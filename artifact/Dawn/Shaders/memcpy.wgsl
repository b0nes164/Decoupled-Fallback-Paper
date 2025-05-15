//***************************************************************************
// Memcpy
//
// This kernel serves as a baseline for comparison against Chained Scans
// since it has identical 2n memory movement.
//
// WARNING: Binding layout is recycled so some bindings
// are unused
//***************************************************************************
enable subgroups;
struct ScanParameters
{
    size: u32,
    vec_size: u32,
    work_tiles: u32,
    unused_0: u32,
};

@group(0) @binding(0)
var<uniform> params : ScanParameters;

@group(0) @binding(1)
var<storage, read_write> scan_in: array<vec4<u32>>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<vec4<u32>>;

@group(0) @binding(3)
var<storage, read_write> scan_bump: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> order: array<atomic<u32>>;

@group(0) @binding(5)
var<storage, read_write> unused_3: array<u32>;

const BLOCK_DIM = 256u;
const VEC4_SPT = 4u;
const VEC_TILE_SIZE = BLOCK_DIM * VEC4_SPT;

var<workgroup> wg_broadcast: u32;

const READY = 1u;
const INC = 2u;
const MASK = 3u;


@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    if(threadid.x == 0u){
        wg_broadcast = atomicAdd(&scan_bump, 1u);
    }
    let tile_id = workgroupUniformLoad(&wg_broadcast);

    let sid = threadid.x / lane_count;
    let s_offset = laneid + sid * lane_count * VEC4_SPT;
    var t = array<vec4<u32>, VEC4_SPT>();
    {
        var i = s_offset + wgid.x * VEC_TILE_SIZE;
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            t[k] = scan_in[i];
            i += lane_count;
        }
    }
    workgroupBarrier();

    if(threadid.x == 0u){
        atomicStore(&order[tile_id], select(READY, INC, tile_id == 0u));
    }

    // With Fallback :^):

    if(tile_id != 0u){
        if (threadid.x == 0u) {
            var lookback_id = tile_id - 1u;
            while (true) {
                var p = 0u;
                var spin_count = 0u;
                while (spin_count < 4u) {
                    p = atomicLoad(&order[lookback_id]);
                    if (p != 0u) {
                        if (p == INC) {
                            atomicStore(&order[tile_id], INC);
                            break;
                        } else {
                            lookback_id -= 1u;
                        }
                    } else {
                        spin_count += 1u;
                    }
                }

                if (p == INC) {
                    break;
                } else if (spin_count == 4u) {
                    lookback_id -= 1u;
                }
            }
        }
        workgroupBarrier();
    }

    // No fallback :^(

    // if(tile_id != 0u){
    //     if (threadid.x == 0u) {
    //         var lookback_id = tile_id - 1u;
    //         while (true) {
    //             var p = atomicLoad(&order[lookback_id]);
    //             if (p != 0u) {
    //                 if (p == INC) {
    //                     atomicStore(&order[tile_id], INC);
    //                     break;
    //                 } else {
    //                     lookback_id -= 1u;
    //                 }
    //             }
    //         }
    //     }
    //     workgroupBarrier();
    // }

    {
        var i = s_offset + wgid.x * VEC_TILE_SIZE;
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            scan_out[i] = t[k];
            i += lane_count;
        }
    }
}
