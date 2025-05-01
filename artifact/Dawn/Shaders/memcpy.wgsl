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
var<storage, read_write> unused_1: u32;

@group(0) @binding(4)
var<storage, read_write> unused_2: array<u32>;

@group(0) @binding(5)
var<storage, read_write> unused_3: array<u32>;

const BLOCK_DIM = 256u;
const VEC4_SPT = 4u;
const VEC_TILE_SIZE = BLOCK_DIM * VEC4_SPT;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    // let end = (wgid.x + 1u) * VEC_TILE_SIZE;
    // if(wgid.x < params.work_tiles - 1u){
    //     for(var i = threadid.x + wgid.x * VEC_TILE_SIZE; i < end; i += BLOCK_DIM){
    //         var t = scan_in[i];
    //         for (var z = 0; z < 16; z += 1) {
    //             t.x = (t.x * 1129100271u) ^ (t.y * 1129100273u) ^
    //                   (t.z * 1129100281u) ^ (t.w * 1129100283u);
    //         }
    //         scan_out[i] = t;
    //     }
    // }

    // if(wgid.x == params.work_tiles - 1u){
    //     for(var i = threadid.x + wgid.x * VEC_TILE_SIZE; i < end; i += BLOCK_DIM){
    //         if(i < params.vec_size){
    //             scan_out[i] = scan_in[i];
    //         }
    //     }
    // }

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

    for (var k = 0u; k < VEC4_SPT; k += 1u) {
        for (var z = 0; z < 32; z += 1) {
            t[k].x = (t[k].x * 1129100271u) ^ (t[k].y * 1129100273u) ^
                     (t[k].z * 1129100281u) ^ (t[k].w * 1129100283u);
        }
    }

    {
        var i = s_offset + wgid.x * VEC_TILE_SIZE;
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            scan_out[i] = t[k];
            i += lane_count;
        }
    }
}
