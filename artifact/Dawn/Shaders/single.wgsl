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
var<storage, read_write> misc: array<u32>;

const BLOCK_DIM = 256u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_PARTIALS_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const VEC4_SPT = 4u;
const VEC_TILE_SIZE = BLOCK_DIM * VEC4_SPT;

var<workgroup> wg_partials: array<u32, MAX_PARTIALS_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    let s_offset = laneid + sid * lane_count * VEC4_SPT;
    var t_scan = array<vec4<u32>, VEC4_SPT>();
    let lane_pred = laneid == lane_count - 1u;
    let lane_log = u32(countTrailingZeros(lane_count));
    let local_spine = BLOCK_DIM >> lane_log;
    let aligned_size = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
    var prev_red = 0u;

    for(var tile_id = 0u; tile_id < params.work_tiles; tile_id += 1u) {
        let dev_offset = tile_id * VEC_TILE_SIZE;
        {
            var i: u32 = s_offset + dev_offset;
            if(tile_id < params.work_tiles- 1u){
                for(var k = 0u; k < VEC4_SPT; k += 1u){
                    t_scan[k] = scan_in[i];
                    t_scan[k].y += t_scan[k].x;
                    t_scan[k].z += t_scan[k].y;
                    t_scan[k].w += t_scan[k].z;
                    i += lane_count;
                }
            }

            if(tile_id == params.work_tiles - 1u){
                for(var k = 0u; k < VEC4_SPT; k += 1u){
                    if(i < params.vec_size){
                        t_scan[k] = scan_in[i];
                        t_scan[k].y += t_scan[k].x;
                        t_scan[k].z += t_scan[k].y;
                        t_scan[k].w += t_scan[k].z;
                    }
                    i += lane_count;
                }
            }
        }

        var prev = 0u;
        let lane_mask = lane_count - 1u;
        let circular_shift = (laneid + lane_mask) & lane_mask;
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            let t = subgroupShuffle(subgroupInclusiveAdd(select(prev, 0u, laneid != 0u) + t_scan[k].w), circular_shift);
            t_scan[k] += select(prev, t, laneid != 0u);
            prev = t;
        }

        if(laneid == 0u){
            wg_partials[sid] = prev;
        }
        workgroupBarrier();

        //Non-divergent subgroup agnostic inclusive scan across subgroup partial reductions
        {   
            var offset = 0u;
            var top_offset = 0u;
            for(var j = lane_count; j <= aligned_size; j <<= lane_log){
                let step = local_spine >> offset;
                let pred = threadid.x < step;
                let t = subgroupInclusiveAdd(select(0u, wg_partials[threadid.x + top_offset], pred));
                if(pred){
                    wg_partials[threadid.x + top_offset] = t;
                    if(lane_pred){
                        wg_partials[sid + step + top_offset] = t;
                    }
                }
                workgroupBarrier();

                if(j != lane_count){
                    let rshift = j >> lane_log;
                    let index = threadid.x + rshift;
                    if(index < local_spine && (index & (j - 1u)) >= rshift){
                        wg_partials[index] += wg_partials[(index >> offset) + top_offset - 1u];
                    }
                }
                top_offset += step;
                offset += lane_log;
            }
        }   
        workgroupBarrier();
        
        {
            let prev = select(0u, wg_partials[sid - 1u], sid != 0u) + prev_red;
            var i = s_offset + dev_offset;
            if(tile_id < params.work_tiles - 1u){
                for(var k = 0u; k < VEC4_SPT; k += 1u){
                    scan_out[i] = t_scan[k] + prev;
                    i += lane_count;
                }
            }

            if(tile_id == params.work_tiles - 1u){
                for(var k = 0u; k < VEC4_SPT; k += 1u){
                    if(i < params.vec_size){
                        scan_out[i] = t_scan[k] + prev;
                    }
                    i += lane_count;
                }
            }
        }

        prev_red += wg_partials[local_spine - 1u];
        workgroupBarrier();
    }
}