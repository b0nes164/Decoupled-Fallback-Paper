//***************************************************************************
// Reduce then Scan
//
// Also known as a "Tree Scan" or "Two-Level Scan," this approach 
// first reduces values to intermediate results, which are then 
// scanned and passed into a final downsweep pass.
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
var<storage, read_write> spine: array<u32>;

@group(0) @binding(5)
var<storage, read_write> misc: array<u32>;

const BLOCK_DIM = 256u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_PARTIALS_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const VEC4_SPT = 4u;
const VEC_TILE_SIZE = BLOCK_DIM * VEC4_SPT;

const SPINE_SPT = 16u;
const SPINE_TILE_SIZE = BLOCK_DIM * SPINE_SPT;

var<workgroup> wg_partials: array<u32, MAX_PARTIALS_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn reduce(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    let s_offset = laneid + sid * lane_count * VEC4_SPT;
    let dev_offset = wgid.x * VEC_TILE_SIZE;
    var i: u32 = s_offset + dev_offset;

    var t_red = 0u;
    if(wgid.x < params.work_tiles - 1u){
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            let t = scan_in[i];
            t_red += t.x + t.y + t.z + t.w;
            i += lane_count;
        }
    }

    if(wgid.x == params.work_tiles - 1u){
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            let t = select(vec4<u32>(0u, 0u, 0u, 0u), scan_in[i], i < params.vec_size);
            t_red += t.x + t.y + t.z + t.w;
            i += lane_count;
        }
    }

    t_red = subgroupAdd(t_red);
    if(laneid == 0u){
        wg_partials[sid] = t_red;
    }
    workgroupBarrier();

    //Non-divergent subgroup agnostic reduction across subgroup partial reductions
    let lane_pred = laneid == lane_count - 1u;
    let lane_log = u32(countTrailingZeros(lane_count));
    let local_spine = BLOCK_DIM >> lane_log;
    let aligned_size = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);

    var w_red = 0u;
    var offset = 0u;
    var top_offset = 0u;
    for(var j = lane_count; j <= aligned_size; j <<= lane_log){
        let step = local_spine >> offset;
        let pred = threadid.x < step;
        w_red = subgroupAdd(select(0u, wg_partials[threadid.x + top_offset], pred));
        if(pred && lane_pred){
            wg_partials[sid + step + top_offset] = w_red;
        }
        workgroupBarrier();
        top_offset += step;
        offset += lane_log;
    }

    if(threadid.x == 0u){
        spine[wgid.x] = w_red;
    }
}

//Spine unvectorized
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn spine_scan(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    let lane_pred = laneid == lane_count - 1u;
    let lane_log = u32(countTrailingZeros(lane_count));
    let s_offset = laneid + sid * lane_count * SPINE_SPT;
    let local_spine = BLOCK_DIM >> lane_log;
    let local_align = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
    let aligned_size = (params.work_tiles + SPINE_TILE_SIZE - 1u) / SPINE_TILE_SIZE * SPINE_TILE_SIZE;
    var t_scan = array<u32, SPINE_SPT>();
    
    var prev_red = 0u;
    for(var dev_offset = 0u; dev_offset < aligned_size; dev_offset += SPINE_TILE_SIZE){
        {
            var i = s_offset + dev_offset;
            for(var k = 0u; k < SPINE_SPT; k += 1u){
                if(i < params.work_tiles){
                    t_scan[k] = spine[i];
                }
                i += lane_count;
            }
        }

        var prev = 0u;
        for(var k = 0u; k < SPINE_SPT; k += 1u){
            t_scan[k] = subgroupInclusiveAdd(t_scan[k]) + prev;
            prev = subgroupShuffle(t_scan[k], lane_count - 1);
        }

        if(laneid == lane_count - 1u){
            wg_partials[sid] = prev;
        }
        workgroupBarrier();

        //Non-divergent subgroup agnostic inclusive scan across subgroup partial reductions
        {   
            var offset = 0u;
            var top_offset = 0u;
            for(var j = lane_count; j <= local_align; j <<= lane_log){
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
            var i: u32 = s_offset + dev_offset;
            for(var k = 0u; k < SPINE_SPT; k += 1u){
                if(i < params.work_tiles){
                    spine[i] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }

        prev_red += subgroupBroadcast(wg_partials[local_spine - 1u], 0u);
        workgroupBarrier();
    }
}    

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn downsweep(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    let s_offset = laneid + sid * lane_count * VEC4_SPT;
    var t_scan = array<vec4<u32>, VEC4_SPT>();

    {
        let dev_offset = wgid.x * VEC_TILE_SIZE;
        var i: u32 = s_offset + dev_offset;

        if(wgid.x < params.work_tiles- 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                t_scan[k] = scan_in[i];
                t_scan[k].y += t_scan[k].x;
                t_scan[k].z += t_scan[k].y;
                t_scan[k].w += t_scan[k].z;
                i += lane_count;
            }
        }

        if(wgid.x == params.work_tiles - 1u){
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
    }
    workgroupBarrier();

    //Non-divergent subgroup agnostic inclusive scan across subgroup partial reductions
    {   
        let lane_pred = laneid == lane_count - 1u;
        let lane_log = u32(countTrailingZeros(lane_count));
        let local_spine = BLOCK_DIM >> lane_log;
        let aligned_size = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
        
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
        let prev = select(0u, spine[wgid.x - 1u], wgid.x != 0u) + select(0u, wg_partials[sid - 1u], sid != 0u);
        let dev_offset =  wgid.x * VEC_TILE_SIZE;
        var i = s_offset + dev_offset;

        if(wgid.x < params.work_tiles - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                scan_out[i] = t_scan[k] + prev;
                i += lane_count;
            }
        }

        if(wgid.x == params.work_tiles - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                if(i < params.vec_size){
                    scan_out[i] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }
    }
}
