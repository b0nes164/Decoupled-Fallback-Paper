struct ScanParameters
{
    size: u32,
    vec_size: u32,
    work_tiles: u32,
};

@group(0) @binding(0)
var<uniform> params : ScanParameters; 

@group(0) @binding(1)
var<storage, read_write> scan_in: array<vec4<u32>>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<vec4<u32>>;

@group(0) @binding(3)
var<storage, read_write> scan_bump: u32;

@group(0) @binding(4)
var<storage, read_write> spine: array<u32>;

@group(0) @binding(5)
var<storage, read_write> misc: array<u32>;

const LAUNCH_DIM = 4096u;
const LAUNCH_MASK = 4095u;
const BLOCK_DIM = 256u;
const MIN_SUBGROUP_SIZE = 4u;
const SPINE_SPT = LAUNCH_DIM / BLOCK_DIM;

const VEC4_SPT = 4u;
const VEC_TILE_SIZE = BLOCK_DIM * VEC4_SPT;
const MAX_PARTIALS_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE * 2u; //Double for conflict avoidance

var<workgroup> wg_partials: array<u32, MAX_PARTIALS_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn reduce(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    let s_offset = laneid + sid * lane_count * VEC4_SPT;
    let lane_log = u32(countTrailingZeros(lane_count));
    let lane_pred = laneid == lane_count - 1u;
    let local_spine = BLOCK_DIM >> lane_log;
    let aligned_size = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);

    var dev_offset = 0u;
    var end = 0u;
    {   
        let div = params.work_tiles / LAUNCH_DIM;
        let partials = params.work_tiles & LAUNCH_MASK;
        let pred = wgid.x < partials;
        let tiles = div + select(0u, 1u, pred);
        if(pred) {
            dev_offset = wgid.x * VEC_TILE_SIZE * (div + 1u);
        } else {
            dev_offset = partials * VEC_TILE_SIZE * (div + 1u) + (wgid.x - partials) * VEC_TILE_SIZE * div; 
        }
        end = dev_offset + tiles * VEC_TILE_SIZE;
    }

    var rake_red = 0u;
    for(; dev_offset < end; dev_offset += VEC_TILE_SIZE){
        var s_red = 0u;
        var i: u32 = s_offset + dev_offset;
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            let t = select(vec4<u32>(0u, 0u, 0u, 0u), scan_in[i], i < params.vec_size);
            s_red += dot(t, vec4(1u, 1u, 1u, 1u));
            i += lane_count;
        }

        s_red = subgroupAdd(s_red);
        if(lane_pred){
            wg_partials[sid] = s_red;
        }
        workgroupBarrier();

        //Non-divergent subgroup agnostic reduction across subgroup partial reductions
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

        rake_red += w_red;
    }
    
    if(threadid.x == 0u){
        spine[wgid.x] = rake_red;
    }
}

//Spine unvectorized
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn spine_scan(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    let s_offset = laneid + sid * lane_count * SPINE_SPT;
    var t_scan = array<u32, SPINE_SPT>();
    
    var i = s_offset;
    for(var k = 0u; k < SPINE_SPT; k += 1u){
        t_scan[k] = spine[i];
        i += lane_count;
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
        let lane_log = u32(countTrailingZeros(lane_count));
        let lane_pred = laneid == lane_count - 1u;
        let local_spine = BLOCK_DIM >> lane_log;
        let aligned_size = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
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
        let prev = select(0u, wg_partials[sid - 1u], sid != 0u);
        var i: u32 = s_offset;
        for(var k = 0u; k < SPINE_SPT; k += 1u){
            spine[i] = t_scan[k] + prev;
            i += lane_count;
        }
    }
}    

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn downsweep(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    let lane_log = u32(countTrailingZeros(lane_count));
    let lane_mask = lane_count - 1u;
    let circular_shift = (laneid + lane_mask) & lane_mask;
    let lane_pred = laneid == lane_mask;
    let local_spine = BLOCK_DIM >> lane_log;
    let aligned_size = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
    let s_offset = laneid + sid * lane_count * VEC4_SPT;
    let rake_prev = select(0u, spine[wgid.x - 1u], wgid.x != 0u);

    var dev_offset = 0u;
    var end = 0u;
    {   
        let div = params.work_tiles / LAUNCH_DIM;
        let partials = params.work_tiles & LAUNCH_MASK;
        let pred = wgid.x < partials;
        let tiles = div + select(0u, 1u, pred);
        if(pred) {
            dev_offset = wgid.x * VEC_TILE_SIZE * (div + 1u);
        } else {
            dev_offset = partials * VEC_TILE_SIZE * (div + 1u) + (wgid.x - partials) * VEC_TILE_SIZE * div; 
        }
        end = dev_offset + tiles * VEC_TILE_SIZE;
    }

    var rake_red = 0u;
    var t_scan = array<vec4<u32>, VEC4_SPT>();
    for(; dev_offset < end; dev_offset += VEC_TILE_SIZE){
        {
            var i: u32 = s_offset + dev_offset;
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

        let total_prev = rake_prev + rake_red + select(0u, wg_partials[sid - 1u], sid != 0u);
        {
            var i = s_offset + dev_offset;
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                if(i < params.vec_size){
                    scan_out[i] = t_scan[k] + total_prev;
                }
                i += lane_count;
            }
        }

        rake_red += wg_partials[local_spine - 1u];
        workgroupBarrier();
    }
}
