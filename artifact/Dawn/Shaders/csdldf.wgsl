//***************************************************************************
// Chained Scan with Decoupled Lookback and Decoupled Fallback
//
// CSDL but with an additional fallback routine, allowing the scan to work
// on hardware without forward thread progress guarantees.
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
var<storage, read_write> spine: array<array<atomic<u32>, 2>>;

@group(0) @binding(5)
var<storage, read_write> unused_1: array<u32>;

const BLOCK_DIM = 256u;
const SPLIT_MEMBERS = 2u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_PARTIALS_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE * 2u; //Double for conflict avoidance

const VEC4_SPT = 4u;
const VEC_TILE_SIZE = BLOCK_DIM * VEC4_SPT;

const FLAG_NOT_READY = 0u;
const FLAG_READY = 0x40000000u;
const FLAG_INCLUSIVE = 0x80000000u;
const FLAG_MASK = 0xC0000000u;
const VALUE_MASK = 0xffffu;
const ALL_READY = 3u;

const MAX_SPIN_COUNT = 4u;
const LOCKED = 1u;
const UNLOCKED = 0u;

var<workgroup> wg_control: u32;
var<workgroup> wg_broadcast: u32;
var<workgroup> wg_partials: array<u32, MAX_PARTIALS_SIZE>;
var<workgroup> wg_fallback: array<u32, MAX_PARTIALS_SIZE>;

@diagnostic(off, subgroup_uniformity)
fn unsafeShuffle(x: u32, source: u32) -> u32 {
    return subgroupShuffle(x, source);
}

//lop off of the upper ballot bits;
//we never need them across all subgroup sizes
@diagnostic(off, subgroup_uniformity)
fn unsafeBallot(pred: bool) -> u32 {
    return subgroupBallot(pred).x;  
}

fn join(mine: u32, tid: u32) -> u32 {
    let xor = tid ^ 1;
    let theirs = unsafeShuffle(mine, xor);
    return (mine << (16u * tid)) | (theirs << (16u * xor));
}

fn split(x: u32, tid: u32) -> u32 {
    return (x >> (tid * 16u)) & VALUE_MASK;
}

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    
    //acquire partition index, set the lock
    if(threadid.x == 0u){
        wg_broadcast = atomicAdd(&scan_bump, 1u);
        wg_control = LOCKED;
    }
    let tile_id = workgroupUniformLoad(&wg_broadcast);
    let s_offset = laneid + sid * lane_count * VEC4_SPT;

    var t_scan = array<vec4<u32>, VEC4_SPT>();
    {
        var i = s_offset + tile_id * VEC_TILE_SIZE;
        if(tile_id < params.work_tiles - 1u){
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
    let lane_log = u32(countTrailingZeros(lane_count));
    let local_spine = BLOCK_DIM >> lane_log;
    let aligned_size = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
    {   
        var offset = 0u;
        var top_offset = 0u;
        let lane_pred = laneid == lane_count - 1u;
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

    //Device broadcast
    if(threadid.x < SPLIT_MEMBERS){
        let t = split(wg_partials[local_spine - 1u], threadid.x) | select(FLAG_READY, FLAG_INCLUSIVE, tile_id == 0u);
        atomicStore(&spine[tile_id][threadid.x], t);
    }

    //lookback, single subgroup
    if(tile_id != 0u){
        var prev_red = 0u;
        var lookback_id = tile_id - 1u;
        var control_flag = workgroupUniformLoad(&wg_control);
        while(control_flag == LOCKED){
            if(threadid.x < lane_count){
                var spin_count = 0u;
                while(spin_count < MAX_SPIN_COUNT){
                    var flag_payload = select(0u, atomicLoad(&spine[lookback_id][threadid.x]), threadid.x < SPLIT_MEMBERS);
                    if(unsafeBallot((flag_payload & FLAG_MASK) > FLAG_NOT_READY) == ALL_READY) {
                        var incl_bal = unsafeBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
                        if(incl_bal != 0u) {
                            //Did we find any inclusive? Alright, the rest are guaranteed to be on their way, lets just wait.
                            //Note, this rests on the assumption that the execution width of the load == store.
                            //If for whatever reason, this is not true, it risks deadlock without FPG.
                            while(incl_bal != ALL_READY){
                                flag_payload = select(0u, atomicLoad(&spine[lookback_id][threadid.x]), threadid.x < SPLIT_MEMBERS);
                                incl_bal = unsafeBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
                            }
                            prev_red += join(flag_payload & VALUE_MASK, threadid.x);
                            if(threadid.x < SPLIT_MEMBERS){
                                let t = split(prev_red + wg_partials[local_spine - 1u], threadid.x) | FLAG_INCLUSIVE;
                                atomicStore(&spine[tile_id][threadid.x], t);
                            }
                            if(threadid.x == 0u){
                                wg_control = UNLOCKED;
                                wg_broadcast = prev_red;
                            }
                            break;
                        } else {
                            prev_red += join(flag_payload & VALUE_MASK, threadid.x);
                            spin_count = 0u;
                            lookback_id -= 1u;
                        }
                    } else {
                        spin_count += 1u;
                    }
                }

                if(threadid.x == 0 && spin_count == MAX_SPIN_COUNT) {
                    wg_broadcast = lookback_id;
                }
            }

            //Fallback if still locked
            control_flag = workgroupUniformLoad(&wg_control);
            if(control_flag == LOCKED){
                let fallback_id = wg_broadcast;
                {
                    var t_red = 0u;
                    var i = s_offset + fallback_id * VEC_TILE_SIZE;
                    for(var k = 0u; k < VEC4_SPT; k += 1u){
                        let t = scan_in[i];
                        t_red += t.x + t.y + t.z + t.w;
                        i += lane_count;
                    }

                    let s_red = subgroupAdd(t_red);
                    if(laneid == 0u){
                        wg_fallback[sid] = s_red;
                    }
                }
                workgroupBarrier();

                //Non-divergent subgroup agnostic reduction across subgroup partial reductions
                var f_red = 0u;
                {
                    var offset = 0u;
                    var top_offset = 0u;
                    let lane_pred = laneid == lane_count - 1u;
                    for(var j = lane_count; j <= aligned_size; j <<= lane_log){
                        let step = local_spine >> offset;
                        let pred = threadid.x < step;
                        f_red = subgroupAdd(select(0u, wg_fallback[threadid.x + top_offset], pred));
                        if(pred && lane_pred){
                            wg_fallback[sid + step + top_offset] = f_red;
                        }
                        workgroupBarrier();
                        top_offset += step;
                        offset += lane_log;
                    }
                }

                if(threadid.x < lane_count){
                    let f_split = split(f_red, threadid.x) | select(FLAG_READY, FLAG_INCLUSIVE, fallback_id == 0u);
                    var f_payload = 0u;
                    if(threadid.x < SPLIT_MEMBERS) {
                        f_payload = atomicMax(&spine[fallback_id][threadid.x], f_split);
                    }
                    let incl_found = unsafeBallot((f_payload & FLAG_MASK) == FLAG_INCLUSIVE) == ALL_READY;
                    if(incl_found){
                        prev_red += join(f_payload & VALUE_MASK, threadid.x); 
                    } else {
                        prev_red += f_red;
                    }

                    if(fallback_id == 0u || incl_found){
                        if(threadid.x < SPLIT_MEMBERS){
                            let t = split(prev_red + wg_partials[local_spine - 1u], threadid.x) | FLAG_INCLUSIVE;
                            atomicStore(&spine[tile_id][threadid.x], t);
                        }
                        if(threadid.x == 0u){
                            wg_control = UNLOCKED;
                            wg_broadcast = prev_red;
                        }
                    } else {
                        lookback_id -= 1u;
                    }
                }
                control_flag = workgroupUniformLoad(&wg_control);
            }
        }
    }

    {
        var i = s_offset + tile_id * VEC_TILE_SIZE;
        let prev = wg_broadcast + select(0u, wg_partials[sid - 1u], sid != 0u); //wg_broadcast is 0 for tile_id 0
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
}
