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
var<storage, read_write> g_in: array<array<atomic<u32>, 2>>;

@group(0) @binding(2)
var<storage, read_write> unused_0: atomic<u32>;

@group(0) @binding(3)
var<storage, read_write> unused_1: array<u32>;

@group(0) @binding(4)
var<storage, read_write> unused_2: array<u32>;

@group(0) @binding(5)
var<storage, read_write> err: array<atomic<u32>>;

const BLOCK_DIM = 256u;
const ALL = 3u;
const SPLIT = 2u;
const SPINS = 65535u;
const SPREAD = 4096u;

@diagnostic(off, subgroup_uniformity)
fn unsafeShuffle(x: u32, source: u32) -> u32 {
    return subgroupShuffle(x, source);
}

@diagnostic(off, subgroup_uniformity)
fn unsafeBallot(pred: bool) -> u32 {
    return subgroupBallot(pred).x;  
}

fn join(mine: u32, tid: u32) -> u32 {
    let xor = tid ^ 1;
    let theirs = unsafeShuffle(mine, xor);
    return (mine << (16u * tid)) | (theirs << (16u * xor));
}

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) wgdim: vec3<u32>) {
    
    let sid = threadid.x / lane_count;
    let s_count = BLOCK_DIM / lane_count;
    let stride = wgdim.x * s_count * SPREAD;
    var j = (sid + wgid.x * s_count) * SPREAD;

    var total = 0u;
    for(var i = 0u; i < SPINS; i += 1u) {
        let x = select(0u, atomicLoad(&g_in[j & 0xffffffu][laneid]), laneid < SPLIT);
        let bal = unsafeBallot(x == 1) == ALL;
        if(bal) {
            total += join(x, threadid.x);
        }
        j += stride;
    }

    if(laneid < SPLIT) {
        let upper = total >> 16u;
        let lower = total & 0xffffu;
        let maxErr = max(SPINS - upper, SPINS - lower);
        atomicAdd(&err[0], maxErr);
    }
}
