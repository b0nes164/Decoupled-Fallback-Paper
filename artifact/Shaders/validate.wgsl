struct ScanParameters
{
    size: u32,
    vec_size: u32,
    work_tiles: u32,
};

@group(0) @binding(0)
var<uniform> params : ScanParameters; 

@group(0) @binding(1)
var<storage, read_write> scan_in: array<u32>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<u32>;

@group(0) @binding(3)
var<storage, read_write> scan_bump: u32;

@group(0) @binding(4)
var<storage, read_write> spine: array<u32>;

@group(0) @binding(5)
var<storage, read_write> misc: array<atomic<u32>>;

const ERR_COUNT_INDEX = 0u;
const BLOCK_DIM = 256u;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {
    for(var i: u32 = id.x; i < params.size; i += griddim.x * BLOCK_DIM){
        let expected = i + 1u;
        if(scan_out[i] != expected){
            atomicAdd(&misc[ERR_COUNT_INDEX], 1u);
        }
    }
}
