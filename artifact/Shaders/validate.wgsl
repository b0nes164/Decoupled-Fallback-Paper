//***************************************************************************
// Validate
//
// This kernel verifies the correctness of results on the GPU.
//
// WARNING: Binding layout is recycled so some bindings
// are unused
//***************************************************************************
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
var<storage, read_write> unused_1: array<u32>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<u32>;

@group(0) @binding(3)
var<storage, read_write> unused_2: u32;

@group(0) @binding(4)
var<storage, read_write> unused_3: array<u32>;

@group(0) @binding(5)
var<storage, read_write> error: array<atomic<u32>>;

const ERR_COUNT_INDEX = 0u;
const BLOCK_DIM = 256u;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {
    for(var i: u32 = id.x; i < params.size; i += griddim.x * BLOCK_DIM){
        //The inclusive scan of 1 . . . is the natural numbers 
        let expected = i + 1u; 
        if(scan_out[i] != expected){
            atomicAdd(&error[ERR_COUNT_INDEX], 1u);
        }
    }
}
