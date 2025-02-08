//***************************************************************************
// Memcpy
//
// This kernel serves as a baseline for comparison against Chained Scans 
// since it has identical 2n memory movement.
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
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {
    let end = params.vec_size;
    for(var i = id.x; i < end; i += griddim.x * BLOCK_DIM){
        scan_out[i] = scan_in[i];
    }
}
