struct InfoStruct
{
    size: u32,
    vec_size: u32,
    work_tiles: u32,
};

@group(0) @binding(0)
var<uniform> info : InfoStruct; 

@group(0) @binding(1)
var<storage, read_write> scan_in: array<u32>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<u32>;

@group(0) @binding(3)
var<storage, read_write> scan_bump: u32;

@group(0) @binding(4)
var<storage, read_write> spine: array<u32>;

@group(0) @binding(5)
var<storage, read_write> misc: array<u32>;

const MISC_SIZE = 5u;
const BLOCK_DIM = 256u;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    for(var i = id.x; i < info.size; i += griddim.x * BLOCK_DIM){
        scan_in[i] = 1u;
        scan_out[i] = 1u << 31u;
    }

    for(var i = id.x; i < info.work_tiles * 4; i += griddim.x * BLOCK_DIM){
        spine[i] = 0u;
    }

    if(id.x == 0u){
        scan_bump = 0u;
    }

    if(id.x < MISC_SIZE){
        misc[id.x] = 0u; 
    }
}
