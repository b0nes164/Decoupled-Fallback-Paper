//***************************************************************************
// Initialization Kernel
//
// Responsible for initializing `scan_bump` to 0,
// setting the state of spine tiles to 0 (NOT_READY),
// and initializing the scan_input.
//
// WARNING: Binding layout is recycled so some bindings
// are unused
//***************************************************************************
struct ScanParameters
{
    size: u32,
    vec_size: u32,
    work_tiles: u32,
    unused: u32,
};

@group(0) @binding(0)
var<uniform> params : ScanParameters;

@group(0) @binding(1)
var<storage, read_write> scan_in: array<u32>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<u32>;

@group(0) @binding(3)
var<storage, read_write> scan_bump: array<u32>;

@group(0) @binding(4)
var<storage, read_write> spine: array<u32>;

@group(0) @binding(5)
var<storage, read_write> misc: array<u32>;

const SPLIT_MEMBERS = 2u;
const BLOCK_DIM = 256u;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn initPass(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    //Initialize all elements to 1
    //Initialize the output to prevent previous runs from
    //potentially resulting in false positive test passes
    for(var i = id.x; i < params.size; i += griddim.x * BLOCK_DIM){
        scan_in[i] = 1u;
        scan_out[i] = 1u << 31u;
    }

    //Set spine states to NOT_READY
    for(var i = id.x; i < params.work_tiles * SPLIT_MEMBERS; i += griddim.x * BLOCK_DIM){
        spine[i] = 0u;
    }

    //Reset the atomic bump
    if(id.x == 0u){
        scan_bump[0u] = 0u;
    }

    //Increment the batch relative index, so we know where to store statistics and error counts.
    if(id.x == 0u) {
        scan_bump[1u] += 1u;
    }
}

const MISC_STRIDE = 4u;
const MAX_PASSES_PER_BATCH = 2048u; // MAX_QUERY_ENTRIES / 2
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn initBatch(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    //Reset the batch relative index.
    if(id.x == 0u) {
        scan_bump[1u] = 0u;
    }

    //Reset the miscellanous buffer, which holds statistics and error counts.
    //The size of the miscellaneous buffer is sized to the max number of launches we can fit into
    //a single command encoder and still be timed. Therefore, this is run once per batch.
    //
    //The first stride is used for capturing the error count, if validation is enabled.
    let total_misc_size = (MAX_PASSES_PER_BATCH + 1u) * MISC_STRIDE;
    for(var i = id.x; i < total_misc_size; i += griddim.x * BLOCK_DIM){
        misc[i] = 0u;
    }
}