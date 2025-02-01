use std::{
    env,
    fs::File,
    io::{self, Write},
    vec,
};

fn div_round_up(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}

struct GPUContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    query_set: wgpu::QuerySet,
    timestamp_freq: f32,
}

impl GPUContext {
    async fn init() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions::default(),
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let adapter_info = adapter.get_info();
        println!("Adapter Info:");
        println!("  Name: {}", adapter_info.name);
        println!("  Vendor: {}", adapter_info.vendor);
        println!("  Device: {}", adapter_info.device);
        println!("  Backend: {:?}", adapter_info.backend);
        println!("  Driver: {}", adapter_info.driver);
        println!("  Driver Info: {}", adapter_info.driver_info);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::SUBGROUP
                        | wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            count: 6,
            ty: wgpu::QueryType::Timestamp,
        });

        let timestamp_freq = queue.get_timestamp_period();

        GPUContext {
            device,
            queue,
            query_set,
            timestamp_freq,
        }
    }
}

struct GPUBuffers {
    params: wgpu::Buffer,
    scan_in: wgpu::Buffer,
    scan_out: wgpu::Buffer,
    scan_bump: wgpu::Buffer,
    spine: wgpu::Buffer,
    timestamp: wgpu::Buffer,
    timestamp_readback: wgpu::Buffer,
    readback: wgpu::Buffer,
    misc: wgpu::Buffer,
}

impl GPUBuffers {
    fn init(
        gpu: &GPUContext,
        size: usize,
        work_tiles: usize,
        max_pass_count: usize,
        max_readback_size: usize,
        misc_size: usize,
    ) -> Self {
        let buffer_size = (size * std::mem::size_of::<u32>()) as u64;
        let params: wgpu::Buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan Parameters"),
            size: (4usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scan_in: wgpu::Buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan In"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let scan_out: wgpu::Buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan Out"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scan_bump: wgpu::Buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan Bump"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let spine: wgpu::Buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spine"),
            size: ((work_tiles * 2usize) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let timestamp_size = (max_pass_count * 2usize * std::mem::size_of::<u64>()) as u64;
        let timestamp = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp"),
            size: timestamp_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let timestamp_readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Readback"),
            size: timestamp_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback"),
            size: ((max_readback_size) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let misc = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Misc"),
            size: (misc_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        GPUBuffers {
            params,
            scan_in,
            scan_out,
            scan_bump,
            spine,
            timestamp,
            timestamp_readback,
            readback,
            misc,
        }
    }
}

//For simplicity we are going to use the bind group and layout
//for all of the kernels except the validation
struct ComputeShader {
    bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    label: String,
}

impl ComputeShader {
    fn init(
        gpu: &GPUContext,
        gpu_buffers: &GPUBuffers,
        entry_point: &str,
        module: &wgpu::ShaderModule,
        cs_label: &str,
    ) -> Self {
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("Bind Group Layout {}", cs_label)),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Bind Group {}", cs_label)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buffers.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_buffers.scan_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_buffers.scan_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu_buffers.scan_bump.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu_buffers.spine.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gpu_buffers.misc.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout_init =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("Pipeline Layout {}", cs_label)),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        
        let mut comp_options = wgpu::PipelineCompilationOptions::default();
        comp_options.zero_initialize_workgroup_memory = false;
        let compute_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Compute Pipeline {}", cs_label)),
                    layout: Some(&pipeline_layout_init),
                    module,
                    entry_point: Some(entry_point),
                    compilation_options: comp_options,
                    cache: Default::default(),
                });

        ComputeShader {
            bind_group,
            compute_pipeline,
            label: cs_label.to_string(),
        }
    }
}

struct Shaders {
    init: ComputeShader,
    reduce: ComputeShader,
    spine_scan: ComputeShader,
    downsweep: ComputeShader,
    csdl: ComputeShader,
    csdldf: ComputeShader,
    csdldf_emu: ComputeShader,
    csdldf_occ: ComputeShader,
    memcpy: ComputeShader,
    validate: ComputeShader,
}

impl Shaders {
    fn init(gpu: &GPUContext, gpu_buffers: &GPUBuffers) -> Self {
        let init_mod;
        let valid_mod;
        let reduce_mod;
        let spine_mod;
        let downsweep_mod;
        let csdl_mod;
        let csdldf_mod;
        let csdldf_emu_mod;
        let csdldf_occ_mod;
        let memcpy_mod;

        unsafe {
            init_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!("../SPIR-V/init.main.spv"));
            valid_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!(
                    "../SPIR-V/validate.main.spv"
                ));
            reduce_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!("../SPIR-V/rts.reduce.spv"));
            spine_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!(
                    "../SPIR-V/rts.spine_scan.spv"
                ));
            downsweep_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!(
                    "../SPIR-V/rts.downsweep.spv"
                ));
            csdl_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!("../SPIR-V/csdl.main.spv"));
            csdldf_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!("../SPIR-V/csdldf.main.spv"));
            csdldf_emu_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!(
                    "../SPIR-V/csdldf_emulate.main.spv"
                ));
            csdldf_occ_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!(
                    "../SPIR-V/csdldf_occ.main.spv"
                ));
            memcpy_mod = gpu
                .device
                .create_shader_module_spirv(&wgpu::include_spirv_raw!("../SPIR-V/memcpy.main.spv"));
        }

        let init = ComputeShader::init(gpu, gpu_buffers, "main", &init_mod, "Init");
        let reduce = ComputeShader::init(gpu, gpu_buffers, "reduce", &reduce_mod, "Reduce");
        let spine_scan =
            ComputeShader::init(gpu, gpu_buffers, "spine_scan", &spine_mod, "Spine Scan");
        let downsweep =
            ComputeShader::init(gpu, gpu_buffers, "downsweep", &downsweep_mod, "Downsweep");
        let csdl = ComputeShader::init(gpu, gpu_buffers, "main", &csdl_mod, "CSDL");
        let csdldf = ComputeShader::init(gpu, gpu_buffers, "main", &csdldf_mod, "CSDLDF");
        let csdldf_emu = ComputeShader::init(
            gpu,
            gpu_buffers,
            "main",
            &csdldf_emu_mod,
            "CSDLDF Emulation",
        );
        let csdldf_occ = ComputeShader::init(
            gpu,
            gpu_buffers,
            "main",
            &csdldf_occ_mod,
            "CSDLDF Occupancy",
        );
        let memcpy = ComputeShader::init(gpu, gpu_buffers, "main", &memcpy_mod, "Memcopy");
        let validate = ComputeShader::init(gpu, gpu_buffers, "main", &valid_mod, "Validate");

        Shaders {
            init,
            reduce,
            spine_scan,
            downsweep,
            csdl,
            csdldf,
            csdldf_emu,
            csdldf_occ,
            memcpy,
            validate,
        }
    }
}

fn set_compute_pass(
    query: &wgpu::QuerySet,
    cs: &ComputeShader,
    com_encoder: &mut wgpu::CommandEncoder,
    work_tiles: u32,
    timestamp_offset: u32,
) {
    let mut pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(&format!("{} Pass", cs.label)),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set: query,
            beginning_of_pass_write_index: Some(timestamp_offset),
            end_of_pass_write_index: Some(timestamp_offset + 1u32),
        }),
    });
    pass.set_pipeline(&cs.compute_pipeline);
    pass.set_bind_group(0, &cs.bind_group, &[]);
    pass.dispatch_workgroups(work_tiles, 1, 1);
}

fn readback_back(tester: &Tester, data_out: &mut Vec<u32>, readback_size: u64) {
    let readback_slice = &tester.gpu_buffers.readback.slice(0..readback_size);
    readback_slice.map_async(wgpu::MapMode::Read, |result| {
        result.unwrap();
    });
    tester.gpu_context.device.poll(wgpu::Maintain::wait());
    let data = readback_slice.get_mapped_range();
    data_out.extend_from_slice(bytemuck::cast_slice(&data));
}

fn validate(tester: &Tester, cs: &ComputeShader) -> bool {
    let mut valid_command =
        tester
            .gpu_context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Valid Command Encoder"),
            });
    {
        let mut valid_pass = valid_command.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Validate Pass"),
            timestamp_writes: None,
        });
        valid_pass.set_pipeline(&cs.compute_pipeline);
        valid_pass.set_bind_group(0, &cs.bind_group, &[]);
        valid_pass.dispatch_workgroups(256, 1, 1);
    }
    valid_command.copy_buffer_to_buffer(
        &tester.gpu_buffers.misc,
        0u64,
        &tester.gpu_buffers.readback,
        0u64,
        std::mem::size_of::<u32>() as u64,
    );
    tester
        .gpu_context
        .queue
        .submit(Some(valid_command.finish()));

    let mut data_out: Vec<u32> = vec![];
    readback_back(tester, &mut data_out, std::mem::size_of::<u32>() as u64);
    tester.gpu_buffers.readback.unmap();

    if data_out[0] != 0 {
        println!("Err count {}", data_out[0]);
    }
    data_out[0] == 0
}

fn get_stats(tester: &Tester, stats_out: &mut Vec<u32>) {
    let mut stats_command =
        tester
            .gpu_context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Stats Command Encoder"),
            });
    stats_command.copy_buffer_to_buffer(
        &tester.gpu_buffers.misc,
        std::mem::size_of::<u32>() as u64,
        &tester.gpu_buffers.readback,
        0u64,
        4u64 * std::mem::size_of::<u32>() as u64,
    );
    tester
        .gpu_context
        .queue
        .submit(Some(stats_command.finish()));
    readback_back(tester, stats_out, 4u64 * std::mem::size_of::<u32>() as u64);
    tester.gpu_buffers.readback.unmap();
}

trait PassTrait {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder);
    fn pass_count(&self) -> u32;
}

struct RtsPass;
impl PassTrait for RtsPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.reduce,
            com_encoder,
            tester.work_tiles,
            0u32,
        );
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.spine_scan,
            com_encoder,
            1u32,
            2u32,
        );
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.downsweep,
            com_encoder,
            tester.work_tiles,
            4u32,
        );
    }

    fn pass_count(&self) -> u32 {
        3u32
    }
}

struct CsdlPass;
impl PassTrait for CsdlPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdl,
            com_encoder,
            tester.work_tiles,
            0u32,
        );
    }

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct CsdldfPass;
impl PassTrait for CsdldfPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdldf,
            com_encoder,
            tester.work_tiles,
            0u32,
        );
    }

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct CsdldfEmulatePass;
impl PassTrait for CsdldfEmulatePass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdldf_emu,
            com_encoder,
            tester.work_tiles,
            0u32,
        );
    }

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct MemcpyPass;
impl PassTrait for MemcpyPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.memcpy,
            com_encoder,
            tester.work_tiles,
            0u32,
        );
    }

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct DataStruct {
    time: Vec<f64>,
    total_spins: Vec<f64>,
    lookback_length: Vec<f64>,
    fallbacks_initiated: Vec<u32>,
    successful_insertions: Vec<u32>,
}

impl DataStruct {
    fn new() -> Self {
        Self {
            time: Vec::new(),
            total_spins: Vec::new(),
            lookback_length: Vec::new(),
            fallbacks_initiated: Vec::new(),
            successful_insertions: Vec::new(),
        }
    }

    fn resize(&mut self, size: usize) {
        self.time.resize(size, 0.0);
        self.total_spins.resize(size, 0.0);
        self.lookback_length.resize(size, 0.0);
        self.fallbacks_initiated.resize(size, 0);
        self.successful_insertions.resize(size, 0);
    }
}

struct Tester {
    gpu_context: GPUContext,
    gpu_buffers: GPUBuffers,
    gpu_shaders: Shaders,
    size: u32,
    work_tiles: u32,
}

impl Tester {
    async fn init(
        size: u32,
        work_tiles: u32,
        max_pass_count: usize,
        max_readback_size: usize,
        misc_size: usize,
    ) -> Self {
        let gpu_context = GPUContext::init().await;
        let gpu_buffers = GPUBuffers::init(
            &gpu_context,
            size as usize,
            work_tiles as usize,
            max_pass_count,
            max_readback_size,
            misc_size,
        );
        let gpu_shaders = Shaders::init(&gpu_context, &gpu_buffers);
        Tester {
            gpu_context,
            gpu_buffers,
            gpu_shaders,
            size,
            work_tiles,
        }
    }

    fn init_pass(&self, com_encoder: &mut wgpu::CommandEncoder) {
        let mut init_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Init Pass"),
            timestamp_writes: None,
        });
        init_pass.set_pipeline(&self.gpu_shaders.init.compute_pipeline);
        init_pass.set_bind_group(0, &self.gpu_shaders.init.bind_group, &[]);
        init_pass.dispatch_workgroups(256, 1, 1);
    }

    fn resolve_time_query(&self, com_encoder: &mut wgpu::CommandEncoder, pass_count: u32) {
        let entries_to_resolve = pass_count * 2;
        com_encoder.resolve_query_set(
            &self.gpu_context.query_set,
            0..entries_to_resolve,
            &self.gpu_buffers.timestamp,
            0u64,
        );
        com_encoder.copy_buffer_to_buffer(
            &self.gpu_buffers.timestamp,
            0u64,
            &self.gpu_buffers.timestamp_readback,
            0u64,
            entries_to_resolve as u64 * std::mem::size_of::<u64>() as u64,
        );
    }

    fn time(&self, pass_count: usize) -> u64 {
        let query_slice = self.gpu_buffers.timestamp_readback.slice(..);
        query_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        self.gpu_context.device.poll(wgpu::Maintain::wait());
        let query_out = query_slice.get_mapped_range();
        let timestamp: Vec<u64> = bytemuck::cast_slice(&query_out).to_vec();
        let mut total_time = 0u64;
        for i in 0..pass_count {
            total_time += u64::wrapping_sub(timestamp[i * 2 + 1], timestamp[i * 2]);
        }
        total_time
    }

    fn readback_results(&self, readback_size: u32) {
        let mut copy_command =
            self.gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Command Encoder"),
                });
        copy_command.copy_buffer_to_buffer(
            &self.gpu_buffers.scan_out,
            0u64,
            &self.gpu_buffers.readback,
            0u64,
            readback_size as u64 * std::mem::size_of::<u32>() as u64,
        );
        self.gpu_context.queue.submit(Some(copy_command.finish()));
        let readback_slice = self
            .gpu_buffers
            .readback
            .slice(0..((readback_size as usize * std::mem::size_of::<u32>()) as u64));
        readback_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        self.gpu_context.device.poll(wgpu::Maintain::wait());
        let data = readback_slice.get_mapped_range();
        let data_out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        println!("{:?}", data_out);
        // for i in 0..readback_size{
        //     println!("{} {}", i, data_out[i as usize]);
        // }
    }

    fn get_occupancy(&self) -> u32 {
        let mut occ_command =
            self.gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Occupancy Command Encoder"),
                });
        self.init_pass(&mut occ_command);
        set_compute_pass(
            &self.gpu_context.query_set,
            &self.gpu_shaders.csdldf_occ,
            &mut occ_command,
            self.work_tiles,
            0u32,
        );
        occ_command.copy_buffer_to_buffer(
            &self.gpu_buffers.misc,
            std::mem::size_of::<u32>() as u64,
            &self.gpu_buffers.readback,
            0u64,
            std::mem::size_of::<u32>() as u64,
        );
        self.gpu_context.queue.submit(Some(occ_command.finish()));
        let mut occ_out: Vec<u32> = vec![];
        readback_back(self, &mut occ_out, std::mem::size_of::<u32>() as u64);
        self.gpu_buffers.readback.unmap();
        occ_out[0]
    }

    fn record_to_csv(
        &self,
        should_get_stats: bool,
        data: &DataStruct,
        filename: &str,
    ) -> io::Result<()> {
        let mut file = File::create(format!("{}.csv", filename))?;

        if should_get_stats {
            // Write full headers
            writeln!(
                file,
                "time,totalSpins,lookbackLength,fallbacksInitiated,successfulInsertions"
            )?;

            // Write full data
            for i in 0..data.time.len() {
                writeln!(
                    file,
                    "{},{},{},{},{}",
                    data.time[i],
                    data.total_spins[i],
                    data.lookback_length[i],
                    data.fallbacks_initiated[i],
                    data.successful_insertions[i]
                )?;
            }
        } else {
            writeln!(file, "time")?;

            for &t in &data.time {
                writeln!(file, "{}", t)?;
            }
        }

        Ok(())
    }

    async fn run(
        &self,
        args: &Args,              // Struct containing test parameters
        mask: u32,                // Mask value for GPU processing
        test_label: &str,         // Label for the test (e.g., CSV file name)
        pass: Box<dyn PassTrait>, // Boxed pass logic
    ) {
        let param_info: Vec<u32> =
            vec![self.size, div_round_up(self.size, 4), self.work_tiles, mask];

        self.gpu_context.queue.write_buffer(
            &self.gpu_buffers.params,
            0,
            bytemuck::cast_slice(&param_info),
        );

        let mut total_spins: u32 = 0;
        let mut fallbacks_initiated: u32 = 0;
        let mut successful_insertions: u32 = 0;
        let mut lookback_length: u32 = 0;

        let mut data = DataStruct::new();
        if args.should_record {
            data.resize(args.batch_size as usize);
        }

        let mut tests_passed: u32 = 0;
        let mut total_time: u64 = 0;
        for i in 0..=args.batch_size {
            let mut command =
                self.gpu_context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Command Encoder"),
                    });

            self.init_pass(&mut command);
            pass.main_pass(self, &mut command); // Using the passed-in main pass logic
            if args.should_time {
                self.resolve_time_query(&mut command, pass.pass_count());
            }
            self.gpu_context.queue.submit(Some(command.finish()));

            // The first test is always discarded to prep caches and TLB
            if i != 0 {
                if args.should_time {
                    let t = self.time(pass.pass_count() as usize);
                    self.gpu_buffers.timestamp_readback.unmap();
                    total_time += t;
                    if args.should_record {
                        data.time[(i - 1) as usize] = self.size as f64 / t as f64;
                    }
                }

                if args.should_validate {
                    let test_passed = validate(self, &self.gpu_shaders.validate);
                    if test_passed {
                        tests_passed += 1;
                    }
                }

                if args.should_get_stats {
                    let mut stats_out: Vec<u32> = vec![];
                    get_stats(self, &mut stats_out);
                    total_spins += stats_out[0];
                    fallbacks_initiated += stats_out[1];
                    successful_insertions += stats_out[2];
                    lookback_length += stats_out[3];
                    if args.should_record {
                        data.total_spins[(i - 1) as usize] =
                            stats_out[0] as f64 / self.work_tiles as f64;
                        data.fallbacks_initiated[(i - 1) as usize] = stats_out[1];
                        data.successful_insertions[(i - 1) as usize] = stats_out[2];
                        data.lookback_length[(i - 1) as usize] =
                            stats_out[3] as f64 / self.work_tiles as f64;
                    }
                }
            }
        }

        println!("\n\n{}", test_label);
        if args.should_readback {
            self.readback_results(args.readback_size);
            self.gpu_buffers.readback.unmap();
        }

        if args.should_time {
            let mut f_time = total_time as f64;
            f_time /= 1_000_000_000.0;
            println!("Total time elapsed: {}", f_time);
            let speed = ((self.size as u64) * (args.batch_size as u64)) as f64
                / (f_time * self.gpu_context.timestamp_freq as f64);
            println!("Estimated speed {:e} ele/s", speed);
        }

        if args.should_validate {
            if tests_passed == args.batch_size {
                println!("ALL TESTS PASSED: {} / {}", tests_passed, args.batch_size);
            } else {
                println!("TESTS FAILED: {} / {}", tests_passed, args.batch_size);
            }
        }

        if args.should_get_stats {
            println!("Estimated Occupancy: {}", self.get_occupancy());
            let avg_total_spins =
                total_spins as f64 / self.work_tiles as f64 / args.batch_size as f64;
            let avg_fallback_init = fallbacks_initiated as f64 / args.batch_size as f64;
            let avg_success_insert = successful_insertions as f64 / args.batch_size as f64;
            let avg_lookback_length =
                lookback_length as f64 / args.batch_size as f64 / self.work_tiles as f64;
            println!("\nThread Blocks Launched: {}", self.work_tiles);
            println!("Average Spins per Workgroup per Pass: {}", avg_total_spins);
            println!(
                "Average Lookback Length per Pass per Workgroup: {}",
                avg_lookback_length
            );
            println!(
                "Average Fallbacks Initiated per Pass: {}",
                avg_fallback_init
            );
            println!(
                "Average Successful Fallback Insertions per Pass: {}",
                avg_success_insert
            );
        }

        if args.should_record {
            if let Err(e) = self.record_to_csv(
                args.should_get_stats,
                &data,
                &format!("{}{}", args.csv_name, test_label),
            ) {
                eprintln!("Error writing to CSV: {}", e);
            }
        }
    }
}

pub enum TestType {
    Csdl,
    Csdldf,
    Full,
}

fn print_usage() {
    eprintln!("Usage: <TestType: \"csdl\"|\"csdldf\"|\"full\"> [\"record\"] [deviceName]");
}

struct Args {
    batch_size: u32,        // How many tests to run in a batch
    should_validate: bool,  // Whether to validate the results
    should_readback: bool,  // Use readback to sanity check results
    should_time: bool,      // Time results?
    should_record: bool,    // Record results?
    should_get_stats: bool, // Collect statistics?
    csv_name: String,       // Name of the CSV file
    readback_size: u32,     // How many elements to readback, must be less than max
}

impl Args {
    fn new(csv_name: &str, record: bool) -> Self {
        Self {
            batch_size: 500,
            should_validate: true,
            should_readback: false,
            should_time: true,
            should_record: record,
            should_get_stats: false,
            csv_name: csv_name.to_string(),
            readback_size: 256,
        }
    }
}

pub async fn run_the_runner(test_type: &TestType, should_record: bool, csv_name: &str) {
    //Stuff required for the GPU
    let size: u32 = 1 << 25; //Input size to test, must be a multiple of 4
    let tile_size: u32 = 4096; //MUST match tile size described in shaders
    let work_tiles =                   //How many work tiles in the scan
        div_round_up(size, tile_size);
    let max_pass_count: usize = 3; //Max number of passes to track with our query set
    let max_readback_size: usize = 1 << 13; //Max size of our readback buffer
    let misc_size: usize = 5; //Max scratch memory we use to track various stats
    let tester = Tester::init(
        size,
        work_tiles,
        max_pass_count,
        max_readback_size,
        misc_size,
    )
    .await;

    //Set the initial test arguments
    let mut test_args: Args = Args::new(csv_name, should_record);

    //Get the Memcopy speed as a baseline
    test_args.should_validate = false;
    tester
        .run(&test_args, 0u32, "Memcopy", Box::new(MemcpyPass))
        .await;
    test_args.should_validate = true;

    match test_type {
        TestType::Csdl => {
            tester
                .run(&test_args, 0u32, "CSDL", Box::new(CsdlPass))
                .await;
        }
        TestType::Csdldf => {
            tester
                .run(&test_args, 0u32, "CSDLDF", Box::new(CsdldfPass))
                .await;

            println!(
                "\n\nEstimated Workgroup Occupancy CSDLDF: {}",
                tester.get_occupancy()
            );
        }
        TestType::Full => {
            tester.run(&test_args, 0u32, "RTS", Box::new(RtsPass)).await;

            tester
                .run(&test_args, 0u32, "CSDLDF", Box::new(CsdldfPass))
                .await;

            test_args.should_get_stats = true;
            tester
                .run(
                    &test_args,
                    0xffffffffu32,
                    "CSDLDF_Stats",
                    Box::new(CsdldfEmulatePass),
                )
                .await;

            for i in (0..10).rev() {
                tester
                    .run(
                        &test_args,
                        (1 << i) - 1,
                        &format!("CSDLDF_{}", 1 << i),
                        Box::new(CsdldfEmulatePass),
                    )
                    .await;
            }

            println!(
                "\n\nEstimated Workgroup Occupancy CSDLDF:{}",
                tester.get_occupancy()
            );
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // We allow:
    //   1) <TestType>                      e.g. "csdl"
    //   2) <TestType> record               e.g. "csdl record"
    //   3) <TestType> record <csv_name>  e.g.   "csdl record csv_name"

    if args.len() < 2 || args.len() > 4 {
        print_usage();
        std::process::exit(1);
    }

    // Parse the test type
    let test_type = match args[1].as_str() {
        "csdl" => TestType::Csdl,
        "csdldf" => TestType::Csdldf,
        "full" => TestType::Full,
        _ => {
            print_usage();
            std::process::exit(1);
        }
    };

    // Should Record?
    let mut should_record = false;
    if let Some(arg) = args.get(2) {
        if arg == "record" {
            should_record = true;
        } else {
            print_usage();
            std::process::exit(1);
        }
    }
    let csv_name = args.get(3).map(|s| format!("{}_", s)).unwrap_or_default();

    pollster::block_on(run_the_runner(&test_type, should_record, &csv_name));
}
