/******************************************************************************
 * GPUSorting
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 12/6/2024
 * https://github.com/b0nes164/GPUSorting
 *
 ******************************************************************************/

use std::{env, vec};

fn div_round_up(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}

const RADIX: u32 = 256;
const RADIX_BITS: u32 = 8;
const KEY_BITS: u32 = 32;
const SORT_PASSES: u32 = KEY_BITS / RADIX_BITS;

enum SortType {
    DeviceRadixSort,
    OneSweep,
}

struct GPUContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    query_set: wgpu::QuerySet,
    timestamp_freq: f32,
}

impl GPUContext {
    async fn init(max_pass_count: usize) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            count: max_pass_count as u32 * 2u32,
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
    info: wgpu::Buffer,
    bump: wgpu::Buffer,
    sort: wgpu::Buffer,
    alt: wgpu::Buffer,
    payload: wgpu::Buffer,
    alt_payload: wgpu::Buffer,
    hist: wgpu::Buffer,
    pass_hist: wgpu::Buffer,
    timestamp: wgpu::Buffer,
    timestamp_readback: wgpu::Buffer,
    readback: wgpu::Buffer,
    misc: wgpu::Buffer,
    copy: wgpu::Buffer,
}

impl GPUBuffers {
    fn init(
        gpu: &GPUContext,
        size: usize,
        thread_blocks: usize,
        max_pass_count: usize,
        max_readback_size: usize,
        misc_size: usize,
    ) -> Self {
        let buffer_size = (size * std::mem::size_of::<u32>()) as u64;
        let info_size = (4usize * std::mem::size_of::<u32>()) as u64;
        let info = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Info"),
            size: info_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bump = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bump"),
            size: (4usize * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sort"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let alt = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alt"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let payload = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Payload"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let alt_payload = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alt Payload"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let hist = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hist"),
            size: (((SORT_PASSES * RADIX) as usize) * std::mem::size_of::<u32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let pass_hist = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pass Hist"),
            size: ((thread_blocks * (RADIX * SORT_PASSES) as usize) * std::mem::size_of::<u32>())
                as wgpu::BufferAddress,
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

        let copy = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Copy"),
            size: info_size * SORT_PASSES as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        GPUBuffers {
            info,
            bump,
            sort,
            alt,
            payload,
            alt_payload,
            hist,
            pass_hist,
            timestamp,
            timestamp_readback,
            readback,
            misc,
            copy,
        }
    }
}

//For simplicity we are going to use the bind group and layout
//for all of the kernels except the validation
struct ComputeShader {
    bind_group_even: wgpu::BindGroup,
    bind_group_odd: wgpu::BindGroup,
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
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

        let bind_group_even = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Bind Group {}", cs_label)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buffers.info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_buffers.bump.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_buffers.sort.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu_buffers.alt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu_buffers.payload.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gpu_buffers.alt_payload.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: gpu_buffers.hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: gpu_buffers.pass_hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: gpu_buffers.misc.as_entire_binding(),
                },
            ],
        });

        let bind_group_odd = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Bind Group {}", cs_label)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buffers.info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_buffers.bump.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_buffers.alt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu_buffers.sort.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu_buffers.alt_payload.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gpu_buffers.payload.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: gpu_buffers.hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: gpu_buffers.pass_hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
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

        let compute_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Compute Pipeline {}", cs_label)),
                    layout: Some(&pipeline_layout_init),
                    module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });

        ComputeShader {
            bind_group_even,
            bind_group_odd,
            compute_pipeline,
            label: cs_label.to_string(),
        }
    }
}

struct Shaders {
    init: ComputeShader,
    reduce_hist: ComputeShader,
    scan: ComputeShader,
    dvr_pass: ComputeShader,
    global_hist: ComputeShader,
    onesweep_scan: ComputeShader,
    onesweep_pass: ComputeShader,
    //forward_sweep: ComputeShader
    validate: ComputeShader,
}

impl Shaders {
    fn init(gpu: &GPUContext, gpu_buffers: &GPUBuffers) -> Self {
        let init_mod = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("../../SharedShaders/init.wgsl"));
        let dvr_mod = gpu.device.create_shader_module(wgpu::include_wgsl!(
            "../../SharedShaders/deviceRadixSort.wgsl"
        ));
        let onesweep_mod = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("../../SharedShaders/oneSweep.wgsl"));
        let valid_mod = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("../../SharedShaders/validate.wgsl"));

        let init = ComputeShader::init(gpu, gpu_buffers, "main", &init_mod, "Init");
        let reduce_hist =
            ComputeShader::init(gpu, gpu_buffers, "reduce_hist", &dvr_mod, "Reduce Hist");
        let scan = ComputeShader::init(gpu, gpu_buffers, "scan", &dvr_mod, "Scan");
        let dvr_pass = ComputeShader::init(gpu, gpu_buffers, "dvr_pass", &dvr_mod, "DVR Pass");
        let global_hist = ComputeShader::init(
            gpu,
            gpu_buffers,
            "global_hist",
            &onesweep_mod,
            "Global Hist",
        );
        let onesweep_scan = ComputeShader::init(
            gpu,
            gpu_buffers,
            "onesweep_scan",
            &onesweep_mod,
            "Onesweep Scan",
        );
        let onesweep_pass = ComputeShader::init(
            gpu,
            gpu_buffers,
            "onesweep_pass",
            &onesweep_mod,
            "Onesweep Pass",
        );
        let validate = ComputeShader::init(gpu, gpu_buffers, "main", &valid_mod, "Validate");
        Shaders {
            init,
            reduce_hist,
            scan,
            dvr_pass,
            global_hist,
            onesweep_scan,
            onesweep_pass,
            validate,
        }
    }
}

//Setup the intial info, awkward because no push constants
fn set_copy_info(seed: u32, tester: &Tester) {
    let mut info_info: Vec<u32> = Vec::new();
    for i in 0..SORT_PASSES {
        info_info.extend([tester.size, i * RADIX_BITS, tester.pass_thread_blocks, seed]);
    }
    let copy_slice = &tester.gpu_buffers.copy.slice(..);
    copy_slice.map_async(wgpu::MapMode::Write, |result| {
        result.unwrap();
    });
    tester.gpu_context.device.poll(wgpu::Maintain::wait());
    {
        let mut mapped = copy_slice.get_mapped_range_mut();
        mapped.copy_from_slice(bytemuck::cast_slice(&info_info));
    }
    tester.gpu_buffers.copy.unmap();
}

fn update_info(pass_index: u32, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
    let info_size = (4 * std::mem::size_of::<u32>()) as u64;
    com_encoder.copy_buffer_to_buffer(
        &tester.gpu_buffers.copy,
        info_size * pass_index as u64,
        &tester.gpu_buffers.info,
        0u64,
        info_size,
    );
}

fn set_compute_pass(
    pass_index: u32,
    query: &wgpu::QuerySet,
    cs: &ComputeShader,
    com_encoder: &mut wgpu::CommandEncoder,
    thread_blocks: u32,
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
    if (pass_index & 1) == 1 {
        pass.set_bind_group(0, &cs.bind_group_odd, &[]);
    } else {
        pass.set_bind_group(0, &cs.bind_group_even, &[]);
    }
    pass.dispatch_workgroups(thread_blocks, 1, 1);
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

fn validate_base(tester: &Tester, cs: &ComputeShader) -> bool {
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
        valid_pass.set_bind_group(0, &cs.bind_group_even, &[]);
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

fn validate_key(tester: &Tester) -> bool {
    validate_base(tester, &tester.gpu_shaders.validate)
}

fn _validate_pair(tester: &Tester) -> bool {
    validate_base(tester, &tester.gpu_shaders.validate) //TODO
}

trait SortLogic {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder);
    fn validation_pass(&self, tester: &Tester) -> bool;
    fn launch_count(&self) -> u32;
}

struct DeviceRadixSort;
impl SortLogic for DeviceRadixSort {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        for i in 0..SORT_PASSES {
            if i != 0 {
                update_info(i, tester, com_encoder);
            }

            set_compute_pass(
                i,
                &tester.gpu_context.query_set,
                &tester.gpu_shaders.reduce_hist,
                com_encoder,
                tester.reduce_thread_blocks,
                i * 6u32,
            );
            set_compute_pass(
                i,
                &tester.gpu_context.query_set,
                &tester.gpu_shaders.scan,
                com_encoder,
                RADIX,
                i * 6u32 + 2u32,
            );
            set_compute_pass(
                i,
                &tester.gpu_context.query_set,
                &tester.gpu_shaders.dvr_pass,
                com_encoder,
                tester.pass_thread_blocks,
                i * 6u32 + 4u32,
            );
        }
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_key(tester)
    }

    fn launch_count(&self) -> u32 {
        3 * SORT_PASSES
    }
}

struct OneSweep;
impl SortLogic for OneSweep {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            0u32,
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.global_hist,
            com_encoder,
            tester.reduce_thread_blocks,
            0u32,
        );
        set_compute_pass(
            0u32,
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.onesweep_scan,
            com_encoder,
            SORT_PASSES,
            2u32,
        );

        for i in 0u32..SORT_PASSES {
            if i != 0u32 {
                update_info(i, tester, com_encoder);
            }
            set_compute_pass(
                i,
                &tester.gpu_context.query_set,
                &tester.gpu_shaders.onesweep_pass,
                com_encoder,
                tester.pass_thread_blocks,
                i * 2u32 + 4u32,
            );
        }
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_key(tester)
    }

    fn launch_count(&self) -> u32 {
        2 + SORT_PASSES
    }
}

struct Tester {
    gpu_context: GPUContext,
    gpu_buffers: GPUBuffers,
    gpu_shaders: Shaders,
    size: u32,
    pass_thread_blocks: u32,
    reduce_thread_blocks: u32,
}

impl Tester {
    async fn init(
        size: u32,
        pass_thread_blocks: u32,
        reduce_thread_blocks: u32,
        max_pass_count: usize,
        max_readback_size: usize,
        misc_size: usize,
    ) -> Self {
        let gpu_context = GPUContext::init(max_pass_count).await;
        let gpu_buffers = GPUBuffers::init(
            &gpu_context,
            size as usize,
            pass_thread_blocks as usize,
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
            pass_thread_blocks,
            reduce_thread_blocks,
        }
    }

    fn init_pass(&self, com_encoder: &mut wgpu::CommandEncoder) {
        update_info(0, &self, com_encoder);
        let mut init_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Init Pass"),
            timestamp_writes: None,
        });
        init_pass.set_pipeline(&self.gpu_shaders.init.compute_pipeline);
        init_pass.set_bind_group(0, &self.gpu_shaders.init.bind_group_even, &[]);
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
            &self.gpu_buffers.sort,
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
        //println!("{:?}", data_out);
        for i in 0..readback_size {
            print!("{}, ", data_out[i as usize]);
        }
    }

    async fn run(
        &self,
        should_readback: bool,
        should_time: bool,
        should_validate: bool,
        readback_size: u32,
        batch_size: u32,
        pass: Box<dyn SortLogic>,
    ) {
        let mut tests_passed: u32 = 0;
        let mut total_time: u64 = 0;
        for i in 0..batch_size {
            let mut command =
                self.gpu_context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Command Encoder"),
                    });
            set_copy_info(i + 10, &self);
            self.init_pass(&mut command);
            pass.main_pass(self, &mut command);
            if should_time {
                self.resolve_time_query(&mut command, pass.launch_count())
            }
            self.gpu_context.queue.submit(Some(command.finish()));

            //The first test is always discarded to prep caches and TLB
            if should_time && i != 0u32 {
                total_time += self.time(pass.launch_count() as usize);
                self.gpu_buffers.timestamp_readback.unmap();
            }

            if should_validate {
                let test_passed = pass.validation_pass(self);
                if test_passed {
                    tests_passed += 1u32;
                }
            }
        }

        if should_readback {
            self.readback_results(readback_size);
            self.gpu_buffers.readback.unmap();
        }

        if should_time {
            let mut f_time = total_time as f64;
            f_time /= 1000000000.0f64;
            println!("\nTotal time elapsed: {}", f_time);
            let speed = ((self.size as u64) * ((batch_size - 1) as u64)) as f64
                / (f_time * self.gpu_context.timestamp_freq as f64);
            println!("Estimated speed {:e} ele/s", speed);
        }

        if should_validate {
            if tests_passed == batch_size {
                println!("ALL TESTS PASSED: {} / {}", tests_passed, batch_size);
            } else {
                println!("TESTS FAILED: {} / {}", tests_passed, batch_size);
            }
        }
    }

    pub async fn run_test(
        &self,
        should_readback: bool,
        should_time: bool,
        should_validate: bool,
        readback_size: u32,
        args: Vec<String>,
    ) {
        let sort_type = match args[1].as_str() {
            "dvr" => Some(SortType::DeviceRadixSort),
            "one" => Some(SortType::OneSweep),
            _ => None,
        };

        let sort_type = match sort_type {
            Some(sort_type) => sort_type,
            None => {
                eprintln!("Error: unknown sort type {}", &args[1]);
                return;
            }
        };

        let pass: Box<dyn SortLogic> = match sort_type {
            SortType::DeviceRadixSort => Box::new(DeviceRadixSort),
            SortType::OneSweep => Box::new(OneSweep),
        };

        let batch_size: u32 = match args[3].parse() {
            Ok(num) => num,
            Err(_) => {
                eprintln!("Error: Batch Size must be a positive integer");
                std::process::exit(1);
            }
        };

        self.run(
            should_readback,
            should_time,
            should_validate,
            readback_size,
            batch_size,
            pass,
        )
        .await;
    }
}

//warning, absolutely no guard rails
pub async fn run_the_runner(args: Vec<String>) {
    let pow_of_two: u32 = match args[2].parse() {
        Ok(num) if (num < 26) => num,
        Ok(_) => {
            eprintln!("Error: input size power must be a value between 0 and 25");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("Error: input size power must be a positive integer");
            std::process::exit(1);
        }
    };

    let size: u32 = 1 << pow_of_two; //Input size to test, must be a multiple of 4
    let part_size: u32 = 3840; //MUST match partition size described in shaders
    let reduce_part_size = 3840; //MUST match partition size described in shaders
    let pass_thread_blocks =                //Thread Blocks to launch based on input
        div_round_up(size, part_size);
    let reduce_thread_blocks = div_round_up(size, reduce_part_size);
    let max_pass_count: usize = 16; //Max number of passes to track with our query set
    let max_readback_size: usize = 8192; //Max size of our readback buffer
    let misc_size: usize = 4; //Max scratch memory we use to track various stats
    let tester = Tester::init(
        size,
        pass_thread_blocks,
        reduce_thread_blocks,
        max_pass_count,
        max_readback_size,
        misc_size,
    )
    .await;

    let should_validate = true; //Perform validation?
    let should_readback = false; //Use readback to sanity check results
    let should_time = true; //Time results?
    let readback_size = 256; //How many elements to readback, must be less than max
    tester
        .run_test(
            should_readback,
            should_time,
            should_validate,
            readback_size,
            args,
        )
        .await;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "Usage: <Sort Type: String>
            <Input Size as Power of Two: u32> <Test Batch Size: u32>"
        );
        std::process::exit(1);
    }
    pollster::block_on(run_the_runner(args));
}
