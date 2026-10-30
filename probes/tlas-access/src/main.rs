//! Driver probe: how compute ray queries reach a TLAS on a descriptor-heap device, plus the
//! heap texture, untyped pointer and mesh-shader paths the GPU interface depends on.
//!
//! `tlas-access` runs every case in its own child process so one device loss cannot hide
//! another case; `tlas-access info` prints device facts; `tlas-access run <case>` runs one case.
//! Each case prints a `RESULT <case> <status> <detail>` line; the exit code is 0 on PASS.
//!
//! Build from this directory with `cargo build --release --target-dir ../../target/probes`.
//! Shaders need the pinned Slang in `.tools/slang-2026.20` (or `ORR_SLANGC`) and `VULKAN_SDK`.
//! Set `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` to count validation messages per case.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use ash::vk::{self, TaggedStructure};
use std::collections::BTreeMap;
use std::ffi::{CStr, c_void};
use std::io::Cursor;
use std::sync::Mutex;
use std::time::Instant;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;
const ITERATIONS: u32 = 32;
const CUSTOM_INDEX: u32 = 41;
const TEX_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const RT_SIZE: u32 = 64;
const FENCE_TIMEOUT_NS: u64 = 10_000_000_000;
/// Triangle in the BLAS, in the unit square the rays cover.
const TRIANGLE: [[f32; 3]; 3] = [[0.1, 0.1, 0.0], [0.9, 0.2, 0.0], [0.4, 0.85, 0.0]];
/// Mesh-shader triangle, clip space.
const MESH_TRIANGLE: [[f32; 4]; 3] = [[-0.8, -0.8, 0.0, 1.0], [0.8, -0.6, 0.0, 1.0], [-0.2, 0.7, 0.0, 1.0]];

const CASES: &[&str] = &[
    "heap_glsl",
    "heap_slang",
    "address",
    "address_ptr",
    "address_null",
    "map_const_offset",
    "map_push_index",
    "map_push_address",
    "map_indirect_address",
    "classic",
    "classic_address",
    "mixed",
    "heap_texture",
    "mesh",
];

macro_rules! spv {
    ($name:literal) => {
        ash::util::read_spv(&mut Cursor::new(&include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))[..]))
            .expect("valid SPIR-V")
    };
}

/// Shader root, C layout; mirrors `Root` in shaders/common.slang.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Root {
    out: u64,
    tlas: u64,
    heap_index: u32,
    width: u32,
    height: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MeshRoot {
    vertices: u64,
    texture_index: u32,
    pad: [u32; 3],
}

/// Push data: root address at 0, TLAS address at 8, heap index at 16.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Push {
    root: u64,
    tlas: u64,
    index: u32,
    pad: u32,
}

static MESSAGES: Mutex<BTreeMap<String, (u32, String)>> = Mutex::new(BTreeMap::new());

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let data = &*data;
    let id = if data.p_message_id_name.is_null() {
        "?".to_owned()
    } else {
        CStr::from_ptr(data.p_message_id_name).to_string_lossy().into_owned()
    };
    let message = if data.p_message.is_null() {
        String::new()
    } else {
        CStr::from_ptr(data.p_message).to_string_lossy().into_owned()
    };
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) { "E" } else { "W" };
    let mut map = MESSAGES.lock().unwrap();
    let entry = map.entry(format!("{level} {id}")).or_insert((0, message));
    entry.0 += 1;
    vk::FALSE
}

struct Buf {
    buffer: vk::Buffer,
    /// Aligned device address.
    addr: u64,
    /// Offset of `addr` inside `buffer`.
    offset: u64,
    ptr: *mut u8,
}

struct Image {
    image: vk::Image,
    view: vk::ImageView,
}

struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    pdev: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    mem: vk::PhysicalDeviceMemoryProperties,
    heap_props: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    as_props: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    limits: vk::PhysicalDeviceLimits,
    heap: ash::ext::descriptor_heap::Device,
    accel: ash::khr::acceleration_structure::Device,
    mesh: ash::ext::mesh_shader::Device,
}

const REQUIRED_EXTENSIONS: &[&CStr] = &[
    ash::khr::acceleration_structure::NAME,
    ash::khr::deferred_host_operations::NAME,
    ash::khr::ray_query::NAME,
    ash::ext::mesh_shader::NAME,
    ash::ext::descriptor_heap::NAME,
    ash::khr::shader_untyped_pointers::NAME,
];

const OPTIONAL_EXTENSIONS: &[&CStr] = &[
    c"VK_NV_cluster_acceleration_structure",
    c"VK_NV_partitioned_acceleration_structure",
    c"VK_EXT_ray_tracing_invocation_reorder",
    c"VK_NV_ray_tracing_invocation_reorder",
    c"VK_KHR_opacity_micromap",
    c"VK_EXT_opacity_micromap",
    c"VK_KHR_device_address_commands",
    c"VK_KHR_unified_image_layouts",
    c"VK_NV_low_latency2",
    c"VK_NV_device_diagnostic_checkpoints",
    c"VK_NV_device_diagnostics_config",
    c"VK_EXT_device_fault",
    c"VK_KHR_swapchain",
    c"VK_KHR_swapchain_maintenance1",
    c"VK_EXT_swapchain_maintenance1",
    c"VK_KHR_present_id2",
    c"VK_KHR_present_wait2",
    c"VK_EXT_hdr_metadata",
    c"VK_KHR_pipeline_binary",
    c"VK_EXT_shader_object",
    c"VK_EXT_descriptor_buffer",
    c"VK_NV_cooperative_vector",
    c"VK_KHR_cooperative_matrix",
    c"VK_EXT_memory_budget",
    c"VK_EXT_memory_priority",
    c"VK_EXT_pageable_device_local_memory",
    c"VK_KHR_maintenance9",
    c"VK_KHR_maintenance10",
    c"VK_NVX_binary_import",
    c"VK_NVX_image_view_handle",
];

fn has_ext(list: &[vk::ExtensionProperties], name: &CStr) -> bool {
    list.iter().any(|e| e.extension_name_as_c_str() == Ok(name))
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

impl Gpu {
    unsafe fn new() -> Gpu {
        let entry = ash::Entry::load().expect("Vulkan loader");
        let instance_exts = entry.enumerate_instance_extension_properties(None).unwrap();
        let debug = has_ext(&instance_exts, ash::ext::debug_utils::NAME);
        let app = vk::ApplicationInfo::default().application_name(c"tlas-access").api_version(vk::API_VERSION_1_4);
        let mut ext_names = Vec::new();
        if debug {
            ext_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let instance = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&ext_names), None)
            .expect("instance");
        if debug {
            let du = ash::ext::debug_utils::Instance::load(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            // Leaked on purpose: lives for the process.
            let _ = du.create_debug_utils_messenger(&info, None);
        }

        let pdev = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .find(|&p| {
                let exts = instance.enumerate_device_extension_properties(p).unwrap();
                instance.get_physical_device_properties(p).api_version >= vk::API_VERSION_1_4
                    && REQUIRED_EXTENSIONS.iter().all(|n| has_ext(&exts, n))
            })
            .expect("no Vulkan 1.4 device with descriptor heap, ray query and mesh shaders");

        let family = instance
            .get_physical_device_queue_family_properties(pdev)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
            .unwrap() as u32;

        let mut f_heap = vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default().descriptor_heap(true);
        let mut f_untyped = vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR::default().shader_untyped_pointers(true);
        let mut f_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
        let mut f_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut f_mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().task_shader(true).mesh_shader(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .scalar_block_layout(true)
            .timeline_semaphore(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true)
            .maintenance4(true);
        let mut f14 = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true).push_descriptor(true);
        let core = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .shader_storage_image_read_without_format(true)
            .shader_storage_image_write_without_format(true);
        let mut f2 = vk::PhysicalDeviceFeatures2::default().features(core);

        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
        let names: Vec<_> = REQUIRED_EXTENSIONS.iter().map(|n| n.as_ptr()).collect();
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&names)
            .push(&mut f2)
            .push(&mut f12)
            .push(&mut f13)
            .push(&mut f14)
            .push(&mut f_heap)
            .push(&mut f_untyped)
            .push(&mut f_as)
            .push(&mut f_rq)
            .push(&mut f_mesh);
        let device = instance.create_device(pdev, &info, None).expect("device");
        let queue = device.get_device_queue(family, 0);
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();

        let mut heap_props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
        let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut heap_props).push(&mut as_props);
        instance.get_physical_device_properties2(pdev, &mut props2);
        let limits = props2.properties.limits;
        heap_props.p_next = std::ptr::null_mut();
        as_props.p_next = std::ptr::null_mut();

        Gpu {
            heap: ash::ext::descriptor_heap::Device::load(&instance, &device),
            accel: ash::khr::acceleration_structure::Device::load(&instance, &device),
            mesh: ash::ext::mesh_shader::Device::load(&instance, &device),
            mem: instance.get_physical_device_memory_properties(pdev),
            _entry: entry,
            instance,
            pdev,
            device,
            queue,
            pool,
            heap_props,
            as_props,
            limits,
        }
    }

    unsafe fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> u32 {
        (0..self.mem.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem.memory_types[i as usize].property_flags.contains(flags))
            .expect("memory type")
    }

    /// Buffer whose returned address is aligned to `align`; host-visible buffers are mapped.
    unsafe fn buffer(&self, size: u64, usage: vk::BufferUsageFlags, host: bool, align: u64) -> Buf {
        let usage = usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let buffer = self
            .device
            .create_buffer(&vk::BufferCreateInfo::default().size(size + align).usage(usage), None)
            .unwrap();
        let req = self.device.get_buffer_memory_requirements(buffer);
        let flags = if host {
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let mut flags_info = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(self.memory_type(req.memory_type_bits, flags))
                    .push(&mut flags_info),
                None,
            )
            .unwrap();
        self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
        let base = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
        let addr = align_up(base, align.max(1));
        let offset = addr - base;
        let ptr = if host {
            (self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap() as *mut u8).add(offset as usize)
        } else {
            std::ptr::null_mut()
        };
        Buf { buffer, addr, offset, ptr }
    }

    unsafe fn image(&self, format: vk::Format, size: u32, usage: vk::ImageUsageFlags, view: bool) -> Image {
        let image = self
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D { width: size, height: size, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage),
                None,
            )
            .unwrap();
        let req = self.device.get_image_memory_requirements(image);
        let memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)),
                None,
            )
            .unwrap();
        self.device.bind_image_memory(image, memory, 0).unwrap();
        let view = if view {
            self.device.create_image_view(&view_info(image, format), None).unwrap()
        } else {
            vk::ImageView::null()
        };
        Image { image, view }
    }

    unsafe fn begin(&self) -> vk::CommandBuffer {
        let cmd = self
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1),
            )
            .unwrap()[0];
        self.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
            .unwrap();
        cmd
    }

    /// Submits and waits; device loss and timeouts end the process with a RESULT line.
    unsafe fn submit_wait(&self, cmd: vk::CommandBuffer, case: &str) {
        self.device.end_command_buffer(cmd).unwrap();
        let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        if let Err(e) = self.device.queue_submit(self.queue, &[submit], fence) {
            finish(case, if e == vk::Result::ERROR_DEVICE_LOST { "DEVICE_LOST" } else { "SUBMIT_FAILED" }, &format!("{e:?}"));
        }
        match self.device.wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS) {
            Ok(()) => {}
            Err(vk::Result::ERROR_DEVICE_LOST) => finish(case, "DEVICE_LOST", "vkWaitForFences"),
            Err(vk::Result::TIMEOUT) => finish(case, "TIMEOUT", "fence not signalled in 10 s"),
            Err(e) => finish(case, "WAIT_FAILED", &format!("{e:?}")),
        }
        self.device.destroy_fence(fence, None);
    }

    unsafe fn barrier(&self, cmd: vk::CommandBuffer, src: vk::PipelineStageFlags2, sa: vk::AccessFlags2, dst: vk::PipelineStageFlags2, da: vk::AccessFlags2) {
        let b = [vk::MemoryBarrier2::default().src_stage_mask(src).src_access_mask(sa).dst_stage_mask(dst).dst_access_mask(da)];
        self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
    }

    unsafe fn to_general(&self, cmd: vk::CommandBuffer, image: vk::Image) {
        let b = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(image)
            .subresource_range(color_range())];
        self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b));
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)
}

fn view_info(image: vk::Image, format: vk::Format) -> vk::ImageViewCreateInfo<'static> {
    vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(color_range())
}

fn finish(case: &str, status: &str, detail: &str) -> ! {
    let map = MESSAGES.lock().unwrap();
    let errors: u32 = map.iter().filter(|(k, _)| k.starts_with('E')).map(|(_, v)| v.0).sum();
    let warnings: u32 = map.iter().filter(|(k, _)| k.starts_with('W')).map(|(_, v)| v.0).sum();
    for (id, (count, message)) in map.iter() {
        let short: String = message.chars().take(400).collect();
        eprintln!("VVL {case} {id} x{count}: {short}");
    }
    println!("RESULT {case} {status} {detail} (validation errors={errors} warnings={warnings})");
    std::process::exit(if status == "PASS" { 0 } else { 2 });
}

/// Scene: one-triangle BLAS, one-instance TLAS.
struct Scene {
    tlas: vk::AccelerationStructureKHR,
    tlas_addr: u64,
}

unsafe fn build_scene(gpu: &Gpu, case: &str) -> Scene {
    let input = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
    let storage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR;
    let scratch_align = u64::from(gpu.as_props.min_acceleration_structure_scratch_offset_alignment);

    let verts = gpu.buffer(36, input, true, 16);
    std::ptr::copy_nonoverlapping(TRIANGLE.as_ptr().cast::<u8>(), verts.ptr, 36);

    let make = |ty: vk::AccelerationStructureTypeKHR, geometry: vk::AccelerationStructureGeometryKHR<'static>, count: u32| {
        let geometries = [geometry];
        let mut build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.accel.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &build, Some(&[count]), &mut sizes);
        let backing = gpu.buffer(sizes.acceleration_structure_size, storage, false, 256);
        let accel = gpu
            .accel
            .create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(backing.buffer)
                    .offset(backing.offset)
                    .size(sizes.acceleration_structure_size)
                    .ty(ty),
                None,
            )
            .unwrap();
        let scratch = gpu.buffer(sizes.build_scratch_size, vk::BufferUsageFlags::STORAGE_BUFFER, false, scratch_align);
        build = build.dst_acceleration_structure(accel).scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch.addr });
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)];
        let cmd = gpu.begin();
        gpu.accel.cmd_build_acceleration_structures(cmd, &[build], &[Some(&ranges)]);
        gpu.barrier(
            cmd,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );
        gpu.submit_wait(cmd, case);
        let addr = gpu
            .accel
            .get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(accel));
        (accel, addr)
    };

    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: verts.addr })
        .vertex_stride(12)
        .max_vertex(2)
        .index_type(vk::IndexType::NONE_KHR);
    let blas_geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE);
    let (_blas, blas_addr) = make(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, blas_geometry, 1);

    let instance = vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR { matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] },
        instance_custom_index_and_mask: vk::Packed24_8::new(CUSTOM_INDEX, 0xff),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            0,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: blas_addr },
    };
    let instances = gpu.buffer(64, input, true, 16);
    std::ptr::copy_nonoverlapping((&raw const instance).cast::<u8>(), instances.ptr, 64);
    let tlas_geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .data(vk::DeviceOrHostAddressConstKHR { device_address: instances.addr }),
        });
    let (tlas, tlas_addr) = make(vk::AccelerationStructureTypeKHR::TOP_LEVEL, tlas_geometry, 1);
    Scene { tlas, tlas_addr }
}

/// Resource heap: storage image at index 0, sampled image at 1 (image-descriptor stride), TLAS
/// descriptor at `as_offset` (a multiple of the buffer-descriptor size). Sampler heap: one
/// nearest sampler at index 0.
struct Heaps {
    resource_bind: vk::BindHeapInfoEXT<'static>,
    sampler_bind: vk::BindHeapInfoEXT<'static>,
    as_offset: u64,
    as_descriptor: Vec<u8>,
}

unsafe fn create_heaps(gpu: &Gpu, scene: &Scene, texture: &Image) -> Heaps {
    let p = &gpu.heap_props;
    let image_size = p.image_descriptor_size;
    let buffer_size = p.buffer_descriptor_size;
    let as_offset = align_up(2 * image_size, buffer_size.max(8));
    let used = as_offset + buffer_size;
    let reserved_offset = align_up(used, p.image_descriptor_alignment.max(p.buffer_descriptor_alignment));
    let resource_size = reserved_offset + p.min_resource_heap_reserved_range;
    let resource = gpu.buffer(resource_size, vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT, true, p.resource_heap_alignment);
    std::ptr::write_bytes(resource.ptr, 0, resource_size as usize);

    let storage_view = view_info(texture.image, TEX_FORMAT);
    let sampled_view = view_info(texture.image, TEX_FORMAT);
    let storage_image = vk::ImageDescriptorInfoEXT::default().view(&storage_view).layout(vk::ImageLayout::GENERAL);
    let sampled_image = vk::ImageDescriptorInfoEXT::default().view(&sampled_view).layout(vk::ImageLayout::GENERAL);
    let as_range = vk::DeviceAddressRangeEXT { address: scene.tlas_addr, size: 0 };
    let infos = [
        vk::ResourceDescriptorInfoEXT::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .data(vk::ResourceDescriptorDataEXT { p_image: &storage_image }),
        vk::ResourceDescriptorInfoEXT::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .data(vk::ResourceDescriptorDataEXT { p_image: &sampled_image }),
        vk::ResourceDescriptorInfoEXT::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .data(vk::ResourceDescriptorDataEXT { p_address_range: &as_range }),
    ];
    let dst = |offset: u64, size: u64| vk::HostAddressRangeEXT::default().address(std::slice::from_raw_parts_mut(resource.ptr.add(offset as usize), size as usize));
    let ranges = [dst(0, image_size), dst(image_size, image_size), dst(as_offset, buffer_size)];
    gpu.heap.write_resource_descriptors(&infos, &ranges).expect("vkWriteResourceDescriptorsEXT");
    let as_descriptor = std::slice::from_raw_parts(resource.ptr.add(as_offset as usize), buffer_size as usize).to_vec();

    let sampler_reserved = align_up(p.sampler_descriptor_size, p.sampler_descriptor_alignment.max(1));
    let sampler_size = sampler_reserved + p.min_sampler_heap_reserved_range;
    let sampler = gpu.buffer(sampler_size, vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT, true, p.sampler_heap_alignment);
    let nearest = [vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)];
    let sampler_dst = [vk::HostAddressRangeEXT::default().address(std::slice::from_raw_parts_mut(sampler.ptr, p.sampler_descriptor_size as usize))];
    gpu.heap.write_sampler_descriptors(&nearest, &sampler_dst).expect("vkWriteSamplerDescriptorsEXT");

    let bind = |addr: u64, reserved_offset: u64, reserved: u64| vk::BindHeapInfoEXT {
        heap_range: vk::DeviceAddressRangeEXT { address: addr, size: reserved_offset + reserved },
        reserved_range_offset: reserved_offset,
        reserved_range_size: reserved,
        ..Default::default()
    };
    Heaps {
        resource_bind: bind(resource.addr, reserved_offset, p.min_resource_heap_reserved_range),
        sampler_bind: bind(sampler.addr, sampler_reserved, p.min_sampler_heap_reserved_range),
        as_offset,
        as_descriptor,
    }
}

#[derive(Clone, Copy)]
enum Binding {
    /// Shader binds nothing at set 0 binding 1.
    None,
    ConstOffset(u32),
    PushIndex { stride: u32 },
    PushAddress,
    IndirectAddress,
}

fn root_mapping() -> vk::DescriptorSetAndBindingMappingEXT<'static> {
    vk::DescriptorSetAndBindingMappingEXT {
        descriptor_set: 0,
        first_binding: 0,
        binding_count: 1,
        resource_mask: vk::SpirvResourceTypeFlagsEXT::UNIFORM_BUFFER | vk::SpirvResourceTypeFlagsEXT::READ_ONLY_STORAGE_BUFFER,
        source: vk::DescriptorMappingSourceEXT::PUSH_ADDRESS,
        source_data: vk::DescriptorMappingSourceDataEXT { push_address_offset: 0 },
        ..Default::default()
    }
}

fn as_mapping(binding: Binding) -> Option<vk::DescriptorSetAndBindingMappingEXT<'static>> {
    let (source, source_data) = match binding {
        Binding::None => return None,
        Binding::ConstOffset(offset) => (
            vk::DescriptorMappingSourceEXT::HEAP_WITH_CONSTANT_OFFSET,
            vk::DescriptorMappingSourceDataEXT {
                constant_offset: vk::DescriptorMappingSourceConstantOffsetEXT { heap_offset: offset, ..Default::default() },
            },
        ),
        Binding::PushIndex { stride } => (
            vk::DescriptorMappingSourceEXT::HEAP_WITH_PUSH_INDEX,
            vk::DescriptorMappingSourceDataEXT {
                push_index: vk::DescriptorMappingSourcePushIndexEXT { heap_offset: 0, push_offset: 16, heap_index_stride: stride, ..Default::default() },
            },
        ),
        Binding::PushAddress => (
            vk::DescriptorMappingSourceEXT::PUSH_ADDRESS,
            vk::DescriptorMappingSourceDataEXT { push_address_offset: 8 },
        ),
        Binding::IndirectAddress => (
            vk::DescriptorMappingSourceEXT::INDIRECT_ADDRESS,
            vk::DescriptorMappingSourceDataEXT {
                indirect_address: vk::DescriptorMappingSourceIndirectAddressEXT { push_offset: 0, address_offset: 8 },
            },
        ),
    };
    Some(vk::DescriptorSetAndBindingMappingEXT {
        descriptor_set: 0,
        first_binding: 1,
        binding_count: 1,
        resource_mask: vk::SpirvResourceTypeFlagsEXT::ACCELERATION_STRUCTURE,
        source,
        source_data,
        ..Default::default()
    })
}

/// Heap-mode compute pipeline when `layout` is null, classic otherwise.
unsafe fn compute_pipeline(gpu: &Gpu, code: &[u32], entry: &CStr, binding: Binding, layout: vk::PipelineLayout) -> vk::Pipeline {
    let mut module = vk::ShaderModuleCreateInfo::default().code(code);
    let mut mappings = vec![root_mapping()];
    mappings.extend(as_mapping(binding));
    let mut mapping_info = vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&mappings);
    let mut flags = vk::PipelineCreateFlags2CreateInfo::default().flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
    let heap_mode = layout == vk::PipelineLayout::null();
    let mut stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).name(entry).push(&mut module);
    if heap_mode {
        stage = stage.push(&mut mapping_info);
    }
    let mut info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
    if heap_mode {
        info = info.push(&mut flags);
    }
    gpu.device
        .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        .map_err(|(_, e)| e)
        .expect("compute pipeline")[0]
}

unsafe fn bind_heaps(gpu: &Gpu, cmd: vk::CommandBuffer, heaps: &Heaps) {
    gpu.heap.cmd_bind_resource_heap(cmd, &heaps.resource_bind);
    gpu.heap.cmd_bind_sampler_heap(cmd, &heaps.sampler_bind);
}

unsafe fn push(gpu: &Gpu, cmd: vk::CommandBuffer, data: &Push) {
    let info = vk::PushDataInfoEXT {
        offset: 0,
        data: vk::HostAddressRangeConstEXT::default().address(std::slice::from_raw_parts((data as *const Push).cast::<u8>(), size_of::<Push>())),
        ..Default::default()
    };
    gpu.heap.cmd_push_data(cmd, &info);
}

/// CPU reference for one ray: Some(expected value), or None within 1e-4 of an edge.
fn expected_hit(x: u32, y: u32) -> Option<u32> {
    let p = [(x as f32 + 0.5) / WIDTH as f32, (y as f32 + 0.5) / HEIGHT as f32];
    let edge = |a: [f32; 3], b: [f32; 3]| {
        let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
        ((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / (ex * ex + ey * ey).sqrt()
    };
    let e = [edge(TRIANGLE[0], TRIANGLE[1]), edge(TRIANGLE[1], TRIANGLE[2]), edge(TRIANGLE[2], TRIANGLE[0])];
    if e.iter().any(|d| d.abs() < 1e-4) {
        return None;
    }
    let inside = e.iter().all(|&d| d > 0.0) || e.iter().all(|&d| d < 0.0);
    Some(if inside { CUSTOM_INDEX + 1 } else { 0 })
}

/// Compares a ray-query output buffer with the CPU reference; returns (mismatches, hits).
unsafe fn check_rays(out: &Buf) -> (u32, u32) {
    let values = std::slice::from_raw_parts(out.ptr.cast::<u32>(), (WIDTH * HEIGHT) as usize);
    let (mut bad, mut hits) = (0, 0);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let v = values[(y * WIDTH + x) as usize];
            if v == CUSTOM_INDEX + 1 {
                hits += 1;
            }
            if let Some(e) = expected_hit(x, y)
                && e != v
            {
                bad += 1;
            }
        }
    }
    (bad, hits)
}

struct Ctx {
    gpu: Gpu,
    scene: Scene,
    texture: Image,
    heaps: Heaps,
    roots: Buf,
    timestamps: vk::QueryPool,
}

impl Ctx {
    unsafe fn out_buffer(&self) -> Buf {
        let out = self.gpu.buffer(u64::from(WIDTH * HEIGHT * 4), vk::BufferUsageFlags::STORAGE_BUFFER, true, 16);
        std::ptr::write_bytes(out.ptr, 0xff, (WIDTH * HEIGHT * 4) as usize);
        out
    }

    /// Writes root `slot` (256-byte stride so classic uniform-buffer offsets stay aligned).
    unsafe fn root(&self, slot: u64, root: Root) -> (u64, u64) {
        let offset = slot * 256;
        std::ptr::write(self.roots.ptr.add(offset as usize).cast::<Root>(), root);
        (self.roots.addr + offset, self.roots.offset + offset)
    }

    unsafe fn ray_root(&self, out: &Buf, heap_index: u32) -> Root {
        Root { out: out.addr, tlas: self.scene.tlas_addr, heap_index, width: WIDTH, height: HEIGHT, pad: 0 }
    }

    unsafe fn dispatch_rays(&self, cmd: vk::CommandBuffer, iterations: u32) {
        for _ in 0..iterations {
            self.gpu.device.cmd_dispatch(cmd, WIDTH / 8, HEIGHT / 8, 1);
            self.gpu.barrier(
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::HOST,
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::HOST_READ,
            );
        }
    }

    unsafe fn gpu_ms(&self) -> f64 {
        let mut ticks = [0u64; 2];
        self.gpu
            .device
            .get_query_pool_results(self.timestamps, 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
            .unwrap();
        (ticks[1] - ticks[0]) as f64 * f64::from(self.gpu.limits.timestamp_period) / 1e6
    }

    /// Zero TLAS address: every ray must miss (no device loss).
    unsafe fn address_null_case(&self, case: &str) {
        let gpu = &self.gpu;
        let pipeline = compute_pipeline(gpu, &spv!("rq_address"), c"main", Binding::None, vk::PipelineLayout::null());
        let out = self.out_buffer();
        let (root_addr, _) = self.root(0, Root { tlas: 0, ..self.ray_root(&out, 0) });
        let cmd = gpu.begin();
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        bind_heaps(gpu, cmd, &self.heaps);
        push(gpu, cmd, &Push { root: root_addr, ..Default::default() });
        self.dispatch_rays(cmd, 1);
        gpu.submit_wait(cmd, case);
        let values = value_histogram(&out);
        let ok = values == [(0, WIDTH * HEIGHT)];
        finish(case, if ok { "PASS" } else { "WRONG" }, &format!("values={values:?}"));
    }

    /// Root holds a pointer to a GPU word holding the TLAS address.
    unsafe fn address_ptr_case(&self, case: &str) {
        let table = self.gpu.buffer(8, vk::BufferUsageFlags::STORAGE_BUFFER, true, 16);
        std::ptr::write(table.ptr.cast::<u64>(), self.scene.tlas_addr);
        self.heap_case_root(case, &spv!("rq_address_ptr"), Binding::None, 0, table.addr);
    }

    /// One heap-mode ray-query case: ITERATIONS dispatches of WIDTH x HEIGHT rays.
    unsafe fn heap_case(&self, case: &str, code: &[u32], binding: Binding, heap_index: u32) {
        self.heap_case_root(case, code, binding, heap_index, self.scene.tlas_addr);
    }

    unsafe fn heap_case_root(&self, case: &str, code: &[u32], binding: Binding, heap_index: u32, tlas_field: u64) {
        let gpu = &self.gpu;
        let pipeline = compute_pipeline(gpu, code, c"main", binding, vk::PipelineLayout::null());
        let out = self.out_buffer();
        let (root_addr, _) = self.root(0, Root { tlas: tlas_field, ..self.ray_root(&out, heap_index) });
        let cmd = gpu.begin();
        gpu.device.cmd_reset_query_pool(cmd, self.timestamps, 0, 2);
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        bind_heaps(gpu, cmd, &self.heaps);
        push(gpu, cmd, &Push { root: root_addr, tlas: self.scene.tlas_addr, index: heap_index, pad: 0 });
        gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, self.timestamps, 0);
        self.dispatch_rays(cmd, ITERATIONS);
        gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, self.timestamps, 1);
        gpu.submit_wait(cmd, case);
        let (bad, hits) = check_rays(&out);
        let ms = self.gpu_ms() / f64::from(ITERATIONS);
        let mut detail = format!("hits={hits} mismatches={bad} gpu_ms_per_1M_rays={ms:.4}");
        if bad > 0 {
            detail += &format!(" values={:?}", value_histogram(&out));
        }
        finish(case, if bad == 0 && hits > 0 { "PASS" } else { "WRONG" }, &detail);
    }

    unsafe fn classic_layout(&self) -> vk::PipelineLayout {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout = self
            .gpu
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
                    .bindings(&bindings),
                None,
            )
            .unwrap();
        let layouts = [set_layout];
        self.gpu
            .device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None)
            .unwrap()
    }

    unsafe fn push_classic(&self, cmd: vk::CommandBuffer, layout: vk::PipelineLayout, root_offset: u64) {
        let buffer_info = [vk::DescriptorBufferInfo { buffer: self.roots.buffer, offset: root_offset, range: size_of::<Root>() as u64 }];
        let structures = [self.scene.tlas];
        let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&structures);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .push(&mut as_write),
        ];
        self.gpu.device.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &writes);
    }

    /// Classic (non-heap) pipeline with push descriptors on the descriptor-heap device.
    unsafe fn classic_case(&self, case: &str, code: &[u32]) {
        let gpu = &self.gpu;
        let layout = self.classic_layout();
        let pipeline = compute_pipeline(gpu, code, c"main", Binding::None, layout);
        let out = self.out_buffer();
        let (_, root_offset) = self.root(0, self.ray_root(&out, 0));
        let cmd = gpu.begin();
        gpu.device.cmd_reset_query_pool(cmd, self.timestamps, 0, 2);
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        self.push_classic(cmd, layout, root_offset);
        gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, self.timestamps, 0);
        self.dispatch_rays(cmd, ITERATIONS);
        gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, self.timestamps, 1);
        gpu.submit_wait(cmd, case);
        let (bad, hits) = check_rays(&out);
        let ms = self.gpu_ms() / f64::from(ITERATIONS);
        let detail = format!("hits={hits} mismatches={bad} gpu_ms_per_1M_rays={ms:.4}");
        finish(case, if bad == 0 && hits > 0 { "PASS" } else { "WRONG" }, &detail);
    }

    /// Heap pipeline, classic pipeline, heap pipeline again in one command buffer.
    unsafe fn mixed_case(&self, case: &str) {
        let gpu = &self.gpu;
        let code = spv!("rq_binding");
        let layout = self.classic_layout();
        let heap_a = compute_pipeline(gpu, &code, c"main", Binding::IndirectAddress, vk::PipelineLayout::null());
        let classic = compute_pipeline(gpu, &code, c"main", Binding::None, layout);
        let heap_b = compute_pipeline(gpu, &code, c"main", Binding::ConstOffset(self.heaps.as_offset as u32), vk::PipelineLayout::null());
        let outs = [self.out_buffer(), self.out_buffer(), self.out_buffer()];
        let roots: Vec<_> = outs.iter().enumerate().map(|(i, o)| self.root(i as u64, self.ray_root(o, 0))).collect();
        let cmd = gpu.begin();
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, heap_a);
        bind_heaps(gpu, cmd, &self.heaps);
        push(gpu, cmd, &Push { root: roots[0].0, ..Default::default() });
        self.dispatch_rays(cmd, 1);
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, classic);
        self.push_classic(cmd, layout, roots[1].1);
        self.dispatch_rays(cmd, 1);
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, heap_b);
        bind_heaps(gpu, cmd, &self.heaps);
        push(gpu, cmd, &Push { root: roots[2].0, ..Default::default() });
        self.dispatch_rays(cmd, 1);
        gpu.submit_wait(cmd, case);
        let results: Vec<_> = outs.iter().map(|o| check_rays(o)).collect();
        let ok = results.iter().all(|&(bad, hits)| bad == 0 && hits > 0);
        let detail = format!("heap_indirect={:?} classic={:?} heap_const={:?} (mismatches, hits)", results[0], results[1], results[2]);
        finish(case, if ok { "PASS" } else { "WRONG" }, &detail);
    }

    unsafe fn record_texture_write(&self, cmd: vk::CommandBuffer, slot: u64) {
        let gpu = &self.gpu;
        let write = compute_pipeline(gpu, &spv!("tex_write"), c"write_main", Binding::None, vk::PipelineLayout::null());
        let (root_addr, _) = self.root(slot, Root { width: WIDTH, height: HEIGHT, ..Default::default() });
        gpu.to_general(cmd, self.texture.image);
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, write);
        bind_heaps(gpu, cmd, &self.heaps);
        push(gpu, cmd, &Push { root: root_addr, ..Default::default() });
        gpu.device.cmd_dispatch(cmd, WIDTH / 8, HEIGHT / 8, 1);
        gpu.barrier(
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
    }

    unsafe fn texture_case(&self, case: &str) {
        let gpu = &self.gpu;
        let read = compute_pipeline(gpu, &spv!("tex_read"), c"read_main", Binding::None, vk::PipelineLayout::null());
        let out = self.out_buffer();
        let (root_addr, _) = self.root(1, Root { out: out.addr, width: WIDTH, height: HEIGHT, ..Default::default() });
        let cmd = gpu.begin();
        self.record_texture_write(cmd, 0);
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, read);
        push(gpu, cmd, &Push { root: root_addr, ..Default::default() });
        self.dispatch_rays(cmd, 1);
        gpu.submit_wait(cmd, case);
        let values = std::slice::from_raw_parts(out.ptr.cast::<u32>(), (WIDTH * HEIGHT) as usize);
        let good = values.iter().filter(|&&v| v == 1).count();
        let total = values.len();
        finish(case, if good == total { "PASS" } else { "WRONG" }, &format!("matching_texels={good}/{total}"));
    }

    unsafe fn mesh_case(&self, case: &str) {
        let gpu = &self.gpu;
        let stages_code = [spv!("mesh_task"), spv!("mesh_mesh"), spv!("mesh_pixel")];
        let stage_flags = [vk::ShaderStageFlags::TASK_EXT, vk::ShaderStageFlags::MESH_EXT, vk::ShaderStageFlags::FRAGMENT];
        let names = [c"task_main", c"mesh_main", c"pixel_main"];
        let mappings = [root_mapping()];
        let mut modules: Vec<_> = stages_code.iter().map(|c| vk::ShaderModuleCreateInfo::default().code(c)).collect();
        let mut mapping_infos: Vec<_> =
            (0..3).map(|_| vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&mappings)).collect();
        let stages: Vec<_> = modules
            .iter_mut()
            .zip(mapping_infos.iter_mut())
            .enumerate()
            .map(|(i, (m, mi))| vk::PipelineShaderStageCreateInfo::default().stage(stage_flags[i]).name(names[i]).push(m).push(mi))
            .collect();
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachments = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [vk::Format::R8G8B8A8_UNORM];
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let mut flags = vk::PipelineCreateFlags2CreateInfo::default().flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .push(&mut rendering)
            .push(&mut flags);
        let pipeline = gpu
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, e)| e)
            .expect("mesh pipeline")[0];

        let target = gpu.image(vk::Format::R8G8B8A8_UNORM, RT_SIZE, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, true);
        let readback = gpu.buffer(u64::from(RT_SIZE * RT_SIZE * 4), vk::BufferUsageFlags::TRANSFER_DST, true, 16);
        let verts = gpu.buffer(48, vk::BufferUsageFlags::STORAGE_BUFFER, true, 16);
        std::ptr::copy_nonoverlapping(MESH_TRIANGLE.as_ptr().cast::<u8>(), verts.ptr, 48);
        let mesh_root = MeshRoot { vertices: verts.addr, texture_index: 1, pad: [0; 3] };
        std::ptr::write(self.roots.ptr.add(512).cast::<MeshRoot>(), mesh_root);
        let root_addr = self.roots.addr + 512;

        let cmd = gpu.begin();
        self.record_texture_write(cmd, 0);
        gpu.to_general(cmd, target.image);
        let color = [vk::RenderingAttachmentInfo::default()
            .image_view(target.view)
            .image_layout(vk::ImageLayout::GENERAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue::default())];
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: RT_SIZE, height: RT_SIZE } };
        gpu.device.cmd_begin_rendering(cmd, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&color));
        gpu.device.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: RT_SIZE as f32, height: RT_SIZE as f32, min_depth: 0.0, max_depth: 1.0 }]);
        gpu.device.cmd_set_scissor(cmd, 0, &[area]);
        gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        bind_heaps(gpu, cmd, &self.heaps);
        push(gpu, cmd, &Push { root: root_addr, ..Default::default() });
        gpu.mesh.cmd_draw_mesh_tasks(cmd, 1, 1, 1);
        gpu.device.cmd_end_rendering(cmd);
        gpu.barrier(
            cmd,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let region = vk::BufferImageCopy::default()
            .buffer_offset(readback.offset)
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: RT_SIZE, height: RT_SIZE, depth: 1 });
        gpu.device.cmd_copy_image_to_buffer(cmd, target.image, vk::ImageLayout::GENERAL, readback.buffer, &[region]);
        gpu.barrier(cmd, vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE, vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ);
        gpu.submit_wait(cmd, case);

        let pixels = std::slice::from_raw_parts(readback.ptr.cast::<[u8; 4]>(), (RT_SIZE * RT_SIZE) as usize);
        let (mut bad, mut covered) = (0, 0);
        for y in 0..RT_SIZE {
            for x in 0..RT_SIZE {
                let px = pixels[(y * RT_SIZE + x) as usize];
                let lit = px[0] == 255 && px[1].abs_diff(128) <= 1 && px[2].abs_diff(64) <= 1 && px[3] == 255;
                let blank = px == [0, 0, 0, 0];
                if lit {
                    covered += 1;
                }
                match mesh_expected(x, y) {
                    Some(true) if !lit => bad += 1,
                    Some(false) if !blank => bad += 1,
                    _ => {}
                }
            }
        }
        finish(case, if bad == 0 && covered > 0 { "PASS" } else { "WRONG" }, &format!("covered={covered} mismatches={bad}"));
    }
}

/// Mesh coverage at a pixel centre, or None within 0.02 clip units of an edge.
fn mesh_expected(x: u32, y: u32) -> Option<bool> {
    let p = [(x as f32 + 0.5) / RT_SIZE as f32 * 2.0 - 1.0, (y as f32 + 0.5) / RT_SIZE as f32 * 2.0 - 1.0];
    let edge = |a: [f32; 4], b: [f32; 4]| {
        let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
        ((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / (ex * ex + ey * ey).sqrt()
    };
    let t = MESH_TRIANGLE;
    let e = [edge(t[0], t[1]), edge(t[1], t[2]), edge(t[2], t[0])];
    if e.iter().any(|d| d.abs() < 0.02) {
        return None;
    }
    Some(e.iter().all(|&d| d > 0.0) || e.iter().all(|&d| d < 0.0))
}

unsafe fn setup(case: &str) -> Ctx {
    let gpu = Gpu::new();
    let scene = build_scene(&gpu, case);
    let texture = gpu.image(TEX_FORMAT, WIDTH, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED, false);
    let heaps = create_heaps(&gpu, &scene, &texture);
    let ub_align = gpu.limits.min_uniform_buffer_offset_alignment.max(256);
    let roots = gpu.buffer(4096, vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER, true, ub_align);
    let timestamps = gpu
        .device
        .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None)
        .unwrap();
    Ctx { gpu, scene, texture, heaps, roots, timestamps }
}

unsafe fn run(case: &str) {
    let ctx = setup(case);
    let heap_index_8 = (ctx.heaps.as_offset / 8) as u32;
    let buffer_size = ctx.gpu.heap_props.buffer_descriptor_size;
    let heap_index_desc = (ctx.heaps.as_offset / buffer_size) as u32;
    match case {
        "heap_glsl" => ctx.heap_case(case, &spv!("rq_heap_glsl"), Binding::None, heap_index_desc),
        "heap_slang" => ctx.heap_case(case, &spv!("rq_heap_slang"), Binding::None, heap_index_8),
        "address" => ctx.heap_case(case, &spv!("rq_address"), Binding::None, 0),
        "address_ptr" => ctx.address_ptr_case(case),
        "address_null" => ctx.address_null_case(case),
        "map_const_offset" => ctx.heap_case(case, &spv!("rq_binding"), Binding::ConstOffset(ctx.heaps.as_offset as u32), 0),
        "map_push_index" => {
            ctx.heap_case(case, &spv!("rq_binding"), Binding::PushIndex { stride: buffer_size as u32 }, heap_index_desc)
        }
        "map_push_address" => ctx.heap_case(case, &spv!("rq_binding"), Binding::PushAddress, 0),
        "map_indirect_address" => ctx.heap_case(case, &spv!("rq_binding"), Binding::IndirectAddress, 0),
        "classic" => ctx.classic_case(case, &spv!("rq_binding")),
        "classic_address" => ctx.classic_case(case, &spv!("rq_address")),
        "mixed" => ctx.mixed_case(case),
        "heap_texture" => ctx.texture_case(case),
        "mesh" => ctx.mesh_case(case),
        _ => finish(case, "UNKNOWN_CASE", "see CASES"),
    }
}

unsafe fn info() {
    let ctx = setup("info");
    let gpu = &ctx.gpu;
    let props = gpu.instance.get_physical_device_properties(gpu.pdev);
    let v = props.driver_version;
    let name = props.device_name_as_c_str().unwrap().to_string_lossy();
    println!("device: {name}");
    println!("driver: {}.{:02} (raw {v:#x})", v >> 22, (v >> 14) & 0xff);
    println!(
        "api: {}.{}.{}",
        vk::api_version_major(props.api_version),
        vk::api_version_minor(props.api_version),
        vk::api_version_patch(props.api_version)
    );
    let h = &gpu.heap_props;
    println!(
        "heap: sampler_size={} image_size={} buffer_size={} sampler_align={} image_align={} buffer_align={}",
        h.sampler_descriptor_size,
        h.image_descriptor_size,
        h.buffer_descriptor_size,
        h.sampler_descriptor_alignment,
        h.image_descriptor_alignment,
        h.buffer_descriptor_alignment
    );
    println!(
        "heap: resource_heap_align={} sampler_heap_align={} max_resource_heap={} max_sampler_heap={} min_resource_reserved={} min_sampler_reserved={} min_sampler_reserved_with_embedded={} max_push_data={} max_embedded_samplers={}",
        h.resource_heap_alignment,
        h.sampler_heap_alignment,
        h.max_resource_heap_size,
        h.max_sampler_heap_size,
        h.min_resource_heap_reserved_range,
        h.min_sampler_heap_reserved_range,
        h.min_sampler_heap_reserved_range_with_embedded,
        h.max_push_data_size,
        h.max_descriptor_heap_embedded_samplers
    );
    let a = &gpu.as_props;
    println!(
        "accel: max_instances={} max_geometry={} max_primitives={} scratch_align={}",
        a.max_instance_count, a.max_geometry_count, a.max_primitive_count, a.min_acceleration_structure_scratch_offset_alignment
    );
    println!("tlas_address: {:#018x}", ctx.scene.tlas_addr);
    println!("tlas_heap_descriptor: {:02x?}", ctx.heaps.as_descriptor);
    println!(
        "tlas_descriptor_is_address: {}",
        ctx.heaps.as_descriptor.len() >= 8 && ctx.heaps.as_descriptor[..8] == ctx.scene.tlas_addr.to_le_bytes()
    );
    println!("timestamp_period_ns: {}", gpu.limits.timestamp_period);
    println!("min_uniform_buffer_offset_alignment: {}", gpu.limits.min_uniform_buffer_offset_alignment);
    let exts = gpu.instance.enumerate_device_extension_properties(gpu.pdev).unwrap();
    for name in OPTIONAL_EXTENSIONS {
        let spec = exts.iter().find(|e| e.extension_name_as_c_str() == Ok(*name)).map(|e| e.spec_version);
        println!("ext {:<45} {}", name.to_string_lossy(), spec.map_or("absent".to_owned(), |s| format!("rev {s}")));
    }
    for name in REQUIRED_EXTENSIONS {
        let spec = exts.iter().find(|e| e.extension_name_as_c_str() == Ok(*name)).map(|e| e.spec_version);
        println!("ext {:<45} rev {} (required)", name.to_string_lossy(), spec.unwrap_or(0));
    }
    let map = MESSAGES.lock().unwrap();
    for (id, (count, message)) in map.iter() {
        let short: String = message.chars().take(400).collect();
        eprintln!("VVL info {id} x{count}: {short}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("info") => unsafe { info() },
        Some("run") => unsafe { run(args.get(2).expect("case name")) },
        None => {
            let exe = std::env::current_exe().unwrap();
            for case in CASES {
                let start = Instant::now();
                let out = std::process::Command::new(&exe).args(["run", case]).output().unwrap();
                let stdout = String::from_utf8_lossy(&out.stdout);
                let line = stdout
                    .lines()
                    .find(|l| l.starts_with("RESULT"))
                    .map_or_else(|| format!("RESULT {case} CRASHED exit={:?}", out.status.code()), str::to_owned);
                println!("{line} [{:.1}s]", start.elapsed().as_secs_f32());
                for l in String::from_utf8_lossy(&out.stderr).lines().filter(|l| l.starts_with("VVL") || l.contains("panicked")) {
                    println!("    {l}");
                }
            }
        }
        Some(other) => panic!("unknown command {other}; use info, run <case>, or no argument"),
    }
}

/// Most common output values, for diagnosing a WRONG case.
unsafe fn value_histogram(out: &Buf) -> Vec<(u32, u32)> {
    let values = std::slice::from_raw_parts(out.ptr.cast::<u32>(), (WIDTH * HEIGHT) as usize);
    let mut counts = BTreeMap::new();
    for &v in values {
        *counts.entry(v).or_insert(0u32) += 1;
    }
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    counts.truncate(4);
    counts
}
