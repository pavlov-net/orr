//! Driver probe: does inline ray query (compute shaders, no ray-tracing pipeline) traverse
//! VK_NV_cluster_acceleration_structure CLAS / cluster BLAS correctly, including through a
//! VK_NV_partitioned_acceleration_structure TLAS, alpha-tested and OMM clusters, templates and
//! sparse-bound CLAS storage.
//!
//! `cluster-rayquery` runs every case in its own child process so one device loss cannot hide
//! another; `cluster-rayquery info` prints device facts; `cluster-rayquery run <case>` runs one.
//! Each case prints `RESULT <case> <status> <detail>`; the exit code is 0 on PASS.
//!
//! Every case traces W x H orthographic +z rays over the unit square from a compute shader and
//! checks each ray against a CPU reference: hit/miss, instance custom index and index, ClusterID,
//! primitive index, geometry index, t, committed-triangle positions (position fetch) and the
//! nearest candidate seen in the candidate loop. Rays within EDGE_EPS of any triangle edge are
//! skipped.
//!
//! Build from this directory with `cargo build --release --target-dir ../../target/probes`.
//! Shaders need the pinned Slang in `.tools/slang-2026.20` (or `ORR_SLANGC`) and `VULKAN_SDK`.
//! `PROBE_RTP=1` also enables VK_KHR_ray_tracing_pipeline (never used) to see whether the
//! cluster SPIR-V capability's implicit RayTracingKHR dependency changes anything.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use ash::vk::{self, TaggedStructure};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, c_void};
use std::io::Cursor;
use std::sync::Mutex;
use std::time::Instant;

const W: u32 = 512;
const H: u32 = 512;
const REC: usize = 20;
const EDGE_EPS: f32 = 2e-4;
const T_EPS: f32 = 1e-4;
const FENCE_TIMEOUT_NS: u64 = 10_000_000_000;

const MODE_FORCE_NON_OPAQUE: u32 = 1;
const MODE_ALPHA: u32 = 2;
const MODE_OMM_2_STATE: u32 = 4;

/// Cluster geometry flag bits (VkClusterAccelerationStructureGeometryFlagBitsNV).
const GF_CULL_DISABLE: u32 = 1;
const GF_NO_DUP: u32 = 2;
const GF_OPAQUE: u32 = 4;

const CASES: &[&str] = &[
    "clas_tlas",
    "clas_tlas_glsl",
    "clas_tlas_nocap",
    "clas_mixed",
    "clas_ptlas",
    "clas_alpha",
    "clas_omm",
    "clas_template",
    "clas_template_src",
    "clas_sparse",
];

macro_rules! spv {
    ($name:literal) => {
        ash::util::read_spv(&mut Cursor::new(&include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))[..]))
            .expect("valid SPIR-V")
    };
}

// ---------------------------------------------------------------------------------------------
// Shader-visible structs (mirrors of shaders/args.slang and shaders/trace.slang).

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ArgsRoot {
    src: u64,
    dst: u64,
    count_dst: u64,
    aux: u64,
    n: u32,
    op: u32,
    pad: [u32; 2],
}

const OP_CLUSTER: u32 = 0;
const OP_TEMPLATE: u32 = 1;
const OP_INSTANTIATE: u32 = 2;
const OP_GATHER: u32 = 3;
const OP_BLAS: u32 = 4;
const OP_INSTANCE: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ClusterDesc {
    index_addr: u64,
    vertex_addr: u64,
    gflags_addr: u64,
    omm_array: u64,
    omm_index: u64,
    bbox_addr: u64,
    id: u32,
    tris: u32,
    verts: u32,
    index_type: u32,
    truncate: u32,
    base_geom: u32,
    geom_flags: u32,
    cluster_flags: u32,
    index_stride: u32,
    vertex_stride: u32,
    gflags_stride: u32,
    omm_index_type: u32,
    omm_index_stride: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct InstantiateDesc {
    template_addr_ptr: u64,
    vertex_addr: u64,
    vertex_stride: u64,
    cluster_id_offset: u32,
    geometry_index_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BlasDesc {
    count: u32,
    first: u32,
    pad: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct InstanceDesc {
    m: [f32; 12],
    custom: u32,
    mask: u32,
    flags: u32,
    blas_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TraceRoot {
    out: u64,
    tlas: u64,
    width: u32,
    height: u32,
    mode: u32,
    pad: u32,
}

const _: () = {
    assert!(size_of::<ClusterDesc>() == 112);
    assert!(size_of::<InstantiateDesc>() == 32);
    assert!(size_of::<BlasDesc>() == 16);
    assert!(size_of::<InstanceDesc>() == 64);
    assert!(size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV>() == 64);
    assert!(size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV>() == 72);
    assert!(size_of::<vk::ClusterAccelerationStructureInstantiateClusterInfoNV>() == 32);
    assert!(size_of::<vk::ClusterAccelerationStructureBuildClustersBottomLevelInfoNV>() == 16);
    assert!(size_of::<vk::AccelerationStructureInstanceKHR>() == 64);
};

fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn translate(t: [f32; 3]) -> [f32; 12] {
    [1.0, 0.0, 0.0, t[0], 0.0, 1.0, 0.0, t[1], 0.0, 0.0, 1.0, t[2]]
}

// ---------------------------------------------------------------------------------------------
// Validation message collection and result reporting.

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

fn dump_messages(case: &str) -> (u32, u32) {
    let map = MESSAGES.lock().unwrap();
    let errors: u32 = map.iter().filter(|(k, _)| k.starts_with('E')).map(|(_, v)| v.0).sum();
    let warnings: u32 = map.iter().filter(|(k, _)| k.starts_with('W')).map(|(_, v)| v.0).sum();
    for (id, (count, message)) in map.iter() {
        let short: String = message.chars().take(600).collect();
        eprintln!("VVL {case} {id} x{count}: {short}");
    }
    (errors, warnings)
}

fn finish(case: &str, status: &str, detail: &str) -> ! {
    let (errors, warnings) = dump_messages(case);
    println!("RESULT {case} {status} {detail} (validation errors={errors} warnings={warnings})");
    std::process::exit(if status == "PASS" { 0 } else { 2 });
}

// ---------------------------------------------------------------------------------------------
// Device.

#[derive(Clone, Copy, PartialEq)]
enum Mem {
    Device,
    Upload,
    Readback,
}

#[derive(Clone, Copy)]
struct Buf {
    buffer: vk::Buffer,
    addr: u64,
    ptr: *mut u8,
    size: u64,
}

impl Buf {
    unsafe fn write<T: Copy>(&self, offset: usize, data: &[T]) {
        let b = bytes(data);
        assert!(offset + b.len() <= self.size as usize);
        std::ptr::copy_nonoverlapping(b.as_ptr(), self.ptr.add(offset), b.len());
    }

    unsafe fn read<T: Copy>(&self, count: usize) -> Vec<T> {
        std::slice::from_raw_parts(self.ptr.cast::<T>(), count).to_vec()
    }
}

struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    pdev: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_flags: vk::QueueFlags,
    pool: vk::CommandPool,
    mem: vk::PhysicalDeviceMemoryProperties,
    limits: vk::PhysicalDeviceLimits,
    exts: Vec<vk::ExtensionProperties>,
    as_props: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    cl_props: vk::PhysicalDeviceClusterAccelerationStructurePropertiesNV<'static>,
    pt_props: vk::PhysicalDevicePartitionedAccelerationStructurePropertiesNV<'static>,
    omm_props: vk::PhysicalDeviceOpacityMicromapPropertiesEXT<'static>,
    features: vk::PhysicalDeviceFeatures,
    has_ptlas: bool,
    has_omm: bool,
    has_rtp: bool,
    extra_usage: vk::BufferUsageFlags,
    accel: ash::khr::acceleration_structure::Device,
    cluster: ash::nv::cluster_acceleration_structure::Device,
    ptlas: Option<ash::nv::partitioned_acceleration_structure::Device>,
    omm: Option<ash::ext::opacity_micromap::Device>,
    layout: vk::PipelineLayout,
    timestamps: vk::QueryPool,
}

const REQUIRED_EXTENSIONS: &[&CStr] = &[
    ash::khr::acceleration_structure::NAME,
    ash::khr::deferred_host_operations::NAME,
    ash::khr::ray_query::NAME,
    ash::khr::ray_tracing_position_fetch::NAME,
    ash::nv::cluster_acceleration_structure::NAME,
];

const REPORTED_EXTENSIONS: &[&CStr] = &[
    c"VK_NV_cluster_acceleration_structure",
    c"VK_NV_partitioned_acceleration_structure",
    c"VK_KHR_acceleration_structure",
    c"VK_KHR_ray_query",
    c"VK_KHR_ray_tracing_pipeline",
    c"VK_KHR_ray_tracing_position_fetch",
    c"VK_EXT_opacity_micromap",
    c"VK_KHR_opacity_micromap",
    c"VK_NV_displacement_micromap",
    c"VK_NV_ray_tracing_linear_swept_spheres",
    c"VK_NV_ray_tracing_validation",
    c"VK_EXT_ray_tracing_invocation_reorder",
    c"VK_NV_ray_tracing_invocation_reorder",
];

fn has_ext(list: &[vk::ExtensionProperties], name: &CStr) -> bool {
    list.iter().any(|e| e.extension_name_as_c_str() == Ok(name))
}

fn ext_rev(list: &[vk::ExtensionProperties], name: &CStr) -> Option<u32> {
    list.iter().find(|e| e.extension_name_as_c_str() == Ok(name)).map(|e| e.spec_version)
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a.max(1)) * a.max(1)
}

impl Gpu {
    unsafe fn new() -> Gpu {
        let entry = ash::Entry::load().expect("Vulkan loader");
        let instance_exts = entry.enumerate_instance_extension_properties(None).unwrap();
        let debug = has_ext(&instance_exts, ash::ext::debug_utils::NAME);
        let app = vk::ApplicationInfo::default().application_name(c"cluster-rayquery").api_version(vk::API_VERSION_1_4);
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
            .expect("no Vulkan 1.4 device with cluster acceleration structures, ray query and position fetch");
        let exts = instance.enumerate_device_extension_properties(pdev).unwrap();
        let has_ptlas = has_ext(&exts, ash::nv::partitioned_acceleration_structure::NAME);
        let has_omm = has_ext(&exts, ash::ext::opacity_micromap::NAME);
        let has_rtp = std::env::var("PROBE_RTP").is_ok_and(|v| v == "1") && has_ext(&exts, ash::khr::ray_tracing_pipeline::NAME);

        let families = instance.get_physical_device_queue_family_properties(pdev);
        let family = families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::SPARSE_BINDING))
            .or_else(|| families.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)))
            .unwrap();
        let queue_flags = families[family].queue_flags;

        let features = instance.get_physical_device_features(pdev);
        let mut f_cluster = vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV::default().cluster_acceleration_structure(true);
        let mut f_ptlas = vk::PhysicalDevicePartitionedAccelerationStructureFeaturesNV::default().partitioned_acceleration_structure(true);
        let mut f_omm = vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default().micromap(true);
        let mut f_fetch = vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default().ray_tracing_position_fetch(true);
        let mut f_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
        let mut f_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut f_rtp = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true).scalar_block_layout(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true).maintenance4(true);
        let mut f14 = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true);
        let core = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .sparse_binding(features.sparse_binding == vk::TRUE)
            .sparse_residency_buffer(features.sparse_residency_buffer == vk::TRUE);
        let mut f2 = vk::PhysicalDeviceFeatures2::default().features(core);

        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family as u32).queue_priorities(&priorities)];
        let mut names: Vec<_> = REQUIRED_EXTENSIONS.iter().map(|n| n.as_ptr()).collect();
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .push(&mut f2)
            .push(&mut f12)
            .push(&mut f13)
            .push(&mut f14)
            .push(&mut f_as)
            .push(&mut f_rq)
            .push(&mut f_fetch)
            .push(&mut f_cluster);
        if has_ptlas {
            names.push(ash::nv::partitioned_acceleration_structure::NAME.as_ptr());
            info = info.push(&mut f_ptlas);
        }
        if has_omm {
            names.push(ash::ext::opacity_micromap::NAME.as_ptr());
            info = info.push(&mut f_omm);
        }
        if has_rtp {
            names.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
            info = info.push(&mut f_rtp);
        }
        let info = info.enabled_extension_names(&names);
        let device = instance.create_device(pdev, &info, None).expect("device");
        let queue = device.get_device_queue(family as u32, 0);
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family as u32)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();

        let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut cl_props = vk::PhysicalDeviceClusterAccelerationStructurePropertiesNV::default();
        let mut pt_props = vk::PhysicalDevicePartitionedAccelerationStructurePropertiesNV::default();
        let mut omm_props = vk::PhysicalDeviceOpacityMicromapPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut as_props).push(&mut cl_props);
        if has_ptlas {
            props2 = props2.push(&mut pt_props);
        }
        if has_omm {
            props2 = props2.push(&mut omm_props);
        }
        instance.get_physical_device_properties2(pdev, &mut props2);
        let limits = props2.properties.limits;
        as_props.p_next = std::ptr::null_mut();
        cl_props.p_next = std::ptr::null_mut();
        pt_props.p_next = std::ptr::null_mut();
        omm_props.p_next = std::ptr::null_mut();

        let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(64)];
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges), None).unwrap();
        let timestamps = device
            .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(32), None)
            .unwrap();
        let extra_usage = if has_omm {
            vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT | vk::BufferUsageFlags::MICROMAP_STORAGE_EXT
        } else {
            vk::BufferUsageFlags::empty()
        };

        Gpu {
            accel: ash::khr::acceleration_structure::Device::load(&instance, &device),
            cluster: ash::nv::cluster_acceleration_structure::Device::load(&instance, &device),
            ptlas: has_ptlas.then(|| ash::nv::partitioned_acceleration_structure::Device::load(&instance, &device)),
            omm: has_omm.then(|| ash::ext::opacity_micromap::Device::load(&instance, &device)),
            mem: instance.get_physical_device_memory_properties(pdev),
            _entry: entry,
            instance,
            pdev,
            device,
            queue,
            queue_flags,
            pool,
            limits,
            exts,
            as_props,
            cl_props,
            pt_props,
            omm_props,
            features,
            has_ptlas,
            has_omm,
            has_rtp,
            extra_usage,
            layout,
            timestamps,
        }
    }

    unsafe fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem.memory_types[i as usize].property_flags.contains(flags))
    }

    fn usage(&self) -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | self.extra_usage
    }

    /// Buffer whose returned address is aligned to `align`; host-visible kinds are mapped.
    unsafe fn buffer(&self, size: u64, kind: Mem, align: u64) -> Buf {
        let size = size.max(16);
        let align = align.max(16);
        let buffer = self
            .device
            .create_buffer(&vk::BufferCreateInfo::default().size(size + align).usage(self.usage()), None)
            .unwrap();
        let req = self.device.get_buffer_memory_requirements(buffer);
        let device_local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let ty = match kind {
            Mem::Device => self.memory_type(req.memory_type_bits, device_local),
            Mem::Upload => self.memory_type(req.memory_type_bits, device_local | host).or_else(|| self.memory_type(req.memory_type_bits, host)),
            Mem::Readback => self
                .memory_type(req.memory_type_bits, host | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| self.memory_type(req.memory_type_bits, host)),
        }
        .expect("memory type");
        let mut flags_info = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = self
            .device
            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push(&mut flags_info), None)
            .unwrap();
        self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
        let base = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
        let addr = align_up(base, align);
        let offset = addr - base;
        let ptr = if kind == Mem::Device {
            std::ptr::null_mut()
        } else {
            let p = (self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap() as *mut u8).add(offset as usize);
            std::ptr::write_bytes(p, 0, size as usize);
            p
        };
        Buf { buffer, addr, ptr, size }
    }

    unsafe fn upload<T: Copy>(&self, data: &[T]) -> Buf {
        let b = self.buffer(std::mem::size_of_val(data) as u64, Mem::Upload, 256);
        b.write(0, data);
        b
    }

    unsafe fn pipeline(&self, code: &[u32]) -> vk::Pipeline {
        let module = vk::ShaderModuleCreateInfo::default().code(code);
        let mut module = module;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).name(c"main").push(&mut module);
        let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout);
        self.device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, e)| e)
            .expect("compute pipeline")[0]
    }

    unsafe fn begin(&self) -> vk::CommandBuffer {
        let cmd = self
            .device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1))
            .unwrap()[0];
        self.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
            .unwrap();
        cmd
    }

    unsafe fn wait(&self, fence: vk::Fence, case: &str, what: &str) {
        match self.device.wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS) {
            Ok(()) => {}
            Err(vk::Result::ERROR_DEVICE_LOST) => finish(case, "DEVICE_LOST", &format!("vkWaitForFences after {what}")),
            Err(vk::Result::TIMEOUT) => finish(case, "TIMEOUT", &format!("fence not signalled in 10 s after {what}")),
            Err(e) => finish(case, "WAIT_FAILED", &format!("{e:?} after {what}")),
        }
        self.device.destroy_fence(fence, None);
    }

    /// Submits and waits; device loss and timeouts end the process with a RESULT line.
    unsafe fn submit_wait(&self, cmd: vk::CommandBuffer, case: &str, what: &str) {
        self.full_barrier(cmd, true);
        self.device.end_command_buffer(cmd).unwrap();
        let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        if let Err(e) = self.device.queue_submit(self.queue, &[submit], fence) {
            finish(case, if e == vk::Result::ERROR_DEVICE_LOST { "DEVICE_LOST" } else { "SUBMIT_FAILED" }, &format!("{e:?} at {what}"));
        }
        self.wait(fence, case, what);
    }

    /// Everything before -> everything after. The probe measures correctness, not overlap.
    unsafe fn full_barrier(&self, cmd: vk::CommandBuffer, host: bool) {
        let mut dst_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
        let mut dst_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        if host {
            dst_stage |= vk::PipelineStageFlags2::HOST;
            dst_access |= vk::AccessFlags2::HOST_READ;
        }
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)];
        self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
    }

    unsafe fn push<T: Copy>(&self, cmd: vk::CommandBuffer, data: &T) {
        self.device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes(std::slice::from_ref(data)));
    }

    unsafe fn timestamp(&self, cmd: vk::CommandBuffer, index: u32) {
        self.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, self.timestamps, index);
    }

    /// Microseconds between consecutive timestamps 0..count.
    unsafe fn read_timestamps(&self, count: u32) -> Vec<f64> {
        let mut ticks = vec![0u64; count as usize];
        self.device
            .get_query_pool_results(self.timestamps, 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
            .unwrap();
        ticks.windows(2).map(|w| (w[1] - w[0]) as f64 * f64::from(self.limits.timestamp_period) / 1e3).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Scene description and CPU reference.

#[derive(Clone)]
struct Cluster {
    id: u32,
    verts: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    /// Index size in bytes: 1, 2 or 4.
    index_bytes: u32,
    truncate: u32,
    base_geom: u32,
    geom_flags: u32,
    /// Per-triangle geometry index (bits 0-23) and geometry flags (bits 29-31).
    tri_gflags: Option<Vec<u32>>,
    /// Per-triangle opacity micromap index (or special index).
    omm_index: Option<Vec<i32>>,
}

/// Opacity of a triangle as traversal sees it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Opaque,
    NonOpaque,
    Transparent,
}

#[derive(Clone)]
struct RefTri {
    v: [[f32; 3]; 3],
    cluster: u32,
    prim: u32,
    geom: u32,
    kind: Kind,
    /// Kind under RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE.
    kind_2state: Kind,
    no_dup: bool,
}

#[derive(Clone)]
struct RefInst {
    custom: u32,
    index: u32,
    offset: [f32; 3],
    tris: Vec<RefTri>,
}

/// OMM states of the four micromap entries the OMM case builds (4-state, level 0).
const OMM_STATES: [u8; 4] = [0, 1, 2, 3];

fn alpha_keep(cluster: u32, prim: u32) -> bool {
    cluster.wrapping_add(prim) % 3 != 0
}

fn truncate(v: f32, bits: u32) -> f32 {
    if bits == 0 { v } else { f32::from_bits(v.to_bits() & !((1u32 << bits) - 1)) }
}

/// Grid cluster over [x0, x1] x [y0, y1] with nx x ny quads and alternating diagonals.
fn grid(id: u32, nx: u32, ny: u32, rect: [f32; 4], z: impl Fn(f32, f32) -> f32, index_bytes: u32) -> Cluster {
    let mut verts = Vec::new();
    for j in 0..=ny {
        for i in 0..=nx {
            let x = rect[0] + (rect[2] - rect[0]) * i as f32 / nx as f32;
            let y = rect[1] + (rect[3] - rect[1]) * j as f32 / ny as f32;
            verts.push([x, y, z(x, y)]);
        }
    }
    let mut tris = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let a = j * (nx + 1) + i;
            let (b, c) = (a + 1, a + nx + 1);
            let d = c + 1;
            if (i + j) % 2 == 0 {
                tris.push([a, b, d]);
                tris.push([a, d, c]);
            } else {
                tris.push([a, b, c]);
                tris.push([b, d, c]);
            }
        }
    }
    Cluster {
        id,
        verts,
        tris,
        index_bytes,
        truncate: 0,
        base_geom: 0,
        geom_flags: GF_OPAQUE | GF_CULL_DISABLE,
        tri_gflags: None,
        omm_index: None,
    }
}

fn omm_kind(index: i32, geometry: Kind) -> (Kind, Kind) {
    let state = |s: u8| match s {
        0 => (Kind::Transparent, Kind::Transparent),
        1 => (Kind::Opaque, Kind::Opaque),
        2 => (Kind::NonOpaque, Kind::Transparent),
        _ => (Kind::NonOpaque, Kind::Opaque),
    };
    match index {
        -1 => state(0),
        -2 => state(1),
        -3 => state(2),
        -4 => state(3),
        -5 => (geometry, geometry),
        i => state(OMM_STATES[i as usize]),
    }
}

/// Reference triangles of `c` as built, optionally with replacement vertices (templates).
fn ref_tris(c: &Cluster, verts: Option<&[[f32; 3]]>, id_offset: u32, geom_offset: u32) -> Vec<RefTri> {
    let verts = verts.unwrap_or(&c.verts);
    let tv = |i: u32| verts[i as usize].map(|x| truncate(x, c.truncate));
    c.tris
        .iter()
        .enumerate()
        .map(|(t, tri)| {
            let word = c.tri_gflags.as_ref().map_or(0, |g| g[t]);
            let flags = c.geom_flags | (word >> 29);
            let geometry = if flags & GF_OPAQUE != 0 { Kind::Opaque } else { Kind::NonOpaque };
            let (kind, kind_2state) = match &c.omm_index {
                Some(omm) => omm_kind(omm[t], geometry),
                None => (geometry, geometry),
            };
            RefTri {
                v: [tv(tri[0]), tv(tri[1]), tv(tri[2])],
                cluster: c.id + id_offset,
                prim: t as u32,
                geom: c.base_geom + (word & 0xff_ffff) + geom_offset,
                kind,
                kind_2state,
                no_dup: flags & GF_NO_DUP != 0,
            }
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Cand {
    /// No non-opaque triangle on the ray: the loop must see no candidate.
    Zero,
    /// The nearest candidate must be this (cluster, prim); `exact` candidates when given.
    Nearest(u32, u32, Option<u32>),
    /// Candidates may or may not arrive (non-opaque triangles behind an opaque one).
    Any,
}

struct Expect {
    /// (instance, triangle, t)
    hit: Option<(usize, usize, f32)>,
    cand: Cand,
}

fn cross(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn seg_dist(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
    let t = (((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
    let (dx, dy) = (p[0] - a[0] - t * ex, p[1] - a[1] - t * ey);
    (dx * dx + dy * dy).sqrt()
}

/// CPU reference for ray (x, y); None when the ray passes within EDGE_EPS of any edge.
fn expect(insts: &[RefInst], x: u32, y: u32, mode: u32) -> Option<Expect> {
    let p = [(x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / H as f32];
    let mut hits = Vec::new();
    for (ii, inst) in insts.iter().enumerate() {
        let q = [p[0] - inst.offset[0], p[1] - inst.offset[1]];
        for (ti, tri) in inst.tris.iter().enumerate() {
            let v = tri.v.map(|v| [v[0], v[1]]);
            let (minx, maxx) = (v[0][0].min(v[1][0]).min(v[2][0]), v[0][0].max(v[1][0]).max(v[2][0]));
            let (miny, maxy) = (v[0][1].min(v[1][1]).min(v[2][1]), v[0][1].max(v[1][1]).max(v[2][1]));
            if q[0] < minx - EDGE_EPS || q[0] > maxx + EDGE_EPS || q[1] < miny - EDGE_EPS || q[1] > maxy + EDGE_EPS {
                continue;
            }
            if seg_dist(v[0], v[1], q) < EDGE_EPS || seg_dist(v[1], v[2], q) < EDGE_EPS || seg_dist(v[2], v[0], q) < EDGE_EPS {
                return None;
            }
            let area = cross(v[0], v[1], v[2]);
            let b = [cross(v[1], v[2], q) / area, cross(v[2], v[0], q) / area, cross(v[0], v[1], q) / area];
            if b.iter().all(|&w| w > 0.0) {
                let t = 1.0 + inst.offset[2] + b[0] * tri.v[0][2] + b[1] * tri.v[1][2] + b[2] * tri.v[2][2];
                let mut kind = if mode & MODE_OMM_2_STATE != 0 { tri.kind_2state } else { tri.kind };
                if mode & MODE_FORCE_NON_OPAQUE != 0 && kind == Kind::Opaque {
                    kind = Kind::NonOpaque;
                }
                if kind != Kind::Transparent {
                    hits.push((t, ii, ti, kind));
                }
            }
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let tri = |h: &(f32, usize, usize, Kind)| &insts[h.1].tris[h.2];
    let committed = hits
        .iter()
        .find(|h| h.3 == Kind::Opaque || mode & MODE_ALPHA == 0 || alpha_keep(tri(h).cluster, tri(h).prim))
        .map(|h| (h.1, h.2, h.0));
    let opaque_t = hits.iter().find(|h| h.3 == Kind::Opaque).map_or(f32::INFINITY, |h| h.0);
    let before: Vec<_> = hits.iter().filter(|h| h.3 == Kind::NonOpaque && h.0 < opaque_t).collect();
    let any_non_opaque = hits.iter().any(|h| h.3 == Kind::NonOpaque);
    let cand = match before.first() {
        Some(h) => {
            let exact = (before.len() == hits.iter().filter(|h| h.3 == Kind::NonOpaque).count()
                && before.iter().all(|h| tri(h).no_dup))
                .then_some(before.len() as u32);
            Cand::Nearest(tri(h).cluster, tri(h).prim, exact)
        }
        None if !any_non_opaque => Cand::Zero,
        None => Cand::Any,
    };
    Some(Expect { hit: committed, cand })
}

#[derive(Default)]
struct Report {
    checked: u32,
    skipped: u32,
    hits: u32,
    candidates: u32,
    clusters: BTreeSet<u32>,
    bad: BTreeMap<&'static str, u32>,
    first_bad: Vec<String>,
}

impl Report {
    fn ok(&self) -> bool {
        self.bad.is_empty() && self.hits > 0
    }

    fn summary(&self) -> String {
        let mut s = format!(
            "checked={} skipped={} hits={} rays_with_candidates={} clusters_seen={:?}",
            self.checked, self.skipped, self.hits, self.candidates, self.clusters
        );
        if !self.bad.is_empty() {
            s += &format!(" mismatches={:?} first=[{}]", self.bad, self.first_bad.join("; "));
        }
        s
    }
}

/// Compares every ray record with the CPU reference. `cluster_fields` is false for the shader
/// without the cluster capability (no ClusterID, no position fetch).
unsafe fn check(out: &Buf, insts: &[RefInst], mode: u32, cluster_fields: bool) -> Report {
    let recs = std::slice::from_raw_parts(out.ptr.cast::<u32>(), (W * H) as usize * REC);
    let mut r = Report::default();
    for y in 0..H {
        for x in 0..W {
            let o = &recs[(y * W + x) as usize * REC..][..REC];
            let Some(e) = expect(insts, x, y, mode) else {
                r.skipped += 1;
                continue;
            };
            r.checked += 1;
            let mut bad = |what: &'static str, detail: String| {
                *r.bad.entry(what).or_insert(0) += 1;
                if r.first_bad.len() < 4 {
                    r.first_bad.push(format!("({x},{y}) {what}: {detail}"));
                }
            };
            if o[19] != 0x600d {
                bad("unwritten", format!("{:#x}", o[19]));
                continue;
            }
            if o[7] > 0 {
                r.candidates += 1;
            }
            match e.cand {
                Cand::Zero if o[7] != 0 => bad("candidate_count", format!("got {} want 0", o[7])),
                Cand::Nearest(c, p, exact) => {
                    if o[7] == 0 {
                        bad("candidate_missing", format!("want cluster {c} prim {p}"));
                    } else if (cluster_fields && o[8] != c) || o[9] != p {
                        bad("candidate_id", format!("got cluster {} prim {} want {c} {p}", o[8], o[9]));
                    } else if let Some(n) = exact
                        && o[7] != n
                    {
                        bad("candidate_dup", format!("got {} want {n} (NO_DUPLICATE_ANYHIT)", o[7]));
                    }
                }
                _ => {}
            }
            let Some((ii, ti, t)) = e.hit else {
                if o[0] != 0 {
                    bad("hit_on_miss", format!("instance {} cluster {} prim {}", o[1], o[3], o[4]));
                }
                continue;
            };
            if o[0] != 1 {
                bad("miss_on_hit", format!("want instance {} cluster {} prim {}", insts[ii].custom, insts[ii].tris[ti].cluster, ti));
                continue;
            }
            r.hits += 1;
            let inst = &insts[ii];
            let tri = &inst.tris[ti];
            r.clusters.insert(o[3]);
            if o[1] != inst.custom || o[2] != inst.index {
                bad("instance", format!("got custom {} index {} want {} {}", o[1], o[2], inst.custom, inst.index));
            }
            if cluster_fields && o[3] != tri.cluster {
                bad("cluster_id", format!("got {} want {}", o[3], tri.cluster));
            }
            if o[4] != tri.prim {
                bad("primitive", format!("got {} want {} (cluster {})", o[4], tri.prim, tri.cluster));
            }
            if o[5] != tri.geom {
                bad("geometry_index", format!("got {} want {} (cluster {})", o[5], tri.geom, tri.cluster));
            }
            let got_t = f32::from_bits(o[6]);
            if (got_t - t).abs() > T_EPS {
                bad("t", format!("got {got_t} want {t}"));
            }
            if cluster_fields {
                let got: Vec<f32> = o[10..19].iter().map(|&b| f32::from_bits(b)).collect();
                let want: Vec<f32> = tri.v.iter().flatten().copied().collect();
                if got != want {
                    let same_set = (0..3).any(|r0| (0..3).all(|k| got[k * 3..k * 3 + 3] == want[((k + r0) % 3) * 3..((k + r0) % 3) * 3 + 3]));
                    let what = if same_set { "position_fetch_rotated" } else { "position_fetch" };
                    bad(what, format!("got {got:?} want {want:?}"));
                }
            }
        }
    }
    r
}

// ---------------------------------------------------------------------------------------------
// Builders: all recorded into one command buffer, arguments written on the GPU.

struct Ctx {
    gpu: Gpu,
    args: vk::Pipeline,
    keep: Mutex<Vec<Buf>>,
}

/// Geometry uploaded for a cluster batch.
struct Uploaded {
    descs: Buf,
    n: u32,
}

/// Output of one indirect cluster operation.
struct ClusterOut {
    /// u64 per op (written by the build).
    addrs: Buf,
    /// u32 per op (written by the build).
    sizes: Buf,
    implicit_size: u64,
    scratch_size: u64,
}

impl Ctx {
    unsafe fn new(case: &str) -> Ctx {
        let _ = case;
        let gpu = Gpu::new();
        let args = gpu.pipeline(&spv!("args"));
        Ctx { gpu, args, keep: Mutex::new(Vec::new()) }
    }

    unsafe fn buf(&self, size: u64, kind: Mem, align: u64) -> Buf {
        let b = self.gpu.buffer(size, kind, align);
        self.keep.lock().unwrap().push(b);
        b
    }

    unsafe fn args(&self, cmd: vk::CommandBuffer, root: ArgsRoot) {
        self.gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.args);
        self.gpu.push(cmd, &root);
        self.gpu.device.cmd_dispatch(cmd, root.n.div_ceil(64).max(1), 1, 1);
        self.gpu.full_barrier(cmd, false);
    }

    /// Uploads vertex/index/flag/OMM-index data and the ClusterDesc array.
    unsafe fn upload_clusters(&self, clusters: &[Cluster], omm_array: u64, bbox: &[u64]) -> Uploaded {
        let mut descs = Vec::new();
        for (ci, c) in clusters.iter().enumerate() {
            let verts = self.gpu.upload(&c.verts);
            let index_bytes: Vec<u8> = c
                .tris
                .iter()
                .flatten()
                .flat_map(|&i| match c.index_bytes {
                    1 => vec![u8::try_from(i).unwrap()],
                    2 => u16::try_from(i).unwrap().to_le_bytes().to_vec(),
                    _ => i.to_le_bytes().to_vec(),
                })
                .collect();
            let indices = self.gpu.upload(&index_bytes);
            let gflags = c.tri_gflags.as_ref().map(|g| self.gpu.upload(g));
            let omm_index = c.omm_index.as_ref().map(|g| self.gpu.upload(g));
            descs.push(ClusterDesc {
                index_addr: indices.addr,
                vertex_addr: verts.addr,
                gflags_addr: gflags.map_or(0, |b| b.addr),
                omm_array: if omm_index.is_some() { omm_array } else { 0 },
                omm_index: omm_index.map_or(0, |b| b.addr),
                bbox_addr: bbox.get(ci).copied().unwrap_or(0),
                id: c.id,
                tris: c.tris.len() as u32,
                verts: c.verts.len() as u32,
                index_type: c.index_bytes,
                truncate: c.truncate,
                base_geom: c.base_geom,
                geom_flags: c.geom_flags,
                cluster_flags: 0,
                index_stride: c.index_bytes,
                vertex_stride: 12,
                gflags_stride: if gflags.is_some() { 4 } else { 0 },
                omm_index_type: if omm_index.is_some() { 4 } else { 1 },
                omm_index_stride: if omm_index.is_some() { 4 } else { 0 },
                pad: [0; 3],
            });
        }
        Uploaded { descs: self.gpu.upload(&descs), n: descs.len() as u32 }
    }

    fn triangle_input(&self, clusters: &[Cluster]) -> vk::ClusterAccelerationStructureTriangleClusterInputNV<'static> {
        let geoms: u32 = clusters
            .iter()
            .map(|c| c.tri_gflags.as_ref().map_or(1, |g| g.iter().map(|w| w & 0xff_ffff).collect::<BTreeSet<_>>().len() as u32))
            .max()
            .unwrap_or(1);
        vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(64)
            .max_cluster_unique_geometry_count(geoms)
            .max_cluster_triangle_count(clusters.iter().map(|c| c.tris.len() as u32).max().unwrap())
            .max_cluster_vertex_count(clusters.iter().map(|c| c.verts.len() as u32).max().unwrap())
            .max_total_triangle_count(clusters.iter().map(|c| c.tris.len() as u32).sum())
            .max_total_vertex_count(clusters.iter().map(|c| c.verts.len() as u32).sum())
            .min_position_truncate_bit_count(clusters.iter().map(|c| c.truncate).min().unwrap())
    }

    /// One indirect cluster op over `n` inputs in implicit-destination mode. Arguments are
    /// written by the args kernel (`args_op`, from `src`); the count is GPU-written too.
    /// `addrs_dst` optionally places the output address array inside a larger table.
    unsafe fn cluster_op(
        &self,
        cmd: vk::CommandBuffer,
        op_type: vk::ClusterAccelerationStructureOpTypeNV,
        args_op: u32,
        src: u64,
        aux: u64,
        n: u32,
        input: &mut vk::ClusterAccelerationStructureOpInputNV<'_>,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        implicit: Option<u64>,
        addrs_dst: Option<u64>,
    ) -> ClusterOut {
        let gpu = &self.gpu;
        let p = &gpu.cl_props;
        let stride: u64 = match args_op {
            OP_CLUSTER => 64,
            OP_TEMPLATE => 72,
            OP_INSTANTIATE => 32,
            _ => 16,
        };
        let infos = self.buf(stride * u64::from(n), Mem::Device, 16);
        let count = self.buf(4, Mem::Device, 16);
        self.args(cmd, ArgsRoot { src, dst: infos.addr, count_dst: count.addr, aux, n, op: args_op, pad: [0; 2] });

        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(n)
            .flags(flags)
            .op_type(op_type)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(*input);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.cluster.get_cluster_acceleration_structure_build_sizes(&info, &mut sizes);
        let align = match op_type {
            vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER_TEMPLATE => p.cluster_template_byte_alignment,
            vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL => p.cluster_bottom_level_byte_alignment,
            _ => p.cluster_byte_alignment,
        };
        let implicit = implicit.unwrap_or_else(|| self.buf(sizes.acceleration_structure_size, Mem::Device, u64::from(align)).addr);
        let scratch = self.buf(sizes.build_scratch_size, Mem::Device, u64::from(p.cluster_scratch_byte_alignment));
        let addrs = self.buf(8 * u64::from(n), Mem::Readback, 16);
        let sizes_out = self.buf(4 * u64::from(n), Mem::Readback, 16);
        let region = |addr: u64, stride: u64| vk::StridedDeviceAddressRegionKHR { device_address: addr, stride, size: stride * u64::from(n) };
        let commands = vk::ClusterAccelerationStructureCommandsInfoNV::default()
            .input(info)
            .dst_implicit_data(implicit)
            .scratch_data(scratch.addr)
            .dst_addresses_array(region(addrs_dst.unwrap_or(addrs.addr), 8))
            .dst_sizes_array(region(sizes_out.addr, 4))
            .src_infos_array(region(infos.addr, stride))
            .src_infos_count(count.addr);
        gpu.cluster.cmd_build_cluster_acceleration_structure_indirect(cmd, &commands);
        gpu.full_barrier(cmd, false);
        ClusterOut {
            addrs: if let Some(a) = addrs_dst { Buf { addr: a, ..addrs } } else { addrs },
            sizes: sizes_out,
            implicit_size: sizes.acceleration_structure_size,
            scratch_size: sizes.build_scratch_size,
        }
    }

    /// CLAS build from uploaded clusters.
    unsafe fn build_clas(
        &self,
        cmd: vk::CommandBuffer,
        up: &Uploaded,
        clusters: &[Cluster],
        flags: vk::BuildAccelerationStructureFlagsKHR,
        implicit: Option<u64>,
        addrs_dst: Option<u64>,
    ) -> ClusterOut {
        let mut tri_input = self.triangle_input(clusters);
        let mut input = vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut tri_input };
        self.cluster_op(cmd, vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER, OP_CLUSTER, up.descs.addr, 0, up.n, &mut input, flags, implicit, addrs_dst)
    }

    /// Cluster BLAS per group of indices into the CLAS address table at `clas_addrs`.
    unsafe fn build_blas(&self, cmd: vk::CommandBuffer, clas_addrs: u64, groups: &[Vec<u32>], implicit: Option<u64>) -> ClusterOut {
        let flat: Vec<u32> = groups.iter().flatten().copied().collect();
        let mut first = 0;
        let descs: Vec<BlasDesc> = groups
            .iter()
            .map(|g| {
                let d = BlasDesc { count: g.len() as u32, first, pad: 0 };
                first += g.len() as u32;
                d
            })
            .collect();
        let indices = self.gpu.upload(&flat);
        let refs = self.buf(8 * flat.len() as u64, Mem::Device, 16);
        self.args(cmd, ArgsRoot { src: indices.addr, dst: refs.addr, count_dst: 0, aux: clas_addrs, n: flat.len() as u32, op: OP_GATHER, pad: [0; 2] });
        let src = self.gpu.upload(&descs);
        let mut bl = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default()
            .max_total_cluster_count(flat.len() as u32)
            .max_cluster_count_per_acceleration_structure(groups.iter().map(|g| g.len() as u32).max().unwrap());
        let mut input = vk::ClusterAccelerationStructureOpInputNV { p_clusters_bottom_level: &mut bl };
        self.cluster_op(
            cmd,
            vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL,
            OP_BLAS,
            src.addr,
            refs.addr,
            groups.len() as u32,
            &mut input,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            implicit,
            None,
        )
    }

    /// Classic TLAS over cluster BLAS; instance records written by the args kernel from the
    /// BLAS build's address output. Returns the TLAS device address.
    unsafe fn build_tlas(&self, cmd: vk::CommandBuffer, blas_addrs: u64, insts: &[InstanceDesc]) -> u64 {
        let gpu = &self.gpu;
        let n = insts.len() as u32;
        let src = gpu.upload(insts);
        let instances = self.buf(64 * u64::from(n), Mem::Device, 16);
        self.args(cmd, ArgsRoot { src: src.addr, dst: instances.addr, count_dst: 0, aux: blas_addrs, n, op: OP_INSTANCE, pad: [0; 2] });
        let geometry = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .data(vk::DeviceOrHostAddressConstKHR { device_address: instances.addr }),
            })];
        let mut build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.accel.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &build, Some(&[n]), &mut sizes);
        let backing = self.buf(sizes.acceleration_structure_size, Mem::Device, 256);
        let base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(backing.buffer));
        let tlas = gpu
            .accel
            .create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(backing.buffer)
                    .offset(backing.addr - base)
                    .size(sizes.acceleration_structure_size)
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                None,
            )
            .unwrap();
        let scratch = self.buf(sizes.build_scratch_size, Mem::Device, u64::from(gpu.as_props.min_acceleration_structure_scratch_offset_alignment));
        build = build.dst_acceleration_structure(tlas).scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch.addr });
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(n)];
        gpu.accel.cmd_build_acceleration_structures(cmd, &[build], &[Some(&ranges)]);
        gpu.full_barrier(cmd, false);
        gpu.accel.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(tlas))
    }

    /// Ordinary (non-cluster) triangle BLAS from one cluster's geometry; returns its address.
    unsafe fn build_triangle_blas(&self, cmd: vk::CommandBuffer, c: &Cluster) -> u64 {
        let gpu = &self.gpu;
        let verts = gpu.upload(&c.verts);
        let indices: Vec<u32> = c.tris.iter().flatten().copied().collect();
        let indices = gpu.upload(&indices);
        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: verts.addr })
            .vertex_stride(12)
            .max_vertex(c.verts.len() as u32 - 1)
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR { device_address: indices.addr });
        let geometry = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
            .flags(vk::GeometryFlagsKHR::OPAQUE)];
        let n = c.tris.len() as u32;
        let mut build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(data_flags())
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.accel.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &build, Some(&[n]), &mut sizes);
        let backing = self.buf(sizes.acceleration_structure_size, Mem::Device, 256);
        let base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(backing.buffer));
        let blas = gpu
            .accel
            .create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(backing.buffer)
                    .offset(backing.addr - base)
                    .size(sizes.acceleration_structure_size)
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                None,
            )
            .unwrap();
        let scratch = self.buf(sizes.build_scratch_size, Mem::Device, u64::from(gpu.as_props.min_acceleration_structure_scratch_offset_alignment));
        build = build.dst_acceleration_structure(blas).scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch.addr });
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(n)];
        gpu.accel.cmd_build_acceleration_structures(cmd, &[build], &[Some(&ranges)]);
        gpu.full_barrier(cmd, false);
        gpu.accel.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blas))
    }

    unsafe fn out_buffer(&self) -> Buf {
        let b = self.buf((W * H) as u64 * REC as u64 * 4, Mem::Readback, 16);
        std::ptr::write_bytes(b.ptr, 0xee, b.size as usize);
        b
    }

    unsafe fn trace(&self, cmd: vk::CommandBuffer, pipeline: vk::Pipeline, out: &Buf, tlas: u64, mode: u32) {
        self.gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        self.gpu.push(cmd, &TraceRoot { out: out.addr, tlas, width: W, height: H, mode, pad: 0 });
        self.gpu.device.cmd_dispatch(cmd, W / 8, H / 8, 1);
        self.gpu.full_barrier(cmd, false);
    }

    /// Warm GPU time of one W x H trace dispatch: one warm-up, then TIMED dispatches into a
    /// device-local buffer, averaged. Separate from the checked dispatch, which writes 80 bytes
    /// per ray to host memory.
    unsafe fn time_trace(&self, case: &str, pipeline: vk::Pipeline, tlas: u64, mode: u32) -> f64 {
        const TIMED: u32 = 16;
        let gpu = &self.gpu;
        let out = self.buf((W * H) as u64 * REC as u64 * 4, Mem::Device, 16);
        let cmd = gpu.begin();
        gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 30, 2);
        self.trace(cmd, pipeline, &out, tlas, mode);
        gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, gpu.timestamps, 30);
        for _ in 0..TIMED {
            self.trace(cmd, pipeline, &out, tlas, mode);
        }
        gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, gpu.timestamps, 31);
        gpu.submit_wait(cmd, case, "timing");
        let mut ticks = [0u64; 2];
        gpu.device
            .get_query_pool_results(gpu.timestamps, 30, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
            .unwrap();
        (ticks[1] - ticks[0]) as f64 * f64::from(gpu.limits.timestamp_period) / 1e3 / f64::from(TIMED)
    }

    fn clamp_grid(&self, mut nx: u32, ny: u32) -> (u32, u32) {
        let p = &self.gpu.cl_props;
        while nx > 1 && (2 * nx * ny > p.max_triangles_per_cluster || (nx + 1) * (ny + 1) > p.max_vertices_per_cluster) {
            nx -= 1;
        }
        (nx, ny)
    }
}

// ---------------------------------------------------------------------------------------------
// Scenes.

const FRONT_CUSTOM: u32 = 41;
const BACK_CUSTOM: u32 = 77;
const FRONT_Z: f32 = 0.05;

fn quadrant(q: u32) -> [f32; 4] {
    let (qx, qy) = ((q % 2) as f32 * 0.5, (q / 2) as f32 * 0.5);
    [qx + 0.03, qy + 0.03, qx + 0.47, qy + 0.47]
}

/// Four opaque grid clusters, one per quadrant (8, 24, 72 and up to 256 triangles; 8-, 16-,
/// 32- and 8-bit indices; cluster 2 truncates 8 mantissa bits), plus one back quad cluster.
fn base_clusters(ctx: &Ctx) -> Vec<Cluster> {
    let grids = [(2, 2, 1), (4, 3, 2), (6, 6, 4), (16, 8, 1)];
    let mut clusters: Vec<Cluster> = grids
        .iter()
        .enumerate()
        .map(|(c, &(nx, ny, ib))| {
            let (nx, ny) = ctx.clamp_grid(nx, ny);
            let cf = c as f32;
            let mut cl = grid(1000 + 17 * c as u32, nx, ny, quadrant(c as u32), move |x, y| 0.2 + 0.1 * x + 0.05 * y + 0.01 * cf + 0.02 * (9.0 * x).sin() * (7.0 * y).cos(), ib);
            cl.base_geom = c as u32;
            cl
        })
        .collect();
    clusters[2].truncate = 8;
    let mut back = grid(5000, 1, 1, [0.0, 0.0, 1.0, 1.0], |_, _| 0.8, 1);
    back.base_geom = 9;
    clusters.push(back);
    clusters
}

fn inst_desc(t: [f32; 3], custom: u32, blas_index: u32) -> InstanceDesc {
    InstanceDesc {
        m: translate(t),
        custom,
        mask: 0xff,
        flags: vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw(),
        blas_index,
    }
}

fn ref_inst(custom: u32, index: u32, offset: [f32; 3], clusters: &[&Cluster]) -> RefInst {
    RefInst { custom, index, offset, tris: clusters.iter().flat_map(|c| ref_tris(c, None, 0, 0)).collect() }
}

fn data_flags() -> vk::BuildAccelerationStructureFlagsKHR {
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS
}

unsafe fn sizes_detail(gpu: &Gpu, clas: &ClusterOut, blas: &ClusterOut, n_clas: usize, n_blas: usize, clusters: &[Cluster]) -> String {
    let _ = gpu;
    let cs: Vec<u32> = clas.sizes.read(n_clas);
    let bs: Vec<u32> = blas.sizes.read(n_blas);
    let per_tri: Vec<String> = cs.iter().zip(clusters).map(|(s, c)| format!("{:.0}", f64::from(*s) / c.tris.len() as f64)).collect();
    format!(
        "clas_bytes={cs:?} clas_bytes_per_tri={per_tri:?} clas_implicit_reserve={} blas_bytes={bs:?} blas_implicit_reserve={} scratch={}/{}",
        clas.implicit_size, blas.implicit_size, clas.scratch_size, blas.scratch_size
    )
}

fn us(t: &[f64]) -> String {
    t.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join("/")
}

/// Case 1 and its shader variants: CLAS -> cluster BLAS -> classic TLAS -> ray query.
unsafe fn case_clas_tlas(ctx: &Ctx, case: &str, pipeline: vk::Pipeline, cluster_fields: bool, sparse: Option<(u64, u64)>) -> (bool, String) {
    let gpu = &ctx.gpu;
    let clusters = base_clusters(ctx);
    let up = ctx.upload_clusters(&clusters, 0, &[]);
    let cmd = gpu.begin();
    gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 0, 32);
    gpu.timestamp(cmd, 0);
    let clas = ctx.build_clas(cmd, &up, &clusters, data_flags(), sparse.map(|s| s.0), None);
    gpu.timestamp(cmd, 1);
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0, 1, 2, 3], vec![4]], sparse.map(|s| s.1));
    gpu.timestamp(cmd, 2);
    let tlas = ctx.build_tlas(cmd, blas.addrs.addr, &[inst_desc([0.0, 0.0, FRONT_Z], FRONT_CUSTOM, 0), inst_desc([0.0; 3], BACK_CUSTOM, 1)]);
    gpu.timestamp(cmd, 3);
    let out = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out, tlas, 0);
    gpu.timestamp(cmd, 4);
    let out_fno = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out_fno, tlas, MODE_FORCE_NON_OPAQUE);
    gpu.timestamp(cmd, 5);
    gpu.submit_wait(cmd, case, "build+trace");
    let t = gpu.read_timestamps(6);
    let insts = [
        ref_inst(FRONT_CUSTOM, 0, [0.0, 0.0, FRONT_Z], &clusters[..4].iter().collect::<Vec<_>>()),
        ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[&clusters[4]]),
    ];
    let r = check(&out, &insts, 0, cluster_fields);
    let r_fno = check(&out_fno, &insts, MODE_FORCE_NON_OPAQUE, cluster_fields);
    let warm = [ctx.time_trace(case, pipeline, tlas, 0), ctx.time_trace(case, pipeline, tlas, MODE_FORCE_NON_OPAQUE)];
    let detail = format!(
        "opaque[{}] force_non_opaque[{}] {} gpu_us(clas/blas/tlas/cold_trace/cold_trace_fno)={} warm_trace_us_per_262k_rays(opaque/fno)={}",
        r.summary(),
        r_fno.summary(),
        sizes_detail(gpu, &clas, &blas, clusters.len(), 2, &clusters),
        us(&t),
        us(&warm)
    );
    (r.ok() && r_fno.ok() && (!cluster_fields || r.clusters.len() == 5), detail)
}

/// Cluster BLAS and an ordinary triangle BLAS under one classic TLAS: hits on the ordinary
/// BLAS must report ClusterID -1 (gl_ClusterIDNoneNV).
unsafe fn case_mixed(ctx: &Ctx, case: &str, pipeline: vk::Pipeline) -> (bool, String) {
    let gpu = &ctx.gpu;
    let clusters = base_clusters(ctx);
    let front = &clusters[..4];
    let up = ctx.upload_clusters(front, 0, &[]);
    let cmd = gpu.begin();
    let clas = ctx.build_clas(cmd, &up, front, data_flags(), None, None);
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0, 1, 2, 3]], None);
    let tri_blas = ctx.build_triangle_blas(cmd, &clusters[4]);
    // Second entry of the BLAS address table: the ordinary BLAS.
    let base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(blas.addrs.buffer));
    gpu.device.cmd_update_buffer(cmd, blas.addrs.buffer, blas.addrs.addr - base + 8, bytes(&[tri_blas]));
    gpu.full_barrier(cmd, false);
    let tlas = ctx.build_tlas(cmd, blas.addrs.addr, &[inst_desc([0.0, 0.0, FRONT_Z], FRONT_CUSTOM, 0), inst_desc([0.0; 3], BACK_CUSTOM, 1)]);
    let out = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out, tlas, 0);
    let out_fno = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out_fno, tlas, MODE_FORCE_NON_OPAQUE);
    gpu.submit_wait(cmd, case, "build+trace");
    let mut back = ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[&clusters[4]]);
    for t in &mut back.tris {
        t.cluster = u32::MAX;
        t.geom = 0;
    }
    let insts = [ref_inst(FRONT_CUSTOM, 0, [0.0, 0.0, FRONT_Z], &front.iter().collect::<Vec<_>>()), back];
    let r = check(&out, &insts, 0, true);
    let r_fno = check(&out_fno, &insts, MODE_FORCE_NON_OPAQUE, true);
    let detail = format!("opaque[{}] force_non_opaque[{}]", r.summary(), r_fno.summary());
    (r.ok() && r_fno.ok() && r.clusters.contains(&u32::MAX), detail)
}

/// Case 2: the same CLAS/BLAS under a partitioned TLAS; write, update, translate.
unsafe fn case_ptlas(ctx: &Ctx, case: &str, pipeline: vk::Pipeline) -> (bool, String) {
    let gpu = &ctx.gpu;
    let Some(pt) = &gpu.ptlas else { finish(case, "UNSUPPORTED", "VK_NV_partitioned_acceleration_structure absent") };
    let clusters = base_clusters(ctx);
    let up = ctx.upload_clusters(&clusters, 0, &[]);
    let cmd = gpu.begin();
    let clas = ctx.build_clas(cmd, &up, &clusters, data_flags(), None, None);
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0, 1, 2, 3], vec![4], vec![0, 2]], None);
    gpu.submit_wait(cmd, case, "clas+blas");
    let blas_addr: Vec<u64> = blas.addrs.read(3);

    let mut flags_nv = vk::PartitionedAccelerationStructureFlagsNV::default().enable_partition_translation(true);
    let input = vk::PartitionedAccelerationStructureInstancesInputNV::default()
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .instance_count(2)
        .max_instance_per_partition_count(2)
        .partition_count(2)
        .max_instance_in_global_partition_count(1)
        .push(&mut flags_nv);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    pt.get_partitioned_acceleration_structures_build_sizes(&input, &mut sizes);
    let dst = ctx.buf(sizes.acceleration_structure_size, Mem::Device, 256);
    let scratch = ctx.buf(sizes.build_scratch_size, Mem::Device, 256);

    let write = |index: u32, partition: u32, custom: u32, t: [f32; 3], blas: u64, aabb: bool| {
        let mut flags = vk::PartitionedAccelerationStructureInstanceFlagsNV::FLAG_TRIANGLE_FACING_CULL_DISABLE;
        if aabb {
            flags |= vk::PartitionedAccelerationStructureInstanceFlagsNV::FLAG_ENABLE_EXPLICIT_BOUNDING_BOX;
        }
        let m = translate(t);
        vk::PartitionedAccelerationStructureWriteInstanceDataNV {
            transform: vk::TransformMatrixKHR { matrix: m },
            explicit_aabb: [-0.1, -0.1, 0.0, 1.1, 1.1, 1.0],
            instance_id: custom,
            instance_mask: 0xff,
            instance_contribution_to_hit_group_index: 0,
            instance_flags: flags,
            instance_index: index,
            partition_index: partition,
            acceleration_structure: blas,
        }
    };
    // One build: per-op argument arrays and the command array, host-written.
    let build = |ops: Vec<(vk::PartitionedAccelerationStructureOpTypeNV, Buf, u32, u64)>, first: bool, label: &str| {
        let commands: Vec<vk::BuildPartitionedAccelerationStructureIndirectCommandNV> = ops
            .iter()
            .map(|(op, buf, count, stride)| vk::BuildPartitionedAccelerationStructureIndirectCommandNV {
                op_type: *op,
                arg_count: *count,
                arg_data: vk::StridedDeviceAddressNV { start_address: buf.addr, stride_in_bytes: *stride },
            })
            .collect();
        let src_infos = gpu.upload(&commands);
        let count = gpu.upload(&[commands.len() as u32]);
        let mut flags_nv = vk::PartitionedAccelerationStructureFlagsNV::default().enable_partition_translation(true);
        let input = vk::PartitionedAccelerationStructureInstancesInputNV::default()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .instance_count(2)
            .max_instance_per_partition_count(2)
            .partition_count(2)
            .max_instance_in_global_partition_count(1)
            .push(&mut flags_nv);
        let info = vk::BuildPartitionedAccelerationStructureInfoNV {
            input,
            src_acceleration_structure_data: if first { 0 } else { dst.addr },
            dst_acceleration_structure_data: dst.addr,
            scratch_data: scratch.addr,
            src_infos: src_infos.addr,
            src_infos_count: count.addr,
            ..Default::default()
        };
        let cmd = gpu.begin();
        gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 0, 32);
        gpu.timestamp(cmd, 0);
        pt.cmd_build_partitioned_acceleration_structures(cmd, &info);
        gpu.full_barrier(cmd, false);
        gpu.timestamp(cmd, 1);
        let out = ctx.out_buffer();
        ctx.trace(cmd, pipeline, &out, dst.addr, 0);
        gpu.timestamp(cmd, 2);
        gpu.submit_wait(cmd, case, label);
        (out, gpu.read_timestamps(3))
    };
    let wsize = size_of::<vk::PartitionedAccelerationStructureWriteInstanceDataNV>() as u64;
    let usize_ = size_of::<vk::PartitionedAccelerationStructureUpdateInstanceDataNV>() as u64;
    let tsize = size_of::<vk::PartitionedAccelerationStructureWritePartitionTranslationDataNV>() as u64;
    let op_w = vk::PartitionedAccelerationStructureOpTypeNV::WRITE_INSTANCE;
    let op_u = vk::PartitionedAccelerationStructureOpTypeNV::UPDATE_INSTANCE;
    let op_t = vk::PartitionedAccelerationStructureOpTypeNV::WRITE_PARTITION_TRANSLATION;

    // Build 1: both instances written, front in partition 0 (explicit AABB), back in 1.
    let w1 = gpu.upload(&[
        write(0, 0, FRONT_CUSTOM, [0.0, 0.0, FRONT_Z], blas_addr[0], true),
        write(1, 1, BACK_CUSTOM, [0.0; 3], blas_addr[1], false),
    ]);
    let (out_a, t_a) = build(vec![(op_w, w1, 2, wsize)], true, "ptlas build 1");
    let front: Vec<&Cluster> = clusters[..4].iter().collect();
    let refs_a = [ref_inst(FRONT_CUSTOM, 0, [0.0, 0.0, FRONT_Z], &front), ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[&clusters[4]])];
    let r_a = check(&out_a, &refs_a, 0, true);

    // Build 2 (in place): front swaps to BLAS {0, 2} by UPDATE_INSTANCE, partition 0 translated,
    // back rewritten into the global partition one eighth up.
    let u2 = gpu.upload(&[vk::PartitionedAccelerationStructureUpdateInstanceDataNV {
        instance_index: 0,
        instance_contribution_to_hit_group_index: 0,
        acceleration_structure: blas_addr[2],
    }]);
    let w2 = gpu.upload(&[write(1, vk::PARTITIONED_ACCELERATION_STRUCTURE_PARTITION_INDEX_GLOBAL_NV, BACK_CUSTOM, [0.0, 0.125, 0.0], blas_addr[1], false)]);
    let t2 = gpu.upload(&[vk::PartitionedAccelerationStructureWritePartitionTranslationDataNV { partition_index: 0, partition_translation: [0.125, -0.0625, 0.0] }]);
    let (out_b, t_b) = build(vec![(op_u, u2, 1, usize_), (op_w, w2, 1, wsize), (op_t, t2, 1, tsize)], false, "ptlas build 2");
    let subset = [&clusters[0], &clusters[2]];
    let refs_b = [ref_inst(FRONT_CUSTOM, 0, [0.125, -0.0625, FRONT_Z], &subset), ref_inst(BACK_CUSTOM, 1, [0.0, 0.125, 0.0], &[&clusters[4]])];
    let r_b = check(&out_b, &refs_b, 0, true);

    // Build 3: a new translation for partition 0 replaces the previous one.
    let t3 = gpu.upload(&[vk::PartitionedAccelerationStructureWritePartitionTranslationDataNV { partition_index: 0, partition_translation: [0.0625, 0.0, 0.0] }]);
    let (out_c, t_c) = build(vec![(op_t, t3, 1, tsize)], false, "ptlas build 3");
    let refs_c = [ref_inst(FRONT_CUSTOM, 0, [0.0625, 0.0, FRONT_Z], &subset), ref_inst(BACK_CUSTOM, 1, [0.0, 0.125, 0.0], &[&clusters[4]])];
    let r_c = check(&out_c, &refs_c, 0, true);
    let warm = ctx.time_trace(case, pipeline, dst.addr, 0);
    let detail = format!(
        "write[{}] update+translate+global[{}] retranslate[{}] ptlas_bytes={} scratch={} gpu_us(build/cold_trace)={} {} {} warm_trace_us={warm:.1}",
        r_a.summary(),
        r_b.summary(),
        r_c.summary(),
        sizes.acceleration_structure_size,
        sizes.build_scratch_size,
        us(&t_a),
        us(&t_b),
        us(&t_c)
    );
    (r_a.ok() && r_b.ok() && r_c.ok(), detail)
}

/// Case 3: non-opaque CLAS geometry, per-triangle flags and geometry indices, manual alpha test.
unsafe fn case_alpha(ctx: &Ctx, case: &str, pipeline: vk::Pipeline) -> (bool, String) {
    let gpu = &ctx.gpu;
    let (nx, ny) = ctx.clamp_grid(8, 8);
    // Mixed cluster: every 4th triangle opaque by its per-triangle flags, geometry index
    // 20 + (t % 4) from base + per-triangle word.
    let mut mixed = grid(2001, nx, ny, [0.05, 0.05, 0.95, 0.48], |x, y| 0.3 + 0.05 * x - 0.03 * y, 1);
    mixed.geom_flags = GF_CULL_DISABLE;
    mixed.base_geom = 20;
    mixed.tri_gflags = Some((0..mixed.tris.len() as u32).map(|t| (t % 4) | if t % 4 == 0 { GF_OPAQUE << 29 } else { 0 }).collect());
    // All non-opaque, NO_DUPLICATE_ANYHIT: exactly one candidate per front triangle.
    let mut nodup = grid(2002, 6, 6, [0.05, 0.52, 0.95, 0.95], |x, y| 0.35 - 0.05 * x + 0.02 * y, 2);
    nodup.geom_flags = GF_CULL_DISABLE | GF_NO_DUP;
    nodup.base_geom = 30;
    let mut back = grid(5000, 1, 1, [0.0, 0.0, 1.0, 1.0], |_, _| 0.8, 1);
    back.base_geom = 9;
    let clusters = vec![mixed, nodup, back];
    let up = ctx.upload_clusters(&clusters, 0, &[]);
    let cmd = gpu.begin();
    gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 0, 32);
    gpu.timestamp(cmd, 0);
    let clas = ctx.build_clas(cmd, &up, &clusters, data_flags(), None, None);
    gpu.timestamp(cmd, 1);
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0, 1], vec![2]], None);
    gpu.timestamp(cmd, 2);
    // No FORCE_OPAQUE on the instance; cull disabled.
    let tlas = ctx.build_tlas(cmd, blas.addrs.addr, &[inst_desc([0.0; 3], FRONT_CUSTOM, 0), inst_desc([0.0; 3], BACK_CUSTOM, 1)]);
    gpu.timestamp(cmd, 3);
    let out = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out, tlas, MODE_ALPHA);
    gpu.timestamp(cmd, 4);
    let out_fno = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out_fno, tlas, MODE_ALPHA | MODE_FORCE_NON_OPAQUE);
    gpu.timestamp(cmd, 5);
    gpu.submit_wait(cmd, case, "build+trace");
    let t = gpu.read_timestamps(6);
    let insts = [ref_inst(FRONT_CUSTOM, 0, [0.0; 3], &[&clusters[0], &clusters[1]]), ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[&clusters[2]])];
    let r = check(&out, &insts, MODE_ALPHA, true);
    let r_fno = check(&out_fno, &insts, MODE_ALPHA | MODE_FORCE_NON_OPAQUE, true);
    let warm = [ctx.time_trace(case, pipeline, tlas, MODE_ALPHA), ctx.time_trace(case, pipeline, tlas, MODE_ALPHA | MODE_FORCE_NON_OPAQUE)];
    let detail = format!(
        "alpha[{}] alpha+force_non_opaque[{}] {} gpu_us(clas/blas/tlas/cold_trace/cold_trace_fno)={} warm_trace_us(alpha/alpha_fno)={}",
        r.summary(),
        r_fno.summary(),
        sizes_detail(gpu, &clas, &blas, 3, 2, &clusters),
        us(&t),
        us(&warm)
    );
    (r.ok() && r_fno.ok() && r.candidates > 0, detail)
}

/// Case 4: opacity micromap referenced from a CLAS (VK_NV_cluster_acceleration_structure rev 4).
unsafe fn case_omm(ctx: &Ctx, case: &str, pipeline: vk::Pipeline) -> (bool, String) {
    let gpu = &ctx.gpu;
    let rev = ext_rev(&gpu.exts, ash::nv::cluster_acceleration_structure::NAME).unwrap_or(0);
    let Some(omm) = &gpu.omm else { finish(case, "UNSUPPORTED", &format!("VK_EXT_opacity_micromap absent (cluster rev {rev})")) };
    if rev < 4 {
        finish(case, "UNSUPPORTED", &format!("VK_NV_cluster_acceleration_structure rev {rev} < 4 (no ALLOW_CLUSTER_OPACITY_MICROMAPS)"));
    }
    // Four level-0 4-state micromap entries: transparent, opaque, unknown-transparent,
    // unknown-opaque.
    let data = gpu.upload(&OMM_STATES);
    let tri_entries: Vec<vk::MicromapTriangleKHR> = (0..4u32)
        .map(|i| vk::MicromapTriangleKHR { data_offset: i, subdivision_level: 0, format: vk::OpacityMicromapFormatKHR::TYPE_4_STATE.as_raw() as u16 })
        .collect();
    let tri_array = gpu.upload(&tri_entries);
    let usage = [vk::MicromapUsageEXT { count: 4, subdivision_level: 0, format: vk::OpacityMicromapFormatKHR::TYPE_4_STATE.as_raw() as u32 }];
    let mut mm_info = vk::MicromapBuildInfoEXT {
        ty: vk::MicromapTypeEXT::OPACITY_MICROMAP,
        flags: vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE,
        mode: vk::BuildMicromapModeEXT::BUILD,
        usage_counts_count: 1,
        p_usage_counts: usage.as_ptr(),
        data: vk::DeviceOrHostAddressConstKHR { device_address: data.addr },
        triangle_array: vk::DeviceOrHostAddressConstKHR { device_address: tri_array.addr },
        triangle_array_stride: 8,
        ..Default::default()
    };
    let mut mm_sizes = vk::MicromapBuildSizesInfoEXT::default();
    (omm.fp().get_micromap_build_sizes_ext)(gpu.device.handle(), vk::AccelerationStructureBuildTypeKHR::DEVICE, &mm_info, &mut mm_sizes);
    let mm_buf = ctx.buf(mm_sizes.micromap_size, Mem::Device, 256);
    let mm_base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(mm_buf.buffer));
    let mut micromap = vk::MicromapEXT::null();
    let create = vk::MicromapCreateInfoEXT {
        buffer: mm_buf.buffer,
        offset: mm_buf.addr - mm_base,
        size: mm_sizes.micromap_size,
        ty: vk::MicromapTypeEXT::OPACITY_MICROMAP,
        ..Default::default()
    };
    let res = (omm.fp().create_micromap_ext)(gpu.device.handle(), &create, std::ptr::null(), &mut micromap);
    if res != vk::Result::SUCCESS {
        finish(case, "WRONG", &format!("vkCreateMicromapEXT {res:?}"));
    }
    let mm_scratch = ctx.buf(mm_sizes.build_scratch_size, Mem::Device, 256);
    mm_info.dst_micromap = micromap;
    mm_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: mm_scratch.addr };

    // Front cluster: OMM index per triangle cycles special indices and the four entries;
    // geometry non-opaque so OMM and the candidate loop decide.
    let (nx, ny) = ctx.clamp_grid(10, 10);
    let mut front = grid(3003, nx, ny, [0.05, 0.05, 0.95, 0.95], |x, y| 0.3 + 0.04 * x + 0.02 * y, 1);
    front.geom_flags = GF_CULL_DISABLE;
    front.base_geom = 2;
    let table: [i32; 8] = [-1, -2, -3, 0, 1, 3, -5, 2];
    front.omm_index = Some((0..front.tris.len()).map(|t| table[t % 8]).collect());
    let mut back = grid(5000, 1, 1, [0.0, 0.0, 1.0, 1.0], |_, _| 0.8, 1);
    back.base_geom = 9;
    let clusters = vec![front, back];
    // The micromap is addressed by its storage address (no micromap device-address query exists).
    let up = ctx.upload_clusters(&clusters, mm_buf.addr, &[]);
    let cmd = gpu.begin();
    gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 0, 32);
    gpu.timestamp(cmd, 0);
    (omm.fp().cmd_build_micromaps_ext)(cmd, 1, &mm_info);
    gpu.full_barrier(cmd, false);
    gpu.timestamp(cmd, 1);
    let flags = data_flags() | vk::BuildAccelerationStructureFlagsKHR::ALLOW_CLUSTER_OPACITY_MICROMAPS_NV;
    let clas = ctx.build_clas(cmd, &up, &clusters, flags, None, None);
    gpu.timestamp(cmd, 2);
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0], vec![1]], None);
    gpu.timestamp(cmd, 3);
    let tlas = ctx.build_tlas(cmd, blas.addrs.addr, &[inst_desc([0.0; 3], FRONT_CUSTOM, 0), inst_desc([0.0; 3], BACK_CUSTOM, 1)]);
    gpu.timestamp(cmd, 4);
    let out = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out, tlas, MODE_ALPHA);
    let out2 = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out2, tlas, MODE_ALPHA | MODE_OMM_2_STATE);
    gpu.timestamp(cmd, 5);
    gpu.submit_wait(cmd, case, "omm+build+trace");
    let t = gpu.read_timestamps(6);
    let insts = [ref_inst(FRONT_CUSTOM, 0, [0.0; 3], &[&clusters[0]]), ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[&clusters[1]])];
    let r = check(&out, &insts, MODE_ALPHA, true);
    let r2 = check(&out2, &insts, MODE_ALPHA | MODE_OMM_2_STATE, true);
    // Control: the same rays as if the OMM were ignored (all front triangles non-opaque).
    let mut plain = clusters[0].clone();
    plain.omm_index = None;
    let r_ignored = check(&out, &[ref_inst(FRONT_CUSTOM, 0, [0.0; 3], &[&plain]), insts[1].clone()], MODE_ALPHA, true);
    let warm = [ctx.time_trace(case, pipeline, tlas, MODE_ALPHA), ctx.time_trace(case, pipeline, tlas, MODE_ALPHA | MODE_OMM_2_STATE)];
    let detail = format!(
        "cluster_rev={rev} micromap_bytes={} 4state[{}] 2state[{}] if_omm_ignored_mismatches={:?} {} gpu_us(omm/clas/blas/tlas/cold_traces)={} warm_trace_us(4state/2state)={}",
        mm_sizes.micromap_size,
        r.summary(),
        r2.summary(),
        r_ignored.bad,
        sizes_detail(gpu, &clas, &blas, 2, 2, &clusters),
        us(&t),
        us(&warm)
    );
    (r.ok() && r2.ok(), detail)
}

/// Case 5: cluster templates instantiated with displaced vertices.
/// `source` adds an instantiation with a zero vertex address (positions from the template),
/// run as its own op after the displaced ones.
unsafe fn case_template(ctx: &Ctx, case: &str, pipeline: vk::Pipeline, source: bool) -> (bool, String) {
    let gpu = &ctx.gpu;
    let p = &gpu.cl_props;
    let base = base_clusters(ctx);
    let back = vec![base[4].clone()];
    // Templates from cluster 1 (16-bit indices, no bounds) and cluster 3 (8-bit, with bounds).
    let templates = vec![base[1].clone(), base[3].clone()];
    let displace = |c: &Cluster, s: f32| -> Vec<[f32; 3]> {
        c.verts
            .iter()
            .map(|v| [v[0] + 0.01 * (31.0 * v[1]).sin(), v[1] + 0.01 * (29.0 * v[0]).cos(), v[2] + s * (13.0 * v[0] + 7.0 * v[1]).sin()])
            .collect()
    };
    let d3 = displace(&templates[1], 0.04);
    let d1 = displace(&templates[0], -0.03);
    let bounds = |c: &Cluster, bloat: f32| -> [f32; 8] {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for v in &c.verts {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k] - bloat);
                hi[k] = hi[k].max(v[k] + bloat);
            }
        }
        [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], 0.0, 0.0]
    };
    let bbox = ctx.buf(64, Mem::Upload, u64::from(p.cluster_template_bounds_byte_alignment).max(32));
    bbox.write(0, &bounds(&templates[1], 0.1));
    let up_t = ctx.upload_clusters(&templates, 0, &[0, bbox.addr]);
    let up_back = ctx.upload_clusters(&back, 0, &[]);
    let v3 = gpu.upload(&d3);
    let v1 = gpu.upload(&d1);
    // CLAS address table: [back, inst0, inst1, inst2].
    let table = ctx.buf(8 * 4, Mem::Readback, 16);

    // Knobs for bisecting: PROBE_TMPL=nobbox,nodata,srcstride12,srct1 (comma list).
    let knobs = std::env::var("PROBE_TMPL").unwrap_or_default();
    let knob = |k: &str| knobs.split(',').any(|x| x == k);
    let tmpl_flags = if knob("nodata") { vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE } else { data_flags() };
    let up_t = if knob("nobbox") { ctx.upload_clusters(&templates, 0, &[]) } else { up_t };

    // One submit per stage so a device loss names its stage.
    let cmd = gpu.begin();
    gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 0, 32);
    gpu.timestamp(cmd, 0);
    let mut tri_input = ctx.triangle_input(&templates);
    let mut input = vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut tri_input };
    let tmpl = ctx.cluster_op(cmd, vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER_TEMPLATE, OP_TEMPLATE, up_t.descs.addr, 0, 2, &mut input, tmpl_flags, None, None);
    gpu.timestamp(cmd, 1);
    gpu.submit_wait(cmd, case, "template build");
    eprintln!("stage template: addrs={:x?} sizes={:?}", tmpl.addrs.read::<u64>(2), tmpl.sizes.read::<u32>(2));

    let mut inst_descs = vec![
        InstantiateDesc { template_addr_ptr: tmpl.addrs.addr + 8, vertex_addr: v3.addr, vertex_stride: 12, cluster_id_offset: 100, geometry_index_offset: 5 },
        InstantiateDesc { template_addr_ptr: tmpl.addrs.addr, vertex_addr: v1.addr, vertex_stride: 12, cluster_id_offset: 200, geometry_index_offset: 0 },
        InstantiateDesc { template_addr_ptr: tmpl.addrs.addr, vertex_addr: 0, vertex_stride: 0, cluster_id_offset: 300, geometry_index_offset: 0 },
    ];
    if knob("srcstride12") {
        inst_descs[2].vertex_stride = 12;
    }
    if knob("srct1") {
        inst_descs[2].template_addr_ptr = tmpl.addrs.addr + 8;
    }
    // srcalone: the template-sourced instantiation runs as its own op after the others.
    let alone = source.then(|| inst_descs[2]);
    inst_descs.pop();
    let n_inst = inst_descs.len() as u32;
    let inst_src = gpu.upload(&inst_descs);
    let cmd = gpu.begin();
    gpu.timestamp(cmd, 2);
    let mut tri_input = ctx.triangle_input(&templates);
    let mut input = vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut tri_input };
    let inst = ctx.cluster_op(
        cmd,
        vk::ClusterAccelerationStructureOpTypeNV::INSTANTIATE_TRIANGLE_CLUSTER,
        OP_INSTANTIATE,
        inst_src.addr,
        0,
        n_inst,
        &mut input,
        tmpl_flags,
        None,
        Some(table.addr + 8),
    );
    gpu.timestamp(cmd, 3);
    gpu.submit_wait(cmd, case, "instantiate");
    eprintln!("stage instantiate: table={:x?} sizes={:?}", table.read::<u64>(4), inst.sizes.read::<u32>(n_inst as usize));
    let n_inst = if let Some(d) = alone {
        let src = gpu.upload(&[d]);
        let cmd = gpu.begin();
        let mut tri_input = ctx.triangle_input(&templates);
        let mut input = vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut tri_input };
        let one = ctx.cluster_op(cmd, vk::ClusterAccelerationStructureOpTypeNV::INSTANTIATE_TRIANGLE_CLUSTER, OP_INSTANTIATE, src.addr, 0, 1, &mut input, tmpl_flags, None, Some(table.addr + 24));
        gpu.submit_wait(cmd, case, "instantiate from template positions (alone)");
        eprintln!("stage instantiate alone: table={:x?} size={:?}", table.read::<u64>(4), one.sizes.read::<u32>(1));
        3
    } else {
        n_inst
    };

    let cmd = gpu.begin();
    gpu.timestamp(cmd, 4);
    let clas_back = ctx.build_clas(cmd, &up_back, &back, data_flags(), None, Some(table.addr));
    let mut groups = vec![vec![1, 2], vec![0]];
    if n_inst == 3 {
        groups.push(vec![3]);
    }
    let blas = ctx.build_blas(cmd, table.addr, &groups, None);
    gpu.timestamp(cmd, 5);
    gpu.submit_wait(cmd, case, "back clas + blas");
    eprintln!("stage blas: table={:x?} blas={:x?} sizes={:?}", table.read::<u64>(4), blas.addrs.read::<u64>(groups.len()), blas.sizes.read::<u32>(groups.len()));

    let cmd = gpu.begin();
    gpu.timestamp(cmd, 6);
    let mut tlas_insts = vec![inst_desc([0.0; 3], FRONT_CUSTOM, 0), inst_desc([0.0; 3], BACK_CUSTOM, 1)];
    if n_inst == 3 {
        tlas_insts.push(inst_desc([-0.5, 0.0, 0.0], 99, 2));
    }
    let tlas = ctx.build_tlas(cmd, blas.addrs.addr, &tlas_insts);
    gpu.timestamp(cmd, 7);
    gpu.submit_wait(cmd, case, "tlas");

    let cmd = gpu.begin();
    let out = ctx.out_buffer();
    gpu.timestamp(cmd, 8);
    ctx.trace(cmd, pipeline, &out, tlas, 0);
    gpu.timestamp(cmd, 9);
    gpu.submit_wait(cmd, case, "trace");
    let ts = gpu.read_timestamps(10);
    let t = [ts[0], ts[2], ts[4], ts[6], ts[8]];
    let insts = [
        RefInst {
            custom: FRONT_CUSTOM,
            index: 0,
            offset: [0.0; 3],
            tris: [ref_tris(&templates[1], Some(&d3), 100, 5), ref_tris(&templates[0], Some(&d1), 200, 0)].concat(),
        },
        ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[&back[0]]),
        RefInst { custom: 99, index: 2, offset: [-0.5, 0.0, 0.0], tris: ref_tris(&templates[0], None, 300, 0) },
    ];
    let r = check(&out, &insts[..tlas_insts.len()], 0, true);
    let warm = ctx.time_trace(case, pipeline, tlas, 0);
    let tsizes: Vec<u32> = tmpl.sizes.read(2);
    let isizes: Vec<u32> = inst.sizes.read(n_inst as usize);
    let detail = format!(
        "{} template_bytes={tsizes:?} instantiated_bytes={isizes:?} back_clas_bytes={:?} gpu_us(template/instantiate/clas+blas/tlas/cold_trace)={} warm_trace_us={warm:.1}",
        r.summary(),
        clas_back.sizes.read::<u32>(1),
        us(&t)
    );
    (r.ok() && r.clusters.len() == 1 + n_inst as usize, detail.replace("gpu_us(", &format!("knobs=[{knobs}] gpu_us(")))
}

/// Case 6: CLAS and cluster BLAS in a sparse-residency buffer whose pages are bound out of
/// order from two allocations; implicit data straddles page boundaries.
unsafe fn case_sparse(ctx: &Ctx, case: &str, pipeline: vk::Pipeline) -> (bool, String) {
    let gpu = &ctx.gpu;
    if gpu.features.sparse_binding != vk::TRUE || gpu.features.sparse_residency_buffer != vk::TRUE {
        finish(case, "UNSUPPORTED", "sparseBinding or sparseResidencyBuffer not supported");
    }
    if !gpu.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
        finish(case, "UNSUPPORTED", "no compute queue with SPARSE_BINDING");
    }
    let clusters = base_clusters(ctx);
    let p = &gpu.cl_props;
    // Size the implicit reservations the same way the builds will.
    let tri_input = ctx.triangle_input(&clusters);
    let mut ti = tri_input;
    let info = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(clusters.len() as u32)
        .flags(data_flags())
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
        .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
        .op_input(vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut ti });
    let mut clas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    gpu.cluster.get_cluster_acceleration_structure_build_sizes(&info, &mut clas_sizes);
    let mut bl = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default().max_total_cluster_count(5).max_cluster_count_per_acceleration_structure(4);
    let info = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(2)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL)
        .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
        .op_input(vk::ClusterAccelerationStructureOpInputNV { p_clusters_bottom_level: &mut bl });
    let mut blas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    gpu.cluster.get_cluster_acceleration_structure_build_sizes(&info, &mut blas_sizes);

    let probe = gpu
        .device
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .size(1 << 20)
                .usage(gpu.usage())
                .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY),
            None,
        )
        .unwrap();
    let page = gpu.device.get_buffer_memory_requirements(probe).alignment;
    gpu.device.destroy_buffer(probe, None);
    let clas_off = page - 512;
    let blas_off = align_up(clas_off + clas_sizes.acceleration_structure_size, u64::from(p.cluster_bottom_level_byte_alignment).max(page / 2)) + page / 2 - 256;
    let used = blas_off + blas_sizes.acceleration_structure_size;
    let bound_pages = used.div_ceil(page);
    let total_pages = bound_pages + 2;
    let buffer = gpu
        .device
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .size(total_pages * page)
                .usage(gpu.usage())
                .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY),
            None,
        )
        .unwrap();
    let req = gpu.device.get_buffer_memory_requirements(buffer);
    let ty = gpu.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL).expect("sparse memory type");
    let half = bound_pages.div_ceil(2);
    let alloc = || {
        let mut flags_info = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        gpu.device
            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(half * page).memory_type_index(ty).push(&mut flags_info), None)
            .unwrap()
    };
    let (mem_a, mem_b) = (alloc(), alloc());
    // Page i -> slot (bound_pages - 1 - i): odd slots in A, even slots in B, reversed.
    let binds: Vec<vk::SparseMemoryBind> = (0..bound_pages)
        .map(|i| {
            let slot = bound_pages - 1 - i;
            vk::SparseMemoryBind {
                resource_offset: i * page,
                size: page,
                memory: if slot % 2 == 1 { mem_a } else { mem_b },
                memory_offset: (slot / 2) * page,
                flags: vk::SparseMemoryBindFlags::empty(),
            }
        })
        .collect();
    let buffer_binds = [vk::SparseBufferMemoryBindInfo::default().buffer(buffer).binds(&binds)];
    let bind_info = [vk::BindSparseInfo::default().buffer_binds(&buffer_binds)];
    let fence = gpu.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
    if let Err(e) = gpu.device.queue_bind_sparse(gpu.queue, &bind_info, fence) {
        finish(case, "WRONG", &format!("vkQueueBindSparse {e:?}"));
    }
    gpu.wait(fence, case, "vkQueueBindSparse");
    let base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
    let (ok, detail) = case_clas_tlas(ctx, case, pipeline, true, Some((base + clas_off, base + blas_off)));
    let detail = format!(
        "page={page} pages_bound={bound_pages}/{total_pages} (reversed, two allocations) clas_at={clas_off} blas_at={blas_off} reserve_end={used} {detail}"
    );
    (ok, detail)
}

unsafe fn run(case: &str) {
    let ctx = Ctx::new(case);
    let gpu = &ctx.gpu;
    let slang = gpu.pipeline(&spv!("trace_cluster"));
    let (ok, detail) = match case {
        "clas_tlas" => case_clas_tlas(&ctx, case, slang, true, None),
        "clas_tlas_glsl" => case_clas_tlas(&ctx, case, gpu.pipeline(&spv!("trace_glsl")), true, None),
        "clas_tlas_nocap" => case_clas_tlas(&ctx, case, gpu.pipeline(&spv!("trace_plain")), false, None),
        "clas_mixed" => case_mixed(&ctx, case, slang),
        "clas_ptlas" => case_ptlas(&ctx, case, slang),
        "clas_alpha" => case_alpha(&ctx, case, slang),
        "clas_omm" => case_omm(&ctx, case, slang),
        "clas_template" => case_template(&ctx, case, slang, false),
        "clas_template_src" => case_template(&ctx, case, slang, true),
        "clas_sparse" => case_sparse(&ctx, case, slang),
        _ => finish(case, "UNKNOWN_CASE", "see CASES"),
    };
    let rtp = if gpu.has_rtp { " [ray_tracing_pipeline enabled]" } else { "" };
    finish(case, if ok { "PASS" } else { "WRONG" }, &format!("{detail}{rtp}"));
}

unsafe fn info() {
    let gpu = Gpu::new();
    let props = gpu.instance.get_physical_device_properties(gpu.pdev);
    let v = props.driver_version;
    println!("device: {}", props.device_name_as_c_str().unwrap().to_string_lossy());
    println!("driver: {}.{:02} (raw {v:#x})", v >> 22, (v >> 14) & 0xff);
    println!(
        "api: {}.{}.{}",
        vk::api_version_major(props.api_version),
        vk::api_version_minor(props.api_version),
        vk::api_version_patch(props.api_version)
    );
    for name in REPORTED_EXTENSIONS {
        println!("ext {:<45} {}", name.to_string_lossy(), ext_rev(&gpu.exts, name).map_or("absent".to_owned(), |s| format!("rev {s}")));
    }
    let c = &gpu.cl_props;
    println!(
        "cluster: maxVerticesPerCluster={} maxTrianglesPerCluster={} clusterScratchByteAlignment={} clusterByteAlignment={} clusterTemplateByteAlignment={} clusterBottomLevelByteAlignment={} clusterTemplateBoundsByteAlignment={} maxClusterGeometryIndex={}",
        c.max_vertices_per_cluster,
        c.max_triangles_per_cluster,
        c.cluster_scratch_byte_alignment,
        c.cluster_byte_alignment,
        c.cluster_template_byte_alignment,
        c.cluster_bottom_level_byte_alignment,
        c.cluster_template_bounds_byte_alignment,
        c.max_cluster_geometry_index
    );
    if gpu.has_ptlas {
        println!("ptlas: maxPartitionCount={}", gpu.pt_props.max_partition_count);
    }
    if gpu.has_omm {
        println!(
            "omm: maxOpacity2StateSubdivisionLevel={} maxOpacity4StateSubdivisionLevel={}",
            gpu.omm_props.max_opacity2_state_subdivision_level, gpu.omm_props.max_opacity4_state_subdivision_level
        );
    }
    let a = &gpu.as_props;
    println!(
        "accel: maxGeometryCount={} maxInstanceCount={} maxPrimitiveCount={} minScratchOffsetAlignment={}",
        a.max_geometry_count, a.max_instance_count, a.max_primitive_count, a.min_acceleration_structure_scratch_offset_alignment
    );
    // Features as reported (not as enabled).
    let mut f_cluster = vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV::default();
    let mut f_ptlas = vk::PhysicalDevicePartitionedAccelerationStructureFeaturesNV::default();
    let mut f_fetch = vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default();
    let mut f_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut f_omm = vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push(&mut f_cluster).push(&mut f_ptlas).push(&mut f_fetch).push(&mut f_as).push(&mut f_omm);
    gpu.instance.get_physical_device_features2(gpu.pdev, &mut f2);
    let f = f2.features;
    println!(
        "features: clusterAccelerationStructure={} partitionedAccelerationStructure={} rayTracingPositionFetch={} micromap={} accelerationStructureIndirectBuild={} accelerationStructureCaptureReplay={}",
        f_cluster.cluster_acceleration_structure,
        f_ptlas.partitioned_acceleration_structure,
        f_fetch.ray_tracing_position_fetch,
        f_omm.micromap,
        f_as.acceleration_structure_indirect_build,
        f_as.acceleration_structure_capture_replay
    );
    println!(
        "sparse: sparseBinding={} sparseResidencyBuffer={} sparseResidencyAliased={} residencyNonResidentStrict={} queue_sparse_binding={}",
        f.sparse_binding,
        f.sparse_residency_buffer,
        f.sparse_residency_aliased,
        props.sparse_properties.residency_non_resident_strict,
        gpu.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING)
    );
    println!("timestamp_period_ns: {}", gpu.limits.timestamp_period);
    // Cluster vertexFormat must have ACCELERATION_STRUCTURE_VERTEX_BUFFER support.
    let candidates = [
        vk::Format::R32G32B32_SFLOAT,
        vk::Format::R32G32_SFLOAT,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::Format::R16G16_SFLOAT,
        vk::Format::R16G16B16A16_SNORM,
        vk::Format::R16G16_SNORM,
        vk::Format::R16G16B16A16_UNORM,
        vk::Format::R16G16_UNORM,
        vk::Format::R8G8B8A8_SNORM,
        vk::Format::R8G8_SNORM,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::R8G8_UNORM,
        vk::Format::A2B10G10R10_UNORM_PACK32,
        vk::Format::R16G16B16_SFLOAT,
        vk::Format::R32G32B32A32_SFLOAT,
    ];
    let supported: Vec<String> = candidates
        .iter()
        .filter(|&&f| {
            gpu.instance
                .get_physical_device_format_properties(gpu.pdev, f)
                .buffer_features
                .contains(vk::FormatFeatureFlags::ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR)
        })
        .map(|f| format!("{f:?}"))
        .collect();
    println!("as_vertex_formats: {}", supported.join(" "));
    // Host size estimates for a 128/128 and 256/256 single-cluster build.
    for (t, vtx) in [(128u32, 128u32), (256, 256)] {
        let mut ti = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(1)
            .max_cluster_unique_geometry_count(1)
            .max_cluster_triangle_count(t)
            .max_cluster_vertex_count(vtx)
            .max_total_triangle_count(t * 1000)
            .max_total_vertex_count(vtx * 1000);
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(1000)
            .flags(data_flags())
            .op_type(vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut ti });
        let mut s = vk::AccelerationStructureBuildSizesInfoKHR::default();
        gpu.cluster.get_cluster_acceleration_structure_build_sizes(&info, &mut s);
        println!(
            "host_estimate 1000 clusters {t}t/{vtx}v data_access: implicit={} ({:.0} B/tri) scratch={}",
            s.acceleration_structure_size,
            s.acceleration_structure_size as f64 / f64::from(t * 1000),
            s.build_scratch_size
        );
    }
    if gpu.has_ptlas {
        for (inst, parts) in [(1u32 << 16, 256u32), (1 << 20, 4096)] {
            let input = vk::PartitionedAccelerationStructureInstancesInputNV::default()
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .instance_count(inst)
                .max_instance_per_partition_count(1024)
                .partition_count(parts)
                .max_instance_in_global_partition_count(1024);
            let mut s = vk::AccelerationStructureBuildSizesInfoKHR::default();
            gpu.ptlas.as_ref().unwrap().get_partitioned_acceleration_structures_build_sizes(&input, &mut s);
            println!("ptlas_estimate instances={inst} partitions={parts} global=1024: size={} scratch={}", s.acceleration_structure_size, s.build_scratch_size);
        }
    }
    dump_messages("info");
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
