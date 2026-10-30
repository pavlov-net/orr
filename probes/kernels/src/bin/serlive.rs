//! Driver probe: Shader Execution Reordering for a foliage path tracer.
//!
//! One Slang kernel body (`shaders/pt.slang`) is compiled to a compute entry (inline ray query,
//! variant A, 8x4 groups) and to ray-generation entries: ray query without reorder (B0), ray
//! query turned into a hit object and reordered (B, Bh), ray query then a coherence-hint-only
//! reorder (Bk), pipeline traversal with an any-hit alpha test and no closest-hit (C0, C, Ch),
//! and an extra reorder by ray bins before each bounce (EB, EBk, EC). Each runs on a plain
//! cluster scene and on the same scene with opacity micromaps on the leaf clusters (`_omm`).
//! Every variant is checked per pixel (path hash and radiance) against A and against B0.
//!
//! The scene is thousands of instances of three cluster-BLAS trees (opaque trunk clusters,
//! alpha-tested leaf-card clusters) on a cluster ground plane, traced at 1920x1080 with up to
//! PROBE_BOUNCES cosine bounces and one sun shadow ray per hit.
//!
//! Build from this directory with `cargo build --release --target-dir ../../target/probes`.
//! Run `ser` (everything), `ser info`, `ser only Bk,C_omm` (A always runs as the reference), or
//! `ser mini` (minimal hit-object cases, one process each; `ser mini <n> [heap|classic]`).
//! Knobs: PROBE_BOUNCES (2), PROBE_SHADE (material cost multiplier, 1), PROBE_MIX (1: leaves of
//! one tree in four take the expensive material), PROBE_ROUNDS (timed samples per variant, 40),
//! PROBE_GAP_MS (idle between samples, 6), PROBE_WARMUP_MS (2000), PROBE_GRID (trees per side,
//! 64), PROBE_OMM_LEVEL (6), PROBE_EXEC_STATS=1. Bisection knobs: PROBE_ANYHIT, PROBE_ROOTMAP,
//! PROBE_MINI, PROBE_MINI_HIT; PROBE_FP_PRECISE=1 at build time compiles with `-fp-mode precise`.
//! Set `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` for validation counts; time without it.
//!
//! Copy of `probes/ser` with one addition: PROBE_LIVE=1 runs A and Bk with 0, 16, 32, 64, 128 and
//! 256 extra bytes of path state live across the coherence point (`LIVE_WORDS` in
//! shaders/ser/pt.slang); PROBE_LIVE_OMM=1 runs them on the OMM scene.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use ash::vk::{self, TaggedStructure};
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, c_void};
use std::io::Cursor;
use std::sync::Mutex;
use std::time::Instant;

const W: u32 = 1920;
const H: u32 = 1080;
const PIXELS: usize = (W * H) as usize;
const ATLAS: u32 = 512;
const FENCE_TIMEOUT_NS: u64 = 60_000_000_000;
const MAX_TRIS: usize = 128;
const MAX_VERTS: usize = 256;

const MAT_GROUND: u32 = 0;
const MAT_BARK: u32 = 1;
const MAT_LEAF: u32 = 2;
const MAT_HEAVY: u32 = 3;
const CF_ALPHA: u32 = 1;

/// Cluster geometry flag bits (VkClusterAccelerationStructureGeometryFlagBitsNV).
const GF_CULL_DISABLE: u32 = 1;
const GF_OPAQUE: u32 = 4;

macro_rules! spv {
    ($name:literal) => {
        ash::util::read_spv(&mut Cursor::new(&include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))[..]))
            .expect("valid SPIR-V")
    };
}

// ---------------------------------------------------------------------------------------------
// Shader-visible structs (mirrors of shaders/common.slang, C layout).

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Root {
    tlas: u64,
    insts: u64,
    clusters: u64,
    pos: u64,
    uv: u64,
    idx: u64,
    out_rad: u64,
    out_hash: u64,
    out_stats: u64,
    width: u32,
    height: u32,
    frame: u32,
    bounces: u32,
    tex: u32,
    flags: u32,
    salt: u32,
    shade_scale: u32,
    cam: [f32; 16],
    sun: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Inst {
    m: [f32; 12],
    proto: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ClusterInfo {
    vbase: u32,
    ibase: u32,
    material: u32,
    flags: u32,
}

const _: () = {
    assert!(size_of::<Root>() == 184);
    assert!(size_of::<Inst>() == 64);
    assert!(size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV>() == 64);
    assert!(size_of::<vk::ClusterAccelerationStructureBuildClustersBottomLevelInfoNV>() == 16);
    assert!(size_of::<vk::AccelerationStructureInstanceKHR>() == 64);
};

fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

// ---------------------------------------------------------------------------------------------
// Validation message collection.

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

fn dump_messages() -> (u32, u32) {
    let map = MESSAGES.lock().unwrap();
    let errors: u32 = map.iter().filter(|(k, _)| k.starts_with('E')).map(|(_, v)| v.0).sum();
    let warnings: u32 = map.iter().filter(|(k, _)| k.starts_with('W')).map(|(_, v)| v.0).sum();
    for (id, (count, message)) in map.iter() {
        let short: String = message.chars().take(700).collect();
        println!("VVL {id} x{count}: {short}");
    }
    (errors, warnings)
}

fn fatal(what: &str) -> ! {
    let (e, w) = dump_messages();
    println!("FATAL {what} (validation errors={e} warnings={w})");
    std::process::exit(2);
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
    /// Offset of `addr` inside `buffer`.
    offset: u64,
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
    pool: vk::CommandPool,
    mem: vk::PhysicalDeviceMemoryProperties,
    limits: vk::PhysicalDeviceLimits,
    exts: Vec<vk::ExtensionProperties>,
    as_props: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    cl_props: vk::PhysicalDeviceClusterAccelerationStructurePropertiesNV<'static>,
    rt_props: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    ser_props: vk::PhysicalDeviceRayTracingInvocationReorderPropertiesEXT<'static>,
    heap_props: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    omm_props: vk::PhysicalDeviceOpacityMicromapPropertiesEXT<'static>,
    accel: ash::khr::acceleration_structure::Device,
    cluster: ash::nv::cluster_acceleration_structure::Device,
    omm: ash::ext::opacity_micromap::Device,
    rt: ash::khr::ray_tracing_pipeline::Device,
    heap: ash::ext::descriptor_heap::Device,
    exec: Option<ash::khr::pipeline_executable_properties::Device>,
    timestamps: vk::QueryPool,
}

const REQUIRED_EXTENSIONS: &[&CStr] = &[
    ash::khr::acceleration_structure::NAME,
    ash::khr::deferred_host_operations::NAME,
    ash::khr::ray_query::NAME,
    ash::khr::ray_tracing_pipeline::NAME,
    ash::ext::ray_tracing_invocation_reorder::NAME,
    ash::nv::cluster_acceleration_structure::NAME,
    ash::ext::opacity_micromap::NAME,
    ash::ext::descriptor_heap::NAME,
    ash::khr::shader_untyped_pointers::NAME,
];

const REPORTED_EXTENSIONS: &[&CStr] = &[
    c"VK_KHR_ray_tracing_pipeline",
    c"VK_EXT_ray_tracing_invocation_reorder",
    c"VK_NV_ray_tracing_invocation_reorder",
    c"VK_NV_cluster_acceleration_structure",
    c"VK_EXT_opacity_micromap",
    c"VK_KHR_opacity_micromap",
    c"VK_EXT_descriptor_heap",
    c"VK_KHR_pipeline_executable_properties",
];

fn has_ext(list: &[vk::ExtensionProperties], name: &CStr) -> bool {
    list.iter().any(|e| e.extension_name_as_c_str() == Ok(name))
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a.max(1)) * a.max(1)
}

impl Gpu {
    unsafe fn new() -> Gpu {
        let entry = ash::Entry::load().expect("Vulkan loader");
        let instance_exts = entry.enumerate_instance_extension_properties(None).unwrap();
        let debug = has_ext(&instance_exts, ash::ext::debug_utils::NAME);
        let app = vk::ApplicationInfo::default().application_name(c"ser").api_version(vk::API_VERSION_1_4);
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
            .expect("no Vulkan 1.4 device with ray tracing pipeline, EXT invocation reorder, clusters, OMM and descriptor heap");
        let exts = instance.enumerate_device_extension_properties(pdev).unwrap();
        let has_exec = has_ext(&exts, ash::khr::pipeline_executable_properties::NAME);

        let family = instance
            .get_physical_device_queue_family_properties(pdev)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
            .unwrap() as u32;

        let mut f_heap = vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default().descriptor_heap(true);
        let mut f_untyped = vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR::default().shader_untyped_pointers(true);
        let mut f_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
        let mut f_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut f_rtp = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
        let mut f_ser = vk::PhysicalDeviceRayTracingInvocationReorderFeaturesEXT::default().ray_tracing_invocation_reorder(true);
        let mut f_cluster = vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV::default().cluster_acceleration_structure(true);
        let mut f_omm = vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default().micromap(true);
        let mut f_exec = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true).scalar_block_layout(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true).maintenance4(true);
        let mut f14 = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true);
        let core = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        let mut f2 = vk::PhysicalDeviceFeatures2::default().features(core);

        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
        let mut names: Vec<_> = REQUIRED_EXTENSIONS.iter().map(|n| n.as_ptr()).collect();
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .push(&mut f2)
            .push(&mut f12)
            .push(&mut f13)
            .push(&mut f14)
            .push(&mut f_heap)
            .push(&mut f_untyped)
            .push(&mut f_as)
            .push(&mut f_rq)
            .push(&mut f_rtp)
            .push(&mut f_ser)
            .push(&mut f_cluster)
            .push(&mut f_omm);
        if has_exec {
            names.push(ash::khr::pipeline_executable_properties::NAME.as_ptr());
            info = info.push(&mut f_exec);
        }
        let info = info.enabled_extension_names(&names);
        let device = instance.create_device(pdev, &info, None).expect("device");
        let queue = device.get_device_queue(family, 0);
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();

        let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut cl_props = vk::PhysicalDeviceClusterAccelerationStructurePropertiesNV::default();
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut ser_props = vk::PhysicalDeviceRayTracingInvocationReorderPropertiesEXT::default();
        let mut heap_props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
        let mut omm_props = vk::PhysicalDeviceOpacityMicromapPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push(&mut as_props)
            .push(&mut cl_props)
            .push(&mut rt_props)
            .push(&mut ser_props)
            .push(&mut heap_props)
            .push(&mut omm_props);
        instance.get_physical_device_properties2(pdev, &mut props2);
        let limits = props2.properties.limits;
        as_props.p_next = std::ptr::null_mut();
        cl_props.p_next = std::ptr::null_mut();
        rt_props.p_next = std::ptr::null_mut();
        ser_props.p_next = std::ptr::null_mut();
        heap_props.p_next = std::ptr::null_mut();
        omm_props.p_next = std::ptr::null_mut();

        let timestamps = device
            .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(256), None)
            .unwrap();

        Gpu {
            accel: ash::khr::acceleration_structure::Device::load(&instance, &device),
            cluster: ash::nv::cluster_acceleration_structure::Device::load(&instance, &device),
            omm: ash::ext::opacity_micromap::Device::load(&instance, &device),
            rt: ash::khr::ray_tracing_pipeline::Device::load(&instance, &device),
            heap: ash::ext::descriptor_heap::Device::load(&instance, &device),
            exec: has_exec.then(|| ash::khr::pipeline_executable_properties::Device::load(&instance, &device)),
            mem: instance.get_physical_device_memory_properties(pdev),
            _entry: entry,
            instance,
            pdev,
            device,
            queue,
            pool,
            limits,
            exts,
            as_props,
            cl_props,
            rt_props,
            ser_props,
            heap_props,
            omm_props,
            timestamps,
        }
    }

    unsafe fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem.memory_type_count).find(|&i| bits & (1 << i) != 0 && self.mem.memory_types[i as usize].property_flags.contains(flags))
    }

    /// Buffer whose returned address is aligned to `align`; host-visible kinds are mapped and zeroed.
    unsafe fn buffer_usage(&self, size: u64, kind: Mem, align: u64, usage: vk::BufferUsageFlags) -> Buf {
        let size = size.max(16);
        let align = align.max(16);
        let usage = usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size + align).usage(usage), None).unwrap();
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
        Buf { buffer, addr, offset, ptr, size }
    }

    unsafe fn buffer(&self, size: u64, kind: Mem, align: u64) -> Buf {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
            | vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT
            | vk::BufferUsageFlags::MICROMAP_STORAGE_EXT
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        self.buffer_usage(size, kind, align, usage)
    }

    unsafe fn upload<T: Copy>(&self, data: &[T]) -> Buf {
        let b = self.buffer(std::mem::size_of_val(data) as u64, Mem::Upload, 256);
        b.write(0, data);
        b
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

    /// Submits and waits; device loss and timeouts end the process.
    unsafe fn submit_wait(&self, cmd: vk::CommandBuffer, what: &str) {
        self.full_barrier(cmd);
        self.device.end_command_buffer(cmd).unwrap();
        let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        if let Err(e) = self.device.queue_submit(self.queue, &[submit], fence) {
            fatal(&format!("submit {what}: {e:?}"));
        }
        if let Err(e) = self.device.wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS) {
            fatal(&format!("wait {what}: {e:?}"));
        }
        self.device.destroy_fence(fence, None);
        self.device.free_command_buffers(self.pool, &cmds);
    }

    unsafe fn full_barrier(&self, cmd: vk::CommandBuffer) {
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::HOST_READ)];
        self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
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
// Leaf atlas and opacity micromaps.

/// Card-local coverage of leaf shape `q` (2x2 atlas quadrants) at (u, v) in [0, 1].
fn leaf_shape(q: u32, u: f32, v: f32) -> bool {
    let (x, y) = (u - 0.5, v - 0.5);
    match q {
        // Broad serrated leaf.
        0 => {
            let a = y.atan2(x);
            let r = 1.0 + 0.08 * (24.0 * a).sin();
            (x / 0.27).powi(2) + (y / 0.44).powi(2) < r * r
        }
        // Three leaflets.
        1 => [(0.0, 0.18, 0.2), (-0.17, -0.12, 0.16), (0.17, -0.12, 0.16)]
            .iter()
            .any(|&(cx, cy, r)| ((x - cx) / (0.7 * r)).powi(2) + ((y - cy) / r).powi(2) < 1.0),
        // Needle tuft: thin blades radiating from the base.
        2 => {
            let (bx, by) = (x, y + 0.45);
            let len = (bx * bx + by * by).sqrt();
            if len > 0.85 || by < 0.0 {
                return false;
            }
            let a = bx.atan2(by);
            let blade = (a * 7.0).rem_euclid(std::f32::consts::PI / 3.0) - std::f32::consts::PI / 6.0;
            (blade * len).abs() < 0.018
        }
        // Lobed leaf.
        _ => {
            let a = y.atan2(x);
            let r = 0.36 * (0.55 + 0.45 * (2.5 * a).cos().abs());
            x * x + y * y < r * r
        }
    }
}

/// RGBA8 atlas: four leaf shapes, alpha 0 or 255.
fn make_atlas() -> Vec<[u8; 4]> {
    let mut px = vec![[0u8; 4]; (ATLAS * ATLAS) as usize];
    let half = ATLAS / 2;
    for y in 0..ATLAS {
        for x in 0..ATLAS {
            let q = (x / half) + 2 * (y / half);
            let u = ((x % half) as f32 + 0.5) / half as f32;
            let v = ((y % half) as f32 + 0.5) / half as f32;
            let inside = leaf_shape(q, u, v);
            let vein = u8::from((u - 0.5).abs() < 0.01);
            let base: [u8; 3] = [[60, 120, 40], [80, 140, 30], [40, 90, 50], [110, 120, 30]][q as usize];
            px[(y * ATLAS + x) as usize] = [base[0] - 20 * vein, base[1] - 30 * vein, base[2], if inside { 255 } else { 0 }];
        }
    }
    px
}

/// Micromap index of barycentrics (u, v) at `level` (Vulkan reference code).
fn bary_to_index(u: f32, v: f32, level: u32) -> u32 {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let n = 1u32 << level;
    let fu = u * n as f32;
    let fv = v * n as f32;
    let mut iu = fu as u32;
    let mut iv = fv as u32;
    let uf = fu - iu as f32;
    let vf = fv - iv as f32;
    if iu >= n {
        iu = n - 1;
    }
    if iv >= n {
        iv = n - 1;
    }
    let iuv = iu + iv;
    if iuv >= n {
        iu -= iuv - n + 1;
    }
    let mut iw = !(iu + iv);
    if uf + vf >= 1.0 && iuv < n - 1 {
        iw = iw.wrapping_sub(1);
    }
    let mut b0 = !(iu ^ iw);
    b0 &= n - 1;
    let t = (iu ^ iv) & b0;
    let mut f = t;
    f ^= f >> 1;
    f ^= f >> 2;
    f ^= f >> 4;
    f ^= f >> 8;
    let mut b1 = ((f ^ iu) & !b0) | t;
    b0 = (b0 | (b0 << 8)) & 0x00ff00ff;
    b0 = (b0 | (b0 << 4)) & 0x0f0f0f0f;
    b0 = (b0 | (b0 << 2)) & 0x33333333;
    b0 = (b0 | (b0 << 1)) & 0x55555555;
    b1 = (b1 | (b1 << 8)) & 0x00ff00ff;
    b1 = (b1 | (b1 << 4)) & 0x0f0f0f0f;
    b1 = (b1 | (b1 << 2)) & 0x33333333;
    b1 = (b1 | (b1 << 1)) & 0x55555555;
    b0 | (b1 << 1)
}

/// Card corner UVs (card-local): triangle 0 = corners 0,1,2; triangle 1 = corners 0,2,3.
const CARD_UV: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
const CARD_TRIS: [[u32; 3]; 2] = [[0, 1, 2], [0, 2, 3]];

fn atlas_uv(q: u32, c: [f32; 2]) -> [f32; 2] {
    [((q % 2) as f32 + c[0]) * 0.5, ((q / 2) as f32 + c[1]) * 0.5]
}

/// 4-state micromap for (quadrant, card triangle): one state per micro-triangle, conservative
/// against nearest sampling of the atlas (5x5 texel dilation around dense samples).
fn omm_states(alpha: &[bool], q: u32, tri: usize, level: u32) -> Vec<u8> {
    let count = 1usize << (2 * level);
    // Bit 0: saw an opaque texel, bit 1: saw a transparent texel.
    let mut seen = vec![0u8; count];
    let n = (1u32 << level) * 8;
    let corners = CARD_TRIS[tri].map(|k| atlas_uv(q, CARD_UV[k as usize]));
    for j in 0..n {
        for i in 0..n - j {
            for (du, dv) in [(0.25, 0.25), (0.6, 0.3)] {
                let u = (i as f32 + du) / n as f32;
                let v = (j as f32 + dv) / n as f32;
                if u + v > 1.0 {
                    continue;
                }
                let w = 1.0 - u - v;
                let s = corners[0][0] * w + corners[1][0] * u + corners[2][0] * v;
                let t = corners[0][1] * w + corners[1][1] * u + corners[2][1] * v;
                let tx = (s * ATLAS as f32).floor() as i32;
                let ty = (t * ATLAS as f32).floor() as i32;
                let idx = bary_to_index(u, v, level) as usize;
                for oy in -2..=2 {
                    for ox in -2..=2 {
                        let x = (tx + ox).clamp(0, ATLAS as i32 - 1);
                        let y = (ty + oy).clamp(0, ATLAS as i32 - 1);
                        seen[idx] |= if alpha[(y * ATLAS as i32 + x) as usize] { 1 } else { 2 };
                    }
                }
            }
        }
    }
    seen.iter()
        .map(|&s| match s {
            1 => 1, // opaque
            2 => 0, // transparent
            _ => 2, // unknown-transparent: the alpha test decides
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Scene generation.

type V3 = [f32; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn mul(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> V3 {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    mul(a, 1.0 / l)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 16777216.0
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    fn dir(&mut self) -> V3 {
        loop {
            let v = [self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0)];
            let l = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if l > 1e-3 && l <= 1.0 {
                return norm(v);
            }
        }
    }
}

#[derive(Default, Clone)]
struct ClusterCpu {
    pos: Vec<V3>,
    uv: Vec<[f32; 2]>,
    tris: Vec<[u32; 3]>,
    /// Per-triangle micromap index (alpha clusters only).
    omm: Vec<u32>,
    material: u32,
    alpha: bool,
}

/// Tapered tube from `a` to `b` appended to an indexed mesh.
fn tube(pos: &mut Vec<V3>, tris: &mut Vec<[u32; 3]>, a: V3, b: V3, r0: f32, r1: f32, sides: u32, segs: u32) {
    let axis = norm(sub(b, a));
    let t = if axis[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let u = norm(cross(axis, t));
    let v = cross(axis, u);
    let base = pos.len() as u32;
    for s in 0..=segs {
        let f = s as f32 / segs as f32;
        let c = add(a, mul(sub(b, a), f));
        let r = r0 + (r1 - r0) * f;
        for k in 0..sides {
            let ang = k as f32 / sides as f32 * std::f32::consts::TAU;
            pos.push(add(c, add(mul(u, ang.cos() * r), mul(v, ang.sin() * r))));
        }
    }
    for s in 0..segs {
        for k in 0..sides {
            let i0 = base + s * sides + k;
            let i1 = base + s * sides + (k + 1) % sides;
            let (j0, j1) = (i0 + sides, i1 + sides);
            tris.push([i0, i1, j1]);
            tris.push([i0, j1, j0]);
        }
    }
}

/// Splits an indexed mesh into clusters of at most MAX_TRIS triangles and MAX_VERTS vertices.
fn clusterize(pos: &[V3], tris: &[[u32; 3]], material: u32) -> Vec<ClusterCpu> {
    let mut out = Vec::new();
    let mut cur = ClusterCpu { material, ..Default::default() };
    let mut map: HashMap<u32, u32> = HashMap::new();
    for t in tris {
        let new = t.iter().filter(|i| !map.contains_key(i)).count();
        if cur.tris.len() == MAX_TRIS || map.len() + new > MAX_VERTS {
            out.push(std::mem::replace(&mut cur, ClusterCpu { material, ..Default::default() }));
            map.clear();
        }
        let local = t.map(|i| {
            *map.entry(i).or_insert_with(|| {
                cur.pos.push(pos[i as usize]);
                cur.uv.push([0.0, 0.0]);
                (cur.pos.len() - 1) as u32
            })
        });
        cur.tris.push(local);
    }
    if !cur.tris.is_empty() {
        out.push(cur);
    }
    out
}

struct TreeParams {
    height: f32,
    r0: f32,
    r1: f32,
    crown_y: f32,
    crown_r: f32,
    crown_rv: f32,
    cards: u32,
    card_size: f32,
    bark: u32,
    quadrants: &'static [u32],
}

const TREES: [TreeParams; 3] = [
    TreeParams { height: 10.0, r0: 0.35, r1: 0.08, crown_y: 7.0, crown_r: 3.5, crown_rv: 2.8, cards: 2400, card_size: 0.55, bark: MAT_BARK, quadrants: &[0, 3] },
    TreeParams { height: 14.0, r0: 0.30, r1: 0.05, crown_y: 9.0, crown_r: 2.2, crown_rv: 4.5, cards: 3000, card_size: 0.42, bark: MAT_BARK, quadrants: &[2, 1] },
    TreeParams { height: 5.5, r0: 0.25, r1: 0.06, crown_y: 3.6, crown_r: 2.6, crown_rv: 1.8, cards: 1600, card_size: 0.5, bark: MAT_HEAVY, quadrants: &[1, 3] },
];

fn tree(p: &TreeParams, rng: &mut Rng) -> Vec<ClusterCpu> {
    let mut pos = Vec::new();
    let mut tris = Vec::new();
    tube(&mut pos, &mut tris, [0.0, -0.5, 0.0], [0.0, p.height, 0.0], p.r0, p.r1, 8, 12);
    for b in 0..7 {
        let y = p.height * (0.45 + 0.4 * b as f32 / 6.0);
        let ang = rng.range(0.0, std::f32::consts::TAU);
        let dir = norm([ang.cos(), rng.range(0.3, 0.8), ang.sin()]);
        let start = [0.0, y, 0.0];
        tube(&mut pos, &mut tris, start, add(start, mul(dir, p.crown_r * 0.9)), p.r0 * 0.3, p.r1 * 0.5, 6, 3);
    }
    let mut clusters = clusterize(&pos, &tris, p.bark);

    let center = [0.0, p.crown_y, 0.0];
    let mut cur = ClusterCpu { material: MAT_LEAF, alpha: true, ..Default::default() };
    for c in 0..p.cards {
        let off = loop {
            let o = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if o[0] * o[0] + o[1] * o[1] + o[2] * o[2] <= 1.0 {
                break [o[0] * p.crown_r, o[1] * p.crown_rv, o[2] * p.crown_r];
            }
        };
        let mid = add(center, off);
        let n = rng.dir();
        let t = if n[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let a = norm(cross(n, t));
        let b = cross(n, a);
        let s = p.card_size * rng.range(0.8, 1.2) * 0.5;
        let q = p.quadrants[(c % p.quadrants.len() as u32) as usize];
        let base = cur.pos.len() as u32;
        for cu in CARD_UV {
            let ofs = add(mul(a, (cu[0] * 2.0 - 1.0) * s), mul(b, (cu[1] * 2.0 - 1.0) * s));
            cur.pos.push(add(mid, ofs));
            cur.uv.push(atlas_uv(q, cu));
        }
        for (ti, t) in CARD_TRIS.iter().enumerate() {
            cur.tris.push(t.map(|i| base + i));
            cur.omm.push(q * 2 + ti as u32);
        }
        if cur.tris.len() == MAX_TRIS {
            clusters.push(std::mem::replace(&mut cur, ClusterCpu { material: MAT_LEAF, alpha: true, ..Default::default() }));
        }
    }
    if !cur.tris.is_empty() {
        clusters.push(cur);
    }
    clusters
}

fn ground() -> Vec<ClusterCpu> {
    let mut out = Vec::new();
    let (extent, tiles, quads) = (320.0f32, 4u32, 8u32);
    let tile = 2.0 * extent / tiles as f32;
    for ty in 0..tiles {
        for tx in 0..tiles {
            let mut c = ClusterCpu { material: MAT_GROUND, ..Default::default() };
            for j in 0..=quads {
                for i in 0..=quads {
                    let x = -extent + tile * (tx as f32 + i as f32 / quads as f32);
                    let z = -extent + tile * (ty as f32 + j as f32 / quads as f32);
                    c.pos.push([x, 0.0, z]);
                    c.uv.push([x * 0.1, z * 0.1]);
                }
            }
            for j in 0..quads {
                for i in 0..quads {
                    let a = j * (quads + 1) + i;
                    let (b, d) = (a + 1, a + quads + 1);
                    c.tris.push([a, b, d + 1]);
                    c.tris.push([a, d + 1, d]);
                }
            }
            out.push(c);
        }
    }
    out
}

struct Camera {
    pos: V3,
    fwd: V3,
    right: V3,
    up: V3,
    tan_half: f32,
}

fn camera() -> Camera {
    let pitch = 6.0f32.to_radians();
    let fwd = norm([0.0, pitch.sin(), pitch.cos()]);
    let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
    let up = cross(right, fwd);
    Camera { pos: [0.0, 1.7, -205.0], fwd, right, up, tan_half: 30.0f32.to_radians().tan() }
}

/// Instance placements: (prototype, transform). Index 0 is the ground (prototype 3).
fn placements(grid: u32, cam: &Camera) -> Vec<(u32, [f32; 12])> {
    let mut rng = Rng(0x5eed_1234);
    let mut out = vec![(3, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])];
    let spacing = 7.0;
    for j in 0..grid {
        for i in 0..grid {
            let x = (i as f32 - grid as f32 / 2.0 + 0.5) * spacing + rng.range(-2.5, 2.5);
            let z = (j as f32 - grid as f32 / 2.0 + 0.5) * spacing + rng.range(-2.5, 2.5);
            let r = rng.f();
            let yaw = rng.range(0.0, std::f32::consts::TAU);
            let s = rng.range(0.8, 1.25);
            if (x - cam.pos[0]).hypot(z - cam.pos[2]) < 5.0 {
                continue;
            }
            let proto = if r < 0.45 {
                0
            } else if r < 0.8 {
                1
            } else {
                2
            };
            let (c, sn) = (yaw.cos() * s, yaw.sin() * s);
            out.push((proto, [c, 0.0, sn, x, 0.0, s, 0.0, 0.0, -sn, 0.0, c, z]));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Scene upload and acceleration structures.

struct Scene {
    tlas_plain: u64,
    tlas_omm: u64,
    insts: Buf,
    clusters: Buf,
    pos: Buf,
    uv: Buf,
    idx: Buf,
    stats: String,
}

/// VkClusterAccelerationStructureBuildTriangleClusterInfoNV, packed (64 bytes), 32-bit indices.
fn pack_cluster(id: u32, tris: u32, verts: u32, geom_flags: u32, index_addr: u64, vertex_addr: u64, omm: Option<(u64, u64)>) -> [u32; 16] {
    let mut o = [0u32; 16];
    let omm_index_type = if omm.is_some() { 4 } else { 1 };
    o[0] = id;
    o[2] = tris | (verts << 9) | (4 << 24) | (omm_index_type << 28);
    // Base geometry index = cluster id: shaders look the cluster up by geometry index.
    o[3] = (id & 0xff_ffff) | (geom_flags << 29);
    o[4] = 4 | (12 << 16);
    o[5] = if omm.is_some() { 4 << 16 } else { 0 };
    let mut put64 = |w: usize, v: u64| {
        o[w] = v as u32;
        o[w + 1] = (v >> 32) as u32;
    };
    put64(6, index_addr);
    put64(8, vertex_addr);
    if let Some((array, index)) = omm {
        put64(12, array);
        put64(14, index);
    }
    o
}

impl Gpu {
    /// One indirect cluster op in implicit-destination mode; returns the implicit data size.
    unsafe fn cluster_op(
        &self,
        cmd: vk::CommandBuffer,
        op_type: vk::ClusterAccelerationStructureOpTypeNV,
        input: vk::ClusterAccelerationStructureOpInputNV<'_>,
        infos: &Buf,
        stride: u64,
        n: u32,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        addrs: &Buf,
    ) -> u64 {
        let p = &self.cl_props;
        let count = self.upload(&[n]);
        let info = vk::ClusterAccelerationStructureInputInfoNV::default()
            .max_acceleration_structure_count(n)
            .flags(flags)
            .op_type(op_type)
            .op_mode(vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS)
            .op_input(input);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        self.cluster.get_cluster_acceleration_structure_build_sizes(&info, &mut sizes);
        let align = if op_type == vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL {
            p.cluster_bottom_level_byte_alignment
        } else {
            p.cluster_byte_alignment
        };
        let implicit = self.buffer(sizes.acceleration_structure_size, Mem::Device, u64::from(align));
        let scratch = self.buffer(sizes.build_scratch_size, Mem::Device, u64::from(p.cluster_scratch_byte_alignment));
        let sizes_out = self.buffer(4 * u64::from(n), Mem::Device, 16);
        let region = |addr: u64, stride: u64| vk::StridedDeviceAddressRegionKHR { device_address: addr, stride, size: stride * u64::from(n) };
        let commands = vk::ClusterAccelerationStructureCommandsInfoNV::default()
            .input(info)
            .dst_implicit_data(implicit.addr)
            .scratch_data(scratch.addr)
            .dst_addresses_array(region(addrs.addr, 8))
            .dst_sizes_array(region(sizes_out.addr, 4))
            .src_infos_array(region(infos.addr, stride))
            .src_infos_count(count.addr);
        self.cluster.cmd_build_cluster_acceleration_structure_indirect(cmd, &commands);
        self.full_barrier(cmd);
        sizes.acceleration_structure_size
    }

    unsafe fn build_tlas(&self, instances: &[vk::AccelerationStructureInstanceKHR]) -> u64 {
        let n = instances.len() as u32;
        let src = self.upload(instances);
        let geometry = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default().data(vk::DeviceOrHostAddressConstKHR { device_address: src.addr }),
            })];
        let mut build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        self.accel.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &build, Some(&[n]), &mut sizes);
        let backing = self.buffer(sizes.acceleration_structure_size, Mem::Device, 256);
        let tlas = self
            .accel
            .create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(backing.buffer)
                    .offset(backing.offset)
                    .size(sizes.acceleration_structure_size)
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                None,
            )
            .unwrap();
        let scratch = self.buffer(sizes.build_scratch_size, Mem::Device, u64::from(self.as_props.min_acceleration_structure_scratch_offset_alignment));
        build = build.dst_acceleration_structure(tlas).scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch.addr });
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(n)];
        let cmd = self.begin();
        self.accel.cmd_build_acceleration_structures(cmd, &[build], &[Some(&ranges)]);
        self.submit_wait(cmd, "tlas");
        self.accel.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(tlas))
    }

    /// 4-state micromap with `entries` triangles at `level`; returns its storage address.
    unsafe fn build_micromap(&self, data: &[u8], entries: u32, level: u32) -> u64 {
        let bytes_per = (1u32 << (2 * level)) / 4;
        let data_buf = self.upload(data);
        let tri: Vec<vk::MicromapTriangleKHR> = (0..entries)
            .map(|i| vk::MicromapTriangleKHR {
                data_offset: i * bytes_per,
                subdivision_level: level as u16,
                format: vk::OpacityMicromapFormatKHR::TYPE_4_STATE.as_raw() as u16,
            })
            .collect();
        let tri_buf = self.upload(&tri);
        let usage = [vk::MicromapUsageEXT { count: entries, subdivision_level: level, format: vk::OpacityMicromapFormatKHR::TYPE_4_STATE.as_raw() as u32 }];
        let mut info = vk::MicromapBuildInfoEXT {
            ty: vk::MicromapTypeEXT::OPACITY_MICROMAP,
            flags: vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE,
            mode: vk::BuildMicromapModeEXT::BUILD,
            usage_counts_count: 1,
            p_usage_counts: usage.as_ptr(),
            data: vk::DeviceOrHostAddressConstKHR { device_address: data_buf.addr },
            triangle_array: vk::DeviceOrHostAddressConstKHR { device_address: tri_buf.addr },
            triangle_array_stride: 8,
            ..Default::default()
        };
        let mut sizes = vk::MicromapBuildSizesInfoEXT::default();
        (self.omm.fp().get_micromap_build_sizes_ext)(self.device.handle(), vk::AccelerationStructureBuildTypeKHR::DEVICE, &info, &mut sizes);
        let storage = self.buffer(sizes.micromap_size, Mem::Device, 256);
        let mut micromap = vk::MicromapEXT::null();
        let create = vk::MicromapCreateInfoEXT {
            buffer: storage.buffer,
            offset: storage.offset,
            size: sizes.micromap_size,
            ty: vk::MicromapTypeEXT::OPACITY_MICROMAP,
            ..Default::default()
        };
        let res = (self.omm.fp().create_micromap_ext)(self.device.handle(), &create, std::ptr::null(), &mut micromap);
        assert_eq!(res, vk::Result::SUCCESS, "vkCreateMicromapEXT");
        let scratch = self.buffer(sizes.build_scratch_size, Mem::Device, 256);
        info.dst_micromap = micromap;
        info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch.addr };
        let cmd = self.begin();
        (self.omm.fp().cmd_build_micromaps_ext)(cmd, 1, &info);
        self.submit_wait(cmd, "micromap");
        storage.addr
    }
}

unsafe fn build_scene(gpu: &Gpu, grid: u32, omm_level: u32) -> Scene {
    let t0 = Instant::now();
    let mut rng = Rng(0x7ee5);
    let mut protos: Vec<Vec<ClusterCpu>> = TREES.iter().map(|p| tree(p, &mut rng)).collect();
    protos.push(ground());

    // Global vertex / index arrays; geometry index = global cluster index.
    let mut pos: Vec<V3> = Vec::new();
    let mut uv: Vec<[f32; 2]> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();
    let mut omm_idx: Vec<u32> = Vec::new();
    let mut infos: Vec<ClusterInfo> = Vec::new();
    let mut ranges = Vec::new(); // (first cluster, count) per prototype
    let mut tri_total = Vec::new();
    for clusters in &protos {
        ranges.push((infos.len() as u32, clusters.len() as u32));
        tri_total.push(clusters.iter().map(|c| c.tris.len()).sum::<usize>());
        for c in clusters {
            assert!(c.tris.len() <= MAX_TRIS && c.pos.len() <= MAX_VERTS);
            infos.push(ClusterInfo {
                vbase: pos.len() as u32,
                ibase: idx.len() as u32,
                material: c.material,
                flags: if c.alpha { CF_ALPHA } else { 0 },
            });
            pos.extend_from_slice(&c.pos);
            uv.extend_from_slice(&c.uv);
            idx.extend(c.tris.iter().flatten());
            if c.alpha {
                omm_idx.extend_from_slice(&c.omm);
            } else {
                omm_idx.extend(std::iter::repeat_n(0, c.tris.len()));
            }
        }
    }
    let n_clusters = infos.len() as u32;
    let pos_buf = gpu.upload(&pos);
    let uv_buf = gpu.upload(&uv);
    let idx_buf = gpu.upload(&idx);
    let omm_idx_buf = gpu.upload(&omm_idx);
    let clusters_buf = gpu.upload(&infos);

    // Micromaps: 4 quadrants x 2 card triangles.
    let atlas = make_atlas();
    let alpha: Vec<bool> = atlas.iter().map(|p| p[3] >= 128).collect();
    let mut omm_data = Vec::new();
    let mut omm_hist = [0usize; 3];
    for e in 0..8u32 {
        let states = omm_states(&alpha, e / 2, (e % 2) as usize, omm_level);
        let mut packed = vec![0u8; states.len() / 4];
        for (i, &s) in states.iter().enumerate() {
            packed[i / 4] |= s << ((i % 4) * 2);
            omm_hist[usize::from(s.min(2))] += 1;
        }
        omm_data.extend_from_slice(&packed);
    }
    let mm_addr = gpu.build_micromap(&omm_data, 8, omm_level);

    // CLAS builds (plain and with micromaps), then one cluster BLAS per prototype for each.
    let mut tlas = [0u64; 2];
    let mut sizes_note = String::new();
    let all: Vec<&ClusterCpu> = protos.iter().flatten().collect();
    let max_tris = all.iter().map(|c| c.tris.len() as u32).max().unwrap();
    let max_verts = all.iter().map(|c| c.pos.len() as u32).max().unwrap();
    let total_tris = all.iter().map(|c| c.tris.len() as u32).sum::<u32>();
    let total_verts = all.iter().map(|c| c.pos.len() as u32).sum::<u32>();
    let place = placements(grid, &camera());
    for (k, with_omm) in [false, true].into_iter().enumerate() {
        let words: Vec<[u32; 16]> = infos
            .iter()
            .zip(&all)
            .enumerate()
            .map(|(gi, (info, c))| {
                let gflags = if c.alpha { GF_CULL_DISABLE } else { GF_CULL_DISABLE | GF_OPAQUE };
                let omm = (with_omm && c.alpha).then(|| (mm_addr, omm_idx_buf.addr + u64::from(info.ibase / 3) * 4));
                pack_cluster(
                    gi as u32,
                    c.tris.len() as u32,
                    c.pos.len() as u32,
                    gflags,
                    idx_buf.addr + u64::from(info.ibase) * 4,
                    pos_buf.addr + u64::from(info.vbase) * 12,
                    omm,
                )
            })
            .collect();
        let clas_infos = gpu.upload(&words);
        let clas_addrs = gpu.buffer(8 * u64::from(n_clusters), Mem::Device, 16);
        let mut tri_input = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .max_geometry_index_value(n_clusters)
            .max_cluster_unique_geometry_count(1)
            .max_cluster_triangle_count(max_tris)
            .max_cluster_vertex_count(max_verts)
            .max_total_triangle_count(total_tris)
            .max_total_vertex_count(total_verts)
            .min_position_truncate_bit_count(0);
        let mut flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
        if with_omm {
            flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_CLUSTER_OPACITY_MICROMAPS_NV;
        }
        let cmd = gpu.begin();
        let clas_size = gpu.cluster_op(
            cmd,
            vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER,
            vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut tri_input },
            &clas_infos,
            64,
            n_clusters,
            flags,
            &clas_addrs,
        );
        // VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV: count, stride 8, references.
        let blas_infos: Vec<[u64; 2]> =
            ranges.iter().map(|&(first, count)| [u64::from(count) | (8u64 << 32), clas_addrs.addr + u64::from(first) * 8]).collect();
        let blas_infos_buf = gpu.upload(&blas_infos);
        let blas_addrs = gpu.buffer(8 * ranges.len() as u64, Mem::Readback, 16);
        let mut bl = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default()
            .max_total_cluster_count(n_clusters)
            .max_cluster_count_per_acceleration_structure(ranges.iter().map(|r| r.1).max().unwrap());
        let blas_size = gpu.cluster_op(
            cmd,
            vk::ClusterAccelerationStructureOpTypeNV::BUILD_CLUSTERS_BOTTOM_LEVEL,
            vk::ClusterAccelerationStructureOpInputNV { p_clusters_bottom_level: &mut bl },
            &blas_infos_buf,
            16,
            ranges.len() as u32,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            &blas_addrs,
        );
        gpu.submit_wait(cmd, "clas+blas");
        let blas: Vec<u64> = blas_addrs.read(ranges.len());
        assert!(blas.iter().all(|&a| a != 0), "cluster BLAS build wrote a null address");
        let instances: Vec<vk::AccelerationStructureInstanceKHR> = place
            .iter()
            .enumerate()
            .map(|(i, (proto, m))| vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: *m },
                // Custom index equals instance index, so either hit-object accessor is correct.
                instance_custom_index_and_mask: vk::Packed24_8::new(i as u32, 0xff),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: blas[*proto as usize] },
            })
            .collect();
        tlas[k] = gpu.build_tlas(&instances);
        sizes_note += &format!(" {}clas_kb={} blas_kb={}", if with_omm { "omm_" } else { "" }, clas_size / 1024, blas_size / 1024);
    }
    let insts: Vec<Inst> = place.iter().map(|&(proto, m)| Inst { m, proto, pad: [0; 3] }).collect();
    let instanced_tris: usize = insts.iter().map(|i| tri_total[i.proto as usize]).sum();
    let stats = format!(
        "clusters={n_clusters} proto_tris={tri_total:?} instances={} instanced_tris={:.1}M omm_level={omm_level} omm_microtris(transparent/opaque/unknown)={omm_hist:?} omm_bytes={}{sizes_note} build_ms={:.0}",
        insts.len(),
        instanced_tris as f64 / 1e6,
        omm_data.len(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    Scene { tlas_plain: tlas[0], tlas_omm: tlas[1], insts: gpu.upload(&insts), clusters: clusters_buf, pos: pos_buf, uv: uv_buf, idx: idx_buf, stats }
}

// ---------------------------------------------------------------------------------------------
// Texture and descriptor heaps.

struct Heaps {
    resource: vk::BindHeapInfoEXT<'static>,
    sampler: vk::BindHeapInfoEXT<'static>,
}

unsafe fn make_texture_and_heaps(gpu: &Gpu) -> Heaps {
    let format = vk::Format::R8G8B8A8_UNORM;
    let image = gpu
        .device
        .create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: ATLAS, height: ATLAS, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
            None,
        )
        .unwrap();
    let req = gpu.device.get_image_memory_requirements(image);
    let memory = gpu
        .device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(gpu.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap()),
            None,
        )
        .unwrap();
    gpu.device.bind_image_memory(image, memory, 0).unwrap();
    let staging = gpu.upload(&make_atlas());
    let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
    let cmd = gpu.begin();
    let to_general = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .image(image)
        .subresource_range(range)];
    gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_general));
    let region = vk::BufferImageCopy::default()
        .buffer_offset(staging.offset)
        .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
        .image_extent(vk::Extent3D { width: ATLAS, height: ATLAS, depth: 1 });
    gpu.device.cmd_copy_buffer_to_image(cmd, staging.buffer, image, vk::ImageLayout::GENERAL, &[region]);
    gpu.submit_wait(cmd, "atlas upload");

    let p = &gpu.heap_props;
    let image_size = p.image_descriptor_size;
    let reserved_offset = align_up(image_size, p.image_descriptor_alignment.max(p.buffer_descriptor_alignment));
    let resource_size = reserved_offset + p.min_resource_heap_reserved_range;
    let resource = gpu.buffer_usage(resource_size, Mem::Upload, p.resource_heap_alignment, vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT);
    let view = vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(range);
    let sampled = vk::ImageDescriptorInfoEXT::default().view(&view).layout(vk::ImageLayout::GENERAL);
    let infos = [vk::ResourceDescriptorInfoEXT::default().ty(vk::DescriptorType::SAMPLED_IMAGE).data(vk::ResourceDescriptorDataEXT { p_image: &sampled })];
    let dst = [vk::HostAddressRangeEXT::default().address(std::slice::from_raw_parts_mut(resource.ptr, image_size as usize))];
    gpu.heap.write_resource_descriptors(&infos, &dst).expect("vkWriteResourceDescriptorsEXT");

    let sampler_reserved = align_up(p.sampler_descriptor_size, p.sampler_descriptor_alignment.max(1));
    let sampler_size = sampler_reserved + p.min_sampler_heap_reserved_range;
    let sampler = gpu.buffer_usage(sampler_size, Mem::Upload, p.sampler_heap_alignment, vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT);
    let nearest = [vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)];
    let sdst = [vk::HostAddressRangeEXT::default().address(std::slice::from_raw_parts_mut(sampler.ptr, p.sampler_descriptor_size as usize))];
    gpu.heap.write_sampler_descriptors(&nearest, &sdst).expect("vkWriteSamplerDescriptorsEXT");

    let bind = |addr: u64, reserved_offset: u64, reserved: u64| vk::BindHeapInfoEXT {
        heap_range: vk::DeviceAddressRangeEXT { address: addr, size: reserved_offset + reserved },
        reserved_range_offset: reserved_offset,
        reserved_range_size: reserved,
        ..Default::default()
    };
    Heaps {
        resource: bind(resource.addr, reserved_offset, p.min_resource_heap_reserved_range),
        sampler: bind(sampler.addr, sampler_reserved, p.min_sampler_heap_reserved_range),
    }
}

// ---------------------------------------------------------------------------------------------
// Pipelines.

#[derive(Clone, Copy, PartialEq)]
enum RootMap {
    /// Push data holds the root address (orr's plan).
    PushAddress,
    /// Push data holds the root itself.
    PushData,
    /// Push data holds the address of a word holding the root address.
    Indirect,
}

fn root_map() -> RootMap {
    match std::env::var("PROBE_ROOTMAP").as_deref() {
        Ok("push_data") => RootMap::PushData,
        Ok("indirect") => RootMap::Indirect,
        _ => RootMap::PushAddress,
    }
}

fn root_mapping() -> vk::DescriptorSetAndBindingMappingEXT<'static> {
    let (source, source_data) = match root_map() {
        RootMap::PushAddress => (vk::DescriptorMappingSourceEXT::PUSH_ADDRESS, vk::DescriptorMappingSourceDataEXT { push_address_offset: 0 }),
        RootMap::PushData => (vk::DescriptorMappingSourceEXT::PUSH_DATA, vk::DescriptorMappingSourceDataEXT { push_data_offset: 0 }),
        RootMap::Indirect => (
            vk::DescriptorMappingSourceEXT::INDIRECT_ADDRESS,
            vk::DescriptorMappingSourceDataEXT {
                indirect_address: vk::DescriptorMappingSourceIndirectAddressEXT { push_offset: 0, address_offset: 0 },
            },
        ),
    };
    vk::DescriptorSetAndBindingMappingEXT {
        descriptor_set: 0,
        first_binding: 0,
        binding_count: 1,
        resource_mask: vk::SpirvResourceTypeFlagsEXT::UNIFORM_BUFFER | vk::SpirvResourceTypeFlagsEXT::READ_ONLY_STORAGE_BUFFER,
        source,
        source_data,
        ..Default::default()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Compute,
    /// Raygen only, one-record shader table.
    RayGen,
    /// Raygen + miss + triangle hit group with an any-hit shader.
    RayGenHit,
    /// Raygen + empty miss + triangle hit group with no shaders: the records a hit object
    /// recorded from a ray query refers to (index 0 in each table).
    RayGenNullHit,
}

struct Pipeline {
    pipeline: vk::Pipeline,
    kind: Kind,
    /// Classic layout with the root in push constants (no descriptor heap).
    classic: Option<vk::PipelineLayout>,
    /// Compute workgroup size.
    wg: (u32, u32),
    /// raygen, miss, hit regions.
    sbt: [vk::StridedDeviceAddressRegionKHR; 3],
    create_ms: f64,
}

unsafe fn make_pipeline(gpu: &Gpu, kind: Kind, code: &[u32], omm: bool, capture: bool, salt: u32) -> Pipeline {
    make_pipeline_in(gpu, kind, code, omm, capture, salt, None)
}

const CLASSIC_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::RAYGEN_KHR.as_raw()
        | vk::ShaderStageFlags::MISS_KHR.as_raw()
        | vk::ShaderStageFlags::ANY_HIT_KHR.as_raw()
        | vk::ShaderStageFlags::COMPUTE.as_raw(),
);

unsafe fn classic_layout(gpu: &Gpu) -> vk::PipelineLayout {
    let ranges = [vk::PushConstantRange::default().stage_flags(CLASSIC_STAGES).size(size_of::<Root>() as u32)];
    gpu.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges), None).unwrap()
}

unsafe fn make_pipeline_in(gpu: &Gpu, kind: Kind, code: &[u32], omm: bool, capture: bool, salt: u32, classic: Option<vk::PipelineLayout>) -> Pipeline {
    let mut flags2 = if classic.is_some() { vk::PipelineCreateFlags2::empty() } else { vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT };
    if capture {
        flags2 |= vk::PipelineCreateFlags2::CAPTURE_STATISTICS_KHR;
    }
    if omm && kind == Kind::RayGenHit {
        flags2 |= vk::PipelineCreateFlags2::RAY_TRACING_OPACITY_MICROMAP_EXT;
    }
    let mut flags_info = vk::PipelineCreateFlags2CreateInfo::default().flags(flags2);
    let mappings = [root_mapping()];
    let spec_entries = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
    let salt_bytes = salt.to_le_bytes();
    let spec = vk::SpecializationInfo::default().map_entries(&spec_entries).data(&salt_bytes);
    let (miss_code, anyhit_code) = if classic.is_some() {
        let anyhit = if std::env::var("PROBE_ANYHIT").as_deref() == Ok("root") { spv!("anyhit_push_root") } else { spv!("anyhit_push") };
        (spv!("miss_push"), anyhit)
    } else {
        // PROBE_ANYHIT selects a reduced any-hit for bisection: simple (no root, no texture),
        // root (root read, no texture); default is the full alpha test.
        let anyhit = match std::env::var("PROBE_ANYHIT").as_deref() {
            Ok("simple") => spv!("anyhit_simple"),
            Ok("root") => spv!("anyhit_root"),
            Ok("root0") => spv!("anyhit_root0"),
            Ok("geom") => spv!("anyhit_geom"),
            Ok("dump") => spv!("anyhit_dump"),
            Ok("check") => spv!("anyhit_check"),
            _ => spv!("anyhit"),
        };
        (spv!("miss"), anyhit)
    };
    let codes: Vec<&[u32]> = match kind {
        Kind::RayGenHit => vec![code, &miss_code, &anyhit_code],
        Kind::RayGenNullHit => vec![code, &miss_code],
        _ => vec![code],
    };
    let mut modules: Vec<_> = codes.iter().map(|c| vk::ShaderModuleCreateInfo::default().code(c)).collect();
    let mut maps: Vec<_> = codes.iter().map(|_| vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&mappings)).collect();
    let first = if kind == Kind::Compute {
        (vk::ShaderStageFlags::COMPUTE, c"compute_main")
    } else {
        (vk::ShaderStageFlags::RAYGEN_KHR, c"raygen_main")
    };
    let stage_info: [(vk::ShaderStageFlags, &CStr); 3] =
        [first, (vk::ShaderStageFlags::MISS_KHR, c"miss_main"), (vk::ShaderStageFlags::ANY_HIT_KHR, c"anyhit_main")];
    let stages: Vec<_> = modules
        .iter_mut()
        .zip(maps.iter_mut())
        .enumerate()
        .map(|(i, (m, mi))| {
            let st = vk::PipelineShaderStageCreateInfo::default().stage(stage_info[i].0).name(stage_info[i].1).specialization_info(&spec).push(m);
            if classic.is_some() { st } else { st.push(mi) }
        })
        .collect();
    let start = Instant::now();
    if kind == Kind::Compute {
        let info = vk::ComputePipelineCreateInfo::default().stage(stages[0]).layout(classic.unwrap_or_default()).push(&mut flags_info);
        let pipeline = gpu.device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e).expect("compute pipeline")[0];
        let create_ms = start.elapsed().as_secs_f64() * 1e3;
        return Pipeline { pipeline, kind, classic, wg: (8, 4), sbt: [vk::StridedDeviceAddressRegionKHR::default(); 3], create_ms };
    }
    let general = |i: u32| {
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(i)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
    };
    let mut groups = vec![general(0)];
    if kind != Kind::RayGen {
        groups.push(general(1));
        groups.push(
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(if kind == Kind::RayGenHit { 2 } else { vk::SHADER_UNUSED_KHR })
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        );
    }
    let info = vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(&stages)
        .groups(&groups)
        // The raygen-only kind never calls traceRay.
        .max_pipeline_ray_recursion_depth(u32::from(kind != Kind::RayGen))
        .layout(classic.unwrap_or_default())
        .push(&mut flags_info);
    let pipeline = gpu
        .rt
        .create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), vk::PipelineCache::null(), &[info], None)
        .map_err(|(_, e)| e)
        .expect("ray tracing pipeline")[0];
    let create_ms = start.elapsed().as_secs_f64() * 1e3;

    let rp = &gpu.rt_props;
    let handle = u64::from(rp.shader_group_handle_size);
    let base = u64::from(rp.shader_group_base_alignment);
    let stride = align_up(handle, u64::from(rp.shader_group_handle_alignment));
    let handles = gpu.rt.get_ray_tracing_shader_group_handles(pipeline, 0, groups.len() as u32, groups.len() * handle as usize).unwrap();
    let table = gpu.buffer(base * 3, Mem::Upload, base);
    for g in 0..groups.len() {
        table.write(g * base as usize, &handles[g * handle as usize..(g + 1) * handle as usize]);
    }
    let region = |g: u64, stride: u64| vk::StridedDeviceAddressRegionKHR { device_address: table.addr + g * base, stride, size: stride };
    let none = vk::StridedDeviceAddressRegionKHR::default();
    let mut sbt = [region(0, align_up(handle, base)), none, none];
    if kind != Kind::RayGen {
        sbt[1] = region(1, stride);
        sbt[2] = region(2, stride);
    }
    Pipeline { pipeline, kind, classic, wg: (0, 0), sbt, create_ms }
}

/// Pipeline-executable statistics, one line per executable.
unsafe fn exec_stats(gpu: &Gpu, pipeline: vk::Pipeline) -> Vec<String> {
    let Some(exec) = &gpu.exec else { return vec!["(no VK_KHR_pipeline_executable_properties)".into()] };
    let pinfo = vk::PipelineInfoKHR::default().pipeline(pipeline);
    let n = exec.get_pipeline_executable_properties_len(&pinfo).unwrap_or(0);
    if n == 0 {
        return vec!["(no executables reported)".into()];
    }
    let mut props = vec![vk::PipelineExecutablePropertiesKHR::default(); n];
    if exec.get_pipeline_executable_properties(&pinfo, &mut props).is_err() {
        return vec!["(executable properties failed)".into()];
    }
    let mut out = Vec::new();
    for (i, p) in props.iter().enumerate() {
        let einfo = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(i as u32);
        let m = exec.get_pipeline_executable_statistics_len(&einfo).unwrap_or(0);
        let mut stats = vec![vk::PipelineExecutableStatisticKHR::default(); m];
        let _ = exec.get_pipeline_executable_statistics(&einfo, &mut stats);
        let name = p.name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let mut line = format!("{name}:");
        for s in &stats {
            let sname = s.name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let v = match s.format {
                vk::PipelineExecutableStatisticFormatKHR::BOOL32 => format!("{}", s.value.b32),
                vk::PipelineExecutableStatisticFormatKHR::INT64 => format!("{}", s.value.i64),
                vk::PipelineExecutableStatisticFormatKHR::UINT64 => format!("{}", s.value.u64),
                _ => format!("{:.2}", s.value.f64),
            };
            line += &format!(" {sname}={v}");
        }
        out.push(line);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Variants, runs and comparison.

struct Variant {
    name: String,
    pipe: usize,
    omm: bool,
}

struct Ctx {
    gpu: Gpu,
    heaps: Heaps,
    scene: Scene,
    roots: Buf,
}

const ROOT_STRIDE: u64 = 256;

impl Ctx {
    unsafe fn root(&self, slot: u64, root: Root) -> u64 {
        std::ptr::write(self.roots.ptr.add((slot * ROOT_STRIDE) as usize).cast::<Root>(), root);
        self.roots.addr + slot * ROOT_STRIDE
    }

    fn base_root(&self, omm: bool, frame: u32, rad: u64, hash: u64, stats: u64) -> Root {
        let cam = camera();
        let sun = norm([0.35, 0.75, 0.45]);
        let s = &self.scene;
        Root {
            tlas: if omm { s.tlas_omm } else { s.tlas_plain },
            insts: s.insts.addr,
            clusters: s.clusters.addr,
            pos: s.pos.addr,
            uv: s.uv.addr,
            idx: s.idx.addr,
            out_rad: rad,
            out_hash: hash,
            out_stats: stats,
            width: W,
            height: H,
            frame,
            bounces: env_u32("PROBE_BOUNCES", 2),
            tex: 0,
            flags: env_u32("PROBE_MIX", 1) & 1,
            salt: 0,
            shade_scale: env_u32("PROBE_SHADE", 1),
            cam: [
                cam.pos[0],
                cam.pos[1],
                cam.pos[2],
                cam.tan_half,
                cam.fwd[0],
                cam.fwd[1],
                cam.fwd[2],
                W as f32 / H as f32,
                cam.right[0],
                cam.right[1],
                cam.right[2],
                0.0,
                cam.up[0],
                cam.up[1],
                cam.up[2],
                0.0,
            ],
            sun: [sun[0], sun[1], sun[2], 2.5],
        }
    }

    unsafe fn record(&self, cmd: vk::CommandBuffer, p: &Pipeline, root_addr: u64) {
        let gpu = &self.gpu;
        let root_bytes = std::slice::from_raw_parts(self.roots.ptr.add((root_addr - self.roots.addr) as usize), size_of::<Root>()).to_vec();
        let data = match root_map() {
            RootMap::PushAddress => root_addr.to_le_bytes().to_vec(),
            RootMap::PushData => root_bytes,
            RootMap::Indirect => {
                // The word after the root in its 256-byte slot holds the root address.
                let word = root_addr + 192;
                std::ptr::write(self.roots.ptr.add((word - self.roots.addr) as usize).cast::<u64>(), root_addr);
                word.to_le_bytes().to_vec()
            }
        };
        let push = vk::PushDataInfoEXT { offset: 0, data: vk::HostAddressRangeConstEXT::default().address(&data), ..Default::default() };
        if let Some(layout) = p.classic {
            let point = if p.kind == Kind::Compute { vk::PipelineBindPoint::COMPUTE } else { vk::PipelineBindPoint::RAY_TRACING_KHR };
            gpu.device.cmd_bind_pipeline(cmd, point, p.pipeline);
            let root = std::ptr::read(self.roots.ptr.add((root_addr - self.roots.addr) as usize).cast::<Root>());
            gpu.device.cmd_push_constants(cmd, layout, CLASSIC_STAGES, 0, bytes(std::slice::from_ref(&root)));
            if p.kind == Kind::Compute {
                gpu.device.cmd_dispatch(cmd, W.div_ceil(p.wg.0), H.div_ceil(p.wg.1), 1);
            } else {
                gpu.rt.cmd_trace_rays(cmd, &p.sbt[0], &p.sbt[1], &p.sbt[2], &vk::StridedDeviceAddressRegionKHR::default(), W, H, 1);
            }
            return;
        }
        if p.kind == Kind::Compute {
            gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            gpu.heap.cmd_push_data(cmd, &push);
            gpu.device.cmd_dispatch(cmd, W.div_ceil(p.wg.0), H.div_ceil(p.wg.1), 1);
        } else {
            gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, p.pipeline);
            gpu.heap.cmd_push_data(cmd, &push);
            let callable = vk::StridedDeviceAddressRegionKHR::default();
            gpu.rt.cmd_trace_rays(cmd, &p.sbt[0], &p.sbt[1], &p.sbt[2], &callable, W, H, 1);
        }
    }

    unsafe fn bind_heaps(&self, cmd: vk::CommandBuffer) {
        self.gpu.heap.cmd_bind_resource_heap(cmd, &self.heaps.resource);
        self.gpu.heap.cmd_bind_sampler_heap(cmd, &self.heaps.sampler);
    }

    /// One frame-0 dispatch into host-visible outputs; returns (radiance, hash, stats).
    unsafe fn check_run(&self, p: &Pipeline, omm: bool, outs: &(Buf, Buf, Buf), stats: bool) -> (Vec<[f32; 4]>, Vec<u32>, Vec<[u32; 2]>) {
        let (rad, hash, st) = outs;
        let root = self.root(0, self.base_root(omm, 0, rad.addr, hash.addr, st.addr));
        let cmd = self.gpu.begin();
        self.bind_heaps(cmd);
        self.record(cmd, p, root);
        self.gpu.submit_wait(cmd, "check");
        (rad.read(PIXELS), hash.read(PIXELS), if stats { st.read(PIXELS) } else { Vec::new() })
    }
}

struct Compare {
    hash_mismatch: usize,
    rad_mismatch: usize,
    max_rel: f32,
}

fn compare(ref_rad: &[[f32; 4]], ref_hash: &[u32], rad: &[[f32; 4]], hash: &[u32]) -> Compare {
    let mut c = Compare { hash_mismatch: 0, rad_mismatch: 0, max_rel: 0.0 };
    for i in 0..PIXELS {
        if ref_hash[i] != hash[i] {
            c.hash_mismatch += 1;
            continue;
        }
        let mut worst = 0.0f32;
        for k in 0..3 {
            let (a, b) = (ref_rad[i][k], rad[i][k]);
            let rel = (a - b).abs() / a.abs().max(1e-3);
            worst = worst.max(if rel.is_nan() { f32::INFINITY } else { rel });
        }
        c.max_rel = c.max_rel.max(worst);
        if worst > 1e-3 {
            c.rad_mismatch += 1;
        }
    }
    c
}

// ---------------------------------------------------------------------------------------------
// PNG (stored deflate) for a visual check of the reference frame.

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &b in data {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

fn write_png(path: &std::path::Path, w: u32, h: u32, rgb: &[u8]) {
    let mut raw = Vec::with_capacity((w * 3 + 1) as usize * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgb[y * w as usize * 3..(y + 1) * w as usize * 3]);
    }
    let mut z = vec![0x78, 0x01];
    let n_chunks = raw.len().div_ceil(65535);
    for (i, chunk) in raw.chunks(65535).enumerate() {
        z.push(u8::from(i + 1 == n_chunks));
        z.extend_from_slice(&(chunk.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(chunk.len() as u16)).to_le_bytes());
        z.extend_from_slice(chunk);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut chunk = |ty: &[u8], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut c = ty.to_vec();
        c.extend_from_slice(data);
        out.extend_from_slice(&c);
        out.extend_from_slice(&crc32(&c).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    std::fs::write(path, out).unwrap();
}

fn tonemap(rad: &[[f32; 4]]) -> Vec<u8> {
    rad.iter()
        .flat_map(|p| {
            [0, 1, 2].map(|k| {
                let x = p[k] / (1.0 + p[k]);
                (x.max(0.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------

unsafe fn info(gpu: &Gpu) {
    let props = gpu.instance.get_physical_device_properties(gpu.pdev);
    let v = props.driver_version;
    println!("device: {}", props.device_name_as_c_str().unwrap().to_string_lossy());
    println!(
        "driver: {}.{:02} api {}.{}.{}",
        v >> 22,
        (v >> 14) & 0xff,
        vk::api_version_major(props.api_version),
        vk::api_version_minor(props.api_version),
        vk::api_version_patch(props.api_version)
    );
    for name in REPORTED_EXTENSIONS {
        let spec = gpu.exts.iter().find(|e| e.extension_name_as_c_str() == Ok(*name)).map(|e| e.spec_version);
        println!("ext {:<42} {}", name.to_string_lossy(), spec.map_or("absent".to_owned(), |s| format!("rev {s}")));
    }
    let s = &gpu.ser_props;
    println!("ser: reordering_hint={:?} max_sbt_record_index={}", s.ray_tracing_invocation_reorder_reordering_hint, s.max_shader_binding_table_record_index);
    let r = &gpu.rt_props;
    println!(
        "rt: handle={} handle_align={} base_align={} max_recursion={} max_stride={} max_dispatch={}",
        r.shader_group_handle_size,
        r.shader_group_handle_alignment,
        r.shader_group_base_alignment,
        r.max_ray_recursion_depth,
        r.max_shader_group_stride,
        r.max_ray_dispatch_invocation_count
    );
    println!(
        "omm: max_2state_level={} max_4state_level={}",
        gpu.omm_props.max_opacity2_state_subdivision_level, gpu.omm_props.max_opacity4_state_subdivision_level
    );
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

unsafe fn run(only: Option<Vec<String>>) {
    let gpu = Gpu::new();
    info(&gpu);
    let grid = env_u32("PROBE_GRID", 64);
    let level = env_u32("PROBE_OMM_LEVEL", 6);
    let scene = build_scene(&gpu, grid, level);
    println!("scene: {}", scene.stats);
    let heaps = make_texture_and_heaps(&gpu);
    let roots = gpu.buffer(ROOT_STRIDE * 256, Mem::Upload, 256);
    let ctx = Ctx { gpu, heaps, scene, roots };
    let gpu = &ctx.gpu;

    // Pipelines: (name, kind, code, OMM pipeline flag). A per-run salt defeats the driver's
    // shader cache, so create times are cold compiles.
    let salt = 0x4000_0000 | (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() as u32 & 0x0fff_ffff);
    // PROBE_LIVE=1 replaces the variant set with A and Bk carrying 0..256 extra bytes of live
    // state across the coherence point (LIVE_WORDS in shaders/ser/pt.slang).
    let live = std::env::var("PROBE_LIVE").is_ok_and(|v| v == "1");
    let specs: Vec<(&str, Kind, Vec<u32>, bool)> = if live {
        vec![
            ("A", Kind::Compute, spv!("pt_a"), false),
            ("A_stats", Kind::Compute, spv!("pt_a_stats"), false),
            ("B0", Kind::RayGen, spv!("pt_b0"), false),
            ("Bk", Kind::RayGen, spv!("pt_bk"), false),
            ("A_L16", Kind::Compute, spv!("pt_a_live4"), false),
            ("Bk_L16", Kind::RayGen, spv!("pt_bk_live4"), false),
            ("A_L32", Kind::Compute, spv!("pt_a_live8"), false),
            ("Bk_L32", Kind::RayGen, spv!("pt_bk_live8"), false),
            ("A_L64", Kind::Compute, spv!("pt_a_live16"), false),
            ("Bk_L64", Kind::RayGen, spv!("pt_bk_live16"), false),
            ("A_L128", Kind::Compute, spv!("pt_a_live32"), false),
            ("Bk_L128", Kind::RayGen, spv!("pt_bk_live32"), false),
            ("A_L256", Kind::Compute, spv!("pt_a_live64"), false),
            ("Bk_L256", Kind::RayGen, spv!("pt_bk_live64"), false),
        ]
    } else {
        vec![
        ("A", Kind::Compute, spv!("pt_a"), false),
            ("A_stats", Kind::Compute, spv!("pt_a_stats"), false),
            ("A_8x8", Kind::Compute, spv!("pt_a_8x8"), false),
            ("A_16x8", Kind::Compute, spv!("pt_a_16x8"), false),
            ("A_16x16", Kind::Compute, spv!("pt_a_16x16"), false),
            ("A_32x1", Kind::Compute, spv!("pt_a_32x1"), false),
            ("B0", Kind::RayGen, spv!("pt_b0"), false),
            ("B", Kind::RayGenNullHit, spv!("pt_b"), false),
            ("Bh", Kind::RayGenNullHit, spv!("pt_bh"), false),
            ("EB", Kind::RayGenNullHit, spv!("pt_eb"), false),
            // Created only: dispatching B with an empty miss/hit table loses the device.
            ("B_emptysbt", Kind::RayGen, spv!("pt_b"), false),
            ("Bk", Kind::RayGen, spv!("pt_bk"), false),
            ("EBk", Kind::RayGen, spv!("pt_ebk"), false),
            ("C0", Kind::RayGenHit, spv!("pt_c0"), false),
            ("C", Kind::RayGenHit, spv!("pt_c"), false),
            ("Ch", Kind::RayGenHit, spv!("pt_ch"), false),
            ("EC", Kind::RayGenHit, spv!("pt_ec"), false),
            ("C0+ommflag", Kind::RayGenHit, spv!("pt_c0"), true),
            ("C+ommflag", Kind::RayGenHit, spv!("pt_c"), true),
            ("Ch+ommflag", Kind::RayGenHit, spv!("pt_ch"), true),
            ("EC+ommflag", Kind::RayGenHit, spv!("pt_ec"), true),
            ]
    };
    let mut pipes = Vec::new();
    println!("pipelines (cold create, salted spec constant):");
    for (i, (name, kind, code, omm)) in specs.iter().enumerate() {
        let mut p = make_pipeline(gpu, *kind, code, *omm, false, salt + i as u32);
        if let Some(wg) = name.strip_prefix("A_").and_then(|d| d.split_once('x')) {
            p.wg = (wg.0.parse().unwrap(), wg.1.parse().unwrap());
        }
        println!("  {name:<12} create_ms={:.1}", p.create_ms);
        pipes.push(p);
    }
    for (i, (name, kind, code, omm)) in specs.iter().enumerate().filter(|(_, s)| !live && ["A", "B0", "Bk", "C0", "C"].contains(&s.0)) {
        let p = make_pipeline(gpu, *kind, code, *omm, false, salt + i as u32);
        println!("  {name:<12} recreate_ms={:.1} (same salt)", p.create_ms);
    }
    if std::env::var("PROBE_EXEC_STATS").is_ok_and(|v| v == "1") {
        for (i, (name, kind, code, omm)) in specs.iter().enumerate() {
            let p = make_pipeline(gpu, *kind, code, *omm, true, salt + 1000 + i as u32);
            for line in exec_stats(gpu, p.pipeline) {
                println!("  stats {name:<12} {line}");
            }
        }
    }
    let pipe = |name: &str| specs.iter().position(|s| s.0 == name).unwrap();

    let mut variants: Vec<Variant> = Vec::new();
    if live {
        let omm = std::env::var("PROBE_LIVE_OMM").is_ok_and(|v| v == "1");
        for name in ["A", "Bk", "A_L16", "Bk_L16", "A_L32", "Bk_L32", "A_L64", "Bk_L64", "A_L128", "Bk_L128", "A_L256", "Bk_L256"] {
            variants.push(Variant { name: name.into(), pipe: pipe(name), omm });
        }
    }
    for omm in [false, true].into_iter().filter(|_| !live) {
        for base in ["A", "B0", "B", "Bh", "EB", "Bk", "EBk", "C0", "C", "Ch", "EC"] {
            let pname = if omm && base.starts_with('C') || omm && base == "EC" { format!("{base}+ommflag") } else { base.to_owned() };
            let name = if omm { format!("{base}_omm") } else { base.to_owned() };
            variants.push(Variant { name, pipe: pipe(&pname), omm });
        }
    }
    // Compute workgroup shapes (plain scene), only when asked for by name.
    if only.is_some() {
        for wg in ["A_8x8", "A_16x8", "A_16x16", "A_32x1"] {
            variants.push(Variant { name: wg.into(), pipe: pipe(wg), omm: false });
        }
    }
    // A pipeline carrying the OMM flag on the plain scene: cost of the flag itself.
    if !live {
        variants.push(Variant { name: "C_ommflag_plain".into(), pipe: pipe("C+ommflag"), omm: false });
    }
    if let Some(only) = &only {
        variants.retain(|v| only.iter().any(|o| o == &v.name) || v.name == "A");
    }

    // Correctness: frame 0 of every variant against A on the plain scene.
    let outs = (
        gpu.buffer(PIXELS as u64 * 16, Mem::Readback, 16),
        gpu.buffer(PIXELS as u64 * 4, Mem::Readback, 16),
        gpu.buffer(PIXELS as u64 * 8, Mem::Readback, 16),
    );
    let (ref_rad, ref_hash, _) = ctx.check_run(&pipes[pipe("A")], false, &outs, false);
    let exe_dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    write_png(&exe_dir.join("ser_A.png"), W, H, &tonemap(&ref_rad));
    let sky = ref_hash.iter().filter(|&&h| h == hash_after_miss()).count();
    println!("reference A: png={} primary_sky_pixels={sky}", exe_dir.join("ser_A.png").display());
    for omm in [false, true] {
        let (_, _, st) = ctx.check_run(&pipes[pipe("A_stats")], omm, &outs, true);
        let tests: u64 = st.iter().map(|s| u64::from(s[0])).sum();
        let segs: u64 = st.iter().map(|s| u64::from(s[1])).sum();
        println!(
            "stats {}: alpha_tests={:.1}M ({:.2}/pixel) path_segments={:.2}M ({:.2}/pixel)",
            if omm { "omm  " } else { "plain" },
            tests as f64 / 1e6,
            tests as f64 / PIXELS as f64,
            segs as f64 / 1e6,
            segs as f64 / PIXELS as f64
        );
    }
    let (rg_rad, rg_hash, _) = ctx.check_run(&pipes[pipe("B0")], false, &outs, false);
    println!("check columns: vs A (compute) | vs B0 (raygen, ray query, no reorder)");
    for v in &variants {
        let (rad, hash, _) = ctx.check_run(&pipes[v.pipe], v.omm, &outs, false);
        let c = compare(&ref_rad, &ref_hash, &rad, &hash);
        let r = compare(&rg_rad, &rg_hash, &rad, &hash);
        println!(
            "check {:<16} hash_mismatch={} rad_mismatch={} max_rel={:.2e} | hash_mismatch={} rad_mismatch={} max_rel={:.2e}",
            v.name, c.hash_mismatch, c.rad_mismatch, c.max_rel, r.hash_mismatch, r.rad_mismatch, r.max_rel
        );
        if c.hash_mismatch > PIXELS / 100 {
            write_png(&exe_dir.join(format!("ser_{}.png", v.name)), W, H, &tonemap(&rad));
        }
    }

    // Timing. At full load the GPU sits on its board power limit and the clock follows the power
    // of whatever runs, so back-to-back dispatches of different variants slow each other. Each
    // sample is therefore its own submission: one warm dispatch of the variant, then the timed
    // dispatch, then PROBE_GAP_MS of idle so average power stays under the limit and the clock
    // stays at boost. Variants rotate round-robin; a PROBE_WARMUP_MS warm-up runs first.
    let rounds = env_u32("PROBE_ROUNDS", 40);
    let gap = std::time::Duration::from_millis(u64::from(env_u32("PROBE_GAP_MS", 6)));
    let n = variants.len() as u32;
    let out_rad = gpu.buffer(PIXELS as u64 * 16, Mem::Device, 16);
    let out_hash = gpu.buffer(PIXELS as u64 * 4, Mem::Device, 16);
    for f in 0..2 {
        ctx.root(1 + f, ctx.base_root(false, f as u32 + 1, out_rad.addr, out_hash.addr, 0));
        ctx.root(101 + f, ctx.base_root(true, f as u32 + 1, out_rad.addr, out_hash.addr, 0));
    }
    let warm_until = Instant::now() + std::time::Duration::from_millis(u64::from(env_u32("PROBE_WARMUP_MS", 2000)));
    while Instant::now() < warm_until {
        let cmd = gpu.begin();
        ctx.bind_heaps(cmd);
        ctx.record(cmd, &pipes[variants[0].pipe], ctx.roots.addr + ROOT_STRIDE);
        gpu.submit_wait(cmd, "warm-up");
        std::thread::sleep(gap);
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    // Time over A's time in the same round.
    let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let mut firsts: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    for round in 0..rounds {
        let mut round_times = vec![0.0; variants.len()];
        for k in 0..n {
            let vi = ((k + round * 7) % n) as usize;
            let v = &variants[vi];
            let base_slot = if v.omm { 101 } else { 1 };
            let cmd = gpu.begin();
            gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 0, 3);
            ctx.bind_heaps(cmd);
            gpu.timestamp(cmd, 0);
            ctx.record(cmd, &pipes[v.pipe], ctx.roots.addr + base_slot * ROOT_STRIDE);
            gpu.full_barrier(cmd);
            gpu.timestamp(cmd, 1);
            ctx.record(cmd, &pipes[v.pipe], ctx.roots.addr + (base_slot + 1) * ROOT_STRIDE);
            gpu.full_barrier(cmd);
            gpu.timestamp(cmd, 2);
            gpu.submit_wait(cmd, "timing");
            let t = gpu.read_timestamps(3);
            firsts[vi].push(t[0]);
            samples[vi].push(t[1]);
            round_times[vi] = t[1];
            std::thread::sleep(gap);
        }
        for vi in 0..variants.len() {
            ratios[vi].push(round_times[vi] / round_times[0]);
        }
    }
    let base = sorted(&samples[0])[samples[0].len() / 2];
    println!("timing: {rounds} samples per variant (warm dispatch + timed dispatch per submission, {gap:?} idle gaps), us per frame");
    println!(
        "  {:<16} {:>9} {:>9} {:>9} {:>7} {:>22} {:>9}",
        "variant", "median", "p10", "p90", "vs A", "per-round vs A (p10-p90)", "warm-up"
    );
    for (vi, v) in variants.iter().enumerate() {
        let s = sorted(&samples[vi]);
        let r = sorted(&ratios[vi]);
        let m = s[s.len() / 2];
        let f = sorted(&firsts[vi]);
        println!(
            "  {:<16} {:>9.0} {:>9.0} {:>9.0} {:>6.2}x {:>8.3} ({:.3}-{:.3}) {:>9.0}",
            v.name,
            m,
            s[s.len() / 10],
            s[s.len() * 9 / 10],
            m / base,
            r[r.len() / 2],
            r[r.len() / 10],
            r[r.len() * 9 / 10],
            f[f.len() / 2]
        );
    }
    let (e, w) = dump_messages();
    println!("validation errors={e} warnings={w}");
}

const MINI_NAMES: [&str; 15] = [
    "nop+reorder",
    "reorder_hint",
    "trace_opaque",
    "trace_anyhit",
    "trace_attr",
    "rq_from_query",
    "rq_from_query+reorder_key",
    "trace+reorder",
    "rq_only",
    "rq_from_query+reorder",
    "rq_from_query+hint_only",
    "glsl_from_query+reorder",
    "trace_geometry_index",
    "rq_geometry_index",
    "rq_from_query_geometry_index",
];

/// One minimal hit-object case (`mini.slang`, MINI = n) in heap or classic mode.
unsafe fn mini(n: usize, classic: bool) {
    let gpu = Gpu::new();
    let scene = build_scene(&gpu, env_u32("PROBE_GRID", 64), env_u32("PROBE_OMM_LEVEL", 6));
    let heaps = make_texture_and_heaps(&gpu);
    let roots = gpu.buffer(ROOT_STRIDE * 4, Mem::Upload, 256);
    let ctx = Ctx { gpu, heaps, scene, roots };
    let gpu = &ctx.gpu;
    let codes: [Vec<u32>; 15] = if classic {
        [
            spv!("mini1_push"),
            spv!("mini2_push"),
            spv!("mini3_push"),
            spv!("mini4_push"),
            spv!("mini5_push"),
            spv!("mini6_push"),
            spv!("mini7_push"),
            spv!("mini8_push"),
            spv!("mini9_push"),
            spv!("mini10_push"),
            spv!("mini11_push"),
            spv!("glsl_mini7_push"),
            spv!("mini13_push"),
            spv!("mini14_push"),
            spv!("mini15_push"),
        ]
    } else {
        [
            spv!("mini1"),
            spv!("mini2"),
            spv!("mini3"),
            spv!("mini4"),
            spv!("mini5"),
            spv!("mini6"),
            spv!("mini7"),
            spv!("mini8"),
            spv!("mini9"),
            spv!("mini10"),
            spv!("mini11"),
            spv!("glsl_mini7_push"),
            spv!("mini13"),
            spv!("mini14"),
            spv!("mini15"),
        ]
    };
    // PROBE_MINI_HIT=1 gives the ray-query cases a miss record and an empty hit group.
    let with_hit = std::env::var("PROBE_MINI_HIT").is_ok_and(|v| v == "1");
    let kind = if [3, 4, 5, 8, 13].contains(&n) {
        Kind::RayGenHit
    } else if with_hit {
        Kind::RayGenNullHit
    } else {
        Kind::RayGen
    };
    let layout = classic.then(|| classic_layout(gpu));
    let p = make_pipeline_in(gpu, kind, &codes[n - 1], false, false, 7, layout);
    let hash = gpu.buffer(PIXELS as u64 * 4, Mem::Readback, 16);
    let dump = gpu.buffer(PIXELS as u64 * 16, Mem::Readback, 16);
    let root = ctx.root(0, ctx.base_root(false, 0, 0, hash.addr, dump.addr));
    let cmd = gpu.begin();
    if !classic {
        ctx.bind_heaps(cmd);
    }
    ctx.record(cmd, &p, root);
    gpu.submit_wait(cmd, "mini");
    let v: Vec<u32> = hash.read(PIXELS);
    let nonzero = v.iter().filter(|&&x| x != 0).count();
    if [13, 14, 15].contains(&n) {
        let over = v.iter().filter(|&&x| x > 138).count();
        let max = v.iter().max().unwrap();
        let sum: u64 = v.iter().map(|&x| u64::from(x & 0xfff)).sum();
        println!("geometry index: hits={nonzero} above_cluster_range={over} max={max:#x} low12_checksum={sum}");
    }
    if std::env::var("PROBE_ANYHIT").as_deref() == Ok("check") {
        let d: Vec<[u32; 4]> = dump.read(PIXELS);
        let seen: Vec<&[u32; 4]> = d.iter().filter(|r| r[3] == 0x600d).collect();
        let count = |bit: u32| seen.iter().filter(|r| r[1] & bit != 0).count();
        println!(
            "anyhit check: candidates={} point_matches(full/low12/high)={}/{}/{}",
            seen.len(),
            count(1),
            count(2),
            count(4)
        );
    }
    if std::env::var("PROBE_ANYHIT").as_deref() == Ok("dump") {
        // Any-hit view (geometry, primitive, instance) of the last candidate per pixel, against
        // the cluster-range ground truth: geometry indices are 0..clusters.
        let d: Vec<[u32; 4]> = dump.read(PIXELS);
        let seen: Vec<&[u32; 4]> = d.iter().filter(|r| r[3] == 0x600d).collect();
        let mut geoms = BTreeMap::new();
        for r in &seen {
            *geoms.entry(r[0]).or_insert(0u32) += 1;
        }
        let in_range = seen.iter().filter(|r| r[0] < 138).count();
        let mut top: Vec<_> = geoms.into_iter().collect();
        top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        top.truncate(12);
        println!("anyhit dump: pixels_with_candidates={} geometry_in_range={in_range} top_geometry={top:?}", seen.len());
        let samples: Vec<_> = seen.iter().step_by(seen.len().max(1) / 8 + 1).map(|r| (r[0], r[0] & 0xff_ffff, r[1], r[2])).collect();
        println!("anyhit samples (geom, geom&0xffffff, prim, inst): {samples:?}");
    }
    let (e, w) = dump_messages();
    println!("RESULT mini{n} ok nonzero={nonzero} (validation errors={e} warnings={w})");
}

/// Path hash of a pixel whose primary ray misses.
fn hash_after_miss() -> u32 {
    (2166136261u32 ^ 0xffff_fffe).wrapping_mul(16777619)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("info") => unsafe { info(&Gpu::new()) },
        Some("mini") if args.len() > 2 => unsafe { mini(args[2].parse().unwrap(), args.get(3).is_some_and(|m| m == "classic")) },
        Some("mini") => {
            let exe = std::env::current_exe().unwrap();
            let list = std::env::var("PROBE_MINI").unwrap_or_else(|_| "1,2,3,4,5,6,7,8,9,10,11,12".into());
            let cases: Vec<usize> = list.split(',').map(|c| c.parse().unwrap()).collect();
            for classic in [false, true] {
                for &n in &cases {
                    let mode = if classic { "classic" } else { "heap" };
                    if n == 12 && !classic {
                        continue;
                    }
                    let out = std::process::Command::new(&exe).args(["mini", &n.to_string(), mode]).output().unwrap();
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let line = stdout
                        .lines()
                        .find(|l| l.starts_with("RESULT") || l.starts_with("FATAL"))
                        .map_or_else(|| format!("CRASH exit={:?}", out.status.code()), str::to_owned);
                    println!("mini{n} {:<22} {mode:<8} {line}", MINI_NAMES[n - 1]);
                    for l in stdout.lines().filter(|l| l.starts_with("VVL")) {
                        println!("    {l}");
                    }
                }
            }
        }
        Some("only") => unsafe { run(Some(args.get(2).expect("variant list").split(',').map(str::to_owned).collect())) },
        None => unsafe { run(None) },
        Some(other) => panic!("unknown command {other}; use info, only <variants>, or no argument"),
    }
}
