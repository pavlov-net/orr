//! Device creation and the state every object shares.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::{CStr, CString, c_char};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::accel::{EmptyTlas, build_empty_tlas};
use super::cache::PipelineCache;
use super::coopvec::component;
use super::debug::{MessageCount, messenger_info, name_object};
use super::diagnostics::{Breadcrumbs, Diagnostics, DiagnosticsReport, FaultFns, Trail};
use super::features::{self, SURFACE_INSTANCE_EXTENSIONS, Selection};
use super::garbage::{Garbage, Item, MAX_TIMELINES};
use super::heap::Heaps;
use super::interop::{ClassicBind, DeviceHooks};
use super::memory::{MemoryKind, MemoryTypes, Pools, Range, Usage, buffer_usage, sharing};
use super::queue::{Calibration, PassResult, Queue, QueueKind, QueueRequest, Timeline};
use super::ray::RecordBlock;
use super::rtas::{RtasCheck, cluster_caps};
use super::texture::{image_info, vk_format};
use crate::{
    Caps, CooperativeVectorCombination, Error, Format, Limits, TextureDesc, TextureUsage,
    TimelinePoint, Tlas,
};
use orr_memory::AllocatorDesc;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The driver exposes `VK_NV_ray_tracing_validation` only with this set to 1.
const RAY_TRACING_VALIDATION_VAR: &str = "NV_ALLOW_RAYTRACING_VALIDATION";

/// How to create a [`Device`].
#[derive(Clone, Copy, Debug)]
pub struct DeviceDesc<'a> {
    /// Application name reported to the driver.
    pub name: &'a str,
    /// Extensions and features a library that records raw Vulkan commands needs.
    pub hooks: Option<&'a dyn DeviceHooks>,
    pub queues: QueueRequest,
    /// Loads the Khronos validation layer with synchronisation checks, when installed.
    /// Messages are logged and counted ([`Device::validation_messages`]).
    pub validation: bool,
    pub diagnostics: Diagnostics<'a>,
    /// The disk pipeline cache: read at creation, written by
    /// [`Device::save_pipeline_cache`]. `None`: no disk cache.
    pub pipeline_cache: Option<&'a Path>,
    /// Optional extensions to leave off even when the device offers them; their `Caps`
    /// stay false.
    pub skip_extensions: &'a [&'a CStr],
    /// Reports `Caps::rebar` and `Caps::host_image_copy` false and puts
    /// [`MemoryKind::Upload`] in host memory whatever the device offers, so callers take the
    /// staging path, as on a device without resizable BAR.
    pub force_staging: bool,
    /// Resource-heap slots for sampled and storage image views.
    pub image_heap_slots: u32,
    pub sampler_heap_slots: u32,
}

impl Default for DeviceDesc<'_> {
    fn default() -> Self {
        Self {
            name: "orr",
            hooks: None,
            queues: QueueRequest::default(),
            validation: false,
            diagnostics: Diagnostics::Off,
            pipeline_cache: None,
            skip_extensions: &[],
            force_staging: false,
            image_heap_slots: 262_144,
            sampler_heap_slots: 1024,
        }
    }
}

/// State every object of one device shares. Objects hold it through an `Arc`, so it is
/// destroyed after the last of them.
pub(crate) struct Shared {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub heap_fn: ash::ext::descriptor_heap::Device,
    pub accel_fn: ash::khr::acceleration_structure::Device,
    pub cluster_fn: ash::nv::cluster_acceleration_structure::Device,
    pub ptlas_fn: ash::nv::partitioned_acceleration_structure::Device,
    /// With `Caps::device_address_commands`.
    pub dac_fn: Option<ash::khr::device_address_commands::Device>,
    /// With `Caps::memory_decompression`.
    pub decompress_fn: Option<ash::ext::memory_decompression::Device>,
    /// With cooperative vectors (`Caps::cooperative_vector` not empty).
    pub coop_fn: Option<ash::nv::cooperative_vector::Device>,
    /// With `Caps::occupancy_priority`.
    pub occupancy_fn: Option<ash::nv::compute_occupancy_priority::Device>,
    /// With `Caps::generated_commands`.
    pub generated_fn: Option<ash::ext::device_generated_commands::Device>,
    pub mesh_fn: ash::ext::mesh_shader::Device,
    pub rt_fn: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub debug_fn: ash::ext::debug_utils::Device,
    pub calibrated_fn: Option<ash::khr::calibrated_timestamps::Device>,
    pub exec_fn: Option<ash::khr::pipeline_executable_properties::Device>,
    /// On devices with the Present queue.
    pub present_fn: Option<PresentFns>,
    messenger: (ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT),
    messages: Box<MessageCount>,
    pub caps: Caps,
    pub pipeline_cache: PipelineCache,
    pub heap_props: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    pub as_scratch_align: u32,
    pub memory: MemoryTypes,
    /// The allocator and its pools; dropped in `Drop` before the device.
    pub pools: ManuallyDrop<Pools>,
    /// Distinct queue families of the created queues; buffers and images share across them.
    pub families: Vec<u32>,
    pub timelines: [Option<Timeline>; MAX_TIMELINES],
    heaps: OnceLock<Heaps>,
    /// Raygen shader-table records; with `Caps::ray_dispatch` only.
    pub records: OnceLock<RecordBlock>,
    empty_tlas: OnceLock<EmptyTlas>,
    /// The debug PTLAS check, created by the first PTLAS build of a debug build.
    pub rtas_check: OnceLock<RtasCheck>,
    /// `gpu_malloc` and sparse buffers by start address.
    pub buffers: RwLock<BTreeMap<u64, Range>>,
    /// The most recently freed ranges, oldest first.
    pub freed: Mutex<VecDeque<(u64, Range)>>,
    pub usage: Usage,
    /// Usage of every `gpu_malloc` buffer.
    pub buffer_usage: vk::BufferUsageFlags,
    pub garbage: Mutex<Garbage>,
    pub results: Mutex<VecDeque<(TimelinePoint, Vec<PassResult>)>>,
    pub calibration: Mutex<Option<Calibration>>,
    pub checkpoint_fn: Option<ash::nv::device_diagnostic_checkpoints::Device>,
    pub fault_fn: Option<FaultFns>,
    /// With `Caps::breadcrumbs`.
    pub breadcrumbs: OnceLock<Breadcrumbs>,
    pub trail: Mutex<Trail>,
    /// The report made when device loss was first seen.
    pub lost: OnceLock<DiagnosticsReport>,
    /// The raw queue of each created kind.
    pub queue_handles: [Option<vk::Queue>; MAX_TIMELINES],
    /// The family of each created kind's queue.
    pub families_by_kind: [u32; MAX_TIMELINES],
    /// Every enabled device extension.
    pub extensions: Vec<CString>,
    classic_bind: OnceLock<ClassicBind>,
}

/// Surface and swapchain entry points of a device that presents.
pub(crate) struct PresentFns {
    pub surface: ash::khr::surface::Instance,
    pub win32: ash::khr::win32_surface::Instance,
    pub caps2: ash::khr::get_surface_capabilities2::Instance,
    pub swapchain: ash::khr::swapchain::Device,
    pub wait2: Option<ash::khr::present_wait2::Device>,
    pub latency: Option<ash::nv::low_latency2::Device>,
    pub hdr: Option<ash::ext::hdr_metadata::Device>,
    pub timing: Option<ash::ext::present_timing::Device>,
}

impl Shared {
    pub(crate) fn classic_bind(&self) -> &ClassicBind {
        self.classic_bind
            .get()
            .expect("the classic bind exists once the device is created")
    }

    pub(crate) fn heaps(&self) -> &Heaps {
        self.heaps
            .get()
            .expect("heaps exist once the device is created")
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: nothing else uses the device any more; every handle below belongs to it
        // and is destroyed once.
        unsafe {
            let _ = self.device.device_wait_idle();
            for item in self.garbage.get_mut().unwrap().take_all() {
                self.destroy(item);
            }
            if let Some(c) = self.rtas_check.take() {
                self.device.destroy_pipeline(c.pipeline, None);
                self.destroy(Item::Buffer(c.failures));
            }
            if let Some(t) = self.empty_tlas.take() {
                self.accel_fn.destroy_acceleration_structure(t.accel, None);
                self.destroy(Item::Buffer(t.storage));
            }
            if let Some(r) = self.records.take() {
                self.destroy(Item::Buffer(r.raw));
            }
            if let Some(b) = self.breadcrumbs.take() {
                self.destroy(Item::Buffer(b.raw));
            }
            if let Some(c) = self.classic_bind.take() {
                c.destroy(&self.device);
            }
            if let Some(h) = self.heaps.take() {
                self.destroy(Item::Buffer(h.resource.raw));
                self.destroy(Item::Buffer(h.sampler.raw));
            }
            for t in self.timelines.iter().flatten() {
                self.device.destroy_semaphore(t.semaphore, None);
            }
            self.pipeline_cache.destroy(&self.device);
            ManuallyDrop::drop(&mut self.pools);
            self.device.destroy_device(None);
            self.messenger
                .0
                .destroy_debug_utils_messenger(self.messenger.1, None);
            self.instance.destroy_instance(None);
        }
    }
}

/// The GPU: memory, heaps, pipelines and queues. Cheap to clone; every clone is the same
/// device.
#[derive(Clone)]
pub struct Device {
    pub(crate) shared: Arc<Shared>,
    queues: Arc<[Option<Queue>; MAX_TIMELINES]>,
}

/// One queue to create: its kind, family and index in the family.
struct QueuePlan {
    kind: QueueKind,
    family: u32,
    index: u32,
}

fn plan_queues(
    props: &[vk::QueueFamilyProperties],
    request: QueueRequest,
) -> Result<Vec<QueuePlan>, String> {
    let find = |want: vk::QueueFlags, avoid: vk::QueueFlags| {
        props
            .iter()
            .position(|p| p.queue_flags.contains(want) && !p.queue_flags.intersects(avoid))
            .map(|i| i as u32)
    };
    let render = find(
        vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
        vk::QueueFlags::empty(),
    )
    .ok_or("a graphics and compute queue family")?;
    let compute = find(vk::QueueFlags::COMPUTE, vk::QueueFlags::GRAPHICS).unwrap_or(render);
    let transfer = find(
        vk::QueueFlags::TRANSFER,
        vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
    )
    .unwrap_or(compute);
    let mut used = vec![0u32; props.len()];
    let mut plans = Vec::new();
    let mut take = |kind: QueueKind, family: u32| -> Result<(), String> {
        let index = used[family as usize];
        if index >= props[family as usize].queue_count {
            return Err(format!("a free queue in family {family} for {kind:?}"));
        }
        used[family as usize] += 1;
        plans.push(QueuePlan {
            kind,
            family,
            index,
        });
        Ok(())
    };
    take(QueueKind::Render, render)?;
    if request.present {
        take(QueueKind::Present, render)?;
    }
    if request.async_compute {
        take(QueueKind::Loading, compute)?;
    }
    if request.render_compute {
        take(QueueKind::Compute, compute)?;
    }
    if request.copy {
        take(QueueKind::Copy, transfer)?;
    }
    Ok(plans)
}

/// Execution sets of compute and of mesh pipelines (task, mesh and pixel stages) can be
/// bound and run per sequence.
fn generated_stages(p: &vk::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT<'_>) -> bool {
    use vk::ShaderStageFlags as S;
    let want = S::COMPUTE | S::TASK_EXT | S::MESH_EXT | S::FRAGMENT;
    p.supported_indirect_commands_shader_stages.contains(want)
        && p.supported_indirect_commands_shader_stages_pipeline_binding
            .contains(want)
}

fn c_string(chars: &[c_char]) -> String {
    // SAFETY: Vulkan property strings are NUL-terminated within their arrays.
    unsafe { CStr::from_ptr(chars.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl Device {
    /// Creates the device: the first Vulkan 1.4 GPU (discrete first) with every required
    /// feature of contracts.md 7. Fails naming the first missing item.
    pub fn new(desc: &DeviceDesc<'_>) -> Result<Device, Error> {
        if cfg!(debug_assertions) && std::env::var_os(RAY_TRACING_VALIDATION_VAR).is_none() {
            // SAFETY: the driver reads the variable at instance creation. Setting it races
            // only with other threads touching the environment, which Windows serialises;
            // elsewhere debug builds create the device before such threads start.
            unsafe { std::env::set_var(RAY_TRACING_VALIDATION_VAR, "1") };
        }
        if let Diagnostics::Aftermath { dump_dir } = desc.diagnostics {
            super::aftermath::enable(dump_dir, desc.name).map_err(Error::Aftermath)?;
        }
        // SAFETY: loading the system Vulkan loader.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| Error::Loader(e.to_string()))?;
        let messages = Box::<MessageCount>::default();
        let instance = create_instance(&entry, desc, &messages)?;
        let debug_instance = ash::ext::debug_utils::Instance::load(&entry, &instance);
        // SAFETY: the count outlives the messenger (both live in `Shared`).
        let messenger = unsafe {
            debug_instance.create_debug_utils_messenger(&messenger_info(&messages), None)
        };
        let messenger = match messenger {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: nothing else uses the instance.
                unsafe { instance.destroy_instance(None) };
                return Err(Error::vk("vkCreateDebugUtilsMessengerEXT")(e));
            }
        };
        let destroy_instance = || {
            // SAFETY: nothing else uses the instance or messenger.
            unsafe {
                debug_instance.destroy_debug_utils_messenger(messenger, None);
                instance.destroy_instance(None);
            }
        };
        let picked = pick_device(
            &instance,
            desc.skip_extensions,
            desc.queues.present,
            desc.diagnostics,
        );
        let (physical, selection) = match picked {
            Ok(p) => p,
            Err(e) => {
                destroy_instance();
                return Err(e);
            }
        };
        let created = create_device(&instance, physical, desc, selection);
        let (device, plans, optional, extensions) = match created {
            Ok(c) => c,
            Err(e) => {
                destroy_instance();
                return Err(e);
            }
        };

        // SAFETY: the physical device belongs to the instance.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let mut heap_props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
        let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut reorder_props =
            vk::PhysicalDeviceRayTracingInvocationReorderPropertiesEXT::default();
        let mut cluster_props =
            vk::PhysicalDeviceClusterAccelerationStructurePropertiesNV::default();
        let mut ptlas_props =
            vk::PhysicalDevicePartitionedAccelerationStructurePropertiesNV::default();
        let mut decompress_props = vk::PhysicalDeviceMemoryDecompressionPropertiesEXT::default();
        let mut generated_props = vk::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push(&mut heap_props)
            .push(&mut as_props)
            .push(&mut driver)
            .push(&mut cluster_props)
            .push(&mut ptlas_props);
        if optional.ray_dispatch {
            props2 = props2.push(&mut rt_props).push(&mut reorder_props);
        }
        if optional.memory_decompression {
            props2 = props2.push(&mut decompress_props);
        }
        if optional.generated_commands {
            props2 = props2.push(&mut generated_props);
        }
        // SAFETY: every struct in the chain belongs to an enabled extension or core 1.4.
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        let props = props2.properties;
        let generated = optional.generated_commands && generated_stages(&generated_props);
        heap_props.p_next = std::ptr::null_mut();
        rt_props.p_next = std::ptr::null_mut();
        let device_name = c_string(&props.device_name);

        let coop = if optional.cooperative_vector {
            let f = ash::nv::cooperative_vector::Instance::load(&entry, &instance);
            // SAFETY: the extension is enabled on this physical device.
            let n = unsafe { f.get_physical_device_cooperative_vector_properties_len(physical) }
                .unwrap_or(0);
            let mut out = vec![vk::CooperativeVectorPropertiesNV::default(); n];
            // SAFETY: as above, with room for `n` entries.
            let _ =
                unsafe { f.get_physical_device_cooperative_vector_properties(physical, &mut out) };
            out.iter()
                .map(|p| CooperativeVectorCombination {
                    input: component(p.input_type),
                    input_interpretation: component(p.input_interpretation),
                    matrix_interpretation: component(p.matrix_interpretation),
                    bias_interpretation: component(p.bias_interpretation),
                    result: component(p.result_type),
                    transpose: p.transpose != 0,
                })
                .collect()
        } else {
            Vec::new()
        };

        // SAFETY: the physical device belongs to the instance.
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let mut family_list: Vec<u32> = plans.iter().map(|p| p.family).collect();
        family_list.sort_unstable();
        family_list.dedup();
        let usage = buffer_usage(optional.ray_dispatch);
        let memory = select_memory(
            &device,
            memory_props,
            &family_list,
            usage,
            !desc.force_staging,
        )
        .map_err(|m| features::unsupported(&device_name, m));
        let (memory, rebar) = match memory {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: nothing else uses the device.
                unsafe { device.destroy_device(None) };
                destroy_instance();
                return Err(e);
            }
        };

        let p = &heap_props;
        let caps = Caps {
            device_name: device_name.clone(),
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            driver: format!(
                "{} {}",
                c_string(&driver.driver_name),
                c_string(&driver.driver_info)
            ),
            driver_version: props.driver_version,
            rebar,
            ray_dispatch: optional.ray_dispatch,
            lss: optional.lss,
            reorder: optional.ray_dispatch
                && reorder_props.ray_tracing_invocation_reorder_reordering_hint
                    == vk::RayTracingInvocationReorderModeEXT::REORDER,
            host_image_copy: optional.host_image_copy && rebar,
            memory_decompression: optional.memory_decompression,
            occupancy_priority: optional.occupancy_priority,
            generated_commands: generated,
            omm: optional.omm,
            device_address_commands: optional.device_address_commands,
            unified_image_layouts: optional.unified_image_layouts,
            memory_budget: optional.memory_budget,
            memory_priority: optional.memory_priority,
            shader_abort: optional.shader_abort,
            device_fault: optional.device_fault || optional.device_fault_ext,
            breadcrumbs: optional.buffer_marker,
            checkpoints: optional.checkpoints,
            aftermath: optional.aftermath,
            pipeline_binary: optional.pipeline_binary,
            cooperative_vector: coop,
            shader_float8: optional.shader_float8,
            calibrated_timestamps: optional.calibrated_timestamps,
            frame_boundary: optional.frame_boundary,
            pipeline_executable_info: optional.executable_info,
            conservative_raster: optional.conservative_raster,
            ray_tracing_validation: optional.ray_tracing_validation,
            present_wait: optional.present_wait,
            low_latency: optional.low_latency,
            present_metering: optional.present_metering,
            present_timing: optional.present_timing,
            hdr_metadata: optional.hdr_metadata,
            cluster: cluster_caps(&cluster_props, &ptlas_props, optional.omm),
            limits: Limits {
                image_descriptor_bytes: p.image_descriptor_size as u32,
                sampler_descriptor_bytes: p.sampler_descriptor_size as u32,
                max_image_slots: ((p.max_resource_heap_size - p.min_resource_heap_reserved_range)
                    / p.image_descriptor_size) as u32,
                max_sampler_slots: ((p.max_sampler_heap_size - p.min_sampler_heap_reserved_range)
                    / p.sampler_descriptor_size) as u32,
                max_push_data_bytes: p.max_push_data_size as u32,
                timestamp_period_ns: props.limits.timestamp_period,
                max_workgroup_size: props.limits.max_compute_work_group_size,
                max_workgroup_invocations: props.limits.max_compute_work_group_invocations,
                max_workgroup_count: props.limits.max_compute_work_group_count,
                max_image_extent_2d: props.limits.max_image_dimension2_d,
                max_image_extent_3d: props.limits.max_image_dimension3_d,
                max_image_layers: props.limits.max_image_array_layers,
                max_color_targets: props.limits.max_color_attachments,
                max_ray_dispatch_invocations: rt_props.max_ray_dispatch_invocation_count,
                device_local_bytes: memory.heap_size(MemoryKind::DeviceLocal),
                upload_bytes: memory.heap_size(MemoryKind::Upload),
                max_decompress_regions: if optional.memory_decompression {
                    decompress_props
                        .max_decompression_indirect_count
                        .min(u64::from(u32::MAX)) as u32
                } else {
                    0
                },
                max_generated_pipelines: if generated {
                    generated_props.max_indirect_pipeline_count
                } else {
                    0
                },
                max_generated_sequences: if generated {
                    generated_props.max_indirect_sequence_count
                } else {
                    0
                },
            },
        };

        let mut timelines: [Option<Timeline>; MAX_TIMELINES] = Default::default();
        for plan in &plans {
            let mut ty = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            // SAFETY: valid create info.
            let semaphore = unsafe {
                device.create_semaphore(&vk::SemaphoreCreateInfo::default().push(&mut ty), None)
            };
            let semaphore = match semaphore {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: nothing else uses the device or the semaphores made so far.
                    unsafe {
                        for t in timelines.iter().flatten() {
                            device.destroy_semaphore(t.semaphore, None);
                        }
                        device.destroy_device(None);
                    }
                    destroy_instance();
                    return Err(Error::vk("vkCreateSemaphore")(e));
                }
            };
            timelines[plan.kind.index()] = Some(Timeline {
                semaphore,
                submitted: AtomicU64::new(0),
            });
        }

        let allocator = AllocatorDesc {
            entry: &entry,
            instance: &instance,
            physical,
            device: &device,
            api_version: vk::API_VERSION_1_4,
            memory_budget: optional.memory_budget,
            memory_priority: optional.memory_priority,
        };
        let placement = placement_align(
            &device,
            &family_list,
            usage,
            props.limits.buffer_image_granularity,
        );
        // SAFETY: the handles are live, buffer device address and the named extensions are
        // enabled, and `Shared` drops the pools before the device.
        let pools = match unsafe { Pools::new(&allocator, &memory, placement) } {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: nothing else uses the device or the semaphores.
                unsafe {
                    for t in timelines.iter().flatten() {
                        device.destroy_semaphore(t.semaphore, None);
                    }
                    device.destroy_device(None);
                }
                destroy_instance();
                return Err(e);
            }
        };

        let pipeline_cache =
            PipelineCache::open(&instance, &device, physical, &caps, desc.pipeline_cache);
        let mut queue_handles = [None; MAX_TIMELINES];
        let mut families_by_kind = [0; MAX_TIMELINES];
        for plan in &plans {
            // SAFETY: the queue was requested at device creation.
            queue_handles[plan.kind.index()] =
                Some(unsafe { device.get_device_queue(plan.family, plan.index) });
            families_by_kind[plan.kind.index()] = plan.family;
        }
        let fault_fn = if optional.device_fault {
            Some(FaultFns::Khr(ash::khr::device_fault::Device::load(
                &instance, &device,
            )))
        } else if optional.device_fault_ext {
            Some(FaultFns::Ext(ash::ext::device_fault::Device::load(
                &instance, &device,
            )))
        } else {
            None
        };
        let shared = Arc::new(Shared {
            heap_fn: ash::ext::descriptor_heap::Device::load(&instance, &device),
            accel_fn: ash::khr::acceleration_structure::Device::load(&instance, &device),
            cluster_fn: ash::nv::cluster_acceleration_structure::Device::load(&instance, &device),
            ptlas_fn: ash::nv::partitioned_acceleration_structure::Device::load(&instance, &device),
            dac_fn: optional
                .device_address_commands
                .then(|| ash::khr::device_address_commands::Device::load(&instance, &device)),
            decompress_fn: optional
                .memory_decompression
                .then(|| ash::ext::memory_decompression::Device::load(&instance, &device)),
            coop_fn: optional
                .cooperative_vector
                .then(|| ash::nv::cooperative_vector::Device::load(&instance, &device)),
            occupancy_fn: optional
                .occupancy_priority
                .then(|| ash::nv::compute_occupancy_priority::Device::load(&instance, &device)),
            generated_fn: generated
                .then(|| ash::ext::device_generated_commands::Device::load(&instance, &device)),
            mesh_fn: ash::ext::mesh_shader::Device::load(&instance, &device),
            rt_fn: optional
                .ray_dispatch
                .then(|| ash::khr::ray_tracing_pipeline::Device::load(&instance, &device)),
            debug_fn: ash::ext::debug_utils::Device::load(&instance, &device),
            calibrated_fn: optional
                .calibrated_timestamps
                .then(|| ash::khr::calibrated_timestamps::Device::load(&instance, &device)),
            exec_fn: optional.executable_info.then(|| {
                ash::khr::pipeline_executable_properties::Device::load(&instance, &device)
            }),
            present_fn: optional.surface.then(|| PresentFns {
                surface: ash::khr::surface::Instance::load(&entry, &instance),
                win32: ash::khr::win32_surface::Instance::load(&entry, &instance),
                caps2: ash::khr::get_surface_capabilities2::Instance::load(&entry, &instance),
                swapchain: ash::khr::swapchain::Device::load(&instance, &device),
                wait2: optional
                    .present_wait
                    .then(|| ash::khr::present_wait2::Device::load(&instance, &device)),
                latency: optional
                    .low_latency
                    .then(|| ash::nv::low_latency2::Device::load(&instance, &device)),
                hdr: optional
                    .hdr_metadata
                    .then(|| ash::ext::hdr_metadata::Device::load(&instance, &device)),
                timing: optional
                    .present_timing
                    .then(|| ash::ext::present_timing::Device::load(&instance, &device)),
            }),
            messenger: (debug_instance, messenger),
            messages,
            caps,
            pipeline_cache,
            heap_props,
            as_scratch_align: as_props.min_acceleration_structure_scratch_offset_alignment,
            memory,
            pools: ManuallyDrop::new(pools),
            families: family_list,
            timelines,
            heaps: OnceLock::new(),
            records: OnceLock::new(),
            empty_tlas: OnceLock::new(),
            rtas_check: OnceLock::new(),
            buffers: RwLock::default(),
            freed: Mutex::default(),
            usage: Usage::default(),
            buffer_usage: usage,
            garbage: Mutex::default(),
            results: Mutex::default(),
            calibration: Mutex::default(),
            checkpoint_fn: optional
                .checkpoints
                .then(|| ash::nv::device_diagnostic_checkpoints::Device::load(&instance, &device)),
            fault_fn,
            breadcrumbs: OnceLock::new(),
            trail: Mutex::default(),
            lost: OnceLock::new(),
            queue_handles,
            families_by_kind,
            extensions,
            classic_bind: OnceLock::new(),
            entry,
            instance,
            physical,
            device,
        });
        for (i, t) in shared.timelines.iter().enumerate() {
            if let Some(t) = t {
                name_object(
                    &shared.debug_fn,
                    t.semaphore,
                    &format!("orr {:?} timeline", QueueKind::ALL[i]),
                );
            }
        }
        let mut queues: [Option<Queue>; MAX_TIMELINES] = Default::default();
        for plan in &plans {
            let q = Queue::new(
                Arc::clone(&shared),
                plan.kind,
                plan.family,
                plan.index,
                &families[plan.family as usize],
            );
            queues[plan.kind.index()] = Some(q);
        }
        if optional.buffer_marker {
            let marker_fn = ash::amd::buffer_marker::Device::load(&shared.instance, &shared.device);
            let _ = shared
                .breadcrumbs
                .set(Breadcrumbs::new(&shared, marker_fn)?);
        }
        if optional.aftermath {
            super::aftermath::add_device(Arc::downgrade(&shared));
        } else if matches!(desc.diagnostics, Diagnostics::Aftermath { .. }) {
            log::warn!(
                "Aftermath asked for, but the device lacks diagnostics config or checkpoints"
            );
        }
        let heaps = Heaps::new(&shared, desc.image_heap_slots, desc.sampler_heap_slots)?;
        let _ = shared.heaps.set(heaps);
        let _ = shared.classic_bind.set(ClassicBind::new(&shared.device)?);
        if optional.ray_dispatch {
            let _ = shared.records.set(RecordBlock::new(&shared, &rt_props)?);
        }
        let device = Device {
            shared,
            queues: Arc::new(queues),
        };
        let tlas = build_empty_tlas(&device)?;
        let _ = device.shared.empty_tlas.set(tlas);
        if let Some(hooks) = desc.hooks {
            hooks.device_created(&device.shared.raw_device());
        }
        log::info!(
            "device {} ({}): rebar {}, ray dispatch {} (reorder {}), omm {}, lss {}, address commands {}, cooperative vector combinations {}, host image copy {}, decompression {}, occupancy priority {}, memory priority {} (pageable {}), generated commands {}, validation layer {}, ray tracing validation {}, diagnostics {:?} (breadcrumbs {}, checkpoints {}, aftermath {}, device fault {})",
            device.shared.caps.device_name,
            device.shared.caps.driver,
            device.shared.caps.rebar,
            device.shared.caps.ray_dispatch,
            device.shared.caps.reorder,
            optional.omm,
            optional.lss,
            optional.device_address_commands,
            device.shared.caps.cooperative_vector.len(),
            device.shared.caps.host_image_copy,
            optional.memory_decompression,
            optional.occupancy_priority,
            optional.memory_priority,
            optional.pageable_memory,
            device.shared.caps.generated_commands,
            desc.validation,
            optional.ray_tracing_validation,
            desc.diagnostics,
            optional.buffer_marker,
            optional.checkpoints,
            optional.aftermath,
            device.shared.caps.device_fault,
        );
        Ok(device)
    }

    pub fn caps(&self) -> &Caps {
        &self.shared.caps
    }

    /// The queue of `kind`, if the device created one.
    pub fn queue(&self, kind: QueueKind) -> Option<&Queue> {
        self.queues[kind.index()].as_ref()
    }

    /// A TLAS with one inactive instance: every ray misses it. For kernels with nothing to
    /// trace; `Tlas(0)` is never traced.
    pub fn empty_tlas(&self) -> Tlas {
        Tlas(
            self.shared
                .empty_tlas
                .get()
                .expect("built in Device::new")
                .address,
        )
    }

    /// Validation-layer and ray-tracing-validation warnings and errors reported so far, and
    /// PTLAS builds the debug check rejected among completed work.
    pub fn validation_messages(&self) -> u32 {
        let rejected = self.shared.rtas_check.get().map_or(0, RtasCheck::failures);
        self.shared.messages.get() + rejected
    }
}

fn create_instance(
    entry: &ash::Entry,
    desc: &DeviceDesc<'_>,
    messages: &MessageCount,
) -> Result<ash::Instance, Error> {
    // SAFETY: plain queries.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
    let has_layer = layers
        .iter()
        .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
    let validation = desc.validation && has_layer;
    if desc.validation && !has_layer {
        log::warn!("validation requested but {VALIDATION_LAYER:?} is not installed");
    }
    // SAFETY: plain query.
    let exts = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(Error::vk("vkEnumerateInstanceExtensionProperties"))?;
    if !exts
        .iter()
        .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME))
    {
        return Err(Error::Unsupported {
            device: "instance".into(),
            missing: "VK_EXT_debug_utils".into(),
        });
    }
    let mut ext_names = vec![ash::ext::debug_utils::NAME.as_ptr()];
    let layer_names = if validation {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };
    if validation {
        ext_names.push(ash::ext::layer_settings::NAME.as_ptr());
    }
    if desc.queues.present {
        for name in SURFACE_INSTANCE_EXTENSIONS {
            if !exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(*name))
            {
                return Err(Error::Unsupported {
                    device: "instance".into(),
                    missing: name.to_string_lossy().into_owned(),
                });
            }
            ext_names.push(name.as_ptr());
        }
    }
    let hooked = desc
        .hooks
        .map(|h| h.instance_extensions(entry))
        .unwrap_or_default();
    for name in &hooked {
        if !exts
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(name.as_c_str()))
        {
            return Err(Error::Unsupported {
                device: "instance".into(),
                missing: name.to_string_lossy().into_owned(),
            });
        }
        // SAFETY: every entry points at a NUL-terminated name.
        let listed = |&p: &*const c_char| unsafe { CStr::from_ptr(p) } == name.as_c_str();
        if !ext_names.iter().any(listed) {
            ext_names.push(name.as_ptr());
        }
    }
    let name = CString::new(desc.name).unwrap_or_default();
    let app = vk::ApplicationInfo::default()
        .application_name(&name)
        .engine_name(c"orr")
        .api_version(vk::API_VERSION_1_4);
    let on = [vk::TRUE];
    let settings = [vk::LayerSettingEXT::default()
        .layer_name(VALIDATION_LAYER)
        .setting_name(c"validate_sync")
        .ty(vk::LayerSettingTypeEXT::BOOL32)
        .values(&on)];
    let mut layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
    let mut messenger = messenger_info(messages);
    let mut info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_layer_names(&layer_names)
        .enabled_extension_names(&ext_names)
        .push(&mut messenger);
    if validation {
        info = info.push(&mut layer_settings);
    }
    // SAFETY: valid create info; the messenger's count outlives the instance.
    unsafe { entry.create_instance(&info, None) }.map_err(Error::vk("vkCreateInstance"))
}

fn pick_device(
    instance: &ash::Instance,
    skip: &[&CStr],
    surface: bool,
    diagnostics: Diagnostics<'_>,
) -> Result<(vk::PhysicalDevice, Selection), Error> {
    // SAFETY: plain query.
    let mut devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(Error::vk("vkEnumeratePhysicalDevices"))?;
    // SAFETY: each handle came from this instance.
    let props = |p: vk::PhysicalDevice| unsafe { instance.get_physical_device_properties(p) };
    devices.retain(|&p| props(p).api_version >= vk::API_VERSION_1_4);
    devices.sort_by_key(|&p| props(p).device_type != vk::PhysicalDeviceType::DISCRETE_GPU);
    let mut first_failure = None;
    for p in devices {
        match features::select(instance, p, skip, surface, diagnostics) {
            Ok(selection) => return Ok((p, selection)),
            Err(missing) => {
                first_failure.get_or_insert_with(|| {
                    features::unsupported(&c_string(&props(p).device_name), missing)
                });
            }
        }
    }
    Err(first_failure.unwrap_or(Error::NoDevice))
}

fn create_device(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    desc: &DeviceDesc<'_>,
    mut selection: Selection,
) -> Result<
    (
        ash::Device,
        Vec<QueuePlan>,
        features::Optional,
        Vec<CString>,
    ),
    Error,
> {
    // SAFETY: the physical device belongs to the instance.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    // SAFETY: as above.
    let name = c_string(&unsafe { instance.get_physical_device_properties(physical) }.device_name);
    let plans = plan_queues(&families, desc.queues).map_err(|m| features::unsupported(&name, m))?;
    let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
    for p in &plans {
        *counts.entry(p.family).or_default() += 1;
    }
    let priorities = [1.0f32; 16];
    let queue_infos: Vec<_> = counts
        .iter()
        .map(|(&family, &n)| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities[..n as usize])
        })
        .collect();
    let mut extensions: Vec<CString> = selection.extensions.iter().map(|&n| n.to_owned()).collect();
    let _hooked_features = match desc.hooks {
        Some(hooks) => {
            for ext in hooks.device_extensions(instance, physical) {
                if !selection.offers(&ext) {
                    return Err(features::unsupported(
                        &name,
                        ext.to_string_lossy().into_owned(),
                    ));
                }
                if !extensions.contains(&ext) {
                    extensions.push(ext);
                }
            }
            selection.enable.hook(hooks, physical)
        }
        None => Vec::new(),
    };
    let ext_names: Vec<_> = extensions.iter().map(|n| n.as_ptr()).collect();
    let mut info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_names);
    info.p_next = selection.enable.head();
    // SAFETY: the feature chain lives in `selection` and `_hooked_features` until the call
    // returns.
    let device = unsafe { instance.create_device(physical, &info, None) }
        .map_err(Error::vk("vkCreateDevice"))?;
    Ok((device, plans, selection.optional, extensions))
}

/// The memory-offset alignment and size granularity of large DeviceLocal and Upload blocks:
/// at least 64 KiB, the standard sparse tile size, at least a sparse buffer's page, and at
/// least `bufferImageGranularity`, so an image placed or a page bound inside a block never
/// shares a granule with a neighboring buffer.
fn placement_align(
    device: &ash::Device,
    families: &[u32],
    usage: vk::BufferUsageFlags,
    granularity: u64,
) -> u64 {
    let (sharing, family_list) = sharing(families);
    let info = vk::BufferCreateInfo::default()
        .size(1 << 20)
        .usage(usage)
        .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY)
        .sharing_mode(sharing)
        .queue_family_indices(family_list);
    let mut out = vk::MemoryRequirements2::default();
    // SAFETY: valid create info; sparse binding and residency are required features.
    unsafe {
        device.get_device_buffer_memory_requirements(
            &vk::DeviceBufferMemoryRequirements::default().create_info(&info),
            &mut out,
        );
    };
    out.memory_requirements
        .alignment
        .max(granularity)
        .max(64 << 10)
        .next_power_of_two()
}

/// Memory requirements of the universal buffer and of optimal images decide the types.
fn select_memory(
    device: &ash::Device,
    props: vk::PhysicalDeviceMemoryProperties,
    families: &[u32],
    usage: vk::BufferUsageFlags,
    allow_rebar: bool,
) -> Result<(MemoryTypes, bool), String> {
    let (sharing, family_list) = sharing(families);
    let buffer = vk::BufferCreateInfo::default()
        .size(65536)
        .usage(usage)
        .sharing_mode(sharing)
        .queue_family_indices(family_list);
    let mut out = vk::MemoryRequirements2::default();
    // SAFETY: valid create info.
    unsafe {
        device.get_device_buffer_memory_requirements(
            &vk::DeviceBufferMemoryRequirements::default().create_info(&buffer),
            &mut out,
        );
    };
    let buffer_bits = out.memory_requirements.memory_type_bits;
    let mut image_bits = u32::MAX;
    for (format, usage) in [
        (
            Format::Rgba8Unorm,
            TextureUsage::SAMPLED | TextureUsage::STORAGE | TextureUsage::COLOR,
        ),
        (
            Format::D32Float,
            TextureUsage::SAMPLED | TextureUsage::DEPTH,
        ),
        (Format::Bc7Unorm, TextureUsage::SAMPLED),
    ] {
        let info = image_info(&TextureDesc::d2(format, 256, 256, usage), false, families);
        debug_assert_eq!(info.format, vk_format(format));
        let mut out = vk::MemoryRequirements2::default();
        // SAFETY: valid create info.
        unsafe {
            device.get_device_image_memory_requirements(
                &vk::DeviceImageMemoryRequirements::default().create_info(&info),
                &mut out,
            );
        };
        image_bits &= out.memory_requirements.memory_type_bits;
    }
    MemoryTypes::select(props, buffer_bits, image_bits, allow_rebar)
}
