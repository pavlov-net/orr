//! Device creation and the state every object shares.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::{CStr, CString, c_char};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::accel::{EmptyTlas, build_empty_tlas};
use super::debug::{MessageCount, messenger_info, name_object};
use super::features::{self, Selection};
use super::garbage::{Garbage, Item, MAX_TIMELINES};
use super::heap::Heaps;
use super::memory::{MemoryKind, MemoryTypes, Usage, buffer_usage, sharing};
use super::queue::{Calibration, PassResult, Queue, QueueKind, QueueRequest, Timeline};
use super::ray::RecordBlock;
use super::texture::{image_info, vk_format};
use crate::{
    Caps, ComponentType, CooperativeVectorCombination, Error, Format, Limits, TextureDesc,
    TextureUsage, TimelinePoint, Tlas,
};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// How to create a [`Device`].
#[derive(Clone, Copy, Debug)]
pub struct DeviceDesc<'a> {
    /// Application name reported to the driver.
    pub name: &'a str,
    pub queues: QueueRequest,
    /// Loads the Khronos validation layer with synchronisation checks, when installed.
    /// Messages are logged and counted ([`Device::validation_messages`]).
    pub validation: bool,
    /// Resource-heap slots for sampled and storage image views.
    pub image_heap_slots: u32,
    pub sampler_heap_slots: u32,
}

impl Default for DeviceDesc<'_> {
    fn default() -> Self {
        Self {
            name: "orr",
            queues: QueueRequest::default(),
            validation: false,
            image_heap_slots: 262_144,
            sampler_heap_slots: 1024,
        }
    }
}

/// State every object of one device shares. Objects hold it through an `Arc`, so it is
/// destroyed after the last of them.
pub(crate) struct Shared {
    // Keeps the loader library loaded.
    _entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub heap_fn: ash::ext::descriptor_heap::Device,
    pub accel_fn: ash::khr::acceleration_structure::Device,
    pub rt_fn: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub debug_fn: ash::ext::debug_utils::Device,
    pub calibrated_fn: Option<ash::khr::calibrated_timestamps::Device>,
    pub exec_fn: Option<ash::khr::pipeline_executable_properties::Device>,
    messenger: (ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT),
    messages: Box<MessageCount>,
    pub caps: Caps,
    pub heap_props: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    pub as_scratch_align: u32,
    pub memory: MemoryTypes,
    /// Distinct queue families of the created queues; buffers and images share across them.
    pub families: Vec<u32>,
    pub timelines: [Option<Timeline>; MAX_TIMELINES],
    heaps: OnceLock<Heaps>,
    /// Raygen shader-table records; with `Caps::ray_dispatch` only.
    pub records: OnceLock<RecordBlock>,
    empty_tlas: OnceLock<EmptyTlas>,
    /// `gpu_malloc` buffers by start address: end address and buffer.
    pub buffers: RwLock<BTreeMap<u64, (u64, vk::Buffer)>>,
    pub usage: Usage,
    /// Usage of every `gpu_malloc` buffer.
    pub buffer_usage: vk::BufferUsageFlags,
    pub garbage: Mutex<Garbage>,
    pub results: Mutex<VecDeque<(TimelinePoint, Vec<PassResult>)>>,
    pub calibration: Mutex<Option<Calibration>>,
}

impl Shared {
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
            if let Some(t) = self.empty_tlas.take() {
                self.accel_fn.destroy_acceleration_structure(t.accel, None);
                self.destroy(Item::Buffer(t.storage));
            }
            if let Some(r) = self.records.take() {
                self.destroy(Item::Buffer(r.raw));
            }
            if let Some(h) = self.heaps.take() {
                self.destroy(Item::Buffer(h.resource.raw));
                self.destroy(Item::Buffer(h.sampler.raw));
            }
            for t in self.timelines.iter().flatten() {
                self.device.destroy_semaphore(t.semaphore, None);
            }
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

fn component(t: vk::ComponentTypeKHR) -> ComponentType {
    use vk::ComponentTypeKHR as C;
    match t {
        C::FLOAT16 => ComponentType::F16,
        C::FLOAT32 => ComponentType::F32,
        C::FLOAT64 => ComponentType::F64,
        C::SINT8 => ComponentType::I8,
        C::SINT16 => ComponentType::I16,
        C::SINT32 => ComponentType::I32,
        C::SINT64 => ComponentType::I64,
        C::UINT8 => ComponentType::U8,
        C::UINT16 => ComponentType::U16,
        C::UINT32 => ComponentType::U32,
        C::UINT64 => ComponentType::U64,
        C::FLOAT8_E4M3_EXT => ComponentType::E4m3,
        C::FLOAT8_E5M2_EXT => ComponentType::E5m2,
        other => ComponentType::Other(other.as_raw()),
    }
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
        let picked = pick_device(&instance);
        let (physical, selection) = match picked {
            Ok(p) => p,
            Err(e) => {
                destroy_instance();
                return Err(e);
            }
        };
        let created = create_device(&instance, physical, desc, selection);
        let (device, plans, optional) = match created {
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
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push(&mut heap_props)
            .push(&mut as_props)
            .push(&mut driver);
        if optional.ray_dispatch {
            props2 = props2.push(&mut rt_props).push(&mut reorder_props);
        }
        // SAFETY: every struct in the chain belongs to an enabled extension or core 1.4.
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        let props = props2.properties;
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
        let memory = select_memory(&device, memory_props, &family_list, usage)
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
            reorder: optional.ray_dispatch
                && reorder_props.ray_tracing_invocation_reorder_reordering_hint
                    == vk::RayTracingInvocationReorderModeEXT::REORDER,
            host_image_copy: optional.host_image_copy,
            memory_decompression: optional.memory_decompression,
            omm: optional.omm,
            device_address_commands: optional.device_address_commands,
            unified_image_layouts: optional.unified_image_layouts,
            memory_budget: optional.memory_budget,
            shader_abort: optional.shader_abort,
            pipeline_binary: optional.pipeline_binary,
            cooperative_vector: coop,
            shader_float8: optional.shader_float8,
            calibrated_timestamps: optional.calibrated_timestamps,
            frame_boundary: optional.frame_boundary,
            pipeline_executable_info: optional.executable_info,
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
                max_ray_dispatch_invocations: rt_props.max_ray_dispatch_invocation_count,
                device_local_bytes: memory.heap_size(MemoryKind::DeviceLocal),
                upload_bytes: memory.heap_size(MemoryKind::Upload),
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

        let shared = Arc::new(Shared {
            heap_fn: ash::ext::descriptor_heap::Device::load(&instance, &device),
            accel_fn: ash::khr::acceleration_structure::Device::load(&instance, &device),
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
            messenger: (debug_instance, messenger),
            messages,
            caps,
            heap_props,
            as_scratch_align: as_props.min_acceleration_structure_scratch_offset_alignment,
            memory,
            families: family_list,
            timelines,
            heaps: OnceLock::new(),
            records: OnceLock::new(),
            empty_tlas: OnceLock::new(),
            buffers: RwLock::default(),
            usage: Usage::default(),
            buffer_usage: usage,
            garbage: Mutex::default(),
            results: Mutex::default(),
            calibration: Mutex::default(),
            _entry: entry,
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
        let heaps = Heaps::new(&shared, desc.image_heap_slots, desc.sampler_heap_slots)?;
        let _ = shared.heaps.set(heaps);
        if optional.ray_dispatch {
            let _ = shared.records.set(RecordBlock::new(&shared, &rt_props)?);
        }
        let device = Device {
            shared,
            queues: Arc::new(queues),
        };
        let tlas = build_empty_tlas(&device)?;
        let _ = device.shared.empty_tlas.set(tlas);
        log::info!(
            "device {} ({}): rebar {}, ray dispatch {} (reorder {}), omm {}, host image copy {}, decompression {}, validation layer {}",
            device.shared.caps.device_name,
            device.shared.caps.driver,
            rebar,
            device.shared.caps.ray_dispatch,
            device.shared.caps.reorder,
            optional.omm,
            optional.host_image_copy,
            optional.memory_decompression,
            desc.validation,
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

    /// Validation-layer warnings and errors reported so far.
    pub fn validation_messages(&self) -> u32 {
        self.shared.messages.get()
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

fn pick_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, Selection), Error> {
    // SAFETY: plain query.
    let mut devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(Error::vk("vkEnumeratePhysicalDevices"))?;
    // SAFETY: each handle came from this instance.
    let props = |p: vk::PhysicalDevice| unsafe { instance.get_physical_device_properties(p) };
    devices.retain(|&p| props(p).api_version >= vk::API_VERSION_1_4);
    devices.sort_by_key(|&p| props(p).device_type != vk::PhysicalDeviceType::DISCRETE_GPU);
    let mut first_failure = None;
    for p in devices {
        match features::select(instance, p) {
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
) -> Result<(ash::Device, Vec<QueuePlan>, features::Optional), Error> {
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
    let ext_names: Vec<_> = selection.extensions.iter().map(|n| n.as_ptr()).collect();
    let mut info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_names);
    info.p_next = selection.enable.head();
    // SAFETY: the feature chain lives in `selection` until the call returns.
    let device = unsafe { instance.create_device(physical, &info, None) }
        .map_err(Error::vk("vkCreateDevice"))?;
    Ok((device, plans, selection.optional))
}

/// Memory requirements of the universal buffer and of optimal images decide the types.
fn select_memory(
    device: &ash::Device,
    props: vk::PhysicalDeviceMemoryProperties,
    families: &[u32],
    usage: vk::BufferUsageFlags,
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
        let info = image_info(&TextureDesc::d2(format, 256, 256, usage), families);
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
    MemoryTypes::select(props, buffer_bits, image_bits)
}
