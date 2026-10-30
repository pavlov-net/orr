//! Device with the three queues of contracts.md 33 (render = family 0, loading = a compute-only
//! family, copy = a transfer-only family), buffers in the three memory kinds of contracts.md 8
//! with concurrent sharing across those families, compute pipelines and per-thread submitters.

use ash::vk::{self, TaggedStructure};
use std::collections::BTreeMap;
use std::ffi::{CStr, c_void};
use std::sync::Mutex;

pub const RENDER: usize = 0;
pub const LOADING: usize = 1;
const TIMEOUT_NS: u64 = 20_000_000_000;

static MESSAGES: Mutex<BTreeMap<String, (u32, String)>> = Mutex::new(BTreeMap::new());

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    unsafe {
        let data = &*data;
        let id = if data.p_message_id_name.is_null() { "?".into() } else { CStr::from_ptr(data.p_message_id_name).to_string_lossy().into_owned() };
        let msg = if data.p_message.is_null() { String::new() } else { CStr::from_ptr(data.p_message).to_string_lossy().into_owned() };
        let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) { "E" } else { "W" };
        MESSAGES.lock().unwrap().entry(format!("{level} {id}")).or_insert((0, msg)).0 += 1;
    }
    vk::FALSE
}

/// Prints collected validation messages; returns (errors, warnings).
pub fn validation_report() -> (u32, u32) {
    let map = MESSAGES.lock().unwrap();
    let e = map.iter().filter(|(k, _)| k.starts_with('E')).map(|(_, v)| v.0).sum();
    let w = map.iter().filter(|(k, _)| k.starts_with('W')).map(|(_, v)| v.0).sum();
    for (id, (n, m)) in map.iter() {
        let short: String = m.chars().take(500).collect();
        eprintln!("VVL {id} x{n}: {short}");
    }
    (e, w)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mem {
    /// Device-local, host-visible, coherent: VRAM through resizable BAR.
    Upload,
    /// Device-local only.
    Device,
    /// Host-cached readback.
    Readback,
}

#[derive(Clone, Copy)]
pub struct Buf {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub addr: u64,
    pub ptr: *mut u8,
    pub size: u64,
}
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    pub unsafe fn write<T: Copy>(&self, offset: usize, data: &[T]) {
        let b = crate::bytes(data);
        assert!(offset + b.len() <= self.size as usize);
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), self.ptr.add(offset), b.len()) };
    }
    pub unsafe fn slice(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }
}

pub fn usage2() -> vk::BufferUsageFlags2 {
    vk::BufferUsageFlags2::STORAGE_BUFFER
        | vk::BufferUsageFlags2::TRANSFER_SRC
        | vk::BufferUsageFlags2::TRANSFER_DST
        | vk::BufferUsageFlags2::INDIRECT_BUFFER
        | vk::BufferUsageFlags2::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags2::ACCELERATION_STRUCTURE_STORAGE_KHR
        | vk::BufferUsageFlags2::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
}

pub struct Gpu {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub families: [u32; 3],
    pub queues: [vk::Queue; 3],
    pub mem: vk::PhysicalDeviceMemoryProperties,
    pub ts_period_ns: f64,
    pub name: String,
    pub driver: String,
    pub cluster: ash::nv::cluster_acceleration_structure::Device,
    pub accel: ash::khr::acceleration_structure::Device,
    pub cl_props: vk::PhysicalDeviceClusterAccelerationStructurePropertiesNV<'static>,
    pub sparse_props: vk::PhysicalDeviceSparseProperties,
    pub queue_flags: [vk::QueueFlags; 3],
    pub layout: vk::PipelineLayout,
    pub tex_set_layout: vk::DescriptorSetLayout,
    pub tex_layout: vk::PipelineLayout,
}
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

/// Images per descriptor set in the sampling kernels.
pub const SET_IMAGES: u32 = 16;

const DEVICE_EXTENSIONS: &[&CStr] = &[
    ash::khr::acceleration_structure::NAME,
    ash::khr::deferred_host_operations::NAME,
    ash::khr::ray_query::NAME,
    ash::nv::cluster_acceleration_structure::NAME,
];

impl Gpu {
    pub unsafe fn new() -> Gpu {
        unsafe {
            let entry = ash::Entry::load().expect("Vulkan loader");
            let iexts = entry.enumerate_instance_extension_properties(None).unwrap();
            let debug = iexts.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME))
                && std::env::var("VK_INSTANCE_LAYERS").is_ok_and(|v| v.contains("validation"));
            let app = vk::ApplicationInfo::default().application_name(c"loading").api_version(vk::API_VERSION_1_4);
            let names: Vec<_> = if debug { vec![ash::ext::debug_utils::NAME.as_ptr()] } else { vec![] };
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&names), None)
                .expect("instance");
            if debug {
                let du = ash::ext::debug_utils::Instance::load(&entry, &instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
                    .pfn_user_callback(Some(debug_callback));
                let _ = du.create_debug_utils_messenger(&info, None);
            }
            let pdev = instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .find(|&p| instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
                .expect("discrete GPU");
            let qf = instance.get_physical_device_queue_family_properties(pdev);
            let find = |want: vk::QueueFlags, not: vk::QueueFlags| {
                qf.iter().position(|q| q.queue_flags.contains(want) && !q.queue_flags.intersects(not)).unwrap() as u32
            };
            let g = vk::QueueFlags::GRAPHICS;
            let c = vk::QueueFlags::COMPUTE;
            let video = vk::QueueFlags::VIDEO_DECODE_KHR | vk::QueueFlags::VIDEO_ENCODE_KHR | vk::QueueFlags::OPTICAL_FLOW_NV;
            let families = [find(g | c, vk::QueueFlags::empty()), find(c, g), find(vk::QueueFlags::TRANSFER, g | c | video)];

            let mut f_cluster = vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV::default().cluster_acceleration_structure(true);
            let mut f_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
            let mut f_rq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .scalar_block_layout(true)
                .timeline_semaphore(true)
                .host_query_reset(true)
                .runtime_descriptor_array(true);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true).maintenance4(true);
            let mut f14 = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true).host_image_copy(true);
            let core = vk::PhysicalDeviceFeatures::default()
                .shader_int64(true)
                .sampler_anisotropy(true)
                .shader_resource_min_lod(true)
                .sparse_binding(true)
                .sparse_residency_buffer(true)
                .sparse_residency_image2_d(true)
                .sparse_residency_aliased(true);
            let mut f2 = vk::PhysicalDeviceFeatures2::default().features(core);
            // `PROBE_LOADING_PRIORITY` sets the loading and copy queues' priority (render is 1.0).
            let low: f32 = std::env::var("PROBE_LOADING_PRIORITY").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
            let prio = [[1.0f32], [low], [low]];
            let qinfo: Vec<_> = families
                .iter()
                .zip(&prio)
                .map(|(&f, p)| vk::DeviceQueueCreateInfo::default().queue_family_index(f).queue_priorities(p))
                .collect();
            let ext_names: Vec<_> = DEVICE_EXTENSIONS.iter().map(|n| n.as_ptr()).collect();
            let info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qinfo)
                .push(&mut f2)
                .push(&mut f12)
                .push(&mut f13)
                .push(&mut f14)
                .push(&mut f_as)
                .push(&mut f_rq)
                .push(&mut f_cluster)
                .enabled_extension_names(&ext_names);
            let device = instance.create_device(pdev, &info, None).expect("device");
            let queues = families.map(|f| device.get_device_queue(f, 0));
            let queue_flags = families.map(|f| qf[f as usize].queue_flags);
            let mut cl_props = vk::PhysicalDeviceClusterAccelerationStructurePropertiesNV::default();
            let mut drv = vk::PhysicalDeviceDriverProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut cl_props).push(&mut drv);
            instance.get_physical_device_properties2(pdev, &mut props2);
            let props = props2.properties;
            cl_props.p_next = std::ptr::null_mut();
            let name = props.device_name_as_c_str().unwrap().to_string_lossy().into_owned();
            let driver = drv.driver_info_as_c_str().unwrap().to_string_lossy().into_owned();
            let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(128)];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges), None).unwrap();
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(SET_IMAGES)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let tex_set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
            let sets = [tex_set_layout];
            let tex_layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges).set_layouts(&sets), None)
                .unwrap();
            Gpu {
                cluster: ash::nv::cluster_acceleration_structure::Device::load(&instance, &device),
                accel: ash::khr::acceleration_structure::Device::load(&instance, &device),
                mem: instance.get_physical_device_memory_properties(pdev),
                ts_period_ns: f64::from(props.limits.timestamp_period),
                sparse_props: props.sparse_properties,
                queue_flags,
                _entry: entry,
                instance,
                pdev,
                device,
                families,
                queues,
                name,
                driver,
                cl_props,
                layout,
                tex_set_layout,
                tex_layout,
            }
        }
    }

    pub fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags, not: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem.memory_type_count).find(|&i| {
            let f = self.mem.memory_types[i as usize].property_flags;
            bits & (1 << i) != 0 && f.contains(want) && !f.intersects(not)
        })
    }

    pub fn kind_type(&self, bits: u32, kind: Mem) -> u32 {
        let dl = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        match kind {
            Mem::Upload => self.memory_type(bits, dl | hv, vk::MemoryPropertyFlags::empty()),
            Mem::Device => self.memory_type(bits, dl, vk::MemoryPropertyFlags::HOST_VISIBLE),
            Mem::Readback => self.memory_type(bits, hv | vk::MemoryPropertyFlags::HOST_CACHED, vk::MemoryPropertyFlags::empty()),
        }
        .expect("memory type")
    }

    /// Every buffer carries the gpu_malloc usage set (gpu.md) and concurrent sharing across
    /// the three families.
    pub unsafe fn buffer(&self, size: u64, kind: Mem) -> Buf {
        unsafe {
            let size = size.max(256);
            let mut usage2 = vk::BufferUsageFlags2CreateInfo::default().usage(usage2());
            self.buffer_with(size, kind, Some(&mut usage2), vk::BufferUsageFlags::empty())
        }
    }

    unsafe fn buffer_with(&self, size: u64, kind: Mem, usage2: Option<&mut vk::BufferUsageFlags2CreateInfo<'_>>, usage: vk::BufferUsageFlags) -> Buf {
        unsafe {
            let fam = self.families;
            let mut ci = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&fam);
            if let Some(u2) = usage2 {
                ci = ci.push(u2);
            }
            let buffer = self.device.create_buffer(&ci, None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let ty = self.kind_type(req.memory_type_bits, kind);
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = self
                .device
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push(&mut flags), None)
                .expect("allocate");
            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let addr = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            let ptr = if kind == Mem::Device {
                std::ptr::null_mut()
            } else {
                self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap() as *mut u8
            };
            Buf { buffer, memory, addr, ptr, size }
        }
    }

    pub unsafe fn free(&self, b: Buf) {
        unsafe {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    /// A SPARSE_BINDING | SPARSE_RESIDENCY buffer with the gpu_malloc usage set, concurrent
    /// across the three families, no memory bound. Returns the buffer, its address and its
    /// memory requirements (alignment = page size).
    pub unsafe fn sparse_buffer(&self, size: u64) -> (vk::Buffer, u64, vk::MemoryRequirements) {
        unsafe {
            let fam = self.families;
            let mut u2 = vk::BufferUsageFlags2CreateInfo::default().usage(usage2());
            let ci = vk::BufferCreateInfo::default()
                .size(size)
                .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY)
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&fam)
                .push(&mut u2);
            let buffer = self.device.create_buffer(&ci, None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let addr = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            (buffer, addr, req)
        }
    }

    pub unsafe fn alloc(&self, size: u64, ty: u32) -> vk::DeviceMemory {
        unsafe {
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            self.device
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty).push(&mut flags), None)
                .expect("allocate")
        }
    }

    /// vkQueueBindSparse on `queue` with one bind info, waiting and signalling timeline values.
    pub unsafe fn bind_sparse(&self, queue: usize, info: vk::BindSparseInfo<'_>, wait: &[(vk::Semaphore, u64)], signal: (vk::Semaphore, u64)) {
        unsafe {
            let ws: Vec<_> = wait.iter().map(|w| w.0).collect();
            let wv: Vec<_> = wait.iter().map(|w| w.1).collect();
            let ss = [signal.0];
            let sv = [signal.1];
            let mut t = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wv).signal_semaphore_values(&sv);
            let info = info.wait_semaphores(&ws).signal_semaphores(&ss).push(&mut t);
            if let Err(e) = self.device.queue_bind_sparse(self.queues[queue], &[info], vk::Fence::null()) {
                crate::fail(&format!("vkQueueBindSparse on queue {queue}: {e:?}"));
            }
        }
    }

    pub unsafe fn pipeline(&self, code: &[u32], layout: vk::PipelineLayout) -> vk::Pipeline {
        unsafe {
            let mut module = vk::ShaderModuleCreateInfo::default().code(code);
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).name(c"main").push(&mut module);
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e).expect("pipeline")[0]
        }
    }

    pub unsafe fn push<T: Copy>(&self, cmd: vk::CommandBuffer, layout: vk::PipelineLayout, data: &T) {
        unsafe { self.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, crate::bytes(std::slice::from_ref(data))) };
    }

    pub unsafe fn barrier(&self, cmd: vk::CommandBuffer) {
        unsafe {
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
            self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
        }
    }

    pub unsafe fn timeline(&self) -> vk::Semaphore {
        unsafe {
            let mut t = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            self.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push(&mut t), None).unwrap()
        }
    }

    pub unsafe fn wait_value(&self, sem: vk::Semaphore, value: u64) -> Result<(), vk::Result> {
        unsafe {
            let s = [sem];
            let v = [value];
            self.device.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&s).values(&v), TIMEOUT_NS)
        }
    }
}

/// One thread's command pool, ring of command buffers, timeline and timestamp pool for one queue.
pub struct Submitter<'g> {
    pub gpu: &'g Gpu,
    pub queue: usize,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    pub timeline: vk::Semaphore,
    pub value: u64,
    pub ts: vk::QueryPool,
    ring: usize,
    ts_per_slot: u32,
}

impl<'g> Submitter<'g> {
    pub unsafe fn new(gpu: &'g Gpu, queue: usize, ring: usize, ts_per_slot: u32) -> Self {
        unsafe {
            let pool = gpu
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(gpu.families[queue]).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let cmds = gpu
                .device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(ring as u32))
                .unwrap();
            let n = (ring as u32 * ts_per_slot).max(2);
            let ts = gpu.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(n), None).unwrap();
            gpu.device.reset_query_pool(ts, 0, n);
            Submitter { gpu, queue, pool, cmds, timeline: gpu.timeline(), value: 0, ts, ring, ts_per_slot }
        }
    }

    pub fn slot(&self, value: u64) -> usize {
        (value % self.ring as u64) as usize
    }

    /// Waits until the slot for submission `value + 1` is free, then begins its command buffer.
    pub unsafe fn begin(&mut self) -> vk::CommandBuffer {
        unsafe {
            let next = self.value + 1;
            if next > self.ring as u64 {
                self.wait(next - self.ring as u64);
            }
            let cmd = self.cmds[self.slot(next)];
            self.gpu.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            self.gpu
                .device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .unwrap();
            let base = self.slot(next) as u32 * self.ts_per_slot;
            self.gpu.device.cmd_reset_query_pool(cmd, self.ts, base, self.ts_per_slot);
            cmd
        }
    }

    pub unsafe fn stamp(&self, cmd: vk::CommandBuffer, index: u32) {
        unsafe {
            let base = self.slot(self.value + 1) as u32 * self.ts_per_slot;
            self.gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, self.ts, base + index);
        }
    }

    /// Ends and submits; returns the timeline value it signals.
    pub unsafe fn submit(&mut self, cmd: vk::CommandBuffer) -> u64 {
        unsafe { self.submit_waiting(cmd, &[]) }
    }

    pub unsafe fn submit_waiting(&mut self, cmd: vk::CommandBuffer, waits: &[(vk::Semaphore, u64)]) -> u64 {
        unsafe {
            self.gpu.device.end_command_buffer(cmd).unwrap();
            self.value += 1;
            let c = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let s = [vk::SemaphoreSubmitInfo::default().semaphore(self.timeline).value(self.value).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let w: Vec<_> = waits
                .iter()
                .map(|&(sem, v)| vk::SemaphoreSubmitInfo::default().semaphore(sem).value(v).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS))
                .collect();
            let info = vk::SubmitInfo2::default().command_buffer_infos(&c).signal_semaphore_infos(&s).wait_semaphore_infos(&w);
            if let Err(e) = self.gpu.device.queue_submit2(self.gpu.queues[self.queue], &[info], vk::Fence::null()) {
                crate::fail(&format!("queue_submit2 on queue {}: {e:?}", self.queue));
            }
            self.value
        }
    }

    pub unsafe fn wait(&self, value: u64) {
        unsafe {
            if let Err(e) = self.gpu.wait_value(self.timeline, value) {
                crate::fail(&format!("timeline wait on queue {}: {e:?}", self.queue));
            }
        }
    }

    /// Milliseconds between timestamps of submission `value` (waits for it).
    pub unsafe fn ms(&self, value: u64, from: u32, to: u32) -> f64 {
        unsafe {
            self.wait(value);
            let base = self.slot(value) as u32 * self.ts_per_slot;
            let mut t = [0u64; 2];
            self.gpu.device.get_query_pool_results(self.ts, base + from, &mut t[..1], vk::QueryResultFlags::TYPE_64).unwrap();
            self.gpu.device.get_query_pool_results(self.ts, base + to, &mut t[1..], vk::QueryResultFlags::TYPE_64).unwrap();
            (t[1].wrapping_sub(t[0])) as f64 * self.gpu.ts_period_ns / 1e6
        }
    }

    pub unsafe fn run<F: FnOnce(vk::CommandBuffer)>(&mut self, f: F) -> u64 {
        unsafe {
            let cmd = self.begin();
            f(cmd);
            let v = self.submit(cmd);
            self.wait(v);
            v
        }
    }
}

impl Drop for Submitter<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.gpu.wait_value(self.timeline, self.value);
            self.gpu.device.destroy_command_pool(self.pool, None);
            self.gpu.device.destroy_semaphore(self.timeline, None);
            self.gpu.device.destroy_query_pool(self.ts, None);
        }
    }
}
