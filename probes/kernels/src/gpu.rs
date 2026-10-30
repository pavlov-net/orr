//! Raw-ash device for the kernel probes: descriptor heap, root by `PUSH_ADDRESS`, buffers by
//! device address, image and sampler heaps, compute and mesh pipelines, isolated timing.

use ash::vk::{self, TaggedStructure};
use std::collections::BTreeMap;
use std::ffi::{CStr, c_void};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const FENCE_TIMEOUT_NS: u64 = 30_000_000_000;

pub fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

pub fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a.max(1)) * a.max(1)
}

static MESSAGES: Mutex<BTreeMap<String, (u32, String)>> = Mutex::new(BTreeMap::new());

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { &*data };
    let text = |p: *const std::ffi::c_char| if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() };
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) { "E" } else { "W" };
    let mut map = MESSAGES.lock().unwrap();
    let entry = map.entry(format!("{level} {}", text(data.p_message_id_name))).or_insert((0, text(data.p_message)));
    entry.0 += 1;
    vk::FALSE
}

/// Prints collected validation messages; returns (errors, warnings).
pub fn dump_messages() -> (u32, u32) {
    let map = MESSAGES.lock().unwrap();
    let count = |p: char| map.iter().filter(|(k, _)| k.starts_with(p)).map(|(_, v)| v.0).sum();
    for (id, (n, message)) in map.iter() {
        let short: String = message.chars().take(600).collect();
        println!("VVL {id} x{n}: {short}");
    }
    (count('E'), count('W'))
}

pub fn fatal(what: &str) -> ! {
    let (e, w) = dump_messages();
    println!("FATAL {what} (validation errors={e} warnings={w})");
    std::process::exit(2);
}

#[derive(Clone, Copy, PartialEq)]
pub enum Mem {
    Device,
    /// Device-local, host-visible (resizable BAR).
    Upload,
    Readback,
}

#[derive(Clone, Copy)]
pub struct Buf {
    pub buffer: vk::Buffer,
    pub addr: u64,
    pub offset: u64,
    pub ptr: *mut u8,
    pub size: u64,
}

impl Buf {
    pub unsafe fn write<T: Copy>(&self, offset: usize, data: &[T]) {
        let b = bytes(data);
        assert!(offset + b.len() <= self.size as usize);
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), self.ptr.add(offset), b.len()) };
    }

    pub unsafe fn read<T: Copy>(&self, count: usize) -> Vec<T> {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<T>(), count) }.to_vec()
    }
}

pub struct Image {
    pub image: vk::Image,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub point: vk::PipelineBindPoint,
}

#[derive(Default)]
pub struct Options {
    /// VK_NV_device_diagnostics_config with every Aftermath flag, plus checkpoints.
    pub diagnostics: bool,
    /// VK_EXT_frame_boundary when the device (or a capture layer) offers it.
    pub frame_boundary: bool,
}

pub struct Gpu {
    pub _entry: ash::Entry,
    pub _instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub pool: vk::CommandPool,
    pub mem: vk::PhysicalDeviceMemoryProperties,
    pub limits: vk::PhysicalDeviceLimits,
    pub heap_props: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    pub heap: ash::ext::descriptor_heap::Device,
    pub mesh: ash::ext::mesh_shader::Device,
    pub coopvec: ash::nv::cooperative_vector::Device,
    pub coopvec_instance: ash::nv::cooperative_vector::Instance,
    pub exec: ash::khr::pipeline_executable_properties::Device,
    pub timestamps: vk::QueryPool,
    pub resource_heap: Buf,
    pub sampler_heap: Buf,
    pub image_slots: u32,
    pub name: String,
    pub driver: String,
    pub frame_boundary: bool,
}

const EXTENSIONS: &[&CStr] = &[
    ash::ext::descriptor_heap::NAME,
    ash::khr::shader_untyped_pointers::NAME,
    ash::ext::shader_image_atomic_int64::NAME,
    ash::ext::mesh_shader::NAME,
    ash::nv::cooperative_vector::NAME,
    ash::ext::shader_replicated_composites::NAME,
    ash::khr::pipeline_executable_properties::NAME,
];

impl Gpu {
    pub unsafe fn new(opts: &Options) -> Gpu {
        unsafe {
            let entry = ash::Entry::load().expect("Vulkan loader");
            let instance_exts = entry.enumerate_instance_extension_properties(None).unwrap();
            let debug = instance_exts.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME));
            let app = vk::ApplicationInfo::default().application_name(c"kernels").api_version(vk::API_VERSION_1_4);
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
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
                    .pfn_user_callback(Some(debug_callback));
                let _ = du.create_debug_utils_messenger(&info, None);
            }
            let mut names: Vec<&CStr> = EXTENSIONS.to_vec();
            if opts.diagnostics {
                names.push(ash::nv::device_diagnostics_config::NAME);
                names.push(ash::nv::device_diagnostic_checkpoints::NAME);
            }
            let pdev = instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .find(|&p| {
                    let exts = instance.enumerate_device_extension_properties(p).unwrap();
                    names.iter().all(|n| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(*n)))
                })
                .expect("no device with descriptor heap, image int64 atomics, mesh shaders and cooperative vectors");
            let dev_exts = instance.enumerate_device_extension_properties(pdev).unwrap();
            let frame_boundary = opts.frame_boundary && dev_exts.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::ext::frame_boundary::NAME));
            if frame_boundary {
                names.push(ash::ext::frame_boundary::NAME);
            }
            let props = instance.get_physical_device_properties(pdev);
            let name = props.device_name_as_c_str().unwrap().to_string_lossy().into_owned();
            let v = props.driver_version;
            let driver = format!("{}.{:02}", v >> 22, (v >> 14) & 0xff);
            let family = instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
                .unwrap() as u32;

            let mut f_heap = vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default().descriptor_heap(true);
            let mut f_untyped = vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR::default().shader_untyped_pointers(true);
            let mut f_img64 = vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT::default().shader_image_int64_atomics(true);
            let mut f_mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true).task_shader(true);
            let mut f_cv = vk::PhysicalDeviceCooperativeVectorFeaturesNV::default().cooperative_vector(true);
            // Slang emits ReplicatedCompositesEXT for CoopVec construction.
            let mut f_rep = vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT::default().shader_replicated_composites(true);
            let mut f_exec = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .scalar_block_layout(true)
                .uniform_buffer_standard_layout(true)
                .shader_buffer_int64_atomics(true)
                .shader_float16(true)
                .shader_int8(true)
                .storage_buffer8_bit_access(true)
                .vulkan_memory_model(true)
                .vulkan_memory_model_device_scope(true)
                .timeline_semaphore(true);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true).maintenance4(true).dynamic_rendering(true);
            let mut f14 = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true);
            let core = vk::PhysicalDeviceFeatures::default().shader_int64(true).shader_int16(true).fragment_stores_and_atomics(true);
            let mut f2 = vk::PhysicalDeviceFeatures2::default().features(core);
            let mut diag = vk::DeviceDiagnosticsConfigCreateInfoNV::default().flags(
                vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_AUTOMATIC_CHECKPOINTS
                    | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_RESOURCE_TRACKING
                    | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_DEBUG_INFO
                    | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_ERROR_REPORTING,
            );
            let priorities = [1.0];
            let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let name_ptrs: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();
            let mut info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&name_ptrs)
                .push(&mut f2)
                .push(&mut f11)
                .push(&mut f12)
                .push(&mut f13)
                .push(&mut f14)
                .push(&mut f_heap)
                .push(&mut f_untyped)
                .push(&mut f_img64)
                .push(&mut f_mesh)
                .push(&mut f_cv)
                .push(&mut f_rep)
                .push(&mut f_exec);
            if opts.diagnostics {
                info = info.push(&mut diag);
            }
            let mut f_fb = vk::PhysicalDeviceFrameBoundaryFeaturesEXT::default().frame_boundary(true);
            if frame_boundary {
                info = info.push(&mut f_fb);
            }
            let device = instance.create_device(pdev, &info, None).expect("device");
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let mut heap_props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut heap_props);
            instance.get_physical_device_properties2(pdev, &mut props2);
            let limits = props2.properties.limits;
            heap_props.p_next = std::ptr::null_mut();
            let timestamps = device
                .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(64), None)
                .unwrap();
            let mut gpu = Gpu {
                heap: ash::ext::descriptor_heap::Device::load(&instance, &device),
                mesh: ash::ext::mesh_shader::Device::load(&instance, &device),
                coopvec: ash::nv::cooperative_vector::Device::load(&instance, &device),
                coopvec_instance: ash::nv::cooperative_vector::Instance::load(&entry, &instance),
                exec: ash::khr::pipeline_executable_properties::Device::load(&instance, &device),
                mem: instance.get_physical_device_memory_properties(pdev),
                _entry: entry,
                _instance: instance,
                pdev,
                device,
                queue,
                pool,
                limits,
                heap_props,
                timestamps,
                resource_heap: Buf { buffer: vk::Buffer::null(), addr: 0, offset: 0, ptr: std::ptr::null_mut(), size: 0 },
                sampler_heap: Buf { buffer: vk::Buffer::null(), addr: 0, offset: 0, ptr: std::ptr::null_mut(), size: 0 },
                image_slots: 16,
                name,
                driver,
                frame_boundary,
            };
            gpu.make_heaps();
            gpu
        }
    }

    unsafe fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem.memory_type_count).find(|&i| bits & (1 << i) != 0 && self.mem.memory_types[i as usize].property_flags.contains(flags))
    }

    pub unsafe fn buffer_usage(&self, size: u64, kind: Mem, align: u64, usage: vk::BufferUsageFlags) -> Buf {
        unsafe {
            let size = size.max(16);
            let align = align.max(16);
            let usage = usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size + align).usage(usage), None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = match kind {
                Mem::Device => self.memory_type(req.memory_type_bits, local),
                Mem::Upload => self.memory_type(req.memory_type_bits, local | host).or_else(|| self.memory_type(req.memory_type_bits, host)),
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
    }

    pub unsafe fn buffer(&self, size: u64, kind: Mem) -> Buf {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        unsafe { self.buffer_usage(size, kind, 256, usage) }
    }

    unsafe fn make_heaps(&mut self) {
        unsafe {
            let p = &self.heap_props;
            let reserved_offset = align_up(p.image_descriptor_size * u64::from(self.image_slots), p.image_descriptor_alignment.max(64));
            let size = reserved_offset + p.min_resource_heap_reserved_range;
            self.resource_heap = self.buffer_usage(size, Mem::Upload, p.resource_heap_alignment, vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT);
            let s_reserved = align_up(p.sampler_descriptor_size, p.sampler_descriptor_alignment.max(1));
            let s_size = s_reserved + p.min_sampler_heap_reserved_range;
            self.sampler_heap = self.buffer_usage(s_size, Mem::Upload, p.sampler_heap_alignment, vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT);
            let linear = [vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)];
            let dst = [vk::HostAddressRangeEXT::default()
                .address(std::slice::from_raw_parts_mut(self.sampler_heap.ptr, self.heap_props.sampler_descriptor_size as usize))];
            self.heap.write_sampler_descriptors(&linear, &dst).expect("sampler descriptor");
        }
    }

    pub unsafe fn bind_heaps(&self, cmd: vk::CommandBuffer) {
        let p = &self.heap_props;
        let reserved_offset = align_up(p.image_descriptor_size * u64::from(self.image_slots), p.image_descriptor_alignment.max(64));
        let s_reserved = align_up(p.sampler_descriptor_size, p.sampler_descriptor_alignment.max(1));
        let bind = |addr: u64, off: u64, reserved: u64| vk::BindHeapInfoEXT {
            heap_range: vk::DeviceAddressRangeEXT { address: addr, size: off + reserved },
            reserved_range_offset: off,
            reserved_range_size: reserved,
            ..Default::default()
        };
        unsafe {
            self.heap.cmd_bind_resource_heap(cmd, &bind(self.resource_heap.addr, reserved_offset, p.min_resource_heap_reserved_range));
            self.heap.cmd_bind_sampler_heap(cmd, &bind(self.sampler_heap.addr, s_reserved, p.min_sampler_heap_reserved_range));
        }
    }

    pub unsafe fn image(&self, format: vk::Format, width: u32, height: u32, usage: vk::ImageUsageFlags) -> Image {
        unsafe {
            let image = self
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D { width, height, depth: 1 })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST),
                    None,
                )
                .unwrap();
            let req = self.device.get_image_memory_requirements(image);
            let ty = self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap();
            let memory = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None).unwrap();
            self.device.bind_image_memory(image, memory, 0).unwrap();
            let cmd = self.begin();
            let b = [vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(color_range())];
            self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b));
            self.submit_wait(cmd, "image layout");
            Image { image, format, width, height }
        }
    }

    pub unsafe fn image_view(&self, img: &Image) -> vk::ImageView {
        let info = vk::ImageViewCreateInfo::default()
            .image(img.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(img.format)
            .subresource_range(color_range());
        unsafe { self.device.create_image_view(&info, None).unwrap() }
    }

    /// Writes a sampled or storage descriptor for `img` into resource-heap slot `slot`.
    pub unsafe fn write_image_slot(&self, slot: u32, img: &Image, storage: bool) {
        assert!(slot < self.image_slots);
        let view = vk::ImageViewCreateInfo::default()
            .image(img.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(img.format)
            .subresource_range(color_range());
        let desc = vk::ImageDescriptorInfoEXT::default().view(&view).layout(vk::ImageLayout::GENERAL);
        let ty = if storage { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::SAMPLED_IMAGE };
        let infos = [vk::ResourceDescriptorInfoEXT::default().ty(ty).data(vk::ResourceDescriptorDataEXT { p_image: &desc })];
        let size = self.heap_props.image_descriptor_size as usize;
        let dst = [vk::HostAddressRangeEXT::default()
            .address(unsafe { std::slice::from_raw_parts_mut(self.resource_heap.ptr.add(slot as usize * size), size) })];
        unsafe { self.heap.write_resource_descriptors(&infos, &dst) }.expect("image descriptor");
    }

    pub unsafe fn begin(&self) -> vk::CommandBuffer {
        unsafe {
            let cmd = self
                .device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1))
                .unwrap()[0];
            self.device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .unwrap();
            cmd
        }
    }

    /// Submits and waits; returns the error instead of exiting.
    pub unsafe fn try_submit_wait(&self, cmd: vk::CommandBuffer) -> Result<(), vk::Result> {
        unsafe {
            self.full_barrier(cmd);
            self.device.end_command_buffer(cmd).unwrap();
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let cmds = [cmd];
            self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence)?;
            self.device.wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS)?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.pool, &cmds);
            Ok(())
        }
    }

    /// Submits with a frame-end boundary (VK_EXT_frame_boundary) and waits.
    pub unsafe fn submit_frame(&self, cmd: vk::CommandBuffer, frame: u64) {
        unsafe {
            self.device.end_command_buffer(cmd).unwrap();
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let cmds = [cmd];
            let mut fb = vk::FrameBoundaryEXT::default().flags(vk::FrameBoundaryFlagsEXT::FRAME_END).frame_id(frame);
            let mut submit = vk::SubmitInfo::default().command_buffers(&cmds);
            if self.frame_boundary {
                submit = submit.push(&mut fb);
            }
            self.device.queue_submit(self.queue, &[submit], fence).unwrap();
            self.device.wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS).unwrap();
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.pool, &cmds);
        }
    }

    pub unsafe fn submit_wait(&self, cmd: vk::CommandBuffer, what: &str) {
        if let Err(e) = unsafe { self.try_submit_wait(cmd) } {
            fatal(&format!("{what}: {e:?}"));
        }
    }

    pub unsafe fn full_barrier(&self, cmd: vk::CommandBuffer) {
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS | vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::HOST_READ)];
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
    }

    pub unsafe fn timestamp(&self, cmd: vk::CommandBuffer, index: u32) {
        unsafe { self.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, self.timestamps, index) };
    }

    /// Microseconds between timestamps (b - a) for each pair.
    pub unsafe fn read_timestamps(&self, count: u32) -> Vec<u64> {
        let mut ticks = vec![0u64; count as usize];
        unsafe {
            self.device
                .get_query_pool_results(self.timestamps, 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                .unwrap()
        };
        ticks
    }

    pub fn ticks_us(&self, t: u64) -> f64 {
        t as f64 * f64::from(self.limits.timestamp_period) / 1e3
    }

    pub unsafe fn push_root(&self, cmd: vk::CommandBuffer, root: u64) {
        let data = root.to_le_bytes();
        let push = vk::PushDataInfoEXT { offset: 0, data: vk::HostAddressRangeConstEXT::default().address(&data), ..Default::default() };
        unsafe { self.heap.cmd_push_data(cmd, &push) };
    }

    /// Heap-mode compute pipeline; binding (0, 0) is the root through `PUSH_ADDRESS`, plus `extra`.
    pub unsafe fn compute(&self, code: &[u32], entry: &CStr, extra: &[vk::DescriptorSetAndBindingMappingEXT<'_>]) -> Pipeline {
        let mut mappings = vec![root_mapping()];
        mappings.extend_from_slice(extra);
        let mut module = vk::ShaderModuleCreateInfo::default().code(code);
        let mut map_info = vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&mappings);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(entry)
            .push(&mut module)
            .push(&mut map_info);
        let mut flags = vk::PipelineCreateFlags2CreateInfo::default()
            .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT | vk::PipelineCreateFlags2::CAPTURE_STATISTICS_KHR);
        let info = vk::ComputePipelineCreateInfo::default().stage(stage).push(&mut flags);
        let pipeline = unsafe { self.device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) }
            .map_err(|(_, e)| e)
            .unwrap_or_else(|e| fatal(&format!("compute pipeline {entry:?}: {e:?}")))[0];
        Pipeline { pipeline, point: vk::PipelineBindPoint::COMPUTE }
    }

    /// Heap-mode mesh + pixel pipeline rendering into one colour target of `format`.
    pub unsafe fn mesh_pipeline(&self, mesh: &[u32], pixel: &[u32], format: vk::Format) -> Pipeline {
        let mappings = [root_mapping()];
        let codes = [mesh, pixel];
        let flags_s = [vk::ShaderStageFlags::MESH_EXT, vk::ShaderStageFlags::FRAGMENT];
        let names = [c"mesh_main", c"pixel_main"];
        let mut modules: Vec<_> = codes.iter().map(|c| vk::ShaderModuleCreateInfo::default().code(c)).collect();
        let mut maps: Vec<_> = (0..2).map(|_| vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&mappings)).collect();
        let stages: Vec<_> = modules
            .iter_mut()
            .zip(maps.iter_mut())
            .enumerate()
            .map(|(i, (m, mi))| vk::PipelineShaderStageCreateInfo::default().stage(flags_s[i]).name(names[i]).push(m).push(mi))
            .collect();
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachments = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::R)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [format];
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
        let pipeline = unsafe { self.device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
            .map_err(|(_, e)| e)
            .unwrap_or_else(|e| fatal(&format!("mesh pipeline: {e:?}")))[0];
        Pipeline { pipeline, point: vk::PipelineBindPoint::GRAPHICS }
    }

    /// Registers, shared memory and spills from pipeline executable statistics.
    pub unsafe fn stats(&self, pipeline: vk::Pipeline) -> String {
        let pinfo = vk::PipelineInfoKHR::default().pipeline(pipeline);
        let n = unsafe { self.exec.get_pipeline_executable_properties_len(&pinfo) }.unwrap_or(0);
        let mut out = String::new();
        for i in 0..n as u32 {
            let einfo = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(i);
            let m = unsafe { self.exec.get_pipeline_executable_statistics_len(&einfo) }.unwrap_or(0);
            let mut stats = vec![vk::PipelineExecutableStatisticKHR::default(); m];
            let _ = unsafe { self.exec.get_pipeline_executable_statistics(&einfo, &mut stats) };
            for s in &stats {
                let name = s.name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
                if name.contains("Register") || name.contains("Spill") || name.contains("Shared") {
                    let v = match s.format {
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => unsafe { s.value.u64 }.to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => unsafe { s.value.i64 }.to_string(),
                        _ => "?".into(),
                    };
                    out += &format!(" {}={v}", name.replace(' ', "_"));
                }
            }
        }
        out
    }

    /// Times each variant: per submission one warm and one timed run, `PROBE_GAP_MS` idle between
    /// submissions, variants round-robin. `pre` runs before each run, outside the timestamps.
    /// Returns (median, p10, p90) in us per variant.
    pub unsafe fn bench(&self, variants: &[&dyn Fn(vk::CommandBuffer)], pre: Option<&dyn Fn(vk::CommandBuffer)>) -> Vec<(f64, f64, f64)> {
        let rounds = env_u32("PROBE_ROUNDS", 31);
        let gap = Duration::from_millis(u64::from(env_u32("PROBE_GAP_MS", 6)));
        let run = |v: &dyn Fn(vk::CommandBuffer)| -> f64 {
            unsafe {
                let cmd = self.begin();
                self.device.cmd_reset_query_pool(cmd, self.timestamps, 0, 4);
                self.bind_heaps(cmd);
                for k in 0..2 {
                    if let Some(p) = pre {
                        p(cmd);
                        self.full_barrier(cmd);
                    }
                    self.timestamp(cmd, k * 2);
                    v(cmd);
                    self.full_barrier(cmd);
                    self.timestamp(cmd, k * 2 + 1);
                }
                self.submit_wait(cmd, "bench");
                let t = self.read_timestamps(4);
                self.ticks_us(t[3] - t[2])
            }
        };
        let warm_until = Instant::now() + Duration::from_millis(u64::from(env_u32("PROBE_WARMUP_MS", 1500)));
        while Instant::now() < warm_until {
            run(variants[0]);
            std::thread::sleep(gap);
        }
        let n = variants.len();
        let mut samples = vec![Vec::new(); n];
        for round in 0..rounds as usize {
            for k in 0..n {
                let vi = (k + round) % n;
                samples[vi].push(run(variants[vi]));
                std::thread::sleep(gap);
            }
        }
        samples
            .into_iter()
            .map(|mut s| {
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (s[s.len() / 2], s[s.len() / 10], s[s.len() * 9 / 10])
            })
            .collect()
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)
}

pub fn root_mapping() -> vk::DescriptorSetAndBindingMappingEXT<'static> {
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

/// Binding (0, `binding`) as a storage buffer whose address is read at root + `address_offset`.
pub fn indirect_mapping(binding: u32, address_offset: u32) -> vk::DescriptorSetAndBindingMappingEXT<'static> {
    vk::DescriptorSetAndBindingMappingEXT {
        descriptor_set: 0,
        first_binding: binding,
        binding_count: 1,
        resource_mask: vk::SpirvResourceTypeFlagsEXT::READ_ONLY_STORAGE_BUFFER | vk::SpirvResourceTypeFlagsEXT::READ_WRITE_STORAGE_BUFFER,
        source: vk::DescriptorMappingSourceEXT::INDIRECT_ADDRESS,
        source_data: vk::DescriptorMappingSourceDataEXT {
            indirect_address: vk::DescriptorMappingSourceIndirectAddressEXT { push_offset: 0, address_offset },
        },
        ..Default::default()
    }
}

#[macro_export]
macro_rules! spv {
    ($name:literal) => {
        ash::util::read_spv(&mut std::io::Cursor::new(&include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))[..]))
            .expect("valid SPIR-V")
    };
}
