//! Headless descriptor-heap device: render queue (graphics family) and the compute-only family,
//! buffers by address, images with resource-heap storage descriptors, heap-mode compute
//! pipelines whose root is a pushed address.

use ash::vk::{self, TaggedStructure};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_void};
use std::sync::Mutex;

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
        let short: String = m.chars().take(700).collect();
        eprintln!("VVL {id} x{n}: {short}");
    }
    (e, w)
}

#[derive(Clone, Copy, PartialEq)]
pub enum Mem {
    /// Device-local, host-visible, coherent (resizable BAR).
    Upload,
    Device,
    /// Host-visible, cached.
    Readback,
}

#[derive(Clone, Copy)]
pub struct Buf {
    pub buffer: vk::Buffer,
    pub addr: u64,
    pub ptr: *mut u8,
    pub size: u64,
}

#[derive(Clone, Copy)]
pub struct Img {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: [u32; 2],
    pub slot: u32,
}

pub struct Gpu {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    /// Graphics family, compute-only family.
    pub families: [u32; 2],
    pub queues: [vk::Queue; 2],
    pub mem: vk::PhysicalDeviceMemoryProperties,
    pub ts_period_ns: f64,
    pub name: String,
    pub driver: String,
    pub heap: ash::ext::descriptor_heap::Device,
    pub heap_props: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    pub validation: bool,
    pub unified: bool,
    pub heaps: bool,
    resource_heap: Buf,
    sampler_heap: Buf,
    resource_bind: vk::BindHeapInfoEXT<'static>,
    sampler_bind: vk::BindHeapInfoEXT<'static>,
    next_slot: std::cell::Cell<u32>,
    roots: Buf,
    next_root: std::cell::Cell<u64>,
}

pub const HEAP_SLOTS: u64 = 64;
const ROOT_SLOTS: u64 = 4096;
const ROOT_STRIDE: u64 = 256;

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a.max(1)) * a.max(1)
}

impl Gpu {
    /// `extra_instance` and `extra_device` are NGX's requirements. `unified` enables
    /// `unifiedImageLayouts`.
    pub unsafe fn new(extra_instance: &[CString], extra_device: &[CString], unified: bool, heaps: bool) -> Gpu {
        unsafe {
            let entry = ash::Entry::load().expect("Vulkan loader");
            let iexts = entry.enumerate_instance_extension_properties(None).unwrap();
            let validation = std::env::var("VK_INSTANCE_LAYERS").is_ok_and(|v| v.contains("validation"));
            let debug = validation && iexts.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME));
            let app = vk::ApplicationInfo::default().application_name(c"ngx-interop").api_version(vk::API_VERSION_1_4);
            let mut inames: Vec<CString> = Vec::new();
            if debug {
                inames.push(ash::ext::debug_utils::NAME.to_owned());
            }
            for e in extra_instance {
                if !inames.contains(e) {
                    inames.push(e.clone());
                }
            }
            let iptrs: Vec<_> = inames.iter().map(|n| n.as_ptr()).collect();
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&iptrs), None)
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
            let g = vk::QueueFlags::GRAPHICS;
            let c = vk::QueueFlags::COMPUTE;
            let render = qf.iter().position(|q| q.queue_flags.contains(g | c)).unwrap() as u32;
            let compute = qf.iter().position(|q| q.queue_flags.contains(c) && !q.queue_flags.contains(g)).unwrap() as u32;
            let families = [render, compute];

            let mut f_heap = vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default().descriptor_heap(true);
            let mut f_untyped = vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR::default().shader_untyped_pointers(true);
            let mut f_unified = vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default().unified_image_layouts(true);
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .scalar_block_layout(true)
                .timeline_semaphore(true)
                .shader_float16(true)
                .shader_int8(true)
                .host_query_reset(true);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true).dynamic_rendering(true).maintenance4(true);
            let mut f14 = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true).push_descriptor(true);
            let core = vk::PhysicalDeviceFeatures::default()
                .shader_int64(true)
                .shader_int16(true)
                .shader_storage_image_read_without_format(true)
                .shader_storage_image_write_without_format(true);
            let mut f2 = vk::PhysicalDeviceFeatures2::default().features(core);
            let prio = [1.0f32];
            let qinfo: Vec<_> = families.iter().map(|&f| vk::DeviceQueueCreateInfo::default().queue_family_index(f).queue_priorities(&prio)).collect();
            let mut dnames: Vec<CString> = vec![ash::ext::memory_budget::NAME.to_owned()];
            if heaps {
                dnames.push(ash::ext::descriptor_heap::NAME.to_owned());
                dnames.push(ash::khr::shader_untyped_pointers::NAME.to_owned());
            }
            if unified {
                dnames.push(c"VK_KHR_unified_image_layouts".to_owned());
            }
            for e in extra_device {
                // Core 1.2 buffer device address replaces the EXT that NGX lists.
                if e.as_c_str() != c"VK_EXT_buffer_device_address" && !dnames.contains(e) {
                    dnames.push(e.clone());
                }
            }
            let dptrs: Vec<_> = dnames.iter().map(|n| n.as_ptr()).collect();
            let mut info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qinfo)
                .enabled_extension_names(&dptrs)
                .push(&mut f2)
                .push(&mut f11)
                .push(&mut f12)
                .push(&mut f13)
                .push(&mut f14);
            if heaps {
                info = info.push(&mut f_heap).push(&mut f_untyped);
            }
            if unified {
                info = info.push(&mut f_unified);
            }
            let device = instance.create_device(pdev, &info, None).expect("device");
            let queues = families.map(|f| device.get_device_queue(f, 0));
            let mut drv = vk::PhysicalDeviceDriverProperties::default();
            let mut heap_props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut drv).push(&mut heap_props);
            instance.get_physical_device_properties2(pdev, &mut props2);
            let props = props2.properties;
            heap_props.p_next = std::ptr::null_mut();
            let name = props.device_name_as_c_str().unwrap().to_string_lossy().into_owned();
            let driver = drv.driver_info_as_c_str().unwrap().to_string_lossy().into_owned();
            let mem = instance.get_physical_device_memory_properties(pdev);
            let heap = ash::ext::descriptor_heap::Device::load(&instance, &device);

            let mut gpu = Gpu {
                entry,
                instance,
                pdev,
                device,
                families,
                queues,
                mem,
                ts_period_ns: f64::from(props.limits.timestamp_period),
                name,
                driver,
                heap,
                heap_props,
                heaps,
                validation: debug,
                unified,
                resource_heap: Buf { buffer: vk::Buffer::null(), addr: 0, ptr: std::ptr::null_mut(), size: 0 },
                sampler_heap: Buf { buffer: vk::Buffer::null(), addr: 0, ptr: std::ptr::null_mut(), size: 0 },
                resource_bind: vk::BindHeapInfoEXT::default(),
                sampler_bind: vk::BindHeapInfoEXT::default(),
                next_slot: std::cell::Cell::new(0),
                roots: Buf { buffer: vk::Buffer::null(), addr: 0, ptr: std::ptr::null_mut(), size: 0 },
                next_root: std::cell::Cell::new(0),
            };
            if heaps {
                gpu.create_heaps();
            }
            gpu.roots = gpu.buffer(ROOT_SLOTS * ROOT_STRIDE, Mem::Upload);
            gpu
        }
    }

    unsafe fn create_heaps(&mut self) {
        unsafe {
            let p = self.heap_props;
            let used = HEAP_SLOTS * p.image_descriptor_size;
            let reserved_offset = align_up(used, p.image_descriptor_alignment.max(p.buffer_descriptor_alignment));
            let size = reserved_offset + p.min_resource_heap_reserved_range;
            self.resource_heap = self.heap_buffer(size, p.resource_heap_alignment);
            std::ptr::write_bytes(self.resource_heap.ptr, 0, size as usize);
            let sampler_reserved = align_up(p.sampler_descriptor_size, p.sampler_descriptor_alignment);
            let ssize = sampler_reserved + p.min_sampler_heap_reserved_range;
            self.sampler_heap = self.heap_buffer(ssize, p.sampler_heap_alignment);
            std::ptr::write_bytes(self.sampler_heap.ptr, 0, ssize as usize);
            let bind = |addr: u64, reserved_offset: u64, reserved: u64| vk::BindHeapInfoEXT {
                heap_range: vk::DeviceAddressRangeEXT { address: addr, size: reserved_offset + reserved },
                reserved_range_offset: reserved_offset,
                reserved_range_size: reserved,
                ..Default::default()
            };
            self.resource_bind = bind(self.resource_heap.addr, reserved_offset, p.min_resource_heap_reserved_range);
            self.sampler_bind = bind(self.sampler_heap.addr, sampler_reserved, p.min_sampler_heap_reserved_range);
        }
    }

    /// Host-visible descriptor-heap buffer whose address is aligned to `align`.
    unsafe fn heap_buffer(&self, size: u64, align: u64) -> Buf {
        unsafe {
            let usage = vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size + align).usage(usage), None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = self.memory_type(req.memory_type_bits, flags, vk::MemoryPropertyFlags::empty());
            let mut fl = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push(&mut fl), None).unwrap();
            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let base = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            let addr = align_up(base, align);
            let ptr: *mut u8 = self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap().cast();
            Buf { buffer, addr, ptr: ptr.add((addr - base) as usize), size }
        }
    }

    pub fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags, not: vk::MemoryPropertyFlags) -> u32 {
        (0..self.mem.memory_type_count)
            .find(|&i| {
                let f = self.mem.memory_types[i as usize].property_flags;
                bits & (1 << i) != 0 && f.contains(want) && !f.intersects(not)
            })
            .expect("memory type")
    }

    pub unsafe fn buffer(&self, size: u64, kind: Mem) -> Buf {
        unsafe {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let dl = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = match kind {
                Mem::Upload => self.memory_type(req.memory_type_bits, dl | hv, vk::MemoryPropertyFlags::empty()),
                Mem::Device => self.memory_type(req.memory_type_bits, dl, vk::MemoryPropertyFlags::HOST_VISIBLE),
                Mem::Readback => self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED, dl),
            };
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = self
                .device
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push(&mut flags), None)
                .expect("allocate");
            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let addr = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            let ptr = if kind == Mem::Device { std::ptr::null_mut() } else { self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap().cast() };
            Buf { buffer, addr, ptr, size }
        }
    }

    /// Image in a resource-heap storage slot (descriptor layout GENERAL).
    pub unsafe fn image(&self, extent: [u32; 2], format: vk::Format) -> Img {
        unsafe {
            let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
            let ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: extent[0], height: extent[1], depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = self.device.create_image(&ci, None).unwrap();
            let req = self.device.get_image_memory_requirements(image);
            let ty = self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::HOST_VISIBLE);
            let memory = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None).expect("allocate");
            self.device.bind_image_memory(image, memory, 0).unwrap();
            let view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(color_range()),
                    None,
                )
                .unwrap();
            let slot = self.next_slot.get();
            if !self.heaps {
                return Img { image, view, format, extent, slot };
            }
            assert!((slot as u64) < HEAP_SLOTS, "heap full");
            self.next_slot.set(slot + 1);
            let view_info = vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(color_range());
            let desc = vk::ImageDescriptorInfoEXT::default().view(&view_info).layout(vk::ImageLayout::GENERAL);
            let infos = [vk::ResourceDescriptorInfoEXT::default().ty(vk::DescriptorType::STORAGE_IMAGE).data(vk::ResourceDescriptorDataEXT { p_image: &desc })];
            let size = self.heap_props.image_descriptor_size;
            let dst = [vk::HostAddressRangeEXT::default().address(std::slice::from_raw_parts_mut(self.resource_heap.ptr.add((slot as u64 * size) as usize), size as usize))];
            self.heap.write_resource_descriptors(&infos, &dst).expect("vkWriteResourceDescriptorsEXT");
            Img { image, view, format, extent, slot }
        }
    }

    /// Heap-mode compute pipeline; set 0 binding 0 (the root) maps to the pushed address.
    pub unsafe fn pipeline(&self, code: &[u32], entry: &CStr) -> vk::Pipeline {
        unsafe {
            let mut module = vk::ShaderModuleCreateInfo::default().code(code);
            let mappings = [vk::DescriptorSetAndBindingMappingEXT {
                descriptor_set: 0,
                first_binding: 0,
                binding_count: 1,
                resource_mask: vk::SpirvResourceTypeFlagsEXT::UNIFORM_BUFFER | vk::SpirvResourceTypeFlagsEXT::READ_ONLY_STORAGE_BUFFER,
                source: vk::DescriptorMappingSourceEXT::PUSH_ADDRESS,
                source_data: vk::DescriptorMappingSourceDataEXT { push_address_offset: 0 },
                ..Default::default()
            }];
            let mut mapping_info = vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&mappings);
            let mut flags = vk::PipelineCreateFlags2CreateInfo::default().flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).name(entry).push(&mut module).push(&mut mapping_info);
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).push(&mut flags);
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e).expect("compute pipeline")[0]
        }
    }

    pub unsafe fn bind_heaps(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.heap.cmd_bind_resource_heap(cmd, &self.resource_bind);
            self.heap.cmd_bind_sampler_heap(cmd, &self.sampler_bind);
        }
    }

    /// Writes `root` into the next upload slot and pushes its address.
    pub unsafe fn push_root<T: Copy>(&self, cmd: vk::CommandBuffer, root: &T) {
        unsafe {
            assert!(size_of::<T>() as u64 <= ROOT_STRIDE);
            let i = self.next_root.get();
            self.next_root.set(i + 1);
            let off = (i % ROOT_SLOTS) * ROOT_STRIDE;
            std::ptr::copy_nonoverlapping((root as *const T).cast::<u8>(), self.roots.ptr.add(off as usize), size_of::<T>());
            let addr = self.roots.addr + off;
            let info = vk::PushDataInfoEXT {
                offset: 0,
                data: vk::HostAddressRangeConstEXT::default().address(std::slice::from_raw_parts((&addr as *const u64).cast::<u8>(), 8)),
                ..Default::default()
            };
            self.heap.cmd_push_data(cmd, &info);
        }
    }

    pub unsafe fn dispatch(&self, cmd: vk::CommandBuffer, pipeline: vk::Pipeline, groups: [u32; 2]) {
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_dispatch(cmd, groups[0], groups[1], 1);
        }
    }

    /// Full memory barrier, all commands to all commands.
    pub unsafe fn barrier(&self, cmd: vk::CommandBuffer) {
        unsafe {
            let mb = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
            self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&mb));
        }
    }

    pub unsafe fn layouts(&self, cmd: vk::CommandBuffer, images: &[vk::Image], from: vk::ImageLayout, to: vk::ImageLayout) {
        unsafe {
            let ib: Vec<_> = images
                .iter()
                .map(|&image| {
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                        .old_layout(from)
                        .new_layout(to)
                        .image(image)
                        .subresource_range(color_range())
                })
                .collect();
            self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&ib));
        }
    }

    pub unsafe fn command_pool(&self, family: u32) -> vk::CommandPool {
        unsafe { self.device.create_command_pool(&vk::CommandPoolCreateInfo::default().flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER).queue_family_index(family), None).unwrap() }
    }

    pub unsafe fn command_buffer(&self, pool: vk::CommandPool) -> vk::CommandBuffer {
        unsafe {
            self.device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
                .unwrap()[0]
        }
    }

    pub unsafe fn begin(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
        }
    }

    /// Ends, submits on `queue` and waits for idle.
    pub unsafe fn submit_wait(&self, queue: vk::Queue, cmd: vk::CommandBuffer) {
        unsafe {
            self.device.end_command_buffer(cmd).unwrap();
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit = [vk::SubmitInfo2::default().command_buffer_infos(&cmds)];
            self.device.queue_submit2(queue, &submit, vk::Fence::null()).expect("submit");
            self.device.queue_wait_idle(queue).expect("queue wait (device lost?)");
        }
    }

    pub unsafe fn copy_to_buffer(&self, cmd: vk::CommandBuffer, img: &Img, buf: &Buf) {
        unsafe {
            let region = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D { width: img.extent[0], height: img.extent[1], depth: 1 })];
            self.device.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::GENERAL, buf.buffer, &region);
        }
    }

    /// Device-local heap usage in bytes (VK_EXT_memory_budget).
    pub unsafe fn vram_used(&self) -> u64 {
        unsafe {
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut p = vk::PhysicalDeviceMemoryProperties2::default().push(&mut budget);
            self.instance.get_physical_device_memory_properties2(self.pdev, &mut p);
            let n = p.memory_properties.memory_heap_count as usize;
            (0..n).filter(|&i| p.memory_properties.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|i| budget.heap_usage[i]).sum()
        }
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)
}

/// A classic (non-heap) compute pipeline with one storage-image descriptor set: the state NGX
/// itself uses, recorded to test whether it restores what a heap bind changed.
pub struct Classic {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
}

impl Classic {
    pub unsafe fn new(gpu: &Gpu, code: &[u32], scratch: &Img) -> Classic {
        unsafe {
            let d = &gpu.device;
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
            let sets = [set_layout];
            let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets), None).unwrap();
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 }];
            let pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).unwrap();
            let set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sets)).unwrap()[0];
            let info = [vk::DescriptorImageInfo::default().image_view(scratch.view).image_layout(vk::ImageLayout::GENERAL)];
            let write = [vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&info)];
            d.update_descriptor_sets(&write, &[]);
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"classic_main");
            let pipeline = d
                .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None)
                .map_err(|(_, e)| e)
                .expect("classic pipeline")[0];
            Classic { pipeline, layout, set }
        }
    }

    /// `mode`: "pipeline" binds the pipeline, "set" also the descriptor set, "full" also
    /// dispatches one workgroup.
    pub unsafe fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, mode: &str) {
        unsafe {
            gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            if mode != "pipeline" {
                gpu.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
            }
            if mode == "full" {
                gpu.device.cmd_dispatch(cmd, 1, 1, 1);
            }
        }
    }
}
