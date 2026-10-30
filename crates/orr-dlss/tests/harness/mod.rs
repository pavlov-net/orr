//! A raw-ash device with descriptor heaps, NGX's extensions and the validation layer when
//! `VK_INSTANCE_LAYERS` names it; images in GENERAL with upload and readback.

#![allow(dead_code, unreachable_pub)]

use ash::vk::{self, TaggedStructure};
use orr_dlss::Uuid;
use orr_dlss::raw::{self, Image};
use std::ffi::{CStr, CString, c_void};
use std::sync::atomic::{AtomicU32, Ordering};

pub const PROJECT_ID: Uuid = Uuid::from_u128(0x6a3c1e0d_9b2f_4c7a_8e15_3d0f7b9a2c41);

static MESSAGES: AtomicU32 = AtomicU32::new(0);
static NGX_HAZARDS: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn on_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the layer passes valid callback data.
    let data = unsafe { &*data };
    let text = |p: *const std::ffi::c_char| {
        if p.is_null() {
            String::new()
        } else {
            // SAFETY: NUL-terminated strings from the layer.
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    };
    let id = text(data.p_message_id_name);
    // The loader and layer-settings reports are not about this program.
    if id.starts_with("Loader") || id.contains("Layer-Settings") {
        return vk::FALSE;
    }
    // NGX's first evaluate of a feature clears its own history image right after its own
    // layout transition, without a barrier to the transfer stage.
    // SAFETY: the layer passes `object_count` valid entries.
    let objects = unsafe { std::slice::from_raw_parts(data.p_objects, data.object_count as usize) };
    if id.starts_with("SYNC-HAZARD")
        && objects
            .iter()
            .any(|o| text(o.p_object_name).starts_with("nv.ngx."))
    {
        NGX_HAZARDS.fetch_add(1, Ordering::Relaxed);
        return vk::FALSE;
    }
    MESSAGES.fetch_add(1, Ordering::Relaxed);
    log::error!("[{severity:?}] {id}: {}", text(data.p_message));
    #[allow(clippy::print_stderr)]
    {
        eprintln!("validation [{severity:?}] {id}: {}", text(data.p_message));
    }
    vk::FALSE
}

pub struct Buffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub address: u64,
    pub ptr: *mut u8,
    pub size: u64,
}

pub struct Gpu {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub validation: bool,
    messenger: vk::DebugUtilsMessengerEXT,
    heap_fn: ash::ext::descriptor_heap::Device,
    resource_heap: Buffer,
    sampler_heap: Buffer,
    resource_bind: vk::BindHeapInfoEXT<'static>,
    sampler_bind: vk::BindHeapInfoEXT<'static>,
    pool: vk::CommandPool,
    pub cmd: vk::CommandBuffer,
    images: Vec<(vk::Image, vk::ImageView, vk::DeviceMemory)>,
    buffers: Vec<Buffer>,
}

impl Gpu {
    /// A device on the first discrete GPU with NGX's extensions when `ngx` is set.
    pub fn new(ngx: bool) -> Gpu {
        // SAFETY: test setup; every create call gets valid infos, and the objects are
        // destroyed in `Drop`.
        unsafe {
            let entry = ash::Entry::load().expect("Vulkan loader");
            let validation =
                std::env::var("VK_INSTANCE_LAYERS").is_ok_and(|v| v.contains("KHRONOS_validation"));
            let mut inames: Vec<CString> = if ngx {
                raw::instance_extensions(PROJECT_ID).unwrap()
            } else {
                Vec::new()
            };
            if validation {
                inames.push(ash::ext::debug_utils::NAME.to_owned());
                inames.push(ash::ext::layer_settings::NAME.to_owned());
            }
            let iptrs: Vec<_> = inames.iter().map(|n| n.as_ptr()).collect();
            let app = vk::ApplicationInfo::default()
                .application_name(c"orr-dlss tests")
                .api_version(vk::API_VERSION_1_4);
            let on = [vk::TRUE];
            let settings = [vk::LayerSettingEXT::default()
                .layer_name(c"VK_LAYER_KHRONOS_validation")
                .setting_name(c"validate_sync")
                .ty(vk::LayerSettingTypeEXT::BOOL32)
                .values(&on)];
            let mut layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
            let messenger = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(on_message));
            let mut info = vk::InstanceCreateInfo::default()
                .application_info(&app)
                .enabled_extension_names(&iptrs);
            if validation {
                info = info.push(&mut layer_settings);
            }
            let instance = entry.create_instance(&info, None).expect("instance");
            let mut messenger_handle = vk::DebugUtilsMessengerEXT::null();
            if validation {
                let du = ash::ext::debug_utils::Instance::load(&entry, &instance);
                messenger_handle = du
                    .create_debug_utils_messenger(&messenger, None)
                    .expect("messenger");
            }
            let physical_device = instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .find(|&p| {
                    instance.get_physical_device_properties(p).device_type
                        == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .expect("discrete GPU");
            let family = instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .position(|q| {
                    q.queue_flags
                        .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .unwrap() as u32;

            let mut dnames: Vec<CString> = vec![
                ash::ext::descriptor_heap::NAME.to_owned(),
                ash::khr::shader_untyped_pointers::NAME.to_owned(),
            ];
            if ngx {
                for e in raw::device_extensions(&instance, physical_device, PROJECT_ID).unwrap() {
                    if !dnames.contains(&e) {
                        dnames.push(e);
                    }
                }
            }
            let dptrs: Vec<_> = dnames.iter().map(|n| n.as_ptr()).collect();
            let mut f_heap =
                vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default().descriptor_heap(true);
            let mut f_untyped = vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR::default()
                .shader_untyped_pointers(true);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .timeline_semaphore(true);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .maintenance4(true);
            let mut f14 = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true);
            if ngx {
                raw::enable_features(&mut f14);
            }
            let prio = [1.0f32];
            let qinfo = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&prio)];
            let info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qinfo)
                .enabled_extension_names(&dptrs)
                .push(&mut f12)
                .push(&mut f13)
                .push(&mut f14)
                .push(&mut f_heap)
                .push(&mut f_untyped);
            let device = instance
                .create_device(physical_device, &info, None)
                .expect("device");
            let queue = device.get_device_queue(family, 0);
            let heap_fn = ash::ext::descriptor_heap::Device::load(&instance, &device);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(family),
                    None,
                )
                .unwrap();
            let cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];

            let mut heap_props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut heap_props);
            instance.get_physical_device_properties2(physical_device, &mut props2);
            let p = heap_props;
            let mut gpu = Gpu {
                entry,
                instance,
                physical_device,
                device,
                queue,
                validation,
                messenger: messenger_handle,
                heap_fn,
                resource_heap: Buffer::null(),
                sampler_heap: Buffer::null(),
                resource_bind: vk::BindHeapInfoEXT::default(),
                sampler_bind: vk::BindHeapInfoEXT::default(),
                pool,
                cmd,
                images: Vec::new(),
                buffers: Vec::new(),
            };
            // Heaps hold only their reserved ranges: binding them is what the tests need.
            let reserved = align_up(
                64 * p.image_descriptor_size,
                p.image_descriptor_alignment
                    .max(p.buffer_descriptor_alignment),
            );
            gpu.resource_heap = gpu.heap_buffer(
                reserved + p.min_resource_heap_reserved_range,
                p.resource_heap_alignment,
            );
            let sampler_reserved =
                align_up(p.sampler_descriptor_size, p.sampler_descriptor_alignment);
            gpu.sampler_heap = gpu.heap_buffer(
                sampler_reserved + p.min_sampler_heap_reserved_range,
                p.sampler_heap_alignment,
            );
            let bind = |address: u64, offset: u64, size: u64| vk::BindHeapInfoEXT {
                heap_range: vk::DeviceAddressRangeEXT {
                    address,
                    size: offset + size,
                },
                reserved_range_offset: offset,
                reserved_range_size: size,
                ..Default::default()
            };
            gpu.resource_bind = bind(
                gpu.resource_heap.address,
                reserved,
                p.min_resource_heap_reserved_range,
            );
            gpu.sampler_bind = bind(
                gpu.sampler_heap.address,
                sampler_reserved,
                p.min_sampler_heap_reserved_range,
            );
            gpu
        }
    }

    pub fn raw(&self) -> raw::Vulkan<'_> {
        raw::Vulkan {
            entry: &self.entry,
            instance: &self.instance,
            physical_device: self.physical_device,
            device: &self.device,
        }
    }

    /// Validation messages reported so far, across every device in the process, leaving out
    /// synchronisation hazards on NGX's own resources.
    pub fn validation_messages() -> u32 {
        MESSAGES.load(Ordering::Relaxed)
    }

    /// Synchronisation hazards on NGX's own resources reported so far.
    pub fn ngx_hazards() -> u32 {
        NGX_HAZARDS.load(Ordering::Relaxed)
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> u32 {
        // SAFETY: a query on a live physical device.
        let mem = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        (0..mem.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0 && mem.memory_types[i as usize].property_flags.contains(want)
            })
            .expect("memory type")
    }

    unsafe fn allocate(
        &self,
        req: vk::MemoryRequirements,
        want: vk::MemoryPropertyFlags,
        address: bool,
    ) -> vk::DeviceMemory {
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(self.memory_type(req.memory_type_bits, want));
        if address {
            info = info.push(&mut flags);
        }
        // SAFETY: valid info on a live device.
        unsafe { self.device.allocate_memory(&info, None) }.expect("allocate")
    }

    unsafe fn heap_buffer(&self, size: u64, align: u64) -> Buffer {
        // SAFETY: test setup on a live device.
        unsafe {
            let usage = vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size + align)
                        .usage(usage),
                    None,
                )
                .unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let memory = self.allocate(
                req,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                true,
            );
            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let base = self
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            let address = align_up(base, align);
            let ptr: *mut u8 = self
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .unwrap()
                .cast();
            std::ptr::write_bytes(ptr, 0, (size + align) as usize);
            Buffer {
                buffer,
                memory,
                address,
                ptr,
                size,
            }
        }
    }

    /// A host-visible transfer buffer, freed with the device.
    pub fn buffer(&mut self, size: u64) -> usize {
        // SAFETY: test setup on a live device.
        unsafe {
            let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(usage),
                    None,
                )
                .unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let memory = self.allocate(
                req,
                vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT
                    | vk::MemoryPropertyFlags::HOST_CACHED,
                false,
            );
            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let ptr = self
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .unwrap()
                .cast();
            self.buffers.push(Buffer {
                buffer,
                memory,
                address: 0,
                ptr,
                size,
            });
            self.buffers.len() - 1
        }
    }

    /// A device-local 2D image left in GENERAL by the next submission's first commands.
    pub fn image(&mut self, extent: [u32; 2], format: vk::Format) -> Image {
        // SAFETY: test setup on a live device.
        unsafe {
            let usage = vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST;
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: extent[0],
                    height: extent[1],
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(usage);
            let image = self.device.create_image(&info, None).unwrap();
            let memory = self.allocate(
                self.device.get_image_memory_requirements(image),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                false,
            );
            self.device.bind_image_memory(image, memory, 0).unwrap();
            let view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(color_range()),
                    None,
                )
                .unwrap();
            self.images.push((image, view, memory));
            Image {
                image,
                view,
                format,
                extent,
            }
        }
    }

    pub fn begin(&self) {
        // SAFETY: the command buffer is not pending (every submit waits).
        unsafe {
            self.device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.device
                .begin_command_buffer(
                    self.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
    }

    pub fn submit_wait(&self) {
        // SAFETY: the command buffer is recording; the queue is idle afterwards.
        unsafe {
            self.device.end_command_buffer(self.cmd).unwrap();
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd)];
            let submit = [vk::SubmitInfo2::default().command_buffer_infos(&cmds)];
            self.device
                .queue_submit2(self.queue, &submit, vk::Fence::null())
                .expect("submit");
            self.device.queue_wait_idle(self.queue).expect("queue wait");
        }
    }

    /// Binds the resource and sampler heaps.
    pub fn bind_heaps(&self) {
        // SAFETY: recording; the heaps live as long as the device.
        unsafe {
            self.heap_fn
                .cmd_bind_resource_heap(self.cmd, &self.resource_bind);
            self.heap_fn
                .cmd_bind_sampler_heap(self.cmd, &self.sampler_bind);
        }
    }

    /// Moves images from UNDEFINED to GENERAL.
    pub fn to_general(&self, images: &[&Image]) {
        let barriers: Vec<_> = images
            .iter()
            .map(|i| {
                vk::ImageMemoryBarrier2::default()
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(i.image)
                    .subresource_range(color_range())
            })
            .collect();
        // SAFETY: recording.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
        }
    }

    /// Every command before against every command after.
    pub fn barrier(&self) {
        let mb = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
        // SAFETY: recording.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().memory_barriers(&mb),
            );
        }
    }

    pub fn clear(&self, image: &Image, value: [f32; 4]) {
        // SAFETY: recording; the image is in GENERAL.
        unsafe {
            self.device.cmd_clear_color_image(
                self.cmd,
                image.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue { float32: value },
                &[color_range()],
            );
        }
    }

    fn copy_region(image: &Image) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: image.extent[0],
                height: image.extent[1],
                depth: 1,
            })
    }

    /// Writes `bytes` into transfer buffer `staging` and records its copy into `image`.
    pub fn upload(&self, staging: usize, image: &Image, bytes: &[u8]) {
        let b = &self.buffers[staging];
        assert!(bytes.len() as u64 <= b.size);
        // SAFETY: the buffer is mapped, large enough and not in use (every submit waits).
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.ptr, bytes.len());
            self.device.cmd_copy_buffer_to_image(
                self.cmd,
                b.buffer,
                image.image,
                vk::ImageLayout::GENERAL,
                &[Self::copy_region(image)],
            );
        }
    }

    pub fn download(&self, image: &Image, readback: usize) {
        // SAFETY: recording; the image is in GENERAL.
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.cmd,
                image.image,
                vk::ImageLayout::GENERAL,
                self.buffers[readback].buffer,
                &[Self::copy_region(image)],
            );
        }
    }

    pub fn bytes(&self, buffer: usize) -> &[u8] {
        let b = &self.buffers[buffer];
        // SAFETY: mapped for the buffer's lifetime; read after the submit has completed.
        unsafe { std::slice::from_raw_parts(b.ptr, b.size as usize) }
    }
}

impl Buffer {
    fn null() -> Buffer {
        Buffer {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            address: 0,
            ptr: std::ptr::null_mut(),
            size: 0,
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: the queue is idle; every object was created on this device.
        unsafe {
            self.device.device_wait_idle().unwrap();
            for (image, view, memory) in self.images.drain(..) {
                self.device.destroy_image_view(view, None);
                self.device.destroy_image(image, None);
                self.device.free_memory(memory, None);
            }
            let heaps = [
                std::mem::replace(&mut self.resource_heap, Buffer::null()),
                std::mem::replace(&mut self.sampler_heap, Buffer::null()),
            ];
            for b in self.buffers.drain(..).chain(heaps) {
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            if self.messenger != vk::DebugUtilsMessengerEXT::null() {
                ash::ext::debug_utils::Instance::load(&self.entry, &self.instance)
                    .destroy_debug_utils_messenger(self.messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a.max(1)) * a.max(1)
}

pub fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
}

pub fn f16_values(bytes: &[u8], count: usize) -> Vec<f32> {
    bytes[..count * 2]
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}
