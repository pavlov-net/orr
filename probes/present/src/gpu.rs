//! Instance, surface device and helpers: render queue (family 0) and loading queue (the
//! compute-only family), buffers by address, images, compute/mesh/compose pipelines.

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
        let short: String = m.chars().take(600).collect();
        eprintln!("VVL {id} x{n}: {short}");
    }
    (e, w)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mem {
    /// Device-local, host-visible, coherent (resizable BAR).
    Upload,
    Device,
}

#[derive(Clone, Copy)]
pub struct Buf {
    pub buffer: vk::Buffer,
    pub addr: u64,
    pub ptr: *mut u8,
    pub size: u64,
}
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

#[derive(Clone, Copy)]
pub struct Img {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: [u32; 2],
}

/// Raw VK_EXT_present_timing entry points (ash has no wrapper for them at this rev).
pub struct Timing {
    pub set_queue_size: vk::PFN_vkSetSwapchainPresentTimingQueueSizeEXT,
    pub properties: vk::PFN_vkGetSwapchainTimingPropertiesEXT,
    pub domains: vk::PFN_vkGetSwapchainTimeDomainPropertiesEXT,
    pub past: vk::PFN_vkGetPastPresentationTimingEXT,
}

pub struct Gpu {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_i: ash::khr::surface::Instance,
    pub caps2_i: ash::khr::get_surface_capabilities2::Instance,
    pub win32_i: ash::khr::win32_surface::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub families: [u32; 2],
    pub queues: [vk::Queue; 2],
    pub mem: vk::PhysicalDeviceMemoryProperties,
    pub ts_period_ns: f64,
    pub name: String,
    pub driver: String,
    pub swapchain: ash::khr::swapchain::Device,
    pub latency: ash::nv::low_latency2::Device,
    pub wait2: ash::khr::present_wait2::Device,
    pub hdr: ash::ext::hdr_metadata::Device,
    pub mesh: ash::ext::mesh_shader::Device,
    pub push_desc: ash::khr::push_descriptor::Device,
    pub timing: Timing,
    pub layout: vk::PipelineLayout,
    pub mesh_layout: vk::PipelineLayout,
    pub compose_layout: vk::PipelineLayout,
    pub validation: bool,
}
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

const DEVICE_EXTENSIONS: &[&CStr] = &[
    c"VK_KHR_swapchain",
    c"VK_KHR_swapchain_maintenance1",
    c"VK_KHR_present_id",
    c"VK_KHR_present_id2",
    c"VK_KHR_present_wait2",
    c"VK_NV_low_latency2",
    c"VK_NV_present_metering",
    c"VK_EXT_present_timing",
    c"VK_KHR_calibrated_timestamps",
    c"VK_EXT_hdr_metadata",
    c"VK_EXT_mesh_shader",
];

const INSTANCE_EXTENSIONS: &[&CStr] = &[
    c"VK_KHR_surface",
    c"VK_KHR_win32_surface",
    c"VK_KHR_get_surface_capabilities2",
    c"VK_KHR_surface_maintenance1",
    c"VK_EXT_swapchain_colorspace",
];

fn load<T>(gpa: vk::PFN_vkGetDeviceProcAddr, device: vk::Device, name: &CStr) -> T {
    unsafe {
        let f = gpa(device, name.as_ptr()).unwrap_or_else(|| panic!("missing {name:?}"));
        std::mem::transmute_copy(&f)
    }
}

impl Gpu {
    /// `extra_instance` and `extra_device` are NGX's requirements.
    pub unsafe fn new(extra_instance: &[CString], extra_device: &[CString]) -> Gpu {
        unsafe {
            let entry = ash::Entry::load().expect("Vulkan loader");
            let iexts = entry.enumerate_instance_extension_properties(None).unwrap();
            let validation = std::env::var("VK_INSTANCE_LAYERS").is_ok_and(|v| v.contains("validation"));
            let debug = validation && iexts.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::ext::debug_utils::NAME));
            let app = vk::ApplicationInfo::default().application_name(c"present").api_version(vk::API_VERSION_1_4);
            let mut inames: Vec<CString> = INSTANCE_EXTENSIONS.iter().map(|&n| n.to_owned()).collect();
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
            let loading = qf.iter().position(|q| q.queue_flags.contains(c) && !q.queue_flags.contains(g)).unwrap() as u32;
            let families = [render, loading];

            let mut f_id = vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);
            let mut f_id2 = vk::PhysicalDevicePresentId2FeaturesKHR::default().present_id2(true);
            let mut f_wait2 = vk::PhysicalDevicePresentWait2FeaturesKHR::default().present_wait2(true);
            let mut f_meter = vk::PhysicalDevicePresentMeteringFeaturesNV::default().present_metering(true);
            let mut f_timing = vk::PhysicalDevicePresentTimingFeaturesEXT::default()
                .present_timing(true)
                .present_at_relative_time(true)
                .present_at_absolute_time(true);
            let mut f_sm1 = vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR::default().swapchain_maintenance1(true);
            let mut f_mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true);
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
            let mut dnames: Vec<CString> = DEVICE_EXTENSIONS.iter().map(|&n| n.to_owned()).collect();
            for e in extra_device {
                // Core 1.2 buffer device address replaces the EXT that NGX lists.
                if e.as_c_str() != c"VK_EXT_buffer_device_address" && !dnames.contains(e) {
                    dnames.push(e.clone());
                }
            }
            let dptrs: Vec<_> = dnames.iter().map(|n| n.as_ptr()).collect();
            let info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qinfo)
                .enabled_extension_names(&dptrs)
                .push(&mut f2)
                .push(&mut f11)
                .push(&mut f12)
                .push(&mut f13)
                .push(&mut f14)
                .push(&mut f_id)
                .push(&mut f_id2)
                .push(&mut f_wait2)
                .push(&mut f_meter)
                .push(&mut f_timing)
                .push(&mut f_sm1)
                .push(&mut f_mesh);
            let device = instance.create_device(pdev, &info, None).expect("device");
            let queues = families.map(|f| device.get_device_queue(f, 0));
            let mut drv = vk::PhysicalDeviceDriverProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut drv);
            instance.get_physical_device_properties2(pdev, &mut props2);
            let props = props2.properties;
            let name = props.device_name_as_c_str().unwrap().to_string_lossy().into_owned();
            let driver = drv.driver_info_as_c_str().unwrap().to_string_lossy().into_owned();

            let all = vk::ShaderStageFlags::COMPUTE;
            let ranges = [vk::PushConstantRange::default().stage_flags(all).size(128)];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges), None).unwrap();
            let mranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT).size(16)];
            let mesh_layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&mranges), None).unwrap();
            let bindings = [0, 1].map(|b| {
                vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(all)
            });
            let compose_set = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR).bindings(&bindings),
                    None,
                )
                .unwrap();
            let sets = [compose_set];
            let cranges = [vk::PushConstantRange::default().stage_flags(all).size(32)];
            let compose_layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets).push_constant_ranges(&cranges), None).unwrap();

            let gdpa = instance.fp_v1_0().get_device_proc_addr;
            let dh = device.handle();
            let timing = Timing {
                set_queue_size: load(gdpa, dh, c"vkSetSwapchainPresentTimingQueueSizeEXT"),
                properties: load(gdpa, dh, c"vkGetSwapchainTimingPropertiesEXT"),
                domains: load(gdpa, dh, c"vkGetSwapchainTimeDomainPropertiesEXT"),
                past: load(gdpa, dh, c"vkGetPastPresentationTimingEXT"),
            };
            Gpu {
                surface_i: ash::khr::surface::Instance::load(&entry, &instance),
                caps2_i: ash::khr::get_surface_capabilities2::Instance::load(&entry, &instance),
                win32_i: ash::khr::win32_surface::Instance::load(&entry, &instance),
                swapchain: ash::khr::swapchain::Device::load(&instance, &device),
                latency: ash::nv::low_latency2::Device::load(&instance, &device),
                wait2: ash::khr::present_wait2::Device::load(&instance, &device),
                hdr: ash::ext::hdr_metadata::Device::load(&instance, &device),
                mesh: ash::ext::mesh_shader::Device::load(&instance, &device),
                push_desc: ash::khr::push_descriptor::Device::load(&instance, &device),
                mem: instance.get_physical_device_memory_properties(pdev),
                ts_period_ns: f64::from(props.limits.timestamp_period),
                timing,
                entry,
                instance,
                pdev,
                device,
                families,
                queues,
                name,
                driver,
                layout,
                mesh_layout,
                compose_layout,
                validation: debug,
            }
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
            let fam = self.families;
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let ci = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&fam);
            let buffer = self.device.create_buffer(&ci, None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let dl = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = match kind {
                Mem::Upload => self.memory_type(req.memory_type_bits, dl | hv, vk::MemoryPropertyFlags::empty()),
                Mem::Device => self.memory_type(req.memory_type_bits, dl, vk::MemoryPropertyFlags::HOST_VISIBLE),
            };
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = self
                .device
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push(&mut flags), None)
                .expect("allocate");
            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let addr = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            let ptr = if kind == Mem::Upload { self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap().cast() } else { std::ptr::null_mut() };
            Buf { buffer, addr, ptr, size }
        }
    }

    pub unsafe fn image(&self, extent: [u32; 2], format: vk::Format, usage: vk::ImageUsageFlags) -> Img {
        unsafe {
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
            let view = self.view(image, format);
            Img { image, view, format, extent }
        }
    }

    pub unsafe fn view(&self, image: vk::Image, format: vk::Format) -> vk::ImageView {
        unsafe {
            self.device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(color_range()),
                    None,
                )
                .unwrap()
        }
    }

    pub unsafe fn compute_pipeline(&self, code: &[u32], layout: vk::PipelineLayout) -> vk::Pipeline {
        unsafe {
            let module = self.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let p = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None)
                .map_err(|e| e.1)
                .expect("compute pipeline")[0];
            self.device.destroy_shader_module(module, None);
            p
        }
    }

    pub unsafe fn mesh_pipeline(&self, format: vk::Format) -> vk::Pipeline {
        unsafe {
            let codes = [crate::spv!("mesh_mesh"), crate::spv!("mesh_pixel")];
            let flags = [vk::ShaderStageFlags::MESH_EXT, vk::ShaderStageFlags::FRAGMENT];
            let names = [c"mesh_main", c"pixel_main"];
            let modules: Vec<_> = codes.iter().map(|c| self.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(c), None).unwrap()).collect();
            let stages: Vec<_> = (0..2).map(|i| vk::PipelineShaderStageCreateInfo::default().stage(flags[i]).module(modules[i]).name(names[i])).collect();
            let formats = [format];
            let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
            let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let rs = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::NONE).line_width(1.0);
            let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let att = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
            let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&att);
            let dyns = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let ds = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .viewport_state(&vp)
                .rasterization_state(&rs)
                .multisample_state(&ms)
                .color_blend_state(&cb)
                .dynamic_state(&ds)
                .layout(self.mesh_layout)
                .push(&mut rendering);
            let p = self.device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|e| e.1).expect("mesh pipeline")[0];
            for m in modules {
                self.device.destroy_shader_module(m, None);
            }
            p
        }
    }

    pub unsafe fn push<T: Copy>(&self, cmd: vk::CommandBuffer, layout: vk::PipelineLayout, stages: vk::ShaderStageFlags, v: &T) {
        unsafe {
            let b = std::slice::from_raw_parts((v as *const T).cast::<u8>(), std::mem::size_of::<T>());
            self.device.cmd_push_constants(cmd, layout, stages, 0, b);
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

    pub unsafe fn layout_barrier(&self, cmd: vk::CommandBuffer, image: vk::Image, from: vk::ImageLayout, to: vk::ImageLayout) {
        unsafe {
            let ib = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .old_layout(from)
                .new_layout(to)
                .image(image)
                .subresource_range(color_range())];
            self.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&ib));
        }
    }

    pub unsafe fn timeline(&self) -> vk::Semaphore {
        unsafe {
            let mut t = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            self.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push(&mut t), None).unwrap()
        }
    }

    pub unsafe fn binary(&self) -> vk::Semaphore {
        unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() }
    }

    pub unsafe fn wait(&self, sem: vk::Semaphore, value: u64) {
        unsafe {
            let s = [sem];
            let v = [value];
            self.device.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&s).values(&v), 20_000_000_000).expect("timeline wait");
        }
    }

    pub unsafe fn command_pool(&self, family: u32) -> vk::CommandPool {
        unsafe { self.device.create_command_pool(&vk::CommandPoolCreateInfo::default().flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER).queue_family_index(family), None).unwrap() }
    }

    pub unsafe fn command_buffers(&self, pool: vk::CommandPool, n: u32) -> Vec<vk::CommandBuffer> {
        unsafe {
            self.device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(n))
                .unwrap()
        }
    }

    pub unsafe fn begin(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .unwrap();
        }
    }

    pub unsafe fn surface(&self, win: &crate::win::Window) -> vk::SurfaceKHR {
        unsafe {
            let info = vk::Win32SurfaceCreateInfoKHR::default().hinstance(win.hinstance).hwnd(win.hwnd as isize);
            self.win32_i.create_win32_surface(&info, None).expect("surface")
        }
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)
}
