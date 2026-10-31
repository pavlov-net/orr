//! Raw Vulkan access for libraries that record their own commands into orr's command
//! buffers: device-creation hooks, raw handles, image handles and deferred release.

use std::any::Any;
use std::ffi::{CStr, CString};
use std::fmt;

use ash::vk;

use super::command::CommandList;
use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::Item;
use super::memory::Allocation;
use super::queue::QueueKind;
use super::texture::Texture;
use crate::{Error, ViewDesc};

/// Additions to device creation. Every method has a no-op default.
pub trait DeviceHooks {
    /// Instance extensions to enable besides orr's.
    fn instance_extensions(&self, entry: &ash::Entry) -> Vec<CString> {
        let _ = entry;
        Vec::new()
    }

    /// Device extensions to enable on `adapter`, the GPU orr picked, besides orr's.
    fn device_extensions(
        &self,
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
    ) -> Vec<CString> {
        let _ = (instance, adapter);
        Vec::new()
    }

    /// Turns on feature bits and chains feature structs orr does not enable itself through
    /// [`Features::push`]. Leaves every `p_next` as it is.
    fn device_features(&self, adapter: vk::PhysicalDevice, features: &mut Features<'_>) {
        let _ = (adapter, features);
    }

    /// Runs once the device exists, before `Device::new` returns.
    fn device_created(&self, raw: &RawDevice<'_>) {
        let _ = raw;
    }
}

impl fmt::Debug for dyn DeviceHooks + '_ {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DeviceHooks")
    }
}

/// The feature structs orr enables at device creation, for [`DeviceHooks::device_features`].
pub struct Features<'a> {
    pub core: &'a mut vk::PhysicalDeviceFeatures,
    pub vulkan_1_1: &'a mut vk::PhysicalDeviceVulkan11Features<'static>,
    pub vulkan_1_2: &'a mut vk::PhysicalDeviceVulkan12Features<'static>,
    pub vulkan_1_3: &'a mut vk::PhysicalDeviceVulkan13Features<'static>,
    pub vulkan_1_4: &'a mut vk::PhysicalDeviceVulkan14Features<'static>,
    /// Structure types orr already chains.
    pub(crate) chained: Vec<vk::StructureType>,
    pub(crate) extra: Vec<Box<dyn Any>>,
}

impl Features<'_> {
    /// Chains `s` into device creation. Panics if orr or an earlier push already chains a
    /// struct of the same type; set its bits through the fields instead.
    pub fn push<T>(&mut self, s: T)
    where
        T: vk::Extends<vk::DeviceCreateInfo<'static>> + vk::TaggedStructure<'static> + 'static,
    {
        assert!(
            !self.chained.contains(&T::STRUCTURE_TYPE),
            "{:?} is already in the device feature chain",
            T::STRUCTURE_TYPE
        );
        self.chained.push(T::STRUCTURE_TYPE);
        self.extra.push(Box::new(s));
    }
}

/// The raw Vulkan objects of a device and its Render queue.
///
/// The queue is for queries only: orr's [`crate::Queue`] owns every submit to it.
pub struct RawDevice<'a> {
    pub entry: &'a ash::Entry,
    pub instance: &'a ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: &'a ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    /// Every enabled device extension, orr's and the hooks'.
    pub extensions: &'a [CString],
}

impl RawDevice<'_> {
    pub fn get_instance_proc_addr(&self) -> vk::PFN_vkGetInstanceProcAddr {
        self.entry.static_fn().get_instance_proc_addr
    }

    pub fn get_device_proc_addr(&self) -> vk::PFN_vkGetDeviceProcAddr {
        self.instance.fp_v1_0().get_device_proc_addr
    }

    pub fn has_extension(&self, name: &CStr) -> bool {
        self.extensions.iter().any(|e| e.as_c_str() == name)
    }
}

/// The raw objects of `device`.
pub fn raw_device(device: &Device) -> RawDevice<'_> {
    device.shared.raw_device()
}

impl Shared {
    pub(crate) fn raw_device(&self) -> RawDevice<'_> {
        let render = QueueKind::Render.index();
        RawDevice {
            entry: &self.entry,
            instance: &self.instance,
            physical_device: self.physical,
            device: &self.device,
            queue: self.queue_handles[render].expect("the Render queue always exists"),
            queue_family: self.families_by_kind[render],
            extensions: &self.extensions,
        }
    }
}

/// A command list's buffer, for recording raw Vulkan commands. Heap-mode commands recorded
/// on the list afterwards rebind the heaps, the pipeline and the root first.
pub struct RawCommandBuffer<'a> {
    pub device: &'a ash::Device,
    pub buffer: vk::CommandBuffer,
    pub queue_family: u32,
}

impl CommandList<'_> {
    /// Hands out the list's command buffer for raw recording, after recording one classic
    /// descriptor-set bind on the compute bind point: on a queue that has bound descriptor
    /// heaps, some libraries' commands write nothing without one. Raw commands leave
    /// textures in `GENERAL`.
    pub fn vulkan_raw(&mut self) -> RawCommandBuffer<'_> {
        assert!(
            !self.in_rendering(),
            "raw commands are recorded outside rendering"
        );
        let queue = self.queue();
        let shared = &queue.shared;
        let cmd = self.cmd();
        if queue.shaders {
            shared.classic_bind().record(&shared.device, cmd);
            self.raw_recorded();
        }
        RawCommandBuffer {
            device: &shared.device,
            buffer: cmd,
            queue_family: queue.family,
        }
    }
}

/// A texture's raw image and a view of all its mips and layers.
#[derive(Clone, Copy, Debug)]
pub struct ImageHandles {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub range: vk::ImageSubresourceRange,
}

/// The raw handles of `t`. The view is created on the first call and lives as long as the
/// texture. Needs a texture whose usage allows views: sampled, storage or an attachment.
pub fn image_handles(t: &Texture) -> Result<ImageHandles, Error> {
    let info = t.view_info(&ViewDesc::default());
    let view = match t.full_view.get() {
        Some(&v) => v,
        None => {
            let shared = t.shared();
            // SAFETY: the image is live and bound to memory.
            let made = unsafe { shared.device.create_image_view(&info, None) }
                .map_err(Error::vk("vkCreateImageView"))?;
            name_object(&shared.debug_fn, made, "orr full view");
            if let Err(made) = t.full_view.set(made) {
                // SAFETY: another thread's view won the race; this one was never handed out.
                unsafe { shared.device.destroy_image_view(made, None) };
            }
            *t.full_view.get().unwrap()
        }
    };
    let d = t.desc();
    Ok(ImageHandles {
        image: t.image,
        view,
        format: info.format,
        extent: vk::Extent3D {
            width: d.extent[0],
            height: d.extent[1],
            depth: d.extent[2],
        },
        range: info.subresource_range,
    })
}

/// The raw buffer behind `a`, which starts at offset 0 and spans `a.size` bytes.
pub fn buffer_handle<T>(a: &Allocation<T>) -> vk::Buffer {
    a.raw().buffer
}

/// Runs `release` once everything submitted to `device` so far has completed, from
/// [`Device::collect_garbage`] or when the device is destroyed, before the Vulkan
/// device is. For objects a library made on the device and its commands use.
pub fn defer_after_submitted(device: &Device, release: impl FnOnce() + Send + 'static) {
    device
        .shared
        .defer_after_submitted(Item::Release(Box::new(release)));
}

/// A static one-binding classic descriptor set with an immutable sampler. Binding it on the
/// compute bind point before NGX lets NGX write its outputs on a queue that has bound
/// descriptor heaps.
pub(crate) struct ClassicBind {
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl ClassicBind {
    pub(crate) fn new(device: &ash::Device) -> Result<Self, Error> {
        let mut this = Self {
            sampler: vk::Sampler::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
        };
        // SAFETY: plain object creation on a live device; on failure `destroy` frees what
        // was made, and null handles are ignored.
        let made = unsafe { this.create(device) };
        if let Err(e) = made {
            // SAFETY: nothing uses the objects yet.
            unsafe { this.destroy(device) };
            return Err(e);
        }
        Ok(this)
    }

    unsafe fn create(&mut self, device: &ash::Device) -> Result<(), Error> {
        // SAFETY: valid create infos on a live device (caller).
        unsafe {
            self.sampler = device
                .create_sampler(&vk::SamplerCreateInfo::default(), None)
                .map_err(Error::vk("vkCreateSampler"))?;
            let samplers = [self.sampler];
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .immutable_samplers(&samplers)];
            self.set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(Error::vk("vkCreateDescriptorSetLayout"))?;
            let set_layouts = [self.set_layout];
            self.layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
                .map_err(Error::vk("vkCreatePipelineLayout"))?;
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 1,
            }];
            self.pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(Error::vk("vkCreateDescriptorPool"))?;
            self.set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.pool)
                        .set_layouts(&set_layouts),
                )
                .map_err(Error::vk("vkAllocateDescriptorSets"))?[0];
        }
        Ok(())
    }

    fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        // SAFETY: `cmd` is recording on a compute-capable queue of the device the set was
        // made on.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
        }
    }

    /// # Safety
    ///
    /// No pending command buffer uses the set.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from the caller; destroying null handles is a no-op.
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
    }
}
