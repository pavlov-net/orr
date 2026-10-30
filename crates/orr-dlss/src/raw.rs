//! NGX on ash handles: device requirements, initialisation, and ray reconstruction recorded
//! into the caller's command buffers.
//!
//! Images passed to NGX stay in `GENERAL`; NGX needs no layout transitions around its calls.

use crate::{DlssError, Quality, Support, Uuid};
use ash::vk;
use std::ffi::CString;

#[cfg(not(ngx))]
use crate::mock as imp;
#[cfg(ngx)]
use crate::ngx as imp;

/// The Vulkan objects NGX runs on.
#[derive(Clone, Copy)]
pub struct Vulkan<'a> {
    pub entry: &'a ash::Entry,
    pub instance: &'a ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: &'a ash::Device,
}

/// Instance extensions NGX needs for ray reconstruction, super resolution and frame
/// generation. Call before creating the instance.
pub fn instance_extensions(project_id: Uuid) -> Result<Vec<CString>, DlssError> {
    imp::instance_extensions(project_id)
}

/// Device extensions NGX needs on `physical_device` for the features it supports there.
///
/// `VK_EXT_buffer_device_address` is left out: the device enables the core
/// `bufferDeviceAddress` feature instead, since enabling both is invalid. NGX searches for
/// device extensions only until its first shutdown in the process; later calls return the
/// list found for the same GPU then.
///
/// # Safety
///
/// `physical_device` belongs to `instance`.
pub unsafe fn device_extensions(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    project_id: Uuid,
) -> Result<Vec<CString>, DlssError> {
    // SAFETY: forwarded from the caller.
    let mut list = unsafe { imp::device_extensions(instance, physical_device, project_id)? };
    list.retain(|e| e.as_c_str() != ash::ext::buffer_device_address::NAME);
    Ok(list)
}

/// Turns on the device features NGX's extensions need: `pushDescriptor`, which Vulkan 1.4
/// requires alongside `VK_KHR_push_descriptor`. NGX also needs the core
/// `bufferDeviceAddress` feature.
pub fn enable_features(vulkan_1_4: &mut vk::PhysicalDeviceVulkan14Features<'_>) {
    vulkan_1_4.push_descriptor = vk::TRUE;
}

/// NGX initialised on one device.
pub struct Ngx(imp::Ngx);

impl Ngx {
    /// Initialises NGX with the application's project ID.
    ///
    /// # Safety
    ///
    /// The device was created with the extensions from [`instance_extensions`] and
    /// [`device_extensions`], and outlives this value and every feature created from it.
    pub unsafe fn new(vulkan: Vulkan<'_>, project_id: Uuid) -> Result<Self, DlssError> {
        // SAFETY: forwarded from the caller.
        unsafe { imp::Ngx::new(vulkan, project_id).map(Self) }
    }

    pub fn support(&self) -> Support {
        self.0.support()
    }

    /// Device memory NGX reports for the features alive now, in bytes.
    pub fn vram_bytes(&self) -> u64 {
        self.0.vram_bytes()
    }

    /// NGX log lines since the last call, from every NGX instance in the process.
    pub fn take_log() -> Vec<String> {
        imp::take_log()
    }

    /// The RR preset named by NGX log lines, as a letter. A driver-wide DLSS override can
    /// replace the requested preset; its log line wins.
    pub fn applied_preset(log: &[String]) -> Option<char> {
        let preset = |line: &str| {
            let at = line.find("Preset_")? + "Preset_".len();
            line[at..].chars().next().filter(char::is_ascii_uppercase)
        };
        let overridden = log
            .iter()
            .filter(|l| l.contains("Overridden Preset"))
            .find_map(|l| preset(l));
        overridden.or_else(|| log.iter().find_map(|l| preset(l)))
    }
}

/// An image NGX reads or writes. `view` covers mip 0, layer 0 of `image`.
#[derive(Clone, Copy, Debug)]
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: [u32; 2],
}

/// Ray reconstruction inputs at render resolution, and the output at output resolution.
#[derive(Clone, Copy)]
pub struct RrInputs<'a> {
    /// Pre-exposed HDR radiance.
    pub color: &'a Image,
    /// Radiance before the transparency layer; not bound by default.
    pub color_before_transparency: Option<&'a Image>,
    /// Reverse-Z device depth.
    pub depth: &'a Image,
    /// Unjittered motion, previous UV minus current UV.
    pub motion: &'a Image,
    pub diffuse_albedo: &'a Image,
    pub specular_albedo: &'a Image,
    /// World-space normal in RGB, linear roughness in A.
    pub normal_roughness: &'a Image,
    /// Motion of reflected surfaces, in the same units as `motion`.
    pub specular_motion: &'a Image,
    /// Premultiplied colour with coverage in alpha, composited after denoising.
    pub transparency: Option<&'a Image>,
    pub depth_of_field_guide: Option<&'a Image>,
    pub output: &'a Image,
    /// Drops the history.
    pub reset: bool,
    /// From [`RayReconstruction::jitter`] for this frame.
    pub jitter: [f32; 2],
}

/// One ray reconstruction feature at a fixed render and output size.
pub struct RayReconstruction {
    inner: imp::Rr,
    render: [u32; 2],
    output: [u32; 2],
    phases: u32,
}

impl RayReconstruction {
    /// Creates the feature, recording NGX's setup work into `cmd`; submit `cmd` before the
    /// first [`record`](Self::record) runs. `preset` is a
    /// `NVSDK_NGX_RayReconstruction_Hint_Render_Preset` value; `None` keeps NGX's default.
    ///
    /// # Safety
    ///
    /// `cmd` is recording on a queue of `ngx`'s device. The feature is dropped before `ngx`,
    /// after the GPU has finished every command buffer that used it.
    pub unsafe fn new(
        ngx: &Ngx,
        cmd: vk::CommandBuffer,
        output: [u32; 2],
        quality: Quality,
        preset: Option<u32>,
    ) -> Result<Self, DlssError> {
        let render = ngx.0.render_size(output, quality)?;
        // SAFETY: forwarded from the caller.
        let inner = unsafe { imp::Rr::new(&ngx.0, cmd, render, output, quality, preset)? };
        Ok(Self {
            inner,
            render,
            output,
            phases: crate::jitter_phases(render, output),
        })
    }

    /// Render resolution, fixed for the feature's lifetime.
    pub fn render_size(&self) -> [u32; 2] {
        self.render
    }

    pub fn output_size(&self) -> [u32; 2] {
        self.output
    }

    /// NGX's jitter offset for `frame` (see [`crate::jitter`]).
    pub fn jitter(&self, frame: u64) -> [f32; 2] {
        crate::jitter(frame, self.phases)
    }

    /// Texture LOD bias for the render resolution (see [`crate::mip_bias`]).
    pub fn mip_bias(&self) -> f32 {
        crate::mip_bias(self.render, self.output)
    }

    /// Records the evaluation into `cmd`. Every image is in `GENERAL`, inputs at
    /// [`render_size`](Self::render_size), the output at
    /// [`output_size`](Self::output_size).
    ///
    /// A queue that has ever bound a descriptor heap needs [`ClassicBind::record`] earlier in
    /// `cmd`; without it NGX writes nothing. NGX leaves its own pipeline and descriptors
    /// bound: rebind heaps, push data and pipeline before the next heap-mode command.
    ///
    /// # Safety
    ///
    /// `cmd` is recording on a queue of the feature's device, and every image outlives its
    /// execution.
    pub unsafe fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        inputs: &RrInputs<'_>,
    ) -> Result<(), DlssError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.inner.record(cmd, inputs, self.render) }
    }
}

/// A static one-binding classic descriptor set, bound before NGX on a queue that binds
/// descriptor heaps.
///
/// Once any command buffer on a queue has bound a descriptor heap, NGX evaluations on that
/// queue report success and write nothing, unless a classic descriptor-set bind precedes them
/// in the same command buffer.
pub struct ClassicBind {
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl ClassicBind {
    pub fn new(device: &ash::Device) -> Result<Self, DlssError> {
        let vk_err = |call| move |result| DlssError::Vulkan { call, result };
        // SAFETY: plain object creation on a live device; every object is destroyed in
        // `destroy` or here on failure.
        unsafe {
            let sampler = device
                .create_sampler(&vk::SamplerCreateInfo::default(), None)
                .map_err(vk_err("vkCreateSampler"))?;
            let samplers = [sampler];
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .immutable_samplers(&samplers)];
            let mut this = Self {
                sampler,
                set_layout: vk::DescriptorSetLayout::null(),
                layout: vk::PipelineLayout::null(),
                pool: vk::DescriptorPool::null(),
                set: vk::DescriptorSet::null(),
            };
            let made = (|| {
                this.set_layout = device
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                        None,
                    )
                    .map_err(vk_err("vkCreateDescriptorSetLayout"))?;
                let set_layouts = [this.set_layout];
                this.layout = device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                        None,
                    )
                    .map_err(vk_err("vkCreatePipelineLayout"))?;
                let sizes = [vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLER,
                    descriptor_count: 1,
                }];
                this.pool = device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(1)
                            .pool_sizes(&sizes),
                        None,
                    )
                    .map_err(vk_err("vkCreateDescriptorPool"))?;
                this.set = device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(this.pool)
                            .set_layouts(&set_layouts),
                    )
                    .map_err(vk_err("vkAllocateDescriptorSets"))?[0];
                Ok(())
            })();
            match made {
                Ok(()) => Ok(this),
                Err(e) => {
                    this.destroy(device);
                    Err(e)
                }
            }
        }
    }

    /// Records the classic bind on the compute bind point.
    ///
    /// # Safety
    ///
    /// `cmd` is recording on `device`, the device this was created on.
    pub unsafe fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        // SAFETY: forwarded from the caller.
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
    /// `device` is the device this was created on, and no pending command buffer uses it.
    pub unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: forwarded from the caller; destroying null handles is a no-op.
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Ngx;

    #[test]
    fn applied_preset_prefers_the_driver_override() {
        let log = [
            "[NgxRayReconstruction::FillCreationParams] Info: (Perf) Using DRS Overridden Preset Preset_F",
            "[NGXOverrideStatusCallback:2325] Feature [SuperSamplingDenoising] Override Reported: ModelPreset, Value: 6",
        ]
        .map(String::from);
        assert_eq!(Ngx::applied_preset(&log), Some('F'));
        assert_eq!(
            Ngx::applied_preset(&["Info: Using Preset_E".to_owned()]),
            Some('E')
        );
        assert_eq!(Ngx::applied_preset(&[]), None);
    }
}
