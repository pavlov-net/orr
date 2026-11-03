//! NGX on ash handles: device requirements, initialization, and ray reconstruction, super
//! resolution and frame generation recorded into the caller's command buffers.
//!
//! Images passed to NGX stay in `GENERAL`; NGX needs no layout transitions around its calls.

use crate::{DlssError, Mode, Quality, Support, Uuid};
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

/// NGX initialized on one device.
pub struct Ngx(imp::Ngx);

impl Ngx {
    /// Initializes NGX with the application's project ID.
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
}

/// The preset NGX applied to the last `upscaler` feature created at `mode`, as a letter,
/// from NGX's log since before the creation. NGX reports the model it applied for the
/// feature (`Override Reported: ModelPreset, Value: n`, n = 1 for A); without that line, its
/// line for the feature's class and mode names the preset. A driver-wide DLSS override can
/// replace the requested preset, and both lines follow it. NGX also logs other features'
/// lines while creating one, so lines are matched by feature and mode.
fn applied_preset(log: &[String], upscaler: Upscaler, mode: Mode) -> Option<char> {
    let (feature, class) = match upscaler {
        Upscaler::Rr => (
            "Feature [SuperSamplingDenoising] ",
            "[NgxRayReconstruction::",
        ),
        Upscaler::Sr => ("Feature [SuperSampling] ", "[NgxDltss::"),
    };
    let reported = log.iter().rev().find_map(|l| {
        let rest = &l[l.find(feature)?..];
        let at = rest.find("ModelPreset, Value: ")? + "ModelPreset, Value: ".len();
        let digits: String = rest[at..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        let n: u8 = digits.parse().ok()?;
        (1..=26).contains(&n).then(|| char::from(b'A' + n - 1))
    });
    let tag = match mode {
        Mode::Dlaa => "(DLAA)",
        Mode::Quality => "(Quality)",
        Mode::Balanced => "(Balanced)",
        Mode::Performance => "(Perf)",
        Mode::UltraPerformance => "(UltraPerf)",
    };
    let named = || {
        log.iter()
            .rev()
            .filter(|l| l.contains(class) && l.contains(tag))
            .find_map(|l| {
                let at = l.rfind("Preset")? + "Preset".len();
                let mut rest = l[at..].chars().skip_while(|c| *c == '_' || *c == ' ');
                rest.next().filter(char::is_ascii_uppercase)
            })
    };
    reported.or_else(named)
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
    /// Premultiplied color with coverage in alpha, composited after denoising.
    pub transparency: Option<&'a Image>,
    pub depth_of_field_guide: Option<&'a Image>,
    pub output: &'a Image,
    /// Drops the history.
    pub reset: bool,
    /// From [`RayReconstruction::jitter`] for this frame.
    pub jitter: [f32; 2],
}

/// Ray reconstruction or super resolution, as NGX tells them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Upscaler {
    Rr,
    Sr,
}

/// One ray reconstruction feature at a fixed render and output size.
pub struct RayReconstruction {
    inner: imp::Upscale,
    mode: Mode,
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
        let mode = quality.resolve(output);
        let render = ngx.0.render_size(output, mode, Upscaler::Rr)?;
        // SAFETY: forwarded from the caller.
        unsafe { Self::create(ngx, cmd, render, output, mode, preset) }
    }

    /// [`new`](Self::new) at `render`, with the mode nearest it. NVIDIA documents only the
    /// render sizes of the modes for ray reconstruction, so NGX may refuse others.
    ///
    /// # Safety
    ///
    /// As [`new`](Self::new).
    pub(crate) unsafe fn with_render_size(
        ngx: &Ngx,
        cmd: vk::CommandBuffer,
        render: [u32; 2],
        output: [u32; 2],
        preset: Option<u32>,
    ) -> Result<Self, DlssError> {
        let mode = Mode::for_render_size(render, output);
        // SAFETY: forwarded from the caller.
        unsafe { Self::create(ngx, cmd, render, output, mode, preset) }
    }

    unsafe fn create(
        ngx: &Ngx,
        cmd: vk::CommandBuffer,
        render: [u32; 2],
        output: [u32; 2],
        mode: Mode,
        preset: Option<u32>,
    ) -> Result<Self, DlssError> {
        // SAFETY: forwarded from the caller.
        let inner =
            unsafe { imp::Upscale::new(&ngx.0, cmd, Upscaler::Rr, render, output, mode, preset)? };
        Ok(Self {
            inner,
            mode,
            render,
            output,
            phases: crate::jitter::jitter_phases(render, output),
        })
    }

    /// The mode the feature was created with.
    pub(crate) fn mode(&self) -> Mode {
        self.mode
    }

    /// Render resolution, fixed for the feature's lifetime.
    pub fn render_size(&self) -> [u32; 2] {
        self.render
    }

    pub(crate) fn output_size(&self) -> [u32; 2] {
        self.output
    }

    /// NGX's jitter offset for `frame`, in render pixels, each axis in [-0.5, 0.5). The
    /// projection places each pixel's sample at the pixel center minus this value on both
    /// axes, with y pointing down.
    pub fn jitter(&self, frame: u64) -> [f32; 2] {
        crate::jitter::jitter(frame, self.phases)
    }

    /// Texture LOD bias for the render resolution: `log2(render / output) - 1`.
    pub fn mip_bias(&self) -> f32 {
        crate::jitter::mip_bias(self.render, self.output)
    }

    /// The preset NGX applied to this feature, read from `log`: [`Ngx::take_log`] from
    /// before the feature was created.
    pub fn applied_preset(&self, log: &[String]) -> Option<char> {
        applied_preset(log, Upscaler::Rr, self.mode)
    }

    /// Records the evaluation into `cmd`. Every image is in `GENERAL`, inputs at
    /// [`render_size`](Self::render_size), the output at
    /// the output size.
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
        unsafe { self.inner.record_rr(cmd, inputs, self.render) }
    }
}

/// Super resolution inputs at render resolution, and the output at output resolution.
#[derive(Clone, Copy)]
#[cfg_attr(not(ngx), allow(dead_code))]
pub(crate) struct SrInputs<'a> {
    /// HDR radiance, multiplied by `pre_exposure`.
    pub(crate) color: &'a Image,
    /// Reverse-Z device depth.
    pub(crate) depth: &'a Image,
    /// Unjittered motion, previous UV minus current UV.
    pub(crate) motion: &'a Image,
    /// 1x1; its first channel is the exposure that brings `color / pre_exposure` to the range
    /// the tonemapper takes (mid gray near 0.18). Presets J and K read it; L meters the color
    /// itself.
    pub(crate) exposure: &'a Image,
    /// The factor `color` was multiplied by, which NGX divides out.
    pub(crate) pre_exposure: f32,
    pub(crate) output: &'a Image,
    /// Drops the history.
    pub(crate) reset: bool,
    /// From [`SuperResolution::jitter`] for this frame.
    pub(crate) jitter: [f32; 2],
}

/// One super resolution (or DLAA) feature at a fixed render and output size.
pub(crate) struct SuperResolution {
    inner: imp::Upscale,
    mode: Mode,
    render: [u32; 2],
    output: [u32; 2],
    phases: u32,
}

impl SuperResolution {
    /// Creates the feature, recording NGX's setup work into `cmd`; submit `cmd` before the
    /// first [`record`](Self::record) runs. `preset` is a `NVSDK_NGX_DLSS_Hint_Render_Preset`
    /// value; `None` keeps NGX's default for the mode.
    ///
    /// # Safety
    ///
    /// As [`RayReconstruction::new`].
    pub(crate) unsafe fn new(
        ngx: &Ngx,
        cmd: vk::CommandBuffer,
        output: [u32; 2],
        quality: Quality,
        preset: Option<u32>,
    ) -> Result<Self, DlssError> {
        let mode = quality.resolve(output);
        let render = ngx.0.render_size(output, mode, Upscaler::Sr)?;
        // SAFETY: forwarded from the caller.
        let inner =
            unsafe { imp::Upscale::new(&ngx.0, cmd, Upscaler::Sr, render, output, mode, preset)? };
        Ok(Self {
            inner,
            mode,
            render,
            output,
            phases: crate::jitter::jitter_phases(render, output),
        })
    }

    /// Render resolution, fixed for the feature's lifetime.
    pub(crate) fn render_size(&self) -> [u32; 2] {
        self.render
    }

    pub(crate) fn output_size(&self) -> [u32; 2] {
        self.output
    }

    /// NGX's jitter offset for `frame`.
    pub(crate) fn jitter(&self, frame: u64) -> [f32; 2] {
        crate::jitter::jitter(frame, self.phases)
    }

    /// Texture LOD bias for the render resolution.
    pub(crate) fn mip_bias(&self) -> f32 {
        crate::jitter::mip_bias(self.render, self.output)
    }

    /// The preset NGX applied to this feature, read from `log`: [`Ngx::take_log`] from
    /// before the feature was created.
    pub(crate) fn applied_preset(&self, log: &[String]) -> Option<char> {
        applied_preset(log, Upscaler::Sr, self.mode)
    }

    /// Records the evaluation into `cmd`, under the same rules as
    /// [`RayReconstruction::record`].
    ///
    /// # Safety
    ///
    /// As [`RayReconstruction::record`].
    pub(crate) unsafe fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        inputs: &SrInputs<'_>,
    ) -> Result<(), DlssError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.inner.record_sr(cmd, inputs, self.render) }
    }
}

/// A buffer NGX writes, from its start.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(ngx), allow(dead_code))]
pub(crate) struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) size: u64,
}

/// The camera frame generation reprojects with. Matrices are column-vector matrices given
/// as their columns, which is NGX's row-major layout for row vectors. None holds jitter.
/// View space looks down -z with y up; clip space is Vulkan's, y down, reverse Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FgCamera {
    pub view_to_clip: [[f32; 4]; 4],
    pub clip_to_view: [[f32; 4]; 4],
    pub clip_to_prev_clip: [[f32; 4]; 4],
    pub prev_clip_to_clip: [[f32; 4]; 4],
    /// Relative to a point that stays fixed between resets.
    pub position: [f32; 3],
    pub up: [f32; 3],
    pub right: [f32; 3],
    pub forward: [f32; 3],
    pub near: f32,
    /// Finite, though the projection has no far plane.
    pub far: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Width over height.
    pub aspect: f32,
}

/// One frame generation evaluate: the rendered frame, its guides and the image to write.
#[derive(Clone, Copy)]
#[cfg_attr(not(ngx), allow(dead_code))]
pub(crate) struct FgInputs<'a> {
    /// Display-encoded, without UI, in the feature's format and size.
    pub(crate) backbuffer: &'a Image,
    /// Reverse-Z device depth.
    pub(crate) depth: &'a Image,
    /// Unjittered motion, previous UV minus current UV.
    pub(crate) motion: &'a Image,
    /// RGBA16F bidirectional distortion field: RG the UV offset from a presented pixel to
    /// its undistorted position, BA the reverse.
    pub(crate) distortion: Option<&'a Image>,
    pub(crate) interpolated: &'a Image,
    /// At least 4 bytes; NGX sets the first byte when the generated images of this frame
    /// should not be shown.
    pub(crate) disable_interpolation: Option<Buffer>,
    pub(crate) camera: &'a FgCamera,
    /// NGX's jitter of the guides, in render pixels.
    pub(crate) jitter: [f32; 2],
    /// Scales `motion` to pixels of the motion image.
    pub(crate) mvec_scale: [f32; 2],
    /// The previous frame is unrelated to this one.
    pub(crate) reset: bool,
    /// Increases by one each rendered frame, with frame generation on or off.
    pub(crate) frame_id: u64,
    /// Menus and loading screens.
    pub(crate) not_rendering_scene: bool,
    /// The backbuffer is PQ.
    pub(crate) hdr: bool,
    /// Scales NGX's linearized depth, `1 / depth`; NGX's default is 1.
    pub(crate) depth_scale: f32,
}

/// One frame generation feature at a fixed output size and backbuffer format.
pub(crate) struct FrameGeneration {
    inner: imp::Fg,
    max_generated: u32,
}

impl FrameGeneration {
    /// Creates the feature, recording NGX's setup work into `cmd`; submit `cmd` before the
    /// first [`record`](Self::record) runs. NGX may finish creating it on its own thread.
    ///
    /// # Safety
    ///
    /// `cmd` is recording on a queue of `ngx`'s device. The feature is dropped before `ngx`,
    /// after the GPU has finished every command buffer that used it.
    pub(crate) unsafe fn new(
        ngx: &Ngx,
        cmd: vk::CommandBuffer,
        output: [u32; 2],
        format: vk::Format,
    ) -> Result<Self, DlssError> {
        let max_generated = ngx.support().max_generated;
        // SAFETY: forwarded from the caller.
        let inner = unsafe { imp::Fg::new(&ngx.0, cmd, output, format)? };
        Ok(Self {
            inner,
            max_generated,
        })
    }

    /// Records generated image `index` of `count` (1-based) between the previous
    /// backbuffer and this one. Call it for every index in order each frame. Every image is
    /// in `GENERAL`. Needs the same classic bind as [`RayReconstruction::record`], and
    /// leaves NGX's pipeline and descriptors bound.
    ///
    /// # Safety
    ///
    /// `cmd` is recording on a queue of the feature's device, and every image and buffer
    /// outlives its execution.
    pub(crate) unsafe fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        inputs: &FgInputs<'_>,
        index: u32,
        count: u32,
    ) -> Result<(), DlssError> {
        assert!(
            (1..=count).contains(&index) && count <= self.max_generated.max(1),
            "generated image {index} of {count} with at most {} per frame",
            self.max_generated
        );
        // SAFETY: forwarded from the caller.
        unsafe { self.inner.record(cmd, inputs, index, count) }
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
    use super::{Mode, Upscaler, applied_preset};

    #[test]
    fn applied_preset_is_read_for_the_feature_and_mode() {
        let log = [
            "[NgxRayReconstruction::FillCreationParams::<lambda>::operator ():1733] Info: (Quality) Using Default Overriden Preset Preset_D",
            "[NgxDltss::FillCreationParams::<lambda>::operator ():2745] Info: (Quality) Using DRS Overridden Preset K",
            "[NgxDltss::FillCreationParams::<lambda>::operator ():2745] Info: (Perf) Using DRS Overridden Preset M",
            "[NGXOverrideStatusCallback:2325] Feature [SuperSampling] Override Reported: ModelPreset, Value: 11, Requested: Yes, Applied: Yes, Is Latest: Yes",
        ]
        .map(String::from);
        assert_eq!(applied_preset(&log, Upscaler::Sr, Mode::Quality), Some('K'));
        assert_eq!(
            applied_preset(&log[..3], Upscaler::Sr, Mode::Performance),
            Some('M')
        );
        assert_eq!(
            applied_preset(&log[..3], Upscaler::Rr, Mode::Quality),
            Some('D')
        );
        assert_eq!(applied_preset(&log, Upscaler::Rr, Mode::Dlaa), None);
        let rr = [
            "[NgxRayReconstruction::FillCreationParams] Info: (Perf) Using DRS Overridden Preset Preset_F",
            "[NGXOverrideStatusCallback:2325] Feature [SuperSamplingDenoising] Override Reported: ModelPreset, Value: 6",
        ]
        .map(String::from);
        assert_eq!(
            applied_preset(&rr, Upscaler::Rr, Mode::Performance),
            Some('F')
        );
        assert_eq!(applied_preset(&rr, Upscaler::Sr, Mode::Performance), None);
    }
}
