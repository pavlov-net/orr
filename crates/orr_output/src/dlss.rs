//! DLSS as the backend of ray reconstruction and frame generation.

use orr_dlss::{
    DlssSdk, FgCamera, FgInputs, FrameGeneration, MIN_OUTPUT, Mode, Quality, RayReconstruction,
    RrInputs,
};
use orr_math::{DMat4, DVec3, Vec3};
use orr_scene::orr_gpu::{CommandList, Device, Texture};

use crate::{
    Error, FrameGenerationBackend, FrameGenerationCamera, FrameGenerationFeature,
    FrameGenerationInputs, InterpolationCheck, ReconstructionBackend, ReconstructionFeature,
    ReconstructionInputs, ReconstructionMode, ReconstructionSupport, Result, UpscaleMode,
    UpscaleQuality,
};

impl ReconstructionBackend for DlssSdk {
    fn support(&self) -> ReconstructionSupport {
        let rr = DlssSdk::support(self).rr;
        ReconstructionSupport {
            available: rr,
            dlaa: rr,
            min_output: MIN_OUTPUT,
            scale: None,
        }
    }

    fn feature(
        &self,
        output: [u32; 2],
        quality: UpscaleQuality,
    ) -> Result<Box<dyn ReconstructionFeature>> {
        let quality = match quality {
            UpscaleQuality::Auto => Quality::Auto,
            UpscaleQuality::Fixed(mode) => Quality::Fixed(match mode {
                UpscaleMode::Quality => Mode::Quality,
                UpscaleMode::Balanced => Mode::Balanced,
                UpscaleMode::Performance => Mode::Performance,
                UpscaleMode::UltraPerformance => Mode::UltraPerformance,
            }),
        };
        Ok(Box::new(RayReconstruction::new(
            self, output, quality, None,
        )?))
    }

    fn feature_at(
        &self,
        render: [u32; 2],
        output: [u32; 2],
    ) -> Result<Box<dyn ReconstructionFeature>> {
        Ok(Box::new(RayReconstruction::with_render_size(
            self, render, output, None,
        )?))
    }

    fn dlaa(&self, output: [u32; 2]) -> Result<Box<dyn ReconstructionFeature>> {
        Ok(Box::new(RayReconstruction::new(
            self,
            output,
            Quality::Fixed(Mode::Dlaa),
            None,
        )?))
    }
}

impl ReconstructionFeature for RayReconstruction {
    fn mode(&self) -> ReconstructionMode {
        match RayReconstruction::mode(self) {
            Mode::Dlaa => ReconstructionMode::Dlaa,
            Mode::Quality => ReconstructionMode::Upscale(UpscaleMode::Quality),
            Mode::Balanced => ReconstructionMode::Upscale(UpscaleMode::Balanced),
            Mode::Performance => ReconstructionMode::Upscale(UpscaleMode::Performance),
            Mode::UltraPerformance => ReconstructionMode::Upscale(UpscaleMode::UltraPerformance),
        }
    }

    fn render_size(&self) -> [u32; 2] {
        RayReconstruction::render_size(self)
    }

    fn output_size(&self) -> [u32; 2] {
        RayReconstruction::output_size(self)
    }

    fn jitter(&self, frame: u64) -> [f32; 2] {
        RayReconstruction::jitter(self, frame)
    }

    fn mip_bias(&self) -> f32 {
        RayReconstruction::mip_bias(self)
    }

    fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &ReconstructionInputs<'_>,
    ) -> Result<()> {
        let inputs = RrInputs {
            color: inputs.color,
            color_before_transparency: inputs.color_before_transparency,
            depth: inputs.depth,
            motion: inputs.motion,
            diffuse_albedo: inputs.diffuse_albedo,
            specular_albedo: inputs.specular_albedo,
            normal_roughness: inputs.normal_roughness,
            specular_motion: inputs.specular_motion,
            transparency: inputs.transparency,
            depth_of_field_guide: inputs.depth_of_field_guide,
            output: inputs.output,
            reset: inputs.reset,
            jitter: inputs.view.jitter.to_array(),
        };
        Ok(RayReconstruction::record(self, list, &inputs)?)
    }
}

impl FrameGenerationBackend for DlssSdk {
    fn max_generated(&self, gpu: &Device, low_latency: bool) -> Result<u32> {
        let support = DlssSdk::support(self);
        let caps = gpu.caps();
        let missing = if !support.fg || support.max_generated == 0 {
            "NGX frame generation"
        } else if !low_latency {
            "VK_NV_low_latency2"
        } else if !caps.present_wait || !caps.present_metering {
            "present ids and present metering"
        } else if hardware_scheduling(gpu) != Some(true) {
            "hardware-accelerated GPU scheduling"
        } else {
            return Ok(support.max_generated);
        };
        Err(Error::FrameGenerationUnavailable(missing))
    }

    fn feature(&self, backbuffer: &Texture, hdr: bool) -> Result<Box<dyn FrameGenerationFeature>> {
        Ok(Box::new(FrameGeneration::new(self, backbuffer, hdr)?))
    }

    fn interpolation_check(&self) -> Option<InterpolationCheck> {
        Some(FrameGeneration::interpolation_disabled)
    }
}

impl FrameGenerationFeature for FrameGeneration {
    fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &FrameGenerationInputs<'_>,
    ) -> Result<()> {
        let camera = fg_camera(&inputs.camera, &inputs.previous_camera, inputs.anchor);
        let count = inputs.outputs.len() as u32;
        for (index, interpolated) in (1..).zip(inputs.outputs) {
            let fg = FgInputs {
                backbuffer: inputs.backbuffer,
                depth: inputs.depth,
                motion: inputs.motion,
                distortion: inputs.distortion,
                interpolated,
                disable_interpolation: inputs.disable_interpolation,
                camera,
                jitter: inputs.jitter,
                mvec_scale: inputs.camera.extent.map(|v| v as f32),
                reset: inputs.reset,
                frame_id: inputs.frame_id,
                not_rendering_scene: inputs.not_rendering_scene,
                depth_scale: 1.0,
            };
            FrameGeneration::record(self, list, &fg, index, count)?;
        }
        Ok(())
    }
}

/// Whether Windows schedules the device's GPU in hardware, which frame generation needs.
/// `None` when it cannot tell; always true off Windows.
#[cfg(windows)]
fn hardware_scheduling(gpu: &Device) -> Option<bool> {
    use orr_scene::orr_gpu::interop::{self, ash::vk};
    use vk::TaggedStructure as _;

    let raw = interop::raw_device(gpu);
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push(&mut id);
    // SAFETY: a query on the device's live physical device.
    unsafe {
        raw.instance
            .get_physical_device_properties2(raw.physical_device, &mut props);
    }
    if id.device_luid_valid == vk::FALSE {
        return None;
    }
    orr_window::hardware_scheduling(id.device_luid)
}

#[cfg(not(windows))]
fn hardware_scheduling(_gpu: &Device) -> Option<bool> {
    Some(true)
}

/// The far plane frame generation is told; orr's projections have none.
const FG_FAR_M: f32 = 100_000.0;

/// NGX's camera for `cur`, reprojecting to `prev`, with the position relative to `anchor`.
fn fg_camera(cur: &FrameGenerationCamera, prev: &FrameGenerationCamera, anchor: DVec3) -> FgCamera {
    let clip = cur.clip_from_view();
    let view_from_clip = clip.inverse();
    let world_from_view = DMat4::from_quat(cur.rotation.as_dquat());
    let prev_view_from_world = DMat4::from_quat(prev.rotation.as_dquat().inverse());
    // Positions relative to each camera: the current one sits at the previous one's
    // position plus the motion.
    let motion = DMat4::from_translation(cur.position - prev.position);
    let clip_to_prev =
        prev.clip_from_view() * prev_view_from_world * motion * world_from_view * view_from_clip;
    let cols = |m: DMat4| m.as_mat4().to_cols_array_2d();
    let dir = |v: Vec3| (cur.rotation * v).to_array();
    FgCamera {
        view_to_clip: cols(clip),
        clip_to_view: cols(view_from_clip),
        clip_to_prev_clip: cols(clip_to_prev),
        prev_clip_to_clip: cols(clip_to_prev.inverse()),
        position: (cur.position - anchor).as_vec3().to_array(),
        up: dir(Vec3::Y),
        right: dir(Vec3::X),
        forward: dir(Vec3::NEG_Z),
        near: cur.projection.near,
        far: FG_FAR_M,
        fov_y: cur.projection.fov_y,
        aspect: cur.output[0] as f32 / cur.output[1].max(1) as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orr_math::{DVec4, Mat4, Quat};
    use orr_scene::{Projection, SceneId};

    /// `m`, a column-vector matrix given as columns, applied to `p`.
    fn apply(m: [[f32; 4]; 4], p: DVec4) -> DVec4 {
        Mat4::from_cols_array_2d(&m).as_dmat4() * p
    }

    #[test]
    fn clip_to_prev_clip_reprojects_a_world_point() {
        let projection = Projection {
            fov_y: 0.9,
            near: 0.1,
        };
        let prev = FrameGenerationCamera {
            scene: SceneId(0),
            position: DVec3::new(1000.0, 2.0, -3.0),
            rotation: Quat::from_rotation_y(0.3),
            projection,
            extent: [1600, 900],
            output: [1600, 900],
        };
        let cur = FrameGenerationCamera {
            position: prev.position + DVec3::new(0.4, 0.1, -0.7),
            rotation: Quat::from_rotation_y(0.34) * Quat::from_rotation_x(-0.05),
            ..prev
        };
        let c = fg_camera(&cur, &prev, prev.position);
        let world = DVec3::new(1003.0, 2.5, -12.0);
        let clip_of = |cam: &FrameGenerationCamera| {
            let view = cam.rotation.as_dquat().inverse() * (world - cam.position);
            let clip = cam.clip_from_view() * view.extend(1.0);
            clip / clip.w
        };
        let here = clip_of(&cur);
        let there = apply(c.clip_to_prev_clip, here);
        let there = there / there.w;
        assert!(
            (there - clip_of(&prev)).abs().max_element() < 1e-5,
            "{there} vs {}",
            clip_of(&prev)
        );
        let back = apply(c.prev_clip_to_clip, clip_of(&prev));
        assert!((back / back.w - here).abs().max_element() < 1e-5);
        let moved = Vec3::from_array(c.position) - Vec3::new(0.4, 0.1, -0.7);
        assert!(moved.abs().max_element() < 1e-6);
    }
}
