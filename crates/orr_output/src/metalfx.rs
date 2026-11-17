//! MetalFX as the backend of ray reconstruction.

use orr_math::jitter::{jitter, jitter_phases, mip_bias};
use orr_math::{DMat4, DVec3};
use orr_metalfx::{DenoisedInputs, DenoisedScaler, Float4x4, ScaleRange};
use orr_scene::orr_gpu::{CommandList, Device};

use crate::{
    Error, GuideReads, ReconstructionBackend, ReconstructionFeature, ReconstructionInputs,
    ReconstructionMode, ReconstructionSupport, Result, UpscaleMode, UpscaleQuality,
};

/// MetalFX's effects on one device.
pub struct MetalFx {
    device: Device,
    denoised: Option<ScaleRange>,
}

impl MetalFx {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            denoised: orr_metalfx::support(device).denoised,
        }
    }

    fn denoiser(
        &self,
        render: [u32; 2],
        output: [u32; 2],
        mode: ReconstructionMode,
    ) -> Result<Box<dyn ReconstructionFeature>> {
        let scaler = DenoisedScaler::new(&self.device, render, output)?;
        Ok(Box::new(Denoiser {
            scaler,
            mode,
            phases: jitter_phases(render, output),
            anchor: None,
        }))
    }
}

/// Output over render size for `mode`: the scales DLSS's modes name.
fn mode_scale(mode: UpscaleMode) -> f64 {
    match mode {
        UpscaleMode::Quality => 1.5,
        UpscaleMode::Balanced => 1.0 / 0.58,
        UpscaleMode::Performance => 2.0,
        UpscaleMode::UltraPerformance => 3.0,
    }
}

/// The mode whose scale is nearest rendering at `render` for `output`, compared as render over
/// output, as DLSS reports it; only a DLAA request runs at the output size.
fn nearest_mode(render: [u32; 2], output: [u32; 2]) -> UpscaleMode {
    let scale = f64::from(render[0]) / f64::from(output[0]);
    [
        UpscaleMode::Quality,
        UpscaleMode::Balanced,
        UpscaleMode::Performance,
        UpscaleMode::UltraPerformance,
    ]
    .into_iter()
    .min_by(|a, b| {
        (1.0 / mode_scale(*a) - scale)
            .abs()
            .total_cmp(&(1.0 / mode_scale(*b) - scale).abs())
    })
    .expect("the list is not empty")
}

/// The mode that suits `output`: Quality up to 1440p, Performance up to 4K, Ultra Performance
/// above.
fn auto_mode(output: [u32; 2]) -> UpscaleMode {
    let pixels = u64::from(output[0]) * u64::from(output[1]);
    if pixels <= 2560 * 1440 {
        UpscaleMode::Quality
    } else if pixels <= 3840 * 2160 {
        UpscaleMode::Performance
    } else {
        UpscaleMode::UltraPerformance
    }
}

impl ReconstructionBackend for MetalFx {
    fn support(&self) -> ReconstructionSupport {
        ReconstructionSupport {
            available: self.denoised.is_some(),
            dlaa: self.denoised.is_some_and(|r| r.contains(1.0)),
            min_output: [1, 1],
            scale: self.denoised.map(|r| [r.min, r.max]),
        }
    }

    fn reads(&self) -> GuideReads {
        GuideReads {
            roughness: true,
            specular_hit_distance: true,
        }
    }

    fn feature(
        &self,
        output: [u32; 2],
        quality: UpscaleQuality,
    ) -> Result<Box<dyn ReconstructionFeature>> {
        let mode = match quality {
            UpscaleQuality::Auto => auto_mode(output),
            UpscaleQuality::Fixed(m) => m,
        };
        let scale = mode_scale(mode);
        let render = output.map(|e| ((f64::from(e) / scale).round() as u32).max(1));
        self.denoiser(render, output, ReconstructionMode::Upscale(mode))
    }

    fn feature_at(
        &self,
        render: [u32; 2],
        output: [u32; 2],
    ) -> Result<Box<dyn ReconstructionFeature>> {
        self.denoiser(
            render,
            output,
            ReconstructionMode::Upscale(nearest_mode(render, output)),
        )
    }

    fn dlaa(&self, output: [u32; 2]) -> Result<Box<dyn ReconstructionFeature>> {
        if !self.support().dlaa {
            return Err(Error::DlaaUnavailable);
        }
        self.denoiser(output, output, ReconstructionMode::Dlaa)
    }
}

struct Denoiser {
    scaler: DenoisedScaler,
    mode: ReconstructionMode,
    phases: u32,
    /// The camera position the matrices' world is relative to, from the last history reset.
    anchor: Option<DVec3>,
}

impl ReconstructionFeature for Denoiser {
    fn mode(&self) -> ReconstructionMode {
        self.mode
    }

    fn render_size(&self) -> [u32; 2] {
        self.scaler.render_size()
    }

    fn output_size(&self) -> [u32; 2] {
        self.scaler.output_size()
    }

    fn jitter(&self, frame: u64) -> [f32; 2] {
        jitter(frame, self.phases)
    }

    fn mip_bias(&self) -> f32 {
        mip_bias(self.scaler.render_size(), self.scaler.output_size())
    }

    fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &ReconstructionInputs<'_>,
    ) -> Result<()> {
        let view = inputs.view;
        let anchor = match self.anchor {
            Some(a) if !inputs.reset => a,
            _ => *self.anchor.insert(view.position),
        };
        let world_from_view =
            DMat4::from_rotation_translation(view.rotation.as_dquat(), view.position - anchor);
        let clip_from_view = view.projection.clip_from_view(view.output_extent);
        let cols = |m: DMat4| Float4x4::from_cols_array_2d(&m.as_mat4().to_cols_array_2d());
        let roughness = inputs
            .roughness
            .expect("MetalFX reads the roughness guide, which views then hold");
        self.scaler.record(
            list,
            &DenoisedInputs {
                color: inputs.color,
                depth: inputs.depth,
                motion: inputs.motion,
                diffuse_albedo: inputs.diffuse_albedo,
                specular_albedo: inputs.specular_albedo,
                normal: inputs.normal_roughness,
                roughness,
                specular_hit_distance: inputs.specular_hit_distance,
                transparency: inputs.transparency,
                output: inputs.output,
                reset: inputs.reset,
                jitter: view.jitter.to_array(),
                world_to_view: cols(world_from_view.inverse()),
                view_to_clip: Float4x4::from_cols_array_2d(&clip_from_view.to_cols_array_2d()),
            },
        )?;
        Ok(())
    }
}
