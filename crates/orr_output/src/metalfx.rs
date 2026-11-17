//! MetalFX as the backend of ray reconstruction, temporal upscaling and frame generation.

use orr_math::jitter::{jitter, jitter_phases, mip_bias};
use orr_math::{DMat4, DVec3};
use orr_metalfx::{
    DenoisedInputs, DenoisedScaler, Float4x4, FrameInterpolator, InterpolatorInputs, ScaleRange,
    TemporalInputs, TemporalScaler,
};
use orr_scene::orr_gpu::{CommandList, Device, Texture};

use crate::frame_generation::FAR_M;
use crate::{
    Error, FrameGenerationBackend, FrameGenerationCamera, FrameGenerationFeature,
    FrameGenerationInputs, GuideReads, InterpolationCheck, ReconstructionBackend,
    ReconstructionFeature, ReconstructionInputs, ReconstructionMode, ReconstructionSupport, Result,
    UpscaleBackend, UpscaleFeature, UpscaleInputs, UpscaleMode, UpscaleQuality,
};

/// MetalFX's effects on one device.
pub struct MetalFx {
    device: Device,
    denoised: Option<ScaleRange>,
    temporal: Option<ScaleRange>,
}

impl MetalFx {
    pub fn new(device: &Device) -> Self {
        let support = orr_metalfx::support(device);
        Self {
            device: device.clone(),
            denoised: support.denoised,
            temporal: support.temporal,
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

/// The render size of `quality` at `output`, and the mode it names.
fn quality_render(output: [u32; 2], quality: UpscaleQuality) -> ([u32; 2], UpscaleMode) {
    let mode = match quality {
        UpscaleQuality::Auto => auto_mode(output),
        UpscaleQuality::Fixed(m) => m,
    };
    let scale = mode_scale(mode);
    (
        output.map(|e| ((f64::from(e) / scale).round() as u32).max(1)),
        mode,
    )
}

fn range_support(range: Option<ScaleRange>) -> ReconstructionSupport {
    ReconstructionSupport {
        available: range.is_some(),
        dlaa: range.is_some_and(|r| r.contains(1.0)),
        min_output: [1, 1],
        scale: range.map(|r| [r.min, r.max]),
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
        range_support(self.denoised)
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
        let (render, mode) = quality_render(output, quality);
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

impl MetalFx {
    fn temporal(
        &self,
        render: [u32; 2],
        output: [u32; 2],
        mode: ReconstructionMode,
    ) -> Result<Box<dyn UpscaleFeature>> {
        let scaler = TemporalScaler::new(&self.device, render, output)?;
        Ok(Box::new(Upscaler {
            scaler,
            mode,
            phases: jitter_phases(render, output),
        }))
    }
}

impl UpscaleBackend for MetalFx {
    fn upscale_support(&self) -> ReconstructionSupport {
        range_support(self.temporal)
    }

    fn upscaler(
        &self,
        output: [u32; 2],
        quality: UpscaleQuality,
    ) -> Result<Box<dyn UpscaleFeature>> {
        let (render, mode) = quality_render(output, quality);
        self.temporal(render, output, ReconstructionMode::Upscale(mode))
    }

    fn upscaler_at(&self, render: [u32; 2], output: [u32; 2]) -> Result<Box<dyn UpscaleFeature>> {
        self.temporal(
            render,
            output,
            ReconstructionMode::Upscale(nearest_mode(render, output)),
        )
    }

    fn antialiaser(&self, output: [u32; 2]) -> Result<Box<dyn UpscaleFeature>> {
        if !self.upscale_support().dlaa {
            return Err(Error::DlaaUnavailable);
        }
        self.temporal(output, output, ReconstructionMode::Dlaa)
    }
}

struct Upscaler {
    scaler: TemporalScaler,
    mode: ReconstructionMode,
    phases: u32,
}

impl UpscaleFeature for Upscaler {
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

    fn record(&mut self, list: &mut CommandList<'_>, inputs: &UpscaleInputs<'_>) -> Result<()> {
        self.scaler.record(
            list,
            &TemporalInputs {
                color: inputs.color,
                depth: inputs.depth,
                motion: inputs.motion,
                exposure: inputs.exposure,
                pre_exposure: inputs.pre_exposure,
                output: inputs.output,
                reset: inputs.reset,
                jitter: inputs.jitter,
            },
        )?;
        Ok(())
    }
}

/// Column-major `m` as MetalFX takes it.
fn float4x4(m: DMat4) -> Float4x4 {
    Float4x4::from_cols_array_2d(&m.as_mat4().to_cols_array_2d())
}

/// View from world, with world relative to `anchor`.
fn view_from_world(camera: &FrameGenerationCamera, anchor: DVec3) -> DMat4 {
    DMat4::from_rotation_translation(camera.rotation.as_dquat(), camera.position - anchor).inverse()
}

impl FrameGenerationBackend for MetalFx {
    /// One interpolated image per rendered frame, on a device with Metal 4 frame interpolation
    /// whose presents can be metered.
    fn max_generated(&self, gpu: &Device, _low_latency: bool) -> Result<u32> {
        if !FrameInterpolator::supported(gpu) {
            Err(Error::FrameGenerationUnavailable(
                "MetalFX frame interpolation",
            ))
        } else if !gpu.caps().present_metering {
            Err(Error::FrameGenerationUnavailable("metered presents"))
        } else {
            Ok(1)
        }
    }

    fn feature(
        &self,
        _backbuffer: &Texture,
        _hdr: bool,
    ) -> Result<Box<dyn FrameGenerationFeature>> {
        Ok(Box::new(Interpolator(FrameInterpolator::new(
            &self.device,
        )?)))
    }

    fn interpolation_check(&self) -> Option<InterpolationCheck> {
        None
    }
}

struct Interpolator(FrameInterpolator);

impl FrameGenerationFeature for Interpolator {
    fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &FrameGenerationInputs<'_>,
    ) -> Result<()> {
        let [output] = inputs.outputs else {
            return Err(Error::FrameGenerationMode(
                crate::FrameGenerationMode::Fixed(inputs.outputs.len() as u32),
            ));
        };
        let camera = &inputs.camera;
        let previous = inputs.previous_backbuffer;
        self.0.record(
            list,
            &InterpolatorInputs {
                color: inputs.backbuffer,
                previous_color: previous.unwrap_or(inputs.backbuffer),
                depth: inputs.depth,
                motion: inputs.motion,
                distortion: inputs.distortion,
                output,
                reset: inputs.reset || previous.is_none(),
                jitter: inputs.jitter,
                delta_time: inputs.delta_time,
                near: camera.projection.near,
                far: FAR_M,
                fov_y: camera.projection.fov_y,
                aspect: camera.output[0] as f32 / camera.output[1].max(1) as f32,
                world_to_view: float4x4(view_from_world(camera, inputs.anchor)),
                view_to_clip: float4x4(camera.clip_from_view()),
            },
        )?;
        Ok(())
    }
}

// SAFETY: the feature is recorded by one thread at a time (`record` takes `&mut self`), and
// shared references to it touch nothing.
unsafe impl Sync for Interpolator {}
