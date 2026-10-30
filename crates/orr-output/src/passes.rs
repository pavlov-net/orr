//! Output's compute passes, recorded into a command list: exposure (meter and adapt),
//! tonemap and compose.

use std::ops::RangeInclusive;

use bytemuck::Pod;
use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, MemoryKind, OutputFormat,
    ShaderStage, SpecData, Stages, StorageImageIndex, TextureIndex,
};
use orr_scene::{SkyStats, ViewConstants};

use crate::exposure::{COMPENSATION_ENTRIES, HISTOGRAM_BINS};
use crate::shaders::compose;
use crate::shaders::exposure::adapt::{self, AdaptParams, ExposureTelemetryGpu};
use crate::shaders::exposure::meter;
use crate::shaders::tonemap::{self, TonemapConstants};
use crate::{
    AutoExposure, CompensationCurve, Exposure, ExposureTelemetry, Gt7Curve, MeterWeights,
    OutputDebug, ResolvedDisplay, Result, SkyTelemetry, ViewOutput,
};

/// Output's compute pipelines on one device.
pub struct OutputKernels {
    meter: ComputePipeline,
    adapt: ComputePipeline,
    tonemap: ComputePipeline,
    compose: ComputePipeline,
}

/// Words of the meter's histogram (`exposure.slang`).
const HISTOGRAM_WORDS: u64 = 3 * HISTOGRAM_BINS as u64;

/// Byte offsets in `OutputRoots::upload`. Roots sit at 256 B, the uniform-buffer offset
/// alignment.
const TONEMAP_ROOT: u64 = 0;
const TONEMAP_CONSTANTS: u64 = 256;
const COMPOSE_ROOT: u64 = 512;
const METER_ROOT: u64 = 768;
const ADAPT_ROOT: u64 = 1024;
const ADAPT_PARAMS: u64 = 1280;
const COMPENSATION: u64 = 1536;
const ZEROS: u64 = COMPENSATION + 4 * COMPENSATION_ENTRIES as u64;
const UPLOAD_BYTES: u64 = ZEROS + 4 * HISTOGRAM_WORDS;

/// GPU memory for one view's output passes in one frame. Recording a pass writes its root
/// here, so reuse it only once the list that last recorded with it has completed; an
/// [`OutputView`] keeps one per frame in flight.
pub struct OutputRoots {
    upload: Allocation<u8>,
    histogram: Allocation<u32>,
    histogram_cleared: bool,
    telemetry: Allocation<ExposureTelemetryGpu>,
    telemetry_frame: Option<u64>,
    compensation: Option<(CompensationCurve, RangeInclusive<f32>)>,
}

impl OutputRoots {
    pub fn new(gpu: &Device) -> Result<Self> {
        let mut upload = gpu.gpu_malloc::<u8>(MemoryKind::Upload, UPLOAD_BYTES, "output roots")?;
        upload.as_mut_slice().expect("upload memory").fill(0);
        Ok(Self {
            upload,
            histogram: gpu.gpu_malloc(MemoryKind::DeviceLocal, HISTOGRAM_WORDS, "histogram")?,
            histogram_cleared: false,
            telemetry: gpu.gpu_malloc(MemoryKind::Readback, 1, "exposure telemetry")?,
            telemetry_frame: None,
            compensation: None,
        })
    }

    /// Writes `value` at byte `at` of the upload block and returns its address.
    fn put<T: Pod>(&mut self, at: u64, value: &T) -> GpuPtr<T> {
        let bytes = bytemuck::bytes_of(value);
        let slice = self.upload.as_mut_slice().expect("upload memory");
        slice[at as usize..at as usize + bytes.len()].copy_from_slice(bytes);
        self.upload.gpu.offset(at).cast()
    }
}

/// Output's state for one view history: its [`OutputRoots`] per frame in flight and the
/// exposure telemetry they read back.
pub struct OutputView {
    frames: Vec<OutputRoots>,
}

impl OutputView {
    pub fn new(gpu: &Device, frames_in_flight: u32) -> Result<Self> {
        let frames = (0..frames_in_flight.max(1))
            .map(|_| OutputRoots::new(gpu))
            .collect::<Result<_>>()?;
        Ok(Self { frames })
    }

    /// The roots frame `frame` records with. Its slot was last used by frame
    /// `frame - frames_in_flight`, which must have completed.
    pub fn roots(&mut self, frame: u64) -> &mut OutputRoots {
        let n = self.frames.len() as u64;
        &mut self.frames[(frame % n) as usize]
    }

    /// The exposure telemetry of the newest frame at or before `completed` that ran the
    /// exposure passes. `completed` must be a frame whose work has completed on the GPU.
    pub fn telemetry(&self, completed: u64) -> Option<ExposureTelemetry> {
        let roots = self
            .frames
            .iter()
            .filter(|r| r.telemetry_frame.is_some_and(|f| f <= completed))
            .max_by_key(|r| r.telemetry_frame)?;
        let t = roots.telemetry.as_slice().expect("readback memory")[0];
        Some(ExposureTelemetry {
            frame: roots.telemetry_frame?,
            adapted_ev100: t.adapted_ev100,
            target_ev100: t.target_ev100,
            long_term_ev100: t.long_term_ev100,
            metered_ev100: (t.metered != 0).then_some(t.metered_ev100),
            pre_exposure: t.pre_exposure,
            white_point_xy: t.white_point_xy,
            sky: SkyTelemetry {
                dome_luminance: t.sky.dome_luminance,
                dome_xy: t.sky.dome_xy,
                directional_lux: [
                    t.sky.directional[0],
                    t.sky.directional[1],
                    t.sky.directional[2],
                ],
            },
        })
    }
}

/// Metering and adapting one view's exposure.
#[derive(Clone, Copy, Debug)]
pub struct ExposurePass<'a> {
    /// The view's pre-exposed radiance, linear Rec.2020.
    pub source: TextureIndex,
    pub extent: [u32; 2],
    /// The render's view constants: the exposure slots and `delta_time`.
    pub view: GpuPtr<ViewConstants>,
    /// The scene's sky statistics, for multi-reference metering and telemetry; null without
    /// a scene.
    pub sky: GpuPtr<SkyStats>,
    /// orr's frame id, which the telemetry carries.
    pub frame: u64,
    pub output: &'a ViewOutput,
}

/// Tonemapping one image: pre-exposed linear Rec.2020 radiance in, display-linear Rec.709
/// (1.0 = paper white) out, at the exposure the view's adapt pass wrote.
#[derive(Clone, Copy, Debug)]
pub struct TonemapPass<'a> {
    /// Radiance, read at each output texel.
    pub source: TextureIndex,
    /// An RGBA16F (or wider) texture of `extent`.
    pub target: StorageImageIndex,
    pub extent: [u32; 2],
    pub view: GpuPtr<ViewConstants>,
    pub display: &'a ResolvedDisplay,
    pub output: &'a ViewOutput,
}

/// Encoding display-linear Rec.709 for the display.
#[derive(Clone, Copy, Debug)]
pub struct ComposePass<'a> {
    /// The tonemapped image.
    pub source: TextureIndex,
    /// The backbuffer, in the format of `display.encoding`.
    pub target: StorageImageIndex,
    pub extent: [u32; 2],
    pub display: &'a ResolvedDisplay,
}

impl OutputKernels {
    pub fn new(gpu: &Device) -> Result<Self> {
        let pipeline = |module, name| -> Result<ComputePipeline> {
            let stage = ShaderStage::new(module, "main", SpecData::default())?;
            Ok(gpu.create_compute_pipeline(&stage, name)?)
        };
        Ok(Self {
            meter: pipeline(&meter::MODULE, "meter")?,
            adapt: pipeline(&adapt::MODULE, "adapt")?,
            tonemap: pipeline(&tonemap::MODULE, "tonemap")?,
            compose: pipeline(&compose::MODULE, "compose")?,
        })
    }

    /// Records `meter` (auto exposure only) and `adapt`, which writes the view's next
    /// exposure slot and the telemetry. Order the source's writes before it with a barrier;
    /// it ends with a barrier, so tonemap may follow directly.
    pub fn exposure(
        &self,
        list: &mut CommandList<'_>,
        roots: &mut OutputRoots,
        pass: &ExposurePass<'_>,
    ) {
        if !roots.histogram_cleared {
            let zeros = roots.upload.gpu.offset(ZEROS);
            list.copy(roots.histogram.gpu.cast(), zeros, 4 * HISTOGRAM_WORDS);
            list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
            roots.histogram_cleared = true;
        }
        let params = match &pass.output.exposure {
            Exposure::Manual { ev100 } => AdaptParams {
                mode: 0,
                manual_ev100: *ev100,
                cut: u32::from(pass.output.cut),
                ..bytemuck::Zeroable::zeroed()
            },
            Exposure::Auto(auto) => {
                let params = adapt_params(auto, pass.output.cut);
                let root = meter::Root {
                    histogram: roots.histogram.gpu,
                    view: pass.view,
                    source: pass.source,
                    mask: auto.metering_mask.unwrap_or(TextureIndex::NONE),
                    width: pass.extent[0],
                    height: pass.extent[1],
                    ev_min: params.ev_min,
                    bins_per_ev: 1.0 / params.bin_ev,
                    _pad0: 0,
                    _pad1: 0,
                };
                let root = roots.put(METER_ROOT, &root);
                list.begin_pass("meter", false);
                list.dispatch(&self.meter, root, groups(pass.extent, 16));
                list.end_pass();
                list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
                let stale = roots.compensation.as_ref().is_none_or(|(curve, range)| {
                    *curve != auto.compensation || *range != auto.range_ev100
                });
                if stale {
                    let lut = auto.compensation.bake(&auto.range_ev100);
                    roots.put(COMPENSATION, &lut);
                    roots.compensation =
                        Some((auto.compensation.clone(), auto.range_ev100.clone()));
                }
                params
            }
        };
        let root = adapt::Root {
            params: roots.put(ADAPT_PARAMS, &params),
            compensation: roots.upload.gpu.offset(COMPENSATION).cast(),
            histogram: roots.histogram.gpu,
            view: pass.view,
            sky: pass.sky,
            telemetry: roots.telemetry.gpu,
        };
        let root = roots.put(ADAPT_ROOT, &root);
        list.begin_pass("adapt", false);
        list.dispatch(&self.adapt, root, [1, 1, 1]);
        list.end_pass();
        list.barrier(
            Stages::COMPUTE,
            Stages::COMPUTE | Stages::HOST,
            Hazards::empty(),
        );
        roots.telemetry_frame = Some(pass.frame);
    }

    /// Records the tonemap pass. Order its source's and the adapt pass's writes before it
    /// with a barrier.
    pub fn tonemap(
        &self,
        list: &mut CommandList<'_>,
        roots: &mut OutputRoots,
        pass: &TonemapPass<'_>,
    ) {
        let d = pass.display;
        let o = pass.output;
        let row = |r: [f32; 3]| [r[0], r[1], r[2], 0.0];
        // DEBUG_* in tonemap.slang.
        let (debug, pattern) = match o.debug {
            OutputDebug::None => (0, None),
            OutputDebug::Linear => (1, None),
            OutputDebug::Pattern(p) => (2, Some(p.levels(d))),
            OutputDebug::ExposureHeatmap => (3, None),
        };
        let constants = TonemapConstants {
            curve: Gt7Curve::new(&o.gt7, d.peak_nits, d.paper_white_nits),
            grade_r: row(o.grade.matrix[0]),
            grade_g: row(o.grade.matrix[1]),
            grade_b: row(o.grade.matrix[2]),
            grade_scale: o.grade.exposure_ev.exp2(),
            sdr: u32::from(d.encoding == OutputFormat::Sdr),
            debug,
            pattern_patch: pattern.map_or(0.0, |p| p.patch),
            pattern_surround: pattern.map_or(0.0, |p| p.surround),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let root = tonemap::Root {
            constants: roots.put(TONEMAP_CONSTANTS, &constants),
            view: pass.view,
            source: pass.source,
            target: pass.target,
            width: pass.extent[0],
            height: pass.extent[1],
        };
        let root = roots.put(TONEMAP_ROOT, &root);
        list.begin_pass("tonemap", false);
        list.dispatch(&self.tonemap, root, groups(pass.extent, 8));
        list.end_pass();
    }

    /// Records the compose pass. Order its source's writes before it with a barrier.
    pub fn compose(
        &self,
        list: &mut CommandList<'_>,
        roots: &mut OutputRoots,
        pass: &ComposePass<'_>,
    ) {
        // ENCODING_* in display.slang.
        let encoding = match pass.display.encoding {
            OutputFormat::Sdr => 0,
            OutputFormat::Hdr10Pq => 1,
            OutputFormat::ScRgb => 2,
        };
        let root = compose::Root {
            source: pass.source,
            target: pass.target,
            width: pass.extent[0],
            height: pass.extent[1],
            encoding,
            paper_white_nits: pass.display.paper_white_nits,
            _pad0: 0,
            _pad1: 0,
        };
        let root = roots.put(COMPOSE_ROOT, &root);
        list.begin_pass("compose", false);
        list.dispatch(&self.compose, root, groups(pass.extent, 8));
        list.end_pass();
    }
}

fn adapt_params(auto: &AutoExposure, cut: bool) -> AdaptParams {
    let range = &auto.range_ev100;
    let weights = auto.references.unwrap_or(MeterWeights {
        histogram: 1.0,
        dome: 0.0,
        directional: 0.0,
    });
    let long = auto.physiological.unwrap_or_default();
    AdaptParams {
        mode: 1,
        manual_ev100: 0.0,
        cut: u32::from(cut),
        physiological: u32::from(auto.physiological.is_some()),
        ev_min: *range.start(),
        bin_ev: (range.end() - range.start()) / HISTOGRAM_BINS as f32,
        filter_low: *auto.filter.start(),
        filter_high: *auto.filter.end(),
        clamp_min: *auto.clamp_ev100.start(),
        clamp_max: *auto.clamp_ev100.end(),
        speed_brighten: auto.speed_brighten,
        speed_darken: auto.speed_darken,
        exponential_ev: auto.exponential_ev,
        weight_histogram: weights.histogram,
        weight_dome: weights.dome,
        weight_directional: weights.directional,
        long_speed_brighten: long.speed_brighten,
        long_speed_darken: long.speed_darken,
        bound_brighten: long.bound_brighten,
        bound_darken: long.bound_darken,
    }
}

/// `size` x `size` workgroups covering `extent`.
fn groups(extent: [u32; 2], size: u32) -> [u32; 3] {
    [extent[0].div_ceil(size), extent[1].div_ceil(size), 1]
}
