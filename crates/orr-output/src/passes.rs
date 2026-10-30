//! Output's compute passes, recorded into a command list: hook inputs, exposure and white
//! balance (meter and adapt), tonemap, compose and compose-UI.

use std::ops::RangeInclusive;

use bytemuck::Pod;
use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, MemoryKind, OutputFormat,
    ShaderStage, SpecData, Stages, StorageImageIndex, TextureIndex,
};
use orr_scene::{SkyStats, ViewConstants};

use crate::exposure::{COMPENSATION_ENTRIES, HISTOGRAM_BINS, HISTOGRAM_WORDS_PER_BIN};
use crate::hooks::HookTargets;
use crate::shaders::compose;
use crate::shaders::exposure::adapt::{self, AdaptParams, ExposureTelemetryGpu};
use crate::shaders::exposure::meter;
use crate::shaders::hook_inputs;
use crate::shaders::tonemap::{self, TonemapConstants};
use crate::{
    AutoExposure, ChainImages, CompensationCurve, DistortionField, Exposure, ExposureTelemetry,
    Gt7Curve, HookInputs, Hooks, MeterWeights, OutputDebug, ResolvedDisplay, Result, SkyTelemetry,
    ViewOutput,
};

/// Output's compute pipelines on one device.
pub struct OutputKernels {
    hook_inputs: ComputePipeline,
    meter: ComputePipeline,
    adapt: ComputePipeline,
    tonemap: ComputePipeline,
    compose: ComputePipeline,
}

/// Words of the meter's histogram (`exposure.slang`).
const HISTOGRAM_WORDS: u64 = HISTOGRAM_WORDS_PER_BIN * HISTOGRAM_BINS as u64;

/// Byte offsets in `OutputRoots::upload`. Roots sit at 256 B, the uniform-buffer offset
/// alignment.
const TONEMAP_ROOT: u64 = 0;
const TONEMAP_CONSTANTS: u64 = 256;
const METER_ROOT: u64 = 512;
const ADAPT_ROOT: u64 = 768;
const ADAPT_PARAMS: u64 = 1024;
const HOOK_ROOT: u64 = 1280;
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
    white_balance: Allocation<[f32; 4]>,
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
            white_balance: gpu.gpu_malloc(MemoryKind::DeviceLocal, 3, "white balance")?,
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

/// Output's state for one view history: its [`OutputRoots`] per frame in flight, the
/// exposure telemetry they read back and, while the view sets a hook, the hook inputs.
pub struct OutputView {
    frames: Vec<OutputRoots>,
    hooks: Option<HookTargets>,
    device: Device,
}

impl OutputView {
    pub fn new(gpu: &Device, frames_in_flight: u32) -> Result<Self> {
        let frames = (0..frames_in_flight.max(1))
            .map(|_| OutputRoots::new(gpu))
            .collect::<Result<_>>()?;
        Ok(Self {
            frames,
            hooks: None,
            device: gpu.clone(),
        })
    }

    /// Bytes of the hook inputs held for the view: 0 unless its last frame set a hook.
    pub fn hook_input_bytes(&self) -> u64 {
        self.hooks.as_ref().map_or(0, HookTargets::bytes)
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
            white_target_xy: t.white_target_xy,
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

/// Building a view's hook inputs from lighting's render-resolution targets.
#[derive(Clone, Copy, Debug)]
pub struct HookInputsPass {
    /// Lighting's `rr_depth`: R32F reverse-Z at render resolution.
    pub rr_depth: TextureIndex,
    /// Lighting's `rr_motion`: RG16F previous minus current UV at render resolution.
    pub rr_motion: TextureIndex,
    pub chain: ChainImages,
    /// The output extent.
    pub extent: [u32; 2],
    /// The render's view constants: the render extent and jitter.
    pub view: GpuPtr<ViewConstants>,
    /// The frame's jitter in render pixels, as in the view constants.
    pub jitter: [f32; 2],
    /// orr's frame id.
    pub frame: u64,
    pub hooks: Hooks,
}

/// Tonemapping one image: pre-exposed linear Rec.2020 radiance in, display-linear Rec.709
/// (1.0 = paper white) out, at the exposure and white balance the view's adapt pass wrote
/// with the same [`OutputRoots`].
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
    /// With a `display_linear` hook, the chain's other image ([`ChainImages::rr_output_store`]),
    /// which tonemap writes too as the hook's source.
    pub hook_source: Option<StorageImageIndex>,
}

/// What compose reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComposeSource {
    /// Display-linear Rec.709, 1.0 = paper white, as tonemap writes it.
    Linear(TextureIndex),
    /// An image already in the display's encoding: the UI-free backbuffer or an interpolated
    /// frame. Texels with no UI over them are copied unchanged.
    Encoded(TextureIndex),
    /// Black, for a frame that renders no view.
    Black,
}

/// Encoding an image for the display, with the UI layer blended over it in extended sRGB
/// relative to paper white: `ui + (1 - ui.a) * srgb_encode(linear)`. SDR is exactly egui's
/// gamma blending; HDR keeps the UI's SDR look at paper white.
///
/// Compose writes the UI-free backbuffer from tonemap's output (`ui: None`); compose-UI
/// blends the frame's UI layer over each presented image (`Encoded`), or over black for a
/// UI-only frame.
#[derive(Clone, Copy, Debug)]
pub struct ComposePass<'a> {
    pub source: ComposeSource,
    /// The UI layer ([`crate::UiLayer::index`]).
    pub ui: Option<TextureIndex>,
    /// The output image, in the format of `display.encoding`.
    pub target: StorageImageIndex,
    pub extent: [u32; 2],
    pub display: &'a ResolvedDisplay,
}

/// Upload memory for one compose pass's root. Recording writes it, so reuse it only once
/// the list that last recorded with it has completed: a display keeps one per presented
/// image and frame in flight.
pub struct ComposeRoot {
    upload: Allocation<compose::Root>,
}

impl ComposeRoot {
    pub fn new(gpu: &Device) -> Result<Self> {
        Ok(Self {
            upload: gpu.gpu_malloc(MemoryKind::Upload, 1, "compose root")?,
        })
    }
}

impl OutputKernels {
    pub fn new(gpu: &Device) -> Result<Self> {
        let pipeline = |module, name| -> Result<ComputePipeline> {
            let stage = ShaderStage::new(module, "main", SpecData::default())?;
            Ok(gpu.create_compute_pipeline(&stage, name)?)
        };
        Ok(Self {
            hook_inputs: pipeline(&hook_inputs::MODULE, "hook_inputs")?,
            meter: pipeline(&meter::MODULE, "meter")?,
            adapt: pipeline(&adapt::MODULE, "adapt")?,
            tonemap: pipeline(&tonemap::MODULE, "tonemap")?,
            compose: pipeline(&compose::MODULE, "compose")?,
        })
    }

    /// Records the `hook_inputs` pass when the view sets a hook: depth and motion at output
    /// resolution, with a `scene_referred` hook the copy of RR's output it reads as its
    /// source, and with `field` the field cleared to identity. Allocates the inputs on the
    /// first frame that sets a hook and frees them on the first that sets none, which
    /// records nothing. Order the writes of lighting's targets and RR before it with a
    /// barrier; [`HookInputs::run`] orders its writes before the hooks.
    pub fn hook_inputs(
        &self,
        list: &mut CommandList<'_>,
        view: &mut OutputView,
        pass: &HookInputsPass,
        field: Option<&mut DistortionField>,
    ) -> Result<Option<HookInputs>> {
        if !pass.hooks.any() {
            view.hooks = None;
            return Ok(None);
        }
        if view
            .hooks
            .as_ref()
            .is_none_or(|t| t.extent() != pass.extent)
        {
            view.hooks = Some(HookTargets::new(&view.device, pass.extent)?);
        }
        let targets = view.hooks.as_mut().expect("hook targets");
        if std::mem::take(&mut targets.fresh) {
            list.discard(targets.depth.texture());
            list.discard(targets.motion.texture());
        }
        let field = field.map_or(StorageImageIndex::NONE, |f| {
            assert_eq!(
                f.extent(),
                [pass.extent[0].div_ceil(2), pass.extent[1].div_ceil(2)],
                "distortion field for another extent"
            );
            f.prepare(list);
            f.current_storage()
        });
        let (copy_source, copy_target) = if pass.hooks.scene_referred {
            (pass.chain.rr_output, pass.chain.linear_store)
        } else {
            (TextureIndex::NONE, StorageImageIndex::NONE)
        };
        let inputs = HookInputs {
            hooks: pass.hooks,
            chain: pass.chain,
            depth: targets.depth.sampled,
            motion: targets.motion.sampled,
            extent: pass.extent,
            view: pass.view,
            jitter: pass.jitter,
        };
        let root = hook_inputs::Root {
            view: pass.view,
            rr_depth: pass.rr_depth,
            rr_motion: pass.rr_motion,
            depth: targets.depth.storage,
            motion: targets.motion.storage,
            copy_source,
            copy_target,
            field,
            width: pass.extent[0],
            height: pass.extent[1],
            _pad: 0,
        };
        let root = view.roots(pass.frame).put(HOOK_ROOT, &root);
        list.begin_pass("hook_inputs", false);
        list.dispatch(&self.hook_inputs, root, groups(pass.extent, 8));
        list.end_pass();
        Ok(Some(inputs))
    }

    /// Records `meter` (with auto exposure or auto white balance) and `adapt`, which writes
    /// the view's next exposure slot, the white balance tonemap applies and the telemetry.
    /// Order the source's writes before it with a barrier; it ends with a barrier, so tonemap
    /// may follow directly.
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
        let output = pass.output;
        let default_meter = AutoExposure::default();
        let (mut params, meter) = match &output.exposure {
            Exposure::Manual { ev100 } => {
                let params = AdaptParams {
                    mode: 0,
                    manual_ev100: *ev100,
                    ..adapt_params(&default_meter, output.cut)
                };
                (params, output.white_balance.map(|_| &default_meter))
            }
            Exposure::Auto(auto) => {
                let stale = roots.compensation.as_ref().is_none_or(|(curve, range)| {
                    *curve != auto.compensation || *range != auto.range_ev100
                });
                if stale {
                    let lut = auto.compensation.bake(&auto.range_ev100);
                    roots.put(COMPENSATION, &lut);
                    roots.compensation =
                        Some((auto.compensation.clone(), auto.range_ev100.clone()));
                }
                (adapt_params(auto, output.cut), Some(auto))
            }
        };
        if let Some(white) = &output.white_balance {
            let weights = white.references.unwrap_or(HISTOGRAM_ONLY);
            params.white_balance = 1;
            params.white_speed = white.speed;
            params.virtual_light_nits = white.virtual_light_nits;
            params.white_histogram = weights.histogram;
            params.white_dome = weights.dome;
            params.white_directional = weights.directional;
        }
        if let Some(auto) = meter {
            let root = meter::Root {
                histogram: roots.histogram.gpu,
                view: pass.view,
                source: pass.source,
                mask: auto.metering_mask.unwrap_or(TextureIndex::NONE),
                width: pass.extent[0],
                height: pass.extent[1],
                ev_min: params.ev_min,
                bins_per_ev: 1.0 / params.bin_ev,
                chroma: params.white_balance,
                _pad: 0,
            };
            let root = roots.put(METER_ROOT, &root);
            list.begin_pass("meter", false);
            list.dispatch(&self.meter, root, groups(pass.extent, 16));
            list.end_pass();
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        }
        let root = adapt::Root {
            params: roots.put(ADAPT_PARAMS, &params),
            compensation: roots.upload.gpu.offset(COMPENSATION).cast(),
            histogram: roots.histogram.gpu,
            view: pass.view,
            sky: pass.sky,
            telemetry: roots.telemetry.gpu,
            white_balance: roots.white_balance.gpu.cast(),
            _pad0: 0,
            _pad1: 0,
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
            white_balance: u32::from(o.white_balance.is_some()),
            _pad1: 0,
            _pad2: 0,
        };
        let root = tonemap::Root {
            constants: roots.put(TONEMAP_CONSTANTS, &constants),
            view: pass.view,
            white_balance: roots.white_balance.gpu.cast(),
            source: pass.source,
            target: pass.target,
            hook_source: pass.hook_source.unwrap_or(StorageImageIndex::NONE),
            width: pass.extent[0],
            height: pass.extent[1],
            _pad: 0,
        };
        let root = roots.put(TONEMAP_ROOT, &root);
        list.begin_pass("tonemap", false);
        list.dispatch(&self.tonemap, root, groups(pass.extent, 8));
        list.end_pass();
    }

    /// Records a compose pass, named `compose`, or `compose_ui` when it blends a UI layer
    /// over an encoded or black image. Order the writes of its source and UI layer before it
    /// with a barrier.
    pub fn compose(
        &self,
        list: &mut CommandList<'_>,
        root: &mut ComposeRoot,
        pass: &ComposePass<'_>,
    ) {
        // ENCODING_* in display.slang.
        let encoding = match pass.display.encoding {
            OutputFormat::Sdr => 0,
            OutputFormat::Hdr10Pq => 1,
            OutputFormat::ScRgb => 2,
        };
        // SOURCE_* in compose.slang.
        let (source, source_kind) = match pass.source {
            ComposeSource::Linear(t) => (t, 0),
            ComposeSource::Encoded(t) => (t, 1),
            ComposeSource::Black => (TextureIndex::NONE, 2),
        };
        root.upload.as_mut_slice().expect("upload memory")[0] = compose::Root {
            source,
            ui: pass.ui.unwrap_or(TextureIndex::NONE),
            target: pass.target,
            width: pass.extent[0],
            height: pass.extent[1],
            source_kind,
            encoding,
            paper_white_nits: pass.display.paper_white_nits,
        };
        let name = match pass.source {
            ComposeSource::Linear(_) => "compose",
            _ => "compose_ui",
        };
        list.begin_pass(name, false);
        list.dispatch(&self.compose, root.upload.gpu, groups(pass.extent, 8));
        list.end_pass();
    }
}

const HISTOGRAM_ONLY: MeterWeights = MeterWeights {
    histogram: 1.0,
    dome: 0.0,
    directional: 0.0,
};

fn adapt_params(auto: &AutoExposure, cut: bool) -> AdaptParams {
    let range = &auto.range_ev100;
    let weights = auto.references.unwrap_or(HISTOGRAM_ONLY);
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
        ..bytemuck::Zeroable::zeroed()
    }
}

/// `size` x `size` workgroups covering `extent`.
fn groups(extent: [u32; 2], size: u32) -> [u32; 3] {
    [extent[0].div_ceil(size), extent[1].div_ceil(size), 1]
}
