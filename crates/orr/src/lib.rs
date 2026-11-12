//! orr: a ray-tracing-first, deferred renderer.
//!
//! A [`Renderer`] holds global asset state and any number of scenes. Each frame runs in a
//! fixed order on one render thread:
//!
//! 1. [`Renderer::begin_frame`] waits until the frame `frames_in_flight` back has
//!    completed and opens the next one.
//! 2. The consumer writes scene changes through [`Frame::scene`] and [`Frame::materials`],
//!    from any threads, straight into mapped upload memory. A view that renders after step 3
//!    says so with [`Frame::prepare_view`], so the scene chooses geometry for its render size.
//! 3. [`Frame::update_scenes`] applies every write and runs the consumer's scene compute. The
//!    first [`Frame::render_view`] runs it when the consumer did not.
//! 4. [`Frame::render_view`] once per view records its raster half,
//! 5. then the view's lighting and output: the scene's lighting pass before its first view,
//!    one path per pixel, ray reconstruction, the hooks, exposure, motion blur, glare,
//!    tonemap, the UI and compose. A consumer may
//!    instead shade the G-buffer [`Frame::raster_view`] returns with its own kernels in
//!    [`Frame::list`] and show the result with [`Frame::compose`].
//! 6. [`Frame::end`] records the display's frame generation, submits the frame and hands
//!    it to the display's present thread.
//!
//! The path tracing kernels compile as jobs on the threads the consumer lends to
//! [`Renderer::loading`], and on a first run they take minutes. Until they finish a view renders
//! black, with [`RenderedView::compiling`] set: nothing blocks a frame, and the UI still draws.
//! See [`Renderer::create_view_history`].
//!
//! There is no scheduler, render graph or per-frame extraction: each step records directly
//! into the frame's command list on the Render queue, and the output passes into
//! [`Frame::output_list`]. On a device created with `QueueRequest::render_compute` that
//! list runs on the Compute queue after the frame's render work, while the next frame's
//! step 3 runs on the Render queue. A frame may list no scenes and render no view;
//! [`Frame::ui_only`] then presents its UI alone, over black or over a calibration pattern
//! from [`Frame::pattern`].

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub use orr_lighting::{
    self, CausticSettings, CausticStats, CompilingKernels, Error as LightingError, GuideSettings,
    LightingDebugView, LightingStats, MediaSettings, ParticleDraw, ParticleLighting, PathEntry,
    ReferenceSettings, SceneLightingSettings, ThinLens, ViewLighting, WorldCacheSettings,
};
pub use orr_output::DisplayWatch;
pub use orr_output::{self, Error as OutputError};
pub use orr_output::{
    AutoExposure, AutoWhiteBalance, CalibrationPattern, CompensationCurve, Display, DisplayFeed,
    DisplayHeadroom, DisplayInfo, DisplayRequest, Exposure, ExposureTelemetry,
    FrameGenerationBackend, FrameGenerationCamera, FrameGenerationFeature, FrameGenerationInputs,
    FrameGenerationMode, GlareMethod, Grade, Gt7Params, HookPass, InterpolationCheck,
    LocalExposure, MeterWeights, OutputDebug, Physiological, PostQuality, Presented, Primaries,
    Reconstruction, ReconstructionBackend, ReconstructionFeature, ReconstructionInputs,
    ReconstructionMode, ReconstructionSupport, ResolvedDisplay, Screenshot, ScreenshotRequest,
    SkyTelemetry, UiPass, UpscaleMode, UpscaleQuality, ViewOutput,
};
#[cfg(dlss)]
pub use orr_output::{DlssSdk, DlssSupport};
pub use orr_scene::Error as SceneError;
pub use orr_scene::{self, RasterView};
pub use orr_scene::{
    AffineRows, ArenaReport, ArenaTarget, BlobId, BlobTarget, BoundTexture, Budgets, ClassStats,
    ClipRef, ClusterTopology, Cone, ConsumerMarker, CopyVariant, Decal, DecalChannels, DecalId,
    DepthOfField, DetailStats, DirectionalLight, DiscProfile, EditDesc, EditId, EditRef,
    EnvironmentData, ExposureState, Focus, FrameStats, FrameTime, Glare, GroupWriter,
    InstanceFlags, InstancePage, InstancePageRef, InstanceRange, InstancesDesc, JobResult, Lens,
    LensDistortion, Light, LightId, LoadState, Loading, MaterialGpu, MaterialId, MaterialWriter,
    MediumData, MeshPipelineId, MeshStages, ModelId, MountId, NeuralTextures, Opacity, OpaqueHook,
    OpaquePass, OriginPolicy, PackSource, PageId, PipelineId, PoolSize, Pose, ProducerBatch,
    ProducerDesc, ProducerId, ProducerStats, ProducerUpdate, Projection, RasterGpu, ReloadReport,
    RendererDesc, RequestSource, ResidencyClass, ResidencyStats, RtStats, SceneCompute, SceneDesc,
    SceneId, SceneMemory, SceneRoot, SkeletonRef, SkinStats, SkinnedDesc, SkinnedPageRef, SkyStats,
    Span, TextureFilter, TextureSamplerGpu, TextureStreamGpu, TransformGroupDesc, TransformGroupId,
    TransparencyPass, ViewConstants, ViewDesc, ViewHistoryId, ViewStep, WorldPos, abi,
    binding_sampler, material_flags, orr_gpu, orr_pack,
};

use orr_gpu::{
    CommandList, Device, Format, GpuPtr, Hazards, KernelSet, OutputFormat, ShaderModule, Stages,
    StorageImageIndex, TextureDesc, TextureIndex, TextureUsage, TimelinePoint,
};
use orr_lighting::{
    BsdfTables, LightingKernels, LightingView, PathNeeds, SceneLighting, ScenePass, ViewPass,
};
use orr_math::{DVec3, Quat, UVec2, Vec2};
use orr_output::{
    CapturePass, ChainImages, ComposePass, ComposeRoot, ComposeSource, DisplayFrame, ExposurePass,
    GlarePass, HookInputsPass, HookPoint, Hooks, Image, MotionBlurPass, OutputKernels, OutputView,
    OverlayFront, SceneCapturePass, TonemapPass, UI_PASS,
};
use orr_scene::{ArenaRequest, Scenes};

mod probe;

pub use orr_lighting::OriginProbe;
use orr_lighting::{OriginProbes, TracedViews};
pub use orr_scene::{PlanOutcome, ProbeTap, ProbeTile, UnitState, UnitStatus};
pub use probe::PixelProbe;
use probe::{ProbeAt, record_probe, take_probe};

/// A scene, lighting or output failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Scene(#[from] SceneError),
    #[error(transparent)]
    Lighting(#[from] LightingError),
    #[error(transparent)]
    Output(#[from] OutputError),
    /// A view's explicit render extent has another aspect ratio than its output extent, so its
    /// pixels would not be square.
    #[error("render extent {render:?} does not keep the aspect ratio of the output {output:?}")]
    RenderAspect { render: [u32; 2], output: [u32; 2] },
    /// The path tracing kernels failed to compile. [`Frame::render_view`] and
    /// [`Renderer::wait_paths`] return this until [`Renderer::create_view_history`] starts the
    /// compile again.
    #[error(transparent)]
    PathKernels(Arc<LightingError>),
    /// The view's target is the display, and the frame has none, or its window has no area or
    /// on Metal is covered.
    #[error("the view targets a display, and the frame has none with any area")]
    NoDisplay,
}

impl From<orr_gpu::Error> for Error {
    fn from(e: orr_gpu::Error) -> Self {
        Self::Scene(e.into())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// What one frame renders, and where it goes.
pub struct FrameDesc<'a> {
    pub time: FrameTime,
    /// Scenes updated this frame, in order. Others keep their tables and record nothing.
    pub scenes: &'a [SceneId],
    /// The window the frame presents to. `begin_frame` sleeps on its low latency handle and
    /// opens its frame slot; `end` hands the frame to its present thread. Its
    /// `frames_in_flight` must equal the renderer's.
    pub display: Option<&'a mut Display>,
}

/// A view's transparency hook: draws into the pass's overlay, lighting and fogging particles
/// with the view's [`ParticleLighting`] of the frame.
pub type TransparencyHook<'a> =
    dyn FnMut(&mut TransparencyPass<'_, '_>, GpuPtr<ParticleLighting>) + 'a;

/// The consumer passes a view runs, each once per `render_view` that sets it.
#[derive(Default)]
pub struct ViewHooks<'a> {
    /// In each pass of the view's raster, into its visibility buffer: the draws of the scene's
    /// drawn producer entries (`ProducerDesc::drawn`), through the consumer's task and mesh
    /// pipelines. It runs once, or twice in a frame whose raster culls against the previous
    /// frame's depth, and also while the path tracing kernels compile.
    pub opaque: Option<&'a mut OpaqueHook<'a>>,
    /// Into the view's transparency overlay after its opaque and dielectric lighting and
    /// media apply, at the render extent, with the view's [`ParticleLighting`] for this
    /// frame, which [`ParticleDraw`] and a hook's own `particle_shade` light and fog
    /// particles with; the overlay is composited over the radiance before `scene_referred`.
    /// Particle producer entries are drawn here or not at all for primary visibility. While it
    /// is set the view's froxels are lit every frame, whether or not anything else reads them.
    pub transparency: Option<&'a mut TransparencyHook<'a>>,
    /// On the view's pre-exposed radiance before metering, at the output extent; compute
    /// work in [`Frame::output_list`].
    pub scene_referred: Option<&'a mut dyn FnMut(&mut HookPass<'_, '_>)>,
    /// On tonemap's display-linear image before compose; not run for `ViewTarget::Radiance`.
    /// Compute work in [`Frame::output_list`].
    pub display_linear: Option<&'a mut dyn FnMut(&mut HookPass<'_, '_>)>,
    /// Into the display's UI layer, in [`Frame::list`]; run only for `ViewTarget::Display`.
    /// With `Display::set_ui_per_present` on and frame generation generating, it runs once
    /// per presented image. Each run records inside a pass named [`orr_output::UI_PASS`], so
    /// it begins no pass of its own.
    pub ui: Option<&'a mut dyn FnMut(&mut UiPass<'_, '_>)>,
}

/// Writes one scene's changes: everything [`orr_scene::SceneWriter`] writes, which it derefs to,
/// and what reaches the scene's lighting. Send.
pub struct SceneWriter<'f> {
    writer: orr_scene::SceneWriter<'f>,
    lighting: &'f SceneLighting,
}

impl<'f> Deref for SceneWriter<'f> {
    type Target = orr_scene::SceneWriter<'f>;

    fn deref(&self) -> &Self::Target {
        &self.writer
    }
}

impl DerefMut for SceneWriter<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.writer
    }
}

impl SceneWriter<'_> {
    /// The world changed at once, as in a time skip: the scene's next lighting pass clears its
    /// world cache and light feedback, and every view of the scene renders as a camera cut the
    /// next time [`Frame::render_view`] renders it, this frame or later. A view
    /// [`Frame::raster_view`] renders sets its own `camera_cut`. A camera that moves:
    /// `camera_cut` on the view; the world that changes: this on the scene; either cuts the view.
    pub fn reset_lighting_history(&self) {
        self.lighting.reset_history();
    }
}

/// One camera that [`Frame::render_view`] renders this frame.
#[derive(Clone, Copy)]
pub struct LitView<'a> {
    /// Must be listed in the frame.
    pub scene: SceneId,
    pub history: ViewHistoryId,
    pub position: DVec3,
    /// World from view.
    pub rotation: Quat,
    pub projection: Projection,
    pub lens: Lens,
    /// A camera cut: the previous camera is taken to be this one; motion blur shows nothing,
    /// frame generation, ray reconstruction and lighting drop their history, and exposure and
    /// white balance start over: the frame is lit as [`Exposure::start`] gives and shown at the
    /// exposure and white point it meters. Ray reconstruction drops its history again on the
    /// frame first lit at the exposure the view sets. Under auto exposure that is the frame
    /// after the cut's when the cut's frame has light to meter. Otherwise the view stays lit
    /// at the reference through its first frame with light, up to and including the first
    /// frame whose exposure pass has read back that the exposure is set, and the frame after
    /// that one drops the history. The frame after the cut's drops it too, unless the cut
    /// frame's telemetry is read back by then and shows no exposure set. Under manual exposure
    /// the view starts at its exposure and only the cut's frame drops the history. A view's
    /// first render, its first on another scene and its first after
    /// [`SceneWriter::reset_lighting_history`] are cuts without it.
    pub camera_cut: bool,
    /// Menus and loading screens: frame generation is told the frame shows no scene.
    pub not_rendering_scene: bool,
    /// A capture of this render, read back through [`Renderer::screenshots`], which holds its
    /// memory until then: set it only on the renders to capture. A display capture needs an
    /// image, so a [`ViewTarget::Radiance`] view takes none, and it holds the UI only on
    /// [`ViewTarget::Display`]. While the path tracing kernels compile it captures black.
    pub screenshot: Option<ScreenshotRequest>,
    /// How the view filters material textures.
    pub texture_filter: TextureFilter,
    pub target: ViewTarget<'a>,
}

/// Where a [`LitView`]'s image goes, which sets its output extent.
#[derive(Clone, Copy)]
pub enum ViewTarget<'a> {
    /// The frame's display, at its window's extent. [`Frame::render_view`] returns
    /// [`Error::NoDisplay`] without a display or while its window has no area.
    Display,
    /// A texture with storage usage (RGBA8 or RGBA16F), overwritten SDR-encoded with paper
    /// white at 250 nits.
    Texture(&'a orr_gpu::Texture),
    /// No image: the view stops after exposure, with [`RenderedView::radiance`] at `extent`.
    Radiance { extent: [u32; 2] },
}

/// What a view's `render_view` wrote. `radiance` and `linear` stay valid until the frame's
/// GPU work completes; of `raster` only depth, motion and the HZB do (see [`RasterView`]).
/// `linear` and the view's target are written on [`Frame::output_list`]: passes that read
/// them this frame record there.
#[derive(Clone)]
pub struct RenderedView {
    /// The path tracing kernels are still compiling: the view rendered its raster half, and
    /// `radiance`, `linear` and its target hold black.
    pub compiling: bool,
    /// The view's paths ran pipelines that hold no more than its scene and settings ask for,
    /// which is everything without [`ViewLighting::specialize`]; false while pipelines that
    /// hold more run them until those have compiled, and while `compiling`.
    pub exact_paths: bool,
    pub raster: RasterView,
    /// The extent the view rendered at.
    pub render_extent: [u32; 2],
    /// What ray reconstruction ran as; `None` when it didn't run.
    pub reconstruction: Option<ReconstructionMode>,
    /// RGBA16F linear Rec.2020 pre-exposed radiance at the output extent, as ray
    /// reconstruction, or lighting without it, `hooks.scene_referred` and motion blur left it.
    /// With a `display_linear` hook it holds that hook's source instead.
    pub radiance: TextureIndex,
    /// RGBA16F display-linear Rec.709, 1.0 = paper white, at the output extent; `NONE`
    /// when the view stopped after exposure.
    pub linear: TextureIndex,
}

/// The renderer: scenes, their lighting, their views' histories and the frame.
pub struct Renderer {
    scenes: Scenes,
    lighting: Lighting,
    output: Output,
}

/// How long the path kernels may wait for a thread before a warning says so.
const UNSTARTED_PATHS_WARNING: Duration = Duration::from_secs(10);

/// Where the path tracing kernels stand.
enum Paths {
    /// No view history has asked for them.
    NotStarted,
    Compiling {
        jobs: CompilingKernels,
        since: Instant,
        warned: bool,
    },
    Ready(Box<LightingKernels>),
    Failed(Arc<LightingError>),
}

impl Paths {
    /// Takes the kernels when their last job has finished, and warns once when nothing runs the
    /// jobs.
    fn poll(&mut self) {
        let Self::Compiling {
            jobs,
            since,
            warned,
        } = self
        else {
            return;
        };
        if jobs.outstanding() > 0 {
            if !*warned && !jobs.started() && since.elapsed() > UNSTARTED_PATHS_WARNING {
                log::warn!(
                    "no thread has started the path tracing kernels' compile jobs, so views stay \
                     black: lend threads to Renderer::loading() with run_worker or run_one, or \
                     call Renderer::wait_paths"
                );
                *warned = true;
            }
            return;
        }
        *self = match jobs.finish() {
            Ok(kernels) => {
                log::info!(
                    "path tracing kernels ready after {:.1} s",
                    since.elapsed().as_secs_f64()
                );
                Self::Ready(Box::new(kernels))
            }
            Err(e) => {
                log::error!("{e}");
                Self::Failed(Arc::new(e))
            }
        };
    }
}

/// orr_lighting's kernels and tables, and each scene's lighting state.
struct Lighting {
    kernels: &'static KernelSet,
    entry: PathEntry,
    try_cached_paths: bool,
    /// Started by the first view history.
    paths: Paths,
    /// [`Renderer::wait_paths`] was called: a view waits for the path pipelines its scene asks
    /// for instead of rendering without them.
    waits: bool,
    /// When the kernels started, until a view first renders lit.
    started: Option<Instant>,
    scenes: HashMap<SceneId, SceneLighting>,
    tables: BsdfTables,
}

impl Lighting {
    /// Starts the lighting kernels unless they are ready or compiling: from the pipeline caches
    /// when they hold every one and `try_cached_paths` allows looking, else as jobs on `loading`,
    /// which report any failure. The path pipelines themselves start as jobs when a view first
    /// asks for them.
    fn start_paths(&mut self, gpu: &Device, loading: &Arc<Loading>) {
        if matches!(self.paths, Paths::Ready(_) | Paths::Compiling { .. }) {
            return;
        }
        self.started = Some(Instant::now());
        if self.try_cached_paths {
            let cached = gpu.cached_pipelines_only();
            if let Ok(kernels) =
                LightingKernels::new(&cached, self.kernels, &self.tables, self.entry)
            {
                kernels.attach(gpu, loading);
                self.paths = Paths::Ready(Box::new(kernels));
                return;
            }
        }
        log::info!(
            "compiling the path tracing kernels on loading jobs; views render black until they \
             finish, which on a first run takes minutes"
        );
        self.paths = Paths::Compiling {
            jobs: LightingKernels::spawn(gpu, self.kernels, &self.tables, self.entry, loading),
            since: Instant::now(),
            warned: false,
        };
    }
}

/// One view history's lighting and output state.
struct ViewState {
    lighting: Option<LightingView>,
    output: OutputView,
    /// Compose roots per frame in flight.
    compose: Vec<ComposeRoot>,
    /// Renders of the view so far: the sample index of its next paths.
    samples: u32,
    /// The scene of the view's last lit render and that scene's lighting resets then.
    scene_resets: Option<(SceneId, u64)>,
    reconstruction: Option<ViewReconstruction>,
    /// The lighting side of the view's pixel probes, and what each probe's render was.
    origin_probes: OriginProbes,
    /// The view's traced-scene debug pass, while its lighting asks for one.
    traced: TracedViews,
    probes: Vec<ProbeAt>,
}

/// A view's ray reconstruction feature and what it was made for.
struct ViewReconstruction {
    feature: Box<dyn ReconstructionFeature>,
    request: Reconstruction,
    /// No render has used it yet, so it has no history.
    fresh: bool,
}

impl ViewState {
    /// The description of the view's next raster half at these extents.
    fn raster_desc(&self, view: &LitView<'_>, extent: [u32; 2], render: [u32; 2]) -> ViewDesc {
        let (jitter, mip_bias) = self.reconstruction.as_ref().map_or(([0.0; 2], 0.0), |r| {
            (
                r.feature.jitter(u64::from(self.samples)),
                r.feature.mip_bias(),
            )
        });
        ViewDesc {
            scene: view.scene,
            history: view.history,
            position: view.position,
            rotation: view.rotation,
            projection: view.projection,
            lens: view.lens,
            render_extent: UVec2::from(render),
            output_extent: UVec2::from(extent),
            jitter: Vec2::from(jitter),
            camera_cut: view.camera_cut,
            mip_bias,
            texture_filter: view.texture_filter,
        }
    }

    /// Makes the view's feature match `request` at the output extent `output`, with no feature
    /// without `backend` or for [`Reconstruction::Off`].
    fn sync_reconstruction(
        &mut self,
        backend: Option<&dyn ReconstructionBackend>,
        output: [u32; 2],
        request: Reconstruction,
    ) -> Result<()> {
        let current = self
            .reconstruction
            .as_ref()
            .is_some_and(|r| r.request == request && r.feature.output_size() == output);
        if current && backend.is_some() {
            return Ok(());
        }
        self.reconstruction = None;
        let Some(backend) = backend else {
            return Ok(());
        };
        let feature = match request {
            Reconstruction::Off => return Ok(()),
            Reconstruction::Dlaa => backend.dlaa(output),
            Reconstruction::Upscale(quality) => backend.feature(output, quality),
            Reconstruction::RenderExtent(render) => {
                let wide = u64::from(render[0]) * u64::from(output[1]);
                let tall = u64::from(render[1]) * u64::from(output[0]);
                // One render pixel of rounding on either axis.
                if wide.abs_diff(tall) > u64::from(output[0].max(output[1])) {
                    return Err(Error::RenderAspect { render, output });
                }
                backend.feature_at(render, output)
            }
        };
        self.reconstruction = Some(ViewReconstruction {
            feature: feature?,
            request,
            fresh: true,
        });
        Ok(())
    }
}

/// orr_output's kernels, the views' output state and the state of frames that render no
/// view.
struct Output {
    device: Device,
    frames_in_flight: u32,
    kernels: OutputKernels,
    /// Tonemap roots per frame in flight.
    roots: OutputView,
    /// Compose roots per frame in flight.
    compose: Vec<ComposeRoot>,
    /// The display-linear image [`Frame::pattern`] draws into, at the display's extent.
    linear: Option<Image>,
    /// What makes ray reconstruction features, while it offers them, and what it offers.
    reconstruction: Option<(Box<dyn ReconstructionBackend>, ReconstructionSupport)>,
    views: HashMap<ViewHistoryId, ViewState>,
}

fn compose_roots(gpu: &Device, frames_in_flight: u32) -> Result<Vec<ComposeRoot>> {
    Ok((0..frames_in_flight.max(1))
        .map(|_| ComposeRoot::new(gpu))
        .collect::<Result<_, _>>()?)
}

/// `image`, recreated at `extent` (with its discard recorded into `list`) when it holds
/// none or one of another extent.
fn image_at<'i>(
    image: &'i mut Option<Image>,
    gpu: &Device,
    list: &mut CommandList<'_>,
    extent: [u32; 2],
    name: &str,
) -> Result<&'i Image> {
    if image
        .as_ref()
        .is_none_or(|i| i.texture().desc().extent[..2] != extent)
    {
        let fresh = Image::new(gpu, Format::Rgba16Float, extent, name)?;
        list.discard(fresh.texture());
        *image = Some(fresh);
    }
    Ok(image.as_ref().expect("image"))
}

impl Output {
    fn new(gpu: &Device, frames_in_flight: u32) -> Result<Self> {
        Ok(Self {
            device: gpu.clone(),
            frames_in_flight,
            kernels: OutputKernels::new(gpu)?,
            roots: OutputView::new(gpu, frames_in_flight)?,
            compose: compose_roots(gpu, frames_in_flight)?,
            linear: None,
            reconstruction: None,
            views: HashMap::new(),
        })
    }

    /// The display-linear image at `extent`, recreated (with its discard recorded into
    /// `list`) when the extent changes.
    fn linear(&mut self, list: &mut CommandList<'_>, extent: [u32; 2]) -> Result<&Image> {
        image_at(
            &mut self.linear,
            &self.device,
            list,
            extent,
            "display linear",
        )
    }

    /// Records compose of the display-linear `source` into `frame`'s backbuffer.
    fn compose(
        &mut self,
        list: &mut CommandList<'_>,
        id: u64,
        frame: &mut DisplayFrame<'_>,
        source: TextureIndex,
    ) {
        let target = frame.backbuffer(list);
        let n = self.compose.len() as u64;
        self.kernels.compose(
            list,
            &mut self.compose[(id % n) as usize],
            &ComposePass {
                source: ComposeSource::Linear(source),
                ui: None,
                target,
                extent: frame.extent,
                display: &frame.display,
            },
        );
    }
}

/// The arena name of a view's display-linear image.
const VIEW_LINEAR: &str = "view linear";
/// The arena name of ray reconstruction's output.
const RECONSTRUCTED: &str = "reconstructed radiance";

/// A view image at the output `extent`, from step `first` to the end of the frame: ray
/// reconstruction's output, which every later pass reads in place of lighting's radiance, or
/// the display-linear image, from the scene-referred hook on the second image of the chain that
/// hooks, motion blur and tonemap alternate over. The consumer may read either after the
/// render.
fn output_image_request(name: &'static str, extent: [u32; 2], first: ViewStep) -> ArenaRequest {
    ArenaRequest::texture(
        name,
        TextureDesc::d2(
            Format::Rgba16Float,
            extent[0],
            extent[1],
            TextureUsage::SAMPLED | TextureUsage::STORAGE,
        ),
        Span::new(first, ViewStep::After),
    )
}

/// Every kernel module compiled into orr.
fn builtin_modules() -> Vec<&'static ShaderModule> {
    orr_scene::MODULES
        .iter()
        .chain(orr_lighting::MODULES)
        .copied()
        .collect()
}

impl Renderer {
    /// Checks `desc.kernels` against orr's compiled-in kernels, then compiles the renderer's
    /// pipelines and uploads the BSDF tables on the calling thread, which must own the
    /// Render queue. The path tracing pipelines start with the first view history
    /// ([`Renderer::create_view_history`]).
    pub fn new(gpu: &Device, desc: RendererDesc) -> Result<Self> {
        desc.kernels
            .check(&builtin_modules())
            .map_err(SceneError::from)?;
        let tables = BsdfTables::new(gpu)?;
        Ok(Self {
            scenes: Scenes::new(gpu, &desc)?,
            lighting: Lighting {
                kernels: desc.kernels,
                entry: desc.path_entry,
                try_cached_paths: desc.try_cached_paths,
                paths: Paths::NotStarted,
                waits: false,
                started: None,
                scenes: HashMap::new(),
                tables,
            },
            output: Output::new(gpu, desc.frames_in_flight)?,
        })
    }

    pub fn device(&self) -> &Device {
        self.scenes.device()
    }

    /// Gives lit views ray reconstruction, when `backend` offers it on the device; returns
    /// whether it does. `backend` must be on this renderer's device. Views drop their features
    /// and history, and make new features on `backend` at their next render.
    pub fn set_reconstruction(&mut self, backend: impl ReconstructionBackend + 'static) -> bool {
        for view in self.output.views.values_mut() {
            view.reconstruction = None;
        }
        let support = backend.support();
        self.output.reconstruction = support
            .available
            .then(|| (Box::new(backend) as Box<dyn ReconstructionBackend>, support));
        self.output.reconstruction.is_some()
    }

    /// What the renderer's ray reconstruction backend offers; `None` without one.
    pub fn reconstruction_support(&self) -> Option<ReconstructionSupport> {
        self.output.reconstruction.as_ref().map(|(_, s)| *s)
    }

    /// A scene's tables and the lighting state every view of the scene shares.
    pub fn create_scene(
        &mut self,
        desc: &SceneDesc,
        lighting: &SceneLightingSettings,
    ) -> Result<SceneId> {
        let gpu = self.scenes.device().clone();
        let state = SceneLighting::new(&gpu, self.lighting.kernels, lighting)?;
        let id = self.scenes.create_scene(desc)?;
        self.scenes.set_near_range(id, lighting.near_range)?;
        self.lighting.scenes.insert(id, state);
        Ok(id)
    }

    pub fn set_scene_lighting(
        &mut self,
        scene: SceneId,
        settings: &SceneLightingSettings,
    ) -> Result<()> {
        let state = self
            .lighting
            .scenes
            .get_mut(&scene)
            .ok_or(SceneError::UnknownScene(scene))?;
        state.set_settings(settings)?;
        self.scenes.set_near_range(scene, settings.near_range)?;
        Ok(())
    }

    /// Destroys a scene; its memory is freed once every submitted frame has completed.
    pub fn destroy_scene(&mut self, scene: SceneId) {
        self.lighting.scenes.remove(&scene);
        self.scenes.destroy_scene(scene);
    }

    /// The scene's lighting statistics of `frame`, a frame whose work has completed
    /// ([`FrameStats::frame`] or older); `None` once they are no longer kept.
    pub fn lighting_stats(&self, scene: SceneId, frame: u64) -> Option<LightingStats> {
        self.lighting.scenes.get(&scene)?.stats(frame)
    }

    /// History for one persistent camera: its previous camera, exposure slots and the
    /// lighting and output state of its renders.
    ///
    /// The first call, and the first after a failed compile, starts the lighting kernels
    /// without compiling on this thread. If the pipeline caches hold those every view runs
    /// ([`RendererDesc::try_cached_paths`]), they are ready when it returns. Otherwise they
    /// compile as jobs on [`Renderer::loading`], one per pipeline, which take up to a minute
    /// on a first run, holding their threads the whole time. The path pipelines are
    /// compiled for what a view's scene and settings may meet, as jobs of their own when the
    /// view first renders. Until both are ready [`Frame::render_view`] renders black.
    /// The jobs need threads: lend them with [`Loading::run_worker`] or
    /// [`Loading::run_one`], as many as the machine has to spare so that they compile in
    /// parallel while loading goes on, or block in [`Renderer::wait_paths`]. With neither, views stay
    /// black, and a warning is logged after a few seconds of frames.
    ///
    /// Each job writes the device's pipeline cache file when it finishes. Dropping the renderer
    /// cancels the jobs no thread has started. A thread already compiling finishes that pipeline,
    /// which a consumer joining its workers waits for, since the driver cannot be interrupted. On
    /// Windows, a process that is ending need not join them: the system stops its threads before
    /// the driver unloads.
    pub fn create_view_history(&mut self) -> Result<ViewHistoryId> {
        let gpu = self.scenes.device().clone();
        let loading = self.scenes.loading();
        self.lighting.start_paths(&gpu, &loading);
        let fif = self.output.frames_in_flight;
        let state = ViewState {
            lighting: None,
            output: OutputView::new(&gpu, fif)?,
            compose: compose_roots(&gpu, fif)?,
            samples: 0,
            scene_resets: None,
            reconstruction: None,
            origin_probes: OriginProbes::default(),
            traced: TracedViews::default(),
            probes: Vec::new(),
        };
        let id = self.scenes.create_view_history()?;
        self.output.views.insert(id, state);
        Ok(id)
    }

    /// Whether the lighting kernels every view runs have compiled. False also before the first
    /// view history and after a failed compile. A view's own path pipelines compile when it
    /// first renders: [`Frame::view_lit`] says whether a view renders lit.
    pub fn kernels_ready(&mut self) -> bool {
        self.lighting.paths.poll();
        matches!(self.lighting.paths, Paths::Ready(_))
    }

    /// How many of the path tracing kernels' compile jobs have not finished.
    pub fn paths_outstanding(&self) -> u32 {
        match &self.lighting.paths {
            Paths::Compiling { jobs, .. } => jobs.outstanding() + jobs.paths_outstanding(),
            Paths::Ready(kernels) => kernels.outstanding(),
            _ => 0,
        }
    }

    /// Blocks until the path tracing kernels have compiled, running queued loading jobs on this
    /// thread so it needs no workers, and returns [`Error::PathKernels`] if they failed. For
    /// tests and headless runs; returns at once when they are ready or never started. From
    /// this call on [`Frame::render_view`] also waits for the path pipelines a view's scene and
    /// settings first ask for, where it would otherwise hold the view black while they compile.
    pub fn wait_paths(&mut self) -> Result<()> {
        self.lighting.waits = true;
        if let Paths::Compiling { jobs, .. } = &self.lighting.paths {
            jobs.wait(&self.scenes.loading());
        }
        self.lighting.paths.poll();
        match &self.lighting.paths {
            Paths::Failed(e) => Err(Error::PathKernels(Arc::clone(e))),
            _ => Ok(()),
        }
    }

    /// History for a camera that only [`Frame::raster_view`] renders: its previous camera
    /// and exposure slots, without lighting or output state, so it compiles no path tracing
    /// pipelines. [`Frame::render_view`] refuses it.
    pub fn create_raster_view_history(&mut self) -> Result<ViewHistoryId> {
        Ok(self.scenes.create_view_history()?)
    }

    /// The transient targets of the view's last render and the memory they take
    /// ([`RendererDesc::alias_view_targets`]).
    pub fn view_arena_report(&self, history: ViewHistoryId) -> Option<ArenaReport> {
        self.scenes.view_arena_report(history)
    }

    pub fn destroy_view_history(&mut self, id: ViewHistoryId) {
        self.output.views.remove(&id);
        self.scenes.destroy_view_history(id);
    }

    /// Takes the view's screenshots of frames at or before `completed`, a frame whose work
    /// has completed ([`FrameStats::frame`]), oldest first.
    pub fn screenshots(&mut self, history: ViewHistoryId, completed: u64) -> Vec<Screenshot> {
        self.output
            .views
            .get_mut(&history)
            .map_or_else(Vec::new, |v| v.output.screenshots(completed))
    }

    /// Takes the view's pixel probes ([`Frame::probe`]) of frames at or before `completed`, a
    /// frame whose work has completed ([`FrameStats::frame`]), oldest first. The names and
    /// residency states in them are what the renderer holds when they are taken.
    pub fn probes(&mut self, history: ViewHistoryId, completed: u64) -> Vec<PixelProbe> {
        let scene = self.scenes.probes(history, completed);
        let (mut lighting, done) = match self.output.views.get_mut(&history) {
            Some(state) => {
                let (done, pending): (Vec<ProbeAt>, _) = std::mem::take(&mut state.probes)
                    .into_iter()
                    .partition(|p| p.frame <= completed);
                state.probes = pending;
                (state.origin_probes.take(completed), done)
            }
            None => (Vec::new(), Vec::new()),
        };
        scene
            .into_iter()
            .map(|s| {
                let at = done.iter().find(|p| p.frame == s.frame);
                let l = lighting.iter().position(|l| l.frame == s.frame);
                PixelProbe {
                    // A view only rasterized was asked for a render pixel.
                    output_pixel: at.map_or(s.pixel, |p| p.output),
                    output_extent: at.map_or(s.view.render_extent, |p| p.output_extent),
                    reconstruction: at.and_then(|p| p.reconstruction),
                    lighting: l.map(|i| lighting.swap_remove(i)),
                    scene: s,
                }
            })
            .collect()
    }

    /// The loading side's job queue, for the threads the consumer lends to it.
    pub fn loading(&self) -> Arc<Loading> {
        self.scenes.loading()
    }

    /// Mounts a pack; returns at once and loads on the loading side.
    pub fn mount(&mut self, source: PackSource) -> MountId {
        self.scenes.mount(source)
    }

    pub fn mount_state(&self, m: MountId) -> LoadState {
        self.scenes.mount_state(m)
    }

    /// Unmounts a pack; its memory is freed once every submitted frame has completed.
    pub fn unmount(&mut self, m: MountId) {
        self.scenes.unmount(m);
    }

    pub fn model(&self, m: MountId, id: orr_pack::AssetId) -> Result<ModelId> {
        Ok(self.scenes.model(m, id)?)
    }

    /// The skeleton `id` of mount `m`, for pose kernels that fill skinned palettes.
    pub fn skeleton(&self, m: MountId, id: orr_pack::AssetId) -> Result<SkeletonRef> {
        Ok(self.scenes.skeleton(m, id)?)
    }

    /// The clip `id` of mount `m`, for pose kernels that sample it.
    pub fn clip(&self, m: MountId, id: orr_pack::AssetId) -> Result<ClipRef> {
        Ok(self.scenes.clip(m, id)?)
    }

    pub fn instance_page(&self, m: MountId, id: orr_pack::AssetId) -> Result<PageId> {
        Ok(self.scenes.instance_page(m, id)?)
    }

    pub fn page(&self, id: PageId) -> Option<&InstancePage> {
        self.scenes.page(id)
    }

    /// The entry of `scene`'s producer table that producer `id` holds while it is set, as
    /// [`orr_scene::SceneWriter::producer_entry`] gives it.
    pub fn producer_entry(&self, scene: SceneId, id: ProducerId) -> Result<Option<u32>> {
        Ok(self.scenes.producer_entry(scene, id)?)
    }

    /// Wants the records of instance page `id` resident from the next frame on. Requests last
    /// one frame.
    pub fn prefetch_page(&mut self, id: PageId, cell_distance: u32) {
        self.scenes.prefetch_page(id, cell_distance);
    }

    /// Wants the instance pages meeting the sphere, and the streamed units of the models they
    /// place, resident from the next frame on. Requests last one frame.
    pub fn prefetch_region(&mut self, center: DVec3, radius: f32) {
        self.scenes.prefetch_region(center, radius);
    }

    /// Starts an edit of a mesh: a new model whose touched pages kernels write at the scene
    /// compute point. Poll [`Renderer::edit_state`].
    pub fn create_edit(&mut self, desc: &EditDesc<'_>) -> Result<EditId> {
        Ok(self.scenes.create_edit(desc)?)
    }

    pub fn edit_state(&self, id: EditId) -> LoadState {
        self.scenes.edit_state(id)
    }

    /// What kernels and instances need of a ready edit.
    pub fn edit(&self, id: EditId) -> Result<EditRef> {
        Ok(self.scenes.edit(id)?)
    }

    /// Ends an edit once every frame so far has completed; remove its copies first.
    pub fn destroy_edit(&mut self, id: EditId) {
        self.scenes.destroy_edit(id);
    }

    pub fn texture(&self, m: MountId, id: orr_pack::AssetId) -> Result<TextureIndex> {
        Ok(self.scenes.texture(m, id)?)
    }

    pub fn material(&self, m: MountId, id: orr_pack::AssetId) -> Result<MaterialId> {
        Ok(self.scenes.material(m, id)?)
    }

    /// Wants every streamed unit of `model` resident from the next frame on, `cell_distance`
    /// world cells from the focus. Requests last one frame: call it every frame while the
    /// model should load or stay.
    pub fn prefetch(&mut self, model: ModelId, cell_distance: u32) {
        self.scenes.prefetch(model, cell_distance);
    }

    /// Gives every copy of `model` up to 3 farther models, each with the distance from the
    /// focus in m at which copies switch to it, ascending; empty: the model at every
    /// distance. A copy that changes step cross-fades between the two over a few frames.
    pub fn set_chain(&mut self, model: ModelId, chain: &[(ModelId, f32)]) -> Result<()> {
        Ok(self.scenes.set_chain(model, chain)?)
    }

    /// Wants every tile of mips `finest..` of the cooked texture `texture` of mount `m`
    /// resident from the next frame on. Requests last one frame.
    pub fn prefetch_texture(&mut self, m: MountId, texture: TextureIndex, finest: u32) {
        self.scenes.prefetch_texture(m, texture, finest);
    }

    /// The stream table for kernels that sample looked-up textures through
    /// `texture_footprint`.
    pub fn texture_streams(&self) -> GpuPtr<GpuPtr<TextureStreamGpu>> {
        self.scenes.texture_streams()
    }

    /// Streams blob `blob` of asset `id` of mount `m` into consumer memory under the user
    /// blob budget; poll [`Renderer::blob_state`].
    pub fn request_blob(
        &mut self,
        m: MountId,
        id: orr_pack::AssetId,
        blob: u32,
        target: BlobTarget,
    ) -> Result<BlobId> {
        Ok(self.scenes.request_blob(m, id, blob, target)?)
    }

    pub fn blob_state(&self, id: BlobId) -> LoadState {
        self.scenes.blob_state(id)
    }

    /// Returns a requested blob's bytes to the user blob budget.
    pub fn release_blob(&mut self, id: BlobId) {
        self.scenes.release_blob(id);
    }

    /// Residency figures of the latest frame's step 1.
    pub fn residency(&self) -> &ResidencyStats {
        self.scenes.residency()
    }

    /// Lowers the budget of `class`'s streamed units to `bytes`, at most what it is: the
    /// next frame evicts down to it.
    pub fn lower_budget(&mut self, class: ResidencyClass, bytes: u64) {
        self.scenes.lower_budget(class, bytes);
    }

    /// What the streamed pools, the CLAS pool and the frames' upload windows hold, gathered
    /// on demand.
    pub fn scene_memory(&self) -> SceneMemory {
        self.scenes.memory()
    }

    /// A compute pipeline of `module`'s `entry`, created on the calling thread. With the
    /// `hot-reload` feature, editing the kernel's sources replaces it at a later step 1
    /// ([`Frame::reloads`]); look it up each frame with [`Frame::pipeline`] or
    /// [`SceneCompute::pipeline`].
    pub fn compute_pipeline(
        &mut self,
        module: &'static ShaderModule,
        entry: &str,
        spec: orr_gpu::SpecData<'_>,
    ) -> Result<PipelineId> {
        Ok(self.scenes.compute_pipeline(module, entry, spec)?)
    }

    /// A mesh pipeline of `module`'s `stages`, created on the calling thread. Hot reload
    /// replaces it with the module's other pipelines, as it does a compute one; look it up
    /// each frame with [`Frame::mesh_pipeline`].
    pub fn mesh_pipeline(
        &mut self,
        module: &'static ShaderModule,
        stages: &MeshStages<'_>,
    ) -> Result<MeshPipelineId> {
        Ok(self.scenes.mesh_pipeline(module, stages)?)
    }

    /// Step 1: waits until the frame `frames_in_flight` back has completed, sleeps on the
    /// display's low latency handle, opens the next frame and the display's slot for it.
    pub fn begin_frame<'a>(&'a mut self, desc: FrameDesc<'a>) -> Result<Frame<'a>> {
        self.lighting.paths.poll();
        let scenes = self.scenes.begin_frame(&orr_scene::FrameDesc {
            time: desc.time,
            scenes: desc.scenes,
            low_latency: desc.display.as_deref().and_then(Display::latency),
        })?;
        let mut display = desc.display;
        if let Some(d) = display.as_deref_mut() {
            d.begin_frame(scenes.id())?;
        }
        Ok(Frame {
            scenes,
            display,
            lighting: &mut self.lighting,
            output: &mut self.output,
            lit: Vec::new(),
            slots: Vec::new(),
            probes: Vec::new(),
        })
    }

    /// Figures of the last frame whose GPU work has completed.
    pub fn stats(&self) -> &FrameStats {
        self.scenes.stats()
    }
}

/// One frame, from [`Renderer::begin_frame`] to [`Frame::end`].
pub struct Frame<'a> {
    scenes: orr_scene::Frame<'a>,
    display: Option<&'a mut Display>,
    lighting: &'a mut Lighting,
    output: &'a mut Output,
    /// Scenes whose lighting pass this frame has recorded.
    lit: Vec<SceneId>,
    /// Heap slots of texture targets' storage views, freed once the frame completes.
    slots: Vec<u32>,
    /// Pixels to probe in the views' renders this frame, in output pixels.
    probes: Vec<(ViewHistoryId, [u32; 2])>,
}

impl<'a> Frame<'a> {
    /// What `scene` may hold now, which the path pipelines of its views are made for.
    pub fn scene_features(&self, scene: SceneId) -> Result<orr_scene::SceneFeatures> {
        Ok(self.scenes.scene_features(scene)?)
    }

    /// The entry of `scene`'s producer table that producer `id` holds while it is set, as
    /// [`orr_scene::SceneWriter::producer_entry`] gives it.
    pub fn producer_entry(&self, scene: SceneId, id: ProducerId) -> Result<Option<u32>> {
        Ok(self.scenes.producer_entry(scene, id)?)
    }

    /// orr's frame id: strictly increasing from 1.
    pub fn id(&self) -> u64 {
        self.scenes.id()
    }

    /// The exposure telemetry of the view's newest completed render. Read it before the
    /// view renders this frame: the frame's render reuses the oldest render's readback.
    pub fn exposure_telemetry(&self, history: ViewHistoryId) -> Option<ExposureTelemetry> {
        self.output
            .views
            .get(&history)?
            .output
            .telemetry(self.scenes.completed())
    }

    /// What the frame's display encodes for, and its extent; `None` without a display or
    /// while its window has no area.
    pub fn display(&mut self) -> Option<(ResolvedDisplay, [u32; 2])> {
        let f = self.display.as_deref_mut()?.frame()?;
        Some((f.display, f.extent))
    }

    /// The current pipeline of `id`.
    pub fn pipeline(&self, id: PipelineId) -> &'a orr_gpu::ComputePipeline {
        self.scenes.pipeline(id)
    }

    /// The current mesh pipeline of `id`.
    pub fn mesh_pipeline(&self, id: MeshPipelineId) -> &'a orr_gpu::MeshPipeline {
        self.scenes.mesh_pipeline(id)
    }

    /// What hot reload applied at this frame's step 1.
    pub fn reloads(&self) -> &[ReloadReport] {
        self.scenes.reloads()
    }

    /// Step 2: the writer of a scene listed in this frame. One per scene at a time; Send.
    pub fn scene(&self, id: SceneId) -> Result<SceneWriter<'_>> {
        Ok(SceneWriter {
            writer: self.scenes.scene(id)?,
            lighting: self
                .lighting
                .scenes
                .get(&id)
                .ok_or(SceneError::UnknownScene(id))?,
        })
    }

    /// Step 2: tells orr that kernels change edit `id` at this frame's scene compute point.
    /// False: this frame cannot rebuild it, and the kernels must leave it unchanged.
    pub fn edit_dirty(&mut self, id: EditId) -> bool {
        self.scenes.edit_dirty(id)
    }

    /// Like [`edit_dirty`](Self::edit_dirty), but what kernels change has no motion this frame.
    pub fn edit_dirty_without_motion(&mut self, id: EditId) -> bool {
        self.scenes.edit_dirty_without_motion(id)
    }

    /// Step 2: a material writer; Send.
    pub fn materials(&self) -> MaterialWriter<'_> {
        self.scenes.materials()
    }

    /// Step 2, on the render thread: sets a latency marker under [`Frame::id`] when the frame
    /// has a low latency handle.
    pub fn latency_marker(&self, marker: ConsumerMarker) {
        self.scenes.latency_marker(marker);
    }

    /// Step 3: applies every write, then hands each listed scene in order to `compute`. Panics
    /// if a listed scene never set the data of an environment or medium its kernels link, as
    /// does any step that runs this one.
    pub fn update_scenes(
        &mut self,
        compute: &mut dyn FnMut(&mut SceneCompute<'_, 'a>),
    ) -> Result<()> {
        Ok(self.scenes.update_scenes(compute)?)
    }

    /// Step 4 alone, for a view the consumer shades itself: returns the view's raster
    /// outputs, the G-buffer that kernels read through `load_gsurface` and the scene root
    /// whose TLAS they trace, for passes it records after it into [`Frame::list`].
    pub fn raster_view(&mut self, view: &ViewDesc) -> Result<RasterView> {
        self.raster_view_with(view, None)
    }

    /// [`Frame::raster_view`], with `opaque` drawing the scene's drawn producer entries
    /// (`ProducerDesc::drawn`) in each pass of the view's raster, as [`ViewHooks::opaque`] does.
    pub fn raster_view_with(
        &mut self,
        view: &ViewDesc,
        opaque: Option<&mut OpaqueHook<'_>>,
    ) -> Result<RasterView> {
        self.prepare_raster_view(view)?;
        let raster = self.scenes.raster_view_with(view, &[], opaque)?;
        if let Some(at) = take_probe(&mut self.probes, view.history) {
            self.scenes.probe(&raster, at)?;
        }
        Ok(raster)
    }

    /// Probes pixel `pixel` of the output image of view `history`'s render this frame, a render
    /// pixel for a view [`Frame::raster_view`] renders: what was drawn there, at which level of
    /// detail and why, the G-buffer and texture taps the lighting read, what ray tracing traces
    /// of the copy and the frame's detail and memory. [`Renderer::probes`] returns it once the
    /// frame has completed. A debug aid: a frame without a probe records and keeps nothing.
    pub fn probe(&mut self, history: ViewHistoryId, pixel: [u32; 2]) {
        self.probes.retain(|(h, _)| *h != history);
        self.probes.push((history, pixel));
    }

    /// Whether [`Frame::render_view`] renders `view` lit this frame under `lighting`, or black
    /// because the kernels or the pipelines its scene and settings ask for are still compiling;
    /// it asks for those pipelines. `transparency` is whether that render gets a
    /// [`ViewHooks::transparency`] hook, whose particles the froxels' light pass lights. The
    /// answer holds for the frame once [`Frame::update_scenes`] has run. A compile that failed
    /// returns [`Error::PathKernels`].
    pub fn view_lit(
        &mut self,
        view: &LitView<'_>,
        lighting: &ViewLighting,
        transparency: bool,
    ) -> Result<bool> {
        if let Paths::Failed(e) = &self.lighting.paths {
            return Err(Error::PathKernels(Arc::clone(e)));
        }
        self.ask_paths(view, lighting, transparency, false)
    }

    /// Asks for the path pipelines `view` takes under `lighting` and says whether it renders
    /// lit: they are ready, or the caller waits for them ([`Renderer::wait_paths`]), which
    /// `wait` does here. While the kernels every view runs compile, the view's pipelines are
    /// asked for all the same, so they compile beside those. A failed compile fails the
    /// kernels.
    fn ask_paths(
        &mut self,
        view: &LitView<'_>,
        lighting: &ViewLighting,
        particle_lighting: bool,
        wait: bool,
    ) -> Result<bool> {
        if !matches!(
            self.lighting.paths,
            Paths::Ready(_) | Paths::Compiling { .. }
        ) {
            return Ok(false);
        }
        let scene = self
            .lighting
            .scenes
            .get(&view.scene)
            .ok_or(SceneError::UnknownScene(view.scene))?;
        let needs = PathNeeds {
            features: self.scenes.scene_features(view.scene)?,
            frosted_glass: self
                .output
                .views
                .get(&view.history)
                .is_some_and(|state| state.reconstruction.is_some()),
            dof_background: false,
            particle_lighting,
        };
        let waits = self.lighting.waits;
        let asked = match &self.lighting.paths {
            Paths::Compiling { jobs, .. } => {
                jobs.prepare_paths(scene, needs, lighting).map(|()| false)
            }
            // The render of a caller that waits asks for them itself, once it has waited.
            Paths::Ready(_) if waits && !wait => return Ok(true),
            Paths::Ready(kernels) => {
                // A caller that waits takes the pipelines made for exactly its view, so its
                // frames do not depend on what else has compiled.
                // Held until the pipelines are asked for below, which marks them as run now:
                // the render records them in this same call.
                let waited = if waits {
                    kernels.wait_paths(scene, needs, lighting).map(Some)
                } else {
                    Ok(None)
                };
                waited
                    .and_then(|_held| kernels.prepare_paths(scene, needs, lighting))
                    .map(|ready| ready || waits)
            }
            Paths::NotStarted | Paths::Failed(_) => unreachable!("returned above"),
        };
        asked.map_err(|e| {
            log::error!("{e}");
            let e = Arc::new(e);
            self.lighting.paths = Paths::Failed(Arc::clone(&e));
            Error::PathKernels(e)
        })
    }

    /// Step 2: works out the render extent [`Frame::render_view`] gives `view` with these
    /// settings, remaking the view's ray reconstruction feature as that call would, and asks the
    /// view's scene to choose geometry for it at this frame's [`Frame::update_scenes`]. A scene
    /// takes the sharpest of its views. Only a view whose render comes after the update needs
    /// this; without it the view counts from the next frame on.
    pub fn prepare_view(
        &mut self,
        view: &LitView<'_>,
        lighting: &ViewLighting,
        output: &ViewOutput,
    ) -> Result<()> {
        self.resolve_view(view, lighting, output).map(|_| ())
    }

    /// [`Frame::prepare_view`], returning the view's output and render extents.
    fn resolve_view(
        &mut self,
        view: &LitView<'_>,
        lighting: &ViewLighting,
        output: &ViewOutput,
    ) -> Result<([u32; 2], [u32; 2])> {
        let extent = match view.target {
            ViewTarget::Display => self.display().ok_or(Error::NoDisplay)?.1,
            ViewTarget::Texture(t) => {
                let [w, h, _] = t.desc().extent;
                [w, h]
            }
            ViewTarget::Radiance { extent } => extent,
        };
        let backend = self
            .output
            .reconstruction
            .as_ref()
            .filter(|(_, support)| {
                lighting.reference.is_none()
                    && lighting.debug_view == LightingDebugView::None
                    && extent[0] >= support.min_output[0]
                    && extent[1] >= support.min_output[1]
            })
            .map(|(backend, _)| &**backend);
        let state = self
            .output
            .views
            .get_mut(&view.history)
            .ok_or(SceneError::UnknownViewHistory(view.history))?;
        state.sync_reconstruction(backend, extent, output.reconstruction)?;
        let render = state
            .reconstruction
            .as_ref()
            .map_or(extent, |r| r.feature.render_size());
        let detail = view.projection.pixels_per_radian(render[1]);
        self.scenes.request_detail(view.scene, detail)?;
        Ok((extent, render))
    }

    /// [`Frame::prepare_view`] for a view [`Frame::raster_view`] renders.
    pub fn prepare_raster_view(&mut self, view: &ViewDesc) -> Result<()> {
        let detail = view.projection.pixels_per_radian(view.render_extent.y);
        Ok(self.scenes.request_detail(view.scene, detail)?)
    }

    /// Steps 4-5 for one view: its raster half; the scene's lighting pass before the scene's
    /// first view this frame; the view's froxel volume; one path per pixel into the view's
    /// radiance, from its first dielectric or presence layer or its surface; media apply over
    /// each pixel's camera segment; the
    /// transparency hook; ray reconstruction; `hooks.scene_referred`; a scene-referred
    /// screenshot; metering and adaptation; then, unless the view stops there, motion blur, the
    /// lens's glare, tonemap, `hooks.display_linear`, `hooks.ui`, compose and a display
    /// screenshot.
    ///
    /// Motion blur smears the radiance along each pixel's motion over the fraction
    /// `lens.shutter` of the time between presented images, centered on the frame. A camera cut
    /// (`view.camera_cut`, or [`SceneWriter::reset_lighting_history`] since the view's last
    /// render) shows none. A camera that jumps without declaring a cut blurs up to
    /// the longest streak, [`orr_output::MOTION_BLUR_MAX_RADIUS`] of the output width each way.
    ///
    /// A `lighting.debug_view` shows no motion blur, glare or lens distortion, since it is not
    /// the final image. The accumulation reference shows glare but no motion blur: each of its
    /// frames is ground truth and a caller simulates the shutter by stepping time, so blurring it
    /// here would blur it twice.
    ///
    /// The target decides the output extent. `output.reconstruction` and the output extent
    /// decide the render extent, the jitter and the texture level-of-detail bias while the
    /// renderer has ray reconstruction ([`Renderer::set_reconstruction`]); otherwise the view
    /// renders at the output extent. A new output extent or request remakes the view's feature,
    /// which waits for its setup on this thread. The jitter follows the view's renders, so a view
    /// not rendered every frame still covers its pixels. A feature the backend refuses is an
    /// error. Ray reconstruction reads lighting's guides
    /// ([`orr_lighting::LightingView::depth_of_field_guide`] among them, with the blur rough
    /// glass gives what lies behind it).
    ///
    /// While the kernels or the view's path pipelines compile ([`Frame::view_lit`] false) only the view's raster
    /// half runs, so streaming goes on while a loading thread is free, and everything after it
    /// holds black:
    /// [`RenderedView::compiling`] is set, ray reconstruction, exposure, motion blur, glare and
    /// frame generation do not run, and the `transparency` and `scene_referred` hooks do not run,
    /// since no radiance exists for them to draw into or to keep. `display_linear` and `ui` run over
    /// the black image, so a consumer can draw its own message there. The first render after that
    /// starts every history as a view's first render does. A compile that failed returns
    /// [`Error::PathKernels`], from here where it is a path pipeline's, which compile when a view
    /// first asks for them.
    pub fn render_view(
        &mut self,
        view: &LitView<'_>,
        lighting: &ViewLighting,
        output: &ViewOutput,
        hooks: &mut ViewHooks<'_>,
    ) -> Result<RenderedView> {
        if let Paths::Failed(e) = &self.lighting.paths {
            return Err(Error::PathKernels(Arc::clone(e)));
        }
        let (extent, render) = self.resolve_view(view, lighting, output)?;
        let lit = self.ask_paths(view, lighting, hooks.transparency.is_some(), true)?;
        let Paths::Ready(kernels) = &self.lighting.paths else {
            return self.render_view_black(view, output, hooks, extent, render);
        };
        if !lit {
            return self.render_view_black(view, output, hooks, extent, render);
        }
        if let Some(started) = self.lighting.started.take() {
            log::info!(
                "first lit view {:.1} s after the path tracing kernels started",
                started.elapsed().as_secs_f64()
            );
        }
        let id = self.scenes.id();
        let sdr = DisplayRequest::default().resolve(&[OutputFormat::Sdr], None, false);
        let frame = self.display.as_deref_mut().and_then(Display::frame);
        // Whatever the view's target: the consumer may compose a texture into the display.
        let generated = frame.as_ref().map_or(0, |f| f.generated);
        let mut display = frame.filter(|_| matches!(view.target, ViewTarget::Display));
        let resolved = display.as_ref().map_or(sdr, |f| f.display);
        let tonemapped = !matches!(view.target, ViewTarget::Radiance { .. });

        let state = self
            .output
            .views
            .get_mut(&view.history)
            .ok_or(SceneError::UnknownViewHistory(view.history))?;
        let reconstruction = state.reconstruction.as_ref().map(|r| r.feature.mode());
        let reconstructed = reconstruction.is_some();
        let mut desc = state.raster_desc(view, extent, render);
        let jitter = desc.jitter.to_array();
        let scene = self
            .lighting
            .scenes
            .get_mut(&view.scene)
            .ok_or(SceneError::UnknownScene(view.scene))?;
        // The view's first lit render, whatever raster-only frames came before it while the path
        // kernels compiled, its first on another scene and a lighting reset since its last are
        // all cuts.
        let scene_resets = (view.scene, scene.resets());
        desc.camera_cut |= state.scene_resets != Some(scene_resets);

        let set = Hooks {
            scene_referred: hooks.scene_referred.is_some(),
            display_linear: tonemapped && hooks.display_linear.is_some(),
        };
        let mut requests = LightingView::requests(render, reconstructed);
        if reconstructed {
            requests.push(output_image_request(RECONSTRUCTED, extent, ViewStep::Rr));
        }
        requests.push(output_image_request(
            VIEW_LINEAR,
            extent,
            ViewStep::SceneReferred,
        ));
        if set.any() {
            requests.extend(orr_output::hook_requests(extent));
        }
        if desc.camera_cut {
            self.scenes
                .restart_exposure(view.history, output.exposure.start())?;
            state.output.exposure_restarted(id);
        }
        let raster = self
            .scenes
            .raster_view_with(&desc, &requests, hooks.opaque.as_deref_mut())?;
        let gpu = self.output.device.clone();
        if let Some(at) = take_probe(&mut self.probes, view.history) {
            let probe = ProbeAt {
                frame: id,
                output: at,
                output_extent: extent,
                reconstruction,
            };
            record_probe(
                &mut self.scenes,
                self.lighting.kernels,
                &gpu,
                state,
                &raster,
                probe,
            )?;
        }
        let (list, arena) = self.scenes.list_and_arena(view.history)?;
        arena.enter(ViewStep::Lighting, list);

        if !self.lit.contains(&view.scene) {
            let pass = ScenePass {
                frame: id,
                scene: raster.scene,
                focus: raster.focus,
                tables: raster.light_tables,
                features: raster.features,
            };
            if let Err(e) = scene.record(list, &pass) {
                // What the pass makes when the scene first needs it finds memory once the
                // budgets have made room.
                if let orr_lighting::Error::Gpu(e) = &e {
                    self.scenes.out_of_memory(e);
                }
                return Err(e.into());
            }
            self.lit.push(view.scene);
        }
        match &mut state.lighting {
            Some(l) if l.extent() == render => l.bind(arena)?,
            l => *l = Some(LightingView::new(&gpu, arena)?),
        }
        let paths = state.lighting.as_mut().expect("lighting view");
        // A debug view is not the final image, so it shows no camera effects. A view that stops
        // after exposure holds none of their images. Depth of field takes this rule too once it
        // runs here: the reference traces its own lens.
        let look = lighting.debug_view == LightingDebugView::None;
        let blurs = tonemapped && look && lighting.reference.is_none();
        let pass = ViewPass {
            frame: id,
            scene: raster.scene,
            view: raster.view,
            gbuffer: raster.gbuffer,
            raster: raster.raster,
            smoke_args: raster.smoke_args,
            sample_index: state.samples,
            seed: 0,
            dof_background: None,
            frosted_glass: reconstructed,
            overlay_front: blurs && view.lens.shutter > 0.0,
            scene_features: raster.features,
            particle_lighting: hooks.transparency.is_some(),
        };
        paths.begin(scene, &desc, lighting);
        state.scene_resets = Some(scene_resets);
        let traced = Stages::COMPUTE | Stages::RAY;
        list.barrier(Stages::ALL, traced, Hazards::empty());
        kernels.record_media(list, scene, paths, &pass, lighting)?;
        kernels.record_paths(list, scene, paths, &pass, lighting)?;
        let exact_paths = paths.paths_ran().is_none_or(|ran| ran.exact());
        state.samples = state.samples.wrapping_add(1);
        list.barrier(traced, Stages::COMPUTE, Hazards::empty());
        kernels.record_media_apply(list, scene, paths, &pass, lighting)?;
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        if let LightingDebugView::Traced(kind) = lighting.debug_view {
            state.traced.record(
                &gpu,
                self.lighting.kernels,
                list,
                id,
                kind,
                pass.scene,
                pass.view,
                pass.gbuffer,
                paths.radiance_storage(),
                paths.extent(),
            )?;
        }
        let mut overlay = None;
        if let Some(hook) = hooks.transparency.as_mut() {
            let particles = paths.particle_lighting(id);
            let mut hook = |pass: &mut TransparencyPass<'_, '_>| hook(pass, particles);
            overlay = Some(if reconstructed {
                self.scenes.transparency_overlay(&raster, &mut hook)?
            } else {
                let store = paths.radiance_storage();
                self.scenes.transparency(&raster, store, &mut hook)?
            });
        }
        let list = self.scenes.list()?;
        if pass.overlay_front {
            let sampled = overlay.as_ref().map(|(_, sampled)| *sampled);
            kernels.record_overlay_front(list, paths, sampled);
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        }
        if reconstructed || matches!(lighting.debug_view, LightingDebugView::Guide(_)) {
            let list = self.scenes.list()?;
            kernels.record_guides(list, paths, &pass, lighting)?;
            if let Some(guide) = paths.depth_of_field_guide_storage() {
                list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
                kernels.record_frosted_glass(list, paths, &pass, guide);
            }
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        }
        let overlay_front = paths
            .overlay_front_indices()
            .map(|[depth, motion, weight]| OverlayFront {
                depth,
                motion,
                weight,
            });
        let delta = self.scenes.time().delta;
        let completed = self.scenes.completed();
        let stepped = state.output.pre_exposure_steps(completed);
        let (list, arena) = self.scenes.output_list_and_arena(view.history)?;
        let (rr_output, rr_output_store) = match state.reconstruction.as_mut() {
            Some(r) => {
                arena.enter(ViewStep::Rr, list);
                let target = arena.texture(RECONSTRUCTED).expect("requested above");
                list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
                list.begin_pass("ray_reconstruction", false);
                let recorded = r.feature.record(
                    list,
                    &ReconstructionInputs {
                        color: paths.radiance(),
                        color_before_transparency: None,
                        depth: paths.depth(),
                        motion: paths.motion(),
                        diffuse_albedo: paths.diffuse_albedo(),
                        specular_albedo: paths.specular_albedo(),
                        normal_roughness: paths.normal_roughness(),
                        specular_motion: paths.specular_motion(),
                        specular_hit_distance: None,
                        transparency: overlay.as_ref().map(|(texture, _)| &**texture),
                        depth_of_field_guide: paths.depth_of_field_guide(),
                        denoise_strength_mask: None,
                        reactive_mask: None,
                        exposure: None,
                        output: target.texture(),
                        reset: std::mem::take(&mut r.fresh) || paths.history_reset() || stepped,
                        view: &desc,
                    },
                );
                list.end_pass();
                recorded?;
                list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
                (target.sampled(), target.storage(0))
            }
            None => (paths.radiance_index(), paths.radiance_storage()),
        };
        arena.enter(ViewStep::SceneReferred, list);
        let linear = arena.texture(VIEW_LINEAR).expect("requested above").clone();
        let mut chain = ChainImages {
            rr_output,
            rr_output_store,
            linear: linear.sampled(),
            linear_store: linear.storage(0),
        };
        let (depth, motion, rr_depth, rr_motion) = if reconstructed {
            (
                paths.depth(),
                paths.motion(),
                paths.depth_index(),
                paths.motion_index(),
            )
        } else {
            (
                &raster.depth,
                &raster.motion,
                raster.targets.depth,
                raster.targets.motion,
            )
        };
        if let Some(frame) = display.as_mut() {
            frame.frame_generation(&desc, depth, motion, view.not_rendering_scene, delta);
        }
        let distortion = output.lens_distortion(view.lens.distortion.filter(|_| look));
        let field_on = distortion.is_some() || (output.distortion_field && set.any());
        let mut field = match display.as_mut() {
            Some(frame) if field_on => Some(frame.distortion_field()?),
            _ => None,
        };
        let kernels = &self.output.kernels;
        let inputs = kernels.hook_inputs(
            list,
            &mut state.output,
            arena,
            &HookInputsPass {
                rr_depth,
                rr_motion,
                chain,
                extent,
                view: raster.view,
                jitter,
                frame: id,
                hooks: set,
            },
            field.as_deref_mut(),
        )?;
        if let (Some(inputs), Some(hook)) = (inputs, hooks.scene_referred.as_mut()) {
            inputs.run(
                list,
                HookPoint::SceneReferred,
                &chain,
                field.as_deref_mut(),
                &mut **hook,
            );
        }
        if let Some(ScreenshotRequest::SceneReferred { rec709 }) = view.screenshot {
            kernels.capture_scene_referred(
                list,
                &mut state.output,
                &SceneCapturePass {
                    source: chain.rr_output,
                    extent,
                    view: raster.view,
                    rec709,
                    display: &resolved,
                    frame: id,
                },
            )?;
        }
        arena.enter(ViewStep::Exposure, list);
        kernels.exposure(
            list,
            &mut state.output,
            &ExposurePass {
                source: chain.rr_output,
                extent,
                view: raster.view,
                sky: raster.sky_stats,
                frame: id,
                completed,
                output,
            },
        )?;
        kernels.motion_blur(
            list,
            &mut state.output,
            &MotionBlurPass {
                rr_depth,
                rr_motion,
                overlay: overlay_front,
                extent,
                view: raster.view,
                shutter: if blurs { view.lens.shutter } else { 0.0 },
                generated,
                camera_cut: desc.camera_cut,
                frame: id,
                output,
            },
            &mut chain,
        )?;
        let glare = kernels.glare(
            list,
            &mut state.output,
            &GlarePass {
                source: chain.rr_output,
                extent,
                view: raster.view,
                sensor_height_mm: view.lens.sensor_height_mm,
                glare: view.lens.glare.filter(|_| tonemapped && look),
                display: &resolved,
                frame: id,
                output,
            },
        )?;
        if !tonemapped {
            arena.enter(ViewStep::After, list);
            return Ok(RenderedView {
                raster,
                render_extent: render,
                reconstruction,
                radiance: chain.rr_output,
                linear: TextureIndex::NONE,
                compiling: false,
                exact_paths,
            });
        }

        arena.enter(ViewStep::Tonemap, list);
        kernels.tonemap(
            list,
            state.output.roots(id),
            &TonemapPass {
                source: chain.rr_output,
                target: chain.linear_store,
                extent,
                view: raster.view,
                display: &resolved,
                output,
                hook_source: set.display_linear.then_some(chain.rr_output_store),
                glare,
                distortion,
                frame: id,
            },
            field.as_deref_mut(),
        );
        list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        arena.enter(ViewStep::DisplayLinear, list);
        if let (Some(inputs), Some(hook)) = (inputs, hooks.display_linear.as_mut()) {
            inputs.run(list, HookPoint::DisplayLinear, &chain, field, &mut **hook);
        }
        finish_view(
            &mut self.scenes,
            &mut self.slots,
            &gpu,
            kernels,
            state,
            display.as_mut(),
            ViewEnd {
                view,
                ui: hooks.ui.as_mut().map(|ui| &mut **ui as _),
                linear: chain.linear,
                extent,
                display: &resolved,
            },
        )?;
        Ok(RenderedView {
            raster,
            render_extent: render,
            reconstruction,
            radiance: chain.rr_output,
            linear: chain.linear,
            compiling: false,
            exact_paths,
        })
    }

    /// [`Frame::render_view`] while the path kernels compile: the raster half, then black. It
    /// leaves the view's lighting, exposure and reconstruction state as it found them, and frame
    /// generation restarts on a frame that does not ask for it.
    fn render_view_black(
        &mut self,
        view: &LitView<'_>,
        output: &ViewOutput,
        hooks: &mut ViewHooks<'_>,
        extent: [u32; 2],
        render: [u32; 2],
    ) -> Result<RenderedView> {
        let id = self.scenes.id();
        let sdr = DisplayRequest::default().resolve(&[OutputFormat::Sdr], None, false);
        let frame = self.display.as_deref_mut().and_then(Display::frame);
        let mut display = frame.filter(|_| matches!(view.target, ViewTarget::Display));
        let resolved = display.as_ref().map_or(sdr, |f| f.display);
        let tonemapped = !matches!(view.target, ViewTarget::Radiance { .. });

        let state = self
            .output
            .views
            .get_mut(&view.history)
            .ok_or(SceneError::UnknownViewHistory(view.history))?;
        // The next lit render starts every history over, as a view's first does.
        state.scene_resets = None;
        // The render size, jitter and bias the first lit render will use, so geometry and
        // streaming settle for it.
        let desc = state.raster_desc(view, extent, render);
        let set = Hooks {
            scene_referred: false,
            display_linear: tonemapped && hooks.display_linear.is_some(),
        };
        let mut requests = vec![
            output_image_request(RECONSTRUCTED, extent, ViewStep::Rr),
            output_image_request(VIEW_LINEAR, extent, ViewStep::SceneReferred),
        ];
        if set.any() {
            requests.extend(orr_output::hook_requests(extent));
        }
        let raster = self
            .scenes
            .raster_view_with(&desc, &requests, hooks.opaque.as_deref_mut())?;
        if let Some(at) = take_probe(&mut self.probes, view.history) {
            let probe = ProbeAt {
                frame: id,
                output: at,
                output_extent: extent,
                reconstruction: None,
            };
            let gpu = self.output.device.clone();
            record_probe(
                &mut self.scenes,
                self.lighting.kernels,
                &gpu,
                state,
                &raster,
                probe,
            )?;
        }
        let (list, arena) = self.scenes.output_list_and_arena(view.history)?;
        arena.enter(ViewStep::Rr, list);
        let radiance = arena
            .texture(RECONSTRUCTED)
            .expect("requested above")
            .clone();
        arena.enter(ViewStep::SceneReferred, list);
        let linear = arena.texture(VIEW_LINEAR).expect("requested above").clone();
        let chain = ChainImages {
            rr_output: radiance.sampled(),
            rr_output_store: radiance.storage(0),
            linear: linear.sampled(),
            linear_store: linear.storage(0),
        };
        let mut field = match display.as_mut() {
            Some(frame) if output.distortion_field && set.any() => Some(frame.distortion_field()?),
            _ => None,
        };
        let kernels = &self.output.kernels;
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        let black = [chain.rr_output_store, chain.linear_store];
        kernels.black(list, &mut state.output, id, black, extent, &sdr);
        list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        let inputs = kernels.hook_inputs(
            list,
            &mut state.output,
            arena,
            &HookInputsPass {
                rr_depth: raster.targets.depth,
                rr_motion: raster.targets.motion,
                chain,
                extent,
                view: raster.view,
                jitter: desc.jitter.to_array(),
                frame: id,
                hooks: set,
            },
            field.as_deref_mut(),
        )?;
        if let Some(ScreenshotRequest::SceneReferred { rec709 }) = view.screenshot {
            kernels.capture_scene_referred(
                list,
                &mut state.output,
                &SceneCapturePass {
                    source: chain.rr_output,
                    extent,
                    view: raster.view,
                    rec709,
                    display: &resolved,
                    frame: id,
                },
            )?;
        }
        if tonemapped {
            arena.enter(ViewStep::DisplayLinear, list);
            if let (Some(inputs), Some(hook)) = (inputs, hooks.display_linear.as_mut()) {
                inputs.run(list, HookPoint::DisplayLinear, &chain, field, &mut **hook);
            }
            finish_view(
                &mut self.scenes,
                &mut self.slots,
                &self.output.device,
                kernels,
                state,
                display.as_mut(),
                ViewEnd {
                    view,
                    ui: hooks.ui.as_mut().map(|ui| &mut **ui as _),
                    linear: chain.linear,
                    extent,
                    display: &resolved,
                },
            )?;
        } else {
            arena.enter(ViewStep::After, list);
        }
        Ok(RenderedView {
            compiling: true,
            exact_paths: false,
            raster,
            render_extent: render,
            reconstruction: None,
            radiance: chain.rr_output,
            linear: if tonemapped {
                chain.linear
            } else {
                TextureIndex::NONE
            },
        })
    }

    /// The frame's command list on the Render queue, for the consumer's own passes.
    /// Everything recorded into it runs in order with orr's raster, lighting and UI passes
    /// and before [`Frame::output_list`], and is submitted by [`Frame::end`]. The first call
    /// runs [`Frame::update_scenes`] if the frame has not.
    pub fn list(&mut self) -> Result<&mut CommandList<'a>> {
        Ok(self.scenes.list()?)
    }

    /// The list orr's output passes record into, for the consumer's ray reconstruction and
    /// passes that read a view's output: compute work only. It runs after everything in
    /// [`Frame::list`], whatever the recording order. On a device with a Compute queue it
    /// runs there, overlapping the next frame's [`Frame::update_scenes`]; without one it is
    /// [`Frame::list`].
    pub fn output_list(&mut self) -> Result<&mut CommandList<'a>> {
        Ok(self.scenes.output_list()?)
    }

    /// Runs `ui` into the display's UI layer, in [`Frame::list`]. In a frame that renders no
    /// view into the display (menus, loading screens) the present thread composes it over
    /// black, or over the frame's [`Frame::pattern`]; after [`Frame::compose`] and
    /// [`Frame::frame_generation`] it goes over the consumer's image and, with
    /// `Display::set_ui_per_present` on, runs once per presented image. Each run records
    /// inside a pass named [`orr_output::UI_PASS`]. Does nothing without a display or while
    /// its window has no area.
    pub fn ui_only(&mut self, ui: &mut dyn FnMut(&mut UiPass<'_, '_>)) -> Result<()> {
        let Some(display) = self.display.as_deref_mut() else {
            return Ok(());
        };
        let delta = self.scenes.time().delta;
        let list = self.scenes.list()?;
        let Some(mut frame) = display.frame() else {
            return Ok(());
        };
        paint_ui(list, &mut frame, delta, ui);
        Ok(())
    }

    /// A frame that renders no view into the display (calibration screens): fills the
    /// display with `pattern`, encoded at the frame's resolved display without
    /// tonemapping, so each level reaches the display at its stated luminance. The patch
    /// is a centered square half the window's height across. Does nothing without a display
    /// or while its window has no area.
    pub fn pattern(&mut self, pattern: CalibrationPattern) -> Result<()> {
        let Some(display) = self.display.as_deref_mut() else {
            return Ok(());
        };
        let id = self.scenes.id();
        let list = self.scenes.output_list()?;
        let Some(mut frame) = display.frame() else {
            return Ok(());
        };
        let output = &mut *self.output;
        let linear = output.linear(list, frame.extent)?;
        let (sampled, storage) = (linear.sampled, linear.storage);
        let view_output = ViewOutput {
            debug: OutputDebug::Pattern(pattern),
            ..ViewOutput::default()
        };
        output.kernels.tonemap(
            list,
            output.roots.roots(id),
            &TonemapPass {
                source: TextureIndex::NONE,
                target: storage,
                extent: frame.extent,
                view: GpuPtr::NULL,
                display: &frame.display,
                output: &view_output,
                hook_source: None,
                glare: None,
                distortion: None,
                frame: id,
            },
            None,
        );
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        output.compose(list, id, &mut frame, sampled);
        Ok(())
    }

    /// For a view the consumer shaded itself and showed with [`Frame::compose`]: the camera
    /// frame generation reprojects the display's image with, and the view's reverse-Z
    /// `depth` and `motion` (previous UV minus current UV) at its render extent.
    /// [`Frame::render_view`] does this for the views it renders into the display. Does
    /// nothing without a display or while its window has no area.
    pub fn frame_generation(
        &mut self,
        view: &ViewDesc,
        depth: &Arc<orr_gpu::Texture>,
        motion: &Arc<orr_gpu::Texture>,
        not_rendering_scene: bool,
    ) {
        let delta = self.scenes.time().delta;
        if let Some(mut frame) = self.display.as_deref_mut().and_then(Display::frame) {
            frame.frame_generation(view, depth, motion, not_rendering_scene, delta);
        }
    }

    /// Encodes `linear`, display-linear Rec.709 with 1.0 at paper white, into the display
    /// at the frame's resolved display, for a view the consumer shaded itself. The image
    /// covers the window's extent; texels past its edge repeat the last row and column.
    /// Records into [`Frame::output_list`] after a barrier, so the image may be written
    /// just before in either list. Does nothing without a display or while its window has
    /// no area.
    pub fn compose(&mut self, linear: TextureIndex) -> Result<()> {
        let Some(display) = self.display.as_deref_mut() else {
            return Ok(());
        };
        let id = self.scenes.id();
        let list = self.scenes.output_list()?;
        let Some(mut frame) = display.frame() else {
            return Ok(());
        };
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.output.compose(list, id, &mut frame, linear);
        Ok(())
    }

    /// Step 6: records the display's frame generation, submits the frame after the
    /// display's slot is free, hands it to the present thread without waiting, and returns
    /// the point its last submit signals, after which all its work has completed.
    pub fn end(mut self) -> Result<TimelinePoint> {
        let wait = self
            .display
            .as_deref_mut()
            .and_then(Display::frame)
            .and_then(|f| f.wait);
        if let Some(d) = self.display.as_deref_mut() {
            // Compose wrote the backbuffer on the output list, if it was written.
            let list = if self.scenes.has_output_list() {
                self.scenes.output_list()?
            } else {
                self.scenes.list()?
            };
            d.record_frame_generation(list)?;
        }
        let gpu = self.output.device.clone();
        let slots = std::mem::take(&mut self.slots);
        let ended = self.scenes.end_submits(wait.as_slice());
        let last = ended.as_ref().ok().and_then(|p| p.last().copied());
        for slot in slots {
            match last {
                Some(point) => gpu.free_image_slot(slot, point),
                None => gpu.free_image_slot_after_submitted(slot),
            }
        }
        let submits = ended?;
        if let Some(d) = self.display {
            d.present(&submits)?;
        }
        Ok(last.expect("a frame submits"))
    }
}

/// What the end of a view's render works on.
struct ViewEnd<'x> {
    view: &'x LitView<'x>,
    ui: Option<&'x mut dyn FnMut(&mut UiPass<'_, '_>)>,
    /// The display-linear image.
    linear: TextureIndex,
    extent: [u32; 2],
    display: &'x ResolvedDisplay,
}

/// The end of a view's render with an image, after its display-linear hook: its UI, compose into
/// its target and a display screenshot.
fn finish_view(
    scenes: &mut orr_scene::Frame<'_>,
    slots: &mut Vec<u32>,
    gpu: &Device,
    kernels: &OutputKernels,
    state: &mut ViewState,
    mut frame: Option<&mut DisplayFrame<'_>>,
    end: ViewEnd<'_>,
) -> Result<()> {
    let ViewEnd {
        view,
        ui,
        linear,
        extent,
        display,
    } = end;
    let id = scenes.id();
    let delta = scenes.time().delta;
    let (list, arena) = scenes.output_list_and_arena(view.history)?;
    arena.enter(ViewStep::Compose, list);
    let ui = match (frame.as_deref_mut(), ui) {
        (Some(frame), Some(hook)) => {
            let list = scenes.list()?;
            let index = paint_ui(list, frame, delta, hook);
            list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
            Some(index)
        }
        _ => None,
    };
    let (list, arena) = scenes.output_list_and_arena(view.history)?;
    let store = match (view.target, frame) {
        (ViewTarget::Texture(t), _) => {
            list.discard_after_use(t);
            let slot = gpu.alloc_image_slot()?;
            slots.push(slot);
            gpu.write_storage(slot, t, &orr_gpu::ViewDesc::default())
        }
        (_, Some(frame)) => frame.backbuffer(list),
        _ => StorageImageIndex::NONE,
    };
    let n = state.compose.len() as u64;
    kernels.compose(
        list,
        &mut state.compose[(id % n) as usize],
        &ComposePass {
            source: ComposeSource::Linear(linear),
            ui: None,
            target: store,
            extent,
            display,
        },
    );
    if let Some(ScreenshotRequest::Display { include_ui }) = view.screenshot {
        kernels.capture(
            list,
            &mut state.output,
            &CapturePass {
                source: ComposeSource::Linear(linear),
                ui: ui.filter(|_| include_ui),
                extent,
                display,
                frame: id,
            },
        )?;
    }
    arena.enter(ViewStep::After, list);
    Ok(())
}

/// Runs `hook` once per UI layer of `frame`, each inside a [`UI_PASS`] pass with its
/// presented image's index and time offset, and returns the sampled view of the frame's own
/// image's layer.
fn paint_ui(
    list: &mut CommandList<'_>,
    frame: &mut DisplayFrame<'_>,
    delta: f32,
    hook: &mut dyn FnMut(&mut UiPass<'_, '_>),
) -> TextureIndex {
    let mut index = TextureIndex::NONE;
    for presented in 0..frame.ui_layers() {
        let time_offset = frame.ui_time_offset(presented, delta);
        let layer = frame.ui_layer(list, presented);
        index = layer.index();
        list.begin_pass(UI_PASS, false);
        hook(&mut UiPass {
            cmd: list,
            layer: layer.texture(),
            presented,
            time_offset,
        });
        list.end_pass();
    }
    index
}
