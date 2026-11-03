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
//!    one path per pixel, DLSS ray reconstruction, the hooks, exposure, motion blur, glare,
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
use std::sync::Arc;
use std::time::{Duration, Instant};

pub use orr_lighting::{
    self, CompilingKernels, Error as LightingError, LightingDebugView, LightingStats,
    MediaSettings, ParticleDraw, ParticleLighting, PathEntry, ReferenceSettings,
    SceneLightingSettings, ThinLens, ViewLighting, WorldCacheSettings,
};
#[cfg(windows)]
pub use orr_output::DisplayWatch;
pub use orr_output::{self, Error as OutputError};
pub use orr_output::{
    AutoExposure, AutoWhiteBalance, CalibrationPattern, CompensationCurve, Display, DisplayFeed,
    DisplayInfo, DisplayRequest, DlssMode, DlssQuality, DlssSdk, DlssSupport, Exposure,
    ExposureTelemetry, FrameGenerationMode, GlareMethod, Grade, Gt7Params, HookPass, LocalExposure,
    MeterWeights, OutputDebug, Physiological, PostQuality, Presented, Primaries, Reconstruction,
    ResolvedDisplay, Screenshot, ScreenshotRequest, SkyTelemetry, UiPass, ViewOutput,
};
pub use orr_scene::Error as SceneError;
pub use orr_scene::{self, RasterView};
pub use orr_scene::{
    Affine3, ArenaReport, ArenaTarget, BlobId, BlobTarget, BoundTexture, Budgets, ClassStats,
    ClipRef, ClusterRef, ClusterTopology, Cone, ConsumerMarker, CopyVariant, Decal, DecalChannels,
    DecalId, DepthOfField, DirectionalLight, DiscProfile, EditDesc, EditGroupGpu, EditId, EditRef,
    EmissiveGpu, EmissiveSpan, ExposureState, Focus, FrameStats, FrameTime, Glare, GroupGpu,
    GroupWriter, InstanceFlags, InstanceGpu, InstancePage, InstancePageRef, InstanceRange,
    InstancesDesc, JobResult, Lens, LensDistortion, Light, LightGpu, LightId, LoadState, Loading,
    MaterialGpu, MaterialId, MaterialWriter, MeshGpu, ModelEmission, ModelGpu, ModelId, MountId,
    NeuralMipGpu, NeuralSetGpu, NeuralTextureGpu, NeuralTextures, Opacity, OriginPolicy,
    PackSource, PageId, PipelineId, Pose, ProducerBatch, ProducerClusterGpu, ProducerDesc,
    ProducerGpu, ProducerId, ProducerStats, ProducerUpdate, Projection, REBASE_GRID_LOG2,
    RasterGpu, ReloadReport, RendererDesc, ResidencyClass, ResidencyStats, RtCutGpu, RtStats,
    SceneCompute, SceneDesc, SceneId, SceneRoot, SceneWriter, SkeletonRef, SkinStats, SkinnedDesc,
    SkyStats, SlotStateGpu, Span, TextureFeedbackGpu, TextureFilter, TextureStreamGpu, TopologyGpu,
    TransformGroupDesc, TransformGroupId, TransparencyPass, ViewConstants, ViewDesc, ViewHistoryId,
    ViewStep, WorldPos, abi, material_flags, orr_gpu, orr_pack,
};

use orr_gpu::{
    CommandList, Device, Format, GpuPtr, Hazards, KernelSet, OutputFormat, ShaderModule, Stages,
    StorageImageIndex, TextureDesc, TextureIndex, TextureUsage, TimelinePoint,
};
use orr_lighting::{BsdfTables, LightingKernels, LightingView, SceneLighting, ScenePass, ViewPass};
use orr_math::{DVec3, Quat, UVec2, Vec2};
use orr_output::orr_dlss::{MIN_OUTPUT as DLSS_MIN_OUTPUT, RayReconstruction, RrInputs};
use orr_output::{
    CapturePass, ChainImages, ComposePass, ComposeRoot, ComposeSource, DisplayFrame, ExposurePass,
    GlarePass, HookInputsPass, HookPoint, Hooks, Image, MotionBlurPass, OutputKernels, OutputView,
    SceneCapturePass, TonemapPass, UI_PASS,
};
use orr_scene::{ArenaRequest, Scenes};

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
    /// The view's target is the display, and the frame has none or its window has no area.
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
    /// Into the view's transparency overlay after its opaque and dielectric lighting and
    /// media apply, at the render extent, with the view's [`ParticleLighting`] for this
    /// frame, which [`ParticleDraw`] and a hook's own `particle_shade` light and fog
    /// particles with; the overlay is composited over the radiance before `scene_referred`.
    /// Particle producer entries are drawn here or not at all for primary visibility.
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
    /// frame generation, ray reconstruction and lighting drop their history, auto exposure and
    /// white balance snap. A view's first render is a cut without it.
    pub camera_cut: bool,
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
    pub raster: RasterView,
    /// The extent the view rendered at.
    pub render_extent: [u32; 2],
    /// The mode ray reconstruction ran in; `None` when it did not run.
    pub reconstruction: Option<DlssMode>,
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
    scenes: HashMap<SceneId, SceneLighting>,
    tables: BsdfTables,
}

impl Lighting {
    /// Starts the path kernels unless they are ready or compiling: from the pipeline caches when
    /// they hold every one and `try_cached_paths` allows looking, else as jobs on `loading`, which
    /// report any failure.
    fn start_paths(&mut self, gpu: &Device, loading: &Loading) {
        if matches!(self.paths, Paths::Ready(_) | Paths::Compiling { .. }) {
            return;
        }
        if self.try_cached_paths {
            let cached = gpu.cached_pipelines_only();
            if let Ok(kernels) =
                LightingKernels::new(&cached, self.kernels, &self.tables, self.entry)
            {
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
    reconstruction: Option<ViewReconstruction>,
}

/// A view's ray reconstruction feature and what it was made for.
struct ViewReconstruction {
    feature: RayReconstruction,
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
    /// without `sdk` or for [`Reconstruction::Off`].
    fn sync_reconstruction(
        &mut self,
        sdk: Option<&DlssSdk>,
        output: [u32; 2],
        request: Reconstruction,
    ) -> Result<()> {
        let current = self
            .reconstruction
            .as_ref()
            .is_some_and(|r| r.request == request && r.feature.output_size() == output);
        if current && sdk.is_some() {
            return Ok(());
        }
        self.reconstruction = None;
        let Some(sdk) = sdk else {
            return Ok(());
        };
        let feature = match request {
            Reconstruction::Off => return Ok(()),
            Reconstruction::Quality(quality) => RayReconstruction::new(sdk, output, quality, None),
            Reconstruction::RenderExtent(render) => {
                let wide = u64::from(render[0]) * u64::from(output[1]);
                let tall = u64::from(render[1]) * u64::from(output[0]);
                // One render pixel of rounding on either axis.
                if wide.abs_diff(tall) > u64::from(output[0].max(output[1])) {
                    return Err(Error::RenderAspect { render, output });
                }
                RayReconstruction::with_render_size(sdk, render, output, None)
            }
        };
        self.reconstruction = Some(ViewReconstruction {
            feature: feature.map_err(OutputError::from)?,
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
    /// NGX, while it offers ray reconstruction.
    dlss: Option<DlssSdk>,
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
            dlss: None,
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
        .chain(orr_lighting::shaders::MODULES)
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
                scenes: HashMap::new(),
                tables,
            },
            output: Output::new(gpu, desc.frames_in_flight)?,
        })
    }

    pub fn device(&self) -> &Device {
        self.scenes.device()
    }

    /// Gives lit views DLSS ray reconstruction, when NGX offers it on the device; returns
    /// whether it does. `sdk` must be on this renderer's device.
    pub fn set_dlss(&mut self, sdk: DlssSdk) -> bool {
        self.output.dlss = sdk.support().rr.then_some(sdk);
        self.output.dlss.is_some()
    }

    /// A scene's tables and the lighting state every view of the scene shares.
    pub fn create_scene(
        &mut self,
        desc: &SceneDesc,
        lighting: &SceneLightingSettings,
    ) -> Result<SceneId> {
        let gpu = self.scenes.device().clone();
        let state = SceneLighting::new(&gpu, self.lighting.kernels, desc, lighting)?;
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
    /// The first call, and the first after a failed compile, starts the path tracing pipelines
    /// without compiling on this thread. If
    /// the pipeline caches hold them all ([`RendererDesc::try_cached_paths`]), they are ready
    /// when it returns, in tens of milliseconds. Otherwise they compile as jobs on
    /// [`Renderer::loading`], [`CompilingKernels::JOBS`] of them, which take minutes on a first
    /// run, holding their threads the whole time. Until then [`Frame::render_view`] renders black.
    /// The jobs need threads: lend them with [`Loading::run_worker`] or
    /// [`Loading::run_one`], at least [`CompilingKernels::JOBS`] to compile in parallel and more
    /// to keep loading meanwhile, or block in [`Renderer::wait_paths`]. With neither, views stay
    /// black, and a warning is logged after a few seconds of frames.
    ///
    /// Each job writes the device's pipeline cache file when it finishes. Dropping the renderer
    /// cancels the jobs no thread has started. A thread already compiling finishes that pipeline,
    /// which a consumer joining its workers waits for, since the driver cannot be interrupted.
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
            reconstruction: None,
        };
        let id = self.scenes.create_view_history()?;
        self.output.views.insert(id, state);
        Ok(id)
    }

    /// Whether the path tracing kernels have compiled, so the next frame's views render lit. False
    /// also before the first view history and after a failed compile.
    pub fn paths_ready(&mut self) -> bool {
        self.lighting.paths.poll();
        matches!(self.lighting.paths, Paths::Ready(_))
    }

    /// How many of the path tracing kernels' compile jobs have not finished.
    pub fn paths_outstanding(&self) -> u32 {
        match &self.lighting.paths {
            Paths::Compiling { jobs, .. } => jobs.outstanding(),
            _ => 0,
        }
    }

    /// Blocks until the path tracing kernels have compiled, running queued loading jobs on this
    /// thread so it needs no workers, and returns [`Error::PathKernels`] if they failed. For
    /// tests and headless runs; returns at once when they are ready or never started.
    pub fn wait_paths(&mut self) -> Result<()> {
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
}

impl<'a> Frame<'a> {
    /// orr's frame id: strictly increasing from 1.
    pub fn id(&self) -> u64 {
        self.scenes.id()
    }

    /// Whether this frame's views render lit, or black because the path tracing kernels are still
    /// compiling. The answer holds for the whole frame.
    pub fn paths_ready(&self) -> bool {
        matches!(self.lighting.paths, Paths::Ready(_))
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

    /// What hot reload applied at this frame's step 1.
    pub fn reloads(&self) -> &[ReloadReport] {
        self.scenes.reloads()
    }

    /// Step 2: the writer of a scene listed in this frame. One per scene at a time; Send.
    pub fn scene(&self, id: SceneId) -> Result<SceneWriter<'_>> {
        Ok(self.scenes.scene(id)?)
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

    /// Step 3: applies every write, then hands each listed scene in order to `compute`.
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
        self.prepare_raster_view(view)?;
        Ok(self.scenes.raster_view(view)?)
    }

    /// Step 2: works out the render extent [`Frame::render_view`] gives `view` with these
    /// settings, remaking the view's NGX feature as that call would, and asks the view's scene
    /// to choose geometry for it at this frame's [`Frame::update_scenes`]. A scene takes the
    /// sharpest of its views. Only a view whose render comes after the update needs this;
    /// without it the view counts from the next frame on.
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
        let dlss = self.output.dlss.as_ref().filter(|_| {
            lighting.reference.is_none()
                && lighting.debug_view == LightingDebugView::None
                && extent[0] >= DLSS_MIN_OUTPUT[0]
                && extent[1] >= DLSS_MIN_OUTPUT[1]
        });
        let state = self
            .output
            .views
            .get_mut(&view.history)
            .ok_or(SceneError::UnknownViewHistory(view.history))?;
        state.sync_reconstruction(dlss, extent, output.reconstruction)?;
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
    /// radiance, then the dielectric dispatch over it, which traces each pixel's first
    /// dielectric or presence layer; media apply over each pixel's camera segment; the
    /// transparency hook; DLSS ray reconstruction; `hooks.scene_referred`; a scene-referred
    /// screenshot; metering and adaptation; then, unless the view stops there, motion blur, the
    /// lens's glare, tonemap, `hooks.display_linear`, `hooks.ui`, compose and a display
    /// screenshot.
    ///
    /// Motion blur smears the radiance along each pixel's motion over the fraction
    /// `lens.shutter` of the time between presented images, centered on the frame. A camera cut
    /// (`view.camera_cut`) shows none. A camera that jumps without declaring a cut blurs up to
    /// the longest streak, [`orr_output::MOTION_BLUR_MAX_RADIUS`] of the output width each way.
    ///
    /// A `lighting.debug_view` shows neither motion blur nor glare, since it is not the final
    /// image. The accumulation reference shows glare but no motion blur: each of its frames is
    /// ground truth and a caller simulates the shutter by stepping time, so blurring it here would
    /// blur it twice.
    ///
    /// The target decides the output extent. `output.reconstruction` and the output extent
    /// decide the render extent, the jitter and the texture level-of-detail bias while the
    /// renderer has ray reconstruction ([`Renderer::set_dlss`]); otherwise the view renders at
    /// the output extent. A new output extent or request remakes the view's NGX feature, which
    /// waits for its setup on this thread. The jitter follows the view's renders, so a view not
    /// rendered every frame still covers its pixels. A feature NGX refuses is an error.
    ///
    /// While the path tracing kernels compile ([`Frame::paths_ready`] false) only the view's raster
    /// half runs, so streaming goes on while a loading thread is free, and everything after it
    /// holds black:
    /// [`RenderedView::compiling`] is set, ray reconstruction, exposure, motion blur, glare and
    /// frame generation do not run, and the `transparency` and `scene_referred` hooks do not run,
    /// since no radiance exists for them to draw into or to keep. `display_linear` and `ui` run over
    /// the black image, so a consumer can draw its own message there. The first render after that
    /// starts every history as a view's first render does. A compile that failed returns
    /// [`Error::PathKernels`].
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
        let Paths::Ready(kernels) = &self.lighting.paths else {
            return self.render_view_black(view, output, hooks, extent, render);
        };
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
        // A first lit render is a cut, whatever raster-only frames came before it while the path
        // kernels compiled.
        desc.camera_cut |= state.lighting.is_none();

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
        let raster = self.scenes.raster_view_with(&desc, &requests)?;
        let gpu = self.output.device.clone();
        let (list, arena) = self.scenes.list_and_arena(view.history)?;
        arena.enter(ViewStep::Lighting, list);

        let scene = self
            .lighting
            .scenes
            .get_mut(&view.scene)
            .ok_or(SceneError::UnknownScene(view.scene))?;
        if !self.lit.contains(&view.scene) {
            scene.record(
                list,
                &ScenePass {
                    frame: id,
                    scene: raster.scene,
                    focus: raster.focus,
                },
            );
            self.lit.push(view.scene);
        }
        match &mut state.lighting {
            Some(l) if l.extent() == render => l.bind(arena)?,
            l => *l = Some(LightingView::new(&gpu, arena)?),
        }
        let paths = state.lighting.as_mut().expect("lighting view");
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
            frosted_glass: false,
        };
        paths.begin(id, scene, &desc, lighting);
        let traced = Stages::COMPUTE | Stages::RAY;
        list.barrier(Stages::ALL, traced, Hazards::empty());
        kernels.record_media(list, scene, paths, &pass, lighting)?;
        kernels.record_opaque(list, scene, paths, &pass, lighting)?;
        list.barrier(traced, traced, Hazards::empty());
        kernels.record_dielectric(list, paths, &pass, lighting);
        state.samples = state.samples.wrapping_add(1);
        list.barrier(traced, Stages::COMPUTE, Hazards::empty());
        kernels.record_media_apply(list, scene, paths, &pass, lighting);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        let mut overlay = None;
        if let Some(hook) = hooks.transparency.as_mut() {
            let particles = paths.particle_lighting(id);
            let mut hook = |pass: &mut TransparencyPass<'_, '_>| hook(pass, particles);
            if reconstructed {
                overlay = Some(self.scenes.transparency_overlay(&raster, &mut hook)?);
            } else {
                let store = paths.radiance_storage();
                self.scenes.transparency(&raster, store, &mut hook)?;
            }
        }
        if reconstructed || matches!(lighting.debug_view, LightingDebugView::Guide(_)) {
            let list = self.scenes.list()?;
            kernels.record_guides(list, paths, &pass, lighting);
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        }
        let (list, arena) = self.scenes.output_list_and_arena(view.history)?;
        let (rr_output, rr_output_store) = match state.reconstruction.as_mut() {
            Some(r) => {
                arena.enter(ViewStep::Rr, list);
                let target = arena.texture(RECONSTRUCTED).expect("requested above");
                list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
                list.begin_pass("ray_reconstruction", false);
                let recorded = r.feature.record(
                    list,
                    &RrInputs {
                        color: paths.radiance(),
                        color_before_transparency: None,
                        depth: paths.depth(),
                        motion: paths.motion(),
                        diffuse_albedo: paths.diffuse_albedo(),
                        specular_albedo: paths.specular_albedo(),
                        normal_roughness: paths.normal_roughness(),
                        specular_motion: paths.specular_motion(),
                        transparency: overlay.as_deref(),
                        depth_of_field_guide: None,
                        output: target.texture(),
                        reset: std::mem::take(&mut r.fresh) || paths.history_reset(),
                        jitter,
                    },
                );
                list.end_pass();
                recorded.map_err(OutputError::from)?;
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
            frame.frame_generation(&desc, depth, motion, output.not_rendering_scene);
        }
        let distortion = output.lens_distortion(view.lens.distortion);
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
        if let Some(ScreenshotRequest::SceneReferred { rec709 }) = output.screenshot {
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
                camera_cut: desc.camera_cut,
                frame: id,
                output,
            },
        )?;
        // A view that stops after exposure shows no camera effects and holds none of their images.
        let look = lighting.debug_view == LightingDebugView::None;
        // Depth of field takes this rule too once it runs here: the reference traces its own lens.
        let blurs = tonemapped && look && lighting.reference.is_none();
        kernels.motion_blur(
            list,
            &mut state.output,
            &MotionBlurPass {
                rr_depth,
                rr_motion,
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
                sensor_width_mm: view.lens.sensor_width_mm,
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
                output,
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
        let raster = self.scenes.raster_view_with(&desc, &requests)?;
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
        if let Some(ScreenshotRequest::SceneReferred { rec709 }) = output.screenshot {
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
                    output,
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
        if let Some(mut frame) = self.display.as_deref_mut().and_then(Display::frame) {
            frame.frame_generation(view, depth, motion, not_rendering_scene);
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
    output: &'x ViewOutput,
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
        output,
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
    if let Some(ScreenshotRequest::Display { include_ui }) = output.screenshot {
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
