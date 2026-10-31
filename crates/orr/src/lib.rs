//! orr: a ray-tracing-first, deferred renderer.
//!
//! A [`Renderer`] holds global asset state and any number of scenes. Each frame runs in a
//! fixed order on one render thread:
//!
//! 1. [`Renderer::begin_frame`] waits until the frame `frames_in_flight` back has
//!    completed and opens the next one.
//! 2. The consumer writes scene changes through [`Frame::scene`] and [`Frame::materials`],
//!    from any threads, straight into mapped upload memory.
//! 3. [`Frame::update_scenes`] applies every write and runs the consumer's scene compute.
//! 4. [`Frame::render_view`] once per view records its raster half,
//! 5. then the view's lighting and output: the scene's lighting pass before its first view,
//!    one path per pixel, the hooks, exposure, tonemap, the UI and compose. A consumer may
//!    instead shade the G-buffer [`Frame::raster_view`] returns with its own kernels in
//!    [`Frame::list`] and show the result with [`Frame::compose`].
//! 6. [`Frame::end`] records the display's frame generation, submits the frame and hands
//!    it to the display's present thread.
//!
//! There is no scheduler, render graph or per-frame extraction: each step records directly
//! into the frame's command list. A frame may list no scenes and render no view;
//! [`Frame::ui_only`] then presents its UI alone, over black or over a calibration pattern
//! from [`Frame::pattern`].

use std::collections::HashMap;
use std::sync::Arc;

pub use orr_lighting::{
    self, Error as LightingError, LightingDebugView, LightingStats, MediaSettings, PathEntry,
    ReferenceSettings, SceneLightingSettings, ThinLens, ViewLighting, WorldCacheSettings,
};
#[cfg(windows)]
pub use orr_output::DisplayWatch;
pub use orr_output::{self, Error as OutputError};
pub use orr_output::{
    AutoExposure, AutoWhiteBalance, CalibrationPattern, CompensationCurve, Display, DisplayFeed,
    DisplayInfo, DisplayRequest, DlssSdk, DlssSupport, Exposure, ExposureTelemetry,
    FrameGenerationMode, Grade, Gt7Params, HookPass, MeterWeights, OutputDebug, Physiological,
    Presented, Primaries, ResolvedDisplay, Screenshot, ScreenshotRequest, SkyTelemetry, UiPass,
    ViewOutput,
};
pub use orr_scene::Error as SceneError;
pub use orr_scene::{self, RasterView};
pub use orr_scene::{
    Affine3, ArenaReport, ArenaTarget, BoundTexture, Budgets, ClassStats, ClipRef, ClusterRef,
    ClusterTopology, Cone, ConsumerMarker, CopyVariant, DepthOfField, DirectionalLight,
    DiscProfile, EmissiveGpu, EmissiveSpan, ExposureState, Focus, FrameStats, FrameTime, Glare,
    GroupGpu, GroupWriter, InstanceFlags, InstanceGpu, InstancePage, InstanceRange, InstancesDesc,
    Lens, LensDistortion, Light, LightGpu, LightId, LoadState, Loading, MaterialGpu, MaterialId,
    MaterialWriter, MeshGpu, ModelEmission, ModelGpu, ModelId, MountId, Opacity, OriginPolicy,
    PackSource, PageId, PipelineId, Pose, ProducerBatch, ProducerClusterGpu, ProducerDesc,
    ProducerGpu, ProducerId, ProducerStats, ProducerUpdate, Projection, REBASE_GRID_LOG2,
    RasterGpu, ReloadReport, RendererDesc, ResidencyClass, ResidencyStats, RtCutGpu, RtStats,
    SceneCompute, SceneDesc, SceneId, SceneRoot, SceneWriter, SkeletonRef, SkinStats, SkinnedDesc,
    SkyStats, SlotStateGpu, Span, TextureFilter, TopologyGpu, TransformGroupDesc, TransformGroupId,
    TransparencyPass, ViewConstants, ViewDesc, ViewHistoryId, ViewStep, ViewTarget, WorldPos, abi,
    material_flags, orr_gpu, orr_pack,
};

use orr_gpu::{
    CommandList, Device, Format, GpuPtr, Hazards, KernelSet, OutputFormat, QueueKind, ShaderModule,
    Stages, StorageImageIndex, TextureDesc, TextureIndex, TextureUsage, TimelinePoint,
};
use orr_lighting::{BsdfTables, LightingKernels, LightingView, SceneLighting, ScenePass, ViewPass};
use orr_output::{
    CapturePass, ChainImages, ComposePass, ComposeRoot, ComposeSource, DisplayFrame, ExposurePass,
    HookInputsPass, HookPoint, Hooks, Image, OutputKernels, OutputView, SceneCapturePass,
    TonemapPass, UI_PASS,
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
    /// The view renders at another extent than its target, which needs ray reconstruction
    /// to upscale.
    #[error("render extent {render:?} differs from the output extent {output:?}")]
    Upscaling { render: [u32; 2], output: [u32; 2] },
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

/// The consumer passes a view runs, each once per `render_view` that sets it.
#[derive(Default)]
pub struct ViewHooks<'a> {
    /// Into the view's transparency overlay after its opaque lighting, at the render extent;
    /// the overlay is composited over the radiance before `scene_referred`. Particle
    /// producer entries are drawn here or not at all for primary visibility.
    pub transparency: Option<&'a mut dyn FnMut(&mut TransparencyPass<'_, '_>)>,
    /// On the view's pre-exposed radiance before metering, at the output extent.
    pub scene_referred: Option<&'a mut dyn FnMut(&mut HookPass<'_, '_>)>,
    /// On tonemap's display-linear image before compose; not run for `ViewTarget::AfterRr`.
    pub display_linear: Option<&'a mut dyn FnMut(&mut HookPass<'_, '_>)>,
    /// Into the display's UI layer; run only for `ViewTarget::Display`. With
    /// `Display::set_ui_per_present` on and frame generation generating, it runs once per
    /// presented image. Each run records inside a pass named [`orr_output::UI_PASS`], so it
    /// begins no pass of its own.
    pub ui: Option<&'a mut dyn FnMut(&mut UiPass<'_, '_>)>,
}

/// What a view's `render_view` wrote. `radiance` and `linear` stay valid until the frame's
/// GPU work completes; of `raster` only depth, motion and the HZB do (see [`RasterView`]).
#[derive(Clone)]
pub struct RenderedView {
    pub raster: RasterView,
    /// RGBA16F linear Rec.2020 pre-exposed radiance at the render extent, as lighting and
    /// `hooks.scene_referred` left it. With a `display_linear` hook it holds that hook's
    /// source instead.
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

/// orr-lighting's kernels and tables, and each scene's lighting state.
struct Lighting {
    kernels: &'static KernelSet,
    paths: LightingKernels,
    scenes: HashMap<SceneId, SceneLighting>,
    _tables: BsdfTables,
}

/// One view history's lighting and output state.
struct ViewState {
    lighting: Option<LightingView>,
    output: OutputView,
    /// Compose roots per frame in flight.
    compose: Vec<ComposeRoot>,
    /// Renders of the view so far: the sample index of its next paths.
    samples: u32,
}

/// orr-output's kernels, the views' output state and the state of frames that render no
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

/// The view's display-linear image at the output `extent`: tonemap's target, and from the
/// scene-referred hook on the hook chain's second image; the consumer may read it after the
/// render.
fn linear_request(extent: [u32; 2]) -> ArenaRequest {
    ArenaRequest::texture(
        VIEW_LINEAR,
        TextureDesc::d2(
            Format::Rgba16Float,
            extent[0],
            extent[1],
            TextureUsage::SAMPLED | TextureUsage::STORAGE,
        ),
        Span::new(ViewStep::SceneReferred, ViewStep::After),
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
    /// Render queue.
    pub fn new(gpu: &Device, desc: RendererDesc) -> Result<Self> {
        desc.kernels
            .check(&builtin_modules())
            .map_err(SceneError::from)?;
        let render = gpu.queue(QueueKind::Render).expect("render queue");
        let mut list = render.command_list()?;
        let tables = BsdfTables::new(gpu, &mut list)?;
        render.submit(vec![list], &[])?;
        Ok(Self {
            scenes: Scenes::new(gpu, &desc)?,
            lighting: Lighting {
                kernels: desc.kernels,
                paths: LightingKernels::new(gpu, desc.kernels, &tables, PathEntry::Fastest)?,
                scenes: HashMap::new(),
                _tables: tables,
            },
            output: Output::new(gpu, desc.frames_in_flight)?,
        })
    }

    pub fn device(&self) -> &Device {
        self.scenes.device()
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
    pub fn create_view_history(&mut self) -> Result<ViewHistoryId> {
        let gpu = self.scenes.device().clone();
        let fif = self.output.frames_in_flight;
        let state = ViewState {
            lighting: None,
            output: OutputView::new(&gpu, fif)?,
            compose: compose_roots(&gpu, fif)?,
            samples: 0,
        };
        let id = self.scenes.create_view_history()?;
        self.output.views.insert(id, state);
        Ok(id)
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
    pub fn raster_view(&mut self, view: &ViewDesc<'_>) -> Result<RasterView> {
        Ok(self.scenes.raster_view(view)?)
    }

    /// Steps 4-5 for one view: its raster half; the scene's lighting pass before the scene's
    /// first view this frame; one path per pixel into the view's radiance;
    /// `hooks.scene_referred`; a scene-referred screenshot; metering and adaptation; then,
    /// unless the view stops there, tonemap, `hooks.display_linear`, `hooks.ui`, compose and
    /// a display screenshot.
    ///
    /// The target decides the output: the display's backbuffer at its window's extent; a
    /// texture with storage usage (RGBA8 or RGBA16F), written SDR-encoded with paper white
    /// at 250 nits; or nothing past exposure for `AfterRr`, and for `Display` without a
    /// display or while its window has no area. The render extent must equal the output
    /// extent.
    pub fn render_view(
        &mut self,
        view: &ViewDesc<'_>,
        lighting: &ViewLighting,
        output: &ViewOutput,
        hooks: &mut ViewHooks<'_>,
    ) -> Result<RenderedView> {
        let render = view.render_extent.to_array();
        let sdr = DisplayRequest::default().resolve(&[OutputFormat::Sdr], None, false);
        let mut display = match view.target {
            ViewTarget::Display => self.display.as_deref_mut().and_then(Display::frame),
            _ => None,
        };
        let target = match (view.target, display.as_ref()) {
            (ViewTarget::Display, Some(f)) => Some((f.display, f.extent)),
            (ViewTarget::Texture(t), _) => {
                let [w, h, _] = t.desc().extent;
                Some((sdr, [w, h]))
            }
            _ => None,
        };
        let tonemapped = target.is_some();
        let (resolved, extent) = target.unwrap_or((sdr, render));
        if extent != render {
            return Err(Error::Upscaling {
                render,
                output: extent,
            });
        }

        let set = Hooks {
            scene_referred: hooks.scene_referred.is_some(),
            display_linear: tonemapped && hooks.display_linear.is_some(),
        };
        let mut requests = LightingView::requests(render);
        requests.push(linear_request(extent));
        if set.any() {
            requests.extend(orr_output::hook_requests(extent));
        }
        let raster = self.scenes.raster_view_with(view, &requests)?;
        let id = self.scenes.id();
        let delta = self.scenes.time().delta;
        let gpu = self.output.device.clone();
        let (list, arena) = self.scenes.list_and_arena(view.history)?;
        let state = self
            .output
            .views
            .get_mut(&view.history)
            .ok_or(SceneError::UnknownViewHistory(view.history))?;
        arena.enter(ViewStep::Lighting, list);

        let lights = &mut *self.lighting;
        let scene = lights
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
        list.barrier(Stages::ALL, Stages::COMPUTE | Stages::RAY, Hazards::empty());
        lights.paths.record_opaque(
            list,
            scene,
            paths,
            &ViewPass {
                frame: id,
                scene: raster.scene,
                view: raster.view,
                gbuffer: raster.gbuffer,
                sample_index: state.samples,
                seed: 0,
            },
            lighting,
        );
        state.samples = state.samples.wrapping_add(1);
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::COMPUTE,
            Hazards::empty(),
        );
        let (radiance, radiance_store) = (paths.radiance_index(), paths.radiance_storage());
        if let Some(hook) = hooks.transparency.as_mut() {
            self.scenes
                .transparency(&raster, radiance_store, &mut **hook)?;
        }
        let (list, arena) = self.scenes.list_and_arena(view.history)?;
        arena.enter(ViewStep::SceneReferred, list);
        let linear = arena.texture(VIEW_LINEAR).expect("requested above").clone();
        let chain = ChainImages {
            rr_output: radiance,
            rr_output_store: radiance_store,
            linear: linear.sampled(),
            linear_store: linear.storage(0),
        };
        if let Some(frame) = display.as_mut() {
            frame.frame_generation(
                view,
                &raster.depth,
                &raster.motion,
                output.not_rendering_scene,
            );
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
                rr_depth: raster.targets.depth,
                rr_motion: raster.targets.motion,
                chain,
                extent,
                view: raster.view,
                jitter: view.jitter.to_array(),
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
                    source: radiance,
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
            state.output.roots(id),
            &ExposurePass {
                source: radiance,
                extent,
                view: raster.view,
                sky: raster.sky_stats,
                frame: id,
                output,
            },
        );
        if !tonemapped {
            arena.enter(ViewStep::After, list);
            return Ok(RenderedView {
                raster,
                radiance,
                linear: TextureIndex::NONE,
            });
        }

        arena.enter(ViewStep::Tonemap, list);
        kernels.tonemap(
            list,
            state.output.roots(id),
            &TonemapPass {
                source: radiance,
                target: chain.linear_store,
                extent,
                view: raster.view,
                display: &resolved,
                output,
                hook_source: set.display_linear.then_some(radiance_store),
                glare: None,
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
        arena.enter(ViewStep::Compose, list);
        let ui = match (display.as_mut(), hooks.ui.as_mut()) {
            (Some(frame), Some(hook)) => {
                let index = paint_ui(list, frame, delta, &mut **hook);
                list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
                Some(index)
            }
            _ => None,
        };
        let store = match (view.target, display.as_mut()) {
            (ViewTarget::Texture(t), _) => {
                let slot = gpu.alloc_image_slot()?;
                self.slots.push(slot);
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
                source: ComposeSource::Linear(chain.linear),
                ui: None,
                target: store,
                extent,
                display: &resolved,
            },
        );
        if let Some(ScreenshotRequest::Display { include_ui }) = output.screenshot {
            kernels.capture(
                list,
                &mut state.output,
                &CapturePass {
                    source: ComposeSource::Linear(chain.linear),
                    ui: ui.filter(|_| include_ui),
                    extent,
                    display: &resolved,
                    frame: id,
                },
            )?;
        }
        arena.enter(ViewStep::After, list);
        Ok(RenderedView {
            raster,
            radiance,
            linear: chain.linear,
        })
    }

    /// The frame's command list, for the consumer's own passes. Everything recorded into it
    /// runs in order with orr's passes and is submitted by [`Frame::end`]. The first call
    /// runs [`Frame::update_scenes`] if the frame has not.
    pub fn list(&mut self) -> Result<&mut CommandList<'a>> {
        Ok(self.scenes.list()?)
    }

    /// Runs `ui` into the display's UI layer. In a frame that renders no view into the
    /// display (menus, loading screens) the present thread composes it over black, or over
    /// the frame's [`Frame::pattern`]; after [`Frame::compose`] and
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
        let list = self.scenes.list()?;
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
        view: &ViewDesc<'_>,
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
    /// Records a barrier after every earlier pass, so the image may be written just
    /// before. Does nothing without a display or while its window has no area.
    pub fn compose(&mut self, linear: TextureIndex) -> Result<()> {
        let Some(display) = self.display.as_deref_mut() else {
            return Ok(());
        };
        let id = self.scenes.id();
        let list = self.scenes.list()?;
        let Some(mut frame) = display.frame() else {
            return Ok(());
        };
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.output.compose(list, id, &mut frame, linear);
        Ok(())
    }

    /// Step 6: records the display's frame generation, submits the frame after the
    /// display's slot is free, hands it to the present thread without waiting, and returns
    /// the render-timeline point its work signals.
    pub fn end(mut self) -> Result<TimelinePoint> {
        let wait = self
            .display
            .as_deref_mut()
            .and_then(Display::frame)
            .and_then(|f| f.wait);
        if let Some(d) = self.display.as_deref_mut() {
            d.record_frame_generation(self.scenes.list()?)?;
        }
        let gpu = self.output.device.clone();
        let slots = std::mem::take(&mut self.slots);
        let ended = self.scenes.end(wait.as_slice());
        for slot in slots {
            match &ended {
                Ok(point) => gpu.free_image_slot(slot, *point),
                Err(_) => gpu.free_image_slot_after_submitted(slot),
            }
        }
        let point = ended?;
        if let Some(d) = self.display {
            d.present(point)?;
        }
        Ok(point)
    }
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
