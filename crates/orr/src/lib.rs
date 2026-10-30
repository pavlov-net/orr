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
//! 5. then the view's lighting and output.
//! 6. [`Frame::end`] submits the frame and hands it to the display's present thread.
//!
//! There is no scheduler, render graph or per-frame extraction: each step records directly
//! into the frame's command list. A frame may list no scenes and render no view;
//! [`Frame::ui_only`] then presents its UI alone.

pub use orr_output::{self, Error as OutputError};
pub use orr_output::{
    Display, DisplayFeed, DisplayInfo, DisplayRequest, Presented, ResolvedDisplay, UiPass,
};
pub use orr_scene::Error as SceneError;
pub use orr_scene::{
    Affine3, Budgets, Cone, ConsumerMarker, CopyVariant, DepthOfField, DirectionalLight,
    DiscProfile, ExposureState, Focus, FrameStats, FrameTime, Glare, GroupGpu, GroupWriter,
    InstanceFlags, InstanceGpu, InstanceRange, InstancesDesc, Lens, LensDistortion, Light,
    LightGpu, LightId, MaterialGpu, MaterialId, MaterialWriter, ModelId, Pose, Projection,
    REBASE_GRID_LOG2, RendererDesc, SceneCompute, SceneDesc, SceneId, SceneRoot, SceneWriter,
    SkyStats, SlotStateGpu, TransformGroupDesc, TransformGroupId, ViewConstants, ViewDesc,
    ViewHistoryId, ViewTarget, WorldPos, abi, material_flags, orr_gpu, orr_pack,
};

use orr_gpu::{Device, ShaderModule, TimelinePoint};
use orr_scene::Scenes;

/// A scene or output failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Scene(#[from] SceneError),
    #[error(transparent)]
    Output(#[from] OutputError),
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

/// The renderer: scenes, their views' histories and the frame.
pub struct Renderer {
    scenes: Scenes,
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
    /// pipelines on the calling thread.
    pub fn new(gpu: &Device, desc: RendererDesc) -> Result<Self> {
        desc.kernels
            .check(&builtin_modules())
            .map_err(SceneError::from)?;
        Ok(Self {
            scenes: Scenes::new(gpu, &desc)?,
        })
    }

    pub fn device(&self) -> &Device {
        self.scenes.device()
    }

    pub fn create_scene(&mut self, desc: &SceneDesc) -> Result<SceneId> {
        Ok(self.scenes.create_scene(desc)?)
    }

    /// Destroys a scene; its memory is freed once every submitted frame has completed.
    pub fn destroy_scene(&mut self, scene: SceneId) {
        self.scenes.destroy_scene(scene);
    }

    /// History for one persistent camera: its previous camera and exposure slots.
    pub fn create_view_history(&mut self) -> Result<ViewHistoryId> {
        Ok(self.scenes.create_view_history()?)
    }

    pub fn destroy_view_history(&mut self, id: ViewHistoryId) {
        self.scenes.destroy_view_history(id);
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
        Ok(Frame { scenes, display })
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
}

impl<'a> Frame<'a> {
    /// orr's frame id: strictly increasing from 1.
    pub fn id(&self) -> u64 {
        self.scenes.id()
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

    /// Steps 4-5 for one view.
    pub fn render_view(&mut self, view: &ViewDesc<'_>) -> Result<()> {
        self.scenes.raster_view(view)?;
        Ok(())
    }

    /// A frame that renders no view into the display (menus, loading screens): runs `ui`
    /// into the display's UI layer, which the present thread composes over black. Does
    /// nothing without a display or while its window has no area.
    pub fn ui_only(&mut self, ui: &mut dyn FnMut(&mut UiPass<'_, '_>)) -> Result<()> {
        let Some(display) = self.display.as_deref_mut() else {
            return Ok(());
        };
        let list = self.scenes.list()?;
        let Some(mut frame) = display.frame() else {
            return Ok(());
        };
        let layer = frame.ui_layer(list).texture();
        ui(&mut UiPass {
            cmd: list,
            layer,
            presented: 0,
            time_offset: 0.0,
        });
        Ok(())
    }

    /// Step 6: submits the frame after the display's slot is free, hands it to the present
    /// thread without waiting, and returns the render-timeline point its work signals.
    pub fn end(mut self) -> Result<TimelinePoint> {
        let wait = self
            .display
            .as_deref_mut()
            .and_then(Display::frame)
            .and_then(|f| f.wait);
        let point = self.scenes.end(wait.as_slice())?;
        if let Some(d) = self.display {
            d.present(point)?;
        }
        Ok(point)
    }
}
