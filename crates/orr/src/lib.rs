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
//! 6. [`Frame::end`] submits the frame.
//!
//! There is no scheduler, render graph or per-frame extraction: each step records directly
//! into the frame's command list. A frame may list no scenes and render no view.

pub use orr_scene::{
    Affine3, Budgets, Cone, CopyVariant, DepthOfField, DirectionalLight, DiscProfile, Error,
    ExposureState, Focus, FrameDesc, FrameStats, FrameTime, Glare, GroupGpu, GroupWriter,
    InstanceFlags, InstanceGpu, InstanceRange, InstancesDesc, Lens, LensDistortion, Light,
    LightGpu, LightId, MaterialGpu, MaterialId, MaterialWriter, ModelId, Pose, Projection,
    REBASE_GRID_LOG2, RendererDesc, Result, SceneCompute, SceneDesc, SceneId, SceneRoot,
    SceneWriter, SkyStats, SlotStateGpu, TransformGroupDesc, TransformGroupId, ViewConstants,
    ViewDesc, ViewHistoryId, ViewTarget, WorldPos, abi, material_flags, orr_gpu, orr_pack,
};

use orr_gpu::{Device, ShaderModule, TimelinePoint};
use orr_scene::Scenes;

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
        desc.kernels.check(&builtin_modules())?;
        Ok(Self {
            scenes: Scenes::new(gpu, &desc)?,
        })
    }

    pub fn device(&self) -> &Device {
        self.scenes.device()
    }

    pub fn create_scene(&mut self, desc: &SceneDesc) -> Result<SceneId> {
        self.scenes.create_scene(desc)
    }

    /// Destroys a scene; its memory is freed once every submitted frame has completed.
    pub fn destroy_scene(&mut self, scene: SceneId) {
        self.scenes.destroy_scene(scene);
    }

    /// History for one persistent camera: its previous camera and exposure slots.
    pub fn create_view_history(&mut self) -> Result<ViewHistoryId> {
        self.scenes.create_view_history()
    }

    pub fn destroy_view_history(&mut self, id: ViewHistoryId) {
        self.scenes.destroy_view_history(id);
    }

    /// Step 1: waits until the frame `frames_in_flight` back has completed and opens the
    /// next one.
    pub fn begin_frame(&mut self, desc: &FrameDesc<'_>) -> Result<Frame<'_>> {
        Ok(Frame {
            scenes: self.scenes.begin_frame(desc)?,
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
}

impl<'a> Frame<'a> {
    /// orr's frame id: strictly increasing from 1.
    pub fn id(&self) -> u64 {
        self.scenes.id()
    }

    /// Step 2: the writer of a scene listed in this frame. One per scene at a time; Send.
    pub fn scene(&self, id: SceneId) -> Result<SceneWriter<'_>> {
        self.scenes.scene(id)
    }

    /// Step 2: a material writer; Send.
    pub fn materials(&self) -> MaterialWriter<'_> {
        self.scenes.materials()
    }

    /// Step 3: applies every write, then hands each listed scene in order to `compute`.
    pub fn update_scenes(
        &mut self,
        compute: &mut dyn FnMut(&mut SceneCompute<'_, 'a>),
    ) -> Result<()> {
        self.scenes.update_scenes(compute)
    }

    /// Steps 4-5 for one view.
    pub fn render_view(&mut self, view: &ViewDesc<'_>) -> Result<()> {
        self.scenes.raster_view(view)?;
        Ok(())
    }

    /// Step 6: submits the frame; returns the render-timeline point its work signals.
    pub fn end(self) -> Result<TimelinePoint> {
        self.scenes.end(&[])
    }
}
