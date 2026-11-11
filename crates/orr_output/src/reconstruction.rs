//! Ray reconstruction apart from what runs it: what a view asks for, what the renderer reads back
//! and the two traits a backend implements.

use orr_scene::ViewDesc;
use orr_scene::orr_gpu::{CommandList, Texture};

use crate::{Error, Result};

/// How a lit view goes through ray reconstruction when the renderer has it. Without it, or under
/// a lighting debug view or the accumulation reference, the view renders one noisy path per
/// pixel at the output extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reconstruction {
    /// No ray reconstruction: the view renders at the output extent.
    Off,
    /// Deep Learning Anti-Aliasing: DLSS's best and most expensive mode, which renders at the
    /// output extent. Only this request runs it, and a backend without it refuses it.
    Dlaa,
    /// At the render size the backend gives this quality for the output extent.
    Upscale(UpscaleQuality),
    /// At this render size, which keeps the output's aspect ratio. A backend may refuse sizes it
    /// doesn't run at; [`ReconstructionSupport::scale`] gives the range where it reports one.
    RenderExtent([u32; 2]),
}

impl Default for Reconstruction {
    fn default() -> Self {
        Self::Upscale(UpscaleQuality::default())
    }
}

/// How far below its output a view renders: a named mode, or the one that suits the output
/// resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UpscaleQuality {
    /// The backend's choice for the output resolution.
    #[default]
    Auto,
    Fixed(UpscaleMode),
}

/// A named ratio of output to render resolution, which the backend turns into a render size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UpscaleMode {
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

/// What a view's ray reconstruction runs as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReconstructionMode {
    Dlaa,
    Upscale(UpscaleMode),
    /// From a backend without named modes, at the view's render extent.
    RenderExtent,
}

/// What a backend offers on its device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructionSupport {
    pub available: bool,
    /// It runs [`Reconstruction::Dlaa`].
    pub dlaa: bool,
    /// The smallest output width and height it takes. A smaller view renders without it.
    pub min_output: [u32; 2],
    /// The smallest and largest output over render resolution it takes, where it reports them.
    pub scale: Option<[f32; 2]>,
}

/// Makes ray reconstruction features on one device.
pub trait ReconstructionBackend: Send {
    /// Read once, when the backend is given to the renderer.
    fn support(&self) -> ReconstructionSupport;

    /// A feature for `output` at the render size of `quality`. May wait for setup on the Render
    /// queue, so call it on that queue's owner thread.
    fn feature(
        &self,
        output: [u32; 2],
        quality: UpscaleQuality,
    ) -> Result<Box<dyn ReconstructionFeature>>;

    /// [`feature`](Self::feature) at the render size `render`.
    fn feature_at(
        &self,
        render: [u32; 2],
        output: [u32; 2],
    ) -> Result<Box<dyn ReconstructionFeature>>;

    /// [`feature`](Self::feature) for [`Reconstruction::Dlaa`], from a backend whose
    /// [`support`](Self::support) has `dlaa`.
    fn dlaa(&self, _output: [u32; 2]) -> Result<Box<dyn ReconstructionFeature>> {
        Err(Error::DlaaUnavailable)
    }
}

/// One view's ray reconstruction at a fixed render and output size. Dropping it releases its
/// resources once everything submitted before the drop has completed.
pub trait ReconstructionFeature: Send {
    /// What it runs as. A backend with named modes reports the one it runs in, which for a
    /// render size given to [`ReconstructionBackend::feature_at`] may be the nearest.
    fn mode(&self) -> ReconstructionMode;

    /// Render resolution, fixed for the feature's lifetime.
    fn render_size(&self) -> [u32; 2];

    fn output_size(&self) -> [u32; 2];

    /// The jitter offset for `frame`, in render pixels, each axis in [-0.5, 0.5). The projection
    /// places each pixel's sample at the pixel center minus this value on both axes, with y
    /// pointing down.
    fn jitter(&self, frame: u64) -> [f32; 2];

    /// Texture LOD bias for the render resolution.
    fn mip_bias(&self) -> f32;

    /// Records the evaluation into `list`, on a compute-capable queue of the feature's device.
    /// The inputs' formats first show here, and stay the same for the feature's lifetime.
    fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &ReconstructionInputs<'_>,
    ) -> Result<()>;
}

/// Ray reconstruction's inputs at render resolution and its output at output resolution: all a
/// backend may use, of which it ignores the rest. Every texture is 2D with one mip and one layer,
/// and has sampled or storage usage.
#[derive(Clone, Copy)]
pub struct ReconstructionInputs<'a> {
    /// Pre-exposed HDR radiance.
    pub color: &'a Texture,
    /// Radiance before the transparency layer.
    pub color_before_transparency: Option<&'a Texture>,
    /// Reverse-Z device depth.
    pub depth: &'a Texture,
    /// Unjittered motion, previous UV minus current UV.
    pub motion: &'a Texture,
    pub diffuse_albedo: &'a Texture,
    pub specular_albedo: &'a Texture,
    /// World-space normal in RGB, linear roughness in A.
    pub normal_roughness: &'a Texture,
    /// Motion of reflected surfaces, in the same units as `motion`.
    pub specular_motion: &'a Texture,
    /// World-space distance from each primary surface to its specular hit, an alternative to
    /// `specular_motion`.
    pub specular_hit_distance: Option<&'a Texture>,
    /// Premultiplied color with coverage in alpha, composited after denoising.
    pub transparency: Option<&'a Texture>,
    pub depth_of_field_guide: Option<&'a Texture>,
    /// One channel: how strongly to denoise each pixel.
    pub denoise_strength_mask: Option<&'a Texture>,
    /// One channel in [0, 1]: how far to trust the current frame over the history, where motion
    /// is inaccurate.
    pub reactive_mask: Option<&'a Texture>,
    /// 1x1: the exposure that brings the color to the tonemapper's range.
    pub exposure: Option<&'a Texture>,
    pub output: &'a Texture,
    /// Drops the history.
    pub reset: bool,
    /// The view as it rendered: its camera, and its jitter from
    /// [`ReconstructionFeature::jitter`].
    pub view: &'a ViewDesc,
}
