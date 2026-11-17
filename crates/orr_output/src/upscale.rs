//! Temporal upscaling apart from what runs it, for views that are not path traced and so need
//! no denoising: the two traits a backend implements and what a feature reads. Ray
//! reconstruction ([`crate::reconstruction`]) is the denoising counterpart.

use orr_scene::orr_gpu::{CommandList, Texture};

use crate::{Error, ReconstructionMode, ReconstructionSupport, Result, UpscaleQuality};

/// Makes temporal upscaling features on one device.
pub trait UpscaleBackend: Send {
    /// Read once, when the backend is given to its user.
    fn upscale_support(&self) -> ReconstructionSupport;

    /// A feature for `output` at the render size of `quality`. May wait for setup on the Render
    /// queue, so call it on that queue's owner thread.
    fn upscaler(
        &self,
        output: [u32; 2],
        quality: UpscaleQuality,
    ) -> Result<Box<dyn UpscaleFeature>>;

    /// [`upscaler`](Self::upscaler) at the render size `render`.
    fn upscaler_at(&self, render: [u32; 2], output: [u32; 2]) -> Result<Box<dyn UpscaleFeature>>;

    /// [`upscaler`](Self::upscaler) at the output size: temporal antialiasing, from a backend
    /// whose [`upscale_support`](Self::upscale_support) has `dlaa`.
    fn antialiaser(&self, _output: [u32; 2]) -> Result<Box<dyn UpscaleFeature>> {
        Err(Error::DlaaUnavailable)
    }
}

/// One view's temporal upscaling at a fixed render and output size. Dropping it releases its
/// resources once everything submitted before the drop has completed.
pub trait UpscaleFeature: Send {
    /// What it runs as; for a render size given to [`UpscaleBackend::upscaler_at`], the nearest
    /// named mode where the backend has them.
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

    /// Records the upscale into `list`, on a compute-capable queue of the feature's device. The
    /// inputs' formats first show here, and stay the same for the feature's lifetime.
    fn record(&mut self, list: &mut CommandList<'_>, inputs: &UpscaleInputs<'_>) -> Result<()>;
}

/// Temporal upscaling's inputs at render resolution and its output at output resolution. Every
/// texture is 2D with one mip and one layer, and has sampled or storage usage.
#[derive(Clone, Copy)]
pub struct UpscaleInputs<'a> {
    /// HDR color, multiplied by `pre_exposure`.
    pub color: &'a Texture,
    /// Reverse-Z device depth.
    pub depth: &'a Texture,
    /// Unjittered motion, previous UV minus current UV.
    pub motion: &'a Texture,
    /// 1x1; its first channel is the exposure that brings `color / pre_exposure` to the range
    /// the tonemapper takes (mid gray near 0.18). `None` for color already exposed, such as
    /// display-linear color with 1.0 at paper white.
    pub exposure: Option<&'a Texture>,
    /// The factor `color` was multiplied by, which the backend divides out; 1 when it was not.
    pub pre_exposure: f32,
    pub output: &'a Texture,
    /// Drops the history.
    pub reset: bool,
    /// From [`UpscaleFeature::jitter`] for this frame.
    pub jitter: [f32; 2],
}
