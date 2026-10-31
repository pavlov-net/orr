//! What output does with one view each frame.

use orr_scene::LensDistortion;

use crate::{
    AutoWhiteBalance, CalibrationPattern, DofPath, Exposure, Grade, Gt7Params, ScreenshotRequest,
};

/// Output's settings for one view, passed with each render of it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewOutput {
    pub exposure: Exposure,
    /// Off by default.
    pub white_balance: Option<AutoWhiteBalance>,
    pub grade: Grade,
    pub gt7: Gt7Params,
    pub post: PostQuality,
    /// A hook warps the image: the view's hooks get the display's distortion field, which
    /// frame generation reads. A lens with distortion turns the field on too.
    pub distortion_field: bool,
    /// A camera cut: auto exposure snaps to the target and resets its long-term stage, and
    /// auto white balance snaps to its target.
    pub cut: bool,
    /// Menus and loading screens: frame generation is told the frame shows no scene.
    pub not_rendering_scene: bool,
    /// A capture of this frame, read back through `OutputView::screenshots`.
    pub screenshot: Option<ScreenshotRequest>,
    pub debug: OutputDebug,
}

impl ViewOutput {
    /// The part of the view's `lens` distortion that tonemap applies: none under a
    /// calibration pattern.
    pub fn lens_distortion(&self, lens: Option<LensDistortion>) -> Option<LensDistortion> {
        lens.filter(|_| !matches!(self.debug, OutputDebug::Pattern(_)))
    }
}

/// Quality settings of the camera effects that the view's lens turns on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostQuality {
    /// Samples per pixel along the blur, in symmetric pairs.
    pub motion_blur_samples: u32,
    /// Rings of the half-resolution depth-of-field gathers; ring k holds 8k samples.
    pub dof_rings: u32,
    pub dof_path: DofPath,
    pub glare: GlareMethod,
    /// Halvings to the resolution glare runs at: 1 (half resolution) or 2. For the bands it
    /// is the pyramid's first level.
    pub glare_first_level: u32,
}

/// How glare convolves the radiance with the diaphragm's PSF.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GlareMethod {
    /// A band pyramid with per-channel weights fitted to the PSF's energy in rings: radially
    /// symmetric, without the spikes.
    #[default]
    Bands,
    /// FFT convolution with the PSF itself: the blades' diffraction spikes and the far tail,
    /// out to a quarter of the image's longer side. Costlier in time and memory (about
    /// 100 MB at 4K).
    Fft,
}

impl Default for PostQuality {
    fn default() -> Self {
        Self {
            motion_blur_samples: 35,
            dof_rings: 5,
            dof_path: DofPath::AfterRr,
            glare: GlareMethod::Bands,
            glare_first_level: 1,
        }
    }
}

/// Debug views in place of the tonemapped image.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum OutputDebug {
    #[default]
    None,
    /// Exposed and graded radiance converted to Rec.709, without GT7.
    Linear,
    /// A calibration pattern in place of the view.
    Pattern(CalibrationPattern),
    /// Each pixel's exposed luminance in stops from mid grey: blue below, green at, red
    /// above, saturating at six stops.
    ExposureHeatmap,
}
