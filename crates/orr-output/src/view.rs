//! What output does with one view each frame.

use crate::{AutoWhiteBalance, CalibrationPattern, Exposure, Grade, Gt7Params, ScreenshotRequest};

/// Output's settings for one view, passed with each render of it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewOutput {
    pub exposure: Exposure,
    /// Off by default.
    pub white_balance: Option<AutoWhiteBalance>,
    pub grade: Grade,
    pub gt7: Gt7Params,
    /// A hook warps the image: the view's hooks get the display's distortion field, which
    /// frame generation reads.
    pub distortion_field: bool,
    /// A camera cut: auto exposure snaps to the target and resets its long-term stage, and
    /// auto white balance snaps to its target.
    pub cut: bool,
    /// A capture of this frame, read back through `OutputView::screenshots`.
    pub screenshot: Option<ScreenshotRequest>,
    pub debug: OutputDebug,
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
