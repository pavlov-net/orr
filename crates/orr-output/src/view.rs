//! What output does with one view each frame.

use crate::{CalibrationPattern, Exposure, Grade, Gt7Params};

/// Output's settings for one view, passed with each render of it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewOutput {
    pub exposure: Exposure,
    pub grade: Grade,
    pub gt7: Gt7Params,
    /// A camera cut: auto exposure snaps to the target and resets its long-term stage.
    pub cut: bool,
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
