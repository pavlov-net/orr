//! Display output for orr: exposure, tonemapping, the display target and presentation.

mod display;
#[cfg(windows)]
mod display_watch;
mod encode;
mod exposure;
mod gt7;

pub use display::{
    CalibrationPattern, DisplayFeed, DisplayInfo, DisplayRequest, PatternLevels, ResolvedDisplay,
};
#[cfg(windows)]
pub use display_watch::DisplayWatch;
pub use encode::{pq_decode, pq_encode, srgb_decode, srgb_encode};
pub use exposure::{
    MID_GREY, ev100_from_exposure, ev100_from_luminance, exposure_from_ev100, luminance_from_ev100,
};
pub use gt7::{GT7_REFERENCE_NITS, GT7_SDR_NITS, Gt7Curve, Gt7Params};
pub use orr_scene::orr_gpu::{HdrMetadata, OutputFormat, PresentMode};

/// Errors from orr-output.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the window handle is not a Win32 window")]
    NotWin32Window,
    #[cfg(windows)]
    #[error("display watch: {0}")]
    Windows(#[from] windows_core::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
