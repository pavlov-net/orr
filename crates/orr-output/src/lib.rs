//! Display output for orr: exposure, tonemapping, the display target and presentation.

mod display;
#[cfg(windows)]
mod display_watch;

pub use display::{
    CalibrationPattern, DisplayFeed, DisplayInfo, DisplayRequest, PatternLevels, ResolvedDisplay,
};
#[cfg(windows)]
pub use display_watch::DisplayWatch;
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
