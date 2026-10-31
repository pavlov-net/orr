//! Display output for orr: exposure, white balance, tonemapping, the display target and
//! presentation.

mod display;
#[cfg(windows)]
mod display_watch;
mod dof;
mod encode;
mod exposure;
mod glare;
mod gt7;
mod hooks;
mod motion_blur;
mod passes;
mod present;
mod screenshot;
mod ui;
mod view;
mod white_balance;

mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

pub use display::{
    CalibrationPattern, DisplayFeed, DisplayInfo, DisplayRequest, PatternLevels, ResolvedDisplay,
};
#[cfg(windows)]
pub use display_watch::DisplayWatch;
pub use dof::{DepthOfFieldPass, DofGuideImages, DofPath};
pub use encode::{pq_decode, pq_encode, srgb_decode, srgb_encode};
pub use exposure::{
    AutoExposure, CompensationCurve, Exposure, ExposureTelemetry, Grade, MID_GREY, MeterWeights,
    Physiological, SkyTelemetry, ev100_from_exposure, ev100_from_luminance, exposure_from_ev100,
    keyed_ev100, luminance_from_ev100,
};
pub use glare::{GlareComposite, GlarePass};
pub use gt7::{GT7_REFERENCE_NITS, GT7_SDR_NITS, Gt7Curve, Gt7Params};
pub use hooks::{ChainImages, DistortionField, HookInputs, HookPass, HookPoint, Hooks, Image};
pub use motion_blur::{MOTION_BLUR_MAX_RADIUS, MotionBlurPass};
pub use orr_scene::orr_gpu::{HdrMetadata, LowLatency, OutputFormat, PresentMode};
pub use passes::{
    CapturePass, ComposePass, ComposeRoot, ComposeSource, ExposurePass, HookInputsPass,
    OutputKernels, OutputRoots, OutputView, SceneCapturePass, TonemapPass,
};
pub use present::{Display, DisplayFrame, Presented};
pub use screenshot::{Primaries, Screenshot, ScreenshotRequest};
pub use ui::{UiLayer, UiPass};
pub use view::{OutputDebug, PostQuality, ViewOutput};
pub use white_balance::{
    AutoWhiteBalance, D65_XY, WHITE_BALANCE_CCT, WHITE_BALANCE_DUV, cct_duv_from_xy,
    xy_from_cct_duv,
};

/// Errors from orr-output.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Gpu(#[from] orr_scene::orr_gpu::Error),
    #[error("the window handle is not a Win32 window")]
    NotWin32Window,
    #[error("present thread: {0}")]
    Thread(String),
    #[cfg(windows)]
    #[error("display watch: {0}")]
    Windows(#[from] windows_core::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
