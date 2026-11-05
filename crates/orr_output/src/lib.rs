//! Display output for orr: exposure, local exposure, white balance, tonemapping, the display
//! target, frame generation and presentation.

mod display;
#[cfg(dlss)]
mod dlss;
mod dof;
mod exposure;
mod fft_glare;
mod frame_generation;
mod glare;
mod glare_tables;
mod gt7;
mod hooks;
mod local_exposure;
mod motion_blur;
mod overlay_front;
mod passes;
#[cfg(all(windows, feature = "pcl-stats"))]
mod pcl_stats;
mod present;
mod reconstruction;
mod screenshot;
mod ui;
mod view;
mod white_balance;

mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

pub use display::{CalibrationPattern, DisplayRequest, PatternLevels, ResolvedDisplay};
pub use dof::{DepthOfFieldPass, DofGuideImages, DofHoleMask, DofPath};
pub use exposure::{
    AutoExposure, CompensationCurve, Exposure, ExposureTelemetry, Grade, LocalExposure, MID_GRAY,
    MeterWeights, Physiological, SkyTelemetry, ev100_from_exposure, ev100_from_luminance,
    exposure_from_ev100, keyed_ev100, luminance_from_ev100,
};
pub use frame_generation::{
    FrameGenerationBackend, FrameGenerationCamera, FrameGenerationFeature, FrameGenerationInputs,
    FrameGenerationMode, InterpolationCheck,
};
pub use glare::{GlareComposite, GlarePass};
pub use gt7::{GT7_REFERENCE_NITS, Gt7Curve, Gt7Params};
pub use hooks::{
    ChainImages, DistortionField, HookInputs, HookPass, HookPoint, Hooks, Image, hook_requests,
};
pub use motion_blur::{MOTION_BLUR_MAX_RADIUS, MotionBlurPass};
pub use orr_color::Primaries;
#[cfg(dlss)]
pub use orr_dlss::{self, DlssSdk, Support as DlssSupport};
pub use orr_scene::orr_gpu::{LowLatency, OutputFormat};
pub use orr_window::{self, DisplayFeed, DisplayInfo, DisplayWatch};
pub use overlay_front::OverlayFront;
pub use passes::{
    CapturePass, ComposePass, ComposeRoot, ComposeSource, ExposurePass, HookInputsPass,
    OutputKernels, OutputRoots, OutputView, SceneCapturePass, TonemapPass,
};
pub use present::{Display, DisplayFrame, Presented, UI_PASS};
pub use reconstruction::{
    Reconstruction, ReconstructionBackend, ReconstructionFeature, ReconstructionInputs,
    ReconstructionSupport, UpscaleMode, UpscaleQuality,
};
pub use screenshot::{Screenshot, ScreenshotRequest};
pub use ui::{UiLayer, UiPass};
pub use view::{GlareMethod, OutputDebug, PostQuality, ViewOutput};
pub use white_balance::{AutoWhiteBalance, WHITE_BALANCE_CCT, WHITE_BALANCE_DUV};

/// Errors from orr_output.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Gpu(#[from] orr_scene::orr_gpu::Error),
    #[error("present thread: {0}")]
    Thread(String),
    #[cfg(dlss)]
    #[error(transparent)]
    Dlss(#[from] orr_dlss::DlssError),
    /// The device or system lacks something frame generation needs.
    #[error("frame generation needs {0}")]
    FrameGenerationUnavailable(&'static str),
    #[error("frame generation mode {0:?} is out of range for this device")]
    FrameGenerationMode(FrameGenerationMode),
    #[error(transparent)]
    Window(#[from] orr_window::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
