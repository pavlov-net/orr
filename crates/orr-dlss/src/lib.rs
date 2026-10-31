//! DLSS ray reconstruction (RR), super resolution, DLAA and frame generation through NVIDIA
//! NGX on Vulkan.
//!
//! The build reads the DLSS SDK from `DLSS_SDK` (an absolute path) and links NGX's static
//! library. Without `DLSS_SDK` the crate still builds, and [`raw::Ngx::new`] returns
//! [`DlssError::NoSdk`]. The `mock` feature needs no SDK: it reports every feature
//! unsupported, and ray reconstruction copies the colour input to the output with a bilinear
//! upscale. The `debug_overlay` feature loads NGX's development DLLs, which draw the DLSS
//! debug overlay.
//!
//! NGX loads its feature DLLs from, in order: the executable's directory; in debug builds,
//! the SDK the crate was built against; the SDK at `DLSS_SDK` at run time.
//!
//! [`DlssSdk`], [`RayReconstruction`] and [`FrameGeneration`] work on an orr-gpu device
//! and record into its command lists, serializing every NGX call. [`raw`] works on ash
//! handles; its caller keeps NGX calls on one thread at a time.

mod error;
mod jitter;
pub mod raw;
mod sdk;

#[cfg(not(ngx))]
mod mock;
#[cfg(ngx)]
mod ngx;
#[cfg(ngx)]
mod sys;

pub use error::DlssError;
pub use jitter::{jitter, jitter_phases, mip_bias};
pub use raw::FgCamera;
pub use sdk::{DlssDeviceHooks, DlssSdk, FgInputs, FrameGeneration, RayReconstruction, RrInputs};
pub use uuid::Uuid;

/// Ratio of output to render resolution. `Dlaa` renders at output resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quality {
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

/// What NGX offers on a device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Support {
    /// Ray reconstruction.
    pub rr: bool,
    /// Super resolution and DLAA.
    pub sr: bool,
    /// Frame generation.
    pub fg: bool,
    /// Most generated frames per rendered frame; 0 without frame generation.
    pub max_generated: u32,
}
