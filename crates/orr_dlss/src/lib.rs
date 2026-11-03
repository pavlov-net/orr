//! DLSS ray reconstruction (RR), super resolution, DLAA and frame generation through NVIDIA
//! NGX on Vulkan.
//!
//! The build reads the DLSS SDK from `DLSS_SDK` (an absolute path) and links NGX's static
//! library. Without `DLSS_SDK` the crate still builds, and [`raw::Ngx::new`] returns
//! [`DlssError::NoSdk`]. The `mock` feature needs no SDK: it reports every feature
//! unsupported, and ray reconstruction and super resolution copy the color input to the
//! output with a bilinear upscale. The `debug_overlay` feature loads NGX's development DLLs,
//! which draw the DLSS debug overlay.
//!
//! NGX loads its feature DLLs from, in order: the executable's directory; in debug builds,
//! the SDK the crate was built against; the SDK at `DLSS_SDK` at run time.
//!
//! [`DlssSdk`], [`RayReconstruction`], [`SuperResolution`] and [`FrameGeneration`] work on
//! an orr_gpu device and record into its command lists, serializing every NGX call. [`raw`]
//! holds the NGX instance and ray reconstruction on ash handles; its caller keeps NGX calls on
//! one thread at a time.

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
pub use raw::FgCamera;
pub use sdk::{
    DlssDeviceHooks, DlssSdk, FgInputs, FrameGeneration, RayReconstruction, RrInputs, SrInputs,
    SuperResolution,
};
pub use uuid::Uuid;

/// The smallest output width and height DLSS takes.
pub const MIN_OUTPUT: [u32; 2] = [64, 32];

/// What a caller asks DLSS for: a [`Mode`], or the one that suits the output resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Quality {
    /// By output pixels: Quality up to 2560 x 1440, Performance up to 3840 x 2160, Ultra
    /// Performance above.
    #[default]
    Auto,
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl Quality {
    /// The mode this asks for at `output`.
    pub fn resolve(self, output: [u32; 2]) -> Mode {
        match self {
            Self::Auto => Mode::for_output(output),
            Self::Dlaa => Mode::Dlaa,
            Self::Quality => Mode::Quality,
            Self::Balanced => Mode::Balanced,
            Self::Performance => Mode::Performance,
            Self::UltraPerformance => Mode::UltraPerformance,
        }
    }
}

/// Ratio of output to render resolution, as NGX takes it. `Dlaa` renders at output resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl Mode {
    /// The mode for an output of this many pixels: Quality up to 2560 x 1440, Performance up to
    /// 3840 x 2160, Ultra Performance above. From 1920 x 1080 up these are the defaults of the
    /// RTX UI Developer Guidelines' table.
    fn for_output([width, height]: [u32; 2]) -> Self {
        match u64::from(width) * u64::from(height) {
            p if p <= 2560 * 1440 => Self::Quality,
            p if p <= 3840 * 2160 => Self::Performance,
            _ => Self::UltraPerformance,
        }
    }

    /// The mode whose nominal scale is closest to rendering at `render` for `output`.
    fn for_render_size(render: [u32; 2], output: [u32; 2]) -> Self {
        let scale = f64::from(render[0]) / f64::from(output[0]);
        [
            Self::Dlaa,
            Self::Quality,
            Self::Balanced,
            Self::Performance,
            Self::UltraPerformance,
        ]
        .into_iter()
        .min_by(|a, b| {
            (a.scale() - scale)
                .abs()
                .total_cmp(&(b.scale() - scale).abs())
        })
        .expect("the list is not empty")
    }

    /// NGX's published ratio of render to output resolution on each axis.
    fn scale(self) -> f64 {
        match self {
            Self::Dlaa => 1.0,
            Self::Quality => 2.0 / 3.0,
            Self::Balanced => 0.58,
            Self::Performance => 0.5,
            Self::UltraPerformance => 1.0 / 3.0,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{Mode, Quality};

    #[test]
    fn auto_follows_the_output_resolution_at_each_boundary() {
        for (output, mode) in [
            ([1919, 1080], Mode::Quality),
            ([2560, 1080], Mode::Quality),
            ([2560, 1440], Mode::Quality),
            ([2561, 1440], Mode::Performance),
            ([3440, 1440], Mode::Performance),
            ([3840, 2160], Mode::Performance),
            ([3841, 2160], Mode::UltraPerformance),
            ([u32::MAX, u32::MAX], Mode::UltraPerformance),
        ] {
            assert_eq!(Quality::default().resolve(output), mode, "{output:?}");
        }
    }

    #[test]
    fn a_render_size_picks_the_nearest_mode() {
        let output = [3840, 2160];
        for (render, mode) in [
            ([3840, 2160], Mode::Dlaa),
            ([2560, 1440], Mode::Quality),
            ([2227, 1253], Mode::Balanced),
            ([1920, 1080], Mode::Performance),
            ([1280, 720], Mode::UltraPerformance),
        ] {
            assert_eq!(Mode::for_render_size(render, output), mode, "{render:?}");
        }
    }
}
