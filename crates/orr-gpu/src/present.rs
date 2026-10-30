/// A swapchain encoding: the surface format and colour space compose writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    /// `B8G8R8A8_UNORM` (or `R8G8B8A8_UNORM`) in `SRGB_NONLINEAR`; compose encodes sRGB.
    Sdr,
    /// `A2B10G10R10_UNORM_PACK32` in `HDR10_ST2084`.
    Hdr10Pq,
    /// `R16G16B16A16_SFLOAT` in `EXTENDED_SRGB_LINEAR`; 1.0 = 80 nits.
    ScRgb,
}

/// Mastering display metadata passed to `vkSetHdrMetadataEXT`. Chromaticities are CIE 1931
/// xy, luminance in nits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrMetadata {
    /// Red, green, blue.
    pub primaries: [[f32; 2]; 3],
    pub white: [f32; 2],
    pub max_luminance: f32,
    pub min_luminance: f32,
    /// Maximum content light level; 0 = unknown.
    pub max_cll: f32,
    /// Maximum frame-average light level; 0 = unknown.
    pub max_fall: f32,
}

/// How presents wait for the display.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PresentMode {
    /// `FIFO`: waits for vertical blank.
    Fifo,
    /// `IMMEDIATE`: presents without waiting, tearing allowed.
    Immediate,
}

/// How to create a [`crate::Swapchain`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwapchainDesc {
    /// At least the largest metered batch plus one; raised to the surface's minimum.
    pub min_images: u32,
    pub format: OutputFormat,
    /// For `Hdr10Pq` and `ScRgb`; ignored for `Sdr`.
    pub hdr: Option<HdrMetadata>,
    pub present_mode: PresentMode,
    /// Used only when the surface leaves the extent to the swapchain.
    pub extent: [u32; 2],
}

impl Default for SwapchainDesc {
    fn default() -> Self {
        Self {
            min_images: 5,
            format: OutputFormat::Sdr,
            hdr: None,
            present_mode: PresentMode::Fifo,
            extent: [1280, 720],
        }
    }
}

/// The display's refresh as the swapchain reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refresh {
    /// The shortest time between two refreshes; `None` when unknown.
    pub interval: Option<std::time::Duration>,
    /// Variable refresh rate. False when unknown.
    pub variable: bool,
}

/// A Reflex latency marker (`VkLatencyMarkerNV`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LatencyMarker {
    SimulationStart,
    SimulationEnd,
    RenderSubmitStart,
    RenderSubmitEnd,
    PresentStart,
    PresentEnd,
    InputSample,
}

/// One frame's latency report (`VkLatencyTimingsFrameReportNV`): times in microseconds on
/// the driver's clock, 0 where nothing was recorded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LatencyTimings {
    /// The frame id the markers carried.
    pub frame_id: u64,
    pub input_sample: u64,
    pub simulation_start: u64,
    pub simulation_end: u64,
    pub render_submit_start: u64,
    pub render_submit_end: u64,
    pub present_start: u64,
    pub present_end: u64,
    pub driver_start: u64,
    pub driver_end: u64,
    pub os_render_queue_start: u64,
    pub os_render_queue_end: u64,
    pub gpu_render_start: u64,
    pub gpu_render_end: u64,
}
