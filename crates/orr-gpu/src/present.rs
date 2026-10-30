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
