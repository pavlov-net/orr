//! Exposure units. Scene luminance is in cd/m2; EV100 is the reflected-light meter value at
//! ISO 100 with K = 12.5. An exposure `s` scales luminance to exposed values, where 1.0 is
//! paper white.

/// The exposed value auto exposure aims the metered average at, before compensation.
pub const MID_GREY: f32 = 0.18;

/// `log2(luminance) + 3`.
pub fn ev100_from_luminance(luminance: f32) -> f32 {
    luminance.log2() + 3.0
}

pub fn luminance_from_ev100(ev100: f32) -> f32 {
    (ev100 - 3.0).exp2()
}

/// Manual exposure at `ev100`: `1 / (1.2 * 2^ev100)`.
pub fn exposure_from_ev100(ev100: f32) -> f32 {
    1.0 / (1.2 * ev100.exp2())
}

/// The EV100 of exposure `s`, as telemetry reports it: `-log2(1.2 * s)`.
pub fn ev100_from_exposure(exposure: f32) -> f32 {
    -(1.2 * exposure).log2()
}
