//! Luminance and luminous efficacy.

use std::sync::LazyLock;

use orr_math::DVec3;

use crate::RgbPrimaries;

/// Luminous efficacy of the CIE photopic curve's peak, lm/W.
pub const LUMINOUS_EFFICACY: f64 = 683.0;

/// Luminance (CIE Y) of a linear Rec.2020 color.
pub fn luminance(rec2020: impl Into<DVec3>) -> f64 {
    static WEIGHTS: LazyLock<DVec3> = LazyLock::new(|| RgbPrimaries::REC2020.luminance_weights());
    WEIGHTS.dot(rec2020.into())
}
