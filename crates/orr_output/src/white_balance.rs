//! Auto white balance settings.

use std::range::RangeInclusive;

use crate::MeterWeights;

/// The correlated color temperatures, in kelvin, a white point is held inside.
pub const WHITE_BALANCE_CCT: RangeInclusive<f32> = RangeInclusive {
    start: 2500.0,
    last: 7000.0,
};

/// The largest distance from the Planckian locus, in CIE 1960 uv, a white point keeps.
pub const WHITE_BALANCE_DUV: f32 = 0.05;

/// Auto white balance: the meter's chromaticity, optionally mixed with the scene's sky
/// statistics, read as the scene's white point and adapted to D65 by CAT16.
///
/// The histogram (between the exposure filter's percentiles), the sky dome and the
/// directional lights (as a mid-gray card) are averaged in Yxy, each weighted by its
/// luminance times its [`MeterWeights`] entry, together with a D65 light of
/// `virtual_light_nits` that is not weighted, so dim scenes stay near D65. The white point
/// is held within [`WHITE_BALANCE_CCT`] and [`WHITE_BALANCE_DUV`] of the locus, and moves
/// toward that target in CIE 1960 uv, closing `1 - exp(-speed * dt)` of the distance a
/// frame. The view's first frame with light, and a camera cut's, take the target as the white
/// point. Frames with nothing to meter hold it.
///
/// With [`crate::Exposure::Manual`], the meter still runs for the chromaticity, over
/// [`crate::AutoExposure::default`]'s range and filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoWhiteBalance {
    /// Per second.
    pub speed: f32,
    /// cd/m2.
    pub virtual_light_nits: f32,
    /// `None` meters the histogram alone.
    pub references: Option<MeterWeights>,
}

impl Default for AutoWhiteBalance {
    fn default() -> Self {
        Self {
            speed: 1.0,
            virtual_light_nits: 1.0,
            references: None,
        }
    }
}
