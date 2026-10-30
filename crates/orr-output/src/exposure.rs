//! Exposure units and the exposure settings of a view. Scene luminance is in cd/m2; EV100 is
//! the reflected-light meter value at ISO 100 with K = 12.5. An exposure `s` scales luminance
//! to exposed values, where 1.0 is paper white.

use std::ops::RangeInclusive;

use orr_scene::orr_gpu::TextureIndex;

/// The exposed value auto exposure aims the metered average at, before compensation.
pub const MID_GREY: f32 = 0.18;

/// Bins of the meter's histogram over [`AutoExposure::range_ev100`].
pub(crate) const HISTOGRAM_BINS: u32 = 128;

/// Words per histogram bin (`exposure.slang`).
pub(crate) const HISTOGRAM_WORDS_PER_BIN: u64 = 7;

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

/// The exposure, as EV100, that shows a scene metered at `metered_ev100` at
/// `MID_GREY * 2^stops`.
pub fn keyed_ev100(metered_ev100: f32, stops: f32) -> f32 {
    ev100_from_exposure(MID_GREY * stops.exp2() / luminance_from_ev100(metered_ev100))
}

/// How a view is exposed.
#[derive(Clone, Debug, PartialEq)]
pub enum Exposure {
    /// A fixed exposure, [`exposure_from_ev100`] of `ev100`, shown from the frame it is set.
    Manual {
        ev100: f32,
    },
    Auto(AutoExposure),
}

impl Default for Exposure {
    fn default() -> Self {
        Self::Auto(AutoExposure::default())
    }
}

/// Auto exposure: a histogram of the view's luminance in EV100, optionally mixed with the
/// scene's sky statistics, keyed to [`MID_GREY`] through the compensation curve and adapted
/// over real time.
///
/// Exposures are EV100 as [`ev100_from_exposure`] gives them. A scene metered at EV100 `m`
/// with no compensation is exposed at `keyed_ev100(m, 0.0)`, about `m - 0.79`.
///
/// A view's first metered frames, and the frames after a cut (`ViewOutput::cut`), snap to
/// the target until two snaps agree within half a stop. A frame with nothing to meter (all
/// black) holds the exposure, so a black first frame leaves the view waiting for one with
/// light.
#[derive(Clone, Debug, PartialEq)]
pub struct AutoExposure {
    /// The histogram's range of scene EV100; luminance outside it counts in the end bins.
    pub range_ev100: RangeInclusive<f32>,
    /// The exposure window: the target and the adapted exposure stay inside it every frame.
    pub clamp_ev100: RangeInclusive<f32>,
    /// The histogram percentiles the mean is taken between.
    pub filter: RangeInclusive<f32>,
    /// Stops per second while the scene gets brighter.
    pub speed_brighten: f32,
    /// Stops per second while the scene gets darker.
    pub speed_darken: f32,
    /// Distance in stops from the target inside which adaptation slows exponentially.
    pub exponential_ev: f32,
    /// Stops above [`MID_GREY`] the metered average is shown at, by metered EV100.
    pub compensation: CompensationCurve,
    /// Mixes the histogram with the scene's sky statistics.
    pub references: Option<MeterWeights>,
    /// A slow long-term stage that bounds the fast one.
    pub physiological: Option<Physiological>,
    /// Per-pixel metering weights in R, quantised to 16 levels; 0 leaves a pixel out. Any
    /// size; it is stretched over the view.
    pub metering_mask: Option<TextureIndex>,
}

impl Default for AutoExposure {
    fn default() -> Self {
        Self {
            range_ev100: -12.0..=20.0,
            clamp_ev100: -12.0..=20.0,
            filter: 0.10..=0.90,
            speed_brighten: 3.0,
            speed_darken: 1.0,
            exponential_ev: 1.5,
            compensation: CompensationCurve::default(),
            references: None,
            physiological: None,
            metering_mask: None,
        }
    }
}

/// Fixed weights mixing EV100 references. Dome: the scene's dome luminance. Directional:
/// the illuminance of every directional light summed after the medium, as a mid-grey card,
/// `L = 0.18 * E / pi`. A reference that is black that frame drops out of the mix.
///
/// Without [`Physiological`] the mix is the target. With it, the mix is the long-term
/// stage's target and the fast stage meters the histogram alone within its bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeterWeights {
    pub histogram: f32,
    pub dome: f32,
    pub directional: f32,
}

/// The slow stage of two-stage adaptation. It follows the metered target (or the reference
/// mix) at its own speeds, and the fast stage's target stays within `bound_darken` stops
/// below and `bound_brighten` stops above it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Physiological {
    pub speed_brighten: f32,
    pub speed_darken: f32,
    pub bound_brighten: f32,
    pub bound_darken: f32,
}

impl Default for Physiological {
    fn default() -> Self {
        Self {
            speed_brighten: 0.5,
            speed_darken: 0.25,
            bound_brighten: 2.0,
            bound_darken: 2.0,
        }
    }
}

/// Exposure compensation in stops by metered EV100: piecewise linear through its knots,
/// constant past the first and last. No knots is 0 everywhere.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompensationCurve {
    knots: Vec<[f32; 2]>,
}

/// Entries of the table the curve is baked into, spread over the histogram range.
pub(crate) const COMPENSATION_ENTRIES: usize = 256;

impl CompensationCurve {
    /// A curve through `(ev100, stops)` knots, in any order.
    pub fn new(knots: impl IntoIterator<Item = (f32, f32)>) -> Self {
        let mut knots: Vec<[f32; 2]> = knots.into_iter().map(|(e, s)| [e, s]).collect();
        knots.sort_by(|a, b| a[0].total_cmp(&b[0]));
        Self { knots }
    }

    /// The compensation at `ev100`.
    pub fn stops(&self, ev100: f32) -> f32 {
        let (Some(first), Some(last)) = (self.knots.first(), self.knots.last()) else {
            return 0.0;
        };
        if ev100 <= first[0] {
            return first[1];
        }
        if ev100 >= last[0] {
            return last[1];
        }
        let i = self.knots.partition_point(|k| k[0] <= ev100);
        let [a, b] = [self.knots[i - 1], self.knots[i]];
        a[1] + (b[1] - a[1]) * (ev100 - a[0]) / (b[0] - a[0])
    }

    /// The table the GPU interpolates: `COMPENSATION_ENTRIES` evenly spaced over `range`.
    pub(crate) fn bake(&self, range: &RangeInclusive<f32>) -> [f32; COMPENSATION_ENTRIES] {
        let step = (range.end() - range.start()) / (COMPENSATION_ENTRIES - 1) as f32;
        std::array::from_fn(|i| self.stops(range.start() + i as f32 * step))
    }
}

/// A per-view exposure bias and colour matrix on linear Rec.2020, applied after metering and
/// before GT7, so the meter never cancels them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grade {
    pub exposure_ev: f32,
    /// Rows; `out = matrix * rgb`.
    pub matrix: [[f32; 3]; 3],
}

impl Default for Grade {
    fn default() -> Self {
        Self {
            exposure_ev: 0.0,
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

/// A view's exposure as adapt left it, read back 1-3 frames late.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureTelemetry {
    /// The frame it belongs to.
    pub frame: u64,
    pub adapted_ev100: f32,
    pub target_ev100: f32,
    pub long_term_ev100: f32,
    /// The histogram's mean scene EV100; `None` when nothing was metered (manual exposure,
    /// or a black frame, which holds the exposure).
    pub metered_ev100: Option<f32>,
    /// The pre-exposure the view's next render is lit with.
    pub pre_exposure: f32,
    /// The white point the view is balanced from, CIE 1931 xy; D65 without auto white
    /// balance.
    pub white_point_xy: [f32; 2],
    /// The white point auto white balance moves toward, held in range.
    pub white_target_xy: [f32; 2],
    pub sky: SkyTelemetry,
}

/// The scene's sky statistics as the meter read them that frame; zero without a scene.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SkyTelemetry {
    pub dome_luminance: f32,
    pub dome_xy: [f32; 2],
    /// Linear Rec.2020, every directional light summed.
    pub directional_lux: [f32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensation_curve_interpolates_and_holds_its_ends() {
        let c = CompensationCurve::new([(10.0, -1.0), (0.0, 1.0), (4.0, 0.0)]);
        assert_eq!(c.stops(-5.0), 1.0);
        assert_eq!(c.stops(2.0), 0.5);
        assert_eq!(c.stops(7.0), -0.5);
        assert_eq!(c.stops(30.0), -1.0);
        let lut = c.bake(&(-12.0..=20.0));
        assert_eq!((lut[0], lut[COMPENSATION_ENTRIES - 1]), (1.0, -1.0));
    }
}
