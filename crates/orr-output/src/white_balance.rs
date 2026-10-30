//! Auto white balance settings and the white point maths `orr.color` runs on the GPU:
//! McCamy's correlated colour temperature and Duv from Kang's Planckian locus.

use std::ops::RangeInclusive;

use crate::MeterWeights;

/// CIE 1931 xy of D65, the white auto white balance adapts to.
pub const D65_XY: [f32; 2] = [0.3127, 0.3290];

/// The correlated colour temperatures, in kelvin, a white point is held inside.
pub const WHITE_BALANCE_CCT: RangeInclusive<f32> = 2500.0..=7000.0;

/// The largest distance from the Planckian locus, in CIE 1960 uv, a white point keeps.
pub const WHITE_BALANCE_DUV: f32 = 0.05;

/// Auto white balance: the meter's chromaticity, optionally mixed with the scene's sky
/// statistics, read as the scene's white point and adapted to D65 by CAT16.
///
/// The histogram (between the exposure filter's percentiles), the sky dome and the
/// directional lights (as a mid-grey card) are averaged in Yxy, each weighted by its
/// luminance times its [`MeterWeights`] entry, together with a D65 light of
/// `virtual_light_nits` that is not weighted, so dim scenes stay near D65. The white point
/// is held within [`WHITE_BALANCE_CCT`] and [`WHITE_BALANCE_DUV`] of the locus, and moves
/// toward that target in CIE 1960 uv, closing `1 - exp(-speed * dt)` of the distance a
/// frame. Cuts, the view's first frame with light, and frames on which auto exposure snaps
/// snap the white point too. Frames with nothing to meter hold it.
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

/// McCamy's correlated colour temperature, in kelvin, and the signed distance in CIE 1960 uv
/// from the Planckian locus at that temperature, positive above the locus (toward green).
/// Chromaticities at or below y = 0.1858 read as the hot end; the cubic turns at about
/// 1620 K, so redder ones read 1620 K.
pub fn cct_duv_from_xy(xy: [f32; 2]) -> (f32, f32) {
    let xy = xy.map(f64::from);
    let cct = mccamy(xy);
    (cct as f32, duv(xy, cct) as f32)
}

/// The chromaticity at `cct` kelvin and `duv` from the Planckian locus (Kang et al. 2002,
/// 1667-25000 K, clamped outside).
pub fn xy_from_cct_duv(cct: f32, duv: f32) -> [f32; 2] {
    let (cct, duv) = (f64::from(cct), f64::from(duv));
    let p = uv(planckian(cct));
    let n = normal(cct);
    xy([p[0] + duv * n[0], p[1] + duv * n[1]]).map(|c| c as f32)
}

fn mccamy([x, y]: [f64; 2]) -> f64 {
    let n = if y > 0.1858 {
        ((x - 0.3320) / (0.1858 - y)).max(-1.28)
    } else {
        4.0
    };
    ((449.0 * n + 3525.0) * n + 6823.3) * n + 5520.33
}

fn planckian(cct: f64) -> [f64; 2] {
    let t = cct.clamp(1667.0, 25000.0);
    let r = 1000.0 / t;
    let x = if t <= 4000.0 {
        ((-0.2661239 * r - 0.2343589) * r + 0.8776956) * r + 0.179910
    } else {
        ((-3.0258469 * r + 2.1070379) * r + 0.2226347) * r + 0.240390
    };
    let y = if t <= 2222.0 {
        ((-1.1063814 * x - 1.34811020) * x + 2.18555832) * x - 0.20219683
    } else if t <= 4000.0 {
        ((-0.9549476 * x - 1.37418593) * x + 2.09137015) * x - 0.16748867
    } else {
        ((3.0817580 * x - 5.87338670) * x + 3.75112997) * x - 0.37001483
    };
    [x, y]
}

fn uv([x, y]: [f64; 2]) -> [f64; 2] {
    let d = -2.0 * x + 12.0 * y + 3.0;
    [4.0 * x / d, 6.0 * y / d]
}

fn xy([u, v]: [f64; 2]) -> [f64; 2] {
    let d = 2.0 * u - 8.0 * v + 4.0;
    [3.0 * u / d, 2.0 * v / d]
}

/// The locus normal in uv, from the same central difference the GPU takes.
fn normal(cct: f64) -> [f64; 2] {
    let (a, b) = (uv(planckian(cct * 0.99)), uv(planckian(cct * 1.01)));
    let d = [b[0] - a[0], b[1] - a[1]];
    let len = d[0].hypot(d[1]);
    [d[1] / len, -d[0] / len]
}

fn duv(xy: [f64; 2], cct: f64) -> f64 {
    let (p, q, n) = (uv(xy), uv(planckian(cct)), normal(cct));
    (p[0] - q[0]) * n[0] + (p[1] - q[1]) * n[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locus_reads_known_illuminants() {
        // CIE A (2856 K) and D65 (6504 K, about 0.0032 above the locus).
        let (a, a_duv) = cct_duv_from_xy([0.44757, 0.40745]);
        assert!(
            (a - 2856.0).abs() < 2.0 && a_duv.abs() < 5e-4,
            "{a} K, {a_duv}"
        );
        let (d, d_duv) = cct_duv_from_xy(D65_XY);
        assert!((d - 6504.0).abs() < 5.0, "{d} K");
        assert!((d_duv - 0.0032).abs() < 3e-4, "{d_duv}");
        let back = xy_from_cct_duv(d, d_duv);
        assert!((back[0] - D65_XY[0]).abs() < 3e-4 && (back[1] - D65_XY[1]).abs() < 3e-4);
    }
}
