//! UsdLux `inputs:colorTemperature` as a linear Rec.709 color.
//!
//! The knot table and the spline are `UsdLuxBlackbodyTemperatureAsRgb` of OpenUSD v26.05
//! (`pxr/usd/usdLux/blackbody.cpp`, Copyright 2016 Pixar, under the Modified Apache 2.0 License
//! in OpenUSD's `LICENSE.txt`), whose table comes from John Walker's public-domain
//! `specrend.c` (<https://www.fourmilab.ch/documents/specrend/specrend.c>): Rec.709 colors of
//! blackbodies from 1000 K to 10000 K in 500 K steps, the end knots repeated.

use orr_math::DVec3;

const KNOTS: [[f64; 3]; 22] = [
    [1.000000, 0.027490, 0.000000],
    [1.000000, 0.027490, 0.000000],
    [1.000000, 0.149664, 0.000000],
    [1.000000, 0.256644, 0.008095],
    [1.000000, 0.372033, 0.067450],
    [1.000000, 0.476725, 0.153601],
    [1.000000, 0.570376, 0.259196],
    [1.000000, 0.653480, 0.377155],
    [1.000000, 0.726878, 0.501606],
    [1.000000, 0.791543, 0.628050],
    [1.000000, 0.848462, 0.753228],
    [1.000000, 0.898581, 0.874905],
    [1.000000, 0.942771, 0.991642],
    [0.906947, 0.890456, 1.000000],
    [0.828247, 0.841838, 1.000000],
    [0.765791, 0.801896, 1.000000],
    [0.715255, 0.768579, 1.000000],
    [0.673683, 0.740423, 1.000000],
    [0.638992, 0.716359, 1.000000],
    [0.609681, 0.695588, 1.000000],
    [0.609681, 0.695588, 1.000000],
    [0.609681, 0.695588, 1.000000],
];

/// `UsdLuxBlackbodyTemperatureAsRgb(kelvin)`: the Catmull-Rom spline through [`KNOTS`] at
/// `kelvin` clamped to [1000, 10000], at the Rec.709 luma of white, negative channels
/// clamped to 0.
fn blackbody_rgb(kelvin: f64) -> DVec3 {
    let k = |i: usize| DVec3::from_array(KNOTS[i]);
    let segments = (KNOTS.len() - 4) as f64;
    let x = ((kelvin - 1000.0) / 9000.0).clamp(0.0, 1.0) * segments;
    let seg = x.floor() as usize;
    let u = x - seg as f64;
    let [k0, k1, k2, k3] = [k(seg), k(seg + 1), k(seg + 2), k(seg + 3)];
    let a = -0.5 * k0 + 1.5 * k1 - 1.5 * k2 + 0.5 * k3;
    let b = k0 - 2.5 * k1 + 2.0 * k2 - 0.5 * k3;
    let c = -0.5 * k0 + 0.5 * k2;
    let rgb = ((a * u + b) * u + c) * u + k1;
    (rgb / rgb.dot(DVec3::new(0.2126, 0.7152, 0.0722))).max(DVec3::ZERO)
}

/// The color `inputs:colorTemperature` multiplies `inputs:color` by: the blackbody's linear
/// Rec.709 color, D65 white, "normalized such that the default value of 6500 will always result
/// in white" (`usdLux/schema.usda`), so divided by its color at 6500 K.
pub(super) fn temperature_color(kelvin: f64) -> DVec3 {
    blackbody_rgb(kelvin) / blackbody_rgb(6500.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spline passes through its knots, at Rec.709 luma 1, and 6500 K is white.
    #[test]
    fn temperature_colors_follow_the_knots_and_6500_is_white() {
        let at = |kelvin: f64| {
            let knot = DVec3::from_array(KNOTS[((kelvin - 1000.0) / 500.0) as usize + 1]);
            knot / knot.dot(DVec3::new(0.2126, 0.7152, 0.0722))
        };
        for kelvin in [2000.0, 3500.0, 6500.0, 9000.0] {
            assert!(
                blackbody_rgb(kelvin).abs_diff_eq(at(kelvin), 1e-12),
                "{kelvin} K"
            );
        }
        assert!(temperature_color(6500.0).abs_diff_eq(DVec3::ONE, 1e-12));
        assert!(blackbody_rgb(20000.0).abs_diff_eq(at(10000.0), 1e-12));
    }
}
