//! Colour-space matrices and chromaticity (D65 white, CIE 1931 2-deg observer).

/// CIE XYZ to linear Rec.2020.
pub const XYZ_TO_REC2020: [[f64; 3]; 3] = [
    [1.716_651_188_0, -0.355_670_783_8, -0.253_366_281_4],
    [-0.666_684_351_8, 1.616_481_236_6, 0.015_768_545_8],
    [0.017_639_857_4, -0.042_770_613_3, 0.942_103_121_2],
];

/// Linear Rec.2020 to CIE XYZ.
pub const REC2020_TO_XYZ: [[f64; 3]; 3] = [
    [0.636_958_048_4, 0.144_616_903_6, 0.168_880_975_2],
    [0.262_700_212_0, 0.677_998_071_5, 0.059_301_716_5],
    [0.0, 0.028_072_693_0, 1.060_985_057_7],
];

/// Linear Rec.709 to linear Rec.2020.
pub const REC709_TO_REC2020: [[f64; 3]; 3] = [
    [0.627_403_895_9, 0.329_283_038_4, 0.043_313_065_7],
    [0.069_097_289_4, 0.919_540_395_1, 0.011_362_315_6],
    [0.016_391_438_9, 0.088_013_307_9, 0.895_595_253_2],
];

/// Linear Rec.2020 to linear Rec.709.
pub const REC2020_TO_REC709: [[f64; 3]; 3] = [
    [1.660_491_002_3, -0.587_641_138_9, -0.072_849_863_3],
    [-0.124_550_474_6, 1.132_899_897_1, -0.008_349_422_6],
    [-0.018_150_763_4, -0.100_578_898_0, 1.118_729_661_4],
];

/// Luminance weights of linear Rec.2020.
pub const REC2020_LUMINANCE: [f64; 3] = [0.262_700_212_0, 0.677_998_071_5, 0.059_301_716_5];

/// Luminous efficacy of the CIE photopic curve's peak, lm/W.
pub const LUMINOUS_EFFICACY: f64 = 683.0;

/// `m` times `v`.
pub fn mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// CIE 1976 u'v' chromaticity of an XYZ colour.
pub fn uv(xyz: [f64; 3]) -> [f64; 2] {
    let d = xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2];
    [4.0 * xyz[0] / d, 9.0 * xyz[1] / d]
}

/// Distance between two XYZ colours' u'v' chromaticities.
pub fn delta_uv(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (p, q) = (uv(a), uv(b));
    (p[0] - q[0]).hypot(p[1] - q[1])
}
