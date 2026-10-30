//! Color constants for orr: RGB primaries, RGB and CIE XYZ matrices, luminance and transfer
//! functions.
//!
//! orr renders in linear Rec.2020. Matrices are row-major `[[f64; 3]; 3]` and act on a
//! linear RGB `[f64; 3]` as [`mul`]`(&m, rgb)`. They are `const fn`s computed in f64 from
//! the chromaticities; [`mat_f32`] and [`vec_f32`] round them for f32 constants:
//!
//! ```
//! use orr_color::{RgbPrimaries, mat_f32};
//!
//! const REC709_TO_REC2020: [[f32; 3]; 3] =
//!     mat_f32(RgbPrimaries::REC709.matrix_to(RgbPrimaries::REC2020));
//! ```

mod transfer;

pub use transfer::{PQ_MAX_NITS, SCRGB_WHITE_NITS, pq_decode, pq_encode, srgb_decode, srgb_encode};

/// A CIE 1931 xy chromaticity.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Chromaticity {
    pub x: f64,
    pub y: f64,
}

impl Chromaticity {
    /// CIE standard illuminant D65, the white point of every set in [`RgbPrimaries`].
    pub const D65: Self = Self::new(0.3127, 0.3290);

    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// CIE XYZ of this chromaticity at luminance (Y) `luminance`.
    pub const fn to_xyz(self, luminance: f64) -> [f64; 3] {
        let s = luminance / self.y;
        [self.x * s, luminance, (1.0 - self.x - self.y) * s]
    }
}

/// The chromaticities of an RGB space's primaries and of its white, the color of
/// `(1, 1, 1)`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RgbPrimaries {
    pub red: Chromaticity,
    pub green: Chromaticity,
    pub blue: Chromaticity,
    pub white: Chromaticity,
}

impl RgbPrimaries {
    /// ITU-R BT.709, shared with sRGB.
    pub const REC709: Self = Self::d65([0.640, 0.330], [0.300, 0.600], [0.150, 0.060]);
    /// Display P3: the DCI-P3 primaries with D65 white.
    pub const DISPLAY_P3: Self = Self::d65([0.680, 0.320], [0.265, 0.690], [0.150, 0.060]);
    /// ITU-R BT.2020, orr's working space.
    pub const REC2020: Self = Self::d65([0.708, 0.292], [0.170, 0.797], [0.131, 0.046]);

    const fn d65(r: [f64; 2], g: [f64; 2], b: [f64; 2]) -> Self {
        Self {
            red: Chromaticity::new(r[0], r[1]),
            green: Chromaticity::new(g[0], g[1]),
            blue: Chromaticity::new(b[0], b[1]),
            white: Chromaticity::D65,
        }
    }

    /// Linear RGB to CIE XYZ. `(1, 1, 1)` maps to the white point at Y = 1.
    pub const fn rgb_to_xyz(self) -> [[f64; 3]; 3] {
        let p = transpose([
            self.red.to_xyz(1.0),
            self.green.to_xyz(1.0),
            self.blue.to_xyz(1.0),
        ]);
        let s = mul(&inverse(&p), self.white.to_xyz(1.0));
        [
            [p[0][0] * s[0], p[0][1] * s[1], p[0][2] * s[2]],
            [p[1][0] * s[0], p[1][1] * s[1], p[1][2] * s[2]],
            [p[2][0] * s[0], p[2][1] * s[1], p[2][2] * s[2]],
        ]
    }

    /// CIE XYZ to linear RGB.
    pub const fn xyz_to_rgb(self) -> [[f64; 3]; 3] {
        inverse(&self.rgb_to_xyz())
    }

    /// Linear RGB in these primaries to linear RGB in `dst`. White points are not adapted.
    pub const fn matrix_to(self, dst: Self) -> [[f64; 3]; 3] {
        mul_mat(&dst.xyz_to_rgb(), &self.rgb_to_xyz())
    }

    /// The weights that give luminance (CIE Y) from linear RGB; they sum to 1.
    pub const fn luminance_weights(self) -> [f64; 3] {
        self.rgb_to_xyz()[1]
    }
}

/// The primaries ids stored in cooked textures.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Primaries {
    Rec709 = 0,
    DisplayP3 = 1,
    Rec2020 = 2,
}

impl Primaries {
    pub const fn id(self) -> u8 {
        self as u8
    }

    pub const fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Rec709),
            1 => Some(Self::DisplayP3),
            2 => Some(Self::Rec2020),
            _ => None,
        }
    }

    pub const fn rgb(self) -> RgbPrimaries {
        match self {
            Self::Rec709 => RgbPrimaries::REC709,
            Self::DisplayP3 => RgbPrimaries::DISPLAY_P3,
            Self::Rec2020 => RgbPrimaries::REC2020,
        }
    }
}

/// Luminance weights of linear Rec.2020.
pub const REC2020_LUMINANCE: [f64; 3] = RgbPrimaries::REC2020.luminance_weights();

/// Luminance (CIE Y) of a linear Rec.2020 color.
pub const fn luminance(rec2020: [f64; 3]) -> f64 {
    let w = REC2020_LUMINANCE;
    w[0] * rec2020[0] + w[1] * rec2020[1] + w[2] * rec2020[2]
}

/// `m` times `v`.
pub const fn mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// `m` times `v`, in f32.
pub const fn mul_f32(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// `m` rounded to f32.
pub const fn mat_f32(m: [[f64; 3]; 3]) -> [[f32; 3]; 3] {
    [vec_f32(m[0]), vec_f32(m[1]), vec_f32(m[2])]
}

/// `v` rounded to f32.
pub const fn vec_f32(v: [f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

const fn transpose(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

const fn mul_mat(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c = transpose(*b);
    transpose([mul(a, c[0]), mul(a, c[1]), mul(a, c[2])])
}

/// The cofactor of row `i`, column `j`; the cyclic index order carries the sign.
const fn cofactor(m: &[[f64; 3]; 3], i: usize, j: usize) -> f64 {
    let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
    let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
    m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
}

const fn inverse(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det =
        m[0][0] * cofactor(m, 0, 0) + m[0][1] * cofactor(m, 0, 1) + m[0][2] * cofactor(m, 0, 2);
    let mut inv = [[0.0; 3]; 3];
    let mut i = 0;
    while i < 3 {
        let mut j = 0;
        while j < 3 {
            inv[j][i] = cofactor(m, i, j) / det;
            j += 1;
        }
        i += 1;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    const SETS: [RgbPrimaries; 3] = [
        RgbPrimaries::REC709,
        RgbPrimaries::DISPLAY_P3,
        RgbPrimaries::REC2020,
    ];
    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn assert_close<T: Copy + Into<f64>>(a: &[[f64; 3]; 3], b: &[[T; 3]; 3], tol: f64, what: &str) {
        for i in 0..3 {
            for j in 0..3 {
                let b = b[i][j].into();
                let d = (a[i][j] - b).abs();
                assert!(d <= tol, "{what}[{i}][{j}]: {} vs {b}", a[i][j]);
            }
        }
    }

    fn assert_vec_close(a: [f64; 3], b: [f64; 3], tol: f64) {
        assert!((0..3).all(|i| (a[i] - b[i]).abs() <= tol), "{a:?} vs {b:?}");
    }

    #[test]
    fn rec709_to_rec2020_matches_bt2087() {
        // ITU-R BT.2087-0, section 2.2.1.
        let bt2087 = [
            [0.6274, 0.3293, 0.0433],
            [0.0691, 0.9195, 0.0114],
            [0.0164, 0.0880, 0.8956],
        ];
        let m = RgbPrimaries::REC709.matrix_to(RgbPrimaries::REC2020);
        assert_close(&m, &bt2087, 5e-5, "BT.2087");
    }

    #[test]
    fn matrices_round_trip_and_keep_white() {
        let d65 = Chromaticity::D65.to_xyz(1.0);
        for a in SETS {
            assert_close(
                &mul_mat(&a.xyz_to_rgb(), &a.rgb_to_xyz()),
                &IDENTITY,
                1e-14,
                "xyz",
            );
            assert_vec_close(mul(&a.rgb_to_xyz(), [1.0; 3]), d65, 1e-14);
            for b in SETS {
                let there_and_back = mul_mat(&b.matrix_to(a), &a.matrix_to(b));
                assert_close(&there_and_back, &IDENTITY, 1e-14, "rgb");
                assert_vec_close(mul(&a.matrix_to(b), [1.0; 3]), [1.0; 3], 1e-14);
            }
        }
    }

    const SLANG: &str = include_str!("../../orr-shaders/shaders/orr/color.slang");

    /// The `float3x3` constant `name` in Slang's `orr.color`.
    fn slang_matrix(name: &str) -> [[f64; 3]; 3] {
        let decl = format!("float3x3 {name} = float3x3(");
        let start = SLANG.find(&decl).unwrap_or_else(|| panic!("{name}")) + decl.len();
        let body = &SLANG[start..start + SLANG[start..].find(')').unwrap()];
        let v: Vec<f64> = body.split(',').map(|s| s.trim().parse().unwrap()).collect();
        [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
    }

    #[test]
    fn matrices_match_slang() {
        let named = [
            (RgbPrimaries::REC709, "REC709"),
            (RgbPrimaries::DISPLAY_P3, "P3"),
            (RgbPrimaries::REC2020, "REC2020"),
        ];
        for (p, n) in named {
            let to = format!("{n}_TO_XYZ");
            let from = format!("XYZ_TO_{n}");
            assert_close(&p.rgb_to_xyz(), &slang_matrix(&to), 1e-7, &to);
            assert_close(&p.xyz_to_rgb(), &slang_matrix(&from), 3e-7, &from);
        }
        let rec2020 = named[2];
        for src in &named[..2] {
            for (a, b) in [(*src, rec2020), (rec2020, *src)] {
                let n = format!("{}_TO_{}", a.1, b.1);
                assert_close(&a.0.matrix_to(b.0), &slang_matrix(&n), 2e-7, &n);
            }
        }
        let d65 = Chromaticity::D65;
        assert!(SLANG.contains(&format!("D65_XY = float2({:.4}, {:.4})", d65.x, d65.y)));
    }

    #[test]
    fn matrices_match_crate_copies() {
        use RgbPrimaries as P;
        // orr-atmosphere, src/color.rs.
        let xyz_to_rec2020 = [
            [1.716_651_188_0, -0.355_670_783_8, -0.253_366_281_4],
            [-0.666_684_351_8, 1.616_481_236_6, 0.015_768_545_8],
            [0.017_639_857_4, -0.042_770_613_3, 0.942_103_121_2],
        ];
        let rec2020_to_xyz = [
            [0.636_958_048_4, 0.144_616_903_6, 0.168_880_975_2],
            [0.262_700_212_0, 0.677_998_071_5, 0.059_301_716_5],
            [0.0, 0.028_072_693_0, 1.060_985_057_7],
        ];
        let rec2020_to_rec709 = [
            [1.660_491_002_3, -0.587_641_138_9, -0.072_849_863_3],
            [-0.124_550_474_6, 1.132_899_897_1, -0.008_349_422_6],
            [-0.018_150_763_4, -0.100_578_898_0, 1.118_729_661_4],
        ];
        assert_close(
            &P::REC2020.xyz_to_rgb(),
            &xyz_to_rec2020,
            1e-10,
            "XYZ_TO_REC2020",
        );
        assert_close(
            &P::REC2020.rgb_to_xyz(),
            &rec2020_to_xyz,
            1e-10,
            "REC2020_TO_XYZ",
        );
        let m = P::REC2020.matrix_to(P::REC709);
        assert_close(&m, &rec2020_to_rec709, 1e-9, "REC2020_TO_REC709");
        assert_vec_close(REC2020_LUMINANCE, rec2020_to_xyz[1], 1e-10);

        // orr-output, src/screenshot.rs.
        let rec709_to_rec2020 = [
            [
                0.627_403_895_934_699,
                0.329_283_038_377_884,
                0.043_313_065_687_417,
            ],
            [
                0.069_097_289_358_232,
                0.919_540_395_075_458,
                0.011_362_315_566_310,
            ],
            [
                0.016_391_438_875_150,
                0.088_013_307_877_226,
                0.895_595_253_247_624,
            ],
        ];
        let m = P::REC709.matrix_to(P::REC2020);
        assert_close(&m, &rec709_to_rec2020, 1e-14, "REC709_TO_REC2020");

        // orr-scene, src/lights.rs and src/material.rs, in f32.
        let lights: [[f32; 3]; 3] = [
            [0.627_403_9, 0.329_283_04, 0.043_313_066],
            [0.069_097_29, 0.919_540_4, 0.011_362_316],
            [0.016_391_438, 0.088_013_31, 0.895_595_25],
        ];
        let material: [[f32; 3]; 3] = [
            [0.627_403_9, 0.329_283_04, 0.043_313_06],
            [0.069_097_29, 0.919_540_4, 0.011_362_3],
            [0.016_391_44, 0.088_013_3, 0.895_595_3],
        ];
        assert_close(&m, &lights, 6e-8, "lights REC709_TO_REC2020");
        assert_close(&m, &material, 1e-7, "material REC709_TO_REC2020");

        // orr-cook, src/texture/mod.rs, to four places.
        let cook = [
            (P::REC709, [0.2126, 0.7152, 0.0722]),
            (P::DISPLAY_P3, [0.2290, 0.6917, 0.0793]),
            (P::REC2020, [0.2627, 0.6780, 0.0593]),
        ];
        for (p, w4) in cook {
            assert_vec_close(p.luminance_weights(), w4, 5e-5);
        }
    }
}
