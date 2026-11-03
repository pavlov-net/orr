//! RGB primaries and the matrices between RGB spaces and CIE XYZ.

use std::sync::LazyLock;

use orr_math::{DMat3, DVec3};

use crate::Chromaticity;

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
    /// ACEScg (the ACES AP1 primaries) with the ACES white.
    pub const ACES_CG: Self = Self {
        white: Chromaticity::D60,
        ..Self::d65([0.713, 0.293], [0.165, 0.830], [0.128, 0.044])
    };

    /// Primaries at `[x, y]` with D65 white.
    const fn d65(r: [f64; 2], g: [f64; 2], b: [f64; 2]) -> Self {
        Self {
            red: Chromaticity::new(r[0], r[1]),
            green: Chromaticity::new(g[0], g[1]),
            blue: Chromaticity::new(b[0], b[1]),
            white: Chromaticity::D65,
        }
    }

    /// Linear RGB to CIE XYZ. `(1, 1, 1)` maps to the white at Y = 1.
    pub fn rgb_to_xyz(self) -> DMat3 {
        let p = DMat3::from_cols(
            self.red.to_xyz(1.0),
            self.green.to_xyz(1.0),
            self.blue.to_xyz(1.0),
        );
        p * DMat3::from_diagonal(p.inverse() * self.white.to_xyz(1.0))
    }

    /// CIE XYZ to linear RGB.
    pub fn xyz_to_rgb(self) -> DMat3 {
        self.rgb_to_xyz().inverse()
    }

    /// Linear RGB in these primaries to linear RGB in `dst`. When the whites differ, the
    /// Bradford transform adapts this white to `dst`'s, so `(1, 1, 1)` maps to `(1, 1, 1)`.
    pub fn matrix_to(self, dst: Self) -> DMat3 {
        let mut to_xyz = self.rgb_to_xyz();
        if self.white != dst.white {
            to_xyz = bradford(self.white, dst.white) * to_xyz;
        }
        dst.xyz_to_rgb() * to_xyz
    }

    /// The weights that give luminance (CIE Y) from linear RGB; they sum to 1.
    pub fn luminance_weights(self) -> DVec3 {
        self.rgb_to_xyz().row(1)
    }
}

/// Linear Rec.709 to linear Rec.2020.
pub fn rec709_to_rec2020(rec709: impl Into<DVec3>) -> DVec3 {
    static MATRIX: LazyLock<DMat3> =
        LazyLock::new(|| RgbPrimaries::REC709.matrix_to(RgbPrimaries::REC2020));
    *MATRIX * rec709.into()
}

/// The Bradford chromatic adaptation from white `src` to white `dst`, on CIE XYZ.
fn bradford(src: Chromaticity, dst: Chromaticity) -> DMat3 {
    let cone = DMat3::from_rows(
        DVec3::new(0.8951, 0.2664, -0.1614),
        DVec3::new(-0.7502, 1.7135, 0.0367),
        DVec3::new(0.0389, -0.0685, 1.0296),
    );
    adaptation(cone, src, dst)
}

/// Full adaptation from white `src` to white `dst` in the cone space `cone`, on CIE XYZ.
pub(crate) fn adaptation(cone: DMat3, src: Chromaticity, dst: Chromaticity) -> DMat3 {
    let gain = (cone * dst.to_xyz(1.0)) / (cone * src.to_xyz(1.0));
    cone.inverse() * DMat3::from_diagonal(gain) * cone
}

/// The primaries ids stored in cooked textures.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(rename_all = "snake_case")
)]
#[repr(u8)]
pub enum Primaries {
    Rec709 = 0,
    DisplayP3 = 1,
    Rec2020 = 2,
}

impl Primaries {
    /// The id stored in a cooked texture.
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// The primaries of a stored id, or `None` for an unknown id.
    pub const fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Rec709),
            1 => Some(Self::DisplayP3),
            2 => Some(Self::Rec2020),
            _ => None,
        }
    }

    /// The chromaticities of these primaries.
    pub const fn rgb(self) -> RgbPrimaries {
        match self {
            Self::Rec709 => RgbPrimaries::REC709,
            Self::DisplayP3 => RgbPrimaries::DISPLAY_P3,
            Self::Rec2020 => RgbPrimaries::REC2020,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SETS: [RgbPrimaries; 4] = [
        RgbPrimaries::REC709,
        RgbPrimaries::DISPLAY_P3,
        RgbPrimaries::REC2020,
        RgbPrimaries::ACES_CG,
    ];

    fn assert_close(a: DMat3, b: DMat3, tol: f64, what: &str) {
        assert!(a.abs_diff_eq(b, tol), "{what}: {a} vs {b}");
    }

    #[test]
    fn rec709_to_rec2020_matches_bt2087() {
        // ITU-R BT.2087-0, section 2.2.1, rows.
        let bt2087 = DMat3::from_cols_array_2d(&[
            [0.6274, 0.3293, 0.0433],
            [0.0691, 0.9195, 0.0114],
            [0.0164, 0.0880, 0.8956],
        ])
        .transpose();
        let m = RgbPrimaries::REC709.matrix_to(RgbPrimaries::REC2020);
        assert_close(m, bt2087, 5e-5, "BT.2087");
    }

    #[test]
    fn aces_cg_to_rec709_matches_aces() {
        // The ACES reference ACEScg to linear Rec.709 matrix (Bradford adaptation), rows.
        let aces = DMat3::from_cols_array_2d(&[
            [1.705_050_993, -0.621_792_121, -0.083_258_872],
            [-0.130_256_418, 1.140_804_737, -0.010_548_319],
            [-0.024_003_357, -0.128_968_976, 1.152_972_333],
        ])
        .transpose();
        let m = RgbPrimaries::ACES_CG.matrix_to(RgbPrimaries::REC709);
        assert_close(m, aces, 1e-6, "ACEScg to Rec.709");
    }

    #[test]
    fn matrices_round_trip_and_keep_white() {
        for a in SETS {
            assert_close(
                a.xyz_to_rgb() * a.rgb_to_xyz(),
                DMat3::IDENTITY,
                1e-14,
                "xyz",
            );
            let white = a.rgb_to_xyz() * DVec3::ONE;
            assert!(white.abs_diff_eq(a.white.to_xyz(1.0), 1e-14), "{white}");
            for b in SETS {
                let there_and_back = b.matrix_to(a) * a.matrix_to(b);
                assert_close(there_and_back, DMat3::IDENTITY, 1e-14, "rgb");
                let white = a.matrix_to(b) * DVec3::ONE;
                assert!(white.abs_diff_eq(DVec3::ONE, 1e-14), "{white}");
            }
        }
    }

    const SLANG: &str = include_str!("../shaders/orr_color/primaries.slang");

    fn named(name: &str) -> RgbPrimaries {
        match name {
            "REC709" => RgbPrimaries::REC709,
            "P3" => RgbPrimaries::DISPLAY_P3,
            "REC2020" => RgbPrimaries::REC2020,
            _ => panic!("{name}"),
        }
    }

    #[test]
    fn slang_matrices_are_the_rust_ones_rounded() {
        let decl = "public static const float3x3 ";
        let mut seen = 0;
        for (at, _) in SLANG.match_indices(decl) {
            let rest = &SLANG[at + decl.len()..];
            let name = &rest[..rest.find(' ').unwrap()];
            let open = rest.find('(').unwrap() + 1;
            let body = &rest[open..rest.find(')').unwrap()];
            let rows: Vec<f32> = body.split(',').map(|s| s.trim().parse().unwrap()).collect();
            let (src, dst) = name.split_once("_TO_").unwrap();
            let m = match (src, dst) {
                ("XYZ", d) => named(d).xyz_to_rgb(),
                (s, "XYZ") => named(s).rgb_to_xyz(),
                (s, d) => named(s).matrix_to(named(d)),
            };
            let want = m.transpose().to_cols_array().map(|v| v as f32);
            // Exact but for the zeros that f64 rounding leaves at about 1e-17.
            let same = rows.iter().zip(want).all(|(a, b)| (a - b).abs() <= 1e-12);
            assert!(same, "{name}: {rows:?} vs {want:?}");
            seen += 1;
        }
        assert_eq!(seen, 10);
        let d65 = Chromaticity::D65;
        let want = format!("D65_XY = float2({:?}, {:?});", d65.x as f32, d65.y as f32);
        assert!(SLANG.contains(&want), "{want}");
        for (name, p) in [
            ("REC709", Primaries::Rec709),
            ("DISPLAY_P3", Primaries::DisplayP3),
            ("REC2020", Primaries::Rec2020),
        ] {
            let want = format!("PRIMARIES_{name} = {};", p.id());
            assert!(SLANG.contains(&want), "{want}");
        }
    }
}
