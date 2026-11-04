//! Chromaticities: CIE 1931 xy, CIE 1960 uv, CIE 1976 u'v', correlated color temperature and
//! Duv, and CAT16 adaptation between white points.

use orr_math::{DMat3, DVec2, DVec3};

use crate::RgbPrimaries;
use crate::primaries::adaptation;

/// A CIE 1931 xy chromaticity.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Chromaticity {
    pub x: f64,
    pub y: f64,
}

impl Chromaticity {
    /// CIE standard illuminant D65, the white of Rec.709, Display P3 and Rec.2020.
    pub const D65: Self = Self::new(0.3127, 0.3290);
    /// The ACES white, close to CIE illuminant D60.
    pub const D60: Self = Self::new(0.32168, 0.33767);

    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// The chromaticity of a CIE XYZ color.
    pub fn from_xyz(xyz: DVec3) -> Self {
        let s = xyz.element_sum();
        Self::new(xyz.x / s, xyz.y / s)
    }

    /// CIE XYZ of this chromaticity at luminance (Y) `luminance`.
    pub const fn to_xyz(self, luminance: f64) -> DVec3 {
        let s = luminance / self.y;
        DVec3::new(self.x * s, luminance, (1.0 - self.x - self.y) * s)
    }

    /// CIE 1960 uv.
    pub fn uv_1960(self) -> DVec2 {
        DVec2::new(4.0 * self.x, 6.0 * self.y) / (-2.0 * self.x + 12.0 * self.y + 3.0)
    }

    /// The chromaticity of CIE 1960 `uv`.
    pub fn from_uv_1960(uv: DVec2) -> Self {
        let d = 2.0 * uv.x - 8.0 * uv.y + 4.0;
        Self::new(3.0 * uv.x / d, 2.0 * uv.y / d)
    }

    /// Correlated color temperature in kelvin by McCamy's cubic. Chromaticities at or below
    /// y = 0.1858 count as the hot end; the cubic turns at about 1620 K, so redder ones read
    /// 1620 K.
    pub fn cct(self) -> f64 {
        let n = if self.y > 0.1858 {
            ((self.x - 0.3320) / (0.1858 - self.y)).max(-1.28)
        } else {
            4.0
        };
        ((449.0 * n + 3525.0) * n + 6823.3) * n + 5520.33
    }

    /// The Planckian locus by Kang et al. (2002), for 1667-25000 K; clamped outside.
    pub fn planckian(cct: f64) -> Self {
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
        Self::new(x, y)
    }

    /// Distance in CIE 1960 uv from the locus point at `cct`, along the locus normal: positive
    /// above the locus (toward green), negative below.
    pub fn duv(self, cct: f64) -> f64 {
        (self.uv_1960() - Self::planckian(cct).uv_1960()).dot(planckian_normal_uv_1960(cct))
    }

    /// The chromaticity `duv` from the locus point at `cct`, along the locus normal in CIE 1960
    /// uv.
    pub fn from_cct_duv(cct: f64, duv: f64) -> Self {
        Self::from_uv_1960(Self::planckian(cct).uv_1960() + duv * planckian_normal_uv_1960(cct))
    }
}

impl From<[f64; 2]> for Chromaticity {
    fn from([x, y]: [f64; 2]) -> Self {
        Self::new(x, y)
    }
}

impl From<[f32; 2]> for Chromaticity {
    fn from([x, y]: [f32; 2]) -> Self {
        Self::new(x.into(), y.into())
    }
}

impl From<Chromaticity> for [f64; 2] {
    fn from(c: Chromaticity) -> Self {
        [c.x, c.y]
    }
}

impl From<Chromaticity> for [f32; 2] {
    fn from(c: Chromaticity) -> Self {
        [c.x as f32, c.y as f32]
    }
}

/// The unit normal of the Planckian locus in CIE 1960 uv at `cct`, pointing above it (toward
/// green). Outside the locus's range it is the normal at the end, where the central difference
/// still spans two points.
fn planckian_normal_uv_1960(cct: f64) -> DVec2 {
    let t = cct.clamp(1667.0 / 0.99, 25000.0 / 1.01);
    let d =
        Chromaticity::planckian(t * 1.01).uv_1960() - Chromaticity::planckian(t * 0.99).uv_1960();
    DVec2::new(d.y, -d.x).normalize()
}

/// CIE 1976 u'v' chromaticity of a CIE XYZ color.
pub fn uv_1976(xyz: impl Into<DVec3>) -> DVec2 {
    let xyz = xyz.into();
    let d = xyz.x + 15.0 * xyz.y + 3.0 * xyz.z;
    DVec2::new(4.0 * xyz.x / d, 9.0 * xyz.y / d)
}

/// Distance between the CIE 1976 u'v' chromaticities of two CIE XYZ colors.
pub fn delta_uv_1976(a: impl Into<DVec3>, b: impl Into<DVec3>) -> f64 {
    uv_1976(a).distance(uv_1976(b))
}

/// CAT16 (Li et al. 2017) from XYZ to its sharpened cone space.
const XYZ_TO_CAT16: DMat3 = DMat3::from_cols(
    DVec3::new(0.401288, -0.250268, -0.002079),
    DVec3::new(0.650173, 1.204414, 0.048952),
    DVec3::new(-0.051461, 0.045854, 0.953127),
);

/// CAT16 with full adaptation, as a matrix on linear Rec.2020 that maps white `from` to white
/// `to` at the same luminance.
pub fn cat16_rec2020(from: Chromaticity, to: Chromaticity) -> DMat3 {
    let p = RgbPrimaries::REC2020;
    p.xyz_to_rgb() * adaptation(XYZ_TO_CAT16, from, to) * p.rgb_to_xyz()
}

/// The CIE 1931 2-deg color matching functions at `nm`, by Wyman, Sloan and Shirley's
/// multi-lobe fit.
pub fn cie_xyz(nm: f64) -> DVec3 {
    let g = |mean: f64, below: f64, above: f64| {
        let t = (nm - mean) / if nm < mean { below } else { above };
        (-0.5 * t * t).exp()
    };
    DVec3::new(
        1.056 * g(599.8, 37.9, 31.0) + 0.362 * g(442.0, 16.0, 26.7) - 0.065 * g(501.1, 20.4, 26.2),
        0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1),
        1.217 * g(437.0, 11.8, 36.0) + 0.681 * g(459.0, 26.0, 13.8),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locus_reads_known_illuminants() {
        // CIE A (2856 K) and D65 (6504 K, about 0.0032 above the locus).
        let a = Chromaticity::new(0.44757, 0.40745);
        let cct = a.cct();
        let duv = a.duv(cct);
        assert!(
            (cct - 2856.0).abs() < 2.0 && duv.abs() < 5e-4,
            "{cct} K, {duv}"
        );
        let d65 = Chromaticity::D65;
        let cct = d65.cct();
        let duv = d65.duv(cct);
        assert!((cct - 6504.0).abs() < 5.0, "{cct} K");
        assert!((duv - 0.0032).abs() < 3e-4, "{duv}");
        let back = Chromaticity::from_cct_duv(cct, duv);
        assert!((back.x - d65.x).abs() < 3e-4 && (back.y - d65.y).abs() < 3e-4);
    }

    #[test]
    fn slang_cat16_is_the_rust_one_and_its_inverse() {
        let slang = include_str!("chromaticity.slang");
        let matrix = |name: &str| {
            let at = slang.find(&format!("{name} = float3x3(")).unwrap();
            let body = &slang[at + name.len() + " = float3x3(".len()..];
            let rows: Vec<f64> = body[..body.find(')').unwrap()]
                .split(',')
                .map(|s| s.trim().parse().unwrap())
                .collect();
            DMat3::from_cols_slice(&rows).transpose()
        };
        assert_eq!(matrix("XYZ_TO_CAT16"), XYZ_TO_CAT16);
        assert!(matrix("CAT16_TO_XYZ").abs_diff_eq(XYZ_TO_CAT16.inverse(), 1e-9));
    }
}
