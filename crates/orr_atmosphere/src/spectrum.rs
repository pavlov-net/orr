//! Spectral coefficients derived on first use from the ASTM G173 solar spectrum, ozone cross
//! sections at 233 K and the CIE 1931 2-deg observer.
//!
//! Three sets describe the same medium at different spectral resolution: [`SPECTRAL`]
//! (48 bins of 10 nm, 360-830 nm, each at its labeled wavelength), [`GROUPS`]
//! ([`SPECTRAL_GROUPS`] runs of adjacent bins, solar-weighted means) and [`RGB`] (the three
//! Rec.2020 channels, weighted by the solar spectrum times each channel's color matching
//! function). Group and band coefficients are means over their bins; their `rec2020` rows
//! sum to [`SOLAR_COLOR`].

use std::array::from_fn;
use std::sync::LazyLock;

use orr_color::{LUMINOUS_EFFICACY, RgbPrimaries};
use orr_math::DVec3;

use crate::spectrum_data::{COLOR_MATCHING, OZONE_CROSS_SECTION, SOLAR_IRRADIANCE};

/// Wavelength groups in the LUT passes; each is `BINS / SPECTRAL_GROUPS` adjacent 10 nm bins.
pub const SPECTRAL_GROUPS: usize = 12;
/// 10 nm bins of the spectral reference, 360-830 nm.
pub const BINS: usize = 48;
/// Entries in each band's aerosol table.
pub const ANGSTROM_STEPS: usize = 26;
/// Angstrom exponent between aerosol table entries.
pub const ANGSTROM_STEP: f64 = 0.1;

const _: () = assert!(
    BINS.is_multiple_of(SPECTRAL_GROUPS),
    "groups must split the bins evenly"
);
const _: () = assert!(
    SPECTRAL_GROUPS.is_multiple_of(3),
    "the LUT passes run three groups at a time"
);

const FIRST_NM: f64 = 360.0;
const BIN_NM: f64 = 10.0;
/// Rayleigh scattering of Earth's sea-level air at 1 um, 1/m (scales as lambda^-4).
const RAYLEIGH_1UM: f64 = 1.24062e-6;

/// Top-of-atmosphere solar illuminance from the spectrum, lux.
pub const SOLAR_ILLUMINANCE: f64 = {
    let mut sum = 0.0;
    let mut i = 0;
    while i < BINS {
        sum += LUMINOUS_EFFICACY * BIN_NM * SOLAR_IRRADIANCE[i] * COLOR_MATCHING[i][1];
        i += 1;
    }
    sum
};

/// Linear Rec.2020 color of sunlight above the atmosphere, luminance 1.
pub static SOLAR_COLOR: LazyLock<[f64; 3]> =
    LazyLock::new(|| from_fn(|c| SPECTRAL.rec2020.iter().map(|w| w[c]).sum()));

/// The 10 nm bins.
pub static SPECTRAL: LazyLock<Bands<BINS>> = LazyLock::new(|| {
    let rbar = rec2020_matching();
    Bands {
        wavelength: from_fn(wavelength),
        rayleigh: from_fn(rayleigh),
        ozone_cross_section: OZONE_CROSS_SECTION,
        aerosol: from_fn(|a| from_fn(|i| aerosol(i, a))),
        rec2020: from_fn(|i| {
            rbar[i]
                .map(|r| LUMINOUS_EFFICACY * BIN_NM * SOLAR_IRRADIANCE[i] * r / SOLAR_ILLUMINANCE)
        }),
    }
});

/// The wavelength groups of the LUT passes.
pub static GROUPS: LazyLock<Bands<SPECTRAL_GROUPS>> = LazyLock::new(|| {
    let per_group = BINS / SPECTRAL_GROUPS;
    let bins = |g: usize| g * per_group..(g + 1) * per_group;
    bands(
        |g, i| {
            if bins(g).contains(&i) {
                SOLAR_IRRADIANCE[i]
            } else {
                0.0
            }
        },
        |g| from_fn(|c| bins(g).map(|i| SPECTRAL.rec2020[i][c]).sum()),
    )
});

/// The three solar-weighted Rec.2020 bands (red, green, blue).
pub static RGB: LazyLock<Bands<3>> = LazyLock::new(|| {
    let rbar = rec2020_matching();
    bands(
        |c, i| SOLAR_IRRADIANCE[i] * rbar[i][c],
        |c| {
            let mut rec2020 = [0.0; 3];
            rec2020[c] = SOLAR_COLOR[c];
            rec2020
        },
    )
});

fn wavelength(i: usize) -> f64 {
    FIRST_NM + BIN_NM * i as f64
}

fn rayleigh(i: usize) -> f64 {
    RAYLEIGH_1UM / (wavelength(i) / 1000.0).powi(4)
}

fn aerosol(i: usize, a: usize) -> f64 {
    (wavelength(i) / 550.0).powf(-(a as f64 * ANGSTROM_STEP))
}

/// Rec.2020 color matching functions of the bins.
fn rec2020_matching() -> [[f64; 3]; BINS] {
    let xyz_to_rec2020 = RgbPrimaries::REC2020.xyz_to_rgb();
    COLOR_MATCHING.map(|c| (xyz_to_rec2020 * DVec3::from(c)).to_array())
}

/// Bands whose coefficients are means over the bins with `weight(band, bin)`.
fn bands<const N: usize>(
    weight: impl Fn(usize, usize) -> f64,
    rec2020: impl Fn(usize) -> [f64; 3],
) -> Bands<N> {
    let mean = |k: usize, f: &dyn Fn(usize) -> f64| {
        let total: f64 = (0..BINS).map(|i| weight(k, i)).sum();
        (0..BINS).map(|i| weight(k, i) * f(i)).sum::<f64>() / total
    };
    Bands {
        wavelength: from_fn(|k| {
            let total: f64 = (0..BINS).map(|i| weight(k, i).max(0.0)).sum();
            (0..BINS)
                .map(|i| weight(k, i).max(0.0) * wavelength(i))
                .sum::<f64>()
                / total
        }),
        rayleigh: from_fn(|k| mean(k, &rayleigh)),
        ozone_cross_section: from_fn(|k| mean(k, &|i| OZONE_CROSS_SECTION[i])),
        aerosol: from_fn(|a| from_fn(|k| mean(k, &|i| aerosol(i, a)))),
        rec2020: from_fn(rec2020),
    }
}

/// Per-band coefficients of the medium.
#[derive(Clone, Copy, Debug)]
pub struct Bands<const N: usize> {
    /// Center wavelength, nm: the bin's wavelength, or the weighted mean over a band.
    pub wavelength: [f64; N],
    /// Rayleigh scattering of Earth's sea-level air, 1/m.
    pub rayleigh: [f64; N],
    /// Ozone absorption cross section, m^2.
    pub ozone_cross_section: [f64; N],
    /// Mean of `(lambda / 550 nm)^-alpha` over the band, at `alpha = i * ANGSTROM_STEP`.
    pub aerosol: [[f64; N]; ANGSTROM_STEPS],
    /// The band's share of a solar-colored light, linear Rec.2020 per lux of
    /// top-of-atmosphere illuminance.
    pub rec2020: [[f64; 3]; N],
}

impl<const N: usize> Bands<N> {
    /// Band `k`'s aerosol extinction relative to 550 nm for Angstrom exponent `alpha`,
    /// interpolated linearly in the table and clamped to its range (0 to 2.5).
    pub(crate) fn aerosol_factor(&self, k: usize, alpha: f64) -> f64 {
        let x = (alpha / ANGSTROM_STEP).clamp(0.0, (ANGSTROM_STEPS - 1) as f64);
        let i = (x as usize).min(ANGSTROM_STEPS - 2);
        let f = x - i as f64;
        self.aerosol[i][k] * (1.0 - f) + self.aerosol[i + 1][k] * f
    }
}

/// A Rec.2020 color's value at wavelength `lambda` (nm): linear between the [`RGB`]
/// bands' center wavelengths, constant beyond the outer two. At a band's center it is that
/// channel's value.
///
/// Colors given per channel (an albedo, or a light's color relative to
/// [`SOLAR_COLOR`]) enter a group or spectral run through this.
pub(crate) fn rgb_at(lambda: f64, rgb: [f64; 3]) -> f64 {
    let [lr, lg, lb] = RGB.wavelength;
    if lambda <= lb {
        rgb[2]
    } else if lambda <= lg {
        rgb[2] + (rgb[1] - rgb[2]) * (lambda - lb) / (lg - lb)
    } else if lambda <= lr {
        rgb[1] + (rgb[0] - rgb[1]) * (lambda - lg) / (lr - lg)
    } else {
        rgb[0]
    }
}
