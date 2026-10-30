//! Spectral coefficients derived at build time from the ASTM G173 solar spectrum, ozone
//! cross sections at 233 K and the CIE 1931 2-deg observer.
//!
//! Three sets describe the same medium at different spectral resolution: [`SPECTRAL`]
//! (48 bins of 10 nm, 360-830 nm, each at its labelled wavelength), [`GROUPS`]
//! ([`SPECTRAL_GROUPS`] runs of adjacent bins, solar-weighted means) and [`RGB`] (the three
//! Rec.2020 channels, weighted by the solar spectrum times each channel's colour matching
//! function). Group and band coefficients are means over their bins; their `rec2020` rows
//! sum to [`SOLAR_COLOUR`].

include!(concat!(env!("OUT_DIR"), "/spectrum.rs"));

/// Per-band coefficients of the medium.
#[derive(Clone, Copy, Debug)]
pub struct Bands<const N: usize> {
    /// Centre wavelength, nm: the bin's wavelength, or the weighted mean over a band.
    pub wavelength: [f64; N],
    /// Rayleigh scattering of Earth's sea-level air, 1/m.
    pub rayleigh: [f64; N],
    /// Ozone absorption cross section, m^2.
    pub ozone_cross_section: [f64; N],
    /// Mean of `(lambda / 550 nm)^-alpha` over the band, at `alpha = i * ANGSTROM_STEP`.
    pub aerosol: [[f64; N]; ANGSTROM_STEPS],
    /// The band's share of a solar-coloured light, linear Rec.2020 per lux of
    /// top-of-atmosphere illuminance.
    pub rec2020: [[f64; 3]; N],
}

impl<const N: usize> Bands<N> {
    /// Band `k`'s aerosol extinction relative to 550 nm for Angstrom exponent `alpha`,
    /// interpolated linearly in the table and clamped to its range (0 to 2.5).
    pub fn aerosol_factor(&self, k: usize, alpha: f64) -> f64 {
        let x = (alpha / ANGSTROM_STEP).clamp(0.0, (ANGSTROM_STEPS - 1) as f64);
        let i = (x as usize).min(ANGSTROM_STEPS - 2);
        let f = x - i as f64;
        self.aerosol[i][k] * (1.0 - f) + self.aerosol[i + 1][k] * f
    }
}

/// A Rec.2020 colour's value at wavelength `lambda` (nm): linear between the [`RGB`]
/// bands' centre wavelengths, constant beyond the outer two.
///
/// Colours given per channel (an albedo, or a light's colour relative to
/// [`SOLAR_COLOUR`]) enter a group or spectral run through this.
pub fn rgb_at(lambda: f64, rgb: [f64; 3]) -> f64 {
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
