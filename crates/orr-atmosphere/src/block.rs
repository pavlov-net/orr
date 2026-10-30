//! An [`Atmosphere`] as the GPU reads it: the medium fields of [`AtmosphereGpu`] and the
//! per-group coefficients of the LUT passes.

use glam::{DVec3, Vec3};
use orr::WorldPos;

use crate::colour::{REC709_TO_REC2020, REC2020_LUMINANCE, mul};
use crate::shaders::types::{AtmosphereGpu, AtmosphereGroups};
use crate::spectrum::{GROUPS, RGB, SOLAR_COLOUR, SPECTRAL_GROUPS, rgb_at};
use crate::{Atmosphere, DOBSON_UNIT};

/// LUT passes per frame: the wavelength groups three at a time.
pub(crate) const PASSES: usize = SPECTRAL_GROUPS / 3;

/// A linear Rec.709 colour in linear Rec.2020.
pub(crate) fn rec2020(c: Vec3) -> [f64; 3] {
    mul(&REC709_TO_REC2020, c.as_dvec3().to_array())
}

fn xyz(v: [f64; 3]) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, 0.0]
}

/// Night background luminance at the ground zenith with the Earth defaults, cd/m^2.
const NIGHT_ZENITH: f64 = 2e-4;

/// Ozone molecules per m^3 at the tent's peak.
pub(crate) fn ozone_peak(a: &Atmosphere) -> f64 {
    f64::from(a.ozone.column) * DOBSON_UNIT / f64::from(a.ozone.half_width)
}

/// Transmittance from the ground to space straight up, per Rec.2020 channel of sunlight, as
/// the transmittance LUT's Rec.2020 layer holds it. Fog is left out.
fn zenith_transmittance(a: &Atmosphere) -> [f64; 3] {
    let thickness = f64::from(a.top_radius - a.planet_radius);
    let column = |scale: f32| {
        let h = f64::from(scale);
        h * (1.0 - (-thickness / h).exp())
    };
    let rayleigh = f64::from(a.rayleigh.density) * column(a.rayleigh.scale_height);
    let aerosol = f64::from(a.aerosol.optical_depth_550) * column(a.aerosol.scale_height)
        / f64::from(a.aerosol.scale_height);
    let ozone = f64::from(a.ozone.column) * DOBSON_UNIT;
    let alpha = f64::from(a.aerosol.angstrom_exponent);
    std::array::from_fn(|c| {
        (0..SPECTRAL_GROUPS)
            .map(|k| {
                let depth = GROUPS.rayleigh[k] * rayleigh
                    + GROUPS.aerosol_factor(k, alpha) * aerosol
                    + GROUPS.ozone_cross_section[k] * ozone;
                GROUPS.rec2020[k][c] * (-depth).exp()
            })
            .sum::<f64>()
            / SOLAR_COLOUR[c]
    })
}

/// Zenith radiance of the night background before extinction, nits: the value that gives
/// [`NIGHT_ZENITH`] at the ground under the Earth defaults.
fn night_zenith() -> f64 {
    let t = zenith_transmittance(&Atmosphere::earth(DVec3::ZERO, Vec3::Y));
    NIGHT_ZENITH / (0..3).map(|c| REC2020_LUMINANCE[c] * t[c]).sum::<f64>()
}

/// Each group's coefficients, three groups per pass.
pub(crate) fn groups(a: &Atmosphere) -> [AtmosphereGroups; PASSES] {
    let alpha = f64::from(a.aerosol.angstrom_exponent);
    let aod = f64::from(a.aerosol.optical_depth_550);
    let ssa = f64::from(a.aerosol.single_scattering_albedo);
    let density = f64::from(a.rayleigh.density);
    let ozone = ozone_peak(a);
    let ground = rec2020(a.ground_albedo);
    let fog = a.height_fog.map_or([0.0; 3], |f| rec2020(f.albedo));
    std::array::from_fn(|p| {
        let per = |f: &dyn Fn(usize) -> f64| xyz(std::array::from_fn(|j| f(3 * p + j)));
        AtmosphereGroups {
            rayleigh: per(&|k| GROUPS.rayleigh[k] * density),
            aerosol_extinction: per(&|k| aod * GROUPS.aerosol_factor(k, alpha)),
            aerosol_scattering: per(&|k| aod * GROUPS.aerosol_factor(k, alpha) * ssa),
            ozone: per(&|k| GROUPS.ozone_cross_section[k] * ozone),
            fog_albedo: per(&|k| rgb_at(GROUPS.wavelength[k], fog)),
            ground_albedo: per(&|k| rgb_at(GROUPS.wavelength[k], ground)),
            rec2020: std::array::from_fn(|c| per(&|k| GROUPS.rec2020[k][c])),
            // A light's colour over the solar colour, carried into the group as `rgb_at` does.
            light_weight: std::array::from_fn(|j| {
                let k = 3 * p + j;
                xyz(std::array::from_fn(|c| {
                    let mut unit = [0.0; 3];
                    unit[c] = 1.0;
                    rgb_at(GROUPS.wavelength[k], unit) / SOLAR_COLOUR[c]
                }))
            }),
        }
    })
}

/// Writes the fields of `out` that describe the medium.
pub(crate) fn medium(a: &Atmosphere, out: &mut AtmosphereGpu) {
    let top = WorldPos::new(a.planet_top, 0);
    let alpha = f64::from(a.aerosol.angstrom_exponent);
    let aod = f64::from(a.aerosol.optical_depth_550);
    let density = f64::from(a.rayleigh.density);
    let ozone = ozone_peak(a);
    let band = |f: &dyn Fn(usize) -> f64| xyz(std::array::from_fn(f));
    out.top_metres = top.cell.extend(0).to_array();
    out.top_offset = top.offset.extend(0.0).to_array();
    out.up = a.up.normalize().extend(0.0).to_array();
    out.solar_colour = xyz(SOLAR_COLOUR);
    out.rayleigh = band(&|c| RGB.rayleigh[c] * density);
    out.aerosol_extinction = band(&|c| aod * RGB.aerosol_factor(c, alpha));
    out.aerosol_scattering = band(&|c| {
        aod * RGB.aerosol_factor(c, alpha) * f64::from(a.aerosol.single_scattering_albedo)
    });
    out.ozone = band(&|c| RGB.ozone_cross_section[c] * ozone);
    out.fog_albedo = xyz(a.height_fog.map_or([0.0; 3], |f| rec2020(f.albedo)));
    out.planet_radius = a.planet_radius;
    out.top_radius = a.top_radius;
    out.rayleigh_scale = a.rayleigh.scale_height;
    out.aerosol_scale = a.aerosol.scale_height;
    out.aerosol_asymmetry = a.aerosol.asymmetry;
    out.ozone_peak = a.ozone.peak_altitude;
    out.ozone_half_width = a.ozone.half_width;
    out.night_background = if a.night_background {
        night_zenith() as f32
    } else {
        0.0
    };
    let fog = a.height_fog.filter(|f| f.density > 0.0);
    out.fog_density = fog.map_or(0.0, |f| f.density);
    out.fog_base = fog.map_or(0.0, |f| f.base_height);
    out.fog_falloff = fog.map_or(1.0, |f| f.falloff_height);
    out.fog_diameter = fog.map_or(10.0, |f| f.droplet_diameter);
    out.sky_lights = a.sky_lights;
}
