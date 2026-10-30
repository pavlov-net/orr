//! Planetary sky and atmosphere: Rayleigh, aerosol and ozone over a planet, with an
//! optional height-fog layer.
//!
//! [`Atmosphere`] describes the medium. [`AtmosphereState`] holds one atmosphere's GPU
//! state: its LUTs, recorded at the scene compute point, and the [`AtmosphereGpu`] block
//! the Slang module `atmosphere` reads (`DEP_ORR_ATMOSPHERE_SHADER_DIRS`), whose
//! `AtmosphereSky` links into orr's kernels as the environment. [`presets`] holds example
//! lights, [`spectrum`] the build-time spectral coefficients and
//! [`reference`](mod@reference) an f64 spectral integrator that the GPU passes are checked
//! against.

mod block;
pub mod colour;
mod luts;
pub mod presets;
pub mod reference;
pub mod spectrum;

pub use luts::{AtmosphereState, MULTIPLE_SCATTERING_SIZE, SKY_VIEW_SIZE, TRANSMITTANCE_SIZE};
pub use shaders::types::{AtmosphereGpu, AtmosphereGroups, FoldGpu};

use glam::{DVec3, Vec3};

/// Generated kernel modules and the Rust types of the crate's Slang structs.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// Molecules per m^2 in one Dobson unit.
pub(crate) const DOBSON_UNIT: f64 = 2.687e20;

/// Directional lights a sky can choose its folded lights from.
pub const MAX_SKY_LIGHTS: usize = 16;
/// Lights folded into one sky-view LUT, fading lights included.
pub const MAX_FOLDED_LIGHTS: u32 = 8;
/// Sky-view LUTs beyond the focus view's.
pub const MAX_EXTRA_VIEWS: u32 = 4;

/// A planet's atmosphere.
///
/// Altitude is height above the flat ground plane through `planet_top` with `up` as the
/// zenith. The froxel range the sky marches from is the scene's `near_range`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atmosphere {
    /// World point where the flat ground plane touches the planet.
    pub planet_top: DVec3,
    /// World zenith, unit length.
    pub up: Vec3,
    /// Planet radius, m.
    pub planet_radius: f32,
    /// Radius of the top of the atmosphere, m.
    pub top_radius: f32,
    /// Ground albedo of the virtual planet below the horizon, linear Rec.709.
    pub ground_albedo: Vec3,
    pub rayleigh: Rayleigh,
    pub aerosol: Aerosol,
    pub ozone: Ozone,
    pub height_fog: Option<HeightFog>,
    /// Airglow, zodiacal light and starlight in the sky: 2e-4 cd/m^2 at the ground zenith with
    /// the Earth defaults, dimmer through a denser medium.
    pub night_background: bool,
    /// Lights folded into the sky-view LUT, `1..=MAX_FOLDED_LIGHTS`.
    pub sky_lights: u32,
}

impl Atmosphere {
    /// Earth: radius 6360 km, top 6460 km, ground albedo 0.1, the default Rayleigh,
    /// aerosol and ozone, no height fog, night background on, 4 sky lights.
    pub fn earth(planet_top: DVec3, up: Vec3) -> Self {
        Atmosphere {
            planet_top,
            up,
            planet_radius: 6_360_000.0,
            top_radius: 6_460_000.0,
            ground_albedo: Vec3::splat(0.1),
            rayleigh: Rayleigh::default(),
            aerosol: Aerosol::default(),
            ozone: Ozone::default(),
            height_fog: None,
            night_background: true,
            sky_lights: 4,
        }
    }
}

/// Rayleigh scattering by air molecules, exponential in altitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rayleigh {
    /// Density at the ground relative to Earth's sea-level air.
    pub density: f32,
    /// Scale height, m.
    pub scale_height: f32,
}

impl Default for Rayleigh {
    /// Earth: density 1, scale height 8 km.
    fn default() -> Self {
        Rayleigh {
            density: 1.0,
            scale_height: 8000.0,
        }
    }
}

/// Aerosol, exponential in altitude, with a Cornette-Shanks phase function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aerosol {
    /// Vertical optical depth at 550 nm.
    pub optical_depth_550: f32,
    /// Angstrom exponent: extinction scales as `(lambda / 550 nm)^-angstrom_exponent`.
    /// The spectral tables cover 0 to 2.5.
    pub angstrom_exponent: f32,
    pub single_scattering_albedo: f32,
    /// Scale height, m.
    pub scale_height: f32,
    /// Cornette-Shanks asymmetry `g`.
    pub asymmetry: f32,
}

impl Default for Aerosol {
    /// Global mean aerosol (MACv2, Kinne 2019): optical depth 0.12, Angstrom exponent 0.8
    /// (450-550 nm), single-scattering albedo 0.94, asymmetry 0.7, scale height 1.2 km.
    fn default() -> Self {
        Aerosol::new(0.12, 0.8, 0.94, 0.7, 1200.0)
    }
}

/// OPAC aerosol types at 80% relative humidity (Hess 1998, Table 3), with the scale height
/// that matches the OPAC layered profile's ground extinction at the same optical depth.
impl Aerosol {
    const fn new(aod: f32, angstrom: f32, ssa: f32, g: f32, scale_height: f32) -> Self {
        Aerosol {
            optical_depth_550: aod,
            angstrom_exponent: angstrom,
            single_scattering_albedo: ssa,
            scale_height,
            asymmetry: g,
        }
    }

    pub const fn clean_continental() -> Self {
        Aerosol::new(0.064, 1.42, 0.972, 0.709, 2460.0)
    }

    pub const fn continental_average() -> Self {
        Aerosol::new(0.151, 1.42, 0.925, 0.703, 2010.0)
    }

    pub const fn urban() -> Self {
        Aerosol::new(0.643, 1.43, 0.817, 0.689, 1820.0)
    }

    pub const fn desert() -> Self {
        Aerosol::new(0.286, 0.17, 0.888, 0.729, 1980.0)
    }

    pub const fn maritime_clean() -> Self {
        Aerosol::new(0.096, 0.08, 0.997, 0.772, 1060.0)
    }
}

/// Ozone absorption, a tent in altitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ozone {
    /// Vertical column, Dobson units.
    pub column: f32,
    /// Altitude of peak density, m.
    pub peak_altitude: f32,
    /// Distance from the peak to where the density reaches zero, m.
    pub half_width: f32,
}

impl Default for Ozone {
    /// 300 DU, peak at 25 km, zero below 10 km and above 40 km.
    fn default() -> Self {
        Ozone {
            column: 300.0,
            peak_altitude: 25_000.0,
            half_width: 15_000.0,
        }
    }
}

/// A layer of water droplets whose density falls exponentially with altitude:
/// `density * exp(-(altitude - base_height) / falloff_height)`. Extinction is the same at
/// every wavelength; the phase function is the Jendersie-d'Eon HG plus Draine fit for the
/// droplet diameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightFog {
    /// Extinction at `base_height`, 1/m.
    pub density: f32,
    /// Altitude over which the density falls by e, m.
    pub falloff_height: f32,
    /// Altitude above the ground plane where the extinction is `density`, m.
    pub base_height: f32,
    /// Single-scattering albedo, linear Rec.709.
    pub albedo: Vec3,
    /// Mean droplet diameter, um; the phase fit covers 5 to 50.
    pub droplet_diameter: f32,
}
