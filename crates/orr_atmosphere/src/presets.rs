//! Example lights for a planet's sky. They are plain [`DirectionalLight`] values: the scene
//! takes them through `SceneWriter::set_light`, and the atmosphere folds in whichever are
//! among its candidates.

use std::f64::consts::PI;

use orr::abi::TextureIndex;
use orr::{DirectionalLight, DiscProfile, LightId};
use orr_color::{RgbPrimaries, luminance};
use orr_math::{DVec3, Vec3};

use crate::spectrum::{SOLAR_COLOR, SOLAR_ILLUMINANCE, SPECTRAL};

/// Angular radius of the solar disc, rad.
pub const SOLAR_ANGULAR_RADIUS: f32 = 4.652e-3;

/// A sun: a limb-darkened disc of [`SOLAR_ANGULAR_RADIUS`] giving the solar spectrum's
/// top-of-atmosphere illuminance (132.6 klx) in its color.
pub fn solar_disc(direction_to_light: Vec3) -> DirectionalLight {
    let to_rec709 = RgbPrimaries::REC2020.matrix_to(RgbPrimaries::REC709);
    let rec709 = to_rec709 * (DVec3::from(*SOLAR_COLOR) * SOLAR_ILLUMINANCE);
    DirectionalLight {
        direction_to_light: direction_to_light.normalize(),
        illuminance: rec709.as_vec3(),
        angular_radius: SOLAR_ANGULAR_RADIUS,
        disc: DiscProfile::LimbDarkened,
    }
}

/// Angular radius of the lunar disc, rad.
pub const LUNAR_ANGULAR_RADIUS: f32 = 4.52e-3;

/// Mean albedo of a lunar albedo texture: with a zero `illuminance` the Lambertian disc it
/// gives matches [`lunar_phase`] at a phase angle of 5 deg (0.30 lx under [`solar_disc`]).
pub const LUNAR_ALBEDO: f64 = 0.167;

/// The Moon's reflectance at `lambda` nm relative to 590 nm (ROLO's ramp, 1.61 from 400 to
/// 700 nm). Moonlight is the solar spectrum times this.
pub fn lunar_reflectance(lambda: f64) -> f64 {
    1.0 + 1.47e-3 * (lambda - 590.0)
}

/// Moonlight's color, linear Rec.2020 per lux (luminance 1).
pub fn lunar_color() -> [f64; 3] {
    let mut c = [0.0; 3];
    for (w, lambda) in SPECTRAL.rec2020.iter().zip(SPECTRAL.wavelength) {
        for (c, w) in c.iter_mut().zip(w) {
            *c += w * lunar_reflectance(lambda);
        }
    }
    let y = luminance(c);
    c.map(|c| c / y)
}

/// A lunar albedo texture's color, linear Rec.2020 as a `Textured` disc reads it: mean
/// [`LUNAR_ALBEDO`] under sunlight, colored so a sunlit disc emits [`lunar_color`].
pub fn lunar_albedo() -> [f64; 3] {
    let c = lunar_color();
    std::array::from_fn(|i| LUNAR_ALBEDO * c[i] / SOLAR_COLOR[i])
}

/// The Moon's disc-equivalent reflectance at phase angle `alpha` (deg): Krisciunas and
/// Schaefer's (1991) phase law, valid above 7 deg, times an opposition surge fitted to ROLO
/// nearer full. Within 1% of ROLO at 1.5-7 deg and 3% at 1.5-120 deg; a quarter moon is 0.09
/// of full.
pub fn lunar_phase(alpha: f64) -> f64 {
    let a = alpha.abs();
    0.1156 * 10f64.powf(-0.4 * (0.026 * a + 4e-9 * a.powi(4))) * (1.0 + 0.37 * (-a / 3.0).exp())
}

/// A moon: a disc of [`LUNAR_ANGULAR_RADIUS`] showing `albedo` (oriented by `up`) lit by
/// the `lit_by` lights, at most four. Its illuminance follows [`lunar_phase`] for each
/// `lit_by` light, so the texture sets only the disc's pattern and the total stays right at
/// every phase; call it again when a light moves. `lit_by` lights that are themselves
/// `Textured` light nothing, as in orr's disc.
pub fn lunar_disc(
    direction_to_light: Vec3,
    albedo: TextureIndex,
    up: Vec3,
    lit_by: &[(LightId, &DirectionalLight)],
) -> DirectionalLight {
    assert!(lit_by.len() <= 4, "a disc is lit by at most four lights");
    let direction = direction_to_light.normalize();
    let r = f64::from(LUNAR_ANGULAR_RADIUS);
    let solid_angle = 2.0 * PI * (1.0 - r.cos());
    let mut ids = [LightId::NONE; 4];
    let mut lux = 0.0;
    for (slot, (id, light)) in ids.iter_mut().zip(lit_by) {
        *slot = *id;
        if matches!(light.disc, DiscProfile::Textured { .. }) {
            continue;
        }
        let e = luminance(crate::block::rec2020(light.illuminance));
        let cos = f64::from(direction.dot(light.direction_to_light.normalize()));
        let alpha = 180.0 - cos.clamp(-1.0, 1.0).acos().to_degrees();
        lux += e * solid_angle * lunar_phase(alpha) / PI;
    }
    let to_rec709 = RgbPrimaries::REC2020.matrix_to(RgbPrimaries::REC709);
    let rec709 = to_rec709 * (DVec3::from(lunar_color()) * lux);
    DirectionalLight {
        direction_to_light: direction,
        illuminance: rec709.as_vec3(),
        angular_radius: LUNAR_ANGULAR_RADIUS,
        disc: DiscProfile::Textured {
            albedo,
            up,
            lit_by: ids,
        },
    }
}
