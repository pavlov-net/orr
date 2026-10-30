//! Example lights for a planet's sky. They are plain [`DirectionalLight`] values: the scene
//! takes them through `SceneWriter::set_light`, and the atmosphere folds in whichever are
//! among its candidates.

use glam::Vec3;
use orr::{DirectionalLight, DiscProfile};

use crate::colour::{REC2020_TO_REC709, mul};
use crate::spectrum::{SOLAR_COLOUR, SOLAR_ILLUMINANCE};

/// Angular radius of the solar disc, rad.
pub const SOLAR_ANGULAR_RADIUS: f32 = 4.652e-3;

/// A sun: a limb-darkened disc of [`SOLAR_ANGULAR_RADIUS`] giving the solar spectrum's
/// top-of-atmosphere illuminance (132.6 klx) in its colour.
pub fn solar_disc(direction_to_light: Vec3) -> DirectionalLight {
    let rec709 = mul(
        &REC2020_TO_REC709,
        SOLAR_COLOUR.map(|c| c * SOLAR_ILLUMINANCE),
    );
    DirectionalLight {
        direction_to_light: direction_to_light.normalize(),
        illuminance: Vec3::from(rec709.map(|c| c as f32)),
        angular_radius: SOLAR_ANGULAR_RADIUS,
        disc: DiscProfile::LimbDarkened,
    }
}
