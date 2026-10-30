//! Directional lights and their GPU records.

use glam::{Mat3, Vec3};
use orr_gpu::TextureIndex;

use crate::shaders::types::DirectionalLightGpu;

/// A light at infinity with a finite disc. Colours are linear Rec.709.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DirectionalLight {
    /// Unit direction from a receiver toward the light.
    pub direction_to_light: Vec3,
    /// Illuminance before the scene's medium, lux.
    pub illuminance: Vec3,
    /// Half angle of the disc, radians, > 0.
    pub angular_radius: f32,
    pub disc: DiscProfile,
}

/// How radiance spreads over a directional light's disc. Every profile's radiance,
/// integrated over the disc against the cosine to its centre, gives the light's
/// illuminance, except a [`DiscProfile::Textured`] disc with zero illuminance.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DiscProfile {
    /// Constant radiance.
    Uniform,
    /// Solar limb darkening (Neckel and Labs 1994).
    LimbDarkened,
    /// A sphere lit by other directional lights, showing phases: radiance at a disc point is
    /// `albedo x sum_i E_i x max(0, n.l_i) / pi`, with `n` the sphere normal there and `E_i`,
    /// `l_i` each lighting light's illuminance and direction. With zero illuminance the disc
    /// emits that integral; otherwise it keeps the lit shape and is scaled per channel to
    /// emit its illuminance, and a disc that receives no light emits nothing.
    Textured {
        /// Albedo viewed face on; `TextureIndex::NONE` is albedo 1.
        albedo: TextureIndex,
        /// Orients the albedo texture: its top row lies toward `up` on the disc.
        up: Vec3,
        /// Indices in the same light table of the `Uniform` or `LimbDarkened` lights that
        /// light the disc.
        lit_by: [Option<u32>; 4],
    },
}

/// Linear Rec.709 to linear Rec.2020 (ITU-R BT.2087).
const REC709_TO_REC2020: Mat3 = Mat3::from_cols_array(&[
    0.627_403_9,
    0.069_097_29,
    0.016_391_44,
    0.329_283_04,
    0.919_540_4,
    0.088_013_31,
    0.043_313_066,
    0.011_362_316,
    0.895_595_3,
]);

impl DirectionalLight {
    /// The record lighting's kernels read, in Rec.2020.
    pub fn gpu(&self) -> DirectionalLightGpu {
        let (disc, albedo, up, lit_by) = match self.disc {
            DiscProfile::Uniform => (0, TextureIndex::NONE, Vec3::ZERO, [u32::MAX; 4]),
            DiscProfile::LimbDarkened => (1, TextureIndex::NONE, Vec3::ZERO, [u32::MAX; 4]),
            DiscProfile::Textured { albedo, up, lit_by } => {
                (2, albedo, up, lit_by.map(|l| l.unwrap_or(u32::MAX)))
            }
        };
        DirectionalLightGpu {
            direction_to_light: self.direction_to_light.normalize().to_array(),
            angular_radius: self.angular_radius,
            illuminance: (REC709_TO_REC2020 * self.illuminance).to_array(),
            disc,
            up: up.to_array(),
            albedo,
            lit_by,
        }
    }
}
