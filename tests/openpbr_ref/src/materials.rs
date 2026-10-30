//! The parameter grid every check runs over: each OpenPBR lobe on its own and in the usual
//! combinations, white where the lobe allows it so that energy checks are tight.

use crate::Inputs;

/// A named point of the grid.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub name: &'static str,
    pub inputs: Inputs,
    /// Has an ideal-specular (delta) lobe, which has no density to bin or integrate.
    pub delta: bool,
    /// f(wo, wi) = f(wi, wo) for directions above the surface. adobe/openpbr-bsdf's coat and
    /// fuzz attenuate the base by the view side only, and its thin-walled transmission
    /// folds the back side into reflection, so materials with those are not.
    pub reciprocal: bool,
}

/// Cosines of the view angle the checks use.
pub const VIEW_COSINES: [f32; 4] = [0.05, 0.3, 0.7, 1.0];

/// The view direction at cosine `mu`, in the x-z plane.
pub fn view(mu: f32) -> [f32; 3] {
    [(1.0 - mu * mu).max(0.0).sqrt(), 0.0, mu]
}

pub fn grid() -> Vec<Material> {
    let d = Inputs::default();
    let white = Inputs {
        base_color: [1.0; 3],
        ..d
    };
    let metal = Inputs {
        base_metalness: 1.0,
        ..white
    };
    let glass = Inputs {
        transmission_weight: 1.0,
        ..white
    };
    let (c30, s30) = (30f32.to_radians().cos(), 30f32.to_radians().sin());
    let m = |name, inputs, delta, reciprocal| Material {
        name,
        inputs,
        delta,
        reciprocal,
    };
    vec![
        m(
            "diffuse",
            Inputs {
                specular_weight: 0.0,
                ..white
            },
            false,
            true,
        ),
        m(
            "diffuse rough",
            Inputs {
                specular_weight: 0.0,
                base_diffuse_roughness: 1.0,
                ..white
            },
            false,
            true,
        ),
        m(
            "dielectric 0.2",
            Inputs {
                specular_roughness: 0.2,
                ..white
            },
            false,
            true,
        ),
        m(
            "dielectric 0.6",
            Inputs {
                specular_roughness: 0.6,
                ..white
            },
            false,
            true,
        ),
        m(
            "dielectric ior 2",
            Inputs {
                specular_roughness: 0.4,
                specular_ior: 2.0,
                ..white
            },
            false,
            true,
        ),
        m(
            "dielectric tinted",
            Inputs {
                specular_roughness: 0.4,
                specular_color: [1.0, 0.7, 0.4],
                specular_weight: 0.6,
                ..d
            },
            false,
            true,
        ),
        m(
            "metal 0.3",
            Inputs {
                specular_roughness: 0.3,
                ..metal
            },
            false,
            true,
        ),
        m(
            "metal 0.8",
            Inputs {
                specular_roughness: 0.8,
                ..metal
            },
            false,
            true,
        ),
        m(
            "metal f82",
            Inputs {
                base_color: [0.95, 0.64, 0.54],
                specular_color: [0.9, 0.95, 1.0],
                specular_roughness: 0.4,
                ..metal
            },
            false,
            true,
        ),
        m(
            "metal anisotropic",
            Inputs {
                specular_roughness: 0.5,
                specular_roughness_anisotropy: 0.8,
                specular_rotation: [c30, s30],
                ..metal
            },
            false,
            true,
        ),
        m(
            "coat",
            Inputs {
                base_color: [0.5; 3],
                coat_weight: 1.0,
                coat_roughness: 0.2,
                ..d
            },
            false,
            false,
        ),
        m(
            "coat on metal",
            Inputs {
                specular_roughness: 0.5,
                coat_weight: 1.0,
                coat_roughness: 0.3,
                coat_color: [0.9, 0.8, 0.7],
                ..metal
            },
            false,
            false,
        ),
        m(
            "fuzz",
            Inputs {
                fuzz_weight: 1.0,
                fuzz_roughness: 0.5,
                ..white
            },
            false,
            false,
        ),
        m(
            "glass 0.3",
            Inputs {
                specular_roughness: 0.3,
                ..glass
            },
            false,
            true,
        ),
        m(
            "glass in water",
            Inputs {
                specular_roughness: 0.3,
                exterior_ior: 1.33,
                ..glass
            },
            false,
            true,
        ),
        m(
            "glass dispersive",
            Inputs {
                specular_roughness: 0.3,
                transmission_dispersion_scale: 1.0,
                ..glass
            },
            false,
            true,
        ),
        m(
            "thin-walled glass",
            Inputs {
                specular_roughness: 0.4,
                geometry_thin_walled: 1,
                ..glass
            },
            false,
            false,
        ),
        m(
            "subsurface",
            Inputs {
                subsurface_weight: 1.0,
                subsurface_color: [0.9, 0.6, 0.4],
                ..d
            },
            false,
            true,
        ),
        m(
            "thin-walled subsurface",
            Inputs {
                subsurface_weight: 1.0,
                geometry_thin_walled: 1,
                ..white
            },
            false,
            true,
        ),
        m(
            "thin film",
            Inputs {
                specular_roughness: 0.3,
                thin_film_weight: 1.0,
                thin_film_thickness: 0.5,
                ..white
            },
            false,
            true,
        ),
        m(
            "mirror",
            Inputs {
                specular_roughness: 0.0,
                ..metal
            },
            true,
            false,
        ),
        m(
            "smooth glass",
            Inputs {
                specular_roughness: 0.0,
                ..glass
            },
            true,
            false,
        ),
        m(
            "smooth coat",
            Inputs {
                coat_weight: 1.0,
                ..white
            },
            true,
            false,
        ),
    ]
}
