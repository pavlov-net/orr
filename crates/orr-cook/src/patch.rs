//! Partial overrides of materials and mesh settings: every field optional, applied over a
//! value field by field.

use orr_pack::material::Material;
use serde::Deserialize;

use crate::mesh::{LodMode, MeshSettings, OmmSettings, StaticDisplacement};

macro_rules! patch {
    ($(#[$doc:meta])* $name:ident for $target:ty { $($field:ident: $ty:ty,)* }) => {
        $(#[$doc])*
        #[derive(Clone, Debug, Default, PartialEq, Deserialize)]
        #[serde(default, deny_unknown_fields)]
        pub struct $name {
            $(pub $field: Option<$ty>,)*
        }

        impl $name {
            /// Sets every field this patch names.
            pub fn apply(&self, to: &mut $target) {
                $(if let Some(v) = &self.$field {
                    to.$field = v.clone();
                })*
            }
        }
    };
}

patch! {
    /// Any [`Material`] field but the texture bindings. glTF has no source for the
    /// subsurface radius, the scatter and anisotropy fields or the linked surface id, so
    /// those come only from here. Colours are linear Rec.709, lengths metres.
    MaterialPatch for Material {
        base_color: [f32; 3],
        base_weight: f32,
        specular_color: [f32; 3],
        specular_weight: f32,
        emission_color: [f32; 3],
        emission_luminance: f32,
        transmission_color: [f32; 3],
        transmission_depth: f32,
        transmission_scatter: [f32; 3],
        transmission_scatter_anisotropy: f32,
        subsurface_color: [f32; 3],
        subsurface_weight: f32,
        subsurface_radius_scale: [f32; 3],
        subsurface_radius: f32,
        coat_color: [f32; 3],
        coat_weight: f32,
        fuzz_color: [f32; 3],
        fuzz_weight: f32,
        specular_roughness: f32,
        specular_ior: f32,
        specular_anisotropy: f32,
        specular_rotation: f32,
        metalness: f32,
        diffuse_roughness: f32,
        transmission_weight: f32,
        subsurface_scatter_anisotropy: f32,
        coat_roughness: f32,
        coat_ior: f32,
        fuzz_roughness: f32,
        thin_film_thickness: f32,
        thin_film_ior: f32,
        geometry_opacity: f32,
        alpha_cutoff: f32,
        transmission_dispersion_scale: f32,
        transmission_dispersion_abbe_number: f32,
        displacement_scale: f32,
        displacement_offset: f32,
        displacement_min: f32,
        displacement_max: f32,
        alpha_mode: u8,
        double_sided: u8,
        thin_walled: u8,
        surface: u8,
    }
}

patch! {
    /// Any [`MeshSettings`] field.
    MeshPatch for MeshSettings {
        lod: LodMode,
        max_vertices: u16,
        max_vertices_foliage: u16,
        max_triangles: u16,
        position_step_log2: i8,
        deformation_bound: f32,
        merge_level: Option<u8>,
        merge_error: Option<f32>,
        dilate_double_sided: bool,
        hierarchy_resident_max: u32,
        omm: Option<OmmSettings>,
        displace_static: Option<StaticDisplacement>,
    }
}

/// The patches of `overrides` whose pattern `name` contains, applied in order.
pub(crate) fn apply_matching<T, P>(
    overrides: &[(String, P)],
    name: &str,
    to: &mut T,
    apply: impl Fn(&P, &mut T),
) {
    for (pattern, p) in overrides {
        if name.contains(pattern.as_str()) {
            apply(p, to);
        }
    }
}
