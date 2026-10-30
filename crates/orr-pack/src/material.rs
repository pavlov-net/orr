//! Material class: one blob holding a [`Material`].

use bytemuck::{Pod, Zeroable};

use crate::{AssetId, Error, Result};

pub const SCHEMA: u16 = 1;

/// A texture bound to a material slot.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct TextureBinding {
    /// [`AssetId::NONE`] when the slot is unbound.
    pub texture: AssetId,
    /// Texture channel the slot reads, 0..15.
    pub channels: u8,
    /// 0 or 1.
    pub uv_set: u8,
    pub _pad: [u8; 2],
}

/// OpenPBR 1.1.1 parameters of one material, as authored: colours linear Rec.709, lengths
/// in metres, thin-film thickness in nanometres, emission luminance in nits.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Material {
    pub base_color: [f32; 3],
    pub base_weight: f32,
    pub specular_color: [f32; 3],
    pub specular_weight: f32,
    pub emission_color: [f32; 3],
    pub emission_luminance: f32,
    pub transmission_color: [f32; 3],
    pub transmission_depth: f32,
    pub transmission_scatter: [f32; 3],
    pub transmission_scatter_anisotropy: f32,
    pub subsurface_color: [f32; 3],
    pub subsurface_weight: f32,
    pub subsurface_radius_scale: [f32; 3],
    pub subsurface_radius: f32,
    pub coat_color: [f32; 3],
    pub coat_weight: f32,
    pub fuzz_color: [f32; 3],
    pub fuzz_weight: f32,
    pub specular_roughness: f32,
    pub specular_ior: f32,
    pub specular_anisotropy: f32,
    pub specular_rotation: f32,
    pub metalness: f32,
    pub diffuse_roughness: f32,
    pub transmission_weight: f32,
    pub subsurface_scatter_anisotropy: f32,
    pub coat_roughness: f32,
    pub coat_ior: f32,
    pub fuzz_roughness: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,
    pub geometry_opacity: f32,
    /// Coverage below which an alpha-tested surface is cut out; used with [`Self::MASK`].
    pub alpha_cutoff: f32,
    /// 0 means no dispersion.
    pub transmission_dispersion_scale: f32,
    pub transmission_dispersion_abbe_number: f32,
    pub displacement_scale: f32,
    pub displacement_offset: f32,
    /// Declared displacement range in metres, used for bounds.
    pub displacement_min: f32,
    pub displacement_max: f32,
    /// [`Self::OPAQUE`], [`Self::MASK`] or [`Self::BLEND`].
    pub alpha_mode: u8,
    pub double_sided: u8,
    pub thin_walled: u8,
    /// Linked surface id; 0 is the standard surface.
    pub surface: u8,
    /// Indexed by the `slot` constants.
    pub textures: [TextureBinding; 12],
}

pub mod slot {
    pub const BASE_COLOR: usize = 0;
    pub const OPACITY: usize = 1;
    pub const NORMAL: usize = 2;
    pub const METALNESS: usize = 3;
    pub const ROUGHNESS: usize = 4;
    pub const OCCLUSION: usize = 5;
    pub const EMISSION: usize = 6;
    pub const SPECULAR_WEIGHT: usize = 7;
    pub const SUBSURFACE_COLOR: usize = 8;
    pub const DISPLACEMENT: usize = 9;
    pub const COAT: usize = 10;
    pub const TRANSMISSION: usize = 11;
}

impl Material {
    pub const OPAQUE: u8 = 0;
    pub const MASK: u8 = 1;
    pub const BLEND: u8 = 2;

    pub fn from_bytes(blob: &[u8]) -> Result<Self> {
        if blob.len() != size_of::<Self>() {
            return Err(Error::Malformed("material size"));
        }
        Ok(bytemuck::pod_read_unaligned(blob))
    }

    /// Whether the material emits light.
    pub fn emits(&self) -> bool {
        self.emission_luminance > 0.0 && self.emission_color.iter().any(|&c| c > 0.0)
    }
}

/// The OpenPBR 1.1.1 defaults, opaque, single-sided, no textures.
impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [0.8; 3],
            base_weight: 1.0,
            specular_color: [1.0; 3],
            specular_weight: 1.0,
            emission_color: [1.0; 3],
            emission_luminance: 0.0,
            transmission_color: [1.0; 3],
            transmission_depth: 0.0,
            transmission_scatter: [0.0; 3],
            transmission_scatter_anisotropy: 0.0,
            subsurface_color: [0.8; 3],
            subsurface_weight: 0.0,
            subsurface_radius_scale: [1.0, 0.5, 0.25],
            subsurface_radius: 1.0,
            coat_color: [1.0; 3],
            coat_weight: 0.0,
            fuzz_color: [1.0; 3],
            fuzz_weight: 0.0,
            specular_roughness: 0.3,
            specular_ior: 1.5,
            specular_anisotropy: 0.0,
            specular_rotation: 0.0,
            metalness: 0.0,
            diffuse_roughness: 0.0,
            transmission_weight: 0.0,
            subsurface_scatter_anisotropy: 0.0,
            coat_roughness: 0.0,
            coat_ior: 1.6,
            fuzz_roughness: 0.5,
            thin_film_thickness: 0.0,
            thin_film_ior: 1.4,
            geometry_opacity: 1.0,
            alpha_cutoff: 0.5,
            transmission_dispersion_scale: 0.0,
            transmission_dispersion_abbe_number: 20.0,
            displacement_scale: 1.0,
            displacement_offset: 0.0,
            displacement_min: 0.0,
            displacement_max: 0.0,
            alpha_mode: Self::OPAQUE,
            double_sided: 0,
            thin_walled: 0,
            surface: 0,
            textures: [TextureBinding::zeroed(); 12],
        }
    }
}
