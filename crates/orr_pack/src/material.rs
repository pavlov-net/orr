//! Material class: one blob holding a [`Material`].

use bytemuck::{Pod, Zeroable};

use crate::{AssetId, Error, Result};

pub const SCHEMA: u16 = 2;

/// A texture bound to a material slot.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct TextureBinding {
    /// A texture, or a [`TextureTiles`](crate::tiles::TextureTiles) set of them; the rest of the
    /// binding then holds for each tile, read at the point's place in its unit square. Where no
    /// tile lies the slot reads [`Self::outside`]. [`AssetId::NONE`] when the slot is unbound.
    pub texture: AssetId,
    /// Texture channel the slot reads, 0..15.
    pub channels: u8,
    /// 0 or 1.
    pub uv_set: u8,
    /// What u and v read past 0..1, as [`wrap`] values.
    pub wrap: [u8; 2],
    /// [`filter`] bits.
    pub filter: u8,
    pub _pad: [u8; 3],
    /// What the slot reads, from its first channel on, past 0..1 on an axis of
    /// [`wrap::CONSTANT`] and where a tile set has no tile, in the units its texels sample as.
    pub outside: [f32; 4],
}

/// `TextureBinding::wrap` values.
pub mod wrap {
    pub const REPEAT: u8 = 0;
    pub const CLAMP: u8 = 1;
    pub const MIRROR: u8 = 2;
    /// Transparent black past 0..1, filtered in with the edge texels.
    pub const BORDER: u8 = 3;
    /// [`TextureBinding::outside`](super::TextureBinding::outside) past 0..1, and [`CLAMP`]
    /// within it.
    pub const CONSTANT: u8 = 4;
}

/// `TextureBinding::filter` bits. Without any, a slot filters trilinearly and anisotropically.
pub mod filter {
    pub const MAG_NEAREST: u8 = 1 << 0;
    pub const MIN_NEAREST: u8 = 1 << 1;
    pub const MIP_NEAREST: u8 = 1 << 2;
    /// Reads mip 0 alone, whatever the footprint; [`MIP_NEAREST`] is set with it.
    pub const ONE_MIP: u8 = 1 << 3;
}

/// OpenPBR 1.1.1 parameters of one material, as authored: colors linear Rec.709, lengths
/// in world space meters, thin-film thickness in nanometers, emission luminance in nits, angles
/// in radians. The displacement fields are the exception: lengths in mesh space.
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
    pub coat_darkening: f32,
    pub coat_anisotropy: f32,
    /// Radians from the mesh tangent toward the bitangent, as [`Self::specular_rotation`].
    pub coat_rotation: f32,
    pub fuzz_roughness: f32,
    pub thin_film_weight: f32,
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
    /// Declared displacement range, used for bounds.
    pub displacement_min: f32,
    pub displacement_max: f32,
    /// [`Self::OPAQUE`], [`Self::MASK`] or [`Self::BLEND`].
    pub alpha_mode: u8,
    /// By default (0), back faces are not drawn and do not emit light, but rays still hit them
    /// and they cast shadows. Nonzero draws both sides and both sides emit. Transparent surfaces
    /// and strands show both sides either way.
    pub double_sided: u8,
    pub thin_walled: u8,
    /// Linked surface id; 0 is the standard surface.
    pub surface: u8,
    /// What the tangent frame follows: [`Self::FRAME_NORMAL_MAP`] by default.
    pub frame_uv: u8,
    /// Whose model the parameters state: [`Self::MODEL_OPENPBR`] by default.
    pub model: u8,
    pub _pad: [u8; 2],
    /// Indexed by the `slot` constants.
    pub textures: [TextureBinding; slot::COUNT],
    /// Per slot, the affine map from mesh uv to the uv its texture is sampled at, stored
    /// as its difference from the identity so that zero is the identity: `[a - 1, b, c, d,
    /// e - 1, f]` for `u' = a u + b v + c`, `v' = d u + e v + f`.
    pub uv_transforms: [[f32; 6]; slot::COUNT],
}

impl Material {
    /// The tangent frame follows the uv set the normal map reads, else the coat's.
    pub const FRAME_NORMAL_MAP: u8 = 0;
    pub const FRAME_UV0: u8 = 1;
    pub const FRAME_UV1: u8 = 2;
    /// It follows uvs a mesh doesn't carry: only a tangent stream holds it.
    pub const FRAME_OTHER: u8 = 3;
    /// [`Self::tangent_set`] of [`Self::FRAME_OTHER`].
    pub const TANGENT_SET_OTHER: u8 = 2;

    /// The fields mean what OpenPBR states.
    pub const MODEL_OPENPBR: u8 = 0;
    /// As glTF's metallic-roughness: the dielectric Fresnel runs from `F0` at normal incidence to
    /// [`Self::specular_weight`] at grazing, with `F0` the IOR's times [`Self::specular_color`]
    /// (at most 1) times the weight, and what it leaves goes to the layer below. Neither field
    /// reaches a metal, whose edge is white, or the angle of refraction. The coat mixes over all of
    /// that by its Fresnel term, and takes nothing from the base's IOR or roughness.
    pub const MODEL_GLTF: u8 = 1;
    /// As [`Self::MODEL_GLTF`] under UsdPreviewSurface's metalness workflow, except that a metal
    /// reflects its base color at every angle and the coat shades on the base's normal, normal
    /// map included.
    pub const MODEL_PREVIEW: u8 = 2;
    /// As [`Self::MODEL_PREVIEW`] under its specular workflow: [`Self::specular_color`] is `F0`
    /// itself.
    pub const MODEL_PREVIEW_SPECULAR: u8 = 3;

    /// The uv set the tangent frame follows, 0 or 1, or [`Self::TANGENT_SET_OTHER`].
    pub fn tangent_set(&self) -> u8 {
        let normal = self.textures[slot::NORMAL];
        let coat = self.textures[slot::COAT_NORMAL];
        let map = if normal.texture == AssetId::NONE && coat.texture != AssetId::NONE {
            coat
        } else {
            normal
        };
        match self.frame_uv {
            Self::FRAME_NORMAL_MAP => map.uv_set & 1,
            f => f - 1,
        }
    }
}

/// Texture slots. All but `NORMAL`, `COAT_NORMAL`, `EMISSION`, `EMISSION_LUMINANCE`,
/// `SPECULAR_WEIGHT`, `SPECULAR_IOR`, `COAT_IOR`, `THIN_FILM_IOR`, `THIN_FILM_THICKNESS`,
/// `SUBSURFACE_RADIUS` and `DISPLACEMENT` read values of 0..1. Nothing clamps what they sample: a
/// signed texture bound to one is the caller's responsibility. A value multiplies its field,
/// with three exceptions: `TANGENT` and `COAT_TANGENT` hold a direction in the tangent plane in
/// their first two channels at `0.5 v + 0.5`, on top of the field's rotation, and
/// `SUBSURFACE_ANISOTROPY` holds the anisotropy at `0.5 g + 0.5`, in place of the field.
pub mod slot {
    pub const BASE_COLOR: usize = 0;
    pub const OPACITY: usize = 1;
    pub const NORMAL: usize = 2;
    pub const METALNESS: usize = 3;
    pub const ROUGHNESS: usize = 4;
    pub const EMISSION: usize = 5;
    pub const SPECULAR_WEIGHT: usize = 6;
    pub const SUBSURFACE_COLOR: usize = 7;
    pub const DISPLACEMENT: usize = 8;
    pub const COAT: usize = 9;
    pub const TRANSMISSION: usize = 10;
    pub const SPECULAR_COLOR: usize = 11;
    pub const SPECULAR_IOR: usize = 12;
    pub const BASE_WEIGHT: usize = 13;
    pub const DIFFUSE_ROUGHNESS: usize = 14;
    pub const ANISOTROPY: usize = 15;
    pub const TANGENT: usize = 16;
    pub const TRANSMISSION_COLOR: usize = 17;
    pub const SUBSURFACE_WEIGHT: usize = 18;
    pub const SUBSURFACE_RADIUS: usize = 19;
    pub const SUBSURFACE_RADIUS_SCALE: usize = 20;
    pub const SUBSURFACE_ANISOTROPY: usize = 21;
    pub const FUZZ_WEIGHT: usize = 22;
    pub const FUZZ_COLOR: usize = 23;
    pub const FUZZ_ROUGHNESS: usize = 24;
    pub const COAT_COLOR: usize = 25;
    pub const COAT_ROUGHNESS: usize = 26;
    pub const COAT_IOR: usize = 27;
    pub const COAT_DARKENING: usize = 28;
    pub const COAT_ANISOTROPY: usize = 29;
    pub const COAT_NORMAL: usize = 30;
    pub const COAT_TANGENT: usize = 31;
    pub const THIN_FILM_WEIGHT: usize = 32;
    pub const THIN_FILM_THICKNESS: usize = 33;
    pub const THIN_FILM_IOR: usize = 34;
    pub const EMISSION_LUMINANCE: usize = 35;
    pub const COUNT: usize = 36;
    /// Slots from here on read only for a material that binds one of them.
    pub const MORE: usize = SPECULAR_COLOR;
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
            coat_darkening: 1.0,
            coat_anisotropy: 0.0,
            coat_rotation: 0.0,
            fuzz_roughness: 0.5,
            thin_film_weight: 0.0,
            thin_film_thickness: 500.0,
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
            frame_uv: Self::FRAME_NORMAL_MAP,
            model: Self::MODEL_OPENPBR,
            _pad: [0; 2],
            textures: [TextureBinding::zeroed(); slot::COUNT],
            uv_transforms: [[0.0; 6]; slot::COUNT],
        }
    }
}
