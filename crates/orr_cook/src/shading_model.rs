//! Materials that sources state in another shading model, as the OpenPBR parameters a cooked
//! [`Material`] holds. Every renderer reads one model; what a conversion gives up is listed with
//! the importer's mapping (`gltf::material`, `shading`).
//!
//! glTF states its arithmetic per channel in linear Rec.709, and the shading networks read
//! UsdPreviewSurface's values there too, so the conversions take a channel's largest value and a
//! product of colors in linear Rec.709, though the fields hold linear Rec.2020.

use std::sync::LazyLock;

use orr_color::{RgbPrimaries, rec709_to_rec2020};
use orr_math::{DMat3, DVec3};
use orr_pack::material::{Material, slot};

use crate::texture::material::{MaterialTextures, TexRef};

/// The shading model a source states a material's parameters in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ShadingModel {
    /// The fields mean what OpenPBR states.
    #[default]
    OpenPbr,
    /// glTF's metallic-roughness and its extensions: the dielectric's F0 is the IOR's times
    /// `specular_color` (at most 1) times `specular_weight`, and the base color tints light
    /// through the surface.
    Gltf,
    /// UsdPreviewSurface's metalness workflow: the coat shades on the base's normal, normal map
    /// included.
    Preview,
    /// UsdPreviewSurface's specular workflow: as [`Self::Preview`], with `specular_color` F0
    /// itself and the IOR left to refraction and the coat.
    PreviewSpecular,
}

/// `m` and `t`, stated in `model`, as OpenPBR parameters. `peak` is the highest value a color
/// texture reads in any channel, `range` the lowest and highest a scalar texture reads. Returns
/// what the conversion drops, for a warning.
pub(crate) fn to_openpbr(
    model: ShadingModel,
    m: &mut Material,
    t: &mut MaterialTextures,
    peak: impl Fn(&TexRef) -> f32,
    range: impl Fn(&TexRef) -> (f32, f32),
) -> Option<String> {
    match model {
        ShadingModel::OpenPbr => None,
        ShadingModel::Gltf => gltf(m, t),
        ShadingModel::Preview => {
            coat_on_base(m, t);
            None
        }
        ShadingModel::PreviewSpecular => {
            coat_on_base(m, t);
            stated_f0(m, t, peak, range);
            None
        }
    }
}

/// OpenPBR's largest F0, past which an IOR leaves what a float holds.
const MAX_F0: f32 = 0.9999;

/// The textures whose whole range [`to_openpbr`] reads of a material stated in `model`.
pub(crate) fn ranged(model: ShadingModel, t: &MaterialTextures) -> impl Iterator<Item = &TexRef> {
    let slots: &[usize] = match model {
        ShadingModel::PreviewSpecular => &[slot::SPECULAR_COLOR, slot::SPECULAR_IOR],
        _ => &[],
    };
    slots
        .iter()
        .filter_map(|s| t.more.get(s).and_then(Option::as_ref))
}

fn max3(c: [f32; 3]) -> f32 {
    c.into_iter().fold(0.0, f32::max)
}

/// Linear Rec.709 `c` in linear Rec.2020.
fn rec2020(c: [f32; 3]) -> [f32; 3] {
    rec709_to_rec2020(c.map(f64::from)).as_vec3().to_array()
}

/// Linear Rec.2020 `c` in linear Rec.709.
fn rec709(c: [f32; 3]) -> [f32; 3] {
    static MATRIX: LazyLock<DMat3> =
        LazyLock::new(|| RgbPrimaries::REC2020.matrix_to(RgbPrimaries::REC709));
    (*MATRIX * DVec3::from(c.map(f64::from)))
        .as_vec3()
        .to_array()
}

/// Takes the color fields of `m` from linear Rec.709, the space glTF's factors and the shading
/// networks' values are in, to the linear Rec.2020 of a cooked [`Material`].
pub(crate) fn colors_to_rec2020(m: &mut Material) {
    for c in [
        &mut m.base_color,
        &mut m.specular_color,
        &mut m.emission_color,
        &mut m.transmission_color,
        &mut m.transmission_scatter,
        &mut m.subsurface_color,
        &mut m.coat_color,
        &mut m.fuzz_color,
    ] {
        *c = rec2020(*c);
    }
}

/// F0 of a dielectric of relative IOR `ior`.
fn f0_of(ior: f32) -> f32 {
    ((ior - 1.0) / (ior + 1.0)).powi(2)
}

/// glTF's model: a specular color past 1 raises F0 past the IOR's, which OpenPBR's weight does
/// and its color, a tint, can't; the base color's tint of light through a surface with no volume
/// becomes the surface tint OpenPBR's transmission color is at depth 0.
fn gltf(m: &mut Material, t: &mut MaterialTextures) -> Option<String> {
    // The texels of `specularColorTexture` read 0..1: the factor holds the rest.
    let c = max3(rec709(m.specular_color));
    if c > 1.0 {
        m.specular_weight *= c;
        m.specular_color = m.specular_color.map(|v| v / c);
    }
    let transmits = m.transmission_weight > 0.0 || t.transmission.is_some();
    if !transmits {
        return None;
    }
    let tinted = m.base_color != [1.0; 3]
        || m.base_weight != 1.0
        || t.base_color.is_some()
        || m.textures[slot::BASE_COLOR].vertex != 0;
    if m.thin_walled == 0 && m.transmission_depth > 0.0 {
        return tinted.then(|| {
            "its base color tints transmission, which OpenPBR has no part for beside its volume's \
             color: the volume's alone stays"
                .to_string()
        });
    }
    let base = rec709(m.base_color).map(|v| v * m.base_weight);
    let color = rec709(m.transmission_color);
    m.transmission_color = rec2020(std::array::from_fn(|i| color[i] * base[i]));
    if let Some(r) = t.base_color {
        t.more.insert(slot::TRANSMISSION_COLOR, Some(r));
    }
    let from = m.textures[slot::BASE_COLOR];
    let to = &mut m.textures[slot::TRANSMISSION_COLOR];
    to.vertex = from.vertex;
    to.vertex_complement = from.vertex_complement;
    None
}

/// UsdPreviewSurface's coat shades on the base's normal, normal map included: OpenPBR's coat has
/// a normal of its own, which reads the base's map.
fn coat_on_base(m: &mut Material, t: &mut MaterialTextures) {
    let coat = m.coat_weight > 0.0 || t.coat.is_some();
    if !coat || t.coat_normal.is_some() {
        return;
    }
    let Some(normal) = t.normal else {
        return;
    };
    t.coat_normal = Some(normal);
    if m.object_normal(slot::NORMAL) {
        m.object_normals |= Material::OBJECT_COAT_NORMAL;
    }
}

/// UsdPreviewSurface's specular workflow states F0, which OpenPBR makes of the IOR, the weight
/// that scales its F0 and the color that tints it: the weight takes the largest F0 the material
/// reads over the IOR's, and the color, a tint, what each channel and texel holds of that. The
/// IOR stays for refraction and the coat, and a texture of it scales F0 as the middle of its
/// range does, unless it is 1 everywhere, whose F0 no weight scales: the largest F0 then states
/// it.
fn stated_f0(
    m: &mut Material,
    t: &MaterialTextures,
    peak: impl Fn(&TexRef) -> f32,
    range: impl Fn(&TexRef) -> (f32, f32),
) {
    let texels = t
        .more
        .get(&slot::SPECULAR_COLOR)
        .copied()
        .flatten()
        .map_or(1.0, |r| peak(&r));
    let largest = max3(rec709(m.specular_color)) * texels;
    if largest <= 0.0 {
        m.specular_weight = 0.0;
        return;
    }
    let texture = t.more.get(&slot::SPECULAR_IOR).copied().flatten();
    let middle = texture.map_or(1.0, |r| {
        let (lo, hi) = range(&r);
        0.5 * (lo + hi)
    });
    let f0 = f0_of(m.specular_ior * middle);
    if f0 > 0.0 {
        m.specular_weight = largest / f0;
    } else if texture.is_none() {
        let root = largest.min(MAX_F0).sqrt();
        m.specular_ior = (1.0 + root) / (1.0 - root);
        m.specular_weight = 1.0;
    } else {
        m.specular_weight = 0.0;
    }
    m.specular_color = m.specular_color.map(|v| v / largest);
}
