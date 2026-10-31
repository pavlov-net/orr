//! glTF materials as OpenPBR parameters.
//!
//! | glTF | `Material` |
//! | --- | --- |
//! | `baseColorFactor` rgb, `metallicFactor`, `roughnessFactor` | `base_color`, `metalness`, `specular_roughness` |
//! | `KHR_materials_ior` `ior` | `specular_ior` |
//! | `emissiveFactor`, `KHR_materials_emissive_strength` (1 without it) | `emission_color`, `emission_luminance` read as nits |
//! | `alphaMode` `MASK` with `alphaCutoff`, `BLEND`; base alpha | `alpha_mode`, `alpha_cutoff`; `geometry_opacity` under `MASK` and `BLEND` |
//! | `doubleSided` | `double_sided` |
//! | `KHR_materials_specular` `specularFactor`, `specularColorFactor`, `specularTexture` (A) | `specular_weight`, `specular_color`, the specular weight texture |
//! | `KHR_materials_clearcoat` `clearcoatFactor`, `clearcoatRoughnessFactor`, `clearcoatTexture` (R) | `coat_weight`, `coat_roughness`, `coat_ior` 1.5, the coat texture |
//! | `KHR_materials_sheen` `sheenColorFactor`, `sheenRoughnessFactor` | `fuzz_weight` 1 when the colour is not black, `fuzz_color`, `fuzz_roughness` (approximate) |
//! | `KHR_materials_anisotropy` `anisotropyStrength`, `anisotropyRotation` | `specular_anisotropy`, `specular_rotation` (radians, as glTF) |
//! | `KHR_materials_iridescence` `iridescenceFactor` > 0, `iridescenceIor`, `iridescenceThicknessMaximum` | `thin_film_thickness` (nm), `thin_film_ior`; the factor and thickness texture have no equivalent |
//! | `KHR_materials_transmission` `transmissionFactor`, `transmissionTexture` (R) | `transmission_weight`, the transmission texture |
//! | `KHR_materials_volume` `attenuationColor`, `attenuationDistance` | `transmission_color`, `transmission_depth`; an infinite distance is white at depth 0 |
//! | transmission without `KHR_materials_volume`, or with `thicknessFactor` 0 | `thin_walled` |
//! | `KHR_materials_diffuse_transmission` `diffuseTransmissionFactor`, `diffuseTransmissionColorFactor`, `diffuseTransmissionColorTexture` | `subsurface_weight`, `subsurface_color`, the subsurface colour texture; thin-walled (approximate: glTF's split of diffuse transmission and volume has no OpenPBR equivalent) |
//! | `KHR_materials_dispersion` `dispersion` > 0 | `transmission_dispersion_scale` 1, `transmission_dispersion_abbe_number` 20 / dispersion |
//! | `KHR_materials_displacement` (unofficial) `displacementGeometryFactor`, `displacementGeometryOffset`, `displacementGeometryTexture` (R) | `displacement_scale`, `displacement_offset`, the displacement texture; declared range from offset to offset + factor |
//!
//! Textures without a slot (clearcoat roughness and normal, sheen, anisotropy, iridescence,
//! specular colour and the diffuse transmission weight textures) are not cooked. Transmission
//! scatter, scatter anisotropy, the subsurface radius fields and the surface id have no
//! ratified glTF source and come from `MaterialPatch`.

use gltf::material::AlphaMode;
use gltf::texture::WrappingMode;
use orr_pack::material::Material;
use serde_json::Value;

use crate::texture::Wrap;
use crate::texture::material::{MaterialTextures, TexRef};

fn num(v: &Value, k: &str, d: f32) -> f32 {
    v.get(k).and_then(Value::as_f64).map_or(d, |x| x as f32)
}

fn rgb(v: &Value, k: &str, d: [f32; 3]) -> [f32; 3] {
    v.get(k)
        .and_then(Value::as_array)
        .filter(|a| a.len() == 3)
        .map_or(d, |a| {
            std::array::from_fn(|i| a[i].as_f64().unwrap_or(0.0) as f32)
        })
}

/// The material's extension objects, by name.
fn extension<'a>(json: &'a Value, m: &gltf::Material<'_>, name: &str) -> Option<&'a Value> {
    json["materials"][m.index()?]["extensions"].get(name)
}

pub(crate) fn material(json: &Value, m: &gltf::Material<'_>) -> Material {
    let ext = |name| extension(json, m, name);
    let pbr = m.pbr_metallic_roughness();
    let [r, g, b, a] = pbr.base_color_factor();
    let emissive = m.emissive_factor();
    let (alpha_mode, opacity) = match m.alpha_mode() {
        AlphaMode::Opaque => (Material::OPAQUE, 1.0),
        AlphaMode::Mask => (Material::MASK, a),
        AlphaMode::Blend => (Material::BLEND, a),
    };
    let mut out = Material {
        base_color: [r, g, b],
        metalness: pbr.metallic_factor(),
        specular_roughness: pbr.roughness_factor(),
        emission_color: emissive,
        emission_luminance: if emissive.iter().any(|&c| c > 0.0) {
            ext("KHR_materials_emissive_strength").map_or(1.0, |e| num(e, "emissiveStrength", 1.0))
        } else {
            0.0
        },
        geometry_opacity: opacity,
        alpha_mode,
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        double_sided: m.double_sided().into(),
        ..Material::default()
    };
    if let Some(e) = ext("KHR_materials_ior") {
        out.specular_ior = num(e, "ior", 1.5);
    }
    if let Some(e) = ext("KHR_materials_specular") {
        out.specular_weight = num(e, "specularFactor", 1.0);
        out.specular_color = rgb(e, "specularColorFactor", [1.0; 3]);
    }
    if let Some(e) = ext("KHR_materials_clearcoat") {
        out.coat_weight = num(e, "clearcoatFactor", 0.0);
        out.coat_roughness = num(e, "clearcoatRoughnessFactor", 0.0);
        out.coat_ior = 1.5;
        out.coat_color = [1.0; 3];
    }
    if let Some(e) = ext("KHR_materials_sheen") {
        let color = rgb(e, "sheenColorFactor", [0.0; 3]);
        if color.iter().any(|&c| c > 0.0) {
            out.fuzz_weight = 1.0;
            out.fuzz_color = color;
            out.fuzz_roughness = num(e, "sheenRoughnessFactor", 0.0);
        }
    }
    if let Some(e) = ext("KHR_materials_anisotropy") {
        out.specular_anisotropy = num(e, "anisotropyStrength", 0.0);
        out.specular_rotation = num(e, "anisotropyRotation", 0.0);
    }
    if let Some(e) = ext("KHR_materials_iridescence")
        && num(e, "iridescenceFactor", 0.0) > 0.0
    {
        out.thin_film_thickness = num(e, "iridescenceThicknessMaximum", 400.0);
        out.thin_film_ior = num(e, "iridescenceIor", 1.3);
    }
    if let Some(e) = ext("KHR_materials_transmission") {
        out.transmission_weight = num(e, "transmissionFactor", 0.0);
    }
    let volume = ext("KHR_materials_volume");
    if let Some(e) = volume {
        let distance = num(e, "attenuationDistance", f32::INFINITY);
        if distance.is_finite() && distance > 0.0 {
            out.transmission_color = rgb(e, "attenuationColor", [1.0; 3]);
            out.transmission_depth = distance;
        }
    }
    let thin_volume = volume.is_none_or(|e| num(e, "thicknessFactor", 0.0) == 0.0);
    if out.transmission_weight > 0.0 && thin_volume {
        out.thin_walled = 1;
    }
    if let Some(e) = ext("KHR_materials_diffuse_transmission") {
        out.subsurface_weight = num(e, "diffuseTransmissionFactor", 0.0);
        out.subsurface_color = rgb(e, "diffuseTransmissionColorFactor", [1.0; 3]);
        if out.subsurface_weight > 0.0 {
            out.thin_walled = 1;
        }
    }
    if let Some(e) = ext("KHR_materials_dispersion") {
        let d = num(e, "dispersion", 0.0);
        if d > 0.0 {
            out.transmission_dispersion_scale = 1.0;
            out.transmission_dispersion_abbe_number = 20.0 / d;
        }
    }
    if let Some(e) = ext("KHR_materials_displacement")
        && e.get("displacementGeometryTexture").is_some()
    {
        let (factor, offset) = (
            num(e, "displacementGeometryFactor", 1.0),
            num(e, "displacementGeometryOffset", 0.0),
        );
        out.displacement_scale = factor;
        out.displacement_offset = offset;
        out.displacement_min = offset.min(offset + factor);
        out.displacement_max = offset.max(offset + factor);
    }
    out
}

pub(crate) fn textures(
    doc: &gltf::Document,
    json: &Value,
    m: &gltf::Material<'_>,
) -> Result<MaterialTextures, String> {
    let tex = |t: gltf::Texture<'_>, coord: u32, channel: u8| -> Result<TexRef, String> {
        if coord > 1 {
            return Err(format!("texCoord {coord}: only uv sets 0 and 1 exist"));
        }
        let wrap = |w: WrappingMode| match w {
            WrappingMode::Repeat => Wrap::Repeat,
            WrappingMode::ClampToEdge => Wrap::Clamp,
            WrappingMode::MirroredRepeat => Wrap::Mirror,
        };
        let s = t.sampler();
        Ok(TexRef {
            image: t.source().index(),
            channel,
            uv_set: coord as u8,
            wrap: [wrap(s.wrap_s()), wrap(s.wrap_t())],
            srgb: None,
        })
    };
    // A textureInfo object of an extension.
    let info = |ext: &str, key: &str, channel: u8| -> Result<Option<TexRef>, String> {
        let Some(i) = extension(json, m, ext).and_then(|e| e.get(key)) else {
            return Ok(None);
        };
        let index = i["index"]
            .as_u64()
            .ok_or(format!("{ext} {key}: no index"))?;
        let t = doc
            .textures()
            .nth(index as usize)
            .ok_or(format!("{ext} {key}: texture {index} does not exist"))?;
        tex(t, i["texCoord"].as_u64().unwrap_or(0) as u32, channel).map(Some)
    };
    let pbr = m.pbr_metallic_roughness();
    let base_color = pbr
        .base_color_texture()
        .map(|i| tex(i.texture(), i.tex_coord(), 0))
        .transpose()?;
    let metal_rough = |channel| {
        pbr.metallic_roughness_texture()
            .map(|i| tex(i.texture(), i.tex_coord(), channel))
            .transpose()
    };
    Ok(MaterialTextures {
        base_color,
        opacity: base_color.map(|r| TexRef { channel: 3, ..r }),
        roughness: metal_rough(1)?,
        metalness: metal_rough(2)?,
        normal: m
            .normal_texture()
            .map(|i| Ok::<_, String>((tex(i.texture(), i.tex_coord(), 0)?, [i.scale(); 2])))
            .transpose()?,
        occlusion: m
            .occlusion_texture()
            .map(|i| Ok::<_, String>((tex(i.texture(), i.tex_coord(), 0)?, i.strength())))
            .transpose()?,
        emissive: m
            .emissive_texture()
            .map(|i| tex(i.texture(), i.tex_coord(), 0))
            .transpose()?,
        specular: info("KHR_materials_specular", "specularTexture", 3)?,
        coat: info("KHR_materials_clearcoat", "clearcoatTexture", 0)?,
        transmission: info("KHR_materials_transmission", "transmissionTexture", 0)?,
        subsurface_color: info(
            "KHR_materials_diffuse_transmission",
            "diffuseTransmissionColorTexture",
            0,
        )?,
        displacement: info(
            "KHR_materials_displacement",
            "displacementGeometryTexture",
            0,
        )?,
    })
}
