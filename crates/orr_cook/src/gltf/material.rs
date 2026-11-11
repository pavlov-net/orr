//! glTF materials as OpenPBR parameters.
//!
//! | glTF | `Material` |
//! | --- | --- |
//! | `baseColorFactor` rgb, `metallicFactor`, `roughnessFactor` | `base_color`, `metalness`, `specular_roughness` |
//! | `COLOR_0` on a primitive | the vertex color stream, multiplying the base color and, by its alpha, the opacity |
//! | `metallicRoughnessTexture` G and B, or R and G of an image holding only red and green | the roughness and metalness textures |
//! | `KHR_texture_transform` on any textureInfo (offset, rotation, scale, texCoord) | the slot's `uv_transforms` entry and uv set |
//! | `KHR_materials_ior` `ior` | `specular_ior` |
//! | `emissiveFactor`, `emissiveStrength` (1 when absent; `KHR_materials_emissive_strength` overrides it) | `emission_color`, `emission_luminance` read as nits |
//! | `alphaMode` `MASK` with `alphaCutoff`, `BLEND`; base alpha | `alpha_mode`, `alpha_cutoff`; `geometry_opacity` under `MASK` and `BLEND` |
//! | `doubleSided` | `double_sided` |
//! | `KHR_materials_specular` `specularFactor`, `specularColorFactor`, `specularTexture` (A), `specularColorTexture` | `specular_weight`, `specular_color`, their textures |
//! | `KHR_materials_clearcoat` `clearcoatFactor`, `clearcoatRoughnessFactor`, `clearcoatTexture` (R), `clearcoatRoughnessTexture` (G), `clearcoatNormalTexture` | `coat_weight`, `coat_roughness`, `coat_ior` 1.5, their textures and the coat normal |
//! | `KHR_materials_sheen` `sheenColorFactor`, `sheenRoughnessFactor`, `sheenColorTexture`, `sheenRoughnessTexture` (A) | `fuzz_weight` 1 when the factor is not black, `fuzz_color`, `fuzz_roughness` (approximate), their textures |
//! | `KHR_materials_anisotropy` `anisotropyStrength`, `anisotropyRotation`, `anisotropyTexture` (RG direction, B strength) | `specular_anisotropy`, `specular_rotation` (radians, as glTF), the anisotropy and tangent textures |
//! | `KHR_materials_iridescence` `iridescenceFactor`, `iridescenceIor`, `iridescenceThicknessMinimum` and `Maximum`, `iridescenceTexture` (R), `iridescenceThicknessTexture` (G) | `thin_film_weight`, `thin_film_ior`, `thin_film_thickness` (nm) and their textures: the thickness texture reads `mix(minimum, maximum, g)` over the larger of the two |
//! | `KHR_materials_transmission` `transmissionFactor`, `transmissionTexture` (R) | `transmission_weight`, the transmission texture; `Material::MODEL_GLTF` tints it with the base color |
//! | `KHR_materials_volume` `attenuationColor`, `attenuationDistance` | `transmission_color`, `transmission_depth` above `thicknessFactor` 0; an infinite distance is white at depth 0 |
//! | transmission without `KHR_materials_volume`, or with `thicknessFactor` 0 | `thin_walled` |
//! | `KHR_materials_diffuse_transmission` `diffuseTransmissionFactor`, `diffuseTransmissionColorFactor`, `diffuseTransmissionTexture` (A), `diffuseTransmissionColorTexture` | `subsurface_weight`, `subsurface_color`, their textures; thin-walled (approximate: glTF's split of diffuse transmission and volume has no OpenPBR equivalent) |
//! | `KHR_materials_dispersion` `dispersion` > 0 | `transmission_dispersion_scale` 1, `transmission_dispersion_abbe_number` 20 / dispersion |
//! | `KHR_materials_displacement` (unofficial, the draft of KhronosGroup/glTF issue 948, in either of its forms) `displacementGeometryFactor`, `displacementGeometryOffset`, `displacementGeometryTexture` (R) | `displacement_scale`, `displacement_offset`, the displacement texture; declared range from offset to offset + factor. Without a texture "the value is take as it is": factor, 0 when absent, plus offset everywhere, through a solid image |
//!
//! `KHR_materials_volume` `thicknessTexture` stays unread: "Ray-tracers should ignore the
//! thickness texture and use the actual, ray-traced distance instead." Transmission scatter,
//! scatter anisotropy, the subsurface radius fields and the surface id have no ratified glTF
//! source and come from `MaterialPatch`.

use gltf::material::AlphaMode;
use gltf::texture::{MagFilter, MinFilter, WrappingMode};
use orr_pack::material::{Material, filter, slot};
use serde_json::Value;

use crate::texture::Wrap;
use crate::texture::color::ImageColor;
use crate::texture::material::{ImageRules, MaterialTextures, TexRef};

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

/// The material's JSON object; `None` for the default material.
fn raw<'a>(json: &'a Value, m: &gltf::Material<'_>) -> Option<&'a Value> {
    json["materials"].get(m.index()?)
}

/// The material's extension objects, by name.
fn extension<'a>(json: &'a Value, m: &gltf::Material<'_>, name: &str) -> Option<&'a Value> {
    raw(json, m)?["extensions"].get(name)
}

pub(super) fn material(json: &Value, m: &gltf::Material<'_>) -> Material {
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
            ext("KHR_materials_emissive_strength")
                .or_else(|| raw(json, m))
                .map_or(1.0, |e| num(e, "emissiveStrength", 1.0))
        } else {
            0.0
        },
        geometry_opacity: opacity,
        alpha_mode,
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        double_sided: m.double_sided().into(),
        model: Material::MODEL_GLTF,
        coat_darkening: 0.0,
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
    if let Some(e) = ext("KHR_materials_iridescence") {
        out.thin_film_weight = num(e, "iridescenceFactor", 0.0);
        // A thickness texture reads between the two, and texels store the share of the larger.
        let (low, high) = film_thickness(e);
        out.thin_film_thickness = match e.get("iridescenceThicknessTexture") {
            Some(_) => low.max(high),
            None => high,
        };
        out.thin_film_ior = num(e, "iridescenceIor", 1.3);
    }
    if let Some(e) = ext("KHR_materials_transmission") {
        out.transmission_weight = num(e, "transmissionFactor", 0.0);
    }
    let volume = ext("KHR_materials_volume");
    let thin_volume = volume.is_none_or(|e| num(e, "thicknessFactor", 0.0) == 0.0);
    // "If the value is 0 the material is thin-walled": nothing lies inside to attenuate.
    if let Some(e) = volume.filter(|_| !thin_volume) {
        let distance = num(e, "attenuationDistance", f32::INFINITY);
        if distance.is_finite() && distance > 0.0 {
            out.transmission_color = rgb(e, "attenuationColor", [1.0; 3]);
            out.transmission_depth = distance;
        }
    }
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
    if let Some(e) = ext("KHR_materials_displacement") {
        let (scale, offset) = match constant_height(e) {
            Some(height) => (0.0, height),
            None => (
                num(e, "displacementGeometryFactor", 1.0),
                num(e, "displacementGeometryOffset", 0.0),
            ),
        };
        out.displacement_scale = scale;
        out.displacement_offset = offset;
        out.displacement_min = offset.min(offset + scale);
        out.displacement_max = offset.max(offset + scale);
    }
    out
}

/// The one height `e`, a material's displacement extension, states everywhere, if it has no
/// texture. The factor's default of 1 scales a texture: without one, no factor is no value.
fn constant_height(e: &Value) -> Option<f32> {
    e.get("displacementGeometryTexture").is_none().then(|| {
        num(e, "displacementGeometryFactor", 0.0) + num(e, "displacementGeometryOffset", 0.0)
    })
}

/// `iridescenceThicknessMinimum` and `iridescenceThicknessMaximum`, in nanometers.
fn film_thickness(e: &Value) -> (f32, f32) {
    (
        num(e, "iridescenceThicknessMinimum", 100.0),
        num(e, "iridescenceThicknessMaximum", 400.0),
    )
}

/// The `KHR_texture_transform` of a textureInfo object in `Material::uv_transforms` form, and
/// the uv set it reads: the extension's `texCoord`, else the object's. `T R S`, with the uvs
/// turning counter-clockwise about their upper left origin, so a positive rotation takes +u
/// toward -v and turns the image clockwise.
fn uv_transform(info: &Value) -> ([f32; 6], u32) {
    let coord = |v: &Value| v["texCoord"].as_u64().map(|c| c as u32);
    let Some(t) = info["extensions"].get("KHR_texture_transform") else {
        return ([0.0; 6], coord(info).unwrap_or(0));
    };
    let pair = |k: &str, d: [f64; 2]| {
        t.get(k)
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .map_or(d, |a| std::array::from_fn(|i| a[i].as_f64().unwrap_or(0.0)))
    };
    let [ox, oy] = pair("offset", [0.0, 0.0]);
    let [sx, sy] = pair("scale", [1.0, 1.0]);
    // An f32 sine or cosine misses a quarter turn's zero, a shear that a large scale shows.
    let rotation = t.get("rotation").and_then(Value::as_f64).unwrap_or(0.0);
    let (sin, cos) = rotation.sin_cos();
    let uv = [cos * sx - 1.0, sin * sy, ox, -sin * sx, cos * sy - 1.0, oy];
    (
        uv.map(|v| v as f32),
        coord(t).or_else(|| coord(info)).unwrap_or(0),
    )
}

pub(super) fn textures(
    doc: &gltf::Document,
    json: &Value,
    m: &gltf::Material<'_>,
) -> Result<MaterialTextures, String> {
    let own = m.index().map_or(&Value::Null, |i| &json["materials"][i]);
    // A texture and the textureInfo object that binds it.
    let tex = |t: gltf::Texture<'_>, info: &Value, channel: u8| -> Result<TexRef, String> {
        let (uv, coord) = uv_transform(info);
        if coord > 1 {
            return Err(format!(
                "texCoord {coord}: meshes carry TEXCOORD_0 and TEXCOORD_1"
            ));
        }
        let wrap = |w: WrappingMode| match w {
            WrappingMode::Repeat => Wrap::Repeat,
            WrappingMode::ClampToEdge => Wrap::Clamp,
            WrappingMode::MirroredRepeat => Wrap::Mirror,
        };
        let s = t.sampler();
        // An unset filter is the renderer's to choose: linear.
        let mag = match s.mag_filter() {
            Some(MagFilter::Nearest) => filter::MAG_NEAREST,
            _ => 0,
        };
        let min = match s.min_filter() {
            Some(MinFilter::Nearest) => filter::MIN_NEAREST | filter::MIP_NEAREST | filter::ONE_MIP,
            Some(MinFilter::Linear) => filter::MIP_NEAREST | filter::ONE_MIP,
            Some(MinFilter::NearestMipmapNearest) => filter::MIN_NEAREST | filter::MIP_NEAREST,
            Some(MinFilter::LinearMipmapNearest) => filter::MIP_NEAREST,
            Some(MinFilter::NearestMipmapLinear) => filter::MIN_NEAREST,
            Some(MinFilter::LinearMipmapLinear) | None => 0,
        };
        Ok(TexRef {
            image: t.source().index(),
            rules: ImageRules::Gltf,
            channel,
            uv_set: coord as u8,
            wrap: [wrap(s.wrap_s()), wrap(s.wrap_t())],
            color: ImageColor::Data,
            file_wrap: [false; 2],
            filter: mag | min,
            outside: [0.0; 4],
            missing: [0.0; 4],
            rg_channel: None,
            input: None,
            uv,
            remap: None,
        })
    };
    // A textureInfo object of an extension.
    let info = |ext: &str, key: &str, channel: u8| -> Result<Option<TexRef>, String> {
        let Some(i) = extension(json, m, ext).and_then(|e| e.get(key)) else {
            return Ok(None);
        };
        let index = i["index"]
            .as_u64()
            .ok_or_else(|| format!("{ext} {key}: no index"))?;
        let t = doc
            .textures()
            .nth(index as usize)
            .ok_or_else(|| format!("{ext} {key}: texture {index} does not exist"))?;
        tex(t, i, channel).map(Some)
    };
    let pbr = m.pbr_metallic_roughness();
    let pbr_json = &own["pbrMetallicRoughness"];
    let color = |r: TexRef| TexRef {
        color: ImageColor::Gltf,
        ..r
    };
    let base_color = pbr
        .base_color_texture()
        .map(|i| tex(i.texture(), &pbr_json["baseColorTexture"], 0).map(color))
        .transpose()?;
    // An image of red and green alone has no blue: it holds roughness and metalness in
    // red and green.
    let metal_rough = |channel: u8| {
        pbr.metallic_roughness_texture()
            .map(|i| {
                let info = &pbr_json["metallicRoughnessTexture"];
                tex(i.texture(), info, channel).map(|r| TexRef {
                    rg_channel: Some(channel - 1),
                    ..r
                })
            })
            .transpose()
    };
    let mut more = std::collections::BTreeMap::new();
    let mut read = |s: usize, r: Result<Option<TexRef>, String>| {
        more.insert(s, r?);
        Ok::<_, String>(())
    };
    let specular = "KHR_materials_specular";
    read(
        slot::SPECULAR_COLOR,
        info(specular, "specularColorTexture", 0).map(|r| r.map(color)),
    )?;
    let coat = "KHR_materials_clearcoat";
    read(
        slot::COAT_ROUGHNESS,
        info(coat, "clearcoatRoughnessTexture", 1),
    )?;
    let sheen = "KHR_materials_sheen";
    read(
        slot::FUZZ_COLOR,
        info(sheen, "sheenColorTexture", 0).map(|r| r.map(color)),
    )?;
    read(
        slot::FUZZ_ROUGHNESS,
        info(sheen, "sheenRoughnessTexture", 3),
    )?;
    let anisotropy = "KHR_materials_anisotropy";
    read(slot::ANISOTROPY, info(anisotropy, "anisotropyTexture", 2))?;
    read(slot::TANGENT, info(anisotropy, "anisotropyTexture", 0))?;
    let film = "KHR_materials_iridescence";
    read(slot::THIN_FILM_WEIGHT, info(film, "iridescenceTexture", 0))?;
    let thickness = info(film, "iridescenceThicknessTexture", 1)?.map(|r| {
        let (low, high) = extension(json, m, film).map_or((0.0, 0.0), film_thickness);
        let peak = low.max(high);
        let span = if peak > 0.0 {
            [low / peak, (high - low) / peak]
        } else {
            [0.0, 0.0]
        };
        TexRef {
            remap: Some([span; 3]),
            ..r
        }
    });
    read(slot::THIN_FILM_THICKNESS, Ok(thickness))?;
    read(
        slot::SUBSURFACE_WEIGHT,
        info(
            "KHR_materials_diffuse_transmission",
            "diffuseTransmissionTexture",
            3,
        ),
    )?;
    let coat_normal = extension(json, m, coat)
        .and_then(|e| e.get("clearcoatNormalTexture"))
        .map(|i| {
            let r = info(coat, "clearcoatNormalTexture", 0)?;
            let r = r.expect("the object is there");
            Ok::<_, String>((r, [num(i, "scale", 1.0); 2]))
        })
        .transpose()?;
    Ok(MaterialTextures {
        base_color,
        opacity: base_color.map(|r| TexRef {
            channel: 3,
            color: ImageColor::Data,
            ..r
        }),
        roughness: metal_rough(1)?,
        metalness: metal_rough(2)?,
        normal: m
            .normal_texture()
            .map(|i| {
                let r = tex(i.texture(), &own["normalTexture"], 0)?;
                Ok::<_, String>((r, [i.scale(); 2]))
            })
            .transpose()?,
        emissive: m
            .emissive_texture()
            .map(|i| tex(i.texture(), &own["emissiveTexture"], 0).map(color))
            .transpose()?,
        specular: info("KHR_materials_specular", "specularTexture", 3)?,
        coat: info("KHR_materials_clearcoat", "clearcoatTexture", 0)?,
        transmission: info("KHR_materials_transmission", "transmissionTexture", 0)?,
        subsurface_color: info(
            "KHR_materials_diffuse_transmission",
            "diffuseTransmissionColorTexture",
            0,
        )?
        .map(color),
        displacement: match extension(json, m, "KHR_materials_displacement") {
            Some(e) if constant_height(e).is_some_and(|h| h != 0.0) => Some(TexRef {
                image: doc.images().len(),
                rules: ImageRules::Gltf,
                channel: 0,
                rg_channel: None,
                input: None,
                uv_set: 0,
                wrap: [Wrap::Repeat; 2],
                file_wrap: [false; 2],
                filter: 0,
                outside: [0.0; 4],
                missing: [0.0; 4],
                uv: [0.0; 6],
                color: ImageColor::Data,
                remap: None,
            }),
            _ => info(
                "KHR_materials_displacement",
                "displacementGeometryTexture",
                0,
            )?,
        },
        coat_normal,
        more,
    })
}
