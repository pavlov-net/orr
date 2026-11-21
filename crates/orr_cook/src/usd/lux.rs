//! UsdLux lights as layout light records ([`LightSource`]), in scene space.
//!
//! Every value converts as the UsdLux schema of OpenUSD v26.05 defines it:
//! `pxr/usd/usdLux/schema.usda` (cited by its class and attribute) and `overview.dox`, with the
//! `hdEmbree` delegate, which `overview.dox` names as the reference implementation, where the
//! text leaves a formula's details to it. An attribute not authored takes the fallback its schema
//! declares, ShadowAPI's `inputs:shadow:enable` `true` without the schema applied.
//!
//! - Emission is `L = intensity · 2^exposure · color / sizeFactor` (LightAPI `inputs:intensity`,
//!   `inputs:exposure`, `inputs:color`), luminance in nits, where `sizeFactor` is 1 without
//!   `inputs:normalize`, and with it a SphereLight's or RectLight's world-space surface area
//!   (m², the stage's `metersPerUnit` and the light's transform applied) and a DistantLight's
//!   `π sin²θmax`.
//! - A SphereLight emits `L` from its sphere: `L π r²` candela toward any point outside it, which
//!   the scene's sphere takes. With `treatAsPoint` it is a point of that intensity, the sphere it
//!   stands for as seen from outside it, and so is a sphere of radius 0.
//! - `inputs:enableColorTemperature` multiplies `inputs:color` by `inputs:colorTemperature`'s
//!   color: `UsdLuxBlackbodyTemperatureAsRgb`'s linear Rec.709 color, D65 white, "normalized
//!   such that the default value of 6500 will always result in white", so divided by its color
//!   at 6500 K.
//! - ShadowAPI `inputs:shadow:enable = false` makes a sphere, point or rect light one surfaces
//!   cast no shadow from.
//! - A RectLight emits `L` from one side of its `width` x `height` rectangle in its XY plane,
//!   along its −Z axis.
//! - A DistantLight shines along its −Z axis from a disc of angular diameter `inputs:angle`
//!   (degrees, clipped to [0, 360)): with `θmax = angle / 2`, a surface facing it receives
//!   `L π sin²θmax` lux, `intensity · 2^exposure` with `normalize`.
//! - `ShapingAPI` shapes a sphere, point or rect light around its −Z axis as
//!   `orr_pack::layout::ShapingSource` states, `inputs:shaping:cone:softness` clamped to
//!   [0, 1] and an `inputs:shaping:focus` below 0 ignored, as the schema says.
//! - `inputs:color` and `inputs:shaping:focusTint` are in the color space `UsdColorSpaceAPI`
//!   resolves for them ([`PrimReader::color`]) and convert to the linear Rec.2020 layout records
//!   take; the color temperature's color multiplies in linear Rec.709, the space it is in.
//! - A `Mesh` with `LightAPI` (`MeshLightAPI`) is no record: its faces emit as their bound
//!   materials do, which is the light where its terms leave that glow as it is ([`mesh_light`]).
//!
//! What a light states that the scene has no term for fails the import, naming the prim: other
//! light types, light filters and linking, IES profiles, rect textures, ShadowAPI's shadow
//! color, distance and falloff, a DistantLight without shadows, the non-physical
//! `inputs:diffuse` and `inputs:specular`, an input whose time samples change, and a mesh light
//! that shows light rays other than the glow of its faces.

use openusd::{usd, vt};
use orr_color::rec709_to_rec2020;
use orr_math::{DMat3, DMat4, DVec3, Vec3};
use orr_pack::layout::{LightSource, ShapingSource};

use openusd_schemas::lux::{self as usd_lux, blackbody_rgb};

use super::read::{self, PrimReader};

/// Whether `prim` is a light: one that has `LightAPI`, which every light type and the API
/// schemas that make other prims lights include.
pub(super) fn is_light(prim: &usd::Prim) -> Result<bool, String> {
    let light = prim.has_api_schema(usd_lux::tokens::LIGHT_API);
    light.map_err(|e| format!("{}: {e}", prim.path().as_str()))
}

/// The bits of a float a record keeps; two values a record can't tell apart are the same.
const RECORD_EPSILON: f64 = 4.0 * f32::EPSILON as f64;

/// Light input `name` at the import's time; one whose samples change over time fails, as the
/// record holds one value.
fn input(r: &PrimReader<'_>, name: &str) -> Result<Option<vt::Value>, String> {
    if r.varies(name)? {
        return Err(format!(
            "{} {name}: a light input sampled over time is not cooked",
            r.path()
        ));
    }
    r.value(name)
}

fn numbers(r: &PrimReader<'_>, name: &str) -> Result<Option<Vec<f64>>, String> {
    input(r, name)?
        .map(|v| read::numbers(&v).ok_or_else(|| format!("{} {name}: not numbers", r.path())))
        .transpose()
}

/// An input the light's schema declares a fallback for, which stands where none is authored.
fn held(r: &PrimReader<'_>, name: &str) -> Result<Vec<f64>, String> {
    numbers(r, name)?.ok_or_else(|| format!("{} {name}: no value", r.path()))
}

fn number(r: &PrimReader<'_>, name: &str) -> Result<f64, String> {
    Ok(held(r, name)?[0])
}

fn triple(r: &PrimReader<'_>, name: &str) -> Result<DVec3, String> {
    match held(r, name)?[..] {
        [x, y, z] => Ok(DVec3::new(x, y, z)),
        _ => Err(format!("{} {name}: not three numbers", r.path())),
    }
}

/// A bool input, `None` where nothing states one: an API schema that is not applied declares
/// no fallback.
fn bool_input(r: &PrimReader<'_>, name: &str) -> Result<Option<bool>, String> {
    match input(r, name)? {
        None => Ok(None),
        Some(vt::Value::Bool(b)) => Ok(Some(b)),
        Some(_) => Err(format!("{} {name}: not a bool", r.path())),
    }
}

/// A bool input the light's schema declares a fallback for.
fn flag(r: &PrimReader<'_>, name: &str) -> Result<bool, String> {
    bool_input(r, name)?.ok_or_else(|| format!("{} {name}: no value", r.path()))
}

/// Whether `name` holds an opinion other than `fallback`, its schema fallback.
fn departs(r: &PrimReader<'_>, name: &str, fallback: &[f64]) -> Result<bool, String> {
    Ok(numbers(r, name)?.is_some_and(|v| v != fallback))
}

/// The linear part of `to_scene` as a uniform scale, or `None` where it stretches some
/// direction more than another past what a record holds.
fn uniform_scale(l: DMat3) -> Option<f64> {
    let g = l.transpose() * l;
    let s2 = (g.x_axis.x + g.y_axis.y + g.z_axis.z) / 3.0;
    let off = (g - DMat3::from_diagonal(DVec3::splat(s2))).to_cols_array();
    off.iter()
        .all(|x| x.abs() <= RECORD_EPSILON * s2)
        .then(|| s2.sqrt())
}

/// Light `r`, a prim of light type `ty`, whose space `to_scene` takes to scene space.
pub(super) fn light(r: &PrimReader<'_>, ty: &str, to_scene: DMat4) -> Result<LightSource, String> {
    let path = r.path();
    let fail = |what: &str| Err(format!("{path}: {what}"));
    if !matches!(ty, "DistantLight" | "SphereLight" | "RectLight") {
        return fail(&format!(
            "{ty} lights are not cooked; DistantLight, SphereLight and RectLight are"
        ));
    }
    unsupported(r)?;
    let normalize = flag(r, "inputs:normalize")?;
    let emitted = emission(r)?;
    // ShadowAPI `inputs:shadow:enable`.
    let shadows = bool_input(r, "inputs:shadow:enable")?.unwrap_or(true);
    let l = DMat3::from_mat4(to_scene);
    // Lights with a primary axis emit along -Z (`overview.dox`).
    let axis = (l * DVec3::NEG_Z).normalize();
    let shaping = shaping(r, axis)?;
    let position = to_scene.w_axis.truncate();
    match ty {
        "DistantLight" => {
            if shaping.is_some() {
                return fail("ShapingAPI on a DistantLight is not cooked");
            }
            if !shadows {
                return fail("a DistantLight surfaces cast no shadow from is not cooked");
            }
            // `inputs:angle` is clipped to [0, 360); `normalize`'s θmax is its half, clamped to
            // [0, π].
            let angle = number(r, "inputs:angle")?.clamp(0.0, 360.0);
            let theta = (angle.to_radians() / 2.0).clamp(0.0, std::f64::consts::PI);
            if theta == 0.0 || theta > std::f64::consts::FRAC_PI_2 {
                return fail(&format!(
                    "inputs:angle {angle}: the scene's distant lights are discs of 0 to 180 \
                     degrees, not parallel light or wider"
                ));
            }
            let projected = std::f64::consts::PI * theta.sin().powi(2);
            let illuminance = if normalize {
                emitted
            } else {
                emitted * projected
            };
            Ok(LightSource::Directional {
                direction_to_light: (-axis).as_vec3(),
                illuminance: illuminance.as_vec3(),
                angular_radius: theta as f32,
            })
        }
        "SphereLight" => {
            let radius = number(r, "inputs:radius")?;
            let Some(scale) = uniform_scale(l) else {
                return fail("its transform stretches the sphere unevenly");
            };
            let radius = radius * scale;
            let disc = std::f64::consts::PI * radius * radius;
            // `inputs:normalize` divides the luminance by the sphere's area, 4π r², "such that the
            // power of the light remains constant while altering the size of the light": L π r²
            // is a quarter of the emission at every radius above 0, so the radius-0 sphere, whose
            // area divides by 0, is that constant's limit.
            let intensity = match normalize {
                true => emitted / 4.0,
                false => emitted * disc,
            };
            let intensity = intensity.as_vec3();
            if flag(r, "treatAsPoint")? || radius == 0.0 {
                Ok(LightSource::Point {
                    position,
                    intensity,
                    shaping,
                    shadows,
                })
            } else {
                Ok(LightSource::Sphere {
                    position,
                    radius: radius as f32,
                    intensity,
                    shaping,
                    shadows,
                })
            }
        }
        _ => {
            if input(r, "inputs:texture:file")?.is_some() {
                return fail("a RectLight's inputs:texture:file is not cooked");
            }
            let ex = l * DVec3::new(number(r, "inputs:width")?, 0.0, 0.0);
            let mut ey = l * DVec3::new(0.0, number(r, "inputs:height")?, 0.0);
            if ex.dot(ey).abs() > RECORD_EPSILON * ex.length() * ey.length() {
                return fail("its transform shears the rectangle");
            }
            let n = ex.cross(ey);
            let area = n.length();
            // The scene's rect emits toward edge_x x edge_y.
            if n.dot(axis) < 0.0 {
                ey = -ey;
            }
            let luminance = if normalize { emitted / area } else { emitted };
            Ok(LightSource::Rect {
                center: position,
                edge_x: ex.as_vec3(),
                edge_y: ey.as_vec3(),
                luminance: luminance.as_vec3(),
                shaping,
                shadows,
            })
        }
    }
}

/// `intensity · 2^exposure · color`, `color` multiplied by its temperature's where that is
/// enabled.
fn emission(r: &PrimReader<'_>) -> Result<DVec3, String> {
    // LightAPI `inputs:intensity`, `inputs:exposure`, `inputs:color`.
    let intensity = number(r, "inputs:intensity")?;
    let exposure = number(r, "inputs:exposure")?;
    let color = r.color("inputs:color", triple(r, "inputs:color")?)?;
    // LightAPI `inputs:enableColorTemperature`, `inputs:colorTemperature`: "the computed result
    // multiplies against the color attribute".
    let color = match flag(r, "inputs:enableColorTemperature")? {
        true => color * temperature_color(number(r, "inputs:colorTemperature")?),
        false => color,
    };
    Ok(rec709_to_rec2020(color) * (intensity * exposure.exp2()))
}

/// Checks mesh light `r`, a `Mesh` with `LightAPI` (which `MeshLightAPI` applies), whose
/// emission is its faces' materials': it cooks where the light's terms leave the glow of its
/// bound material as it is.
///
/// LightAPI `light:materialSyncMode` (`materialGlowTintsLight` under `MeshLightAPI`,
/// `noMaterialResponse` otherwise): with `materialGlowTintsLight` "all primary and secondary
/// rays see the emissive/glow response as dictated by the bound Material while the base color
/// seen by light rays (which is then modulated by all of the other LightAPI controls) is the
/// multiplication of the color feeding the emission/glow input of the Material [...] with the
/// scalar or pattern input to *inputs:color*". The scene's emitters show light rays what other
/// rays see, so only a light whose terms multiply that glow by 1 cooks: `intensity · 2^exposure
/// · color` (and its temperature's color) of 1, without `inputs:normalize`, ShapingAPI or
/// shadows turned off. Where no face's material `glows`, light rays see none of the light's
/// terms, as no rays do. The other modes give light rays the light's own color, which differs
/// from what the bound Material shows.
pub(super) fn mesh_light(r: &PrimReader<'_>, glows: bool) -> Result<(), String> {
    let path = r.path();
    let fail = |what: &str| Err(format!("{path}: {what}"));
    let mode = r.token("light:materialSyncMode")?;
    let mode = mode.ok_or_else(|| format!("{path} light:materialSyncMode: no value"))?;
    match mode.as_str() {
        "materialGlowTintsLight" => {}
        "noMaterialResponse" => {
            return fail(
                "a mesh light of light:materialSyncMode noMaterialResponse is not cooked: its \
                 faces show the light's own emission and no response of their material",
            );
        }
        _ => {
            return fail(&format!(
                "a mesh light of light:materialSyncMode {mode} is not cooked: its light rays see \
                 the light's own color, not the glow its faces show"
            ));
        }
    }
    if !glows {
        return Ok(());
    }
    unsupported(r)?;
    if flag(r, "inputs:normalize")? {
        return fail(
            "a mesh light's inputs:normalize is not cooked: it divides what light rays see by \
             the surface area of each copy",
        );
    }
    let tint = emission(r)?;
    if (tint - DVec3::ONE).abs().max_element() > RECORD_EPSILON {
        return fail(&format!(
            "a mesh light that multiplies its material's glow by {tint} for light rays is not \
             cooked: other rays see the glow as it is"
        ));
    }
    if r.prim
        .has_api_schema("ShapingAPI")
        .map_err(|e| e.to_string())?
    {
        return fail("ShapingAPI on a mesh light is not cooked");
    }
    if bool_input(r, "inputs:shadow:enable")? == Some(false) {
        return fail("a mesh light surfaces cast no shadow from is not cooked");
    }
    Ok(())
}

/// The color `inputs:colorTemperature` multiplies `inputs:color` by, white at 6500 K.
fn temperature_color(kelvin: f64) -> DVec3 {
    let rgb = |k: f32| Vec3::from(<[f32; 3]>::from(blackbody_rgb(k))).as_dvec3();
    rgb(kelvin as f32) / rgb(6500.0)
}

/// `ShapingAPI`'s shaping of a light whose −Z axis is `axis` in scene space, where it is applied.
fn shaping(r: &PrimReader<'_>, axis: DVec3) -> Result<Option<ShapingSource>, String> {
    if !r
        .prim
        .has_api_schema("ShapingAPI")
        .map_err(|e| e.to_string())?
    {
        return Ok(None);
    }
    if input(r, "inputs:shaping:ies:file")?.is_some() {
        return Err(format!(
            "{}: an IES profile (inputs:shaping:ies:file) is not cooked",
            r.path()
        ));
    }
    // `inputs:shaping:cone:angle` in degrees, `cone:softness` "clamped to [0, 1]", `focus`
    // whose "values < 0 are ignored", `focusTint`.
    let focus = number(r, "inputs:shaping:focus")?;
    Ok(Some(ShapingSource {
        axis: axis.as_vec3(),
        cone_angle: number(r, "inputs:shaping:cone:angle")?.to_radians() as f32,
        cone_softness: number(r, "inputs:shaping:cone:softness")?.clamp(0.0, 1.0) as f32,
        focus: focus.max(0.0) as f32,
        focus_tint: rec709_to_rec2020(r.color(
            "inputs:shaping:focusTint",
            triple(r, "inputs:shaping:focusTint")?,
        )?)
        .as_vec3(),
    }))
}

/// Fails on what light `r` states that the scene has no term for.
fn unsupported(r: &PrimReader<'_>) -> Result<(), String> {
    let path = r.path();
    // Non-physical multipliers (LightAPI `inputs:diffuse`, `inputs:specular`) and ShadowAPI's
    // controls, at their fallbacks a light the scene renders.
    for (name, fallback) in [
        ("inputs:diffuse", &[1.0][..]),
        ("inputs:specular", &[1.0]),
        ("inputs:shadow:color", &[0.0, 0.0, 0.0]),
        ("inputs:shadow:distance", &[-1.0]),
        ("inputs:shadow:falloff", &[-1.0]),
        ("inputs:shadow:falloffGamma", &[1.0]),
    ] {
        if departs(r, name, fallback)? {
            return Err(format!("{path}: {name} is not cooked"));
        }
    }
    let fail = |e: openusd::Error| format!("{path}: {e}");
    let light = usd_lux::LightAPI::from_prim_unchecked(r.prim.clone());
    if !light.filters_rel().targets().map_err(fail)?.is_empty() {
        return Err(format!("{path}: light filters are not cooked"));
    }
    // Linking collections, which include the root by default and list nothing.
    for c in [
        light.light_link_collection(),
        light.shadow_link_collection(),
    ] {
        let narrowed = c.include_root().map_err(fail)? == Some(false)
            || !c.includes().map_err(fail)?.is_empty()
            || !c.excludes().map_err(fail)?.is_empty()
            || c.membership_expression_attr()
                .has_authored_value()
                .map_err(fail)?;
        if narrowed {
            return Err(format!(
                "{path}: collection:{} narrows what the light reaches, which is not cooked",
                c.name()
            ));
        }
    }
    Ok(())
}
