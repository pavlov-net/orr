//! UsdLux lights as layout light records ([`LightSource`]), in scene space.
//!
//! Every value converts as the UsdLux schema of OpenUSD v26.05 defines it, the release whose
//! schemas `openusd` 0.8.0 carries: `pxr/usd/usdLux/schema.usda` (cited by its class and
//! attribute) and `overview.dox`, with the `hdEmbree` delegate, which `overview.dox` names as
//! the reference implementation, where the text leaves a formula's details to it. An attribute
//! not authored takes the fallback that class states.
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
//!   color ([`super::blackbody`]).
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
//! - Colors are in the stage's color space: linear Rec.709 where none is authored
//!   (`UsdColorSpaceAPI`, `pxr/usd/usd/schema.usda`), the space layout records take.
//!
//! What a light states that the scene has no term for fails the import, naming the prim: other
//! light types, light filters and linking, IES profiles, rect textures, ShadowAPI's shadow
//! color, distance and falloff, a DistantLight without shadows, the non-physical
//! `inputs:diffuse` and `inputs:specular`, and an input whose time samples change.

use openusd::sdf;
use orr_math::{DMat3, DMat4, DVec3};
use orr_pack::layout::{LightSource, ShapingSource};

use super::read::{self, PrimReader};

/// The UsdLux concrete light types (`schema.usda`).
pub(super) const LIGHT_TYPES: &[&str] = &[
    "CylinderLight",
    "DiskLight",
    "DistantLight",
    "DomeLight",
    "DomeLight_1",
    "GeometryLight",
    "PluginLight",
    "PortalLight",
    "RectLight",
    "SphereLight",
];

/// API schemas that make another prim a light.
const LIGHT_APIS: &[&str] = &["LightAPI", "MeshLightAPI", "VolumeLightAPI"];

/// Whether `prim`, of no light type, is made a light by an applied API schema.
pub(super) fn light_api(prim: &openusd::usd::Prim) -> Result<Option<String>, String> {
    let applied = prim.api_schemas().map_err(|e| e.to_string())?;
    Ok(applied
        .iter()
        .find(|a| LIGHT_APIS.contains(&a.as_str()))
        .map(|a| a.as_str().to_owned()))
}

/// The bits of a float a record keeps; two values a record can't tell apart are the same.
const RECORD_EPSILON: f64 = 4.0 * f32::EPSILON as f64;

/// Light input `name` at the import's time; one whose samples change over time fails, as the
/// record holds one value.
fn input(r: &PrimReader<'_>, name: &str) -> Result<Option<sdf::Value>, String> {
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

fn number(r: &PrimReader<'_>, name: &str, fallback: f64) -> Result<f64, String> {
    Ok(numbers(r, name)?.map_or(fallback, |v| v[0]))
}

fn triple(r: &PrimReader<'_>, name: &str, fallback: [f64; 3]) -> Result<DVec3, String> {
    match numbers(r, name)? {
        None => Ok(DVec3::from_array(fallback)),
        Some(v) if v.len() == 3 => Ok(DVec3::new(v[0], v[1], v[2])),
        Some(_) => Err(format!("{} {name}: not three numbers", r.path())),
    }
}

fn flag(r: &PrimReader<'_>, name: &str, fallback: bool) -> Result<bool, String> {
    match input(r, name)? {
        None => Ok(fallback),
        Some(sdf::Value::Bool(b)) => Ok(b),
        Some(_) => Err(format!("{} {name}: not a bool", r.path())),
    }
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

/// The color space `inputs:color` is in, where one is authored: on the attribute, else by
/// `UsdColorSpaceAPI` on the prim or its nearest ancestor that states one.
fn color_space(r: &PrimReader<'_>) -> Result<Option<String>, String> {
    if let Some(space) = r.metadata_token("inputs:color", "colorSpace") {
        return Ok(Some(space));
    }
    let mut at = Some(r.prim.clone());
    while let Some(p) = at {
        let reader = PrimReader {
            prim: &p,
            time: r.time,
        };
        if let Some(space) = reader.token("colorSpace:name")? {
            return Ok(Some(space));
        }
        at = p.parent();
    }
    Ok(None)
}

/// Light `r`, a prim of light type `ty`, whose space `to_scene` takes to scene space.
pub(super) fn light(r: &PrimReader<'_>, ty: &str, to_scene: DMat4) -> Result<LightSource, String> {
    let path = r.path();
    let fail = |what: &str| Err(format!("{path}: {what}"));
    if !matches!(ty, "DistantLight" | "SphereLight" | "RectLight") {
        return fail(&format!(
            "a {ty} is not cooked; DistantLight, SphereLight and RectLight are"
        ));
    }
    unsupported(r)?;
    // LightAPI `inputs:intensity` (DistantLight overrides its fallback), `inputs:exposure`,
    // `inputs:color`.
    let intensity = number(
        r,
        "inputs:intensity",
        match ty {
            "DistantLight" => 50000.0,
            _ => 1.0,
        },
    )?;
    let exposure = number(r, "inputs:exposure", 0.0)?;
    let color = triple(r, "inputs:color", [1.0; 3])?;
    if let Some(space) = color_space(r)?
        && space != "lin_rec709_scene"
    {
        return fail(&format!(
            "inputs:color in color space {space}, not linear Rec.709"
        ));
    }
    // LightAPI `inputs:enableColorTemperature`, `inputs:colorTemperature`: "the computed result
    // multiplies against the color attribute".
    let color = match flag(r, "inputs:enableColorTemperature", false)? {
        true => {
            color
                * super::blackbody::temperature_color(number(r, "inputs:colorTemperature", 6500.0)?)
        }
        false => color,
    };
    let normalize = flag(r, "inputs:normalize", false)?;
    let emitted = color * (intensity * exposure.exp2());
    // ShadowAPI `inputs:shadow:enable`.
    let shadows = flag(r, "inputs:shadow:enable", true)?;
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
            let angle = number(r, "inputs:angle", 0.53)?.clamp(0.0, 360.0);
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
            let radius = number(r, "inputs:radius", 0.5)?;
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
            if flag(r, "treatAsPoint", false)? || radius == 0.0 {
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
            let ex = l * DVec3::new(number(r, "inputs:width", 1.0)?, 0.0, 0.0);
            let mut ey = l * DVec3::new(0.0, number(r, "inputs:height", 1.0)?, 0.0);
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
    let focus = number(r, "inputs:shaping:focus", 0.0)?;
    Ok(Some(ShapingSource {
        axis: axis.as_vec3(),
        cone_angle: number(r, "inputs:shaping:cone:angle", 90.0)?.to_radians() as f32,
        cone_softness: number(r, "inputs:shaping:cone:softness", 0.0)?.clamp(0.0, 1.0) as f32,
        focus: focus.max(0.0) as f32,
        focus_tint: triple(r, "inputs:shaping:focusTint", [0.0; 3])?.as_vec3(),
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
    if !r.targets("light:filters")?.is_empty() {
        return Err(format!("{path}: light filters are not cooked"));
    }
    // Linking collections, which include the root by default and list nothing.
    for link in ["lightLink", "shadowLink"] {
        let c = format!("collection:{link}");
        let narrowed = !flag(r, &format!("{c}:includeRoot"), true)?
            || !r.targets(&format!("{c}:includes"))?.is_empty()
            || !r.targets(&format!("{c}:excludes"))?.is_empty()
            || input(r, &format!("{c}:membershipExpression"))?.is_some();
        if narrowed {
            return Err(format!(
                "{path}: {c} narrows what the light reaches, which is not cooked"
            ));
        }
    }
    Ok(())
}
