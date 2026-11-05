//! UsdSkel: one `Skeleton`, the skin bindings of the meshes bound to it, and
//! `SkelAnimation`s as its clips.
//!
//! Joints keep the skeleton's order, which must list parents first; a joint's parent is its
//! nearest ancestor path among them, a joint without one being a root. Model space is the
//! stage's (Y up, meters): [`SkeletonInfo::root`] is the skeleton prim's world transform,
//! inverse binds are those of the bind transforms in model space, and rest transforms come
//! from `restTransforms` (from the bind transforms without them). A bind transform that
//! collapses an axis takes its pseudo-inverse, so the points bound to it land on the surface
//! it flattens to. Skinned points are `geomBindTransform` (unset if it is not a matrix) times
//! the authored points in the skeleton's model space, or its own space in scene units and axes
//! where the skeleton prim's world transform collapses an axis and has no inverse for the binds
//! to undo; the mesh prim's own transform is not used, as UsdSkel specifies. A mesh's
//! `skel:joints` maps its joint indices to skeleton joints by name. An animation that names one
//! of the joints is a clip, named after its prim (by its index among the skeleton's animations
//! when the name is shared): at each time any of its attributes is sampled, the authored
//! joints' local transforms, sampled linearly between those times (rotations by slerp).

use std::collections::HashMap;

use openusd::usd;
use orr_math::{Affine3, DMat3, DMat4, DVec3, DVec4};
use orr_pack::clip::Clip;
use orr_pack::skeleton::{MAX_JOINTS, NO_PARENT, SkeletonInfo, Trs};

use super::mesh::Influence;
use super::read::{PrimReader, matrix, quat};
use crate::import::collapses;
use crate::skin::{self, Channel, ClipSettings, Interpolation, Property};

pub(super) struct Skel {
    pub(super) info: SkeletonInfo,
    /// Joint paths as the skeleton names them.
    joints: Vec<String>,
    by_name: HashMap<String, u16>,
    /// Bind space from skeleton space: model space if the skeleton's root has an inverse, else
    /// the skeleton's space in scene units and axes.
    to_bind: DMat4,
}

/// The path above joint path `joint`.
fn above(joint: &str) -> Option<&str> {
    joint.rsplit_once('/').map(|a| a.0)
}

/// A transform as translation, rotation and scale, `None` if it shears.
fn trs(m: DMat4) -> Option<Trs> {
    let (s, r, t) = skin::decompose(m.as_mat4())?;
    Some(Trs {
        rotation: r.normalize().to_array(),
        translation: t.to_array(),
        scale: s.to_array(),
    })
}

/// The Moore-Penrose inverse of rank-deficient `l`, from the characteristic polynomial of
/// `l^T l` (Decell, 1965). The second invariant separates ranks 2 and 1 at the ratio of singular
/// values that [`collapses`] takes for rounding, squared as `l^T l` squares them.
fn pseudo_inverse(l: DMat3) -> DMat3 {
    let trace = |m: DMat3| m.x_axis.x + m.y_axis.y + m.z_axis.z;
    let g = l.transpose() * l;
    let t = trace(g);
    let invariant = (t * t - trace(g * g)) / 2.0;
    if invariant > t * t * 1e-10 {
        (DMat3::from_diagonal(DVec3::splat(t)) - g) * l.transpose() * invariant.recip()
    } else if t > 0.0 {
        l.transpose() * t.recip()
    } else {
        DMat3::ZERO
    }
}

/// The inverse of a bind transform; where it collapses an axis, the transform that takes a
/// point to the nearest one the bind reaches and back through it.
fn bind_inverse(m: DMat4) -> DMat4 {
    let l = DMat3::from_mat4(m);
    if collapses(l) {
        DMat4::from_mat3(pseudo_inverse(l)) * DMat4::from_translation(-m.w_axis.truncate())
    } else {
        m.inverse()
    }
}

impl Skel {
    /// The skeleton of prim `r`, at world transform `to_model` on a stage that `stage` takes
    /// to scene units and axes.
    pub(super) fn build(r: &PrimReader<'_>, to_model: DMat4, stage: DMat4) -> Result<Self, String> {
        let path = r.path().to_string();
        if !to_model.as_mat4().is_finite() {
            return Err(format!("{path}: its world transform is not finite"));
        }
        let joints = r.tokens("joints")?;
        if joints.is_empty() || joints.len() > MAX_JOINTS {
            return Err(format!(
                "{path}: {} joints; 1 to {MAX_JOINTS}",
                joints.len()
            ));
        }
        let mats = |name: &str| -> Result<Option<Vec<DMat4>>, String> {
            let Some(v) = r.numbers(name)? else {
                return Ok(None);
            };
            if v.len() != 16 * joints.len() {
                return Err(format!("{path} {name}: not one matrix per joint"));
            }
            let m: Vec<DMat4> = v.chunks_exact(16).map(matrix).collect();
            if !m
                .iter()
                .all(|m| m.is_finite() && m.row(3).abs_diff_eq(DVec4::W, 1e-5))
            {
                return Err(format!("{path} {name}: not all affine transforms"));
            }
            Ok(Some(m))
        };
        let binds = mats("bindTransforms")?.ok_or_else(|| format!("{path}: no bindTransforms"))?;
        let by_name: HashMap<String, u16> = joints
            .iter()
            .enumerate()
            .map(|(i, j)| (j.clone(), i as u16))
            .collect();
        let mut parents = Vec::with_capacity(joints.len());
        for (i, j) in joints.iter().enumerate() {
            let parent = std::iter::successors(above(j), |p| above(p))
                .find_map(|a| by_name.get(a).copied())
                .unwrap_or(NO_PARENT);
            if parent != NO_PARENT && usize::from(parent) >= i {
                return Err(format!("{path}: joint {j} comes before its parent"));
            }
            parents.push(parent);
        }
        let rest = match mats("restTransforms")? {
            Some(r) => r,
            None => (0..joints.len())
                .map(|i| match parents[i] {
                    NO_PARENT => binds[i],
                    p => bind_inverse(binds[usize::from(p)]) * binds[i],
                })
                .collect(),
        };
        let to_bind = if collapses(DMat3::from_mat4(to_model)) {
            stage
        } else {
            to_model
        };
        let from_bind = bind_inverse(to_bind);
        let mut info = SkeletonInfo {
            root: Affine3::from_mat4(to_model.as_mat4()),
            parents,
            rest: Vec::with_capacity(joints.len()),
            inverse_binds: Vec::with_capacity(joints.len()),
            names: joints
                .iter()
                .map(|j| j.rsplit('/').next().unwrap_or(j).to_string())
                .collect(),
        };
        for i in 0..joints.len() {
            info.rest.push(trs(rest[i]).ok_or_else(|| {
                format!(
                    "{path} joint {}: shear; joints need translation, rotation and scale",
                    joints[i]
                )
            })?);
            info.inverse_binds.push(Affine3::from_mat4(
                (bind_inverse(binds[i]) * from_bind).as_mat4(),
            ));
        }
        Ok(Self {
            info,
            joints,
            by_name,
            to_bind,
        })
    }

    /// Skin influences per point of a bound mesh, and bind space from its points.
    pub(super) fn binding(
        &self,
        r: &PrimReader<'_>,
        points: usize,
    ) -> Result<(Vec<Influence>, DMat4), String> {
        let path = r.path();
        let map: Vec<u16> = match r.tokens("skel:joints")?.as_slice() {
            [] => (0..self.joints.len() as u16).collect(),
            names => names
                .iter()
                .map(|n| {
                    self.by_name
                        .get(n)
                        .copied()
                        .ok_or_else(|| format!("{path}: skel:joints names {n}, not a joint"))
                })
                .collect::<Result<_, _>>()?,
        };
        let indices = r
            .numbers("primvars:skel:jointIndices")?
            .ok_or_else(|| format!("{path}: no primvars:skel:jointIndices"))?;
        let weights = r
            .numbers("primvars:skel:jointWeights")?
            .ok_or_else(|| format!("{path}: no primvars:skel:jointWeights"))?;
        if indices.len() != weights.len() {
            return Err(format!(
                "{path}: joint indices and weights differ in length"
            ));
        }
        let size = r
            .prim
            .attribute("primvars:skel:jointIndices")
            .get_metadata::<openusd::sdf::Value>("elementSize")
            .ok()
            .flatten()
            .and_then(|v| super::read::numbers(&v))
            .map_or(1, |v| v[0] as usize)
            .max(1);
        let constant = r
            .metadata_token("primvars:skel:jointIndices", "interpolation")
            .as_deref()
            == Some("constant");
        let elements = if constant { 1 } else { points };
        if indices.len() != elements * size {
            return Err(format!(
                "{path}: {} joint influences for {elements} elements of {size}",
                indices.len()
            ));
        }
        // Influences in sets of four, as glTF's JOINTS_n/WEIGHTS_n.
        let sets = size.div_ceil(4);
        let mut influence_sets: Vec<skin::InfluenceSet> = Vec::with_capacity(sets);
        for s in 0..sets {
            let mut js = Vec::with_capacity(elements);
            let mut ws = Vec::with_capacity(elements);
            for e in 0..elements {
                let mut j = [0u16; 4];
                let mut w = [0f32; 4];
                for k in 0..4 {
                    let i = 4 * s + k;
                    if i < size {
                        let index = indices[e * size + i];
                        w[k] = weights[e * size + i] as f32;
                        if index < 0.0 && w[k] != 0.0 {
                            return Err(format!("{path}: joint index {index}"));
                        }
                        j[k] = index as u16;
                    }
                }
                js.push(j);
                ws.push(w);
            }
            influence_sets.push((js, ws));
        }
        let influences: Vec<Influence> = (0..elements)
            .map(|e| skin::influences(&influence_sets, &map, e))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{path}: {e}"))?;
        let influences = if constant {
            vec![influences[0]; points]
        } else {
            influences
        };
        let geom_bind = match r.value("primvars:skel:geomBindTransform")? {
            Some(openusd::sdf::Value::Matrix4d(m)) => matrix(&m.0),
            _ => DMat4::IDENTITY,
        };
        Ok((influences, self.to_bind * geom_bind))
    }

    /// The clip of one animation prim, or `None` when it names none of the joints or has no
    /// time samples.
    pub(super) fn clip(
        &self,
        anim: &usd::Prim,
        per_second: f64,
        settings: &ClipSettings,
    ) -> Result<Option<Clip>, String> {
        let path = anim.path().as_str().to_string();
        let names = PrimReader {
            prim: anim,
            time: usd::TimeCode::new(0.0),
        }
        .tokens("joints")?;
        let joints: Vec<Option<u16>> = names.iter().map(|n| self.by_name.get(n).copied()).collect();
        if joints.iter().all(Option::is_none) {
            return Ok(None);
        }
        let attrs = ["translations", "rotations", "scales"];
        let mut times: Vec<f64> = Vec::new();
        for a in attrs {
            times.extend(
                anim.attribute(a)
                    .time_sample_times()
                    .map_err(|e| format!("{path} {a}: {e}"))?,
            );
        }
        times.sort_by(f64::total_cmp);
        times.dedup();
        if times.is_empty() {
            return Ok(None);
        }
        let n = names.len();
        let mut keys: Vec<Vec<Trs>> = Vec::with_capacity(times.len());
        for &t in &times {
            let r = PrimReader {
                prim: anim,
                time: usd::TimeCode::new(t),
            };
            let read = |a: &str, k: usize| -> Result<Option<Vec<f64>>, String> {
                let v = r.numbers(a)?;
                if let Some(v) = &v
                    && v.len() != n * k
                {
                    return Err(format!("{path} {a}: not one value per joint at time {t}"));
                }
                Ok(v)
            };
            let (tr, ro, sc) = (
                read("translations", 3)?,
                read("rotations", 4)?,
                read("scales", 3)?,
            );
            let mut locals = Vec::with_capacity(n);
            for (i, j) in joints.iter().enumerate() {
                let Some(j) = *j else {
                    locals.push(Trs::IDENTITY);
                    continue;
                };
                let rest = &self.info.rest[usize::from(j)];
                let three = |v: &Option<Vec<f64>>, rest: [f32; 3]| {
                    v.as_ref()
                        .map_or(rest, |v| std::array::from_fn(|k| v[3 * i + k] as f32))
                };
                locals.push(Trs {
                    rotation: ro.as_ref().map_or(rest.rotation, |v| {
                        quat(&v[4 * i..]).normalize().as_quat().to_array()
                    }),
                    translation: three(&tr, rest.translation),
                    scale: three(&sc, rest.scale),
                });
            }
            keys.push(locals);
        }
        let seconds: Vec<f32> = times.iter().map(|t| (t / per_second) as f32).collect();
        let mut channels = Vec::new();
        for (i, j) in joints.iter().enumerate() {
            let Some(j) = *j else { continue };
            let joint_keys: Vec<Trs> = keys.iter().map(|k| k[i]).collect();
            let three = |f: fn(&Trs) -> [f32; 3]| {
                joint_keys
                    .iter()
                    .map(|d| {
                        let v = f(d);
                        [v[0], v[1], v[2], 0.0]
                    })
                    .collect()
            };
            for (property, values) in [
                (Property::Translation, three(|d| d.translation)),
                (
                    Property::Rotation,
                    joint_keys.iter().map(|d| d.rotation).collect(),
                ),
                (Property::Scale, three(|d| d.scale)),
            ] {
                let ch = Channel {
                    joint: usize::from(j),
                    property,
                    interpolation: Interpolation::Linear,
                    times: seconds.clone(),
                    values,
                };
                ch.check()
                    .map_err(|e| format!("{path} joint {}: {e}", names[i]))?;
                channels.push(ch);
            }
        }
        Ok(Some(skin::resample(&self.info, &channels, settings)))
    }
}

#[cfg(test)]
mod tests {
    use orr_math::{Mat3, Quat, Vec3};

    use super::*;

    #[test]
    fn pseudo_inverse_meets_the_penrose_conditions() {
        let turn = |a: f32, b: f32| Quat::from_axis_angle(Vec3::new(a, b, 1.0).normalize(), a);
        let flat = |s: Vec3| {
            let product = Mat3::from_quat(turn(0.3, 2.0))
                * Mat3::from_diagonal(s)
                * Mat3::from_quat(turn(1.1, -0.4));
            product.as_dmat3()
        };
        for l in [
            flat(Vec3::new(2.0, 3.0, 0.0)),
            flat(Vec3::new(5.0, 0.0, 0.0)),
            DMat3::from_cols(DVec3::X, DVec3::Y, DVec3::new(1.0, 1.0, 0.0)),
            DMat3::ZERO,
        ] {
            let p = pseudo_inverse(l);
            let near = |a: DMat3, b: DMat3| a.abs_diff_eq(b, 1e-5);
            assert!(near(l * p * l, l), "{l}");
            assert!(near(p * l * p, p), "{l}");
            assert!(near((l * p).transpose(), l * p), "{l}");
            assert!(near((p * l).transpose(), p * l), "{l}");
        }
    }
}
