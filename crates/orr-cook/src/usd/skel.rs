//! UsdSkel: one `Skeleton`, the skin bindings of the meshes bound to it, and
//! `SkelAnimation`s as its clips.
//!
//! Joints keep the skeleton's order, which must list parents first. Model space is the
//! stage's (Y up, meters): root joints take the skeleton prim's world transform, inverse
//! binds are those of the bind transforms in model space, and rest transforms come from
//! `restTransforms` (from the bind transforms without them). Skinned points are
//! `geomBindTransform` times the authored points in the skeleton's model space; the mesh
//! prim's own transform is not used, as UsdSkel specifies. A mesh's `skel:joints` maps its
//! joint indices to skeleton joints by name. An animation that names one of the joints is
//! a clip, named after its prim (by its index among the skeleton's animations when the name
//! is shared): at each time any of its attributes is sampled, the authored joints' local
//! transforms, sampled linearly between those times (rotations by slerp).

use std::collections::HashMap;

use openusd::usd;
use orr_math::{DMat4, DQuat, DVec3};
use orr_pack::clip::Clip;
use orr_pack::skeleton::{MAX_JOINTS, NO_PARENT, SkeletonInfo, Trs};

use super::mesh::Influence;
use super::read::{PrimReader, matrix, quat};
use crate::skin::{self, Channel, ClipSettings, Interpolation, Property};

pub(super) struct Skel {
    pub info: SkeletonInfo,
    /// Joint paths as the skeleton names them.
    joints: Vec<String>,
    by_name: HashMap<String, u16>,
    /// Model space from skeleton space.
    pub to_model: DMat4,
    /// Rest local transforms, as matrices.
    rest: Vec<DMat4>,
}

/// A transform as translation, rotation and scale; shear fails.
fn trs(m: DMat4, what: &dyn Fn() -> String) -> Result<Trs, String> {
    let (s, r, t) = m.to_scale_rotation_translation();
    let back = DMat4::from_scale_rotation_translation(s, r, t);
    if !skin::close(&m.as_mat4(), &back.as_mat4()) {
        return Err(format!(
            "{}: shear; joints need translation, rotation and scale",
            what()
        ));
    }
    Ok(Trs {
        rotation: r.normalize().as_quat().to_array(),
        translation: t.as_vec3().to_array(),
        scale: s.as_vec3().to_array(),
    })
}

impl Skel {
    pub(super) fn build(r: &PrimReader<'_>, to_model: DMat4) -> Result<Self, String> {
        let path = r.path().to_string();
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
            Ok(Some(v.chunks_exact(16).map(matrix).collect()))
        };
        let binds = mats("bindTransforms")?.ok_or_else(|| format!("{path}: no bindTransforms"))?;
        let by_name: HashMap<String, u16> = joints
            .iter()
            .enumerate()
            .map(|(i, j)| (j.clone(), i as u16))
            .collect();
        let mut parents = Vec::with_capacity(joints.len());
        for (i, j) in joints.iter().enumerate() {
            let parent = match j.rsplit_once('/') {
                None => NO_PARENT,
                Some((p, _)) => *by_name
                    .get(p)
                    .ok_or_else(|| format!("{path}: joint {j} has no parent joint {p}"))?,
            };
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
                    p => binds[usize::from(p)].inverse() * binds[i],
                })
                .collect(),
        };
        let mut info = SkeletonInfo {
            parents: parents.clone(),
            rest: Vec::with_capacity(joints.len()),
            inverse_binds: Vec::with_capacity(joints.len()),
            names: joints
                .iter()
                .map(|j| j.rsplit('/').next().unwrap_or(j).to_string())
                .collect(),
        };
        for i in 0..joints.len() {
            let local = match parents[i] {
                NO_PARENT => to_model * rest[i],
                _ => rest[i],
            };
            info.rest
                .push(trs(local, &|| format!("{path} joint {}", joints[i]))?);
            info.inverse_binds
                .push(skin::affine(&(to_model * binds[i]).inverse().as_mat4()));
        }
        Ok(Self {
            info,
            joints,
            by_name,
            to_model,
            rest,
        })
    }

    /// Skin influences per point of a bound mesh, and model space from its points.
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
                        j[k] = indices[e * size + i] as u16;
                        w[k] = weights[e * size + i] as f32;
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
        let geom_bind = r
            .numbers("primvars:skel:geomBindTransform")?
            .map_or(DMat4::IDENTITY, |m| matrix(&m));
        Ok((influences, self.to_model * geom_bind))
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
        let mut keys: Vec<Vec<DMat4>> = Vec::with_capacity(times.len());
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
                    locals.push(DMat4::IDENTITY);
                    continue;
                };
                let (rs, rr, rt) = self.rest[usize::from(j)].to_scale_rotation_translation();
                let t3 = tr.as_ref().map_or(rt, |v| DVec3::from_slice(&v[3 * i..]));
                let q: DQuat = ro.as_ref().map_or(rr, |v| quat(&v[4 * i..]).normalize());
                let s3 = sc.as_ref().map_or(rs, |v| DVec3::from_slice(&v[3 * i..]));
                let local = DMat4::from_scale_rotation_translation(s3, q, t3);
                locals.push(match self.info.parents[usize::from(j)] {
                    NO_PARENT => self.to_model * local,
                    _ => local,
                });
            }
            keys.push(locals);
        }
        let seconds: Vec<f32> = times.iter().map(|t| (t / per_second) as f32).collect();
        let mut channels = Vec::new();
        for (i, j) in joints.iter().enumerate() {
            let Some(j) = *j else { continue };
            let what = || format!("{path} joint {}", names[i]);
            let decomposed: Vec<Trs> = keys
                .iter()
                .map(|k| trs(k[i], &what))
                .collect::<Result<_, _>>()?;
            let three = |f: fn(&Trs) -> [f32; 3]| {
                decomposed
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
                    decomposed.iter().map(|d| d.rotation).collect(),
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
                ch.check().map_err(|e| format!("{}: {e}", what()))?;
                channels.push(ch);
            }
        }
        Ok(Some(skin::resample(&self.info, &channels, settings)))
    }
}
