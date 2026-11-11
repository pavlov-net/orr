//! UsdSkel: one `Skeleton`, the skin bindings of the meshes bound to it, and
//! `SkelAnimation`s as its clips.
//!
//! Joints keep the skeleton's order, which must list parents first; a joint's parent is its
//! nearest ancestor path among them, a joint without one being a root. Model space is the
//! stage's (Y up, meters): the root is the skeleton prim's world transform, inverse binds are
//! those of the bind transforms in model space, and rest transforms come from `restTransforms`
//! (from the bind transforms without them); [`Skel::settle`] then moves them into a space of
//! their own within the rig ([`RigSpace`]). A bind transform with no
//! inverse takes its pseudo-inverse, so the points bound to it land on the surface it flattens
//! to. Skinned points are `geomBindTransform` (unset if it is not a matrix) times
//! the authored points in the skeleton's model space, or its own space in scene units and axes
//! where the skeleton prim's world transform collapses an axis and has no inverse for the binds
//! to undo; the mesh prim's own transform is not used, as UsdSkel specifies. A mesh's
//! `skel:joints` maps its joint indices to skeleton joints by name; that, `geomBindTransform`
//! and the joint influences inherit, as [`SkinSource`] says. An animation that names one of
//! the joints is a clip, named after its prim (by its index among the skeleton's animations
//! when the name is shared): at each time any of its attributes is sampled, the authored
//! joints' local transforms, sampled linearly between those times (rotations by slerp). At a
//! time it does not hold translations, rotations and scales for all its joints it is not valid,
//! and those joints hold their rest transforms; invalid at every time, it is no clip. Every clip
//! runs over the stage's timeline ([`Timeline`]), holding its first pose before its first sample
//! and its last after its last, so the clips of one stage keep time with each other and with
//! its animated pages.
//!
//! Prims whose xforms are sampled over time move what stands under them by the same means, as
//! [`PrimRig`]s: what moves is a [`Rig`](crate::skin::Rig) of prims, and a `Skeleton` under such a
//! prim takes it and every prim between them as joints above its own ([`Skel::hang`]). An xform
//! joint's clip keys come from the prim's transform at its samples and between them, every
//! [`ClipSettings::max_rate`](crate::skin::ClipSettings::max_rate), so an op that turns past half
//! a circle between two samples keeps its turn.

use std::collections::HashMap;

use openusd::usd;
use orr_math::{Affine3, DMat3, DMat4, DVec3, DVec4};
use orr_pack::skeleton::{MAX_SKINNED_JOINTS, NO_PARENT, SkeletonInfo, Trs};

use super::Timeline;
use super::mesh::Influence;
use super::read::{PrimReader, matrix, quat};
use crate::import::{collapses, singular};
use crate::skin::{self, Channel, ClipSource, Interpolation, Property, RigSpace};

/// The attributes of a `SkelAnimation` that pose its joints.
pub(super) const ANIMATED: [&str; 3] = ["translations", "rotations", "scales"];

pub(super) struct Skel {
    pub(super) info: SkeletonInfo,
    /// Joint paths as the skeleton names them.
    joints: Vec<String>,
    by_name: HashMap<String, u16>,
    /// The first of its own joints, after any prims above it.
    first: u16,
    /// The prims above it that are joints, with their joints.
    pub(super) prims: Vec<(usize, u16)>,
    /// The joint of meshes it does not skin, beside those it does.
    pub(super) still: Option<u16>,
    /// Bind space from skeleton space: model space if the skeleton's root has an inverse, else
    /// the skeleton's space in scene units and axes.
    to_bind: DMat4,
    /// Model space from the space of its root joints.
    root: DMat4,
    /// Per joint, the joint's bind space from model space, which [`Self::settle`] puts in
    /// `info`.
    inverse_binds: Vec<DMat4>,
}

/// The prims a mesh's skinning attributes come from: its own, or the nearest ancestor's up to
/// its `SkelRoot` that authors the attribute and applies `SkelBindingAPI`.
pub(super) struct SkinSource<'a> {
    pub(super) joints: Option<PrimReader<'a>>,
    pub(super) indices: PrimReader<'a>,
    pub(super) weights: PrimReader<'a>,
    pub(super) geom_bind: Option<PrimReader<'a>>,
}

/// The path above joint path `joint`.
fn above(joint: &str) -> Option<&str> {
    joint.rsplit_once('/').map(|a| a.0)
}

/// A skeleton of prims.
pub(super) struct PrimRig {
    pub(super) info: SkeletonInfo,
    pub(super) space: RigSpace,
    /// The prim of each joint that has one, with its joint.
    pub(super) prims: Vec<(usize, u16)>,
    pub(super) joint_of: HashMap<usize, u16>,
    /// Per joint, bind space from its space, in model space.
    pub(super) binds: Vec<DMat4>,
    pub(super) still: Option<u16>,
    /// One rigid part per joint, where the model's cages take no parts.
    pub(super) rigid: bool,
}

/// A local transform per key time in `seconds` of `joint`, as linear channels.
pub(super) fn trs_channels(
    joint: usize,
    seconds: &[f32],
    keys: &[Trs],
) -> Result<Vec<Channel>, String> {
    let three = |f: fn(&Trs) -> [f32; 3]| {
        keys.iter()
            .map(|d| {
                let v = f(d);
                [v[0], v[1], v[2], 0.0]
            })
            .collect()
    };
    let mut channels = Vec::with_capacity(4);
    for (property, values) in [
        (Property::Translation, three(|d| d.translation)),
        (
            Property::Rotation,
            keys.iter().map(|d| d.rotation).collect(),
        ),
        (Property::Scale, three(|d| d.scale)),
        (Property::Shear, three(|d| d.shear)),
    ] {
        let ch = Channel {
            joint,
            property,
            interpolation: Interpolation::Linear,
            times: seconds.to_vec(),
            values,
        };
        ch.check()?;
        channels.push(ch);
    }
    Ok(channels)
}

/// `what`'s local transform as a joint's.
pub(super) fn prim_trs(what: &str, m: DMat4) -> Result<Trs, String> {
    skin::decompose(m).ok_or_else(|| format!("{what}: not a finite affine transform"))
}

/// The Moore-Penrose inverse of rank-deficient `l`, from the characteristic polynomial of
/// `l^T l` (Decell, 1965). The second invariant separates ranks 2 and 1 at a ratio of singular
/// values of 1e-5, squared as `l^T l` squares them.
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

/// The inverse of a bind transform; where it has none, the transform that takes a point to the
/// nearest one the bind reaches and back through it.
fn bind_inverse(m: DMat4) -> DMat4 {
    let l = DMat3::from_mat4(m);
    if singular(l) {
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
        if joints.is_empty() || joints.len() > MAX_SKINNED_JOINTS {
            return Err(format!(
                "{path}: {} joints; 1 to {MAX_SKINNED_JOINTS}",
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
            let m: Vec<DMat4> = v
                .as_chunks::<16>()
                .0
                .iter()
                .map(DMat4::from_cols_array)
                .collect();
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
        // `Self::settle` sets the root, the inverse binds and the origin.
        let mut info = SkeletonInfo {
            root: Affine3::IDENTITY,
            origin: DVec3::ZERO,
            parents,
            rest: Vec::with_capacity(joints.len()),
            inverse_binds: Vec::new(),
            names: joints
                .iter()
                .map(|j| j.rsplit('/').next().unwrap_or(j).to_string())
                .collect(),
        };
        let mut inverse_binds = Vec::with_capacity(joints.len());
        for i in 0..joints.len() {
            info.rest
                .push(prim_trs(&format!("{path} joint {}", joints[i]), rest[i])?);
            inverse_binds.push(bind_inverse(binds[i]) * from_bind);
        }
        Ok(Self {
            info,
            joints,
            by_name,
            first: 0,
            prims: Vec::new(),
            still: None,
            to_bind,
            root: to_model,
            inverse_binds,
        })
    }

    /// Adds a joint that holds still in model space, after the others: its bind undoes the root
    /// as it stands, after any [`Self::hang`]. A root that flattens holds still only what lies
    /// on its plane.
    pub(super) fn add_still(&mut self) -> Result<(), String> {
        let joint = self.info.joint_count();
        if joint == MAX_SKINNED_JOINTS {
            return Err(format!("more than {MAX_SKINNED_JOINTS} joints"));
        }
        self.info.parents.push(NO_PARENT);
        self.info.rest.push(Trs::IDENTITY);
        self.inverse_binds.push(bind_inverse(self.root));
        self.info.names.push(String::new());
        self.still = Some(joint as u16);
        Ok(())
    }

    /// Puts `prims`, top first, above the skeleton's joints, the last of them holding its root
    /// joints: each prim's node, name, rest transform and inverse bind. `root` is model space
    /// from the space of the first.
    pub(super) fn hang(
        &mut self,
        prims: Vec<(usize, String, Trs, DMat4)>,
        root: DMat4,
    ) -> Result<(), String> {
        let added = prims.len();
        if self.info.joint_count() + added > MAX_SKINNED_JOINTS {
            return Err(format!(
                "{} joints with the prims moving them; at most {MAX_SKINNED_JOINTS}",
                self.info.joint_count() + added
            ));
        }
        let by = added as u16;
        let shift = |p: u16| match p {
            NO_PARENT => by - 1,
            p => p + by,
        };
        let parents = (0..by).map(|k| k.checked_sub(1).unwrap_or(NO_PARENT));
        let own = self.info.parents.iter().map(|&p| shift(p));
        self.info.parents = parents.chain(own).collect();
        self.by_name.values_mut().for_each(|j| *j += by);
        self.first = by;
        self.root = root;
        let mut rest = Vec::with_capacity(added);
        let mut binds = Vec::with_capacity(added);
        let mut names = Vec::with_capacity(added);
        for (k, (node, name, local, ib)) in prims.into_iter().enumerate() {
            self.prims.push((node, k as u16));
            rest.push(local);
            binds.push(ib);
            names.push(name);
        }
        self.info.rest.splice(0..0, rest);
        self.inverse_binds.splice(0..0, binds);
        self.info.names.splice(0..0, names);
        Ok(())
    }

    /// Settles the skeleton in a space of its own, with `after` applied to model space after its
    /// root, once its joints are all there.
    pub(super) fn settle(&mut self, after: DMat4) -> RigSpace {
        let after = DMat3::from_mat4(after);
        RigSpace::settle(
            &mut self.info,
            self.root,
            after,
            &mut self.inverse_binds,
            self.still,
        )
    }

    /// Skin influences per point of a bound mesh `path`, and bind space from its points.
    pub(super) fn binding(
        &self,
        path: &str,
        source: &SkinSource<'_>,
        points: usize,
    ) -> Result<(Vec<Influence>, DMat4), String> {
        let joints = match &source.joints {
            Some(r) => r.tokens("skel:joints")?,
            None => Vec::new(),
        };
        let map: Vec<u16> = match joints.as_slice() {
            [] => (self.first..self.first + self.joints.len() as u16).collect(),
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
        let r = &source.indices;
        let indices = r
            .numbers("primvars:skel:jointIndices")?
            .ok_or_else(|| format!("{path}: no primvars:skel:jointIndices"))?;
        let weights = source
            .weights
            .numbers("primvars:skel:jointWeights")?
            .ok_or_else(|| format!("{path}: no primvars:skel:jointWeights"))?;
        if indices.len() != weights.len() {
            return Err(format!(
                "{path}: joint indices and weights differ in length"
            ));
        }
        let size = r.element_size("primvars:skel:jointIndices");
        // A primvar unauthored is constant, unless it holds more than one element's influences.
        let constant = match r.metadata_token("primvars:skel:jointIndices", "interpolation") {
            Some(i) => i == "constant",
            None => indices.len() == size,
        };
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
            .map(|e| {
                let (joints, weights) = skin::influences(&influence_sets, &map, e)?;
                Ok::<_, String>((joints.map(|j| j as u8), weights))
            })
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{path}: {e}"))?;
        let influences = if constant {
            vec![influences[0]; points]
        } else {
            influences
        };
        let bind = match &source.geom_bind {
            Some(r) => r.value("primvars:skel:geomBindTransform")?,
            None => None,
        };
        let geom_bind = match bind {
            Some(openusd::sdf::Value::Matrix4d(m)) => matrix(&m.0),
            _ => DMat4::IDENTITY,
        };
        Ok((influences, self.to_bind * geom_bind))
    }

    /// The clip of one animation prim beside the channels `xforms` of the prims above, over the
    /// stage's `timeline`, or `None` when it names none of the joints, has no time samples or is
    /// not valid at any of them. The skeleton is settled in `space`, and `xforms` with it.
    pub(super) fn clip(
        &self,
        anim: &usd::Prim,
        timeline: &Timeline,
        xforms: &[Channel],
        space: &RigSpace,
    ) -> Result<Option<ClipSource>, String> {
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
        let mut times: Vec<f64> = Vec::new();
        for a in ANIMATED {
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
        let mut valid = false;
        for &t in &times {
            let r = PrimReader {
                prim: anim,
                time: usd::TimeCode::new(t),
            };
            let read = |a: &str, k: usize| -> Result<Option<Vec<f64>>, String> {
                Ok(r.numbers(a)?.filter(|v| v.len() == n * k))
            };
            let trs = (
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
                let three = |v: &[f64]| std::array::from_fn(|k| v[3 * i + k] as f32);
                locals.push(match &trs {
                    (Some(tr), Some(ro), Some(sc)) => Trs {
                        rotation: quat(&ro[4 * i..]).normalize().as_quat().to_array(),
                        translation: space.translation(&self.info, usize::from(j), three(tr)),
                        scale: three(sc),
                        shear: [0.0; 3],
                    },
                    _ => self.info.rest[usize::from(j)],
                });
            }
            valid |= matches!(&trs, (Some(_), Some(_), Some(_)));
            keys.push(locals);
        }
        if !valid {
            return Ok(None);
        }
        let seconds: Vec<f32> = times.iter().map(|&t| timeline.seconds(t) as f32).collect();
        let mut channels = xforms.to_vec();
        for (i, j) in joints.iter().enumerate() {
            let Some(j) = *j else { continue };
            let joint_keys: Vec<Trs> = keys.iter().map(|k| k[i]).collect();
            channels.extend(
                trs_channels(usize::from(j), &seconds, &joint_keys)
                    .map_err(|e| format!("{path} joint {}: {e}", names[i]))?,
            );
        }
        Ok(Some(ClipSource {
            channels,
            span: timeline.span(),
        }))
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
