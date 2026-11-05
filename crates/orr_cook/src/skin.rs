//! glTF skins and animations: one skeleton per import and its clips; clip resampling from
//! keyed channels, which USD animations share ([`resample`]).
//!
//! The skeleton holds every joint of the skins the scene's meshes use, plus their ancestors
//! up to the scene roots, so model space is the scene's space and an animated ancestor
//! animates the whole skeleton. Joints are ordered parents first (scene order, depth
//! first); mesh joint indices are remapped to that order. Ancestors that no skin names get
//! the inverse of their rest transform as inverse bind, or identity when it collapses an
//! axis. Skins that name the same node must agree on its inverse bind.
//!
//! Clips: every animation with a translation, rotation or scale channel on a skeleton
//! joint. A clip runs from its first key to its last over every channel; it is resampled
//! at the frame rate nearest the setting that fits a whole number of frame intervals, so
//! both ends are frames, and the rate doubles until the clip's sampler holds every joint
//! within [`ClipSettings`]' bounds of the source. Channels follow glTF sampling (`STEP`,
//! `LINEAR` with slerp on the shorter arc, `CUBICSPLINE`; values clamp outside their keys);
//! morph weight channels and channels on other nodes are ignored. A clip is named after its
//! animation, or by the animation's index when the name is empty, all digits or shared with
//! another clip.

use std::collections::HashMap;

use orr_math::{Affine3, Mat3, Mat4, Quat, Vec3};
use orr_pack::AssetId;
use orr_pack::clip::{Clip, ClipHeader};
use orr_pack::skeleton::{MAX_JOINTS, NO_PARENT, SkeletonInfo, Trs, slerp};
use serde::Deserialize;

use crate::gltf::data::Data;
use crate::import::collapses;

/// The default scene's node tree: parent and world transform of each glTF node in it.
pub(crate) struct Graph {
    pub(crate) parent: Vec<Option<usize>>,
    local: Vec<Mat4>,
    pub(crate) world: Vec<Mat4>,
    in_scene: Vec<bool>,
    /// Scene nodes depth first, parents before children, in document order.
    pub(crate) order: Vec<usize>,
}

impl Graph {
    pub(crate) fn new(doc: &gltf::Document, scene: &gltf::Scene<'_>) -> Self {
        let n = doc.nodes().len();
        let mut g = Self {
            parent: vec![None; n],
            local: vec![Mat4::IDENTITY; n],
            world: vec![Mat4::IDENTITY; n],
            in_scene: vec![false; n],
            order: Vec::new(),
        };
        let mut stack: Vec<_> = scene.nodes().map(|n| (n, None)).collect();
        stack.reverse();
        while let Some((node, parent)) = stack.pop() {
            let i = node.index();
            if g.in_scene[i] {
                continue;
            }
            g.local[i] = Mat4::from_cols_array_2d(&node.transform().matrix());
            g.in_scene[i] = true;
            g.parent[i] = parent;
            g.world[i] = parent.map_or(Mat4::IDENTITY, |p| g.world[p]) * g.local[i];
            g.order.push(i);
            let at = stack.len();
            stack.extend(node.children().map(|c| (c, Some(i))));
            stack[at..].reverse();
        }
        g
    }

    /// Node `ancestor`'s space from the space of `node`, in its subtree. The local transforms
    /// between them give it even where a world transform on the way has no inverse.
    pub(crate) fn relative(&self, ancestor: usize, node: usize) -> Mat4 {
        let (mut m, mut at) = (Mat4::IDENTITY, node);
        while at != ancestor {
            m = self.local[at] * m;
            at = self.parent[at].expect("an ancestor of the node");
        }
        m
    }
}

pub(crate) struct Skeleton {
    pub(crate) info: SkeletonInfo,
    /// glTF node per joint.
    nodes: Vec<usize>,
    joint_of: HashMap<usize, u16>,
    /// Per glTF skin: the skeleton joint of each of its joints.
    skins: HashMap<usize, Vec<u16>>,
    inverse_binds: Vec<Mat4>,
}

impl Skeleton {
    /// The skeleton of `skins` (glTF skin indices), or `None` without skins.
    pub(crate) fn build(
        data: &Data<'_>,
        graph: &Graph,
        skins: &[usize],
    ) -> Result<Option<Self>, String> {
        if skins.is_empty() {
            return Ok(None);
        }
        let doc = &data.doc.doc;
        let mut wanted = vec![false; graph.parent.len()];
        let mut bind: HashMap<usize, Mat4> = HashMap::new();
        for &s in skins {
            let skin = doc.skins().nth(s).expect("skin index");
            let mats: Vec<Mat4> = match skin.inverse_bind_matrices() {
                Some(a) => data
                    .floats(&a, 16)
                    .map_err(|m| format!("skin {s}: {m}"))?
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .map(Mat4::from_cols_array)
                    .collect(),
                None => vec![Mat4::IDENTITY; skin.joints().len()],
            };
            if mats.len() != skin.joints().len() {
                return Err(format!(
                    "skin {s}: {} inverse bind matrices for {} joints",
                    mats.len(),
                    skin.joints().len()
                ));
            }
            for (joint, m) in skin.joints().zip(mats) {
                let i = joint.index();
                if !graph.in_scene[i] {
                    return Err(format!("skin {s}: joint node {i} is not in the scene"));
                }
                if m.row(3).distance(orr_math::Vec4::W) > 1e-5 || !m.is_finite() {
                    return Err(format!(
                        "skin {s}: inverse bind of node {i} is not an affine transform"
                    ));
                }
                if let Some(prev) = bind.insert(i, m)
                    && !close(&prev, &m)
                {
                    return Err(format!(
                        "skins disagree on the inverse bind of node {i}; one skeleton needs one"
                    ));
                }
                let mut a = Some(i);
                while let Some(n) = a.filter(|&n| !wanted[n]) {
                    wanted[n] = true;
                    a = graph.parent[n];
                }
            }
        }
        let nodes: Vec<usize> = graph.order.iter().copied().filter(|&n| wanted[n]).collect();
        if nodes.len() > MAX_JOINTS {
            return Err(format!(
                "skeleton of {} joints with ancestors; at most {MAX_JOINTS}",
                nodes.len()
            ));
        }
        let joint_of: HashMap<usize, u16> = nodes
            .iter()
            .enumerate()
            .map(|(j, &n)| (n, j as u16))
            .collect();
        let mut info = SkeletonInfo {
            root: Affine3::IDENTITY,
            parents: Vec::with_capacity(nodes.len()),
            rest: Vec::with_capacity(nodes.len()),
            inverse_binds: Vec::with_capacity(nodes.len()),
            names: Vec::with_capacity(nodes.len()),
        };
        let mut inverse_binds = Vec::with_capacity(nodes.len());
        for &n in &nodes {
            let node = doc.nodes().nth(n).expect("node index");
            let world = graph.world[n];
            if !world.is_finite() {
                return Err(format!("node {n}: its transform is not finite"));
            }
            info.parents
                .push(graph.parent[n].map_or(NO_PARENT, |p| joint_of[&p]));
            info.rest.push(rest(&node)?);
            // Without an inverse, the identity is a bind that poses what `attach` hangs on the
            // joint: its space is the bind space, which keeps the depth a flat rest pose hides
            // for a pose that unflattens the joint.
            let ib = bind.get(&n).copied().unwrap_or_else(|| {
                if collapses(Mat3::from_mat4(world).as_dmat3()) {
                    Mat4::IDENTITY
                } else {
                    world.inverse()
                }
            });
            inverse_binds.push(ib);
            info.inverse_binds.push(Affine3::from_mat4(ib));
            info.names.push(node.name().unwrap_or_default().to_string());
        }
        let skins = skins
            .iter()
            .map(|&s| {
                let skin = doc.skins().nth(s).expect("skin index");
                (s, skin.joints().map(|j| joint_of[&j.index()]).collect())
            })
            .collect();
        Ok(Some(Self {
            info,
            nodes,
            joint_of,
            skins,
            inverse_binds,
        }))
    }

    /// Bind space from skeleton joint `j`'s space. A skin may give a joint an inverse bind
    /// that has no inverse: its vertices still skin, but nothing can sit on the joint.
    fn bind(&self, j: u16) -> Result<Mat4, String> {
        let j = usize::from(j);
        let ib = self.inverse_binds[j];
        if collapses(Mat3::from_mat4(ib).as_dmat3()) {
            return Err(format!(
                "the inverse bind of node {} has no inverse",
                self.nodes[j]
            ));
        }
        Ok(ib.inverse())
    }

    /// Bind-space position of skeleton joint `j`, the origin if its inverse bind has no inverse.
    pub(crate) fn bind_origin(&self, j: u16) -> [f32; 3] {
        self.bind(j)
            .map_or([0.0; 3], |b| b.w_axis.truncate().to_array())
    }

    /// Skeleton joint of each joint of glTF skin `s`.
    pub(crate) fn skin_joints(&self, s: usize) -> &[u16] {
        &self.skins[&s]
    }

    /// For a mesh node without a skin: the joint it follows rigidly (itself or its nearest
    /// skeleton ancestor) and the transform from its local space to bind space, so the
    /// joint's skinning matrix moves it with the joint.
    pub(crate) fn attach(&self, node: usize, graph: &Graph) -> Result<(u16, Mat4), String> {
        let mut a = Some(node);
        while let Some(n) = a {
            if let Some(&j) = self.joint_of.get(&n) {
                return Ok((j, self.bind(j)? * graph.relative(n, node)));
            }
            a = graph.parent[n];
        }
        Err(format!(
            "node {node} has a mesh without a skin and no skeleton joint above it"
        ))
    }

    /// Clips of every animation that moves the skeleton, with their names.
    pub(crate) fn clips(
        &self,
        data: &Data<'_>,
        settings: &ClipSettings,
    ) -> Result<Vec<(String, Clip)>, String> {
        settings.check()?;
        let mut clips = Vec::new();
        for anim in data.doc.doc.animations() {
            let what = || format!("animation {}", anim.index());
            let mut channels = Vec::new();
            for ch in anim.channels() {
                let Some(&joint) = self.joint_of.get(&ch.target().node().index()) else {
                    continue;
                };
                use gltf::animation::{Interpolation as I, Property as P};
                let property = ch.target().property();
                if property == P::MorphTargetWeights {
                    continue;
                }
                let fail = |m: String| format!("{}: {m}", what());
                let times = data.floats(&ch.sampler().input(), 1).map_err(fail)?;
                let values: Vec<[f32; 4]> = data
                    .floats(&ch.sampler().output(), 4)
                    .map_err(fail)?
                    .as_chunks::<4>()
                    .0
                    .to_vec();
                let ch = Channel {
                    joint: usize::from(joint),
                    property: match property {
                        P::Translation => Property::Translation,
                        P::Rotation => Property::Rotation,
                        P::Scale => Property::Scale,
                        P::MorphTargetWeights => unreachable!("skipped"),
                    },
                    interpolation: match ch.sampler().interpolation() {
                        I::Step => Interpolation::Step,
                        I::Linear => Interpolation::Linear,
                        I::CubicSpline => Interpolation::CubicSpline,
                    },
                    times,
                    values,
                };
                ch.check().map_err(|m| format!("{}: {m}", what()))?;
                channels.push(ch);
            }
            if channels.is_empty() {
                continue;
            }
            let clip = resample(&self.info, &channels, settings);
            clips.push((anim.index(), anim.name().unwrap_or_default(), clip));
        }
        let mut count: HashMap<&str, usize> = HashMap::new();
        for (_, n, _) in &clips {
            *count.entry(n).or_default() += 1;
        }
        Ok(clips
            .into_iter()
            .map(|(i, n, c)| {
                let plain =
                    !n.is_empty() && !n.bytes().all(|b| b.is_ascii_digit()) && count[n] == 1;
                (if plain { n.to_string() } else { i.to_string() }, c)
            })
            .collect())
    }
}

/// The clip of `channels` (checked), resampled at the rate nearest `settings.rate` that fits
/// a whole number of frame intervals, doubling until the clip's sampler holds every joint
/// within the settings' bounds or the rate reaches `max_rate`.
pub(crate) fn resample(info: &SkeletonInfo, channels: &[Channel], settings: &ClipSettings) -> Clip {
    let mut rate = settings.rate;
    loop {
        let clip = sample(info, channels, rate);
        let (position, rotation) = error(info, channels, &clip);
        if (position <= settings.position_error && rotation <= settings.rotation_error)
            || rate * 2.0 > settings.max_rate
        {
            return clip;
        }
        rate *= 2.0;
    }
}

/// Local transforms at time `t`: the rest pose with the channels applied.
fn pose(info: &SkeletonInfo, channels: &[Channel], t: f64) -> Vec<Trs> {
    let mut pose = info.rest.clone();
    for c in channels {
        let v = c.eval(t);
        let p = &mut pose[c.joint];
        match c.property {
            Property::Translation => p.translation = [v[0], v[1], v[2]],
            Property::Rotation => p.rotation = v,
            Property::Scale => p.scale = [v[0], v[1], v[2]],
        }
    }
    pose
}

/// The channels sampled from their first key to their last at the rate nearest `rate`
/// that fits a whole number of frame intervals.
fn sample(info: &SkeletonInfo, channels: &[Channel], rate: f32) -> Clip {
    let (start, duration) = span(channels);
    let intervals = if duration > 0.0 {
        (duration * f64::from(rate)).round().max(1.0) as u32
    } else {
        0
    };
    let sample_rate = if intervals > 0 {
        (f64::from(intervals) / duration) as f32
    } else {
        rate
    };
    let n = info.joint_count();
    let mut samples: Vec<Trs> = Vec::with_capacity((intervals as usize + 1) * n);
    for f in 0..=intervals {
        let t = start + duration * f64::from(f) / f64::from(intervals.max(1));
        let mut pose = pose(info, channels, t);
        if f > 0 {
            for (p, q) in pose.iter_mut().zip(&samples[samples.len() - n..]) {
                let d: f32 = (0..4).map(|i| p.rotation[i] * q.rotation[i]).sum();
                if d < 0.0 {
                    p.rotation = p.rotation.map(|x| -x);
                }
            }
        }
        samples.extend(pose);
    }
    Clip {
        header: ClipHeader {
            skeleton: AssetId::NONE,
            joint_count: n as u32,
            frame_count: intervals + 1,
            sample_rate,
            duration: intervals as f32 / sample_rate,
        },
        samples,
    }
}

/// Largest model-space joint error of `clip` against the channels, in meters and
/// degrees, over every key time and the middle of every frame interval.
fn error(info: &SkeletonInfo, channels: &[Channel], clip: &Clip) -> (f32, f32) {
    let (start, duration) = span(channels);
    let h = &clip.header;
    let mut times: Vec<f64> = channels
        .iter()
        .flat_map(|c| c.times.iter().map(|&t| f64::from(t) - start))
        .collect();
    times.extend((0..h.frame_count - 1).map(|f| (f64::from(f) + 0.5) / f64::from(h.sample_rate)));
    let n = info.joint_count();
    let (mut sampled, mut model, mut want) = (
        vec![Trs::IDENTITY; n],
        vec![Affine3::IDENTITY; n],
        vec![Affine3::IDENTITY; n],
    );
    let (mut position, mut rotation) = (0.0f32, 0.0f32);
    for t in times {
        let t = t.clamp(0.0, duration);
        clip.sample(t as f32, &mut sampled);
        info.model_pose(&sampled, &mut model);
        info.model_pose(&pose(info, channels, start + t), &mut want);
        for (a, b) in model.iter().zip(&want) {
            let (p, r) = pose_error(a, b);
            position = position.max(p);
            rotation = rotation.max(r);
        }
    }
    (position, rotation)
}

/// How clips are resampled.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClipSettings {
    /// Frames per second a clip starts from. Each clip takes the nearest rate that fits a
    /// whole number of frame intervals.
    pub rate: f32,
    /// While some joint's model-space pose is further than these (m, degrees) from the glTF
    /// sampler, at a key time or between two frames, the rate doubles.
    pub position_error: f32,
    pub rotation_error: f32,
    /// The rate stops doubling here; a clip that still misses the bounds (a `STEP` channel
    /// that jumps, for one) keeps the last rate.
    pub max_rate: f32,
}

impl Default for ClipSettings {
    fn default() -> Self {
        Self {
            rate: 30.0,
            position_error: 0.0005,
            rotation_error: 0.1,
            max_rate: 240.0,
        }
    }
}

impl ClipSettings {
    pub(crate) fn check(&self) -> Result<(), String> {
        let positive = |v: f32| v.is_finite() && v > 0.0;
        if !positive(self.rate) || !(self.max_rate.is_finite() && self.max_rate >= self.rate) {
            return Err(format!(
                "clip rates {} and {} are not positive and ascending",
                self.rate, self.max_rate
            ));
        }
        if !(self.position_error >= 0.0 && self.rotation_error >= 0.0) {
            return Err("clip error bounds are negative".into());
        }
        Ok(())
    }
}

/// First key time and the time to the last, over every channel.
fn span(channels: &[Channel]) -> (f64, f64) {
    let start = channels.iter().map(|c| c.times[0]).fold(f32::MAX, f32::min);
    let end = channels
        .iter()
        .map(|c| c.times[c.times.len() - 1])
        .fold(f32::MIN, f32::max);
    (f64::from(start), f64::from(end) - f64::from(start))
}

/// Translation distance (m) and rotation angle (degrees) between two transforms; the
/// rotations are the frames of their linear parts, which a collapsed axis still completes.
fn pose_error(a: &Affine3, b: &Affine3) -> (f32, f32) {
    let rot = |m: &Affine3| Mat3::from_quat(scale_rotation(m.matrix3).1.normalize());
    let d = rot(a).transpose() * rot(b);
    let cos = ((d.x_axis.x + d.y_axis.y + d.z_axis.z - 1.0) / 2.0).clamp(-1.0, 1.0);
    (
        a.translation.distance(b.translation),
        cos.acos().to_degrees(),
    )
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Property {
    Translation,
    Rotation,
    Scale,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Interpolation {
    Step,
    /// Rotations slerp on the shorter arc.
    Linear,
    CubicSpline,
}

/// One animated property of one joint, keyed in seconds.
pub(crate) struct Channel {
    /// Skeleton joint.
    pub(crate) joint: usize,
    pub(crate) property: Property,
    pub(crate) interpolation: Interpolation,
    pub(crate) times: Vec<f32>,
    /// xyz(w); three per key (in-tangent, value, out-tangent) for cubic splines.
    pub(crate) values: Vec<[f32; 4]>,
}

impl Channel {
    pub(crate) fn check(&self) -> Result<(), String> {
        let n = self.times.len();
        let per = if self.interpolation == Interpolation::CubicSpline {
            3
        } else {
            1
        };
        if n == 0 || self.values.len() != n * per {
            return Err(format!("{} keys and {} values", n, self.values.len()));
        }
        if !self.times.iter().all(|t| t.is_finite()) || !self.times.is_sorted() {
            return Err("key times are not finite and ascending".into());
        }
        if !self.values.iter().flatten().all(|v| v.is_finite()) {
            return Err("non-finite key value".into());
        }
        Ok(())
    }

    fn key(&self, k: usize) -> [f32; 4] {
        match self.interpolation {
            Interpolation::CubicSpline => self.values[3 * k + 1],
            _ => self.values[k],
        }
    }

    fn eval(&self, t: f64) -> [f32; 4] {
        let n = self.times.len();
        if t <= f64::from(self.times[0]) {
            return self.fix(self.key(0));
        }
        if t >= f64::from(self.times[n - 1]) {
            return self.fix(self.key(n - 1));
        }
        let k = self.times.partition_point(|&x| f64::from(x) <= t) - 1;
        let (t0, t1) = (f64::from(self.times[k]), f64::from(self.times[k + 1]));
        let td = t1 - t0;
        let u = ((t - t0) / td) as f32;
        let rotation = self.property == Property::Rotation;
        match self.interpolation {
            Interpolation::Step => self.fix(self.key(k)),
            Interpolation::Linear if rotation => slerp(self.key(k), self.key(k + 1), u),
            Interpolation::Linear => {
                let (a, b) = (self.key(k), self.key(k + 1));
                std::array::from_fn(|i| a[i] + (b[i] - a[i]) * u)
            }
            Interpolation::CubicSpline => {
                let td = td as f32;
                let (u2, u3) = (u * u, u * u * u);
                let (v0, b0) = (self.values[3 * k + 1], self.values[3 * k + 2]);
                let (a1, v1) = (self.values[3 * k + 3], self.values[3 * k + 4]);
                let p: [f32; 4] = std::array::from_fn(|i| {
                    (2.0 * u3 - 3.0 * u2 + 1.0) * v0[i]
                        + td * (u3 - 2.0 * u2 + u) * b0[i]
                        + (-2.0 * u3 + 3.0 * u2) * v1[i]
                        + td * (u3 - u2) * a1[i]
                });
                self.fix(p)
            }
        }
    }

    /// Rotations normalized; other values as they are.
    fn fix(&self, v: [f32; 4]) -> [f32; 4] {
        if self.property == Property::Rotation {
            let l = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.map(|x| x / l)
        } else {
            v
        }
    }
}

/// The per-axis scale and rotation of `l`, if it has them. A mirror is a negative x scale, and
/// a column of zeros takes the direction that completes the frame.
fn scale_rotation(l: Mat3) -> (Vec3, Quat) {
    let cols = [l.x_axis, l.y_axis, l.z_axis];
    let mut scale = Vec3::from(cols.map(Vec3::length));
    let mut frame = cols.map(Vec3::normalize_or_zero);
    for k in 0..3 {
        if frame[k] == Vec3::ZERO {
            let (a, b) = (frame[(k + 1) % 3], frame[(k + 2) % 3]);
            frame[k] = match [a, b].into_iter().find(|v| *v != Vec3::ZERO) {
                Some(seed) => a
                    .cross(b)
                    .try_normalize()
                    .unwrap_or(seed.any_orthonormal_vector()),
                None => Vec3::X,
            };
        }
    }
    let [mut x, y, z] = frame;
    if x.cross(y).dot(z) < 0.0 {
        scale.x = -scale.x;
        x = -x;
    }
    (scale, Quat::from_mat3(&Mat3::from_cols(x, y, z)))
}

/// The scale, rotation and translation of `m`, `None` if it shears.
pub(crate) fn decompose(m: Mat4) -> Option<(Vec3, Quat, Vec3)> {
    let (s, r) = scale_rotation(Mat3::from_mat4(m));
    let t = m.w_axis.truncate();
    close(&m, &Mat4::from_scale_rotation_translation(s, r, t)).then_some((s, r, t))
}

/// A node's local transform as translation, rotation and scale; a matrix must decompose
/// into them.
fn rest(node: &gltf::Node<'_>) -> Result<Trs, String> {
    let (t, r, s) = match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => {
            let (s, r, t) = decompose(Mat4::from_cols_array_2d(&matrix)).ok_or_else(|| {
                format!(
                    "node {}: its matrix is not translation, rotation and scale, as skeleton nodes need",
                    node.index()
                )
            })?;
            (t.to_array(), r.to_array(), s.to_array())
        }
        transform => transform.decomposed(),
    };
    let q = Quat::from_array(r).normalize();
    Ok(Trs {
        rotation: q.to_array(),
        translation: t,
        scale: s,
    })
}

fn close(a: &Mat4, b: &Mat4) -> bool {
    let tol = 1e-5 * a.to_cols_array().iter().fold(1.0f32, |m, v| m.max(v.abs()));
    a.to_cols_array()
        .iter()
        .zip(b.to_cols_array())
        .all(|(x, y)| (x - y).abs() <= tol)
}

/// One `JOINTS_n`/`WEIGHTS_n` pair of a primitive.
pub(crate) type InfluenceSet = (Vec<[u16; 4]>, Vec<[f32; 4]>);

/// Up to four influences of a vertex from every `JOINTS_n`/`WEIGHTS_n` pair, mapped to
/// skeleton joints: the largest weights, renormalized and quantized to unorm8 summing to
/// 255. Slots without weight repeat the first joint.
pub(crate) fn influences(
    sets: &[InfluenceSet],
    map: &[u16],
    v: usize,
) -> Result<([u8; 4], [u8; 4]), String> {
    let mut list: Vec<(u16, f32)> = Vec::with_capacity(4 * sets.len());
    for (joints, weights) in sets {
        for (&j, &w) in joints[v].iter().zip(&weights[v]) {
            if !(w.is_finite() && w >= 0.0) {
                return Err(format!("vertex {v}: weight {w}"));
            }
            if w == 0.0 {
                continue;
            }
            let j = *map
                .get(usize::from(j))
                .ok_or_else(|| format!("vertex {v}: joint {j} past the skin's {}", map.len()))?;
            match list.iter_mut().find(|(k, _)| *k == j) {
                Some((_, acc)) => *acc += w,
                None => list.push((j, w)),
            }
        }
    }
    let sum: f32 = list.iter().map(|(_, w)| w).sum();
    if (sum - 1.0).abs() > 0.01 {
        return Err(format!("vertex {v}: weights sum to {sum}"));
    }
    list.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    list.truncate(4);
    let kept: f32 = list.iter().map(|(_, w)| w).sum();
    let scaled: Vec<f32> = list.iter().map(|(_, w)| w / kept * 255.0).collect();
    let mut q: Vec<u8> = scaled.iter().map(|s| s.floor() as u8).collect();
    let short = 255 - q.iter().map(|&x| u32::from(x)).sum::<u32>();
    let mut by_frac: Vec<usize> = (0..q.len()).collect();
    by_frac.sort_by(|&a, &b| {
        (scaled[b] - scaled[b].floor()).total_cmp(&(scaled[a] - scaled[a].floor()))
    });
    for &i in by_frac.iter().take(short as usize) {
        q[i] += 1;
    }
    let first = list[0].0 as u8;
    let mut joints = [first; 4];
    let mut weights = [0; 4];
    for (s, ((j, _), w)) in list.iter().zip(q).enumerate() {
        if w > 0 {
            joints[s] = *j as u8;
            weights[s] = w;
        }
    }
    Ok((joints, weights))
}
