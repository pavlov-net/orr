//! glTF skins and animations: one skeleton per import and its clips; clip resampling from
//! keyed channels, which USD animations share ([`resample`]).
//!
//! The skeleton holds every joint of the skins the scene's meshes use, plus their ancestors
//! up to the scene roots, so model space is the scene's space and an animated ancestor
//! animates the whole skeleton. Joints are ordered parents first (scene order, depth
//! first); mesh joint indices are remapped to that order. Ancestors that no skin names get
//! the inverse of their rest transform as inverse bind. Skins that name the same node must
//! agree on its inverse bind.
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

use glam::{Mat3, Mat4, Quat, Vec3};
use gltf::animation::util::ReadOutputs;
use orr_pack::Affine3;
use orr_pack::AssetId;
use orr_pack::clip::{Clip, ClipHeader};
use orr_pack::skeleton::{MAX_JOINTS, NO_PARENT, SkeletonInfo, Trs, slerp};
use serde::Deserialize;

/// The default scene's node tree: parent and world transform of each glTF node in it.
pub(crate) struct Graph {
    pub parent: Vec<Option<usize>>,
    pub world: Vec<Mat4>,
    pub in_scene: Vec<bool>,
    /// Scene nodes depth first, parents before children, in document order.
    pub order: Vec<usize>,
}

impl Graph {
    pub(crate) fn new(doc: &gltf::Document, scene: &gltf::Scene<'_>) -> Self {
        let n = doc.nodes().len();
        let mut g = Self {
            parent: vec![None; n],
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
            let local = Mat4::from_cols_array_2d(&node.transform().matrix());
            g.in_scene[i] = true;
            g.parent[i] = parent;
            g.world[i] = parent.map_or(Mat4::IDENTITY, |p| g.world[p]) * local;
            g.order.push(i);
            let at = stack.len();
            stack.extend(node.children().map(|c| (c, Some(i))));
            stack[at..].reverse();
        }
        g
    }
}

pub(crate) struct Skeleton {
    pub info: SkeletonInfo,
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
        doc: &gltf::Document,
        buffers: &[Vec<u8>],
        graph: &Graph,
        skins: &[usize],
    ) -> Result<Option<Self>, String> {
        if skins.is_empty() {
            return Ok(None);
        }
        let mut wanted = vec![false; graph.parent.len()];
        let mut bind: HashMap<usize, Mat4> = HashMap::new();
        for &s in skins {
            let skin = doc.skins().nth(s).expect("skin index");
            let mats: Vec<Mat4> = skin
                .reader(|b| buffers.get(b.index()).map(Vec::as_slice))
                .read_inverse_bind_matrices()
                .map_or_else(
                    || vec![Mat4::IDENTITY; skin.joints().len()],
                    |it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
                );
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
                if m.row(3).distance(glam::Vec4::W) > 1e-5 || !m.is_finite() {
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
            parents: Vec::with_capacity(nodes.len()),
            rest: Vec::with_capacity(nodes.len()),
            inverse_binds: Vec::with_capacity(nodes.len()),
            names: Vec::with_capacity(nodes.len()),
        };
        let mut inverse_binds = Vec::with_capacity(nodes.len());
        for &n in &nodes {
            let node = doc.nodes().nth(n).expect("node index");
            info.parents
                .push(graph.parent[n].map_or(NO_PARENT, |p| joint_of[&p]));
            info.rest.push(rest(&node)?);
            let ib = bind
                .get(&n)
                .copied()
                .unwrap_or_else(|| graph.world[n].inverse());
            inverse_binds.push(ib);
            info.inverse_binds.push(affine(&ib));
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

    /// Bind-space position of skeleton joint `j`.
    pub(crate) fn bind_origin(&self, j: u16) -> [f32; 3] {
        self.inverse_binds[usize::from(j)]
            .inverse()
            .w_axis
            .truncate()
            .to_array()
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
                let rest = graph.world[self.nodes[usize::from(j)]];
                let to_bind = self.inverse_binds[usize::from(j)].inverse() * rest.inverse();
                return Ok((j, to_bind * graph.world[node]));
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
        doc: &gltf::Document,
        buffers: &[Vec<u8>],
        settings: &ClipSettings,
    ) -> Result<Vec<(String, Clip)>, String> {
        settings.check()?;
        let mut clips = Vec::new();
        for anim in doc.animations() {
            let what = || format!("animation {}", anim.index());
            let mut channels = Vec::new();
            for ch in anim.channels() {
                let Some(&joint) = self.joint_of.get(&ch.target().node().index()) else {
                    continue;
                };
                let r = ch.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
                let times: Vec<f32> = r
                    .read_inputs()
                    .ok_or_else(|| format!("{}: channel without input", what()))?
                    .collect();
                let values: Vec<[f32; 4]> = match r.read_outputs() {
                    Some(ReadOutputs::Translations(v) | ReadOutputs::Scales(v)) => {
                        v.map(|[x, y, z]| [x, y, z, 0.0]).collect()
                    }
                    Some(ReadOutputs::Rotations(v)) => v.into_f32().collect(),
                    Some(ReadOutputs::MorphTargetWeights(_)) => continue,
                    None => return Err(format!("{}: channel without output", what())),
                };
                use gltf::animation::{Interpolation as I, Property as P};
                let ch = Channel {
                    joint: usize::from(joint),
                    property: match ch.target().property() {
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

/// Largest model-space joint error of `clip` against the channels, in metres and
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
/// rotations are their linear parts with normalised columns.
fn pose_error(a: &Affine3, b: &Affine3) -> (f32, f32) {
    let rot = |m: &Affine3| {
        Mat3::from_cols_array_2d(&[0, 1, 2].map(|c| {
            Vec3::from([0, 1, 2].map(|r| m.rows[r][c]))
                .normalize_or_zero()
                .to_array()
        }))
    };
    let d = rot(a).transpose() * rot(b);
    let cos = ((d.x_axis.x + d.y_axis.y + d.z_axis.z - 1.0) / 2.0).clamp(-1.0, 1.0);
    let t = |m: &Affine3| Vec3::from([0, 1, 2].map(|r| m.rows[r][3]));
    (t(a).distance(t(b)), cos.acos().to_degrees())
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
    pub joint: usize,
    pub property: Property,
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    /// xyz(w); three per key (in-tangent, value, out-tangent) for cubic splines.
    pub values: Vec<[f32; 4]>,
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

    /// Rotations normalised; other values as they are.
    fn fix(&self, v: [f32; 4]) -> [f32; 4] {
        if self.property == Property::Rotation {
            let l = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.map(|x| x / l)
        } else {
            v
        }
    }
}

/// A node's local transform as translation, rotation and scale; a matrix must decompose
/// into them.
fn rest(node: &gltf::Node<'_>) -> Result<Trs, String> {
    let (t, r, s) = node.transform().decomposed();
    if let gltf::scene::Transform::Matrix { matrix } = node.transform() {
        let m = Mat4::from_cols_array_2d(&matrix);
        let back = Mat4::from_scale_rotation_translation(s.into(), Quat::from_array(r), t.into());
        if !close(&m, &back) {
            return Err(format!(
                "node {}: its matrix has shear; skeleton nodes need translation, rotation and scale",
                node.index()
            ));
        }
    }
    let q = Quat::from_array(r).normalize();
    Ok(Trs {
        rotation: q.to_array(),
        translation: t,
        scale: s,
    })
}

pub(crate) fn close(a: &Mat4, b: &Mat4) -> bool {
    let tol = 1e-5 * a.to_cols_array().iter().fold(1.0f32, |m, v| m.max(v.abs()));
    a.to_cols_array()
        .iter()
        .zip(b.to_cols_array())
        .all(|(x, y)| (x - y).abs() <= tol)
}

pub(crate) fn affine(m: &Mat4) -> Affine3 {
    Affine3 {
        rows: std::array::from_fn(|i| m.row(i).to_array()),
    }
}

/// One `JOINTS_n`/`WEIGHTS_n` pair of a primitive.
pub(crate) type InfluenceSet = (Vec<[u16; 4]>, Vec<[f32; 4]>);

/// Up to four influences of a vertex from every `JOINTS_n`/`WEIGHTS_n` pair, mapped to
/// skeleton joints: the largest weights, renormalised and quantised to unorm8 summing to
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
