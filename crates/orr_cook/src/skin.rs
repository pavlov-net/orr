//! glTF skins and animations: skeletons and their clips, one skeleton per model; clip resampling
//! from keyed channels, which USD animations share ([`resample`]).
//!
//! A [`Rig`] is what one skeleton holds: the joints of the skins its meshes use, the nodes that
//! animate what it moves, and every node between those. A node that animates nothing cooked is no
//! joint. Static nodes above them fold into [`SkeletonInfo::root`], which keeps model space the
//! scene's (or the frame's, in a framed model) and spends no joints; rigs that share no joint and
//! no skin are separate rigs. Joints are ordered parents first (scene order, depth first); mesh
//! joint indices are remapped to that order. A joint no skin names gets the inverse of its rest
//! transform as inverse bind, or identity when that collapses an axis. A node that skins bind
//! differently has one joint per inverse bind, neighbors with the same parent and pose. A rigid
//! mesh sits on a joint in bind space, so a bind that stretches would stretch the rounding of its
//! vertices, and one that flattens would lose them: a node with no other bind takes one more
//! joint for it, bound at the node's rest pose.
//!
//! A rig of no skin cooks as rigid parts, one per joint ([`Rig::rigid`]). The renderer poses every
//! part but part 0, so such a skeleton starts with a still joint: unnamed, at identity, holding
//! what a clip must not move. A skinned model's mesh with no joint above it takes the same joint.
//!
//! Clips: every animation with a translation, rotation or scale channel on a skeleton
//! joint. A clip runs from the animation's first key to its last, over every channel it holds,
//! whatever they target, so the clips one animation leaves on several skeletons keep time. It is
//! resampled at the frame rate nearest the setting that fits a whole number of frame intervals,
//! so both ends are frames, and the rate doubles until the clip's sampler holds every joint
//! within [`ClipSettings`]' bounds of the source, measured where the joint's geometry is.
//! Channels follow glTF sampling (`STEP`,
//! `LINEAR` with slerp on the shorter arc, `CUBICSPLINE`; values clamp outside their keys);
//! morph weight channels and channels on other nodes are ignored. A clip is named after its
//! animation, or by the animation's index when the name is empty, all digits or shared with
//! another clip.

use std::collections::HashMap;
use std::ops::Range;

use orr_math::{Affine3, DMat3, DMat4, DQuat, DVec3, DVec4, Mat4, Quat, Vec4};
use orr_pack::AssetId;
use orr_pack::clip::{Clip, ClipHeader};
use orr_pack::skeleton::{MAX_JOINTS, MAX_SKINNED_JOINTS, NO_PARENT, SkeletonInfo, Trs, slerp};
use serde::Deserialize;

use crate::gltf::data::Data;
use crate::import::collapses;
use crate::par::prelude::*;

/// A node's local transform.
fn local(node: &gltf::Node<'_>) -> DMat4 {
    match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => Mat4::from_cols_array_2d(&matrix).as_dmat4(),
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            let q = DVec4::from(rotation.map(f64::from)).try_normalize();
            DMat4::from_scale_rotation_translation(
                scale.map(f64::from).into(),
                q.map_or(DQuat::IDENTITY, DQuat::from_vec4),
                translation.map(f64::from).into(),
            )
        }
    }
}

/// The default scene's node tree: parent and world transform of each glTF node in it, in
/// double precision as glTF asks.
pub(crate) struct Graph {
    pub(crate) parent: Vec<Option<usize>>,
    local: Vec<DMat4>,
    pub(crate) world: Vec<DMat4>,
    pub(crate) in_scene: Vec<bool>,
    /// Scene nodes depth first, parents before children, in document order.
    pub(crate) order: Vec<usize>,
}

impl Graph {
    pub(crate) fn new(doc: &gltf::Document, scene: &gltf::Scene<'_>) -> Self {
        let n = doc.nodes().len();
        let mut g = Self {
            parent: vec![None; n],
            local: vec![DMat4::IDENTITY; n],
            world: vec![DMat4::IDENTITY; n],
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
            g.local[i] = local(&node);
            g.in_scene[i] = true;
            g.parent[i] = parent;
            g.world[i] = parent.map_or(DMat4::IDENTITY, |p| g.world[p]) * g.local[i];
            g.order.push(i);
            let at = stack.len();
            stack.extend(node.children().map(|c| (c, Some(i))));
            stack[at..].reverse();
        }
        g
    }

    /// `node` and the nodes above it, short of `stop`.
    fn chain(&self, node: usize, stop: Option<usize>) -> impl Iterator<Item = usize> {
        chain(&self.parent, node, stop)
    }

    /// Model space from the space of node `n`, in a model framed by `frame`, or in the scene
    /// without one. The local transforms give it inside the frame, wherever a world transform
    /// on the way has no inverse.
    fn in_model(&self, frame: Option<usize>, n: usize) -> Result<DMat4, String> {
        let Some(f) = frame else {
            return Ok(self.world[n]);
        };
        if self.chain(n, None).any(|a| a == f) {
            return Ok(self.relative(f, n));
        }
        Ok(self.outside(f)? * self.world[n])
    }

    /// Model space from the scene, in a model framed by `f`.
    fn outside(&self, f: usize) -> Result<DMat4, String> {
        if collapses(DMat3::from_mat4(self.world[f])) {
            return Err(format!(
                "node {f} frames a model with joints outside it, and collapses an axis"
            ));
        }
        Ok(self.world[f].inverse())
    }

    /// Node `ancestor`'s space from the space of `node`, in its subtree. The local transforms
    /// between them give it even where a world transform on the way has no inverse.
    pub(crate) fn relative(&self, ancestor: usize, node: usize) -> DMat4 {
        let (mut m, mut at) = (DMat4::IDENTITY, node);
        while at != ancestor {
            m = self.local[at] * m;
            at = self.parent[at].expect("an ancestor of the node");
        }
        m
    }
}

/// `node` and the nodes above it by `parent`, short of `stop`.
pub(crate) fn chain(
    parent: &[Option<usize>],
    node: usize,
    stop: Option<usize>,
) -> impl Iterator<Item = usize> {
    let above = |&n: &usize| parent[n];
    std::iter::successors(Some(node), above).take_while(move |&n| Some(n) != stop)
}

/// The nodes an animation's translation, rotation or scale channel targets in the scene.
pub(crate) fn animated(doc: &gltf::Document, graph: &Graph) -> Vec<bool> {
    use gltf::animation::Property as P;
    let mut animated = vec![false; graph.parent.len()];
    for ch in doc
        .animations()
        .flat_map(|a| a.channels().collect::<Vec<_>>())
    {
        let n = ch.target().node().index();
        animated[n] |= graph.in_scene[n] && ch.target().property() != P::MorphTargetWeights;
    }
    animated
}

/// What one skeleton holds.
pub(crate) struct Rig {
    /// Per node, whether it is a joint.
    pub(crate) joints: Vec<bool>,
    /// glTF skins of its meshes.
    skins: Vec<usize>,
    /// The mesh nodes that hang on it.
    pub(crate) meshes: Vec<usize>,
    /// The first joint in scene order, which names a model of its own.
    pub(crate) top: usize,
    /// The node that frames its model; without one, model space is the scene's.
    pub(crate) frame: Option<usize>,
    /// The nearest node above every root joint. Without one they hang from the frame, or from
    /// the scene where one of them lies outside the frame.
    pub(crate) above: Option<usize>,
    pub(crate) outside: bool,
    /// Its meshes cook as rigid parts instead of skinned vertices.
    pub(crate) rigid: bool,
    /// Holds a still joint.
    pub(crate) still: bool,
}

/// How [`Rig::find`] reads the scene.
pub(crate) struct Rigging<'a> {
    /// Per node, the node whose space it stands in.
    pub(crate) parent: &'a [Option<usize>],
    /// Nodes in scene order, parents first.
    pub(crate) order: &'a [usize],
    pub(crate) in_scene: &'a [bool],
    pub(crate) animated: &'a [bool],
    /// The skin of a mesh node, and the nodes it binds.
    pub(crate) skin_of: &'a dyn Fn(usize) -> Option<(usize, Vec<usize>)>,
    /// Skins cook as rigid parts.
    pub(crate) parts_from_skin: bool,
    /// Parts come from a vertex attribute, which leaves joints to skinning.
    pub(crate) parts_from_attribute: bool,
}

impl Rig {
    /// The rigs of mesh nodes `meshes`, which share a model's frame `frame`: each moving mesh in
    /// the one its motion belongs to, and the meshes with no motion in none. With `one`, the
    /// meshes make up a model, and if any moves, a single rig holds them all.
    pub(crate) fn find(
        r: &Rigging<'_>,
        meshes: &[usize],
        frame: Option<usize>,
        one: bool,
    ) -> Vec<Self> {
        let count = r.parent.len();
        let skin_of = r.skin_of;
        let mut sets: Vec<usize> = (0..count).collect();
        fn find(sets: &mut [usize], mut n: usize) -> usize {
            while sets[n] != n {
                sets[n] = sets[sets[n]];
                n = sets[n];
            }
            n
        }
        let mut pinned = vec![false; count];
        for (_, skin_joints) in meshes.iter().filter_map(|&m| skin_of(m)) {
            let mut skin_joints = skin_joints.into_iter();
            let Some(first) = skin_joints.next() else {
                continue;
            };
            pinned[first] = true;
            for j in skin_joints {
                pinned[j] = true;
                let (a, b) = (find(&mut sets, j), find(&mut sets, first));
                sets[a] = b;
            }
        }
        let mut joints = vec![false; count];
        // The movers at or above `from`, short of `stop`, with what is between them; the nearest
        // of them.
        let mut mark = |from: usize, stop: Option<usize>, sets: &mut [usize]| {
            let chain: Vec<usize> = chain(r.parent, from, stop).collect();
            let moves = |n: &usize| r.animated[*n] || pinned[*n];
            let near = chain.iter().position(moves)?;
            let far = chain.iter().rposition(moves)?;
            for &n in &chain[near..=far] {
                joints[n] = true;
                let (a, b) = (find(sets, n), find(sets, chain[near]));
                sets[a] = b;
            }
            Some(chain[near])
        };
        // A skin may name the frame itself, which poses it like any joint.
        for j in (0..count).filter(|&j| pinned[j]) {
            mark(j, frame.filter(|&f| f != j), &mut sets);
        }
        // Per mesh, a joint of the rig that moves it.
        let mut anchors: Vec<(usize, Option<usize>)> = Vec::with_capacity(meshes.len());
        for &m in meshes {
            anchors.push(match skin_of(m) {
                Some((_, joints)) => (m, joints.first().copied()),
                None => (m, mark(m, frame, &mut sets)),
            });
        }
        // Scene order. A joint out of the scene still makes a rig, for `Skeleton::build` to report.
        let out_of_scene = (0..count).filter(|&n| joints[n] && !r.in_scene[n]);
        let in_scene = r.order.iter().copied().filter(|&n| joints[n]);
        let members: Vec<usize> = in_scene.chain(out_of_scene).collect();
        if one && let Some(&first) = members.first() {
            for &n in &members {
                let (a, b) = (find(&mut sets, n), find(&mut sets, first));
                sets[a] = b;
            }
        }
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut group_of: HashMap<usize, usize> = HashMap::new();
        for &n in &members {
            let g = *group_of.entry(find(&mut sets, n)).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(n);
        }
        let mut held: Vec<Vec<usize>> = vec![Vec::new(); groups.len()];
        let mut still = false;
        for &(m, anchor) in &anchors {
            match anchor {
                Some(a) => held[group_of[&find(&mut sets, a)]].push(m),
                None if one && !held.is_empty() => {
                    held[0].push(m);
                    still = true;
                }
                None => {}
            }
        }
        let mut rigs = Vec::with_capacity(groups.len());
        for (own, held) in groups.into_iter().zip(held) {
            let mut skins: Vec<usize> = held.iter().filter_map(|&m| Some(skin_of(m)?.0)).collect();
            skins.sort_unstable();
            skins.dedup();
            let rigid = r.parts_from_skin || (skins.is_empty() && !r.parts_from_attribute);
            let mut rig = Self {
                joints: vec![false; count],
                skins,
                meshes: held,
                top: own[0],
                frame,
                above: None,
                outside: false,
                rigid,
                still: still || rigid,
            };
            own.iter().for_each(|&n| rig.joints[n] = true);
            rig.fold(r.parent, &own, still && !rigid);
            rigs.push(rig);
        }
        rigs
    }

    /// Settles the space root joints hang from among `own`: the nearest node above them all.
    /// The nodes short of it become joints, and with `unfolded`, every node up to the frame or
    /// the scene does, for a still joint a root that does not invert could not undo.
    fn fold(&mut self, parent: &[Option<usize>], own: &[usize], unfolded: bool) {
        let roots: Vec<usize> = own
            .iter()
            .copied()
            .filter(|&n| parent[n].is_none_or(|p| !self.joints[p]))
            .collect();
        let below = |r: usize, f: usize| chain(parent, r, None).skip(1).any(|a| a == f);
        self.outside = self
            .frame
            .is_some_and(|f| !roots.iter().all(|&r| below(r, f)));
        let stop = self.frame.filter(|_| !self.outside);
        let paths: Vec<Vec<usize>> = roots
            .iter()
            .map(|&r| {
                let mut path: Vec<usize> = chain(parent, r, stop).skip(1).collect();
                path.reverse();
                path
            })
            .collect();
        let common = match paths.split_first() {
            _ if unfolded => 0,
            Some((first, rest)) => (0..first.len())
                .take_while(|&i| rest.iter().all(|p| p.get(i) == first.get(i)))
                .count(),
            None => 0,
        };
        self.above = common.checked_sub(1).map(|i| paths[0][i]);
        for &n in paths.iter().flat_map(|p| &p[common..]) {
            self.joints[n] = true;
        }
    }

    /// Model space from the space of the root joints.
    fn root(&self, graph: &Graph) -> Result<DMat4, String> {
        match (self.above, self.frame) {
            (Some(n), frame) => graph.in_model(frame, n),
            (None, Some(f)) if self.outside => graph.outside(f),
            (None, _) => Ok(DMat4::IDENTITY),
        }
    }
}

/// The inverse bind of a joint no skin binds: its rest pose's, or the identity where that
/// collapses an axis.
fn rest_bind(rest: DMat4) -> DMat4 {
    if collapses(DMat3::from_mat4(rest)) {
        DMat4::IDENTITY
    } else {
        rest.inverse()
    }
}

/// Whether `m` turns and scales alike on every axis, which keeps what it takes in shape.
fn holds_shape(m: DMat4) -> bool {
    let l = DMat3::from_mat4(m);
    let g = l.transpose() * l;
    let s = (g.x_axis.x + g.y_axis.y + g.z_axis.z) / 3.0;
    s > 0.0 && (g - DMat3::from_diagonal(DVec3::splat(s))).abs_diff_eq(DMat3::ZERO, 1e-3 * s)
}

pub(crate) struct Skeleton {
    pub(crate) info: SkeletonInfo,
    /// glTF node per joint; a still joint has none.
    nodes: Vec<Option<usize>>,
    /// The first joint of each node.
    joint_of: HashMap<usize, u16>,
    /// Per glTF skin: the skeleton joint of each of its joints.
    skins: HashMap<usize, Vec<u16>>,
    inverse_binds: Vec<DMat4>,
    /// The frame of its model.
    frame: Option<usize>,
    /// The joint of meshes with no joint above them.
    still: Option<u16>,
}

impl Skeleton {
    /// The skeleton of `rig`, with `after` applied to its model space: the mesh keeps its bind
    /// pose, so a linear transform no placement holds goes here.
    pub(crate) fn build(
        data: &Data<'_>,
        graph: &Graph,
        rig: &Rig,
        after: DMat4,
    ) -> Result<Self, String> {
        let doc = &data.doc.doc;
        // Per node its distinct inverse binds, and per skin each joint's node and bind among them.
        let mut binds: HashMap<usize, Vec<Mat4>> = HashMap::new();
        let mut bound: Vec<(usize, Vec<(usize, usize)>)> = Vec::with_capacity(rig.skins.len());
        for &s in &rig.skins {
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
            if mats.len() < skin.joints().len() {
                return Err(format!(
                    "skin {s}: {} inverse bind matrices for {} joints",
                    mats.len(),
                    skin.joints().len()
                ));
            }
            let mut joints = Vec::with_capacity(skin.joints().len());
            for (joint, m) in skin.joints().zip(mats) {
                let i = joint.index();
                if !graph.in_scene[i] {
                    return Err(format!("skin {s}: joint node {i} is not in the scene"));
                }
                if m.row(3).distance(Vec4::W) > 1e-5 || !m.is_finite() {
                    return Err(format!(
                        "skin {s}: inverse bind of node {i} is not an affine transform"
                    ));
                }
                let own = binds.entry(i).or_default();
                let k = own.iter().position(|b| close(b, &m)).unwrap_or_else(|| {
                    own.push(m);
                    own.len() - 1
                });
                joints.push((i, k));
            }
            bound.push((s, joints));
        }
        for &m in &rig.meshes {
            if doc.nodes().nth(m).is_some_and(|d| d.skin().is_some()) {
                continue;
            }
            let joint = graph.chain(m, rig.frame).find(|&n| rig.joints[n]);
            if let Some((n, own)) = joint.and_then(|n| Some((n, binds.get_mut(&n)?)))
                && !own.iter().any(|b| holds_shape(b.as_dmat4()))
            {
                let m = rest_bind(graph.in_model(rig.frame, n)?).as_mat4();
                if !own.iter().any(|b| close(b, &m)) {
                    own.push(m);
                }
            }
        }
        let still = rig.still.then_some(0u16);
        let nodes: Vec<Option<usize>> = still
            .iter()
            .map(|_| None)
            .chain(
                graph
                    .order
                    .iter()
                    .filter(|&&n| rig.joints[n])
                    .flat_map(|&n| std::iter::repeat_n(Some(n), binds.get(&n).map_or(1, Vec::len))),
            )
            .collect();
        let most = if rig.rigid {
            MAX_JOINTS
        } else {
            MAX_SKINNED_JOINTS
        };
        if nodes.len() > most {
            return Err(format!(
                "skeleton of {} joints with the nodes between them; at most {most}",
                nodes.len()
            ));
        }
        let mut joint_of: HashMap<usize, u16> = HashMap::new();
        for (j, n) in nodes.iter().enumerate() {
            if let Some(n) = n {
                joint_of.entry(*n).or_insert(j as u16);
            }
        }
        let root = rig.root(graph)?;
        let mut info = SkeletonInfo {
            root: Affine3::from_mat4((after * root).as_mat4()),
            parents: Vec::with_capacity(nodes.len()),
            rest: Vec::with_capacity(nodes.len()),
            inverse_binds: Vec::with_capacity(nodes.len()),
            names: Vec::with_capacity(nodes.len()),
        };
        let mut inverse_binds = Vec::with_capacity(nodes.len());
        for (j, &n) in nodes.iter().enumerate() {
            let Some(n) = n else {
                // Undoes the root, where it can: a rigid body's part takes no pose.
                let ib = match collapses(DMat3::from_mat4(root)) {
                    true => DMat4::IDENTITY,
                    false => root.inverse(),
                };
                info.parents.push(NO_PARENT);
                info.rest.push(Trs::IDENTITY);
                inverse_binds.push(ib);
                info.inverse_binds.push(Affine3::from_mat4(ib.as_mat4()));
                info.names.push(String::new());
                continue;
            };
            let node = doc.nodes().nth(n).expect("node index");
            let rest_pose = graph.in_model(rig.frame, n)?;
            if !rest_pose.as_mat4().is_finite() {
                return Err(format!("node {n}: its transform is not finite"));
            }
            let parent = graph.parent[n].filter(|&p| rig.joints[p]);
            info.parents
                .push(parent.map_or(NO_PARENT, |p| joint_of[&p]));
            info.rest.push(rest(&node)?);
            // Without an inverse, the identity is a bind that poses what `attach` hangs on the
            // joint: its space is the bind space, which keeps the depth a flat rest pose hides
            // for a pose that unflattens the joint.
            let ib = match binds.get(&n) {
                Some(own) => own[j - usize::from(joint_of[&n])].as_dmat4(),
                None => rest_bind(rest_pose),
            };
            inverse_binds.push(ib);
            info.inverse_binds.push(Affine3::from_mat4(ib.as_mat4()));
            info.names.push(node.name().unwrap_or_default().to_string());
        }
        let skins = bound
            .into_iter()
            .map(|(s, joints)| {
                let joint = |(n, k): (usize, usize)| joint_of[&n] + k as u16;
                (s, joints.into_iter().map(joint).collect())
            })
            .collect();
        Ok(Self {
            info,
            nodes,
            joint_of,
            skins,
            inverse_binds,
            frame: rig.frame,
            still,
        })
    }

    /// The joints of the node whose first joint is `first`.
    fn joints_from(&self, first: usize) -> Range<usize> {
        let node = self.nodes[first];
        let same = self.nodes[first..].iter().take_while(|&&n| n == node);
        first..first + same.count()
    }

    /// Bind space from skeleton joint `j`'s space. A skin may give a joint an inverse bind
    /// that has no inverse: its vertices still skin, but nothing can sit on the joint.
    fn bind(&self, j: u16) -> Result<DMat4, String> {
        let j = usize::from(j);
        let ib = self.inverse_binds[j];
        if collapses(DMat3::from_mat4(ib)) {
            let node = self.nodes[j].expect("a still joint's bind inverts");
            return Err(format!("the inverse bind of node {node} has no inverse"));
        }
        Ok(ib.inverse())
    }

    /// Bind-space position of skeleton joint `j`, the origin if its inverse bind has no inverse.
    pub(crate) fn bind_origin(&self, j: u16) -> [f32; 3] {
        self.bind(j)
            .map_or([0.0; 3], |b| b.w_axis.truncate().as_vec3().to_array())
    }

    /// Skeleton joint of each joint of glTF skin `s`.
    pub(crate) fn skin_joints(&self, s: usize) -> &[u16] {
        &self.skins[&s]
    }

    /// For a mesh node without a skin: the joint it follows rigidly (itself or its nearest
    /// skeleton ancestor in the model, else the still joint) and the transform from its local
    /// space to bind space, so the joint's skinning matrix moves it with the joint. The joints of
    /// one node pose alike, so it takes one whose bind has an inverse, keeping shape if one does.
    pub(crate) fn attach(&self, node: usize, graph: &Graph) -> Result<(u16, DMat4), String> {
        for n in graph.chain(node, self.frame) {
            if let Some(&first) = self.joint_of.get(&n) {
                let joints = self.joints_from(usize::from(first));
                let mut seated = joints.filter(|&j| self.bind(j as u16).is_ok());
                let shaped = |&j: &usize| holds_shape(self.inverse_binds[j]);
                let j = seated.clone().find(shaped).or_else(|| seated.next_back());
                let j = j.map_or(first, |j| j as u16);
                return Ok((j, self.bind(j)? * graph.relative(n, node)));
            }
        }
        let still = self
            .still
            .expect("`Rig::find` saw every mesh without a joint");
        Ok((still, graph.in_model(self.frame, node)?))
    }

    /// Clips of every animation that moves the skeleton, with their names, for [`resample`].
    /// `spans` is [`animation_spans`].
    pub(crate) fn clips(
        &self,
        data: &Data<'_>,
        spans: &[(f64, f64)],
    ) -> Result<Vec<(String, ClipSource)>, String> {
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
                for joint in self.joints_from(ch.joint).skip(1) {
                    channels.push(Channel {
                        joint,
                        ..ch.clone()
                    });
                }
                channels.push(ch);
            }
            if channels.is_empty() {
                continue;
            }
            let clip = ClipSource {
                channels,
                span: spans[anim.index()],
            };
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

/// Per glTF animation the start and length in seconds of its keys, over every channel it holds.
pub(crate) fn animation_spans(data: &Data<'_>) -> Result<Vec<(f64, f64)>, String> {
    let mut spans = Vec::with_capacity(data.doc.doc.animations().len());
    for anim in data.doc.doc.animations() {
        let (mut first, mut last) = (f32::MAX, f32::MIN);
        for ch in anim.channels() {
            let times = data.floats(&ch.sampler().input(), 1);
            let times = times.map_err(|m| format!("animation {}: {m}", anim.index()))?;
            if let (Some(&a), Some(&b)) = (times.first(), times.last()) {
                first = first.min(a);
                last = last.max(b);
            }
        }
        spans.push((f64::from(first), f64::from(last) - f64::from(first)));
    }
    Ok(spans)
}

/// A clip before it is resampled: its channels (checked) and the span, start and length in
/// seconds, it runs over.
pub(crate) struct ClipSource {
    pub(crate) channels: Vec<Channel>,
    pub(crate) span: (f64, f64),
}

/// The clip of `source` resampled at the rate nearest `settings.rate` that fits a whole number of
/// frame intervals, doubling until the clip's sampler holds every joint within the settings'
/// bounds of the source or the rate reaches `max_rate`. `reach` is each joint's
/// ([`crate::import::Scene::reaches`]).
pub(crate) fn resample(
    info: &SkeletonInfo,
    source: &ClipSource,
    reach: &[f64],
    settings: &ClipSettings,
) -> Clip {
    let mut rate = settings.rate;
    loop {
        let clip = sample(info, &source.channels, source.span, rate);
        if rate * 2.0 > settings.max_rate {
            return clip;
        }
        let (position, rotation) = error(info, source, &clip, reach, CHECKS);
        if position <= f64::from(settings.position_error)
            && rotation <= f64::from(settings.rotation_error)
        {
            return clip;
        }
        rate *= 2.0;
    }
}

/// Times [`error`] takes in each interval between consecutive frame and key times, from its
/// start: enough that the largest error it finds is the largest of a far denser search.
const CHECKS: usize = 4;

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
            Property::Shear => p.shear = [v[0], v[1], v[2]],
        }
    }
    pose
}

/// The channels sampled over `span` at the rate nearest `rate` that fits a whole number of frame
/// intervals.
fn sample(info: &SkeletonInfo, channels: &[Channel], span: (f64, f64), rate: f32) -> Clip {
    let (start, duration) = span;
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

/// Largest error of `clip` against the source over every joint, at `per_interval` evenly spaced
/// times in each interval between consecutive frame and key times: in meters, the farthest a
/// point within the joint's `reach` of its origin lands from where the source puts it, and in
/// degrees, the angle between the joint's frames.
fn error(
    info: &SkeletonInfo,
    source: &ClipSource,
    clip: &Clip,
    reach: &[f64],
    per_interval: usize,
) -> (f64, f64) {
    let (start, duration) = source.span;
    let h = &clip.header;
    let inside = |t: &f64| 0.0 < *t && *t < duration;
    let keys = source.channels.iter().flat_map(|c| &c.times);
    let mut ends: Vec<f64> = keys.map(|&t| f64::from(t) - start).filter(inside).collect();
    let frames = (0..h.frame_count).map(|f| f64::from(f) / f64::from(h.sample_rate));
    ends.extend(frames.filter(inside));
    ends.extend([0.0, duration]);
    ends.sort_by(f64::total_cmp);
    ends.dedup();
    let mut times = Vec::with_capacity(ends.len() * per_interval);
    for w in ends.windows(2) {
        let at = |i: usize| w[0] + (w[1] - w[0]) * i as f64 / per_interval as f64;
        times.extend((0..per_interval).map(at));
    }
    times.extend(ends.last());
    let n = info.joint_count();
    let worst = |chunk: &[f64]| {
        let (mut sampled, mut model, mut want) = (
            vec![Trs::IDENTITY; n],
            vec![Affine3::IDENTITY; n],
            vec![Affine3::IDENTITY; n],
        );
        let mut out = (0.0f64, 0.0f64);
        for &t in chunk {
            clip.sample(t as f32, &mut sampled);
            info.model_pose(&sampled, &mut model);
            info.model_pose(&pose(info, &source.channels, start + t), &mut want);
            for ((a, b), &r) in model.iter().zip(&want).zip(reach) {
                let (p, angle) = pose_error(a, b, r);
                out = (out.0.max(p), out.1.max(angle));
            }
        }
        out
    };
    let max = |a: (f64, f64), b: (f64, f64)| (a.0.max(b.0), a.1.max(b.1));
    times.par_chunks(256).map(worst).reduce(|| (0.0, 0.0), max)
}

/// How clips are resampled.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClipSettings {
    /// Frames per second a clip starts from. Each clip takes the nearest rate that fits a
    /// whole number of frame intervals.
    pub rate: f32,
    /// While, at some time, a point that a joint moves lands further than `position_error`
    /// meters from where the source puts it, or the joint's frame turns further than
    /// `rotation_error` degrees from the source's, the rate doubles. The points are those
    /// within the joint's reach, the farthest of what it moves from its origin in its own
    /// space; the distance is in model space and units, without the scale of any placement.
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

/// How far apart two joint transforms put the points within `reach` of the joint's origin, at
/// most (m), and the rotation angle between them (degrees). A point `x` of the joint's space
/// lands `|(a - b) x|` apart, which is at most the distance of the origins and the most the
/// difference of the linear parts stretches a vector of length `reach`: that covers turns,
/// scales and shears alike. The angle is the larger of the angle between the frames of their
/// linear parts, which a collapsed axis still completes, and the angle between any axis and its
/// counterpart, which shows a shear the frames don't.
fn pose_error(a: &Affine3, b: &Affine3, reach: f64) -> (f64, f64) {
    let linear = |m: &Affine3| DMat3::from_cols_array(&m.matrix3.to_cols_array().map(f64::from));
    let origin = |m: &Affine3| DVec3::from_array(m.translation.to_array().map(f64::from));
    let (la, lb) = (linear(a), linear(b));
    let apart = origin(a).distance(origin(b)) + spectral_norm(la - lb) * reach;
    let rot = |l: DMat3| DMat3::from_quat(stretch_rotation(l).1.normalize());
    let d = rot(la).transpose() * rot(lb);
    // Rounding can take the trace of a rotation past 3.
    let cos = ((d.x_axis.x + d.y_axis.y + d.z_axis.z - 1.0) / 2.0).clamp(-1.0, 1.0);
    let lean = [
        (la.x_axis, lb.x_axis),
        (la.y_axis, lb.y_axis),
        (la.z_axis, lb.z_axis),
    ]
    .map(|(u, v)| u.cross(v).length().atan2(u.dot(v)))
    .into_iter()
    .fold(0.0, f64::max);
    (apart, cos.acos().max(lean).to_degrees())
}

/// The largest singular value of `m`, the most it stretches a vector: the root of the largest
/// eigenvalue of `m^T m`, from the closed form of a symmetric 3 x 3 matrix's (Smith, 1961).
pub(crate) fn spectral_norm(m: DMat3) -> f64 {
    let s = m.transpose() * m;
    let diagonal = DVec3::new(s.x_axis.x, s.y_axis.y, s.z_axis.z);
    let q = diagonal.element_sum() / 3.0;
    let off = s.y_axis.x.powi(2) + s.z_axis.x.powi(2) + s.z_axis.y.powi(2);
    let p2 = (diagonal - q).length_squared() + 2.0 * off;
    if p2 == 0.0 {
        return q.sqrt();
    }
    let p = (p2 / 6.0).sqrt();
    let b = (s - DMat3::from_diagonal(DVec3::splat(q))) * p.recip();
    // Rounding can take the half determinant past 1 where eigenvalues meet.
    let r = (b.determinant() / 2.0).clamp(-1.0, 1.0);
    (q + 2.0 * p * (r.acos() / 3.0).cos()).sqrt()
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Property {
    Translation,
    Rotation,
    Scale,
    Shear,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Interpolation {
    Step,
    /// Rotations slerp on the shorter arc.
    Linear,
    CubicSpline,
}

/// One animated property of one joint, keyed in seconds.
#[derive(Clone)]
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

/// `l` as a rotation times an upper triangular stretch (Gram-Schmidt in column order): the
/// rotation, the stretch's diagonal and what lies above it, [`Trs::shear`]. Every `l` has one. A
/// mirror is a negative x scale, and a column with nothing past those before it takes the
/// direction that completes the frame.
fn stretch_rotation(l: DMat3) -> (DVec3, DQuat, DVec3) {
    let cols = [l.x_axis, l.y_axis, l.z_axis];
    let mut frame: [Option<DVec3>; 3] = [None; 3];
    let mut scale = DVec3::ZERO;
    let mut above = [0.0; 3];
    for k in 0..3 {
        let mut rest = cols[k];
        // Twice: what one pass leaves of a nearly dependent column is rounding as much as
        // direction, and the rotation needs the frame orthonormal.
        for _ in 0..2 {
            for i in 0..k {
                if let Some(q) = frame[i] {
                    let d = q.dot(rest);
                    rest -= q * d;
                    above[i + k - 1] += d;
                }
            }
        }
        scale[k] = rest.length();
        if scale[k] > 1e-14 * cols[k].length() {
            frame[k] = Some(rest / scale[k]);
        } else {
            scale[k] = 0.0;
        }
    }
    // In cyclic order, so that a filled axis leaves the frame right-handed: only a mirror may
    // turn x over, or a key that flattens y would spin half a turn from its neighbors.
    while let Some(k) = frame.iter().position(Option::is_none) {
        frame[k] = Some(match (frame[(k + 1) % 3], frame[(k + 2) % 3]) {
            (Some(a), Some(b)) => a.cross(b).normalize(),
            (Some(a), None) | (None, Some(a)) => a.any_orthonormal_vector(),
            (None, None) => DVec3::X,
        });
    }
    let [mut x, y, z] = frame.map(|q| q.expect("every axis is filled"));
    if x.cross(y).dot(z) < 0.0 {
        x = -x;
        scale.x = -scale.x;
        above[0] = -above[0];
        above[1] = -above[1];
    }
    (
        scale,
        DQuat::from_mat3(&DMat3::from_cols(x, y, z)),
        DVec3::from(above),
    )
}

/// `m` as a joint's local transform, `None` unless it is affine and finite in f32.
pub(crate) fn decompose(m: DMat4) -> Option<Trs> {
    let (s, r, h) = stretch_rotation(DMat3::from_mat4(m));
    let (s, h, t) = (s.as_vec3(), h.as_vec3(), m.w_axis.truncate().as_vec3());
    let held = m.is_finite() && s.is_finite() && h.is_finite() && t.is_finite();
    (held && m.row(3).abs_diff_eq(DVec4::W, 1e-5)).then(|| Trs {
        rotation: r.normalize().as_quat().to_array(),
        translation: t.to_array(),
        scale: s.to_array(),
        shear: h.to_array(),
    })
}

/// A node's local transform as a joint's.
fn rest(node: &gltf::Node<'_>) -> Result<Trs, String> {
    match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => {
            decompose(Mat4::from_cols_array_2d(&matrix).as_dmat4()).ok_or_else(|| {
                format!(
                    "node {}: its matrix is not a finite affine transform",
                    node.index()
                )
            })
        }
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            let q = Vec4::from(rotation).try_normalize();
            Ok(Trs {
                rotation: q.map_or(Quat::IDENTITY, Quat::from_vec4).to_array(),
                translation,
                scale,
                shear: [0.0; 3],
            })
        }
    }
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
/// skeleton joints: the largest weights, renormalized and quantized to unorm16 summing to
/// 65535 by largest remainder. Unorm8 and unorm16 sources stay exact, as `257 k` and `k`, and
/// other weights come within a step. Slots without weight repeat the first joint.
pub(crate) fn influences(
    sets: &[InfluenceSet],
    map: &[u16],
    v: usize,
) -> Result<([u16; 4], [u16; 4]), String> {
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
    let kept: f64 = list.iter().map(|&(_, w)| f64::from(w)).sum();
    let scaled: Vec<f64> = list
        .iter()
        .map(|&(_, w)| f64::from(w) / kept * 65535.0)
        .collect();
    let mut q: Vec<u16> = scaled.iter().map(|s| s.floor() as u16).collect();
    let short = 65535 - q.iter().map(|&x| u32::from(x)).sum::<u32>();
    let mut by_frac: Vec<usize> = (0..q.len()).collect();
    by_frac.sort_by(|&a, &b| {
        (scaled[b] - scaled[b].floor()).total_cmp(&(scaled[a] - scaled[a].floor()))
    });
    for &i in by_frac.iter().take(short as usize) {
        q[i] += 1;
    }
    let first = list[0].0;
    let mut joints = [first; 4];
    let mut weights = [0; 4];
    for (s, ((j, _), w)) in list.iter().zip(q).enumerate() {
        if w > 0 {
            joints[s] = *j;
            weights[s] = w;
        }
    }
    Ok((joints, weights))
}

#[cfg(test)]
mod tests {
    use orr_math::Vec3;

    use super::*;

    /// One joint at the origin, keyed by `channels` over `length` seconds.
    fn one_joint(channels: Vec<Channel>, length: f64) -> (SkeletonInfo, ClipSource) {
        let info = SkeletonInfo {
            root: Affine3::IDENTITY,
            parents: vec![NO_PARENT],
            rest: vec![Trs::IDENTITY],
            inverse_binds: vec![Affine3::IDENTITY],
            names: vec!["j".into()],
        };
        (
            info,
            ClipSource {
                channels,
                span: (0.0, length),
            },
        )
    }

    /// A turn about z of `amplitude` degrees, `sin(2 pi t / period)`, keyed `rate` times a
    /// second for `length` seconds and slerped between keys.
    fn swing(rate: f64, amplitude: f64, period: f64, length: f64) -> (SkeletonInfo, ClipSource) {
        let keys = (length * rate).round() as usize;
        let times: Vec<f32> = (0..=keys).map(|k| (k as f64 / rate) as f32).collect();
        let values = times
            .iter()
            .map(|&t| {
                let a =
                    amplitude.to_radians() * (std::f64::consts::TAU * f64::from(t) / period).sin();
                DQuat::from_rotation_z(a).as_quat().to_array()
            })
            .collect();
        let ch = Channel {
            joint: 0,
            property: Property::Rotation,
            interpolation: Interpolation::Linear,
            times,
            values,
        };
        one_joint(vec![ch], length)
    }

    /// A scale from 0 to 1 over `length` seconds, a cubic spline with flat ends.
    fn pop(length: f64) -> (SkeletonInfo, ClipSource) {
        let flat = [0.0; 4];
        let ch = Channel {
            joint: 0,
            property: Property::Scale,
            interpolation: Interpolation::CubicSpline,
            times: vec![0.0, length as f32],
            values: vec![
                flat,
                [0.0, 0.0, 0.0, 0.0],
                flat,
                flat,
                [1.0, 1.0, 1.0, 0.0],
                flat,
            ],
        };
        one_joint(vec![ch], length)
    }

    /// Two joints, the child 1 m out along x: the parent turns about z, keyed at uneven
    /// times, and the child moves along a cubic spline with tangents.
    fn chain() -> (SkeletonInfo, ClipSource) {
        let times = vec![0.0, 0.3, 0.45, 1.1, 1.5];
        let turn = |a: f64| DQuat::from_rotation_z(a.to_radians()).as_quat().to_array();
        let parent = Channel {
            joint: 0,
            property: Property::Rotation,
            interpolation: Interpolation::Linear,
            values: [0.0, 40.0, -10.0, 70.0, 0.0].map(turn).to_vec(),
            times: times.clone(),
        };
        let point = |x: f32, y: f32| [x, y, 0.0, 0.0];
        let child = Channel {
            joint: 1,
            property: Property::Translation,
            interpolation: Interpolation::CubicSpline,
            values: [
                [point(0.0, 0.0), point(1.0, 0.0), point(0.0, 2.0)],
                [point(-1.0, 0.0), point(1.0, 0.5), point(3.0, -1.0)],
                [point(0.0, 4.0), point(1.5, 0.0), point(0.0, -4.0)],
                [point(2.0, 0.0), point(1.0, -0.5), point(-2.0, 1.0)],
                [point(0.0, 0.0), point(1.0, 0.0), point(0.0, 0.0)],
            ]
            .concat(),
            times,
        };
        let (mut info, mut source) = one_joint(vec![parent, child], 1.5);
        info.parents.push(0);
        info.rest.push(Trs {
            translation: [1.0, 0.0, 0.0],
            ..Trs::IDENTITY
        });
        info.inverse_binds.push(Affine3::from_translation(-Vec3::X));
        info.names.push("child".into());
        source.span = (0.0, 1.5);
        (info, source)
    }

    /// The largest error [`error`] finds, against that of a search 128 times as dense, over the
    /// cases that show the most between the frames and keys it starts from: slow and fast swings,
    /// scale pops, and a chain of uneven keys and cubic splines.
    #[test]
    fn checks_find_the_largest_error_of_a_dense_search() {
        let cases = [
            (swing(24.0, 45.0, 10.0, 10.0), [20.0, 0.0]),
            (swing(24.0, 90.0, 4.0, 8.0), [20.0, 0.0]),
            (swing(24.0, 90.0, 2.0, 4.0), [20.0, 0.0]),
            (swing(24.0, 30.0, 1.0, 4.0), [20.0, 0.0]),
            (swing(25.0, 90.0, 3.0, 6.0), [20.0, 0.0]),
            (pop(0.25), [1.0, 0.0]),
            (pop(0.5), [1.0, 0.0]),
            (pop(1.0), [1.0, 0.0]),
            (chain(), [0.0, 3.0]),
        ];
        for ((info, source), reach) in &cases {
            for rate in [30.0, 60.0, 120.0] {
                let clip = sample(info, &source.channels, source.span, rate);
                let reach = &reach[..info.joint_count()];
                let dense = error(info, source, &clip, reach, 128 * CHECKS);
                let got = error(info, source, &clip, reach, CHECKS);
                // Short of the rounding of f32 poses, which has no shape to search.
                let rounding = 1e-6 * reach.iter().fold(1.0, |a, &b| f64::max(a, b));
                let found = got.0 >= 0.999 * dense.0 - rounding && got.1 >= 0.999 * dense.1 - 1e-4;
                assert!(found, "{rate} fps: {got:?} of {dense:?}");
            }
        }
    }

    #[test]
    fn spectral_norm_is_the_largest_singular_value() {
        let turn = |a: f64, b: f64| {
            DMat3::from_quat(DQuat::from_axis_angle(DVec3::new(a, b, 1.0).normalize(), a))
        };
        for (s, want) in [
            (DVec3::new(3.0, 1.0, 0.5), 3.0),
            (DVec3::new(0.25, -2.0, 2.0), 2.0),
            (DVec3::new(1e-9, 0.0, 0.0), 1e-9),
            (DVec3::ZERO, 0.0),
        ] {
            let m = turn(0.3, 2.0) * DMat3::from_diagonal(s) * turn(1.1, -0.4);
            let got = spectral_norm(m);
            // The closed form keeps half its digits where singular values meet.
            assert!((got - want).abs() <= 1e-7 * want, "{s}: {got}");
        }
    }
}
