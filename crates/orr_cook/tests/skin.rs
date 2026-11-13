#![allow(clippy::print_stdout)] // the external-set test reports measured errors

//! Skeletons, clips and CPU skinning against an independent glTF reference: a sampler that
//! evaluates the source channels at 1 kHz, and linear blend skinning of the source vertices
//! with the source weights.
//!
//! `skinned_set_matches_gltf` cooks an external set and is ignored unless asked for:
//! `ORR_SKINNED_SET=<dir>` holds `.glb` files with skins and animations.
//!
//! `cargo test --release -p orr_cook --test skin -- --ignored --nocapture`

#[path = "common/skinned.rs"]
mod common_skinned;
#[allow(dead_code)]
#[path = "common/glb.rs"]
mod glb;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glb::Glb;
use gltf::animation::util::ReadOutputs;
use gltf::animation::{Interpolation, Property};
use orr_cook::{
    ArchiveSpec, AssetSpec, ClipSettings, GltfSettings, LodMode, Manifest, MeshSettings, Source,
};
use orr_math::{Affine3, Mat3, Mat4, Quat, Vec3, Vec4};
use orr_pack::clip::Clip;
use orr_pack::mesh::{self, Cluster, MeshHeader, MeshInfo, PageView};
use orr_pack::model::ModelInfo;
use orr_pack::skeleton::{NO_PARENT, SkeletonInfo, Trs};
use orr_pack::{Archive, AssetId, Class, Target};

const POSITION_MM: f64 = 0.5;
const ROTATION_DEG: f64 = 0.1;
const SKIN_MM: f64 = 0.5;

/// The source document and its buffers.
struct Doc {
    doc: gltf::Document,
    buffers: Vec<Vec<u8>>,
}

fn load(path: &Path) -> Doc {
    let bytes = std::fs::read(path).unwrap();
    let g = gltf::Gltf::from_slice(&bytes).unwrap();
    let buffers = g
        .buffers()
        .map(|b| match b.source() {
            gltf::buffer::Source::Bin => g.blob.clone().unwrap(),
            gltf::buffer::Source::Uri(uri) => {
                std::fs::read(path.parent().unwrap().join(uri)).unwrap()
            }
        })
        .collect();
    Doc {
        doc: g.document,
        buffers,
    }
}

impl Doc {
    fn buf<'s>(&'s self) -> impl Clone + Fn(gltf::Buffer<'_>) -> Option<&'s [u8]> + 's {
        move |b| self.buffers.get(b.index()).map(Vec::as_slice)
    }

    fn parents(&self) -> Vec<Option<usize>> {
        let mut p = vec![None; self.doc.nodes().len()];
        for n in self.doc.nodes() {
            for c in n.children() {
                p[c.index()] = Some(n.index());
            }
        }
        p
    }

    /// World transform of every node with `local` per node.
    fn world(&self, local: &[(Vec3, Quat, Vec3)]) -> Vec<Mat4> {
        let parents = self.parents();
        let mut world: Vec<Option<Mat4>> = vec![None; local.len()];
        fn at(
            i: usize,
            parents: &[Option<usize>],
            local: &[(Vec3, Quat, Vec3)],
            world: &mut [Option<Mat4>],
        ) -> Mat4 {
            if let Some(m) = world[i] {
                return m;
            }
            let (t, r, s) = local[i];
            let l = Mat4::from_scale_rotation_translation(s, r, t);
            let m = match parents[i] {
                Some(p) => at(p, parents, local, world) * l,
                None => l,
            };
            world[i] = Some(m);
            m
        }
        Vec::from_fn(local.len(), |i| at(i, &parents, local, &mut world))
    }

    fn rest(&self) -> Vec<(Vec3, Quat, Vec3)> {
        self.doc
            .nodes()
            .map(|n| {
                let (t, r, s) = n.transform().decomposed();
                (t.into(), Quat::from_array(r), s.into())
            })
            .collect()
    }

    /// The transform channels of an animation.
    fn channels(&self, a: &gltf::Animation<'_>) -> Vec<Channel> {
        let mut out = Vec::new();
        for ch in a.channels() {
            let r = ch.reader(self.buf());
            let values: Vec<Vec4> = match r.read_outputs().unwrap() {
                ReadOutputs::Translations(v) | ReadOutputs::Scales(v) => {
                    v.map(|x| Vec3::from(x).extend(0.0)).collect()
                }
                ReadOutputs::Rotations(v) => v.into_f32().map(Vec4::from).collect(),
                ReadOutputs::MorphTargetWeights(_) => continue,
            };
            out.push(Channel {
                node: ch.target().node().index(),
                property: ch.target().property(),
                interpolation: ch.sampler().interpolation(),
                times: r.read_inputs().unwrap().collect(),
                values,
            });
        }
        out
    }

    /// Local transforms of every node at glTF time `t`.
    fn pose(&self, channels: &[Channel], t: f32) -> Vec<(Vec3, Quat, Vec3)> {
        let mut local = self.rest();
        for c in channels {
            let rot = c.property == Property::Rotation;
            let v = sample(&c.times, &c.values, c.interpolation, rot, t);
            let l = &mut local[c.node];
            match c.property {
                Property::Translation => l.0 = v.truncate(),
                Property::Rotation => l.1 = Quat::from_vec4(v),
                Property::Scale => l.2 = v.truncate(),
                Property::MorphTargetWeights => {}
            }
        }
        local
    }
}

struct Channel {
    node: usize,
    property: Property,
    interpolation: Interpolation,
    times: Vec<f32>,
    values: Vec<Vec4>,
}

fn time_range(channels: &[Channel]) -> (f32, f32) {
    channels.iter().fold((f32::MAX, f32::MIN), |r, c| {
        (r.0.min(c.times[0]), r.1.max(*c.times.last().unwrap()))
    })
}

/// The glTF sampler: clamped ends, STEP, LINEAR (slerp for rotations), CUBICSPLINE.
fn sample(times: &[f32], values: &[Vec4], interp: Interpolation, rot: bool, t: f32) -> Vec4 {
    let cubic = interp == Interpolation::CubicSpline;
    let key = |k: usize| if cubic { values[3 * k + 1] } else { values[k] };
    let norm = |v: Vec4| if rot { v.normalize() } else { v };
    let n = times.len();
    if t <= times[0] {
        return norm(key(0));
    }
    if t >= times[n - 1] {
        return norm(key(n - 1));
    }
    let k = times.iter().rposition(|&x| x <= t).unwrap();
    let td = times[k + 1] - times[k];
    let u = (t - times[k]) / td;
    match interp {
        Interpolation::Step => norm(key(k)),
        Interpolation::Linear if rot => {
            let (a, b) = (Quat::from_vec4(key(k)), Quat::from_vec4(key(k + 1)));
            Vec4::from(a.slerp(b, u))
        }
        Interpolation::Linear => key(k).lerp(key(k + 1), u),
        Interpolation::CubicSpline => {
            let (u2, u3) = (u * u, u * u * u);
            norm(
                (2.0 * u3 - 3.0 * u2 + 1.0) * values[3 * k + 1]
                    + td * (u3 - 2.0 * u2 + u) * values[3 * k + 2]
                    + (-2.0 * u3 + 3.0 * u2) * values[3 * k + 4]
                    + td * (u3 - u2) * values[3 * k + 3],
            )
        }
    }
}

/// Rotation angle in degrees between two transforms.
fn rotation_error(a: &Mat4, b: &Mat4) -> f64 {
    let rot = |m: &Mat4| {
        let m3 = Mat3::from_mat4(*m);
        Mat3::from_cols(
            m3.x_axis.normalize(),
            m3.y_axis.normalize(),
            m3.z_axis.normalize(),
        )
    };
    let d = rot(a).transpose() * rot(b);
    let c = ((d.x_axis.x + d.y_axis.y + d.z_axis.z - 1.0) / 2.0).clamp(-1.0, 1.0);
    f64::from(c.acos()).to_degrees()
}

/// The cooked entries of one glTF source.
struct Cooked {
    archive: Archive,
    name: String,
}

impl Cooked {
    fn blob(&self, e: &orr_pack::Entry, i: usize) -> Vec<u8> {
        let b = &self.archive.blobs(e)[i];
        let mut v = vec![0; b.raw_len as usize];
        self.archive.read(b, &mut v).unwrap();
        v
    }

    fn entry(&self, suffix: &str) -> &orr_pack::Entry {
        let id = AssetId::from_name(&format!("{}{suffix}", self.name));
        self.archive
            .find(id)
            .unwrap_or_else(|| panic!("no entry {}{suffix}", self.name))
    }

    fn skeleton(&self) -> SkeletonInfo {
        let e = self.entry("/skeleton");
        assert_eq!(e.class, Class::Skeleton as u16);
        SkeletonInfo::parse(&self.blob(e, 0)).unwrap()
    }

    fn clips(&self) -> Vec<(String, Clip)> {
        let prefix = format!("{}/clip/", self.name);
        self.archive
            .entries()
            .iter()
            .filter(|e| e.class == Class::Clip as u16)
            .map(|e| {
                let name = self
                    .archive
                    .name(e)
                    .strip_prefix(&prefix)
                    .unwrap()
                    .to_string();
                (
                    name,
                    Clip::parse(&self.blob(e, 0), &self.blob(e, 1)).unwrap(),
                )
            })
            .collect()
    }
}

fn cook(path: &Path, name: &str, settings: GltfSettings) -> Cooked {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: name.into(),
                source: Source::Gltf {
                    path: path.into(),
                    settings,
                },
            }],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let archive = Archive::from_memory(bytes).unwrap();
    for e in archive.entries() {
        archive.verify(e).unwrap();
    }
    Cooked {
        archive,
        name: name.into(),
    }
}

/// Worst errors over one source.
#[derive(Default, Debug)]
struct Errors {
    /// Points the skeleton moves, posed by the cooked clips against the glTF sampler at 4 kHz:
    /// the joints' origins, and every source vertex in the space of each joint that moves it.
    /// Over the clips short of [`ClipSettings::max_rate`], which hold the bound.
    position_mm: f64,
    /// Joint frames, at the same times.
    rotation_deg: f64,
    /// `mesh::skin` against reference skinning of the cooked vertex (position, joints and
    /// weights as decoded) with the glTF sampler's joint matrices.
    skin_mm: f64,
    /// `mesh::skin` against reference skinning of the source vertex: adds the grid and the
    /// unorm16 weights.
    source_mm: f64,
    /// Cooked weights against the source, in unorm16 steps.
    weight_steps: f64,
    clips: usize,
    rates: (f32, f32),
    clip_bytes: u64,
}

/// glTF node of each skeleton joint, by name; the unnamed still joint has none.
fn joint_nodes(src: &Doc, skel: &SkeletonInfo) -> Vec<Option<usize>> {
    skel.names
        .iter()
        .map(|n| {
            if n.is_empty() {
                return None;
            }
            let found: Vec<usize> = src
                .doc
                .nodes()
                .filter(|x| x.name().unwrap_or_default() == n)
                .map(|x| x.index())
                .collect();
            assert_eq!(found.len(), 1, "joint name {n:?} is not unique");
            Some(found[0])
        })
        .collect()
}

/// A skinned source vertex: bind-space position and influences as skeleton joints.
struct Skinned {
    position: Vec3,
    influences: Vec<(usize, f32)>,
}

/// Skinned vertices, and the vertices of meshes without a skin as (node, local position).
fn source_vertices(src: &Doc, nodes: &[Option<usize>]) -> (Vec<Skinned>, Vec<(usize, Vec3)>) {
    let joint_of = joint_of(nodes);
    let (mut skinned, mut rigid) = (Vec::new(), Vec::new());
    for node in src.doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        for prim in mesh.primitives() {
            let r = prim.reader(src.buf());
            let pos: Vec<Vec3> = r.read_positions().unwrap().map(Vec3::from).collect();
            let Some(skin) = node.skin() else {
                rigid.extend(pos.into_iter().map(|p| (node.index(), p)));
                continue;
            };
            let map: Vec<usize> = skin.joints().map(|j| joint_of[&j.index()]).collect();
            let mut infl = vec![Vec::new(); pos.len()];
            let mut set = 0;
            while let (Some(j), Some(w)) = (r.read_joints(set), r.read_weights(set)) {
                for (v, (j, w)) in j.into_u16().zip(w.into_f32()).enumerate() {
                    for k in 0..4 {
                        if w[k] > 0.0 {
                            infl[v].push((map[usize::from(j[k])], w[k]));
                        }
                    }
                }
                set += 1;
            }
            skinned.extend(
                pos.into_iter()
                    .zip(infl)
                    .map(|(position, influences)| Skinned {
                        position,
                        influences,
                    }),
            );
        }
    }
    (skinned, rigid)
}

/// Skeleton joint of each glTF node that is one.
fn joint_of(nodes: &[Option<usize>]) -> HashMap<usize, usize> {
    let joints = nodes.iter().enumerate();
    joints.filter_map(|(j, n)| Some(((*n)?, j))).collect()
}

/// What the skeleton moves, as (joint, point in its space): every joint's origin, each skinned
/// vertex in the space of every joint that weighs on it (through the source's inverse binds
/// `ibm`), and each vertex of a mesh without a skin in the space of its nearest joint.
fn moved_points(
    src: &Doc,
    nodes: &[Option<usize>],
    ibm: &[Mat4],
    vertices: &(Vec<Skinned>, Vec<(usize, Vec3)>),
) -> Vec<(usize, Vec3)> {
    let mut out: Vec<(usize, Vec3)> = Vec::from_fn(nodes.len(), |j| (j, Vec3::ZERO));
    for v in &vertices.0 {
        let inf = v.influences.iter().filter(|i| i.1 > 0.0);
        out.extend(inf.map(|&(j, _)| (j, ibm[j].transform_point3(v.position))));
    }
    let joint_of = joint_of(nodes);
    let parents = src.parents();
    let world = src.world(&src.rest());
    for &(node, p) in &vertices.1 {
        let mut above = std::iter::successors(Some(node), |&n| parents[n]);
        let held = above.find(|n| joint_of.contains_key(n)).unwrap();
        let q = world[held].inverse() * world[node];
        out.push((joint_of[&held], q.transform_point3(p)));
    }
    out
}

/// The farthest the clip's poses put a point of `points` from where the source's put it, in mm,
/// and the largest angle between a joint's frames, in degrees, at 4 kHz.
fn clip_error(
    src: &Doc,
    channels: &[Channel],
    start: f32,
    clip: &Clip,
    (skel, nodes): (&SkeletonInfo, &[Option<usize>]),
    points: &[(usize, Vec3)],
) -> (f64, f64) {
    let n = skel.joint_count();
    let mut locals = vec![Trs::IDENTITY; n];
    let mut pose = vec![Affine3::IDENTITY; n];
    let (mut far, mut angle) = (0.0f64, 0.0f64);
    let end = clip.header.duration;
    for i in 0..=(end * 4000.0).ceil() as u32 {
        let t = (i as f32 / 4000.0).min(end);
        clip.sample(t, &mut locals);
        skel.model_pose(&locals, &mut pose);
        let world = src.world(&src.pose(channels, start + t));
        let model = |j: usize| to_model(skel) * Mat4::from(pose[j]);
        for &(j, q) in points {
            let Some(node) = nodes[j] else { continue };
            let apart = model(j).transform_point3(q) - world[node].transform_point3(q);
            far = far.max(f64::from(apart.length()) * 1000.0);
        }
        for (j, node) in nodes.iter().enumerate() {
            if let Some(node) = *node {
                angle = angle.max(rotation_error(&model(j), &world[node]));
            }
        }
    }
    (far, angle)
}

/// Model space from the space of `skel`, which its palettes and its mesh stand in.
fn to_model(skel: &SkeletonInfo) -> Mat4 {
    Mat4::from_translation(skel.origin.as_vec3())
}

/// The inverse bind of each joint of `skel` from model space.
fn model_binds(skel: &SkeletonInfo) -> Vec<Mat4> {
    let from_model = to_model(skel).inverse();
    let ib = skel.inverse_binds.iter();
    ib.map(|&a| Mat4::from(a) * from_model).collect()
}

/// Whether `clip` stopped doubling its rate at [`ClipSettings::max_rate`].
fn at_max_rate(clip: &Clip) -> bool {
    clip.header.sample_rate > 0.75 * ClipSettings::default().max_rate
}

fn lbs(reference: &[Mat4], influences: impl Iterator<Item = (usize, f32)>, p: Vec3) -> Vec3 {
    influences
        .map(|(j, w)| w * reference[j].transform_point3(p))
        .sum()
}

fn check(path: &Path, name: &str, settings: GltfSettings) -> (Cooked, Errors) {
    let src = load(path);
    let cooked = cook(path, name, settings);
    let skel = cooked.skeleton();
    let nodes = joint_nodes(&src, &skel);
    let n = skel.joint_count();
    let model = ModelInfo::parse(&cooked.blob(cooked.entry(""), 0)).unwrap();
    assert_eq!(
        model.header.skeleton,
        AssetId::from_name(&format!("{name}/skeleton"))
    );
    let mesh_e = cooked.entry("/mesh");
    let info = MeshInfo::parse(&cooked.blob(mesh_e, 0)).unwrap();
    assert_ne!(info.header.attributes & MeshHeader::SKIN, 0);

    // Inverse binds: the source skins', the cooked ones for joints no skin names.
    let mut ibm = model_binds(&skel);
    for skin in src.doc.skins() {
        if let Some(m) = skin.reader(src.buf()).read_inverse_bind_matrices() {
            for (joint, m) in skin.joints().zip(m) {
                let j = nodes
                    .iter()
                    .position(|&n| n == Some(joint.index()))
                    .unwrap();
                let m = Mat4::from_cols_array_2d(&m);
                let tol = 1e-5 * m.to_cols_array().iter().fold(1.0f32, |a, b| a.max(b.abs()));
                assert!(ibm[j].abs_diff_eq(m, tol), "joint {j}: inverse bind");
                ibm[j] = m;
            }
        }
    }

    // Source vertices by grid cell, to pair them with cooked vertices.
    let vertices = source_vertices(&src, &nodes);
    let points = moved_points(&src, &nodes, &ibm, &vertices);
    let (skinned, rigid) = vertices;
    let step = f64::from(info.header.step());
    let origin = skel.origin;
    let cell = |p: Vec3| {
        let p = p.as_dvec3() - origin;
        p.to_array().map(|x| (x / step + 0.5).floor() as i32)
    };
    let mut by_cell: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (i, v) in skinned.iter().enumerate() {
        by_cell.entry(cell(v.position)).or_default().push(i);
    }
    let pages: Vec<Vec<u8>> = info
        .pages
        .iter()
        .map(|p| {
            let stored = cooked.blob(mesh_e, p.blob as usize);
            let mut out = vec![0; p.resident_len as usize];
            mesh::decode_page(&stored, &mut out).unwrap();
            out
        })
        .collect();
    let views: Vec<PageView<'_>> = pages
        .iter()
        .map(|p| PageView::new(p, &info.header).unwrap())
        .collect();
    let candidates = |view: &PageView<'_>, c: &Cluster, v: u32| {
        by_cell.get(&view.grid(c, v)).map_or_default(Vec::as_slice)
    };

    let mut err = Errors {
        rates: (f32::MAX, 0.0),
        ..Errors::default()
    };
    for view in &views {
        for c in (0..view.cluster_count()).map(|i| view.cluster(i)) {
            for v in 0..u32::from(c.vertex_count) {
                let joints = view.joints(&c, v).unwrap();
                let weights = view.weights(&c, v).unwrap();
                let steps = candidates(view, &c, v)
                    .iter()
                    .map(|&s| {
                        (0..n)
                            .map(|j| {
                                let cooked: f32 = (0..4)
                                    .filter(|&k| usize::from(joints[k]) == j)
                                    .map(|k| weights[k])
                                    .sum();
                                let source: f32 = skinned[s]
                                    .influences
                                    .iter()
                                    .filter(|i| i.0 == j)
                                    .map(|i| i.1)
                                    .sum();
                                f64::from((cooked - source).abs()) * 65535.0
                            })
                            .fold(0.0, f64::max)
                    })
                    .fold(f64::MAX, f64::min);
                if steps < f64::MAX {
                    err.weight_steps = err.weight_steps.max(steps);
                }
            }
        }
    }

    let anims: Vec<gltf::Animation<'_>> = src.doc.animations().collect();
    let mut locals = vec![Trs::IDENTITY; n];
    let mut pose = vec![Affine3::IDENTITY; n];
    let mut palette = vec![Affine3::IDENTITY; n];
    for (clip_name, clip) in cooked.clips() {
        let anim = match clip_name.parse::<usize>() {
            Ok(i) => &anims[i],
            Err(_) => anims
                .iter()
                .find(|a| a.name() == Some(clip_name.as_str()))
                .unwrap(),
        };
        let channels = src.channels(anim);
        let (start, end) = time_range(&channels);
        assert!(
            (clip.header.duration - (end - start)).abs() < 1e-4,
            "{clip_name} duration"
        );
        err.clips += 1;
        err.rates = (
            err.rates.0.min(clip.header.sample_rate),
            err.rates.1.max(clip.header.sample_rate),
        );
        err.clip_bytes += (clip.samples.len() * size_of::<Trs>()) as u64;
        let (p, r) = clip_error(&src, &channels, start, &clip, (&skel, &nodes), &points);
        if !at_max_rate(&clip) {
            err.position_mm = err.position_mm.max(p);
        }
        err.rotation_deg = err.rotation_deg.max(r);
        for i in 0..10 {
            let t = clip.header.duration * i as f32 / 9.0;
            clip.sample(t, &mut locals);
            skel.model_pose(&locals, &mut pose);
            skel.palette(&pose, &mut palette);
            let world = src.world(&src.pose(&channels, start + t));
            // A still joint's palette entry is the identity.
            let reference: Vec<Mat4> = Vec::from_fn(n, |j| {
                nodes[j].map_or(Mat4::IDENTITY, |node| world[node] * ibm[j])
            });
            for view in &views {
                for c in (0..view.cluster_count()).map(|i| view.cluster(i)) {
                    let mut out = vec![[0.0; 3]; usize::from(c.vertex_count)];
                    mesh::skin(view, &c, &palette, &mut out);
                    let placed = to_model(&skel);
                    let out = out
                        .into_iter()
                        .map(|o| placed.transform_point3(Vec3::from(o)));
                    for (v, got) in (0..).zip(out) {
                        let joints = view.joints(&c, v).unwrap();
                        let weights = view.weights(&c, v).unwrap();
                        let decoded = placed.transform_point3(Vec3::from(view.position(&c, v)));
                        let infl = (0..4).map(|k| (usize::from(joints[k]), weights[k]));
                        let want = lbs(&reference, infl, decoded);
                        err.skin_mm = err.skin_mm.max(f64::from(want.distance(got)) * 1000.0);
                        let skinned_best = candidates(view, &c, v).iter().map(|&s| {
                            let sv = &skinned[s];
                            lbs(&reference, sv.influences.iter().copied(), sv.position)
                        });
                        let rigid_best = rigid
                            .iter()
                            .map(|&(node, p)| world[node].transform_point3(p));
                        let best = skinned_best
                            .chain(rigid_best)
                            .map(|w| f64::from(w.distance(got)) * 1000.0)
                            .fold(f64::MAX, f64::min);
                        err.source_mm = err.source_mm.max(best);
                    }
                }
            }
        }
    }
    (cooked, err)
}

#[test]
fn skinned_fixture_matches_gltf() {
    let path = common_skinned::skinned_glb("skinned", true);
    let (cooked, e) = check(&path, "rig", GltfSettings::default());
    let skel = cooked.skeleton();
    assert_eq!(skel.names, ["root", "hip", "knee", "foot"]);
    assert_eq!(skel.parents, [NO_PARENT, 0, 1, 2]);
    let mut clips: Vec<String> = cooked.clips().into_iter().map(|c| c.0).collect();
    clips.sort();
    assert_eq!(clips, ["1", "bend"]);
    assert!(
        e.position_mm <= POSITION_MM && e.rotation_deg <= ROTATION_DEG,
        "{e:?}"
    );
    assert!(
        e.skin_mm <= SKIN_MM && e.weight_steps <= 1.0 && e.source_mm < 1.0,
        "{e:?}"
    );
    // A DAG keeps source vertices with their joints and weights at every depth.
    let dag = GltfSettings {
        mesh: MeshSettings {
            lod: LodMode::Dag,
            ..Default::default()
        },
        ..Default::default()
    };
    let (cooked, e) = check(&path, "rig", dag);
    let info = MeshInfo::parse(&cooked.blob(cooked.entry("/mesh"), 0)).unwrap();
    assert!(!info.groups.is_empty());
    assert!(e.skin_mm <= SKIN_MM && e.weight_steps <= 1.0, "{e:?}");
}

#[test]
#[ignore = "needs ORR_SKINNED_SET"]
fn skinned_set_matches_gltf() {
    let dir = PathBuf::from(std::env::var("ORR_SKINNED_SET").expect("ORR_SKINNED_SET"));
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "glb"))
        .collect();
    files.sort();
    let mut failed = Vec::new();
    for f in &files {
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        let joints: usize = load(f).doc.skins().map(|s| s.joints().len()).sum();
        let (_, e) = check(f, &name, GltfSettings::default());
        println!(
            "{name}: {joints} skin joints, {} clips at {:.1}-{:.1} Hz, {} KiB of samples; \
             pose {:.4} mm {:.4} deg; skin {:.4} mm; against the source vertex {:.4} mm, \
             weights {:.2} steps",
            e.clips,
            e.rates.0,
            e.rates.1,
            e.clip_bytes / 1024,
            e.position_mm,
            e.rotation_deg,
            e.skin_mm,
            e.source_mm,
            e.weight_steps
        );
        if e.position_mm > POSITION_MM
            || e.rotation_deg > ROTATION_DEG
            || e.skin_mm > SKIN_MM
            || e.weight_steps > 1.0
        {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "over the bounds: {failed:?}");
}

/// A node `arm` holding a triangle `reach` m out along x, or with `instanced` one copy of it
/// placed there by `EXT_mesh_gpu_instancing`, with one animation `move` of its `path`, keyed at
/// `times` by `values` (three per key for a cubic spline).
fn arm_glb(
    name: &str,
    (reach, instanced): (f32, bool),
    path: &str,
    times: &[f32],
    values: &[[f32; 4]],
    cubic: bool,
) {
    let mut g = Glb::default();
    let x = if instanced { 0.0 } else { reach };
    let corners = [[x, 0.0, 0.0], [x, 0.01, 0.0], [x, 0.0, 0.01]];
    let pos = g.floats(&corners, "VEC3");
    let placed = g.floats(&[[reach, 0.0, 0.0]], "VEC3");
    let copies = match instanced {
        true => format!(
            r#","extensions":{{"EXT_mesh_gpu_instancing":{{"attributes":{{"TRANSLATION":{placed}}}}}}}"#
        ),
        false => String::new(),
    };
    let idx = g.indices(&[0, 1, 2]);
    let input = g.floats(&times.iter().map(|&t| [t]).collect::<Vec<_>>(), "SCALAR");
    let output = match path {
        "rotation" => g.floats(values, "VEC4"),
        _ => {
            let three: Vec<[f32; 3]> = values.iter().map(|v| [v[0], v[1], v[2]]).collect();
            g.floats(&three, "VEC3")
        }
    };
    let interpolation = if cubic { "CUBICSPLINE" } else { "LINEAR" };
    let rest = format!(
        r#""extensionsUsed":["EXT_mesh_gpu_instancing"],"scene":0,"scenes":[{{"nodes":[0]}}],
"nodes":[{{"name":"arm","mesh":0{copies}}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pos}}},"indices":{idx}}}]}}],
"animations":[{{"name":"move","samplers":[{{"input":{input},"output":{output},"interpolation":"{interpolation}"}}],"channels":[{{"sampler":0,"target":{{"node":0,"path":"{path}"}}}}]}}]"#
    );
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.glb"));
    g.write(&path, &rest);
}

/// The cooked clip of [`arm_glb`] `name`: its rate, and the farthest it puts the triangle from
/// where the source does, in mm, at 4 kHz.
fn arm_error(name: &str) -> (f32, f64, bool) {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.glb"));
    let src = load(&path);
    let cooked = cook(&path, name, GltfSettings::default());
    let skel = cooked.skeleton();
    let nodes = joint_nodes(&src, &skel);
    let vertices = source_vertices(&src, &nodes);
    let ibm = model_binds(&skel);
    let points = moved_points(&src, &nodes, &ibm, &vertices);
    let (_, clip) = cooked.clips().pop().unwrap();
    let anim = src.doc.animations().next().unwrap();
    let channels = src.channels(&anim);
    let start = time_range(&channels).0;
    let (far, _) = clip_error(&src, &channels, start, &clip, (&skel, &nodes), &points);
    (clip.header.sample_rate, far, at_max_rate(&clip))
}

/// A scale that pops from 0 to 1 over a quarter of a second, a cubic spline with flat ends,
/// moves nothing at the joint's origin and turns no axis, but carries the triangle a tenth of a
/// meter out: the clip holds it within the bound there.
#[test]
fn a_scale_pop_is_held_where_its_geometry_is() {
    let zero = [0.0; 4];
    let values = [zero, zero, zero, zero, [1.0, 1.0, 1.0, 0.0], zero];
    arm_glb(
        "arm-pop",
        (0.1, false),
        "scale",
        &[0.0, 0.25],
        &values,
        true,
    );
    let (rate, far, capped) = arm_error("arm-pop");
    assert!(
        far <= POSITION_MM || capped,
        "{far} mm at {rate} fps, past {POSITION_MM} mm"
    );
}

/// A swing of 90 degrees each way over 4 s, keyed at 24 fps, turns a triangle 20 m out by
/// arcs a joint's angle bound alone lets through: the clip holds it within the bound there.
#[test]
fn a_swing_is_held_at_the_reach_of_its_geometry() {
    let times: Vec<f32> = (0..=8 * 24).map(|k| k as f32 / 24.0).collect();
    let values: Vec<[f32; 4]> = times
        .iter()
        .map(|&t| {
            let a = 90f32.to_radians() * (std::f32::consts::TAU * t / 4.0).sin();
            Quat::from_rotation_z(a).to_array()
        })
        .collect();
    arm_glb(
        "arm-swing",
        (20.0, false),
        "rotation",
        &times,
        &values,
        false,
    );
    let (rate, far, capped) = arm_error("arm-swing");
    assert!(
        far <= POSITION_MM || capped,
        "{far} mm at {rate} fps, past {POSITION_MM} mm"
    );
    // A copy that hangs from the joint there is as far from it.
    arm_glb(
        "arm-swing-copy",
        (20.0, true),
        "rotation",
        &times,
        &values,
        false,
    );
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("arm-swing-copy.glb");
    let (_, clip) = cook(&path, "arm-swing-copy", GltfSettings::default())
        .clips()
        .pop()
        .unwrap();
    assert_eq!(clip.header.sample_rate, rate);
}

/// Local transforms of [`far_rig`]'s nodes `far`, `a` and `b`, and the turns `a` and `b` end at.
struct FarRig {
    far: (Vec3, Quat),
    a: Vec3,
    b: (Vec3, Quat),
    a_end: Quat,
    b_end: Quat,
}

impl FarRig {
    fn at(d: f32) -> Self {
        let turn = |axis: Vec3, deg: f32| Quat::from_axis_angle(axis.normalize(), deg.to_radians());
        Self {
            far: (Vec3::new(d, 3.0, -0.6 * d), turn(Vec3::Y, 37.0)),
            a: Vec3::new(0.3, 1.2, 0.1),
            b: (
                Vec3::new(0.4, 0.8, -0.2),
                turn(Vec3::new(1.0, 2.0, 3.0), 20.0),
            ),
            a_end: turn(Vec3::new(0.0, 1.0, 1.0), 50.0),
            b_end: turn(Vec3::X, -35.0),
        }
    }

    /// Model space from joint `a`'s and `b`'s, in f64, at the rest pose or the end of the clip.
    fn world(&self, end: bool) -> [orr_math::DMat4; 2] {
        use orr_math::{DMat4, DQuat};
        let m = |t: Vec3, q: Quat| {
            let q = DQuat::from_array(q.to_array().map(f64::from)).normalize();
            DMat4::from_rotation_translation(q, t.as_dvec3())
        };
        let (a, b) = match end {
            true => (self.a_end, self.b_end),
            false => (Quat::IDENTITY, self.b.1),
        };
        let a = m(self.far.0, self.far.1) * m(self.a, a);
        [a, a * m(self.b.0, b)]
    }
}

/// A static node `d` m from the model's origin, turned 37 degrees, over two animated nodes, `a`
/// and `b` below it, each holding a triangle: a rig of rigid parts whose clip turns both.
fn far_rig(name: &str, d: f32) -> PathBuf {
    let r = FarRig::at(d);
    let mut g = Glb::default();
    let pos = g.floats(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.2]], "VEC3");
    let idx = g.indices(&[0, 1, 2]);
    let input = g.floats(&[[0.0], [1.0]], "SCALAR");
    let a_keys = g.floats(&[Quat::IDENTITY.to_array(), r.a_end.to_array()], "VEC4");
    let b_keys = g.floats(&[r.b.1.to_array(), r.b_end.to_array()], "VEC4");
    let rest = format!(
        r#""scene":0,"scenes":[{{"nodes":[0]}}],
"nodes":[{{"name":"far","translation":{:?},"rotation":{:?},"children":[1]}},
{{"name":"a","translation":{:?},"mesh":0,"children":[2]}},
{{"name":"b","translation":{:?},"rotation":{:?},"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pos}}},"indices":{idx}}}]}}],
"animations":[{{"name":"go","samplers":[{{"input":{input},"output":{a_keys}}},{{"input":{input},"output":{b_keys}}}],
"channels":[{{"sampler":0,"target":{{"node":1,"path":"rotation"}}}},{{"sampler":1,"target":{{"node":2,"path":"rotation"}}}}]}}]"#,
        r.far.0.to_array(),
        r.far.1.to_array(),
        r.a.to_array(),
        r.b.0.to_array(),
        r.b.1.to_array(),
    );
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.glb"));
    g.write(&path, &rest);
    path
}

/// A rig 1 km and 4 km from its model's origin poses in a space of its own: its rest palette is
/// the identity, and a posed vertex lands where the same pose in f64 puts it, both to within the
/// rounding of the rig's own size, not of its distance.
#[test]
fn a_far_rig_poses_to_the_rounding_of_its_own_size() {
    // Rounding of f32 at a few meters, across a chain of three transforms.
    const NEAR: f64 = 4e-6;
    for d in [1000.0, 4000.0] {
        let name = format!("far-rig-{d}");
        let path = far_rig(&name, d);
        let cooked = cook(&path, &name, GltfSettings::default());
        let skel = cooked.skeleton();
        assert_eq!(skel.names, ["", "a", "b"]);
        let origin = skel.origin;
        let mesh_e = cooked.entry("/mesh");
        let info = MeshInfo::parse(&cooked.blob(mesh_e, 0)).unwrap();
        // Each vertex in mesh space with its joint.
        let mut vertices: Vec<(Vec3, usize)> = Vec::new();
        for p in &info.pages {
            let mut page = vec![0; p.resident_len as usize];
            mesh::decode_page(&cooked.blob(mesh_e, p.blob as usize), &mut page).unwrap();
            let view = PageView::new(&page, &info.header).unwrap();
            for c in (0..view.cluster_count()).map(|i| view.cluster(i)) {
                let joint = usize::from(info.parts[usize::from(c.part)].joint);
                for v in 0..u32::from(c.vertex_count) {
                    vertices.push((Vec3::from(view.position(&c, v)), joint));
                }
            }
        }
        assert!(vertices.iter().any(|v| v.1 == 2), "{d} m: no vertex on b");

        let n = skel.joint_count();
        let palette = |locals: &[Trs]| {
            let (mut model, mut out) = (vec![Affine3::IDENTITY; n], vec![Affine3::IDENTITY; n]);
            skel.model_pose(locals, &mut model);
            skel.palette(&model, &mut out);
            out
        };
        let rest = palette(&skel.rest);
        let still = vertices
            .iter()
            .map(|&(x, j)| f64::from(rest[j].transform_point3(x).distance(x)))
            .fold(0.0, f64::max);
        assert!(
            still <= NEAR,
            "{d} m: the rest pose moves a vertex {still} m"
        );

        let (_, clip) = cooked.clips().pop().unwrap();
        let mut locals = vec![Trs::IDENTITY; n];
        clip.sample(clip.header.duration, &mut locals);
        let posed = palette(&locals);
        let r = FarRig::at(d);
        let (from, to) = (r.world(false), r.world(true));
        let apart = vertices
            .iter()
            .map(|&(x, j)| {
                let at = x.as_dvec3() + origin;
                let want = to[j - 1] * from[j - 1].inverse();
                let got = posed[j].transform_point3(x).as_dvec3() + origin;
                got.distance(want.transform_point3(at))
            })
            .fold(0.0, f64::max);
        assert!(apart <= NEAR, "{d} m: a posed vertex lands {apart} m off");
    }
}

/// A skinned node bound and standing 1 km out, whose `EXT_mesh_gpu_instancing` copy flattens it
/// (scale 10, 10, 0): its baked model's skeleton takes the flattening after model space. Corners
/// sit on the mesh grid, so the cooked vertices are the source's shifted.
fn flattened_far_glb(name: &str) -> PathBuf {
    let hip = [1000.0f32, 2.0, -600.0];
    let corners = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.25]]
        .map(|c: [f32; 3]| std::array::from_fn::<f32, 3, _>(|a| c[a] + hip[a]));
    let mut g = Glb::default();
    let pos = g.floats(&corners, "VEC3");
    let jv = g.view(&[0u8; 12], None);
    let joints = g.accessor(jv, 5121, 3, "VEC4", "");
    let weights = g.floats(&[[1.0f32, 0.0, 0.0, 0.0]; 3], "VEC4");
    let bind = Mat4::from_translation(-Vec3::from(hip)).to_cols_array();
    let ibm = g.floats(&[bind], "MAT4");
    let scale = g.floats(&[[10.0f32, 10.0, 0.0]], "VEC3");
    let input = g.floats(&[[0.0], [1.0]], "SCALAR");
    let turn = Quat::from_rotation_y(0.6).to_array();
    let keys = g.floats(&[Quat::IDENTITY.to_array(), turn], "VEC4");
    let idx = g.indices(&[0, 1, 2]);
    let rest = format!(
        r#""extensionsUsed":["EXT_mesh_gpu_instancing"],"scene":0,"scenes":[{{"nodes":[0,1]}}],
"nodes":[{{"name":"hip","translation":{hip:?}}},
{{"name":"body","mesh":0,"skin":0,"extensions":{{"EXT_mesh_gpu_instancing":{{"attributes":{{"SCALE":{scale}}}}}}}}}],
"skins":[{{"joints":[0],"inverseBindMatrices":{ibm}}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pos},"JOINTS_0":{joints},"WEIGHTS_0":{weights}}},"indices":{idx}}}]}}],
"animations":[{{"name":"go","samplers":[{{"input":{input},"output":{keys}}}],"channels":[{{"sampler":0,"target":{{"node":0,"path":"rotation"}}}}]}}]"#
    );
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.glb"));
    g.write(&path, &rest);
    path
}

/// A skeleton that flattens after model space keeps its mesh in a bind space within the rig,
/// apart from the skeleton space the flattening takes it to: the vertices stay the size of the
/// rig, and the poses land where the same poses in f64 put the source's.
#[test]
fn a_far_rig_that_flattens_after_model_space_poses_to_its_own_rounding() {
    use orr_math::{DMat4, DQuat, DVec3};
    // [`a_far_rig_poses_to_the_rounding_of_its_own_size`]'s bound, at the copy's scale.
    const NEAR: f64 = 4e-5;
    let path = flattened_far_glb("far-flat");
    let cooked = cook(&path, "far-flat", GltfSettings::default());
    let model = "/body/baked/0";
    let e = cooked.entry(&format!("{model}/skeleton"));
    let skel = SkeletonInfo::parse(&cooked.blob(e, 0)).unwrap();
    let mesh_e = cooked.entry(&format!("{model}/mesh"));
    let info = MeshInfo::parse(&cooked.blob(mesh_e, 0)).unwrap();
    let mut vertices: Vec<(Vec3, usize)> = Vec::new();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        mesh::decode_page(&cooked.blob(mesh_e, p.blob as usize), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for c in (0..view.cluster_count()).map(|i| view.cluster(i)) {
            for v in 0..u32::from(c.vertex_count) {
                let j = usize::from(view.joints(&c, v).unwrap()[0]);
                vertices.push((Vec3::from(view.position(&c, v)), j));
            }
        }
    }
    let far = vertices.iter().map(|v| v.0.length()).fold(0.0, f32::max);
    assert!(far < 5.0, "a cooked vertex stands {far} m from its origin");

    let src = load(&path);
    let flat = DMat4::from_scale(DVec3::new(10.0, 10.0, 0.0));
    let hip = DVec3::new(1000.0, 2.0, -600.0);
    let source: Vec<DVec3> = source_vertices(&src, &[Some(0)])
        .0
        .iter()
        .map(|v| v.position.as_dvec3())
        .collect();
    let e = cooked.entry(&format!("{model}/clip/go"));
    let clip = Clip::parse(&cooked.blob(e, 0), &cooked.blob(e, 1)).unwrap();
    let n = skel.joint_count();
    let mut locals = vec![Trs::IDENTITY; n];
    for (t, turn) in [(0.0, 0.0), (clip.header.duration, 0.6)] {
        clip.sample(t, &mut locals);
        let (mut pose, mut palette) = (vec![Affine3::IDENTITY; n], vec![Affine3::IDENTITY; n]);
        skel.model_pose(&locals, &mut pose);
        skel.palette(&pose, &mut palette);
        let joint = DMat4::from_rotation_translation(DQuat::from_rotation_y(turn), hip);
        let want: Vec<DVec3> = source
            .iter()
            .map(|&v| (flat * joint * DMat4::from_translation(-hip)).transform_point3(v))
            .collect();
        for &(x, j) in &vertices {
            let got = palette[j].transform_point3(x).as_dvec3() + skel.origin;
            let off = want
                .iter()
                .map(|w| w.distance(got))
                .fold(f64::MAX, f64::min);
            assert!(off <= NEAR, "at {t} s a posed vertex lands {off} m off");
        }
    }
}
