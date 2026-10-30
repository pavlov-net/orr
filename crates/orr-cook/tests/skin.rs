#![allow(clippy::print_stdout)] // the external-set test reports measured errors

//! Skeletons, clips and CPU skinning against an independent glTF reference: a sampler that
//! evaluates the source channels at 1 kHz, and linear blend skinning of the source vertices
//! with the source weights.
//!
//! `skinned_set_matches_gltf` cooks an external set and is ignored unless asked for:
//! `ORR_SKINNED_SET=<dir>` holds `.glb` files with skins and animations.
//!
//! `cargo test --release -p orr-cook --test skin -- --ignored --nocapture`

#[path = "common/skinned.rs"]
mod common_skinned;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use gltf::animation::util::ReadOutputs;
use gltf::animation::{Interpolation, Property};
use orr_cook::{ArchiveSpec, AssetSpec, GltfSettings, LodMode, Manifest, MeshSettings, Source};
use orr_pack::clip::Clip;
use orr_pack::mesh::{self, Cluster, MeshHeader, MeshInfo, PageView};
use orr_pack::model::ModelInfo;
use orr_pack::skeleton::{NO_PARENT, SkeletonInfo, Trs};
use orr_pack::{Affine3, Archive, AssetId, Class, Target};

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
        (0..local.len())
            .map(|i| at(i, &parents, local, &mut world))
            .collect()
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

fn mat4(a: &Affine3) -> Mat4 {
    Mat4::from_cols_array_2d(&[0, 1, 2, 3].map(|j| {
        [
            a.rows[0][j],
            a.rows[1][j],
            a.rows[2][j],
            if j == 3 { 1.0 } else { 0.0 },
        ]
    }))
}

/// Translation distance in mm and rotation angle in degrees between two transforms.
fn pose_error(a: &Mat4, b: &Mat4) -> (f64, f64) {
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
    (
        f64::from(a.w_axis.truncate().distance(b.w_axis.truncate())) * 1000.0,
        f64::from(c.acos()).to_degrees(),
    )
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
    /// Joint poses from the cooked clips against the glTF sampler at 1 kHz.
    position_mm: f64,
    rotation_deg: f64,
    /// `mesh::skin` against reference skinning of the cooked vertex (position, joints and
    /// weights as decoded) with the glTF sampler's joint matrices.
    skin_mm: f64,
    /// `mesh::skin` against reference skinning of the source vertex: adds the grid and the
    /// unorm8 weights.
    source_mm: f64,
    /// Cooked weights against the source, in unorm8 steps.
    weight_steps: f64,
    clips: usize,
    rates: (f32, f32),
    clip_bytes: u64,
}

/// glTF node of each skeleton joint, by name.
fn joint_nodes(src: &Doc, skel: &SkeletonInfo) -> Vec<usize> {
    skel.names
        .iter()
        .map(|n| {
            let found: Vec<usize> = src
                .doc
                .nodes()
                .filter(|x| x.name().unwrap_or_default() == n)
                .map(|x| x.index())
                .collect();
            assert_eq!(found.len(), 1, "joint name {n:?} is not unique");
            found[0]
        })
        .collect()
}

/// A skinned source vertex: bind-space position and influences as skeleton joints.
struct Skinned {
    position: Vec3,
    influences: Vec<(usize, f32)>,
}

/// Skinned vertices, and the vertices of meshes without a skin as (node, local position).
fn source_vertices(src: &Doc, nodes: &[usize]) -> (Vec<Skinned>, Vec<(usize, Vec3)>) {
    let joint_of: HashMap<usize, usize> = nodes.iter().enumerate().map(|(j, &n)| (n, j)).collect();
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
    let mut ibm: Vec<Mat4> = skel.inverse_binds.iter().map(mat4).collect();
    for skin in src.doc.skins() {
        if let Some(m) = skin.reader(src.buf()).read_inverse_bind_matrices() {
            for (joint, m) in skin.joints().zip(m) {
                let j = nodes.iter().position(|&n| n == joint.index()).unwrap();
                let m = Mat4::from_cols_array_2d(&m);
                let tol = 1e-5 * m.to_cols_array().iter().fold(1.0f32, |a, b| a.max(b.abs()));
                assert!(ibm[j].abs_diff_eq(m, tol), "joint {j}: inverse bind");
                ibm[j] = m;
            }
        }
    }

    // Source vertices by grid cell, to pair them with cooked vertices.
    let (skinned, rigid) = source_vertices(&src, &nodes);
    let step = f64::from(info.header.step());
    let cell = |p: Vec3| {
        p.to_array()
            .map(|x| (f64::from(x) / step + 0.5).floor() as i32)
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
        by_cell.get(&view.grid(c, v)).map_or(&[][..], Vec::as_slice)
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
                                f64::from((cooked - source).abs()) * 255.0
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
        for i in 0..=((end - start) * 1000.0).ceil() as u32 {
            let t = (i as f32 / 1000.0).min(end - start);
            clip.sample(t, &mut locals);
            skel.model_pose(&locals, &mut pose);
            let world = src.world(&src.pose(&channels, start + t));
            for (j, &node) in nodes.iter().enumerate() {
                let (p, r) = pose_error(&mat4(&pose[j]), &world[node]);
                err.position_mm = err.position_mm.max(p);
                err.rotation_deg = err.rotation_deg.max(r);
            }
        }
        for i in 0..10 {
            let t = clip.header.duration * i as f32 / 9.0;
            clip.sample(t, &mut locals);
            skel.model_pose(&locals, &mut pose);
            skel.palette(&pose, &mut palette);
            let world = src.world(&src.pose(&channels, start + t));
            let reference: Vec<Mat4> = (0..n).map(|j| world[nodes[j]] * ibm[j]).collect();
            for view in &views {
                for c in (0..view.cluster_count()).map(|i| view.cluster(i)) {
                    let mut out = vec![[0.0; 3]; usize::from(c.vertex_count)];
                    mesh::skin(view, &c, &palette, &mut out);
                    for (v, got) in (0..).zip(out.into_iter().map(Vec3::from)) {
                        let joints = view.joints(&c, v).unwrap();
                        let weights = view.weights(&c, v).unwrap();
                        let decoded = Vec3::from(view.position(&c, v));
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
