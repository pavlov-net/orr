//! What a scene importer (glTF, USD) hands the planner: models with their meshes, the
//! materials and images they use, skeletons with their clips, instanced models and their
//! placements, and the files it read.

use std::collections::HashMap;
use std::sync::Arc;

use orr_math::{DMat3, DMat4, DQuat, DVec3, Mat3, Mat4, Vec3};
use orr_pack::instance::MotionRow;
use orr_pack::material::{Material, slot};
use orr_pack::skeleton::SkeletonInfo;
use serde::Deserialize;

use crate::instances::InstanceMotion;
use crate::mesh::{self, MeshSettings, MeshSource};
use crate::skin::{ClipSettings, ClipSource, spectral_norm};
use crate::state::Input;
use crate::subd::{CageData, SubdSettings};
use crate::texture::material::{ImageFile, MaterialTextures};

/// Selects nodes of a scene (glTF nodes, USD prims) as models.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeSelect {
    /// A node whose name contains this becomes model `<asset>/<node name>`, unless an
    /// ancestor already did.
    pub node: String,
}

/// Rigid parts from a per-vertex attribute.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PartAttribute {
    /// Custom vertex attribute name, such as `_ID`: integer, or float scalar rounded to the
    /// nearest integer.
    pub name: String,
    /// Declared value count; every value is below it, and values without triangles stay as
    /// empty parts.
    pub parts: u16,
    /// One part per (value, section): part = value x sections + section.
    pub by_section: bool,
    /// Custom per-vertex vec3 pivot attribute, read without a skin.
    pub pivot: Option<String>,
}

pub(crate) struct Scene {
    pub(crate) models: Vec<SceneModel>,
    pub(crate) materials: Vec<SceneMaterial>,
    pub(crate) images: Vec<ImageFile>,
    pub(crate) skeletons: Vec<SceneSkeleton>,
    /// How the skeletons' clips are resampled.
    pub(crate) clips: ClipSettings,
    pub(crate) instances: Vec<PlacedModel>,
    /// Every file read.
    pub(crate) files: Vec<Input>,
    pub(crate) warnings: Vec<String>,
}

impl Scene {
    /// Per skeleton, each joint's reach: the farthest, in the joint's own space (its inverse
    /// bind of model space), from its origin of anything it moves. That is the vertices and cage
    /// points weighted to it and the parts it poses, in their bind pose, and the copies that
    /// hang from it: their model's points as placed, or as their rows move them while the
    /// skeleton's clips run.
    pub(crate) fn reaches(&self) -> Vec<Vec<f64>> {
        let mut out: Vec<Vec<f64>> = self
            .skeletons
            .iter()
            .map(|s| vec![0.0; s.info.joint_count()])
            .collect();
        let binds: Vec<Vec<DMat4>> = self
            .skeletons
            .iter()
            .map(|s| {
                let ib = s.info.inverse_binds.iter();
                ib.map(|&b| Mat4::from(b).as_dmat4()).collect()
            })
            .collect();
        for m in &self.models {
            let Some(s) = m.skeleton else { continue };
            let (reach, binds) = (&mut out[s], &binds[s]);
            let mut moves = |j: usize, p: DVec3| {
                reach[j] = reach[j].max(binds[j].transform_point3(p).length());
            };
            let mesh = m.mesh.data();
            for (i, &p) in mesh.positions.iter().enumerate() {
                let p = Vec3::from(p).as_dvec3();
                if let Some(skin) = &mesh.skin {
                    for k in (0..4).filter(|&k| skin.weights[i][k] > 0) {
                        moves(usize::from(skin.joints[i][k]), p);
                    }
                }
                // A part's joint poses it.
                if let Some(parts) = &mesh.parts {
                    let v = parts.values[i];
                    let j = parts.joints.as_ref().map_or(v, |j| j[usize::from(v)]);
                    moves(usize::from(j), p);
                }
            }
            if let Some(cage) = &m.subd
                && let Some(skin) = &cage.cage.skin
            {
                for (p, influence) in cage.cage.cage.positions.iter().zip(skin) {
                    for k in (0..4).filter(|&k| influence.weights[k] > 0) {
                        moves(usize::from(influence.joints[k]), p.as_dvec3());
                    }
                }
            }
        }
        for copies in &self.instances {
            let Some((s, joint)) = copies.frame else {
                continue;
            };
            let model = self
                .models
                .iter()
                .find(|m| m.node.as_ref() == Some(&copies.model));
            let model = model.expect("copies place a model of the scene");
            let ib = binds[s][usize::from(joint)];
            let reach = &mut out[s][usize::from(joint)];
            let points: Vec<DVec3> = model.points().collect();
            match &copies.motion {
                None => {
                    let (center, radius) = sphere(&points);
                    for p in &copies.placed {
                        let linear = DMat3::from_mat4(ib) * p.linear.as_dmat3();
                        let at = ib.transform_point3(p.position + p.linear.as_dmat3() * center);
                        *reach = reach.max(at.length() + spectral_norm(linear) * radius);
                    }
                }
                Some(motion) => {
                    let spans = self.skeletons[s].clips.iter().map(|c| c.1.span);
                    let (lo, hi) = spans.fold((f64::MAX, f64::MIN), |(lo, hi), (start, length)| {
                        (lo.min(start), hi.max(start + length))
                    });
                    let right = points.iter().map(|&p| motion.right.transform_point3(p));
                    let radius = right.map(DVec3::length).fold(0.0, f64::max);
                    *reach = reach.max(moving_reach(ib, motion, (lo, hi), radius));
                }
            }
        }
        out
    }
}

/// The farthest a copy of `motion`'s rows comes from the origin of the space `ib` takes theirs
/// to, while the clock runs over `clock` (s), for a model whose points `right` takes within
/// `radius` of its origin. A row at `dt` into its interval stands at `p + dt (v + a dt / 2)`
/// from its anchor, turned, and scaled by at most the largest of `s + g dt` on an axis.
fn moving_reach(ib: DMat4, motion: &InstanceMotion, clock: (f64, f64), radius: f64) -> f64 {
    let to = ib * motion.left;
    let stretch = spectral_norm(DMat3::from_mat4(to));
    let records = motion.anchors.len();
    let mut reach: f64 = 0.0;
    for (k, &start) in motion.times.iter().enumerate() {
        let start = f64::from(start);
        // The first interval runs back without end, and the last on.
        let from = if k == 0 { f64::MIN } else { start };
        let to_time = motion.times.get(k + 1).map_or(f64::MAX, |&t| f64::from(t));
        let (a, b) = (from.max(clock.0), to_time.min(clock.1));
        if a > b {
            continue;
        }
        let (da, db) = (a - start, b - start);
        let far = da.abs().max(db.abs());
        for (i, anchor) in motion.anchors.iter().enumerate() {
            let row = &motion.rows[k * records + i];
            if row.rotation == [0.0; 4] {
                continue;
            }
            let v = |x: [f32; 3]| Vec3::from(x).as_dvec3();
            let at = DVec3::from_array(*anchor) + v(row.position);
            let path =
                v(row.velocity).length() * far + v(row.acceleration).length() * far * far / 2.0;
            let scale = |dt: f64| (v(row.scale) + v(row.growth) * dt).abs().max_element();
            let size = scale(da).max(scale(db)) * radius;
            reach = reach.max(to.transform_point3(at).length() + stretch * (path + size));
        }
    }
    reach
}

/// A sphere around `points`: the center of their box, and the farthest of them from it.
fn sphere(points: &[DVec3]) -> (DVec3, f64) {
    let (lo, hi) = points
        .iter()
        .fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        });
    let center = (lo + hi) / 2.0;
    let radius = points
        .iter()
        .map(|&p| p.distance(center))
        .fold(0.0, f64::max);
    (center, radius)
}

impl SceneModel {
    /// The points of its mesh and cage, in its space.
    fn points(&self) -> impl Iterator<Item = DVec3> {
        let mesh = self
            .mesh
            .data()
            .positions
            .iter()
            .map(|&p| Vec3::from(p).as_dvec3());
        let cage = self.subd.iter().flat_map(|c| &c.cage.cage.positions);
        mesh.chain(cage.map(|p| p.as_dvec3()))
    }
}

/// The placements of one model that share pages.
pub(crate) struct PlacedModel {
    pub(crate) model: String,
    /// In scene space, or the bind space of `frame`'s joint.
    pub(crate) placed: Vec<Placed>,
    /// Index into [`Scene::skeletons`] and the joint the placements hang from.
    pub(crate) frame: Option<(usize, u16)>,
    /// On pages of its own, whose records are its copies alone: for the palettes of a model with
    /// a skeleton, and for a frame or motion, which are a page's.
    pub(crate) alone: bool,
    /// How the placements move over time, in their order.
    pub(crate) motion: Option<Arc<InstanceMotion>>,
}

/// One placement: position and a linear transform that [`collapses`] passes.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Placed {
    pub(crate) position: DVec3,
    pub(crate) linear: Mat3,
}

pub(crate) struct SceneSkeleton {
    /// Entry names `<asset>/<prefix>/skeleton` and `<asset>/<prefix>/clip/<clip>`; an empty
    /// prefix leaves its part out.
    pub(crate) prefix: String,
    pub(crate) info: SkeletonInfo,
    /// Clips by name, which the planner resamples once it knows what each joint moves.
    pub(crate) clips: Vec<(String, ClipSource)>,
}

pub(crate) struct SceneModel {
    /// The selected node's name; `None` for the whole scene.
    pub(crate) node: Option<String>,
    /// A mesh without sections cooks no mesh entry.
    pub(crate) mesh: MeshSource,
    pub(crate) subd: Option<SceneSubd>,
    /// Index into [`Scene::skeletons`] of a skinned mesh.
    pub(crate) skeleton: Option<usize>,
    pub(crate) settings: MeshSettings,
    /// Index into [`Scene::materials`] per mesh section.
    pub(crate) materials: Vec<usize>,
}

/// A model's subdivision cage.
pub(crate) struct SceneSubd {
    pub(crate) cage: CageData,
    /// Index into [`Scene::materials`] per cage section.
    pub(crate) materials: Vec<usize>,
    pub(crate) settings: SubdSettings,
}

#[derive(Clone)]
pub(crate) struct SceneMaterial {
    /// Entry name suffix: `<asset>/material/<suffix>`, textures under
    /// `<asset>/texture/<suffix>/`; empty names the material `<asset>` itself.
    pub(crate) suffix: String,
    pub(crate) material: Material,
    pub(crate) textures: MaterialTextures,
    /// The index among the scene's materials of its back.
    pub(crate) back: Option<usize>,
}

impl SceneMaterial {
    /// The uv set a surface with this material takes its tangents from; `None` when it reads
    /// none. Without a normal map the shaders read set 0.
    pub(crate) fn tangent_set(&self) -> Option<usize> {
        let m = &self.material;
        let normal = self.textures.normal.as_ref();
        let normal = normal.filter(|_| !m.object_normal(slot::NORMAL));
        let coat = self.textures.coat_normal.as_ref();
        let coat = coat.filter(|_| !m.object_normal(slot::COAT_NORMAL));
        match normal.or(coat) {
            Some((r, _)) => Some(usize::from(r.uv_set)),
            None => mesh::reads_tangent(m).then_some(0),
        }
    }
}

/// Copies whose linear transform collapses: the transform, for the importer to bake into a
/// model of its own, and the copies of that model, placed by translation alone.
pub(crate) type Baked = (DMat4, Vec<Placed>);

pub(crate) fn largest(m: DMat3) -> f64 {
    m.to_cols_array()
        .into_iter()
        .fold(0.0, |a, x| a.max(x.abs()))
}

pub(crate) fn column_lengths(l: DMat3) -> DVec3 {
    DVec3::new(l.x_axis.length(), l.y_axis.length(), l.z_axis.length())
}

/// Whether finite `l` does not invert in f32, the arithmetic of ray tracing instances and the
/// shaders' `linear_inverse`, which is the one rule for what cannot be an instance. Scale, on
/// either side, costs nothing there, so `diag(1, 1e-12, 1)` inverts in any orientation it can be
/// stored in. Inverting needs a determinant: at unit size a normal f32, and above the rounding of
/// the six products it sums, which is all that is left of a product of transforms that flattens.
/// It needs an inverse that fits an f32, which small entries that nearly cancel do not leave.
/// Shader arithmetic need not keep subnormal entries.
pub(crate) fn collapses(l: DMat3) -> bool {
    let floor = f64::from(f32::MIN_POSITIVE);
    let [x, y, z] = l
        .to_cols_array_2d()
        .map(|c| c.map(|v| if v.abs() < floor { 0.0 } else { v }));
    let products = [
        x[0] * y[1] * z[2],
        x[1] * y[2] * z[0],
        x[2] * y[0] * z[1],
        -x[2] * y[1] * z[0],
        -x[1] * y[0] * z[2],
        -x[0] * y[2] * z[1],
    ];
    let kept = DMat3::from_cols_array_2d(&[x, y, z]);
    let rounding = products.iter().map(|p| p.abs()).sum::<f64>() * 1e-5;
    products.iter().sum::<f64>().abs() <= rounding.max(floor * largest(kept).powi(3))
        || largest(kept.inverse()) > f64::from(f32::MAX)
}

/// Whether finite `l` does not invert in f64, the arithmetic of the cook's own transforms, which
/// inverts what [`collapses`] keeps from an instance: the determinant is lost only to the rounding
/// of the six products it sums.
pub(crate) fn singular(l: DMat3) -> bool {
    let [x, y, z] = l.to_cols_array_2d();
    let products = [
        x[0] * y[1] * z[2],
        x[1] * y[2] * z[0],
        x[2] * y[0] * z[1],
        -x[2] * y[1] * z[0],
        -x[1] * y[0] * z[2],
        -x[0] * y[2] * z[1],
    ];
    let rounding = products.iter().map(|p| p.abs()).sum::<f64>() * 1e-14;
    products.iter().sum::<f64>().abs() <= rounding || !l.inverse().is_finite()
}

/// The inverse transpose of `l` up to a positive scale, which exists where `l` collapses. Its
/// largest entry is 1, so it fits an f32 whatever the size of `l`.
pub(crate) fn normal_matrix(l: DMat3) -> DMat3 {
    let [x, y, z] = [l.x_axis, l.y_axis, l.z_axis];
    let m = DMat3::from_cols(y.cross(z), z.cross(x), x.cross(y));
    let sign = if l.determinant() < 0.0 { -1.0 } else { 1.0 };
    m * (sign / largest(m).max(f64::MIN_POSITIVE))
}

/// Whether `l` keeps lengths.
fn rigid(l: DMat3) -> bool {
    largest(l.transpose() * l - DMat3::IDENTITY) <= 1e-5
}

/// A transform that goes into a mesh, with the records and intervals it stands in.
type BakedSet = (DMat4, Vec<(usize, Vec<usize>)>);

/// Moving copies of a model that states world lengths, split as [`kept_lengths`] says.
pub(crate) struct KeptLengths {
    /// What stays on the model, its rows as they were.
    pub(crate) direct: Option<(Vec<Placed>, Arc<InstanceMotion>)>,
    /// What cooks a model of its own with the transform applied, and moves rigidly.
    pub(crate) baked: Vec<(DMat4, Vec<Placed>, Arc<InstanceMotion>)>,
    /// Whether some of `direct` stretches over time, which no single transform can hold.
    pub(crate) stretched: bool,
}

/// A copy's transform is `left * T(t) * R(t) * S(t) * right`. A height baked into a mesh keeps
/// its world length through a rigid transform only, so what stretches has to go into the mesh, and
/// that takes a stretch that stands still: `Q(t) * B` with `Q` rigid and `B` constant. Under a
/// uniform `left` that is `B = c S right`, for any `R(t)`. Otherwise `R` has to hold as well, and
/// `B` is the whole linear part. `S` and `R` that step make one `B` per step, each with the
/// intervals it stands in, so only a `growth`, or a `spin` under a `left` that isn't uniform, has
/// no `B`: its copy stays on the model, stretching.
pub(crate) fn kept_lengths(placed: &[Placed], m: &InstanceMotion) -> KeptLengths {
    let n = placed.len();
    let intervals = m.times.len();
    let row = |k: usize, i: usize| &m.rows[k * n + i];
    let lin = DMat3::from_mat4(m.left);
    let gram = lin.transpose() * lin;
    let unit = (gram.x_axis.x + gram.y_axis.y + gram.z_axis.z) / 3.0;
    let uniform = largest(gram * (1.0 / unit) - DMat3::IDENTITY) <= 1e-5;
    let c = unit.sqrt();
    let vector = |v: [f32; 3]| DVec3::from_array(v.map(f64::from));
    let quat = |q: [f32; 4]| DQuat::from_array(q.map(f64::from));
    let bake = |r: &MotionRow| match uniform {
        true => DMat4::from_scale(vector(r.scale) * c) * m.right,
        false => {
            let stretch = DMat3::from_diagonal(vector(r.scale));
            DMat4::from_mat3(lin * DMat3::from_quat(quat(r.rotation)) * stretch) * m.right
        }
    };
    let there = |k: &usize, i: usize| row(*k, i).rotation != [0.0; 4];

    let mut stretched = false;
    // Per record the intervals that stay, and per transform the records and intervals it takes.
    let mut staying: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut sets: Vec<BakedSet> = Vec::new();
    let mut set_of: HashMap<[u64; 16], usize> = HashMap::new();
    for (i, stays) in staying.iter_mut().enumerate() {
        let alive: Vec<usize> = (0..intervals).filter(|k| there(k, i)).collect();
        let grows = |k: &usize| {
            let r = row(*k, i);
            r.growth != [0.0; 3] || (!uniform && r.spin != [0.0; 3])
        };
        if alive.iter().any(grows) {
            stretched = true;
            *stays = alive;
            continue;
        }
        for k in alive {
            let b = bake(row(k, i));
            if rigid(DMat3::from_mat4(b)) {
                stays.push(k);
                continue;
            }
            let key = b.to_cols_array().map(|x| (x + 0.0).to_bits());
            let s = *set_of.entry(key).or_insert_with(|| {
                sets.push((b, Vec::new()));
                sets.len() - 1
            });
            let records = &mut sets[s].1;
            match records.iter_mut().find(|r| r.0 == i) {
                Some(r) => r.1.push(k),
                None => records.push((i, vec![k])),
            }
        }
    }

    let kept: Vec<usize> = (0..n).filter(|&i| !staying[i].is_empty()).collect();
    let direct = (!kept.is_empty()).then(|| {
        let mut rows = vec![MotionRow::default(); intervals * kept.len()];
        for (slot, &i) in kept.iter().enumerate() {
            for &k in &staying[i] {
                rows[k * kept.len() + slot] = *row(k, i);
            }
        }
        let table = InstanceMotion {
            rows,
            anchors: kept.iter().map(|&i| m.anchors[i]).collect(),
            times: m.times.clone(),
            left: m.left,
            right: m.right,
            start: m.start,
        };
        let records = kept.iter().map(|&i| placed[i]).collect();
        (records, Arc::new(table))
    });

    // What the rows keep once `B` holds the stretch: the rest of `left` goes to them.
    let left = match uniform {
        true => DMat4::from_cols(
            m.left.x_axis / c,
            m.left.y_axis / c,
            m.left.z_axis / c,
            m.left.w_axis,
        ),
        false => DMat4::from_translation(m.left.w_axis.truncate()),
    };
    let through = |v: DVec3| if uniform { v * c } else { lin * v };
    let moved = |v: [f32; 3]| through(vector(v)).as_vec3().to_array();
    let rigid_row = |r: &MotionRow| MotionRow {
        position: moved(r.position),
        velocity: moved(r.velocity),
        acceleration: moved(r.acceleration),
        rotation: if uniform {
            r.rotation
        } else {
            [0.0, 0.0, 0.0, 1.0]
        },
        spin: r.spin,
        scale: [1.0; 3],
        growth: [0.0; 3],
    };
    let baked = sets.into_iter().map(|(b, records)| {
        let mut rows = vec![MotionRow::default(); intervals * records.len()];
        let mut anchors = Vec::with_capacity(records.len());
        let mut at = Vec::with_capacity(records.len());
        for (slot, (i, held)) in records.iter().enumerate() {
            for &k in held {
                rows[k * records.len() + slot] = rigid_row(row(k, *i));
            }
            let anchor = through(DVec3::from_array(m.anchors[*i]));
            anchors.push(anchor.to_array());
            let first = rows[held[0] * records.len() + slot];
            let stands = left
                * DMat4::from_rotation_translation(
                    quat(first.rotation),
                    anchor + vector(first.position),
                );
            at.push(Placed {
                position: stands.w_axis.truncate(),
                linear: DMat3::from_mat4(stands).as_mat3(),
            });
        }
        let table = InstanceMotion {
            rows,
            anchors,
            times: m.times.clone(),
            left,
            right: DMat4::IDENTITY,
            start: m.start,
        };
        (b, at, Arc::new(table))
    });
    KeptLengths {
        baked: baked.collect(),
        direct,
        stretched,
    }
}

/// The copies `transforms` place: those an instance can hold, then the rest by linear
/// transform. With `lengths` a model states world lengths its mesh holds, so an instance can hold
/// only what keeps them.
pub(crate) fn placements(
    transforms: &[DMat4],
    lengths: bool,
) -> Result<(Vec<Placed>, Vec<Baked>), String> {
    let mut placed = Vec::new();
    let mut baked: Vec<Baked> = Vec::new();
    let mut baked_of: HashMap<[u32; 9], usize> = HashMap::new();
    for (i, m) in transforms.iter().enumerate() {
        if !m.as_mat4().is_finite() {
            return Err(format!("the transform of copy {i} is not finite"));
        }
        let position = m.w_axis.truncate();
        let exact = DMat3::from_mat4(*m);
        let l = exact.as_mat3();
        if !collapses(l.as_dmat3()) && (!lengths || rigid(exact)) {
            placed.push(Placed {
                position,
                linear: l,
            });
            continue;
        }
        // Adding zero turns -0.0 into 0.0: copies scaled to zero differ only in such signs.
        let k = *baked_of
            .entry(l.to_cols_array().map(|x| (x + 0.0).to_bits()))
            .or_insert_with(|| {
                baked.push((DMat4::from_mat3(exact), Vec::new()));
                baked.len() - 1
            });
        baked[k].1.push(Placed {
            position,
            linear: Mat3::IDENTITY,
        });
    }
    Ok((placed, baked))
}
