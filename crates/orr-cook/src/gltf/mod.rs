//! glTF 2.0 import: the default scene's triangle primitives as one mesh per model, the
//! materials they use, and with skins a skeleton and its clips ([`crate::skin`]).
//!
//! With [`GltfLayout::Scene`] and without `GltfSettings::models` the whole scene is one model.
//! With models, each node whose name contains a pattern (and has no selected ancestor) is a
//! model holding its subtree, in the node's own space; meshes outside every selected subtree
//! are not cooked. Node transforms are applied to the vertices, and morph targets at their
//! default weights (the node's `weights`, else the mesh's). With skins, the mesh is in
//! bind space: a skinned primitive's vertices stay as stored (glTF ignores its node's
//! transform), take their joints and weights from `JOINTS_n`/`WEIGHTS_n`, and a primitive
//! without a skin follows its nearest skeleton joint rigidly; skins and model selection do
//! not mix.
//!
//! With [`GltfLayout::Meshes`] every glTF mesh the scene places is a model `mesh/<i>` in the
//! mesh's own space (`mesh/<i>/morph/<k>` for each other set of node morph weights), and every
//! node holding it, and every `EXT_mesh_gpu_instancing` instance, is a placement of it
//! ([`Scene::instances`]). Meshes are read when their cook starts, so
//! the import holds no geometry. Skins, model selection, joined models and assemblies need
//! the scene layout.
//!
//! Nodes hidden by `KHR_node_visibility` or the core `visible` property of glTF 2.1 are not
//! cooked, nor are their subtrees; hidden joints still pose their skins.
//!
//! In the scene layout a node with `EXT_mesh_gpu_instancing` is a model of its own, named
//! after the node and in its mesh's space, whatever the selection; its instances, scene space
//! from the node's transform times each instance's, become placements. Such models do not
//! mix with skins.
//!
//! A placement keeps rotation, per-axis scale and translation; a mirroring transform is a
//! negative scale, and a transform that collapses an axis places nothing. A transform that shears further than [`crate::SHEAR_MAX`] is not a placement:
//! the mesh is cooked once per such linear transform with it applied, as model
//! `<model>/baked/<k>`, placed by translation alone.
//!
//! Rigid parts come from the custom vertex attribute `PartAttribute::name` (integer, or
//! float rounded to the nearest integer; 0 where a primitive lacks it). With skins the
//! values are joints of the primitive's skin, pivots their bind positions and `Part::joint`
//! the skeleton joint; without, pivots come from the `PartAttribute::pivot` attribute, else
//! the model origin. With `GltfSettings::parts_from_skin`, a skin whose vertices each follow
//! one joint cooks as rigid parts instead of a skinned mesh: part `j` is skeleton joint `j`,
//! pivoted at its bind position.
//!
//! Joined models (`GltfSettings::joined`) share the coarsest grid step among them, and each
//! one's node must sit on that grid: translation a multiple of the step, rotation a multiple
//! of 90 degrees, unit scale, every placed grid coordinate within 24 bits.
//!
//! Materials map as [`material`] lists. Textures become the material's canonical channel
//! set ([`crate::texture::material`]); `texCoord` 0 or 1 becomes the binding's uv set and
//! the sampler's wrap modes set the mip filter's edges. Images are recognized by their
//! bytes, whatever their `mimeType`.

pub(crate) mod data;
mod material;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gltf::image::Source as ImageSource;
use gltf::mesh::{Mode, Semantic};
use orr_math::{Mat3, Mat4, Vec3};
use orr_pack::Hash32;
use rayon::prelude::*;
use serde::Deserialize;

use crate::import::{
    Decomposed, Placed, Scene, SceneMaterial, SceneModel, SceneSkeleton, placement,
};
use crate::mesh::{
    self, LazyMesh, MeshData, MeshSettings, MeshSource, PartData, Pivots, SectionData, Skin,
};
use crate::patch::{self, MaterialPatch, MeshPatch};
use crate::skin::{Graph, Skeleton, influences};
use crate::state::{Input, Memo};
use crate::texture::material::{ImageData, ImageFile};
use crate::{Error, GltfSettings, PartAttribute, Result};
use data::{Buffer, Data, Document};

/// How a glTF scene becomes models.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GltfLayout {
    /// The whole scene, or each selected node's subtree, is one model with node transforms
    /// applied; instanced nodes are models of their own.
    #[default]
    Scene,
    /// Every placed glTF mesh is a model in its own space, and every node and instance
    /// holding it a placement.
    Meshes,
}

#[derive(Default)]
struct Vertices {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    uvs1: Vec<[f32; 2]>,
    joints: Vec<[u8; 4]>,
    weights: Vec<[u8; 4]>,
    values: Vec<u16>,
    pivots: Vec<[f32; 3]>,
}

/// How a primitive's vertices are skinned.
enum Binding<'a> {
    /// A skin: skeleton joint per skin joint.
    Skin(&'a [u16]),
    /// Rigidly on one joint.
    Rigid(u16),
}

/// One model being assembled: its vertices and triangles per glTF material.
struct Building {
    node: Option<usize>,
    /// Model space from scene space.
    from_scene: Mat4,
    v: Vertices,
    tangents: bool,
    /// A primitive has TEXCOORD_1.
    uv1: bool,
    groups: BTreeMap<usize, Vec<u32>>,
    /// Bind position and skeleton joint per part value, from skins.
    joint_pivots: Vec<Option<([f32; 3], u16)>>,
}

impl Building {
    fn new(node: Option<usize>, from_scene: Mat4, parts: usize) -> Self {
        Self {
            node,
            from_scene,
            v: Vertices::default(),
            tangents: true,
            uv1: false,
            groups: BTreeMap::new(),
            joint_pivots: vec![None; parts],
        }
    }

    /// The model's mesh without sections, and the glTF material and index range of each
    /// non-empty triangle group, in material order.
    fn finish(
        self,
        parts: Option<&PartAttribute>,
        skinned: bool,
    ) -> (MeshData, Vec<(usize, Range<usize>)>) {
        let mut indices = Vec::new();
        let mut groups = Vec::new();
        for (index, tris) in self.groups {
            if tris.is_empty() {
                continue;
            }
            groups.push((index, indices.len()..indices.len() + tris.len()));
            indices.extend(tris);
        }
        let jp = &self.joint_pivots;
        let v = self.v;
        let mesh_parts = parts.map(|pa| PartData {
            values: v.values,
            count: pa.parts,
            by_section: pa.by_section,
            pivots: match (skinned, &pa.pivot) {
                (true, _) => {
                    Pivots::PerValue(jp.iter().map(|p| p.map_or([0.0; 3], |p| p.0)).collect())
                }
                (false, Some(_)) => Pivots::PerVertex(v.pivots),
                (false, None) => Pivots::Origin,
            },
            joints: skinned.then(|| jp.iter().map(|p| p.map_or(0, |p| p.1)).collect()),
        });
        let mesh = MeshData {
            positions: v.positions,
            normals: v.normals,
            tangents: self.tangents.then_some(v.tangents),
            uvs: std::iter::once(v.uvs)
                .chain(self.uv1.then_some(v.uvs1))
                .collect(),
            indices,
            sections: Vec::new(),
            skin: skinned.then_some(Skin {
                joints: v.joints,
                weights: v.weights,
            }),
            parts: mesh_parts,
            min_exponent: None,
        };
        (mesh, groups)
    }
}

/// The glTF material index of a primitive; `usize::MAX` for the default material.
fn material_index(prim: &gltf::Primitive<'_>) -> usize {
    prim.material().index().unwrap_or(usize::MAX)
}

/// Adds the buffer views a primitive's attributes and indices read.
fn primitive_views(prim: &gltf::Primitive<'_>, out: &mut Vec<usize>) {
    for (_, a) in prim.attributes() {
        Document::accessor_views(&a, out);
    }
    if let Some(a) = prim.indices() {
        Document::accessor_views(&a, out);
    }
}

/// What a primitive's vertices need besides the primitive.
struct Context<'a> {
    skeleton: Option<&'a Skeleton>,
    parts: Option<&'a PartAttribute>,
    /// Parts come from the skin's joints, one per vertex.
    rigid: bool,
}

/// Appends a triangle primitive to `b` with its morph targets at `weights`, its positions
/// taken to model space by `world` (winding and tangent signs flipped under a mirroring
/// `world`).
fn append_primitive(
    src: &Data<'_>,
    prim: &gltf::Primitive<'_>,
    world: Mat4,
    weights: &[f32],
    binding: Option<&Binding<'_>>,
    cx: &Context<'_>,
    b: &mut Building,
) -> Result<(), String> {
    if prim.mode() != Mode::Triangles {
        return Err(format!("mode {:?} is not triangles", prim.mode()));
    }
    let linear = Mat3::from_mat4(world);
    let flip = linear.determinant() < 0.0;
    let normal_m = linear.inverse().transpose();
    let position = prim.get(&Semantic::Positions).ok_or("no positions")?;
    let n = position.count();
    let counted = |a: &gltf::Accessor<'_>, k| {
        if a.count() != n {
            return Err("attribute counts differ".to_string());
        }
        src.floats(a, k)
    };
    // An attribute as rows of `k` floats, `None` when the primitive lacks it; positions,
    // normals and tangents with the weighted morph target deltas added.
    let rows = |s: Semantic, k: usize| -> Result<Option<Vec<f32>>, String> {
        let Some(a) = prim.get(&s) else {
            return Ok(None);
        };
        let mut out = counted(&a, k)?;
        for (t, &w) in prim
            .morph_targets()
            .zip(weights)
            .filter(|&(_, &w)| w != 0.0)
        {
            let delta = match s {
                Semantic::Positions => t.positions(),
                Semantic::Normals => t.normals(),
                Semantic::Tangents => t.tangents(),
                _ => None,
            };
            let Some(d) = delta else { continue };
            let d = counted(&d, 3)?;
            let xyz = out.chunks_exact_mut(k).flat_map(|r| &mut r[..3]);
            for (x, d) in xyz.zip(&d) {
                *x += w * d;
            }
        }
        Ok(Some(out))
    };
    let custom = |name: &str, k| -> Result<Option<Vec<f32>>, String> {
        let key = name
            .strip_prefix('_')
            .ok_or_else(|| format!("{name}: custom vertex attributes start with _"))?;
        rows(Semantic::Extras(key.into()), k)
    };
    let positions: Vec<[f32; 3]> = rows(Semantic::Positions, 3)?
        .expect("checked above")
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&p| world.transform_point3(Vec3::from(p)).to_array())
        .collect();
    let mut indices: Vec<u32> = match prim.indices() {
        Some(a) => src.indices(&a)?,
        None => (0..n as u32).collect(),
    };
    if !indices.len().is_multiple_of(3) || indices.iter().any(|&i| i as usize >= n) {
        return Err("bad index list".into());
    }
    if flip {
        indices.chunks_exact_mut(3).for_each(|t| t.swap(1, 2));
    }
    let [uvs, uvs1] = [0, 1].map(|set| rows(Semantic::TexCoords(set), 2));
    let (uvs, uvs1) = (uvs?, uvs1?);
    b.uv1 |= uvs1.is_some();
    let zeros = |v: Option<Vec<f32>>| {
        v.map_or_else(|| vec![[0.0; 2]; n], |v| v.as_chunks::<2>().0.to_vec())
    };
    let (uvs, uvs1) = (zeros(uvs), zeros(uvs1));
    let tangents = rows(Semantic::Tangents, 4)?.map(|t| {
        t.as_chunks::<4>()
            .0
            .iter()
            .map(|&[x, y, z, w]| {
                let t = (linear * Vec3::new(x, y, z)).to_array();
                [t[0], t[1], t[2], if flip { -w } else { w }]
            })
            .collect::<Vec<_>>()
    });
    b.tangents &= tangents.is_some();
    let tangents = tangents.unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; n]);
    let normals: Option<Vec<[f32; 3]>> = rows(Semantic::Normals, 3)?.map(|ns| {
        ns.as_chunks::<3>()
            .0
            .iter()
            .map(|&nv| (normal_m * Vec3::from(nv)).to_array())
            .collect()
    });
    let skin: Vec<([u8; 4], [u8; 4])> = match binding {
        None => Vec::new(),
        Some(Binding::Rigid(j)) => vec![([*j as u8; 4], [255, 0, 0, 0]); n],
        Some(Binding::Skin(map)) => {
            let mut sets = Vec::new();
            for set in 0.. {
                match (
                    rows(Semantic::Joints(set), 4)?,
                    rows(Semantic::Weights(set), 4)?,
                ) {
                    (None, None) => break,
                    (Some(j), Some(w)) => sets.push((
                        j.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|v| v.map(|x| x as u16))
                            .collect::<Vec<_>>(),
                        w.as_chunks::<4>().0.to_vec(),
                    )),
                    _ => return Err(format!("JOINTS_{set} and WEIGHTS_{set} come in pairs")),
                }
            }
            if sets.is_empty() {
                return Err("skinned without JOINTS_0".into());
            }
            (0..n)
                .map(|i| influences(&sets, map, i))
                .collect::<Result<_, _>>()?
        }
    };
    let (values, pivots) = match cx.parts {
        _ if cx.rigid => {
            let values = skin
                .iter()
                .enumerate()
                .map(|(i, (j, w))| match w[1..] {
                    [0, 0, 0] => Ok(u16::from(j[0])),
                    _ => Err(format!(
                        "vertex {i} follows {} joints; parts from a skin take one",
                        w.iter().filter(|&&x| x > 0).count()
                    )),
                })
                .collect::<Result<_, _>>()?;
            (values, Vec::new())
        }
        None => (Vec::new(), Vec::new()),
        Some(pa) => {
            let values: Vec<u16> = match custom(&pa.name, 1)? {
                Some(v) => v
                    .iter()
                    .map(|&x| {
                        let r = x.round();
                        (0.0..=f32::from(u16::MAX))
                            .contains(&r)
                            .then_some(r as u16)
                            .ok_or_else(|| format!("{} value {x} is not a part", pa.name))
                    })
                    .collect::<Result<_, _>>()?,
                None => vec![0; n],
            };
            let pivots = match (&pa.pivot, cx.skeleton) {
                (Some(name), None) => custom(name, 3)?.map_or_else(
                    || vec![[0.0; 3]; n],
                    |p| {
                        p.as_chunks::<3>()
                            .0
                            .iter()
                            .map(|&q| world.transform_point3(Vec3::from(q)).to_array())
                            .collect()
                    },
                ),
                _ => Vec::new(),
            };
            if let (Some(sk), Some(binding)) = (cx.skeleton, binding) {
                for &v in &values {
                    if v >= pa.parts {
                        continue;
                    }
                    let joint = match binding {
                        Binding::Skin(map) => *map
                            .get(usize::from(v))
                            .ok_or_else(|| format!("part value {v} is not a joint of the skin"))?,
                        Binding::Rigid(j) => *j,
                    };
                    let slot = &mut b.joint_pivots[usize::from(v)];
                    if slot.is_some_and(|s| s.1 != joint) {
                        return Err(format!("part value {v} names two skeleton joints"));
                    }
                    *slot = Some((sk.bind_origin(joint), joint));
                }
            }
            (values, pivots)
        }
    };

    let base = b.v.positions.len() as u32;
    let tris = b.groups.entry(material_index(prim)).or_default();
    let v = &mut b.v;
    match normals {
        Some(normals) => {
            v.positions.extend(positions);
            v.normals.extend(normals);
            v.tangents.extend(tangents);
            v.uvs.extend(uvs);
            v.uvs1.extend(uvs1);
            v.joints.extend(skin.iter().map(|s| s.0));
            v.weights.extend(skin.iter().map(|s| s.1));
            v.values.extend(values);
            v.pivots.extend(pivots);
            tris.extend(indices.iter().map(|i| base + i));
        }
        // Without normals, glTF asks for flat shading: one vertex per corner.
        None => {
            for t in indices.as_chunks::<3>().0 {
                let p = t.map(|i| Vec3::from(positions[i as usize]));
                let face = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
                for &i in t {
                    let i = i as usize;
                    tris.push(v.positions.len() as u32);
                    v.positions.push(positions[i]);
                    v.normals.push(face.to_array());
                    v.tangents.push(tangents[i]);
                    v.uvs.push(uvs[i]);
                    v.uvs1.push(uvs1[i]);
                    if let Some(&(j, w)) = skin.get(i) {
                        v.joints.push(j);
                        v.weights.push(w);
                    }
                    if let Some(&x) = values.get(i) {
                        v.values.push(x);
                    }
                    if let Some(&x) = pivots.get(i) {
                        v.pivots.push(x);
                    }
                }
            }
        }
    }
    Ok(())
}

/// The buffer files that `views` read.
fn buffer_files<'d>(src: &'d Document, views: &[usize]) -> Vec<&'d Path> {
    let mut paths: Vec<&Path> = views
        .iter()
        .filter_map(|&v| match src.buffers.get(src.view_source(v).0) {
            Some(Buffer::File(p)) => Some(p.as_path()),
            _ => None,
        })
        .collect();
    paths.sort_unstable();
    paths.dedup();
    paths
}

/// The images of the document: embedded ones read now, files when they are needed.
fn images(asset: &str, src: &Document, dir: &Path) -> Result<Vec<ImageFile>> {
    let fail = |message: String| Error::Import {
        asset: asset.into(),
        message,
    };
    let mut out = Vec::new();
    for img in src.doc.images() {
        let what = || format!("image {}", img.index());
        let (name, data) = match img.source() {
            ImageSource::View { view, .. } => {
                let data = Data::load(src, [view.index()], asset)?;
                let bytes = data.view(&view).map_err(&fail)?.to_vec();
                (what(), ImageData::Bytes(bytes))
            }
            ImageSource::Uri { uri, .. } => match data::data_uri(uri) {
                Some(bytes) => {
                    let bytes = bytes.map_err(|e| fail(format!("{}: {e}", what())))?;
                    (what(), ImageData::Bytes(bytes))
                }
                None => (
                    uri.to_string(),
                    ImageData::File {
                        path: dir.join(uri),
                        read: None,
                    },
                ),
            },
        };
        out.push(ImageFile {
            name: img.name().map_or(name, str::to_string),
            data,
        });
    }
    Ok(out)
}

pub(crate) fn import(
    asset: &str,
    path: &Path,
    memo: &Memo,
    settings: &GltfSettings,
) -> Result<Scene> {
    let fail = |message: String| Error::Import {
        asset: asset.into(),
        message,
    };
    let (bytes, hash) = memo.read(path)?;
    let mut files = vec![Input {
        path: path.to_owned(),
        hash,
    }];
    let dir = path.parent().unwrap_or(Path::new("")).to_owned();
    let src = data::open(&bytes, &dir).map_err(|m| fail(format!("{}: {m}", path.display())))?;
    drop(bytes);
    let src = Arc::new(src);
    let images = images(asset, &src, &dir)?;
    let scene = src
        .doc
        .default_scene()
        .or_else(|| src.doc.scenes().next())
        .ok_or_else(|| fail("no scene".into()))?;
    let graph = Graph::new(&src.doc, &scene);
    let mut b = Imported {
        asset,
        src: &src,
        settings,
        memo,
        materials: BTreeMap::new(),
        models: Vec::new(),
        names: HashSet::new(),
        shown: shown_nodes(&src.json, &graph),
        hashes: BTreeMap::new(),
        instances: Vec::new(),
        files: &mut files,
    };
    let skeleton = match settings.layout {
        GltfLayout::Scene => scene_layout(&mut b, &graph)?,
        GltfLayout::Meshes => {
            meshes_layout(&mut b, &graph)?;
            None
        }
    };
    let Imported {
        materials,
        mut models,
        instances,
        ..
    } = b;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    files.dedup_by(|a, b| a.path == b.path);
    // Material slots in glTF index order, the default material last; only those in use.
    let used: Vec<usize> = materials.keys().copied().collect();
    for m in &mut models {
        for slot in &mut m.materials {
            *slot = used.binary_search(slot).expect("material in use");
        }
    }
    Ok(Scene {
        models,
        materials: materials.into_values().collect(),
        images,
        skeletons: skeleton.into_iter().collect(),
        instances,
        files,
    })
}

/// What both layouts build.
struct Imported<'a> {
    asset: &'a str,
    src: &'a Arc<Document>,
    settings: &'a GltfSettings,
    memo: &'a Memo,
    /// By glTF material index.
    materials: BTreeMap<usize, SceneMaterial>,
    /// [`SceneModel::materials`] holds glTF material indices until the import ends.
    models: Vec<SceneModel>,
    names: HashSet<String>,
    /// The scene's nodes in order, without those hidden ([`shown_nodes`]).
    shown: Vec<usize>,
    /// Hashes of the buffer files read so far.
    hashes: BTreeMap<PathBuf, Hash32>,
    instances: Vec<(String, Vec<Placed>)>,
    files: &'a mut Vec<Input>,
}

impl Imported<'_> {
    fn fail(&self, message: String) -> Error {
        Error::Import {
            asset: self.asset.into(),
            message,
        }
    }

    /// Sorts and dedups `views` and records the buffer files they read (hashing new ones in
    /// parallel through the memo); returns those files' hashes.
    fn record(&mut self, views: &mut Vec<usize>) -> Result<Vec<(PathBuf, Hash32)>> {
        views.sort_unstable();
        views.dedup();
        let paths = buffer_files(self.src, views);
        let new: Vec<(PathBuf, Hash32)> = paths
            .par_iter()
            .filter(|p| !self.hashes.contains_key(**p))
            .map(|p| Ok((p.to_path_buf(), self.memo.hash(p)?)))
            .collect::<Result<_>>()?;
        for (path, hash) in new {
            self.files.push(Input {
                path: path.clone(),
                hash,
            });
            self.hashes.insert(path, hash);
        }
        Ok(paths
            .into_iter()
            .map(|p| (p.to_path_buf(), self.hashes[p]))
            .collect())
    }

    /// Reserves a model name.
    fn name(&mut self, name: &str) -> Result<()> {
        if !self.names.insert(name.to_string()) {
            return Err(self.fail(format!("two models are named {name:?}")));
        }
        Ok(())
    }

    /// Registers the material of `prim`.
    fn material(&mut self, prim: &gltf::Primitive<'_>) -> Result<()> {
        let index = material_index(prim);
        if self.materials.contains_key(&index) {
            return Ok(());
        }
        let mat = prim.material();
        let t = material::textures(&self.src.doc, &self.src.json, &mat)
            .map_err(|m| self.fail(format!("material {index}: {m}")))?;
        let mut m = material::material(&self.src.json, &mat);
        patch::apply_matching(
            &self.settings.material_overrides,
            mat.name().unwrap_or_default(),
            &mut m,
            MaterialPatch::apply,
        );
        self.materials.insert(
            index,
            SceneMaterial {
                suffix: match index {
                    usize::MAX => "default".to_string(),
                    i => i.to_string(),
                },
                material: m,
                textures: t,
            },
        );
        Ok(())
    }

    /// Mesh settings of the model named `name`.
    fn mesh_settings(&self, name: &str) -> MeshSettings {
        let mut s = self.settings.mesh.clone();
        patch::apply_matching(
            &self.settings.mesh_overrides,
            name,
            &mut s,
            MeshPatch::apply,
        );
        s
    }

    /// The models of glTF mesh `mesh` placed by `transforms` under name `name`: the mesh in
    /// its own space for the transforms a placement holds, and a baked model per other
    /// linear transform.
    fn placed(
        &mut self,
        name: &str,
        mesh: usize,
        weights: &[f32],
        transforms: &[Mat4],
    ) -> Result<()> {
        let mut direct = Vec::new();
        let mut baked: Vec<(Mat4, Vec<Placed>)> = Vec::new();
        let mut baked_of: HashMap<[u32; 16], usize> = HashMap::new();
        for m in transforms {
            match placement(*m) {
                Decomposed::Placed(p) => direct.push(p),
                Decomposed::Nothing => {}
                Decomposed::Sheared => {
                    let linear = Mat4::from_mat3(Mat3::from_mat4(*m));
                    let k = *baked_of
                        .entry(linear.to_cols_array().map(f32::to_bits))
                        .or_insert_with(|| {
                            baked.push((linear, Vec::new()));
                            baked.len() - 1
                        });
                    baked[k].1.push(Placed {
                        position: m.w_axis.truncate().as_dvec3(),
                        rotation: orr_math::Quat::IDENTITY,
                        scale: Vec3::ONE,
                    });
                }
            }
        }
        if !direct.is_empty() {
            self.lazy_model(name, mesh, weights, Mat4::IDENTITY)?;
            self.instances.push((name.to_string(), direct));
        }
        for (k, (linear, placed)) in baked.into_iter().enumerate() {
            let name = format!("{name}/baked/{k}");
            self.lazy_model(&name, mesh, weights, linear)?;
            self.instances.push((name, placed));
        }
        Ok(())
    }

    /// Model `name`: glTF mesh `mesh` at morph `weights` with `linear` applied, read when
    /// it cooks.
    fn lazy_model(&mut self, name: &str, mesh: usize, weights: &[f32], linear: Mat4) -> Result<()> {
        self.name(name)?;
        let src = Arc::clone(self.src);
        let m = src.doc.meshes().nth(mesh).expect("mesh index");
        let mut views = Vec::new();
        let mut groups: BTreeMap<usize, u64> = BTreeMap::new();
        for prim in m.primitives() {
            self.material(&prim)?;
            primitive_views(&prim, &mut views);
            for t in prim.morph_targets() {
                for a in [t.positions(), t.normals(), t.tangents()]
                    .into_iter()
                    .flatten()
                {
                    Document::accessor_views(&a, &mut views);
                }
            }
            let what = format!("mesh {mesh} primitive {}", prim.index());
            if prim.mode() != Mode::Triangles {
                return Err(self.fail(format!("{what}: mode {:?} is not triangles", prim.mode())));
            }
            let positions = prim
                .get(&Semantic::Positions)
                .ok_or_else(|| self.fail(format!("{what}: no positions")))?;
            let corners = prim.indices().map_or(positions.count(), |a| a.count());
            if !corners.is_multiple_of(3) {
                return Err(self.fail(format!("{what}: bad index list")));
            }
            *groups.entry(material_index(&prim)).or_default() += corners as u64;
        }
        let checked = self.record(&mut views)?;
        let mut sections = Vec::new();
        let mut materials = Vec::new();
        let mut at = 0usize;
        for (&index, &corners) in &groups {
            if corners == 0 {
                continue;
            }
            let end = at + corners as usize;
            sections.push(SectionData::new(
                at..end,
                &self.materials[&index].material,
                Default::default(),
            ));
            materials.push(index);
            at = end;
        }
        if sections.is_empty() {
            return Err(self.fail(format!("model {name} has no triangles")));
        }
        // Everything the loaded mesh is made of: the mesh's JSON, its accessors and views,
        // the bytes of the buffers they read, the transform and the part attribute.
        let mut h = blake3::Hasher::new();
        let mut json = |v: &serde_json::Value| {
            let s = serde_json::to_string(v).expect("JSON serializes");
            h.update(&(s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        };
        json(&src.json["meshes"][mesh]);
        for prim in m.primitives() {
            for a in prim.attributes().map(|a| a.1).chain(prim.indices()) {
                json(&src.json["accessors"][a.index()]);
            }
        }
        for &v in &views {
            json(&src.json["bufferViews"][v]);
        }
        for &v in &views {
            match &src.buffers[src.view_source(v).0] {
                Buffer::File(p) => h.update(&self.hashes[p].0),
                Buffer::Bytes(_, hash) => h.update(&hash.0),
                Buffer::Fallback => h.update(b"fallback"),
            };
        }
        h.update(bytemuck::cast_slice(&linear.to_cols_array()));
        h.update(bytemuck::cast_slice(weights));
        h.update(format!("{:?}", self.settings.parts_from_attribute).as_bytes());
        let parts = self.settings.parts_from_attribute.clone();
        let asset = self.asset.to_string();
        let weights = weights.to_vec();
        let load = move |memo: &Memo| -> Result<MeshData> {
            for (path, hash) in &checked {
                memo.check(path, *hash)?;
            }
            let data = Data::load(&src, views.iter().copied(), &asset)?;
            let cx = Context {
                skeleton: None,
                parts: parts.as_ref(),
                rigid: false,
            };
            let count = parts.as_ref().map_or(0, |p| usize::from(p.parts));
            let mut b = Building::new(None, Mat4::IDENTITY, count);
            let gm = src.doc.meshes().nth(mesh).expect("mesh index");
            for prim in gm.primitives() {
                append_primitive(&data, &prim, linear, &weights, None, &cx, &mut b).map_err(
                    |m| Error::Import {
                        asset: asset.clone(),
                        message: format!("mesh {mesh} primitive {}: {m}", prim.index()),
                    },
                )?;
            }
            Ok(b.finish(parts.as_ref(), false).0)
        };
        let lazy = LazyMesh {
            sections,
            triangles: at as u64 / 3,
            source: Hash32(*h.finalize().as_bytes()),
            load: Arc::new(load),
        };
        let settings = self.mesh_settings(name);
        // Assemblies split the mesh when it is planned, so such a model is read now.
        let mesh = if settings.assemblies.is_some() {
            MeshSource::Data(Box::new(
                MeshSource::Lazy(lazy).load(self.asset, self.memo)?,
            ))
        } else {
            MeshSource::Lazy(lazy)
        };
        self.models.push(SceneModel {
            node: Some(name.to_string()),
            mesh,
            subd: None,
            skeleton: None,
            settings,
            materials,
        });
        Ok(())
    }
}

/// Node `n`'s `EXT_mesh_gpu_instancing` object, if it has one and a mesh.
fn instancing(src: &Document, n: usize) -> Option<&serde_json::Value> {
    src.doc.nodes().nth(n)?.mesh()?;
    src.json["nodes"][n]
        .get("extensions")
        .and_then(|e| e.get("EXT_mesh_gpu_instancing"))
}

/// The buffer views the accessors of `EXT_mesh_gpu_instancing` nodes read.
fn instancing_views(src: &Document, graph: &Graph) -> Vec<usize> {
    let mut views = Vec::new();
    for &n in &graph.order {
        let Some(attributes) = instancing(src, n).and_then(|e| e["attributes"].as_object()) else {
            continue;
        };
        for i in attributes.values().filter_map(serde_json::Value::as_u64) {
            if let Some(a) = src.doc.accessors().nth(i as usize) {
                Document::accessor_views(&a, &mut views);
            }
        }
    }
    views
}

/// The world transforms of `n`'s copies: its world transform times each
/// `EXT_mesh_gpu_instancing` instance, or its world transform alone.
fn node_transforms(src: &Data<'_>, graph: &Graph, n: usize) -> Result<Vec<Mat4>, String> {
    let world = graph.world[n];
    let Some(ext) = instancing(src.doc, n) else {
        return Ok(vec![world]);
    };
    let what = format!("node {n} EXT_mesh_gpu_instancing");
    let read = |attribute: &str, k: usize| -> Result<Option<Vec<f32>>, String> {
        let Some(i) = ext["attributes"].get(attribute) else {
            return Ok(None);
        };
        let a = i
            .as_u64()
            .and_then(|i| src.doc.doc.accessors().nth(i as usize))
            .ok_or_else(|| format!("{what}: {attribute} names no accessor"))?;
        src.floats(&a, k).map(Some)
    };
    let t = read("TRANSLATION", 3)?;
    let r = read("ROTATION", 4)?;
    let sc = read("SCALE", 3)?;
    let counts: Vec<usize> = [(&t, 3), (&r, 4), (&sc, 3)]
        .iter()
        .filter_map(|(v, k)| v.as_ref().map(|v| v.len() / k))
        .collect();
    let Some(&count) = counts.first() else {
        return Err(format!("{what}: no TRANSLATION, ROTATION or SCALE"));
    };
    if counts.iter().any(|&c| c != count) {
        return Err(format!("{what}: attribute counts differ"));
    }
    Ok((0..count)
        .map(|i| {
            let t = t
                .as_ref()
                .map_or(Vec3::ZERO, |v| Vec3::from_slice(&v[3 * i..]));
            let r = r.as_ref().map_or(orr_math::Quat::IDENTITY, |v| {
                orr_math::Quat::from_slice(&v[4 * i..]).normalize()
            });
            let s = sc
                .as_ref()
                .map_or(Vec3::ONE, |v| Vec3::from_slice(&v[3 * i..]));
            world * Mat4::from_scale_rotation_translation(s, r, t)
        })
        .collect())
}

/// Loads the instancing accessors and records the files they read.
fn instancing_data<'d>(b: &mut Imported<'_>, src: &'d Document, graph: &Graph) -> Result<Data<'d>> {
    let mut views = instancing_views(src, graph);
    b.record(&mut views)?;
    Data::load(src, views, b.asset)
}

/// Every glTF mesh the scene places is a model, every node and instance holding it a
/// placement.
fn meshes_layout(b: &mut Imported<'_>, graph: &Graph) -> Result<()> {
    let s = b.settings;
    let src = Arc::clone(b.src);
    let doc = &src.doc;
    let skinned = graph.order.iter().any(|&n| {
        doc.nodes()
            .nth(n)
            .is_some_and(|d| d.mesh().is_some() && d.skin().is_some())
    });
    let needs_scene = [
        (!s.models.is_empty(), "model selection"),
        (!s.joined.is_empty(), "joined models"),
        (skinned, "skins"),
    ];
    if let Some((_, what)) = needs_scene.iter().find(|n| n.0) {
        return Err(b.fail(format!("{what} need the scene layout")));
    }
    // Every placed mesh's buffer files, hashed together.
    let shown = b.shown.clone();
    let mut views = Vec::new();
    for &n in &shown {
        if let Some(mesh) = doc.nodes().nth(n).and_then(|d| d.mesh()) {
            mesh.primitives()
                .for_each(|p| primitive_views(&p, &mut views));
        }
    }
    b.record(&mut views)?;
    let data = instancing_data(b, &src, graph)?;
    // Per mesh, its placements per set of morph weights, the mesh's own first.
    type Placings = Vec<(Vec<f32>, Vec<Mat4>)>;
    let mut by_mesh: BTreeMap<usize, Placings> = BTreeMap::new();
    for &n in &shown {
        let node = doc.nodes().nth(n).expect("node index");
        let Some(mesh) = node.mesh() else { continue };
        let own = mesh.weights().unwrap_or_default();
        let weights = node.weights().unwrap_or(own);
        let transforms = node_transforms(&data, graph, n).map_err(|m| b.fail(m))?;
        let sets = by_mesh
            .entry(mesh.index())
            .or_insert_with(|| vec![(own.to_vec(), Vec::new())]);
        match sets.iter_mut().find(|s| s.0 == weights) {
            Some(s) => s.1.extend(transforms),
            None => sets.push((weights.to_vec(), transforms)),
        }
    }
    for (mesh, sets) in by_mesh {
        for (k, (weights, transforms)) in sets.iter().enumerate() {
            let name = match k {
                0 => format!("mesh/{mesh}"),
                k => format!("mesh/{mesh}/morph/{k}"),
            };
            if !transforms.is_empty() {
                b.placed(&name, mesh, weights, transforms)?;
            }
        }
    }
    Ok(())
}

/// The whole scene, or the selected nodes' subtrees, as models with node transforms
/// applied, and instanced nodes as placed models. Returns the skeleton.
fn scene_layout(b: &mut Imported<'_>, graph: &Graph) -> Result<Option<SceneSkeleton>> {
    let (asset, settings) = (b.asset, b.settings);
    let src = Arc::clone(b.src);
    let doc = &src.doc;
    let fail = |message: String| Error::Import {
        asset: asset.into(),
        message,
    };
    let shown = b.shown.clone();
    let instanced: Vec<usize> = shown
        .iter()
        .copied()
        .filter(|&n| instancing(&src, n).is_some())
        .collect();
    let mut is_instanced = vec![false; graph.parent.len()];
    instanced.iter().for_each(|&n| is_instanced[n] = true);
    let inst = instancing_data(b, &src, graph)?;
    for &n in &instanced {
        let node = doc.nodes().nth(n).expect("node index");
        let name = node
            .name()
            .map_or_else(|| format!("node{n}"), str::to_string);
        let transforms = node_transforms(&inst, graph, n).map_err(fail)?;
        let mesh = node.mesh().expect("instanced nodes have meshes");
        let weights = node.weights().or(mesh.weights()).unwrap_or_default();
        b.placed(&name, mesh.index(), weights, &transforms)?;
    }

    // The views of the other nodes' meshes, the skins and the animations; instanced nodes'
    // meshes are read when they cook.
    let mut views = Vec::new();
    for &n in &shown {
        if is_instanced[n] {
            continue;
        }
        if let Some(mesh) = doc.nodes().nth(n).and_then(|d| d.mesh()) {
            for p in mesh.primitives() {
                primitive_views(&p, &mut views);
                for t in p.morph_targets() {
                    for a in [t.positions(), t.normals(), t.tangents()]
                        .into_iter()
                        .flatten()
                    {
                        Document::accessor_views(&a, &mut views);
                    }
                }
            }
        }
    }
    for skin in doc.skins() {
        if let Some(a) = skin.inverse_bind_matrices() {
            Document::accessor_views(&a, &mut views);
        }
    }
    for s in doc
        .animations()
        .flat_map(|a| a.samplers().collect::<Vec<_>>())
    {
        Document::accessor_views(&s.input(), &mut views);
        Document::accessor_views(&s.output(), &mut views);
    }
    b.record(&mut views)?;
    let data = Data::load(&src, views, asset)?;

    let mut skins: Vec<usize> = shown
        .iter()
        .filter_map(|&n| {
            let node = doc.nodes().nth(n)?;
            node.mesh()?;
            Some(node.skin()?.index())
        })
        .collect();
    skins.sort_unstable();
    skins.dedup();
    let skeleton = Skeleton::build(&data, graph, &skins).map_err(fail)?;
    if skeleton.is_some() && !settings.models.is_empty() {
        return Err(fail(
            "model selection needs a scene without skins: skinned vertices stay in bind space"
                .into(),
        ));
    }
    if !instanced.is_empty() && skeleton.is_some() {
        return Err(fail("instanced nodes need a scene without skins".into()));
    }
    if settings.parts_from_skin && settings.parts_from_attribute.is_some() {
        return Err(fail(
            "parts_from_skin and parts_from_attribute do not mix".into(),
        ));
    }
    let rigid = skeleton.as_ref().filter(|_| settings.parts_from_skin);
    let (mut models, mut owner) = select_models(doc, graph, &shown, settings).map_err(fail)?;
    for &n in &instanced {
        owner[n] = None;
    }
    let parts = settings.parts_from_attribute.as_ref();
    let cx = Context {
        skeleton: skeleton.as_ref(),
        parts,
        rigid: rigid.is_some(),
    };
    for &n in &shown {
        let node = doc.nodes().nth(n).expect("node index");
        let (Some(mesh), Some(m)) = (node.mesh(), owner[n]) else {
            continue;
        };
        let (world, binding) = match (&skeleton, node.skin()) {
            (None, _) => (models[m].from_scene * graph.world[n], None),
            (Some(sk), Some(s)) => (
                Mat4::IDENTITY,
                Some(Binding::Skin(sk.skin_joints(s.index()))),
            ),
            (Some(sk), None) => {
                let (j, mat) = sk.attach(n, graph).map_err(fail)?;
                (mat, Some(Binding::Rigid(j)))
            }
        };
        let weights = node.weights().or(mesh.weights()).unwrap_or_default();
        for prim in mesh.primitives() {
            b.material(&prim)?;
            let reads_uv1 = b.materials[&material_index(&prim)]
                .textures
                .refs()
                .any(|r| r.uv_set == 1);
            if reads_uv1 && prim.get(&Semantic::TexCoords(1)).is_none() {
                return Err(fail(format!(
                    "mesh {} primitive {}: its material reads TEXCOORD_1, which the primitive does not have",
                    mesh.index(),
                    prim.index()
                )));
            }
            append_primitive(
                &data,
                &prim,
                world,
                weights,
                binding.as_ref(),
                &cx,
                &mut models[m],
            )
            .map_err(|e| {
                fail(format!(
                    "mesh {} primitive {}: {e}",
                    mesh.index(),
                    prim.index()
                ))
            })?;
        }
    }

    let mut out = Vec::new();
    for building in models {
        let node = building.node;
        // The whole scene may hold nothing but instanced nodes.
        if node.is_none() && building.groups.values().all(Vec::is_empty) && !instanced.is_empty() {
            continue;
        }
        let name = node.map(|n| node_name(doc, n).to_string());
        if let Some(name) = &name {
            b.name(name)?;
        }
        let joint_values = rigid.map(|_| building.v.values.clone());
        let (mut mesh, groups) = building.finish(parts, skeleton.is_some() && rigid.is_none());
        if let (Some(sk), Some(values)) = (rigid, joint_values) {
            let joints = 0..sk.info.joint_count() as u16;
            mesh.parts = Some(PartData {
                values,
                count: joints.end,
                by_section: false,
                pivots: Pivots::PerValue(joints.map(|j| sk.bind_origin(j)).collect()),
                joints: None,
            });
        }
        if groups.is_empty() {
            return Err(fail(match &name {
                Some(n) => format!("model {n} has no triangles"),
                None => "the scene has no triangles".into(),
            }));
        }
        mesh.sections = groups
            .iter()
            .map(|(index, range)| {
                SectionData::new(
                    range.clone(),
                    &b.materials[index].material,
                    Default::default(),
                )
            })
            .collect();
        out.push((
            node,
            SceneModel {
                settings: b.mesh_settings(name.as_deref().unwrap_or_default()),
                node: name,
                subd: None,
                mesh: MeshSource::Data(Box::new(mesh)),
                skeleton: skeleton.as_ref().map(|_| 0),
                materials: groups.into_iter().map(|g| g.0).collect(),
            },
        ));
    }
    join(&mut out, graph, &settings.joined).map_err(fail)?;
    let clips = match &skeleton {
        Some(sk) => sk.clips(&data, &settings.clips).map_err(fail)?,
        None => Vec::new(),
    };
    b.models.extend(out.into_iter().map(|m| m.1));
    Ok(skeleton.map(|s| SceneSkeleton {
        prefix: String::new(),
        info: s.info,
        clips,
    }))
}

/// The scene's nodes in [`Graph::order`] without those hidden, with their subtrees, by
/// `KHR_node_visibility` or the core `visible` property of glTF 2.1.
fn shown_nodes(json: &serde_json::Value, graph: &Graph) -> Vec<usize> {
    let mut hidden = vec![false; graph.parent.len()];
    let mut shown = Vec::with_capacity(graph.order.len());
    for &n in &graph.order {
        let node = &json["nodes"][n];
        hidden[n] = node["visible"] == false
            || node["extensions"]["KHR_node_visibility"]["visible"] == false
            || graph.parent[n].is_some_and(|p| hidden[p]);
        if !hidden[n] {
            shown.push(n);
        }
    }
    shown
}

/// The name of node `i`, empty when it has none.
fn node_name(doc: &gltf::Document, i: usize) -> &str {
    doc.nodes()
        .nth(i)
        .and_then(|d| d.name())
        .unwrap_or_default()
}

/// The models to build and, per visible glTF node, the model it belongs to.
fn select_models(
    doc: &gltf::Document,
    graph: &Graph,
    shown: &[usize],
    settings: &GltfSettings,
) -> Result<(Vec<Building>, Vec<Option<usize>>), String> {
    let parts = settings
        .parts_from_attribute
        .as_ref()
        .map_or(0, |p| usize::from(p.parts));
    let building = |node, from_scene| Building::new(node, from_scene, parts);
    let mut owner: Vec<Option<usize>> = vec![None; graph.parent.len()];
    if settings.models.is_empty() {
        for &i in shown {
            owner[i] = Some(0);
        }
        return Ok((vec![building(None, Mat4::IDENTITY)], owner));
    }
    let mut models = Vec::new();
    let mut names: Vec<&str> = Vec::new();
    let mut matched = vec![false; settings.models.len()];
    for &i in shown {
        if let Some(m) = graph.parent[i].and_then(|p| owner[p]) {
            owner[i] = Some(m);
            continue;
        }
        let name = node_name(doc, i);
        let mut hit = false;
        for (s, sel) in settings.models.iter().enumerate() {
            if name.contains(sel.node.as_str()) {
                matched[s] = true;
                hit = true;
            }
        }
        if !hit {
            continue;
        }
        if names.contains(&name) {
            return Err(format!("two selected nodes are named {name:?}"));
        }
        names.push(name);
        owner[i] = Some(models.len());
        models.push(building(Some(i), graph.world[i].inverse()));
    }
    if let Some(s) = matched.iter().position(|m| !m) {
        let pattern = settings.models[s].node.as_str();
        let names = |hidden: bool| -> Vec<&str> {
            let matching = graph
                .order
                .iter()
                .map(|&n| (n, node_name(doc, n)))
                .filter(|(n, name)| name.contains(pattern) && shown.contains(n) != hidden);
            matching.map(|(_, name)| name).collect()
        };
        let (nested, hidden) = (names(false), names(true));
        let mut only = Vec::new();
        if !nested.is_empty() {
            only.push(format!("nodes inside other selected models: {nested:?}"));
        }
        if !hidden.is_empty() {
            only.push(format!("hidden nodes: {hidden:?}"));
        }
        return Err(if only.is_empty() {
            format!("model pattern {pattern:?} matches no node")
        } else {
            format!(
                "model pattern {pattern:?} matches only {}",
                only.join(" and ")
            )
        });
    }
    Ok((models, owner))
}

/// Puts the joined models on one grid and checks that their nodes keep them on it.
fn join(
    models: &mut [(Option<usize>, SceneModel)],
    graph: &Graph,
    patterns: &[String],
) -> Result<(), String> {
    if patterns.is_empty() {
        return Ok(());
    }
    let matches = |m: &SceneModel, p: &str| m.node.as_deref().is_some_and(|n| n.contains(p));
    let joined: Vec<usize> = (0..models.len())
        .filter(|&i| patterns.iter().any(|p| matches(&models[i].1, p)))
        .collect();
    if let Some(p) = patterns
        .iter()
        .find(|p| !joined.iter().any(|&i| matches(&models[i].1, p)))
    {
        return Err(format!("joined pattern {p:?} matches no selected model"));
    }
    fn positions(m: &SceneModel) -> &[[f32; 3]] {
        match &m.mesh {
            MeshSource::Data(d) => &d.positions,
            MeshSource::Lazy(_) => unreachable!("selected models are built in memory"),
        }
    }
    let k = joined
        .iter()
        .map(|&i| {
            let m = &models[i].1;
            mesh::grid_exponent(positions(m), m.settings.position_step_log2)
        })
        .max()
        .expect("a joined model");
    let step = 2f64.powi(i32::from(k));
    for &i in &joined {
        let (node, m) = &mut models[i];
        let name = m.node.clone().unwrap_or_default();
        let world = graph.world[node.expect("selected models have nodes")];
        let cols = world.to_cols_array_2d();
        // Each column one unit axis: entries 0 or +-1, one non-zero per column and row.
        let axis = cols[..3].iter().all(|c| {
            c[..3]
                .iter()
                .all(|&x| (x - x.round()).abs() <= 1e-6 && x.round().abs() <= 1.0)
                && c[..3].iter().filter(|x| x.abs() > 0.5).count() == 1
        });
        let det = Mat3::from_mat4(world).determinant();
        if !axis || (det - 1.0).abs() > 1e-5 {
            return Err(format!(
                "joined model {name}: rotation is not a multiple of 90 degrees at unit scale"
            ));
        }
        let mut placed = 0f64;
        for &x in &cols[3][..3] {
            let g = f64::from(x) / step;
            if g != g.round() {
                return Err(format!(
                    "joined model {name}: translation {:?} m is not on the 2^{k} m grid",
                    &cols[3][..3]
                ));
            }
            placed = placed.max(g.abs());
        }
        let MeshSource::Data(mesh) = &mut m.mesh else {
            unreachable!("selected models are built in memory");
        };
        let local = mesh
            .positions
            .iter()
            .flatten()
            .map(|&x| (f64::from(x) / step + 0.5).floor().abs())
            .fold(0.0, f64::max);
        if placed + local >= f64::from(1 << 23) {
            return Err(format!(
                "joined model {name}: placed grid coordinates need more than 24 bits"
            ));
        }
        mesh.min_exponent = Some(k);
    }
    Ok(())
}
