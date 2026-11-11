//! glTF 2.0 import: the default scene's triangle primitives as one mesh per model, the
//! materials they use, and what moves as models with skeletons and clips ([`crate::skin`]).
//! Strips and fans become triangle lists. Points, lines and primitives without positions have no
//! surface to cook: they are skipped, and each one placed is a warning in the report.
//!
//! With [`GltfLayout::Scene`] and without `GltfSettings::models` the whole scene is one model.
//! With models, each node whose name contains a pattern (and has no selected ancestor) is a
//! model holding its subtree, in the node's own space; meshes outside every selected subtree
//! are not cooked. Node transforms are applied to the vertices, and morph targets at their
//! default weights (the node's `weights`, else the mesh's).
//!
//! With [`GltfLayout::Meshes`] every glTF mesh the scene places is a model `mesh/<i>` in the
//! mesh's own space (`mesh/<i>/morph/<k>` for each other set of node morph weights), and every
//! node holding it, and every `EXT_mesh_gpu_instancing` instance, is a placement of it
//! ([`Scene::instances`]). Meshes are read when their cook starts, so
//! the import holds no geometry. Model selection and joined models need the scene layout.
//!
//! What moves does not stay in those models. A mesh moves if it has a skin, or a joint of a skin
//! or a node an animation targets stands at or above it. Each [`Rig`] of them is a model of its
//! own in scene space, named after its first joint (`node<i>` where that is empty or taken),
//! with its skeleton and clips under that name; a scene that is all one rig stays the
//! whole-scene model. A selected model keeps what moves in it, with one skeleton: it holds what
//! its meshes need of the scene below its node, and a node above or at it that moves it is the
//! placement's to follow. Such a model's mesh is in bind space: a skinned primitive's vertices
//! stay as stored (glTF ignores its node's transform), take their joints and weights from
//! `JOINTS_n`/`WEIGHTS_n`, and a primitive without a skin follows its nearest joint rigidly, or
//! with none stays where its node puts it, on the skeleton's still joint. A rig without a skin
//! cooks as rigid parts, one per joint, unless parts come from an attribute, which leaves its
//! vertices skinned to their joints.
//!
//! Nodes hidden by `KHR_node_visibility` or the core `visible` property of glTF 2.1 are not
//! cooked, nor are their subtrees; hidden joints still pose their skins.
//!
//! In either layout a node with `EXT_mesh_gpu_instancing` is placed as a model of its own, in the
//! scene layout named after the node and in its mesh's space, whatever the selection; its
//! instances, scene space from the node's transform times each instance's, become placements.
//! Those of a node that moves hang from its rig's joint, in the joint's bind space, on pages of
//! their own. A skinned one is a model of its own with its skin's skeleton, each instance a copy
//! posed on its own. glTF ignores a skinned mesh's node transform, and the extension says nothing
//! of skins, so a copy's placement is the instance's transform alone.
//!
//! A displacement is a mesh space length along the mesh's normal, as `KHR_materials_volume` has
//! thickness, so a transform moves it with the mesh, placed or applied. Only a static bake holds
//! that of a transform applied to the vertices: at run time such a height keeps its stated length.
//!
//! A placement keeps the whole transform, mirrored or sheared. A transform that collapses an
//! axis is not a placement: the mesh is cooked once per such linear transform with it
//! applied, as model `<model>/baked/<k>`, placed by translation alone; a skinned mesh keeps its
//! bind pose, and the transform goes to a skeleton of that model's own. A node scaled to zero
//! on every axis cooks like any other, as a surface of no area.
//!
//! Rigid parts come from the custom vertex attribute `PartAttribute::name` (integer, or
//! float rounded to the nearest integer; 0 where a primitive lacks it). With skins the
//! values are joints of the primitive's skin, pivots their bind positions and `Part::joint`
//! the skeleton joint; without, pivots come from the `PartAttribute::pivot` attribute, else
//! the model origin. With `GltfSettings::parts_from_skin`, a skin whose vertices each follow
//! one joint cooks as rigid parts instead of a skinned mesh, as a rig without a skin does:
//! part `j` is skeleton joint `j`, pivoted at its bind position, and part 0, which the renderer
//! does not pose, is the still joint.
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

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::par::prelude::*;
use gltf::image::Source as ImageSource;
use gltf::mesh::{Mode, Semantic};
use orr_math::{DMat3, DMat4, Mat3, Quat, Vec3};
use orr_pack::Hash32;
use orr_pack::material::slot;
use serde::Deserialize;

use crate::import::{
    Placed, PlacedModel, Scene, SceneMaterial, SceneModel, SceneSkeleton, normal_matrix, placements,
};
use crate::mesh::{
    self, LazyMesh, MeshData, MeshSettings, MeshSource, PartData, Pivots, SectionData, Skin,
    tangents,
};
use crate::patch::{self, MaterialPatch, MeshPatch};
use crate::skin::{self, Graph, Rig, Rigging, Skeleton, influences};
use crate::state::{Input, Memo};
use crate::texture::material::{ImageData, ImageFile, Layer};
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

/// How a primitive's vertices are skinned.
enum Binding<'a> {
    /// A skin: skeleton joint per skin joint.
    Skin(&'a [u16]),
    /// Rigidly on one joint.
    Rigid(u16),
}

/// One model being assembled: the primitives read for it, in order, joined when it is done so
/// each stream is built once at its size.
struct Building {
    /// The node that frames it; without one, its space is the scene's.
    frame: Option<usize>,
    /// The model carries tangents, on every primitive: one states them or its material reads
    /// them, and a prototype's material can read them on the others.
    tangents: bool,
    parts: Vec<Part>,
    /// Vertices of `parts`.
    vertices: usize,
    /// Bind position and skeleton joint per part value, from skins.
    joint_pivots: Vec<Option<([f32; 3], u16)>>,
}

impl Building {
    fn new(frame: Option<usize>, parts: usize) -> Self {
        Self {
            frame,
            tangents: false,
            parts: Vec::new(),
            vertices: 0,
            joint_pivots: vec![None; parts],
        }
    }

    /// Whether the model has no triangles.
    fn is_empty(&self) -> bool {
        self.parts.iter().all(|p| p.indices.is_empty())
    }

    /// The part value of each vertex.
    fn values(&self) -> Vec<u16> {
        self.parts
            .iter()
            .flat_map(|p| p.values.iter().copied())
            .collect()
    }

    /// The model's mesh without sections, and the glTF material and index range of each
    /// non-empty triangle group, in material order.
    fn finish(
        mut self,
        parts: Option<&PartAttribute>,
        skinned: bool,
    ) -> (MeshData, Vec<(usize, Range<usize>)>) {
        let ps = &mut self.parts;
        // A stable sort keeps each material's primitives in order.
        let mut order: Vec<usize> = (0..ps.len()).collect();
        order.sort_by_key(|&i| ps[i].material);
        let mut indices = Vec::with_capacity(ps.iter().map(|p| p.indices.len()).sum());
        let mut groups: Vec<(usize, Range<usize>)> = Vec::new();
        for i in order {
            let p = &mut ps[i];
            if p.indices.is_empty() {
                continue;
            }
            let start = indices.len();
            indices.extend(std::mem::take(&mut p.indices));
            match groups.last_mut() {
                Some((m, range)) if *m == p.material => range.end = indices.len(),
                _ => groups.push((p.material, start..indices.len())),
            }
        }
        let counts: Vec<usize> = ps.iter().map(|p| p.positions.len()).collect();
        let uv1 = ps.iter().any(|p| p.uvs1.is_some());
        let colors = ps.iter().any(|p| p.colors.is_some());
        let moves = ps.iter().any(|p| p.moves.is_some());
        let jp = &self.joint_pivots;
        let mesh_parts = parts.map(|pa| PartData {
            values: joined(ps, |p| &mut p.values),
            count: pa.parts,
            by_section: pa.by_section,
            pivots: match (skinned, &pa.pivot) {
                (true, _) => {
                    Pivots::PerValue(jp.iter().map(|p| p.map_or([0.0; 3], |p| p.0)).collect())
                }
                (false, Some(_)) => Pivots::PerVertex(joined(ps, |p| &mut p.pivots)),
                (false, None) => Pivots::Origin,
            },
            joints: skinned.then(|| jp.iter().map(|p| p.map_or(0, |p| p.1)).collect()),
        });
        let mesh = MeshData {
            positions: joined(ps, |p| &mut p.positions),
            normals: joined(ps, |p| &mut p.normals),
            tangents: self
                .tangents
                .then(|| joined_or(ps, &counts, |p| &mut p.tangents, tangents::UNREAD)),
            uvs: std::iter::once(joined(ps, |p| &mut p.uvs))
                .chain(uv1.then(|| joined_or(ps, &counts, |p| &mut p.uvs1, [0.0; 2])))
                .collect(),
            colors: colors.then(|| joined_or(ps, &counts, |p| &mut p.colors, [1.0; 4])),
            moves: moves.then(|| joined_or(ps, &counts, |p| &mut p.moves, [[0.0; 3]; 2])),
            indices,
            sections: Vec::new(),
            skin: skinned.then(|| Skin {
                joints: joined(ps, |p| &mut p.joints),
                weights: joined(ps, |p| &mut p.weights),
            }),
            parts: mesh_parts,
            min_exponent: None,
        };
        (mesh, groups)
    }
}

/// One stream of `parts` joined, taken from them.
fn joined<T>(parts: &mut [Part], stream: impl Fn(&mut Part) -> &mut Vec<T>) -> Vec<T> {
    if let [p] = parts {
        return std::mem::take(stream(p));
    }
    let mut out = Vec::with_capacity(parts.iter_mut().map(|p| stream(p).len()).sum());
    for p in parts {
        out.extend(std::mem::take(stream(p)));
    }
    out
}

/// [`joined`] for a stream parts may lack, with `fill` for the `counts` vertices of each part
/// without it.
fn joined_or<T: Clone>(
    parts: &mut [Part],
    counts: &[usize],
    stream: impl Fn(&mut Part) -> &mut Option<Vec<T>>,
    fill: T,
) -> Vec<T> {
    if let [p] = parts
        && let Some(v) = stream(p).take()
    {
        return v;
    }
    let mut out = Vec::with_capacity(counts.iter().sum());
    for (p, &n) in parts.iter_mut().zip(counts) {
        match stream(p).take() {
            Some(v) => out.extend(v),
            None => out.resize(out.len() + n, fill.clone()),
        }
    }
    out
}

/// The glTF material index of a primitive; `usize::MAX` for the default material.
fn material_index(prim: &gltf::Primitive<'_>) -> usize {
    prim.material().index().unwrap_or(usize::MAX)
}

/// Whether `prim` has a surface to cook.
fn has_surface(prim: &gltf::Primitive<'_>) -> bool {
    let triangles = matches!(
        prim.mode(),
        Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan
    );
    triangles && prim.get(&Semantic::Positions).is_some()
}

/// The primitives of `mesh` with a surface to cook.
fn surfaces<'a>(mesh: &gltf::Mesh<'a>) -> impl Iterator<Item = gltf::Primitive<'a>> {
    mesh.primitives().filter(has_surface)
}

/// The mesh of `node`, if it has a surface to cook.
fn surface_mesh<'a>(node: &gltf::Node<'a>) -> Option<gltf::Mesh<'a>> {
    node.mesh().filter(|m| surfaces(m).next().is_some())
}

/// The corners of the triangles of vertices `v` connected as `mode`, in glTF's order.
fn corners(mode: Mode, v: Vec<u32>) -> Vec<u32> {
    let triangles = 0..v.len().saturating_sub(2);
    match mode {
        Mode::TriangleStrip => triangles
            .flat_map(|i| [v[i], v[i + 1 + i % 2], v[i + 2 - i % 2]])
            .collect(),
        Mode::TriangleFan => triangles.flat_map(|i| [v[i + 1], v[i + 2], v[0]]).collect(),
        _ => v,
    }
}

/// A warning for each primitive without a surface of the meshes nodes `placed` hold.
fn skipped_primitives(doc: &gltf::Document, placed: impl Iterator<Item = usize>) -> Vec<String> {
    let meshes = placed.filter_map(|n| doc.nodes().nth(n)?.mesh().map(|m| m.index()));
    let mut warnings = Vec::new();
    for mesh in meshes.collect::<BTreeSet<_>>() {
        let mesh = doc.meshes().nth(mesh).expect("mesh index");
        for p in mesh.primitives().filter(|p| !has_surface(p)) {
            let why = match p.get(&Semantic::Positions) {
                Some(_) => format!("mode {:?}", p.mode()),
                None => "no POSITION".to_string(),
            };
            warnings.push(format!(
                "mesh {} primitive {}: {why} has no surface, skipped",
                mesh.index(),
                p.index()
            ));
        }
    }
    warnings
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

/// How one primitive lands in a model.
struct Placing<'a> {
    /// Its glTF mesh.
    mesh: usize,
    /// Model space from the primitive's.
    world: DMat4,
    /// Of its morph targets.
    weights: &'a [f32],
    binding: Option<&'a Binding<'a>>,
    /// [`SceneMaterial::tangent_set`] of its material; set 0 where that reads none and the
    /// model carries tangents.
    tangent_set: Option<usize>,
    /// Its material displaces it in a space `world` has already moved it to.
    displaced: bool,
}

/// A primitive attribute as rows of `K` floats with the weighted morph target deltas added,
/// `None` when the primitive lacks it; every one holds `n` rows. L1991: a tangent's deltas
/// leave its w. L1926 gives joints and their weights no deltas.
fn morphed_rows<const K: usize>(
    src: &Data<'_>,
    prim: &gltf::Primitive<'_>,
    at: &Placing<'_>,
    n: usize,
    s: Semantic,
) -> Result<Option<Vec<[f32; K]>>, String> {
    let counted = |a: &gltf::Accessor<'_>| {
        if a.count() != n {
            return Err("attribute counts differ".to_string());
        }
        Ok(())
    };
    let Some(a) = prim.get(&s) else {
        return Ok(None);
    };
    counted(&a)?;
    let mut out = src.rows::<K>(&a)?;
    if matches!(s, Semantic::Joints(_) | Semantic::Weights(_)) {
        return Ok(Some(out));
    }
    let width = if s == Semantic::Tangents { 3 } else { K };
    let name = s.to_string();
    for (t, &w) in at.weights.iter().enumerate().filter(|&(_, &w)| w != 0.0) {
        let Some(d) = src.doc.morph_deltas(at.mesh, prim.index(), t, &name) else {
            continue;
        };
        counted(&d)?;
        let d = src.floats(&d, width)?;
        let components = out.iter_mut().flat_map(|r| &mut r[..width]);
        for (x, d) in components.zip(&d) {
            *x += w * d;
        }
    }
    Ok(Some(out))
}

/// The world transform's linear part, and whether it mirrors. As `normal_matrix` takes it:
/// where `world` flattens, the sign is rounding, and the winding and the normals have to
/// agree on it.
fn placing_linear(world: DMat4) -> (DMat3, bool) {
    let linear = DMat3::from_mat4(world);
    (linear, linear.determinant() < 0.0)
}

/// MikkTSpace's tangents of a primitive per corner of `indices`, in its own space from its
/// morphed `local` positions, normals (flat ones without) and uvs of set `set`, then placed by
/// `linear` and with their sign flipped where it mirrors.
fn corner_tangents(
    indices: &[u32],
    local: &[[f32; 3]],
    local_normals: Option<&[[f32; 3]]>,
    uv: &[[f32; 2]],
    linear: Mat3,
    flip: bool,
) -> Vec<[f32; 4]> {
    let place_tangent = |[x, y, z, w]: [f32; 4]| {
        let t = (linear * Vec3::new(x, y, z)).to_array();
        [t[0], t[1], t[2], if flip { -w } else { w }]
    };
    let corners = match local_normals {
        Some(n) => {
            let unit: Vec<[f32; 3]> = n
                .par_iter()
                .light()
                .map(|&n| Vec3::from(n).normalize_or_zero().to_array())
                .collect();
            tangents::generate(Some(indices), local, &unit, uv)
        }
        None => tangents::generate_flat(indices, local, uv),
    };
    corners.into_iter().map(place_tangent).collect()
}

/// The uv set of the tangents [`read_primitive`] computes for `prim`, if it computes them:
/// when the model carries tangents (`carries`) and the primitive states none. glTF ignores the
/// tangents of a primitive without normals.
fn tangents_set(prim: &gltf::Primitive<'_>, at: &Placing<'_>, carries: bool) -> Option<usize> {
    let stated = prim.get(&Semantic::Normals).is_some() && prim.get(&Semantic::Tangents).is_some();
    at.tangent_set
        .or_else(|| carries.then_some(0))
        .filter(|_| !stated)
}

/// A primitive's vertices and triangles as its model takes them ([`Building::append`]). The
/// streams a primitive may lack are `None`, and the model fills them as it does without them.
struct Part {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Option<Vec<[f32; 4]>>,
    uvs: Vec<[f32; 2]>,
    uvs1: Option<Vec<[f32; 2]>>,
    colors: Option<Vec<[f32; 4]>>,
    moves: Option<Vec<[[f32; 3]; 2]>>,
    /// Empty without a binding.
    joints: Vec<[u8; 4]>,
    weights: Vec<[u16; 4]>,
    /// Empty without parts from an attribute or a skin.
    values: Vec<u16>,
    /// Empty without pivots from an attribute.
    pivots: Vec<[f32; 3]>,
    /// Triangles over the part's vertices, and over the model's once appended.
    indices: Vec<u32>,
    material: usize,
}

/// Reads a triangle primitive with its morph targets at the weights, its positions taken to
/// model space by the world transform (winding and tangent signs flipped under a mirroring
/// one), for a model that carries tangents or not (`carries`). Tangents are the primitive's
/// own, or MikkTSpace's where the model carries them, computed from its morphed attributes in
/// its own space as the glTF spec asks. It reads on its own, so primitives read in parallel;
/// what the generator reads comes first, and the rest once its corners are gone.
fn read_primitive(
    src: &Data<'_>,
    prim: &gltf::Primitive<'_>,
    at: &Placing<'_>,
    cx: &Context<'_>,
    carries: bool,
) -> Result<Part, String> {
    let (world, weights, binding) = (at.world, at.weights, at.binding);
    let (linear, flip) = placing_linear(world);
    let normal_m = normal_matrix(linear).as_mat3();
    let linear = linear.as_mat3();
    let place = |p: [f32; 3]| {
        let p = world.transform_point3(Vec3::from(p).as_dvec3());
        p.as_vec3().to_array()
    };
    let position = prim.get(&Semantic::Positions).ok_or("no positions")?;
    let n = position.count();
    let custom = |name: &str| {
        name.strip_prefix('_')
            .map(|key| Semantic::Extras(key.into()))
            .ok_or_else(|| format!("{name}: custom vertex attributes start with _"))
    };
    let mut positions: Vec<[f32; 3]> =
        morphed_rows(src, prim, at, n, Semantic::Positions)?.expect("checked above");
    let vertices: Vec<u32> = match prim.indices() {
        Some(a) => src.indices(&a)?,
        None => (0..n as u32).collect(),
    };
    let mut indices = corners(prim.mode(), vertices);
    if !indices.len().is_multiple_of(3) || indices.iter().any(|&i| i as usize >= n) {
        return Err("bad index list".into());
    }
    let zeros = |v: Option<Vec<[f32; 2]>>| v.unwrap_or_else(|| vec![[0.0; 2]; n]);
    let set = tangents_set(prim, at, carries);
    let mut uv_sets: [Option<Option<Vec<[f32; 2]>>>; 2] = [None, None];
    if let Some(s) = set {
        let s = usize::from(s == 1);
        uv_sets[s] = Some(morphed_rows(
            src,
            prim,
            at,
            n,
            Semantic::TexCoords(s as u32),
        )?);
    }
    let mut normals: Option<Vec<[f32; 3]>> = morphed_rows(src, prim, at, n, Semantic::Normals)?;
    let place_tangent = |[x, y, z, w]: [f32; 4]| {
        let t = (linear * Vec3::new(x, y, z)).to_array();
        [t[0], t[1], t[2], if flip { -w } else { w }]
    };
    // glTF ignores the tangents of a primitive without normals.
    let stated: Option<Vec<[f32; 4]>> = match normals {
        Some(_) => morphed_rows(src, prim, at, n, Semantic::Tangents)?.map(|mut t| {
            t.iter_mut().for_each(|t| *t = place_tangent(*t));
            t
        }),
        None => None,
    };
    let mut computed: Option<Vec<[f32; 4]>> = match set {
        Some(s) => {
            let unread;
            let uv = match &uv_sets[usize::from(s == 1)] {
                Some(Some(uv)) => uv.as_slice(),
                _ => {
                    unread = vec![[0.0; 2]; n];
                    unread.as_slice()
                }
            };
            Some(corner_tangents(
                &indices,
                &positions,
                normals.as_deref(),
                uv,
                linear,
                flip,
            ))
        }
        None => None,
    };
    if flip {
        indices
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .for_each(|t| t.swap(1, 2));
        if let Some(c) = &mut computed {
            c.as_chunks_mut::<3>()
                .0
                .iter_mut()
                .for_each(|t| t.swap(1, 2));
        }
    }
    let moved = |n: Vec3| {
        let unit = n.normalize_or_zero();
        [unit.to_array(), (linear * unit).to_array()]
    };
    // Per vertex: tangents and the copies corners of differing tangents move to; without
    // normals, the corners' own tangents and the moves of their faces.
    let (tangents, copies, face_moves) = match &normals {
        Some(_) => {
            let (tangents, copies) = match computed.take() {
                Some(c) => {
                    let (t, copies) = tangents::per_vertex(n, &mut indices, &c);
                    (Some(t), copies)
                }
                None => (stated, Vec::new()),
            };
            (tangents, copies, None)
        }
        None => {
            // `t` winds as the placed triangle does, against the mesh's under a mirror.
            let moves = at.displaced.then(|| {
                let faces = indices.as_chunks::<3>().0.iter();
                faces
                    .map(|t| {
                        let n = tangents::face_normal(t.map(|i| Vec3::from(positions[i as usize])));
                        moved(if flip { -n } else { n })
                    })
                    .collect::<Vec<_>>()
            });
            (computed.take(), Vec::new(), moves)
        }
    };
    let moves = match &normals {
        Some(ns) if at.displaced => Some(ns.iter().map(|&n| moved(Vec3::from(n))).collect()),
        _ => None,
    };
    positions
        .par_iter_mut()
        .light()
        .for_each(|p| *p = place(*p));
    if let Some(ns) = &mut normals {
        ns.par_iter_mut()
            .light()
            .for_each(|nv| *nv = (normal_m * Vec3::from(*nv)).normalize_or_zero().to_array());
    }
    for (s, uv) in (0..).zip(&mut uv_sets) {
        if uv.is_none() {
            *uv = Some(morphed_rows(src, prim, at, n, Semantic::TexCoords(s))?);
        }
    }
    let [uvs, uvs1] = uv_sets.map(|s| s.expect("read above"));
    let uvs = zeros(uvs);
    // L1707 keeps them in 0..1, L2008 keeps them there once morphed, and a primitive without any
    // multiplies by nothing.
    let colors: Option<Vec<[f32; 4]>> =
        morphed_rows(src, prim, at, n, Semantic::Colors(0))?.map(|mut colors| {
            let opaque = prim
                .get(&Semantic::Colors(0))
                .is_some_and(|a| a.dimensions() == gltf::accessor::Dimensions::Vec3);
            if opaque {
                colors.iter_mut().for_each(|r| r[3] += 1.0);
            }
            if weights.iter().any(|&w| w != 0.0) {
                colors
                    .as_flattened_mut()
                    .iter_mut()
                    .for_each(|c| *c = c.clamp(0.0, 1.0));
            }
            colors
        });
    let skin: Vec<([u16; 4], [u16; 4])> = match binding {
        None => Vec::new(),
        Some(Binding::Rigid(j)) => vec![([*j; 4], [65535, 0, 0, 0]); n],
        Some(Binding::Skin(map)) => {
            let mut sets = Vec::new();
            for set in 0.. {
                match (
                    morphed_rows::<4>(src, prim, at, n, Semantic::Joints(set))?,
                    morphed_rows::<4>(src, prim, at, n, Semantic::Weights(set))?,
                ) {
                    (None, None) => break,
                    (Some(j), Some(w)) => {
                        let j: Vec<[u16; 4]> = j.iter().map(|v| v.map(|x| x as u16)).collect();
                        sets.push((j, w));
                    }
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
                    [0, 0, 0] => Ok(j[0]),
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
            let values: Vec<u16> = match morphed_rows::<1>(src, prim, at, n, custom(&pa.name)?)? {
                Some(v) => v
                    .iter()
                    .map(|&[x]| {
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
                (Some(name), None) => morphed_rows::<3>(src, prim, at, n, custom(name)?)?
                    .map_or_else(
                        || vec![[0.0; 3]; n],
                        |mut p| {
                            p.iter_mut().for_each(|q| *q = place(*q));
                            p
                        },
                    ),
                _ => Vec::new(),
            };
            (values, pivots)
        }
    };
    let (joints, weights): (Vec<[u8; 4]>, Vec<[u16; 4]>) =
        skin.iter().map(|s| (s.0.map(|j| j as u8), s.1)).unzip();
    let material = material_index(prim);
    let Some(normals) = normals else {
        // Without normals, glTF asks for flat shading: one vertex per corner.
        let positions_c = per_corner(&positions, &indices);
        let normals = positions_c
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|t| [tangents::face_normal(t.map(Vec3::from)).to_array(); 3])
            .collect();
        return Ok(Part {
            normals,
            tangents,
            uvs: per_corner(&uvs, &indices),
            uvs1: uvs1.map(|u| per_corner(&u, &indices)),
            colors: colors.map(|c| per_corner(&c, &indices)),
            moves: face_moves.map(|m| m.iter().flat_map(|&m| [m; 3]).collect()),
            joints: per_corner(&joints, &indices),
            weights: per_corner(&weights, &indices),
            values: per_corner(&values, &indices),
            pivots: per_corner(&pivots, &indices),
            indices: (0..positions_c.len() as u32).collect(),
            positions: positions_c,
            material,
        });
    };
    let c = &copies;
    Ok(Part {
        positions: tangents::with_copies(positions, c),
        normals: tangents::with_copies(normals, c),
        tangents,
        uvs: tangents::with_copies(uvs, c),
        uvs1: uvs1.map(|u| tangents::with_copies(u, c)),
        colors: colors.map(|v| tangents::with_copies(v, c)),
        moves: moves.map(|m| tangents::with_copies(m, c)),
        joints: tangents::with_copies(joints, c),
        weights: tangents::with_copies(weights, c),
        values: tangents::with_copies(values, c),
        pivots: tangents::with_copies(pivots, c),
        indices,
        material,
    })
}

/// Appends `prims` to `b` in order. With `parallel`, the primitives that compute tangents read
/// first, in parallel, and the rest as they are appended. Fails with the first error in
/// primitive order, and the primitive's index.
fn append_all(
    src: &Data<'_>,
    prims: &[(gltf::Primitive<'_>, Placing<'_>)],
    cx: &Context<'_>,
    b: &mut Building,
    parallel: bool,
) -> Result<(), (usize, String)> {
    let carries = b.tangents;
    let read = |(prim, at): &(gltf::Primitive<'_>, Placing<'_>)| {
        read_primitive(src, prim, at, cx, carries)
    };
    let early: Vec<Option<Result<Part, String>>> = if parallel {
        prims
            .par_iter()
            .map(|p| tangents_set(&p.0, &p.1, carries).is_some().then(|| read(p)))
            .collect()
    } else {
        prims.iter().map(|_| None).collect()
    };
    for (p, part) in prims.iter().zip(early) {
        part.unwrap_or_else(|| read(p))
            .and_then(|part| b.append(part, &p.1, cx))
            .map_err(|m| (p.0.index(), m))?;
    }
    Ok(())
}

/// Per corner of `indices`, its vertex's element of `v`, unless `v` is empty.
fn per_corner<T: Copy>(v: &[T], indices: &[u32]) -> Vec<T> {
    if v.is_empty() {
        return Vec::new();
    }
    indices.iter().map(|&i| v[i as usize]).collect()
}

impl Building {
    /// Appends a primitive read for this model, placed as `at`.
    fn append(&mut self, mut part: Part, at: &Placing<'_>, cx: &Context<'_>) -> Result<(), String> {
        if let (Some(pa), Some(sk), Some(binding), false) =
            (cx.parts, cx.skeleton, at.binding, cx.rigid)
        {
            for &v in &part.values {
                if v >= pa.parts {
                    continue;
                }
                let joint = match binding {
                    Binding::Skin(map) => *map
                        .get(usize::from(v))
                        .ok_or_else(|| format!("part value {v} is not a joint of the skin"))?,
                    Binding::Rigid(j) => *j,
                };
                let slot = &mut self.joint_pivots[usize::from(v)];
                if slot.is_some_and(|s| s.1 != joint) {
                    return Err(format!("part value {v} names two skeleton joints"));
                }
                *slot = Some((sk.bind_origin(joint), joint));
            }
        }
        let base = self.vertices as u32;
        part.indices.iter_mut().for_each(|i| *i += base);
        self.vertices += part.positions.len();
        self.parts.push(part);
        Ok(())
    }
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
                        layer: Layer::Whole,
                        path: data::uri_path(dir, uri),
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
    // What a constant displacement reads, past the document's images.
    out.push(ImageFile {
        name: "constant displacement".into(),
        data: ImageData::Solid([0.0; 4]),
    });
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
    let dir = path.parent().unwrap_or_else(|| Path::new("")).to_owned();
    let src = data::open(&bytes, &dir, settings.ignore_version)
        .map_err(|m| fail(format!("{}: {m}", path.display())))?;
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
        skeletons: Vec::new(),
        instances: Vec::new(),
        files: &mut files,
        warnings: Vec::new(),
    };
    node_models(&mut b, &graph)?;
    let Imported {
        materials,
        mut models,
        skeletons,
        instances,
        warnings,
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
        skeletons,
        clips: settings.clips.clone(),
        instances,
        files,
        warnings,
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
    skeletons: Vec<SceneSkeleton>,
    instances: Vec<PlacedModel>,
    files: &'a mut Vec<Input>,
    warnings: Vec<String>,
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

    /// Registers the material of `prim` of glTF mesh `mesh`; the primitive must hold the uv
    /// sets it reads.
    fn material(&mut self, mesh: usize, prim: &gltf::Primitive<'_>) -> Result<()> {
        let index = material_index(prim);
        if !self.materials.contains_key(&index) {
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
                    back: None,
                    suffix: match index {
                        usize::MAX => "default".to_string(),
                        i => i.to_string(),
                    },
                    material: m,
                    textures: t,
                },
            );
        }
        // L2666: an extra multiplier for the base color, alpha included by the example below it.
        if prim.get(&Semantic::Colors(0)).is_some() {
            let m = &mut self
                .materials
                .get_mut(&index)
                .expect("registered above")
                .material;
            m.textures[slot::BASE_COLOR].vertex = 1;
            m.textures[slot::OPACITY].vertex = 4;
        }
        let reads_uv1 = self.materials[&index]
            .textures
            .refs()
            .any(|r| r.uv_set == 1);
        // L2628 "**MUST** define" them and gives no reading; TEXCOORD_0 reads zero alike.
        if reads_uv1 && prim.get(&Semantic::TexCoords(1)).is_none() {
            let warning = format!(
                "mesh {mesh} primitive {}: its material reads TEXCOORD_1, which the primitive does not have; it reads uv 0",
                prim.index()
            );
            // Every node placing the mesh comes through here.
            if !self.warnings.contains(&warning) {
                self.warnings.push(warning);
            }
        }
        Ok(())
    }

    /// [`SceneMaterial::tangent_set`] of `prim`'s registered material.
    fn tangent_set(&self, prim: &gltf::Primitive<'_>) -> Option<usize> {
        self.materials[&material_index(prim)].tangent_set()
    }

    /// Whether `prim`'s registered material displaces it.
    fn displaces(&self, prim: &gltf::Primitive<'_>) -> bool {
        self.materials[&material_index(prim)]
            .textures
            .displacement
            .is_some()
    }

    /// Whether `prim` makes its model carry tangents. glTF ignores those of a primitive without
    /// normals.
    fn carries_tangents(&self, prim: &gltf::Primitive<'_>) -> bool {
        let stated = [Semantic::Normals, Semantic::Tangents];
        self.tangent_set(prim).is_some() || stated.iter().all(|s| prim.get(s).is_some())
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
    /// linear transform. They hang from `frame`, in its joint's bind space.
    fn placed(
        &mut self,
        name: &str,
        mesh: usize,
        weights: &[f32],
        transforms: &[DMat4],
        frame: Option<(usize, u16)>,
    ) -> Result<()> {
        let (direct, baked) =
            placements(transforms, false).map_err(|m| self.fail(format!("model {name}: {m}")))?;
        let hung = frame.is_some();
        if !direct.is_empty() {
            self.lazy_model(name, mesh, weights, DMat4::IDENTITY, hung)?;
            self.instances.push(PlacedModel {
                model: name.to_string(),
                placed: direct,
                frame,
                alone: frame.is_some(),
                motion: None,
            });
        }
        for (k, (linear, placed)) in baked.into_iter().enumerate() {
            let name = format!("{name}/baked/{k}");
            self.lazy_model(&name, mesh, weights, linear, hung)?;
            self.instances.push(PlacedModel {
                model: name,
                placed,
                frame,
                alone: frame.is_some(),
                motion: None,
            });
        }
        Ok(())
    }

    /// Model `name`: glTF mesh `mesh` at morph `weights` with `linear` applied, read when
    /// it cooks, or now when it is `hung` from a joint, whose reach it sets.
    fn lazy_model(
        &mut self,
        name: &str,
        mesh: usize,
        weights: &[f32],
        linear: DMat4,
        hung: bool,
    ) -> Result<()> {
        self.name(name)?;
        let src = Arc::clone(self.src);
        let m = src.doc.meshes().nth(mesh).expect("mesh index");
        let mut views = Vec::new();
        let mut groups: BTreeMap<usize, u64> = BTreeMap::new();
        let mut vertex_count = 0;
        let mut tangent_sets = Vec::new();
        let mut displaced = Vec::new();
        for prim in surfaces(&m) {
            self.material(mesh, &prim)?;
            tangent_sets.push(self.tangent_set(&prim));
            displaced.push(linear != DMat4::IDENTITY && self.displaces(&prim));
            primitive_views(&prim, &mut views);
            for a in src.morph_accessors(mesh, prim.index()) {
                Document::accessor_views(&a, &mut views);
            }
            let positions = prim.get(&Semantic::Positions).expect("a surface");
            let vertices = prim
                .indices()
                .map_or_else(|| positions.count(), |a| a.count());
            let count = match prim.mode() {
                Mode::Triangles => vertices,
                _ => 3 * vertices.saturating_sub(2),
            };
            if !count.is_multiple_of(3) {
                let what = format!("mesh {mesh} primitive {}", prim.index());
                return Err(self.fail(format!("{what}: bad index list")));
            }
            // Without normals, a primitive takes a vertex per corner.
            vertex_count += match prim.get(&Semantic::Normals) {
                Some(_) => positions.count(),
                None => count,
            } as u64;
            *groups.entry(material_index(&prim)).or_default() += count as u64;
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
                None,
                Default::default(),
            ));
            materials.push(index);
            at = end;
        }
        if sections.is_empty() {
            return Err(self.fail(format!("model {name} has no triangles")));
        }
        // Everything the loaded mesh is made of: the mesh's JSON, its accessors and views,
        // the bytes of the buffers they read, the transform, the part attribute, whether the
        // model carries tangents and the uv sets they follow, and what the transform displaces.
        let mut h = blake3::Hasher::new();
        let mut json = |v: &serde_json::Value| {
            let s = serde_json::to_string(v).expect("JSON serializes");
            h.update(&(s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        };
        json(&src.json["meshes"][mesh]);
        for prim in surfaces(&m) {
            let attributes = prim.attributes().map(|a| a.1);
            for a in attributes
                .chain(prim.indices())
                .chain(src.morph_accessors(mesh, prim.index()))
            {
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
        let tangents = surfaces(&m).any(|p| self.carries_tangents(&p));
        h.update(format!("{tangents} {tangent_sets:?} {displaced:?}").as_bytes());
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
            let mut b = Building::new(None, count);
            b.tangents = tangents;
            let gm = src.doc.meshes().nth(mesh).expect("mesh index");
            let placings = tangent_sets.iter().zip(&displaced);
            let prims: Vec<_> = surfaces(&gm)
                .zip(placings)
                .map(|(prim, (&tangent_set, &displaced))| {
                    let at = Placing {
                        mesh,
                        world: linear,
                        weights: &weights,
                        binding: None,
                        tangent_set,
                        displaced,
                    };
                    (prim, at)
                })
                .collect();
            let computing = prims
                .iter()
                .filter(|(prim, at)| tangents_set(prim, at, tangents).is_some())
                .count();
            append_all(&data, &prims, &cx, &mut b, computing > 1).map_err(|(p, m)| {
                Error::Import {
                    asset: asset.clone(),
                    message: format!("mesh {mesh} primitive {p}: {m}"),
                }
            })?;
            Ok(b.finish(parts.as_ref(), false).0)
        };
        let lazy = LazyMesh {
            sections,
            triangles: at as u64 / 3,
            vertices: vertex_count,
            source: Hash32(*h.finalize().as_bytes()),
            load: Arc::new(load),
        };
        let settings = self.mesh_settings(name);
        // Assemblies split the mesh when it is planned, so such a model is read now.
        let mesh = if settings.assemblies.is_some() || hung {
            MeshSource::Data(Box::new(
                MeshSource::Lazy(lazy).read(self.asset, self.memo)?,
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

/// Node `n`'s `EXT_mesh_gpu_instancing` object, if it has one and a mesh with a surface.
fn instancing(src: &Document, n: usize) -> Option<&serde_json::Value> {
    surface_mesh(&src.doc.nodes().nth(n)?)?;
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

/// The world transforms of `n`'s copies: its world transform times each of [`instance_locals`].
fn node_transforms(src: &Data<'_>, graph: &Graph, n: usize) -> Result<Vec<DMat4>, String> {
    let locals = instance_locals(src, n)?;
    Ok(locals.into_iter().map(|l| graph.world[n] * l).collect())
}

/// Each `EXT_mesh_gpu_instancing` instance's transform of node `n`, the identity alone without
/// the extension.
fn instance_locals(src: &Data<'_>, n: usize) -> Result<Vec<DMat4>, String> {
    let Some(ext) = instancing(src.doc, n) else {
        return Ok(vec![DMat4::IDENTITY]);
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
            let r = r
                .as_ref()
                .map_or(Quat::IDENTITY, |v| Quat::from_slice(&v[4 * i..]));
            let s = sc
                .as_ref()
                .map_or(Vec3::ONE, |v| Vec3::from_slice(&v[3 * i..]));
            DMat4::from_scale_rotation_translation(
                s.as_dvec3(),
                r.as_dquat().normalize(),
                t.as_dvec3(),
            )
        })
        .collect())
}

/// Loads the instancing accessors and records the files they read.
fn instancing_data<'d>(b: &mut Imported<'_>, src: &'d Document, graph: &Graph) -> Result<Data<'d>> {
    let mut views = instancing_views(src, graph);
    b.record(&mut views)?;
    Data::load(src, views, b.asset)
}

/// Every glTF mesh that nodes `placed` hold is a model, every node and instance holding it a
/// placement.
fn meshes_layout(b: &mut Imported<'_>, graph: &Graph, placed: &[usize]) -> Result<()> {
    let src = Arc::clone(b.src);
    let doc = &src.doc;
    // Every placed mesh's buffer files, hashed together.
    let shown = b.shown.clone();
    b.warnings
        .extend(skipped_primitives(doc, shown.iter().copied()));
    let mut views = Vec::new();
    for &n in placed {
        if let Some(mesh) = doc.nodes().nth(n).and_then(|d| d.mesh()) {
            surfaces(&mesh).for_each(|p| primitive_views(&p, &mut views));
        }
    }
    b.record(&mut views)?;
    let data = instancing_data(b, &src, graph)?;
    // Per mesh, its placements per set of morph weights, the mesh's own first.
    type Placings = Vec<(Vec<f32>, Vec<DMat4>)>;
    let mut by_mesh: BTreeMap<usize, Placings> = BTreeMap::new();
    for &n in placed {
        let node = doc.nodes().nth(n).expect("node index");
        let Some(mesh) = surface_mesh(&node) else {
            continue;
        };
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
                b.placed(&name, mesh, weights, transforms, None)?;
            }
        }
    }
    Ok(())
}

/// A model built from nodes, beside its [`Building`].
struct Slot {
    /// `None` for the whole scene.
    name: Option<String>,
    rig: Option<Rig>,
    /// What its skeleton applies after model space ([`Skeleton::build`]).
    after: DMat4,
    /// Its instances' placements, if it is placed by them.
    copies: Option<Vec<Placed>>,
}

impl Slot {
    fn new(name: Option<String>, rig: Option<Rig>) -> Self {
        Self {
            name,
            rig,
            after: DMat4::IDENTITY,
            copies: None,
        }
    }
}

/// The models of both layouts: what moves in rigs, and what stands still as the layout says.
fn node_models(b: &mut Imported<'_>, graph: &Graph) -> Result<()> {
    let (asset, settings) = (b.asset, b.settings);
    let src = Arc::clone(b.src);
    let doc = &src.doc;
    let fail = |message: String| Error::Import {
        asset: asset.into(),
        message,
    };
    let selecting = !settings.models.is_empty();
    let by_mesh = settings.layout == GltfLayout::Meshes;
    if by_mesh {
        let needs_scene = [
            (selecting, "model selection"),
            (!settings.joined.is_empty(), "joined models"),
        ];
        if let Some((_, what)) = needs_scene.iter().find(|n| n.0) {
            return Err(b.fail(format!("{what} need the scene layout")));
        }
    }
    if settings.parts_from_skin && settings.parts_from_attribute.is_some() {
        return Err(fail(
            "parts_from_skin and parts_from_attribute do not mix".into(),
        ));
    }
    let node = |n: usize| doc.nodes().nth(n).expect("node index");
    let shown = b.shown.clone();
    let surfaced: Vec<usize> = shown
        .iter()
        .copied()
        .filter(|&n| surface_mesh(&node(n)).is_some())
        .collect();
    let instanced: Vec<usize> = surfaced
        .iter()
        .copied()
        .filter(|&n| instancing(&src, n).is_some())
        .collect();
    let mut is_instanced = vec![false; graph.parent.len()];
    instanced.iter().for_each(|&n| is_instanced[n] = true);
    let skinned_copies = |n: usize| is_instanced[n] && node(n).skin().is_some();
    let animated = skin::animated(doc, graph);
    let skin_of = |n: usize| {
        let skin = node(n).skin()?;
        Some((skin.index(), skin.joints().map(|j| j.index()).collect()))
    };
    let rigging = Rigging {
        parent: &graph.parent,
        order: &graph.order,
        in_scene: &graph.in_scene,
        animated: &animated,
        skin_of: &skin_of,
        parts_from_skin: settings.parts_from_skin,
        parts_from_attribute: settings.parts_from_attribute.is_some(),
    };
    let (mut models, mut owner) = if by_mesh {
        (Vec::new(), vec![None; graph.parent.len()])
    } else {
        select_models(doc, graph, &shown, settings).map_err(fail)?
    };
    let placed = shown.iter().copied();
    let skipped = skipped_primitives(
        doc,
        placed.filter(|&n| owner[n].is_some() || instancing(&src, n).is_some()),
    );
    let elsewhere = !instanced.is_empty() || !skipped.is_empty();
    if !by_mesh {
        b.warnings.extend(skipped);
    }
    for &n in &instanced {
        owner[n] = None;
    }
    let mut slots: Vec<Slot> = models
        .iter()
        .map(|m| {
            let name = m.frame.map(|n| node_name(doc, n).to_string());
            Slot::new(name, None)
        })
        .collect();
    if selecting {
        for (m, slot) in slots.iter_mut().enumerate() {
            let held: Vec<usize> = surfaced
                .iter()
                .copied()
                .filter(|&n| owner[n] == Some(m))
                .collect();
            slot.rig = Rig::find(&rigging, &held, models[m].frame, true).pop();
        }
    }
    // What no selected model holds: with selection, only instanced nodes are cooked of that.
    let free: Vec<usize> = surfaced
        .iter()
        .copied()
        .filter(|&n| !skinned_copies(n) && (!selecting || is_instanced[n]))
        .collect();
    let rigs = Rig::find(&rigging, &free, None, false);
    let mut rigged = vec![false; graph.parent.len()];
    rigs.iter()
        .flat_map(|r| &r.meshes)
        .for_each(|&n| rigged[n] = true);
    let still: Vec<usize> = free.iter().copied().filter(|&n| !rigged[n]).collect();

    let inst = instancing_data(b, &src, graph)?;
    if !by_mesh {
        for &n in still.iter().filter(|&&n| is_instanced[n]) {
            let name = instanced_name(doc, n);
            let transforms = node_transforms(&inst, graph, n).map_err(fail)?;
            let at = node(n);
            let mesh = at.mesh().expect("instanced nodes have meshes");
            let weights = at.weights().or_else(|| mesh.weights()).unwrap_or_default();
            b.placed(&name, mesh.index(), weights, &transforms, None)?;
        }
    }
    let part_values = settings
        .parts_from_attribute
        .as_ref()
        .map_or(0, |p| usize::from(p.parts));
    let whole = !selecting && !by_mesh && rigs.len() == 1 && still.iter().all(|&n| is_instanced[n]);
    let mut hung: Vec<(usize, usize)> = Vec::new();
    for rig in rigs {
        let m = if whole {
            0
        } else {
            let top = node_name(doc, rig.top);
            let taken = |s: &Slot| s.name.as_deref() == Some(top);
            let copied = |&n: &usize| skinned_copies(n) && instanced_name(doc, n) == top;
            // `mesh` is the entry of the whole scene's mesh.
            let unusable = top.is_empty() || top == "mesh" || b.names.contains(top);
            let name = if unusable || slots.iter().any(taken) || instanced.iter().any(copied) {
                format!("node{}", rig.top)
            } else {
                top.to_string()
            };
            models.push(Building::new(None, part_values));
            slots.push(Slot::new(Some(name), None));
            models.len() - 1
        };
        for &n in &rig.meshes {
            // An instanced node's copies stay placements: they hang from the rig.
            if is_instanced[n] {
                hung.push((n, m));
            } else {
                owner[n] = Some(m);
            }
        }
        slots[m].rig = Some(rig);
    }
    // A skinned instanced node: per transform an instance holds or bakes, a model of its copies.
    let mut placing: Vec<(usize, usize)> = surfaced
        .iter()
        .filter_map(|&n| Some((n, owner[n]?)))
        .collect();
    for &n in instanced.iter().filter(|&&n| skinned_copies(n)) {
        let name = instanced_name(doc, n);
        let locals = instance_locals(&inst, n).map_err(fail)?;
        let (direct, baked) =
            placements(&locals, false).map_err(|m| fail(format!("model {name}: {m}")))?;
        let baked = baked
            .into_iter()
            .enumerate()
            .map(|(k, (linear, copies))| (format!("{name}/baked/{k}"), linear, copies));
        let direct = (!direct.is_empty()).then(|| (name.clone(), DMat4::IDENTITY, direct));
        for (name, after, copies) in direct.into_iter().chain(baked) {
            let rig = Rig::find(&rigging, &[n], None, true).pop();
            let rig = rig.ok_or_else(|| fail(format!("node {n}: a skin without joints")))?;
            placing.push((n, models.len()));
            models.push(Building::new(None, part_values));
            slots.push(Slot {
                after,
                copies: Some(copies),
                ..Slot::new(Some(name), Some(rig))
            });
        }
    }

    // The views of what cooks here, the skins and the animations; other meshes are read when
    // they cook.
    let mut views = Vec::new();
    for &(n, _) in &placing {
        if let Some(mesh) = node(n).mesh() {
            for p in surfaces(&mesh) {
                primitive_views(&p, &mut views);
                for a in src.morph_accessors(mesh.index(), p.index()) {
                    Document::accessor_views(&a, &mut views);
                }
            }
        }
    }
    let moves = slots.iter().any(|s| s.rig.is_some());
    if moves {
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
    }
    b.record(&mut views)?;
    let data = Data::load(&src, views, asset)?;
    let spans = match moves {
        true => skin::animation_spans(&data).map_err(fail)?,
        false => Vec::new(),
    };
    let build = |s: &Slot| {
        let rig = s.rig.as_ref();
        rig.map(|r| Skeleton::build(&data, graph, r, s.after))
            .transpose()
    };
    let skeletons: Vec<Option<Skeleton>> = slots
        .iter()
        .map(build)
        .collect::<Result<_, _>>()
        .map_err(fail)?;
    let rigid: Vec<bool> = slots
        .iter()
        .map(|s| s.rig.as_ref().is_some_and(|r| r.rigid))
        .collect();

    let parts = settings.parts_from_attribute.as_ref();
    for &(n, m) in &placing {
        let mesh = surface_mesh(&node(n)).expect("a mesh with a surface");
        for prim in surfaces(&mesh) {
            b.material(mesh.index(), &prim)?;
            models[m].tangents |= b.carries_tangents(&prim);
        }
    }
    for &(n, m) in &placing {
        let at = node(n);
        let mesh = surface_mesh(&at).expect("a mesh with a surface");
        let cx = Context {
            skeleton: skeletons[m].as_ref(),
            parts,
            rigid: rigid[m],
        };
        let (world, binding) = match (&skeletons[m], at.skin()) {
            (None, _) => (
                models[m]
                    .frame
                    .map_or(graph.world[n], |model| graph.relative(model, n)),
                None,
            ),
            (Some(sk), Some(s)) => (
                DMat4::IDENTITY,
                Some(Binding::Skin(sk.skin_joints(s.index()))),
            ),
            (Some(sk), None) => {
                let (j, mat) = sk.attach(n, graph).map_err(fail)?;
                (mat, Some(Binding::Rigid(j)))
            }
        };
        let weights = at.weights().or_else(|| mesh.weights()).unwrap_or_default();
        for prim in surfaces(&mesh) {
            let placing = Placing {
                mesh: mesh.index(),
                world,
                weights,
                binding: binding.as_ref(),
                tangent_set: b.tangent_set(&prim),
                displaced: b.displaces(&prim),
            };
            let carries = models[m].tangents;
            read_primitive(&data, &prim, &placing, &cx, carries)
                .and_then(|part| models[m].append(part, &placing, &cx))
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
    let mut skeleton_of: Vec<Option<usize>> = vec![None; slots.len()];
    for (m, (building, slot)) in models.into_iter().zip(slots.iter_mut()).enumerate() {
        let frame = building.frame;
        let empty = building.is_empty();
        let sk = skeletons[m].as_ref();
        let skeleton = match sk {
            Some(sk) => {
                b.skeletons.push(SceneSkeleton {
                    prefix: slot.name.clone().unwrap_or_default(),
                    info: sk.info.clone(),
                    clips: sk.clips(&data, &spans).map_err(fail)?,
                });
                Some(b.skeletons.len() - 1)
            }
            None => None,
        };
        skeleton_of[m] = skeleton;
        // The whole scene may hold nothing but what moves, instanced nodes and primitives
        // without a surface, and a rig nothing but the copies that hang from it.
        if empty && (sk.is_some() || (slot.name.is_none() && (elsewhere || skeletons.len() > 1))) {
            continue;
        }
        let name = slot.name.take();
        if let Some(name) = &name {
            b.name(name)?;
        }
        let parts_of = sk.filter(|_| rigid[m]);
        let joint_values = parts_of.map(|_| building.values());
        let (mut mesh, groups) = building.finish(parts, sk.is_some() && parts_of.is_none());
        if let (Some(sk), Some(values)) = (parts_of, joint_values) {
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
                    None,
                    Default::default(),
                )
            })
            .collect();
        if let (Some(name), Some(copies)) = (&name, slot.copies.take()) {
            b.instances.push(PlacedModel {
                model: name.clone(),
                placed: copies,
                frame: None,
                alone: true,
                motion: None,
            });
        }
        out.push((
            frame,
            SceneModel {
                settings: b.mesh_settings(name.as_deref().unwrap_or_default()),
                node: name,
                subd: None,
                mesh: MeshSource::Data(Box::new(mesh)),
                skeleton,
                materials: groups.into_iter().map(|g| g.0).collect(),
            },
        ));
    }
    join(&mut out, graph, &settings.joined).map_err(fail)?;
    b.models.extend(out.into_iter().map(|m| m.1));
    // Hung copies stand in the bind space of the joint that moves them, so that its skinning
    // matrix takes them from the rest pose to the pose.
    for (n, m) in hung {
        let sk = skeletons[m].as_ref().expect("a rig has a skeleton");
        let (joint, to_bind) = sk.attach(n, graph).map_err(fail)?;
        let locals = instance_locals(&inst, n).map_err(fail)?;
        let transforms: Vec<DMat4> = locals.into_iter().map(|l| to_bind * l).collect();
        let at = node(n);
        let mesh = at.mesh().expect("instanced nodes have meshes");
        let weights = at.weights().or_else(|| mesh.weights()).unwrap_or_default();
        let frame = skeleton_of[m].map(|s| (s, joint));
        // Node names repeat, and a rig may have taken this one.
        let name = instanced_name(doc, n);
        let unusable = name.is_empty() || name == "mesh" || b.names.contains(&name);
        let name = if unusable { format!("node{n}") } else { name };
        b.placed(&name, mesh.index(), weights, &transforms, frame)?;
    }
    if by_mesh {
        meshes_layout(b, graph, &still)?;
    }
    Ok(())
}

/// The model name of instanced node `n`.
fn instanced_name(doc: &gltf::Document, n: usize) -> String {
    let node = doc.nodes().nth(n).expect("node index");
    node.name()
        .map_or_else(|| format!("node{n}"), str::to_string)
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
    let building = |node| Building::new(node, parts);
    let mut owner: Vec<Option<usize>> = vec![None; graph.parent.len()];
    if settings.models.is_empty() {
        for &i in shown {
            owner[i] = Some(0);
        }
        return Ok((vec![building(None)], owner));
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
        models.push(building(Some(i)));
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
        .filter(|&i| models[i].0.is_some())
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
        // In f32, the precision of glTF's numbers: a chain of decimal translations that meets
        // the grid sums just off it in f64.
        let cols = world.as_mat4().to_cols_array_2d();
        // Each column one unit axis: entries 0 or +-1, one non-zero per column and row.
        let axis = cols[..3].iter().all(|c| {
            c[..3]
                .iter()
                .all(|&x| (x - x.round()).abs() <= 1e-6 && x.round().abs() <= 1.0)
                && c[..3].iter().filter(|x| x.abs() > 0.5).count() == 1
        });
        let det = DMat3::from_mat4(world).determinant();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Primitives that compute tangents read in parallel give the model they give read one by
    /// one: one whose uvs fold, so corners move to copies, and one placed by a mirror.
    #[test]
    fn primitives_read_in_parallel_append_as_read_in_turn() {
        let quad = |uv3: [f32; 2]| {
            let pos: [[f32; 3]; 4] = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.2],
            ];
            let nrm = [[0.0f32, 0.0, 1.0]; 4];
            let uv: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], uv3];
            let idx: [u32; 6] = [0, 1, 2, 0, 2, 3];
            let mut bin: Vec<u8> = Vec::new();
            bin.extend_from_slice(bytemuck::cast_slice(&pos));
            bin.extend_from_slice(bytemuck::cast_slice(&nrm));
            bin.extend_from_slice(bytemuck::cast_slice(&uv));
            bin.extend_from_slice(bytemuck::cast_slice(&idx));
            bin
        };
        let mut bin = quad([2.0, 0.0]);
        bin.extend(quad([0.0, 1.0]));
        let uri: String = bin.iter().map(|b| format!("%{b:02X}")).collect();
        let streams = [
            (0, 48, "VEC3", 5126),
            (48, 48, "VEC3", 5126),
            (96, 32, "VEC2", 5126),
            (128, 24, "SCALAR", 5125),
        ];
        let (mut views, mut accessors) = (Vec::new(), Vec::new());
        for q in 0..2 {
            for (i, (at, len, ty, ct)) in streams.into_iter().enumerate() {
                let at = 152 * q + at;
                views.push(format!(
                    r#"{{"buffer":0,"byteOffset":{at},"byteLength":{len}}}"#
                ));
                let count = if ty == "SCALAR" { 6 } else { 4 };
                let bounds = if i == 0 {
                    r#","min":[0,0,0],"max":[1,1,0.2]"#
                } else {
                    ""
                };
                accessors.push(format!(
                    r#"{{"bufferView":{},"componentType":{ct},"count":{count},"type":"{ty}"{bounds}}}"#,
                    views.len() - 1
                ));
            }
        }
        let prim = |q: usize| {
            let a = 4 * q;
            format!(
                r#"{{"attributes":{{"POSITION":{a},"NORMAL":{},"TEXCOORD_0":{}}},"indices":{}}}"#,
                a + 1,
                a + 2,
                a + 3
            )
        };
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"meshes":[{{"primitives":[{},{}]}}],
"buffers":[{{"byteLength":{},"uri":"data:,{uri}"}}],"bufferViews":[{}],"accessors":[{}]}}"#,
            prim(0),
            prim(1),
            bin.len(),
            views.join(","),
            accessors.join(",")
        );
        let doc = data::open(json.as_bytes(), Path::new(""), false).unwrap();
        let data = Data::load(&doc, 0..views.len(), "t").unwrap();
        let mesh = doc.doc.meshes().next().unwrap();
        let mirror = DMat4::from_scale(orr_math::DVec3::new(-1.0, 1.0, 1.0));
        let prims: Vec<_> = mesh
            .primitives()
            .zip([DMat4::IDENTITY, mirror])
            .map(|(prim, world)| {
                let at = Placing {
                    mesh: 0,
                    world,
                    weights: &[],
                    binding: None,
                    tangent_set: Some(0),
                    displaced: true,
                };
                (prim, at)
            })
            .collect();
        let cx = Context {
            skeleton: None,
            parts: None,
            rigid: false,
        };
        let build = |parallel| {
            let mut b = Building::new(None, 0);
            b.tangents = true;
            append_all(&data, &prims, &cx, &mut b, parallel).unwrap();
            b.finish(None, false)
        };
        let ((a, ga), (b, gb)) = (build(true), build(false));
        assert!(a.positions.len() > 8, "folded uvs copy corners");
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.normals, b.normals);
        assert_eq!(a.tangents, b.tangents);
        assert_eq!(a.uvs, b.uvs);
        assert_eq!(a.moves, b.moves);
        assert_eq!(a.indices, b.indices);
        assert_eq!(ga, gb);
    }
}
