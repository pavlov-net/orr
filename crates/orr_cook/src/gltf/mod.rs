//! glTF 2.0 import: the default scene's triangle primitives as one mesh per model, the
//! materials they use, and with skins a skeleton and its clips ([`crate::skin`]).
//!
//! Without `GltfSettings::models` the whole scene is one model. With it, each node whose
//! name contains a pattern (and has no selected ancestor) is a model holding its subtree, in
//! the node's own space; meshes outside every selected subtree are not cooked. Node
//! transforms are applied to the vertices. With skins, the mesh is in bind space: a skinned
//! primitive's vertices stay as stored (glTF ignores its node's transform), take their
//! joints and weights from `JOINTS_n`/`WEIGHTS_n`, and a primitive without a skin follows
//! its nearest skeleton joint rigidly; skins and model selection do not mix.
//!
//! Rigid parts come from the custom vertex attribute `PartAttribute::name` (integer, or
//! float rounded to the nearest integer; 0 where a primitive lacks it). With skins the
//! values are joints of the primitive's skin, pivots their bind positions and `Part::joint`
//! the skeleton joint; without, pivots come from the `PartAttribute::pivot` attribute, else
//! the model origin.
//!
//! Joined models (`GltfSettings::joined`) share the coarsest grid step among them, and each
//! one's node must sit on that grid: translation a multiple of the step, rotation a multiple
//! of 90 degrees, unit scale, every placed grid coordinate within 24 bits.
//!
//! Materials map as [`material`] lists. Textures become the material's canonical channel
//! set ([`crate::texture::material`]); `texCoord` 0 or 1 becomes the binding's uv set and
//! the sampler's wrap modes set the mip filter's edges.

mod data;
mod material;

use std::collections::BTreeMap;
use std::path::Path;

use base64::Engine;
use glam::{Mat3, Mat4, Vec3};
use gltf::image::Source as ImageSource;
use gltf::mesh::{Mode, Semantic};
use orr_pack::clip::Clip;
use orr_pack::material::Material;
use orr_pack::skeleton::SkeletonInfo;
use serde::Deserialize;

use crate::mesh::{self, MeshData, MeshSettings, PartData, Pivots, SectionData, Skin};
use crate::patch::{self, MaterialPatch, MeshPatch};
use crate::skin::{Graph, Skeleton, influences};
use crate::state::{Input, Memo};
use crate::texture::material::{ImageFile, MaterialTextures};
use crate::{Error, GltfSettings, Result};

/// Selects nodes of a glTF scene as models.
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

pub(crate) struct Imported {
    pub models: Vec<ImportedModel>,
    pub materials: Vec<ImportedMaterial>,
    pub images: Vec<ImageFile>,
    pub skeleton: Option<SkeletonInfo>,
    /// Clips by name; their skeleton id is left for the caller to set.
    pub clips: Vec<(String, Clip)>,
    /// Every file read: the document, its buffers and images.
    pub files: Vec<Input>,
}

pub(crate) struct ImportedModel {
    /// The selected node's name; `None` for the whole scene.
    pub node: Option<String>,
    pub mesh: MeshData,
    pub settings: MeshSettings,
    /// Index into [`Imported::materials`] per mesh section.
    pub materials: Vec<usize>,
}

pub(crate) struct ImportedMaterial {
    /// Entry name suffix: the glTF material index, or `default`.
    pub suffix: String,
    pub material: Material,
    pub textures: MaterialTextures,
}

#[derive(Default)]
struct Vertices {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
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
    groups: BTreeMap<usize, Vec<u32>>,
    /// Bind position and skeleton joint per part value, from skins.
    joint_pivots: Vec<Option<([f32; 3], u16)>>,
}

pub(crate) fn import(
    asset: &str,
    path: &Path,
    memo: &Memo,
    settings: &GltfSettings,
) -> Result<Imported> {
    let fail = |message: String| Error::Import {
        asset: asset.into(),
        message,
    };
    let mut files = Vec::new();
    let mut read = |p: &Path| {
        let (bytes, hash) = memo.read(p)?;
        files.push(Input {
            path: p.to_owned(),
            hash,
        });
        Ok::<_, Error>(bytes)
    };
    let bytes = read(path)?;
    let dir = path.parent().unwrap_or(Path::new("")).to_owned();
    // A data URI's bytes, or a file next to the document.
    let mut load_uri = |uri: &str, what: &str| match uri.strip_prefix("data:") {
        Some(data) => {
            let (_, b64) = data
                .split_once(";base64,")
                .ok_or_else(|| fail(format!("{what}: data URI is not base64")))?;
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| fail(format!("{what}: {e}")))
        }
        None => read(&dir.join(uri)),
    };
    let src = data::open(&bytes, &mut load_uri, &|m| {
        fail(format!("{}: {m}", path.display()))
    })?;
    let (doc, buffers) = (&src.doc, &src.buffers);

    let mut images = Vec::new();
    for img in doc.images() {
        let (name, bytes) = match img.source() {
            ImageSource::View { view, .. } => {
                let data = &buffers[view.buffer().index()];
                let bytes = data
                    .get(view.offset()..view.offset() + view.length())
                    .ok_or_else(|| fail(format!("image {}: view outside its buffer", img.index())))?
                    .to_vec();
                (format!("image {}", img.index()), bytes)
            }
            ImageSource::Uri { uri, .. } => (
                uri.to_string(),
                load_uri(uri, &format!("image {}", img.index()))?,
            ),
        };
        images.push(ImageFile {
            name: img.name().map_or(name, str::to_string),
            bytes,
        });
    }

    let scene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .ok_or_else(|| fail("no scene".into()))?;
    let graph = Graph::new(doc, &scene);
    let mut skins: Vec<usize> = graph
        .order
        .iter()
        .filter_map(|&n| {
            let node = doc.nodes().nth(n)?;
            node.mesh()?;
            Some(node.skin()?.index())
        })
        .collect();
    skins.sort_unstable();
    skins.dedup();
    let skeleton = Skeleton::build(doc, buffers, &graph, &skins).map_err(fail)?;
    if skeleton.is_some() && !settings.models.is_empty() {
        return Err(fail(
            "model selection needs a scene without skins: skinned vertices stay in bind space"
                .into(),
        ));
    }

    let (mut models, owner) = select_models(doc, &graph, settings).map_err(fail)?;
    let parts = settings.parts_from_attribute.as_ref();
    let mut materials: BTreeMap<usize, ImportedMaterial> = BTreeMap::new();
    for &n in &graph.order {
        let node = doc.nodes().nth(n).expect("node index");
        let Some(mesh) = node.mesh() else { continue };
        let Some(b) = owner[n].map(|m| &mut models[m]) else {
            continue;
        };
        let (world, binding) = match (&skeleton, node.skin()) {
            (None, _) => (b.from_scene * graph.world[n], None),
            (Some(sk), Some(s)) => (
                Mat4::IDENTITY,
                Some(Binding::Skin(sk.skin_joints(s.index()))),
            ),
            (Some(sk), None) => {
                let (j, m) = sk.attach(n, &graph).map_err(fail)?;
                (m, Some(Binding::Rigid(j)))
            }
        };
        let linear = Mat3::from_mat4(world);
        let flip = linear.determinant() < 0.0;
        let normal_m = linear.inverse().transpose();
        for prim in mesh.primitives() {
            let what = format!("mesh {} primitive {}", mesh.index(), prim.index());
            let fail_prim = |m: String| fail(format!("{what}: {m}"));
            if prim.mode() != Mode::Triangles {
                return Err(fail_prim(format!(
                    "mode {:?} is not triangles",
                    prim.mode()
                )));
            }
            let position = prim
                .get(&Semantic::Positions)
                .ok_or_else(|| fail_prim("no positions".into()))?;
            let n = position.count();
            // An attribute as rows of `k` floats, `None` when the primitive lacks it.
            let rows = |s: Semantic, k: usize| -> Result<Option<Vec<f32>>> {
                prim.get(&s)
                    .map(|a| {
                        if a.count() != n {
                            return Err(fail_prim("attribute counts differ".into()));
                        }
                        src.floats(&a, k).map_err(fail_prim)
                    })
                    .transpose()
            };
            let custom = |name: &str, k| -> Result<Option<Vec<f32>>> {
                let key = name.strip_prefix('_').ok_or_else(|| {
                    fail_prim(format!("{name}: custom vertex attributes start with _"))
                })?;
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
                Some(a) => src.indices(&a).map_err(fail_prim)?,
                None => (0..n as u32).collect(),
            };
            if !indices.len().is_multiple_of(3) || indices.iter().any(|&i| i as usize >= n) {
                return Err(fail_prim("bad index list".into()));
            }
            if flip {
                indices.chunks_exact_mut(3).for_each(|t| t.swap(1, 2));
            }
            let uvs: Vec<[f32; 2]> = rows(Semantic::TexCoords(0), 2)?
                .map_or_else(|| vec![[0.0; 2]; n], |v| v.as_chunks::<2>().0.to_vec());
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
            let skin: Vec<([u8; 4], [u8; 4])> = match &binding {
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
                            _ => {
                                return Err(fail_prim(format!(
                                    "JOINTS_{set} and WEIGHTS_{set} come in pairs"
                                )));
                            }
                        }
                    }
                    if sets.is_empty() {
                        return Err(fail_prim("skinned without JOINTS_0".into()));
                    }
                    (0..n)
                        .map(|i| influences(&sets, map, i))
                        .collect::<Result<_, _>>()
                        .map_err(fail_prim)?
                }
            };
            let (values, pivots) = match parts {
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
                                    .ok_or_else(|| {
                                        fail_prim(format!("{} value {x} is not a part", pa.name))
                                    })
                            })
                            .collect::<Result<_>>()?,
                        None => vec![0; n],
                    };
                    let pivots = match (&pa.pivot, &skeleton) {
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
                    if let (Some(sk), Some(binding)) = (&skeleton, &binding) {
                        for &v in &values {
                            if v >= pa.parts {
                                continue;
                            }
                            let joint = match binding {
                                Binding::Skin(map) => {
                                    *map.get(usize::from(v)).ok_or_else(|| {
                                        fail_prim(format!(
                                            "part value {v} is not a joint of the skin"
                                        ))
                                    })?
                                }
                                Binding::Rigid(j) => *j,
                            };
                            let slot = &mut b.joint_pivots[usize::from(v)];
                            if slot.is_some_and(|s| s.1 != joint) {
                                return Err(fail_prim(format!(
                                    "part value {v} names two skeleton joints"
                                )));
                            }
                            *slot = Some((sk.bind_origin(joint), joint));
                        }
                    }
                    (values, pivots)
                }
            };

            let base = b.v.positions.len() as u32;
            let mat = prim.material();
            let index = mat.index().unwrap_or(usize::MAX);
            if let std::collections::btree_map::Entry::Vacant(e) = materials.entry(index) {
                let t = material::textures(doc, &src.json, &mat)
                    .map_err(|m| fail(format!("material {index}: {m}")))?;
                let mut m = material::material(&src.json, &mat);
                patch::apply_matching(
                    &settings.material_overrides,
                    mat.name().unwrap_or_default(),
                    &mut m,
                    MaterialPatch::apply,
                );
                e.insert(ImportedMaterial {
                    suffix: match index {
                        usize::MAX => "default".to_string(),
                        i => i.to_string(),
                    },
                    material: m,
                    textures: t,
                });
            }
            let tris = b.groups.entry(index).or_default();
            let v = &mut b.v;
            match normals {
                Some(normals) => {
                    v.positions.extend(positions);
                    v.normals.extend(normals);
                    v.tangents.extend(tangents);
                    v.uvs.extend(uvs);
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
        }
    }

    // Material slots in glTF index order, the default material last; only those in use.
    let used: Vec<usize> = materials.keys().copied().collect();
    let mut out = Vec::new();
    for b in models {
        let node = b.node;
        let name = node.map(|n| {
            doc.nodes()
                .nth(n)
                .and_then(|d| d.name())
                .unwrap_or_default()
                .to_string()
        });
        let mut mesh_settings = settings.mesh.clone();
        patch::apply_matching(
            &settings.mesh_overrides,
            name.as_deref().unwrap_or_default(),
            &mut mesh_settings,
            MeshPatch::apply,
        );
        let mut indices = Vec::new();
        let mut sections = Vec::new();
        let mut slots = Vec::new();
        for (index, tris) in &b.groups {
            if tris.is_empty() {
                continue;
            }
            sections.push(SectionData::new(
                indices.len()..indices.len() + tris.len(),
                &materials[index].material,
                Default::default(),
            ));
            indices.extend_from_slice(tris);
            slots.push(used.binary_search(index).expect("material in use"));
        }
        if sections.is_empty() {
            return Err(fail(match &name {
                Some(n) => format!("model {n} has no triangles"),
                None => "the scene has no triangles".into(),
            }));
        }
        let mesh_parts = parts.map(|pa| PartData {
            values: b.v.values.clone(),
            count: pa.parts,
            by_section: pa.by_section,
            pivots: match (&skeleton, &pa.pivot) {
                (Some(_), _) => Pivots::PerValue(
                    b.joint_pivots
                        .iter()
                        .map(|p| p.map_or([0.0; 3], |p| p.0))
                        .collect(),
                ),
                (None, Some(_)) => Pivots::PerVertex(b.v.pivots.clone()),
                (None, None) => Pivots::Origin,
            },
            joints: skeleton.as_ref().map(|_| {
                b.joint_pivots
                    .iter()
                    .map(|p| p.map_or(0, |p| p.1))
                    .collect()
            }),
        });
        let v = b.v;
        out.push((
            node,
            ImportedModel {
                node: name,
                mesh: MeshData {
                    positions: v.positions,
                    normals: v.normals,
                    tangents: b.tangents.then_some(v.tangents),
                    uvs: v.uvs,
                    indices,
                    sections,
                    skin: skeleton.is_some().then_some(Skin {
                        joints: v.joints,
                        weights: v.weights,
                    }),
                    parts: mesh_parts,
                    min_exponent: None,
                },
                settings: mesh_settings,
                materials: slots,
            },
        ));
    }
    join(&mut out, &graph, &settings.joined).map_err(fail)?;
    let clips = match &skeleton {
        Some(sk) => sk.clips(doc, buffers, &settings.clips).map_err(fail)?,
        None => Vec::new(),
    };

    Ok(Imported {
        models: out.into_iter().map(|m| m.1).collect(),
        materials: materials.into_values().collect(),
        images,
        skeleton: skeleton.map(|s| s.info),
        clips,
        files,
    })
}

/// The models to build and, per glTF node, the model it belongs to.
fn select_models(
    doc: &gltf::Document,
    graph: &Graph,
    settings: &GltfSettings,
) -> Result<(Vec<Building>, Vec<Option<usize>>), String> {
    let parts = settings
        .parts_from_attribute
        .as_ref()
        .map_or(0, |p| usize::from(p.parts));
    let building = |node, from_scene| Building {
        node,
        from_scene,
        v: Vertices::default(),
        tangents: true,
        groups: BTreeMap::new(),
        joint_pivots: vec![None; parts],
    };
    let n = graph.parent.len();
    if settings.models.is_empty() {
        let owner = (0..n).map(|i| graph.in_scene[i].then_some(0)).collect();
        return Ok((vec![building(None, Mat4::IDENTITY)], owner));
    }
    let mut models = Vec::new();
    let mut owner: Vec<Option<usize>> = vec![None; n];
    let mut names: Vec<&str> = Vec::new();
    let mut matched = vec![false; settings.models.len()];
    for &i in &graph.order {
        if let Some(m) = graph.parent[i].and_then(|p| owner[p]) {
            owner[i] = Some(m);
            continue;
        }
        let name = doc
            .nodes()
            .nth(i)
            .and_then(|d| d.name())
            .unwrap_or_default();
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
        return Err(format!(
            "model pattern {:?} matches no node",
            settings.models[s].node
        ));
    }
    Ok((models, owner))
}

/// Puts the joined models on one grid and checks that their nodes keep them on it.
fn join(
    models: &mut [(Option<usize>, ImportedModel)],
    graph: &Graph,
    patterns: &[String],
) -> Result<(), String> {
    if patterns.is_empty() {
        return Ok(());
    }
    let matches = |m: &ImportedModel, p: &str| m.node.as_deref().is_some_and(|n| n.contains(p));
    let joined: Vec<usize> = (0..models.len())
        .filter(|&i| patterns.iter().any(|p| matches(&models[i].1, p)))
        .collect();
    if let Some(p) = patterns
        .iter()
        .find(|p| !joined.iter().any(|&i| matches(&models[i].1, p)))
    {
        return Err(format!("joined pattern {p:?} matches no selected model"));
    }
    let k = joined
        .iter()
        .map(|&i| {
            let m = &models[i].1;
            mesh::grid_exponent(&m.mesh.positions, m.settings.position_step_log2)
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
        let local = m
            .mesh
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
        m.mesh.min_exponent = Some(k);
    }
    Ok(())
}
