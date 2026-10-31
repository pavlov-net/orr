//! USD import (`.usda`, `.usdc`, `.usdz`) through `openusd`: the composed stage's meshes as
//! one mesh per model, the UsdShade materials they bind, `PointInstancer`s as instance
//! records, and with UsdSkel a skeleton and its clips ([`skel`]).
//!
//! The stage composes its references, payloads (all loaded) and variants, with the
//! selections of [`UsdSettings::variants`] in a session layer; any composition error fails
//! the import. Values are read at [`UsdSettings::time`]. The stage's `upAxis` and
//! `metersPerUnit` (0.01 when unauthored, as USD's) turn stage space into scene space (Y
//! up, meters). Prims that are inactive, abstract or not defined (and their subtrees) are not
//! read; geometry that is `invisible` or of purpose `guide` or `proxy` (or under such a
//! prim) is not cooked; instanceable prims are expanded.
//!
//! Models, every one in scene units and axes:
//! - Each `SkelRoot` is a model named after it, in scene space (bind space), of the skinned
//!   meshes under it (those with a `primvars:skel:jointIndices` primvar), which must bind one
//!   skeleton, cooked as `<asset>/<root>/skeleton` with clips
//!   `<asset>/<root>/clip/<animation>`. Its other meshes are static and cook with the rest
//!   of the stage; a skinned mesh outside a `SkelRoot` fails.
//! - Each prototype of a `PointInstancer` is a model named after its root prim, framed by
//!   the root prim's parent, so the root's own transform is part of the model. The
//!   instancer's `positions`, `orientations`, `scales` and `protoIndices` place it (records
//!   whose `ids` are in `invisibleIds` are dropped) as [`crate::import::Scene::instances`].
//!   Prims under an instancer are cooked only as its prototypes.
//! - Without [`UsdSettings::models`], the rest of the stage is one model, in scene space.
//!   With it, each prim whose name contains a pattern (and has no selected ancestor) is a
//!   model holding its subtree, framed by the prim; meshes outside every model are not
//!   cooked.
//!
//! Meshes are [`mesh`] lists. A mesh with a `subdivisionScheme` other than `none` (USD's
//! fallback is `catmullClark`) is a subdivision cage: [`Subdivision::Hull`] cooks its
//! control faces as they are, [`Subdivision::Cage`] cooks the model's cages as one
//! subdivision cage entry (Catmull-Clark only; every cage of a model shares its
//! `interpolateBoundary`, `faceVaryingLinearInterpolation` and `triangleSubdivisionRule`).
//! Faces take the material of their `GeomSubset` of family `materialBind`, else of the
//! mesh: the direct binding `material:binding:full`, else `material:binding`, of the prim or
//! its nearest ancestor (collection bindings are not read). Materials map as [`crate::shading`] lists and are named `<asset>/material/<path>`
//! by their prim path (without the leading `/`), `default` for unbound faces; faces of a
//! `doubleSided` mesh take a double-sided copy, `<path>/double_sided`.

mod mesh;
mod read;
mod shade;
mod skel;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use openusd::{sdf, usd};
use orr_math::{Affine2, DMat4, DQuat, DVec3, DVec4, Mat4};
use serde::Deserialize;

use crate::import::SceneSubd;
use crate::import::{NodeSelect, Scene, SceneMaterial, SceneModel, SceneSkeleton};
use crate::mesh::{MeshData, MeshSettings, SectionData, Skin};
use crate::patch::{self, MaterialPatch, MeshPatch};
use crate::shading;
use crate::skin::ClipSettings;
use crate::subd::SubdSettings;
use crate::texture::material::{ImageFile, TextureRules};
use crate::{Error, Result};

use mesh::{CageBuilder, MeshPrim, Placing, Welder};
use read::{PrimReader, Reads, Recorder};

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UsdSettings {
    /// Prims that become models; empty makes the whole stage one model.
    pub models: Vec<NodeSelect>,
    pub mesh: MeshSettings,
    /// Applied in order to the mesh of every model whose prim name contains the pattern.
    pub mesh_overrides: Vec<(String, MeshPatch)>,
    pub textures: TextureRules,
    /// Applied in order to every material whose prim name contains the pattern.
    pub material_overrides: Vec<(String, MaterialPatch)>,
    /// Variant selections the stage composes with.
    pub variants: Vec<VariantSelection>,
    pub subdivision: Subdivision,
    /// Refinement of cages with [`Subdivision::Cage`].
    pub subd: SubdSettings,
    /// Time code values are read at; the stage's start time code without it (0 when the
    /// stage authors none).
    pub time: Option<f64>,
    /// The mesh primvar of uv set 0, which shader primvar readers name.
    pub uv: String,
    pub clips: ClipSettings,
}

impl Default for UsdSettings {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            mesh: MeshSettings::default(),
            mesh_overrides: Vec::new(),
            textures: TextureRules::default(),
            material_overrides: Vec::new(),
            variants: Vec::new(),
            subdivision: Subdivision::default(),
            subd: SubdSettings::default(),
            time: None,
            uv: "st".into(),
            clips: ClipSettings::default(),
        }
    }
}

/// A variant selection on one prim.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariantSelection {
    /// Absolute prim path, such as `/World/Tree`.
    pub prim: String,
    pub set: String,
    pub selection: String,
}

/// What a subdivision cage cooks to.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Subdivision {
    /// The control faces, as polygons.
    #[default]
    Hull,
    /// The cage as a subdivision surface; needs the `subd` feature.
    Cage,
}

/// One prim of the traversal.
struct Node {
    prim: usd::Prim,
    parent: Option<usize>,
    /// One past the last node of the prim's subtree, which follows the prim.
    end: usize,
    /// Scene space (Y up, meters) from the prim's space.
    world: DMat4,
    ty: String,
    name: String,
    /// Under a `PointInstancer`, which cooks it only as a prototype.
    instanced: bool,
    /// Invisible or of purpose `guide` or `proxy`, or under such a prim: not drawn, but read
    /// as a skeleton or animation.
    hidden: bool,
}

/// One model being assembled.
struct Building {
    name: Option<String>,
    /// Model space from scene space.
    from_scene: DMat4,
    /// Holds skinned meshes, in bind space.
    skinned: bool,
    /// The node of the prim a skeletal root model holds.
    root: Option<usize>,
    skeleton: Option<usize>,
    welder: Welder,
    cage: CageBuilder,
}

pub(crate) fn import(asset: &str, path: &Path, settings: &UsdSettings) -> Result<Scene> {
    let fail = |message: String| Error::Import {
        asset: asset.into(),
        message: format!("{}: {message}", path.display()),
    };
    Importer::open(path, settings)
        .and_then(|mut i| i.run())
        .map_err(fail)
}

struct Importer<'s> {
    settings: &'s UsdSettings,
    stage: usd::Stage,
    reads: Rc<RefCell<Reads>>,
    time: usd::TimeCode,
    /// Scene space from stage space.
    root: DMat4,
    nodes: Vec<Node>,
    by_path: HashMap<String, usize>,
    images: Vec<ImageFile>,
    image_of: HashMap<String, usize>,
    materials: Vec<SceneMaterial>,
    /// Scene material and uv transform by (material prim path, double-sided).
    material_of: HashMap<(Option<String>, bool), (usize, Option<Affine2>)>,
}

impl<'s> Importer<'s> {
    fn open(path: &Path, settings: &'s UsdSettings) -> Result<Self, String> {
        let reads = Rc::new(RefCell::new(Reads::default()));
        reads.borrow_mut().session = read::session_layer(&settings.variants)?;
        let root = path.to_str().ok_or("the path is not UTF-8")?;
        let stage = usd::Stage::builder()
            .resolver(Recorder::new(Rc::clone(&reads)))
            .session_layer(read::SESSION)
            .open(root)
            .map_err(|e| e.to_string())?;
        if let Some(e) = stage.composition_errors().first() {
            return Err(format!("composition: {e}"));
        }
        for v in &settings.variants {
            let prim = stage.prim(v.prim.as_str()).map_err(|e| e.to_string())?;
            let chosen = prim
                .variant_sets()
                .get_all_variant_selections()
                .map_err(|e| e.to_string())?;
            if !chosen
                .iter()
                .any(|(s, sel)| *s == v.set && *sel == v.selection)
            {
                return Err(format!(
                    "{} has no variant {} in set {}",
                    v.prim, v.selection, v.set
                ));
            }
        }
        let time = settings
            .time
            .unwrap_or(if stage.has_authored_time_code_range() {
                stage.start_time_code()
            } else {
                0.0
            });
        Ok(Self {
            settings,
            stage,
            reads,
            time: usd::TimeCode::new(time),
            root: DMat4::IDENTITY,
            nodes: Vec::new(),
            by_path: HashMap::new(),
            images: Vec::new(),
            image_of: HashMap::new(),
            materials: Vec::new(),
            material_of: HashMap::new(),
        })
    }

    fn reader<'p>(&self, prim: &'p usd::Prim) -> PrimReader<'p> {
        PrimReader {
            prim,
            time: self.time,
        }
    }

    /// Scene space from stage space: Y up, meters.
    fn stage_to_scene(&self) -> Result<DMat4, String> {
        let meta = |k: &str| self.stage.stage_metadata(k).map_err(|e| e.to_string());
        let up = match meta("upAxis")? {
            Some(sdf::Value::Token(t)) => t.as_str().to_string(),
            Some(sdf::Value::String(s)) => s,
            _ => "Y".into(),
        };
        let meters = meta("metersPerUnit")?
            .and_then(|v| read::numbers(&v))
            .map_or(0.01, |v| v[0]);
        let axis = match up.as_str() {
            "Y" => DMat4::IDENTITY,
            // Z up: +Z becomes +Y, +Y becomes -Z.
            "Z" => DMat4::from_cols(DVec4::X, DVec4::NEG_Z, DVec4::Y, DVec4::W),
            a => return Err(format!("upAxis {a}")),
        };
        Ok(DMat4::from_scale(DVec3::splat(meters)) * axis)
    }

    /// Every prim worth reading, parents first, with its scene transform.
    fn traverse(&mut self) -> Result<(), String> {
        let root = self.stage_to_scene()?;
        self.root = root;
        let mut stack: Vec<(usd::Prim, Option<usize>)> = Vec::new();
        let pseudo = self.stage.prim("/").map_err(|e| e.to_string())?;
        let mut kids = pseudo.children().map_err(|e| e.to_string())?;
        kids.reverse();
        stack.extend(kids.into_iter().map(|p| (p, None)));
        while let Some((prim, parent)) = stack.pop() {
            let skip = !prim.is_active().map_err(|e| e.to_string())?
                || !prim.is_defined().map_err(|e| e.to_string())?
                || prim.is_abstract().map_err(|e| e.to_string())?;
            if skip {
                continue;
            }
            let r = self.reader(&prim);
            let up = parent.map(|p| &self.nodes[p]);
            let hidden = up.is_some_and(|p| p.hidden)
                || r.token("visibility")?.as_deref() == Some("invisible")
                || matches!(r.token("purpose")?.as_deref(), Some("guide" | "proxy"));
            let instanced = up.is_some_and(|p| p.instanced || p.ty == "PointInstancer");
            let (local, reset) = r.local_transform()?;
            let world = match (parent, reset) {
                (Some(p), false) => self.nodes[p].world * local,
                _ => root * local,
            };
            let ty = prim
                .type_name()
                .map_err(|e| e.to_string())?
                .map(|t| t.as_str().to_string())
                .unwrap_or_default();
            let name = prim.path().name().unwrap_or_default().to_string();
            let i = self.nodes.len();
            self.by_path.insert(prim.path().as_str().to_string(), i);
            let mut kids = prim.children().map_err(|e| e.to_string())?;
            kids.reverse();
            stack.extend(kids.into_iter().map(|p| (p, Some(i))));
            self.nodes.push(Node {
                prim,
                parent,
                end: i + 1,
                world,
                ty,
                name,
                instanced,
                hidden,
            });
        }
        for i in (0..self.nodes.len()).rev() {
            if let Some(p) = self.nodes[i].parent {
                self.nodes[p].end = self.nodes[p].end.max(self.nodes[i].end);
            }
        }
        Ok(())
    }

    /// The nodes of node `i`'s subtree, `i` first.
    fn subtree(&self, i: usize) -> std::ops::Range<usize> {
        i..self.nodes[i].end
    }

    fn run(&mut self) -> Result<Scene, String> {
        self.traverse()?;
        let n = self.nodes.len();
        let mut models: Vec<Building> = Vec::new();
        let mut owner: Vec<Option<usize>> = vec![None; n];
        let new_model =
            |models: &mut Vec<Building>, node: &Node, from_scene, root: Option<usize>| {
                models.push(Building {
                    name: Some(node.name.clone()),
                    from_scene,
                    skinned: root.is_some(),
                    root,
                    skeleton: None,
                    welder: Welder::default(),
                    cage: CageBuilder::default(),
                });
                models.len() - 1
            };

        // Skeletal roots hold their skinned meshes, in bind space.
        for i in 0..n {
            let node = &self.nodes[i];
            if node.ty != "SkelRoot" || owner[i].is_some() || node.instanced {
                continue;
            }
            let m = new_model(&mut models, node, DMat4::IDENTITY, Some(i));
            for k in self.subtree(i) {
                if owner[k].is_none() && self.nodes[k].ty == "Mesh" && self.skinned(k) {
                    owner[k] = Some(m);
                }
            }
        }

        // Prototype models, and the placements of every instancer.
        let instancers: Vec<usize> = (0..n)
            .filter(|&i| {
                let node = &self.nodes[i];
                node.ty == "PointInstancer" && !node.instanced && !node.hidden
            })
            .collect();
        let mut instances: HashMap<usize, Vec<Mat4>> = HashMap::new();
        let mut proto_model: HashMap<String, usize> = HashMap::new();
        for &ins in &instancers {
            let r = self.reader(&self.nodes[ins].prim);
            let path = r.path().to_string();
            let protos = r.targets("prototypes")?;
            let mut proto_models = Vec::with_capacity(protos.len());
            for p in &protos {
                let key = p.as_str().to_string();
                let m = match proto_model.get(&key) {
                    Some(&m) => m,
                    None => {
                        let root = *self
                            .by_path
                            .get(&key)
                            .ok_or_else(|| format!("{path}: prototype {key} is not cooked"))?;
                        let parent_world = self.nodes[root]
                            .parent
                            .map_or(DMat4::IDENTITY, |q| self.nodes[q].world);
                        let from_scene = self.root * parent_world.inverse();
                        let m = new_model(&mut models, &self.nodes[root], from_scene, None);
                        for o in &mut owner[self.subtree(root)] {
                            o.get_or_insert(m);
                        }
                        proto_model.insert(key, m);
                        m
                    }
                };
                proto_models.push(m);
            }
            for (proto, transform) in self.placements(ins, protos.len())? {
                instances
                    .entry(proto_models[proto])
                    .or_default()
                    .push(transform);
            }
        }
        self.select_models(&mut models, &mut owner)?;

        let meshes: Vec<usize> = (0..n)
            .filter(|&i| self.nodes[i].ty == "Mesh" && owner[i].is_some() && !self.nodes[i].hidden)
            .collect();
        // One skeleton per skeletal root, from what its skinned meshes bind: its prim path,
        // the skeleton, and the root's node.
        let mut skels: Vec<(String, skel::Skel, usize)> = Vec::new();
        for (m, b) in models.iter_mut().enumerate() {
            if !b.skinned {
                continue;
            }
            let mut bound: Option<sdf::Path> = None;
            for &i in meshes
                .iter()
                .filter(|&&i| owner[i] == Some(m) && self.skinned(i))
            {
                let p = self.bound_skeleton(i)?;
                match &bound {
                    Some(b) if *b != p => {
                        return Err(format!(
                            "{}: binds {p}, another mesh of its SkelRoot {b}",
                            self.nodes[i].prim.path().as_str()
                        ));
                    }
                    _ => bound = Some(p),
                }
            }
            let Some(path) = bound else { continue };
            let k = match skels.iter().position(|s| s.0 == path.as_str()) {
                Some(k) => k,
                None => {
                    let node = *self
                        .by_path
                        .get(path.as_str())
                        .ok_or_else(|| format!("skeleton {path} is not cooked"))?;
                    let built = skel::Skel::build(
                        &self.reader(&self.nodes[node].prim),
                        self.nodes[node].world,
                    )?;
                    let root = b.root.expect("skeletal roots have prims");
                    skels.push((path.as_str().to_string(), built, root));
                    skels.len() - 1
                }
            };
            b.skeleton = Some(k);
        }
        for &i in &meshes {
            let prim = self.nodes[i].prim.clone();
            let r = self.reader(&prim);
            let path = r.path().to_string();
            let o = owner[i].expect("owned");
            let mesh = MeshPrim::read(&r, &self.settings.uv)?;
            let as_cage = mesh.scheme != "none" && self.settings.subdivision == Subdivision::Cage;
            let skeleton = models[o].skeleton.map(|k| &skels[k].1);
            let (skin, transform) = match (models[o].skinned, self.skinned(i), skeleton) {
                (true, true, Some(sk)) => {
                    let (s, to_model) = sk.binding(&r, mesh.points.len())?;
                    (Some(s), to_model)
                }
                (false, false, _) => (None, models[o].from_scene * self.nodes[i].world),
                _ => return Err(format!("{path}: a skinned mesh outside a SkelRoot")),
            };
            let faces = self.face_materials(i, &mesh)?;
            let face_material = |f: usize| faces[f];
            let placing = Placing {
                transform,
                face_material: &face_material,
                skin: skin.as_deref(),
            };
            if as_cage {
                models[o].cage.add(&mesh, &placing, &path)?;
            } else {
                models[o].welder.add(&mesh, &placing, &path)?;
            }
        }

        let mut out = Vec::new();
        let mut names = std::collections::HashSet::new();
        let mut instance_out = Vec::new();
        for (mi, b) in models.into_iter().enumerate() {
            // The rest of the stage, or a skeletal root with no skinned mesh, may be empty.
            if (b.name.is_none() || b.skinned)
                && b.welder.groups.values().all(Vec::is_empty)
                && b.cage.is_empty()
            {
                continue;
            }
            if let Some(n) = &b.name
                && !names.insert(n.clone())
            {
                return Err(format!("two models are named {n:?}"));
            }
            if let (Some(n), Some(t)) = (&b.name, instances.get(&mi)) {
                instance_out.push((n.clone(), t.clone()));
            }
            out.push(self.model(b)?);
        }
        if out.is_empty() {
            return Err("the stage has no triangles".into());
        }
        let mut skeletons = Vec::with_capacity(skels.len());
        for (path, sk, root) in skels {
            let clips = self.clips(&path, &sk, root)?;
            skeletons.push(SceneSkeleton {
                prefix: self.nodes[root].name.clone(),
                info: sk.info,
                clips,
            });
        }
        Ok(Scene {
            models: out,
            materials: std::mem::take(&mut self.materials),
            images: std::mem::take(&mut self.images),
            skeletons,
            instances: instance_out,
            files: self.reads.borrow().inputs(),
        })
    }

    /// A built model's mesh, one section per scene material.
    fn model(&self, b: Building) -> Result<SceneModel, String> {
        let mut settings = self.settings.mesh.clone();
        patch::apply_matching(
            &self.settings.mesh_overrides,
            b.name.as_deref().unwrap_or_default(),
            &mut settings,
            MeshPatch::apply,
        );
        let w = b.welder;
        let mut indices = Vec::new();
        let mut sections = Vec::new();
        let mut slots = Vec::new();
        for (&m, tris) in &w.groups {
            if tris.is_empty() {
                continue;
            }
            sections.push(SectionData::new(
                indices.len()..indices.len() + tris.len(),
                &self.materials[m].material,
                Default::default(),
            ));
            indices.extend_from_slice(tris);
            slots.push(m);
        }
        let subd = (!b.cage.is_empty()).then(|| {
            let mut slots: Vec<usize> = b.cage.materials.clone();
            slots.sort_unstable();
            slots.dedup();
            let mut cage = b.cage.cage;
            cage.face_sections = b
                .cage
                .materials
                .iter()
                .map(|m| slots.binary_search(m).expect("listed") as u16)
                .collect();
            SceneSubd {
                cage: crate::subd::CageData {
                    cage,
                    skin: b.skinned.then_some(b.cage.skin),
                },
                materials: slots,
                settings: self.settings.subd.clone(),
            }
        });
        if sections.is_empty() && subd.is_none() {
            return Err(match &b.name {
                Some(n) => format!("model {n} has no triangles"),
                None => "the stage has no triangles".into(),
            });
        }
        Ok(SceneModel {
            node: b.name,
            subd,
            mesh: MeshData {
                positions: w.positions,
                normals: w.normals,
                tangents: None,
                uvs: w.uvs,
                indices,
                sections,
                skin: b.skinned.then_some(Skin {
                    joints: w.joints,
                    weights: w.weights,
                }),
                parts: None,
                min_exponent: None,
            },
            skeleton: b.skeleton,
            settings,
            materials: slots,
        })
    }

    /// Adds the selected models (or the whole-stage model) for prims no model holds yet.
    fn select_models(
        &self,
        models: &mut Vec<Building>,
        owner: &mut [Option<usize>],
    ) -> Result<(), String> {
        let n = self.nodes.len();
        if self.settings.models.is_empty() {
            let whole = models.len();
            models.push(Building {
                name: None,
                from_scene: DMat4::IDENTITY,
                skinned: false,
                root: None,
                skeleton: None,
                welder: Welder::default(),
                cage: CageBuilder::default(),
            });
            for (i, o) in owner.iter_mut().enumerate() {
                if o.is_none() && !self.nodes[i].instanced {
                    *o = Some(whole);
                }
            }
            return Ok(());
        }
        let first = models.len();
        let mut matched = vec![false; self.settings.models.len()];
        for i in 0..n {
            let node = &self.nodes[i];
            if node.instanced || owner[i].is_some() {
                continue;
            }
            if let Some(m) = node.parent.and_then(|p| owner[p]).filter(|&m| m >= first) {
                owner[i] = Some(m);
                continue;
            }
            let mut hit = false;
            for (s, sel) in self.settings.models.iter().enumerate() {
                if node.name.contains(sel.node.as_str()) {
                    matched[s] = true;
                    hit = true;
                }
            }
            if !hit {
                continue;
            }
            if models[first..]
                .iter()
                .any(|m| m.name.as_deref() == Some(&node.name))
            {
                return Err(format!("two selected prims are named {:?}", node.name));
            }
            owner[i] = Some(models.len());
            models.push(Building {
                name: Some(node.name.clone()),
                from_scene: self.root * node.world.inverse(),
                skinned: false,
                root: None,
                skeleton: None,
                welder: Welder::default(),
                cage: CageBuilder::default(),
            });
        }
        if let Some(s) = matched.iter().position(|m| !m) {
            return Err(format!(
                "model pattern {:?} matches no prim",
                self.settings.models[s].node
            ));
        }
        Ok(())
    }

    /// Per instance of a `PointInstancer`: its prototype index and scene transform.
    fn placements(&self, ins: usize, protos: usize) -> Result<Vec<(usize, Mat4)>, String> {
        let node = &self.nodes[ins];
        let r = self.reader(&node.prim);
        let path = r.path();
        let index = r.ints("protoIndices")?.unwrap_or_default();
        let positions = r.numbers("positions")?.unwrap_or_default();
        let orientations = match r.numbers("orientations")? {
            Some(o) => Some(o),
            None => r.numbers("orientationsf")?,
        };
        let scales = r.numbers("scales")?;
        let ids = r.ints("ids")?;
        let invisible: std::collections::HashSet<i64> = r
            .ints("invisibleIds")?
            .unwrap_or_default()
            .into_iter()
            .collect();
        let count = index.len();
        if positions.len() != 3 * count {
            return Err(format!(
                "{path}: {} positions for {count} protoIndices",
                positions.len() / 3
            ));
        }
        let check = |v: &Option<Vec<f64>>, k: usize, what: &str| match v {
            Some(v) if v.len() != k * count => Err(format!(
                "{path}: {} {what} for {count} instances",
                v.len() / k
            )),
            _ => Ok(()),
        };
        check(&orientations, 4, "orientations")?;
        check(&scales, 3, "scales")?;
        let mut out = Vec::with_capacity(count);
        for (i, &p) in index.iter().enumerate() {
            let id = ids
                .as_ref()
                .and_then(|ids| ids.get(i).copied())
                .unwrap_or(i as i64);
            if invisible.contains(&id) {
                continue;
            }
            let proto = usize::try_from(p)
                .ok()
                .filter(|&p| p < protos)
                .ok_or_else(|| format!("{path}: protoIndex {p} of {protos} prototypes"))?;
            let t = DVec3::from_slice(&positions[3 * i..]);
            let q = orientations
                .as_ref()
                .map_or(DQuat::IDENTITY, |o| read::quat(&o[4 * i..]).normalize());
            let s = scales
                .as_ref()
                .map_or(DVec3::ONE, |s| DVec3::from_slice(&s[3 * i..]));
            let m =
                node.world * DMat4::from_scale_rotation_translation(s, q, t) * self.root.inverse();
            out.push((proto, m.as_mat4()));
        }
        Ok(out)
    }

    fn skinned(&self, i: usize) -> bool {
        self.nodes[i]
            .prim
            .attribute("primvars:skel:jointIndices")
            .has_authored_value()
            .unwrap_or(false)
    }

    /// The skeleton a skinned mesh binds: `skel:skeleton` on it or its nearest ancestor.
    fn bound_skeleton(&self, m: usize) -> Result<sdf::Path, String> {
        let mut at = Some(m);
        while let Some(n) = at {
            let r = self.reader(&self.nodes[n].prim);
            if let Some(t) = r.targets("skel:skeleton")?.into_iter().next() {
                return Ok(t);
            }
            at = self.nodes[n].parent;
        }
        Err(format!(
            "{}: skinned without skel:skeleton",
            self.nodes[m].prim.path().as_str()
        ))
    }

    /// Clips of skeleton `path`: its `skel:animationSource` (on the skeleton, else its
    /// skeletal root, node `root`) and every animation under the root.
    fn clips(
        &self,
        path: &str,
        sk: &skel::Skel,
        root: usize,
    ) -> Result<Vec<(String, orr_pack::clip::Clip)>, String> {
        self.settings.clips.check()?;
        let per_second = self.stage.time_codes_per_second();
        let mut sources: Vec<usize> = Vec::new();
        for p in [path, self.nodes[root].prim.path().as_str()] {
            let Some(&n) = self.by_path.get(p) else {
                continue;
            };
            let r = self.reader(&self.nodes[n].prim);
            if let Some(t) = r.targets("skel:animationSource")?.into_iter().next() {
                let a = *self
                    .by_path
                    .get(t.as_str())
                    .ok_or_else(|| format!("{p}: animation {t} is not cooked"))?;
                sources.push(a);
                break;
            }
        }
        sources.extend(
            self.subtree(root)
                .filter(|&i| self.nodes[i].ty == "SkelAnimation"),
        );
        sources.sort_unstable();
        sources.dedup();
        let mut clips = Vec::new();
        for i in sources {
            let node = &self.nodes[i];
            if let Some(c) = sk.clip(&node.prim, per_second, &self.settings.clips)? {
                clips.push((i, node.name.clone(), c));
            }
        }
        let mut count: HashMap<String, usize> = HashMap::new();
        for (_, n, _) in &clips {
            *count.entry(n.clone()).or_default() += 1;
        }
        Ok(clips
            .into_iter()
            .enumerate()
            .map(|(k, (_, n, c))| (if count[&n] == 1 { n } else { k.to_string() }, c))
            .collect())
    }

    /// The material prim bound to prim `i` or its nearest ancestor.
    fn bound_material(&self, mut at: Option<usize>) -> Result<Option<sdf::Path>, String> {
        while let Some(n) = at {
            if let Some(m) = self.direct_binding(&self.nodes[n].prim)? {
                return Ok(Some(m));
            }
            at = self.nodes[n].parent;
        }
        Ok(None)
    }

    fn direct_binding(&self, prim: &usd::Prim) -> Result<Option<sdf::Path>, String> {
        let r = self.reader(prim);
        for rel in ["material:binding:full", "material:binding"] {
            if let Some(t) = r.targets(rel)?.into_iter().next() {
                return Ok(Some(t));
            }
        }
        Ok(None)
    }

    /// Scene material and uv transform per face of mesh prim `i`.
    fn face_materials(
        &mut self,
        i: usize,
        mesh: &MeshPrim,
    ) -> Result<Vec<(usize, Option<Affine2>)>, String> {
        let base = self.bound_material(Some(i))?;
        let base = self.material(base, mesh.double_sided)?;
        let mut faces = vec![base; mesh.counts.len()];
        let subsets: Vec<usize> = self
            .subtree(i)
            .filter(|&k| self.nodes[k].parent == Some(i) && self.nodes[k].ty == "GeomSubset")
            .collect();
        for k in subsets {
            let r = self.reader(&self.nodes[k].prim);
            if r.token("familyName")?.as_deref() != Some("materialBind")
                || r.token("elementType")?
                    .as_deref()
                    .is_some_and(|e| e != "face")
            {
                continue;
            }
            let Some(m) = self.direct_binding(&self.nodes[k].prim)? else {
                continue;
            };
            let path = r.path().to_string();
            let indices = r.ints("indices")?.unwrap_or_default();
            let m = self.material(Some(m), mesh.double_sided)?;
            for f in indices {
                let slot = usize::try_from(f)
                    .ok()
                    .and_then(|f| faces.get_mut(f))
                    .ok_or_else(|| format!("{path}: face {f} past the mesh"))?;
                *slot = m;
            }
        }
        Ok(faces)
    }

    /// The scene material of a material prim (`None`: the default material).
    fn material(
        &mut self,
        prim: Option<sdf::Path>,
        double_sided: bool,
    ) -> Result<(usize, Option<Affine2>), String> {
        let key = (prim.as_ref().map(|p| p.as_str().to_string()), double_sided);
        if let Some(&m) = self.material_of.get(&key) {
            return Ok(m);
        }
        let (mut mapped, name) = match &prim {
            None => (
                shading::Mapped {
                    material: Default::default(),
                    textures: Default::default(),
                    uv: None,
                },
                "default".to_string(),
            ),
            Some(p) => {
                let images = &mut self.images;
                let image_of = &mut self.image_of;
                let reads = Rc::clone(&self.reads);
                let mut load = |a: &sdf::AssetPath| -> Result<usize, String> {
                    let resolved = a
                        .resolved_path()
                        .ok_or_else(|| format!("asset {} does not resolve", a.asset_path()))?
                        .to_string();
                    if let Some(&i) = image_of.get(&resolved) {
                        return Ok(i);
                    }
                    let bytes = Recorder::new(Rc::clone(&reads))
                        .read(&resolved)
                        .map_err(|e| format!("{resolved}: {e}"))?;
                    images.push(ImageFile {
                        name: a.asset_path().to_string(),
                        bytes,
                    });
                    image_of.insert(resolved, images.len() - 1);
                    Ok(images.len() - 1)
                };
                let (net, surface, displacement) =
                    shade::network(&self.stage, self.time, p, &mut load)?;
                let uv = self.settings.uv.clone();
                let uv_set = |name: &str| {
                    if name == uv {
                        Ok(0)
                    } else {
                        Err(format!(
                            "reads primvar {name}; meshes cook uv set 0 from {uv}"
                        ))
                    }
                };
                let mapped = shading::map(&net, surface, displacement, &uv_set)
                    .map_err(|e| format!("material {p}: {e}"))?;
                (mapped, p.as_str().trim_start_matches('/').to_string())
            }
        };
        let prim_name = prim
            .as_ref()
            .and_then(|p| p.name().map(str::to_string))
            .unwrap_or_default();
        patch::apply_matching(
            &self.settings.material_overrides,
            &prim_name,
            &mut mapped.material,
            MaterialPatch::apply,
        );
        if double_sided {
            mapped.material.double_sided = 1;
        }
        let suffix = if double_sided {
            format!("{name}/double_sided")
        } else {
            name
        };
        self.materials.push(SceneMaterial {
            suffix,
            material: mapped.material,
            textures: mapped.textures,
        });
        let m = (self.materials.len() - 1, mapped.uv);
        self.material_of.insert(key, m);
        Ok(m)
    }
}
