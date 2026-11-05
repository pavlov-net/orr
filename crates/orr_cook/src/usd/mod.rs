//! USD import (`.usda`, `.usdc`, `.usdz`) through `openusd`: the composed stage's meshes as
//! one mesh per model, the UsdShade materials they bind, `PointInstancer`s as instance
//! records, and with UsdSkel a skeleton and its clips ([`skel`]).
//!
//! The stage composes its references, payloads (all loaded) and variants, with the
//! selections of [`UsdSettings::variants`] in a session layer; any composition error fails
//! the import. Values are read at [`UsdSettings::time`]. The stage's `upAxis` and
//! `metersPerUnit` (0.01 when unauthored, as USD's) turn stage space into scene space (Y
//! up, meters). Inactive prims (and their subtrees) are not read, and prims under a `class` or
//! `over` are read only as prototypes. An Imageable prim that is `invisible`, or has an
//! effective purpose of `guide` or `proxy`, is not drawn: invisibility holds for the subtree,
//! and a prim takes the purpose of the nearest Imageable prim at or above it that authors one.
//! Instanceable prims are expanded.
//!
//! Models, every one in scene units and axes:
//! - Each `SkelRoot` is a model named after it, in scene space (bind space), of the skinned
//!   meshes under it (those with joint influences and a bound skeleton), which must bind one
//!   skeleton, cooked as `<asset>/<root>/skeleton` with clips
//!   `<asset>/<root>/clip/<animation>`. Its other meshes are static and cook with the rest
//!   of the stage, as do meshes outside a `SkelRoot`, whose bindings mean nothing. A skinned
//!   mesh under a `SkelRoot` in a prototype fails: a copy has no pose of its own.
//! - Each prototype of a `PointInstancer` is a model named after its root prim (after its path
//!   if another model has the name). The root's own transform is part of the model, as is
//!   what each prim under it adds, up to any prim that resets the transform stack. Only the
//!   prototype's own subtree decides what is hidden, and a prototype with nothing to draw
//!   places nothing. The instancer's `positions`, `orientations` (`orientationsf` where its
//!   first value is not empty), `scales` and `protoIndices` place it, without the instances
//!   whose `ids` are in `invisibleIds` or `inactiveIds`, as
//!   [`crate::import::Scene::instances`]. Where `velocities` or `angularVelocities` bracket
//!   the time at the samples of the values they move, those extrapolate from the lower sample
//!   (with `accelerations` likewise) and neither value interpolates. An instancer under a
//!   prototype is placed by each copy of it. Prims under an instancer are cooked only as
//!   prototypes; one elsewhere that is defined also draws where it stands. A transform that
//!   collapses an axis is not a placement: the prototype is cooked once per such linear
//!   transform with it applied, as model `<model>/baked/<k>`, placed by translation alone.
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
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::rc::Rc;

use openusd::{sdf, usd};
use orr_math::{Affine2, DMat3, DMat4, DQuat, DVec3, DVec4};
use serde::Deserialize;

use crate::import::SceneSubd;
use crate::import::{
    NodeSelect, Placed, Scene, SceneMaterial, SceneModel, SceneSkeleton, collapses, placements,
};
use crate::mesh::{MeshData, MeshSettings, MeshSource, SectionData, Skin};
use crate::patch::{self, MaterialPatch, MeshPatch};
use crate::shading;
use crate::skin::ClipSettings;
use crate::subd::SubdSettings;
use crate::texture::material::{ImageData, ImageFile, TextureRules};
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
    /// The mesh primvar of uv set 1; without it meshes carry one uv set.
    pub uv1: Option<String>,
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
            uv1: None,
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

/// Schemas derived from `UsdGeomImageable`; other prims have no visibility or purpose.
const IMAGEABLE: &[&str] = &[
    "BasisCurves",
    "Camera",
    "Capsule",
    "Capsule_1",
    "Cone",
    "Cube",
    "Cylinder",
    "Cylinder_1",
    "CylinderLight",
    "DiskLight",
    "DistantLight",
    "DomeLight",
    "DomeLight_1",
    "Field3DAsset",
    "GenerativeProcedural",
    "GeometryLight",
    "HermiteCurves",
    "LightFilter",
    "Mesh",
    "NurbsCurves",
    "NurbsPatch",
    "OpenVDBAsset",
    "PhysicsDistanceJoint",
    "PhysicsFixedJoint",
    "PhysicsJoint",
    "PhysicsPrismaticJoint",
    "PhysicsRevoluteJoint",
    "PhysicsSphericalJoint",
    "Plane",
    "PluginLight",
    "PluginLightFilter",
    "PointInstancer",
    "Points",
    "PortalLight",
    "RectLight",
    "Scope",
    "SkelRoot",
    "Skeleton",
    "SpatialAudio",
    "Sphere",
    "SphereLight",
    "TetMesh",
    "Volume",
    "Xform",
];

/// One prim of the traversal.
struct Node {
    prim: usd::Prim,
    parent: Option<usize>,
    /// One past the last node of the prim's subtree, which follows the prim.
    end: usize,
    /// Scene space (Y up, meters) from the prim's space.
    world: DMat4,
    /// The prim's own transform.
    local: DMat4,
    /// `local` starts from the stage, not the parent.
    reset: bool,
    ty: String,
    name: String,
    /// Under a `PointInstancer`, or a `class` or `over` (or one): cooked only as a prototype.
    instanced: bool,
    /// Imageable and `invisible`: not drawn with its subtree, but read as a skeleton or
    /// animation.
    invisible: bool,
    /// The purpose it authors, if Imageable.
    purpose: Option<String>,
}

/// One model being assembled.
struct Building {
    name: Option<String>,
    /// The node that frames the model, in scene units and axes; scene space without one.
    frame: Option<usize>,
    /// Holds skinned meshes, in bind space.
    skinned: bool,
    /// The node of the prim a skeletal root model holds.
    root: Option<usize>,
    /// The root node of the prototype the model holds.
    prototype: Option<usize>,
    skeleton: Option<usize>,
    welder: Welder,
    cage: CageBuilder,
}

impl Building {
    fn new(name: Option<String>, frame: Option<usize>, root: Option<usize>) -> Self {
        Self {
            name,
            frame,
            skinned: root.is_some(),
            root,
            prototype: None,
            skeleton: None,
            welder: Welder::default(),
            cage: CageBuilder::default(),
        }
    }
}

/// How a mesh under a `SkelRoot` binds a skeleton: nodes of the prims that say so.
struct Bound {
    skeleton: sdf::Path,
    /// Where `skel:skeleton` is authored.
    site: usize,
    joints: Option<usize>,
    indices: usize,
    weights: usize,
    geom_bind: Option<usize>,
}

/// One prototype of the stage.
#[derive(Clone)]
struct Prototype {
    model: usize,
    /// Instancers in its subtree, each placed by every copy of it.
    nested: Vec<usize>,
}

/// What an instancer places, the same for every copy of it.
struct Instancer {
    /// Per prototype target its root node and prototype, if it has any.
    protos: Vec<Option<(usize, Prototype)>>,
    /// [`Importer::placements`].
    placed: Vec<(usize, DMat4)>,
}

/// The stage's instancers as they are placed.
#[derive(Default)]
struct Instancing {
    /// Prototypes by their root nodes.
    prototypes: HashMap<usize, Prototype>,
    /// Instancers by their nodes.
    instancers: HashMap<usize, Rc<Instancer>>,
    /// Placements per model.
    copies: BTreeMap<usize, Vec<DMat4>>,
    /// Meshes the prototypes draw, with their models.
    meshes: Vec<(usize, usize)>,
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
    warnings: Vec<String>,
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
            warnings: Vec::new(),
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
            if !prim.is_active().map_err(|e| e.to_string())? {
                continue;
            }
            let r = self.reader(&prim);
            let up = parent.map(|p| &self.nodes[p]);
            let instanced = !prim.is_defined().map_err(|e| e.to_string())?
                || prim.is_abstract().map_err(|e| e.to_string())?
                || up.is_some_and(|p| p.instanced || p.ty == "PointInstancer");
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
            let imageable = IMAGEABLE.contains(&ty.as_str());
            let invisible = imageable && r.token("visibility")?.as_deref() == Some("invisible");
            let purpose = if imageable { r.token("purpose")? } else { None };
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
                local,
                reset,
                ty,
                name,
                instanced,
                invisible,
                purpose,
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

    /// Whether node `n` is not drawn, by what it and the prims above it say up to and with
    /// `top`, or the stage's root without it.
    fn hidden(&self, n: usize, top: Option<usize>) -> bool {
        let mut purpose = None;
        let mut at = Some(n);
        while let Some(i) = at {
            let node = &self.nodes[i];
            if node.invisible {
                return true;
            }
            purpose = purpose.or(node.purpose.as_deref());
            at = node.parent.filter(|_| Some(i) != top);
        }
        matches!(purpose, Some("guide" | "proxy"))
    }

    fn run(&mut self) -> Result<Scene, String> {
        self.traverse()?;
        let n = self.nodes.len();
        let mut models: Vec<Building> = Vec::new();
        let mut owner: Vec<Option<usize>> = vec![None; n];
        let mut bound: HashMap<usize, Bound> = HashMap::new();

        // Skeletal roots hold their skinned meshes, in bind space.
        for i in 0..n {
            let node = &self.nodes[i];
            if node.ty != "SkelRoot" || owner[i].is_some() || node.instanced {
                continue;
            }
            let m = models.len();
            models.push(Building::new(Some(node.name.clone()), None, Some(i)));
            for k in self.subtree(i) {
                let node = &self.nodes[k];
                if owner[k].is_some() || node.ty != "Mesh" || node.instanced {
                    continue;
                }
                if let Some(b) = self.binding(k, i)? {
                    bound.insert(k, b);
                    owner[k] = Some(m);
                }
            }
        }

        self.select_models(&mut models, &mut owner)?;

        // Prototype models, which name themselves around the rest, and the placements of every
        // instancer.
        let mut instancing = Instancing::default();
        for i in 0..n {
            let node = &self.nodes[i];
            if node.ty == "PointInstancer" && !node.instanced && !self.hidden(i, None) {
                let world = node.world;
                self.place(i, world, &mut models, &mut instancing, &mut Vec::new())?;
            }
        }

        // Per instanced model its placements, and per model the models its meshes go to, with
        // what they apply: the model itself unless it places nothing, and those that bake its
        // other copies.
        let mut placed: HashMap<usize, Vec<Placed>> = HashMap::new();
        let mut holders: Vec<Vec<(usize, DMat4)>> = (0..models.len())
            .map(|m| vec![(m, DMat4::IDENTITY)])
            .collect();
        for (m, transforms) in instancing.copies {
            let name = models[m].name.clone().unwrap_or_default();
            let (direct, baked) = placements(&transforms).map_err(|e| format!("{name}: {e}"))?;
            if direct.is_empty() {
                holders[m].clear();
            }
            placed.insert(m, direct);
            for (k, (linear, copies)) in baked.into_iter().enumerate() {
                placed.insert(models.len(), copies);
                holders[m].push((models.len(), linear));
                let prototype = models[m].prototype;
                models.push(Building::new(Some(format!("{name}/baked/{k}")), None, None));
                models.last_mut().expect("pushed").prototype = prototype;
            }
        }

        // Every mesh drawn, with the model it goes to.
        let mut meshes: Vec<(usize, usize)> = (0..n)
            .filter(|&i| self.nodes[i].ty == "Mesh" && !self.hidden(i, None))
            .filter_map(|i| Some((i, owner[i]?)))
            .collect();
        meshes.extend(instancing.meshes);
        // One skeleton per skeletal root, from what its skinned meshes bind: its prim path,
        // the skeleton, the root's node and the prims that bind it.
        let mut skels: Vec<(String, skel::Skel, usize, Vec<usize>)> = Vec::new();
        for (m, b) in models.iter_mut().enumerate() {
            if !b.skinned {
                continue;
            }
            let mut skeleton: Option<&sdf::Path> = None;
            let mut sites = Vec::new();
            for &(i, _) in meshes.iter().filter(|h| h.1 == m) {
                let binding = &bound[&i];
                match skeleton {
                    Some(s) if *s != binding.skeleton => {
                        return Err(format!(
                            "{}: binds {}, another mesh of its SkelRoot {s}",
                            self.nodes[i].prim.path().as_str(),
                            binding.skeleton,
                        ));
                    }
                    _ => skeleton = Some(&binding.skeleton),
                }
                sites.push(binding.site);
            }
            let Some(path) = skeleton else { continue };
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
                        self.root,
                    )?;
                    let root = b.root.expect("skeletal roots have prims");
                    skels.push((path.as_str().to_string(), built, root, Vec::new()));
                    skels.len() - 1
                }
            };
            skels[k].3.extend(sites);
            b.skeleton = Some(k);
        }
        for &(i, o) in &meshes {
            let prim = self.nodes[i].prim.clone();
            let r = self.reader(&prim);
            let path = r.path().to_string();
            let mesh = MeshPrim::read(&r, &self.settings.uv, self.settings.uv1.as_deref())?;
            let as_cage = !mesh.polygonal() && self.settings.subdivision == Subdivision::Cage;
            let skeleton = models[o].skeleton.map(|k| &skels[k].1);
            let (skin, transform) = match (bound.get(&i), skeleton) {
                (Some(b), Some(sk)) => {
                    let (s, to_bind) =
                        sk.binding(&path, &self.skin_source(b), mesh.points.len())?;
                    (Some(s), to_bind)
                }
                _ => (None, self.to_model(&models[o], i)?),
            };
            let faces = self.face_materials(i, &mesh)?;
            let face_material = |f: usize| faces[f];
            for &(m, linear) in &holders[o] {
                let placing = Placing {
                    transform: linear * transform,
                    face_material: &face_material,
                    skin: skin.as_deref(),
                };
                if as_cage {
                    models[m].cage.add(&mesh, &placing, &path)?;
                } else {
                    models[m].welder.add(&mesh, &placing, &path)?;
                }
            }
        }

        let mut out = Vec::new();
        let mut names = std::collections::HashSet::new();
        let mut instance_out = Vec::new();
        for (mi, b) in models.into_iter().enumerate() {
            let copies = placed.remove(&mi);
            // Models that bake them hold all of a prototype's copies.
            if copies.as_ref().is_some_and(Vec::is_empty) {
                continue;
            }
            // The rest of the stage, a skeletal root with no skinned mesh or a prototype may be
            // empty.
            if (b.name.is_none() || b.skinned || b.prototype.is_some())
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
            if let (Some(n), Some(copies)) = (&b.name, copies) {
                instance_out.push((n.clone(), copies));
            }
            out.push(self.model(b)?);
        }
        if out.is_empty() {
            return Err("the stage has no triangles".into());
        }
        let mut skeletons = Vec::with_capacity(skels.len());
        for (path, sk, root, sites) in skels {
            let clips = self.clips(&path, &sk, root, &sites)?;
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
            warnings: std::mem::take(&mut self.warnings),
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
            if w.missing_uv1.contains(&m)
                && self.materials[m].textures.refs().any(|r| r.uv_set == 1)
            {
                return Err(format!(
                    "material {m} reads uv set 1, which a mesh drawn with it does not author"
                ));
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
            mesh: MeshSource::Data(Box::new(MeshData {
                positions: w.positions,
                normals: w.normals,
                tangents: None,
                uvs: std::iter::once(w.uvs)
                    .chain(self.settings.uv1.is_some().then_some(w.uvs1))
                    .collect(),
                indices,
                sections,
                skin: b.skinned.then_some(Skin {
                    joints: w.joints,
                    weights: w.weights,
                }),
                parts: None,
                min_exponent: None,
            })),
            skeleton: b.skeleton,
            settings,
            materials: slots,
        })
    }

    /// Prototype `root`'s model space from node `n`'s under it: the local transforms from
    /// `root`, or from a prim that resets the transform stack.
    fn to_prototype(&self, root: usize, n: usize) -> DMat4 {
        let (mut m, mut at) = (DMat4::IDENTITY, n);
        loop {
            let node = &self.nodes[at];
            m = node.local * m;
            match node.parent.filter(|_| at != root && !node.reset) {
                Some(p) => at = p,
                None => return self.root * m,
            }
        }
    }

    /// Model `b`'s space from node `n`'s. The local transforms up to the frame give it where
    /// the frame's world transform has no inverse; only that inverse reaches a prim that
    /// resets the transform stack.
    fn to_model(&self, b: &Building, n: usize) -> Result<DMat4, String> {
        if let Some(root) = b.prototype {
            return Ok(self.to_prototype(root, n));
        }
        let Some(frame) = b.frame else {
            return Ok(self.nodes[n].world);
        };
        let (mut m, mut at) = (DMat4::IDENTITY, n);
        while at != frame {
            let node = &self.nodes[at];
            m = node.local * m;
            match node.parent.filter(|_| !node.reset) {
                Some(p) => at = p,
                None => {
                    let world = self.nodes[frame].world;
                    if collapses(DMat3::from_mat4(world)) {
                        return Err(format!(
                            "{}: resets the transform stack under {}, which collapses an axis",
                            self.nodes[n].prim.path().as_str(),
                            self.nodes[frame].prim.path().as_str(),
                        ));
                    }
                    return Ok(self.root * world.inverse() * self.nodes[n].world);
                }
            }
        }
        Ok(self.root * m)
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
            models.push(Building::new(None, None, None));
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
            models.push(Building::new(Some(node.name.clone()), Some(i), None));
        }
        if let Some(s) = matched.iter().position(|m| !m) {
            return Err(format!(
                "model pattern {:?} matches no prim",
                self.settings.models[s].node
            ));
        }
        Ok(())
    }

    /// The prototype with root node `root`, made if it is new: its model holds the meshes its
    /// subtree draws, and instancers there are left to be placed by its copies.
    fn prototype(
        &self,
        root: usize,
        models: &mut Vec<Building>,
        instancing: &mut Instancing,
    ) -> Result<Prototype, String> {
        if let Some(p) = instancing.prototypes.get(&root) {
            return Ok(p.clone());
        }
        let node = &self.nodes[root];
        let taken = models
            .iter()
            .any(|m| m.name.as_deref() == Some(node.name.as_str()));
        let name = if taken {
            node.prim.path().as_str().trim_start_matches('/')
        } else {
            node.name.as_str()
        };
        let model = models.len();
        models.push(Building::new(Some(name.to_string()), None, None));
        models[model].prototype = Some(root);
        let mut nested = Vec::new();
        let mut k = root;
        while k < node.end {
            let at = &self.nodes[k];
            let drawn = !self.hidden(k, Some(root));
            if at.ty == "PointInstancer" {
                if drawn {
                    nested.push(k);
                }
                k = at.end;
                continue;
            }
            if at.ty == "Mesh" && drawn {
                if let Some(skel_root) = self.skel_root(k)
                    && self.binding(k, skel_root)?.is_some()
                {
                    return Err(format!(
                        "{}: a skinned mesh in a prototype",
                        at.prim.path().as_str()
                    ));
                }
                instancing.meshes.push((k, model));
            }
            k += 1;
        }
        let p = Prototype { model, nested };
        instancing.prototypes.insert(root, p.clone());
        Ok(p)
    }

    /// Records the copies of instancer `ins`, whose space `to_scene` takes to scene space, and
    /// of the instancers under its prototypes. `above` lists the instancers being placed.
    fn place(
        &self,
        ins: usize,
        to_scene: DMat4,
        models: &mut Vec<Building>,
        instancing: &mut Instancing,
        above: &mut Vec<usize>,
    ) -> Result<(), String> {
        let r = self.reader(&self.nodes[ins].prim);
        if above.contains(&ins) {
            return Err(format!("{}: the instancer is its own prototype", r.path()));
        }
        above.push(ins);
        let instancer = match instancing.instancers.get(&ins) {
            Some(i) => Rc::clone(i),
            None => {
                // A prototype that is inactive or not there has no copies.
                let mut protos = Vec::new();
                for path in r.targets("prototypes")? {
                    let root = self.by_path.get(path.as_str()).copied();
                    protos.push(match root {
                        Some(root) => Some((root, self.prototype(root, models, instancing)?)),
                        None => None,
                    });
                }
                let placed = self.placements(ins, protos.len())?;
                let i = Rc::new(Instancer { protos, placed });
                instancing.instancers.insert(ins, Rc::clone(&i));
                i
            }
        };
        for &(proto, local) in &instancer.placed {
            let Some((root, p)) = &instancer.protos[proto] else {
                continue;
            };
            let transform = to_scene * local;
            let copies = instancing.copies.entry(p.model).or_default();
            copies.push(transform);
            for &nested in &p.nested {
                let to_scene = transform * self.to_prototype(*root, nested);
                self.place(nested, to_scene, models, instancing, above)?;
            }
        }
        above.pop();
        Ok(())
    }

    /// Per instance of a `PointInstancer`: its prototype index and the instancer's space from
    /// the prototype's model.
    fn placements(&self, ins: usize, protos: usize) -> Result<Vec<(usize, DMat4)>, String> {
        let node = &self.nodes[ins];
        let r = self.reader(&node.prim);
        let path = r.path();
        let index = r.ints("protoIndices")?.unwrap_or_default();
        let count = index.len();
        let time = f64::from(self.time);
        let bracket = |name: &str| {
            node.prim
                .attribute(name)
                .bracketing_time_samples(self.time)
                .map_err(|e| format!("{path} {name}: {e}"))
        };
        // `name` at the lower sample of `base`, if it brackets the time at the same samples and
        // has one value per instance.
        let lined_up = |name: &str, base: Option<(f64, f64)>, width: usize| match base {
            Some((lower, _)) if bracket(name)? == base => Ok::<_, String>(
                r.at(lower)
                    .numbers(name)?
                    .filter(|v| v.len() == width * count),
            ),
            _ => Ok(None),
        };
        let orientation = self.orientation_attribute(&r)?;
        let moved = bracket("positions")?;
        let turned = bracket(orientation)?;
        let velocities = lined_up("velocities", moved, 3)?;
        let accelerations = match velocities {
            Some(_) => lined_up("accelerations", moved, 3)?,
            None => None,
        };
        let spins = lined_up("angularVelocities", turned, 3)?;
        // Without a rate to extrapolate by, values interpolate; with one, both hold.
        let held = |b: Option<(f64, f64)>| match b {
            Some((lower, _)) if velocities.is_some() || spins.is_some() => lower,
            _ => time,
        };
        let positions = r.at(held(moved)).numbers("positions")?.unwrap_or_default();
        let orientations = r
            .at(held(turned))
            .numbers(orientation)?
            .filter(|v| !v.is_empty());
        let scales = r.numbers("scales")?.filter(|v| !v.is_empty());
        let per_second = self.stage.time_codes_per_second();
        let since = |b: Option<(f64, f64)>| b.map_or(0.0, |b| (time - b.0) / per_second);
        let (moved_for, turned_for) = (since(moved), since(turned));
        let ids = r.ints("ids")?;
        let mut masked: std::collections::HashSet<i64> = r
            .ints("invisibleIds")?
            .unwrap_or_default()
            .into_iter()
            .collect();
        let inactive = node
            .prim
            .get_metadata::<sdf::Value>("inactiveIds")
            .map_err(|e| format!("{path} inactiveIds: {e}"))?;
        // A crate file holds a list op; text gives a plain list its own type.
        match inactive {
            Some(sdf::Value::Int64ListOp(op)) => masked.extend(op.flatten()),
            Some(v) => masked.extend(read::numbers(&v).iter().flatten().map(|&x| x as i64)),
            None => {}
        }
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
        let to_model = self.root.inverse();
        let mut out = Vec::with_capacity(count);
        for (i, &p) in index.iter().enumerate() {
            let id = ids
                .as_ref()
                .and_then(|ids| ids.get(i).copied())
                .unwrap_or(i as i64);
            if masked.contains(&id) {
                continue;
            }
            let proto = usize::try_from(p)
                .ok()
                .filter(|&p| p < protos)
                .ok_or_else(|| format!("{path}: protoIndex {p} of {protos} prototypes"))?;
            let vector = |v: &Option<Vec<f64>>| {
                v.as_ref()
                    .map_or(DVec3::ZERO, |v| DVec3::from_slice(&v[3 * i..]))
            };
            let velocity = vector(&velocities) + moved_for * vector(&accelerations) * 0.5;
            let t = DVec3::from_slice(&positions[3 * i..]) + moved_for * velocity;
            let q = orientations.as_ref().map_or(DQuat::IDENTITY, |o| {
                let spin = vector(&spins);
                let turn = (turned_for * spin.length()).to_radians();
                let by = DQuat::from_axis_angle(spin.normalize_or(DVec3::X), turn);
                by * read::quat(&o[4 * i..]).normalize()
            });
            let s = scales
                .as_ref()
                .map_or(DVec3::ONE, |s| DVec3::from_slice(&s[3 * i..]));
            out.push((
                proto,
                DMat4::from_scale_rotation_translation(s, q, t) * to_model,
            ));
        }
        Ok(out)
    }

    /// The attribute orientations come from: `orientationsf`, where its earliest sample (or
    /// default without samples) is not empty.
    fn orientation_attribute(&self, r: &PrimReader<'_>) -> Result<&'static str, String> {
        let name = "orientationsf";
        let times = r
            .prim
            .attribute(name)
            .time_sample_times()
            .map_err(|e| format!("{} {name}: {e}", r.path()))?;
        let first = times.first().map_or(f64::from(self.time), |&t| t);
        let empty = r.at(first).numbers(name)?.is_none_or(|v| v.is_empty());
        Ok(if empty { "orientations" } else { name })
    }

    /// The nearest `SkelRoot` at or above node `n`.
    fn skel_root(&self, n: usize) -> Option<usize> {
        std::iter::successors(Some(n), |&i| self.nodes[i].parent)
            .find(|&i| self.nodes[i].ty == "SkelRoot")
    }

    /// Node `m` or the nearest prim above it, up to `root`, that authors `attribute`; a prim
    /// above it counts only with `SkelBindingAPI`, as `UsdSkelCache` reads them.
    fn inherited(&self, m: usize, root: usize, attribute: &str) -> Result<Option<usize>, String> {
        let mut at = Some(m);
        while let Some(i) = at {
            let prim = &self.nodes[i].prim;
            let fail = |e: openusd::Error| format!("{} {attribute}: {e}", prim.path().as_str());
            if (i == m || prim.has_api_schema("SkelBindingAPI").map_err(fail)?)
                && prim
                    .attribute(attribute)
                    .has_authored_value()
                    .map_err(fail)?
            {
                return Ok(Some(i));
            }
            at = self.nodes[i].parent.filter(|_| i != root);
        }
        Ok(None)
    }

    /// How mesh `m` under `SkelRoot` `root` binds a skeleton, if it has joint influences and
    /// a bound `Skeleton`.
    fn binding(&self, m: usize, root: usize) -> Result<Option<Bound>, String> {
        let from = |attribute: &str| self.inherited(m, root, attribute);
        let (Some(indices), Some(weights)) = (
            from("primvars:skel:jointIndices")?,
            from("primvars:skel:jointWeights")?,
        ) else {
            return Ok(None);
        };
        let Some((skeleton, site)) = self.bound_skeleton(m)? else {
            return Ok(None);
        };
        let is_skeleton = self
            .by_path
            .get(skeleton.as_str())
            .is_some_and(|&n| self.nodes[n].ty == "Skeleton");
        if !is_skeleton {
            return Ok(None);
        }
        Ok(Some(Bound {
            skeleton,
            site,
            joints: from("skel:joints")?,
            indices,
            weights,
            geom_bind: from("primvars:skel:geomBindTransform")?,
        }))
    }

    /// The readers of a binding's attributes.
    fn skin_source(&self, b: &Bound) -> skel::SkinSource<'_> {
        let reader = |n: usize| self.reader(&self.nodes[n].prim);
        skel::SkinSource {
            joints: b.joints.map(reader),
            indices: reader(b.indices),
            weights: reader(b.weights),
            geom_bind: b.geom_bind.map(reader),
        }
    }

    /// The skeleton mesh `m` binds and the node that says so: `skel:skeleton` on it or its
    /// nearest ancestor.
    fn bound_skeleton(&self, m: usize) -> Result<Option<(sdf::Path, usize)>, String> {
        let mut at = Some(m);
        while let Some(n) = at {
            let r = self.reader(&self.nodes[n].prim);
            if let Some(t) = r.targets("skel:skeleton")?.into_iter().next() {
                return Ok(Some((t, n)));
            }
            at = self.nodes[n].parent;
        }
        Ok(None)
    }

    /// The animation bound at node `n`: the `skel:animationSource` there or on its nearest
    /// ancestor that authors one. A target that is deactivated binds nothing.
    fn animation_source(&self, n: usize) -> Result<Option<usize>, String> {
        let mut at = Some(n);
        while let Some(i) = at {
            let r = self.reader(&self.nodes[i].prim);
            if let Some(t) = r.targets("skel:animationSource")?.first() {
                return Ok(self.by_path.get(t.as_str()).copied());
            }
            at = self.nodes[i].parent;
        }
        Ok(None)
    }

    /// Clips of skeleton `path`: the `skel:animationSource` in force where it is bound (nodes
    /// `sites`) and on the skeleton, and every animation under its skeletal root, node `root`.
    fn clips(
        &self,
        path: &str,
        sk: &skel::Skel,
        root: usize,
        sites: &[usize],
    ) -> Result<Vec<(String, orr_pack::clip::Clip)>, String> {
        self.settings.clips.check()?;
        let per_second = self.stage.time_codes_per_second();
        let mut sources: Vec<usize> = Vec::new();
        for &n in sites.iter().chain(self.by_path.get(path)) {
            sources.extend(self.animation_source(n)?);
        }
        sources.extend(self.subtree(root).filter(|&i| {
            let node = &self.nodes[i];
            node.ty == "SkelAnimation" && !node.instanced
        }));
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
                    warnings: Vec::new(),
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
                        data: ImageData::Bytes(bytes),
                    });
                    image_of.insert(resolved, images.len() - 1);
                    Ok(images.len() - 1)
                };
                let (net, surface, displacement) =
                    shade::network(&self.stage, self.time, p, &mut load)?;
                let primvars: Vec<&str> = std::iter::once(self.settings.uv.as_str())
                    .chain(self.settings.uv1.as_deref())
                    .collect();
                let mapped = shading::map(&net, surface, displacement, &primvars)
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
        self.warnings.extend(mapped.warnings);
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
