//! USD import (`.usda`, `.usdc`, `.usdz`) through `openusd`: the composed stage's meshes as
//! one mesh per model, the UsdShade materials they bind, `PointInstancer`s as instance
//! records, and with UsdSkel a skeleton and its clips ([`skel`]).
//!
//! The stage composes its references, payloads (all loaded) and variants, with the
//! selections of [`UsdSettings::variants`] in a session layer; any composition error fails
//! the import. Values are read at [`UsdSettings::time`]. The stage's `upAxis` and
//! `metersPerUnit` (0.01 when unauthored, as USD's) turn stage space into scene space (Y
//! up, meters), and `metersPerUnit` is the length unit of materials. Inactive prims (and their
//! subtrees) are not read, and prims under a `class` or `over` are read only as prototypes. An
//! Imageable prim that is `invisible`, or has an effective purpose of `guide` or `proxy`, is not
//! drawn: invisibility holds for the subtree, and a prim takes the purpose of the nearest
//! Imageable prim at or above it that authors one.
//! Instanceable prims are expanded.
//!
//! Models, every one in scene units and axes:
//! - Each `SkelRoot` is a model named after it, in scene space (bind space), of the skinned
//!   meshes under it (those with joint influences and a bound skeleton), which must bind one
//!   skeleton, cooked as `<asset>/<root>/skeleton` with clips
//!   `<asset>/<root>/clip/<animation>`. Its other meshes are static and cook with the rest
//!   of the stage, as do meshes outside a `SkelRoot`, whose bindings mean nothing. A skinned
//!   mesh under a `SkelRoot` in a prototype fails: a copy has no pose of its own.
//! - What moves by an xform sampled over time, where a sample differs from the first: every mesh
//!   drawn under such a prim. In the rest of the stage each rig of them is a model of its own in
//!   scene space, named after its first joint (after its path, or `node<i>` for its node, if
//!   another model has the name or it would be `mesh`), unless it is all the stage holds; a selected model or a prototype keeps its own, with one
//!   skeleton. Prims become joints as glTF nodes do ([`crate::skin`]), and a rig cooks as rigid
//!   parts, one per joint, part 0 on a still joint, unless a subdivision cage in it takes its
//!   joints by skinning. Its one clip is `xform`, over the samples of every prim on the stage
//!   so that such clips keep time. A `Skeleton` under
//!   such a prim keeps its model, with the prims above it as joints, and each of its clips
//!   holds them too, beside an `xform` clip without a `SkelAnimation`.
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
//!   transform with it applied, as model `<model>/baked/<k>`, placed by translation alone. A
//!   displacement is a world length, so one that bakes into a mesh makes any transform that
//!   changes lengths no placement of it either.
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
//! `doubleSided` mesh take a double-sided copy, `<path>/double_sided`. Meshes that differ in
//! what they hold for a material's primvar readers take a copy each, `<path>/primvars/<k>`.

mod mesh;
mod read;
mod shade;
mod skel;

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::rc::Rc;

use openusd::{sdf, usd};
use orr_math::{Affine3, DMat3, DMat4, DQuat, DVec3, DVec4};
use orr_pack::material::Material;
use orr_pack::skeleton::{NO_PARENT, SkeletonInfo, Trs};
use serde::Deserialize;

use crate::import::SceneSubd;
use crate::import::{
    NodeSelect, Placed, Scene, SceneMaterial, SceneModel, SceneSkeleton, collapses, placements,
};
use crate::mesh::{MeshData, MeshSettings, MeshSource, PartData, Pivots, SectionData, Skin};
use crate::patch::{self, MaterialPatch, MeshPatch};
use crate::shading;
use crate::skin::{self, Channel, ClipSettings, Rig, Rigging};
use crate::subd::SubdSettings;
use crate::texture::material::{ImageData, ImageFile, ImageTile, TextureRules};
use crate::texture::tiles::{UDIM, udim_cell};
use crate::{Error, Result};

use mesh::{CageBuilder, FaceFrame, Faces, FrameUvs, MeshFacts, MeshPrim, Placing, Welder};
use read::{PrimReader, Primvar, Reads, Recorder};

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

impl UsdSettings {
    /// The primvars of the uv sets, by set.
    fn uv_sets(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.uv.as_str()).chain(self.uv1.as_deref())
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

/// The clip of a stage's xforms, which have no animation prim to name them.
const XFORM_CLIP: &str = "xform";

/// A joint's inverse bind from its bind pose: the identity where that collapses an axis, which
/// leaves the joint's own space as bind space.
fn inverse_bind(m: DMat4) -> DMat4 {
    match collapses(DMat3::from_mat4(m)) {
        true => DMat4::IDENTITY,
        false => m.inverse(),
    }
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
    /// The times its xform is sampled at, where they differ; empty for a prim that holds still.
    times: Vec<f64>,
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
    /// Its skeleton among those of rigs.
    rig_skeleton: Option<usize>,
    /// Bind position per joint of a model of rigid parts.
    pivots: Option<Vec<[f32; 3]>>,
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
            rig_skeleton: None,
            pivots: None,
            welder: Welder::default(),
            cage: CageBuilder::default(),
        }
    }
}

/// The scene materials a mesh prim binds.
struct Bindings {
    /// Its own, or its nearest ancestor's.
    base: usize,
    /// Each `GeomSubset` that binds faces: its node and material.
    subsets: Vec<(usize, usize)>,
}

impl Bindings {
    fn materials(&self) -> impl Iterator<Item = usize> {
        std::iter::once(self.base).chain(self.subsets.iter().map(|s| s.1))
    }
}

/// The primvars a material's tangent frame reads: PS `frame:tangentsPrimvarName`,
/// `frame:binormalsPrimvarName` and `frame:stPrimvarName`.
struct Frame {
    tangents: String,
    binormals: String,
    /// The uv set it follows.
    set: usize,
    /// Or this primvar outside the uv sets, on the meshes that hold it.
    st: Option<String>,
    /// The primvars its material reads through the vertex color stream.
    vertex: Vec<shading::VertexRead>,
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

/// The models being assembled, the models their meshes go to and the meshes drawn.
struct Assembly<'a> {
    models: &'a mut Vec<Building>,
    holders: &'a mut Vec<Vec<(usize, DMat4)>>,
    meshes: &'a mut Vec<(usize, usize)>,
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

/// The shading of a material prim.
struct Shading {
    net: shading::Network,
    surface: usize,
    displacement: Option<usize>,
    /// [`shading::Network::primvar_reads`].
    reads: Vec<String>,
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
    /// Per scene material, the primvars of its tangent frame.
    frames: Vec<Frame>,
    /// Scene material by material prim path, double-sided, and what the mesh holds for the
    /// material's primvar readers.
    material_of: HashMap<(Option<String>, bool, String), usize>,
    /// Shading networks by material prim path.
    networks: HashMap<String, Rc<Shading>>,
    /// Scene materials so far by material prim path and double-sided.
    copies_of: HashMap<(String, bool), usize>,
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
            frames: Vec::new(),
            material_of: HashMap::new(),
            warnings: Vec::new(),
            networks: HashMap::new(),
            copies_of: HashMap::new(),
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
        let up = self.stage.stage_metadata("upAxis");
        let up = match up.map_err(|e| e.to_string())? {
            Some(sdf::Value::Token(t)) => t.as_str().to_string(),
            Some(sdf::Value::String(s)) => s,
            _ => "Y".into(),
        };
        let axis = match up.as_str() {
            "Y" => DMat4::IDENTITY,
            // Z up: +Z becomes +Y, +Y becomes -Z.
            "Z" => DMat4::from_cols(DVec4::X, DVec4::NEG_Z, DVec4::Y, DVec4::W),
            a => return Err(format!("upAxis {a}")),
        };
        Ok(DMat4::from_scale(DVec3::splat(self.meters_per_unit()?)) * axis)
    }

    /// The stage's `metersPerUnit`.
    fn meters_per_unit(&self) -> Result<f64, String> {
        let meters = self
            .stage
            .stage_metadata("metersPerUnit")
            .map_err(|e| e.to_string())?;
        Ok(meters
            .and_then(|v| read::numbers(&v))
            .map_or(0.01, |v| v[0]))
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
            let times = r.xform_times()?;
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
                times,
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

        // Every mesh drawn, with the model it goes to.
        let mut meshes: Vec<(usize, usize)> = (0..n)
            .filter(|&i| self.nodes[i].ty == "Mesh" && !self.hidden(i, None))
            .filter_map(|i| Some((i, owner[i]?)))
            .collect();
        meshes.append(&mut instancing.meshes);
        // What each mesh binds and whether it is a cage, and the models that bake a height into a
        // mesh of theirs.
        let mut bindings = Vec::with_capacity(meshes.len());
        let mut baking = std::collections::HashSet::new();
        for &(i, o) in &meshes {
            let prim = self.nodes[i].prim.clone();
            let facts = MeshFacts::read(&self.reader(&prim))?;
            let as_cage = !facts.polygonal && self.settings.subdivision == Subdivision::Cage;
            let bound_here = self.bindings(i, facts.double_sided)?;
            let displaced = bound_here
                .materials()
                .any(|m| self.materials[m].textures.displacement.is_some());
            let settings = || self.mesh_settings(models[o].name.as_deref());
            if !as_cage && displaced && settings().displace_static.is_some() {
                baking.insert(o);
            }
            bindings.push((bound_here, as_cage));
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
            let (direct, baked) =
                placements(&transforms, baking.contains(&m)).map_err(|e| format!("{name}: {e}"))?;
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

        let order: Vec<usize> = (0..n).collect();
        let animated: Vec<bool> = self.nodes.iter().map(|d| !d.times.is_empty()).collect();
        let parents: Vec<Option<usize>> = self
            .nodes
            .iter()
            .map(|d| d.parent.filter(|_| !d.reset))
            .collect();
        let mut rigs: HashMap<usize, skel::PrimRig> = HashMap::new();
        let mut rig_skeletons = Vec::new();
        if animated.contains(&true) {
            let mut held: Vec<Vec<usize>> = vec![Vec::new(); models.len()];
            meshes.iter().for_each(|&(i, o)| held[o].push(i));
            let moved = |&i: &usize| skin::chain(&parents, i, None).any(|n| animated[n]);
            for (o, held) in held.iter().enumerate() {
                if models[o].skinned || !held.iter().any(moved) {
                    continue;
                }
                self.rig_model(
                    o,
                    held,
                    (&order, &animated, &parents),
                    Assembly {
                        models: &mut models,
                        holders: &mut holders,
                        meshes: &mut meshes,
                    },
                    (&mut rigs, &mut rig_skeletons),
                )?;
            }
        }
        // One skeleton per skeletal root, from what its skinned meshes bind: its prim path,
        // the skeleton, the root's node and the prims that bind it.
        let mut skels: Vec<(String, skel::Skel, usize, Vec<usize>)> = Vec::new();
        for (m, b) in models.iter_mut().enumerate() {
            if !b.skinned || b.rig_skeleton.is_some() {
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
                    let mut built = skel::Skel::build(
                        &self.reader(&self.nodes[node].prim),
                        self.nodes[node].world,
                        self.root,
                    )?;
                    self.hang(&mut built, node, &animated, &parents)?;
                    let root = b.root.expect("skeletal roots have prims");
                    skels.push((path.as_str().to_string(), built, root, Vec::new()));
                    skels.len() - 1
                }
            };
            skels[k].3.extend(sites);
            b.skeleton = Some(k);
        }
        for b in &mut models {
            if let Some(k) = b.rig_skeleton {
                b.skeleton = Some(skels.len() + k);
            }
        }
        // A model's meshes carry tangents if one of them does, which each must know as it lands
        // in the model. That takes which materials a mesh binds, and none of its arrays.
        for (&(i, o), (bound_here, as_cage)) in meshes.iter().zip(&bindings) {
            let settled = holders[o]
                .iter()
                .all(|h| models[h.0].welder.carries_tangents);
            if !as_cage && !settled {
                let prim = self.nodes[i].prim.clone();
                let r = self.reader(&prim);
                let mut carries = false;
                for m in bound_here.materials() {
                    carries = carries
                        || self.materials[m].tangent_set().is_some()
                        || r.holds_primvar(&self.frames[m].tangents)?;
                }
                if carries {
                    for &(m, _) in &holders[o] {
                        models[m].welder.carries_tangents = true;
                    }
                }
            }
        }
        for (&(i, o), (bound_here, _)) in meshes.iter().zip(&bindings) {
            let prim = self.nodes[i].prim.clone();
            let r = self.reader(&prim);
            let path = r.path().to_string();
            let mesh = MeshPrim::read(&r, &self.settings.uv, self.settings.uv1.as_deref())?;
            let as_cage = !mesh.polygonal() && self.settings.subdivision == Subdivision::Cage;
            let faces = self.face_materials(bound_here, &mesh)?;
            let mut used = faces.clone();
            used.sort_unstable();
            used.dedup();
            let r = self.reader(&prim);
            let mut primvars: HashMap<String, Option<Primvar>> = HashMap::new();
            let mut read_primvars: HashMap<String, Option<Primvar>> = HashMap::new();
            for &m in &used {
                let frame = &self.frames[m];
                for name in [&frame.tangents, &frame.binormals] {
                    if !primvars.contains_key(name) {
                        primvars.insert(name.clone(), r.frame_primvar(name)?);
                    }
                }
                if let Some(name) = &frame.st
                    && !primvars.contains_key(name)
                {
                    primvars.insert(name.clone(), r.primvar(name, 2, None)?);
                }
                for read in &frame.vertex {
                    if !read_primvars.contains_key(&read.name) {
                        let held = r.wide_primvar(&read.name)?;
                        if let Some(p) = held.as_ref().filter(|p| p.width() > 4 && !as_cage) {
                            self.warnings.push(format!(
                                "{path}: primvar {} is {} wide; a reader takes the first 4",
                                read.name,
                                p.width()
                            ));
                        }
                        let missing = held
                            .as_ref()
                            .map_or(0, |p| p.missing(mesh.domain(p.interp)));
                        if missing > 0 && !as_cage {
                            self.warnings.push(format!(
                                "{path}: primvar {} holds no value for {missing} elements of the \
                                 mesh; its reader's fallback stands in for them",
                                read.name
                            ));
                        }
                        read_primvars.insert(read.name.clone(), held);
                    }
                }
                let reads = self.materials[m].tangent_set().is_some();
                if let Some(st) = &frame.st
                    && reads
                    && !as_cage
                    && primvars[st].is_none()
                {
                    self.warnings.push(format!(
                        "{path}: a material's tangents follow {st}, which the mesh does not \
                         author; they follow uv set {}",
                        frame.set
                    ));
                }
                if as_cage && primvars[&frame.tangents].is_some() {
                    // `mesh.h`: "Normals should not be authored on a subdivision mesh, since
                    // subdivision algorithms define their own normals." The frame around them is
                    // the limit surface's too.
                    self.warnings.push(format!(
                        "{path}: {} on a subdivision cage is not read; tangents follow the limit \
                         surface",
                        frame.tangents
                    ));
                }
            }
            let colored = used.iter().any(|&m| !self.frames[m].vertex.is_empty());
            if colored {
                if as_cage {
                    self.warnings.push(format!(
                        "{path}: a subdivision cage holds no vertex colors; what a material reads \
                         of them reads 1"
                    ));
                } else {
                    for &(m, _) in &holders[o] {
                        models[m].welder.carries_colors = true;
                    }
                }
            }
            let skeleton = models[o].skeleton.and_then(|k| skels.get(k)).map(|s| &s.1);
            let rig = rigs.get(&o);
            let (skin, transform, part) = match (bound.get(&i), skeleton, rig) {
                (Some(b), Some(sk), _) => {
                    let (s, to_bind) =
                        sk.binding(&path, &self.skin_source(b), mesh.points.len())?;
                    (Some(s), to_bind, 0)
                }
                (_, _, Some(r)) => {
                    let (j, to_bind) = self.attach(r, &models[o], i)?;
                    let skin = (!r.rigid).then(|| {
                        let one = ([j as u8; 4], [255, 0, 0, 0]);
                        vec![one; mesh.points.len()]
                    });
                    (skin, to_bind, j)
                }
                _ => (None, self.to_model(&models[o], i)?, 0),
            };
            let face_material = |f: usize| faces[f];
            let surface = (!as_cage).then(|| Faces::new(&mesh));
            let carries = holders[o]
                .iter()
                .any(|h| models[h.0].welder.carries_tangents);
            let corners = match &surface {
                Some(s) if carries => {
                    let frame = |face: usize| {
                        let f = &self.frames[faces[face]];
                        let held = |name: &String| primvars[name].as_ref();
                        // Clusters of a section reading none say set 0, for a copy's material.
                        let reads = self.materials[faces[face]].tangent_set().is_some();
                        let named = f.st.as_ref().filter(|_| reads);
                        let named = named.and_then(|n| Some((n.as_str(), held(n)?)));
                        let set = FrameUvs::Set(if reads { f.set } else { 0 });
                        let uvs = named.map_or(set, |(n, p)| FrameUvs::Primvar(n, p));
                        FaceFrame {
                            uvs,
                            tangents: held(&f.tangents),
                            binormals: held(&f.binormals),
                        }
                    };
                    Some(mesh::corner_tangents(&mesh, s, &frame, &path)?)
                }
                _ => None,
            };
            let color = |face: usize, corner: usize, point: usize| {
                let mut c = [1.0; 4];
                for read in &self.frames[faces[face]].vertex {
                    let held = read_primvars[&read.name].as_ref();
                    let v = held.and_then(|p| p.at(point, face, corner));
                    for k in 0..usize::from(read.width) {
                        let x = v.map_or(f64::from(read.fallback[k]), |v| v[k]);
                        c[usize::from(read.first) + k] = (x / f64::from(read.scale)) as f32;
                    }
                }
                c
            };
            let object_space = used
                .iter()
                .any(|&m| self.materials[m].material.object_normals != 0);
            for &(m, linear) in &holders[o] {
                let baked = DMat3::from_mat4(linear * transform);
                if object_space && !baked.abs_diff_eq(DMat3::IDENTITY, 1e-6) {
                    self.warnings.push(format!(
                        "{path}: its object space normal maps don't turn with the transform \
                         baked into its model"
                    ));
                }
                let placing = Placing {
                    // A skeleton takes what a mesh in its bind pose cannot, but it poses no
                    // part 0.
                    transform: if rig.is_some_and(|r| !r.rigid || part != 0) {
                        transform
                    } else {
                        linear * transform
                    },
                    face_material: &face_material,
                    skin: skin.as_deref(),
                    part,
                    color: &color,
                };
                match &surface {
                    None => models[m].cage.add(&mesh, &placing, &path)?,
                    Some(s) => {
                        let welder = &mut models[m].welder;
                        welder.add(&mesh, s, corners.as_deref(), &placing, &path)?;
                    }
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
        skeletons.extend(rig_skeletons);
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

    /// Mesh settings of the model named `name`.
    fn mesh_settings(&self, name: Option<&str>) -> MeshSettings {
        let mut s = self.settings.mesh.clone();
        patch::apply_matching(
            &self.settings.mesh_overrides,
            name.unwrap_or_default(),
            &mut s,
            MeshPatch::apply,
        );
        s
    }

    /// A built model's mesh, one section per scene material.
    fn model(&mut self, b: Building) -> Result<SceneModel, String> {
        let settings = self.mesh_settings(b.name.as_deref());
        let w = b.welder;
        let mut indices = Vec::new();
        let mut sections = Vec::new();
        let mut slots = Vec::new();
        for (&m, tris) in &w.groups {
            if tris.is_empty() {
                continue;
            }
            // PS `UsdPrimvarReader`: `fallback` is "returned if geometry fetch failed".
            let material = &self.materials[m];
            let frame = material.tangent_set().is_some()
                && (self.frames[m].set == 1
                    || material.material.coat_frame_uv == Material::FRAME_UV1);
            if w.missing_uv1.contains(&m)
                && (frame || material.textures.refs().any(|r| r.uv_set == 1))
            {
                self.warnings.push(format!(
                    "material {m} reads uv set 1, which a mesh drawn with it does not author; \
                     it reads uv 0"
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
                tangents: w.carries_tangents.then_some(w.tangents),
                uvs: std::iter::once(w.uvs)
                    .chain(self.settings.uv1.is_some().then_some(w.uvs1))
                    .collect(),
                colors: w.carries_colors.then_some(w.colors),
                moves: None,
                indices,
                sections,
                skin: b.skinned.then_some(Skin {
                    joints: w.joints,
                    weights: w.weights,
                }),
                parts: b.pivots.map(|pivots| PartData {
                    values: w.parts,
                    count: pivots.len() as u16,
                    by_section: false,
                    pivots: Pivots::PerValue(pivots),
                    joints: None,
                }),
                min_exponent: None,
            })),
            skeleton: b.skeleton,
            settings,
            materials: slots,
        })
    }

    /// Finds what moves among model `o`'s meshes `held`, and gives it skeletons: on the model
    /// where it is the whole model, a selection or a prototype, else in models of its own.
    fn rig_model(
        &self,
        o: usize,
        held: &[usize],
        (order, animated, parents): (&[usize], &[bool], &[Option<usize>]),
        Assembly {
            models,
            holders,
            meshes,
        }: Assembly<'_>,
        (rigs, skeletons): (&mut HashMap<usize, skel::PrimRig>, &mut Vec<SceneSkeleton>),
    ) -> Result<(), String> {
        let in_scene = vec![true; parents.len()];
        // A prototype's model ends at its root.
        let mut cut = parents.to_vec();
        if let Some(root) = models[o].prototype {
            cut[root] = None;
        }
        let no_skin = |_: usize| None;
        let mut cages = std::collections::HashSet::new();
        for &i in held {
            let facts = MeshFacts::read(&self.reader(&self.nodes[i].prim))?;
            if !facts.polygonal && self.settings.subdivision == Subdivision::Cage {
                cages.insert(i);
            }
        }
        let one = models[o].frame.is_some() || models[o].prototype.is_some();
        let rigging = Rigging {
            parent: &cut,
            order,
            in_scene: &in_scene,
            animated,
            skin_of: &no_skin,
            parts_from_skin: false,
            // A cage skins the one rig of such a model, which settles where its root hangs.
            parts_from_attribute: one && !cages.is_empty(),
        };
        let found = Rig::find(&rigging, held, models[o].frame, one);
        let whole = found.len() == 1 && found[0].meshes.len() == held.len();
        let mut moved: HashMap<usize, usize> = HashMap::new();
        for mut rig in found {
            let m = if one || whole {
                o
            } else {
                let top = &self.nodes[rig.top];
                let path = top.prim.path().as_str().trim_start_matches('/');
                // `mesh` is the entry of the whole stage's mesh.
                let free = |s: &&str| {
                    *s != "mesh" && !models.iter().any(|m| m.name.as_deref() == Some(*s))
                };
                let name = [top.name.as_str(), path]
                    .into_iter()
                    .find(free)
                    .map_or_else(|| format!("node{}", rig.top), str::to_string);
                models.push(Building::new(Some(name), None, None));
                holders.push(vec![(models.len() - 1, DMat4::IDENTITY)]);
                let m = models.len() - 1;
                moved.extend(rig.meshes.iter().map(|&i| (i, m)));
                m
            };
            rig.rigid = !rig.meshes.iter().any(|i| cages.contains(i));
            let mut channels = Vec::new();
            for (k, &(h, linear)) in holders[m].clone().iter().enumerate() {
                let built = self.prim_rig(&rig, &models[h], linear)?;
                models[h].skinned = !rig.rigid;
                let joints = 0..built.info.joint_count();
                models[h].pivots = rig.rigid.then(|| {
                    let at = |j: usize| built.binds[j].w_axis.truncate().as_vec3().to_array();
                    joints.map(at).collect()
                });
                models[h].rig_skeleton = Some(skeletons.len());
                if k == 0 {
                    channels = self.xform_channels(&built.prims)?;
                }
                let clip = self.xform_clip(&built.info, &channels)?;
                skeletons.push(SceneSkeleton {
                    prefix: models[h].name.clone().unwrap_or_default(),
                    info: built.info.clone(),
                    clips: clip
                        .map(|c| (XFORM_CLIP.to_string(), c))
                        .into_iter()
                        .collect(),
                });
                if k == 0 {
                    rigs.insert(m, built);
                }
            }
        }
        for h in meshes.iter_mut().filter(|h| h.1 == o) {
            h.1 = moved.get(&h.0).copied().unwrap_or(o);
        }
        Ok(())
    }

    /// The skeleton of `rig` in the space of model `b`, with `after` applied to that space.
    fn prim_rig(&self, rig: &Rig, b: &Building, after: DMat4) -> Result<skel::PrimRig, String> {
        let nodes: Vec<usize> = (0..self.nodes.len()).filter(|&n| rig.joints[n]).collect();
        let first = u16::from(rig.still);
        let count = nodes.len() + usize::from(first);
        let most = if rig.rigid {
            orr_pack::skeleton::MAX_JOINTS
        } else {
            orr_pack::skeleton::MAX_SKINNED_JOINTS
        };
        if count > most {
            return Err(format!(
                "{} moves {count} joints with the prims between them; at most {most}",
                self.nodes[rig.top].prim.path().as_str()
            ));
        }
        let root = match (rig.above, rig.frame) {
            (Some(n), _) => self.to_model(b, n)?,
            (None, Some(f)) if rig.outside => {
                let world = self.nodes[f].world;
                if collapses(DMat3::from_mat4(world)) {
                    return Err(format!(
                        "{}: frames a model with prims that reset the transform stack, and \
                         collapses an axis",
                        self.nodes[f].prim.path().as_str()
                    ));
                }
                self.root * world.inverse() * self.root
            }
            (None, _) => self.root,
        };
        let joint_of: HashMap<usize, u16> = nodes
            .iter()
            .enumerate()
            .map(|(k, &n)| (n, k as u16 + first))
            .collect();
        let mut info = SkeletonInfo {
            root: Affine3::from_mat4((after * root).as_mat4()),
            parents: Vec::with_capacity(count),
            rest: Vec::with_capacity(count),
            inverse_binds: Vec::with_capacity(count),
            names: Vec::with_capacity(count),
        };
        let mut binds = Vec::with_capacity(count);
        if rig.still {
            let ib = inverse_bind(root);
            info.parents.push(NO_PARENT);
            info.rest.push(Trs::IDENTITY);
            info.inverse_binds.push(Affine3::from_mat4(ib.as_mat4()));
            info.names.push(String::new());
            binds.push(inverse_bind(ib));
        }
        for &n in &nodes {
            let node = &self.nodes[n];
            let path = node.prim.path().as_str();
            let rest = self.to_model(b, n)?;
            let parent = node.parent.filter(|_| !node.reset);
            let parent = parent.and_then(|p| joint_of.get(&p).copied());
            info.parents.push(parent.unwrap_or(NO_PARENT));
            info.rest.push(skel::prim_trs(path, node.local)?);
            let ib = inverse_bind(rest);
            info.inverse_binds.push(Affine3::from_mat4(ib.as_mat4()));
            info.names.push(node.name.clone());
            binds.push(inverse_bind(ib));
        }
        Ok(skel::PrimRig {
            info,
            prims: nodes.iter().map(|&n| (n, joint_of[&n])).collect(),
            joint_of,
            binds,
            still: rig.still.then_some(0),
            rigid: rig.rigid,
        })
    }

    /// The joint mesh `i` of model `b` follows, its nearest prim that has one, else the still
    /// joint, and bind space from its points.
    fn attach(&self, r: &skel::PrimRig, b: &Building, i: usize) -> Result<(u16, DMat4), String> {
        let (mut m, mut at) = (DMat4::IDENTITY, Some(i));
        while let Some(n) = at.filter(|&n| Some(n) != b.frame) {
            if let Some(&j) = r.joint_of.get(&n) {
                return Ok((j, r.binds[usize::from(j)] * m));
            }
            let node = &self.nodes[n];
            m = node.local * m;
            at = node
                .parent
                .filter(|_| !node.reset && Some(n) != b.prototype);
        }
        let still = r.still.ok_or("a mesh with no joint above it")?;
        Ok((still, self.to_model(b, i)?))
    }

    /// The channels of `prims`' xforms: keys at their samples and between them, every
    /// [`ClipSettings::max_rate`].
    fn xform_channels(&self, prims: &[(usize, u16)]) -> Result<Vec<Channel>, String> {
        let per_second = self.stage.time_codes_per_second();
        let rate = f64::from(self.settings.clips.max_rate);
        let mut channels = Vec::new();
        for &(n, joint) in prims {
            let node = &self.nodes[n];
            let Some(&last) = node.times.last() else {
                continue;
            };
            let mut times = Vec::new();
            for w in node.times.windows(2) {
                let steps = ((w[1] - w[0]) / per_second * rate).ceil().max(1.0) as u32;
                let at = |k: u32| w[0] + (w[1] - w[0]) * f64::from(k) / f64::from(steps);
                times.extend((0..steps).map(at));
            }
            times.push(last);
            let path = node.prim.path().as_str();
            let r = self.reader(&node.prim);
            let mut keys = Vec::with_capacity(times.len());
            for &t in &times {
                let local = r.at(t).local_transform()?.0;
                keys.push(skel::prim_trs(&format!("{path} at {t}"), local)?);
            }
            let seconds: Vec<f32> = times.iter().map(|t| (t / per_second) as f32).collect();
            let keyed = skel::trs_channels(usize::from(joint), &seconds, &keys);
            channels.extend(keyed.map_err(|e| format!("{path}: {e}"))?);
        }
        Ok(channels)
    }

    /// The start and length in seconds of every xform on the stage, as [`skin::span`] gives them.
    fn xform_span(&self) -> (f64, f64) {
        let per_second = self.stage.time_codes_per_second();
        let seconds = |t: f64| f64::from((t / per_second) as f32);
        let held = self.nodes.iter().filter(|d| !d.times.is_empty());
        let (first, last) = held.fold((f64::MAX, f64::MIN), |(a, b), d| {
            (a.min(d.times[0]), b.max(d.times[d.times.len() - 1]))
        });
        (seconds(first), seconds(last) - seconds(first))
    }

    /// The clip of `channels`, what xforms make up alone, or `None` without any. It runs over
    /// every xform, so the clips of separate skeletons keep time.
    fn xform_clip(
        &self,
        info: &SkeletonInfo,
        channels: &[Channel],
    ) -> Result<Option<orr_pack::clip::Clip>, String> {
        self.settings.clips.check()?;
        let span = self.xform_span();
        let clips = &self.settings.clips;
        Ok((!channels.is_empty()).then(|| skin::resample(info, channels, span, clips)))
    }

    /// Hangs skeleton `sk` of prim `node` from the prims that move it, if any do: every prim
    /// from the highest of them to its own.
    fn hang(
        &self,
        sk: &mut skel::Skel,
        node: usize,
        animated: &[bool],
        parents: &[Option<usize>],
    ) -> Result<(), String> {
        let chain: Vec<usize> = skin::chain(parents, node, None).collect();
        let Some(far) = chain.iter().rposition(|&n| animated[n]) else {
            return Ok(());
        };
        let root = match chain.get(far + 1) {
            Some(&above) => self.nodes[above].world,
            None => self.root,
        };
        let mut prims = Vec::with_capacity(far + 1);
        for &n in chain[..=far].iter().rev() {
            let d = &self.nodes[n];
            let local = skel::prim_trs(d.prim.path().as_str(), d.local)?;
            prims.push((n, d.name.clone(), local, inverse_bind(d.world)));
        }
        sk.hang(prims, root)
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
        let xforms = self.xform_channels(&sk.prims)?;
        let mut clips = Vec::new();
        for i in sources {
            let node = &self.nodes[i];
            if let Some(c) = sk.clip(&node.prim, per_second, &xforms, &self.settings.clips)? {
                clips.push((node.name.clone(), c));
            }
        }
        if clips.is_empty()
            && let Some(c) = self.xform_clip(&sk.info, &xforms)?
        {
            clips.push((XFORM_CLIP.to_string(), c));
        }
        let mut count: HashMap<String, usize> = HashMap::new();
        for (n, _) in &clips {
            *count.entry(n.clone()).or_default() += 1;
        }
        Ok(clips
            .into_iter()
            .enumerate()
            .map(|(k, (n, c))| (if count[&n] == 1 { n } else { k.to_string() }, c))
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

    /// The scene materials mesh prim `i` binds.
    fn bindings(&mut self, i: usize, double_sided: bool) -> Result<Bindings, String> {
        let base = self.bound_material(Some(i))?;
        let base = self.material(base, double_sided, i)?;
        let subsets: Vec<usize> = self
            .subtree(i)
            .filter(|&k| self.nodes[k].parent == Some(i) && self.nodes[k].ty == "GeomSubset")
            .collect();
        let mut bound = Vec::new();
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
            bound.push((k, self.material(Some(m), double_sided, i)?));
        }
        Ok(Bindings {
            base,
            subsets: bound,
        })
    }

    /// Scene material per face of `mesh`, which binds `bindings`.
    fn face_materials(&self, bindings: &Bindings, mesh: &MeshPrim) -> Result<Vec<usize>, String> {
        let mut faces = vec![bindings.base; mesh.counts.len()];
        for &(k, m) in &bindings.subsets {
            let r = self.reader(&self.nodes[k].prim);
            let path = r.path().to_string();
            let indices = r.ints("indices")?.unwrap_or_default();
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

    /// The shading of a material prim, read the first time it is asked for.
    fn network(&mut self, p: &sdf::Path) -> Result<Rc<Shading>, String> {
        if let Some(s) = self.networks.get(p.as_str()) {
            return Ok(Rc::clone(s));
        }
        let images = &mut self.images;
        let image_of = &mut self.image_of;
        let reads = Rc::clone(&self.reads);
        let mut load = |a: &sdf::AssetPath| -> usize {
            let name = a.asset_path().to_string();
            let key = a.resolved_path().unwrap_or(&name).to_string();
            if let Some(&i) = image_of.get(&key) {
                return i;
            }
            let read = |name: String, resolved: &str| match Recorder::new(Rc::clone(&reads))
                .read(resolved)
            {
                Ok(bytes) => ImageFile {
                    name,
                    data: ImageData::Bytes(bytes),
                },
                Err(e) => ImageFile::unread(name, e.to_string()),
            };
            let tiles = a
                .resolved_path()
                .and_then(|r| Some(reads.borrow().tiles(r)?.to_vec()));
            let image = match (tiles, a.resolved_path()) {
                (Some(found), _) => {
                    let mut tiles = Vec::with_capacity(found.len());
                    for (n, path) in found {
                        images.push(read(name.replace(UDIM, &n.to_string()), &path));
                        tiles.push(ImageTile {
                            cell: udim_cell(n),
                            image: images.len() - 1,
                        });
                    }
                    ImageFile {
                        name,
                        data: ImageData::Tiles(tiles),
                    }
                }
                (None, Some(resolved)) => read(name, resolved),
                (None, None) => ImageFile::unread(name, "it does not resolve".into()),
            };
            images.push(image);
            image_of.insert(key, images.len() - 1);
            images.len() - 1
        };
        let (net, surface, displacement) = shade::network(&self.stage, self.time, p, &mut load)?;
        let reads = net.primvar_reads();
        let s = Rc::new(Shading {
            net,
            surface,
            displacement,
            reads,
        });
        self.networks.insert(p.as_str().to_string(), Rc::clone(&s));
        Ok(s)
    }

    /// The tangent frame of a material prim with `material`: the primvars PS's `frame:*` inputs
    /// name, by default `tangents`, `binormals` and the uv set `material` follows.
    fn frame(&self, prim: Option<&sdf::Path>, material: &mut Material) -> Result<Frame, String> {
        let mut frame = Frame {
            tangents: "tangents".into(),
            binormals: "binormals".into(),
            set: usize::from(material.tangent_set()),
            st: None,
            vertex: Vec::new(),
        };
        let Some(p) = prim else {
            return Ok(frame);
        };
        let prim = self.stage.prim(p.clone()).map_err(|e| e.to_string())?;
        let r = self.reader(&prim);
        if let Some(n) = r.token("inputs:frame:tangentsPrimvarName")? {
            frame.tangents = n;
        }
        if let Some(n) = r.token("inputs:frame:binormalsPrimvarName")? {
            frame.binormals = n;
        }
        if let Some(n) = r.token("inputs:frame:stPrimvarName")? {
            match self.settings.uv_sets().position(|s| s == n) {
                Some(0) => material.frame_uv = Material::FRAME_UV0,
                Some(_) => material.frame_uv = Material::FRAME_UV1,
                None => material.frame_uv = Material::FRAME_OTHER,
            }
            frame.st = (material.frame_uv == Material::FRAME_OTHER).then_some(n);
            frame.set = match frame.st {
                Some(_) => 0,
                None => usize::from(material.tangent_set()),
            };
        }
        Ok(frame)
    }

    /// What mesh prim `mesh` holds for primvar `name`. A constant primvar inherits down
    /// namespace. A cage holds no stream, so nothing is scaled to fit one.
    fn primvar(&self, mesh: usize, name: &str) -> Result<shading::Primvar, String> {
        let set = self.settings.uv_sets().position(|s| s == name);
        let facts = MeshFacts::read(&self.reader(&self.nodes[mesh].prim))?;
        let cage = !facts.polygonal && self.settings.subdivision == Subdivision::Cage;
        let mut at = Some(mesh);
        while let Some(i) = at {
            match (self.reader(&self.nodes[i].prim).whole_primvar(name)?, set) {
                (shading::Primvar::Missing, _) => {}
                (_, Some(set)) if i == mesh => return Ok(shading::Primvar::Uv(set as u8)),
                (shading::Primvar::Varying { .. }, _) if i != mesh => {}
                (shading::Primvar::Varying { width, .. }, _) if cage => {
                    return Ok(shading::Primvar::Varying { width, scale: None });
                }
                (held, _) => return Ok(held),
            }
            at = self.nodes[i].parent;
        }
        Ok(shading::Primvar::Missing)
    }

    /// The scene material of a material prim (`None`: the default material) on mesh prim
    /// `mesh`.
    fn material(
        &mut self,
        prim: Option<sdf::Path>,
        double_sided: bool,
        mesh: usize,
    ) -> Result<usize, String> {
        let shaded = prim.as_ref().map(|p| self.network(p)).transpose()?;
        let mut reads = Vec::new();
        for name in shaded.iter().flat_map(|s| &s.reads) {
            reads.push((name.clone(), self.primvar(mesh, name)?));
        }
        let key = (
            prim.as_ref().map(|p| p.as_str().to_string()),
            double_sided,
            format!("{reads:?}"),
        );
        if let Some(&m) = self.material_of.get(&key) {
            return Ok(m);
        }
        let (mut mapped, name) = match (&prim, &shaded) {
            (Some(p), Some(s)) => {
                let meters_per_unit = self.meters_per_unit()?;
                let primvars: Vec<&str> = self.settings.uv_sets().collect();
                let images = &mut self.images;
                let image_of = &mut self.image_of;
                let mut add = |image: ImageFile| {
                    *image_of.entry(image.name.clone()).or_insert_with(|| {
                        images.push(image);
                        images.len() - 1
                    })
                };
                let mut host = shading::Host {
                    meters_per_unit,
                    primvars: &primvars,
                    reads: &reads,
                    any_mesh: false,
                    add: &mut add,
                };
                let mapped = shading::map(&s.net, s.surface, s.displacement, &mut host)
                    .map_err(|e| format!("material {p}: {e}"))?;
                let path = p.as_str().trim_start_matches('/');
                let copy = self
                    .copies_of
                    .entry((path.to_string(), double_sided))
                    .or_default();
                *copy += 1;
                let name = match *copy {
                    1 => path.to_string(),
                    k => format!("{path}/primvars/{}", k - 1),
                };
                (mapped, name)
            }
            _ => (
                shading::Mapped {
                    material: Default::default(),
                    textures: Default::default(),
                    vertex: Vec::new(),
                    warnings: Vec::new(),
                },
                "default".to_string(),
            ),
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
        let mut frame = self.frame(prim.as_ref(), &mut mapped.material)?;
        frame.vertex = std::mem::take(&mut mapped.vertex);
        shading::fold_stream_scales(&mut mapped.material, &frame.vertex);
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
            back: None,
        });
        self.frames.push(frame);
        let m = self.materials.len() - 1;
        self.material_of.insert(key, m);
        Ok(m)
    }
}
