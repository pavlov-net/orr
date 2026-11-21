//! USD import (`.usda`, `.usdc`, `.usdz`) through `openusd`: the composed stage's meshes as
//! one mesh per model, the UsdShade materials they bind, `PointInstancer`s as instance
//! records, with UsdSkel a skeleton and its clips ([`skel`]), and its UsdLux lights as layout
//! light records of their partitions ([`lux`]), a prototype's once per still copy.
//!
//! The stage composes its references, payloads (all loaded) and variants, with the
//! selections of [`UsdSettings::variants`] in a session layer; any composition error fails
//! the import. It knows the schemas of `openusd_schemas`, which give unauthored attributes their
//! fallbacks and say which types derive from which. Values are read at [`UsdSettings::time`].
//! The stage's `upAxis` and `metersPerUnit` (0.01 when unauthored, as USD's) turn stage space
//! into scene space (Y up, meters), and `metersPerUnit` is the length unit of displacements. A
//! prim's transform is its `UsdGeomXformable` op stack; a prim that is not Xformable has none.
//! Inactive prims (and their subtrees) are not read, and prims under a `class` or `over` are
//! read only as prototypes. An Imageable prim that is `invisible`, or has an effective purpose
//! of `guide` or `proxy`, or of `render` under a `renderVisibility` of `invisible`
//! (`UsdGeomVisibilityAPI`), is not drawn: invisibility holds for the subtree, and a prim takes
//! the purpose (`ComputePurposeInfo`) and render visibility of the nearest Imageable prim at or
//! above it that authors one. Instanceable prims are expanded.
//!
//! Models, every one in scene units and axes; a model with a skeleton stands in the skeleton's
//! space, at its origin in the model's ([`crate::skin::RigSpace`]), and its copies take it:
//! - Each `SkelRoot` is a model named after it, in scene space (bind space), of the skinned
//!   meshes under it (those with joint influences and a bound skeleton, as `UsdSkelCache`
//!   resolves them through `SkelBindingAPI`; a prim under the root that is not Imageable keeps
//!   every mesh below it unskinned), with its skeleton and clips per animation. `schemas.dox`
//!   makes each `skel:skeleton` binding a skeleton instance of its own, so with several under a
//!   root each is a model of the root's derived by the binding
//!   prim's path under it ([`Derived::Binding`]), whichever skeleton it binds. Its
//!   other meshes are static and cook with the rest of the stage, as do meshes outside a
//!   `SkelRoot`, whose bindings mean nothing. A prototype's skinned meshes make it such a model,
//!   its other meshes on a still joint, a binding past the first a model of the prototype's
//!   derived by the binding prim's path under it placed with it, and what a sampled xform moves
//!   among its other meshes a model [`Derived::Moved`] placed with it. One skeleton serves every
//!   copy, and its pages hold only that model's copies, which a palette per copy poses. A
//!   skeleton outside the prototype is seen from the first copy, following the common space
//!   UsdSkel's scene index defines: that copy stands at the skeleton's world pose, and the others
//!   stand off from it as they do from the first.
//! - What moves by an xform sampled over time, where a sample differs from the first: every mesh
//!   drawn under such a prim. In the rest of the stage each rig of them is a model of its own in
//!   scene space, named after its first joint (after its path, or its node, if a model other
//!   than a rig of that joint has the name), unless it is all its partition's rest holds; one
//!   prim heads a rig in each partition whose rest it moves, named for a partition held by a
//!   group by [`Derived::Partition`]. A selected model or a prototype keeps its own, with one
//!   skeleton. Prims become joints as glTF nodes do ([`crate::skin`]), and a rig cooks as rigid
//!   parts, one per joint, part 0 on a still joint, unless a subdivision cage in it takes its
//!   joints by skinning. Its one clip is `xform`. Every clip of the stage, this one and each
//!   `SkelAnimation`'s, runs over the stage's one timeline, from the first sample of a moving
//!   xform, moving instancer or `SkelAnimation` to the last, so that its clips keep time. A
//!   `Skeleton` under
//!   such a prim keeps its model, with the prims above it as joints, and each of its clips
//!   holds them too, beside an `xform` clip without a `SkelAnimation`.
//! - An instancer whose own attributes move its instances cooks to animated pages: per interval
//!   between the samples of `positions`, orientations, `scales`, `protoIndices`, `ids` and
//!   `invisibleIds`, a row per instance that carries it to the next ([`InstanceState`]). A copy
//!   is a slot of one prototype and index, as USD interpolates by index, and has no row where a
//!   sample has no such instance, so `ids` and counts step at the lower sample, unlike what
//!   interpolates between them. A rate that survives alignment holds all three at their lower
//!   samples; without one, values interpolate unless samples differ in length. A time whose
//!   arrays disagree in length, or with a `protoIndices` entry past the prototypes, holds no
//!   instances, as USD refuses that call alone, and the instancer stands again at the next good
//!   sample. USD defines no non-finite value and no format here carries one, so that instance
//!   alone is skipped there. Both warn, at any time and for an instancer that holds still as well.
//!   Before the first sample and after the last, rates extrapolate and all else clamps.
//!   `motion:velocityScale`
//!   is deprecated and not read. Every copy shares its prototype's model, and a kernel poses
//!   them each frame by [`orr_pack::instance::motion_transform`]; the page's clock runs on the
//!   timeline of the stage's clips. An instancer under one that moves this way stays where the
//!   time read puts it, with a warning.
//! - What a sampled xform moves above an instancer takes its copies along: the rig of those prims
//!   holds the instancer, and its records stand in the bind space of the rig's nearest joint, on
//!   pages of their own that name the joint (`InstancePageHeader::frame_skeleton`), so one
//!   palette entry poses them all. Copies still share their prototype's model and BLAS, and
//!   skinned copies its skeleton, with a palette per copy. A rig of nothing else is a skeleton and
//!   its clips with no model of its own. An instancer under a prim that moves inside a prototype
//!   stays where the time read puts it, with a warning: its copies would need a frame per copy
//!   of the outer prototype.
//! - Each prototype of a `PointInstancer` is a model named after its root prim (after its path
//!   if another model has the name, and [`Derived::Prototype`] of that if one has the path too).
//!   The root's own transform is part of the model, as is
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
//!   transform with it applied, as a model [`Derived::Baked`] of it, placed by translation alone. A
//!   displacement is a world length, so one that bakes into a mesh makes any transform that
//!   changes lengths no placement of it either. Moving copies of such a mesh follow the same
//!   rule while what stretches them stands still between samples ([`crate::import::kept_lengths`]):
//!   each stretch cooks its own baked model, and the rows move that rigidly. One that
//!   changes between samples has no mesh to go into, as only run-time displacement keeps a length
//!   through it: those copies stay on the model, with a warning, and the height stretches.
//! - Without [`UsdSettings::models`], the rest of the stage is one model per partition
//!   ([`UsdSettings::groups`]), in scene space. Every other model in scene space that no
//!   instancer places belongs to a partition too: a `SkelRoot`'s model to its prim's, a rig's
//!   to the partition whose rest it moves.
//!   With it, each prim whose name contains a pattern (and has no selected ancestor) is a
//!   model holding its subtree, framed by the prim, and placed once where the prim stands, in
//!   the prim's partition, as an instancer places a prototype: what moves the prim, at or above
//!   it, is a rig its copy hangs from, and a transform that is no placement bakes a model
//!   beside the selected one. Meshes outside every model are not cooked.
//!
//! Meshes are [`mesh`] lists. A mesh with a `subdivisionScheme` other than `none` (USD's
//! fallback is `catmullClark`) is a subdivision cage: [`Subdivision::Hull`] cooks its
//! control faces as they are, [`Subdivision::Cage`] cooks the model's cages as one
//! subdivision cage entry (Catmull-Clark only; every cage of a model shares its
//! `interpolateBoundary`, `faceVaryingLinearInterpolation` and `triangleSubdivisionRule`).
//! Its uv sets and the primvars its materials read through the vertex color stream come out of
//! the tessellator by their interpolation ([`mesh::CageBuilder`]).
//! Faces take the material of their `GeomSubset` of family `materialBind`, else of the
//! mesh: the `Material` `ComputeBoundMaterial` resolves for the `full` purpose, through direct
//! and collection bindings, `bindMaterialAs` and the binding's target, counting bindings on
//! prims without `MaterialBindingAPI` as USD does by default, which one warning of the stage
//! counts; a binding to another type of prim binds nothing, with a warning. Materials map as
//! [`crate::shading`] lists and are named by their prim path ([`MaterialBase::Path`]),
//! [`MaterialBase::Default`] for unbound faces; faces of a `doubleSided` mesh take a
//! double-sided copy ([`Variant::DoubleSided`]). Meshes that differ in what they hold for a
//! material's primvar readers take a copy each ([`Variant::Primvars`]).

mod lux;
mod mesh;
mod read;
mod shade;
mod skel;

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use openusd::usd::PathRule;
use openusd::{ar, sdf, tf, usd, vt};
use openusd_schemas::geom::{
    self, ImageableExt, ImageableSchema, PointBasedSchema, PointInstancerSchema, Purpose,
    PurposeVisibility, SubsetSchema, UpAxis, Visibility,
};
use openusd_schemas::shade as usd_shade;
use openusd_schemas::skel as usd_skel;
use orr_math::{Affine3, DMat3, DMat4, DQuat, DVec3, DVec4};
use orr_pack::instance::MotionRow;
use orr_pack::layout::{Attribute, AttributeValue, Group, LightSource};
use orr_pack::material::Material;
use orr_pack::names::{Derived, MaterialBase, MaterialName, ModelBase, ModelName, Variant};
use orr_pack::skeleton::{NO_PARENT, SkeletonInfo, Trs};
use serde::Deserialize;

use crate::import::SceneSubd;
use crate::import::{
    NodeSelect, Placed, PlacedModel, Scene, SceneMaterial, SceneModel, ScenePartition,
    SceneSkeleton, collapses, kept_lengths, placements,
};
use crate::instances::InstanceMotion;
use crate::mesh::{MeshData, MeshSettings, MeshSource, PartData, Pivots, SectionData, Skin};
use crate::patch::{self, MaterialPatch, MeshPatch};
use crate::shading;
use crate::shading_model::ShadingModel;
use crate::skin::{self, Channel, ClipSettings, ClipSource, Rig, RigSpace, Rigging};
use crate::subd::SubdSettings;
use crate::texture::material::{ImageData, ImageFile, ImageTile, TextureRules};
use crate::texture::tiles::udim_cell;
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
    /// Collections whose members cook into partitions of their own, in the layout
    /// ([`orr_pack::layout`]): group `k` of the layout is `groups[k]`.
    pub groups: Vec<GroupSpec>,
}

/// A collection that makes a group of the layout.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupSpec {
    /// Absolute path of the prim that applies the collection, such as `/World/Layers/Night`.
    pub prim: String,
    /// The collection's instance name: `members` for `collection:members`.
    pub collection: String,
    /// Namespaces of the prim's authored attributes the group carries, such as `dataLayer`.
    #[serde(default)]
    pub attributes: Vec<String>,
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
            groups: Vec::new(),
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

/// The stage's one timeline, from the first sample of what moves on it (a sampled xform, an
/// instancer whose instances move, a `SkelAnimation`) to the last: every clip and animated page
/// of the stage runs on it, from its start. USD holds a value before its first sample and after
/// its last, so a source that starts late or ends early holds there.
#[derive(Clone, Copy)]
pub(super) struct Timeline {
    /// Time codes of its ends.
    first: f64,
    last: f64,
    per_second: f64,
}

impl Timeline {
    /// Seconds from its start to time code `t`.
    pub(super) fn seconds(&self, t: f64) -> f64 {
        (t - self.first) / self.per_second
    }

    /// The span clips resample over, from its start: `(0, length)` in seconds.
    pub(super) fn span(&self) -> (f64, f64) {
        (0.0, self.seconds(self.last))
    }
}

/// A joint's inverse bind from its bind pose: the identity where that collapses an axis, which
/// leaves the joint's own space as bind space.
fn inverse_bind(m: DMat4) -> DMat4 {
    match collapses(DMat3::from_mat4(m)) {
        true => DMat4::IDENTITY,
        false => m.inverse(),
    }
}

/// A group attribute's value, its numbers as [`read::integers`] and [`read::numbers`] read
/// them; `None` for a value of no attribute value type.
fn attribute_value(v: vt::Value) -> Option<AttributeValue> {
    use AttributeValue as A;
    use vt::Value as V;
    fn elements<const N: usize>(v: &vt::Value) -> Option<Vec<[f64; N]>> {
        Some(read::numbers(v)?.as_chunks::<N>().0.to_vec())
    }
    fn element<const N: usize>(v: &vt::Value) -> Option<[f64; N]> {
        read::numbers(v)?.try_into().ok()
    }
    // USD's integer vectors are of `int`, which an `f64` holds exactly.
    fn ints<const N: usize>(v: &vt::Value) -> Option<Vec<[i32; N]>> {
        let wide = elements::<N>(v)?;
        Some(wide.into_iter().map(|e| e.map(|x| x as i32)).collect())
    }
    fn quats(v: &vt::Value) -> Option<Vec<DQuat>> {
        Some(elements::<4>(v)?.iter().map(|q| read::quat(q)).collect())
    }
    let one = |v: Option<Vec<f64>>| v.map(|v| v[0]);
    Some(match &v {
        V::Bool(b) => A::Bool(*b),
        V::BoolVec(b) => A::BoolArray(b.clone()),
        V::Uchar(_) | V::Int(_) | V::Uint(_) | V::Int64(_) => A::Int(read::integers(&v)?[0]),
        V::UcharVec(_) | V::IntVec(_) | V::UintVec(_) | V::Int64Vec(_) => {
            A::IntArray(read::integers(&v)?)
        }
        V::Uint64(u) => A::Uint64(*u),
        V::Uint64Vec(u) => A::Uint64Array(u.clone()),
        V::Half(_) | V::Float(_) | V::Double(_) => A::Float(one(read::numbers(&v))?),
        V::TimeCode(t) => A::Float(t.value()),
        V::HalfVec(_) | V::FloatVec(_) | V::DoubleVec(_) => A::FloatArray(read::numbers(&v)?),
        V::TimeCodeVec(t) => A::FloatArray(t.iter().map(|t| t.value()).collect()),
        V::Vec2h(_) | V::Vec2f(_) | V::Vec2d(_) => A::Float2(element(&v)?),
        V::Vec3h(_) | V::Vec3f(_) | V::Vec3d(_) => A::Float3(element(&v)?),
        V::Vec4h(_) | V::Vec4f(_) | V::Vec4d(_) => A::Float4(element(&v)?),
        V::Vec2hVec(_) | V::Vec2fVec(_) | V::Vec2dVec(_) => A::Float2Array(elements(&v)?),
        V::Vec3hVec(_) | V::Vec3fVec(_) | V::Vec3dVec(_) => A::Float3Array(elements(&v)?),
        V::Vec4hVec(_) | V::Vec4fVec(_) | V::Vec4dVec(_) => A::Float4Array(elements(&v)?),
        V::Vec2i(_) => A::Int2(ints(&v)?[0]),
        V::Vec3i(_) => A::Int3(ints(&v)?[0]),
        V::Vec4i(_) => A::Int4(ints(&v)?[0]),
        V::Vec2iVec(_) => A::Int2Array(ints(&v)?),
        V::Vec3iVec(_) => A::Int3Array(ints(&v)?),
        V::Vec4iVec(_) => A::Int4Array(ints(&v)?),
        V::Quath(_) | V::Quatf(_) | V::Quatd(_) => A::Quat(quats(&v)?[0]),
        V::QuathVec(_) | V::QuatfVec(_) | V::QuatdVec(_) => A::QuatArray(quats(&v)?),
        V::Matrix2d(m) => A::Matrix2(m.0),
        V::Matrix3d(m) => A::Matrix3(m.0),
        V::Matrix4d(m) => A::Matrix4(m.0),
        V::Matrix2dVec(m) => A::Matrix2Array(m.iter().map(|m| m.0).collect()),
        V::Matrix3dVec(m) => A::Matrix3Array(m.iter().map(|m| m.0).collect()),
        V::Matrix4dVec(m) => A::Matrix4Array(m.iter().map(|m| m.0).collect()),
        V::String(t) => A::String(t.clone()),
        V::StringVec(t) => A::StringArray(t.clone()),
        V::Token(t) => A::Token(t.as_str().to_owned()),
        V::TokenVec(t) => A::TokenArray(t.iter().map(|t| t.as_str().to_owned()).collect()),
        V::AssetPath(a) => A::Asset(a.authored_path.clone()),
        V::AssetPathVec(a) => A::AssetArray(a.iter().map(|a| a.authored_path.clone()).collect()),
        V::PathExpression(e) => A::PathExpression(e.to_string()),
        V::PathExpressionVec(e) => {
            A::PathExpressionArray(e.iter().map(ToString::to_string).collect())
        }
        _ => return None,
    })
}

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
    /// Its transform stack, whose `times` are empty for a prim that holds still.
    stack: read::Stack,
    /// What it and the prims above it say of whether it is drawn.
    shown: Shown,
}

/// What a prim and the prims above it say of whether it is drawn: `UsdGeomImageable`'s
/// visibility, purpose and `UsdGeomVisibilityAPI` render visibility, which only Imageable prims
/// state. A prim that is not drawn may still be read, as a skeleton or animation.
#[derive(Clone, Copy, Default)]
struct Shown {
    /// `invisible` here or above.
    invisible: bool,
    purpose: geom::PurposeInfo,
    /// The nearest authored `renderVisibility`, which only VisibilityAPI states.
    render: Option<PurposeVisibility>,
}

impl Shown {
    /// Not drawn: invisible, of purpose `guide` or `proxy`, or of purpose `render` with render
    /// visibility `invisible`.
    fn hidden(self) -> bool {
        self.invisible
            || match self.purpose.purpose {
                Purpose::Default => false,
                Purpose::Guide | Purpose::Proxy => true,
                Purpose::Render => self.render == Some(PurposeVisibility::Invisible),
            }
    }
}

/// One model being assembled.
struct Building {
    /// [`ModelName::WHOLE`] for the part of the stage no instancer places that partition 0
    /// holds, [`ModelBase::Partition`] for another partition's.
    name: ModelName,
    /// The node that frames the model, in scene units and axes; scene space without one.
    frame: Option<usize>,
    /// Holds skinned meshes, in bind space.
    skinned: bool,
    /// The node of the prim a skeletal root model holds.
    root: Option<usize>,
    /// The root node of the prototype the model holds.
    prototype: Option<usize>,
    /// The instancer of its first copy, and scene space from the model at that copy.
    first_copy: Option<(usize, DMat4)>,
    skeleton: Option<usize>,
    /// Its skeleton among those of rigs.
    rig_skeleton: Option<usize>,
    /// Bind position per joint of a model of rigid parts.
    pivots: Option<Vec<[f32; 3]>>,
    /// [`SceneModel::partition`].
    partition: Option<usize>,
    /// The top prim of the rig the model was made for, out of the rest of the stage.
    rig_top: Option<usize>,
    welder: Welder,
    cage: CageBuilder,
}

impl Building {
    fn new(
        name: ModelName,
        frame: Option<usize>,
        root: Option<usize>,
        partition: Option<usize>,
    ) -> Self {
        Self {
            name,
            frame,
            partition,
            rig_top: None,
            skinned: root.is_some(),
            root,
            prototype: None,
            first_copy: None,
            skeleton: None,
            rig_skeleton: None,
            pivots: None,
            welder: Welder::default(),
            cage: CageBuilder::default(),
        }
    }

    /// The model of the part of the stage no instancer places that partition `p` holds.
    fn rest(p: usize) -> Self {
        let name = match p {
            0 => ModelName::WHOLE,
            p => ModelName::new(ModelBase::Partition(p as u32)),
        };
        Self::new(name, None, None, Some(p))
    }
}

/// The scene materials a mesh prim binds.
struct Bindings {
    /// The mesh's.
    base: usize,
    /// Each `GeomSubset` of family `materialBind` that holds faces: its path, the faces it holds and
    /// their material.
    subsets: Vec<(String, Vec<i32>, usize)>,
}

impl Bindings {
    fn materials(&self) -> impl Iterator<Item = usize> {
        std::iter::once(self.base).chain(self.subsets.iter().map(|s| s.2))
    }
}

/// The primvars a material's tangent frame reads: PS `frame:tangentsPrimvarName`,
/// `frame:binormalsPrimvarName` and `frame:stPrimvarName`.
#[derive(Clone)]
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

/// How a mesh under a `SkelRoot` binds a skeleton.
struct Bound {
    skeleton: sdf::Path,
    /// Where `skel:skeleton` is authored.
    site: usize,
    /// The binding properties it inherits.
    query: usd_skel::SkinningQuery,
    /// Its joint indices and weights at the time read, one run for every point or, constant,
    /// one for all.
    influences: usd_skel::JointInfluences,
}

/// The models being assembled, the models their meshes go to and the meshes drawn.
struct Assembly<'a> {
    models: &'a mut Vec<Building>,
    holders: &'a mut Vec<Vec<(usize, DMat4)>>,
    meshes: &'a mut Vec<(usize, usize)>,
}

/// How far past its largest value the stream keeps room for a cage's primvar: a patch near an
/// extraordinary point approximates the limit surface, which stays in the hull of the cage, and
/// can pass it. A power of two more leaves every mantissa as it is.
const CAGE_COLOR_REACH: f64 = 2.0;

/// `UsdTimeCode::SafeStep`: past rounding at any time code a stage holds.
const SAFE_STEP: f64 = f64::EPSILON * 1e7;

/// Where an attribute's samples stand around a time ([`Importer::samples_around`]).
#[derive(Copy, Clone, PartialEq)]
struct Around {
    /// What it reads there, without interpolating.
    sample: f64,
    lower: f64,
    upper: f64,
}

/// An instance from a time until the next sample: [`MotionRow`] in double precision, in the
/// instancer's space, with rates in seconds.
#[derive(Copy, Clone, PartialEq)]
struct InstanceState {
    /// Its index in the instancer's arrays.
    index: usize,
    /// Index into the instancer's prototypes.
    proto: usize,
    position: DVec3,
    velocity: DVec3,
    acceleration: DVec3,
    rotation: DQuat,
    /// Degrees a second, its direction the axis.
    spin: DVec3,
    scale: DVec3,
    growth: DVec3,
}

impl InstanceState {
    /// Where it stands at the time it was read.
    fn transform(&self) -> DMat4 {
        DMat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    fn is_finite(&self) -> bool {
        let vectors = [
            self.position,
            self.velocity,
            self.acceleration,
            self.spin,
            self.scale,
            self.growth,
        ];
        self.rotation.as_quat().is_finite() && vectors.iter().all(|v| v.as_vec3().is_finite())
    }

    fn holds_still(&self) -> bool {
        [self.velocity, self.acceleration, self.spin, self.growth]
            .iter()
            .all(|&v| v == DVec3::ZERO)
    }

    /// Its row, the position from `anchor`.
    fn row(&self, anchor: DVec3) -> MotionRow {
        let three = |v: DVec3| v.as_vec3().to_array();
        MotionRow {
            position: three(self.position - anchor),
            velocity: three(self.velocity),
            acceleration: three(self.acceleration),
            rotation: self.rotation.normalize().as_quat().to_array(),
            spin: three(self.spin),
            scale: three(self.scale),
            growth: three(self.growth),
        }
    }
}

/// An attribute's values from a time on: the lower sample, and to interpolate, the upper one with
/// how far the time stands between them and the seconds they span.
struct Sampled {
    lower: Vec<f64>,
    upper: Option<(Vec<f64>, f64, f64)>,
}

/// The instances of an instancer that move by its own attributes.
struct Motion {
    /// Time codes each interval starts at.
    times: Vec<f64>,
    /// Prototype and index of each instance, whatever its id: a copy of one prototype.
    slots: Vec<(usize, usize)>,
    /// Per interval a row per slot, its position from the slot's `rest`.
    rows: Vec<MotionRow>,
    /// Per slot, how it stands at the time the stage is read, else at its first interval.
    rest: Vec<InstanceState>,
}

/// Where [`Importer::place`] puts an instancer's copies.
struct Placer<'a> {
    models: &'a mut Vec<Building>,
    instancing: &'a mut Instancing,
    /// The instancers being placed.
    above: &'a mut Vec<usize>,
    /// The joint they hang from.
    frame: Option<RigJoint>,
    /// The partition of the instancer placed from the stage.
    partition: usize,
}

/// The joint of a rig: the model that holds the rig, and the joint.
type RigJoint = (usize, u16);

/// Whether model `m` is named by `base` or derives from a model it names.
fn taken(m: &Building, base: &ModelBase) -> bool {
    m.name.base() == base
}

/// Gives model `m` its `k`th baked model, which holds its meshes with `linear` applied.
fn bake_model(
    models: &mut Vec<Building>,
    holders: &mut [Vec<(usize, DMat4)>],
    m: usize,
    k: usize,
    linear: DMat4,
) {
    holders[m].push((models.len(), linear));
    let name = models[m].name.with(Derived::Baked(k as u32));
    let (prototype, skel_root) = (models[m].prototype, models[m].root);
    let (first_copy, partition) = (models[m].first_copy, models[m].partition);
    let baked = models.push_mut(Building::new(name, models[m].frame, None, partition));
    baked.prototype = prototype;
    baked.root = skel_root;
    baked.first_copy = first_copy;
}

/// Where copies of a model go: the joint they hang from and the partition of
/// [`Importer::partitions`] that holds them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Dest {
    frame: Option<RigJoint>,
    partition: usize,
}

/// Placements of a model, and how they move over time.
struct Copies {
    dest: Dest,
    placed: Vec<Placed>,
    motion: Option<Arc<InstanceMotion>>,
}

/// Moving copies of `model`, which instancer `instancer` places.
struct Animated {
    model: usize,
    instancer: usize,
    dest: Dest,
    placed: Vec<Placed>,
    motion: Arc<InstanceMotion>,
}

/// The baked models of the stage's models ([`bake_model`]): one per model, joint and linear
/// transform, whichever partitions hold their copies.
#[derive(Default)]
struct Bakes {
    of: HashMap<(usize, Option<RigJoint>, [u64; 16]), usize>,
    /// Per model, how many it has.
    count: HashMap<usize, usize>,
}

impl Bakes {
    /// Model `m`'s baked model for copies hanging from `frame` that take `linear`.
    fn model(
        &mut self,
        models: &mut Vec<Building>,
        holders: &mut [Vec<(usize, DMat4)>],
        m: usize,
        frame: Option<RigJoint>,
        linear: DMat4,
    ) -> usize {
        let Self { of, count } = self;
        // Adding zero turns -0.0 into 0.0, as `placements` tells transforms apart.
        let key = linear.to_cols_array().map(|x| (x + 0.0).to_bits());
        *of.entry((m, frame, key)).or_insert_with(|| {
            let k = count.entry(m).or_default();
            let baked = models.len();
            bake_model(models, holders, m, *k, linear);
            *k += 1;
            baked
        })
    }
}

/// One prototype of the stage.
#[derive(Clone)]
struct Prototype {
    model: usize,
    /// Models of its skeleton instances past the first, placed with it.
    also: Vec<usize>,
    /// Instancers in its subtree, each placed by every copy of it.
    nested: Vec<usize>,
    /// Lights it draws, outside its instancers.
    lights: Vec<usize>,
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
    /// Placements per model and where they go.
    copies: BTreeMap<(usize, Dest), Vec<DMat4>>,
    /// What placing them warned of.
    warnings: Vec<String>,
    animated: Vec<Animated>,
    /// Meshes the prototypes draw, with their models.
    meshes: Vec<(usize, usize)>,
    /// How the skinned ones among them bind a skeleton.
    bound: HashMap<usize, Bound>,
    /// Lights the prototypes draw: per copy, the light's node, scene space from its space, and
    /// the partition that holds it.
    lights: Vec<(usize, DMat4, usize)>,
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
    /// By instancer node, those whose instances move.
    motions: HashMap<usize, Motion>,
    timeline: Timeline,
    images: Vec<ImageFile>,
    image_of: HashMap<String, usize>,
    materials: Vec<SceneMaterial>,
    /// Copies of materials whose object space maps read through a baked transform, by material
    /// and the bits of the frame.
    object_frames: HashMap<(usize, [[u32; 3]; 3]), usize>,
    /// Per scene material, the primvars of its tangent frame.
    frames: Vec<Frame>,
    /// Scene material by material prim path, double-sided, and what the mesh holds for the
    /// material's primvar readers.
    material_of: HashMap<(Option<String>, bool, String), usize>,
    /// Shading networks by material prim path.
    networks: HashMap<String, Rc<Shading>>,
    /// Scene materials so far by material prim path and double-sided.
    copies_of: HashMap<(String, bool), usize>,
    /// The layout's groups, and its partitions by the ascending groups that hold them, the
    /// first held by none.
    groups: Vec<Group>,
    partitions: Vec<Vec<u32>>,
    /// Per node, its partition.
    partition_of: Vec<usize>,
    /// Material binding resolutions so far.
    binding_cache: usd_shade::BindingCache,
    /// The prims that bind a material without `MaterialBindingAPI`, which warn together.
    unapplied: std::collections::BTreeSet<String>,
    /// The color spaces of the prims read so far.
    spaces: usd::ColorSpaceCache,
    /// The skinnable prims of the `SkelRoot` nodes populated so far.
    skel_cache: RefCell<(usd_skel::Cache, std::collections::HashSet<usize>)>,
    /// What reading skeleton bindings warned of.
    skel_warnings: RefCell<Vec<String>>,
    warnings: Vec<String>,
}

impl<'s> Importer<'s> {
    fn open(path: &Path, settings: &'s UsdSettings) -> Result<Self, String> {
        let reads = Rc::new(RefCell::new(Reads::default()));
        reads.borrow_mut().session = read::session_layer(&settings.variants)?;
        let root = path.to_str().ok_or("the path is not UTF-8")?;
        let stage = usd::Stage::builder()
            .schema_registry(openusd_schemas::schema_registry())
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
                .selections()
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
        let time = settings.time.unwrap_or_else(|| {
            if stage.has_authored_time_code_range() {
                stage.start_time_code()
            } else {
                0.0
            }
        });
        let per_second = stage.time_codes_per_second();
        Ok(Self {
            settings,
            stage,
            reads,
            time: usd::TimeCode::new(time),
            root: DMat4::IDENTITY,
            nodes: Vec::new(),
            by_path: HashMap::new(),
            motions: HashMap::new(),
            timeline: Timeline {
                first: f64::MAX,
                last: f64::MIN,
                per_second,
            },
            images: Vec::new(),
            image_of: HashMap::new(),
            materials: Vec::new(),
            frames: Vec::new(),
            material_of: HashMap::new(),
            warnings: Vec::new(),
            networks: HashMap::new(),
            object_frames: HashMap::new(),
            copies_of: HashMap::new(),
            groups: Vec::new(),
            partitions: Vec::new(),
            partition_of: Vec::new(),
            binding_cache: usd_shade::BindingCache::default(),
            unapplied: std::collections::BTreeSet::new(),
            spaces: usd::ColorSpaceCache::default(),
            skel_cache: RefCell::default(),
            skel_warnings: RefCell::default(),
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
        let up = geom::stage_up_axis(&self.stage).map_err(|e| format!("upAxis: {e}"))?;
        let axis = match up {
            UpAxis::Y => DMat4::IDENTITY,
            // Z up: +Z becomes +Y, +Y becomes -Z.
            UpAxis::Z => DMat4::from_cols(DVec4::X, DVec4::NEG_Z, DVec4::Y, DVec4::W),
        };
        Ok(DMat4::from_scale(DVec3::splat(self.meters_per_unit()?)) * axis)
    }

    /// The stage's `metersPerUnit`.
    fn meters_per_unit(&self) -> Result<f64, String> {
        geom::stage_meters_per_unit(&self.stage).map_err(|e| format!("metersPerUnit: {e}"))
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
            let up = parent.map(|p| &self.nodes[p]);
            let instanced = !prim.is_defined().map_err(|e| e.to_string())?
                || prim.is_abstract().map_err(|e| e.to_string())?
                || up.is_some_and(|p| p.instanced || p.ty == "PointInstancer");
            let xform = read::Stack::read(&prim)?;
            let (local, reset) = (xform.local(self.time)?, xform.resets());
            let shown = self.shown(&prim, up.map_or_default(|p| p.shown))?;
            let world = match (parent, reset) {
                (Some(p), false) => self.nodes[p].world * local,
                _ => root * local,
            };
            let ty = prim
                .type_name()
                .map_err(|e| e.to_string())?
                .map_or_default(|t| t.as_str().to_string());
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
                stack: xform,
                shown,
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

    /// What `prim`, under a prim that says `parent`, says of whether it is drawn.
    fn shown(&self, prim: &usd::Prim, parent: Shown) -> Result<Shown, String> {
        let fail = |e: openusd::Error| format!("{}: {e}", prim.path().as_str());
        let purpose = geom::Imageable::from_prim_unchecked(prim.clone())
            .purpose_info_from(parent.purpose)
            .map_err(fail)?;
        let Some(ip) = geom::Imageable::from_prim(prim.clone()).map_err(fail)? else {
            return Ok(Shown { purpose, ..parent });
        };
        let visibility = ip.visibility_attr().get_at::<tf::Token>(self.time);
        let invisible =
            visibility.map_err(fail)?.as_deref() == Some(Visibility::Invisible.as_token());
        let render = match ip.purpose_visibility_attr(Purpose::Render).map_err(fail)? {
            Some(a) if a.has_authored_value().map_err(fail)? => a.get().map_err(fail)?,
            _ => parent.render,
        };
        Ok(Shown {
            invisible: parent.invisible || invisible,
            purpose,
            render,
        })
    }

    /// Per node of prototype `root`'s subtree, from `root` on, whether it is drawn by what that
    /// subtree alone says.
    fn drawn_in(&self, root: usize) -> Result<Vec<bool>, String> {
        let mut shown: Vec<Shown> = Vec::with_capacity(self.nodes[root].end - root);
        for k in self.subtree(root) {
            let node = &self.nodes[k];
            let parent = match node.parent.filter(|_| k != root) {
                Some(p) => shown[p - root],
                None => Shown::default(),
            };
            shown.push(self.shown(&node.prim, parent)?);
        }
        Ok(shown.iter().map(|s| !s.hidden()).collect())
    }

    fn run(&mut self) -> Result<Scene, String> {
        self.traverse()?;
        self.find_partitions()?;
        self.find_motions()?;
        self.find_timeline()?;
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
            let mut held = Vec::new();
            for k in self.subtree(i) {
                let node = &self.nodes[k];
                if owner[k].is_some() || node.ty != "Mesh" || node.instanced {
                    continue;
                }
                if let Some(b) = self.binding(k, i)? {
                    held.push((k, b));
                }
            }
            let mut sites: Vec<usize> = held.iter().map(|h| h.1.site).collect();
            sites.sort_unstable();
            sites.dedup();
            for &site in &sites {
                let name = ModelName::source(self.nodes[i].name.as_str());
                let name = match sites.len() {
                    1 => name,
                    _ => name.with(Derived::Binding(self.site_name(i, site).into())),
                };
                let p = self.partition_of[i];
                models.push(Building::new(name, None, Some(i), Some(p)));
                let m = models.len() - 1;
                held.iter()
                    .filter(|h| h.1.site == site)
                    .for_each(|h| owner[h.0] = Some(m));
            }
            bound.extend(held);
        }

        self.select_models(&mut models, &mut owner)?;

        // Every mesh of the stage drawn, with the model it goes to.
        let mut meshes: Vec<(usize, usize)> = (0..n)
            .filter(|&i| self.nodes[i].ty == "Mesh" && !self.nodes[i].shown.hidden())
            .filter_map(|i| Some((i, owner[i]?)))
            .collect();
        let order: Vec<usize> = (0..n).collect();
        let animated: Vec<bool> = self
            .nodes
            .iter()
            .map(|d| !d.stack.times.is_empty())
            .collect();
        let parents: Vec<Option<usize>> = self
            .nodes
            .iter()
            .map(|d| d.parent.filter(|_| !d.reset))
            .collect();
        let moved = |&i: &usize| skin::chain(&parents, i, None).any(|n| animated[n]);
        let instancers: Vec<usize> = (0..n)
            .filter(|&i| {
                let node = &self.nodes[i];
                node.ty == "PointInstancer" && !node.instanced && !node.shown.hidden()
            })
            .collect();

        // The rigs of the stage's models, before what they move places anything. Per model the
        // models its meshes go to, with what they apply: the model itself unless it places
        // nothing, and those that bake its other copies.
        let mut holders: Vec<Vec<(usize, DMat4)>> =
            Vec::from_fn(models.len(), |m| vec![(m, DMat4::IDENTITY)]);
        let mut rigs: HashMap<usize, skel::PrimRig> = HashMap::new();
        let mut rig_skeletons = Vec::new();
        let mut rig_of: HashMap<usize, usize> = HashMap::new();
        // Each selected model stands once where its prim does, a copy of the model.
        let selected: Vec<(usize, usize)> = (0..models.len())
            .filter_map(|m| Some((m, models[m].frame?)))
            .collect();
        let hung: Vec<usize> = instancers
            .iter()
            .copied()
            .chain(selected.iter().map(|s| s.1))
            .filter(|i| moved(i))
            .collect();
        let mut held: Vec<Vec<usize>> = vec![Vec::new(); models.len()];
        meshes.iter().for_each(|&(i, o)| held[o].push(i));
        // Copies stand in scene space, so what moves them joins the rest of the stage. A
        // selection has none of that: each instancer or selected prim takes a model of its own.
        let selecting = !self.settings.models.is_empty();
        let mut rest: HashMap<u32, usize> = (0..models.len())
            .filter_map(|m| Some((models[m].name.partition()?, m)))
            .collect();
        for &i in &hung {
            let p = self.partition_of[i];
            let m = match rest.get(&(p as u32)) {
                Some(&m) => m,
                None => {
                    let b = match selecting {
                        false => Building::rest(p),
                        true => {
                            let path = ModelName::source(self.nodes[i].prim.path().as_str());
                            Building::new(path, None, None, Some(p))
                        }
                    };
                    let m = models.len();
                    models.push(b);
                    holders.push(vec![(m, DMat4::IDENTITY)]);
                    held.push(Vec::new());
                    if !selecting {
                        rest.insert(p as u32, m);
                    }
                    m
                }
            };
            held[m].push(i);
        }
        // A selected model's own rig waits for its baked models, as a prototype's does.
        for (o, held) in held.iter().enumerate() {
            if models[o].skinned || models[o].frame.is_some() || !held.iter().any(moved) {
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
                (&mut rigs, &mut rig_skeletons, &mut rig_of),
            )?;
        }

        // Prototype models, which name themselves around the rest, and the placements of every
        // instancer.
        let mut instancing = Instancing::default();
        for &i in &instancers {
            let (to_scene, frame) = match rig_of.get(&i) {
                Some(&m) => {
                    let (joint, to_bind) = self.attach(&rigs[&m], &models[m], i)?;
                    (to_bind, Some((m, joint)))
                }
                None => (self.nodes[i].world, None),
            };
            let place = Placer {
                models: &mut models,
                instancing: &mut instancing,
                above: &mut Vec::new(),
                frame,
                partition: self.partition_of[i],
            };
            self.place(i, to_scene, place)?;
        }
        // A selected model holds its prim's space as `root` turns the stage's.
        for &(m, f) in &selected {
            let (to_scene, frame) = match rig_of.get(&f) {
                Some(&h) => {
                    let (joint, to_bind) = self.attach(&rigs[&h], &models[h], f)?;
                    (to_bind, Some((h, joint)))
                }
                None => (self.nodes[f].world, None),
            };
            let partition = self.partition_of[f];
            let key = (m, Dest { frame, partition });
            let copy = to_scene * self.root.inverse();
            instancing.copies.entry(key).or_default().push(copy);
        }
        // An instancer that holds still warned of its samples already, the placed one among them.
        let heard: std::collections::HashSet<String> = self.warnings.iter().cloned().collect();
        instancing.warnings.retain(|w| !heard.contains(w));
        self.warnings.append(&mut instancing.warnings);
        for m in holders.len()..models.len() {
            holders.push(vec![(m, DMat4::IDENTITY)]);
        }

        meshes.append(&mut instancing.meshes);
        let lights = self.lights(&instancing.lights)?;
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
            let settings = || self.mesh_settings(&models[o].name);
            if !as_cage && displaced && settings().displace_static.is_some() {
                baking.insert(o);
            }
            bindings.push((bound_here, as_cage));
        }

        // Per instanced model its placements by where they go.
        let mut placed: HashMap<usize, Vec<Copies>> = HashMap::new();
        let mut bakes = Bakes::default();
        for ((m, dest), transforms) in instancing.copies {
            let (direct, baked) = placements(&transforms, baking.contains(&m))
                .map_err(|e| format!("{}: {e}", models[m].name))?;
            placed.entry(m).or_default().push(Copies {
                dest,
                placed: direct,
                motion: None,
            });
            for (linear, copies) in baked {
                let b = bakes.model(&mut models, &mut holders, m, dest.frame, linear);
                placed.entry(b).or_default().push(Copies {
                    dest,
                    placed: copies,
                    motion: None,
                });
            }
        }
        for a in instancing.animated {
            let (m, dest) = (a.model, a.dest);
            if !baking.contains(&m) {
                placed.entry(m).or_default().push(Copies {
                    dest,
                    placed: a.placed,
                    motion: Some(a.motion),
                });
                continue;
            }
            let kept = kept_lengths(&a.placed, &a.motion);
            if kept.stretched {
                let r = self.reader(&self.nodes[a.instancer].prim);
                self.warnings.push(format!(
                    "{}: a copy's scale changes over time, so the height baked into {} \
                     stretches with it, not a world length",
                    r.path(),
                    models[m].name
                ));
            }
            let (direct, motion) = kept.direct.unzip();
            placed.entry(m).or_default().push(Copies {
                dest,
                placed: direct.unwrap_or_default(),
                motion,
            });
            for (linear, copies, motion) in kept.baked {
                let b = bakes.model(&mut models, &mut holders, m, dest.frame, linear);
                placed.entry(b).or_default().push(Copies {
                    dest,
                    placed: copies,
                    motion: Some(motion),
                });
            }
        }
        // Models that bake them hold all of a prototype's copies; a selected model stays.
        for (&m, lists) in &placed {
            let framed = models[m].frame.is_some();
            if !framed && bakes.count.contains_key(&m) && lists.iter().all(|l| l.placed.is_empty())
            {
                holders[m].retain(|h| h.0 != m);
            }
        }
        bound.extend(instancing.bound);
        // The rigs of prototypes and selected models, which their copies carry on their pages.
        let mut held: Vec<Vec<usize>> = vec![Vec::new(); models.len()];
        meshes.iter().for_each(|&(i, o)| held[o].push(i));
        for (o, held) in held.iter().enumerate() {
            let copied = models[o].prototype.is_some() || models[o].frame.is_some();
            if !copied || models[o].skinned || !held.iter().any(moved) {
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
                (&mut rigs, &mut rig_skeletons, &mut rig_of),
            )?;
        }
        // One skeleton per model that skins, from what its meshes bind: its prim path, the
        // skeleton and its space, the root's node, the prims that bind it and the model whose
        // skeleton it is.
        let mut skels: Vec<(String, skel::Skel, RigSpace, usize, Vec<usize>, ModelName)> =
            Vec::new();
        for m in 0..models.len() {
            if !models[m].skinned || models[m].rig_skeleton.is_some() {
                continue;
            }
            let mut skeleton: Option<&sdf::Path> = None;
            let mut sites = Vec::new();
            let mut still = false;
            for &(i, _) in meshes.iter().filter(|h| h.1 == m) {
                match bound.get(&i) {
                    Some(binding) => {
                        skeleton = Some(&binding.skeleton);
                        sites.push(binding.site);
                    }
                    None => still = true,
                }
            }
            let Some(path) = skeleton else { continue };
            let node = *self
                .by_path
                .get(path.as_str())
                .ok_or_else(|| format!("skeleton {path} is not cooked"))?;
            for (h, linear) in holders[m].clone() {
                let to_model = self.to_model(&models[h], node)?;
                let mut built =
                    skel::Skel::build(&self.reader(&self.nodes[node].prim), to_model, self.root)?;
                self.hang(&mut built, node, &models[h], &animated, &parents)?;
                if still {
                    built.add_still()?;
                }
                // A baked copy keeps its bind pose: the skeleton takes the transform.
                let space = built.settle(linear);
                let root = models[h].root.expect("skeletal roots have prims");
                let model = models[h].name.clone();
                let path = path.as_str().to_string();
                skels.push((path, built, space, root, sites.clone(), model));
                models[h].skinned = true;
                models[h].skeleton = Some(skels.len() - 1);
            }
            // Its meshes bind through it even where it bakes every copy and holds none.
            models[m].skeleton = models[holders[m][0].0].skeleton;
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
                        if let Some(p) = held.as_ref().filter(|p| p.width() > 4) {
                            self.warnings.push(format!(
                                "{path}: primvar {} is {} wide; a reader takes the first 4",
                                read.name,
                                p.width()
                            ));
                        }
                        let missing = held
                            .as_ref()
                            .map_or_default(|p| p.missing(mesh.domain(p.interp)));
                        if missing > 0 {
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
                if let Some(st) = &frame.st
                    && reads
                    && as_cage
                {
                    self.warnings.push(format!(
                        "{path}: a material's tangents follow {st}, which a subdivision cage does \
                         not carry; they follow uv set 0"
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
            if colored && !as_cage {
                for &(m, _) in &holders[o] {
                    models[m].welder.carries_colors = true;
                }
            }
            let skeleton = models[o].skeleton.and_then(|k| skels.get(k)).map(|s| &s.1);
            // Each holder's skeleton stands in spaces of its own, which its vertices take.
            let space = |m: usize| match models[m].skeleton {
                Some(k) if k < skels.len() => Some(skels[k].2),
                Some(k) => Some(rig_skeletons[k - skels.len()].space),
                None => None,
            };
            let spaces: Vec<Option<RigSpace>> = holders[o].iter().map(|&(m, _)| space(m)).collect();
            let rig = rigs.get(&o);
            let (skin, transform, part) = match (bound.get(&i), skeleton, rig) {
                (Some(b), Some(sk), _) => {
                    let (s, to_bind) = sk.binding(&path, b, mesh.points.len(), self.time)?;
                    (Some(s), to_bind, 0)
                }
                (None, Some(sk), _) => {
                    let still = sk
                        .still
                        .expect("`add_still` saw every mesh without a binding");
                    let one = ([still as u8; 4], [65535, 0, 0, 0]);
                    let to_model = self.to_model(&models[o], i)?;
                    (Some(vec![one; mesh.points.len()]), to_model, 0)
                }
                (_, _, Some(r)) => {
                    let (j, to_bind) = self.attach(r, &models[o], i)?;
                    let skin = (!r.rigid).then(|| {
                        let one = ([j as u8; 4], [65535, 0, 0, 0]);
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
            // A skeleton takes what a mesh in its bind pose cannot, but it poses no part 0, which
            // stands where its poses would take it.
            let posed = skin.is_some() || rig.is_some_and(|r| part != 0 && r.rigid);
            let applied = |linear: DMat4| if posed { transform } else { linear * transform };
            let from_model = |space: &Option<RigSpace>| match space {
                Some(s) if posed => s.to_bind_space(),
                Some(s) => DMat4::from_translation(-s.origin),
                None => DMat4::IDENTITY,
            };
            if skin.is_some()
                && (bound.contains_key(&i) || skeleton.is_none())
                && used
                    .iter()
                    .any(|&m| self.materials[m].material.object_normals != 0)
            {
                self.warnings.push(format!(
                    "{path}: its object space normal maps don't turn with its skeleton's pose"
                ));
            }
            // A map states directions in the prim's space, which a model's vertices left.
            let mut turned = Vec::with_capacity(holders[o].len());
            for &(_, linear) in &holders[o] {
                let baked = DMat3::from_mat4(applied(linear));
                let mut copies = HashMap::new();
                for &m in &used {
                    if self.materials[m].material.object_normals != 0 {
                        copies.insert(m, self.object_frame_material(m, baked));
                    }
                }
                turned.push(copies);
            }
            let color = |face: usize, corner: usize, point: usize| {
                let mut c = [1.0; 4];
                for read in &self.frames[faces[face]].vertex {
                    let held = read_primvars[&read.name].as_ref();
                    read.put(held.and_then(|p| p.at(point, face, corner)), &mut c);
                }
                c
            };
            let mut color_reads: Vec<mesh::ColorRead<'_>> = Vec::new();
            if as_cage {
                for (face, &m) in faces.iter().enumerate() {
                    for read in &self.frames[m].vertex {
                        match color_reads.iter_mut().find(|c| c.read == read) {
                            Some(c) => c.faces.push(face),
                            None => color_reads.push(mesh::ColorRead {
                                read,
                                held: read_primvars[&read.name].as_ref(),
                                faces: vec![face],
                            }),
                        }
                    }
                }
            }
            let placed = holders[o].iter().zip(&turned).zip(&spaces);
            for ((&(m, linear), turned), space) in placed {
                let turned_material = |f: usize| {
                    let m = face_material(f);
                    turned.get(&m).copied().unwrap_or(m)
                };
                let placing = Placing {
                    transform: from_model(space) * applied(linear),
                    face_material: &turned_material,
                    skin: skin.as_deref(),
                    part,
                    color: &color,
                    color_reads: &color_reads,
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
        let rig_skeleton: Vec<Option<usize>> = models
            .iter()
            .map(|b| b.rig_skeleton.map(|k| skels.len() + k))
            .collect();
        for (mi, b) in models.into_iter().enumerate() {
            let copies = placed.remove(&mi);
            // Models that bake them hold all of a prototype's copies.
            if b.frame.is_none()
                && copies
                    .as_ref()
                    .is_some_and(|c| c.iter().all(|l| l.placed.is_empty()))
            {
                continue;
            }
            // The rest of the stage, a skeletal root with no skinned mesh or a prototype may be
            // empty.
            let rig = b.rig_skeleton.is_some();
            if (b.name.partition().is_some() || b.skinned || b.prototype.is_some() || rig)
                && b.welder.groups.values().all(Vec::is_empty)
                && b.cage.is_empty()
            {
                continue;
            }
            if !names.insert(b.name.clone()) {
                return Err(format!("two models are named {:?}", b.name.to_string()));
            }
            for c in copies
                .into_iter()
                .flatten()
                .filter(|c| !c.placed.is_empty())
            {
                let frame = c.dest.frame;
                instance_out.push(PlacedModel {
                    model: b.name.clone(),
                    placed: c.placed,
                    alone: frame.is_some() || b.skeleton.is_some() || c.motion.is_some(),
                    motion: c.motion,
                    frame: frame
                        .map(|(m, joint)| (rig_skeleton[m].expect("a rig has a skeleton"), joint)),
                    partition: c.dest.partition,
                });
            }
            out.push(self.model(b)?);
        }
        if out.is_empty() {
            return Err("the stage has no triangles".into());
        }
        let mut skeletons = Vec::with_capacity(skels.len());
        for (path, sk, space, root, sites, model) in skels {
            let clips = self.clips(&path, (&sk, &space), root, &sites)?;
            skeletons.push(SceneSkeleton {
                model,
                info: sk.info,
                space,
                clips,
            });
        }
        skeletons.extend(rig_skeletons);
        if !self.unapplied.is_empty() {
            let first: Vec<&str> = self.unapplied.iter().take(5).map(String::as_str).collect();
            let more = if self.unapplied.len() > first.len() {
                ", ..."
            } else {
                ""
            };
            self.warnings.push(format!(
                "{} prims bind materials without MaterialBindingAPI applied, read as USD reads \
                 legacy bindings: {}{more}",
                self.unapplied.len(),
                first.join(", ")
            ));
        }
        let mut heard = std::collections::HashSet::new();
        let skel = std::mem::take(&mut *self.skel_warnings.borrow_mut());
        self.warnings
            .extend(skel.into_iter().filter(|w| heard.insert(w.clone())));
        Ok(Scene {
            models: out,
            materials: std::mem::take(&mut self.materials),
            images: std::mem::take(&mut self.images),
            skeletons,
            clips: self.settings.clips.clone(),
            instances: instance_out,
            groups: std::mem::take(&mut self.groups),
            partitions: std::mem::take(&mut self.partitions)
                .into_iter()
                .zip(lights)
                .map(|(groups, lights)| ScenePartition { groups, lights })
                .collect(),
            files: self.reads.borrow().inputs(),
            warnings: std::mem::take(&mut self.warnings),
        })
    }

    /// Mesh settings of model `name`.
    fn mesh_settings(&self, name: &ModelName) -> MeshSettings {
        let mut s = self.settings.mesh.clone();
        let overrides = &self.settings.mesh_overrides;
        if !overrides.is_empty() {
            patch::apply_matching(overrides, &name.to_string(), &mut s, MeshPatch::apply);
        }
        s
    }

    /// A built model's mesh, one section per scene material.
    fn model(&mut self, b: Building) -> Result<SceneModel, String> {
        let settings = self.mesh_settings(&b.name);
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
                (indices.len()..indices.len() + tris.len()).into(),
                &self.materials[m].material,
                None,
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
            return Err(match b.name.partition() {
                None => format!("model {} has no triangles", b.name),
                Some(_) => "the stage has no triangles".into(),
            });
        }
        Ok(SceneModel {
            name: b.name,
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
            partition: b.partition,
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
        (rigs, skeletons, rig_of): (
            &mut HashMap<usize, skel::PrimRig>,
            &mut Vec<SceneSkeleton>,
            &mut HashMap<usize, usize>,
        ),
    ) -> Result<(), String> {
        let in_scene = vec![true; parents.len()];
        // A prototype's model ends at its root.
        let mut cut = parents.to_vec();
        if let Some(root) = models[o].prototype {
            cut[root] = None;
        }
        let no_skin = |_: usize| None;
        let mut cages = std::collections::HashSet::new();
        // A selected prim it hangs may be a mesh, of the selected model.
        for &(i, _) in meshes.iter().filter(|h| h.1 == o) {
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
                // A prim heads a rig in each partition whose rest it moves: those rigs share a
                // name, told apart by partition, which no other model takes.
                let top = &self.nodes[rig.top];
                let free = |s: &ModelName| {
                    let other = |m: &Building| taken(m, s.base()) && m.rig_top != Some(rig.top);
                    !models.iter().any(other)
                };
                let name = [top.name.as_str(), top.prim.path().as_str()]
                    .map(ModelName::source)
                    .into_iter()
                    .find(free)
                    .unwrap_or_else(|| ModelName::new(ModelBase::Node(rig.top as u32)));
                let partition = models[o].partition;
                let name = match partition {
                    Some(p) if p != 0 => name.with(Derived::Partition(p as u32)),
                    _ => name,
                };
                let b = models.push_mut(Building::new(name, None, None, partition));
                b.rig_top = Some(rig.top);
                holders.push(vec![(models.len() - 1, DMat4::IDENTITY)]);
                let m = models.len() - 1;
                moved.extend(rig.meshes.iter().map(|&i| (i, m)));
                m
            };
            rig_of.extend(rig.meshes.iter().map(|&i| (i, m)));
            rig.rigid = !rig.meshes.iter().any(|i| cages.contains(i));
            let mut channels = Vec::new();
            for (k, &(h, linear)) in holders[m].clone().iter().enumerate() {
                let built = self.prim_rig(&rig, &models[h], linear)?;
                models[h].skinned = !rig.rigid;
                let joints = 0..built.info.joint_count();
                models[h].pivots = rig.rigid.then(|| {
                    let at = |j: usize| built.binds[j].w_axis.truncate() - built.space.bind_origin;
                    joints.map(|j| at(j).as_vec3().to_array()).collect()
                });
                models[h].rig_skeleton = Some(skeletons.len());
                if k == 0 {
                    channels = self.xform_channels(&built.prims)?;
                }
                let mut clip = self.xform_clip(&channels);
                if let Some(c) = &mut clip {
                    built.space.channels(&built.info, &mut c.channels);
                }
                skeletons.push(SceneSkeleton {
                    model: models[h].name.clone(),
                    info: built.info.clone(),
                    space: built.space,
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
        // `RigSpace::settle` sets the root, the inverse binds and the origin.
        let mut info = SkeletonInfo {
            root: Affine3::IDENTITY,
            origin: DVec3::ZERO,
            parents: Vec::with_capacity(count),
            rest: Vec::with_capacity(count),
            inverse_binds: Vec::new(),
            names: Vec::with_capacity(count),
        };
        let mut inverse_binds = Vec::with_capacity(count);
        let mut binds = Vec::with_capacity(count);
        if rig.still {
            let ib = inverse_bind(root);
            info.parents.push(NO_PARENT);
            info.rest.push(Trs::IDENTITY);
            inverse_binds.push(ib);
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
            inverse_binds.push(ib);
            info.names.push(node.name.clone());
            binds.push(inverse_bind(ib));
        }
        let still = rig.still.then_some(0);
        let after = DMat3::from_mat4(after);
        let space = RigSpace::settle(&mut info, root, after, &mut inverse_binds, still);
        Ok(skel::PrimRig {
            info,
            space,
            prims: nodes.iter().map(|&n| (n, joint_of[&n])).collect(),
            joint_of,
            binds,
            still,
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
    /// [`ClipSettings::max_rate`], in seconds of the stage's timeline.
    fn xform_channels(&self, prims: &[(usize, u16)]) -> Result<Vec<Channel>, String> {
        let per_second = self.stage.time_codes_per_second();
        let rate = f64::from(self.settings.clips.max_rate);
        let mut channels = Vec::new();
        for &(n, joint) in prims {
            let node = &self.nodes[n];
            let Some(&last) = node.stack.times.last() else {
                continue;
            };
            let mut times = Vec::new();
            for w in node.stack.times.windows(2) {
                let steps = ((w[1] - w[0]) / per_second * rate).ceil().max(1.0) as u32;
                let at = |k: u32| w[0] + (w[1] - w[0]) * f64::from(k) / f64::from(steps);
                times.extend((0..steps).map(at));
            }
            times.push(last);
            let path = node.prim.path().as_str();
            let mut keys = Vec::with_capacity(times.len());
            for &t in &times {
                let local = node.stack.local(usd::TimeCode::new(t))?;
                keys.push(skel::prim_trs(&format!("{path} at {t}"), local)?);
            }
            let timeline = &self.timeline;
            let seconds: Vec<f32> = times.iter().map(|&t| timeline.seconds(t) as f32).collect();
            let keyed = skel::trs_channels(usize::from(joint), &seconds, &keys);
            channels.extend(keyed.map_err(|e| format!("{path}: {e}"))?);
        }
        Ok(channels)
    }

    /// Settles the stage's [`Timeline`], over the samples of every moving xform, every instancer
    /// whose instances move and every `SkelAnimation`.
    fn find_timeline(&mut self) -> Result<(), String> {
        let held = self.nodes.iter().filter(|d| !d.stack.times.is_empty());
        let mut ends: Vec<(f64, f64)> = held
            .map(|d| (d.stack.times[0], d.stack.times[d.stack.times.len() - 1]))
            .collect();
        // An instancer's first interval runs back from its first sample.
        let moved = self.motions.values();
        ends.extend(moved.map(|m| (m.times[1], m.times[m.times.len() - 1])));
        for node in self.nodes.iter().filter(|d| d.ty == "SkelAnimation") {
            let times = skel::anim_query(&node.prim)?.joint_transform_time_samples();
            let times = times.map_err(|e| format!("{}: {e}", node.prim.path().as_str()))?;
            ends.extend(times.iter().map(|&t| (t, t)));
        }
        for (first, last) in ends {
            self.timeline.first = self.timeline.first.min(first);
            self.timeline.last = self.timeline.last.max(last);
        }
        Ok(())
    }

    /// The clip of `channels`, what xforms make up alone, or `None` without any. It runs over
    /// the stage's timeline, so the clips of separate skeletons keep time.
    fn xform_clip(&self, channels: &[Channel]) -> Option<ClipSource> {
        (!channels.is_empty()).then(|| ClipSource {
            channels: channels.to_vec(),
            span: self.timeline.span(),
        })
    }

    /// Hangs skeleton `sk` of prim `node` in model `b` from the prims that move it, if any do:
    /// every prim from the highest of them to its own.
    fn hang(
        &self,
        sk: &mut skel::Skel,
        node: usize,
        b: &Building,
        animated: &[bool],
        parents: &[Option<usize>],
    ) -> Result<(), String> {
        let mut chain: Vec<usize> = skin::chain(parents, node, None).collect();
        let mut moving = chain.len();
        if let Some(root) = b.prototype {
            match chain.iter().position(|&n| n == root) {
                Some(end) => {
                    chain.truncate(end + 1);
                    moving = chain.len();
                }
                // Past where it meets the instancer, a prim moves copies and skeleton alike.
                None => {
                    let holds = |&n: &usize| {
                        let ins = b.first_copy.map_or(root, |c| c.0);
                        self.subtree(n).contains(&ins)
                    };
                    moving = chain.iter().position(holds).unwrap_or(moving);
                }
            }
        }
        let Some(far) = chain[..moving].iter().rposition(|&n| animated[n]) else {
            return Ok(());
        };
        let root = match chain.get(far + 1) {
            Some(&above) => self.to_model(b, above)?,
            // The space above a prototype's highest prim is its model's, without what stands
            // above the prototype.
            None if b.prototype.is_some_and(|r| self.under(chain[far], r)) => self.root,
            None => self.stage_to_model(b),
        };
        let mut prims = Vec::with_capacity(far + 1);
        for &n in chain[..=far].iter().rev() {
            let d = &self.nodes[n];
            let local = skel::prim_trs(d.prim.path().as_str(), d.local)?;
            let ib = inverse_bind(self.to_model(b, n)?);
            prims.push((n, d.name.clone(), local, ib));
        }
        sk.hang(prims, root)
    }

    /// Model `b`'s space from stage space. For what stands outside a prototype that space is its
    /// first copy's, the common space of `UsdSkelImagingDataSourceXformResolver`.
    fn stage_to_model(&self, b: &Building) -> DMat4 {
        match b.first_copy {
            Some((_, copy)) => inverse_bind(copy) * self.root,
            None => self.root,
        }
    }

    /// Whether a prim at or above node `n` moves it over time, up to and with `top`, or the
    /// stage's root without it, or a prim that resets the transform stack.
    fn moves(&self, n: usize, top: Option<usize>) -> bool {
        let mut at = Some(n);
        while let Some(i) = at {
            let node = &self.nodes[i];
            if !node.stack.times.is_empty() {
                return true;
            }
            at = node.parent.filter(|_| Some(i) != top && !node.reset);
        }
        false
    }

    /// The lights of the stage, per partition: those it draws outside the instancers', and the
    /// copies of those its instancers' prototypes draw, `placed`.
    fn lights(&self, placed: &[(usize, DMat4, usize)]) -> Result<Vec<Vec<LightSource>>, String> {
        let mut out = vec![Vec::new(); self.partitions.len()];
        let mut found = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if !lux::is_light(&node.prim)? {
                continue;
            }
            if node.instanced || node.shown.hidden() {
                continue;
            }
            if self.moves(i, None) {
                return Err(format!(
                    "{}: a light that a transform sampled over time moves is not cooked",
                    node.prim.path().as_str()
                ));
            }
            found.push((i, node.world, self.partition_of[i]));
        }
        for &(i, to_scene, p) in found.iter().chain(placed) {
            let node = &self.nodes[i];
            out[p].push(lux::light(&self.reader(&node.prim), &node.ty, to_scene)?);
        }
        Ok(out)
    }

    /// Whether node `n` stands at or under node `top`.
    fn under(&self, n: usize, top: usize) -> bool {
        std::iter::successors(Some(n), |&k| self.nodes[k].parent).any(|k| k == top)
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
            if self.under(n, root) {
                return Ok(self.to_prototype(root, n));
            }
            return Ok(self.stage_to_model(b) * self.root.inverse() * self.nodes[n].world);
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

    /// The layout's groups from [`UsdSettings::groups`], and each node's partition: the set of
    /// groups whose collection includes its prim, as UsdCollectionAPI's membership rules say.
    fn find_partitions(&mut self) -> Result<(), String> {
        let mut queries = Vec::with_capacity(self.settings.groups.len());
        for g in &self.settings.groups {
            let fail =
                |e: openusd::Error| format!("group {}.collection:{}: {e}", g.prim, g.collection);
            let &node = self
                .by_path
                .get(g.prim.as_str())
                .ok_or_else(|| format!("group prim {} is not on the stage", g.prim))?;
            let prim = &self.nodes[node].prim;
            let collection = usd::CollectionAPI::from_prim(prim, g.collection.as_str())
                .map_err(fail)?
                .ok_or_else(|| format!("{} does not apply collection:{}", g.prim, g.collection))?;
            queries.push(collection.membership_query().map_err(fail)?);
            let mut attributes = Vec::new();
            let r = self.reader(prim);
            for ns in &g.attributes {
                for a in prim.authored_attributes_in_namespace(ns).map_err(fail)? {
                    let Some(value) = r.value(a.name())? else {
                        continue;
                    };
                    let value = attribute_value(value).ok_or_else(|| {
                        format!("{}.{}: no attribute value type holds it", g.prim, a.name())
                    })?;
                    attributes.push(Attribute {
                        name: a.name().to_owned(),
                        value,
                    });
                }
            }
            self.groups.push(Group {
                name: format!("{}.collection:{}", g.prim, g.collection),
                attributes,
            });
        }
        // Each node's membership follows from its own opinion and the rule its parent passes
        // down, which the pre-order of the nodes holds before it.
        let root = sdf::Path::abs_root();
        let seeds: Vec<PathRule> = queries
            .iter()
            .map(|q| q.is_included_below(&root, PathRule::Exclude).1)
            .collect();
        let mut rules: Vec<Vec<PathRule>> = Vec::with_capacity(self.nodes.len());
        let mut index: HashMap<Vec<u32>, usize> = HashMap::from([(Vec::new(), 0)]);
        self.partitions = vec![Vec::new()];
        self.partition_of = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let above = node.parent.map_or(&seeds, |p| &rules[p]);
            let mut held = Vec::new();
            let mut own = Vec::with_capacity(queries.len());
            for (g, q) in queries.iter().enumerate() {
                let (included, rule) = q.is_included_below(node.prim.path(), above[g]);
                if included {
                    held.push(g as u32);
                }
                own.push(rule);
            }
            rules.push(own);
            let p = match index.get(&held) {
                Some(&p) => p,
                None => {
                    index.insert(held.clone(), self.partitions.len());
                    self.partitions.push(held);
                    self.partitions.len() - 1
                }
            };
            self.partition_of.push(p);
        }
        Ok(())
    }

    /// Adds the selected models (or the whole-stage model) for prims no model holds yet.
    fn select_models(
        &self,
        models: &mut Vec<Building>,
        owner: &mut [Option<usize>],
    ) -> Result<(), String> {
        let n = self.nodes.len();
        if self.settings.models.is_empty() {
            let mut rest: HashMap<usize, usize> = HashMap::new();
            for (i, o) in owner.iter_mut().enumerate() {
                if o.is_none() && !self.nodes[i].instanced {
                    let p = self.partition_of[i];
                    *o = Some(*rest.entry(p).or_insert_with(|| {
                        models.push(Building::rest(p));
                        models.len() - 1
                    }));
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
            let name = ModelName::source(node.name.as_str());
            if models[first..].iter().any(|m| m.name == name) {
                return Err(format!("two selected prims are named {:?}", node.name));
            }
            owner[i] = Some(models.len());
            models.push(Building::new(name, Some(i), None, None));
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
        // Models derived from a name would meet those of the prototype's own binding prims.
        let named = ModelName::source(node.name.as_str());
        let path = ModelName::source(node.prim.path().as_str());
        let name = match [&named, &path].map(|n| models.iter().any(|m| taken(m, n.base()))) {
            [false, _] => named,
            [_, false] => path,
            _ => path.with(Derived::Prototype),
        };
        let model = models.len();
        models.push(Building::new(name.clone(), None, None, None));
        models[model].prototype = Some(root);
        let mut nested = Vec::new();
        let mut lights = Vec::new();
        let mut also = Vec::new();
        // Each skeleton binding met so far, with its model.
        let mut sites: Vec<(usize, usize)> = Vec::new();
        let drawn_in = self.drawn_in(root)?;
        let mut k = root;
        while k < node.end {
            let at = &self.nodes[k];
            let drawn = drawn_in[k - root];
            if at.ty == "PointInstancer" {
                if drawn {
                    nested.push(k);
                    let up = |&n: &usize| {
                        let d = &self.nodes[n];
                        d.parent.filter(|_| n != root && !d.reset)
                    };
                    let mut above = std::iter::successors(Some(k), up);
                    if above.any(|n| !self.nodes[n].stack.times.is_empty()) {
                        instancing.warnings.push(format!(
                            "{}: a prim of its prototype moves this instancer, whose copies stay \
                             where the time read puts them",
                            at.prim.path().as_str()
                        ));
                    }
                }
                k = at.end;
                continue;
            }
            if drawn && lux::is_light(&at.prim)? {
                lights.push(k);
            }
            if at.ty == "Mesh" && drawn {
                let skel_root = self.skel_root(k);
                let binding = match skel_root {
                    Some(r) => self.binding(k, r)?,
                    None => None,
                };
                let held = match binding {
                    None => model,
                    Some(b) => {
                        let m = match sites.iter().find(|s| s.0 == b.site) {
                            Some(s) => s.1,
                            None if sites.is_empty() => model,
                            None => {
                                let site = self.site_name(root, b.site).into();
                                let name = name.with(Derived::Binding(site));
                                let m = models.len();
                                let b = Building::new(name, None, None, None);
                                models.push_mut(b).prototype = Some(root);
                                also.push(m);
                                m
                            }
                        };
                        sites.push((b.site, m));
                        models[m].skinned = true;
                        models[m].root = skel_root;
                        instancing.bound.insert(k, b);
                        m
                    }
                };
                instancing.meshes.push((k, held));
            }
            k += 1;
        }
        // A skeleton's model takes no rig, so what moves beside its skins takes one of its own.
        let moves = |mesh: usize| {
            let (mut at, mut moved) = (Some(mesh), false);
            while let Some(n) = at {
                let d = &self.nodes[n];
                moved |= !d.stack.times.is_empty();
                at = d.parent.filter(|_| n != root && !d.reset);
            }
            moved
        };
        let bound = &instancing.bound;
        let mut rigged = None;
        for h in instancing.meshes.iter_mut().filter(|h| h.1 == model) {
            if !sites.is_empty() && !bound.contains_key(&h.0) && moves(h.0) {
                h.1 = *rigged.get_or_insert_with(|| {
                    let m = models.len();
                    let moved = Building::new(name.with(Derived::Moved), None, None, None);
                    models.push_mut(moved).prototype = Some(root);
                    also.push(m);
                    m
                });
            }
        }
        let p = Prototype {
            model,
            also,
            nested,
            lights,
        };
        instancing.prototypes.insert(root, p.clone());
        Ok(p)
    }

    /// Records the copies of instancer `ins`, whose space `to_scene` takes to scene space, and
    /// of the instancers under its prototypes. `above` lists the instancers being placed.
    fn place(&self, ins: usize, to_scene: DMat4, at: Placer<'_>) -> Result<(), String> {
        let Placer {
            models,
            instancing,
            above,
            frame,
            partition,
        } = at;
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
                let instancer = geom::PointInstancer::from_prim_unchecked(r.prim.clone());
                let targets = instancer.prototypes_rel().targets();
                for path in targets.map_err(|e| format!("{} prototypes: {e}", r.path()))? {
                    let root = self.by_path.get(path.as_str()).copied();
                    protos.push(match root {
                        Some(root) => Some((root, self.prototype(root, models, instancing)?)),
                        None => None,
                    });
                }
                // Those that move warned at every sample.
                let mut unheard = Vec::new();
                let warnings = match self.motions.contains_key(&ins) {
                    true => &mut unheard,
                    false => &mut instancing.warnings,
                };
                let placed = self.placements(ins, protos.len(), warnings)?;
                let i = Rc::new(Instancer { protos, placed });
                instancing.instancers.insert(ins, Rc::clone(&i));
                i
            }
        };
        let motion = self.motions.get(&ins);
        if let Some(motion) = motion {
            let at = Placer {
                models: &mut *models,
                instancing: &mut *instancing,
                above: &mut *above,
                frame,
                partition,
            };
            self.place_moving(ins, motion, &instancer, to_scene, at)?;
        }
        for (root, p) in instancer.protos.iter().flatten() {
            let carried = motion.is_some() || frame.is_some();
            let moved = p
                .lights
                .iter()
                .find(|&&l| carried || self.moves(l, Some(*root)));
            if let Some(&l) = moved {
                return Err(format!(
                    "{}: a light that moves with its prototype or instancer is not cooked",
                    self.nodes[l].prim.path().as_str()
                ));
            }
        }
        for &(proto, local) in &instancer.placed {
            let Some((root, p)) = &instancer.protos[proto] else {
                continue;
            };
            let transform = to_scene * local;
            for &l in &p.lights {
                let light = transform * self.to_prototype(*root, l);
                instancing.lights.push((l, light, partition));
            }
            let copied = std::iter::once(&p.model).chain(&p.also);
            for &m in copied.filter(|_| motion.is_none()) {
                let key = (m, Dest { frame, partition });
                instancing.copies.entry(key).or_default().push(transform);
                models[m].first_copy.get_or_insert((ins, transform));
            }
            for &nested in &p.nested {
                let to_scene = transform * self.to_prototype(*root, nested);
                let at = Placer {
                    models: &mut *models,
                    instancing: &mut *instancing,
                    above: &mut *above,
                    frame,
                    partition,
                };
                self.place(nested, to_scene, at)?;
            }
        }
        above.pop();
        Ok(())
    }

    /// Records the moving copies of instancer `ins`, a set per prototype. What an instancer in
    /// a prototype places stays out of them: a page poses a copy, not the pages it would carry.
    fn place_moving(
        &self,
        ins: usize,
        motion: &Motion,
        instancer: &Instancer,
        to_scene: DMat4,
        at: Placer<'_>,
    ) -> Result<(), String> {
        let Placer {
            models,
            instancing,
            frame,
            partition,
            ..
        } = at;
        let right = self.root.inverse();
        let seconds = |t: &f64| self.timeline.seconds(*t);
        let times: Vec<f64> = motion.times.iter().map(seconds).collect();
        for (proto, entry) in instancer.protos.iter().enumerate() {
            let Some((_, p)) = entry else { continue };
            let held = |&k: &usize| motion.slots[k].0 == proto;
            let slots: Vec<usize> = (0..motion.slots.len()).filter(held).collect();
            if slots.is_empty() {
                continue;
            }
            if !p.nested.is_empty() {
                let r = self.reader(&self.nodes[ins].prim);
                let path = r.path();
                instancing.warnings.push(format!(
                    "{path}: an instancer in a prototype of this one stays where the time read \
                     puts it, as its instances move"
                ));
            }
            let rest: Vec<DMat4> = slots
                .iter()
                .map(|&k| to_scene * motion.rest[k].transform() * right)
                .collect();
            let mut rows = Vec::with_capacity(times.len() * slots.len());
            for k in 0..times.len() {
                let at = k * motion.slots.len();
                rows.extend(slots.iter().map(|&s| motion.rows[at + s]));
            }
            let table = Arc::new(InstanceMotion {
                times: times.clone(),
                rows,
                anchors: slots
                    .iter()
                    .map(|&k| motion.rest[k].position.to_array())
                    .collect(),
                left: to_scene,
                right,
                start: seconds(&f64::from(self.time)),
            });
            // The kernel replaces a record, which need only stand somewhere it can invert.
            let record = |m: &DMat4| {
                let l = DMat3::from_mat4(*m);
                Placed {
                    position: m.w_axis.truncate(),
                    linear: if collapses(l) {
                        orr_math::Mat3::IDENTITY
                    } else {
                        l.as_mat3()
                    },
                }
            };
            let placed: Vec<Placed> = rest.iter().map(record).collect();
            for &m in std::iter::once(&p.model).chain(&p.also) {
                models[m].first_copy.get_or_insert((ins, rest[0]));
                instancing.animated.push(Animated {
                    model: m,
                    instancer: ins,
                    dest: Dest { frame, partition },
                    placed: placed.clone(),
                    motion: Arc::clone(&table),
                });
            }
        }
        Ok(())
    }

    /// Per instance of a `PointInstancer`: its prototype index and the instancer's space from
    /// the prototype's model.
    fn placements(
        &self,
        ins: usize,
        protos: usize,
        warnings: &mut Vec<String>,
    ) -> Result<Vec<(usize, DMat4)>, String> {
        let to_model = self.root.inverse();
        let states = self.instance_states(ins, protos, f64::from(self.time), warnings)?;
        let placed = states.iter().map(|s| (s.proto, s.transform() * to_model));
        Ok(placed.collect())
    }

    /// The instances of `ins` that `time` holds, from there until the next sample of what they
    /// read. `pointInstancer.h` and `UsdGeom_VelocityInterpolation` in `overview.dox` decide it:
    /// `placements` reads them as they stand, and an animated page keeps them per interval.
    fn instance_states(
        &self,
        ins: usize,
        protos: usize,
        time: f64,
        warnings: &mut Vec<String>,
    ) -> Result<Vec<InstanceState>, String> {
        let node = &self.nodes[ins];
        let r = self.reader(&node.prim);
        let r = r.at(time);
        let path = r.path();
        let instancer = geom::PointInstancer::from_prim_unchecked(node.prim.clone());
        let at = usd::TimeCode::new(time);
        let index = instancer.proto_indices_at(at);
        let index = index.map_err(|e| format!("{path} protoIndices: {e}"))?;
        let index = index.unwrap_or_default();
        let count = index.len();
        let around = |name: &str| self.samples_around(&node.prim, name, time);
        // `name` at the sample of `base`, if it stands around the time as `base` does and has one
        // value per instance.
        let mut dropped = Vec::new();
        let mut lined_up = |name: &str, base: Option<Around>, width: usize| match base {
            Some(b) if around(name)? == base => {
                let held = r.at(b.sample).numbers(name)?;
                let (fits, wrong): (Option<Vec<f64>>, _) = match held {
                    Some(v) if v.len() == width * count => (Some(v), None),
                    Some(v) => (None, Some(v.len() / width)),
                    None => (None, None),
                };
                dropped.extend(wrong.map(|n| {
                    format!("{path} {name} at {time}: {n} for {count} protoIndices; dropped")
                }));
                Ok::<_, String>(fits)
            }
            _ => Ok(None),
        };
        let orientation = self.orientation_attribute(&r)?;
        let moved = around("positions")?;
        let turned = around(orientation)?;
        let velocities = lined_up("velocities", moved, 3)?;
        let accelerations = match velocities {
            Some(_) => lined_up("accelerations", around("velocities")?, 3)?,
            None => None,
        };
        let spins = lined_up("angularVelocities", turned, 3)?;
        warnings.append(&mut dropped);
        // Without a rate to extrapolate by, values interpolate, but for samples of different
        // lengths, which hold the lower; with one, all hold.
        let rated = velocities.is_some() || spins.is_some();
        let per_second = self.stage.time_codes_per_second();
        let read = |name: &str, base: Option<Around>, width: usize| {
            let Some(b) = base else {
                let lower = r.numbers(name)?;
                return Ok::<_, String>(lower.map(|lower| Sampled { lower, upper: None }));
            };
            let Some(lower) = r.at(b.sample).numbers(name)? else {
                return Ok(None);
            };
            let between = !rated && b.lower <= time && time < b.upper;
            let upper = match between && lower.len() == width * count {
                true => r
                    .at(b.upper)
                    .numbers(name)?
                    .filter(|u| u.len() == lower.len()),
                false => None,
            };
            let span = b.upper - b.lower;
            let upper = upper.map(|u| (u, (time - b.lower) / span, span / per_second));
            Ok(Some(Sampled { lower, upper }))
        };
        let positions = read("positions", moved, 3)?;
        let orientations = read(orientation, turned, 4)?.filter(|v| !v.lower.is_empty());
        let scales = read("scales", around("scales")?, 3)?.filter(|v| !v.lower.is_empty());
        let since = |b: Option<Around>| b.map_or_default(|b| (time - b.sample) / per_second);
        let (moved_for, turned_for) = (since(moved), since(turned));
        let ids = instancer
            .ids_at(at)
            .map_err(|e| format!("{path} ids: {e}"))?;
        // An attribute of another length than `protoIndices` leaves nothing to place at this time.
        let lengths = [
            (
                "positions",
                Some(positions.as_ref().map_or_default(|v| v.lower.len() / 3)),
            ),
            (
                orientation,
                orientations.as_ref().map(|v| v.lower.len() / 4),
            ),
            ("scales", scales.as_ref().map(|v| v.lower.len() / 3)),
            // An empty `ids` names no instance, so it masks none (`ComputeMaskAtTime`).
            ("ids", ids.as_ref().filter(|v| !v.is_empty()).map(Vec::len)),
        ];
        if let Some((name, n)) = lengths
            .iter()
            .find_map(|&(name, n)| Some((name, n?)).filter(|l| l.1 != count))
        {
            warnings.push(format!(
                "{path} {name} at {time}: {n} for {count} protoIndices; no instances there"
            ));
            return Ok(Vec::new());
        }
        let known = |&p: &i32| usize::try_from(p).is_ok_and(|p| p < protos);
        if let Some(p) = index.iter().find(|p| !known(p)) {
            warnings.push(format!(
                "{path}: protoIndex {p} of {protos} prototypes; no instances at that sample"
            ));
            return Ok(Vec::new());
        }
        let inactive = node.prim.metadata::<vt::Value>("inactiveIds");
        let inactive = inactive.map_err(|e| format!("{path} inactiveIds: {e}"))?;
        if inactive.is_some_and(|v| !matches!(v, vt::Value::Int64ListOp(_))) {
            return Err(format!("{path} inactiveIds: not an int64 list op"));
        }
        // An instance's id is its index where `ids` holds none.
        let named = ids.unwrap_or_else(|| (0..count as i64).collect());
        let mask = instancer.mask_at(at, Some(&named));
        let mask = mask.map_err(|e| format!("{path} at {time}: {e}"))?;
        let mut out = Vec::with_capacity(count);
        for (i, &p) in index.iter().enumerate() {
            if mask.get(i) == Some(&false) {
                continue;
            }
            let proto = p as usize;
            let vector = |v: &Option<Vec<f64>>| {
                v.as_ref()
                    .map_or(DVec3::ZERO, |v| DVec3::from_slice(&v[3 * i..]))
            };
            // Between samples a line is a velocity, and a slerp a spin.
            let three = |v: &[f64]| DVec3::from_slice(&v[3 * i..]);
            let (mut position, mut velocity, mut acceleration) = match &positions {
                Some(Sampled {
                    lower,
                    upper: Some((upper, u, span)),
                }) => {
                    let (a, b) = (three(lower), three(upper));
                    (a + (b - a) * *u, (b - a) / *span, DVec3::ZERO)
                }
                Some(s) => (three(&s.lower), DVec3::ZERO, DVec3::ZERO),
                None => (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO),
            };
            if velocities.is_some() {
                let (v, a) = (vector(&velocities), vector(&accelerations));
                position += moved_for * (v + a * (moved_for * 0.5));
                velocity = v + a * moved_for;
                acceleration = a;
            }
            let (rotation, spin) = match &orientations {
                Some(Sampled {
                    lower,
                    upper: Some((upper, u, span)),
                }) => {
                    let a = read::quat(&lower[4 * i..]).normalize();
                    let mut b = read::quat(&upper[4 * i..]).normalize();
                    if a.dot(b) < 0.0 {
                        b = -b;
                    }
                    let (axis, angle) = (b * a.inverse()).to_axis_angle();
                    let at = DQuat::from_axis_angle(axis, angle * *u) * a;
                    (at, axis * angle.to_degrees() / *span)
                }
                Some(s) => {
                    let spin = vector(&spins);
                    let turn = (turned_for * spin.length()).to_radians();
                    let by = DQuat::from_axis_angle(spin.normalize_or(DVec3::X), turn);
                    (by * read::quat(&s.lower[4 * i..]).normalize(), spin)
                }
                None => (DQuat::IDENTITY, DVec3::ZERO),
            };
            let (scale, growth) = match &scales {
                Some(Sampled {
                    lower,
                    upper: Some((upper, u, span)),
                }) => {
                    let (a, b) = (three(lower), three(upper));
                    (a + (b - a) * *u, (b - a) / *span)
                }
                Some(s) => (three(&s.lower), DVec3::ZERO),
                None => (DVec3::ONE, DVec3::ZERO),
            };
            let state = InstanceState {
                index: i,
                proto,
                position,
                velocity,
                acceleration,
                rotation,
                spin,
                scale,
                growth,
            };
            if state.is_finite() {
                out.push(state);
            } else {
                warnings.push(format!(
                    "{path}: instance {i} is not finite; it is skipped at that sample"
                ));
            }
        }
        Ok(out)
    }

    /// Finds the instancers whose instances move by their own attributes.
    fn find_motions(&mut self) -> Result<(), String> {
        let mut motions = HashMap::new();
        for i in 0..self.nodes.len() {
            if self.nodes[i].ty != "PointInstancer" {
                continue;
            }
            let mut warnings = Vec::new();
            if let Some(m) = self.instance_motion(i, &mut warnings)? {
                motions.insert(i, m);
            }
            let mut seen = std::collections::HashSet::new();
            warnings.retain(|w| seen.insert(w.clone()));
            self.warnings.append(&mut warnings);
        }
        self.motions = motions;
        Ok(())
    }

    /// The motion of instancer `ins`, `None` where nothing moves. Every interval runs from a
    /// sample of what steps or interpolates to the next, all else is constant in it: rates apply
    /// only between samples of their own at both its ends, or none at all.
    fn instance_motion(
        &self,
        ins: usize,
        warnings: &mut Vec<String>,
    ) -> Result<Option<Motion>, String> {
        let node = &self.nodes[ins];
        let r = self.reader(&node.prim);
        let path = r.path();
        let mut samples = Vec::new();
        for name in [
            "positions",
            self.orientation_attribute(&r)?,
            "scales",
            "protoIndices",
            "ids",
            "invisibleIds",
        ] {
            let held = node.prim.attribute(name).time_sample_times();
            samples.extend(held.map_err(|e| format!("{path} {name}: {e}"))?);
        }
        samples.sort_by(f64::total_cmp);
        samples.dedup();
        let Some(&first) = samples.first() else {
            return Ok(None);
        };
        let instancer = geom::PointInstancer::from_prim_unchecked(node.prim.clone());
        let protos = instancer.prototypes_rel().targets();
        let protos = protos.map_err(|e| format!("{path} prototypes: {e}"))?.len();
        // The first interval runs back from the first sample, which only a rate can move.
        let times: Vec<f64> = std::iter::once(first - 1.0).chain(samples).collect();
        let mut keys = Vec::with_capacity(times.len());
        for &t in &times {
            keys.push(self.instance_states(ins, protos, t, warnings)?);
        }
        let same = |a: &InstanceState, b: &InstanceState| {
            a.index == b.index
                && a.proto == b.proto
                && a.transform() == b.transform()
                && a.holds_still()
                && b.holds_still()
        };
        let still = keys
            .iter()
            .all(|k| k.len() == keys[0].len() && k.iter().zip(&keys[0]).all(|(a, b)| same(a, b)));
        if still {
            return Ok(None);
        }
        let mut slots: Vec<(usize, usize)> =
            keys.iter().flatten().map(|s| (s.proto, s.index)).collect();
        slots.sort_unstable();
        slots.dedup();
        let slot_of = |s: &InstanceState| {
            slots
                .binary_search(&(s.proto, s.index))
                .expect("every state has a slot")
        };
        let mut rest: Vec<Option<InstanceState>> = vec![None; slots.len()];
        for s in keys.iter().flatten() {
            rest[slot_of(s)].get_or_insert(*s);
        }
        let now = self.instance_states(ins, protos, f64::from(self.time), &mut Vec::new())?;
        for s in &now {
            rest[slot_of(s)] = Some(*s);
        }
        let rest: Vec<InstanceState> = rest
            .into_iter()
            .map(|s| s.expect("every slot has a key"))
            .collect();
        let mut rows = vec![MotionRow::default(); times.len() * slots.len()];
        for (k, states) in keys.iter().enumerate() {
            for s in states {
                let slot = slot_of(s);
                rows[k * slots.len() + slot] = s.row(rest[slot].position);
            }
        }
        Ok(Some(Motion {
            times,
            slots,
            rows,
            rest,
        }))
    }

    /// Where attribute `name` of `prim` stands around `time`, `None` without samples: as
    /// `_GetAttrForTransforms` in `samplingUtils.cpp` finds it, the samples around the time just
    /// past it where the time is on a sample, so that alignment is judged on the interval ahead.
    fn samples_around(
        &self,
        prim: &usd::Prim,
        name: &str,
        time: f64,
    ) -> Result<Option<Around>, String> {
        let bracket = |t: f64| {
            prim.attribute(name)
                .bracketing_time_samples(usd::TimeCode::new(t))
                .map_err(|e| format!("{} {name}: {e}", prim.path().as_str()))
        };
        let Some((sample, upper)) = bracket(time)? else {
            return Ok(None);
        };
        let (lower, upper) = if sample == upper {
            bracket(time + SAFE_STEP)?.unwrap_or((sample, upper))
        } else {
            (sample, upper)
        };
        Ok(Some(Around {
            sample,
            lower,
            upper,
        }))
    }

    /// The attribute orientations come from (`UsesOrientationsf`): `orientationsf`, where its
    /// earliest sample (or default without samples) is not empty.
    fn orientation_attribute(&self, r: &PrimReader<'_>) -> Result<&'static str, String> {
        let instancer = geom::PointInstancer::from_prim_unchecked(r.prim.clone());
        let (f, _) = instancer
            .uses_orientationsf()
            .map_err(|e| format!("{}: {e}", r.path()))?;
        Ok(if f { "orientationsf" } else { "orientations" })
    }

    /// What names skeleton binding `site` beside others under node `root`: its path under
    /// `root`, or its absolute path, which starts with `/`, where it stands elsewhere.
    fn site_name(&self, root: usize, site: usize) -> &str {
        let path = self.nodes[site].prim.path().as_str();
        let under = path.strip_prefix(self.nodes[root].prim.path().as_str());
        under.and_then(|p| p.strip_prefix('/')).unwrap_or(path)
    }

    /// Material `m`, or its copy whose object space maps read through `baked`.
    fn object_frame_material(&mut self, m: usize, baked: DMat3) -> usize {
        let rows = crate::import::normal_matrix(baked)
            .transpose()
            .as_mat3()
            .to_cols_array_2d();
        // On a grid far finer than a map's texels, turns that only rounding parts share a copy: a
        // cosine of 90 degrees reads 6e-17 where another prim states 0. Adding zero turns -0.0
        // into 0.0.
        let step = 2f32.powi(20);
        let frame: [[f32; 3]; 3] = std::array::from_fn(|r| {
            std::array::from_fn(|c| ((rows[r][c] - f32::from(r == c)) * step).round() / step + 0.0)
        });
        if frame == [[0.0; 3]; 3] {
            return m;
        }
        let key = (m, frame.map(|row| row.map(f32::to_bits)));
        if let Some(&turned) = self.object_frames.get(&key) {
            return turned;
        }
        let mut copy = self.materials[m].clone();
        copy.name = copy
            .name
            .with(Variant::Object(self.object_frames.len() as u32));
        copy.material.object_frame = frame;
        self.materials.push(copy);
        self.frames.push(self.frames[m].clone());
        let turned = self.materials.len() - 1;
        self.object_frames.insert(key, turned);
        turned
    }

    /// The nearest `SkelRoot` at or above node `n`.
    fn skel_root(&self, n: usize) -> Option<usize> {
        std::iter::successors(Some(n), |&i| self.nodes[i].parent)
            .find(|&i| self.nodes[i].ty == "SkelRoot")
    }

    /// How mesh `m` under `SkelRoot` `root` binds a skeleton, as `UsdSkelCache` resolves it: by
    /// the joint influences and `skel:skeleton` it or the nearest prim above it up to the root
    /// states through `SkelBindingAPI`, where that names a `Skeleton`.
    fn binding(&self, m: usize, root: usize) -> Result<Option<Bound>, String> {
        let prim = &self.nodes[m].prim;
        let path = prim.path().as_str();
        let query = {
            let (cache, populated) = &mut *self.skel_cache.borrow_mut();
            if populated.insert(root) {
                let r = usd_skel::Root::from_prim_unchecked(self.nodes[root].prim.clone());
                let all = usd::PrimPredicate::ALL;
                let filled = cache.populate(&r, all);
                filled.map_err(|e| format!("{}: {e}", self.nodes[root].prim.path().as_str()))?;
            }
            cache.skinning_query(prim).cloned()
        };
        let Some(query) = query else {
            return Ok(None);
        };
        if !query.has_joint_influences() {
            if query.joint_indices_primvar().is_some() || query.joint_weights_primvar().is_some() {
                self.skel_warnings.borrow_mut().push(format!(
                    "{path}: joint influences that are not both constant or both vertex with one \
                     elementSize bind nothing"
                ));
            }
            return Ok(None);
        }
        let held = query.joint_influences(self.time);
        let held = held.map_err(|e| format!("{path} joint influences: {e}"))?;
        let points = || {
            let mesh = geom::Mesh::from_prim_unchecked(prim.clone());
            let points = mesh.points_at(self.time);
            let points = points.map_err(|e| format!("{path} points: {e}"))?;
            Ok::<_, String>(points.map_or(0, |p| p.len()))
        };
        let size = query.influences_per_component();
        let influences = match held {
            Some(h) if query.is_rigidly_deformed() || h.0.len() == points()? * size => h,
            // `ComputeJointInfluences` and `ComputeVaryingJointInfluences` give none.
            _ => {
                self.skel_warnings.borrow_mut().push(format!(
                    "{path}: joint influences that hold no value at the time read, or not one run \
                     of {size} for each point, bind nothing"
                ));
                return Ok(None);
            }
        };
        let fail = |e: openusd::Error| format!("{path} skel:skeleton: {e}");
        let mut at = Some(m);
        while let Some(n) = at {
            let api = usd_skel::BindingAPI::from_prim(self.nodes[n].prim.clone()).map_err(fail)?;
            if let Some(api) = api
                && api.skeleton_rel().has_authored_targets().map_err(fail)?
            {
                let skeleton = api.skeleton().map_err(fail)?;
                let Some(skeleton) =
                    skeleton.filter(|s| self.by_path.contains_key(s.path().as_str()))
                else {
                    return Ok(None);
                };
                return Ok(Some(Bound {
                    skeleton: skeleton.path().clone(),
                    site: n,
                    query,
                    influences,
                }));
            }
            at = self.nodes[n].parent.filter(|_| n != root);
        }
        Ok(None)
    }

    /// The animation bound at node `n`: the `skel:animationSource` there or on its nearest
    /// ancestor that states one through `SkelBindingAPI`. A target that is deactivated binds
    /// nothing.
    fn animation_source(&self, n: usize) -> Result<Option<usize>, String> {
        let prim = &self.nodes[n].prim;
        let api = usd_skel::BindingAPI::from_prim_unchecked(prim.clone());
        let anim = api.inherited_animation_source();
        let anim =
            anim.map_err(|e| format!("{} skel:animationSource: {e}", prim.path().as_str()))?;
        Ok(anim.and_then(|a| self.by_path.get(a.path().as_str()).copied()))
    }

    /// Clips of skeleton `path`: the `skel:animationSource` in force where it is bound (nodes
    /// `sites`) and on the skeleton, and every animation under its skeletal root, node `root`.
    fn clips(
        &self,
        path: &str,
        (sk, space): (&skel::Skel, &RigSpace),
        root: usize,
        sites: &[usize],
    ) -> Result<Vec<(String, ClipSource)>, String> {
        let mut sources: Vec<usize> = Vec::new();
        for &n in sites.iter().chain(self.by_path.get(path)) {
            sources.extend(self.animation_source(n)?);
        }
        sources.extend(self.subtree(root).filter(|&i| {
            let node = &self.nodes[i];
            node.ty == "SkelAnimation" && node.instanced == self.nodes[root].instanced
        }));
        sources.sort_unstable();
        sources.dedup();
        let mut xforms = self.xform_channels(&sk.prims)?;
        space.channels(&sk.info, &mut xforms);
        let mut clips = Vec::new();
        for i in sources {
            let node = &self.nodes[i];
            if let Some(c) = sk.clip(&node.prim, &self.timeline, &xforms, space)? {
                clips.push((node.name.clone(), c));
            }
        }
        if clips.is_empty()
            && let Some(c) = self.xform_clip(&xforms)
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

    /// The `Material` bound to `prim` for the `full` purpose, as `ComputeBoundMaterial` resolves
    /// it with `supportLegacyBindings`, its default: a binding counts on a prim without
    /// `MaterialBindingAPI`, which the stage's warnings list as USD warns of them. A binding to a
    /// prim that is no `Material` binds nothing.
    fn bound_material(&mut self, prim: &usd::Prim) -> Result<Option<sdf::Path>, String> {
        let path = prim.path().as_str();
        let fail = |e: openusd::Error| format!("{path} material binding: {e}");
        let api = usd_shade::MaterialBindingAPI::from_prim_unchecked(prim.clone());
        let resolved = api.resolve_binding(usd_shade::tokens::FULL, true, &mut self.binding_cache);
        let Some(resolved) = resolved.map_err(fail)? else {
            return Ok(None);
        };
        let site = resolved.rel().prim();
        let applied = site.has_api_schema(usd_shade::tokens::MATERIAL_BINDING_API);
        if !applied.map_err(fail)? {
            self.unapplied.insert(site.path().as_str().to_string());
        }
        let target = resolved.material_path();
        match resolved.material().map_err(fail)? {
            Some(_) => Ok(Some(target.clone())),
            None => {
                self.warnings.push(format!(
                    "{path}: binds {target}, which is not a Material; it takes the default material"
                ));
                Ok(None)
            }
        }
    }

    /// The scene materials mesh prim `i` binds.
    fn bindings(&mut self, i: usize, double_sided: bool) -> Result<Bindings, String> {
        let prim = self.nodes[i].prim.clone();
        let base = self.bound_material(&prim)?;
        let base = self.material(base, double_sided, i)?;
        // The mesh's own children, which an instance proxy's are too.
        let subsets: Vec<usize> = self
            .subtree(i)
            .filter(|&k| self.nodes[k].parent == Some(i) && self.nodes[k].ty == "GeomSubset")
            .collect();
        let mut bound = Vec::new();
        for k in subsets {
            let subset = geom::Subset::from_prim_unchecked(self.nodes[k].prim.clone());
            let path = subset.path().as_str().to_string();
            let fail = |e: openusd::Error| format!("{path}: {e}");
            let family = subset.family_name().map_err(fail)?;
            let element = subset
                .element_type_attr()
                .get::<tf::Token>()
                .map_err(fail)?;
            if family.as_deref() != Some(usd_shade::tokens::MATERIAL_BIND)
                || element.as_deref() != Some(geom::ElementType::Face.as_token())
            {
                continue;
            }
            let indices = subset.indices_at(self.time).map_err(fail)?;
            let m = self.bound_material(&subset)?;
            let m = self.material(m, double_sided, i)?;
            bound.push((path, indices.unwrap_or_default(), m));
        }
        Ok(Bindings {
            base,
            subsets: bound,
        })
    }

    /// Scene material per face of `mesh`, which binds `bindings`.
    fn face_materials(&self, bindings: &Bindings, mesh: &MeshPrim) -> Result<Vec<usize>, String> {
        let mut faces = vec![bindings.base; mesh.counts.len()];
        for (path, indices, m) in &bindings.subsets {
            for &f in indices {
                let slot = usize::try_from(f)
                    .ok()
                    .and_then(|f| faces.get_mut(f))
                    .ok_or_else(|| format!("{path}: face {f} past the mesh"))?;
                *slot = *m;
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
                    let udim = ar::UdimPath::parse(&name).expect("a path with tiles names them");
                    let mut tiles = Vec::with_capacity(found.len());
                    for (n, path) in found {
                        images.push(read(udim.tile(n), &path));
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
        let material =
            usd_shade::Material::get(&self.stage, p.clone()).map_err(|e| e.to_string())?;
        let material = material.ok_or_else(|| format!("{p} is not a Material"))?;
        let shaded = shade::network(&material, self.time, &mut load, &mut self.spaces)?;
        let shade::Shaded {
            net,
            surface,
            displacement,
            warnings,
        } = shaded;
        self.warnings.extend(warnings);
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

    /// What mesh prim `mesh` holds for primvar `name`, as `FindPrimvarWithInheritance` finds it:
    /// its own, else the nearest ancestor's that is constant.
    fn primvar(&self, mesh: usize, name: &str) -> Result<shading::Primvar, String> {
        let set = self.settings.uv_sets().position(|s| s == name);
        let prim = &self.nodes[mesh].prim;
        let facts = MeshFacts::read(&self.reader(prim))?;
        let cage = !facts.polygonal && self.settings.subdivision == Subdivision::Cage;
        let reach = if cage { CAGE_COLOR_REACH } else { 1.0 };
        let api = geom::PrimvarsAPI::from_prim_unchecked(prim.clone());
        let found = match api.primvar_with_inheritance(name) {
            Ok(Some(found)) => found,
            // `GetPrimvar` gives an invalid primvar for a name no primvar has.
            Ok(None) | Err(openusd_schemas::SchemaError::InvalidPrimvarName { .. }) => {
                return Ok(shading::Primvar::Missing);
            }
            Err(e) => return Err(format!("{} primvar {name}: {e}", prim.path().as_str())),
        };
        let owner = found.attribute().prim();
        Ok(
            match (self.reader(&owner).whole_primvar(name, reach)?, set) {
                (shading::Primvar::Missing, _) => shading::Primvar::Missing,
                (_, Some(set)) if owner.path() == prim.path() => shading::Primvar::Uv(set as u8),
                (held, _) => held,
            },
        )
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
                let path = MaterialName::new(MaterialBase::Path(p.as_str().to_string()));
                let copy = self
                    .copies_of
                    .entry((p.as_str().to_string(), double_sided))
                    .or_default();
                *copy += 1;
                let name = match *copy {
                    1 => path,
                    k => path.with(Variant::Primvars((k - 1) as u32)),
                };
                (mapped, name)
            }
            _ => (
                shading::Mapped {
                    material: Default::default(),
                    textures: Default::default(),
                    model: ShadingModel::OpenPbr,
                    vertex: Vec::new(),
                    warnings: Vec::new(),
                },
                MaterialName::new(MaterialBase::Default),
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
        let name = match double_sided {
            true => name.with(Variant::DoubleSided),
            false => name,
        };
        self.warnings.extend(mapped.warnings);
        self.materials.push(SceneMaterial {
            name,
            material: mapped.material,
            textures: mapped.textures,
            model: mapped.model,
            back: None,
        });
        self.frames.push(frame);
        let m = self.materials.len() - 1;
        self.material_of.insert(key, m);
        Ok(m)
    }
}
