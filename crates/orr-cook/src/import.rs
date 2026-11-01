//! What a scene importer (glTF, USD) hands the planner: models with their meshes, the
//! materials and images they use, skeletons with their clips, instanced models and their
//! placements, and the files it read.

use orr_math::{DMat3, DQuat, DVec3, Mat3, Mat4, Quat, Vec3};
use orr_pack::clip::Clip;
use orr_pack::material::Material;
use orr_pack::skeleton::SkeletonInfo;
use serde::Deserialize;

use crate::mesh::{MeshSettings, MeshSource};
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
    pub models: Vec<SceneModel>,
    pub materials: Vec<SceneMaterial>,
    pub images: Vec<ImageFile>,
    pub skeletons: Vec<SceneSkeleton>,
    /// Per instanced model: its name and its instances' scene-space placements.
    pub instances: Vec<(String, Vec<Placed>)>,
    /// Every file read.
    pub files: Vec<Input>,
}

/// One placement: position, rotation and per-axis scale, applied before the rotation; a
/// negative scale component mirrors.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Placed {
    pub position: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

pub(crate) struct SceneSkeleton {
    /// Entry names `<asset>/<prefix>/skeleton` and `<asset>/<prefix>/clip/<clip>`; an empty
    /// prefix leaves its part out.
    pub prefix: String,
    pub info: SkeletonInfo,
    /// Clips by name; their skeleton id is left for the planner to set.
    pub clips: Vec<(String, Clip)>,
}

pub(crate) struct SceneModel {
    /// The selected node's name; `None` for the whole scene.
    pub node: Option<String>,
    /// A mesh without sections cooks no mesh entry.
    pub mesh: MeshSource,
    pub subd: Option<SceneSubd>,
    /// Index into [`Scene::skeletons`] of a skinned mesh.
    pub skeleton: Option<usize>,
    pub settings: MeshSettings,
    /// Index into [`Scene::materials`] per mesh section.
    pub materials: Vec<usize>,
}

/// A model's subdivision cage.
pub(crate) struct SceneSubd {
    pub cage: CageData,
    /// Index into [`Scene::materials`] per cage section.
    pub materials: Vec<usize>,
    pub settings: SubdSettings,
}

pub(crate) struct SceneMaterial {
    /// Entry name suffix: `<asset>/material/<suffix>`, textures under
    /// `<asset>/texture/<suffix>/`; empty names the material `<asset>` itself.
    pub suffix: String,
    pub material: Material,
    pub textures: MaterialTextures,
}

/// Largest shear a placement drops: the largest entry of `linear - R diag(s)` over the
/// largest entry of `linear`. It is below the instance record's own rotation error.
pub const SHEAR_MAX: f64 = 1e-3;

/// What a scene-space transform places.
pub(crate) enum Decomposed {
    Placed(Placed),
    /// The transform collapses an axis: nothing shows.
    Nothing,
    /// The transform shears further than [`SHEAR_MAX`].
    Sheared,
}

fn largest(m: DMat3) -> f64 {
    m.to_cols_array()
        .into_iter()
        .fold(0.0, |a, x| a.max(x.abs()))
}

/// The placement of a scene-space transform.
pub(crate) fn placement(m: Mat4) -> Decomposed {
    let l = Mat3::from_mat4(m).as_dmat3();
    let mut s = DVec3::new(l.x_axis.length(), l.y_axis.length(), l.z_axis.length());
    let largest_entry = largest(l);
    if s.min_element() <= largest_entry * 1e-9 {
        return Decomposed::Nothing;
    }
    if l.determinant() < 0.0 {
        s.x = -s.x;
    }
    // The rotation nearest `l diag(1 / s)`, by polar iteration.
    let mut r = l * DMat3::from_diagonal(s.recip());
    for _ in 0..16 {
        r = (r + r.inverse().transpose()) * 0.5;
    }
    if largest(l - r * DMat3::from_diagonal(s)) > SHEAR_MAX * largest_entry {
        return Decomposed::Sheared;
    }
    Decomposed::Placed(Placed {
        position: m.w_axis.truncate().as_dvec3(),
        rotation: DQuat::from_mat3(&r).normalize().as_quat(),
        scale: s.as_vec3(),
    })
}
