//! What a scene importer (glTF, USD) hands the planner: models with their meshes, the
//! materials and images they use, skeletons with their clips, instanced models and their
//! placements, and the files it read.

use orr_math::Mat4;
use orr_pack::clip::Clip;
use orr_pack::material::Material;
use orr_pack::skeleton::SkeletonInfo;
use serde::Deserialize;

use crate::mesh::{MeshData, MeshSettings};
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
    /// Per instanced model: its name and its instances' scene-space transforms.
    pub instances: Vec<(String, Vec<Mat4>)>,
    /// Every file read.
    pub files: Vec<Input>,
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
    pub mesh: MeshData,
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
