//! What a scene importer (glTF, USD) hands the planner: models with their meshes, the
//! materials and images they use, skeletons with their clips, instanced models and their
//! placements, and the files it read.

use std::collections::HashMap;

use orr_math::{DMat3, DVec3, Mat3, Mat4};
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
    pub(crate) models: Vec<SceneModel>,
    pub(crate) materials: Vec<SceneMaterial>,
    pub(crate) images: Vec<ImageFile>,
    pub(crate) skeletons: Vec<SceneSkeleton>,
    /// Per instanced model: its name and its instances' scene-space placements.
    pub(crate) instances: Vec<(String, Vec<Placed>)>,
    /// Every file read.
    pub(crate) files: Vec<Input>,
    pub(crate) warnings: Vec<String>,
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
    /// Clips by name; their skeleton id is left for the planner to set.
    pub(crate) clips: Vec<(String, Clip)>,
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

pub(crate) struct SceneMaterial {
    /// Entry name suffix: `<asset>/material/<suffix>`, textures under
    /// `<asset>/texture/<suffix>/`; empty names the material `<asset>` itself.
    pub(crate) suffix: String,
    pub(crate) material: Material,
    pub(crate) textures: MaterialTextures,
}

/// Copies whose linear transform collapses: the transform, for the importer to bake into a
/// model of its own, and the copies of that model, placed by translation alone.
pub(crate) type Baked = (Mat4, Vec<Placed>);

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

/// The inverse transpose of `l` up to a positive scale, which exists where `l` collapses. Its
/// largest entry is 1, so it fits an f32 whatever the size of `l`.
pub(crate) fn normal_matrix(l: DMat3) -> DMat3 {
    let [x, y, z] = [l.x_axis, l.y_axis, l.z_axis];
    let m = DMat3::from_cols(y.cross(z), z.cross(x), x.cross(y));
    let sign = if l.determinant() < 0.0 { -1.0 } else { 1.0 };
    m * (sign / largest(m).max(f64::MIN_POSITIVE))
}

/// The copies `transforms` place: those an instance can hold, then the rest by linear
/// transform.
pub(crate) fn placements(transforms: &[Mat4]) -> Result<(Vec<Placed>, Vec<Baked>), String> {
    let mut placed = Vec::new();
    let mut baked: Vec<Baked> = Vec::new();
    let mut baked_of: HashMap<[u32; 16], usize> = HashMap::new();
    for (i, m) in transforms.iter().enumerate() {
        if !m.is_finite() {
            return Err(format!("the transform of copy {i} is not finite"));
        }
        let position = m.w_axis.truncate().as_dvec3();
        let l = Mat3::from_mat4(*m);
        if !collapses(l.as_dmat3()) {
            placed.push(Placed {
                position,
                linear: l,
            });
            continue;
        }
        let linear = Mat4::from_mat3(l);
        // Adding zero turns -0.0 into 0.0: copies scaled to zero differ only in such signs.
        let k = *baked_of
            .entry(linear.to_cols_array().map(|x| (x + 0.0).to_bits()))
            .or_insert_with(|| {
                baked.push((linear, Vec::new()));
                baked.len() - 1
            });
        baked[k].1.push(Placed {
            position,
            linear: Mat3::IDENTITY,
        });
    }
    Ok((placed, baked))
}
