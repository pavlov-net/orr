//! Subdivision cages: Catmull-Clark topology refined by OpenSubdiv Far into feature-adaptive
//! patches, cooked as a [`orr_pack::subd`] entry.
//!
//! Every ptex face's patches come from one patch table refined to
//! [`SubdSettings::isolation`] around creases, corners and boundaries and to
//! [`SubdSettings::smooth_isolation`] around smooth extraordinary vertices, with Gregory
//! basis patches where isolation stops. Faces are ordered by section, then by cage face, and
//! packed whole into pages; a page holds the stencils of the points its patches read. A
//! face's bounds are the box of its patches' Bezier control points (the limit surface lies
//! in their hull), grown by its section's declared displacement range.

#[cfg(feature = "subd")]
mod build;
#[cfg(feature = "subd")]
mod osd;

use orr_math::{Vec2, Vec3};
use orr_pack::material::Material;
use orr_pack::subd::{Influence, SubdInfo};
use serde::Deserialize;

/// A Catmull-Clark cage in model space, built by a library caller.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CageInput {
    pub positions: Vec<Vec3>,
    /// Corner count per face, 3 to 255.
    pub face_counts: Vec<u32>,
    /// Point per face corner, faces in order.
    pub face_indices: Vec<u32>,
    /// Section (material) per face; empty puts every face in section 0.
    pub face_sections: Vec<u16>,
    pub creases: Vec<Crease>,
    pub corners: Vec<Corner>,
    /// Faces left out of the surface.
    pub holes: Vec<u32>,
    pub uvs: Option<FaceVarying>,
    /// Faces wind clockwise.
    pub left_handed: bool,
    pub boundary: Boundary,
    pub uv_linear: UvLinear,
    /// Triangles use the smooth-triangle rule rather than plain Catmull-Clark.
    pub smooth_triangles: bool,
}

/// An edge between two points and its sharpness; 10 and above is infinitely sharp.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Crease {
    pub edge: [u32; 2],
    pub sharpness: f32,
}

/// A sharp corner point and its sharpness.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Corner {
    pub point: u32,
    pub sharpness: f32,
}

/// Face-varying uvs, top-left origin: values and one value per face corner.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FaceVarying {
    pub values: Vec<Vec2>,
    pub indices: Vec<u32>,
}

/// How boundary edges and corners interpolate.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Boundary {
    /// Boundary faces are not part of the surface.
    None,
    /// Boundary edges are sharp.
    Edges,
    /// Boundary edges are sharp, and so are corners of two faces or fewer.
    #[default]
    EdgesAndCorners,
}

/// Where face-varying uvs interpolate linearly rather than smoothly.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UvLinear {
    /// Smooth everywhere.
    None = 0,
    /// Linear at corners of uv islands.
    CornersOnly = 1,
    /// Linear at corners, at junctions of three or more islands and at darts.
    #[default]
    CornersPlus1 = 2,
    /// As `CornersPlus1`, and at concave corners.
    CornersPlus2 = 3,
    /// Linear along island boundaries.
    Boundaries = 4,
    /// Linear everywhere.
    All = 5,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SubdSettings {
    /// Refinement levels around creases, corners and boundaries, 1 to 10.
    pub isolation: u8,
    /// Refinement levels around smooth extraordinary vertices, 1 to `isolation`.
    pub smooth_isolation: u8,
}

impl Default for SubdSettings {
    fn default() -> Self {
        Self {
            isolation: 4,
            smooth_isolation: 2,
        }
    }
}

/// A cage with what the importer adds: skin influences per point.
pub(crate) struct CageData {
    pub cage: CageInput,
    pub skin: Option<Vec<Influence>>,
}

impl CageData {
    pub(crate) fn hash(&self) -> orr_pack::Hash32 {
        orr_pack::Hash32::of(format!("{:?} {:?}", self.cage, self.skin).as_bytes())
    }

    pub(crate) fn faces(&self) -> u64 {
        self.cage.face_counts.len() as u64
    }
}

/// A cooked cage: blob 0 and the pages.
pub(crate) struct Cooked {
    pub info: SubdInfo,
    pub pages: Vec<Vec<u8>>,
}

/// Checks the cage against its section count.
pub(crate) fn check(cage: &CageInput, sections: usize) -> Result<(), String> {
    let n = cage.positions.len();
    if cage.face_counts.is_empty() {
        return Err("the cage has no faces".into());
    }
    if let Some(c) = cage.face_counts.iter().find(|&&c| !(3..=255).contains(&c)) {
        return Err(format!("a face has {c} corners; faces have 3 to 255"));
    }
    let corners: u64 = cage.face_counts.iter().map(|&c| u64::from(c)).sum();
    if corners != cage.face_indices.len() as u64 {
        return Err(format!(
            "face counts add up to {corners} corners, the cage has {} indices",
            cage.face_indices.len()
        ));
    }
    if let Some(i) = cage.face_indices.iter().find(|&&i| i as usize >= n) {
        return Err(format!("face corner names point {i} of {n}"));
    }
    if !cage.face_sections.is_empty() && cage.face_sections.len() != cage.face_counts.len() {
        return Err(format!(
            "{} face sections for {} faces",
            cage.face_sections.len(),
            cage.face_counts.len()
        ));
    }
    if let Some(s) = cage
        .face_sections
        .iter()
        .find(|&&s| usize::from(s) >= sections)
    {
        return Err(format!("a face is in section {s} of {sections}"));
    }
    let point = |i: u32| (i as usize) < n;
    if cage.creases.iter().any(|c| !c.edge.into_iter().all(point)) {
        return Err("a crease names a point past the cage".into());
    }
    if cage.corners.iter().any(|c| !point(c.point)) {
        return Err("a corner names a point past the cage".into());
    }
    if let Some(h) = cage
        .holes
        .iter()
        .find(|&&h| h as usize >= cage.face_counts.len())
    {
        return Err(format!("hole {h} is past the faces"));
    }
    let sharp = cage.creases.iter().map(|c| c.sharpness);
    if sharp
        .chain(cage.corners.iter().map(|c| c.sharpness))
        .any(|s| s.is_nan() || s < 0.0)
    {
        return Err("a sharpness is negative or not a number".into());
    }
    if let Some(uv) = &cage.uvs {
        if uv.indices.len() != cage.face_indices.len() {
            return Err(format!(
                "{} uv indices for {} face corners",
                uv.indices.len(),
                cage.face_indices.len()
            ));
        }
        if let Some(i) = uv.indices.iter().find(|&&i| i as usize >= uv.values.len()) {
            return Err(format!("a uv index names value {i} of {}", uv.values.len()));
        }
    }
    Ok(())
}

#[cfg(feature = "subd")]
pub(crate) fn cook(
    data: &CageData,
    materials: &[Material],
    settings: &SubdSettings,
) -> Result<Cooked, String> {
    if !(1..=10).contains(&settings.isolation)
        || !(1..=settings.isolation).contains(&settings.smooth_isolation)
    {
        return Err(format!(
            "isolation {} and smooth isolation {}: 1 <= smooth <= isolation <= 10",
            settings.isolation, settings.smooth_isolation
        ));
    }
    let cage = &data.cage;
    check(cage, materials.len())?;
    if let Some(skin) = &data.skin
        && skin.len() != cage.positions.len()
    {
        return Err(format!(
            "{} skin influences for {} points",
            skin.len(),
            cage.positions.len()
        ));
    }
    let refined = osd::refine(cage, settings)?;
    build::build(data, materials, settings, &refined)
}

#[cfg(not(feature = "subd"))]
pub(crate) fn cook(_: &CageData, _: &[Material], _: &SubdSettings) -> Result<Cooked, String> {
    Err("subdivision cages need orr-cook's subd feature".into())
}
