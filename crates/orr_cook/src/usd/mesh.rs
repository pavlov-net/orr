//! `UsdGeomMesh` faces as welded triangles: ngons by ear clipping, holes dropped, normals
//! and `st` by their interpolation (where no normals are authored, face normals on the
//! polygons of schemes `none` and `bilinear` and smooth point normals on other hulls), uvs
//! flipped from USD's bottom-left origin to the cook's top-left. Or as a subdivision cage
//! ([`CageBuilder`]): faces as authored with their creases, corners, holes and boundary
//! rules, and its uv sets and each primvar a material reads through the vertex color stream as
//! their interpolation asks: `vertex` on the surface's basis, `faceVarying` on its own topology
//! (unindexed values weld where a point's corners agree), `varying`, `uniform` and `constant`
//! linear over each face.
//! Tangents are the frame primvars a material names, or MikkTSpace-compatible ones from its
//! frame's uvs, in the mesh's own space; a cage's come from its limit surface on the GPU.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use orr_math::{DMat3, DMat4, DVec3, Vec2, Vec4};
use orr_pack::subd::Influence as SkinInfluence;

use super::read::{Interp, PrimReader, Primvar};
use crate::import::normal_matrix;
use crate::mesh::tangents;
use crate::shading::VertexRead;
use crate::subd::{
    Boundary, CageColor, CageInput, CageUv, CageValues, Corner, Crease, FaceVarying, UvLinear,
};

/// The authored topology and attributes of one mesh prim.
pub(super) struct MeshPrim {
    pub(super) counts: Vec<usize>,
    indices: Vec<usize>,
    pub(super) points: Vec<DVec3>,
    normals: Option<Primvar>,
    uvs: Option<Primvar>,
    uvs1: Option<Primvar>,
    left_handed: bool,
    holes: Vec<usize>,
    scheme: String,
    creases: Vec<Crease>,
    corners: Vec<Corner>,
    boundary: Boundary,
    uv_linear: UvLinear,
    smooth_triangles: bool,
}

/// What a mesh prim says of itself without any of its arrays.
pub(super) struct MeshFacts {
    /// Its surface is its faces, which no scheme smooths.
    pub(super) polygonal: bool,
    pub(super) double_sided: bool,
}

impl MeshFacts {
    pub(super) fn read(r: &PrimReader<'_>) -> Result<Self, String> {
        Ok(Self {
            polygonal: polygonal(&scheme(r)?),
            double_sided: double_sided(r)?,
        })
    }
}

fn scheme(r: &PrimReader<'_>) -> Result<String, String> {
    Ok(r.token("subdivisionScheme")?
        .unwrap_or_else(|| "catmullClark".into()))
}

fn polygonal(scheme: &str) -> bool {
    matches!(scheme, "none" | "bilinear")
}

fn double_sided(r: &PrimReader<'_>) -> Result<bool, String> {
    Ok(r.numbers("doubleSided")?.is_some_and(|v| v[0] != 0.0))
}

impl MeshPrim {
    /// `uv` and `uv1` name the primvars of uv sets 0 and 1.
    pub(super) fn read(r: &PrimReader<'_>, uv: &str, uv1: Option<&str>) -> Result<Self, String> {
        let path = r.path();
        let points: Vec<DVec3> = r
            .numbers("points")?
            .ok_or_else(|| format!("{path}: no points"))?
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&p| DVec3::from_array(p))
            .collect();
        let usize_list = |name: &str| -> Result<Vec<usize>, String> {
            r.ints(name)?
                .unwrap_or_default()
                .into_iter()
                .map(|i| usize::try_from(i).map_err(|_| format!("{path} {name}: {i}")))
                .collect()
        };
        let counts = usize_list("faceVertexCounts")?;
        let indices = usize_list("faceVertexIndices")?;
        if counts.iter().sum::<usize>() != indices.len() {
            return Err(format!(
                "{path}: face counts do not add up to the index count"
            ));
        }
        if let Some(i) = indices.iter().find(|&&i| i >= points.len()) {
            return Err(format!(
                "{path}: face vertex index {i} past {} points",
                points.len()
            ));
        }
        let index = |i: i64, what: &str| {
            u32::try_from(i)
                .ok()
                .filter(|&i| (i as usize) < points.len())
                .ok_or_else(|| format!("{path} {what}: point {i}"))
        };
        let mut creases = Vec::new();
        let chain = r.ints("creaseIndices")?.unwrap_or_default();
        let lengths = r.ints("creaseLengths")?.unwrap_or_default();
        let sharp = r.numbers("creaseSharpnesses")?.unwrap_or_default();
        let edges: i64 = lengths.iter().map(|&l| (l - 1).max(0)).sum();
        let per_edge = sharp.len() as i64 == edges && sharp.len() != lengths.len();
        if !per_edge && sharp.len() != lengths.len() {
            return Err(format!(
                "{path}: {} crease sharpnesses for {} creases",
                sharp.len(),
                lengths.len()
            ));
        }
        let (mut at, mut edge) = (0usize, 0usize);
        for (c, &len) in lengths.iter().enumerate() {
            let len = usize::try_from(len).map_err(|_| format!("{path}: crease length {len}"))?;
            let run = chain
                .get(at..at + len)
                .ok_or_else(|| format!("{path}: crease lengths pass the crease indices"))?;
            for pair in run.windows(2) {
                creases.push(Crease {
                    edge: [
                        index(pair[0], "creaseIndices")?,
                        index(pair[1], "creaseIndices")?,
                    ],
                    sharpness: sharp[if per_edge { edge } else { c }] as f32,
                });
                edge += 1;
            }
            at += len;
        }
        let corner_points = r.ints("cornerIndices")?.unwrap_or_default();
        let corner_sharp = r.numbers("cornerSharpnesses")?.unwrap_or_default();
        if corner_points.len() != corner_sharp.len() {
            return Err(format!(
                "{path}: corner indices and sharpnesses differ in count"
            ));
        }
        let corners = corner_points
            .iter()
            .zip(&corner_sharp)
            .map(|(&p, &s)| {
                Ok(Corner {
                    point: index(p, "cornerIndices")?,
                    sharpness: s as f32,
                })
            })
            .collect::<Result<_, String>>()?;
        let boundary = match r.token("interpolateBoundary")?.as_deref() {
            Some("none") => Boundary::None,
            Some("edgeOnly") => Boundary::Edges,
            Some("edgeAndCorner") | None => Boundary::EdgesAndCorners,
            Some(t) => return Err(format!("{path}: interpolateBoundary {t}")),
        };
        let uv_linear = match r.token("faceVaryingLinearInterpolation")?.as_deref() {
            Some("none") => UvLinear::None,
            Some("cornersOnly") => UvLinear::CornersOnly,
            Some("cornersPlus1") | None => UvLinear::CornersPlus1,
            Some("cornersPlus2") => UvLinear::CornersPlus2,
            Some("boundaries") => UvLinear::Boundaries,
            Some("all") => UvLinear::All,
            Some(t) => return Err(format!("{path}: faceVaryingLinearInterpolation {t}")),
        };
        let smooth_triangles = match r.token("triangleSubdivisionRule")?.as_deref() {
            Some("smooth") => true,
            Some("catmullClark") | None => false,
            Some(t) => return Err(format!("{path}: triangleSubdivisionRule {t}")),
        };
        Ok(Self {
            creases,
            corners,
            boundary,
            uv_linear,
            smooth_triangles,
            counts,
            indices,
            points,
            normals: r.primvar("normals", 3, Some("normals"))?,
            uvs: r.primvar(uv, 2, None)?,
            uvs1: match uv1 {
                Some(name) => r.primvar(name, 2, None)?,
                None => None,
            },
            left_handed: r.token("orientation")?.as_deref() == Some("leftHanded"),
            holes: usize_list("holeIndices")?,
            scheme: scheme(r)?,
        })
    }

    /// Whether its surface is its faces, which no scheme smooths.
    pub(super) fn polygonal(&self) -> bool {
        polygonal(&self.scheme)
    }

    /// How many elements a primvar of interpolation `interp` holds over it.
    pub(super) fn domain(&self, interp: Interp) -> usize {
        match interp {
            Interp::Constant => 1,
            Interp::Uniform => self.counts.len(),
            Interp::Vertex | Interp::Varying => self.points.len(),
            Interp::FaceVarying => self.indices.len(),
        }
    }
}

/// One triangle of a face.
struct Triangle {
    face: usize,
    /// Per corner, as the face winds: its face-vertex and its point.
    corners: [(usize, usize); 3],
}

/// A polygonal mesh's triangles, without its holes, and its normals where none are authored.
pub(super) struct Faces {
    triangles: Vec<Triangle>,
    made: Vec<DVec3>,
}

impl Faces {
    pub(super) fn new(mesh: &MeshPrim) -> Self {
        let made = match &mesh.normals {
            Some(_) => Vec::new(),
            None if mesh.polygonal() => face_normals(mesh),
            None => smooth_normals(mesh),
        };
        let holes: std::collections::HashSet<usize> = mesh.holes.iter().copied().collect();
        let mut triangles = Vec::new();
        let mut base = 0;
        for (face, &count) in mesh.counts.iter().enumerate() {
            let ring: Vec<usize> = (base..base + count).collect();
            base += count;
            if count < 3 || holes.contains(&face) {
                continue;
            }
            let pts: Vec<DVec3> = ring.iter().map(|&c| mesh.points[mesh.indices[c]]).collect();
            triangles.extend(triangulate(&pts).into_iter().map(|t| Triangle {
                face,
                corners: t.map(|k| (ring[k], mesh.indices[ring[k]])),
            }));
        }
        Self { triangles, made }
    }

    /// The mesh-space normal of corner `c` of triangle `t`.
    fn normal(&self, mesh: &MeshPrim, t: &Triangle, c: usize, path: &str) -> Result<DVec3, String> {
        let (corner, point) = t.corners[c];
        Ok(match &mesh.normals {
            Some(nv) => {
                let v = nv
                    .at(point, t.face, corner)
                    .ok_or_else(|| format!("{path}: normals do not cover face {}", t.face))?;
                DVec3::from_slice(v)
            }
            None if mesh.polygonal() => self.made[t.face],
            None => self.made[point],
        })
    }
}

/// The uv of corner `c` of triangle `t` in `uvs`, zero without them, in the cook's top-left
/// origin.
fn corner_uv(
    uvs: Option<&Primvar>,
    t: &Triangle,
    c: usize,
    path: &str,
) -> Result<[f32; 2], String> {
    let (corner, point) = t.corners[c];
    let st = match uvs {
        Some(uv) => {
            let v = uv
                .at(point, t.face, corner)
                .ok_or_else(|| format!("{path}: uvs do not cover face {}", t.face))?;
            Vec2::new(v[0] as f32, v[1] as f32)
        }
        None => Vec2::ZERO,
    };
    Ok([st.x, 1.0 - st.y])
}

/// The uvs a face's tangent frame follows.
#[derive(Copy, Clone)]
pub(super) enum FrameUvs<'a> {
    /// A uv set.
    Set(usize),
    /// A primvar outside the uv sets, by name.
    Primvar(&'a str, &'a Primvar),
}

/// What a face's tangent frame reads, PS `frame:*`.
#[derive(Copy, Clone)]
pub(super) struct FaceFrame<'a> {
    pub(super) uvs: FrameUvs<'a>,
    pub(super) tangents: Option<&'a Primvar>,
    pub(super) binormals: Option<&'a Primvar>,
}

/// The tangent of each triangle corner of `faces` in the mesh's own space, xyz and bitangent
/// sign: `frame` states it per face, or the corners of faces on the same uvs make one
/// MikkTSpace-compatible surface. A sign no binormal states is the one that surface gives.
pub(super) fn corner_tangents<'a>(
    mesh: &'a MeshPrim,
    faces: &Faces,
    frame: &dyn Fn(usize) -> FaceFrame<'a>,
    path: &str,
) -> Result<Vec<[f32; 4]>, String> {
    let mut surfaces: BTreeMap<(usize, &str), (FrameUvs<'_>, Vec<usize>)> = BTreeMap::new();
    for (k, t) in faces.triangles.iter().enumerate() {
        let f = frame(t.face);
        if f.tangents.is_some() && f.binormals.is_some() {
            continue;
        }
        let key = match f.uvs {
            FrameUvs::Set(s) => (s, ""),
            FrameUvs::Primvar(name, _) => (2, name),
        };
        surfaces.entry(key).or_insert((f.uvs, Vec::new())).1.push(k);
    }
    // MikkTSpace reads the winding, which a left-handed mesh reverses.
    let order = if mesh.left_handed {
        [0, 2, 1]
    } else {
        [0, 1, 2]
    };
    let mut out = vec![tangents::UNREAD; 3 * faces.triangles.len()];
    for (uvs, triangles) in surfaces.into_values() {
        let source = match uvs {
            FrameUvs::Set(0) => mesh.uvs.as_ref(),
            FrameUvs::Set(_) => mesh.uvs1.as_ref(),
            FrameUvs::Primvar(_, p) => Some(p),
        };
        let (mut positions, mut normals, mut sts) = (Vec::new(), Vec::new(), Vec::new());
        for &k in &triangles {
            let t = &faces.triangles[k];
            for c in order {
                positions.push(mesh.points[t.corners[c].1].as_vec3().to_array());
                let n = faces.normal(mesh, t, c, path)?.normalize_or_zero();
                normals.push(n.as_vec3().to_array());
                sts.push(corner_uv(source, t, c, path)?);
            }
        }
        let made = tangents::generate(None, &positions, &normals, &sts);
        for (&k, made) in triangles.iter().zip(made.as_chunks::<3>().0) {
            for (&c, &t) in order.iter().zip(made) {
                out[3 * k + c] = t;
            }
        }
    }
    for (k, t) in faces.triangles.iter().enumerate() {
        let f = frame(t.face);
        let Some(stated) = f.tangents else { continue };
        for c in 0..3 {
            let (corner, point) = t.corners[c];
            let missing = |what: &str| format!("{path}: {what} do not cover face {}", t.face);
            let v = stated
                .at(point, t.face, corner)
                .ok_or_else(|| missing("tangents"))?;
            let tangent = DVec3::from_slice(v);
            let sign = match f.binormals {
                Some(b) => {
                    let b = b
                        .at(point, t.face, corner)
                        .ok_or_else(|| missing("binormals"))?;
                    let n = faces.normal(mesh, t, c, path)?;
                    if n.cross(tangent).dot(DVec3::from_slice(b)) < 0.0 {
                        -1.0
                    } else {
                        1.0
                    }
                }
                None => match v.get(3) {
                    Some(&w) if w < 0.0 => -1.0,
                    Some(_) => 1.0,
                    None => out[3 * k + c][3],
                },
            };
            let [x, y, z] = tangent.as_vec3().to_array();
            out[3 * k + c] = [x, y, z, sign];
        }
    }
    Ok(out)
}

/// Per-vertex skin influences: four joints and unorm16 weights.
pub(super) type Influence = ([u8; 4], [u16; 4]);

/// One model's welded vertices and its triangles per scene material.
#[derive(Default)]
pub(super) struct Welder {
    pub(super) positions: Vec<[f32; 3]>,
    pub(super) normals: Vec<[f32; 3]>,
    pub(super) tangents: Vec<[f32; 4]>,
    /// The model carries tangents, on every mesh: one states them or a material reads them,
    /// and a prototype's material can read them on the others.
    pub(super) carries_tangents: bool,
    pub(super) uvs: Vec<[f32; 2]>,
    pub(super) uvs1: Vec<[f32; 2]>,
    pub(super) colors: Vec<[f32; 4]>,
    /// A material of the model reads the vertex color stream.
    pub(super) carries_colors: bool,
    pub(super) joints: Vec<[u8; 4]>,
    /// Per vertex, its rigid part.
    pub(super) parts: Vec<u16>,
    pub(crate) weights: Vec<[u16; 4]>,
    pub(super) groups: BTreeMap<usize, Vec<u32>>,
    /// Scene materials drawn on a mesh without the uv set 1 primvar.
    pub(super) missing_uv1: BTreeSet<usize>,
    seen: HashMap<[u32; 22], u32>,
}

/// How a mesh's faces land in a model.
pub(super) struct Placing<'a> {
    /// Model space from the mesh's points.
    pub(super) transform: DMat4,
    /// Scene material per face.
    pub(super) face_material: &'a dyn Fn(usize) -> usize,
    /// Influences per point when skinned.
    pub(super) skin: Option<&'a [Influence]>,
    /// The rigid part the mesh makes up.
    pub(super) part: u16,
    /// The vertex color stream at a triangle's corner: its face, face corner and point.
    pub(super) color: &'a dyn Fn(usize, usize, usize) -> [f32; 4],
    /// What a cage's faces read of the stream.
    pub(super) color_reads: &'a [ColorRead<'a>],
}

/// A primvar faces of a cage read through the vertex color stream.
pub(super) struct ColorRead<'a> {
    pub(super) read: &'a VertexRead,
    /// What the mesh holds of it.
    pub(super) held: Option<&'a Primvar>,
    /// The faces that read it.
    pub(super) faces: Vec<usize>,
}

/// Which [`CageColor`] of a model a read lands in: meshes share one where they read a primvar
/// alike, so a model's face-varying channels don't grow with its meshes.
#[derive(PartialEq, Eq, Hash)]
struct ColorKey {
    name: String,
    first: u8,
    width: u8,
    fallback: [u32; 4],
    scale: u32,
    interpolation: CageBasis,
}

/// How a primvar interpolates over a cage, as [`CageValues`] states.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum CageBasis {
    Surface,
    Own,
    Linear,
}

impl CageBasis {
    /// `UsdGeomPrimvar`: `vertex` takes "the basis function of the surface", `varying` "bilinear
    /// interpolation", and only `faceVarying` values may differ across an edge. A primvar the mesh
    /// doesn't hold reads its fallback, which is linear.
    fn of(held: Option<&Primvar>) -> Self {
        match held.map(|p| p.interp) {
            Some(Interp::Vertex) => Self::Surface,
            Some(Interp::FaceVarying) => Self::Own,
            _ => Self::Linear,
        }
    }

    /// Values of this basis with no mesh in them yet. Corners of a mesh that doesn't read them
    /// take value 0.
    fn values<T: Default>(self) -> CageValues<T> {
        let corners = || FaceVarying {
            values: vec![T::default()],
            indices: Vec::new(),
        };
        match self {
            Self::Surface => CageValues::Vertex(Vec::new()),
            Self::Own => CageValues::FaceVarying(corners()),
            Self::Linear => CageValues::Linear(corners()),
        }
    }
}

impl ColorKey {
    fn new(r: &ColorRead<'_>) -> Self {
        Self {
            name: r.read.name.clone(),
            first: r.read.first,
            width: r.read.width,
            fallback: r.read.fallback.map(f32::to_bits),
            scale: r.read.scale.to_bits(),
            interpolation: CageBasis::of(r.held),
        }
    }

    /// The color it names, with no mesh in it yet.
    fn color(&self) -> CageColor {
        CageColor {
            values: self.interpolation.values(),
            channels: ((1 << self.width) - 1) << self.first,
            faces: Vec::new(),
        }
    }
}

impl Welder {
    fn vertex(
        &mut self,
        p: [f32; 3],
        n: [f32; 3],
        t: [f32; 4],
        uv: [[f32; 2]; 2],
        color: [f32; 4],
        (skin, part): (Influence, u16),
    ) -> u32 {
        let mut key = [0u32; 22];
        let values = p.iter().chain(&n).chain(&t).chain(uv.as_flattened());
        for (k, v) in key.iter_mut().zip(values) {
            *k = v.to_bits();
        }
        let w = skin.1.map(u32::from);
        key[14] = u32::from_le_bytes(skin.0);
        key[15] = u32::from(part);
        key[16] = w[0] | w[1] << 16;
        key[17] = w[2] | w[3] << 16;
        key[18..].copy_from_slice(&color.map(f32::to_bits));
        if let Some(&i) = self.seen.get(&key) {
            return i;
        }
        let i = self.positions.len() as u32;
        self.positions.push(p);
        self.normals.push(n);
        self.tangents.push(t);
        self.uvs.push(uv[0]);
        self.uvs1.push(uv[1]);
        self.colors.push(color);
        self.joints.push(skin.0);
        self.weights.push(skin.1);
        self.parts.push(part);
        self.seen.insert(key, i);
        i
    }

    /// Adds the mesh's triangles, with `corners` from [`corner_tangents`].
    pub(super) fn add(
        &mut self,
        mesh: &MeshPrim,
        faces: &Faces,
        corners: Option<&[[f32; 4]]>,
        at: &Placing<'_>,
        path: &str,
    ) -> Result<(), String> {
        let linear = DMat3::from_mat4(at.transform);
        let mirrors = linear.determinant() < 0.0;
        let flip = mirrors != mesh.left_handed;
        let normal_m = normal_matrix(linear);
        for (k, t) in faces.triangles.iter().enumerate() {
            let material = (at.face_material)(t.face);
            let mut out = [0u32; 3];
            for (c, o) in out.iter_mut().enumerate() {
                let point = t.corners[c].1;
                let p = at.transform.transform_point3(mesh.points[point]);
                let n = (normal_m * faces.normal(mesh, t, c, path)?).normalize_or_zero();
                let tangent = corners.map_or(tangents::UNREAD, |corners| {
                    let [x, y, z, w] = corners[3 * k + c];
                    let v = linear * DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                    let [x, y, z] = v.as_vec3().to_array();
                    [x, y, z, if mirrors { -w } else { w }]
                });
                let uv = [
                    corner_uv(mesh.uvs.as_ref(), t, c, path)?,
                    corner_uv(mesh.uvs1.as_ref(), t, c, path)?,
                ];
                let skin = at.skin.map_or(([0; 4], [0; 4]), |s| s[point]);
                *o = self.vertex(
                    p.as_vec3().to_array(),
                    n.as_vec3().to_array(),
                    tangent,
                    uv,
                    (at.color)(t.face, t.corners[c].0, point),
                    (skin, at.part),
                );
            }
            if flip {
                out.swap(1, 2);
            }
            if mesh.uvs1.is_none() {
                self.missing_uv1.insert(material);
            }
            self.groups.entry(material).or_default().extend(out);
        }
        Ok(())
    }
}

/// One model's subdivision cage, appended mesh by mesh, in model space and right-handed.
#[derive(Default)]
pub(super) struct CageBuilder {
    pub(super) cage: CageInput,
    pub(super) skin: Vec<SkinInfluence>,
    /// Scene material per face.
    pub(super) materials: Vec<usize>,
    /// Each uv set's place in the cage's uvs, per basis: meshes of a model share one.
    uvs: HashMap<(u8, CageBasis), usize>,
    /// Each read's place in the cage's colors.
    colors: HashMap<ColorKey, usize>,
    /// The first mesh's path, naming the rules every other must share.
    first: Option<String>,
}

impl CageBuilder {
    pub(super) fn is_empty(&self) -> bool {
        self.cage.face_counts.is_empty()
    }

    /// Adds the mesh's faces, creases, corners, holes and uvs.
    pub(super) fn add(
        &mut self,
        mesh: &MeshPrim,
        at: &Placing<'_>,
        path: &str,
    ) -> Result<(), String> {
        if mesh.scheme != "catmullClark" {
            return Err(format!(
                "{path}: subdivision scheme {}; cages cook as Catmull-Clark",
                mesh.scheme
            ));
        }
        let rules = (mesh.boundary, mesh.uv_linear, mesh.smooth_triangles);
        let c = &mut self.cage;
        match &self.first {
            None => {
                (c.boundary, c.uv_linear, c.smooth_triangles) = rules;
                self.first = Some(path.into());
            }
            Some(first) if (c.boundary, c.uv_linear, c.smooth_triangles) != rules => {
                return Err(format!(
                    "{path}: its boundary, face-varying or triangle rules differ from {first}'s \
                     in the same model; select them as separate models"
                ));
            }
            Some(_) => {}
        }
        let flip = (DMat3::from_mat4(at.transform).determinant() < 0.0) != mesh.left_handed;
        let base = c.positions.len() as u32;
        let first_face = c.face_counts.len() as u32;
        let first_corner = c.face_indices.len();
        for &p in &mesh.points {
            c.positions.push(at.transform.transform_point3(p).as_vec3());
        }
        if let Some(s) = at.skin {
            self.skin.extend(
                s.iter()
                    .map(|&(joints, weights)| SkinInfluence { joints, weights }),
            );
        }
        let mut corner = 0;
        // Face corners as the cage holds them.
        let mut read_order = Vec::with_capacity(mesh.indices.len());
        for (face, &count) in mesh.counts.iter().enumerate() {
            self.materials.push((at.face_material)(face));
            c.face_counts.push(count as u32);
            let ring: Vec<usize> = if flip {
                std::iter::once(corner)
                    .chain((corner + 1..corner + count).rev())
                    .collect()
            } else {
                (corner..corner + count).collect()
            };
            corner += count;
            read_order.extend(&ring);
            c.face_indices
                .extend(ring.into_iter().map(|k| base + mesh.indices[k] as u32));
        }
        for cr in &mesh.creases {
            c.creases.push(Crease {
                edge: cr.edge.map(|p| base + p),
                sharpness: cr.sharpness,
            });
        }
        for co in &mesh.corners {
            c.corners.push(Corner {
                point: base + co.point,
                sharpness: co.sharpness,
            });
        }
        let holes: BTreeSet<usize> = mesh.holes.iter().copied().collect();
        for h in holes {
            if h >= mesh.counts.len() {
                return Err(format!("{path}: hole {h} past the faces"));
            }
            c.holes.push(first_face + h as u32);
        }
        let spot = CageSpot {
            read_order: &read_order,
            first_point: base as usize,
            first_corner,
        };
        self.add_uvs(mesh, &spot, first_face, path)?;
        self.add_colors(mesh, at, &spot, first_face)?;
        let c = &mut self.cage;
        let (points, corners) = (c.positions.len(), c.face_indices.len());
        for uv in &mut c.uvs {
            pad(&mut uv.values, points, corners);
        }
        for color in &mut c.colors {
            pad(&mut color.values, points, corners);
        }
        Ok(())
    }

    /// Adds the mesh's uv sets. A face of a mesh without a set reads uv 0.
    fn add_uvs(
        &mut self,
        mesh: &MeshPrim,
        spot: &CageSpot<'_>,
        first_face: u32,
        path: &str,
    ) -> Result<(), String> {
        let c = &mut self.cage;
        let first_faces = first_face..first_face + mesh.counts.len() as u32;
        for (set, held) in [(0, &mesh.uvs), (1, &mesh.uvs1)] {
            let Some(held) = held else { continue };
            let basis = CageBasis::of(Some(held));
            let l = *self.uvs.entry((set, basis)).or_insert_with(|| {
                c.uvs.push(CageUv {
                    values: basis.values(),
                    set,
                    faces: Vec::new(),
                });
                c.uvs.len() - 1
            });
            let uv = &mut c.uvs[l];
            uv.faces.extend(first_faces.clone());
            let element = |v: Option<&[f64]>, face: usize| {
                let v = v.ok_or_else(|| format!("{path}: uvs do not cover face {face}"))?;
                Ok(Vec2::new(v[0] as f32, 1.0 - v[1] as f32))
            };
            add_values(&mut uv.values, mesh, Some(held), element, spot)?;
        }
        Ok(())
    }

    /// Adds what the mesh's faces read of the vertex color stream, its faces starting at
    /// `first_face`. A color spans the cage, so the values of faces that don't read it stay
    /// beside those of faces that do, and the surface between them shapes both.
    fn add_colors(
        &mut self,
        mesh: &MeshPrim,
        at: &Placing<'_>,
        spot: &CageSpot<'_>,
        first_face: u32,
    ) -> Result<(), String> {
        let c = &mut self.cage;
        for r in at.color_reads {
            let element = |v: Option<&[f64]>, _: usize| {
                let mut color = [0.0; 4];
                r.read.put(v, &mut color);
                Ok(Vec4::from_array(color))
            };
            let l = *self
                .colors
                .entry(ColorKey::new(r))
                .or_insert_with_key(|key| {
                    c.colors.push(key.color());
                    c.colors.len() - 1
                });
            let color = &mut c.colors[l];
            color
                .faces
                .extend(r.faces.iter().map(|&f| first_face + f as u32));
            add_values(&mut color.values, mesh, r.held, element, spot)?;
        }
        Ok(())
    }
}

/// Where a mesh lands in a cage: its corners in the cage's order, and its first point and corner.
struct CageSpot<'a> {
    read_order: &'a [usize],
    first_point: usize,
    first_corner: usize,
}

/// Adds a mesh's `held` primvar to `values`, the elements of the mesh and face `element` takes;
/// without a primvar it takes none, on every corner.
fn add_values<T: Default + Clone>(
    values: &mut CageValues<T>,
    mesh: &MeshPrim,
    held: Option<&Primvar>,
    element: impl Fn(Option<&[f64]>, usize) -> Result<T, String>,
    at: &CageSpot<'_>,
) -> Result<(), String> {
    match (values, held) {
        (CageValues::Vertex(values), Some(held)) => {
            values.resize(at.first_point, T::default());
            for p in 0..mesh.points.len() {
                values.push(element(held.at(p, 0, 0), 0)?);
            }
        }
        (CageValues::Vertex(_), None) => {
            unreachable!("a primvar the mesh doesn't hold has no point values")
        }
        (CageValues::FaceVarying(corners) | CageValues::Linear(corners), held) => {
            let (values, index) = match held {
                Some(p) => corner_values(mesh, p, &element)?,
                None => (vec![element(None, 0)?], vec![0; mesh.indices.len()]),
            };
            let first = corners.values.len() as u32;
            corners.values.extend(values);
            corners.indices.resize(at.first_corner, 0);
            corners
                .indices
                .extend(at.read_order.iter().map(|&k| first + index[k]));
        }
    }
    Ok(())
}

/// Gives `values` an entry for every point and face corner of the cage so far.
fn pad<T: Default + Clone>(values: &mut CageValues<T>, points: usize, corners: usize) {
    match values {
        CageValues::Vertex(values) => values.resize(points, T::default()),
        CageValues::FaceVarying(c) | CageValues::Linear(c) => c.indices.resize(corners, 0),
    }
}

/// A primvar's values by `value`, which takes an element and its face, and one value index per
/// face corner: indexed and point primvars keep their sharing, unindexed face-varying values
/// weld where a point's corners agree in the four numbers a reader can take of them.
fn corner_values<T>(
    mesh: &MeshPrim,
    p: &Primvar,
    value: impl Fn(Option<&[f64]>, usize) -> Result<T, String>,
) -> Result<(Vec<T>, Vec<u32>), String> {
    let mut values = Vec::new();
    let mut index = Vec::with_capacity(mesh.indices.len());
    let mut seen: HashMap<(usize, Option<[u64; 4]>), u32> = HashMap::new();
    let mut corner = 0;
    for (face, &count) in mesh.counts.iter().enumerate() {
        for (k, &point) in (corner..).zip(&mesh.indices[corner..corner + count]) {
            let v = p.at(point, face, k);
            let element = match (p.interp, &p.indices) {
                (Interp::FaceVarying, Some(ix)) => ix.get(k).map_or(usize::MAX, |&i| i as usize),
                (Interp::FaceVarying, None) | (Interp::Vertex | Interp::Varying, _) => point,
                (Interp::Uniform, _) => usize::MAX - face,
                (Interp::Constant, _) => usize::MAX,
            };
            let bits = v.map(|v| std::array::from_fn(|i| v.get(i).map_or_default(|x| x.to_bits())));
            let key = (element, bits);
            let id = match seen.get(&key) {
                Some(&id) => id,
                None => {
                    values.push(value(v, face)?);
                    let id = values.len() as u32 - 1;
                    seen.insert(key, id);
                    id
                }
            };
            index.push(id);
        }
        corner += count;
    }
    Ok((values, index))
}

/// Per face, Newell's normal: twice the area along the face's normal.
fn face_normals(mesh: &MeshPrim) -> Vec<DVec3> {
    let mut base = 0;
    let faces = mesh.counts.iter().map(|&count| {
        let ring = &mesh.indices[base..base + count];
        base += count;
        let mut n = DVec3::ZERO;
        for (i, &a) in ring.iter().enumerate() {
            let (p, q) = (mesh.points[a], mesh.points[ring[(i + 1) % ring.len()]]);
            n += DVec3::new(
                (p.y - q.y) * (p.z + q.z),
                (p.z - q.z) * (p.x + q.x),
                (p.x - q.x) * (p.y + q.y),
            );
        }
        if mesh.left_handed { -n } else { n }
    });
    faces.collect()
}

/// Area-weighted point normals from the faces around each point.
fn smooth_normals(mesh: &MeshPrim) -> Vec<DVec3> {
    let mut acc = vec![DVec3::ZERO; mesh.points.len()];
    let mut base = 0;
    for (&count, n) in mesh.counts.iter().zip(face_normals(mesh)) {
        for &a in &mesh.indices[base..base + count] {
            acc[a] += n;
        }
        base += count;
    }
    acc.into_iter().map(DVec3::normalize_or_zero).collect()
}

/// Triangles of a polygon as corner triplets, wound like the polygon: a fan for triangles
/// and convex quads, ear clipping otherwise (a fan where clipping finds no ear).
fn triangulate(pts: &[DVec3]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n == 3 {
        return vec![[0, 1, 2]];
    }
    let mut normal = DVec3::ZERO;
    for i in 0..n {
        let (p, q) = (pts[i], pts[(i + 1) % n]);
        normal += p.cross(q);
    }
    let fan = || (1..n - 1).map(|i| [0, i, i + 1]).collect();
    if normal.length_squared() == 0.0 {
        return fan();
    }
    let turn = |a: DVec3, b: DVec3, c: DVec3| (b - a).cross(c - b).dot(normal);
    let convex = |a, b, c| turn(a, b, c) > 0.0;
    let mut ring: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(n - 2);
    while ring.len() > 3 {
        let m = ring.len();
        let ear = (0..m).find(|&i| {
            let (a, b, c) = (ring[(i + m - 1) % m], ring[i], ring[(i + 1) % m]);
            if !convex(pts[a], pts[b], pts[c]) {
                return false;
            }
            // No other corner inside the ear or on its edges.
            ring.iter().all(|&j| {
                j == a
                    || j == b
                    || j == c
                    || !(turn(pts[a], pts[b], pts[j]) >= 0.0
                        && turn(pts[b], pts[c], pts[j]) >= 0.0
                        && turn(pts[c], pts[a], pts[j]) >= 0.0)
            })
        });
        let Some(i) = ear else {
            return fan();
        };
        out.push([ring[(i + m - 1) % m], ring[i], ring[(i + 1) % m]]);
        ring.remove(i);
    }
    out.push([ring[0], ring[1], ring[2]]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(pts: &[DVec3], tris: &[[usize; 3]]) -> f64 {
        tris.iter()
            .map(|t| (pts[t[1]] - pts[t[0]]).cross(pts[t[2]] - pts[t[0]]).z / 2.0)
            .sum()
    }

    #[test]
    fn ear_clipping_keeps_concave_polygons_inside() {
        // An L shape, counter-clockwise, area 3; a fan from corner 0 would leave it.
        let pts: Vec<DVec3> = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
            .iter()
            .map(|&[x, y]| DVec3::new(f64::from(x), f64::from(y), 0.0))
            .collect();
        let tris = triangulate(&pts);
        assert_eq!(tris.len(), 4);
        assert!((area(&pts, &tris) - 3.0).abs() < 1e-12);
        for t in &tris {
            let a = (pts[t[1]] - pts[t[0]]).cross(pts[t[2]] - pts[t[0]]).z;
            assert!(a > 0.0, "{t:?} keeps the polygon's winding");
        }
    }
}
