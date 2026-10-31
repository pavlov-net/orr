//! `UsdGeomMesh` faces as welded triangles: ngons by ear clipping, holes dropped, normals
//! and `st` by their interpolation (smooth point normals where none are authored), uvs
//! flipped from USD's bottom-left origin to the cook's top-left. Or as a subdivision cage
//! ([`CageBuilder`]): faces as authored with their creases, corners, holes and boundary
//! rules, `st` as face-varying values (unindexed values weld where a point's corners agree).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use orr_math::{Affine2, DMat3, DMat4, DVec3, Vec2};
use orr_pack::subd::Influence as SkinInfluence;

use super::read::{Interp, PrimReader, Primvar};
use crate::subd::{Boundary, CageInput, Corner, Crease, FaceVarying, UvLinear};

/// The authored topology and attributes of one mesh prim.
pub(super) struct MeshPrim {
    pub counts: Vec<usize>,
    pub indices: Vec<usize>,
    pub points: Vec<DVec3>,
    pub normals: Option<Primvar>,
    pub uvs: Option<Primvar>,
    pub left_handed: bool,
    pub holes: Vec<usize>,
    pub scheme: String,
    pub double_sided: bool,
    pub creases: Vec<Crease>,
    pub corners: Vec<Corner>,
    pub boundary: Boundary,
    pub uv_linear: UvLinear,
    pub smooth_triangles: bool,
}

impl MeshPrim {
    pub(super) fn read(r: &PrimReader<'_>, uv: &str) -> Result<Self, String> {
        let path = r.path();
        let points: Vec<DVec3> = r
            .numbers("points")?
            .ok_or_else(|| format!("{path}: no points"))?
            .chunks_exact(3)
            .map(DVec3::from_slice)
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
            left_handed: r.token("orientation")?.as_deref() == Some("leftHanded"),
            holes: usize_list("holeIndices")?,
            scheme: r
                .token("subdivisionScheme")?
                .unwrap_or_else(|| "catmullClark".into()),
            double_sided: r.numbers("doubleSided")?.is_some_and(|v| v[0] != 0.0),
        })
    }
}

/// Per-vertex skin influences: four joints and unorm8 weights.
pub(super) type Influence = ([u8; 4], [u8; 4]);

/// One model's welded vertices and its triangles per scene material.
#[derive(Default)]
pub(super) struct Welder {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub joints: Vec<[u8; 4]>,
    pub weights: Vec<[u8; 4]>,
    pub groups: BTreeMap<usize, Vec<u32>>,
    seen: HashMap<[u32; 16], u32>,
}

/// How a mesh's faces land in a model.
pub(super) struct Placing<'a> {
    /// Model space from the mesh's points.
    pub transform: DMat4,
    /// Scene material and uv transform per face.
    pub face_material: &'a dyn Fn(usize) -> (usize, Option<Affine2>),
    /// Influences per point when skinned.
    pub skin: Option<&'a [Influence]>,
}

impl Welder {
    fn vertex(&mut self, p: [f32; 3], n: [f32; 3], uv: [f32; 2], skin: Influence) -> u32 {
        let mut key = [0u32; 16];
        for (k, v) in key.iter_mut().zip(p.iter().chain(&n).chain(&uv)) {
            *k = v.to_bits();
        }
        key[8] = u32::from_le_bytes(skin.0);
        key[9] = u32::from_le_bytes(skin.1);
        if let Some(&i) = self.seen.get(&key) {
            return i;
        }
        let i = self.positions.len() as u32;
        self.positions.push(p);
        self.normals.push(n);
        self.uvs.push(uv);
        self.joints.push(skin.0);
        self.weights.push(skin.1);
        self.seen.insert(key, i);
        i
    }

    /// Adds the mesh's triangles.
    pub(super) fn add(
        &mut self,
        mesh: &MeshPrim,
        at: &Placing<'_>,
        path: &str,
    ) -> Result<(), String> {
        let linear = DMat3::from_mat4(at.transform);
        let flip = (linear.determinant() < 0.0) != mesh.left_handed;
        let normal_m = linear.inverse().transpose();
        let smooth = match &mesh.normals {
            Some(_) => Vec::new(),
            None => smooth_normals(mesh),
        };
        let holes: std::collections::HashSet<usize> = mesh.holes.iter().copied().collect();
        let mut base = 0;
        for (face, &count) in mesh.counts.iter().enumerate() {
            let corners = base..base + count;
            base += count;
            if count < 3 || holes.contains(&face) {
                continue;
            }
            let (material, uv_t) = (at.face_material)(face);
            let ring: Vec<usize> = corners.clone().collect();
            let pts: Vec<DVec3> = ring.iter().map(|&c| mesh.points[mesh.indices[c]]).collect();
            for tri in triangulate(&pts) {
                let mut out = [0u32; 3];
                for (o, &k) in out.iter_mut().zip(&tri) {
                    let corner = ring[k];
                    let point = mesh.indices[corner];
                    let p = at.transform.transform_point3(mesh.points[point]);
                    let n = match &mesh.normals {
                        Some(nv) => {
                            let v = nv.at(point, face, corner).ok_or_else(|| {
                                format!("{path}: normals do not cover face {face}")
                            })?;
                            DVec3::from_slice(v)
                        }
                        None => smooth[point],
                    };
                    let n = (normal_m * n).normalize_or_zero();
                    let st = match &mesh.uvs {
                        Some(uv) => {
                            let v = uv
                                .at(point, face, corner)
                                .ok_or_else(|| format!("{path}: st does not cover face {face}"))?;
                            Vec2::new(v[0] as f32, v[1] as f32)
                        }
                        None => Vec2::ZERO,
                    };
                    let st = uv_t.map_or(st, |t| t.transform_point2(st));
                    let skin = at.skin.map_or(([0; 4], [0; 4]), |s| s[point]);
                    *o = self.vertex(
                        p.as_vec3().to_array(),
                        n.as_vec3().to_array(),
                        [st.x, 1.0 - st.y],
                        skin,
                    );
                }
                if flip {
                    out.swap(1, 2);
                }
                self.groups.entry(material).or_default().extend(out);
            }
        }
        Ok(())
    }
}

/// One model's subdivision cage, appended mesh by mesh, in model space and right-handed.
#[derive(Default)]
pub(super) struct CageBuilder {
    pub cage: CageInput,
    pub skin: Vec<SkinInfluence>,
    /// Scene material per face.
    pub materials: Vec<usize>,
    uvs: HashMap<[u32; 2], u32>,
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
        for &p in &mesh.points {
            c.positions.push(at.transform.transform_point3(p).as_vec3());
        }
        if let Some(s) = at.skin {
            self.skin.extend(
                s.iter()
                    .map(|&(joints, weights)| SkinInfluence { joints, weights }),
            );
        }
        let had_uvs = c.uvs.is_some();
        if mesh.uvs.is_some() && !had_uvs {
            // Earlier meshes without uvs read value 0.
            c.uvs = Some(FaceVarying {
                values: vec![Vec2::ZERO],
                indices: vec![0; c.face_indices.len()],
            });
        }
        let welds = mesh
            .uvs
            .as_ref()
            .map(|uv| uv_values(mesh, uv, path))
            .transpose()?;
        let mut corner = 0;
        for (face, &count) in mesh.counts.iter().enumerate() {
            let (material, uv_t) = (at.face_material)(face);
            self.materials.push(material);
            c.face_counts.push(count as u32);
            let ring: Vec<usize> = if flip {
                std::iter::once(corner)
                    .chain((corner + 1..corner + count).rev())
                    .collect()
            } else {
                (corner..corner + count).collect()
            };
            corner += count;
            for k in ring {
                c.face_indices.push(base + mesh.indices[k] as u32);
                let Some(fv) = c.uvs.as_mut() else { continue };
                let value = match &welds {
                    Some((values, index)) => {
                        let v = values[index[k] as usize];
                        let v = uv_t.map_or(v, |t| t.transform_point2(v));
                        Vec2::new(v.x, 1.0 - v.y)
                    }
                    None => Vec2::ZERO,
                };
                let id = match &welds {
                    // A value stays shared among the corners of one source value and one
                    // uv transform.
                    Some((_, index)) => {
                        let source = [index[k], uv_t.map_or(u32::MAX, |_| material as u32)];
                        *self.uvs.entry(source).or_insert_with(|| {
                            fv.values.push(value);
                            fv.values.len() as u32 - 1
                        })
                    }
                    None => 0,
                };
                fv.indices.push(id);
            }
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
        // Per-mesh uv welding must not join another mesh's values.
        self.uvs.clear();
        Ok(())
    }
}

/// The mesh's uv values and one value index per face corner: indexed and vertex primvars
/// keep their sharing, unindexed face-varying values weld where a point's corners agree.
fn uv_values(mesh: &MeshPrim, uv: &Primvar, path: &str) -> Result<(Vec<Vec2>, Vec<u32>), String> {
    let mut values = Vec::new();
    let mut index = Vec::with_capacity(mesh.indices.len());
    let mut seen: HashMap<(usize, [u64; 2]), u32> = HashMap::new();
    let mut corner = 0;
    for (face, &count) in mesh.counts.iter().enumerate() {
        for k in corner..corner + count {
            let point = mesh.indices[k];
            let v = uv
                .at(point, face, k)
                .ok_or_else(|| format!("{path}: st does not cover face {face}"))?;
            let element = match (uv.interp, &uv.indices) {
                (Interp::FaceVarying, Some(ix)) => ix[k] as usize,
                (Interp::FaceVarying, None) => point,
                (Interp::Vertex, _) => point,
                (Interp::Uniform, _) => usize::MAX - face,
                (Interp::Constant, _) => usize::MAX,
            };
            let key = (element, [v[0].to_bits(), v[1].to_bits()]);
            let id = *seen.entry(key).or_insert_with(|| {
                values.push(Vec2::new(v[0] as f32, v[1] as f32));
                values.len() as u32 - 1
            });
            index.push(id);
        }
        corner += count;
    }
    Ok((values, index))
}

/// Area-weighted point normals from the faces around each point.
fn smooth_normals(mesh: &MeshPrim) -> Vec<DVec3> {
    let mut acc = vec![DVec3::ZERO; mesh.points.len()];
    let mut base = 0;
    for &count in &mesh.counts {
        let ring = &mesh.indices[base..base + count];
        base += count;
        // Newell's normal: twice the area along the face's normal.
        let mut n = DVec3::ZERO;
        for (i, &a) in ring.iter().enumerate() {
            let (p, q) = (mesh.points[a], mesh.points[ring[(i + 1) % ring.len()]]);
            n += DVec3::new(
                (p.y - q.y) * (p.z + q.z),
                (p.z - q.z) * (p.x + q.x),
                (p.x - q.x) * (p.y + q.y),
            );
        }
        for &a in ring {
            acc[a] += n;
        }
    }
    if mesh.left_handed {
        acc.iter_mut().for_each(|n| *n = -*n);
    }
    acc.into_iter().map(DVec3::normalize_or_zero).collect()
}

/// Triangles of a polygon as corner triplets, wound like the polygon: a fan for triangles
/// and convex quads, ear clipping otherwise (a fan where clipping finds no ear).
pub(super) fn triangulate(pts: &[DVec3]) -> Vec<[usize; 3]> {
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
