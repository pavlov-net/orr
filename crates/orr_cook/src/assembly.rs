//! Assemblies: rigid repeats inside one mesh, cooked once as unit models and placed by
//! ordinary instance records.
//!
//! [`split`] cuts a static mesh into connected pieces (triangles joined by shared positions)
//! and buckets the pieces of at least `min_triangles` triangles by their per-section
//! triangle counts. Within a bucket, a copy is fitted to the bucket's first piece from two
//! anchor points (the point farthest from the area-weighted centroid, then the point
//! farthest from that axis), refined by Gauss-Newton over nearest points, and accepted when
//! every triangle lands on one of the copy's within half the grid step with equal section,
//! uvs, normals and tangents. Pieces and their triangles are sorted by position first, so
//! the result does not depend on index order.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

use orr_math::{DMat3, DQuat, DVec3, Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::instances::{Instances, Placement, PrototypeSpec, SectionSpec};
use crate::mesh::{MeshData, PartData, Pivots, SectionData, grid_exponent};
use crate::{Error, Result};

/// Repeated rigid pieces of a mesh cooked once as unit models.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AssemblySettings {
    /// Smallest piece, in triangles, that becomes a unit.
    pub min_triangles: u32,
}

impl Default for AssemblySettings {
    fn default() -> Self {
        Self { min_triangles: 128 }
    }
}

/// A unit copy in its model's space: `rotation * unit + translation`.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct UnitPlacement {
    pub unit: u32,
    pub rotation: [f32; 4],
    pub translation: [f32; 3],
}

/// A mesh cut into its repeated units and the rest.
pub(crate) struct Split {
    /// Per unit: its mesh in unit space and the source section of each of its sections.
    pub units: Vec<(MeshData, Vec<u16>)>,
    pub placements: Vec<UnitPlacement>,
    /// Every source section, without the units' triangles.
    pub remainder: MeshData,
}

/// What instance sources need to expand placements of model `model`; kept in the cookstate
/// with the source's record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Assembly {
    pub model: String,
    pub units: Vec<Unit>,
    pub placements: Vec<UnitPlacement>,
    /// Triangles of the source mesh and of the model's own mesh after the split.
    pub source_triangles: u64,
    pub kept_triangles: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Unit {
    pub model: String,
    pub sections: Vec<u16>,
    pub triangles: u32,
}

/// A triangle: its section and corners, the corner with the smallest position first.
#[derive(Copy, Clone)]
struct Tri {
    section: u16,
    v: [u32; 3],
}

struct Piece {
    /// Sorted by section, then corner positions.
    tris: Vec<Tri>,
    /// Section triangle counts, the bucket key.
    counts: Vec<(u16, u32)>,
    /// Area-weighted centroid.
    center: DVec3,
    area: f64,
    perimeter: f64,
    /// Distinct corner positions, sorted.
    points: Vec<DVec3>,
    radius: f64,
}

fn dvec_cmp(a: DVec3, b: DVec3) -> Ordering {
    (0..3)
        .map(|i| a[i].total_cmp(&b[i]))
        .find(|o| o.is_ne())
        .unwrap_or(Ordering::Equal)
}

fn point(m: &MeshData, v: u32) -> DVec3 {
    DVec3::from(m.positions[v as usize].map(f64::from))
}

/// Positions with `-0.0` as `0.0`, as bits.
fn bits(p: [f32; 3]) -> [u32; 3] {
    p.map(|x| (x + 0.0).to_bits())
}

/// Cuts the repeated pieces out of `m`; `None` when no piece repeats or the mesh is skinned
/// or joined.
pub(crate) fn split(m: &MeshData, s: &AssemblySettings, step_log2: i8) -> Option<Split> {
    if m.skin.is_some() || m.min_exponent.is_some() || m.indices.is_empty() {
        return None;
    }
    let tol = 0.5 * 2f64.powi(grid_exponent(&m.positions, step_log2).into());
    let pieces = pieces(m, s.min_triangles);
    let mut buckets: BTreeMap<&[(u16, u32)], Vec<usize>> = BTreeMap::new();
    for (i, p) in pieces.iter().enumerate() {
        buckets.entry(&p.counts).or_default().push(i);
    }
    // Per group: the canonical piece, then every copy with its rotation from it.
    let mut groups: Vec<Vec<(usize, DQuat)>> = Vec::new();
    for members in buckets.values() {
        let mut free = members.clone();
        while let Some((&first, rest)) = free.split_first() {
            let mut group = vec![(first, DQuat::IDENTITY)];
            let mut left = Vec::new();
            for &other in rest {
                match fit(m, &pieces[first], &pieces[other], tol) {
                    Some(r) => group.push((other, r)),
                    None => left.push(other),
                }
            }
            if group.len() > 1 {
                groups.push(group);
            }
            free = left;
        }
    }
    if groups.is_empty() {
        return None;
    }
    groups.sort_by(|a, b| dvec_cmp(pieces[a[0].0].center, pieces[b[0].0].center));

    let mut taken = vec![false; pieces.len()];
    let mut units = Vec::with_capacity(groups.len());
    let mut placements = Vec::new();
    for (u, group) in groups.iter().enumerate() {
        let canonical = &pieces[group[0].0];
        units.push(unit_mesh(m, canonical));
        for &(p, r) in group {
            taken[p] = true;
            let mut q = r.normalize().as_quat();
            if q.w < 0.0 {
                q = -q;
            }
            placements.push(UnitPlacement {
                unit: u as u32,
                rotation: q.to_array(),
                translation: pieces[p].center.as_vec3().to_array(),
            });
        }
    }
    let mut removed: Vec<Vec<[u32; 3]>> = vec![Vec::new(); m.sections.len()];
    for (p, piece) in pieces.iter().enumerate() {
        if taken[p] {
            for t in &piece.tris {
                let mut v = t.v;
                v.sort_unstable();
                removed[usize::from(t.section)].push(v);
            }
        }
    }
    Some(Split {
        units,
        placements,
        remainder: remainder(m, removed),
    })
}

/// Connected pieces of at least `min` triangles, none touching an animated part, sorted by
/// centroid.
fn pieces(m: &MeshData, min: u32) -> Vec<Piece> {
    let n = m.positions.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn root(parent: &mut [u32], mut v: u32) -> u32 {
        while parent[v as usize] != v {
            parent[v as usize] = parent[parent[v as usize] as usize];
            v = parent[v as usize];
        }
        v
    }
    let join = |parent: &mut Vec<u32>, a: u32, b: u32| {
        let (ra, rb) = (root(parent, a), root(parent, b));
        if ra != rb {
            parent[ra.max(rb) as usize] = ra.min(rb);
        }
    };
    let mut by_pos: HashMap<[u32; 3], u32> = HashMap::with_capacity(n);
    for (v, p) in m.positions.iter().enumerate() {
        let first = *by_pos.entry(bits(*p)).or_insert(v as u32);
        join(&mut parent, first, v as u32);
    }
    for sec in &m.sections {
        for c in m.indices[sec.indices.clone()].chunks_exact(3) {
            join(&mut parent, c[0], c[1]);
            join(&mut parent, c[0], c[2]);
        }
    }
    let animated = |v: u32| m.parts.as_ref().is_some_and(|p| p.values[v as usize] != 0);
    let mut tris_of: HashMap<u32, Vec<Tri>> = HashMap::new();
    let mut skip: HashSet<u32> = HashSet::new();
    for (s, sec) in m.sections.iter().enumerate() {
        for c in m.indices[sec.indices.clone()].chunks_exact(3) {
            let r = root(&mut parent, c[0]);
            if c.iter().any(|&v| animated(v)) {
                skip.insert(r);
            }
            tris_of.entry(r).or_default().push(Tri {
                section: s as u16,
                v: [c[0], c[1], c[2]],
            });
        }
    }
    let mut out: Vec<Piece> = tris_of
        .into_iter()
        .filter(|(r, t)| t.len() >= min as usize && !skip.contains(r))
        .map(|(_, tris)| piece(m, tris))
        .collect();
    out.sort_by(|a, b| {
        dvec_cmp(a.center, b.center).then_with(|| {
            let (ta, tb) = (a.tris[0].v, b.tris[0].v);
            (0..3)
                .map(|k| dvec_cmp(point(m, ta[k]), point(m, tb[k])))
                .find(|o| o.is_ne())
                .unwrap_or(Ordering::Equal)
        })
    });
    out
}

/// A piece from its triangles: corners and triangles in position order, its centroid and
/// its distinct points.
fn piece(m: &MeshData, mut tris: Vec<Tri>) -> Piece {
    let corner_cmp = |a: u32, b: u32| dvec_cmp(point(m, a), point(m, b)).then(a.cmp(&b));
    for t in &mut tris {
        let first = (0..3)
            .min_by(|&i, &j| corner_cmp(t.v[i], t.v[j]))
            .expect("three corners");
        t.v.rotate_left(first);
    }
    tris.sort_by(|a, b| {
        a.section.cmp(&b.section).then_with(|| {
            (0..3)
                .map(|k| corner_cmp(a.v[k], b.v[k]))
                .find(|o| o.is_ne())
                .unwrap_or(Ordering::Equal)
        })
    });
    let mut counts: Vec<(u16, u32)> = Vec::new();
    let (mut weighted, mut area, mut perimeter, mut mean) = (DVec3::ZERO, 0.0, 0.0, DVec3::ZERO);
    let mut points = Vec::with_capacity(tris.len());
    for t in &tris {
        match counts.last_mut() {
            Some((s, c)) if *s == t.section => *c += 1,
            _ => counts.push((t.section, 1)),
        }
        let p = t.v.map(|v| point(m, v));
        let a = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length();
        weighted += a * (p[0] + p[1] + p[2]) / 3.0;
        mean += p[0] + p[1] + p[2];
        area += a;
        perimeter += p[0].distance(p[1]) + p[1].distance(p[2]) + p[2].distance(p[0]);
        points.extend(p);
    }
    let center = if area > 0.0 {
        weighted / area
    } else {
        mean / (3 * tris.len()) as f64
    };
    points.sort_by(|a, b| dvec_cmp(*a, *b));
    points.dedup();
    let radius = points
        .iter()
        .map(|p| p.distance(center))
        .fold(0.0, f64::max);
    Piece {
        tris,
        counts,
        center,
        area,
        perimeter,
        points,
        radius,
    }
}

/// Points of a piece hashed by cell.
struct PointGrid<'a> {
    cell: f64,
    points: &'a [DVec3],
    map: HashMap<[i64; 3], Vec<u32>>,
}

impl<'a> PointGrid<'a> {
    fn new(points: &'a [DVec3], cell: f64) -> Self {
        let mut map: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for (i, p) in points.iter().enumerate() {
            map.entry(cell_of(*p, cell)).or_default().push(i as u32);
        }
        Self { cell, points, map }
    }

    /// The nearest point within one cell of `p`.
    fn nearest(&self, p: DVec3) -> Option<DVec3> {
        let c = cell_of(p, self.cell);
        let mut best: Option<(f64, DVec3)> = None;
        for d in neighbors() {
            let k = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
            for &i in self.map.get(&k).into_iter().flatten() {
                let q = self.points[i as usize];
                let e = q.distance_squared(p);
                if best.is_none_or(|b| e < b.0) {
                    best = Some((e, q));
                }
            }
        }
        best.filter(|b| b.0 <= self.cell * self.cell).map(|b| b.1)
    }
}

fn cell_of(p: DVec3, cell: f64) -> [i64; 3] {
    (p / cell).floor().as_i64vec3().to_array()
}

fn neighbors() -> impl Iterator<Item = [i64; 3]> {
    (0..27).map(|i| [i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1])
}

/// An orthonormal frame from an axis point and a second point off that axis.
fn frame(center: DVec3, a: DVec3, b: DVec3) -> Option<DMat3> {
    let x = (a - center).try_normalize()?;
    let y = (b - center).reject_from_normalized(x).try_normalize()?;
    Some(DMat3::from_cols(x, y, x.cross(y)))
}

/// The rotation taking piece `a` onto piece `b` about their centroids, if `b` is a copy.
fn fit(m: &MeshData, a: &Piece, b: &Piece, tol: f64) -> Option<DQuat> {
    let slack = 2.0 * tol + 1e-6 * a.radius;
    if (a.radius - b.radius).abs() > slack
        || (a.area - b.area).abs() > a.perimeter.max(b.perimeter) * tol + 1e-9
        || a.points.len() != b.points.len()
    {
        return None;
    }
    let off_axis = |p: &Piece, axis: DVec3| {
        let (c, x) = (p.center, (axis - p.center).normalize_or_zero());
        move |q: DVec3| (q - c).reject_from_normalized(x).length()
    };
    // The first point of `a` farthest by `d`.
    let farthest = |d: &dyn Fn(DVec3) -> f64| {
        a.points
            .iter()
            .map(|&q| (q, d(q)))
            .fold((DVec3::ZERO, -1.0), |b, c| if c.1 > b.1 { c } else { b })
    };
    let (a1, da) = farthest(&|q| q.distance(a.center));
    let (a2, ha) = farthest(&off_axis(a, a1));
    if ha < 1e-9 || da < 1e-9 {
        return None;
    }
    let fa = frame(a.center, a1, a2)?;
    let da2 = a2.distance(a.center);
    let reach = 4.0 * tol * (1.0 + a.radius / ha) + 1e-6 * a.radius;
    let grid = PointGrid::new(&b.points, reach);

    let near = |want: f64, d: &dyn Fn(DVec3) -> f64| {
        let mut c: Vec<(f64, DVec3)> = b
            .points
            .iter()
            .map(|&q| ((d(q) - want).abs(), q))
            .filter(|c| c.0 <= slack)
            .collect();
        c.sort_by(|x, y| x.0.total_cmp(&y.0).then(dvec_cmp(x.1, y.1)));
        c.truncate(8);
        c
    };
    for (_, b1) in near(da, &|q| q.distance(b.center)) {
        let h_b = off_axis(b, b1);
        let second = near(ha, &|q| h_b(q));
        for (_, b2) in second {
            if (b2.distance(b.center) - da2).abs() > slack
                || (b2.distance(b1) - a2.distance(a1)).abs() > slack
            {
                continue;
            }
            let Some(fb) = frame(b.center, b1, b2) else {
                continue;
            };
            let mut r = DQuat::from_mat3(&(fb * fa.transpose())).normalize();
            if !refine(a, b, &grid, &mut r) {
                continue;
            }
            if matches(m, a, b, r, tol) {
                return Some(r);
            }
        }
    }
    None
}

/// Gauss-Newton steps on the rotation over nearest points; false when a point of `a` has
/// no point of `b` near it.
fn refine(a: &Piece, b: &Piece, grid: &PointGrid<'_>, r: &mut DQuat) -> bool {
    for _ in 0..3 {
        let (mut lhs, mut rhs) = (DMat3::ZERO, DVec3::ZERO);
        for &p in &a.points {
            let q = *r * (p - a.center);
            let Some(t) = grid.nearest(q + b.center) else {
                return false;
            };
            let d = t - b.center - q;
            lhs += DMat3::from_diagonal(DVec3::splat(q.length_squared()))
                - DMat3::from_cols(q * q.x, q * q.y, q * q.z);
            rhs += q.cross(d);
        }
        if lhs.determinant().abs() < 1e-30 {
            return true;
        }
        let w = lhs.inverse() * rhs;
        *r = (DQuat::from_scaled_axis(w) * *r).normalize();
        if w.length() < 1e-12 {
            break;
        }
    }
    true
}

/// Whether every triangle of `a` under `r` lands on its own triangle of `b`: corners within
/// `tol`, the same section, uvs within 1e-4, normals and tangents within 0.9995.
fn matches(m: &MeshData, a: &Piece, b: &Piece, r: DQuat, tol: f64) -> bool {
    let cell = 2.0 * tol;
    let pos = |v: u32| point(m, v);
    let centroid = |p: [DVec3; 3]| (p[0] + p[1] + p[2]) / 3.0;
    let mut map: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for (i, t) in b.tris.iter().enumerate() {
        map.entry(cell_of(centroid(t.v.map(pos)), cell))
            .or_default()
            .push(i as u32);
    }
    let rf = r.as_quat();
    let dir_ok = |x: Vec3, y: Vec3| match (x.try_normalize(), y.try_normalize()) {
        (Some(x), Some(y)) => x.dot(y) >= 0.9995,
        (None, None) => true,
        _ => false,
    };
    let same_vertex = |va: u32, vb: u32| {
        let (va, vb) = (va as usize, vb as usize);
        let uv = (0..2).all(|k| (m.uvs[va][k] - m.uvs[vb][k]).abs() <= 1e-4);
        let normal = dir_ok(rf * Vec3::from(m.normals[va]), Vec3::from(m.normals[vb]));
        let tangent = m.tangents.as_ref().is_none_or(|t| {
            let (ta, tb) = (t[va], t[vb]);
            (ta[3] < 0.0) == (tb[3] < 0.0)
                && dir_ok(
                    rf * Vec3::new(ta[0], ta[1], ta[2]),
                    Vec3::new(tb[0], tb[1], tb[2]),
                )
        });
        uv && normal && tangent
    };
    let mut used = vec![false; b.tris.len()];
    a.tris.iter().all(|ta| {
        let p = ta.v.map(|v| r * (pos(v) - a.center) + b.center);
        let c = cell_of(centroid(p), cell);
        for d in neighbors() {
            let k = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
            for &j in map.get(&k).into_iter().flatten() {
                let tb = &b.tris[j as usize];
                if used[j as usize] || tb.section != ta.section {
                    continue;
                }
                let on = (0..3).find(|&s| {
                    (0..3).all(|i| {
                        let vb = tb.v[(i + s) % 3];
                        p[i].distance(pos(vb)) <= tol && same_vertex(ta.v[i], vb)
                    })
                });
                if on.is_some() {
                    used[j as usize] = true;
                    return true;
                }
            }
        }
        false
    })
}

/// The canonical piece as a mesh about its centroid, with the source section of each of
/// its sections.
fn unit_mesh(m: &MeshData, p: &Piece) -> (MeshData, Vec<u16>) {
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut out = MeshData {
        positions: Vec::new(),
        normals: Vec::new(),
        tangents: m.tangents.as_ref().map(|_| Vec::new()),
        uvs: Vec::new(),
        indices: Vec::with_capacity(3 * p.tris.len()),
        sections: Vec::new(),
        skin: None,
        parts: None,
        min_exponent: None,
    };
    let mut sources = Vec::new();
    for t in &p.tris {
        if sources.last() != Some(&t.section) {
            sources.push(t.section);
            let at = out.indices.len();
            out.sections.push(SectionData {
                indices: at..at,
                ..m.sections[usize::from(t.section)].clone()
            });
        }
        for &v in &t.v {
            let n = *remap.entry(v).or_insert_with(|| {
                let i = v as usize;
                let q = point(m, v) - p.center;
                out.positions.push(q.as_vec3().to_array());
                out.normals.push(m.normals[i]);
                out.uvs.push(m.uvs[i]);
                if let (Some(dst), Some(src)) = (&mut out.tangents, &m.tangents) {
                    dst.push(src[i]);
                }
                (out.positions.len() - 1) as u32
            });
            out.indices.push(n);
        }
        out.sections.last_mut().expect("a section").indices.end = out.indices.len();
    }
    (out, sources)
}

/// `m` without the `removed` triangles (corners sorted, per section), its vertices
/// compacted.
fn remainder(m: &MeshData, mut removed: Vec<Vec<[u32; 3]>>) -> MeshData {
    let mut remap = vec![u32::MAX; m.positions.len()];
    let mut order = Vec::new();
    let mut indices = Vec::new();
    let mut sections = Vec::with_capacity(m.sections.len());
    for (s, sec) in m.sections.iter().enumerate() {
        removed[s].sort_unstable();
        let start = indices.len();
        for c in m.indices[sec.indices.clone()].chunks_exact(3) {
            let mut k = [c[0], c[1], c[2]];
            k.sort_unstable();
            if removed[s].binary_search(&k).is_ok() {
                continue;
            }
            for &v in c {
                if remap[v as usize] == u32::MAX {
                    remap[v as usize] = order.len() as u32;
                    order.push(v as usize);
                }
                indices.push(remap[v as usize]);
            }
        }
        sections.push(SectionData {
            indices: start..indices.len(),
            ..sec.clone()
        });
    }
    let pick = |a: &[[f32; 3]]| order.iter().map(|&v| a[v]).collect::<Vec<_>>();
    MeshData {
        positions: pick(&m.positions),
        normals: pick(&m.normals),
        tangents: m
            .tangents
            .as_ref()
            .map(|t| order.iter().map(|&v| t[v]).collect()),
        uvs: order.iter().map(|&v| m.uvs[v]).collect(),
        indices,
        sections,
        skin: None,
        parts: m.parts.as_ref().map(|p| PartData {
            values: order.iter().map(|&v| p.values[v]).collect(),
            count: p.count,
            by_section: p.by_section,
            pivots: match &p.pivots {
                Pivots::PerValue(v) => Pivots::PerValue(v.clone()),
                Pivots::PerVertex(v) => Pivots::PerVertex(order.iter().map(|&i| v[i]).collect()),
                Pivots::Origin => Pivots::Origin,
            },
            joints: p.joints.clone(),
        }),
        min_exponent: None,
    }
}

/// A placement of a unit copy from a placement of its model; `None` when a non-uniform
/// scale meets a unit rotation that does not map axes to axes.
pub(crate) fn compose(p: &Placement, u: &UnitPlacement) -> Option<Placement> {
    let r = Quat::from_array(u.rotation);
    let s = p.scale;
    let uniform = s.max_element() - s.min_element() <= 1e-6 * s.max_element();
    let scale = if uniform {
        s
    } else {
        // S R = R S' when R e_j = +-e_i: S'_j = S_i.
        let m = Mat3::from_quat(r);
        let mut out = [0.0; 3];
        for (j, o) in out.iter_mut().enumerate() {
            let col = m.col(j).abs();
            let i = (0..3).find(|&i| col[i] > 1.0 - 1e-4)?;
            *o = s[i];
        }
        Vec3::from(out)
    };
    let t = DVec3::from(u.translation.map(f64::from));
    Some(Placement {
        position: p.position + p.rotation.as_dquat() * (s.as_dvec3() * t),
        rotation: (p.rotation * r).normalize(),
        scale,
        prototype: p.prototype,
    })
}

/// `src` with every placement of a model with an assembly also placing its units, and the
/// model's own placement dropped when its mesh kept no triangle; `None` when no prototype
/// names such a model.
pub(crate) fn expand<'a>(
    name: &str,
    src: &Instances,
    of: impl Fn(&str) -> Option<&'a Assembly>,
) -> Result<Option<Instances>> {
    let found: Vec<Option<&Assembly>> = src.prototypes.iter().map(|p| of(&p.model)).collect();
    if found.iter().all(Option::is_none) {
        return Ok(None);
    }
    let fail = |message: String| Error::Instances {
        asset: name.into(),
        message,
    };
    if src.joined {
        let model = found
            .iter()
            .flatten()
            .next()
            .expect("an assembly")
            .model
            .clone();
        return Err(fail(format!(
            "joined placements cannot place {model}, whose units rotate freely"
        )));
    }
    // Per source prototype: its own index, if it stays, and the index of its first unit.
    let mut out = Instances {
        user_stride: src.user_stride,
        ..Instances::default()
    };
    let mut map = Vec::with_capacity(src.prototypes.len());
    for (p, a) in src.prototypes.iter().zip(&found) {
        let own = match a {
            Some(a) if a.kept_triangles == 0 => None,
            _ => {
                out.prototypes.push(p.clone());
                Some(out.prototypes.len() as u16 - 1)
            }
        };
        let first = out.prototypes.len() as u16;
        for unit in a.iter().flat_map(|a| &a.units) {
            let sections = if p.sections.is_empty() {
                Vec::new()
            } else {
                unit.sections
                    .iter()
                    .map(|&s| p.sections.get(usize::from(s)).cloned().unwrap_or_default())
                    .collect::<Vec<SectionSpec>>()
            };
            out.prototypes.push(PrototypeSpec {
                model: unit.model.clone(),
                flags: p.flags,
                pose: 0,
                sections,
            });
        }
        map.push((own, first));
    }
    let stride = usize::from(src.user_stride);
    for (i, rec) in src.records.iter().enumerate() {
        let user = src.user.get(i * stride..(i + 1) * stride).unwrap_or(&[]);
        let (own, first) = map[usize::from(rec.prototype)];
        if let Some(own) = own {
            out.records.push(Placement {
                prototype: own,
                ..rec.clone()
            });
            out.user.extend_from_slice(user);
        }
        let Some(a) = found[usize::from(rec.prototype)] else {
            continue;
        };
        for u in &a.placements {
            let mut c = compose(rec, u).ok_or_else(|| {
                fail(format!(
                    "placement {i} scales {} unevenly by {}, and a unit copy is not rotated about the axes",
                    a.model, rec.scale
                ))
            })?;
            c.prototype = first + u.unit as u16;
            out.records.push(c);
            out.user.extend_from_slice(user);
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MapSpec;
    use orr_math::{EulerRot, IVec3};

    struct Fixture {
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        sections: Vec<Vec<[u32; 3]>>,
        unit_tris: usize,
    }

    impl Fixture {
        fn mesh(&self, sections: &[Vec<[u32; 3]>]) -> MeshData {
            let mut indices = Vec::new();
            let mut out = Vec::new();
            for tris in sections {
                let first = indices.len();
                indices.extend(tris.iter().flatten());
                out.push(SectionData {
                    indices: first..indices.len(),
                    flags: 0,
                    subsurface: false,
                    maps: MapSpec::default(),
                });
            }
            MeshData {
                positions: self.positions.clone(),
                normals: self.normals.clone(),
                tangents: None,
                uvs: self.uvs.clone(),
                indices,
                sections: out,
                skin: None,
                parts: None,
                min_exponent: None,
            }
        }
    }

    /// `copies` rotated copies of a bumpy, partial disc in section 0 over a sheet in
    /// section 1.
    fn fixture(copies: usize) -> Fixture {
        let n = 120;
        let mut disc = vec![([0.0, 0.0, 0.02], [0.5, 0.5])];
        for i in 0..n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            let r = 0.2 + 0.03 * (3.0 * a).sin();
            disc.push((
                [0.1 * a.cos(), 0.1 * a.sin(), 0.01 * (5.0 * a).cos()],
                [a, 0.3],
            ));
            disc.push(([r * a.cos(), r * a.sin(), 0.0], [a, 0.9]));
        }
        let mut tris = Vec::new();
        for i in 0..n * 2 / 3 {
            let (a, b) = (1 + 2 * i, 1 + 2 * (i + 1));
            tris.extend([[0, a, b], [a, a + 1, b + 1], [a, b + 1, b]]);
        }
        let mut f = Fixture {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            sections: vec![Vec::new(), Vec::new()],
            unit_tris: tris.len(),
        };
        for c in 0..copies {
            let c = c as f32;
            let q = Quat::from_euler(EulerRot::XYZ, 0.37 * c, 1.1 * c, -0.23 * c);
            let t = Vec3::new(c.sin() * 3.0, 0.4 * c, (0.7 * c).cos() * 2.0);
            let base = f.positions.len() as u32;
            for (p, uv) in &disc {
                f.positions.push((q * Vec3::from(*p) + t).to_array());
                f.normals.push((q * Vec3::Z).to_array());
                f.uvs.push(*uv);
            }
            f.sections[0].extend(tris.iter().map(|t: &[u32; 3]| t.map(|v| v + base)));
        }
        let base = f.positions.len() as u32;
        for j in 0..9 {
            for i in 0..9 {
                let h = 0.01 * (i * j) as f32;
                f.positions.push([i as f32 * 0.5, -1.0 + h, j as f32 * 0.5]);
                f.normals.push([0.0, 1.0, 0.0]);
                f.uvs.push([i as f32, j as f32]);
            }
        }
        for j in 0..8 {
            for i in 0..8 {
                let c = base + j * 9 + i;
                f.sections[1].extend([[c, c + 9, c + 1], [c + 1, c + 9, c + 10]]);
            }
        }
        f
    }

    type TriSet = Vec<(u16, [Vec3; 3])>;

    fn tri_set(m: &MeshData, sections: &[u16], t: impl Fn(Vec3) -> Vec3) -> TriSet {
        let mut out = Vec::new();
        for (s, sec) in m.sections.iter().enumerate() {
            for c in m.indices[sec.indices.clone()].chunks_exact(3) {
                let p = [0, 1, 2].map(|k| t(Vec3::from(m.positions[c[k] as usize])));
                out.push((sections[s], p));
            }
        }
        out
    }

    fn expanded(split: &Split) -> TriSet {
        let mut out = tri_set(&split.remainder, &[0, 1], |p| p);
        for u in &split.placements {
            let (mesh, sections) = &split.units[u.unit as usize];
            let r = Quat::from_array(u.rotation);
            let t = Vec3::from(u.translation);
            out.extend(tri_set(mesh, sections, |p| r * p + t));
        }
        out
    }

    /// Whether `a` and `b` hold the same triangles, corners within `tol` in some rotation.
    fn same_set(a: &TriSet, b: &TriSet, tol: f32) -> bool {
        let key = |t: &[Vec3; 3]| ((t[0] + t[1] + t[2]) / 3.0 / 0.01).floor().as_ivec3();
        let mut grid: HashMap<IVec3, Vec<usize>> = HashMap::new();
        for (j, (_, t)) in b.iter().enumerate() {
            grid.entry(key(t)).or_default().push(j);
        }
        let mut used = vec![false; b.len()];
        a.len() == b.len()
            && a.iter().all(|(s, ta)| {
                let c = key(ta);
                let mut hit = None;
                for x in -1..=1 {
                    for y in -1..=1 {
                        for z in -1..=1 {
                            for &j in grid.get(&(c + IVec3::new(x, y, z))).into_iter().flatten() {
                                let (sb, tb) = &b[j];
                                let on = (0..3).any(|k| {
                                    (0..3).all(|i| ta[i].distance(tb[(i + k) % 3]) <= tol)
                                });
                                if !used[j] && s == sb && on {
                                    hit = Some(j);
                                }
                            }
                        }
                    }
                }
                hit.map(|j| used[j] = true).is_some()
            })
    }

    #[test]
    fn forty_copies_become_one_unit() {
        let f = fixture(40);
        assert!(f.unit_tris >= 200, "{}", f.unit_tris);
        let m = f.mesh(&f.sections);
        let split = split(&m, &AssemblySettings::default(), -11).expect("units");
        assert_eq!(split.units.len(), 1);
        assert_eq!(split.placements.len(), 40);
        assert_eq!(split.units[0].0.indices.len() / 3, f.unit_tris);
        assert_eq!(split.units[0].1, [0]);
        assert!(split.remainder.sections[0].indices.is_empty());
        let step = 2f32.powi(grid_exponent(&m.positions, -11).into());
        let source = tri_set(&m, &[0, 1], |p| p);
        assert!(same_set(&source, &expanded(&split), 0.5 * step));
    }

    #[test]
    fn shuffled_indices_find_the_same_units() {
        let f = fixture(40);
        let s = AssemblySettings::default();
        let a = split(&f.mesh(&f.sections), &s, -11).expect("units");
        let mut sections = f.sections.clone();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for tris in &mut sections {
            for i in (1..tris.len()).rev() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                tris.swap(i, (state % (i as u64 + 1)) as usize);
                tris[i].rotate_left((state >> 32) as usize % 3);
            }
        }
        let b = split(&f.mesh(&sections), &s, -11).expect("units");
        assert_eq!(a.placements, b.placements);
        assert_eq!(a.units.len(), b.units.len());
        assert_eq!(a.units[0].0.positions, b.units[0].0.positions);
        assert_eq!(a.units[0].0.indices, b.units[0].0.indices);
    }

    #[test]
    fn small_and_single_pieces_stay() {
        let f = fixture(1);
        assert!(split(&f.mesh(&f.sections), &AssemblySettings::default(), -11).is_none());
        let f = fixture(3);
        let s = AssemblySettings {
            min_triangles: f.unit_tris as u32 + 1,
        };
        assert!(split(&f.mesh(&f.sections), &s, -11).is_none());
    }

    #[test]
    fn uneven_scale_keeps_axis_rotations_only() {
        let p = Placement {
            position: DVec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_y(0.5),
            scale: Vec3::new(1.0, 2.0, 3.0),
            prototype: 0,
        };
        let quarter = UnitPlacement {
            unit: 0,
            rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).to_array(),
            translation: [1.0, 1.0, 0.0],
        };
        let c = compose(&p, &quarter).expect("axis rotation");
        assert_eq!(c.scale, Vec3::new(2.0, 1.0, 3.0));
        let x = Vec3::new(0.3, -0.2, 0.7);
        let unit = Quat::from_array(quarter.rotation) * x + Vec3::from(quarter.translation);
        let want = p.position.as_vec3() + p.rotation * (p.scale * unit);
        let got = c.position.as_vec3() + c.rotation * (c.scale * x);
        assert!(want.distance(got) < 1e-5, "{want} {got}");
        let tilted = UnitPlacement {
            rotation: Quat::from_rotation_z(0.3).to_array(),
            ..quarter
        };
        assert!(compose(&p, &tilted).is_none());
    }
}
