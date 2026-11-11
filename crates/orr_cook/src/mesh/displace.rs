//! Static displacement: the displacement texture baked into a section's geometry.
//!
//! Triangles split at their longest edge until no edge is over `max_edge`; a midpoint is
//! shared by every triangle on its edge, and the two sides of a uv seam compute it from the
//! same positions, so the split mesh stays watertight. Every vertex at a position moves
//! along the mean normal of the baked vertices there, by their mean displacement, so seams
//! and borders with other sections move alike. [`MeshData::moves`](super::MeshData::moves) takes
//! that normal to the vertices' space where it is held. A height across the surface moves it
//! along the direction its uvs give there. Normals become the area-weighted face
//! normals at each position. The baked sections are then simplified to `target_error`.

use orr_math::{DVec3, Vec3};
use rustc_hash::FxHashMap;

use super::lod::{self, Parts, Welded};
use super::{Grid, SectionMaps, StaticDisplacement};

/// Splits and displaces every section with a readable displacement texture; returns which
/// sections were baked.
pub(super) fn bake(
    w: &mut Welded,
    parts: &mut Parts,
    maps: &[SectionMaps],
    d: StaticDisplacement,
) -> Vec<bool> {
    let _z = crate::profile::zone!("mesh.displace");
    let baked: Vec<bool> = maps.iter().map(SectionMaps::bakes).collect();
    if !baked.iter().any(|&b| b) {
        return baked;
    }
    split(w, parts, &baked, d.max_edge);

    // Mean displacement, and summed normals and moves of a height of 1, per position over the
    // baked vertices.
    let mut at: FxHashMap<[u32; 3], (f64, DVec3, DVec3, u32, DVec3)> = FxHashMap::default();
    for (s, r) in w.sections.iter().enumerate() {
        let Some(disp) = maps[s].displacement.as_ref().filter(|_| baked[s]) else {
            continue;
        };
        let heights = disp.vertex_heights(&w.uvs[disp.uv_set], w.section_triangles(s));
        let [along_u, along_v] = maps[s].tangent_displacement.each_ref().map(|d| {
            d.as_ref()
                .map(|d| d.vertex_heights(&w.uvs[d.uv_set], w.section_triangles(s)))
        });
        let frames = (along_u.is_some() || along_v.is_some())
            .then(|| uv_frames(w, &w.indices[r.clone()], disp.uv_set));
        for &v in &w.indices[r.clone()] {
            let height = heights[&v];
            let frame = frames.as_ref().and_then(|f| f.get(&v)).copied();
            let [u, t] = [&along_u, &along_v].map(|h| h.as_ref().map_or(0.0, |h| h[&v]));
            let v = v as usize;
            let e = at.entry(w.positions[v].map(f32::to_bits)).or_insert((
                0.0,
                DVec3::ZERO,
                DVec3::ZERO,
                0,
                DVec3::ZERO,
            ));
            if let Some([tangent, bitangent]) = frame {
                e.4 += tangent * f64::from(u) + bitangent * f64::from(t);
            }
            e.0 += f64::from(height);
            let unit = Vec3::from(w.normals[v]).normalize_or_zero().to_array();
            let [normal, moved] = w.moves.as_ref().map_or([unit; 2], |m| m[v]);
            e.1 += Vec3::from(normal).as_dvec3();
            e.2 += Vec3::from(moved).as_dvec3();
            e.3 += 1;
        }
    }
    for p in &mut w.positions {
        if let Some(&(sum, n, moved, count, across)) = at.get(&p.map(f32::to_bits)) {
            // A move is linear in its normal, so the unit mean normal's is the sum's over its size.
            let unit = n.length_recip();
            let along = if unit.is_finite() {
                moved * unit
            } else {
                DVec3::ZERO
            };
            let offset = (along * sum + across) / f64::from(count);
            *p = (Vec3::from(*p).as_dvec3() + offset).as_vec3().to_array();
        }
    }
    smooth_normals(w, &baked);
    baked
}

/// Per vertex of triangle list `indices`, unit directions in the surface where u grows and where
/// st's t grows, which is against the cook's v.
fn uv_frames(w: &Welded, indices: &[u32], set: usize) -> FxHashMap<u32, [DVec3; 2]> {
    let mut sums: FxHashMap<u32, [DVec3; 2]> = FxHashMap::default();
    for t in indices.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|v| v as usize);
        let point = |i: usize| Vec3::from(w.positions[i]).as_dvec3();
        let uv = |i: usize| w.uvs[set][i].map(f64::from);
        let (e1, e2) = (point(b) - point(a), point(c) - point(a));
        let ([ua, va], [ub, vb], [uc, vc]) = (uv(a), uv(b), uv(c));
        let (u1, v1, u2, v2) = (ub - ua, vb - va, uc - ua, vc - va);
        let r = u1 * v2 - u2 * v1;
        if r == 0.0 {
            continue;
        }
        let along = ((e1 * v2 - e2 * v1) / r).normalize_or_zero();
        let across = ((e1 * u2 - e2 * u1) / r).normalize_or_zero();
        for &v in t {
            let sum = sums.entry(v).or_default();
            sum[0] += along;
            sum[1] += across;
        }
    }
    for (&v, frame) in &mut sums {
        let n = Vec3::from(w.normals[v as usize])
            .normalize_or_zero()
            .as_dvec3();
        *frame = frame.map(|d| (d - n * n.dot(d)).normalize_or_zero());
    }
    sums
}

/// Longest-edge bisection of the baked sections' triangles until no edge is over
/// `max_edge`; triangle parts follow their source triangle.
fn split(w: &mut Welded, parts: &mut Parts, baked: &[bool], max_edge: f32) {
    let max2 = f64::from(max_edge) * f64::from(max_edge);
    let mut midpoints: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut indices = Vec::with_capacity(w.indices.len());
    let mut of_triangle = Vec::with_capacity(parts.of_triangle.len());
    let mut sections = Vec::with_capacity(w.sections.len());
    for (s, r) in w.sections.clone().into_iter().enumerate() {
        let start = indices.len();
        for t in r.start / 3..r.end / 3 {
            let tri = [0, 1, 2].map(|k| w.indices[3 * t + k]);
            let part = parts.of_triangle[t];
            if !baked[s] {
                indices.extend_from_slice(&tri);
                of_triangle.push(part);
                continue;
            }
            let mut stack = vec![tri];
            while let Some(t) = stack.pop() {
                let len2 = |k: usize| edge2(w, t[k], t[(k + 1) % 3]);
                let (k, l) = (0..3)
                    .map(|k| (k, len2(k)))
                    .fold((0, -1.0), |a, b| if b.1 > a.1 { b } else { a });
                if l <= max2 {
                    indices.extend_from_slice(&t);
                    of_triangle.push(part);
                    continue;
                }
                let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                let m = *midpoints
                    .entry((a.min(b), a.max(b)))
                    .or_insert_with(|| midpoint(w, a.min(b), a.max(b)));
                stack.push([a, m, c]);
                stack.push([m, b, c]);
            }
        }
        sections.push(start..indices.len());
    }
    w.indices = indices;
    w.sections = sections;
    parts.of_triangle = of_triangle;
}

fn edge2(w: &Welded, a: u32, b: u32) -> f64 {
    let p = |v: u32| Vec3::from(w.positions[v as usize]).as_dvec3();
    p(a).distance_squared(p(b))
}

/// A new vertex halfway along `a`-`b`: its position from the two positions in a fixed
/// order, so both sides of a uv seam get the same bits.
fn midpoint(w: &mut Welded, a: u32, b: u32) -> u32 {
    let (a, b) = (a as usize, b as usize);
    let (pa, pb) = (w.positions[a], w.positions[b]);
    let (lo, hi) = if pa.map(f32::to_bits) <= pb.map(f32::to_bits) {
        (pa, pb)
    } else {
        (pb, pa)
    };
    let mid = |x: f32, y: f32| ((f64::from(x) + f64::from(y)) * 0.5) as f32;
    w.positions.push([0, 1, 2].map(|i| mid(lo[i], hi[i])));
    let n = Vec3::from(w.normals[a]) + Vec3::from(w.normals[b]);
    w.normals
        .push(n.normalize_or(Vec3::from(w.normals[a])).to_array());
    for u in &mut w.uvs {
        let (ua, ub) = (u[a], u[b]);
        u.push([mid(ua[0], ub[0]), mid(ua[1], ub[1])]);
    }
    if let Some(c) = &mut w.colors {
        let (ca, cb) = (c[a], c[b]);
        c.push(std::array::from_fn(|i| mid(ca[i], cb[i])));
    }
    if let Some(c) = &mut w.moves {
        // A move is linear in its normal, and the midpoint's normal is the ends' unit sum.
        let [na, ma] = c[a].map(Vec3::from);
        let [nb, mb] = c[b].map(Vec3::from);
        let unit = (na + nb).length_recip();
        c.push(match unit.is_finite() {
            true => [(na + nb) * unit, (ma + mb) * unit].map(<[f32; 3]>::from),
            false => c[a],
        });
    }
    if let Some(t) = &mut w.tangents {
        // A placed mesh's tangents are not unit, and the longer one would pull the midpoint.
        let (ta, tb) = (t[a], t[b]);
        let [ua, ub] = [ta, tb].map(|t| Vec3::from_slice(&t).normalize_or_zero());
        let v = (ua + ub).normalize_or(ua);
        t.push([v.x, v.y, v.z, ta[3]]);
    }
    if let Some(k) = &mut w.skin {
        k.joints.push(k.joints[a]);
        k.weights.push(k.weights[a]);
    }
    if let Some(v) = &mut w.values {
        v.push(v[a]);
    }
    if let Some(p) = &mut w.pivots {
        p.push(p[a]);
    }
    w.positions.len() as u32 - 1
}

/// Normals of the baked sections' vertices: the area-weighted face normals at their
/// position.
fn smooth_normals(w: &mut Welded, baked: &[bool]) {
    let mut sums: FxHashMap<[u32; 3], DVec3> = FxHashMap::default();
    let p = |w: &Welded, v: u32| Vec3::from(w.positions[v as usize]).as_dvec3();
    for (s, r) in w.sections.iter().enumerate() {
        if !baked[s] {
            continue;
        }
        for t in w.indices[r.clone()].as_chunks::<3>().0 {
            let face = (p(w, t[1]) - p(w, t[0])).cross(p(w, t[2]) - p(w, t[0]));
            for &v in t {
                *sums
                    .entry(w.positions[v as usize].map(f32::to_bits))
                    .or_default() += face;
            }
        }
    }
    for (s, r) in w.sections.clone().into_iter().enumerate() {
        if !baked[s] {
            continue;
        }
        for &v in &w.indices[r] {
            let v = v as usize;
            let n = sums[&w.positions[v].map(f32::to_bits)];
            if n != DVec3::ZERO {
                w.normals[v] = n.normalize().as_vec3().to_array();
            }
        }
    }
}

/// Simplifies each (part, section) piece of the baked sections to `target` meters, with
/// grid positions shared with any other piece locked, uv seams kept, and the open borders of
/// sections in `keep_borders` kept ([`lod::simplify`]).
pub(super) fn simplify(
    w: &mut Welded,
    parts: &mut Parts,
    grid: &mut Grid,
    baked: &[bool],
    keep_borders: &[bool],
    target: f32,
) {
    let _z = crate::profile::zone!("mesh.displace_simplify");
    if target <= 0.0 {
        return;
    }
    let mut pieces = lod::pieces(w, parts);
    let (n, id) = (grid.reps.len(), |v: &u32| grid.reps[*v as usize]);
    let shared = lod::shared(n, pieces.iter().map(|p| p.indices.iter().map(id)));
    let seams = lod::seams(w, n, w.indices.iter().map(|v| (*v, id(v))));
    for p in &mut pieces {
        if baked[usize::from(p.section)] {
            let how = lod::Simplify {
                target,
                sloppy: false,
                keep_borders: keep_borders[usize::from(p.section)],
            };
            p.indices = lod::simplify(&p.indices, w, grid, &shared, &seams, how).0;
        }
    }
    let mut indices = Vec::with_capacity(w.indices.len());
    let mut of_triangle = Vec::new();
    let mut sections = Vec::with_capacity(w.sections.len());
    for s in 0..w.sections.len() {
        let start = indices.len();
        for p in pieces.iter().filter(|p| usize::from(p.section) == s) {
            indices.extend_from_slice(&p.indices);
            of_triangle.extend(std::iter::repeat_n(p.part, p.indices.len() / 3));
        }
        sections.push(start..indices.len());
    }
    w.indices = indices;
    w.sections = sections;
    parts.of_triangle = of_triangle;
}
