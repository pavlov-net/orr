//! Static displacement: the displacement texture baked into a section's geometry.
//!
//! Triangles split at their longest edge until no edge is over `max_edge`; a midpoint is
//! shared by every triangle on its edge, and the two sides of a uv seam compute it from the
//! same positions, so the split mesh stays watertight. Every vertex at a position moves
//! along the mean normal of the baked vertices there, by their mean displacement, so seams
//! and borders with other sections move alike. Normals become the area-weighted face
//! normals at each position. The baked sections are then simplified to `target_error`.

use std::collections::HashMap;

use glam::{DVec3, Vec3};

use super::lod::{self, Parts, Welded};
use super::{Grid, SectionMaps, StaticDisplacement};

/// Splits and displaces every section with a readable displacement texture; returns which
/// sections were baked.
pub(crate) fn bake(
    w: &mut Welded,
    parts: &mut Parts,
    maps: &[SectionMaps],
    d: &StaticDisplacement,
) -> Vec<bool> {
    let baked: Vec<bool> = maps.iter().map(SectionMaps::bakes).collect();
    if !baked.iter().any(|&b| b) {
        return baked;
    }
    split(w, parts, &baked, d.max_edge);

    // Mean displacement and normal per position over the baked vertices.
    let mut at: HashMap<[u32; 3], (f64, DVec3, u32)> = HashMap::new();
    for (s, r) in w.sections.iter().enumerate() {
        let Some(disp) = maps[s].displacement.as_ref().filter(|_| baked[s]) else {
            continue;
        };
        let mip0 = &disp.mips[0];
        for &v in &w.indices[r.clone()] {
            let v = v as usize;
            let e = at
                .entry(w.positions[v].map(f32::to_bits))
                .or_insert((0.0, DVec3::ZERO, 0));
            e.0 += f64::from(disp.scale * mip0.sample(w.uvs[v]) + disp.offset);
            e.1 += Vec3::from(w.normals[v]).normalize_or_zero().as_dvec3();
            e.2 += 1;
        }
    }
    for p in &mut w.positions {
        if let Some(&(sum, n, count)) = at.get(&p.map(f32::to_bits)) {
            let offset = n.normalize_or_zero() * (sum / f64::from(count));
            *p = (Vec3::from(*p).as_dvec3() + offset).as_vec3().to_array();
        }
    }
    smooth_normals(w, &baked);
    baked
}

/// Longest-edge bisection of the baked sections' triangles until no edge is over
/// `max_edge`; triangle parts follow their source triangle.
fn split(w: &mut Welded, parts: &mut Parts, baked: &[bool], max_edge: f32) {
    let max2 = f64::from(max_edge) * f64::from(max_edge);
    let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
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
    let (ua, ub) = (w.uvs[a], w.uvs[b]);
    w.uvs.push([mid(ua[0], ub[0]), mid(ua[1], ub[1])]);
    if let Some(t) = &mut w.tangents {
        let (ta, tb) = (t[a], t[b]);
        let v = Vec3::new(ta[0] + tb[0], ta[1] + tb[1], ta[2] + tb[2]).normalize_or(Vec3::X);
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
    let mut sums: HashMap<[u32; 3], DVec3> = HashMap::new();
    let p = |w: &Welded, v: u32| Vec3::from(w.positions[v as usize]).as_dvec3();
    for (s, r) in w.sections.iter().enumerate() {
        if !baked[s] {
            continue;
        }
        for t in w.indices[r.clone()].chunks_exact(3) {
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

/// Simplifies each (part, section) piece of the baked sections to `target` metres, with
/// grid positions shared with any other piece locked and uv seams kept.
pub(crate) fn simplify(
    w: &mut Welded,
    parts: &mut Parts,
    grid: &Grid,
    baked: &[bool],
    target: f32,
) {
    if target <= 0.0 {
        return;
    }
    let mut pieces = lod::pieces(w, parts);
    let cell = |v: &u32| grid.cells[*v as usize];
    let shared = lod::shared_cells(pieces.iter().map(|p| p.indices.iter().map(cell)));
    let seams = lod::seam_cells(w, w.indices.iter().map(|v| (*v, cell(v))));
    for p in &mut pieces {
        if baked[usize::from(p.section)] {
            p.indices = lod::simplify(&p.indices, w, grid, &shared, &seams, target, false).0;
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
