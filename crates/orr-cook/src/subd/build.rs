//! Patches into pages.

use std::collections::{HashMap, HashSet};

use orr_math::Vec3;
use orr_pack::material::Material;
use orr_pack::mesh::Section;
use orr_pack::subd::{
    Face, PAGE_MAX, Page, PageHeader, PageInfo, Patch, PatchParam, SubdHeader, SubdInfo,
};
use orr_pack::{Aabb, Span};

use super::osd::Refined;
use super::{Boundary, CageData, Cooked, SubdSettings};

/// Ptex faces of a cage face with `corners` corners: one for a quad, one per corner
/// otherwise.
fn ptex_count(corners: u32) -> u32 {
    if corners == 4 { 1 } else { corners }
}

/// Far's packed patch param as the cooked one.
fn param(kind: u8, p: [u32; 2]) -> (u32, PatchParam) {
    let bits = |v: u32, width: u32, offset: u32| (v >> offset) & ((1 << width) - 1);
    let face = bits(p[0], 28, 0);
    let param = PatchParam {
        u: bits(p[1], 10, 22) as u16,
        v: bits(p[1], 10, 12) as u16,
        depth: bits(p[1], 4, 0) as u8,
        boundary: bits(p[1], 5, 7) as u8,
        transition: bits(p[0], 4, 28) as u8,
        kind,
        flags: if bits(p[1], 1, 4) != 0 {
            PatchParam::NON_QUAD
        } else {
            0
        },
        _pad: [0; 3],
    };
    (face, param)
}

pub(super) fn build(
    data: &CageData,
    materials: &[Material],
    settings: &SubdSettings,
    refined: &Refined,
) -> Result<Cooked, String> {
    let cage = &data.cage;
    let points: Vec<[f32; 3]> = cage.positions.iter().map(Vec3::to_array).collect();
    let rest = refined.stencils.apply(&points);
    let section_of = |f: u32| cage.face_sections.get(f as usize).copied().unwrap_or(0);

    // Patches by ptex face, in table order.
    let mut first_ptex = Vec::with_capacity(cage.face_counts.len());
    let mut ptex = 0u32;
    for &c in &cage.face_counts {
        first_ptex.push(ptex);
        ptex += ptex_count(c);
    }
    let mut by_face: Vec<Vec<usize>> = vec![Vec::new(); ptex as usize];
    let patches = &refined.patches;
    for (p, &raw) in patches.params.iter().enumerate() {
        let (face, _) = param(patches.types[p], raw);
        let list = by_face
            .get_mut(face as usize)
            .ok_or_else(|| format!("OpenSubdiv made a patch of ptex face {face}"))?;
        list.push(p);
    }
    // Ptex faces grouped by cage face, cage faces by section.
    let mut order: Vec<u32> = (0..cage.face_counts.len() as u32).collect();
    order.sort_by_key(|&f| (section_of(f), f));

    let reach: Vec<f32> = materials
        .iter()
        .map(|m| m.displacement_min.abs().max(m.displacement_max.abs()))
        .collect();
    let mut pages = Vec::new();
    let mut infos: Vec<PageInfo> = Vec::new();
    let mut page = PageBuilder::default();
    let mut bounds = Aabb::EMPTY;
    let mut face_count = 0u32;
    for f in order {
        let first = first_ptex[f as usize];
        let faces: Vec<u32> = (first..first + ptex_count(cage.face_counts[f as usize]))
            .filter(|&q| !by_face[q as usize].is_empty())
            .collect();
        if faces.is_empty() {
            continue;
        }
        let section = section_of(f);
        let grow = reach.get(usize::from(section)).copied().unwrap_or(0.0);
        let mut cost = page.cost(&faces, &by_face, refined);
        if page.bytes() + cost > PAGE_MAX && !page.faces.is_empty() {
            let (bytes, info) = page.finish(pages.len() as u32 + 1);
            pages.push(bytes);
            infos.push(info);
            page = PageBuilder::default();
            cost = page.cost(&faces, &by_face, refined);
        }
        if page.bytes() + cost > PAGE_MAX {
            return Err(format!(
                "face {f} needs more than a {} KiB page; lower the isolation",
                PAGE_MAX >> 10
            ));
        }
        for q in faces {
            let corner = (q - first) as u8;
            let mut b = Aabb::EMPTY;
            let first_patch = page.patches.len() as u32;
            for &p in &by_face[q as usize] {
                let cvs = &patches.cvs[patches.first[p]..patches.first[p + 1]];
                let (_, prm) = param(patches.types[p], patches.params[p]);
                for c in hull(&prm, cvs, &rest) {
                    b.extend(c);
                }
                page.patch(p, prm, refined);
            }
            // Grown by the displacement reach and by rounding of the evaluated surface.
            let size = b
                .min
                .iter()
                .chain(&b.max)
                .fold(0.0f32, |a, c| a.max(c.abs()));
            let grow = grow + size * 4.0 * f32::EPSILON;
            b.min = b.min.map(|m| m - grow);
            b.max = b.max.map(|m| m + grow);
            bounds.extend(b.min);
            bounds.extend(b.max);
            page.faces.push(Face {
                bounds: b,
                ptex: q,
                base: f,
                section,
                corner,
                _pad: 0,
                patches: Span {
                    first: first_patch,
                    count: page.patches.len() as u32 - first_patch,
                },
            });
            face_count += 1;
        }
    }
    if !page.faces.is_empty() {
        let (bytes, info) = page.finish(pages.len() as u32 + 1);
        pages.push(bytes);
        infos.push(info);
    }
    if pages.is_empty() {
        return Err("the cage has no patches (every face is a hole)".into());
    }
    let patch_count = infos.iter().map(|i| i.patch_count).sum::<u32>();
    let sections = materials
        .iter()
        .enumerate()
        .map(|(i, m)| Section {
            material_slot: i as u16,
            flags: crate::mesh::section_flags(m),
            _pad: 0,
        })
        .collect();
    let mut attributes = 0;
    if cage.uvs.is_some() {
        attributes |= SubdHeader::UV;
    }
    if data.skin.is_some() {
        attributes |= SubdHeader::SKIN;
    }
    let info = SubdInfo {
        header: SubdHeader {
            bounds,
            face_count,
            patch_count,
            isolation: settings.isolation,
            smooth_isolation: settings.smooth_isolation,
            boundary: match cage.boundary {
                Boundary::None => SubdHeader::BOUNDARY_NONE,
                Boundary::Edges => SubdHeader::BOUNDARY_EDGES,
                Boundary::EdgesAndCorners => SubdHeader::BOUNDARY_EDGES_AND_CORNERS,
            },
            uv_linear: cage.uv_linear as u8,
            attributes,
            ..bytemuck::Zeroable::zeroed()
        },
        sections,
        points,
        uvs: cage
            .uvs
            .as_ref()
            .map(|u| u.values.iter().map(orr_math::Vec2::to_array).collect())
            .unwrap_or_default(),
        skin: data.skin.clone().unwrap_or_default(),
        pages: infos,
    };
    Ok(Cooked { info, pages })
}

/// The points whose box bounds a patch: the Bezier points of a B-spline patch (with its
/// missing points extrapolated across boundaries), the points of the other kinds.
fn hull(p: &PatchParam, cvs: &[u32], rest: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let at = |i: usize| Vec3::from_array(rest[cvs[i] as usize]);
    if p.kind != PatchParam::REGULAR {
        return (0..cvs.len()).map(|i| at(i).to_array()).collect();
    }
    let mut q: [Vec3; 16] = std::array::from_fn(at);
    let b = p.boundary;
    if b & 8 != 0 {
        for r in 0..4 {
            q[4 * r] = 2.0 * q[4 * r + 1] - q[4 * r + 2];
        }
    }
    if b & 4 != 0 {
        for c in 0..4 {
            q[12 + c] = 2.0 * q[8 + c] - q[4 + c];
        }
    }
    if b & 2 != 0 {
        for r in 0..4 {
            q[4 * r + 3] = 2.0 * q[4 * r + 2] - q[4 * r + 1];
        }
    }
    if b & 1 != 0 {
        for c in 0..4 {
            q[c] = 2.0 * q[4 + c] - q[8 + c];
        }
    }
    let bezier = |p: [Vec3; 4]| {
        [
            (p[0] + 4.0 * p[1] + p[2]) / 6.0,
            (4.0 * p[1] + 2.0 * p[2]) / 6.0,
            (2.0 * p[1] + 4.0 * p[2]) / 6.0,
            (p[1] + 4.0 * p[2] + p[3]) / 6.0,
        ]
    };
    let rows: [[Vec3; 4]; 4] =
        std::array::from_fn(|r| bezier([q[4 * r], q[4 * r + 1], q[4 * r + 2], q[4 * r + 3]]));
    (0..4)
        .flat_map(|c| bezier(rows.map(|r| r[c])).map(|v| v.to_array()))
        .collect()
}

/// A page being filled: its tables and where each refined point landed.
#[derive(Default)]
struct PageBuilder {
    page: PageTables,
    faces: Vec<Face>,
    patches: Vec<Patch>,
    points: HashMap<u32, u16>,
    uv_points: HashMap<u32, u16>,
}

#[derive(Default)]
struct PageTables {
    cvs: Vec<u16>,
    stencils: Vec<u32>,
    stencil_points: Vec<u32>,
    stencil_weights: Vec<f32>,
    uv_cvs: Vec<u16>,
    uv_stencils: Vec<u32>,
    uv_stencil_points: Vec<u32>,
    uv_stencil_weights: Vec<f32>,
}

impl PageBuilder {
    fn bytes(&self) -> usize {
        let t = &self.page;
        // Header, ten tables each padded to 4 bytes, and the closing stencil offsets.
        size_of::<PageHeader>()
            + 40
            + 8
            + self.faces.len() * size_of::<Face>()
            + self.patches.len() * size_of::<Patch>()
            + 2 * (t.cvs.len() + t.uv_cvs.len())
            + 4 * (t.stencils.len() + t.uv_stencils.len())
            + 8 * (t.stencil_points.len() + t.uv_stencil_points.len())
    }

    /// Bytes the ptex faces would add.
    fn cost(&self, faces: &[u32], by_face: &[Vec<usize>], r: &Refined) -> usize {
        let mut new = HashSet::new();
        let mut new_uv = HashSet::new();
        let mut bytes = faces.len() * size_of::<Face>();
        for &q in faces {
            for &p in &by_face[q as usize] {
                bytes += size_of::<Patch>();
                let cvs = &r.patches.cvs[r.patches.first[p]..r.patches.first[p + 1]];
                bytes += 2 * cvs.len();
                for &c in cvs {
                    if !self.points.contains_key(&c) && new.insert(c) {
                        bytes += 4 + 8 * r.stencils.terms(c as usize).len();
                    }
                }
                if let Some((s, uv)) = &r.uv {
                    let cvs = &uv.cvs[uv.first[p]..uv.first[p + 1]];
                    bytes += 2 * cvs.len();
                    for &c in cvs {
                        if !self.uv_points.contains_key(&c) && new_uv.insert(c) {
                            bytes += 4 + 8 * s.terms(c as usize).len();
                        }
                    }
                }
            }
        }
        bytes
    }

    fn patch(&mut self, p: usize, prm: PatchParam, r: &Refined) {
        let t = &mut self.page;
        let cvs = t.cvs.len() as u32;
        for &c in &r.patches.cvs[r.patches.first[p]..r.patches.first[p + 1]] {
            let local = *self.points.entry(c).or_insert_with(|| {
                let s = &r.stencils;
                t.stencils.push(t.stencil_points.len() as u32);
                for k in s.terms(c as usize) {
                    t.stencil_points.push(s.indices[k]);
                    t.stencil_weights.push(s.weights[k]);
                }
                (t.stencils.len() - 1) as u16
            });
            t.cvs.push(local);
        }
        let uv_cvs = t.uv_cvs.len() as u32;
        let mut uv_param = PatchParam {
            kind: PatchParam::NONE,
            ..Default::default()
        };
        if let Some((s, uv)) = &r.uv {
            uv_param = param(uv.types[p], uv.params[p]).1;
            for &c in &uv.cvs[uv.first[p]..uv.first[p + 1]] {
                let local = *self.uv_points.entry(c).or_insert_with(|| {
                    t.uv_stencils.push(t.uv_stencil_points.len() as u32);
                    for k in s.terms(c as usize) {
                        t.uv_stencil_points.push(s.indices[k]);
                        t.uv_stencil_weights.push(s.weights[k]);
                    }
                    (t.uv_stencils.len() - 1) as u16
                });
                t.uv_cvs.push(local);
            }
        }
        self.patches.push(Patch {
            param: prm,
            uv_param,
            cvs,
            uv_cvs,
        });
    }

    fn finish(mut self, blob: u32) -> (Vec<u8>, PageInfo) {
        let t = &mut self.page;
        t.stencils.push(t.stencil_points.len() as u32);
        if !t.uv_stencils.is_empty() {
            t.uv_stencils.push(t.uv_stencil_points.len() as u32);
        }
        let mut bounds = Aabb::EMPTY;
        for f in &self.faces {
            bounds.extend(f.bounds.min);
            bounds.extend(f.bounds.max);
        }
        let page = Page {
            faces: self.faces,
            patches: self.patches,
            cvs: std::mem::take(&mut t.cvs),
            stencils: std::mem::take(&mut t.stencils),
            stencil_points: std::mem::take(&mut t.stencil_points),
            stencil_weights: std::mem::take(&mut t.stencil_weights),
            uv_cvs: std::mem::take(&mut t.uv_cvs),
            uv_stencils: std::mem::take(&mut t.uv_stencils),
            uv_stencil_points: std::mem::take(&mut t.uv_stencil_points),
            uv_stencil_weights: std::mem::take(&mut t.uv_stencil_weights),
        };
        let bytes = page.to_bytes();
        let info = PageInfo {
            blob,
            len: bytes.len() as u32,
            face_count: page.faces.len() as u32,
            patch_count: page.patches.len() as u32,
            bounds,
        };
        (bytes, info)
    }
}

#[cfg(test)]
mod tests {
    use orr_math::Vec2;
    use orr_pack::subd::Page;

    use super::*;
    use crate::subd::{CageInput, Corner, Crease, FaceVarying, UvLinear, cook, osd};

    /// Points of a disc of `k` quads around a center of valence `k`, lifted off the plane,
    /// at `origin`; returns the cage's first point.
    fn star(c: &mut CageInput, k: u32, origin: Vec3) -> u32 {
        let base = c.positions.len() as u32;
        let lift = |p: Vec3| p + Vec3::new(0.0, 0.0, 0.3 * (p.x * 1.7).sin() * (p.y * 1.3).cos());
        c.positions.push(origin + lift(Vec3::ZERO));
        for i in 0..2 * k {
            let a = std::f32::consts::PI * i as f32 / k as f32;
            let r = if i % 2 == 0 { 1.0 } else { 1.3 };
            c.positions
                .push(origin + lift(Vec3::new(r * a.cos(), r * a.sin(), 0.0)));
        }
        for i in 0..k {
            let ring = |j: u32| base + 1 + (2 * i + j) % (2 * k);
            c.face_counts.push(4);
            c.face_indices.extend([base, ring(0), ring(1), ring(2)]);
        }
        base
    }

    /// Stars of valence 3 to 8, a closed cube, and a pentagon, with creases of finite and
    /// infinite sharpness, a corner, a hole, boundaries, and planar uvs with one seam.
    fn test_cage() -> CageInput {
        let mut c = CageInput::default();
        let mut first = Vec::new();
        for k in 3..=8 {
            first.push(star(&mut c, k, Vec3::new(3.0 * k as f32, 0.0, 0.0)));
        }
        // Crease a spoke of the valence-5 star, sharpen a spoke of the valence-6 one, put a
        // corner on the valence-7 ring and a hole in the valence-8 disc.
        c.creases.push(Crease {
            edge: [first[2], first[2] + 1],
            sharpness: 2.5,
        });
        c.creases.push(Crease {
            edge: [first[3], first[3] + 3],
            sharpness: 10.0,
        });
        c.corners.push(Corner {
            point: first[4] + 5,
            sharpness: 3.0,
        });
        let star_faces: u32 = (3..=8).sum();
        c.holes.push(star_faces - 2);
        // A cube: valence-3 corners on a closed surface, one edge sharp.
        let base = c.positions.len() as u32;
        for i in 0..8 {
            let b = |k: u32| if i & (1 << k) != 0 { 1.0 } else { -1.0 };
            c.positions.push(Vec3::new(b(0), b(1) + 6.0, b(2)));
        }
        for f in [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ] {
            c.face_counts.push(4);
            c.face_indices.extend(f.map(|v| base + v));
        }
        c.creases.push(Crease {
            edge: [base, base + 1],
            sharpness: 1.0,
        });
        // A lone pentagon: five ptex faces, every edge a boundary.
        let base = c.positions.len() as u32;
        for i in 0..5 {
            let a = std::f32::consts::TAU * i as f32 / 5.0;
            c.positions
                .push(Vec3::new(a.cos(), a.sin() - 6.0, 0.2 * (3.0 * a).sin()));
        }
        c.face_counts.push(5);
        c.face_indices.extend(base..base + 5);
        // Planar uvs, one value per point, with a seam along the valence-4 star's first
        // spoke: its first face reads a copy of the spoke's points.
        let mut values: Vec<Vec2> = c
            .positions
            .iter()
            .map(|p| Vec2::new(p.x, p.y) * 0.1)
            .collect();
        let mut indices = c.face_indices.clone();
        let seam = (3 * 4) as usize;
        for k in [seam, seam + 1] {
            values.push(values[indices[k] as usize] + Vec2::new(0.01, 0.0));
            indices[k] = values.len() as u32 - 1;
        }
        c.uvs = Some(FaceVarying { values, indices });
        c.uv_linear = UvLinear::CornersPlus1;
        c.face_sections = (0..c.face_counts.len()).map(|f| (f % 2) as u16).collect();
        c
    }

    #[test]
    fn limit_matches_opensubdiv_and_bounds_hold() {
        let cage = test_cage();
        let settings = SubdSettings::default();
        let mut materials = [Material::default(); 2];
        materials[1].displacement_min = -0.05;
        materials[1].displacement_max = 0.1;
        let data = CageData {
            cage: cage.clone(),
            skin: None,
        };
        let cooked = cook(&data, &materials, &settings).unwrap();
        let info = &cooked.info;
        let size =
            Vec3::from_array(info.header.bounds.max) - Vec3::from_array(info.header.bounds.min);
        let size = size.max_element();
        let steps = [0.0, 0.13, 0.5, 0.77, 1.0];
        let mut at = Vec::new();
        let mut ours = Vec::new();
        let mut corner_counts = std::collections::BTreeSet::new();
        for (bytes, page_info) in cooked.pages.iter().zip(&info.pages) {
            let page = Page::parse(bytes).unwrap();
            assert_eq!(page.faces.len() as u32, page_info.face_count);
            let points = page.points(&info.points);
            let uv_points = page.uv_points(&info.uvs);
            for (f, face) in page.faces.iter().enumerate() {
                let reach = if face.section == 1 { 0.1 } else { 0.0 };
                for &u in &steps {
                    for &v in &steps {
                        let l = page.limit(f, u, v, &points).unwrap();
                        let uv = page.limit_uv(f, u, v, &uv_points).unwrap();
                        let n = l.normal();
                        for d in [-reach, reach] {
                            let p = std::array::from_fn::<f32, 3, _>(|k| l.position[k] + d * n[k]);
                            let b = face.bounds;
                            assert!(
                                (0..3).all(|k| b.min[k] <= p[k] && p[k] <= b.max[k]),
                                "face {} at ({u}, {v}): {p:?} outside {b:?}",
                                face.ptex
                            );
                        }
                        at.push((face.ptex, u, v));
                        ours.push((l.position, uv));
                    }
                }
                corner_counts.insert(cage.face_counts[face.base as usize]);
            }
        }
        let faces: u32 = cage.face_counts.iter().map(|&c| ptex_count(c)).sum();
        assert_eq!(info.header.face_count, faces - 1, "every face but the hole");
        assert!(corner_counts.contains(&5));
        let (theirs, their_uvs) = osd::limit(&cage, &settings, &at).unwrap();
        for (i, ((p, uv), (q, quv))) in ours.iter().zip(theirs.iter().zip(&their_uvs)).enumerate() {
            let e = (0..3).map(|k| (p[k] - q[k]).abs()).fold(0.0, f32::max);
            assert!(
                e <= 1e-5 * size,
                "{:?}: position {p:?}, OpenSubdiv {q:?}",
                at[i]
            );
            let e = (0..2).map(|k| (uv[k] - quv[k]).abs()).fold(0.0, f32::max);
            assert!(e <= 1e-5, "{:?}: uv {uv:?}, OpenSubdiv {quv:?}", at[i]);
        }
    }
}
