//! Patches into pages.

use std::collections::{HashMap, HashSet};
use std::ops::{Add, Div, Mul, Sub};

use orr_math::{Vec3, Vec4};
use orr_pack::material::Material;
use orr_pack::mesh::Section;
use orr_pack::subd::{
    ColorPart, Face, PAGE_MAX, Page, PageHeader, PageInfo, Patch, PatchParam, SubdHeader, SubdInfo,
};
use orr_pack::{Aabb, Span};

use super::osd::{Patches, Refined, Stencils};
use super::{Boundary, CageData, CageInput, ColorValues, Cooked, SubdSettings};
use crate::shading::stream_holds;

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

/// A page color point.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum ColorPoint {
    /// Point `point` of the refined points [`ColorLayer`] `layer` interpolates on.
    Refined { layer: u32, point: u32 },
    /// Face coordinates `(u, v)`, as bits, of the ptex face at corner `corner` of cage face
    /// `face`, in linear layer `layer`.
    Linear {
        layer: u32,
        face: u32,
        corner: u32,
        u: u32,
        v: u32,
    },
}

/// One [`ColorPart`] of a patch before it has a place in a page.
struct PatchColor {
    param: PatchParam,
    channels: u32,
    points: Vec<ColorPoint>,
}

/// A [`super::CageColor`] in the blob's color table.
struct ColorLayer<'a> {
    /// Its first value in the table.
    first: u32,
    /// The refined points and patches of a face-varying one.
    refined: Option<&'a (Stencils, Patches)>,
}

/// A cage's colors over its refinement.
struct Colors<'a> {
    cage: &'a CageInput,
    refined: &'a Refined,
    layers: Vec<ColorLayer<'a>>,
    /// The blob's color table.
    values: Vec<[f32; 4]>,
    /// Layers of each cage face, in order.
    of_face: Vec<Vec<u32>>,
    /// First face corner of each cage face.
    first_corner: Vec<u32>,
}

impl<'a> Colors<'a> {
    fn new(cage: &'a CageInput, refined: &'a Refined) -> Self {
        let mut values = Vec::new();
        let mut face_varying = refined.colors.iter();
        let mut of_face = vec![Vec::new(); cage.face_counts.len()];
        let mut layers = Vec::with_capacity(cage.colors.len());
        for (l, c) in cage.colors.iter().enumerate() {
            let first = values.len() as u32;
            let (own, refined) = match &c.values {
                ColorValues::Vertex(v) => (v, None),
                ColorValues::FaceVarying(f) => (&f.values, face_varying.next()),
                ColorValues::Linear(f) => (&f.values, None),
            };
            values.extend(own.iter().map(Vec4::to_array));
            layers.push(ColorLayer { first, refined });
            if c.faces.is_empty() {
                of_face.iter_mut().for_each(|f| f.push(l as u32));
            }
            for &f in &c.faces {
                of_face[f as usize].push(l as u32);
            }
        }
        let mut corner = 0;
        let first_corner = cage
            .face_counts
            .iter()
            .map(|&c| {
                corner += c;
                corner - c
            })
            .collect();
        Self {
            cage,
            refined,
            layers,
            values,
            of_face,
            first_corner,
        }
    }

    /// The parts of patch `p`, at `corner` of cage face `face`.
    fn parts(&self, face: u32, corner: u32, p: usize) -> Vec<PatchColor> {
        let patches = &self.refined.patches;
        let (_, surface) = param(patches.types[p], patches.params[p]);
        let on = |layer: u32, table: &Patches| {
            let points = &table.cvs[table.first[p]..table.first[p + 1]];
            let refined = |&point| ColorPoint::Refined { layer, point };
            points.iter().map(refined).collect()
        };
        self.of_face[face as usize]
            .iter()
            .map(|&layer| {
                let l = layer as usize;
                let (param, points) = match (&self.cage.colors[l].values, self.layers[l].refined) {
                    (ColorValues::Vertex(_), _) => (surface, on(layer, patches)),
                    (_, Some((_, own))) => (param(own.types[p], own.params[p]).1, on(layer, own)),
                    _ => {
                        let n = f32::from(1u16 << surface.levels());
                        let at = |d: (u16, u16)| ColorPoint::Linear {
                            layer,
                            face,
                            corner,
                            u: (f32::from(surface.u + d.0) / n).to_bits(),
                            v: (f32::from(surface.v + d.1) / n).to_bits(),
                        };
                        let param = PatchParam {
                            kind: PatchParam::LINEAR,
                            boundary: 0,
                            ..surface
                        };
                        (param, [(0, 0), (1, 0), (1, 1), (0, 1)].map(at).to_vec())
                    }
                };
                PatchColor {
                    param,
                    channels: u32::from(self.cage.colors[l].channels),
                    points,
                }
            })
            .collect()
    }

    /// The stencil of a color point over the blob's color table.
    fn terms(&self, point: ColorPoint) -> Vec<(u32, f32)> {
        match point {
            ColorPoint::Refined { layer, point } => {
                let l = &self.layers[layer as usize];
                let s = l.refined.map_or(&self.refined.stencils, |r| &r.0);
                s.terms(point as usize)
                    .map(|k| (l.first + s.indices[k], s.weights[k]))
                    .collect()
            }
            ColorPoint::Linear {
                layer,
                face,
                corner,
                u,
                v,
            } => self.linear_terms(layer, face, corner, f32::from_bits(u), f32::from_bits(v)),
        }
    }

    /// Linear layer `layer` at `(u, v)` of a ptex face, whose corners [`SubdInfo::ptex_corners`]
    /// states: a quad's are its own, an n-sided face's a corner, its edges' middles and its center.
    fn linear_terms(&self, layer: u32, face: u32, corner: u32, u: f32, v: f32) -> Vec<(u32, f32)> {
        let l = layer as usize;
        let ColorValues::Linear(values) = &self.cage.colors[l].values else {
            unreachable!("a linear color point of a layer of another interpolation");
        };
        let f = face as usize;
        let n = self.cage.face_counts[f] as usize;
        let first = self.first_corner[f] as usize;
        // As the refiner reads the face: a left-handed one's corners after its first reverse.
        let at = |i: usize| {
            let i = i % n;
            let k = if self.cage.left_handed && i != 0 {
                n - i
            } else {
                i
            };
            self.layers[l].first + values.indices[first + k]
        };
        let w = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
        let mut terms = Vec::new();
        if n == 4 {
            terms.extend((0..4).map(|i| (at(i), w[i])));
        } else {
            let k = corner as usize;
            terms.push((at(k), w[0] + 0.5 * (w[1] + w[3])));
            terms.push((at(k + 1), 0.5 * w[1]));
            terms.push((at(k + n - 1), 0.5 * w[3]));
            terms.extend((0..n).map(|i| (at(i), w[2] / n as f32)));
        }
        terms.retain(|t| t.1 != 0.0);
        terms
    }
}

pub(super) fn build(
    data: &CageData,
    materials: &[Material],
    settings: &SubdSettings,
    refined: &Refined,
) -> Result<Cooked, String> {
    let cage = &data.cage;
    let points: Vec<[f32; 3]> = cage.positions.iter().map(Vec3::to_array).collect();
    let rest: Vec<Vec3> = refined
        .stencils
        .apply(&points)
        .into_iter()
        .map(Vec3::from_array)
        .collect();
    let colors = Colors::new(cage, refined);
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

    let reach: Vec<f32> = materials.iter().map(Material::displacement_reach).collect();
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
        let mut cost = page.cost(f, first, &faces, &by_face, refined, &colors);
        if page.bytes() + cost > PAGE_MAX && !page.faces.is_empty() {
            let (bytes, info) = page.finish(pages.len() as u32 + 1, &colors.values)?;
            pages.push(bytes);
            infos.push(info);
            page = PageBuilder::default();
            cost = page.cost(f, first, &faces, &by_face, refined, &colors);
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
                    b.extend(c.to_array());
                }
                let parts = colors.parts(f, u32::from(corner), p);
                page.patch(p, prm, refined, &parts, &colors);
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
        let (bytes, info) = page.finish(pages.len() as u32 + 1, &colors.values)?;
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
    // Corners as the refiner reads them: a left-handed face keeps its first corner and
    // reverses the rest.
    let mut cage_faces = Vec::with_capacity(cage.face_counts.len() + 1);
    let mut cage_corners = Vec::with_capacity(cage.face_indices.len());
    for &c in &cage.face_counts {
        let first = cage_corners.len();
        cage_faces.push(first as u32);
        let face = &cage.face_indices[first..first + c as usize];
        cage_corners.push(face[0]);
        if cage.left_handed {
            cage_corners.extend(face[1..].iter().rev());
        } else {
            cage_corners.extend(&face[1..]);
        }
    }
    cage_faces.push(cage_corners.len() as u32);
    let mut attributes = 0;
    if cage.uvs.is_some() {
        attributes |= SubdHeader::UV;
    }
    if data.skin.is_some() {
        attributes |= SubdHeader::SKIN;
    }
    if !cage.colors.is_empty() {
        attributes |= SubdHeader::COLOR;
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
        colors: colors.values,
        skin: data.skin.clone().unwrap_or_default(),
        pages: infos,
        cage_faces,
        cage_corners,
    };
    Ok(Cooked { info, pages })
}

/// The points whose box bounds a patch: the Bezier points of a B-spline patch (with its
/// missing points extrapolated across boundaries), the points of the other kinds.
fn hull<T>(p: &PatchParam, cvs: &[u32], rest: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<f32, Output = T> + Div<f32, Output = T>,
{
    let at = |i: usize| rest[cvs[i] as usize];
    if p.kind != PatchParam::REGULAR {
        return (0..cvs.len()).map(at).collect();
    }
    let mut q: [T; 16] = std::array::from_fn(at);
    let b = p.boundary;
    if b & 8 != 0 {
        for r in 0..4 {
            q[4 * r] = q[4 * r + 1] * 2.0 - q[4 * r + 2];
        }
    }
    if b & 4 != 0 {
        for c in 0..4 {
            q[12 + c] = q[8 + c] * 2.0 - q[4 + c];
        }
    }
    if b & 2 != 0 {
        for r in 0..4 {
            q[4 * r + 3] = q[4 * r + 2] * 2.0 - q[4 * r + 1];
        }
    }
    if b & 1 != 0 {
        for c in 0..4 {
            q[c] = q[4 + c] * 2.0 - q[8 + c];
        }
    }
    let bezier = |p: [T; 4]| {
        [
            (p[0] + p[1] * 4.0 + p[2]) / 6.0,
            (p[1] * 4.0 + p[2] * 2.0) / 6.0,
            (p[1] * 2.0 + p[2] * 4.0) / 6.0,
            (p[1] + p[2] * 4.0 + p[3]) / 6.0,
        ]
    };
    let rows: [[T; 4]; 4] =
        std::array::from_fn(|r| bezier([q[4 * r], q[4 * r + 1], q[4 * r + 2], q[4 * r + 3]]));
    (0..4).flat_map(|c| bezier(rows.map(|r| r[c]))).collect()
}

/// A page being filled: its tables and where each refined point landed.
#[derive(Default)]
struct PageBuilder {
    page: PageTables,
    faces: Vec<Face>,
    patches: Vec<Patch>,
    points: HashMap<u32, u16>,
    uv_points: HashMap<u32, u16>,
    color_parts: Vec<ColorPart>,
    color_points: HashMap<ColorPoint, u16>,
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
    color_cvs: Vec<u16>,
    color_stencils: Vec<u32>,
    color_stencil_points: Vec<u32>,
    color_stencil_weights: Vec<f32>,
}

impl PageBuilder {
    fn bytes(&self) -> usize {
        let t = &self.page;
        // Header, fifteen tables each padded to 4 bytes, and the closing stencil offsets.
        size_of::<PageHeader>()
            + 60
            + 12
            + self.faces.len() * size_of::<Face>()
            + self.patches.len() * size_of::<Patch>()
            + self.color_parts.len() * size_of::<ColorPart>()
            + 2 * (t.cvs.len() + t.uv_cvs.len() + t.color_cvs.len())
            + 4 * (t.stencils.len() + t.uv_stencils.len() + t.color_stencils.len())
            + 8 * (t.stencil_points.len()
                + t.uv_stencil_points.len()
                + t.color_stencil_points.len())
    }

    /// Bytes the ptex faces of cage face `face`, whose first is `first`, would add.
    fn cost(
        &self,
        face: u32,
        first: u32,
        faces: &[u32],
        by_face: &[Vec<usize>],
        r: &Refined,
        colors: &Colors<'_>,
    ) -> usize {
        let mut new = HashSet::new();
        let mut new_uv = HashSet::new();
        let mut new_color = HashSet::new();
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
                for part in colors.parts(face, q - first, p) {
                    bytes += size_of::<ColorPart>() + 2 * part.points.len();
                    for c in part.points {
                        if !self.color_points.contains_key(&c) && new_color.insert(c) {
                            bytes += 4 + 8 * colors.terms(c).len();
                        }
                    }
                }
            }
        }
        bytes
    }

    fn patch(
        &mut self,
        p: usize,
        prm: PatchParam,
        r: &Refined,
        parts: &[PatchColor],
        colors: &Colors<'_>,
    ) {
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
        let first_part = self.color_parts.len() as u32;
        for part in parts {
            let cvs = t.color_cvs.len() as u32;
            for &c in &part.points {
                let local = *self.color_points.entry(c).or_insert_with(|| {
                    t.color_stencils.push(t.color_stencil_points.len() as u32);
                    for (value, weight) in colors.terms(c) {
                        t.color_stencil_points.push(value);
                        t.color_stencil_weights.push(weight);
                    }
                    (t.color_stencils.len() - 1) as u16
                });
                t.color_cvs.push(local);
            }
            self.color_parts.push(ColorPart {
                param: part.param,
                cvs,
                channels: part.channels,
            });
        }
        self.patches.push(Patch {
            param: prm,
            uv_param,
            cvs,
            uv_cvs,
            colors: Span {
                first: first_part,
                count: parts.len() as u32,
            },
        });
    }

    fn finish(mut self, blob: u32, colors: &[[f32; 4]]) -> Result<(Vec<u8>, PageInfo), String> {
        let t = &mut self.page;
        t.stencils.push(t.stencil_points.len() as u32);
        if !t.uv_stencils.is_empty() {
            t.uv_stencils.push(t.uv_stencil_points.len() as u32);
        }
        if !t.color_stencils.is_empty() {
            t.color_stencils.push(t.color_stencil_points.len() as u32);
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
            color_parts: self.color_parts,
            color_cvs: std::mem::take(&mut t.color_cvs),
            color_stencils: std::mem::take(&mut t.color_stencils),
            color_stencil_points: std::mem::take(&mut t.color_stencil_points),
            color_stencil_weights: std::mem::take(&mut t.color_stencil_weights),
        };
        check_color_reach(&page, colors)?;
        let bytes = page.to_bytes();
        let info = PageInfo {
            blob,
            len: bytes.len() as u32,
            face_count: page.faces.len() as u32,
            patch_count: page.patches.len() as u32,
            bounds,
        };
        Ok((bytes, info))
    }
}

/// Refuses colors whose patches can pass what the stream's half floats hold: a patch's limit
/// lies in its hull, which a Gregory patch's points or a boundary's extrapolated ones can take
/// past the cage's values.
fn check_color_reach(page: &Page, colors: &[[f32; 4]]) -> Result<(), String> {
    let points: Vec<Vec4> = page
        .color_points(colors)
        .into_iter()
        .map(Vec4::from_array)
        .collect();
    for part in &page.color_parts {
        let first = part.cvs as usize;
        let cvs: Vec<u32> = page.color_cvs[first..first + part.param.cv_count()]
            .iter()
            .map(|&c| u32::from(c))
            .collect();
        for c in hull(&part.param, &cvs, &points) {
            let given = (0..4).filter(|k| part.channels & (1 << k) != 0);
            let peak = given.fold(0.0f32, |p, k| p.max(c[k].abs()));
            if !stream_holds(peak) {
                return Err(format!(
                    "a cage color reaches {peak} between its values, past what the stream's half \
                     floats hold"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use orr_math::Vec2;
    use orr_pack::subd::Page;

    use super::*;
    use crate::subd::{CageColor, Corner, Crease, FaceVarying, UvLinear, cook, osd};

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
        // Colors with a seam of their own, along another spoke.
        let mut colors: Vec<Vec4> = c
            .positions
            .iter()
            .map(|p| Vec4::new(p.x, p.y, p.z, p.x + p.y) * 0.25)
            .collect();
        let mut color_indices = c.face_indices.clone();
        for k in [0, 1] {
            colors.push(colors[color_indices[k] as usize] + Vec4::splat(0.5));
            color_indices[k] = colors.len() as u32 - 1;
        }
        c.colors.push(CageColor {
            values: ColorValues::FaceVarying(FaceVarying {
                values: colors,
                indices: color_indices,
            }),
            channels: 0b1111,
            faces: Vec::new(),
        });
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
        materials[1].displacement_tangent = [[0.0, 0.0, 0.0, 0.2], [0.0, 0.0, -0.05, 0.0]];
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
        let mut shared: HashMap<orr_pack::subd::CagePoint, [f32; 3]> = HashMap::new();
        for (bytes, page_info) in cooked.pages.iter().zip(&info.pages) {
            let page = Page::parse(bytes).unwrap();
            assert_eq!(page.faces.len() as u32, page_info.face_count);
            let points = page.points(&info.points);
            let uv_points = page.uv_points(&info.uvs);
            let color_points = page.color_points(&info.colors);
            for (f, face) in page.faces.iter().enumerate() {
                let corners = info.ptex_corners(face);
                for (k, (u, v)) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                    .into_iter()
                    .enumerate()
                {
                    let p = page.limit(f, u, v, &points).unwrap().position;
                    let q = *shared.entry(corners[k]).or_insert(p);
                    let e = (0..3).map(|i| (p[i] - q[i]).abs()).fold(0.0, f32::max);
                    assert!(e <= 1e-5 * size, "{:?} of face {}", corners[k], face.ptex);
                }
                let reach = if face.section == 1 { 0.35 } else { 0.0 };
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
                        let color = page.limit_color(f, u, v, &color_points).unwrap();
                        at.push((face.ptex, u, v));
                        ours.push((l.position, uv, color));
                    }
                }
                corner_counts.insert(cage.face_counts[face.base as usize]);
            }
        }
        let faces: u32 = cage.face_counts.iter().map(|&c| ptex_count(c)).sum();
        assert_eq!(info.header.face_count, faces - 1, "every face but the hole");
        assert!(corner_counts.contains(&5));
        let (theirs, their_uvs) = osd::limit(&cage, &settings, &at).unwrap();
        let their_colors = osd::limit_color(&cage, &settings, &at).unwrap();
        for (i, (p, uv, color)) in ours.iter().enumerate() {
            let (q, quv, qcolor) = (&theirs[i], &their_uvs[i], &their_colors[i]);
            let e = (0..3).map(|k| (p[k] - q[k]).abs()).fold(0.0, f32::max);
            assert!(
                e <= 1e-5 * size,
                "{:?}: position {p:?}, OpenSubdiv {q:?}",
                at[i]
            );
            let e = (0..2).map(|k| (uv[k] - quv[k]).abs()).fold(0.0, f32::max);
            assert!(e <= 1e-5, "{:?}: uv {uv:?}, OpenSubdiv {quv:?}", at[i]);
            let e = (0..4)
                .map(|k| (color[k] - qcolor[k]).abs())
                .fold(0.0, f32::max);
            assert!(
                e <= 1e-5,
                "{:?}: color {color:?}, OpenSubdiv {qcolor:?}",
                at[i]
            );
        }
    }
}
