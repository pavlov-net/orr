//! Patches into pages.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::{Add, Div, Mul, Sub};

use orr_math::{Vec2, Vec3, Vec4};
use orr_pack::material::Material;
use orr_pack::mesh::Section;
use orr_pack::subd::{
    Face, PAGE_MAX, Page, PageHeader, PageInfo, Part, Patch, PatchParam, SubdHeader, SubdInfo,
};
use orr_pack::{Aabb, Span};

use super::osd::{Patches, Refined, Stencils};
use super::{Boundary, CageData, CageInput, CageValues, Cooked, SubdSettings};
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

/// A page value point.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum ValuePoint {
    /// Point `point` of the refined points [`Layer`] `layer` interpolates on.
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

/// One [`Part`] of a patch before it has a place in a page.
struct PatchPart {
    param: PatchParam,
    channels: u32,
    points: Vec<ValuePoint>,
}

/// What a [`Layer`] interpolates on.
#[derive(Copy, Clone)]
enum Basis<'a> {
    /// The surface's points and patches.
    Surface,
    /// Its own refined points and patches.
    Own(&'a (Stencils, Patches)),
    /// Each face's corners, by these value indices.
    Linear(&'a [u32]),
}

/// A [`super::CageUv`] or [`super::CageColor`] in the blob's value table.
struct Layer<'a> {
    /// Its first value in the table.
    first: u32,
    basis: Basis<'a>,
    /// [`Part::channels`].
    channels: u32,
}

/// A cage's uvs and colors over its refinement.
struct Layers<'a> {
    cage: &'a CageInput,
    refined: &'a Refined,
    layers: Vec<Layer<'a>>,
    /// The blob's value table.
    values: Vec<[f32; 4]>,
    /// Layers of each cage face, in order.
    of_face: Vec<Vec<u32>>,
    /// First face corner of each cage face.
    first_corner: Vec<u32>,
}

impl<'a> Layers<'a> {
    fn new(cage: &'a CageInput, refined: &'a Refined) -> Self {
        let mut corner = 0;
        let first_corner = cage
            .face_counts
            .iter()
            .map(|&c| {
                corner += c;
                corner - c
            })
            .collect();
        let mut out = Self {
            cage,
            refined,
            layers: Vec::with_capacity(cage.uvs.len() + cage.colors.len()),
            values: Vec::new(),
            of_face: vec![Vec::new(); cage.face_counts.len()],
            first_corner,
        };
        let mut fvar = refined.fvar.iter();
        for u in &cage.uvs {
            let wide = |v: &Vec2| [v.x, v.y, 0.0, 0.0];
            let channels = Part::UV << u.set;
            out.add(&u.values, channels, &u.faces, wide, &mut fvar);
        }
        for c in &cage.colors {
            let channels = u32::from(c.channels);
            out.add(&c.values, channels, &c.faces, Vec4::to_array, &mut fvar);
        }
        out
    }

    /// Adds a layer of `values`, four wide by `wide`, that gives `channels` over `faces`.
    fn add<T>(
        &mut self,
        values: &'a CageValues<T>,
        channels: u32,
        faces: &[u32],
        wide: impl Fn(&T) -> [f32; 4],
        fvar: &mut std::slice::Iter<'a, (Stencils, Patches)>,
    ) {
        let (own, basis) = match values {
            CageValues::Vertex(v) => (v, Basis::Surface),
            CageValues::FaceVarying(f) => {
                let channel = fvar.next().expect("OpenSubdiv refined every channel");
                (&f.values, Basis::Own(channel))
            }
            CageValues::Linear(f) => (&f.values, Basis::Linear(&f.indices)),
        };
        let l = self.layers.len() as u32;
        self.layers.push(Layer {
            first: self.values.len() as u32,
            basis,
            channels,
        });
        self.values.extend(own.iter().map(wide));
        if faces.is_empty() {
            self.of_face.iter_mut().for_each(|f| f.push(l));
        }
        for &f in faces {
            self.of_face[f as usize].push(l);
        }
    }

    /// The parts of patch `p`, at `corner` of cage face `face`.
    fn parts(&self, face: u32, corner: u32, p: usize) -> Vec<PatchPart> {
        let patches = &self.refined.patches;
        let (_, surface) = param(patches.types[p], patches.params[p]);
        let on = |layer: u32, table: &Patches| {
            let points = &table.cvs[table.first[p]..table.first[p + 1]];
            let refined = |&point| ValuePoint::Refined { layer, point };
            points.iter().map(refined).collect()
        };
        self.of_face[face as usize]
            .iter()
            .map(|&layer| {
                let l = &self.layers[layer as usize];
                let (param, points) = match l.basis {
                    Basis::Surface => (surface, on(layer, patches)),
                    Basis::Own((_, own)) => (param(own.types[p], own.params[p]).1, on(layer, own)),
                    Basis::Linear(_) => {
                        let n = f32::from(1u16 << surface.levels());
                        let at = |d: (u16, u16)| ValuePoint::Linear {
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
                PatchPart {
                    param,
                    channels: l.channels,
                    points,
                }
            })
            .collect()
    }

    /// The stencil of a value point over the blob's value table.
    fn terms(&self, point: ValuePoint) -> Vec<(u32, f32)> {
        match point {
            ValuePoint::Refined { layer, point } => {
                let l = &self.layers[layer as usize];
                let s = match l.basis {
                    Basis::Own((s, _)) => s,
                    _ => &self.refined.stencils,
                };
                s.terms(point as usize)
                    .map(|k| (l.first + s.indices[k], s.weights[k]))
                    .collect()
            }
            ValuePoint::Linear {
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
        let l = &self.layers[layer as usize];
        let Basis::Linear(indices) = l.basis else {
            unreachable!("a linear value point of a layer of another interpolation");
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
            l.first + indices[first + k]
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
    let layers = Layers::new(cage, refined);
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
        let mut cost = page.cost(f, first, &faces, &by_face, refined, &layers);
        if page.bytes() + cost > PAGE_MAX && !page.faces.is_empty() {
            let (bytes, info) = page.finish(pages.len() as u32 + 1, &layers.values)?;
            pages.push(bytes);
            infos.push(info);
            page = PageBuilder::default();
            cost = page.cost(f, first, &faces, &by_face, refined, &layers);
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
                let parts = layers.parts(f, u32::from(corner), p);
                page.patch(p, prm, refined, &parts, &layers);
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
        let (bytes, info) = page.finish(pages.len() as u32 + 1, &layers.values)?;
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
    if !cage.uvs.is_empty() {
        attributes |= SubdHeader::UV;
    }
    if cage.uvs.iter().any(|u| u.set == 1) || materials.iter().any(crate::mesh::reads_uv1) {
        attributes |= SubdHeader::UV1;
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
        values: layers.values,
        skin: data.skin.clone().unwrap_or_default(),
        pages: infos,
        cage_faces,
        cage_corners,
    };
    Ok(Cooked {
        info,
        pages,
        warnings: uv1_warnings(cage, materials),
    })
}

/// A warning per section that reads uv set 1 on drawn faces no [`super::CageUv`] of the set
/// covers, which read uv 0.
fn uv1_warnings(cage: &CageInput, materials: &[Material]) -> Vec<String> {
    let mut covered = vec![false; cage.face_counts.len()];
    for &hole in &cage.holes {
        covered[hole as usize] = true;
    }
    for u in cage.uvs.iter().filter(|u| u.set == 1) {
        if u.faces.is_empty() {
            covered.fill(true);
        }
        for &f in &u.faces {
            covered[f as usize] = true;
        }
    }
    let lacking: BTreeSet<u16> = (0..covered.len())
        .filter(|&f| !covered[f])
        .map(|f| cage.face_sections.get(f).copied().unwrap_or(0))
        .collect();
    // A library caller's cage may name no materials: its faces are in a section 0 that isn't there.
    lacking
        .into_iter()
        .filter(|&s| {
            materials
                .get(usize::from(s))
                .is_some_and(crate::mesh::reads_uv1)
        })
        .map(|s| format!("section {s} reads uv set 1, which faces of it lack; they read uv 0"))
        .collect()
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
    parts: Vec<Part>,
    value_points: HashMap<ValuePoint, u16>,
}

#[derive(Default)]
struct PageTables {
    cvs: Vec<u16>,
    stencils: Vec<u32>,
    stencil_points: Vec<u32>,
    stencil_weights: Vec<f32>,
    part_cvs: Vec<u16>,
    value_stencils: Vec<u32>,
    value_stencil_points: Vec<u32>,
    value_stencil_weights: Vec<f32>,
}

impl PageBuilder {
    fn bytes(&self) -> usize {
        let t = &self.page;
        // Header, eleven tables each padded to 4 bytes, and the closing stencil offsets.
        size_of::<PageHeader>()
            + 44
            + 8
            + self.faces.len() * size_of::<Face>()
            + self.patches.len() * size_of::<Patch>()
            + self.parts.len() * size_of::<Part>()
            + 2 * (t.cvs.len() + t.part_cvs.len())
            + 4 * (t.stencils.len() + t.value_stencils.len())
            + 8 * (t.stencil_points.len() + t.value_stencil_points.len())
    }

    /// Bytes the ptex faces of cage face `face`, whose first is `first`, would add.
    fn cost(
        &self,
        face: u32,
        first: u32,
        faces: &[u32],
        by_face: &[Vec<usize>],
        r: &Refined,
        layers: &Layers<'_>,
    ) -> usize {
        let mut new = HashSet::new();
        let mut new_value = HashSet::new();
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
                for part in layers.parts(face, q - first, p) {
                    bytes += size_of::<Part>() + 2 * part.points.len();
                    for c in part.points {
                        if !self.value_points.contains_key(&c) && new_value.insert(c) {
                            bytes += 4 + 8 * layers.terms(c).len();
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
        parts: &[PatchPart],
        layers: &Layers<'_>,
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
        let first_part = self.parts.len() as u32;
        for part in parts {
            let cvs = t.part_cvs.len() as u32;
            for &c in &part.points {
                let local = *self.value_points.entry(c).or_insert_with(|| {
                    t.value_stencils.push(t.value_stencil_points.len() as u32);
                    for (value, weight) in layers.terms(c) {
                        t.value_stencil_points.push(value);
                        t.value_stencil_weights.push(weight);
                    }
                    (t.value_stencils.len() - 1) as u16
                });
                t.part_cvs.push(local);
            }
            self.parts.push(Part {
                param: part.param,
                cvs,
                channels: part.channels,
            });
        }
        self.patches.push(Patch {
            param: prm,
            cvs,
            parts: Span {
                first: first_part,
                count: parts.len() as u32,
            },
        });
    }

    fn finish(mut self, blob: u32, values: &[[f32; 4]]) -> Result<(Vec<u8>, PageInfo), String> {
        let t = &mut self.page;
        t.stencils.push(t.stencil_points.len() as u32);
        if !t.value_stencils.is_empty() {
            t.value_stencils.push(t.value_stencil_points.len() as u32);
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
            parts: self.parts,
            part_cvs: std::mem::take(&mut t.part_cvs),
            value_stencils: std::mem::take(&mut t.value_stencils),
            value_stencil_points: std::mem::take(&mut t.value_stencil_points),
            value_stencil_weights: std::mem::take(&mut t.value_stencil_weights),
        };
        check_color_reach(&page, values)?;
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
fn check_color_reach(page: &Page, values: &[[f32; 4]]) -> Result<(), String> {
    let points: Vec<Vec4> = page
        .value_points(values)
        .into_iter()
        .map(Vec4::from_array)
        .collect();
    for part in page
        .parts
        .iter()
        .filter(|p| p.channels & (Part::UV - 1) != 0)
    {
        let first = part.cvs as usize;
        let cvs: Vec<u32> = page.part_cvs[first..first + part.param.cv_count()]
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
    use orr_pack::subd::Page;

    use super::*;
    use crate::subd::{CageColor, CageUv, Corner, Crease, FaceVarying, UvLinear, cook, osd};

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
            values: CageValues::FaceVarying(FaceVarying {
                values: colors,
                indices: color_indices,
            }),
            channels: 0b1111,
            faces: Vec::new(),
        });
        c.uvs.push(CageUv {
            values: CageValues::FaceVarying(FaceVarying { values, indices }),
            set: 0,
            faces: Vec::new(),
        });
        c.uv_linear = UvLinear::CornersPlus1;
        c.face_sections = (0..c.face_counts.len()).map(|f| (f % 2) as u16).collect();
        c
    }

    #[test]
    fn uvs_read_their_interpolation_and_only_face_varying_refines_the_surface() {
        const N: u32 = 6;
        let point = |i: u32, j: u32| j * (N + 1) + i;
        let mut c = CageInput::default();
        let mut values = Vec::new();
        for j in 0..=N {
            for i in 0..=N {
                c.positions.push(Vec3::new(i as f32, j as f32, 0.0));
                // Too rough for a plane, so what smooths it moves it.
                values.push(Vec2::new(
                    (i * i % 5) as f32 * 0.2,
                    ((i + 2 * j) % 3) as f32 * 0.3,
                ));
            }
        }
        for j in 0..N {
            for i in 0..N {
                c.face_counts.push(4);
                c.face_indices.extend([
                    point(i, j),
                    point(i + 1, j),
                    point(i + 1, j + 1),
                    point(i, j + 1),
                ]);
            }
            c.creases.push(Crease {
                edge: [point(3, j), point(3, j + 1)],
                sharpness: 10.0,
            });
        }
        // Faces above row 3 read copies of its points, so a seam runs along it.
        let mut seamed = values.clone();
        let mut seam_indices = c.face_indices.clone();
        for (k, index) in seam_indices.iter_mut().enumerate() {
            if *index / (N + 1) == 3 && (k / 4) as u32 / N >= 3 {
                seamed.push(seamed[*index as usize] + Vec2::new(0.4, 0.0));
                *index = seamed.len() as u32 - 1;
            }
        }
        let rows = |r: std::ops::Range<u32>| (r.start * N..r.end * N).collect::<Vec<u32>>();
        let uv = |values, set, faces| CageUv { values, set, faces };
        let linear = FaceVarying {
            values: values.clone(),
            indices: c.face_indices.clone(),
        };
        let seam = FaceVarying {
            values: seamed,
            indices: seam_indices,
        };
        let surface = CageValues::Vertex(values.clone());
        let bare = c.clone();
        c.uvs = vec![
            uv(surface.clone(), 0, rows(0..2)),
            uv(CageValues::FaceVarying(seam.clone()), 0, rows(2..4)),
            uv(CageValues::Linear(linear.clone()), 0, rows(4..N)),
            uv(CageValues::Linear(linear.clone()), 1, Vec::new()),
        ];
        let settings = SubdSettings::default();
        let materials = [Material::default()];
        let cook_cage = |cage: &CageInput| {
            let data = CageData {
                cage: cage.clone(),
                skin: None,
            };
            cook(&data, &materials, &settings).unwrap()
        };
        let cooked = cook_cage(&c);

        let (mut at, mut ours, mut sets_one) = (Vec::new(), Vec::new(), Vec::new());
        for bytes in &cooked.pages {
            let page = Page::parse(bytes).unwrap();
            let points = page.value_points(&cooked.info.values);
            for (f, face) in page.faces.iter().enumerate() {
                for u in [0.0, 0.13, 0.5, 0.77, 1.0] {
                    for v in [0.0, 0.13, 0.5, 0.77, 1.0] {
                        at.push((face.ptex, u, v));
                        ours.push(Vec2::from(page.limit_uv(f, u, v, 0, &points).unwrap()));
                        sets_one.push(Vec2::from(page.limit_uv(f, u, v, 1, &points).unwrap()));
                    }
                }
            }
        }
        // OpenSubdiv's limit of the uvs as points, on the same refinement as the cooked cage's.
        let as_points = CageInput {
            positions: values.iter().map(|v| v.extend(0.0)).collect(),
            ..c.clone()
        };
        let on_surface = osd::limit(&as_points, &settings, &at).unwrap().0;
        let on_channel = osd::limit(&c, &settings, &at).unwrap().1;
        let bilinear = |&(f, u, v): &(u32, f32, f32)| {
            let k = &c.face_indices[4 * f as usize..4 * f as usize + 4];
            let p = |i: usize| values[k[i] as usize];
            (p(0) * (1.0 - u) + p(1) * u) * (1.0 - v) + (p(3) * (1.0 - u) + p(2) * u) * v
        };
        let mut clear = [0.0f32; 3];
        for (i, place) in at.iter().enumerate() {
            let readings = [
                Vec3::from(on_surface[i]).truncate(),
                Vec2::from(on_channel[i]),
                bilinear(place),
            ];
            let which = match place.0 / N {
                0..2 => 0,
                2..4 => 1,
                _ => 2,
            };
            let e = (ours[i] - readings[which]).abs().max_element();
            assert!(e <= 1e-5, "{place:?}: {}, {:?}", ours[i], readings[which]);
            let e = (sets_one[i] - bilinear(place)).abs().max_element();
            assert!(e <= 1e-5, "{place:?}: set 1 reads {}", sets_one[i]);
            for (k, other) in readings.iter().enumerate() {
                if k != which {
                    let d = (readings[which] - *other).abs().max_element();
                    clear[which] = clear[which].max(d);
                }
            }
        }
        assert!(clear.iter().all(|&d| d > 0.05), "{clear:?}: alike");

        let patches = |uvs: Vec<CageUv>| {
            let cage = CageInput {
                uvs,
                ..bare.clone()
            };
            cook_cage(&cage).info.header.patch_count
        };
        let plain = patches(Vec::new());
        assert_eq!(patches(vec![uv(surface, 0, Vec::new())]), plain);
        let linear = CageValues::Linear(linear);
        assert_eq!(patches(vec![uv(linear, 0, Vec::new())]), plain);
        let seam = CageValues::FaceVarying(seam);
        assert!(patches(vec![uv(seam, 0, Vec::new())]) > plain);
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
        let mut corner_counts = BTreeSet::new();
        let mut shared: HashMap<orr_pack::subd::CagePoint, [f32; 3]> = HashMap::new();
        for (bytes, page_info) in cooked.pages.iter().zip(&info.pages) {
            let page = Page::parse(bytes).unwrap();
            assert_eq!(page.faces.len() as u32, page_info.face_count);
            let points = page.points(&info.points);
            let value_points = page.value_points(&info.values);
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
                        let uv = page.limit_uv(f, u, v, 0, &value_points).unwrap();
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
                        let color = page.limit_color(f, u, v, &value_points).unwrap();
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
