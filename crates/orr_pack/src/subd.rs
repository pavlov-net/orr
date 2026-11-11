//! Subdivision cage class: a Catmull-Clark cage and the patches of its limit surface.
//!
//! Blob 0 holds a [`SubdHeader`] and its tables ([`SubdInfo`]): the cage's control points and
//! faces, the values its uvs and colors read, per-point skin influences, the sections and the
//! pages. The blobs after it are pages of whole faces ([`Page`]), each read alone.
//!
//! Faces are ptex faces: a quad of the cage is one face, an n-sided face is n faces, one per
//! corner. Each face is covered by patches: one bicubic B-spline patch for a regular face,
//! or the feature-adaptive patches of its isolated sub-faces, B-spline where regular and
//! Gregory basis patches at the isolation level. A patch's points are page points; each page
//! point is a stencil, a weighted sum of cage points, so an animated cage deforms by moving its
//! points and re-applying the page stencils. [`Page::limit`] is the reference evaluator.
//!
//! A patch's uvs and colors are [`Part`]s, each a patch of its own over value points, which are
//! stencils of the cage's values: on the position patch for values the surface's basis carries,
//! on a face-varying patch, or bilinear over the patch's corners. Each gives a uv set or some of
//! the vertex color stream's four channels, so values of different interpolations share a face.

use bytemuck::{Pod, Zeroable};
use orr_math::{Affine3, Vec3};

use crate::blob::{put_table, read, table};
use crate::mesh::Section;
use crate::{Aabb, Error, Result, Span};

pub const SCHEMA: u16 = 5;

/// Largest page.
pub const PAGE_MAX: usize = 128 << 10;

/// Blob 0 header. Spans name tables in blob 0: `first` is a byte offset, `count` a record
/// count.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct SubdHeader {
    /// Rest-pose bounds of the limit surface, displaced by every section's declared range.
    pub bounds: Aabb,
    /// Ptex faces and patches over all pages.
    pub face_count: u32,
    pub patch_count: u32,
    /// Isolation levels the cage was refined to: sharp features, smooth irregular vertices.
    pub isolation: u8,
    pub smooth_isolation: u8,
    /// [`Self::BOUNDARY_NONE`], [`Self::BOUNDARY_EDGES`] or [`Self::BOUNDARY_EDGES_AND_CORNERS`].
    pub boundary: u8,
    /// Face-varying linear interpolation, [`Self::UV_LINEAR_NONE`] to [`Self::UV_LINEAR_ALL`].
    pub uv_linear: u8,
    /// [`Self::UV`], [`Self::SKIN`], [`Self::COLOR`] and [`Self::UV1`].
    pub attributes: u8,
    pub _pad: [u8; 3],
    /// [`Section`] per material slot.
    pub sections: Span,
    /// `[f32; 3]` rest position per cage point, model space.
    pub points: Span,
    /// `[f32; 4]` per cage value [`Part`]s read: a color as the vertex color stream holds it, a
    /// uv in its first two, top-left origin.
    pub values: Span,
    /// [`Influence`] per cage point; empty without [`Self::SKIN`].
    pub skin: Span,
    /// [`PageInfo`] per page.
    pub pages: Span,
    /// u32 per cage face plus one: face `f`'s corners are
    /// `cage_corners[cage_faces[f]..cage_faces[f + 1]]`.
    pub cage_faces: Span,
    /// u32 cage point per face corner, counter-clockwise from the face's first corner as the
    /// refiner reads it (a left-handed face's corners after its first are reversed).
    pub cage_corners: Span,
}

impl SubdHeader {
    pub const UV: u8 = 1;
    pub const SKIN: u8 = 2;
    pub const COLOR: u8 = 4;
    /// Its copies carry uv set 1; without it that set reads uv set 0, as other producers' do.
    pub const UV1: u8 = 8;

    /// Boundary edges and corners are smooth.
    pub const BOUNDARY_NONE: u8 = 0;
    /// Boundary edges are sharp.
    pub const BOUNDARY_EDGES: u8 = 1;
    /// Boundary edges are sharp, and corners with two faces or fewer.
    pub const BOUNDARY_EDGES_AND_CORNERS: u8 = 2;

    pub const UV_LINEAR_NONE: u8 = 0;
    pub const UV_LINEAR_CORNERS_ONLY: u8 = 1;
    pub const UV_LINEAR_CORNERS_PLUS1: u8 = 2;
    pub const UV_LINEAR_CORNERS_PLUS2: u8 = 3;
    pub const UV_LINEAR_BOUNDARIES: u8 = 4;
    pub const UV_LINEAR_ALL: u8 = 5;
}

/// A cage point's skin: four skeleton joints and unorm16 weights ([`crate::mesh::unorm16`])
/// that sum to 65535.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Influence {
    pub joints: [u8; 4],
    pub weights: [u16; 4],
}

/// One page: the blob that holds it, its size and what it covers.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct PageInfo {
    /// Blob index within the entry.
    pub blob: u32,
    pub len: u32,
    pub face_count: u32,
    pub patch_count: u32,
    /// The union of its faces' bounds.
    pub bounds: Aabb,
}

/// A blob-0 table set.
#[derive(Clone, Debug, PartialEq)]
pub struct SubdInfo {
    pub header: SubdHeader,
    pub sections: Vec<Section>,
    pub points: Vec<[f32; 3]>,
    pub values: Vec<[f32; 4]>,
    pub skin: Vec<Influence>,
    pub pages: Vec<PageInfo>,
    pub cage_faces: Vec<u32>,
    pub cage_corners: Vec<u32>,
}

/// A point of the cage's topology that ptex faces meet at.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CagePoint {
    /// A cage point.
    Point(u32),
    /// The middle of the cage edge between two points, the lower first.
    EdgeMiddle(u32, u32),
    /// The center of a cage face that is not a quad.
    FaceCenter(u32),
}

impl SubdInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let header: SubdHeader = read(blob, 0).ok_or(Error::Malformed("subd header"))?;
        let info = Self {
            sections: table(blob, header.sections)?,
            points: table(blob, header.points)?,
            values: table(blob, header.values)?,
            skin: table(blob, header.skin)?,
            pages: table(blob, header.pages)?,
            cage_faces: table(blob, header.cage_faces)?,
            cage_corners: table(blob, header.cage_corners)?,
            header,
        };
        let faces_ok = info.cage_faces.windows(2).all(|w| w[0] < w[1])
            && info
                .cage_faces
                .last()
                .is_some_and(|&l| l as usize == info.cage_corners.len());
        if !faces_ok
            || info
                .cage_corners
                .iter()
                .any(|&c| c as usize >= info.points.len())
        {
            return Err(Error::Malformed("subd cage faces"));
        }
        Ok(info)
    }

    /// The cage points of cage face `face`, counter-clockwise.
    fn face_points(&self, face: u32) -> &[u32] {
        let f = face as usize;
        &self.cage_corners[self.cage_faces[f] as usize..self.cage_faces[f + 1] as usize]
    }

    /// The corners of ptex face `face` at face coordinates (0, 0), (1, 0), (1, 1) and
    /// (0, 1). A quad's are its points; the face of corner `k` of an n-sided face runs
    /// from point `k` to the middle of the edge after it, the face center and the middle
    /// of the edge before it.
    pub fn ptex_corners(&self, face: &Face) -> [CagePoint; 4] {
        let p = self.face_points(face.base);
        let edge = |a: u32, b: u32| CagePoint::EdgeMiddle(a.min(b), a.max(b));
        if p.len() == 4 {
            return [0, 1, 2, 3].map(|i| CagePoint::Point(p[i]));
        }
        let n = p.len();
        let k = usize::from(face.corner);
        let (here, next, before) = (p[k], p[(k + 1) % n], p[(k + n - 1) % n]);
        [
            CagePoint::Point(here),
            edge(here, next),
            CagePoint::FaceCenter(face.base),
            edge(before, here),
        ]
    }

    /// The cage points posed by a skinning palette (`model[j] x inverse_binds[j]` per joint).
    pub fn skinned_points(&self, palette: &[Affine3]) -> Vec<[f32; 3]> {
        self.points
            .iter()
            .zip(&self.skin)
            .map(|(&p, s)| {
                let p = Vec3::from(p);
                let mut out = Vec3::ZERO;
                for (&j, &w) in s.joints.iter().zip(&s.weights) {
                    let q = palette[usize::from(j)].transform_point3(p);
                    out += crate::mesh::unorm16(w) * q;
                }
                out.to_array()
            })
            .collect()
    }

    /// Blob bytes; the header's spans are set from the tables.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = self.header;
        let mut out = vec![0; size_of::<SubdHeader>()];
        h.sections = put_table(&mut out, &self.sections);
        h.points = put_table(&mut out, &self.points);
        h.values = put_table(&mut out, &self.values);
        h.skin = put_table(&mut out, &self.skin);
        h.pages = put_table(&mut out, &self.pages);
        h.cage_faces = put_table(&mut out, &self.cage_faces);
        h.cage_corners = put_table(&mut out, &self.cage_corners);
        out[..size_of::<SubdHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }
}

/// Page header. Spans name tables in the page: `first` is a byte offset, `count` a record
/// count.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct PageHeader {
    /// [`Face`] per face.
    pub faces: Span,
    /// [`Patch`] per patch; a face's patches are contiguous.
    pub patches: Span,
    /// u16 page point per patch point.
    pub cvs: Span,
    /// u32 per page point plus one: point `i`'s stencil terms are
    /// `stencils[i]..stencils[i + 1]`.
    pub stencils: Span,
    /// u32 cage point per stencil term.
    pub stencil_points: Span,
    /// f32 weight per stencil term.
    pub stencil_weights: Span,
    /// [`Part`] per part; a patch's parts are contiguous.
    pub parts: Span,
    /// The same four tables for parts, over the cage's values; empty without any.
    pub part_cvs: Span,
    pub value_stencils: Span,
    pub value_stencil_points: Span,
    pub value_stencil_weights: Span,
}

/// One ptex face of the cage.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Face {
    /// Rest-pose bounds of its limit surface displaced by its section's declared range.
    pub bounds: Aabb,
    /// Ptex face index: cage faces in order, an n-sided face taking n.
    pub ptex: u32,
    /// Cage face.
    pub base: u32,
    pub section: u16,
    /// For a face of an n-sided cage face, the corner it starts at; 0 for a quad.
    pub corner: u8,
    pub _pad: u8,
    /// Range of the page's patches.
    pub patches: Span,
}

/// Where a patch sits in its face and how its points combine.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct PatchParam {
    /// Position of the patch in its face's grid of `2^depth` (`2^(depth - 1)` for the root
    /// of an n-sided face's sub-face) patches per side.
    pub u: u16,
    pub v: u16,
    pub depth: u8,
    /// Edges 0..3 (bits 0..3: v = 0, u = 1, v = 1, u = 0) beyond which a B-spline patch
    /// extrapolates its missing points.
    pub boundary: u8,
    /// Edges whose neighbor is one level finer, for crack-free tessellation.
    pub transition: u8,
    /// [`Self::REGULAR`], [`Self::GREGORY`] or [`Self::LINEAR`].
    pub kind: u8,
    /// [`Self::NON_QUAD`].
    pub flags: u8,
    pub _pad: [u8; 3],
}

impl PatchParam {
    /// Bicubic B-spline, 16 points.
    pub const REGULAR: u8 = 0;
    /// Gregory basis, 20 points.
    pub const GREGORY: u8 = 1;
    /// Bilinear, 4 points.
    pub const LINEAR: u8 = 2;
    /// The patch belongs to a sub-face of an n-sided cage face.
    pub const NON_QUAD: u8 = 1;

    pub fn cv_count(&self) -> usize {
        match self.kind {
            Self::REGULAR => 16,
            Self::GREGORY => 20,
            Self::LINEAR => 4,
            _ => 0,
        }
    }

    /// Subdivision levels of the face's grid of patches: `2^levels` patches per side.
    pub fn levels(&self) -> u8 {
        self.depth - u8::from(self.flags & Self::NON_QUAD != 0)
    }

    /// Whether face coordinates `(u, v)` lie in the patch.
    fn contains(&self, u: f32, v: f32) -> bool {
        let n = f32::from(1u16 << self.levels());
        let (s, t) = (u * n - f32::from(self.u), v * n - f32::from(self.v));
        (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t)
    }

    /// Weights of the patch's points and of their derivatives in face coordinates at face
    /// coordinates `(u, v)`; returns the point count.
    fn basis(
        &self,
        u: f32,
        v: f32,
        w: &mut [f32; 20],
        du: &mut [f32; 20],
        dv: &mut [f32; 20],
    ) -> usize {
        let n = f32::from(1u16 << self.levels());
        let s = u * n - f32::from(self.u);
        let t = v * n - f32::from(self.v);
        let count = match self.kind {
            Self::REGULAR => {
                bspline(s, t, w, du, dv);
                if self.boundary != 0 {
                    for x in [&mut *w, &mut *du, &mut *dv] {
                        bspline_boundary(self.boundary, x);
                    }
                }
                16
            }
            Self::GREGORY => {
                gregory(s, t, w, du, dv);
                20
            }
            Self::LINEAR => {
                linear(s, t, w, du, dv);
                4
            }
            _ => 0,
        };
        let scale = f32::from(1u16 << self.depth);
        for i in 0..count {
            du[i] *= scale;
            dv[i] *= scale;
        }
        count
    }
}

/// One patch: its parameters and first point for positions, and its parts.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Patch {
    pub param: PatchParam,
    /// First entry of the page's `cvs` table.
    pub cvs: u32,
    /// Range of the page's `parts`. Color channels none of them gives read 1, and a uv set none
    /// of them gives reads 0.
    pub parts: Span,
}

/// A uv set or some channels of the vertex color stream over one patch.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Part {
    /// Where the part's own patch sits in the face.
    pub param: PatchParam,
    /// First entry of the page's `part_cvs` table.
    pub cvs: u32,
    /// What it gives: bit `i` below [`Self::UV`] for color channel `i`, [`Self::UV`] for uv set 0
    /// and the next bit for set 1. A later part of a patch takes either from an earlier one.
    pub channels: u32,
}

impl Part {
    /// [`Self::channels`] of uv set 0.
    pub const UV: u32 = 16;
}

/// A limit surface sample: position and derivatives in face coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Limit {
    pub position: [f32; 3],
    pub du: [f32; 3],
    pub dv: [f32; 3],
}

impl Limit {
    /// The unit normal, oriented by the face winding; zero where the derivatives degenerate.
    pub fn normal(&self) -> [f32; 3] {
        Vec3::from(self.du)
            .cross(Vec3::from(self.dv))
            .normalize_or_zero()
            .to_array()
    }
}

/// A parsed page.
#[derive(Clone, Debug, PartialEq)]
pub struct Page {
    pub faces: Vec<Face>,
    pub patches: Vec<Patch>,
    pub cvs: Vec<u16>,
    pub stencils: Vec<u32>,
    pub stencil_points: Vec<u32>,
    pub stencil_weights: Vec<f32>,
    pub parts: Vec<Part>,
    pub part_cvs: Vec<u16>,
    pub value_stencils: Vec<u32>,
    pub value_stencil_points: Vec<u32>,
    pub value_stencil_weights: Vec<f32>,
}

impl Page {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let h: PageHeader = read(blob, 0).ok_or(Error::Malformed("subd page header"))?;
        let page = Self {
            faces: table(blob, h.faces)?,
            patches: table(blob, h.patches)?,
            cvs: table(blob, h.cvs)?,
            stencils: table(blob, h.stencils)?,
            stencil_points: table(blob, h.stencil_points)?,
            stencil_weights: table(blob, h.stencil_weights)?,
            parts: table(blob, h.parts)?,
            part_cvs: table(blob, h.part_cvs)?,
            value_stencils: table(blob, h.value_stencils)?,
            value_stencil_points: table(blob, h.value_stencil_points)?,
            value_stencil_weights: table(blob, h.value_stencil_weights)?,
        };
        page.check()?;
        Ok(page)
    }

    fn check(&self) -> Result<()> {
        let bad = Error::Malformed("subd page tables disagree");
        let stencils_ok = |s: &[u32], terms: usize| {
            s.windows(2).all(|w| w[0] <= w[1]) && s.last().is_none_or(|&l| l as usize == terms)
        };
        if !stencils_ok(&self.stencils, self.stencil_points.len())
            || !stencils_ok(&self.value_stencils, self.value_stencil_points.len())
            || self.stencil_points.len() != self.stencil_weights.len()
            || self.value_stencil_points.len() != self.value_stencil_weights.len()
        {
            return Err(bad);
        }
        let points = self.stencils.len().saturating_sub(1);
        let value_points = self.value_stencils.len().saturating_sub(1);
        let cvs = |at: u32, n: usize, table: &[u16], limit: usize| {
            table
                .get(at as usize..at as usize + n)
                .is_some_and(|c| c.iter().all(|&i| usize::from(i) < limit))
        };
        for p in &self.patches {
            if !cvs(p.cvs, p.param.cv_count(), &self.cvs, points)
                || p.parts.first as usize + p.parts.count as usize > self.parts.len()
            {
                return Err(bad);
            }
        }
        for c in &self.parts {
            if !cvs(c.cvs, c.param.cv_count(), &self.part_cvs, value_points) {
                return Err(bad);
            }
        }
        if self
            .faces
            .iter()
            .any(|f| f.patches.first as usize + f.patches.count as usize > self.patches.len())
        {
            return Err(bad);
        }
        Ok(())
    }

    /// Blob bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![0; size_of::<PageHeader>()];
        let h = PageHeader {
            faces: put_table(&mut out, &self.faces),
            patches: put_table(&mut out, &self.patches),
            cvs: put_table(&mut out, &self.cvs),
            stencils: put_table(&mut out, &self.stencils),
            stencil_points: put_table(&mut out, &self.stencil_points),
            stencil_weights: put_table(&mut out, &self.stencil_weights),
            parts: put_table(&mut out, &self.parts),
            part_cvs: put_table(&mut out, &self.part_cvs),
            value_stencils: put_table(&mut out, &self.value_stencils),
            value_stencil_points: put_table(&mut out, &self.value_stencil_points),
            value_stencil_weights: put_table(&mut out, &self.value_stencil_weights),
        };
        out[..size_of::<PageHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }

    /// The page points from the cage points.
    pub fn points(&self, cage: &[[f32; 3]]) -> Vec<[f32; 3]> {
        apply_stencils(
            &self.stencils,
            &self.stencil_points,
            &self.stencil_weights,
            cage,
        )
    }

    /// The page value points from the cage's values.
    pub fn value_points(&self, values: &[[f32; 4]]) -> Vec<[f32; 4]> {
        apply_stencils(
            &self.value_stencils,
            &self.value_stencil_points,
            &self.value_stencil_weights,
            values,
        )
    }

    /// The patch of face `face` (an index into [`Self::faces`]) that holds face coordinates
    /// `(u, v)` in `[0, 1]`.
    fn patch_at(&self, face: usize, u: f32, v: f32) -> Option<usize> {
        let s = self.faces.get(face)?.patches;
        (s.first as usize..(s.first + s.count) as usize)
            .find(|&p| self.patches[p].param.contains(u, v))
    }

    /// The limit surface of face `face` at face coordinates `(u, v)`, from [`Self::points`].
    pub fn limit(&self, face: usize, u: f32, v: f32, points: &[[f32; 3]]) -> Option<Limit> {
        Some(self.patch_limit(self.patch_at(face, u, v)?, u, v, points))
    }

    /// The limit surface of patch `patch` at face coordinates `(u, v)` of its face, from
    /// [`Self::points`].
    pub fn patch_limit(&self, patch: usize, u: f32, v: f32, points: &[[f32; 3]]) -> Limit {
        let p = &self.patches[patch];
        let (mut w, mut du, mut dv) = ([0.0; 20], [0.0; 20], [0.0; 20]);
        let n = p.param.basis(u, v, &mut w, &mut du, &mut dv);
        let cvs = &self.cvs[p.cvs as usize..p.cvs as usize + n];
        let sum = |w: &[f32; 20]| {
            let mut acc = [0.0f64; 3];
            for (i, &c) in cvs.iter().enumerate() {
                let q = points[usize::from(c)];
                for k in 0..3 {
                    acc[k] += f64::from(w[i]) * f64::from(q[k]);
                }
            }
            acc.map(|a| a as f32)
        };
        Limit {
            position: sum(&w),
            du: sum(&du),
            dv: sum(&dv),
        }
    }

    /// The parts of the patch of face `face` that holds face coordinates `(u, v)`.
    fn parts_at(&self, face: usize, u: f32, v: f32) -> Option<&[Part]> {
        let s = self.patches[self.patch_at(face, u, v)?].parts;
        Some(&self.parts[s.first as usize..(s.first + s.count) as usize])
    }

    /// Part `part` at face coordinates `(u, v)`, from [`Self::value_points`].
    fn part_value(&self, part: &Part, u: f32, v: f32, points: &[[f32; 4]]) -> [f32; 4] {
        let (mut w, mut du, mut dv) = ([0.0; 20], [0.0; 20], [0.0; 20]);
        let n = part.param.basis(u, v, &mut w, &mut du, &mut dv);
        let mut acc = [0.0f64; 4];
        let cvs = &self.part_cvs[part.cvs as usize..part.cvs as usize + n];
        for (i, &c) in cvs.iter().enumerate() {
            let q = points[usize::from(c)];
            for k in 0..4 {
                acc[k] += f64::from(w[i]) * f64::from(q[k]);
            }
        }
        acc.map(|a| a as f32)
    }

    /// The limit uv of set `set` of face `face` at face coordinates `(u, v)`, from
    /// [`Self::value_points`].
    pub fn limit_uv(
        &self,
        face: usize,
        u: f32,
        v: f32,
        set: u32,
        points: &[[f32; 4]],
    ) -> Option<[f32; 2]> {
        let parts = self.parts_at(face, u, v)?;
        let last = parts.iter().rfind(|p| p.channels & (Part::UV << set) != 0);
        Some(last.map_or([0.0; 2], |part| {
            let [x, y, ..] = self.part_value(part, u, v, points);
            [x, y]
        }))
    }

    /// The vertex color stream of face `face` at face coordinates `(u, v)`, from
    /// [`Self::value_points`].
    pub fn limit_color(
        &self,
        face: usize,
        u: f32,
        v: f32,
        points: &[[f32; 4]],
    ) -> Option<[f32; 4]> {
        let mut color = [1.0; 4];
        for part in self.parts_at(face, u, v)? {
            let value = self.part_value(part, u, v, points);
            for k in (0..4).filter(|k| part.channels & (1 << k) != 0) {
                color[k] = value[k];
            }
        }
        Some(color)
    }
}

/// Points from stencils over `source`: point `i` is the sum of `weights[t] x source[points[t]]`
/// for `t` in `offsets[i]..offsets[i + 1]`, accumulated in f64.
pub fn apply_stencils<const N: usize>(
    offsets: &[u32],
    points: &[u32],
    weights: &[f32],
    source: &[[f32; N]],
) -> Vec<[f32; N]> {
    offsets
        .windows(2)
        .map(|r| {
            let mut acc = [0.0f64; N];
            for t in r[0] as usize..r[1] as usize {
                let q = source[points[t] as usize];
                for k in 0..N {
                    acc[k] += f64::from(weights[t]) * f64::from(q[k]);
                }
            }
            acc.map(|a| a as f32)
        })
        .collect()
}

/// Uniform cubic B-spline basis at `t` and its derivative.
fn bspline_curve(t: f32) -> ([f32; 4], [f32; 4]) {
    let (t2, t3) = (t * t, t * t * t);
    (
        [
            (1.0 - 3.0 * (t - t2) - t3) / 6.0,
            (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
            (1.0 + 3.0 * (t + t2 - t3)) / 6.0,
            t3 / 6.0,
        ],
        [
            -0.5 * t2 + t - 0.5,
            1.5 * t2 - 2.0 * t,
            -1.5 * t2 + t + 0.5,
            0.5 * t2,
        ],
    )
}

/// Bicubic B-spline weights, points row-major from `(0, 0)` with `s` along a row.
fn bspline(s: f32, t: f32, w: &mut [f32; 20], du: &mut [f32; 20], dv: &mut [f32; 20]) {
    let (bs, ds) = bspline_curve(s);
    let (bt, dt) = bspline_curve(t);
    for i in 0..4 {
        for j in 0..4 {
            w[4 * i + j] = bs[j] * bt[i];
            du[4 * i + j] = ds[j] * bt[i];
            dv[4 * i + j] = bs[j] * dt[i];
        }
    }
}

/// Moves the weights of points beyond boundary edges onto the points that extrapolate them:
/// a missing point is `2 x next - next but one`.
fn bspline_boundary(boundary: u8, w: &mut [f32; 20]) {
    if boundary & 1 != 0 {
        for i in 0..4 {
            w[i + 8] -= w[i];
            w[i + 4] += 2.0 * w[i];
            w[i] = 0.0;
        }
    }
    if boundary & 2 != 0 {
        for i in (0..16).step_by(4) {
            w[i + 1] -= w[i + 3];
            w[i + 2] += 2.0 * w[i + 3];
            w[i + 3] = 0.0;
        }
    }
    if boundary & 4 != 0 {
        for i in 0..4 {
            w[i + 4] -= w[i + 12];
            w[i + 8] += 2.0 * w[i + 12];
            w[i + 12] = 0.0;
        }
    }
    if boundary & 8 != 0 {
        for i in (0..16).step_by(4) {
            w[i + 2] -= w[i];
            w[i + 1] += 2.0 * w[i];
            w[i] = 0.0;
        }
    }
}

fn bezier_curve(t: f32) -> ([f32; 4], [f32; 4]) {
    let c = 1.0 - t;
    (
        [c * c * c, 3.0 * c * c * t, 3.0 * t * t * c, t * t * t],
        [
            -3.0 * c * c,
            9.0 * t * t - 12.0 * t + 3.0,
            -9.0 * t * t + 6.0 * t,
            3.0 * t * t,
        ],
    )
}

/// Gregory basis weights. The 20 points group by corner (corner, edge points, face points);
/// the face-point pairs blend rationally into the four interior Bezier points. Derivatives
/// are those of the Bezier patch at `(s, t)`, as OpenSubdiv's.
fn gregory(s: f32, t: f32, w: &mut [f32; 20], du: &mut [f32; 20], dv: &mut [f32; 20]) {
    const BOUNDARY: [usize; 12] = [0, 1, 7, 5, 2, 6, 16, 12, 15, 17, 11, 10];
    const BOUNDARY_COL: [usize; 12] = [0, 1, 2, 3, 0, 3, 0, 3, 0, 1, 2, 3];
    const BOUNDARY_ROW: [usize; 12] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3];
    const INTERIOR: [usize; 8] = [3, 4, 8, 9, 13, 14, 18, 19];
    const INTERIOR_COL: [usize; 8] = [1, 1, 2, 2, 2, 2, 1, 1];
    const INTERIOR_ROW: [usize; 8] = [1, 1, 1, 1, 2, 2, 2, 2];
    let (bs, ds) = bezier_curve(s);
    let (bt, dt) = bezier_curve(t);
    let (sc, tc) = (1.0 - s, 1.0 - t);
    let inv = |d: f32| if d <= 0.0 { 1.0 } else { 1.0 / d };
    let (d0, d1, d2, d3) = (inv(s + t), inv(sc + t), inv(sc + tc), inv(s + tc));
    let g = [
        s * d0,
        1.0 - s * d0,
        t * d1,
        1.0 - t * d1,
        sc * d2,
        1.0 - sc * d2,
        tc * d3,
        1.0 - tc * d3,
    ];
    for i in 0..12 {
        let (c, r) = (BOUNDARY_COL[i], BOUNDARY_ROW[i]);
        w[BOUNDARY[i]] = bs[c] * bt[r];
        du[BOUNDARY[i]] = ds[c] * bt[r];
        dv[BOUNDARY[i]] = bs[c] * dt[r];
    }
    for i in 0..8 {
        let (c, r) = (INTERIOR_COL[i], INTERIOR_ROW[i]);
        w[INTERIOR[i]] = bs[c] * bt[r] * g[i];
        du[INTERIOR[i]] = ds[c] * bt[r] * g[i];
        dv[INTERIOR[i]] = bs[c] * dt[r] * g[i];
    }
}

/// Bilinear weights, points counter-clockwise from `(0, 0)`.
fn linear(s: f32, t: f32, w: &mut [f32; 20], du: &mut [f32; 20], dv: &mut [f32; 20]) {
    let (sc, tc) = (1.0 - s, 1.0 - t);
    w[..4].copy_from_slice(&[sc * tc, s * tc, s * t, sc * t]);
    du[..4].copy_from_slice(&[-tc, tc, t, -t]);
    dv[..4].copy_from_slice(&[-sc, -s, s, sc]);
}
