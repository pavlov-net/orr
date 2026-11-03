//! Subdivision cages: a mounted cage's GPU tables, built once from its pages, and the copies
//! that dice its limit surface as tessellated producer entries.
//!
//! Each patch of the cage's patch table is a quad in its own coordinates (s, t) that becomes
//! two base triangles, or a fan of four to eight around its center where a neighbor patch
//! is one level finer (its transition edges split at their middles). Base vertices at the
//! same point of the cage's topology evaluate their position on one patch, so every base
//! triangle that meets there reads the same bits: the tessellator's edge factors and weld
//! then agree across patches and faces. A copy's fill writes the base triangles into its
//! entry's patch batch every frame, from the rest page points or, for a skinned copy, from
//! cage points posed by its palette and the cage's stencils.

use std::collections::HashMap;

use orr_pack::subd::{CagePoint, Face, Page, SubdInfo};

use crate::material::MaterialId;
use crate::producer::{Opacity, OriginPolicy};
use crate::scene::{InstanceFlags, ModelId, TransformGroupId};
use crate::shaders::types::{SubdCageGpu, SubdPatchGpu, SubdTriangleGpu, SubdVertexGpu};
use crate::{Error, Result};
use orr_gpu::GpuPtr;

/// One copy of a model's subdivision cage, diced as tessellated producer entry `id` of
/// [`crate::SceneWriter::set_subd`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubdDesc<'a> {
    pub model: ModelId,
    /// Group and index of the copy's transform.
    pub transform: (TransformGroupId, u32),
    /// A skinned cage's palette group and first entry, one skinning matrix per skeleton
    /// joint as for skinned meshes; `None` keeps the cage at rest.
    pub palette: Option<(TransformGroupId, u32)>,
    /// One per cage section; empty takes the model's.
    pub materials: &'a [MaterialId],
    /// Generated clusters and vertices the tessellator dices the copy into, at most.
    pub max_clusters: u32,
    pub max_vertices: u32,
    /// The linked `IDisplacement` id the limit surface displaces through; 0 is the
    /// material's displacement texture.
    pub displacement: u8,
    /// The most the displacement moves a point, in m of the copy's space.
    pub displacement_reach: f32,
    pub origin: OriginPolicy,
    pub flags: InstanceFlags,
    /// Read by linked surfaces and hit shading through `SurfaceInput`, never by orr.
    pub user_tag: u32,
}

impl Default for SubdDesc<'_> {
    fn default() -> Self {
        Self {
            model: ModelId(0),
            transform: (TransformGroupId(0), 0),
            palette: None,
            materials: &[],
            max_clusters: 0,
            max_vertices: 0,
            displacement: 0,
            displacement_reach: 0.0,
            origin: OriginPolicy::Default,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            user_tag: 0,
        }
    }
}

/// The sizes of a cage's tables.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CageShape {
    /// Patch clusters and base vertices of its patch batch.
    pub(crate) clusters: u32,
    pub(crate) vertices: u32,
    /// Page points, and cage points.
    pub(crate) points: u32,
    pub(crate) cage_points: u32,
    pub(crate) sections: u32,
    pub(crate) skinned: bool,
    /// The farthest the limit surface lies from its base triangles at rest, m.
    pub(crate) deviation: f32,
}

/// A model's cage as its copies take it.
#[derive(Clone, Debug)]
pub(crate) struct ModelCage {
    pub(crate) cage: GpuPtr<SubdCageGpu>,
    /// Its rest page points.
    pub(crate) points: GpuPtr<[f32; 3]>,
    pub(crate) shape: CageShape,
    /// One per cage section.
    pub(crate) materials: Vec<MaterialId>,
    /// From the cage sections' materials; `None` where they mix dielectric or presence
    /// materials with others.
    pub(crate) opacity: Option<Opacity>,
}

/// What a copy's tessellated entry is built with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SubdBuild {
    pub(crate) cage: GpuPtr<SubdCageGpu>,
    pub(crate) points: GpuPtr<[f32; 3]>,
    pub(crate) shape: CageShape,
    /// Palette group, first entry and joints of a skinned copy.
    pub(crate) palette: Option<(u32, u32, u32)>,
}

/// Face coordinates are numerators over `ONE`, exact for patches up to 15 levels deep.
const ONE: u32 = 1 << 16;

/// A point of the cage's limit surface named by its topology: the same for every ptex face
/// it lies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    At(CagePoint),
    /// On the segment between two topology points (the lower first), at a fraction of
    /// `ONE` from the first.
    On(CagePoint, CagePoint, u32),
    /// Inside ptex face `0`.
    Inside(u32, u32, u32),
}

fn rank(p: CagePoint) -> (u8, u32, u32) {
    match p {
        CagePoint::Point(a) => (0, a, 0),
        CagePoint::EdgeMiddle(a, b) => (1, a, b),
        CagePoint::FaceCenter(f) => (2, f, 0),
    }
}

/// The point `t` from `p` toward `q` along the cage edge between them.
fn on_edge(p: u32, q: u32, t: u32) -> Key {
    let (a, b, t) = if p < q { (p, q, t) } else { (q, p, ONE - t) };
    match t {
        0 => Key::At(CagePoint::Point(a)),
        ONE => Key::At(CagePoint::Point(b)),
        t if t == ONE / 2 => Key::At(CagePoint::EdgeMiddle(a, b)),
        t => Key::On(CagePoint::Point(a), CagePoint::Point(b), t),
    }
}

/// The point `t` from `a` toward `b` along a ptex face edge. A half cage edge (an n-sided
/// face's) names its point on the whole edge, which the face across it shares.
fn on_segment(a: CagePoint, b: CagePoint, t: u32) -> Key {
    let other = |p: u32, x: u32, y: u32| if x == p { y } else { x };
    match (a, b) {
        (CagePoint::Point(p), CagePoint::Point(q)) => on_edge(p, q, t),
        (CagePoint::Point(p), CagePoint::EdgeMiddle(x, y)) => on_edge(p, other(p, x, y), t / 2),
        (CagePoint::EdgeMiddle(x, y), CagePoint::Point(p)) => {
            on_edge(p, other(p, x, y), (ONE - t) / 2)
        }
        _ if rank(a) <= rank(b) => Key::On(a, b, t),
        _ => Key::On(b, a, ONE - t),
    }
}

/// The topology point at face coordinates `(u, v)` (over `ONE`) of ptex face `face`.
fn face_point(info: &SubdInfo, face: &Face, u: u32, v: u32) -> Key {
    let c = info.ptex_corners(face);
    match (u, v) {
        (0, 0) => Key::At(c[0]),
        (ONE, 0) => Key::At(c[1]),
        (ONE, ONE) => Key::At(c[2]),
        (0, ONE) => Key::At(c[3]),
        (_, 0) => on_segment(c[0], c[1], u),
        (ONE, _) => on_segment(c[1], c[2], v),
        (_, ONE) => on_segment(c[2], c[3], ONE - u),
        (0, _) => on_segment(c[3], c[0], ONE - v),
        _ => Key::Inside(face.ptex, u, v),
    }
}

/// Patch corners (0, 0), (1, 0), (1, 1), (0, 1), edge middles and center in halves.
const CORNERS: [[u32; 2]; 4] = [[0, 0], [2, 0], [2, 2], [0, 2]];
const MIDDLES: [[u32; 2]; 4] = [[1, 0], [2, 1], [1, 2], [0, 1]];
const CENTER: [u32; 2] = [1, 1];

/// The base triangles of a patch whose `transition` edges meet a finer neighbor,
/// counter-clockwise in patch coordinates.
fn base_triangles(transition: u8) -> Vec<[[u32; 2]; 3]> {
    if transition & 15 == 0 {
        let c = CORNERS;
        return vec![[c[0], c[1], c[2]], [c[0], c[2], c[3]]];
    }
    let mut out = Vec::new();
    for e in 0..4 {
        let (a, b) = (CORNERS[e], CORNERS[(e + 1) % 4]);
        if transition & (1 << e) != 0 {
            out.push([a, MIDDLES[e], CENTER]);
            out.push([MIDDLES[e], b, CENTER]);
        } else {
            out.push([a, b, CENTER]);
        }
    }
    out
}

fn halves(h: [u32; 2]) -> [f32; 2] {
    h.map(|x| x as f32 * 0.5)
}

/// Patch `patch` of `page` at patch coordinates `st`, from the page's points.
fn limit(page: &Page, patch: usize, points: &[[f32; 3]], st: [f32; 2]) -> [f32; 3] {
    let param = &page.patches[patch].param;
    let n = f32::from(1u16 << param.levels());
    let u = (f32::from(param.u) + st[0]) / n;
    let v = (f32::from(param.v) + st[1]) / n;
    page.patch_limit(patch, u, v, points).position
}

/// A cage's tables before they have an address.
pub(crate) struct CageTables {
    patches: Vec<SubdPatchGpu>,
    cvs: Vec<u32>,
    uv_cvs: Vec<u32>,
    points: Vec<[f32; 3]>,
    uv_points: Vec<[f32; 2]>,
    vertices: Vec<SubdVertexGpu>,
    triangles: Vec<SubdTriangleGpu>,
    clusters: Vec<[u32; 4]>,
    indices: Vec<u8>,
    stencils: Vec<u32>,
    stencil_points: Vec<u32>,
    stencil_weights: Vec<f32>,
    cage_points: Vec<[f32; 3]>,
    skin: Vec<[u32; 2]>,
    pub(crate) shape: CageShape,
}

/// Base triangles per patch cluster, at most; triangles of a patch stay in one cluster.
const CLUSTER_TRIANGLES: usize = 128;
/// Base vertices per patch cluster, at most.
const CLUSTER_VERTICES: usize = 256;

impl CageTables {
    /// The tables of a cage from blob 0 and its pages, in page order.
    pub(crate) fn new(info: &SubdInfo, pages: &[Page]) -> Result<Self> {
        let skinned = !info.skin.is_empty();
        let mut t = Self {
            patches: Vec::new(),
            cvs: Vec::new(),
            uv_cvs: Vec::new(),
            points: Vec::new(),
            uv_points: Vec::new(),
            vertices: Vec::new(),
            triangles: Vec::new(),
            clusters: Vec::new(),
            indices: Vec::new(),
            stencils: Vec::new(),
            stencil_points: Vec::new(),
            stencil_weights: Vec::new(),
            cage_points: if skinned {
                info.points.clone()
            } else {
                Vec::new()
            },
            skin: info
                .skin
                .iter()
                .map(|s| [u32::from_le_bytes(s.joints), u32::from_le_bytes(s.weights)])
                .collect(),
            shape: CageShape {
                cage_points: info.points.len() as u32,
                sections: info.sections.len() as u32,
                skinned,
                ..CageShape::default()
            },
        };
        // Every page's patches, points and stencils into one table.
        // Per face: its page, first patch and first point in the tables.
        let mut faces: Vec<(Face, usize, usize, usize)> = Vec::new();
        for (k, page) in pages.iter().enumerate() {
            let point_base = t.points.len() as u32;
            let uv_base = t.uv_points.len() as u32;
            let patch_base = t.patches.len();
            t.points.extend(page.points(&info.points));
            t.uv_points.extend(page.uv_points(&info.uvs));
            if skinned {
                let term_base = t.stencil_points.len() as u32;
                let offsets = &page.stencils[..page.stencils.len().saturating_sub(1)];
                t.stencils.extend(offsets.iter().map(|&o| o + term_base));
                t.stencil_points.extend(&page.stencil_points);
                t.stencil_weights.extend(&page.stencil_weights);
            }
            for p in &page.patches {
                let (lp, luv) = (p.param.levels(), p.uv_param.levels());
                let scale = 2f32.powi(i32::from(luv) - i32::from(lp));
                let n = p.param.cv_count();
                let n_uv = p.uv_param.cv_count();
                t.patches.push(SubdPatchGpu {
                    bits: u32::from(p.param.kind)
                        | u32::from(p.param.boundary) << 8
                        | u32::from(p.uv_param.kind) << 16
                        | u32::from(p.uv_param.boundary) << 24,
                    cvs: t.cvs.len() as u32,
                    uv_cvs: t.uv_cvs.len() as u32,
                    uv_scale: scale,
                    uv_offset: [
                        f32::from(p.uv_param.u) - f32::from(p.param.u) * scale,
                        f32::from(p.uv_param.v) - f32::from(p.param.v) * scale,
                    ],
                    _pad: [0; 2],
                });
                let cvs = &page.cvs[p.cvs as usize..p.cvs as usize + n];
                t.cvs.extend(cvs.iter().map(|&c| u32::from(c) + point_base));
                let uv_cvs = &page.uv_cvs[p.uv_cvs as usize..p.uv_cvs as usize + n_uv];
                t.uv_cvs
                    .extend(uv_cvs.iter().map(|&c| u32::from(c) + uv_base));
            }
            faces.extend(
                page.faces
                    .iter()
                    .map(|&f| (f, k, patch_base, point_base as usize)),
            );
        }
        if skinned {
            t.stencils.push(t.stencil_points.len() as u32);
        }
        t.shape.points = t.points.len() as u32;

        // Base triangles by patch, patches packed into clusters of one section.
        let mut sources: HashMap<Key, (u32, [f32; 2])> = HashMap::new();
        let mut cluster: Option<([u32; 4], u16)> = None;
        let mut deviation = 0.0f32;
        for &(ref face, page, patch_base, point_base) in &faces {
            let (page, page_points) = (&pages[page], &t.points[point_base..]);
            for local in
                face.patches.first as usize..(face.patches.first + face.patches.count) as usize
            {
                let patch = patch_base + local;
                let param = page.patches[local].param;
                let lp = u32::from(param.levels());
                if lp > 15 {
                    return Err(
                        orr_pack::Error::Malformed("subd patch deeper than 15 levels").into(),
                    );
                }
                let tris = base_triangles(param.transition);
                let mut local_vertices: Vec<[u32; 2]> = Vec::new();
                for tri in &tris {
                    for h in tri {
                        if !local_vertices.contains(h) {
                            local_vertices.push(*h);
                        }
                    }
                }
                let open = cluster.is_some_and(|(c, section)| {
                    section == face.section
                        && c[3] as usize + tris.len() <= CLUSTER_TRIANGLES
                        && c[1] as usize + local_vertices.len() <= CLUSTER_VERTICES
                });
                if !open {
                    if let Some((c, section)) = cluster.take() {
                        t.clusters.push(close(c, section));
                    }
                    cluster = Some((
                        [t.vertices.len() as u32, 0, t.triangles.len() as u32, 0],
                        face.section,
                    ));
                }
                let (c, _) = cluster.as_mut().expect("opened above");
                let first_local = c[1];
                for h in &local_vertices {
                    let u = (2 * u32::from(param.u) + h[0]) << (15 - lp);
                    let v = (2 * u32::from(param.v) + h[1]) << (15 - lp);
                    let key = face_point(info, face, u, v);
                    let own = (patch as u32, halves(*h));
                    let (source, source_st) = *sources.entry(key).or_insert(own);
                    t.vertices.push(SubdVertexGpu {
                        patch: own.0,
                        source,
                        st: own.1,
                        source_st,
                    });
                }
                c[1] += local_vertices.len() as u32;
                for tri in &tris {
                    let mut corners = 0u32;
                    for (k, h) in tri.iter().enumerate() {
                        let i = local_vertices.iter().position(|x| x == h).expect("listed");
                        t.indices.push((first_local as usize + i) as u8);
                        corners |= (h[0] | h[1] << 2) << (4 * k);
                    }
                    t.triangles.push(SubdTriangleGpu {
                        patch: patch as u32,
                        corners,
                    });
                    // How far the limit surface bulges from the flat triangle.
                    let p = tri.map(|h| limit(page, local, page_points, halves(h)));
                    for w in [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                    ] {
                        let sum: f32 = w.iter().sum();
                        let st = std::array::from_fn(|k| {
                            (0..3).map(|j| w[j] * halves(tri[j])[k]).sum::<f32>() / sum
                        });
                        let on = limit(page, local, page_points, st);
                        let flat: [f32; 3] = std::array::from_fn(|k| {
                            (0..3).map(|j| w[j] * p[j][k]).sum::<f32>() / sum
                        });
                        let d = (0..3).map(|k| (on[k] - flat[k]).powi(2)).sum::<f32>();
                        deviation = deviation.max(d.sqrt());
                    }
                }
                c[3] += tris.len() as u32;
            }
        }
        if let Some((c, section)) = cluster {
            t.clusters.push(close(c, section));
        }
        t.shape.clusters = t.clusters.len() as u32;
        t.shape.vertices = t.vertices.len() as u32;
        t.shape.deviation = deviation * 1.25;
        Ok(t)
    }

    fn arrays(&self) -> [&[u8]; 14] {
        use bytemuck::cast_slice as b;
        [
            b(&self.patches),
            b(&self.cvs),
            b(&self.uv_cvs),
            b(&self.points),
            b(&self.uv_points),
            b(&self.vertices),
            b(&self.triangles),
            b(&self.clusters),
            &self.indices,
            b(&self.stencils),
            b(&self.stencil_points),
            b(&self.stencil_weights),
            b(&self.cage_points),
            b(&self.skin),
        ]
    }

    /// Offsets of the record and each array.
    fn layout(&self) -> (Vec<u64>, u64) {
        let mut at = size_of::<SubdCageGpu>() as u64;
        let offsets = self
            .arrays()
            .iter()
            .map(|a| {
                at = at.next_multiple_of(16);
                let o = at;
                at += a.len() as u64;
                o
            })
            .collect();
        (offsets, at.next_multiple_of(16))
    }

    pub(crate) fn len(&self) -> u64 {
        self.layout().1
    }

    /// The rest page points of the tables at `base`.
    pub(crate) fn rest_points(&self, base: GpuPtr<u8>) -> GpuPtr<[f32; 3]> {
        base.offset(self.layout().0[3]).cast()
    }

    /// The table bytes for an allocation at `base`: the [`SubdCageGpu`] record, then its
    /// arrays.
    pub(crate) fn bytes(&self, base: GpuPtr<u8>) -> Vec<u8> {
        let (offsets, len) = self.layout();
        let at = |i: usize| base.offset(offsets[i]);
        let record = SubdCageGpu {
            patches: at(0).cast(),
            cvs: at(1).cast(),
            uv_cvs: at(2).cast(),
            points: at(3).cast(),
            uv_points: at(4).cast(),
            vertices: at(5).cast(),
            triangles: at(6).cast(),
            clusters: at(7).cast(),
            indices: at(8),
            stencils: at(9).cast(),
            stencil_points: at(10).cast(),
            stencil_weights: at(11).cast(),
            cage_points: at(12).cast(),
            skin: at(13).cast(),
            point_count: self.shape.points,
            cage_point_count: self.shape.cage_points,
            vertex_count: self.shape.vertices,
            cluster_count: self.shape.clusters,
        };
        let mut bytes = vec![0u8; len as usize];
        bytes[..size_of::<SubdCageGpu>()].copy_from_slice(bytemuck::bytes_of(&record));
        for (a, &o) in self.arrays().iter().zip(&offsets) {
            bytes[o as usize..o as usize + a.len()].copy_from_slice(a);
        }
        bytes
    }
}

/// A patch cluster's record: first vertex, vertex count, first triangle and triangle count
/// with its section above.
fn close(c: [u32; 4], section: u16) -> [u32; 4] {
    [c[0], c[1], c[2], c[3] | u32::from(section) << 8]
}

/// The opacity copies of a cage take from its sections' material flags.
pub(crate) fn opacity(flags: &[u32]) -> Option<Opacity> {
    use crate::material::material_flags as f;
    match crate::material::candidate_class(flags.iter().copied())? {
        1 => Some(Opacity::Dielectric),
        2 => Some(Opacity::Presence),
        _ if flags.iter().any(|&x| x & f::ALPHA_TESTED != 0) => Some(Opacity::AlphaTested),
        _ => Some(Opacity::Opaque),
    }
}

/// Checks a copy against its model's cage.
pub(crate) fn check(desc: &SubdDesc<'_>, cage: Option<&ModelCage>, joints: u32) -> Result<()> {
    let cage = cage.ok_or(Error::NoCage(desc.model))?;
    assert!(
        desc.materials.is_empty() || desc.materials.len() == cage.shape.sections as usize,
        "{} materials for {} cage sections",
        desc.materials.len(),
        cage.shape.sections
    );
    assert!(
        desc.palette.is_none() || (cage.shape.skinned && joints > 0),
        "{:?} has no skinned cage and skeleton to pose",
        desc.model
    );
    if cage.opacity.is_none() {
        return Err(Error::MixedClasses(desc.model));
    }
    Ok(())
}
