//! Mesh cook: welding, rigid parts, discrete levels, grid quantisation, spatial clusters and
//! the stored pages.
//!
//! Per (part, section) the cook welds, reindexes and spatially sorts the triangles. Level 0
//! is the source; level `i` simplifies level `i - 1` to an absolute error of
//! `errors[i - 1]` (permissive simplification that keeps uv seams, with part and section
//! borders locked), falling back to sloppy simplification when that keeps more than three
//! quarters of its input. From `merge_level` on, the parts of a section simplify together
//! into the body (part 0, [`Cluster::MERGED`]). Each piece of each level is cut into spatial
//! clusters; alpha-tested clusters put opaque triangles first and carry opacity micromaps
//! over the rest. Pages hold whole clusters of one level, coarsest level first, and the
//! coarsest level's pages are pinned.

mod lod;
pub(crate) mod maps;
mod omm;

use std::ops::Range;

use glam::{Vec2, Vec3};
use meshopt::VertexDataAdapter;
use orr_pack::material::Material;
use orr_pack::mesh::{
    Cluster, Level, MeshHeader, MeshInfo, OMM_ALIGN, PAGE_MAX, PageHeader, PageInfo, Part,
    PartLevel, Section, oct_encode,
};
use orr_pack::{Aabb, Hash32, Span, Sphere};
use rayon::prelude::*;
use serde::Deserialize;

use crate::texture::Wrap;
use crate::{Error, Result};
use maps::{Displacement, Opacity};
pub use omm::OmmSettings;
use omm::{ClusterOmm, OmmAlpha};

/// How meshes are simplified, clustered and quantised.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MeshSettings {
    pub lod: LodMode,
    /// Cluster vertex limit on solid sections.
    pub max_vertices: u16,
    /// Cluster vertex limit on alpha-tested or double-sided sections.
    pub max_vertices_foliage: u16,
    /// Cluster triangle limit, a multiple of 16 up to 128; clusters aim for at least a quarter.
    pub max_triangles: u16,
    /// Finest grid step, log2 of metres. A mesh too large to span 16 bits per axis at this
    /// step gets the smallest coarser step that does.
    pub position_step_log2: i8,
    /// Metres a skinned vertex may move from its rest position, for every cluster of a
    /// skinned mesh outside animated parts (the bound GPU skinning builds cluster templates
    /// with); ignored without a skin.
    pub deformation_bound: f32,
    /// First discrete level whose parts merge into the body; `None` keeps parts at every
    /// level. Ignored on meshes without parts.
    pub merge_level: Option<u8>,
    /// Opacity micromaps on alpha-tested clusters; `None` for content whose coverage
    /// changes at run time, which never uses them.
    pub omm: Option<OmmSettings>,
}

impl Default for MeshSettings {
    fn default() -> Self {
        Self {
            lod: LodMode::default(),
            max_vertices: 128,
            max_vertices_foliage: 256,
            max_triangles: 128,
            position_step_log2: -11,
            deformation_bound: 0.0,
            merge_level: None,
            omm: Some(OmmSettings::default()),
        }
    }
}

/// Level of detail representation.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum LodMode {
    /// Discrete levels: level 0 is the source and level `i` simplifies level `i - 1` to an
    /// absolute error of `errors[i - 1]` metres. Empty gives one level.
    Discrete { errors: Vec<f32> },
}

impl Default for LodMode {
    fn default() -> Self {
        Self::Discrete { errors: Vec::new() }
    }
}

/// A mesh built by a library caller ([`crate::Source::Mesh`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshInput {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// xyz and bitangent sign.
    pub tangents: Option<Vec<glam::Vec4>>,
    pub uvs: Option<Vec<Vec2>>,
    pub indices: Vec<u32>,
    /// Index ranges, one per material.
    pub sections: Vec<Span>,
    pub parts: Option<MeshParts>,
}

/// Rigid parts of a [`MeshInput`], as `parts_from_attribute` reads them from glTF.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshParts {
    /// Per vertex: the part value.
    pub values: Vec<u16>,
    /// Declared value count; every value is below it.
    pub count: u16,
    /// One part per (value, section): part = value x sections + section.
    pub by_section: bool,
    /// Per vertex: the pivot of its part, alike for every vertex of a part within one grid
    /// step. Without it pivots are the model origin.
    pub pivots: Option<Vec<Vec3>>,
}

/// A mesh ready to cook: one vertex array, index ranges per section.
pub(crate) struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Option<Vec<[f32; 4]>>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub sections: Vec<SectionData>,
    pub skin: Option<Skin>,
    pub parts: Option<PartData>,
    /// The grid step may not be finer than `2^min_exponent` m (models joined crack-free).
    pub min_exponent: Option<i8>,
}

/// Per vertex: four skeleton joints and their unorm8 weights, summing to 255.
pub(crate) struct Skin {
    pub joints: Vec<[u8; 4]>,
    pub weights: Vec<[u8; 4]>,
}

/// Rigid parts from a per-vertex value.
pub(crate) struct PartData {
    pub values: Vec<u16>,
    pub count: u16,
    pub by_section: bool,
    pub pivots: Pivots,
    /// `Part::joint` per value; the value itself without it.
    pub joints: Option<Vec<u16>>,
}

pub(crate) enum Pivots {
    PerValue(Vec<[f32; 3]>),
    PerVertex(Vec<[f32; 3]>),
    Origin,
}

pub(crate) struct SectionData {
    pub indices: Range<usize>,
    /// [`Section`] flags.
    pub flags: u8,
    /// The material scatters below the surface (not thin-walled, no transmission).
    pub subsurface: bool,
    pub maps: MapSpec,
}

impl SectionData {
    /// A section of `indices` drawn with material `m`.
    pub(crate) fn new(indices: Range<usize>, m: &Material, maps: MapSpec) -> Self {
        let mut flags = 0;
        if m.alpha_mode == Material::MASK {
            flags |= Section::ALPHA_TESTED;
        }
        if m.double_sided != 0 {
            flags |= Section::DOUBLE_SIDED;
        }
        if m.emits() {
            flags |= Section::EMISSIVE;
        }
        Self {
            indices,
            flags,
            subsurface: m.subsurface_weight > 0.0
                && m.thin_walled == 0
                && m.transmission_weight == 0.0,
            maps,
        }
    }
}

/// The cooked textures a section's clusters depend on.
#[derive(Clone, Debug, Default)]
pub(crate) struct MapSpec {
    pub opacity: OpacitySpec,
    pub displacement: Option<DisplacementSpec>,
}

#[derive(Clone, Debug, Default)]
pub(crate) enum OpacitySpec {
    /// Not alpha-tested.
    #[default]
    None,
    Constant(bool),
    /// The cooked opacity texture of this unit.
    Texture {
        unit: String,
        key: Hash32,
        cutoff: f32,
        wrap: [Wrap; 2],
    },
    Unknown,
}

#[derive(Clone, Debug)]
pub(crate) struct DisplacementSpec {
    pub scale: f32,
    pub offset: f32,
    pub min: f32,
    pub max: f32,
    /// The cooked displacement texture's unit, key and edge modes.
    pub texture: Option<(String, Hash32, [Wrap; 2])>,
}

impl MapSpec {
    /// The texture units the spec reads.
    pub(crate) fn units(&self) -> impl Iterator<Item = &str> {
        let opacity = match &self.opacity {
            OpacitySpec::Texture { unit, .. } => Some(unit.as_str()),
            _ => None,
        };
        let displacement = self
            .displacement
            .as_ref()
            .and_then(|d| d.texture.as_ref().map(|t| t.0.as_str()));
        opacity.into_iter().chain(displacement)
    }

    /// Everything of the spec that shapes the cooked mesh: values and texture keys.
    fn fingerprint(&self) -> String {
        let opacity = match &self.opacity {
            OpacitySpec::Texture {
                key, cutoff, wrap, ..
            } => format!("texture {key:?} {cutoff:?} {wrap:?}"),
            o => format!("{o:?}"),
        };
        let displacement = self.displacement.as_ref().map(|d| {
            (
                [d.scale, d.offset, d.min, d.max],
                d.texture.as_ref().map(|t| (t.1, t.2)),
            )
        });
        format!("{opacity} {displacement:?}")
    }
}

/// What a section's cluster build reads, resolved from its [`MapSpec`].
#[derive(Default)]
pub(crate) struct SectionMaps {
    pub opacity: Option<Opacity>,
    pub displacement: Option<Displacement>,
}

pub(crate) struct CookedMesh {
    pub info: MeshInfo,
    /// Stored page blobs, in [`MeshInfo::pages`] order.
    pub pages: Vec<Vec<u8>>,
    pub warnings: Vec<String>,
}

pub(crate) fn cook(
    asset: &str,
    m: &MeshData,
    s: &MeshSettings,
    maps: &[SectionMaps],
) -> Result<CookedMesh> {
    let fail = |message: String| Error::Mesh {
        asset: asset.into(),
        message,
    };
    check(m, s).map_err(fail)?;
    let mut warnings = Vec::new();
    let w = lod::Welded::new(m);
    let grid = Grid::new(&w.positions, s.position_step_log2, m.min_exponent).map_err(fail)?;
    let parts = lod::Parts::new(&w, m, &grid).map_err(fail)?;
    for (si, sec) in m.sections.iter().enumerate() {
        if let Some(d) = &maps[si].displacement {
            d.check_range()
                .map_err(|e| fail(format!("section {si}: {e}")))?;
            warnings.extend(lod::displacement_seams(&w, &grid, si, d));
        }
        if sec.subsurface {
            let n = lod::border_edges(&w, &grid, si);
            if n > 0 {
                warnings.push(format!(
                    "section {si}: subsurface material on {n} border edges; the subsurface walk leaks through them"
                ));
            }
        }
    }
    let mut omm_alpha = Vec::with_capacity(maps.len());
    for (si, sm) in maps.iter().enumerate() {
        let alpha = match (&s.omm, &sm.opacity) {
            (Some(_), Some(Opacity::Texture { mip0, cutoff })) => {
                let a = OmmAlpha::new(mip0, *cutoff);
                if a.is_none() {
                    warnings.push(format!(
                        "section {si}: opacity texture {}x{} is over {} texels a side; no micromaps",
                        mip0.w,
                        mip0.h,
                        OmmAlpha::MAX_SIZE
                    ));
                }
                a
            }
            _ => None,
        };
        omm_alpha.push(alpha);
    }
    let levels = lod::levels(&w, &grid, &parts, s, m.sections.len());

    let attributes = if m.tangents.is_some() {
        MeshHeader::TANGENT
    } else {
        0
    } | if m.skin.is_some() {
        MeshHeader::SKIN
    } else {
        0
    };
    let deformation_bound = if m.skin.is_some() {
        f16_up(s.deformation_bound)
    } else {
        0
    };
    let stride = orr_pack::mesh::attr_stride(attributes) as usize;
    let adapter = VertexDataAdapter::new(bytemuck::cast_slice(&grid.floats), 12, 0)
        .expect("positions are 12-byte vertices");

    // Clusters per level, part-major within a level so each part's clusters are contiguous.
    let level_count = levels.len();
    let part_count = parts.table.len();
    let mut max_vertices = 0;
    let mut by_level: Vec<Vec<ClusterData>> = Vec::with_capacity(level_count);
    for (li, level) in levels.iter().enumerate() {
        let parent_error = levels.get(li + 1).map_or(f32::INFINITY, |l| l.error);
        let mut clusters = Vec::new();
        for piece in &level.pieces {
            let sec = &m.sections[usize::from(piece.section)];
            let foliage = sec.flags & (Section::ALPHA_TESTED | Section::DOUBLE_SIDED) != 0;
            let maxv = if foliage {
                s.max_vertices_foliage
            } else {
                s.max_vertices
            };
            max_vertices = max_vertices.max(maxv);
            let maxt = usize::from(s.max_triangles);
            let mut ml = meshopt::build_meshlets_spatial(
                &piece.indices,
                &adapter,
                usize::from(maxv),
                maxt / 4,
                maxt,
                0.5,
            );
            let section_maps = &maps[usize::from(piece.section)];
            let animated = parts.animated[usize::from(piece.part)] && !piece.merged;
            for i in 0..ml.len() {
                let r = ml.meshlets[i];
                let (vo, vc) = (r.vertex_offset as usize, r.vertex_count as usize);
                let (to, tc) = (r.triangle_offset as usize, r.triangle_count as usize);
                meshopt::optimize_meshlet_level(
                    &mut ml.vertices[vo..vo + vc],
                    &mut ml.triangles[to..to + tc * 3],
                    3,
                );
                let verts = &ml.vertices[vo..vo + vc];
                let tris = &mut ml.triangles[to..to + tc * 3];
                let uv = ClusterUvs::new(&w, verts);
                let uvs: Vec<[f32; 2]> = verts
                    .iter()
                    .map(|&v| uv.stored(w.uvs[v as usize]))
                    .collect();
                let opaque = partition_opaque(&uvs, tris, section_maps.opacity.as_ref());
                stored_corner_order(tris);
                let meshlet = meshopt::Meshlet {
                    vertices: verts,
                    triangles: tris,
                };
                let bounds = meshopt::compute_meshlet_bounds(meshlet, &adapter);
                let mut c = build_cluster(&w, &grid, verts, tris, &bounds, sec.flags, &uv);
                if s.omm.is_some() && section_maps.opacity.is_some() {
                    c.uvs = uvs;
                }
                c.record.section = piece.section;
                c.record.part = piece.part;
                c.record.alpha_first = opaque;
                c.record.self_error = level.error;
                c.record.parent_error = parent_error;
                if piece.merged {
                    c.record.flags |= Cluster::MERGED;
                }
                if animated {
                    c.record.flags |= Cluster::ANIMATED_PART;
                } else {
                    c.record.deformation_bound = deformation_bound;
                }
                if let Some(d) = &section_maps.displacement {
                    let uv = |i: u8| w.uvs[verts[usize::from(i)] as usize];
                    let [lo, hi] = d.bounds(tris.as_chunks::<3>().0.iter().map(|t| t.map(uv)));
                    c.record.displacement_min = f16_down(lo);
                    c.record.displacement_max = f16_up(hi);
                }
                clusters.push(c);
            }
        }
        by_level.push(clusters);
    }
    if let Some(o) = &s.omm {
        let clusters: Vec<&mut ClusterData> = by_level.iter_mut().flatten().collect();
        clusters.into_par_iter().for_each(|c| {
            let section = usize::from(c.record.section);
            if let Some(opacity) = &maps[section].opacity {
                let alpha = &c.tris.as_chunks::<3>().0[usize::from(c.record.alpha_first)..];
                let texture = omm_alpha[section].as_ref();
                c.omm = omm::cluster_omm(opacity, texture, &c.uvs, alpha, o);
            }
            c.uvs = Vec::new();
        });
    }

    // Pages, coarsest level first; mesh cluster indices count in page order.
    let mut pages = Vec::new();
    let mut page_info = Vec::new();
    let mut level_spans = vec![Span::default(); level_count];
    let mut part_level_spans = vec![Span::default(); part_count * level_count];
    let mut part_bounds = vec![Aabb::EMPTY; part_count];
    let mut next = 0u32;
    let mut pinned_pages = 0;
    for li in (0..level_count).rev() {
        let clusters = &by_level[li];
        level_spans[li] = Span {
            first: next,
            count: clusters.len() as u32,
        };
        for p in 0..part_count {
            part_level_spans[p * level_count + li].first = next;
        }
        for (i, c) in clusters.iter().enumerate() {
            let span = &mut part_level_spans[usize::from(c.record.part) * level_count + li];
            if span.count == 0 {
                span.first = next + i as u32;
            }
            span.count += 1;
            if c.record.flags & Cluster::MERGED == 0 {
                c.extend(&mut part_bounds[usize::from(c.record.part)], &grid);
            }
        }
        for (bytes, n) in paginate(clusters, stride) {
            let stored = orr_pack::mesh::encode_page(&bytes)?;
            // Micromaps follow the stored corner order, so the page must decode as written.
            let mut decoded = vec![0; bytes.len()];
            orr_pack::mesh::decode_page(&stored, &mut decoded)?;
            assert!(decoded == bytes, "a cooked page decodes to itself");
            page_info.push(PageInfo {
                blob: pages.len() as u32 + 1,
                resident_len: bytes.len() as u32,
                cluster_count: n as u16,
                _pad: 0,
                deps: Span::default(),
            });
            pages.push(stored);
        }
        if li + 1 == level_count {
            pinned_pages = pages.len() as u32;
        }
        next += clusters.len() as u32;
    }

    let mut bounds = Aabb::EMPTY;
    grid.floats.iter().for_each(|&p| bounds.extend(p));
    let sphere = meshopt::compute_sphere_bounds(
        meshopt::PositionDataAdapter {
            data: bytemuck::cast_slice(&grid.floats),
            position_count: grid.floats.len(),
            position_stride: 12,
            position_offset: 0,
        },
        None,
    );
    let merge_level = levels
        .iter()
        .position(|l| l.pieces.iter().any(|p| p.merged))
        .map_or(MeshHeader::NO_MERGE_LEVEL, |l| l as u8);
    let info = MeshInfo {
        header: MeshHeader {
            bounds,
            sphere: Sphere {
                center: sphere.center,
                radius: sphere.radius,
            },
            position_step_log2: s.position_step_log2,
            position_exponent: grid.k,
            clas_truncate_bits: grid.truncate_bits,
            attributes,
            max_vertices,
            max_triangles: s.max_triangles,
            merge_level,
            merge_error: f32::INFINITY,
            pinned_pages,
            ..bytemuck::Zeroable::zeroed()
        },
        sections: m
            .sections
            .iter()
            .enumerate()
            .map(|(i, sec)| Section {
                material_slot: i as u16,
                flags: sec.flags,
                _pad: 0,
            })
            .collect(),
        parts: parts
            .table
            .iter()
            .zip(part_bounds)
            .map(|(p, bounds)| Part { bounds, ..*p })
            .collect(),
        part_levels: part_level_spans
            .into_iter()
            .map(|clusters| PartLevel { clusters })
            .collect(),
        levels: levels
            .iter()
            .zip(level_spans)
            .map(|(l, clusters)| Level {
                error: l.error,
                clusters,
            })
            .collect(),
        roots: Vec::new(),
        pages: page_info,
    };
    Ok(CookedMesh {
        info,
        pages,
        warnings,
    })
}

fn check(m: &MeshData, s: &MeshSettings) -> Result<(), String> {
    let n = m.positions.len();
    if s.max_triangles == 0 || s.max_triangles > 128 || !s.max_triangles.is_multiple_of(16) {
        return Err(format!(
            "max_triangles {} is not a multiple of 16 up to 128",
            s.max_triangles
        ));
    }
    for v in [s.max_vertices, s.max_vertices_foliage] {
        if !(3..=256).contains(&v) {
            return Err(format!("cluster vertex limit {v} is outside 3..=256"));
        }
    }
    let LodMode::Discrete { errors } = &s.lod;
    if errors.len() > 254 {
        return Err(format!("{} levels; at most 255", errors.len() + 1));
    }
    if errors.iter().any(|e| !(e.is_finite() && *e > 0.0)) || !errors.is_sorted() {
        return Err(format!(
            "level errors {errors:?} are not positive, finite and ascending"
        ));
    }
    if let Some(l) = s.merge_level
        && (l == 0 || usize::from(l) > errors.len())
    {
        return Err(format!(
            "merge_level {l} is not a level after the first of {}",
            errors.len() + 1
        ));
    }
    if m.normals.len() != n
        || m.uvs.len() != n
        || m.tangents.as_ref().is_some_and(|t| t.len() != n)
        || m.skin
            .as_ref()
            .is_some_and(|k| k.joints.len() != n || k.weights.len() != n)
        || m.parts.as_ref().is_some_and(|p| {
            p.values.len() != n || matches!(&p.pivots, Pivots::PerVertex(v) if v.len() != n)
        })
    {
        return Err("vertex attribute counts differ".into());
    }
    if let Some(o) = &s.omm {
        o.check()?;
    }
    if !(0.0..=65504.0).contains(&s.deformation_bound) {
        return Err(format!(
            "deformation_bound {} m is outside 0..=65504",
            s.deformation_bound
        ));
    }
    if m.sections.is_empty() || m.sections.len() > 256 {
        return Err(format!("{} sections, expected 1 to 256", m.sections.len()));
    }
    for sec in &m.sections {
        let r = &sec.indices;
        if r.start > r.end || r.end > m.indices.len() || !r.len().is_multiple_of(3) {
            return Err("section index range".into());
        }
    }
    if let Some(&i) = m.indices.iter().find(|&&i| i as usize >= n) {
        return Err(format!("index {i} past {n} vertices"));
    }
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
    if !m.positions.iter().all(|p| finite(p))
        || !m.normals.iter().all(|p| finite(p))
        || !m.uvs.iter().all(|p| finite(p))
        || !m.tangents.iter().flatten().all(|p| finite(p))
    {
        return Err("non-finite vertex attribute".into());
    }
    Ok(())
}

/// Stable partition of a cluster's triangles over its stored vertex uvs: opaque ones first.
/// Returns their count.
fn partition_opaque(uvs: &[[f32; 2]], tris: &mut [u8], opacity: Option<&Opacity>) -> u8 {
    let n = tris.len() / 3;
    let Some(opacity) = opacity else {
        return n as u8;
    };
    let (mut opaque, mut tested) = (Vec::with_capacity(n * 3), Vec::with_capacity(n * 3));
    for t in tris.as_chunks::<3>().0 {
        let uv = t.map(|i| uvs[usize::from(i)]);
        if opacity.opaque(uv) {
            opaque.extend_from_slice(t);
        } else {
            tested.extend_from_slice(t);
        }
    }
    let first = opaque.len() / 3;
    opaque.extend_from_slice(&tested);
    tris.copy_from_slice(&opaque);
    first as u8
}

/// Rotates each triangle's corners as the page codec stores them; order and winding stay.
fn stored_corner_order(tris: &mut [u8]) {
    let n = tris.len() / 3;
    let meshlet = meshopt::Meshlet {
        vertices: &[],
        triangles: tris,
    };
    let stored = meshopt::encode_meshlet(meshlet).expect("the meshlet codec takes a cluster");
    let (_, decoded) =
        meshopt::decode_meshlet(&stored, 0, n).expect("the meshlet codec decodes its own stream");
    tris.copy_from_slice(&decoded);
}

/// A cluster's uvs as the page stores them: unorm16 over the cluster's uv range.
struct ClusterUvs {
    offset: [f32; 2],
    scale: [f32; 2],
}

impl ClusterUvs {
    fn new(w: &lod::Welded, verts: &[u32]) -> Self {
        let (lo, hi) = verts
            .iter()
            .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &v| {
                let uv = Vec2::from(w.uvs[v as usize]);
                (lo.min(uv), hi.max(uv))
            });
        Self {
            offset: lo.to_array(),
            scale: (hi - lo).to_array(),
        }
    }

    fn quantise(&self, uv: [f32; 2]) -> [u32; 2] {
        [0, 1].map(|a| {
            if self.scale[a] > 0.0 {
                ((uv[a] - self.offset[a]) / self.scale[a] * 65535.0)
                    .round()
                    .clamp(0.0, 65535.0) as u32
            } else {
                0
            }
        })
    }

    /// The uv a page reader decodes for `uv`.
    fn stored(&self, uv: [f32; 2]) -> [f32; 2] {
        let q = self.quantise(uv);
        [0, 1].map(|a| self.offset[a] + q[a] as f32 / 65535.0 * self.scale[a])
    }
}

/// f16 bits of the smallest half at or above `v`.
fn f16_up(v: f32) -> u16 {
    let h = half::f16::from_f32(v);
    if h.to_f32() < v {
        f16_next(h, true)
    } else {
        h.to_bits()
    }
}

/// f16 bits of the largest half at or below `v`.
fn f16_down(v: f32) -> u16 {
    let h = half::f16::from_f32(v);
    if h.to_f32() > v {
        f16_next(h, false)
    } else {
        h.to_bits()
    }
}

/// Bits of the neighbouring finite half above or below `h`.
fn f16_next(h: half::f16, up: bool) -> u16 {
    let b = h.to_bits();
    if h.to_f32() == 0.0 {
        return if up { 0x0001 } else { 0x8001 };
    }
    if (b & 0x8000 == 0) == up {
        b + 1
    } else {
        b - 1
    }
}

/// The grid exponent `k` of positions: `max(step_log2, ceil(log2(extent / 65535)))`,
/// computed without log2 rounding.
pub(crate) fn grid_exponent(positions: &[[f32; 3]], step_log2: i8) -> i8 {
    let mut b = Aabb::EMPTY;
    positions.iter().for_each(|&p| b.extend(p));
    let extent = (0..3)
        .map(|a| f64::from(b.max[a]) - f64::from(b.min[a]))
        .fold(0.0, f64::max);
    let mut k = i32::from(step_log2);
    while extent > 65535.0 * 2f64.powi(k) {
        k += 1;
    }
    k as i8
}

/// The mesh grid: every position as an integer multiple of `2^k` metres.
pub(crate) struct Grid {
    pub k: i8,
    truncate_bits: u8,
    pub cells: Vec<[i32; 3]>,
    pub floats: Vec<[f32; 3]>,
}

impl Grid {
    fn new(positions: &[[f32; 3]], step_log2: i8, min: Option<i8>) -> Result<Self, String> {
        let own = grid_exponent(positions, step_log2);
        let k = i32::from(own.max(min.unwrap_or(i8::MIN)));
        let step = 2f64.powi(k);
        let mut max_abs = 0;
        let mut cells = Vec::with_capacity(positions.len());
        for p in positions {
            let g = p.map(|x| (f64::from(x) / step + 0.5).floor());
            if g.iter().any(|v| v.abs() >= f64::from(1 << 23)) {
                return Err(format!(
                    "position {p:?} m needs more than 24 bits on the 2^{k} m grid"
                ));
            }
            let g = g.map(|v| v as i32);
            max_abs = g.iter().fold(max_abs, |m, v| m.max(v.unsigned_abs()));
            cells.push(g);
        }
        let floats = cells
            .iter()
            .map(|g| g.map(|v| (f64::from(v) * step) as f32))
            .collect();
        let mut b = Aabb::EMPTY;
        positions.iter().for_each(|&p| b.extend(p));
        let m = meshopt::compute_position_exponent(b.min, b.max, step_log2.into(), 16);
        let own = i32::from(own);
        assert!(
            m == own || m == own + 1,
            "grid exponent {own}, meshoptimizer {m}"
        );
        Ok(Self {
            k: k as i8,
            truncate_bits: (24 - (32 - max_abs.leading_zeros())) as u8,
            cells,
            floats,
        })
    }

    pub(crate) fn step(&self) -> f64 {
        2f64.powi(i32::from(self.k))
    }
}

struct ClusterData {
    record: Cluster,
    local: Vec<[u16; 3]>,
    attrs: Vec<u32>,
    tris: Vec<u8>,
    /// Vertex uvs as the page stores them, until the micromaps are made.
    uvs: Vec<[f32; 2]>,
    omm: Option<ClusterOmm>,
}

impl ClusterData {
    fn slots(&self) -> usize {
        self.local.len().next_multiple_of(2)
    }

    /// Extends `b` by the cluster's rest AABB.
    fn extend(&self, b: &mut Aabb, grid: &Grid) {
        let corner = |l: [u16; 3]| {
            [0, 1, 2].map(|a| {
                ((f64::from(self.record.anchor[a]) + f64::from(l[a])) * grid.step()) as f32
            })
        };
        b.extend(corner([0; 3]));
        b.extend(corner(self.record.local_max));
    }
}

fn build_cluster(
    w: &lod::Welded,
    grid: &Grid,
    verts: &[u32],
    tris: &[u8],
    bounds: &meshopt::Bounds,
    flags: u8,
    uv: &ClusterUvs,
) -> ClusterData {
    let anchor = [0, 1, 2].map(|a| {
        verts
            .iter()
            .map(|&v| grid.cells[v as usize][a])
            .min()
            .unwrap()
    });
    let local: Vec<[u16; 3]> = verts
        .iter()
        .map(|&v| {
            let g = grid.cells[v as usize];
            [0, 1, 2].map(|a| u16::try_from(g[a] - anchor[a]).expect("cluster spans 16 bits"))
        })
        .collect();
    let local_max = [0, 1, 2].map(|a| local.iter().map(|l| l[a]).max().unwrap());

    let mut attrs = Vec::with_capacity(verts.len() * 3);
    for &v in verts {
        let v = v as usize;
        attrs.push(oct_encode(
            Vec3::from(w.normals[v]).normalize_or(Vec3::Z).to_array(),
        ));
        if let Some(t) = &w.tangents {
            let [x, y, z, s] = t[v];
            let e = oct_encode(Vec3::new(x, y, z).normalize_or(Vec3::X).to_array());
            attrs.push(e & !(1 << 16) | u32::from(s < 0.0) << 16);
        }
        let q = uv.quantise(w.uvs[v]);
        attrs.push(q[0] | q[1] << 16);
        if let Some(k) = &w.skin {
            attrs.push(u32::from_le_bytes(k.joints[v]));
            attrs.push(u32::from_le_bytes(k.weights[v]));
        }
    }

    let area: f64 = tris
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            let p = t.map(|i| Vec3::from(grid.floats[verts[usize::from(i)] as usize]).as_dvec3());
            0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
        })
        .sum();
    let tri_count = (tris.len() / 3) as u8;
    let sphere = Sphere {
        center: bounds.center,
        radius: bounds.radius,
    };
    let record = Cluster {
        self_sphere: sphere,
        parent_sphere: sphere,
        cull_sphere: sphere,
        cone: [
            bounds.cone_axis_s8[0],
            bounds.cone_axis_s8[1],
            bounds.cone_axis_s8[2],
            bounds.cone_cutoff_s8,
        ],
        anchor,
        vertex_count: verts.len() as u16,
        tri_count,
        alpha_first: tri_count,
        kind: Cluster::TRIANGLES,
        flags: if flags & Section::EMISSIVE != 0 {
            Cluster::EMISSIVE
        } else {
            0
        },
        local_max,
        uv_offset: uv.offset,
        uv_scale: uv.scale,
        omm_first: Cluster::NO_OMM,
        area: area as f32,
        ..bytemuck::Zeroable::zeroed()
    };
    ClusterData {
        record,
        local,
        attrs,
        tris: tris.to_vec(),
        uvs: Vec::new(),
        omm: None,
    }
}

/// Packs clusters in order into resident pages of at most [`PAGE_MAX`] bytes; returns each
/// page and its cluster count.
fn paginate(clusters: &[ClusterData], stride: usize) -> Vec<(Vec<u8>, usize)> {
    let cost = |c: &ClusterData| {
        size_of::<Cluster>()
            + c.slots() * (6 + stride)
            + c.tris.len().next_multiple_of(4)
            + c.omm.as_ref().map_or(0, ClusterOmm::bytes)
    };
    let mut pages = Vec::new();
    let mut first = 0;
    while first < clusters.len() {
        let mut size = size_of::<PageHeader>();
        let mut end = first;
        let mut aligned = false;
        while end < clusters.len() {
            let c = &clusters[end];
            // Room to align the micromap region, once per page.
            let align = c.omm.is_some() && !aligned;
            let add = cost(c) + if align { OMM_ALIGN as usize - 1 } else { 0 };
            if size + add > PAGE_MAX {
                break;
            }
            size += add;
            aligned |= align;
            end += 1;
        }
        pages.push((write_page(&clusters[first..end], stride), end - first));
        first = end;
    }
    pages
}

fn write_page(clusters: &[ClusterData], stride: usize) -> Vec<u8> {
    let slots: usize = clusters.iter().map(ClusterData::slots).sum();
    let tris_len: usize = clusters
        .iter()
        .map(|c| c.tris.len().next_multiple_of(4))
        .sum();
    let positions = size_of::<PageHeader>() + clusters.len() * size_of::<Cluster>();
    let attrs = positions + slots * 6;
    let tris = attrs + slots * stride;
    let region = omm::page_region(&clusters.iter().map(|c| c.omm.as_ref()).collect::<Vec<_>>());
    let omm = if region.bytes.is_empty() {
        tris + tris_len
    } else {
        (tris + tris_len).next_multiple_of(OMM_ALIGN as usize)
    };
    let len = omm + region.bytes.len();
    let header = PageHeader {
        cluster_count: clusters.len() as u32,
        attr_stride: stride as u32,
        positions_offset: positions as u32,
        attrs_offset: attrs as u32,
        tris_offset: tris as u32,
        omm_offset: omm as u32,
        len: len as u32,
        omm_usage: region.usage,
        ..bytemuck::Zeroable::zeroed()
    };
    let mut page = vec![0u8; len];
    page[..size_of::<PageHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
    page[omm..].copy_from_slice(&region.bytes);
    let (mut vertex_first, mut tri_first) = (0, 0);
    for (i, c) in clusters.iter().enumerate() {
        let mut r = c.record;
        r.vertex_first = vertex_first as u32;
        r.tri_first = tri_first as u32;
        r.omm_first = region.omm_first[i].unwrap_or(Cluster::NO_OMM);
        let at = size_of::<PageHeader>() + i * size_of::<Cluster>();
        page[at..at + size_of::<Cluster>()].copy_from_slice(bytemuck::bytes_of(&r));
        let p = positions + vertex_first * 6;
        page[p..p + c.local.len() * 6].copy_from_slice(bytemuck::cast_slice(&c.local));
        let a = attrs + vertex_first * stride;
        page[a..a + c.attrs.len() * 4].copy_from_slice(bytemuck::cast_slice(&c.attrs));
        let t = tris + tri_first;
        page[t..t + c.tris.len()].copy_from_slice(&c.tris);
        vertex_first += c.slots();
        tri_first += c.tris.len().next_multiple_of(4);
    }
    page
}

impl MeshData {
    /// Hash of everything the mesh cook reads but its settings and the cooked textures,
    /// which enter through their keys.
    pub(crate) fn hash(&self) -> Hash32 {
        let mut h = blake3::Hasher::new();
        let mut part = |b: &[u8]| {
            h.update(&(b.len() as u64).to_le_bytes());
            h.update(b);
        };
        part(bytemuck::cast_slice(&self.positions));
        part(bytemuck::cast_slice(&self.normals));
        part(self.tangents.as_deref().map_or(&[], bytemuck::cast_slice));
        part(bytemuck::cast_slice(&self.uvs));
        part(bytemuck::cast_slice(&self.indices));
        for s in &self.sections {
            part(&(s.indices.start as u64).to_le_bytes());
            part(&(s.indices.end as u64).to_le_bytes());
            part(&[s.flags, s.subsurface.into()]);
            part(s.maps.fingerprint().as_bytes());
        }
        if let Some(k) = &self.skin {
            part(bytemuck::cast_slice(&k.joints));
            part(bytemuck::cast_slice(&k.weights));
        }
        if let Some(p) = &self.parts {
            part(bytemuck::cast_slice(&p.values));
            part(&p.count.to_le_bytes());
            part(&[p.by_section.into()]);
            match &p.pivots {
                Pivots::PerValue(v) => {
                    part(b"value");
                    part(bytemuck::cast_slice(v));
                }
                Pivots::PerVertex(v) => {
                    part(b"vertex");
                    part(bytemuck::cast_slice(v));
                }
                Pivots::Origin => part(b"origin"),
            }
            part(p.joints.as_deref().map_or(&[], bytemuck::cast_slice));
        }
        part(format!("{:?}", self.min_exponent).as_bytes());
        Hash32(*h.finalize().as_bytes())
    }

    pub(crate) fn triangles(&self) -> u64 {
        self.indices.len() as u64 / 3
    }
}
