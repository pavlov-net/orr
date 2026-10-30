//! Level-0 mesh cook: grid quantisation, spatial clusters, resident pages.

use std::ops::Range;

use glam::{Vec2, Vec3};
use meshopt::VertexDataAdapter;
use orr_pack::mesh::{
    Cluster, Level, MeshHeader, MeshInfo, PAGE_MAX, PageHeader, PageInfo, Part, PartLevel, Section,
    oct_encode,
};
use orr_pack::{Aabb, Span, Sphere};
use serde::Deserialize;

use crate::{Error, Result};

/// How meshes are clustered and quantised.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MeshSettings {
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
    /// skinned mesh (the bound GPU skinning builds cluster templates with); ignored without
    /// a skin.
    pub deformation_bound: f32,
}

impl Default for MeshSettings {
    fn default() -> Self {
        Self {
            max_vertices: 128,
            max_vertices_foliage: 256,
            max_triangles: 128,
            position_step_log2: -11,
            deformation_bound: 0.0,
        }
    }
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
}

/// Per vertex: four skeleton joints and their unorm8 weights, summing to 255.
pub(crate) struct Skin {
    pub joints: Vec<[u8; 4]>,
    pub weights: Vec<[u8; 4]>,
}

pub(crate) struct SectionData {
    pub indices: Range<usize>,
    /// [`Section`] flags.
    pub flags: u8,
}

pub(crate) struct CookedMesh {
    pub info: MeshInfo,
    /// Stored page blobs, in [`MeshInfo::pages`] order.
    pub pages: Vec<Vec<u8>>,
}

pub(crate) fn cook(asset: &str, m: &MeshData, s: &MeshSettings) -> Result<CookedMesh> {
    let fail = |message: String| Error::Mesh {
        asset: asset.into(),
        message,
    };
    check(m, s).map_err(fail)?;
    let grid = Grid::new(&m.positions, s.position_step_log2).map_err(fail)?;
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
    let mut clusters = Vec::new();
    let mut max_vertices = 0;
    for (si, sec) in m.sections.iter().enumerate() {
        let foliage = sec.flags & (Section::ALPHA_TESTED | Section::DOUBLE_SIDED) != 0;
        let maxv = if foliage {
            s.max_vertices_foliage
        } else {
            s.max_vertices
        };
        max_vertices = max_vertices.max(maxv);
        let maxt = usize::from(s.max_triangles);
        let mut ml = meshopt::build_meshlets_spatial(
            &m.indices[sec.indices.clone()],
            &adapter,
            usize::from(maxv),
            maxt / 4,
            maxt,
            0.5,
        );
        for i in 0..ml.len() {
            let r = ml.meshlets[i];
            let (vo, vc) = (r.vertex_offset as usize, r.vertex_count as usize);
            let (to, tc) = (r.triangle_offset as usize, r.triangle_count as usize);
            meshopt::optimize_meshlet_level(
                &mut ml.vertices[vo..vo + vc],
                &mut ml.triangles[to..to + tc * 3],
                3,
            );
            let bounds = meshopt::compute_meshlet_bounds(ml.get(i), &adapter);
            let mut c = build_cluster(m, &grid, ml.get(i), &bounds, si as u8, sec.flags);
            c.record.deformation_bound = deformation_bound;
            clusters.push(c);
        }
    }

    let pages = paginate(&clusters, stride);
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
    let all = Span {
        first: 0,
        count: clusters.len() as u32,
    };
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
            merge_level: MeshHeader::NO_MERGE_LEVEL,
            merge_error: f32::INFINITY,
            // A single level is the coarsest, so every page is pinned.
            pinned_pages: pages.len() as u32,
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
        parts: vec![Part {
            bounds,
            pivot: [0.0; 3],
            joint: 0,
            _pad: 0,
        }],
        part_levels: vec![PartLevel { clusters: all }],
        levels: vec![Level {
            error: 0.0,
            clusters: all,
        }],
        roots: Vec::new(),
        pages: pages
            .iter()
            .enumerate()
            .map(|(i, (bytes, n))| PageInfo {
                blob: i as u32 + 1,
                resident_len: bytes.len() as u32,
                cluster_count: *n as u16,
                _pad: 0,
                deps: Span::default(),
            })
            .collect(),
    };
    Ok(CookedMesh {
        info,
        pages: pages.into_iter().map(|(p, _)| p).collect(),
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
    if m.normals.len() != n
        || m.uvs.len() != n
        || m.tangents.as_ref().is_some_and(|t| t.len() != n)
        || m.skin
            .as_ref()
            .is_some_and(|k| k.joints.len() != n || k.weights.len() != n)
    {
        return Err("vertex attribute counts differ".into());
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

/// f16 bits of the smallest half at or above `v`.
fn f16_up(v: f32) -> u16 {
    let h = half::f16::from_f32(v);
    if h.to_f32() < v {
        h.to_bits() + 1
    } else {
        h.to_bits()
    }
}

/// The mesh grid: every position as an integer multiple of `2^k` metres.
struct Grid {
    k: i8,
    truncate_bits: u8,
    cells: Vec<[i32; 3]>,
    floats: Vec<[f32; 3]>,
}

impl Grid {
    fn new(positions: &[[f32; 3]], step_log2: i8) -> Result<Self, String> {
        let mut b = Aabb::EMPTY;
        positions.iter().for_each(|&p| b.extend(p));
        let extent = (0..3)
            .map(|a| f64::from(b.max[a]) - f64::from(b.min[a]))
            .fold(0.0, f64::max);
        // k = max(step, ceil(log2(extent / 65535))), without log2 rounding.
        let mut k = i32::from(step_log2);
        while extent > 65535.0 * 2f64.powi(k) {
            k += 1;
        }
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
        let k8 = k as i8;
        let floats = cells
            .iter()
            .map(|g| g.map(|v| (f64::from(v) * step) as f32))
            .collect();
        let m = meshopt::compute_position_exponent(b.min, b.max, step_log2.into(), 16);
        assert!(m == k || m == k + 1, "grid exponent {k}, meshoptimizer {m}");
        Ok(Self {
            k: k8,
            truncate_bits: (24 - (32 - max_abs.leading_zeros())) as u8,
            cells,
            floats,
        })
    }
}

struct ClusterData {
    record: Cluster,
    local: Vec<[u16; 3]>,
    attrs: Vec<u32>,
    tris: Vec<u8>,
}

impl ClusterData {
    fn slots(&self) -> usize {
        self.local.len().next_multiple_of(2)
    }
}

fn build_cluster(
    m: &MeshData,
    grid: &Grid,
    ml: meshopt::Meshlet<'_>,
    bounds: &meshopt::Bounds,
    section: u8,
    flags: u8,
) -> ClusterData {
    let verts = ml.vertices;
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

    let (uv_lo, uv_hi) = verts
        .iter()
        .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(lo, hi), &v| {
            let uv = Vec2::from(m.uvs[v as usize]);
            (lo.min(uv), hi.max(uv))
        });
    let (uv_lo, uv_hi) = (uv_lo.to_array(), uv_hi.to_array());
    let uv_scale = [0, 1].map(|a| uv_hi[a] - uv_lo[a]);
    let unorm = |v: f32, lo: f32, scale: f32| {
        if scale > 0.0 {
            ((v - lo) / scale * 65535.0).round().clamp(0.0, 65535.0) as u32
        } else {
            0
        }
    };
    let mut attrs = Vec::with_capacity(verts.len() * 3);
    for &v in verts {
        let v = v as usize;
        attrs.push(oct_encode(
            Vec3::from(m.normals[v]).normalize_or(Vec3::Z).to_array(),
        ));
        if let Some(t) = &m.tangents {
            let [x, y, z, w] = t[v];
            let e = oct_encode(Vec3::new(x, y, z).normalize_or(Vec3::X).to_array());
            attrs.push(e & !(1 << 16) | u32::from(w < 0.0) << 16);
        }
        let uv = m.uvs[v];
        attrs.push(unorm(uv[0], uv_lo[0], uv_scale[0]) | unorm(uv[1], uv_lo[1], uv_scale[1]) << 16);
        if let Some(k) = &m.skin {
            attrs.push(u32::from_le_bytes(k.joints[v]));
            attrs.push(u32::from_le_bytes(k.weights[v]));
        }
    }

    let tris = ml.triangles.to_vec();
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
        self_error: 0.0,
        parent_sphere: sphere,
        parent_error: f32::INFINITY,
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
        alpha_first: if flags & Section::ALPHA_TESTED != 0 {
            0
        } else {
            tri_count
        },
        section,
        kind: Cluster::TRIANGLES,
        flags: if flags & Section::EMISSIVE != 0 {
            Cluster::EMISSIVE
        } else {
            0
        },
        local_max,
        uv_offset: uv_lo,
        uv_scale,
        omm_first: Cluster::NO_OMM,
        area: area as f32,
        ..bytemuck::Zeroable::zeroed()
    };
    ClusterData {
        record,
        local,
        attrs,
        tris,
    }
}

/// Packs clusters in order into pages of at most [`PAGE_MAX`] bytes; returns each page and
/// its cluster count.
fn paginate(clusters: &[ClusterData], stride: usize) -> Vec<(Vec<u8>, usize)> {
    let cost = |c: &ClusterData| {
        size_of::<Cluster>() + c.slots() * (6 + stride) + (c.tris.len()).next_multiple_of(4)
    };
    let mut pages = Vec::new();
    let mut first = 0;
    while first < clusters.len() {
        let mut size = size_of::<PageHeader>();
        let mut end = first;
        while end < clusters.len() && size + cost(&clusters[end]) <= PAGE_MAX {
            size += cost(&clusters[end]);
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
    let omm = tris + tris_len;
    let header = PageHeader {
        cluster_count: clusters.len() as u32,
        attr_stride: stride as u32,
        positions_offset: positions as u32,
        attrs_offset: attrs as u32,
        tris_offset: tris as u32,
        omm_offset: omm as u32,
        len: omm as u32,
        ..bytemuck::Zeroable::zeroed()
    };
    let mut page = vec![0u8; omm];
    page[..size_of::<PageHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
    let (mut vertex_first, mut tri_first) = (0, 0);
    for (i, c) in clusters.iter().enumerate() {
        let mut r = c.record;
        r.vertex_first = vertex_first as u32;
        r.tri_first = tri_first as u32;
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
    /// Hash of everything the mesh cook reads.
    pub(crate) fn hash(&self) -> orr_pack::Hash32 {
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
            part(&[s.flags]);
        }
        if let Some(k) = &self.skin {
            part(bytemuck::cast_slice(&k.joints));
            part(bytemuck::cast_slice(&k.weights));
        }
        orr_pack::Hash32(*h.finalize().as_bytes())
    }

    pub(crate) fn triangles(&self) -> u64 {
        self.indices.len() as u64 / 3
    }
}
