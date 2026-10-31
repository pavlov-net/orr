//! The cook report: what each asset read and wrote, how each entry was obtained, and sizes.

use std::path::PathBuf;

use orr_pack::mesh::{
    Cluster, MeshHeader, MeshInfo, Node, OMM_STATE_UNKNOWN_TRANSPARENT, OMM_UNKNOWN_TRANSPARENT,
    PageView, Section, cluster_resident_len, decode_page,
};
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, Class, Entry};
use serde::Serialize;

use crate::RasterReport;

#[derive(Clone, Debug, Default, Serialize)]
pub struct CookReport {
    /// Wall time of the whole cook.
    pub seconds: f64,
    pub archives: Vec<ArchiveReport>,
    /// One per manifest asset, in manifest order.
    pub assets: Vec<AssetReport>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ArchiveReport {
    /// `None` for a memory archive.
    pub output: Option<PathBuf>,
    /// False when every entry matched the archive on disk, which was left as it was.
    pub written: bool,
    pub entries: usize,
    pub bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AssetReport {
    pub name: String,
    pub archive: usize,
    /// Files the importer read.
    pub inputs: Vec<PathBuf>,
    pub entries: Vec<EntryReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EntryReport {
    pub name: String,
    pub class: String,
    pub outcome: Outcome,
    /// Time of the job that cooked it; 0 when linked.
    pub seconds: f64,
    /// The scheduler's estimate of the job's peak bytes.
    pub working_set: u64,
    /// Decoded and stored bytes over its blobs.
    pub raw: u64,
    pub stored: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mesh: Option<MeshReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub texture: Option<TextureReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raster: Option<RasterReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neural: Option<NeuralReport>,
    /// Content the cook accepted but flags: displacement that differs across uv seams,
    /// subsurface materials on open meshes.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// How an entry was obtained.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Cooked,
    /// Linked from the cook cache.
    Cache,
    /// Linked from the archive the cook replaces.
    Archive,
}

/// Sizes are statistics, never caps.
#[derive(Clone, Debug, Serialize)]
pub struct MeshReport {
    pub position_exponent: i8,
    pub clas_truncate_bits: u8,
    pub sections: usize,
    pub parts: usize,
    pub pages: usize,
    pub pinned_pages: u32,
    pub clusters: u32,
    /// First merged level, if any.
    pub merge_level: Option<u8>,
    /// DAG mode: the error the merged stage serves from, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge_error: Option<f32>,
    /// DAG mode: groups, the deepest group depth plus one, hierarchy nodes, their bytes and
    /// node pages (0 when the hierarchy is resident).
    pub groups: usize,
    pub dag_depth: u32,
    pub hierarchy_nodes: usize,
    pub hierarchy_bytes: u64,
    pub node_pages: usize,
    /// Stored bytes of the whole entry per source triangle.
    pub bytes_per_triangle: f64,
    /// Mean triangles per cluster on alpha-tested or double-sided sections, if any.
    pub foliage_cluster_fill: Option<f64>,
    /// Finest first: discrete levels, or in DAG mode the part stage's group depths.
    pub levels: Vec<LevelReport>,
    /// DAG mode: the merged stage's group depths.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub merged_levels: Vec<LevelReport>,
}

/// A page counts toward every level it holds clusters of, and its bytes are shared among
/// them by the clusters' resident sizes.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LevelReport {
    /// Simplification error from the source, metres; in DAG mode the largest of the
    /// depth's clusters.
    pub error: f32,
    pub triangles: u64,
    pub clusters: u32,
    pub pages: usize,
    /// Page bytes as stored and as resident.
    pub stored: u64,
    pub resident: u64,
    pub stored_per_triangle: f64,
    pub resident_per_triangle: f64,
    /// Resident bytes of the pages' micromap regions.
    pub omm_bytes: u64,
    /// Alpha-tested triangles with micromaps.
    pub omm_triangles: u64,
    /// Share of those triangles' micro-triangles in an unknown state, each triangle weighted
    /// alike; a special index counts for its whole triangle.
    pub omm_unknown: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TextureReport {
    pub vk_format: u32,
    pub width: u32,
    pub height: u32,
    pub layers: u16,
    pub mips: u8,
    /// Stored bytes per mip blob, then the tail.
    pub blob_bytes: Vec<u32>,
}

/// A cooked neural texture set.
#[derive(Clone, Debug, Serialize)]
pub struct NeuralReport {
    /// LibNTC's bits per mip 0 texel of the chosen run.
    pub bits_per_pixel: f32,
    /// PSNR over all channels of the chosen run, dB.
    pub psnr: f32,
    /// Compression runs of the adaptive search.
    pub runs: u32,
    /// Raw bytes of the latents and weights, and of the BC textures of the members.
    pub bytes: u64,
    pub bc_bytes: u64,
    pub textures: Vec<NeuralTextureReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NeuralTextureReport {
    /// The BC texture entry.
    pub name: String,
    /// Per channel at mip 0, dB.
    pub psnr: Vec<f32>,
    /// Whether the set stands in for the texture.
    pub kept: bool,
}

/// Sizes of an entry; mesh and texture details only for entries this cook made.
/// `triangles` counts a mesh's source triangles.
pub(crate) fn entry(
    a: &Archive,
    e: &Entry,
    outcome: Outcome,
    seconds: f64,
    working_set: u64,
    triangles: u64,
) -> EntryReport {
    let blobs = a.blobs(e);
    let head = || {
        let mut b = vec![0; blobs[0].raw_len as usize];
        a.read(&blobs[0], &mut b).ok().map(|()| b)
    };
    let stored: u64 = blobs.iter().map(|b| u64::from(b.stored_len)).sum();
    let detail = outcome == Outcome::Cooked;
    let mesh = (detail && e.class == Class::Mesh as u16)
        .then(head)
        .flatten()
        .and_then(|b| MeshInfo::parse(&b).ok())
        .and_then(|m| mesh_report(a, e, &m, stored, triangles));
    let texture = (detail && e.class == Class::Texture as u16)
        .then(head)
        .flatten()
        .and_then(|b| TextureInfo::parse(&b).ok())
        .map(|t| TextureReport {
            vk_format: t.header.vk_format,
            width: t.header.width,
            height: t.header.height,
            layers: t.header.layers,
            mips: t.header.mip_count,
            blob_bytes: blobs[1..].iter().map(|b| b.stored_len).collect(),
        });
    EntryReport {
        name: a.name(e).into(),
        class: match Class::from_u16(e.class) {
            Some(c) => c.name().into(),
            None => format!("user:{:#06x}", e.class),
        },
        outcome,
        seconds,
        working_set,
        raw: blobs.iter().map(|b| u64::from(b.raw_len)).sum(),
        stored,
        mesh,
        texture,
        raster: None,
        neural: None,
        warnings: Vec::new(),
    }
}

/// Per level: triangles, clusters and page bytes, from the decoded pages.
fn mesh_report(
    a: &Archive,
    e: &Entry,
    m: &MeshInfo,
    stored: u64,
    triangles: u64,
) -> Option<MeshReport> {
    let blobs = a.blobs(e);
    let dag = !m.groups.is_empty();
    let clusters: u32 = m.pages.iter().map(|p| u32::from(p.cluster_count)).sum();
    // Level of each mesh cluster: a discrete level, or a DAG group depth.
    let mut level_of = vec![0usize; clusters as usize];
    if dag {
        let mut page_first = Vec::with_capacity(m.pages.len());
        let mut first = 0u32;
        for p in &m.pages {
            page_first.push(first);
            first += u32::from(p.cluster_count);
        }
        for g in &m.groups {
            let at = (page_first[g.page as usize] + u32::from(g.first_cluster)) as usize;
            level_of[at..at + usize::from(g.cluster_count)].fill(usize::from(g.depth));
        }
    } else {
        for (li, l) in m.levels.iter().enumerate() {
            let r = l.clusters.first as usize..(l.clusters.first + l.clusters.count) as usize;
            level_of[r].fill(li);
        }
    }
    let mut levels: Vec<LevelReport> = m
        .levels
        .iter()
        .map(|l| LevelReport {
            error: l.error,
            clusters: l.clusters.count,
            ..LevelReport::default()
        })
        .collect();
    let mut merged_levels: Vec<LevelReport> = Vec::new();
    let (mut foliage_tris, mut foliage_clusters) = (0u64, 0u64);
    let mut first = 0usize;
    for p in &m.pages {
        let b = blobs.get(p.blob as usize)?;
        let mut raw = vec![0; b.raw_len as usize];
        a.read(b, &mut raw).ok()?;
        let mut page = vec![0; p.resident_len as usize];
        let layout = decode_page(&raw, &mut page).ok()?;
        let view = PageView::new(&page, &m.header).ok()?;
        let maps = view.micromaps().ok()?;
        // Per cluster: its level and resident bytes, which share out the page's bytes.
        let mut shares: Vec<(bool, usize, f64)> = Vec::new();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            let level = level_of[first + i as usize];
            let merged = c.flags & Cluster::MERGED != 0 && dag;
            let list = if merged {
                &mut merged_levels
            } else {
                &mut levels
            };
            if list.len() <= level {
                list.resize(level + 1, LevelReport::default());
            }
            let r = &mut list[level];
            if dag {
                r.clusters += 1;
                r.error = r.error.max(c.self_error);
            }
            let resident = cluster_resident_len(
                usize::from(c.vertex_count),
                usize::from(c.tri_count),
                layout.attr_stride as usize,
            );
            shares.push((merged, level, resident as f64));
            r.triangles += u64::from(c.tri_count);
            for t in u32::from(c.alpha_first)..u32::from(c.tri_count) {
                let Some(index) = view.omm_index(&c, t) else {
                    break;
                };
                r.omm_triangles += 1;
                r.omm_unknown += match u32::try_from(index) {
                    Ok(e) => {
                        let micro = 1u32 << (2 * maps.entry(e).level);
                        let unknown = (0..micro)
                            .filter(|&k| maps.state(e, k) >= OMM_STATE_UNKNOWN_TRANSPARENT)
                            .count();
                        unknown as f64 / f64::from(micro)
                    }
                    Err(_) => f64::from(u8::from(index <= OMM_UNKNOWN_TRANSPARENT)),
                };
            }
            let foliage = Section::ALPHA_TESTED | Section::DOUBLE_SIDED;
            if m.sections.get(usize::from(c.section))?.flags & foliage != 0 {
                foliage_tris += u64::from(c.tri_count);
                foliage_clusters += 1;
            }
        }
        let total: f64 = shares.iter().map(|s| s.2).sum();
        let mut counted = Vec::new();
        for (merged, level, w) in shares {
            let list = if merged {
                &mut merged_levels
            } else {
                &mut levels
            };
            let r = &mut list[level];
            let share = |bytes: u32| (f64::from(bytes) * w / total).round() as u64;
            r.stored += share(b.stored_len);
            r.resident += share(p.resident_len);
            r.omm_bytes += share(layout.len - layout.omm_offset);
            if !counted.contains(&(merged, level)) {
                counted.push((merged, level));
                r.pages += 1;
            }
        }
        first += usize::from(p.cluster_count);
    }
    for l in levels.iter_mut().chain(&mut merged_levels) {
        l.stored_per_triangle = l.stored as f64 / l.triangles.max(1) as f64;
        l.resident_per_triangle = l.resident as f64 / l.triangles.max(1) as f64;
        l.omm_unknown /= l.omm_triangles.max(1) as f64;
    }
    let hierarchy_nodes = if m.node_pages.is_empty() {
        m.nodes.len()
    } else {
        m.node_pages.iter().map(|p| p.node_count as usize).sum()
    };
    Some(MeshReport {
        position_exponent: m.header.position_exponent,
        clas_truncate_bits: m.header.clas_truncate_bits,
        sections: m.sections.len(),
        parts: m.parts.len(),
        pages: m.pages.len(),
        pinned_pages: m.header.pinned_pages,
        clusters,
        merge_level: (m.header.merge_level != MeshHeader::NO_MERGE_LEVEL)
            .then_some(m.header.merge_level),
        merge_error: (dag && m.header.merge_error.is_finite()).then_some(m.header.merge_error),
        groups: m.groups.len(),
        dag_depth: m
            .groups
            .iter()
            .map(|g| u32::from(g.depth) + 1)
            .max()
            .unwrap_or(0),
        hierarchy_nodes,
        hierarchy_bytes: (hierarchy_nodes * size_of::<Node>()) as u64,
        node_pages: m.node_pages.len(),
        bytes_per_triangle: stored as f64 / triangles.max(1) as f64,
        foliage_cluster_fill: (foliage_clusters > 0)
            .then(|| foliage_tris as f64 / foliage_clusters as f64),
        levels,
        merged_levels,
    })
}
