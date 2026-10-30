//! The cook report: what each asset read and wrote, how each entry was obtained, and sizes.

use std::path::PathBuf;

use orr_pack::mesh::{MeshHeader, MeshInfo, PageView, Section, decode_page};
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
    /// Stored bytes of the whole entry per source triangle.
    pub bytes_per_triangle: f64,
    /// Mean triangles per cluster on alpha-tested or double-sided sections, if any.
    pub foliage_cluster_fill: Option<f64>,
    /// Finest first.
    pub levels: Vec<LevelReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LevelReport {
    /// Simplification error from the source, metres.
    pub error: f32,
    pub triangles: u64,
    pub clusters: u32,
    pub pages: usize,
    /// Page bytes as stored and as resident.
    pub stored: u64,
    pub resident: u64,
    pub stored_per_triangle: f64,
    pub resident_per_triangle: f64,
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
    let mut levels: Vec<LevelReport> = m
        .levels
        .iter()
        .map(|l| LevelReport {
            error: l.error,
            triangles: 0,
            clusters: l.clusters.count,
            pages: 0,
            stored: 0,
            resident: 0,
            stored_per_triangle: 0.0,
            resident_per_triangle: 0.0,
        })
        .collect();
    let (mut foliage_tris, mut foliage_clusters) = (0u64, 0u64);
    let mut first = 0u32;
    for p in &m.pages {
        let b = blobs.get(p.blob as usize)?;
        let level = m.levels.iter().position(|l| {
            (l.clusters.first..l.clusters.first + l.clusters.count).contains(&first)
        })?;
        let mut raw = vec![0; b.raw_len as usize];
        a.read(b, &mut raw).ok()?;
        let mut page = vec![0; p.resident_len as usize];
        decode_page(&raw, &mut page).ok()?;
        let view = PageView::new(&page, &m.header).ok()?;
        let r = &mut levels[level];
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            r.triangles += u64::from(c.tri_count);
            let foliage = Section::ALPHA_TESTED | Section::DOUBLE_SIDED;
            if m.sections.get(usize::from(c.section))?.flags & foliage != 0 {
                foliage_tris += u64::from(c.tri_count);
                foliage_clusters += 1;
            }
        }
        r.pages += 1;
        r.stored += u64::from(b.stored_len);
        r.resident += u64::from(p.resident_len);
        first += u32::from(p.cluster_count);
    }
    for l in &mut levels {
        l.stored_per_triangle = l.stored as f64 / l.triangles.max(1) as f64;
        l.resident_per_triangle = l.resident as f64 / l.triangles.max(1) as f64;
    }
    Some(MeshReport {
        position_exponent: m.header.position_exponent,
        clas_truncate_bits: m.header.clas_truncate_bits,
        sections: m.sections.len(),
        parts: m.parts.len(),
        pages: m.pages.len(),
        pinned_pages: m.header.pinned_pages,
        clusters: m.pages.iter().map(|p| u32::from(p.cluster_count)).sum(),
        merge_level: (m.header.merge_level != MeshHeader::NO_MERGE_LEVEL)
            .then_some(m.header.merge_level),
        bytes_per_triangle: stored as f64 / triangles.max(1) as f64,
        foliage_cluster_fill: (foliage_clusters > 0)
            .then(|| foliage_tris as f64 / foliage_clusters as f64),
        levels,
    })
}
