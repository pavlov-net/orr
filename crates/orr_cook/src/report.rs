//! The cook report: what each asset read and wrote, how each entry was obtained, and sizes.

use std::path::PathBuf;

use orr_pack::mesh::MeshInfo;
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, Class, Entry};
use serde::Serialize;

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

#[derive(Clone, Debug, Serialize)]
pub struct MeshReport {
    pub position_exponent: i8,
    pub clas_truncate_bits: u8,
    pub sections: usize,
    pub parts: usize,
    pub levels: usize,
    pub pages: usize,
    pub pinned_pages: u32,
    pub clusters: u32,
    /// Stored bytes per source triangle.
    pub bytes_per_triangle: f64,
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
        .map(|m| {
            let clusters: u32 = m.pages.iter().map(|p| u32::from(p.cluster_count)).sum();
            MeshReport {
                position_exponent: m.header.position_exponent,
                clas_truncate_bits: m.header.clas_truncate_bits,
                sections: m.sections.len(),
                parts: m.parts.len(),
                levels: m.levels.len(),
                pages: m.pages.len(),
                pinned_pages: m.header.pinned_pages,
                clusters,
                bytes_per_triangle: stored as f64 / triangles.max(1) as f64,
            }
        });
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
    }
}
