use bytemuck::{Pod, Zeroable};

use crate::{AssetId, Codec, Error, Filter, Hash32, Result};

pub const FORMAT_VERSION: u32 = 1;
pub(crate) const MAGIC: [u8; 8] = *b"ORRPACK\0";
pub(crate) const PAGE: u64 = 4096;
pub(crate) const MOUNT_ALIGN: usize = 16;

/// First 128 bytes of an archive. The table of contents follows in this order: entries,
/// blobs, chunks, names; then the mount section at 16-byte alignment, then zeros up to
/// `index_len`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Header {
    pub magic: [u8; 8],
    pub version: u32,
    /// Length of the index range at offset 0, a multiple of 4096.
    pub index_len: u32,
    /// Length of the whole archive, a multiple of 4096.
    pub file_len: u64,
    pub entry_count: u32,
    pub blob_count: u32,
    pub chunk_count: u32,
    pub names_len: u32,
    pub mount_offset: u32,
    pub mount_len: u32,
    /// blake3 of the bytes from the end of the header to `mount_offset`.
    pub toc_hash: Hash32,
    pub _reserved: [u32; 12],
}

/// A string in the name table.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct StrRef {
    pub offset: u32,
    pub len: u32,
}

/// One asset in the table of contents (128 bytes). Entries are sorted by `id`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Entry {
    pub id: AssetId,
    pub class: u16,
    pub schema: u16,
    pub flags: u32,
    /// blake3 over every blob in order: its decoded length as a little-endian u64, then its
    /// decoded bytes.
    pub content_hash: Hash32,
    /// Hash of everything the cook read to produce the entry.
    pub cook_key: Hash32,
    pub first_blob: u32,
    pub blob_count: u32,
    pub name: StrRef,
    pub _reserved: [u32; 6],
}

/// One blob (24 bytes). Payload blobs start on a 4096-byte boundary; mount blobs sit in the
/// index range at 16-byte alignment. Identical blobs share their stored bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Blob {
    /// Offset of the stored bytes in the archive.
    pub offset: u64,
    pub stored_len: u32,
    /// Decoded length.
    pub raw_len: u32,
    pub first_chunk: u32,
    pub chunk_count: u16,
    /// Bits 0-3 [`Codec`], bits 4-7 [`Filter`].
    pub codec: u8,
    /// 1 for a blob in the mount section.
    pub mount: u8,
}

/// Stored length of one chunk. Chunks are packed back to back inside their blob.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Chunk {
    pub stored_len: u32,
}

const _: () = {
    assert!(size_of::<Header>() == 128);
    assert!(size_of::<Entry>() == 128);
    assert!(size_of::<Blob>() == 24);
};

impl Blob {
    pub fn codec(&self) -> Codec {
        match self.codec & 0xF {
            1 => Codec::Zstd,
            2 => Codec::GDeflate,
            _ => Codec::None,
        }
    }

    pub fn filter(&self) -> Filter {
        match self.codec >> 4 {
            1 => Filter::BytePlanes4,
            _ => Filter::None,
        }
    }

    pub fn is_mount(&self) -> bool {
        self.mount != 0
    }

    /// Decoded length of every chunk but the last, which may be shorter. Chunk lengths are
    /// powers of two, so this follows from `raw_len` and `chunk_count`.
    pub fn chunk_raw_len(&self) -> u32 {
        if self.chunk_count <= 1 {
            self.raw_len
        } else {
            self.raw_len
                .div_ceil(u32::from(self.chunk_count))
                .next_power_of_two()
        }
    }

    /// Decoded length of chunk `i`.
    pub fn chunk_len(&self, i: u32) -> u32 {
        let c = self.chunk_raw_len();
        self.raw_len.saturating_sub(i.saturating_mul(c)).min(c)
    }
}

pub(crate) fn codec_byte(codec: Codec, filter: Filter) -> u8 {
    codec as u8 | (filter as u8) << 4
}

pub(crate) const HEADER_LEN: usize = size_of::<Header>();

/// Byte offsets of the tables inside the index range.
#[derive(Copy, Clone)]
pub(crate) struct Layout {
    pub(crate) entries: usize,
    pub(crate) blobs: usize,
    pub(crate) chunks: usize,
    pub(crate) names: usize,
    pub(crate) mount: usize,
    pub(crate) index_len: usize,
}

impl Layout {
    pub(crate) fn new(
        entries: usize,
        blobs: usize,
        chunks: usize,
        names: usize,
        mount: usize,
    ) -> Self {
        let e = HEADER_LEN;
        let b = e + entries * size_of::<Entry>();
        let c = b + blobs * size_of::<Blob>();
        let n = c + chunks * size_of::<Chunk>();
        let m = (n + names).next_multiple_of(MOUNT_ALIGN);
        Self {
            entries: e,
            blobs: b,
            chunks: c,
            names: n,
            mount: m,
            index_len: (m + mount).next_multiple_of(PAGE as usize),
        }
    }

    fn of(h: &Header) -> Self {
        Self::new(
            h.entry_count as usize,
            h.blob_count as usize,
            h.chunk_count as usize,
            h.names_len as usize,
            h.mount_len as usize,
        )
    }
}

/// Adds one blob to an entry's content hash: its decoded length as a little-endian u64, then
/// its decoded bytes.
pub(crate) fn hash_blob(h: &mut blake3::Hasher, raw: &[u8]) {
    h.update(&(raw.len() as u64).to_le_bytes());
    h.update(raw);
}

/// Reads the header from the first bytes of an archive.
pub(crate) fn read_header(head: &[u8]) -> Result<Header> {
    let Some(bytes) = head.get(..HEADER_LEN) else {
        return Err(Error::Truncated {
            need: HEADER_LEN as u64,
            have: head.len() as u64,
        });
    };
    let h: Header = bytemuck::pod_read_unaligned(bytes);
    if h.magic != MAGIC {
        return Err(Error::NotArchive);
    }
    if h.version != FORMAT_VERSION {
        return Err(Error::Version {
            found: h.version,
            expected: FORMAT_VERSION,
        });
    }
    if h.index_len == 0
        || !u64::from(h.index_len).is_multiple_of(PAGE)
        || !h.file_len.is_multiple_of(PAGE)
        || h.file_len < u64::from(h.index_len)
    {
        return Err(Error::Malformed("index or file length"));
    }
    Ok(h)
}

/// Checks the table of contents of a complete index range against its header and returns
/// its layout.
pub(crate) fn validate(h: &Header, index: &[u8]) -> Result<Layout> {
    if index.len() != h.index_len as usize {
        return Err(Error::Truncated {
            need: h.index_len.into(),
            have: index.len() as u64,
        });
    }
    let l = Layout::of(h);
    if l.mount != h.mount_offset as usize || l.index_len != index.len() {
        return Err(Error::CorruptToc);
    }
    if Hash32::of(&index[HEADER_LEN..l.mount]) != h.toc_hash {
        return Err(Error::CorruptToc);
    }
    let entries: &[Entry] = bytemuck::cast_slice(&index[l.entries..l.blobs]);
    let blobs: &[Blob] = bytemuck::cast_slice(&index[l.blobs..l.chunks]);
    let chunks: &[Chunk] = bytemuck::cast_slice(&index[l.chunks..l.names]);
    let names = std::str::from_utf8(&index[l.names..l.names + h.names_len as usize])
        .map_err(|_| Error::Malformed("name table is not UTF-8"))?;

    if !entries.windows(2).all(|w| w[0].id < w[1].id) {
        return Err(Error::Malformed("entries not sorted by id"));
    }
    for e in entries {
        let end = u64::from(e.first_blob) + u64::from(e.blob_count);
        if end > blobs.len() as u64 {
            return Err(Error::Malformed("entry blob range"));
        }
        let (o, n) = (e.name.offset as usize, e.name.len as usize);
        if names.get(o..o + n).is_none() {
            return Err(Error::Malformed("entry name"));
        }
    }
    let mount = u64::from(h.mount_offset)..u64::from(h.mount_offset) + u64::from(h.mount_len);
    for b in blobs {
        let first = b.first_chunk as usize;
        let Some(cs) = chunks.get(first..first + usize::from(b.chunk_count)) else {
            return Err(Error::Malformed("blob chunk range"));
        };
        if b.chunk_count == 0
            || b.codec & 0xF > Codec::GDeflate as u8
            || b.codec >> 4 > Filter::BytePlanes4 as u8
            || b.mount > 1
        {
            return Err(Error::Malformed("blob record"));
        }
        let c = u64::from(b.chunk_raw_len());
        let n = u64::from(b.chunk_count);
        if n > 1 && (!c.is_power_of_two() || (n - 1) * c >= u64::from(b.raw_len)) {
            return Err(Error::Malformed("blob chunk count"));
        }
        if cs.iter().map(|c| u64::from(c.stored_len)).sum::<u64>() != u64::from(b.stored_len) {
            return Err(Error::Malformed("chunk lengths do not sum to the blob"));
        }
        if b.codec() == Codec::GDeflate && cs.iter().any(|c| !c.stored_len.is_multiple_of(4)) {
            return Err(Error::Malformed(
                "GDeflate chunk length not a multiple of 4",
            ));
        }
        let end = b.offset + u64::from(b.stored_len);
        let placed = if b.is_mount() {
            b.offset >= mount.start && end <= mount.end
        } else {
            b.offset.is_multiple_of(PAGE) && b.offset >= u64::from(h.index_len) && end <= h.file_len
        };
        if !placed {
            return Err(Error::Malformed("blob outside its section"));
        }
    }
    Ok(l)
}
