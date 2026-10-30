use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::codec::decode_chunk;
use crate::format::{Layout, PAGE, hash_blob, read_header, validate};
use crate::{AssetId, Blob, Chunk, Class, Entry, Error, Hash32, Header, Result};

/// A 4096-byte-aligned file range covering stored bytes, for unbuffered reads. The stored
/// bytes start `skip` bytes into it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct StoredRange {
    pub offset: u64,
    pub len: u32,
    pub skip: u32,
}

/// Archive bytes with the alignment the tables need.
enum Bytes {
    Boxed(Box<[u8]>),
    Shared(Arc<[u8]>),
    Copied(Box<[u64]>, usize),
}

impl Bytes {
    fn boxed(b: Box<[u8]>) -> Self {
        if b.as_ptr().cast::<u64>().is_aligned() {
            Self::Boxed(b)
        } else {
            Self::copy(&b)
        }
    }

    fn shared(b: Arc<[u8]>) -> Self {
        if b.as_ptr().cast::<u64>().is_aligned() {
            Self::Shared(b)
        } else {
            Self::copy(&b)
        }
    }

    fn copy(b: &[u8]) -> Self {
        let mut words = vec![0u64; b.len().div_ceil(8)].into_boxed_slice();
        bytemuck::cast_slice_mut::<u64, u8>(&mut words)[..b.len()].copy_from_slice(b);
        Self::Copied(words, b.len())
    }

    fn get(&self) -> &[u8] {
        match self {
            Self::Boxed(b) => b,
            Self::Shared(b) => b,
            Self::Copied(w, n) => &bytemuck::cast_slice(w)[..*n],
        }
    }
}

/// A mounted archive: its index range in memory, plus either the open file or the whole
/// archive in memory. `Send + Sync`; every read takes `&self`.
pub struct Archive {
    bytes: Bytes,
    file: Option<(File, PathBuf)>,
    header: Header,
    layout: Layout,
}

impl Archive {
    /// Bytes to read first: enough to find the index range.
    pub const HEAD_LEN: usize = 4096;

    /// The index range (header, table of contents and mount section) from the first
    /// [`Archive::HEAD_LEN`] bytes, or fewer if the archive is shorter.
    pub fn index_range(head: &[u8]) -> Result<StoredRange> {
        let h = read_header(head)?;
        Ok(StoredRange {
            offset: 0,
            len: h.index_len,
            skip: 0,
        })
    }

    /// A file archive from the bytes of its index range. Opens the file with read and delete
    /// sharing, so a writer can replace it while it stays mounted.
    pub fn from_index(path: &Path, index: Box<[u8]>) -> Result<Archive> {
        Self::with_file(File::open(path)?, path, index)
    }

    /// A file archive from a handle the caller opened with read and delete sharing, and the
    /// bytes of its index range: the archive stays the file that handle names even after
    /// another file is renamed over `path`.
    pub fn from_index_file(file: File, path: &Path, index: Box<[u8]>) -> Result<Archive> {
        Self::with_file(file, path, index)
    }

    /// An archive held entirely in memory; reads never do IO.
    pub fn from_memory(bytes: Arc<[u8]>) -> Result<Archive> {
        let header = read_header(&bytes)?;
        if (bytes.len() as u64) < header.file_len {
            return Err(Error::Truncated {
                need: header.file_len,
                have: bytes.len() as u64,
            });
        }
        let bytes = Bytes::shared(bytes);
        let layout = validate(&header, &bytes.get()[..header.index_len as usize])?;
        Ok(Self {
            bytes,
            file: None,
            header,
            layout,
        })
    }

    /// Opens a file archive with two positioned reads: the head, then the index range.
    pub fn open(path: &Path) -> Result<Archive> {
        let file = File::open(path)?;
        let mut head = vec![0; Self::HEAD_LEN];
        let n = read_some(&file, 0, &mut head)?;
        let range = Self::index_range(&head[..n])?;
        let mut index = vec![0; range.len as usize].into_boxed_slice();
        read_exact(&file, 0, &mut index)?;
        Self::with_file(file, path, index)
    }

    fn with_file(file: File, path: &Path, index: Box<[u8]>) -> Result<Archive> {
        let header = read_header(&index)?;
        let len = file.metadata()?.len();
        if len < header.file_len {
            return Err(Error::Truncated {
                need: header.file_len,
                have: len,
            });
        }
        let layout = validate(&header, &index)?;
        Ok(Self {
            bytes: Bytes::boxed(index),
            file: Some((file, path.to_owned())),
            header,
            layout,
        })
    }

    /// The file path; `None` for a memory archive.
    pub fn path(&self) -> Option<&Path> {
        self.file.as_ref().map(|(_, p)| p.as_path())
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Every entry, sorted by id.
    pub fn entries(&self) -> &[Entry] {
        let l = self.layout;
        bytemuck::cast_slice(&self.bytes.get()[l.entries..l.blobs])
    }

    pub fn find(&self, id: AssetId) -> Option<&Entry> {
        let entries = self.entries();
        entries
            .binary_search_by_key(&id, |e| e.id)
            .ok()
            .map(|i| &entries[i])
    }

    /// The entry for `id`, checked to be of `class` at `schema`.
    pub fn expect(&self, id: AssetId, class: u16, schema: u16) -> Result<&Entry> {
        let e = self.find(id).ok_or(Error::Missing(id))?;
        if e.class != class {
            return Err(Error::Class {
                name: self.name(e).into(),
                found: e.class,
                expected: class,
            });
        }
        if e.schema != schema {
            return Err(Error::Schema {
                name: self.name(e).into(),
                found: e.schema,
                expected: schema,
            });
        }
        Ok(e)
    }

    pub fn blobs(&self, e: &Entry) -> &[Blob] {
        let l = self.layout;
        let all: &[Blob] = bytemuck::cast_slice(&self.bytes.get()[l.blobs..l.chunks]);
        &all[e.first_blob as usize..(e.first_blob + e.blob_count) as usize]
    }

    pub fn chunks(&self, b: &Blob) -> &[Chunk] {
        let l = self.layout;
        let all: &[Chunk] = bytemuck::cast_slice(&self.bytes.get()[l.chunks..l.names]);
        &all[b.first_chunk as usize..b.first_chunk as usize + usize::from(b.chunk_count)]
    }

    pub fn name(&self, e: &Entry) -> &str {
        let l = self.layout;
        let o = l.names + e.name.offset as usize;
        // The name table was checked to be UTF-8 and every name to lie on char boundaries.
        std::str::from_utf8(&self.bytes.get()[o..o + e.name.len as usize]).unwrap_or_default()
    }

    /// Archive offset and stored length of the whole blob or of one chunk.
    fn span(&self, b: &Blob, chunk: Option<u32>) -> (u64, u32) {
        match chunk {
            None => (b.offset, b.stored_len),
            Some(i) => {
                let cs = &self.chunks(b)[..=i as usize];
                let before: u64 = cs[..i as usize]
                    .iter()
                    .map(|c| u64::from(c.stored_len))
                    .sum();
                (b.offset + before, cs[i as usize].stored_len)
            }
        }
    }

    /// The 4096-byte-aligned range covering the stored bytes of a blob or one chunk. Panics if
    /// `chunk` is not below `b.chunk_count`.
    pub fn stored_range(&self, b: &Blob, chunk: Option<u32>) -> StoredRange {
        let (offset, len) = self.span(b, chunk);
        let start = offset / PAGE * PAGE;
        let end = (offset + u64::from(len)).next_multiple_of(PAGE);
        StoredRange {
            offset: start,
            len: (end - start) as u32,
            skip: (offset - start) as u32,
        }
    }

    /// Stored bytes in place, for memory archives and mount blobs; `None` otherwise. Panics if
    /// `chunk` is not below `b.chunk_count`.
    pub fn stored(&self, b: &Blob, chunk: Option<u32>) -> Option<&[u8]> {
        if self.file.is_some() && !b.is_mount() {
            return None;
        }
        let (offset, len) = self.span(b, chunk);
        Some(&self.bytes.get()[offset as usize..offset as usize + len as usize])
    }

    /// Decodes stored bytes the caller already has. `stored` starts at the stored bytes of the
    /// blob or chunk and may run past them. `dst` holds exactly the decoded length:
    /// `b.raw_len` for the whole blob, `b.chunk_len(i)` for chunk `i`.
    pub fn decode(
        &self,
        b: &Blob,
        chunk: Option<u32>,
        stored: &[u8],
        dst: &mut [u8],
    ) -> Result<()> {
        check_chunk(b, chunk)?;
        let (codec, filter) = (b.codec(), b.filter());
        let (_, len) = self.span(b, chunk);
        let stored = stored.get(..len as usize).ok_or(Error::BufferSize {
            need: len as usize,
            have: stored.len(),
        })?;
        match chunk {
            Some(i) => {
                check_len(dst, b.chunk_len(i) as usize)?;
                decode_chunk(codec, filter, stored, dst)
            }
            None => {
                check_len(dst, b.raw_len as usize)?;
                let (mut src, mut out) = (0, 0);
                for (i, c) in (0..).zip(self.chunks(b)) {
                    let (n, m) = (c.stored_len as usize, b.chunk_len(i) as usize);
                    decode_chunk(codec, filter, &stored[src..src + n], &mut dst[out..out + m])?;
                    src += n;
                    out += m;
                }
                Ok(())
            }
        }
    }

    /// Reads and decodes a whole blob into `dst` (`b.raw_len` bytes). Positioned reads on a
    /// file archive, none for memory archives and mount blobs.
    pub fn read(&self, b: &Blob, dst: &mut [u8]) -> Result<()> {
        self.read_part(b, None, dst)
    }

    /// Reads and decodes chunk `chunk` into `dst` (`b.chunk_len(chunk)` bytes).
    pub fn read_chunk(&self, b: &Blob, chunk: u32, dst: &mut [u8]) -> Result<()> {
        check_chunk(b, Some(chunk))?;
        self.read_part(b, Some(chunk), dst)
    }

    fn read_part(&self, b: &Blob, chunk: Option<u32>, dst: &mut [u8]) -> Result<()> {
        if let Some(s) = self.stored(b, chunk) {
            return self.decode(b, chunk, s, dst);
        }
        let (offset, len) = self.span(b, chunk);
        let mut stored = vec![0; len as usize];
        self.read_file(offset, &mut stored)?;
        self.decode(b, chunk, &stored, dst)
    }

    /// Reads the stored bytes of a blob as they are (`b.stored_len` bytes).
    pub fn read_stored(&self, b: &Blob, dst: &mut [u8]) -> Result<()> {
        check_len(dst, b.stored_len as usize)?;
        match self.stored(b, None) {
            Some(s) => dst.copy_from_slice(s),
            None => self.read_file(b.offset, dst)?,
        }
        Ok(())
    }

    fn read_file(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        let (file, _) = self.file.as_ref().expect("memory archives read in place");
        read_exact(file, offset, dst)
    }

    /// Decodes every blob of `e`, checks its content hash, and checks its schema when this
    /// crate defines the class.
    pub fn verify(&self, e: &Entry) -> Result<()> {
        if let Some(expected) = Class::from_u16(e.class).and_then(Class::schema)
            && e.schema != expected
        {
            return Err(Error::Schema {
                name: self.name(e).into(),
                found: e.schema,
                expected,
            });
        }
        let mut hasher = blake3::Hasher::new();
        let mut raw = Vec::new();
        for b in self.blobs(e) {
            raw.resize(b.raw_len as usize, 0);
            self.read(b, &mut raw)?;
            hash_blob(&mut hasher, &raw);
        }
        if Hash32::finish(&hasher) != e.content_hash {
            return Err(Error::ContentHash {
                name: self.name(e).into(),
            });
        }
        Ok(())
    }
}

fn check_chunk(b: &Blob, chunk: Option<u32>) -> Result<()> {
    match chunk {
        Some(i) if i >= u32::from(b.chunk_count) => {
            Err(Error::InvalidBlob("chunk index out of range"))
        }
        _ => Ok(()),
    }
}

fn check_len(dst: &[u8], need: usize) -> Result<()> {
    if dst.len() == need {
        Ok(())
    } else {
        Err(Error::BufferSize {
            need,
            have: dst.len(),
        })
    }
}

/// Positioned read; never moves a shared cursor, so many threads can read one `File`.
fn read_some(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        #[cfg(windows)]
        let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
        #[cfg(unix)]
        let n = file.read_at(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            break;
        }
        done += n;
    }
    Ok(done)
}

fn read_exact(file: &File, offset: u64, buf: &mut [u8]) -> Result<()> {
    let n = read_some(file, offset, buf)?;
    if n == buf.len() {
        Ok(())
    } else {
        Err(Error::Truncated {
            need: offset + buf.len() as u64,
            have: offset + n as u64,
        })
    }
}
