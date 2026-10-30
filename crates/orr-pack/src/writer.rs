use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::codec::split_planes;
use crate::format::{HEADER_LEN, Layout, MAGIC, MOUNT_ALIGN, PAGE, codec_byte, hash_blob};
use crate::{
    Archive, AssetId, Blob, Chunk, Codec, Entry, Error, FORMAT_VERSION, Filter, Hash32, Header,
    Result, StrRef,
};

/// One blob to add.
#[derive(Copy, Clone, Debug)]
pub struct BlobIn<'a> {
    pub bytes: &'a [u8],
    pub codec: Codec,
    pub filter: Filter,
    /// Store the blob in the mount section, resident from mount.
    pub mount: bool,
    /// Decoded chunk length: a power of two from 4096 up, or 0 for the default (256 KiB
    /// chunks for payload blobs over 256 KiB, else one chunk). Mount blobs are one chunk.
    pub chunk_len: u32,
}

impl<'a> BlobIn<'a> {
    /// A zstd payload blob with default chunking.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            codec: Codec::Zstd,
            filter: Filter::None,
            mount: false,
            chunk_len: 0,
        }
    }

    /// Compresses the blob as the writer would, on the calling thread, for
    /// [`EntryBlobs::push_encoded`].
    pub fn encode(&self, zstd_level: i32) -> Result<EncodedBlob<'a>> {
        let chunk = chunk_len(self)?;
        self.encode_chunked(chunk, raw_key(self, chunk), zstd_level)
    }

    fn encode_chunked(&self, chunk: u32, key: Hash32, zstd_level: i32) -> Result<EncodedBlob<'a>> {
        if self.codec == Codec::GDeflate {
            return Err(Error::InvalidBlob("GDeflate encoding is not available"));
        }
        let mut stored = Vec::new();
        let mut lens = Vec::new();
        let mut planes = Vec::new();
        let raw_chunks: Vec<&[u8]> = if self.bytes.is_empty() {
            vec![&[]]
        } else {
            self.bytes.chunks(chunk as usize).collect()
        };
        let (mut codec, mut filter) = (self.codec, self.filter);
        for c in &raw_chunks {
            let src = if filter == Filter::BytePlanes4 {
                split_planes(c, &mut planes);
                &planes[..]
            } else {
                c
            };
            let n = stored.len();
            match codec {
                Codec::Zstd => stored.extend(zstd::bulk::compress(src, zstd_level)?),
                _ => stored.extend_from_slice(src),
            }
            lens.push((stored.len() - n) as u32);
        }
        if codec == Codec::Zstd && stored.len() >= self.bytes.len() {
            (codec, filter) = (Codec::None, Filter::None);
            stored.clear();
            stored.extend_from_slice(self.bytes);
            lens = raw_chunks.iter().map(|c| c.len() as u32).collect();
        }
        Ok(EncodedBlob {
            rec: Blob {
                offset: 0,
                stored_len: 0,
                raw_len: self.bytes.len() as u32,
                first_chunk: 0,
                chunk_count: lens.len() as u16,
                codec: codec_byte(codec, filter),
                mount: u8::from(self.mount),
            },
            key,
            raw: self.bytes,
            stored,
            chunks: lens,
        })
    }
}

/// A blob compressed by [`BlobIn::encode`]; it borrows the decoded bytes for the entry's
/// content hash.
pub struct EncodedBlob<'a> {
    rec: Blob,
    /// Dedup key of the decoded bytes and settings.
    key: Hash32,
    raw: &'a [u8],
    stored: Vec<u8>,
    chunks: Vec<u32>,
}

impl EncodedBlob<'_> {
    pub fn stored_len(&self) -> usize {
        self.stored.len()
    }

    pub fn filter(&self) -> Filter {
        self.rec.filter()
    }
}

/// The blobs of an entry being added by [`ArchiveWriter::add_streamed`].
pub struct EntryBlobs<'w> {
    writer: &'w mut ArchiveWriter,
    content: &'w mut blake3::Hasher,
}

impl EntryBlobs<'_> {
    /// Encodes and appends a blob.
    pub fn push(&mut self, b: &BlobIn) -> Result<()> {
        let rec = self.writer.encode(b)?;
        hash_blob(self.content, b.bytes);
        self.writer.blobs.push(rec);
        Ok(())
    }

    /// Appends a blob encoded elsewhere.
    pub fn push_encoded(&mut self, e: &EncodedBlob) -> Result<()> {
        let rec = match self.writer.seen.get(&e.key) {
            Some(r) => *r,
            None => self
                .writer
                .place(e.rec, &e.stored, &e.chunks, Some(e.key))?,
        };
        hash_blob(self.content, e.raw);
        self.writer.blobs.push(rec);
        Ok(())
    }
}

/// Chunk length of a blob: one chunk for mount blobs, else as asked or by the default rule.
fn chunk_len(b: &BlobIn) -> Result<u32> {
    let raw_len =
        u32::try_from(b.bytes.len()).map_err(|_| Error::InvalidBlob("blob over 4 GiB"))?;
    let chunk = match b.chunk_len {
        _ if b.mount => raw_len.max(1),
        0 if raw_len > DEFAULT_CHUNK => DEFAULT_CHUNK,
        0 => raw_len.max(1),
        c if c.is_power_of_two() && u64::from(c) >= PAGE => c,
        _ => {
            return Err(Error::InvalidBlob(
                "chunk length is not a power of two from 4096",
            ));
        }
    };
    if raw_len.div_ceil(chunk) > u32::from(u16::MAX) {
        return Err(Error::InvalidBlob("more than 65535 chunks"));
    }
    Ok(chunk)
}

fn raw_key(b: &BlobIn, chunk: u32) -> Hash32 {
    let mut key = blake3::Hasher::new();
    key.update(b"raw");
    key.update(&[b.codec as u8, b.filter as u8, u8::from(b.mount)]);
    key.update(&chunk.to_le_bytes());
    key.update(b.bytes);
    Hash32::finish(&key)
}

/// Where [`ArchiveWriter::finish`] puts the archive.
#[derive(Clone, Debug)]
pub enum Target {
    /// Written to `<path>.tmp`, then renamed over the path.
    File(PathBuf),
    /// Returned as bytes; the same bytes a file target would hold.
    Memory,
}

const DEFAULT_CHUNK: u32 = 256 << 10;

enum Payload {
    Memory(Vec<u8>),
    File {
        out: BufWriter<File>,
        scratch: PathBuf,
        path: PathBuf,
        len: u64,
    },
}

/// Builds an archive. Mount blobs and payload blobs each land in call order, so the same
/// calls give the same bytes.
pub struct ArchiveWriter {
    payload: Option<Payload>,
    entries: Vec<Entry>,
    blobs: Vec<Blob>,
    chunks: Vec<Chunk>,
    names: String,
    mount: Vec<u8>,
    ids: HashSet<AssetId>,
    /// Blob records by a hash of their decoded bytes and settings, and by a hash of their
    /// stored bytes and layout.
    seen: HashMap<Hash32, Blob>,
    zstd_level: i32,
}

impl ArchiveWriter {
    pub fn create(target: Target) -> Result<Self> {
        let payload = match target {
            Target::Memory => Payload::Memory(Vec::new()),
            Target::File(path) => {
                let scratch = suffixed(&path, ".payload.tmp");
                let out = BufWriter::with_capacity(
                    1 << 20,
                    File::options()
                        .read(true)
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&scratch)?,
                );
                Payload::File {
                    out,
                    scratch,
                    path,
                    len: 0,
                }
            }
        };
        Ok(Self {
            payload: Some(payload),
            entries: Vec::new(),
            blobs: Vec::new(),
            chunks: Vec::new(),
            names: String::new(),
            mount: Vec::new(),
            ids: HashSet::new(),
            seen: HashMap::new(),
            zstd_level: 19,
        })
    }

    /// zstd compression level for blobs added from now on (default 19).
    pub fn set_zstd_level(&mut self, level: i32) {
        self.zstd_level = level;
    }

    /// Adds an asset named `name` and returns its id.
    pub fn add(
        &mut self,
        name: &str,
        class: u16,
        schema: u16,
        cook_key: Hash32,
        blobs: &[BlobIn],
    ) -> Result<AssetId> {
        self.add_streamed(name, class, schema, cook_key, |e| {
            blobs.iter().try_for_each(|b| e.push(b))
        })
    }

    /// Adds an asset whose blobs `push_blobs` pushes in order, so they need not all be in
    /// memory at once. On failure the entry and its blob records are dropped.
    pub fn add_streamed<E: From<Error>>(
        &mut self,
        name: &str,
        class: u16,
        schema: u16,
        cook_key: Hash32,
        push_blobs: impl FnOnce(&mut EntryBlobs<'_>) -> Result<(), E>,
    ) -> Result<AssetId, E> {
        let entry = Entry {
            id: AssetId::NONE,
            class,
            schema,
            flags: 0,
            content_hash: Hash32([0; 32]),
            cook_key,
            first_blob: 0,
            blob_count: 0,
            name: StrRef::default(),
            _reserved: [0; 6],
        };
        let mut content = blake3::Hasher::new();
        let first = self.blobs.len();
        let id = self.insert(name, entry, |w| {
            push_blobs(&mut EntryBlobs {
                writer: w,
                content: &mut content,
            })
        })?;
        let e = self.entries.last_mut().expect("just inserted");
        e.content_hash = Hash32::finish(&content);
        e.blob_count = (self.blobs.len() - first) as u32;
        Ok(id)
    }

    /// Copies an entry from another archive with its stored bytes as they are.
    pub fn copy_entry(&mut self, from: &Archive, e: &Entry) -> Result<()> {
        self.insert::<Error>(from.name(e), *e, |w| {
            let mut buf = Vec::new();
            for b in from.blobs(e) {
                let stored = match from.stored(b, None) {
                    Some(s) => s,
                    None => {
                        buf.resize(b.stored_len as usize, 0);
                        from.read_stored(b, &mut buf)?;
                        &buf
                    }
                };
                let chunks: Vec<u32> = from.chunks(b).iter().map(|c| c.stored_len).collect();
                let rec = w.place(*b, stored, &chunks, None)?;
                w.blobs.push(rec);
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Adds `entry` under `name` with the blob records `push_blobs` appends. On failure the
    /// entry and its blob records are dropped.
    fn insert<E: From<Error>>(
        &mut self,
        name: &str,
        entry: Entry,
        push_blobs: impl FnOnce(&mut Self) -> Result<(), E>,
    ) -> Result<AssetId, E> {
        let id = AssetId::from_name(name);
        if id == AssetId::NONE || self.ids.contains(&id) {
            return Err(Error::Duplicate(name.into()).into());
        }
        let first_blob = self.blobs.len();
        if let Err(e) = push_blobs(self) {
            self.blobs.truncate(first_blob);
            return Err(e);
        }
        self.ids.insert(id);
        self.entries.push(Entry {
            id,
            first_blob: first_blob as u32,
            name: StrRef {
                offset: self.names.len() as u32,
                len: name.len() as u32,
            },
            ..entry
        });
        self.names.push_str(name);
        Ok(id)
    }

    fn encode(&mut self, b: &BlobIn) -> Result<Blob> {
        let chunk = chunk_len(b)?;
        let key = raw_key(b, chunk);
        if let Some(rec) = self.seen.get(&key) {
            return Ok(*rec);
        }
        let e = b.encode_chunked(chunk, key, self.zstd_level)?;
        self.place(e.rec, &e.stored, &e.chunks, Some(e.key))
    }

    /// Stores encoded bytes once and returns the blob record pointing at them.
    fn place(
        &mut self,
        mut rec: Blob,
        stored: &[u8],
        chunks: &[u32],
        raw_key: Option<Hash32>,
    ) -> Result<Blob> {
        let mut key = blake3::Hasher::new();
        key.update(b"stored");
        key.update(&[rec.codec, rec.mount]);
        key.update(&rec.raw_len.to_le_bytes());
        key.update(bytemuck::cast_slice(chunks));
        key.update(stored);
        let key = Hash32::finish(&key);
        if let Some(&r) = self.seen.get(&key) {
            if let Some(k) = raw_key {
                self.seen.insert(k, r);
            }
            return Ok(r);
        }
        rec.stored_len =
            u32::try_from(stored.len()).map_err(|_| Error::InvalidBlob("blob over 4 GiB"))?;
        rec.first_chunk = self.chunks.len() as u32;
        rec.chunk_count = chunks.len() as u16;
        self.chunks
            .extend(chunks.iter().map(|&stored_len| Chunk { stored_len }));
        rec.offset = if rec.is_mount() {
            let at = self.mount.len().next_multiple_of(MOUNT_ALIGN);
            self.mount.resize(at, 0);
            self.mount.extend_from_slice(stored);
            at as u64
        } else {
            self.write_payload(stored)?
        };
        self.seen.insert(key, rec);
        if let Some(k) = raw_key {
            self.seen.insert(k, rec);
        }
        Ok(rec)
    }

    /// Appends to the payload and pads it to 4096 bytes; returns the payload-relative offset.
    fn write_payload(&mut self, bytes: &[u8]) -> Result<u64> {
        let pad = &[0; PAGE as usize][..bytes.len().next_multiple_of(PAGE as usize) - bytes.len()];
        Ok(match self.payload.as_mut().expect("writer is live") {
            Payload::Memory(v) => {
                let at = v.len() as u64;
                v.extend_from_slice(bytes);
                v.extend_from_slice(pad);
                at
            }
            Payload::File { out, len, .. } => {
                let at = *len;
                out.write_all(bytes)?;
                out.write_all(pad)?;
                *len += (bytes.len() + pad.len()) as u64;
                at
            }
        })
    }

    /// Writes the index range and completes the archive. A file target is written to
    /// `<path>.tmp`, synced and renamed over the path; readers that opened the old file with
    /// delete sharing keep reading the old bytes. Returns the bytes for a memory target.
    pub fn finish(mut self) -> Result<Option<Arc<[u8]>>> {
        let l = Layout::new(
            self.entries.len(),
            self.blobs.len(),
            self.chunks.len(),
            self.names.len(),
            self.mount.len(),
        );
        for b in &mut self.blobs {
            b.offset += if b.is_mount() {
                l.mount as u64
            } else {
                l.index_len as u64
            };
        }
        self.entries.sort_unstable_by_key(|e| e.id);

        let mut index = vec![0u8; l.index_len];
        index[l.entries..l.blobs].copy_from_slice(bytemuck::cast_slice(&self.entries));
        index[l.blobs..l.chunks].copy_from_slice(bytemuck::cast_slice(&self.blobs));
        index[l.chunks..l.names].copy_from_slice(bytemuck::cast_slice(&self.chunks));
        index[l.names..l.names + self.names.len()].copy_from_slice(self.names.as_bytes());
        index[l.mount..l.mount + self.mount.len()].copy_from_slice(&self.mount);

        let payload = self.payload.take().expect("writer is live");
        let payload_len = match &payload {
            Payload::Memory(v) => v.len() as u64,
            Payload::File { len, .. } => *len,
        };
        let header = Header {
            magic: MAGIC,
            version: FORMAT_VERSION,
            index_len: l.index_len as u32,
            file_len: l.index_len as u64 + payload_len,
            entry_count: self.entries.len() as u32,
            blob_count: self.blobs.len() as u32,
            chunk_count: self.chunks.len() as u32,
            names_len: self.names.len() as u32,
            mount_offset: l.mount as u32,
            mount_len: self.mount.len() as u32,
            toc_hash: Hash32::of(&index[HEADER_LEN..l.mount]),
            _reserved: [0; 12],
        };
        index[..HEADER_LEN].copy_from_slice(bytemuck::bytes_of(&header));

        match payload {
            Payload::Memory(v) => {
                index.extend_from_slice(&v);
                Ok(Some(index.into()))
            }
            Payload::File {
                out, scratch, path, ..
            } => {
                let result = write_file(&path, &index, out);
                let _ = fs::remove_file(&scratch);
                result.map(|()| None)
            }
        }
    }
}

impl Drop for ArchiveWriter {
    fn drop(&mut self) {
        if let Some(Payload::File { scratch, .. }) = self.payload.take() {
            let _ = fs::remove_file(scratch);
        }
    }
}

fn write_file(path: &Path, index: &[u8], payload: BufWriter<File>) -> Result<()> {
    let mut payload = payload
        .into_inner()
        .map_err(io::IntoInnerError::into_error)?;
    payload.rewind()?;
    let tmp = suffixed(path, ".tmp");
    let mut out = File::create(&tmp)?;
    out.write_all(index)?;
    io::copy(&mut payload, &mut out)?;
    out.sync_all()?;
    drop(out);
    fs::rename(&tmp, path).map_err(|source| Error::Replace {
        path: path.to_owned(),
        source,
    })
}

fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut s = OsString::from(path.as_os_str());
    s.push(suffix);
    s.into()
}
