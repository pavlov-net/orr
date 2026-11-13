//! The disk pipeline cache over Metal 4 pipeline archives. The device's compiler records every
//! pipeline it compiles in a pipeline data set serializer; saving writes those as an archive into
//! the same file the Vulkan backend writes (`cache_file`). Its payload starts with the archive's
//! own header, as Vulkan's starts with the driver's. Loading checks both, writes the archive
//! beside the cache file for Metal to open, and looks pipelines up in it: a pipeline the archive
//! holds is a hit. Hits compile with the archive to look binaries up in, so the serializer
//! records them too and a save keeps them.
//!
//! Serializing empties the serializer, so a save first compiles again, from the last saved
//! archive, the pipelines an earlier save took out of it. A saved file holds the pipelines this
//! session created, not ones the loaded archive held that it never asked for.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, RwLock};

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSProcessInfo, NSString, NSURL};
use objc2_metal::{
    MTL4Archive, MTL4Compiler, MTL4CompilerTaskOptions, MTL4ComputePipelineDescriptor,
    MTL4PipelineDataSetSerializer, MTL4PipelineDataSetSerializerConfiguration,
    MTL4PipelineDataSetSerializerDescriptor, MTL4PipelineDescriptor, MTLComputePipelineState,
    MTLDevice, MTLRenderPipelineState,
};

use super::device::{Device, Shared, metal_error};
use super::{Obj, lock};
use crate::cache_file::{Identity, decode, encode, mark_used, write};
use crate::{Error, PipelineCacheReport};

/// Apple's PCI vendor ID.
const APPLE: u32 = 0x106b;
/// The archive header: its size, version, vendor, device and the cache UUID, laid out as
/// Vulkan's `VkPipelineCacheHeaderVersionOne`.
const ARCHIVE_HEADER: usize = 32;
const ARCHIVE_VERSION: u32 = 1;

type Archive = ProtocolObject<dyn MTL4Archive>;
pub(super) type Serializer = ProtocolObject<dyn MTL4PipelineDataSetSerializer>;
type ComputeState = ProtocolObject<dyn MTLComputePipelineState>;
type RenderState = ProtocolObject<dyn MTLRenderPipelineState>;

fn word(hash: &blake3::Hash) -> u32 {
    u32::from_le_bytes(*hash.as_bytes().first_chunk().unwrap())
}

/// The device's identity: the GPU by name, and the OS, whose Metal compiler builds the binaries.
fn identity(device: &ProtocolObject<dyn MTLDevice>) -> Identity {
    let name = device.name().to_string();
    let os = NSProcessInfo::processInfo()
        .operatingSystemVersionString()
        .to_string();
    let uuid = blake3::hash(format!("orr Metal pipeline archive\0{name}\0{os}").as_bytes());
    Identity {
        vendor: APPLE,
        device: word(&blake3::hash(name.as_bytes())),
        driver_version: word(&blake3::hash(os.as_bytes())),
        cache_uuid: *uuid.as_bytes().first_chunk().unwrap(),
    }
}

/// The archive header for `id`.
fn archive_header(id: &Identity) -> [u8; ARCHIVE_HEADER] {
    let mut h = [0; ARCHIVE_HEADER];
    for (i, v) in [ARCHIVE_HEADER as u32, ARCHIVE_VERSION, id.vendor, id.device]
        .into_iter()
        .enumerate()
    {
        h[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    h[16..].copy_from_slice(&id.cache_uuid);
    h
}

fn url(path: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
}

/// `path` with `.{pid}.{what}` appended: a file of this process beside it.
fn beside(path: &Path, what: &str) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(format!(".{}.{what}", std::process::id()));
    PathBuf::from(p)
}

/// A pipeline this session created, to compile again when a save has emptied the serializer.
/// The descriptors are not changed once made.
pub(super) enum Record {
    Compute(Obj<MTL4ComputePipelineDescriptor>),
    Render(Obj<MTL4PipelineDescriptor>),
}

/// What the last save wrote.
#[derive(Default)]
struct Saved {
    /// The misses counted then.
    misses: u32,
    /// The records then, which that save took out of the serializer.
    flushed: usize,
    /// How many archives this session has saved, which names their files.
    archives: u32,
}

pub(super) struct PipelineCache {
    path: Option<PathBuf>,
    identity: Identity,
    /// The archive pipelines are looked up in, the loaded or last saved one, and the file
    /// Metal reads it from.
    archive: Mutex<Option<(Obj<Archive>, PathBuf)>>,
    /// This session's pipelines in creation order; with a path only.
    records: Mutex<Vec<Record>>,
    /// Held to read by pipeline creation, so a save, which holds it to write, sees every
    /// pipeline the serializer recorded among the records.
    saving: RwLock<()>,
    /// Records what the compiler compiles; with a path only.
    serializer: Option<Obj<Serializer>>,
    loaded: bool,
    rejected: Option<String>,
    hits: AtomicU32,
    misses: AtomicU32,
    /// Held while saving.
    saved: Mutex<Saved>,
    /// This session's pipelines by key.
    computes: Mutex<HashMap<blake3::Hash, Obj<ComputeState>>>,
    renders: Mutex<HashMap<blake3::Hash, Obj<RenderState>>>,
    /// The keys of the pipelines being created now: a thread asking for one of them waits,
    /// then finds it among this session's.
    compiling: Mutex<HashSet<blake3::Hash>>,
    compiled: Condvar,
}

/// Marks a key as being created until dropped.
struct CompileGuard<'a> {
    cache: &'a PipelineCache,
    key: blake3::Hash,
}

impl Drop for CompileGuard<'_> {
    fn drop(&mut self) {
        lock(&self.cache.compiling).remove(&self.key);
        self.cache.compiled.notify_all();
    }
}

impl PipelineCache {
    /// The serializer the device's compiler records into, when the cache has a path.
    pub(super) fn serializer(
        device: &ProtocolObject<dyn MTLDevice>,
        path: Option<&Path>,
    ) -> Option<Obj<Serializer>> {
        path?;
        let desc = MTL4PipelineDataSetSerializerDescriptor::new();
        // Binaries alone: with descriptors captured too, serializing an archive fails
        // without an error (macOS 27.0.1).
        desc.setConfiguration(MTL4PipelineDataSetSerializerConfiguration::CaptureBinaries);
        Some(Obj(device.newPipelineDataSetSerializerWithDescriptor(&desc)))
    }

    /// Opens the cache at `path`. A missing, damaged or foreign file starts it empty; so does one
    /// whose archive was written for another device or OS, which the next save replaces, or one
    /// Metal refuses, which is removed.
    pub(super) fn open(
        device: &ProtocolObject<dyn MTLDevice>,
        path: Option<&Path>,
        serializer: Option<Obj<Serializer>>,
    ) -> PipelineCache {
        let mut cache = PipelineCache {
            path: path.map(Path::to_path_buf),
            identity: identity(device),
            archive: Mutex::default(),
            records: Mutex::default(),
            saving: RwLock::default(),
            serializer,
            loaded: false,
            rejected: None,
            hits: AtomicU32::new(0),
            misses: AtomicU32::new(0),
            saved: Mutex::default(),
            computes: Mutex::default(),
            renders: Mutex::default(),
            compiling: Mutex::default(),
            compiled: Condvar::new(),
        };
        let Some(path) = path else { return cache };
        let file = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return cache,
            Err(e) => {
                cache.rejected = Some(format!("unreadable: {e}"));
                return cache;
            }
        };
        let archive = match decode(&file, &cache.identity) {
            Ok(payload) => payload,
            Err(why) => {
                log::info!("pipeline cache {}: file ignored ({why})", path.display());
                cache.rejected = Some(why);
                return cache;
            }
        };
        let header = archive_header(&cache.identity);
        let Some(binaries) = archive.strip_prefix(&header[..]) else {
            let why = "its archive was written for another device or OS".to_owned();
            log::warn!("pipeline cache {}: {why}", path.display());
            cache.rejected = Some(why);
            return cache;
        };
        let file = beside(path, "mtl4archive");
        let opened = std::fs::write(&file, binaries)
            .map_err(|e| e.to_string())
            .and_then(|()| {
                device
                    .newArchiveWithURL_error(&url(&file))
                    .map_err(|e| e.localizedDescription().to_string())
            });
        match opened {
            Ok(a) => {
                log::info!(
                    "pipeline cache {}: {} bytes loaded",
                    path.display(),
                    binaries.len()
                );
                cache.archive = Mutex::new(Some((Obj(a), file)));
                cache.loaded = true;
                mark_used(path);
            }
            Err(e) => {
                let _ = std::fs::remove_file(&file);
                let why = format!("Metal refused its archive ({e})");
                log::warn!("pipeline cache {}: {why}", path.display());
                cache.rejected = Some(why);
                let _ = std::fs::remove_file(path);
            }
        }
        cache
    }

    /// Waits until no other thread is creating `key`, then marks it as being created.
    fn begin_compile(&self, key: blake3::Hash) -> CompileGuard<'_> {
        let compiling = lock(&self.compiling);
        self.compiled
            .wait_while(compiling, |c| c.contains(&key))
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key);
        CompileGuard { cache: self, key }
    }

    /// The pipeline of `key`: this session's, else compiled, with the archive to look its
    /// binary up in when the archive holds it (a hit). `from_archive` asks the archive alone;
    /// `compile` compiles with the given options. With `cached_only`, a pipeline neither
    /// holds is refused. `record` describes it for later saves.
    fn get<T: Message>(
        &self,
        key: blake3::Hash,
        cached_only: bool,
        record: impl FnOnce() -> Record,
        session: &Mutex<HashMap<blake3::Hash, Obj<T>>>,
        from_archive: impl FnOnce(&Archive) -> Option<Retained<T>>,
        compile: impl FnOnce(Option<&MTL4CompilerTaskOptions>) -> Result<Retained<T>, Error>,
    ) -> Result<Obj<T>, Error> {
        let counted = self.path.is_some();
        let _guard = self.begin_compile(key);
        if let Some(p) = lock(session).get(&key) {
            if counted {
                self.hits.fetch_add(1, Ordering::Release);
            }
            return Ok(p.clone());
        }
        let _saving = self
            .saving
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let archive = lock(&self.archive).as_ref().map(|(a, _)| a.clone());
        let archived = archive
            .as_ref()
            .and_then(|a| from_archive(a).map(|p| (a, p)));
        let hit = archived.is_some();
        let raw = match archived {
            Some((archive, found)) => {
                let options = MTL4CompilerTaskOptions::new();
                options.setLookupArchives(Some(&NSArray::from_slice(&[&**archive])));
                // Compiled again so the serializer records it, taking the binary from the
                // archive; the archive's own copy serves when that fails.
                compile(Some(&options)).unwrap_or(found)
            }
            None if cached_only => return Err(Error::CompileRequired),
            None => compile(None)?,
        };
        // Counted once the serializer holds the pipeline, so a save that sees the count
        // writes it.
        if counted {
            lock(&self.records).push(record());
            if hit { &self.hits } else { &self.misses }.fetch_add(1, Ordering::Release);
        }
        let raw = Obj(raw);
        lock(session).insert(key, raw.clone());
        Ok(raw)
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        let archive = self.archive.get_mut().map(Option::take);
        if let Ok(Some((archive, file))) = archive {
            drop(archive);
            let _ = std::fs::remove_file(file);
        }
    }
}

impl Shared {
    /// The compute pipeline of `key` through the cache; `compile` compiles it with the given
    /// options, and `from_archive` asks the archive alone.
    pub(super) fn cached_compute(
        &self,
        key: blake3::Hash,
        cached_only: bool,
        from_archive: impl FnOnce(&Archive) -> Option<Retained<ComputeState>>,
        compile: impl FnOnce(Option<&MTL4CompilerTaskOptions>) -> Result<Retained<ComputeState>, Error>,
        desc: &Retained<MTL4ComputePipelineDescriptor>,
    ) -> Result<Obj<ComputeState>, Error> {
        let c = &self.pipeline_cache;
        let record = || Record::Compute(Obj(desc.clone()));
        c.get(key, cached_only, record, &c.computes, from_archive, compile)
    }

    /// The render pipeline of `key` through the cache, as [`Shared::cached_compute`].
    pub(super) fn cached_render(
        &self,
        key: blake3::Hash,
        cached_only: bool,
        from_archive: impl FnOnce(&Archive) -> Option<Retained<RenderState>>,
        compile: impl FnOnce(Option<&MTL4CompilerTaskOptions>) -> Result<Retained<RenderState>, Error>,
        desc: Retained<MTL4PipelineDescriptor>,
    ) -> Result<Obj<RenderState>, Error> {
        let c = &self.pipeline_cache;
        let record = || Record::Render(Obj(desc));
        c.get(key, cached_only, record, &c.renders, from_archive, compile)
    }
}

impl Device {
    /// Writes the pipeline cache to the path given at device creation, replacing the file
    /// whole: every pipeline compiled since the device opened, hits included. Does nothing
    /// without a path, or when no pipeline has missed the cache since the device opened it or
    /// last wrote it. Safe to call from several threads and processes at once: each writes a
    /// whole file and the last one stays.
    pub fn save_pipeline_cache(&self) -> Result<(), Error> {
        let _zone = crate::zone!("save pipeline cache");
        let cache = &self.shared.pipeline_cache;
        let (Some(path), Some(serializer)) = (&cache.path, &cache.serializer) else {
            return Ok(());
        };
        let mut saved = lock(&cache.saved);
        let misses = cache.misses.load(Ordering::Acquire);
        if misses == saved.misses {
            return Ok(());
        }
        // The archive is serialized beside the file, before `write` would create its directory.
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| Error::Io {
                path: dir.to_path_buf(),
                source: e,
            })?;
        }
        let _saving = cache
            .saving
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let records = lock(&cache.records);
        // The pipelines an earlier save took out of the serializer, compiled again from the
        // archive it wrote: quick, and recorded again.
        if saved.flushed > 0 {
            let options = MTL4CompilerTaskOptions::new();
            if let Some((archive, _)) = &*lock(&cache.archive) {
                options.setLookupArchives(Some(&NSArray::from_slice(&[&**archive])));
            }
            let compiler = &self.shared.compiler;
            for r in &records[..saved.flushed] {
                let again = match r {
                    Record::Compute(d) => compiler
                        .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(
                            d,
                            Some(&options),
                        )
                        .map(drop),
                    Record::Render(d) => compiler
                        .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(
                            d,
                            Some(&options),
                        )
                        .map(drop),
                };
                if let Err(e) = again {
                    log::warn!("pipeline cache: compiling a saved pipeline again failed: {e}");
                }
            }
        }
        let file = beside(path, &format!("{}.mtl4archive", saved.archives + 1));
        let serialized = serializer
            .serializeAsArchiveAndFlushToURL_error(&url(&file))
            .map_err(|e| metal_error("serializeAsArchiveAndFlushToURL", &e));
        let binaries = serialized.and_then(|()| {
            std::fs::read(&file).map_err(|e| Error::Io {
                path: file.clone(),
                source: e,
            })
        });
        let binaries = match binaries {
            Ok(b) => b,
            Err(e) => {
                let _ = std::fs::remove_file(&file);
                return Err(e);
            }
        };
        let mut payload = archive_header(&cache.identity).to_vec();
        payload.extend_from_slice(&binaries);
        let bytes = encode(&cache.identity, &payload);
        write(path, &bytes)?;
        // Later lookups, and the next save, read the archive just written.
        match self.shared.device.newArchiveWithURL_error(&url(&file)) {
            Ok(a) => {
                let old = lock(&cache.archive).replace((Obj(a), file));
                if let Some((old, old_file)) = old {
                    drop(old);
                    let _ = std::fs::remove_file(old_file);
                }
            }
            Err(e) => {
                let _ = std::fs::remove_file(&file);
                log::warn!(
                    "pipeline cache: Metal refused the archive just saved: {}",
                    e.localizedDescription()
                );
            }
        }
        saved.misses = misses;
        saved.flushed = records.len();
        saved.archives += 1;
        log::debug!(
            "pipeline cache {}: saved {} bytes",
            path.display(),
            bytes.len()
        );
        Ok(())
    }

    pub fn pipeline_cache_report(&self) -> PipelineCacheReport {
        let c = &self.shared.pipeline_cache;
        PipelineCacheReport {
            loaded: c.loaded,
            hits: c.hits.load(Ordering::Relaxed),
            misses: c.misses.load(Ordering::Relaxed),
            rejected: c.rejected.clone(),
        }
    }
}
