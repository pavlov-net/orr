//! The disk pipeline cache: one `VkPipelineCache` that every pipeline compiles through. The
//! file (`cache_file`) holds the device identity, the cache data and a checksum; a file from
//! another device, driver or format, or one the driver refuses or ignores, is rebuilt.
//!
//! The pipeline store is the same thing per pipeline: a directory of those files named by what
//! the pipeline compiles from. A process writes only the pipelines it compiled, each into its own
//! file as it finishes and adding to what the file holds, so processes share a store without
//! erasing each other's work, a killed process keeps what it finished, and an unchanged pipeline
//! keeps its file whatever else changes. A kill while writing leaves a bad checksum, which reads
//! as a miss.
//!
//! A device merges the whole store into one `VkPipelineCache` before it creates any pipeline, and
//! creates pipelines through that. The order comes from measurement, not from Vulkan: on an RTX
//! 5090 with driver 621.14 every pipeline hit from a store merged before the device's first
//! pipeline, while files loaded only when their pipelines were asked for left many path and media
//! kernels missing. Other drivers may differ. A miss costs only a compile: the driver decides
//! every hit by its own keys, so nothing here can change what a pipeline computes.
//!
//! A process waits for a pipeline another one is compiling, on the lock of its file, so that it
//! hits instead of compiling too. The system drops the lock of a process that dies. A wait pays
//! only if the file then hits, so nothing waits on a store that was empty at open, and the first
//! file from another process that misses ends waiting for the rest of the process.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::fs::{File, TryLockError};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use ash::vk;

use super::device::{Device, Shared};
use super::pipeline::StageParts;
use crate::cache_file::{Identity, decode, encode, write};
use crate::{Error, PipelineCacheReport};

/// The bytes `vkGetPipelineCacheData` would write for `cache`.
fn data_size(device: &ash::Device, cache: vk::PipelineCache) -> Option<usize> {
    let mut size = 0;
    // SAFETY: the cache is live, and a null data pointer asks only for the size.
    let queried = unsafe {
        (device.fp_v1_0().get_pipeline_cache_data)(
            device.handle(),
            cache,
            &mut size,
            std::ptr::null_mut(),
        )
    };
    (queried == vk::Result::SUCCESS).then_some(size)
}

/// Running out of memory says nothing about a cache file, which stays for the next run.
const OUT_OF_MEMORY: [vk::Result; 2] = [
    vk::Result::ERROR_OUT_OF_HOST_MEMORY,
    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
];

/// How often a process waiting for another's compile checks the lock. It polls so that any of
/// its threads can end the wait for all of them.
const WAIT_POLL: Duration = Duration::from_millis(50);

fn stored_path(dir: &Path, key: blake3::Hash) -> PathBuf {
    dir.join(key.to_hex().as_str()).with_extension("bin")
}

/// `data`, empty or from a file `decode` accepted, as a pipeline cache.
fn create_cache(device: &ash::Device, data: &[u8]) -> Result<vk::PipelineCache, vk::Result> {
    let info = vk::PipelineCacheCreateInfo::default().initial_data(data);
    // SAFETY: data that passed `decode` came from vkGetPipelineCacheData on this device and
    // driver; the driver checks its own header too.
    unsafe { device.create_pipeline_cache(&info, None) }
}

pub(super) struct PipelineCache {
    path: Option<PathBuf>,
    /// The pipeline store's directory.
    store: Option<PathBuf>,
    identity: Identity,
    /// Null without a path or a store, or when the driver could not create it.
    cache: vk::PipelineCache,
    /// With a store, read-locked to use `cache` and write-locked to merge into it, which needs it
    /// to itself, so nothing compiles through it. The keys of the pipelines already in it, and
    /// whether this process has marked each one's file as used.
    merged: RwLock<HashMap<blake3::Hash, AtomicBool>>,
    /// Whether to wait for a pipeline another process is compiling.
    waits: AtomicBool,
    /// The file or the store held pipelines at open.
    loaded: bool,
    rejected: Option<String>,
    hits: AtomicU32,
    misses: AtomicU32,
    /// Held while saving; the misses counted when the file was last written.
    saved: Mutex<u32>,
    /// The keys of the pipelines being compiled now: a thread asking for one of them waits, then
    /// gets the pipeline from the driver's or this cache without compiling.
    compiling: Mutex<HashSet<blake3::Hash>>,
    compiled: Condvar,
}

/// One pipeline's cache from the store, for one creation.
struct StoredPipeline<'a> {
    device: &'a ash::Device,
    cache: vk::PipelineCache,
    /// Whether the file held data for this device and driver.
    filled: bool,
    /// The pipeline's file. `None` when the request compiles nothing, or when the file can't be
    /// opened and the pipeline goes unstored.
    file: Option<File>,
    locked: bool,
}

impl StoredPipeline<'_> {
    /// Replaces the cache with an empty one, for a compile: data a miss left out of the file comes
    /// only from compiling into a cache that doesn't already hold the pipeline.
    fn empty(&mut self) -> Result<(), Error> {
        let empty = create_cache(self.device, &[]).map_err(Error::vk("vkCreatePipelineCache"))?;
        // SAFETY: no creation is using the old cache.
        unsafe { self.device.destroy_pipeline_cache(self.cache, None) };
        self.cache = empty;
        Ok(())
    }

    /// Writes the cache to the file with what the file holds now added to it, after a miss put
    /// the pipeline in the cache. Adding, not replacing: on the driver in the module's measurement,
    /// two processes replacing files left a store whose largest kernel missed for good, with
    /// every file present. The pipeline exists whether or not this works, so a failure only warns.
    fn write(&self, identity: &Identity) {
        let Some(mut file) = self.file.as_ref() else {
            return;
        };
        if !self.locked
            && let Err(e) = file.lock()
        {
            log::warn!("pipeline store: {e}");
            return;
        }
        let mut held = Vec::new();
        let read = file.rewind().and_then(|()| file.read_to_end(&mut held));
        if let Err(e) = read {
            log::warn!("pipeline store: {e}");
            return;
        }
        if let Some(data) = decode(&held, identity).ok().filter(|d| !d.is_empty())
            && let Ok(held) = create_cache(self.device, data)
        {
            // SAFETY: both caches are live, and no creation is using the destination.
            let merged = unsafe { self.device.merge_pipeline_caches(self.cache, &[held]) };
            // SAFETY: nothing else uses the source, and the merge has returned.
            unsafe { self.device.destroy_pipeline_cache(held, None) };
            if let Err(e) = merged {
                log::warn!("pipeline store: vkMergePipelineCaches failed ({e:?})");
            }
        }
        // SAFETY: the cache is live; VkPipelineCache is internally synchronized.
        let data = match unsafe { self.device.get_pipeline_cache_data(self.cache) } {
            Ok(data) => data,
            Err(e) => {
                log::warn!("pipeline store: vkGetPipelineCacheData failed ({e:?})");
                return;
            }
        };
        let bytes = encode(identity, &data);
        let written = file
            .rewind()
            .and_then(|()| file.set_len(0))
            .and_then(|()| file.write_all(&bytes));
        if let Err(e) = written {
            log::warn!("pipeline store: {e}");
        }
    }
}

impl Drop for StoredPipeline<'_> {
    fn drop(&mut self) {
        // SAFETY: the one creation that used the cache has returned.
        unsafe { self.device.destroy_pipeline_cache(self.cache, None) };
    }
}

/// A pipeline, and whether the cache it came through held it.
struct Created {
    pipeline: vk::Pipeline,
    hit: bool,
}

/// Marks a key as compiling until dropped.
struct CompileGuard<'a> {
    cache: &'a PipelineCache,
    key: blake3::Hash,
}

impl Drop for CompileGuard<'_> {
    fn drop(&mut self) {
        self.cache.compiling.lock().unwrap().remove(&self.key);
        self.cache.compiled.notify_all();
    }
}

impl PipelineCache {
    /// Opens the store at `store`, else the cache at `path`. A missing, damaged or foreign file
    /// starts the cache empty; so does one the driver refuses, which is removed, or ignores, which
    /// the next save replaces.
    pub(super) fn open(
        device: &ash::Device,
        props: &vk::PhysicalDeviceProperties,
        path: Option<&Path>,
        store: Option<&Path>,
    ) -> PipelineCache {
        if let Some(Err(e)) = store.map(std::fs::create_dir_all) {
            log::warn!("pipeline store: {e}");
        }
        let mut cache = PipelineCache {
            path: path.filter(|_| store.is_none()).map(Path::to_path_buf),
            store: store.map(Path::to_path_buf),
            identity: Identity {
                vendor: props.vendor_id,
                device: props.device_id,
                driver_version: props.driver_version,
                cache_uuid: props.pipeline_cache_uuid,
            },
            cache: vk::PipelineCache::null(),
            merged: RwLock::default(),
            waits: AtomicBool::new(false),
            loaded: false,
            rejected: None,
            hits: AtomicU32::new(0),
            misses: AtomicU32::new(0),
            saved: Mutex::new(0),
            compiling: Mutex::default(),
            compiled: Condvar::new(),
        };
        if let Some(dir) = store {
            match create_cache(device, &[]) {
                Ok(pc) => {
                    cache.cache = pc;
                    cache.load_store(device, dir);
                }
                Err(e) => {
                    log::warn!("pipeline store: vkCreatePipelineCache failed ({e:?})");
                    cache.store = None;
                }
            }
            return cache;
        }
        let Some(path) = path else { return cache };
        let file = match std::fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                cache.rejected = Some(format!("unreadable: {e}"));
                None
            }
        };
        let initial = match file.as_deref().map(|f| decode(f, &cache.identity)) {
            Some(Ok(data)) => data,
            Some(Err(why)) => {
                cache.rejected = Some(why);
                &[]
            }
            None => &[],
        };
        let create = |data: &[u8]| create_cache(device, data);
        // A driver takes data it finds incompatible as an empty cache and reports success. The
        // cache is then no larger than a new one, and a file with pipelines in it is larger.
        let ignored = |pc: vk::PipelineCache| {
            let Ok(empty) = create(&[]) else {
                return false;
            };
            let sizes = (data_size(device, pc), data_size(device, empty));
            // SAFETY: the cache was never used.
            unsafe { device.destroy_pipeline_cache(empty, None) };
            matches!(sizes, (Some(size), Some(new)) if size <= new && initial.len() > new)
        };
        let created = create(initial);
        let unusable = match &created {
            Err(e) if !initial.is_empty() && !OUT_OF_MEMORY.contains(e) => {
                Some(format!("the driver refused it ({e:?})"))
            }
            Ok(pc) if !initial.is_empty() && ignored(*pc) => Some("the driver ignored it".into()),
            _ => None,
        };
        let created = match unusable {
            Some(why) => {
                log::warn!("pipeline cache {}: {why}", path.display());
                cache.rejected = Some(why);
                created.or_else(|_| {
                    let _ = std::fs::remove_file(path);
                    create(&[])
                })
            }
            None => {
                cache.loaded = created.is_ok() && !initial.is_empty();
                created
            }
        };
        match created {
            Ok(pc) => cache.cache = pc,
            Err(e) => log::warn!("pipeline cache: vkCreatePipelineCache failed ({e:?})"),
        }
        match &cache.rejected {
            Some(why) => log::info!("pipeline cache {}: file ignored ({why})", path.display()),
            None => log::info!(
                "pipeline cache {}: {} bytes loaded",
                path.display(),
                initial.len()
            ),
        }
        cache
    }

    /// Waits until no other thread is compiling `key`, then marks it as compiling.
    fn begin_compile(&self, key: blake3::Hash) -> CompileGuard<'_> {
        let compiling = self.compiling.lock().unwrap();
        self.compiled
            .wait_while(compiling, |compiling| compiling.contains(&key))
            .unwrap()
            .insert(key);
        CompileGuard { cache: self, key }
    }

    /// Merges every usable file in `dir` into the cache. A file being compiled or written, or
    /// one the driver refuses, stays out: its pipeline misses and the file is rewritten.
    fn load_store(&mut self, device: &ash::Device, dir: &Path) {
        let start = Instant::now();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let merged = self.merged.get_mut().unwrap();
        let mut size = 0;
        for path in entries.flatten().map(|entry| entry.path()) {
            let key = path.file_stem().and_then(|stem| stem.to_str());
            let Some(key) = key.and_then(|key| blake3::Hash::from_hex(key).ok()) else {
                continue;
            };
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(data) = decode(&bytes, &self.identity) else {
                continue;
            };
            let Ok(file) = create_cache(device, data) else {
                continue;
            };
            // SAFETY: the device has just opened, so nothing else uses the destination; the
            // source is live.
            let result = unsafe { device.merge_pipeline_caches(self.cache, &[file]) };
            // SAFETY: nothing else uses the source, and the merge has returned.
            unsafe { device.destroy_pipeline_cache(file, None) };
            match result {
                Ok(()) => {
                    merged.insert(key, AtomicBool::new(false));
                    size += data.len();
                }
                Err(e) => log::warn!("pipeline store: vkMergePipelineCaches failed ({e:?})"),
            }
        }
        self.loaded = !merged.is_empty();
        *self.waits.get_mut() = self.loaded;
        log::info!(
            "pipeline store {}: {} pipelines, {size} bytes, loaded in {:.0} ms",
            dir.display(),
            merged.len(),
            start.elapsed().as_secs_f64() * 1e3
        );
    }

    /// Takes `file`'s lock, waiting for a process that holds it, unless waiting has ended. Returns
    /// whether it did.
    fn wait_for(&self, file: &File) -> std::io::Result<bool> {
        while self.waits.load(Ordering::Relaxed) {
            match file.try_lock() {
                Ok(()) => return Ok(true),
                Err(TryLockError::WouldBlock) => thread::sleep(WAIT_POLL),
                Err(TryLockError::Error(e)) => return Err(e),
            }
        }
        Ok(false)
    }

    /// The cache of the pipeline with `key`, from its file in `dir`. A request that compiles
    /// holds the file's lock until it drops the result, if it could wait for it. Without the lock
    /// the file reads as it stands: Windows refuses a read of a locked file, and a file mid-write
    /// fails its checksum, so either way it misses.
    fn stored<'a>(
        &self,
        device: &'a ash::Device,
        dir: &Path,
        key: blake3::Hash,
        compiles: bool,
    ) -> Result<StoredPipeline<'a>, Error> {
        let path = stored_path(dir, key);
        let opened = if compiles {
            File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .and_then(|file| self.wait_for(&file).map(|locked| (file, locked)))
        } else {
            File::open(&path).map(|file| (file, false))
        };
        let mut bytes = Vec::new();
        let (file, locked) = match opened {
            Ok((mut file, locked)) => {
                if file.read_to_end(&mut bytes).is_err() {
                    bytes.clear();
                }
                (compiles.then_some(file), locked)
            }
            Err(_) if !compiles => (None, false),
            Err(e) => {
                log::warn!("pipeline store {}: {e}", path.display());
                (None, false)
            }
        };
        let data = decode(&bytes, &self.identity).unwrap_or_default();
        let (cache, filled) = match create_cache(device, data) {
            Ok(cache) => (cache, !data.is_empty()),
            Err(e) if !data.is_empty() && !OUT_OF_MEMORY.contains(&e) => {
                let empty = create_cache(device, &[]);
                (empty.map_err(Error::vk("vkCreatePipelineCache"))?, false)
            }
            Err(e) => return Err(Error::vk("vkCreatePipelineCache")(e)),
        };
        Ok(StoredPipeline {
            device,
            cache,
            filled,
            file,
            locked,
        })
    }

    /// Asks `attempt` for a pipeline from the merged cache, first merging `stored`'s into it, and
    /// marks the pipeline's file as used the first time this process hits it.
    fn through_merged(
        &self,
        stored: Option<&StoredPipeline<'_>>,
        dir: &Path,
        key: blake3::Hash,
        attempt: &impl Fn(vk::PipelineCache, bool) -> Result<Created, Error>,
    ) -> Result<Created, Error> {
        if let Some(stored) = stored {
            self.merge(stored, key);
        }
        let merged = self.merged.read().unwrap();
        let created = attempt(self.cache, true);
        let first_use = created.is_ok()
            && merged
                .get(&key)
                .is_some_and(|used| !used.swap(true, Ordering::Relaxed));
        drop(merged);
        if first_use {
            let touched = File::options()
                .write(true)
                .open(stored_path(dir, key))
                .and_then(|file| file.set_modified(SystemTime::now()));
            if let Err(e) = touched {
                log::warn!("pipeline store: {e}");
            }
        }
        created
    }

    fn merge(&self, stored: &StoredPipeline<'_>, key: blake3::Hash) {
        let mut merged = self.merged.write().unwrap();
        // SAFETY: the write lock keeps everything else off the destination; the source is live.
        let result = unsafe {
            stored
                .device
                .merge_pipeline_caches(self.cache, &[stored.cache])
        };
        match result {
            Ok(()) => {
                merged.entry(key).or_default();
            }
            Err(e) => log::warn!("pipeline store: vkMergePipelineCaches failed ({e:?})"),
        }
    }

    /// Creates the pipeline with `key` through the store in `dir`. `attempt` takes the cache and
    /// whether to fail rather than compile. Without `compiles` the pipeline comes from the store
    /// or the driver's own cache, or this fails.
    fn create_stored(
        &self,
        device: &ash::Device,
        dir: &Path,
        key: blake3::Hash,
        compiles: bool,
        attempt: impl Fn(vk::PipelineCache, bool) -> Result<Created, Error>,
    ) -> Result<Created, Error> {
        if self.merged.read().unwrap().contains_key(&key) {
            match self.through_merged(None, dir, key, &attempt) {
                Err(Error::CompileRequired) => {}
                created => return created,
            }
        }
        let mut stored = self.stored(device, dir, key, compiles)?;
        if stored.filled || !compiles {
            match self.through_merged(stored.filled.then_some(&stored), dir, key, &attempt) {
                Err(Error::CompileRequired) if compiles => {
                    if self.waits.swap(false, Ordering::Relaxed) {
                        log::info!("pipeline store: a stored pipeline missed, so waiting ends");
                    }
                }
                created => return created,
            }
        }
        stored.empty()?;
        let created = attempt(stored.cache, false)?;
        if !created.hit {
            stored.write(&self.identity);
        }
        self.merge(&stored, key);
        Ok(created)
    }

    pub(super) fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: no pipeline creation is running; the device is being destroyed. Null is
        // ignored.
        unsafe { device.destroy_pipeline_cache(self.cache, None) };
        self.cache = vk::PipelineCache::null();
    }
}

impl Shared {
    /// Creates a pipeline through the cache with `extra` creation flags, compiling it once when
    /// several threads ask for it. `stages` and `state`, whatever else the create info depends on,
    /// tell pipelines apart. `create` builds the create info with `next` as its `p_next` (creation
    /// flags, then creation feedback) and calls the create function with `cache`.
    pub(super) fn create_pipeline(
        &self,
        stages: &[&StageParts<'_>],
        state: &[u8],
        extra: vk::PipelineCreateFlags2,
        name: &str,
        call: &'static str,
        create: impl Fn(vk::PipelineCache, *const c_void) -> Result<vk::Pipeline, vk::Result>,
    ) -> Result<vk::Pipeline, Error> {
        let cache = &self.pipeline_cache;
        let cached_only = vk::PipelineCreateFlags2::FAIL_ON_PIPELINE_COMPILE_REQUIRED;
        let compiles = !extra.contains(cached_only);
        // A request for cached pipelines only compiles nothing, so nothing waits for it. It needs
        // the key only to find in the store what a request that compiles wrote there.
        let key = (compiles || cache.store.is_some()).then(|| {
            let mut key = blake3::Hasher::new();
            for s in stages {
                s.key(&mut key);
            }
            key.update(state);
            key.update(&(extra & !cached_only).as_raw().to_le_bytes());
            key.finalize()
        });
        let guard = key.filter(|_| compiles).map(|k| cache.begin_compile(k));
        let start = Instant::now();
        let attempt = |handle: vk::PipelineCache, only_cached: bool| {
            let mut feedback = vk::PipelineCreationFeedback::default();
            let mut feedback_info = vk::PipelineCreationFeedbackCreateInfo::default()
                .pipeline_creation_feedback(&mut feedback);
            let extra = if only_cached {
                extra | cached_only
            } else {
                extra
            };
            let mut flags =
                vk::PipelineCreateFlags2CreateInfo::default().flags(self.pipeline_flags() | extra);
            flags.p_next = std::ptr::from_mut(&mut feedback_info).cast();
            let pipeline =
                create(handle, std::ptr::from_ref(&flags).cast()).map_err(Error::vk(call))?;
            // Drivers need not report a hit, and a creation that may not compile compiled nothing.
            let hit = only_cached
                || feedback
                    .flags
                    .contains(vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT);
            Ok(Created { pipeline, hit })
        };
        let created = match (&cache.store, key) {
            (Some(dir), Some(key)) => {
                cache.create_stored(&self.device, dir, key, compiles, attempt)?
            }
            _ => attempt(cache.cache, false)?,
        };
        let raw = created.pipeline;
        if cache.cache != vk::PipelineCache::null() {
            // Counted after the pipeline is in the cache, so a save that sees the count sees
            // what it must write.
            let count = if created.hit {
                &cache.hits
            } else {
                &cache.misses
            };
            count.fetch_add(1, Ordering::Release);
        }
        drop(guard);
        self.created(raw, name, start);
        Ok(raw)
    }
}

impl Device {
    /// Writes the pipeline cache to the path given at device creation, replacing the file
    /// whole. Does nothing without a path, or when no pipeline has missed the cache since
    /// the device opened it or last wrote it. Safe to call from several threads and
    /// processes at once: each writes a whole file and the last one stays.
    pub fn save_pipeline_cache(&self) -> Result<(), Error> {
        let cache = &self.shared.pipeline_cache;
        let Some(path) = &cache.path else {
            return Ok(());
        };
        let mut saved = cache.saved.lock().unwrap();
        let misses = cache.misses.load(Ordering::Acquire);
        if misses == *saved {
            return Ok(());
        }
        // SAFETY: the cache is live (a miss was counted through it); VkPipelineCache is
        // internally synchronized.
        let data = unsafe { self.shared.device.get_pipeline_cache_data(cache.cache) }
            .map_err(Error::vk("vkGetPipelineCacheData"))?;
        let bytes = encode(&cache.identity, &data);
        write(path, &bytes)?;
        *saved = misses;
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
