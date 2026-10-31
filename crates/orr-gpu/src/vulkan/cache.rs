//! The disk pipeline cache. With `VK_KHR_pipeline_binary` each pipeline's binaries are kept
//! under orr's hash of its creation inputs. The first pipeline of each kind checks that its
//! binaries load back; kinds whose binaries the driver refuses compile through a
//! `VkPipelineCache` kept beside them. Without the extension only the `VkPipelineCache` is
//! kept. Either way the file starts with the device identity and ends with a checksum, and a
//! file from another device, driver or format is ignored and rebuilt.

use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::device::{Device, Shared};
use crate::{Caps, Error};

/// Which cache a device keeps.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PipelineCacheKind {
    /// No cache path was given.
    None,
    /// Pipeline binaries per pipeline (`VK_KHR_pipeline_binary`). Kinds of pipeline whose
    /// binaries the driver does not take back compile through a `VkPipelineCache` kept
    /// beside them.
    Binaries,
    /// One `VkPipelineCache`.
    File,
}

/// What the pipeline cache loaded and how creation has used it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineCacheReport {
    pub kind: PipelineCacheKind,
    /// Pipelines read from the file: entries for binaries, 1 for a loaded `VkPipelineCache`.
    pub loaded: u32,
    /// Pipelines created from cached binaries, or compiled with the driver reporting a
    /// `VkPipelineCache` hit.
    pub hits: u32,
    /// Pipelines compiled with no cache hit.
    pub misses: u32,
    /// Why the file was ignored, when it existed and was.
    pub rejected: Option<String>,
}

const MAGIC: [u8; 8] = *b"orrpipe\0";
const FORMAT: u32 = 1;
const CHECKSUM: usize = 32;

/// What a cache file must match to be used.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Identity {
    pub kind: u32,
    pub vendor: u32,
    pub device: u32,
    pub driver_version: u32,
    /// The driver UUID for binaries, `pipelineCacheUUID` for a `VkPipelineCache`.
    pub uuid: [u8; 16],
    /// The driver's global pipeline key for binaries; empty otherwise.
    pub global_key: Vec<u8>,
}

/// The file: identity, payload, then a BLAKE3 hash of everything before it.
pub(crate) fn encode(id: &Identity, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + id.global_key.len() + payload.len() + CHECKSUM);
    out.extend_from_slice(&MAGIC);
    for v in [FORMAT, id.kind, id.vendor, id.device, id.driver_version] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&id.uuid);
    out.extend_from_slice(&(id.global_key.len() as u32).to_le_bytes());
    out.extend_from_slice(&id.global_key);
    out.extend_from_slice(payload);
    let sum = blake3::hash(&out);
    out.extend_from_slice(sum.as_bytes());
    out
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.0.len() < n {
            return Err("truncated".into());
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// The payload of `file` if it is intact and was written for `expect`; otherwise why not.
pub(crate) fn decode<'a>(file: &'a [u8], expect: &Identity) -> Result<&'a [u8], String> {
    let Some(body_len) = file.len().checked_sub(CHECKSUM) else {
        return Err("truncated".into());
    };
    let (body, sum) = file.split_at(body_len);
    if blake3::hash(body).as_bytes() != sum {
        return Err("checksum mismatch".into());
    }
    let mut r = Reader(body);
    if r.take(8)? != MAGIC {
        return Err("not an orr pipeline cache".into());
    }
    let format = r.u32()?;
    if format != FORMAT {
        return Err(format!("format {format}, expected {FORMAT}"));
    }
    let found = Identity {
        kind: r.u32()?,
        vendor: r.u32()?,
        device: r.u32()?,
        driver_version: r.u32()?,
        uuid: r.take(16)?.try_into().unwrap(),
        global_key: {
            let n = r.u32()? as usize;
            r.take(n)?.to_vec()
        },
    };
    let differs = |what: &str| Err(format!("written for another {what}"));
    if found.kind != expect.kind {
        return differs("cache kind");
    }
    if (found.vendor, found.device) != (expect.vendor, expect.device) {
        return differs("device");
    }
    if found.driver_version != expect.driver_version {
        return differs("driver version");
    }
    if found.uuid != expect.uuid {
        return differs("driver or cache UUID");
    }
    if found.global_key != expect.global_key {
        return differs("driver pipeline key");
    }
    Ok(r.0)
}

/// One pipeline binary: the driver's key and data.
struct Binary {
    key: Vec<u8>,
    data: Vec<u8>,
}

type Entries = HashMap<[u8; 32], Arc<[Binary]>>;

fn encode_entries(entries: &Entries) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (hash, binaries) in entries {
        out.extend_from_slice(hash);
        out.extend_from_slice(&(binaries.len() as u32).to_le_bytes());
        for b in binaries.iter() {
            out.extend_from_slice(&(b.key.len() as u32).to_le_bytes());
            out.extend_from_slice(&b.key);
            out.extend_from_slice(&(b.data.len() as u64).to_le_bytes());
            out.extend_from_slice(&b.data);
        }
    }
    out
}

fn decode_entries(payload: &[u8]) -> Result<Entries, String> {
    let mut r = Reader(payload);
    let mut entries = Entries::new();
    for _ in 0..r.u32()? {
        let hash: [u8; 32] = r.take(32)?.try_into().unwrap();
        let mut binaries = Vec::new();
        for _ in 0..r.u32()? {
            let key_len = r.u32()? as usize;
            if key_len > vk::MAX_PIPELINE_BINARY_KEY_SIZE_KHR {
                return Err("binary key too long".into());
            }
            let key = r.take(key_len)?.to_vec();
            let data_len = usize::try_from(r.u64()?).map_err(|_| "binary too large")?;
            let data = r.take(data_len)?.to_vec();
            binaries.push(Binary { key, data });
        }
        entries.insert(hash, binaries.into());
    }
    if !r.0.is_empty() {
        return Err("trailing bytes".into());
    }
    Ok(entries)
}

/// The kinds of pipeline, which differ in whether the driver takes their binaries back.
#[derive(Clone, Copy, Debug)]
pub(super) enum PipelineKind {
    Compute,
    Mesh,
    RayDispatch,
}

const KINDS: usize = 3;
const UNKNOWN: u8 = 0;
const TAKES_BINARIES: u8 = 1;
const REFUSES_BINARIES: u8 = 2;

enum Backing {
    None,
    Binaries {
        fns: ash::khr::pipeline_binary::Device,
        entries: Mutex<Entries>,
        /// Per kind: whether the driver takes its binaries back.
        verdicts: [AtomicU8; KINDS],
        /// Compiles of kinds whose binaries are refused go through it; null when it could
        /// not be created.
        cache: vk::PipelineCache,
    },
    File(vk::PipelineCache),
}

type Verdicts = [u8; KINDS];

/// The binaries payload: the verdicts, the entries, then the `VkPipelineCache` data.
fn split_binaries_payload(payload: &[u8]) -> Result<(Verdicts, Entries, &[u8]), String> {
    let mut r = Reader(payload);
    let verdicts: Verdicts = r.take(KINDS)?.try_into().unwrap();
    if verdicts.iter().any(|&v| v > REFUSES_BINARIES) {
        return Err("bad verdict".into());
    }
    let len = usize::try_from(r.u64()?).map_err(|_| "entries too large")?;
    let entries = decode_entries(r.take(len)?)?;
    Ok((verdicts, entries, r.0))
}

fn create_vk_cache(device: &ash::Device, initial: &[u8]) -> Option<vk::PipelineCache> {
    let info = vk::PipelineCacheCreateInfo::default().initial_data(initial);
    // SAFETY: the data was written by vkGetPipelineCacheData for this device and driver;
    // the driver checks its own header too.
    match unsafe { device.create_pipeline_cache(&info, None) } {
        Ok(pc) => Some(pc),
        Err(e) => {
            log::warn!("pipeline cache: vkCreatePipelineCache failed ({e:?})");
            None
        }
    }
}

pub(crate) struct PipelineCache {
    path: Option<PathBuf>,
    identity: Identity,
    backing: Backing,
    loaded: u32,
    rejected: Option<String>,
    hits: AtomicU32,
    misses: AtomicU32,
}

impl PipelineCache {
    /// Opens the cache at `path`: pipeline binaries when the device has them, a
    /// `VkPipelineCache` otherwise. A missing, damaged or foreign file starts it empty.
    pub(crate) fn open(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        caps: &Caps,
        path: Option<&Path>,
    ) -> PipelineCache {
        let mut id_props = vk::PhysicalDeviceIDProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut id_props);
        // SAFETY: core 1.1 properties of a device of this instance.
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        let cache_uuid = props2.properties.pipeline_cache_uuid;
        let mut cache = PipelineCache {
            path: path.map(Path::to_path_buf),
            identity: Identity {
                kind: 0,
                vendor: caps.vendor_id,
                device: caps.device_id,
                driver_version: caps.driver_version,
                uuid: [0; 16],
                global_key: Vec::new(),
            },
            backing: Backing::None,
            loaded: 0,
            rejected: None,
            hits: AtomicU32::new(0),
            misses: AtomicU32::new(0),
        };
        let Some(path) = path else { return cache };
        let file = match std::fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                cache.rejected = Some(format!("unreadable: {e}"));
                None
            }
        };

        if caps.pipeline_binary {
            let fns = ash::khr::pipeline_binary::Device::load(instance, device);
            let mut key = vk::PipelineBinaryKeyKHR::default();
            // SAFETY: a null create info asks for the global key.
            let got = unsafe {
                (fns.fp().get_pipeline_key_khr)(device.handle(), std::ptr::null(), &mut key)
            };
            if got == vk::Result::SUCCESS {
                cache.identity.kind = PipelineCacheKind::Binaries as u32;
                cache.identity.uuid = id_props.driver_uuid;
                cache.identity.global_key = key.key[..key.key_size as usize].to_vec();
                let empty = ([UNKNOWN; KINDS], Entries::new(), &[][..]);
                let (verdicts, entries, initial) = match file
                    .as_deref()
                    .map(|f| decode(f, &cache.identity).and_then(split_binaries_payload))
                {
                    Some(Ok(loaded)) => loaded,
                    Some(Err(why)) => {
                        cache.rejected = Some(why);
                        empty
                    }
                    None => empty,
                };
                cache.loaded = entries.len() as u32;
                cache.backing = Backing::Binaries {
                    fns,
                    entries: Mutex::new(entries),
                    verdicts: verdicts.map(AtomicU8::new),
                    cache: create_vk_cache(device, initial).unwrap_or_default(),
                };
                cache.log(path);
                return cache;
            }
            log::warn!(
                "pipeline cache: vkGetPipelineKeyKHR failed ({got:?}); using a VkPipelineCache"
            );
        }

        cache.identity.kind = PipelineCacheKind::File as u32;
        cache.identity.uuid = cache_uuid;
        let initial = match file.as_deref().map(|f| decode(f, &cache.identity)) {
            Some(Ok(data)) => data,
            Some(Err(why)) => {
                cache.rejected = Some(why);
                &[]
            }
            None => &[],
        };
        if let Some(pc) = create_vk_cache(device, initial) {
            cache.loaded = u32::from(!initial.is_empty());
            cache.backing = Backing::File(pc);
        }
        cache.log(path);
        cache
    }

    fn log(&self, path: &Path) {
        match &self.rejected {
            Some(why) => log::info!(
                "pipeline cache {}: {:?}, file ignored ({why})",
                path.display(),
                self.kind()
            ),
            None => log::info!(
                "pipeline cache {}: {:?}, {} loaded",
                path.display(),
                self.kind(),
                self.loaded
            ),
        }
    }

    fn kind(&self) -> PipelineCacheKind {
        match self.backing {
            Backing::None => PipelineCacheKind::None,
            Backing::Binaries { .. } => PipelineCacheKind::Binaries,
            Backing::File(_) => PipelineCacheKind::File,
        }
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device) {
        let pc = match std::mem::replace(&mut self.backing, Backing::None) {
            Backing::File(pc) | Backing::Binaries { cache: pc, .. } => pc,
            Backing::None => return,
        };
        // SAFETY: no pipeline creation is running; the device is being destroyed. Null is
        // ignored.
        unsafe { device.destroy_pipeline_cache(pc, None) };
    }
}

/// Feeds `Hash` input into BLAKE3 for pipeline keys. The device and driver are not part
/// of the key: a file written for another one is refused whole.
pub(super) struct KeyHasher(blake3::Hasher);

impl KeyHasher {
    pub(super) fn new(kind: PipelineKind) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(&[kind as u8]);
        KeyHasher(h)
    }

    pub(super) fn finish_key(&self) -> [u8; 32] {
        *self.0.finalize().as_bytes()
    }
}

impl Hasher for KeyHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn finish(&self) -> u64 {
        u64::from_le_bytes(self.finish_key()[..8].try_into().unwrap())
    }
}

/// The pNext chain a creator puts on its create info: creation flags, creation feedback and
/// the binaries to build from.
pub(super) struct Chain<'a> {
    pub cache: vk::PipelineCache,
    flags: vk::PipelineCreateFlags2CreateInfo<'a>,
    feedback: vk::PipelineCreationFeedbackCreateInfo<'a>,
    binaries: vk::PipelineBinaryInfoKHR<'a>,
}

impl<'a> Chain<'a> {
    fn new(
        cache: vk::PipelineCache,
        flags: vk::PipelineCreateFlags2,
        feedback: &'a mut vk::PipelineCreationFeedback,
        binaries: &'a [vk::PipelineBinaryKHR],
    ) -> Self {
        Chain {
            cache,
            flags: vk::PipelineCreateFlags2CreateInfo::default().flags(flags),
            feedback: vk::PipelineCreationFeedbackCreateInfo::default()
                .pipeline_creation_feedback(feedback),
            binaries: vk::PipelineBinaryInfoKHR::default().pipeline_binaries(binaries),
        }
    }

    /// The chain's head, for the create info's `p_next`. Valid while `self` is neither moved
    /// nor dropped.
    pub(super) fn head(&mut self) -> *const c_void {
        self.flags.p_next = std::ptr::from_mut(&mut self.feedback).cast();
        self.feedback.p_next = if self.binaries.binary_count == 0 {
            std::ptr::null()
        } else {
            std::ptr::from_ref(&self.binaries).cast()
        };
        std::ptr::from_ref(&self.flags).cast()
    }
}

impl Shared {
    /// Creates a pipeline through the cache with `extra` creation flags, which `key` covers.
    /// `create` builds the create info with the chain's head as its `p_next` and the chain's
    /// cache, and calls the create function.
    pub(super) fn create_pipeline(
        &self,
        kind: PipelineKind,
        key: [u8; 32],
        extra: vk::PipelineCreateFlags2,
        name: &str,
        call: &'static str,
        mut create: impl FnMut(&mut Chain<'_>) -> Result<vk::Pipeline, vk::Result>,
    ) -> Result<vk::Pipeline, Error> {
        let start = Instant::now();
        let cache = &self.pipeline_cache;
        let flags = self.pipeline_flags() | extra;
        let count = |hit: bool| {
            if hit { &cache.hits } else { &cache.misses }.fetch_add(1, Ordering::Relaxed);
        };
        let raw = match &cache.backing {
            Backing::None => {
                self.compile(vk::PipelineCache::null(), flags, call, &mut create)?
                    .0
            }
            Backing::File(pc) => {
                let (raw, hit) = self.compile(*pc, flags, call, &mut create)?;
                count(hit);
                raw
            }
            Backing::Binaries {
                fns,
                entries,
                verdicts,
                cache: pc,
            } => {
                let verdict = &verdicts[kind as usize];
                let found = entries.lock().unwrap().get(&key).cloned();
                let warm = found.and_then(|binaries| {
                    match self.create_from_binaries(fns, &binaries, flags, &mut create) {
                        Ok(raw) => Some(raw),
                        Err(e) => {
                            log::info!("pipeline {name}: cached binaries refused ({e:?})");
                            verdict.store(REFUSES_BINARIES, Ordering::Relaxed);
                            entries.lock().unwrap().remove(&key);
                            None
                        }
                    }
                });
                if let Some(raw) = warm {
                    count(true);
                    raw
                } else if verdict.load(Ordering::Relaxed) == REFUSES_BINARIES {
                    let (raw, hit) = self.compile(*pc, flags, call, &mut create)?;
                    count(hit);
                    raw
                } else {
                    count(false);
                    let capture = flags | vk::PipelineCreateFlags2::CAPTURE_DATA_KHR;
                    let null = vk::PipelineCache::null();
                    let raw = self.compile(null, capture, call, &mut create)?.0;
                    match self.capture_binaries(fns, raw) {
                        Ok(b) => {
                            if verdict.load(Ordering::Relaxed) == UNKNOWN {
                                self.check_binaries(kind, verdict, fns, &b, flags, &mut create);
                            }
                            if verdict.load(Ordering::Relaxed) == TAKES_BINARIES {
                                entries.lock().unwrap().insert(key, b.into());
                                raw
                            } else {
                                // Compile once more so the VkPipelineCache holds it.
                                let seeded = self.compile(*pc, flags, call, &mut create)?.0;
                                // SAFETY: `raw` was never used.
                                unsafe { self.device.destroy_pipeline(raw, None) };
                                seeded
                            }
                        }
                        Err(e) => {
                            log::warn!("pipeline {name}: capturing binaries failed ({e:?})");
                            raw
                        }
                    }
                }
            }
        };
        self.created(raw, name, start);
        Ok(raw)
    }

    /// Settles whether the driver takes back binaries of `kind` by loading `binaries`, just
    /// captured, into a second pipeline.
    fn check_binaries(
        &self,
        kind: PipelineKind,
        verdict: &AtomicU8,
        fns: &ash::khr::pipeline_binary::Device,
        binaries: &[Binary],
        flags: vk::PipelineCreateFlags2,
        create: &mut impl FnMut(&mut Chain<'_>) -> Result<vk::Pipeline, vk::Result>,
    ) {
        match self.create_from_binaries(fns, binaries, flags, create) {
            Ok(check) => {
                // SAFETY: `check` was never used.
                unsafe { self.device.destroy_pipeline(check, None) };
                verdict.store(TAKES_BINARIES, Ordering::Relaxed);
            }
            Err(e) => {
                log::info!(
                    "pipeline cache: the driver refuses {kind:?} pipeline binaries ({e:?}); \
                     they compile through a VkPipelineCache"
                );
                verdict.store(REFUSES_BINARIES, Ordering::Relaxed);
            }
        }
    }

    /// Compiles through `pc`; the flag is set when the driver reports a cache hit.
    fn compile(
        &self,
        pc: vk::PipelineCache,
        flags: vk::PipelineCreateFlags2,
        call: &'static str,
        create: &mut impl FnMut(&mut Chain<'_>) -> Result<vk::Pipeline, vk::Result>,
    ) -> Result<(vk::Pipeline, bool), Error> {
        let mut feedback = vk::PipelineCreationFeedback::default();
        let raw =
            create(&mut Chain::new(pc, flags, &mut feedback, &[])).map_err(Error::vk(call))?;
        let hit = feedback
            .flags
            .contains(vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT);
        Ok((raw, hit))
    }

    fn create_from_binaries(
        &self,
        fns: &ash::khr::pipeline_binary::Device,
        binaries: &[Binary],
        flags: vk::PipelineCreateFlags2,
        create: &mut impl FnMut(&mut Chain<'_>) -> Result<vk::Pipeline, vk::Result>,
    ) -> Result<vk::Pipeline, vk::Result> {
        let keys: Vec<_> = binaries
            .iter()
            .map(|b| {
                let mut k = vk::PipelineBinaryKeyKHR {
                    key_size: b.key.len() as u32,
                    ..Default::default()
                };
                k.key[..b.key.len()].copy_from_slice(&b.key);
                k
            })
            .collect();
        let data: Vec<_> = binaries
            .iter()
            .map(|b| vk::PipelineBinaryDataKHR {
                data_size: b.data.len(),
                p_data: b.data.as_ptr().cast_mut().cast(),
                ..Default::default()
            })
            .collect();
        let keys_and_data = vk::PipelineBinaryKeysAndDataKHR {
            binary_count: binaries.len() as u32,
            p_pipeline_binary_keys: keys.as_ptr(),
            p_pipeline_binary_data: data.as_ptr(),
            ..Default::default()
        };
        let info = vk::PipelineBinaryCreateInfoKHR {
            p_keys_and_data_info: &keys_and_data,
            ..Default::default()
        };
        let mut handles = vec![vk::PipelineBinaryKHR::null(); binaries.len()];
        let mut out = vk::PipelineBinaryHandlesInfoKHR {
            pipeline_binary_count: handles.len() as u32,
            p_pipeline_binaries: handles.as_mut_ptr(),
            ..Default::default()
        };
        // SAFETY: the keys and data came from vkGetPipelineBinaryDataKHR on a device whose
        // identity matched; the handle array has room for every binary.
        let r = unsafe {
            (fns.fp().create_pipeline_binaries_khr)(
                self.device.handle(),
                &info,
                std::ptr::null(),
                &mut out,
            )
        };
        let made = &handles[..out.pipeline_binary_count as usize];
        let result = if r == vk::Result::SUCCESS {
            let mut feedback = vk::PipelineCreationFeedback::default();
            create(&mut Chain::new(
                vk::PipelineCache::null(),
                flags,
                &mut feedback,
                made,
            ))
        } else {
            Err(r)
        };
        self.destroy_binaries(fns, made);
        result
    }

    fn capture_binaries(
        &self,
        fns: &ash::khr::pipeline_binary::Device,
        raw: vk::Pipeline,
    ) -> Result<Vec<Binary>, vk::Result> {
        let fp = fns.fp();
        let device = self.device.handle();
        let info = vk::PipelineBinaryCreateInfoKHR {
            pipeline: raw,
            ..Default::default()
        };
        let mut out = vk::PipelineBinaryHandlesInfoKHR::default();
        // SAFETY: `raw` was created with CAPTURE_DATA and its data is not released yet; a
        // null handle array asks for the count.
        let mut r =
            unsafe { (fp.create_pipeline_binaries_khr)(device, &info, std::ptr::null(), &mut out) };
        let mut handles = Vec::new();
        if r == vk::Result::SUCCESS {
            handles = vec![vk::PipelineBinaryKHR::null(); out.pipeline_binary_count as usize];
            out.p_pipeline_binaries = handles.as_mut_ptr();
            // SAFETY: as above, with room for every binary.
            r = unsafe {
                (fp.create_pipeline_binaries_khr)(device, &info, std::ptr::null(), &mut out)
            };
            handles.truncate(out.pipeline_binary_count as usize);
        }
        let release = vk::ReleaseCapturedPipelineDataInfoKHR {
            pipeline: raw,
            ..Default::default()
        };
        // SAFETY: `raw` was created with CAPTURE_DATA and is released once.
        let _ =
            unsafe { (fp.release_captured_pipeline_data_khr)(device, &release, std::ptr::null()) };
        if r != vk::Result::SUCCESS {
            self.destroy_binaries(fns, &handles);
            return Err(r);
        }
        let mut binaries = Vec::with_capacity(handles.len());
        for &h in &handles {
            let data_info = vk::PipelineBinaryDataInfoKHR {
                pipeline_binary: h,
                ..Default::default()
            };
            let mut key = vk::PipelineBinaryKeyKHR::default();
            let mut size = 0usize;
            // SAFETY: a live binary; a null data pointer asks for the size.
            let mut r = unsafe {
                (fp.get_pipeline_binary_data_khr)(
                    device,
                    &data_info,
                    &mut key,
                    &mut size,
                    std::ptr::null_mut(),
                )
            };
            let mut data = vec![0u8; size];
            if r == vk::Result::SUCCESS {
                // SAFETY: as above, with `size` bytes of room.
                r = unsafe {
                    (fp.get_pipeline_binary_data_khr)(
                        device,
                        &data_info,
                        &mut key,
                        &mut size,
                        data.as_mut_ptr().cast(),
                    )
                };
            }
            if r != vk::Result::SUCCESS {
                self.destroy_binaries(fns, &handles);
                return Err(r);
            }
            binaries.push(Binary {
                key: key.key[..key.key_size as usize].to_vec(),
                data,
            });
        }
        self.destroy_binaries(fns, &handles);
        Ok(binaries)
    }

    fn destroy_binaries(
        &self,
        fns: &ash::khr::pipeline_binary::Device,
        handles: &[vk::PipelineBinaryKHR],
    ) {
        for &h in handles {
            // SAFETY: each binary is ours and used by no pending creation.
            unsafe {
                (fns.fp().destroy_pipeline_binary_khr)(self.device.handle(), h, std::ptr::null());
            }
        }
    }
}

impl Device {
    /// Writes the pipeline cache to the path given at device creation, replacing the file
    /// whole. Does nothing without a path.
    pub fn save_pipeline_cache(&self) -> Result<(), Error> {
        let cache = &self.shared.pipeline_cache;
        let Some(path) = &cache.path else {
            return Ok(());
        };
        let payload = match &cache.backing {
            Backing::None => return Ok(()),
            Backing::Binaries {
                entries,
                verdicts,
                cache: pc,
                ..
            } => {
                let entries = encode_entries(&entries.lock().unwrap());
                let mut payload: Vec<u8> =
                    verdicts.iter().map(|v| v.load(Ordering::Relaxed)).collect();
                payload.extend_from_slice(&(entries.len() as u64).to_le_bytes());
                payload.extend_from_slice(&entries);
                if *pc != vk::PipelineCache::null() {
                    payload.extend_from_slice(&self.vk_cache_data(*pc)?);
                }
                payload
            }
            Backing::File(pc) => self.vk_cache_data(*pc)?,
        };
        let bytes = encode(&cache.identity, &payload);
        let io = |e| Error::Io {
            path: path.clone(),
            source: e,
        };
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        let mut tmp = path.clone().into_os_string();
        tmp.push(".tmp");
        std::fs::write(&tmp, &bytes).map_err(io)?;
        std::fs::rename(&tmp, path).map_err(io)?;
        log::debug!(
            "pipeline cache {}: saved {} bytes",
            path.display(),
            bytes.len()
        );
        Ok(())
    }

    fn vk_cache_data(&self, pc: vk::PipelineCache) -> Result<Vec<u8>, Error> {
        // SAFETY: the cache is live; VkPipelineCache is internally synchronised.
        unsafe { self.shared.device.get_pipeline_cache_data(pc) }
            .map_err(Error::vk("vkGetPipelineCacheData"))
    }

    pub fn pipeline_cache_report(&self) -> PipelineCacheReport {
        let c = &self.shared.pipeline_cache;
        PipelineCacheReport {
            kind: c.kind(),
            loaded: c.loaded,
            hits: c.hits.load(Ordering::Relaxed),
            misses: c.misses.load(Ordering::Relaxed),
            rejected: c.rejected.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Identity {
        Identity {
            kind: PipelineCacheKind::Binaries as u32,
            vendor: 0x10de,
            device: 0x2b85,
            driver_version: 7,
            uuid: [3; 16],
            global_key: vec![9; 20],
        }
    }

    #[test]
    fn foreign_or_damaged_files_are_rejected() {
        let id = identity();
        let file = encode(&id, b"payload");
        assert_eq!(decode(&file, &id), Ok(&b"payload"[..]));

        let mut other = id.clone();
        other.driver_version += 1;
        assert!(
            decode(&file, &other)
                .unwrap_err()
                .contains("driver version")
        );
        let mut other = id.clone();
        other.device += 1;
        assert!(decode(&file, &other).unwrap_err().contains("device"));
        let mut other = id.clone();
        other.global_key[0] ^= 1;
        assert!(decode(&file, &other).unwrap_err().contains("key"));
        let mut other = id.clone();
        other.kind = PipelineCacheKind::File as u32;
        assert!(decode(&file, &other).is_err());

        let mut flipped = file.clone();
        flipped[40] ^= 1;
        assert!(decode(&flipped, &id).unwrap_err().contains("checksum"));
        assert!(decode(&file[..file.len() - 1], &id).is_err());
        assert!(decode(&[], &id).is_err());
    }

    #[test]
    fn entries_round_trip() {
        let mut entries = Entries::new();
        entries.insert(
            [1; 32],
            vec![
                Binary {
                    key: vec![1, 2, 3],
                    data: vec![4; 1000],
                },
                Binary {
                    key: vec![5; 32],
                    data: Vec::new(),
                },
            ]
            .into(),
        );
        entries.insert([2; 32], Vec::new().into());
        let back = decode_entries(&encode_entries(&entries)).unwrap();
        assert_eq!(back.len(), 2);
        let b = &back[&[1; 32]];
        assert_eq!(
            (b[0].key.as_slice(), b[0].data.len()),
            (&[1, 2, 3][..], 1000)
        );
        assert_eq!((b[1].key.len(), b[1].data.len()), (32, 0));
        assert!(back[&[2; 32]].is_empty());
        let bytes = encode_entries(&entries);
        assert!(decode_entries(&bytes[..bytes.len() - 1]).is_err());
    }
}
