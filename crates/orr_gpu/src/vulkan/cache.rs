//! The disk pipeline cache: one `VkPipelineCache` that every pipeline compiles through. The
//! file holds the device identity, the cache data and a checksum; a file from another device,
//! driver or format is ignored and rebuilt.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use ash::vk;

use super::device::{Device, Shared};
use crate::Error;

/// What the pipeline cache loaded and how creation has used it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineCacheReport {
    /// The file existed and its data was loaded.
    pub loaded: bool,
    /// Pipelines compiled with the driver reporting a cache hit.
    pub hits: u32,
    /// Pipelines compiled with no cache hit.
    pub misses: u32,
    /// Why the file was ignored, when it existed and was.
    pub rejected: Option<String>,
}

const MAGIC: [u8; 8] = *b"orrpipe\0";
const FORMAT: u32 = 2;
/// Magic, format, vendor, device, driver version, then the 16-byte cache UUID.
const HEADER: usize = 24 + 16;
const CHECKSUM: usize = 32;

/// What a cache file must match to be used.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Identity {
    pub vendor: u32,
    pub device: u32,
    pub driver_version: u32,
    pub cache_uuid: [u8; 16],
}

/// The file: identity, payload, then a BLAKE3 hash of everything before it.
pub(crate) fn encode(id: &Identity, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + payload.len() + CHECKSUM);
    out.extend_from_slice(&MAGIC);
    for v in [FORMAT, id.vendor, id.device, id.driver_version] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&id.cache_uuid);
    out.extend_from_slice(payload);
    let sum = blake3::hash(&out);
    out.extend_from_slice(sum.as_bytes());
    out
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
    let Some((header, payload)) = body.split_at_checked(HEADER) else {
        return Err("truncated".into());
    };
    if header[..8] != MAGIC {
        return Err("not an orr pipeline cache".into());
    }
    let word = |i: usize| u32::from_le_bytes(header[8 + 4 * i..12 + 4 * i].try_into().unwrap());
    let format = word(0);
    if format != FORMAT {
        return Err(format!("format {format}, expected {FORMAT}"));
    }
    let differs = |what: &str| Err(format!("written for another {what}"));
    if (word(1), word(2)) != (expect.vendor, expect.device) {
        return differs("device");
    }
    if word(3) != expect.driver_version {
        return differs("driver version");
    }
    if header[24..] != expect.cache_uuid {
        return differs("pipeline cache UUID");
    }
    Ok(payload)
}

pub(crate) struct PipelineCache {
    path: Option<PathBuf>,
    identity: Identity,
    /// Null without a path, or when the driver could not create it.
    cache: vk::PipelineCache,
    loaded: bool,
    rejected: Option<String>,
    hits: AtomicU32,
    misses: AtomicU32,
    /// Held while saving; the misses counted when the file was last written.
    saved: Mutex<u32>,
}

impl PipelineCache {
    /// Opens the cache at `path`. A missing, damaged or foreign file starts it empty.
    pub(crate) fn open(
        device: &ash::Device,
        props: &vk::PhysicalDeviceProperties,
        path: Option<&Path>,
    ) -> PipelineCache {
        let mut cache = PipelineCache {
            path: path.map(Path::to_path_buf),
            identity: Identity {
                vendor: props.vendor_id,
                device: props.device_id,
                driver_version: props.driver_version,
                cache_uuid: props.pipeline_cache_uuid,
            },
            cache: vk::PipelineCache::null(),
            loaded: false,
            rejected: None,
            hits: AtomicU32::new(0),
            misses: AtomicU32::new(0),
            saved: Mutex::new(0),
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
        let initial = match file.as_deref().map(|f| decode(f, &cache.identity)) {
            Some(Ok(data)) => data,
            Some(Err(why)) => {
                cache.rejected = Some(why);
                &[]
            }
            None => &[],
        };
        let info = vk::PipelineCacheCreateInfo::default().initial_data(initial);
        // SAFETY: the data was written by vkGetPipelineCacheData for this device and driver;
        // the driver checks its own header too.
        match unsafe { device.create_pipeline_cache(&info, None) } {
            Ok(pc) => {
                cache.cache = pc;
                cache.loaded = !initial.is_empty();
            }
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

    pub(crate) fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: no pipeline creation is running; the device is being destroyed. Null is
        // ignored.
        unsafe { device.destroy_pipeline_cache(self.cache, None) };
        self.cache = vk::PipelineCache::null();
    }
}

impl Shared {
    /// Creates a pipeline through the cache with `extra` creation flags. `create` builds the
    /// create info with `next` as its `p_next` (creation flags, then creation feedback) and
    /// calls the create function with `cache`.
    pub(super) fn create_pipeline(
        &self,
        extra: vk::PipelineCreateFlags2,
        name: &str,
        call: &'static str,
        create: impl FnOnce(vk::PipelineCache, *const c_void) -> Result<vk::Pipeline, vk::Result>,
    ) -> Result<vk::Pipeline, Error> {
        let start = Instant::now();
        let cache = &self.pipeline_cache;
        let mut feedback = vk::PipelineCreationFeedback::default();
        let mut feedback_info = vk::PipelineCreationFeedbackCreateInfo::default()
            .pipeline_creation_feedback(&mut feedback);
        let mut flags =
            vk::PipelineCreateFlags2CreateInfo::default().flags(self.pipeline_flags() | extra);
        flags.p_next = std::ptr::from_mut(&mut feedback_info).cast();
        let raw =
            create(cache.cache, std::ptr::from_ref(&flags).cast()).map_err(Error::vk(call))?;
        if cache.cache != vk::PipelineCache::null() {
            let hit = feedback
                .flags
                .contains(vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT);
            // Counted after the pipeline is in the cache, so a save that sees the count sees
            // what it must write.
            if hit { &cache.hits } else { &cache.misses }.fetch_add(1, Ordering::Release);
        }
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
        let io = |e| Error::Io {
            path: path.clone(),
            source: e,
        };
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        let mut tmp = path.clone().into_os_string();
        tmp.push(format!(".{}.tmp", std::process::id()));
        let written = std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, path));
        if let Err(e) = written {
            let _ = std::fs::remove_file(&tmp);
            return Err(io(e));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_or_damaged_files_are_rejected() {
        let id = Identity {
            vendor: 0x10de,
            device: 0x2b85,
            driver_version: 7,
            cache_uuid: [3; 16],
        };
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
        other.cache_uuid[0] ^= 1;
        assert!(decode(&file, &other).unwrap_err().contains("UUID"));

        let mut flipped = file.clone();
        flipped[30] ^= 1;
        assert!(decode(&flipped, &id).unwrap_err().contains("checksum"));
        assert!(decode(&file[..file.len() - 1], &id).is_err());
        assert!(decode(&[], &id).is_err());
    }
}
