//! How to create a device, alike on every backend.

use std::ffi::CStr;
use std::io::ErrorKind;
use std::path::Path;

use crate::{DeviceHooks, Diagnostics, Error, QueueRequest};

/// How to create a [`Device`](crate::Device). Fields only the Vulkan backend reads say so.
#[derive(Clone, Copy, Debug)]
pub struct DeviceDesc<'a> {
    /// Application name: reported to the Vulkan driver, and labels Metal's objects.
    pub name: &'a str,
    /// Extensions and features a library that records raw Vulkan commands needs. Vulkan only.
    pub hooks: Option<&'a dyn DeviceHooks>,
    pub queues: QueueRequest,
    pub diagnostics: Diagnostics<'a>,
    /// The file of the pipeline cache every pipeline compiles through (a `VkPipelineCache`,
    /// or Metal pipeline archives): read at creation, written by
    /// [`Device::save_pipeline_cache`](crate::Device::save_pipeline_cache). Loading it sets its
    /// modification time, so a directory of cache files trims as a pipeline store does. `None`:
    /// no cache beyond the driver's own.
    pub pipeline_cache: Option<&'a Path>,
    /// A directory holding each pipeline's compiled code in a file of its own, which takes the
    /// place of `pipeline_cache`. The device loads all of it at creation, so its size is a cost
    /// to every process. A pipeline's file is written when the pipeline compiles, so nothing
    /// needs saving and a killed process keeps what it finished. Processes share it: one compiles
    /// a pipeline while the others wait for it, for as long as what they wait for hits. Files
    /// stay until the caller removes them, as [`trim_pipeline_store`] does; a file's
    /// modification time says when a pipeline last came from it. Vulkan only.
    pub pipeline_store: Option<&'a Path>,
    /// Optional extensions to leave off even when the device offers them; their `Caps`
    /// stay false. Vulkan only.
    pub skip_extensions: &'a [&'a CStr],
    /// Reports `Caps::rebar`, `Caps::staging_device_local` and `Caps::host_image_copy` false
    /// and, on Vulkan, puts [`MemoryKind::Upload`](crate::MemoryKind::Upload) in host memory
    /// whatever the device offers, so callers take the staging path, as on a discrete device
    /// without resizable BAR.
    pub force_staging: bool,
    /// Loads the Nsight Perf SDK and enables the extensions its range profiler needs, for
    /// counter frames ([`crate::Caps::perf_counters`]). Builds without `NSIGHT_PERF_SDK`
    /// ignore it. Vulkan only.
    pub perf_counters: bool,
    /// Image heap slots for sampled and storage image views.
    pub image_heap_slots: u32,
    pub sampler_heap_slots: u32,
    /// Reports at most this many [`ClusterCaps::max_partitions`](crate::ClusterCaps), so a test
    /// reaches the limit with a small scene. `None`: what the device offers.
    pub max_partitions: Option<u32>,
}

impl Default for DeviceDesc<'_> {
    fn default() -> Self {
        Self {
            name: "orr",
            hooks: None,
            queues: QueueRequest::default(),
            diagnostics: Diagnostics::Off,
            pipeline_cache: None,
            pipeline_store: None,
            skip_extensions: &[],
            force_staging: false,
            perf_counters: false,
            image_heap_slots: 262_144,
            sampler_heap_slots: 1024,
            max_partitions: None,
        }
    }
}

/// Removes the files of the pipeline store at `dir` used longest ago, by modification time,
/// until the rest hold at most `max_bytes`; a missing `dir` is an empty store. A file the OS
/// won't remove, as Windows won't one another process holds open, stays; one removed while
/// another process reads the store is only a miss there.
pub fn trim_pipeline_store(dir: &Path, max_bytes: u64) -> Result<(), Error> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(Error::Io {
                path: dir.to_path_buf(),
                source: e,
            });
        }
    };
    // An entry whose metadata is gone was removed since the listing.
    let mut files: Vec<_> = entries
        .flatten()
        .filter_map(|entry| {
            let meta = entry.metadata().ok()?;
            Some((meta.modified().ok()?, meta.len(), entry.path()))
        })
        .collect();
    files.sort();
    let mut total: u64 = files.iter().map(|(_, len, _)| len).sum();
    for (_, len, path) in files {
        if total <= max_bytes {
            break;
        }
        match std::fs::remove_file(&path) {
            Err(e) if e.kind() != ErrorKind::NotFound => {}
            _ => total -= len,
        }
    }
    Ok(())
}
