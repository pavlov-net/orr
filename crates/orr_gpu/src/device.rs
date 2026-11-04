//! How to create a device, alike on every backend.

use std::ffi::CStr;
use std::path::Path;

use crate::{DeviceHooks, Diagnostics, QueueRequest};

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
    /// [`Device::save_pipeline_cache`](crate::Device::save_pipeline_cache). `None`: no cache
    /// beyond the driver's own.
    pub pipeline_cache: Option<&'a Path>,
    /// Optional extensions to leave off even when the device offers them; their `Caps`
    /// stay false. Vulkan only.
    pub skip_extensions: &'a [&'a CStr],
    /// Reports `Caps::rebar` and `Caps::host_image_copy` false and, on Vulkan, puts
    /// [`MemoryKind::Upload`](crate::MemoryKind::Upload) in host memory whatever the device
    /// offers, so callers take the staging path, as on a device without resizable BAR.
    pub force_staging: bool,
    /// Loads the Nsight Perf SDK and enables the extensions its range profiler needs, for
    /// counter frames ([`crate::Caps::perf_counters`]). Builds without `NSIGHT_PERF_SDK`
    /// ignore it. Vulkan only.
    pub perf_counters: bool,
    /// Image heap slots for sampled and storage image views.
    pub image_heap_slots: u32,
    pub sampler_heap_slots: u32,
}

impl Default for DeviceDesc<'_> {
    fn default() -> Self {
        Self {
            name: "orr",
            hooks: None,
            queues: QueueRequest::default(),
            diagnostics: Diagnostics::Off,
            pipeline_cache: None,
            skip_extensions: &[],
            force_staging: false,
            perf_counters: false,
            image_heap_slots: 262_144,
            sampler_heap_slots: 1024,
        }
    }
}
