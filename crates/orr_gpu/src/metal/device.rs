use std::collections::HashMap;
use std::ffi::CStr;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Compiler, MTL4CompilerDescriptor, MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent,
    MTLGPUFamily, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor, MTLSharedEvent,
};

use super::heap::Bindings;
use super::queue::{Queue, QueueKind, QueueRequest};
use super::unsupported::Diagnostics;
use super::{Buffers, Garbage, MAX_TIMELINES, Obj, Release, Timeline, lock};
use crate::rtas::ClusterCaps;
use crate::{Caps, Error, Limits, MAX_INLINE_ROOT_BYTES, Tlas};

/// Additions to device creation. Metal devices take none, so the trait has no methods here.
pub trait DeviceHooks {}

impl std::fmt::Debug for dyn DeviceHooks + '_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeviceHooks")
    }
}

/// How to create a [`Device`].
#[derive(Clone, Copy, Debug)]
pub struct DeviceDesc<'a> {
    /// Application name; labels the device's objects.
    pub name: &'a str,
    /// Unused on Metal.
    pub hooks: Option<&'a dyn DeviceHooks>,
    pub queues: QueueRequest,
    pub diagnostics: Diagnostics<'a>,
    /// Unused on Metal: there is no pipeline cache yet.
    pub pipeline_cache: Option<&'a Path>,
    /// Unused on Metal.
    pub skip_extensions: &'a [&'a CStr],
    /// Unused on Metal: memory is unified, so there is no staging path.
    pub force_staging: bool,
    /// Unused on Metal.
    pub perf_counters: bool,
    /// Texture view pool slots for sampled and storage image views.
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

/// What every handle of one device shares.
pub(super) struct Shared {
    pub(super) device: Obj<ProtocolObject<dyn MTLDevice>>,
    pub(super) compiler: Obj<ProtocolObject<dyn MTL4Compiler>>,
    /// Every allocation; committed before a submit when `dirty`.
    pub(super) residency: Mutex<Residency>,
    /// Indexed by [`QueueKind`].
    pub(super) timelines: [Timeline; MAX_TIMELINES],
    pub(super) garbage: Mutex<Garbage>,
    pub(super) buffers: Mutex<Buffers>,
    /// Each module's libraries by (module hash, fragment library).
    pub(super) libraries: Mutex<HashMap<(u64, bool), Obj<ProtocolObject<dyn MTLLibrary>>>>,
    pub(super) bindings: Bindings,
    pub(super) caps: Caps,
}

pub(super) struct Residency {
    pub(super) set: Obj<ProtocolObject<dyn MTLResidencySet>>,
    pub(super) dirty: bool,
}

impl Shared {
    /// Runs `release` once everything submitted so far has completed.
    pub(super) fn defer(&self, release: Release) {
        let after = self.submitted();
        lock(&self.garbage).push(after, release);
    }

    /// Runs `release` once `value` of `timeline` has completed.
    pub(super) fn defer_until(&self, timeline: u32, value: u64, release: Release) {
        let mut after = [0; MAX_TIMELINES];
        after[timeline as usize] = value;
        lock(&self.garbage).push(after, release);
    }

    pub(super) fn submitted(&self) -> [u64; MAX_TIMELINES] {
        std::array::from_fn(|i| self.timelines[i].submitted.load(Ordering::Acquire))
    }

    pub(super) fn collect(&self) {
        let completed = std::array::from_fn(|i| self.timelines[i].completed());
        lock(&self.garbage).collect(&completed);
    }

    /// Makes the allocations added or removed since the last commit resident for the next
    /// submits.
    pub(super) fn commit_residency(&self) {
        let mut r = lock(&self.residency);
        if r.dirty {
            r.set.commit();
            r.dirty = false;
        }
    }
}

/// The GPU: memory, heaps, pipelines and queues. Cheap to clone; every clone is the same
/// device.
#[derive(Clone)]
pub struct Device {
    pub(super) shared: Arc<Shared>,
    queues: Arc<[Option<Queue>; MAX_TIMELINES]>,
}

impl Device {
    /// The system's default GPU, which must support Metal 4.
    pub fn new(desc: &DeviceDesc<'_>) -> Result<Device, Error> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)?;
        let name = device.name().to_string();
        if !device.supportsFamily(MTLGPUFamily::Metal4) {
            return Err(Error::Unsupported {
                device: name,
                missing: "Metal 4".to_owned(),
            });
        }
        let compiler = device
            .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
            .map_err(|e| metal_error("newCompilerWithDescriptor", &e))?;
        let residency_desc = MTLResidencySetDescriptor::new();
        residency_desc.setLabel(Some(&NSString::from_str(desc.name)));
        let set = device
            .newResidencySetWithDescriptor_error(&residency_desc)
            .map_err(|e| metal_error("newResidencySetWithDescriptor", &e))?;
        let timelines = std::array::from_fn(|_| Timeline {
            event: Obj(device.newSharedEvent().expect("newSharedEvent")),
            submitted: AtomicU64::new(0),
        });
        let caps = caps(&device, desc);
        let device = Obj(device);
        let bindings = Bindings::new(&device, desc)?;
        let shared = Arc::new(Shared {
            device,
            compiler: Obj(compiler),
            residency: Mutex::new(Residency {
                set: Obj(set),
                dirty: true,
            }),
            timelines,
            garbage: Mutex::default(),
            buffers: Mutex::default(),
            libraries: Mutex::default(),
            bindings,
            caps,
        });
        bindings_resident(&shared);
        let q = desc.queues;
        let wanted = [true, q.render_compute, q.async_compute, q.copy, q.present];
        let mut queues: [Option<Queue>; MAX_TIMELINES] = Default::default();
        for (kind, slot) in QueueKind::ALL.into_iter().zip(&mut queues) {
            if wanted[kind as usize] {
                *slot = Some(Queue::new(&shared, kind)?);
            }
        }
        Ok(Device {
            shared,
            queues: Arc::new(queues),
        })
    }

    pub fn caps(&self) -> &Caps {
        &self.shared.caps
    }

    /// The queue of `kind`, if the device created it.
    pub fn queue(&self, kind: QueueKind) -> Option<&Queue> {
        self.queues[kind as usize].as_ref()
    }

    /// The same device. Metal has no pipeline cache yet, so pipelines compile as usual.
    pub fn cached_pipelines_only(&self) -> Device {
        self.clone()
    }

    /// Waits for everything submitted so far on every queue.
    pub fn wait_submitted(&self, timeout: Duration) -> Result<(), Error> {
        let submitted = self.shared.submitted();
        for (t, value) in self.shared.timelines.iter().zip(submitted) {
            let ms = timeout.as_millis().try_into().unwrap_or(u64::MAX);
            if !t.event.waitUntilSignaledValue_timeoutMS(value, ms) {
                return Err(Error::Timeout);
            }
        }
        Ok(())
    }

    /// A TLAS with no instances.
    pub fn empty_tlas(&self) -> Tlas {
        super::not_implemented("Device::empty_tlas")
    }

    /// Messages from Metal's validation; the backend does not collect them yet.
    pub fn validation_messages(&self) -> u32 {
        0
    }
}

/// Adds the binding tables to the residency set.
fn bindings_resident(shared: &Shared) {
    let r = lock(&shared.residency);
    for buffer in shared.bindings.buffers() {
        r.set.addAllocation(ProtocolObject::from_ref(&**buffer));
    }
}

pub(super) fn metal_error(call: &'static str, e: &objc2_foundation::NSError) -> Error {
    Error::Metal {
        call,
        message: e.localizedDescription().to_string(),
    }
}

fn caps(device: &ProtocolObject<dyn MTLDevice>, desc: &DeviceDesc<'_>) -> Caps {
    let threads = device.maxThreadsPerThreadgroup();
    let working_set = device.recommendedMaxWorkingSetSize();
    Caps {
        device_name: device.name().to_string(),
        vendor_id: 0x106b,
        device_id: device.registryID() as u32,
        driver: "Metal 4".to_owned(),
        driver_version: 0,
        rebar: true,
        host_image_copy: false,
        memory_decompression: false,
        occupancy_priority: false,
        generated_commands: false,
        ray_dispatch: false,
        reorder: false,
        lss: false,
        omm: false,
        device_address_commands: false,
        indirect_copy: false,
        unified_image_layouts: true,
        memory_budget: false,
        memory_priority: false,
        shader_abort: false,
        device_fault: false,
        breadcrumbs: false,
        checkpoints: false,
        aftermath: false,
        cooperative_vector: Vec::new(),
        shader_float8: false,
        calibrated_timestamps: false,
        perf_counters: false,
        frame_boundary: false,
        pipeline_executable_info: false,
        conservative_raster: false,
        ray_tracing_validation: false,
        present_wait: false,
        low_latency: false,
        present_metering: false,
        present_timing: false,
        hdr_metadata: false,
        cluster: ClusterCaps {
            formats: &[],
            max_tris: 0,
            max_verts: 0,
            align_clas: 0,
            align_template: 0,
            align_blas: 0,
            align_scratch: 0,
            align_bounds: 0,
            max_geometry_index: 0,
            omm_in_clusters: false,
            max_partitions: 0,
        },
        limits: Limits {
            image_descriptor_bytes: 8,
            sampler_descriptor_bytes: 8,
            max_image_slots: desc.image_heap_slots,
            max_sampler_slots: desc.sampler_heap_slots,
            max_push_data_bytes: MAX_INLINE_ROOT_BYTES as u32 + 8,
            timestamp_period_ns: 1.0,
            max_workgroup_size: [
                threads.width as u32,
                threads.height as u32,
                threads.depth as u32,
            ],
            max_workgroup_invocations: 1024,
            max_workgroup_count: [u32::MAX; 3],
            max_image_extent_2d: 16384,
            max_image_extent_3d: 2048,
            max_image_layers: 2048,
            max_color_targets: 8,
            max_ray_dispatch_invocations: 0,
            device_local_bytes: working_set,
            upload_bytes: working_set,
            max_decompress_regions: 0,
            max_generated_pipelines: 0,
            max_generated_sequences: 0,
        },
    }
}
