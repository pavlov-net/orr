use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Compiler, MTL4CompilerDescriptor, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily,
    MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor, MTLSharedEvent,
};

use orr_memory::HeapPool;

use super::accel::{EmptyTlas, build_empty_tlas};
use super::cache::PipelineCache;
use super::cluster::{self, Kernels, Objects, PtlasCheck, StructureIds};
use super::heap::Bindings;
use super::queue::{Queue, TimestampRing};
use super::{Buffers, Garbage, MAX_TIMELINES, Obj, Release, Timeline, lock};
use crate::rtas::{ClusterCaps, ClusterVertexFormat};
use crate::{
    Caps, DeviceDesc, Error, Limits, MAX_INLINE_ROOT_BYTES, MemoryUsage, PassResult, QueueKind,
    TimelinePoint, Tlas,
};

/// Additions to device creation. Metal devices take none, so the trait has no methods here.
pub trait DeviceHooks {}

impl std::fmt::Debug for dyn DeviceHooks + '_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeviceHooks")
    }
}

/// What every handle of one device shares.
pub(super) struct Shared {
    pub(super) device: Obj<ProtocolObject<dyn MTLDevice>>,
    pub(super) compiler: Obj<ProtocolObject<dyn MTL4Compiler>>,
    pub(super) pipeline_cache: PipelineCache,
    /// Every allocation; committed before a submit when `dirty`.
    pub(super) residency: Mutex<Residency>,
    /// Indexed by [`QueueKind`].
    pub(super) timelines: [Timeline; MAX_TIMELINES],
    /// The events of [`HostTimeline`]s, whose points' timelines count on from
    /// `MAX_TIMELINES`. Kept with the device.
    ///
    /// [`HostTimeline`]: super::HostTimeline
    pub(super) host_timelines: Mutex<Vec<Obj<ProtocolObject<dyn MTLSharedEvent>>>>,
    pub(super) garbage: Mutex<Garbage>,
    /// Command lists that releases deferred while they are open wait for.
    pub(super) lists: Mutex<crate::queue::OpenLists>,
    /// Held while releases run, so that a thread finding nothing left to take still returns
    /// after what another thread took: [`Device::pass_results`] reads what they write.
    pub(super) collecting: Mutex<()>,
    pub(super) buffers: Mutex<Buffers>,
    /// The heaps `gpu_malloc` places buffers in, by [`MemoryKind`](crate::MemoryKind).
    pub(super) heaps: [Mutex<HeapPool>; crate::MemoryKind::ALL.len()],
    /// Live `gpu_malloc` memory and textures of their own by label and kind.
    pub(super) ledger: crate::memory::Ledger,
    /// Textures with memory of their own, outside the heaps.
    pub(super) own_textures: Mutex<MemoryUsage>,
    /// Heaps the pools have created so far.
    pub(super) device_allocations: AtomicU64,
    /// Bytes of freed buffers' heaps of their own whose release waits for the GPU.
    pub(super) freeing: AtomicU64,
    /// Pass results by the point of their submit, oldest first.
    pub(super) results: Mutex<VecDeque<(TimelinePoint, Vec<PassResult>)>>,
    pub(super) timestamps: TimestampRing,
    /// Cluster and partitioned acceleration structures by the address of their object.
    pub(super) objects: Mutex<Objects>,
    pub(super) structure_ids: StructureIds,
    /// The debug PTLAS check, created by the first PTLAS build of a debug build.
    pub(super) ptlas_check: OnceLock<PtlasCheck>,
    pub(super) cluster_kernels: OnceLock<Kernels>,
    /// Command buffers that completed with an error.
    pub(super) gpu_errors: AtomicU32,
    /// Metal libraries by the hash that names them (a module's own by the module's hash, a
    /// reload's by its contents') and whether they hold the pixel entries.
    pub(super) libraries: Mutex<HashMap<(u64, bool), Obj<Library>>>,
    pub(super) bindings: Bindings,
    pub(super) caps: Caps,
}

type Library = ProtocolObject<dyn MTLLibrary>;

pub(super) struct Residency {
    pub(super) set: Obj<ProtocolObject<dyn MTLResidencySet>>,
    pub(super) dirty: bool,
}

impl Shared {
    /// Runs `release` once everything submitted so far has completed and every command list
    /// open now has resolved: the lists may have recorded uses of what it releases.
    pub(super) fn defer(&self, release: Release) {
        let lists = lock(&self.lists).next();
        let after = self.submitted();
        lock(&self.garbage).push(after, lists, release);
    }

    /// Runs `release` once `value` of `timeline` has completed.
    pub(super) fn defer_until(&self, timeline: u32, value: u64, release: Release) {
        assert!(
            (timeline as usize) < MAX_TIMELINES,
            "frees wait for a queue's points, not a host timeline's"
        );
        let mut after = [0; MAX_TIMELINES];
        after[timeline as usize] = value;
        lock(&self.garbage).push(after, 0, release);
    }

    pub(super) fn submitted(&self) -> [u64; MAX_TIMELINES] {
        std::array::from_fn(|i| self.timelines[i].submitted.load(Ordering::Acquire))
    }

    /// Runs the releases whose points have completed, and those they queue in turn, such as
    /// a finished list's allocations, which free once what was submitted when they dropped has.
    pub(super) fn collect(&self) {
        let _collecting = lock(&self.collecting);
        let completed: [u64; MAX_TIMELINES] =
            std::array::from_fn(|i| self.timelines[i].completed());
        let first_unresolved = lock(&self.lists).first_unresolved(&completed);
        autoreleasepool(|_| {
            loop {
                let done = lock(&self.garbage).take_done(&completed, first_unresolved);
                if done.is_empty() {
                    break;
                }
                for release in done {
                    release();
                }
            }
        });
    }

    /// The event behind `timeline`: a queue's or a host timeline's.
    pub(super) fn event(&self, timeline: u32) -> Obj<ProtocolObject<dyn MTLSharedEvent>> {
        match self.timelines.get(timeline as usize) {
            Some(t) => t.event.clone(),
            None => lock(&self.host_timelines)[timeline as usize - MAX_TIMELINES].clone(),
        }
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
    /// Built in [`Device::new`]; outside `Shared`, since it holds the shared state itself.
    empty_tlas: Arc<OnceLock<EmptyTlas>>,
    /// Pipelines come only from the cache: [`Device::cached_pipelines_only`].
    pub(super) cached_only: bool,
}

impl PartialEq for Device {
    /// Whether both handles are clones of one device.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }
}

impl Eq for Device {}

impl Device {
    /// The system's default GPU, which must support Metal 4 and filter 32-bit floats.
    pub fn new(desc: &DeviceDesc<'_>) -> Result<Device, Error> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)?;
        let name = device.name().to_string();
        if !device.supportsFamily(MTLGPUFamily::Metal4) {
            return Err(Error::Unsupported {
                device: name,
                missing: "Metal 4".to_owned(),
            });
        }
        // Float height maps are R32Float, read through the material's linear sampler.
        if !device.supports32BitFloatFiltering() {
            return Err(Error::Unsupported {
                device: name,
                missing: "filtering of 32-bit float textures".to_owned(),
            });
        }
        let serializer = PipelineCache::serializer(&device, desc.pipeline_cache);
        let compiler_desc = MTL4CompilerDescriptor::new();
        compiler_desc.setPipelineDataSetSerializer(serializer.as_deref());
        let compiler = device
            .newCompilerWithDescriptor_error(&compiler_desc)
            .map_err(|e| metal_error("newCompilerWithDescriptor", &e))?;
        let pipeline_cache = PipelineCache::open(&device, desc.pipeline_cache, serializer);
        let residency_desc = MTLResidencySetDescriptor::new();
        residency_desc.setLabel(Some(&NSString::from_str(desc.name)));
        let set = device
            .newResidencySetWithDescriptor_error(&residency_desc)
            .map_err(|e| metal_error("newResidencySetWithDescriptor", &e))?;
        let timelines = std::array::from_fn(|_| {
            Timeline::new(Obj(device.newSharedEvent().expect("newSharedEvent")))
        });
        let caps = caps(&device, desc);
        let timestamps = TimestampRing::new(&device)?;
        let device = Obj(device);
        let bindings = Bindings::new(&device, desc)?;
        let heaps = super::memory::heap_pools(&device).map(Mutex::new);
        let shared = Arc::new(Shared {
            device,
            compiler: Obj(compiler),
            pipeline_cache,
            residency: Mutex::new(Residency {
                set: Obj(set),
                dirty: true,
            }),
            timelines,
            host_timelines: Mutex::default(),
            garbage: Mutex::default(),
            lists: Mutex::default(),
            collecting: Mutex::default(),
            buffers: Mutex::default(),
            heaps,
            ledger: crate::memory::Ledger::default(),
            own_textures: Mutex::default(),
            device_allocations: AtomicU64::new(0),
            freeing: AtomicU64::new(0),
            results: Mutex::default(),
            timestamps,
            objects: Mutex::default(),
            structure_ids: StructureIds::default(),
            ptlas_check: OnceLock::new(),
            cluster_kernels: OnceLock::new(),
            gpu_errors: AtomicU32::new(0),
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
        let device = Device {
            shared,
            queues: Arc::new(queues),
            empty_tlas: Arc::default(),
            cached_only: false,
        };
        let empty = build_empty_tlas(&device)?;
        let _ = device.empty_tlas.set(empty);
        Ok(device)
    }

    pub fn caps(&self) -> &Caps {
        &self.shared.caps
    }

    /// The queue of `kind`, if the device created it.
    pub fn queue(&self, kind: QueueKind) -> Option<&Queue> {
        self.queues[kind as usize].as_ref()
    }

    /// A handle to the same device whose pipeline creation compiles nothing: a pipeline this
    /// session or the cache's archive does not hold fails with [`Error::CompileRequired`].
    pub fn cached_pipelines_only(&self) -> Device {
        Device {
            cached_only: true,
            ..self.clone()
        }
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

    /// Whether a call has reported the device lost; Metal reports no loss.
    pub fn is_lost(&self) -> bool {
        false
    }

    /// A TLAS with one inactive instance: every ray misses it. For kernels with nothing to
    /// trace; `Tlas(0)` is never traced.
    pub fn empty_tlas(&self) -> Tlas {
        self.empty_tlas
            .get()
            .expect("built in Device::new")
            .tlas
            .tlas()
    }

    /// Command buffers the GPU failed: a fault or a validation error, each logged with
    /// Metal's description as it completes; and PTLAS builds the debug check rejected among
    /// completed work.
    pub fn validation_messages(&self) -> u32 {
        let rejected = self
            .shared
            .ptlas_check
            .get()
            .map_or(0, PtlasCheck::failures);
        self.shared.gpu_errors.load(Ordering::Relaxed) + rejected
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
        rebar: !desc.force_staging,
        staging_device_local: !desc.force_staging,
        host_image_copy: false,
        memory_decompression: false,
        host_memory_import: false,
        occupancy_priority: false,
        generated_commands: false,
        ray_dispatch: false,
        reorder: false,
        lss: false,
        omm: false,
        device_address_commands: false,
        indirect_copy: false,
        unified_image_layouts: true,
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
        // Presented handlers report each drawable's present.
        present_wait: desc.queues.present,
        low_latency: false,
        present_metering: false,
        present_timing: false,
        // `CAEDRMetadata` on the layer.
        hdr_metadata: desc.queues.present,
        // Emulated over classic structures (cluster.rs).
        cluster: ClusterCaps {
            formats: &[ClusterVertexFormat::F32x3],
            max_tris: 256,
            max_verts: 256,
            align_clas: cluster::ALIGN,
            align_template: cluster::ALIGN,
            align_blas: cluster::ALIGN,
            align_scratch: 256,
            align_bounds: 32,
            max_geometry_index: (1 << 24) - 1,
            omm_in_clusters: false,
            max_partitions: desc.max_partitions.map_or(1 << 16, |max| max.min(1 << 16)),
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
            // Metal's feature tables name no such total for mesh grids that this was checked
            // against, so draws keep to the least `VK_EXT_mesh_shader` guarantees.
            max_mesh_draw_groups: 1 << 22,
            max_ray_dispatch_invocations: 0,
            device_local_bytes: working_set,
            upload_bytes: working_set,
            max_memory_allocations: u32::MAX,
            max_decompress_regions: 0,
            max_generated_pipelines: 0,
            max_generated_sequences: 0,
            host_import_align: 0,
        },
    }
}
