//! Cluster and partitioned acceleration structures, which Metal lacks, emulated over its
//! classic ones.
//!
//! Every object lives in the caller's memory behind a 64-byte header (its kind, size and the
//! Metal resource ID of the acceleration structure standing for it), so moves are copies and
//! kernels find a structure from an address. The Metal structures themselves live in the
//! backend's own allocations; [`Objects`] keeps them by the address of their object and
//! releases one when another object is placed over its memory or that memory is freed.
//!
//! - A CLAS holds its triangles (vertices, indices and per-triangle data: cluster ID,
//!   geometry index and flags, triangle in the cluster) and a Metal BLAS over them.
//!   A CLAS of an op marked [`RtasFlags::FLAT_ONLY`] has no Metal BLAS.
//! - A cluster BLAS marked [`RtasFlags::TRANSIENT`] is an instance structure over its CLAS'
//!   BLAS, a cheap build that traces through three levels; any other is one flat BLAS over
//!   its CLAS' triangles, slower to build and faster to trace.
//! - A PTLAS keeps its instances' last writes and its partitions' translations, and is
//!   rebuilt whole from them, with the instance count read on the GPU.
//!
//! Metal builds only what the CPU encodes with exact counts, so cluster ops copy their
//! records back and finish in a CPU step ([`CommandList::split`]) once the GPU has written
//! them. PTLAS builds need none.
//!
//! Metal finds a structure from its ID through a table that grows as structures are created,
//! and work encoded before a growth misses IDs past the old size: its traces and TLAS builds
//! see no geometry. A CPU step creates structures after the work that uses them was encoded,
//! so recording the op grows the table first ([`StructureIds`]).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString};
use objc2_metal::{
    MTL4AccelerationStructureDescriptor, MTL4AccelerationStructureGeometryDescriptor,
    MTL4AccelerationStructureTriangleGeometryDescriptor, MTL4BufferRange,
    MTL4ComputeCommandEncoder, MTL4IndirectInstanceAccelerationStructureDescriptor,
    MTL4InstanceAccelerationStructureDescriptor, MTL4LibraryFunctionDescriptor,
    MTL4PrimitiveAccelerationStructureDescriptor, MTLAccelerationStructure,
    MTLAccelerationStructureDescriptor, MTLAccelerationStructureInstanceDescriptorType,
    MTLAccelerationStructureUsage, MTLAttributeFormat, MTLBuffer, MTLComputePipelineState,
    MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLIndexType,
    MTLResidencySet, MTLResource, MTLResourceOptions, MTLStorageMode,
};

use super::command::CommandList;
use super::device::{Device, Shared, metal_error};
use super::heap::id_bits;
use super::memory::{Allocation, malloc};
use super::{Obj, lock};
use crate::rtas::{PtlasArgs, RtasIo, RtasLimits, RtasMode, RtasOp, RtasOpDesc, RtasSizes};
use crate::{Error, GpuAddress, GpuPtr, Hazards, MemoryKind, RtasFlags, Stages};

type Structure = ProtocolObject<dyn MTLAccelerationStructure>;

/// A structure, what to build it from and the scratch the build needs.
type Build = (
    Obj<Structure>,
    Retained<MTL4AccelerationStructureDescriptor>,
    u64,
);

/// Builds in one encoder. Metal flushes an encoder's builds as one batch, which at hundreds of
/// thousands takes minutes on the CPU and again on the GPU, past any timeout that detects a
/// hang; smaller batches let the queue wait for each in turn.
const BUILDS_PER_ENCODER: usize = 4096;

/// The builds left for a later CPU step.
struct RemainingBuilds(Vec<Build>);

// SAFETY: the step takes the descriptors whole; no other thread holds them.
unsafe impl Send for RemainingBuilds {}

const SOURCE: &str = include_str!("cluster.msl");
const HEADER: u64 = 64;
/// Bytes of a triangle's data: cluster ID, geometry index and flags, triangle in the
/// cluster, and its three corners, which hits fetch since Metal cannot.
const DATA_BYTES: u64 = 48;
/// Alignment of objects in the caller's memory.
pub(super) const ALIGN: u32 = 64;
/// Alignment of a PTLAS.
const PTLAS_ALIGN: u64 = 256;
const PTLAS_HEADER: u64 = 256;
/// Bytes of a stored PTLAS write and of a Metal indirect instance descriptor.
const WRITE_BYTES: u64 = 104;
const INSTANCE_BYTES: u64 = 72;
/// Threads per group of every kernel here.
const GROUP: u32 = 64;

/// Strides of the records `rtas` `store` writes: the Vulkan structs.
const CLAS_BUILD_STRIDE: u32 = 64;
const TEMPLATE_BUILD_STRIDE: u32 = 72;
const INSTANTIATE_STRIDE: u32 = 32;
const CLUSTER_BLAS_STRIDE: u32 = 16;
const MOVE_STRIDE: u32 = 8;
const PTLAS_COMMAND_STRIDE: u32 = 24;
const PTLAS_WRITE_STRIDE: u32 = 104;
const PTLAS_UPDATE_STRIDE: u32 = 16;
const PTLAS_TRANSLATE_STRIDE: u32 = 16;

/// Object kinds, as header word 0 and the kernels name them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kind {
    Clas = 1,
    Template = 2,
    BlasInstanced = 3,
    BlasFlat = 4,
    Ptlas = 5,
}

/// What the CPU knows of an object at an address.
pub(super) struct Object {
    end: u64,
    kind: Kind,
    tris: u32,
    verts: u32,
    /// Every triangle opaque, so its geometry is.
    opaque: bool,
    raw: Option<Obj<Structure>>,
}

/// The objects the backend placed, by start address.
#[derive(Default)]
pub(super) struct Objects {
    by_start: BTreeMap<u64, Object>,
}

impl Objects {
    /// Removes the objects overlapping `[start, end)` and returns their structures.
    fn evict(&mut self, start: u64, end: u64) -> Vec<Obj<Structure>> {
        let overlapping: Vec<u64> = self
            .by_start
            .range(..end)
            .rev()
            .take_while(|(_, o)| o.end > start)
            .map(|(&s, _)| s)
            .collect();
        overlapping
            .into_iter()
            .filter_map(|s| self.by_start.remove(&s).and_then(|o| o.raw))
            .collect()
    }

    fn insert(&mut self, start: u64, o: Object) -> Vec<Obj<Structure>> {
        let gone = self.evict(start, o.end);
        self.by_start.insert(start, o);
        gone
    }
}

/// The acceleration-structure IDs Metal's table holds. A CPU step hands out IDs no higher
/// than the highest so far plus the structures recorded steps may still create, so
/// recording an op makes the table hold those first.
#[derive(Default)]
pub(super) struct StructureIds {
    /// The highest ID a structure the backend keeps has had.
    high: AtomicU64,
    /// Structures that recorded CPU steps may still create.
    pending: AtomicU64,
    /// IDs the table holds at least. Held while growing it.
    held: std::sync::Mutex<u64>,
}

impl StructureIds {
    pub(super) fn note(&self, s: &Structure) {
        self.high
            .fetch_max(id_bits(s.gpuResourceID()), Ordering::Relaxed);
    }
}

/// Structures a recorded CPU step may create; released with the step.
struct Reservation {
    shared: Arc<Shared>,
    count: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.shared
            .structure_ids
            .pending
            .fetch_sub(self.count, Ordering::Relaxed);
    }
}

impl Shared {
    /// Makes room in Metal's ID table for `count` more structures, created by a CPU step
    /// after work recorded before it. Grows the table by creating throwaway structures,
    /// to the next power of two past what is needed, so growths stay rare.
    fn reserve_structures(self: &Arc<Self>, count: u64) -> Reservation {
        let ids = &self.structure_ids;
        let pending = ids.pending.fetch_add(count, Ordering::Relaxed) + count;
        let reservation = Reservation {
            shared: Arc::clone(self),
            count,
        };
        let need = ids.high.load(Ordering::Relaxed) + pending + 1;
        let mut held = lock(&ids.held);
        if need > *held {
            let target = need.next_power_of_two().max(256);
            // SAFETY: any size is valid to ask about.
            let placed = unsafe {
                self.device
                    .heapAccelerationStructureSizeAndAlignWithSize(16)
            };
            let desc = MTLHeapDescriptor::new();
            desc.setType(MTLHeapType::Placement);
            desc.setStorageMode(MTLStorageMode::Private);
            desc.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
            desc.setSize(placed.size);
            let heap = self
                .device
                .newHeapWithDescriptor(&desc)
                .expect("newHeapWithDescriptor");
            // New IDs take the lowest free ones: the last of `target` structures has an ID
            // at least `target - 1`, which makes the table hold `target`. They all alias one
            // heap range because device allocations scan a free list per object, which takes
            // minutes at a million structures.
            let mut throwaway = Vec::new();
            while (throwaway.len() as u64) < target {
                // SAFETY: the structure fills the heap, and nothing builds or traces it.
                let s = unsafe { heap.newAccelerationStructureWithSize_offset(16, 0) }
                    .expect("newAccelerationStructureWithSize");
                let id = id_bits(s.gpuResourceID());
                throwaway.push(s);
                if id + 1 >= target {
                    break;
                }
            }
            *held = target;
        }
        reservation
    }

    /// Forgets the objects in `[start, end)`, memory being freed, and releases their
    /// structures; runs when nothing submitted uses them any more.
    pub(super) fn forget_objects(&self, start: u64, end: u64) {
        let gone = lock(&self.objects).evict(start, end);
        self.release_structures(gone);
    }

    fn release_structures(&self, gone: Vec<Obj<Structure>>) {
        if gone.is_empty() {
            return;
        }
        let mut r = lock(&self.residency);
        for s in &gone {
            r.set.removeAllocation(ProtocolObject::from_ref(&**s));
        }
        r.dirty = true;
    }

    /// Places `o` at `start`; the structures it covers are released once everything
    /// submitted before has completed.
    fn place(self: &Arc<Self>, start: u64, o: Object) {
        if let Some(raw) = &o.raw {
            let mut r = lock(&self.residency);
            r.set.addAllocation(ProtocolObject::from_ref(&**raw));
            r.dirty = true;
        }
        let gone = lock(&self.objects).insert(start, o);
        if !gone.is_empty() {
            let shared = Arc::clone(self);
            self.defer(Box::new(move || shared.release_structures(gone)));
        }
    }

    /// A Metal acceleration structure for `desc`, and the scratch its build needs.
    fn structure(&self, desc: &MTL4AccelerationStructureDescriptor) -> (Obj<Structure>, u64) {
        let base: &MTLAccelerationStructureDescriptor = desc;
        let sizes = self.device.accelerationStructureSizesWithDescriptor(base);
        let raw = self
            .device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize.max(1))
            .expect("newAccelerationStructureWithSize");
        self.structure_ids.note(&raw);
        (Obj(raw), sizes.buildScratchBufferSize as u64)
    }
}

/// The kernels of `cluster.msl`, compiled once per device.
pub(super) struct Kernels {
    pack: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    blas_instances: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    blas_gather: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    copy_words: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    ptlas_clear: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    ptlas_apply: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    ptlas_descriptors: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    ptlas_check: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    ptlas_check_writes: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Kernels {
    fn get(shared: &Shared) -> &Kernels {
        shared.cluster_kernels.get_or_init(|| {
            Kernels::new(shared).unwrap_or_else(|e| panic!("compiling the cluster kernels: {e}"))
        })
    }

    fn new(shared: &Shared) -> Result<Kernels, Error> {
        let library = shared
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|e| metal_error("newLibraryWithSource", &e))?;
        let state = |name: &str| {
            let f = MTL4LibraryFunctionDescriptor::new();
            f.setName(Some(&NSString::from_str(name)));
            f.setLibrary(Some(&library));
            shared.compute_state(&f, [GROUP, 1, 1], name)
        };
        Ok(Kernels {
            pack: state("pack")?,
            blas_instances: state("blas_instances")?,
            blas_gather: state("blas_gather")?,
            copy_words: state("copy_words")?,
            ptlas_clear: state("ptlas_clear")?,
            ptlas_apply: state("ptlas_apply")?,
            ptlas_descriptors: state("ptlas_descriptors")?,
            ptlas_check: state("ptlas_check")?,
            ptlas_check_writes: state("ptlas_check_writes")?,
        })
    }
}

/// The debug PTLAS check's count of rejected builds, which the GPU adds to.
pub(super) struct PtlasCheck {
    failures: Obj<super::Buffer>,
    reported: std::sync::atomic::AtomicU32,
}

impl PtlasCheck {
    fn get(shared: &Shared) -> &PtlasCheck {
        shared.ptlas_check.get_or_init(|| {
            let options = MTLResourceOptions::StorageModeShared
                | MTLResourceOptions::HazardTrackingModeUntracked;
            let failures = shared
                .device
                .newBufferWithLength_options(4, options)
                .expect("the PTLAS check's failure count");
            failures.setLabel(Some(&NSString::from_str("orr PTLAS check failures")));
            // SAFETY: shared storage is mapped and zeroed; nothing else uses it yet.
            unsafe { failures.contents().cast::<u32>().as_ptr().write(0) };
            let mut r = lock(&shared.residency);
            r.set.addAllocation(ProtocolObject::from_ref(&*failures));
            r.dirty = true;
            PtlasCheck {
                failures: Obj(failures),
                reported: Default::default(),
            }
        })
    }

    /// Rejected builds among completed work; logs the ones not reported before.
    pub(super) fn failures(&self) -> u32 {
        // SAFETY: the word stays mapped while the buffer lives; the GPU only adds to it.
        let n = unsafe {
            self.failures
                .contents()
                .cast::<u32>()
                .as_ptr()
                .read_volatile()
        };
        let before = self.reported.swap(n, Ordering::Relaxed);
        if n > before {
            log::error!(
                "{} PTLAS build(s) rejected: more commands than PtlasLimits::max_ops, an unknown op, two commands of one op type or a write's custom index above 24 bits",
                n - before
            );
        }
        n
    }
}

/// Little-endian word `i` of `bytes`.
fn word(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
}

fn word64(bytes: &[u8], i: usize) -> u64 {
    u64::from(word(bytes, i)) | u64::from(word(bytes, i + 1)) << 32
}

fn range(address: u64, length: u64) -> MTL4BufferRange {
    MTL4BufferRange {
        bufferAddress: address,
        length,
    }
}

/// Bytes of a CLAS (`clas`) or template with `tris` triangles and `verts` vertices.
fn object_bytes(clas: bool, tris: u32, verts: u32) -> u64 {
    let v = if clas { 12 * u64::from(verts) } else { 0 };
    (HEADER + v + (12 + DATA_BYTES) * u64::from(tris)).next_multiple_of(16)
}

/// Bytes of a PTLAS's state.
fn ptlas_bytes(instances: u32, partitions: u32) -> u64 {
    PTLAS_HEADER
        + 16 * (u64::from(partitions) + 1)
        + (WRITE_BYTES + INSTANCE_BYTES) * u64::from(instances)
}

fn args_stride(op: RtasOp) -> u32 {
    match op {
        RtasOp::BuildClas => CLAS_BUILD_STRIDE,
        RtasOp::BuildTemplates => TEMPLATE_BUILD_STRIDE,
        RtasOp::InstantiateTemplates => INSTANTIATE_STRIDE,
        RtasOp::BuildClusterBlas => CLUSTER_BLAS_STRIDE,
        RtasOp::Move => MOVE_STRIDE,
        RtasOp::Ptlas => PTLAS_COMMAND_STRIDE,
    }
}

fn max_args(d: &RtasOpDesc) -> u32 {
    match (d.op, d.limits) {
        (
            RtasOp::BuildClas | RtasOp::BuildTemplates | RtasOp::InstantiateTemplates,
            RtasLimits::Cluster(l),
        ) => l.max_args,
        (RtasOp::BuildClusterBlas, RtasLimits::Blas(l)) => l.max_args,
        (RtasOp::Move, RtasLimits::Move(l)) => l.max_args,
        (RtasOp::Ptlas, RtasLimits::Ptlas(l)) => l.max_ops,
        (op, limits) => panic!("{op:?} takes other limits than {limits:?}"),
    }
}

/// The triangle geometry of a CLAS at `at`, or of the opaque or other triangles of a flat
/// BLAS: unindexed when `indices` is 0.
fn triangles(
    vertices: u64,
    verts: u64,
    indices: u64,
    tris: u32,
    data: u64,
    opaque: bool,
) -> Retained<MTL4AccelerationStructureGeometryDescriptor> {
    let g = MTL4AccelerationStructureTriangleGeometryDescriptor::new();
    // SAFETY: the ranges are objects the build's kernels wrote, live until it completes.
    unsafe {
        g.setVertexBuffer(range(vertices, 12 * verts.max(1)));
        g.setVertexStride(12);
        g.setTriangleCount(tris as usize);
        if indices != 0 {
            g.setIndexBuffer(range(indices, 12 * u64::from(tris)));
        }
        g.setPrimitiveDataBuffer(range(data, DATA_BYTES * u64::from(tris)));
        g.setPrimitiveDataElementSize(DATA_BYTES as usize);
        g.setPrimitiveDataStride(DATA_BYTES as usize);
    }
    g.setVertexFormat(MTLAttributeFormat::Float3);
    g.setIndexType(MTLIndexType::UInt32);
    g.setOpaque(opaque);
    Retained::into_super(g)
}

fn primitive(
    geometries: &[Retained<MTL4AccelerationStructureGeometryDescriptor>],
    fast_build: bool,
) -> Retained<MTL4AccelerationStructureDescriptor> {
    let p = MTL4PrimitiveAccelerationStructureDescriptor::new();
    p.setGeometryDescriptors(Some(&NSArray::from_retained_slice(geometries)));
    if fast_build {
        p.setUsage(MTLAccelerationStructureUsage::PreferFastBuild);
    }
    Retained::into_super(p)
}

fn instances(descriptors: u64, count: u32) -> Retained<MTL4AccelerationStructureDescriptor> {
    let t = MTL4InstanceAccelerationStructureDescriptor::new();
    // SAFETY: a kernel writes `count` descriptors at `descriptors` before the build.
    unsafe {
        t.setInstanceDescriptorBuffer(range(descriptors, INSTANCE_BYTES * u64::from(count.max(1))));
        t.setInstanceDescriptorStride(INSTANCE_BYTES as usize);
        t.setInstanceCount(count as usize);
    }
    t.setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::Indirect);
    t.setUsage(MTLAccelerationStructureUsage::PreferFastBuild);
    Retained::into_super(t)
}

/// The rebuilt TLAS of a PTLAS whose state is at `state`.
fn ptlas_descriptor(
    state: u64,
    instances: u32,
    partitions: u32,
) -> Retained<MTL4AccelerationStructureDescriptor> {
    let t = MTL4IndirectInstanceAccelerationStructureDescriptor::new();
    let descriptors = state
        + PTLAS_HEADER
        + 16 * (u64::from(partitions) + 1)
        + WRITE_BYTES * u64::from(instances);
    // SAFETY: the state holds `instances` descriptors and the count at header word 4.
    unsafe {
        t.setInstanceDescriptorBuffer(range(
            descriptors,
            INSTANCE_BYTES * u64::from(instances.max(1)),
        ));
        t.setInstanceDescriptorStride(INSTANCE_BYTES as usize);
        t.setMaxInstanceCount(instances as usize);
        t.setInstanceCountBuffer(range(state + 16, 4));
    }
    t.setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::Indirect);
    t.setUsage(MTLAccelerationStructureUsage::PreferFastBuild);
    Retained::into_super(t)
}

/// A header: kind, size, resource ID, then `extra` words.
fn header(kind: Kind, size: u64, id: u64, extra: &[u32]) -> [u32; 16] {
    let mut h = [0u32; 16];
    h[0] = kind as u32;
    h[1] = size as u32;
    h[2] = id as u32;
    h[3] = (id >> 32) as u32;
    h[4..4 + extra.len()].copy_from_slice(extra);
    h
}

impl Device {
    /// The memory `d` needs at most: the implicit result (or the PTLAS) and scratch. One
    /// query sizes a whole batch; for an [`RtasMode::Explicit`] build of CLAS or templates,
    /// `result_max` is the most one of its results takes.
    pub fn rtas_sizes(&self, d: &RtasOpDesc) -> RtasSizes {
        let n = u64::from(max_args(d));
        let align = u64::from(ALIGN);
        let result_max = match d.limits {
            RtasLimits::Cluster(l) if d.mode == RtasMode::Explicit => object_bytes(
                d.op != RtasOp::BuildTemplates,
                l.max_tris_per_cluster,
                l.max_verts_per_cluster,
            ),
            RtasLimits::Cluster(l) => {
                let clas = d.op != RtasOp::BuildTemplates;
                let verts = if clas {
                    12 * u64::from(l.max_total_verts)
                } else {
                    0
                };
                n * (HEADER + align) + verts + (12 + DATA_BYTES) * u64::from(l.max_total_tris)
            }
            RtasLimits::Blas(_) => n * HEADER,
            RtasLimits::Move(l) => l.max_bytes + n * align,
            RtasLimits::Ptlas(l) => {
                let desc = ptlas_descriptor(0, l.instances, l.partitions);
                let base: &MTLAccelerationStructureDescriptor = &desc;
                let sizes = self
                    .shared
                    .device
                    .accelerationStructureSizesWithDescriptor(base);
                return RtasSizes {
                    result_max: ptlas_bytes(l.instances, l.partitions),
                    scratch: (sizes.buildScratchBufferSize as u64).max(256),
                };
            }
        };
        // The backend allocates its own scratch for cluster ops.
        RtasSizes {
            result_max,
            scratch: 256,
        }
    }

    /// Bytes per argument record of `op` in the layout `rtas` `store` writes. For
    /// [`RtasOp::Ptlas`], the command record.
    pub fn rtas_args_stride(&self, op: RtasOp) -> u32 {
        args_stride(op)
    }

    /// Bytes per record of the arrays PTLAS commands point at.
    pub fn ptlas_args_stride(&self, args: PtlasArgs) -> u32 {
        match args {
            PtlasArgs::Write => PTLAS_WRITE_STRIDE,
            PtlasArgs::Update => PTLAS_UPDATE_STRIDE,
            PtlasArgs::Translate => PTLAS_TRANSLATE_STRIDE,
        }
    }
}

/// A cluster op's records as the GPU wrote them, read back for its CPU step.
struct Readback {
    memory: Allocation<u8>,
    stride: usize,
    /// Byte offset of the explicit destinations, after the records.
    destinations: usize,
}

impl Readback {
    fn bytes(&self) -> &[u8] {
        self.memory.as_slice().expect("readback is mapped")
    }

    fn count(&self, max: u32) -> u32 {
        word(self.bytes(), 0).min(max)
    }

    fn record(&self, i: usize) -> &[u8] {
        &self.bytes()[16 + i * self.stride..][..self.stride]
    }

    fn destination(&self, i: usize) -> u64 {
        u64::from_le_bytes(
            self.bytes()[self.destinations + 8 * i..][..8]
                .try_into()
                .unwrap(),
        )
    }
}

impl CommandList<'_> {
    /// Records `d` over the records and count kernels wrote at `io`, then makes the results
    /// visible to traces, builds and kernels. Order the argument writes before it with a
    /// barrier into [`Stages::AS_BUILD`].
    pub fn rtas_execute(&mut self, d: &RtasOpDesc, io: &RtasIo) {
        if d.op == RtasOp::Ptlas {
            self.ptlas(d, io);
        } else {
            let (d, io) = (*d, *io);
            let readback = self.read_records(&d, &io);
            let creates = match d.op {
                RtasOp::BuildTemplates | RtasOp::Move => 0,
                _ if d.flags.contains(RtasFlags::FLAT_ONLY) => 0,
                _ => u64::from(max_args(&d)),
            };
            let reservation = self.queue.shared.reserve_structures(creates);
            self.split(Box::new(move |list: &mut CommandList<'_>| {
                list.cluster_step(&d, &io, &readback, reservation);
            }));
        }
        self.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }

    /// Copies the count, the records and any explicit destinations into memory the CPU
    /// reads.
    fn read_records(&mut self, d: &RtasOpDesc, io: &RtasIo) -> Readback {
        let n = u64::from(max_args(d));
        let stride = u64::from(args_stride(d.op));
        let explicit = d.mode == RtasMode::Explicit;
        let destinations = 16 + n * stride;
        let bytes = destinations + if explicit { 8 * n } else { 0 };
        let memory = malloc::<u8>(
            &self.queue.shared,
            MemoryKind::Readback,
            bytes,
            "rtas records",
        )
        .expect("rtas readback");
        let at = memory.gpu().address().0;
        self.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
        // SAFETY: each range lies in memory the caller or this list owns.
        unsafe {
            let ptr = |a: u64| GpuPtr::<u8>::from_address(GpuAddress(a));
            self.copy(ptr(at), io.count.cast(), 4);
            if n > 0 {
                self.copy(ptr(at + 16), ptr(io.args.0), n * stride);
            }
            if explicit {
                self.copy(ptr(at + destinations), io.dst_addresses.cast(), 8 * n);
            }
        }
        Readback {
            memory,
            stride: stride as usize,
            destinations: destinations as usize,
        }
    }

    /// The CPU half of a cluster op: encodes it from the records the GPU wrote. The
    /// structures it creates are within `reservation`, which it keeps until they exist.
    fn cluster_step(
        &mut self,
        d: &RtasOpDesc,
        io: &RtasIo,
        rb: &Readback,
        reservation: Reservation,
    ) {
        let count = rb.count(max_args(d)) as usize;
        if count == 0 {
            return;
        }
        match d.op {
            RtasOp::BuildClas | RtasOp::BuildTemplates | RtasOp::InstantiateTemplates => {
                self.build_objects(d, io, rb, count);
            }
            RtasOp::BuildClusterBlas => self.build_blas(d, io, rb, count, reservation),
            RtasOp::Move => self.move_objects(d, io, rb, count),
            RtasOp::Ptlas => unreachable!("PTLAS builds take no CPU step"),
        }
    }

    /// Where `sizes` go: packed from `dst_implicit`, or at the explicit destinations; none
    /// when only sizing.
    fn destinations(d: &RtasOpDesc, io: &RtasIo, rb: &Readback, sizes: &[u64]) -> Vec<u64> {
        match d.mode {
            RtasMode::Implicit => {
                let mut at = io.dst_implicit.0;
                sizes
                    .iter()
                    .map(|&s| {
                        let a = at;
                        at = (at + s).next_multiple_of(u64::from(ALIGN));
                        a
                    })
                    .collect()
            }
            RtasMode::Explicit => (0..sizes.len()).map(|i| rb.destination(i)).collect(),
            RtasMode::GetSizes => Vec::new(),
        }
    }

    /// Memory this list keeps until its submit completes.
    fn scratch_memory(&mut self, kind: MemoryKind, bytes: u64, name: &str) -> Allocation<u8> {
        malloc::<u8>(&self.queue.shared, kind, bytes.max(16), name).expect(name)
    }

    /// Writes the results' addresses and sizes where the caller asked for them.
    fn write_results(&mut self, io: &RtasIo, addresses: &[u64], sizes: &[u64]) {
        let mut put = |dst: GpuPtr<u8>, bytes: Vec<u8>| {
            if dst == GpuPtr::NULL || bytes.is_empty() {
                return;
            }
            let mut m = self.scratch_memory(MemoryKind::Upload, bytes.len() as u64, "rtas results");
            m.as_mut_slice().unwrap()[..bytes.len()].copy_from_slice(&bytes);
            self.copy(dst, m.gpu(), bytes.len() as u64);
            self.keep.push(Box::new(m));
        };
        put(
            io.dst_addresses.cast(),
            addresses.iter().flat_map(|a| a.to_le_bytes()).collect(),
        );
        put(
            io.dst_sizes.cast(),
            sizes
                .iter()
                .flat_map(|&s| (s as u32).to_le_bytes())
                .collect(),
        );
    }

    /// Uploads `jobs` and runs `state` with one group per job.
    fn run_jobs<T: bytemuck::Pod>(
        &mut self,
        state: &Obj<ProtocolObject<dyn MTLComputePipelineState>>,
        jobs: &[T],
    ) {
        if jobs.is_empty() {
            return;
        }
        let bytes = bytemuck::cast_slice::<T, u8>(jobs);
        let mut m = self.scratch_memory(MemoryKind::Upload, bytes.len() as u64, "cluster jobs");
        m.as_mut_slice().unwrap()[..bytes.len()].copy_from_slice(bytes);
        let root = m.gpu().address().0;
        self.dispatch_state(
            state,
            &root.to_le_bytes(),
            [jobs.len() as u32, 1, 1],
            [GROUP, 1, 1],
        );
        self.keep.push(Box::new(m));
    }

    /// Writes 64-byte headers to their addresses.
    fn write_headers(&mut self, headers: &[(u64, [u32; 16])]) {
        let k = Kernels::get(&self.queue.shared);
        let bytes: Vec<u8> = headers
            .iter()
            .flat_map(|(_, h)| bytemuck::cast_slice::<u32, u8>(h).to_vec())
            .collect();
        let mut m = self.scratch_memory(MemoryKind::Upload, bytes.len() as u64, "cluster headers");
        m.as_mut_slice().unwrap()[..bytes.len()].copy_from_slice(&bytes);
        let src = m.gpu().address().0;
        let jobs: Vec<CopyJob> = headers
            .iter()
            .enumerate()
            .map(|(i, &(dst, _))| CopyJob {
                src: src + HEADER * i as u64,
                dst,
                words: 16,
                pad: 0,
            })
            .collect();
        self.keep.push(Box::new(m));
        let state = k.copy_words.clone();
        self.run_jobs(&state, &jobs);
    }

    /// Encodes `builds` with scratch of their own, in batches.
    fn encode_builds(&mut self, mut builds: Vec<Build>) {
        if builds.is_empty() {
            return;
        }
        let rest = builds.split_off(builds.len().min(BUILDS_PER_ENCODER));
        let offsets: Vec<u64> = builds
            .iter()
            .scan(0u64, |at, (_, _, s)| {
                let o = *at;
                *at += s.next_multiple_of(256).max(256);
                Some(o)
            })
            .collect();
        let total = offsets.last().unwrap() + builds.last().unwrap().2.max(256);
        let scratch = self.scratch_memory(MemoryKind::DeviceLocal, total, "cluster scratch");
        let base = scratch.gpu().address().0;
        self.barrier(Stages::ALL, Stages::AS_BUILD, Hazards::empty());
        let e: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>> = self.compute();
        for ((raw, desc, s), o) in builds.iter().zip(offsets) {
            // SAFETY: the structure, its inputs and its scratch range are live until the
            // submit completes.
            unsafe {
                e.buildAccelerationStructure_descriptor_scratchBuffer(
                    raw,
                    desc,
                    range(base + o, (*s).max(256)),
                );
            }
            self.keep.push(Box::new(raw.clone()));
        }
        self.keep.push(Box::new(scratch));
        if !rest.is_empty() {
            let rest = RemainingBuilds(rest);
            self.split(Box::new(move |list: &mut CommandList<'_>| {
                let rest = rest;
                list.encode_builds(rest.0);
            }));
        }
    }

    /// CLAS builds, template builds and instantiations.
    fn build_objects(&mut self, d: &RtasOpDesc, io: &RtasIo, rb: &Readback, count: usize) {
        let shared = Arc::clone(&self.queue.shared);
        let clas = d.op != RtasOp::BuildTemplates;
        let structures = clas && !d.flags.contains(RtasFlags::FLAT_ONLY);
        // Triangles, vertices and opacity of each record.
        let shapes: Vec<(u32, u32, bool)> = (0..count)
            .map(|i| {
                let r = rb.record(i);
                if d.op == RtasOp::InstantiateTemplates {
                    let template = word64(r, 2);
                    let objects = lock(&shared.objects);
                    let t = objects.by_start.get(&template).unwrap_or_else(|| {
                        panic!("instantiating {template:#x}, which holds no template")
                    });
                    (t.tris, t.verts, t.opaque)
                } else {
                    let bits = word(r, 2);
                    let base = word(r, 3);
                    let flags = word64(r, 10);
                    (
                        bits & 0x1ff,
                        (bits >> 9) & 0x1ff,
                        base >> 31 != 0 && flags == 0,
                    )
                }
            })
            .collect();
        let sizes: Vec<u64> = shapes
            .iter()
            .map(|&(t, v, _)| object_bytes(clas, t, v))
            .collect();
        if d.mode == RtasMode::GetSizes {
            self.write_results(io, &[], &sizes);
            return;
        }
        let addresses = Self::destinations(d, io, rb, &sizes);
        let mut builds = Vec::new();
        let mut jobs = Vec::with_capacity(count);
        for (i, (&(tris, verts, opaque), &at)) in shapes.iter().zip(&addresses).enumerate() {
            let mut id = 0;
            let mut raw = None;
            if structures {
                let vertices = at + HEADER;
                let indices = vertices + 12 * u64::from(verts);
                let data = indices + 12 * u64::from(tris);
                let desc = primitive(
                    &[triangles(
                        vertices,
                        u64::from(verts),
                        indices,
                        tris,
                        data,
                        opaque,
                    )],
                    false,
                );
                let (s, scratch) = shared.structure(&desc);
                id = id_bits(s.gpuResourceID());
                builds.push((s.clone(), desc, scratch));
                raw = Some(s);
            }
            shared.place(
                at,
                Object {
                    end: at + sizes[i],
                    kind: if clas { Kind::Clas } else { Kind::Template },
                    tris,
                    verts,
                    opaque,
                    raw,
                },
            );
            jobs.push(PackJob {
                dst: at,
                id,
                record: io.args.0 + i as u64 * u64::from(args_stride(d.op)),
                kind: match d.op {
                    RtasOp::BuildClas => 1,
                    RtasOp::BuildTemplates => 2,
                    _ => 3,
                },
                pad: 0,
            });
        }
        let pack = Kernels::get(&shared).pack.clone();
        self.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.run_jobs(&pack, &jobs);
        self.encode_builds(builds);
        self.write_results(io, &addresses, &sizes);
    }

    /// Cluster BLAS builds.
    fn build_blas(
        &mut self,
        d: &RtasOpDesc,
        io: &RtasIo,
        rb: &Readback,
        count: usize,
        reservation: Reservation,
    ) {
        let blas: Vec<(u64, u32)> = (0..count)
            .map(|i| {
                let r = rb.record(i);
                (word64(r, 2), word(r, 0))
            })
            .collect();
        let sizes = vec![HEADER; count];
        if d.mode == RtasMode::GetSizes {
            self.write_results(io, &[], &sizes);
            return;
        }
        let addresses = Self::destinations(d, io, rb, &sizes);
        if d.flags.contains(RtasFlags::TRANSIENT) {
            self.build_instanced(&blas, &addresses);
            self.write_results(io, &addresses, &sizes);
        } else {
            // A flat BLAS needs its CLAS' triangle counts: the CLAS addresses come back in
            // a second step.
            let total: u64 = blas.iter().map(|&(_, n)| 8 * u64::from(n)).sum();
            let memory = self.scratch_memory(MemoryKind::Readback, total, "cluster BLAS CLAS");
            let at = memory.gpu().address().0;
            let mut offset = 0;
            self.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
            for &(list, n) in &blas {
                if n > 0 {
                    // SAFETY: the caller's CLAS address arrays and this list's readback.
                    unsafe {
                        self.copy(
                            GpuPtr::from_address(GpuAddress(at + offset)),
                            GpuPtr::from_address(GpuAddress(list)),
                            8 * u64::from(n),
                        );
                    }
                }
                offset += 8 * u64::from(n);
            }
            let io = *io;
            self.split(Box::new(move |list: &mut CommandList<'_>| {
                let _reservation = reservation;
                let bytes = memory.as_slice().expect("readback is mapped");
                let mut at = 0;
                let clas: Vec<Vec<u64>> = blas
                    .iter()
                    .map(|&(_, n)| {
                        let v = (0..n as usize)
                            .map(|i| {
                                u64::from_le_bytes(bytes[at + 8 * i..][..8].try_into().unwrap())
                            })
                            .collect();
                        at += 8 * n as usize;
                        v
                    })
                    .collect();
                list.build_flat(&clas, &addresses);
                list.write_results(&io, &addresses, &sizes);
            }));
        }
    }

    /// Instance structures over the CLAS' BLAS at each `(list, count)`.
    fn build_instanced(&mut self, blas: &[(u64, u32)], addresses: &[u64]) {
        let shared = Arc::clone(&self.queue.shared);
        let total: u64 = blas.iter().map(|&(_, n)| u64::from(n.max(1))).sum();
        let descriptors = self.scratch_memory(
            MemoryKind::DeviceLocal,
            INSTANCE_BYTES * total,
            "cluster BLAS instances",
        );
        let mut out = descriptors.gpu().address().0;
        let mut jobs = Vec::new();
        let mut builds = Vec::new();
        let mut headers = Vec::new();
        for (&(list, n), &at) in blas.iter().zip(addresses) {
            let desc = instances(out, n);
            let (s, scratch) = shared.structure(&desc);
            let id = id_bits(s.gpuResourceID());
            jobs.push(BlasJob {
                clas: list,
                out,
                count: n,
                pad: 0,
            });
            builds.push((s.clone(), desc, scratch));
            headers.push((at, header(Kind::BlasInstanced, HEADER, id, &[n])));
            shared.place(
                at,
                Object {
                    end: at + HEADER,
                    kind: Kind::BlasInstanced,
                    tris: 0,
                    verts: 0,
                    opaque: false,
                    raw: Some(s),
                },
            );
            out += INSTANCE_BYTES * u64::from(n.max(1));
        }
        let instances_state = Kernels::get(&shared).blas_instances.clone();
        self.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.run_jobs(&instances_state, &jobs);
        self.write_headers(&headers);
        self.encode_builds(builds);
        self.keep.push(Box::new(descriptors));
    }

    /// Flat BLAS over the triangles of each list of CLAS addresses.
    fn build_flat(&mut self, clas: &[Vec<u64>], addresses: &[u64]) {
        let shared = Arc::clone(&self.queue.shared);
        // Each BLAS's opaque and other triangles, gathered unindexed.
        let shapes: Vec<Vec<(u64, u32, bool)>> = {
            let objects = lock(&shared.objects);
            clas.iter()
                .map(|list| {
                    list.iter()
                        .map(|a| {
                            let o = objects.by_start.get(a).unwrap_or_else(|| {
                                panic!("a cluster BLAS over {a:#x}, which holds no CLAS")
                            });
                            (*a, o.tris, o.opaque)
                        })
                        .collect()
                })
                .collect()
        };
        let total: u64 = shapes.iter().flatten().map(|&(_, t, _)| u64::from(t)).sum();
        let vertices =
            self.scratch_memory(MemoryKind::DeviceLocal, 36 * total, "flat BLAS vertices");
        let data = self.scratch_memory(
            MemoryKind::DeviceLocal,
            DATA_BYTES * total,
            "flat BLAS data",
        );
        let (mut v_at, mut d_at) = (vertices.gpu().address().0, data.gpu().address().0);
        let mut jobs = Vec::new();
        let mut builds = Vec::new();
        let mut headers = Vec::new();
        for (list, &at) in shapes.iter().zip(addresses) {
            let mut geometries = Vec::new();
            for opaque in [true, false] {
                let (v0, d0) = (v_at, d_at);
                let mut tris = 0;
                for &(clas, t, _) in list.iter().filter(|&&(_, _, o)| o == opaque) {
                    jobs.push(GatherJob {
                        clas,
                        vertices: v_at,
                        data: d_at,
                    });
                    v_at += 36 * u64::from(t);
                    d_at += DATA_BYTES * u64::from(t);
                    tris += t;
                }
                if tris > 0 {
                    geometries.push(triangles(v0, 3 * u64::from(tris), 0, tris, d0, opaque));
                }
            }
            let (s, kind) = if geometries.is_empty() {
                // No triangles: an instance structure over none.
                (instances(v_at, 0), Kind::BlasInstanced)
            } else {
                (primitive(&geometries, false), Kind::BlasFlat)
            };
            let (raw, scratch) = shared.structure(&s);
            let id = id_bits(raw.gpuResourceID());
            builds.push((raw.clone(), s, scratch));
            headers.push((at, header(kind, HEADER, id, &[])));
            shared.place(
                at,
                Object {
                    end: at + HEADER,
                    kind,
                    tris: 0,
                    verts: 0,
                    opaque: false,
                    raw: Some(raw),
                },
            );
        }
        let gather = Kernels::get(&shared).blas_gather.clone();
        self.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.run_jobs(&gather, &jobs);
        self.write_headers(&headers);
        self.encode_builds(builds);
        self.keep.push(Box::new(vertices));
        self.keep.push(Box::new(data));
    }

    /// Moves of CLAS, templates and cluster BLAS: copies through scratch, so destinations
    /// may overlap sources.
    fn move_objects(&mut self, d: &RtasOpDesc, io: &RtasIo, rb: &Readback, count: usize) {
        let shared = Arc::clone(&self.queue.shared);
        let sources: Vec<u64> = (0..count).map(|i| word64(rb.record(i), 0)).collect();
        let sizes: Vec<u64> = {
            let objects = lock(&shared.objects);
            sources
                .iter()
                .map(|s| {
                    let o = objects
                        .by_start
                        .get(s)
                        .unwrap_or_else(|| panic!("moving {s:#x}, which holds no object"));
                    o.end - s
                })
                .collect()
        };
        if d.mode == RtasMode::GetSizes {
            self.write_results(io, &[], &sizes);
            return;
        }
        let addresses = Self::destinations(d, io, rb, &sizes);
        let total: u64 = sizes.iter().sum();
        let temp = self.scratch_memory(MemoryKind::DeviceLocal, total, "cluster moves");
        let mut at = temp.gpu().address().0;
        let (mut there, mut back) = (Vec::new(), Vec::new());
        for ((&src, &dst), &size) in sources.iter().zip(&addresses).zip(&sizes) {
            let words = (size / 4) as u32;
            there.push(CopyJob {
                src,
                dst: at,
                words,
                pad: 0,
            });
            back.push(CopyJob {
                src: at,
                dst,
                words,
                pad: 0,
            });
            at += size;
        }
        let copy = Kernels::get(&shared).copy_words.clone();
        self.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.run_jobs(&copy, &there);
        self.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        self.run_jobs(&copy, &back);
        self.keep.push(Box::new(temp));
        // The objects keep their structures at their new addresses.
        let moved: Vec<Object> = {
            let mut objects = lock(&shared.objects);
            sources
                .iter()
                .map(|s| objects.by_start.remove(s).expect("looked up above"))
                .collect()
        };
        for ((o, &dst), &size) in moved.into_iter().zip(&addresses).zip(&sizes) {
            let o = Object {
                end: dst + size,
                ..o
            };
            // The structure is in the residency set already.
            let gone = lock(&shared.objects).insert(dst, o);
            if !gone.is_empty() {
                let s = Arc::clone(&shared);
                shared.defer(Box::new(move || s.release_structures(gone)));
            }
        }
        self.write_results(io, &addresses, &sizes);
    }

    /// A PTLAS build: the commands apply to the stored state on the GPU, then the TLAS is
    /// rebuilt from it.
    fn ptlas(&mut self, d: &RtasOpDesc, io: &RtasIo) {
        let RtasLimits::Ptlas(l) = d.limits else {
            panic!("{:?} takes other limits than {:?}", d.op, d.limits);
        };
        let shared = Arc::clone(&self.queue.shared);
        let state = io.dst_implicit.0;
        assert!(
            state.is_multiple_of(PTLAS_ALIGN),
            "PTLAS {state:#x} is not aligned to {PTLAS_ALIGN}"
        );
        let size = ptlas_bytes(l.instances, l.partitions);
        let desc = ptlas_descriptor(state, l.instances, l.partitions);
        // The TLAS stays with its state while the capacity does not change.
        let existing = {
            let objects = lock(&shared.objects);
            objects
                .by_start
                .get(&state)
                .filter(|o| o.kind == Kind::Ptlas && o.end == state + size)
                .and_then(|o| o.raw.clone())
        };
        let raw = match existing {
            Some(raw) => raw,
            None => {
                let (raw, _) = shared.structure(&desc);
                shared.place(
                    state,
                    Object {
                        end: state + size,
                        kind: Kind::Ptlas,
                        tris: 0,
                        verts: 0,
                        opaque: false,
                        raw: Some(raw.clone()),
                    },
                );
                raw
            }
        };
        let id = id_bits(raw.gpuResourceID());
        let k = Kernels::get(&shared);
        let (clear, apply, descriptors) = (
            k.ptlas_clear.clone(),
            k.ptlas_apply.clone(),
            k.ptlas_descriptors.clone(),
        );
        let args = |op: u32| PtlasKernelArgs {
            state,
            commands: io.args.0,
            count: io.count.address().0,
            instances: l.instances,
            partitions: l.partitions,
            op,
            pad: 0,
        };
        let threads = l.instances.max(l.partitions + 1);
        let groups = [threads.div_ceil(GROUP), 1, 1];
        if cfg!(debug_assertions) {
            let check = PtlasCheck::get(&shared);
            let args = PtlasCheckArgs {
                commands: io.args.0,
                count: io.count.address().0,
                failures: check.failures.gpuAddress(),
                max_ops: l.max_ops,
                pad: 0,
            };
            let (check, writes) = (k.ptlas_check.clone(), k.ptlas_check_writes.clone());
            let args = bytemuck::bytes_of(&args);
            self.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
            // Its pipeline takes groups of GROUP threads; thread 0 checks.
            self.dispatch_state(&check, args, [1, 1, 1], [GROUP, 1, 1]);
            self.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            self.dispatch_state(&writes, args, groups, [GROUP, 1, 1]);
        }
        self.barrier(
            Stages::ALL,
            Stages::COMPUTE | Stages::TRANSFER,
            Hazards::empty(),
        );
        if io.src.0 == 0 {
            self.dispatch_state(&clear, bytemuck::bytes_of(&args(0)), groups, [GROUP, 1, 1]);
        } else if io.src.0 != state {
            // SAFETY: both states are `size` bytes of the caller's memory.
            unsafe {
                self.copy(
                    GpuPtr::from_address(GpuAddress(state)),
                    GpuPtr::from_address(io.src),
                    size,
                );
            }
        }
        self.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        let h = header(Kind::Ptlas, size, id, &[l.instances, l.partitions]);
        self.write_headers(&[(state, h)]);
        for op in 0..3 {
            self.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            self.dispatch_state(&apply, bytemuck::bytes_of(&args(op)), groups, [GROUP, 1, 1]);
        }
        self.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        self.dispatch_state(
            &descriptors,
            bytemuck::bytes_of(&args(0)),
            groups,
            [GROUP, 1, 1],
        );
        self.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
        let e = self.compute();
        // SAFETY: the TLAS, its descriptors and the caller's scratch are live until the
        // submit completes.
        unsafe {
            e.buildAccelerationStructure_descriptor_scratchBuffer(
                &raw,
                &desc,
                range(io.scratch.0, u64::MAX),
            );
        }
        self.keep.push(Box::new(raw));
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackJob {
    dst: u64,
    id: u64,
    record: u64,
    kind: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlasJob {
    clas: u64,
    out: u64,
    count: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherJob {
    clas: u64,
    vertices: u64,
    data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CopyJob {
    src: u64,
    dst: u64,
    words: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PtlasCheckArgs {
    commands: u64,
    count: u64,
    failures: u64,
    max_ops: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PtlasKernelArgs {
    state: u64,
    commands: u64,
    count: u64,
    instances: u32,
    partitions: u32,
    op: u32,
    pad: u32,
}
