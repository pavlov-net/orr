//! The Metal 4 backend: orr_gpu's interface on Apple GPUs, with the public items of the Vulkan
//! backend under the same names, so the crates above orr_gpu build unchanged.
//!
//! Kernels reach everything as `abi.slang` lays it out for Metal: argument-table buffer 0 holds
//! the root's address, buffer 1 the address of the texture view pool's base resource ID and
//! buffer 2 the address of the sampler resource IDs. Memory is reached by address
//! (`MTLBuffer::gpuAddress`), every allocation sits in one residency set, and each queue's
//! timeline is an `MTLSharedEvent`.
//!
//! What the backend does not do yet still exists: calls that return a `Result` fail with
//! [`Error::NotImplemented`](crate::Error::NotImplemented), the others panic saying so, and
//! handles only those calls create cannot be made.
//!
//! The descriptor and report types repeat the Vulkan backend's definitions field for field.

mod accel;
mod command;
mod device;
mod heap;
mod memory;
mod mesh;
mod pipeline;
mod queue;
mod sparse;
mod texture;
mod unsupported;

use std::collections::BTreeMap;
use std::sync::Mutex;

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLSharedEvent};

pub use accel::{
    ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicInstance, ClassicKind,
    InstanceFlags, LssEndCaps, LssIndexing, Micromap, MicromapDesc, MicromapTriangle,
    MicromapUsage, OmmFormat,
};
pub use command::{CommandList, DecompressRegion, DispatchArgs, MemoryCopy, MeshArgs, TextureCopy};
pub use device::{Device, DeviceDesc, DeviceHooks};
pub use memory::{
    AllocError, Allocation, DeviceArenas, HeapReport, HostArenas, MemoryKind, MemoryPriority,
    MemoryReport, MemoryUsage,
};
pub use mesh::{ColorAttachment, DepthAttachment, MeshPipeline, MeshPipelineDesc, Rendering};
pub use pipeline::{ComputePipeline, ShaderStage};
pub use queue::{PassResult, PipelineStats, Queue, QueueKind, QueueRequest};
pub use sparse::{PageBind, SparseAllocation, SparseBinds, TileBind};
pub use texture::{Texture, TextureMemory};
pub use unsupported::{
    COUNTER_METRICS, Checkpoint, CounterRange, CounterReport, DeviceFault, Diagnostics,
    DiagnosticsReport, ExecutionSet, FaultAccess, FaultAddress, Generated, LowLatency,
    MATRIX_ALIGN, MatrixConversion, MatrixFormat, MatrixLayout, PipelineCacheReport, QueueProgress,
    RayDispatchPipeline, Sequence, SetKind, Swapchain, SwapchainFrame,
};

/// The most queues a device has, one per [`QueueKind`].
const MAX_TIMELINES: usize = 5;

/// A Metal object shared between threads. Metal's devices, resources (buffers, heaps,
/// textures), events, libraries and pipeline states are safe to use from any thread; objc2
/// marks only some of their protocols `Send + Sync`. Only those kinds go in here.
pub(super) struct Obj<T: Message>(pub(super) Retained<T>);

// SAFETY: only thread-safe Metal objects are wrapped (see above); the wrapper hands out
// shared references only.
unsafe impl<T: Message> Send for Obj<T> {}
// SAFETY: as above.
unsafe impl<T: Message> Sync for Obj<T> {}

impl<T: Message> std::ops::Deref for Obj<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Message> Clone for Obj<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

type Buffer = ProtocolObject<dyn MTLBuffer>;

/// One queue's timeline: values up to `submitted` have been submitted, and the event reaches
/// each as its work completes.
pub(super) struct Timeline {
    pub(super) event: Obj<ProtocolObject<dyn MTLSharedEvent>>,
    pub(super) submitted: std::sync::atomic::AtomicU64,
}

impl Timeline {
    pub(super) fn completed(&self) -> u64 {
        self.event.signaledValue()
    }
}

/// Work to run once the GPU has finished with something.
type Release = Box<dyn FnOnce() + Send>;

/// Releases waiting on timeline values: each entry runs once every timeline has reached the
/// value it had been submitted to when the entry was queued.
#[derive(Default)]
pub(super) struct Garbage {
    pending: Vec<([u64; MAX_TIMELINES], Release)>,
}

impl Garbage {
    pub(super) fn push(&mut self, after: [u64; MAX_TIMELINES], release: Release) {
        self.pending.push((after, release));
    }

    /// Takes every release whose values `completed` has reached. The caller runs them after
    /// unlocking the list, since a release may defer releases of its own.
    #[must_use]
    pub(super) fn take_done(&mut self, completed: &[u64; MAX_TIMELINES]) -> Vec<Release> {
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|(after, _)| after.iter().zip(completed).all(|(a, c)| c >= a));
        self.pending = keep;
        done.into_iter().map(|(_, release)| release).collect()
    }
}

/// The live buffers by GPU address, for the Metal calls that take a buffer and an offset.
#[derive(Default)]
pub(super) struct Buffers {
    /// Start address to (end address, buffer).
    by_start: BTreeMap<u64, (u64, Obj<Buffer>)>,
}

impl Buffers {
    pub(super) fn insert(&mut self, start: u64, size: u64, buffer: Obj<Buffer>) {
        self.by_start.insert(start, (start + size, buffer));
    }

    pub(super) fn remove(&mut self, start: u64) {
        self.by_start.remove(&start);
    }

    /// The buffer holding `bytes` bytes from `address`, and the offset of `address` in it.
    pub(super) fn find(&self, address: u64, bytes: u64) -> Option<(Obj<Buffer>, u64)> {
        let (start, (end, buffer)) = self.by_start.range(..=address).next_back()?;
        (address + bytes <= *end).then(|| (buffer.clone(), address - start))
    }
}

/// Locks `m`, carrying on past a panic on another thread: every guarded structure stays
/// consistent between statements.
pub(super) fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Panics for a call the Metal backend does not implement yet.
#[track_caller]
pub(super) fn not_implemented(what: &str) -> ! {
    panic!("orr_gpu's Metal backend does not implement {what} yet")
}
