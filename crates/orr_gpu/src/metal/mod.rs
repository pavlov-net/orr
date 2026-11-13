//! The Metal 4 backend: orr_gpu's interface on Apple GPUs, with the public items of the Vulkan
//! backend under the same names, so the crates above orr_gpu build unchanged.
//!
//! Kernels reach everything as `abi.slang` lays it out for Metal: argument-table buffer 0 holds
//! the root's address, buffer 1 the address of the texture view pool's base resource ID and
//! buffer 2 the address of the sampler resource IDs. Memory is reached by address
//! (`MTLBuffer::gpuAddress`): each allocation is a buffer placed in a heap that orr_memory
//! sub-allocates, every heap sits in one residency set, and each queue's timeline is an
//! `MTLSharedEvent`.
//!
//! What the backend does not do yet still exists: calls that return a `Result` fail with
//! [`Error::NotImplemented`](crate::Error::NotImplemented), the others panic saying so, and
//! handles only those calls create cannot be made.
//!
//! Types alike on both backends are the crate root's; the ones here hold Metal objects and
//! have the Vulkan backend's public fields and methods.

mod accel;
mod cache;
mod cluster;
mod command;
mod device;
mod heap;
mod memory;
mod mesh;
mod pipeline;
mod queue;
mod sparse;
mod swapchain;
mod texture;
mod unsupported;

use std::collections::BTreeMap;
use std::sync::Mutex;

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLSharedEvent};

pub use accel::{ClassicAs, Micromap};
pub use command::CommandList;
pub use device::{Device, DeviceHooks};
pub use memory::{Allocation, DeviceArenas, HostArenas};
pub use mesh::MeshPipeline;
pub use pipeline::{ComputePipeline, ShaderStage};
pub use queue::{HostTimeline, Queue};
pub use sparse::{SparseAllocation, SparseBinds};
pub use swapchain::{LowLatency, Swapchain, SwapchainFrame, WindowExtent};
pub use texture::Texture;
pub use unsupported::{
    Diagnostics, ExecutionSet, ImportedMemory, RayDispatchPipeline, RayPipeline,
};

/// The most queues a device has, one per [`QueueKind`].
const MAX_TIMELINES: usize = 5;

/// A Metal object shared between threads. Metal's devices, resources (buffers, heaps,
/// textures), events, libraries and pipeline states are safe to use from any thread, as are
/// Core Animation's Metal layers and drawables; objc2 marks only some of their protocols
/// `Send + Sync`. Only those kinds go in here.
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

/// One queue's timeline: values up to `submitted` have been submitted, those up to `issued`
/// have had their signals handed to Metal, and the event reaches each as its work completes.
pub(super) struct Timeline {
    pub(super) event: Obj<ProtocolObject<dyn MTLSharedEvent>>,
    pub(super) submitted: std::sync::atomic::AtomicU64,
    issued: Mutex<u64>,
    issued_changed: std::sync::Condvar,
}

impl Timeline {
    pub(super) fn new(event: Obj<ProtocolObject<dyn MTLSharedEvent>>) -> Self {
        Self {
            event,
            submitted: std::sync::atomic::AtomicU64::new(0),
            issued: Mutex::new(0),
            issued_changed: std::sync::Condvar::new(),
        }
    }

    pub(super) fn completed(&self) -> u64 {
        self.event.signaledValue()
    }

    pub(super) fn issue(&self, value: u64) {
        let mut issued = lock(&self.issued);
        *issued = value.max(*issued);
        self.issued_changed.notify_all();
    }

    pub(super) fn is_issued(&self, value: u64) -> bool {
        *lock(&self.issued) >= value
    }

    /// Blocks until `value`'s signal has been handed to Metal. Every point waited on comes
    /// from an earlier submit, so these waits cannot form a cycle.
    pub(super) fn wait_issued(&self, value: u64) {
        let issued = lock(&self.issued);
        drop(
            self.issued_changed
                .wait_while(issued, |issued| *issued < value)
                .unwrap(),
        );
    }
}

/// Work to run once the GPU has finished with something.
type Release = Box<dyn FnOnce() + Send>;

/// Releases waiting on timeline values and command lists: each entry runs once every timeline
/// has reached the value it had been submitted to when the entry was queued, and every list
/// open then has resolved (see [`crate::queue::OpenLists`]).
#[derive(Default)]
pub(super) struct Garbage {
    /// Each release's values, the first list sequence number it does not wait for and the
    /// release.
    pending: Vec<([u64; MAX_TIMELINES], u64, Release)>,
}

impl Garbage {
    pub(super) fn push(&mut self, after: [u64; MAX_TIMELINES], lists: u64, release: Release) {
        self.pending.push((after, lists, release));
    }

    /// Takes every release whose values `completed` has reached and whose lists have all
    /// resolved, `first_unresolved` being the smallest sequence number of a list that has not.
    /// The caller runs them after unlocking the list, since a release may defer releases of
    /// its own.
    #[must_use]
    pub(super) fn take_done(
        &mut self,
        completed: &[u64; MAX_TIMELINES],
        first_unresolved: u64,
    ) -> Vec<Release> {
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|(after, lists, _)| {
                *lists <= first_unresolved && after.iter().zip(completed).all(|(a, c)| c >= a)
            });
        self.pending = keep;
        done.into_iter().map(|(_, _, release)| release).collect()
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
