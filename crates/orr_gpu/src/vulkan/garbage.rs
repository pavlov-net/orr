//! Deferred destruction: objects freed while the GPU may still use them wait for timeline
//! points, then `Device::collect_garbage` destroys them.

use ash::vk;

use super::device::Shared;
use crate::MemoryKind;
use crate::memory::Charge;

/// Timelines a device can have: one per queue kind.
pub(super) const MAX_TIMELINES: usize = 5;

/// Per timeline, the value that must complete before an item is destroyed (0: none).
pub(super) type After = [u64; MAX_TIMELINES];

pub(super) enum Item {
    Buffer(RawBuffer),
    Image {
        image: vk::Image,
        memory: Option<(orr_memory::Allocation, Charge)>,
        views: [vk::ImageView; 2],
    },
    Pipeline(vk::Pipeline),
    Accel(vk::AccelerationStructureKHR),
    /// A sparse buffer; the memory bound to it belongs to others.
    SparseBuffer {
        buffer: vk::Buffer,
        address: u64,
    },
    /// An execution set, its command layouts and its pipelines, destroyed in that order.
    ExecutionSet {
        set: vk::IndirectExecutionSetEXT,
        layouts: Vec<vk::IndirectCommandsLayoutEXT>,
        pipelines: Vec<vk::Pipeline>,
    },
    ImageSlot(u32),
    RaygenRecord(u32),
    /// A library's objects, released by its closure.
    Release(Box<dyn FnOnce() + Send>),
    /// An import of host memory.
    Imported(super::host_memory::Imported),
}

/// A buffer bound to memory from orr_gpu's pools, as `gpu_malloc` creates it.
pub(super) struct RawBuffer {
    pub(super) buffer: vk::Buffer,
    pub(super) allocation: orr_memory::Allocation,
    pub(super) kind: MemoryKind,
    pub(super) size: u64,
    pub(super) address: u64,
    pub(super) charge: Charge,
}

impl RawBuffer {
    /// The memory and memory offset of byte `offset` of the buffer.
    pub(super) fn memory_at(&self, offset: u64) -> (vk::DeviceMemory, u64) {
        (self.allocation.memory, self.allocation.offset + offset)
    }
}

/// Due allocations with device memory of their own that one collection frees, when as many
/// are due: a `vkFreeMemory` costs the driver real time, and an eviction can empty hundreds of
/// pool blocks at once.
const OWN_FREES: usize = 2;
/// One collection frees one in this many of the due allocations with device memory of their
/// own when that is more than [`OWN_FREES`], so what waits shrinks by that share each
/// collection however much was freed at once, and stays within this many collections' worth of
/// what is freed steadily.
const OWN_SHARE: usize = 32;

/// The device memory of its own an item gives back to the system: its heap and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Own {
    pub(super) heap: usize,
    pub(super) bytes: u64,
}

impl Item {
    /// The memory type and bytes of the `VkDeviceMemory` of its own it frees.
    fn own_memory(&self) -> Option<(u32, u64)> {
        let a = match self {
            Item::Buffer(b) => &b.allocation,
            Item::Image { memory, .. } => &memory.as_ref()?.0,
            _ => return None,
        };
        a.dedicated.then_some((a.memory_type, a.size))
    }
}

/// Items waiting for timeline points, in the order they were deferred.
pub(super) struct Waiting<T> {
    entries: Vec<(After, Option<Own>, T)>,
    /// Per heap, the bytes of the entries' device memory of their own.
    own_bytes: [u64; vk::MAX_MEMORY_HEAPS],
}

impl<T> Default for Waiting<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            own_bytes: [0; vk::MAX_MEMORY_HEAPS],
        }
    }
}

pub(super) type Garbage = Waiting<Item>;

impl<T> Waiting<T> {
    fn push(&mut self, after: After, own: Option<Own>, item: T) {
        if let Some(o) = own {
            self.own_bytes[o.heap] += o.bytes;
        }
        self.entries.push((after, own, item));
    }

    /// Removes and returns the items whose points have all completed, oldest first. Of those
    /// with device memory of their own it takes the oldest [`OWN_FREES`], or one in
    /// [`OWN_SHARE`] of them when that is more; the rest wait for the next collections.
    fn take_due(&mut self, completed: &After) -> Vec<T> {
        let ready = |after: &After| after.iter().zip(completed).all(|(a, c)| a <= c);
        let due_own = self
            .entries
            .iter()
            .filter(|(after, own, _)| own.is_some() && ready(after))
            .count();
        let mut quota = due_own.div_ceil(OWN_SHARE).max(OWN_FREES);
        let own_bytes = &mut self.own_bytes;
        self.entries
            .extract_if(.., |(after, own, _)| {
                if !ready(after) {
                    return false;
                }
                let Some(o) = own else { return true };
                if quota == 0 {
                    return false;
                }
                quota -= 1;
                own_bytes[o.heap] -= o.bytes;
                true
            })
            .map(|(_, _, item)| item)
            .collect()
    }

    /// Bytes of device memory of their own on heap `heap` that the waiting items give back.
    pub(super) fn own_bytes(&self, heap: usize) -> u64 {
        self.own_bytes[heap]
    }

    pub(super) fn take_all(&mut self) -> Vec<T> {
        self.own_bytes = [0; vk::MAX_MEMORY_HEAPS];
        self.entries.drain(..).map(|(_, _, item)| item).collect()
    }
}

impl Shared {
    /// Destroys `item` once every timeline reaches its value in `after`.
    pub(super) fn defer(&self, after: After, item: Item) {
        let types = &self.memory.properties.memory_types;
        let own = item.own_memory().map(|(ty, bytes)| Own {
            heap: types[ty as usize].heap_index as usize,
            bytes,
        });
        self.garbage.lock().unwrap().push(after, own, item);
    }

    /// Destroys `item` once everything submitted so far has completed.
    pub(super) fn defer_after_submitted(&self, item: Item) {
        self.defer(self.submitted(), item);
    }

    pub(super) fn collect(&self) {
        self.pools.next_frame();
        let completed = self.completed();
        let due = self.garbage.lock().unwrap().take_due(&completed);
        for item in due {
            self.destroy(item);
        }
    }

    pub(super) fn destroy(&self, item: Item) {
        let d = &self.device;
        // SAFETY: every item is owned by the garbage list alone and no submitted work
        // references it any more.
        unsafe {
            match item {
                Item::Buffer(b) => {
                    self.unregister_buffer(&b);
                    self.pools.allocator.destroy_buffer(b.buffer, b.allocation);
                }
                Item::Image {
                    image,
                    memory,
                    views,
                } => {
                    for v in views {
                        d.destroy_image_view(v, None);
                    }
                    match memory {
                        Some((m, charge)) => {
                            self.ledger.release(charge);
                            self.pools.allocator.destroy_image(image, m);
                        }
                        None => d.destroy_image(image, None),
                    }
                }
                Item::Pipeline(p) => d.destroy_pipeline(p, None),
                Item::Accel(a) => self.accel_fn.destroy_acceleration_structure(a, None),
                Item::SparseBuffer { buffer, address } => {
                    self.unregister_range(address);
                    d.destroy_buffer(buffer, None);
                }
                Item::ExecutionSet {
                    set,
                    layouts,
                    pipelines,
                } => {
                    let f = self
                        .generated_fn
                        .as_ref()
                        .expect("execution sets need Caps::generated_commands");
                    if set != vk::IndirectExecutionSetEXT::null() {
                        f.destroy_indirect_execution_set(set, None);
                    }
                    for l in layouts {
                        f.destroy_indirect_commands_layout(l, None);
                    }
                    for p in pipelines {
                        d.destroy_pipeline(p, None);
                    }
                }
                Item::ImageSlot(slot) => self.heaps().release_image_slot(slot),
                Item::RaygenRecord(slot) => {
                    if let Some(r) = self.records.get() {
                        r.release(slot);
                    }
                }
                Item::Release(release) => release(),
                Item::Imported(i) => self.destroy_imported(i),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DONE: After = [1; MAX_TIMELINES];

    fn own(bytes: u64) -> Option<Own> {
        Some(Own { heap: 1, bytes })
    }

    /// A burst of frees of own memory and a steady run of them after it leave oldest first,
    /// past entries that are not due and beside every due entry without memory of its own, and
    /// what waits falls to a few collections' worth of the steady frees.
    #[test]
    fn own_memory_frees_oldest_first_at_a_rate_that_follows_what_waits() {
        const BURST: u32 = 400;
        const STEADY: u32 = 3;
        let mut w = Waiting::<u32>::default();
        // Not due until the end, ahead of everything else.
        w.push([2; MAX_TIMELINES], own(1), u32::MAX);
        for i in 0..BURST {
            w.push([0; MAX_TIMELINES], own(10), i);
        }
        let mut next = BURST;
        let mut freed = Vec::new();
        let mut longest = 0;
        for collection in 0..200u32 {
            for _ in 0..STEADY {
                w.push([1; MAX_TIMELINES], own(10), next);
                next += 1;
            }
            w.push([0; MAX_TIMELINES], None, u32::MAX - 1);
            let due = w.take_due(&DONE);
            assert_eq!(due.last(), Some(&(u32::MAX - 1)), "an entry without memory");
            let own_due = &due[..due.len() - 1];

            for &i in own_due {
                // Entry `i` was deferred at the collection its number gives.
                let deferred = i.saturating_sub(BURST) / STEADY;
                longest = longest.max(collection - deferred);
            }
            freed.extend_from_slice(own_due);
            assert_eq!(
                w.own_bytes(1),
                1 + 10 * (u64::from(next) - freed.len() as u64)
            );
        }
        assert!(freed.is_sorted(), "oldest first");
        let waiting = next as usize - freed.len();
        assert!(
            waiting <= OWN_SHARE * STEADY as usize,
            "{waiting} still wait"
        );
        assert!(longest <= 64, "an entry waited {longest} collections");
        assert_eq!(w.take_due(&[2; MAX_TIMELINES])[0], u32::MAX);
    }
}
