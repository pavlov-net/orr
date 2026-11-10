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

/// Allocations with device memory of their own one collection frees at most: a
/// `vkFreeMemory` costs the driver real time, and an eviction can empty hundreds of pool
/// blocks at once. The rest wait for the next collections.
const OWN_FREES: usize = 2;

impl Item {
    /// It frees a `VkDeviceMemory` of its own.
    fn owns_memory(&self) -> bool {
        match self {
            Item::Buffer(b) => b.allocation.dedicated,
            Item::Image { memory, .. } => memory.as_ref().is_some_and(|(m, _)| m.dedicated),
            _ => false,
        }
    }
}

#[derive(Default)]
pub(super) struct Garbage {
    entries: Vec<(After, Item)>,
}

impl Garbage {
    fn push(&mut self, after: After, item: Item) {
        self.entries.push((after, item));
    }

    /// Removes and returns the items whose points have all completed, with at most
    /// [`OWN_FREES`] of those holding device memory of their own.
    fn take_due(&mut self, completed: &After) -> Vec<Item> {
        let mut due = Vec::new();
        let mut own = 0;
        let mut i = 0;
        while i < self.entries.len() {
            let (after, item) = &self.entries[i];
            let ready = after.iter().zip(completed).all(|(a, c)| a <= c);
            if ready && (own < OWN_FREES || !item.owns_memory()) {
                own += usize::from(item.owns_memory());
                due.push(self.entries.swap_remove(i).1);
            } else {
                i += 1;
            }
        }
        due
    }

    pub(super) fn take_all(&mut self) -> Vec<Item> {
        self.entries.drain(..).map(|(_, item)| item).collect()
    }
}

impl Shared {
    /// Destroys `item` once every timeline reaches its value in `after`.
    pub(super) fn defer(&self, after: After, item: Item) {
        self.garbage.lock().unwrap().push(after, item);
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
