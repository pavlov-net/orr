//! Deferred destruction: objects freed while the GPU may still use them wait for timeline
//! points, then `Device::collect_garbage` destroys them.

use ash::vk;

use super::device::Shared;
use super::memory::MemoryKind;

/// Timelines a device can have: one per queue kind.
pub(crate) const MAX_TIMELINES: usize = 5;

/// Per timeline, the value that must complete before an item is destroyed (0: none).
pub(crate) type After = [u64; MAX_TIMELINES];

pub(crate) enum Item {
    Buffer(RawBuffer),
    Image {
        image: vk::Image,
        memory: Option<vk::DeviceMemory>,
        view: vk::ImageView,
    },
    Pipeline(vk::Pipeline),
    Accel(vk::AccelerationStructureKHR),
    /// A sparse buffer; the memory bound to it belongs to others.
    SparseBuffer {
        buffer: vk::Buffer,
        address: u64,
    },
    ImageSlot(u32),
    RaygenRecord(u32),
}

/// A buffer with its own memory, as `gpu_malloc` creates it.
#[derive(Clone, Copy)]
pub(crate) struct RawBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub kind: MemoryKind,
    pub size: u64,
    pub address: u64,
}

#[derive(Default)]
pub(crate) struct Garbage {
    entries: Vec<(After, Item)>,
}

impl Garbage {
    pub(crate) fn push(&mut self, after: After, item: Item) {
        self.entries.push((after, item));
    }

    /// Removes and returns the items whose points have all completed.
    pub(crate) fn take_due(&mut self, completed: &After) -> Vec<Item> {
        let mut due = Vec::new();
        let mut i = 0;
        while i < self.entries.len() {
            if self.entries[i].0.iter().zip(completed).all(|(a, c)| a <= c) {
                due.push(self.entries.swap_remove(i).1);
            } else {
                i += 1;
            }
        }
        due
    }

    pub(crate) fn take_all(&mut self) -> Vec<Item> {
        self.entries.drain(..).map(|(_, item)| item).collect()
    }
}

impl Shared {
    /// Destroys `item` once every timeline reaches its value in `after`.
    pub(crate) fn defer(&self, after: After, item: Item) {
        self.garbage.lock().unwrap().push(after, item);
    }

    /// Destroys `item` once everything submitted so far has completed.
    pub(crate) fn defer_after_submitted(&self, item: Item) {
        self.defer(self.submitted(), item);
    }

    pub(crate) fn collect(&self) {
        let completed = self.completed();
        let due = self.garbage.lock().unwrap().take_due(&completed);
        for item in due {
            self.destroy(item);
        }
    }

    pub(crate) fn destroy(&self, item: Item) {
        let d = &self.device;
        // SAFETY: every item is owned by the garbage list alone and no submitted work
        // references it any more.
        unsafe {
            match item {
                Item::Buffer(b) => {
                    self.unregister_buffer(&b);
                    d.destroy_buffer(b.buffer, None);
                    d.free_memory(b.memory, None);
                }
                Item::Image {
                    image,
                    memory,
                    view,
                } => {
                    d.destroy_image_view(view, None);
                    d.destroy_image(image, None);
                    if let Some(m) = memory {
                        d.free_memory(m, None);
                    }
                }
                Item::Pipeline(p) => d.destroy_pipeline(p, None),
                Item::Accel(a) => self.accel_fn.destroy_acceleration_structure(a, None),
                Item::SparseBuffer { buffer, address } => {
                    self.unregister_range(address);
                    d.destroy_buffer(buffer, None);
                }
                Item::ImageSlot(slot) => self.heaps().release_image_slot(slot),
                Item::RaygenRecord(slot) => {
                    if let Some(r) = self.records.get() {
                        r.release(slot);
                    }
                }
            }
        }
    }
}
