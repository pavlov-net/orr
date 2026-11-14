//! The resource heap (image descriptors only) and the sampler heap.

use std::mem::DropGuard;
use std::ptr::NonNull;
use std::sync::Mutex;

use ash::vk;

use super::device::{Device, Shared};
use super::garbage::{Item, RawBuffer};
use super::texture::{Texture, vk_address_mode, vk_compare, vk_filter, vk_mipmap_mode};
use crate::{
    Error, FIRST_FREE_IMAGE_SLOT, MemoryKind, SamplerDesc, SamplerIndex, StorageImageIndex,
    TextureIndex, TimelinePoint, ViewDesc,
};

/// Descriptor sizes and limits of one heap.
struct HeapLayout {
    stride: u64,
    stride_align: u64,
    heap_align: u64,
    reserved: u64,
    max: u64,
}

/// One heap buffer and how it is bound.
pub(crate) struct HeapBuffer {
    pub(super) raw: RawBuffer,
    /// Where the heap starts in `raw`.
    address: u64,
    ptr: NonNull<u8>,
    stride: u64,
    slots: u32,
    reserved_offset: u64,
    reserved_size: u64,
}

impl HeapBuffer {
    /// `slots` descriptors, then the driver's reserved range.
    fn new(shared: &Shared, name: &str, slots: u32, l: HeapLayout) -> Result<HeapBuffer, Error> {
        let reserved_offset =
            (u64::from(slots) * l.stride).next_multiple_of(l.stride_align.max(64));
        let size = reserved_offset + l.reserved;
        if size > l.max {
            return Err(Error::Invalid(format!(
                "{name}: {slots} slots need {size} bytes, the device allows {}",
                l.max
            )));
        }
        let usage =
            vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        // The buffer's address need not meet the heap's alignment, so the heap starts at the
        // first address in it that does.
        let heap_align = l.heap_align.max(1);
        let (raw, ptr) =
            shared.create_buffer(MemoryKind::Upload, size + heap_align - 1, usage, 1, name)?;
        let Some(ptr) = ptr else {
            shared.destroy(Item::Buffer(raw));
            return Err(Error::Invalid(format!(
                "{name}: upload memory is not mapped"
            )));
        };
        let address = raw.address.next_multiple_of(heap_align);
        // SAFETY: the buffer holds `heap_align - 1` bytes beyond the heap.
        let ptr = unsafe { ptr.add((address - raw.address) as usize) };
        Ok(HeapBuffer {
            raw,
            address,
            ptr,
            stride: l.stride,
            slots,
            reserved_offset,
            reserved_size: l.reserved,
        })
    }

    fn bind_info(&self) -> vk::BindHeapInfoEXT<'static> {
        vk::BindHeapInfoEXT {
            heap_range: vk::DeviceAddressRangeEXT {
                address: self.address,
                size: self.reserved_offset + self.reserved_size,
            },
            reserved_range_offset: self.reserved_offset,
            reserved_range_size: self.reserved_size,
            ..Default::default()
        }
    }

    /// The host bytes of `slot`.
    fn slot(&self, slot: u32, what: &str) -> vk::HostAddressRangeEXT<'_> {
        assert!(
            slot < self.slots,
            "{what} slot {slot} is outside the heap's {} slots",
            self.slots
        );
        // SAFETY: the slot lies inside the mapped heap, which lives as long as `self`.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr
                    .as_ptr()
                    .add((u64::from(slot) * self.stride) as usize),
                self.stride as usize,
            )
        };
        vk::HostAddressRangeEXT::default().address(bytes)
    }
}

/// Free slots below a capacity: never-used slots count up from `next`; freed ones are reused
/// first.
#[derive(Default)]
pub(super) struct SlotAllocator {
    next: u32,
    free: Vec<u32>,
}

impl SlotAllocator {
    /// Image slots, which never include slot 0, [`TextureIndex::NONE`], or the slots reserved for
    /// the OpenPBR tables.
    fn images() -> Self {
        Self {
            next: FIRST_FREE_IMAGE_SLOT,
            free: Vec::new(),
        }
    }

    pub(super) fn take(&mut self, capacity: u32) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        (self.next < capacity).then(|| {
            self.next += 1;
            self.next - 1
        })
    }

    /// The first of `count` consecutive slots: a run of freed ones, else never-used ones.
    fn take_run(&mut self, count: u32, capacity: u32) -> Option<u32> {
        assert!(count > 0, "a run of no slots");
        let n = count as usize;
        self.free.sort_unstable();
        if let Some(i) = self
            .free
            .windows(n)
            .position(|w| w[n - 1] - w[0] == count - 1)
        {
            let first = self.free[i];
            self.free.drain(i..i + n);
            return Some(first);
        }
        (capacity.saturating_sub(self.next) >= count).then(|| {
            self.next += count;
            self.next - count
        })
    }

    pub(super) fn release(&mut self, slot: u32) {
        self.free.push(slot);
    }
}

/// Panics unless the heap could have handed out `slot`.
fn assert_freeable(slot: u32) {
    assert!(
        slot >= FIRST_FREE_IMAGE_SLOT,
        "freeing the empty or a reserved image slot: {slot}"
    );
}

pub(super) struct Heaps {
    pub(crate) resource: HeapBuffer,
    pub(crate) sampler: HeapBuffer,
    slots: Mutex<SlotAllocator>,
    /// Samplers [`Device::sampler`] has written, from the top slot down.
    shared_samplers: Mutex<Vec<SamplerDesc>>,
}

// SAFETY: the mapped pointers are plain memory; slots are written by one owner at a time.
unsafe impl Send for Heaps {}
// SAFETY: as above.
unsafe impl Sync for Heaps {}

impl Heaps {
    pub(crate) fn new(
        shared: &Shared,
        image_slots: u32,
        sampler_slots: u32,
    ) -> Result<Heaps, Error> {
        let p = &shared.heap_props;
        let resource = HeapBuffer::new(
            shared,
            "orr resource heap",
            image_slots,
            HeapLayout {
                stride: p.image_descriptor_size,
                stride_align: p.image_descriptor_alignment,
                heap_align: p.resource_heap_alignment,
                reserved: p.min_resource_heap_reserved_range,
                max: p.max_resource_heap_size,
            },
        )?;
        let resource = DropGuard::new(resource, |r| shared.destroy(Item::Buffer(r.raw)));
        let sampler = HeapBuffer::new(
            shared,
            "orr sampler heap",
            sampler_slots,
            HeapLayout {
                stride: p.sampler_descriptor_size,
                stride_align: p.sampler_descriptor_alignment,
                heap_align: p.sampler_heap_alignment,
                reserved: p.min_sampler_heap_reserved_range,
                max: p.max_sampler_heap_size,
            },
        )?;
        Ok(Heaps {
            resource: DropGuard::dismiss(resource),
            sampler,
            slots: Mutex::new(SlotAllocator::images()),
            shared_samplers: Mutex::default(),
        })
    }

    pub(super) fn bind(&self, heap_fn: &ash::ext::descriptor_heap::Device, cmd: vk::CommandBuffer) {
        // SAFETY: `cmd` is recording on a graphics or compute queue; both heaps are live.
        unsafe {
            heap_fn.cmd_bind_resource_heap(cmd, &self.resource.bind_info());
            heap_fn.cmd_bind_sampler_heap(cmd, &self.sampler.bind_info());
        }
    }

    pub(super) fn release_image_slot(&self, slot: u32) {
        self.slots.lock().unwrap().release(slot);
    }
}

impl Device {
    /// Image slots of the resource heap; every slot index is below it.
    pub fn image_slots(&self) -> u32 {
        self.shared.heaps().resource.slots
    }

    /// Takes a free image slot, never [`TextureIndex::NONE`]. Sampled and storage views
    /// share the slot space.
    pub fn alloc_image_slot(&self) -> Result<u32, Error> {
        let heaps = self.shared.heaps();
        heaps
            .slots
            .lock()
            .unwrap()
            .take(heaps.resource.slots)
            .ok_or(Error::HeapFull("image"))
    }

    /// Takes `count` consecutive free image slots and returns the first, for views a shader
    /// reaches as a base slot plus an offset. Each slot is freed on its own.
    pub fn alloc_image_slots(&self, count: u32) -> Result<u32, Error> {
        let heaps = self.shared.heaps();
        heaps
            .slots
            .lock()
            .unwrap()
            .take_run(count, heaps.resource.slots)
            .ok_or(Error::HeapFull("image"))
    }

    /// Returns `slot` to the free list once `after` has completed.
    pub fn free_image_slot(&self, slot: u32, after: TimelinePoint) {
        assert_freeable(slot);
        self.shared
            .defer(self.shared.after_point(after), Item::ImageSlot(slot));
    }

    /// Returns `slot` to the free list once the GPU is done with the work submitted so far
    /// and with the command lists open now, as dropping a [`Texture`] does.
    pub fn free_image_slot_after_submitted(&self, slot: u32) {
        assert_freeable(slot);
        self.shared.defer_after_submitted(Item::ImageSlot(slot));
    }

    /// Writes a sampled view of `tex` into `slot`. A slot is written once while live; a
    /// changed view takes a new slot.
    pub fn write_sampled(&self, slot: u32, tex: &Texture, view: &ViewDesc) -> TextureIndex {
        self.write_image(slot, tex, view, vk::DescriptorType::SAMPLED_IMAGE);
        TextureIndex(slot)
    }

    /// Writes a storage view of `tex` into `slot`; `tex` needs `TextureUsage::STORAGE`.
    pub fn write_storage(&self, slot: u32, tex: &Texture, view: &ViewDesc) -> StorageImageIndex {
        self.write_image(slot, tex, view, vk::DescriptorType::STORAGE_IMAGE);
        StorageImageIndex(slot)
    }

    fn write_image(&self, slot: u32, tex: &Texture, view: &ViewDesc, ty: vk::DescriptorType) {
        let heaps = self.shared.heaps();
        let view = tex.view_info(view);
        let image = vk::ImageDescriptorInfoEXT::default()
            .view(&view)
            .layout(vk::ImageLayout::GENERAL);
        let info =
            [vk::ResourceDescriptorInfoEXT::default()
                .ty(ty)
                .data(vk::ResourceDescriptorDataEXT {
                    p_image: &raw const image,
                })];
        let dst = [heaps.resource.slot(slot, "image")];
        // SAFETY: the destination is one descriptor's bytes inside the mapped heap.
        unsafe { self.shared.heap_fn.write_resource_descriptors(&info, &dst) }
            .expect("vkWriteResourceDescriptorsEXT");
    }

    /// A sampler slot holding `desc`, shared by every caller asking for the same sampler and
    /// live as long as the device. These slots are taken from the top of the sampler heap
    /// down; slots written with [`Device::write_sampler`] must stay below them.
    pub fn sampler(&self, desc: &SamplerDesc) -> Result<SamplerIndex, Error> {
        let heaps = self.shared.heaps();
        let mut shared = heaps.shared_samplers.lock().unwrap();
        let top = heaps.sampler.slots;
        if let Some(i) = shared.iter().position(|d| d == desc) {
            return Ok(SamplerIndex(top - 1 - i as u32));
        }
        if shared.len() as u32 == top {
            return Err(Error::HeapFull("sampler"));
        }
        let slot = SamplerIndex(top - 1 - shared.len() as u32);
        self.write_sampler(slot, desc);
        shared.push(*desc);
        Ok(slot)
    }

    /// Writes a sampler into sampler-heap slot `slot`.
    pub fn write_sampler(&self, slot: SamplerIndex, desc: &SamplerDesc) {
        let heaps = self.shared.heaps();
        let anisotropy = desc.max_anisotropy > 1.0;
        let info = [vk::SamplerCreateInfo::default()
            .mag_filter(vk_filter(desc.mag))
            .min_filter(vk_filter(desc.min))
            .mipmap_mode(vk_mipmap_mode(desc.mip))
            .address_mode_u(vk_address_mode(desc.address[0]))
            .address_mode_v(vk_address_mode(desc.address[1]))
            .address_mode_w(vk_address_mode(desc.address[2]))
            .anisotropy_enable(anisotropy)
            .max_anisotropy(if anisotropy { desc.max_anisotropy } else { 1.0 })
            .compare_enable(desc.compare.is_some())
            .compare_op(desc.compare.map_or(vk::CompareOp::NEVER, vk_compare))
            .mip_lod_bias(desc.mip_bias)
            .min_lod(desc.min_lod)
            .max_lod(desc.max_lod)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)];
        let dst = [heaps.sampler.slot(slot.0, "sampler")];
        // SAFETY: the destination is one descriptor's bytes inside the mapped heap.
        unsafe { self.shared.heap_fn.write_sampler_descriptors(&info, &dst) }
            .expect("vkWriteSamplerDescriptorsEXT");
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;

    use super::SlotAllocator;
    use crate::{StorageImageIndex, TextureIndex};

    #[test]
    fn zeroed_image_indices_name_a_slot_the_heap_never_hands_out() {
        assert_eq!(TextureIndex::zeroed(), TextureIndex::NONE);
        assert_eq!(StorageImageIndex::zeroed(), StorageImageIndex::NONE);
        let mut a = SlotAllocator::images();
        let run = a.take_run(3, 16).unwrap();
        let rest: Vec<u32> = std::iter::from_fn(|| a.take(16)).collect();
        assert_eq!((run, rest), (9, vec![12, 13, 14, 15]));
    }

    #[test]
    fn runs_are_consecutive_and_reuse_freed_runs() {
        let mut a = SlotAllocator::default();
        let singles: Vec<u32> = Vec::from_fn(6, |_| a.take(100).unwrap());
        for s in [4, 1, 2, 5] {
            a.release(singles[s]);
        }
        assert_eq!(a.take_run(3, 100), Some(6), "no three freed slots in a row");
        assert_eq!(a.take_run(2, 100), Some(1));
        assert_eq!(a.take_run(2, 100), Some(4));
        assert_eq!(a.take_run(91, 100), Some(9));
        assert_eq!(a.take_run(1, 100), None);
    }
}
