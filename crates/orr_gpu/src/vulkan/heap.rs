//! The resource heap (image descriptors only) and the sampler heap (contracts.md 9).

use std::ptr::NonNull;
use std::sync::Mutex;

use ash::vk;

use super::device::{Device, Shared};
use super::garbage::{Item, RawBuffer};
use super::memory::MemoryKind;
use super::texture::{Texture, vk_address_mode, vk_compare, vk_filter, vk_mipmap_mode};
use crate::{
    Error, SamplerDesc, SamplerIndex, StorageImageIndex, TextureIndex, TimelinePoint, ViewDesc,
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
    pub raw: RawBuffer,
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
        let (raw, ptr) =
            shared.create_buffer(MemoryKind::Upload, size, usage, l.heap_align.max(1), name)?;
        let ptr =
            ptr.ok_or_else(|| Error::Invalid(format!("{name}: upload memory is not mapped")))?;
        Ok(HeapBuffer {
            raw,
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
                address: self.raw.address,
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

/// Free image slots: never-used slots count up from `next`; freed ones are reused first.
struct SlotAllocator {
    next: u32,
    free: Vec<u32>,
}

pub(crate) struct Heaps {
    pub resource: HeapBuffer,
    pub sampler: HeapBuffer,
    slots: Mutex<SlotAllocator>,
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
        );
        let sampler = match sampler {
            Ok(s) => s,
            Err(e) => {
                shared.destroy(Item::Buffer(resource.raw));
                return Err(e);
            }
        };
        Ok(Heaps {
            resource,
            sampler,
            slots: Mutex::new(SlotAllocator {
                next: 0,
                free: Vec::new(),
            }),
        })
    }

    pub(crate) fn bind(&self, heap_fn: &ash::ext::descriptor_heap::Device, cmd: vk::CommandBuffer) {
        // SAFETY: `cmd` is recording on a graphics or compute queue; both heaps are live.
        unsafe {
            heap_fn.cmd_bind_resource_heap(cmd, &self.resource.bind_info());
            heap_fn.cmd_bind_sampler_heap(cmd, &self.sampler.bind_info());
        }
    }

    pub(crate) fn release_image_slot(&self, slot: u32) {
        self.slots.lock().unwrap().free.push(slot);
    }
}

impl Device {
    /// Takes a free image slot. Sampled and storage views share the slot space.
    pub fn alloc_image_slot(&self) -> Result<u32, Error> {
        let heaps = self.shared.heaps();
        let mut s = heaps.slots.lock().unwrap();
        if let Some(slot) = s.free.pop() {
            return Ok(slot);
        }
        if s.next >= heaps.resource.slots {
            return Err(Error::HeapFull("image"));
        }
        s.next += 1;
        Ok(s.next - 1)
    }

    /// Returns `slot` to the free list once `after` has completed.
    pub fn free_image_slot(&self, slot: u32, after: TimelinePoint) {
        self.shared
            .defer(self.shared.after_point(after), Item::ImageSlot(slot));
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
        let info = [vk::ResourceDescriptorInfoEXT::default()
            .ty(ty)
            .data(vk::ResourceDescriptorDataEXT { p_image: &image })];
        let dst = [heaps.resource.slot(slot, "image")];
        // SAFETY: the destination is one descriptor's bytes inside the mapped heap.
        unsafe { self.shared.heap_fn.write_resource_descriptors(&info, &dst) }
            .expect("vkWriteResourceDescriptorsEXT");
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
