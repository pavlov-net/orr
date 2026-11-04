//! The texture heap and sampler table as `abi.slang` reads them on Metal: argument-table
//! buffer 1 holds the base resource ID of a texture view pool, so heap slot `i` is that ID
//! plus `i`, and buffer 2 holds one sampler resource ID per sampler slot.

use std::sync::Mutex;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLResource, MTLResourceOptions};

use super::device::{Device, DeviceDesc};
use super::texture::Texture;
use super::{Buffer, Obj, lock};
use crate::{
    Error, SamplerDesc, SamplerIndex, StorageImageIndex, TextureIndex, TimelinePoint, ViewDesc,
};

/// The buffers argument-table slots 1 and 2 point at, and the heap's free slots.
pub(super) struct Bindings {
    /// One `u64`: the texture view pool's base resource ID.
    pub(super) pool_base: Obj<Buffer>,
    /// One `u64` resource ID per sampler slot.
    pub(super) samplers: Obj<Buffer>,
    slots: Mutex<Slots>,
}

struct Slots {
    capacity: u32,
    /// Slots never handed out start here; slot 0 never is.
    next: u32,
    free: Vec<u32>,
}

impl Bindings {
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        desc: &DeviceDesc<'_>,
    ) -> Result<Self, Error> {
        let table = |bytes: usize, label: &str| {
            let options = MTLResourceOptions::StorageModeShared
                | MTLResourceOptions::HazardTrackingModeUntracked;
            let b = device
                .newBufferWithLength_options(bytes.max(8), options)
                .ok_or(Error::Metal {
                    call: "newBufferWithLength",
                    message: format!("{label}: no memory"),
                })?;
            b.setLabel(Some(&NSString::from_str(label)));
            Ok::<_, Error>(Obj(b))
        };
        Ok(Self {
            pool_base: table(8, "texture view pool base")?,
            samplers: table(8 * desc.sampler_heap_slots as usize, "sampler table")?,
            slots: Mutex::new(Slots {
                capacity: desc.image_heap_slots,
                next: 1,
                free: Vec::new(),
            }),
        })
    }

    pub(super) fn buffers(&self) -> [&Obj<Buffer>; 2] {
        [&self.pool_base, &self.samplers]
    }
}

impl Device {
    /// The number of image slots, slot 0 included.
    pub fn image_slots(&self) -> u32 {
        lock(&self.shared.bindings.slots).capacity
    }

    /// A free image slot.
    pub fn alloc_image_slot(&self) -> Result<u32, Error> {
        let mut s = lock(&self.shared.bindings.slots);
        if let Some(slot) = s.free.pop() {
            return Ok(slot);
        }
        if s.next >= s.capacity {
            return Err(Error::HeapFull("image"));
        }
        s.next += 1;
        Ok(s.next - 1)
    }

    /// `count` consecutive free image slots; returns the first.
    pub fn alloc_image_slots(&self, count: u32) -> Result<u32, Error> {
        let mut s = lock(&self.shared.bindings.slots);
        if s.capacity - s.next < count {
            return Err(Error::HeapFull("image"));
        }
        s.next += count;
        Ok(s.next - count)
    }

    /// Returns `slot` once `after` has completed.
    pub fn free_image_slot(&self, slot: u32, after: TimelinePoint) {
        let shared = std::sync::Arc::clone(&self.shared);
        self.shared.defer_until(
            after.timeline,
            after.value,
            Box::new(move || lock(&shared.bindings.slots).free.push(slot)),
        );
    }

    /// Returns `slot` once everything submitted so far has completed.
    pub fn free_image_slot_after_submitted(&self, slot: u32) {
        let shared = std::sync::Arc::clone(&self.shared);
        self.shared.defer(Box::new(move || {
            lock(&shared.bindings.slots).free.push(slot)
        }));
    }

    /// Writes a sampled view of `tex` into `slot`.
    pub fn write_sampled(&self, slot: u32, tex: &Texture, view: &ViewDesc) -> TextureIndex {
        let _ = (slot, tex, view);
        super::not_implemented("Device::write_sampled")
    }

    /// Writes a storage view of `tex` into `slot`.
    pub fn write_storage(&self, slot: u32, tex: &Texture, view: &ViewDesc) -> StorageImageIndex {
        let _ = (slot, tex, view);
        super::not_implemented("Device::write_storage")
    }

    /// A sampler slot holding `desc`.
    pub fn sampler(&self, desc: &SamplerDesc) -> Result<SamplerIndex, Error> {
        let _ = desc;
        Err(Error::NotImplemented("Device::sampler"))
    }

    /// Writes `desc` into sampler slot `slot`.
    pub fn write_sampler(&self, slot: SamplerIndex, desc: &SamplerDesc) {
        let _ = (slot, desc);
        super::not_implemented("Device::write_sampler")
    }
}
