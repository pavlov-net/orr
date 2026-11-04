//! The texture heap and sampler table as `abi.slang` reads them on Metal: argument-table
//! buffer 1 holds the base resource ID of a texture view pool, so heap slot `i` is that ID
//! plus `i`, and buffer 2 holds one sampler resource ID per sampler slot.

use std::sync::Mutex;

use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBuffer, MTLCompareFunction, MTLDevice, MTLResource, MTLResourceID, MTLResourceOptions,
    MTLResourceViewPool, MTLResourceViewPoolDescriptor, MTLSamplerAddressMode,
    MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLSamplerState, MTLTextureType, MTLTextureViewDescriptor, MTLTextureViewPool,
};

use super::device::{Device, DeviceDesc, metal_error};
use super::texture::{Texture, pixel_format};
use super::{Buffer, Obj, lock};
use crate::{
    AddressMode, CompareOp, Error, Filter, SamplerDesc, SamplerIndex, StorageImageIndex,
    TextureIndex, TextureKind, TimelinePoint, ViewDesc, ViewKind,
};

/// The view pool, the buffers argument-table slots 1 and 2 point at, and the free slots.
pub(super) struct Bindings {
    pool: Obj<ProtocolObject<dyn MTLTextureViewPool>>,
    /// One `u64`: the texture view pool's base resource ID.
    pub(super) pool_base: Obj<Buffer>,
    /// One `u64` resource ID per sampler slot.
    pub(super) samplers: Obj<Buffer>,
    slots: Mutex<Slots>,
    sampler_states: Mutex<Samplers>,
}

struct Slots {
    capacity: u32,
    /// Slots never handed out start here; slot 0 never is.
    next: u32,
    free: Vec<u32>,
}

struct Samplers {
    /// The state behind each slot, kept alive while the slot names it.
    states: Vec<Option<Obj<ProtocolObject<dyn MTLSamplerState>>>>,
    /// Descriptions [`Device::sampler`] handed out, from the top slot down.
    shared: Vec<SamplerDesc>,
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
        let pool_desc = MTLResourceViewPoolDescriptor::new();
        // SAFETY: a count the device validates.
        unsafe { pool_desc.setResourceViewCount(desc.image_heap_slots as usize) };
        pool_desc.setLabel(Some(&NSString::from_str("image heap")));
        let pool = device
            .newTextureViewPoolWithDescriptor_error(&pool_desc)
            .map_err(|e| metal_error("newTextureViewPoolWithDescriptor", &e))?;
        let pool_base = table(8, "texture view pool base")?;
        write_id(&pool_base, 0, pool.baseResourceID());
        let sampler_slots = desc.sampler_heap_slots as usize;
        Ok(Self {
            pool: Obj(pool),
            pool_base,
            samplers: table(8 * sampler_slots, "sampler table")?,
            slots: Mutex::new(Slots {
                capacity: desc.image_heap_slots,
                next: 1,
                free: Vec::new(),
            }),
            sampler_states: Mutex::new(Samplers {
                states: vec![None; sampler_slots],
                shared: Vec::new(),
            }),
        })
    }

    pub(super) fn buffers(&self) -> [&Obj<Buffer>; 2] {
        [&self.pool_base, &self.samplers]
    }
}

/// The 64-bit value of a resource ID, which shaders reinterpret as the resource.
pub(super) fn id_bits(id: MTLResourceID) -> u64 {
    const _: () = assert!(size_of::<MTLResourceID>() == 8);
    // SAFETY: `MTLResourceID` is a `repr(C)` struct of one `u64`.
    unsafe { std::mem::transmute::<MTLResourceID, u64>(id) }
}

/// Writes `id` as the `index`th `u64` of the shared `buffer`.
fn write_id(buffer: &Buffer, index: usize, id: MTLResourceID) {
    assert!(
        (index + 1) * 8 <= buffer.length(),
        "resource ID {index} out of range"
    );
    // SAFETY: shared storage stays mapped; the index is in range (checked above) and the
    // contents are 8-byte aligned.
    unsafe {
        buffer
            .contents()
            .cast::<u64>()
            .as_ptr()
            .add(index)
            .write(id_bits(id));
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
        self.write_view(slot, tex, view);
        TextureIndex(slot)
    }

    /// Writes a storage view of `tex` into `slot`.
    pub fn write_storage(&self, slot: u32, tex: &Texture, view: &ViewDesc) -> StorageImageIndex {
        self.write_view(slot, tex, view);
        StorageImageIndex(slot)
    }

    fn write_view(&self, slot: u32, tex: &Texture, view: &ViewDesc) {
        let b = &self.shared.bindings;
        assert!(
            slot < lock(&b.slots).capacity,
            "image slot {slot} out of range"
        );
        let id = if *view == ViewDesc::default() {
            // SAFETY: the slot is within the pool (checked above).
            unsafe { b.pool.setTextureView_atIndex(&tex.raw, slot as usize) }
        } else {
            let d = view_descriptor(tex, view);
            // SAFETY: as above.
            unsafe {
                b.pool
                    .setTextureView_descriptor_atIndex(&tex.raw, &d, slot as usize)
            }
        };
        debug_assert_eq!(
            id_bits(id),
            id_bits(b.pool.baseResourceID()) + u64::from(slot),
            "view pool IDs are not the base ID plus the slot"
        );
    }

    /// A sampler slot holding `desc`; equal descriptions share one, handed out from the top
    /// slot down.
    pub fn sampler(&self, desc: &SamplerDesc) -> Result<SamplerIndex, Error> {
        let top = self.caps().limits.max_sampler_slots;
        let slot = {
            let mut s = lock(&self.shared.bindings.sampler_states);
            if let Some(i) = s.shared.iter().position(|d| d == desc) {
                return Ok(SamplerIndex(top - 1 - i as u32));
            }
            if s.shared.len() as u32 == top {
                return Err(Error::HeapFull("sampler"));
            }
            s.shared.push(*desc);
            SamplerIndex(top - s.shared.len() as u32)
        };
        self.write_sampler(slot, desc);
        Ok(slot)
    }

    /// Writes `desc` into sampler slot `slot`.
    pub fn write_sampler(&self, slot: SamplerIndex, desc: &SamplerDesc) {
        let d = MTLSamplerDescriptor::new();
        d.setSupportArgumentBuffers(true);
        let filter = |f| match f {
            Filter::Nearest => MTLSamplerMinMagFilter::Nearest,
            Filter::Linear => MTLSamplerMinMagFilter::Linear,
        };
        d.setMagFilter(filter(desc.mag));
        d.setMinFilter(filter(desc.min));
        d.setMipFilter(match desc.mip {
            Filter::Nearest => MTLSamplerMipFilter::Nearest,
            Filter::Linear => MTLSamplerMipFilter::Linear,
        });
        let address = |a| match a {
            AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
            AddressMode::MirroredRepeat => MTLSamplerAddressMode::MirrorRepeat,
            AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
            AddressMode::ClampToBorder => MTLSamplerAddressMode::ClampToBorderColor,
        };
        d.setSAddressMode(address(desc.address[0]));
        d.setTAddressMode(address(desc.address[1]));
        d.setRAddressMode(address(desc.address[2]));
        d.setBorderColor(MTLSamplerBorderColor::TransparentBlack);
        d.setMaxAnisotropy(desc.max_anisotropy.clamp(1.0, 16.0) as usize);
        if let Some(c) = desc.compare {
            d.setCompareFunction(compare(c));
        }
        d.setLodBias(desc.mip_bias);
        d.setLodMinClamp(desc.min_lod);
        d.setLodMaxClamp(desc.max_lod);
        let state = self
            .shared
            .device
            .newSamplerStateWithDescriptor(&d)
            .expect("newSamplerStateWithDescriptor");
        let b = &self.shared.bindings;
        write_id(&b.samplers, slot.0 as usize, state.gpuResourceID());
        let old = lock(&b.sampler_states).states[slot.0 as usize].replace(Obj(state));
        if let Some(old) = old {
            self.shared.defer(Box::new(move || drop(old)));
        }
    }
}

pub(super) fn compare(c: CompareOp) -> MTLCompareFunction {
    match c {
        CompareOp::Never => MTLCompareFunction::Never,
        CompareOp::Less => MTLCompareFunction::Less,
        CompareOp::LessOrEqual => MTLCompareFunction::LessEqual,
        CompareOp::Equal => MTLCompareFunction::Equal,
        CompareOp::NotEqual => MTLCompareFunction::NotEqual,
        CompareOp::Greater => MTLCompareFunction::Greater,
        CompareOp::GreaterOrEqual => MTLCompareFunction::GreaterEqual,
        CompareOp::Always => MTLCompareFunction::Always,
    }
}

/// The view of `view`'s mips and layers of `tex`, in its natural kind unless `view` names one.
fn view_descriptor(
    tex: &Texture,
    view: &ViewDesc,
) -> objc2::rc::Retained<MTLTextureViewDescriptor> {
    let desc = tex.desc();
    let d = MTLTextureViewDescriptor::new();
    d.setPixelFormat(pixel_format(desc.format).expect("the texture's format"));
    let natural = match desc.kind {
        TextureKind::D2 if desc.layers > 1 => ViewKind::D2Array,
        TextureKind::D2 => ViewKind::D2,
        TextureKind::D3 => ViewKind::D3,
        TextureKind::Cube if desc.layers > 6 => ViewKind::CubeArray,
        TextureKind::Cube => ViewKind::Cube,
    };
    d.setTextureType(match view.kind.unwrap_or(natural) {
        ViewKind::D2 => MTLTextureType::Type2D,
        ViewKind::D2Array => MTLTextureType::Type2DArray,
        ViewKind::D3 => MTLTextureType::Type3D,
        ViewKind::Cube => MTLTextureType::TypeCube,
        ViewKind::CubeArray => MTLTextureType::TypeCubeArray,
    });
    let mips = if view.mips == 0 {
        desc.mips - view.base_mip
    } else {
        view.mips
    };
    let layers = if view.layers == 0 {
        desc.layers - view.base_layer
    } else {
        view.layers
    };
    // SAFETY: ranges the device validates against the texture.
    unsafe {
        d.setLevelRange(NSRange::new(view.base_mip as usize, mips as usize));
        d.setSliceRange(NSRange::new(view.base_layer as usize, layers as usize));
    }
    d
}
