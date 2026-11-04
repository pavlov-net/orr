//! The tables the `orr_lighting.bsdf` module reads on the GPU.

use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::Duration;

use crate::openpbr::{LUTS, LutFormat};
use orr_scene::orr_gpu::{
    self, Device, FIRST_FREE_IMAGE_SLOT, Format, Hazards, MemoryKind, OPENPBR_LUT_SLOT, QueueKind,
    Stages, Texture, TextureDesc, TextureKind, TextureMemory, TextureRegion, TextureUsage,
    ViewDesc,
};

use crate::Error;

/// How long to wait for the GPU before giving up on the tables.
const WAIT: Duration = Duration::from_secs(60);

/// [`LUTS`] as textures with sampled views in the reserved heap slots from
/// `OPENPBR_LUT_SLOT`, the layout `orr_lighting.bsdf` (through `orr_lighting.openpbr`) reads, and
/// where specialization constant `openpbr_lut_slot` (104) points by default. Keep a set alive while
/// kernels that import `orr_lighting.bsdf` run. Every set on a device shares one upload, freed once
/// everything submitted before the last set's drop has completed.
#[derive(Clone)]
pub struct BsdfTables {
    _live: Arc<Live>,
}

/// One device's tables.
struct Live {
    device: Device,
    _textures: Vec<Texture>,
}

/// The tables of each device that has a set.
static LIVE: Mutex<Vec<Weak<Live>>> = Mutex::new(Vec::new());

impl BsdfTables {
    /// The device's tables, uploaded on the first call and when the last set has been dropped.
    /// Kernels may read them when this returns. An upload submits to the render queue, so call
    /// this on its owner thread, and blocks until all submitted work on the device has
    /// completed; other calls wait for it.
    pub fn new(device: &Device) -> Result<Self, Error> {
        // The list of weak references stays valid if an upload panicked.
        let mut live = LIVE.lock().unwrap_or_else(PoisonError::into_inner);
        live.retain(|l| l.strong_count() > 0);
        if let Some(l) = live
            .iter()
            .filter_map(Weak::upgrade)
            .find(|l| l.device == *device)
        {
            return Ok(Self { _live: l });
        }
        let l = Arc::new(Live {
            device: device.clone(),
            _textures: upload(device)?,
        });
        live.push(Arc::downgrade(&l));
        Ok(Self { _live: l })
    }
}

/// Uploads the tables and writes the reserved slots. Work from an earlier set may still read the
/// slots, so this first waits for everything submitted.
fn upload(device: &Device) -> Result<Vec<Texture>, Error> {
    if device.image_slots() < FIRST_FREE_IMAGE_SLOT {
        return Err(orr_gpu::Error::HeapFull("image").into());
    }
    device.wait_submitted(WAIT)?;
    let queue = device.queue(QueueKind::Render).expect("render queue");
    let mut list = queue.command_list()?;
    let total: usize = LUTS.iter().map(|l| l.bytes.len()).sum();
    let mut staging = device.gpu_malloc::<u8>(MemoryKind::Upload, total as u64, "bsdf tables")?;
    let base = staging.gpu();
    let mapped = staging.as_mut_slice().expect("upload memory is mapped");
    let mut textures = Vec::with_capacity(LUTS.len());
    let mut offset = 0;
    for lut in &LUTS {
        let [w, h, d] = lut.extent;
        let desc = TextureDesc {
            kind: if d > 1 {
                TextureKind::D3
            } else {
                TextureKind::D2
            },
            format: match lut.format {
                LutFormat::R16Unorm => Format::R16Unorm,
                LutFormat::Rgba32Float => Format::Rgba32Float,
            },
            extent: [w, h, d],
            mips: 1,
            layers: 1,
            usage: TextureUsage::SAMPLED,
        };
        let texture = device.create_texture(&desc, TextureMemory::Dedicated, lut.name)?;
        mapped[offset..offset + lut.bytes.len()].copy_from_slice(lut.bytes);
        list.discard(&texture);
        list.copy_to_texture(
            &texture,
            &TextureRegion::whole_mip(&desc, 0),
            base.offset(offset as u64),
        );
        device.write_sampled(OPENPBR_LUT_SLOT + lut.id, &texture, &ViewDesc::default());
        offset += lut.bytes.len();
        textures.push(texture);
    }
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
    let point = queue.submit(vec![list], &[])?;
    queue.wait(point, WAIT)?;
    Ok(textures)
}

#[cfg(test)]
mod tests {
    use super::OPENPBR_LUT_SLOT;
    use crate::shaders::path::opaque;

    #[test]
    fn kernels_default_to_the_reserved_table_slot() {
        assert_eq!(opaque::Spec::default().openpbr_lut_slot, OPENPBR_LUT_SLOT);
    }
}
