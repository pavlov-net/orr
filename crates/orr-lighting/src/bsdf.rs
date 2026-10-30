//! The tables the `bsdf` module reads on the GPU.

use orr_scene::orr_gpu::{
    Allocation, CommandList, Device, Format, Hazards, MemoryKind, Stages, Texture, TextureDesc,
    TextureKind, TextureMemory, TextureRegion, TextureUsage, ViewDesc,
};
use orr_shaders::openpbr::{LUTS, LutFormat};

use crate::Error;

/// `orr_shaders::openpbr::LUTS` as textures with sampled views in eight consecutive heap
/// slots, the layout `bsdf` (through `orr.openpbr`) reads. A kernel that imports `bsdf` sets
/// specialisation constant `openpbr_lut_slot` (104) to [`first_slot`](Self::first_slot).
/// Dropping the tables frees the slots once everything submitted before the drop has
/// completed.
pub struct BsdfTables {
    first_slot: u32,
    device: Device,
    _textures: Vec<Texture>,
    _staging: Allocation<u8>,
}

impl BsdfTables {
    /// Records the uploads into `list`; kernels may read the tables once it has run.
    pub fn new(device: &Device, list: &mut CommandList<'_>) -> Result<Self, Error> {
        let total: usize = LUTS.iter().map(|l| l.bytes.len()).sum();
        let mut staging =
            device.gpu_malloc::<u8>(MemoryKind::Upload, total as u64, "bsdf tables")?;
        let first_slot = device.alloc_image_slots(LUTS.len() as u32)?;
        let base = staging.gpu;
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
            device.write_sampled(first_slot + lut.id, &texture, &ViewDesc::default());
            offset += lut.bytes.len();
            textures.push(texture);
        }
        list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
        Ok(Self {
            first_slot,
            device: device.clone(),
            _textures: textures,
            _staging: staging,
        })
    }

    /// The heap slot of the first table, the value of specialisation constant
    /// `openpbr_lut_slot`.
    pub fn first_slot(&self) -> u32 {
        self.first_slot
    }
}

impl Drop for BsdfTables {
    fn drop(&mut self) {
        for slot in self.first_slot..self.first_slot + LUTS.len() as u32 {
            self.device.free_image_slot_after_submitted(slot);
        }
    }
}
