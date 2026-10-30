//! The baked environment: each frame the linked environment is evaluated at the scene focus
//! into an equal-area octahedral map, then reduced to its mean, the dome statistics and an
//! importance pyramid for sampling.

use bytemuck::{Pod, Zeroable};
use orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, Format, GpuPtr, Hazards, MemoryKind,
    SamplerIndex, Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory,
    TextureUsage, TimelinePoint, ViewDesc,
};

use crate::shaders::environment::{bake, pyramid};
use crate::shaders::types::{EnvironmentGpu, SkyStats};
use crate::{Error, KernelSet, pipeline};

/// Texels per side of the baked map.
pub const ENVIRONMENT_SIZE: u32 = 256;

const LEVELS: u32 = ENVIRONMENT_SIZE.trailing_zeros();
const GROUPS: u32 = ENVIRONMENT_SIZE / 8;

/// Floats in the importance pyramid: every level from `ENVIRONMENT_SIZE`^2 down to 1.
const PYRAMID_LEN: u64 = (4 * ENVIRONMENT_SIZE as u64 * ENVIRONMENT_SIZE as u64 - 1) / 3;

/// Data of the default environment, a cooked equal-area octahedral HDR map: what
/// `SceneWriter::set_environment` points at when nothing else is linked.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, Pod, Zeroable)]
pub struct EnvironmentMapGpu {
    pub map: TextureIndex,
    pub sampler: SamplerIndex,
    /// Scales the map's texels to nits.
    pub intensity: f32,
    /// The cooked map's `TextureHeader::primaries`; texels convert to Rec.2020 at sampling.
    pub primaries: u32,
    /// Rotates world directions into map space, rows `[R | 0]`.
    pub from_world: [[f32; 4]; 3],
}

pub(crate) struct EnvironmentBake {
    bake: ComputePipeline,
    mean: ComputePipeline,
    build: ComputePipeline,
    map: Texture,
    map_slots: [u32; 2],
    storage: StorageImageIndex,
    pyramid: Allocation<f32>,
    partials: Allocation<[f32; 4]>,
    record: Allocation<EnvironmentGpu>,
    /// Index 0 is `mean`'s root, index `l` the root building level `l`.
    pyramid_roots: Allocation<pyramid::Root>,
    initialised: bool,
}

impl EnvironmentBake {
    pub(crate) fn new(
        device: &Device,
        kernels: &KernelSet,
        sky: GpuPtr<SkyStats>,
    ) -> Result<Self, Error> {
        let desc = TextureDesc::d2(
            Format::Rgba16Float,
            ENVIRONMENT_SIZE,
            ENVIRONMENT_SIZE,
            TextureUsage::SAMPLED | TextureUsage::STORAGE,
        );
        let map = device.create_texture(&desc, TextureMemory::Dedicated, "environment map")?;
        let map_slots = [device.alloc_image_slot()?, device.alloc_image_slot()?];
        let storage = device.write_storage(map_slots[0], &map, &ViewDesc::default());
        let sampled = device.write_sampled(map_slots[1], &map, &ViewDesc::default());
        let pyramid = device.gpu_malloc::<f32>(
            MemoryKind::DeviceLocal,
            PYRAMID_LEN,
            "environment pyramid",
        )?;
        let partials = device.gpu_malloc::<[f32; 4]>(
            MemoryKind::DeviceLocal,
            u64::from(GROUPS * GROUPS),
            "environment partials",
        )?;
        let mut record =
            device.gpu_malloc::<EnvironmentGpu>(MemoryKind::Upload, 1, "environment record")?;
        record.as_mut_slice().unwrap()[0] = EnvironmentGpu {
            mean: [0.0; 4],
            pyramid: pyramid.gpu,
            map: sampled,
            size: ENVIRONMENT_SIZE,
        };
        let mut pyramid_roots = device.gpu_malloc::<pyramid::Root>(
            MemoryKind::Upload,
            u64::from(LEVELS + 1),
            "environment pyramid roots",
        )?;
        for (level, root) in pyramid_roots.as_mut_slice().unwrap().iter_mut().enumerate() {
            *root = pyramid::Root {
                partials: partials.gpu,
                pyramid: pyramid.gpu,
                environment: record.gpu,
                sky,
                partial_count: GROUPS * GROUPS,
                size: ENVIRONMENT_SIZE,
                level: level as u32,
                _pad: 0,
            };
        }
        Ok(Self {
            bake: pipeline(device, kernels, &bake::MODULE, "main")?,
            mean: pipeline(device, kernels, &pyramid::MODULE, "mean")?,
            build: pipeline(device, kernels, &pyramid::MODULE, "build")?,
            map,
            map_slots,
            storage,
            pyramid,
            partials,
            record,
            pyramid_roots,
            initialised: false,
        })
    }

    /// The bake root for one frame.
    pub(crate) fn root(&self, environment: u64, focus: [f32; 3]) -> bake::Root {
        bake::Root {
            environment,
            partials: self.partials.gpu,
            luminance: self.pyramid.gpu,
            map: self.storage,
            size: ENVIRONMENT_SIZE,
            focus: [focus[0], focus[1], focus[2], 0.0],
        }
    }

    pub(crate) fn record_bake(&mut self, list: &mut CommandList<'_>, root: GpuPtr<bake::Root>) {
        if !self.initialised {
            list.discard(&self.map);
            self.initialised = true;
        }
        list.dispatch(&self.bake, root, [GROUPS, GROUPS, 1]);
    }

    /// The mean, dome statistics and pyramid; after the bake's writes are visible.
    pub(crate) fn record_reduce(&self, list: &mut CommandList<'_>) {
        list.dispatch(&self.mean, self.pyramid_roots.gpu, [1, 1, 1]);
        for level in 1..=LEVELS {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            let side = ENVIRONMENT_SIZE >> level;
            list.dispatch(
                &self.build,
                self.pyramid_roots.gpu.offset(u64::from(level)),
                [side.div_ceil(8), side.div_ceil(8), 1],
            );
        }
    }

    pub(crate) fn environment(&self) -> GpuPtr<EnvironmentGpu> {
        self.record.gpu
    }

    pub(crate) fn free(self, device: &Device, after: TimelinePoint) {
        for slot in self.map_slots {
            device.free_image_slot(slot, after);
        }
    }
}
