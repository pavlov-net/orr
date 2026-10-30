//! One atmosphere's GPU state and the LUT passes recorded at the scene compute point.

use orr::abi::{GpuAddress, SamplerIndex, StorageImageIndex, TextureIndex};
use orr::orr_gpu::{
    AddressMode, Allocation, ComputePipeline, Device, Filter, Format, Hazards, MemoryKind,
    SamplerDesc, ShaderStage, SpecData, Stages, Texture, TextureDesc, TextureMemory, TextureUsage,
    TimelinePoint, ViewDesc,
};
use orr::{LightId, Result, SceneCompute};
use orr_shaders::bytemuck::Zeroable;

use crate::block::{self, PASSES};
use crate::shaders::luts::{MODULE, Root};
use crate::shaders::types::{AtmosphereGpu, AtmosphereGroups, FoldGpu};
use crate::{Atmosphere, MAX_EXTRA_VIEWS, MAX_FOLDED_LIGHTS, MAX_SKY_LIGHTS};

/// Transmittance LUT texels: zenith cosine (Bruneton's distance mapping) by altitude.
pub const TRANSMITTANCE_SIZE: [u32; 2] = [256, 128];
/// Multiple-scattering LUT texels: the light's zenith cosine by altitude.
pub const MULTIPLE_SCATTERING_SIZE: [u32; 2] = [64, 32];
/// Sky-view LUT texels: longitude by warped latitude.
pub const SKY_VIEW_SIZE: [u32; 2] = [400, 200];
/// Entries of the ground irradiance table, over the light's zenith cosine.
const GROUND_SIZE: u32 = 64;

/// Sky-view LUTs: the focus view's and up to [`MAX_EXTRA_VIEWS`] more.
const SKY_VIEWS: usize = 1 + MAX_EXTRA_VIEWS as usize;

/// A LUT texture with its sampled and storage views.
struct Lut {
    texture: Texture,
    sampled: TextureIndex,
    storage: StorageImageIndex,
    slots: [u32; 2],
}

impl Lut {
    fn new(device: &Device, size: [u32; 2], layers: u32, name: &str) -> Result<Self> {
        let desc = TextureDesc {
            layers,
            ..TextureDesc::d2(
                Format::Rgba32Float,
                size[0],
                size[1],
                TextureUsage::SAMPLED | TextureUsage::STORAGE,
            )
        };
        let texture = device.create_texture(&desc, TextureMemory::Dedicated, name)?;
        let slots = [device.alloc_image_slot()?, device.alloc_image_slot()?];
        Ok(Self {
            sampled: device.write_sampled(slots[0], &texture, &ViewDesc::default()),
            storage: device.write_storage(slots[1], &texture, &ViewDesc::default()),
            texture,
            slots,
        })
    }
}

struct Pipelines {
    transmittance: ComputePipeline,
    multiple_scattering: ComputePipeline,
    ground_irradiance: ComputePipeline,
    fold: ComputePipeline,
    sky_view: ComputePipeline,
}

/// Workgroups of 8x8 threads over a LUT.
fn tiles(size: [u32; 2]) -> [u32; 3] {
    [size[0].div_ceil(8), size[1].div_ceil(8), 1]
}

fn pipeline(device: &Device, entry: &str) -> Result<ComputePipeline> {
    let stage = ShaderStage::new(&MODULE, entry, SpecData::default())?;
    Ok(device.create_compute_pipeline(&stage, entry)?)
}

/// One atmosphere's GPU state: its LUTs, the fold of candidate lights and the
/// [`AtmosphereGpu`] block that [`AtmosphereState::data`] points at.
///
/// After a [`set`](Self::set), the next [`record`](Self::record) rebuilds the
/// transmittance, multiple-scattering and ground irradiance tables, which depend only on the
/// medium. Every `record` refolds the candidate lights and rebuilds the sky-view LUT at the
/// scene focus altitude, in [`SPECTRAL_GROUPS`](crate::spectrum::SPECTRAL_GROUPS) wavelength
/// groups summed to Rec.2020.
pub struct AtmosphereState {
    desc: Atmosphere,
    medium_changed: bool,
    discarded: bool,
    pipelines: Pipelines,
    /// The block as the next `record` writes it.
    record: AtmosphereGpu,
    block: Allocation<AtmosphereGpu>,
    groups: Allocation<[AtmosphereGroups; PASSES]>,
    _fold: Allocation<FoldGpu>,
    _ground: Allocation<[f32; 4]>,
    transmittance: Lut,
    multiple_scattering: Lut,
    sky_views: Vec<Lut>,
}

fn check(desc: &Atmosphere) {
    assert!(
        (1..=MAX_FOLDED_LIGHTS).contains(&desc.sky_lights),
        "sky_lights is 1..={MAX_FOLDED_LIGHTS}"
    );
    assert!(
        desc.planet_radius > 0.0 && desc.top_radius > desc.planet_radius,
        "the top of the atmosphere is above the planet"
    );
    assert!(desc.up.length() > 0.0, "up is a direction");
    assert!(
        desc.rayleigh.scale_height > 0.0
            && desc.aerosol.scale_height > 0.0
            && desc.ozone.half_width > 0.0,
        "scale heights and the ozone half width are positive"
    );
    if let Some(f) = desc.height_fog {
        assert!(
            f.falloff_height > 0.0,
            "height fog falls off over a positive height"
        );
    }
}

impl AtmosphereState {
    /// Allocates the LUTs, their heap slots and the block, writes `sampler` (a slot the state
    /// keeps) and creates the pipelines on the calling thread. Nothing is allocated after
    /// this.
    pub fn new(device: &Device, desc: &Atmosphere, sampler: SamplerIndex) -> Result<Self> {
        check(desc);
        device.write_sampler(
            sampler,
            &SamplerDesc {
                mag: Filter::Linear,
                min: Filter::Linear,
                mip: Filter::Nearest,
                address: [AddressMode::Repeat; 3],
                ..SamplerDesc::default()
            },
        );
        let layers = PASSES as u32 + 1;
        let transmittance = Lut::new(
            device,
            TRANSMITTANCE_SIZE,
            layers,
            "atmosphere transmittance",
        )?;
        let multiple_scattering = Lut::new(
            device,
            MULTIPLE_SCATTERING_SIZE,
            layers,
            "atmosphere multiple scattering",
        )?;
        let sky_views = vec![Lut::new(device, SKY_VIEW_SIZE, 1, "atmosphere sky view")?];
        let block = device.gpu_malloc(MemoryKind::DeviceLocal, 1, "atmosphere")?;
        let groups = device.gpu_malloc(MemoryKind::DeviceLocal, 1, "atmosphere groups")?;
        let fold = device.gpu_malloc(MemoryKind::DeviceLocal, 1, "atmosphere fold")?;
        let ground = device.gpu_malloc(
            MemoryKind::DeviceLocal,
            u64::from(GROUND_SIZE) * PASSES as u64,
            "atmosphere ground irradiance",
        )?;
        let mut record = AtmosphereGpu::zeroed();
        record.groups = groups.gpu.cast();
        record.fold = fold.gpu;
        record.ground_irradiance = ground.gpu;
        record.group_passes = PASSES as u32;
        record.transmittance = transmittance.sampled;
        record.multiple_scattering = multiple_scattering.sampled;
        record.sampler = sampler;
        record.sky_view = [TextureIndex::NONE; SKY_VIEWS];
        for (i, lut) in sky_views.iter().enumerate() {
            record.sky_view[i] = lut.sampled;
        }
        record.sky_view_count = sky_views.len() as u32;
        record.multiple_scattering_scale = 1.0;
        block::medium(desc, &mut record);
        Ok(Self {
            desc: *desc,
            medium_changed: true,
            discarded: false,
            pipelines: Pipelines {
                transmittance: pipeline(device, "transmittance")?,
                multiple_scattering: pipeline(device, "multiple_scattering")?,
                ground_irradiance: pipeline(device, "ground_irradiance")?,
                fold: pipeline(device, "fold")?,
                sky_view: pipeline(device, "sky_view")?,
            },
            record,
            block,
            groups,
            _fold: fold,
            _ground: ground,
            transmittance,
            multiple_scattering,
            sky_views,
        })
    }

    pub fn desc(&self) -> &Atmosphere {
        &self.desc
    }

    /// Changes the medium; the next [`record`](Self::record) rebuilds every LUT.
    pub fn set(&mut self, desc: &Atmosphere) {
        check(desc);
        self.desc = *desc;
        block::medium(desc, &mut self.record);
        self.medium_changed = true;
    }

    /// The candidate lights, at most [`MAX_SKY_LIGHTS`]: ids in the scene's light table. The
    /// sky folds in the `sky_lights` brightest by top-of-atmosphere luminance among those that
    /// are directional and no more than 18 deg below the focus horizon; ties go to the earlier
    /// candidate.
    pub fn set_lights(&mut self, lights: &[LightId]) {
        assert!(
            lights.len() <= MAX_SKY_LIGHTS,
            "at most {MAX_SKY_LIGHTS} candidate lights"
        );
        let r = &mut self.record;
        r.candidate_count = lights.len() as u32;
        for (slot, id) in r.candidates.iter_mut().zip(lights) {
            *slot = id.0;
        }
    }

    /// Whether the sky-view LUTs add multiple scattering (on by default). Off, they hold single
    /// scattering only and the ground below the horizon gets only the direct light: a view
    /// for debugging and for comparison with single-scattering references.
    pub fn set_multiple_scattering(&mut self, on: bool) {
        self.record.multiple_scattering_scale = if on { 1.0 } else { 0.0 };
    }

    /// The block's address, for `SceneWriter::set_environment` and `set_medium`.
    pub fn data(&self) -> GpuAddress {
        self.block.gpu.address()
    }

    /// Records the atmosphere's passes for the scene `cx` hands over: the medium's tables
    /// after a [`set`](Self::set), then the fold and the sky-view LUT at the scene focus
    /// altitude (0 without a focus). Its writes are visible to every later stage of the list.
    pub fn record(&mut self, cx: &mut SceneCompute<'_, '_>) -> Result<()> {
        let d = &self.desc;
        let altitude = cx.focus.map_or(0.0, |(focus, _)| {
            (focus - d.planet_top).dot(d.up.normalize().as_dvec3())
        });
        let ceiling = f64::from(d.top_radius - d.planet_radius) - 1.0;
        let r = &mut self.record;
        r.scene = cx.root.address().0;
        r.sky_view_altitude[0] = altitude.clamp(0.0, ceiling) as f32;

        let block = cx.put(&self.record)?;
        cx.cmd.copy(
            self.block.gpu.cast(),
            block.cast(),
            size_of::<AtmosphereGpu>() as u64,
        );
        if self.medium_changed {
            let groups = cx.put(&block::groups(&self.desc))?;
            cx.cmd.copy(
                self.groups.gpu.cast(),
                groups.cast(),
                size_of::<[AtmosphereGroups; PASSES]>() as u64,
            );
        }
        if !self.discarded {
            cx.cmd.discard(&self.transmittance.texture);
            cx.cmd.discard(&self.multiple_scattering.texture);
            for lut in &self.sky_views {
                cx.cmd.discard(&lut.texture);
            }
            self.discarded = true;
        }
        cx.cmd
            .barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());

        let root = |output: StorageImageIndex, view: u32| Root {
            atmosphere: self.block.gpu,
            output,
            view,
        };
        if self.medium_changed {
            cx.cmd.begin_pass("atmosphere medium", false);
            let t = cx.put_root(&root(self.transmittance.storage, 0))?;
            cx.cmd
                .dispatch(&self.pipelines.transmittance, t, tiles(TRANSMITTANCE_SIZE));
            cx.cmd
                .barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            let ms = cx.put_root(&root(self.multiple_scattering.storage, 0))?;
            let [w, h] = MULTIPLE_SCATTERING_SIZE;
            cx.cmd
                .dispatch(&self.pipelines.multiple_scattering, ms, [w, h, 1]);
            cx.cmd
                .barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            let g = cx.put_root(&root(StorageImageIndex::NONE, 0))?;
            cx.cmd
                .dispatch(&self.pipelines.ground_irradiance, g, [GROUND_SIZE, 1, 1]);
            cx.cmd
                .barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            cx.cmd.end_pass();
            self.medium_changed = false;
        }

        cx.cmd.begin_pass("atmosphere sky view", false);
        let f = cx.put_root(&root(StorageImageIndex::NONE, 0))?;
        cx.cmd.dispatch(&self.pipelines.fold, f, [1, 1, 1]);
        cx.cmd
            .barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        for (i, lut) in self.sky_views.iter().enumerate() {
            let s = cx.put_root(&root(lut.storage, i as u32))?;
            cx.cmd
                .dispatch(&self.pipelines.sky_view, s, tiles(SKY_VIEW_SIZE));
        }
        cx.cmd
            .barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        cx.cmd.end_pass();
        Ok(())
    }

    /// Frees the LUTs' heap slots once `after` has completed; the rest is freed when
    /// dropped.
    pub fn free(self, device: &Device, after: TimelinePoint) {
        let luts = [self.transmittance, self.multiple_scattering]
            .into_iter()
            .chain(self.sky_views);
        for lut in luts {
            for slot in lut.slots {
                device.free_image_slot(slot, after);
            }
        }
    }
}
