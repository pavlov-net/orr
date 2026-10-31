//! One atmosphere's GPU state and the LUT passes recorded at the scene compute point.

use glam::DVec3;
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

/// An eye this far from the focus altitude gets a sky-view LUT of its own, m.
const VIEW_ENTER: f32 = 25.0;
/// An extra LUT this far from the focus altitude is read alone; nearer, it is blended with
/// the focus LUT, rising from nothing at [`VIEW_ENTER`], m.
const VIEW_FULL: f32 = 50.0;
/// Eyes this close to an extra LUT's altitude share it, m.
const VIEW_SHARE: f32 = 25.0;

/// The altitudes of the extra sky-view LUTs for eyes at altitudes `eyes`, given the focus
/// altitude; returns how many of `out` it filled. Eyes more than [`VIEW_ENTER`] from the
/// focus are placed farthest first; an eye within [`VIEW_SHARE`] of a LUT already placed
/// shares it.
fn assign_views(focus: f32, eyes: impl Iterator<Item = f32> + Clone, out: &mut [f32]) -> usize {
    let mut count = 0;
    while count < out.len() {
        let placed = &out[..count];
        let Some(h) = eyes
            .clone()
            .filter(|h| {
                (h - focus).abs() > VIEW_ENTER && placed.iter().all(|a| (a - h).abs() > VIEW_SHARE)
            })
            .min_by(|a, b| (b - focus).abs().total_cmp(&(a - focus).abs()))
        else {
            break;
        };
        out[count] = h;
        count += 1;
    }
    count
}

/// How much of an extra LUT at altitude `h` a view reads, the rest coming from the focus
/// LUT: 0 at [`VIEW_ENTER`] from the focus, rising linearly to 1 at [`VIEW_FULL`], so the
/// sky changes smoothly as an eye moves through the threshold.
fn view_weight(focus: f32, h: f32) -> f32 {
    (((h - focus).abs() - VIEW_ENTER) / (VIEW_FULL - VIEW_ENTER)).clamp(0.0, 1.0)
}

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
/// medium. Every `record` refolds the candidate lights and rebuilds the sky-view LUTs, in
/// [`SPECTRAL_GROUPS`](crate::spectrum::SPECTRAL_GROUPS) wavelength groups summed to
/// Rec.2020: one at the scene focus altitude and one for each other view far enough above or
/// below it (see [`record`](Self::record)).
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
    /// this. `extra_views`, at most [`MAX_EXTRA_VIEWS`], is how many sky-view LUTs it
    /// holds for views other than the focus view.
    pub fn new(
        device: &Device,
        desc: &Atmosphere,
        sampler: SamplerIndex,
        extra_views: u32,
    ) -> Result<Self> {
        check(desc);
        assert!(
            extra_views <= MAX_EXTRA_VIEWS,
            "at most {MAX_EXTRA_VIEWS} extra views"
        );
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
        let sky_views = (0..=extra_views)
            .map(|_| Lut::new(device, SKY_VIEW_SIZE, 1, "atmosphere sky view"))
            .collect::<Result<Vec<_>>>()?;
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
        record.sky_view_count = 1;
        record.sky_view_weight[0] = 1.0;
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
    /// candidate. A `Textured` light with zero illuminance counts with its disc integral.
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
    /// after a [`set`](Self::set), then the fold and the sky-view LUTs. Its writes are visible
    /// to every later stage of the list.
    ///
    /// `eyes` are the world positions of the scene's other views this frame. The focus LUT
    /// sits at the scene focus altitude (0 without a focus). An eye more than 25 m above or
    /// below it gets an extra LUT at its own altitude, farthest eyes first; eyes within 25 m
    /// of an extra LUT's altitude share it. Every view reads the LUT nearest its altitude, so
    /// eyes left over once the extra LUTs run out read the nearest one. An extra LUT less
    /// than 50 m from the focus altitude is read blended with the focus LUT, from none of it
    /// at 25 m to all of it at 50 m, so an eye crossing 25 m sees no step.
    pub fn record(&mut self, cx: &mut SceneCompute<'_, '_>, eyes: &[DVec3]) -> Result<()> {
        let d = &self.desc;
        let up = d.up.normalize().as_dvec3();
        let ceiling = f64::from(d.top_radius - d.planet_radius) - 1.0;
        let altitude = |p: DVec3| ((p - d.planet_top).dot(up)).clamp(0.0, ceiling) as f32;
        let focus = cx.focus.map_or(0.0, |(focus, _)| altitude(focus));
        let r = &mut self.record;
        let views = 1 + assign_views(
            focus,
            eyes.iter().map(|&e| altitude(e)),
            &mut r.sky_view_altitude[1..self.sky_views.len()],
        );
        r.scene = cx.root.address().0;
        r.sky_view_altitude[0] = focus;
        r.sky_view_count = views as u32;
        for (w, &h) in r.sky_view_weight[..views]
            .iter_mut()
            .zip(&r.sky_view_altitude)
        {
            *w = view_weight(focus, h);
        }

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
            cx.cmd.end_pass();
            self.medium_changed = false;
        }

        cx.cmd.begin_pass("atmosphere sky view", false);
        let f = cx.put_root(&root(StorageImageIndex::NONE, 0))?;
        cx.cmd.dispatch(&self.pipelines.fold, f, [1, 1, 1]);
        cx.cmd
            .barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        for (i, lut) in self.sky_views[..views].iter().enumerate() {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assign(focus: f32, eyes: &[f32], capacity: usize) -> Vec<f32> {
        let mut out = vec![0.0; capacity];
        let n = assign_views(focus, eyes.iter().copied(), &mut out);
        out.truncate(n);
        out
    }

    #[test]
    fn thresholds() {
        assert!(assign(100.0, &[124.0, 76.0, 100.0], 4).is_empty());
        assert_eq!(assign(100.0, &[126.0], 4), [126.0]);
        assert_eq!(assign(100.0, &[74.0], 4), [74.0]);
    }

    #[test]
    fn sharing_and_overflow() {
        // Within 25 m of a placed LUT shares it; the farthest eye places it.
        assert_eq!(assign(0.0, &[60.0, 80.0, 100.0], 4), [100.0, 60.0]);
        assert_eq!(assign(0.0, &[-60.0, 60.0], 4), [-60.0, 60.0]);
        // More eyes than LUTs: the farthest get them, the rest read the nearest.
        assert_eq!(
            assign(0.0, &[100.0, 1000.0, 400.0, 10_000.0], 2),
            [10_000.0, 1000.0]
        );
        assert!(assign(0.0, &[500.0], 0).is_empty());
    }
}
