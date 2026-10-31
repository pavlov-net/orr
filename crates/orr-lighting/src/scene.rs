//! Per-scene lighting state and the scene pass that runs before the scene's first view.

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, MemoryKind, Stages,
};
use orr_scene::{SceneDesc, SceneRoot, SkyStats};

use crate::environment::EnvironmentBake;
use crate::shaders::light_sampling::{directional, tiles};
use crate::shaders::types::{
    DirectionalState, EmissiveSelection, EnvironmentGpu, LightSample, LightSetGpu, LightSetStats,
};
use crate::{Error, KernelSet, pipeline};

/// Frames whose scene-pass roots and statistics are kept apart, so the CPU can write one
/// frame's while the GPU still reads earlier ones.
pub const FRAME_SLOTS: usize = 3;

/// Presampled light tiles redrawn each frame, and samples per tile (`light_sampling`'s
/// `TILE_COUNT` and `TILE_SIZE`).
pub const LIGHT_TILES: u32 = 128;
pub const LIGHT_TILE_SIZE: u32 = 1024;

/// Lighting state every view of a scene shares.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SceneLightingSettings {
    pub world_cache: WorldCacheSettings,
    /// Metres from the eye that every view's froxel volume covers; linked environments that
    /// march from it are built with the same value.
    pub near_range: f32,
    /// Emissive clusters the scene's light selection holds per frame. Emitting entries past
    /// it are left out of next-event estimation (counted in [`LightingStats`]); BSDF hits
    /// still see them.
    pub emissive_clusters: u32,
}

impl Default for SceneLightingSettings {
    fn default() -> Self {
        Self {
            world_cache: WorldCacheSettings::default(),
            near_range: 256.0,
            emissive_clusters: 1 << 20,
        }
    }
}

/// The scene's world-space radiance cache.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WorldCacheSettings {
    pub capacity_log2: u32,
    /// Base voxel size as a power of two, metres.
    pub base_cell_log2: i32,
    pub lod_scale: f32,
    /// Frames without an update before a cell is evicted, at most 255.
    pub cell_lifetime: u32,
    pub max_temporal_samples: f32,
    /// Pixels per side covered by one update path.
    pub update_block: u32,
}

impl Default for WorldCacheSettings {
    fn default() -> Self {
        Self {
            capacity_log2: 22,
            base_cell_log2: -3,
            lod_scale: 15.0,
            cell_lifetime: 10,
            max_temporal_samples: 32.0,
            update_block: 4,
        }
    }
}

/// What the scene pass reads for one frame.
#[derive(Clone, Copy, Debug)]
pub struct ScenePass {
    pub frame: u64,
    /// The scene's root after this frame's `update_scenes`.
    pub scene: GpuPtr<SceneRoot>,
    /// The scene focus in render space.
    pub focus: [f32; 3],
}

/// Statistics of one frame, read back after it completes.
#[derive(Clone, Copy, Debug)]
pub struct LightingStats {
    pub frame: u64,
    pub sky: SkyStats,
    pub directional_lights: u32,
    /// Sphere and rect lights with non-zero power.
    pub local_lights: u32,
    /// Emissive entries next-event estimation can pick.
    pub emissive_entries: u32,
    /// Clusters in those entries' cluster CDFs.
    pub emissive_clusters: u32,
    /// Emitting entries left out because `SceneLightingSettings::emissive_clusters` was full.
    pub dropped_entries: u32,
    /// Share of tile samples taken from the environment.
    pub environment_probability: f32,
}

/// Per-root ring of one root per frame slot, in upload memory.
pub(crate) struct Ring<T: 'static>(Allocation<T>);

impl<T: bytemuck::Pod> Ring<T> {
    pub(crate) fn new(device: &Device, name: &str) -> Result<Self, Error> {
        Ok(Self(device.gpu_malloc::<T>(
            MemoryKind::Upload,
            FRAME_SLOTS as u64,
            name,
        )?))
    }

    pub(crate) fn write(&mut self, slot: usize, value: T) -> GpuPtr<T> {
        self.0.as_mut_slice().unwrap()[slot] = value;
        self.0.gpu.offset(slot as u64)
    }
}

fn device_local<T>(device: &Device, count: u64, name: &str) -> Result<Allocation<T>, Error> {
    Ok(device.gpu_malloc(MemoryKind::DeviceLocal, count.max(1), name)?)
}

/// Lighting state of one scene: the baked environment, the directional light states, the
/// light selection and the presampled light tiles.
pub struct SceneLighting {
    device: Device,
    settings: SceneLightingSettings,
    producers: u32,
    environment: EnvironmentBake,
    directional: ComputePipeline,
    select: ComputePipeline,
    clusters: ComputePipeline,
    presample: ComputePipeline,
    states: Allocation<DirectionalState>,
    directional_ids: Allocation<u32>,
    set: Allocation<LightSetGpu>,
    cdf: Allocation<f32>,
    light_probability: Allocation<f32>,
    emissive: Allocation<EmissiveSelection>,
    cluster_cdf: Allocation<f32>,
    tiles: Allocation<LightSample>,
    readback: Allocation<LightSetStats>,
    bake_roots: Ring<crate::shaders::environment::bake::Root>,
    directional_roots: Ring<directional::Root>,
    tile_roots: Ring<tiles::Root>,
    frames: [Option<u64>; FRAME_SLOTS],
}

impl SceneLighting {
    /// Lighting for a scene created with `scene`. `kernels` replaces the crate's kernels that
    /// share a kernel name with one of its modules, as a build linking another environment or
    /// medium produces them.
    pub fn new(
        device: &Device,
        kernels: &KernelSet,
        scene: &SceneDesc,
        settings: &SceneLightingSettings,
    ) -> Result<Self, Error> {
        let lights = u64::from(scene.light_capacity);
        let entries = u64::from(scene.producer_capacity) + u64::from(scene.emissive_capacity);
        Ok(Self {
            device: device.clone(),
            settings: *settings,
            producers: scene.producer_capacity,
            environment: EnvironmentBake::new(device, kernels)?,
            directional: pipeline(device, kernels, &directional::MODULE, "main")?,
            select: pipeline(device, kernels, &tiles::MODULE, "select")?,
            clusters: pipeline(device, kernels, &tiles::MODULE, "clusters")?,
            presample: pipeline(device, kernels, &tiles::MODULE, "presample")?,
            states: device_local(device, lights, "directional states")?,
            directional_ids: device_local(device, lights, "directional ids")?,
            set: device_local(device, 1, "light set")?,
            cdf: device_local(device, lights + entries, "light cdf")?,
            light_probability: device_local(device, lights, "light probabilities")?,
            emissive: device_local(device, entries, "emissive selection")?,
            cluster_cdf: device_local(
                device,
                u64::from(settings.emissive_clusters),
                "emissive cluster cdf",
            )?,
            tiles: device_local(
                device,
                u64::from(LIGHT_TILES * LIGHT_TILE_SIZE),
                "light tiles",
            )?,
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                FRAME_SLOTS as u64,
                "lighting stats readback",
            )?,
            bake_roots: Ring::new(device, "environment bake roots")?,
            directional_roots: Ring::new(device, "directional roots")?,
            tile_roots: Ring::new(device, "light tile roots")?,
            frames: [None; FRAME_SLOTS],
        })
    }

    pub fn settings(&self) -> &SceneLightingSettings {
        &self.settings
    }

    /// A change of `emissive_clusters` reallocates the cluster pool; the old one is freed once
    /// the work submitted so far has completed.
    pub fn set_settings(&mut self, settings: &SceneLightingSettings) -> Result<(), Error> {
        if settings.emissive_clusters != self.settings.emissive_clusters {
            self.cluster_cdf = device_local(
                &self.device,
                u64::from(settings.emissive_clusters),
                "emissive cluster cdf",
            )?;
        }
        self.settings = *settings;
        Ok(())
    }

    /// Records the scene pass after the scene's `update_scenes` in the same or an earlier
    /// list: bakes the environment and builds its pyramid, derives every directional light's
    /// state, builds the light selection and the emissive cluster CDFs, presamples the light
    /// tiles, and writes `SkyStats` and the statistics [`Self::stats`] reads. Its outputs are
    /// visible to every later stage of the list.
    pub fn record(&mut self, list: &mut CommandList<'_>, pass: &ScenePass) {
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        self.frames[slot] = Some(pass.frame);
        let focus = [pass.focus[0], pass.focus[1], pass.focus[2], 0.0];
        let bake_root = self
            .bake_roots
            .write(slot, self.environment.root(pass.scene, pass.focus));
        let directional_root = self.directional_roots.write(
            slot,
            directional::Root {
                scene: pass.scene,
                states: self.states.gpu,
                ids: self.directional_ids.gpu,
                set: self.set.gpu,
                focus,
            },
        );
        let tile_root = self.tile_roots.write(
            slot,
            tiles::Root {
                scene: pass.scene,
                set: self.set.gpu,
                environment: self.environment.environment(),
                directional: self.states.gpu,
                directional_ids: self.directional_ids.gpu,
                cdf: self.cdf.gpu,
                light_probability: self.light_probability.gpu,
                emissive: self.emissive.gpu,
                cluster_cdf: self.cluster_cdf.gpu,
                tiles: self.tiles.gpu,
                stats: self.readback.gpu.offset(slot as u64),
                cluster_capacity: self.settings.emissive_clusters,
                frame: pass.frame as u32,
            },
        );

        list.begin_pass("lighting scene", false);
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.environment.record_bake(list, bake_root);
        list.dispatch(&self.directional, directional_root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        self.environment.record_reduce(list);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.select, tile_root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        if self.producers > 0 {
            list.dispatch(&self.clusters, tile_root, [self.producers, 1, 1]);
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        }
        list.dispatch(
            &self.presample,
            tile_root,
            [LIGHT_TILES * LIGHT_TILE_SIZE / 64, 1, 1],
        );
        list.barrier(
            Stages::COMPUTE,
            Stages::ALL | Stages::HOST,
            Hazards::empty(),
        );
        list.end_pass();
    }

    /// Statistics of `frame`, once its submit has completed; `None` when that frame's slot
    /// holds another frame.
    pub fn stats(&self, frame: u64) -> Option<LightingStats> {
        let slot = (frame % FRAME_SLOTS as u64) as usize;
        (self.frames[slot] == Some(frame)).then(|| {
            let s = self.readback.as_slice().unwrap()[slot];
            LightingStats {
                frame,
                sky: s.sky,
                directional_lights: s.directional_lights,
                local_lights: s.local_lights,
                emissive_entries: s.emissive_entries,
                emissive_clusters: s.emissive_clusters,
                dropped_entries: s.dropped_entries,
                environment_probability: s.environment_probability,
            }
        })
    }

    /// The light set kernels sample: selection, tiles, directional states and the baked
    /// environment, as the last recorded scene pass left them.
    pub fn light_set(&self) -> GpuPtr<LightSetGpu> {
        self.set.gpu
    }

    /// The baked environment, for sampling and miss lookups.
    pub fn environment(&self) -> GpuPtr<EnvironmentGpu> {
        self.environment.environment()
    }

    /// Per directional light, by LightId, its state for the last recorded frame.
    pub fn directional_states(&self) -> GpuPtr<DirectionalState> {
        self.states.gpu
    }
}
