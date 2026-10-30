//! Per-scene lighting state and the scene pass that runs before the scene's first view.

use orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuAddress, GpuPtr, Hazards, MemoryKind,
    Stages, TimelinePoint,
};

use crate::environment::EnvironmentBake;
use crate::shaders::environment::bake;
use crate::shaders::light_sampling::directional;
use crate::shaders::types::{DirectionalLightGpu, DirectionalState, EnvironmentGpu, SkyStats};
use crate::{Error, KernelSet, pipeline};

/// Frames whose scene-pass roots and statistics are kept apart, so the CPU can write one
/// frame's while the GPU still reads earlier ones.
pub const FRAME_SLOTS: usize = 3;

/// Lighting state every view of a scene shares.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SceneLightingSettings {
    pub world_cache: WorldCacheSettings,
    /// Metres from the eye that every view's froxel volume covers; linked environments that
    /// march from it are built with the same value.
    pub near_range: f32,
}

impl Default for SceneLightingSettings {
    fn default() -> Self {
        Self {
            world_cache: WorldCacheSettings::default(),
            near_range: 256.0,
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
    /// The scene focus in render space.
    pub focus: [f32; 3],
    /// The linked environment's data address.
    pub environment: GpuAddress,
    /// The linked medium's data address.
    pub medium: GpuAddress,
    /// The scene's directional lights.
    pub lights: GpuPtr<DirectionalLightGpu>,
    pub light_count: u32,
}

/// Statistics of one frame, read back after it completes.
#[derive(Clone, Copy, Debug)]
pub struct LightingStats {
    pub frame: u64,
    pub sky: SkyStats,
}

/// Per-root ring of one root per frame slot, in upload memory.
struct Ring<T: 'static>(Allocation<T>);

impl<T: bytemuck::Pod> Ring<T> {
    fn new(device: &Device, name: &str) -> Result<Self, Error> {
        Ok(Self(device.gpu_malloc::<T>(
            MemoryKind::Upload,
            FRAME_SLOTS as u64,
            name,
        )?))
    }

    fn write(&mut self, slot: usize, value: T) -> GpuPtr<T> {
        self.0.as_mut_slice().unwrap()[slot] = value;
        self.0.gpu.offset(slot as u64)
    }
}

/// Lighting state of one scene: the baked environment, the directional light states and the
/// sky statistics.
pub struct SceneLighting {
    settings: SceneLightingSettings,
    environment: EnvironmentBake,
    directional: ComputePipeline,
    states: Allocation<DirectionalState>,
    sky: Allocation<SkyStats>,
    readback: Allocation<SkyStats>,
    bake_roots: Ring<bake::Root>,
    directional_roots: Ring<directional::Root>,
    frames: [Option<u64>; FRAME_SLOTS],
}

impl SceneLighting {
    /// `light_capacity` bounds the directional lights a scene pass reads. `kernels` replaces
    /// the crate's kernels that share a kernel name with one of its modules, as a build
    /// linking another environment or medium produces them.
    pub fn new(
        device: &Device,
        kernels: &KernelSet,
        settings: &SceneLightingSettings,
        light_capacity: u32,
    ) -> Result<Self, Error> {
        let sky = device.gpu_malloc::<SkyStats>(MemoryKind::DeviceLocal, 1, "sky stats")?;
        Ok(Self {
            settings: *settings,
            environment: EnvironmentBake::new(device, kernels, sky.gpu)?,
            directional: pipeline(device, kernels, &directional::MODULE, "main")?,
            states: device.gpu_malloc(
                MemoryKind::DeviceLocal,
                u64::from(light_capacity.max(1)),
                "directional states",
            )?,
            sky,
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                FRAME_SLOTS as u64,
                "sky stats readback",
            )?,
            bake_roots: Ring::new(device, "environment bake roots")?,
            directional_roots: Ring::new(device, "directional roots")?,
            frames: [None; FRAME_SLOTS],
        })
    }

    pub fn settings(&self) -> &SceneLightingSettings {
        &self.settings
    }

    pub fn set_settings(&mut self, settings: &SceneLightingSettings) {
        self.settings = *settings;
    }

    /// Records the scene pass: bakes the environment, reduces it to its mean, dome
    /// statistics and importance pyramid, derives every directional light's state, and copies
    /// `SkyStats` for [`Self::stats`]. Its outputs are visible to every later stage of the
    /// list.
    pub fn record(&mut self, list: &mut CommandList<'_>, pass: &ScenePass) {
        assert!(
            u64::from(pass.light_count) <= self.states.len(),
            "more directional lights than the scene's light capacity"
        );
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        self.frames[slot] = Some(pass.frame);
        let bake_root = self
            .bake_roots
            .write(slot, self.environment.root(pass.environment.0, pass.focus));
        let directional_root = self.directional_roots.write(
            slot,
            directional::Root {
                medium: pass.medium.0,
                lights: pass.lights,
                states: self.states.gpu,
                sky: self.sky.gpu,
                count: pass.light_count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                focus: [pass.focus[0], pass.focus[1], pass.focus[2], 0.0],
            },
        );

        list.begin_pass("lighting scene", false);
        self.environment.record_bake(list, bake_root);
        list.dispatch(&self.directional, directional_root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        self.environment.record_reduce(list);
        list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
        list.copy(
            self.readback.gpu.offset(slot as u64).cast(),
            self.sky.gpu.cast(),
            size_of::<SkyStats>() as u64,
        );
        list.barrier(
            Stages::COMPUTE | Stages::TRANSFER,
            Stages::ALL | Stages::HOST,
            Hazards::empty(),
        );
        list.end_pass();
    }

    /// Statistics of `frame`, once its submit has completed; `None` when that frame's slot
    /// holds another frame.
    pub fn stats(&self, frame: u64) -> Option<LightingStats> {
        let slot = (frame % FRAME_SLOTS as u64) as usize;
        (self.frames[slot] == Some(frame)).then(|| LightingStats {
            frame,
            sky: self.readback.as_slice().unwrap()[slot],
        })
    }

    /// The baked environment, for sampling and miss lookups.
    pub fn environment(&self) -> GpuPtr<EnvironmentGpu> {
        self.environment.environment()
    }

    /// Per directional light, its state for the last recorded frame.
    pub fn directional_states(&self) -> GpuPtr<DirectionalState> {
        self.states.gpu
    }

    pub fn sky_stats(&self) -> GpuPtr<SkyStats> {
        self.sky.gpu
    }

    /// Frees the heap slots of the baked map once `after` has completed; the rest is freed
    /// when dropped.
    pub fn free(self, device: &Device, after: TimelinePoint) {
        self.environment.free(device, after);
    }
}
