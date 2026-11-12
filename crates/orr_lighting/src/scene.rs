//! Per-scene lighting state and the scene pass that runs before the scene's first view.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuAddress, GpuPtr, Hazards, MemoryKind,
    ROOT_ALIGN, Stages,
};
use orr_scene::{LightTables, SceneRoot, SkyStats};

use crate::caustics::{CausticSettings, CausticStats, Caustics, PerEntry, feedback_ids};
use crate::environment::EnvironmentBake;
use crate::shaders::light_sampling::{directional, tiles};
use crate::shaders::types::{
    DirectionalState, EmissiveSelection, EnvironmentGpu, LightSample, LightSetGpu, LightSetStats,
    WorldCacheGpu,
};
use crate::world_cache::{ResolvePipelines, WorldCache, WorldCacheStats};
use crate::{Error, KernelSet, pipeline};

/// Frames whose scene-pass roots and statistics are kept apart, so the CPU can write one
/// frame's while the GPU still reads earlier ones.
pub(crate) const FRAME_SLOTS: usize = 3;

/// Presampled light tiles redrawn each frame, and samples per tile (`light_sampling`'s
/// `TILE_COUNT` and `TILE_SIZE`).
pub const LIGHT_TILES: u32 = 128;
pub const LIGHT_TILE_SIZE: u32 = 1024;

/// Lighting state every view of a scene shares.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SceneLightingSettings {
    pub world_cache: WorldCacheSettings,
    pub caustics: CausticSettings,
    /// Meters from the eye that every view's froxel volume covers; linked environments that
    /// march from it are built with the same value.
    pub near_range: f32,
    /// Producer entries' emitted clusters the scene's light selection holds per frame;
    /// instance entries read their copies' cooked sets and gathered lists in place. Emitting
    /// producer entries past it are left out of next-event estimation (counted in
    /// [`LightingStats`]); BSDF hits still see them, at full weight.
    pub emissive_clusters: u32,
    /// Share of the light selection placed by where the views' pixels found their light in
    /// earlier frames, the rest by emitted power; in [0, 1). 0 selects by power alone.
    pub feedback_share: f32,
}

impl Default for SceneLightingSettings {
    fn default() -> Self {
        Self {
            world_cache: WorldCacheSettings::default(),
            caustics: CausticSettings::default(),
            near_range: 256.0,
            emissive_clusters: 1 << 20,
            feedback_share: 0.5,
        }
    }
}

/// The scene's world-space radiance cache: a hash grid of 2^`capacity_log2` cells of 28 B
/// on voxels whose edge doubles with each level, from 2^`base_cell_log2` m within
/// `lod_scale` m of the scene focus.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WorldCacheSettings {
    /// 3 to [`crate::MAX_CAPACITY_LOG2`].
    pub capacity_log2: u32,
    /// Base voxel size as a power of two, meters.
    pub base_cell_log2: i32,
    /// The level of a point is log2 of its distance to the focus over this, in m.
    pub lod_scale: f32,
    /// Frames without an update before a cell is evicted, at most 255.
    pub cell_lifetime: u32,
    /// Samples of history a cell keeps; cells whose luminance is changing keep 8.
    pub max_temporal_samples: f32,
    /// Pixels per side covered by one update path in the scene's first view of a frame;
    /// later views of the scene cover twice as many per side.
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
    /// What `scene` says the scene's light tables hold: `RasterView::light_tables` or
    /// `SceneCompute::light_tables` of the frame.
    pub tables: LightTables,
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
    /// Clusters in those entries' lists.
    pub emissive_clusters: u32,
    /// Emitting entries left out because `SceneLightingSettings::emissive_clusters` was full.
    pub dropped_entries: u32,
    /// Share of tile samples taken from the environment.
    pub environment_probability: f32,
    /// Pixels of the frame before whose light feedback named a light.
    pub feedback_pixels: u32,
    /// Shadow rays of the frame before that met more non-opaque candidates (dielectric and
    /// presence layers, particles, medium proxies) than the shadow product multiplies; each
    /// kept the product so far.
    pub shadow_overflows: u32,
    /// Walks of reference views in the frame before that reached their step cap; each ended its
    /// path.
    pub capped_walks: u32,
    /// Deposits of reference views' photon maps in the frame before that found their map full;
    /// each dropped its photon.
    pub photon_overflows: u32,
    /// Caster entries whose material could be a plane stack face. With none, directional
    /// lights' rays look for no plane stack.
    pub stack_entries: u32,
    pub world_cache: WorldCacheStats,
    pub caustics: CausticStats,
}

/// `N` kernel roots in upload memory, [`ROOT_ALIGN`] bytes apart.
pub(crate) struct Roots<T: 'static, const N: usize>(Allocation<u8>, PhantomData<fn() -> T>);

/// One root per frame slot.
pub(crate) type Ring<T> = Roots<T, FRAME_SLOTS>;

impl<T: bytemuck::Pod, const N: usize> Roots<T, N> {
    pub(crate) fn new(device: &Device, name: &str) -> Result<Self, Error> {
        const { assert!(size_of::<T>() as u64 <= ROOT_ALIGN) };
        Ok(Self(
            device.gpu_malloc::<u8>(MemoryKind::Upload, N as u64 * ROOT_ALIGN, name)?,
            PhantomData,
        ))
    }

    pub(crate) fn write(&mut self, index: usize, value: T) -> GpuPtr<T> {
        let at = index * ROOT_ALIGN as usize;
        self.0.as_mut_slice().unwrap()[at..at + size_of::<T>()]
            .copy_from_slice(bytemuck::bytes_of(&value));
        self.at(index)
    }

    pub(crate) fn at(&self, index: usize) -> GpuPtr<T> {
        assert!(index < N);
        let address = self.0.gpu().address().0 + index as u64 * ROOT_ALIGN;
        // SAFETY: root `index` has room for a `T` at a multiple of ROOT_ALIGN, which `write`
        // fills.
        unsafe { GpuPtr::from_address(GpuAddress(address)) }
    }
}

fn check_settings(settings: &SceneLightingSettings) -> Result<(), Error> {
    if !(0.0..1.0).contains(&settings.feedback_share) {
        return Err(Error::Settings("feedback_share must lie in [0, 1)"));
    }
    crate::caustics::check_settings(&settings.caustics)
}

pub(crate) fn device_local<T>(
    device: &Device,
    count: u64,
    name: &str,
) -> Result<Allocation<T>, Error> {
    Ok(device.gpu_malloc(MemoryKind::DeviceLocal, count.max(1), name)?)
}

/// What the scene pass holds per light, per emissive list entry and per feedback id, for light
/// tables `tables`.
struct PerTable {
    tables: LightTables,
    states: Allocation<DirectionalState>,
    directional_ids: Allocation<u32>,
    cdf: Allocation<f32>,
    light_probability: Allocation<f32>,
    emissive: Allocation<EmissiveSelection>,
    feedback_counts: Allocation<u32>,
    feedback_weights: Allocation<f32>,
}

impl PerTable {
    fn new(device: &Device, tables: LightTables) -> Result<Self, Error> {
        let n = u64::from(tables.lights);
        let entries = u64::from(tables.entries);
        let ids = feedback_ids(tables);
        Ok(Self {
            tables,
            states: device_local(device, n, "directional states")?,
            directional_ids: device_local(device, n, "directional ids")?,
            cdf: device_local(device, n + entries, "light cdf")?,
            light_probability: device_local(device, n, "light probabilities")?,
            emissive: device_local(device, entries, "emissive selection")?,
            feedback_counts: device_local(device, ids, "light feedback counts")?,
            feedback_weights: device_local(device, ids, "light feedback weights")?,
        })
    }
}

/// Lighting state of one scene: the baked environment, the directional light states, the
/// light selection, the presampled light tiles, the caustics' photon source and grid, and the
/// world cache.
pub struct SceneLighting {
    device: Device,
    settings: SceneLightingSettings,
    environment: EnvironmentBake,
    directional: ComputePipeline,
    select: ComputePipeline,
    clusters: ComputePipeline,
    presample: ComputePipeline,
    /// For the light tables of the last scene pass recorded.
    per: PerTable,
    /// What grown light tables replaced, with the caustics', by the frame whose scene pass
    /// replaced it: that pass reads its feedback and its casters' yields, and the frames in
    /// flight before it read all of it.
    replaced: Vec<(u64, PerTable, PerEntry)>,
    set: Allocation<LightSetGpu>,
    cluster_cdf: Allocation<u32>,
    tiles: Allocation<LightSample>,
    shadow_overflows: Allocation<u32>,
    reference_counts: Allocation<u32>,
    readback: Allocation<LightSetStats>,
    bake_roots: Ring<crate::shaders::environment::bake::Root>,
    directional_roots: Ring<directional::Root>,
    tile_roots: Ring<tiles::Root>,
    cache: WorldCache,
    cache_pipelines: ResolvePipelines,
    caustics: Caustics,
    /// One past the frame whose first view has recorded its update paths.
    first_view: AtomicU64,
    /// One past the frame whose first real-time view has traced its photons.
    first_photons: AtomicU64,
    frames: [Option<u64>; FRAME_SLOTS],
    /// Calls of [`Self::reset_history`] so far.
    resets: AtomicU64,
    /// `resets` as the last [`Self::record`] found it.
    resets_applied: u64,
    clear_feedback: bool,
}

impl SceneLighting {
    /// Lighting for a scene; what it holds per light and per emissive list entry follows the
    /// scene's light tables from pass to pass. `kernels` replaces the crate's kernels that
    /// share a kernel name with one of its modules, as a build linking another environment or
    /// medium produces them.
    pub fn new(
        device: &Device,
        kernels: &KernelSet,
        settings: &SceneLightingSettings,
    ) -> Result<Self, Error> {
        check_settings(settings)?;
        Ok(Self {
            device: device.clone(),
            settings: *settings,
            environment: EnvironmentBake::new(device, kernels)?,
            directional: pipeline(device, kernels, &directional::MODULE, "main")?,
            select: pipeline(device, kernels, &tiles::MODULE, "select")?,
            clusters: pipeline(device, kernels, &tiles::MODULE, "clusters")?,
            presample: pipeline(device, kernels, &tiles::MODULE, "presample")?,
            per: PerTable::new(device, LightTables::default())?,
            replaced: Vec::new(),
            set: device_local(device, 1, "light set")?,
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
            shadow_overflows: device_local(device, 1, "shadow overflows")?,
            reference_counts: device_local(device, 2, "reference counts")?,
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                FRAME_SLOTS as u64,
                "lighting stats readback",
            )?,
            bake_roots: Ring::new(device, "environment bake roots")?,
            directional_roots: Ring::new(device, "directional roots")?,
            tile_roots: Ring::new(device, "light tile roots")?,
            cache: WorldCache::new(device, &settings.world_cache)?,
            cache_pipelines: ResolvePipelines::new(device, kernels)?,
            caustics: Caustics::new(device, kernels, &settings.caustics)?,
            first_view: AtomicU64::new(0),
            first_photons: AtomicU64::new(0),
            frames: [None; FRAME_SLOTS],
            resets: AtomicU64::new(0),
            resets_applied: 0,
            clear_feedback: true,
        })
    }

    pub fn settings(&self) -> &SceneLightingSettings {
        &self.settings
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// A change of `emissive_clusters` reallocates the producer cluster pool, and a change of
    /// `world_cache` reallocates and clears the cache. The caustics take theirs as
    /// [`CausticSettings`] describes. Old buffers are freed once the work submitted so far has
    /// completed.
    pub fn set_settings(&mut self, settings: &SceneLightingSettings) -> Result<(), Error> {
        check_settings(settings)?;
        if settings.world_cache != *self.cache.settings() {
            self.cache = WorldCache::new(&self.device, &settings.world_cache)?;
        }
        self.caustics
            .set_settings(&self.device, &settings.caustics)?;
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
    /// state, builds the light selection from emitted power and the light feedback of the
    /// views of earlier frames, builds the emissive cluster CDFs, presamples the light tiles,
    /// writes `SkyStats`, resolves the world cache at the focus, and writes the statistics
    /// [`Self::stats`] reads. Its outputs are visible to every later stage of the list. A pass
    /// whose light tables have grown since the last makes what the scene holds per light and
    /// per emissive list entry anew, and takes the light feedback and the casters' yields so
    /// far by the ids the smaller tables gave. Fails without device memory for that, recording
    /// nothing.
    pub fn record(&mut self, list: &mut CommandList<'_>, pass: &ScenePass) -> Result<(), Error> {
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        // A frame's slot is written again `FRAME_SLOTS` frames on, when its work has completed.
        self.replaced
            .retain(|r| pass.frame < r.0 + FRAME_SLOTS as u64);
        // What is held follows the lights and the emissive list; the producer table's entries
        // only say how far the per-producer kernels run.
        let held = self.per.tables;
        let grown = (pass.tables.lights, pass.tables.entries) != (held.lights, held.entries);
        if grown {
            let next = PerTable::new(&self.device, pass.tables)?;
            let casters = self.caustics.room(&self.device, pass.tables)?;
            let before = std::mem::replace(&mut self.per, next);
            self.replaced
                .push((pass.frame, before, self.caustics.follow(casters)));
        }
        self.per.tables.producers = pass.tables.producers;
        self.caustics.set_producers(pass.tables.producers);
        let now = &self.per;
        // The frames before left their feedback in what this pass's tables replaced.
        let replaced = self.replaced.last().filter(|_| grown);
        let before = replaced.map_or(now, |r| &r.1);
        self.frames[slot] = Some(pass.frame);
        let resets = self.resets();
        if self.resets_applied != resets {
            self.resets_applied = resets;
            self.cache.clear();
            self.caustics.clear();
            self.clear_feedback = true;
        }
        let focus = [pass.focus[0], pass.focus[1], pass.focus[2], 0.0];
        let bake_root = self
            .bake_roots
            .write(slot, self.environment.root(pass.scene, pass.focus));
        let directional_root = self.directional_roots.write(
            slot,
            directional::Root {
                scene: pass.scene,
                states: now.states.gpu(),
                ids: now.directional_ids.gpu(),
                set: self.set.gpu(),
                focus,
            },
        );
        let tile_root = self.tile_roots.write(
            slot,
            tiles::Root {
                scene: pass.scene,
                set: self.set.gpu(),
                environment: self.environment.environment(),
                directional: now.states.gpu(),
                directional_ids: now.directional_ids.gpu(),
                cdf: now.cdf.gpu(),
                light_probability: now.light_probability.gpu(),
                emissive: now.emissive.gpu(),
                cluster_cdf: self.cluster_cdf.gpu(),
                tiles: self.tiles.gpu(),
                feedback_counts: now.feedback_counts.gpu(),
                feedback_weights: now.feedback_weights.gpu(),
                feedback_counts_before: before.feedback_counts.gpu(),
                feedback_weights_before: before.feedback_weights.gpu(),
                shadow_overflows: self.shadow_overflows.gpu(),
                reference_counts: self.reference_counts.gpu(),
                stats: self.readback.gpu().offset(slot as u64),
                cluster_capacity: self.settings.emissive_clusters,
                frame: pass.frame as u32,
                feedback_share: self.settings.feedback_share,
                clear_feedback: u32::from(std::mem::take(&mut self.clear_feedback)),
                lights_before: before.tables.lights,
                entries_before: before.tables.entries,
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
        if now.tables.producers > 0 {
            list.dispatch(&self.clusters, tile_root, [now.tables.producers, 1, 1]);
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        }
        list.dispatch(
            &self.presample,
            tile_root,
            [LIGHT_TILES * LIGHT_TILE_SIZE / 64, 1, 1],
        );
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        self.caustics.record(
            list,
            slot,
            pass.scene,
            self.set.gpu(),
            pass.focus,
            replaced.map(|r| &r.2),
        );
        self.cache
            .record(list, &self.cache_pipelines, slot, pass.scene, pass.focus);
        list.barrier(
            Stages::COMPUTE,
            Stages::ALL | Stages::HOST,
            Hazards::empty(),
        );
        list.end_pass();
        Ok(())
    }

    /// A lighting discontinuity, such as a time skip: the next [`Self::record`] clears the
    /// world cache and the light feedback, and every view of the scene resets its history at its
    /// next [`crate::LightingView::begin`], however many frames later. A camera that moves sets
    /// `camera_cut` on the view instead; either resets the view's history.
    pub fn reset_history(&self) {
        self.resets.fetch_add(1, Ordering::Relaxed);
    }

    /// How many times [`Self::reset_history`] has been called. A view that remembers the count
    /// sees a reset as a change of it, however long ago the reset came.
    pub fn resets(&self) -> u64 {
        self.resets.load(Ordering::Relaxed)
    }

    /// Whether this is the first view of the scene in `frame` to trace update paths.
    pub(crate) fn first_view(&self, frame: u64) -> bool {
        self.first_view.swap(frame + 1, Ordering::Relaxed) != frame + 1
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
                feedback_pixels: s.feedback_pixels,
                shadow_overflows: s.shadow_overflows,
                capped_walks: s.capped_walks,
                photon_overflows: s.photon_overflows,
                stack_entries: s.stack_entries,
                world_cache: self.cache.stats(slot),
                caustics: self.caustics.stats(slot),
            }
        })
    }

    /// The light set kernels sample: selection, tiles, directional states and the baked
    /// environment, as the last recorded scene pass left them.
    pub fn light_set(&self) -> GpuPtr<LightSetGpu> {
        self.set.gpu()
    }

    /// The baked environment, for sampling and miss lookups.
    pub fn environment(&self) -> GpuPtr<EnvironmentGpu> {
        self.environment.environment()
    }

    /// The world cache as kernels of `frame` read it, after that frame's [`Self::record`]:
    /// `world_cache`'s `cache_lookup` and `world_cache_irradiance` take it.
    pub fn world_cache(&self, frame: u64) -> GpuPtr<WorldCacheGpu> {
        self.cache.root((frame % FRAME_SLOTS as u64) as usize)
    }

    pub(crate) fn world_cache_settings(&self) -> &WorldCacheSettings {
        self.cache.settings()
    }

    pub(crate) fn caustics(&self) -> &Caustics {
        &self.caustics
    }

    /// Whether this is the first real-time view of the scene in `frame` to trace its photons.
    pub(crate) fn first_photons(&self, frame: u64) -> bool {
        self.first_photons.swap(frame + 1, Ordering::Relaxed) != frame + 1
    }

    /// Per directional light, by LightId, its state for the last recorded frame.
    pub fn directional_states(&self) -> GpuPtr<DirectionalState> {
        self.per.states.gpu()
    }

    /// The light tables of the last recorded scene pass, which give the light set's feedback
    /// ids.
    pub(crate) fn light_tables(&self) -> LightTables {
        self.per.tables
    }
}
