//! The scene's world cache: its cells, the resolve that runs in the scene pass, and its
//! statistics.

use orr_scene::SceneRoot;
use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, MemoryKind, Stages,
};

use crate::scene::{FRAME_SLOTS, Ring, WorldCacheSettings, device_local};
use crate::shaders::types::{WorldCacheGpu, WorldCacheStatsGpu};
use crate::shaders::world_cache::resolve;
use crate::{Error, KernelSet, pipeline};

/// Counters the cache keeps on the GPU (`world_cache`'s `COUNTERS`).
const COUNTERS: u64 = 8;

/// Threads per resolve workgroup and workgroups per dispatch row (`world_cache/resolve`).
const RESOLVE_GROUP: u64 = 256;
const RESOLVE_ROW: u64 = 4096;

/// Largest `WorldCacheSettings::capacity_log2`: 2 GiB of cells.
pub const MAX_CAPACITY_LOG2: u32 = 26;

/// The world cache after one frame's resolve.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct WorldCacheStats {
    pub capacity: u32,
    /// Cells holding a key after the resolve.
    pub occupied: u32,
    /// Cells the resolve evicted.
    pub evicted: u32,
    /// Lookups that could end a path, by the views of the frame before.
    pub queries: u32,
    /// Those lookups that found a resolved cell.
    pub hits: u32,
    /// Samples update paths added, in the frame before.
    pub updates: u32,
    /// Samples whose cell's probe was full, so they were dropped.
    pub overflows: u32,
    /// The scale the resolve applied to every cell for a fall of the global light; 1 when it
    /// did not fall.
    pub rescale: f32,
    /// Directional illuminance at the focus plus the dome's irradiance, lux.
    pub light_estimate: f32,
}

pub(crate) struct WorldCache {
    settings: WorldCacheSettings,
    checks: Allocation<u32>,
    accumulated: Allocation<[u32; 4]>,
    resolved: Allocation<[u32; 2]>,
    counters: Allocation<u32>,
    readback: Allocation<WorldCacheStatsGpu>,
    roots: Ring<WorldCacheGpu>,
    resolve_roots: Ring<resolve::Root>,
    clear: bool,
}

pub(crate) struct ResolvePipelines {
    begin: ComputePipeline,
    resolve: ComputePipeline,
    finish: ComputePipeline,
}

impl ResolvePipelines {
    pub(crate) fn new(device: &Device, kernels: &KernelSet) -> Result<Self, Error> {
        Ok(Self {
            begin: pipeline(device, kernels, &resolve::MODULE, "begin")?,
            resolve: pipeline(device, kernels, &resolve::MODULE, "resolve")?,
            finish: pipeline(device, kernels, &resolve::MODULE, "finish")?,
        })
    }
}

impl WorldCache {
    /// An empty cache; the first resolve clears its cells.
    pub(crate) fn new(device: &Device, settings: &WorldCacheSettings) -> Result<Self, Error> {
        if !(3..=MAX_CAPACITY_LOG2).contains(&settings.capacity_log2) {
            return Err(Error::Settings(
                "world cache capacity_log2 must lie in 3..=26",
            ));
        }
        if settings.cell_lifetime > 255 {
            return Err(Error::Settings(
                "world cache cell_lifetime must be at most 255",
            ));
        }
        if settings.update_block == 0 {
            return Err(Error::Settings(
                "world cache update_block must be at least 1",
            ));
        }
        let capacity = 1u64 << settings.capacity_log2;
        Ok(Self {
            settings: *settings,
            checks: device_local(device, capacity, "world cache checks")?,
            accumulated: device_local(device, capacity, "world cache samples")?,
            resolved: device_local(device, capacity, "world cache radiance")?,
            counters: device_local(device, COUNTERS, "world cache counters")?,
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                FRAME_SLOTS as u64,
                "world cache stats readback",
            )?,
            roots: Ring::new(device, "world cache roots")?,
            resolve_roots: Ring::new(device, "world cache resolve roots")?,
            clear: true,
        })
    }

    pub(crate) fn settings(&self) -> &WorldCacheSettings {
        &self.settings
    }

    /// Empties every cell at the next resolve.
    pub(crate) fn clear(&mut self) {
        self.clear = true;
    }

    /// Writes slot `slot`'s cache root for a frame focused on `focus` and records its
    /// resolve.
    pub(crate) fn record(
        &mut self,
        list: &mut CommandList<'_>,
        pipelines: &ResolvePipelines,
        slot: usize,
        scene: GpuPtr<SceneRoot>,
        focus: [f32; 3],
    ) {
        let s = &self.settings;
        let capacity = 1u64 << s.capacity_log2;
        let cache = self.roots.write(
            slot,
            WorldCacheGpu {
                checks: self.checks.gpu,
                accumulated: self.accumulated.gpu,
                resolved: self.resolved.gpu,
                counters: self.counters.gpu,
                focus,
                lod_scale: s.lod_scale,
                capacity_mask: (capacity - 1) as u32,
                base_cell_log2: s.base_cell_log2,
                _pad0: 0,
                _pad1: 0,
            },
        );
        let root = self.resolve_roots.write(
            slot,
            resolve::Root {
                cache,
                scene,
                stats: self.readback.gpu.offset(slot as u64),
                capacity: capacity as u32,
                cell_lifetime: s.cell_lifetime,
                max_temporal_samples: s.max_temporal_samples,
                clear: u32::from(std::mem::take(&mut self.clear)),
                _pad0: 0,
                _pad1: 0,
            },
        );
        let groups = capacity.div_ceil(RESOLVE_GROUP);
        let grid = [
            groups.min(RESOLVE_ROW) as u32,
            groups.div_ceil(RESOLVE_ROW) as u32,
            1,
        ];
        list.dispatch(&pipelines.begin, root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&pipelines.resolve, root, grid);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&pipelines.finish, root, [1, 1, 1]);
    }

    /// The cache as slot `slot`'s frame reads it.
    pub(crate) fn root(&self, slot: usize) -> GpuPtr<WorldCacheGpu> {
        self.roots.at(slot)
    }

    pub(crate) fn stats(&self, slot: usize) -> WorldCacheStats {
        let s = self.readback.as_slice().unwrap()[slot];
        WorldCacheStats {
            capacity: 1 << self.settings.capacity_log2,
            occupied: s.occupied,
            evicted: s.evicted,
            queries: s.queries,
            hits: s.hits,
            updates: s.updates,
            overflows: s.overflows,
            rescale: s.rescale,
            light_estimate: s.light_estimate,
        }
    }
}
