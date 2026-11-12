//! Caustics: what photons start from each frame, built in the scene pass; the scene's
//! real-time grid of what they deposit and its resolve; and a reference view's photon map.

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, MemoryKind, Stages,
};
use orr_scene::{LightTables, SceneRoot};

use crate::scene::{FRAME_SLOTS, Retired, Ring, device_local};
use crate::shaders::caustics::{casters, resolve};
use crate::shaders::types::{
    CausticStatsGpu, CausticsGpu, EmissiveSelection, EyeInterior, LightSetGpu, PhotonGpu,
    PhotonSourceGpu,
};
use crate::{Error, KernelSet, pipeline};

/// Counters the caustics keep on the GPU (`caustics`' `CAUSTIC_COUNTERS`).
const COUNTERS: u64 = 8;

/// `caustics`' `CAUSTICS_GRID` and `CAUSTICS_MAP`.
const CAUSTICS_GRID: u32 = 0;
const CAUSTICS_MAP: u32 = 1;

/// Threads per resolve workgroup and workgroups per dispatch row (`caustics/resolve`).
const RESOLVE_GROUP: u64 = 256;
const RESOLVE_ROW: u64 = 4096;

/// Largest [`CausticSettings::capacity_log2`]: 768 MiB of cells.
pub const MAX_CAUSTIC_CAPACITY_LOG2: u32 = 24;

/// Most photons a frame starts: rows of 4096 in one dispatch.
pub const MAX_PHOTONS: u32 = 4096 * 65535;

/// Caustics a scene's views share: photons each frame from its lights toward its caster
/// surfaces (smooth dielectrics and metals the cook ranks as casters, and producer clusters of
/// casting materials), deposited in a hash grid of 2^`capacity_log2` cells of 48 B on voxels
/// whose edge doubles with each level, from 2^`base_cell_log2` m within `lod_scale` m of the
/// scene focus; while the grid runs full the levels start coarser ([`CausticStats::finest`]).
/// Each cell keeps a running mean of the frames' irradiance.
///
/// A scene holds the grid while it may hold a caster ([`orr_scene::SceneFeatures::CASTERS`])
/// and `photons` is above 0. The scene pass that first finds both makes it, empty. The pass that
/// finds either gone gives it up with what its cells held: nothing deposits any more, so the
/// light they would lose over `cell_lifetime` frames leaves at once, as in a scene that never
/// had a caster.
///
/// Through [`crate::SceneLighting::set_settings`], `photons`, `max_temporal_samples`,
/// `cell_lifetime` and `jitter` take effect at the next frame and keep the grid. A new
/// `capacity_log2` reallocates it, and that or a new `base_cell_log2` or `lod_scale` clears it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CausticSettings {
    /// Photons each frame, at most [`MAX_PHOTONS`]; 0 turns caustics off, which leaves the
    /// light they carry to BSDF rays.
    pub photons: u32,
    /// 3 to [`MAX_CAUSTIC_CAPACITY_LOG2`].
    pub capacity_log2: u32,
    /// Base voxel size as a power of two, meters.
    pub base_cell_log2: i32,
    /// The level of a point is log2 of its distance to the focus over this, in m, at least
    /// twenty-four base cells, so every photon in a cell a point reads deposited there; a
    /// caster's share of the photons falls with the square of its distance from the focus over
    /// it.
    pub lod_scale: f32,
    /// Frames without a deposit before a cell is evicted, at most 255.
    pub cell_lifetime: u32,
    /// Frames of history a cell keeps; cells whose light is changing keep 8.
    pub max_temporal_samples: f32,
    /// The share of a cell's edge each read's point jitters across, in [0, 1]: 1 blends
    /// neighboring cells over frames, 0 reads the cell holding the point.
    pub jitter: f32,
}

impl Default for CausticSettings {
    fn default() -> Self {
        Self {
            photons: 1 << 18,
            capacity_log2: 20,
            base_cell_log2: -8,
            lod_scale: 2.0,
            cell_lifetime: 8,
            max_temporal_samples: 16.0,
            jitter: 1.0,
        }
    }
}

/// The caustics after one frame's resolve.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct CausticStats {
    /// Photons started from caster entries.
    pub started: u32,
    /// Photons that deposited.
    pub deposits: u32,
    /// Deposits whose cell's probe was full, so they were dropped.
    pub overflows: u32,
    /// Cells holding a key after the resolve.
    pub occupied: u32,
    pub evicted: u32,
    /// Flux per fixed-point step of the frame's deposits.
    pub quantum: f32,
    /// The finest level the frame's reads and deposits took, whatever level a point's distance
    /// gives it: one coarser while more than half the grid is occupied, one finer again below
    /// 8%, each move held for longer than a cell's lifetime.
    pub finest: u32,
}

pub(crate) fn check_settings(s: &CausticSettings) -> Result<(), Error> {
    if !(3..=MAX_CAUSTIC_CAPACITY_LOG2).contains(&s.capacity_log2) {
        return Err(Error::Settings("caustics capacity_log2 must lie in 3..=24"));
    }
    if s.photons > MAX_PHOTONS {
        return Err(Error::Settings(
            "caustics photons must be at most MAX_PHOTONS",
        ));
    }
    if !(0.0..=1.0).contains(&s.jitter) {
        return Err(Error::Settings("caustics jitter must lie in [0, 1]"));
    }
    if s.cell_lifetime > 255 {
        return Err(Error::Settings(
            "caustics cell_lifetime must be at most 255",
        ));
    }
    // A point reads level-L cells from level L - 1 on, at least lod_scale x 2^(L - 1) from the
    // focus. A cell of edge base x 2^L is a column one edge square across its bucket's axis,
    // and along it the points whose tangent planes cross the center line in one voxel lie
    // within two edges either side of it, for any normal whose bucket it is, so the photons in
    // the cell a point reads, jittered across it, lie within (sqrt(27) + sqrt(1/2)) x base x 2^L
    // of it. Above 4 x (sqrt(27) + sqrt(1/2)) = 23.6 base cells they lie at level L - 2 or
    // coarser, whose deposits reach level L.
    if s.lod_scale.is_nan() || s.lod_scale < 24.0 * 2f32.powi(s.base_cell_log2) {
        return Err(Error::Settings(
            "caustics lod_scale must be at least twenty-four base cells",
        ));
    }
    Ok(())
}

pub(crate) struct CausticPipelines {
    pool: ComputePipeline,
    lists: ComputePipeline,
    weigh: ComputePipeline,
    interiors: ComputePipeline,
    clear: ComputePipeline,
    heads: ComputePipeline,
    begin: ComputePipeline,
    resolve: ComputePipeline,
    finish: ComputePipeline,
}

impl CausticPipelines {
    pub(crate) fn new(device: &Device, kernels: &KernelSet) -> Result<Self, Error> {
        Ok(Self {
            pool: pipeline(device, kernels, &casters::MODULE, "pool")?,
            lists: pipeline(device, kernels, &casters::MODULE, "lists")?,
            weigh: pipeline(device, kernels, &casters::MODULE, "weigh")?,
            interiors: pipeline(device, kernels, &casters::MODULE, "interiors")?,
            clear: pipeline(device, kernels, &resolve::MODULE, "clear")?,
            heads: pipeline(device, kernels, &resolve::MODULE, "heads")?,
            begin: pipeline(device, kernels, &resolve::MODULE, "begin")?,
            resolve: pipeline(device, kernels, &resolve::MODULE, "resolve")?,
            finish: pipeline(device, kernels, &resolve::MODULE, "finish")?,
        })
    }
}

/// The workgroups of a dispatch over `count` threads of the resolve's group size.
fn resolve_grid(count: u64) -> [u32; 3] {
    let groups = count.div_ceil(RESOLVE_GROUP);
    [
        groups.min(RESOLVE_ROW) as u32,
        groups.div_ceil(RESOLVE_ROW) as u32,
        1,
    ]
}

/// One scene's caustics: the photon source and the real-time grid.
pub(crate) struct Caustics {
    settings: CausticSettings,
    pipelines: CausticPipelines,
    counters: Allocation<u32>,
    readback: Allocation<CausticStatsGpu>,
    source: Allocation<PhotonSourceGpu>,
    /// Held while the scene passes find a caster with photons on.
    grid: Option<Grid>,
    /// The producer entries' caster lists' CDFs, for `pool_room` clusters.
    pool: Allocation<u32>,
    pool_room: u32,
    /// For the scene's light tables at the last scene pass.
    per: PerEntry,
    roots: Ring<CausticsGpu>,
    caster_roots: Ring<casters::Root>,
    resolve_roots: Ring<resolve::Root>,
    clear: bool,
    /// Frame slots whose scene pass recorded the photon source.
    recorded: [bool; FRAME_SLOTS],
}

/// The cells of a scene's real-time grid.
struct Grid {
    checks: Allocation<u32>,
    accumulated: Allocation<[u32; 4]>,
    resolved: Allocation<[u32; 2]>,
    cell_photons: Allocation<f32>,
}

/// The memory a scene pass takes for the caustics before anything of the pass changes.
pub(crate) struct Room {
    per: Option<PerEntry>,
    pool: Option<(Allocation<u32>, u32)>,
    grid: Option<Grid>,
}

/// What a scene's caustics hold per emissive list entry and per feedback id, for light tables
/// `tables`.
pub(crate) struct PerEntry {
    tables: LightTables,
    selections: Allocation<EmissiveSelection>,
    cdf: Allocation<f32>,
    started: Allocation<u32>,
    deposited: Allocation<u32>,
    yields: Allocation<f32>,
    areas: Allocation<f32>,
    interiors: Allocation<EyeInterior>,
}

/// Feedback ids of light tables `tables`: every light, every emissive list entry, then the
/// environment.
pub(crate) fn feedback_ids(tables: LightTables) -> u64 {
    u64::from(tables.lights) + u64::from(tables.entries) + 1
}

impl PerEntry {
    fn new(device: &Device, tables: LightTables) -> Result<Self, Error> {
        let entries = u64::from(tables.entries);
        Ok(Self {
            tables,
            selections: device_local(device, entries, "caster lists")?,
            cdf: device_local(device, entries, "caster cdf")?,
            started: device_local(device, entries, "caster photons started")?,
            deposited: device_local(device, entries, "caster photons deposited")?,
            yields: device_local(device, entries, "caster yields")?,
            areas: device_local(device, entries, "caster areas")?,
            interiors: device_local(device, feedback_ids(tables), "light interiors")?,
        })
    }
}

impl Grid {
    fn new(device: &Device, settings: &CausticSettings) -> Result<Self, Error> {
        let capacity = 1u64 << settings.capacity_log2;
        Ok(Self {
            checks: device_local(device, capacity, "caustic grid checks")?,
            accumulated: device_local(device, 2 * capacity, "caustic grid deposits")?,
            resolved: device_local(device, capacity, "caustic grid irradiance")?,
            cell_photons: device_local(device, capacity, "caustic grid photons")?,
        })
    }
}

impl Caustics {
    pub(crate) fn new(
        device: &Device,
        kernels: &KernelSet,
        settings: &CausticSettings,
    ) -> Result<Self, Error> {
        check_settings(settings)?;
        Ok(Self {
            settings: *settings,
            pipelines: CausticPipelines::new(device, kernels)?,
            counters: device_local(device, COUNTERS, "caustic counters")?,
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                FRAME_SLOTS as u64,
                "caustic stats readback",
            )?,
            source: device_local(device, 1, "photon source")?,
            grid: None,
            pool: device_local(device, 0, "producer caster cdf")?,
            pool_room: 0,
            per: PerEntry::new(device, LightTables::default())?,
            roots: Ring::new(device, "caustic roots")?,
            caster_roots: Ring::new(device, "caster roots")?,
            resolve_roots: Ring::new(device, "caustic resolve roots")?,
            clear: true,
            recorded: [false; FRAME_SLOTS],
        })
    }

    /// Takes `settings`. A new size makes a held grid anew; that and new cells or a new focus
    /// falloff clear the grid and the casters' yields at the next scene pass. A new photon
    /// count, history, lifetime or jitter keeps them.
    pub(crate) fn set_settings(
        &mut self,
        device: &Device,
        settings: &CausticSettings,
    ) -> Result<(), Error> {
        check_settings(settings)?;
        let old = self.settings;
        if settings.capacity_log2 != old.capacity_log2 {
            if self.grid.is_some() {
                self.grid = Some(Grid::new(device, settings)?);
            }
            self.recorded = [false; FRAME_SLOTS];
            self.clear = true;
        }
        if settings.base_cell_log2 != old.base_cell_log2 || settings.lod_scale != old.lod_scale {
            self.clear = true;
        }
        self.settings = *settings;
        Ok(())
    }

    /// Empties every cell and forgets the casters' yields at the next scene pass.
    pub(crate) fn clear(&mut self) {
        self.clear = true;
    }

    /// Whether a scene pass that finds a caster where `casters` records the caustics.
    fn casts(&self, casters: bool) -> bool {
        casters && self.settings.photons > 0
    }

    /// The memory the caustics lack for a scene pass over light tables `tables`, for
    /// [`Self::follow`] to take once nothing else can fail: what they hold per entry and per
    /// feedback id where `grown`, the producers' caster lists where the tables hold room for
    /// other clusters, and the grid where the scene has come to hold `casters`. Fails without
    /// device memory.
    pub(crate) fn room(
        &self,
        device: &Device,
        tables: LightTables,
        grown: bool,
        casters: bool,
    ) -> Result<Room, Error> {
        let per = grown.then(|| PerEntry::new(device, tables)).transpose()?;
        let clusters = tables.producer_clusters;
        let pool = (clusters != self.pool_room)
            .then(|| device_local(device, u64::from(clusters), "producer caster cdf"))
            .transpose()?;
        let grid = (self.casts(casters) && self.grid.is_none())
            .then(|| Grid::new(device, &self.settings))
            .transpose()?;
        Ok(Room {
            per,
            pool: pool.map(|pool| (pool, clusters)),
            grid,
        })
    }

    /// Takes `room`, from [`Self::room`] with the same `casters`, in the scene pass of `frame`,
    /// and gives up the grid where the scene holds no caster: what the frames in flight read
    /// goes to `retired`. Returns what the caustics held per entry before, which the next
    /// [`Self::record`] takes the casters' yields from.
    pub(crate) fn follow(
        &mut self,
        room: Room,
        casters: bool,
        frame: u64,
        retired: &mut Retired,
    ) -> Option<PerEntry> {
        if let Some((pool, clusters)) = room.pool {
            retired.push(frame, std::mem::replace(&mut self.pool, pool));
            self.pool_room = clusters;
        }
        if let Some(grid) = room.grid {
            self.grid = Some(grid);
            self.clear = true;
        }
        if !self.casts(casters)
            && let Some(grid) = self.grid.take()
        {
            retired.push(frame, grid);
        }
        room.per.map(|per| std::mem::replace(&mut self.per, per))
    }

    /// Takes the entries of the scene's producer table, which the producers' caster lists
    /// run over.
    pub(crate) fn set_producers(&mut self, producers: u32) {
        self.per.tables.producers = producers;
    }

    /// Records slot `slot`'s photon source after the light set's selection and starts the
    /// grid's frame, for a frame focused on `focus`. `before` is what [`Self::follow`]
    /// returned since the last record, when the scene's light tables have grown since.
    /// Records nothing without the grid.
    pub(crate) fn record(
        &mut self,
        list: &mut CommandList<'_>,
        slot: usize,
        scene: GpuPtr<SceneRoot>,
        set: GpuPtr<LightSetGpu>,
        focus: [f32; 3],
        before: Option<&PerEntry>,
    ) {
        let s = self.settings;
        self.recorded[slot] = self.grid.is_some();
        let Some(grid) = &self.grid else {
            return;
        };
        let capacity = 1u64 << s.capacity_log2;
        let clear = std::mem::take(&mut self.clear);
        let per = &self.per;
        let before = before.unwrap_or(per);
        self.roots.write(
            slot,
            CausticsGpu {
                source: self.source.gpu(),
                checks: grid.checks.gpu(),
                accumulated: grid.accumulated.gpu(),
                resolved: grid.resolved.gpu(),
                photons: grid.cell_photons.gpu(),
                counters: self.counters.gpu(),
                map: GpuPtr::NULL,
                heads: GpuPtr::NULL,
                focus,
                lod_scale: s.lod_scale,
                capacity_mask: (capacity - 1) as u32,
                base_cell_log2: s.base_cell_log2,
                mode: CAUSTICS_GRID,
                map_capacity: 0,
                offset: [0.0; 3],
                radius: 0.0,
                photon_count: s.photons,
                jitter: s.jitter,
                _pad1: 0,
                _pad2: 0,
            },
        );
        let casters = self.caster_roots.write(
            slot,
            casters::Root {
                scene,
                set,
                source: self.source.gpu(),
                entries: per.selections.gpu(),
                cdf: per.cdf.gpu(),
                started: per.started.gpu(),
                deposited: per.deposited.gpu(),
                yields: per.yields.gpu(),
                areas: per.areas.gpu(),
                started_before: before.started.gpu(),
                deposited_before: before.deposited.gpu(),
                yields_before: before.yields.gpu(),
                _pad4: 0,
                pool: self.pool.gpu(),
                interiors: per.interiors.gpu(),
                focus: [focus[0], focus[1], focus[2], s.lod_scale],
                clear: u32::from(clear),
                counters: self.counters.gpu(),
                capacity: capacity as u32,
                base_cell_log2: s.base_cell_log2,
                cell_lifetime: s.cell_lifetime,
                entries_before: before.tables.entries,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            },
        );
        let grid = self.resolve_roots.write(
            slot,
            resolve::Root {
                caustics: self.roots.at(slot),
                stats: self.readback.gpu().offset(slot as u64),
                capacity: capacity as u32,
                cell_lifetime: s.cell_lifetime,
                max_temporal_samples: s.max_temporal_samples,
                clear: u32::from(clear),
            },
        );
        let p = &self.pipelines;
        list.dispatch(&p.pool, casters, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        if per.tables.producers > 0 {
            list.dispatch(&p.lists, casters, [per.tables.producers, 1, 1]);
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        }
        list.dispatch(&p.weigh, casters, [1, 1, 1]);
        let ids = feedback_ids(per.tables);
        list.dispatch(&p.interiors, casters, [ids.div_ceil(64) as u32, 1, 1]);
        if clear {
            list.dispatch(&p.clear, grid, resolve_grid(capacity));
        }
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&p.begin, grid, [1, 1, 1]);
    }

    /// The photons a frame starts, 0 with caustics off.
    pub(crate) fn photons(&self) -> u32 {
        self.settings.photons
    }

    /// The grid as slot `slot`'s frame reads it, after its scene pass; null where that pass
    /// recorded no caustics.
    pub(crate) fn root(&self, slot: usize) -> GpuPtr<CausticsGpu> {
        if self.recorded[slot] {
            self.roots.at(slot)
        } else {
            GpuPtr::NULL
        }
    }

    /// Records slot `slot`'s resolve after its photons, before any path gathers.
    pub(crate) fn record_resolve(&self, list: &mut CommandList<'_>, slot: usize) {
        if !self.recorded[slot] {
            return;
        }
        let capacity = 1u64 << self.settings.capacity_log2;
        let root = self.resolve_roots.at(slot);
        list.dispatch(&self.pipelines.resolve, root, resolve_grid(capacity));
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.pipelines.finish, root, [1, 1, 1]);
    }

    /// The statistics of slot `slot`'s frame; none where its scene pass recorded no caustics.
    pub(crate) fn stats(&self, slot: usize) -> CausticStats {
        if !self.recorded[slot] {
            return CausticStats::default();
        }
        let s = self.readback.as_slice().unwrap()[slot];
        CausticStats {
            started: s.started,
            deposits: s.deposits,
            overflows: s.overflows,
            occupied: s.occupied,
            evicted: s.evicted,
            quantum: s.quantum,
            finest: s.finest,
        }
    }

    /// Writes slot `slot`'s root of a reference view's `map` for this frame's `photons` and
    /// records the emptying of its lists before them.
    pub(crate) fn record_map(
        &self,
        list: &mut CommandList<'_>,
        map: &mut PhotonMap,
        slot: usize,
        photons: u32,
        offset: [f32; 3],
    ) -> GpuPtr<CausticsGpu> {
        let heads = 1u64 << map.heads_log2;
        let radius = map.radius();
        let root = map.roots.write(
            slot,
            CausticsGpu {
                source: self.source.gpu(),
                checks: GpuPtr::NULL,
                accumulated: GpuPtr::NULL,
                resolved: GpuPtr::NULL,
                photons: GpuPtr::NULL,
                counters: map.counters.gpu(),
                map: map.photons.gpu(),
                heads: map.heads.gpu(),
                focus: [0.0; 3],
                lod_scale: self.settings.lod_scale,
                capacity_mask: (heads - 1) as u32,
                base_cell_log2: 0,
                mode: CAUSTICS_MAP,
                map_capacity: map.capacity,
                offset: offset.map(|o| o * 2.0 * radius),
                radius,
                photon_count: photons,
                jitter: 0.0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        let clear = map.resolve_roots.write(
            slot,
            resolve::Root {
                caustics: root,
                stats: GpuPtr::NULL,
                capacity: heads as u32,
                cell_lifetime: 0,
                max_temporal_samples: 0.0,
                clear: 0,
            },
        );
        list.dispatch(&self.pipelines.heads, clear, resolve_grid(heads));
        root
    }
}

/// A reference view's photon map: one frame's photons in lists on a hash of cells two radii
/// wide, and the radius, which shrinks each frame as progressive photon mapping's from
/// `ReferenceSettings::photon_radius`.
pub(crate) struct PhotonMap {
    photons: Allocation<PhotonGpu>,
    heads: Allocation<u32>,
    counters: Allocation<u32>,
    roots: Ring<CausticsGpu>,
    resolve_roots: Ring<resolve::Root>,
    capacity: u32,
    heads_log2: u32,
    /// The squared radius of the next frame, m^2, and the frames before it.
    radius2: f64,
    frames: u32,
}

/// Progressive photon mapping's alpha: each frame keeps this share of the photons a disc of
/// the last radius would hold.
const PROGRESSIVE_ALPHA: f64 = 2.0 / 3.0;

impl PhotonMap {
    /// A map for `photons` photons a frame, room for two deposits each: a photon deposits where
    /// it ends and at each chain vertex with diffuse lobes on its way. A deposit past the room
    /// counts in [`crate::LightingStats::photon_overflows`].
    pub(crate) fn new(device: &Device, photons: u32) -> Result<Self, Error> {
        let capacity = photons.saturating_mul(2).max(1);
        let heads_log2 = capacity.next_power_of_two().trailing_zeros();
        Ok(Self {
            photons: device_local(device, u64::from(capacity), "photon map")?,
            heads: device_local(device, 1u64 << heads_log2, "photon map lists")?,
            counters: device_local(device, COUNTERS, "photon map counters")?,
            roots: Ring::new(device, "photon map roots")?,
            resolve_roots: Ring::new(device, "photon map clear roots")?,
            capacity,
            heads_log2,
            radius2: 0.0,
            frames: 0,
        })
    }

    /// Whether the map was made for `photons` photons a frame.
    pub(crate) fn holds(&self, photons: u32) -> bool {
        self.capacity == photons.saturating_mul(2).max(1)
    }

    /// Starts the next frame: from `radius` m after a reset, else shrinking the last radius.
    pub(crate) fn advance(&mut self, reset: bool, radius: f32) {
        if reset || self.frames == 0 {
            self.radius2 = f64::from(radius) * f64::from(radius);
            self.frames = 0;
        } else {
            let i = f64::from(self.frames - 1);
            self.radius2 *= (i + PROGRESSIVE_ALPHA) / (i + 1.0);
        }
        self.frames += 1;
    }

    fn radius(&self) -> f32 {
        self.radius2.sqrt() as f32
    }
}
