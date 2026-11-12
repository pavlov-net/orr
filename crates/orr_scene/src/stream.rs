//! Streamed units and user blobs: the render thread's residency step at step 1, and the
//! loading side's reads that place each unit.
//!
//! At step 1 the render thread takes the stream batches whose GPU work has landed, writes
//! each unit's address into its table entry or marks its texture tile resident, hands the
//! requests since the last step to the [`Residency`] manager, writes null over the entries
//! of the units it evicts or coarsens their textures' residency maps (their memory returns
//! once the frames that could read it have completed) and sends the loads to the loading
//! thread in tasks of a few MiB, so the GPU work of a task starts while the later ones still
//! read. The frame's submit waits the taken batches' points. Under memory pressure the step
//! shrinks the residency budgets, evicting streamed units, and the stream pool's blocks sit
//! at low priority so the OS pages them out before anything else.
//!
//! On the loading side a unit's blob, or a tile's chunk, is read through its mount's path
//! and decompressed on a worker into staging, cached host memory: a range of the read arena
//! where the device imports it, else a slot of the staging pool; mesh pages stay in their
//! stored form there. A blob stored as it decodes is read straight into its staging. Where
//! staging memory is device-local, as on unified memory, hierarchy and instance pages take
//! their slots in staging memory and land there. Once a task's reads and those of the tasks
//! sent before it have ended, the loading thread unbinds the tiles evicted since the last
//! task and binds the new ones in one sparse bind, copies the tiles in and decodes the mesh
//! pages into their stream pool slots on the loading queue, submits the copies into the
//! stream pool and the users' targets, builds the mesh pages' CLAS and hands the batch to the
//! render thread, which takes batches in the order their tasks were sent and drops the pages
//! whose decode failed.
//!
//! The step also builds a streaming mesh's finer cached cuts that copies want once their pages
//! are resident, admitting each whole when the mesh page pool has room for it, holding its pages
//! locked until no copy uses the cut, and moves tiles of a texture that lost one back beside its
//! others.
//!
//! Evictions leave the pools' blocks partly used, and a block's memory goes back to the system
//! only once it is empty. Each step the pools choose the blocks to empty ([`Pool::plan_drain`]),
//! all that the others have room for while the heap is over its mark, and the step moves the
//! mesh pages and the other pool units in them to other blocks, a few MiB a step: a unit takes
//! a new slot, the frame copies it there on the GPU before anything else, and the step's writes
//! point its table entries at the new slot, so the frame reads the copy and the frames before
//! it the old slot, which returns once this frame has completed.
//!
//! A tile is bound to its slot, and frames sample it at every step, so a tile moves over two
//! steps without leaving its residency map: the first has its frame copy the tile's texels out
//! to scratch memory; the next binds the tile to its new slot on the render queue, between that
//! frame's work and its own, and has its frame copy the texels back in before anything samples.
//! Where the render queue takes no sparse binds, tiles stay where they are.
//!
//! A step's copies stay queued until a frame whose list carries them is submitted
//! ([`Streaming::submitted`]), so a frame that is dropped leaves them to the next, and a tile is
//! bound to its new slot only once the copy of its texels out has been submitted. A moving
//! tile that is evicted, or whose mount goes, leaves its move: it stays bound where it was, as
//! its eviction expects, and the slot it would have taken returns unused.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use orr_gpu::{
    Allocation, CommandList, Device, GpuAddress, GpuPtr, Hazards, MemoryKind, QueueKind,
    SparseBinds, Stages, Texture, TextureRegion, TileBind, TimelinePoint,
};
use orr_io::{AlignedBuf, SECTOR};
use orr_pack::mesh::{Group, MeshHeader, StoredPage};
use orr_pack::{Blob, Codec};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::accel::{
    self, AccelShared, Animated, ClusterInfo, Cut, CutMember, CutRow, FineCut, PageMicromap,
};
use crate::clas_pool::{self, ClasPool, NO_UNIT, PageBuild, PoolBatch, PoolMicromap, Shortfall};
use crate::clusters::ClusterTable;
use crate::detail::LOW;
use crate::edit::Mirror;
use crate::loading::{
    BlobSource, Loading, Msg, Queue, Resumable, landed, read_into, reads_straight,
};
use crate::memory::{Block, Pool, PoolSlot, SceneMemory};
use crate::mount::{Apply, MeshPages, PageId, page_infos};
use crate::neural::NeuralLink;
use crate::residency::{
    Demand, HOLD_FRAMES, HeapUse, Held, Holder, InFlight, Key, Limits, MARK_TENTHS, Plan, Priority,
    RequestSource, Residency, ResidencyClass, ResidencyStats, UnitDesc, response_period,
};
use crate::scenes::PoolBytes;
use crate::shaders::page_decode::PageDecodeRoot;
use crate::shares::HeapShare;
use crate::tiles::{TexResidency, TexStream, TextureFeedback, Tile, compaction_move};
use crate::upload::{Op, Record};
use crate::{Budgets, ClusterRef, Error, LoadState, MountId, Result, RtCutGpu, TextureStreamGpu};

mod probe;

/// Tiles one step moves beside their texture's others, at most.
const MOVES_PER_STEP: usize = 8;
/// Bytes of a pool's units one step moves out of its draining blocks, at most, past the first
/// unit: each takes a GPU copy and the writes of its table entries.
const DRAIN_BYTES: u64 = 8 << 20;

/// Tiles one step starts moving out of the tile pool's draining blocks, at most: each takes a
/// sparse bind and two copies of its texels.
const DRAIN_TILES: usize = 64;

/// A unit a step moved: the frame copies `bytes` bytes from `src` to `dst` before anything
/// reads either.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnitMove {
    pub(crate) dst: u64,
    pub(crate) src: u64,
    pub(crate) bytes: u64,
}

/// A tile's texels on their way between the tile and scratch memory at `at`: out of the tile
/// when `out`, else back into it, before anything of the frame samples it.
pub(crate) struct TileCopy {
    pub(crate) stream: Arc<TexStream>,
    pub(crate) tile: u32,
    pub(crate) at: u64,
    pub(crate) out: bool,
}

/// What one step has a frame copy first; a later step's copies may read what these wrote.
#[derive(Default)]
pub(crate) struct Moves {
    pub(crate) units: Vec<UnitMove>,
    /// The tiles copied back in, then those copied out.
    pub(crate) tiles: Vec<TileCopy>,
}

/// Tiles of one texture and the slots they move to.
type TileSlots = Vec<(u32, PoolSlot)>;

/// A tile on its way to slot `to`, its texels at `at` in its batch's scratch.
struct TileMove {
    key: Key,
    to: PoolSlot,
    at: u64,
}

/// The tiles a step started moving. Its copies take each one's texels to the scratch; once a
/// frame carrying them is submitted (`copied`), a later step binds the tiles to their new
/// slots.
struct TileMoves {
    scratch: Allocation<u8>,
    tiles: Vec<TileMove>,
    copied: bool,
}

/// The resident units in each block of a pool.
#[derive(Default)]
struct BlockUnits(Vec<BTreeSet<Key>>);

impl BlockUnits {
    fn insert(&mut self, block: u32, k: Key) {
        let b = block as usize;
        if b >= self.0.len() {
            self.0.resize_with(b + 1, BTreeSet::new);
        }
        self.0[b].insert(k);
    }

    fn remove(&mut self, block: u32, k: Key) {
        self.0[block as usize].remove(&k);
    }

    /// A unit of `block`, when it holds one.
    fn any(&self, block: u32) -> Option<Key> {
        self.0.get(block as usize)?.first().copied()
    }
}

/// Frames a tile goes unused before it may move: its map coarsens until it lands again.
const MOVE_COLD_FRAMES: u64 = 8;

/// Who learns the address of a unit in a stream pool slot, null while it is not resident.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PoolTarget {
    /// An entry of the mesh page table.
    Table(u32),
    /// The transform groups placing an instance page.
    InstancePage(PageId),
    /// A window of an animated page's rows.
    PageWindow { page: PageId, window: u32 },
}

/// What [`Streaming::hold`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hold {
    Held,
    Waiting,
    Failed,
}

/// How a streamed unit loads.
#[derive(Clone, Copy, Debug)]
pub(crate) enum StreamUnit {
    /// A blob into a stream pool slot, whose address goes to `target` while it is resident.
    Pool {
        entry: u32,
        blob: u32,
        bytes: u32,
        target: PoolTarget,
    },
    /// Tile `tile` of the mount's streamed texture `texture`.
    Tile { texture: u32, tile: u32 },
    /// Page `page` of the mount's streamed mesh `mesh`.
    MeshPage { mesh: u32, page: u32 },
}

/// A mesh whose pages past its pinned ones stream: how they load and where they land.
pub(crate) struct MeshStream {
    pub(crate) entry: u32,
    pub(crate) header: MeshHeader,
    pub(crate) pages: Vec<StreamedPage>,
    /// Its first entry of the mesh page table.
    pub(crate) entries: u32,
    /// The resident cluster table.
    pub(crate) clusters: GpuPtr<ClusterRef>,
    /// Sections whose every triangle is non-opaque.
    pub(crate) non_opaque: Vec<bool>,
    /// Its group table.
    pub(crate) groups: Arc<[Group]>,
    /// Its index in its mount, and the first page that streams.
    pub(crate) mesh: u32,
    pub(crate) pinned: u32,
    /// Edits of the mesh, whose tables follow its landings and evictions.
    pub(crate) mirrors: Mutex<Vec<Arc<Mirror>>>,
}

/// One page of a [`MeshStream`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamedPage {
    pub(crate) blob: u32,
    /// Resident bytes.
    pub(crate) bytes: u32,
    /// Its first mesh cluster, cluster count and triangle count.
    pub(crate) first: u32,
    pub(crate) clusters: u32,
    pub(crate) triangles: u32,
}

/// What a streaming mesh's finer cached cuts are built from.
pub(crate) struct FineInit {
    pub(crate) mesh: u32,
    pub(crate) stream: Arc<MeshStream>,
    /// The table entries of its finer cuts, finest first, and those cuts.
    pub(crate) table: GpuPtr<RtCutGpu>,
    pub(crate) cuts: Vec<CutRow>,
    /// For each cut, the streamed pages holding a group whose error is over its threshold.
    pub(crate) pages: Vec<Vec<u32>>,
    pub(crate) merge: f32,
    pub(crate) animated: Animated,
    /// By page: its clusters and their CLAS while it is resident, the pinned pages' from
    /// mount. A cut's clusters sit in the pinned pages and its `pages`.
    pub(crate) resident: Vec<Option<Box<PageClusters>>>,
}

/// A resident page's clusters as its records name them, and the CLAS of each.
pub(crate) struct PageClusters {
    pub(crate) infos: Vec<ClusterInfo>,
    pub(crate) clas: Vec<u64>,
}

/// The streamed pages a finer cut holds locked, and the bytes they hold against the mesh
/// page budget.
struct Locks {
    pages: Vec<u32>,
    bytes: u64,
}

enum FineState {
    /// Not built: last wanted at frame `wanted` (0 never) to refine `error` pixels, and while
    /// `admitted` loading its pages to be built.
    Empty {
        wanted: u64,
        error: f32,
        admitted: bool,
    },
    /// Being built, holding `locks`.
    Building { locks: Locks },
    /// Built, last reported in use at `touched`, referencing `locks`. Once `shed` its table
    /// entry is empty, so no copy takes it up, and it goes like any cut no frame uses.
    Built {
        cut: FineCut,
        touched: u64,
        locks: Locks,
        shed: bool,
    },
    /// Its build failed, or a page of it failed to load; not built again while mounted.
    Failed,
}

impl FineState {
    const EMPTY: Self = Self::Empty {
        wanted: 0,
        error: 0.0,
        admitted: false,
    };
}

/// A streaming mesh's finer cached cuts on the render thread.
struct MeshFine {
    init: FineInit,
    slots: Vec<FineState>,
    /// By cut, the pages its build locks, those of its pages holding its clusters, once its
    /// pages have all been resident.
    locks: Vec<Option<Vec<u32>>>,
}

/// A finer cut for the loading thread to build.
pub(crate) struct CutTask {
    mount: MountId,
    mesh: u32,
    slot: u32,
    accel: Arc<AccelShared>,
    /// The pool of the CLAS its build reads.
    clas: Option<Arc<ClasPool>>,
    cut: Cut,
    members: Vec<CutMember>,
    /// The mesh's animated parts.
    parts: u32,
}

/// A finer cut built, handed over once its build has completed.
pub(crate) struct CutBatch {
    mount: MountId,
    mesh: u32,
    slot: u32,
    point: Option<TimelinePoint>,
    /// Its table record and memory.
    cut: Result<(RtCutGpu, FineCut)>,
}

impl CutBatch {
    pub(crate) fn landed(&self, device: &Device) -> bool {
        landed(device, self.point)
    }
}

/// Builds a finer cut on the loading thread.
pub(crate) fn build_cut<'a>(loading: &'a Loading, device: &'a Device, t: CutTask) -> Resumable<'a> {
    Box::pin(async move {
        let k = &t.accel.kernels;
        let clas = t.clas.as_deref();
        let built = accel::build_fine_cut(device, k, clas, t.cut, &t.members, t.parts).await;
        let (cut, point) = match built {
            Ok((r, c, p)) => (Ok((r, c)), Some(p)),
            Err(e) => (Err(e), None),
        };
        loading.push_cut(CutBatch {
            mount: t.mount,
            mesh: t.mesh,
            slot: t.slot,
            point,
            cut,
        });
    })
}

/// Where a requested blob lands: consumer memory that stays alive until the blob is ready
/// or failed.
#[derive(Clone)]
pub enum BlobTarget {
    /// `len` bytes from `at`; the blob's decoded length must fit.
    Buffer { at: GpuPtr<u8>, len: u64 },
    /// `region` of `texture`, whose texels the blob holds tightly packed.
    Texture {
        texture: Arc<Texture>,
        region: TextureRegion,
    },
}

/// A blob requested with [`crate::Scenes::request_blob`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlobId(pub u32);

/// A mount's units as the stream step sees them.
pub(crate) struct MountUnits {
    /// The unit of each instance page, which its windows of rows follow.
    pub(crate) page_units: Vec<u32>,
    /// Every unit, and how to load the ones that start absent.
    pub(crate) units: Vec<(UnitDesc, Option<StreamUnit>)>,
    /// Streamed units of each mesh.
    pub(crate) meshes: Vec<Vec<u32>>,
    /// The unit of each node page of each mesh.
    pub(crate) node_pages: Vec<Vec<u32>>,
    /// The mount's streamed textures and the unit of each one's first tile.
    pub(crate) textures: Vec<(Arc<TexStream>, u32)>,
    /// The unit of each page of each mesh.
    pub(crate) mesh_pages: Vec<Vec<u32>>,
    /// The mount's meshes whose pages stream, as `MeshPage` units name them.
    pub(crate) mesh_streams: Vec<Arc<MeshStream>>,
    /// Those of them with finer cached cuts.
    pub(crate) fine: Vec<FineInit>,
}

struct MountStreams {
    source: Arc<BlobSource>,
    /// By unit; `None` for units the mount placed.
    units: Vec<Option<StreamUnit>>,
    meshes: Vec<Vec<u32>>,
    node_pages: Vec<Vec<u32>>,
    textures: Vec<TexResidency>,
    mesh_pages: Vec<Vec<u32>>,
    mesh_streams: Vec<Arc<MeshStream>>,
    neural: Vec<NeuralStreams>,
}

/// A texture a neural set stands in for.
struct NeuralStreams {
    index: u32,
    /// The mount wrote the stream table entry for it, since it does not stream.
    own: bool,
    /// The index in `textures` of the set's latents when they stream.
    latents: Option<usize>,
    /// The latent mip of each color mip.
    latent_mips: Vec<u32>,
}

impl MountStreams {
    /// The stream of mesh `mesh`, when its pages stream.
    fn stream_of_mesh(&self, mesh: u32) -> Option<&Arc<MeshStream>> {
        let pages = self.mesh_pages.get(mesh as usize)?;
        pages.iter().find_map(|&u| match self.units[u as usize] {
            Some(StreamUnit::MeshPage { mesh, .. }) => self.mesh_streams.get(mesh as usize),
            _ => None,
        })
    }

    /// The texture and tile that unit `unit` is.
    fn tile(&mut self, unit: u32) -> (&mut TexResidency, u32) {
        match self.units[unit as usize] {
            Some(StreamUnit::Tile { texture, tile }) => {
                (&mut self.textures[texture as usize], tile)
            }
            _ => unreachable!("unit {unit} is a tile"),
        }
    }
}

/// Where a resident unit's memory is.
enum Placed {
    Pool(PoolSlot, PoolTarget),
    Tile(PoolSlot),
    MeshPage(MeshPlaced),
}

/// A streamed mesh page: its slot, the cluster id of its first cluster, the unit ranges its
/// CLAS take and its micromap.
struct MeshPlaced {
    slot: PoolSlot,
    stream: Arc<MeshStream>,
    page: u32,
    ids: u32,
    /// Its resident address, once it has landed.
    address: u64,
    clas: Vec<(u32, u32)>,
    micromap: Option<PoolMicromap>,
}

impl MeshPlaced {
    /// Its mesh page table entry and its clusters' entries in the resident cluster table.
    fn entries(&self) -> (u32, GpuPtr<ClusterRef>, u32) {
        (
            self.stream.entries + self.page,
            self.stream.clusters.offset(u64::from(self.ids)),
            self.clusters(),
        )
    }

    fn clusters(&self) -> u32 {
        self.stream.pages[self.page as usize].clusters
    }

    /// What it holds besides its slot.
    fn free(self) -> (PoolSlot, PageFree) {
        let ids = (self.ids, self.clusters());
        let free = PageFree {
            clas: self.clas,
            micromap: self.micromap,
            ids,
        };
        (self.slot, free)
    }
}

/// A streamed page's CLAS, micromap and cluster ids, freed once no frame reads them.
struct PageFree {
    clas: Vec<(u32, u32)>,
    micromap: Option<PoolMicromap>,
    ids: (u32, u32),
}

/// A bound tile to unbind and free, whose slot is retired.
#[derive(Clone)]
struct TileFree {
    key: Key,
    stream: Arc<TexStream>,
    tile: u32,
    slot: PoolSlot,
}

enum BlobState {
    Queued(Box<BlobRequest>),
    Loading { released: bool },
    Ready,
    Failed(Arc<Error>),
}

struct BlobRequest {
    source: Arc<BlobSource>,
    blob: Blob,
    target: BlobTarget,
}

struct UserBlob {
    state: BlobState,
    bytes: u64,
}

pub(crate) struct Streaming {
    residency: Residency,
    accel: Arc<AccelShared>,
    /// The mesh page table, where the entries of what lands and leaves are written.
    mesh_pages: Arc<MeshPages>,
    /// Streamed pages' CLAS; none without cluster acceleration structures.
    clas: Option<Arc<ClasPool>>,
    /// Streamed pages' CLAS and cluster ids, freed once the frame has completed.
    retired_pages: Vec<(u64, PageFree)>,
    /// The first cluster id each streamed mesh page loading holds, and its cluster count.
    loading_ids: FxHashMap<Key, (u32, u32)>,
    /// Mesh pages.
    pool: Arc<Mutex<Pool>>,
    /// Hierarchy and instance pages, in blocks of their own: DeviceLocal memory, or staging
    /// memory where it is device-local, which they land in with no copy.
    units: Arc<Mutex<Pool>>,
    staging: Arc<Mutex<Pool>>,
    tiles: Arc<Mutex<Pool>>,
    /// Finer cached cuts by mount and mesh.
    fine: BTreeMap<(MountId, u32), MeshFine>,
    /// Bytes of mesh pages they hold locked.
    fine_locked: u64,
    /// What the latest sample since the last step found each class wants.
    demand: Option<[Demand; ResidencyClass::ALL.len()]>,
    /// Finer cuts freed once the frame has completed.
    retired_cuts: Vec<(u64, FineCut)>,
    mounts: FxHashMap<MountId, MountStreams>,
    resident: FxHashMap<Key, Placed>,
    /// The resident mesh pages by block of the stream pool, the other pool units by block of
    /// theirs, and the resident tiles by block of the tile pool.
    page_blocks: BlockUnits,
    unit_blocks: BlockUnits,
    tile_blocks: BlockUnits,
    /// The copies of the units and tiles the steps moved, a step's after those of the steps
    /// before, until a frame whose list carries them is submitted.
    moves: Vec<Moves>,
    /// The tiles a step started moving and no step has bound yet.
    tile_moves: Option<TileMoves>,
    /// The tiles bound to new slots whose copy back is in `moves`.
    tile_finished: FxHashSet<Key>,
    /// A test holds the binds of moving tiles back, as a busy loading thread does.
    #[cfg(feature = "test-hooks")]
    tile_binds_held: bool,
    /// A test fails the scratch allocations of tile moves, as a device out of memory does.
    #[cfg(feature = "test-hooks")]
    tile_scratch_fails: bool,
    /// Whether the render queue takes the sparse binds that move tiles.
    moves_tiles: bool,
    /// The slots moved tiles left and the scratch their texels crossed, freed once the frame
    /// has completed.
    retired_tile_slots: Vec<(u64, PoolSlot)>,
    retired_scratch: Vec<(u64, Allocation<u8>)>,
    /// This manager's part of what the device's residency managers hold on its heaps.
    share: HeapShare,
    /// The heap index of the stream pool's and the tile pool's memory, and of the other pool
    /// units'.
    pool_heaps: [usize; 2],
    /// Pool slots freed once the frame has completed.
    retired: Vec<(u64, PoolSlot)>,
    /// `units` slots freed once the frame has completed.
    retired_units: Vec<(u64, PoolSlot)>,
    /// Tiles unbound once the frame has completed.
    retired_tiles: Vec<(u64, TileFree)>,
    /// Tiles the next task unbinds.
    unbinds: Vec<TileFree>,
    /// Unbinds of tasks whose sparse bind failed, until it is known whether the tile was
    /// bound again.
    carried: Vec<TileFree>,
    /// Tiles whose unbind is queued and not yet sent to the loading side; they do not load
    /// again until then, so their unbind never shares a sparse bind with a new one.
    unbinding: FxHashSet<Key>,
    feedback: TextureFeedback,
    /// Feedback cells freed once the frame has completed.
    retired_cells: Vec<(u64, u32)>,
    streams: GpuPtr<GpuPtr<TextureStreamGpu>>,
    /// Tiles leaving to land beside their texture's others: the slot each should follow and
    /// the frame it stops asking.
    moving: HashMap<Key, (PoolSlot, u64)>,
    /// Textures that lost a tile, whose others may move, until the given frame: the room a
    /// move wants comes free some frames after other textures' evictions. Moves take their
    /// per-step limit in this order.
    fragmented: BTreeMap<(MountId, u32), u64>,
    /// Heap indices whose stream table entry the next step writes null.
    nulls: Vec<u32>,
    /// A texture has streamed: a record with a residency map has been written.
    streamed: bool,
    /// The memory kinds on a device-local heap, whose pressure the residency budgets follow.
    governed: Vec<MemoryKind>,
    /// Instance pages that landed (their records' address) or left (0) since the last take.
    page_events: Vec<(PageId, u64)>,
    window_events: Vec<(PageId, u32, u64)>,
    blobs: Vec<Option<UserBlob>>,
    queued: VecDeque<u32>,
    fif: u32,
}

impl Streaming {
    /// Whether a texture has streamed since the renderer was made.
    pub(crate) fn streamed(&self) -> bool {
        self.streamed
    }

    pub(crate) fn new(
        device: &Device,
        b: &Budgets,
        pools: PoolBytes,
        accel: &Arc<AccelShared>,
        mesh_pages: &Arc<MeshPages>,
        streams: GpuPtr<GpuPtr<TextureStreamGpu>>,
        fif: u32,
    ) -> Result<Self> {
        let mut budgets = [0; ResidencyClass::ALL.len()];
        budgets[ResidencyClass::MeshPages as usize] = pools.mesh;
        budgets[ResidencyClass::Textures as usize] = pools.textures;
        budgets[ResidencyClass::InstancePages as usize] = b.instance_page_pool;
        budgets[ResidencyClass::UserBlobs as usize] = b.user_blob_pool;
        let caps = device.caps();
        // Mesh pages take the stream pool and the CLAS pool, tiles the tile pool, and the other
        // units, node pages among them, take `units`, as they come and go apart from mesh pages;
        // user blobs take the consumer's memory.
        let units_kind = if caps.staging_device_local {
            MemoryKind::Staging
        } else {
            MemoryKind::DeviceLocal
        };
        let report = device.memory_report();
        let heap = |kinds: &[MemoryKind]| {
            kinds.iter().fold(0u32, |bits, k| {
                let kind = MemoryKind::ALL.iter().position(|a| a == k).expect("a kind");
                bits | 1 << report.kind_heaps[kind]
            })
        };
        let mut heaps = [0; ResidencyClass::ALL.len()];
        heaps[ResidencyClass::MeshPages as usize] = heap(&[MemoryKind::DeviceLocal, units_kind]);
        heaps[ResidencyClass::Textures as usize] = heap(&[units_kind, MemoryKind::DeviceLocal]);
        heaps[ResidencyClass::InstancePages as usize] = heap(&[units_kind]);
        let block = pool_block(device, b, pools);
        let pool = Arc::new(Mutex::new(Pool::new(
            MemoryKind::DeviceLocal,
            256,
            block,
            "stream pool",
        )));
        let clas = if caps.cluster.max_tris > 0 && caps.cluster.max_partitions > 0 {
            let pool = ClasPool::new(device, pools.mesh, block)?;
            Some(Arc::new(pool))
        } else {
            None
        };
        Ok(Self {
            accel: Arc::clone(accel),
            mesh_pages: Arc::clone(mesh_pages),
            clas,
            retired_pages: Vec::new(),
            loading_ids: FxHashMap::default(),
            residency: Residency::new(Limits {
                budgets,
                heaps,
                period: response_period(fif),
                in_flight: b.in_flight_bytes,
            }),
            units: Arc::new(Mutex::new(Pool::new(
                units_kind,
                256,
                block,
                "stream units",
            ))),
            pool,
            // Sector units, so that reads go straight into a slot.
            staging: Arc::new(Mutex::new(
                Pool::new(MemoryKind::Staging, SECTOR as u64, block, "stream staging").keep_spare(),
            )),
            tiles: Arc::new(Mutex::new(Pool::new(
                MemoryKind::DeviceLocal,
                orr_pack::texture::TILE_BYTES as u64,
                block,
                "texture tile pool",
            ))),
            fine: BTreeMap::new(),
            fine_locked: 0,
            demand: None,
            governed: {
                let report = device.memory_report();
                MemoryKind::ALL
                    .into_iter()
                    .filter(|&k| report.heap_of(k).device_local)
                    .collect()
            },
            retired_cuts: Vec::new(),
            mounts: FxHashMap::default(),
            resident: FxHashMap::default(),
            page_blocks: BlockUnits::default(),
            unit_blocks: BlockUnits::default(),
            tile_blocks: BlockUnits::default(),
            moves: Vec::new(),
            tile_moves: None,
            tile_finished: FxHashSet::default(),
            #[cfg(feature = "test-hooks")]
            tile_binds_held: false,
            #[cfg(feature = "test-hooks")]
            tile_scratch_fails: false,
            moves_tiles: device.binds_sparse_on(QueueKind::Render),
            retired_tile_slots: Vec::new(),
            retired_scratch: Vec::new(),
            share: HeapShare::new(device, response_period(fif)),
            pool_heaps: {
                let heap = |kind: MemoryKind| {
                    let k = MemoryKind::ALL.iter().position(|&a| a == kind);
                    report.kind_heaps[k.expect("a kind")]
                };
                [heap(MemoryKind::DeviceLocal), heap(units_kind)]
            },
            retired: Vec::new(),
            retired_units: Vec::new(),
            retired_tiles: Vec::new(),
            unbinds: Vec::new(),
            carried: Vec::new(),
            unbinding: FxHashSet::default(),
            feedback: TextureFeedback::new(device, fif)?,
            retired_cells: Vec::new(),
            streams,
            moving: HashMap::new(),
            fragmented: BTreeMap::new(),
            nulls: Vec::new(),
            streamed: false,
            page_events: Vec::new(),
            window_events: Vec::new(),
            blobs: Vec::new(),
            queued: VecDeque::new(),
            fif: fif.max(1),
        })
    }

    pub(crate) fn stats(&self) -> &ResidencyStats {
        self.residency.stats()
    }

    /// See [`Residency::lower_budget`].
    pub(crate) fn lower_budget(&mut self, class: ResidencyClass, bytes: u64) {
        self.residency.lower_budget(class, bytes);
    }

    /// The streamed pools' fill and the CLAS pool's pages.
    pub(crate) fn memory(&self, out: &mut SceneMemory) {
        let pools = [&self.pool, &self.units, &self.staging, &self.tiles];
        out.pools
            .extend(pools.into_iter().map(|p| p.lock().unwrap().fill()));
        if let Some(c) = &self.clas {
            let (pages, fills) = c.fill();
            out.clas = Some(pages);
            out.pools.extend(fills);
        }
    }

    /// What mount reads share the in-flight limit with streamed loads through.
    pub(crate) fn in_flight(&self) -> &Arc<InFlight> {
        self.residency.in_flight()
    }

    /// Mesh `mesh` of `mount`'s page `page` where it streams: its address and the cluster id
    /// of its first cluster, or `None` while it is not resident. The outer `None`: the page
    /// does not stream.
    pub(crate) fn streamed_page(
        &self,
        mount: MountId,
        mesh: u32,
        page: u32,
    ) -> Option<Option<(u64, u32)>> {
        let m = self.mounts.get(&mount)?;
        let unit = *m.mesh_pages.get(mesh as usize)?.get(page as usize)?;
        m.units[unit as usize]?;
        Some(match self.resident.get(&Key { mount, unit }) {
            Some(Placed::MeshPage(p)) => Some((p.address, p.ids)),
            _ => None,
        })
    }

    /// Keeps `mirror` in step with mesh `mesh` of `mount` when its pages stream.
    pub(crate) fn add_mirror(&self, mount: MountId, mesh: u32, mirror: &Arc<Mirror>) {
        let Some(m) = self.mounts.get(&mount) else {
            return;
        };
        if let Some(s) = m.stream_of_mesh(mesh) {
            s.mirrors.lock().unwrap().push(Arc::clone(mirror));
        }
    }

    pub(crate) fn remove_mirror(&self, mount: MountId, mesh: u32, mirror: &Arc<Mirror>) {
        let Some(m) = self.mounts.get(&mount) else {
            return;
        };
        if let Some(s) = m.stream_of_mesh(mesh) {
            s.mirrors
                .lock()
                .unwrap()
                .retain(|o| !Arc::ptr_eq(o, mirror));
        }
    }

    /// Adds a mount's units, points the stream table at its streamed textures and at the
    /// records of textures that a neural set stands in for and that do not stream, and
    /// writes their neural records.
    pub(crate) fn register(
        &mut self,
        mount: MountId,
        source: Arc<BlobSource>,
        units: MountUnits,
        neural: &[(u32, &NeuralLink)],
        w: &mut Apply<'_>,
    ) -> Result<()> {
        let mut textures: Vec<TexResidency> = Vec::with_capacity(units.textures.len());
        for (k, (stream, first_unit)) in units.textures.into_iter().enumerate() {
            let g = &stream.grid;
            let [cx, cy] = g.cells();
            let cells = cx * cy * g.layers;
            let first_cell = match self
                .feedback
                .alloc((mount, k as u32), cells, g.tail_mip(), w)
            {
                Ok(first) => first,
                Err(e) => {
                    for t in &textures {
                        self.feedback.free(t.first_cell);
                    }
                    return Err(e);
                }
            };
            textures.push(TexResidency {
                resident: vec![false; g.tile_count() as usize],
                heard: vec![u64::MAX; g.tile_count() as usize],
                first_unit,
                first_cell,
                dirty: false,
                stream,
            });
        }
        let descs: Vec<UnitDesc> = units
            .units
            .iter()
            .map(|(d, s)| {
                let mut d = d.clone();
                if let (Some(StreamUnit::MeshPage { mesh, page }), Some(_)) = (s, &self.clas) {
                    let p = units.mesh_streams[*mesh as usize].pages[*page as usize];
                    d.built = clas_pool::estimate(p.triangles, p.clusters) * clas_pool::UNIT;
                }
                d
            })
            .collect();
        if let Err(e) = self.residency.register(mount, &descs) {
            for t in &textures {
                self.feedback.free(t.first_cell);
            }
            return Err(e);
        }
        for t in &textures {
            let s = &t.stream;
            let record = TextureStreamGpu {
                map: s.map,
                feedback: self.feedback.gpu.gpu(),
                cells: s.grid.cells(),
                first_cell: t.first_cell,
                tail_mip: s.grid.tail_mip(),
                neural: s.neural,
            };
            w.put(s.record, &[record])?;
            w.put(self.streams.offset(u64::from(s.index.0)), &[s.record])?;
            self.streamed = true;
        }
        for init in units.fine {
            let slots = init.cuts.iter().map(|_| FineState::EMPTY).collect();
            let locks = vec![None; init.cuts.len()];
            self.fine
                .insert((mount, init.mesh), MeshFine { init, slots, locks });
        }
        let mut linked = Vec::with_capacity(neural.len());
        for &(index, n) in neural {
            w.put(n.record, &[n.gpu])?;
            let own = n.own.address().0 != 0;
            if own {
                let record = TextureStreamGpu {
                    map: GpuPtr::NULL,
                    feedback: GpuPtr::NULL,
                    cells: [0; 2],
                    first_cell: 0,
                    tail_mip: 0,
                    neural: n.record,
                };
                w.put(n.own, &[record])?;
                w.put(self.streams.offset(u64::from(index)), &[n.own])?;
            }
            let latents = n.latents.as_ref().map(|l| {
                textures
                    .iter()
                    .position(|t| Arc::ptr_eq(&t.stream, l))
                    .expect("streamed latents are the mount's")
            });
            linked.push(NeuralStreams {
                index,
                own,
                latents,
                latent_mips: n.latent_mips.clone(),
            });
        }
        self.mounts.insert(
            mount,
            MountStreams {
                source,
                units: units.units.into_iter().map(|(_, s)| s).collect(),
                meshes: units.meshes,
                node_pages: units.node_pages,
                textures,
                mesh_pages: units.mesh_pages,
                mesh_streams: units.mesh_streams,
                neural: linked,
            },
        );
        Ok(())
    }

    /// Drops a mount's units; their memory is freed once `frame` has completed. Its queued
    /// blobs fail.
    pub(crate) fn unregister(&mut self, mount: MountId, frame: u64) {
        let Some(mut m) = self.mounts.remove(&mount) else {
            return;
        };
        self.residency.unregister(mount);
        let cuts: Vec<(MountId, u32)> =
            self.fine.keys().filter(|k| k.0 == mount).copied().collect();
        let mut failed = 0;
        for k in cuts {
            let f = self.fine.remove(&k).expect("listed");
            for state in f.slots {
                let locks = match state {
                    FineState::Empty { .. } => continue,
                    FineState::Failed => {
                        failed += 1;
                        continue;
                    }
                    FineState::Building { locks } => locks,
                    FineState::Built { cut, locks, .. } => {
                        self.retired_cuts.push((frame, cut));
                        locks
                    }
                };
                self.fine_locked -= locks.bytes;
            }
        }
        self.residency.forget_failed_cuts(failed);
        let gone: Vec<Key> = self
            .resident
            .keys()
            .filter(|k| k.mount == mount)
            .copied()
            .collect();
        for k in gone {
            match self.resident.remove(&k).expect("resident") {
                Placed::Pool(slot, _) => self.retire_unit(k, slot, frame),
                Placed::MeshPage(p) => {
                    let (slot, free) = p.free();
                    self.retire_page(k, slot, frame);
                    self.retired_pages.push((frame, free));
                }
                Placed::Tile(slot) => {
                    let (t, tile) = m.tile(k.unit);
                    let free = TileFree {
                        key: k,
                        stream: Arc::clone(&t.stream),
                        tile,
                        slot,
                    };
                    self.tile_blocks.remove(slot.block, k);
                    self.tiles.lock().unwrap().retire(slot);
                    self.retired_tiles.push((frame, free));
                }
            }
        }
        // Its tiles on their way to new slots stay bound where they were, which the frees
        // above cover, and the slots they would have taken return unused.
        if let Some(moves) = &mut self.tile_moves {
            let mut tiles = self.tiles.lock().unwrap();
            moves.tiles.retain(|t| {
                if t.key.mount == mount {
                    tiles.free(t.to);
                }
                t.key.mount != mount
            });
        }
        self.tile_finished.retain(|k| k.mount != mount);
        self.unbinding.retain(|k| k.mount != mount);
        self.moving.retain(|k, _| k.mount != mount);
        self.fragmented.retain(|k, _| k.0 != mount);
        for t in &m.textures {
            self.nulls.push(t.stream.index.0);
            self.retired_cells.push((frame, t.first_cell));
        }
        self.nulls
            .extend(m.neural.iter().filter(|n| n.own).map(|n| n.index));
        for id in std::mem::take(&mut self.queued) {
            let b = self.blobs[id as usize].as_mut().expect("queued blob");
            let BlobState::Queued(r) = &b.state else {
                unreachable!("a queued blob is queued")
            };
            if Arc::ptr_eq(&r.source, &m.source) {
                b.state = BlobState::Failed(Arc::new(Error::Unmounted(mount)));
            } else {
                self.queued.push_back(id);
            }
        }
    }

    /// The instance pages that landed or left since the last call.
    pub(crate) fn take_page_events(&mut self) -> Vec<(PageId, u64)> {
        std::mem::take(&mut self.page_events)
    }

    pub(crate) fn note_pages_over_budget(&mut self, count: u32) {
        self.residency.note_pages_over_budget(count);
    }

    /// The windows of rows, by page, that landed or left since the last call.
    pub(crate) fn take_window_events(&mut self) -> Vec<(PageId, u32, u64)> {
        std::mem::take(&mut self.window_events)
    }

    /// The units an edit of mesh `mesh` of `mount` with copies of pages `own` keeps resident
    /// while it is drawn, since its copies are: the node pages over the copies' groups' leaves
    /// and, with `coarser`, the pages holding the coarser clusters those groups simplify into,
    /// each with what it depends on. None where the mesh's pages do not stream.
    pub(crate) fn edit_holds(
        &self,
        mount: MountId,
        mesh: u32,
        own: &[u32],
        coarser: bool,
    ) -> Vec<Key> {
        let Some(m) = self.mounts.get(&mount) else {
            return Vec::new();
        };
        if m.stream_of_mesh(mesh).is_none() {
            return Vec::new();
        }
        let pages = &m.mesh_pages[mesh as usize];
        let own: FxHashSet<u32> = own.iter().map(|&q| pages[q as usize]).collect();
        let mut held: Vec<u32> = own
            .iter()
            .flat_map(|&unit| self.residency.deps(Key { mount, unit }))
            .copied()
            .filter(|d| !own.contains(d) && (coarser || pages.binary_search(d).is_err()))
            .collect();
        held.sort_unstable();
        held.dedup();
        held.into_iter().map(|unit| Key { mount, unit }).collect()
    }

    /// Asks for `keys` for the next step, which keeps those already resident in their hold
    /// while the rest load; returns whether any is not resident.
    pub(crate) fn want(&mut self, keys: &[Key]) -> bool {
        let p = Priority {
            error: 0.0,
            cell_distance: 0,
        };
        let mut waiting = false;
        for &k in keys {
            self.residency.request(k, p, RequestSource::Hints);
            waiting |= !self.residency.resident(k);
        }
        waiting
    }

    /// Locks `keys` once they are all resident, and asks for them until then. A unit that
    /// failed, or that depends on one that did, never loads.
    pub(crate) fn hold(&mut self, keys: &[Key]) -> Hold {
        if keys.iter().any(|&k| self.residency.failed(k)) {
            return Hold::Failed;
        }
        if self.want(keys) {
            return Hold::Waiting;
        }
        for &k in keys {
            self.residency.lock(k);
        }
        Hold::Held
    }

    /// Unlocks what [`Self::hold`] locked.
    pub(crate) fn release(&mut self, keys: &[Key]) {
        for &k in keys {
            self.residency.unlock(k);
        }
    }

    /// Requests unit `unit` of `mount` with hint priority for the next step.
    pub(crate) fn request_unit(
        &mut self,
        mount: MountId,
        unit: u32,
        cell_distance: u32,
        source: RequestSource,
    ) {
        let p = Priority {
            error: 0.0,
            cell_distance,
        };
        self.residency.request(Key { mount, unit }, p, source);
    }

    /// The budget instance pages plan under.
    pub(crate) fn instance_page_budget(&self) -> u64 {
        self.residency
            .stats()
            .class(ResidencyClass::InstancePages)
            .budget
    }

    /// Requests every streamed unit of mesh `mesh` of `mount` for the next step.
    pub(crate) fn prefetch(&mut self, mount: MountId, mesh: u32, cell_distance: u32) {
        let Some(m) = self.mounts.get(&mount) else {
            return;
        };
        let p = Priority {
            error: 0.0,
            cell_distance,
        };
        for &unit in &m.meshes[mesh as usize] {
            self.residency
                .request(Key { mount, unit }, p, RequestSource::Hints);
        }
    }

    /// Requests every tile of mips `finest..` of `mount`'s streamed texture at heap index
    /// `index` for the next step, and for a texture a neural set stands in for, the tiles of
    /// the latent mips that its color mips from `finest` on read.
    pub(crate) fn prefetch_texture(&mut self, mount: MountId, index: u32, finest: u32) {
        let Some(m) = self.mounts.get(&mount) else {
            return;
        };
        let mut wanted = Vec::new();
        if let Some(t) = m.textures.iter().position(|t| t.stream.index.0 == index) {
            wanted.push((t, finest));
        }
        if let Some(n) = m.neural.iter().find(|n| n.index == index)
            && let Some(k) = n.latents
        {
            let mips = &n.latent_mips;
            wanted.push((k, mips[(finest as usize).min(mips.len() - 1)]));
        }
        let p = Priority {
            error: 0.0,
            cell_distance: 0,
        };
        for (t, finest) in wanted {
            let t = &m.textures[t];
            let g = &t.stream.grid;
            for tile in (0..g.tile_count()).filter(|&tile| g.tile(tile).mip >= finest) {
                let unit = t.first_unit + tile;
                self.residency
                    .request(Key { mount, unit }, p, RequestSource::Hints);
            }
        }
    }

    /// Node page or mesh page `page` of mesh `mesh` of `mount`: requested for the next step
    /// with the projected error it would refine, or read by a traversal in `frame`.
    pub(crate) fn page(
        &mut self,
        mount: MountId,
        mesh: u32,
        page: u32,
        mesh_page: bool,
        request: Option<f32>,
        frame: u64,
    ) {
        let Some(&unit) = self
            .mounts
            .get(&mount)
            .and_then(|m| {
                if mesh_page {
                    m.mesh_pages.get(mesh as usize)
                } else {
                    m.node_pages.get(mesh as usize)
                }
            })
            .and_then(|p| p.get(page as usize))
        else {
            return;
        };
        let k = Key { mount, unit };
        match request {
            Some(error) => self.residency.request(
                k,
                Priority {
                    error,
                    cell_distance: 0,
                },
                RequestSource::Traversal,
            ),
            None => self.residency.touch(k, frame),
        }
    }

    /// Finer cut `k` of mesh `mesh` of `mount`: wanted by copies whose cached cut leaves
    /// `request` pixels of error, or reported in use, in `frame`.
    pub(crate) fn fine_cut(
        &mut self,
        mount: MountId,
        mesh: u32,
        k: u32,
        request: Option<f32>,
        frame: u64,
    ) {
        let Some(f) = self.fine.get_mut(&(mount, mesh)) else {
            return;
        };
        match (f.slots.get_mut(k as usize), request) {
            (Some(FineState::Empty { wanted, error, .. }), Some(e)) => {
                // The completed frames a step reads report at that step's frame.
                *error = if *wanted == frame { error.max(e) } else { e };
                *wanted = frame;
            }
            // A shed cut stays while an instance still traces it, not for being wanted.
            (Some(FineState::Built { touched, shed, .. }), _) if !*shed || request.is_none() => {
                *touched = frame;
            }
            _ => {}
        }
    }

    /// Starts the sample of a completed frame's feedback ([`Residency::begin_sample`]).
    pub(crate) fn begin_sample(&mut self) {
        self.residency.begin_sample();
    }

    /// Ends the sample [`Self::begin_sample`] started, which the next step's finer cuts admit
    /// by; returns the mesh page pool's demand.
    pub(crate) fn end_sample(&mut self) -> Demand {
        let demand = self.residency.end_sample();
        self.demand = Some(demand);
        demand[ResidencyClass::MeshPages as usize]
    }

    /// Lands built finer cuts, retires those unused past their hold (only once `settled`, the
    /// last frame by which every scene has reported what it holds, has passed their last
    /// touch), admits wanted ones, loads the pages of admitted ones and sends those whose
    /// pages are resident to be built.
    ///
    /// A cut is admitted whole, the larger error it would remove first, while the pages it
    /// would lock fit half the mesh page budget beside those of the cuts built and admitted,
    /// and what its loads add to the latest sample's demand keeps it, beside what those of the
    /// cuts admitted add, under [`LOW`] of the budget. Which of its pages it locks, and what
    /// they hold once built, shows only after they have all been resident: until then its loads
    /// alone admit it, and the bytes of all its pages stand for its locks beside the cuts after
    /// it. None is admitted under `pressure`, while the detail is coarsened for the pool: the
    /// traversal's cut comes first. An admitted cut asks for its pages and keeps them in their
    /// hold until it builds; it leaves admission when no copy has wanted it for a hold, or under
    /// pressure.
    fn fine_step(
        &mut self,
        frame: u64,
        settled: u64,
        pressure: bool,
        loading: &Arc<Loading>,
        w: &mut Apply<'_>,
    ) -> Result<Vec<TimelinePoint>> {
        let mut points = Vec::new();
        for batch in loading.take_cuts(w.device) {
            points.extend(batch.point);
            let Some(f) = self.fine.get_mut(&(batch.mount, batch.mesh)) else {
                continue;
            };
            let slot = batch.slot as usize;
            let at = f.init.table.offset(u64::from(batch.slot));
            let FineState::Building { locks } =
                std::mem::replace(&mut f.slots[slot], FineState::EMPTY)
            else {
                unreachable!("a slot has one build at a time")
            };
            let (record, cut) = match batch.cut {
                Ok(c) => c,
                Err(e) => {
                    if e.is_out_of_memory() {
                        // The slot stays empty, so a copy that still wants it asks again.
                        self.residency.out_of_memory(governs(&self.governed, &e));
                    } else {
                        log::warn!(
                            "finer cut {} of mesh {} of {:?} failed to build: {e}",
                            batch.slot,
                            batch.mesh,
                            batch.mount
                        );
                        f.slots[slot] = FineState::Failed;
                        self.residency.note_failed_cut();
                    }
                    let m = &self.mounts[&batch.mount];
                    for &p in &locks.pages {
                        let unit = m.mesh_pages[batch.mesh as usize][p as usize];
                        self.residency.unlock_while_used(Key {
                            mount: batch.mount,
                            unit,
                        });
                    }
                    self.fine_locked -= locks.bytes;
                    continue;
                }
            };
            w.put(at, &[record])?;
            f.slots[slot] = FineState::Built {
                cut,
                touched: frame,
                locks,
                shed: false,
            };
        }
        let budget = self
            .residency
            .stats()
            .class(ResidencyClass::MeshPages)
            .budget;
        // Cuts by mount, mesh and slot: admitted ones, and those wanted with their error.
        let mut admitted_cuts = Vec::new();
        let mut wanted_cuts = Vec::new();
        for (&(mount, mesh), f) in &mut self.fine {
            let m = &self.mounts[&mount];
            let unit = |p: u32| Key {
                mount,
                unit: m.mesh_pages[mesh as usize][p as usize],
            };
            for k in 0..f.slots.len() {
                let CutRow { threshold: t, row } = f.init.cuts[k];
                match &mut f.slots[k] {
                    FineState::Built { touched, .. }
                        if *touched + HOLD_FRAMES < frame && *touched < settled =>
                    {
                        let FineState::Built { cut, locks, .. } =
                            std::mem::replace(&mut f.slots[k], FineState::EMPTY)
                        else {
                            unreachable!("matched above")
                        };
                        let empty = RtCutGpu {
                            error: t,
                            row,
                            ..bytemuck::Zeroable::zeroed()
                        };
                        w.put(f.init.table.offset(k as u64), &[empty])?;
                        self.retired_cuts.push((frame - 1, cut));
                        for &p in &locks.pages {
                            self.residency.unlock_while_used(unit(p));
                        }
                        self.fine_locked -= locks.bytes;
                    }
                    FineState::Empty {
                        wanted: at,
                        admitted,
                        ..
                    } if *admitted => {
                        if pressure || *at + HOLD_FRAMES < frame {
                            *admitted = false;
                        } else {
                            admitted_cuts.push((mount, mesh, k));
                        }
                    }
                    FineState::Empty {
                        wanted: at, error, ..
                    } if *at != 0 && *at + 2 >= frame && !pressure => {
                        wanted_cuts.push((*error, mount, mesh, k));
                    }
                    _ => {}
                }
            }
        }
        // What the admitted cuts hold back of the lock share and add to the demand.
        let (mut locking, mut adding) = (0, 0);
        let mut seen = FxHashSet::default();
        admitted_cuts.retain(|&(mount, mesh, k)| {
            let (keys, bytes) = self.fine_pages(mount, mesh, k);
            match self
                .residency
                .unsampled(keys, &FxHashSet::default(), &mut seen)
            {
                Some(more) => {
                    locking += self.fine_lock_bytes(mount, mesh, k).unwrap_or(bytes);
                    adding += more;
                    true
                }
                None => {
                    self.fail_cut(mount, mesh, k);
                    false
                }
            }
        });
        let demand = self.demand.take();
        let shed = match demand {
            Some(d) => self.shed_cuts(d[ResidencyClass::MeshPages as usize], budget, frame, w)?,
            None => false,
        };
        if let Some(demand) = demand.filter(|_| !shed) {
            let room = (budget as f64 * LOW) as u64;
            // What the frame wants with the pages the built cuts hold beside it.
            let demand = demand[ResidencyClass::MeshPages as usize].bytes
                + self.residency.unsampled_held(ResidencyClass::MeshPages);
            wanted_cuts.sort_by(|a, b| {
                b.0.total_cmp(&a.0)
                    .then((a.1, a.2, a.3).cmp(&(b.1, b.2, b.3)))
            });
            for (_, mount, mesh, k) in wanted_cuts {
                let (keys, bytes) = self.fine_pages(mount, mesh, k);
                let mut mine = FxHashSet::default();
                let Some(more) = self.residency.unsampled(keys, &seen, &mut mine) else {
                    self.fail_cut(mount, mesh, k);
                    continue;
                };
                let lock = self.fine_lock_bytes(mount, mesh, k);
                let over = lock.is_some_and(|b| self.fine_locked + locking + b > budget / 2);
                if over || demand + adding + more > room {
                    continue;
                }
                locking += lock.unwrap_or(bytes);
                adding += more;
                seen.extend(mine);
                let f = self.fine.get_mut(&(mount, mesh)).expect("listed");
                if let FineState::Empty { admitted, .. } = &mut f.slots[k] {
                    *admitted = true;
                }
                admitted_cuts.push((mount, mesh, k));
            }
        }
        for (mount, mesh, k) in admitted_cuts {
            self.build_admitted(mount, mesh, k, budget, frame, loading);
        }
        Ok(points)
    }

    /// Sheds built cuts while what the sample's frame wanted, `demand`, does not fit the mesh
    /// page `budget` beside the pages the cuts hold, or the cuts hold more than their half of
    /// it: the traversal's pages come first. The cuts holding the most pages the frame did not
    /// read go first ([`Residency::shed`]). A shed cut's table entry is emptied, so copies
    /// trace a coarser cut, and the cut goes with its locks once no frame has used it for its
    /// hold. Returns whether any was shed.
    fn shed_cuts(
        &mut self,
        demand: Demand,
        budget: u64,
        frame: u64,
        w: &mut Apply<'_>,
    ) -> Result<bool> {
        let class = ResidencyClass::MeshPages;
        if demand.bytes + self.residency.unsampled_held(class) <= budget
            && self.fine_locked <= budget / 2
        {
            return Ok(false);
        }
        // Each built cut's locked pages and where it is, the shed ones apart. One past its hold
        // stays while a scene not updated since it was used may trace it, so neither shedding
        // it nor its having been shed returns anything until that scene reports again, and its
        // locks, which go then, are not for shedding other cuts to make up for.
        let (mut keys, mut cuts, mut at) = (Vec::new(), Vec::new(), Vec::new());
        let mut kept = 0;
        for (&(mount, mesh), f) in &self.fine {
            let pages = &self.mounts[&mount].mesh_pages[mesh as usize];
            for (k, slot) in f.slots.iter().enumerate() {
                let FineState::Built {
                    locks,
                    shed,
                    touched,
                    ..
                } = slot
                else {
                    continue;
                };
                if touched + HOLD_FRAMES < frame {
                    kept += locks.bytes;
                    continue;
                }
                let first = keys.len();
                keys.extend(locks.pages.iter().map(|&p| Key {
                    mount,
                    unit: pages[p as usize],
                }));
                cuts.push((*shed, first..keys.len(), locks.bytes));
                if !*shed {
                    at.push((mount, mesh, k));
                }
            }
        }
        let holders = |shed: bool| -> Vec<Holder<'_>> {
            (cuts.iter())
                .filter(|c| c.0 == shed)
                .map(|(_, units, bytes)| Holder {
                    units: &keys[units.clone()],
                    bytes: *bytes,
                })
                .collect()
        };
        let order = self.residency.shed(
            class,
            demand.bytes,
            budget,
            &holders(true),
            &holders(false),
            self.fine_locked - kept,
        );
        for &i in &order {
            let (mount, mesh, k) = at[i];
            let f = self.fine.get_mut(&(mount, mesh)).expect("listed");
            let CutRow { threshold, row } = f.init.cuts[k];
            let empty = RtCutGpu {
                error: threshold,
                row,
                ..bytemuck::Zeroable::zeroed()
            };
            w.put(f.init.table.offset(k as u64), &[empty])?;
            if let FineState::Built { shed, .. } = &mut f.slots[k] {
                *shed = true;
            }
        }
        Ok(!order.is_empty())
    }

    /// The keys of the pages finer cut `k` of mesh `mesh` of `mount` is built from, and the
    /// bytes they hold.
    fn fine_pages(&self, mount: MountId, mesh: u32, k: usize) -> (Vec<Key>, u64) {
        let m = &self.mounts[&mount];
        let keys: Vec<Key> = self.fine[&(mount, mesh)].init.pages[k]
            .iter()
            .map(|&p| Key {
                mount,
                unit: m.mesh_pages[mesh as usize][p as usize],
            })
            .collect();
        let bytes = keys.iter().map(|&k| self.residency.held(k)).sum();
        (keys, bytes)
    }

    /// The bytes finer cut `k` of mesh `mesh` of `mount` locks once built, when its pages have
    /// all been resident.
    fn fine_lock_bytes(&self, mount: MountId, mesh: u32, k: usize) -> Option<u64> {
        let m = &self.mounts[&mount];
        let pages = self.fine[&(mount, mesh)].locks[k].as_ref()?;
        let held = |&p: &u32| {
            self.residency.held(Key {
                mount,
                unit: m.mesh_pages[mesh as usize][p as usize],
            })
        };
        Some(pages.iter().map(held).sum())
    }

    /// Finer cut `k` of mesh `mesh` of `mount` never builds: a page of it failed to load.
    fn fail_cut(&mut self, mount: MountId, mesh: u32, k: usize) {
        let f = self.fine.get_mut(&(mount, mesh)).expect("listed");
        f.slots[k] = FineState::Failed;
        self.residency.note_failed_cut();
    }

    /// Asks for the absent pages of admitted finer cut `k` of mesh `mesh` of `mount` and keeps
    /// the resident ones in their hold at `frame`; once they are all resident, locks the ones
    /// holding its clusters and sends it to be built, unless those do not fit half of the mesh
    /// page `budget` beside the locks of the other cuts, which takes back its admission.
    fn build_admitted(
        &mut self,
        mount: MountId,
        mesh: u32,
        k: usize,
        budget: u64,
        frame: u64,
        loading: &Arc<Loading>,
    ) {
        let m = &self.mounts[&mount];
        let unit = |p: u32| Key {
            mount,
            unit: m.mesh_pages[mesh as usize][p as usize],
        };
        let f = self.fine.get_mut(&(mount, mesh)).expect("listed");
        let FineState::Empty { error, .. } = f.slots[k] else {
            unreachable!("an admitted cut is not built")
        };
        let mut absent = false;
        for &q in &f.init.pages[k] {
            if self.residency.resident(unit(q)) {
                self.residency.touch(unit(q), frame);
            } else {
                absent = true;
                let p = Priority {
                    error,
                    cell_distance: 0,
                };
                self.residency.request(unit(q), p, RequestSource::FinerCuts);
            }
        }
        if absent {
            return;
        }
        // A page lists its clusters here in the step the residency takes it as resident, and
        // the cut's clusters sit in these pages.
        let CutRow { threshold: t, row } = f.init.cuts[k];
        let stream = &f.init.stream;
        let (mut clusters, mut members) = (Vec::new(), Vec::new());
        let mut pages = Vec::new();
        let mut error = 0.0f32;
        let held = (0..stream.pinned).chain(f.init.pages[k].iter().copied());
        for q in held {
            let r = f.init.resident[q as usize]
                .as_ref()
                .expect("a resident page's clusters");
            let first = stream.pages[q as usize].first;
            let before = clusters.len();
            for ((c, &clas), i) in r.infos.iter().zip(&r.clas).zip(first..) {
                if c.in_cut(t, f.init.merge) {
                    clusters.push(i);
                    members.push(CutMember {
                        clas,
                        part: f.init.animated.index(c),
                    });
                    error = error.max(c.self_error);
                }
            }
            if q >= stream.pinned && clusters.len() > before {
                pages.push(q);
            }
        }
        if clusters.is_empty() {
            return;
        }
        let bytes = pages.iter().map(|&q| self.residency.held(unit(q))).sum();
        f.locks[k] = Some(pages.clone());
        if self.fine_locked + bytes > budget / 2 {
            f.slots[k] = FineState::EMPTY;
            return;
        }
        for &q in &pages {
            self.residency.lock_while_used(unit(q));
        }
        self.fine_locked += bytes;
        f.slots[k] = FineState::Building {
            locks: Locks { pages, bytes },
        };
        loading.send(Msg::Cut(CutTask {
            mount,
            mesh,
            slot: k as u32,
            accel: Arc::clone(&self.accel),
            clas: self.clas.clone(),
            cut: Cut {
                error,
                clusters,
                row,
            },
            members,
            parts: f.init.animated.count,
        }));
    }

    /// Hands the texture feedback of `frame`, read back in slot `slot`, to the next step. The
    /// residency map holds a cell at the coarsest of itself and its eight neighbors, so a
    /// wanted mip asks for the tiles of that mip over the cell and its neighbors, each once a
    /// frame: a resident one is touched; an absent one is requested, with the mips it would add
    /// below the finest resident tile over it as its error, and that tile, which the cells read
    /// meanwhile, is touched.
    pub(crate) fn texture_feedback(&mut self, slot: usize, frame: u64) {
        let (wanted, lost) = self.feedback.read(slot);
        self.residency.note_texture_feedback_lost(lost);
        for ((mount, texture), cell, mip) in wanted {
            let Some(t) = self
                .mounts
                .get_mut(&mount)
                .and_then(|m| m.textures.get_mut(texture as usize))
            else {
                continue;
            };
            let g = &t.stream.grid;
            let [cx, cy] = g.cells();
            let [gx, gy] = g.grids[mip as usize];
            let (layer, c) = (cell / (cx * cy), cell % (cx * cy));
            let (x, y) = (c % cx, c / cx);
            let span = |at: u32, cells: u32, tiles: u32| {
                let tile = |n: u32| (n >> mip).min(tiles - 1);
                tile(at.saturating_sub(1))..=tile((at + 1).min(cells - 1))
            };
            for ty in span(y, cy, gy) {
                for tx in span(x, cx, gx) {
                    let xy = [tx, ty];
                    let tile = g.index_of(Tile { mip, layer, xy });
                    if t.heard[tile as usize] == frame {
                        continue;
                    }
                    t.heard[tile as usize] = frame;
                    let key = |tile: u32| Key {
                        mount,
                        unit: t.first_unit + tile,
                    };
                    if t.resident[tile as usize] {
                        self.residency.touch(key(tile), frame);
                        continue;
                    }
                    let cell = [tx << mip, ty << mip];
                    let mut above = mip + 1;
                    while above < g.tail_mip()
                        && !t.resident[g.index_of(g.covering(layer, above, cell)) as usize]
                    {
                        above += 1;
                    }
                    let p = Priority {
                        error: (above - mip) as f32,
                        cell_distance: 0,
                    };
                    self.residency
                        .request(key(tile), p, RequestSource::Textures);
                    if above < g.tail_mip() {
                        let over = g.index_of(g.covering(layer, above, cell));
                        self.residency.touch(key(over), frame);
                    }
                }
            }
        }
    }

    /// Queues `blob` of `source` into `target`. It loads once the user blob budget has room
    /// and the in-flight limit has room for it beside the shares of the other requests, in
    /// request order.
    pub(crate) fn request_blob(
        &mut self,
        source: &Arc<BlobSource>,
        blob: Blob,
        target: BlobTarget,
    ) -> BlobId {
        let id = match self.blobs.iter().position(Option::is_none) {
            Some(i) => i,
            None => {
                self.blobs.push(None);
                self.blobs.len() - 1
            }
        };
        self.blobs[id] = Some(UserBlob {
            state: BlobState::Queued(Box::new(BlobRequest {
                source: Arc::clone(source),
                blob,
                target,
            })),
            bytes: u64::from(blob.raw_len),
        });
        self.queued.push_back(id as u32);
        BlobId(id as u32)
    }

    pub(crate) fn blob_state(&self, id: BlobId) -> LoadState {
        match self.blobs.get(id.0 as usize).and_then(Option::as_ref) {
            Some(b) => match &b.state {
                BlobState::Queued(_) | BlobState::Loading { released: false } => LoadState::Pending,
                BlobState::Ready => LoadState::Ready,
                BlobState::Failed(e) => LoadState::Failed(Arc::clone(e)),
                BlobState::Loading { released: true } => {
                    LoadState::Failed(Arc::new(Error::UnknownBlob(id)))
                }
            },
            None => LoadState::Failed(Arc::new(Error::UnknownBlob(id))),
        }
    }

    /// Forgets blob `id` and returns its bytes to the user blob budget. A queued blob is
    /// not loaded; one loading lands and is then dropped.
    pub(crate) fn release_blob(&mut self, id: BlobId) {
        let Some(slot) = self.blobs.get_mut(id.0 as usize) else {
            return;
        };
        let Some(b) = slot.as_mut() else { return };
        match b.state {
            BlobState::Queued(_) => {
                *slot = None;
                self.queued.retain(|&q| q != id.0);
            }
            BlobState::Ready => {
                self.residency.release_blob(b.bytes);
                *slot = None;
            }
            BlobState::Loading { .. } => b.state = BlobState::Loading { released: true },
            BlobState::Failed(_) => *slot = None,
        }
    }

    /// Step 1 of frame `frame`: applies the landed batches, plans this step and sends its
    /// loads. Under `pressure` the detail is coarsened for the mesh page pool. `pixels` are
    /// the render pixels of the views of the frame before. Returns the points the frame's
    /// submit waits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step(
        &mut self,
        frame: u64,
        settled: u64,
        pressure: bool,
        pixels: u64,
        output: Option<TimelinePoint>,
        loading: &Arc<Loading>,
        w: &mut Apply<'_>,
    ) -> Result<Vec<TimelinePoint>> {
        let mut points = self.finish_tile_moves(frame, output, w.device)?;
        for mut batch in loading.take_streams(w.device) {
            points.extend(batch.points);
            self.carried.append(&mut batch.unbinds);
            match batch.error {
                Some(e) if e.is_out_of_memory() => {
                    self.residency.out_of_memory(governs(&self.governed, &e));
                }
                Some(e) => log::warn!("a stream task's GPU work failed: {e}"),
                None => {}
            }
            let mut staging = self.staging.lock().unwrap();
            if let Some(d) = batch.decode {
                d.fail_pages(&mut batch.loads);
                staging.free(d.slot);
            }
            for l in &mut batch.loads {
                if let Some(Staged::Pool(s)) = l.staging.take() {
                    staging.free(s);
                }
            }
            drop(staging);
            let mut clas = Vec::new();
            if let Some(mut c) = batch.clas {
                match c.take_shortfall() {
                    Some(Shortfall::Memory(e)) => {
                        self.residency.out_of_memory(governs(&self.governed, &e));
                    }
                    // The range is as large as the mesh page budget, but beside the CLAS that
                    // budget counts it holds those of evicted pages until their frees apply
                    // and the pages reserved for batches: that budget alone shrinks.
                    Some(Shortfall::Range) => {
                        self.residency.range_full(ResidencyClass::MeshPages);
                    }
                    None => {}
                }
                clas = self.replay(c);
            }
            self.note_failures(&batch.loads);
            self.residency
                .note_read(batch.loads.iter().map(|l| l.read).sum());
            // A load lands after the loads of its batch it depends on.
            let mut loads = batch.loads;
            while !loads.is_empty() {
                let pending: FxHashSet<Key> = loads
                    .iter()
                    .filter_map(|l| match l.what {
                        What::Unit(k) => Some(k),
                        What::Blob(_) => None,
                    })
                    .collect();
                let (ready, later): (Vec<Landed>, Vec<Landed>) =
                    loads.into_iter().partition(|l| match l.what {
                        What::Unit(k) => !self.depends_on(k, &pending),
                        What::Blob(_) => true,
                    });
                for l in ready {
                    self.landed(l, &clas, frame, w)?;
                }
                loads = later;
            }
        }
        if let Some(c) = &self.clas {
            let (moved, bound) = c.finish_moves(w.device, output)?;
            points.extend(bound);
            self.residency.note_clas_pages_moved(moved);
        }
        // A carried unbind's key left `unbinding` when its failed task was sent, so a later
        // task may have loaded the tile again, binding it over the old slot, or a later
        // eviction may have queued its own unbind: then only the slot returns. While such a
        // load is in flight its bind decides.
        for u in std::mem::take(&mut self.carried) {
            let live = self.mounts.contains_key(&u.key.mount);
            if live && self.residency.loading(u.key) {
                self.carried.push(u);
            } else if (live && self.residency.resident(u.key)) || self.unbinding.contains(&u.key) {
                self.tiles.lock().unwrap().free_retired(u.slot);
            } else {
                // A load of the tile waits for the task that unbinds it, as after an eviction.
                self.unbinding.insert(u.key);
                self.unbinds.push(u);
            }
        }
        let slot = (frame % u64::from(self.fif)) as usize;
        let period = response_period(self.fif);
        if let Some(e) = self.feedback.begin(frame, slot, period, pixels, w)? {
            self.residency.out_of_memory(governs(&self.governed, &e));
        }
        for index in std::mem::take(&mut self.nulls) {
            w.put(self.streams.offset(u64::from(index)), &[GpuPtr::NULL])?;
        }
        points.extend(self.fine_step(frame, settled, pressure, loading, w)?);
        let report = w.device.memory_report();
        // Over its mark a heap's pools empty every block they can; under it they keep a
        // block's room for loads.
        let over = |h: usize| report.heaps[h].usage > report.heaps[h].budget / 10 * MARK_TENTHS;
        let mut mine: Vec<[u64; 2]> = (0..report.heaps.len())
            .map(|h| [self.residency.held_on(h), 0])
            .collect();
        for (pool, &heap) in [&self.pool, &self.units].into_iter().zip(&self.pool_heaps) {
            mine[heap][1] += pool.lock().unwrap().plan_drain(over(heap));
        }
        if let Some(c) = &self.clas {
            mine[self.pool_heaps[0]][1] += c.draining();
        }
        if self.moves_tiles {
            let heap = self.pool_heaps[0];
            mine[heap][1] += self.tiles.lock().unwrap().plan_drain(over(heap));
        }
        let heaps: Vec<HeapUse> = report
            .heaps
            .iter()
            .zip(self.share.post(&mine))
            .map(|(h, [held, coming])| HeapUse {
                usage: h.usage,
                budget: h.budget,
                coming: h.freeing + coming,
                held,
            })
            .collect();
        self.residency.react(frame, &heaps);
        let staging = report.heap_of(MemoryKind::Staging);
        if staging.usage > staging.budget / 10 * MARK_TENTHS {
            self.staging.lock().unwrap().release_spare();
        }
        if let Some(c) = &self.clas {
            let (bound, used, range_full) = c.end_step(over(self.pool_heaps[0]));
            self.residency.note_clas_pool(bound, used, range_full);
        }
        self.moving.retain(|_, m| m.1 >= frame);
        let waiting: Vec<Key> = self.moving.keys().copied().collect();
        for k in waiting {
            self.residency.request(
                k,
                Priority {
                    error: 0.0,
                    cell_distance: 0,
                },
                RequestSource::Textures,
            );
        }
        let queued = self
            .queued
            .iter()
            .map(|&id| self.blobs[id as usize].as_ref().expect("queued blob").bytes);
        let Plan {
            loads,
            evictions,
            blobs,
        } = self.residency.plan(frame, queued);
        let blob_loads: Vec<StreamLoad> = self
            .queued
            .drain(..blobs)
            .map(|id| {
                let b = self.blobs[id as usize].as_mut().expect("queued blob");
                let state = std::mem::replace(&mut b.state, BlobState::Loading { released: false });
                let BlobState::Queued(r) = state else {
                    unreachable!("a queued blob is queued")
                };
                StreamLoad {
                    what: What::Blob(id),
                    source: r.source,
                    kind: LoadKind::User {
                        blob: r.blob,
                        target: r.target,
                    },
                }
            })
            .collect();
        for k in &evictions {
            let m = &self.mounts[&k.mount];
            if let Some(StreamUnit::Tile { texture, .. }) = m.units[k.unit as usize] {
                self.fragmented
                    .insert((k.mount, texture), frame + HOLD_FRAMES);
            }
        }
        for k in evictions {
            self.evict_placed(k, frame, w)?;
        }
        self.compact(frame, w)?;
        self.drain(frame, w)?;
        self.start_tile_moves(w.device)?;
        for m in self.mounts.values_mut() {
            for t in &mut m.textures {
                t.write_map(w)?;
            }
        }
        let (ids, device) = (&self.accel.clusters, w.device);
        let mut taken = FxHashMap::default();
        let mut failed = None;
        let loads = self.residency.cancel_held(loads, |k| {
            if self.unbinding.contains(&k) {
                return Some(Held::Waiting);
            }
            let m = &self.mounts[&k.mount];
            let Some(StreamUnit::MeshPage { mesh, page }) = m.units[k.unit as usize] else {
                return None;
            };
            let n = m.mesh_streams[mesh as usize].pages[page as usize].clusters;
            match ids.alloc(device, n) {
                Ok(Some(first)) => {
                    taken.insert(k, (first, n));
                    None
                }
                Ok(None) => Some(Held::ClusterIds),
                Err(e) => {
                    failed.get_or_insert(e);
                    Some(Held::Waiting)
                }
            }
        });
        self.loading_ids.extend(taken);
        // The loads kept go out as tasks before another error returns, so none stays loading
        // without one.
        let failed = match failed {
            Some(e) if e.is_out_of_memory() => {
                self.residency.out_of_memory(governs(&self.governed, &e));
                None
            }
            failed => failed,
        };
        for u in &self.unbinds {
            self.unbinding.remove(&u.key);
        }
        let moving = std::mem::take(&mut self.moving);
        let mut loads: Vec<StreamLoad> = loads
            .into_iter()
            .map(|key| {
                let m = &self.mounts[&key.mount];
                let kind = match m.units[key.unit as usize].expect("a streamed unit") {
                    StreamUnit::Pool {
                        entry,
                        blob,
                        bytes,
                        target,
                    } => LoadKind::Pool {
                        entry,
                        blob,
                        bytes,
                        target,
                    },
                    StreamUnit::Tile { texture, tile } => LoadKind::Tile {
                        stream: Arc::clone(&m.textures[texture as usize].stream),
                        tile,
                        after: moving.get(&key).map(|m| m.0),
                    },
                    StreamUnit::MeshPage { mesh, page } => LoadKind::MeshPage {
                        stream: Arc::clone(&m.mesh_streams[mesh as usize]),
                        page,
                        ids: self.loading_ids[&key].0,
                    },
                };
                StreamLoad {
                    what: What::Unit(key),
                    source: Arc::clone(&m.source),
                    kind,
                }
            })
            .collect();
        // A tile still waiting on its unbind keeps asking.
        self.moving = moving
            .into_iter()
            .filter(|(k, _)| self.residency.absent(*k))
            .collect();
        loads.extend(blob_loads);
        let mut tasks = split_tasks(loads, self.clas.is_some());
        let clas_waits = self.clas.as_ref().is_some_and(|c| c.pending(w.device));
        if tasks.is_empty() && (!self.unbinds.is_empty() || clas_waits) {
            tasks.push(Vec::new());
        }
        // The first task unbinds, before any later one binds a tile again.
        let mut unbinds = std::mem::take(&mut self.unbinds);
        let near = Arc::new(Mutex::new(FxHashMap::default()));
        for loads in tasks {
            let task = StreamTask {
                accel: Arc::clone(&self.accel),
                clas: self.clas.clone(),
                pool: Arc::clone(&self.pool),
                units: Arc::clone(&self.units),
                staging: Arc::clone(&self.staging),
                tiles: Arc::clone(&self.tiles),
                device: w.device.clone(),
                outstanding: AtomicUsize::new(1),
                done: Mutex::new(Vec::with_capacity(loads.len())),
                copies: Mutex::new(Vec::new()),
                decodes: Mutex::new(Vec::new()),
                unbinds: std::mem::take(&mut unbinds),
                near: Arc::clone(&near),
                loads,
            };
            loading.send(Msg::Stream(Arc::new(task)));
        }
        // Landings and the plan may have made room for mount reads.
        loading.wake_mounts();
        failed.map_or(Ok(points), Err)
    }

    /// Gives a stream pool unit's target its address, 0 when the unit leaves.
    fn point_target(&mut self, target: PoolTarget, at: u64, w: &mut Apply<'_>) -> Result<()> {
        match target {
            PoolTarget::Table(entry) => w.put(self.mesh_pages.table.entry(entry), &[at])?,
            PoolTarget::InstancePage(page) => self.page_events.push((page, at)),
            PoolTarget::PageWindow { page, window } => {
                self.window_events.push((page, window, at));
            }
        }
        Ok(())
    }

    /// Writes evicted unit `k` out of its tables; its memory returns once the frame before
    /// has completed.
    fn evict_placed(&mut self, k: Key, frame: u64, w: &mut Apply<'_>) -> Result<()> {
        match self.resident.remove(&k).expect("evicted unit is resident") {
            Placed::Pool(slot, target) => {
                self.point_target(target, 0, w)?;
                self.retire_unit(k, slot, frame - 1);
            }
            Placed::MeshPage(p) => {
                let (entry, refs, count) = p.entries();
                let zeros = vec![bytemuck::Zeroable::zeroed(); count as usize];
                let table = &self.mesh_pages.table;
                w.put(table.entry(entry), &[0])?;
                w.put(refs, &zeros)?;
                if let Some(f) = self.fine.get_mut(&(k.mount, p.stream.mesh)) {
                    f.init.resident[p.page as usize] = None;
                }
                for mirror in p.stream.mirrors.lock().unwrap().iter() {
                    if !mirror.own[p.page as usize] {
                        w.put(table.entry(mirror.entries + p.page), &[0])?;
                    }
                }
                let (slot, free) = p.free();
                self.retire_page(k, slot, frame - 1);
                self.retired_pages.push((frame - 1, free));
            }
            Placed::Tile(slot) => {
                let m = self.mounts.get_mut(&k.mount).expect("mounted");
                let (t, tile) = m.tile(k.unit);
                t.resident[tile as usize] = false;
                t.dirty = true;
                let free = TileFree {
                    key: k,
                    stream: Arc::clone(&t.stream),
                    tile,
                    slot,
                };
                self.unbinding.insert(k);
                self.tile_blocks.remove(slot.block, k);
                let mut pool = self.tiles.lock().unwrap();
                pool.retire(slot);
                // A tile leaving mid-move stays bound to `slot`.
                if let Some(moves) = &mut self.tile_moves
                    && let Some(i) = moves.tiles.iter().position(|t| t.key == k)
                {
                    pool.free(moves.tiles.swap_remove(i).to);
                }
                // This frame's list may still copy the texels of a tile just bound back in.
                let last_use = if self.tile_finished.contains(&k) {
                    frame
                } else {
                    frame - 1
                };
                self.retired_tiles.push((last_use, free));
            }
        }
        Ok(())
    }

    /// Binds the tiles a step started moving to their new slots, on the render queue after the
    /// submitted frame whose list copied their texels out and after `output`, the work of
    /// that frame on another queue, and queues the copies of their texels back in. Their old
    /// slots return once this frame has completed. The moves wait for a later step while no
    /// submitted list has copied them out, or while the loading thread is binding. Returns the
    /// points this frame's work waits for.
    fn finish_tile_moves(
        &mut self,
        frame: u64,
        output: Option<TimelinePoint>,
        device: &Device,
    ) -> Result<Vec<TimelinePoint>> {
        let mut points = Vec::new();
        #[cfg(feature = "test-hooks")]
        if self.tile_binds_held {
            return Ok(points);
        }
        let Some(moves) = self.tile_moves.as_ref().filter(|m| m.copied) else {
            return Ok(points);
        };
        let mut per_texture: Vec<(Arc<TexStream>, TileSlots)> = Vec::new();
        for t in &moves.tiles {
            let m = self
                .mounts
                .get_mut(&t.key.mount)
                .expect("a moving tile's mount");
            let (texture, tile) = m.tile(t.key.unit);
            match per_texture
                .iter_mut()
                .find(|(s, _)| Arc::ptr_eq(s, &texture.stream))
            {
                Some((_, list)) => list.push((tile, t.to)),
                None => per_texture.push((Arc::clone(&texture.stream), vec![(tile, t.to)])),
            }
        }
        let mut pool = self.tiles.lock().unwrap();
        if !per_texture.is_empty() {
            let lists: Vec<(&TexStream, Vec<TileBind<'_>>)> = per_texture
                .iter()
                .map(|(s, list)| {
                    let binds = list
                        .iter()
                        .map(|&(t, to)| {
                            let tile = s.grid.tile(t);
                            TileBind {
                                mip: tile.mip,
                                layer: tile.layer,
                                tile: tile.xy,
                                memory: Some(pool.memory(to)),
                            }
                        })
                        .collect();
                    (&**s, binds)
                })
                .collect();
            let mut binds = SparseBinds::new();
            for (s, list) in &lists {
                binds.tiles(&s.texture, list)?;
            }
            let render = device.queue(QueueKind::Render).expect("render queue");
            // A bind follows the work submitted to its queue before it and nothing else, so
            // the other queue's work goes ahead of it through a list that waits for it.
            if let Some(output) = output.filter(|&p| !device.is_complete(p)) {
                points.push(render.submit(vec![render.command_list()?], &[output])?);
            }
            let Some(point) = device.try_bind_sparse_on(QueueKind::Render, &binds)? else {
                return Ok(points);
            };
            points.push(point);
        }
        let TileMoves { scratch, tiles, .. } = self.tile_moves.take().expect("checked above");
        let base = scratch.gpu().address().0;
        self.retired_scratch.push((frame, scratch));
        let mut copies = Moves::default();
        for t in tiles {
            let m = self
                .mounts
                .get_mut(&t.key.mount)
                .expect("a moving tile's mount");
            let (texture, tile) = m.tile(t.key.unit);
            copies.tiles.push(TileCopy {
                stream: Arc::clone(&texture.stream),
                tile,
                at: base + t.at,
                out: false,
            });
            let Some(Placed::Tile(from)) = self.resident.insert(t.key, Placed::Tile(t.to)) else {
                unreachable!("an evicted tile leaves its move")
            };
            self.tile_blocks.remove(from.block, t.key);
            self.tile_blocks.insert(t.to.block, t.key);
            self.tile_finished.insert(t.key);
            pool.retire(from);
            self.retired_tile_slots.push((frame, from));
        }
        self.moves.push(copies);
        Ok(points)
    }

    /// Starts moving resident tiles out of the tile pool's draining blocks, [`DRAIN_TILES`] at
    /// most: each takes a slot in a block that is not draining, and a frame copies its texels
    /// to scratch memory. [`Self::finish_tile_moves`] binds them there. Without memory for
    /// the scratch no tile moves, and the budgets hear of it.
    fn start_tile_moves(&mut self, device: &Device) -> Result<()> {
        if self.tile_moves.is_some() {
            return Ok(());
        }
        let tile_bytes = orr_pack::texture::TILE_BYTES as u64;
        let mut pool = self.tiles.lock().unwrap();
        let mut tiles: Vec<TileMove> = Vec::new();
        'blocks: for &block in pool.draining().to_vec().iter() {
            let Some(keys) = self.tile_blocks.0.get(block as usize) else {
                continue;
            };
            // A tile whose copy back is still queued moves again once that has run.
            for &key in keys.iter().filter(|k| !self.tile_finished.contains(k)) {
                if tiles.len() == DRAIN_TILES {
                    break 'blocks;
                }
                let Some((to, _)) = pool.alloc_moved(tile_bytes, block) else {
                    continue 'blocks;
                };
                let at = tiles.len() as u64 * tile_bytes;
                tiles.push(TileMove { key, to, at });
            }
        }
        if tiles.is_empty() {
            return Ok(());
        }
        let bytes = tiles.len() as u64 * tile_bytes;
        let scratch = device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, bytes, "tile moves");
        #[cfg(feature = "test-hooks")]
        let scratch = scratch.and_then(|s| match self.tile_scratch_fails {
            true => Err(orr_gpu::Error::OutOfMemory {
                call: "tile moves",
                device: true,
                kind: Some(MemoryKind::DeviceLocal),
            }),
            false => Ok(s),
        });
        let scratch = match scratch {
            Ok(s) => s,
            Err(e) => {
                for t in tiles {
                    pool.free(t.to);
                }
                let e = Error::from(e);
                if !e.is_out_of_memory() {
                    return Err(e);
                }
                self.residency.out_of_memory(governs(&self.governed, &e));
                return Ok(());
            }
        };
        drop(pool);
        let base = scratch.gpu().address().0;
        let mut copies = Moves::default();
        for t in &tiles {
            let m = self
                .mounts
                .get_mut(&t.key.mount)
                .expect("a resident tile's mount");
            let (texture, tile) = m.tile(t.key.unit);
            copies.tiles.push(TileCopy {
                stream: Arc::clone(&texture.stream),
                tile,
                at: base + t.at,
                out: true,
            });
        }
        self.moves.push(copies);
        self.tile_moves = Some(TileMoves {
            scratch,
            tiles,
            copied: false,
        });
        Ok(())
    }

    /// Takes mesh page `k` out of stream pool slot `slot`, which returns once `frame` has
    /// completed.
    fn retire_page(&mut self, k: Key, slot: PoolSlot, frame: u64) {
        self.page_blocks.remove(slot.block, k);
        self.pool.lock().unwrap().retire(slot);
        self.retired.push((frame, slot));
    }

    /// Takes pool unit `k` out of `units` slot `slot`, which returns once `frame` has
    /// completed.
    fn retire_unit(&mut self, k: Key, slot: PoolSlot, frame: u64) {
        self.unit_blocks.remove(slot.block, k);
        self.units.lock().unwrap().retire(slot);
        self.retired_units.push((frame, slot));
    }

    /// Moves units out of the stream pool's and the unit pool's draining blocks, the block
    /// with the fewest live units first, [`DRAIN_BYTES`] of each pool at most: each takes a slot
    /// in a block that is not draining, its table entries point there from this frame on, and
    /// its old slot returns once this frame, whose copy reads it, has completed. A block one of
    /// whose units finds no room stops draining.
    fn drain(&mut self, frame: u64, w: &mut Apply<'_>) -> Result<()> {
        let mut units = Vec::new();
        for pages in [true, false] {
            let pool = Arc::clone(if pages { &self.pool } else { &self.units });
            let blocks = pool.lock().unwrap().draining().to_vec();
            let mut moved = 0;
            'blocks: for block in blocks {
                loop {
                    let index = if pages {
                        &self.page_blocks
                    } else {
                        &self.unit_blocks
                    };
                    let Some(k) = index.any(block) else { break };
                    if moved >= DRAIN_BYTES {
                        break 'blocks;
                    }
                    let (bytes, to) = {
                        let mut pool = pool.lock().unwrap();
                        let bytes = match &self.resident[&k] {
                            Placed::MeshPage(p) => pool.bytes(p.slot),
                            Placed::Pool(slot, _) => pool.bytes(*slot),
                            Placed::Tile(_) => unreachable!("tiles are in their own pool"),
                        };
                        (bytes, pool.alloc_moved(bytes, block))
                    };
                    let Some((to, at)) = to else { break };
                    moved += bytes;
                    units.push(self.move_unit(k, to, at.gpu(), bytes, frame, w)?);
                }
            }
        }
        if !units.is_empty() {
            let tiles = Vec::new();
            self.moves.push(Moves { units, tiles });
        }
        Ok(())
    }

    /// Moves resident mesh page or pool unit `k` to slot `to` at address `at`; returns the
    /// copy that takes it there.
    fn move_unit(
        &mut self,
        k: Key,
        to: PoolSlot,
        at: u64,
        bytes: u64,
        frame: u64,
        w: &mut Apply<'_>,
    ) -> Result<UnitMove> {
        let src = match self.resident.remove(&k).expect("a resident unit") {
            Placed::Pool(slot, target) => {
                let from = self.units.lock().unwrap().address(slot);
                self.point_target(target, at, w)?;
                self.retire_unit(k, slot, frame);
                self.unit_blocks.insert(to.block, k);
                self.resident.insert(k, Placed::Pool(to, target));
                from
            }
            Placed::MeshPage(mut p) => {
                let from = p.address;
                let (entry, refs, count) = p.entries();
                let table = &self.mesh_pages.table;
                w.put(table.entry(entry), &[at])?;
                // Each cluster's entry names the page first; the rest of it stays.
                let stride = (size_of::<ClusterRef>() / 4) as u32;
                for (word, value) in [(0, at as u32), (4, (at >> 32) as u32)] {
                    w.records.push(Record {
                        seq: w.seq.fetch_add(1, Ordering::Relaxed),
                        dst: refs.address().0 + word,
                        count,
                        stride,
                        op: Op::Fill(value),
                    });
                }
                for mirror in p.stream.mirrors.lock().unwrap().iter() {
                    if !mirror.own[p.page as usize] {
                        w.put(table.entry(mirror.entries + p.page), &[at])?;
                    }
                }
                self.retire_page(k, p.slot, frame);
                p.slot = to;
                p.address = at;
                self.page_blocks.insert(to.block, k);
                self.resident.insert(k, Placed::MeshPage(p));
                from
            }
            Placed::Tile(_) => unreachable!("tiles are in their own pool"),
        };
        Ok(UnitMove {
            dst: at,
            src,
            bytes,
        })
    }

    #[cfg(feature = "test-hooks")]
    pub(crate) fn hold_tile_binds(&mut self, hold: bool) {
        self.tile_binds_held = hold;
    }

    #[cfg(feature = "test-hooks")]
    pub(crate) fn fail_tile_scratch(&mut self, fail: bool) {
        self.tile_scratch_fails = fail;
    }

    #[cfg(feature = "test-hooks")]
    pub(crate) fn hold_clas_binds(&self, hold: bool) {
        if let Some(c) = &self.clas {
            c.hold_binds(hold);
        }
    }

    #[cfg(feature = "test-hooks")]
    pub(crate) fn plan_clas_memory_tight(&self, tight: bool) {
        if let Some(c) = &self.clas {
            c.plan_tight(tight);
        }
    }

    /// The copies of the units and tiles the steps moved that no submitted list has carried,
    /// oldest step first, for the frame being begun to run before anything reads the scenes'
    /// tables or samples a texture.
    pub(crate) fn moves(&self) -> &[Moves] {
        &self.moves
    }

    /// The list that carried [`Self::moves`] was submitted: the copies have their place in
    /// the render queue's order.
    pub(crate) fn submitted(&mut self) {
        self.moves.clear();
        self.tile_finished.clear();
        if let Some(moves) = &mut self.tile_moves {
            moves.copied = true;
        }
    }

    /// Moves the tiles of the textures that lost one and sit apart from the rest beside them:
    /// a tile unused for a few frames leaves, its map coarsening, and loads again after its
    /// texture's largest run of adjacent slots with room.
    fn compact(&mut self, frame: u64, w: &mut Apply<'_>) -> Result<()> {
        self.fragmented.retain(|_, until| *until >= frame);
        let mut left = MOVES_PER_STEP.saturating_sub(self.moving.len());
        let textures: Vec<(MountId, u32)> = self.fragmented.keys().copied().collect();
        for (mount, texture) in textures {
            if left == 0 {
                break;
            }
            let t = &self.mounts[&mount].textures[texture as usize];
            let first_unit = t.first_unit;
            let slots: Vec<(u32, PoolSlot)> = (0..t.resident.len() as u32)
                .filter_map(|tile| {
                    let key = Key {
                        mount,
                        unit: first_unit + tile,
                    };
                    match self.resident.get(&key) {
                        Some(Placed::Tile(slot)) => Some((tile, *slot)),
                        _ => None,
                    }
                })
                .collect();
            let residency = &self.residency;
            let pool = self.tiles.lock().unwrap();
            let chosen = compaction_move(
                &slots,
                |tile| {
                    let key = Key {
                        mount,
                        unit: first_unit + tile,
                    };
                    residency.movable(key, frame, MOVE_COLD_FRAMES)
                },
                |slot| pool.free_after(slot),
            );
            drop(pool);
            let Some((tile, after)) = chosen else {
                continue;
            };
            let key = Key {
                mount,
                unit: first_unit + tile,
            };
            self.residency.evict(key);
            self.evict_placed(key, frame, w)?;
            self.moving.insert(key, (after, frame + HOLD_FRAMES));
            self.residency.note_tile_move();
            left -= 1;
        }
        Ok(())
    }

    /// Replays a landed batch's CLAS work on the pool's reference; returns the unit and the
    /// unit count of each of its CLAS.
    fn replay(&self, batch: PoolBatch) -> Vec<(u32, u32)> {
        let pool = self.clas.as_ref().expect("a CLAS batch has its pool");
        let (placed, held) = pool.replay(batch);
        let mut staging = self.staging.lock().unwrap();
        for slot in held {
            staging.free(slot);
        }
        placed
    }

    /// Whether `k` depends on one of `keys`.
    fn depends_on(&self, k: Key, keys: &FxHashSet<Key>) -> bool {
        self.residency.deps(k).iter().any(|&unit| {
            keys.contains(&Key {
                mount: k.mount,
                unit,
            })
        })
    }

    /// Tells the residency manager how the load of `key` ended: resident, released to load
    /// again once memory frees, or failed. False when it isn't resident or its mount is gone.
    fn unit_landed(&mut self, key: Key, result: &Result<(), Arc<Error>>, frame: u64) -> bool {
        match result {
            Ok(()) => self.residency.landed(key, true, frame),
            Err(_) if !self.mounts.contains_key(&key.mount) => false,
            Err(e) if e.is_out_of_memory() => {
                self.residency.retry(key);
                false
            }
            Err(_) => self.residency.landed(key, false, frame),
        }
    }

    /// Reports each failure among a landed batch's loads once, however many loads share it:
    /// one for lack of memory to the residency manager, another of units of live mounts as
    /// a warning. A user blob's failure is its consumer's to report.
    fn note_failures(&mut self, loads: &[Landed]) {
        let mut failures: Vec<(&Arc<Error>, Vec<Key>)> = Vec::new();
        for l in loads {
            let Err(e) = &l.result else { continue };
            let i = match failures.iter().position(|(f, _)| Arc::ptr_eq(f, e)) {
                Some(i) => i,
                None => {
                    failures.push((e, Vec::new()));
                    failures.len() - 1
                }
            };
            if let What::Unit(k) = l.what
                && self.mounts.contains_key(&k.mount)
            {
                failures[i].1.push(k);
            }
        }
        for (e, units) in failures {
            if e.is_out_of_memory() {
                self.residency.out_of_memory(governs(&self.governed, e));
            } else if let [k] = units[..] {
                log::warn!("{k:?} failed to load: {e}");
            } else if !units.is_empty() {
                log::warn!(
                    "{} units failed to load, {:?} first: {e}",
                    units.len(),
                    units[0]
                );
            }
        }
    }

    fn landed(
        &mut self,
        mut l: Landed,
        clas: &[(u32, u32)],
        frame: u64,
        w: &mut Apply<'_>,
    ) -> Result<()> {
        let ok = l.result.is_ok();
        // A mesh page's placement holds its ids from here; one that failed before it was
        // placed returns them now.
        if let What::Unit(key) = l.what
            && let Some((first, n)) = self.loading_ids.remove(&key)
            && !matches!(l.placed, Some(Placed::MeshPage(_)))
        {
            self.accel.clusters.free(first, n);
        }
        if let (What::Unit(key), Some(Placed::MeshPage(_))) = (l.what, &l.placed) {
            return self.mesh_landed(key, l, clas, frame, w);
        }
        match l.what {
            What::Unit(key) => {
                let live = self.unit_landed(key, &l.result, frame);
                match (live && ok, l.placed.take()) {
                    (true, Some(Placed::Pool(slot, target))) => {
                        self.point_target(target, l.gpu, w)?;
                        self.unit_blocks.insert(slot.block, key);
                        self.resident.insert(key, Placed::Pool(slot, target));
                    }
                    (true, Some(Placed::Tile(slot))) => {
                        let m = self.mounts.get_mut(&key.mount).expect("live mount");
                        let (t, tile) = m.tile(key.unit);
                        t.resident[tile as usize] = true;
                        t.dirty = true;
                        self.tile_blocks.insert(slot.block, key);
                        self.resident.insert(key, Placed::Tile(slot));
                    }
                    (_, Some(Placed::Pool(slot, _))) => self.units.lock().unwrap().free(slot),
                    (_, Some(Placed::MeshPage(_))) => unreachable!("mesh pages land apart"),
                    (_, Some(Placed::Tile(slot))) => {
                        let (stream, tile) = l.tile.expect("a tile load names its tile");
                        self.unbinding.insert(key);
                        self.tiles.lock().unwrap().retire(slot);
                        self.unbinds.push(TileFree {
                            key,
                            stream,
                            tile,
                            slot,
                        });
                    }
                    (_, None) => {}
                }
            }
            What::Blob(id) => {
                let slot = &mut self.blobs[id as usize];
                let b = slot.as_mut().expect("a loading blob");
                let released = matches!(b.state, BlobState::Loading { released: true });
                self.residency.blob_landed(b.bytes, ok);
                if released {
                    if ok {
                        self.residency.release_blob(b.bytes);
                    }
                    *slot = None;
                } else {
                    b.state = match l.result {
                        Ok(()) => BlobState::Ready,
                        Err(e) => BlobState::Failed(e),
                    };
                }
            }
        }
        Ok(())
    }

    /// A streamed mesh page whose CLAS have built: its page table entry and its clusters'
    /// entries are written, or with a CLAS the pool could not place its load is canceled to
    /// run again later.
    fn mesh_landed(
        &mut self,
        key: Key,
        mut l: Landed,
        clas: &[(u32, u32)],
        frame: u64,
        w: &mut Apply<'_>,
    ) -> Result<()> {
        let Some(Placed::MeshPage(mut p)) = l.placed.take() else {
            unreachable!("a mesh page load")
        };
        let mut mesh = l.mesh.take();
        let infos = mesh.as_mut().map(|m| std::mem::take(&mut m.clusters));
        let units: Vec<(u32, u32)> = match (&mesh, &self.clas) {
            (Some(m), Some(_)) => m.slots.iter().map(|&s| clas[s as usize]).collect(),
            _ => Vec::new(),
        };
        p.micromap = mesh.and_then(|m| m.micromap);
        p.clas = clas_pool::ranges(&units);
        let placed = l.result.is_ok() && units.iter().all(|&(u, _)| u != NO_UNIT);
        if l.result.is_ok() && self.mounts.contains_key(&key.mount) {
            let clas: u64 = units.iter().map(|&(_, n)| u64::from(n)).sum();
            let micromap = p.micromap.as_ref().map_or(0, |m| m.bytes);
            self.residency.built(key, clas * clas_pool::UNIT + micromap);
        }
        let live = if l.result.is_ok() && !placed && self.mounts.contains_key(&key.mount) {
            self.residency.retry(key);
            self.residency.note_clas_retry();
            false
        } else {
            self.unit_landed(key, &l.result, frame)
        };
        if !(live && placed) {
            self.free_now(p);
            return Ok(());
        }
        let (entry, refs, count) = p.entries();
        let first = p.stream.pages[p.page as usize].first;
        let pool = self.clas.as_deref();
        let entries: Vec<ClusterRef> = (0..count)
            .map(|i| ClusterRef {
                // SAFETY: the page is resident in its slot until it is evicted, which writes
                // these entries null first.
                page: unsafe { GpuPtr::from_address(GpuAddress(l.gpu)) },
                local: i,
                mesh_cluster: first + i,
                clas: match (pool, units.get(i as usize)) {
                    (Some(pool), Some(&(u, _))) => pool.address(u),
                    _ => 0,
                },
            })
            .collect();
        let (table, bases) = (&self.mesh_pages.table, &self.mesh_pages.bases);
        w.put(table.entry(entry), &[l.gpu])?;
        w.put(bases.entry(entry), &[p.ids])?;
        w.put(refs, &entries)?;
        for mirror in p.stream.mirrors.lock().unwrap().iter() {
            if !mirror.own[p.page as usize] {
                let entry = mirror.entries + p.page;
                w.put(table.entry(entry), &[l.gpu])?;
                w.put(bases.entry(entry), &[p.ids])?;
            }
        }
        p.address = l.gpu;
        if let Some(f) = self.fine.get_mut(&(key.mount, p.stream.mesh)) {
            f.init.resident[p.page as usize] = Some(Box::new(PageClusters {
                infos: infos.expect("a page that loaded has its clusters"),
                clas: entries.iter().map(|e| e.clas).collect(),
            }));
        }
        self.page_blocks.insert(p.slot.block, key);
        self.resident.insert(key, Placed::MeshPage(p));
        Ok(())
    }

    /// Frees a mesh page no frame has read.
    fn free_now(&self, p: MeshPlaced) {
        let (slot, free) = p.free();
        self.pool.lock().unwrap().free(slot);
        self.free_page(free);
    }

    fn free_page(&self, f: PageFree) {
        if let Some(pool) = &self.clas {
            pool.free(&f.clas, f.micromap);
        }
        self.accel.clusters.free(f.ids.0, f.ids.1);
    }

    /// Copies frame `frame`'s texture feedback back after every tap of the frame on `list`.
    pub(crate) fn record_feedback(&self, list: &mut CommandList<'_>, frame: u64) {
        let slot = (frame % u64::from(self.fif)) as usize;
        self.feedback.record_end(list, slot);
    }

    /// Memory outside the streamed units ran out with `e`: the next step shrinks the budgets
    /// when they govern its kind.
    pub(crate) fn out_of_memory(&mut self, e: &Error) {
        self.residency.out_of_memory(governs(&self.governed, e));
    }

    /// Frees the slots retired at or before `completed`; the tiles among them unbind with
    /// the next task.
    pub(crate) fn retire(&mut self, completed: u64) {
        self.retired_cuts.retain(|(frame, _)| *frame > completed);
        self.retired_scratch.retain(|(frame, _)| *frame > completed);
        let mut tiles = self.tiles.lock().unwrap();
        self.retired_tile_slots.retain(|&(frame, slot)| {
            let done = frame <= completed;
            if done {
                tiles.free_retired(slot);
            }
            !done
        });
        drop(tiles);
        let mut pool = self.pool.lock().unwrap();
        self.retired.retain(|&(frame, slot)| {
            let done = frame <= completed;
            if done {
                pool.free_retired(slot);
            }
            !done
        });
        drop(pool);
        let mut units = self.units.lock().unwrap();
        self.retired_units.retain(|&(frame, slot)| {
            let done = frame <= completed;
            if done {
                units.free_retired(slot);
            }
            !done
        });
        drop(units);
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired_pages)
            .into_iter()
            .partition(|(f, _)| *f <= completed);
        self.retired_pages = keep;
        for (_, f) in done {
            self.free_page(f);
        }
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired_tiles)
            .into_iter()
            .partition(|(f, _)| *f <= completed);
        self.retired_tiles = keep;
        self.unbinds.extend(done.into_iter().map(|(_, t)| t));
        let feedback = &mut self.feedback;
        self.retired_cells.retain(|&(frame, first)| {
            let done = frame <= completed;
            if done {
                feedback.free(first);
            }
            !done
        });
    }
}

#[derive(Clone, Copy)]
enum What {
    Unit(Key),
    Blob(u32),
}

enum LoadKind {
    Pool {
        entry: u32,
        blob: u32,
        bytes: u32,
        target: PoolTarget,
    },
    Tile {
        stream: Arc<TexStream>,
        tile: u32,
        /// The slot it should follow when moved.
        after: Option<PoolSlot>,
    },
    User {
        blob: Blob,
        target: BlobTarget,
    },
    MeshPage {
        stream: Arc<MeshStream>,
        page: u32,
        /// The cluster id of its first cluster.
        ids: u32,
    },
}

struct StreamLoad {
    what: What,
    source: Arc<BlobSource>,
    kind: LoadKind,
}

impl StreamLoad {
    /// The bytes it reads and places, and the CLAS it builds.
    fn size(&self) -> (u64, u32) {
        match &self.kind {
            LoadKind::Pool { bytes, .. } => (u64::from(*bytes), 0),
            LoadKind::Tile { stream, .. } => (stream.layout.tile_bytes, 0),
            LoadKind::User { blob, .. } => (u64::from(blob.raw_len), 0),
            LoadKind::MeshPage { stream, page, .. } => {
                let p = stream.pages[*page as usize];
                (u64::from(p.bytes), p.clusters)
            }
        }
    }
}

/// Bytes the loads of one stream task read and place, at most, past its first load. A task
/// lands once every one of its reads and its GPU work have, so a step's loads go in tasks
/// whose GPU work submits as their own reads end, while the later tasks still read.
const TASK_BYTES: u64 = 8 << 20;

/// A step's loads, in order, in tasks of at most [`TASK_BYTES`] and, when `builds` (mesh
/// pages build CLAS), [`BATCH_CLAS`] CLAS each past their first load.
fn split_tasks(loads: Vec<StreamLoad>, builds: bool) -> Vec<Vec<StreamLoad>> {
    let mut tasks: Vec<Vec<StreamLoad>> = Vec::new();
    let (mut bytes, mut clas) = (0, 0);
    for load in loads {
        let (b, c) = load.size();
        let c = if builds { c } else { 0 };
        match tasks.last_mut() {
            Some(task) if bytes + b <= TASK_BYTES && clas + c <= BATCH_CLAS => {
                task.push(load);
                bytes += b;
                clas += c;
            }
            _ => {
                tasks.push(vec![load]);
                (bytes, clas) = (b, c);
            }
        }
    }
    tasks
}

/// A copy from the staging pool.
enum Copy {
    Buffer {
        src: u64,
        dst: u64,
        bytes: u64,
    },
    Texture {
        src: u64,
        texture: Arc<Texture>,
        region: TextureRegion,
    },
}

/// Each listed texture's last tile slot, by its stream's address.
type NearTiles = FxHashMap<usize, (Arc<TexStream>, PoolSlot)>;

/// One step's loads on the loading side.
pub(crate) struct StreamTask {
    accel: Arc<AccelShared>,
    clas: Option<Arc<ClasPool>>,
    pool: Arc<Mutex<Pool>>,
    units: Arc<Mutex<Pool>>,
    staging: Arc<Mutex<Pool>>,
    tiles: Arc<Mutex<Pool>>,
    device: Device,
    loads: Vec<StreamLoad>,
    unbinds: Vec<TileFree>,
    /// Each texture's last tile slot among its step's tasks, by its stream's address, so its
    /// next tile goes beside it; the stream stays alive while listed.
    near: Arc<Mutex<NearTiles>>,
    /// The start chain plus one per read.
    outstanding: AtomicUsize,
    done: Mutex<Vec<Landed>>,
    copies: Mutex<Vec<Copy>>,
    /// The staged mesh pages to decode, and each one's cluster count.
    decodes: Mutex<Vec<(StoredPage, u32)>>,
}

/// A finished load.
struct Landed {
    what: What,
    placed: Option<Placed>,
    /// The pool unit's address.
    gpu: u64,
    /// A tile's texture and index, for its bind and copy.
    tile: Option<(Arc<TexStream>, u32)>,
    /// Where it staged, and the address.
    staging: Option<Staged>,
    staged_at: u64,
    /// A mesh page's clusters as it decoded, and what its CLAS build made.
    mesh: Option<MeshLanded>,
    /// The stored bytes its read returned.
    read: u64,
    result: Result<(), Arc<Error>>,
}

/// A streamed mesh page's clusters as its records name them, its decode and its CLAS build.
struct MeshLanded {
    clusters: Vec<ClusterInfo>,
    region: Option<PageMicromap>,
    /// Its page in the batch's [`PageDecodes`].
    decode: u32,
    /// The results index of each cluster's CLAS.
    slots: Vec<u32>,
    micromap: Option<PoolMicromap>,
}

/// A stream task's loads, handed to the render thread once its GPU work has landed.
pub(crate) struct StreamBatch {
    points: Vec<TimelinePoint>,
    loads: Vec<Landed>,
    /// Its CLAS pool batch, whose results its pages' slots index.
    clas: Option<PoolBatch>,
    decode: Option<PageDecodes>,
    /// The task's unbinds when its sparse bind failed, for the next task.
    unbinds: Vec<TileFree>,
    /// A failure of the task's GPU work that no load carries: unbinds or the CLAS pool's
    /// own work.
    error: Option<Error>,
}

/// A batch's mesh page decode: the page records, the status word of each page and the
/// kernels' root, in one staging slot.
struct PageDecodes {
    slot: PoolSlot,
    block: Block,
    status_at: u64,
    count: u32,
}

impl PageDecodes {
    /// Fails the landed mesh pages whose decode set a status bit, all with one error.
    fn fail_pages(&self, loads: &mut [Landed]) {
        let mut status = vec![0u32; self.count as usize];
        let ok = self
            .block
            .read_mapped(self.status_at, bytemuck::cast_slice_mut(&mut status));
        assert!(ok, "the decode status is mapped");
        let mut error: Option<Arc<Error>> = None;
        for l in loads {
            let failed = l
                .mesh
                .as_ref()
                .is_some_and(|m| status[m.decode as usize] != 0);
            if failed && l.result.is_ok() {
                let e = error.get_or_insert_with(|| {
                    Arc::new(orr_pack::Error::Malformed("streamed mesh page streams").into())
                });
                l.result = Err(Arc::clone(e));
            }
        }
    }
}

impl StreamBatch {
    pub(crate) fn landed(&self, device: &Device) -> bool {
        self.points.iter().all(|&p| landed(device, Some(p)))
    }
}

impl StreamTask {
    fn end(&self, loading: &Loading) {
        if self.outstanding.fetch_sub(1, Ordering::AcqRel) == 1 {
            loading.send(Msg::StreamDone);
        }
    }

    /// Whether every one of its reads has ended.
    pub(crate) fn read(&self) -> bool {
        self.outstanding.load(Ordering::Acquire) == 0
    }

    fn fail(&self, what: What, e: Error) {
        self.done.lock().unwrap().push(Landed {
            what,
            placed: None,
            gpu: 0,
            tile: None,
            staging: None,
            staged_at: 0,
            mesh: None,
            read: 0,
            result: Err(Arc::new(e)),
        });
    }

    /// Returns what a failed load took.
    fn free(&self, placed: Option<Placed>, staging: Option<Staged>) {
        match placed {
            Some(Placed::Pool(slot, _)) => self.units.lock().unwrap().free(slot),
            Some(Placed::Tile(slot)) => self.tiles.lock().unwrap().free(slot),
            Some(Placed::MeshPage(p)) => self.pool.lock().unwrap().free(p.slot),
            None => {}
        }
        if let Some(Staged::Pool(s)) = staging {
            self.staging.lock().unwrap().free(s);
        }
    }
}

/// The staging a load reads or decodes into: a slot of the staging pool, or a range of the
/// read arena, which returns when the buffer drops.
enum Staged {
    Pool(PoolSlot),
    Arena { _range: AlignedBuf },
}

/// The blob and chunk a load reads, the length it decodes to and the bytes it lands in.
fn source_of(load: &StreamLoad) -> Result<(Blob, Option<u32>, u64, u64)> {
    let a = &load.source.archive;
    let entry_blob = |entry: u32, blob: u32| {
        a.entries()
            .get(entry as usize)
            .and_then(|e| a.blobs(e).get(blob as usize).copied())
            .ok_or(orr_pack::Error::Malformed(
                "streamed blob outside its entry",
            ))
    };
    let tile_bytes = orr_pack::texture::TILE_BYTES as u32;
    let (b, chunk, len, slot) = match &load.kind {
        LoadKind::Pool {
            entry, blob, bytes, ..
        } => {
            let len = u64::from(*bytes);
            (entry_blob(*entry, *blob)?, None, len, len)
        }
        LoadKind::MeshPage { stream, page, .. } => {
            let p = stream.pages[*page as usize];
            let b = entry_blob(stream.entry, p.blob)?;
            (b, None, u64::from(b.raw_len), u64::from(p.bytes))
        }
        LoadKind::Tile { stream, tile, .. } => {
            let g = &stream.grid;
            let t = g.tile(*tile);
            let b = entry_blob(stream.entry, stream.blobs[t.mip as usize])?;
            let [gx, gy] = g.grids[t.mip as usize];
            let chunk = t.layer * gx * gy + t.xy[1] * gx + t.xy[0];
            if chunk >= u32::from(b.chunk_count) || b.chunk_len(chunk) != tile_bytes {
                return Err(orr_pack::Error::Malformed("texture tile chunk").into());
            }
            let len = u64::from(tile_bytes);
            (b, Some(chunk), len, len)
        }
        LoadKind::User { blob, target } => {
            let len = u64::from(blob.raw_len);
            let fits = match target {
                BlobTarget::Buffer { len: room, .. } => len <= *room,
                BlobTarget::Texture { texture, region } => {
                    len == region.bytes(texture.desc().format)
                }
            };
            if !fits {
                return Err(orr_pack::Error::Malformed("blob does not fit its target").into());
            }
            (*blob, None, len, len)
        }
    };
    if b.codec() == Codec::GDeflate {
        return Err(orr_pack::Error::GDeflate.into());
    }
    if u64::from(chunk.map_or(b.raw_len, |i| b.chunk_len(i))) != len {
        return Err(orr_pack::Error::Malformed("streamed blob length").into());
    }
    Ok((b, chunk, len, slot))
}

/// Places a stream task's loads on the loading thread: memory for each, then its read.
pub(crate) fn start(loading: &Arc<Loading>, task: &Arc<StreamTask>) {
    let mut near = task.near.lock().unwrap();
    let units_staged = task.units.lock().unwrap().kind() == MemoryKind::Staging;
    for (i, load) in task.loads.iter().enumerate() {
        let (b, chunk, len, slot_len) = match source_of(load) {
            Ok(s) => s,
            Err(e) => {
                task.fail(load.what, e);
                continue;
            }
        };
        // A pool slot in staging memory is where the load lands: it reads or decodes into
        // the slot, and no copy follows.
        let direct = units_staged && matches!(load.kind, LoadKind::Pool { .. });
        let (placed, block) = match &load.kind {
            LoadKind::Pool { target, .. } => {
                let mut pool = task.units.lock().unwrap();
                let slot = if direct && reads_straight(&load.source, &b, chunk, false) {
                    let sector = SECTOR as u64;
                    pool.alloc_aligned(&task.device, len.next_multiple_of(sector), sector)
                } else {
                    pool.alloc(&task.device, len, None)
                };
                match slot {
                    Ok((slot, block)) => (Some(Placed::Pool(slot, *target)), Some(block)),
                    Err(e) => {
                        task.fail(load.what, e);
                        continue;
                    }
                }
            }
            LoadKind::MeshPage { stream, page, ids } => {
                match task
                    .pool
                    .lock()
                    .unwrap()
                    .alloc(&task.device, slot_len, None)
                {
                    Ok((slot, block)) => {
                        let p = MeshPlaced {
                            slot,
                            stream: Arc::clone(stream),
                            page: *page,
                            ids: *ids,
                            address: 0,
                            clas: Vec::new(),
                            micromap: None,
                        };
                        (Some(Placed::MeshPage(p)), Some(block))
                    }
                    Err(e) => {
                        task.fail(load.what, e);
                        continue;
                    }
                }
            }
            LoadKind::Tile { stream, after, .. } => {
                let key = Arc::as_ptr(stream).addr();
                let slot = task.tiles.lock().unwrap().alloc(
                    &task.device,
                    len,
                    after.or_else(|| near.get(&key).map(|n| n.1)),
                );
                match slot {
                    Ok((slot, _)) => {
                        near.insert(key, (Arc::clone(stream), slot));
                        (Some(Placed::Tile(slot)), None)
                    }
                    Err(e) => {
                        task.fail(load.what, e);
                        continue;
                    }
                }
            }
            LoadKind::User { .. } => (None, None),
        };
        // Decoders read what they write, so they write cached memory, which a copy or the
        // mesh page decode then reads: the read arena where the device imports it, else the
        // staging pool. A read straight into the slot fills whole sectors.
        let staged_len = len.next_multiple_of(SECTOR as u64);
        let (staging, staged) = if direct {
            (None, block.expect("a pool slot"))
        } else if let Some((buf, staged)) = loading.staging(staged_len) {
            (Some(Staged::Arena { _range: buf }), staged)
        } else {
            let staged = task
                .staging
                .lock()
                .unwrap()
                .alloc(&task.device, staged_len, None);
            match staged {
                Ok((slot, staged)) => (Some(Staged::Pool(slot)), staged),
                Err(e) => {
                    task.free(placed, None);
                    task.fail(load.what, e);
                    continue;
                }
            }
        };
        task.outstanding.fetch_add(1, Ordering::AcqRel);
        let t = Arc::clone(task);
        let l = Arc::downgrade(loading);
        read_into(
            &load.source,
            b,
            chunk,
            staged,
            Queue::Stream,
            Box::new(move |decoded, read| {
                let load = &t.loads[i];
                let mut mesh = None;
                let result = decoded.and_then(|bytes| {
                    if let Some(l) = l.upgrade() {
                        l.flush(&staged)?;
                    }
                    if let LoadKind::MeshPage { stream, page, .. } = &load.kind {
                        let to = (block.expect("a pool slot").gpu(), slot_len);
                        mesh = Some(stage_page(stream, *page, bytes, &staged, to, &t.decodes)?);
                    }
                    Ok(())
                });
                let mut landed = Landed {
                    what: load.what,
                    placed,
                    gpu: block.map_or(0, |b| b.gpu()),
                    tile: match &load.kind {
                        LoadKind::Tile { stream, tile, .. } => Some((Arc::clone(stream), *tile)),
                        _ => None,
                    },
                    staging,
                    staged_at: staged.gpu(),
                    mesh,
                    read,
                    result: result.map_err(Arc::new),
                };
                if landed.result.is_err() {
                    t.free(landed.placed.take(), landed.staging.take());
                } else if !direct {
                    let copy = match (&load.kind, block) {
                        (LoadKind::Pool { .. }, Some(block)) => Some(Copy::Buffer {
                            src: staged.gpu(),
                            dst: block.gpu(),
                            bytes: slot_len,
                        }),
                        (LoadKind::User { target, .. }, _) => Some(match target {
                            BlobTarget::Buffer { at, .. } => Copy::Buffer {
                                src: staged.gpu(),
                                dst: at.address().0,
                                bytes: len,
                            },
                            BlobTarget::Texture { texture, region } => Copy::Texture {
                                src: staged.gpu(),
                                texture: Arc::clone(texture),
                                region: *region,
                            },
                        }),
                        _ => None,
                    };
                    // An empty blob copies nothing, and a copy of no bytes is invalid.
                    let copy = copy.filter(|c| !matches!(c, Copy::Buffer { bytes: 0, .. }));
                    t.copies.lock().unwrap().extend(copy);
                }
                t.done.lock().unwrap().push(landed);
                if let Some(l) = l.upgrade() {
                    t.end(&l);
                }
            }),
        );
    }
    task.end(loading);
}

/// Queues the decode of a streamed mesh page's stored bytes, which its staging slot `staged`
/// holds, into the pool slot at `to` (address and bytes); reads its clusters from its
/// records, which the decode copies unchanged.
fn stage_page(
    stream: &MeshStream,
    page: u32,
    stored: &[u8],
    staged: &Block,
    to: (u64, u64),
    decodes: &Mutex<Vec<(StoredPage, u32)>>,
) -> Result<MeshLanded> {
    let (h, records) = orr_pack::mesh::check_stored_page(stored)?;
    if h.attr_stride != stream.header.attr_stride() {
        return Err(orr_pack::Error::Malformed("page attribute stride").into());
    }
    if h.cluster_count != stream.pages[page as usize].clusters {
        return Err(orr_pack::Error::Malformed("streamed page cluster count").into());
    }
    if u64::from(h.len) > to.1 {
        return Err(orr_pack::Error::Malformed("streamed page length").into());
    }
    let (clusters, region) = page_infos(&h, &records, page, &stream.groups)?;
    let record = StoredPage {
        stored: staged.gpu(),
        resident: to.0,
        stored_len: stored.len() as u32,
        resident_capacity: to.1 as u32,
    };
    let mut decodes = decodes.lock().unwrap();
    decodes.push((record, h.cluster_count));
    Ok(MeshLanded {
        clusters,
        region,
        decode: decodes.len() as u32 - 1,
        slots: Vec::new(),
        micromap: None,
    })
}

/// Ends a stream task on the loading thread: unbinds the evicted tiles and binds the new
/// ones, copies the tiles in on the loading queue, submits the copies into the stream pool
/// and the users' targets, builds the streamed mesh pages' CLAS and hands the batch to the
/// render thread.
pub(crate) fn finish(loading: &Loading, device: &Device, task: &Arc<StreamTask>) {
    let mut loads = std::mem::take(&mut *task.done.lock().unwrap());
    let copies = std::mem::take(&mut *task.copies.lock().unwrap());
    let mut decode = None;
    let mut points = Vec::new();
    let mut unbound = false;
    let mut error = None;
    let clas = match submit(
        device,
        task,
        &loads,
        &copies,
        &mut decode,
        &mut points,
        &mut unbound,
    ) {
        Ok(()) => match &task.clas {
            // The other loads' work is submitted; only the pages it builds need it.
            Some(pool) => match build_clas(device, task, pool, &mut loads, &mut points) {
                Ok(batch) => batch,
                Err(e) => {
                    error = fail_loads(&mut loads, e, |l| l.mesh.is_some());
                    None
                }
            },
            None => None,
        },
        Err(e) => {
            error = fail_loads(&mut loads, e, |_| true);
            None
        }
    };
    loading.push_stream(StreamBatch {
        points,
        loads,
        clas,
        decode,
        unbinds: if unbound {
            Vec::new()
        } else {
            task.unbinds.clone()
        },
        error,
    });
}

/// Bytes of a block of the streamed pools and the resident cluster table on `device`: see
/// [`block_bytes`].
pub(crate) fn pool_block(device: &Device, b: &Budgets, pools: PoolBytes) -> u64 {
    let full = pools.mesh
        + pools.textures
        + b.instance_page_pool
        + 2 * b.in_flight_bytes
        + ClusterTable::max_bytes(pools.mesh);
    let max = device.caps().limits.max_memory_allocations;
    block_bytes(b.stream_block, full, max)
}

/// Bytes of a block of the streamed pools: `stream_block`, or the next power of two past it
/// that keeps `full` bytes, the pools at their full budgets with staging and the resident
/// cluster table, within a quarter of the device's `max_allocations`, as each block is device
/// memory of its own.
fn block_bytes(stream_block: u64, full: u64, max_allocations: u32) -> u64 {
    let least = full.div_ceil(u64::from(max_allocations / 4));
    if least <= stream_block {
        stream_block
    } else {
        least.next_power_of_two()
    }
}

/// Whether `e` ran out of memory the residency budgets follow: of a kind in `governed`, on a
/// device-local heap, or device memory of no named kind.
fn governs(governed: &[MemoryKind], e: &Error) -> bool {
    match e {
        Error::Gpu(orr_gpu::Error::OutOfMemory { device, kind, .. }) => {
            kind.map_or(*device, |k| governed.contains(&k))
        }
        _ => false,
    }
}

/// Fails with `e` each load still on course that `which` picks; one that failed on its own
/// keeps its error. Returns `e` when it picks none.
fn fail_loads(loads: &mut [Landed], e: Error, which: impl Fn(&Landed) -> bool) -> Option<Error> {
    let mut failing = loads
        .iter_mut()
        .filter(|l| l.result.is_ok() && which(l))
        .peekable();
    if failing.peek().is_none() {
        return Some(e);
    }
    let e = Arc::new(e);
    for l in failing {
        l.result = Err(Arc::clone(&e));
    }
    None
}

/// Submits a task's GPU work up to its CLAS build. `decode` takes the mesh page decode and
/// `points` each submission's point as soon as it is submitted, so the batch frees nothing
/// the work submitted before a failure still uses; `unbound` is set once the task's
/// unbinds are queued.
fn submit(
    device: &Device,
    task: &StreamTask,
    loads: &[Landed],
    copies: &[Copy],
    decode: &mut Option<PageDecodes>,
    points: &mut Vec<TimelinePoint>,
    unbound: &mut bool,
) -> Result<()> {
    *decode = decode_pages(device, task, points)?;
    let bound: Vec<(&Arc<TexStream>, u32, PoolSlot, u64)> = loads
        .iter()
        .filter(|l| l.result.is_ok())
        .filter_map(|l| match (&l.placed, &l.tile) {
            (Some(Placed::Tile(slot)), Some((s, t))) => Some((s, *t, *slot, l.staged_at)),
            _ => None,
        })
        .collect();
    let mut tiles = task.tiles.lock().unwrap();
    if !bound.is_empty() || !task.unbinds.is_empty() {
        let mut per_texture: Vec<(&TexStream, Vec<TileBind<'_>>)> = Vec::new();
        let entries = task
            .unbinds
            .iter()
            .map(|u| (&*u.stream, u.tile, None))
            .chain(
                bound
                    .iter()
                    .map(|&(s, t, slot, _)| (&**s, t, Some(tiles.memory(slot)))),
            );
        for (s, t, memory) in entries {
            let tile = s.grid.tile(t);
            let bind = TileBind {
                mip: tile.mip,
                layer: tile.layer,
                tile: tile.xy,
                memory,
            };
            match per_texture.iter_mut().find(|(p, _)| std::ptr::eq(*p, s)) {
                Some((_, list)) => list.push(bind),
                None => per_texture.push((s, vec![bind])),
            }
        }
        let mut binds = SparseBinds::new();
        for (s, list) in &per_texture {
            binds.tiles(&s.texture, list)?;
        }
        points.push(device.bind_sparse(&binds)?);
    }
    *unbound = true;
    for u in &task.unbinds {
        tiles.free_retired(u.slot);
    }
    drop(tiles);
    if !bound.is_empty() {
        let queue = first_queue(device, &[QueueKind::Loading, QueueKind::Render]);
        let mut list = queue.command_list()?;
        for &(s, t, _, staged) in &bound {
            let tile = s.grid.tile(t);
            let region = s
                .layout
                .tile_region(s.texture.desc(), tile.mip, tile.layer, tile.xy);
            list.copy_rows_to_texture(
                &s.texture,
                &region,
                address(staged),
                s.layout.tile_extent[0],
            );
        }
        points.push(queue.submit(vec![list], points)?);
    }
    if !copies.is_empty() {
        let queue = first_queue(
            device,
            &[QueueKind::Copy, QueueKind::Loading, QueueKind::Render],
        );
        let mut list = queue.command_list()?;
        for c in copies {
            match c {
                Copy::Buffer { src, dst, bytes } => list.copy(address(*dst), address(*src), *bytes),
                Copy::Texture {
                    src,
                    texture,
                    region,
                } => list.copy_to_texture(texture, region, address(*src)),
            }
        }
        points.push(queue.submit(vec![list], &[])?);
    }
    Ok(())
}

/// CLAS one stream task builds at most, past its first page: its pages build as one pool
/// batch, and every build of a batch takes its own region of the build scratch, so this
/// bounds the scratch and staging a batch holds. The allocator itself places 40,000 CLAS in
/// 20 to 40 ms in a 98,304-page (6 GiB) pool, about 10 to 20 us per run of CLAS.
const BATCH_CLAS: u32 = 4096;

/// Builds the CLAS of a task's streamed mesh pages on the loading queue after its other work
/// as one pool batch, which also applies the pool's queued frees and takes. Returns the batch
/// submitted; none when there is nothing to build or apply.
fn build_clas(
    device: &Device,
    task: &StreamTask,
    pool: &ClasPool,
    loads: &mut [Landed],
    points: &mut Vec<TimelinePoint>,
) -> Result<Option<PoolBatch>> {
    let mut side = pool.side();
    let pages: Vec<usize> = (0..loads.len())
        .filter(|&i| loads[i].result.is_ok() && loads[i].mesh.is_some())
        .collect();
    if pages.is_empty() && !pool.pending(device) {
        return Ok(None);
    }
    let builds: Vec<PageBuild<'_>> = pages
        .iter()
        .map(|&i| {
            let l = &loads[i];
            let Some(Placed::MeshPage(p)) = &l.placed else {
                unreachable!("a mesh page load")
            };
            let m = l.mesh.as_ref().expect("decoded");
            let header = &p.stream.header;
            PageBuild {
                page: l.gpu,
                clusters: &m.clusters,
                cluster_base: p.ids,
                micromap: m.region,
                step: header.step(),
                truncate_bits: u32::from(header.clas_truncate_bits),
                non_opaque: &p.stream.non_opaque,
            }
        })
        .collect();
    let queue = first_queue(device, &[QueueKind::Loading, QueueKind::Render]);
    let mut list = queue.command_list()?;
    let k = &task.accel.kernels;
    let triangles = task.accel.limits.clas_triangles;
    let out = clas_pool::build_pages(
        pool,
        &mut side,
        device,
        k,
        &mut list,
        &builds,
        triangles,
        &task.staging,
    )?;
    drop(builds);
    let mut wait = points.clone();
    wait.extend(&out.binds);
    match queue.submit(vec![list], &wait) {
        Ok(p) => points.push(p),
        Err(e) => {
            out.abandon(pool, &mut side, &task.staging);
            return Err(e.into());
        }
    }
    for ((&i, slots), micromap) in pages.iter().zip(out.slots).zip(out.micromaps) {
        let m = loads[i].mesh.as_mut().expect("decoded");
        m.slots = slots;
        m.micromap = micromap;
    }
    Ok(Some(out.batch))
}

/// Decodes the task's staged mesh pages into their pool slots on the loading queue.
fn decode_pages(
    device: &Device,
    task: &StreamTask,
    points: &mut Vec<TimelinePoint>,
) -> Result<Option<PageDecodes>> {
    let pages = std::mem::take(&mut *task.decodes.lock().unwrap());
    if pages.is_empty() {
        return Ok(None);
    }
    let n = pages.len() as u64;
    let status_at = (n * size_of::<StoredPage>() as u64).next_multiple_of(16);
    let root_at = status_at + (4 * n).next_multiple_of(16);
    let bytes = root_at + size_of::<PageDecodeRoot>() as u64;
    let (slot, block) = task.staging.lock().unwrap().alloc(device, bytes, None)?;
    let records: Vec<StoredPage> = pages.iter().map(|&(p, _)| p).collect();
    let root = PageDecodeRoot {
        pages: address(block.gpu()),
        status: address(block.gpu() + status_at),
    };
    let wrote = block.write_mapped(0, bytemuck::cast_slice(&records))
        && block.write_mapped(status_at, &vec![0; 4 * n as usize])
        && block.write_mapped(root_at, bytemuck::bytes_of(&root));
    assert!(wrote, "a mapped staging slot");
    let dwords = pages
        .iter()
        .map(|(p, _)| p.resident_capacity / 4)
        .max()
        .unwrap_or(0);
    let clusters = pages.iter().map(|&(_, c)| c).max().unwrap_or(0);
    let rows = n as u32;
    let queue = first_queue(device, &[QueueKind::Loading, QueueKind::Render]);
    let submitted = queue.command_list().and_then(|mut list| {
        let k = &task.accel.kernels;
        let root = address::<PageDecodeRoot>(block.gpu() + root_at);
        list.dispatch(&k.page_copy, root, [dwords.div_ceil(256), rows, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&k.page_decode, root, [clusters.max(1), rows, 1]);
        queue.submit(vec![list], &[])
    });
    match submitted {
        Ok(p) => points.push(p),
        Err(e) => {
            task.staging.lock().unwrap().free(slot);
            return Err(e.into());
        }
    }
    Ok(Some(PageDecodes {
        slot,
        block,
        status_at,
        count: rows,
    }))
}

fn first_queue<'d>(device: &'d Device, kinds: &[QueueKind]) -> &'d orr_gpu::Queue {
    kinds
        .iter()
        .find_map(|&k| device.queue(k))
        .expect("the render queue always exists")
}

/// A device address a stream task copies from or to, or a decode reads.
fn address<T>(at: u64) -> GpuPtr<T> {
    // SAFETY: the address lies in a staging slot, a pool slot or a consumer target, which
    // the task holds until its batch has landed.
    unsafe { GpuPtr::from_address(GpuAddress(at)) }
}
