//! Streamed units and user blobs: the render thread's residency step at step 1, and the
//! loading side's reads that place each unit.
//!
//! At step 1 the render thread takes the stream batches whose GPU work has landed, writes
//! each unit's address into its table entry or marks its texture tile resident, hands the
//! requests since the last step to the [`Residency`] manager, writes null over the entries
//! of the units it evicts or coarsens their textures' residency maps (their memory returns
//! once the frames that could read it have completed) and sends the loads to the loading
//! thread. The frame's submit waits the taken batches' points. Under memory pressure the
//! step shrinks the residency budgets, evicting streamed units, and the stream pool's
//! blocks sit at low priority so the OS pages them out before anything else.
//!
//! On the loading side a unit's blob, or a tile's chunk, is read through its mount's path
//! and decompressed on a worker: hierarchy pages into their stream pool slot (or the
//! staging pool and a copy), texture tiles, user blobs and mesh pages, still in their
//! stored form, into the staging pool. When a task's last read ends, the loading thread
//! unbinds the tiles evicted since the last task and binds the new ones in one sparse bind,
//! copies the tiles in and decodes the mesh pages into their stream pool slots on the
//! loading queue, submits the other copies, builds the mesh pages' CLAS and hands the batch
//! to the render thread, which drops the pages whose decode failed.
//!
//! The step also builds a streaming mesh's finer cached cuts once their pages are resident,
//! holding those pages locked until no copy uses the cut, and moves tiles of a texture that
//! lost one back beside its others.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use orr_gpu::{
    CommandList, Device, GpuAddress, GpuPtr, Hazards, MemoryKind, Micromap, QueueKind, SparseBinds,
    Stages, Texture, TextureRegion, TileBind, TimelinePoint,
};
use orr_pack::mesh::{MeshHeader, StoredPage};
use orr_pack::{Blob, Codec};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::accel::{self, AccelShared, Animated, ClusterInfo, FineCut, PageMicromap};
use crate::clas_pool::{self, ClasPool, NO_UNIT, PageBuild, PoolBatch};
use crate::edit::Mirror;
use crate::loading::{BlobSource, Loading, Msg, decoded, landed, read_stored};
use crate::memory::{Block, Pool, PoolSlot};
use crate::mount::{Apply, page_infos};
use crate::residency::{
    HOLD_FRAMES, Key, Limits, Plan, Priority, Residency, ResidencyClass, ResidencyStats, UnitDesc,
};
use crate::shaders::page_decode::PageDecodeRoot;
use crate::tiles::{TexResidency, TexStream, TextureFeedback, compaction_move};
use crate::upload::Record;
use crate::{Budgets, ClusterRef, Error, LoadState, MountId, Result, RtCutGpu, TextureStreamGpu};

/// Tiles one step moves beside their texture's others, at most.
const MOVES_PER_STEP: usize = 8;
/// Frames a tile goes unused before it may move: its map coarsens until it lands again.
const MOVE_COLD_FRAMES: u64 = 8;

/// Who learns the address of a unit in a stream pool slot, null while it is not resident.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PoolTarget {
    /// A table entry on the GPU.
    Table(GpuPtr<u64>),
    /// The transform groups placing an instance page.
    InstancePage(u32),
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
    pub entry: u32,
    pub header: MeshHeader,
    pub pages: Vec<StreamedPage>,
    /// The mesh's page table.
    pub table: GpuPtr<u64>,
    /// The entry of mesh cluster 0 in the resident cluster table, and its cluster id.
    pub clusters: GpuPtr<ClusterRef>,
    pub cluster_base: u32,
    /// Sections whose every triangle is non-opaque.
    pub non_opaque: Vec<bool>,
    /// Its index in its mount, and the first page that streams.
    pub mesh: u32,
    pub pinned: u32,
    /// Edits of the mesh, whose tables follow its landings and evictions.
    pub mirrors: Mutex<Vec<Arc<Mirror>>>,
}

impl MeshStream {
    /// The resident bytes of `pages`.
    fn pages_bytes(&self, pages: &[u32]) -> u64 {
        pages
            .iter()
            .map(|&p| u64::from(self.pages[p as usize].bytes))
            .sum()
    }
}

/// One page of a [`MeshStream`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamedPage {
    pub blob: u32,
    /// Resident bytes.
    pub bytes: u32,
    /// Its first mesh cluster and cluster count.
    pub first: u32,
    pub clusters: u32,
}

/// What a streaming mesh's finer cached cuts are built from.
pub(crate) struct FineInit {
    pub mesh: u32,
    pub stream: Arc<MeshStream>,
    /// The table entries of its finer cuts, finest first, and their thresholds.
    pub table: GpuPtr<RtCutGpu>,
    pub thresholds: Vec<f32>,
    /// For each threshold, the streamed pages holding a group whose error is over it.
    pub pages: Vec<Vec<u32>>,
    pub merge: f32,
    pub animated: Animated,
    /// By mesh cluster: as known so far, and its CLAS.
    pub clusters: Vec<ClusterInfo>,
    pub clas: Vec<u64>,
}

enum FineState {
    /// Not built; asked for at this frame, if ever.
    Empty(u64),
    /// Being built, holding `locks`.
    Building { locks: Vec<u32> },
    /// Built, last reported in use at `touched`, referencing `locks`.
    Built {
        cut: FineCut,
        touched: u64,
        locks: Vec<u32>,
    },
}

/// A streaming mesh's finer cached cuts on the render thread.
struct MeshFine {
    init: FineInit,
    slots: Vec<FineState>,
}

/// A finer cut for the loading thread to build.
pub(crate) struct CutTask {
    mount: MountId,
    mesh: u32,
    slot: u32,
    accel: Arc<AccelShared>,
    clusters: Vec<u32>,
    infos: Vec<ClusterInfo>,
    animated: Animated,
    clas: Vec<u64>,
}

/// A finer cut built, handed over once its build has completed.
pub(crate) struct CutBatch {
    mount: MountId,
    mesh: u32,
    slot: u32,
    point: Option<TimelinePoint>,
    cut: Result<FineCut>,
}

impl CutBatch {
    pub(crate) fn landed(&self, device: &Device) -> bool {
        landed(device, self.point)
    }
}

/// Builds a finer cut on the loading thread.
pub(crate) fn build_cut(loading: &Loading, device: &Device, t: CutTask) {
    let built = accel::build_fine_cut(
        device,
        &t.accel.kernels,
        t.clusters,
        &t.infos,
        t.animated,
        &t.clas,
    );
    let (cut, point) = match built {
        Ok((c, p)) => (Ok(c), Some(p)),
        Err(e) => (Err(e), None),
    };
    loading.push_cut(CutBatch {
        mount: t.mount,
        mesh: t.mesh,
        slot: t.slot,
        point,
        cut,
    });
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
    /// The unit of each instance page.
    pub page_units: Vec<u32>,
    /// Every unit, and how to load the ones that start absent.
    pub units: Vec<(UnitDesc, Option<StreamUnit>)>,
    /// Streamed units of each mesh.
    pub meshes: Vec<Vec<u32>>,
    /// The unit of each node page of each mesh.
    pub node_pages: Vec<Vec<u32>>,
    /// The mount's streamed textures and the unit of each one's first tile.
    pub textures: Vec<(Arc<TexStream>, u32)>,
    /// The unit of each page of each mesh.
    pub mesh_pages: Vec<Vec<u32>>,
    /// The mount's meshes whose pages stream, as `MeshPage` units name them.
    pub mesh_streams: Vec<Arc<MeshStream>>,
    /// Those of them with finer cached cuts.
    pub fine: Vec<FineInit>,
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

/// A streamed mesh page: its slot, its clusters' CLAS units and its micromap.
struct MeshPlaced {
    slot: PoolSlot,
    stream: Arc<MeshStream>,
    page: u32,
    /// Its resident address and each cluster's CLAS, once it has landed.
    address: u64,
    clas_addresses: Vec<u64>,
    clas: Vec<(u32, u32)>,
    micromap: Option<(Micromap, PoolSlot)>,
}

impl MeshPlaced {
    /// Its page table entry and its clusters' entries in the resident cluster table.
    fn entries(&self) -> (GpuPtr<u64>, GpuPtr<ClusterRef>, u32) {
        let p = self.stream.pages[self.page as usize];
        (
            self.stream.table.offset(u64::from(self.page)),
            self.stream.clusters.offset(u64::from(p.first)),
            p.clusters,
        )
    }
}

/// A streamed page's CLAS and micromap, freed once no frame reads them.
struct ClasFree {
    clas: Vec<(u32, u32)>,
    micromap: Option<(Micromap, PoolSlot)>,
}

/// A bound tile to unbind and free.
pub(crate) struct TileFree {
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
    /// Streamed pages' CLAS; none without cluster acceleration structures.
    clas: Option<Arc<Mutex<ClasPool>>>,
    /// CLAS freed once the frame has completed.
    retired_clas: Vec<(u64, ClasFree)>,
    pool: Arc<Mutex<Pool>>,
    staging: Arc<Mutex<Pool>>,
    tiles: Arc<Mutex<Pool>>,
    /// Finer cached cuts by mount and mesh.
    fine: HashMap<(MountId, u32), MeshFine>,
    /// Bytes of mesh pages they hold locked.
    fine_locked: u64,
    /// Finer cuts freed once the frame has completed.
    retired_cuts: Vec<(u64, FineCut)>,
    mounts: FxHashMap<MountId, MountStreams>,
    resident: FxHashMap<Key, Placed>,
    /// Pool slots freed once the frame has completed.
    retired: Vec<(u64, PoolSlot)>,
    /// Tiles unbound once the frame has completed.
    retired_tiles: Vec<(u64, TileFree)>,
    /// Tiles the next task unbinds.
    unbinds: Vec<TileFree>,
    /// Tiles evicted and not yet unbound by a task sent to the loading side; they do not
    /// load again until then, so their unbind never lands after a new bind.
    unbinding: FxHashSet<Key>,
    feedback: TextureFeedback,
    /// Feedback cells freed once the frame has completed.
    retired_cells: Vec<(u64, u32)>,
    streams: GpuPtr<GpuPtr<TextureStreamGpu>>,
    /// Tiles leaving to land beside their texture's others: the slot each should follow and
    /// the frame it stops asking.
    moving: HashMap<Key, (PoolSlot, u64)>,
    /// Textures that lost a tile, whose others may move, until the given frame: the room a
    /// move wants comes free some frames after other textures' evictions.
    fragmented: HashMap<(MountId, u32), u64>,
    /// Heap indices whose stream table entry the next step writes null.
    nulls: Vec<u32>,
    /// Instance pages that landed (their records' address) or left (0) since the last take.
    page_events: Vec<(u32, u64)>,
    blobs: Vec<Option<UserBlob>>,
    queued: VecDeque<u32>,
    fif: u32,
}

impl Streaming {
    pub(crate) fn new(
        device: &Device,
        b: &Budgets,
        accel: &Arc<AccelShared>,
        streams: GpuPtr<GpuPtr<TextureStreamGpu>>,
        fif: u32,
    ) -> Result<Self> {
        let mut budgets = [0; ResidencyClass::ALL.len()];
        budgets[ResidencyClass::MeshPages as usize] = b.mesh_page_pool;
        budgets[ResidencyClass::HierarchyPages as usize] = b.hierarchy_page_pool;
        budgets[ResidencyClass::Textures as usize] = b.texture_pool;
        budgets[ResidencyClass::InstancePages as usize] = b.instance_page_pool;
        budgets[ResidencyClass::UserBlobs as usize] = b.user_blob_pool;
        let kind = if device.caps().rebar {
            MemoryKind::Upload
        } else {
            MemoryKind::DeviceLocal
        };
        let caps = device.caps();
        let clas = if caps.cluster.max_tris > 0 && caps.cluster.max_partitions > 0 {
            let pool = ClasPool::new(device, b.clas_pool, b.stream_block)?;
            Some(Arc::new(Mutex::new(pool)))
        } else {
            None
        };
        Ok(Self {
            accel: Arc::clone(accel),
            clas,
            retired_clas: Vec::new(),
            residency: Residency::new(Limits {
                budgets,
                io_requests: b.io_requests,
                io_bytes: b.io_bytes,
            }),
            pool: Arc::new(Mutex::new(Pool::new(
                kind,
                256,
                b.stream_block,
                "stream pool",
            ))),
            staging: Arc::new(Mutex::new(Pool::new(
                MemoryKind::Upload,
                256,
                b.stream_block,
                "stream staging",
            ))),
            tiles: Arc::new(Mutex::new(Pool::new(
                MemoryKind::DeviceLocal,
                orr_pack::texture::TILE_BYTES as u64,
                b.stream_block,
                "texture tile pool",
            ))),
            fine: HashMap::new(),
            fine_locked: 0,
            retired_cuts: Vec::new(),
            mounts: FxHashMap::default(),
            resident: FxHashMap::default(),
            retired: Vec::new(),
            retired_tiles: Vec::new(),
            unbinds: Vec::new(),
            unbinding: FxHashSet::default(),
            feedback: TextureFeedback::new(
                device,
                b.texture_feedback_cells,
                b.texture_feedback_records,
                fif,
            )?,
            retired_cells: Vec::new(),
            streams,
            moving: HashMap::new(),
            fragmented: HashMap::new(),
            nulls: Vec::new(),
            page_events: Vec::new(),
            blobs: Vec::new(),
            queued: VecDeque::new(),
            fif: fif.max(1),
        })
    }

    /// What the first frame's scatter clears.
    pub(crate) fn init(&self) -> Record {
        self.feedback.init()
    }

    pub(crate) fn stats(&self) -> &ResidencyStats {
        self.residency.stats()
    }

    /// Mesh `mesh` of `mount`'s page `page` where it streams: its address and its clusters'
    /// CLAS, or `None` while it is not resident. The outer `None`: the page does not stream.
    pub(crate) fn streamed_page(
        &self,
        mount: MountId,
        mesh: u32,
        page: u32,
    ) -> Option<Option<(u64, Vec<u64>)>> {
        let m = self.mounts.get(&mount)?;
        let unit = *m.mesh_pages.get(mesh as usize)?.get(page as usize)?;
        m.units[unit as usize]?;
        Some(match self.resident.get(&Key { mount, unit }) {
            Some(Placed::MeshPage(p)) => Some((p.address, p.clas_addresses.clone())),
            _ => None,
        })
    }

    /// Keeps `mirror` in step with mesh `mesh` of `mount` when its pages stream.
    pub(crate) fn add_mirror(&mut self, mount: MountId, mesh: u32, mirror: &Arc<Mirror>) {
        let Some(m) = self.mounts.get(&mount) else {
            return;
        };
        if let Some(s) = m.stream_of_mesh(mesh) {
            s.mirrors.lock().unwrap().push(Arc::clone(mirror));
        }
    }

    pub(crate) fn remove_mirror(&mut self, mount: MountId, mesh: u32, mirror: &Arc<Mirror>) {
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

    /// Adds a mount's units and points the stream table at its streamed textures.
    pub(crate) fn register(
        &mut self,
        mount: MountId,
        source: Arc<BlobSource>,
        units: MountUnits,
        w: &mut Apply<'_>,
    ) -> Result<()> {
        let mut textures: Vec<TexResidency> = Vec::with_capacity(units.textures.len());
        for (k, (stream, first_unit)) in units.textures.into_iter().enumerate() {
            let g = &stream.grid;
            let [cx, cy] = g.cells();
            let cells = cx * cy * g.layers;
            let Some(first_cell) = self.feedback.alloc((mount, k as u32), cells) else {
                for t in &textures {
                    self.feedback.free(t.first_cell);
                }
                return Err(Error::Capacity {
                    what: "texture feedback cells",
                    capacity: self.feedback.cell_capacity,
                });
            };
            textures.push(TexResidency {
                resident: vec![false; g.tile_count() as usize],
                finest: vec![g.tail_mip(); cells as usize],
                first_unit,
                first_cell,
                dirty: false,
                stream,
            });
        }
        let descs: Vec<UnitDesc> = units.units.iter().map(|(d, _)| d.clone()).collect();
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
            };
            w.put(s.record, &[record])?;
            w.put(self.streams.offset(u64::from(s.index.0)), &[s.record])?;
        }
        for init in units.fine {
            let slots = init
                .thresholds
                .iter()
                .map(|_| FineState::Empty(0))
                .collect();
            self.fine
                .insert((mount, init.mesh), MeshFine { init, slots });
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
        for k in cuts {
            let f = self.fine.remove(&k).expect("listed");
            for state in f.slots {
                let locks = match state {
                    FineState::Empty(_) => continue,
                    FineState::Building { locks } => locks,
                    FineState::Built { cut, locks, .. } => {
                        self.retired_cuts.push((frame, cut));
                        locks
                    }
                };
                self.fine_locked -= f.init.stream.pages_bytes(&locks);
            }
        }
        let gone: Vec<Key> = self
            .resident
            .keys()
            .filter(|k| k.mount == mount)
            .copied()
            .collect();
        for k in gone {
            match self.resident.remove(&k).expect("resident") {
                Placed::Pool(slot, _) => self.retired.push((frame, slot)),
                Placed::MeshPage(p) => {
                    self.retired.push((frame, p.slot));
                    let free = ClasFree {
                        clas: p.clas,
                        micromap: p.micromap,
                    };
                    self.retired_clas.push((frame, free));
                }
                Placed::Tile(slot) => {
                    let (t, tile) = m.tile(k.unit);
                    let free = TileFree {
                        key: k,
                        stream: Arc::clone(&t.stream),
                        tile,
                        slot,
                    };
                    self.retired_tiles.push((frame, free));
                }
            }
        }
        self.unbinding.retain(|k| k.mount != mount);
        self.moving.retain(|k, _| k.mount != mount);
        self.fragmented.retain(|k, _| k.0 != mount);
        for t in &m.textures {
            self.nulls.push(t.stream.index.0);
            self.retired_cells.push((frame, t.first_cell));
        }
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
    pub(crate) fn take_page_events(&mut self) -> Vec<(u32, u64)> {
        std::mem::take(&mut self.page_events)
    }

    /// Requests unit `unit` of `mount` with hint priority for the next step.
    pub(crate) fn request_unit(&mut self, mount: MountId, unit: u32, cell_distance: u32) {
        let p = Priority {
            error: 0.0,
            cell_distance,
        };
        self.residency.request(Key { mount, unit }, p);
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
            self.residency.request(Key { mount, unit }, p);
        }
    }

    /// Requests every tile of mips `finest..` of `mount`'s streamed texture at heap index
    /// `index` for the next step.
    pub(crate) fn prefetch_texture(&mut self, mount: MountId, index: u32, finest: u32) {
        let Some(t) = self
            .mounts
            .get(&mount)
            .and_then(|m| m.textures.iter().find(|t| t.stream.index.0 == index))
        else {
            return;
        };
        let p = Priority {
            error: 0.0,
            cell_distance: 0,
        };
        let g = &t.stream.grid;
        for tile in (0..g.tile_count()).filter(|&tile| g.tile(tile).mip >= finest) {
            let unit = t.first_unit + tile;
            self.residency.request(Key { mount, unit }, p);
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
            ),
            None => self.residency.touch(k, frame),
        }
    }

    /// Finer cut `k` of mesh `mesh` of `mount`: wanted, or reported in use in `frame`.
    pub(crate) fn fine_cut(&mut self, mount: MountId, mesh: u32, k: u32, wanted: bool, frame: u64) {
        let Some(f) = self.fine.get_mut(&(mount, mesh)) else {
            return;
        };
        match f.slots.get_mut(k as usize) {
            Some(FineState::Empty(at)) if wanted => *at = frame,
            Some(FineState::Built { touched, .. }) => *touched = frame,
            _ => {}
        }
    }

    /// Lands built finer cuts, retires those unused past their hold (only once `settled`, the
    /// last frame by which every scene has reported what it holds, has passed their last
    /// touch), asks for the pages of wanted ones and sends those whose pages are resident to
    /// be built.
    fn fine_step(
        &mut self,
        frame: u64,
        settled: u64,
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
                std::mem::replace(&mut f.slots[slot], FineState::Empty(0))
            else {
                unreachable!("a slot has one build at a time")
            };
            match batch.cut {
                Ok(cut) => {
                    w.put(at, &[cut.record])?;
                    f.slots[slot] = FineState::Built {
                        cut,
                        touched: frame,
                        locks,
                    };
                }
                // The slot stays empty, so a copy that still wants it asks again.
                Err(_) => {
                    let m = &self.mounts[&batch.mount];
                    for &p in &locks {
                        let unit = m.mesh_pages[batch.mesh as usize][p as usize];
                        self.residency.unlock(Key {
                            mount: batch.mount,
                            unit,
                        });
                    }
                    self.fine_locked -= f.init.stream.pages_bytes(&locks);
                }
            }
        }
        let budget = self
            .residency
            .stats()
            .class(ResidencyClass::MeshPages)
            .budget;
        for (&(mount, mesh), f) in &mut self.fine {
            let m = &self.mounts[&mount];
            let unit = |p: u32| Key {
                mount,
                unit: m.mesh_pages[mesh as usize][p as usize],
            };
            for k in 0..f.slots.len() {
                let t = f.init.thresholds[k];
                match &f.slots[k] {
                    FineState::Built { touched, .. }
                        if touched + HOLD_FRAMES < frame && *touched < settled =>
                    {
                        let FineState::Built { cut, locks, .. } =
                            std::mem::replace(&mut f.slots[k], FineState::Empty(0))
                        else {
                            unreachable!("matched above")
                        };
                        let empty = RtCutGpu {
                            error: t,
                            ..bytemuck::Zeroable::zeroed()
                        };
                        w.put(f.init.table.offset(k as u64), &[empty])?;
                        self.retired_cuts.push((frame - 1, cut));
                        for &p in &locks {
                            self.residency.unlock(unit(p));
                        }
                        self.fine_locked -= f.init.stream.pages_bytes(&locks);
                    }
                    FineState::Empty(at) if *at + 2 >= frame && *at != 0 => {
                        let absent: Vec<u32> = f.init.pages[k]
                            .iter()
                            .copied()
                            .filter(|&p| !self.residency.resident(unit(p)))
                            .collect();
                        let p = Priority {
                            error: 1.0,
                            cell_distance: 0,
                        };
                        if !absent.is_empty() {
                            for q in absent {
                                self.residency.request(unit(q), p);
                            }
                            continue;
                        }
                        let clusters = accel::cut_clusters(&f.init.clusters, t, f.init.merge);
                        let stream = &f.init.stream;
                        let mut pages: Vec<u32> = clusters
                            .iter()
                            .map(|&c| f.init.clusters[c as usize].page)
                            .filter(|&q| q >= stream.pinned)
                            .collect();
                        pages.sort_unstable();
                        pages.dedup();
                        let bytes = stream.pages_bytes(&pages);
                        if clusters.is_empty() || self.fine_locked + bytes > budget / 2 {
                            continue;
                        }
                        // A landing's CLAS may still be on its way into the pool.
                        if clusters.iter().any(|&c| f.init.clas[c as usize] == 0) {
                            continue;
                        }
                        for &q in &pages {
                            self.residency.lock(unit(q));
                        }
                        self.fine_locked += bytes;
                        f.slots[k] = FineState::Building { locks: pages };
                        loading.send(Msg::Cut(CutTask {
                            mount,
                            mesh,
                            slot: k as u32,
                            accel: Arc::clone(&self.accel),
                            clusters,
                            infos: f.init.clusters.clone(),
                            animated: f.init.animated,
                            clas: f.init.clas.clone(),
                        }));
                    }
                    _ => {}
                }
            }
        }
        Ok(points)
    }

    /// Hands the texture feedback of `frame`, read back in slot `slot`, to the next step:
    /// a wanted mip finer than the resident one requests the tile that would serve it, with
    /// the mips it would add as its error; one already resident is touched.
    pub(crate) fn texture_feedback(&mut self, slot: usize, frame: u64) {
        for ((mount, texture), cell, mip) in self.feedback.read(slot, frame) {
            let Some(t) = self
                .mounts
                .get(&mount)
                .and_then(|m| m.textures.get(texture as usize))
            else {
                continue;
            };
            let g = &t.stream.grid;
            let [cx, cy] = g.cells();
            let (layer, c) = (cell / (cx * cy), cell % (cx * cy));
            if mip >= g.tail_mip() || layer >= g.layers {
                continue;
            }
            let tile = g.index_of(g.covering(layer, mip, [c % cx, c / cx]));
            let k = Key {
                mount,
                unit: t.first_unit + tile,
            };
            let finest = t.finest[cell as usize];
            if mip < finest {
                let p = Priority {
                    error: (finest - mip) as f32,
                    cell_distance: 0,
                };
                self.residency.request(k, p);
            } else {
                self.residency.touch(k, frame);
            }
        }
    }

    /// Queues `blob` of `source` into `target`. It loads once the user blob budget and the
    /// in-flight limits have room, in request order.
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
                self.residency
                    .release_loose(ResidencyClass::UserBlobs, b.bytes);
                *slot = None;
            }
            BlobState::Loading { .. } => b.state = BlobState::Loading { released: true },
            BlobState::Failed(_) => *slot = None,
        }
    }

    /// Step 1 of frame `frame`: applies the landed batches, plans this step and sends its
    /// loads. Returns the points the frame's submit waits.
    pub(crate) fn step(
        &mut self,
        frame: u64,
        settled: u64,
        loading: &Arc<Loading>,
        w: &mut Apply<'_>,
    ) -> Result<Vec<TimelinePoint>> {
        let mut points = Vec::new();
        for mut batch in loading.take_streams(w.device) {
            points.extend(batch.points);
            let mut staging = self.staging.lock().unwrap();
            if let Some(d) = batch.decode {
                d.fail_pages(&mut batch.loads);
                staging.free(d.slot);
            }
            for l in &batch.loads {
                if let Some(s) = l.staging {
                    staging.free(s);
                }
            }
            drop(staging);
            let clas = match batch.clas {
                Some(c) => self.replay(c),
                None => Vec::new(),
            };
            for l in batch.loads {
                self.landed(l, &clas, frame, w)?;
            }
        }
        let slot = (frame % u64::from(self.fif)) as usize;
        self.feedback.begin(frame, slot, w)?;
        for index in std::mem::take(&mut self.nulls) {
            w.put(self.streams.offset(u64::from(index)), &[GpuPtr::NULL])?;
        }
        points.extend(self.fine_step(frame, settled, loading, w)?);
        self.residency.react(w.device.memory_report().pressure());
        if let Some(c) = &self.clas {
            let bytes = c.lock().unwrap().bound_bytes();
            self.residency.note_clas_pool(bytes);
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
            );
        }
        let Plan { loads, evictions } = self.residency.plan(frame);
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
        for m in self.mounts.values_mut() {
            for t in &mut m.textures {
                t.write_map(w)?;
            }
        }
        let (loads, waiting): (Vec<Key>, Vec<Key>) =
            loads.into_iter().partition(|k| !self.unbinding.contains(k));
        for k in waiting {
            self.residency.cancel(k);
        }
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
        while let Some(&id) = self.queued.front() {
            let b = self.blobs[id as usize].as_mut().expect("queued blob");
            if !self
                .residency
                .start_loose(ResidencyClass::UserBlobs, b.bytes)
            {
                break;
            }
            self.queued.pop_front();
            let state = std::mem::replace(&mut b.state, BlobState::Loading { released: false });
            let BlobState::Queued(r) = state else {
                unreachable!("a queued blob is queued")
            };
            loads.push(StreamLoad {
                what: What::Blob(id),
                source: r.source,
                kind: LoadKind::User {
                    blob: r.blob,
                    target: r.target,
                },
            });
        }
        let clas_waits = self
            .clas
            .as_ref()
            .is_some_and(|c| c.lock().unwrap().pending());
        if !loads.is_empty() || !self.unbinds.is_empty() || clas_waits {
            let task = StreamTask {
                accel: Arc::clone(&self.accel),
                clas: self.clas.clone(),
                pool: Arc::clone(&self.pool),
                staging: Arc::clone(&self.staging),
                tiles: Arc::clone(&self.tiles),
                device: w.device.clone(),
                outstanding: AtomicUsize::new(1),
                done: Mutex::new(Vec::with_capacity(loads.len())),
                copies: Mutex::new(Vec::new()),
                decodes: Mutex::new(Vec::new()),
                unbinds: std::mem::take(&mut self.unbinds),
                loads,
            };
            loading.send(Msg::Stream(Arc::new(task)));
        }
        Ok(points)
    }

    /// Gives a stream pool unit's target its address, 0 when the unit leaves.
    fn point_target(&mut self, target: PoolTarget, at: u64, w: &mut Apply<'_>) -> Result<()> {
        match target {
            PoolTarget::Table(table) => w.put(table, &[at])?,
            PoolTarget::InstancePage(page) => self.page_events.push((page, at)),
        }
        Ok(())
    }

    /// Writes evicted unit `k` out of its tables; its memory returns once the frame before
    /// has completed.
    fn evict_placed(&mut self, k: Key, frame: u64, w: &mut Apply<'_>) -> Result<()> {
        match self.resident.remove(&k).expect("evicted unit is resident") {
            Placed::Pool(slot, target) => {
                self.point_target(target, 0, w)?;
                self.retired.push((frame - 1, slot));
            }
            Placed::MeshPage(p) => {
                let (table, refs, count) = p.entries();
                let zeros = vec![bytemuck::Zeroable::zeroed(); count as usize];
                w.put(table, &[0])?;
                w.put(refs, &zeros)?;
                let first = p.stream.pages[p.page as usize].first;
                if let Some(f) = self.fine.get_mut(&(k.mount, p.stream.mesh)) {
                    let at = first as usize;
                    f.init.clas[at..at + count as usize].fill(0);
                }
                for mirror in p.stream.mirrors.lock().unwrap().iter() {
                    if !mirror.own[p.page as usize] {
                        w.put(mirror.table.offset(u64::from(p.page)), &[0])?;
                        w.put(mirror.clusters.offset(u64::from(first)), &zeros)?;
                    }
                }
                self.retired.push((frame - 1, p.slot));
                let free = ClasFree {
                    clas: p.clas,
                    micromap: p.micromap,
                };
                self.retired_clas.push((frame - 1, free));
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
                self.retired_tiles.push((frame - 1, free));
            }
        }
        Ok(())
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
    fn replay(&mut self, batch: PoolBatch) -> Vec<(u32, u32)> {
        let pool = self.clas.as_ref().expect("a CLAS batch has its pool");
        let (placed, held) = pool.lock().unwrap().replay(batch);
        let mut staging = self.staging.lock().unwrap();
        for slot in held {
            staging.free(slot);
        }
        placed
    }

    fn landed(
        &mut self,
        mut l: Landed,
        clas: &[(u32, u32)],
        frame: u64,
        w: &mut Apply<'_>,
    ) -> Result<()> {
        let ok = l.result.is_ok();
        if let (What::Unit(key), Some(Placed::MeshPage(_))) = (l.what, &l.placed) {
            return self.mesh_landed(key, l, clas, frame, w);
        }
        match l.what {
            What::Unit(key) => {
                let live = self.residency.landed(key, ok, frame);
                match (live && ok, l.placed.take()) {
                    (true, Some(Placed::Pool(slot, target))) => {
                        self.point_target(target, l.gpu, w)?;
                        self.resident.insert(key, Placed::Pool(slot, target));
                    }
                    (true, Some(Placed::Tile(slot))) => {
                        let m = self.mounts.get_mut(&key.mount).expect("live mount");
                        let (t, tile) = m.tile(key.unit);
                        t.resident[tile as usize] = true;
                        t.dirty = true;
                        self.resident.insert(key, Placed::Tile(slot));
                    }
                    (_, Some(Placed::Pool(slot, _))) => self.pool.lock().unwrap().free(slot),
                    (_, Some(Placed::MeshPage(_))) => unreachable!("mesh pages land apart"),
                    (_, Some(Placed::Tile(slot))) => {
                        let (stream, tile) = l.tile.expect("a tile load names its tile");
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
                self.residency
                    .loose_landed(ResidencyClass::UserBlobs, b.bytes, ok);
                if released {
                    if ok {
                        self.residency
                            .release_loose(ResidencyClass::UserBlobs, b.bytes);
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
    /// entries are written, or with a CLAS the pool could not place its load is cancelled to
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
        let mesh = l.mesh.take();
        let landed_clusters = mesh.as_ref().map(|m| m.clusters.clone());
        let units: Vec<(u32, u32)> = match (&mesh, &self.clas) {
            (Some(m), Some(_)) => m.slots.iter().map(|&s| clas[s as usize]).collect(),
            _ => Vec::new(),
        };
        p.micromap = mesh.and_then(|m| m.micromap);
        p.clas = units
            .iter()
            .copied()
            .filter(|&(u, _)| u != NO_UNIT)
            .collect();
        let placed = l.result.is_ok() && units.iter().all(|&(u, _)| u != NO_UNIT);
        let live = if !self.mounts.contains_key(&key.mount) {
            false
        } else if l.result.is_ok() && !placed {
            self.residency.cancel(key);
            false
        } else {
            self.residency.landed(key, l.result.is_ok(), frame)
        };
        if !(live && placed) {
            self.free_now(p);
            return Ok(());
        }
        let (table, refs, count) = p.entries();
        let first = p.stream.pages[p.page as usize].first;
        let pool = self.clas.as_ref().map(|c| c.lock().unwrap());
        let entries: Vec<ClusterRef> = (0..count)
            .map(|i| ClusterRef {
                // SAFETY: the page is resident in its slot until it is evicted, which writes
                // these entries null first.
                page: unsafe { GpuPtr::from_address(GpuAddress(l.gpu)) },
                local: i,
                mesh_cluster: first + i,
                clas: match (&pool, units.get(i as usize)) {
                    (Some(pool), Some(&(u, _))) => pool.address(u),
                    _ => 0,
                },
            })
            .collect();
        drop(pool);
        w.put(table, &[l.gpu])?;
        w.put(refs, &entries)?;
        for mirror in p.stream.mirrors.lock().unwrap().iter() {
            if !mirror.own[p.page as usize] {
                w.put(mirror.table.offset(u64::from(p.page)), &[l.gpu])?;
                w.put(mirror.clusters.offset(u64::from(first)), &entries)?;
            }
        }
        p.address = l.gpu;
        p.clas_addresses = entries.iter().map(|e| e.clas).collect();
        if let Some(f) = self.fine.get_mut(&(key.mount, p.stream.mesh)) {
            let at = first as usize;
            if let Some(clusters) = landed_clusters {
                f.init.clusters[at..at + clusters.len()].clone_from_slice(&clusters);
            }
            let n = p.clas_addresses.len();
            f.init.clas[at..at + n].copy_from_slice(&p.clas_addresses);
        }
        self.resident.insert(key, Placed::MeshPage(p));
        Ok(())
    }

    /// Frees a mesh page no frame has read.
    fn free_now(&mut self, p: MeshPlaced) {
        self.pool.lock().unwrap().free(p.slot);
        self.free_clas(ClasFree {
            clas: p.clas,
            micromap: p.micromap,
        });
    }

    fn free_clas(&mut self, f: ClasFree) {
        let Some(pool) = &self.clas else { return };
        let mut pool = pool.lock().unwrap();
        for (u, n) in f.clas {
            pool.free(u, n);
        }
        if let Some((micromap, slot)) = f.micromap {
            drop(micromap);
            pool.micromaps.free(slot);
        }
    }

    /// Copies frame `frame`'s texture feedback back after every tap of the frame on `list`.
    pub(crate) fn record_feedback(&self, list: &mut CommandList<'_>, frame: u64) {
        let slot = (frame % u64::from(self.fif)) as usize;
        self.feedback.record_end(list, slot);
    }

    /// Frees the slots retired at or before `completed`; the tiles among them unbind with
    /// the next task.
    pub(crate) fn retire(&mut self, completed: u64) {
        self.retired_cuts.retain(|(frame, _)| *frame > completed);
        let mut pool = self.pool.lock().unwrap();
        self.retired.retain(|&(frame, slot)| {
            let done = frame <= completed;
            if done {
                pool.free(slot);
            }
            !done
        });
        drop(pool);
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired_clas)
            .into_iter()
            .partition(|(f, _)| *f <= completed);
        self.retired_clas = keep;
        for (_, f) in done {
            self.free_clas(f);
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
    },
}

struct StreamLoad {
    what: What,
    source: Arc<BlobSource>,
    kind: LoadKind,
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

/// One step's loads on the loading side.
pub(crate) struct StreamTask {
    accel: Arc<AccelShared>,
    clas: Option<Arc<Mutex<ClasPool>>>,
    pool: Arc<Mutex<Pool>>,
    staging: Arc<Mutex<Pool>>,
    tiles: Arc<Mutex<Pool>>,
    device: Device,
    loads: Vec<StreamLoad>,
    unbinds: Vec<TileFree>,
    /// The start chain plus one per read.
    outstanding: AtomicUsize,
    done: Mutex<Vec<Landed>>,
    copies: Mutex<Vec<Copy>>,
    /// The staged mesh pages to decode, and each one's cluster count.
    decodes: Mutex<Vec<(StoredPage, u32)>>,
}

/// A finished load.
pub(crate) struct Landed {
    what: What,
    placed: Option<Placed>,
    /// The pool unit's address.
    gpu: u64,
    /// A tile's texture and index, for its bind and copy.
    tile: Option<(Arc<TexStream>, u32)>,
    /// The staging slot and its address.
    staging: Option<PoolSlot>,
    staged_at: u64,
    /// A mesh page's clusters as it decoded, and what its CLAS build made.
    mesh: Option<MeshLanded>,
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
    micromap: Option<(Micromap, PoolSlot)>,
}

/// A stream task's loads, handed to the render thread once its GPU work has landed.
pub(crate) struct StreamBatch {
    pub points: Vec<TimelinePoint>,
    loads: Vec<Landed>,
    clas: Option<PoolBatch>,
    decode: Option<PageDecodes>,
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
    /// Fails the landed mesh pages whose decode set a status bit.
    fn fail_pages(&self, loads: &mut [Landed]) {
        let mut status = vec![0u32; self.count as usize];
        let ok = self
            .block
            .read_mapped(self.status_at, bytemuck::cast_slice_mut(&mut status));
        assert!(ok, "the decode status is mapped");
        for l in loads {
            let failed = l
                .mesh
                .as_ref()
                .is_some_and(|m| status[m.decode as usize] != 0);
            if failed && l.result.is_ok() {
                let e = orr_pack::Error::Malformed("streamed mesh page streams");
                l.result = Err(Arc::new(e.into()));
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
    fn end(self: &Arc<Self>, loading: &Loading) {
        if self.outstanding.fetch_sub(1, Ordering::AcqRel) == 1 {
            loading.send(Msg::StreamDone(Arc::clone(self)));
        }
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
            result: Err(Arc::new(e)),
        });
    }

    /// Returns what a failed load took.
    fn free(&self, placed: Option<Placed>, staging: Option<PoolSlot>) {
        match placed {
            Some(Placed::Pool(slot, _)) => self.pool.lock().unwrap().free(slot),
            Some(Placed::Tile(slot)) => self.tiles.lock().unwrap().free(slot),
            Some(Placed::MeshPage(p)) => self.pool.lock().unwrap().free(p.slot),
            None => {}
        }
        if let Some(s) = staging {
            self.staging.lock().unwrap().free(s);
        }
    }
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
        LoadKind::MeshPage { stream, page } => {
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
    Ok((b, chunk, len, slot))
}

/// Places a stream task's loads on the loading thread: memory for each, then its read.
pub(crate) fn start(loading: &Arc<Loading>, task: &Arc<StreamTask>) {
    // Each texture's last tile slot, so its next tile goes beside it.
    let mut near: HashMap<*const TexStream, PoolSlot> = HashMap::new();
    for (i, load) in task.loads.iter().enumerate() {
        let (b, chunk, len, slot_len) = match source_of(load) {
            Ok(s) => s,
            Err(e) => {
                task.fail(load.what, e);
                continue;
            }
        };
        let (placed, block) = match &load.kind {
            LoadKind::Pool { target, .. } => {
                match task.pool.lock().unwrap().alloc(&task.device, len, None) {
                    Ok((slot, block)) => (Some(Placed::Pool(slot, *target)), Some(block)),
                    Err(e) => {
                        task.fail(load.what, e);
                        continue;
                    }
                }
            }
            LoadKind::MeshPage { stream, page } => {
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
                            address: 0,
                            clas_addresses: Vec::new(),
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
                let key = Arc::as_ptr(stream);
                let slot = task.tiles.lock().unwrap().alloc(
                    &task.device,
                    len,
                    after.or_else(|| near.get(&key).copied()),
                );
                match slot {
                    Ok((slot, _)) => {
                        near.insert(key, slot);
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
        // Bytes go straight into a mapped pool slot, else into the staging pool; mesh pages
        // stage their stored form for the decode.
        let mesh_page = matches!(load.kind, LoadKind::MeshPage { .. });
        let direct = !mesh_page && block.is_some_and(|b| b.cpu().is_some());
        let staged_len = if mesh_page { len } else { slot_len };
        let staging = if direct {
            None
        } else {
            let slot = task
                .staging
                .lock()
                .unwrap()
                .alloc(&task.device, staged_len, None);
            match slot {
                Ok(s) => Some(s),
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
        read_stored(
            &load.source,
            b,
            chunk,
            Box::new(move |stored| {
                let load = &t.loads[i];
                let mut mesh = None;
                let result = stored.and_then(|stored| {
                    decoded(&load.source.archive, &b, chunk, stored, |bytes| {
                        if bytes.len() as u64 != len {
                            return Err(orr_pack::Error::Malformed("streamed blob length").into());
                        }
                        let dst = staging.map_or_else(|| block.expect("a pool slot"), |s| s.1);
                        if let LoadKind::MeshPage { stream, page } = &load.kind {
                            let to = (block.expect("a pool slot").gpu(), slot_len);
                            mesh = Some(stage_page(stream, *page, bytes, &dst, to, &t.decodes)?);
                            return Ok(());
                        }
                        let wrote = dst.write_mapped(0, bytes);
                        assert!(wrote, "a mapped slot");
                        Ok(())
                    })
                });
                let mut landed = Landed {
                    what: load.what,
                    placed,
                    gpu: block.map_or(0, |b| b.gpu()),
                    tile: match &load.kind {
                        LoadKind::Tile { stream, tile, .. } => Some((Arc::clone(stream), *tile)),
                        _ => None,
                    },
                    staging: staging.map(|s| s.0),
                    staged_at: staging.map_or(0, |s| s.1.gpu()),
                    mesh,
                    result: result.map_err(Arc::new),
                };
                if landed.result.is_err() {
                    t.free(landed.placed.take(), landed.staging.take());
                } else if let Some((_, s)) = staging {
                    let copy = match (&load.kind, block) {
                        (LoadKind::Pool { .. }, Some(block)) => Some(Copy::Buffer {
                            src: s.gpu(),
                            dst: block.gpu(),
                            bytes: slot_len,
                        }),
                        (LoadKind::User { target, .. }, _) => Some(match target {
                            BlobTarget::Buffer { at, .. } => Copy::Buffer {
                                src: s.gpu(),
                                dst: at.address().0,
                                bytes: len,
                            },
                            BlobTarget::Texture { texture, region } => Copy::Texture {
                                src: s.gpu(),
                                texture: Arc::clone(texture),
                                region: *region,
                            },
                        }),
                        _ => None,
                    };
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

/// Writes a streamed mesh page's stored bytes into its staging slot `staged` and queues
/// their decode into the pool slot at `to` (address and bytes); reads its clusters from its
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
    let wrote = staged.write_mapped(0, stored);
    assert!(wrote, "a mapped staging slot");
    let (clusters, region) = page_infos(&h, records, page);
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
/// ones, copies the tiles in on the loading queue, submits the other copies, builds the
/// streamed mesh pages' CLAS and hands the batch to the render thread.
pub(crate) fn finish(loading: &Loading, device: &Device, task: &Arc<StreamTask>) {
    let mut loads = std::mem::take(&mut *task.done.lock().unwrap());
    let copies = std::mem::take(&mut *task.copies.lock().unwrap());
    let mut decode = None;
    let (points, clas) = match submit(device, task, &mut loads, &copies, &mut decode) {
        Ok(p) => p,
        Err(e) => {
            let e = Arc::new(e);
            for l in &mut loads {
                l.result = Err(Arc::clone(&e));
            }
            (Vec::new(), None)
        }
    };
    loading.push_stream(StreamBatch {
        points,
        loads,
        clas,
        decode,
    });
}

/// Submits a task's GPU work. `decode` takes the mesh page decode as soon as it is
/// submitted, so its slot returns whatever fails after it.
fn submit(
    device: &Device,
    task: &StreamTask,
    loads: &mut [Landed],
    copies: &[Copy],
    decode: &mut Option<PageDecodes>,
) -> Result<(Vec<TimelinePoint>, Option<PoolBatch>)> {
    let mut points = Vec::new();
    *decode = decode_pages(device, task, &mut points)?;
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
    for u in &task.unbinds {
        tiles.free(u.slot);
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
        points.push(queue.submit(vec![list], &points)?);
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
    let clas = match &task.clas {
        Some(pool) => build_clas(device, task, pool, loads, &mut points)?,
        None => None,
    };
    Ok((points, clas))
}

/// Builds the CLAS of a batch's streamed mesh pages on the loading queue after its other
/// work, and applies the pool's queued frees and takes; none when there is neither.
fn build_clas(
    device: &Device,
    task: &StreamTask,
    pool: &Mutex<ClasPool>,
    loads: &mut [Landed],
    points: &mut Vec<TimelinePoint>,
) -> Result<Option<PoolBatch>> {
    let mut pool = pool.lock().unwrap();
    let pages: Vec<usize> = (0..loads.len())
        .filter(|&i| loads[i].result.is_ok() && loads[i].mesh.is_some())
        .collect();
    if pages.is_empty() && !pool.pending() {
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
                cluster_base: p.stream.cluster_base + p.stream.pages[p.page as usize].first,
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
        &mut pool,
        device,
        k,
        &mut list,
        &builds,
        triangles,
        &task.staging,
    )?;
    drop(builds);
    let mut wait = points.clone();
    wait.extend(out.bind);
    points.push(queue.submit(vec![list], &wait)?);
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
    let mut list = queue.command_list()?;
    let k = &task.accel.kernels;
    let root = address::<PageDecodeRoot>(block.gpu() + root_at);
    list.dispatch(&k.page_copy, root, [dwords.div_ceil(256), rows, 1]);
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    list.dispatch(&k.page_decode, root, [clusters.max(1), rows, 1]);
    points.push(queue.submit(vec![list], &[])?);
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
