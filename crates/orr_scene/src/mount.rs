//! Packs mounted from a file or from memory. The loading side reads and places every asset
//! of a pack except what streams on request: the hierarchy pages below a mesh's top page, a
//! large DAG mesh's finer pages, instance pages' records and the tiles above the mip tails of
//! the textures its materials bind and of neural sets' latents. The render thread takes the
//! finished batch at step 1 and applies its tables with the frame's scatter.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use orr_gpu::{
    Allocation, Device, GpuPtr, Hazards, MemoryKind, MemoryPriority, QueueKind, SamplerIndex,
    SparseBinds, Stages, Texture, TextureDesc, TextureIndex, TextureKind, TextureMemory,
    TextureRegion, TextureUsage, TimelinePoint, ViewDesc,
};
use orr_io::{AlignedBuf, Read, SECTOR};
use orr_math::{DVec3, IVec3, Vec3};
use orr_pack::instance::{InstancePageInfo, InstanceRecord};
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{Cluster, Group, MeshHeader, MeshInfo, NodePage, PageHeader, PageView, Part};
use orr_pack::model::ModelInfo;
use orr_pack::neural::{Constants, Fp8Network, NeuralInfo, NeuralTexture};
use orr_pack::subd::{Page as SubdPage, SubdInfo};
use orr_pack::texture::{TextureHeader, TextureInfo};
use orr_pack::tiles::TextureTiles;
use orr_pack::volume::VolumeInfo;
use orr_pack::{Aabb, Archive, AssetId, Class, Codec};
use rustc_hash::FxHashMap;

use crate::accel::{
    self, AccelShared, ClusterInfo, CutRow, MaskBlas, MeshInput, MeshRt, MountAccel, PageMicromap,
};
use crate::edit::{
    self, ClasInputs, EDIT_SPHERES, EditBatch, EditClas, EditDesc, EditDone, EditId, EditLoaded,
    EditRecorder, EditRef, EditSkin, EditSource, EditTables, EditTablesDesc, EditTask, Mirror,
};
use crate::lights::GATHERED_CLUSTERS;
use crate::loading::{
    ArchiveFile, BlobSource, Loading, Msg, Queue, Resumable, Ticket, completion, decoded,
    read_stored,
};
use crate::material::{BoundTexture, MaterialId, MaterialTable, binding_sampler};
use crate::medium_proxy::VolumeRef;
use crate::memory::{Block, RangeAllocator, UploadWindow};
use crate::neural::{self, NeuralLink, Precision};
use crate::residency::{Key, RequestSource, ResidencyClass, Start, UnitDesc};
use crate::scene::{ModelCpu, ModelId, SkinShape};
use crate::shaders::types::SubdCageGpu;
use crate::skin;
use crate::stream::{
    FineInit, Hold, MeshStream, MountUnits, PageClusters, PoolTarget, StreamUnit, StreamedPage,
    Streaming,
};
use crate::subd::{CageShape, CageTables, ModelCage};
use crate::tiles::{TexStream, TileGrid};
use crate::traverse::{PAGED_DEPTH, PageFeedback};
use crate::upload::Record;
use crate::{
    ClusterRef, EditGroupGpu, Error, MaskGpu, MaterialGpu, MeshGpu, ModelGpu, NeuralSetGpu,
    NeuralTextureGpu, PartGpu, Result, SetTablesGpu, SkinMeshGpu, TextureStreamGpu, material_flags,
};

mod probe;

/// Where a pack's bytes come from.
#[derive(Clone, Debug)]
pub enum PackSource {
    /// An archive file. Its handles open at mount with read and delete sharing, so a re-cook
    /// can rename a new archive over the path; the mount keeps reading the old file until it
    /// is remounted.
    File(PathBuf),
    /// A whole archive in memory, such as an orr_cook cook run in the consumer's process.
    Memory(Arc<[u8]>),
}

/// A mount's progress, polled with [`crate::Scenes::mount_state`].
#[derive(Clone, Debug)]
pub enum LoadState {
    Pending,
    /// Every asset is resident and its lookups succeed.
    Ready,
    Failed(Arc<Error>),
}

/// A mounted pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MountId(pub u32);

/// A mounted instance page. It names no page once its pack is gone, whatever later takes its
/// slot in the page table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
}

/// A heap image slot, sampling a texture, that returns to the device once the work submitted
/// before it drops has completed: the device can outlive the scenes that placed it.
struct ImageSlot {
    device: Device,
    index: TextureIndex,
}

impl ImageSlot {
    fn sampling(device: &Device, texture: &Texture) -> Result<Self> {
        let slot = device.alloc_image_slot()?;
        Ok(Self {
            device: device.clone(),
            index: device.write_sampled(slot, texture, &ViewDesc::default()),
        })
    }
}

impl Drop for ImageSlot {
    fn drop(&mut self) {
        self.device.free_image_slot_after_submitted(self.index.0);
    }
}

/// Heap image slots and their textures.
pub(crate) struct PlacedTexture {
    id: AssetId,
    texture: Arc<Texture>,
    slot: ImageSlot,
    primaries: u8,
    /// Bytes resident from mount: the whole texture, or a streamed one's mip tails.
    bytes: u64,
    /// How a streamed texture's tiles load; none for whole textures.
    stream: Option<Arc<TexStream>>,
    /// The neural set that stands in for the texture.
    neural: Option<NeuralLink>,
}

struct PlacedMesh {
    id: AssetId,
    entry: u32,
    /// Blob 0, which holds the header's tables, in the mesh's block.
    blob0: u64,
    pages: Vec<u64>,
    node_pages: Vec<u64>,
    /// Per page, its resident bytes, the pages it depends on and the node pages past the top
    /// one that hold its groups' leaves.
    page_units: Vec<(u64, Vec<u32>, Vec<u32>)>,
    /// Per node page, its blob, bytes and parent page.
    node_units: Vec<(u32, u32, u32)>,
    /// First mesh cluster per page, then the cluster count, in the mesh's block.
    page_clusters: u64,
    cluster_count: u32,
    sections: u32,
    header: MeshHeader,
    /// The ranks of its emissive and caster sets.
    ranks: [u32; 2],
    /// Whether copies of the mesh trace cached cuts for each purpose's sets: a cut can hold more
    /// clusters of one section than a gathered list keeps.
    cached_only: [bool; 2],
    /// Discrete levels, finest first: error and mesh cluster range.
    level_ranges: Vec<(f32, u32, u32)>,
    /// The clusters of its pinned pages by mesh cluster index, as their pages decoded.
    clusters: Vec<(u32, ClusterInfo)>,
    /// Per page, its micromap region.
    micromaps: Vec<Option<PageMicromap>>,
    parts: Vec<Part>,
    /// Levels of a resident hierarchy; None when it is paged.
    depth: Option<u32>,
    /// Pages past the pinned ones stream: per page its blob, bytes and clusters.
    streamed: Option<Vec<StreamedPage>>,
    /// DAG mode: the cuts whose clusters are all resident from mount.
    cached: Vec<CutRow>,
    /// What edits of the mesh read: its groups, and every page's blob, bytes and clusters.
    groups: Arc<[Group]>,
    all_pages: Vec<StreamedPage>,
    /// Its finer cached cuts where its pages stream.
    fine: Vec<CutRow>,
}

/// The levels of a hierarchy from `roots`, whose nodes follow their parents.
fn hierarchy_depth(nodes: &[orr_pack::mesh::Node], roots: &[u32]) -> u32 {
    let mut depth = vec![0u32; nodes.len()];
    let mut levels = 0;
    for &r in roots.iter().filter(|&&r| r != MeshHeader::NO_ROOT) {
        if let Some(d) = depth.get_mut(r as usize) {
            *d = 1;
        }
    }
    for (i, n) in nodes.iter().enumerate() {
        levels = levels.max(depth[i]);
        if n.group >= 0 {
            continue;
        }
        let children = n.child_first as usize..n.child_first as usize + usize::from(n.child_count);
        let next = depth[i] + 1;
        for c in children {
            if let Some(d) = depth.get_mut(c) {
                *d = next;
            }
        }
    }
    levels
}

/// A resident skeleton: its blob (`orr_pack::skeleton`), which pose kernels read in place
/// through `orr_pack.skeleton`, valid until its pack is unmounted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkeletonRef {
    pub blob: GpuPtr<u8>,
    pub joint_count: u32,
}

/// A resident clip (`orr_pack::clip`): its header and samples, which pose kernels read in
/// place through `orr_pack.skeleton`, valid until its pack is unmounted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipRef {
    pub header: GpuPtr<u8>,
    pub samples: GpuPtr<u8>,
    pub skeleton: AssetId,
    pub joint_count: u32,
    pub frame_count: u32,
    pub sample_rate: f32,
    pub duration: f32,
}

/// A mounted instance page: its tables on the CPU, valid until its pack is unmounted. Its
/// records stream: placed in a scene, or requested with [`crate::Scenes::prefetch_page`] or
/// [`crate::Scenes::prefetch_region`], they land in the instance page pool. Its consumer data
/// (blob 2) is requested with [`crate::Scenes::request_blob`].
#[derive(Debug)]
pub struct InstancePage {
    pub info: InstancePageInfo,
    /// Null while the records are not resident. The page's exact table follows them.
    pub records: GpuPtr<InstanceRecord>,
    /// Per window of an animated page's rows, its address, 0 while it is not resident.
    pub windows: Vec<u64>,
    /// Bytes of the records and the table.
    pub bytes: u32,
    /// The mount holding it, whose models its prototypes name first.
    pub mount: MountId,
}

struct PlacedPage {
    id: AssetId,
    info: InstancePageInfo,
    entry: u32,
    bytes: u32,
}

/// A prototype of an instance page, resolved to this run's ids.
pub(crate) struct PagePrototype {
    pub(crate) model: ModelId,
    /// The page's records that name it.
    pub(crate) records: u32,
    /// Its section materials where it overrides any.
    pub(crate) sets: Option<Vec<MaterialId>>,
    pub(crate) mask: u16,
    pub(crate) pose: u8,
    pub(crate) flags: u32,
}

pub(crate) enum StagedCopy {
    /// A texture's first use, ahead of its copies: the mip tails of its stream bound, then its
    /// contents discarded.
    Discard {
        texture: Arc<Texture>,
        stream: Option<Arc<TexStream>>,
    },
    Buffer {
        src: Allocation<u8>,
        dst: u64,
    },
    Texture {
        src: Allocation<u8>,
        texture: Arc<Texture>,
        region: TextureRegion,
    },
}

impl StagedCopy {
    /// Its staging bytes.
    fn bytes(&self) -> u64 {
        match self {
            Self::Discard { .. } => 0,
            Self::Buffer { src, .. } | Self::Texture { src, .. } => src.size(),
        }
    }
}

/// A mount's staged copies not yet submitted, with their bytes in the in-flight limit, and
/// the points of the batches the loading thread has submitted.
struct Staging {
    copies: Vec<StagedCopy>,
    ticket: Ticket,
    /// The latest batch, whose completion follows the earlier ones on its queue.
    last: Option<TimelinePoint>,
    /// The latest bind of mip tails, which every later batch waits on.
    bound: Option<TimelinePoint>,
}

/// A subdivision cage's tables, resident in a block of the mount.
#[derive(Clone, Copy)]
struct PlacedCage {
    id: AssetId,
    cage: GpuPtr<SubdCageGpu>,
    points: GpuPtr<[f32; 3]>,
    shape: CageShape,
}

/// What the jobs of one mount have placed so far.
#[derive(Default)]
struct Placed {
    blocks: Vec<Allocation<u8>>,
    textures: Vec<PlacedTexture>,
    materials: Vec<(AssetId, Material)>,
    meshes: Vec<PlacedMesh>,
    models: Vec<(AssetId, ModelInfo)>,
    pages: Vec<PlacedPage>,
    skeletons: Vec<(AssetId, SkeletonRef)>,
    clips: Vec<(AssetId, ClipRef)>,
    cages: Vec<PlacedCage>,
    volumes: Vec<(AssetId, VolumeRef)>,
    neural: Vec<PlacedNeural>,
    tiles: Vec<PlacedTiles>,
}

/// A texture tile set, and the table `resolve` fills in a block of the mount.
struct PlacedTiles {
    id: AssetId,
    set: TextureTiles,
    table: GpuPtr<u32>,
}

/// A neural texture set: its weights in a block of the mount, and its record, written by
/// the render thread once its latents are placed.
struct PlacedNeural {
    id: AssetId,
    latents: AssetId,
    record: GpuPtr<NeuralSetGpu>,
    gpu: NeuralSetGpu,
}

/// One mount on the loading side, shared by its jobs and reads.
pub(crate) struct MountTask {
    id: MountId,
    loading: Weak<Loading>,
    device: Device,
    /// Writes go through staging and copies (no resizable BAR).
    staged: bool,
    host_copy: bool,
    /// The sampler of neural sets' latents.
    sampler: SamplerIndex,
    /// The precision neural sets decode at; none when textures sample their BC entries.
    neural: Option<Precision>,
    cancel: Arc<AtomicBool>,
    outstanding: AtomicUsize,
    error: Mutex<Option<Error>>,
    archive: OnceLock<Arc<Archive>>,
    file: OnceLock<Arc<ArchiveFile>>,
    ring: OnceLock<bool>,
    placed: Mutex<Placed>,
    staging: Mutex<Staging>,
    /// Staged bytes at which the loading thread submits a batch while the mount places: a
    /// quarter of the in-flight limit, so batches copy while the reads after them run.
    batch: u64,
    accel: Arc<AccelShared>,
    /// The device builds cluster acceleration structures.
    traced: bool,
}

impl MountTask {
    pub(crate) fn new(
        id: MountId,
        loading: &Arc<Loading>,
        device: &Device,
        sampler: SamplerIndex,
        neural: Option<Precision>,
        cancel: Arc<AtomicBool>,
        accel: &Arc<AccelShared>,
    ) -> Self {
        let caps = device.caps();
        Self {
            id,
            loading: Arc::downgrade(loading),
            device: device.clone(),
            staged: !caps.rebar,
            host_copy: caps.host_image_copy,
            sampler,
            neural,
            cancel,
            // The chain from the archive's first read to the placement plan.
            outstanding: AtomicUsize::new(1),
            error: Mutex::new(None),
            archive: OnceLock::new(),
            file: OnceLock::new(),
            ring: OnceLock::new(),
            placed: Mutex::new(Placed::default()),
            staging: Mutex::new(Staging {
                copies: Vec::new(),
                ticket: Ticket::empty(Arc::downgrade(loading)),
                last: None,
                bound: None,
            }),
            batch: loading.in_flight().limit() / 4,
            accel: Arc::clone(accel),
            traced: caps.cluster.max_tris > 0 && caps.cluster.max_partitions > 0,
        }
    }

    fn live(&self) -> bool {
        !self.cancel.load(Ordering::Relaxed) && self.error.lock().unwrap().is_none()
    }

    fn fail(&self, e: Error) {
        self.error.lock().unwrap().get_or_insert(e);
    }

    fn begin(&self) {
        self.outstanding.fetch_add(1, Ordering::AcqRel);
    }

    fn end(self: &Arc<Self>) {
        if self.outstanding.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(l) = self.loading.upgrade()
        {
            l.send(Msg::Done(Arc::clone(self)));
        }
    }

    /// Runs `f` unless the mount was canceled or failed, records its error, and ends one
    /// outstanding step.
    fn step(self: &Arc<Self>, f: impl FnOnce() -> Result<()>) {
        if self.live()
            && let Err(e) = f()
        {
            self.fail(e);
        }
        self.end();
    }

    fn spawn(self: &Arc<Self>, job: impl FnOnce(&Arc<Self>) + Send + 'static) {
        let t = Arc::clone(self);
        match self.loading.upgrade() {
            Some(l) => l.queue(Queue::Mount, move || job(&t)),
            None => self.end(),
        }
    }

    fn archive(&self) -> &Arc<Archive> {
        self.archive.get().expect("archive parsed")
    }

    fn source(&self) -> BlobSource {
        BlobSource {
            loading: Weak::clone(&self.loading),
            archive: Arc::clone(self.archive()),
            file: self.file.get().cloned(),
            ring: *self.ring.get().expect("read path chosen"),
        }
    }

    fn block(&self, bytes: u64, name: &str) -> Result<Block> {
        let kind = if self.staged {
            MemoryKind::DeviceLocal
        } else {
            MemoryKind::Upload
        };
        let a = self.device.gpu_malloc::<u8>(kind, bytes, name)?;
        let b = Block::of(&a);
        self.placed.lock().unwrap().blocks.push(a);
        Ok(b)
    }

    fn write(&self, block: Block, at: u64, bytes: &[u8]) -> Result<()> {
        if let Some(copy) = write(&self.device, block, at, bytes)? {
            self.stage(copy);
        }
        Ok(())
    }

    fn write_texture(&self, t: &Arc<Texture>, region: &TextureRegion, texels: &[u8]) -> Result<()> {
        if self.host_copy {
            self.device.host_copy_to_texture(t, region, texels)?;
        } else {
            self.stage_texture(t, region, texels)?;
        }
        Ok(())
    }

    fn stage_texture(&self, t: &Arc<Texture>, region: &TextureRegion, texels: &[u8]) -> Result<()> {
        self.stage(StagedCopy::Texture {
            src: staging(&self.device, texels)?,
            texture: Arc::clone(t),
            region: *region,
        });
        Ok(())
    }

    /// Queues `copy` for the loading thread's next batch, its bytes in flight until that
    /// completes, and wakes the loading thread once the queued bytes reach a batch.
    fn stage(&self, copy: StagedCopy) {
        let mut s = self.staging.lock().unwrap();
        let before = s.ticket.bytes();
        s.ticket.add(copy.bytes());
        s.copies.push(copy);
        if before < self.batch
            && s.ticket.bytes() >= self.batch
            && let Some(l) = self.loading.upgrade()
        {
            l.wake();
        }
    }

    /// Whether the copies staged since the last batch submit now: they hold staging bytes, and
    /// unless `any` those bytes reach a batch. Discards alone free no room, so they wait for
    /// copies or the mount's end.
    pub(crate) fn batch_due(&self, any: bool) -> bool {
        let bytes = self.staging.lock().unwrap().ticket.bytes();
        bytes > 0 && (any || bytes >= self.batch)
    }
}

/// `bytes` into `block` at `at`: one sequential write into mapped memory, or into staging
/// and the copy that moves it.
fn write(device: &Device, block: Block, at: u64, bytes: &[u8]) -> Result<Option<StagedCopy>> {
    if bytes.is_empty() || block.write_mapped(at, bytes) {
        return Ok(None);
    }
    Ok(Some(StagedCopy::Buffer {
        src: staging(device, bytes)?,
        dst: block.gpu() + at,
    }))
}

fn staging(device: &Device, bytes: &[u8]) -> Result<Allocation<u8>> {
    let mut src =
        device.gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, "loading staging")?;
    src.as_mut_slice()
        .expect("upload is mapped")
        .copy_from_slice(bytes);
    Ok(src)
}

thread_local! {
    pub(crate) static PAGE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A mount's read from its file, waiting on the loading thread for room in the in-flight
/// limit.
pub(crate) struct MountRead {
    task: Arc<MountTask>,
    read: Box<dyn FnOnce(Option<Ticket>) + Send>,
}

impl MountRead {
    /// Starts the read, its `bytes` in flight until its placement has run; a read of a mount
    /// canceled or failed meanwhile ends unread.
    pub(crate) fn start(self, loading: &Arc<Loading>, bytes: u64) {
        let ticket = Ticket::new(loading, bytes);
        if self.task.live() {
            (self.read)(Some(ticket));
        } else {
            drop((self.read, ticket));
            self.task.end();
        }
    }
}

/// Reads and decodes blob `blob` of entry `entry`, then runs `then` on a worker with its
/// decoded bytes. A read from the file, or one whose placement `stages` copies, waits for room
/// in the in-flight limit, which its stored bytes read from the file and, when it stages, its
/// decoded bytes take until `then` has run; the decoded bytes stand in for those it stages,
/// which then take their own until their batch completes.
fn fetch(
    task: &Arc<MountTask>,
    entry: usize,
    blob: usize,
    stages: bool,
    then: impl FnOnce(&Arc<MountTask>, &[u8]) -> Result<()> + Send + 'static,
) {
    let archive = task.archive();
    let e = &archive.entries()[entry];
    let Some(&b) = archive.blobs(e).get(blob) else {
        task.fail(orr_pack::Error::Malformed("blob index outside its entry").into());
        return;
    };
    if b.codec() == Codec::GDeflate {
        task.fail(orr_pack::Error::GDeflate.into());
        return;
    }
    task.begin();
    let t = Arc::clone(task);
    let source = task.source();
    let read = move |ticket: Option<Ticket>| {
        read_stored(
            &source,
            b,
            None,
            Queue::Mount,
            Box::new(move |stored| {
                t.step(|| decoded(t.archive(), &b, None, stored?, |bytes| then(&t, bytes)));
                drop(ticket);
            }),
        );
    };
    let in_place = archive.stored(&b, None).is_some();
    if in_place && !stages {
        read(None);
    } else if let Some(loading) = task.loading.upgrade() {
        let stored = if in_place {
            0
        } else {
            u64::from(archive.stored_range(&b, None).len)
        };
        let bytes = stored + if stages { u64::from(b.raw_len) } else { 0 };
        let read = MountRead {
            task: Arc::clone(task),
            read: Box::new(read),
        };
        loading.send(Msg::MountRead(bytes, read));
    }
}

/// Starts a mount on the loading thread: the first reads of a file archive, or the parse of
/// a memory archive.
pub(crate) fn start(loading: &Arc<Loading>, ring: bool, task: Arc<MountTask>, source: PackSource) {
    task.ring.set(ring).expect("started once");
    match source {
        PackSource::Memory(bytes) => task.spawn(move |t| {
            t.step(|| {
                let archive = Archive::from_memory(bytes)?;
                plan(t, archive)
            });
        }),
        PackSource::File(path) => {
            let file = match ArchiveFile::open(&path) {
                Ok(f) => Arc::new(f),
                Err(e) => {
                    task.fail(e.into());
                    task.end();
                    return;
                }
            };
            let _ = task.file.set(Arc::clone(&file));
            if ring {
                read_head(loading, task, file, path);
            } else {
                let buffers = loading.buffers().clone();
                task.spawn(move |t| {
                    t.step(|| {
                        let mut head = buffers.take(Archive::HEAD_LEN);
                        file.reads.read(0, &mut head)?;
                        let range = Archive::index_range(head.as_slice())?;
                        drop(head);
                        let mut index = buffers.take(range.len as usize);
                        file.reads.read(range.offset, &mut index)?;
                        open_index(t, &file, &path, &index, range.len)
                    });
                });
            }
        }
    }
}

/// The head read, then the index read, on the ring; the parse runs as a job.
fn read_head(loading: &Arc<Loading>, task: Arc<MountTask>, file: Arc<ArchiveFile>, path: PathBuf) {
    let l = Arc::downgrade(loading);
    let f = Arc::clone(&file);
    let head = Read {
        file: Arc::clone(&file.reads),
        offset: 0,
        len: Archive::HEAD_LEN,
        into: None,
        done: Box::new(move |r| {
            let range = r
                .map_err(Error::from)
                .and_then(|buf| Ok(Archive::index_range(buf.as_slice())?));
            let range = match (range, l.upgrade()) {
                (Ok(range), Some(l)) => (range, l),
                (Err(e), _) => {
                    task.fail(e);
                    task.end();
                    return;
                }
                (Ok(_), None) => return,
            };
            let (range, l) = range;
            let index = Read {
                file: Arc::clone(&f.reads),
                offset: range.offset,
                len: (range.len as usize).next_multiple_of(SECTOR),
                into: None,
                done: Box::new(move |r| match r {
                    Ok(buf) => task.spawn(move |t| {
                        t.step(|| open_index(t, &f, &path, &buf, range.len));
                    }),
                    Err(e) => {
                        task.fail(e.into());
                        task.end();
                    }
                }),
            };
            l.send(Msg::Read(index));
        }),
    };
    loading.send(Msg::Read(head));
}

fn open_index(
    t: &Arc<MountTask>,
    file: &ArchiveFile,
    path: &std::path::Path,
    buf: &AlignedBuf,
    len: u32,
) -> Result<()> {
    let len = len as usize;
    if buf.len() < len {
        return Err(orr_pack::Error::Truncated {
            need: len as u64,
            have: buf.len() as u64,
        }
        .into());
    }
    let handle = file.archive.lock().unwrap().take().expect("archive handle");
    let index = buf.as_slice()[..len].to_vec().into_boxed_slice();
    let archive = Archive::from_index_file(handle, path, index)?;
    plan(t, archive)
}

/// Entry `e`'s blob 0 when it is resident in the mount section.
fn mount_blob0(archive: &Archive, e: &orr_pack::Entry) -> Result<Option<Vec<u8>>> {
    let Some(&b) = archive.blobs(e).first() else {
        return Ok(None);
    };
    let Some(stored) = archive.stored(&b, None) else {
        return Ok(None);
    };
    let mut bytes = vec![0; b.raw_len as usize];
    archive.decode(&b, None, stored, &mut bytes)?;
    Ok(Some(bytes))
}

/// The textures the pack's materials bind, the tiles of their tile sets among them.
fn material_textures(archive: &Archive) -> Result<HashSet<AssetId>> {
    let mut bound = HashSet::new();
    for e in archive
        .entries()
        .iter()
        .filter(|e| e.class == Class::Material as u16)
    {
        if let Some(bytes) = mount_blob0(archive, e)? {
            let m = Material::from_bytes(&bytes)?;
            bound.extend(m.textures.iter().map(|t| t.texture));
        }
    }
    for e in archive
        .entries()
        .iter()
        .filter(|e| e.class == Class::TextureTiles as u16)
    {
        if bound.contains(&e.id)
            && let Some(bytes) = mount_blob0(archive, e)?
        {
            let set = TextureTiles::parse(&bytes)?;
            bound.extend(set.tiles.iter().map(|t| t.texture));
        }
    }
    Ok(bound)
}

/// A neural texture set the mount decodes.
struct DecodedSet {
    entry: usize,
    id: AssetId,
    info: NeuralInfo,
    /// The blob of the weights it reads, and their constants.
    blob: u32,
    constants: Constants,
}

/// The pack's neural texture sets: those the mount decodes, none without `decode`, and the
/// latents of the others, which nothing reads.
fn neural_sets(archive: &Archive, decode: bool) -> Result<(Vec<DecodedSet>, HashSet<AssetId>)> {
    let mut sets = Vec::new();
    let mut unread = HashSet::new();
    for (entry, e) in archive.entries().iter().enumerate() {
        if e.class != Class::NeuralTextureSet as u16 {
            continue;
        }
        let e = archive.expect(e.id, e.class, orr_pack::neural::SCHEMA)?;
        let Some(bytes) = mount_blob0(archive, e)? else {
            continue;
        };
        let info = NeuralInfo::parse(&bytes)?;
        match neural::readable(&info).filter(|_| decode) {
            Some((row, constants)) => sets.push(DecodedSet {
                entry,
                id: e.id,
                blob: row.blob,
                constants,
                info,
            }),
            None => {
                unread.insert(info.header.latents);
            }
        }
    }
    Ok((sets, unread))
}

/// Queues the reads of every asset the mount makes resident. With neural decoding, the
/// textures in a readable set's table get a link to the set, which kernels that decode
/// read in place of the texture, and the set's latents stream like the textures materials
/// bind. Latents of sets the mount does not decode stay out of it.
fn plan(t: &Arc<MountTask>, archive: Archive) -> Result<()> {
    let archive = Arc::new(archive);
    let _ = t.archive.set(Arc::clone(&archive));
    let mut bound = material_textures(&archive)?;
    let (sets, unread_latents) = neural_sets(&archive, t.neural.is_some())?;
    let mut linked: HashMap<AssetId, (AssetId, NeuralTexture)> = HashMap::new();
    for s in &sets {
        bound.insert(s.info.header.latents);
        for tex in &s.info.textures {
            linked.entry(tex.texture).or_insert((s.id, *tex));
        }
    }
    for (i, e) in archive.entries().iter().enumerate() {
        let Some(class) = Class::from_u16(e.class) else {
            continue;
        };
        if !matches!(
            class,
            Class::Mesh
                | Class::Texture
                | Class::Material
                | Class::Model
                | Class::InstancePage
                | Class::Skeleton
                | Class::Clip
                | Class::Subd
                | Class::Volume
                | Class::TextureTiles
        ) {
            continue;
        }
        let schema = class.schema().expect("built-in class format");
        let e = archive.expect(e.id, e.class, schema)?;
        let id = e.id;
        let staged = t.staged;
        match class {
            Class::Mesh => fetch(t, i, 0, staged, move |t, b| place_mesh(t, i, id, b)),
            Class::Texture => {
                if unread_latents.contains(&id) {
                    continue;
                }
                let stream = bound.contains(&id);
                let neural = linked.get(&id).copied();
                fetch(t, i, 0, staged, move |t, b| {
                    place_texture(t, i, id, b, stream, neural)
                });
            }
            Class::Material => fetch(t, i, 0, false, move |t, b| {
                let m = Material::from_bytes(b)?;
                t.placed.lock().unwrap().materials.push((id, m));
                Ok(())
            }),
            Class::Model => fetch(t, i, 0, false, move |t, b| {
                let m = ModelInfo::parse(b)?;
                t.placed.lock().unwrap().models.push((id, m));
                Ok(())
            }),
            Class::Skeleton => fetch(t, i, 0, staged, move |t, b| place_skeleton(t, id, b)),
            Class::Clip => fetch(t, i, 0, staged, move |t, b| place_clip(t, i, id, b)),
            Class::Subd => fetch(t, i, 0, false, move |t, b| place_subd(t, i, id, b)),
            Class::Volume => fetch(t, i, 0, staged, move |t, b| place_volume(t, i, id, b)),
            Class::TextureTiles => fetch(t, i, 0, false, move |t, b| place_tiles(t, id, b)),
            _ => fetch(t, i, 0, false, move |t, b| place_page(t, i, id, b)),
        }
    }
    for s in sets {
        fetch(t, s.entry, s.blob as usize, t.staged, move |t, b| {
            place_neural(t, &s, b)
        });
    }
    Ok(())
}

/// A neural texture set's decoder weights, converted into the device's layout in a block of
/// their own that also holds the set's record.
fn place_neural(t: &Arc<MountTask>, s: &DecodedSet, blob: &[u8]) -> Result<()> {
    let p = t.neural.expect("sets are read with neural decoding");
    let net = Fp8Network::parse(blob)?;
    let mut w = neural::device_weights(&t.device, &s.info, &s.constants, &net, p)?;
    let record_at = w.bytes().len() as u64;
    let record_bytes = size_of::<NeuralSetGpu>() as u64;
    let block = t.block(record_at + record_bytes, "neural texture set")?;
    assert!(
        block.gpu().is_multiple_of(orr_gpu::MATRIX_ALIGN),
        "neural weights at {:#x}",
        block.gpu()
    );
    t.write(block, 0, w.bytes())?;
    w.record.weights = block.gpu();
    w.record.sampler = t.sampler;
    t.placed.lock().unwrap().neural.push(PlacedNeural {
        id: s.id,
        latents: s.info.header.latents,
        // SAFETY: the record's bytes in the set's block.
        record: unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu() + record_at)) },
        gpu: w.record,
    });
    Ok(())
}

/// A volume's blob 0, then its majorants and every page, into one block laid out by
/// `VolumeInfo::resident_layout`.
fn place_volume(t: &Arc<MountTask>, entry: usize, id: AssetId, blob0: &[u8]) -> Result<()> {
    let info = VolumeInfo::parse(blob0)?;
    let layout = info.resident_layout();
    let block = t.block(layout.size, "volume")?;
    t.write(block, 0, &info.resident_head(&layout, block.gpu()))?;
    let r = VolumeRef {
        // SAFETY: the record is resident at the start of the mount's block.
        volume: unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu())) },
        bounds: info.header.bounds,
        grids: info.names.clone().into(),
    };
    t.placed.lock().unwrap().volumes.push((id, r));
    let cells = info.header.majorant_cells.iter().product::<u32>() as usize;
    let majorant_bytes = 4 * cells * info.grids.len();
    let at = layout.majorants;
    fetch(t, entry, 1, t.staged, move |t, b| {
        if b.len() != majorant_bytes {
            return Err(orr_pack::Error::Malformed("volume majorants").into());
        }
        t.write(block, at, b)
    });
    for (g, grid) in info.grids.iter().enumerate() {
        let per_page = grid.bricks_per_page();
        let count = grid.bricks.count as usize;
        for p in 0..grid.pages.count as usize {
            let bytes = (count - p * per_page).min(per_page) * grid.brick_bytes();
            let at = layout.page(&info, g, p);
            fetch(
                t,
                entry,
                grid.pages.first as usize + p,
                t.staged,
                move |t, b| {
                    if b.len() != bytes {
                        return Err(orr_pack::Error::Malformed("volume page").into());
                    }
                    t.write(block, at, b)
                },
            );
        }
    }
    Ok(())
}

/// A subdivision cage's blob 0, then every page; the last page read builds the cage's
/// tables into a block of its own.
fn place_subd(t: &Arc<MountTask>, entry: usize, id: AssetId, blob0: &[u8]) -> Result<()> {
    struct Pending {
        info: SubdInfo,
        pages: Vec<Option<SubdPage>>,
        left: usize,
    }
    let info = SubdInfo::parse(blob0)?;
    let blobs: Vec<u32> = info.pages.iter().map(|p| p.blob).collect();
    let pending = Arc::new(Mutex::new(Pending {
        pages: vec![None; blobs.len()],
        left: blobs.len(),
        info,
    }));
    for (k, blob) in blobs.into_iter().enumerate() {
        let pending = Arc::clone(&pending);
        fetch(t, entry, blob as usize, t.staged, move |t, b| {
            let page = SubdPage::parse(b)?;
            let mut p = pending.lock().unwrap();
            p.pages[k] = Some(page);
            p.left -= 1;
            if p.left > 0 {
                return Ok(());
            }
            let pages: Vec<SubdPage> = p.pages.drain(..).flatten().collect();
            let tables = CageTables::new(&p.info, &pages)?;
            drop(p);
            let kind = if t.staged {
                MemoryKind::DeviceLocal
            } else {
                MemoryKind::Upload
            };
            let a = t.device.gpu_malloc::<u8>(kind, tables.len(), "subd cage")?;
            let base = a.gpu();
            let block = Block::of(&a);
            t.placed.lock().unwrap().blocks.push(a);
            t.write(block, 0, &tables.bytes(base))?;
            t.placed.lock().unwrap().cages.push(PlacedCage {
                id,
                cage: base.cast(),
                points: tables.rest_points(base),
                shape: tables.shape,
            });
            Ok(())
        });
    }
    Ok(())
}

/// A skeleton's blob into a block of its own.
fn place_skeleton(t: &Arc<MountTask>, id: AssetId, blob: &[u8]) -> Result<()> {
    let info = orr_pack::skeleton::SkeletonInfo::parse(blob)?;
    let block = t.block(blob.len() as u64, "skeleton")?;
    t.write(block, 0, blob)?;
    let r = SkeletonRef {
        // SAFETY: the blob is resident in the mount's block.
        blob: unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu())) },
        joint_count: info.joint_count() as u32,
    };
    t.placed.lock().unwrap().skeletons.push((id, r));
    Ok(())
}

/// A clip's header and samples (blobs 0 and 1) into one block.
fn place_clip(t: &Arc<MountTask>, entry: usize, id: AssetId, head: &[u8]) -> Result<()> {
    let h: orr_pack::clip::ClipHeader = bytemuck::pod_read_unaligned(
        head.get(..size_of::<orr_pack::clip::ClipHeader>())
            .ok_or(orr_pack::Error::Malformed("clip header"))?,
    );
    let samples = u64::from(h.frame_count)
        * u64::from(h.joint_count)
        * size_of::<orr_pack::skeleton::Trs>() as u64;
    let at = (head.len() as u64).next_multiple_of(256);
    let block = t.block(at + samples.max(16), "clip")?;
    t.write(block, 0, head)?;
    // SAFETY: the blobs are resident in the mount's block.
    let ptr = |o: u64| unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu() + o)) };
    let r = ClipRef {
        header: ptr(0),
        samples: ptr(at),
        skeleton: h.skeleton,
        joint_count: h.joint_count,
        frame_count: h.frame_count,
        sample_rate: h.sample_rate,
        duration: h.duration,
    };
    t.placed.lock().unwrap().clips.push((id, r));
    fetch(t, entry, 1, t.staged, move |t, bytes| {
        orr_pack::clip::Clip::parse(bytemuck::bytes_of(&h), bytes)?;
        t.write(block, at, bytes)
    });
    Ok(())
}

/// Blob 0, then every page and the top node page of a mesh, into one block. The other node
/// pages stream where the mesh pages do and join the block otherwise: a resident page's groups
/// are drawn, so their leaves are resident too.
fn place_mesh(t: &Arc<MountTask>, entry: usize, id: AssetId, blob0: &[u8]) -> Result<()> {
    let mut info = MeshInfo::parse(blob0)?;
    let header = info.header;
    let streams = streams_pages(&info, t.accel.limits.mesh_resident_max);
    let pinned = if streams {
        header.pinned_pages as usize
    } else {
        info.pages.len()
    };
    let node_bytes = |p: &NodePage| p.node_count * size_of::<orr_pack::mesh::Node>() as u32;
    let mut size = (blob0.len() as u64).next_multiple_of(256);
    let mut place = |bytes: u64| {
        let at = size;
        size += bytes.next_multiple_of(256);
        at
    };
    let page_clusters_at = place(4 * (info.pages.len() as u64 + 1));
    let groups: Arc<[Group]> = std::mem::take(&mut info.groups).into();
    let (fine, cached) = if info.levels.is_empty() {
        accel::cut_rows(&info.cut_rows, streams, t.traced)
    } else {
        Default::default()
    };
    let page_at: Vec<u64> = info.pages[..pinned]
        .iter()
        .map(|p| place(u64::from(p.resident_len)))
        .collect();
    let node_at: Vec<Option<u64>> = info
        .node_pages
        .iter()
        .map(|p| {
            (!streams || p.parent_page == NodePage::TOP).then(|| place(u64::from(node_bytes(p))))
        })
        .collect();
    let mut leaf_pages = vec![Vec::new(); info.pages.len()];
    for g in groups.iter().filter(|g| g.node_page != 0) {
        leaf_pages[g.page as usize].push(g.node_page);
    }
    let page_units = info
        .pages
        .iter()
        .zip(leaf_pages)
        .map(|(p, mut leaves)| {
            let deps = p.deps.first as usize..(p.deps.first + p.deps.count) as usize;
            let deps = info
                .page_deps
                .get(deps)
                .ok_or(orr_pack::Error::Malformed("page dependencies"))?;
            leaves.sort_unstable();
            leaves.dedup();
            Ok((u64::from(p.resident_len), deps.to_vec(), leaves))
        })
        .collect::<Result<Vec<_>>>()?;
    let node_units = info
        .node_pages
        .iter()
        .map(|p| (p.blob, node_bytes(p), p.parent_page))
        .collect();
    let block = t.block(size, "mesh")?;
    t.write(block, 0, blob0)?;
    let depth = info
        .node_pages
        .is_empty()
        .then(|| hierarchy_depth(&info.nodes, &info.roots));
    let mut first = 0;
    let firsts: Vec<u32> = info
        .pages
        .iter()
        .map(|p| {
            let f = first;
            first += u32::from(p.cluster_count);
            f
        })
        .collect();
    let mut page_clusters = firsts.clone();
    page_clusters.push(first);
    t.write(
        block,
        page_clusters_at,
        bytemuck::cast_slice(&page_clusters),
    )?;
    let all_pages: Vec<StreamedPage> = info
        .pages
        .iter()
        .zip(&firsts)
        .map(|(p, &first)| StreamedPage {
            blob: p.blob,
            bytes: p.resident_len,
            first,
            clusters: u32::from(p.cluster_count),
            triangles: p.tri_count,
        })
        .collect();
    let level_ranges = info
        .levels
        .iter()
        .map(|l| (l.error, l.clusters.first, l.clusters.count))
        .collect();
    let mesh = {
        let mut placed = t.placed.lock().unwrap();
        placed.meshes.push(PlacedMesh {
            id,
            entry: entry as u32,
            blob0: block.gpu(),
            pages: vec![0; info.pages.len()],
            node_pages: vec![0; info.node_pages.len()],
            page_units,
            node_units,
            page_clusters: block.gpu() + page_clusters_at,
            cluster_count: first,
            sections: info.sections.len() as u32,
            header,
            ranks: [info.emissive.rank_count(), info.casters.rank_count()],
            cached_only: [&info.emissive, &info.casters].map(|t| t.widest > GATHERED_CLUSTERS),
            level_ranges,
            clusters: Vec::new(),
            micromaps: vec![None; info.pages.len()],
            parts: info.parts.clone(),
            depth,
            streamed: streams.then(|| all_pages.clone()),
            cached,
            groups: Arc::clone(&groups),
            all_pages,
            fine,
        });
        placed.meshes.len() - 1
    };
    for (p, page) in info.pages[..pinned].iter().enumerate() {
        let (at, first, len) = (page_at[p], firsts[p], page.resident_len as usize);
        let groups = Arc::clone(&groups);
        fetch(t, entry, page.blob as usize, t.staged, move |t, stored| {
            PAGE.with_borrow_mut(|resident| {
                resident.resize(len, 0);
                orr_pack::mesh::decode_page(stored, resident)?;
                t.write(block, at, resident)?;
                let view = PageView::new(resident, &header)?;
                let h: PageHeader =
                    bytemuck::pod_read_unaligned(&resident[..size_of::<PageHeader>()]);
                let records: Vec<Cluster> =
                    (0..view.cluster_count()).map(|i| view.cluster(i)).collect();
                let (infos, micromap) = page_infos(&h, &records, p as u32, &groups)?;
                let clusters: Vec<(u32, ClusterInfo)> = (first..).zip(infos).collect();
                let mut placed = t.placed.lock().unwrap();
                let m = &mut placed.meshes[mesh];
                m.pages[p] = block.gpu() + at;
                m.clusters.extend(clusters);
                m.micromaps[p] = micromap;
                Ok(())
            })
        });
    }
    for (n, node_page) in info.node_pages.iter().enumerate() {
        let Some(at) = node_at[n] else { continue };
        let len = node_page.node_count as usize * size_of::<orr_pack::mesh::Node>();
        fetch(
            t,
            entry,
            node_page.blob as usize,
            t.staged,
            move |t, nodes| {
                if nodes.len() != len {
                    return Err(orr_pack::Error::Malformed("node page length").into());
                }
                t.write(block, at, nodes)?;
                t.placed.lock().unwrap().meshes[mesh].node_pages[n] = block.gpu() + at;
                Ok(())
            },
        );
    }
    Ok(())
}

/// Whether a mesh's pages past its pinned ones stream: a DAG mesh whose pages take more than
/// `resident_max` bytes, unless it is skinned, which reads cluster by cluster from mount.
fn streams_pages(info: &MeshInfo, resident_max: u64) -> bool {
    let h = &info.header;
    let bytes: u64 = info.pages.iter().map(|p| u64::from(p.resident_len)).sum();
    info.levels.is_empty()
        && h.attributes & MeshHeader::SKIN == 0
        && (h.pinned_pages as usize) < info.pages.len()
        && bytes > resident_max
}

/// A page's clusters as the loading side keeps them, and its micromap region, from its
/// header and cluster records. Each cluster of a group at depth 0 of the mesh's `groups` is a
/// source cluster and every other one names a group, which the traversal reads.
pub(crate) fn page_infos(
    h: &PageHeader,
    clusters: &[Cluster],
    page: u32,
    groups: &[Group],
) -> Result<(Vec<ClusterInfo>, Option<PageMicromap>)> {
    let first = groups.partition_point(|g| g.page < page);
    for g in groups[first..].iter().take_while(|g| g.page == page) {
        let at = usize::from(g.first_cluster)..;
        let Some(members) = clusters
            .get(at)
            .and_then(|c| c.get(..usize::from(g.cluster_count)))
        else {
            return Err(orr_pack::Error::Malformed("group outside its page").into());
        };
        let named = |c: &Cluster| match g.depth {
            0 => c.from_group == Cluster::SOURCE,
            _ => (c.from_group as usize) < groups.len(),
        };
        if !members.iter().all(named) {
            return Err(orr_pack::Error::Malformed("cluster group").into());
        }
    }
    let clusters = (0..)
        .zip(clusters)
        .map(|(i, c)| ClusterInfo::new(page, i, c))
        .collect();
    let micromap =
        (h.omm_offset < h.len && h.omm_usage.iter().any(|&n| n > 0)).then_some(PageMicromap {
            offset: h.omm_offset,
            usage: h.omm_usage,
        });
    Ok((clusters, micromap))
}

fn gpu_format(vk_format: u32) -> Option<orr_gpu::Format> {
    use orr_gpu::Format as F;
    use orr_pack::texture::vk;
    Some(match vk_format {
        vk::R8_UNORM => F::R8Unorm,
        vk::R8G8_UNORM => F::Rg8Unorm,
        vk::R8G8B8A8_UNORM => F::Rgba8Unorm,
        vk::R8G8B8A8_SRGB => F::Rgba8Srgb,
        vk::A4R4G4B4_UNORM_PACK16 => F::A4r4g4b4Unorm,
        vk::R16_UNORM => F::R16Unorm,
        vk::R16_SNORM => F::R16Snorm,
        vk::R16_SFLOAT => F::R16Float,
        vk::R16G16_UNORM => F::Rg16Unorm,
        vk::R16G16_SFLOAT => F::Rg16Float,
        vk::R16G16B16A16_UNORM => F::Rgba16Unorm,
        vk::R16G16B16A16_SFLOAT => F::Rgba16Float,
        vk::R32_SFLOAT => F::R32Float,
        vk::R32G32_SFLOAT => F::Rg32Float,
        vk::R32G32B32A32_SFLOAT => F::Rgba32Float,
        vk::BC1_RGBA_UNORM => F::Bc1Unorm,
        vk::BC1_RGBA_SRGB => F::Bc1Srgb,
        vk::BC3_UNORM => F::Bc3Unorm,
        vk::BC3_SRGB => F::Bc3Srgb,
        vk::BC4_UNORM => F::Bc4Unorm,
        vk::BC4_SNORM => F::Bc4Snorm,
        vk::BC5_UNORM => F::Bc5Unorm,
        vk::BC5_SNORM => F::Bc5Snorm,
        vk::BC6H_UFLOAT => F::Bc6hUfloat,
        vk::BC6H_SFLOAT => F::Bc6hSfloat,
        vk::BC7_UNORM => F::Bc7Unorm,
        vk::BC7_SRGB => F::Bc7Srgb,
        _ => return None,
    })
}

/// A texture from blob 0: streamed when `stream` and it has mips above a mip tail, which
/// stays resident as the floor the residency clamp falls back to; else whole with every mip
/// copied in from its blob. Textures streamed are those the pack's materials bind and neural
/// sets' latents, which kernels sample through the footprint function's residency clamp. With
/// `neural`, the texture links to the set that stands in for it.
fn place_texture(
    t: &Arc<MountTask>,
    entry: usize,
    id: AssetId,
    blob0: &[u8],
    stream: bool,
    neural: Option<(AssetId, NeuralTexture)>,
) -> Result<()> {
    let info = Arc::new(TextureInfo::parse(blob0)?);
    let h = info.header;
    let format = gpu_format(h.vk_format).ok_or(Error::TextureFormat(h.vk_format))?;
    let mut desc = TextureDesc {
        kind: if h.kind == TextureHeader::CUBE {
            TextureKind::Cube
        } else {
            TextureKind::D2
        },
        format,
        extent: [h.width, h.height, 1],
        mips: u32::from(h.mip_count),
        layers: u32::from(h.layers),
        usage: TextureUsage::SAMPLED,
    };
    let mut neural = neural.map(|n| neural_link(t, n)).transpose()?;
    let has_tail = (1..h.mip_count).contains(&h.tail_first_mip);
    if stream && has_tail && place_streamed(t, entry, id, &info, &desc, &mut neural)? {
        return Ok(());
    }
    if t.host_copy {
        desc.usage |= TextureUsage::HOST_COPY;
    }
    let texture = Arc::new(t.device.create_texture(
        &desc,
        TextureMemory::Dedicated,
        "cooked texture",
    )?);
    let slot = ImageSlot::sampling(&t.device, &texture)?;
    t.placed.lock().unwrap().textures.push(PlacedTexture {
        id,
        texture: Arc::clone(&texture),
        slot,
        primaries: h.primaries,
        bytes: t.device.texture_requirements(&desc).0,
        stream: None,
        neural,
    });
    if !t.host_copy {
        t.stage(StagedCopy::Discard {
            texture: Arc::clone(&texture),
            stream: None,
        });
    }
    for blob in 1..h.blob_count() {
        let (info, texture) = (Arc::clone(&info), Arc::clone(&texture));
        fetch(t, entry, blob, !t.host_copy, move |t, bytes| {
            let h = &info.header;
            for mip in (0..u32::from(h.mip_count)).filter(|&m| h.mip_blob(m) == blob) {
                let [w, hh] = h.mip_size(mip);
                for layer in 0..u32::from(h.layers) {
                    let texels = info.mip_texels(bytes, mip, layer)?;
                    let region = TextureRegion {
                        mip,
                        base_layer: layer,
                        layers: 1,
                        offset: [0; 3],
                        extent: [w, hh, 1],
                    };
                    t.write_texture(&texture, &region, &texels)?;
                }
            }
            Ok(())
        });
    }
    Ok(())
}

/// A tile set's table, filled once its tiles have heap indices.
fn place_tiles(t: &Arc<MountTask>, id: AssetId, blob: &[u8]) -> Result<()> {
    let set = TextureTiles::parse(blob)?;
    let block = t.block(4 * set.words() as u64, "texture tiles")?;
    // SAFETY: the block holds the table.
    let table = unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu())) };
    let tiles = PlacedTiles { id, set, table };
    t.placed.lock().unwrap().tiles.push(tiles);
    Ok(())
}

/// A neural link of its own for a texture that set `set` stands in for, its record (and the
/// texture's own stream record, should it not stream) in a block of their own.
fn neural_link(t: &Arc<MountTask>, (set, tex): (AssetId, NeuralTexture)) -> Result<NeuralLink> {
    let record_bytes = size_of::<NeuralTextureGpu>() as u64;
    let block = t.block(
        record_bytes + size_of::<TextureStreamGpu>() as u64,
        "neural texture record",
    )?;
    // SAFETY: both records are the block's bytes.
    let (record, own) = unsafe {
        (
            GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu())),
            GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu() + record_bytes)),
        )
    };
    Ok(NeuralLink {
        set,
        record,
        own,
        gpu: NeuralTextureGpu {
            set: GpuPtr::NULL,
            first: u32::from(tex.first_channel),
            count: u32::from(tex.channel_count.min(4)),
        },
        latents: None,
        latent_mips: Vec::new(),
    })
}

/// A streamed texture: a sparse image whose mip tails are bound at mount and filled from
/// the last blob, and whose tiles above them stream; its stream record points at `neural`,
/// which it takes. False, placing nothing, when the device's tile layout differs from the
/// cooked one.
fn place_streamed(
    t: &Arc<MountTask>,
    entry: usize,
    id: AssetId,
    info: &Arc<TextureInfo>,
    desc: &TextureDesc,
    neural: &mut Option<NeuralLink>,
) -> Result<bool> {
    let h = info.header;
    let texture = t
        .device
        .create_texture(desc, TextureMemory::Sparse, "streamed texture")?;
    let layout = *texture.sparse_layout().expect("a sparse texture");
    let [tw, th] = orr_pack::texture::tile_shape(h.format()?);
    if layout.tail_first_mip != u32::from(h.tail_first_mip)
        || layout.tile_bytes != orr_pack::texture::TILE_BYTES as u64
        || layout.tile_extent[..2] != [tw, th]
    {
        return Ok(false);
    }
    let texture = Arc::new(texture);
    let neural = neural.take();
    let tail = u32::from(h.tail_first_mip);
    let grid = TileGrid::new(
        (0..tail).map(|m| layout.tiles(desc, m)).collect(),
        desc.layers,
    );
    let tails = layout.tails(desc);
    let tail_bytes = u64::from(tails) * layout.tail_bytes;
    let [cx, cy] = grid.cells();
    let cells = cx * cy * desc.layers;
    let record_bytes = size_of::<TextureStreamGpu>() as u64;
    let block = t.block(record_bytes + 4 * u64::from(cells), "texture stream")?;
    let map = vec![tail; cells as usize];
    t.write(block, record_bytes, bytemuck::cast_slice(&map))?;
    // SAFETY: the record and the map are the block's bytes.
    let (record, map) = unsafe {
        (
            GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu())),
            GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu() + record_bytes)),
        )
    };
    let tails = match tail_bytes {
        0 => None,
        bytes => Some((
            tails,
            t.device.gpu_malloc_with_priority::<u8>(
                MemoryKind::DeviceLocal,
                MemoryPriority::High,
                bytes,
                "texture mip tails",
            )?,
        )),
    };
    let slot = ImageSlot::sampling(&t.device, &texture)?;
    let stream = Arc::new(TexStream {
        texture: Arc::clone(&texture),
        index: slot.index,
        layout,
        grid,
        entry: entry as u32,
        blobs: (0..tail).map(|m| h.mip_blob(m) as u32).collect(),
        record,
        map,
        tails,
        neural: neural.as_ref().map_or(GpuPtr::NULL, |n| n.record),
    });
    t.placed.lock().unwrap().textures.push(PlacedTexture {
        id,
        texture: Arc::clone(&texture),
        slot,
        primaries: h.primaries,
        bytes: tail_bytes,
        stream: Some(Arc::clone(&stream)),
        neural: neural.map(|n| NeuralLink {
            own: GpuPtr::NULL,
            ..n
        }),
    });
    t.stage(StagedCopy::Discard {
        texture: Arc::clone(&texture),
        stream: Some(stream),
    });
    if tail < desc.mips {
        let info = Arc::clone(info);
        fetch(t, entry, h.mip_blob(tail), true, move |t, bytes| {
            let h = &info.header;
            for mip in tail..u32::from(h.mip_count) {
                let [w, hh] = h.mip_size(mip);
                for layer in 0..u32::from(h.layers) {
                    let texels = info.mip_texels(bytes, mip, layer)?;
                    let region = TextureRegion {
                        mip,
                        base_layer: layer,
                        layers: 1,
                        offset: [0; 3],
                        extent: [w, hh, 1],
                    };
                    t.stage_texture(&texture, &region, &texels)?;
                }
            }
            Ok(())
        });
    }
    Ok(true)
}

/// An instance page: blob 0 stays on the CPU; its records stream.
fn place_page(t: &Arc<MountTask>, entry: usize, id: AssetId, blob0: &[u8]) -> Result<()> {
    let info = InstancePageInfo::parse(blob0)?;
    let a = t.archive();
    let blobs = a.blobs(&a.entries()[entry]);
    let bytes = blobs.get(1).map_or(0, |b| b.raw_len);
    let h = &info.header;
    h.exact_rows(bytes as usize)?;
    for w in 0..h.motion_windows() {
        let held = blobs.get(h.motion_blob(w) as usize).map(|b| b.raw_len);
        if held != Some(h.motion_window_bytes(w)) {
            return Err(orr_pack::Error::Malformed("motion rows").into());
        }
    }
    t.placed.lock().unwrap().pages.push(PlacedPage {
        id,
        info,
        entry: entry as u32,
        bytes,
    });
    Ok(())
}

/// A finished mount, handed to the render thread once its loading-queue work has landed.
pub(crate) struct Batch {
    mount: MountId,
    /// The point its staged copies signal; none when nothing was staged.
    pub(crate) point: Option<TimelinePoint>,
    result: Result<Resolved, Arc<Error>>,
}

impl Batch {
    pub(crate) fn landed(&self, device: &Device) -> bool {
        crate::loading::landed(device, self.point)
    }
}

struct ResolvedMesh {
    id: AssetId,
    entry: u32,
    page_units: Vec<(u64, Vec<u32>, Vec<u32>)>,
    node_units: Vec<(u32, u32, u32)>,
    gpu: MeshGpu,
    page_addresses: Vec<u64>,
    /// A skinned mesh's skin table.
    skin_record: Option<SkinMeshGpu>,
    sections: u32,
    header: MeshHeader,
    levels: Vec<(f32, u32, u32)>,
    /// The clusters of its pinned pages, which come first, by mesh cluster index.
    clusters: Vec<ClusterInfo>,
    micromaps: Vec<Option<PageMicromap>>,
    parts: Vec<Part>,
    /// The cluster id of mesh cluster 0: its pinned pages' clusters take the ids that follow.
    cluster_base: u32,
    /// Resident cluster table entries of its pinned pages' clusters, from `cluster_base` on.
    refs: Vec<ClusterRef>,
    rt: MeshRt,
    depth: Option<u32>,
    streamed: Option<Vec<StreamedPage>>,
    cached: Vec<CutRow>,
    non_opaque: Vec<bool>,
    groups: Arc<[Group]>,
    all_pages: Vec<StreamedPage>,
    fine: Vec<CutRow>,
    animated: accel::Animated,
}

struct ResolvedModel {
    id: AssetId,
    mesh: Option<usize>,
    cage: Option<usize>,
    /// Its skeleton's joints; 0 without a skeleton.
    joints: u32,
    /// Indices into `Resolved::materials`, one per section.
    materials: Vec<usize>,
    bounds: Aabb,
}

/// A mount's assets with every reference between them resolved inside the pack.
struct Resolved {
    source: Arc<BlobSource>,
    blocks: Vec<Allocation<u8>>,
    textures: Vec<PlacedTexture>,
    materials: Vec<ResolvedMaterial>,
    meshes: Vec<ResolvedMesh>,
    models: Vec<ResolvedModel>,
    pages: Vec<PlacedPage>,
    skeletons: Vec<(AssetId, SkeletonRef)>,
    clips: Vec<(AssetId, ClipRef)>,
    cages: Vec<PlacedCage>,
    volumes: Vec<(AssetId, VolumeRef)>,
    neural: Vec<PlacedNeural>,
    /// Tile tables and what they hold.
    tiles: Vec<(GpuPtr<u32>, Vec<u32>)>,
    accel: MountAccel,
    /// Resident cluster id ranges taken for the mount's meshes.
    cluster_ranges: Vec<(u32, u32)>,
}

fn resolve(source: BlobSource, p: Placed, device: &Device) -> Result<Resolved> {
    let missing = |id| Error::Pack(orr_pack::Error::Missing(id));
    let mut textures: HashMap<AssetId, BoundTexture> = p
        .textures
        .iter()
        .map(|t| {
            let b = BoundTexture {
                index: t.slot.index,
                primaries: t.primaries,
                ..BoundTexture::NONE
            };
            (t.id, b)
        })
        .collect();
    let mut tiles = Vec::with_capacity(p.tiles.len());
    let mut sets = Vec::with_capacity(p.tiles.len());
    for s in &p.tiles {
        let mut shared: Option<BoundTexture> = None;
        let mut mixed = false;
        let words = s
            .set
            .table(|id| {
                let b = textures.get(&id)?;
                mixed |= b.primaries != shared.get_or_insert(*b).primaries;
                Some(b.index.0)
            })
            .map_err(missing)?;
        if mixed {
            return Err(orr_pack::Error::Malformed("texture tiles of mixed primaries").into());
        }
        let bound = BoundTexture {
            tiles: s.table,
            ..shared.expect("a tile set has tiles")
        };
        sets.push((s.id, bound));
        tiles.push((s.table, words));
    }
    textures.extend(sets);
    let materials = p
        .materials
        .iter()
        .map(|(id, m)| {
            let mut bound = [BoundTexture::NONE; slot::COUNT];
            for (b, t) in bound.iter_mut().zip(&m.textures) {
                if t.texture != AssetId::NONE {
                    *b = BoundTexture {
                        sampler: device.sampler(&binding_sampler(t))?,
                        ..*textures.get(&t.texture).ok_or_else(|| missing(t.texture))?
                    };
                }
            }
            Ok((*id, MaterialGpu::new(m, &bound), m.back))
        })
        .collect::<Result<Vec<_>>>()?;
    let material_index: HashMap<AssetId, usize> = materials
        .iter()
        .enumerate()
        .map(|(i, (id, ..))| (*id, i))
        .collect();
    let materials: Vec<ResolvedMaterial> = materials
        .into_iter()
        .map(|(id, m, back)| {
            let back = (back != AssetId::NONE)
                .then(|| {
                    material_index
                        .get(&back)
                        .copied()
                        .ok_or_else(|| missing(back))
                })
                .transpose()?;
            Ok((id, m, back))
        })
        .collect::<Result<_>>()?;
    let meshes: Vec<ResolvedMesh> = p
        .meshes
        .into_iter()
        .map(|mut m| {
            m.clusters.sort_by_key(|(i, _)| *i);
            let clusters: Vec<ClusterInfo> = m.clusters.into_iter().map(|(_, c)| c).collect();
            let mut page_addresses = m.pages;
            let pages = page_addresses.len() as u32;
            let node_pages = m.node_pages.len() as u32;
            page_addresses.extend(m.node_pages);
            let table = |span: orr_pack::Span| {
                if span.count == 0 {
                    return GpuPtr::NULL;
                }
                // SAFETY: a table of blob 0, resident in the mount's block.
                unsafe {
                    GpuPtr::from_address(orr_gpu::GpuAddress(m.blob0 + u64::from(span.first)))
                }
            };
            let sets = |s: orr_pack::mesh::SetSpans, count: u32, cached: bool| SetTablesGpu {
                sets: table(s.sets).cast(),
                ranks: table(s.ranks).cast(),
                clusters: table(s.clusters).cast(),
                cdf: table(s.cdf).cast(),
                index: table(s.index).cast(),
                nodes: table(s.nodes).cast(),
                count,
                cached: u32::from(cached),
            };
            ResolvedMesh {
                id: m.id,
                entry: m.entry,
                page_units: m.page_units,
                node_units: m.node_units,
                // SAFETY: blob 0 of the mesh and its page cluster table, resident in the
                // mount's block.
                gpu: unsafe {
                    MeshGpu {
                        pages: GpuPtr::NULL,
                        node_pages: GpuPtr::NULL,
                        page_clusters: GpuPtr::from_address(orr_gpu::GpuAddress(m.page_clusters)),
                        page_bases: GpuPtr::NULL,
                        levels: table(m.header.levels).cast(),
                        page_count: pages,
                        node_page_count: node_pages,
                        cluster_count: m.cluster_count,
                        level_count: m.header.levels.count,
                        position_step: m.header.step(),
                        attributes: u32::from(m.header.attributes),
                        merge_error: m.header.merge_error,
                        _pad: 0,
                        rt_cut_count: 0,
                        part_count: m.header.parts.count.max(1),
                        rt_cuts: GpuPtr::NULL,
                        parts: GpuPtr::NULL,
                        masks: GpuPtr::NULL,
                        first_animated: 0,
                        animated_parts: 0,
                        mask_count: 0,
                        animated_sections: 0,
                        nodes: table(m.header.nodes),
                        node_table: table(m.header.node_pages),
                        groups: table(m.header.groups),
                        roots: table(m.header.roots).cast(),
                        root_count: m.header.roots.count,
                        pinned_pages: if m.streamed.is_some() {
                            m.header.pinned_pages
                        } else {
                            pages
                        },
                        skin: GpuPtr::NULL,
                        edit: GpuPtr::NULL,
                        feedback_pages: 0,
                        feedback_nodes: 0,
                        fine_cuts: 0,
                        section_count: m.sections,
                        opaque_sections: m.header.opaque_sections,
                        layer_sections: m.header.layer_sections,
                        emissive: sets(m.header.emissive, m.ranks[0], m.cached_only[0]),
                        casters: sets(m.header.casters, m.ranks[1], m.cached_only[1]),
                    }
                },
                page_addresses,
                skin_record: None,
                sections: m.sections,
                header: m.header,
                levels: m.level_ranges,
                clusters,
                micromaps: m.micromaps,
                parts: m.parts,
                cluster_base: 0,
                refs: Vec::new(),
                rt: MeshRt::default(),
                depth: m.depth,
                streamed: m.streamed,
                cached: m.cached,
                non_opaque: Vec::new(),
                groups: m.groups,
                all_pages: m.all_pages,
                fine: m.fine,
                animated: accel::Animated::default(),
            }
        })
        .collect();
    let mesh_index: HashMap<AssetId, usize> =
        meshes.iter().enumerate().map(|(i, m)| (m.id, i)).collect();
    let models = p
        .models
        .iter()
        .map(|(id, m)| {
            let mesh = match m.header.mesh {
                AssetId::NONE => None,
                k => Some(*mesh_index.get(&k).ok_or_else(|| missing(k))?),
            };
            let cage = match m.header.subd {
                AssetId::NONE => None,
                k => Some(
                    p.cages
                        .iter()
                        .position(|c| c.id == k)
                        .ok_or_else(|| missing(k))?,
                ),
            };
            let sections = mesh.map_or(0, |i| meshes[i].sections)
                + cage.map_or(0, |i| p.cages[i].shape.sections);
            if mesh.is_none() && cage.is_none() || m.materials.len() != sections as usize {
                return Err(orr_pack::Error::Malformed("model materials per section").into());
            }
            let materials = m
                .materials
                .iter()
                .map(|id| material_index.get(id).copied().ok_or_else(|| missing(*id)))
                .collect::<Result<Vec<_>>>()?;
            let joints = match m.header.skeleton {
                AssetId::NONE => 0,
                k => {
                    p.skeletons
                        .iter()
                        .find(|(id, _)| *id == k)
                        .ok_or_else(|| missing(k))?
                        .1
                        .joint_count
                }
            };
            Ok(ResolvedModel {
                id: *id,
                mesh,
                cage,
                joints,
                materials,
                bounds: m.header.bounds,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Resolved {
        source: Arc::new(source),
        blocks: p.blocks,
        textures: p.textures,
        materials,
        meshes,
        models,
        pages: p.pages,
        skeletons: p.skeletons,
        clips: p.clips,
        cages: p.cages,
        volumes: p.volumes,
        neural: p.neural,
        tiles,
        accel: MountAccel::default(),
        cluster_ranges: Vec::new(),
    })
}

/// Points each neural set at its latents and each linked texture at its set.
fn link_neural(r: &mut Resolved) -> Result<()> {
    let missing = |id| Error::Pack(orr_pack::Error::Missing(id));
    for n in &mut r.neural {
        let latents = r
            .textures
            .iter()
            .find(|t| t.id == n.latents)
            .ok_or_else(|| missing(n.latents))?;
        let d = latents.texture.desc();
        if d.format != orr_gpu::Format::A4r4g4b4Unorm || d.layers != n.gpu.latent_layers {
            return Err(orr_pack::Error::Malformed("neural set latents").into());
        }
        n.gpu.latents = latents.slot.index;
    }
    let streams: HashMap<AssetId, Option<Arc<TexStream>>> = r
        .textures
        .iter()
        .map(|t| (t.id, t.stream.clone()))
        .collect();
    for t in &mut r.textures {
        let Some(l) = &mut t.neural else { continue };
        let n = r
            .neural
            .iter()
            .find(|n| n.id == l.set)
            .ok_or_else(|| missing(l.set))?;
        l.latents.clone_from(&streams[&n.latents]);
        l.latent_mips = n.gpu.color_mips[..n.gpu.mips as usize]
            .iter()
            .map(|m| m.latent_mip as u32)
            .collect();
        l.gpu.set = n.record;
    }
    Ok(())
}

/// Takes resident cluster ids for every mesh of `r` and builds their acceleration
/// structures after `wait`. Returns the point the build's last work signals.
async fn build_accel(
    device: &Device,
    shared: &AccelShared,
    r: &mut Resolved,
    wait: Option<TimelinePoint>,
) -> Result<Option<TimelinePoint>> {
    // Sections whose linked coverage decides every triangle at run time, and thin dielectric and
    // presence sections, whose every candidate goes to the ray's candidate handler. Solid glass
    // traces opaque (`CLUSTER_SOLID`).
    let mut non_opaque: Vec<Vec<bool>> = r
        .meshes
        .iter()
        .map(|m| vec![false; m.sections as usize])
        .collect();
    for model in &r.models {
        let Some(mesh) = model.mesh else { continue };
        let sections = r.meshes[mesh].sections as usize;
        for (section, &mi) in model.materials.iter().take(sections).enumerate() {
            let m = &r.materials[mi].1;
            let thin_dielectric = m.flags & material_flags::DIELECTRIC != 0
                && m.flags & material_flags::THIN_WALLED != 0;
            let layered = thin_dielectric || m.flags & material_flags::PRESENCE != 0;
            if layered || (m.surface != 0 && m.flags & material_flags::ALPHA_TESTED != 0) {
                non_opaque[mesh][section] = true;
            }
        }
    }
    for m in &mut r.meshes {
        let n = m.clusters.len() as u32;
        let base = shared
            .cluster_ids
            .lock()
            .unwrap()
            .alloc(n)
            .ok_or(Error::Capacity {
                what: "resident clusters",
                capacity: shared.cluster_capacity,
            })?;
        r.cluster_ranges.push((base, n));
        m.cluster_base = base;
    }
    let inputs: Vec<MeshInput<'_>> = r
        .meshes
        .iter()
        .zip(&non_opaque)
        .map(|(m, non_opaque)| MeshInput {
            header: &m.header,
            pages: &m.page_addresses[..m.gpu.page_count as usize],
            clusters: &m.clusters,
            micromaps: &m.micromaps,
            levels: &m.levels,
            non_opaque,
            cluster_base: m.cluster_base,
            parts: &m.parts,
            cached: &m.cached,
            fine: &m.fine,
        })
        .collect();
    let refs: Vec<Vec<ClusterRef>> = inputs.iter().map(accel::cluster_refs).collect();
    let built = accel::build(device, &shared.kernels, shared.limits, &inputs, wait).await?;
    let mut skins = Vec::with_capacity(inputs.len());
    for (input, templates) in inputs.iter().zip(&built.templates) {
        skins.push(if input.header.attributes & MeshHeader::SKIN != 0 {
            Some(skin::mesh_table(device, input, templates)?)
        } else {
            None
        });
    }
    drop(inputs);
    r.accel = built.accel;
    let meshes = r.meshes.iter_mut().zip(refs).zip(built.cuts).zip(skins);
    for ((((m, mut refs), cuts), skin), non_opaque) in meshes.zip(non_opaque) {
        m.non_opaque = non_opaque;
        if let Some(table) = skin {
            m.gpu.skin = table.gpu().cast();
            let mapped = table.as_slice().expect("upload is mapped");
            m.skin_record = Some(bytemuck::pod_read_unaligned(
                &mapped[..size_of::<SkinMeshGpu>()],
            ));
            r.accel.skins.push(table);
        }
        for (r, &clas) in refs.iter_mut().zip(&cuts.clas) {
            r.clas = clas;
        }
        m.refs = refs;
        m.gpu.rt_cuts = cuts.table;
        m.gpu.rt_cut_count = cuts.count;
        m.gpu.parts = cuts.parts;
        m.gpu.first_animated = cuts.animated.first;
        m.gpu.animated_parts = cuts.animated.count;
        m.gpu.animated_sections = cuts.animated_sections;
        m.gpu.fine_cuts = cuts.fine;
        m.animated = cuts.animated;
        m.rt = cuts.rt;
    }
    Ok(built.point)
}

/// Records staged copies and discards, in their order, in one list on the copy queue, else the
/// loading queue, else the render queue, submitted after `wait`. None when there are none.
pub(crate) fn submit_copies(
    device: &Device,
    copies: &[StagedCopy],
    wait: &[TimelinePoint],
) -> Result<Option<TimelinePoint>> {
    if copies.is_empty() {
        return Ok(None);
    }
    let kind = [QueueKind::Copy, QueueKind::Loading, QueueKind::Render]
        .into_iter()
        .find(|&k| device.queue(k).is_some())
        .expect("render queue");
    let queue = device.queue(kind).expect("chosen queue");
    let mut list = queue.command_list()?;
    for c in copies {
        match c {
            StagedCopy::Discard { texture, .. } => list.discard(texture),
            StagedCopy::Buffer { src, dst } => {
                // SAFETY: `dst` lies inside one of the mount's blocks.
                let dst = unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(*dst)) };
                list.copy(dst, src.gpu(), src.size());
            }
            StagedCopy::Texture {
                src,
                texture,
                region,
            } => list.copy_to_texture(texture, region, src.gpu()),
        }
    }
    Ok(Some(queue.submit(vec![list], wait)?))
}

/// Submits the copies `task` has staged since its last batch as one batch, after binding the
/// mip tails of the streamed textures it discards, and returns the work that frees their
/// staging and in-flight bytes once it completes. Each batch waits on the mount's latest bind,
/// so a texture's copies in later batches follow its bind as those in its first do. None when
/// nothing was staged, or the mount was canceled or failed, which drops its copies unsubmitted.
pub(crate) fn submit_staged<'a>(
    device: &'a Device,
    task: &Arc<MountTask>,
) -> Option<Resumable<'a>> {
    let (copies, ticket, bound) = {
        let mut s = task.staging.lock().unwrap();
        if s.copies.is_empty() {
            return None;
        }
        (std::mem::take(&mut s.copies), s.ticket.take(), s.bound)
    };
    if !task.live() {
        return None;
    }
    let submitted = bind_tails(device, &copies).and_then(|binds| {
        let bound = binds.or(bound);
        Ok((submit_copies(device, &copies, bound.as_slice())?, bound))
    });
    let point = match submitted {
        Ok((point, bound)) => {
            let mut s = task.staging.lock().unwrap();
            s.bound = bound;
            s.last = point;
            point.expect("staged copies")
        }
        Err(e) => {
            task.fail(e);
            return None;
        }
    };
    let task = Arc::clone(task);
    Some(Box::pin(async move {
        if let Err(e) = completion(device, point).await {
            task.fail(e);
        }
        drop((copies, ticket));
    }))
}

/// Binds the mip tails of the streamed textures `copies` discards; none when they have none.
fn bind_tails(device: &Device, copies: &[StagedCopy]) -> Result<Option<TimelinePoint>> {
    let mut binds = SparseBinds::new();
    let mut any = false;
    for c in copies {
        let StagedCopy::Discard {
            stream: Some(s), ..
        } = c
        else {
            continue;
        };
        if let Some((tails, memory)) = &s.tails {
            for layer in 0..*tails {
                let at = u64::from(layer) * s.layout.tail_bytes;
                binds.tail(&s.texture, layer, Some((memory, at)))?;
                any = true;
            }
        }
    }
    Ok(if any {
        Some(device.bind_sparse(&binds)?)
    } else {
        None
    })
}

/// Ends a mount on the loading thread once its last staged copies are submitted: resolves its
/// assets, builds its acceleration structures after its copies and hands the batch to the
/// render thread.
pub(crate) fn finish<'a>(
    loading: &'a Loading,
    device: &'a Device,
    task: Arc<MountTask>,
) -> Resumable<'a> {
    Box::pin(async move { finish_mount(loading, device, &task).await })
}

async fn finish_mount(loading: &Loading, device: &Device, task: &MountTask) {
    let placed = std::mem::take(&mut *task.placed.lock().unwrap());
    let copied = task.staging.lock().unwrap().last;
    let error = task.error.lock().unwrap().take();
    let state = match error {
        _ if task.cancel.load(Ordering::Relaxed) => Err(Error::Unmounted(task.id)),
        Some(e) => Err(e),
        None => Ok(()),
    };
    let mut ranges = Vec::new();
    let result = match state.and_then(|()| resolve(task.source(), placed, device)) {
        Ok(mut r) => {
            let built = build_resolved(device, task, copied, &mut r).await;
            ranges.clone_from(&r.cluster_ranges);
            built.map(|point| (point, r))
        }
        Err(e) => Err(e),
    };
    let (point, result) = match result {
        Ok((point, r)) => (point, Ok(r)),
        Err(e) => {
            let mut ids = task.accel.cluster_ids.lock().unwrap();
            for (first, count) in ranges {
                ids.free(first, count);
            }
            (None, Err(Arc::new(e)))
        }
    };
    loading.push_batch(Batch {
        mount: task.id,
        point,
        result,
    });
}

/// Links a resolved mount's neural sets and builds its acceleration structures after its last
/// batch of staged copies, `copied`. Returns the point the last of that work signals.
async fn build_resolved(
    device: &Device,
    task: &MountTask,
    copied: Option<TimelinePoint>,
    r: &mut Resolved,
) -> Result<Option<TimelinePoint>> {
    link_neural(r)?;
    Ok(build_accel(device, &task.accel, r, copied)
        .await?
        .or(copied))
}

enum Entry {
    Pending(Arc<AtomicBool>),
    Ready(Box<Resident>),
    Failed(Arc<Error>),
    /// Unmounted while pending: its batch is dropped when it arrives.
    Unmounted,
}

/// What a ready mount holds: its lookups and everything to free when it is unmounted.
pub(crate) struct Resident {
    pub(crate) models: HashMap<AssetId, ModelId>,
    pub(crate) textures: HashMap<AssetId, TextureIndex>,
    pub(crate) materials: HashMap<AssetId, MaterialId>,
    pub(crate) pages: HashMap<AssetId, PageId>,
    pub(crate) skeletons: HashMap<AssetId, SkeletonRef>,
    pub(crate) clips: HashMap<AssetId, ClipRef>,
    pub(crate) volumes: HashMap<AssetId, VolumeRef>,
    /// Where the mount's blobs are read from, for requested blobs.
    source: Arc<BlobSource>,
    meshes: Vec<u32>,
    /// Per mesh of `meshes`: what its section-mask BLAS build from.
    rt: Vec<MeshRt>,
    /// Per mesh of `meshes`: the levels of its hierarchy, None when paged.
    hierarchy: Vec<Option<u32>>,
    /// Per mesh of `meshes`: what an edit starts from.
    edit_sources: Vec<EditMesh>,
    /// The record of each model.
    model_gpus: HashMap<ModelId, ModelGpu>,
    /// The material of each section of each model.
    model_materials: HashMap<ModelId, Vec<MaterialId>>,
    page_ranges: Vec<(u32, u32)>,
    material_ranges: Vec<(u32, u32)>,
    cluster_ranges: Vec<(u32, u32)>,
    _blocks: Vec<Allocation<u8>>,
    _accel: MountAccel,
    placed_textures: Vec<PlacedTexture>,
}

/// A mesh's record, resident page addresses and the source of its edits.
struct EditMesh {
    gpu: MeshGpu,
    addresses: Vec<u64>,
    /// The cluster id of mesh cluster 0: its pinned pages' clusters take the ids that follow.
    cluster_base: u32,
    source: Arc<EditSource>,
    /// A skinned mesh's skin table.
    skin: Option<SkinMeshGpu>,
}

/// An edit on the render thread.
enum EditEntry {
    Pending(EditPending),
    Live(Box<EditLive>),
    Failed(Arc<Error>),
}

/// A range of the mesh page table by its owner's mesh: `pages` entries for its pages, then
/// `nodes` for its node pages, then one per finer cut.
#[derive(Clone, Copy, Debug)]
struct EntryOwner {
    mount: MountId,
    /// Of the mount's meshes.
    mesh: u32,
    pages: u32,
    nodes: u32,
}

impl EntryOwner {
    /// Hands entry `local` of this range to `s`: wanted to refine `request` pixels, or read in
    /// `frame`.
    fn feedback(self, s: &mut Streaming, local: u32, request: Option<f32>, frame: u64) {
        if local < self.pages {
            s.page(self.mount, self.mesh, local, true, request, frame);
        } else if local < self.pages + self.nodes {
            s.page(
                self.mount,
                self.mesh,
                local - self.pages,
                false,
                request,
                frame,
            );
        } else {
            let k = local - self.pages - self.nodes;
            s.fine_cut(self.mount, self.mesh, k, request, frame);
        }
    }
}

/// The table ranges an edit holds: the cluster ids of its own pages' clusters, page after page,
/// and its mesh, mesh page table range and model.
#[derive(Clone)]
struct EditIds {
    cluster_base: u32,
    clusters: u32,
    mesh: Option<u32>,
    pages_at: Option<u32>,
    page_count: u32,
    model: Option<u32>,
}

/// What an edit's landing still needs: its copies, and the source's units they depend on.
struct EditPending {
    model: ModelId,
    mount: MountId,
    mesh: u32,
    own: Vec<u32>,
    groups: Vec<EditGroupGpu>,
    spheres: Option<Vec<[f32; 4]>>,
    bounds: Option<Aabb>,
    cancel: Arc<AtomicBool>,
    /// What [`Streaming::edit_holds`] keeps resident for it.
    holds: Vec<Key>,
    /// Its copies once they have landed.
    loaded: Option<Box<EditLoaded>>,
}

/// A landed edit.
struct EditLive {
    model: ModelId,
    mesh_gpu: GpuPtr<MeshGpu>,
    mount: MountId,
    source_mesh: u32,
    ids: EditIds,
    tables: EditTables,
    group_count: u32,
    clas: EditClas,
    /// Kernels change it this frame.
    dirty: bool,
    /// What they do this frame has no motion.
    without_motion: bool,
    /// The copy of last frame's pages is behind: kernels changed it last frame.
    stale: bool,
    /// Its mesh has animated parts.
    parted: bool,
    mirror: Arc<Mirror>,
    loaded: EditLoaded,
    /// What it keeps locked.
    holds: Vec<Key>,
}

impl EditLive {
    /// Where the copy of last frame's pages goes, where it comes from and its bytes.
    fn page_copy(&self) -> (GpuPtr<u8>, GpuPtr<u8>, u64) {
        let pages = self.loaded.block.gpu();
        let bytes = self.loaded.pages_bytes;
        (pages.offset(bytes), pages, bytes)
    }
}

fn record_page_copies(
    list: &mut orr_gpu::CommandList<'_>,
    copies: &[(GpuPtr<u8>, GpuPtr<u8>, u64)],
) {
    if copies.is_empty() {
        return;
    }
    list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
    for &(dst, src, bytes) in copies {
        list.copy(dst, src, bytes);
    }
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
}

/// The traversal's view of the resident hierarchies.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Hierarchy {
    /// Levels of the deepest hierarchy; paged ones count as `PAGED_DEPTH`.
    pub(crate) depth: u32,
}

/// Capacities of the global model and mesh tables.
#[derive(Clone, Copy)]
pub(crate) struct TableSizes {
    pub(crate) models: u32,
    pub(crate) meshes: u32,
    pub(crate) model_materials: u32,
    pub(crate) page_table: u32,
}

/// The render thread's mounts and the global model and mesh tables they fill.
pub(crate) struct Mounts {
    entries: Vec<Option<Entry>>,
    pub(crate) models: Allocation<ModelGpu>,
    model_ids: RangeAllocator,
    pub(crate) meshes: Allocation<MeshGpu>,
    mesh_ids: RangeAllocator,
    /// The resident cluster table, by cluster id.
    pub(crate) clusters: Allocation<ClusterRef>,
    accel: Arc<AccelShared>,
    model_materials: Allocation<u32>,
    model_material_ranges: RangeAllocator,
    page_table: Allocation<u64>,
    /// Beside each page's entry of the page table, the cluster id of its first cluster.
    page_bases: Allocation<u32>,
    page_ranges: RangeAllocator,
    sizes: TableSizes,
    pages: Vec<Option<InstancePage>>,
    /// By page: the mount and unit of its records, which its windows of rows follow.
    page_keys: Vec<Option<(MountId, u32)>>,
    /// Per slot of `pages`, the generation of the page it holds or holds next.
    page_generations: Vec<u32>,
    retired: Vec<(u64, Resident)>,
    pub(crate) streaming: Streaming,
    /// The mount and mesh of each resident model.
    owners: FxHashMap<ModelId, (MountId, u32)>,
    /// The mount and mesh whose pages, node pages and finer cuts take each range of the mesh
    /// page table, by its first entry. An edit's range takes its source mesh's pages.
    entry_owners: BTreeMap<u32, EntryOwner>,
    /// Section-mask BLAS by mesh id.
    masks: HashMap<u32, MeshMasks>,
    /// Scratch of mask builds, freed once their frame has completed.
    transient: Vec<(u64, Vec<Allocation<u8>>)>,
    edits: Vec<Option<EditEntry>>,
    /// Edits destroyed, freed once that frame has completed.
    retired_edits: Vec<(u64, Box<EditLive>)>,
    /// Triangles this frame's edits may still rebuild.
    edit_room: u32,
    /// No edit has rebuilt this frame yet.
    edit_first: bool,
    /// The rebuilds' positions and build scratch, reused every frame.
    edit_scratch: Option<Allocation<u8>>,
}

/// Section-mask BLAS a mesh can hold, at most.
const MASKS_PER_MESH: u32 = 64;

/// A mesh's section-mask BLAS and the table `MeshGpu::masks` points at.
struct MeshMasks {
    table: Allocation<MaskGpu>,
    built: Vec<(u16, MaskBlas)>,
}

/// What applying a batch writes into: the frame's records and the global tables.
pub(crate) struct Apply<'a> {
    pub(crate) device: &'a Device,
    pub(crate) window: &'a UploadWindow,
    pub(crate) seq: &'a AtomicU64,
    pub(crate) records: &'a mut Vec<Record>,
    pub(crate) materials: &'a MaterialTable,
    /// Sections and animated parts by `ModelId`.
    pub(crate) model_cpu: &'a mut Vec<ModelCpu>,
    /// Subdivision cages by `ModelId`.
    pub(crate) cages: &'a mut Vec<Option<ModelCage>>,
}

impl Apply<'_> {
    /// Sets `words` 32-bit words from `dst` to `value`.
    pub(crate) fn fill(&mut self, dst: GpuPtr<u32>, words: u32, value: u32) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.records
            .push(Record::fill(seq, dst.address().0, words, value));
    }

    pub(crate) fn put<T: bytemuck::Pod>(&mut self, dst: GpuPtr<T>, values: &[T]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let (src, slot) = self.window.slice::<T>(self.device, values.len() as u32)?;
        slot.copy_from_slice(values);
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.records
            .push(Record::values(seq, dst, src, values.len() as u32));
        Ok(())
    }
}

/// A material, its record, and the index among the pack's materials of its back.
type ResolvedMaterial = (AssetId, MaterialGpu, Option<usize>);

type ResolvedTables = (
    Vec<ResolvedMaterial>,
    Vec<ResolvedMesh>,
    Vec<ResolvedModel>,
    Vec<PlacedPage>,
    Vec<PlacedCage>,
);

/// A mount's residency units. Mesh pages are pinned up to `pinned_pages`; past it they
/// stream, each after the pages it depends on and the node pages holding its groups' leaves,
/// in a mesh whose pages stream, and are locked otherwise (the mount's acceleration
/// structures reference every cluster). A mesh's top node page is pinned and its other node
/// pages stream, each after its parent page, where its mesh pages do, and are pinned
/// otherwise.
/// Whole textures and streamed textures' mip tails are pinned; a streamed texture's tiles
/// stream, each after the coarser tile covering it, and instance pages' records stream.
fn units(
    meshes: &[ResolvedMesh],
    tables: &[(GpuPtr<u64>, GpuPtr<u32>)],
    clusters: GpuPtr<ClusterRef>,
    textures: &[PlacedTexture],
    pages: &[(PageId, &PlacedPage)],
) -> Result<MountUnits> {
    let malformed = |what| Error::Pack(orr_pack::Error::Malformed(what));
    let mut units = Vec::new();
    let mut streamed = Vec::with_capacity(meshes.len());
    let mut node_pages = Vec::with_capacity(meshes.len());
    let mut mesh_pages = Vec::with_capacity(meshes.len());
    let mut mesh_streams = Vec::new();
    let mut fine = Vec::new();
    for (mesh, (m, &(table, bases))) in (0u32..).zip(meshes.iter().zip(tables)) {
        let base = units.len() as u32;
        let mut mesh_streamed = Vec::new();
        let stream = m.streamed.as_ref().map(|pages| {
            Arc::new(MeshStream {
                entry: m.entry,
                header: m.header,
                pages: pages.clone(),
                table,
                bases,
                clusters,
                non_opaque: m.non_opaque.clone(),
                groups: Arc::clone(&m.groups),
                mesh,
                pinned: m.header.pinned_pages,
                mirrors: Mutex::new(Vec::new()),
            })
        });
        if let (Some(s), true) = (&stream, m.gpu.fine_cuts > 0) {
            let pages = m
                .fine
                .iter()
                .map(|c| {
                    let mut pages: Vec<u32> = m
                        .groups
                        .iter()
                        .filter(|g| g.error > c.threshold && g.page >= m.header.pinned_pages)
                        .map(|g| g.page)
                        .collect();
                    pages.sort_unstable();
                    pages.dedup();
                    pages
                })
                .collect();
            let mut resident: Vec<Option<Box<PageClusters>>> = std::iter::repeat_with(|| None)
                .take(m.all_pages.len())
                .collect();
            for (r, p) in resident.iter_mut().zip(&m.all_pages[..s.pinned as usize]) {
                let at = p.first as usize..(p.first + p.clusters) as usize;
                *r = Some(Box::new(PageClusters {
                    infos: m.clusters[at.clone()].to_vec(),
                    clas: m.refs[at].iter().map(|r| r.clas).collect(),
                }));
            }
            fine.push(FineInit {
                mesh,
                stream: Arc::clone(s),
                table: m.gpu.rt_cuts,
                cuts: m.fine.clone(),
                pages,
                merge: m.header.merge_error,
                animated: m.animated,
                resident,
            });
        }
        let stream_index = mesh_streams.len() as u32;
        let nodes = base + m.page_units.len() as u32;
        for (p, (bytes, deps, leaves)) in m.page_units.iter().enumerate() {
            if deps.iter().any(|&d| d as usize >= p) {
                return Err(malformed("a page depends on a later page"));
            }
            let pinned = (p as u32) < m.header.pinned_pages;
            let (start, load) = match (&stream, pinned) {
                (_, true) => (Start::Pinned, None),
                (Some(_), false) => {
                    mesh_streamed.push(units.len() as u32);
                    let load = StreamUnit::MeshPage {
                        mesh: stream_index,
                        page: p as u32,
                    };
                    (Start::Absent, Some(load))
                }
                (None, false) => (Start::Locked, None),
            };
            let leaves = leaves.iter().map(|n| nodes + n);
            let unit = UnitDesc {
                class: ResidencyClass::MeshPages,
                bytes: bytes.next_multiple_of(256),
                built: 0,
                deps: deps.iter().map(|d| base + d).chain(leaves).collect(),
                start,
            };
            units.push((unit, load));
        }
        mesh_streams.extend(stream);
        mesh_pages.push((base..nodes).collect());
        node_pages.push((nodes..nodes + m.node_units.len() as u32).collect());
        for (n, &(blob, bytes, parent)) in m.node_units.iter().enumerate() {
            let (start, deps, stream) = if parent == NodePage::TOP || m.streamed.is_none() {
                (Start::Pinned, Vec::new(), None)
            } else {
                if parent as usize >= n {
                    return Err(malformed("a node page's parent follows it"));
                }
                let stream = StreamUnit::Pool {
                    entry: m.entry,
                    blob,
                    bytes,
                    target: PoolTarget::Table(table.offset(u64::from(m.gpu.page_count) + n as u64)),
                };
                mesh_streamed.push(units.len() as u32);
                (Start::Absent, vec![nodes + parent], Some(stream))
            };
            let unit = UnitDesc {
                class: ResidencyClass::MeshPages,
                bytes: u64::from(bytes),
                built: 0,
                deps,
                start,
            };
            units.push((unit, stream));
        }
        streamed.push(mesh_streamed);
    }
    let pinned = |class, bytes| {
        let unit = UnitDesc {
            class,
            bytes,
            built: 0,
            deps: Vec::new(),
            start: Start::Pinned,
        };
        (unit, None)
    };
    let mut streams = Vec::new();
    for t in textures {
        units.push(pinned(ResidencyClass::Textures, t.bytes));
        let Some(s) = &t.stream else { continue };
        let first = units.len() as u32;
        let texture = streams.len() as u32;
        for tile in 0..s.grid.tile_count() {
            let unit = UnitDesc {
                class: ResidencyClass::Textures,
                bytes: s.layout.tile_bytes,
                built: 0,
                deps: s.grid.parent(tile).map(|p| first + p).into_iter().collect(),
                start: Start::Absent,
            };
            units.push((unit, Some(StreamUnit::Tile { texture, tile })));
        }
        streams.push((Arc::clone(s), first));
    }
    let mut page_units = Vec::with_capacity(pages.len());
    for &(id, p) in pages {
        page_units.push(units.len() as u32);
        if p.info.header.record_count == 0 {
            units.push(pinned(ResidencyClass::InstancePages, 0));
            continue;
        }
        let unit = UnitDesc {
            class: ResidencyClass::InstancePages,
            bytes: u64::from(p.bytes).next_multiple_of(256),
            built: 0,
            deps: Vec::new(),
            start: Start::Absent,
        };
        let load = StreamUnit::Pool {
            entry: p.entry,
            blob: 1,
            bytes: p.bytes,
            target: PoolTarget::InstancePage(id),
        };
        units.push((unit, Some(load)));
        let h = &p.info.header;
        for window in 0..h.motion_windows() {
            let bytes = h.motion_window_bytes(window);
            let unit = UnitDesc {
                class: ResidencyClass::InstancePages,
                bytes: u64::from(bytes).next_multiple_of(256),
                built: 0,
                deps: Vec::new(),
                start: Start::Absent,
            };
            let load = StreamUnit::Pool {
                entry: p.entry,
                blob: h.motion_blob(window),
                bytes,
                target: PoolTarget::PageWindow { page: id, window },
            };
            units.push((unit, Some(load)));
        }
    }
    Ok(MountUnits {
        fine,
        page_units,
        units,
        meshes: streamed,
        node_pages,
        textures: streams,
        mesh_pages,
        mesh_streams,
    })
}

fn capacity(what: &'static str, capacity: u32) -> Error {
    Error::Capacity { what, capacity }
}

impl Mounts {
    pub(crate) fn new(
        device: &Device,
        sizes: TableSizes,
        budgets: &crate::Budgets,
        pools: crate::scenes::PoolBytes,
        accel: &Arc<AccelShared>,
        streams: GpuPtr<GpuPtr<TextureStreamGpu>>,
        fif: u32,
    ) -> Result<Self> {
        let m = MemoryKind::DeviceLocal;
        Ok(Self {
            entries: Vec::new(),
            clusters: device.gpu_malloc_with_priority(
                m,
                MemoryPriority::High,
                u64::from(accel.cluster_capacity.max(1)),
                "resident clusters",
            )?,
            accel: Arc::clone(accel),
            models: device.gpu_malloc(m, u64::from(sizes.models), "models")?,
            model_ids: RangeAllocator::new(sizes.models),
            meshes: device.gpu_malloc(m, u64::from(sizes.meshes), "meshes")?,
            mesh_ids: RangeAllocator::new(sizes.meshes),
            model_materials: device.gpu_malloc(
                m,
                u64::from(sizes.model_materials),
                "model materials",
            )?,
            model_material_ranges: RangeAllocator::new(sizes.model_materials),
            page_table: device.gpu_malloc(m, u64::from(sizes.page_table), "mesh page table")?,
            page_bases: device.gpu_malloc(m, u64::from(sizes.page_table), "mesh page bases")?,
            page_ranges: RangeAllocator::new(sizes.page_table),
            sizes,
            pages: Vec::new(),
            page_keys: Vec::new(),
            page_generations: Vec::new(),
            retired: Vec::new(),
            streaming: Streaming::new(device, budgets, pools, accel, streams, fif)?,
            owners: FxHashMap::default(),
            entry_owners: BTreeMap::new(),
            masks: HashMap::new(),
            transient: Vec::new(),
            edits: Vec::new(),
            retired_edits: Vec::new(),
            edit_room: 0,
            edit_first: true,
            edit_scratch: None,
        })
    }

    /// Starts an edit of `desc.model` on the loading side.
    pub(crate) fn create_edit(
        &mut self,
        desc: &EditDesc<'_>,
        loading: &Arc<Loading>,
        device: &Device,
    ) -> Result<EditId> {
        let model = desc.model;
        let refuse = |reason| Err(Error::NotEditable { model, reason });
        let Some(&(mount, mesh)) = self.owners.get(&model) else {
            return refuse("the model has no resident mesh");
        };
        let r = self.resident(mount)?;
        if !r.models.values().any(|&m| m == model) {
            return refuse("the model is an edit");
        }
        let m = &r.edit_sources[mesh as usize];
        let (spheres, (own, groups)) = match m.source.finest_level {
            Some(level) => (None, edit::plan_level(&m.source.pages, level)),
            None => {
                // Emissive and caster sets hold level 0 of the whole mesh.
                let spheres = desc.touched.filter(|t| {
                    t.len() <= EDIT_SPHERES
                        && m.skin.is_none()
                        && m.gpu.emissive.count == 0
                        && m.gpu.casters.count == 0
                });
                (spheres, edit::plan(&m.source.groups, spheres))
            }
        };
        let holds = self
            .streaming
            .edit_holds(mount, mesh, &own, spheres.is_some());
        let cancel = Arc::new(AtomicBool::new(false));
        let task = EditTask {
            id: EditId(self.edits.len() as u32),
            loading: Arc::downgrade(loading),
            device: device.clone(),
            source: Arc::clone(&r.source),
            stream: Arc::clone(&m.source),
            own: own.clone(),
            cancel: Arc::clone(&cancel),
            micromaps_per_submit: self.accel.limits.micromaps,
            done: Mutex::new(EditDone::default()),
            outstanding: AtomicUsize::new(1),
        };
        let id = task.id;
        self.edits.push(Some(EditEntry::Pending(EditPending {
            model,
            mount,
            mesh,
            own,
            groups,
            spheres: spheres.map(<[_]>::to_vec),
            bounds: desc.bounds,
            cancel,
            holds,
            loaded: None,
        })));
        loading.send(Msg::Edit(Arc::new(task)));
        Ok(id)
    }

    pub(crate) fn edit_state(&self, id: EditId) -> LoadState {
        match self.edits.get(id.0 as usize).and_then(Option::as_ref) {
            Some(EditEntry::Pending(_)) => LoadState::Pending,
            Some(EditEntry::Live(e)) if !e.clas.built => LoadState::Pending,
            Some(EditEntry::Live(_)) => LoadState::Ready,
            Some(EditEntry::Failed(e)) => LoadState::Failed(Arc::clone(e)),
            None => LoadState::Failed(Arc::new(Error::UnknownEdit(id))),
        }
    }

    pub(crate) fn edit(&self, id: EditId) -> Result<EditRef> {
        match self.edits.get(id.0 as usize).and_then(Option::as_ref) {
            Some(EditEntry::Live(e)) if e.clas.built => Ok(EditRef {
                model: e.model,
                mesh: e.mesh_gpu,
                groups: e.tables.groups,
                group_count: e.group_count,
            }),
            _ => Err(Error::UnknownEdit(id)),
        }
    }

    /// Keeps an edit's landed copies until the units they depend on are resident.
    pub(crate) fn land_edit(&mut self, batch: EditBatch) {
        let slot = &mut self.edits[batch.id.0 as usize];
        let Some(EditEntry::Pending(pending)) = slot else {
            return;
        };
        match batch.result {
            Err(e) => *slot = Some(EditEntry::Failed(e)),
            Ok(loaded) => pending.loaded = Some(Box::new(loaded)),
        }
    }

    /// Asks for what pending edits depend on, and takes the ids and table ranges of each whose
    /// copies have landed beside it, with it locked, and writes their tables. An edit's copies
    /// are drawn whatever the source streams, so they are drawn only with what the traversal
    /// reads around them.
    pub(crate) fn step_edits(&mut self, w: &mut Apply<'_>) {
        for i in 0..self.edits.len() {
            let Some(EditEntry::Pending(p)) = &self.edits[i] else {
                continue;
            };
            if p.loaded.is_none() {
                self.streaming.want(&p.holds);
                continue;
            }
            let entry = match self.streaming.hold(&p.holds) {
                Hold::Waiting => continue,
                Hold::Failed => EditEntry::Failed(Arc::new(Error::NotEditable {
                    model: p.model,
                    reason: "a page its copies depend on failed to load",
                })),
                Hold::Held => {
                    let Some(EditEntry::Pending(mut p)) = self.edits[i].take() else {
                        unreachable!("matched above")
                    };
                    let loaded = *p.loaded.take().expect("landed");
                    let holds = p.holds.clone();
                    match self.place_edit(p, loaded, w) {
                        Ok(live) => EditEntry::Live(Box::new(live)),
                        Err(e) => {
                            self.streaming.release(&holds);
                            EditEntry::Failed(Arc::new(e))
                        }
                    }
                }
            };
            self.edits[i] = Some(entry);
        }
    }

    fn place_edit(
        &mut self,
        p: EditPending,
        loaded: EditLoaded,
        w: &mut Apply<'_>,
    ) -> Result<EditLive> {
        let source = &self.resident(p.mount)?.edit_sources[p.mesh as usize];
        let page_count = source.gpu.page_count;
        let pages = &source.source.pages;
        let clusters: u32 = p.own.iter().map(|&q| pages[q as usize].clusters).sum();
        let ids = EditIds {
            cluster_base: self
                .accel
                .cluster_ids
                .lock()
                .unwrap()
                .alloc(clusters)
                .ok_or_else(|| capacity("resident clusters", self.accel.cluster_capacity))?,
            clusters,
            mesh: self.mesh_ids.alloc(1),
            pages_at: self.page_ranges.alloc(page_count),
            page_count,
            model: self.model_ids.alloc(1),
        };
        let placed = match (ids.mesh, ids.pages_at, ids.model) {
            (Some(mesh), Some(start), Some(model)) => {
                self.write_edit(&p, loaded, &ids, (mesh, start, model), w)
            }
            _ => Err(capacity("meshes, mesh pages or models", self.sizes.meshes)),
        };
        if placed.is_err() {
            self.free_edit_ids(&ids);
        }
        placed
    }

    fn free_edit_ids(&mut self, ids: &EditIds) {
        self.accel
            .cluster_ids
            .lock()
            .unwrap()
            .free(ids.cluster_base, ids.clusters);
        if let Some(mesh) = ids.mesh {
            self.mesh_ids.free(mesh, 1);
        }
        if let Some(start) = ids.pages_at {
            self.page_ranges.free(start, ids.page_count);
        }
        if let Some(model) = ids.model {
            self.model_ids.free(model, 1);
        }
    }

    /// Writes a landed edit's tables under the ids it took.
    fn write_edit(
        &mut self,
        p: &EditPending,
        loaded: EditLoaded,
        ids: &EditIds,
        (mesh, start, model): (u32, u32, u32),
        w: &mut Apply<'_>,
    ) -> Result<EditLive> {
        let r = self.resident(p.mount)?;
        let source = &r.edit_sources[p.mesh as usize];
        let stream = Arc::clone(&source.source);
        let src_gpu = source.gpu;
        let addresses = source.addresses.clone();
        let src_base = source.cluster_base;
        let src_model = r.model_gpus[&p.model];
        let src_skin = source.skin;
        let page_count = ids.page_count;
        let base = ids.cluster_base;
        let bounds = p.bounds.unwrap_or(Aabb {
            min: src_model.bounds_min,
            max: src_model.bounds_max,
        });
        let parts = vec![
            PartGpu {
                bounds_min: bounds.min,
                sections: src_gpu.animated_sections,
                bounds_max: bounds.max,
                _pad: 0,
            };
            src_gpu.animated_parts as usize
        ];
        let level0 = src_skin.map_or_else(Vec::new, |_| {
            edit::level0_clusters(&p.own, &stream.pages, &loaded.clusters)
        });
        let tables = EditTables::new(
            w.device,
            &EditTablesDesc {
                spheres: p.spheres.as_deref(),
                groups: &p.groups,
                pages: (loaded.block.gpu(), loaded.pages_bytes),
                clusters: src_gpu.cluster_count,
                row: u32::from(stream.finest_level.is_none()),
                parts: &parts,
                skin: src_skin.map(|source| EditSkin {
                    source,
                    cut: &level0,
                }),
            },
        )?;
        let table = self.page_table.gpu().offset(u64::from(start));
        let bases = self.page_bases.gpu().offset(u64::from(start));

        // Its own pages' clusters take its ids, page after page; the rest of its pages are the
        // source's, under the source's ids.
        let mut own = vec![false; page_count as usize];
        let mut own_bases = Vec::with_capacity(p.own.len());
        let mut refs: Vec<ClusterRef> = Vec::with_capacity(ids.clusters as usize);
        for (&q, &address) in p.own.iter().zip(&loaded.addresses) {
            own[q as usize] = true;
            own_bases.push(base + refs.len() as u32);
            let info = stream.pages[q as usize];
            refs.extend((0..info.clusters).map(|local| ClusterRef {
                // SAFETY: the page lives in the edit's block.
                page: unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(address)) },
                local,
                mesh_cluster: info.first + local,
                clas: 0,
            }));
        }
        let at_own = |q: u32| p.own.binary_search(&q).ok();
        let mut page_table = vec![0u64; page_count as usize];
        let mut page_bases = vec![0u32; page_count as usize];
        for (q, info) in stream.pages.iter().enumerate() {
            let q32 = q as u32;
            (page_table[q], page_bases[q]) = match at_own(q32) {
                Some(k) => (loaded.addresses[k], own_bases[k]),
                None => match self.streaming.streamed_page(p.mount, p.mesh, q32) {
                    Some(Some(state)) => state,
                    Some(None) => (0, 0),
                    None => (addresses[q], src_base + info.first),
                },
            };
        }
        w.put(table, &page_table)?;
        w.put(bases, &page_bases)?;
        w.put(self.clusters.gpu().offset(u64::from(base)), &refs)?;

        let micromap = |k: usize| {
            loaded.micromaps[k]
                .as_ref()
                .map_or(0, |(m, _)| m.address().0)
        };
        let clas = EditClas::new(
            w.device,
            &ClasInputs {
                pages: (0..p.own.len())
                    .map(|k| (loaded.addresses[k], micromap(k), &loaded.clusters[k][..]))
                    .collect(),
                cluster_bases: own_bases,
                firsts: p
                    .own
                    .iter()
                    .map(|&q| stream.pages[q as usize].first)
                    .collect(),
                header: &stream.header,
                non_opaque: &stream.non_opaque,
                whole: p.spheres.is_none(),
                clas_triangles: self.accel.limits.clas_triangles,
                templates: src_skin.map(|_| tables.templates),
            },
        )?;

        // The merged stage would bypass what the edit pins. Part BLAS are gathered with the cut.
        let gpu = MeshGpu {
            pages: table,
            page_bases: bases,
            merge_error: f32::INFINITY,
            root_count: src_gpu
                .root_count
                .min(src_gpu.part_count * src_gpu.section_count),
            level_count: src_gpu.level_count.min(1),
            rt_cuts: tables.rt_cut,
            rt_cut_count: 1,
            parts: tables.parts,
            masks: GpuPtr::NULL,
            mask_count: 0,
            skin: tables.skin,
            edit: tables.edit,
            feedback_pages: start,
            fine_cuts: 0,
            ..src_gpu
        };
        let mesh_gpu = self.meshes.gpu().offset(u64::from(mesh));
        w.put(mesh_gpu, &[gpu])?;
        let record = ModelGpu {
            mesh,
            bounds_min: bounds.min,
            bounds_max: bounds.max,
            chain_count: 0,
            ..src_model
        };
        w.put(self.models.gpu().offset(u64::from(model)), &[record])?;
        let id = ModelId(model);
        let i = id.0 as usize;
        if w.model_cpu.len() <= i {
            w.model_cpu.resize(i + 1, ModelCpu::default());
        }
        if w.cages.len() <= i {
            w.cages.resize(i + 1, None);
        }
        w.model_cpu[i] = ModelCpu {
            edit: true,
            ..w.model_cpu[p.model.0 as usize]
        };
        w.cages[i] = None;
        self.owners.insert(id, (p.mount, p.mesh));
        self.entry_owners.insert(
            start,
            EntryOwner {
                mount: p.mount,
                mesh: p.mesh,
                pages: page_count,
                nodes: 0,
            },
        );
        let mirror = Arc::new(Mirror { table, bases, own });
        self.streaming.add_mirror(p.mount, p.mesh, &mirror);
        Ok(EditLive {
            model: id,
            mesh_gpu,
            mount: p.mount,
            source_mesh: p.mesh,
            ids: ids.clone(),
            group_count: p.groups.len() as u32,
            tables,
            clas,
            dirty: false,
            without_motion: false,
            // Nothing has filled the copy of last frame's pages yet.
            stale: true,
            parted: src_gpu.animated_parts > 0,
            mirror,
            loaded,
            holds: p.holds.clone(),
        })
    }

    /// Opens the frame's edit budget of `triangles`.
    pub(crate) fn begin_edits(&mut self, triangles: u32) {
        self.edit_room = triangles;
        self.edit_first = true;
    }

    /// Whether an edit of `triangles` may rebuild this frame, within the frame's budget.
    fn admit_edit(room: &mut u32, first: &mut bool, triangles: u32) -> bool {
        if !*first && triangles > *room {
            return false;
        }
        *room = room.saturating_sub(triangles);
        *first = false;
        true
    }

    pub(crate) fn edit_dirty(&mut self, id: EditId, motion: bool) -> bool {
        let Some(Some(EditEntry::Live(e))) = self.edits.get_mut(id.0 as usize) else {
            return false;
        };
        if !e.clas.built {
            return false;
        }
        if !e.dirty {
            e.dirty = Self::admit_edit(&mut self.edit_room, &mut self.edit_first, e.clas.triangles);
        }
        e.without_motion = !motion;
        e.dirty
    }

    /// Whether a copy of an edit may gather one BLAS per animated part.
    pub(crate) fn has_parted_edit(&self) -> bool {
        self.edits
            .iter()
            .flatten()
            .any(|e| matches!(e, EditEntry::Live(e) if e.parted))
    }

    /// Brings the copy of last frame's pages up to date before the scene compute point, for
    /// each edit its kernels changed last frame.
    pub(crate) fn record_edit_history(&mut self, list: &mut orr_gpu::CommandList<'_>) {
        let mut copies = Vec::new();
        for e in self.edits.iter_mut().flatten() {
            let EditEntry::Live(e) = e else { continue };
            if std::mem::take(&mut e.stale) {
                copies.push(e.page_copy());
            }
        }
        record_page_copies(list, &copies);
    }

    /// Records this frame's edit rebuilds after the scene compute point: the dirty edits and
    /// those not yet built.
    pub(crate) fn record_edits(
        &mut self,
        device: &Device,
        window: &UploadWindow,
        list: &mut orr_gpu::CommandList<'_>,
    ) -> Result<()> {
        let (room, first) = (&mut self.edit_room, &mut self.edit_first);
        let mut todo: Vec<&mut EditClas> = Vec::new();
        let mut without_motion = Vec::new();
        for e in self.edits.iter_mut().flatten() {
            let EditEntry::Live(e) = e else { continue };
            let build = if e.clas.built {
                let dirty = std::mem::take(&mut e.dirty);
                if dirty && e.without_motion {
                    without_motion.push(e.page_copy());
                } else if dirty {
                    e.stale = true;
                }
                dirty
            } else {
                Self::admit_edit(room, first, e.clas.triangles)
            };
            if build {
                todo.push(&mut e.clas);
            }
        }
        if todo.is_empty() {
            return Ok(());
        }
        let recorder = EditRecorder {
            device,
            window,
            kernels: &self.accel.kernels,
            clusters: self.clusters.gpu(),
        };
        list.begin_pass("edits", false);
        recorder.record(list, &mut todo, &mut self.edit_scratch)?;
        record_page_copies(list, &without_motion);
        list.end_pass();
        Ok(())
    }

    /// Ends an edit once `frame` has completed.
    pub(crate) fn destroy_edit(&mut self, id: EditId, frame: u64, cages: &mut [Option<ModelCage>]) {
        let Some(slot) = self.edits.get_mut(id.0 as usize) else {
            return;
        };
        match slot.take() {
            Some(EditEntry::Pending(p)) => p.cancel.store(true, Ordering::Relaxed),
            Some(EditEntry::Live(e)) => {
                if let Some(x) = cages.get_mut(e.model.0 as usize) {
                    *x = None;
                }
                self.streaming
                    .remove_mirror(e.mount, e.source_mesh, &e.mirror);
                self.streaming.release(&e.holds);
                self.owners.remove(&e.model);
                self.retired_edits.push((frame, e));
            }
            Some(EditEntry::Failed(_)) | None => {}
        }
    }

    /// Requests every streamed unit of `model`'s mesh for the next step 1.
    pub(crate) fn prefetch(&mut self, model: ModelId, cell_distance: u32) {
        if let Some(&(mount, mesh)) = self.owners.get(&model) {
            self.streaming.prefetch(mount, mesh, cell_distance);
        }
    }

    /// The prototypes of page `id`: their models looked up in the page's mount, then in the
    /// other ready mounts, and their overrides.
    pub(crate) fn page_prototypes(&self, id: PageId) -> Result<Vec<PagePrototype>> {
        let page = self.page(id).ok_or(Error::UnknownPage(id))?;
        let home = self.resident(page.mount)?;
        let others = self.entries.iter().flatten().filter_map(|e| match e {
            Entry::Ready(r) => Some(&**r),
            _ => None,
        });
        let search: Vec<&Resident> = std::iter::once(home).chain(others).collect();
        page.info
            .prototypes
            .iter()
            .map(|p| {
                let missing = || Error::Missing {
                    mount: page.mount,
                    asset: p.model,
                };
                let (r, model) = search
                    .iter()
                    .find_map(|r| r.models.get(&p.model).map(|&m| (*r, m)))
                    .ok_or_else(missing)?;
                let overrides = page.info.sections_of(p);
                let mut mask = 0u16;
                let mut sets = None;
                let base = &r.model_materials[&model];
                for (s, o) in overrides.iter().enumerate() {
                    if o.hidden != 0 && s < 16 {
                        mask |= 1 << s;
                    }
                    if o.material != AssetId::NONE {
                        let m = search
                            .iter()
                            .find_map(|r| r.materials.get(&o.material))
                            .ok_or(Error::Missing {
                                mount: page.mount,
                                asset: o.material,
                            })?;
                        let set = sets.get_or_insert_with(|| base.clone());
                        if s >= set.len() {
                            return Err(Error::Pack(orr_pack::Error::Malformed(
                                "prototype overrides past its model's sections",
                            )));
                        }
                        set[s] = *m;
                    }
                }
                let pose = u8::try_from(p.pose).map_err(|_| {
                    Error::Pack(orr_pack::Error::Malformed("prototype pose past 255"))
                })?;
                Ok(PagePrototype {
                    model,
                    records: u32::from(p.records),
                    sets,
                    mask,
                    pose,
                    flags: p.flags,
                })
            })
            .collect()
    }

    /// The slot of page `id`, unless its pack has gone.
    fn page_slot(&self, id: PageId) -> Option<usize> {
        let slot = id.slot as usize;
        (self.page_generations.get(slot) == Some(&id.generation)).then_some(slot)
    }

    /// Page `id`, while its pack holds it.
    pub(crate) fn page(&self, id: PageId) -> Option<&InstancePage> {
        self.pages[self.page_slot(id)?].as_ref()
    }

    pub(crate) fn page_mut(&mut self, id: PageId) -> Option<&mut InstancePage> {
        let slot = self.page_slot(id)?;
        self.pages[slot].as_mut()
    }

    /// The box of page `id` in world space, as a center and radius.
    pub(crate) fn page_sphere(&self, id: PageId, cell_log2: u32) -> Option<(DVec3, f64)> {
        let h = self.page(id)?.info.header;
        let cell = IVec3::from_array(h.cell).as_dvec3() * f64::from(1u32 << cell_log2);
        let half = Vec3::from_array(h.extent).as_dvec3() * 0.5;
        Some((
            cell + Vec3::from_array(h.origin).as_dvec3() + half,
            half.length(),
        ))
    }

    /// The bytes page `id`'s records and exact table take in the instance page pool.
    pub(crate) fn page_bytes(&self, id: PageId) -> u64 {
        self.page(id)
            .map_or(0, |p| u64::from(p.bytes).next_multiple_of(256))
    }

    /// The mount and records unit of page `id`.
    fn page_key(&self, id: PageId) -> Option<(MountId, u32)> {
        self.page_keys[self.page_slot(id)?]
    }

    /// Requests the records of page `id` for the next step, from `source`.
    pub(crate) fn request_page(&mut self, id: PageId, cell_distance: u32, source: RequestSource) {
        if let Some((mount, unit)) = self.page_key(id) {
            self.streaming
                .request_unit(mount, unit, cell_distance, source);
        }
    }

    /// The bytes window `window` of page `id`'s rows takes in the instance page pool.
    pub(crate) fn window_bytes(&self, id: PageId, window: u32) -> u64 {
        self.page(id).map_or(0, |p| {
            u64::from(p.info.header.motion_window_bytes(window)).next_multiple_of(256)
        })
    }

    /// Requests window `window` of page `id`'s rows for the next step, from `source`.
    pub(crate) fn request_window(
        &mut self,
        id: PageId,
        window: u32,
        cell_distance: u32,
        source: RequestSource,
    ) {
        if let Some((mount, unit)) = self.page_key(id) {
            self.streaming
                .request_unit(mount, unit + 1 + window, cell_distance, source);
        }
    }

    /// Requests what page `id` needs to stand where it starts: its records, and on an animated
    /// page the window of its start.
    pub(crate) fn prefetch_page(&mut self, id: PageId, cell_distance: u32) {
        self.request_page(id, cell_distance, RequestSource::Hints);
        let Some(p) = self.page(id) else {
            return;
        };
        if !p.info.motion_times.is_empty() {
            let h = p.info.header;
            let (k, _) = orr_pack::instance::motion_interval(&p.info.motion_times, h.motion_start);
            let window = k as u32 / h.motion_window;
            self.request_window(id, window, cell_distance, RequestSource::Hints);
        }
    }

    /// Requests the records of the pages meeting the sphere and every streamed unit of the
    /// models they place, nearer pages first, for the next step.
    pub(crate) fn prefetch_region(&mut self, center: DVec3, radius: f64, cell_log2: u32) {
        let edge = f64::from(1u32 << cell_log2);
        for slot in 0..self.pages.len() {
            let id = PageId {
                slot: slot as u32,
                generation: self.page_generations[slot],
            };
            let Some((c, r)) = self.page_sphere(id, cell_log2) else {
                continue;
            };
            let gap = (c.distance(center) - r - radius).max(0.0);
            if gap > 0.0 {
                continue;
            }
            let cells = (c.distance(center) / edge) as u32;
            self.prefetch_page(id, cells);
            let Ok(protos) = self.page_prototypes(id) else {
                continue;
            };
            for p in protos {
                self.prefetch(p.model, cells);
            }
        }
    }

    /// What the traversal needs of the resident meshes' hierarchies.
    pub(crate) fn hierarchy(&self) -> Hierarchy {
        let mut h = Hierarchy::default();
        for e in self.entries.iter().flatten() {
            let Entry::Ready(r) = e else { continue };
            for &depth in &r.hierarchy {
                h.depth = h.depth.max(depth.unwrap_or(PAGED_DEPTH));
            }
        }
        h
    }

    /// Queues blob `blob` of asset `id` of mount `m` into `target`.
    pub(crate) fn request_blob(
        &mut self,
        m: MountId,
        id: AssetId,
        blob: u32,
        target: crate::BlobTarget,
    ) -> Result<crate::BlobId> {
        let r = self.resident(m)?;
        let a = &r.source.archive;
        let missing = Error::Missing {
            mount: m,
            asset: id,
        };
        let e = a.find(id).ok_or(missing)?;
        let b = *a
            .blobs(e)
            .get(blob as usize)
            .ok_or(Error::Pack(orr_pack::Error::Malformed(
                "blob index outside its entry",
            )))?;
        let source = Arc::clone(&r.source);
        Ok(self.streaming.request_blob(&source, b, target))
    }

    /// Hands the traversal's page requests and reads of a completed frame to the residency
    /// step of `frame`.
    pub(crate) fn page_feedback(&mut self, feedback: &PageFeedback, frame: u64) {
        for &(entry, error) in &feedback.wanted {
            let owner = self.entry_owners.range(..=entry).next_back();
            if let Some((&start, &o)) = owner {
                o.feedback(&mut self.streaming, entry - start, Some(error), frame);
            }
        }
        // Touched entries rise, so the owner of each is the last range starting at or before it.
        let mut owners = self.entry_owners.iter().peekable();
        let mut owner: Option<(u32, EntryOwner)> = None;
        for (w, &word) in feedback.touched.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let entry = 32 * w as u32 + bits.trailing_zeros();
                bits &= bits - 1;
                while let Some((&start, &o)) = owners.next_if(|&(&start, _)| start <= entry) {
                    owner = Some((start, o));
                }
                if let Some((start, o)) = owner {
                    o.feedback(&mut self.streaming, entry - start, None, frame);
                }
            }
        }
    }

    /// A new pending mount and the flag that cancels it.
    pub(crate) fn add(&mut self) -> (MountId, Arc<AtomicBool>) {
        let cancel = Arc::new(AtomicBool::new(false));
        self.entries.push(Some(Entry::Pending(Arc::clone(&cancel))));
        (MountId(self.entries.len() as u32 - 1), cancel)
    }

    pub(crate) fn state(&self, m: MountId) -> LoadState {
        match self.entries.get(m.0 as usize).and_then(Option::as_ref) {
            Some(Entry::Pending(_)) => LoadState::Pending,
            Some(Entry::Ready(_)) => LoadState::Ready,
            Some(Entry::Failed(e)) => LoadState::Failed(Arc::clone(e)),
            Some(Entry::Unmounted) | None => LoadState::Failed(Arc::new(Error::UnknownMount(m))),
        }
    }

    fn resident(&self, m: MountId) -> Result<&Resident> {
        match self.entries.get(m.0 as usize).and_then(Option::as_ref) {
            Some(Entry::Ready(r)) => Ok(r),
            Some(Entry::Pending(_)) => Err(Error::NotReady(m)),
            Some(Entry::Failed(_)) => Err(Error::MountFailed(m)),
            Some(Entry::Unmounted) | None => Err(Error::UnknownMount(m)),
        }
    }

    /// Whether `model` belongs to a ready mount.
    pub(crate) fn is_live(&self, model: ModelId) -> bool {
        self.owners
            .get(&model)
            .is_some_and(|&(m, _)| self.resident(m).is_ok())
    }

    /// The pages of ready mount `m`; none for other mounts.
    pub(crate) fn pages_of(&self, m: MountId) -> Vec<PageId> {
        self.resident(m)
            .map(|r| r.pages.values().copied().collect())
            .unwrap_or_default()
    }

    /// Ends every edit of mount `m`'s meshes with it.
    fn destroy_edits_of(&mut self, m: MountId, frame: u64, cages: &mut [Option<ModelCage>]) {
        let of: Vec<EditId> = (0..self.edits.len())
            .filter(|&i| match &self.edits[i] {
                Some(EditEntry::Pending(p)) => p.mount == m,
                Some(EditEntry::Live(e)) => e.mount == m,
                _ => false,
            })
            .map(|i| EditId(i as u32))
            .collect();
        for id in of {
            self.destroy_edit(id, frame, cages);
        }
    }

    /// The models of ready mount `m`, and of the edits of its meshes; none for other mounts.
    pub(crate) fn models_of(&self, m: MountId) -> Vec<ModelId> {
        let own = self.resident(m).map(|r| r.models.values().copied());
        let edited = self.edits.iter().flatten().filter_map(|e| match e {
            EditEntry::Live(e) if e.mount == m => Some(e.model),
            _ => None,
        });
        own.into_iter().flatten().chain(edited).collect()
    }

    /// The model of live edit `id`.
    pub(crate) fn edit_model(&self, id: EditId) -> Option<ModelId> {
        match self.edits.get(id.0 as usize)? {
            Some(EditEntry::Live(e)) => Some(e.model),
            _ => None,
        }
    }

    pub(crate) fn lookup<T: Clone>(
        &self,
        m: MountId,
        id: AssetId,
        table: impl FnOnce(&Resident) -> &HashMap<AssetId, T>,
    ) -> Result<T> {
        table(self.resident(m)?)
            .get(&id)
            .cloned()
            .ok_or(Error::Missing {
                mount: m,
                asset: id,
            })
    }

    /// The texture object of a cooked texture of a ready mount.
    #[cfg(feature = "testing")]
    pub(crate) fn texture_object(&self, m: MountId, id: AssetId) -> Option<Arc<Texture>> {
        let r = self.resident(m).ok()?;
        let t = r.placed_textures.iter().find(|t| t.id == id)?;
        Some(Arc::clone(&t.texture))
    }

    /// Unmounts `m`: a pending mount is canceled, a ready one is freed once `frame` has
    /// completed.
    pub(crate) fn unmount(&mut self, m: MountId, frame: u64, cages: &mut [Option<ModelCage>]) {
        self.destroy_edits_of(m, frame, cages);
        let Some(slot) = self.entries.get_mut(m.0 as usize) else {
            return;
        };
        match slot.take() {
            Some(Entry::Pending(cancel)) => {
                cancel.store(true, Ordering::Relaxed);
                *slot = Some(Entry::Unmounted);
            }
            Some(Entry::Ready(r)) => {
                let mut r = *r;
                self.free_pages(&mut r);
                for id in r.models.values() {
                    if let Some(c) = cages.get_mut(id.0 as usize) {
                        *c = None;
                    }
                }
                self.streaming.unregister(m, frame);
                self.retired.push((frame, r));
            }
            Some(Entry::Unmounted) => *slot = Some(Entry::Unmounted),
            Some(Entry::Failed(_)) | None => {}
        }
    }

    /// Frees the mounts unmounted at or before `completed`.
    pub(crate) fn retire(&mut self, completed: u64, materials: &MaterialTable) {
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired)
            .into_iter()
            .partition(|(f, _)| *f <= completed);
        self.retired = keep;
        for (_, r) in done {
            self.release(r, materials);
        }
        self.transient.retain(|(f, _)| *f > completed);
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired_edits)
            .into_iter()
            .partition(|(f, _)| *f <= completed);
        self.retired_edits = keep;
        for (_, e) in done {
            if let Some(start) = e.ids.pages_at {
                self.entry_owners.remove(&start);
            }
            self.free_edit_ids(&e.ids);
        }
    }

    /// Ends the pages of `r` at once: nothing on the GPU follows a slot, and what a frame in
    /// flight reads of them lives in the stream pool.
    fn free_pages(&mut self, r: &mut Resident) {
        for (_, id) in r.pages.drain() {
            let slot = id.slot as usize;
            self.pages[slot] = None;
            self.page_keys[slot] = None;
            self.page_generations[slot] += 1;
        }
    }

    fn release(&mut self, mut r: Resident, materials: &MaterialTable) {
        for id in r.models.values() {
            self.owners.remove(id);
            self.model_ids.free(id.0, 1);
        }
        for &id in &r.meshes {
            self.mesh_ids.free(id, 1);
            self.masks.remove(&id);
        }
        for &(s, n) in &r.page_ranges {
            self.entry_owners.remove(&s);
            self.page_ranges.free(s, n);
        }
        for &(s, n) in &r.material_ranges {
            self.model_material_ranges.free(s, n);
        }
        for id in r.materials.values() {
            materials.free(*id);
        }
        self.free_pages(&mut r);
        let mut ids = self.accel.cluster_ids.lock().unwrap();
        for &(first, count) in &r.cluster_ranges {
            ids.free(first, count);
        }
    }

    /// Applies a batch the loading side finished: allocates its ids and table ranges and
    /// writes its table records, or drops it when the mount was unmounted.
    pub(crate) fn apply(&mut self, batch: Batch, w: &mut Apply<'_>) {
        let i = batch.mount.0 as usize;
        if !matches!(self.entries.get(i), Some(Some(Entry::Pending(_)))) {
            if let Ok(r) = batch.result {
                let mut ids = self.accel.cluster_ids.lock().unwrap();
                for &(first, count) in &r.cluster_ranges {
                    ids.free(first, count);
                }
            }
            if let Some(slot) = self.entries.get_mut(i) {
                *slot = None;
            }
            return;
        }
        let entry = match batch.result {
            Err(e) => Entry::Failed(e),
            Ok(r) => match self.place(batch.mount, r, w) {
                Ok(r) => Entry::Ready(Box::new(r)),
                Err(e) => Entry::Failed(Arc::new(e)),
            },
        };
        self.entries[i] = Some(entry);
    }

    fn place(&mut self, mount: MountId, r: Resolved, w: &mut Apply<'_>) -> Result<Resident> {
        let mut res = Resident {
            models: HashMap::new(),
            textures: r.textures.iter().map(|t| (t.id, t.slot.index)).collect(),
            materials: HashMap::new(),
            pages: HashMap::new(),
            skeletons: r.skeletons.into_iter().collect(),
            clips: r.clips.into_iter().collect(),
            volumes: r.volumes.into_iter().collect(),
            source: Arc::clone(&r.source),
            meshes: Vec::new(),
            rt: Vec::new(),
            hierarchy: Vec::new(),
            edit_sources: Vec::new(),
            model_gpus: HashMap::new(),
            model_materials: HashMap::new(),
            page_ranges: Vec::new(),
            material_ranges: Vec::new(),
            cluster_ranges: r.cluster_ranges,
            _blocks: r.blocks,
            _accel: r.accel,
            placed_textures: r.textures,
        };
        let resolved = (r.materials, r.meshes, r.models, r.pages, r.cages);
        for n in &r.neural {
            if let Err(e) = w.put(n.record, &[n.gpu]) {
                self.release(res, w.materials);
                return Err(e);
            }
        }
        for (table, words) in &r.tiles {
            if let Err(e) = w.put(*table, words) {
                self.release(res, w.materials);
                return Err(e);
            }
        }
        let result = self.fill(mount, &r.source, &mut res, resolved, w);
        match result {
            Ok(()) => Ok(res),
            Err(e) => {
                for id in res.models.values() {
                    if let Some(c) = w.cages.get_mut(id.0 as usize) {
                        *c = None;
                    }
                }
                self.streaming.unregister(mount, 0);
                self.release(res, w.materials);
                Err(e)
            }
        }
    }

    fn fill(
        &mut self,
        mount: MountId,
        source: &Arc<BlobSource>,
        res: &mut Resident,
        (materials, meshes, models, pages, cages): ResolvedTables,
        w: &mut Apply<'_>,
    ) -> Result<()> {
        let mut material_ids = Vec::with_capacity(materials.len());
        for (id, ..) in &materials {
            let mid = w.materials.alloc()?;
            res.materials.insert(*id, mid);
            material_ids.push(mid);
        }
        let mut records: Vec<MaterialGpu> = materials.iter().map(|m| m.1).collect();
        for (i, &(.., back)) in materials.iter().enumerate() {
            let Some(b) = back else { continue };
            records[i].back = material_ids[b].0;
            records[i].flags |= material_flags::HAS_BACK;
        }
        for (m, &mid) in records.iter().zip(&material_ids) {
            w.materials.note(mid, m);
            w.put(
                w.materials.table.gpu().offset(u64::from(mid.0)),
                &[w.materials.record(m)],
            )?;
        }
        let mut mesh_ids = Vec::with_capacity(meshes.len());
        let mut tables = Vec::with_capacity(meshes.len());
        for m in &meshes {
            let id = self
                .mesh_ids
                .alloc(1)
                .ok_or_else(|| capacity("meshes", self.sizes.meshes))?;
            res.meshes.push(id);
            let n = m.page_addresses.len() as u32 + m.gpu.fine_cuts;
            let start = self
                .page_ranges
                .alloc(n)
                .ok_or_else(|| capacity("mesh page table", self.sizes.page_table))?;
            res.page_ranges.push((start, n));
            let table = self.page_table.gpu().offset(u64::from(start));
            w.put(table, &m.page_addresses)?;
            let bases = self.page_bases.gpu().offset(u64::from(start));
            tables.push((table, bases));
            let pinned = &m.all_pages[..m.gpu.pinned_pages as usize];
            let firsts: Vec<u32> = pinned.iter().map(|p| m.cluster_base + p.first).collect();
            w.put(bases, &firsts)?;
            let gpu = MeshGpu {
                pages: table,
                page_bases: bases,
                node_pages: table.offset(u64::from(m.gpu.page_count)),
                feedback_pages: start,
                feedback_nodes: start + m.gpu.page_count,
                ..m.gpu
            };
            w.put(self.meshes.gpu().offset(u64::from(id)), &[gpu])?;
            w.put(
                self.clusters.gpu().offset(u64::from(m.cluster_base)),
                &m.refs,
            )?;
            res.rt.push(m.rt.clone());
            res.hierarchy.push(m.depth);
            res.edit_sources.push(EditMesh {
                gpu,
                addresses: m.page_addresses.clone(),
                cluster_base: m.cluster_base,
                source: Arc::new(EditSource {
                    entry: m.entry,
                    header: m.header,
                    groups: Arc::clone(&m.groups),
                    pages: m.all_pages.clone(),
                    non_opaque: m.non_opaque.clone(),
                    finest_level: m.levels.first().map(|&(_, first, count)| (first, count)),
                }),
                skin: m.skin_record,
            });
            self.entry_owners.insert(
                start,
                EntryOwner {
                    mount,
                    mesh: res.meshes.len() as u32 - 1,
                    pages: gpu.page_count,
                    nodes: gpu.node_page_count,
                },
            );
            mesh_ids.push(id);
        }
        for m in &models {
            let id = ModelId(
                self.model_ids
                    .alloc(1)
                    .ok_or_else(|| capacity("models", self.sizes.models))?,
            );
            res.models.insert(m.id, id);
            let n = m.materials.len() as u32;
            let start = self
                .model_material_ranges
                .alloc(n)
                .ok_or_else(|| capacity("model materials", self.sizes.model_materials))?;
            res.material_ranges.push((start, n));
            let section_materials: Vec<MaterialId> =
                m.materials.iter().map(|&i| material_ids[i]).collect();
            let list = self.model_materials.gpu().offset(u64::from(start));
            let raw: Vec<u32> = section_materials.iter().map(|id| id.0).collect();
            w.put(list, &raw)?;
            let gpu = ModelGpu {
                mesh: m.mesh.map_or(u32::MAX, |i| mesh_ids[i]),
                section_count: n,
                materials: list,
                bounds_min: m.bounds.min,
                bounds_max: m.bounds.max,
                ..bytemuck::Zeroable::zeroed()
            };
            w.put(self.models.gpu().offset(u64::from(id.0)), &[gpu])?;
            res.model_gpus.insert(id, gpu);
            res.model_materials.insert(id, section_materials.clone());
            let i = id.0 as usize;
            if w.model_cpu.len() <= i {
                w.model_cpu.resize(i + 1, ModelCpu::default());
            }
            if w.cages.len() <= i {
                w.cages.resize(i + 1, None);
            }
            let mesh_sections = m.mesh.map_or(0, |k| meshes[k].sections as usize);
            w.cages[i] = m.cage.map(|k| {
                let c = &cages[k];
                ModelCage {
                    cage: c.cage,
                    points: c.points,
                    shape: c.shape,
                    materials: section_materials[mesh_sections..].to_vec(),
                }
            });
            let Some(mesh_index) = m.mesh else {
                w.model_cpu[i] = ModelCpu {
                    sections: n,
                    joints: m.joints,
                    cage_only: true,
                    ..ModelCpu::default()
                };
                continue;
            };
            self.owners.insert(id, (mount, mesh_index as u32));
            let mesh = &meshes[mesh_index];
            let skinned = mesh.header.attributes & MeshHeader::SKIN != 0;
            w.model_cpu[i] = ModelCpu {
                sections: n,
                animated_parts: mesh.gpu.animated_parts,
                joints: m.joints,
                skin: skinned.then(|| SkinShape {
                    vertices: mesh.clusters.iter().map(|c| c.vertex_count).sum(),
                    clusters: mesh.gpu.cluster_count,
                }),
                emissive_sections: mesh.gpu.emissive.count + mesh.gpu.casters.count,
                emissive_stride: mesh.gpu.emissive.count + mesh.gpu.casters.count,
                cage_only: false,
                edit: false,
            };
        }
        let mut page_entries = Vec::with_capacity(pages.len());
        for p in &pages {
            let i = match self.pages.iter().position(Option::is_none) {
                Some(i) => i,
                None => {
                    self.pages.push(None);
                    self.page_keys.push(None);
                    self.page_generations.push(0);
                    self.pages.len() - 1
                }
            };
            self.pages[i] = Some(InstancePage {
                info: p.info.clone(),
                records: GpuPtr::NULL,
                windows: vec![0; p.info.header.motion_windows() as usize],
                bytes: p.bytes,
                mount,
            });
            let id = PageId {
                slot: i as u32,
                generation: self.page_generations[i],
            };
            res.pages.insert(p.id, id);
            page_entries.push((id, p));
        }
        let units = units(
            &meshes,
            &tables,
            self.clusters.gpu(),
            &res.placed_textures,
            &page_entries,
        )?;
        for (&(id, _), &unit) in page_entries.iter().zip(&units.page_units) {
            self.page_keys[id.slot as usize] = Some((mount, unit));
        }
        let neural: Vec<(u32, &NeuralLink)> = res
            .placed_textures
            .iter()
            .filter_map(|t| Some((t.slot.index.0, t.neural.as_ref()?)))
            .collect();
        self.streaming
            .register(mount, Arc::clone(source), units, &neural, w)?;
        Ok(())
    }

    /// Records on `list` the section-mask BLAS that copies of these models name and no frame
    /// has built, and the records that add them to their meshes' mask tables. Returns the
    /// masks left unbuilt: their mesh's table is full.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_masks(
        &mut self,
        device: &Device,
        window: &UploadWindow,
        list: &mut orr_gpu::CommandList<'_>,
        wanted: impl IntoIterator<Item = (ModelId, u16)>,
        frame: u64,
        seq: &AtomicU64,
        records: &mut Vec<Record>,
    ) -> Result<u32> {
        let mut todo: Vec<(u32, u16)> = Vec::new();
        let mut jobs: Vec<(&MeshRt, u16)> = Vec::new();
        let mut refused = 0;
        for (model, mask) in wanted {
            let Some(&(mount, mesh)) = self.owners.get(&model) else {
                continue;
            };
            let Ok(r) = self.resident(mount) else {
                continue;
            };
            // An edit's copies gather their cuts without their hidden sections.
            if !r.models.values().any(|&m| m == model) {
                continue;
            }
            let id = r.meshes[mesh as usize];
            let rt = &r.rt[mesh as usize];
            let built = self.masks.get(&id).map_or(&[][..], |m| &m.built[..]);
            if rt.cuts.is_empty() || built.iter().any(|b| b.0 == mask) || todo.contains(&(id, mask))
            {
                continue;
            }
            if built.len() + todo.iter().filter(|t| t.0 == id).count() >= MASKS_PER_MESH as usize {
                refused += 1;
                continue;
            }
            todo.push((id, mask));
            jobs.push((rt, mask));
        }
        if todo.is_empty() {
            return Ok(refused);
        }
        let mut keep = Vec::new();
        let built =
            accel::build_masks(device, &self.accel.kernels, window, list, &jobs, &mut keep)?;
        drop(jobs);
        self.transient.push((frame, keep));
        let next = |seq: &AtomicU64| seq.fetch_add(1, Ordering::Relaxed);
        for ((id, mask), blas) in todo.into_iter().zip(built) {
            let masks = match self.masks.entry(id) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => e.insert(MeshMasks {
                    table: device.gpu_malloc(
                        MemoryKind::DeviceLocal,
                        u64::from(MASKS_PER_MESH),
                        "mask table",
                    )?,
                    built: Vec::new(),
                }),
            };
            let entry = MaskGpu {
                mask: u32::from(mask),
                _pad: 0,
                blas: blas.table.gpu(),
            };
            let at = masks.table.gpu().offset(masks.built.len() as u64);
            records.push(Record::values(
                next(seq),
                at,
                window.put(device, &entry)?,
                1,
            ));
            masks.built.push((mask, blas));
            let mesh = self.meshes.gpu().offset(u64::from(id)).address().0;
            let table = window.put(device, &masks.table.gpu().address().0)?;
            let count = window.put(device, &(masks.built.len() as u32))?;
            let field = |offset: usize| mesh + offset as u64;
            records.push(Record::copy(
                next(seq),
                field(std::mem::offset_of!(MeshGpu, masks)),
                table.address().0,
                2,
            ));
            records.push(Record::copy(
                next(seq),
                field(std::mem::offset_of!(MeshGpu, mask_count)),
                count.address().0,
                1,
            ));
        }
        Ok(refused)
    }

    /// Frees every mount at once; the device is idle.
    pub(crate) fn clear(&mut self, materials: &MaterialTable) {
        for e in std::mem::take(&mut self.entries).into_iter().flatten() {
            if let Entry::Ready(r) = e {
                self.release(*r, materials);
            }
        }
        for (_, r) in std::mem::take(&mut self.retired) {
            self.release(r, materials);
        }
        self.transient.clear();
        self.edits.clear();
        self.retired_edits.clear();
    }
}
