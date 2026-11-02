//! Packs mounted from a file or from memory. The loading side reads and places every asset
//! of a pack except the hierarchy pages below a mesh's top page and the tiles above the mip
//! tails of the textures its materials bind, which stream on request; the render thread
//! takes the finished batch at step 1 and applies its tables with the frame's scatter.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use orr_gpu::{
    Allocation, Device, GpuPtr, MemoryKind, MemoryPriority, QueueKind, SamplerIndex, SparseBinds,
    Texture, TextureDesc, TextureIndex, TextureKind, TextureMemory, TextureRegion, TextureUsage,
    TimelinePoint, ViewDesc,
};
use orr_io::{AlignedBuf, Read, SECTOR};
use orr_pack::instance::{InstancePageInfo, InstanceRecord};
use orr_pack::material::Material;
use orr_pack::mesh::{Cluster, MeshHeader, MeshInfo, NodePage, PageHeader, PageView, Part};
use orr_pack::model::ModelInfo;
use orr_pack::subd::{Page as SubdPage, SubdInfo};
use orr_pack::texture::{TextureHeader, TextureInfo};
use orr_pack::volume::VolumeInfo;
use orr_pack::{Aabb, Archive, AssetId, Class, Codec};

use crate::accel::{
    self, AccelShared, ClusterInfo, MaskBlas, MeshInput, MeshRt, MountAccel, PageMicromap,
};
use crate::emissive::{EmissiveSpan, ModelEmission, indexed_spans};
use crate::loading::{ArchiveFile, BlobSource, Loading, Msg, decoded, read_stored};
use crate::material::{BoundTexture, MaterialId, MaterialTable, candidate_class};
use crate::medium_proxy::VolumeRef;
use crate::memory::{Block, RangeAllocator, UploadWindow};
use crate::residency::{ResidencyClass, Start, UnitDesc};
use crate::scene::{ModelCpu, ModelId, SkinShape};
use crate::shaders::types::SubdCageGpu;
use crate::skin;
use crate::stream::{MeshStream, MountUnits, StreamUnit, StreamedPage, Streaming};
use crate::subd::{self, CageShape, CageTables, ModelCage};
use crate::tiles::{TexStream, TileGrid};
use crate::traverse::{PAGED_DEPTH, PageFeedback};
use crate::upload::Record;
use crate::{
    ClusterRef, Error, MaskGpu, MaterialGpu, MeshGpu, ModelGpu, Result, TextureStreamGpu,
    material_flags,
};

/// Where a pack's bytes come from.
#[derive(Clone, Debug)]
pub enum PackSource {
    /// An archive file. Its handles open at mount with read and delete sharing, so a re-cook
    /// can rename a new archive over the path; the mount keeps reading the old file until it
    /// is remounted.
    File(PathBuf),
    /// A whole archive in memory, such as an orr-cook cook run in the consumer's process.
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

/// A resident instance page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub u32);

/// Heap image slots and their textures.
pub(crate) struct PlacedTexture {
    id: AssetId,
    texture: Arc<Texture>,
    index: TextureIndex,
    primaries: u8,
    staged: bool,
    /// Bytes resident from mount: the whole texture, or a streamed one's mip tails.
    bytes: u64,
    /// How a streamed texture's tiles load; none for whole textures.
    stream: Option<Arc<TexStream>>,
}

struct PlacedMesh {
    id: AssetId,
    entry: u32,
    info: u64,
    pages: Vec<u64>,
    node_pages: Vec<u64>,
    /// Per page, its resident bytes and the pages it depends on.
    page_units: Vec<(u64, Vec<u32>)>,
    /// Per node page, its blob, bytes and parent page.
    node_units: Vec<(u32, u32, u32)>,
    /// First mesh cluster per page, then the cluster count, in the mesh's block.
    page_clusters: u64,
    /// The discrete levels' table in blob 0, and their count.
    levels: (u64, u32),
    cluster_count: u32,
    sections: u32,
    header: MeshHeader,
    /// EMISSIVE clusters of the finest level, by mesh cluster index.
    emissive: Vec<(u32, Cluster)>,
    /// Discrete levels, finest first: error and mesh cluster range.
    level_ranges: Vec<(f32, u32, u32)>,
    /// Every cluster by mesh cluster index, as its page decoded.
    clusters: Vec<(u32, ClusterInfo)>,
    /// Per page, its micromap region.
    micromaps: Vec<Option<PageMicromap>>,
    parts: Vec<Part>,
    /// Per node page, the frame the traversal last reported it, in the mesh's block.
    node_stamps: u64,
    /// Levels of a resident hierarchy; None when it is paged.
    depth: Option<u32>,
    /// Pages past the pinned ones stream: per page its blob, bytes and clusters, and the
    /// frame the traversal last reported each, in the mesh's block.
    streamed: Option<(Vec<StreamedPage>, u64)>,
    /// The smallest error of a cut whose clusters all sit in pinned pages.
    min_cut: f32,
}

/// The levels of a hierarchy whose nodes are numbered breadth first from `roots`.
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

/// A resident instance page: its tables on the CPU, its records and consumer data on the
/// GPU, valid until its pack is unmounted.
#[derive(Debug)]
pub struct InstancePage {
    pub info: InstancePageInfo,
    pub records: GpuPtr<InstanceRecord>,
    pub record_count: u32,
    /// `info.header.user_stride` bytes per record; null when the stride is 0.
    pub user: GpuPtr<u8>,
}

struct PlacedPage {
    id: AssetId,
    page: InstancePage,
    bytes: u64,
}

pub(crate) enum StagedCopy {
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
    copies: Vec<StagedCopy>,
    textures: Vec<PlacedTexture>,
    materials: Vec<(AssetId, Material)>,
    meshes: Vec<PlacedMesh>,
    models: Vec<(AssetId, ModelInfo)>,
    pages: Vec<PlacedPage>,
    skeletons: Vec<(AssetId, SkeletonRef)>,
    clips: Vec<(AssetId, ClipRef)>,
    cages: Vec<PlacedCage>,
    volumes: Vec<(AssetId, VolumeRef)>,
}

/// One mount on the loading side, shared by its jobs and reads.
pub(crate) struct MountTask {
    pub id: MountId,
    loading: Weak<Loading>,
    device: Device,
    /// Writes go through staging and copies (no resizable BAR).
    staged: bool,
    host_copy: bool,
    sampler: SamplerIndex,
    cancel: Arc<AtomicBool>,
    outstanding: AtomicUsize,
    error: Mutex<Option<Error>>,
    archive: OnceLock<Arc<Archive>>,
    file: OnceLock<Arc<ArchiveFile>>,
    ring: OnceLock<bool>,
    placed: Mutex<Placed>,
    accel: Arc<AccelShared>,
}

impl MountTask {
    pub(crate) fn new(
        id: MountId,
        loading: &Arc<Loading>,
        device: &Device,
        sampler: SamplerIndex,
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
            cancel,
            // The chain from the archive's first read to the placement plan.
            outstanding: AtomicUsize::new(1),
            error: Mutex::new(None),
            archive: OnceLock::new(),
            file: OnceLock::new(),
            ring: OnceLock::new(),
            placed: Mutex::new(Placed::default()),
            accel: Arc::clone(accel),
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

    /// Runs `f` unless the mount was cancelled or failed, records its error, and ends one
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
            Some(l) => l.spawn(move || job(&t)),
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
            self.placed.lock().unwrap().copies.push(copy);
        }
        Ok(())
    }

    fn write_texture(&self, t: &Arc<Texture>, region: &TextureRegion, texels: &[u8]) -> Result<()> {
        if self.host_copy {
            self.device.host_copy_to_texture(t, region, texels)?;
        } else {
            let src = staging(&self.device, texels)?;
            let copy = StagedCopy::Texture {
                src,
                texture: Arc::clone(t),
                region: *region,
            };
            self.placed.lock().unwrap().copies.push(copy);
        }
        Ok(())
    }
}

/// `bytes` into `block` at `at`: one sequential write into mapped memory, or into staging
/// and the copy that moves it.
pub(crate) fn write(
    device: &Device,
    block: Block,
    at: u64,
    bytes: &[u8],
) -> Result<Option<StagedCopy>> {
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

/// Reads and decodes blob `blob` of entry `entry`, then runs `then` on a worker with its
/// decoded bytes.
fn fetch(
    task: &Arc<MountTask>,
    entry: usize,
    blob: usize,
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
    read_stored(
        &task.source(),
        b,
        None,
        Box::new(move |stored| {
            t.step(|| decoded(t.archive(), &b, None, stored?, |bytes| then(&t, bytes)));
        }),
    );
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
                task.spawn(move |t| {
                    t.step(|| {
                        let mut head = AlignedBuf::new(Archive::HEAD_LEN);
                        let n = file.reads.read(0, &mut head)?;
                        let range = Archive::index_range(&head.as_slice()[..n])?;
                        let mut index = AlignedBuf::new(range.len as usize);
                        let n = file.reads.read(0, &mut index)?;
                        open_index(t, &file, &path, &index, n, range.len)
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
        done: Box::new(move |r| {
            let range = r
                .map_err(Error::from)
                .and_then(|(buf, n)| Ok(Archive::index_range(&buf.as_slice()[..n])?));
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
                offset: 0,
                len: (range.len as usize).next_multiple_of(SECTOR),
                done: Box::new(move |r| match r {
                    Ok((buf, n)) => task.spawn(move |t| {
                        t.step(|| open_index(t, &f, &path, &buf, n, range.len));
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
    n: usize,
    len: u32,
) -> Result<()> {
    let len = len as usize;
    if n < len {
        return Err(orr_pack::Error::Truncated {
            need: len as u64,
            have: n as u64,
        }
        .into());
    }
    let handle = file.archive.lock().unwrap().take().expect("archive handle");
    let index = buf.as_slice()[..len].to_vec().into_boxed_slice();
    let archive = Archive::from_index_file(handle, path, index)?;
    plan(t, archive)
}

/// The textures the pack's materials bind, from the materials' blobs resident in the mount
/// section.
fn material_textures(archive: &Archive) -> Result<HashSet<AssetId>> {
    let mut bound = HashSet::new();
    for e in archive
        .entries()
        .iter()
        .filter(|e| e.class == Class::Material as u16)
    {
        let Some(&b) = archive.blobs(e).first() else {
            continue;
        };
        let Some(stored) = archive.stored(&b, None) else {
            continue;
        };
        let mut bytes = vec![0; b.raw_len as usize];
        archive.decode(&b, None, stored, &mut bytes)?;
        let m = Material::from_bytes(&bytes)?;
        bound.extend(m.textures.iter().map(|t| t.texture));
    }
    Ok(bound)
}

/// Queues the reads of every asset the mount makes resident.
fn plan(t: &Arc<MountTask>, archive: Archive) -> Result<()> {
    let archive = Arc::new(archive);
    let _ = t.archive.set(Arc::clone(&archive));
    let bound = material_textures(&archive)?;
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
        ) {
            continue;
        }
        let schema = class.schema().expect("built-in class format");
        let e = archive.expect(e.id, e.class, schema)?;
        let id = e.id;
        match class {
            Class::Mesh => fetch(t, i, 0, move |t, b| place_mesh(t, i, id, b)),
            Class::Texture => {
                let stream = bound.contains(&id);
                fetch(t, i, 0, move |t, b| place_texture(t, i, id, b, stream));
            }
            Class::Material => fetch(t, i, 0, move |t, b| {
                let m = Material::from_bytes(b)?;
                t.placed.lock().unwrap().materials.push((id, m));
                Ok(())
            }),
            Class::Model => fetch(t, i, 0, move |t, b| {
                let m = ModelInfo::parse(b)?;
                t.placed.lock().unwrap().models.push((id, m));
                Ok(())
            }),
            Class::Skeleton => fetch(t, i, 0, move |t, b| place_skeleton(t, id, b)),
            Class::Clip => fetch(t, i, 0, move |t, b| place_clip(t, i, id, b)),
            Class::Subd => fetch(t, i, 0, move |t, b| place_subd(t, i, id, b)),
            Class::Volume => fetch(t, i, 0, move |t, b| place_volume(t, i, id, b)),
            _ => fetch(t, i, 0, move |t, b| place_page(t, i, id, b)),
        }
    }
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
    fetch(t, entry, 1, move |t, b| {
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
            fetch(t, entry, grid.pages.first as usize + p, move |t, b| {
                if b.len() != bytes {
                    return Err(orr_pack::Error::Malformed("volume page").into());
                }
                t.write(block, at, b)
            });
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
        fetch(t, entry, blob as usize, move |t, b| {
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
    fetch(t, entry, 1, move |t, bytes| {
        orr_pack::clip::Clip::parse(bytemuck::bytes_of(&h), bytes)?;
        t.write(block, at, bytes)
    });
    Ok(())
}

/// Blob 0, then every page and the top node page of a mesh, into one block. The other node
/// pages stream.
fn place_mesh(t: &Arc<MountTask>, entry: usize, id: AssetId, blob0: &[u8]) -> Result<()> {
    let info = MeshInfo::parse(blob0)?;
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
    let stamps_at = place(4 * info.node_pages.len() as u64);
    let page_stamps_at = streams.then(|| place(4 * info.pages.len() as u64));
    let page_at: Vec<u64> = info.pages[..pinned]
        .iter()
        .map(|p| place(u64::from(p.resident_len)))
        .collect();
    let node_at: Vec<Option<u64>> = info
        .node_pages
        .iter()
        .map(|p| (p.parent_page == NodePage::TOP).then(|| place(u64::from(node_bytes(p)))))
        .collect();
    let page_units = info
        .pages
        .iter()
        .map(|p| {
            let deps = p.deps.first as usize..(p.deps.first + p.deps.count) as usize;
            let deps = info
                .page_deps
                .get(deps)
                .ok_or(orr_pack::Error::Malformed("page dependencies"))?;
            Ok((u64::from(p.resident_len), deps.to_vec()))
        })
        .collect::<Result<Vec<_>>>()?;
    let node_units = info
        .node_pages
        .iter()
        .map(|p| (p.blob, node_bytes(p), p.parent_page))
        .collect();
    let block = t.block(size, "mesh")?;
    t.write(block, 0, blob0)?;
    if !info.node_pages.is_empty() {
        t.write(block, stamps_at, &vec![0u8; 4 * info.node_pages.len()])?;
    }
    if let Some(at) = page_stamps_at {
        t.write(block, at, &vec![0u8; 4 * info.pages.len()])?;
    }
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
    let streamed_pages: Vec<StreamedPage> = info
        .pages
        .iter()
        .zip(&firsts)
        .map(|(p, &first)| StreamedPage {
            blob: p.blob,
            bytes: p.resident_len,
            first,
            clusters: u32::from(p.cluster_count),
        })
        .collect();
    // Clusters of streamed pages are known once they load; until then they take no cut.
    let absent: Vec<(u32, ClusterInfo)> = streamed_pages[pinned..]
        .iter()
        .zip(pinned as u32..)
        .flat_map(|(p, page)| {
            (0..p.clusters).map(move |i| (p.first + i, ClusterInfo::absent(page, i)))
        })
        .collect();
    let min_cut = info
        .groups
        .iter()
        .filter(|g| g.page as usize >= pinned && g.error.is_finite())
        .map(|g| g.error)
        .fold(0.0, f32::max);
    let finest = info.levels.first().map(|l| l.clusters);
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
            info: block.gpu(),
            pages: vec![0; info.pages.len()],
            node_pages: vec![0; info.node_pages.len()],
            page_units,
            node_units,
            page_clusters: block.gpu() + page_clusters_at,
            levels: (
                block.gpu() + u64::from(header.levels.first),
                header.levels.count,
            ),
            cluster_count: first,
            sections: info.sections.len() as u32,
            header,
            emissive: Vec::new(),
            level_ranges,
            clusters: absent,
            micromaps: vec![None; info.pages.len()],
            parts: info.parts.clone(),
            node_stamps: block.gpu() + stamps_at,
            depth,
            streamed: page_stamps_at.map(|at| (streamed_pages, block.gpu() + at)),
            min_cut,
        });
        placed.meshes.len() - 1
    };
    for (p, page) in info.pages[..pinned].iter().enumerate() {
        let (at, first, len) = (page_at[p], firsts[p], page.resident_len as usize);
        fetch(t, entry, page.blob as usize, move |t, stored| {
            PAGE.with_borrow_mut(|resident| {
                resident.resize(len, 0);
                orr_pack::mesh::decode_page(stored, resident)?;
                t.write(block, at, resident)?;
                let view = PageView::new(resident, &header)?;
                let emissive = finest_emissive(&view, first, finest);
                let h: PageHeader =
                    bytemuck::pod_read_unaligned(&resident[..size_of::<PageHeader>()]);
                let records = (0..view.cluster_count()).map(|i| view.cluster(i));
                let (infos, micromap) = page_infos(&h, records, p as u32);
                let clusters: Vec<(u32, ClusterInfo)> = (first..).zip(infos).collect();
                let mut placed = t.placed.lock().unwrap();
                let m = &mut placed.meshes[mesh];
                m.pages[p] = block.gpu() + at;
                m.emissive.extend(emissive);
                m.clusters.extend(clusters);
                m.micromaps[p] = micromap;
                Ok(())
            })
        });
    }
    for (n, node_page) in info.node_pages.iter().enumerate() {
        let Some(at) = node_at[n] else { continue };
        let len = node_page.node_count as usize * size_of::<orr_pack::mesh::Node>();
        fetch(t, entry, node_page.blob as usize, move |t, nodes| {
            if nodes.len() != len {
                return Err(orr_pack::Error::Malformed("node page length").into());
            }
            t.write(block, at, nodes)?;
            t.placed.lock().unwrap().meshes[mesh].node_pages[n] = block.gpu() + at;
            Ok(())
        });
    }
    Ok(())
}

/// Whether a mesh's pages past its pinned ones stream: a DAG mesh whose pages take more than
/// `resident_max` bytes, unless it is skinned or has emissive sections, which light and
/// skinning read cluster by cluster from mount.
fn streams_pages(info: &MeshInfo, resident_max: u64) -> bool {
    let h = &info.header;
    let bytes: u64 = info.pages.iter().map(|p| u64::from(p.resident_len)).sum();
    let emissive = info
        .sections
        .iter()
        .any(|s| s.flags & orr_pack::mesh::Section::EMISSIVE != 0);
    info.levels.is_empty()
        && h.attributes & MeshHeader::SKIN == 0
        && !emissive
        && (h.pinned_pages as usize) < info.pages.len()
        && bytes > resident_max
}

/// A page's clusters as the loading side keeps them, and its micromap region, from its
/// header and cluster records.
pub(crate) fn page_infos(
    h: &PageHeader,
    clusters: impl IntoIterator<Item = Cluster>,
    page: u32,
) -> (Vec<ClusterInfo>, Option<PageMicromap>) {
    let clusters = (0..)
        .zip(clusters)
        .map(|(i, c)| ClusterInfo::new(page, i, &c))
        .collect();
    let micromap =
        (h.omm_offset < h.len && h.omm_usage.iter().any(|&n| n > 0)).then_some(PageMicromap {
            offset: h.omm_offset,
            usage: h.omm_usage,
        });
    (clusters, micromap)
}

/// The EMISSIVE clusters of a page at the mesh's finest level: the first discrete level, or
/// in DAG mode the clusters of zero error.
fn finest_emissive(
    view: &PageView<'_>,
    first: u32,
    finest: Option<orr_pack::Span>,
) -> Vec<(u32, Cluster)> {
    (0..view.cluster_count())
        .map(|i| (first + i, view.cluster(i)))
        .filter(|(index, c)| {
            let in_finest = match finest {
                Some(s) => (s.first..s.first + s.count).contains(index),
                None => c.self_error == 0.0,
            };
            in_finest && c.flags & Cluster::EMISSIVE != 0
        })
        .collect()
}

fn gpu_format(vk_format: u32) -> Option<orr_gpu::Format> {
    use orr_gpu::Format as F;
    use orr_pack::texture::vk;
    Some(match vk_format {
        vk::R8_UNORM => F::R8Unorm,
        vk::R8G8_UNORM => F::Rg8Unorm,
        vk::R8G8B8A8_UNORM => F::Rgba8Unorm,
        vk::R8G8B8A8_SRGB => F::Rgba8Srgb,
        vk::R16_UNORM => F::R16Unorm,
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

/// A texture from blob 0: streamed when `stream` and it has mips above its tail, else whole
/// with every mip copied in from its blob. Textures streamed are those the pack's materials
/// bind, which kernels sample through the footprint function's residency clamp.
fn place_texture(
    t: &Arc<MountTask>,
    entry: usize,
    id: AssetId,
    blob0: &[u8],
    stream: bool,
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
    if stream && h.tail_first_mip > 0 && place_streamed(t, entry, id, &info, &desc)? {
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
    let slot = t.device.alloc_image_slot()?;
    let index = t.device.write_sampled(slot, &texture, &ViewDesc::default());
    t.placed.lock().unwrap().textures.push(PlacedTexture {
        id,
        texture: Arc::clone(&texture),
        index,
        primaries: h.primaries,
        staged: !t.host_copy,
        bytes: t.device.texture_requirements(&desc).0,
        stream: None,
    });
    for blob in 1..h.blob_count() {
        let (info, texture) = (Arc::clone(&info), Arc::clone(&texture));
        fetch(t, entry, blob, move |t, bytes| {
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

/// A streamed texture: a sparse image whose mip tails are bound at mount and filled from
/// the last blob, and whose tiles above them stream. False, placing nothing, when the
/// device's tile layout differs from the cooked one.
fn place_streamed(
    t: &Arc<MountTask>,
    entry: usize,
    id: AssetId,
    info: &Arc<TextureInfo>,
    desc: &TextureDesc,
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
    let slot = t.device.alloc_image_slot()?;
    let index = t.device.write_sampled(slot, &texture, &ViewDesc::default());
    let stream = Arc::new(TexStream {
        texture: Arc::clone(&texture),
        index,
        layout,
        grid,
        entry: entry as u32,
        blobs: (0..tail).map(|m| h.mip_blob(m) as u32).collect(),
        record,
        map,
        tails,
    });
    t.placed.lock().unwrap().textures.push(PlacedTexture {
        id,
        texture: Arc::clone(&texture),
        index,
        primaries: h.primaries,
        staged: true,
        bytes: tail_bytes,
        stream: Some(stream),
    });
    if tail < desc.mips {
        let info = Arc::clone(info);
        fetch(t, entry, h.mip_blob(tail), move |t, bytes| {
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
                    let copy = StagedCopy::Texture {
                        src: staging(&t.device, &texels)?,
                        texture: Arc::clone(&texture),
                        region,
                    };
                    t.placed.lock().unwrap().copies.push(copy);
                }
            }
            Ok(())
        });
    }
    Ok(true)
}

/// An instance page: blob 0 stays on the CPU; records and consumer data go into one block.
fn place_page(t: &Arc<MountTask>, entry: usize, id: AssetId, blob0: &[u8]) -> Result<()> {
    let info = InstancePageInfo::parse(blob0)?;
    let a = t.archive();
    let blobs = a.blobs(&a.entries()[entry]);
    let user_stride = u64::from(info.header.user_stride);
    let records_len = blobs.get(1).map_or(0, |b| u64::from(b.raw_len));
    let record_count = records_len / size_of::<InstanceRecord>() as u64;
    if !records_len.is_multiple_of(size_of::<InstanceRecord>() as u64)
        || record_count as usize > orr_pack::instance::MAX_RECORDS
    {
        return Err(orr_pack::Error::Malformed("instance records").into());
    }
    let user_len = record_count * user_stride;
    let user_at = records_len.next_multiple_of(256);
    let bytes = (user_at + user_len).max(16);
    let block = t.block(bytes, "instance page")?;
    let page = InstancePage {
        info,
        // SAFETY: the block holds the records from its start.
        records: unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu())) },
        record_count: record_count as u32,
        user: if user_stride == 0 {
            GpuPtr::NULL
        } else {
            // SAFETY: the block holds the consumer data from `user_at`.
            unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu() + user_at)) }
        },
    };
    t.placed
        .lock()
        .unwrap()
        .pages
        .push(PlacedPage { id, page, bytes });
    if record_count > 0 {
        fetch(t, entry, 1, move |t, records| t.write(block, 0, records));
    }
    if user_len > 0 {
        fetch(t, entry, 2, move |t, user| {
            if user.len() as u64 != user_len {
                return Err(orr_pack::Error::Malformed("instance consumer data").into());
            }
            t.write(block, user_at, user)
        });
    }
    Ok(())
}

/// A finished mount, handed to the render thread once its loading-queue work has landed.
pub(crate) struct Batch {
    pub mount: MountId,
    /// The point its staged copies signal; none when nothing was staged.
    pub point: Option<TimelinePoint>,
    pub result: Result<Resolved, Arc<Error>>,
}

impl Batch {
    pub(crate) fn landed(&self, device: &Device) -> bool {
        crate::loading::landed(device, self.point)
    }
}

struct ResolvedMesh {
    id: AssetId,
    entry: u32,
    page_units: Vec<(u64, Vec<u32>)>,
    node_units: Vec<(u32, u32, u32)>,
    gpu: MeshGpu,
    page_addresses: Vec<u64>,
    sections: u32,
    spans: Vec<EmissiveSpan>,
    header: MeshHeader,
    levels: Vec<(f32, u32, u32)>,
    clusters: Vec<ClusterInfo>,
    micromaps: Vec<Option<PageMicromap>>,
    parts: Vec<Part>,
    /// Resident cluster table entries from `gpu.cluster_base` on.
    refs: Vec<ClusterRef>,
    rt: MeshRt,
    depth: Option<u32>,
    streamed: Option<Vec<StreamedPage>>,
    min_cut: f32,
    non_opaque: Vec<bool>,
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
pub(crate) struct Resolved {
    source: Arc<BlobSource>,
    blocks: Vec<Allocation<u8>>,
    textures: Vec<PlacedTexture>,
    materials: Vec<(AssetId, MaterialGpu)>,
    meshes: Vec<ResolvedMesh>,
    models: Vec<ResolvedModel>,
    pages: Vec<PlacedPage>,
    skeletons: Vec<(AssetId, SkeletonRef)>,
    clips: Vec<(AssetId, ClipRef)>,
    cages: Vec<PlacedCage>,
    volumes: Vec<(AssetId, VolumeRef)>,
    accel: MountAccel,
    /// Resident cluster id ranges taken for the mount's meshes.
    cluster_ranges: Vec<(u32, u32)>,
}

fn resolve(source: BlobSource, p: Placed, sampler: SamplerIndex) -> Result<Resolved> {
    let missing = |id| Error::Pack(orr_pack::Error::Missing(id));
    let textures: HashMap<AssetId, BoundTexture> = p
        .textures
        .iter()
        .map(|t| {
            let b = BoundTexture {
                index: t.index,
                primaries: t.primaries,
            };
            (t.id, b)
        })
        .collect();
    let materials = p
        .materials
        .iter()
        .map(|(id, m)| {
            let mut bound = [BoundTexture::NONE; 12];
            for (b, t) in bound.iter_mut().zip(&m.textures) {
                if t.texture != AssetId::NONE {
                    *b = *textures.get(&t.texture).ok_or_else(|| missing(t.texture))?;
                }
            }
            Ok((*id, MaterialGpu::new(m, &bound, sampler)))
        })
        .collect::<Result<Vec<_>>>()?;
    let material_index: HashMap<AssetId, usize> = materials
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, i))
        .collect();
    let meshes: Vec<ResolvedMesh> = p
        .meshes
        .into_iter()
        .map(|mut m| {
            m.emissive.sort_by_key(|(i, _)| *i);
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
                unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(m.info + u64::from(span.first))) }
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
                        info: GpuPtr::from_address(orr_gpu::GpuAddress(m.info)),
                        pages: GpuPtr::NULL,
                        node_pages: GpuPtr::NULL,
                        page_clusters: GpuPtr::from_address(orr_gpu::GpuAddress(m.page_clusters)),
                        levels: GpuPtr::from_address(orr_gpu::GpuAddress(m.levels.0)),
                        page_count: pages,
                        node_page_count: node_pages,
                        cluster_count: m.cluster_count,
                        level_count: m.levels.1,
                        position_step: m.header.step(),
                        attributes: u32::from(m.header.attributes),
                        merge_error: m.header.merge_error,
                        cluster_base: 0,
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
                        node_stamps: GpuPtr::from_address(orr_gpu::GpuAddress(m.node_stamps)),
                        root_count: m.header.roots.count,
                        pinned_pages: m.header.pinned_pages,
                        skin: GpuPtr::NULL,
                        page_stamps: m.streamed.as_ref().map_or(GpuPtr::NULL, |(_, at)| {
                            GpuPtr::from_address(orr_gpu::GpuAddress(*at))
                        }),
                    }
                },
                page_addresses,
                sections: m.sections,
                spans: indexed_spans(m.emissive),
                header: m.header,
                levels: m.level_ranges,
                clusters,
                micromaps: m.micromaps,
                parts: m.parts,
                refs: Vec::new(),
                rt: MeshRt::default(),
                depth: m.depth,
                streamed: m.streamed.map(|(pages, _)| pages),
                min_cut: m.min_cut,
                non_opaque: Vec::new(),
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
        accel: MountAccel::default(),
        cluster_ranges: Vec::new(),
    })
}

/// Takes resident cluster ids for every mesh of `r` and builds their acceleration
/// structures after `wait`. Returns the point the build's last work signals.
fn build_accel(
    device: &Device,
    shared: &AccelShared,
    r: &mut Resolved,
    wait: Option<TimelinePoint>,
) -> Result<Option<TimelinePoint>> {
    // Sections whose linked coverage decides every triangle at run time, and dielectric and
    // presence sections, whose every candidate goes to the ray's candidate handler.
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
            let layered = m.flags & (material_flags::DIELECTRIC | material_flags::PRESENCE) != 0;
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
        m.gpu.cluster_base = base;
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
            cluster_base: m.gpu.cluster_base,
            parts: &m.parts,
            min_cut: m.min_cut,
        })
        .collect();
    let refs: Vec<Vec<ClusterRef>> = inputs.iter().map(accel::cluster_refs).collect();
    let built = accel::build(device, &shared.kernels, shared.limits, &inputs, wait)?;
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
            r.accel.skins.push(table);
        }
        for (r, &clas) in refs.iter_mut().zip(&cuts.rt.clas) {
            r.clas = clas;
        }
        m.refs = refs;
        m.gpu.rt_cuts = cuts.table;
        m.gpu.rt_cut_count = cuts.count;
        m.gpu.parts = cuts.parts;
        m.gpu.first_animated = cuts.animated.first;
        m.gpu.animated_parts = cuts.animated.count;
        m.gpu.animated_sections = cuts.animated_sections;
        m.rt = cuts.rt;
    }
    Ok(built.point)
}

/// Records the staged copies of a mount (and the discards of its staged textures) in one
/// list on the copy queue, else the loading queue, else the render queue. None when nothing
/// was staged.
pub(crate) fn submit_copies(
    device: &Device,
    copies: &[StagedCopy],
    textures: &[PlacedTexture],
    wait: Option<TimelinePoint>,
) -> Result<Option<TimelinePoint>> {
    if copies.is_empty() {
        return Ok(wait);
    }
    let kind = [QueueKind::Copy, QueueKind::Loading, QueueKind::Render]
        .into_iter()
        .find(|&k| device.queue(k).is_some())
        .expect("render queue");
    let queue = device.queue(kind).expect("chosen queue");
    let mut list = queue.command_list()?;
    for t in textures.iter().filter(|t| t.staged) {
        list.discard(&t.texture);
    }
    for c in copies {
        match c {
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
    Ok(Some(queue.submit(vec![list], wait.as_slice())?))
}

/// Binds the mip tails of a mount's streamed textures; none when they have none.
fn bind_tails(device: &Device, textures: &[PlacedTexture]) -> Result<Option<TimelinePoint>> {
    let mut binds = SparseBinds::new();
    let mut any = false;
    for s in textures.iter().filter_map(|t| t.stream.as_ref()) {
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

/// Ends a mount on the loading thread: resolves its assets, submits its staged copies and
/// hands the batch to the render thread.
pub(crate) fn finish(loading: &Loading, device: &Device, task: &Arc<MountTask>) {
    let mut placed = std::mem::take(&mut *task.placed.lock().unwrap());
    let copies = std::mem::take(&mut placed.copies);
    let slots: Vec<u32> = placed.textures.iter().map(|t| t.index.0).collect();
    let error = task.error.lock().unwrap().take();
    let state = match error {
        _ if task.cancel.load(Ordering::Relaxed) => Err(Error::Unmounted(task.id)),
        Some(e) => Err(e),
        None => Ok(()),
    };
    let mut ranges = Vec::new();
    let result = state
        .and_then(|()| resolve(task.source(), placed, task.sampler))
        .and_then(|mut r| {
            let tails = bind_tails(device, &r.textures)?;
            let copied = submit_copies(device, &copies, &r.textures, tails)?;
            let built = build_accel(device, &task.accel, &mut r, copied);
            ranges.clone_from(&r.cluster_ranges);
            Ok((built?.or(copied), r))
        });
    let (point, result) = match result {
        Ok((point, r)) => (point, Ok(r)),
        Err(e) => {
            for s in slots {
                device.free_image_slot_after_submitted(s);
            }
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

enum Entry {
    Pending(Arc<AtomicBool>),
    Ready(Box<Resident>),
    Failed(Arc<Error>),
    /// Unmounted while pending: its batch is dropped when it arrives.
    Unmounted,
}

/// What a ready mount holds: its lookups and everything to free when it is unmounted.
pub(crate) struct Resident {
    pub models: HashMap<AssetId, ModelId>,
    pub textures: HashMap<AssetId, TextureIndex>,
    pub materials: HashMap<AssetId, MaterialId>,
    pub pages: HashMap<AssetId, PageId>,
    pub skeletons: HashMap<AssetId, SkeletonRef>,
    pub clips: HashMap<AssetId, ClipRef>,
    pub volumes: HashMap<AssetId, VolumeRef>,
    /// Where the mount's blobs are read from, for requested blobs.
    source: Arc<BlobSource>,
    meshes: Vec<u32>,
    /// Per mesh of `meshes`: what its section-mask BLAS build from.
    rt: Vec<MeshRt>,
    /// Per mesh of `meshes`: the levels of its hierarchy (None when paged) and its clusters.
    hierarchy: Vec<(Option<u32>, u32)>,
    page_ranges: Vec<(u32, u32)>,
    material_ranges: Vec<(u32, u32)>,
    cluster_ranges: Vec<(u32, u32)>,
    _blocks: Vec<Allocation<u8>>,
    _accel: MountAccel,
    placed_textures: Vec<PlacedTexture>,
}

/// The traversal's view of the resident hierarchies.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Hierarchy {
    /// Levels of the deepest hierarchy; paged ones count as `PAGED_DEPTH`.
    pub depth: u32,
    /// Some resident mesh has a paged hierarchy.
    pub paged: bool,
    /// Clusters of the largest resident mesh.
    pub max_clusters: u32,
}

/// Capacities of the global model and mesh tables.
#[derive(Clone, Copy)]
pub(crate) struct TableSizes {
    pub models: u32,
    pub meshes: u32,
    pub model_materials: u32,
    pub page_table: u32,
}

/// The render thread's mounts and the global model and mesh tables they fill.
pub(crate) struct Mounts {
    entries: Vec<Option<Entry>>,
    pub models: Allocation<ModelGpu>,
    model_ids: RangeAllocator,
    pub meshes: Allocation<MeshGpu>,
    mesh_ids: RangeAllocator,
    /// The resident cluster table, by cluster id.
    pub clusters: Allocation<ClusterRef>,
    accel: Arc<AccelShared>,
    model_materials: Allocation<u32>,
    model_material_ranges: RangeAllocator,
    page_table: Allocation<u64>,
    page_ranges: RangeAllocator,
    sizes: TableSizes,
    pub pages: Vec<Option<InstancePage>>,
    retired: Vec<(u64, Resident)>,
    pub streaming: Streaming,
    /// The mount and mesh of each resident model.
    owners: HashMap<ModelId, (MountId, u32)>,
    /// The mount and mesh of each resident mesh id.
    mesh_owners: HashMap<u32, (MountId, u32)>,
    /// Section-mask BLAS by mesh id.
    masks: HashMap<u32, MeshMasks>,
    /// Scratch of mask builds, freed once their frame has completed.
    transient: Vec<(u64, Vec<Allocation<u8>>)>,
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
    pub device: &'a Device,
    pub window: &'a UploadWindow,
    pub seq: &'a AtomicU64,
    pub records: &'a mut Vec<Record>,
    pub materials: &'a MaterialTable,
    pub emission: &'a mut Vec<Option<ModelEmission>>,
    /// Sections and animated parts by `ModelId`.
    pub model_cpu: &'a mut Vec<ModelCpu>,
    /// Subdivision cages by `ModelId`.
    pub cages: &'a mut Vec<Option<ModelCage>>,
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

type ResolvedTables = (
    Vec<(AssetId, MaterialGpu)>,
    Vec<ResolvedMesh>,
    Vec<ResolvedModel>,
    Vec<PlacedPage>,
    Vec<PlacedCage>,
);

/// A mount's residency units. Mesh pages are pinned up to `pinned_pages`; past it they
/// stream, each after the pages it depends on, in a mesh whose pages stream, and are locked
/// otherwise (the mount's acceleration structures reference every cluster). A mesh's top
/// node page is pinned and its other node pages stream, each after its parent page.
/// Whole textures, streamed textures' mip tails and instance pages are pinned; a streamed
/// texture's tiles stream, each after the coarser tile covering it.
fn units(
    meshes: &[ResolvedMesh],
    tables: &[GpuPtr<u64>],
    clusters: GpuPtr<ClusterRef>,
    textures: &[PlacedTexture],
    pages: &[PlacedPage],
) -> Result<MountUnits> {
    let malformed = |what| Error::Pack(orr_pack::Error::Malformed(what));
    let mut units = Vec::new();
    let mut streamed = Vec::with_capacity(meshes.len());
    let mut node_pages = Vec::with_capacity(meshes.len());
    let mut mesh_pages = Vec::with_capacity(meshes.len());
    let mut mesh_streams = Vec::new();
    for (m, table) in meshes.iter().zip(tables) {
        let base = units.len() as u32;
        let mut mesh_streamed = Vec::new();
        let stream = m.streamed.as_ref().map(|pages| {
            Arc::new(MeshStream {
                entry: m.entry,
                header: m.header,
                pages: pages.clone(),
                table: *table,
                clusters: clusters.offset(u64::from(m.gpu.cluster_base)),
                cluster_base: m.gpu.cluster_base,
                non_opaque: m.non_opaque.clone(),
            })
        });
        let stream_index = mesh_streams.len() as u32;
        for (p, (bytes, deps)) in m.page_units.iter().enumerate() {
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
            let unit = UnitDesc {
                class: ResidencyClass::MeshPages,
                bytes: bytes.next_multiple_of(256),
                deps: deps.iter().map(|d| base + d).collect(),
                start,
            };
            units.push((unit, load));
        }
        mesh_streams.extend(stream);
        mesh_pages.push((base..base + m.page_units.len() as u32).collect());
        let nodes = units.len() as u32;
        node_pages.push((nodes..nodes + m.node_units.len() as u32).collect());
        for (n, &(blob, bytes, parent)) in m.node_units.iter().enumerate() {
            let (start, deps, stream) = if parent == NodePage::TOP {
                (Start::Pinned, Vec::new(), None)
            } else {
                if parent as usize >= n {
                    return Err(malformed("a node page's parent follows it"));
                }
                let stream = StreamUnit::Pool {
                    entry: m.entry,
                    blob,
                    bytes,
                    table: table.offset(u64::from(m.gpu.page_count) + n as u64),
                };
                mesh_streamed.push(units.len() as u32);
                (Start::Absent, vec![nodes + parent], Some(stream))
            };
            let unit = UnitDesc {
                class: ResidencyClass::HierarchyPages,
                bytes: u64::from(bytes),
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
                deps: s.grid.parent(tile).map(|p| first + p).into_iter().collect(),
                start: Start::Absent,
            };
            units.push((unit, Some(StreamUnit::Tile { texture, tile })));
        }
        streams.push((Arc::clone(s), first));
    }
    units.extend(
        pages
            .iter()
            .map(|p| pinned(ResidencyClass::InstancePages, p.bytes)),
    );
    Ok(MountUnits {
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
            page_ranges: RangeAllocator::new(sizes.page_table),
            sizes,
            pages: Vec::new(),
            retired: Vec::new(),
            streaming: Streaming::new(device, budgets, accel, streams, fif)?,
            owners: HashMap::new(),
            mesh_owners: HashMap::new(),
            masks: HashMap::new(),
            transient: Vec::new(),
        })
    }

    /// Requests every streamed unit of `model`'s mesh for the next step 1.
    pub(crate) fn prefetch(&mut self, model: ModelId, cell_distance: u32) {
        if let Some(&(mount, mesh)) = self.owners.get(&model) {
            self.streaming.prefetch(mount, mesh, cell_distance);
        }
    }

    /// What the traversal needs of the resident meshes' hierarchies.
    pub(crate) fn hierarchy(&self) -> Hierarchy {
        let mut h = Hierarchy::default();
        for e in self.entries.iter().flatten() {
            let Entry::Ready(r) = e else { continue };
            for &(depth, clusters) in &r.hierarchy {
                h.depth = h.depth.max(depth.unwrap_or(PAGED_DEPTH));
                h.paged |= depth.is_none();
                h.max_clusters = h.max_clusters.max(clusters);
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
    pub(crate) fn page_feedback(&mut self, feedback: &[PageFeedback], frame: u64) {
        for f in feedback {
            if let Some(&(mount, mesh)) = self.mesh_owners.get(&f.mesh) {
                let request = f.request.then_some(f.error);
                self.streaming
                    .page(mount, mesh, f.page, f.mesh_page, request, frame);
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

    /// The models of ready mount `m`; none for other mounts.
    pub(crate) fn models_of(&self, m: MountId) -> Vec<ModelId> {
        self.resident(m)
            .map(|r| r.models.values().copied().collect())
            .unwrap_or_default()
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

    /// Unmounts `m`: a pending mount is cancelled, a ready one is freed once `frame` has
    /// completed.
    pub(crate) fn unmount(
        &mut self,
        m: MountId,
        frame: u64,
        emission: &mut [Option<ModelEmission>],
        cages: &mut [Option<ModelCage>],
    ) {
        let Some(slot) = self.entries.get_mut(m.0 as usize) else {
            return;
        };
        match slot.take() {
            Some(Entry::Pending(cancel)) => {
                cancel.store(true, Ordering::Relaxed);
                *slot = Some(Entry::Unmounted);
            }
            Some(Entry::Ready(r)) => {
                let r = *r;
                for id in r.models.values() {
                    if let Some(e) = emission.get_mut(id.0 as usize) {
                        *e = None;
                    }
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
    pub(crate) fn retire(&mut self, completed: u64, device: &Device, materials: &MaterialTable) {
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired)
            .into_iter()
            .partition(|(f, _)| *f <= completed);
        self.retired = keep;
        for (_, r) in done {
            self.release(r, device, materials);
        }
        self.transient.retain(|(f, _)| *f > completed);
    }

    fn release(&mut self, r: Resident, device: &Device, materials: &MaterialTable) {
        for id in r.models.values() {
            self.owners.remove(id);
            self.model_ids.free(id.0, 1);
        }
        for &id in &r.meshes {
            self.mesh_ids.free(id, 1);
            self.masks.remove(&id);
            self.mesh_owners.remove(&id);
        }
        for &(s, n) in &r.page_ranges {
            self.page_ranges.free(s, n);
        }
        for &(s, n) in &r.material_ranges {
            self.model_material_ranges.free(s, n);
        }
        for id in r.materials.values() {
            materials.free(*id);
        }
        for id in r.pages.values() {
            self.pages[id.0 as usize] = None;
        }
        for t in &r.placed_textures {
            device.free_image_slot_after_submitted(t.index.0);
        }
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
                for t in &r.textures {
                    w.device.free_image_slot_after_submitted(t.index.0);
                }
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
            textures: r.textures.iter().map(|t| (t.id, t.index)).collect(),
            materials: HashMap::new(),
            pages: HashMap::new(),
            skeletons: r.skeletons.into_iter().collect(),
            clips: r.clips.into_iter().collect(),
            volumes: r.volumes.into_iter().collect(),
            source: Arc::clone(&r.source),
            meshes: Vec::new(),
            rt: Vec::new(),
            hierarchy: Vec::new(),
            page_ranges: Vec::new(),
            material_ranges: Vec::new(),
            cluster_ranges: r.cluster_ranges,
            _blocks: r.blocks,
            _accel: r.accel,
            placed_textures: r.textures,
        };
        let resolved = (r.materials, r.meshes, r.models, r.pages, r.cages);
        let result = self.fill(mount, &r.source, &mut res, resolved, w);
        match result {
            Ok(()) => Ok(res),
            Err(e) => {
                for id in res.models.values() {
                    w.emission[id.0 as usize] = None;
                    if let Some(c) = w.cages.get_mut(id.0 as usize) {
                        *c = None;
                    }
                }
                self.streaming.unregister(mount, 0);
                self.release(res, w.device, w.materials);
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
        for (id, m) in &materials {
            let mid = w.materials.alloc()?;
            res.materials.insert(*id, mid);
            material_ids.push(mid);
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
                .ok_or(capacity("meshes", self.sizes.meshes))?;
            res.meshes.push(id);
            let n = m.page_addresses.len() as u32;
            let start = self
                .page_ranges
                .alloc(n)
                .ok_or(capacity("mesh page table", self.sizes.page_table))?;
            res.page_ranges.push((start, n));
            let table = self.page_table.gpu().offset(u64::from(start));
            tables.push(table);
            w.put(table, &m.page_addresses)?;
            let gpu = MeshGpu {
                pages: table,
                node_pages: table.offset(u64::from(m.gpu.page_count)),
                ..m.gpu
            };
            w.put(self.meshes.gpu().offset(u64::from(id)), &[gpu])?;
            w.put(
                self.clusters.gpu().offset(u64::from(m.gpu.cluster_base)),
                &m.refs,
            )?;
            res.rt.push(m.rt.clone());
            res.hierarchy.push((m.depth, m.gpu.cluster_count));
            self.mesh_owners
                .insert(id, (mount, res.meshes.len() as u32 - 1));
            mesh_ids.push(id);
        }
        for m in &models {
            let id = ModelId(
                self.model_ids
                    .alloc(1)
                    .ok_or(capacity("models", self.sizes.models))?,
            );
            res.models.insert(m.id, id);
            let n = m.materials.len() as u32;
            let start = self
                .model_material_ranges
                .alloc(n)
                .ok_or(capacity("model materials", self.sizes.model_materials))?;
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
            let i = id.0 as usize;
            if w.emission.len() <= i {
                w.emission.resize(i + 1, None);
            }
            if w.model_cpu.len() <= i {
                w.model_cpu.resize(i + 1, ModelCpu::default());
            }
            if w.cages.len() <= i {
                w.cages.resize(i + 1, None);
            }
            let mesh_sections = m.mesh.map_or(0, |k| meshes[k].sections as usize);
            w.cages[i] = m.cage.map(|k| {
                let c = &cages[k];
                let flags: Vec<u32> = m.materials[mesh_sections..]
                    .iter()
                    .map(|&k| materials[k].1.flags)
                    .collect();
                ModelCage {
                    cage: c.cage,
                    points: c.points,
                    shape: c.shape,
                    materials: section_materials[mesh_sections..].to_vec(),
                    opacity: subd::opacity(&flags),
                }
            });
            let Some(mesh_index) = m.mesh else {
                w.model_cpu[i] = ModelCpu {
                    sections: n,
                    joints: m.joints,
                    cage_only: true,
                    ..ModelCpu::default()
                };
                w.emission[i] = None;
                continue;
            };
            self.owners.insert(id, (mount, mesh_index as u32));
            let mesh = &meshes[mesh_index];
            let skinned = mesh.header.attributes & MeshHeader::SKIN != 0;
            let mesh_materials = &m.materials[..mesh_sections];
            w.model_cpu[i] = ModelCpu {
                sections: n,
                animated_parts: mesh.gpu.animated_parts,
                joints: m.joints,
                skin: skinned.then(|| SkinShape {
                    vertices: mesh.clusters.iter().map(|c| c.vertex_count).sum(),
                    clusters: mesh.gpu.cluster_count,
                }),
                candidate_class: candidate_class(
                    mesh_materials.iter().map(|&k| materials[k].1.flags),
                ),
                cage_only: false,
            };
            let spans = &mesh.spans;
            w.emission[i] = (!spans.is_empty()).then(|| ModelEmission {
                sections: n,
                materials: section_materials,
                spans: spans.clone(),
            });
        }
        let units = units(
            &meshes,
            &tables,
            self.clusters.gpu(),
            &res.placed_textures,
            &pages,
        )?;
        self.streaming
            .register(mount, Arc::clone(source), units, w)?;
        for p in pages {
            let i = match self.pages.iter().position(Option::is_none) {
                Some(i) => i,
                None => {
                    self.pages.push(None);
                    self.pages.len() - 1
                }
            };
            self.pages[i] = Some(p.page);
            res.pages.insert(p.id, PageId(i as u32));
        }
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
    pub(crate) fn clear(&mut self, device: &Device, materials: &MaterialTable) {
        for e in std::mem::take(&mut self.entries).into_iter().flatten() {
            if let Entry::Ready(r) = e {
                self.release(*r, device, materials);
            }
        }
        for (_, r) in std::mem::take(&mut self.retired) {
            self.release(r, device, materials);
        }
        self.transient.clear();
    }
}
