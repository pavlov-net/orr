//! Packs mounted from a file or from memory. The loading side reads and places every asset
//! of a pack except the hierarchy pages below a mesh's top page, which stream on request;
//! the render thread takes the finished batch at step 1 and applies its tables with the
//! frame's scatter.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use orr_gpu::{
    Allocation, Device, GpuPtr, MemoryKind, QueueKind, SamplerIndex, Texture, TextureDesc,
    TextureIndex, TextureKind, TextureMemory, TextureRegion, TextureUsage, TimelinePoint, ViewDesc,
};
use orr_pack::instance::{InstancePageInfo, InstanceRecord};
use orr_pack::material::Material;
use orr_pack::mesh::{Cluster, MeshHeader, MeshInfo, NodePage, PageHeader, PageView};
use orr_pack::model::ModelInfo;
use orr_pack::texture::{TextureHeader, TextureInfo};
use orr_pack::{Aabb, Archive, AssetId, Class, Codec};

use crate::accel::{self, AccelShared, ClusterInfo, MeshInput, MountAccel, PageMicromap};
use crate::emissive::{EmissiveSpan, ModelEmission, indexed_spans};
use crate::loading::io::{AlignedBuf, ArchiveFile, Read, SECTOR, read_positioned};
use crate::loading::{BlobSource, Loading, Msg, decoded, read_stored};
use crate::material::{BoundTexture, MaterialId, MaterialTable};
use crate::memory::{Block, RangeAllocator, UploadWindow};
use crate::residency::{ResidencyClass, Start, UnitDesc};
use crate::scene::ModelId;
use crate::stream::{StreamUnit, Streaming};
use crate::upload::Record;
use crate::{ClusterRef, Error, MaterialGpu, MeshGpu, ModelGpu, Result, material_flags};

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
    bytes: u64,
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
        let b = Block {
            size: bytes,
            ..Block::of(&a)
        };
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
    assert!(
        at + bytes.len() as u64 <= block.size,
        "write outside its block"
    );
    if bytes.is_empty() {
        return Ok(None);
    }
    match block.cpu {
        // SAFETY: the block is mapped and outlives the jobs writing it, and every write
        // covers its own range.
        Some(p) => unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (p as *mut u8).add(at as usize),
                bytes.len(),
            );
            Ok(None)
        },
        None => Ok(Some(StagedCopy::Buffer {
            src: staging(device, bytes)?,
            dst: block.gpu + at,
        })),
    }
}

fn staging(device: &Device, bytes: &[u8]) -> Result<Allocation<u8>> {
    let src = device.gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, "loading staging")?;
    let p = src.cpu.expect("upload is mapped");
    // SAFETY: a fresh mapping of `bytes.len()` bytes.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.as_ptr(), bytes.len()) };
    Ok(src)
}

thread_local! {
    static PAGE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
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
        Box::new(move |stored| {
            t.step(|| decoded(t.archive(), &b, stored?, |bytes| then(&t, bytes)));
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
                        let n = read_positioned(&file, 0, &mut head)?;
                        let range = Archive::index_range(&head.as_slice()[..n])?;
                        let mut index = AlignedBuf::new(range.len as usize);
                        let n = read_positioned(&file, 0, &mut index)?;
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
        file,
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
                file: Arc::clone(&f),
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

/// Queues the reads of every asset the mount makes resident.
fn plan(t: &Arc<MountTask>, archive: Archive) -> Result<()> {
    let archive = Arc::new(archive);
    let _ = t.archive.set(Arc::clone(&archive));
    for (i, e) in archive.entries().iter().enumerate() {
        let Some(class) = Class::from_u16(e.class) else {
            continue;
        };
        if !matches!(
            class,
            Class::Mesh | Class::Texture | Class::Material | Class::Model | Class::InstancePage
        ) {
            continue;
        }
        let schema = class.schema().expect("built-in class format");
        let e = archive.expect(e.id, e.class, schema)?;
        let id = e.id;
        match class {
            Class::Mesh => fetch(t, i, 0, move |t, b| place_mesh(t, i, id, b)),
            Class::Texture => fetch(t, i, 0, move |t, b| place_texture(t, i, id, b)),
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
            _ => fetch(t, i, 0, move |t, b| place_page(t, i, id, b)),
        }
    }
    Ok(())
}

/// Blob 0, then every page and the top node page of a mesh, into one block. The other node
/// pages stream.
fn place_mesh(t: &Arc<MountTask>, entry: usize, id: AssetId, blob0: &[u8]) -> Result<()> {
    let info = MeshInfo::parse(blob0)?;
    let header = info.header;
    let node_bytes = |p: &NodePage| p.node_count * size_of::<orr_pack::mesh::Node>() as u32;
    let mut size = (blob0.len() as u64).next_multiple_of(256);
    let mut place = |bytes: u64| {
        let at = size;
        size += bytes.next_multiple_of(256);
        at
    };
    let page_clusters_at = place(4 * (info.pages.len() as u64 + 1));
    let page_at: Vec<u64> = info
        .pages
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
            info: block.gpu,
            pages: vec![0; info.pages.len()],
            node_pages: vec![0; info.node_pages.len()],
            page_units,
            node_units,
            page_clusters: block.gpu + page_clusters_at,
            levels: (
                block.gpu + u64::from(header.levels.first),
                header.levels.count,
            ),
            cluster_count: first,
            sections: info.sections.len() as u32,
            header,
            emissive: Vec::new(),
            level_ranges,
            clusters: Vec::with_capacity(first as usize),
            micromaps: vec![None; info.pages.len()],
        });
        placed.meshes.len() - 1
    };
    for (p, page) in info.pages.iter().enumerate() {
        let (at, first, len) = (page_at[p], firsts[p], page.resident_len as usize);
        fetch(t, entry, page.blob as usize, move |t, stored| {
            PAGE.with_borrow_mut(|resident| {
                resident.resize(len, 0);
                orr_pack::mesh::decode_page(stored, resident)?;
                t.write(block, at, resident)?;
                let view = PageView::new(resident, &header)?;
                let emissive = finest_emissive(&view, first, finest);
                let clusters: Vec<(u32, ClusterInfo)> = (0..view.cluster_count())
                    .map(|i| (first + i, ClusterInfo::new(p as u32, i, &view.cluster(i))))
                    .collect();
                let h: PageHeader =
                    bytemuck::pod_read_unaligned(&resident[..size_of::<PageHeader>()]);
                let micromap = (h.omm_offset < h.len && h.omm_usage.iter().any(|&n| n > 0))
                    .then_some(PageMicromap {
                        offset: h.omm_offset,
                        usage: h.omm_usage,
                    });
                let mut placed = t.placed.lock().unwrap();
                let m = &mut placed.meshes[mesh];
                m.pages[p] = block.gpu + at;
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
            t.placed.lock().unwrap().meshes[mesh].node_pages[n] = block.gpu + at;
            Ok(())
        });
    }
    Ok(())
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

/// A whole texture: created from blob 0, then every mip copied in from its blob.
fn place_texture(t: &Arc<MountTask>, entry: usize, id: AssetId, blob0: &[u8]) -> Result<()> {
    let info = Arc::new(TextureInfo::parse(blob0)?);
    let h = info.header;
    let format = gpu_format(h.vk_format).ok_or(Error::TextureFormat(h.vk_format))?;
    let mut usage = TextureUsage::SAMPLED;
    if t.host_copy {
        usage |= TextureUsage::HOST_COPY;
    }
    let desc = TextureDesc {
        kind: if h.kind == TextureHeader::CUBE {
            TextureKind::Cube
        } else {
            TextureKind::D2
        },
        format,
        extent: [h.width, h.height, 1],
        mips: u32::from(h.mip_count),
        layers: u32::from(h.layers),
        usage,
    };
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
        records: unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu)) },
        record_count: record_count as u32,
        user: if user_stride == 0 {
            GpuPtr::NULL
        } else {
            // SAFETY: the block holds the consumer data from `user_at`.
            unsafe { GpuPtr::from_address(orr_gpu::GpuAddress(block.gpu + user_at)) }
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
    pub point: Option<(QueueKind, TimelinePoint)>,
    pub result: Result<Resolved, Arc<Error>>,
}

impl Batch {
    pub(crate) fn landed(&self, device: &Device) -> bool {
        crate::loading::landed(device, self.point.as_ref())
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
    /// Resident cluster table entries from `gpu.cluster_base` on.
    refs: Vec<ClusterRef>,
}

struct ResolvedModel {
    id: AssetId,
    mesh: usize,
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
                        _pad: 0,
                        rt_cuts: GpuPtr::NULL,
                    }
                },
                page_addresses,
                sections: m.sections,
                spans: indexed_spans(m.emissive),
                header: m.header,
                levels: m.level_ranges,
                clusters,
                micromaps: m.micromaps,
                refs: Vec::new(),
            }
        })
        .collect();
    let mesh_index: HashMap<AssetId, usize> =
        meshes.iter().enumerate().map(|(i, m)| (m.id, i)).collect();
    let models = p
        .models
        .iter()
        .map(|(id, m)| {
            let mesh = *mesh_index
                .get(&m.header.mesh)
                .ok_or_else(|| missing(m.header.mesh))?;
            if m.materials.len() != meshes[mesh].sections as usize {
                return Err(orr_pack::Error::Malformed("model materials per mesh section").into());
            }
            let materials = m
                .materials
                .iter()
                .map(|id| material_index.get(id).copied().ok_or_else(|| missing(*id)))
                .collect::<Result<Vec<_>>>()?;
            Ok(ResolvedModel {
                id: *id,
                mesh,
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
) -> Result<Option<(QueueKind, TimelinePoint)>> {
    // Sections whose linked coverage decides every triangle at run time.
    let mut non_opaque: Vec<Vec<bool>> = r
        .meshes
        .iter()
        .map(|m| vec![false; m.sections as usize])
        .collect();
    for model in &r.models {
        for (section, &mi) in model.materials.iter().enumerate() {
            let m = &r.materials[mi].1;
            if m.surface != 0 && m.flags & material_flags::ALPHA_TESTED != 0 {
                non_opaque[model.mesh][section] = true;
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
        })
        .collect();
    let refs: Vec<Vec<ClusterRef>> = inputs.iter().map(accel::cluster_refs).collect();
    let built = accel::build(device, &shared.kernels, shared.limits, &inputs, wait)?;
    drop(inputs);
    for ((m, refs), cuts) in r.meshes.iter_mut().zip(refs).zip(built.cuts) {
        m.refs = refs;
        m.gpu.rt_cuts = cuts.table;
        m.gpu.rt_cut_count = cuts.count;
    }
    r.accel = built.accel;
    Ok(built.point)
}

/// Records the staged copies of a mount (and the discards of its staged textures) in one
/// list on the copy queue, else the loading queue, else the render queue. None when nothing
/// was staged.
pub(crate) fn submit_copies(
    device: &Device,
    copies: &[StagedCopy],
    textures: &[PlacedTexture],
) -> Result<Option<(QueueKind, TimelinePoint)>> {
    if copies.is_empty() {
        return Ok(None);
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
                list.copy(dst, src.gpu, src.size);
            }
            StagedCopy::Texture {
                src,
                texture,
                region,
            } => list.copy_to_texture(texture, region, src.gpu),
        }
    }
    Ok(Some((kind, queue.submit(vec![list], &[])?)))
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
            let copied = submit_copies(device, &copies, &r.textures)?;
            let built = build_accel(device, &task.accel, &mut r, copied.map(|(_, p)| p));
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
    meshes: Vec<u32>,
    page_ranges: Vec<(u32, u32)>,
    material_ranges: Vec<(u32, u32)>,
    cluster_ranges: Vec<(u32, u32)>,
    _blocks: Vec<Allocation<u8>>,
    _accel: MountAccel,
    placed_textures: Vec<PlacedTexture>,
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
}

/// What applying a batch writes into: the frame's records and the global tables.
pub(crate) struct Apply<'a> {
    pub device: &'a Device,
    pub window: &'a UploadWindow,
    pub seq: &'a AtomicU64,
    pub records: &'a mut Vec<Record>,
    pub materials: &'a MaterialTable,
    pub emission: &'a mut Vec<Option<ModelEmission>>,
}

impl Apply<'_> {
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
);

/// A mount's residency units, with how to load the streamed ones, and the streamed units
/// of each mesh. Mesh pages are pinned up to `pinned_pages` and locked after it (the mount's
/// acceleration structures reference every cluster); a mesh's top node page is pinned and
/// its other node pages stream, each after its parent page. Textures and instance pages are
/// pinned.
type Units = (Vec<(UnitDesc, Option<StreamUnit>)>, Vec<Vec<u32>>);

fn units(
    meshes: &[ResolvedMesh],
    tables: &[GpuPtr<u64>],
    textures: &[PlacedTexture],
    pages: &[PlacedPage],
) -> Result<Units> {
    let malformed = |what| Error::Pack(orr_pack::Error::Malformed(what));
    let mut units = Vec::new();
    let mut streamed = Vec::with_capacity(meshes.len());
    for (m, table) in meshes.iter().zip(tables) {
        let base = units.len() as u32;
        for (p, (bytes, deps)) in m.page_units.iter().enumerate() {
            if deps.iter().any(|&d| d as usize >= p) {
                return Err(malformed("a page depends on a later page"));
            }
            let start = if (p as u32) < m.header.pinned_pages {
                Start::Pinned
            } else {
                Start::Locked
            };
            let unit = UnitDesc {
                class: ResidencyClass::MeshPages,
                bytes: bytes.next_multiple_of(256),
                deps: deps.iter().map(|d| base + d).collect(),
                start,
            };
            units.push((unit, None));
        }
        let nodes = units.len() as u32;
        let mut mesh_streamed = Vec::new();
        for (n, &(blob, bytes, parent)) in m.node_units.iter().enumerate() {
            let (start, deps, stream) = if parent == NodePage::TOP {
                (Start::Pinned, Vec::new(), None)
            } else {
                if parent as usize >= n {
                    return Err(malformed("a node page's parent follows it"));
                }
                let stream = StreamUnit {
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
    units.extend(
        textures
            .iter()
            .map(|t| pinned(ResidencyClass::Textures, t.bytes)),
    );
    units.extend(
        pages
            .iter()
            .map(|p| pinned(ResidencyClass::InstancePages, p.bytes)),
    );
    Ok((units, streamed))
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
    ) -> Result<Self> {
        let m = MemoryKind::DeviceLocal;
        Ok(Self {
            entries: Vec::new(),
            clusters: device.gpu_malloc(
                m,
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
            streaming: Streaming::new(device, budgets),
            owners: HashMap::new(),
        })
    }

    /// Requests every streamed unit of `model`'s mesh for the next step 1.
    pub(crate) fn prefetch(&mut self, model: ModelId, cell_distance: u32) {
        if let Some(&(mount, mesh)) = self.owners.get(&model) {
            self.streaming.prefetch(mount, mesh, cell_distance);
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

    pub(crate) fn lookup<T: Copy>(
        &self,
        m: MountId,
        id: AssetId,
        table: impl FnOnce(&Resident) -> &HashMap<AssetId, T>,
    ) -> Result<T> {
        table(self.resident(m)?)
            .get(&id)
            .copied()
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
    }

    fn release(&mut self, r: Resident, device: &Device, materials: &MaterialTable) {
        for id in r.models.values() {
            self.owners.remove(id);
            self.model_ids.free(id.0, 1);
        }
        for &id in &r.meshes {
            self.mesh_ids.free(id, 1);
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
            meshes: Vec::new(),
            page_ranges: Vec::new(),
            material_ranges: Vec::new(),
            cluster_ranges: r.cluster_ranges,
            _blocks: r.blocks,
            _accel: r.accel,
            placed_textures: r.textures,
        };
        let resolved = (r.materials, r.meshes, r.models, r.pages);
        let result = self.fill(mount, &r.source, &mut res, resolved, w);
        match result {
            Ok(()) => Ok(res),
            Err(e) => {
                for id in res.models.values() {
                    w.emission[id.0 as usize] = None;
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
        (materials, meshes, models, pages): ResolvedTables,
        w: &mut Apply<'_>,
    ) -> Result<()> {
        let mut material_ids = Vec::with_capacity(materials.len());
        for (id, m) in &materials {
            let mid = w.materials.alloc()?;
            res.materials.insert(*id, mid);
            material_ids.push(mid);
            w.materials.note(mid, m);
            w.put(
                w.materials.table.gpu.offset(u64::from(mid.0)),
                std::slice::from_ref(m),
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
            let table = self.page_table.gpu.offset(u64::from(start));
            tables.push(table);
            w.put(table, &m.page_addresses)?;
            let gpu = MeshGpu {
                pages: table,
                node_pages: table.offset(u64::from(m.gpu.page_count)),
                ..m.gpu
            };
            w.put(self.meshes.gpu.offset(u64::from(id)), &[gpu])?;
            w.put(
                self.clusters.gpu.offset(u64::from(m.gpu.cluster_base)),
                &m.refs,
            )?;
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
            let list = self.model_materials.gpu.offset(u64::from(start));
            let raw: Vec<u32> = section_materials.iter().map(|id| id.0).collect();
            w.put(list, &raw)?;
            let gpu = ModelGpu {
                mesh: mesh_ids[m.mesh],
                section_count: n,
                materials: list,
                bounds_min: m.bounds.min,
                bounds_max: m.bounds.max,
            };
            w.put(self.models.gpu.offset(u64::from(id.0)), &[gpu])?;
            self.owners.insert(id, (mount, m.mesh as u32));
            let i = id.0 as usize;
            if w.emission.len() <= i {
                w.emission.resize(i + 1, None);
            }
            let spans = &meshes[m.mesh].spans;
            w.emission[i] = (!spans.is_empty()).then(|| ModelEmission {
                sections: n,
                materials: section_materials,
                spans: spans.clone(),
            });
        }
        let units = units(&meshes, &tables, &res.placed_textures, &pages)?;
        self.streaming
            .register(mount, Arc::clone(source), units.0, units.1)?;
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
    }
}
