//! Packs mounted from a file or from memory. The loading side reads and places every asset
//! of a pack (whole-pack residency); the render thread takes the finished batch at step 1
//! and applies its tables with the frame's scatter.

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
use orr_pack::mesh::{Cluster, MeshHeader, MeshInfo, PageHeader, PageView};
use orr_pack::model::ModelInfo;
use orr_pack::texture::{TextureHeader, TextureInfo};
use orr_pack::{Aabb, Archive, AssetId, Blob, Class, Codec};

use crate::accel::{self, AccelShared, ClusterInfo, MeshInput, MountAccel, PageMicromap};
use crate::emissive::{EmissiveSpan, ModelEmission, indexed_spans};
use crate::loading::io::{AlignedBuf, ArchiveFile, Read, SECTOR, read_positioned};
use crate::loading::{Loading, Msg};
use crate::material::{BoundTexture, MaterialId, MaterialTable};
use crate::memory::{RangeAllocator, UploadWindow};
use crate::scene::ModelId;
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
}

struct PlacedMesh {
    id: AssetId,
    info: u64,
    pages: Vec<u64>,
    node_pages: Vec<u64>,
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
}

enum StagedCopy {
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

/// Memory every job of a mount writes into: mapped Upload memory, or with
/// [`MountTask::staged`] DeviceLocal memory the copy queue fills from staging.
#[derive(Clone, Copy)]
struct Block {
    gpu: u64,
    cpu: Option<usize>,
    size: u64,
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

    fn block(&self, bytes: u64, name: &str) -> Result<Block> {
        let kind = if self.staged {
            MemoryKind::DeviceLocal
        } else {
            MemoryKind::Upload
        };
        let a = self.device.gpu_malloc::<u8>(kind, bytes, name)?;
        let b = Block {
            gpu: a.gpu.address().0,
            cpu: a.cpu.map(|p| p.as_ptr() as usize),
            size: bytes,
        };
        self.placed.lock().unwrap().blocks.push(a);
        Ok(b)
    }

    /// `bytes` into `block` at `at`: one sequential write into mapped memory, or into staging
    /// that the batch copies.
    fn write(&self, block: Block, at: u64, bytes: &[u8]) -> Result<()> {
        assert!(
            at + bytes.len() as u64 <= block.size,
            "write outside its block"
        );
        if bytes.is_empty() {
            return Ok(());
        }
        match block.cpu {
            // SAFETY: the block is mapped, outlives the mount's jobs, and every write of a
            // mount covers its own range.
            Some(p) => unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    (p as *mut u8).add(at as usize),
                    bytes.len(),
                );
            },
            None => {
                let src = self.staging(bytes)?;
                let copy = StagedCopy::Buffer {
                    src,
                    dst: block.gpu + at,
                };
                self.placed.lock().unwrap().copies.push(copy);
            }
        }
        Ok(())
    }

    fn staging(&self, bytes: &[u8]) -> Result<Allocation<u8>> {
        let src = self.device.gpu_malloc::<u8>(
            MemoryKind::Upload,
            bytes.len() as u64,
            "loading staging",
        )?;
        let p = src.cpu.expect("upload is mapped");
        // SAFETY: a fresh mapping of `bytes.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.as_ptr(), bytes.len()) };
        Ok(src)
    }

    fn write_texture(&self, t: &Arc<Texture>, region: &TextureRegion, texels: &[u8]) -> Result<()> {
        if self.host_copy {
            self.device.host_copy_to_texture(t, region, texels)?;
        } else {
            let src = self.staging(texels)?;
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

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    static PAGE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Decodes blob `b` from its stored bytes into the calling worker's scratch and hands the
/// decoded bytes to `then`.
fn decoded(
    archive: &Archive,
    b: &Blob,
    stored: &[u8],
    then: impl FnOnce(&[u8]) -> Result<()>,
) -> Result<()> {
    SCRATCH.with_borrow_mut(|s| {
        s.resize(b.raw_len as usize, 0);
        archive.decode(b, None, stored, s)?;
        then(s)
    })
}

type Then = Box<dyn FnOnce(&Arc<MountTask>, &[u8]) -> Result<()> + Send>;

/// Reads and decodes blob `blob` of entry `entry`, then runs `then` on a worker with its
/// decoded bytes.
fn fetch(
    task: &Arc<MountTask>,
    entry: usize,
    blob: usize,
    then: impl FnOnce(&Arc<MountTask>, &[u8]) -> Result<()> + Send + 'static,
) {
    let then: Then = Box::new(then);
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
    if archive.stored(&b, None).is_some() {
        task.spawn(move |t| {
            t.step(|| {
                let a = t.archive();
                let stored = a.stored(&b, None).expect("stored in place");
                decoded(a, &b, stored, |bytes| then(t, bytes))
            });
        });
        return;
    }
    let range = archive.stored_range(&b, None);
    let file = Arc::clone(task.file.get().expect("file archive"));
    let check = move |buf: &AlignedBuf, n: usize, t: &Arc<MountTask>| -> Result<()> {
        let end = range.skip as usize + b.stored_len as usize;
        if n < end {
            return Err(orr_pack::Error::Truncated {
                need: range.offset + end as u64,
                have: range.offset + n as u64,
            }
            .into());
        }
        let a = t.archive();
        decoded(a, &b, &buf.as_slice()[range.skip as usize..end], |bytes| {
            then(t, bytes)
        })
    };
    if *task.ring.get().expect("read path chosen") {
        let t = Arc::clone(task);
        let read = Read {
            file,
            offset: range.offset,
            len: range.len as usize,
            done: Box::new(move |r| match r {
                Ok((buf, n)) => t.spawn(move |t| t.step(|| check(&buf, n, t))),
                Err(e) => {
                    t.fail(e.into());
                    t.end();
                }
            }),
        };
        if let Some(l) = task.loading.upgrade() {
            l.send(Msg::Read(read));
        }
    } else {
        task.spawn(move |t| {
            t.step(|| {
                let mut buf = AlignedBuf::new(range.len as usize);
                let n = read_positioned(&file, range.offset, &mut buf)?;
                check(&buf, n, t)
            });
        });
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

/// Blob 0, then every page and node page of a mesh, into one block.
fn place_mesh(t: &Arc<MountTask>, entry: usize, id: AssetId, blob0: &[u8]) -> Result<()> {
    let info = MeshInfo::parse(blob0)?;
    let header = info.header;
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
    let node_at: Vec<u64> = info
        .node_pages
        .iter()
        .map(|p| place(u64::from(p.node_count) * size_of::<orr_pack::mesh::Node>() as u64))
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
            info: block.gpu,
            pages: vec![0; info.pages.len()],
            node_pages: vec![0; info.node_pages.len()],
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
        let at = node_at[n];
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
    let block = t.block((user_at + user_len).max(16), "instance page")?;
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
    t.placed.lock().unwrap().pages.push(PlacedPage { id, page });
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
        self.point.is_none_or(|(kind, point)| {
            device
                .queue(kind)
                .is_some_and(|q| q.completed() >= point.value())
        })
    }
}

struct ResolvedMesh {
    id: AssetId,
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
    archive: Arc<Archive>,
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

fn resolve(archive: Arc<Archive>, p: Placed, sampler: SamplerIndex) -> Result<Resolved> {
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
        archive,
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
fn submit_copies(
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
        .and_then(|()| resolve(Arc::clone(task.archive()), placed, task.sampler))
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
    _archive: Arc<Archive>,
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
    fn put<T: bytemuck::Pod>(&mut self, dst: GpuPtr<T>, values: &[T]) -> Result<()> {
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

fn capacity(what: &'static str, capacity: u32) -> Error {
    Error::Capacity { what, capacity }
}

impl Mounts {
    pub(crate) fn new(
        device: &Device,
        sizes: TableSizes,
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
        })
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
            Ok(r) => match self.place(r, w) {
                Ok(r) => Entry::Ready(Box::new(r)),
                Err(e) => Entry::Failed(Arc::new(e)),
            },
        };
        self.entries[i] = Some(entry);
    }

    fn place(&mut self, r: Resolved, w: &mut Apply<'_>) -> Result<Resident> {
        let mut res = Resident {
            models: HashMap::new(),
            textures: r.textures.iter().map(|t| (t.id, t.index)).collect(),
            materials: HashMap::new(),
            pages: HashMap::new(),
            meshes: Vec::new(),
            page_ranges: Vec::new(),
            material_ranges: Vec::new(),
            cluster_ranges: r.cluster_ranges,
            _archive: r.archive,
            _blocks: r.blocks,
            _accel: r.accel,
            placed_textures: r.textures,
        };
        let result = self.fill(&mut res, r.materials, r.meshes, r.models, r.pages, w);
        match result {
            Ok(()) => Ok(res),
            Err(e) => {
                for id in res.models.values() {
                    w.emission[id.0 as usize] = None;
                }
                self.release(res, w.device, w.materials);
                Err(e)
            }
        }
    }

    fn fill(
        &mut self,
        res: &mut Resident,
        materials: Vec<(AssetId, MaterialGpu)>,
        meshes: Vec<ResolvedMesh>,
        models: Vec<ResolvedModel>,
        pages: Vec<PlacedPage>,
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
