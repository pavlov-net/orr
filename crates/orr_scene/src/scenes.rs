//! `Scenes`, the frame and its fixed order.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use orr_gpu::{
    Allocation, ClearColor, ColorAttachment, ColorLoad, CommandList, ComputePipeline,
    DEFAULT_ENVIRONMENT, DEFAULT_MEDIUM, Device, GpuAddress, GpuPtr, Hazards, KernelSet,
    LatencyMarker, LowLatency, MemoryKind, PassResult, QueueKind, Rendering, SamplerDesc,
    SamplerIndex, ShaderModule, SpecData, Stages, StorageImageIndex, TextureIndex, TimelinePoint,
};
use orr_math::{DVec3, Vec3};
use orr_pack::AssetId;
use orr_shaders::PathEntry;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::accel::{AccelKernels, AccelLimits, AccelShared};
use crate::arena::{ArenaReport, ArenaRequest, ViewArena, ViewStep};
use crate::decals::{self, build_bins};
use crate::detail::{Detail, Sample};
use crate::features::SceneFeatures;
use crate::loading::{self, Loading, Msg};
use crate::material::{MaterialId, MaterialTable, MaterialWriter};
use crate::medium_proxy::{ProxyBlas, VolumeRef};
use crate::memory::{BestFit, PreprocessArenas, UploadWindow};
use crate::mount::{
    Apply, ClipRef, InstancePage, LoadState, MountId, MountTask, Mounts, PackSource, PageId,
    SkeletonRef, TableSizes,
};
use crate::neural::{NeuralTextures, Precision};
use crate::pipelines::{PipelineId, Pipelines, ReloadReport};
use crate::producer::{self, Job, ProducerKernels, ProducerStats, ProducerUpdate, Readback};
use crate::ptlas::{FrameInputs, PtlasPipelines, RtStats};
use crate::raster::{self, ListSizes, RasterPipelines, TargetSizes, ViewLists, ViewTargets};
#[cfg(feature = "hot-reload")]
use crate::reload::Reloader;
use crate::residency::{RequestSource, ResidencyStats, response_period};
use crate::scene::{GroupWriter, ModelCpu, ModelId, Scene, SceneDesc, SceneId, SceneWriter};
use crate::shaders::transparency::CompositeRoot;
use crate::shaders::{animation, emissive, instance_page, page_motion, world as world_kernel};
use crate::skin::{SkinFrame, SkinKernels, SkinStats};
use crate::subd::ModelCage;
use crate::tessellate::{self, Patterns, SeenBy, TessKernels, TessellationStats};
use crate::time::FrameTime;
use crate::transparency::TransparencyPass;
use crate::traverse::{CutInputs, CutPipelines, CutSizes, CutStats, Cuts, PageFeedback};
use crate::upload::{Record, scatter};
use crate::view::{RasterView, ViewDesc, ViewHistory, ViewHistoryId, view_constants};
use crate::world::{WorldPos, needs_rebase, origin_for};
use crate::{
    BlobId, BlobTarget, EditDesc, EditId, EditRef, Error, ExposureState, ProducerBatch, Result,
    SceneRoot, TextureStreamGpu,
};

/// The size of a pool: bytes, or a share of the device-local heap its memory is on, which
/// [`Scenes::new`] resolves once from the heap's size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolSize {
    Bytes(u64),
    /// Percent of the heap.
    HeapPercent(u32),
}

impl PoolSize {
    /// Its bytes on a heap of `heap` bytes.
    pub fn bytes(self, heap: u64) -> u64 {
        match self {
            Self::Bytes(b) => b,
            Self::HeapPercent(p) => (u128::from(heap) * u128::from(p) / 100) as u64,
        }
    }
}

/// The bytes of the pools [`PoolSize`] gives, on the device.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PoolBytes {
    pub(crate) mesh: u64,
    pub(crate) textures: u64,
}

impl PoolBytes {
    fn new(device: &Device, b: &Budgets) -> Self {
        let heap = device.memory_report().heap_of(MemoryKind::DeviceLocal).size;
        Self {
            mesh: b.mesh_page_pool.bytes(heap),
            textures: b.texture_pool.bytes(heap),
        }
    }
}

/// Pool sizes and per-frame limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budgets {
    /// Bytes per upload block; a window that outgrows its blocks takes another.
    pub upload_block: u64,
    /// Entries in the global material table.
    pub materials: u32,
    /// Entries in the global model table: [`ModelId`]s of mounted packs.
    pub models: u32,
    /// Entries in the global mesh table.
    pub meshes: u32,
    /// Section materials of every resident model.
    pub model_materials: u32,
    /// Entries of the mesh page table: per mounted mesh one per page and node page, which
    /// hold their addresses, and one per finer cached cut of a streaming mesh, and per edit one
    /// per page. A scene's traversal reports what it reads and wants by entry, a bit each.
    pub mesh_pages: u32,
    /// Visible clusters one view's list holds at first. A list grows a few frames after a
    /// frame wanted more than it holds, and gives memory back after a long run of frames that
    /// want far less. It holds 2^25 at most, the clusters a visibility buffer word can name.
    pub initial_visible_clusters: u32,
    /// Items of up to 32 clusters a scene's raster candidate cut holds at first; it grows like
    /// the visible clusters. The work a view's first culling pass leaves to its second holds
    /// as many as its scene's candidate cut.
    pub initial_candidate_items: u32,
    /// Hierarchy nodes each of a scene's traversal queues holds at first, the nodes one pass
    /// visits; they grow like the visible clusters.
    pub initial_traversal_nodes: u32,
    /// Mesh pages, hierarchy pages and finer cached cuts a scene's traversal asks the loading
    /// side for in a frame, at most: those of the largest projected errors. The others wait
    /// for a later frame.
    pub traversal_requests: u32,
    /// CLAS the cluster BLAS a scene builds in a frame for the ray-tracing cuts its traversal
    /// gathers take at first, and those BLAS. Their tables and arenas grow to what completed
    /// frames wanted within `rt_cut_memory`. While a frame wants more than they hold, the
    /// copies nearest the focus build theirs and the farther ones trace cached cuts, the same
    /// copies every frame of a still view. Above 0.
    pub initial_rt_cut_clusters: u32,
    pub initial_rt_cut_blas: u32,
    /// Device memory a scene's tables of built ray-tracing cuts and their BLAS arenas grow
    /// to at most; the first sizes take what they take. Past it the gathered cuts coarsen
    /// evenly until their CLAS fit ([`DetailStats::rt_scale`]), and builds past what fits
    /// leave the farthest copies their cached cuts.
    pub rt_cut_memory: PoolSize,
    /// Triangles of edited clusters whose CLAS rebuild in one frame, at most; a first edit
    /// past it still rebuilds alone.
    pub edit_triangles: u32,
    /// Triangles one CLAS build on the loading queue takes, at most. On 621.01 an implicit
    /// build of about 413k triangles in 3,875 clusters of up to 256 vertices faults the
    /// device while 270k builds, so the default stays under that.
    pub stream_clas_triangles: u32,
    /// Micromaps one loading-queue submit builds, at most.
    pub micromap_pages: u32,
    /// Per-frame PTLAS work beyond mandatory work, in rebuilt instances: the full instance
    /// count of every partition a write touches plus one per update.
    pub ptlas_rebuilt: u32,
    /// Resident mesh pages, the CLAS and micromaps streamed pages build, which count at an
    /// estimate from their triangles until a page's first load has built them, and resident
    /// hierarchy pages, which a streamed page's groups need beside it. The CLAS pool's address
    /// range is as large, up to the 512 GiB its 32-bit unit indices name, and bound in pages
    /// as they need. While the frames want more than the pool's budget holds, the error
    /// thresholds geometry is chosen by coarsen evenly until they want what it holds
    /// ([`DetailStats`]).
    pub mesh_page_pool: PoolSize,
    /// Bytes of `mesh_page_pool` per entry of the resident cluster table, which holds the
    /// clusters of resident mesh pages and of edits' own pages, 24 B each. A mount whose
    /// pinned pages' clusters find no room fails, and a streamed page that finds none loads
    /// later while the mesh page budget shrinks. A cooked scene of 101.8 M clusters holds
    /// 1,376 B of resident page per cluster, before a streamed page's CLAS. Above 0.
    pub resident_cluster_bytes: u64,
    /// Bytes of pages a DAG mesh keeps resident from mount; the pages of a larger one past
    /// its pinned coarsest pages stream on the traversal's requests.
    pub mesh_resident_max: u64,
    /// Resident textures: whole textures, streamed textures' mip tails and tiles.
    pub texture_pool: PoolSize,
    /// Bytes of resident instance pages.
    pub instance_page_pool: u64,
    /// Bytes of requested blobs loading or landed and not released.
    pub user_blob_pool: u64,
    /// Records of raises of a texture feedback cell's wanted mip a frame keeps at first; the
    /// list grows like the visible clusters. While a frame raises more, each frame keeps a
    /// different run of its raises, so every cell is heard within a few frames.
    pub initial_texture_feedback_records: u32,
    /// Bytes loading at once: of streamed units and requested blobs from the step that starts
    /// their reads to the one their GPU work lands in, which hold staging and read buffers that
    /// long, and mounts' reads: their stored bytes, and the decoded bytes of those that stage
    /// writes, from the read's start until it is placed, then the bytes it staged until their
    /// copies complete. Loads start while fewer are in flight, so the last one started may
    /// pass it. Each [`RequestSource`], the requested blobs and the mounts share it equally
    /// while they all have loads waiting; a share one leaves unused goes to the others. A mount
    /// larger than it loads in pieces. Twice this sizes the arena archive reads and streamed
    /// units' staging take, which the system's ring registers and, where the device imports
    /// host memory, the GPU reads in place: room for the limit's worth of streamed units'
    /// staging and as much again of the stored bytes archive reads take. Reads and staging that
    /// find no room take memory of their own. Above 0.
    pub in_flight_bytes: u64,
    /// Bytes per block of the stream pool.
    pub stream_block: u64,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            upload_block: 16 << 20,
            materials: 65_536,
            models: 65_536,
            meshes: 65_536,
            model_materials: 262_144,
            mesh_pages: 1 << 20,
            initial_visible_clusters: 1 << 20,
            initial_candidate_items: 1 << 18,
            initial_traversal_nodes: 1 << 18,
            traversal_requests: 4096,
            initial_rt_cut_clusters: 1 << 18,
            initial_rt_cut_blas: 1024,
            rt_cut_memory: PoolSize::HeapPercent(10),
            edit_triangles: 1 << 18,
            stream_clas_triangles: 1 << 18,
            micromap_pages: 64,
            ptlas_rebuilt: 250_000,
            mesh_page_pool: PoolSize::HeapPercent(40),
            resident_cluster_bytes: 1024,
            mesh_resident_max: 1 << 20,
            texture_pool: PoolSize::HeapPercent(30),
            instance_page_pool: 256 << 20,
            user_blob_pool: 256 << 20,
            initial_texture_feedback_records: 1 << 14,
            in_flight_bytes: 256 << 20,
            stream_block: 16 << 20,
        }
    }
}

/// Farther models a representation chain holds: the length of `ModelGpu::chain`.
const MAX_CHAIN: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct RendererDesc {
    pub budgets: Budgets,
    /// Frames the CPU records ahead of the GPU.
    pub frames_in_flight: u32,
    /// Consumer-linked kernels.
    pub kernels: &'static KernelSet,
    /// Which entry of the path tracing kernels views run.
    pub path_entry: PathEntry,
    /// The first view history looks in the pipeline caches for the lighting kernels that every
    /// view runs, on the calling thread, so a warm start waits only for the path pipelines its
    /// first view asks for, which come from the caches as jobs. Off, the calling thread never
    /// touches the driver for them: they always start as jobs too.
    pub try_cached_paths: bool,
    /// World cells are 2^cell_log2 m on a side.
    pub cell_log2: u32,
    /// Archive reads are positioned reads on workers even where the system's completion
    /// ring (IoRing, io_uring) is available.
    pub positioned_reads: bool,
    /// Raster LOD: the largest cluster error drawn, in render pixels at the scene focus, times
    /// [`DetailStats::memory_scale`].
    pub raster_error_pixels: f32,
    /// Ray-tracing LOD: the largest cluster error traced, in render pixels at the scene focus,
    /// times [`DetailStats::memory_scale`] and, in a gathered cut, its scene's
    /// [`DetailStats::rt_scale`].
    pub rt_error_pixels: f32,
    /// Copies no view drew last frame trace at this many times `rt_error_pixels`.
    pub culled_error_scale: f32,
    /// Copies whose bounding sphere projects to a radius under this many render pixels at the
    /// focus trace their model's merged cut: the finest cached cut any of them needs. 0 turns
    /// merging off.
    pub rt_merge_pixels: f32,
    /// Clusters whose screen rect's longer side is at most this many render pixels raster
    /// in compute, unless they cross the near plane; others take the mesh-shader path. 0
    /// turns the software raster off; values above 128 act as 128.
    pub software_raster_pixels: f32,
    /// A view's transient targets whose spans in the frame order do not overlap share
    /// memory ([`ViewArena`]).
    pub alias_view_targets: bool,
    /// The material resolve runs its surface bins as one device-generated command that
    /// skips empty bins, on devices with `Caps::generated_commands`; off, or without them,
    /// it records one indirect dispatch per bin.
    pub generated_resolve: bool,
    /// Tessellation: the longest generated edge, in render pixels at the scene focus.
    pub dice_pixels: f32,
    /// Whether textures that a pack's neural texture sets stand in for decode from the sets.
    pub neural_textures: NeuralTextures,
}

impl Default for RendererDesc {
    fn default() -> Self {
        Self {
            budgets: Budgets::default(),
            frames_in_flight: 2,
            kernels: &KernelSet::NONE,
            path_entry: PathEntry::Fastest,
            try_cached_paths: true,
            cell_log2: 8,
            positioned_reads: false,
            raster_error_pixels: 1.0,
            rt_error_pixels: 1.0,
            culled_error_scale: 2.0,
            rt_merge_pixels: 32.0,
            software_raster_pixels: 32.0,
            alias_view_targets: true,
            generated_resolve: true,
            dice_pixels: 2.0,
            neural_textures: NeuralTextures::Fastest,
        }
    }
}

/// What one frame renders.
#[derive(Clone, Copy)]
pub struct FrameDesc<'a> {
    pub time: FrameTime,
    /// Scenes updated this frame, in order. Others keep their tables and record nothing.
    pub scenes: &'a [SceneId],
    /// The display's Reflex handle while its low latency is on: `begin_frame` sleeps on it
    /// and the frame sets its markers under [`Frame::id`].
    pub low_latency: Option<&'a LowLatency>,
}

/// The latency markers a consumer sets, on the render thread during the write phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsumerMarker {
    SimulationStart,
    SimulationEnd,
    InputSample,
}

/// Figures of the last frame whose GPU work has completed.
#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    /// The frame these figures describe; 0 before any has completed.
    pub frame: u64,
    /// GPU passes of the frame's submit in order, consumer passes included.
    pub gpu: Vec<PassResult>,
    /// Render-thread time per step: `begin_frame`, `write` (the consumer's write phase),
    /// `update_scenes`, `raster_view`, `end`.
    pub cpu: Vec<(&'static str, Duration)>,
    pub upload_bytes: u64,
    /// Most upload-window bytes any frame has used.
    pub upload_high_water: u64,
    pub scatter_records: u32,
    /// Words written by the scatter.
    pub scatter_words: u64,
    pub scatter_dispatches: u32,
    pub scenes_updated: u32,
    /// PTLAS work of the frame's scenes.
    pub rt: RtStats,
    /// Producer entries of the frame's scenes: emitted against capacity, drops, builds.
    pub producers: ProducerStats,
    /// The DAG traversals of the frame's scenes.
    pub cuts: CutStats,
    /// The skinning of the frame's scenes.
    pub skin: SkinStats,
    /// The tessellated producer entries of the frame's scenes.
    pub tessellation: TessellationStats,
    pub detail: DetailStats,
}

/// The detail a frame ran at: the scales of the error thresholds geometry is chosen by
/// ([`RendererDesc::raster_error_pixels`], [`RendererDesc::rt_error_pixels`]) that keep what
/// the frames want within the mesh page pool and the gathers' capacity.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DetailStats {
    /// The scale of both thresholds, from the mesh page pool's demand; 1 at the configured
    /// detail.
    pub memory_scale: f32,
    /// The largest further scale of the ray-tracing threshold of gathered cuts in its scenes,
    /// from the CLAS they wanted against what their tables hold once those can grow no
    /// further ([`Budgets::rt_cut_memory`]).
    pub rt_scale: f32,
    /// What the mesh page pool was asked for at the frame's detail: its pinned, locked and
    /// loading pages, and those the frame read or requested with what they depend on, whether
    /// or not the budget took them; and the pool's budget then. 0 when the frame updated no
    /// scene.
    pub mesh_demand: u64,
    pub mesh_budget: u64,
}

/// Scene compute: a listed scene's turn in `update_scenes`, after its uploads have landed.
pub struct SceneCompute<'a, 'q> {
    pub cmd: &'a mut CommandList<'q>,
    pub scene: SceneId,
    pub root: GpuPtr<SceneRoot>,
    /// The scene's producer batches, indexed by `ProducerId`.
    pub batches: GpuPtr<ProducerBatch>,
    pub time: FrameTime,
    /// The scene focus and the render pixels per radian this update chooses geometry for.
    pub focus: Option<(DVec3, f32)>,
    device: &'a Device,
    window: &'a UploadWindow,
    pipelines: &'a Pipelines,
}

impl<'a> SceneCompute<'a, '_> {
    /// The current pipeline of `id`.
    pub fn pipeline(&self, id: PipelineId) -> &'a ComputePipeline {
        self.pipelines.get(id)
    }

    /// `value` in this frame's upload window, 16-byte aligned; the GPU can read it until
    /// the frame completes.
    pub fn put<T: bytemuck::Pod>(&self, value: &T) -> Result<GpuPtr<T>> {
        self.window.put(self.device, value)
    }

    /// `root` in this frame's upload window at an address a dispatch can take as its root.
    pub fn put_root<T: bytemuck::Pod>(&self, root: &T) -> Result<GpuPtr<T>> {
        self.window.put_root(self.device, root)
    }
}

struct InFlight {
    id: u64,
    /// The frame's submits in order; the last completes after the others.
    points: Vec<TimelinePoint>,
    stats: FrameStats,
    /// Scenes whose PTLAS the frame built.
    traced: Vec<SceneId>,
    /// Scenes the frame updated.
    updated: Vec<SceneId>,
}

struct Core {
    fif: u32,
    cell_log2: u32,
    budgets: Budgets,
    raster_error_pixels: f32,
    rt_error_pixels: f32,
    culled_error_scale: f32,
    /// The scale of both thresholds for the mesh page pool ([`DetailStats::memory_scale`]).
    memory: Detail,
    rt_merge_pixels: f32,
    software_raster_pixels: f32,
    alias_view_targets: bool,
    raster: RasterPipelines,
    cuts: CutPipelines,
    materials: MaterialTable,
    /// Sections and animated parts by `ModelId`.
    model_cpu: Vec<ModelCpu>,
    /// Subdivision cages by `ModelId`.
    cages: Vec<Option<ModelCage>>,
    /// The farther models of every model with a representation chain.
    chains: HashMap<ModelId, Vec<ModelId>>,
    /// A model has had a chain: a copy that fades between two steps is written with every
    /// triangle a candidate, and stays so until its instance is next written.
    chained: bool,
    /// Records outside any scene: materials, view-history initialization.
    global: Mutex<Vec<Record>>,
    scenes: Vec<Option<Scene>>,
    histories: Vec<Option<ViewHistory>>,
    windows: Vec<UploadWindow>,
    preprocess: PreprocessArenas,
    seq: AtomicU64,
    next_frame: u64,
    completed_frame: u64,
    in_flight: VecDeque<InFlight>,
    /// The last output list submitted on the Compute queue.
    output: Option<TimelinePoint>,
    stats: FrameStats,
    high_water: u64,
    mounts: Mounts,
    accel: Arc<AccelShared>,
    ptlas: PtlasPipelines,
    /// Scratch of every scene's PTLAS build; the builds of a frame run one after another.
    ptlas_scratch: Mutex<Option<Allocation<u8>>>,
    /// Made at the first traced frame with a medium proxy.
    proxy_blas: Option<ProxyBlas>,
    producers: ProducerKernels,
    skin: SkinKernels,
    /// Scratch of producer builds, grown to the largest frame's.
    producer_scratch: Mutex<Option<Allocation<u8>>>,
    patterns: Patterns,
    tess: TessKernels,
    dice_pixels: f32,
}

/// Global scene state, mounted packs and every resident scene. Render thread only.
pub struct Scenes {
    device: Device,
    kernels: &'static KernelSet,
    pipelines: Pipelines,
    #[cfg(feature = "hot-reload")]
    reloader: Reloader,
    core: Core,
    loading: Arc<Loading>,
    loading_thread: Option<JoinHandle<()>>,
    /// The sampler of neural sets' latents.
    sampler: SamplerIndex,
    /// The precision mounts decode neural texture sets at; none when they sample BC.
    neural: Option<Precision>,
}

/// Puts `value` in the first free entry of `slots` and returns its index.
fn insert<T>(slots: &mut Vec<Option<T>>, value: T) -> u32 {
    match slots.iter().position(Option::is_none) {
        Some(i) => {
            slots[i] = Some(value);
            i as u32
        }
        None => {
            slots.push(Some(value));
            (slots.len() - 1) as u32
        }
    }
}

impl Scenes {
    /// Compiles orr_scene's pipelines on the calling thread, from `desc.kernels`' module
    /// wherever it holds one for an orr_scene kernel, and starts the loading thread. With
    /// the `hot-reload` feature it watches the sources of every module it creates a
    /// pipeline from.
    pub fn new(device: &Device, desc: &RendererDesc) -> Result<Self> {
        assert!(desc.frames_in_flight >= 1, "frames_in_flight is at least 1");
        assert!(desc.cell_log2 <= 16, "cells of 2^{} m", desc.cell_log2);
        let b = &desc.budgets;
        assert!(b.in_flight_bytes > 0, "in_flight_bytes is above 0");
        assert!(
            b.resident_cluster_bytes > 0,
            "resident_cluster_bytes is above 0"
        );
        assert!(
            b.initial_rt_cut_clusters > 0 && b.initial_rt_cut_blas > 0,
            "initial_rt_cut_clusters and initial_rt_cut_blas are above 0"
        );
        let pools = PoolBytes::new(device, b);
        // Cluster ids are 32 bits.
        let clusters = u32::try_from(pools.mesh / b.resident_cluster_bytes).unwrap_or(u32::MAX);
        let accel = Arc::new(AccelShared {
            kernels: AccelKernels::new(device, desc.kernels)?,
            limits: AccelLimits {
                clas_triangles: b.stream_clas_triangles,
                micromaps: b.micromap_pages,
                mesh_resident_max: b.mesh_resident_max,
            },
            cluster_ids: Mutex::new(BestFit::new(clusters)),
            cluster_capacity: clusters,
        });
        let materials = MaterialTable::new(device, b.materials)?;
        let mounts = Mounts::new(
            device,
            TableSizes {
                models: b.models,
                meshes: b.meshes,
                model_materials: b.model_materials,
                page_table: b.mesh_pages,
            },
            b,
            pools,
            &accel,
            materials.streams.gpu(),
            desc.frames_in_flight,
        )?;
        let sampler = device.sampler(&SamplerDesc {
            max_anisotropy: 16.0,
            ..SamplerDesc::default()
        })?;
        let mut pipelines = Pipelines::new(device, desc.kernels)?;
        let generated = desc.generated_resolve && device.caps().generated_commands;
        let raster = RasterPipelines::new(device, desc.kernels, &mut pipelines, generated)?;
        let ptlas = PtlasPipelines::new(device, desc.kernels, &mut pipelines)?;
        let producers = ProducerKernels::new(device, desc.kernels, &mut pipelines)?;
        let skin = SkinKernels::new(device, desc.kernels, &mut pipelines)?;
        let cuts = CutPipelines::new(device, desc.kernels, &mut pipelines)?;
        let tess = TessKernels::new(device, desc.kernels, &mut pipelines)?;
        let patterns = Patterns::new(device)?;
        #[cfg(feature = "hot-reload")]
        let mut reloader = Reloader::new();
        #[cfg(feature = "hot-reload")]
        for r in &pipelines.recipes {
            reloader.watch(r.module);
        }
        let streams = materials.streams.gpu().address().0;
        let global = vec![Record::fill(0, streams, 2 * device.image_slots(), 0)];
        // Last, so no failure after it leaves the threads running with no one to stop them.
        let in_flight = Arc::clone(mounts.streaming.in_flight());
        let loading = Arc::new(Loading::new(
            desc.positioned_reads,
            2 * b.in_flight_bytes,
            Some(device),
            in_flight,
        )?);
        let thread = loading::spawn(&loading, device)?;
        Ok(Self {
            loading,
            loading_thread: Some(thread),
            sampler,
            neural: crate::neural::precision(device, desc.neural_textures),
            device: device.clone(),
            kernels: desc.kernels,
            pipelines,
            #[cfg(feature = "hot-reload")]
            reloader,
            core: Core {
                fif: desc.frames_in_flight,
                cell_log2: desc.cell_log2,
                budgets: desc.budgets,
                raster_error_pixels: desc.raster_error_pixels,
                rt_error_pixels: desc.rt_error_pixels,
                culled_error_scale: desc.culled_error_scale,
                memory: Detail::new(),
                rt_merge_pixels: desc.rt_merge_pixels,
                software_raster_pixels: desc.software_raster_pixels,
                alias_view_targets: desc.alias_view_targets,
                raster,
                cuts,
                materials,
                model_cpu: Vec::new(),
                cages: Vec::new(),
                chains: HashMap::new(),
                chained: false,
                global: Mutex::new(global),
                scenes: Vec::new(),
                histories: Vec::new(),
                windows: (0..desc.frames_in_flight)
                    .map(|_| UploadWindow::new(desc.budgets.upload_block))
                    .collect(),
                preprocess: PreprocessArenas::new(desc.frames_in_flight),
                seq: AtomicU64::new(1),
                next_frame: 1,
                completed_frame: 0,
                in_flight: VecDeque::new(),
                output: None,
                stats: FrameStats::default(),
                high_water: 0,
                mounts,
                accel,
                ptlas,
                ptlas_scratch: Mutex::new(None),
                proxy_blas: None,
                producers,
                skin,
                producer_scratch: Mutex::new(None),
                patterns,
                tess,
                dice_pixels: desc.dice_pixels,
            },
        })
    }

    /// A compute pipeline of `module`'s `entry`, created on the calling thread from the
    /// renderer's kernel set where it holds a module of the same kernel file. With the
    /// `hot-reload` feature, editing the kernel's sources replaces the pipeline at a later
    /// step 1 ([`Frame::reloads`]); look it up each frame with [`Frame::pipeline`] or
    /// [`SceneCompute::pipeline`].
    pub fn compute_pipeline(
        &mut self,
        module: &'static ShaderModule,
        entry: &str,
        spec: SpecData<'_>,
    ) -> Result<PipelineId> {
        let id = self
            .pipelines
            .create(&self.device, self.kernels, module, entry, spec)?;
        #[cfg(feature = "hot-reload")]
        self.reloader
            .watch(self.pipelines.recipes.last().expect("just created").module);
        Ok(id)
    }

    /// The loading side's job queue, for the threads the consumer lends to it.
    pub fn loading(&self) -> Arc<Loading> {
        Arc::clone(&self.loading)
    }

    /// Mounts a pack. Returns at once: its index and every asset load on the loading side,
    /// and lookups fail with [`Error::NotReady`] until [`Scenes::mount_state`] is `Ready`.
    pub fn mount(&mut self, source: PackSource) -> MountId {
        let (id, cancel) = self.core.mounts.add();
        let task = MountTask::new(
            id,
            &self.loading,
            &self.device,
            self.sampler,
            self.neural,
            cancel,
            &self.core.accel,
        );
        self.loading.send(Msg::Mount(Arc::new(task), source));
        id
    }

    /// The progress of mount `m`; `Failed` with [`Error::UnknownMount`] once unmounted.
    pub fn mount_state(&self, m: MountId) -> LoadState {
        self.core.mounts.state(m)
    }

    /// Unmounts a pack. A pending mount stops loading; a ready one's memory is freed once
    /// every frame submitted so far has completed. Copies of its models and their edits in every
    /// scene are gone from the next frame, placed by hand, on pages or as subdivision copies, and
    /// their [`InstanceRange`]s and producer entries stay the consumer's.
    pub fn unmount(&mut self, m: MountId) {
        let frame = self.core.next_frame - 1;
        let core = &mut self.core;
        let gone: FxHashSet<u32> = core.mounts.models_of(m).iter().map(|id| id.0).collect();
        let holds = |page: PageId| {
            let protos = core.mounts.page_prototypes(page);
            protos.is_ok_and(|p| p.iter().any(|q| gone.contains(&q.model.0)))
        };
        for scene in core.scenes.iter().flatten() {
            scene.empty_copies(&gone, holds, frame, &core.seq);
        }
        for page in core.mounts.pages_of(m) {
            core.land_page(page, 0);
        }
        let gone = core.mounts.models_of(m);
        core.mounts.unmount(m, frame, &mut core.cages);
        // Chains that name a model of the mount end; their copies take their model again.
        let ended: Vec<ModelId> = core
            .chains
            .iter()
            .filter(|(model, steps)| gone.contains(model) || steps.iter().any(|s| gone.contains(s)))
            .map(|(&model, _)| model)
            .collect();
        for model in ended {
            core.chains.remove(&model);
            if !gone.contains(&model) {
                core.write_chain(model, &[]);
                // New copies take entries for the model alone; live ones keep theirs.
                let m = &mut core.model_cpu[model.0 as usize];
                m.emissive_stride = m.emissive_sections;
            }
        }
    }

    /// Starts an edit of a mesh: a new model the consumer's kernels change at the scene
    /// compute point (see [`EditRef`]). Its pages read on the loading side, and
    /// [`Scenes::edit`] succeeds once [`Scenes::edit_state`] is `Ready`.
    pub fn create_edit(&mut self, desc: &EditDesc<'_>) -> Result<EditId> {
        self.core
            .mounts
            .create_edit(desc, &self.loading, &self.device)
    }

    /// The progress of an edit; `Failed` with [`Error::UnknownEdit`] once destroyed.
    pub fn edit_state(&self, id: EditId) -> LoadState {
        self.core.mounts.edit_state(id)
    }

    /// What kernels and instances need of a ready edit.
    pub fn edit(&self, id: EditId) -> Result<EditRef> {
        self.core.mounts.edit(id)
    }

    /// Ends an edit. Its copies in every scene are gone from the next frame, and their
    /// [`InstanceRange`]s stay the consumer's to remove; its memory is freed once every frame
    /// submitted so far has completed.
    pub fn destroy_edit(&mut self, id: EditId) {
        let frame = self.core.next_frame - 1;
        let core = &mut self.core;
        if let Some(model) = core.mounts.edit_model(id) {
            let gone = FxHashSet::from_iter([model.0]);
            for scene in core.scenes.iter().flatten() {
                scene.empty_copies(&gone, |_| false, frame, &core.seq);
            }
        }
        core.mounts.destroy_edit(id, frame, &mut core.cages);
    }

    /// Gives every copy of `model` up to 3 farther models, each with the distance from the
    /// focus in m at which copies switch to it, ascending; empty: the model at every
    /// distance. Copies take their step from the next frame on, and a copy that changes step
    /// cross-fades between the two over a few frames. Steps past the model itself draw and
    /// trace rigid, with their own materials, every section shown and no pose. A chain ends
    /// when a pack holding one of its models is unmounted. Each copy holds an emissive list
    /// entry per emissive section of its chain's most emissive model; a chain that needs more
    /// gives every copy of the model in every scene more, on pages too, and fails with
    /// [`Error::Capacity`], changing nothing, where a scene's list has no room.
    pub fn set_chain(&mut self, model: ModelId, chain: &[(ModelId, f32)]) -> Result<()> {
        let invalid = |reason| Err(Error::InvalidChain { model, reason });
        if !self.core.mounts.is_live(model) {
            return invalid("the model is not resident");
        }
        if chain.len() > MAX_CHAIN {
            return invalid("a chain holds at most 3 farther models");
        }
        let mut last = 0.0;
        for &(step, distance) in chain {
            if !self.core.mounts.is_live(step) {
                return invalid("a farther model is not resident");
            }
            let skinned = self
                .core
                .model_cpu
                .get(step.0 as usize)
                .is_some_and(|m| m.skin.is_some());
            if skinned {
                return invalid("a farther model is skinned");
            }
            let edit = self
                .core
                .model_cpu
                .get(step.0 as usize)
                .is_some_and(|m| m.edit);
            if edit {
                return invalid("a farther model is an edit");
            }
            if !(distance.is_finite() && distance > last) {
                return invalid("switch distances must be finite, positive and ascending");
            }
            last = distance;
        }
        let core = &mut self.core;
        // Every copy holds an emissive list entry per section rank of its most emissive step.
        let ranks = |m: ModelId| {
            core.model_cpu
                .get(m.0 as usize)
                .map_or(0, |c| c.emissive_sections)
        };
        let stride = chain
            .iter()
            .map(|&(m, _)| ranks(m))
            .fold(ranks(model), u32::max);
        let held = core
            .model_cpu
            .get(model.0 as usize)
            .map_or(0, |c| c.emissive_stride);
        if stride > held {
            let frame = core.next_frame - 1;
            // A page's other prototypes keep what they hold.
            let page_strides = |page: PageId| {
                let protos = core.mounts.page_prototypes(page).ok()?;
                protos.iter().any(|p| p.model == model).then(|| {
                    protos
                        .iter()
                        .map(|p| if p.model == model { stride } else { 0 })
                        .collect()
                })
            };
            // Every scene's ranges first, so a scene without room changes none.
            let mut growths = Vec::new();
            for scene in core.scenes.iter().flatten() {
                match scene.reserve_entries(model.0, stride, page_strides) {
                    Ok(g) => growths.push((scene, g)),
                    Err(e) => {
                        for (scene, g) in growths {
                            scene.release_entries(g);
                        }
                        return Err(e);
                    }
                }
            }
            for (scene, g) in growths {
                scene.grow_entries(g, frame, &core.seq);
            }
            core.model_cpu[model.0 as usize].emissive_stride = stride;
        }
        if chain.is_empty() {
            core.chains.remove(&model);
        } else {
            core.chains
                .insert(model, chain.iter().map(|&(m, _)| m).collect());
            core.chained = true;
        }
        core.write_chain(model, chain);
        Ok(())
    }

    /// The model `id` of mount `m`.
    pub fn model(&self, m: MountId, id: AssetId) -> Result<ModelId> {
        self.core.mounts.lookup(m, id, |r| &r.models)
    }

    /// The volume `id` of mount `m`, for medium proxies and kernels that sample it through
    /// `orr_pack.volume`.
    pub fn volume(&self, m: MountId, id: AssetId) -> Result<VolumeRef> {
        self.core.mounts.lookup(m, id, |r| &r.volumes)
    }

    /// The skeleton `id` of mount `m`, for pose kernels that fill skinned palettes.
    pub fn skeleton(&self, m: MountId, id: AssetId) -> Result<SkeletonRef> {
        self.core.mounts.lookup(m, id, |r| &r.skeletons)
    }

    /// The clip `id` of mount `m`, for pose kernels that sample it.
    pub fn clip(&self, m: MountId, id: AssetId) -> Result<ClipRef> {
        self.core.mounts.lookup(m, id, |r| &r.clips)
    }

    /// The instance page `id` of mount `m`.
    pub fn instance_page(&self, m: MountId, id: AssetId) -> Result<PageId> {
        self.core.mounts.lookup(m, id, |r| &r.pages)
    }

    /// The cooked texture `id` of mount `m`, for consumer data that samples it. A texture the
    /// pack's materials bind streams by tile: sample it through `texture_footprint` with
    /// [`Scenes::texture_streams`], which never reads a tile that is not resident. A texture
    /// a neural set stands in for reads its own texels, and decodes from the set in kernels
    /// that prepare it (`texture_prepare`).
    pub fn texture(&self, m: MountId, id: AssetId) -> Result<TextureIndex> {
        self.core.mounts.lookup(m, id, |r| &r.textures)
    }

    /// The cooked material `id` of mount `m`, for consumer data and [`MaterialWriter`] edits.
    pub fn material(&self, m: MountId, id: AssetId) -> Result<MaterialId> {
        self.core.mounts.lookup(m, id, |r| &r.materials)
    }

    /// Wants every streamed unit of `model` resident from the next step 1 on, `cell_distance`
    /// world cells from the focus. Requests last one step: call it every frame while the
    /// model should load or stay.
    pub fn prefetch(&mut self, model: ModelId, cell_distance: u32) {
        self.core.mounts.prefetch(model, cell_distance);
    }

    /// Wants every tile of mips `finest..` of the cooked texture `texture` of mount `m`
    /// resident from the next step 1 on. Requests last one step: call it every frame while
    /// the tiles should load or stay. Textures that do not stream are always resident.
    pub fn prefetch_texture(&mut self, m: MountId, texture: TextureIndex, finest: u32) {
        self.core
            .mounts
            .streaming
            .prefetch_texture(m, texture.0, finest);
    }

    /// Wants the records of instance page `id`, and on an animated page the rows of its start,
    /// resident from the next frame on. Requests last one frame.
    pub fn prefetch_page(&mut self, id: PageId, cell_distance: u32) {
        self.core.mounts.prefetch_page(id, cell_distance);
    }

    /// Wants the records of every mounted instance page meeting the sphere, and the streamed
    /// units of the models they place, resident from the next frame on, nearer first. Requests
    /// last one frame and rank after what the frames' traversals request.
    pub fn prefetch_region(&mut self, center: DVec3, radius: f32) {
        let log2 = self.core.cell_log2;
        self.core
            .mounts
            .prefetch_region(center, f64::from(radius), log2);
    }

    /// The stream table `MaterialGpu::texture_streams` points at, for kernels that sample
    /// looked-up textures through `texture_footprint`; it lives as long as `self`.
    pub fn texture_streams(&self) -> GpuPtr<GpuPtr<TextureStreamGpu>> {
        self.core.materials.streams.gpu()
    }

    /// Streams blob `blob` of asset `id` of mount `m` into `target` through the loading
    /// side, under the user blob budget. Poll [`Scenes::blob_state`]; the data is there for
    /// frames that begin after it reads `Ready`.
    pub fn request_blob(
        &mut self,
        m: MountId,
        id: AssetId,
        blob: u32,
        target: BlobTarget,
    ) -> Result<BlobId> {
        self.core.mounts.request_blob(m, id, blob, target)
    }

    /// The progress of a requested blob; `Failed` with [`Error::UnknownBlob`] once released.
    pub fn blob_state(&self, id: BlobId) -> LoadState {
        self.core.mounts.streaming.blob_state(id)
    }

    /// Returns a requested blob's bytes to the user blob budget; its target is the
    /// consumer's again once the blob is not loading.
    pub fn release_blob(&mut self, id: BlobId) {
        self.core.mounts.streaming.release_blob(id);
    }

    /// Residency figures of the latest step 1.
    pub fn residency(&self) -> &ResidencyStats {
        self.core.mounts.streaming.stats()
    }

    /// A resident instance page.
    pub fn page(&self, id: PageId) -> Option<&InstancePage> {
        self.core.mounts.page(id)
    }

    /// What placing instance page `id` takes of a scene's capacities.
    pub fn page_needs(&self, id: PageId) -> Result<crate::PageNeeds> {
        crate::scene::page_needs(&self.core.mounts, &self.core.model_cpu, id).map(|(n, ..)| n)
    }

    #[cfg(feature = "testing")]
    pub(crate) fn texture_object(&self, m: MountId, id: AssetId) -> Option<Arc<orr_gpu::Texture>> {
        self.core.mounts.texture_object(m, id)
    }

    #[cfg(feature = "testing")]
    pub(crate) fn tables(
        &self,
    ) -> (
        GpuPtr<crate::ModelGpu>,
        GpuPtr<crate::MeshGpu>,
        GpuPtr<crate::MaterialGpu>,
    ) {
        let m = &self.core.mounts;
        (
            m.models.gpu(),
            m.meshes.gpu(),
            self.core.materials.table.gpu(),
        )
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// A new scene with empty tables.
    pub fn create_scene(&mut self, desc: &SceneDesc) -> Result<SceneId> {
        let scene = Scene::new(&self.device, desc)?;
        Ok(SceneId(insert(&mut self.core.scenes, scene)))
    }

    /// Destroys a scene; its memory is freed once every submitted frame has completed.
    pub fn destroy_scene(&mut self, id: SceneId) {
        if let Some(s) = self.core.scenes.get_mut(id.0 as usize) {
            *s = None;
        }
    }

    /// The froxel depth every view of the scene uses, in m (`SceneRoot::near_range`).
    pub fn set_near_range(&mut self, id: SceneId, near_range: f32) -> Result<()> {
        self.core.scene(id)?.cpu.lock().unwrap().near_range = near_range;
        Ok(())
    }

    /// History for one persistent camera.
    pub fn create_view_history(&mut self) -> Result<ViewHistoryId> {
        let exposure =
            self.device
                .gpu_malloc::<ExposureState>(MemoryKind::DeviceLocal, 2, "exposure")?;
        // Both slots start with pre-exposure 1 and `initialized` 0.
        let base = exposure.gpu().address().0;
        let words = (size_of::<ExposureState>() / 4) as u32;
        let one = 1.0f32.to_bits();
        self.core
            .global
            .get_mut()
            .unwrap()
            .extend((0..2u64).flat_map(|slot| {
                let at = base + slot * size_of::<ExposureState>() as u64;
                [
                    Record::fill(0, at, 2, one),
                    Record::fill(0, at + 8, words - 2, 0),
                ]
            }));
        let arena = ViewArena::new(&self.device, self.core.alias_view_targets);
        let readback = self.device.gpu_malloc(
            MemoryKind::Readback,
            u64::from(self.core.fif),
            "visible cluster readback",
        )?;
        let visible = self.core.budgets.initial_visible_clusters;
        let history = ViewHistory::new(exposure, arena, visible, readback);
        Ok(ViewHistoryId(insert(&mut self.core.histories, history)))
    }

    /// What the arena of view `history` holds, as its last render laid it out.
    pub fn view_arena_report(&self, history: ViewHistoryId) -> Option<ArenaReport> {
        self.core
            .histories
            .get(history.0 as usize)
            .and_then(Option::as_ref)
            .map(|h| h.arena.report())
    }

    pub fn destroy_view_history(&mut self, id: ViewHistoryId) {
        let Some(h) = self
            .core
            .histories
            .get_mut(id.0 as usize)
            .and_then(Option::take)
        else {
            return;
        };
        // Its slots' initialization may not have been applied yet.
        let base = h.exposure.gpu().address().0;
        let end = base + h.exposure.size();
        self.core
            .global
            .get_mut()
            .unwrap()
            .retain(|r| !(base..end).contains(&r.dst));
    }

    /// Figures of the last completed frame.
    pub fn stats(&self) -> &FrameStats {
        &self.core.stats
    }

    /// Step 1: waits until frame N - frames_in_flight has completed, for as long as the render
    /// queue goes on completing work (an error once it completes nothing for 10 s), sleeps on
    /// the low latency handle when one is given, recycles the upload window and the slots
    /// freed before it, swaps in hot-reloaded pipelines, and opens frame N. Raises a panic of
    /// the loading thread.
    pub fn begin_frame(&mut self, desc: &FrameDesc<'_>) -> Result<Frame<'_>> {
        let started = Instant::now();
        // The loading thread ends only at `Msg::Stop`, so one that has ended panicked, and
        // nothing would load again.
        if self
            .loading_thread
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
        {
            let thread = self.loading_thread.take().expect("a loading thread");
            if let Err(panic) = thread.join() {
                std::panic::resume_unwind(panic);
            }
        }
        let core = &mut self.core;
        for (i, id) in desc.scenes.iter().enumerate() {
            core.scene(*id)?;
            assert!(!desc.scenes[..i].contains(id), "{id:?} listed twice");
        }
        let frame = core.next_frame;
        let render = self.device.queue(QueueKind::Render).expect("render queue");
        while let Some(f) = core.in_flight.front() {
            let due = f.id + u64::from(core.fif) <= frame;
            let last = *f.points.last().expect("a frame submits");
            if !due && !self.device.is_complete(last) {
                break;
            }
            render.wait_while_completing(last, Duration::from_secs(10))?;
            let mut f = core.in_flight.pop_front().unwrap();
            f.stats.gpu = f
                .points
                .iter()
                .flat_map(|p| self.device.pass_results(*p).unwrap_or_default())
                .collect();
            let slot = (f.id % u64::from(core.fif)) as usize;
            for id in &f.traced {
                let Ok(scene) = core.scene(*id) else { continue };
                if let Some(figures) = scene.cpu.lock().unwrap().ptlas.figures(slot) {
                    f.stats.rt.add(&figures);
                }
            }
            let mut feedback = PageFeedback::default();
            for id in &f.updated {
                let Ok(scene) = core.scene(*id) else { continue };
                let mut cpu = scene.cpu.lock().unwrap();
                if let Some(r) = &cpu.producer_readback {
                    let (producers, tessellation) = r.figures(slot);
                    f.stats.producers.add(&producers);
                    f.stats.tessellation.add(&tessellation);
                }
                if let Some(cuts) = &mut cpu.cuts {
                    cuts.figures(slot, &mut f.stats.cuts);
                    cuts.feedback(slot, &mut feedback);
                    cuts.observe(slot);
                    let rt = &mut f.stats.detail.rt_scale;
                    *rt = rt.max(cuts.rt_scale(slot));
                }
                if let Some(skin) = &cpu.skin_frame {
                    f.stats.skin.add(&skin.figures(slot));
                }
            }
            for h in core.histories.iter_mut().flatten() {
                if h.rendered[slot] == Some(f.id) {
                    let wanted = h.readback.as_slice().expect("readback is mapped")[slot];
                    h.visible.observe(wanted);
                }
            }
            // A frame that updated no scene read nothing, so it says nothing of what one wants.
            let sampled = !f.updated.is_empty();
            if sampled {
                core.mounts.streaming.begin_sample();
            }
            core.mounts.page_feedback(&feedback, frame);
            core.mounts.streaming.texture_feedback(slot, f.id);
            if sampled {
                let mesh = core.mounts.streaming.end_sample();
                core.memory.step(Sample {
                    scale: f.stats.detail.memory_scale,
                    demand: mesh.bytes,
                    capacity: mesh.budget,
                    sheddable: mesh.streamed > 0,
                    whole: mesh.absent == 0,
                });
                f.stats.detail.mesh_demand = mesh.bytes;
                f.stats.detail.mesh_budget = mesh.budget;
            }
            core.completed_frame = f.id;
            core.stats = f.stats;
        }
        if let Some(latency) = desc.low_latency {
            latency.sleep();
        }
        core.next_frame += 1;
        let completed = core.completed_frame;
        for scene in core.scenes.iter().flatten() {
            let cpu = &mut *scene.cpu.lock().unwrap();
            cpu.removed.retain(|&(frame, range)| {
                let done = frame <= completed;
                if done {
                    cpu.slots.free(range.first, range.count);
                }
                !done
            });
            cpu.emissive_removed.retain(|&(frame, first, count)| {
                let done = frame <= completed;
                if done {
                    cpu.emissive.free(first, count);
                }
                !done
            });
            cpu.retired.retain(|(frame, _)| *frame > completed);
            cpu.groups_removed.retain(|(frame, _)| *frame > completed);
            let sets = &mut cpu.page_sets;
            cpu.pages_removed.retain(|(frame, p)| {
                let done = *frame <= completed;
                if done {
                    sets.free(p.sets.0, p.sets.1);
                }
                !done
            });
            cpu.skinned_sets_removed.retain(|&(frame, (first, count))| {
                let done = frame <= completed;
                if done {
                    sets.free(first, count);
                }
                !done
            });
            cpu.skins_removed.retain(|&(frame, first, count, _)| {
                let done = frame <= completed;
                if done {
                    cpu.skinned.free(first, count);
                }
                !done
            });
            cpu.ptlas.retire(completed);
        }
        core.mounts.retire(completed, &core.materials);
        core.mounts.streaming.retire(completed);
        self.device.collect_garbage();
        let slot = (frame % u64::from(core.fif)) as usize;
        core.windows[slot].reset();
        core.request_pages();
        // Scenes report what they hold in their updates; a skipped scene's last one bounds it.
        let settled = core
            .scenes
            .iter()
            .flatten()
            .filter_map(|s| {
                let cpu = s.cpu.lock().unwrap();
                (cpu.ptlas.high_water > 0).then_some(cpu.last_update)
            })
            .min()
            .unwrap_or(frame);
        let mut apply = Apply {
            device: &self.device,
            window: &core.windows[slot],
            seq: &core.seq,
            records: core.global.get_mut().unwrap(),
            materials: &core.materials,
            model_cpu: &mut core.model_cpu,
            cages: &mut core.cages,
        };
        let mut loading_points = Vec::new();
        for batch in self.loading.take_batches(&self.device) {
            loading_points.extend(batch.point);
            core.mounts.apply(batch, &mut apply);
        }
        for batch in self.loading.take_edits(&self.device) {
            loading_points.extend(batch.point);
            core.mounts.land_edit(batch);
        }
        core.mounts.step_edits(&mut apply);
        loading_points.extend(core.mounts.streaming.step(
            frame,
            settled,
            core.memory.scale() > 1.0,
            &self.loading,
            &mut apply,
        )?);
        core.mounts.begin_edits(core.budgets.edit_triangles);
        let events = core.mounts.streaming.take_page_events();
        for (page, records) in events {
            core.land_page(page, records);
        }
        for (page, window, at) in core.mounts.streaming.take_window_events() {
            if let Some(p) = core.mounts.page_mut(page) {
                p.windows[window as usize] = at;
            }
        }
        #[cfg(feature = "hot-reload")]
        let reloads = self
            .reloader
            .poll(&mut self.pipelines, &self.loading, &self.device);
        #[cfg(not(feature = "hot-reload"))]
        let reloads = Vec::new();
        let mut list = render.command_list()?;
        // Work from earlier frames on this queue finishes before this frame's writes.
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        let now = Instant::now();
        Ok(Frame {
            device: &self.device,
            kernels: self.kernels,
            pipelines: &self.pipelines,
            core: &mut self.core,
            id: frame,
            time: desc.time,
            listed: desc.scenes.to_vec(),
            window: slot,
            list: Some(list),
            output: None,
            submits: Vec::new(),
            after_output: None,
            updated: false,
            cpu: vec![("begin_frame", now - started)],
            step_start: now,
            scatter: Default::default(),
            latency: desc.low_latency.cloned(),
            loading: loading_points,
            reloads,
            traced: Vec::new(),
            masks_refused: 0,
        })
    }
}

impl Core {
    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// What any scene may hold by the tables every scene shares: what a material may show,
    /// candidates on every triangle of a copy fading between the steps of a chain, and textures
    /// that have streamed.
    fn features(&self) -> SceneFeatures {
        let fades = if self.chained {
            SceneFeatures::COVERAGE
        } else {
            SceneFeatures::NONE
        };
        let streams = if self.mounts.streaming.streamed() {
            SceneFeatures::STREAMED_TEXTURES
        } else {
            SceneFeatures::NONE
        };
        self.materials.features() | fades | streams
    }

    /// Writes `model`'s chain into its `ModelGpu` with the next frame's records.
    fn write_chain(&mut self, model: ModelId, chain: &[(ModelId, f32)]) {
        let mut words = [0u32; 1 + 2 * MAX_CHAIN];
        words[0] = chain.len() as u32;
        for (i, &(m, distance)) in chain.iter().enumerate() {
            words[1 + i] = m.0;
            words[1 + MAX_CHAIN + i] = distance.to_bits();
        }
        let at = self
            .mounts
            .models
            .gpu()
            .offset(u64::from(model.0))
            .address()
            .0
            + std::mem::offset_of!(crate::ModelGpu, chain_count) as u64;
        let seq = self.next_seq();
        let records = self.global.get_mut().unwrap();
        for (i, &w) in words.iter().enumerate() {
            records.push(Record::fill(seq, at + 4 * i as u64, 1, w));
        }
    }

    /// Wants the records of every placed page and the windows of rows its clock needs for this
    /// step, nearest a focus first, as far as the instance page budget holds them; the others
    /// are left to leave after their hold.
    fn request_pages(&mut self) {
        let mut wanted: FxHashMap<PageId, f64> = FxHashMap::default();
        let edge = f64::from(1u32 << self.cell_log2);
        let mut windows: FxHashMap<(PageId, u32), f64> = FxHashMap::default();
        for scene in self.scenes.iter().flatten() {
            let cpu = scene.cpu.lock().unwrap();
            let skinned = cpu.skinned_pages.iter().map(|p| (p.page, p.group));
            for (page, group) in cpu.pages.iter().map(|p| (p.page, p.group)).chain(skinned) {
                let Some((center, radius)) = self.mounts.page_sphere(page, self.cell_log2) else {
                    continue;
                };
                // The sphere is the rest pose: only the GPU knows where a joint has a page now.
                let moves = self
                    .mounts
                    .page(page)
                    .is_some_and(|p| !p.info.motion_times.is_empty());
                let hung = moves
                    || cpu.groups[group.0 as usize]
                        .as_ref()
                        .is_some_and(|g| g.frame.is_some());
                let distance = cpu
                    .focus
                    .filter(|_| !hung)
                    .map_or(0.0, |f| (f.distance(center) - radius).max(0.0));
                let e = wanted.entry(page).or_insert(f64::INFINITY);
                *e = e.min(distance / edge);
            }
            for m in &cpu.page_motions {
                let Some(p) = self.mounts.page(m.page) else {
                    continue;
                };
                for (window, rank) in m.windows(p.info.header.motion_window) {
                    let e = windows.entry((m.page, window)).or_insert(f64::INFINITY);
                    *e = e.min(rank);
                }
            }
        }
        // Records and windows in one order: what stands now comes before what lies ahead.
        let records = wanted.into_iter().map(|(page, at)| (at, page, None));
        let rows = windows
            .into_iter()
            .map(|((page, window), rank)| (rank, page, Some(window)));
        let mut units: Vec<(f64, PageId, Option<u32>)> = records.chain(rows).collect();
        units.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        let budget = self.mounts.streaming.instance_page_budget();
        let mut bytes = 0;
        let mut over = 0;
        let source = RequestSource::InstancePages;
        for (at, page, window) in units {
            let size = match window {
                Some(w) => self.mounts.window_bytes(page, w),
                None => self.mounts.page_bytes(page),
            };
            if bytes + size > budget {
                over += u32::from(window.is_none() || at == 0.0);
                continue;
            }
            bytes += size;
            match window {
                Some(w) => self.mounts.request_window(page, w, at as u32, source),
                None => self.mounts.request_page(page, at as u32, source),
            }
        }
        self.mounts.streaming.note_pages_over_budget(over);
    }

    /// Points the page groups of `page` in every scene at its records, or hides their copies.
    /// An event that outlived the page's pack names memory already given back.
    fn land_page(&mut self, page: PageId, records: u64) {
        let Some(p) = self.mounts.page_mut(page) else {
            return;
        };
        // SAFETY: the records stay resident until the event that writes this null.
        p.records = unsafe { GpuPtr::from_address(GpuAddress(records)) };
        for scene in self.scenes.iter().flatten() {
            let mut cpu = scene.cpu.lock().unwrap();
            for g in cpu.groups.iter_mut().flatten() {
                if let Some(q) = &mut g.page
                    && q.page == page
                {
                    q.records = records;
                    q.changed = true;
                }
            }
        }
    }

    fn scene(&self, id: SceneId) -> Result<&Scene> {
        self.scenes
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(Error::UnknownScene(id))
    }
}

/// One frame, from `begin_frame` to `end`. Steps run in the fixed order: writes, then
/// `update_scenes`, then views, then `end`.
pub struct Frame<'a> {
    device: &'a Device,
    kernels: &'static KernelSet,
    pipelines: &'a Pipelines,
    core: &'a mut Core,
    id: u64,
    time: FrameTime,
    listed: Vec<SceneId>,
    window: usize,
    list: Option<CommandList<'a>>,
    /// The output list on the Compute queue, once the frame records into it.
    output: Option<CommandList<'a>>,
    /// Render submits made before `end`.
    submits: Vec<TimelinePoint>,
    /// The previous frame's output submit, which the frame's render work after
    /// `update_scenes` waits for.
    after_output: Option<TimelinePoint>,
    updated: bool,
    cpu: Vec<(&'static str, Duration)>,
    step_start: Instant,
    scatter: crate::upload::ScatterTotals,
    latency: Option<LowLatency>,
    /// Loading batches taken at step 1; the frame's submit waits them.
    loading: Vec<TimelinePoint>,
    reloads: Vec<ReloadReport>,
    /// Scenes whose PTLAS this frame builds.
    traced: Vec<SceneId>,
    /// Section masks this frame could not build.
    masks_refused: u32,
}

impl<'a> Frame<'a> {
    /// orr's frame id: strictly increasing from 1.
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn time(&self) -> FrameTime {
        self.time
    }

    /// The newest frame whose work had completed when this one began.
    pub fn completed(&self) -> u64 {
        self.core.completed_frame
    }

    /// The current pipeline of `id`.
    pub fn pipeline(&self, id: PipelineId) -> &'a ComputePipeline {
        self.pipelines.get(id)
    }

    /// Tells orr that kernels change edit `id` at this frame's scene compute point, so its
    /// clusters' culling data and CLAS rebuild right after it. False, with nothing to
    /// rebuild: the frame's [`Budgets::edit_triangles`] is spent or the edit is not ready,
    /// and the kernels must leave the edit unchanged this frame so that raster and ray
    /// tracing keep agreeing. Vertices they move take motion from where they were the frame
    /// before. Call it before [`Frame::update_scenes`].
    pub fn edit_dirty(&mut self, id: EditId) -> bool {
        assert!(!self.updated, "edits are marked before update_scenes");
        self.core.mounts.edit_dirty(id, true)
    }

    /// Like [`edit_dirty`](Self::edit_dirty), but what kernels change has no motion this
    /// frame: a vertex that jumps leaves no history to smear.
    pub fn edit_dirty_without_motion(&mut self, id: EditId) -> bool {
        assert!(!self.updated, "edits are marked before update_scenes");
        self.core.mounts.edit_dirty(id, false)
    }

    /// What hot reload applied at this frame's step 1; always empty without the
    /// `hot-reload` feature.
    pub fn reloads(&self) -> &[ReloadReport] {
        &self.reloads
    }

    /// Asks `update_scenes` to choose the scene's geometry for a view of this many render
    /// pixels per radian ([`crate::Projection::pixels_per_radian`]); once this frame's update
    /// has run, the next frame's. The largest request wins; an update without a request keeps
    /// the value of the one before it, at first 1000.
    pub fn request_detail(&self, id: SceneId, pixels_per_radian: f32) -> Result<()> {
        let scene = self.core.scene(id)?;
        let mut cpu = scene.cpu.lock().unwrap();
        let largest = cpu
            .requested_detail
            .map_or(pixels_per_radian, |r| r.max(pixels_per_radian));
        cpu.requested_detail = Some(largest);
        Ok(())
    }

    /// Step 2: the writer of listed scene `id`. One writer per scene at a time.
    pub fn scene(&self, id: SceneId) -> Result<SceneWriter<'_>> {
        if !self.listed.contains(&id) {
            return Err(Error::NotListed(id));
        }
        assert!(!self.updated, "scene writes after update_scenes");
        let scene = self.core.scene(id)?;
        if scene.writing.swap(true, Ordering::Acquire) {
            return Err(Error::WriterTaken(id));
        }
        Ok(SceneWriter {
            scene,
            device: self.device,
            kernels: self.kernels,
            window: &self.core.windows[self.window],
            seq: &self.core.seq,
            frame: self.id,
            cell_log2: self.core.cell_log2,
            model_cpu: &self.core.model_cpu,
            cages: &self.core.cages,
            patterns: &self.core.patterns,
            mounts: &self.core.mounts,
            materials: &self.core.materials,
        })
    }

    /// Step 2: the material writer. Any number, on any threads.
    pub fn materials(&self) -> MaterialWriter<'_> {
        assert!(!self.updated, "material writes after update_scenes");
        MaterialWriter {
            device: self.device,
            table: &self.core.materials,
            window: &self.core.windows[self.window],
            seq: &self.core.seq,
            records: &self.core.global,
        }
    }

    /// Sets `marker` under this frame's id when the frame has a low latency handle. Call on
    /// the render thread.
    pub fn latency_marker(&self, marker: ConsumerMarker) {
        self.marker(match marker {
            ConsumerMarker::SimulationStart => LatencyMarker::SimulationStart,
            ConsumerMarker::SimulationEnd => LatencyMarker::SimulationEnd,
            ConsumerMarker::InputSample => LatencyMarker::InputSample,
        });
    }

    fn marker(&self, marker: LatencyMarker) {
        if let Some(latency) = &self.latency {
            latency.marker(self.id, marker);
        }
    }

    fn step(&mut self, name: &'static str) {
        let now = Instant::now();
        self.cpu.push((name, now - self.step_start));
        self.step_start = now;
    }

    /// Step 3: applies every write of this frame, then gives each listed scene, in order,
    /// to `compute` with the frame's command list. Panics if a listed scene never set the data
    /// of an environment or medium its kernels link.
    pub fn update_scenes(
        &mut self,
        compute: &mut dyn FnMut(&mut SceneCompute<'_, 'a>),
    ) -> Result<()> {
        assert!(!self.updated, "update_scenes runs once a frame");
        self.step("write");
        self.marker(LatencyMarker::RenderSubmitStart);
        self.updated = true;
        let device = self.device;
        let core = &mut *self.core;
        let window = &core.windows[self.window];
        let mut list = self.list.take().expect("frame list");
        let mut records = std::mem::take(&mut *core.global.lock().unwrap());
        let mut copies = Vec::new();
        let mut rebased = Vec::new();
        let mut roots = Vec::new();
        let mut traced = Vec::new();
        let mut masks = Vec::new();
        // Cut tables that found no memory to grow.
        let mut short = Vec::new();
        for &id in &self.listed {
            let scene = core.scene(id)?;
            let cpu = &mut *scene.cpu.lock().unwrap();
            assert!(
                cpu.environment.is_some() || self.kernels.environment == DEFAULT_ENVIRONMENT,
                "{id:?} has no data for the environment its kernels link: call set_environment"
            );
            assert!(
                cpu.medium.is_some() || self.kernels.medium == DEFAULT_MEDIUM,
                "{id:?} has no data for the medium its kernels link: call set_medium"
            );
            cpu.last_update = self.id;
            records.append(&mut cpu.records);
            masks.extend(std::mem::take(&mut cpu.masks));
            for p in cpu.producers.iter().flatten() {
                if p.update == ProducerUpdate::EveryFrame {
                    let at = p.emit_counts.address().0;
                    records.push(Record::fill(core.next_seq(), at, 4, 0));
                }
            }
            if let Some(detail) = cpu.requested_detail.take() {
                cpu.pixels_per_radian = detail;
            }
            let mut rebase = false;
            if let Some(focus) = cpu.focus
                && needs_rebase(cpu.origin_cell, focus, core.cell_log2)
            {
                cpu.origin_cell = origin_for(focus, core.cell_log2);
                rebased.push((scene.tables.root.gpu(), scene.desc.light_capacity));
                rebase = true;
            }
            let producers = cpu.producers.iter().any(Option::is_some);
            let tlas = cpu.ptlas.tlas(device, core.fif, producers)?;
            if cpu.cuts.is_none() {
                let heap = device.memory_report().heap_of(MemoryKind::DeviceLocal).size;
                let sizes = CutSizes {
                    slots: scene.desc.instance_capacity,
                    queue: core.budgets.initial_traversal_nodes,
                    candidates: core.budgets.initial_candidate_items,
                    rt_clas: core.budgets.initial_rt_cut_clusters,
                    rt_blas: core.budgets.initial_rt_cut_blas,
                    rt_bytes: core.budgets.rt_cut_memory.bytes(heap),
                    models: core.budgets.models,
                    entries: core.budgets.mesh_pages,
                    requests: core.budgets.traversal_requests,
                };
                cpu.cuts = Some(Cuts::new(device, sizes, core.fif)?);
            }
            let cuts = cpu.cuts.as_mut().expect("made above");
            match cuts.fit(device, self.id, response_period(core.fif)) {
                Err(e) if e.is_out_of_memory() => short.push(e),
                r => r?,
            }
            let cut = cuts.gpu();
            if cpu.skinned_high_water > 0 && cpu.skin_frame.is_none() {
                cpu.skin_frame = Some(SkinFrame::new(
                    device,
                    scene.desc.skinned_clusters,
                    scene.desc.skinned_capacity,
                    core.fif,
                )?);
            }
            let skin_work = cpu.skin_frame.as_ref().map(|f| f.work);
            records.append(&mut cpu.ptlas.partition_records(device, window, &core.seq)?);
            for (index, g) in cpu.groups.iter_mut().enumerate() {
                let Some(g) = g else { continue };
                let before = g.record();
                let fresh = g.created_frame == self.id;
                match g.writer {
                    GroupWriter::Cpu => {
                        if g.written && !fresh {
                            copies.push((
                                g.buffers[1].gpu(),
                                g.buffers[0].gpu(),
                                g.buffers[0].size(),
                            ));
                        }
                        g.previous_is_current = !g.written || fresh;
                    }
                    GroupWriter::Gpu if !fresh => {
                        g.current = 1 - g.current;
                        g.previous_is_current = false;
                    }
                    GroupWriter::Page | GroupWriter::Gpu => {}
                }
                g.written = false;
                let after = g.record();
                let landed = g
                    .page
                    .as_mut()
                    .is_some_and(|p| std::mem::take(&mut p.changed));
                if landed || bytemuck::bytes_of(&before) != bytemuck::bytes_of(&after) {
                    let src = window.put(device, &after)?;
                    let dst = scene.tables.groups.gpu().offset(index as u64);
                    records.push(Record::values(core.next_seq(), dst, src, 1));
                }
            }
            let t = &scene.tables;
            if std::mem::take(&mut cpu.decals_changed) {
                let bins = build_bins(&cpu.decals);
                let slots: &[u32] = bytemuck::cast_slice(&bins.slots);
                for (dst, words) in [
                    (t.decal_bins.gpu().cast::<u32>(), slots),
                    (t.decal_refs.gpu(), &bins.refs),
                ] {
                    if !words.is_empty() {
                        let n = words.len() as u32;
                        let (src, out) = window.slice::<u32>(device, n)?;
                        out.copy_from_slice(words);
                        records.push(Record::values(core.next_seq(), dst, src, n));
                    }
                }
                cpu.decal_bin_slots = bins.slots.len() as u32;
                cpu.decal_globals = bins.globals;
            }
            let root = SceneRoot {
                instances: t.instances.gpu(),
                slot_state: t.slot_state.gpu(),
                groups: t.groups.gpu(),
                models: core.mounts.models.gpu(),
                meshes: core.mounts.meshes.gpu(),
                materials: core.materials.table.gpu(),
                clusters: core.mounts.clusters.gpu(),
                material_sets: t.material_sets.gpu().cast(),
                lights: t.lights.gpu(),
                emissive: t.emissive.gpu(),
                producers: t.producers.gpu(),
                batches: t.batches.gpu(),
                patches: t.patches.gpu(),
                producer_state: t.producer_state.gpu(),
                sky_stats: t.sky_stats.gpu(),
                environment: cpu.environment.unwrap_or(0),
                medium: cpu.medium.unwrap_or(0),
                surface_data: cpu.surface_data,
                tlas: tlas.unwrap_or_else(|| device.empty_tlas()),
                part_slots: cpu.ptlas.part_slots.gpu(),
                origin_cell: cpu.origin_cell.extend(core.cell_log2 as i32).to_array(),
                delta_time: self.time.delta,
                frame_index: self.id as u32,
                instance_capacity: scene.desc.instance_capacity,
                group_capacity: scene.desc.group_capacity,
                light_capacity: scene.desc.light_capacity,
                material_set_capacity: scene.desc.material_set_capacity
                    + scene.desc.page_material_sets,
                producer_capacity: scene.desc.producer_capacity,
                emissive_capacity: scene.desc.emissive_capacity,
                near_range: cpu.near_range,
                part_base: scene.desc.instance_capacity,
                part_capacity: scene.desc.part_capacity,
                producer_base: scene.desc.instance_capacity + scene.desc.part_capacity,
                cut,
                skinned: t.skinned.gpu(),
                skin_state: t.skin_state.gpu(),
                skin_work: skin_work.unwrap_or(GpuPtr::NULL).cast(),
                skinned_capacity: scene.desc.skinned_capacity,
                decal_capacity: scene.desc.decal_capacity,
                decals: t.decals.gpu(),
                decal_bins: t.decal_bins.gpu(),
                decal_refs: t.decal_refs.gpu(),
                decal_globals: cpu.decal_globals,
                decal_bin_slots: cpu.decal_bin_slots,
                decal_bin_log2: decals::bin_log2(core.cell_log2),
                proxy_base: scene.desc.instance_capacity
                    + scene.desc.part_capacity
                    + scene.desc.producer_capacity,
                proxies: t.proxies.gpu(),
                proxy_capacity: scene.desc.medium_proxy_capacity,
                light_base: scene.desc.instance_capacity
                    + scene.desc.part_capacity
                    + scene.desc.producer_capacity
                    + scene.desc.medium_proxy_capacity,
                emissive_lists: t.emissive_lists.gpu(),
            };
            let src = window.put(device, &root)?;
            records.push(Record::values(core.next_seq(), t.root.gpu(), src, 1));
            if tlas.is_some() {
                traced.push((id, rebase));
            }
            roots.push((
                id,
                t.root.gpu(),
                t.batches.gpu(),
                cpu.focus_detail(),
                scene.desc,
            ));
        }
        for e in &short {
            core.mounts.streaming.out_of_memory(e);
        }

        if !masks.is_empty() {
            list.begin_pass("masks", false);
            self.masks_refused += core.mounts.build_masks(
                device,
                window,
                &mut list,
                masks,
                self.id,
                &core.seq,
                &mut records,
            )?;
            list.end_pass();
        }
        list.begin_pass("upload", true);
        for &(dst, src, bytes) in &copies {
            list.copy(dst, src, bytes);
        }
        if !copies.is_empty() {
            list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        }
        let own = &self.pipelines.own;
        let scatter_pipeline = self.pipelines.get(own.scatter);
        self.scatter = scatter(device, window, &mut list, scatter_pipeline, records)?;
        list.end_pass();
        list.barrier(
            Stages::COMPUTE | Stages::TRANSFER,
            Stages::ALL,
            Hazards::empty(),
        );
        if !rebased.is_empty() {
            list.begin_pass("rebase", false);
            for &(root, lights) in &rebased {
                let r = world_kernel::Root {
                    scene: root,
                    _pad0: 0,
                    _pad1: 0,
                };
                let ptr = window.put_root(device, &r)?;
                let p = self.pipelines.get(own.rebase);
                list.dispatch(p, ptr, [lights.div_ceil(64), 1, 1]);
            }
            list.end_pass();
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        }
        let mut expansions = Vec::new();
        let mut entry_jobs = Vec::new();
        for &id in &self.listed {
            let scene = core.scene(id)?;
            let root = scene.tables.root.gpu();
            let mut cpu = scene.cpu.lock().unwrap();
            let cpu = &mut *cpu;
            let mut rewritten: Vec<std::ops::Range<u32>> = Vec::new();
            for p in cpu.pages.iter_mut() {
                if p.expanded && !p.entries_stale {
                    continue;
                }
                let records = cpu.groups[p.group.0 as usize]
                    .as_ref()
                    .and_then(|g| g.page.as_ref())
                    .map_or(0, |g| g.records);
                if records == 0 {
                    continue;
                }
                let entries_only = p.expanded;
                if entries_only {
                    rewritten.push(p.range.first..p.range.first + p.range.count);
                }
                p.expanded = true;
                p.entries_stale = false;
                let layout = p.entry_layout();
                let (entries, out) = window.slice(device, layout.len() as u32)?;
                out.copy_from_slice(&layout);
                expansions.push(instance_page::PageExpandRoot {
                    scene: root,
                    // SAFETY: the records stay resident through this frame: their eviction is
                    // written at step 1 before the event that hides them.
                    records: unsafe { GpuPtr::from_address(GpuAddress(records)) },
                    prototypes: p.prototypes.gpu().cast(),
                    entries,
                    first_slot: p.range.first,
                    count: p.range.count,
                    prototype_count: p.prototype_count,
                    first_entry: p.first_entry,
                    linear_rows: p.linear_rows,
                    entries_only: u32::from(entries_only),
                    _pad0: 0,
                    _pad1: 0,
                });
            }
            if !rewritten.is_empty() {
                // Their slots take their entries here in place of holding none. Page ranges
                // are disjoint, so one search finds the only one a slot can lie in.
                rewritten.sort_unstable_by_key(|r| r.start);
                cpu.entry_jobs.retain(|j| {
                    let k = rewritten.partition_point(|r| r.start <= j[0]);
                    k == 0 || !rewritten[k - 1].contains(&j[0])
                });
            }
            if !cpu.entry_jobs.is_empty() {
                let jobs = std::mem::take(&mut cpu.entry_jobs);
                let (src, out) = window.slice::<[u32; 4]>(device, jobs.len() as u32)?;
                out.copy_from_slice(&jobs);
                entry_jobs.push(emissive::Root {
                    scene: root,
                    jobs: src.cast(),
                    job_count: jobs.len() as u32,
                    _pad0: 0,
                    _pad1: 0,
                });
            }
        }
        let mut motions = Vec::new();
        for &id in &self.listed {
            let scene = core.scene(id)?;
            let cpu = &mut *scene.cpu.lock().unwrap();
            for m in &mut cpu.page_motions {
                let records = cpu.groups[m.records.0 as usize]
                    .as_ref()
                    .and_then(|g| g.page.as_ref())
                    .map_or(0, |g| g.records);
                let page = core.mounts.page(m.page);
                let row_size = size_of::<orr_pack::instance::MotionRow>() as u64;
                // The address of interval `k`'s rows, 0 while its window is not resident.
                let row_at = |k: u32| {
                    let p = page.filter(|_| records != 0)?;
                    let per_window = p.info.header.motion_window;
                    let at = *p.windows.get((k / per_window) as usize)?;
                    let k = u64::from(k % per_window);
                    (at != 0).then(|| at + k * u64::from(m.count) * row_size)
                };
                let (k, dt) = orr_pack::instance::motion_interval(&m.times, m.time);
                let now = (k as u32, dt);
                if row_at(now.0).is_some() {
                    m.held = Some(now);
                }
                let (rows, dt) = match m.held.and_then(|h| Some((row_at(h.0)?, h.1))) {
                    // SAFETY: an evicted window stays alive through this frame, as records do for
                    // the expansions above.
                    Some((at, dt)) => (unsafe { GpuPtr::from_address(GpuAddress(at)) }, dt),
                    None => (GpuPtr::NULL, 0.0),
                };
                motions.push(page_motion::PageMotionRoot {
                    scene: scene.tables.root.gpu(),
                    rows,
                    left0: m.left[0],
                    left1: m.left[1],
                    left2: m.left[2],
                    right0: m.right[0],
                    right1: m.right[1],
                    right2: m.right[2],
                    group: m.group.0,
                    count: m.count,
                    dt,
                    _pad: 0,
                });
            }
        }
        if !expansions.is_empty() || !motions.is_empty() || !entry_jobs.is_empty() {
            list.begin_pass("instance pages", false);
            for r in &entry_jobs {
                let ptr = window.put_root(device, r)?;
                let p = self.pipelines.get(own.emissive_entries);
                list.dispatch(p, ptr, [r.job_count.div_ceil(64), 1, 1]);
            }
            for r in &expansions {
                let ptr = window.put_root(device, r)?;
                let p = self.pipelines.get(own.expand_page);
                list.dispatch(p, ptr, [r.count.div_ceil(64), 1, 1]);
            }
            for r in &motions {
                let ptr = window.put_root(device, r)?;
                let p = self.pipelines.get(own.page_motion);
                list.dispatch(p, ptr, [r.count.div_ceil(64), 1, 1]);
            }
            list.end_pass();
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        }
        core.mounts.record_edit_history(&mut list);
        for &(scene, root, batches, focus, _) in &roots {
            let mut sc = SceneCompute {
                cmd: &mut list,
                scene,
                root,
                batches,
                time: self.time,
                focus,
                device,
                window,
                pipelines: self.pipelines,
            };
            compute(&mut sc);
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        }
        core.mounts.record_edits(device, window, &mut list)?;
        record_producers(
            device,
            core,
            &self.listed,
            window,
            &mut list,
            self.pipelines,
            self.id,
        )?;
        list.begin_pass("animate", false);
        for &(id, root, _, focus, _) in &roots {
            let slots = core.scene(id)?.cpu.lock().unwrap().ptlas.high_water;
            if slots == 0 {
                continue;
            }
            let (focus_cell, focus_offset) = match focus {
                Some((f, _)) => {
                    let w = WorldPos::new(f, core.cell_log2);
                    (w.cell.extend(1).to_array(), w.offset.extend(0.0).to_array())
                }
                None => ([0; 4], [0.0; 4]),
            };
            let r = animation::AnimationRoot {
                scene: root,
                _pad0: 0,
                focus_cell,
                focus_offset,
                slots,
                _pad1: 0,
                _pad2: 0,
            };
            let ptr = window.put_root(device, &r)?;
            list.dispatch(
                self.pipelines.get(own.animate),
                ptr,
                [slots.div_ceil(64), 1, 1],
            );
        }
        list.end_pass();
        list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        for &(id, root, _, focus, _) in &roots {
            let cpu = core.scene(id)?.cpu.lock().unwrap();
            let Some(skin) = &cpu.skin_frame else {
                continue;
            };
            let focus = focus.map(|(f, ppr)| {
                let w = WorldPos::new(f, core.cell_log2);
                (w.cell.extend(1).to_array(), w.offset.extend(ppr).to_array())
            });
            skin.record(
                device,
                window,
                &mut list,
                self.pipelines,
                &core.skin,
                root,
                focus,
                core.raster_error_pixels * core.memory.scale(),
                cpu.skinned_high_water,
                (self.id % u64::from(core.fif)) as usize,
            )?;
        }
        let hierarchy = core.mounts.hierarchy();
        for &(id, root, _, focus, _) in &roots {
            let scene = core.scene(id)?;
            let cpu = &mut *scene.cpu.lock().unwrap();
            let focus = focus.map(|(f, ppr)| {
                let w = WorldPos::new(f, core.cell_log2);
                (w.cell.extend(1).to_array(), w.offset.extend(ppr).to_array())
            });
            let memory_scale = core.memory.scale();
            let inputs = CutInputs {
                scene: root,
                focus,
                raster_pixels: core.raster_error_pixels * memory_scale,
                rt_pixels: core.rt_error_pixels * memory_scale,
                memory_scale,
                culled_scale: core.culled_error_scale,
                merge_pixels: core.rt_merge_pixels,
                slots: cpu.ptlas.high_water,
                passes: hierarchy.depth,
                parted_edits: core.mounts.has_parted_edit(),
                slot: (self.id % u64::from(core.fif)) as usize,
            };
            let cuts = cpu.cuts.as_mut().expect("made with the root");
            cuts.record(
                device,
                window,
                &mut list,
                self.pipelines,
                &core.cuts,
                &inputs,
            )?;
        }
        if !traced.is_empty() {
            let proxies = traced.iter().any(|&(id, _)| {
                core.scene(id)
                    .is_ok_and(|s| s.cpu.lock().unwrap().ptlas.has_proxy_work())
            });
            if proxies && core.proxy_blas.is_none() {
                core.proxy_blas = Some(ProxyBlas::new(device)?);
            }
            let proxy_blas = core
                .proxy_blas
                .as_mut()
                .map_or(GpuAddress(0), |b| b.record(&mut list));
            let mut scratch = core.ptlas_scratch.lock().unwrap();
            let need = traced
                .iter()
                .filter_map(|&(id, _)| core.scene(id).ok())
                .map(|s| s.cpu.lock().unwrap().ptlas.scratch_bytes(device))
                .max()
                .unwrap_or(0);
            if scratch.as_ref().is_none_or(|a| a.size() < need) {
                *scratch = Some(device.gpu_malloc::<u8>(
                    MemoryKind::DeviceLocal,
                    need.max(256),
                    "PTLAS scratch",
                )?);
            }
            let scratch = scratch.as_ref().expect("made above").gpu().address();
            for &(id, rebased) in &traced {
                let scene = core.scene(id)?;
                let cpu = &mut *scene.cpu.lock().unwrap();
                let (focus_cell, focus_offset) = match cpu.focus_detail() {
                    Some((f, ppr)) => {
                        let w = WorldPos::new(f, core.cell_log2);
                        (w.cell.extend(1).to_array(), w.offset.extend(ppr).to_array())
                    }
                    None => ([0; 4], [0.0; 4]),
                };
                let inputs = FrameInputs {
                    scene: scene.tables.root.gpu(),
                    focus_cell,
                    focus_offset,
                    origin_cell: cpu.origin_cell,
                    cell_log2: core.cell_log2,
                    rebased,
                    budget: core.budgets.ptlas_rebuilt,
                    proxy_blas,
                    slot: (self.id % u64::from(core.fif)) as usize,
                };
                let kernels = &core.ptlas;
                cpu.ptlas.record(
                    device,
                    window,
                    &mut list,
                    self.pipelines,
                    kernels,
                    &inputs,
                    scratch,
                )?;
            }
        }
        self.traced = traced.iter().map(|&(id, _)| id).collect();
        if !roots.is_empty() {
            // The PTLAS kernels write the cuts this pass reads.
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            list.begin_pass("emissive", false);
            for &(_, scene, _, _, desc) in &roots {
                let r = emissive::Root {
                    scene,
                    jobs: GpuPtr::NULL,
                    job_count: 0,
                    _pad0: 0,
                    _pad1: 0,
                };
                let ptr = window.put_root(device, &r)?;
                if desc.producer_capacity > 0 {
                    let p = self.pipelines.get(own.emissive_producers);
                    list.dispatch(p, ptr, [desc.producer_capacity, 1, 1]);
                }
                if desc.emissive_capacity > 0 {
                    let groups = desc.emissive_capacity.div_ceil(64);
                    let p = self.pipelines.get(own.emissive_instances);
                    list.dispatch(p, ptr, [groups, 1, 1]);
                }
            }
            list.end_pass();
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        }
        self.list = Some(list);
        self.overlap_output()?;
        self.step("update_scenes");
        Ok(())
    }

    /// While the previous frame's output list may still run on the Compute queue, submits
    /// the scene update so it overlaps that work, and has the rest of the frame's render
    /// work, which overwrites the targets the output passes read, wait for it.
    fn overlap_output(&mut self) -> Result<()> {
        let Some(previous) = self.core.output.filter(|p| !self.device.is_complete(*p)) else {
            return Ok(());
        };
        let render = self.device.queue(QueueKind::Render).expect("render queue");
        let list = self.list.take().expect("frame list");
        self.submits.push(render.submit(vec![list], &self.loading)?);
        let mut list = render.command_list()?;
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        self.list = Some(list);
        self.after_output = Some(previous);
        Ok(())
    }

    fn ensure_updated(&mut self) -> Result<()> {
        if !self.updated {
            self.update_scenes(&mut |_| {})?;
        }
        Ok(())
    }

    /// The frame's command list on the Render queue, for the steps after `update_scenes`
    /// (views, lighting, UI); runs `update_scenes` first when the consumer did not.
    pub fn list(&mut self) -> Result<&mut CommandList<'a>> {
        self.ensure_updated()?;
        Ok(self.list.as_mut().expect("frame list"))
    }

    /// The list for ray reconstruction and the output passes after it: compute work only.
    /// On a device with a Compute queue it records there and runs after everything in
    /// [`Frame::list`], whatever the recording order, overlapping the next frame's
    /// `update_scenes`; without one it is [`Frame::list`].
    pub fn output_list(&mut self) -> Result<&mut CommandList<'a>> {
        self.ensure_updated()?;
        let Some(compute) = self.device.queue(QueueKind::Compute) else {
            return Ok(self.list.as_mut().expect("frame list"));
        };
        if self.output.is_none() {
            let mut list = compute.command_list()?;
            // Earlier frames' output work finishes before this frame's writes.
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            self.output = Some(list);
        }
        Ok(self.output.as_mut().expect("output list"))
    }

    /// Whether the frame has recorded into an output list of its own on the Compute queue.
    pub fn has_output_list(&self) -> bool {
        self.output.is_some()
    }

    /// Step 4, the raster half of a view: writes its `ViewConstants`, culls and rasters the
    /// scene into the visibility buffer, resolves the G-buffer, depth, motion and HZB, and
    /// moves its history on. Lays out the view's arena with the raster's transient targets
    /// alone and enters [`ViewStep::Raster`].
    pub fn raster_view(&mut self, view: &ViewDesc) -> Result<RasterView> {
        self.raster_view_with(view, &[])
    }

    /// What `scene` may hold now: [`RasterView::features`] before the view is rastered.
    pub fn scene_features(&self, scene: SceneId) -> Result<SceneFeatures> {
        let cpu = self.core.scene(scene)?.cpu.lock().unwrap();
        Ok(self.core.features() | cpu.features())
    }

    /// [`Self::raster_view`], with `requests`, the transient targets of the passes that
    /// follow it, laid out in the view's arena beside the raster's.
    pub fn raster_view_with(
        &mut self,
        view: &ViewDesc,
        requests: &[ArenaRequest],
    ) -> Result<RasterView> {
        self.ensure_updated()?;
        if !self.listed.contains(&view.scene) {
            return Err(Error::NotListed(view.scene));
        }
        let device = self.device;
        let core = &mut *self.core;
        let scene = core.scene(view.scene)?;
        let (origin, focus) = {
            let cpu = scene.cpu.lock().unwrap();
            (cpu.origin_cell, cpu.focus.unwrap_or(view.position))
        };
        let root = scene.tables.root.gpu();
        let sky_stats = scene.tables.sky_stats.gpu();
        let focus = WorldPos::new(focus, core.cell_log2)
            .relative_to(
                WorldPos {
                    cell: origin,
                    offset: Vec3::ZERO,
                },
                core.cell_log2,
            )
            .to_array();
        let (slots, smoke, scene_features, work) = {
            let cpu = scene.cpu.lock().unwrap();
            let features = cpu.features();
            let smoke = features.contains(SceneFeatures::SMOKE);
            let cuts = cpu.cuts.as_ref().expect("an updated scene has cut tables");
            (
                cpu.ptlas.high_water,
                smoke,
                features,
                cuts.candidate_capacity(),
            )
        };
        let features = core.features() | scene_features;
        let producer_capacity = scene.desc.producer_capacity;
        let sizes = TargetSizes {
            extent: view.render_extent.to_array(),
            extended: core.materials.any_extended(),
            bins: core.raster.bins,
            slots: scene.desc.instance_capacity,
        };
        let preprocess = match core.raster.preprocess_layout(self.pipelines) {
            Some(layout) => Some(core.preprocess.allocate(device, layout)?),
            None => None,
        };
        let history = core
            .histories
            .get_mut(view.history.0 as usize)
            .and_then(Option::as_mut)
            .ok_or(Error::UnknownViewHistory(view.history))?;
        let constants = view_constants(
            view,
            history,
            origin,
            core.cell_log2,
            self.id,
            self.time.delta,
        );
        let window = &core.windows[self.window];
        let list = self.list.as_mut().expect("frame list");
        let mut all = ViewTargets::requests(&sizes);
        all.extend_from_slice(requests);
        history.arena.layout(&all, list)?;
        // The second culling pass takes at most one work item per candidate item.
        let want = ListSizes {
            visible_clusters: history.visible.wanted(),
            work,
        };
        let held = history.lists.as_ref().map(|l| l.sizes);
        let mut short = None;
        if held != Some(want) && (held.is_none() || self.id >= history.lists_retry) {
            match ViewLists::new(device, want) {
                Ok(lists) => {
                    history.visible.set(want.visible_clusters);
                    history.lists = Some(lists);
                }
                // The view keeps the lists it has while the streaming budgets make room.
                Err(e) if e.is_out_of_memory() && held.is_some() => {
                    history.lists_retry = self.id + response_period(core.fif);
                    short = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        history.arena.enter(ViewStep::Raster, list);
        match &mut history.targets {
            Some(t) if t.sizes == sizes => t.bind(&history.arena),
            t => *t = Some(ViewTargets::new(device, sizes, &history.arena, list)?),
        }
        let lists = history.lists.as_ref().expect("lists made above");
        let targets = history.targets.as_mut().expect("view targets");
        let hzb_view = (view.scene, view.output_extent.to_array());
        let occlusion = targets.hzb_view.replace(hzb_view) == Some(hzb_view) && !view.camera_cut;
        let targets = &*targets;
        let seen = constants;
        let constants = window.put(device, &constants)?;
        let draws = raster::record(
            device,
            &core.raster,
            self.pipelines,
            window,
            list,
            targets,
            lists,
            root,
            slots,
            producer_capacity,
            constants,
            core.software_raster_pixels,
            occlusion,
            preprocess,
        )?;
        let raster = draws[0];
        let slot = (self.id % u64::from(core.fif)) as usize;
        let wanted = history.readback.gpu().offset(slot as u64);
        list.copy(wanted.cast(), targets.counts().cast(), 4);
        list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
        history.rendered[slot] = Some(self.id);
        let (gbuf_a, gbuf_b, gbuf_x, gbuf_y) = targets.gbuffer_textures();
        let out = RasterView {
            view: constants,
            scene: root,
            gbuffer: raster.cast(),
            targets: targets.gbuffer(lists),
            depth: Arc::clone(&targets.depth.texture),
            motion: Arc::clone(&targets.motion.texture),
            gbuf_a,
            gbuf_b,
            gbuf_x,
            gbuf_y,
            hzb: Arc::clone(&targets.hzb.texture),
            counts: targets.counts(),
            raster,
            smoke_args: smoke.then(|| targets.smoke_args()),
            features,
            history: view.history,
            sky_stats,
            focus,
        };
        history.draws = draws;
        if let Some(e) = short {
            core.mounts.streaming.out_of_memory(&e);
        }
        let mut cpu = core.scene(view.scene)?.cpu.lock().unwrap();
        if cpu.seen_by.is_none_or(|s| s.frame != self.id) {
            cpu.seen_by = Some(crate::scene::LastView {
                history: view.history,
                frame: self.id,
                constants: seen,
            });
        }
        drop(cpu);
        self.step("raster_view");
        Ok(out)
    }

    /// Step 5's transparency point for a view `raster_view` rendered this frame: enters
    /// [`ViewStep::Transparency`], clears the view's overlay, runs `hook` to draw into it,
    /// then composites the overlay over `radiance` (RGBA16F at the render extent,
    /// pre-exposed) in place. Returns the overlay and its sampled view, premultiplied with
    /// coverage in alpha, which holds its contents through [`ViewStep::Transparency`].
    pub fn transparency(
        &mut self,
        view: &RasterView,
        radiance: StorageImageIndex,
        hook: &mut dyn FnMut(&mut TransparencyPass<'_, '_>),
    ) -> Result<(Arc<orr_gpu::Texture>, TextureIndex)> {
        self.transparency_point(view, Some(radiance), hook)
    }

    /// [`Self::transparency`] without the composite: the overlay, RGBA16F at the render extent,
    /// is for ray reconstruction to composite after denoising, and holds its contents through
    /// [`ViewStep::Rr`].
    pub fn transparency_overlay(
        &mut self,
        view: &RasterView,
        hook: &mut dyn FnMut(&mut TransparencyPass<'_, '_>),
    ) -> Result<(Arc<orr_gpu::Texture>, TextureIndex)> {
        self.transparency_point(view, None, hook)
    }

    fn transparency_point(
        &mut self,
        view: &RasterView,
        radiance: Option<StorageImageIndex>,
        hook: &mut dyn FnMut(&mut TransparencyPass<'_, '_>),
    ) -> Result<(Arc<orr_gpu::Texture>, TextureIndex)> {
        self.ensure_updated()?;
        let device = self.device;
        let composite = self.pipelines.get(self.pipelines.own.composite);
        let core = &mut *self.core;
        let window = &core.windows[self.window];
        let list = self.list.as_mut().expect("frame list");
        let history = core
            .histories
            .get_mut(view.history.0 as usize)
            .and_then(Option::as_mut)
            .filter(|h| h.targets.is_some())
            .ok_or(Error::UnknownViewHistory(view.history))?;
        history.arena.enter(ViewStep::Transparency, list);
        let targets = history.targets.as_ref().expect("view targets");
        let extent = targets.sizes.extent;
        let lists = history.lists.as_ref().expect("lists of the view's raster");
        let particle_draws = targets.particle_draws(lists, &history.draws);
        let depth = Arc::clone(&targets.depth.texture);
        let depth_index = targets.depth.sampled;
        let overlay = targets.overlay();
        let (overlay_texture, overlay_index) = (Arc::clone(overlay.texture()), overlay.sampled());
        list.barrier(Stages::ALL, Stages::COLOR_OUT, Hazards::empty());
        list.begin_rendering(&Rendering {
            extent,
            colors: &[ColorAttachment {
                texture: &overlay_texture,
                load: ColorLoad::Clear(ClearColor::Float([0.0; 4])),
                store: true,
            }],
            depth: None,
        });
        list.end_rendering();
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        hook(&mut TransparencyPass {
            cmd: list,
            overlay: &overlay_texture,
            depth: &depth,
            depth_index,
            extent,
            view: view.view,
            scene: view.scene,
            raster: view.raster,
            particle_draws,
            device,
            window,
        });
        if let Some(radiance) = radiance {
            list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
            let root = window.put_root(
                device,
                &CompositeRoot {
                    overlay: overlay_index,
                    radiance,
                    width: extent[0],
                    height: extent[1],
                },
            )?;
            list.dispatch(
                composite,
                root,
                [extent[0].div_ceil(8), extent[1].div_ceil(8), 1],
            );
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        } else {
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        }
        self.step("transparency");
        Ok((overlay_texture, overlay_index))
    }

    /// The frame's command list and the arena of view `history`, as its last
    /// `raster_view_with` laid it out, for the passes of the view's render after its raster
    /// half: they enter their steps ([`ViewArena::enter`]) and take their targets from it.
    pub fn list_and_arena(
        &mut self,
        history: ViewHistoryId,
    ) -> Result<(&mut CommandList<'a>, &mut ViewArena)> {
        self.ensure_updated()?;
        let arena = &mut self
            .core
            .histories
            .get_mut(history.0 as usize)
            .and_then(Option::as_mut)
            .ok_or(Error::UnknownViewHistory(history))?
            .arena;
        Ok((self.list.as_mut().expect("frame list"), arena))
    }

    /// [`Self::list_and_arena`] with [`Frame::output_list`] in place of the frame's list, for
    /// the view's passes from ray reconstruction on.
    pub fn output_list_and_arena(
        &mut self,
        history: ViewHistoryId,
    ) -> Result<(&mut CommandList<'a>, &mut ViewArena)> {
        self.output_list()?;
        let arena = &mut self
            .core
            .histories
            .get_mut(history.0 as usize)
            .and_then(Option::as_mut)
            .ok_or(Error::UnknownViewHistory(history))?
            .arena;
        let list = match self.output.as_mut() {
            Some(output) => output,
            None => self.list.as_mut().expect("frame list"),
        };
        Ok((list, arena))
    }

    /// Step 6: submits the frame, its render work after `wait` and then its output list, and
    /// returns the point its work signals.
    pub fn end(self, wait: &[TimelinePoint]) -> Result<TimelinePoint> {
        let points = self.end_submits(wait)?;
        Ok(*points.last().expect("a frame submits"))
    }

    /// [`Frame::end`], returning every submit of the frame in order; the last completes
    /// after the others.
    pub fn end_submits(mut self, wait: &[TimelinePoint]) -> Result<Vec<TimelinePoint>> {
        self.ensure_updated()?;
        let render = self.device.queue(QueueKind::Render).expect("render queue");
        let mut list = self.list.take().expect("frame list");
        self.core
            .mounts
            .streaming
            .record_feedback(&mut list, self.id);
        let mut waits = wait.to_vec();
        waits.append(&mut self.loading);
        waits.extend(self.after_output);
        let mut points = std::mem::take(&mut self.submits);
        points.push(render.submit(vec![list], &waits)?);
        if let Some(output) = self.output.take() {
            let compute = self
                .device
                .queue(QueueKind::Compute)
                .expect("compute queue");
            let point = compute.submit(vec![output], &points[points.len() - 1..])?;
            points.push(point);
            self.core.output = Some(point);
        }
        self.core
            .preprocess
            .end_frame(*points.last().expect("a frame submits"))?;
        self.marker(LatencyMarker::RenderSubmitEnd);
        render.mark_frame_end(self.id)?;
        self.step("end");
        let used = self.core.windows[self.window].used();
        self.core.high_water = self.core.high_water.max(used);
        let stats = FrameStats {
            frame: self.id,
            gpu: Vec::new(),
            cpu: std::mem::take(&mut self.cpu),
            upload_bytes: used,
            upload_high_water: self.core.high_water,
            scatter_records: self.scatter.records,
            scatter_words: self.scatter.words,
            scatter_dispatches: self.scatter.dispatches,
            scenes_updated: self.listed.len() as u32,
            rt: RtStats {
                masks_refused: self.masks_refused,
                ..RtStats::default()
            },
            producers: ProducerStats::default(),
            cuts: CutStats::default(),
            skin: SkinStats::default(),
            tessellation: TessellationStats::default(),
            // `begin_frame` alone steps the scale, so it is the one this frame recorded at.
            detail: DetailStats {
                memory_scale: self.core.memory.scale(),
                rt_scale: 1.0,
                ..DetailStats::default()
            },
        };
        self.core.in_flight.push_back(InFlight {
            id: self.id,
            points: points.clone(),
            stats,
            traced: std::mem::take(&mut self.traced),
            updated: if self.updated {
                std::mem::take(&mut self.listed)
            } else {
                Vec::new()
            },
        });
        Ok(points)
    }
}

/// Builds the producer entries of the listed scenes that build this frame, then copies each
/// scene's producer counts and states back for [`FrameStats::producers`].
fn record_producers(
    device: &Device,
    core: &Core,
    listed: &[SceneId],
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    pipelines: &Pipelines,
    frame: u64,
) -> Result<()> {
    let slot = (frame % u64::from(core.fif)) as usize;
    let mut scenes = Vec::with_capacity(listed.len());
    for &id in listed {
        let scene = core.scene(id)?;
        scenes.push((scene, scene.cpu.lock().unwrap()));
    }
    let mut tess_jobs = Vec::new();
    for (&id, (scene, cpu)) in listed.iter().zip(&scenes) {
        if !cpu.producers.iter().flatten().any(|p| p.tess.is_some()) {
            continue;
        }
        let focus = cpu.focus_detail().map(|(f, ppr)| {
            let w = WorldPos::new(f, core.cell_log2);
            (w.cell.extend(1).to_array(), w.offset.extend(ppr).to_array())
        });
        let seen = match cpu.seen_by {
            Some(s) if s.frame + 1 == frame => seen_by(device, core, window, id, &s)?,
            _ => None,
        };
        for (i, p) in cpu.producers.iter().enumerate() {
            let Some(tess) = p.as_ref().and_then(|p| p.tess) else {
                continue;
            };
            tess_jobs.push(tessellate::Job {
                scene: scene.tables.root.gpu(),
                entry: i as u32,
                tess,
                state: scene.tables.tess_state.gpu().offset(i as u64),
                focus,
                seen,
                subd: p.as_ref().and_then(|p| p.subd),
            });
        }
    }
    let jobs: Vec<Job<'_>> = scenes
        .iter()
        .flat_map(|(scene, cpu)| {
            cpu.producers
                .iter()
                .enumerate()
                .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)))
                .filter(|(_, p)| p.builds())
                .map(|(i, producer)| Job {
                    scene: scene.tables.root.gpu(),
                    entry: i as u32,
                    producer,
                })
        })
        .collect();
    if !tess_jobs.is_empty() || !jobs.is_empty() {
        // One scratch for the frame: a block replaced while this list records would be
        // freed before the list runs.
        let templates = if tess_jobs.is_empty() {
            0
        } else {
            core.patterns.scratch(device)
        };
        let need = producer::scratch_bytes(device, &jobs)
            .max(templates)
            .max(256);
        let mut held = core.producer_scratch.lock().unwrap();
        if held.as_ref().is_none_or(|a| a.size() < need) {
            *held =
                Some(device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, need, "producer scratch")?);
        }
        let scratch = held.as_ref().expect("made above").gpu().address();
        tessellate::record(
            device,
            window,
            list,
            pipelines,
            &core.tess,
            &core.patterns,
            &tess_jobs,
            core.dice_pixels,
            scratch,
        )?;
        if !jobs.is_empty() {
            producer::record(
                device,
                window,
                list,
                pipelines,
                &core.producers,
                &jobs,
                scratch,
            )?;
        }
    }
    drop(jobs);
    for (scene, cpu) in &mut scenes {
        let capacity = scene.desc.producer_capacity;
        if capacity == 0 {
            continue;
        }
        for p in cpu.producers.iter_mut().flatten() {
            p.templates_built = true;
            p.built = true;
        }
        let set = cpu
            .producers
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                p.as_ref()
                    .map(|p| (i as u32, p.capacity(), p.tess.is_some()))
            })
            .collect();
        if cpu.producer_readback.is_none() {
            cpu.producer_readback = Some(Readback::new(device, capacity, core.fif)?);
        }
        let t = &scene.tables;
        cpu.producer_readback.as_mut().expect("made above").record(
            list,
            slot,
            t.producer_counts.gpu(),
            t.producer_state.gpu(),
            t.tess_state.gpu(),
            set,
        );
    }
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
    Ok(())
}

/// Last frame's first view of scene `id` as the tessellator reads it, from `seen`: its
/// constants and, while its history's HZB still holds that render, the HZB.
fn seen_by(
    device: &Device,
    core: &Core,
    window: &UploadWindow,
    id: SceneId,
    seen: &crate::scene::LastView,
) -> Result<Option<SeenBy>> {
    let Some(history) = core
        .histories
        .get(seen.history.0 as usize)
        .and_then(Option::as_ref)
    else {
        return Ok(None);
    };
    let raster = match &history.targets {
        Some(t)
            if t.hzb_view.is_some_and(|(scene, _)| scene == id)
                && t.sizes.extent == seen.constants.extent =>
        {
            window.put(device, &t.hzb_record())?
        }
        _ => GpuPtr::NULL,
    };
    Ok(Some(SeenBy {
        view: window.put(device, &seen.constants)?,
        raster,
    }))
}

impl Drop for Scenes {
    fn drop(&mut self) {
        self.loading.send(Msg::Stop);
        self.loading.stop_workers();
        let ended = self.loading_thread.take().map(JoinHandle::join);
        self.core.mounts.clear(&self.core.materials);
        // A panic of the loading thread is the scenes'. Raising it while another unwinds would
        // abort, and the panic hook printed it when it happened.
        if let Some(Err(panic)) = ended
            && !std::thread::panicking()
        {
            std::panic::resume_unwind(panic);
        }
    }
}
