//! `Scenes`, the frame and its fixed order.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use orr_gpu::{
    Allocation, ClearColor, ColorAttachment, ColorLoad, CommandList, ComputePipeline,
    DEFAULT_ENVIRONMENT, DEFAULT_MEDIUM, Device, GpuAddress, GpuPtr, Hazards, KernelSet,
    LatencyMarker, LowLatency, MemoryKind, MeshPipeline, PassResult, QueueKind, Rendering,
    SamplerDesc, SamplerIndex, ShaderModule, SpecData, Stages, StorageImageIndex, TextureIndex,
    TimelinePoint,
};
use orr_math::{DVec3, Vec3};
use orr_pack::AssetId;
use orr_shaders::PathEntry;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::accel::{AccelKernels, AccelLimits, AccelShared};
use crate::arena::{ArenaReport, ArenaRequest, ViewArena, ViewStep};
use crate::clusters::ClusterTable;
use crate::decals::{self, build_bins};
use crate::detail::{Detail, Sample};
use crate::drawn::OpaqueHook;
use crate::features::SceneFeatures;
use crate::lights::LightTables;
use crate::loading::{self, Loading, Msg};
use crate::material::{MaterialId, MaterialTable, MaterialWriter};
use crate::medium_proxy::{ProxyBlas, VolumeRef};
use crate::memory::{PreprocessArenas, SceneMemory, UploadWindow};
use crate::mount::{
    Apply, ClipRef, InstancePage, LoadState, MountId, MountTask, Mounts, PackSource, PageId,
    SkeletonRef,
};
use crate::neural::{NeuralTextures, Precision};
use crate::pipelines::{MeshPipelineId, MeshStages, PipelineId, Pipelines, ReloadReport};
use crate::producer::{
    self, Job, ProducerId, ProducerKernels, ProducerStats, ProducerUpdate, Readback,
};
use crate::ptlas::{FrameInputs, PtlasPipelines, RtStats};
use crate::raster::{self, ListSizes, RasterPipelines, TargetSizes, ViewLists, ViewTargets};
#[cfg(feature = "hot-reload")]
use crate::reload::Reloader;
use crate::residency::{HOLD_FRAMES, RequestSource, ResidencyStats, response_period};
use crate::scene::{
    GroupWriter, ModelCpu, ModelId, Scene, SceneCpu, SceneDesc, SceneId, SceneWriter,
};
use crate::shaders::transparency::CompositeRoot;
use crate::shaders::{animation, emissive, instance_page, page_motion, world as world_kernel};
use crate::skin::{SkinFrame, SkinKernels, SkinStats};
use crate::stream::pool_block;
use crate::subd::ModelCage;
use crate::tessellate::{self, Patterns, TessKernels, TessellationStats};
use crate::time::FrameTime;
use crate::transparency::TransparencyPass;
use crate::traverse::{CutInputs, CutPipelines, CutSizes, CutStats, Cuts, PageFeedback};
use crate::upload::{Record, scatter};
use crate::view::{RasterView, ViewDesc, ViewHistory, ViewHistoryId, view_constants};
use crate::work::{DetailTarget, Floors};
use crate::world::{WorldPos, needs_rebase, origin_for};
mod probe;

use crate::{
    BlobId, BlobTarget, EditDesc, EditId, EditRef, Error, ExposureState, ProducerBatch, Result,
    SceneRoot, SeenView, TextureStreamGpu,
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
///
/// The per-frame work lists have no size here. A view's visible clusters, a scene's traversal
/// queues and raster candidate cut, the tables of the ray-tracing cuts it builds and the
/// records of texture feedback each hold what the content can ask of them in a frame: the
/// hierarchy nodes, cluster groups and clusters of the scene's copies, the clusters its
/// producers emit, the feedback cells of the streamed textures. A scene that asks less than
/// its views are estimated to drops nothing from its first frame. Past the estimate a list
/// holds it at least, grows a few frames after a frame wanted more than it holds, and gives
/// back what it grew by after a long run of frames that want far less. The estimate takes a
/// cluster to cover the square of the error it is drawn at,
/// [`RendererDesc::raster_error_pixels`] on a side: a view's visible clusters over the pixels
/// of its render extent, 2^25 at most, the clusters a visibility buffer word can name; a
/// scene's traversal over the pixels of the sphere of directions around its focus at the
/// detail its views ask for, since it chooses geometry whatever they face, beside the roots
/// and the one candidate item every copy asks however coarse it draws; texture feedback a
/// raise for each pixel that reported in the frame before.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budgets {
    /// Bytes per upload block; a window that outgrows its blocks takes another.
    pub upload_block: u64,
    /// Mesh pages, hierarchy pages and finer cached cuts a scene's traversal asks the loading
    /// side for in a frame, at most: those of the largest projected errors. The others wait
    /// for a later frame.
    pub traversal_requests: u32,
    /// Render pixels, at the scene's detail and an error of one pixel, to each CLAS the
    /// tables of a scene's built ray-tracing cuts hold room for before the frames have told
    /// what they want, over the sphere of directions around its focus; a scene whose copies
    /// can gather fewer holds those. The BLAS they hold room for are the copies that tile
    /// that sphere at [`RendererDesc::rt_merge_pixels`], under which a copy gathers no cut.
    /// Above 0.
    ///
    /// The gathered cuts hold far fewer clusters than the raster's candidates, by a factor
    /// the scene sets and nothing the renderer holds gives: only copies near the focus gather,
    /// and copies with equal cuts share one. A CLAS and the BLAS over it take many times the
    /// device memory of a candidate, so the tables do not start at the raster's estimate of a
    /// cluster to the error's square either. This value stands in for that factor. Lower it
    /// where a scene's first frames or a camera cut leave copies tracing cached cuts
    /// ([`CutStats::rt_fallback`]) for want of CLAS ([`CutStats::rt_wanted_clas`] over
    /// `rt_clas_capacity`).
    ///
    /// While a frame wants more than the tables hold, its cuts build nearest the focus first,
    /// up to the first whose CLAS or BLAS find no room left: the copies of that cut and of the
    /// cuts after it trace cached cuts. A copy is as far as its model's bounds, so every copy
    /// whose bounds hold the focus is at distance 0; cuts equally far go in the order of their
    /// first copies' instance slots. The same copies build every frame of a still view.
    pub rt_cut_pixels: f32,
    /// Device memory a scene's tables of built ray-tracing cuts and their BLAS arenas grow
    /// to at most, past what the content or `rt_cut_pixels` gives them, which they take
    /// whatever it is. Past it the gathered cuts coarsen evenly until their CLAS fit
    /// ([`DetailStats::rt_scale`]), and builds past what fits leave the farthest copies their
    /// cached cuts.
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
    /// as they need. The resident cluster table names an id for each 156-byte cluster record
    /// the pool can hold, 24 B each, bound as ids are taken. While the frames want more than
    /// the pool's budget holds, the error thresholds geometry is chosen by coarsen evenly until
    /// they want what it holds ([`DetailStats`]).
    pub mesh_page_pool: PoolSize,
    /// Bytes of pages a DAG mesh keeps resident from mount; the pages of a larger one past
    /// its pinned coarsest pages stream on the traversal's requests.
    pub mesh_resident_max: u64,
    /// Resident textures: whole textures, streamed textures' mip tails and tiles.
    pub texture_pool: PoolSize,
    /// Bytes of resident instance pages.
    pub instance_page_pool: u64,
    /// Bytes of requested blobs loading or landed and not released.
    pub user_blob_pool: u64,
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
    /// Bytes per block of the streamed pools and the resident cluster table, each block device
    /// memory of its own that goes back to the system once eviction empties it. Where the pools
    /// and the table at their full budgets would take more than a quarter of the device's
    /// `Limits::max_memory_allocations` in blocks this large, they take blocks of the next
    /// power of two that keeps them within it.
    pub stream_block: u64,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            upload_block: 16 << 20,
            traversal_requests: 4096,
            rt_cut_pixels: 32.0,
            rt_cut_memory: PoolSize::HeapPercent(10),
            edit_triangles: 1 << 18,
            stream_clas_triangles: 1 << 18,
            micromap_pages: 64,
            ptlas_rebuilt: 250_000,
            mesh_page_pool: PoolSize::HeapPercent(40),
            mesh_resident_max: 1 << 20,
            texture_pool: PoolSize::HeapPercent(30),
            instance_page_pool: 256 << 20,
            user_blob_pool: 256 << 20,
            in_flight_bytes: 256 << 20,
            stream_block: 16 << 20,
        }
    }
}

/// Farther models a representation chain holds: the length of `ModelGpu::chain`.
pub(crate) const MAX_CHAIN: usize = 3;

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
    /// [`DetailStats::visible_scale`].
    pub raster_error_pixels: f32,
    /// Ray-tracing LOD: the largest cluster error traced, in render pixels at the scene focus,
    /// times [`DetailStats::visible_scale`] and, in a gathered cut, its scene's
    /// [`DetailStats::rt_scale`].
    pub rt_error_pixels: f32,
    /// Where no view saw a group's bounds in its last render of the scene (outside its
    /// frustum or behind the depth it drew), raster and ray tracing take this many times
    /// their thresholds, times [`DetailStats::hidden_scale`]; so do the ray-tracing thresholds
    /// of copies no view drew last frame. At least 1.
    pub culled_error_scale: f32,
    /// How far the mesh page pool's detail scale ([`DetailStats::memory_scale`]) coarsens
    /// hidden geometry before visible geometry: hidden thresholds take up to this much of it
    /// ([`DetailStats::hidden_scale`]), visible ones the rest. At least 1; 1 coarsens both
    /// alike.
    pub hidden_detail_lead: f32,
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
            hidden_detail_lead: 4.0,
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
    /// detail. It is `visible_scale` times `hidden_scale`.
    pub memory_scale: f32,
    /// The part of `memory_scale` every threshold takes, over visible geometry too.
    pub visible_scale: f32,
    /// The part of `memory_scale` only the thresholds of hidden geometry take: up to
    /// [`RendererDesc::hidden_detail_lead`].
    pub hidden_scale: f32,
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
    /// The bytes beside `mesh_demand` of the resident pages that ray-tracing cuts, built or
    /// building, lock and the frame did not read, with what they depend on: they stay while
    /// copies trace the cuts, and leave the pool with the cuts.
    pub mesh_held: u64,
    /// The bytes of `mesh_demand` no detail sheds (pinned and locked pages and what they
    /// depend on) past `mesh_budget`. While it is above 0 the scales rise only until the frames
    /// want nothing else, and the pool stays over its budget by this much.
    pub mesh_shortfall: u64,
}

/// Scene compute: a listed scene's turn in `update_scenes`, after its uploads have landed.
pub struct SceneCompute<'a, 'q> {
    pub cmd: &'a mut CommandList<'q>,
    pub scene: SceneId,
    pub root: GpuPtr<SceneRoot>,
    /// The scene's producer batches, by producer entry ([`Self::producer_entry`]).
    pub batches: GpuPtr<ProducerBatch>,
    pub time: FrameTime,
    /// The scene focus and the render pixels per radian this update chooses geometry for.
    pub focus: Option<(DVec3, f32)>,
    /// What `root` says the scene's light tables hold.
    pub light_tables: LightTables,
    /// What the scene may hold this frame.
    pub features: SceneFeatures,
    device: &'a Device,
    window: &'a UploadWindow,
    pipelines: &'a Pipelines,
    entries: &'a Scene,
}

impl<'a> SceneCompute<'a, '_> {
    /// The entry of the scene's producer table that producer `id` holds while it is set, as
    /// [`SceneWriter::producer_entry`] gives it.
    pub fn producer_entry(&self, id: ProducerId) -> Option<u32> {
        self.entries.producer_entry(id)
    }

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
    hidden_detail_lead: f32,
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
    /// The pixel probe's pipelines, made at the first probe.
    probe: Option<[ComputePipeline; 2]>,
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
        assert!(b.rt_cut_pixels > 0.0, "rt_cut_pixels is above 0");
        assert!(
            desc.culled_error_scale >= 1.0 && desc.hidden_detail_lead >= 1.0,
            "culled_error_scale and hidden_detail_lead are at least 1"
        );
        let pools = PoolBytes::new(device, b);
        let accel = Arc::new(AccelShared {
            kernels: AccelKernels::new(device, desc.kernels)?,
            limits: AccelLimits {
                clas_triangles: b.stream_clas_triangles,
                micromaps: b.micromap_pages,
                mesh_resident_max: b.mesh_resident_max,
            },
            clusters: ClusterTable::new(device, pools.mesh, pool_block(device, b, pools))?,
        });
        let materials = MaterialTable::new(device)?;
        let mounts = Mounts::new(
            device,
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
                hidden_detail_lead: desc.hidden_detail_lead,
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
                probe: None,
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

    /// A mesh pipeline of `module`'s `stages`, created on the calling thread, as
    /// [`Self::compute_pipeline`] creates a compute one: hot reload replaces it with the
    /// module's other pipelines, so the stages of a drawn entry's draw change in the frame
    /// the resolve's linked geometry does. Look it up each frame with
    /// [`Frame::mesh_pipeline`].
    pub fn mesh_pipeline(
        &mut self,
        module: &'static ShaderModule,
        stages: &MeshStages<'_>,
    ) -> Result<MeshPipelineId> {
        let id = self
            .pipelines
            .create_mesh(&self.device, self.kernels, module, stages)?;
        #[cfg(feature = "hot-reload")]
        self.reloader
            .watch(self.pipelines.recipes.last().expect("just created").module);
        Ok(id)
    }

    /// The loading side's job queue, for the threads the consumer lends to it.
    pub fn loading(&self) -> Arc<Loading> {
        Arc::clone(&self.loading)
    }

    /// What the loading side, the queues and the streamed pools hold, for reporting a mount or
    /// a load that does not land.
    pub fn loading_report(&self) -> String {
        use std::fmt::Write;
        let d = &self.device;
        let mut out = format!("device lost {}\n", d.is_lost());
        let kinds = [
            QueueKind::Render,
            QueueKind::Compute,
            QueueKind::Loading,
            QueueKind::Copy,
            QueueKind::Present,
        ];
        for kind in kinds {
            if let Some(q) = d.queue(kind) {
                let _ = writeln!(out, "{kind:?} queue completed {}", q.completed());
            }
        }
        out += &self.loading.report(d);
        let _ = writeln!(
            out,
            "resident cluster table: {}",
            self.core.mounts.clusters().report()
        );
        let _ = writeln!(out, "{:?}", self.residency());
        out
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
    /// their [`InstanceRange`]s stay the consumer's; a subdivision copy's [`ProducerId`] names
    /// nothing until it is set again.
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
        for scene in core.scenes.iter().flatten() {
            scene.cpu.lock().unwrap().models_changed();
        }
        for model in ended {
            core.chains.remove(&model);
            if !gone.contains(&model) {
                core.write_chain(model, &[]);
                // New copies take entries for the model alone; live ones keep theirs.
                let m = &mut core.model_cpu[model.0 as usize];
                m.emissive_stride = m.emissive_sections;
                m.chained = false;
            }
        }
    }

    /// Starts an edit of a mesh: a new model the consumer's kernels change at the scene
    /// compute point (see [`EditRef`]). Its pages read on the loading side, and
    /// [`Scenes::edit`] succeeds once [`Scenes::edit_state`] is `Ready`.
    pub fn create_edit(&mut self, desc: &EditDesc<'_>) -> Result<EditId> {
        self.core.mounts.create_edit(desc, &self.loading)
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
        for scene in core.scenes.iter().flatten() {
            scene.cpu.lock().unwrap().models_changed();
        }
    }

    /// Gives every copy of `model` up to 3 farther models, each with the distance from the
    /// focus in m at which copies switch to it, ascending; empty: the model at every
    /// distance. Copies take their step from the next frame on, and a copy that changes step
    /// cross-fades between the two over a few frames. Steps past the model itself draw and
    /// trace rigid, with their own materials, every section shown and no pose. A chain ends
    /// when a pack holding one of its models is unmounted. Each copy holds an emissive list
    /// entry per emissive section of its chain's most emissive model; a chain that needs more
    /// gives every copy of the model in every scene more, on pages too, and fails with
    /// [`Error::Capacity`], changing nothing, where a scene's list would pass what it can hold
    /// or its PTLAS part pool has no room for the far instances of the model's copies.
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
        // Copies placed while the model has a chain count their far instances as they are
        // placed; the others count them when it gains one.
        let gains_far = !chain.is_empty() && !core.model_cpu[model.0 as usize].chained;
        if gains_far {
            for scene in core.scenes.iter().flatten() {
                scene.check_chain_far(model.0)?;
            }
        }
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
                match scene.reserve_entries(&self.device, model.0, stride, page_strides) {
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
        for scene in core.scenes.iter().flatten() {
            scene.cpu.lock().unwrap().models_changed();
        }
        if gains_far {
            for scene in core.scenes.iter().flatten() {
                scene.chain_far(model.0);
            }
        }
        core.model_cpu[model.0 as usize].chained = !chain.is_empty();
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

    /// Lowers the budget of `class`'s streamed units to `bytes`, at most what it is: the
    /// next step 1 evicts down to it, and the pools give back the blocks that empties.
    pub fn lower_budget(&mut self, class: crate::ResidencyClass, bytes: u64) {
        self.core.mounts.streaming.lower_budget(class, bytes);
    }

    /// What the streamed pools, the CLAS pool and the frames' upload windows hold, gathered
    /// on demand; it can wait for a streamed CLAS build to submit.
    pub fn memory(&self) -> SceneMemory {
        let mut out = SceneMemory {
            windows: self.core.windows.iter().map(UploadWindow::fill).collect(),
            ..SceneMemory::default()
        };
        self.core.mounts.streaming.memory(&mut out);
        out.pools.push(self.core.mounts.clusters().fill());
        out
    }

    /// A resident instance page.
    pub fn page(&self, id: PageId) -> Option<&InstancePage> {
        self.core.mounts.page(id)
    }

    /// The entry of `scene`'s producer table that producer `id` holds while it is set, as
    /// [`SceneWriter::producer_entry`] gives it.
    pub fn producer_entry(&self, scene: SceneId, id: ProducerId) -> Result<Option<u32>> {
        Ok(self.core.scene(scene)?.producer_entry(id))
    }

    /// Holds back the binds that finish tile moves, as a loading thread busy binding does, or
    /// lets them go.
    #[cfg(feature = "test-hooks")]
    pub fn hold_tile_binds(&mut self, hold: bool) {
        self.core.mounts.streaming.hold_tile_binds(hold);
    }

    /// Fails the scratch allocations of tile moves as a device out of memory does, or lets
    /// them be.
    #[cfg(feature = "test-hooks")]
    pub fn fail_tile_scratch(&mut self, fail: bool) {
        self.core.mounts.streaming.fail_tile_scratch(fail);
    }

    /// Holds back the binds that move CLAS pool pages to other memory, as a loading side
    /// holding the pool does, or lets them go.
    #[cfg(feature = "test-hooks")]
    pub fn hold_clas_binds(&mut self, hold: bool) {
        self.core.mounts.streaming.hold_clas_binds(hold);
    }

    /// Has the memory of the CLAS pool's pages empty every block the others have room for,
    /// as it does while its heap is over its mark, or lets the heap decide.
    #[cfg(feature = "test-hooks")]
    pub fn plan_clas_memory_tight(&mut self, tight: bool) {
        self.core.mounts.streaming.plan_clas_memory_tight(tight);
    }

    #[cfg(feature = "testing")]
    pub(crate) fn texture_object(&self, m: MountId, id: AssetId) -> Option<Arc<orr_gpu::Texture>> {
        self.core.mounts.texture_object(m, id)
    }

    #[cfg(feature = "testing")]
    pub(crate) fn model_materials(&self) -> GpuPtr<u32> {
        self.core.mounts.model_materials.gpu()
    }

    #[cfg(feature = "testing")]
    pub(crate) fn mesh_page_tables(&self) -> (GpuPtr<u64>, GpuPtr<u32>, u32) {
        let pages = &self.core.mounts.mesh_pages;
        (pages.table.gpu(), pages.bases.gpu(), pages.entries())
    }

    #[cfg(feature = "testing")]
    pub(crate) fn table_entries(&self, scene: SceneId) -> Result<crate::testing::TableEntries> {
        let t = &self.core.scene(scene)?.tables;
        Ok(crate::testing::TableEntries {
            slots: t.instances.capacity(),
            groups: t.groups.capacity(),
            material_sets: t.material_sets.capacity(),
            page_material_sets: t.page_sets.capacity(),
            skinned: t.skinned.capacity(),
        })
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
        let history = ViewHistory::new(exposure, arena, readback);
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
        // A later history may take its id.
        for scene in self.core.scenes.iter().flatten() {
            scene
                .cpu
                .lock()
                .unwrap()
                .seen_by
                .retain(|s| s.history != id);
        }
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
                f.stats.detail.mesh_held = mesh.held;
                f.stats.detail.mesh_shortfall =
                    (mesh.bytes - mesh.streamed).saturating_sub(mesh.budget);
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
            let removed = std::mem::take(&mut cpu.producers_removed);
            for (frame, first, count) in removed {
                if frame <= completed {
                    scene.free_producers(cpu, first, count);
                } else {
                    cpu.producers_removed.push((frame, first, count));
                }
            }
            cpu.retired.retain(|(frame, _)| *frame > completed);
            cpu.groups_removed.retain(|(frame, _)| *frame > completed);
            let sets = &scene.tables.page_sets;
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
            let mut freed = Vec::new();
            cpu.skins_removed.retain(|&(frame, first, count, _)| {
                let done = frame <= completed;
                if done {
                    freed.push((first, count));
                }
                !done
            });
            for (first, count) in freed {
                scene.free_skinned(cpu, first, count);
            }
            cpu.ptlas.retire(completed);
            scene.retire(cpu, completed);
        }
        core.mounts.retire(completed, &core.materials);
        core.materials.table.retire(completed);
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
        // The views of the frame before, whose pixels report texture feedback.
        let last = frame - 1;
        let pixels = core
            .histories
            .iter()
            .flatten()
            .filter(|h| h.rendered[(last % u64::from(core.fif)) as usize] == Some(last))
            .filter_map(|h| h.previous)
            .map(|v| u64::from(v.render_extent.x) * u64::from(v.render_extent.y))
            .sum();
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
            pixels,
            core.output,
            &self.loading,
            &mut apply,
        )?);
        loading_points.extend(core.mounts.clusters().bind(&self.device)?);
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
        for (i, moves) in core.mounts.streaming.moves().iter().enumerate() {
            // A later step's copies may read what an earlier step's wrote.
            if i > 0 {
                list.barrier(Stages::TRANSFER, Stages::TRANSFER, Hazards::empty());
            }
            for m in &moves.units {
                // SAFETY: both slots are the streamed pools', held until this frame has
                // completed.
                let (dst, src) = unsafe {
                    (
                        GpuPtr::from_address(GpuAddress(m.dst)),
                        GpuPtr::from_address(GpuAddress(m.src)),
                    )
                };
                list.copy(dst, src, m.bytes);
            }
            for t in &moves.tiles {
                let s = &t.stream;
                let tile = s.grid.tile(t.tile);
                let region = s
                    .layout
                    .tile_region(s.texture.desc(), tile.mip, tile.layer, tile.xy);
                // SAFETY: the tile's place in the scratch its move holds until this frame has
                // completed.
                let at = unsafe { GpuPtr::from_address(GpuAddress(t.at)) };
                if t.out {
                    list.copy_from_texture(at, &s.texture, &region);
                } else {
                    list.copy_to_texture(&s.texture, &region, at);
                }
            }
            list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
        }
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

    /// What the scene of `cpu` may hold now, under kernels that link `environment`: what any
    /// scene may, what it holds itself, and what the two decide together.
    fn scene_features(&self, cpu: &mut SceneCpu, environment: &str) -> SceneFeatures {
        let mut features = self.features() | cpu.features();
        let work = cpu.work(&self.model_cpu, &self.chains, |m| self.mounts.is_live(m));
        let traced = cpu.producers.iter().flatten().any(|p| p.casts);
        if work.caster_copies > 0 || (traced && self.materials.casts()) {
            features |= SceneFeatures::CASTERS;
        }
        // The default map without data is black.
        if environment != DEFAULT_ENVIRONMENT || cpu.environment.is_some_and(|data| data != 0) {
            features |= SceneFeatures::ENVIRONMENT;
        }
        features
    }

    /// The memory scale's parts ([`DetailStats::visible_scale`], `hidden_scale`): hidden
    /// geometry takes up to the lead first.
    fn detail_scales(&self) -> (f32, f32) {
        let m = self.memory.scale();
        let hidden = m.min(self.hidden_detail_lead);
        (m / hidden, hidden)
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

    /// The current mesh pipeline of `id`.
    pub fn mesh_pipeline(&self, id: MeshPipelineId) -> &'a MeshPipeline {
        self.pipelines.mesh(id)
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
                rebased.push((scene.tables.root.gpu(), scene.tables.lights.capacity()));
                rebase = true;
            }
            let producers = cpu.producers.iter().any(Option::is_some);
            let tlas = cpu.ptlas.tlas(device, core.fif, producers)?;
            let layout = cpu.ptlas.layout();
            let work = cpu.work(&core.model_cpu, &core.chains, |m| core.mounts.is_live(m));
            let detail = DetailTarget {
                pixels_per_radian: cpu.pixels_per_radian,
                raster_pixels: core.raster_error_pixels,
                rt_pixels: core.rt_error_pixels,
                merge_pixels: core.rt_merge_pixels,
                rt_cut_pixels: core.budgets.rt_cut_pixels,
            };
            cpu.sized = Some((work, detail));
            let floors = detail.floors(&work);
            if cpu.cuts.is_none() {
                let heap = device.memory_report().heap_of(MemoryKind::DeviceLocal).size;
                let sizes = CutSizes {
                    slots: cpu.slot_capacity(),
                    rt_bytes: core.budgets.rt_cut_memory.bytes(heap),
                    models: core.mounts.models.capacity(),
                    entries: core.mounts.mesh_pages.entries(),
                    requests: core.budgets.traversal_requests,
                };
                // Lists of nothing where the device lacks the memory: `fit` says so and tries
                // again.
                let cuts = match Cuts::new(device, sizes, core.fif, floors) {
                    Err(e) if e.is_out_of_memory() => {
                        Cuts::new(device, sizes, core.fif, Floors::default())?
                    }
                    cuts => cuts?,
                };
                cpu.cuts = Some(cuts);
            }
            let cuts = cpu.cuts.as_mut().expect("made above");
            cuts.follow_models(device, core.mounts.models.capacity())?;
            cuts.follow_entries(device, core.mounts.mesh_pages.entries())?;
            match cuts.fit(device, self.id, response_period(core.fif), floors) {
                Err(e) if e.is_out_of_memory() => short.push(e),
                r => r?,
            }
            let cut = cuts.gpu();
            if cpu.skinned_high_water > 0 && cpu.skin_frame.is_none() {
                cpu.skin_frame = Some(SkinFrame::new(
                    device,
                    scene.desc.skinned_clusters,
                    scene.tables.skinned.capacity(),
                    core.fif,
                )?);
            }
            let skin_work = cpu.skin_frame.as_ref().map(|f| f.work);
            records.append(&mut cpu.ptlas.partition_records(device, window, &core.seq)?);
            for (index, g) in cpu.groups.iter_mut().enumerate() {
                let Some(g) = g else { continue };
                let before = g.record(index as u32);
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
                let after = g.record(index as u32);
                let landed = g
                    .page
                    .as_mut()
                    .is_some_and(|p| std::mem::take(&mut p.changed));
                if landed || bytemuck::bytes_of(&before) != bytemuck::bytes_of(&after) {
                    let src = window.put(device, &after)?;
                    let dst = scene.tables.groups.entry(index as u32);
                    records.push(Record::values(core.next_seq(), dst, src, 1));
                }
            }
            let t = &scene.tables;
            if cpu.decals_changed {
                let bins = build_bins(&cpu.decals);
                let slots: &[u32] = bytemuck::cast_slice(&bins.slots);
                // Window room and then the tables' come first, so a failure leaves the bins
                // the kernels read as they were.
                let (slots_src, slots_out) = window.slice::<u32>(device, slots.len() as u32)?;
                let (refs_src, refs_out) = window.slice::<u32>(device, bins.refs.len() as u32)?;
                cpu.fit_decal_bins(device, &bins)?;
                slots_out.copy_from_slice(slots);
                refs_out.copy_from_slice(&bins.refs);
                for (dst, src, n) in [
                    (cpu.decal_bins.gpu().cast::<u32>(), slots_src, slots.len()),
                    (cpu.decal_refs.gpu(), refs_src, bins.refs.len()),
                ] {
                    records.push(Record::values(core.next_seq(), dst, src, n as u32));
                }
                cpu.decal_bin_slots = bins.slots.len() as u32;
                cpu.decal_globals = bins.globals;
                cpu.decals_changed = false;
            }
            let root = SceneRoot {
                instances: t.instances.gpu(),
                slot_state: t.slot_state.gpu(),
                groups: t.groups.gpu(),
                models: core.mounts.models.gpu(),
                meshes: core.mounts.meshes.gpu(),
                materials: core.materials.table.gpu(),
                clusters: core.mounts.clusters().gpu(),
                mesh_pages: core.mounts.mesh_pages.table.gpu(),
                mesh_page_bases: core.mounts.mesh_pages.bases.gpu(),
                model_materials: core.mounts.model_materials.gpu(),
                material_sets: t.material_sets.gpu().cast(),
                page_material_sets: t.page_sets.gpu().cast(),
                lights: t.lights.gpu(),
                emissive: t.emissive.gpu(),
                producers: t.producers.gpu(),
                batches: t.batches.gpu(),
                patches: t.patches.gpu(),
                producer_state: t.producer_state.gpu(),
                tess_state: t.tess_state.gpu(),
                sky_stats: t.sky_stats.gpu(),
                environment: cpu.environment.unwrap_or(0),
                medium: cpu.medium.unwrap_or(0),
                surface_data: cpu.surface_data,
                tlas: tlas.unwrap_or_else(|| device.empty_tlas()),
                part_slots: cpu.ptlas.part_slots(),
                origin_cell: cpu.origin_cell.extend(core.cell_log2 as i32).to_array(),
                delta_time: self.time.delta,
                frame_index: self.id as u32,
                skinned_capacity: t.skinned.capacity(),
                light_capacity: t.lights.capacity(),
                producer_capacity: t.producers.capacity(),
                emissive_capacity: t.emissive.capacity(),
                near_range: cpu.near_range,
                part_base: layout.slots,
                part_capacity: layout.parts,
                producer_base: layout.slots + layout.parts,
                cut,
                skinned: t.skinned.gpu(),
                skin_state: t.skin_state.gpu(),
                skin_work: skin_work.unwrap_or(GpuPtr::NULL).cast(),
                decals: t.decals.gpu(),
                decal_bins: cpu.decal_bins.gpu(),
                decal_refs: cpu.decal_refs.gpu(),
                decal_globals: cpu.decal_globals,
                decal_bin_slots: cpu.decal_bin_slots,
                decal_bin_log2: decals::bin_log2(core.cell_log2),
                proxy_base: layout.slots + layout.parts + cpu.ptlas.producer_entries(),
                proxies: t.proxies.gpu(),
                proxy_capacity: t.proxies.capacity(),
                light_base: layout.slots
                    + layout.parts
                    + cpu.ptlas.producer_entries()
                    + cpu.ptlas.proxy_entries(),
                emissive_lists: t.emissive_lists.gpu().cast(),
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
                scene.light_tables(cpu),
                core.scene_features(cpu, self.kernels.environment),
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
        // Tables that grew move before the records that write them.
        let mut moves = Vec::new();
        core.materials.table.land(self.id, &mut moves, &mut records);
        core.mounts.land(self.id, &mut moves, &mut records);
        for &id in &self.listed {
            let scene = core.scene(id)?;
            let cpu = scene.cpu.lock().unwrap();
            scene.land(&cpu, self.id, &mut moves, &mut records);
        }
        if !moves.is_empty() {
            list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
        }
        for &(dst, src, bytes) in &moves {
            list.copy(dst, src, bytes);
        }
        for &(dst, src, bytes) in &copies {
            list.copy(dst, src, bytes);
        }
        if !copies.is_empty() || !moves.is_empty() {
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
            // A scene without lights has none to move.
            for &(root, lights) in rebased.iter().filter(|r| r.1 > 0) {
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
                    exact_rows: p.exact_rows,
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
                let dt = orr_pack::instance::motion_dt(dt);
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
                    dt: dt[0],
                    dt_lo: dt[1],
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
        for &(scene, root, batches, focus, light_tables, features) in &roots {
            let entries = core.scene(scene)?;
            let mut sc = SceneCompute {
                cmd: &mut list,
                scene,
                root,
                batches,
                time: self.time,
                focus,
                light_tables,
                features,
                device,
                window,
                pipelines: self.pipelines,
                entries,
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
        for &(id, root, _, focus, ..) in &roots {
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
        for &(id, root, _, focus, ..) in &roots {
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
                core.raster_error_pixels * core.detail_scales().0,
                cpu.skinned_high_water,
                (self.id % u64::from(core.fif)) as usize,
            )?;
        }
        let hierarchy = core.mounts.hierarchy();
        for &(id, root, _, focus, ..) in &roots {
            let scene = core.scene(id)?;
            let cpu = &mut *scene.cpu.lock().unwrap();
            let focus = focus.map(|(f, ppr)| {
                let w = WorldPos::new(f, core.cell_log2);
                (w.cell.extend(1).to_array(), w.offset.extend(ppr).to_array())
            });
            let (visible, hidden) = core.detail_scales();
            let views = seen_views(
                device,
                &core.histories,
                window,
                id,
                &mut cpu.seen_by,
                self.id,
            )?;
            let inputs = CutInputs {
                scene: root,
                focus,
                raster_pixels: core.raster_error_pixels * visible,
                rt_pixels: core.rt_error_pixels * visible,
                visible_scale: visible,
                culled_scale: core.culled_error_scale * hidden,
                views: views.0,
                view_count: views.1,
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
                    producers: scene.tables.producers.capacity(),
                    budget: core.budgets.ptlas_rebuilt,
                    proxy_blas,
                    slot: (self.id % u64::from(core.fif)) as usize,
                    frame: self.id,
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
            for &(_, scene, _, _, tables, _) in &roots {
                let r = emissive::Root {
                    scene,
                    jobs: GpuPtr::NULL,
                    job_count: 0,
                    _pad0: 0,
                    _pad1: 0,
                };
                let ptr = window.put_root(device, &r)?;
                if tables.producers > 0 {
                    let p = self.pipelines.get(own.emissive_producers);
                    list.dispatch(p, ptr, [tables.producers, 1, 1]);
                }
                if tables.entries > 0 {
                    let groups = tables.entries.div_ceil(64);
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
        self.core.mounts.streaming.submitted();
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

    /// Starts the exposure state of view `history` over at `start`, in both slots. Call it
    /// before the view's [`Self::raster_view`] of this frame, which then reads it. `start`
    /// replaces `prev_pre_exposure` too, so a history in pre-exposed units cannot rescale
    /// across the restart and has to start over on the same frame.
    pub fn restart_exposure(&mut self, history: ViewHistoryId, start: ExposureState) -> Result<()> {
        self.ensure_updated()?;
        let history = self
            .core
            .histories
            .get(history.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(Error::UnknownViewHistory(history))?;
        let window = &self.core.windows[self.window];
        let source = window.put(self.device, &[start; 2])?;
        let list = self.list.as_mut().expect("frame list");
        list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
        list.copy(
            history.exposure.gpu().cast(),
            source.cast(),
            history.exposure.size(),
        );
        list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
        Ok(())
    }

    /// Step 4, the raster half of a view: writes its `ViewConstants`, culls and rasters the
    /// scene into the visibility buffer, resolves the G-buffer, depth, motion and HZB, and
    /// moves its history on. Lays out the view's arena with the raster's transient targets
    /// alone and enters [`ViewStep::Raster`].
    pub fn raster_view(&mut self, view: &ViewDesc) -> Result<RasterView> {
        self.raster_view_with(view, &[], None)
    }

    /// The entry of `scene`'s producer table that producer `id` holds while it is set, as
    /// [`SceneWriter::producer_entry`] gives it.
    pub fn producer_entry(&self, scene: SceneId, id: ProducerId) -> Result<Option<u32>> {
        Ok(self.core.scene(scene)?.producer_entry(id))
    }

    /// Says that an allocation made for the frame beside the scenes' own found no device
    /// memory with `e`, as one of a scene's lighting does: the mesh and texture budgets then
    /// make room where they govern its kind, as they do for the scenes' own.
    pub fn out_of_memory(&mut self, e: &orr_gpu::Error) {
        self.core.mounts.streaming.device_out_of_memory(e);
    }

    /// What `scene` may hold now: [`RasterView::features`] before the view is rastered.
    pub fn scene_features(&self, scene: SceneId) -> Result<SceneFeatures> {
        let cpu = &mut *self.core.scene(scene)?.cpu.lock().unwrap();
        Ok(self.core.scene_features(cpu, self.kernels.environment))
    }

    /// [`Self::raster_view`], with `requests`, the transient targets of the passes that
    /// follow it, laid out in the view's arena beside the raster's, and with `opaque`
    /// drawing the scene's drawn producer entries in each pass of the raster.
    pub fn raster_view_with(
        &mut self,
        view: &ViewDesc,
        requests: &[ArenaRequest],
        opaque: Option<&mut OpaqueHook<'_>>,
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
        let (slots, slot_capacity, smoke, features, light_tables, work, producers, sized) = {
            let cpu = &mut *scene.cpu.lock().unwrap();
            let features = core.scene_features(cpu, self.kernels.environment);
            let smoke = features.contains(SceneFeatures::SMOKE);
            let cuts = cpu.cuts.as_ref().expect("an updated scene has cut tables");
            let producers: Vec<(u32, u32)> = (0..)
                .zip(&cpu.producers)
                .filter_map(|(entry, p)| Some((entry, p.as_ref()?.raster_clusters)))
                .filter(|&(_, clusters)| clusters > 0)
                .collect();
            (
                cpu.ptlas.high_water,
                cpu.slot_capacity(),
                smoke,
                features,
                scene.light_tables(cpu),
                cuts.candidate_capacity(),
                producers,
                cpu.sized.expect("an updated scene has sized its lists"),
            )
        };
        let sizes = TargetSizes {
            extent: view.render_extent.to_array(),
            extended: core.materials.any_extended(),
            bins: core.raster.bins,
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
        let produced = producers.iter().map(|&(_, n)| u64::from(n)).sum();
        let pixels = u64::from(view.render_extent.x) * u64::from(view.render_extent.y);
        history
            .visible
            .floor(sized.1.visible(&sized.0, produced, pixels));
        // The second culling pass takes at most one work item per candidate item.
        let want = ListSizes {
            visible_clusters: history.visible.wanted(),
            work,
        };
        let mut short = None;
        if history.lists.is_none() {
            // Lists of nothing where the device lacks the memory: the resize below says so and
            // tries again.
            let made = match ViewLists::new(device, want) {
                Err(e) if e.is_out_of_memory() => {
                    let none = ListSizes {
                        visible_clusters: 0,
                        work: 0,
                    };
                    ViewLists::new(device, none)?
                }
                made => made?,
            };
            history.lists = Some(made);
        }
        let lists = history.lists.as_mut().expect("lists made above");
        if lists.sizes() != want && self.id >= history.lists_retry {
            match lists.resize(device, want) {
                Ok(()) => {}
                // The view keeps the lists it has while the streaming budgets make room.
                Err(e) if e.is_out_of_memory() => {
                    history.lists_retry = self.id + response_period(core.fif);
                    short = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        history.visible.set(lists.sizes().visible_clusters);
        history.arena.enter(ViewStep::Raster, list);
        match &mut history.targets {
            Some(t) if t.sizes == sizes => t.bind(&history.arena),
            t => {
                let new = ViewTargets::new(device, sizes, slot_capacity, &history.arena, list)?;
                *t = Some(new);
            }
        }
        let lists = history.lists.as_ref().expect("lists made above");
        let targets = history.targets.as_mut().expect("view targets");
        targets.follow_slots(device, slot_capacity, self.id)?;
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
            &producers,
            constants,
            core.software_raster_pixels,
            occlusion,
            preprocess,
            opaque,
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
            light_tables,
        };
        history.draws = draws;
        if let Some(e) = short {
            core.mounts.streaming.out_of_memory(&e);
        }
        let mut cpu = core.scene(view.scene)?.cpu.lock().unwrap();
        let last = crate::scene::LastView {
            history: view.history,
            frame: self.id,
            constants: seen,
        };
        // The views stay in the order of their last renders; a second render in one frame
        // keeps its place, its constants those of the render the HZB holds.
        let views = &mut cpu.seen_by;
        match views.iter().position(|s| s.history == view.history) {
            Some(i) if views[i].frame == self.id => views[i] = last,
            Some(i) => {
                views.remove(i);
                views.push(last);
            }
            None => views.push(last),
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
        self.core.mounts.streaming.submitted();
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
                visible_scale: self.core.detail_scales().0,
                hidden_scale: self.core.detail_scales().1,
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

/// Builds the producer entries of the listed scenes that build this frame, then sums each
/// scene's producer figures and copies them back for [`FrameStats::producers`].
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
        // Last frame's first view of the scene.
        let seen = match cpu.seen_by.iter().find(|s| s.frame + 1 == frame) {
            Some(s) => seen_view(device, &core.histories, window, id, s)?,
            None => None,
        };
        for (i, p) in cpu.producers.iter().enumerate() {
            let Some(tess) = p.as_ref().and_then(|p| p.tess) else {
                continue;
            };
            tess_jobs.push(tessellate::Job {
                scene: scene.tables.root.gpu(),
                entry: i as u32,
                tess,
                state: scene.tables.tess_state.entry(i as u32),
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
    let mut summed = Vec::new();
    for (scene, cpu) in &mut scenes {
        if scene.tables.producers.capacity() == 0 {
            continue;
        }
        for p in cpu.producers.iter_mut().flatten() {
            p.templates_built = true;
            p.built = true;
        }
        let readback = match &mut cpu.producer_readback {
            Some(readback) => readback,
            none => none.insert(Readback::new(device, core.fif)?),
        };
        summed.push((readback, scene.tables.root.gpu()));
    }
    if !summed.is_empty() {
        list.begin_pass("producer figures", false);
        for (readback, scene) in &summed {
            readback.sum(device, window, list, pipelines, &core.producers, *scene)?;
        }
        list.end_pass();
        list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
        for (readback, _) in summed {
            readback.copy(list, slot);
        }
    }
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
    Ok(())
}

/// A render of scene `id` as kernels that test what it saw read it, from `seen`: its constants
/// and, while its history's HZB still holds that render, the HZB. None once the history is
/// gone.
fn seen_view(
    device: &Device,
    histories: &[Option<ViewHistory>],
    window: &UploadWindow,
    id: SceneId,
    seen: &crate::scene::LastView,
) -> Result<Option<SeenView>> {
    let Some(history) = histories
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
    Ok(Some(SeenView {
        view: window.put(device, &seen.constants)?,
        raster,
    }))
}

/// The views the traversal of scene `id` at frame `frame` tests, each history's last render of
/// the scene ([`seen_view`]), and their count. First drops from `seen` the renders of histories
/// that are gone or have rendered another scene since, and those older than the page hold: a
/// view that renders the scene now and then keeps its place between its renders.
fn seen_views(
    device: &Device,
    histories: &[Option<ViewHistory>],
    window: &UploadWindow,
    id: SceneId,
    seen: &mut Vec<crate::scene::LastView>,
    frame: u64,
) -> Result<(GpuPtr<SeenView>, u32)> {
    seen.retain(|s| {
        let Some(h) = histories.get(s.history.0 as usize).and_then(Option::as_ref) else {
            return false;
        };
        let other = h
            .targets
            .as_ref()
            .and_then(|t| t.hzb_view)
            .is_some_and(|(scene, _)| scene != id);
        !other && s.frame + HOLD_FRAMES >= frame
    });
    let mut views = Vec::with_capacity(seen.len());
    for s in seen.iter() {
        views.extend(seen_view(device, histories, window, id, s)?);
    }
    if views.is_empty() {
        return Ok((GpuPtr::NULL, 0));
    }
    let (at, out) = window.slice::<SeenView>(device, views.len() as u32)?;
    out.copy_from_slice(&views);
    Ok((at, views.len() as u32))
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
