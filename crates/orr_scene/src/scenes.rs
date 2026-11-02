//! `Scenes`, the frame and its fixed order.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use glam::{DVec3, Vec3};
use orr_gpu::{
    Allocation, ClearColor, ColorAttachment, ColorLoad, CommandList, ComputePipeline, Device,
    GpuAddress, GpuPtr, Hazards, KernelSet, LatencyMarker, LowLatency, MemoryKind, PassResult,
    QueueKind, Rendering, SamplerDesc, SamplerIndex, ShaderModule, SpecData, Stages,
    StorageImageIndex, TextureIndex, TimelinePoint,
};
use orr_pack::AssetId;
use orr_shaders::PathEntry;
use rustc_hash::FxHashMap;

use crate::accel::{AccelKernels, AccelLimits, AccelShared};
use crate::arena::{ArenaReport, ArenaRequest, ViewArena, ViewStep};
use crate::decals::{self, build_bins};
use crate::emissive::ModelEmission;
use crate::loading::{self, Loading, Msg};
use crate::material::{MaterialId, MaterialTable, MaterialWriter};
use crate::medium_proxy::{ProxyBlas, VolumeRef};
use crate::memory::{PreprocessArenas, RangeAllocator, UploadWindow};
use crate::mount::{
    Apply, ClipRef, InstancePage, LoadState, MountId, MountTask, Mounts, PackSource, PageId,
    SkeletonRef, TableSizes,
};
use crate::neural::{NeuralTextures, Precision};
use crate::pipelines::{PipelineId, Pipelines, ReloadReport};
use crate::producer::{self, Job, ProducerKernels, ProducerStats, ProducerUpdate, Readback};
use crate::ptlas::{FrameInputs, PtlasPipelines, RtStats};
use crate::raster::{self, RasterPipelines, TargetSizes, ViewTargets};
#[cfg(feature = "hot-reload")]
use crate::reload::Reloader;
use crate::residency::ResidencyStats;
use crate::scene::{GroupWriter, ModelCpu, ModelId, Scene, SceneDesc, SceneId, SceneWriter};
use crate::shaders::transparency::CompositeRoot;
use crate::shaders::{animation, emissive, instance_page, world as world_kernel};
use crate::skin::{SkinFrame, SkinKernels, SkinStats};
use crate::subd::ModelCage;
use crate::tessellate::{self, Patterns, SeenBy, TessKernels, TessellationStats};
use crate::time::FrameTime;
use crate::transparency::TransparencyPass;
use crate::traverse::{CutInputs, CutPipelines, CutSizes, CutStats, Cuts};
use crate::upload::{Record, scatter};
use crate::view::{RasterView, ViewDesc, ViewHistory, ViewHistoryId, view_constants};
use crate::world::{WorldPos, needs_rebase, origin_for};
use crate::{
    BlobId, BlobTarget, EditDesc, EditId, EditRef, Error, ExposureState, ProducerBatch, Result,
    SceneRoot, TextureStreamGpu,
};

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
    /// Page and node page addresses of every resident mesh.
    pub mesh_pages: u32,
    /// Clusters one view can draw.
    pub visible_clusters: u32,
    /// Items of up to 32 clusters in a scene's raster candidate cut, and in the work one
    /// view's first culling pass leaves to its second.
    pub cluster_work: u32,
    /// Hierarchy nodes one traversal pass of a scene visits, at most.
    pub traversal_nodes: u32,
    /// Clusters of the ray-tracing cuts a scene's traversal gathers in a frame, and CLAS
    /// that frame's cluster BLAS take; copies past it trace a cached cut.
    pub rt_cut_clusters: u32,
    /// Cluster BLAS a scene builds in a frame for the cuts of copies near the focus; copies
    /// past it trace a cached cut.
    pub rt_cut_blas: u32,
    /// Entries in the resident cluster table: every cluster of every resident mesh.
    pub resident_clusters: u32,
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
    /// Bytes of resident mesh pages.
    pub mesh_page_pool: u64,
    /// Bytes of pages a DAG mesh keeps resident from mount; the pages of a larger one past
    /// its pinned coarsest pages stream on the traversal's requests.
    pub mesh_resident_max: u64,
    /// Bytes of the address range streamed pages' CLAS take, bound in pages as they need.
    pub clas_pool: u64,
    /// Bytes of resident hierarchy pages.
    pub hierarchy_page_pool: u64,
    /// Bytes of resident textures: whole textures, streamed textures' mip tails and tiles.
    pub texture_pool: u64,
    /// Bytes of resident instance pages.
    pub instance_page_pool: u64,
    /// Bytes of requested blobs loading or landed and not released.
    pub user_blob_pool: u64,
    /// Feedback cells of streamed textures: one per mip 0 tile of each layer.
    pub texture_feedback_cells: u32,
    /// Texture feedback records a frame keeps; taps past it report nothing that frame.
    pub texture_feedback_records: u32,
    /// Streamed units the loading side reads at once, at most.
    pub io_requests: u32,
    /// Bytes of streamed units the loading side reads at once, at most.
    pub io_bytes: u64,
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
            visible_clusters: 1 << 20,
            cluster_work: 1 << 18,
            traversal_nodes: 1 << 18,
            rt_cut_clusters: 1 << 18,
            rt_cut_blas: 1024,
            resident_clusters: 1 << 21,
            edit_triangles: 1 << 18,
            stream_clas_triangles: 1 << 18,
            micromap_pages: 64,
            ptlas_rebuilt: 250_000,
            mesh_page_pool: 3 << 30,
            mesh_resident_max: 1 << 20,
            clas_pool: 1 << 30,
            hierarchy_page_pool: 256 << 20,
            texture_pool: 4 << 30,
            instance_page_pool: 256 << 20,
            user_blob_pool: 256 << 20,
            texture_feedback_cells: 1 << 20,
            texture_feedback_records: 1 << 14,
            io_requests: 64,
            io_bytes: 64 << 20,
            stream_block: 16 << 20,
        }
    }
}

/// Farther models a representation chain holds: the length of `ModelGpu::chain`.
const MAX_CHAIN: usize = 3;

static NO_KERNELS: KernelSet = KernelSet {
    modules: &[],
    surfaces: 0,
};

#[derive(Clone, Copy, Debug)]
pub struct RendererDesc {
    pub budgets: Budgets,
    /// Frames the CPU records ahead of the GPU.
    pub frames_in_flight: u32,
    /// Consumer-linked kernels.
    pub kernels: &'static KernelSet,
    /// Which entry of the path tracing kernels views run.
    pub path_entry: PathEntry,
    /// World cells are 2^cell_log2 m on a side.
    pub cell_log2: u32,
    /// Archive reads are positioned reads on workers even where the system's completion
    /// ring (IoRing, io_uring) is available.
    pub positioned_reads: bool,
    /// Raster LOD: the largest cluster error drawn, in render pixels at the scene focus.
    pub raster_error_pixels: f32,
    /// Ray-tracing LOD: the largest cluster error traced, in render pixels at the scene focus.
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
            kernels: &NO_KERNELS,
            path_entry: PathEntry::Fastest,
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
    rt_merge_pixels: f32,
    software_raster_pixels: f32,
    alias_view_targets: bool,
    raster: RasterPipelines,
    cuts: CutPipelines,
    materials: MaterialTable,
    /// Emission by `ModelId`.
    models: Vec<Option<ModelEmission>>,
    /// Sections and animated parts by `ModelId`.
    model_cpu: Vec<ModelCpu>,
    /// Subdivision cages by `ModelId`.
    cages: Vec<Option<ModelCage>>,
    /// The farther models of every model with a representation chain.
    chains: HashMap<ModelId, Vec<ModelId>>,
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
        let accel = Arc::new(AccelShared {
            kernels: AccelKernels::new(device, desc.kernels)?,
            limits: AccelLimits {
                clas_triangles: b.stream_clas_triangles,
                micromaps: b.micromap_pages,
                mesh_resident_max: b.mesh_resident_max,
            },
            cluster_ids: Mutex::new(RangeAllocator::new(b.resident_clusters)),
            cluster_capacity: b.resident_clusters,
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
            &accel,
            materials.streams.gpu(),
            desc.frames_in_flight,
        )?;
        let sampler = device.sampler(&SamplerDesc {
            max_anisotropy: 16.0,
            ..SamplerDesc::default()
        })?;
        let loading = Arc::new(Loading::new(desc.positioned_reads));
        let thread = {
            let (l, d) = (Arc::clone(&loading), device.clone());
            std::thread::Builder::new()
                .name("orr loading".into())
                .spawn(move || loading::thread(l, d))?
        };
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
        let global = vec![
            Record::fill(0, streams, 2 * device.image_slots(), 0),
            mounts.streaming.init(),
        ];
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
                rt_merge_pixels: desc.rt_merge_pixels,
                software_raster_pixels: desc.software_raster_pixels,
                alias_view_targets: desc.alias_view_targets,
                raster,
                cuts,
                materials,
                models: Vec::new(),
                model_cpu: Vec::new(),
                cages: Vec::new(),
                chains: HashMap::new(),
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
    /// every frame submitted so far has completed.
    pub fn unmount(&mut self, m: MountId) {
        let frame = self.core.next_frame - 1;
        let core = &mut self.core;
        for page in core.mounts.pages_of(m) {
            core.land_page(page.0, 0);
        }
        let gone = core.mounts.models_of(m);
        core.mounts
            .unmount(m, frame, &mut core.models, &mut core.cages);
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

    /// Ends an edit. Its copies must be removed first; its memory is freed once every frame
    /// submitted so far has completed.
    pub fn destroy_edit(&mut self, id: EditId) {
        let frame = self.core.next_frame - 1;
        let core = &mut self.core;
        core.mounts
            .destroy_edit(id, frame, &mut core.models, &mut core.cages);
    }

    /// Gives every copy of `model` up to 3 farther models, each with the distance from the
    /// focus in m at which copies switch to it, ascending; empty: the model at every
    /// distance. Copies take their step from the next frame on, and a copy that changes step
    /// cross-fades between the two over a few frames. Steps past the model itself draw and
    /// trace rigid, with their own materials, every section shown and no pose. A chain ends
    /// when a pack holding one of its models is unmounted.
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
        if chain.is_empty() {
            core.chains.remove(&model);
        } else {
            core.chains
                .insert(model, chain.iter().map(|&(m, _)| m).collect());
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

    /// Wants the records of instance page `id` resident from the next frame on. Requests last
    /// one frame.
    pub fn prefetch_page(&mut self, id: PageId, cell_distance: u32) {
        self.core.mounts.request_page(id.0, cell_distance);
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
        self.core.mounts.pages.get(id.0 as usize)?.as_ref()
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

    pub fn cell_log2(&self) -> u32 {
        self.core.cell_log2
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

    /// What copies of `model` add to the emissive list from now on; copies added before
    /// keep their entries. Mounting sets it for models of packs.
    #[cfg(feature = "testing")]
    pub(crate) fn set_model_emission(&mut self, model: ModelId, emission: Option<ModelEmission>) {
        let i = model.0 as usize;
        if self.core.models.len() <= i {
            self.core.models.resize(i + 1, None);
        }
        self.core.models[i] = emission;
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
        let history = ViewHistory::new(exposure, arena);
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

    /// Step 1: waits until frame N - frames_in_flight has completed, sleeps on the low
    /// latency handle when one is given, recycles the upload window and the slots freed
    /// before it, swaps in hot-reloaded pipelines, and opens frame N.
    pub fn begin_frame(&mut self, desc: &FrameDesc<'_>) -> Result<Frame<'_>> {
        let started = Instant::now();
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
            render.wait(last, Duration::from_secs(10))?;
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
            let mut feedback = Vec::new();
            for id in &f.updated {
                let Ok(scene) = core.scene(*id) else { continue };
                let cpu = scene.cpu.lock().unwrap();
                if let Some(r) = &cpu.producer_readback {
                    let (producers, tessellation) = r.figures(slot);
                    f.stats.producers.add(&producers);
                    f.stats.tessellation.add(&tessellation);
                }
                if let Some(cuts) = &cpu.cuts {
                    cuts.figures(slot, &mut f.stats.cuts);
                    feedback.extend(cuts.feedback(slot));
                }
                if let Some(skin) = &cpu.skin_frame {
                    f.stats.skin.add(&skin.figures(slot));
                }
            }
            core.mounts.page_feedback(&feedback, frame);
            core.mounts.streaming.texture_feedback(slot, f.id);
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
            let sets = &mut cpu.page_sets;
            cpu.pages_removed.retain(|(frame, p)| {
                let done = *frame <= completed;
                if done {
                    sets.free(p.sets.0, p.sets.1);
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
        core.mounts.retire(completed, &self.device, &core.materials);
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
            emission: &mut core.models,
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
            core.mounts.land_edit(batch, &mut apply);
        }
        loading_points.extend(core.mounts.streaming.step(
            frame,
            settled,
            &self.loading,
            &mut apply,
        )?);
        core.mounts.begin_edits(core.budgets.edit_triangles);
        let events = core.mounts.streaming.take_page_events();
        for (page, records) in events {
            core.land_page(page, records);
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

    /// Wants the records of every placed page for this step, nearest a focus first, as far as
    /// the instance page budget holds them; the others are left to leave after their hold.
    fn request_pages(&mut self) {
        let mut wanted: FxHashMap<u32, f64> = FxHashMap::default();
        let edge = f64::from(1u32 << self.cell_log2);
        for scene in self.scenes.iter().flatten() {
            let cpu = scene.cpu.lock().unwrap();
            for p in &cpu.pages {
                let Some((center, radius)) = self.mounts.page_sphere(p.page.0, self.cell_log2)
                else {
                    continue;
                };
                let distance = cpu
                    .focus
                    .map_or(0.0, |f| (f.distance(center) - radius).max(0.0));
                let e = wanted.entry(p.page.0).or_insert(f64::INFINITY);
                *e = e.min(distance);
            }
        }
        let mut pages: Vec<(u32, f64)> = wanted.into_iter().collect();
        pages.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let budget = self.mounts.streaming.instance_page_budget();
        let mut bytes = 0;
        for (page, distance) in pages {
            bytes += self.mounts.page_bytes(page);
            if bytes > budget {
                break;
            }
            self.mounts.request_page(page, (distance / edge) as u32);
        }
    }

    /// Points the page groups of `page` in every scene at its records, or hides their copies.
    fn land_page(&mut self, page: u32, records: u64) {
        if let Some(Some(p)) = self.mounts.pages.get_mut(page as usize) {
            // SAFETY: the records stay resident until the event that writes this null.
            p.records = unsafe { GpuPtr::from_address(GpuAddress(records)) };
        }
        for scene in self.scenes.iter().flatten() {
            let mut cpu = scene.cpu.lock().unwrap();
            for g in cpu.groups.iter_mut().flatten() {
                if let Some(q) = &mut g.page
                    && q.page.0 == page
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
            id,
            scene,
            device: self.device,
            window: &self.core.windows[self.window],
            seq: &self.core.seq,
            frame: self.id,
            cell_log2: self.core.cell_log2,
            models: &self.core.models,
            model_cpu: &self.core.model_cpu,
            cages: &self.core.cages,
            patterns: &self.core.patterns,
            mounts: &self.core.mounts,
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
    /// to `compute` with the frame's command list.
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
        for &id in &self.listed {
            let scene = core.scene(id)?;
            let cpu = &mut *scene.cpu.lock().unwrap();
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
                let sizes = CutSizes {
                    slots: scene.desc.instance_capacity,
                    queue: core.budgets.traversal_nodes,
                    candidates: core.budgets.cluster_work,
                    rt_clusters: core.budgets.rt_cut_clusters,
                    rt_blas: core.budgets.rt_cut_blas,
                    models: core.budgets.models,
                };
                cpu.cuts = Some(Cuts::new(device, sizes, core.fif)?);
            }
            let cut = cpu.cuts.as_ref().expect("made above").gpu();
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
                    GroupWriter::Page => {}
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
                    GroupWriter::Gpu => {}
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
                environment: cpu.environment,
                medium: cpu.medium,
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
                _pad: 0,
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
        for &id in &self.listed {
            let scene = core.scene(id)?;
            let root = scene.tables.root.gpu();
            let mut cpu = scene.cpu.lock().unwrap();
            let cpu = &mut *cpu;
            for p in cpu.pages.iter_mut().filter(|p| !p.expanded) {
                let records = cpu.groups[p.group.0 as usize]
                    .as_ref()
                    .and_then(|g| g.page.as_ref())
                    .map_or(0, |g| g.records);
                if records == 0 {
                    continue;
                }
                p.expanded = true;
                expansions.push(instance_page::PageExpandRoot {
                    scene: root,
                    // SAFETY: the records stay resident through this frame: their eviction is
                    // written at step 1 before the event that hides them.
                    records: unsafe { GpuPtr::from_address(GpuAddress(records)) },
                    prototypes: p.prototypes.gpu().cast(),
                    spans: p.spans,
                    first_slot: p.range.first,
                    count: p.range.count,
                    prototype_count: p.prototype_count,
                    first_entry: p.first_entry,
                    entry_stride: p.entry_stride,
                    _pad0: 0,
                    _pad1: 0,
                });
            }
        }
        if !expansions.is_empty() {
            list.begin_pass("instance pages", false);
            for r in &expansions {
                let ptr = window.put_root(device, r)?;
                let p = self.pipelines.get(own.expand_page);
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
                core.raster_error_pixels,
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
            let inputs = CutInputs {
                scene: root,
                focus,
                raster_pixels: core.raster_error_pixels,
                rt_pixels: core.rt_error_pixels,
                culled_scale: core.culled_error_scale,
                merge_pixels: core.rt_merge_pixels,
                slots: cpu.ptlas.high_water,
                passes: hierarchy.depth,
                paged: hierarchy.paged,
                max_clas: hierarchy.max_clusters,
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
            list.begin_pass("emissive", false);
            for &(_, scene, _, _, desc) in &roots {
                let r = emissive::Root {
                    scene,
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
        let (slots, smoke) = {
            let cpu = scene.cpu.lock().unwrap();
            let smoke = cpu.producers.iter().flatten().any(|p| p.smoke);
            (cpu.ptlas.high_water, smoke)
        };
        let producer_capacity = scene.desc.producer_capacity;
        let sizes = TargetSizes {
            extent: view.render_extent.to_array(),
            extended: core.materials.any_extended(),
            visible_clusters: core.budgets.visible_clusters,
            work: core.budgets.cluster_work,
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
        history.arena.enter(ViewStep::Raster, list);
        match &mut history.targets {
            Some(t) if t.sizes == sizes => t.bind(&history.arena),
            t => *t = Some(ViewTargets::new(device, sizes, &history.arena, list)?),
        }
        let targets = history.targets.as_mut().expect("view targets");
        let occlusion =
            targets.hzb_scene.replace(view.scene) == Some(view.scene) && !view.reset_history;
        let targets = &*targets;
        let seen = constants;
        let constants = window.put(device, &constants)?;
        let raster = raster::record(
            device,
            &core.raster,
            self.pipelines,
            window,
            list,
            targets,
            root,
            slots,
            producer_capacity,
            constants,
            core.software_raster_pixels,
            occlusion,
            preprocess,
        )?;
        let (gbuf_a, gbuf_b, gbuf_x) = targets.gbuffer_textures();
        let out = RasterView {
            view: constants,
            scene: root,
            gbuffer: raster.cast(),
            targets: targets.gbuffer(),
            depth: Arc::clone(&targets.depth.texture),
            motion: Arc::clone(&targets.motion.texture),
            gbuf_a,
            gbuf_b,
            gbuf_x,
            hzb: Arc::clone(&targets.hzb.texture),
            counts: targets.counts(),
            raster,
            smoke_args: smoke.then(|| targets.smoke_args()),
            history: view.history,
            sky_stats,
            focus,
        };
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
    /// pre-exposed) in place.
    pub fn transparency(
        &mut self,
        view: &RasterView,
        radiance: StorageImageIndex,
        hook: &mut dyn FnMut(&mut TransparencyPass<'_, '_>),
    ) -> Result<()> {
        self.transparency_point(view, Some(radiance), hook)?;
        Ok(())
    }

    /// [`Self::transparency`] without the composite: returns the overlay, premultiplied
    /// RGBA16F at the render extent with coverage in alpha, for ray reconstruction to
    /// composite after denoising. It holds its contents through [`ViewStep::Rr`].
    pub fn transparency_overlay(
        &mut self,
        view: &RasterView,
        hook: &mut dyn FnMut(&mut TransparencyPass<'_, '_>),
    ) -> Result<Arc<orr_gpu::Texture>> {
        self.transparency_point(view, None, hook)
    }

    fn transparency_point(
        &mut self,
        view: &RasterView,
        radiance: Option<StorageImageIndex>,
        hook: &mut dyn FnMut(&mut TransparencyPass<'_, '_>),
    ) -> Result<Arc<orr_gpu::Texture>> {
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
        let (particle_args, particle_draws) = targets.particle_draws();
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
            particle_args,
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
        Ok(overlay_texture)
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
        Some(t) if t.hzb_scene == Some(id) && t.sizes.extent == seen.constants.extent => {
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
        if let Some(t) = self.loading_thread.take() {
            let _ = t.join();
        }
        self.core.mounts.clear(&self.device, &self.core.materials);
    }
}
