//! `Scenes`, the frame and its fixed order.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use glam::{DVec3, Vec3};
use orr_gpu::{
    Allocation, ClearColor, ColorAttachment, ColorLoad, CommandList, ComputePipeline, Device,
    GpuPtr, Hazards, KernelSet, LatencyMarker, LowLatency, MemoryKind, PassResult, QueueKind,
    Rendering, SamplerDesc, SamplerIndex, ShaderModule, SpecData, Stages, StorageImageIndex,
    TextureIndex, TimelinePoint,
};
use orr_pack::AssetId;

use crate::accel::{AccelKernels, AccelLimits, AccelShared};
use crate::arena::{ArenaReport, ArenaRequest, ViewArena, ViewStep};
use crate::emissive::ModelEmission;
use crate::loading::{self, Loading, Msg};
use crate::material::{MaterialId, MaterialTable, MaterialWriter};
use crate::memory::RangeAllocator;
use crate::memory::UploadWindow;
use crate::mount::{
    Apply, ClipRef, InstancePage, LoadState, MountId, MountTask, Mounts, PackSource, PageId,
    SkeletonRef, TableSizes,
};
use crate::pipelines::{PipelineId, Pipelines, ReloadReport};
use crate::producer::{self, Job, ProducerKernels, ProducerStats, ProducerUpdate, Readback};
use crate::ptlas::{FrameInputs, PtlasPipelines, RtStats};
use crate::raster::{self, RasterPipelines, TargetSizes, ViewTargets};
#[cfg(feature = "hot-reload")]
use crate::reload::Reloader;
use crate::residency::ResidencyStats;
use crate::scene::{GroupWriter, ModelCpu, ModelId, Scene, SceneDesc, SceneId, SceneWriter};
use crate::shaders::transparency::CompositeRoot;
use crate::shaders::{animation, emissive, world as world_kernel};
use crate::skin::{SkinFrame, SkinKernels, SkinStats};
use crate::time::FrameTime;
use crate::transparency::TransparencyPass;
use crate::traverse::{CutInputs, CutPipelines, CutSizes, CutStats, Cuts};
use crate::upload::{Record, scatter};
use crate::view::{RasterView, ViewDesc, ViewHistory, ViewHistoryId, view_constants};
use crate::world::{WorldPos, needs_rebase, origin_for};
use crate::{Error, ExposureState, ProducerBatch, Result, SceneRoot};

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
    /// Bytes of resident hierarchy pages.
    pub hierarchy_page_pool: u64,
    /// Bytes of resident textures.
    pub texture_pool: u64,
    /// Bytes of resident instance pages.
    pub instance_page_pool: u64,
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
            stream_clas_triangles: 1 << 18,
            micromap_pages: 64,
            ptlas_rebuilt: 250_000,
            mesh_page_pool: 3 << 30,
            hierarchy_page_pool: 256 << 20,
            texture_pool: 4 << 30,
            instance_page_pool: 256 << 20,
            io_requests: 64,
            io_bytes: 64 << 20,
            stream_block: 16 << 20,
        }
    }
}

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
    /// World cells are 2^cell_log2 m on a side.
    pub cell_log2: u32,
    /// Archive reads are positioned reads on workers even where the system's completion
    /// ring (IoRing, io_uring) is available.
    pub positioned_reads: bool,
    /// Raster LOD: the largest cluster error drawn, in pixels at the scene focus.
    pub raster_error_pixels: f32,
    /// Ray-tracing LOD: the largest cluster error traced, in pixels at the scene focus.
    pub rt_error_pixels: f32,
    /// Copies no view drew last frame trace at this many times `rt_error_pixels`.
    pub culled_error_scale: f32,
    /// Copies whose bounding sphere projects to a radius under this many pixels at the focus
    /// trace their model's merged cut: the finest cached cut any of them needs. 0 turns
    /// merging off.
    pub rt_merge_pixels: f32,
    /// Clusters whose screen rect's longer side is at most this many render pixels raster
    /// in compute, unless they cross the near plane; others take the mesh-shader path. 0
    /// turns the software raster off; values above 128 act as 128.
    pub software_raster_pixels: f32,
    /// A view's transient targets whose spans in the frame order do not overlap share
    /// memory ([`ViewArena`]).
    pub alias_view_targets: bool,
}

impl Default for RendererDesc {
    fn default() -> Self {
        Self {
            budgets: Budgets::default(),
            frames_in_flight: 2,
            kernels: &NO_KERNELS,
            cell_log2: 8,
            positioned_reads: false,
            raster_error_pixels: 1.0,
            rt_error_pixels: 1.0,
            culled_error_scale: 2.0,
            rt_merge_pixels: 32.0,
            software_raster_pixels: 32.0,
            alias_view_targets: true,
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
}

/// Scene compute: a listed scene's turn in `update_scenes`, after its uploads have landed.
pub struct SceneCompute<'a, 'q> {
    pub cmd: &'a mut CommandList<'q>,
    pub scene: SceneId,
    pub root: GpuPtr<SceneRoot>,
    /// The scene's producer batches, indexed by `ProducerId`.
    pub batches: GpuPtr<ProducerBatch>,
    pub time: FrameTime,
    /// The scene focus and its pixels per radian, as last set.
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
    point: TimelinePoint,
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
    /// Records outside any scene: materials, view-history initialisation.
    global: Mutex<Vec<Record>>,
    scenes: Vec<Option<Scene>>,
    histories: Vec<Option<ViewHistory>>,
    windows: Vec<UploadWindow>,
    seq: AtomicU64,
    next_frame: u64,
    completed_frame: u64,
    in_flight: VecDeque<InFlight>,
    stats: FrameStats,
    high_water: u64,
    mounts: Mounts,
    accel: Arc<AccelShared>,
    ptlas: PtlasPipelines,
    /// Scratch of every scene's PTLAS build; the builds of a frame run one after another.
    ptlas_scratch: Mutex<Option<Allocation<u8>>>,
    producers: ProducerKernels,
    skin: SkinKernels,
    /// Scratch of producer builds, grown to the largest frame's.
    producer_scratch: Mutex<Option<Allocation<u8>>>,
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
    /// Compiles orr-scene's pipelines on the calling thread, from `desc.kernels`' module
    /// wherever it holds one for an orr-scene kernel, and starts the loading thread. With
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
            },
            cluster_ids: Mutex::new(RangeAllocator::new(b.resident_clusters)),
            cluster_capacity: b.resident_clusters,
        });
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
        let raster = RasterPipelines::new(device, desc.kernels, &mut pipelines)?;
        let ptlas = PtlasPipelines::new(device, desc.kernels, &mut pipelines)?;
        let producers = ProducerKernels::new(device, desc.kernels, &mut pipelines)?;
        let skin = SkinKernels::new(device, desc.kernels, &mut pipelines)?;
        let cuts = CutPipelines::new(device, desc.kernels, &mut pipelines)?;
        #[cfg(feature = "hot-reload")]
        let mut reloader = Reloader::new();
        #[cfg(feature = "hot-reload")]
        for r in &pipelines.recipes {
            reloader.watch(r.module);
        }
        Ok(Self {
            loading,
            loading_thread: Some(thread),
            sampler,
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
                materials: MaterialTable::new(device, desc.budgets.materials)?,
                models: Vec::new(),
                model_cpu: Vec::new(),
                global: Mutex::new(Vec::new()),
                scenes: Vec::new(),
                histories: Vec::new(),
                windows: (0..desc.frames_in_flight)
                    .map(|_| UploadWindow::new(desc.budgets.upload_block))
                    .collect(),
                seq: AtomicU64::new(1),
                next_frame: 1,
                completed_frame: 0,
                in_flight: VecDeque::new(),
                stats: FrameStats::default(),
                high_water: 0,
                mounts,
                accel,
                ptlas,
                ptlas_scratch: Mutex::new(None),
                producers,
                skin,
                producer_scratch: Mutex::new(None),
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
        core.mounts.unmount(m, frame, &mut core.models);
    }

    /// The model `id` of mount `m`.
    pub fn model(&self, m: MountId, id: AssetId) -> Result<ModelId> {
        self.core.mounts.lookup(m, id, |r| &r.models)
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

    /// The cooked texture `id` of mount `m`, for consumer data that samples it.
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
        (m.models.gpu, m.meshes.gpu, self.core.materials.table.gpu)
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
        let base = exposure.gpu.address().0;
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
        // Its slots' initialisation may not have been applied yet.
        let base = h.exposure.gpu.address().0;
        let end = base + h.exposure.size;
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
            if !due && render.completed() < f.point.value() {
                break;
            }
            render.wait(f.point, Duration::from_secs(10))?;
            let mut f = core.in_flight.pop_front().unwrap();
            f.stats.gpu = self.device.pass_results(f.point).unwrap_or_default();
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
                    f.stats.producers.add(&r.figures(slot));
                }
                if let Some(cuts) = &cpu.cuts {
                    cuts.figures(slot, &mut f.stats.cuts);
                    feedback.extend(cuts.feedback(slot));
                }
                if let Some(skin) = &cpu.skin_frame {
                    f.stats.skin.add(&skin.figures(slot));
                }
            }
            core.mounts.node_feedback(&feedback, frame);
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
        let mut apply = Apply {
            device: &self.device,
            window: &core.windows[slot],
            seq: &core.seq,
            records: core.global.get_mut().unwrap(),
            materials: &core.materials,
            emission: &mut core.models,
            model_cpu: &mut core.model_cpu,
        };
        let mut loading_points = Vec::new();
        for batch in self.loading.take_batches(&self.device) {
            loading_points.extend(batch.point.map(|(_, p)| p));
            core.mounts.apply(batch, &mut apply);
        }
        loading_points.extend(
            core.mounts
                .streaming
                .step(frame, &self.loading, &mut apply)?,
        );
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

    /// What hot reload applied at this frame's step 1; always empty without the
    /// `hot-reload` feature.
    pub fn reloads(&self) -> &[ReloadReport] {
        &self.reloads
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
            records.append(&mut cpu.records);
            masks.extend(std::mem::take(&mut cpu.masks));
            let counts = scene.tables.producer_counts.gpu;
            for (i, p) in cpu.producers.iter().enumerate() {
                if p.as_ref()
                    .is_some_and(|p| p.update == ProducerUpdate::EveryFrame)
                {
                    let at = counts.offset(i as u64).address().0;
                    records.push(Record::fill(core.next_seq(), at, 2, 0));
                }
            }
            let mut rebase = false;
            if let Some((focus, _)) = cpu.focus
                && needs_rebase(cpu.origin_cell, focus, core.cell_log2)
            {
                cpu.origin_cell = origin_for(focus, core.cell_log2);
                rebased.push((scene.tables.root.gpu, scene.desc.light_capacity));
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
                    GroupWriter::Cpu => {
                        if g.written && !fresh {
                            copies.push((g.buffers[1].gpu, g.buffers[0].gpu, g.buffers[0].size));
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
                if bytemuck::bytes_of(&before) != bytemuck::bytes_of(&after) {
                    let src = window.put(device, &after)?;
                    let dst = scene.tables.groups.gpu.offset(index as u64);
                    records.push(Record::values(core.next_seq(), dst, src, 1));
                }
            }
            let t = &scene.tables;
            let root = SceneRoot {
                instances: t.instances.gpu,
                slot_state: t.slot_state.gpu,
                groups: t.groups.gpu,
                models: core.mounts.models.gpu,
                meshes: core.mounts.meshes.gpu,
                materials: core.materials.table.gpu,
                clusters: core.mounts.clusters.gpu,
                material_sets: t.material_sets.gpu.cast(),
                lights: t.lights.gpu,
                emissive: t.emissive.gpu,
                producers: t.producers.gpu,
                batches: t.batches.gpu,
                producer_state: t.producer_state.gpu,
                sky_stats: t.sky_stats.gpu,
                environment: cpu.environment,
                medium: cpu.medium,
                surface_data: cpu.surface_data,
                tlas: tlas.unwrap_or_else(|| device.empty_tlas()),
                part_slots: cpu.ptlas.part_slots.gpu,
                origin_cell: cpu.origin_cell.extend(core.cell_log2 as i32).to_array(),
                delta_time: self.time.delta,
                frame_index: self.id as u32,
                instance_capacity: scene.desc.instance_capacity,
                group_capacity: scene.desc.group_capacity,
                light_capacity: scene.desc.light_capacity,
                material_set_capacity: scene.desc.material_set_capacity,
                producer_capacity: scene.desc.producer_capacity,
                emissive_capacity: scene.desc.emissive_capacity,
                near_range: cpu.near_range,
                part_base: scene.desc.instance_capacity,
                part_capacity: scene.desc.part_capacity,
                producer_base: scene.desc.instance_capacity + scene.desc.part_capacity,
                cut,
                skinned: t.skinned.gpu,
                skin_state: t.skin_state.gpu,
                skin_work: skin_work.unwrap_or(GpuPtr::NULL).cast(),
                skinned_capacity: scene.desc.skinned_capacity,
                _pad: 0,
            };
            let src = window.put(device, &root)?;
            records.push(Record::values(core.next_seq(), t.root.gpu, src, 1));
            if tlas.is_some() {
                traced.push((id, rebase));
            }
            roots.push((id, t.root.gpu, t.batches.gpu, cpu.focus, scene.desc));
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
            let mut scratch = core.ptlas_scratch.lock().unwrap();
            let need = traced
                .iter()
                .filter_map(|&(id, _)| core.scene(id).ok())
                .map(|s| s.cpu.lock().unwrap().ptlas.scratch_bytes(device))
                .max()
                .unwrap_or(0);
            if scratch.as_ref().is_none_or(|a| a.size < need) {
                *scratch = Some(device.gpu_malloc::<u8>(
                    MemoryKind::DeviceLocal,
                    need.max(256),
                    "PTLAS scratch",
                )?);
            }
            let scratch = scratch.as_ref().expect("made above").gpu.address();
            for &(id, rebased) in &traced {
                let scene = core.scene(id)?;
                let cpu = &mut *scene.cpu.lock().unwrap();
                let (focus_cell, focus_offset) = match cpu.focus {
                    Some((f, ppr)) => {
                        let w = WorldPos::new(f, core.cell_log2);
                        (w.cell.extend(1).to_array(), w.offset.extend(ppr).to_array())
                    }
                    None => ([0; 4], [0.0; 4]),
                };
                let inputs = FrameInputs {
                    scene: scene.tables.root.gpu,
                    focus_cell,
                    focus_offset,
                    origin_cell: cpu.origin_cell,
                    cell_log2: core.cell_log2,
                    rebased,
                    budget: core.budgets.ptlas_rebuilt,
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
        self.step("update_scenes");
        Ok(())
    }

    fn ensure_updated(&mut self) -> Result<()> {
        if !self.updated {
            self.update_scenes(&mut |_| {})?;
        }
        Ok(())
    }

    /// The frame's command list, for the steps after `update_scenes` (views, output, UI);
    /// runs `update_scenes` first when the consumer did not.
    pub fn list(&mut self) -> Result<&mut CommandList<'a>> {
        self.ensure_updated()?;
        Ok(self.list.as_mut().expect("frame list"))
    }

    /// Step 4, the raster half of a view: writes its `ViewConstants`, culls and rasters the
    /// scene into the visibility buffer, resolves the G-buffer, depth, motion and HZB, and
    /// moves its history on. Lays out the view's arena with the raster's transient targets
    /// alone and enters [`ViewStep::Raster`].
    pub fn raster_view(&mut self, view: &ViewDesc<'_>) -> Result<RasterView> {
        self.raster_view_with(view, &[])
    }

    /// [`Self::raster_view`], with `requests`, the transient targets of the passes that
    /// follow it, laid out in the view's arena beside the raster's.
    pub fn raster_view_with(
        &mut self,
        view: &ViewDesc<'_>,
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
            (cpu.origin_cell, cpu.focus.map_or(view.position, |f| f.0))
        };
        let root = scene.tables.root.gpu;
        let sky_stats = scene.tables.sky_stats.gpu;
        let focus = WorldPos::new(focus, core.cell_log2)
            .relative_to(
                WorldPos {
                    cell: origin,
                    offset: Vec3::ZERO,
                },
                core.cell_log2,
            )
            .to_array();
        let slots = scene.cpu.lock().unwrap().ptlas.high_water;
        let producer_capacity = scene.desc.producer_capacity;
        let sizes = TargetSizes {
            extent: view.render_extent.to_array(),
            extended: core.materials.any_extended(),
            visible_clusters: core.budgets.visible_clusters,
            work: core.budgets.cluster_work,
            bins: core.raster.bins,
            slots: scene.desc.instance_capacity,
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
            history: view.history,
            sky_stats,
            focus,
        };
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
        self.step("transparency");
        Ok(())
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

    /// Step 6: submits the frame on the render queue after `wait`, and returns the point
    /// its work signals.
    pub fn end(mut self, wait: &[TimelinePoint]) -> Result<TimelinePoint> {
        self.ensure_updated()?;
        let render = self.device.queue(QueueKind::Render).expect("render queue");
        let list = self.list.take().expect("frame list");
        let mut waits = wait.to_vec();
        waits.append(&mut self.loading);
        let point = render.submit(vec![list], &waits)?;
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
        };
        self.core.in_flight.push_back(InFlight {
            id: self.id,
            point,
            stats,
            traced: std::mem::take(&mut self.traced),
            updated: if self.updated {
                std::mem::take(&mut self.listed)
            } else {
                Vec::new()
            },
        });
        Ok(point)
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
    let jobs: Vec<Job<'_>> = scenes
        .iter()
        .flat_map(|(scene, cpu)| {
            cpu.producers
                .iter()
                .enumerate()
                .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)))
                .filter(|(_, p)| p.builds())
                .map(|(i, producer)| Job {
                    scene: scene.tables.root.gpu,
                    entry: i as u32,
                    producer,
                })
        })
        .collect();
    if !jobs.is_empty() {
        let need = producer::scratch_bytes(device, &jobs).max(256);
        let mut held = core.producer_scratch.lock().unwrap();
        if held.as_ref().is_none_or(|a| a.size < need) {
            *held =
                Some(device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, need, "producer scratch")?);
        }
        let scratch = held.as_ref().expect("made above").gpu.address();
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
                    .map(|p| (i as u32, p.max_clusters, p.max_vertices))
            })
            .collect();
        if cpu.producer_readback.is_none() {
            cpu.producer_readback = Some(Readback::new(device, capacity, core.fif)?);
        }
        let t = &scene.tables;
        cpu.producer_readback.as_mut().expect("made above").record(
            list,
            slot,
            t.producer_counts.gpu,
            t.producer_state.gpu,
            set,
        );
    }
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
    Ok(())
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
