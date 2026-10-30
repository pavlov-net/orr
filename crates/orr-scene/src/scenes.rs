//! `Scenes`, the frame and its fixed order.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use glam::DVec3;
use orr_gpu::{
    CommandList, ComputePipeline, Device, GpuPtr, Hazards, KernelSet, LatencyMarker, LowLatency,
    MemoryKind, PassResult, QueueKind, ShaderStage, SpecData, Stages, TimelinePoint,
};

use crate::emissive::ModelEmission;
use crate::material::{MaterialTable, MaterialWriter};
use crate::memory::UploadWindow;
use crate::producer::ProducerUpdate;
use crate::scene::{GroupWriter, ModelId, Scene, SceneDesc, SceneId, SceneWriter};
use crate::shaders::{emissive, upload, world as world_kernel};
use crate::time::FrameTime;
use crate::upload::{Record, scatter};
use crate::view::{RasterView, ViewDesc, ViewHistory, ViewHistoryId, view_constants};
use crate::world::{needs_rebase, origin_for};
use crate::{Error, ExposureState, ProducerBatch, Result, SceneRoot};

/// Pool sizes and per-frame limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budgets {
    /// Bytes per upload block; a window that outgrows its blocks takes another.
    pub upload_block: u64,
    /// Entries in the global material table.
    pub materials: u32,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            upload_block: 16 << 20,
            materials: 65_536,
        }
    }
}

static NO_KERNELS: KernelSet = KernelSet { modules: &[] };

#[derive(Clone, Copy, Debug)]
pub struct RendererDesc {
    pub budgets: Budgets,
    /// Frames the CPU records ahead of the GPU.
    pub frames_in_flight: u32,
    /// Consumer-linked kernels.
    pub kernels: &'static KernelSet,
    /// World cells are 2^cell_log2 m on a side.
    pub cell_log2: u32,
}

impl Default for RendererDesc {
    fn default() -> Self {
        Self {
            budgets: Budgets::default(),
            frames_in_flight: 2,
            kernels: &NO_KERNELS,
            cell_log2: 8,
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
}

impl SceneCompute<'_, '_> {
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

struct Pipelines {
    scatter: ComputePipeline,
    rebase: ComputePipeline,
    emissive_instances: ComputePipeline,
    emissive_producers: ComputePipeline,
}

struct InFlight {
    id: u64,
    point: TimelinePoint,
    stats: FrameStats,
}

struct Core {
    fif: u32,
    cell_log2: u32,
    materials: MaterialTable,
    /// Emission by `ModelId`.
    models: Vec<Option<ModelEmission>>,
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
}

/// Global scene state and every resident scene. Render thread only.
pub struct Scenes {
    device: Device,
    pipelines: Pipelines,
    core: Core,
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

fn pipeline(
    device: &Device,
    kernels: &KernelSet,
    own: &'static orr_gpu::ShaderModule,
    entry: &str,
) -> Result<ComputePipeline> {
    let module = kernels.module(own)?;
    let stage = ShaderStage::new(module, entry, SpecData::default())?;
    Ok(device.create_compute_pipeline(&stage, module.name)?)
}

impl Scenes {
    /// Compiles orr-scene's pipelines on the calling thread, from `desc.kernels`' module
    /// wherever it holds one for an orr-scene kernel.
    pub fn new(device: &Device, desc: &RendererDesc) -> Result<Self> {
        assert!(desc.frames_in_flight >= 1, "frames_in_flight is at least 1");
        assert!(desc.cell_log2 <= 16, "cells of 2^{} m", desc.cell_log2);
        Ok(Self {
            device: device.clone(),
            pipelines: Pipelines {
                scatter: pipeline(device, desc.kernels, &upload::MODULE, "scatter")?,
                rebase: pipeline(device, desc.kernels, &world_kernel::MODULE, "rebase_lights")?,
                emissive_instances: pipeline(device, desc.kernels, &emissive::MODULE, "instances")?,
                emissive_producers: pipeline(device, desc.kernels, &emissive::MODULE, "producers")?,
            },
            core: Core {
                fif: desc.frames_in_flight,
                cell_log2: desc.cell_log2,
                materials: MaterialTable::new(device, desc.budgets.materials)?,
                models: Vec::new(),
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
            },
        })
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
    /// keep their entries. Model loading sets it when a model becomes resident.
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
        let history = ViewHistory::new(exposure);
        Ok(ViewHistoryId(insert(&mut self.core.histories, history)))
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
    /// before it, and opens frame N.
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
        }
        self.device.collect_garbage();
        let slot = (frame % u64::from(core.fif)) as usize;
        core.windows[slot].reset();
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
}

impl<'a> Frame<'a> {
    /// orr's frame id: strictly increasing from 1.
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn time(&self) -> FrameTime {
        self.time
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
        let core = &*self.core;
        let window = &core.windows[self.window];
        let mut list = self.list.take().expect("frame list");
        let mut records = std::mem::take(&mut *core.global.lock().unwrap());
        let mut copies = Vec::new();
        let mut rebased = Vec::new();
        let mut roots = Vec::new();
        for &id in &self.listed {
            let scene = core.scene(id)?;
            let cpu = &mut *scene.cpu.lock().unwrap();
            records.append(&mut cpu.records);
            let counts = scene.tables.producer_counts.gpu;
            for (i, p) in cpu.producers.iter().enumerate() {
                if p.as_ref()
                    .is_some_and(|p| p.update == ProducerUpdate::EveryFrame)
                {
                    let at = counts.offset(i as u64).address().0;
                    records.push(Record::fill(core.next_seq(), at, 2, 0));
                }
            }
            if let Some((focus, _)) = cpu.focus
                && needs_rebase(cpu.origin_cell, focus, core.cell_log2)
            {
                cpu.origin_cell = origin_for(focus, core.cell_log2);
                rebased.push((scene.tables.root.gpu, scene.desc.light_capacity));
            }
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
                materials: core.materials.table.gpu,
                material_sets: t.material_sets.gpu.cast(),
                lights: t.lights.gpu,
                emissive: t.emissive.gpu,
                producers: t.producers.gpu,
                batches: t.batches.gpu,
                sky_stats: t.sky_stats.gpu,
                environment: cpu.environment,
                medium: cpu.medium,
                tlas: device.empty_tlas(),
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
                _pad: 0,
            };
            let src = window.put(device, &root)?;
            records.push(Record::values(core.next_seq(), t.root.gpu, src, 1));
            roots.push((id, t.root.gpu, t.batches.gpu, cpu.focus, scene.desc));
        }

        list.begin_pass("upload", true);
        for &(dst, src, bytes) in &copies {
            list.copy(dst, src, bytes);
        }
        if !copies.is_empty() {
            list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        }
        self.scatter = scatter(device, window, &mut list, &self.pipelines.scatter, records)?;
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
                list.dispatch(&self.pipelines.rebase, ptr, [lights.div_ceil(64), 1, 1]);
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
            };
            compute(&mut sc);
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        }
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
                    let p = &self.pipelines.emissive_producers;
                    list.dispatch(p, ptr, [desc.producer_capacity, 1, 1]);
                }
                if desc.emissive_capacity > 0 {
                    let groups = desc.emissive_capacity.div_ceil(64);
                    list.dispatch(&self.pipelines.emissive_instances, ptr, [groups, 1, 1]);
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

    /// Step 4, the raster half of a view: writes its `ViewConstants` and moves its history
    /// on.
    pub fn raster_view(&mut self, view: &ViewDesc<'_>) -> Result<RasterView> {
        self.ensure_updated()?;
        if !self.listed.contains(&view.scene) {
            return Err(Error::NotListed(view.scene));
        }
        let core = &mut *self.core;
        let scene = core.scene(view.scene)?;
        let origin = scene.cpu.lock().unwrap().origin_cell;
        let root = scene.tables.root.gpu;
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
        let view = core.windows[self.window].put(self.device, &constants)?;
        self.step("raster_view");
        Ok(RasterView { view, scene: root })
    }

    /// Step 6: submits the frame on the render queue after `wait`, and returns the point
    /// its work signals.
    pub fn end(mut self, wait: &[TimelinePoint]) -> Result<TimelinePoint> {
        self.ensure_updated()?;
        let render = self.device.queue(QueueKind::Render).expect("render queue");
        let list = self.list.take().expect("frame list");
        let point = render.submit(vec![list], wait)?;
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
        };
        self.core.in_flight.push_back(InFlight {
            id: self.id,
            point,
            stats,
        });
        Ok(point)
    }
}
