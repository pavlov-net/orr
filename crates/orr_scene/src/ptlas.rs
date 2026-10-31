//! A scene's partitioned TLAS: instance index = slot, part instances from `part_base`,
//! producer entries from `producer_base`. Copies take cell partitions in copy order, per
//! transform-group anchor cell, at most `partition_instances` to a partition; movers sit in
//! the global partition until their transform has rested for 30 frames. The `ptlas` kernels
//! decide each frame's writes, null writes and BLAS updates on the GPU and spend the
//! frame's budget of rebuilt instances from the focus outward, mandatory work first.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use orr_gpu::{
    Allocation, ClusterVertexFormat, CommandList, Device, GpuAddress, GpuPtr, Hazards, MemoryKind,
    PtlasArgs, PtlasLimits, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp, RtasOpDesc, Stages,
    Tlas, partition_translation,
};
use orr_math::IVec3;

use crate::Result;
use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::scene::SceneDesc;
use crate::shaders::ptlas::{PartitionGpu, PendingGpu, PtlasGpu, PtlasRoot, TranslateJob};
use crate::upload::Record;

/// `ptlas.slang` constants.
const PARTITION_NONE: u32 = u32::MAX;
const DISTANCE_BUCKETS: u32 = 64;
const COUNTERS: usize = 14;
const COUNTERS_CLEARED: u32 = 13;
const PTLAS_REBASE: u32 = 1;

/// `PtlasGpu.counters` read back for [`RtStats`].
const COUNTER_WRITES: usize = 0;
const COUNTER_UPDATES: usize = 1;
const COUNTER_TRANSLATES: usize = 2;
const COUNTER_REBUILT: usize = 4;
const COUNTER_DEFERRED: usize = 5;
const COUNTER_MAX_AGE: usize = 6;
const COUNTER_DROPPED: usize = 7;
const COUNTER_GLOBAL: usize = 13;

/// Ray-tracing figures of a frame, summed over its scenes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtStats {
    /// PTLAS instance writes, null writes included.
    pub writes: u32,
    /// PTLAS BLAS updates.
    pub updates: u32,
    pub translations: u32,
    /// The clamp's unit: instances in partitions a write rebuilt, plus updates.
    pub rebuilt: u32,
    /// Operations left for a later frame.
    pub deferred: u32,
    /// Frames the oldest deferred operation has waited.
    pub max_age: u32,
    /// Records past their array's capacity.
    pub dropped: u32,
    /// Instances in global partitions.
    pub global: u32,
    /// Copies with no free partition, not traced.
    pub unplaced: u32,
}

impl RtStats {
    pub(crate) fn add(&mut self, c: &[u32; COUNTERS], unplaced: u32) {
        self.writes += c[COUNTER_WRITES];
        self.updates += c[COUNTER_UPDATES];
        self.translations += c[COUNTER_TRANSLATES];
        self.rebuilt += c[COUNTER_REBUILT];
        self.deferred += c[COUNTER_DEFERRED];
        self.max_age = self.max_age.max(c[COUNTER_MAX_AGE]);
        self.dropped += c[COUNTER_DROPPED];
        self.global += c[COUNTER_GLOBAL];
        self.unplaced += unplaced;
    }
}

/// The `ptlas` pipelines.
pub(crate) struct PtlasPipelines {
    clear: PipelineId,
    want: PipelineId,
    cutoff: PipelineId,
    admit: PipelineId,
    emit: PipelineId,
    translate: PipelineId,
    commands: PipelineId,
}

impl PtlasPipelines {
    pub(crate) fn new(
        device: &Device,
        kernels: &orr_gpu::KernelSet,
        p: &mut Pipelines,
    ) -> Result<Self> {
        let m = &crate::shaders::ptlas::MODULE;
        let mut c = |entry| p.create(device, kernels, m, entry, Default::default());
        Ok(Self {
            clear: c("clear")?,
            want: c("want")?,
            cutoff: c("cutoff")?,
            admit: c("admit")?,
            emit: c("emit")?,
            translate: c("translate")?,
            commands: c("commands")?,
        })
    }
}

struct PartitionCpu {
    cell: IVec3,
    count: u32,
}

/// The PTLAS and the arrays its builds read, made at the scene's first traced frame.
struct Storage {
    tlas: Allocation<u8>,
    record: Allocation<PtlasGpu>,
    _blocks: Vec<Allocation<u8>>,
    _pending: Allocation<PendingGpu>,
    commands: GpuPtr<u8>,
    command_count: GpuPtr<u32>,
    counters: GpuPtr<u32>,
    /// One counter block per frame in flight.
    readback: Allocation<[u32; COUNTERS]>,
    built: bool,
}

/// One scene's PTLAS state.
pub(crate) struct Ptlas {
    limits: PtlasLimits,
    max_per_partition: u32,
    partitions: Vec<PartitionCpu>,
    free: Vec<u32>,
    /// The partition of each cell that takes its next copies.
    open: HashMap<IVec3, u32>,
    slot_partition: Vec<u32>,
    /// Partitions whose record changed since the last frame.
    changed: BTreeSet<u32>,
    /// Partitions given a cell since the last frame: their translation is written.
    assigned: BTreeSet<u32>,
    /// Partitions emptied in a frame, free once it has completed.
    emptied: Vec<(u64, u32)>,
    unplaced: u32,
    /// Slots below it have been handed out at some time.
    pub high_water: u32,
    pub partition_table: Allocation<PartitionGpu>,
    pub slot_table: Allocation<u32>,
    pub part_slots: Allocation<u32>,
    storage: Option<Storage>,
}

impl Ptlas {
    pub(crate) fn new(device: &Device, desc: &SceneDesc) -> Result<Self> {
        let n = |c: u32| u64::from(c.max(1));
        let dl = MemoryKind::DeviceLocal;
        let partitions = desc.ptlas_partitions;
        Ok(Self {
            limits: PtlasLimits {
                instances: desc.instance_capacity + desc.part_capacity + desc.producer_capacity,
                partitions,
                max_per_partition: desc.partition_instances,
                global_instances: desc.global_instances,
                max_ops: 3,
                translation: true,
            },
            max_per_partition: desc.partition_instances,
            partitions: (0..partitions)
                .map(|_| PartitionCpu {
                    cell: IVec3::ZERO,
                    count: 0,
                })
                .collect(),
            free: (0..partitions).rev().collect(),
            open: HashMap::new(),
            slot_partition: vec![PARTITION_NONE; desc.instance_capacity as usize],
            changed: BTreeSet::new(),
            assigned: BTreeSet::new(),
            emptied: Vec::new(),
            unplaced: 0,
            high_water: 0,
            partition_table: device.gpu_malloc(dl, n(partitions), "partitions")?,
            slot_table: device.gpu_malloc(dl, n(desc.instance_capacity), "slot partitions")?,
            part_slots: device.gpu_malloc(dl, n(desc.part_capacity), "part slots")?,
            storage: None,
        })
    }

    /// Puts `count` copies from slot `first` into partitions of `cell`, in order; returns
    /// each copy's partition.
    pub(crate) fn place(&mut self, first: u32, count: u32, cell: IVec3) -> Vec<u32> {
        self.high_water = self.high_water.max(first + count);
        (first..first + count)
            .map(|slot| {
                let open = self
                    .open
                    .get(&cell)
                    .copied()
                    .filter(|&p| self.partitions[p as usize].count < self.max_per_partition);
                let Some(p) = open.or_else(|| self.free.pop()) else {
                    self.unplaced += 1;
                    self.slot_partition[slot as usize] = PARTITION_NONE;
                    return PARTITION_NONE;
                };
                let part = &mut self.partitions[p as usize];
                if part.count == 0 {
                    part.cell = cell;
                    self.assigned.insert(p);
                }
                part.count += 1;
                self.open.insert(cell, p);
                self.changed.insert(p);
                self.slot_partition[slot as usize] = p;
                p
            })
            .collect()
    }

    /// Takes slots `first..first + count` out of their partitions in `frame`.
    pub(crate) fn remove(&mut self, first: u32, count: u32, frame: u64) {
        for slot in first..first + count {
            let p = std::mem::replace(&mut self.slot_partition[slot as usize], PARTITION_NONE);
            if p == PARTITION_NONE {
                self.unplaced = self.unplaced.saturating_sub(1);
                continue;
            }
            let part = &mut self.partitions[p as usize];
            part.count -= 1;
            self.changed.insert(p);
            if part.count == 0 {
                if self.open.get(&part.cell) == Some(&p) {
                    self.open.remove(&part.cell);
                }
                self.emptied.push((frame, p));
            }
        }
    }

    /// Frees the partitions emptied in frames at or before `completed`.
    pub(crate) fn retire(&mut self, completed: u64) {
        let partitions = &self.partitions;
        let free = &mut self.free;
        self.emptied.retain(|&(frame, p)| {
            let done = frame <= completed;
            // A partition refilled before its frame completed stays in use.
            if done && partitions[p as usize].count == 0 {
                free.push(p);
            }
            !done
        });
    }

    /// The scene's TLAS for the frame about to record, after making the PTLAS storage the
    /// first time a slot has been handed out.
    pub(crate) fn tlas(&mut self, device: &Device, fif: u32) -> Result<Option<Tlas>> {
        let caps = device.caps();
        if caps.cluster.max_partitions == 0 || self.high_water == 0 {
            return Ok(None);
        }
        if self.storage.is_none() {
            self.storage = Some(self.create(device, fif)?);
        }
        let s = self.storage.as_ref().expect("made above");
        Ok(Some(Tlas(s.tlas.gpu.address().0)))
    }

    fn create(&self, device: &Device, fif: u32) -> Result<Storage> {
        let desc = desc(self.limits);
        let sizes = device.rtas_sizes(&desc);
        let dl = MemoryKind::DeviceLocal;
        let mut blocks = Vec::new();
        let mut block = |bytes: u64, name: &str| -> Result<GpuPtr<u8>> {
            let a = device.gpu_malloc::<u8>(dl, bytes.max(16), name)?;
            let gpu = a.gpu;
            blocks.push(a);
            Ok(gpu)
        };
        let instances = u64::from(self.limits.instances);
        let partitions = u64::from(self.limits.partitions);
        let stride = |a| u64::from(device.ptlas_args_stride(a));
        let pending = device.gpu_malloc::<PendingGpu>(
            dl,
            u64::from(self.limits.instances.max(1)),
            "PTLAS pending",
        )?;
        let writes = block(stride(PtlasArgs::Write) * instances, "PTLAS writes")?;
        let updates = block(stride(PtlasArgs::Update) * instances, "PTLAS updates")?;
        let translates = block(
            stride(PtlasArgs::Translate) * partitions,
            "PTLAS translations",
        )?;
        let commands = block(
            u64::from(device.rtas_args_stride(RtasOp::Ptlas)) * 3,
            "PTLAS commands",
        )?;
        let command_count = block(4, "PTLAS command count")?.cast();
        let state = block(4 * (partitions + 1), "partition state")?.cast();
        let buckets = block(4 * u64::from(DISTANCE_BUCKETS), "PTLAS buckets")?.cast();
        let counters = block(4 * COUNTERS as u64, "PTLAS counters")?.cast();
        let record = PtlasGpu {
            partitions: self.partition_table.gpu,
            slot_partition: self.slot_table.gpu,
            pending: pending.gpu,
            partition_state: state,
            buckets,
            counters,
            writes,
            updates,
            translates,
            commands,
            command_count,
            partition_count: self.limits.partitions,
            write_capacity: self.limits.instances,
            update_capacity: self.limits.instances,
            global_capacity: self.limits.global_instances,
        };
        let mut gpu = device.gpu_malloc::<PtlasGpu>(MemoryKind::Upload, 1, "PTLAS tables")?;
        gpu.as_mut_slice().expect("upload is mapped")[0] = record;
        Ok(Storage {
            tlas: device.gpu_malloc::<u8>(dl, sizes.result_max, "PTLAS")?,
            record: gpu,
            _blocks: blocks,
            _pending: pending,
            commands,
            command_count,
            counters,
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                u64::from(fif.max(1)),
                "PTLAS stats",
            )?,
            built: false,
        })
    }

    /// Scratch bytes a build of this PTLAS needs.
    pub(crate) fn scratch_bytes(&self, device: &Device) -> u64 {
        device.rtas_sizes(&desc(self.limits)).scratch
    }

    /// Scatter records of the partitions changed since the last frame.
    pub(crate) fn partition_records(
        &mut self,
        device: &Device,
        window: &UploadWindow,
        seq: &AtomicU64,
    ) -> Result<Vec<Record>> {
        std::mem::take(&mut self.changed)
            .into_iter()
            .map(|p| {
                let part = &self.partitions[p as usize];
                let value = PartitionGpu {
                    cell: part.cell.to_array(),
                    count: part.count,
                };
                let src = window.put(device, &value)?;
                let dst = self.partition_table.gpu.offset(u64::from(p));
                Ok(Record::values(
                    seq.fetch_add(1, Ordering::Relaxed),
                    dst,
                    src,
                    1,
                ))
            })
            .collect()
    }

    /// The figures of the frame that used readback slot `slot`.
    pub(crate) fn counters(&self, slot: usize) -> Option<([u32; COUNTERS], u32)> {
        let s = self.storage.as_ref()?;
        let c = s.readback.as_slice()?.get(slot)?;
        Some((*c, self.unplaced))
    }

    /// Records this frame's PTLAS work after the scene's tables and transforms are final.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &mut self,
        device: &Device,
        window: &UploadWindow,
        list: &mut CommandList<'_>,
        p: &Pipelines,
        k: &PtlasPipelines,
        f: &FrameInputs,
        scratch: GpuAddress,
    ) -> Result<()> {
        let Some(s) = self.storage.as_mut() else {
            return Ok(());
        };
        let jobs: Vec<TranslateJob> = if f.rebased || !s.built {
            (0..self.limits.partitions)
                .filter(|&i| self.partitions[i as usize].count > 0)
                .collect::<Vec<_>>()
        } else {
            self.assigned.iter().copied().collect()
        }
        .into_iter()
        .map(|i| TranslateJob {
            translation: partition_translation(
                self.partitions[i as usize].cell.to_array(),
                f.origin_cell.to_array(),
                f.cell_log2,
            ),
            partition: i,
        })
        .collect();
        self.assigned.clear();
        let translate_jobs = if jobs.is_empty() {
            GpuPtr::NULL
        } else {
            let (src, out) = window.slice::<TranslateJob>(device, jobs.len() as u32)?;
            out.copy_from_slice(&jobs);
            src
        };
        let root = PtlasRoot {
            scene: f.scene,
            ptlas: s.record.gpu,
            focus_cell: f.focus_cell,
            focus_offset: f.focus_offset,
            slots: self.high_water,
            budget: f.budget,
            flags: if f.rebased { PTLAS_REBASE } else { 0 },
            error_pixels: f.error_pixels,
            translate_jobs,
            translate_count: jobs.len() as u32,
            _pad: 0,
        };
        let root = window.put_root(device, &root)?;
        let groups = |n: u32| [n.div_ceil(64).max(1), 1, 1];
        let step = |list: &mut CommandList<'_>| {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        };
        let partitions = self.limits.partitions + 1;
        list.begin_pass("ptlas", false);
        list.dispatch(
            p.get(k.clear),
            root,
            groups(partitions.max(DISTANCE_BUCKETS).max(COUNTERS_CLEARED)),
        );
        step(list);
        list.dispatch(p.get(k.want), root, groups(self.high_water));
        step(list);
        list.dispatch(p.get(k.cutoff), root, [1, 1, 1]);
        step(list);
        list.dispatch(p.get(k.admit), root, groups(partitions));
        step(list);
        list.dispatch(p.get(k.emit), root, groups(self.high_water));
        if !jobs.is_empty() {
            list.dispatch(p.get(k.translate), root, groups(jobs.len() as u32));
        }
        step(list);
        list.dispatch(p.get(k.commands), root, [1, 1, 1]);
        list.barrier(
            Stages::COMPUTE,
            Stages::AS_BUILD | Stages::TRANSFER,
            Hazards::empty(),
        );
        let tlas = s.tlas.gpu.address();
        list.rtas_execute(
            &desc(self.limits),
            &RtasIo {
                args: s.commands.address(),
                count: s.command_count,
                dst_implicit: tlas,
                dst_addresses: GpuPtr::NULL,
                dst_sizes: GpuPtr::NULL,
                scratch,
                src: if s.built { tlas } else { GpuAddress(0) },
            },
        );
        let at = s.readback.gpu.offset(f.slot as u64);
        list.copy(at.cast(), s.counters.cast(), 4 * COUNTERS as u64);
        list.end_pass();
        s.built = true;
        Ok(())
    }
}

/// What one frame's PTLAS work reads from the frame.
pub(crate) struct FrameInputs {
    pub scene: GpuPtr<crate::SceneRoot>,
    pub focus_cell: [i32; 4],
    pub focus_offset: [f32; 4],
    pub origin_cell: IVec3,
    pub cell_log2: u32,
    pub rebased: bool,
    pub budget: u32,
    pub error_pixels: f32,
    /// The frame's readback slot.
    pub slot: usize,
}

fn desc(limits: PtlasLimits) -> RtasOpDesc {
    RtasOpDesc {
        op: RtasOp::Ptlas,
        mode: RtasMode::Implicit,
        flags: RtasFlags::FAST_TRACE,
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Ptlas(limits),
    }
}
