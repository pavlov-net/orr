//! A scene's GPU tables, its CPU bookkeeping and the writer consumers fill it through.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, IVec3};
use orr_gpu::{Allocation, Device, GpuAddress, GpuPtr, MemoryKind};

use crate::emissive::{ModelEmission, Run, Runs};
use crate::lights::{EMISSIVE_INSTANCE, EMISSIVE_PRODUCER, Light, LightId, light_gpu};
use crate::material::MaterialId;
use crate::memory::{RangeAllocator, UploadWindow};
use crate::producer::{self, Producer, ProducerDesc, ProducerId};
use crate::ptlas::Ptlas;
use crate::upload::{Op, Record};
use crate::world::WorldPos;
use crate::{
    EmissiveGpu, Error, GroupGpu, InstanceGpu, LightGpu, ProducerBatch, ProducerGpu, Result,
    SceneRoot, SkyStats, SlotStateGpu,
};

/// A scene in [`crate::Scenes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SceneId(pub u32);

/// Table sizes of a scene, fixed at creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneDesc {
    pub instance_capacity: u32,
    /// Transform groups, the null group included.
    pub group_capacity: u32,
    pub light_capacity: u32,
    /// `MaterialId` entries in the scene's material-set table.
    pub material_set_capacity: u32,
    /// Producer entries: every `ProducerId` is below it.
    pub producer_capacity: u32,
    /// Emissive list entries for instances: one per copy per EMISSIVE span of its model.
    pub emissive_capacity: u32,
    /// PTLAS instances for the animated parts of copies inside their animation range.
    pub part_capacity: u32,
    /// PTLAS cell partitions.
    pub ptlas_partitions: u32,
    /// Instances one PTLAS partition holds, at most.
    pub partition_instances: u32,
    /// Instances the PTLAS global partition holds, at most: movers beyond it stay in their
    /// cell partitions.
    pub global_instances: u32,
}

impl Default for SceneDesc {
    fn default() -> Self {
        Self {
            instance_capacity: 65_536,
            group_capacity: 1024,
            light_capacity: 256,
            material_set_capacity: 4096,
            producer_capacity: 256,
            emissive_capacity: 16_384,
            part_capacity: 524_288,
            ptlas_partitions: 2048,
            partition_instances: 1000,
            global_instances: 4096,
        }
    }
}

/// A 3x4 row-major affine transform, rows [R | t].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Affine3 {
    pub rows: [[f32; 4]; 3],
}

impl Affine3 {
    pub const IDENTITY: Self = Self {
        rows: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };
}

/// A transform group in one scene. Group 0 is the null group: it holds nothing and hides
/// whatever names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransformGroupId(pub u32);

/// Who writes a group's transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupWriter {
    /// `SceneWriter::transforms`; previous transforms are kept for motion.
    Cpu,
    /// Consumer kernels in scene compute, through `GroupGpu::current`. The previous
    /// frame's transforms become `previous` each frame the scene is updated, so a GPU
    /// writer rewrites every transform each such frame. A kernel removes an entry's motion
    /// (a teleport) by writing the same transform to `previous` as well.
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformGroupDesc {
    pub count: u32,
    pub writer: GroupWriter,
    /// The group's transforms are relative to this world position.
    pub anchor: DVec3,
}

/// Slots `first..first + count` of a scene's instance table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceRange {
    pub first: u32,
    pub count: u32,
}

/// A cooked model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelId(pub u32);

/// `InstanceGpu::flags`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceFlags(pub u8);

impl InstanceFlags {
    pub const RASTER: Self = Self(1);
    pub const RAY: Self = Self(2);
    /// Traced for shadows only.
    pub const CAST_ONLY: Self = Self(4);
}

impl std::ops::BitOr for InstanceFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Where a copy's part poses live:
/// `group[(index + copy * stride + pose_variant) * parts + part]`, with `copy` the copy's
/// transform index in its group and `parts` the mesh's part count. Each entry is the part's
/// mesh-space transform from its rest pose; the pose group's anchor is not used. Only
/// clusters of animated parts take their pose. An all-zero entry, or one past the group's
/// count, hides its part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pose {
    pub group: TransformGroupId,
    pub index: u32,
    pub variants: u32,
    pub stride: u32,
}

/// Per-copy variety: pose variant, material set and hidden sections (bit set = hidden).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct CopyVariant {
    pub pose: u8,
    pub material_set: u8,
    pub section_mask: u16,
}

/// `copies` copies of `model`, one instance slot each.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstancesDesc {
    pub model: ModelId,
    pub copies: u32,
    /// One anchor-relative transform per copy, from `first` on.
    pub transforms: TransformGroupId,
    pub first: u32,
    /// None: no animated parts.
    pub pose: Option<Pose>,
    /// Base in the scene's material-set table; None: the model's own materials.
    pub material_set: Option<u32>,
    pub flags: InstanceFlags,
    /// m from the scene focus within which the copies are traced.
    pub rt_range: f32,
    /// m from the scene focus within which the copies take their pose; beyond it they take
    /// the rest pose in raster and ray tracing.
    pub animation_range: f32,
    /// Never read by orr.
    pub user_tag: u32,
}

pub(crate) struct Tables {
    pub root: Allocation<SceneRoot>,
    pub instances: Allocation<InstanceGpu>,
    pub slot_state: Allocation<SlotStateGpu>,
    pub groups: Allocation<GroupGpu>,
    pub material_sets: Allocation<MaterialId>,
    pub lights: Allocation<LightGpu>,
    /// Producer entries first, then instance spans.
    pub emissive: Allocation<EmissiveGpu>,
    pub producers: Allocation<ProducerGpu>,
    pub batches: Allocation<ProducerBatch>,
    /// Cluster and vertex count of each producer batch.
    pub producer_counts: Allocation<[u32; 2]>,
    pub sky_stats: Allocation<SkyStats>,
}

pub(crate) struct Group {
    pub writer: GroupWriter,
    pub count: u32,
    pub anchor: WorldPos,
    /// `current` and the other buffer; GPU groups swap them every updated frame.
    pub buffers: [Allocation<u8>; 2],
    pub current: usize,
    /// `GroupGpu::previous` points at the current buffer.
    pub previous_is_current: bool,
    pub created_frame: u64,
    pub written: bool,
}

impl Group {
    pub(crate) fn record(&self) -> GroupGpu {
        let current = self.buffers[self.current].gpu;
        let previous = if self.previous_is_current {
            current
        } else {
            self.buffers[1 - self.current].gpu
        };
        GroupGpu {
            current,
            previous,
            count: self.count,
            format: 0,
            anchor_cell: self.anchor.cell.to_array(),
            anchor_offset: self.anchor.offset.to_array(),
            ..Zeroable::zeroed()
        }
    }
}

/// What writers know of a resident model.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ModelCpu {
    pub sections: u32,
    pub animated_parts: u32,
}

pub(crate) struct SceneCpu {
    pub slots: RangeAllocator,
    /// The model of each slot handed out.
    pub slot_models: Vec<u32>,
    /// Section masks copies name since the last frame: their BLAS build if missing.
    pub masks: HashSet<(ModelId, u16)>,
    /// Ranges removed in a frame, freed once that frame has completed.
    pub removed: Vec<(u64, InstanceRange)>,
    /// Instance entries of the emissive list, counted after the producer entries.
    pub emissive: RangeAllocator,
    pub emissive_runs: Runs,
    /// Entry ranges removed in a frame, freed once that frame has completed.
    pub emissive_removed: Vec<(u64, u32, u32)>,
    pub producers: Vec<Option<Producer>>,
    /// Memory of replaced or cleared producer entries, freed once that frame has completed.
    pub retired: Vec<(u64, Producer)>,
    /// Index = TransformGroupId; entry 0 is the null group.
    pub groups: Vec<Option<Group>>,
    pub records: Vec<Record>,
    pub focus: Option<(DVec3, f32)>,
    pub origin_cell: IVec3,
    pub environment: u64,
    pub medium: u64,
    pub surface_data: u64,
    pub near_range: f32,
    pub ptlas: Ptlas,
}

pub(crate) struct Scene {
    pub desc: SceneDesc,
    pub tables: Tables,
    pub cpu: Mutex<SceneCpu>,
    /// Set while a writer for this scene is alive.
    pub writing: AtomicBool,
}

/// 32-bit words in `bytes`.
fn words(bytes: usize) -> u32 {
    (bytes / 4) as u32
}

impl Scene {
    pub(crate) fn new(device: &Device, desc: &SceneDesc) -> Result<Self> {
        let n = |c: u32| u64::from(c.max(1));
        let dl = MemoryKind::DeviceLocal;
        let tables = Tables {
            root: device.gpu_malloc(dl, 1, "scene root")?,
            instances: device.gpu_malloc(dl, n(desc.instance_capacity), "instances")?,
            slot_state: device.gpu_malloc(dl, n(desc.instance_capacity), "slot state")?,
            groups: device.gpu_malloc(dl, n(desc.group_capacity), "transform groups")?,
            material_sets: device.gpu_malloc(dl, n(desc.material_set_capacity), "material sets")?,
            lights: device.gpu_malloc(dl, n(desc.light_capacity), "lights")?,
            emissive: device.gpu_malloc(
                dl,
                n(desc.producer_capacity + desc.emissive_capacity),
                "emissive list",
            )?,
            producers: device.gpu_malloc(dl, n(desc.producer_capacity), "producers")?,
            batches: device.gpu_malloc(dl, n(desc.producer_capacity), "producer batches")?,
            producer_counts: device.gpu_malloc(dl, n(desc.producer_capacity), "producer counts")?,
            sky_stats: device.gpu_malloc(dl, 1, "sky stats")?,
        };
        // Tables start zeroed: empty slots, the null group, no lights. These apply before
        // any write (sequence 0).
        let zero = |a: GpuAddress, bytes: u64| Record::fill(0, a.0, words(bytes as usize), 0);
        let records = vec![
            zero(tables.instances.gpu.address(), tables.instances.size),
            zero(tables.slot_state.gpu.address(), tables.slot_state.size),
            zero(tables.groups.gpu.address(), tables.groups.size),
            zero(
                tables.material_sets.gpu.address(),
                tables.material_sets.size,
            ),
            zero(tables.lights.gpu.address(), tables.lights.size),
            zero(tables.emissive.gpu.address(), tables.emissive.size),
            zero(tables.producers.gpu.address(), tables.producers.size),
            zero(tables.batches.gpu.address(), tables.batches.size),
            zero(
                tables.producer_counts.gpu.address(),
                tables.producer_counts.size,
            ),
            zero(tables.sky_stats.gpu.address(), tables.sky_stats.size),
        ];
        Ok(Self {
            desc: *desc,
            tables,
            cpu: Mutex::new(SceneCpu {
                slots: RangeAllocator::new(desc.instance_capacity),
                slot_models: vec![0; desc.instance_capacity as usize],
                masks: HashSet::new(),
                removed: Vec::new(),
                emissive: RangeAllocator::new(desc.emissive_capacity),
                emissive_runs: Runs::default(),
                emissive_removed: Vec::new(),
                producers: (0..desc.producer_capacity).map(|_| None).collect(),
                retired: Vec::new(),
                groups: vec![None],
                records,
                focus: None,
                origin_cell: IVec3::ZERO,
                environment: 0,
                medium: 0,
                surface_data: 0,
                near_range: 256.0,
                ptlas: Ptlas::new(device, desc)?,
            }),
            writing: AtomicBool::new(false),
        })
    }
}

/// Writes one scene's changes into this frame's upload window. Returned slices are
/// write-only (write-combined memory) and live until `update_scenes`. Send: different
/// scenes' writers, and disjoint pieces of one returned slice, can be filled on different
/// threads.
pub struct SceneWriter<'f> {
    pub(crate) id: SceneId,
    pub(crate) scene: &'f Scene,
    pub(crate) device: &'f Device,
    pub(crate) window: &'f UploadWindow,
    pub(crate) seq: &'f AtomicU64,
    pub(crate) frame: u64,
    pub(crate) cell_log2: u32,
    /// Emission by `ModelId`.
    pub(crate) models: &'f [Option<ModelEmission>],
    /// Sections and animated parts by `ModelId`.
    pub(crate) model_cpu: &'f [ModelCpu],
}

impl Drop for SceneWriter<'_> {
    fn drop(&mut self) {
        self.scene.writing.store(false, Ordering::Release);
    }
}

impl<'f> SceneWriter<'f> {
    pub fn id(&self) -> SceneId {
        self.id
    }

    /// Window space for `count` values of `T`, recorded as a copy to `dst`.
    fn upload<T: Pod>(&self, dst: GpuPtr<T>, count: u32) -> Result<&'f mut [T]> {
        let (src, out) = self.window.slice::<T>(self.device, count)?;
        self.push(Record::values(self.next_seq(), dst, src, count));
        Ok(out)
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    fn push(&self, r: Record) {
        self.scene.cpu.lock().unwrap().records.push(r);
    }

    fn write<T: Pod>(&self, dst: GpuPtr<T>, value: T) -> Result<()> {
        self.upload::<T>(dst, 1)?[0] = value;
        Ok(())
    }

    /// A new transform group; its transforms start all-zero (hidden).
    pub fn create_transform_group(
        &mut self,
        desc: &TransformGroupDesc,
    ) -> Result<TransformGroupId> {
        let bytes = u64::from(desc.count.max(1)) * size_of::<Affine3>() as u64;
        let buffers = [0, 1].map(|_| {
            self.device
                .gpu_malloc::<u8>(MemoryKind::DeviceLocal, bytes, "transforms")
        });
        let [a, b] = buffers;
        let group = Group {
            writer: desc.writer,
            count: desc.count,
            anchor: WorldPos::new(desc.anchor, self.cell_log2),
            buffers: [a?, b?],
            current: 0,
            previous_is_current: true,
            created_frame: self.frame,
            written: false,
        };
        let record = group.record();
        let zero = Record::fill(
            self.next_seq(),
            group.buffers[0].gpu.address().0,
            words(bytes as usize),
            0,
        );
        let id = {
            let mut cpu = self.scene.cpu.lock().unwrap();
            let id = cpu.groups.len() as u32;
            if id >= self.scene.desc.group_capacity {
                return Err(Error::Capacity {
                    what: "transform groups",
                    capacity: self.scene.desc.group_capacity,
                });
            }
            cpu.groups.push(Some(group));
            cpu.records.push(zero);
            id
        };
        self.write(self.scene.tables.groups.gpu.offset(u64::from(id)), record)?;
        Ok(TransformGroupId(id))
    }

    /// Transforms `first..first + count` of a CPU-written group, relative to its anchor.
    pub fn transforms(
        &mut self,
        group: TransformGroupId,
        first: u32,
        count: u32,
    ) -> Result<&'f mut [Affine3]> {
        self.write_transforms(group, first, count, false)
    }

    /// Like [`transforms`](Self::transforms), but these entries have no motion this frame:
    /// their previous transform is the one written here (teleports, respawns, entries
    /// taken over by new instances).
    pub fn transforms_without_motion(
        &mut self,
        group: TransformGroupId,
        first: u32,
        count: u32,
    ) -> Result<&'f mut [Affine3]> {
        self.write_transforms(group, first, count, true)
    }

    fn write_transforms(
        &mut self,
        group: TransformGroupId,
        first: u32,
        count: u32,
        without_motion: bool,
    ) -> Result<&'f mut [Affine3]> {
        let (current, previous) = {
            let mut cpu = self.scene.cpu.lock().unwrap();
            let g = cpu
                .groups
                .get_mut(group.0 as usize)
                .and_then(Option::as_mut)
                .unwrap_or_else(|| panic!("{group:?} does not exist"));
            assert_eq!(
                g.writer,
                GroupWriter::Cpu,
                "{group:?} is written on the GPU"
            );
            assert!(
                u64::from(first) + u64::from(count) <= u64::from(g.count),
                "transforms {first}+{count} outside {group:?} of {}",
                g.count
            );
            g.written = true;
            let at = |b: &Allocation<u8>| b.gpu.cast::<Affine3>().offset(u64::from(first));
            (at(&g.buffers[g.current]), at(&g.buffers[1 - g.current]))
        };
        let (src, out) = self.window.slice::<Affine3>(self.device, count)?;
        self.push(Record::values(self.next_seq(), current, src, count));
        // The current-to-previous copy runs before the scatter, so this lands on top of it.
        if without_motion {
            self.push(Record::values(self.next_seq(), previous, src, count));
        }
        Ok(out)
    }

    fn instance_word(&self, slot: u32, byte: usize) -> u64 {
        self.scene
            .tables
            .instances
            .gpu
            .offset(u64::from(slot))
            .address()
            .0
            + byte as u64
    }

    /// `desc.copies` new instance slots, and an emissive list entry per copy for each
    /// EMISSIVE span of the model. With a pose, each copy's PTLAS partition keeps room for
    /// its part run.
    pub fn add_instances(&mut self, desc: &InstancesDesc) -> Result<InstanceRange> {
        let emission = self
            .models
            .get(desc.model.0 as usize)
            .and_then(Option::as_ref)
            .filter(|m| !m.spans.is_empty());
        let (first, entry, partitions, table) = {
            let cpu = &mut *self.scene.cpu.lock().unwrap();
            let first = cpu.slots.alloc(desc.copies).ok_or(Error::Capacity {
                what: "instances",
                capacity: self.scene.desc.instance_capacity,
            })?;
            let cell = cpu
                .groups
                .get(desc.transforms.0 as usize)
                .and_then(Option::as_ref)
                .map_or(IVec3::ZERO, |g| g.anchor.cell);
            let entry = match emission {
                None => None,
                Some(m) => {
                    let spans = m.spans.len() as u32;
                    let Some(entry) = cpu.emissive.alloc(desc.copies * spans) else {
                        cpu.slots.free(first, desc.copies);
                        return Err(Error::Capacity {
                            what: "emissive list entries",
                            capacity: self.scene.desc.emissive_capacity,
                        });
                    };
                    let run = Run {
                        copies: desc.copies,
                        entry,
                        spans,
                    };
                    cpu.emissive_runs.insert(first, run);
                    Some(entry)
                }
            };
            let parts = self
                .model_cpu
                .get(desc.model.0 as usize)
                .map_or(0, |m| m.animated_parts);
            let cost = 1 + if desc.pose.is_some() { parts } else { 0 };
            cpu.slot_models[first as usize..(first + desc.copies) as usize].fill(desc.model.0);
            let partitions = cpu.ptlas.place(first, desc.copies, cell, cost);
            (first, entry, partitions, cpu.ptlas.slot_table.gpu)
        };
        self.upload(table.offset(u64::from(first)), desc.copies)?
            .copy_from_slice(&partitions);
        if let (Some(m), Some(entry)) = (emission, entry) {
            let entries = self.upload(
                self.emissive_entry(entry),
                desc.copies * m.spans.len() as u32,
            )?;
            for (e, (copy, span)) in entries
                .iter_mut()
                .zip((0..desc.copies).flat_map(|c| m.spans.iter().map(move |s| (c, s))))
            {
                *e = EmissiveGpu {
                    kind: EMISSIVE_INSTANCE,
                    source: first + copy,
                    first_cluster: span.first_cluster,
                    cluster_count: span.cluster_count,
                    section: u32::from(span.section),
                    sections: m.sections,
                    material: m.materials[usize::from(span.section)].0,
                    model_area: span.area,
                    ..Zeroable::zeroed()
                };
            }
        }
        let range = InstanceRange {
            first,
            count: desc.copies,
        };
        let pose = desc.pose.unwrap_or(Pose {
            group: TransformGroupId(0),
            index: 0,
            variants: 0,
            stride: 0,
        });
        let records = self.upload(
            self.scene.tables.instances.gpu.offset(u64::from(first)),
            desc.copies,
        )?;
        for (copy, r) in records.iter_mut().enumerate() {
            *r = InstanceGpu {
                group: desc.transforms.0,
                group_index: desc.first + copy as u32,
                model: desc.model.0,
                material_base: desc.material_set.unwrap_or(u32::MAX),
                pose_group: pose.group.0,
                pose_index: pose.index,
                first_part: u32::MAX,
                rt_range: half::f16::from_f32(desc.rt_range).to_bits(),
                animation_range: half::f16::from_f32(desc.animation_range).to_bits(),
                flags: desc.flags.0,
                candidate_class: 0,
                pose_variants: pose.variants as u16,
                pose_stride: pose.stride as u16,
                _pad: 0,
                user_tag: desc.user_tag,
                variant: 0,
            };
        }
        // The fields from `part_run` on describe a run the PTLAS kernels still release.
        let state = self
            .scene
            .tables
            .slot_state
            .gpu
            .offset(u64::from(first))
            .address()
            .0;
        let seq = self.next_seq();
        for word in 0..words(std::mem::offset_of!(SlotStateGpu, part_run)) {
            self.push(Record {
                seq,
                dst: state + 4 * u64::from(word),
                count: desc.copies,
                stride: words(size_of::<SlotStateGpu>()),
                op: Op::Fill(0),
            });
        }
        Ok(range)
    }

    /// Replaces the flags of every copy in `range`.
    pub fn set_instance_flags(&mut self, range: InstanceRange, flags: InstanceFlags) {
        let byte = std::mem::offset_of!(InstanceGpu, flags);
        let shift = 8 * (byte % 4);
        self.push(Record {
            seq: self.next_seq(),
            dst: self.instance_word(range.first, byte & !3),
            count: range.count,
            stride: words(size_of::<InstanceGpu>()),
            op: Op::Masked {
                value: u32::from(flags.0) << shift,
                mask: 0xff << shift,
            },
        });
    }

    /// Sets the variant words of copies `first..` of `range`. Ray tracing hides a copy's
    /// masked sections through BLAS built per (model, cut, mask) the first frame a copy
    /// names the mask; a mask hides an animated part's instance only when it covers every
    /// section of the part.
    pub fn copy_variants(
        &mut self,
        range: InstanceRange,
        first: u32,
        variants: &[CopyVariant],
    ) -> Result<()> {
        let count = variants.len() as u32;
        assert!(
            first + count <= range.count,
            "variants {first}+{count} outside {range:?}"
        );
        let (src, out) = self.window.slice::<CopyVariant>(self.device, count)?;
        out.copy_from_slice(variants);
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        let slot = (range.first + first) as usize;
        for (v, &model) in variants.iter().zip(&cpu.slot_models[slot..]) {
            let sections = self.model_cpu.get(model as usize).map_or(0, |m| m.sections);
            let mask = u32::from(v.section_mask) & ((1 << sections.min(16)) - 1);
            if mask != 0 {
                cpu.masks.insert((ModelId(model), mask as u16));
            }
        }
        cpu.records.push(Record {
            seq: self.next_seq(),
            dst: self.instance_word(
                range.first + first,
                std::mem::offset_of!(InstanceGpu, variant),
            ),
            count,
            stride: words(size_of::<InstanceGpu>()),
            op: Op::Copy(src.address().0),
        });
        Ok(())
    }

    /// Entries `first..first + count` of the scene's material-set table.
    pub fn material_sets(&mut self, first: u32, count: u32) -> Result<&'f mut [MaterialId]> {
        assert!(
            u64::from(first) + u64::from(count) <= u64::from(self.scene.desc.material_set_capacity),
            "material sets {first}+{count} outside the table"
        );
        self.upload(
            self.scene.tables.material_sets.gpu.offset(u64::from(first)),
            count,
        )
    }

    /// Instance entry `entry` of the emissive list.
    fn emissive_entry(&self, entry: u32) -> GpuPtr<EmissiveGpu> {
        self.scene
            .tables
            .emissive
            .gpu
            .offset(u64::from(self.scene.desc.producer_capacity + entry))
    }

    /// Empties the slots of `range` and their emissive list entries; they are reused once
    /// this frame has completed.
    pub fn remove(&mut self, range: InstanceRange) {
        let n = range.count * words(size_of::<InstanceGpu>());
        let seq = self.next_seq();
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        cpu.records
            .push(Record::fill(seq, self.instance_word(range.first, 0), n, 0));
        cpu.removed.push((self.frame, range));
        cpu.ptlas.remove(range.first, range.count, self.frame);
        for (entry, count) in cpu.emissive_runs.remove(range.first, range.count) {
            let at = self.emissive_entry(entry).address().0;
            let n = count * words(size_of::<EmissiveGpu>());
            cpu.records.push(Record::fill(seq, at, n, 0));
            cpu.emissive_removed.push((self.frame, entry, count));
        }
    }

    /// Sets or clears producer entry `id`. Setting allocates the entry's batch and resets
    /// its counts; the previous batch, if any, is freed once this frame has completed.
    pub fn set_producer(&mut self, id: ProducerId, desc: Option<&ProducerDesc<'_>>) -> Result<()> {
        assert!(
            id.0 < self.scene.desc.producer_capacity,
            "{id:?} outside the producer table"
        );
        let t = &self.scene.tables;
        let i = u64::from(id.0);
        let counts = t.producer_counts.gpu.offset(i);
        let (record, batch, entry, producer) = match desc {
            None => (
                ProducerGpu::zeroed(),
                ProducerBatch::zeroed(),
                EmissiveGpu::zeroed(),
                None,
            ),
            Some(d) => {
                let b = producer::build(self.device, d, counts)?;
                let words = self.upload(b.static_at, (b.static_bytes.len() / 4) as u32)?;
                bytemuck::cast_slice_mut(words).copy_from_slice(&b.static_bytes);
                let entry = EmissiveGpu {
                    kind: EMISSIVE_PRODUCER,
                    source: id.0,
                    ..Zeroable::zeroed()
                };
                (b.record, b.batch, entry, Some(b.producer))
            }
        };
        self.write(t.producers.gpu.offset(i), record)?;
        self.write(t.batches.gpu.offset(i), batch)?;
        self.write(t.emissive.gpu.offset(i), entry)?;
        self.push(Record::fill(self.next_seq(), counts.address().0, 2, 0));
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        if let Some(old) = std::mem::replace(&mut cpu.producers[id.0 as usize], producer) {
            cpu.retired.push((self.frame, old));
        }
        Ok(())
    }

    /// Sets or clears light `id`.
    pub fn set_light(&mut self, id: LightId, light: Option<&Light>) -> Result<()> {
        assert!(
            id.0 < self.scene.desc.light_capacity,
            "{id:?} outside the light table"
        );
        let origin = self.scene.cpu.lock().unwrap().origin_cell;
        let record = light.map_or(LightGpu::zeroed(), |l| light_gpu(l, origin, self.cell_log2));
        self.write(self.scene.tables.lights.gpu.offset(u64::from(id.0)), record)
    }

    /// Consumer data for the linked environment.
    pub fn set_environment(&mut self, data: GpuAddress) {
        self.scene.cpu.lock().unwrap().environment = data.0;
    }

    /// Consumer data for the linked medium.
    pub fn set_medium(&mut self, data: GpuAddress) {
        self.scene.cpu.lock().unwrap().medium = data.0;
    }

    /// Consumer data for linked surfaces whose material names none (`MaterialGpu::data`).
    pub fn set_surface_data(&mut self, data: GpuAddress) {
        self.scene.cpu.lock().unwrap().surface_data = data.0;
    }

    /// The primary view's camera and pixels per radian (render height / vertical fov):
    /// the eye of the scene's per-frame work and the anchor of its render origin.
    pub fn set_focus(&mut self, focus: DVec3, pixels_per_radian: f32) {
        self.scene.cpu.lock().unwrap().focus = Some((focus, pixels_per_radian));
    }
}
