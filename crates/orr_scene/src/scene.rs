//! A scene's GPU tables, its CPU bookkeeping and the writer consumers fill it through.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use orr_gpu::{Allocation, Device, GpuAddress, GpuPtr, KernelSet, MemoryKind};
use orr_math::{DVec3, IVec3};
use rustc_hash::FxHashSet;

use crate::decals::{Decal, DecalId, Listing, decal_gpu, listing, table_sizes};
use crate::lights::{
    EMISSIVE_INSTANCE, EMISSIVE_PRODUCER, GATHERED_CLUSTERS, Light, LightId, NO_EMISSIVE, NO_SET,
    light_gpu,
};
use crate::material::{MaterialId, MaterialTable};
use crate::medium_proxy::{MediumProxy, MediumProxyId, proxy_gpu};
use crate::memory::{RangeAllocator, UploadWindow};
use crate::mount::{InstancePage, Mounts, PageId, PagePrototype};
use crate::producer::{self, Opacity, Producer, ProducerDesc, ProducerId, ProducerUpdate};
use crate::ptlas::Ptlas;
use crate::shaders::instance_page::{PageEntriesGpu, PagePrototypeGpu};
use crate::skin::{CopyArrays, SkinFrame, SkinnedDesc};
use crate::subd::{self, ModelCage, SubdBuild, SubdDesc};
use crate::tessellate::Patterns;
use crate::traverse::Cuts;
use crate::upload::{Op, Record};
use crate::view::ViewHistoryId;
use crate::world::WorldPos;
use crate::{
    DecalBinGpu, DecalGpu, EmissiveGpu, Error, GroupGpu, InstanceGpu, LightGpu, MediumProxyGpu,
    ProducerBatch, ProducerGpu, ProducerStateGpu, Result, SceneRoot, SkinStateGpu, SkinnedGpu,
    SkyStats, SlotStateGpu, TessStateGpu, ViewConstants,
};

/// `SlotHold::model` of a slot no model placed by hand holds.
const NO_MODEL: u32 = u32::MAX;

/// `tables.slang` `NO_MATERIAL_SET`.
const NO_MATERIAL_SET: u32 = u32::MAX;

/// Seconds of an animated page's rows wanted ahead of its clock.
const MOTION_AHEAD: f32 = 1.0;

/// `tables.slang` `GROUP_QUANTIZED_PAGE`.
const GROUP_QUANTIZED_PAGE: u32 = 1;

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
    /// Emissive list entries for instances: one per copy per emissive and per caster section of
    /// the chain step of its model that has the most of them.
    pub emissive_capacity: u32,
    /// PTLAS instances for the animated parts of copies inside their animation range, and
    /// one for each copy fading between two chain steps. At most 2^24.
    pub part_capacity: u32,
    /// PTLAS cell partitions.
    pub ptlas_partitions: u32,
    /// Instances the copies of one PTLAS partition count, at most: their bodies and part runs.
    /// A copy fading between two chain steps holds one more.
    pub partition_instances: u32,
    /// Instances the PTLAS global partition holds, at most: movers beyond it stay in their
    /// cell partitions.
    pub global_instances: u32,
    /// Skinned copies (`add_skinned`), at most.
    pub skinned_capacity: u32,
    /// Clusters all skinned copies skin and instantiate in one frame, at most.
    pub skinned_clusters: u32,
    /// Decals: every `DecalId` is below it.
    pub decal_capacity: u32,
    /// Medium proxies: every `MediumProxyId` is below it. They trace from the PTLAS global
    /// partition.
    pub medium_proxy_capacity: u32,
    /// `MaterialId` entries for the sets of cooked page prototypes that override materials,
    /// after the material-set table's `material_set_capacity` entries.
    pub page_material_sets: u32,
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
            skinned_capacity: 256,
            skinned_clusters: 8192,
            decal_capacity: 1024,
            medium_proxy_capacity: 64,
            page_material_sets: 4096,
        }
    }
}

pub use orr_pack::AffineRows;

/// What placing an instance page takes of a scene's capacities ([`SceneDesc`]), with the chains
/// its models have now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageNeeds {
    /// Of `instance_capacity`: one per record.
    pub instances: u32,
    /// Of `page_material_sets`: the sets of the prototypes that override materials. A placement
    /// takes none for a prototype whose set `InstancePageRef::material_sets` replaces, so this
    /// is the most it takes.
    pub material_sets: u32,
    /// Of `emissive_capacity`: every copy takes room for the emissive and caster sections of
    /// the most emissive step of its model's chain.
    pub emissive: u32,
}

/// What the copies of one prototype of a placed page take of the emissive list: `stride`
/// entries each for its `records`, after the entries of the prototypes before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrototypeEntries {
    pub(crate) records: u32,
    pub(crate) stride: u32,
}

/// The emissive list entries of a page whose prototypes take `entries`.
fn page_entries(entries: &[PrototypeEntries]) -> u32 {
    entries.iter().map(|e| e.records * e.stride).sum()
}

/// What placing page `id` takes, the emissive entries of each of its prototypes, and its
/// prototypes.
pub(crate) fn page_needs(
    mounts: &Mounts,
    model_cpu: &[ModelCpu],
    id: PageId,
) -> Result<(PageNeeds, Vec<PrototypeEntries>, Vec<PagePrototype>)> {
    let page = mounts.page(id).ok_or(Error::UnknownPage(id))?;
    let instances = u32::from(page.info.header.record_count);
    let protos = mounts.page_prototypes(id)?;
    let material_sets = protos
        .iter()
        .map(|p| p.sets.as_ref().map_or(0, |s| s.len() as u32))
        .sum();
    let entries: Vec<PrototypeEntries> = protos
        .iter()
        .map(|p| PrototypeEntries {
            records: p.records,
            stride: model_cpu
                .get(p.model.0 as usize)
                .map_or(0, |m| m.emissive_stride),
        })
        .collect();
    let needs = PageNeeds {
        instances,
        material_sets,
        emissive: page_entries(&entries),
    };
    Ok((needs, entries, protos))
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
    /// A cooked instance page's records, from [`SceneWriter::add_instance_page`]. Its copies
    /// are hidden while the page is not resident.
    Page,
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

/// A debug hold on the level of detail of copies, in raster and ray tracing alike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DebugLod {
    /// DAG meshes draw the part stage's clusters made by this many simplification steps
    /// (and coarser terminal groups where the DAG stops earlier); discrete meshes draw this
    /// level, or their coarsest.
    Level(u8),
    /// Scales the error threshold: above 1 coarser, below 1 finer.
    ErrorScale(f32),
}

impl DebugLod {
    /// `InstanceGpu::debug_lod`.
    fn bits(lod: Option<Self>) -> u16 {
        const LEVEL: u16 = 0x8000;
        match lod {
            None => 0,
            Some(Self::Level(l)) => LEVEL | u16::from(l),
            // A positive f16 below LEVEL; 0 stands for none.
            Some(Self::ErrorScale(s)) => half::f16::from_f32(s.clamp(1e-4, 65504.0)).to_bits(),
        }
    }
}

/// `InstanceGpu::flags`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceFlags(pub u8);

impl InstanceFlags {
    pub const RASTER: Self = Self(1);
    pub const RAY: Self = Self(2);
    /// Traced for shadows only.
    pub const CAST_ONLY: Self = Self(4);
    /// The bits consumers set; orr keeps the others.
    const CONSUMER: u8 = 7;
    /// `tables.slang` `INSTANCE_SKINNED`.
    const SKINNED: u8 = 8;
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

/// A cooked instance page placed in a scene with [`SceneWriter::add_instance_page`].
#[derive(Clone, Copy, Debug)]
pub struct InstancePageRef<'a> {
    pub page: PageId,
    /// The palette group and entry of the joint the page hangs from, for a page that names one
    /// (`InstancePageHeader::frame_skeleton`): the entry takes its records from the joint's bind
    /// space to its pose. None places them as stored.
    pub frame: Option<(TransformGroupId, u32)>,
    /// Where copies of each model take their part poses; the copy's index in its pose block
    /// is its record index. Models not listed take none.
    pub poses: &'a [(ModelId, Pose)],
    /// Replaces a prototype's material set by a base in the scene's material-set table, the
    /// prototype by its index in the page's prototypes, each at most once.
    pub material_sets: &'a [(u16, u32)],
    /// A copy's flags are its prototype's and these; a prototype's flags of 0 take these.
    pub flags: InstanceFlags,
    /// m from the scene focus within which the copies are traced.
    pub rt_range: f32,
    /// m from the scene focus within which the copies take their pose.
    pub animation_range: f32,
    /// Never read by orr.
    pub user_tag: u32,
}

/// A cooked instance page of a skinned model, placed in a scene with
/// [`SceneWriter::add_skinned_page`]: its records are the copies, as [`SkinnedDesc`] places them,
/// each taking `joints` palette entries from `palette_first` on in record order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkinnedPageRef {
    pub page: PageId,
    /// As [`InstancePageRef::frame`].
    pub frame: Option<(TransformGroupId, u32)>,
    pub palette: TransformGroupId,
    pub palette_first: u32,
    /// Base in the scene's material-set table; None: the model's own materials.
    pub material_set: Option<u32>,
    pub flags: InstanceFlags,
    /// m from the scene focus within which the copies are traced.
    pub rt_range: f32,
    /// Never read by orr.
    pub user_tag: u32,
}

/// Per-copy variety: pose variant, material set and hidden sections (bit set = hidden). Set `k`
/// of a copy whose base is `b` and whose model has `n` sections is entries
/// `b + k * n..b + (k + 1) * n` of the scene's material-set table; a cooked page's base holds set
/// 0 only.
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
    /// Base in the scene's material-set table; None: the model's own materials. Light sampling
    /// takes what the set's materials emit on the sections the model cooked as emissive; a
    /// set that makes another section emit leaves it to BSDF rays, at full weight.
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
    pub(crate) root: Allocation<SceneRoot>,
    pub(crate) instances: Allocation<InstanceGpu>,
    pub(crate) slot_state: Allocation<SlotStateGpu>,
    pub(crate) groups: Allocation<GroupGpu>,
    pub(crate) material_sets: Allocation<MaterialId>,
    pub(crate) lights: Allocation<LightGpu>,
    /// Producer entries first, then instance entries.
    pub(crate) emissive: Allocation<EmissiveGpu>,
    /// Per instance entry, the list its copy's traversal gathers (`SceneRoot.emissive_lists`).
    pub(crate) emissive_lists: Allocation<u32>,
    pub(crate) producers: Allocation<ProducerGpu>,
    pub(crate) batches: Allocation<ProducerBatch>,
    /// Tessellated entries' patch batches, zero for others.
    pub(crate) patches: Allocation<ProducerBatch>,
    pub(crate) tess_state: Allocation<TessStateGpu>,
    /// Cluster, vertex and segment count of each producer batch, and a spare word.
    pub(crate) producer_counts: Allocation<[u32; 4]>,
    pub(crate) producer_state: Allocation<ProducerStateGpu>,
    pub(crate) sky_stats: Allocation<SkyStats>,
    pub(crate) skinned: Allocation<SkinnedGpu>,
    pub(crate) skin_state: Allocation<SkinStateGpu>,
    pub(crate) decals: Allocation<DecalGpu>,
    pub(crate) decal_bins: Allocation<DecalBinGpu>,
    pub(crate) decal_refs: Allocation<u32>,
    pub(crate) proxies: Allocation<MediumProxyGpu>,
}

pub(crate) struct Group {
    pub(crate) writer: GroupWriter,
    count: u32,
    anchor: WorldPos,
    /// `current` and the other buffer; GPU groups swap them every updated frame.
    pub(crate) buffers: Vec<Allocation<u8>>,
    pub(crate) current: usize,
    /// `GroupGpu::previous` points at the current buffer.
    pub(crate) previous_is_current: bool,
    pub(crate) created_frame: u64,
    pub(crate) written: bool,
    /// A page group's record; its buffers are empty.
    pub(crate) page: Option<PageGroup>,
    /// The joint it hangs from.
    pub(crate) frame: Option<GroupFrame>,
}

/// [`GroupGpu::frame`], `frame_index` and `frame_origin`.
#[derive(Clone, Copy)]
pub(crate) struct GroupFrame {
    palette: GpuPtr<GroupGpu>,
    index: u32,
    origin: [f32; 3],
}

/// An animated page placed in a scene.
pub(crate) struct PageMotion {
    pub(crate) range: InstanceRange,
    /// Names no page once its pack unmounts, when its copies hide for good.
    pub(crate) page: PageId,
    /// The page's records.
    pub(crate) records: TransformGroupId,
    /// The transforms its copies read.
    pub(crate) group: TransformGroupId,
    pub(crate) times: Vec<f32>,
    pub(crate) left: [[f32; 4]; 3],
    pub(crate) right: [[f32; 4]; 3],
    pub(crate) count: u32,
    /// s of the page's clock, from [`SceneWriter::set_page_time`].
    pub(crate) time: f32,
    /// Whether the clock last ran forward, the way its windows stream ahead.
    pub(crate) forward: bool,
    /// The interval and seconds into it last posed, where copies hold while the window of the
    /// clock is on its way.
    pub(crate) held: Option<(u32, f32)>,
}

impl PageMotion {
    /// The windows of rows wanted now, with their rank: the clock's, then the one copies hold in
    /// while that is on its way, then those [`MOTION_AHEAD`] seconds on along the clock. A pool
    /// with room for one window gives it to the clock's, or copies would hold for good.
    pub(crate) fn windows(&self, per_window: u32) -> Vec<(u32, f64)> {
        let window =
            |t: f32| orr_pack::instance::motion_interval(&self.times, t).0 as u32 / per_window;
        let now = window(self.time);
        let last = (self.times.len() as u32 - 1) / per_window;
        let (first, end) = if self.forward {
            let far = window(self.time + MOTION_AHEAD);
            (now, far.max(now + 1).min(last))
        } else {
            let far = window(self.time - MOTION_AHEAD);
            (far.min(now.saturating_sub(1)), now)
        };
        let rank = |w: u32| f64::from(w.abs_diff(now));
        let mut out: Vec<(u32, f64)> = (first..=end).map(|w| (w, rank(w))).collect();
        out.extend(self.held.map(|h| (h.0 / per_window, 0.5)));
        out
    }
}

/// What a cooked page's transform group holds.
#[derive(Clone, Copy)]
pub(crate) struct PageGroup {
    pub(crate) page: PageId,
    /// Where its records are resident; 0: not resident, which hides its copies.
    pub(crate) records: u64,
    /// `records` changed since the group record was last written.
    pub(crate) changed: bool,
    origin: [f32; 3],
    step: [f32; 3],
    log2_scale_min: f32,
    log2_scale_step: f32,
}

impl Group {
    /// The group of `page`'s records.
    fn of_page(id: PageId, page: &InstancePage, frame: u64) -> Self {
        let header = page.info.header;
        Self {
            writer: GroupWriter::Page,
            count: u32::from(header.record_count),
            anchor: WorldPos {
                cell: IVec3::from_array(header.cell),
                offset: orr_math::Vec3::ZERO,
            },
            buffers: Vec::new(),
            current: 0,
            previous_is_current: true,
            created_frame: frame,
            written: false,
            frame: None,
            page: Some(PageGroup {
                page: id,
                records: page.records.address().0,
                changed: false,
                origin: header.origin,
                step: header.step,
                log2_scale_min: header.log2_scale_min,
                log2_scale_step: header.log2_scale_step,
            }),
        }
    }

    pub(crate) fn record(&self) -> GroupGpu {
        let frame = self.frame.unwrap_or(GroupFrame {
            palette: GpuPtr::NULL,
            index: 0,
            origin: [0.0; 3],
        });
        if let Some(p) = &self.page {
            // SAFETY: the records stay resident while this record names them: their eviction
            // and their pack's unmount write it null first.
            let records = unsafe { GpuPtr::from_address(GpuAddress(p.records)) };
            return GroupGpu {
                frame: frame.palette.cast(),
                frame_index: frame.index,
                frame_origin: frame.origin,
                current: records,
                previous: records,
                count: self.count,
                format: GROUP_QUANTIZED_PAGE,
                anchor_cell: self.anchor.cell.to_array(),
                anchor_offset: self.anchor.offset.to_array(),
                origin: p.origin,
                step: p.step,
                log2_scale_min: p.log2_scale_min,
                log2_scale_step: p.log2_scale_step,
            };
        }
        let current = self.buffers[self.current].gpu();
        let previous = if self.previous_is_current {
            current
        } else {
            self.buffers[1 - self.current].gpu()
        };
        GroupGpu {
            current,
            previous,
            count: self.count,
            format: 0,
            anchor_cell: self.anchor.cell.to_array(),
            anchor_offset: self.anchor.offset.to_array(),
            frame: frame.palette.cast(),
            frame_index: frame.index,
            frame_origin: frame.origin,
            ..Zeroable::zeroed()
        }
    }
}

/// What writers know of a resident model.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ModelCpu {
    pub(crate) sections: u32,
    pub(crate) animated_parts: u32,
    /// Its mesh's emissive section ranks and then its caster section ranks, an emissive list
    /// entry each.
    pub(crate) emissive_sections: u32,
    /// The emissive list entries of each copy: the most of `emissive_sections` of its mesh and
    /// of its chain's.
    pub(crate) emissive_stride: u32,
    /// Its skeleton's joints; 0 without a skeleton.
    pub(crate) joints: u32,
    /// Set when its mesh is skinned.
    pub(crate) skin: Option<SkinShape>,
    /// It has a subdivision cage and no mesh.
    pub(crate) cage_only: bool,
    /// It is an edit of another model's mesh.
    pub(crate) edit: bool,
}

/// A cooked page placed in a scene: its prototypes until its records first land, which write
/// its slots, and what its removal frees. Its copies' emissive list entries are one range from
/// `first_entry`, by prototype, and in a prototype's by record rank; only its records say which
/// slot holds which.
pub(crate) struct PagePlacement {
    pub(crate) range: InstanceRange,
    pub(crate) group: TransformGroupId,
    pub(crate) page: PageId,
    pub(crate) prototypes: Allocation<u8>,
    pub(crate) prototype_count: u32,
    pub(crate) first_entry: u32,
    pub(crate) exact_rows: u32,
    pub(crate) entries: Vec<PrototypeEntries>,
    pub(crate) expanded: bool,
    /// Its slots hold no entries since its range moved, and take their new ones the next time
    /// its records are resident.
    pub(crate) entries_stale: bool,
    /// Its prototypes' entries of the page part of the material-set table.
    pub(crate) sets: (u32, u32),
}

impl PagePlacement {
    /// Where each prototype's copies take their entries, as its expansion reads it.
    pub(crate) fn entry_layout(&self) -> Vec<PageEntriesGpu> {
        let mut first = 0;
        self.entries
            .iter()
            .map(|e| {
                let at = first;
                first += e.records * e.stride;
                PageEntriesGpu {
                    first: at,
                    count: e.stride,
                }
            })
            .collect()
    }
}

/// A cooked page of skinned copies placed in a scene.
pub(crate) struct SkinnedPage {
    pub(crate) range: InstanceRange,
    pub(crate) group: TransformGroupId,
    pub(crate) page: PageId,
    /// Its prototype's entries of the page part of the material-set table.
    pub(crate) sets: (u32, u32),
}

/// The vertex and cluster counts of a skinned mesh.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SkinShape {
    pub(crate) vertices: u32,
    pub(crate) clusters: u32,
}

/// The copies of one `add_skinned` call: their slots, skinned-copy entries and vertex arrays.
struct SkinRange {
    slots: InstanceRange,
    entry: u32,
    arrays: CopyArrays,
    /// Copies not yet removed.
    live: u32,
}

/// What a scene chooses geometry for before any view has asked ([`crate::Frame::request_detail`]).
const DEFAULT_PIXELS_PER_RADIAN: f32 = 1000.0;

/// Emissive list ranges taken for a chain that grows `model`'s copies to `stride` entries:
/// one per slot, then one per page with its prototypes' new entries.
pub(crate) struct EntryGrowth {
    model: u32,
    stride: u32,
    slots: Vec<u32>,
    pages: Vec<(usize, Vec<PrototypeEntries>)>,
    taken: Vec<(u32, u32)>,
}

/// What a slot holds.
#[derive(Clone, Copy)]
struct SlotHold {
    model: u32,
    /// Its entries of the instance part of the emissive list; none on a page, which holds its
    /// copies' entries.
    first_entry: u32,
    entries: u32,
    /// Its copy's `InstanceGpu::material_base`.
    material_base: u32,
}

impl SlotHold {
    /// A slot that holds nothing.
    const FREE: Self = Self::of(NO_MODEL);

    const fn of(model: u32) -> Self {
        Self {
            model,
            first_entry: 0,
            entries: 0,
            material_base: NO_MATERIAL_SET,
        }
    }
}

impl SceneCpu {
    /// What the scene itself may hold, beside what the material table's records may show.
    pub(crate) fn features(&self) -> crate::SceneFeatures {
        let mut out = self.lights.features();
        if self.ptlas.has_medium_proxies() {
            out |= crate::SceneFeatures::MEDIUM_PROXIES;
        }
        if self.producers.iter().flatten().any(|p| p.smoke) {
            out |= crate::SceneFeatures::SMOKE;
        }
        // A replaced entry's geometry stays until its frame has completed.
        let retired = self.retired.iter().map(|(_, p)| p);
        for p in self.producers.iter().flatten().chain(retired) {
            out |= p.features;
        }
        // The bins a frame's kernels read are remade from the list once it has changed.
        if self.decals_changed || self.decal_globals > 0 || self.decal_bin_slots > 0 {
            out |= crate::SceneFeatures::DECALS;
        }
        if self.hung || self.skinned_high_water > 0 {
            out |= crate::SceneFeatures::SKINNING;
        }
        out
    }
}

pub(crate) struct SceneCpu {
    pub(crate) slots: RangeAllocator,
    /// By slot.
    slot_holds: Vec<SlotHold>,
    /// Section masks copies name since the last frame: their BLAS build if missing.
    pub(crate) masks: HashSet<(ModelId, u16)>,
    /// Ranges removed in a frame, freed once that frame has completed.
    pub(crate) removed: Vec<(u64, InstanceRange)>,
    /// Instance entries of the emissive list, counted after the producer entries.
    pub(crate) emissive: RangeAllocator,
    /// Entry ranges removed in a frame, freed once that frame has completed.
    pub(crate) emissive_removed: Vec<(u64, u32, u32)>,
    /// Slots of written records that take new emissive list entries at the next update:
    /// (slot, first instance entry, entries, 0).
    pub(crate) entry_jobs: Vec<[u32; 4]>,
    pub(crate) producers: Vec<Option<Producer>>,
    /// Memory of replaced or cleared producer entries, freed once that frame has completed.
    pub(crate) retired: Vec<(u64, Producer)>,
    /// Producer figures copied back each frame; made at the first update.
    pub(crate) producer_readback: Option<producer::Readback>,
    /// Index = TransformGroupId; entry 0 is the null group.
    pub(crate) groups: Vec<Option<Group>>,
    pub(crate) records: Vec<Record>,
    pub(crate) focus: Option<DVec3>,
    /// What the scene's per-frame work chooses geometry for, from the requests the last
    /// update took.
    pub(crate) pixels_per_radian: f32,
    /// The largest request since the last update.
    pub(crate) requested_detail: Option<f32>,
    pub(crate) origin_cell: IVec3,
    /// `None` until `set_environment`: a data-free type's call sets null.
    pub(crate) environment: Option<u64>,
    /// `None` until `set_medium`: a data-free type's call sets null.
    pub(crate) medium: Option<u64>,
    pub(crate) surface_data: u64,
    pub(crate) near_range: f32,
    pub(crate) ptlas: Ptlas,
    pub(crate) lights: crate::lights::LightKinds,
    /// Made at the scene's first update.
    pub(crate) cuts: Option<Cuts>,
    /// Skinned-copy entries.
    pub(crate) skinned: RangeAllocator,
    /// One past the highest skinned-copy entry handed out.
    pub(crate) skinned_high_water: u32,
    /// A group has hung from a palette's joint.
    pub(crate) hung: bool,
    skins: Vec<SkinRange>,
    /// Entry ranges and vertex arrays of removed skinned copies, freed once that frame has
    /// completed.
    pub(crate) skins_removed: Vec<(u64, u32, u32, Option<CopyArrays>)>,
    /// Made at the first update with skinned copies.
    pub(crate) skin_frame: Option<SkinFrame>,
    /// Each view history's last render of the scene, in the order the renders ran: the
    /// traversal tests what they saw, the tessellator's edge factors what last frame's first
    /// saw.
    pub(crate) seen_by: Vec<LastView>,
    /// Where each decal is listed, by `DecalId`.
    pub(crate) decals: Vec<Option<Listing>>,
    /// Set when a decal changed since the bin table was last written.
    pub(crate) decals_changed: bool,
    pub(crate) decal_bin_slots: u32,
    pub(crate) decal_globals: u32,
    /// The last frame `update_scenes` updated the scene.
    pub(crate) last_update: u64,
    /// Cooked pages placed in the scene.
    pub(crate) pages: Vec<PagePlacement>,
    /// Pages removed in a frame, freed once that frame has completed.
    pub(crate) pages_removed: Vec<(u64, PagePlacement)>,
    pub(crate) skinned_pages: Vec<SkinnedPage>,
    /// Animated pages, which a kernel poses every frame.
    pub(crate) page_motions: Vec<PageMotion>,
    /// Groups removed in a frame, whose memory that frame's records may still write.
    pub(crate) groups_removed: Vec<(u64, Group)>,
    /// Entries of skinned pages' material sets by the frames that removed them, freed once
    /// those have completed.
    pub(crate) skinned_sets_removed: Vec<(u64, (u32, u32))>,
    /// Entries of the page part of the material-set table, from `material_set_capacity`.
    pub(crate) page_sets: RangeAllocator,
}

/// A view of a scene: its history, the frame and its constants.
#[derive(Clone, Copy)]
pub(crate) struct LastView {
    pub(crate) history: ViewHistoryId,
    pub(crate) frame: u64,
    pub(crate) constants: ViewConstants,
}

pub(crate) struct Scene {
    pub(crate) desc: SceneDesc,
    pub(crate) tables: Tables,
    pub(crate) cpu: Mutex<SceneCpu>,
    /// Set while a writer for this scene is alive.
    pub(crate) writing: AtomicBool,
}

/// 32-bit words in `bytes`.
fn words(bytes: usize) -> u32 {
    (bytes / 4) as u32
}

impl SceneCpu {
    /// The lowest id no group holds; 0 means none.
    fn free_group(&self) -> usize {
        (1..self.groups.len())
            .find(|&i| self.groups[i].is_none())
            .unwrap_or(self.groups.len())
    }

    /// Puts `group` at an id of [`Self::free_group`].
    fn put_group(&mut self, id: usize, group: Group) {
        self.hung |= group.frame.is_some();
        if id == self.groups.len() {
            self.groups.push(Some(group));
        } else {
            self.groups[id] = Some(group);
        }
    }

    /// The focus and the pixels per radian of the scene's per-frame work.
    pub(crate) fn focus_detail(&self) -> Option<(DVec3, f32)> {
        self.focus.map(|f| (f, self.pixels_per_radian))
    }
}

impl Scene {
    /// Instance entry `entry` of the emissive list.
    fn emissive_entry(&self, entry: u32) -> GpuPtr<EmissiveGpu> {
        self.tables
            .emissive
            .gpu()
            .offset(u64::from(self.desc.producer_capacity + entry))
    }

    /// Empties group `id` and its record.
    fn clear_group(&self, cpu: &mut SceneCpu, frame: u64, id: TransformGroupId, seq: u64) {
        let group = cpu.groups[id.0 as usize].take();
        cpu.groups_removed.extend(group.map(|g| (frame, g)));
        let at = self.tables.groups.gpu().offset(u64::from(id.0));
        let n = words(size_of::<GroupGpu>());
        cpu.records.push(Record::fill(seq, at.address().0, n, 0));
    }

    /// Ends what the slots of `range` hold, and leaves them with their owner, who still removes
    /// them.
    fn empty_slots(&self, cpu: &mut SceneCpu, range: InstanceRange, frame: u64, seq: u64) {
        cpu.entry_jobs
            .retain(|j| !(range.first..range.first + range.count).contains(&j[0]));
        let n = range.count * words(size_of::<InstanceGpu>());
        let first = self.tables.instances.gpu().offset(u64::from(range.first));
        cpu.records.push(Record::fill(seq, first.address().0, n, 0));
        let mut entries: Vec<(u32, u32)> = Vec::new();
        if let Some(i) = cpu.pages.iter().position(|p| p.range == range) {
            let p = cpu.pages.swap_remove(i);
            self.clear_group(cpu, frame, p.group, seq);
            let n = page_entries(&p.entries);
            if n > 0 {
                entries.push((p.first_entry, n));
            }
            cpu.pages_removed.push((frame, p));
        }
        if let Some(i) = cpu.page_motions.iter().position(|m| m.range == range) {
            let m = cpu.page_motions.swap_remove(i);
            self.clear_group(cpu, frame, m.group, seq);
        }
        if let Some(i) = cpu.skinned_pages.iter().position(|p| p.range == range) {
            let p = cpu.skinned_pages.swap_remove(i);
            self.clear_group(cpu, frame, p.group, seq);
            cpu.skinned_sets_removed.push((frame, p.sets));
        }
        let end = range.first + range.count;
        let mut i = 0;
        while i < cpu.skins.len() {
            let r = &mut cpu.skins[i];
            let (a, b) = (
                range.first.max(r.slots.first),
                end.min(r.slots.first + r.slots.count),
            );
            if a >= b {
                i += 1;
                continue;
            }
            let entry = r.entry + (a - r.slots.first);
            let n = b - a;
            r.live -= n;
            let at = self.tables.skinned.gpu().offset(u64::from(entry));
            let w = n * words(size_of::<SkinnedGpu>());
            cpu.records.push(Record::fill(seq, at.address().0, w, 0));
            let arrays = if r.live == 0 {
                Some(cpu.skins.swap_remove(i).arrays)
            } else {
                i += 1;
                None
            };
            cpu.skins_removed.push((frame, entry, n, arrays));
        }
        let holds = &mut cpu.slot_holds[range.first as usize..end as usize];
        for hold in holds {
            match entries.last_mut() {
                Some((first, count)) if *first + *count == hold.first_entry => {
                    *count += hold.entries;
                }
                _ if hold.entries > 0 => entries.push((hold.first_entry, hold.entries)),
                _ => {}
            }
            *hold = SlotHold::FREE;
        }
        for (entry, count) in entries {
            let at = self.emissive_entry(entry).address().0;
            let n = count * words(size_of::<EmissiveGpu>());
            cpu.records.push(Record::fill(seq, at, n, 0));
            cpu.emissive_removed.push((frame, entry, count));
        }
    }

    /// Takes the emissive list ranges that give every copy of `model` placed by hand `stride`
    /// entries where it holds fewer, and every page a range for its prototypes' strides from
    /// `page_strides` where one is larger than the page holds, as a chain that grows the model's
    /// most emissive step asks. `page_strides` gives one per prototype, or `None` for a page
    /// the chain leaves alone. Fails without a change when the list has no room.
    pub(crate) fn reserve_entries(
        &self,
        model: u32,
        stride: u32,
        page_strides: impl Fn(PageId) -> Option<Vec<u32>>,
    ) -> Result<EntryGrowth> {
        let cpu = &mut *self.cpu.lock().unwrap();
        let slots: Vec<u32> = (0..cpu.slot_holds.len() as u32)
            .filter(|&i| {
                let h = &cpu.slot_holds[i as usize];
                h.model == model && h.entries < stride
            })
            .collect();
        let pages: Vec<(usize, Vec<PrototypeEntries>)> = cpu
            .pages
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let strides = page_strides(p.page)?;
                let grown: Vec<PrototypeEntries> = p
                    .entries
                    .iter()
                    .zip(&strides)
                    .map(|(&e, &stride)| PrototypeEntries {
                        stride: e.stride.max(stride),
                        ..e
                    })
                    .collect();
                (grown != p.entries).then_some((i, grown))
            })
            .collect();
        let mut taken: Vec<(u32, u32)> = Vec::new();
        let wanted = slots
            .iter()
            .map(|_| stride)
            .chain(pages.iter().map(|(_, e)| page_entries(e)));
        for n in wanted {
            match cpu.emissive.alloc(n) {
                Some(first) => taken.push((first, n)),
                None => {
                    for (first, n) in taken {
                        cpu.emissive.free(first, n);
                    }
                    return Err(Error::Capacity {
                        what: "emissive list entries",
                        capacity: self.desc.emissive_capacity,
                    });
                }
            }
        }
        Ok(EntryGrowth {
            model,
            stride,
            slots,
            pages,
            taken,
        })
    }

    /// Frees the ranges of a growth that is not applied.
    pub(crate) fn release_entries(&self, growth: EntryGrowth) {
        let cpu = &mut *self.cpu.lock().unwrap();
        for (first, n) in growth.taken {
            cpu.emissive.free(first, n);
        }
    }

    /// Moves the copies of a growth from [`reserve_entries`](Self::reserve_entries) to its
    /// ranges. New entries are written at the next update, a page's at the first one its
    /// records are resident for; the old ones are emptied and freed once this frame has
    /// completed.
    pub(crate) fn grow_entries(&self, growth: EntryGrowth, frame: u64, seq: &AtomicU64) {
        let cpu = &mut *self.cpu.lock().unwrap();
        let EntryGrowth {
            model,
            stride,
            slots,
            pages,
            taken,
        } = growth;
        let seq = seq.fetch_add(1, Ordering::Relaxed);
        let retire = |cpu: &mut SceneCpu, first: u32, count: u32| {
            if count == 0 {
                return;
            }
            let at = self.emissive_entry(first).address().0;
            cpu.records.push(Record::fill(
                seq,
                at,
                count * words(size_of::<EmissiveGpu>()),
                0,
            ));
            cpu.emissive_removed.push((frame, first, count));
        };
        let mut ranges = taken.into_iter();
        for slot in slots {
            let (first, _) = ranges.next().expect("one range per slot");
            let hold = cpu.slot_holds[slot as usize];
            retire(cpu, hold.first_entry, hold.entries);
            cpu.slot_holds[slot as usize] = SlotHold {
                model,
                first_entry: first,
                entries: stride,
                material_base: hold.material_base,
            };
            cpu.entry_jobs.push([slot, first, stride, 0]);
        }
        for (i, grown) in pages {
            let (first, _) = ranges.next().expect("one range per page");
            let p = &cpu.pages[i];
            let (range, old, held) = (p.range, p.first_entry, page_entries(&p.entries));
            let expanded = p.expanded;
            retire(cpu, old, held);
            // Only the records say which slot takes which entries, so the slots of a page
            // written before hold none until its records are resident again.
            if expanded {
                let slots = range.first..range.first + range.count;
                cpu.entry_jobs.extend(slots.map(|slot| [slot, 0, 0, 0]));
            }
            let p = &mut cpu.pages[i];
            p.entries_stale = expanded;
            p.first_entry = first;
            p.entries = grown;
        }
    }

    /// Empties every copy of `gone`, whose memory is about to go: placed by hand, or on a page
    /// `holds` says names one of them. Their ranges stay with their owners.
    pub(crate) fn empty_copies(
        &self,
        gone: &FxHashSet<u32>,
        holds: impl Fn(PageId) -> bool,
        frame: u64,
        seq: &AtomicU64,
    ) {
        let cpu = &mut *self.cpu.lock().unwrap();
        let pages = cpu.pages.iter().map(|p| (p.page, p.range));
        let skinned = cpu.skinned_pages.iter().map(|p| (p.page, p.range));
        let mut ranges: Vec<InstanceRange> = pages
            .chain(skinned)
            .filter(|p| holds(p.0))
            .map(|p| p.1)
            .collect();
        for &r in &ranges {
            let slots = r.first as usize..(r.first + r.count) as usize;
            cpu.slot_holds[slots]
                .iter_mut()
                .for_each(|h| h.model = NO_MODEL);
        }
        let held = |&(_, h): &(usize, &SlotHold)| gone.contains(&h.model);
        let mut run: Option<InstanceRange> = None;
        for (slot, _) in cpu.slot_holds.iter().enumerate().filter(held) {
            let slot = slot as u32;
            match &mut run {
                Some(r) if r.first + r.count == slot => r.count += 1,
                _ => {
                    ranges.extend(run.take());
                    run = Some(InstanceRange {
                        first: slot,
                        count: 1,
                    });
                }
            }
        }
        ranges.extend(run);
        for r in ranges {
            self.empty_slots(cpu, r, frame, seq.fetch_add(1, Ordering::Relaxed));
            cpu.ptlas.touch(r.first, r.count);
        }
        for i in 0..cpu.producers.len() {
            let held = cpu.producers[i].as_ref().and_then(|p| p.model);
            if held.is_some_and(|m| gone.contains(&m.0)) {
                self.end_producer(cpu, i, frame, seq.fetch_add(1, Ordering::Relaxed));
            }
        }
    }

    /// Clears producer entry `i` as `SceneWriter::set_producer` with nothing does, between frames.
    fn end_producer(&self, cpu: &mut SceneCpu, i: usize, frame: u64, seq: u64) {
        let t = &self.tables;
        let at = i as u64;
        let state = t.producer_state.gpu().offset(at).address().0;
        let blas = std::mem::offset_of!(ProducerStateGpu, blas) as u64;
        let clusters = std::mem::offset_of!(ProducerStateGpu, clusters) as u64;
        let zeroed = [
            (
                t.producers.gpu().offset(at).address().0,
                words(size_of::<ProducerGpu>()),
            ),
            (
                t.batches.gpu().offset(at).address().0,
                words(size_of::<ProducerBatch>()),
            ),
            (
                t.patches.gpu().offset(at).address().0,
                words(size_of::<ProducerBatch>()),
            ),
            (
                t.tess_state.gpu().offset(at).address().0,
                words(size_of::<TessStateGpu>()),
            ),
            (
                t.emissive.gpu().offset(at).address().0,
                words(size_of::<EmissiveGpu>()),
            ),
            (t.producer_counts.gpu().offset(at).address().0, 4),
            (state + blas, 2),
            (state + clusters, 1),
        ];
        for (dst, n) in zeroed {
            cpu.records.push(Record::fill(seq, dst, n, 0));
        }
        if let Some(old) = cpu.producers[i].take() {
            cpu.retired.push((frame, old));
        }
    }

    pub(crate) fn new(device: &Device, desc: &SceneDesc) -> Result<Self> {
        // A released part run packs its first part instance into 24 bits (ptlas.slang).
        assert!(
            desc.part_capacity <= 1 << 24,
            "part_capacity {} is past 2^24",
            desc.part_capacity
        );
        let n = |c: u32| u64::from(c.max(1));
        let dl = MemoryKind::DeviceLocal;
        let (decal_slots, decal_refs) = table_sizes(desc.decal_capacity);
        let tables = Tables {
            root: device.gpu_malloc(dl, 1, "scene root")?,
            instances: device.gpu_malloc(dl, n(desc.instance_capacity), "instances")?,
            slot_state: device.gpu_malloc(dl, n(desc.instance_capacity), "slot state")?,
            groups: device.gpu_malloc(dl, n(desc.group_capacity), "transform groups")?,
            material_sets: device.gpu_malloc(
                dl,
                n(desc.material_set_capacity + desc.page_material_sets),
                "material sets",
            )?,
            lights: device.gpu_malloc(dl, n(desc.light_capacity), "lights")?,
            emissive: device.gpu_malloc(
                dl,
                n(desc.producer_capacity + desc.emissive_capacity),
                "emissive list",
            )?,
            emissive_lists: device.gpu_malloc(
                dl,
                n(desc.emissive_capacity * 2 * GATHERED_CLUSTERS),
                "gathered emissive lists",
            )?,
            producers: device.gpu_malloc(dl, n(desc.producer_capacity), "producers")?,
            batches: device.gpu_malloc(dl, n(desc.producer_capacity), "producer batches")?,
            patches: device.gpu_malloc(dl, n(desc.producer_capacity), "producer patches")?,
            tess_state: device.gpu_malloc(dl, n(desc.producer_capacity), "tessellation state")?,
            producer_counts: device.gpu_malloc(dl, n(desc.producer_capacity), "producer counts")?,
            producer_state: device.gpu_malloc(dl, n(desc.producer_capacity), "producer state")?,
            sky_stats: device.gpu_malloc(dl, 1, "sky stats")?,
            skinned: device.gpu_malloc(dl, n(desc.skinned_capacity), "skinned copies")?,
            skin_state: device.gpu_malloc(dl, n(desc.skinned_capacity), "skin state")?,
            decals: device.gpu_malloc(dl, n(desc.decal_capacity), "decals")?,
            decal_bins: device.gpu_malloc(dl, n(decal_slots), "decal bins")?,
            decal_refs: device.gpu_malloc(dl, n(decal_refs), "decal runs")?,
            proxies: device.gpu_malloc(dl, n(desc.medium_proxy_capacity), "medium proxies")?,
        };
        // Tables start zeroed: empty slots, the null group, no lights. These apply before
        // any write (sequence 0).
        let zero = |a: GpuAddress, bytes: u64| Record::fill(0, a.0, words(bytes as usize), 0);
        let records = vec![
            zero(tables.instances.gpu().address(), tables.instances.size()),
            zero(tables.slot_state.gpu().address(), tables.slot_state.size()),
            zero(tables.groups.gpu().address(), tables.groups.size()),
            zero(
                tables.material_sets.gpu().address(),
                tables.material_sets.size(),
            ),
            zero(tables.lights.gpu().address(), tables.lights.size()),
            zero(tables.emissive.gpu().address(), tables.emissive.size()),
            zero(tables.producers.gpu().address(), tables.producers.size()),
            zero(tables.batches.gpu().address(), tables.batches.size()),
            zero(tables.patches.gpu().address(), tables.patches.size()),
            zero(tables.tess_state.gpu().address(), tables.tess_state.size()),
            zero(
                tables.producer_counts.gpu().address(),
                tables.producer_counts.size(),
            ),
            zero(
                tables.producer_state.gpu().address(),
                tables.producer_state.size(),
            ),
            zero(tables.sky_stats.gpu().address(), tables.sky_stats.size()),
            zero(tables.skinned.gpu().address(), tables.skinned.size()),
            zero(tables.skin_state.gpu().address(), tables.skin_state.size()),
            zero(tables.decals.gpu().address(), tables.decals.size()),
            zero(tables.proxies.gpu().address(), tables.proxies.size()),
        ];
        Ok(Self {
            desc: *desc,
            tables,
            cpu: Mutex::new(SceneCpu {
                slots: RangeAllocator::new(desc.instance_capacity),
                slot_holds: vec![SlotHold::FREE; desc.instance_capacity as usize],
                masks: HashSet::new(),
                removed: Vec::new(),
                emissive: RangeAllocator::new(desc.emissive_capacity),
                emissive_removed: Vec::new(),
                entry_jobs: Vec::new(),
                producers: (0..desc.producer_capacity).map(|_| None).collect(),
                retired: Vec::new(),
                producer_readback: None,
                groups: vec![None],
                records,
                focus: None,
                pixels_per_radian: DEFAULT_PIXELS_PER_RADIAN,
                requested_detail: None,
                origin_cell: IVec3::ZERO,
                environment: None,
                medium: None,
                surface_data: 0,
                near_range: 256.0,
                ptlas: Ptlas::new(device, desc)?,
                lights: crate::lights::LightKinds::default(),
                cuts: None,
                skinned: RangeAllocator::new(desc.skinned_capacity),
                skinned_high_water: 0,
                hung: false,
                skins: Vec::new(),
                skins_removed: Vec::new(),
                skin_frame: None,
                seen_by: Vec::new(),
                decals: vec![None; desc.decal_capacity as usize],
                decals_changed: false,
                decal_bin_slots: 0,
                decal_globals: 0,
                last_update: 0,
                pages: Vec::new(),
                pages_removed: Vec::new(),
                skinned_pages: Vec::new(),
                page_motions: Vec::new(),
                groups_removed: Vec::new(),
                skinned_sets_removed: Vec::new(),
                page_sets: RangeAllocator::new(desc.page_material_sets),
            }),
            writing: AtomicBool::new(false),
        })
    }
}

/// A record a linked environment reads; `TYPE` is the Slang type that reads it.
pub trait EnvironmentData: 'static {
    const TYPE: &'static str;
    /// `TYPE` is safe on a null pointer, as it reads no data or checks for null, so
    /// `SceneWriter::set_environment` takes one.
    const NULL_SAFE: bool = false;
}

/// A record a linked medium reads; `TYPE` is the Slang type that reads it.
pub trait MediumData: 'static {
    const TYPE: &'static str;
    /// `TYPE` is safe on a null pointer, as it reads no data or checks for null, so
    /// `SceneWriter::set_medium` takes one.
    const NULL_SAFE: bool = false;
}

/// Writes one scene's changes into this frame's upload window. Returned slices are
/// write-only (write-combined memory) and live until `update_scenes`. Send: different
/// scenes' writers, and disjoint pieces of one returned slice, can be filled on different
/// threads.
pub struct SceneWriter<'f> {
    pub(crate) scene: &'f Scene,
    pub(crate) device: &'f Device,
    pub(crate) kernels: &'static KernelSet,
    pub(crate) window: &'f UploadWindow,
    pub(crate) seq: &'f AtomicU64,
    pub(crate) frame: u64,
    pub(crate) cell_log2: u32,
    /// Sections and animated parts by `ModelId`.
    pub(crate) model_cpu: &'f [ModelCpu],
    /// Subdivision cages by `ModelId`.
    pub(crate) cages: &'f [Option<ModelCage>],
    pub(crate) patterns: &'f Patterns,
    pub(crate) mounts: &'f Mounts,
    pub(crate) materials: &'f MaterialTable,
}

impl Drop for SceneWriter<'_> {
    fn drop(&mut self) {
        self.scene.writing.store(false, Ordering::Release);
    }
}

impl<'f> SceneWriter<'f> {
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

    /// `value` in the window and its copy to `dst`, for the caller to push once nothing it
    /// does after can fail.
    fn staged<T: Pod>(&self, dst: GpuPtr<T>, value: &T) -> Result<Record> {
        let src = self.window.put(self.device, value)?;
        Ok(Record::values(self.next_seq(), dst, src, 1))
    }

    /// Fills window room `out` at `src` with the emissive list entries of the copies from slot
    /// `first`, `stride` each, recorded as a copy to instance entry `entry`.
    fn put_entries(
        &self,
        (src, out): (GpuPtr<EmissiveGpu>, &mut [EmissiveGpu]),
        first: u32,
        entry: u32,
        stride: u32,
    ) {
        for (e, i) in out.iter_mut().zip(0..) {
            *e = EmissiveGpu {
                kind: EMISSIVE_INSTANCE,
                source: first + i / stride,
                rank: i % stride,
                set: NO_SET,
                ..Zeroable::zeroed()
            };
        }
        self.push(Record::values(
            self.next_seq(),
            self.scene.emissive_entry(entry),
            src,
            out.len() as u32,
        ));
    }

    /// A new transform group; its transforms start all-zero (hidden).
    pub fn create_transform_group(
        &mut self,
        desc: &TransformGroupDesc,
    ) -> Result<TransformGroupId> {
        self.create_group(desc, None)
    }

    fn create_group(
        &self,
        desc: &TransformGroupDesc,
        frame: Option<GroupFrame>,
    ) -> Result<TransformGroupId> {
        let bytes = u64::from(desc.count.max(1)) * size_of::<AffineRows>() as u64;
        let buffers = [0, 1].map(|_| {
            self.device
                .gpu_malloc::<u8>(MemoryKind::DeviceLocal, bytes, "transforms")
        });
        let [a, b] = buffers;
        let group = Group {
            writer: desc.writer,
            count: desc.count,
            anchor: WorldPos::new(desc.anchor, self.cell_log2),
            buffers: vec![a?, b?],
            current: 0,
            previous_is_current: true,
            created_frame: self.frame,
            written: false,
            page: None,
            frame,
        };
        let record = self.window.put(self.device, &group.record())?;
        let zero = Record::fill(
            self.next_seq(),
            group.buffers[0].gpu().address().0,
            words(bytes as usize),
            0,
        );
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        let id = cpu.free_group();
        if id as u32 >= self.scene.desc.group_capacity {
            return Err(Error::Capacity {
                what: "transform groups",
                capacity: self.scene.desc.group_capacity,
            });
        }
        cpu.put_group(id, group);
        cpu.records.push(zero);
        cpu.records.push(Record::values(
            self.next_seq(),
            self.scene.tables.groups.gpu().offset(id as u64),
            record,
            1,
        ));
        Ok(TransformGroupId(id as u32))
    }

    /// Transforms `first..first + count` of a CPU-written group, relative to its anchor.
    pub fn transforms(
        &mut self,
        group: TransformGroupId,
        first: u32,
        count: u32,
    ) -> Result<&'f mut [AffineRows]> {
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
    ) -> Result<&'f mut [AffineRows]> {
        self.write_transforms(group, first, count, true)
    }

    fn write_transforms(
        &self,
        group: TransformGroupId,
        first: u32,
        count: u32,
        without_motion: bool,
    ) -> Result<&'f mut [AffineRows]> {
        let (src, out) = self.window.slice::<AffineRows>(self.device, count)?;
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
            let at = |b: &Allocation<u8>| b.gpu().cast::<AffineRows>().offset(u64::from(first));
            (at(&g.buffers[g.current]), at(&g.buffers[1 - g.current]))
        };
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
            .gpu()
            .offset(u64::from(slot))
            .address()
            .0
            + byte as u64
    }

    /// `desc.copies` new instance slots, and an emissive list entry per copy for each
    /// emissive section of the model. With a pose, each copy's PTLAS partition keeps room for
    /// its part run. Each cluster takes the class its section cooked with (a `material_set`
    /// does not change it): dielectric and presence sections are traced by rays that look for
    /// them and never rasterized into the visibility buffer, and one model may mix them with
    /// opaque sections. Fails with [`Error::NotResident`] for a model no ready mount or live
    /// edit holds.
    ///
    /// Panics if `desc.material_set`'s set 0 ends past `material_set_capacity`, or if
    /// `desc.transforms` or the pose's group is neither the null group nor one from
    /// `create_transform_group`.
    pub fn add_instances(&mut self, desc: &InstancesDesc) -> Result<InstanceRange> {
        self.check_groups(std::iter::once(desc.transforms).chain(desc.pose.map(|p| p.group)));
        self.placeable(desc.model)?;
        if let Some(base) = desc.material_set {
            self.check_material_set(desc.model, base, 0);
        }
        let model = self.model_cpu[desc.model.0 as usize];
        let ranks = model.emissive_stride;
        // Window room comes first, so nothing fails once the copies hold slots.
        let (partitions_src, partitions_out) =
            self.window.slice::<u32>(self.device, desc.copies)?;
        let entries = (ranks > 0)
            .then(|| {
                self.window
                    .slice::<EmissiveGpu>(self.device, desc.copies * ranks)
            })
            .transpose()?;
        let (records_src, records) = self.window.slice::<InstanceGpu>(self.device, desc.copies)?;
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
            let entry = if ranks > 0 {
                cpu.emissive.alloc(desc.copies * ranks)
            } else {
                Some(0)
            };
            let Some(entry) = entry else {
                cpu.slots.free(first, desc.copies);
                return Err(Error::Capacity {
                    what: "emissive list entries",
                    capacity: self.scene.desc.emissive_capacity,
                });
            };
            let cost = 1 + if desc.pose.is_some() {
                model.animated_parts
            } else {
                0
            };
            let holds = &mut cpu.slot_holds[first as usize..(first + desc.copies) as usize];
            for (hold, copy) in holds.iter_mut().zip(0..) {
                *hold = SlotHold {
                    model: desc.model.0,
                    first_entry: entry + copy * ranks,
                    entries: ranks,
                    material_base: desc.material_set.unwrap_or(NO_MATERIAL_SET),
                };
            }
            let partitions = cpu.ptlas.place(first, desc.copies, cell, cost);
            (first, entry, partitions, cpu.ptlas.slot_table.gpu())
        };
        partitions_out.copy_from_slice(&partitions);
        self.push(Record::values(
            self.next_seq(),
            table.offset(u64::from(first)),
            partitions_src,
            desc.copies,
        ));
        if let Some(room) = entries {
            self.put_entries(room, first, entry, ranks);
        }
        let first_entry = self.scene.desc.producer_capacity + entry;
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
        self.push(Record::values(
            self.next_seq(),
            self.scene.tables.instances.gpu().offset(u64::from(first)),
            records_src,
            desc.copies,
        ));
        for (copy, r) in records.iter_mut().enumerate() {
            *r = InstanceGpu {
                first_emissive: if ranks > 0 {
                    first_entry + copy as u32 * ranks
                } else {
                    NO_EMISSIVE
                },
                emissive_count: ranks,
                group: desc.transforms.0,
                group_index: desc.first + copy as u32,
                model: desc.model.0,
                material_base: desc.material_set.unwrap_or(NO_MATERIAL_SET),
                pose_group: pose.group.0,
                pose_index: pose.index,
                first_part: u32::MAX,
                rt_range: half::f16::from_f32(desc.rt_range).to_bits(),
                animation_range: half::f16::from_f32(desc.animation_range).to_bits(),
                flags: desc.flags.0 & InstanceFlags::CONSUMER,
                _pad: 0,
                pose_variants: pose.variants as u16,
                pose_stride: pose.stride as u16,
                debug_lod: 0,
                user_tag: desc.user_tag,
                variant: 0,
            };
        }
        self.reset_slot_state(range);
        Ok(range)
    }

    /// One slot per record of a cooked instance page, partitioned as the cook laid them out,
    /// in a transform group of its own over the page's records. The page's records load on
    /// the loading side: the scene wants them every frame, nearest the focus first, within
    /// the instance page budget, and its copies are hidden while they are not resident. The
    /// range stays valid as the page leaves and returns. Its slots take their records from the
    /// page's prototypes the first time the records land, so their variants, flags and other
    /// per-copy settings come from the page.
    ///
    /// Panics if `r.material_sets` names a prototype the page lacks or one prototype twice, if
    /// set 0 of a base there ends past `material_set_capacity` on its prototype's model, or if
    /// a group in `r.poses` is neither the null group nor one from `create_transform_group`.
    pub fn add_instance_page(&mut self, r: &InstancePageRef<'_>) -> Result<InstanceRange> {
        self.check_groups(r.poses.iter().map(|(_, pose)| pose.group));
        let page = page_needs(self.mounts, self.model_cpu, r.page)?;
        let protos = &page.2;
        let mut named = vec![None; protos.len()];
        for &(index, base) in r.material_sets {
            let i = usize::from(index);
            assert!(
                i < protos.len(),
                "material set for prototype {index} of a page of {} prototypes",
                protos.len()
            );
            assert!(
                named[i].is_none(),
                "two material sets for prototype {index}"
            );
            self.check_material_set(protos[i].model, base, 0);
            named[i] = Some(base);
        }
        let motion = self.motion_group(r.page, r.frame)?;
        let placed = self.place_instance_page(r, page, &named, motion);
        self.settle_motion(r.page, motion, placed)
    }

    fn place_instance_page(
        &self,
        r: &InstancePageRef<'_>,
        (needs, entries, protos): (PageNeeds, Vec<PrototypeEntries>, Vec<PagePrototype>),
        named: &[Option<u32>],
        motion: Option<TransformGroupId>,
    ) -> Result<(InstanceRange, TransformGroupId)> {
        let page = self.mounts.page(r.page).ok_or(Error::UnknownPage(r.page))?;
        if let Some(p) = protos.iter().find(|p| self.is_skinned(p.model)) {
            return Err(Error::SkinnedPage(p.model));
        }
        let header = page.info.header;
        let exact_rows = header.exact_rows(page.bytes as usize)? as u32;
        let PageNeeds {
            instances: count,
            emissive,
            ..
        } = needs;
        let set_entries = protos
            .iter()
            .zip(named)
            .filter(|(_, base)| base.is_none())
            .map(|(p, _)| p.sets.as_ref().map_or(0, |s| s.len() as u32))
            .sum();
        // Window room comes first, so nothing fails once the page holds slots.
        let (sets_src, sets_out) = self.window.slice::<u32>(self.device, set_entries)?;
        let (slots_src, slots_out) = self.window.slice::<u32>(self.device, count)?;
        let (group_src, group_out) = self.window.slice::<GroupGpu>(self.device, 1)?;
        let copy_bytes = (protos.len() * size_of::<PagePrototypeGpu>()) as u64;
        let mut table = self.device.gpu_malloc::<u8>(
            MemoryKind::Upload,
            copy_bytes.max(16),
            "page prototypes",
        )?;

        let cpu = &mut *self.scene.cpu.lock().unwrap();
        let cell = IVec3::from_array(header.cell);
        let frame = r
            .frame
            .map(|f| self.group_frame(cpu, f, cell))
            .transpose()?;
        let group_id = cpu.free_group() as u32;
        if group_id >= self.scene.desc.group_capacity {
            return Err(Error::Capacity {
                what: "transform groups",
                capacity: self.scene.desc.group_capacity,
            });
        }
        let first = cpu.slots.alloc(count).ok_or(Error::Capacity {
            what: "instances",
            capacity: self.scene.desc.instance_capacity,
        })?;
        let Some(set_base) = cpu.page_sets.alloc(set_entries) else {
            cpu.slots.free(first, count);
            return Err(Error::Capacity {
                what: "page material sets",
                capacity: self.scene.desc.page_material_sets,
            });
        };
        let entry = if emissive > 0 {
            cpu.emissive.alloc(emissive)
        } else {
            Some(0)
        };
        let Some(entry) = entry else {
            cpu.slots.free(first, count);
            cpu.page_sets.free(set_base, set_entries);
            return Err(Error::Capacity {
                what: "emissive list entries",
                capacity: self.scene.desc.emissive_capacity,
            });
        };
        if emissive > 0 {
            let at = self.scene.emissive_entry(entry).address().0;
            let n = emissive * words(size_of::<EmissiveGpu>());
            cpu.records.push(Record::fill(self.next_seq(), at, n, 0));
        }
        cpu.slot_holds[first as usize..(first + count) as usize].fill(SlotHold::FREE);

        let mut copies = Vec::with_capacity(protos.len());
        let first_set = self.scene.desc.material_set_capacity + set_base;
        let mut at = 0;
        for (p, &named) in protos.iter().zip(named) {
            let base = match (named, &p.sets) {
                (Some(base), _) => base,
                (None, Some(set)) => {
                    for (o, m) in sets_out[at..at + set.len()].iter_mut().zip(set) {
                        *o = m.0;
                    }
                    at += set.len();
                    first_set + (at - set.len()) as u32
                }
                (None, None) => NO_MATERIAL_SET,
            };
            let pose = r.poses.iter().find(|(m, _)| *m == p.model).map(|&(_, q)| q);
            let flags = if p.flags == 0 {
                r.flags.0
            } else {
                p.flags as u8 & r.flags.0
            };
            if p.mask != 0 {
                cpu.masks.insert((p.model, p.mask));
            }
            copies.push(PagePrototypeGpu {
                instance: InstanceGpu {
                    group: motion.map_or(group_id, |m| m.0),
                    group_index: 0,
                    model: p.model.0,
                    material_base: base,
                    pose_group: pose.map_or(0, |q| q.group.0),
                    pose_index: pose.map_or(0, |q| q.index),
                    first_part: u32::MAX,
                    rt_range: half::f16::from_f32(r.rt_range).to_bits(),
                    animation_range: half::f16::from_f32(r.animation_range).to_bits(),
                    flags: flags & InstanceFlags::CONSUMER,
                    _pad: 0,
                    pose_variants: pose.map_or(0, |q| q.variants as u16),
                    pose_stride: pose.map_or(0, |q| q.stride as u16),
                    debug_lod: 0,
                    user_tag: r.user_tag,
                    variant: u32::from(p.pose) | u32::from(p.mask) << 16,
                    first_emissive: NO_EMISSIVE,
                    emissive_count: 0,
                },
                records: p.records,
                _pad: [0; 3],
            });
        }
        if set_entries > 0 {
            cpu.records.push(Record::values(
                self.next_seq(),
                self.scene
                    .tables
                    .material_sets
                    .gpu()
                    .offset(u64::from(first_set))
                    .cast::<u32>(),
                sets_src,
                set_entries,
            ));
        }
        let out = table.as_mut_slice().expect("upload is mapped");
        out[..copy_bytes as usize].copy_from_slice(bytemuck::cast_slice(&copies));
        let mut ends: Vec<u32> = page.info.partitions.iter().map(|&p| u32::from(p)).collect();
        ends.push(count);
        let mut runs: Vec<u32> = ends
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|&n| n > 0)
            .collect();
        if runs.is_empty() && count > 0 {
            runs.push(count);
        }
        let partitions = cpu.ptlas.place_runs(first, &runs, cell);
        let mut group = Group::of_page(r.page, page, self.frame);
        group.frame = frame;
        let record = group.record();
        cpu.put_group(group_id as usize, group);
        let range = InstanceRange { first, count };
        let seq = self.next_seq();
        let n = count * words(size_of::<InstanceGpu>());
        cpu.records
            .push(Record::fill(seq, self.instance_word(first, 0), n, 0));
        let slot_table = cpu.ptlas.slot_table.gpu();
        cpu.pages.push(PagePlacement {
            range,
            group: TransformGroupId(group_id),
            page: r.page,
            prototypes: table,
            prototype_count: copies.len() as u32,
            first_entry: entry,
            exact_rows,
            entries,
            expanded: false,
            entries_stale: false,
            sets: (set_base, set_entries),
        });
        slots_out.copy_from_slice(&partitions);
        cpu.records.push(Record::values(
            self.next_seq(),
            slot_table.offset(u64::from(first)),
            slots_src,
            count,
        ));
        group_out[0] = record;
        cpu.records.push(Record::values(
            self.next_seq(),
            self.scene.tables.groups.gpu().offset(u64::from(group_id)),
            group_src,
            1,
        ));
        Self::reset_slot_state_with(&self.scene.tables, self.next_seq(), range, &mut cpu.records);
        Ok((range, TransformGroupId(group_id)))
    }

    /// What hangs a group anchored at `cell` from entry `frame.1` of palette group `frame.0`. A
    /// page's group holds records, not the matrices a palette entry is read as.
    fn group_frame(
        &self,
        cpu: &SceneCpu,
        (palette, index): (TransformGroupId, u32),
        cell: IVec3,
    ) -> Result<GroupFrame> {
        let anchor = cpu
            .groups
            .get(palette.0 as usize)
            .and_then(Option::as_ref)
            .filter(|g| g.page.is_none() && index < g.count)
            .ok_or(Error::NoFrameJoint(palette, index))?
            .anchor;
        let edge = 2f64.powi(self.cell_log2 as i32);
        let cells = (cell - anchor.cell).as_dvec3() * edge - anchor.offset.as_dvec3();
        Ok(GroupFrame {
            palette: self.scene.tables.groups.gpu().offset(u64::from(palette.0)),
            index,
            origin: cells.as_vec3().to_array(),
        })
    }

    /// The group an animated page's copies read, hung as the page is; `None` for a page that
    /// holds still. A kernel writes it every frame, from [`Self::set_page_time`].
    fn motion_group(
        &self,
        page: PageId,
        frame: Option<(TransformGroupId, u32)>,
    ) -> Result<Option<TransformGroupId>> {
        let held = self.mounts.page(page).ok_or(Error::UnknownPage(page))?;
        if held.info.motion_times.is_empty() {
            return Ok(None);
        }
        let header = held.info.header;
        let cell = IVec3::from_array(header.cell);
        let edge = 2f64.powi(self.cell_log2 as i32);
        let hung = {
            let cpu = self.scene.cpu.lock().unwrap();
            frame.map(|f| self.group_frame(&cpu, f, cell)).transpose()?
        };
        let desc = TransformGroupDesc {
            count: u32::from(header.record_count),
            writer: GroupWriter::Gpu,
            anchor: cell.as_dvec3() * edge,
        };
        self.create_group(&desc, hung).map(Some)
    }

    /// Lists the animated page just placed, or frees its group if that failed. `placed` is the
    /// page's range and the group of its records.
    fn settle_motion(
        &self,
        page: PageId,
        motion: Option<TransformGroupId>,
        placed: Result<(InstanceRange, TransformGroupId)>,
    ) -> Result<InstanceRange> {
        let Some(group) = motion else {
            return placed.map(|p| p.0);
        };
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        let (range, records) = match placed {
            Ok(p) => p,
            Err(e) => {
                self.scene
                    .clear_group(cpu, self.frame, group, self.next_seq());
                return Err(e);
            }
        };
        let p = self.mounts.page(page).ok_or(Error::UnknownPage(page))?;
        let h = p.info.header;
        cpu.page_motions.push(PageMotion {
            range,
            page,
            records,
            group,
            times: p.info.motion_times.clone(),
            left: h.motion_left.rows,
            right: h.motion_right.rows,
            count: u32::from(h.record_count),
            time: h.motion_start,
            forward: true,
            held: None,
        });
        Ok(range)
    }

    /// Sets the clock of the animated page placed at `range`, in seconds, on the timeline of its
    /// skeleton clips. Copies of a page that holds still ignore it. A clock that wraps reads as a
    /// seek: the page can't know it loops, so its copies hold until the rows of the new time land.
    pub fn set_page_time(&mut self, range: InstanceRange, seconds: f32) {
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        if let Some(m) = cpu.page_motions.iter_mut().find(|m| m.range == range) {
            if seconds != m.time {
                m.forward = seconds > m.time;
            }
            m.time = seconds;
        }
    }

    /// Whether `model` has a skinned mesh.
    fn is_skinned(&self, model: ModelId) -> bool {
        let cpu = self.model_cpu.get(model.0 as usize);
        cpu.is_some_and(|m| m.skin.is_some())
    }

    /// The copies of a page of one skinned model, one instance slot each, posed like
    /// [`Self::add_skinned`]'s from the page's records, and hidden while those are not resident.
    /// They take the page's flags, section mask and materials as [`Self::add_instance_page`]'s
    /// do, unless `r` names a material set. A page of anything else fails with
    /// [`Error::NotSkinnedPage`].
    ///
    /// Panics if set 0 of `r.material_set` ends past `material_set_capacity` on the page's model,
    /// or if `r.palette` is neither the null group nor one from `create_transform_group`.
    pub fn add_skinned_page(&mut self, r: &SkinnedPageRef) -> Result<InstanceRange> {
        self.check_groups([r.palette]);
        let protos = self.mounts.page_prototypes(r.page)?;
        let [proto] = protos.as_slice() else {
            return Err(Error::NotSkinnedPage(r.page));
        };
        if let Some(base) = r.material_set {
            self.check_material_set(proto.model, base, 0);
        }
        let motion = self.motion_group(r.page, r.frame)?;
        let placed = self.place_skinned_page(r, proto, motion);
        self.settle_motion(r.page, motion, placed)
    }

    fn place_skinned_page(
        &mut self,
        r: &SkinnedPageRef,
        proto: &PagePrototype,
        motion: Option<TransformGroupId>,
    ) -> Result<(InstanceRange, TransformGroupId)> {
        let page = self.mounts.page(r.page).ok_or(Error::UnknownPage(r.page))?;
        let posable = |m: &ModelCpu| m.skin.is_some() && m.joints > 0;
        if !self
            .model_cpu
            .get(proto.model.0 as usize)
            .is_some_and(posable)
        {
            return Err(Error::NotSkinnedPage(r.page));
        }
        let mut group = Group::of_page(r.page, page, self.frame);
        let cell = IVec3::from_array(page.info.header.cell);
        group.frame = {
            let cpu = self.scene.cpu.lock().unwrap();
            r.frame
                .map(|f| self.group_frame(&cpu, f, cell))
                .transpose()?
        };
        let copies = group.count;
        let record = group.record();
        let cooked = proto.sets.as_ref().filter(|_| r.material_set.is_none());
        let set_entries = cooked.map_or(0, |s| s.len() as u32);
        let (sets_src, sets_out) = self.window.slice::<u32>(self.device, set_entries)?;
        let (group_src, group_out) = self.window.slice::<GroupGpu>(self.device, 1)?;
        let (id, sets) = {
            let cpu = &mut *self.scene.cpu.lock().unwrap();
            let id = cpu.free_group();
            if id as u32 >= self.scene.desc.group_capacity {
                return Err(Error::Capacity {
                    what: "transform groups",
                    capacity: self.scene.desc.group_capacity,
                });
            }
            let Some(set_base) = cpu.page_sets.alloc(set_entries) else {
                return Err(Error::Capacity {
                    what: "page material sets",
                    capacity: self.scene.desc.page_material_sets,
                });
            };
            if let Some(set) = cooked {
                for (o, m) in sets_out.iter_mut().zip(set) {
                    *o = m.0;
                }
                let at = self.scene.desc.material_set_capacity + set_base;
                let table = self.scene.tables.material_sets.gpu();
                cpu.records.push(Record::values(
                    self.next_seq(),
                    table.offset(u64::from(at)).cast::<u32>(),
                    sets_src,
                    set_entries,
                ));
            }
            cpu.put_group(id, group);
            group_out[0] = record;
            cpu.records.push(Record::values(
                self.next_seq(),
                self.scene.tables.groups.gpu().offset(id as u64),
                group_src,
                1,
            ));
            (TransformGroupId(id as u32), (set_base, set_entries))
        };
        let cooked_set = cooked.map(|_| self.scene.desc.material_set_capacity + sets.0);
        let flags = match proto.flags {
            0 => r.flags,
            f => InstanceFlags(f as u8 & r.flags.0),
        };
        let desc = SkinnedDesc {
            model: proto.model,
            copies,
            transforms: motion.unwrap_or(id),
            first: 0,
            palette: r.palette,
            palette_first: r.palette_first,
            material_set: r.material_set.or(cooked_set),
            flags,
            rt_range: r.rt_range,
            user_tag: r.user_tag,
        };
        let placed = self.place_skinned(&desc, proto.mask);
        let range = match placed {
            Ok(range) => range,
            Err(e) => {
                let seq = self.next_seq();
                let cpu = &mut *self.scene.cpu.lock().unwrap();
                self.scene.clear_group(cpu, self.frame, id, seq);
                cpu.skinned_sets_removed.push((self.frame, sets));
                return Err(e);
            }
        };
        let placement = SkinnedPage {
            range,
            group: id,
            page: r.page,
            sets,
        };
        self.scene.cpu.lock().unwrap().skinned_pages.push(placement);
        Ok((range, id))
    }

    /// Zeroes the slot state of new slots. The fields from `part_run` on describe runs the
    /// PTLAS kernels still release.
    fn reset_slot_state(&self, range: InstanceRange) {
        let mut records = Vec::new();
        Self::reset_slot_state_with(&self.scene.tables, self.next_seq(), range, &mut records);
        self.scene.cpu.lock().unwrap().records.append(&mut records);
    }

    fn reset_slot_state_with(
        tables: &Tables,
        seq: u64,
        range: InstanceRange,
        records: &mut Vec<Record>,
    ) {
        let state = tables
            .slot_state
            .gpu()
            .offset(u64::from(range.first))
            .address()
            .0;
        for word in 0..words(std::mem::offset_of!(SlotStateGpu, part_run)) {
            records.push(Record {
                seq,
                dst: state + 4 * u64::from(word),
                count: range.count,
                stride: words(size_of::<SlotStateGpu>()),
                op: Op::Fill(0),
            });
        }
    }

    /// `desc.copies` skinned copies of a model with a skinned mesh, one instance slot each.
    /// Every frame the scene is updated each copy takes one cut by its distance from the
    /// focus, skins it with its palette entries into its own vertex arrays, rasters it and,
    /// inside `rt_range`, traces it through a cluster BLAS built from the cut's cluster
    /// templates, in the PTLAS global partition. Copies skin whether or not a view shows
    /// them, and take the candidate class of the model's own materials as in
    /// [`Self::add_instances`]. Fails with [`Error::NotResident`] as that does. Panics if the
    /// model's mesh has no skin or the model no skeleton, if `desc.material_set`'s set 0 ends
    /// past `material_set_capacity`, or if `desc.transforms` or `desc.palette` is neither the
    /// null group nor one from `create_transform_group`.
    pub fn add_skinned(&mut self, desc: &SkinnedDesc) -> Result<InstanceRange> {
        self.check_groups([desc.transforms, desc.palette]);
        self.placeable(desc.model)?;
        if let Some(base) = desc.material_set {
            self.check_material_set(desc.model, base, 0);
        }
        self.place_skinned(desc, 0)
    }

    /// [`Self::add_skinned`] for a base that may lie in the page part of the material-set table,
    /// with every copy hiding the sections of `section_mask`.
    fn place_skinned(&mut self, desc: &SkinnedDesc, section_mask: u16) -> Result<InstanceRange> {
        let model = self.model_cpu[desc.model.0 as usize];
        let shape = model
            .skin
            .unwrap_or_else(|| panic!("{:?} has no skinned mesh", desc.model));
        assert!(model.joints > 0, "{:?} has no skeleton", desc.model);
        let arrays = CopyArrays::new(self.device, desc.copies, shape.vertices, shape.clusters)?;
        let ranks = model.emissive_stride;
        // Window room comes first, so nothing fails once the copies hold slots.
        let entries = (ranks > 0)
            .then(|| {
                self.window
                    .slice::<EmissiveGpu>(self.device, desc.copies * ranks)
            })
            .transpose()?;
        let (records_src, records) = self.window.slice::<InstanceGpu>(self.device, desc.copies)?;
        let (skinned_src, skinned) = self.window.slice::<SkinnedGpu>(self.device, desc.copies)?;
        let (states_src, states) = self
            .window
            .slice::<SkinStateGpu>(self.device, desc.copies)?;
        let (first, entry, emissive) = {
            let cpu = &mut *self.scene.cpu.lock().unwrap();
            let first = cpu.slots.alloc(desc.copies).ok_or(Error::Capacity {
                what: "instances",
                capacity: self.scene.desc.instance_capacity,
            })?;
            let Some(entry) = cpu.skinned.alloc(desc.copies) else {
                cpu.slots.free(first, desc.copies);
                return Err(Error::Capacity {
                    what: "skinned copies",
                    capacity: self.scene.desc.skinned_capacity,
                });
            };
            let emissive = if ranks > 0 {
                cpu.emissive.alloc(desc.copies * ranks)
            } else {
                Some(0)
            };
            let Some(emissive) = emissive else {
                cpu.slots.free(first, desc.copies);
                cpu.skinned.free(entry, desc.copies);
                return Err(Error::Capacity {
                    what: "emissive list entries",
                    capacity: self.scene.desc.emissive_capacity,
                });
            };
            cpu.skinned_high_water = cpu.skinned_high_water.max(entry + desc.copies);
            let holds = &mut cpu.slot_holds[first as usize..(first + desc.copies) as usize];
            for (hold, copy) in holds.iter_mut().zip(0..) {
                *hold = SlotHold {
                    model: desc.model.0,
                    first_entry: emissive + copy * ranks,
                    entries: ranks,
                    material_base: desc.material_set.unwrap_or(NO_MATERIAL_SET),
                };
            }
            if section_mask != 0 {
                cpu.masks.insert((desc.model, section_mask));
            }
            cpu.ptlas.place_skinned(first, desc.copies);
            (first, entry, emissive)
        };
        if let Some(room) = entries {
            self.put_entries(room, first, emissive, ranks);
        }
        let first_emissive = self.scene.desc.producer_capacity + emissive;
        let range = InstanceRange {
            first,
            count: desc.copies,
        };
        self.push(Record::values(
            self.next_seq(),
            self.scene.tables.instances.gpu().offset(u64::from(first)),
            records_src,
            desc.copies,
        ));
        for (copy, r) in (0..).zip(records.iter_mut()) {
            *r = InstanceGpu {
                first_emissive: if ranks > 0 {
                    first_emissive + copy * ranks
                } else {
                    NO_EMISSIVE
                },
                emissive_count: ranks,
                group: desc.transforms.0,
                group_index: desc.first + copy,
                model: desc.model.0,
                material_base: desc.material_set.unwrap_or(NO_MATERIAL_SET),
                pose_group: desc.palette.0,
                pose_index: entry + copy,
                first_part: u32::MAX,
                rt_range: half::f16::from_f32(desc.rt_range).to_bits(),
                animation_range: 0,
                flags: (desc.flags.0 & InstanceFlags::CONSUMER) | InstanceFlags::SKINNED,
                _pad: 0,
                pose_variants: 0,
                pose_stride: 0,
                debug_lod: 0,
                user_tag: desc.user_tag,
                variant: u32::from(section_mask) << 16,
            };
        }
        self.push(Record::values(
            self.next_seq(),
            self.scene.tables.skinned.gpu().offset(u64::from(entry)),
            skinned_src,
            desc.copies,
        ));
        for (copy, k) in (0..).zip(skinned.iter_mut()) {
            let [positions, previous, normals, tangents, boxes] = arrays.of(copy);
            *k = SkinnedGpu {
                positions: positions.cast(),
                previous: previous.cast(),
                normals: normals.cast(),
                tangents: tangents.cast(),
                boxes: boxes.cast(),
                slot: first + copy,
                palette_group: desc.palette.0,
                palette_first: desc.palette_first + copy * model.joints,
                joint_count: model.joints,
            };
        }
        self.push(Record::values(
            self.next_seq(),
            self.scene.tables.skin_state.gpu().offset(u64::from(entry)),
            states_src,
            desc.copies,
        ));
        states.fill(SkinStateGpu {
            cut: u32::MAX,
            ..Zeroable::zeroed()
        });
        self.reset_slot_state(range);
        self.scene.cpu.lock().unwrap().skins.push(SkinRange {
            slots: range,
            entry,
            arrays,
            live: desc.copies,
        });
        Ok(range)
    }

    /// Panics unless material set `set` from consumer base `base`, one entry per section of
    /// `model`, ends within the consumer's part of the material-set table.
    fn check_material_set(&self, model: ModelId, base: u32, set: u8) {
        let sections = self
            .model_cpu
            .get(model.0 as usize)
            .map_or(0, |m| m.sections);
        let capacity = self.scene.desc.material_set_capacity;
        let end = u64::from(base) + (u64::from(set) + 1) * u64::from(sections);
        assert!(
            end <= u64::from(capacity),
            "material set {set} from base {base} of {model:?}'s {sections} sections ends past \
             the {capacity} entries of material_set_capacity"
        );
    }

    /// Panics unless each group is the null group or one the consumer made. Kernels index the
    /// group table by it unbounded, and a group created later can take an id no group holds. A
    /// page's group is orr's, and holds records where a pose or palette reads matrices.
    fn check_groups(&self, groups: impl IntoIterator<Item = TransformGroupId>) {
        // Found under the lock and raised after it, so a refused call leaves the scene usable.
        let refused = {
            let cpu = self.scene.cpu.lock().unwrap();
            groups.into_iter().find(|group| {
                group.0 != 0
                    && !cpu
                        .groups
                        .get(group.0 as usize)
                        .and_then(Option::as_ref)
                        .is_some_and(|g| g.page.is_none())
            })
        };
        if let Some(group) = refused {
            panic!("{group:?} is not a group the consumer made");
        }
    }

    /// Refuses a model without a mesh and one no ready mount or live edit holds: the model
    /// table holds no record for it, or that of an unmounted pack's model.
    fn placeable(&self, model: ModelId) -> Result<()> {
        if self.mounts.is_live(model) {
            return Ok(());
        }
        // A resident cage-only model keeps its cage; unmounting clears it.
        let i = model.0 as usize;
        let cage_only = self.model_cpu.get(i).is_some_and(|m| m.cage_only)
            && self.cages.get(i).is_some_and(Option::is_some);
        Err(if cage_only {
            Error::NoMesh(model)
        } else {
            Error::NotResident(model)
        })
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
                value: u32::from(flags.0 & InstanceFlags::CONSUMER) << shift,
                mask: u32::from(InstanceFlags::CONSUMER) << shift,
            },
        });
    }

    /// Holds the copies of `range` at one DAG level or scales their errors, in raster and ray
    /// tracing; `None` lets them choose their cuts again.
    pub fn debug_lod(&mut self, range: InstanceRange, lod: Option<DebugLod>) {
        let byte = std::mem::offset_of!(InstanceGpu, debug_lod);
        let shift = 8 * (byte % 4);
        self.push(Record {
            seq: self.next_seq(),
            dst: self.instance_word(range.first, byte & !3),
            count: range.count,
            stride: words(size_of::<InstanceGpu>()),
            op: Op::Masked {
                value: u32::from(DebugLod::bits(lod)) << shift,
                mask: 0xffff << shift,
            },
        });
    }

    /// Sets the variant words of copies `first..` of `range`. Ray tracing hides a copy's
    /// masked sections through BLAS built per (model, cut, mask) the first frame a copy
    /// names the mask; a mask hides an animated part's instance only when it covers every
    /// section of the part.
    ///
    /// Panics if a variant's material set ends past `material_set_capacity` on its copy's base,
    /// names a set other than 0 on a cooked page's base, or lands on a copy of a cooked
    /// instance page.
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
        let slot = range.first + first;
        let (on_page, holds) = {
            let cpu = self.scene.cpu.lock().unwrap();
            let on_page = cpu
                .pages
                .iter()
                .any(|p| p.range.first < slot + count && slot < p.range.first + p.range.count);
            let holds = cpu.slot_holds[slot as usize..(slot + count) as usize].to_vec();
            (on_page, holds)
        };
        assert!(
            !on_page,
            "copies of a cooked page take their variants from the page"
        );
        let capacity = self.scene.desc.material_set_capacity;
        for (v, hold) in variants.iter().zip(&holds) {
            match hold.material_base {
                NO_MATERIAL_SET => {}
                b if b >= capacity => assert!(
                    v.material_set == 0,
                    "material set {} of a cooked page's base, which holds one",
                    v.material_set
                ),
                b => self.check_material_set(ModelId(hold.model), b, v.material_set),
            }
        }
        let (src, out) = self.window.slice::<CopyVariant>(self.device, count)?;
        out.copy_from_slice(variants);
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        for (v, hold) in variants.iter().zip(&holds) {
            let sections = self
                .model_cpu
                .get(hold.model as usize)
                .map_or(0, |m| m.sections);
            let mask = u32::from(v.section_mask) & ((1 << sections.min(16)) - 1);
            if mask != 0 {
                cpu.masks.insert((ModelId(hold.model), mask as u16));
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
            self.scene
                .tables
                .material_sets
                .gpu()
                .offset(u64::from(first)),
            count,
        )
    }

    /// Empties the slots of `range` and their emissive list entries; they are reused once
    /// this frame has completed.
    pub fn remove(&mut self, range: InstanceRange) {
        let seq = self.next_seq();
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        self.scene.empty_slots(cpu, range, self.frame, seq);
        cpu.removed.push((self.frame, range));
        cpu.ptlas.remove(range.first, range.count, self.frame);
    }

    /// Sets or clears producer entry `id`. Setting allocates the entry's batch and build
    /// memory and resets its counts; its templates build this frame. The previous entry's
    /// memory, if any, is freed once this frame has completed. Panics if `desc`'s transform
    /// group is neither the null group nor one from `create_transform_group`.
    pub fn set_producer(&mut self, id: ProducerId, desc: Option<&ProducerDesc<'_>>) -> Result<()> {
        self.set_entry(id, desc, None)
    }

    /// Sets or clears producer entry `id` as one copy of a model's subdivision cage. Every
    /// frame the scene is updated, the copy evaluates its cage's base patches on the limit
    /// surface (skinned copies after posing the cage by their palette entries) and orr's
    /// tessellator dices them at rates from the scene focus, displaced through
    /// `desc.displacement`, into the clusters raster and ray tracing draw. Faces near
    /// extraordinary vertices and creases were refined to the cage's isolation level when
    /// it cooked; each patch dices like any other. Each diced cluster takes the class and alpha
    /// test of the material it binds. The copy ends with its model's pack or edit, and `id` stays
    /// the consumer's. Panics with materials for another section count, a palette for a cage
    /// without skin or a model without a skeleton, or a transform or palette group that is
    /// neither the null group nor one from `create_transform_group`.
    pub fn set_subd(&mut self, id: ProducerId, desc: Option<&SubdDesc<'_>>) -> Result<()> {
        let Some(d) = desc else {
            return self.set_entry(id, None, None);
        };
        self.check_groups(d.palette.map(|(palette, _)| palette));
        let cage = self.cages.get(d.model.0 as usize).and_then(Option::as_ref);
        let joints = self
            .model_cpu
            .get(d.model.0 as usize)
            .map_or(0, |m| m.joints);
        subd::check(d, cage, joints)?;
        let cage = cage.expect("checked");
        let materials = if d.materials.is_empty() {
            &cage.materials[..]
        } else {
            d.materials
        };
        let producer = ProducerDesc {
            max_clusters: d.max_clusters,
            max_vertices: d.max_vertices,
            max_segments: 0,
            update: ProducerUpdate::EveryFrame,
            topologies: &[],
            tessellated: true,
            max_patch_clusters: cage.shape.clusters,
            max_patch_vertices: cage.shape.vertices,
            displacement: d.displacement,
            displacement_reach: d.displacement_reach + cage.shape.deviation,
            opacity: Opacity::Sections,
            materials,
            transform: d.transform,
            origin: d.origin,
            // A world length is another entry space length once the placement's scale changes.
            previous_positions: d.palette.is_some() || self.materials.any_world_length(materials),
            vertex_user: false,
            generated_records: false,
            flags: d.flags,
            user_tag: d.user_tag,
        };
        let build = SubdBuild {
            cage: cage.cage,
            points: cage.points,
            shape: cage.shape,
            palette: d.palette.map(|(g, first)| (g.0, first, joints)),
        };
        self.set_entry(id, Some(&producer), Some(&build))?;
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        if let Some(p) = &mut cpu.producers[id.0 as usize] {
            p.model = Some(d.model);
        }
        Ok(())
    }

    fn set_entry(
        &self,
        id: ProducerId,
        desc: Option<&ProducerDesc<'_>>,
        subd: Option<&SubdBuild>,
    ) -> Result<()> {
        assert!(
            id.0 < self.scene.desc.producer_capacity,
            "{id:?} outside the producer table"
        );
        self.check_groups(desc.map(|d| d.transform.0));
        let t = &self.scene.tables;
        let i = u64::from(id.0);
        let counts = t.producer_counts.gpu().offset(i);
        // Window room comes first, so nothing is recorded unless all of it is.
        let mut copies = Vec::with_capacity(6);
        let (record, batch, patches, entry, producer) = match desc {
            None => (
                ProducerGpu::zeroed(),
                ProducerBatch::zeroed(),
                ProducerBatch::zeroed(),
                EmissiveGpu::zeroed(),
                None,
            ),
            Some(d) => {
                let b = producer::build(
                    self.device,
                    d,
                    counts,
                    t.batches.gpu().offset(i),
                    self.patterns,
                    subd,
                )?;
                let n = (b.static_bytes.len() / 4) as u32;
                let (src, words) = self.window.slice::<u32>(self.device, n)?;
                bytemuck::cast_slice_mut(words).copy_from_slice(&b.static_bytes);
                copies.push(Record::values(self.next_seq(), b.static_at, src, n));
                let entry = EmissiveGpu {
                    kind: EMISSIVE_PRODUCER,
                    source: id.0,
                    ..Zeroable::zeroed()
                };
                (b.record, b.batch, b.patches, entry, Some(b.producer))
            }
        };
        copies.push(self.staged(t.producers.gpu().offset(i), &record)?);
        copies.push(self.staged(t.batches.gpu().offset(i), &batch)?);
        copies.push(self.staged(t.patches.gpu().offset(i), &patches)?);
        copies.push(self.staged(t.tess_state.gpu().offset(i), &TessStateGpu::zeroed())?);
        copies.push(self.staged(t.emissive.gpu().offset(i), &entry)?);
        let emit_counts = producer.as_ref().map_or(counts, |p| p.emit_counts);
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        cpu.records.append(&mut copies);
        if emit_counts != counts {
            cpu.records
                .push(Record::fill(self.next_seq(), emit_counts.address().0, 4, 0));
        }
        cpu.records
            .push(Record::fill(self.next_seq(), counts.address().0, 4, 0));
        // The new entry has built nothing; its PTLAS instance follows from the record.
        let state = t.producer_state.gpu().offset(i);
        let blas = std::mem::offset_of!(ProducerStateGpu, blas) as u64;
        cpu.records.push(Record::fill(
            self.next_seq(),
            state.address().0 + blas,
            2,
            0,
        ));
        let clusters = std::mem::offset_of!(ProducerStateGpu, clusters) as u64;
        cpu.records.push(Record::fill(
            self.next_seq(),
            state.address().0 + clusters,
            1,
            0,
        ));
        if let Some(old) = std::mem::replace(&mut cpu.producers[id.0 as usize], producer) {
            cpu.retired.push((self.frame, old));
        }
        Ok(())
    }

    /// Sets or clears light `id`. A sphere of non-zero radius and a rect also stand in the
    /// scene's ray tracing as a proxy that only the BSDF rays of paths meet, weighed against
    /// light sampling. Camera, lens, shadow and G-buffer rays do not see it, so a light casts
    /// no shadow and has no body in the image; a lamp the camera sees is an emissive instance.
    pub fn set_light(&mut self, id: LightId, light: Option<&Light>) -> Result<()> {
        assert!(
            id.0 < self.scene.desc.light_capacity,
            "{id:?} outside the light table"
        );
        let proxied = matches!(
            light,
            Some(Light::Sphere { radius, .. }) if *radius > 0.0
        ) || matches!(light, Some(Light::Rect { .. }));
        let (src, out) = self.window.slice::<LightGpu>(self.device, 1)?;
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        cpu.ptlas.set_light(id.0, proxied);
        cpu.lights.set(id.0, light);
        let origin = cpu.origin_cell;
        out[0] = light.map_or_else(LightGpu::zeroed, |l| light_gpu(l, origin, self.cell_log2));
        cpu.records.push(Record::values(
            self.next_seq(),
            self.scene.tables.lights.gpu().offset(u64::from(id.0)),
            src,
            1,
        ));
        Ok(())
    }

    /// Sets or clears decal `id`, from this frame on.
    pub fn set_decal(&mut self, id: DecalId, decal: Option<&Decal>) -> Result<()> {
        assert!(
            id.0 < self.scene.desc.decal_capacity,
            "{id:?} outside the decal table"
        );
        let record = decal.map_or_else(DecalGpu::zeroed, |d| decal_gpu(d, self.cell_log2));
        let copy = self.staged(
            self.scene.tables.decals.gpu().offset(u64::from(id.0)),
            &record,
        )?;
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        cpu.decals[id.0 as usize] = decal.map(|d| listing(d, self.cell_log2));
        cpu.decals_changed = true;
        cpu.records.push(copy);
        Ok(())
    }

    /// Sets or clears medium proxy `id`, from this frame on.
    pub fn set_medium_proxy(
        &mut self,
        id: MediumProxyId,
        proxy: Option<&MediumProxy>,
    ) -> Result<()> {
        assert!(
            id.0 < self.scene.desc.medium_proxy_capacity,
            "{id:?} outside the medium proxy table"
        );
        let record = proxy.map_or_else(MediumProxyGpu::zeroed, |p| proxy_gpu(p, self.cell_log2));
        let copy = self.staged(
            self.scene.tables.proxies.gpu().offset(u64::from(id.0)),
            &record,
        )?;
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        cpu.ptlas.set_proxy(id.0, proxy.is_some());
        cpu.records.push(copy);
        Ok(())
    }

    /// Consumer data for the linked environment, which kernels that link one need before the
    /// scene's first `update_scenes`. An environment that reads no data takes a null pointer to
    /// its `T`. Panics unless the kernels link the type that reads `T`, and on a null pointer
    /// unless `T` is [`NULL_SAFE`](EnvironmentData::NULL_SAFE).
    pub fn set_environment<T: EnvironmentData>(&mut self, data: GpuPtr<T>) {
        assert!(
            self.kernels.environment == T::TYPE,
            "the kernels link {} as the environment; this data is for {}",
            self.kernels.environment,
            T::TYPE
        );
        assert!(
            T::NULL_SAFE || data != GpuPtr::NULL,
            "{} reads its data, and this data is null",
            T::TYPE
        );
        self.scene.cpu.lock().unwrap().environment = Some(data.address().0);
    }

    /// Consumer data for the linked medium, which kernels that link one need before the scene's
    /// first `update_scenes`. A medium that reads no data takes a null pointer to its `T`.
    /// Panics unless the kernels link the type that reads `T`, and on a null pointer unless `T`
    /// is [`NULL_SAFE`](MediumData::NULL_SAFE).
    pub fn set_medium<T: MediumData>(&mut self, data: GpuPtr<T>) {
        assert!(
            self.kernels.medium == T::TYPE,
            "the kernels link {} as the medium; this data is for {}",
            self.kernels.medium,
            T::TYPE
        );
        assert!(
            T::NULL_SAFE || data != GpuPtr::NULL,
            "{} reads its data, and this data is null",
            T::TYPE
        );
        self.scene.cpu.lock().unwrap().medium = Some(data.address().0);
    }

    /// Consumer data for linked surfaces whose material names none (`MaterialGpu::data`).
    pub fn set_surface_data(&mut self, data: GpuAddress) {
        self.scene.cpu.lock().unwrap().surface_data = data.0;
    }

    /// The eye of the scene's per-frame work and the anchor of its render origin.
    pub fn set_focus(&mut self, focus: DVec3) {
        self.scene.cpu.lock().unwrap().focus = Some(focus);
    }
}
