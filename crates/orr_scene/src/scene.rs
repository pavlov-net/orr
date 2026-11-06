//! A scene's GPU tables, its CPU bookkeeping and the writer consumers fill it through.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use orr_gpu::{Allocation, Device, GpuAddress, GpuPtr, MemoryKind};
use orr_math::{DVec3, IVec3};

use crate::decals::{Decal, DecalId, Listing, decal_gpu, listing, table_sizes};
use crate::emissive::{ModelEmission, Run, Runs};
use crate::lights::{EMISSIVE_INSTANCE, EMISSIVE_PRODUCER, Light, LightId, light_gpu};
use crate::material::{MaterialId, MaterialTable};
use crate::medium_proxy::{MediumProxy, MediumProxyId, proxy_gpu};
use crate::memory::{RangeAllocator, UploadWindow};
use crate::mount::{InstancePage, Mounts, PageId};
use crate::producer::{self, Producer, ProducerDesc, ProducerId, ProducerUpdate};
use crate::ptlas::Ptlas;
use crate::shaders::instance_page::PagePrototypeGpu;
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
    /// Emissive list entries for instances: one per copy per EMISSIVE span of its model.
    pub emissive_capacity: u32,
    /// PTLAS instances for the animated parts of copies inside their animation range, and
    /// one for each copy fading between two chain steps.
    pub part_capacity: u32,
    /// PTLAS cell partitions.
    pub ptlas_partitions: u32,
    /// Instances one PTLAS partition holds, at most.
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
    /// Replaces a prototype's material set by a base in the scene's material-set table.
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
    pub(crate) root: Allocation<SceneRoot>,
    pub(crate) instances: Allocation<InstanceGpu>,
    pub(crate) slot_state: Allocation<SlotStateGpu>,
    pub(crate) groups: Allocation<GroupGpu>,
    pub(crate) material_sets: Allocation<MaterialId>,
    pub(crate) lights: Allocation<LightGpu>,
    /// Producer entries first, then instance spans.
    pub(crate) emissive: Allocation<EmissiveGpu>,
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
    /// `None` once its pack unmounts, when its copies hide for good: the id can name another
    /// page by then.
    pub(crate) page: Option<PageId>,
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
    extent: [f32; 3],
    log2_scale_min: f32,
    log2_scale_max: f32,
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
                extent: header.extent,
                log2_scale_min: header.log2_scale_min,
                log2_scale_max: header.log2_scale_max,
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
                frame: frame.palette,
                frame_index: frame.index,
                frame_origin: frame.origin,
                current: records,
                previous: records,
                count: self.count,
                format: GROUP_QUANTIZED_PAGE,
                anchor_cell: self.anchor.cell.to_array(),
                anchor_offset: self.anchor.offset.to_array(),
                origin: p.origin,
                extent: p.extent,
                log2_scale_min: p.log2_scale_min,
                log2_scale_max: p.log2_scale_max,
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
            frame: frame.palette,
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
    /// The `orr_shaders.trace` candidate class its copies take; `None` where its sections mix
    /// dielectric or presence materials with others.
    pub(crate) candidate_class: Option<u8>,
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
/// its slots, and what its removal frees.
pub(crate) struct PagePlacement {
    pub(crate) range: InstanceRange,
    pub(crate) group: TransformGroupId,
    pub(crate) page: PageId,
    pub(crate) prototypes: Allocation<u8>,
    pub(crate) prototype_count: u32,
    pub(crate) spans: GpuPtr<EmissiveGpu>,
    pub(crate) first_entry: u32,
    pub(crate) linear_rows: u32,
    pub(crate) entry_stride: u32,
    pub(crate) expanded: bool,
    /// Its prototypes' entries of the page part of the material-set table.
    pub(crate) sets: (u32, u32),
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

pub(crate) struct SceneCpu {
    pub(crate) slots: RangeAllocator,
    /// The model of each slot handed out.
    slot_models: Vec<u32>,
    /// Section masks copies name since the last frame: their BLAS build if missing.
    pub(crate) masks: HashSet<(ModelId, u16)>,
    /// Ranges removed in a frame, freed once that frame has completed.
    pub(crate) removed: Vec<(u64, InstanceRange)>,
    /// Instance entries of the emissive list, counted after the producer entries.
    pub(crate) emissive: RangeAllocator,
    emissive_runs: Runs,
    /// Entry ranges removed in a frame, freed once that frame has completed.
    pub(crate) emissive_removed: Vec<(u64, u32, u32)>,
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
    pub(crate) environment: u64,
    pub(crate) medium: u64,
    pub(crate) surface_data: u64,
    pub(crate) near_range: f32,
    pub(crate) ptlas: Ptlas,
    /// Made at the scene's first update.
    pub(crate) cuts: Option<Cuts>,
    /// Skinned-copy entries.
    pub(crate) skinned: RangeAllocator,
    /// One past the highest skinned-copy entry handed out.
    pub(crate) skinned_high_water: u32,
    skins: Vec<SkinRange>,
    /// Entry ranges and vertex arrays of removed skinned copies, freed once that frame has
    /// completed.
    pub(crate) skins_removed: Vec<(u64, u32, u32, Option<CopyArrays>)>,
    /// Made at the first update with skinned copies.
    pub(crate) skin_frame: Option<SkinFrame>,
    /// The first view `raster_view` rendered of the scene in its last frame, which the
    /// tessellator's edge factors test for what it saw.
    pub(crate) seen_by: Option<LastView>,
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
    pub(crate) fn new(device: &Device, desc: &SceneDesc) -> Result<Self> {
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
                slot_models: vec![0; desc.instance_capacity as usize],
                masks: HashSet::new(),
                removed: Vec::new(),
                emissive: RangeAllocator::new(desc.emissive_capacity),
                emissive_runs: Runs::default(),
                emissive_removed: Vec::new(),
                producers: (0..desc.producer_capacity).map(|_| None).collect(),
                retired: Vec::new(),
                producer_readback: None,
                groups: vec![None],
                records,
                focus: None,
                pixels_per_radian: DEFAULT_PIXELS_PER_RADIAN,
                requested_detail: None,
                origin_cell: IVec3::ZERO,
                environment: 0,
                medium: 0,
                surface_data: 0,
                near_range: 256.0,
                ptlas: Ptlas::new(device, desc)?,
                cuts: None,
                skinned: RangeAllocator::new(desc.skinned_capacity),
                skinned_high_water: 0,
                skins: Vec::new(),
                skins_removed: Vec::new(),
                skin_frame: None,
                seen_by: None,
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

/// Writes one scene's changes into this frame's upload window. Returned slices are
/// write-only (write-combined memory) and live until `update_scenes`. Send: different
/// scenes' writers, and disjoint pieces of one returned slice, can be filled on different
/// threads.
pub struct SceneWriter<'f> {
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

    fn write<T: Pod>(&self, dst: GpuPtr<T>, value: T) -> Result<()> {
        self.upload::<T>(dst, 1)?[0] = value;
        Ok(())
    }

    /// A new transform group; its transforms start all-zero (hidden).
    pub fn create_transform_group(
        &mut self,
        desc: &TransformGroupDesc,
    ) -> Result<TransformGroupId> {
        self.create_group(desc, None)
    }

    fn create_group(
        &mut self,
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
        let record = group.record();
        let zero = Record::fill(
            self.next_seq(),
            group.buffers[0].gpu().address().0,
            words(bytes as usize),
            0,
        );
        let id = {
            let mut cpu = self.scene.cpu.lock().unwrap();
            let id = cpu.free_group();
            if id as u32 >= self.scene.desc.group_capacity {
                return Err(Error::Capacity {
                    what: "transform groups",
                    capacity: self.scene.desc.group_capacity,
                });
            }
            cpu.put_group(id, group);
            cpu.records.push(zero);
            id as u32
        };
        self.write(self.scene.tables.groups.gpu().offset(u64::from(id)), record)?;
        Ok(TransformGroupId(id))
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
        &mut self,
        group: TransformGroupId,
        first: u32,
        count: u32,
        without_motion: bool,
    ) -> Result<&'f mut [AffineRows]> {
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
        let (src, out) = self.window.slice::<AffineRows>(self.device, count)?;
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
    /// EMISSIVE span of the model. With a pose, each copy's PTLAS partition keeps room for
    /// its part run. Copies take the candidate class of the model's own materials (a
    /// `material_set` does not change it): dielectric or presence models are traced by
    /// dielectric rays only and never rasterized; a model that mixes either with other
    /// materials fails with [`Error::MixedClasses`].
    pub fn add_instances(&mut self, desc: &InstancesDesc) -> Result<InstanceRange> {
        self.has_mesh(desc.model)?;
        let candidate_class = self
            .model_cpu
            .get(desc.model.0 as usize)
            .map_or(Some(0), |m| m.candidate_class)
            .ok_or(Error::MixedClasses(desc.model))?;
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
            (first, entry, partitions, cpu.ptlas.slot_table.gpu())
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
            self.scene.tables.instances.gpu().offset(u64::from(first)),
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
                flags: desc.flags.0 & InstanceFlags::CONSUMER,
                candidate_class,
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
    pub fn add_instance_page(&mut self, r: &InstancePageRef<'_>) -> Result<InstanceRange> {
        let motion = self.motion_group(r.page, r.frame)?;
        let placed = self.place_instance_page(r, motion);
        self.settle_motion(r.page, motion, placed)
    }

    fn place_instance_page(
        &mut self,
        r: &InstancePageRef<'_>,
        motion: Option<TransformGroupId>,
    ) -> Result<(InstanceRange, TransformGroupId)> {
        let page = self
            .mounts
            .pages
            .get(r.page.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(Error::UnknownPage(r.page))?;
        let count = u32::from(page.info.header.record_count);
        let protos = self.mounts.page_prototypes(r.page)?;
        if let Some(p) = protos.iter().find(|p| self.is_skinned(p.model)) {
            return Err(Error::SkinnedPage(p.model));
        }
        let header = page.info.header;
        let linear_rows = header.linear_rows(page.bytes as usize)? as u32;
        let set_entries: u32 = protos
            .iter()
            .map(|p| p.sets.as_ref().map_or(0, |s| s.len() as u32))
            .sum();
        let empty = ModelCpu::default();
        let model_cpu = |model: ModelId| self.model_cpu.get(model.0 as usize).unwrap_or(&empty);
        let classes = protos
            .iter()
            .map(|p| {
                model_cpu(p.model)
                    .candidate_class
                    .ok_or(Error::MixedClasses(p.model))
            })
            .collect::<Result<Vec<_>>>()?;
        // Every copy of the page takes room for the emissive spans of the most emissive model.
        let spans_of = |model: ModelId| {
            self.models
                .get(model.0 as usize)
                .and_then(Option::as_ref)
                .map_or(&[][..], |m| &m.spans[..])
        };
        let stride = protos
            .iter()
            .map(|p| spans_of(p.model).len() as u32)
            .max()
            .unwrap_or(0);
        let span_total: usize = protos.iter().map(|p| spans_of(p.model).len()).sum();
        let copy_bytes = (protos.len() * size_of::<PagePrototypeGpu>()) as u64;
        let spans_at = copy_bytes.next_multiple_of(16);
        let span_bytes = (span_total * size_of::<EmissiveGpu>()) as u64;
        let mut table = self.device.gpu_malloc::<u8>(
            MemoryKind::Upload,
            (spans_at + span_bytes).max(16),
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
        let entry = if stride > 0 {
            let Some(e) = cpu.emissive.alloc(count * stride) else {
                cpu.slots.free(first, count);
                cpu.page_sets.free(set_base, set_entries);
                return Err(Error::Capacity {
                    what: "emissive list entries",
                    capacity: self.scene.desc.emissive_capacity,
                });
            };
            cpu.emissive_runs.insert(
                first,
                Run {
                    copies: count,
                    entry: e,
                    spans: stride,
                },
            );
            let at = self.emissive_entry(e).address().0;
            let n = count * stride * words(size_of::<EmissiveGpu>());
            cpu.records.push(Record::fill(self.next_seq(), at, n, 0));
            e
        } else {
            0
        };

        let mut copies = Vec::with_capacity(protos.len());
        let mut spans: Vec<EmissiveGpu> = Vec::with_capacity(span_total);
        let mut at_set = self.scene.desc.material_set_capacity + set_base;
        for ((index, p), &class) in protos.iter().enumerate().zip(&classes) {
            let cooked_base = match &p.sets {
                Some(set) => {
                    let values: Vec<u32> = set.iter().map(|m| m.0).collect();
                    let (src, out) = self.window.slice::<u32>(self.device, values.len() as u32)?;
                    out.copy_from_slice(&values);
                    let dst = self.scene.tables.material_sets.gpu();
                    cpu.records.push(Record::values(
                        self.next_seq(),
                        dst.offset(u64::from(at_set)).cast::<u32>(),
                        src,
                        values.len() as u32,
                    ));
                    at_set += values.len() as u32;
                    at_set - values.len() as u32
                }
                None => u32::MAX,
            };
            let base = r
                .material_sets
                .iter()
                .find(|(q, _)| usize::from(*q) == index)
                .map_or(cooked_base, |&(_, b)| b);
            let pose = r.poses.iter().find(|(m, _)| *m == p.model).map(|&(_, q)| q);
            let flags = if p.flags == 0 {
                r.flags.0
            } else {
                p.flags as u8 & r.flags.0
            };
            if p.mask != 0 {
                cpu.masks.insert((p.model, p.mask));
            }
            let first_span = spans.len() as u32;
            if let Some(m) = self.models.get(p.model.0 as usize).and_then(Option::as_ref) {
                spans.extend(m.spans.iter().map(|s| EmissiveGpu {
                    kind: EMISSIVE_INSTANCE,
                    first_cluster: s.first_cluster,
                    cluster_count: s.cluster_count,
                    section: u32::from(s.section),
                    sections: m.sections,
                    material: m.materials[usize::from(s.section)].0,
                    model_area: s.area,
                    ..Zeroable::zeroed()
                }));
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
                    candidate_class: class,
                    pose_variants: pose.map_or(0, |q| q.variants as u16),
                    pose_stride: pose.map_or(0, |q| q.stride as u16),
                    debug_lod: 0,
                    user_tag: r.user_tag,
                    variant: u32::from(p.pose) | u32::from(p.mask) << 16,
                },
                first_span,
                span_count: spans.len() as u32 - first_span,
                _pad: [0; 2],
            });
        }
        let out = table.as_mut_slice().expect("upload is mapped");
        out[..copy_bytes as usize].copy_from_slice(bytemuck::cast_slice(&copies));
        out[spans_at as usize..(spans_at + span_bytes) as usize]
            .copy_from_slice(bytemuck::cast_slice(&spans));
        let spans_gpu = table.gpu().offset(spans_at).cast();
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
            spans: spans_gpu,
            first_entry: entry,
            linear_rows,
            entry_stride: stride,
            expanded: false,
            sets: (set_base, set_entries),
        });
        let (src, out) = self.window.slice::<u32>(self.device, count)?;
        out.copy_from_slice(&partitions);
        cpu.records.push(Record::values(
            self.next_seq(),
            slot_table.offset(u64::from(first)),
            src,
            count,
        ));
        let (src, out) = self.window.slice::<GroupGpu>(self.device, 1)?;
        out[0] = record;
        cpu.records.push(Record::values(
            self.next_seq(),
            self.scene.tables.groups.gpu().offset(u64::from(group_id)),
            src,
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
        &mut self,
        page: PageId,
        frame: Option<(TransformGroupId, u32)>,
    ) -> Result<Option<TransformGroupId>> {
        let held = self
            .mounts
            .pages
            .get(page.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(Error::UnknownPage(page))?;
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
                self.clear_group(cpu, group, self.next_seq());
                return Err(e);
            }
        };
        let held = self.mounts.pages[page.0 as usize].as_ref();
        let p = held.ok_or(Error::UnknownPage(page))?;
        let h = p.info.header;
        cpu.page_motions.push(PageMotion {
            range,
            page: Some(page),
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
    /// skeleton clips. Copies of a page that holds still ignore it.
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
    pub fn add_skinned_page(&mut self, r: &SkinnedPageRef) -> Result<InstanceRange> {
        let motion = self.motion_group(r.page, r.frame)?;
        let placed = self.place_skinned_page(r, motion);
        self.settle_motion(r.page, motion, placed)
    }

    fn place_skinned_page(
        &mut self,
        r: &SkinnedPageRef,
        motion: Option<TransformGroupId>,
    ) -> Result<(InstanceRange, TransformGroupId)> {
        let page = self
            .mounts
            .pages
            .get(r.page.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(Error::UnknownPage(r.page))?;
        let protos = self.mounts.page_prototypes(r.page)?;
        let [proto] = protos.as_slice() else {
            return Err(Error::NotSkinnedPage(r.page));
        };
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
                let values: Vec<u32> = set.iter().map(|m| m.0).collect();
                let (src, out) = self.window.slice::<u32>(self.device, set_entries)?;
                out.copy_from_slice(&values);
                let at = self.scene.desc.material_set_capacity + set_base;
                let table = self.scene.tables.material_sets.gpu();
                cpu.records.push(Record::values(
                    self.next_seq(),
                    table.offset(u64::from(at)).cast::<u32>(),
                    src,
                    set_entries,
                ));
            }
            cpu.put_group(id, group);
            let (src, out) = self.window.slice::<GroupGpu>(self.device, 1)?;
            out[0] = record;
            cpu.records.push(Record::values(
                self.next_seq(),
                self.scene.tables.groups.gpu().offset(id as u64),
                src,
                1,
            ));
            (TransformGroupId(id as u32), (set_base, set_entries))
        };
        let cooked_set = cooked.map(|_| self.scene.desc.material_set_capacity + sets.0);
        let flags = match proto.flags {
            0 => r.flags,
            f => InstanceFlags(f as u8 & r.flags.0),
        };
        let placed = self.add_skinned(&SkinnedDesc {
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
        });
        let range = match placed {
            Ok(range) => range,
            Err(e) => {
                let seq = self.next_seq();
                let cpu = &mut *self.scene.cpu.lock().unwrap();
                self.clear_group(cpu, id, seq);
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
        if proto.mask != 0 {
            let variant = CopyVariant {
                section_mask: proto.mask,
                ..CopyVariant::default()
            };
            self.copy_variants(range, 0, &vec![variant; copies as usize])?;
        }
        Ok((range, id))
    }

    /// Empties group `id` and its record.
    fn clear_group(&self, cpu: &mut SceneCpu, id: TransformGroupId, seq: u64) {
        let group = cpu.groups[id.0 as usize].take();
        cpu.groups_removed.extend(group.map(|g| (self.frame, g)));
        let at = self.scene.tables.groups.gpu().offset(u64::from(id.0));
        let n = words(size_of::<GroupGpu>());
        cpu.records.push(Record::fill(seq, at.address().0, n, 0));
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
    /// [`Self::add_instances`]. Panics if the model's mesh has no skin or the model no
    /// skeleton.
    pub fn add_skinned(&mut self, desc: &SkinnedDesc) -> Result<InstanceRange> {
        self.has_mesh(desc.model)?;
        let model = self
            .model_cpu
            .get(desc.model.0 as usize)
            .copied()
            .unwrap_or_default();
        let shape = model
            .skin
            .unwrap_or_else(|| panic!("{:?} has no skinned mesh", desc.model));
        assert!(model.joints > 0, "{:?} has no skeleton", desc.model);
        let candidate_class = model
            .candidate_class
            .ok_or(Error::MixedClasses(desc.model))?;
        let arrays = CopyArrays::new(self.device, desc.copies, shape.vertices, shape.clusters)?;
        let (first, entry) = {
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
            cpu.skinned_high_water = cpu.skinned_high_water.max(entry + desc.copies);
            cpu.slot_models[first as usize..(first + desc.copies) as usize].fill(desc.model.0);
            cpu.ptlas.place_skinned(first, desc.copies);
            (first, entry)
        };
        let range = InstanceRange {
            first,
            count: desc.copies,
        };
        let records = self.upload(
            self.scene.tables.instances.gpu().offset(u64::from(first)),
            desc.copies,
        )?;
        for (copy, r) in (0..).zip(records.iter_mut()) {
            *r = InstanceGpu {
                group: desc.transforms.0,
                group_index: desc.first + copy,
                model: desc.model.0,
                material_base: desc.material_set.unwrap_or(u32::MAX),
                pose_group: desc.palette.0,
                pose_index: entry + copy,
                first_part: u32::MAX,
                rt_range: half::f16::from_f32(desc.rt_range).to_bits(),
                animation_range: 0,
                flags: (desc.flags.0 & InstanceFlags::CONSUMER) | InstanceFlags::SKINNED,
                candidate_class,
                pose_variants: 0,
                pose_stride: 0,
                debug_lod: 0,
                user_tag: desc.user_tag,
                variant: 0,
            };
        }
        let skinned = self.upload(
            self.scene.tables.skinned.gpu().offset(u64::from(entry)),
            desc.copies,
        )?;
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
        let states = self.upload(
            self.scene.tables.skin_state.gpu().offset(u64::from(entry)),
            desc.copies,
        )?;
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

    fn has_mesh(&self, model: ModelId) -> Result<()> {
        match self.model_cpu.get(model.0 as usize) {
            Some(m) if m.cage_only => Err(Error::NoMesh(model)),
            _ => Ok(()),
        }
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
        assert!(
            !cpu.pages.iter().any(|p| p.range.first == range.first),
            "copies of a cooked page take their variants from the page"
        );
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
            self.scene
                .tables
                .material_sets
                .gpu()
                .offset(u64::from(first)),
            count,
        )
    }

    /// Instance entry `entry` of the emissive list.
    fn emissive_entry(&self, entry: u32) -> GpuPtr<EmissiveGpu> {
        self.scene
            .tables
            .emissive
            .gpu()
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
        if let Some(i) = cpu.pages.iter().position(|p| p.range == range) {
            let p = cpu.pages.swap_remove(i);
            self.clear_group(cpu, p.group, seq);
            cpu.pages_removed.push((self.frame, p));
        }
        if let Some(i) = cpu.page_motions.iter().position(|m| m.range == range) {
            let m = cpu.page_motions.swap_remove(i);
            self.clear_group(cpu, m.group, seq);
        }
        if let Some(i) = cpu.skinned_pages.iter().position(|p| p.range == range) {
            let p = cpu.skinned_pages.swap_remove(i);
            self.clear_group(cpu, p.group, seq);
            cpu.skinned_sets_removed.push((self.frame, p.sets));
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
            let at = self.scene.tables.skinned.gpu().offset(u64::from(entry));
            let w = n * words(size_of::<SkinnedGpu>());
            cpu.records.push(Record::fill(seq, at.address().0, w, 0));
            let arrays = if r.live == 0 {
                Some(cpu.skins.swap_remove(i).arrays)
            } else {
                i += 1;
                None
            };
            cpu.skins_removed.push((self.frame, entry, n, arrays));
        }
        for (entry, count) in cpu.emissive_runs.remove(range.first, range.count) {
            let at = self.emissive_entry(entry).address().0;
            let n = count * words(size_of::<EmissiveGpu>());
            cpu.records.push(Record::fill(seq, at, n, 0));
            cpu.emissive_removed.push((self.frame, entry, count));
        }
    }

    /// Sets or clears producer entry `id`. Setting allocates the entry's batch and build
    /// memory and resets its counts; its templates build this frame. The previous entry's
    /// memory, if any, is freed once this frame has completed.
    pub fn set_producer(&mut self, id: ProducerId, desc: Option<&ProducerDesc<'_>>) -> Result<()> {
        self.set_entry(id, desc, None)
    }

    /// Sets or clears producer entry `id` as one copy of a model's subdivision cage. Every
    /// frame the scene is updated, the copy evaluates its cage's base patches on the limit
    /// surface (skinned copies after posing the cage by their palette entries) and orr's
    /// tessellator dices them at rates from the scene focus, displaced through
    /// `desc.displacement`, into the clusters raster and ray tracing draw. Faces near
    /// extraordinary vertices and creases were refined to the cage's isolation level when
    /// it cooked; each patch dices like any other. The copy takes the opacity of the cage
    /// sections' own materials; a cage that mixes dielectric or presence materials with
    /// others fails with [`Error::MixedClasses`]. Its cage stays valid until its pack is
    /// unmounted. Panics with materials for another section count, or a palette for a cage
    /// without skin or a model without a skeleton.
    pub fn set_subd(&mut self, id: ProducerId, desc: Option<&SubdDesc<'_>>) -> Result<()> {
        let Some(d) = desc else {
            return self.set_entry(id, None, None);
        };
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
            opacity: cage.opacity.expect("checked"),
            materials,
            transform: d.transform,
            origin: d.origin,
            // A world length is another entry space length once the placement's scale changes.
            previous_positions: d.palette.is_some() || self.materials.any_world_length(materials),
            vertex_user: false,
            flags: d.flags,
            user_tag: d.user_tag,
        };
        let build = SubdBuild {
            cage: cage.cage,
            points: cage.points,
            shape: cage.shape,
            palette: d.palette.map(|(g, first)| (g.0, first, joints)),
        };
        self.set_entry(id, Some(&producer), Some(&build))
    }

    fn set_entry(
        &mut self,
        id: ProducerId,
        desc: Option<&ProducerDesc<'_>>,
        subd: Option<&SubdBuild>,
    ) -> Result<()> {
        assert!(
            id.0 < self.scene.desc.producer_capacity,
            "{id:?} outside the producer table"
        );
        let t = &self.scene.tables;
        let i = u64::from(id.0);
        let counts = t.producer_counts.gpu().offset(i);
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
                let words = self.upload(b.static_at, (b.static_bytes.len() / 4) as u32)?;
                bytemuck::cast_slice_mut(words).copy_from_slice(&b.static_bytes);
                let entry = EmissiveGpu {
                    kind: EMISSIVE_PRODUCER,
                    source: id.0,
                    ..Zeroable::zeroed()
                };
                (b.record, b.batch, b.patches, entry, Some(b.producer))
            }
        };
        let emit_counts = producer.as_ref().map_or(counts, |p| p.emit_counts);
        self.write(t.producers.gpu().offset(i), record)?;
        self.write(t.batches.gpu().offset(i), batch)?;
        self.write(t.patches.gpu().offset(i), patches)?;
        self.write(t.tess_state.gpu().offset(i), TessStateGpu::zeroed())?;
        if emit_counts != counts {
            self.push(Record::fill(self.next_seq(), emit_counts.address().0, 4, 0));
        }
        self.write(t.emissive.gpu().offset(i), entry)?;
        self.push(Record::fill(self.next_seq(), counts.address().0, 4, 0));
        // The new entry has built nothing; its PTLAS instance follows from the record.
        let state = t.producer_state.gpu().offset(i);
        let blas = std::mem::offset_of!(ProducerStateGpu, blas) as u64;
        self.push(Record::fill(
            self.next_seq(),
            state.address().0 + blas,
            2,
            0,
        ));
        let clusters = std::mem::offset_of!(ProducerStateGpu, clusters) as u64;
        self.push(Record::fill(
            self.next_seq(),
            state.address().0 + clusters,
            1,
            0,
        ));
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
        self.write(
            self.scene.tables.lights.gpu().offset(u64::from(id.0)),
            record,
        )
    }

    /// Sets or clears decal `id`, from this frame on.
    pub fn set_decal(&mut self, id: DecalId, decal: Option<&Decal>) -> Result<()> {
        assert!(
            id.0 < self.scene.desc.decal_capacity,
            "{id:?} outside the decal table"
        );
        let record = decal.map_or(DecalGpu::zeroed(), |d| decal_gpu(d, self.cell_log2));
        {
            let mut cpu = self.scene.cpu.lock().unwrap();
            cpu.decals[id.0 as usize] = decal.map(|d| listing(d, self.cell_log2));
            cpu.decals_changed = true;
        }
        self.write(
            self.scene.tables.decals.gpu().offset(u64::from(id.0)),
            record,
        )
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
        let record = proxy.map_or(MediumProxyGpu::zeroed(), |p| proxy_gpu(p, self.cell_log2));
        self.scene
            .cpu
            .lock()
            .unwrap()
            .ptlas
            .set_proxy(id.0, proxy.is_some());
        self.write(
            self.scene.tables.proxies.gpu().offset(u64::from(id.0)),
            record,
        )
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

    /// The eye of the scene's per-frame work and the anchor of its render origin.
    pub fn set_focus(&mut self, focus: DVec3) {
        self.scene.cpu.lock().unwrap().focus = Some(focus);
    }
}
