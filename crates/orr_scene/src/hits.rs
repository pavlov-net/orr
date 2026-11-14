//! Scenes' hit tables: for one ray tracing pipeline, a hit record per ray type for each run of a
//! scene's material rows (`runs.slang`), which `sbt.slang` writes from the scene's tables.

use orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, HitTable, MemoryKind,
    RayPipeline, Stages,
};

use crate::shaders::sbt::HitTableRoot;
use crate::{Error, Result, SceneRoot};

/// `runs.slang`'s RAY_TYPES, FIXED_RUNS and the size of `HitData`, and `sbt.slang`'s
/// FIXED_KINDS and PINNED_KINDS.
const RAY_TYPES: u32 = 3;
const HIT_DATA_BYTES: u64 = 16;
const FIXED_RUNS: u32 = 3;
const FIXED_KINDS: usize = 6;
const PINNED_KINDS: usize = 6;

/// The hit groups of a ray tracing pipeline that a scene's hit records name, by what each
/// record serves: a group index of the pipeline, or `None` for records of zeros. A record of
/// zeros runs no shader and confirms every candidate a trace finds in it.
///
/// The path and shadow groups are pinned: each is the first of [`Self::pins`] groups, one after
/// another, for the materials of one linked surface id (`scene_may.slang`'s `hit_surface`). The
/// first serves any id, and is what a material whose back shows another id takes; where surfaces
/// link, the pin of id `k` follows at `k + 1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HitGroups {
    /// The linked surface types the pinned groups serve, ids 1 to `surfaces`.
    pub surfaces: u32,
    /// Triangles of copies' clusters under path rays: a closest-hit that resolves the hit and an
    /// any-hit that decides by `orr_scene.candidates`'s `cover_path`.
    pub path: Option<u32>,
    /// Triangles of producer entries' clusters under path rays.
    pub producer_path: Option<u32>,
    /// Strand entries' segments under path rays.
    pub strand_path: Option<u32>,
    /// Triangles under shadow rays: an any-hit that decides by `cover_shadow`.
    pub shadow: Option<u32>,
    /// Triangles under layer rays: an any-hit that keeps particles and ignores the rest, since a
    /// record of zeros would commit them.
    pub layers: Option<u32>,
    /// Medium proxies under shadow rays: a procedural group of `box_medium` and `proxy_shadow`.
    pub medium_shadow: Option<u32>,
    /// Medium proxies under layer rays.
    pub medium_layers: Option<u32>,
    /// Sphere and rect light proxies under layer rays.
    pub light_layers: Option<u32>,
    /// The closest-hit a walk's step runs on a triangle it met, in place of the triangle's own,
    /// to read its geometry.
    pub walk_meet: Option<u32>,
    /// The closest-hit a plane stack's trace runs on a dielectric triangle it met, in place of the
    /// triangle's own, to read its cluster.
    pub stack_meet: Option<u32>,
    /// Dielectric and presence layer triangles of copies' clusters under path rays: `path`'s
    /// any-hit with the closest-hit that resolves a layer.
    pub layer_path: Option<u32>,
    /// Dielectric and presence layer triangles of producer entries' clusters under path rays.
    pub producer_layer_path: Option<u32>,
}

impl HitGroups {
    /// The groups of each pinned kind: one where no surface links, else one for any id and one
    /// per id from 0 on.
    pub fn pins(&self) -> u32 {
        if self.surfaces == 0 {
            1
        } else {
            self.surfaces + 2
        }
    }

    /// In `sbt.slang`'s handle table order: the KIND_* groups held once, then each pinned
    /// kind's pins.
    fn slots(&self) -> Vec<Option<u32>> {
        let fixed: [Option<u32>; FIXED_KINDS] = [
            self.layers,
            self.medium_shadow,
            self.medium_layers,
            self.light_layers,
            self.walk_meet,
            self.stack_meet,
        ];
        let pinned: [Option<u32>; PINNED_KINDS] = [
            self.path,
            self.producer_path,
            self.strand_path,
            self.shadow,
            self.layer_path,
            self.producer_layer_path,
        ];
        let pins = self.pins();
        fixed
            .into_iter()
            .chain(
                pinned
                    .into_iter()
                    .flat_map(|first| (0..pins).map(move |p| first.map(|g| g + p))),
            )
            .collect()
    }
}

/// Where a scene's runs start this frame (`SceneRoot::run_*`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RunLayout {
    pub(crate) models: u32,
    pub(crate) sets: u32,
    pub(crate) producers: u32,
    pub(crate) pages: u32,
    pub(crate) count: u32,
}

impl RunLayout {
    /// The runs of `models` section materials of the mounts, `sets` material set entries,
    /// `producers` producer runs and `pages` page material set entries.
    pub(crate) fn new(models: u32, sets: u32, producers: u32, pages: u32) -> Self {
        let models_at = FIXED_RUNS;
        let sets_at = models_at.saturating_add(models);
        let producers_at = sets_at.saturating_add(sets);
        let pages_at = producers_at.saturating_add(producers);
        Self {
            models: models_at,
            sets: sets_at,
            producers: producers_at,
            pages: pages_at,
            count: pages_at.saturating_add(pages),
        }
    }
}

/// A scene's hit tables, one per pipeline's handles that a frame asked for.
#[derive(Default)]
pub(crate) struct HitTables {
    tables: Vec<Table>,
}

struct Table {
    /// The handles of the groups `HitGroups` names, in `sbt.slang`'s order, zeros for none.
    handles: Vec<u8>,
    stride: u64,
    handle_table: Allocation<u8>,
    records: Allocation<u8>,
    /// Records `records` holds room for.
    room: u64,
    /// The frame that last asked for it.
    used: u64,
}

/// Frames a table nobody asks for is kept.
const KEEP_FRAMES: u64 = 8;

impl HitTables {
    /// The table of `pipeline`'s `groups` over a scene laid out as `layout` with
    /// `producer_entries` producer entries, whose root is `root`: written whole by `list`'s next
    /// work, after the traces before it and before those after it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn table(
        &mut self,
        device: &Device,
        list: &mut CommandList<'_>,
        roots: &dyn Fn(&HitTableRoot) -> Result<GpuPtr<HitTableRoot>>,
        kernels: [&ComputePipeline; 2],
        root: GpuPtr<SceneRoot>,
        layout: RunLayout,
        producer_entries: u32,
        pipeline: &RayPipeline,
        groups: &HitGroups,
        frame: u64,
    ) -> Result<HitTable> {
        // An instance's contribution, 24 bits, names every run's records or the table is none.
        let limit = (1u32 << 24) / RAY_TYPES;
        if layout.count > limit {
            return Err(Error::Capacity {
                what: "hit table runs",
                capacity: limit,
            });
        }
        let limits = &device.caps().limits;
        let handle = u64::from(limits.ray_handle_bytes);
        let data = HIT_DATA_BYTES;
        let stride = pipeline
            .hit_stride()
            .max(handle + data)
            .next_multiple_of(u64::from(limits.ray_handle_align));
        let slots = groups.slots();
        let mut handles = vec![0u8; slots.len() * handle as usize];
        for (k, g) in slots.iter().enumerate() {
            if let Some(g) = *g {
                handles[k * handle as usize..][..handle as usize]
                    .copy_from_slice(pipeline.hit_handle(g));
            }
        }
        self.tables.retain(|t| t.used + KEEP_FRAMES >= frame);
        let at = match self
            .tables
            .iter()
            .position(|t| t.handles == handles && t.stride == stride)
        {
            Some(at) => at,
            None => {
                let mut handle_table = device.gpu_malloc::<u8>(
                    MemoryKind::Upload,
                    handles.len() as u64,
                    "hit group handles",
                )?;
                handle_table
                    .as_mut_slice()
                    .expect("upload is mapped")
                    .copy_from_slice(&handles);
                self.tables.push(Table {
                    handles,
                    stride,
                    handle_table,
                    records: device.gpu_malloc::<u8>(
                        MemoryKind::DeviceLocal,
                        stride,
                        "hit table",
                    )?,
                    room: 1,
                    used: frame,
                });
                self.tables.len() - 1
            }
        };
        let t = &mut self.tables[at];
        t.used = frame;
        let records = u64::from(layout.count) * u64::from(RAY_TYPES);
        if records > t.room {
            let room = records.max(2 * t.room);
            t.records =
                device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, room * stride, "hit table")?;
            t.room = room;
        }
        let r = roots(&HitTableRoot {
            scene: root,
            records: t.records.gpu().cast(),
            handles: t.handle_table.gpu().cast(),
            stride_words: (stride / 4) as u32,
            handle_words: (handle / 4) as u32,
            surfaces: groups.surfaces,
            pins: groups.pins(),
            _pad: 0,
        })?;
        list.barrier(Stages::RAY, Stages::COMPUTE, Hazards::empty());
        list.dispatch(kernels[0], r, [layout.count.div_ceil(64).max(1), 1, 1]);
        list.dispatch(kernels[1], r, [producer_entries.max(1), 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::RAY, Hazards::empty());
        Ok(HitTable {
            address: t.records.gpu().address(),
            stride,
            count: records,
        })
    }
}
