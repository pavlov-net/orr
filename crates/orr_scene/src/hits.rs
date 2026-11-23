//! Scenes' hit tables: for one ray tracing pipeline, a hit record per ray type for each run of a
//! scene's material rows (`runs.slang`), which `sbt.slang` writes from the scene's runs. A table
//! is written where it changed: the runs logged since it last wrote, or all of them when it is
//! new, has grown, has new handles or a material's hit records changed.

use orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, HitTable, MemoryKind,
    RayPipeline, Stages,
};

use crate::memory::UploadWindow;
use crate::shaders::consts::{
    FIXED_KINDS, HIT_DATA_BYTES, KIND_EMISSIVE_PATH, KIND_FRONT, KIND_LAYER_PATH,
    KIND_LAYER_SHADOW, KIND_LIGHT_FRONT, KIND_MEDIUM_SHADOW, KIND_PATH,
    KIND_PRODUCER_EMISSIVE_PATH, KIND_PRODUCER_LAYER_PATH, KIND_PRODUCER_LAYER_SHADOW,
    KIND_PRODUCER_PATH, KIND_PRODUCER_RICH_PATH, KIND_RICH_PATH, KIND_SHADOW, KIND_STRAND_PATH,
    PINNED_KINDS, RAY_TYPES, RUNS_PER_GROUP,
};
use crate::shaders::sbt::HitTableRoot;
use crate::{Result, SceneRoot};

/// The hit groups of a ray tracing pipeline that a scene's hit records name, by what each
/// record serves: a group index of the pipeline, or `None` for records of zeros. A record of
/// zeros runs no shader and confirms every candidate a trace finds in it.
///
/// The path and shadow groups are pinned: each is the first of [`Self::pins`] groups, one after
/// another, for the materials of one linked surface id (`scene_may.slang`'s `hit_surface`). The
/// first serves any id, and is what a material whose back shows another id takes: a pipeline
/// gives it shaders where a linked type declares its other side, which alone lets such a
/// material be. Where surfaces link, the pin of id `k` follows at `k + 1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HitGroups {
    /// The linked surface types the pinned groups serve, ids 1 to `surfaces`.
    pub surfaces: u32,
    /// Triangles of copies' clusters under path rays: a closest-hit that shades the hit and an
    /// any-hit that decides by `orr_scene.candidates`'s `cover_path`.
    pub path: Option<u32>,
    /// As [`Self::path`], for a material that may have a coat, fuzz, thin film or subsurface;
    /// `None` serves those by [`Self::path`].
    pub rich_path: Option<u32>,
    /// As [`Self::path`], for clusters that trace as a dielectric or a presence layer; `None`
    /// serves those by [`Self::path`].
    pub layer_path: Option<u32>,
    /// As [`Self::path`], for a material that may emit and has none of those; `None` serves
    /// those by [`Self::path`].
    pub emissive_path: Option<u32>,
    /// Triangles of producer entries' clusters under path rays.
    pub producer_path: Option<u32>,
    /// As [`Self::rich_path`] and [`Self::layer_path`], of producer entries' clusters, served
    /// by [`Self::producer_path`] where `None`.
    pub producer_rich_path: Option<u32>,
    pub producer_layer_path: Option<u32>,
    pub producer_emissive_path: Option<u32>,
    /// Strand entries' segments under path rays.
    pub strand_path: Option<u32>,
    /// Triangles under shadow rays: an any-hit that decides by `cover_shadow`.
    pub shadow: Option<u32>,
    /// As [`Self::shadow`], for copies' and producer entries' clusters that trace as a
    /// dielectric, with the closest-hit of a manifold walk's face ray on them, which the face
    /// ray's hit object runs from this record; `None` serves those by [`Self::shadow`].
    pub layer_shadow: Option<u32>,
    pub producer_layer_shadow: Option<u32>,
    /// Particles under path rays: an any-hit that keeps them as what lies in front of the ray's
    /// hit and ignores them, since a record of zeros would commit them.
    pub front: Option<u32>,
    /// Sphere and rect light proxies under path rays.
    pub light_front: Option<u32>,
    /// Medium proxies under shadow rays: a procedural group of `box_medium` and `proxy_shadow`.
    pub medium_shadow: Option<u32>,
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
        let mut fixed = [None; FIXED_KINDS as usize];
        for (kind, group) in [
            (KIND_MEDIUM_SHADOW, self.medium_shadow),
            (KIND_FRONT, self.front),
            (KIND_LIGHT_FRONT, self.light_front),
        ] {
            fixed[kind as usize] = group;
        }
        let mut pinned = [None; PINNED_KINDS as usize];
        for (kind, group) in [
            (KIND_PATH, self.path),
            (KIND_RICH_PATH, self.rich_path.or(self.path)),
            (KIND_LAYER_PATH, self.layer_path.or(self.path)),
            (KIND_EMISSIVE_PATH, self.emissive_path.or(self.path)),
            (KIND_PRODUCER_PATH, self.producer_path),
            (
                KIND_PRODUCER_RICH_PATH,
                self.producer_rich_path.or(self.producer_path),
            ),
            (
                KIND_PRODUCER_LAYER_PATH,
                self.producer_layer_path.or(self.producer_path),
            ),
            (
                KIND_PRODUCER_EMISSIVE_PATH,
                self.producer_emissive_path.or(self.producer_path),
            ),
            (KIND_STRAND_PATH, self.strand_path),
            (KIND_SHADOW, self.shadow),
            (KIND_LAYER_SHADOW, self.layer_shadow.or(self.shadow)),
            (
                KIND_PRODUCER_LAYER_SHADOW,
                self.producer_layer_shadow.or(self.shadow),
            ),
        ] {
            pinned[kind as usize] = group;
        }
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

/// A scene's hit tables, one per pipeline's handles that a frame asked for, and the runs written
/// since the oldest of them last wrote.
#[derive(Default)]
pub(crate) struct HitTables {
    tables: Vec<Table>,
    /// Ranges of runs written, from position `logged` on: each table writes those past its own
    /// position.
    log: Vec<(u32, u32)>,
    logged: u64,
}

struct Table {
    /// The handles of the groups `HitGroups` names, in `sbt.slang`'s order, zeros for none.
    handles: Vec<u8>,
    stride: u64,
    handle_table: Allocation<u8>,
    records: Allocation<u8>,
    /// Runs `records` holds room for.
    room: u32,
    /// The log position up to which it holds the runs.
    applied: u64,
    /// The material table's hit epoch it holds the runs of.
    epoch: u64,
    /// The frame that last asked for it.
    used: u64,
}

/// Frames a table nobody asks for is kept.
const KEEP_FRAMES: u64 = 8;

impl HitTables {
    /// Logs that runs `first..first + count` hold other records from the frame's update on.
    pub(crate) fn note(&mut self, first: u32, count: u32) {
        if count > 0 && !self.tables.is_empty() {
            self.log.push((first, count));
        }
    }

    /// Drops the tables no frame asked for in [`KEEP_FRAMES`] by `frame`, and the log every
    /// table left holds.
    pub(crate) fn retire(&mut self, frame: u64) {
        self.tables.retain(|t| t.used + KEEP_FRAMES >= frame);
        let end = self.logged + self.log.len() as u64;
        let held = self.tables.iter().map(|t| t.applied).min().unwrap_or(end);
        self.log.drain(..(held - self.logged) as usize);
        self.logged = held;
    }

    /// The table of `pipeline`'s `groups` over a scene of `runs` runs (below what an instance's
    /// 24-bit contribution names, as the scene's run table keeps them), whose root is `root`, at
    /// material hit epoch `epoch`: what has changed in it is written by `list`'s next work, after
    /// the traces before it and before those after it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn table(
        &mut self,
        device: &Device,
        list: &mut CommandList<'_>,
        window: &UploadWindow,
        kernel: &ComputePipeline,
        root: GpuPtr<SceneRoot>,
        runs: u32,
        epoch: u64,
        pipeline: &RayPipeline,
        groups: &HitGroups,
        frame: u64,
    ) -> Result<HitTable> {
        let limits = &device.caps().limits;
        let handle = u64::from(limits.ray_handle_bytes);
        let data = u64::from(HIT_DATA_BYTES);
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
        self.retire(frame);
        let end = self.logged + self.log.len() as u64;
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
                    room: 0,
                    applied: end,
                    epoch,
                    used: frame,
                });
                self.tables.len() - 1
            }
        };
        let t = &mut self.tables[at];
        t.used = frame;
        let mut whole = t.room == 0 || t.epoch != epoch;
        if runs > t.room {
            let room = runs.max(2 * t.room);
            t.records = device.gpu_malloc::<u8>(
                MemoryKind::DeviceLocal,
                u64::from(room) * u64::from(RAY_TYPES) * stride,
                "hit table",
            )?;
            t.room = room;
            whole = true;
        }
        let mut ranges: Vec<(u32, u32)> = if whole {
            vec![(0, runs)]
        } else {
            self.log[(t.applied - self.logged) as usize..].to_vec()
        };
        t.applied = end;
        t.epoch = epoch;
        let table = HitTable {
            address: t.records.gpu().address(),
            stride,
            count: u64::from(runs) * u64::from(RAY_TYPES),
        };
        let groups_of = split(&mut ranges);
        if groups_of.is_empty() {
            return Ok(table);
        }
        let (at_ranges, out) = window.slice::<[u32; 2]>(device, groups_of.len() as u32)?;
        for (o, &(first, count)) in out.iter_mut().zip(&groups_of) {
            *o = [first, count];
        }
        let r = window.put(
            device,
            &HitTableRoot {
                scene: root,
                records: t.records.gpu().cast(),
                handles: t.handle_table.gpu().cast(),
                ranges: at_ranges.cast(),
                stride_words: (stride / 4) as u32,
                handle_words: (handle / 4) as u32,
                surfaces: groups.surfaces,
                pins: groups.pins(),
            },
        )?;
        list.barrier(Stages::RAY, Stages::COMPUTE, Hazards::empty());
        list.dispatch(kernel, r, [groups_of.len() as u32, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::RAY, Hazards::empty());
        Ok(table)
    }
}

/// `ranges` merged where they overlap or touch, cut into the ranges of one workgroup each.
fn split(ranges: &mut [(u32, u32)]) -> Vec<(u32, u32)> {
    ranges.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for &(first, count) in ranges.iter().filter(|r| r.1 > 0) {
        match merged.last_mut() {
            Some((f, n)) if first <= *f + *n => *n = (*n).max(first + count - *f),
            _ => merged.push((first, count)),
        }
    }
    merged
        .into_iter()
        .flat_map(|(first, count)| {
            (0..count.div_ceil(RUNS_PER_GROUP)).map(move |k| {
                let at = k * RUNS_PER_GROUP;
                (first + at, (count - at).min(RUNS_PER_GROUP))
            })
        })
        .collect()
}
