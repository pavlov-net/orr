//! The pipeline path tracer (`path/launch.slang`): one ray tracing pipeline whose raygens run a
//! source's paths from their start to their end, a launch per slice of them, through the scene's
//! hit records; and the memory a view's paths keep in it.

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Instant;

use orr_scene::HitGroups;
use orr_scene::orr_gpu::{
    Allocation, CommandList, CompileThreads, Device, GpuPtr, Hazards, HitGroup, HitTable,
    MemoryKind, RayPipeline, RayPipelineDesc, RayPipelineFlags, ShaderRecord, ShaderStage,
    SpecData, Stages,
};

use crate::shaders::path::pipeline;
use crate::shaders::types::{PathRoot, PathStages};
use crate::staged::{RECORD_WORDS, UPDATE_RECORD_WORDS};
use crate::{Error, KernelSet, module};

/// `path/launch.slang`'s SLOT_WORDS: the words of a path's slot.
const SLOT_WORDS: u64 = 19;

/// Paths of a source one launch runs at most, which bounds the memory a view's paths keep: a
/// source with more runs them in slices.
const MAX_SLOTS: u32 = 1 << 20;

/// The pipeline's stages, its groups' members by index: the raygens, the miss shader, the hit
/// groups' shaders and the callables in `path/launch.slang`'s CALL_* order.
const STAGES: [&str; 15] = [
    "path_view",
    "path_update",
    "miss_path",
    "resolve",
    "cover_path_hit",
    "cover_shadow_hit",
    "box_medium_hit",
    "proxy_shadow_hit",
    "start_view",
    "start_update",
    "emission",
    "direct",
    "scatter",
    "finish_view",
    "finish_update",
];
const FIRST_CALLABLE: u32 = 8;

/// A source's raygen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Raygen {
    View = 0,
    Update = 1,
}

/// The ray tracing pipeline of the path tracer, and the groups a scene's hit records name.
pub(crate) struct PathPipeline {
    pipeline: RayPipeline,
    groups: HitGroups,
    /// The mixes of what frames asked of it that it does not render, each logged once: views
    /// of different scenes or settings ask for different mixes in the same frames.
    unlaunched: Mutex<HashSet<Vec<&'static str>>>,
}

impl PathPipeline {
    /// Compiles the pipeline from `kernels` on `threads`.
    pub(crate) fn new(
        device: &Device,
        kernels: &KernelSet,
        threads: CompileThreads<'_>,
    ) -> Result<Self, Error> {
        let m = module(kernels, &pipeline::MODULE)?;
        let stages = STAGES
            .iter()
            .map(|e| ShaderStage::new(m, e, SpecData::default()))
            .collect::<Result<Vec<_>, _>>()?;
        let caps = device.caps();
        let mut flags = RayPipelineFlags::CLUSTERS;
        if caps.omm {
            flags |= RayPipelineFlags::OPACITY_MICROMAPS;
        }
        if caps.lss {
            flags |= RayPipelineFlags::SWEPT_SPHERES;
        }
        let callables: Vec<ShaderRecord<'_>> = (FIRST_CALLABLE..STAGES.len() as u32)
            .map(ShaderRecord::new)
            .collect();
        let desc = RayPipelineDesc {
            stages: &stages,
            raygens: &[ShaderRecord::new(0), ShaderRecord::new(1)],
            misses: &[ShaderRecord::new(2)],
            hit_groups: &[
                HitGroup {
                    closest_hit: Some(3),
                    any_hit: Some(4),
                    intersection: None,
                },
                HitGroup {
                    any_hit: Some(5),
                    ..HitGroup::default()
                },
                HitGroup {
                    closest_hit: None,
                    any_hit: Some(7),
                    intersection: Some(6),
                },
            ],
            callables: &callables,
            max_recursion: 1,
            flags,
            threads,
        };
        let started = Instant::now();
        let pipeline = device.create_ray_pipeline(&desc, "path pipeline")?;
        log::debug!(
            "path pipeline made in {:.1} s",
            started.elapsed().as_secs_f64()
        );
        Ok(Self {
            pipeline,
            groups: HitGroups {
                path: Some(0),
                shadow: Some(1),
                medium_shadow: Some(2),
                ..HitGroups::default()
            },
            unlaunched: Mutex::new(HashSet::new()),
        })
    }

    /// Logs what a frame asks of the pipeline that it does not render yet, once for each mix.
    pub(crate) fn note_unlaunched(&self, what: Vec<&'static str>) {
        if what.is_empty() {
            return;
        }
        let mut logged = self.unlaunched.lock().unwrap();
        if !logged.contains(&what) {
            log::warn!(
                "the path tracing pipeline renders this scene without: {}",
                what.join(", ")
            );
            logged.insert(what);
        }
    }

    pub(crate) fn hit_pipeline(&self) -> (&RayPipeline, &HitGroups) {
        (&self.pipeline, &self.groups)
    }

    /// Records `items` paths of `root`'s source through `raygen`, over `hits`, in slices of
    /// `memory`'s slots, each after the one before it. Orders nothing after them.
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        hits: &HitTable,
        raygen: Raygen,
        root: &PathRoot,
        memory: &SourceMemory,
        items: u32,
    ) {
        let mut root = PathRoot {
            stages: memory.header.gpu(),
            ..*root
        };
        let mut base = 0;
        while base < items {
            if base > 0 {
                list.barrier(Stages::RAY, Stages::RAY, Hazards::empty());
            }
            root.slice_base = base;
            root.slice_count = (items - base).min(memory.slots);
            list.trace_rays(
                &self.pipeline,
                raygen as u32,
                hits,
                &root,
                [root.slice_count, 1, 1],
            );
            base += root.slice_count;
        }
    }
}

/// The slots of a slice of one source's paths in the pipeline and their records.
pub(crate) struct SourceMemory {
    slots: u32,
    header: Allocation<PathStages>,
    _words: Allocation<[u32; 4]>,
    _records: Allocation<[u32; 4]>,
}

impl SourceMemory {
    /// The slots of `items` paths, at most [`MAX_SLOTS`], with records of `record_words` words.
    fn new(device: &Device, items: u32, record_words: u64, name: &str) -> Result<Self, Error> {
        let slots = slots(items);
        let n = u64::from(slots);
        let words = |count: u64, what: &str| {
            device.gpu_malloc::<[u32; 4]>(MemoryKind::DeviceLocal, count, &format!("{name} {what}"))
        };
        let state = words(SLOT_WORDS * n, "paths")?;
        let records = words(record_words * n, "path records")?;
        let mut header = device.gpu_malloc::<PathStages>(MemoryKind::Upload, 1, name)?;
        header.as_mut_slice().expect("mapped")[0] = PathStages {
            capacity: slots,
            rounds: 0,
            queues: GpuPtr::NULL,
            lists: GpuPtr::NULL,
            words: state.gpu(),
            shadows: GpuPtr::NULL,
            records: records.gpu(),
            update_records: GpuPtr::NULL,
            update_first: slots,
            _pad0: 0,
        };
        Ok(Self {
            slots,
            header,
            _words: state,
            _records: records,
        })
    }
}

fn slots(items: u32) -> u32 {
    items.clamp(1, MAX_SLOTS)
}

/// What a view's paths keep in the pipeline: its own paths' slots and its update paths'.
pub(crate) struct LaunchMemory {
    pub(crate) view: SourceMemory,
    pub(crate) update: SourceMemory,
}

impl LaunchMemory {
    /// Slots for a slice of a view of `pixels` pixels and one of its `update` update paths.
    pub(crate) fn new(device: &Device, pixels: u32, update: u32) -> Result<Self, Error> {
        let memory = Self {
            view: SourceMemory::new(device, pixels, RECORD_WORDS, "launched view")?,
            update: SourceMemory::new(device, update, UPDATE_RECORD_WORDS, "launched update")?,
        };
        log::debug!(
            "launched paths: {} view and {} update slots of {} bytes",
            memory.view.slots,
            memory.update.slots,
            16 * SLOT_WORDS
        );
        Ok(memory)
    }

    /// Whether it holds the slots [`Self::new`] makes for `pixels` and `update`.
    pub(crate) fn fits(&self, pixels: u32, update: u32) -> bool {
        self.view.slots == slots(pixels) && self.update.slots == slots(update)
    }
}
