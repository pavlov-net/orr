//! The pipeline path tracer (`path/launch.slang`): one ray tracing pipeline whose raygens run a
//! source's paths from their start to their end, a launch per slice of them, through the scene's
//! hit records; and the memory a view's paths keep in it.

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Instant;

use orr_scene::orr_gpu::{
    Allocation, CommandList, CompileThreads, Device, GpuPtr, Hazards, HitGroup, HitTable,
    MemoryKind, RayPipeline, RayPipelineDesc, RayPipelineFlags, ShaderRecord, ShaderStage,
    SpecData, Stages,
};
use orr_scene::{HitGroups, SceneFeatures};

use crate::shaders::path::pipeline::{self, Spec};
use crate::shaders::types::{PathRoot, PathStages};
use crate::staged::{
    FEATURE_INTERIORS, FEATURE_WALK, MAY_ALL, MAY_RICH, RECORD_WORDS, UPDATE_RECORD_WORDS,
};
use crate::{Error, KernelSet, module};

/// `path/launch.slang`'s SLOT_WORDS, WALK_WORDS and INTERIORS_WORDS: the words of a path's slot,
/// and those of the parts the paths that walk and that track interiors add to it.
const SLOT_WORDS: u64 = 28;
const WALK_WORDS: u64 = 15;
const INTERIORS_WORDS: u64 = 7;

/// The words of a slot of paths whose root has `features`, of a view's (`view`) or not, as
/// `path/launch.slang`'s `slot_stride` lays it out.
fn slot_words(features: u32, view: bool) -> u64 {
    let walk = view && features & (FEATURE_WALK | FEATURE_INTERIORS) != 0;
    let interiors = features & FEATURE_INTERIORS != 0;
    SLOT_WORDS + if walk { WALK_WORDS } else { 0 } + if interiors { INTERIORS_WORDS } else { 0 }
}

/// Paths of a source one launch runs at most, which bounds the memory a view's paths keep: a
/// source with more runs them in slices.
const MAX_SLOTS: u32 = 1 << 20;

/// The stages the pipeline holds one of, first, by index: the raygens, the miss shader, the
/// medium proxies' intersection and any-hit shaders, the walk meet's closest-hit, and what layer
/// rays keep of triangles, medium proxies and light proxies.
const FIXED: [&str; 10] = [
    "path_view",
    "path_update",
    "miss_path",
    "box_medium_hit",
    "proxy_shadow_hit",
    "walk_meet",
    "cover_layers_hit",
    "cover_layers_medium_hit",
    "box_light_hit",
    "cover_layers_light_hit",
];
const MISS: u32 = 2;
const MEDIUM_INTERSECTION: u32 = 3;
const MEDIUM_ANY_HIT: u32 = 4;
const WALK_MEET: u32 = 5;
const LAYERS_ANY_HIT: u32 = 6;
const MEDIUM_LAYERS_ANY_HIT: u32 = 7;
const LIGHT_INTERSECTION: u32 = 8;
const LIGHT_LAYERS_ANY_HIT: u32 = 9;

/// What the pipeline holds per pin after those (`HitGroups`): the closest-hit shaders of a copy's
/// clusters, of a producer entry's clusters and of strands, and the any-hit shaders under path and
/// shadow rays, with what each may meet of the scene's geometry.
const PINNED: [(&str, Geometry); 5] = [
    ("resolve", Geometry::Copies),
    ("resolve", Geometry::Producers),
    ("resolve_strand", Geometry::Any),
    ("cover_path_hit", Geometry::Any),
    ("cover_shadow_hit", Geometry::Any),
];
const RESOLVE_COPIES: usize = 0;
const RESOLVE_PRODUCERS: usize = 1;
const RESOLVE_STRANDS: usize = 2;
const COVER_PATH: usize = 3;
const COVER_SHADOW: usize = 4;

/// Then the callables, in `path/launch.slang`'s CALL_* order, each with the lobes of the BSDF its
/// vertices may have: next-event estimation and the scatter are made for a plain surface's and a
/// rich one's. A walk's ends are made rich, as a walk starts only from a vertex with a subsurface
/// weight. What lies along a ray and a scatter in a medium shade no surface.
const CALLABLES: [(&str, bool); 19] = [
    ("start_view", true),
    ("start_update", true),
    ("emission", true),
    ("direct", false),
    ("direct", true),
    ("scatter", false),
    ("scatter", true),
    ("finish_view", true),
    ("finish_update", true),
    ("walk_end", true),
    ("direct_exit", true),
    ("scatter_exit", true),
    ("direct_leave", true),
    ("scatter_leave", true),
    ("segment", false),
    ("layers", false),
    ("march_proxy", false),
    ("direct_medium", false),
    ("scatter_medium", false),
];

/// The geometry a hit shader runs on, which its stage is made for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Geometry {
    /// Copies' clusters, cooked or skinned.
    Copies,
    /// Producer entries' clusters, not strands.
    Producers,
    Any,
}

impl Geometry {
    /// `path_may2` of its stage.
    fn may2(self) -> u32 {
        let out = match self {
            Self::Copies => SceneFeatures::PRODUCERS.0 | SceneFeatures::STRANDS.0,
            Self::Producers => SceneFeatures::STRANDS.0,
            Self::Any => 0,
        };
        SceneFeatures(!out).surface_may()
    }
}

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
        let groups = HitGroups {
            surfaces: kernels.surfaces,
            ..HitGroups::default()
        };
        let pins = groups.pins();
        // `hit_surface` of the hit shaders of each pin: any id, then each id from 0 on.
        let pin_surface = |pin: u32| if pin == 0 { u32::MAX } else { pin - 1 };
        let mut specs: Vec<(&str, Spec)> = FIXED.iter().map(|&e| (e, Spec::default())).collect();
        for pin in 0..pins {
            for &(entry, geometry) in &PINNED {
                let spec = Spec {
                    path_may2: geometry.may2(),
                    hit_surface: pin_surface(pin),
                    ..Spec::default()
                };
                specs.push((entry, spec));
            }
        }
        let first_callable = specs.len() as u32;
        for &(entry, rich) in &CALLABLES {
            let may = if rich { MAY_ALL } else { MAY_ALL & !MAY_RICH };
            let spec = Spec {
                path_may: may as u32,
                ..Spec::default()
            };
            specs.push((entry, spec));
        }
        let bytes: Vec<_> = specs.iter().map(|(_, s)| s.bytes()).collect();
        let stages = specs
            .iter()
            .zip(&bytes)
            .map(|(&(entry, spec), bytes)| {
                let data = if spec == Spec::default() {
                    SpecData::default()
                } else {
                    SpecData {
                        entries: Spec::ENTRIES,
                        bytes,
                    }
                };
                ShaderStage::new(m, entry, data)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let caps = device.caps();
        let mut flags = RayPipelineFlags::CLUSTERS;
        if caps.omm {
            flags |= RayPipelineFlags::OPACITY_MICROMAPS;
        }
        if caps.lss {
            flags |= RayPipelineFlags::SWEPT_SPHERES;
        }
        let callables: Vec<ShaderRecord<'_>> = (first_callable..stages.len() as u32)
            .map(ShaderRecord::new)
            .collect();
        // Each pinned kind's groups one after another by pin, then the medium proxies'.
        let pinned = |pin: u32, k: usize| FIXED.len() as u32 + pin * PINNED.len() as u32 + k as u32;
        let mut hit_groups = Vec::new();
        for (resolve, any_hit) in [
            (Some(RESOLVE_COPIES), Some(COVER_PATH)),
            (Some(RESOLVE_PRODUCERS), Some(COVER_PATH)),
            (Some(RESOLVE_STRANDS), None),
            (None, Some(COVER_SHADOW)),
        ] {
            for pin in 0..pins {
                hit_groups.push(HitGroup {
                    closest_hit: resolve.map(|k| pinned(pin, k)),
                    any_hit: any_hit.map(|k| pinned(pin, k)),
                    intersection: None,
                });
            }
        }
        hit_groups.push(HitGroup {
            closest_hit: None,
            any_hit: Some(MEDIUM_ANY_HIT),
            intersection: Some(MEDIUM_INTERSECTION),
        });
        hit_groups.push(HitGroup {
            closest_hit: Some(WALK_MEET),
            ..HitGroup::default()
        });
        for (any_hit, intersection) in [
            (LAYERS_ANY_HIT, None),
            (MEDIUM_LAYERS_ANY_HIT, Some(MEDIUM_INTERSECTION)),
            (LIGHT_LAYERS_ANY_HIT, Some(LIGHT_INTERSECTION)),
        ] {
            hit_groups.push(HitGroup {
                closest_hit: None,
                any_hit: Some(any_hit),
                intersection,
            });
        }
        let desc = RayPipelineDesc {
            stages: &stages,
            raygens: &[ShaderRecord::new(0), ShaderRecord::new(1)],
            misses: &[ShaderRecord::new(MISS)],
            hit_groups: &hit_groups,
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
                producer_path: Some(pins),
                strand_path: Some(2 * pins),
                shadow: Some(3 * pins),
                medium_shadow: Some(4 * pins),
                walk_meet: Some(4 * pins + 1),
                layers: Some(4 * pins + 2),
                medium_layers: Some(4 * pins + 3),
                light_layers: Some(4 * pins + 4),
                ..groups
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
    slot_words: u64,
    header: Allocation<PathStages>,
    _words: Allocation<[u32; 4]>,
    _records: Allocation<[u32; 4]>,
}

impl SourceMemory {
    /// The slots of `items` paths, at most [`MAX_SLOTS`], of `slot_words` words, with records of
    /// `record_words` words.
    fn new(
        device: &Device,
        items: u32,
        slot_words: u64,
        record_words: u64,
        name: &str,
    ) -> Result<Self, Error> {
        let slots = slots(items);
        let n = u64::from(slots);
        let words = |count: u64, what: &str| {
            device.gpu_malloc::<[u32; 4]>(MemoryKind::DeviceLocal, count, &format!("{name} {what}"))
        };
        let state = words(slot_words * n, "paths")?;
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
            slot_words,
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
    /// Slots for a slice of a view of `pixels` pixels whose paths' root has `features`, and one
    /// of its `update` update paths, whose root has `update_features`.
    pub(crate) fn new(
        device: &Device,
        pixels: u32,
        features: u32,
        update: u32,
        update_features: u32,
    ) -> Result<Self, Error> {
        let memory = Self {
            view: SourceMemory::new(
                device,
                pixels,
                slot_words(features, true),
                RECORD_WORDS,
                "launched view",
            )?,
            update: SourceMemory::new(
                device,
                update,
                slot_words(update_features, false),
                UPDATE_RECORD_WORDS,
                "launched update",
            )?,
        };
        log::debug!(
            "launched paths: {} view slots of {} bytes and {} update slots of {} bytes",
            memory.view.slots,
            16 * memory.view.slot_words,
            memory.update.slots,
            16 * memory.update.slot_words
        );
        Ok(memory)
    }

    /// Whether it holds the slots [`Self::new`] makes for its arguments.
    pub(crate) fn fits(
        &self,
        pixels: u32,
        features: u32,
        update: u32,
        update_features: u32,
    ) -> bool {
        self.view.slots == slots(pixels)
            && self.view.slot_words == slot_words(features, true)
            && self.update.slots == slots(update)
            && self.update.slot_words == slot_words(update_features, false)
    }
}
