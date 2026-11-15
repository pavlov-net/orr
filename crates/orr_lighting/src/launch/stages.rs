//! The pipelines of the path tracer: one for each mask a source's paths may meet, its stages
//! linked by what the mask holds and each folded with it, its hit groups, and its creation.

use std::sync::Arc;
use std::time::Instant;

use orr_scene::orr_gpu::{
    CompileThreads, Device, HitGroup, RayPipeline, RayPipelineDesc, RayPipelineFlags, ShaderModule,
    ShaderRecord, ShaderStage, SpecData, SpecEntry,
};
use orr_scene::{HitGroups, Loading, SceneFeatures};

use crate::mask_sets::Members;
use crate::shaders::consts::{
    AT_EXIT, AT_LAYER, AT_LEAVE, AT_SCATTER, AT_SURFACE, CALL_BENT, CALL_CACHE_VERTEX,
    CALL_CROSS_FOLD, CALL_DEPOSIT, CALL_EMISSION, CALL_EMITTER_VIEW, CALL_FINISH_BACKGROUND,
    CALL_FINISH_RECORD, CALL_FINISH_VIEW, CALL_GATHER, CALL_GBUFFER, CALL_LAYERS, CALL_PROXY_MARCH,
    CALL_RESOLVE_GIVEN, CALL_SEGMENT, CALL_STACK_MISS, CALL_STACK_STEP, CALL_START_BACKGROUND,
    CALL_START_FIRST_VERTEX, CALL_START_PHOTON, CALL_START_UPDATE, CALL_START_VIEW, CALL_VERTEX,
    CALL_WALK_END, CALLABLES,
};
use crate::shaders::path::pipeline;
use crate::staged::{
    MAY_BACKGROUND, MAY_CAUSTICS, MAY_DEBUG, MAY_EMISSIVE, MAY_INTERIORS, MAY_LAYERS,
    MAY_LOCAL_LIGHTS, MAY_PARTICLES, MAY_PHOTON, MAY_PROXIES, MAY_RICH, MAY_STACK, MAY_UPDATE,
    MAY_VIEW, MAY_WALK,
};
use crate::{Error, KernelSet, module};

/// The pipeline's kernel files (`path/pipeline/*.slang`), which hold its entries by family.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Raygen,
    Hits,
    Vertex,
    Sources,
}

impl Kernel {
    const ALL: [Self; 4] = [Self::Raygen, Self::Hits, Self::Vertex, Self::Sources];

    /// orr_lighting's own module of the file.
    fn own(self) -> &'static ShaderModule {
        match self {
            Self::Raygen => &pipeline::raygen::MODULE,
            Self::Hits => &pipeline::hits::MODULE,
            Self::Vertex => &pipeline::vertex::MODULE,
            Self::Sources => &pipeline::sources::MODULE,
        }
    }

    /// The specialization of a stage of the file with `c`, its other constants at the file's
    /// defaults: a constant the file reads keeps the default its declaration gives, which a file
    /// that does not read it reflects as 0.
    fn spec(self, c: Constants) -> ([u8; 32], &'static [SpecEntry]) {
        macro_rules! spec {
            ($file:ident) => {{
                use pipeline::$file::Spec;
                let d = Spec::default();
                let s = Spec {
                    path_may: c.may as u32,
                    path_may2: (c.may >> 32) as u32,
                    hit_surface: c.hit_surface.unwrap_or(d.hit_surface),
                    ..d
                };
                (s.bytes(), Spec::ENTRIES)
            }};
        }
        match self {
            Self::Raygen => spec!(raygen),
            Self::Hits => spec!(hits),
            Self::Vertex => spec!(vertex),
            Self::Sources => spec!(sources),
        }
    }
}

/// The constants a stage of the pipeline is made with: what it may meet (`path_may` in the low
/// word, `path_may2` in the high one) and, for a hit shader made for one linked surface id, that
/// id.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Constants {
    may: u64,
    hit_surface: Option<u32>,
}

/// What a pipeline is made for: what its paths may meet.
#[derive(Clone, Copy)]
struct Mask {
    may: u64,
}

impl Mask {
    /// Whether its paths may meet any of `bits`.
    fn any(self, bits: u64) -> bool {
        self.may & bits != 0
    }

    /// Whether they may meet all of `bits`.
    fn all(self, bits: u64) -> bool {
        self.may & bits == bits
    }

    /// Whether they may meet what `feature` decides of `path_may2`.
    fn may2(self, feature: SceneFeatures) -> bool {
        self.any(may2(feature))
    }
}

/// The bit of `path_may2`, in the high word, that `feature` decides.
fn may2(feature: SceneFeatures) -> u64 {
    u64::from(!SceneFeatures(!feature.0).surface_may()) << 32
}

/// The `path_may` bits of every source.
const SOURCES: u64 = MAY_VIEW | MAY_BACKGROUND | MAY_UPDATE | MAY_PHOTON;

/// A callable of the table: its index (`path/launch/steps.slang`'s CALL_*), its kernel file and
/// entry, the bits it is made without beyond its pipeline's, and whether a pipeline made for a
/// mask links it. A pipeline that does not link a callable holds a record of zeros at its index,
/// so each call site of it must lie behind a fold of the bits its row tests or a value that only
/// a scene with them makes.
#[derive(Clone, Copy)]
struct Callable {
    index: u32,
    kernel: Kernel,
    entry: &'static str,
    without: u64,
    linked: fn(Mask) -> bool,
}

const fn row(
    index: u32,
    kernel: Kernel,
    entry: &'static str,
    without: u64,
    linked: fn(Mask) -> bool,
) -> Callable {
    Callable {
        index,
        kernel,
        entry,
        without,
        linked,
    }
}

use Kernel::{Sources as S, Vertex as V};

/// The callables, one row each. A record's callables are made for their own source; what shades
/// no surface is made without the rich lobes and walks.
#[rustfmt::skip]
const CALLED: [Callable; CALLABLES as usize] = [
    row(CALL_START_VIEW, S, "start_view", 0, |m| m.any(MAY_VIEW)),
    row(CALL_START_UPDATE, S, "start_update", 0, |m| m.any(MAY_UPDATE)),
    row(CALL_START_PHOTON, S, "start_photon", 0, |m| m.any(MAY_PHOTON)),
    row(CALL_START_BACKGROUND, S, "start_background", 0, |m| m.any(MAY_BACKGROUND)),
    row(CALL_RESOLVE_GIVEN, S, "resolve_given", 0, |m| m.any(MAY_PHOTON)),
    row(CALL_EMISSION, V, "emission", 0, |m| m.any(MAY_EMISSIVE)),
    row(CALL_VERTEX + AT_SURFACE, V, "vertex", 0, |_| true),
    row(CALL_VERTEX + AT_EXIT, V, "vertex_exit", 0, |m| m.any(MAY_WALK)),
    row(CALL_VERTEX + AT_LEAVE, V, "vertex_leave", 0, |m| m.any(MAY_INTERIORS)),
    row(CALL_VERTEX + AT_SCATTER, V, "vertex_medium", MAY_RICH, |m| m.any(MAY_INTERIORS)),
    row(CALL_VERTEX + AT_LAYER, V, "vertex_layer", 0, |m| m.any(MAY_LAYERS)),
    row(CALL_GATHER, V, "gather", MAY_RICH, |m| m.any(MAY_CAUSTICS)),
    row(CALL_DEPOSIT, V, "deposit", SOURCES & !MAY_PHOTON, |m| m.any(MAY_PHOTON)),
    row(CALL_CACHE_VERTEX, V, "cache_vertex", SOURCES & !MAY_UPDATE, |m| m.any(MAY_UPDATE)),
    row(CALL_BENT, V, "bent", 0, |m| m.any(MAY_STACK)),
    row(CALL_FINISH_VIEW, S, "finish_view", 0, |m| m.any(MAY_VIEW)),
    row(CALL_FINISH_RECORD, S, "finish_record", 0, |m| m.any(MAY_UPDATE | MAY_PHOTON)),
    row(CALL_FINISH_BACKGROUND, S, "finish_background", 0, |m| m.any(MAY_BACKGROUND)),
    row(CALL_WALK_END, V, "walk_end", 0, |m| m.any(MAY_INTERIORS)),
    row(CALL_SEGMENT, S, "segment", MAY_RICH, |m| m.any(MAY_INTERIORS)),
    row(CALL_LAYERS, S, "layers", MAY_RICH, |m| m.any(MAY_PARTICLES | MAY_LOCAL_LIGHTS)),
    row(CALL_PROXY_MARCH, S, "march_proxy", MAY_RICH, |m| m.any(MAY_PROXIES)),
    row(CALL_GBUFFER, S, "gbuffer_start", 0, |m| m.any(MAY_VIEW)),
    row(CALL_STACK_STEP, V, "stack_step", MAY_RICH, |m| m.any(MAY_STACK)),
    row(CALL_STACK_MISS, V, "stack_miss", MAY_RICH, |m| m.any(MAY_STACK)),
    row(CALL_EMITTER_VIEW, V, "emitter_view", 0, |m| m.any(MAY_EMISSIVE) && m.may2(SceneFeatures::COATED_EMITTERS)),
    row(CALL_START_FIRST_VERTEX, S, "start_first_vertex", 0, |m| m.all(MAY_VIEW | MAY_DEBUG)),
    row(CALL_CROSS_FOLD, S, "cross_fold", MAY_RICH, |m| m.any(MAY_INTERIORS) && m.any(MAY_VIEW | MAY_UPDATE)),
];

/// [`CALLED`] in the order of the table, checked to name each index once.
fn callables() -> [Option<Callable>; CALLABLES as usize] {
    let mut table = [None; CALLABLES as usize];
    for c in CALLED {
        let at = &mut table[c.index as usize];
        assert!(at.is_none(), "two callables at index {}", c.index);
        *at = Some(c);
    }
    table
}

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
    /// The `path_may2` bits (in the high word) its stage is made without.
    fn without(self) -> u64 {
        let producers = may2(SceneFeatures::PRODUCERS);
        let strands = may2(SceneFeatures::STRANDS);
        match self {
            Self::Copies => producers | strands,
            Self::Producers => strands,
            Self::Any => 0,
        }
    }
}

/// The stages of a pipeline being listed, each once.
struct Listed {
    specs: Vec<(Kernel, &'static str, Constants)>,
}

impl Listed {
    /// The index of the stage of `entry` of `kernel` made with `constants`.
    fn stage(&mut self, kernel: Kernel, entry: &'static str, constants: Constants) -> u32 {
        let spec = (kernel, entry, constants);
        let at = self
            .specs
            .iter()
            .position(|s| (s.0, s.1, s.2) == (spec.0, spec.1, spec.2));
        at.unwrap_or_else(|| {
            self.specs.push(spec);
            self.specs.len() - 1
        }) as u32
    }
}

/// A ray tracing pipeline of the path tracer, made for what its paths may meet, and the groups a
/// scene's hit records name.
pub(crate) struct PathPipeline {
    pub(super) pipeline: RayPipeline,
    groups: HitGroups,
}

impl PathPipeline {
    /// Compiles the pipeline from `kernels` on `threads` for paths that may meet `may`: the
    /// stages a scene of that mask can reach, each folded with it.
    pub(crate) fn new(
        device: &Device,
        kernels: &KernelSet,
        threads: CompileThreads<'_>,
        may: u64,
    ) -> Result<Self, Error> {
        let mut modules = [&pipeline::raygen::MODULE; Kernel::ALL.len()];
        for k in Kernel::ALL {
            modules[k as usize] = module(kernels, k.own())?;
        }
        let mask = Mask { may };
        let mut groups = HitGroups {
            surfaces: kernels.surfaces,
            ..HitGroups::default()
        };
        let pins = groups.pins();
        let mut listed = Listed { specs: Vec::new() };
        let folded = Constants {
            may,
            hit_surface: None,
        };
        let raygen = listed.stage(Kernel::Raygen, "paths", folded);
        let miss = listed.stage(Kernel::Raygen, "miss_path", folded);
        // `hit_surface` of the hit shaders of each pin: any id, then each id from 0 on.
        let pinned = |pin: u32, geometry: Geometry, without: u64| Constants {
            may: may & !geometry.without() & !without,
            hit_surface: Some(if pin == 0 { u32::MAX } else { pin - 1 }),
        };
        let producers = mask.any(may2(SceneFeatures::PRODUCERS));
        let rich = mask.any(MAY_RICH);
        let layers = mask.any(MAY_LAYERS);
        let cover = Some("cover_path_hit");
        let mut hit_groups = Vec::new();
        // Each pinned kind's groups one after another by pin, the first named in `groups`: a
        // path ray's hit shaders by what its record's material and geometry need, the opaque
        // ones without the rich lobes where the record's material has none.
        for (first, hit, any_hit, geometry, without, linked) in [
            (
                &mut groups.path,
                Some("vertex_hit"),
                cover,
                Geometry::Copies,
                MAY_RICH,
                true,
            ),
            (
                &mut groups.rich_path,
                Some("vertex_hit"),
                cover,
                Geometry::Copies,
                0,
                rich,
            ),
            (
                &mut groups.layer_path,
                Some("layer_hit"),
                cover,
                Geometry::Copies,
                0,
                layers,
            ),
            (
                &mut groups.producer_path,
                Some("vertex_hit"),
                cover,
                Geometry::Producers,
                MAY_RICH,
                producers,
            ),
            (
                &mut groups.producer_rich_path,
                Some("vertex_hit"),
                cover,
                Geometry::Producers,
                0,
                producers && rich,
            ),
            (
                &mut groups.producer_layer_path,
                Some("layer_hit"),
                cover,
                Geometry::Producers,
                0,
                producers && layers,
            ),
            (
                &mut groups.strand_path,
                Some("strand_hit"),
                None,
                Geometry::Any,
                0,
                mask.any(may2(SceneFeatures::STRANDS)),
            ),
            (
                &mut groups.shadow,
                None,
                Some("cover_shadow_hit"),
                Geometry::Any,
                0,
                true,
            ),
        ] {
            if !linked {
                continue;
            }
            *first = Some(hit_groups.len() as u32);
            for pin in 0..pins {
                // The coverage's any-hit shaders are made for any geometry, once per pin.
                let (c, any) = (
                    pinned(pin, geometry, without),
                    pinned(pin, Geometry::Any, 0),
                );
                hit_groups.push(HitGroup {
                    closest_hit: hit.map(|e| listed.stage(Kernel::Hits, e, c)),
                    any_hit: any_hit.map(|e| listed.stage(Kernel::Hits, e, any)),
                    intersection: None,
                });
            }
        }
        // The groups held once: the medium proxies' under shadow and layer rays, what path and
        // layer rays keep of particles and of light proxies, and the stack meet's closest-hit.
        let medium = mask.any(MAY_PROXIES);
        for (group, closest_hit, any_hit, intersection, linked) in [
            (
                &mut groups.medium_shadow,
                None,
                Some("proxy_shadow_hit"),
                Some("box_medium_hit"),
                medium,
            ),
            (
                &mut groups.front,
                None,
                Some("front_path_hit"),
                None,
                mask.any(MAY_PARTICLES),
            ),
            (
                &mut groups.light_front,
                None,
                Some("front_path_light_hit"),
                Some("box_light_hit"),
                mask.any(MAY_LOCAL_LIGHTS),
            ),
            (
                &mut groups.layers,
                None,
                Some("cover_layers_hit"),
                None,
                mask.any(MAY_PARTICLES),
            ),
            (
                &mut groups.medium_layers,
                None,
                Some("cover_layers_medium_hit"),
                Some("box_medium_hit"),
                medium,
            ),
            (
                &mut groups.light_layers,
                None,
                Some("cover_layers_light_hit"),
                Some("box_light_hit"),
                mask.any(MAY_LOCAL_LIGHTS),
            ),
            (
                &mut groups.stack_meet,
                Some("stack_meet"),
                None,
                None,
                mask.any(MAY_STACK),
            ),
        ] {
            if !linked {
                continue;
            }
            let mut stage =
                |e: Option<&'static str>| e.map(|e| listed.stage(Kernel::Hits, e, folded));
            *group = Some(hit_groups.len() as u32);
            hit_groups.push(HitGroup {
                closest_hit: stage(closest_hit),
                any_hit: stage(any_hit),
                intersection: stage(intersection),
            });
        }
        let records: Vec<Option<u32>> = callables()
            .into_iter()
            .map(|c| {
                let c = c.expect("a callable at every index");
                (c.linked)(mask).then(|| {
                    let constants = Constants {
                        may: may & !c.without,
                        hit_surface: None,
                    };
                    listed.stage(c.kernel, c.entry, constants)
                })
            })
            .collect();
        let bytes: Vec<_> = listed.specs.iter().map(|&(k, _, c)| k.spec(c)).collect();
        // Every stage takes its constants: a stage with none is handed the whole module, with
        // every other entry point in it.
        let stages = listed
            .specs
            .iter()
            .zip(&bytes)
            .map(|(&(kernel, entry, _), (bytes, entries))| {
                let data = SpecData { entries, bytes };
                ShaderStage::new(modules[kernel as usize], entry, data)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let callables: Vec<_> = records.iter().map(|r| r.map(ShaderRecord::new)).collect();
        let desc = RayPipelineDesc {
            stages: &stages,
            raygens: &[ShaderRecord::new(raygen)],
            misses: &[ShaderRecord::new(miss)],
            hit_groups: &hit_groups,
            callables: &callables,
            max_recursion: 1,
            flags: RayPipelineFlags::scene(device.caps()),
            threads,
        };
        let started = Instant::now();
        let name = format!("path pipeline {may:#x}");
        let pipeline = device.create_ray_pipeline(&desc, &name)?;
        log::debug!(
            "{name}: {} stages, {} of {CALLABLES} callables, made in {:.1} s",
            stages.len(),
            records.iter().flatten().count(),
            started.elapsed().as_secs_f64()
        );
        Ok(Self { pipeline, groups })
    }

    /// The pipeline whose hit table its launches trace through, and the groups the table names.
    pub(crate) fn hit_pipeline(&self) -> (&RayPipeline, &HitGroups) {
        (&self.pipeline, &self.groups)
    }
}

/// A stage of [`lean_probe`]: its module, entry and specialization.
type ProbeStage = (
    &'static ShaderModule,
    &'static str,
    ([u8; 32], &'static [SpecEntry]),
);

/// TEMP-EXPERIMENT: makes the lean tracer's stage set as one pipeline and logs its stage count,
/// what it hands the driver and how long the driver took, for the cold-compile check.
pub(crate) fn lean_probe(
    device: &Device,
    kernels: &KernelSet,
    threads: CompileThreads<'_>,
) -> Result<(), Error> {
    use crate::shaders::path::lean_paths;
    let lean = module(kernels, &lean_paths::MODULE)?;
    let hits = module(kernels, &pipeline::hits::MODULE)?;
    let pins = HitGroups {
        surfaces: kernels.surfaces,
        ..HitGroups::default()
    }
    .pins();
    let lean_spec = |pin: Option<u32>| {
        let d = lean_paths::Spec::default();
        let s = lean_paths::Spec {
            hit_surface: pin.unwrap_or(d.hit_surface),
            ..d
        };
        (s.bytes(), lean_paths::Spec::ENTRIES)
    };
    let pin_of = |pin: u32| Some(if pin == 0 { u32::MAX } else { pin - 1 });
    let mut specs: Vec<ProbeStage> = vec![
        (lean, "paths", lean_spec(None)),
        (lean, "miss_scatter", lean_spec(None)),
        (lean, "miss_shadow", lean_spec(None)),
    ];
    let mut hit_groups = Vec::new();
    for pin in 0..pins {
        let c = Constants {
            may: u64::MAX,
            hit_surface: pin_of(pin),
        };
        let at = specs.len() as u32;
        specs.push((lean, "surface_hit", lean_spec(pin_of(pin))));
        specs.push((hits, "cover_path_hit", Kernel::Hits.spec(c)));
        hit_groups.push(HitGroup {
            closest_hit: Some(at),
            any_hit: Some(at + 1),
            intersection: None,
        });
    }
    for pin in 0..pins {
        let c = Constants {
            may: u64::MAX,
            hit_surface: pin_of(pin),
        };
        let at = specs.len() as u32;
        specs.push((hits, "cover_shadow_hit", Kernel::Hits.spec(c)));
        hit_groups.push(HitGroup {
            closest_hit: None,
            any_hit: Some(at),
            intersection: None,
        });
    }
    let stages = specs
        .iter()
        .map(|(m, entry, (bytes, entries))| ShaderStage::new(m, entry, SpecData { entries, bytes }))
        .collect::<Result<Vec<_>, _>>()?;
    let instructions: u32 = stages.iter().map(|s| s.entry.instructions).sum();
    for s in &stages {
        log::info!(
            "lean probe stage {:?}: {} instructions",
            s.entry.name,
            s.entry.instructions
        );
    }
    let desc = RayPipelineDesc {
        stages: &stages,
        raygens: &[ShaderRecord::new(0)],
        misses: &[ShaderRecord::new(1), ShaderRecord::new(2)],
        hit_groups: &hit_groups,
        callables: &[],
        max_recursion: 1,
        flags: RayPipelineFlags::scene(device.caps()),
        threads,
    };
    let started = Instant::now();
    let _pipeline = device.create_ray_pipeline(&desc, "lean probe")?;
    log::info!(
        "lean probe: {} stages, {} instructions handed over, {} linked surfaces, made in {:.2} s",
        stages.len(),
        instructions,
        kernels.surfaces,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The path tracer's pipelines as [`crate::mask_sets::MaskSets`] members: one per mask.
pub(crate) struct PathPipelines {
    pub(crate) kernels: KernelSet,
}

impl Members for PathPipelines {
    type Pipeline = PathPipeline;

    fn names(&self) -> &'static [&'static str] {
        &["path pipeline"]
    }

    fn runs(&self, _: u64, _: &[usize]) -> Vec<usize> {
        vec![0]
    }

    fn compile(
        &self,
        device: &Device,
        loading: Option<&Arc<Loading>>,
        _: usize,
        may: u64,
    ) -> Result<PathPipeline, Error> {
        // The driver compiles the pipeline's shaders on the loading workers that join it.
        let Some(pool) = loading else {
            return PathPipeline::new(device, &self.kernels, CompileThreads::default(), may);
        };
        let spawn = |task: Box<dyn FnOnce() + Send>| pool.spawn(task);
        let threads = CompileThreads::Pool {
            spawn: &spawn,
            max: std::thread::available_parallelism().map_or(1, |n| n.get() as u32),
        };
        // TEMP-EXPERIMENT
        if std::env::var_os("ORR_LEAN_PROBE").is_some() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            let mut result = Ok(());
            ONCE.call_once(|| {
                let again = CompileThreads::Pool {
                    spawn: &spawn,
                    max: std::thread::available_parallelism().map_or(1, |n| n.get() as u32),
                };
                result = lean_probe(device, &self.kernels, again);
            });
            result?;
            log::info!("lean probe done; exiting");
            std::process::exit(0);
        }
        PathPipeline::new(device, &self.kernels, threads, may)
    }
}
