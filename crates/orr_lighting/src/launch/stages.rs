//! The pipeline of the path tracer: its stage table for what its paths may meet, each stage
//! folded with that and held to a cap on what it hands the driver, its hit groups, and its
//! creation.

use std::ffi::CString;
use std::sync::Arc;
use std::time::Instant;

use orr_scene::orr_gpu::{
    CompileThreads, Device, HitGroup, RayPipeline, RayPipelineDesc, RayPipelineFlags, ShaderRecord,
    ShaderStage, SpecData, SpecEntry, handed_instructions,
};
use orr_scene::{HitGroups, Loading, SceneFeatures};
use orr_shaders::surface_declares;

use crate::mask_sets::Members;
use crate::shaders::consts::{
    ANY_HIT_INSTRUCTIONS, HIT_INSTRUCTIONS, MISS_INSTRUCTIONS, PATH_SOURCE_UPDATE,
    PATH_SOURCE_VIEW, PIPELINE_INSTRUCTIONS, PRODUCER_ANY_HIT_INSTRUCTIONS,
    PRODUCER_HIT_INSTRUCTIONS, RAYGEN_INSTRUCTIONS, SAMPLER_INSTRUCTIONS,
};
use crate::shaders::path::lean_paths;
use crate::staged::{MAY_LOCAL, MAY_UPDATE, MAY_VIEW};
use crate::{Error, KernelSet, module};

/// The bit of `path_may2`, in the high word, that `feature` decides.
fn may2(feature: SceneFeatures) -> u64 {
    u64::from(!SceneFeatures(!feature.0).surface_may()) << 32
}

/// What a stage does in the pipeline, which decides the cap on its size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Raygen,
    Miss,
    /// A closest-hit shader or the first vertex's callable, which run a whole vertex.
    Hit,
    /// An any-hit shader of a candidate that is not opaque.
    AnyHit,
    /// The callable that draws a first vertex's candidate from its local light sampler.
    Sampler,
}

/// The most instructions the stages of a pipeline may hand the driver together.
pub const PIPELINE_CAP: u32 = PIPELINE_INSTRUCTIONS;

/// The geometry a hit group's shaders run on, which their stages are made for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Geometry {
    /// Copies' clusters, cooked or skinned.
    Copies,
    /// Producer entries' clusters, not strands.
    Producers,
    /// Either of those.
    Triangles,
}

impl Geometry {
    fn label(self) -> &'static str {
        match self {
            Self::Copies => ".copies",
            Self::Producers => ".producers",
            Self::Triangles => ".triangles",
        }
    }

    /// The `path_may2` bits (in the high word) its stages are made without.
    fn without(self) -> u64 {
        let producers = may2(SceneFeatures::PRODUCERS);
        let strands = may2(SceneFeatures::STRANDS);
        match self {
            Self::Copies => producers | strands,
            Self::Producers | Self::Triangles => strands,
        }
    }
}

/// A stage of the table: its entry in the lean kernel, its role, and the constants it is made
/// with: what it may meet (`path_may` in the low word, `path_may2` in the high one) and the
/// linked surface id its shader is made for.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Spec {
    entry: &'static str,
    role: Role,
    /// The geometry of the hit group the stage serves; none for a stage no group holds.
    geometry: Option<Geometry>,
    may: u64,
    hit_surface: u32,
}

impl Spec {
    /// The name a driver's tools show the stage by: its entry, the geometry it is made for and
    /// the linked surface id it is made for, so a trace tells stages of one entry apart.
    fn label(&self) -> CString {
        let surface = match (self.geometry, self.hit_surface) {
            (None, _) => String::new(),
            (Some(_), u32::MAX) => ".any".into(),
            (Some(_), id) => format!(".surface{id}"),
        };
        let geometry = self.geometry.map_or("", Geometry::label);
        CString::new(format!("{}{geometry}{surface}", self.entry)).expect("no nul in a label")
    }

    /// Whether the stage is made for a linked surface type, or for any of them.
    fn linked(&self, surfaces: u32) -> bool {
        surfaces != 0 && self.hit_surface != 0 && self.geometry.is_some()
    }

    /// The most instructions the stage may hand the driver: its role's, with what a producer
    /// entry's geometry adds to a hit or an any-hit shader. A stage made for a linked surface
    /// has none of its own: it holds the application's code beside the standard stage's.
    fn cap(&self, surfaces: u32) -> Option<u32> {
        if self.linked(surfaces) {
            return None;
        }
        let producers = !matches!(self.geometry, None | Some(Geometry::Copies));
        let (base, producer) = match self.role {
            Role::Raygen => return Some(RAYGEN_INSTRUCTIONS),
            Role::Miss => return Some(MISS_INSTRUCTIONS),
            Role::Sampler => return Some(SAMPLER_INSTRUCTIONS),
            Role::Hit => (HIT_INSTRUCTIONS, PRODUCER_HIT_INSTRUCTIONS),
            Role::AnyHit => (ANY_HIT_INSTRUCTIONS, PRODUCER_ANY_HIT_INSTRUCTIONS),
        };
        Some(base + if producers { producer } else { 0 })
    }

    fn constants(&self) -> ([u8; 32], &'static [SpecEntry]) {
        let s = lean_paths::Spec {
            path_may: self.may as u32,
            path_may2: (self.may >> 32) as u32,
            hit_surface: self.hit_surface,
            ..lean_paths::Spec::default()
        };
        (s.bytes(), lean_paths::Spec::ENTRIES)
    }
}

/// The stages of a pipeline, each once, with its hit groups and what the scene's hit records
/// name of them.
struct Table {
    specs: Vec<Spec>,
    /// The raygen stage of each source by its id, where the pipeline runs the source.
    raygens: [Option<u32>; 4],
    misses: [u32; 2],
    /// The callable of a path's first vertex.
    first: u32,
    /// The callable of that vertex's local light sampler, where paths may read one.
    local: Option<u32>,
    hit_groups: Vec<HitGroup>,
    groups: HitGroups,
}

impl Table {
    /// The index of the stage `spec`.
    fn stage(specs: &mut Vec<Spec>, spec: Spec) -> u32 {
        let at = specs.iter().position(|s| *s == spec);
        at.unwrap_or_else(|| {
            specs.push(spec);
            specs.len() - 1
        }) as u32
    }

    /// The table of a pipeline whose paths may meet `may`, with the linked surface types of
    /// `kernels`.
    fn new(may: u64, kernels: &KernelSet) -> Self {
        let surfaces = kernels.surfaces;
        // A triangle of two surface ids is shaded by the groups that serve any id, which only
        // a type that declares its other side can stand on.
        let two_sided = kernels
            .surface_declares
            .iter()
            .any(|d| d & surface_declares::OTHER_SIDE != 0);
        let mut specs = Vec::new();
        let mut groups = HitGroups {
            surfaces,
            ..HitGroups::default()
        };
        let folded = |entry, role| Spec {
            entry,
            role,
            geometry: None,
            may,
            hit_surface: lean_paths::Spec::default().hit_surface,
        };
        let mut raygens = [None; 4];
        for (source, bit, entry) in [
            (PATH_SOURCE_VIEW, MAY_VIEW, "paths"),
            (PATH_SOURCE_UPDATE, MAY_UPDATE, "updates"),
        ] {
            if may & bit != 0 {
                raygens[source as usize] =
                    Some(Self::stage(&mut specs, folded(entry, Role::Raygen)));
            }
        }
        let misses = [
            Self::stage(&mut specs, folded("miss_scatter", Role::Miss)),
            Self::stage(&mut specs, folded("miss_shadow", Role::Miss)),
        ];
        let first = Self::stage(&mut specs, folded("first_vertex", Role::Hit));
        let local = (may & MAY_LOCAL != 0)
            .then(|| Self::stage(&mut specs, folded("local_light", Role::Sampler)));
        // `hit_surface` of the hit shaders of each pin: any id, then each id from 0 on. Where
        // no surface links every material shows id 0.
        let pinned = |entry, role, pin: u32, geometry: Geometry| Spec {
            entry,
            role,
            geometry: Some(geometry),
            may: may & !geometry.without(),
            hit_surface: if surfaces == 0 {
                0
            } else if pin == 0 {
                u32::MAX
            } else {
                pin - 1
            },
        };
        let producers = may & may2(SceneFeatures::PRODUCERS) != 0;
        let mut hit_groups = Vec::new();
        let pins = groups.pins();
        // Each pinned kind's groups one after another by pin, the first named in `groups`.
        for (first, hit, any_hit, geometry, linked) in [
            (
                &mut groups.path,
                Some("surface_hit"),
                "cover_hit",
                Geometry::Copies,
                true,
            ),
            (
                &mut groups.producer_path,
                Some("surface_hit"),
                "cover_hit",
                Geometry::Producers,
                producers,
            ),
            (
                &mut groups.shadow,
                None,
                "shadow_hit",
                Geometry::Triangles,
                true,
            ),
        ] {
            if !linked {
                continue;
            }
            *first = Some(hit_groups.len() as u32);
            for pin in 0..pins {
                // The pin keeps its place among the kind's groups, with no shader.
                if surfaces != 0 && pin == 0 && !two_sided {
                    hit_groups.push(HitGroup {
                        closest_hit: None,
                        any_hit: None,
                        intersection: None,
                    });
                    continue;
                }
                hit_groups.push(HitGroup {
                    closest_hit: hit
                        .map(|e| Self::stage(&mut specs, pinned(e, Role::Hit, pin, geometry))),
                    any_hit: Some(Self::stage(
                        &mut specs,
                        pinned(any_hit, Role::AnyHit, pin, geometry),
                    )),
                    intersection: None,
                });
            }
        }
        Self {
            specs,
            raygens,
            misses,
            first,
            local,
            hit_groups,
            groups,
        }
    }
}

/// A stage of a path pipeline as the driver is handed it.
#[derive(Clone, Debug)]
pub struct StageSize {
    pub entry: &'static str,
    pub role: Role,
    /// The geometry of the hit group it serves, as its label names it; empty for a stage no
    /// group holds.
    pub geometry: &'static str,
    /// The linked surface id the stage is made for; `u32::MAX` for any.
    pub hit_surface: u32,
    pub instructions: u32,
    /// The most of orr's own code it may hand the driver; none for a stage made for a linked
    /// surface.
    pub cap: Option<u32>,
    /// What the kernels' linked types add to it: its size beyond the same stage of orr's
    /// kernels with nothing linked, where a stage made for a linked surface is the standard
    /// surface's. None for those kernels themselves.
    pub linked: Option<i64>,
    /// Of a stage made for a linked surface, what it holds beyond the stage of its entry and
    /// geometry made for the standard surface: the surface's own code, less the standard code
    /// it replaces.
    pub own: Option<i64>,
}

/// What `spec`'s stage of `module` hands the driver: its entry folded with its constants.
fn handed(module: &'static orr_shaders::ShaderModule, spec: &Spec) -> Result<u32, Error> {
    let (bytes, entries) = spec.constants();
    let stage = ShaderStage::new(
        module,
        spec.entry,
        SpecData {
            entries,
            bytes: &bytes,
        },
    )?;
    Ok(handed_instructions(
        stage.spirv,
        stage.entry.spirv_name,
        stage.spec,
    ))
}

/// What each stage of the pipeline of paths that may meet `may` hands the driver, from
/// `kernels`, and what of it is orr's own.
pub fn path_stage_sizes(kernels: &KernelSet, may: u64) -> Result<Vec<StageSize>, Error> {
    let lean = module(kernels, &lean_paths::MODULE)?;
    // Kernels that link a type compile the lean stages anew, with the types' code in them.
    let own: &'static orr_shaders::ShaderModule = &lean_paths::MODULE;
    let plain = (!std::ptr::eq(lean, own)).then_some(own);
    let mut sizes = Table::new(may, kernels)
        .specs
        .iter()
        .map(|spec| {
            let instructions = handed(lean, spec)?;
            let linked = plain
                .map(|plain| {
                    let standard = Spec {
                        hit_surface: match spec.geometry {
                            Some(_) => 0,
                            None => spec.hit_surface,
                        },
                        ..*spec
                    };
                    handed(plain, &standard).map(|own| i64::from(instructions) - i64::from(own))
                })
                .transpose()?;
            Ok(StageSize {
                entry: spec.entry,
                role: spec.role,
                geometry: spec.geometry.map_or("", Geometry::label),
                hit_surface: spec.hit_surface,
                instructions,
                cap: spec.cap(kernels.surfaces),
                linked,
                own: None,
            })
        })
        .collect::<Result<Vec<StageSize>, Error>>()?;
    let standard: Vec<_> = sizes
        .iter()
        .filter(|s| s.hit_surface == 0)
        .map(|s| (s.entry, s.geometry, s.instructions))
        .collect();
    for s in sizes.iter_mut().filter(|s| s.cap.is_none()) {
        s.own = standard
            .iter()
            .find(|&&(entry, geometry, _)| (entry, geometry) == (s.entry, s.geometry))
            .map(|&(.., instructions)| i64::from(s.instructions) - i64::from(instructions));
    }
    Ok(sizes)
}

/// The first of `sizes` whose own code, without what linked types add to it, is over its cap,
/// or the pipeline's total, the linked types' code among it, where that is over
/// [`PIPELINE_CAP`].
pub fn over_cap(sizes: &[StageSize]) -> Option<String> {
    for s in sizes {
        let own = i64::from(s.instructions) - s.linked.unwrap_or(0);
        if let Some(cap) = s.cap.filter(|&cap| own > i64::from(cap)) {
            return Some(format!(
                "stage {}{} hands the driver {own} instructions of orr's own, over its {cap} as \
                 a {:?}",
                s.entry, s.geometry, s.role
            ));
        }
    }
    let total: u32 = sizes.iter().map(|s| s.instructions).sum();
    (total > PIPELINE_CAP).then(|| {
        format!(
            "{} stages hand the driver {total} instructions, over the pipeline's {PIPELINE_CAP}",
            sizes.len()
        )
    })
}

/// A ray tracing pipeline of the path tracer, made for what its paths may meet, and the groups a
/// scene's hit records name.
pub(crate) struct PathPipeline {
    pub(super) pipeline: RayPipeline,
    /// The raygen record of each source by its id.
    pub(super) raygens: [Option<u32>; 4],
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
        let lean = module(kernels, &lean_paths::MODULE)?;
        let table = Table::new(may, kernels);
        let name = format!("path pipeline {may:#x}");
        let sizes = path_stage_sizes(kernels, may)?;
        for s in &sizes {
            let stage = format!("{name}: {}{}", s.entry, s.geometry);
            match (s.own, s.linked) {
                (Some(own), _) => log::info!(
                    "{stage} for surface {:#x}: {} instructions, {own:+} its surface's own",
                    s.hit_surface,
                    s.instructions
                ),
                (None, Some(linked)) => log::info!(
                    "{stage}: {} instructions, {linked:+} of the linked types",
                    s.instructions
                ),
                (None, None) => log::debug!("{stage}: {} instructions", s.instructions),
            }
        }
        if let Some(message) = over_cap(&sizes) {
            return Err(Error::StageSize(message));
        }
        let bytes: Vec<_> = table.specs.iter().map(Spec::constants).collect();
        let labels: Vec<CString> = table.specs.iter().map(Spec::label).collect();
        // Every stage takes its constants: a stage with none is handed the whole module, with
        // every other entry point in it.
        let stages = table
            .specs
            .iter()
            .zip(&bytes)
            .zip(&labels)
            .map(|((spec, (bytes, entries)), label)| {
                Ok(ShaderStage::new(lean, spec.entry, SpecData { entries, bytes })?.labeled(label))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        // A source's record is its place among the raygens the pipeline holds.
        let mut records = Vec::new();
        let mut raygens = [None; 4];
        for (source, stage) in table.raygens.iter().enumerate() {
            if let Some(stage) = stage {
                raygens[source] = Some(records.len() as u32);
                records.push(ShaderRecord::new(*stage));
            }
        }
        let desc = RayPipelineDesc {
            stages: &stages,
            raygens: &records,
            misses: &table.misses.map(ShaderRecord::new),
            hit_groups: &table.hit_groups,
            // In `raygen.slang`'s CALL_* order.
            callables: &[
                Some(ShaderRecord::new(table.first)),
                table.local.map(ShaderRecord::new),
            ],
            max_recursion: 1,
            flags: RayPipelineFlags::scene(device.caps()),
            threads,
        };
        let started = Instant::now();
        let pipeline = device.create_ray_pipeline(&desc, &name)?;
        log::info!(
            "{name}: {} stages, {} instructions handed over, made in {:.1} s",
            stages.len(),
            sizes.iter().map(|s| s.instructions).sum::<u32>(),
            started.elapsed().as_secs_f64()
        );
        Ok(Self {
            pipeline,
            raygens,
            groups: table.groups,
        })
    }

    /// The pipeline whose hit table its launches trace through, and the groups the table names.
    pub(crate) fn hit_pipeline(&self) -> (&RayPipeline, &HitGroups) {
        (&self.pipeline, &self.groups)
    }
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
        PathPipeline::new(device, &self.kernels, threads, may)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stage over its cap compiles for longer than a cold start may wait, whatever the scene
    /// holds: the pipeline that may meet everything is the one a session runs.
    #[test]
    fn the_stages_of_the_pipeline_that_may_meet_everything_hold_their_caps() {
        let sizes = path_stage_sizes(&KernelSet::default(), u64::MAX).unwrap();
        assert_eq!(over_cap(&sizes), None);
    }
}
