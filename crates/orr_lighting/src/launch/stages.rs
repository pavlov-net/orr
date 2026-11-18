//! The pipeline of the path tracer: its stage table for what its paths may meet, each stage
//! folded with that and measured against its cap on what it hands the driver, its hit groups, and
//! its creation.

use std::ffi::CString;
use std::sync::Arc;
use std::time::Instant;

use orr_scene::orr_gpu::{
    CompileThreads, Device, HitGroup, RayPipeline, RayPipelineDesc, RayPipelineFlags, ShaderRecord,
    ShaderStage, SpecData, SpecEntry, handed_instructions,
};
use orr_scene::{HitGroups, Loading, SceneFeatures, Urgency};
use orr_shaders::surface_declares;

use crate::mask_sets::{Members, driver_cost};
use crate::shaders::consts::{
    ANY_HIT_INSTRUCTIONS, CAUSTIC_HIT_INSTRUCTIONS, COATED_EMITTER_HIT_INSTRUCTIONS,
    DEPOSIT_INSTRUCTIONS, DISPERSE_INSTRUCTIONS, DISPERSION_LAYER_INSTRUCTIONS,
    EMITTER_HIT_INSTRUCTIONS, EMITTER_VIEW_INSTRUCTIONS, FRONT_ANY_HIT_INSTRUCTIONS,
    FRONT_INSTRUCTIONS, GATHER_INSTRUCTIONS, HIT_INSTRUCTIONS, INTERIOR_HIT_INSTRUCTIONS,
    LAYER_HIT_INSTRUCTIONS, LIGHT_BOX_INSTRUCTIONS, LOCAL_FIRST_HIT_INSTRUCTIONS,
    MAP_GATHER_HIT_INSTRUCTIONS, MAP_GATHER_INSTRUCTIONS, MEDIUM_VERTEX_INSTRUCTIONS,
    MISS_INSTRUCTIONS, PATH_SOURCE_BACKGROUND, PATH_SOURCE_PHOTON, PATH_SOURCE_UPDATE,
    PATH_SOURCE_VIEW, PHOTON_HIT_INSTRUCTIONS, PHOTON_START_INSTRUCTIONS, PIPELINE_INSTRUCTIONS,
    PRODUCER_ANY_HIT_INSTRUCTIONS, PRODUCER_HIT_INSTRUCTIONS, PROXY_ANY_HIT_INSTRUCTIONS,
    PROXY_INTERSECTION_INSTRUCTIONS, PROXY_MARCH_INSTRUCTIONS, RAYGEN_DISPERSION_INSTRUCTIONS,
    RAYGEN_FRONT_INSTRUCTIONS, RAYGEN_INSTRUCTIONS, RAYGEN_LAYER_INSTRUCTIONS,
    RAYGEN_LENS_INSTRUCTIONS, RAYGEN_PHOTON_INSTRUCTIONS, RAYGEN_PROXY_INSTRUCTIONS,
    RAYGEN_STACK_INSTRUCTIONS, RAYGEN_WALK_INSTRUCTIONS, RICH_HIT_INSTRUCTIONS,
    STACK_LAYER_INSTRUCTIONS, STACK_STEP_INSTRUCTIONS, STACK_VERTEX_INSTRUCTIONS,
    TRACED_HIT_INSTRUCTIONS, WALK_EXIT_INSTRUCTIONS, WALK_HIT_INSTRUCTIONS,
    WALK_LAYER_INSTRUCTIONS, WALK_MEETS_INSTRUCTIONS,
};
use crate::shaders::path::{layer, plain, rich, walk};
use crate::specialize::{
    MAY_ALL, MAY_BACKGROUND, MAY_CAUSTICS, MAY_DEBUG, MAY_DISPERSION, MAY_EMISSIVE, MAY_EMITTER,
    MAY_INTERIORS, MAY_LAYERS, MAY_LENS, MAY_LOBES, MAY_LOCAL, MAY_LOCAL_LIGHTS, MAY_MEDIUM,
    MAY_PARTICLES, MAY_PHOTON, MAY_PROXIES, MAY_REFERENCE, MAY_SCATTERING, MAY_STACK, MAY_TRACED,
    MAY_UPDATE, MAY_VIEW, MAY_WALK,
};
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
    /// The callable of a path's vertex in an interior's medium: at a scatter, or where a walk
    /// stopped at its cap leaves the interior.
    Medium,
    /// The callable of the vertex where an interior's real-time walk meets its surface.
    Meets,
    /// The callable of the vertex where a subsurface walk leaves.
    Exit,
    /// The callable that weighs a coated emitter's light by what its coat and fuzz pass.
    View,
    /// The callable of a step of a directional job's walk through a plane stack.
    Stack,
    /// The callable that takes what a path's ray meets on its way to its hit: particles, the
    /// proxies of sphere and rect lights, and the march of the linked medium's in-scatter.
    Front,
    /// The any-hit that marks a path ray's payload where it met a particle or a light's proxy,
    /// and the intersection shader of a light's proxy under a path ray.
    FrontAnyHit,
    LightBox,
    /// A medium proxy's intersection shader under a shadow ray, its any-hit there, and the
    /// callable of the marches of those with a volume that the ray passed.
    ProxyBox,
    ProxyAnyHit,
    ProxyMarch,
    /// The raygen of the frame's caustic photons, the callable that starts one from a caster and
    /// the one of its deposit.
    PhotonRaygen,
    PhotonStart,
    Deposit,
    /// The callables of the caustic photons' gather at a path's vertex: the grid's, and the
    /// reference's map's.
    Gather,
    MapGather,
    /// The callable that weighs a dispersive layer's companion wavelengths.
    Disperse,
}

/// What the stages of a pipeline may hand the driver together, linked types' code among it.
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
    /// Strand entries' segments, linear swept spheres.
    Strands,
}

impl Geometry {
    fn label(self) -> &'static str {
        match self {
            Self::Copies => ".copies",
            Self::Producers => ".producers",
            Self::Triangles => ".triangles",
            Self::Strands => ".strands",
        }
    }

    /// The `path_may2` bits (in the high word) its stages are made without.
    fn without(self) -> u64 {
        let producers = may2(SceneFeatures::PRODUCERS);
        let strands = may2(SceneFeatures::STRANDS);
        match self {
            Self::Copies => producers | strands,
            Self::Producers | Self::Triangles => strands,
            Self::Strands => 0,
        }
    }
}

/// The kernel a stage comes from: the one whose BSDF has the plain lobes alone, the one
/// with a coat, a fuzz, a thin film and a thin wall's transmission too, that one with the
/// subsurface lobe and its walks, which takes its place where the scene's surfaces may walk, or
/// the dielectric class's, whose BSDF transmits and whose vertex crosses interiors.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kernel {
    Plain,
    Rich,
    Walk,
    Layer,
}

impl Kernel {
    fn own(self) -> &'static orr_shaders::ShaderModule {
        match self {
            Self::Plain => &plain::MODULE,
            Self::Rich => &rich::MODULE,
            Self::Walk => &walk::MODULE,
            Self::Layer => &layer::MODULE,
        }
    }
}

/// A stage of the table: its entry in its kernel, its role, and the constants it is made
/// with: what it may meet (`path_may` in the low word, `path_may2` in the high one) and the
/// linked surface id its shader is made for.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Spec {
    kernel: Kernel,
    entry: &'static str,
    role: Role,
    /// The geometry of the hit group the stage serves; none for a stage no group holds.
    geometry: Option<Geometry>,
    may: u64,
    hit_surface: u32,
}

impl Spec {
    /// The name a driver's tools show the stage by: its entry, the geometry it is made for, the
    /// linked surface id it is made for and whether it shades emitters, so a trace tells stages
    /// of one entry apart.
    fn label(&self) -> CString {
        let surface = match (self.geometry, self.hit_surface) {
            (None, _) => String::new(),
            (Some(_), u32::MAX) => ".any".into(),
            (Some(_), id) => format!(".surface{id}"),
        };
        let geometry = self.geometry.map_or("", Geometry::label);
        let emitter = if self.geometry.is_some() && self.may & MAY_EMITTER != 0 {
            ".emitter"
        } else {
            ""
        };
        let rich = match self.kernel {
            Kernel::Plain => "",
            Kernel::Rich => ".rich",
            Kernel::Walk => ".walk",
            Kernel::Layer => ".layer",
        };
        CString::new(format!("{}{geometry}{surface}{rich}{emitter}", self.entry))
            .expect("no nul in a label")
    }

    /// Whether the stage is made for a linked surface type, or for any of them.
    fn linked(&self, surfaces: u32) -> bool {
        surfaces != 0 && self.hit_surface != 0 && self.geometry.is_some()
    }

    /// The cap on orr's own code in the stage: its role's, with each allowance its mask and kind
    /// keep. A stage made for a linked surface has none of its own: it holds the application's
    /// code beside the standard stage's.
    fn cap(&self, surfaces: u32) -> Option<u32> {
        if self.linked(surfaces) {
            return None;
        }
        let base = match self.role {
            Role::Raygen => RAYGEN_INSTRUCTIONS,
            Role::Miss => MISS_INSTRUCTIONS,
            Role::Hit => HIT_INSTRUCTIONS,
            Role::AnyHit => ANY_HIT_INSTRUCTIONS,
            Role::Medium => MEDIUM_VERTEX_INSTRUCTIONS,
            Role::Meets => WALK_MEETS_INSTRUCTIONS,
            Role::Exit => WALK_EXIT_INSTRUCTIONS,
            Role::PhotonRaygen => RAYGEN_PHOTON_INSTRUCTIONS,
            Role::PhotonStart => PHOTON_START_INSTRUCTIONS,
            Role::Deposit => DEPOSIT_INSTRUCTIONS,
            Role::Gather => GATHER_INSTRUCTIONS,
            Role::MapGather => MAP_GATHER_INSTRUCTIONS,
            Role::Disperse => DISPERSE_INSTRUCTIONS,
            Role::View => EMITTER_VIEW_INSTRUCTIONS,
            Role::Stack => STACK_STEP_INSTRUCTIONS,
            Role::Front => FRONT_INSTRUCTIONS,
            Role::FrontAnyHit => FRONT_ANY_HIT_INSTRUCTIONS,
            Role::LightBox => LIGHT_BOX_INSTRUCTIONS,
            Role::ProxyBox => PROXY_INTERSECTION_INSTRUCTIONS,
            Role::ProxyAnyHit => PROXY_ANY_HIT_INSTRUCTIONS,
            Role::ProxyMarch => PROXY_MARCH_INSTRUCTIONS,
        };
        let allowed = allowances()
            .iter()
            .filter(|a| self.may & a.bits != 0 && (a.holds)(self))
            .map(|a| a.instructions)
            .sum::<u32>();
        Some(base + allowed)
    }

    fn constants(&self) -> ([u8; 32], &'static [SpecEntry]) {
        let (may, may2) = (self.may as u32, (self.may >> 32) as u32);
        match self.kernel {
            Kernel::Plain => {
                let s = plain::Spec {
                    path_may: may,
                    path_may2: may2,
                    hit_surface: self.hit_surface,
                    ..plain::Spec::default()
                };
                (s.bytes(), plain::Spec::ENTRIES)
            }
            Kernel::Rich => {
                let s = rich::Spec {
                    path_may: may,
                    path_may2: may2,
                    hit_surface: self.hit_surface,
                    ..rich::Spec::default()
                };
                (s.bytes(), rich::Spec::ENTRIES)
            }
            Kernel::Walk => {
                let s = walk::Spec {
                    path_may: may,
                    path_may2: may2,
                    hit_surface: self.hit_surface,
                    ..walk::Spec::default()
                };
                (s.bytes(), walk::Spec::ENTRIES)
            }
            Kernel::Layer => {
                let s = layer::Spec {
                    path_may: may,
                    path_may2: may2,
                    hit_surface: self.hit_surface,
                    ..layer::Spec::default()
                };
                (s.bytes(), layer::Spec::ENTRIES)
            }
        }
    }
}

/// What a stage may hold beyond its role's cap where its mask has any of `bits` and `holds` says
/// the stage is one that holds the feature's code.
struct Allowance {
    bits: u64,
    instructions: u32,
    holds: fn(&Spec) -> bool,
}

/// Whether a stage runs a whole vertex: a hit, the first vertex, or a medium's, a walk's meeting
/// or a walk's exit callable.
fn runs_vertex(s: &Spec) -> bool {
    matches!(s.role, Role::Hit | Role::Medium | Role::Meets | Role::Exit)
}

/// Every allowance of a stage's cap.
fn allowances() -> [Allowance; 24] {
    let raygen = |s: &Spec| s.role == Role::Raygen;
    let hit = |s: &Spec| s.role == Role::Hit;
    let layer = |s: &Spec| s.role == Role::Hit && s.kernel == Kernel::Layer;
    fn allow(bits: u64, instructions: u32, holds: fn(&Spec) -> bool) -> Allowance {
        Allowance {
            bits,
            instructions,
            holds,
        }
    }
    fn producer(s: &Spec) -> bool {
        !matches!(s.geometry, None | Some(Geometry::Copies))
    }
    [
        // What a raygen holds per feature its paths may meet.
        allow(MAY_LAYERS, RAYGEN_LAYER_INSTRUCTIONS, raygen),
        allow(MAY_SCATTERING | MAY_WALK, RAYGEN_WALK_INSTRUCTIONS, raygen),
        allow(MAY_STACK, RAYGEN_STACK_INSTRUCTIONS, raygen),
        allow(
            MAY_PARTICLES | MAY_LOCAL_LIGHTS | MAY_MEDIUM,
            RAYGEN_FRONT_INSTRUCTIONS,
            raygen,
        ),
        allow(MAY_PROXIES, RAYGEN_PROXY_INSTRUCTIONS, raygen),
        allow(MAY_LENS | MAY_BACKGROUND, RAYGEN_LENS_INSTRUCTIONS, raygen),
        allow(MAY_DISPERSION, RAYGEN_DISPERSION_INSTRUCTIONS, raygen),
        // Where the paths may meet a plane stack, a stage that runs a vertex holds its re-run for
        // a bent job, and a layer's stage the stack question's answer and a crossing's track.
        allow(MAY_STACK, STACK_VERTEX_INSTRUCTIONS, runs_vertex),
        allow(MAY_STACK, STACK_LAYER_INSTRUCTIONS, layer),
        // Where the paths may gather caustic photons, a stage that runs a vertex holds whether it
        // is a chain vertex, the light its paths find past chain vertices and the gather's call; a
        // hit, a photon's vertex where the pipeline runs photons and the call of the reference
        // map's gather where it may be the reference's.
        allow(MAY_CAUSTICS, CAUSTIC_HIT_INSTRUCTIONS, runs_vertex),
        allow(MAY_PHOTON, PHOTON_HIT_INSTRUCTIONS, hit),
        allow(MAY_REFERENCE, MAP_GATHER_HIT_INSTRUCTIONS, |s| {
            s.role == Role::Hit && s.may & MAY_CAUSTICS != 0
        }),
        // A hit of a view's camera ray writes the view's G-buffer too, but for a layer's, which
        // writes none.
        allow(MAY_TRACED, TRACED_HIT_INSTRUCTIONS, |s| {
            s.role == Role::Hit && s.kernel != Kernel::Layer && s.geometry.is_some()
        }),
        // A hit a camera ray may reach may be its path's first vertex, which draws with its
        // pixel's local sampler.
        allow(MAY_LOCAL, LOCAL_FIRST_HIT_INSTRUCTIONS, |s| {
            s.role == Role::Hit
                && s.geometry.is_some()
                && s.may & (MAY_LAYERS | MAY_TRACED | MAY_LENS | MAY_BACKGROUND) != 0
        }),
        allow(
            may2(SceneFeatures::COATED_EMITTERS),
            COATED_EMITTER_HIT_INSTRUCTIONS,
            hit,
        ),
        allow(MAY_EMITTER, EMITTER_HIT_INSTRUCTIONS, hit),
        allow(MAY_INTERIORS, INTERIOR_HIT_INSTRUCTIONS, hit),
        // The lobes of the kernel a hit comes from: the rich ones, the walk's beside them, or a
        // dielectric's.
        allow(MAY_LOBES, RICH_HIT_INSTRUCTIONS, |s| {
            s.role == Role::Hit && matches!(s.kernel, Kernel::Rich | Kernel::Walk)
        }),
        allow(MAY_WALK, WALK_HIT_INSTRUCTIONS, |s| {
            s.role == Role::Hit && s.kernel == Kernel::Walk
        }),
        allow(MAY_LAYERS, LAYER_HIT_INSTRUCTIONS, layer),
        // A layer's stage where an interior's medium may scatter holds the call of the vertex where
        // a real-time walk meets its surface, and where its paths may split by wavelength the
        // start of the path's wavelengths, the hero's IOR and the call that weighs the companions.
        allow(MAY_SCATTERING, WALK_LAYER_INSTRUCTIONS, layer),
        allow(MAY_DISPERSION, DISPERSION_LAYER_INSTRUCTIONS, layer),
        // What resolves a producer entry's triangle, in a stage made for its geometry where the
        // scene may hold producer entries.
        allow(
            may2(SceneFeatures::PRODUCERS),
            PRODUCER_HIT_INSTRUCTIONS,
            |s| s.role == Role::Hit && producer(s),
        ),
        allow(
            may2(SceneFeatures::PRODUCERS),
            PRODUCER_ANY_HIT_INSTRUCTIONS,
            |s| s.role == Role::AnyHit && producer(s),
        ),
    ]
}

/// The stages of a pipeline, each once, with its hit groups and what the scene's hit records
/// name of them.
struct Table {
    specs: Vec<Spec>,
    /// The raygen stage of each source by its id, where the pipeline runs the source.
    raygens: [Option<u32>; 4],
    misses: [u32; 2],
    /// The callable of a path's first vertex, where the pipeline runs paths of a view, and where
    /// it may show a debug view, the one of a debug view of the first vertex alone.
    first: Option<u32>,
    first_view: Option<u32>,
    /// The callable that starts a path in the interiors around the eye, where the paths may lie
    /// in one.
    eye_interior: Option<u32>,
    /// The callable of a path's vertex at a scatter in an interior's medium, where one may scatter.
    medium: Option<u32>,
    /// The callable of the vertex where a subsurface walk leaves, where surfaces may walk.
    walk_exit: Option<u32>,
    /// The callable that weighs a coated emitter's light by its coat and fuzz, where an emitter
    /// may have one.
    emitter_view: Option<u32>,
    /// The callable of a step of a directional job's walk through a plane stack, where the paths
    /// may meet one.
    stack_step: Option<u32>,
    /// The callable that takes what a path's ray meets on its way to its hit, where the paths may
    /// meet particles, sphere and rect lights or a linked medium.
    front: Option<u32>,
    /// The callable that marches the medium proxies with a volume a shadow ray passed, where the
    /// scene may hold medium proxies.
    proxy_march: Option<u32>,
    /// The callable that starts a thin lens's path in the interiors at the lens's point, where the
    /// view traces its camera rays through one and the paths may lie in an interior.
    eye_lens: Option<u32>,
    /// The callable of the caustic photons' gather, where paths of a view may gather them, and
    /// those of a photon's start and of its deposit, where the pipeline runs photons.
    gather: Option<u32>,
    photon_start: Option<u32>,
    deposit: Option<u32>,
    /// The callable of the reference map's gather, where paths of a view may gather one.
    map_gather: Option<u32>,
    /// The callable that weighs a dispersive layer's companion wavelengths, where the paths may
    /// meet a layer and split by wavelength.
    disperse: Option<u32>,
    /// The callable of the vertex where an interior's real-time walk meets its surface, where an
    /// interior's medium may scatter.
    walk_meets: Option<u32>,
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
        // The pipeline that may meet everything stands in while a scene's own compiles, and runs a
        // view that specializes nothing: each pin's group shades its surfaces with the stage that
        // serves any id, so the table holds one per kind and not one per linked type. A scene's
        // own mask is never all of it.
        let general = may == MAY_ALL && surfaces != 0;
        // Whether it runs paths of a view (its own, its background layer's or its update paths),
        // whose stages a pipeline that runs caustic photons alone holds none of.
        let viewed = may & (MAY_VIEW | MAY_BACKGROUND | MAY_UPDATE) != 0;
        let photons = may & MAY_PHOTON != 0;
        let emits = may & MAY_EMISSIVE != 0;
        // The stage made for a surface that may emit is the only one with `MAY_EMITTER`.
        let may = may & !MAY_EMITTER;
        // Whether the scene's surfaces may have a coat, a fuzz, a thin film or subsurface, and the
        // kernel of those lobes: the one that walks where they may.
        let lobes = may & MAY_LOBES != 0;
        let rich = if may & MAY_WALK != 0 {
            Kernel::Walk
        } else {
            Kernel::Rich
        };
        let folded = |entry, role| Spec {
            kernel: Kernel::Plain,
            entry,
            role,
            geometry: None,
            may,
            hit_surface: plain::Spec::default().hit_surface,
        };
        let mut raygens = [None; 4];
        for (source, bit, entry, role) in [
            (PATH_SOURCE_VIEW, MAY_VIEW, "paths", Role::Raygen),
            (
                PATH_SOURCE_BACKGROUND,
                MAY_BACKGROUND,
                "backgrounds",
                Role::Raygen,
            ),
            (PATH_SOURCE_UPDATE, MAY_UPDATE, "updates", Role::Raygen),
            (
                PATH_SOURCE_PHOTON,
                MAY_PHOTON,
                "photons",
                Role::PhotonRaygen,
            ),
        ] {
            if may & bit != 0 {
                raygens[source as usize] = Some(Self::stage(&mut specs, folded(entry, role)));
            }
        }
        let misses = [
            Self::stage(&mut specs, folded("miss_scatter", Role::Miss)),
            Self::stage(&mut specs, folded("miss_shadow", Role::Miss)),
        ];
        // Any surface of the G-buffer may be a path's first vertex.
        let first = viewed.then(|| {
            Self::stage(
                &mut specs,
                Spec {
                    kernel: if lobes { rich } else { Kernel::Plain },
                    may: may | if emits { MAY_EMITTER } else { 0 },
                    ..folded("first_vertex", Role::Hit)
                },
            )
        });
        let first_view = (viewed && may & MAY_DEBUG != 0).then(|| {
            Self::stage(
                &mut specs,
                Spec {
                    kernel: if lobes { rich } else { Kernel::Plain },
                    ..folded("first_view", Role::Hit)
                },
            )
        });
        let eye_interior = (viewed && may & MAY_INTERIORS != 0)
            .then(|| Self::stage(&mut specs, folded("eye_interior", Role::Hit)));
        let eye_lens = (viewed && may & MAY_INTERIORS != 0 && may & MAY_LENS != 0)
            .then(|| Self::stage(&mut specs, folded("eye_lens", Role::Hit)));
        // A photon scatters in no medium and walks no surface.
        let medium = (viewed && may & MAY_SCATTERING != 0)
            .then(|| Self::stage(&mut specs, folded("medium_vertex", Role::Medium)));
        let walk_exit = (viewed && may & MAY_WALK != 0).then(|| {
            Self::stage(
                &mut specs,
                Spec {
                    kernel: Kernel::Walk,
                    ..folded("walk_leaves", Role::Exit)
                },
            )
        });
        let coated = emits && lobes && may & may2(SceneFeatures::COATED_EMITTERS) != 0;
        let emitter_view = coated.then(|| {
            Self::stage(
                &mut specs,
                Spec {
                    kernel: rich,
                    ..folded("emitter_view", Role::View)
                },
            )
        });
        // A photon draws no light, so no job of its walks through a plane stack.
        let stack_step = (viewed && may & MAY_STACK != 0)
            .then(|| Self::stage(&mut specs, folded("stack_step", Role::Stack)));
        let gather = (viewed && may & MAY_CAUSTICS != 0)
            .then(|| Self::stage(&mut specs, folded("gather", Role::Gather)));
        let photon_start =
            photons.then(|| Self::stage(&mut specs, folded("caster_photon", Role::PhotonStart)));
        let deposit = photons.then(|| Self::stage(&mut specs, folded("deposit", Role::Deposit)));
        let map_gather = (viewed && may & MAY_CAUSTICS != 0 && may & MAY_REFERENCE != 0)
            .then(|| Self::stage(&mut specs, folded("map_gather", Role::MapGather)));
        let disperse = (may & MAY_DISPERSION != 0 && may & MAY_LAYERS != 0).then(|| {
            Self::stage(
                &mut specs,
                Spec {
                    kernel: Kernel::Layer,
                    ..folded("disperse", Role::Disperse)
                },
            )
        });
        let walk_meets = (viewed && may & MAY_SCATTERING != 0).then(|| {
            Self::stage(
                &mut specs,
                Spec {
                    kernel: Kernel::Layer,
                    ..folded("walk_meets", Role::Meets)
                },
            )
        });
        let front = (may & (MAY_PARTICLES | MAY_LOCAL_LIGHTS | MAY_MEDIUM) != 0)
            .then(|| Self::stage(&mut specs, folded("front", Role::Front)));
        let proxies = may & MAY_PROXIES != 0;
        let proxy_march =
            proxies.then(|| Self::stage(&mut specs, folded("volume_march", Role::ProxyMarch)));
        // `hit_surface` of the hit shaders of each pin: any id, then each id from 0 on. Where
        // no surface links every material shows id 0. A closest-hit reaches a linked type only
        // through its shading and an any-hit only through its coverage, so the shader of a type
        // that declares neither is the standard surface's, made for id 0.
        let reaches = |role| match role {
            Role::Hit => surface_declares::SHADING,
            _ => surface_declares::COVERAGE,
        };
        let pinned = |entry, role, pin: u32, geometry: Geometry| Spec {
            kernel: Kernel::Plain,
            entry,
            role,
            geometry: Some(geometry),
            may: may & !geometry.without(),
            hit_surface: if surfaces == 0 {
                0
            } else if pin == 0 {
                u32::MAX
            } else if pin >= 2 && kernels.surface_declares[pin as usize - 2] & reaches(role) == 0 {
                0
            } else {
                pin - 1
            },
        };
        // Whether the closest-hit of pin `pin` shades a surface that may emit, where it is the
        // one made for materials that may (`emissive`). A linked type that declares its emission
        // has it in its one stage, which every material of its id takes. The standard surface's
        // materials that emit by their own factor take a stage of their own, so the others' holds
        // none of it; a linked type's that do share their type's, unless the type shades every
        // hit itself, where no factor of its records shows.
        let factors = may & may2(SceneFeatures::EMISSION_FACTORS) != 0;
        let linked_factors = may & may2(SceneFeatures::LINKED_EMISSION_FACTORS) != 0;
        // A photon takes no light of what it meets: a pipeline of photons alone shades an emitter as
        // any other surface.
        let emitter = |pin: u32, emissive: bool| {
            if !emits || !viewed {
                false
            } else if surfaces == 0 || pin == 1 {
                emissive && factors
            } else if pin == 0 {
                true
            } else {
                let declared = kernels.surface_declares[pin as usize - 2];
                let falls_back = declared & surface_declares::ALWAYS_SHADES == 0;
                declared & surface_declares::EMISSION != 0 || falls_back && linked_factors
            }
        };
        // Whether the closest-hit of pin `pin` comes from the kernel with the rich lobes, where
        // it is the one made for materials that may have one (`rich`): as its emission, a linked
        // type that declares a lobe has it in its one stage, and the standard surface's
        // materials whose weights give them one take a stage of their own.
        let factors = may & may2(SceneFeatures::LOBE_FACTORS) != 0;
        let linked_factors = may & may2(SceneFeatures::LINKED_LOBE_FACTORS) != 0;
        let rich_lobes = surface_declares::SUBSURFACE
            | surface_declares::COAT
            | surface_declares::FUZZ
            | surface_declares::THIN_FILM;
        let lobed = |pin: u32, rich: bool| {
            if !lobes {
                false
            } else if surfaces == 0 || pin == 1 {
                rich && factors
            } else if pin == 0 {
                true
            } else {
                let declared = kernels.surface_declares[pin as usize - 2];
                let falls_back = declared & surface_declares::ALWAYS_SHADES == 0;
                declared & rich_lobes != 0 || falls_back && linked_factors
            }
        };
        // Whether pin `pin` may shade a dielectric layer: the standard surface, any id, a linked
        // type that declares its transmission, and one that leaves hits to the standard surface
        // where such a type's records transmit by their own weights.
        let layers = may & MAY_LAYERS != 0;
        let linked_transmits = may & may2(SceneFeatures::LINKED_TRANSMISSION_FACTORS) != 0;
        let transmits = |pin: u32| {
            if surfaces == 0 || pin <= 1 {
                true
            } else {
                let declared = kernels.surface_declares[pin as usize - 2];
                let falls_back = declared & surface_declares::ALWAYS_SHADES == 0;
                declared & surface_declares::TRANSMISSION != 0 || falls_back && linked_transmits
            }
        };
        let producers = may & may2(SceneFeatures::PRODUCERS) != 0;
        let strands = producers && may & may2(SceneFeatures::STRANDS) != 0;
        let mut hit_groups = Vec::new();
        let pins = groups.pins();
        // Each pinned kind's groups one after another by pin, the first named in `groups`.
        // Each kind: its groups, shaders and geometry, whether the table holds it, and whether its
        // materials may emit and may have a rich lobe. A rich material that emits takes the rich
        // kind.
        for (first, hit, any_hit, geometry, linked, emissive, lobed_kind, layer) in [
            (
                &mut groups.path,
                Some("surface_hit"),
                Some("cover_hit"),
                Geometry::Copies,
                true,
                false,
                false,
                false,
            ),
            (
                &mut groups.emissive_path,
                Some("surface_hit"),
                Some("cover_hit"),
                Geometry::Copies,
                emits,
                true,
                false,
                false,
            ),
            (
                &mut groups.rich_path,
                Some("surface_hit"),
                Some("cover_hit"),
                Geometry::Copies,
                lobes,
                true,
                true,
                false,
            ),
            (
                &mut groups.producer_path,
                Some("surface_hit"),
                Some("cover_hit"),
                Geometry::Producers,
                producers,
                false,
                false,
                false,
            ),
            (
                &mut groups.producer_emissive_path,
                Some("surface_hit"),
                Some("cover_hit"),
                Geometry::Producers,
                producers && emits,
                true,
                false,
                false,
            ),
            (
                &mut groups.producer_rich_path,
                Some("surface_hit"),
                Some("cover_hit"),
                Geometry::Producers,
                producers && lobes,
                true,
                true,
                false,
            ),
            (
                &mut groups.layer_path,
                Some("surface_hit"),
                Some("cover_hit"),
                Geometry::Copies,
                layers,
                true,
                false,
                true,
            ),
            (
                &mut groups.producer_layer_path,
                Some("surface_hit"),
                Some("cover_hit"),
                Geometry::Producers,
                producers && layers,
                true,
                false,
                true,
            ),
            // A strand is opaque: its rays take no any-hit. Its kind serves all of its materials,
            // which may emit or have a rich lobe.
            (
                &mut groups.strand_path,
                Some("strand_hit"),
                None,
                Geometry::Strands,
                strands,
                true,
                true,
                false,
            ),
            (
                &mut groups.shadow,
                None,
                Some("shadow_hit"),
                Geometry::Triangles,
                true,
                false,
                false,
                false,
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
                let pin = if general { 0 } else { pin };
                hit_groups.push(HitGroup {
                    closest_hit: hit.map(|e| {
                        let spec = pinned(e, Role::Hit, pin, geometry);
                        let bit = if emitter(pin, emissive) {
                            MAY_EMITTER
                        } else {
                            0
                        };
                        let kernel = if layer && transmits(pin) {
                            Kernel::Layer
                        } else if lobed(pin, lobed_kind || layer) {
                            rich
                        } else {
                            Kernel::Plain
                        };
                        Self::stage(
                            &mut specs,
                            Spec {
                                kernel,
                                may: spec.may | bit,
                                ..spec
                            },
                        )
                    }),
                    any_hit: any_hit
                        .map(|e| Self::stage(&mut specs, pinned(e, Role::AnyHit, pin, geometry))),
                    intersection: None,
                });
            }
        }
        // A path ray whose mask takes particles or lights' proxies marks its payload where it meets
        // one, and goes on.
        if may & MAY_PARTICLES != 0 {
            groups.front = Some(hit_groups.len() as u32);
            hit_groups.push(HitGroup {
                closest_hit: None,
                any_hit: Some(Self::stage(
                    &mut specs,
                    folded("front_met_hit", Role::FrontAnyHit),
                )),
                intersection: None,
            });
        }
        if may & MAY_LOCAL_LIGHTS != 0 {
            groups.light_front = Some(hit_groups.len() as u32);
            hit_groups.push(HitGroup {
                closest_hit: None,
                any_hit: Some(Self::stage(
                    &mut specs,
                    folded("light_met_hit", Role::FrontAnyHit),
                )),
                intersection: Some(Self::stage(
                    &mut specs,
                    folded("light_box_hit", Role::LightBox),
                )),
            });
        }
        // A medium proxy is a box under shadow rays alone.
        if proxies {
            groups.medium_shadow = Some(hit_groups.len() as u32);
            hit_groups.push(HitGroup {
                closest_hit: None,
                any_hit: Some(Self::stage(
                    &mut specs,
                    folded("proxy_shadow_hit", Role::ProxyAnyHit),
                )),
                intersection: Some(Self::stage(
                    &mut specs,
                    folded("box_medium_hit", Role::ProxyBox),
                )),
            });
        }
        Self {
            specs,
            raygens,
            misses,
            first,
            first_view,
            eye_interior,
            medium,
            walk_exit,
            emitter_view,
            stack_step,
            front,
            proxy_march,
            eye_lens,
            gather,
            photon_start,
            deposit,
            map_gather,
            disperse,
            walk_meets,
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
    /// Whether it comes from a kernel with the coat, fuzz, thin film and thin wall lobes.
    pub rich: bool,
    pub instructions: u32,
    /// The cap on orr's own code in it; none for a stage made for a linked surface.
    pub cap: Option<u32>,
    /// What the kernels' linked types add to it: its size beyond the same stage of orr's
    /// kernels with nothing linked, where a stage made for a linked surface is the standard
    /// surface's. None for those kernels themselves.
    pub linked: Option<i64>,
    /// Of a stage made for a linked surface, what it holds beyond the stage of its entry,
    /// geometry and constants made for the standard surface: the surface's own code, less the
    /// standard code it replaces.
    pub own: Option<i64>,
}

/// The stages of the pipeline of paths that may meet `may`, from `kernels`: each folded with its
/// constants once, for the sizes and for the driver, and what each hands the driver.
struct Folded {
    table: Table,
    /// Each stage's code as the driver is handed it; none where that is its module's code.
    code: Vec<Option<Vec<u32>>>,
    sizes: Vec<StageSize>,
}

impl Folded {
    /// Folds every stage the sizes need on `threads`: the table's, and those they are measured
    /// against.
    fn new(kernels: &KernelSet, may: u64, threads: CompileThreads<'_>) -> Result<Self, Error> {
        let table = Table::new(may, kernels);
        let specs = &table.specs;
        let modules: Vec<&'static orr_shaders::ShaderModule> = specs
            .iter()
            .map(|spec| module(kernels, spec.kernel.own()))
            .collect::<Result<_, Error>>()?;
        // The folds, each once: the table's stages, then the stages they are measured against.
        let mut folds: Vec<_> = modules.iter().copied().zip(specs.iter().copied()).collect();
        let mut fold = |module: &'static orr_shaders::ShaderModule, spec: Spec| {
            let at = folds
                .iter()
                .position(|&(m, s)| std::ptr::eq(m, module) && s == spec);
            at.unwrap_or_else(|| {
                folds.push((module, spec));
                folds.len() - 1
            })
        };
        // Kernels that link a type compile the path stages anew, with the types' code in them:
        // a stage is measured against the same stage of orr's own kernel for the standard surface.
        let plain: Vec<Option<usize>> = specs
            .iter()
            .enumerate()
            .map(|(at, spec)| {
                let own = spec.kernel.own();
                let standard = Spec {
                    hit_surface: match spec.geometry {
                        Some(_) => 0,
                        None => spec.hit_surface,
                    },
                    ..*spec
                };
                (!std::ptr::eq(modules[at], own)).then(|| fold(own, standard))
            })
            .collect();
        // A stage made for a linked surface is measured against the standard surface's stage of
        // the same constants, which the table need not hold: the standard surface's materials
        // may emit by no factor where a linked type emits.
        let standard: Vec<Option<usize>> = specs
            .iter()
            .map(|spec| {
                spec.linked(kernels.surfaces)
                    .then(|| {
                        let standard = Spec {
                            hit_surface: 0,
                            ..*spec
                        };
                        Ok(fold(module(kernels, standard.kernel.own())?, standard))
                    })
                    .transpose()
            })
            .collect::<Result<_, Error>>()?;
        let constants: Vec<_> = folds.iter().map(|(_, spec)| spec.constants()).collect();
        let stages = folds
            .iter()
            .zip(&constants)
            .map(|(&(module, spec), (bytes, entries))| {
                ShaderStage::new(module, spec.entry, SpecData { entries, bytes })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut code = threads.map(stages.len(), |i| stages[i].fold());
        let handed: Vec<u32> = code
            .iter()
            .zip(&stages)
            .map(|(code, stage)| handed_instructions(code.as_deref().unwrap_or(stage.spirv)))
            .collect();
        let sizes = specs
            .iter()
            .enumerate()
            .map(|(at, spec)| {
                let instructions = handed[at];
                let beyond =
                    |i: Option<usize>| i.map(|i| i64::from(instructions) - i64::from(handed[i]));
                StageSize {
                    entry: spec.entry,
                    role: spec.role,
                    geometry: spec.geometry.map_or("", Geometry::label),
                    hit_surface: spec.hit_surface,
                    rich: matches!(spec.kernel, Kernel::Rich | Kernel::Walk),
                    instructions,
                    cap: spec.cap(kernels.surfaces),
                    linked: beyond(plain[at]),
                    own: beyond(standard[at]),
                }
            })
            .collect();
        code.truncate(specs.len());
        Ok(Self { table, code, sizes })
    }
}

/// What each stage of the pipeline of paths that may meet `may` hands the driver, from
/// `kernels`, and what of it is orr's own.
pub fn path_stage_sizes(kernels: &KernelSet, may: u64) -> Result<Vec<StageSize>, Error> {
    Folded::new(kernels, may, CompileThreads::default()).map(|f| f.sizes)
}

/// Each of `sizes` whose own code, without what linked types add to it, is over its cap, and the
/// pipeline's total, the linked types' code among it, where that is over [`PIPELINE_CAP`].
pub fn over_cap(sizes: &[StageSize]) -> Vec<String> {
    let over = |size: i64, cap: u32| 100.0 * (size as f64 / f64::from(cap) - 1.0);
    let mut breaches: Vec<String> = sizes
        .iter()
        .filter_map(|s| {
            let own = i64::from(s.instructions) - s.linked.unwrap_or(0);
            let cap = s.cap.filter(|&cap| own > i64::from(cap))?;
            Some(format!(
                "stage {}{} hands the driver {own} instructions of orr's own, {:.1} % over its \
                 {cap} as a {:?}",
                s.entry,
                s.geometry,
                over(own, cap),
                s.role
            ))
        })
        .collect();
    let total: u32 = sizes.iter().map(|s| s.instructions).sum();
    if total > PIPELINE_CAP {
        breaches.push(format!(
            "{} stages hand the driver {total} instructions, {:.1} % over the pipeline's \
             {PIPELINE_CAP}",
            sizes.len(),
            over(total.into(), PIPELINE_CAP)
        ));
    }
    breaches
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
        let started = Instant::now();
        let Folded { table, code, sizes } = Folded::new(kernels, may, threads)?;
        let folded = started.elapsed().as_secs_f64();
        let name = format!("path pipeline {may:#x}");
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
        // The caps are a growth budget that tests hold orr's own code to, not a limit on what a
        // scene may run: a pipeline over one still compiles.
        for breach in over_cap(&sizes) {
            log::warn!("{name}: {breach}");
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
            .zip(&code)
            .map(|(((spec, (bytes, entries)), label), code)| {
                let m = module(kernels, spec.kernel.own())?;
                let stage = ShaderStage::new(m, spec.entry, SpecData { entries, bytes })?;
                Ok(stage
                    .folded(code.as_deref().unwrap_or(m.spirv))
                    .labeled(label))
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
            callables: &[
                table.first.map(ShaderRecord::new),
                table.first_view.map(ShaderRecord::new),
                table.eye_interior.map(ShaderRecord::new),
                table.medium.map(ShaderRecord::new),
                table.walk_exit.map(ShaderRecord::new),
                table.emitter_view.map(ShaderRecord::new),
                table.stack_step.map(ShaderRecord::new),
                table.front.map(ShaderRecord::new),
                table.proxy_march.map(ShaderRecord::new),
                table.eye_lens.map(ShaderRecord::new),
                table.gather.map(ShaderRecord::new),
                table.photon_start.map(ShaderRecord::new),
                table.deposit.map(ShaderRecord::new),
                table.map_gather.map(ShaderRecord::new),
                table.disperse.map(ShaderRecord::new),
                table.walk_meets.map(ShaderRecord::new),
            ],
            max_recursion: 1,
            flags: RayPipelineFlags::scene(device.caps()),
            threads,
        };
        let pipeline = device.create_ray_pipeline(&desc, &name)?;
        log::info!(
            "{name}: {} stages, {} instructions handed over, made in {:.1} s, {folded:.1} s of \
             it folding{}",
            stages.len(),
            sizes.iter().map(|s| s.instructions).sum::<u32>(),
            started.elapsed().as_secs_f64(),
            driver_cost(pipeline.compile_cost())
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
        urgency: Urgency,
    ) -> Result<PathPipeline, Error> {
        // The driver compiles the pipeline's shaders on the loading workers that join it, ahead
        // of the queued jobs where a frame waits on it.
        let Some(pool) = loading else {
            return PathPipeline::new(device, &self.kernels, CompileThreads::default(), may);
        };
        let spawn = |task: Box<dyn FnOnce() + Send>| pool.spawn_joining(urgency, task);
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

    /// `path_may` and `path_may2` of the pipelines of the bench's synthetic walk, as it logs them:
    /// the one of its view's and update paths and the one of its caustic photons. Both are over
    /// cooked clusters and producer entries, with interiors, layers, particles, coverage,
    /// caustics, a plane stack, froxels, a medium, directional and emissive lights; the view's
    /// paths have local light samplers and the cache too.
    const BENCH_MAY: [u64; 2] = [0xfffe_1430_2378_702d, 0xfffe_1430_2378_2045];

    /// The caps are a growth budget on orr's own code, held here and not at a pipeline's
    /// creation: on the pipeline that may meet everything, on the bench's, and on the first with
    /// each allowance taken away in turn, so the room one feature's allowance leaves at the widest
    /// mask hides no growth in the rest of a stage.
    #[test]
    fn the_stages_of_orrs_own_kernels_hold_their_caps() {
        let mut bits: Vec<u64> = allowances().iter().map(|a| a.bits).collect();
        bits.sort_unstable();
        bits.dedup();
        let masks = [MAY_ALL]
            .into_iter()
            .chain(BENCH_MAY)
            .chain(bits.into_iter().map(|bits| MAY_ALL & !bits));
        let mut breaches = Vec::new();
        for may in masks {
            let sizes = path_stage_sizes(&KernelSet::default(), may).unwrap();
            breaches.extend(
                over_cap(&sizes)
                    .into_iter()
                    .map(|b| format!("{may:#018x}: {b}")),
            );
        }
        assert!(breaches.is_empty(), "{}", breaches.join("\n"));
    }
}
