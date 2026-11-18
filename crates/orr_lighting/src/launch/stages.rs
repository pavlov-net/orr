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
use orr_scene::{HitGroups, Loading, SceneFeatures, Urgency};
use orr_shaders::surface_declares;

use crate::mask_sets::Members;
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
    /// leaves the interior.
    Medium,
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

    /// The most instructions the stage may hand the driver: its role's, with what a producer
    /// entry's geometry adds to a hit or an any-hit shader. A stage made for a linked surface
    /// has none of its own: it holds the application's code beside the standard stage's.
    fn cap(&self, surfaces: u32) -> Option<u32> {
        if self.linked(surfaces) {
            return None;
        }
        let producers = !matches!(self.geometry, None | Some(Geometry::Copies));
        // Where the paths may meet a plane stack, a stage that runs a vertex holds its re-run for
        // a bent job, and a layer's stage the stack question's answer and a crossing's track.
        let stacks = self.may & MAY_STACK != 0;
        let vertex_stack = if stacks { STACK_VERTEX_INSTRUCTIONS } else { 0 };
        // Where the paths may gather caustic photons, a stage that runs a vertex holds whether it
        // is a chain vertex, the light its paths find past chain vertices and the gather's call; a
        // hit, a photon's vertex where the pipeline runs photons and the call of the reference
        // map's gather where it may be the reference's.
        let caustics = if self.may & MAY_CAUSTICS != 0 {
            CAUSTIC_HIT_INSTRUCTIONS
        } else {
            0
        };
        let photon = if self.may & MAY_PHOTON != 0 {
            PHOTON_HIT_INSTRUCTIONS
        } else {
            0
        };
        let map = if self.may & MAY_CAUSTICS != 0 && self.may & MAY_REFERENCE != 0 {
            MAP_GATHER_HIT_INSTRUCTIONS
        } else {
            0
        };
        let (base, producer) = match self.role {
            Role::Raygen => {
                let layers = self.may & MAY_LAYERS != 0;
                let walks = self.may & (MAY_SCATTERING | MAY_WALK) != 0;
                let front = self.may & (MAY_PARTICLES | MAY_LOCAL_LIGHTS | MAY_MEDIUM) != 0;
                let proxies = self.may & MAY_PROXIES != 0;
                let lens = self.may & (MAY_LENS | MAY_BACKGROUND) != 0;
                let dispersion = self.may & MAY_DISPERSION != 0;
                return Some(
                    RAYGEN_INSTRUCTIONS
                        + if layers { RAYGEN_LAYER_INSTRUCTIONS } else { 0 }
                        + if walks { RAYGEN_WALK_INSTRUCTIONS } else { 0 }
                        + if stacks { RAYGEN_STACK_INSTRUCTIONS } else { 0 }
                        + if front { RAYGEN_FRONT_INSTRUCTIONS } else { 0 }
                        + if proxies {
                            RAYGEN_PROXY_INSTRUCTIONS
                        } else {
                            0
                        }
                        + if lens { RAYGEN_LENS_INSTRUCTIONS } else { 0 }
                        + if dispersion {
                            RAYGEN_DISPERSION_INSTRUCTIONS
                        } else {
                            0
                        },
                );
            }
            Role::Miss => return Some(MISS_INSTRUCTIONS),
            Role::Medium => return Some(MEDIUM_VERTEX_INSTRUCTIONS + vertex_stack + caustics),
            Role::Exit => return Some(WALK_EXIT_INSTRUCTIONS + vertex_stack + caustics),
            Role::PhotonRaygen => return Some(RAYGEN_PHOTON_INSTRUCTIONS),
            Role::PhotonStart => return Some(PHOTON_START_INSTRUCTIONS),
            Role::Deposit => return Some(DEPOSIT_INSTRUCTIONS),
            Role::Gather => return Some(GATHER_INSTRUCTIONS),
            Role::MapGather => return Some(MAP_GATHER_INSTRUCTIONS),
            Role::Disperse => return Some(DISPERSE_INSTRUCTIONS),
            Role::View => return Some(EMITTER_VIEW_INSTRUCTIONS),
            Role::Stack => return Some(STACK_STEP_INSTRUCTIONS),
            Role::Front => return Some(FRONT_INSTRUCTIONS),
            Role::FrontAnyHit => return Some(FRONT_ANY_HIT_INSTRUCTIONS),
            Role::LightBox => return Some(LIGHT_BOX_INSTRUCTIONS),
            Role::ProxyBox => return Some(PROXY_INTERSECTION_INSTRUCTIONS),
            Role::ProxyAnyHit => return Some(PROXY_ANY_HIT_INSTRUCTIONS),
            Role::ProxyMarch => return Some(PROXY_MARCH_INSTRUCTIONS),
            Role::Hit => (HIT_INSTRUCTIONS, PRODUCER_HIT_INSTRUCTIONS),
            Role::AnyHit => (ANY_HIT_INSTRUCTIONS, PRODUCER_ANY_HIT_INSTRUCTIONS),
        };
        // A hit of a view's camera ray writes the view's G-buffer too, but for a layer's, which
        // writes none.
        let primary = self.role == Role::Hit
            && self.kernel != Kernel::Layer
            && self.geometry.is_some()
            && self.may & MAY_TRACED != 0;
        let traced = if primary { TRACED_HIT_INSTRUCTIONS } else { 0 };
        // A hit a camera ray may reach may be its path's first vertex, which draws with its
        // pixel's local sampler.
        let local = self.role == Role::Hit
            && self.geometry.is_some()
            && self.may & MAY_LOCAL != 0
            && self.may & (MAY_LAYERS | MAY_TRACED | MAY_LENS | MAY_BACKGROUND) != 0;
        let local = if local {
            LOCAL_FIRST_HIT_INSTRUCTIONS
        } else {
            0
        };
        let coated = self.role == Role::Hit && self.may & may2(SceneFeatures::COATED_EMITTERS) != 0;
        let coated = if coated {
            COATED_EMITTER_HIT_INSTRUCTIONS
        } else {
            0
        };
        let emitter = self.role == Role::Hit && self.may & MAY_EMITTER != 0;
        let emitter = if emitter { EMITTER_HIT_INSTRUCTIONS } else { 0 };
        let interiors = self.role == Role::Hit && self.may & MAY_INTERIORS != 0;
        let interiors = if interiors {
            INTERIOR_HIT_INSTRUCTIONS
        } else {
            0
        };
        let rich = match (self.role, self.kernel) {
            (Role::Hit, Kernel::Rich) => RICH_HIT_INSTRUCTIONS,
            (Role::Hit, Kernel::Walk) => RICH_HIT_INSTRUCTIONS + WALK_HIT_INSTRUCTIONS,
            (Role::Hit, Kernel::Layer) => LAYER_HIT_INSTRUCTIONS,
            _ => 0,
        };
        let stack = match (self.role, self.kernel) {
            (Role::Hit, Kernel::Layer) if stacks => vertex_stack + STACK_LAYER_INSTRUCTIONS,
            (Role::Hit, _) => vertex_stack,
            _ => 0,
        };
        // A layer's stage where its paths may split by wavelength holds the start of the path's
        // wavelengths, the hero's IOR and the call that weighs the companions.
        let dispersion = match (self.role, self.kernel) {
            (Role::Hit, Kernel::Layer) if self.may & MAY_DISPERSION != 0 => {
                DISPERSION_LAYER_INSTRUCTIONS
            }
            _ => 0,
        };
        Some(
            base + if producers { producer } else { 0 }
                + traced
                + local
                + emitter
                + interiors
                + rich
                + coated
                + stack
                + dispersion
                + if self.role == Role::Hit {
                    caustics + photon + map
                } else {
                    0
                },
        )
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
        let front = (may & (MAY_PARTICLES | MAY_LOCAL_LIGHTS | MAY_MEDIUM) != 0)
            .then(|| Self::stage(&mut specs, folded("front", Role::Front)));
        let proxies = may & MAY_PROXIES != 0;
        let proxy_march =
            proxies.then(|| Self::stage(&mut specs, folded("volume_march", Role::ProxyMarch)));
        // `hit_surface` of the hit shaders of each pin: any id, then each id from 0 on. Where
        // no surface links every material shows id 0.
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
    /// The most of orr's own code it may hand the driver; none for a stage made for a linked
    /// surface.
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
            ],
            max_recursion: 1,
            // A closest-hit or callable a raygen runs calls the gather, map gather, disperse
            // and emitter view callables.
            call_depth: 2,
            flags: RayPipelineFlags::scene(device.caps()),
            threads,
        };
        let pipeline = device.create_ray_pipeline(&desc, &name)?;
        log::info!(
            "{name}: {} stages, {} instructions handed over, made in {:.1} s, {folded:.1} s of \
             it folding",
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

    /// A stage over its cap compiles for longer than a cold start may wait, whatever the scene
    /// holds: the pipeline that may meet everything is the one a session runs. It holds the
    /// camera rays' hits, with their allowance; the same pipeline without them holds the plain
    /// caps.
    #[test]
    fn the_stages_of_the_pipeline_that_may_meet_everything_hold_their_caps() {
        for may in [u64::MAX, !MAY_TRACED] {
            let sizes = path_stage_sizes(&KernelSet::default(), may).unwrap();
            assert_eq!(over_cap(&sizes), None, "{may:#x}");
        }
    }
}
