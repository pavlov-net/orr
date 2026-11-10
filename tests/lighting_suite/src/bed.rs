//! The lighting test bed: analytic scenes cooked, mounted and placed through orr_scene's
//! public API, rendered from a fixed camera, with a kernel's radiance accumulated over
//! frames and compared with FLIP.

use std::path::PathBuf;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

pub use openpbr_ref::walk::fresnel;
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, GridInput, GridVoxels, LodMode, Manifest, MeshInput,
    MeshSettings, Source, TextureSettings, Usage, VolumeGrids, VolumeInput, VolumeSettings,
};
use orr_gpu::{
    Allocation, CommandList, ComputePipeline, Format, GpuPtr, Hazards, KernelSet, MemoryKind,
    Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory, TextureRegion,
    TextureUsage,
};
use orr_lighting::{
    BsdfTables, CompilingKernels, DirectionalLight, DiscProfile, DofBackgroundPass,
    EnvironmentMapGpu, Light, LightId, LightingKernels, LightingView, ParticleDraw, PathEntry,
    PathNeeds, ReferenceSettings, SceneLighting, SceneLightingSettings, ScenePass, ViewLighting,
    ViewPass,
};
use orr_math::{Affine3, DVec3, Mat3, Quat, UVec2, Vec2, Vec3};
use orr_pack::material::{Material, slot};
use orr_pack::{AssetId, Span, Target};
use orr_scene::testing::{FlipError, Image, Pixels, flip};
use orr_scene::{
    ClusterTopology, FrameDesc, FrameTime, GroupWriter, InstanceFlags, InstancesDesc, Lens,
    LoadState, MaterialGpu, MaterialId, MediumProxy, MediumProxyId, MountId, Opacity, PackSource,
    ProducerDesc, ProducerId, ProducerUpdate, Projection, RasterView, RendererDesc, SceneCompute,
    SceneDesc, SceneId, SceneRoot, SceneWriter, Scenes, SkinStateGpu, SkinnedDesc, TextureFilter,
    TransformGroupDesc, TransformGroupId, ViewArena, ViewDesc, ViewHistoryId, VolumeRef,
    material_flags,
};
use orr_shaders::bytemuck::Zeroable;

use crate::shaders::{accumulate, exposure, flat_color, guide_read, particle_quads};
use crate::{TestTexture, Workers, device, linear_clamp, pipeline, read, render, texture, upload};

/// Render extent of every bed view.
pub const EXTENT: [u32; 2] = [128, 96];

/// Near plane of every bed view, m.
pub const NEAR: f32 = 0.05;

const PIXELS: u64 = (EXTENT[0] * EXTENT[1]) as u64;

/// How far away the bed's copies are ray traced from, m.
pub const RT_RANGE: f32 = 1000.0;

/// The scenes lighting is checked on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Analytic {
    /// A white diffuse sphere (albedo 1, no specular) in a uniform environment.
    Furnace,
    /// A diffuse Cornell box, lit only by an emissive quad under its ceiling.
    CornellBox,
    /// A tilted emissive quad over a gray ground, nothing else lit.
    EmissiveQuad,
    /// A white sphere on a gray ground under a sun and a uniform sky.
    SunAndGround,
    /// An upright alpha-tested quad of rings over the ground, sun and sky.
    AlphaQuad,
    /// A smooth glass slab over the ground, sun and sky.
    DielectricSlab,
    /// A subsurface slab on the ground, sun and sky.
    SubsurfaceSlab,
}

impl Analytic {
    pub const ALL: [Analytic; 7] = [
        Analytic::Furnace,
        Analytic::CornellBox,
        Analytic::EmissiveQuad,
        Analytic::SunAndGround,
        Analytic::AlphaQuad,
        Analytic::DielectricSlab,
        Analytic::SubsurfaceSlab,
    ];

    /// What the scene holds and where its camera stands.
    pub fn spec(self) -> Spec {
        let at = |model, at: Vec3| Item {
            model,
            linear: Mat3::IDENTITY,
            at,
        };
        let floor = at(FLOOR, Vec3::ZERO);
        let lit = |copies: Vec<Item>, eye: Vec3, target: Vec3| Spec {
            copies: [vec![floor], copies].concat(),
            lights: vec![Light::Directional(SUN)],
            environment: Some(SKY),
            eye,
            target,
            fov_y: 0.8,
            focus: None,
        };
        match self {
            Analytic::Furnace => Spec {
                copies: vec![at(WHITE_SPHERE, Vec3::ZERO)],
                lights: Vec::new(),
                environment: Some(FURNACE_RADIANCE),
                eye: Vec3::new(0.0, 0.0, 4.0),
                target: Vec3::ZERO,
                fov_y: 0.6,
                focus: None,
            },
            Analytic::CornellBox => Spec {
                copies: vec![
                    at(CORNELL, Vec3::ZERO),
                    Item {
                        model: TALL_BOX,
                        linear: Mat3::from_rotation_y(0.3),
                        at: Vec3::new(-0.35, 0.6, -0.35),
                    },
                    Item {
                        model: SHORT_BOX,
                        linear: Mat3::from_rotation_y(-0.3),
                        at: Vec3::new(0.35, 0.3, 0.3),
                    },
                    at(EMITTER, Vec3::new(0.0, 1.98, 0.0)),
                ],
                lights: Vec::new(),
                environment: None,
                eye: Vec3::new(0.0, 1.0, 3.4),
                target: Vec3::new(0.0, 1.0, 0.0),
                fov_y: 0.75,
                focus: None,
            },
            Analytic::EmissiveQuad => Spec {
                copies: vec![
                    floor,
                    Item {
                        model: EMITTER,
                        linear: Mat3::from_rotation_x(-0.6),
                        at: Vec3::new(0.0, 1.0, 0.0),
                    },
                ],
                lights: Vec::new(),
                environment: None,
                eye: Vec3::new(0.0, 1.5, 4.0),
                target: Vec3::new(0.0, 0.5, 0.0),
                fov_y: 0.8,
                focus: None,
            },
            Analytic::SunAndGround => lit(
                vec![at(WHITE_SPHERE, Vec3::new(0.0, 1.0, 0.0))],
                Vec3::new(0.0, 2.0, 5.0),
                Vec3::new(0.0, 0.7, 0.0),
            ),
            Analytic::AlphaQuad => lit(
                vec![at(LEAF, Vec3::new(0.0, 1.1, 0.0))],
                Vec3::new(0.5, 1.5, 3.5),
                Vec3::new(0.0, 0.9, 0.0),
            ),
            Analytic::DielectricSlab => lit(
                vec![at(GLASS_SLAB, Vec3::new(0.0, 0.5, 0.0))],
                Vec3::new(0.0, 2.0, 3.5),
                Vec3::new(0.0, 0.4, 0.0),
            ),
            Analytic::SubsurfaceSlab => lit(
                vec![at(SKIN_SLAB, Vec3::new(0.0, 0.15, 0.0))],
                Vec3::new(0.0, 1.5, 3.0),
                Vec3::new(0.0, 0.2, 0.0),
            ),
        }
    }
}

/// One placed copy: a cooked model of the bed's pack, its rotation and translation.
#[derive(Clone, Copy, Debug)]
pub struct Item {
    pub model: &'static str,
    pub linear: Mat3,
    pub at: Vec3,
}

/// An analytic scene: copies, lights and a uniform environment (nits, linear Rec.2020;
/// none is black), and its camera at `eye` looking at `target`, y up. With a `focus` (a
/// point and pixels per radian), ray tracing picks each copy's cut as for a primary view
/// there; without one it traces every copy's finest cut.
#[derive(Clone, Debug)]
pub struct Spec {
    pub copies: Vec<Item>,
    pub lights: Vec<Light>,
    pub environment: Option<[f32; 3]>,
    pub eye: Vec3,
    pub target: Vec3,
    /// Radians.
    pub fov_y: f32,
    pub focus: Option<(Vec3, f32)>,
}

/// A particle of the bed's particle entry: a square of half edge `size` m around `center`,
/// facing the camera.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub center: Vec3,
    pub size: f32,
}

/// What the bed's particles are made of: albedo (linear Rec.2020), emission (nits), opacity
/// and the Henyey-Greenstein parameter of their response to directional light.
#[derive(Clone, Copy, Debug)]
pub struct ParticleLook {
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
    pub opacity: f32,
    pub anisotropy: f32,
}

/// The double-sided material of a particle of `look`.
fn particle_material(look: &ParticleLook) -> MaterialGpu {
    let mut m = MaterialGpu::zeroed();
    m.base_color = look.albedo;
    m.base_weight = 1.0;
    m.geometry_opacity = look.opacity;
    m.emission_color = look.emission;
    m.emission_luminance = 1.0;
    m.flags = material_flags::DOUBLE_SIDED;
    if look.emission.iter().any(|&e| e > 0.0) {
        m.flags |= material_flags::EMISSIVE;
    }
    m
}

/// The bed's particle entry: its particles, the producer kernel that emits them each frame
/// and the stock draw at the transparency point.
struct Particles {
    list: Allocation<[f32; 4]>,
    count: u32,
    emit: ComputePipeline,
    draw: ParticleDraw,
    anisotropy: f32,
}

/// The producer entry of the bed's particles.
const PARTICLES: ProducerId = ProducerId(0);

/// `SceneDesc::producer_capacity` of every bed scene.
pub const PRODUCER_CAPACITY: u32 = 1;

/// A particle square's two triangles.
const PARTICLE_QUAD: [[u8; 3]; 2] = [[0, 1, 2], [0, 2, 3]];

pub const SUN: DirectionalLight = DirectionalLight {
    direction_to_light: Vec3::new(0.35043832, 0.8760958, 0.26282874),
    illuminance: Vec3::new(100_000.0, 98_000.0, 92_000.0),
    angular_radius: 4.65e-3,
    disc: DiscProfile::Uniform,
};

pub const SKY: [f32; 3] = [3000.0, 4000.0, 6000.0];
pub const FURNACE_RADIANCE: [f32; 3] = [1000.0; 3];

// Models of the bed's pack.
pub const FLOOR: &str = "floor";
/// [`FLOOR`]'s 10 m square as a diffuse surface of albedo [`WHITE_FLOOR_ALBEDO`] with no
/// specular lobe, which takes the caustic photons' irradiance exactly.
pub const WHITE_FLOOR: &str = "white floor";
pub const WHITE_FLOOR_ALBEDO: f32 = 0.8;
pub const WHITE_SPHERE: &str = "white sphere";
/// A unit sphere with no reflection at all, so only the medium in front of it shows.
pub const BLACK_SPHERE: &str = "black sphere";
pub const CORNELL: &str = "cornell box";
pub const TALL_BOX: &str = "tall box";
pub const SHORT_BOX: &str = "short box";
pub const EMITTER: &str = "emitter";
/// [`EMITTER`] under a full blue-tinted coat and a full fuzz, which pass its light by angle.
pub const COATED_EMITTER: &str = "coated emitter";
/// [`EMITTER`] with an emission texture of four quadrants that emit 1, 0.22, 0.05 and 0 of its
/// luminance, so no flip or swap of its uv leaves the light it gives where it is.
pub const QUADRANT_EMITTER: &str = "quadrant emitter";
/// A 0.8 x 0.3 m strip along x facing -y, emitting [`EMITTER_LUMINANCE`] nits, skinned to two
/// joints at its origin: the first holds x = -0.4, and the second's weight grows to 1 at
/// x = 0.4. Placed with [`Bed::add_skinned`].
pub const SKINNED_EMITTER: &str = "skinned emitter";
/// The alpha-tested 2 m quad of rings, facing +z, with opacity micromaps.
pub const LEAF: &str = "leaf";
/// [`LEAF`] cooked without opacity micromaps.
pub const BARE_LEAF: &str = "bare leaf";
/// [`LEAF`] as thin-walled clear glass of IOR 1.5, cut out by the same rings, in one model with
/// a small opaque quad 2 m below it.
pub const MASKED_PANE: &str = "masked pane";
/// [`LEAF`] without its opacity texture, its material naming linked surface 2, whose
/// coverage works the rings out from the uv.
pub const RINGS_LEAF: &str = "rings leaf";
/// [`LEAF`] without its opacity texture, its material naming linked surface 3, whose
/// coverage is a quarter everywhere.
pub const VEIL: &str = "veil";
pub const GLASS_SLAB: &str = "glass slab";
pub const SKIN_SLAB: &str = "skin slab";
/// An upright 2 m square perfect mirror (metal), facing +z.
pub const MIRROR: &str = "mirror";
/// The floor as a perfectly smooth black dielectric: only its mirror reflection shows.
pub const SMOOTH_FLOOR: &str = "smooth floor";
/// The floor with a strong bump normal map.
pub const BUMPY_FLOOR: &str = "bumpy floor";
/// The floor as a glossy black dielectric, roughness [`GLOSSY_ROUGHNESS`]: it shows lights by a
/// blurred reflection that next-event estimation and BSDF rays both find.
pub const GLOSSY_FLOOR: &str = "glossy floor";
pub const GLOSSY_ROUGHNESS: f32 = 0.25;
/// A 6 m square of 48 x 48 emitting tiles facing -y around its origin, each a fifth of its
/// cell's side: a sparse field with clusters enough for its cooked set to hold a tree.
pub const EMITTER_FIELD: &str = "emitter field";
/// An upright 1 m square facing +z whose material names linked surface 1.
pub const PANEL: &str = "panel";
/// A valley 3 m across and 8 m long along z, y = 0.5 x^2, finely tessellated, with
/// discrete levels of 1 cm and 3 cm error.
pub const TROUGH: &str = "trough";
/// A 2 m square facing +y, emitting [`EMITTER_LUMINANCE`] nits, nothing else.
pub const LIGHT_FLOOR: &str = "light floor";
/// A smooth glass slab 1.6 x [`SLAB_THICKNESS`] x 1 m around its origin (IOR 1.5) whose
/// interior transmits half the light over its thickness.
pub const TINTED_SLAB: &str = "tinted slab";
/// [`TINTED_SLAB`] whose `transmission_depth` a texture of 128 / 255 multiplies.
pub const DEPTH_TEXTURED_SLAB: &str = "depth textured slab";
/// [`TINTED_SLAB`] whose `transmission_depth` a texture multiplies by 128 / 255 where its top and
/// bottom faces' u is below 1/2, which is local z below 0, and by 1 elsewhere.
pub const DEPTH_SPLIT_SLAB: &str = "depth split slab";
/// [`DEPTH_TEXTURED_SLAB`] [`SHEET_THICKNESS`] thick, with a `transmission_depth` of
/// [`SHEET_DEPTH`].
pub const DEPTH_TEXTURED_SHEET: &str = "depth textured sheet";
/// A length the mesh's position grid holds exactly, since an optical depth this steep shows its
/// rounding.
pub const SHEET_THICKNESS: f32 = 0.0625;
pub const SHEET_DEPTH: f32 = 0.06;
/// [`TINTED_SLAB`], double-sided.
pub const TWO_SIDED_SLAB: &str = "two-sided slab";
/// [`LIGHT_FLOOR`] 0.5 m below [`TINTED_SLAB`] in one model.
pub const MIXED_SLAB: &str = "mixed slab";
/// [`TINTED_SLAB`] with a small emitter at its middle, facing up, in one model.
pub const LIT_SLAB: &str = "lit slab";
pub const SLAB_THICKNESS: f32 = 0.3;
/// [`TINTED_SLAB`] with specular roughness 0.3.
pub const ROUGH_SLAB: &str = "rough slab";
/// A smooth glass sphere of radius 1 m (IOR 1.5), clear.
pub const GLASS_SPHERE: &str = "glass sphere";
/// [`GLASS_SPHERE`] of a dispersive glass: IOR [`PRISM_IOR`], Abbe number [`PRISM_ABBE`].
pub const DISPERSIVE_SPHERE: &str = "dispersive sphere";
/// An upright prism of the dispersive glass: [`prism_corners`] at y = -1 m to 1 m.
pub const PRISM: &str = "prism";
pub const PRISM_IOR: f32 = 1.6;
pub const PRISM_ABBE: f32 = 20.0;
pub const PRISM_HALF_HEIGHT: f32 = 1.0;
/// A white emitter of [`EMITTER_LUMINANCE`] facing +z: a strip 2 [`LINE_HALF_WIDTH`] wide
/// and 2 [`LINE_HALF_HEIGHT`] tall.
pub const LINE: &str = "line";
pub const LINE_HALF_WIDTH: f32 = 0.01;
pub const LINE_HALF_HEIGHT: f32 = 2.0;

/// The corners of [`PRISM`]'s cross section in the xz plane: an equilateral triangle of
/// circumradius 0.5 m around the origin, apex toward +z, on the cook's 2^-11 m position
/// grid.
pub fn prism_corners() -> [Vec3; 3] {
    let grid = |v: f32| (v * 2048.0).round() / 2048.0;
    [90.0f32, 210.0, 330.0].map(|a| {
        let a = a.to_radians();
        Vec3::new(grid(0.5 * a.cos()), 0.0, grid(0.5 * a.sin()))
    })
}

/// A smooth glass box 4 x 1 x 4 m around its origin (IOR 1.5) holding a scattering medium
/// of [`MILK_SIGMA_T`] and [`MILK_SIGMA_S`] per m, isotropic.
pub const MILKY_SLAB: &str = "milky slab";
pub const MILK_SIGMA_T: f32 = 2.0;
pub const MILK_SIGMA_S: f32 = 0.5;
/// A smooth glass sphere of radius 1 m (IOR 1.5) holding a non-absorbing medium that
/// scatters 3 times per m.
pub const CLOUD_SPHERE: &str = "cloud sphere";
/// A water box 6 x 4 x 6 m around its origin (IOR 1.33) whose transmission color over 1 m
/// is exp(-[`WATER_SIGMA`]) in Rec.709.
pub const WATER_BOX: &str = "water box";
pub const WATER_SIGMA: [f32; 3] = [0.4, 0.2, 0.1];
/// A water box 40 x 8 x 40 m around its origin (IOR 1.33), turbid: extinction
/// [`TURBID_SIGMA_T`] and isotropic scattering [`TURBID_SIGMA_S`] per m, in Rec.709.
pub const TURBID_WATER: &str = "turbid water";
pub const TURBID_SIGMA_T: [f32; 3] = [0.45, 0.25, 0.2];
pub const TURBID_SIGMA_S: [f32; 3] = [0.1, 0.12, 0.13];

/// A smoke plume: the cooked volume's grid `density`, a column about 1.5 m across and 3 m
/// tall on 0.1 m voxels, base at the origin, thinning upward and broken into puffs under
/// 0.5 m.
pub const PLUME: &str = "plume";
const PLUME_MIN: [i32; 3] = [-12, 0, -12];
const PLUME_SIZE: [u32; 3] = [24, 30, 24];
const PLUME_VOXEL: f32 = 0.1;

/// The plume's input density at voxel `i`.
fn plume_density(i: [i32; 3]) -> f32 {
    let p = i.map(|c| c as f32 * PLUME_VOXEL);
    let radius = 0.75 - 0.15 * p[1] / 3.0;
    let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
    let core = (1.0 - r / radius).clamp(0.0, 1.0);
    let puffs = 0.55 + 0.45 * (7.1 * p[0] + 3.0 * p[1]).sin() * (5.3 * p[2] - 4.1 * p[1]).sin();
    2.0 * core.powf(0.7) * puffs * (1.0 - p[1] / 3.2)
}

/// The cooked plume: blob 0 parsed and the density grid's decoded pages.
pub fn plume_volume() -> (orr_pack::volume::VolumeInfo, Vec<Vec<u8>>) {
    let a = orr_pack::Archive::from_memory(Arc::clone(pack())).unwrap();
    let e = a
        .expect(AssetId::from_name(PLUME), orr_pack::Class::Volume as u16, 1)
        .unwrap();
    let blobs = a.blobs(e);
    let read = |b: u32| {
        let mut v = vec![0; blobs[b as usize].raw_len as usize];
        a.read(&blobs[b as usize], &mut v).unwrap();
        v
    };
    let info = orr_pack::volume::VolumeInfo::parse(&read(0)).unwrap();
    let g = info.grids[0].pages;
    let pages = (g.first..g.first + g.count).map(read).collect();
    (info, pages)
}

fn plume() -> AssetSpec {
    let n = PLUME_SIZE.map(|s| s as i32);
    let mut values = Vec::new();
    for z in 0..n[2] {
        for y in 0..n[1] {
            for x in 0..n[0] {
                values.push(plume_density([
                    PLUME_MIN[0] + x,
                    PLUME_MIN[1] + y,
                    PLUME_MIN[2] + z,
                ]));
            }
        }
    }
    AssetSpec {
        name: PLUME.into(),
        source: Source::Volume {
            input: VolumeInput::Grids(VolumeGrids {
                origin: [0.0; 3],
                voxel_size: [PLUME_VOXEL; 3],
                grids: vec![GridInput {
                    name: "density".into(),
                    background: 0.0,
                    voxels: GridVoxels::Dense {
                        min: PLUME_MIN,
                        size: PLUME_SIZE,
                        values,
                    },
                }],
            }),
            settings: VolumeSettings::default(),
        },
    }
}

/// The water box's transmission color over 1 m in the linear Rec.2020 the kernels work in.
pub fn water_transmission() -> [f64; 3] {
    let color = WATER_SIGMA.map(|s| f64::from(-s).exp());
    orr_color::rec709_to_rec2020(color).to_array()
}

/// A smooth dielectric's dense scattering interior, per m in Rec.709.
#[derive(Clone, Copy, Debug)]
pub struct DenseInterior {
    pub sigma_t: [f32; 3],
    pub sigma_s: [f32; 3],
    pub anisotropy: f32,
    pub ior: f32,
    /// The material's `transmission_depth`, m.
    pub depth: f32,
}

impl DenseInterior {
    fn material(&self, glass: Material) -> Material {
        Material {
            transmission_color: self.sigma_t.map(|s| (-s * self.depth).exp()),
            transmission_depth: self.depth,
            transmission_scatter: self.sigma_s.map(|s| s * self.depth),
            transmission_scatter_anisotropy: self.anisotropy,
            specular_ior: self.ior,
            ..glass
        }
    }

    /// Extinction and single-scattering albedo in linear Rec.2020, as the cook converts the
    /// material's colors.
    pub fn rec2020(&self) -> (DVec3, DVec3) {
        use orr_color::rec709_to_rec2020;
        let d = f64::from(self.depth);
        let color = DVec3::from(self.sigma_t.map(|s| (f64::from(-s) * d).exp()));
        let scatter = DVec3::from(self.sigma_s.map(|s| f64::from(s) * d));
        let extinction = DVec3::from(rec709_to_rec2020(color).to_array().map(|c| -c.ln() / d));
        (extinction, rec709_to_rec2020(scatter) / d / extinction)
    }
}

/// A smooth sphere of radius 1 m holding [`MILK`].
pub const MILK_SPHERE: &str = "milk sphere";
/// Bright and forward-scattering, about 40 mean free paths to the radius.
pub const MILK: DenseInterior = DenseInterior {
    sigma_t: [40.0, 42.0, 45.0],
    sigma_s: [39.8, 41.7, 44.0],
    anisotropy: 0.7,
    ior: 1.35,
    depth: 0.02,
};
/// A smooth sphere of radius 1 m holding [`JADE`].
pub const JADE_SPHERE: &str = "jade sphere";
/// Green, absorbing and nearly isotropic, a few mean free paths to the radius.
pub const JADE: DenseInterior = DenseInterior {
    sigma_t: [8.0, 4.0, 6.0],
    sigma_s: [5.0, 3.8, 4.5],
    anisotropy: 0.3,
    ior: 1.6,
    depth: 0.25,
};
/// A black 2 m square facing +y with opacity [`PRESENCE_VEIL_OPACITY`], blended: a presence layer.
pub const PRESENCE_VEIL: &str = "presence veil";
pub const PRESENCE_VEIL_OPACITY: f32 = 0.25;
/// Subsurface boxes 2 m square of [`walk_material`] with their top face at y = 0, a quarter,
/// 1, 4 and 16 of its longest mean free path thick, rounded to the cook's 2^-11 m position
/// grid.
pub const WALK_SLABS: [(&str, f32); 4] = [
    ("walk slab 0.25", 5.0 / 2048.0),
    ("walk slab 1", 20.0 / 2048.0),
    ("walk slab 4", 82.0 / 2048.0),
    ("walk slab 16", 328.0 / 2048.0),
];
/// A subsurface sphere of radius 1 m, of [`walk_material`].
pub const WALK_SPHERE: &str = "walk sphere";
/// A subsurface sphere of radius 1 m of [`walk_material`] without specular reflection, so
/// light crosses its surface unchanged.
pub const WALK_FURNACE: &str = "walk furnace";
/// The walk models' material: gray subsurface color 0.8, isotropic, mean free paths of
/// 10, 5 and 2.5 mm per channel, specular roughness 0.4 at IOR 1.4.
pub fn walk_material() -> Material {
    Material {
        subsurface_weight: 1.0,
        subsurface_color: [0.8; 3],
        subsurface_radius: 0.01,
        subsurface_radius_scale: [1.0, 0.5, 0.25],
        specular_roughness: 0.4,
        specular_ior: 1.4,
        ..Material::default()
    }
}

/// A dark horizontal canopy 160 m square at y = 0: slats 1 m wide along z with 1 m gaps
/// between them, faced up and down.
pub const CANOPY: &str = "canopy";

/// The bed's emitter: nits, white.
pub const EMITTER_LUMINANCE: f32 = 50.0;

/// Triangles per material section, each wound so its front faces its vertices' normal.
#[derive(Default)]
struct MeshBuilder {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    sections: Vec<Vec<[u32; 3]>>,
}

impl MeshBuilder {
    fn vertex(&mut self, p: Vec3, n: Vec3, uv: Vec2) -> u32 {
        self.positions.push(p);
        self.normals.push(n);
        self.uvs.push(uv);
        self.positions.len() as u32 - 1
    }

    fn triangle(&mut self, section: usize, t: [u32; 3]) {
        let [a, b, c] = t.map(|i| self.positions[i as usize]);
        let n = self.normals[t[0] as usize];
        let t = if (b - a).cross(c - a).dot(n) >= 0.0 {
            t
        } else {
            [t[0], t[2], t[1]]
        };
        if self.sections.len() <= section {
            self.sections.resize(section + 1, Vec::new());
        }
        self.sections[section].push(t);
    }

    /// A quad spanning `corners` in order around its edge, facing `normal`, split into
    /// `n` x `n` cells.
    fn quad(&mut self, section: usize, corners: [Vec3; 4], normal: Vec3, n: u32) {
        let first = self.positions.len() as u32;
        for j in 0..=n {
            for i in 0..=n {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                let p = corners[0]
                    .lerp(corners[1], u)
                    .lerp(corners[3].lerp(corners[2], u), v);
                self.vertex(p, normal, Vec2::new(u, 1.0 - v));
            }
        }
        let w = n + 1;
        for j in 0..n {
            for i in 0..n {
                let c = first + j * w + i;
                self.triangle(section, [c, c + 1, c + w + 1]);
                self.triangle(section, [c, c + w + 1, c + w]);
            }
        }
    }

    /// An axis-aligned box of `half` extents around the origin, faces outward.
    fn cuboid(&mut self, section: usize, half: Vec3) {
        for axis in 0..3 {
            for sign in [-1.0f32, 1.0] {
                let n = Vec3::AXES[axis] * sign;
                let (u, v) = (Vec3::AXES[(axis + 1) % 3], Vec3::AXES[(axis + 2) % 3]);
                let c = n * half;
                let (du, dv) = (u * half, v * half);
                self.quad(
                    section,
                    [c - du - dv, c + du - dv, c + du + dv, c - du + dv],
                    n,
                    1,
                );
            }
        }
    }

    /// A UV sphere of `radius` around the origin in `rings` latitude bands.
    fn sphere(&mut self, section: usize, rings: u32, radius: f32) {
        let segments = 2 * rings;
        let first = self.positions.len() as u32;
        for r in 0..=rings {
            for s in 0..=segments {
                let (v, u) = (r as f32 / rings as f32, s as f32 / segments as f32);
                let (theta, phi) = (v * std::f32::consts::PI, u * std::f32::consts::TAU);
                let n = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );
                self.vertex(n * radius, n, Vec2::new(u, v));
            }
        }
        let w = segments + 1;
        for r in 0..rings {
            for s in 0..segments {
                let c = first + r * w + s;
                if r > 0 {
                    self.triangle(section, [c, c + 1, c + w]);
                }
                if r < rings - 1 {
                    self.triangle(section, [c + 1, c + w + 1, c + w]);
                }
            }
        }
    }

    fn finish(self) -> MeshInput {
        let mut m = MeshInput {
            positions: self.positions,
            normals: self.normals,
            uvs: Some(self.uvs),
            ..MeshInput::default()
        };
        for tris in self.sections {
            let first = m.indices.len() as u32;
            m.indices.extend(tris.iter().flatten());
            m.sections.push(Span {
                first,
                count: m.indices.len() as u32 - first,
            });
        }
        m
    }
}

/// The corners of a square of half side `half` at height `y`, in the xz plane.
fn square(half: f32, y: f32) -> [Vec3; 4] {
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .map(|(x, z)| Vec3::new(x * half, y, z * half))
}

/// The corners of an upright square of half side `half` around the origin, facing +z.
fn upright_square(half: f32) -> [Vec3; 4] {
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .map(|(x, y)| Vec3::new(x * half, y * half, 0.0))
}

/// An upright square of half side `half` around the origin, facing +z.
fn upright(half: f32) -> MeshInput {
    mesh(|b| b.quad(0, upright_square(half), Vec3::Z, 1))
}

fn mesh(build: impl FnOnce(&mut MeshBuilder)) -> MeshInput {
    let mut b = MeshBuilder::default();
    build(&mut b);
    b.finish()
}

/// [`QUADRANT_EMITTER`]'s emission texture: a 2 x 2 sRGB PNG.
fn quadrants_png() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr_lighting_bed_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("quadrants.png");
    let mut enc = png::Encoder::new(std::fs::File::create(&path).unwrap(), 2, 2);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&[255, 128, 64, 0])
        .unwrap();
    path
}

/// [`SKINNED_EMITTER`] as a binary glTF.
fn skinned_strip_glb() -> PathBuf {
    let n = 8u32;
    let (mut positions, mut normals, mut joints, mut weights) = (
        Vec::<f32>::new(),
        Vec::<f32>::new(),
        Vec::<u8>::new(),
        Vec::<f32>::new(),
    );
    for i in 0..=n {
        let x = -0.4 + 0.8 * i as f32 / n as f32;
        let w = (x + 0.4) / 0.8;
        for z in [-0.15f32, 0.15] {
            positions.extend([x, 0.0, z]);
            normals.extend([0.0, -1.0, 0.0]);
            joints.extend([0, 1, 0, 0]);
            weights.extend([1.0 - w, w, 0.0, 0.0]);
        }
    }
    // Vertex 2 i lies at z = -0.15 and 2 i + 1 at z = 0.15; both triangles face -y.
    let indices: Vec<u32> = (0..n)
        .flat_map(|i| {
            let (a, b, c, d) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
            [a, c, b, c, d, b]
        })
        .collect();
    let identity: Vec<f32> = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    ]
    .repeat(2);
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut add = |bytes: &[u8]| {
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            bin.len(),
            bytes.len()
        ));
        bin.extend_from_slice(bytes);
        bin.resize(bin.len().next_multiple_of(4), 0);
        views.len() - 1
    };
    let v_pos = add(bytemuck::cast_slice(&positions));
    let v_nrm = add(bytemuck::cast_slice(&normals));
    let v_jnt = add(&joints);
    let v_wgt = add(bytemuck::cast_slice(&weights));
    let v_idx = add(bytemuck::cast_slice(&indices));
    let v_ibm = add(bytemuck::cast_slice(&identity));
    let vertices = positions.len() / 3;
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"extensionsUsed":["KHR_materials_emissive_strength"],
"scene":0,"scenes":[{{"nodes":[0,2]}}],
"nodes":[{{"name":"root","children":[1]}},{{"name":"tip"}},{{"name":"strip","mesh":0,"skin":0}}],
"skins":[{{"joints":[0,1],"inverseBindMatrices":5}}],
"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[0,0,0,1],"metallicFactor":0}},
"emissiveFactor":[1,1,1],"extensions":{{"KHR_materials_emissive_strength":{{"emissiveStrength":{EMITTER_LUMINANCE}}}}}}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3}},"indices":4,"material":0}}]}}],
"accessors":[
{{"bufferView":{v_pos},"componentType":5126,"count":{vertices},"type":"VEC3","min":[-0.4,0,-0.15],"max":[0.4,0,0.15]}},
{{"bufferView":{v_nrm},"componentType":5126,"count":{vertices},"type":"VEC3"}},
{{"bufferView":{v_jnt},"componentType":5121,"count":{vertices},"type":"VEC4"}},
{{"bufferView":{v_wgt},"componentType":5126,"count":{vertices},"type":"VEC4"}},
{{"bufferView":{v_idx},"componentType":5125,"count":{n_idx},"type":"SCALAR"}},
{{"bufferView":{v_ibm},"componentType":5126,"count":2,"type":"MAT4"}}],
"bufferViews":[{views}],
"buffers":[{{"byteLength":{len}}}]}}"#,
        n_idx = indices.len(),
        views = views.join(","),
        len = bin.len(),
    );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend(b"glTF");
    glb.extend(2u32.to_le_bytes());
    glb.extend(((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    glb.extend((json.len() as u32).to_le_bytes());
    glb.extend(b"JSON");
    glb.extend(&json);
    glb.extend((bin.len() as u32).to_le_bytes());
    glb.extend(b"BIN\0");
    glb.extend(&bin);
    let dir = std::env::temp_dir().join(format!("orr_lighting_bed_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("skinned_strip.glb");
    std::fs::write(&path, glb).unwrap();
    path
}

/// A 16 x 16 grayscale PNG, `left` where u is below 1/2 and `right` elsewhere.
fn halves_png(name: &str, left: u8, right: u8) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr_lighting_bed_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    let mut enc = png::Encoder::new(std::fs::File::create(&path).unwrap(), 16, 16);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    let data: Vec<u8> = (0..256)
        .map(|i| if i % 16 < 8 { left } else { right })
        .collect();
    enc.write_header().unwrap().write_image_data(&data).unwrap();
    path
}

/// A grayscale PNG of concentric rings, opaque on the inner 60% of each ring.
fn rings_png() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr_lighting_bed_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rings.png");
    let n = 128;
    let data: Vec<u8> = (0..n * n)
        .map(|i| {
            let (u, v) = (
                (i % n) as f32 / n as f32 - 0.5,
                (i / n) as f32 / n as f32 - 0.5,
            );
            if ((u * u + v * v).sqrt() * 8.0).fract() < 0.6 {
                255
            } else {
                0
            }
        })
        .collect();
    let mut enc = png::Encoder::new(std::fs::File::create(&path).unwrap(), n, n);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&data).unwrap();
    path
}

/// A tangent-space normal map of round bumps tilting normals up to about 50 degrees.
fn bumps_png() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr_lighting_bed_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bumps.png");
    let n = 128;
    let data: Vec<u8> = (0..n * n)
        .flat_map(|i| {
            let tau = std::f32::consts::TAU;
            let (u, v) = ((i % n) as f32 / n as f32, (i / n) as f32 / n as f32);
            let (dx, dy) = (
                1.2 * (8.0 * tau * u).cos() * (8.0 * tau * v).sin(),
                1.2 * (8.0 * tau * u).sin() * (8.0 * tau * v).cos(),
            );
            let normal = Vec3::new(-dx, -dy, 1.0).normalize();
            normal
                .to_array()
                .map(|c| ((c * 0.5 + 0.5) * 255.0).round() as u8)
        })
        .collect();
    let mut enc = png::Encoder::new(std::fs::File::create(&path).unwrap(), n, n);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&data).unwrap();
    path
}

fn diffuse(base_color: [f32; 3]) -> Material {
    Material {
        base_color,
        specular_weight: 0.0,
        ..Material::default()
    }
}

fn cook() -> Arc<[u8]> {
    let model = |name: &str, mesh: MeshInput, materials: Vec<Material>| AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings::default(),
        },
    };
    let floor = Material {
        base_color: [0.5; 3],
        specular_roughness: 0.8,
        ..Material::default()
    };
    let emitter = Material {
        base_color: [0.0; 3],
        emission_color: [1.0; 3],
        emission_luminance: EMITTER_LUMINANCE,
        ..Material::default()
    };
    let mut quadrant_emitter = emitter;
    quadrant_emitter.textures[slot::EMISSION].texture = AssetId::from_name("quadrants");
    let coated_emitter = Material {
        coat_weight: 1.0,
        coat_color: [0.4, 0.6, 1.0],
        fuzz_weight: 1.0,
        fuzz_color: [1.0; 3],
        ..emitter
    };
    let mut leaf = Material {
        base_color: [0.2, 0.6, 0.15],
        alpha_mode: Material::MASK,
        alpha_cutoff: 0.5,
        double_sided: 1,
        ..Material::default()
    };
    let linked_leaf = move |surface| Material { surface, ..leaf };
    leaf.textures[slot::OPACITY].texture = AssetId::from_name("rings");
    let glass = Material {
        base_color: [1.0; 3],
        specular_roughness: 0.0,
        specular_ior: 1.5,
        transmission_weight: 1.0,
        ..Material::default()
    };
    let masked_glass = Material {
        alpha_mode: Material::MASK,
        alpha_cutoff: 0.5,
        double_sided: 1,
        thin_walled: 1,
        textures: leaf.textures,
        ..glass
    };
    let skin = Material {
        base_color: [0.9, 0.6, 0.5],
        subsurface_weight: 1.0,
        subsurface_color: [0.9, 0.6, 0.5],
        subsurface_radius: 0.05,
        subsurface_radius_scale: [1.0, 0.4, 0.2],
        ..Material::default()
    };
    let white = diffuse([0.73; 3]);
    let mirror = Material {
        base_color: [1.0; 3],
        metalness: 1.0,
        specular_roughness: 0.0,
        ..Material::default()
    };
    let smooth = Material {
        base_color: [0.0; 3],
        specular_roughness: 0.0,
        specular_ior: 1.5,
        ..Material::default()
    };
    let glossy = Material {
        specular_roughness: GLOSSY_ROUGHNESS,
        ..smooth
    };
    let mut bumpy = floor;
    bumpy.textures[slot::NORMAL].texture = AssetId::from_name("bumps");
    let panel = Material {
        base_color: [0.0; 3],
        surface: 1,
        ..Material::default()
    };
    let trough = mesh(|b| {
        let (nx, nz) = (48u32, 16u32);
        let first = b.positions.len() as u32;
        for j in 0..=nz {
            for i in 0..=nx {
                let x = -1.5 + 3.0 * i as f32 / nx as f32;
                let z = -4.0 + 8.0 * j as f32 / nz as f32;
                let n = Vec3::new(-x, 1.0, 0.0).normalize();
                b.vertex(
                    Vec3::new(x, 0.5 * x * x, z),
                    n,
                    Vec2::new(i as f32 / nx as f32, j as f32 / nz as f32),
                );
            }
        }
        let w = nx + 1;
        for j in 0..nz {
            for i in 0..nx {
                let c = first + j * w + i;
                b.triangle(0, [c, c + 1, c + w + 1]);
                b.triangle(0, [c, c + w + 1, c + w]);
            }
        }
    });
    let (x, y, z) = (Vec3::X, Vec3::Y, Vec3::Z);
    let cornell = mesh(|b| {
        let c = |px: f32, py: f32, pz: f32| Vec3::new(px, py, pz);
        b.quad(0, square(1.0, 0.0), y, 4);
        b.quad(0, square(1.0, 2.0), -y, 4);
        b.quad(
            0,
            [
                c(-1., 0., -1.),
                c(1., 0., -1.),
                c(1., 2., -1.),
                c(-1., 2., -1.),
            ],
            z,
            4,
        );
        b.quad(
            1,
            [
                c(-1., 0., -1.),
                c(-1., 0., 1.),
                c(-1., 2., 1.),
                c(-1., 2., -1.),
            ],
            x,
            4,
        );
        b.quad(
            2,
            [c(1., 0., -1.), c(1., 0., 1.), c(1., 2., 1.), c(1., 2., -1.)],
            -x,
            4,
        );
    });
    let slab = || mesh(|b| b.cuboid(0, Vec3::new(0.8, 0.15, 0.5)));
    let light_floor = Material {
        base_color: [0.0; 3],
        specular_weight: 0.0,
        emission_color: [1.0; 3],
        emission_luminance: EMITTER_LUMINANCE,
        ..Material::default()
    };
    let tinted = Material {
        transmission_color: [0.5; 3],
        transmission_depth: SLAB_THICKNESS,
        ..glass
    };
    let mut depth_textured = tinted;
    depth_textured.textures[slot::TRANSMISSION_DEPTH].texture = AssetId::from_name("gray");
    let mut depth_split = tinted;
    depth_split.textures[slot::TRANSMISSION_DEPTH].texture = AssetId::from_name("split");
    let depth_textured_sheet = Material {
        transmission_depth: SHEET_DEPTH,
        ..depth_textured
    };
    let two_sided = Material {
        double_sided: 1,
        ..tinted
    };
    let rough = Material {
        specular_roughness: 0.3,
        ..tinted
    };
    // OpenPBR's interior: sigma_t = -ln(color) / depth, sigma_s = scatter / depth.
    let milk = Material {
        transmission_color: [(-MILK_SIGMA_T).exp(); 3],
        transmission_depth: 1.0,
        transmission_scatter: [MILK_SIGMA_S; 3],
        ..glass
    };
    let cloud = Material {
        transmission_color: [(-3.0f32).exp(); 3],
        transmission_depth: 1.0,
        transmission_scatter: [3.0; 3],
        ..glass
    };
    let dispersive = Material {
        specular_ior: PRISM_IOR,
        transmission_dispersion_scale: 1.0,
        transmission_dispersion_abbe_number: PRISM_ABBE,
        ..glass
    };
    let prism = mesh(|b| {
        let c = prism_corners();
        let h = Vec3::Y * PRISM_HALF_HEIGHT;
        for i in 0..3 {
            let (p, q) = (c[i], c[(i + 1) % 3]);
            let mid = 0.5 * (p + q);
            let n = Vec3::Y.cross(q - p).normalize();
            let n = if n.dot(mid) < 0.0 { -n } else { n };
            b.quad(0, [p - h, q - h, q + h, p + h], n, 1);
        }
        for side in [-1.0f32, 1.0] {
            let n = Vec3::Y * side;
            let v = c.map(|p| b.vertex(p + h * side, n, Vec2::ZERO));
            b.triangle(0, v);
        }
    });
    let line = mesh(|b| {
        let (w, h) = (LINE_HALF_WIDTH, LINE_HALF_HEIGHT);
        b.quad(
            0,
            [
                Vec3::new(-w, -h, 0.0),
                Vec3::new(w, -h, 0.0),
                Vec3::new(w, h, 0.0),
                Vec3::new(-w, h, 0.0),
            ],
            Vec3::Z,
            1,
        );
    });
    let water = Material {
        specular_ior: 1.33,
        transmission_color: WATER_SIGMA.map(|s| (-s).exp()),
        transmission_depth: 1.0,
        ..glass
    };
    let turbid = Material {
        transmission_color: TURBID_SIGMA_T.map(|s| (-s).exp()),
        transmission_scatter: TURBID_SIGMA_S,
        ..water
    };
    let veil = Material {
        base_color: [0.0; 3],
        specular_weight: 0.0,
        geometry_opacity: PRESENCE_VEIL_OPACITY,
        alpha_mode: Material::BLEND,
        double_sided: 1,
        ..Material::default()
    };
    let canopy = mesh(|b| {
        for i in 0..80 {
            let x0 = -80.0 + 2.0 * i as f32;
            let corners = [
                Vec3::new(x0, 0.0, -80.0),
                Vec3::new(x0 + 1.0, 0.0, -80.0),
                Vec3::new(x0 + 1.0, 0.0, 80.0),
                Vec3::new(x0, 0.0, 80.0),
            ];
            b.quad(0, corners, y, 1);
            b.quad(0, corners, -y, 1);
        }
    });
    let walk_slabs = WALK_SLABS.map(|(name, thickness)| {
        let slab = mesh(|b| {
            b.cuboid(0, Vec3::new(1.0, 0.5 * thickness, 1.0));
            for p in &mut b.positions {
                p.y -= 0.5 * thickness;
            }
        });
        model(name, slab, vec![walk_material()])
    });
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: "gray".into(),
                    source: Source::Texture {
                        path: halves_png("gray", 128, 128),
                        settings: TextureSettings {
                            usage: Usage::Data(orr_cook::DataFormat::R8),
                            ..TextureSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: "split".into(),
                    source: Source::Texture {
                        path: halves_png("split", 128, 255),
                        settings: TextureSettings {
                            usage: Usage::Data(orr_cook::DataFormat::R8),
                            ..TextureSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: "rings".into(),
                    source: Source::Texture {
                        path: rings_png(),
                        settings: TextureSettings {
                            usage: Usage::Linear1,
                            ..TextureSettings::default()
                        },
                    },
                },
                model(
                    FLOOR,
                    mesh(|b| b.quad(0, square(5.0, 0.0), y, 8)),
                    vec![floor],
                ),
                model(
                    WHITE_FLOOR,
                    mesh(|b| b.quad(0, square(5.0, 0.0), y, 8)),
                    vec![diffuse([WHITE_FLOOR_ALBEDO; 3])],
                ),
                model(
                    WHITE_SPHERE,
                    mesh(|b| b.sphere(0, 24, 1.0)),
                    vec![diffuse([1.0; 3])],
                ),
                model(
                    BLACK_SPHERE,
                    mesh(|b| b.sphere(0, 24, 1.0)),
                    vec![diffuse([0.0; 3])],
                ),
                model(
                    CORNELL,
                    cornell,
                    vec![
                        white,
                        diffuse([0.63, 0.065, 0.05]),
                        diffuse([0.14, 0.45, 0.091]),
                    ],
                ),
                model(
                    TALL_BOX,
                    mesh(|b| b.cuboid(0, Vec3::new(0.3, 0.6, 0.3))),
                    vec![white],
                ),
                model(
                    SHORT_BOX,
                    mesh(|b| b.cuboid(0, Vec3::splat(0.3))),
                    vec![white],
                ),
                model(
                    EMITTER,
                    mesh(|b| b.quad(0, square(0.3, 0.0), -y, 1)),
                    vec![emitter],
                ),
                model(
                    COATED_EMITTER,
                    mesh(|b| b.quad(0, square(0.3, 0.0), -y, 1)),
                    vec![coated_emitter],
                ),
                AssetSpec {
                    name: "quadrants".into(),
                    source: Source::Texture {
                        path: quadrants_png(),
                        settings: TextureSettings {
                            usage: Usage::Color,
                            ..TextureSettings::default()
                        },
                    },
                },
                model(
                    QUADRANT_EMITTER,
                    mesh(|b| b.quad(0, square(0.3, 0.0), -y, 1)),
                    vec![quadrant_emitter],
                ),
                AssetSpec {
                    name: SKINNED_EMITTER.into(),
                    source: Source::Gltf {
                        path: skinned_strip_glb(),
                        settings: GltfSettings {
                            mesh: MeshSettings {
                                deformation_bound: 1.0,
                                ..MeshSettings::default()
                            },
                            ..GltfSettings::default()
                        },
                    },
                },
                model(
                    EMITTER_FIELD,
                    mesh(|b| {
                        let (n, side) = (48, 6.0f32);
                        let cell = side / n as f32;
                        for j in 0..n {
                            for i in 0..n {
                                let c = Vec3::new(
                                    (i as f32 + 0.5) * cell - 0.5 * side,
                                    0.0,
                                    (j as f32 + 0.5) * cell - 0.5 * side,
                                );
                                let tile = square(0.1 * cell, 0.0).map(|p| p + c);
                                b.quad(0, tile, -y, 1);
                            }
                        }
                    }),
                    vec![emitter],
                ),
                model(LEAF, upright(1.0), vec![leaf]),
                AssetSpec {
                    name: BARE_LEAF.into(),
                    source: Source::Mesh {
                        mesh: upright(1.0),
                        materials: vec![leaf],
                        settings: MeshSettings {
                            omm: None,
                            ..MeshSettings::default()
                        },
                    },
                },
                model(
                    MASKED_PANE,
                    mesh(|b| {
                        b.quad(0, upright_square(1.0), Vec3::Z, 1);
                        b.quad(1, square(0.3, -2.0), y, 1);
                    }),
                    vec![masked_glass, floor],
                ),
                model(RINGS_LEAF, upright(1.0), vec![linked_leaf(2)]),
                model(VEIL, upright(1.0), vec![linked_leaf(3)]),
                model(GLASS_SLAB, slab(), vec![glass]),
                model(SKIN_SLAB, slab(), vec![skin]),
                AssetSpec {
                    name: "bumps".into(),
                    source: Source::Texture {
                        path: bumps_png(),
                        settings: TextureSettings {
                            usage: Usage::Normal,
                            ..TextureSettings::default()
                        },
                    },
                },
                model(MIRROR, upright(1.0), vec![mirror]),
                model(
                    SMOOTH_FLOOR,
                    mesh(|b| b.quad(0, square(5.0, 0.0), y, 8)),
                    vec![smooth],
                ),
                model(
                    GLOSSY_FLOOR,
                    mesh(|b| b.quad(0, square(5.0, 0.0), y, 8)),
                    vec![glossy],
                ),
                model(
                    BUMPY_FLOOR,
                    mesh(|b| b.quad(0, square(5.0, 0.0), y, 8)),
                    vec![bumpy],
                ),
                model(PANEL, upright(0.5), vec![panel]),
                model(
                    LIGHT_FLOOR,
                    mesh(|b| b.quad(0, square(1.0, 0.0), y, 1)),
                    vec![light_floor],
                ),
                model(
                    TINTED_SLAB,
                    mesh(|b| b.cuboid(0, Vec3::new(0.8, SLAB_THICKNESS * 0.5, 0.5))),
                    vec![tinted],
                ),
                model(
                    DEPTH_TEXTURED_SLAB,
                    mesh(|b| b.cuboid(0, Vec3::new(0.8, SLAB_THICKNESS * 0.5, 0.5))),
                    vec![depth_textured],
                ),
                model(
                    DEPTH_SPLIT_SLAB,
                    mesh(|b| b.cuboid(0, Vec3::new(0.8, SLAB_THICKNESS * 0.5, 0.5))),
                    vec![depth_split],
                ),
                model(
                    DEPTH_TEXTURED_SHEET,
                    mesh(|b| b.cuboid(0, Vec3::new(0.8, SHEET_THICKNESS * 0.5, 0.5))),
                    vec![depth_textured_sheet],
                ),
                model(
                    TWO_SIDED_SLAB,
                    mesh(|b| b.cuboid(0, Vec3::new(0.8, SLAB_THICKNESS * 0.5, 0.5))),
                    vec![two_sided],
                ),
                model(
                    MIXED_SLAB,
                    mesh(|b| {
                        b.quad(0, square(1.0, -0.5), y, 1);
                        b.cuboid(1, Vec3::new(0.8, SLAB_THICKNESS * 0.5, 0.5));
                    }),
                    vec![light_floor, tinted],
                ),
                model(
                    LIT_SLAB,
                    mesh(|b| {
                        b.quad(0, square(0.3, 0.0), y, 1);
                        b.cuboid(1, Vec3::new(0.8, SLAB_THICKNESS * 0.5, 0.5));
                    }),
                    vec![light_floor, tinted],
                ),
                model(
                    ROUGH_SLAB,
                    mesh(|b| b.cuboid(0, Vec3::new(0.8, SLAB_THICKNESS * 0.5, 0.5))),
                    vec![rough],
                ),
                model(GLASS_SPHERE, mesh(|b| b.sphere(0, 32, 1.0)), vec![glass]),
                model(
                    DISPERSIVE_SPHERE,
                    mesh(|b| b.sphere(0, 32, 1.0)),
                    vec![dispersive],
                ),
                model(PRISM, prism, vec![dispersive]),
                model(LINE, line, vec![light_floor]),
                model(
                    MILKY_SLAB,
                    mesh(|b| b.cuboid(0, Vec3::new(2.0, 0.5, 2.0))),
                    vec![milk],
                ),
                model(CLOUD_SPHERE, mesh(|b| b.sphere(0, 32, 1.0)), vec![cloud]),
                model(
                    MILK_SPHERE,
                    mesh(|b| b.sphere(0, 48, 1.0)),
                    vec![MILK.material(glass)],
                ),
                model(
                    JADE_SPHERE,
                    mesh(|b| b.sphere(0, 48, 1.0)),
                    vec![JADE.material(glass)],
                ),
                model(
                    WATER_BOX,
                    mesh(|b| b.cuboid(0, Vec3::new(3.0, 2.0, 3.0))),
                    vec![water],
                ),
                model(
                    TURBID_WATER,
                    mesh(|b| b.cuboid(0, Vec3::new(20.0, 4.0, 20.0))),
                    vec![turbid],
                ),
                model(
                    PRESENCE_VEIL,
                    mesh(|b| b.quad(0, square(1.0, 0.0), y, 1)),
                    vec![veil],
                ),
                model(CANOPY, canopy, vec![diffuse([0.1; 3])]),
                model(
                    WALK_SPHERE,
                    mesh(|b| b.sphere(0, 48, 1.0)),
                    vec![walk_material()],
                ),
                model(
                    WALK_FURNACE,
                    mesh(|b| b.sphere(0, 48, 1.0)),
                    vec![Material {
                        specular_weight: 0.0,
                        subsurface_radius_scale: [1.0; 3],
                        ..walk_material()
                    }],
                ),
                AssetSpec {
                    name: TROUGH.into(),
                    source: Source::Mesh {
                        mesh: trough,
                        materials: vec![white],
                        settings: MeshSettings {
                            lod: LodMode::Discrete {
                                errors: vec![0.01, 0.03],
                            },
                            ..MeshSettings::default()
                        },
                    },
                },
            ]
            .into_iter()
            .chain(walk_slabs)
            .chain([plume()])
            .collect(),
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0)
}

/// The bed's pack, cooked once per process.
fn pack() -> &'static Arc<[u8]> {
    static PACK: OnceLock<Arc<[u8]>> = OnceLock::new();
    PACK.get_or_init(cook)
}

fn time(frame: u64) -> FrameTime {
    FrameTime {
        absolute: frame as f64 / 60.0,
        delta: 1.0 / 60.0,
    }
}

/// World from view for a camera at `eye` looking at `target`, y up.
pub fn look(eye: Vec3, target: Vec3) -> Quat {
    orr_math::camera::rh::view::look_at_quat(eye, target, Vec3::Y).inverse()
}

/// The image point, in pixels from the top left, where `spec`'s camera sees `p`.
pub fn pixel_of(spec: &Spec, p: Vec3) -> Vec2 {
    let q = look(spec.eye, spec.target).inverse() * (p - spec.eye);
    let t = (0.5 * spec.fov_y).tan();
    let aspect = EXTENT[0] as f32 / EXTENT[1] as f32;
    let ndc = Vec2::new(q.x / (-q.z * t * aspect), q.y / (-q.z * t));
    Vec2::new(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y) * Vec2::new(EXTENT[0] as f32, EXTENT[1] as f32)
}

/// Scene compute work a bed runs every frame before its scene pass.
pub type SceneComputeHook = Box<dyn FnMut(&mut SceneCompute<'_, '_>)>;

/// Bytes of root memory each frame's kernels may put.
const ROOT_BYTES: u64 = 64 * 1024;

/// What a kernel recording into a bed frame reads and writes.
pub struct BedFrame<'a> {
    /// The frame's index in the accumulation.
    pub index: u32,
    pub seed: u32,
    /// The orr_scene frame id.
    pub frame: u64,
    pub view: &'a RasterView,
    /// The view as the frame rendered it.
    pub desc: ViewDesc,
    pub lighting: &'a SceneLighting,
    /// The radiance target the kernel writes, at [`EXTENT`].
    pub target: &'a mut LightingView,
    /// The bed draws particles after the kernel.
    particles: bool,
    roots: &'a mut Roots,
}

impl BedFrame<'_> {
    /// The frame as the view dispatch reads it.
    pub fn pass(&self) -> ViewPass {
        ViewPass {
            frame: self.frame,
            scene: self.view.scene,
            view: self.view.view,
            gbuffer: self.view.gbuffer,
            raster: self.view.raster,
            smoke_args: self.view.smoke_args,
            sample_index: self.index,
            seed: self.seed,
            dof_background: None,
            frosted_glass: false,
            overlay_front: false,
            scene_features: self.view.features,
            particle_lighting: self.particles,
        }
    }

    /// `value` in root memory that lives until the frame completes.
    pub fn put<T: bytemuck::Pod>(&mut self, value: &T) -> GpuPtr<T> {
        self.roots.put(value)
    }
}

struct Roots {
    memory: Allocation<u8>,
    used: u64,
}

impl Roots {
    fn put<T: bytemuck::Pod>(&mut self, value: &T) -> GpuPtr<T> {
        let bytes = bytemuck::bytes_of(value);
        let at = self.used;
        self.used = (at + bytes.len() as u64).next_multiple_of(256);
        assert!(self.used <= ROOT_BYTES, "bed frame roots overflow");
        self.memory.as_mut_slice().unwrap()[at as usize..][..bytes.len()].copy_from_slice(bytes);
        self.memory.gpu().offset(at).cast()
    }
}

/// The mean of a bed's accumulated radiance: RGBA per pixel, rows of [`EXTENT`] top to
/// bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Accumulated {
    pub mean: Vec<[f32; 4]>,
}

impl Accumulated {
    /// Mean rgb of the `2 half` x `2 half` pixels around image point `at`.
    pub fn around(&self, at: Vec2, half: u32) -> [f64; 3] {
        let [w, h] = EXTENT;
        let (cx, cy) = (at.x.round() as u32, at.y.round() as u32);
        assert!(
            cx >= half && cy >= half && cx + half <= w && cy + half <= h,
            "{at} near the edge"
        );
        let mut sum = [0.0f64; 3];
        for y in cy - half..cy + half {
            for x in cx - half..cx + half {
                let p = self.mean[(y * w + x) as usize];
                for c in 0..3 {
                    sum[c] += f64::from(p[c]);
                }
            }
        }
        sum.map(|s| s / f64::from(4 * half * half))
    }

    /// The mean as a linear float image.
    pub fn image(&self) -> Image {
        Image {
            width: EXTENT[0],
            height: EXTENT[1],
            pixels: Pixels::Rgba32F(self.mean.iter().flatten().copied().collect()),
        }
    }

    /// The mean scaled by `exposure`, clamped and sRGB encoded, alpha opaque.
    pub fn display(&self, exposure: f32) -> Image {
        let encode = |v: f32| {
            let s = orr_color::srgb_encode(f64::from((v * exposure).clamp(0.0, 1.0)));
            (s * 255.0).round() as u8
        };
        Image {
            width: EXTENT[0],
            height: EXTENT[1],
            pixels: Pixels::Rgba8(
                self.mean
                    .iter()
                    .flat_map(|p| [encode(p[0]), encode(p[1]), encode(p[2]), 255])
                    .collect(),
            ),
        }
    }

    /// FLIP of `self` against `reference`, both displayed at `exposure`.
    pub fn flip(&self, reference: &Accumulated, exposure: f32) -> FlipError {
        flip(&reference.display(exposure), &self.display(exposure))
    }
}

/// A bed view's guide targets, RGBA per pixel in rows of [`EXTENT`] top to bottom; unused
/// channels 0.
#[derive(Clone, Debug)]
pub struct GuideImages {
    pub diffuse_albedo: Vec<[f32; 4]>,
    pub specular_albedo: Vec<[f32; 4]>,
    pub normal_roughness: Vec<[f32; 4]>,
    pub specular_motion: Vec<[f32; 4]>,
    pub depth: Vec<[f32; 4]>,
    pub motion: Vec<[f32; 4]>,
}

/// A bed view's radiance and guide targets read at one point of a frame's list.
pub struct Capture {
    slots: Vec<u32>,
    root: Allocation<guide_read::Root>,
    read: ComputePipeline,
    guides: Allocation<[f32; 4]>,
    radiance: Allocation<[u16; 4]>,
}

impl Capture {
    pub fn new(view: &LightingView) -> Self {
        let d = device();
        let textures = [
            view.diffuse_albedo(),
            view.specular_albedo(),
            view.normal_roughness(),
            view.specular_motion(),
            view.depth(),
            view.motion(),
        ];
        let slots: Vec<u32> = textures
            .iter()
            .map(|_| d.alloc_image_slot().unwrap())
            .collect();
        let index: Vec<_> = textures
            .iter()
            .zip(&slots)
            .map(|(t, &slot)| d.write_sampled(slot, t, &orr_gpu::ViewDesc::default()))
            .collect();
        let guides = crate::readback::<[f32; 4]>(PIXELS * 6, "guide readback");
        let root = upload(
            &[guide_read::Root {
                diffuse_albedo: index[0],
                specular_albedo: index[1],
                normal_roughness: index[2],
                specular_motion: index[3],
                depth: index[4],
                motion: index[5],
                out: guides.gpu(),
                width: EXTENT[0],
                count: PIXELS as u32,
                _pad0: 0,
            }],
            "guide read root",
        );
        Self {
            slots,
            root,
            read: pipeline(&guide_read::MODULE, "main"),
            guides,
            radiance: crate::readback::<[u16; 4]>(PIXELS, "radiance readback"),
        }
    }

    /// Copies the targets as the work before it in `list` leaves them; order that work
    /// before compute and transfer.
    pub fn record(&self, list: &mut CommandList<'_>, view: &LightingView) {
        list.dispatch(
            &self.read,
            self.root.gpu(),
            [PIXELS.div_ceil(64) as u32, 1, 1],
        );
        let radiance = view.radiance();
        list.copy_from_texture(
            self.radiance.gpu().cast(),
            radiance,
            &TextureRegion::whole_mip(radiance.desc(), 0),
        );
        list.barrier(
            Stages::COMPUTE | Stages::TRANSFER,
            Stages::HOST,
            Hazards::empty(),
        );
    }

    /// The guides as the last completed recording read them.
    pub fn guides(&self) -> GuideImages {
        let v = self.guides.as_slice().unwrap();
        let plane = |k: usize| (0..PIXELS as usize).map(|i| v[i * 6 + k]).collect();
        GuideImages {
            diffuse_albedo: plane(0),
            specular_albedo: plane(1),
            normal_roughness: plane(2),
            specular_motion: plane(3),
            depth: plane(4),
            motion: plane(5),
        }
    }

    /// The RGBA16F radiance's bits as the last completed recording read them, in rows of
    /// [`EXTENT`].
    pub fn radiance(&self) -> &[[u16; 4]] {
        self.radiance.as_slice().unwrap()
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        for &slot in &self.slots {
            device().free_image_slot_after_submitted(slot);
        }
    }
}

/// One analytic scene mounted, placed and viewed, ready to accumulate frames.
pub struct Bed {
    pub spec: Spec,
    pub lighting: SceneLighting,
    scenes: Scenes,
    scene: SceneId,
    history: ViewHistoryId,
    mount: MountId,
    _environment: Option<(TestTexture, Allocation<EnvironmentMapGpu>)>,
    target: LightingView,
    /// Holds `target`'s targets.
    _arena: ViewArena,
    sum: Allocation<[f32; 4]>,
    out: Allocation<[f32; 4]>,
    roots: Roots,
    accumulate: ComputePipeline,
    exposure: ComputePipeline,
    pre_exposure: f32,
    frame: u64,
    last_id: u64,
    /// The scene root of the last frame run.
    last_root: GpuPtr<SceneRoot>,
    last_overlay: Option<TextureIndex>,
    light_writes: Vec<(LightId, Option<Light>)>,
    proxy_writes: Vec<(MediumProxyId, Option<MediumProxy>)>,
    scene_compute: Option<SceneComputeHook>,
    /// Frames add the background layer instead of the radiance.
    background_layer: bool,
    kernels: &'static KernelSet,
    particle_group: TransformGroupId,
    particles: Option<Particles>,
    _workers: Workers,
}

impl Bed {
    pub fn new(analytic: Analytic) -> Self {
        Self::with(analytic.spec())
    }

    pub fn with(spec: Spec) -> Self {
        Self::with_renderer(spec, &RendererDesc::default())
    }

    /// The bed for `spec` on a renderer created with `renderer`; orr_lighting's modules are
    /// replaced from its kernels too.
    pub fn with_renderer(spec: Spec, renderer: &RendererDesc) -> Self {
        Self::with_linked(spec, renderer, RT_RANGE, |_| {})
    }

    /// As [`Self::with_renderer`], with the copies ray traced from up to `rt_range` m away and
    /// `linked` setting the scene's data for a linked environment and medium in its first
    /// frame.
    pub fn with_linked(
        spec: Spec,
        renderer: &RendererDesc,
        rt_range: f32,
        linked: impl FnOnce(&mut SceneWriter<'_>),
    ) -> Self {
        let kernels = renderer.kernels;
        let d = device();
        let mut scenes = Scenes::new(d, renderer).unwrap();
        let workers = Workers::start(&scenes);
        let mount = scenes.mount(PackSource::Memory(Arc::clone(pack())));
        let start = Instant::now();
        let mut frame = 0;
        while matches!(scenes.mount_state(mount), LoadState::Pending) {
            frame += 1;
            let f = scenes
                .begin_frame(&FrameDesc {
                    time: time(frame),
                    scenes: &[],
                    low_latency: None,
                })
                .unwrap();
            f.end(&[]).unwrap();
            assert!(start.elapsed() < Duration::from_secs(60), "mount pending");
            std::thread::sleep(Duration::from_millis(1));
        }
        if let LoadState::Failed(e) = scenes.mount_state(mount) {
            panic!("mount failed: {e}");
        }

        let desc = SceneDesc {
            light_capacity: spec.lights.len().max(8) as u32,
            producer_capacity: PRODUCER_CAPACITY,
            ..SceneDesc::default()
        };
        let scene = scenes.create_scene(&desc).unwrap();
        let history = scenes.create_view_history().unwrap();
        let environment = spec.environment.map(|c| {
            let map = texture(2, 2, &[[c[0], c[1], c[2], 1.0]; 4], "bed environment");
            let data = upload(
                &[EnvironmentMapGpu {
                    map: map.index,
                    sampler: linear_clamp(),
                    intensity: 1.0,
                    primaries: orr_color::Primaries::Rec2020.id().into(),
                    from_world: Affine3::IDENTITY.into(),
                }],
                "bed environment data",
            );
            (map, data)
        });
        let models: Vec<_> = spec
            .copies
            .iter()
            .map(|c| {
                scenes
                    .model(mount, AssetId::from_name(c.model))
                    .unwrap_or_else(|e| panic!("model {}: {e}", c.model))
            })
            .collect();
        frame += 1;
        let list = [scene];
        let f = scenes
            .begin_frame(&FrameDesc {
                time: time(frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        let mut w = f.scene(scene).unwrap();
        let group = w
            .create_transform_group(&TransformGroupDesc {
                count: spec.copies.len() as u32,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        for (i, (c, &model)) in spec.copies.iter().zip(&models).enumerate() {
            w.transforms(group, i as u32, 1).unwrap()[0] =
                Affine3::from_mat3_translation(c.linear, c.at).into();
            w.add_instances(&InstancesDesc {
                model,
                copies: 1,
                transforms: group,
                first: i as u32,
                pose: None,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range,
                animation_range: 0.0,
                user_tag: i as u32,
            })
            .unwrap();
        }
        let particle_group = w
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        w.transforms(particle_group, 0, 1).unwrap()[0] = Affine3::IDENTITY.into();
        for (i, l) in spec.lights.iter().enumerate() {
            w.set_light(LightId(i as u32), Some(l)).unwrap();
        }
        if let Some((_, data)) = &environment {
            w.set_environment(data.gpu());
        }
        linked(&mut w);
        if let Some((at, pixels_per_radian)) = spec.focus {
            w.set_focus(at.as_dvec3());
            f.request_detail(scene, pixels_per_radian).unwrap();
        }
        drop(w);
        let point = f.end(&[]).unwrap();
        render().wait(point, Duration::from_secs(10)).unwrap();

        let lighting =
            SceneLighting::new(d, kernels, &desc, &SceneLightingSettings::default()).unwrap();
        let buffer = |kind, name| d.gpu_malloc::<[f32; 4]>(kind, PIXELS, name).unwrap();
        let mut arena = ViewArena::new(d, false);
        let mut list = render().command_list().unwrap();
        arena
            .layout(&LightingView::requests(EXTENT, false), &mut list)
            .unwrap();
        let point = render().submit(vec![list], &[]).unwrap();
        render().wait(point, Duration::from_secs(10)).unwrap();
        Self {
            spec,
            lighting,
            scenes,
            scene,
            history,
            mount,
            _environment: environment,
            target: LightingView::new(d, &arena).unwrap(),
            _arena: arena,
            sum: buffer(MemoryKind::DeviceLocal, "bed sum"),
            out: buffer(MemoryKind::Readback, "bed readback"),
            roots: Roots {
                memory: d
                    .gpu_malloc::<u8>(MemoryKind::Upload, ROOT_BYTES, "bed roots")
                    .unwrap(),
                used: 0,
            },
            accumulate: pipeline(&accumulate::MODULE, "main"),
            exposure: pipeline(&exposure::MODULE, "main"),
            pre_exposure: 1.0,
            frame,
            last_id: 0,
            last_root: GpuPtr::NULL,
            last_overlay: None,
            light_writes: Vec::new(),
            proxy_writes: Vec::new(),
            scene_compute: None,
            background_layer: false,
            kernels,
            particle_group,
            particles: None,
            _workers: workers,
        }
    }

    /// Adds a skinned copy of model `model` from the next frame on, placed by `place` and posed
    /// by the skinning matrices `palette`, one per joint of its skeleton.
    pub fn add_skinned(&mut self, model: &str, place: Affine3, palette: &[Affine3]) {
        let model = self
            .scenes
            .model(self.mount, AssetId::from_name(model))
            .unwrap_or_else(|e| panic!("model {model}: {e}"));
        self.frame += 1;
        let list = [self.scene];
        let f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        {
            let mut w = f.scene(self.scene).unwrap();
            let group = |w: &mut SceneWriter<'_>, transforms: &[Affine3]| {
                let g = w
                    .create_transform_group(&TransformGroupDesc {
                        count: transforms.len() as u32,
                        writer: GroupWriter::Cpu,
                        anchor: DVec3::ZERO,
                    })
                    .unwrap();
                let out = w.transforms(g, 0, transforms.len() as u32).unwrap();
                for (o, &t) in out.iter_mut().zip(transforms) {
                    *o = t.into();
                }
                g
            };
            let transforms = group(&mut w, &[place]);
            let palette = group(&mut w, palette);
            w.add_skinned(&SkinnedDesc {
                model,
                copies: 1,
                transforms,
                first: 0,
                palette,
                palette_first: 0,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: RT_RANGE,
                user_tag: 0,
            })
            .unwrap();
        }
        let point = f.end(&[]).unwrap();
        render().wait(point, Duration::from_secs(10)).unwrap();
    }

    /// The state the last frame left skinned entry `entry` in, the bed's skinned copies taking
    /// entries in the order [`Self::add_skinned`] added them.
    pub fn skin_state(&self, entry: u32) -> SkinStateGpu {
        let root = read(self.last_root, 1)[0];
        read(root.skin_state.offset(u64::from(entry)), 1)[0]
    }

    /// Gives the bed's particle entry `particles` of `look` from the next frame on, traced
    /// and drawn as `opacity` says (`Opacity::Particle` for particles, `Opacity::Smoke` for
    /// smoke); each frame emits them and draws particle entries at the transparency point after
    /// the frame's kernel, lit by the lighting view's particle lighting of that frame, where it
    /// wrote one.
    pub fn set_particles(&mut self, particles: &[Particle], look: ParticleLook, opacity: Opacity) {
        let mut m = particle_material(&look);
        if matches!(opacity, Opacity::Particle | Opacity::Smoke) {
            m.flags |= material_flags::PRESENCE;
        }
        self.set_quads(
            particles,
            m,
            None,
            Affine3::IDENTITY,
            look.anisotropy,
            opacity,
        );
    }

    /// As [`Self::set_particles`], but opaque squares showing `front` on their front and `back`
    /// on a back of its own, placed by `transform`.
    pub fn set_sided_quads(
        &mut self,
        particles: &[Particle],
        front: ParticleLook,
        back: ParticleLook,
        transform: Affine3,
    ) {
        let (front, back) = (particle_material(&front), particle_material(&back));
        self.set_quads(
            particles,
            front,
            Some(back),
            transform,
            0.0,
            Opacity::Opaque,
        );
    }

    fn set_quads(
        &mut self,
        particles: &[Particle],
        mut m: MaterialGpu,
        back: Option<MaterialGpu>,
        transform: Affine3,
        anisotropy: f32,
        opacity: Opacity,
    ) {
        self.frame += 1;
        let list = [self.scene];
        let f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        if let Some(b) = back {
            m.back = f.materials().create(&b).unwrap().0;
            m.flags |= material_flags::HAS_BACK;
        }
        let material = f.materials().create(&m).unwrap();
        let count = particles.len() as u32;
        let topology = [ClusterTopology {
            vertex_count: 4,
            triangles: &PARTICLE_QUAD,
        }];
        {
            let mut w = f.scene(self.scene).unwrap();
            w.transforms(self.particle_group, 0, 1).unwrap()[0] = transform.into();
            w.set_producer(
                PARTICLES,
                Some(&ProducerDesc {
                    max_clusters: count.max(1),
                    max_vertices: 4 * count.max(1),
                    update: ProducerUpdate::EveryFrame,
                    topologies: &topology,
                    materials: &[material],
                    transform: (self.particle_group, 0),
                    opacity,
                    ..ProducerDesc::default()
                }),
            )
            .unwrap();
        }
        let point = f.end(&[]).unwrap();
        render().wait(point, Duration::from_secs(10)).unwrap();
        let mut list = device()
            .gpu_malloc::<[f32; 4]>(MemoryKind::Upload, u64::from(count.max(1)), "bed particles")
            .unwrap();
        for (o, p) in list.as_mut_slice().unwrap().iter_mut().zip(particles) {
            *o = [p.center.x, p.center.y, p.center.z, p.size];
        }
        self.particles = Some(Particles {
            list,
            count,
            emit: pipeline(&particle_quads::MODULE, "main"),
            draw: ParticleDraw::new(device(), self.kernels).unwrap(),
            anisotropy,
        });
    }

    /// Runs `frames` frames of the scene, each recording `kernel` after the scene pass and
    /// adding the radiance it wrote, and returns the mean. A frame's index and `seed`
    /// choose its samples, except the light tiles', which follow the scene's frame id. That
    /// starts wherever mounting left it and keeps counting across calls, so with an
    /// environment, local lights or emitters the same `frames` and `seed` can give different
    /// images.
    pub fn accumulate(
        &mut self,
        frames: u32,
        seed: u32,
        kernel: &mut dyn FnMut(&mut CommandList<'_>, &mut BedFrame<'_>),
    ) -> Accumulated {
        self.accumulate_from(0, frames, seed, kernel)
    }

    /// As [`Self::accumulate`], with frame indices from `first`.
    pub fn accumulate_from(
        &mut self,
        first: u32,
        frames: u32,
        seed: u32,
        kernel: &mut dyn FnMut(&mut CommandList<'_>, &mut BedFrame<'_>),
    ) -> Accumulated {
        assert!(frames > 0);
        for k in 0..frames {
            self.run_frame(first + k, seed, k == 0, k == 0, k + 1 == frames, kernel);
        }
        self.mean(frames)
    }

    /// As [`Self::accumulate_from`], with no camera cut: the view's history, light feedback and
    /// local light samplers carry on from the frames before.
    pub fn accumulate_on(
        &mut self,
        first: u32,
        frames: u32,
        seed: u32,
        kernel: &mut dyn FnMut(&mut CommandList<'_>, &mut BedFrame<'_>),
    ) -> Accumulated {
        assert!(frames > 0);
        for k in 0..frames {
            self.run_frame(first + k, seed, false, k == 0, k + 1 == frames, kernel);
        }
        self.mean(frames)
    }

    /// Moves the camera to `eye`, looking at `target`, and renders one frame as the view's
    /// next, with no camera cut; returns its radiance.
    pub fn follow(
        &mut self,
        eye: Vec3,
        target: Vec3,
        index: u32,
        kernel: &mut dyn FnMut(&mut CommandList<'_>, &mut BedFrame<'_>),
    ) -> Accumulated {
        self.spec.eye = eye;
        self.spec.target = target;
        self.run_frame(index, 0, false, true, true, kernel);
        self.mean(1)
    }

    fn mean(&self, frames: u32) -> Accumulated {
        let n = frames as f32;
        Accumulated {
            mean: self
                .out
                .as_slice()
                .unwrap()
                .iter()
                .map(|p| p.map(|v| v / n))
                .collect(),
        }
    }

    /// The first material of the bed's model `model`.
    pub fn material(&self, model: &str) -> MaterialId {
        self.scenes
            .material(
                self.mount,
                AssetId::from_name(&format!("{model}/material/0")),
            )
            .unwrap_or_else(|e| panic!("material of {model}: {e}"))
    }

    /// The orr_scene frame id of the last frame run.
    pub fn last_frame(&self) -> u64 {
        self.last_id
    }

    /// Sets light `id` from the next frame on.
    pub fn set_light(&mut self, id: LightId, light: Option<Light>) {
        self.light_writes.push((id, light));
    }

    /// The froxel volume's depth from the next frame on, m (`SceneRoot::near_range`).
    pub fn set_near_range(&mut self, near_range: f32) {
        self.scenes.set_near_range(self.scene, near_range).unwrap();
    }

    /// Sets medium proxy `id` from the next frame on.
    pub fn set_medium_proxy(&mut self, id: MediumProxyId, proxy: Option<MediumProxy>) {
        self.proxy_writes.push((id, proxy));
    }

    /// The bed's cooked volume `name`.
    pub fn volume(&self, name: &str) -> VolumeRef {
        self.scenes
            .volume(self.mount, AssetId::from_name(name))
            .unwrap_or_else(|e| panic!("volume {name}: {e}"))
    }

    /// Runs `hook` at the scene compute point of every frame from the next on.
    pub fn set_scene_compute(&mut self, hook: SceneComputeHook) {
        self.scene_compute = Some(hook);
    }

    /// The bed view's lighting targets as the last frame left them.
    pub fn view(&self) -> &LightingView {
        &self.target
    }

    /// Whether later frames add the view's background layer to the accumulation instead of
    /// its radiance.
    pub fn accumulate_background_layer(&mut self, on: bool) {
        self.background_layer = on;
    }

    /// Renders later frames at pre-exposure `p`, as the output's meter would set it.
    pub fn set_pre_exposure(&mut self, p: f32) {
        self.pre_exposure = p;
        self.run_frame(0, 0, true, true, false, &mut |_, _| {});
    }

    /// Reads the guide targets of the last frame.
    pub fn guides(&self) -> GuideImages {
        let capture = Capture::new(&self.target);
        let mut list = render().command_list().unwrap();
        list.barrier(
            Stages::ALL,
            Stages::COMPUTE | Stages::TRANSFER,
            Hazards::empty(),
        );
        capture.record(&mut list, &self.target);
        crate::finish(list);
        capture.guides()
    }

    fn run_frame(
        &mut self,
        index: u32,
        seed: u32,
        cut: bool,
        first: bool,
        last: bool,
        kernel: &mut dyn FnMut(&mut CommandList<'_>, &mut BedFrame<'_>),
    ) {
        let eye = self.spec.eye;
        self.frame += 1;
        self.roots.used = 0;
        let list = [self.scene];
        let mut f = self
            .scenes
            .begin_frame(&FrameDesc {
                time: time(self.frame),
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        let desc = ViewDesc {
            scene: self.scene,
            history: self.history,
            position: eye.as_dvec3(),
            rotation: look(eye, self.spec.target),
            projection: Projection {
                fov_y: self.spec.fov_y,
                near: NEAR,
            },
            lens: Lens::default(),
            render_extent: UVec2::from(EXTENT),
            output_extent: UVec2::from(EXTENT),
            jitter: Vec2::ZERO,
            camera_cut: cut,
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
        };
        if !self.light_writes.is_empty() || !self.proxy_writes.is_empty() {
            let mut w = f.scene(self.scene).unwrap();
            for (id, l) in self.light_writes.drain(..) {
                w.set_light(id, l.as_ref()).unwrap();
            }
            for (id, p) in self.proxy_writes.drain(..) {
                w.set_medium_proxy(id, p.as_ref()).unwrap();
            }
        }
        if self.scene_compute.is_some() || self.particles.is_some() {
            let (hook, particles) = (&mut self.scene_compute, &self.particles);
            let eye = [eye.x, eye.y, eye.z, 0.0];
            f.update_scenes(&mut |c| {
                if let Some(hook) = hook {
                    hook(c);
                }
                let Some(p) = particles else {
                    return;
                };
                let root = c
                    .put_root(&particle_quads::Root {
                        scene: c.root,
                        particles: p.list.gpu(),
                        eye,
                        producer: PARTICLES.0,
                        count: p.count,
                        _pad: 0,
                    })
                    .unwrap();
                c.cmd
                    .dispatch(&p.emit, root, [p.count.div_ceil(64).max(1), 1, 1]);
            })
            .unwrap();
        }
        let view = f.raster_view(&desc).unwrap();
        let id = f.id();
        self.last_id = id;
        self.last_root = view.scene;
        let list = f.list().unwrap();
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.lighting.record(
            list,
            &ScenePass {
                frame: id,
                scene: view.scene,
                focus: eye.to_array(),
            },
        );
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        kernel(
            list,
            &mut BedFrame {
                index,
                seed,
                frame: id,
                view: &view,
                desc,
                lighting: &self.lighting,
                target: &mut self.target,
                particles: self.particles.is_some(),
                roots: &mut self.roots,
            },
        );
        if let Some(p) = &self.particles {
            let lighting = self.target.particle_lighting(id);
            let radiance = self.target.radiance_storage();
            let (_, overlay) = f
                .transparency(&view, radiance, &mut |pass| {
                    p.draw.record(pass, lighting, p.anisotropy).unwrap();
                })
                .unwrap();
            self.last_overlay = Some(overlay);
        }
        let list = f.list().unwrap();
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::COMPUTE,
            Hazards::empty(),
        );
        let root = self.roots.put(&accumulate::Root {
            sum: self.sum.gpu(),
            radiance: if self.background_layer {
                self.target
                    .dof_background_index()
                    .expect("the frame traced no background layer")
            } else {
                self.target.radiance_index()
            },
            width: EXTENT[0],
            count: PIXELS as u32,
            first: u32::from(first),
            _pad0: 0,
            _pad1: 0,
        });
        list.dispatch(&self.accumulate, root, [PIXELS.div_ceil(64) as u32, 1, 1]);
        let root = self.roots.put(&exposure::Root {
            view: view.view,
            pre_exposure: self.pre_exposure,
            _pad: 0,
        });
        list.dispatch(&self.exposure, root, [1, 1, 1]);
        if last {
            list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
            list.copy(
                self.out.gpu().cast(),
                self.sum.gpu().cast(),
                PIXELS * size_of::<[f32; 4]>() as u64,
            );
            list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
        }
        let point = f.end(&[]).unwrap();
        render().wait(point, Duration::from_secs(10)).unwrap();
        let stats = self.lighting.stats(id).unwrap();
        assert_eq!(
            stats.photon_overflows, 0,
            "the frame before {id} filled a reference photon map"
        );
    }
}

/// The flat-color stand-in for a lighting kernel (`flat_color.slang`): each pixel's base
/// color, or `sky` where nothing was drawn, times a uniform draw in [0, 2). Alpha is 1 where
/// a surface was drawn and 0 for sky.
pub struct FlatColor {
    pipeline: ComputePipeline,
    pub sky: [f32; 3],
}

impl FlatColor {
    pub fn new(sky: [f32; 3]) -> Self {
        Self {
            pipeline: pipeline(&flat_color::MODULE, "main"),
            sky,
        }
    }

    pub fn record(&self, list: &mut CommandList<'_>, f: &mut BedFrame<'_>) {
        let root = f.put(&flat_color::Root {
            view: f.view.view,
            gbuffer: f.view.gbuffer,
            radiance: f.target.radiance_storage(),
            sky: [self.sky[0], self.sky[1], self.sky[2], 0.0],
            frame: f.index,
            seed: f.seed,
            _pad0: 0,
        });
        list.dispatch(
            &self.pipeline,
            root,
            [EXTENT[0].div_ceil(8), EXTENT[1].div_ceil(8), 1],
        );
    }
}

/// Settings of the reference integrator with `max_bounces`.
pub fn reference(max_bounces: u32) -> ViewLighting {
    ViewLighting {
        max_bounces,
        reference: Some(ReferenceSettings::default()),
        ..ViewLighting::default()
    }
}

/// Luminance of a pixel of linear Rec.2020 radiance.
pub fn luminance(p: &[f32; 4]) -> f64 {
    orr_color::luminance(DVec3::new(p[0].into(), p[1].into(), p[2].into()))
}

/// Mean rgb over the pixels that pass `keep`.
pub fn mean_where(a: &Accumulated, keep: impl Fn(&[f32; 4]) -> bool) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    let mut n = 0.0;
    for p in a.mean.iter().filter(|p| keep(p)) {
        for c in 0..3 {
            sum[c] += f64::from(p[c]);
        }
        n += 1.0;
    }
    assert!(n > 0.0, "no pixels selected");
    sum.map(|s| s / n)
}

/// Mean rgb over the pixels a surface or layer was drawn on.
pub fn surface_mean(a: &Accumulated) -> [f64; 3] {
    mean_where(a, |p| p[3] > 0.5)
}

/// Mean rgb over the whole image.
pub fn image_mean(a: &Accumulated) -> [f64; 3] {
    mean_where(a, |_| true)
}

/// Asserts each channel of `a` is within `tolerance` of `b`'s, relative to `b`.
pub fn assert_close(what: &str, a: [f64; 3], b: [f64; 3], tolerance: f64) {
    for c in 0..3 {
        let e = (a[c] - b[c]).abs() / b[c].abs().max(1e-12);
        assert!(e <= tolerance, "{what}: channel {c} {a:?} against {b:?}");
    }
}

/// The alpha quad scene with `model` for its quad.
pub fn quad_scene(model: &'static str) -> Spec {
    let mut spec = Analytic::AlphaQuad.spec();
    spec.copies[1].model = model;
    spec
}

/// Pixels a side of the blocks [`assert_within_noise`] averages.
const NOISE_BLOCK: u32 = 8;

/// How far above the seed-to-seed difference [`assert_within_noise`] lets a block difference
/// sit. Over 48 seeds correct images reach 1.25 of it with linked coverage and 0.86 with
/// micromaps; coverage wrong only in the shadow it casts starts at 2.0.
const BLOCK_MARGIN: f64 = 1.6;

/// Rms over [`NOISE_BLOCK`] pixel blocks of the difference in mean luminance of `a` and `b`.
fn block_difference(a: &Accumulated, b: &Accumulated) -> f64 {
    let half = NOISE_BLOCK / 2;
    let [columns, rows] = EXTENT.map(|e| e / NOISE_BLOCK);
    let block = |image: &Accumulated, at| orr_color::luminance(DVec3::from(image.around(at, half)));
    let squares = (0..columns * rows)
        .map(|i| {
            let at = (UVec2::new(i % columns, i / columns) * NOISE_BLOCK + half).as_vec2();
            (block(a, at) - block(b, at)).powi(2)
        })
        .sum::<f64>();
    (squares / f64::from(columns * rows)).sqrt()
}

/// Asserts `a` shows what `b` shows up to sampling noise, where `a` has `b`'s seed and `noise`
/// is `b`'s scene at another seed. Block means catch a wrong shadow, too small a share of the
/// image to lift FLIP clear of the noise; FLIP, at up to `flip_margin` times its value for
/// `noise` and an exposure putting `b`'s mean at 0.18, catches detail finer than a block.
pub fn assert_within_noise(
    what: &str,
    a: &Accumulated,
    b: &Accumulated,
    noise: &Accumulated,
    flip_margin: f32,
) {
    let mean = b.mean.iter().map(luminance).sum::<f64>() / b.mean.len() as f64;
    let exposure = (0.18 / mean) as f32;
    let (e, floor) = (a.flip(b, exposure).mean, noise.flip(b, exposure).mean);
    assert!(
        e <= flip_margin * floor,
        "{what}: FLIP {e}, {floor} between seeds"
    );
    let (e, floor) = (block_difference(a, b), block_difference(noise, b));
    assert!(floor > 0.0, "{what}: no difference between seeds");
    assert!(
        e <= BLOCK_MARGIN * floor,
        "{what}: block difference {e}, {floor} between seeds"
    );
}

/// orr_lighting's view dispatch and guide resolve recording into bed frames with
/// `settings`.
pub struct Paths {
    kernels: &'static LightingKernels,
    /// Where the kernels compile the path pipelines a frame first asks for.
    loading: &'static orr_scene::Loading,
    pub settings: ViewLighting,
    /// The background layer request of every frame.
    pub dof_background: Option<DofBackgroundPass>,
    /// The depth-of-field guide every frame adds the frosted-glass term into.
    pub frost_guide: Option<FrostGuide>,
    /// Every frame keeps the view's front layer, which [`Self::overlay_front`] reads.
    pub overlay_front: bool,
    /// The kernels read a G-buffer without layer tiles, so the layer guides trace
    /// every tile, as a full-screen dispatch does.
    pub every_tile: bool,
}

impl Paths {
    /// The kernels of a set and entry compile once per process, on threads of their own, and
    /// stay: under the validation layer each compile takes minutes.
    pub fn new(kernels: &KernelSet, entry: PathEntry, settings: ViewLighting) -> Self {
        type Key = (Vec<usize>, PathEntry);
        type Compiled = (LightingKernels, Arc<orr_scene::Loading>);
        static COMPILED: Mutex<Vec<(Key, &'static OnceLock<Compiled>)>> = Mutex::new(Vec::new());
        let key = (
            kernels
                .modules
                .iter()
                .map(|m| ptr::from_ref(*m).addr())
                .collect(),
            entry,
        );
        let mut compiled = COMPILED.lock().unwrap_or_else(PoisonError::into_inner);
        let slot = match compiled.iter().find(|(k, _)| *k == key) {
            Some((_, slot)) => *slot,
            None => {
                let slot: &'static OnceLock<_> = Box::leak(Box::default());
                compiled.push((key, slot));
                slot
            }
        };
        drop(compiled);
        let (kernels, loading) = slot.get_or_init(|| {
            let tables = BsdfTables::new(device()).unwrap();
            let kernels: &'static KernelSet = Box::leak(Box::new(*kernels));
            let loading = Arc::new(orr_scene::testing::loading());
            let jobs = LightingKernels::spawn(device(), kernels, &tables, entry, &loading);
            std::thread::scope(|threads| {
                for _ in 1..CompilingKernels::JOBS {
                    threads.spawn(|| while loading.run_one() {});
                }
                jobs.wait(&loading);
            });
            (jobs.finish().unwrap(), loading)
        });
        Self {
            kernels,
            loading,
            // The beds' scenes are many and small: one set of pipelines runs them all.
            settings: ViewLighting {
                specialize: false,
                ..settings
            },
            dof_background: None,
            frost_guide: None,
            overlay_front: false,
            every_tile: false,
        }
    }

    /// Starts the view's frame, then records the froxel passes, the view's paths, media apply,
    /// the guide resolve and, with a frost guide, the frosted-glass term.
    pub fn record(&self, list: &mut CommandList<'_>, f: &mut BedFrame<'_>) {
        let pass = self.view_pass(f);
        self.record_into(list, f, &pass, None);
    }

    /// As [`Self::record`], into `target` instead of the frame's view.
    pub fn record_on(
        &self,
        list: &mut CommandList<'_>,
        f: &mut BedFrame<'_>,
        target: &mut LightingView,
    ) {
        let pass = self.view_pass(f);
        self.record_into(list, f, &pass, Some(target));
    }

    fn view_pass(&self, f: &mut BedFrame<'_>) -> ViewPass {
        let mut pass = ViewPass {
            dof_background: self.dof_background,
            frosted_glass: self.frost_guide.is_some(),
            overlay_front: self.overlay_front,
            ..f.pass()
        };
        if self.every_tile {
            let mut g = f.view.targets;
            g.layer_tiles = GpuPtr::NULL;
            pass.gbuffer = f.put(&g);
        }
        pass
    }

    fn record_into(
        &self,
        list: &mut CommandList<'_>,
        f: &mut BedFrame<'_>,
        pass: &ViewPass,
        target: Option<&mut LightingView>,
    ) {
        let target = match target {
            Some(t) => t,
            None => &mut *f.target,
        };
        let (lighting, settings) = (f.lighting, &self.settings);
        target.begin(lighting, &f.desc, settings);
        let needs = PathNeeds {
            features: pass.scene_features,
            frosted_glass: pass.frosted_glass,
            dof_background: pass.dof_background.is_some(),
        };
        if !self.kernels.prepare_paths(lighting, &needs, settings) {
            // The pipelines these paths first ask for compile on threads of their own.
            std::thread::scope(|threads| {
                for _ in 1..CompilingKernels::JOBS {
                    threads.spawn(|| while self.loading.run_one() {});
                }
                self.kernels.wait_paths(lighting, &needs, settings).unwrap();
            });
        }
        self.kernels
            .record_media(list, lighting, target, pass, settings)
            .unwrap();
        self.kernels
            .record_paths(list, lighting, target, pass, settings)
            .unwrap();
        let paths = Stages::COMPUTE | Stages::RAY;
        list.barrier(paths, Stages::COMPUTE, Hazards::empty());
        self.kernels
            .record_media_apply(list, lighting, target, pass, settings);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        self.kernels.record_guides(list, target, pass, settings);
        if let Some(g) = &self.frost_guide {
            g.fill(list);
            self.kernels
                .record_frosted_glass(list, target, pass, g.store);
        }
    }

    /// Resolves the front layer of `bed`'s last frame and reads it: depth, motion and weight per
    /// pixel, in rows of [`EXTENT`] top to bottom.
    pub fn overlay_front(&self, bed: &mut Bed) -> OverlayFrontLayer {
        let overlay = bed.last_overlay;
        let mut list = render().command_list().unwrap();
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.kernels
            .record_overlay_front(&mut list, &mut bed.target, overlay);
        let view = bed.view();
        let depth = view.overlay_depth().expect("the frame kept no front layer");
        let motion = view
            .overlay_motion()
            .expect("the frame kept no front layer");
        let weight = view
            .overlay_weight()
            .expect("the frame kept no front layer");
        let depth_out = crate::readback::<f32>(PIXELS, "overlay depth readback");
        let motion_out = crate::readback::<[u16; 2]>(PIXELS, "overlay motion readback");
        let weight_out = crate::readback::<u16>(PIXELS, "overlay weight readback");
        list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
        list.copy_from_texture(
            weight_out.gpu().cast(),
            weight,
            &TextureRegion::whole_mip(weight.desc(), 0),
        );
        list.copy_from_texture(
            depth_out.gpu().cast(),
            depth,
            &TextureRegion::whole_mip(depth.desc(), 0),
        );
        list.copy_from_texture(
            motion_out.gpu().cast(),
            motion,
            &TextureRegion::whole_mip(motion.desc(), 0),
        );
        crate::finish(list);
        let half = |b: u16| half::f16::from_bits(b).to_f32();
        OverlayFrontLayer {
            depth: depth_out.as_slice().unwrap().to_vec(),
            motion: motion_out
                .as_slice()
                .unwrap()
                .iter()
                .map(|m| m.map(half))
                .collect(),
            weight: weight_out
                .as_slice()
                .unwrap()
                .iter()
                .map(|w| half(*w))
                .collect(),
        }
    }
}

/// A view's front layer per pixel.
pub struct OverlayFrontLayer {
    pub depth: Vec<f32>,
    pub motion: Vec<[f32; 2]>,
    pub weight: Vec<f32>,
}

/// A lighting view at [`EXTENT`] besides a bed's own, for kernels that record into a second
/// set of targets in the same frame.
pub struct SpareView {
    pub view: LightingView,
    _arena: ViewArena,
}

impl SpareView {
    pub fn new() -> Self {
        let d = device();
        let mut arena = ViewArena::new(d, false);
        let mut list = render().command_list().unwrap();
        arena
            .layout(&LightingView::requests(EXTENT, false), &mut list)
            .unwrap();
        let point = render().submit(vec![list], &[]).unwrap();
        render().wait(point, Duration::from_secs(10)).unwrap();
        Self {
            view: LightingView::new(d, &arena).unwrap(),
            _arena: arena,
        }
    }
}

impl Default for SpareView {
    fn default() -> Self {
        Self::new()
    }
}

/// An R16F depth-of-field guide at [`EXTENT`] that each frame sets to one value, as output's
/// guide pass would write it, before the frosted-glass term.
pub struct FrostGuide {
    texture: Texture,
    region: TextureRegion,
    store: StorageImageIndex,
    fill: Allocation<u16>,
}

impl FrostGuide {
    pub fn new(value: f32) -> Self {
        let d = device();
        let desc = TextureDesc::d2(
            Format::R16Float,
            EXTENT[0],
            EXTENT[1],
            TextureUsage::SAMPLED | TextureUsage::STORAGE,
        );
        let texture = d
            .create_texture(&desc, TextureMemory::Dedicated, "frost guide")
            .unwrap();
        let store = d.write_storage(
            d.alloc_image_slot().unwrap(),
            &texture,
            &orr_gpu::ViewDesc::default(),
        );
        let bits = half::f16::from_f32(value).to_bits();
        Self {
            texture,
            region: TextureRegion::whole_mip(&desc, 0),
            store,
            fill: upload(&vec![bits; PIXELS as usize], "frost guide fill"),
        }
    }

    fn fill(&self, list: &mut CommandList<'_>) {
        list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
        list.discard(&self.texture);
        list.copy_to_texture(&self.texture, &self.region, self.fill.gpu().cast());
        list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    }

    /// The guide as the last frame left it, in rows of [`EXTENT`] top to bottom.
    pub fn read(&self) -> Vec<f32> {
        let out = crate::readback::<u16>(PIXELS, "frost guide readback");
        let mut list = render().command_list().unwrap();
        list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
        list.copy_from_texture(out.gpu().cast(), &self.texture, &self.region);
        crate::finish(list);
        let bits = out.as_slice().unwrap();
        bits.iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect()
    }
}

impl Drop for FrostGuide {
    fn drop(&mut self) {
        device().free_image_slot_after_submitted(self.store.0);
    }
}
