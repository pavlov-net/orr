//! The dense vegetation test scene: a 1 km heightfield tile of grid-patch producer clusters,
//! a forest of trees with rigid parts and discrete levels swaying inside their animation
//! range, procedural grass in a meadow, a wave-displaced dielectric pond, emissive particles
//! drawn at the transparency point, a low limb-darkened sun and a textured moon it lights,
//! all path traced through `render_view`. Trees come from cooked archives in
//! `$ORR_CONTENT/cooked`, or with `--synthetic` from stand-in trees of four species and three
//! variants cooked in process. Camera scripts walk, turn, look from above and fly through it
//! on a fixed clock, so runs repeat; each frame's figures go into a [`Report`].
//!
//! Switches (all optional):
//! `--synthetic`, `--trees N`, `--species N`, `--grass-density D` (blades per m^2),
//! `--grass-radius M`, `--parts all|trunk`, `--poses-per-variant N`, `--pose-stride`,
//! `--no-part-animation`, `--rt-range M`, `--animation-range M`, `--directional-lights L,T`,
//! `--sw-raster-px P`, `--lod0-only`, `--world-km K`, `--world-density N` (trees per km^2
//! outside the tile), `--particles N`, `--stream-load`, `--extent WxH`, `--script NAME`
//! (walk, turn, overview, flythrough) and `--script-frames N`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orr::orr_gpu::{
    Allocation, Blend, ColorAttachment, ColorLoad, ColorMask, ColorTarget, CompareOp,
    DepthAttachment, DepthLoad, DepthTarget, Device, Format, Hazards, MemoryKind, MeshPipeline,
    MeshPipelineDesc, QueueKind, RasterState, Rendering, SamplerIndex, ShaderStage, SpecData,
    Stages, Texture, TextureDesc, TextureMemory, TextureUsage, TimelinePoint,
};
use orr::orr_lighting::EnvironmentMapGpu;
use orr::orr_pack::material::{Material, slot};
use orr::orr_pack::mesh::MeshInfo;
use orr::orr_pack::model::ModelInfo;
use orr::orr_pack::{Archive, AssetId, Target};
use orr::{
    Affine3, BoundTexture, ClusterTopology, CopyVariant, DirectionalLight, DiscProfile, Display,
    DisplayRequest, FrameDesc, FrameStats, FrameTime, GroupWriter, InstanceFlags, InstancesDesc,
    Lens, Light, LightId, LightingStats, LoadState, MaterialGpu, ModelId, MountId, Opacity,
    PackSource, PipelineId, Pose, ProducerDesc, ProducerId, ProducerUpdate, Projection, Renderer,
    RendererDesc, SceneCompute, SceneDesc, SceneId, SceneLightingSettings, ScreenshotRequest,
    TransformGroupDesc, TransformGroupId, TransparencyPass, ViewDesc, ViewHistoryId, ViewHooks,
    ViewLighting, ViewOutput, ViewTarget,
};
use orr_cook::{ArchiveSpec, AssetSpec, LodMode, Manifest, MeshSettings, Quality, Source};
use orr_cook::{TextureSettings, Usage};
use orr_math::camera::rh::view;
use orr_math::{DVec3, Mat3, UVec2, Vec2, Vec3};

use crate::field::{Grass, Heights, Slab, Terrain, blade_triangles, patch_triangles};
use crate::mesh::MeshBuilder;
use crate::shaders::{fireflies, glow, grass, slab, sway, terrain};
use crate::sky::{self, Sky};
use crate::window::Window;
use crate::{Capture, Options, Workers, affine, device, fixed_time, gray_png};

/// The tile's side, m; it spans -TILE / 2 to TILE / 2 on x and z.
const TILE: f32 = 1024.0;
/// Height samples per m on the tile.
const TILE_SPACING: f32 = 1.0;
/// Patches of 8 m on the tile.
const TILE_PATCH: f32 = 8.0;
/// Far terrain, past the tile under `--world-km`: samples and patches.
const FAR_SPACING: f32 = 8.0;
const FAR_PATCH: f32 = 64.0;
/// The meadow: a 100 m square of grass from its first corner, across the trail.
const MEADOW: [f32; 2] = [-150.0, -70.0];
const MEADOW_SIDE: f32 = 100.0;
/// The pond: a 200 m square of water around its center, in a basin.
const POND: [f32; 2] = [160.0, -150.0];
const POND_SIDE: f32 = 200.0;
/// A trail along x at this z, kept free of trees, for the walk.
const TRAIL_Z: f32 = -20.0;
const EYE_HEIGHT: f32 = 1.7;
const WALK_SPEED: f32 = 1.4;
const FOV_Y: f32 = 0.9;
/// World cells, 2^8 = 256 m.
const CELL_LOG2: u32 = 8;
const CELL: f32 = 256.0;
const BLADES_PER_CLUSTER: f32 = 8.0;
const PARTICLES_PER_CLUSTER: u32 = 16;
/// The swarm of particles: its center and radius over the meadow.
const SWARM_RADIUS: f32 = 40.0;

const TERRAIN: ProducerId = ProducerId(0);
const FAR_TERRAIN: ProducerId = ProducerId(1);
const GRASS: ProducerId = ProducerId(2);
const WATER: ProducerId = ProducerId(3);
const PARTICLES: ProducerId = ProducerId(4);

const SKY: &str = "dense vegetation sky";
const MOON: &str = "dense vegetation moon";
const LEAF: &str = "tex/leaf";

/// The sun: low in the south-west, ahead and left of the walk.
const SUN_ELEVATION: f32 = 0.2;
const SUN_AZIMUTH: f32 = 2.6;
const SUN_ILLUMINANCE: Vec3 = Vec3::new(60_000.0, 52_000.0, 42_000.0);
const SUN_RADIUS: f32 = 0.004_65;
const MOON_ILLUMINANCE: Vec3 = Vec3::new(0.3, 0.3, 0.3);
const MOON_RADIUS: f32 = 0.004_5;

/// A camera script, on a fixed 60 Hz clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Script {
    /// Along the trail at eye height, looking ahead, from the forest into the meadow.
    Walk,
    /// Standing on the trail, then a 180-degree turn in half a second.
    Turn,
    /// From 300 m up, circling the tile's center.
    Overview,
    /// Across the tile at 25 m, twice walking speed.
    Flythrough,
}

impl Script {
    pub const ALL: [Script; 4] = [
        Script::Walk,
        Script::Turn,
        Script::Overview,
        Script::Flythrough,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Script::Walk => "walk",
            Script::Turn => "turn",
            Script::Overview => "overview",
            Script::Flythrough => "flythrough",
        }
    }

    fn parse(s: &str) -> Self {
        Self::ALL
            .into_iter()
            .find(|c| c.name() == s)
            .unwrap_or_else(|| panic!("unknown script {s}"))
    }

    /// The eye and the point it looks at, `t` s into the script.
    fn camera(self, h: &Heights, t: f32) -> (Vec3, Vec3) {
        let ground = |x: f32, z: f32| h.at(x, z);
        match self {
            Script::Walk => {
                let x = -175.0 + WALK_SPEED * t;
                let eye = Vec3::new(x, ground(x, TRAIL_Z) + EYE_HEIGHT, TRAIL_Z);
                let yaw = 0.25 * (0.4 * t).sin();
                let ahead = Vec3::new(yaw.cos(), 0.0, yaw.sin()) * 12.0;
                let target = eye + ahead + Vec3::Y * (ground(x + 12.0, TRAIL_Z) - eye.y + 2.5);
                (eye, target)
            }
            Script::Turn => {
                let x = -100.0;
                let eye = Vec3::new(x, ground(x, TRAIL_Z) + EYE_HEIGHT, TRAIL_Z);
                let turn = ((t - 0.5) / 0.5).clamp(0.0, 1.0);
                let yaw = std::f32::consts::PI * turn;
                (eye, eye + Vec3::new(yaw.cos(), 0.1, yaw.sin()) * 10.0)
            }
            Script::Overview => {
                let a = 0.8 + 0.05 * t;
                let eye = Vec3::new(a.cos() * 400.0, 300.0 + ground(0.0, 0.0), a.sin() * 400.0);
                (eye, Vec3::new(0.0, ground(0.0, 0.0), 0.0))
            }
            Script::Flythrough => {
                let s = -420.0 + 2.0 * WALK_SPEED * t * std::f32::consts::FRAC_1_SQRT_2;
                let eye = Vec3::new(s, ground(s, s) + 25.0, s);
                (eye, eye + Vec3::new(30.0, -8.0, 30.0))
            }
        }
    }
}

/// Which parts the animation kernel moves; deeper parts always ride on their parents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parts {
    /// Every part flutters about its own base.
    All,
    /// Only the parts on the trunk (the body) flutter.
    Trunk,
}

/// What the scene holds and how it runs, from the command line.
#[derive(Clone, Debug)]
pub struct Settings {
    pub synthetic: bool,
    pub trees: u32,
    pub species: u32,
    pub grass_density: f32,
    pub grass_radius: f32,
    pub parts: Parts,
    pub poses_per_variant: u32,
    pub pose_stride: bool,
    pub part_animation: bool,
    pub rt_range: f32,
    pub animation_range: f32,
    /// Limb-darkened and textured directional lights.
    pub directional_lights: (u32, u32),
    pub sw_raster_px: f32,
    pub lod0_only: bool,
    pub world_km: f32,
    pub world_density: f32,
    pub particles: u32,
    pub stream_load: bool,
    pub extent: [u32; 2],
    pub scripts: Vec<Script>,
    /// Frames per script; `None` runs the window's script until it closes.
    pub script_frames: Option<u64>,
}

impl Settings {
    /// The defaults, then `rest` over them.
    pub fn parse(rest: &[String]) -> Self {
        let mut s = Self {
            synthetic: false,
            trees: 10_000,
            species: 4,
            grass_density: 500.0,
            grass_radius: 75.0,
            parts: Parts::All,
            poses_per_variant: 8,
            pose_stride: false,
            part_animation: true,
            rt_range: 400.0,
            animation_range: 50.0,
            directional_lights: (1, 1),
            sw_raster_px: RendererDesc::default().software_raster_pixels,
            lod0_only: false,
            world_km: 1.0,
            world_density: 400.0,
            particles: 320,
            stream_load: false,
            extent: [1920, 1080],
            scripts: vec![Script::Walk],
            script_frames: None,
        };
        let mut args = rest.iter();
        while let Some(a) = args.next() {
            let mut value = || {
                args.next()
                    .unwrap_or_else(|| panic!("{a} takes a value"))
                    .as_str()
            };
            let number = |v: &str| -> f32 { v.parse().unwrap_or_else(|_| panic!("{a} {v}")) };
            match a.as_str() {
                "--synthetic" => s.synthetic = true,
                "--trees" => s.trees = number(value()) as u32,
                "--species" => s.species = number(value()) as u32,
                "--grass-density" => s.grass_density = number(value()),
                "--grass-radius" => s.grass_radius = number(value()),
                "--parts" => {
                    s.parts = match value() {
                        "all" => Parts::All,
                        "trunk" => Parts::Trunk,
                        v => panic!("--parts all|trunk, not {v}"),
                    }
                }
                "--poses-per-variant" => s.poses_per_variant = (number(value()) as u32).max(1),
                "--pose-stride" => s.pose_stride = true,
                "--no-part-animation" => s.part_animation = false,
                "--rt-range" => s.rt_range = number(value()),
                "--animation-range" => s.animation_range = number(value()),
                "--directional-lights" => {
                    let v = value();
                    let (l, t) = v.split_once(',').expect("--directional-lights L,T");
                    s.directional_lights = (number(l) as u32, number(t) as u32);
                }
                "--sw-raster-px" => s.sw_raster_px = number(value()),
                "--lod0-only" => s.lod0_only = true,
                "--world-km" => s.world_km = number(value()).max(1.0),
                "--world-density" => s.world_density = number(value()),
                "--particles" => s.particles = number(value()) as u32,
                "--stream-load" => s.stream_load = true,
                "--extent" => {
                    let v = value();
                    let (w, h) = v.split_once('x').expect("--extent WxH");
                    s.extent = [number(w) as u32, number(h) as u32];
                }
                "--script" => s.scripts = vec![Script::parse(value())],
                "--script-frames" => s.script_frames = Some(number(value()) as u64),
                _ => panic!("unknown argument {a}"),
            }
        }
        assert!(
            s.species > 0 && s.poses_per_variant <= 256,
            "at least one species and at most 256 poses per variant"
        );
        s
    }
}

// --- Content ----------------------------------------------------------------------------

/// Variants of each species.
const VARIANTS: u32 = 3;

/// One stand-in species' shape: height, crown radius, branches and leaves.
struct Species {
    height: f32,
    crown: f32,
    branches: u32,
    leaves: u32,
    leaf_size: f32,
}

const SPECIES: [Species; 4] = [
    Species {
        height: 22.0,
        crown: 3.0,
        branches: 32,
        leaves: 1600,
        leaf_size: 0.45,
    },
    Species {
        height: 16.0,
        crown: 6.0,
        branches: 24,
        leaves: 1400,
        leaf_size: 0.6,
    },
    Species {
        height: 12.0,
        crown: 4.5,
        branches: 20,
        leaves: 1000,
        leaf_size: 0.55,
    },
    Species {
        height: 7.0,
        crown: 3.5,
        branches: 12,
        leaves: 700,
        leaf_size: 0.5,
    },
];

/// A tree model's asset name.
fn tree_name(species: u32, variant: u32) -> String {
    format!("tree {species} {variant}")
}

/// A small xorshift generator, so content and placement repeat.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

/// A stand-in tree: section 0 bark, section 1 leaves under the leaf mask. The trunk is the
/// body; part `b + 1` is branch `b`, pivoted where it leaves the trunk, and part
/// `branches + b + 1` its twig, pivoted partway along it. Leaves sit on branches and twigs.
fn tree(species: u32, variant: u32) -> orr_cook::MeshInput {
    let s = &SPECIES[species as usize];
    let mut rng = Rng::new(u64::from(species * 16 + variant + 1));
    let mut m = MeshBuilder::default();
    let trunk = s.height * 0.02;
    m.tube(
        0,
        Vec3::ZERO,
        Vec3::Y * s.height,
        trunk,
        trunk * 0.3,
        16,
        12,
    );
    let crown_base = s.height * 0.35;
    // Per branch: its part's pivot and tip, then its twig's.
    let mut limbs = Vec::new();
    for b in 0..s.branches {
        let t = b as f32 / s.branches as f32;
        let y = crown_base + (s.height - crown_base) * t * 0.95;
        let yaw = b as f32 * 2.399 + rng.range(-0.3, 0.3);
        let reach = s.crown * (1.0 - 0.7 * t) * rng.range(0.8, 1.1);
        let base = Vec3::new(0.0, y, 0.0);
        let dir = Vec3::new(yaw.cos(), rng.range(0.2, 0.6), yaw.sin()).normalize();
        let tip = base + dir * reach;
        m.part(b as u16 + 1, base);
        m.tube(0, base, tip, trunk * 0.35, trunk * 0.08, 8, 4);
        let fork = base.lerp(tip, 0.6);
        let side = Vec3::new(-dir.z, 0.3, dir.x).normalize() * if b % 2 == 0 { 1.0 } else { -1.0 };
        let twig = fork + (dir + side).normalize() * reach * 0.45;
        m.part((s.branches + b) as u16 + 1, fork);
        m.tube(0, fork, twig, trunk * 0.12, trunk * 0.04, 6, 2);
        limbs.push([(base, tip), (fork, twig)]);
    }
    for l in 0..s.leaves {
        let b = l % s.branches;
        let on_twig = (l / s.branches) % 2 == 1;
        let (base, tip) = limbs[b as usize][usize::from(on_twig)];
        let along = base.lerp(tip, rng.range(0.4, 1.0));
        let offset = Vec3::new(
            rng.range(-1.0, 1.0),
            rng.range(-0.6, 0.8),
            rng.range(-1.0, 1.0),
        ) * s.crown
            * 0.2;
        let normal = Vec3::new(
            rng.range(-1.0, 1.0),
            rng.range(-0.2, 1.0),
            rng.range(-1.0, 1.0),
        )
        .normalize_or(Vec3::Y);
        let part = if on_twig { s.branches + b } else { b };
        m.part(part as u16 + 1, base);
        m.leaf(1, along + offset, normal, s.leaf_size);
    }
    m.finish()
}

/// The pack every run cooks in process: the sky, the moon's albedo and, with `--synthetic`,
/// the stand-in trees under their leaf mask.
fn cook(settings: &Settings) -> Arc<[u8]> {
    let started = Instant::now();
    let sky = Sky::new("dense-vegetation", sun_direction(0));
    let dir = std::env::temp_dir().join(format!("orr-dense-vegetation-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let texture = |name: &str, path: PathBuf, usage: Usage| AssetSpec {
        name: name.into(),
        source: Source::Texture {
            path,
            settings: TextureSettings {
                usage,
                quality: Quality::Fast,
                ..TextureSettings::default()
            },
        },
    };
    // Maria on a bright disc.
    let moon = dir.join("moon.png");
    gray_png(&moon, 64, |u, v| {
        let mare = |cu: f32, cv: f32, r: f32| ((u - cu).powi(2) + (v - cv).powi(2)) < r * r;
        if mare(0.35, 0.4, 0.15) || mare(0.62, 0.55, 0.12) || mare(0.5, 0.75, 0.08) {
            90
        } else {
            190
        }
    });
    // Four leaves around the patch's center.
    let leaf = dir.join("leaf.png");
    gray_png(&leaf, 128, |u, v| {
        let (u, v) = (u - 0.5, v - 0.5);
        let a = v.atan2(u) * 2.0;
        let r = (u * u + v * v).sqrt();
        if r < 0.45 * a.cos().abs().sqrt() && r > 0.04 {
            255
        } else {
            0
        }
    });
    let mut assets = vec![
        sky.asset(SKY),
        texture(MOON, moon, Usage::Color),
        texture(LEAF, leaf, Usage::Linear1),
    ];
    if settings.synthetic {
        let bark = Material {
            base_color: [0.35, 0.27, 0.2],
            specular_roughness: 0.8,
            ..Material::default()
        };
        let mut leaves = Material {
            base_color: [0.18, 0.4, 0.12],
            alpha_mode: Material::MASK,
            alpha_cutoff: 0.5,
            double_sided: 1,
            ..Material::default()
        };
        leaves.textures[slot::OPACITY].texture = AssetId::from_name(LEAF);
        for s in 0..settings.species.min(SPECIES.len() as u32) {
            for v in 0..VARIANTS {
                assets.push(AssetSpec {
                    name: tree_name(s, v),
                    source: Source::Mesh {
                        mesh: tree(s, v),
                        materials: vec![bark, leaves],
                        settings: MeshSettings {
                            lod: LodMode::Discrete {
                                errors: vec![0.01, 0.05, 0.2, 0.8],
                            },
                            merge_level: Some(4),
                            ..MeshSettings::default()
                        },
                    },
                });
            }
        }
    }
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, assets)],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0);
    let _ = std::fs::remove_dir_all(&dir);
    log::info!(
        "cooked {:.1} MB in {:.1} s",
        pack.len() as f64 / 1e6,
        started.elapsed().as_secs_f64()
    );
    pack
}

/// The cooked archives: every `.orr` file in `$ORR_CONTENT/cooked`.
fn cooked_archives() -> Vec<PathBuf> {
    let root = std::env::var_os("ORR_CONTENT")
        .map(PathBuf::from)
        .expect("ORR_CONTENT names the content directory; run with --synthetic without it");
    let dir = root.join("cooked");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "orr"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .orr archives in {}", dir.display());
    files
}

/// A tree model and what its pose kernel needs: its part count and, per part, the joint it
/// tilts about.
struct TreeModel {
    model: ModelId,
    parts: u32,
    joints: Vec<sway::Joint>,
}

/// The parts of model `name` in `archive`, from its mesh header. Each part flutters about
/// its pivot, tilting about the horizontal axis across its limb: the line from its pivot to
/// its bounds' center. A part pivoted inside the body's bounds hangs on the body; any other
/// on the body-hung part whose bounds hold its pivot and whose limb passes nearest it.
fn tree_parts(archive: &Archive, name: &str) -> Option<(u32, Vec<sway::Joint>)> {
    let read = |id: AssetId| {
        let e = archive.find(id)?;
        let b = archive.blobs(e).first()?;
        let mut bytes = vec![0; b.raw_len as usize];
        archive.read(b, &mut bytes).ok()?;
        Some(bytes)
    };
    let model = ModelInfo::parse(&read(AssetId::from_name(name))?).ok()?;
    let mesh = MeshInfo::parse(&read(model.header.mesh)?).ok()?;
    struct Limb {
        pivot: Vec3,
        center: Vec3,
        lo: Vec3,
        hi: Vec3,
    }
    let limbs: Vec<Limb> = mesh
        .parts
        .iter()
        .map(|p| {
            let (lo, hi) = (Vec3::from(p.bounds.min), Vec3::from(p.bounds.max));
            Limb {
                pivot: Vec3::from(p.pivot),
                center: (lo + hi) * 0.5,
                lo,
                hi,
            }
        })
        .collect();
    let holds = |l: &Limb, at: Vec3| at.cmpge(l.lo).all() && at.cmple(l.hi).all();
    let on_body: Vec<bool> = limbs
        .iter()
        .enumerate()
        .map(|(i, l)| i > 0 && holds(&limbs[0], l.pivot))
        .collect();
    let distance = |l: &Limb, at: Vec3| {
        let d = l.center - l.pivot;
        let t = ((at - l.pivot).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0);
        (l.pivot + d * t).distance(at)
    };
    let mut joints = vec![sway::Joint {
        pivot: [0.0; 3],
        parent: 0,
        axis: [0.0; 3],
        _pad0: 0.0,
    }];
    for (i, l) in limbs.iter().enumerate().skip(1) {
        let out = (l.center - l.pivot) * Vec3::new(1.0, 0.0, 1.0);
        let axis = if out.length() < 1e-3 {
            Vec3::ZERO
        } else {
            Vec3::Y.cross(out).normalize()
        };
        let parent = if on_body[i] {
            0
        } else {
            (1..limbs.len())
                .filter(|&k| on_body[k] && holds(&limbs[k], l.pivot))
                .min_by(|&a, &b| {
                    distance(&limbs[a], l.pivot).total_cmp(&distance(&limbs[b], l.pivot))
                })
                .unwrap_or(0) as u32
        };
        joints.push(sway::Joint {
            pivot: l.pivot.to_array(),
            parent,
            axis: axis.to_array(),
            _pad0: 0.0,
        });
    }
    Some((mesh.parts.len() as u32, joints))
}

// --- Terrain ------------------------------------------------------------------------------

/// Ground height at (x, z): rolling hills with a basin under the pond.
fn ground(x: f32, z: f32) -> f32 {
    let (dx, dz) = (x - POND[0], z - POND[1]);
    4.0 * (0.011 * x).sin() * (0.009 * z).cos()
        + 1.5 * (0.031 * x + 0.02 * z).sin()
        + 0.4 * (0.13 * x).sin() * (0.11 * z).cos()
        - 3.0 * (-(dx * dx + dz * dz) / (2.0 * 70.0 * 70.0)).exp()
}

/// The water's level: over the basin's floor.
fn water_level() -> f32 {
    ground(POND[0], POND[1]) + 1.8
}

fn inside(p: [f32; 2], corner: [f32; 2], side: f32, margin: f32) -> bool {
    (0..2).all(|i| p[i] > corner[i] - margin && p[i] < corner[i] + side + margin)
}

/// Whether a tree may stand at (x, z): not on the trail, in the meadow or in the pond.
fn plantable(x: f32, z: f32) -> bool {
    let pond = [POND[0] - POND_SIDE / 2.0, POND[1] - POND_SIDE / 2.0];
    (z - TRAIL_Z).abs() > 2.5
        && !inside([x, z], MEADOW, MEADOW_SIDE, 3.0)
        && !inside([x, z], pond, POND_SIDE, -20.0)
}

// --- Directional lights --------------------------------------------------------------------

/// Unit direction to limb-darkened light `i`: the sun, then others spread around the sky.
fn sun_direction(i: u32) -> Vec3 {
    let az = SUN_AZIMUTH + i as f32 * 2.1;
    let el = SUN_ELEVATION + i as f32 * 0.15;
    Vec3::new(el.cos() * az.cos(), el.sin(), el.cos() * az.sin())
}

/// Unit direction to textured light `i`, high in the sky across from the sun.
fn moon_direction(i: u32) -> Vec3 {
    let az = SUN_AZIMUTH + std::f32::consts::PI + i as f32 * 0.7;
    let el = 0.6 + i as f32 * 0.12;
    Vec3::new(el.cos() * az.cos(), el.sin(), el.cos() * az.sin())
}

// --- The scene ---------------------------------------------------------------------------

/// The trees of one model in one cell, as transforms from the cell's anchor.
struct Stand {
    model: usize,
    cell: (i32, i32),
    at: Vec<Affine3>,
}

/// A tree model's GPU-written pose group and what its pose kernel reads.
struct Poses {
    group: TransformGroupId,
    blocks: u32,
    parts: u32,
    joints: Allocation<sway::Joint>,
}

/// The dense vegetation test scene loaded into a renderer.
struct DenseVegetation {
    scene: SceneId,
    history: ViewHistoryId,
    /// The frame id the scene was created in.
    frame: u64,
    tree_count: u32,
    settings: Settings,
    heights: Heights,
    far: Option<(Heights, Terrain)>,
    land: Terrain,
    grass: Grass,
    water: Slab,
    poses: Vec<Poses>,
    kernels: Kernels,
    glow: MeshPipeline,
    stream_pack: Option<PackSource>,
    streaming: Option<MountId>,
    _environment: Allocation<EnvironmentMapGpu>,
}

struct Kernels {
    terrain: PipelineId,
    grass: PipelineId,
    water: PipelineId,
    fireflies: PipelineId,
    sway: PipelineId,
}

fn cell_of(x: f32, z: f32) -> (i32, i32) {
    ((x / CELL).floor() as i32, (z / CELL).floor() as i32)
}

fn anchor(cell: (i32, i32)) -> Vec3 {
    Vec3::new(
        (cell.0 as f32 + 0.5) * CELL,
        0.0,
        (cell.1 as f32 + 0.5) * CELL,
    )
}

/// The forest: trees on a jittered grid over the tile at `trees` per km^2, then past it to
/// the world's edge at `world_density`, grouped by cell and model.
fn plant(settings: &Settings, models: usize, heights: &dyn Fn(f32, f32) -> f32) -> Vec<Stand> {
    let mut rng = Rng::new(5);
    let mut stands: Vec<Stand> = Vec::new();
    let mut place = |x: f32, z: f32, rng: &mut Rng| {
        if !plantable(x, z) {
            return;
        }
        let model = ((rng.next() * models as f32) as usize).min(models - 1);
        let cell = cell_of(x, z);
        let turn = Mat3::from_rotation_y(rng.range(0.0, std::f32::consts::TAU));
        let at = Vec3::new(x, heights(x, z) - 0.1, z) - anchor(cell);
        let t = affine(turn * rng.range(0.8, 1.2), at);
        match stands
            .iter_mut()
            .find(|s| s.cell == cell && s.model == model)
        {
            Some(s) => s.at.push(t),
            None => stands.push(Stand {
                model,
                cell,
                at: vec![t],
            }),
        }
    };
    let grid = |count: f32, side: f32| (side / count.sqrt().max(1.0)).max(1.0);
    let spacing = grid(settings.trees as f32, TILE);
    let n = (TILE / spacing) as u32;
    for j in 0..n {
        for i in 0..n {
            let x = -TILE / 2.0 + (i as f32 + rng.range(0.1, 0.9)) * spacing;
            let z = -TILE / 2.0 + (j as f32 + rng.range(0.1, 0.9)) * spacing;
            place(x, z, &mut rng);
        }
    }
    if settings.world_km > 1.0 && settings.world_density > 0.0 {
        let side = settings.world_km * 1000.0;
        let spacing = 1000.0 / settings.world_density.sqrt();
        let n = (side / spacing) as u32;
        for j in 0..n {
            for i in 0..n {
                let x = -side / 2.0 + (i as f32 + rng.range(0.1, 0.9)) * spacing;
                let z = -side / 2.0 + (j as f32 + rng.range(0.1, 0.9)) * spacing;
                if x.abs() > TILE / 2.0 || z.abs() > TILE / 2.0 {
                    place(x, z, &mut rng);
                }
            }
        }
    }
    stands.sort_by_key(|s| (s.cell, s.model));
    stands
}

/// Mounts `source`, rendering empty frames from `frame` until it has mounted; returns the
/// mount and the last frame id used.
fn mount(renderer: &mut Renderer, source: PackSource, mut frame: u64) -> (MountId, u64) {
    let m = renderer.mount(source);
    let start = Instant::now();
    while matches!(renderer.mount_state(m), LoadState::Pending) {
        frame += 1;
        let f = renderer
            .begin_frame(FrameDesc {
                time: fixed_time(frame),
                scenes: &[],
                display: None,
            })
            .expect("begin frame");
        f.end().expect("end frame");
        assert!(start.elapsed() < Duration::from_secs(600), "mount pending");
        std::thread::sleep(Duration::from_millis(1));
    }
    if let LoadState::Failed(e) = renderer.mount_state(m) {
        panic!("mount failed: {e}");
    }
    (m, frame)
}

/// The frame id the scene is created in, after its packs have mounted, so light sampling's
/// sequence repeats across runs.
const FIRST_FRAME: u64 = 1000;

/// The renderer's settings for `settings`.
fn renderer_desc(settings: &Settings) -> RendererDesc {
    RendererDesc {
        cell_log2: CELL_LOG2,
        software_raster_pixels: settings.sw_raster_px,
        raster_error_pixels: if settings.lod0_only {
            0.0
        } else {
            RendererDesc::default().raster_error_pixels
        },
        ..RendererDesc::default()
    }
}

impl DenseVegetation {
    /// Cooks and mounts the content, then creates the scene and sets its entries in one
    /// frame.
    fn load(renderer: &mut Renderer, gpu: &Device, settings: &Settings) -> Self {
        let extras = cook(settings);
        let (extras_mount, mut frame) = mount(renderer, PackSource::Memory(Arc::clone(&extras)), 0);
        // Tree archives: the extras pack itself, or every cooked archive.
        let mut archives: Vec<(MountId, Archive, PackSource)> = Vec::new();
        if settings.synthetic {
            let archive = Archive::from_memory(Arc::clone(&extras)).expect("archive");
            archives.push((extras_mount, archive, PackSource::Memory(extras)));
        } else {
            for path in cooked_archives() {
                let (m, f) = mount(renderer, PackSource::File(path.clone()), frame);
                frame = f;
                let archive = Archive::open(&path).expect("archive");
                archives.push((m, archive, PackSource::File(path)));
            }
        }
        let mut trees: Vec<TreeModel> = Vec::new();
        for s in 0..settings.species {
            for v in 0..VARIANTS {
                let name = tree_name(s, v);
                let found = archives.iter().find_map(|(m, a, _)| {
                    let (parts, joints) = tree_parts(a, &name)?;
                    let model = renderer.model(*m, AssetId::from_name(&name)).ok()?;
                    Some(TreeModel {
                        model,
                        parts,
                        joints,
                    })
                });
                match found {
                    Some(t) => trees.push(t),
                    None => log::warn!("no model {name}"),
                }
            }
        }
        assert!(!trees.is_empty(), "no tree models");
        let stream_pack = archives.first().map(|(_, _, s)| s.clone());
        while frame < FIRST_FRAME {
            frame += 1;
            let f = renderer
                .begin_frame(FrameDesc {
                    time: fixed_time(frame),
                    scenes: &[],
                    display: None,
                })
                .expect("begin frame");
            f.end().expect("end frame");
        }
        if frame > FIRST_FRAME {
            log::warn!("mounting took {frame} frames; this run will not repeat others exactly");
        }

        let heights = Heights::new(
            gpu,
            (TILE / TILE_SPACING) as u32 + 1,
            TILE_SPACING,
            [-TILE / 2.0; 2],
            ground,
        )
        .expect("heights");
        let land = Terrain {
            patches: (TILE / TILE_PATCH) as u32,
            patch_size: TILE_PATCH,
            corner: [-TILE / 2.0; 2],
            hole: None,
        };
        let far = (settings.world_km > 1.0).then(|| {
            // A side of whole patch pairs keeps the patch grid on the tile's edges.
            let side = (settings.world_km * 1000.0 / (2.0 * FAR_PATCH)).ceil() * 2.0 * FAR_PATCH;
            let h = Heights::new(
                gpu,
                (side / FAR_SPACING) as u32 + 1,
                FAR_SPACING,
                [-side / 2.0; 2],
                ground,
            )
            .expect("far heights");
            let t = Terrain {
                patches: (side / FAR_PATCH) as u32,
                patch_size: FAR_PATCH,
                corner: [-side / 2.0; 2],
                hole: Some(([-TILE / 2.0; 2], TILE)),
            };
            (h, t)
        });
        let cell = (BLADES_PER_CLUSTER / settings.grass_density.max(1e-3)).sqrt();
        let cells = (MEADOW_SIDE / cell).ceil() as u32;
        let reach = (std::f32::consts::PI * settings.grass_radius.powi(2) / (cell * cell)) as u64;
        let grass_field = Grass {
            cells,
            cell_size: MEADOW_SIDE / cells as f32,
            corner: MEADOW,
            radius: settings.grass_radius,
            capacity: reach.min(u64::from(cells) * u64::from(cells)).max(1) as u32,
        };
        let water = Slab {
            patches: (POND_SIDE / TILE_PATCH) as u32,
            patch_size: TILE_PATCH,
            corner: [POND[0] - POND_SIDE / 2.0, POND[1] - POND_SIDE / 2.0],
            level: water_level(),
            amplitude: 0.04,
        };

        let stands = plant(settings, trees.len(), &|x, z| {
            if x.abs() <= TILE / 2.0 && z.abs() <= TILE / 2.0 {
                heights.at(x, z)
            } else {
                far.as_ref()
                    .map_or_else(|| ground(x, z), |(h, _)| h.at(x, z))
            }
        });
        let tree_count: u32 = stands.iter().map(|s| s.at.len() as u32).sum();
        let mut per_model = vec![0u32; trees.len()];
        for s in &stands {
            per_model[s.model] += s.at.len() as u32;
        }
        let max_parts = trees.iter().map(|t| t.parts).max().unwrap_or(1);
        // Copies inside the animation range (with its hysteresis and a crown's reach) around
        // a focus, half again for room; a copy's run holds its part count rounded up to a
        // power of two.
        let density = settings.trees as f32 / (TILE * TILE);
        let reach = settings.animation_range * 1.15 + 10.0;
        let near = (density * std::f32::consts::PI * reach * reach * 1.5).ceil() as u32;
        let (lights, textured) = settings.directional_lights;
        let groups = stands.len() as u32 + trees.len() as u32 + 4;
        let scene = renderer
            .create_scene(
                &SceneDesc {
                    instance_capacity: tree_count + 64,
                    group_capacity: groups,
                    light_capacity: (lights + textured).max(1),
                    producer_capacity: 8,
                    part_capacity: (near * max_parts.next_power_of_two()).max(4096),
                    ..SceneDesc::default()
                },
                &SceneLightingSettings::default(),
            )
            .expect("scene");
        let history = renderer.create_view_history().expect("view history");
        let kernels = Kernels {
            terrain: compute(renderer, &terrain::MODULE),
            grass: compute(renderer, &grass::MODULE),
            water: compute(renderer, &slab::MODULE),
            fireflies: compute(renderer, &fireflies::MODULE),
            sway: compute(renderer, &sway::MODULE),
        };
        let glow = gpu
            .create_mesh_pipeline(
                &MeshPipelineDesc {
                    task: None,
                    mesh: ShaderStage::new(&glow::MODULE, "mesh", SpecData::default())
                        .expect("glow mesh"),
                    pixel: Some(
                        ShaderStage::new(&glow::MODULE, "pixel", SpecData::default())
                            .expect("glow pixel"),
                    ),
                    colors: &[ColorTarget {
                        format: Format::Rgba16Float,
                        blend: Some(Blend::PREMULTIPLIED),
                        write: ColorMask::ALL,
                    }],
                    depth: Some(DepthTarget {
                        format: Format::D32Float,
                        compare: CompareOp::GreaterOrEqual,
                        write: false,
                    }),
                    raster: RasterState::default(),
                    samples: 1,
                },
                "dense vegetation particles",
            )
            .expect("particle pipeline");
        let environment = sky::environment(renderer, gpu, extras_mount, SKY);
        let moon = renderer
            .texture(extras_mount, AssetId::from_name(MOON))
            .expect("moon texture");

        frame += 1;
        let listed = [scene];
        let mut f = renderer
            .begin_frame(FrameDesc {
                time: fixed_time(frame),
                scenes: &listed,
                display: None,
            })
            .expect("begin frame");
        let material =
            |m: Material| MaterialGpu::new(&m, &[BoundTexture::NONE; 12], SamplerIndex(0));
        let (soil, blades, pond, glowing) = {
            let mut mats = f.materials();
            let mut create = |m: Material| mats.create(&material(m)).expect("material");
            (
                create(Material {
                    base_color: [0.3, 0.26, 0.18],
                    specular_roughness: 0.9,
                    ..Material::default()
                }),
                create(Material {
                    base_color: [0.25, 0.45, 0.15],
                    specular_roughness: 0.6,
                    double_sided: 1,
                    ..Material::default()
                }),
                create(Material {
                    base_color: [1.0; 3],
                    specular_roughness: 0.05,
                    specular_ior: 1.333,
                    transmission_weight: 1.0,
                    ..Material::default()
                }),
                create(Material {
                    base_color: [0.1; 3],
                    emission_color: [1.0, 0.8, 0.35],
                    emission_luminance: 4000.0,
                    ..Material::default()
                }),
            )
        };
        let mut poses = Vec::new();
        {
            let mut w = f.scene(scene).expect("scene writer");
            let world = w
                .create_transform_group(&TransformGroupDesc {
                    count: 1,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::ZERO,
                })
                .expect("transform group");
            w.transforms(world, 0, 1).expect("transforms")[0] = Affine3::IDENTITY;

            let animated = settings.part_animation;
            for (i, t) in trees.iter().enumerate() {
                let blocks = if settings.pose_stride {
                    per_model[i]
                } else {
                    settings.poses_per_variant
                };
                let group = if animated && t.parts > 1 {
                    w.create_transform_group(&TransformGroupDesc {
                        count: (blocks * t.parts).max(1),
                        writer: GroupWriter::Gpu,
                        anchor: DVec3::ZERO,
                    })
                    .expect("pose group")
                } else {
                    TransformGroupId(0)
                };
                let mut joints = gpu
                    .gpu_malloc::<sway::Joint>(MemoryKind::Upload, t.joints.len() as u64, "joints")
                    .expect("joints");
                joints
                    .as_mut_slice()
                    .expect("joints")
                    .copy_from_slice(&t.joints);
                poses.push(Poses {
                    group,
                    blocks,
                    parts: t.parts,
                    joints,
                });
            }
            let mut first_copy = vec![0u32; trees.len()];
            let mut rng = Rng::new(11);
            for s in &stands {
                let n = s.at.len() as u32;
                let group = w
                    .create_transform_group(&TransformGroupDesc {
                        count: n,
                        writer: GroupWriter::Cpu,
                        anchor: anchor(s.cell).as_dvec3(),
                    })
                    .expect("transform group");
                w.transforms(group, 0, n)
                    .expect("transforms")
                    .copy_from_slice(&s.at);
                let p = &poses[s.model];
                let pose = (p.group != TransformGroupId(0)).then(|| {
                    if settings.pose_stride {
                        Pose {
                            group: p.group,
                            index: first_copy[s.model],
                            variants: 1,
                            stride: 1,
                        }
                    } else {
                        Pose {
                            group: p.group,
                            index: 0,
                            variants: settings.poses_per_variant,
                            stride: 0,
                        }
                    }
                });
                let range = w
                    .add_instances(&InstancesDesc {
                        model: trees[s.model].model,
                        copies: n,
                        transforms: group,
                        first: 0,
                        pose,
                        material_set: None,
                        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                        rt_range: settings.rt_range,
                        animation_range: if pose.is_some() {
                            settings.animation_range
                        } else {
                            0.0
                        },
                        user_tag: 0,
                    })
                    .expect("trees");
                if pose.is_some() && !settings.pose_stride && settings.poses_per_variant > 1 {
                    let variants: Vec<CopyVariant> = (0..n)
                        .map(|_| CopyVariant {
                            pose: (rng.next() * settings.poses_per_variant as f32) as u8,
                            ..CopyVariant::default()
                        })
                        .collect();
                    w.copy_variants(range, 0, &variants).expect("variants");
                }
                first_copy[s.model] += n;
            }

            let patch = patch_triangles();
            let patch_topology = [ClusterTopology {
                vertex_count: 81,
                triangles: &patch,
            }];
            let (tall, short) = (blade_triangles(true), blade_triangles(false));
            let blade_topologies = [
                ClusterTopology {
                    vertex_count: 40,
                    triangles: &tall,
                },
                ClusterTopology {
                    vertex_count: 24,
                    triangles: &short,
                },
            ];
            let octahedra = octahedra_triangles();
            let particle_topology = [ClusterTopology {
                vertex_count: 6 * PARTICLES_PER_CLUSTER,
                triangles: &octahedra,
            }];
            let (soil, blades, pond, glowing) = ([soil], [blades], [pond], [glowing]);
            w.set_producer(
                TERRAIN,
                Some(&land.desc(&patch_topology, &soil, (world, 0))),
            )
            .expect("terrain");
            if let Some((_, t)) = far.as_ref() {
                w.set_producer(
                    FAR_TERRAIN,
                    Some(&t.desc(&patch_topology, &soil, (world, 0))),
                )
                .expect("far terrain");
            }
            w.set_producer(
                GRASS,
                Some(&grass_field.desc(&blade_topologies, &blades, (world, 0))),
            )
            .expect("grass");
            w.set_producer(WATER, Some(&water.desc(&patch_topology, &pond, (world, 0))))
                .expect("pond");
            let clusters = settings.particles.div_ceil(PARTICLES_PER_CLUSTER);
            if clusters > 0 {
                w.set_producer(
                    PARTICLES,
                    Some(&ProducerDesc {
                        max_clusters: clusters,
                        max_vertices: clusters * 6 * PARTICLES_PER_CLUSTER,
                        update: ProducerUpdate::EveryFrame,
                        topologies: &particle_topology,
                        materials: &glowing,
                        transform: (world, 0),
                        opacity: Opacity::Particle,
                        previous_positions: true,
                        ..ProducerDesc::default()
                    }),
                )
                .expect("particles");
            }

            let sun = LightId(0);
            for i in 0..lights {
                w.set_light(
                    LightId(i),
                    Some(&Light::Directional(DirectionalLight {
                        direction_to_light: sun_direction(i),
                        illuminance: SUN_ILLUMINANCE / (1 + i) as f32,
                        angular_radius: SUN_RADIUS,
                        disc: DiscProfile::LimbDarkened,
                    })),
                )
                .expect("sun");
            }
            for i in 0..textured {
                w.set_light(
                    LightId(lights + i),
                    Some(&Light::Directional(DirectionalLight {
                        direction_to_light: moon_direction(i),
                        illuminance: MOON_ILLUMINANCE,
                        angular_radius: MOON_RADIUS,
                        disc: DiscProfile::Textured {
                            albedo: moon,
                            up: Vec3::Y,
                            lit_by: [
                                if lights > 0 { sun } else { LightId::NONE },
                                LightId::NONE,
                                LightId::NONE,
                                LightId::NONE,
                            ],
                        },
                    })),
                )
                .expect("moon");
            }
            w.set_environment(environment.gpu.address());
        }
        let land_kernel = kernels.terrain;
        f.update_scenes(&mut |c| {
            land.record(c, TERRAIN, land_kernel, &heights);
            if let Some((h, t)) = far.as_ref() {
                t.record(c, FAR_TERRAIN, land_kernel, h);
            }
        })
        .expect("update scenes");
        f.end().expect("end frame");
        log::info!(
            "{tree_count} trees of {} models in {} stands, {} grass clusters at most",
            trees.len(),
            stands.len(),
            grass_field.capacity
        );

        Self {
            scene,
            history,
            frame,
            tree_count,
            settings: settings.clone(),
            heights,
            far,
            land,
            grass: grass_field,
            water,
            poses,
            kernels,
            glow,
            stream_pack: settings.stream_load.then_some(stream_pack).flatten(),
            streaming: None,
            _environment: environment,
        }
    }

    /// Records the scene's kernels for a camera at `eye` at times `t` and `previous`.
    fn record(&self, c: &mut SceneCompute<'_, '_>, eye: Vec3, t: f32, previous: f32) {
        self.grass.record(
            c,
            GRASS,
            self.kernels.grass,
            &self.heights,
            [eye.x, eye.z],
            t,
            previous,
        );
        self.water.record(c, WATER, self.kernels.water, t);
        let clusters = self.settings.particles.div_ceil(PARTICLES_PER_CLUSTER);
        if clusters > 0 {
            let center = [MEADOW[0] + MEADOW_SIDE / 2.0, MEADOW[1] + MEADOW_SIDE / 2.0];
            let root = c
                .put_root(&fireflies::Root {
                    scene: c.root,
                    heights: self.heights.samples(),
                    side: self.heights.side,
                    spacing: self.heights.spacing,
                    origin_x: self.heights.origin[0],
                    origin_z: self.heights.origin[1],
                    producer: PARTICLES.0,
                    clusters,
                    center_x: center[0],
                    center_z: center[1],
                    radius: SWARM_RADIUS,
                    size: 0.05,
                    time: t,
                    previous_time: previous,
                })
                .expect("particle root");
            c.cmd.dispatch(
                c.pipeline(self.kernels.fireflies),
                root,
                [clusters.div_ceil(64), 1, 1],
            );
        }
        for p in self.poses.iter().filter(|p| p.group != TransformGroupId(0)) {
            let root = c
                .put_root(&sway::Root {
                    scene: c.root,
                    joints: p.joints.gpu,
                    group: p.group.0,
                    blocks: p.blocks,
                    parts: p.parts,
                    all_parts: u32::from(self.settings.parts == Parts::All),
                    time: t,
                    phase_step: if self.settings.pose_stride {
                        2.399
                    } else {
                        std::f32::consts::TAU / p.blocks as f32
                    },
                    flutter: 0.15,
                    _pad0: 0,
                })
                .expect("sway root");
            c.cmd.dispatch(
                c.pipeline(self.kernels.sway),
                root,
                [(p.blocks * p.parts).div_ceil(64), 1, 1],
            );
        }
    }

    /// The terrain entries' clusters: the tile's patches and the far terrain's outside it.
    fn terrain_patches(&self) -> u32 {
        let hole = (TILE / FAR_PATCH) as u32;
        self.land.patches * self.land.patches
            + self
                .far
                .as_ref()
                .map_or(0, |(_, t)| t.patches * t.patches - hole * hole)
    }
}

fn compute(renderer: &mut Renderer, module: &'static orr::orr_gpu::ShaderModule) -> PipelineId {
    renderer
        .compute_pipeline(module, "main", SpecData::default())
        .expect("pipeline")
}

/// The triangles of a particle cluster: 16 octahedra of six vertices, eight faces each.
fn octahedra_triangles() -> Vec<[u8; 3]> {
    // Vertices +x, -x, +y, -y, +z, -z.
    let faces = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    (0..PARTICLES_PER_CLUSTER)
        .flat_map(|p| faces.iter().map(move |f| f.map(|k: u32| (p * 6 + k) as u8)))
        .collect()
}

// --- Running --------------------------------------------------------------------------------

/// What a run collected: per script, the figures of each frame it rendered.
pub struct Run {
    pub capture: Option<Capture>,
    pub scripts: Vec<(Script, Vec<Sample>)>,
    pub tree_count: u32,
    pub terrain_patches: u32,
    /// Cook, mounts and scene setup, ms.
    pub load_ms: f64,
    pub memory: orr::orr_gpu::MemoryReport,
}

/// One completed frame's figures.
#[derive(Clone)]
pub struct Sample {
    pub stats: FrameStats,
    pub lighting: Option<LightingStats>,
    /// `RasterView::counts`.
    pub counts: [u32; COUNTS],
}

/// Words of `RasterView::counts` read back.
pub const COUNTS: usize = 16;
/// Frames of each script whose figures are left out, while histories and exposure settle.
const WARM_UP: u64 = 20;

/// Renders `settings.scripts` one after another, into a window unless `options.headless`;
/// headless runs capture the last frame of the last script.
pub fn run(options: &Options, settings: &Settings) -> Run {
    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new("orr dense vegetation", 1280, 720));
    let desc = renderer_desc(settings);
    let mut display = window.as_ref().map(|w| {
        Display::new(
            gpu,
            w.window_handle(),
            w.display_handle(),
            &DisplayRequest::default(),
            None,
            desc.frames_in_flight,
        )
        .expect("display")
    });
    let mut renderer = Renderer::new(gpu, desc).expect("renderer");
    let workers = Workers::new(renderer.loading(), 4);
    let started = Instant::now();
    let mut scene = DenseVegetation::load(&mut renderer, gpu, settings);
    let load_ms = started.elapsed().as_secs_f64() * 1e3;
    let listed = [scene.scene];

    let headless_target: Option<Texture> = options.headless.then(|| {
        let [w, h] = settings.extent;
        gpu.create_texture(
            &TextureDesc::d2(
                Format::Rgba8Unorm,
                w,
                h,
                TextureUsage::STORAGE | TextureUsage::SAMPLED,
            ),
            TextureMemory::Dedicated,
            "dense vegetation target",
        )
        .expect("headless target")
    });
    let slots = u64::from(desc.frames_in_flight) + 1;
    let readback = gpu
        .gpu_malloc::<[u32; COUNTS]>(MemoryKind::Readback, slots, "raster counts")
        .expect("counts readback");
    let mut glow_roots = gpu
        .gpu_malloc::<glow::Root>(MemoryKind::Upload, slots, "particle draw roots")
        .expect("particle roots");

    let frames_per_script = options.frames.or(settings.script_frames);
    let mut collected: Vec<(Script, Vec<Sample>)> = Vec::new();
    let mut frame_ids: Vec<(u64, usize)> = Vec::new();
    let mut last: Option<(u64, TimelinePoint)> = None;
    let mut closed = false;
    let base = fixed_time(scene.frame + 1).absolute;
    let mut clock = 0u64;
    let wall = Instant::now();
    let mut previous_t = 0.0f64;
    for (si, &script) in settings.scripts.iter().enumerate() {
        collected.push((script, Vec::new()));
        let final_script = si + 1 == settings.scripts.len();
        for i in 0.. {
            if frames_per_script.is_some_and(|n| i >= n) || closed {
                break;
            }
            if let Some(w) = window.as_mut()
                && !w.pump()
            {
                closed = true;
                break;
            }
            let final_frame = final_script && frames_per_script.is_some_and(|n| i + 1 == n);
            take_sample(
                &renderer,
                &scene,
                &readback,
                slots,
                &mut frame_ids,
                &mut collected,
            );
            let t = if options.headless || frames_per_script.is_some() {
                i as f64 / 60.0
            } else {
                wall.elapsed().as_secs_f64()
            };
            clock += 1;
            let time = FrameTime {
                absolute: base + clock as f64 / 60.0,
                delta: if i == 0 {
                    1.0 / 60.0
                } else {
                    (t - previous_t) as f32
                },
            };
            let previous = if i == 0 { t } else { previous_t };
            previous_t = t;
            let (eye, target) = script.camera(&scene.heights, t as f32);

            if let Some(source) = scene.stream_pack.as_ref()
                && clock.is_multiple_of(20)
            {
                match scene.streaming.take() {
                    Some(m) => renderer.unmount(m),
                    None => scene.streaming = Some(renderer.mount(source.clone())),
                }
            }

            let mut f = renderer
                .begin_frame(FrameDesc {
                    time,
                    scenes: &listed,
                    display: display.as_mut(),
                })
                .expect("begin frame");
            let id = f.id();
            let (extent, view_target) = match (&headless_target, f.display()) {
                (Some(t), _) => (settings.extent, ViewTarget::Texture(t)),
                (None, Some((_, extent))) => (extent, ViewTarget::Display),
                (None, None) => {
                    last = Some((id, f.end().expect("end frame")));
                    continue;
                }
            };
            f.scene(scene.scene)
                .expect("scene writer")
                .set_focus(eye.as_dvec3(), extent[1] as f32 / FOV_Y);
            f.update_scenes(&mut |c| scene.record(c, eye, t as f32, previous as f32))
                .expect("update scenes");
            if let Some(tex) = headless_target.as_ref().filter(|_| clock == 1) {
                f.list().expect("frame list").discard(tex);
            }
            let slot = id % slots;
            let glow = &scene.glow;
            let roots = &mut glow_roots;
            let mut particles = |pass: &mut TransparencyPass<'_, '_>| {
                roots.as_mut_slice().expect("particle roots")[slot as usize] = glow::Root {
                    scene: pass.scene,
                    view: pass.view,
                    raster: pass.raster,
                    _pad0: 0,
                    radiance: [4000.0, 3200.0, 1400.0, 0.0],
                };
                pass.cmd.begin_rendering(&Rendering {
                    extent: pass.extent,
                    colors: &[ColorAttachment {
                        texture: pass.overlay,
                        load: ColorLoad::Keep,
                        store: true,
                    }],
                    depth: Some(DepthAttachment {
                        texture: pass.depth,
                        load: DepthLoad::Keep,
                        store: true,
                    }),
                });
                pass.cmd.draw_mesh_tasks_indirect_count(
                    glow,
                    roots.gpu.offset(slot),
                    pass.particle_args,
                    pass.particle_draws,
                    1,
                );
                pass.cmd.end_rendering();
            };
            let mut hooks = ViewHooks {
                transparency: Some(&mut particles),
                ..ViewHooks::default()
            };
            let output = ViewOutput {
                screenshot: (options.headless && final_frame)
                    .then_some(ScreenshotRequest::Display { include_ui: false }),
                ..ViewOutput::default()
            };
            let rendered = f
                .render_view(
                    &ViewDesc {
                        scene: scene.scene,
                        history: scene.history,
                        position: eye.as_dvec3(),
                        rotation: view::look_at_quat(eye, target, Vec3::Y).inverse(),
                        projection: Projection {
                            fov_y: FOV_Y,
                            near: 0.1,
                        },
                        lens: Lens::default(),
                        render_extent: UVec2::from(extent),
                        jitter: Vec2::ZERO,
                        reset_history: i == 0,
                        mip_bias: 0.0,
                        target: view_target,
                    },
                    &ViewLighting::default(),
                    &output,
                    &mut hooks,
                )
                .expect("render view");
            let list = f.list().expect("frame list");
            list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
            list.copy(
                readback.gpu.offset(slot).cast(),
                rendered.raster.counts.cast(),
                (COUNTS * 4) as u64,
            );
            list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
            frame_ids.push((id, si));
            last = Some((id, f.end().expect("end frame")));
        }
    }

    if let Some(d) = display.as_ref() {
        d.wait_idle();
    }
    if let Some((_, point)) = last {
        gpu.queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(30))
            .expect("wait");
    }
    // Empty frames take in the figures of the frames still in flight.
    for _ in 0..slots {
        take_sample(
            &renderer,
            &scene,
            &readback,
            slots,
            &mut frame_ids,
            &mut collected,
        );
        clock += 1;
        let f = renderer
            .begin_frame(FrameDesc {
                time: FrameTime {
                    absolute: base + clock as f64 / 60.0,
                    delta: 1.0 / 60.0,
                },
                scenes: &[],
                display: None,
            })
            .expect("begin frame");
        let point = f.end().expect("end frame");
        gpu.queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(30))
            .expect("wait");
    }
    take_sample(
        &renderer,
        &scene,
        &readback,
        slots,
        &mut frame_ids,
        &mut collected,
    );
    let capture = last
        .filter(|_| options.headless)
        .and_then(|(id, _)| renderer.screenshots(scene.history, id).pop())
        .map(|shot| Capture::from_screenshot(&shot));
    let memory = gpu.memory_report();
    let run = Run {
        capture,
        scripts: collected,
        tree_count: scene.tree_count,
        terrain_patches: scene.terrain_patches(),
        load_ms,
        memory,
    };
    if let Some(m) = scene.streaming.take() {
        renderer.unmount(m);
    }
    drop(glow_roots);
    drop(readback);
    drop(headless_target);
    drop(scene);
    drop(display);
    drop(workers);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    run
}

/// Files the figures of the frame that last completed under its script, past the script's
/// warm-up frames.
fn take_sample(
    renderer: &Renderer,
    scene: &DenseVegetation,
    readback: &Allocation<[u32; COUNTS]>,
    slots: u64,
    ids: &mut Vec<(u64, usize)>,
    collected: &mut [(Script, Vec<Sample>)],
) {
    let stats = renderer.stats();
    let Some(k) = ids.iter().position(|&(id, _)| id == stats.frame) else {
        return;
    };
    let (id, script) = ids.remove(k);
    let counts = readback.as_slice().expect("counts readback")[(id % slots) as usize];
    collected[script].1.push(Sample {
        stats: stats.clone(),
        lighting: renderer.lighting_stats(scene.scene, id),
        counts,
    });
}

/// The samples of a script past its warm-up.
pub fn settled(samples: &[Sample]) -> &[Sample] {
    &samples[(WARM_UP as usize).min(samples.len().saturating_sub(1))..]
}
