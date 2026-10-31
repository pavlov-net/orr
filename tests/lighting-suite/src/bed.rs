//! The lighting test bed: analytic scenes cooked, mounted and placed through orr-scene's
//! public API, rendered from a fixed camera, with a kernel's radiance accumulated over
//! frames and compared with FLIP.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use orr_cook::{
    ArchiveSpec, AssetSpec, LodMode, Manifest, MeshInput, MeshSettings, Quality, Source,
    TextureSettings, Usage,
};
use orr_gpu::{
    Allocation, CommandList, ComputePipeline, GpuPtr, Hazards, KernelSet, MemoryKind, Stages,
};
use orr_lighting::{
    BsdfTables, DirectionalLight, DiscProfile, EnvironmentMapGpu, Light, LightId, LightingKernels,
    LightingView, PathEntry, SceneLighting, SceneLightingSettings, ScenePass, ViewLighting,
    ViewPass,
};
use orr_math::{DVec3, Mat3, Quat, UVec2, Vec2, Vec3};
use orr_pack::material::{Material, slot};
use orr_pack::{AssetId, Span, Target};
use orr_scene::testing::{FlipError, Image, Pixels, flip};
use orr_scene::{
    FrameDesc, FrameTime, GroupWriter, InstanceFlags, InstancesDesc, Lens, LoadState, Loading,
    MountId, PackSource, Projection, RasterView, RendererDesc, SceneDesc, SceneId, Scenes,
    TransformGroupDesc, ViewDesc, ViewHistoryId, ViewTarget,
};

use crate::shaders::{accumulate, exposure, flat_color, guide_read};
use crate::{TestTexture, affine, device, linear_clamp, pipeline, render, texture, upload};

/// Render extent of every bed view.
pub const EXTENT: [u32; 2] = [128, 96];

/// Near plane of every bed view, m.
pub const NEAR: f32 = 0.05;

const PIXELS: u64 = (EXTENT[0] * EXTENT[1]) as u64;

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
pub const WHITE_SPHERE: &str = "white sphere";
pub const CORNELL: &str = "cornell box";
pub const TALL_BOX: &str = "tall box";
pub const SHORT_BOX: &str = "short box";
pub const EMITTER: &str = "emitter";
pub const LEAF: &str = "leaf";
pub const GLASS_SLAB: &str = "glass slab";
pub const SKIN_SLAB: &str = "skin slab";
/// An upright 2 m square perfect mirror (metal), facing +z.
pub const MIRROR: &str = "mirror";
/// The floor as a perfectly smooth black dielectric: only its mirror reflection shows.
pub const SMOOTH_FLOOR: &str = "smooth floor";
/// The floor with a strong bump normal map.
pub const BUMPY_FLOOR: &str = "bumpy floor";
/// An upright 1 m square facing +z whose material names linked surface 1.
pub const PANEL: &str = "panel";
/// A valley 3 m across and 8 m long along z, y = 0.5 x^2, finely tessellated, with
/// discrete levels of 1 cm and 3 cm error.
pub const TROUGH: &str = "trough";

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

/// An upright square of half side `half` around the origin, facing +z.
fn upright(half: f32) -> MeshInput {
    mesh(|b| {
        b.quad(
            0,
            [
                Vec3::new(-half, -half, 0.0),
                Vec3::new(half, -half, 0.0),
                Vec3::new(half, half, 0.0),
                Vec3::new(-half, half, 0.0),
            ],
            Vec3::Z,
            1,
        );
    })
}

fn mesh(build: impl FnOnce(&mut MeshBuilder)) -> MeshInput {
    let mut b = MeshBuilder::default();
    build(&mut b);
    b.finish()
}

/// A grayscale PNG of concentric rings, opaque on the inner 60% of each ring.
fn rings_png() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr-lighting-bed-{}", std::process::id()));
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
    let dir = std::env::temp_dir().join(format!("orr-lighting-bed-{}", std::process::id()));
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
    let mut leaf = Material {
        base_color: [0.2, 0.6, 0.15],
        alpha_mode: Material::MASK,
        alpha_cutoff: 0.5,
        double_sided: 1,
        ..Material::default()
    };
    leaf.textures[slot::OPACITY].texture = AssetId::from_name("rings");
    let glass = Material {
        base_color: [1.0; 3],
        specular_roughness: 0.0,
        specular_ior: 1.5,
        transmission_weight: 1.0,
        ..Material::default()
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
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: "rings".into(),
                    source: Source::Texture {
                        path: rings_png(),
                        settings: TextureSettings {
                            usage: Usage::Linear1,
                            quality: Quality::Fast,
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
                    WHITE_SPHERE,
                    mesh(|b| b.sphere(0, 24, 1.0)),
                    vec![diffuse([1.0; 3])],
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
                    LEAF,
                    mesh(|b| {
                        b.quad(
                            0,
                            [
                                Vec3::new(-1.0, -1.0, 0.0),
                                Vec3::new(1.0, -1.0, 0.0),
                                Vec3::new(1.0, 1.0, 0.0),
                                Vec3::new(-1.0, 1.0, 0.0),
                            ],
                            z,
                            1,
                        );
                    }),
                    vec![leaf],
                ),
                model(GLASS_SLAB, slab(), vec![glass]),
                model(SKIN_SLAB, slab(), vec![skin]),
                AssetSpec {
                    name: "bumps".into(),
                    source: Source::Texture {
                        path: bumps_png(),
                        settings: TextureSettings {
                            usage: Usage::Normal,
                            quality: Quality::Fast,
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
                    BUMPY_FLOOR,
                    mesh(|b| b.quad(0, square(5.0, 0.0), y, 8)),
                    vec![bumpy],
                ),
                model(PANEL, upright(0.5), vec![panel]),
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
            ],
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

/// Threads lent to the loading side until dropped.
struct Workers {
    loading: Arc<Loading>,
    threads: Vec<JoinHandle<()>>,
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.loading.stop_workers();
        for t in self.threads.drain(..) {
            t.join().unwrap();
        }
    }
}

fn time(frame: u64) -> FrameTime {
    FrameTime {
        absolute: frame as f64 / 60.0,
        delta: 1.0 / 60.0,
    }
}

/// World from view for a camera at `eye` looking at `target`, y up.
pub fn look(eye: Vec3, target: Vec3) -> Quat {
    let back = (eye - target).normalize();
    let right = Vec3::Y.cross(back).normalize();
    Quat::from_mat3(&Mat3::from_cols(right, back.cross(right), back))
}

/// Bytes of root memory each frame's kernels may put.
const ROOT_BYTES: u64 = 64 * 1024;

/// What a kernel recording into a bed frame reads and writes.
pub struct BedFrame<'a> {
    /// The frame's index in the accumulation.
    pub index: u32,
    pub seed: u32,
    /// The orr-scene frame id.
    pub frame: u64,
    pub view: &'a RasterView,
    /// The view as the frame rendered it.
    pub desc: ViewDesc<'static>,
    pub lighting: &'a SceneLighting,
    /// The radiance target the kernel writes, at [`EXTENT`].
    pub target: &'a mut LightingView,
    roots: &'a mut Roots,
}

impl BedFrame<'_> {
    /// The frame as the opaque dispatch reads it.
    pub fn pass(&self) -> ViewPass {
        ViewPass {
            frame: self.frame,
            scene: self.view.scene,
            view: self.view.view,
            gbuffer: self.view.gbuffer,
            sample_index: self.index,
            seed: self.seed,
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
        self.memory.gpu.offset(at).cast()
    }
}

/// The mean of a bed's accumulated radiance: RGBA per pixel, rows of [`EXTENT`] top to
/// bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Accumulated {
    pub mean: Vec<[f32; 4]>,
}

impl Accumulated {
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

/// One analytic scene mounted, placed and viewed, ready to accumulate frames.
pub struct Bed {
    pub spec: Spec,
    pub lighting: SceneLighting,
    scenes: Scenes,
    scene: SceneId,
    history: ViewHistoryId,
    _mount: MountId,
    _environment: Option<(TestTexture, Allocation<EnvironmentMapGpu>)>,
    target: LightingView,
    sum: Allocation<[f32; 4]>,
    out: Allocation<[f32; 4]>,
    roots: Roots,
    accumulate: ComputePipeline,
    exposure: ComputePipeline,
    pre_exposure: f32,
    frame: u64,
    last_id: u64,
    light_writes: Vec<(LightId, Option<Light>)>,
    _workers: Workers,
}

impl Bed {
    pub fn new(analytic: Analytic) -> Self {
        Self::with(analytic.spec())
    }

    pub fn with(spec: Spec) -> Self {
        Self::with_renderer(spec, &RendererDesc::default())
    }

    /// The bed for `spec` on a renderer created with `renderer`; orr-lighting's modules are
    /// replaced from its kernels too.
    pub fn with_renderer(spec: Spec, renderer: &RendererDesc) -> Self {
        let kernels = renderer.kernels;
        let d = device();
        let mut scenes = Scenes::new(d, renderer).unwrap();
        let loading = scenes.loading();
        let threads = (0..2)
            .map(|_| {
                let l = Arc::clone(&loading);
                std::thread::spawn(move || l.run_worker())
            })
            .collect();
        let workers = Workers { loading, threads };
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
            light_capacity: 8,
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
                    from_world: [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
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
            w.transforms(group, i as u32, 1).unwrap()[0] = affine(c.linear, c.at);
            w.add_instances(&InstancesDesc {
                model,
                copies: 1,
                transforms: group,
                first: i as u32,
                pose: None,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 1000.0,
                animation_range: 0.0,
                user_tag: i as u32,
            })
            .unwrap();
        }
        for (i, l) in spec.lights.iter().enumerate() {
            w.set_light(LightId(i as u32), Some(l)).unwrap();
        }
        if let Some((_, data)) = &environment {
            w.set_environment(data.gpu.address());
        }
        if let Some((at, pixels_per_radian)) = spec.focus {
            w.set_focus(at.as_dvec3(), pixels_per_radian);
        }
        drop(w);
        let point = f.end(&[]).unwrap();
        render().wait(point, Duration::from_secs(10)).unwrap();

        let lighting =
            SceneLighting::new(d, kernels, &desc, &SceneLightingSettings::default()).unwrap();
        let buffer = |kind, name| d.gpu_malloc::<[f32; 4]>(kind, PIXELS, name).unwrap();
        Self {
            spec,
            lighting,
            scenes,
            scene,
            history,
            _mount: mount,
            _environment: environment,
            target: LightingView::new(d, EXTENT).unwrap(),
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
            light_writes: Vec::new(),
            _workers: workers,
        }
    }

    /// Runs `frames` frames of the scene, each recording `kernel` after the scene pass and
    /// adding the radiance it wrote, and returns the mean. A frame's index and `seed`
    /// choose its samples, so the same `frames` and `seed` give the same image.
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

    /// The orr-scene frame id of the last frame run.
    pub fn last_frame(&self) -> u64 {
        self.last_id
    }

    /// Sets light `id` from the next frame on.
    pub fn set_light(&mut self, id: LightId, light: Option<Light>) {
        self.light_writes.push((id, light));
    }

    /// Renders later frames at pre-exposure `p`, as the output's meter would set it.
    pub fn set_pre_exposure(&mut self, p: f32) {
        self.pre_exposure = p;
        self.run_frame(0, 0, true, true, false, &mut |_, _| {});
    }

    /// Reads the guide targets of the last frame.
    pub fn guides(&self) -> GuideImages {
        let d = device();
        let t = &self.target;
        let textures = [
            t.diffuse_albedo(),
            t.specular_albedo(),
            t.normal_roughness(),
            t.specular_motion(),
            t.depth(),
            t.motion(),
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
        let out = crate::readback::<[f32; 4]>(PIXELS * 6, "guide readback");
        let read = pipeline(&guide_read::MODULE, "main");
        let mut list = render().command_list().unwrap();
        let root = upload(
            &[guide_read::Root {
                diffuse_albedo: index[0],
                specular_albedo: index[1],
                normal_roughness: index[2],
                specular_motion: index[3],
                depth: index[4],
                motion: index[5],
                out: out.gpu,
                width: EXTENT[0],
                count: PIXELS as u32,
                _pad0: 0,
            }],
            "guide read root",
        );
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&read, root.gpu, [PIXELS.div_ceil(64) as u32, 1, 1]);
        crate::finish(list);
        for slot in slots {
            d.free_image_slot_after_submitted(slot);
        }
        let v = out.as_slice().unwrap();
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
            jitter: Vec2::ZERO,
            reset_history: cut,
            mip_bias: 0.0,
            target: ViewTarget::AfterRr,
        };
        if !self.light_writes.is_empty() {
            let mut w = f.scene(self.scene).unwrap();
            for (id, l) in self.light_writes.drain(..) {
                w.set_light(id, l.as_ref()).unwrap();
            }
        }
        let view = f.raster_view(&desc).unwrap();
        let id = f.id();
        self.last_id = id;
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
                roots: &mut self.roots,
            },
        );
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::COMPUTE,
            Hazards::empty(),
        );
        let root = self.roots.put(&accumulate::Root {
            sum: self.sum.gpu,
            radiance: self.target.radiance_index(),
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
                self.out.gpu.cast(),
                self.sum.gpu.cast(),
                PIXELS * size_of::<[f32; 4]>() as u64,
            );
            list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
        }
        let point = f.end(&[]).unwrap();
        render().wait(point, Duration::from_secs(10)).unwrap();
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
            scene: f.view.scene,
            view: f.view.view,
            gbuffer: f.view.gbuffer,
            radiance: f.target.radiance_storage(),
            _pad2: 0,
            sky: [self.sky[0], self.sky[1], self.sky[2], 0.0],
            frame: f.index,
            seed: f.seed,
            _pad0: 0,
            _pad1: 0,
        });
        list.dispatch(
            &self.pipeline,
            root,
            [EXTENT[0].div_ceil(8), EXTENT[1].div_ceil(8), 1],
        );
    }
}

/// orr-lighting's opaque dispatch and guide resolve recording into bed frames with
/// `settings`.
pub struct Paths {
    kernels: LightingKernels,
    _tables: BsdfTables,
    pub settings: ViewLighting,
}

impl Paths {
    pub fn new(kernels: &KernelSet, entry: PathEntry, settings: ViewLighting) -> Self {
        let mut list = render().command_list().unwrap();
        let tables = BsdfTables::new(device(), &mut list).unwrap();
        crate::run(list);
        Self {
            kernels: LightingKernels::new(device(), kernels, &tables, entry).unwrap(),
            _tables: tables,
            settings,
        }
    }

    /// Starts the view's frame, then records the opaque dispatch and the guide resolve.
    pub fn record(&self, list: &mut CommandList<'_>, f: &mut BedFrame<'_>) {
        let pass = f.pass();
        f.target
            .begin(f.frame, f.lighting, &f.desc, false, &self.settings);
        self.kernels
            .record_opaque(list, f.lighting, f.target, &pass, &self.settings);
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::COMPUTE,
            Hazards::empty(),
        );
        self.kernels
            .record_guides(list, f.target, &pass, &self.settings);
    }
}
