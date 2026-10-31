//! The examples' cooked scene: a gold knot, a glossy and a mirror sphere, a block and a
//! glowing panel on a ground, under a cooked sky and a 100 klx sun.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orr::orr_gpu::{AddressMode, Allocation, Device, Filter, MemoryKind, SamplerDesc};
use orr::orr_lighting::EnvironmentMapGpu;
use orr::orr_pack::material::Material;
use orr::orr_pack::{AssetId, Target};
use orr::{
    Affine3, DirectionalLight, DiscProfile, FrameDesc, GroupWriter, InstanceFlags, InstancesDesc,
    Light, LightId, LoadState, PackSource, Renderer, SceneDesc, SceneId, SceneLightingSettings,
    TransformGroupDesc, ViewHistoryId,
};
use orr_cook::{
    ArchiveSpec, AssetSpec, Manifest, MeshInput, MeshSettings, Quality, Source, TextureSettings,
    Usage,
};
use orr_math::{DVec3, Vec3};

use crate::fixed_time;
use crate::mesh::MeshBuilder;

const OBJECTS: &str = "objects";
const GROUND: &str = "ground";
const SKY: &str = "sky";

/// The frame id the scene is created in, after the pack has mounted.
const FIRST_FRAME: u64 = 500;
pub const FOV_Y: f32 = 0.8;
/// Unit direction to the sun.
const SUN: Vec3 = Vec3::new(-0.453_413_4, 0.654_930_5, 0.604_551_3);
/// Lux, linear Rec.709.
const SUN_ILLUMINANCE: Vec3 = Vec3::new(100_000.0, 95_000.0, 88_000.0);
/// The sun's angular radius, radians.
const SUN_RADIUS: f32 = 0.004_65;
/// What the camera looks at.
pub const TARGET: Vec3 = Vec3::new(0.0, 0.9, 0.0);

/// The scene loaded into a renderer, with a view history for one camera.
pub struct Showcase {
    pub scene: SceneId,
    pub history: ViewHistoryId,
    /// The last frame id `load` used.
    pub frame: u64,
    _environment: Allocation<EnvironmentMapGpu>,
}

impl Showcase {
    /// Cooks and mounts the pack, renders empty frames until it has mounted and the frame
    /// id reaches a fixed value (light sampling's random sequence follows the frame id, so
    /// headless runs repeat), then creates the scene in one more frame.
    pub fn load(renderer: &mut Renderer, gpu: &Device) -> Self {
        let mount = renderer.mount(PackSource::Memory(cook()));
        let mut frame = 0;
        let start = Instant::now();
        while matches!(renderer.mount_state(mount), LoadState::Pending) || frame < FIRST_FRAME {
            frame += 1;
            let f = renderer
                .begin_frame(FrameDesc {
                    time: fixed_time(frame),
                    scenes: &[],
                    display: None,
                })
                .expect("begin frame");
            f.end().expect("end frame");
            assert!(start.elapsed() < Duration::from_secs(60), "mount pending");
            if matches!(renderer.mount_state(mount), LoadState::Pending) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        if frame > FIRST_FRAME {
            log::warn!("mounting took {frame} frames; this run will not repeat others exactly");
        }
        if let LoadState::Failed(e) = renderer.mount_state(mount) {
            panic!("mount failed: {e}");
        }

        let scene = renderer
            .create_scene(
                &SceneDesc {
                    instance_capacity: 16,
                    group_capacity: 4,
                    light_capacity: 1,
                    material_set_capacity: 16,
                    emissive_capacity: 16,
                    part_capacity: 0,
                    ptlas_partitions: 16,
                    global_instances: 16,
                    ..SceneDesc::default()
                },
                &SceneLightingSettings::default(),
            )
            .expect("scene");
        let history = renderer.create_view_history().expect("view history");
        let models = [OBJECTS, GROUND].map(|name| {
            renderer
                .model(mount, AssetId::from_name(name))
                .unwrap_or_else(|e| panic!("model {name}: {e}"))
        });
        let mut environment = gpu
            .gpu_malloc::<EnvironmentMapGpu>(MemoryKind::Upload, 1, "showcase sky")
            .expect("environment");
        environment.as_mut_slice().expect("environment")[0] = EnvironmentMapGpu {
            map: renderer
                .texture(mount, AssetId::from_name(SKY))
                .expect("sky texture"),
            sampler: gpu
                .sampler(&SamplerDesc {
                    mag: Filter::Linear,
                    min: Filter::Linear,
                    mip: Filter::Linear,
                    address: [AddressMode::ClampToEdge; 3],
                    ..SamplerDesc::default()
                })
                .expect("sky sampler"),
            intensity: 1.0,
            primaries: orr_color::Primaries::Rec709.id().into(),
            from_world: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        };

        frame += 1;
        let listed = [scene];
        let f = renderer
            .begin_frame(FrameDesc {
                time: fixed_time(frame),
                scenes: &listed,
                display: None,
            })
            .expect("begin frame");
        let mut w = f.scene(scene).expect("scene writer");
        let group = w
            .create_transform_group(&TransformGroupDesc {
                count: models.len() as u32,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .expect("transform group");
        w.transforms(group, 0, models.len() as u32)
            .expect("transforms")
            .fill(Affine3::IDENTITY);
        for (i, &model) in models.iter().enumerate() {
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
                user_tag: 0,
            })
            .expect("instances");
        }
        w.set_light(
            LightId(0),
            Some(&Light::Directional(DirectionalLight {
                direction_to_light: SUN,
                illuminance: SUN_ILLUMINANCE,
                angular_radius: SUN_RADIUS,
                disc: DiscProfile::LimbDarkened,
            })),
        )
        .expect("sun");
        w.set_environment(environment.gpu.address());
        drop(w);
        f.end().expect("end frame");
        Self {
            scene,
            history,
            frame,
            _environment: environment,
        }
    }
}

/// The camera on its orbit around [`TARGET`] at `angle` radians.
pub fn orbit_eye(angle: f64) -> Vec3 {
    TARGET + Vec3::new(angle.sin() as f32 * 6.5, 1.1, angle.cos() as f32 * 6.5)
}

/// The objects, the ground and the sky, cooked into an in-memory pack.
fn cook() -> Arc<[u8]> {
    let gold = Material {
        base_color: [1.0, 0.77, 0.34],
        metalness: 1.0,
        specular_roughness: 0.25,
        ..Material::default()
    };
    let glossy = Material {
        base_color: [0.6, 0.05, 0.04],
        specular_roughness: 0.12,
        ..Material::default()
    };
    let mirror = Material {
        base_color: [0.95, 0.95, 0.95],
        metalness: 1.0,
        specular_roughness: 0.02,
        ..Material::default()
    };
    let block = Material {
        base_color: [0.8, 0.8, 0.78],
        specular_roughness: 0.6,
        ..Material::default()
    };
    let glow = Material {
        base_color: [0.1, 0.1, 0.1],
        emission_color: [1.0, 0.55, 0.25],
        emission_luminance: 20_000.0,
        ..Material::default()
    };
    let ground = Material {
        base_color: [0.5, 0.5, 0.48],
        specular_roughness: 0.75,
        ..Material::default()
    };

    let mut objects = MeshBuilder::default();
    objects.knot(0, Vec3::new(0.0, 1.4, 0.0), 512, 32);
    objects.sphere(1, Vec3::new(-2.3, 0.8, 1.0), 0.8, 48);
    objects.sphere(2, Vec3::new(2.3, 0.7, 1.2), 0.7, 48);
    objects.cuboid(3, Vec3::new(-1.6, 0.0, -2.6), Vec3::new(-0.2, 0.9, -1.6), 4);
    objects.quad(
        4,
        [
            Vec3::new(0.8, 0.1, -2.4),
            Vec3::new(2.4, 0.1, -1.8),
            Vec3::new(2.4, 1.3, -1.8),
            Vec3::new(0.8, 1.3, -2.4),
        ],
        2,
    );
    let mut floor = MeshBuilder::default();
    let h = 60.0;
    floor.quad(
        0,
        [
            Vec3::new(-h, 0.0, h),
            Vec3::new(h, 0.0, h),
            Vec3::new(h, 0.0, -h),
            Vec3::new(-h, 0.0, -h),
        ],
        24,
    );
    let mesh = |name: &str, mesh: MeshInput, materials: Vec<Material>| AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings::default(),
        },
    };
    let sky = sky_hdr();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: SKY.into(),
                    source: Source::Texture {
                        path: sky.clone(),
                        settings: TextureSettings {
                            usage: Usage::Environment {
                                light_threshold: None,
                                max_lights: 0,
                            },
                            quality: Quality::Fast,
                            ..TextureSettings::default()
                        },
                    },
                },
                mesh(
                    OBJECTS,
                    objects.finish(),
                    vec![gold, glossy, mirror, block, glow],
                ),
                mesh(GROUND, floor.finish(), vec![ground]),
            ],
        )],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0);
    let _ = std::fs::remove_dir_all(sky.parent().expect("temp dir"));
    pack
}

/// Sky radiance in nits, linear Rec.709, along unit direction `d`: a gradient from a bright
/// horizon to a blue zenith with a glow around the sun, over a dim ground.
fn sky(d: Vec3) -> Vec3 {
    let glow = 6.0 * (60.0 * (d.dot(SUN) - 1.0)).exp();
    if d.y < 0.0 {
        return Vec3::new(0.55, 0.52, 0.48) * 6000.0;
    }
    let horizon = Vec3::new(0.85, 0.9, 1.0) * 5000.0;
    let zenith = Vec3::new(0.2, 0.38, 1.0) * 4000.0;
    horizon.lerp(zenith, d.y.sqrt()) * (1.0 + glow)
}

/// The sky as a 1024 x 512 equirectangular Radiance HDR file in a temp directory, for the
/// cook to read.
fn sky_hdr() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orr-showcase-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("sky.hdr");
    let (w, h) = (1024u32, 512u32);
    let mut bytes = format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            let (theta, phi) = (
                (y as f32 + 0.5) / h as f32 * std::f32::consts::PI,
                (x as f32 + 0.5) / w as f32 * std::f32::consts::TAU - std::f32::consts::PI,
            );
            let d = Vec3::new(
                theta.sin() * phi.sin(),
                theta.cos(),
                -theta.sin() * phi.cos(),
            );
            bytes.extend(rgbe(sky(d)));
        }
    }
    std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    path
}

/// Radiance's shared-exponent encoding of `c`.
fn rgbe(c: Vec3) -> [u8; 4] {
    let v = c.max_element();
    if v < 1e-30 {
        return [0; 4];
    }
    let e = v.log2().floor() as i32 + 1;
    let scale = 256.0 / 2f32.powi(e);
    let m = |x: f32| (x * scale).clamp(0.0, 255.0) as u8;
    [m(c.x), m(c.y), m(c.z), (e + 128) as u8]
}
