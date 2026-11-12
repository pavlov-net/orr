//! The examples' cooked scene: a gold knot, a glossy and a mirror sphere, a block and a
//! glowing panel on a ground, under `orr_atmosphere`'s sky and medium and a sun. A renderer
//! that lights it takes `linked::KERNELS` and records [`Showcase::sky`] in every frame's
//! `update_scenes`.

use std::sync::Arc;

use orr::orr_gpu::Device;
use orr::orr_pack::material::Material;
use orr::orr_pack::{AssetId, Target};
use orr::{
    FrameDesc, GroupWriter, InstanceFlags, InstancesDesc, Light, LightId, PackSource, Renderer,
    SceneDesc, SceneId, SceneLightingSettings, TransformGroupDesc,
};
use orr_atmosphere::presets::solar_disc;
use orr_atmosphere::{Atmosphere, AtmosphereState};
use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshInput, MeshSettings, Source};
use orr_math::{Affine3, DVec3, Vec3};

use crate::mesh::MeshBuilder;
use crate::{fixed_time, mount_at};

const OBJECTS: &str = "objects";
const GROUND: &str = "ground";

/// The frame id the scene is created in, after the pack has mounted.
const FIRST_FRAME: u64 = 500;
pub const FOV_Y: f32 = 0.8;
/// Unit direction to the sun.
pub const SUN: Vec3 = Vec3::new(-0.453_413_4, 0.654_930_5, 0.604_551_3);
/// The ground's color, which the atmosphere's ground carries on to the horizon.
const GROUND_COLOR: [f32; 3] = [0.5, 0.5, 0.48];
/// What the camera looks at.
pub const TARGET: Vec3 = Vec3::new(0.0, 0.9, 0.0);

/// The scene loaded into a renderer.
pub struct Showcase {
    pub scene: SceneId,
    /// The last frame id `load` used.
    pub frame: u64,
    pub sky: AtmosphereState,
}

impl Showcase {
    /// Cooks and mounts the pack, renders empty frames until it has mounted and the frame
    /// id reaches a fixed value (light sampling's random sequence follows the frame id, so
    /// headless runs repeat), then creates the scene in one more frame.
    pub fn load(renderer: &mut Renderer, gpu: &Device) -> Self {
        let (mount, mut frame) = mount_at(renderer, PackSource::Memory(cook()), FIRST_FRAME);

        let scene = renderer
            .create_scene(
                &SceneDesc {
                    emissive_capacity: 16,
                    global_instances: 16,
                    ..SceneDesc::default()
                },
                &SceneLightingSettings::default(),
            )
            .expect("scene");
        let models = [OBJECTS, GROUND].map(|name| {
            renderer
                .model(mount, AssetId::from_name(name))
                .unwrap_or_else(|e| panic!("model {name}: {e}"))
        });
        let atmosphere = Atmosphere {
            ground_albedo: Vec3::from(GROUND_COLOR),
            ..Atmosphere::earth(DVec3::ZERO, Vec3::Y)
        };
        let mut sky = AtmosphereState::new(gpu, &atmosphere, 0).expect("atmosphere");
        sky.set_lights(&[LightId(0)]);
        sky.skip_fades();

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
            .fill(Affine3::IDENTITY.into());
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
        w.set_light(LightId(0), Some(&sun())).expect("sun");
        w.set_environment(sky.data());
        w.set_medium(sky.data());
        drop(w);
        f.end().expect("end frame");
        Self { scene, frame, sky }
    }
}

/// The sun along [`SUN`].
pub fn sun() -> Light {
    Light::Directional(solar_disc(SUN))
}

/// The camera on its orbit around [`TARGET`] at `angle` radians.
pub fn orbit_eye(angle: f64) -> Vec3 {
    TARGET + Vec3::new(angle.sin() as f32 * 6.5, 1.1, angle.cos() as f32 * 6.5)
}

/// The objects and the ground, cooked into an in-memory pack.
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
        double_sided: 1,
        ..Material::default()
    };
    let ground = Material {
        base_color: GROUND_COLOR,
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
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                mesh(
                    OBJECTS,
                    objects.finish(),
                    vec![gold, glossy, mirror, block, glow],
                ),
                mesh(GROUND, floor.finish(), vec![ground]),
            ],
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0)
}
