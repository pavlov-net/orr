//! Dielectrics under a low sun: a pool of wave-displaced water (a producer entry) holding a
//! half-submerged glass sphere and a white ball on its floor, an amber glass slab and a
//! scattering jade block on the ground. Each frame traces the opaque paths, then one camera
//! ray per pixel against the dielectric mask and the path through what it finds: Fresnel
//! reflection or refraction, the water's and the amber's absorption, the jade's scattering,
//! and nested media where the glass meets the water. The camera orbits the pool, or with
//! `--dive` bobs through the water's surface, the eye's medium following it. A still camera
//! averages its frames in a scene-referred hook.
//!
//! `dielectrics [--dive] [--still] [--frames N] [--headless] [--out file.png]`: headless runs
//! keep the camera still, render 480 x 270 for 120 frames unless told otherwise and write the
//! last frame as a PNG.

use std::sync::Arc;
use std::time::{Duration, Instant};

use examples::average::Average;
use examples::field::{Slab, patch_triangles};
use examples::mesh::MeshBuilder;
use examples::shaders::slab;
use examples::sky::{self, Sky};
use examples::window::Window;
use examples::{Capture, Options, Workers, assert_clean, device, fixed_time, mount_at};
use orr::orr_gpu::{
    Format, QueueKind, SamplerIndex, SpecData, TextureDesc, TextureMemory, TextureUsage,
    TimelinePoint,
};
use orr::orr_pack::material::Material;
use orr::orr_pack::{AssetId, Target};
use orr::{
    Affine3, BoundTexture, ClusterTopology, DirectionalLight, DiscProfile, Display, DisplayRequest,
    FrameDesc, FrameTime, GroupWriter, HookPass, InstanceFlags, InstancesDesc, Lens, Light,
    LightId, MaterialGpu, PackSource, ProducerId, Projection, Renderer, RendererDesc, SceneDesc,
    SceneLightingSettings, ScreenshotRequest, TextureFilter, TransformGroupDesc, ViewDesc,
    ViewHooks, ViewLighting, ViewOutput, ViewTarget,
};
use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshInput, MeshSettings, Source};
use orr_math::camera::rh::view;
use orr_math::{DVec3, UVec2, Vec2, Vec3};

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
/// The frame id the scene is created in, after the pack has mounted.
const FIRST_FRAME: u64 = 500;
const FOV_Y: f32 = 0.8;
/// Unit direction to the sun, 15 degrees above the horizon.
const SUN: Vec3 = Vec3::new(-0.683_012_7, 0.258_819, 0.683_012_7);
/// Lux, linear Rec.709.
const SUN_ILLUMINANCE: Vec3 = Vec3::new(100_000.0, 92_000.0, 80_000.0);
const SKY: &str = "sky";
/// The cooked models, one candidate class each.
const MODELS: [&str; 5] = ["ground", "pool", "glass", "amber", "jade"];
const POND: ProducerId = ProducerId(0);
/// The pool: 4 m square around the origin, 1 m deep, water 0.15 m below its rim.
const POOL_HALF: f32 = 2.0;
const POOL_DEPTH: f32 = 1.0;
const WATER_LEVEL: f32 = -0.15;
/// Radians per second.
const ORBIT_RATE: f64 = 0.15;
/// Where a still camera stands on the orbit, radians.
const STILL_ANGLE: f64 = 0.5;
/// Frames a still camera renders before it starts averaging, while exposure settles.
const ACCUMULATE_AFTER: u64 = 30;

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("dielectrics");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// The camera at time `t`: on its orbit around the pool, or bobbing through the water.
fn camera(t: f64, still: bool, dive: bool) -> (Vec3, Vec3) {
    if dive {
        let y = WATER_LEVEL + 0.45 * (t * 0.7).sin() as f32;
        return (
            Vec3::new(1.4, y, 1.4),
            Vec3::new(-1.0, WATER_LEVEL - 0.1, -1.2),
        );
    }
    let angle = if still {
        STILL_ANGLE
    } else {
        STILL_ANGLE + t * ORBIT_RATE
    };
    let eye = Vec3::new(angle.sin() as f32 * 6.0, 1.7, angle.cos() as f32 * 6.0);
    (eye, Vec3::new(0.0, -0.3, 0.0))
}

/// Runs the example; headless runs and runs with `--out` return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new("orr dielectrics", 1280, 720));
    let desc = RendererDesc::default();
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
    let _workers = Workers::new(renderer.loading(), 2);
    let (mount, mut frame) = mount_at(&mut renderer, PackSource::Memory(cook()), FIRST_FRAME);

    let scene = renderer
        .create_scene(
            &SceneDesc {
                instance_capacity: 8,
                group_capacity: 2,
                light_capacity: 1,
                material_set_capacity: 4,
                emissive_capacity: 4,
                producer_capacity: 1,
                part_capacity: 0,
                ptlas_partitions: 4,
                global_instances: 8,
                ..SceneDesc::default()
            },
            &SceneLightingSettings::default(),
        )
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    let models = MODELS.map(|name| {
        renderer
            .model(mount, AssetId::from_name(name))
            .unwrap_or_else(|e| panic!("model {name}: {e}"))
    });
    let environment = sky::environment(&renderer, gpu, mount, SKY);
    let slab_kernel = renderer
        .compute_pipeline(&slab::MODULE, "main", SpecData::default())
        .expect("slab pipeline");
    let pond = Slab {
        patches: 2,
        patch_size: POOL_HALF,
        corner: [-POOL_HALF; 2],
        level: WATER_LEVEL,
        amplitude: 0.015,
    };
    let patch = patch_triangles();
    let topology = [ClusterTopology {
        vertex_count: 81,
        triangles: &patch,
    }];

    frame += 1;
    let listed = [scene];
    let f = renderer
        .begin_frame(FrameDesc {
            time: fixed_time(frame),
            scenes: &listed,
            display: None,
        })
        .expect("begin frame");
    let water = Material {
        base_color: [1.0; 3],
        specular_roughness: 0.02,
        specular_ior: 1.333,
        transmission_weight: 1.0,
        transmission_color: [0.62, 0.86, 0.9],
        transmission_depth: 1.5,
        ..Material::default()
    };
    let water = f
        .materials()
        .create(&MaterialGpu::new(
            &water,
            &[BoundTexture::NONE; 12],
            SamplerIndex(0),
        ))
        .expect("water");
    let mut w = f.scene(scene).expect("scene writer");
    let group = w
        .create_transform_group(&TransformGroupDesc {
            count: models.len() as u32 + 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .expect("transform group");
    w.transforms(group, 0, models.len() as u32 + 1)
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
    let water_set = [water];
    w.set_producer(
        POND,
        Some(&pond.desc(&topology, &water_set, (group, models.len() as u32))),
    )
    .expect("pond");
    w.set_light(
        LightId(0),
        Some(&Light::Directional(DirectionalLight {
            direction_to_light: SUN,
            illuminance: SUN_ILLUMINANCE,
            angular_radius: 0.004_65,
            disc: DiscProfile::LimbDarkened,
        })),
    )
    .expect("sun");
    w.set_environment(environment.gpu.address());
    drop(w);
    f.end().expect("end frame");

    let headless_target = options.headless.then(|| {
        let [w, h] = HEADLESS_EXTENT;
        gpu.create_texture(
            &TextureDesc::d2(
                Format::Rgba8Unorm,
                w,
                h,
                TextureUsage::STORAGE | TextureUsage::SAMPLED,
            ),
            TextureMemory::Dedicated,
            "dielectrics target",
        )
        .expect("headless target")
    });
    let capture = options.headless || options.out.is_some();
    let still = options.headless || options.has("--still");
    let dive = options.has("--dive");
    let average_frames = still && !dive;
    let mut average = Average::new(&mut renderer, gpu, desc.frames_in_flight, ACCUMULATE_AFTER);
    let frames = options
        .frames
        .or(options.headless.then_some(HEADLESS_FRAMES));
    let started = Instant::now();
    let mut previous = 0.0;
    let mut last: Option<(u64, TimelinePoint)> = None;
    for i in 0.. {
        if frames.is_some_and(|n| i >= n) {
            break;
        }
        if let Some(w) = window.as_mut()
            && !w.pump()
        {
            break;
        }
        let final_frame = frames.is_some_and(|n| i + 1 == n);
        // Headless runs step a fixed clock, so they repeat exactly.
        let t = if options.headless {
            fixed_time(i).absolute
        } else {
            started.elapsed().as_secs_f64()
        };
        let time = FrameTime {
            absolute: fixed_time(frame + 1).absolute + t,
            delta: (t - previous) as f32,
        };
        previous = t;
        let (eye, target) = camera(t, still, dive);
        // The pixels flip the eye's medium where the near plane crosses the surface.
        let lighting = ViewLighting {
            camera_medium: (dive && eye.y < WATER_LEVEL).then_some(water),
            ..ViewLighting::default()
        };

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: display.as_mut(),
            })
            .expect("begin frame");
        let id = f.id();
        let (extent, view_target) = match (&headless_target, f.display()) {
            (Some(t), _) => (HEADLESS_EXTENT, ViewTarget::Texture(t)),
            (None, Some((_, extent))) => (extent, ViewTarget::Display),
            (None, None) => {
                last = Some((id, f.end().expect("end frame")));
                continue;
            }
        };
        f.scene(scene)
            .expect("scene writer")
            .set_focus(eye.as_dvec3(), extent[1] as f32 / FOV_Y);
        f.update_scenes(&mut |c| pond.record(c, POND, slab_kernel, t as f32))
            .expect("update scenes");
        if let Some(t) = headless_target.as_ref().filter(|_| i == 0) {
            f.list().expect("frame list").discard(t);
        }
        let mut average = average.hook(&f, i, extent);
        let mut hooks = ViewHooks {
            scene_referred: average_frames
                .then_some(&mut average as &mut dyn FnMut(&mut HookPass<'_, '_>)),
            ..ViewHooks::default()
        };
        let output = ViewOutput {
            screenshot: (capture && final_frame)
                .then_some(ScreenshotRequest::Display { include_ui: false }),
            ..ViewOutput::default()
        };
        f.render_view(
            &ViewDesc {
                scene,
                history,
                position: eye.as_dvec3(),
                rotation: view::look_at_quat(eye, target, Vec3::Y).inverse(),
                projection: Projection {
                    fov_y: FOV_Y,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: UVec2::from(extent),
                jitter: Vec2::ZERO,
                reset_history: i == 0,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
                target: view_target,
            },
            &lighting,
            &output,
            &mut hooks,
        )
        .expect("render view");
        last = Some((id, f.end().expect("end frame")));
    }

    if let Some(d) = display.as_ref() {
        d.wait_idle();
    }
    if let Some((_, point)) = last {
        gpu.queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(10))
            .expect("wait");
    }
    let captured = last
        .filter(|_| capture)
        .and_then(|(id, _)| renderer.screenshots(history, id).pop())
        .map(|shot| Capture::from_screenshot(&shot));
    drop(average);
    drop(headless_target);
    drop(environment);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}

/// The ground around the pool, the pool's tiled inside, and the glass, amber and jade
/// objects, cooked with the sky into an in-memory pack.
fn cook() -> Arc<[u8]> {
    let ground = Material {
        base_color: [0.48, 0.45, 0.4],
        specular_roughness: 0.8,
        ..Material::default()
    };
    let tiles = Material {
        base_color: [0.55, 0.75, 0.78],
        specular_roughness: 0.35,
        ..Material::default()
    };
    let ball = Material {
        base_color: [0.85; 3],
        specular_roughness: 0.4,
        ..Material::default()
    };
    let glass = Material {
        base_color: [1.0; 3],
        specular_roughness: 0.0,
        specular_ior: 1.5,
        transmission_weight: 1.0,
        ..Material::default()
    };
    let amber = Material {
        transmission_color: [0.95, 0.55, 0.15],
        transmission_depth: 0.2,
        ..glass
    };
    let jade = Material {
        specular_roughness: 0.15,
        transmission_color: [0.35, 0.7, 0.45],
        transmission_depth: 0.1,
        transmission_scatter: [0.5, 0.8, 0.6],
        transmission_scatter_anisotropy: 0.3,
        ..glass
    };

    let (h, p) = (20.0, POOL_HALF);
    let flat = |x0: f32, z0: f32, x1: f32, z1: f32| {
        [
            Vec3::new(x0, 0.0, z1),
            Vec3::new(x1, 0.0, z1),
            Vec3::new(x1, 0.0, z0),
            Vec3::new(x0, 0.0, z0),
        ]
    };
    let mut around = MeshBuilder::default();
    for corners in [
        flat(-h, p, h, h),
        flat(-h, -h, h, -p),
        flat(-h, -p, -p, p),
        flat(p, -p, h, p),
    ] {
        around.quad(0, corners, 8);
    }
    // The pool's floor and walls, facing inward.
    let d = POOL_DEPTH;
    let c = Vec3::new;
    let mut walls = MeshBuilder::default();
    for corners in [
        [c(-p, -d, p), c(p, -d, p), c(p, -d, -p), c(-p, -d, -p)],
        [c(-p, -d, p), c(-p, -d, -p), c(-p, 0.0, -p), c(-p, 0.0, p)],
        [c(p, -d, -p), c(p, -d, p), c(p, 0.0, p), c(p, 0.0, -p)],
        [c(-p, -d, -p), c(p, -d, -p), c(p, 0.0, -p), c(-p, 0.0, -p)],
        [c(p, -d, p), c(-p, -d, p), c(-p, 0.0, p), c(p, 0.0, p)],
    ] {
        walls.quad(0, corners, 4);
    }
    walls.sphere(1, Vec3::new(-0.9, -POOL_DEPTH + 0.35, -0.6), 0.35, 24);
    let mut sphere = MeshBuilder::default();
    sphere.sphere(0, Vec3::new(0.8, WATER_LEVEL, 0.6), 0.5, 32);
    let mut slab = MeshBuilder::default();
    slab.cuboid(0, Vec3::new(-4.6, 0.0, 0.6), Vec3::new(-3.2, 0.12, 1.6), 1);
    let mut block = MeshBuilder::default();
    block.cuboid(0, Vec3::new(3.0, 0.0, -2.0), Vec3::new(3.8, 0.8, -1.2), 1);

    let mesh = |name: &str, mesh: MeshInput, materials: Vec<Material>| AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings::default(),
        },
    };
    let sky = Sky::new("dielectrics", SUN);
    let [g, pool, glass_name, amber_name, jade_name] = MODELS;
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                sky.asset(SKY),
                mesh(g, around.finish(), vec![ground]),
                mesh(pool, walls.finish(), vec![tiles, ball]),
                mesh(glass_name, sphere.finish(), vec![glass]),
                mesh(amber_name, slab.finish(), vec![amber]),
                mesh(jade_name, block.finish(), vec![jade]),
            ],
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0)
}
