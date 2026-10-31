//! A square at night lit by 304 local lights (160 lamps along its streets, 32 spotlights on
//! its buildings and 112 lit windows) and 16 glowing signs, under a moon and two faint
//! colored skies. Each frame traces one path per pixel. Next-event estimation draws its
//! candidates from where the frames before found their light: the scene's light selection
//! follows the lights that reached the view's pixels, and each screen tile also draws from
//! the lights its own pixels found. The camera orbits the square.
//!
//! `many-lights [--still] [--power] [--frames N] [--headless] [--out file.png]`. `--power`
//! selects lights by emitted power alone, without the feedback. Headless runs keep the camera
//! still, render 480 x 270 for 120 frames unless told otherwise and write the last frame as a
//! PNG.

use std::sync::Arc;
use std::time::{Duration, Instant};

use examples::mesh::MeshBuilder;
use examples::window::Window;
use examples::{Capture, Options, Workers, assert_clean, device, fixed_time, mount_at};
use orr::orr_gpu::{Format, QueueKind, TextureDesc, TextureMemory, TextureUsage, TimelinePoint};
use orr::orr_pack::material::Material;
use orr::orr_pack::{AssetId, Target};
use orr::{
    Affine3, Cone, DirectionalLight, DiscProfile, Display, DisplayRequest, FrameDesc, FrameTime,
    GroupWriter, InstanceFlags, InstancesDesc, Lens, Light, LightId, PackSource, Projection,
    Renderer, RendererDesc, SceneDesc, SceneLightingSettings, ScreenshotRequest, TextureFilter,
    TransformGroupDesc, ViewDesc, ViewHooks, ViewLighting, ViewOutput, ViewTarget,
};
use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshSettings, Source};
use orr_math::camera::rh::view;
use orr_math::{DVec3, UVec2, Vec2, Vec3};

const TOWN: &str = "town";

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
/// The frame id the scene is created in, after the pack has mounted.
const FIRST_FRAME: u64 = 500;
const FOV_Y: f32 = 0.9;
const TARGET: Vec3 = Vec3::new(0.0, 1.5, 0.0);
/// Radians per second.
const ORBIT_RATE: f64 = 0.1;
/// Where a still camera stands on the orbit, radians.
const STILL_ANGLE: f64 = 0.7;

/// Buildings per side of the square, their spacing and footprint, m.
const BLOCKS: i32 = 4;
const BLOCK_SPACING: f32 = 9.0;
const BLOCK_SIZE: f32 = 5.0;
/// Street lamps per side of each street.
const LAMPS: i32 = 16;
const LAMP_HEIGHT: f32 = 3.2;
/// Glowing signs, one per building face.
const SIGNS: usize = 4;

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("many-lights");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// The center of building `(i, j)` on the ground.
fn block(i: i32, j: i32) -> Vec3 {
    let offset = (BLOCKS - 1) as f32 * 0.5;
    Vec3::new(
        (i as f32 - offset) * BLOCK_SPACING,
        0.0,
        (j as f32 - offset) * BLOCK_SPACING,
    )
}

/// The height of building `(i, j)`, m.
fn block_height(i: i32, j: i32) -> f32 {
    4.0 + ((i * 7 + j * 3) % 5) as f32 * 1.5
}

/// A lamp color from a warm to a cool white by `t` in [0, 1].
fn lamp_color(t: f32) -> Vec3 {
    Vec3::new(1.0, 0.72, 0.42).lerp(Vec3::new(0.8, 0.9, 1.0), t)
}

/// Every local light of the square: sphere lamps along the streets, spotlights on the
/// buildings and rect windows.
fn lights() -> Vec<Light> {
    let mut lights = Vec::new();
    let half = BLOCKS as f32 * BLOCK_SPACING * 0.5;
    for street in 0..=BLOCKS {
        let at = (street as f32 - BLOCKS as f32 * 0.5) * BLOCK_SPACING;
        for k in 0..LAMPS {
            let along = -half + (k as f32 + 0.5) * 2.0 * half / LAMPS as f32;
            let t = ((street * 5 + k * 3) % 7) as f32 / 6.0;
            for p in [
                Vec3::new(at, LAMP_HEIGHT, along),
                Vec3::new(along, LAMP_HEIGHT, at),
            ] {
                lights.push(Light::Sphere {
                    position: p.as_dvec3(),
                    radius: 0.12,
                    intensity: lamp_color(t) * 220.0,
                    cone: None,
                });
            }
        }
    }
    for i in 0..BLOCKS {
        for j in 0..BLOCKS {
            let c = block(i, j);
            let h = block_height(i, j);
            let s = BLOCK_SIZE * 0.5 + 0.6;
            for (k, d) in [Vec3::X, Vec3::Z, Vec3::NEG_X, Vec3::NEG_Z]
                .into_iter()
                .enumerate()
            {
                if (i + j + k as i32) % 2 == 0 {
                    lights.push(Light::Sphere {
                        position: (c + d * s + Vec3::Y * 0.3).as_dvec3(),
                        radius: 0.08,
                        intensity: lamp_color(0.3) * 900.0,
                        cone: Some(Cone {
                            direction: (Vec3::Y * 2.0 - d).normalize(),
                            inner: 0.25,
                            outer: 0.4,
                        }),
                    });
                }
                let up = Vec3::Y;
                let across = up.cross(d);
                let face = c + d * (BLOCK_SIZE * 0.5 + 0.02);
                for floor in 1..=((h / 2.5) as i32).min(2) {
                    lights.push(Light::Rect {
                        center: (face + up * (floor as f32 * 2.3)).as_dvec3(),
                        edge_x: across * 1.4,
                        edge_y: up * 0.9,
                        luminance: lamp_color(((i + 2 * j + floor) % 3) as f32 / 2.0) * 35.0,
                    });
                }
            }
        }
    }
    lights
}

/// The moon and two faint skies of colored light.
fn directional_lights() -> [DirectionalLight; 3] {
    [
        DirectionalLight {
            direction_to_light: Vec3::new(0.3, 0.6, -0.74).normalize(),
            illuminance: Vec3::new(0.25, 0.26, 0.3),
            angular_radius: 0.0045,
            disc: DiscProfile::Uniform,
        },
        DirectionalLight {
            direction_to_light: Vec3::new(-0.5, 0.3, 0.8).normalize(),
            illuminance: Vec3::new(0.02, 0.03, 0.08),
            angular_radius: 0.3,
            disc: DiscProfile::Uniform,
        },
        DirectionalLight {
            direction_to_light: Vec3::new(0.8, 0.2, 0.4).normalize(),
            illuminance: Vec3::new(0.06, 0.03, 0.02),
            angular_radius: 0.3,
            disc: DiscProfile::Uniform,
        },
    ]
}

/// Runs the example; headless runs and runs with `--out` return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new("orr many-lights", 1280, 720));
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

    let power = options.has("--power");
    let lighting = ViewLighting {
        local_light_share: if power { 0.0 } else { 0.5 },
        ..ViewLighting::default()
    };
    let local = lights();
    let directional = directional_lights();
    let scene = renderer
        .create_scene(
            &SceneDesc {
                instance_capacity: 4,
                group_capacity: 2,
                light_capacity: (local.len() + directional.len()) as u32,
                material_set_capacity: 4,
                emissive_capacity: 16,
                part_capacity: 0,
                ptlas_partitions: 4,
                global_instances: 4,
                ..SceneDesc::default()
            },
            &SceneLightingSettings {
                feedback_share: if power { 0.0 } else { 0.5 },
                ..SceneLightingSettings::default()
            },
        )
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    let model = renderer
        .model(mount, AssetId::from_name(TOWN))
        .expect("town model");

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
            count: 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .expect("transform group");
    w.transforms(group, 0, 1).expect("transforms")[0] = Affine3::IDENTITY;
    w.add_instances(&InstancesDesc {
        model,
        copies: 1,
        transforms: group,
        first: 0,
        pose: None,
        material_set: None,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 1000.0,
        animation_range: 0.0,
        user_tag: 0,
    })
    .expect("instances");
    for (i, light) in directional
        .into_iter()
        .map(Light::Directional)
        .chain(local)
        .enumerate()
    {
        w.set_light(LightId(i as u32), Some(&light)).expect("light");
    }
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
            "many-lights target",
        )
        .expect("headless target")
    });
    let capture = options.headless || options.out.is_some();
    let still = options.headless || options.has("--still");
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
        let angle = if still {
            STILL_ANGLE
        } else {
            STILL_ANGLE + t * ORBIT_RATE
        };
        let eye = TARGET + Vec3::new(angle.sin() as f32 * 16.0, 4.5, angle.cos() as f32 * 16.0);

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: display.as_mut(),
            })
            .expect("begin frame");
        let id = f.id();
        let (extent, target) = match (&headless_target, f.display()) {
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
        if let Some(t) = headless_target.as_ref().filter(|_| i == 0) {
            f.list().expect("frame list").discard(t);
        }
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
                rotation: view::look_at_quat(eye, TARGET, Vec3::Y).inverse(),
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
                target,
            },
            &lighting,
            &output,
            &mut ViewHooks::default(),
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
    drop(headless_target);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}

/// The ground, the buildings, the lamp posts and the signs, cooked into an in-memory pack.
fn cook() -> Arc<[u8]> {
    let ground = Material {
        base_color: [0.35, 0.34, 0.32],
        specular_roughness: 0.55,
        ..Material::default()
    };
    let wall = Material {
        base_color: [0.6, 0.55, 0.5],
        specular_roughness: 0.8,
        ..Material::default()
    };
    let post = Material {
        base_color: [0.1, 0.1, 0.11],
        metalness: 1.0,
        specular_roughness: 0.35,
        ..Material::default()
    };
    let sign = |color: [f32; 3]| Material {
        base_color: [0.05, 0.05, 0.05],
        emission_color: color,
        emission_luminance: 60.0,
        ..Material::default()
    };
    let signs = [
        sign([1.0, 0.2, 0.3]),
        sign([0.2, 0.6, 1.0]),
        sign([0.3, 1.0, 0.4]),
        sign([1.0, 0.7, 0.1]),
    ];

    let mut town = MeshBuilder::default();
    let h = 60.0;
    town.quad(
        0,
        [
            Vec3::new(-h, 0.0, h),
            Vec3::new(h, 0.0, h),
            Vec3::new(h, 0.0, -h),
            Vec3::new(-h, 0.0, -h),
        ],
        32,
    );
    for i in 0..BLOCKS {
        for j in 0..BLOCKS {
            let c = block(i, j);
            let s = Vec3::new(BLOCK_SIZE * 0.5, 0.0, BLOCK_SIZE * 0.5);
            town.cuboid(1, c - s, c + s + Vec3::Y * block_height(i, j), 4);
            let d = [Vec3::X, Vec3::Z, Vec3::NEG_X, Vec3::NEG_Z][((i + j) % 4) as usize];
            let across = Vec3::Y.cross(d) * 0.9;
            let face = c + d * (BLOCK_SIZE * 0.5 + 0.03) + Vec3::Y * 1.2;
            town.quad(
                3 + ((i * BLOCKS + j) as usize % SIGNS),
                [
                    face - across,
                    face + across,
                    face + across + Vec3::Y * 0.5,
                    face - across + Vec3::Y * 0.5,
                ],
                1,
            );
        }
    }
    for light in lights() {
        if let Light::Sphere {
            position,
            cone: None,
            ..
        } = light
        {
            let p = position.as_vec3();
            town.cuboid(
                2,
                Vec3::new(p.x - 0.05, 0.0, p.z - 0.05),
                Vec3::new(p.x + 0.05, p.y - 0.15, p.z + 0.05),
                1,
            );
        }
    }
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: TOWN.into(),
                source: Source::Mesh {
                    mesh: town.finish(),
                    materials: [vec![ground, wall, post], signs.to_vec()].concat(),
                    settings: MeshSettings::default(),
                },
            }],
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0)
}
