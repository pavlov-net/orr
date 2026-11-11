//! A planet's sky and medium over a plain, under two suns and a moon they light. Towers from
//! 200 m to 30 km fade into the air. Every path traced shadow, sky and haze comes from
//! `orr_atmosphere`, whose `AtmosphereSky` and `AtmosphereMedium` are linked into orr's kernels
//! as its environment and medium; both suns and the moon go into the sky-view LUT. The suns move
//! along their arcs and the moon's phase follows.
//!
//! `sky [--still] [--fog] [--frames N] [--headless] [--out file.png]`. `--fog` adds a layer of
//! height fog. Headless runs keep the lights and the camera still, render 480 x 270 for 120
//! frames unless told otherwise and write the last frame as a PNG.

use std::sync::Arc;
use std::time::{Duration, Instant};

use orr::orr_gpu::{
    Device, Format, Hazards, MemoryKind, QueueKind, Stages, Texture, TextureDesc, TextureIndex,
    TextureMemory, TextureRegion, TextureUsage, TimelinePoint, ViewDesc as TextureView,
};
use orr::orr_pack::material::Material;
use orr::orr_pack::{AssetId, Target};
use orr::{
    DirectionalLight, Display, DisplayRequest, FrameDesc, FrameTime, GroupWriter, HookPass,
    InstanceFlags, InstancesDesc, Lens, Light, LightId, LitView, PackSource, Projection, Renderer,
    RendererDesc, SceneDesc, SceneLightingSettings, ScreenshotRequest, TextureFilter,
    TransformGroupDesc, ViewHooks, ViewLighting, ViewOutput, ViewTarget,
};
use orr_atmosphere::presets::{LUNAR_ANGULAR_RADIUS, lunar_albedo, lunar_disc, solar_disc};
use orr_atmosphere::{Atmosphere, AtmosphereState, HeightFog};
use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshSettings, Source};
use orr_examples::average::Average;
use orr_examples::linked::KERNELS;
use orr_examples::mesh::MeshBuilder;
use orr_examples::window::Window;
use orr_examples::{Capture, Options, Workers, assert_clean, device, fixed_time, mount_at};
use orr_math::camera::rh::view;
use orr_math::{Affine3, DVec3, Vec3};

const NEAR: &str = "near";
const FAR: &str = "far";

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
/// The frame id the scene is created in, after the pack has mounted.
const FIRST_FRAME: u64 = 500;
/// Frames a still camera renders before it starts averaging, while exposure settles.
const ACCUMULATE_AFTER: u64 = 30;
const FOV_Y: f32 = 0.8;
const EYE: Vec3 = Vec3::new(0.0, 2.0, 12.0);
const TARGET: Vec3 = Vec3::new(0.0, 5.0, 0.0);

const SUN: LightId = LightId(0);
const SECOND_SUN: LightId = LightId(1);
const MOON: LightId = LightId(2);
/// Degrees per second along the suns' arcs.
const ARC_RATE: f64 = 0.5;
/// The moon is drawn this many times its size, with its radiance kept, so it reads at any
/// resolution.
const MOON_SCALE: f32 = 6.0;

/// Half the side of the ground, m.
const GROUND: f32 = 60_000.0;
/// The towers' distance, m, across the view from left to right.
const TOWERS: [f32; 7] = [200.0, 600.0, 1_500.0, 4_000.0, 9_000.0, 18_000.0, 30_000.0];

/// Unit direction at an elevation and an azimuth from the view's -z toward +x, in degrees.
fn dir(elevation: f64, azimuth: f64) -> Vec3 {
    let (e, a) = (elevation.to_radians(), azimuth.to_radians());
    Vec3::new(
        (e.cos() * a.sin()) as f32,
        e.sin() as f32,
        (-e.cos() * a.cos()) as f32,
    )
}

/// The lights `degrees` along their arcs: a yellow sun, a dim red sun with a larger disc, and
/// the moon, lit by both.
fn lights(degrees: f64, moon_albedo: TextureIndex) -> [DirectionalLight; 3] {
    let sun = solar_disc(dir(3.0 + 0.4 * degrees, -30.0 + degrees));
    let base = solar_disc(dir(8.0 - 0.1 * degrees, -8.0 + 0.5 * degrees));
    let second = DirectionalLight {
        illuminance: base.illuminance * Vec3::new(0.30, 0.13, 0.06),
        angular_radius: 3.0 * base.angular_radius,
        ..base
    };
    let moon = lunar_disc(
        dir(28.0, 30.0),
        moon_albedo,
        Vec3::Y,
        &[(SUN, &sun), (SECOND_SUN, &second)],
    );
    let moon = DirectionalLight {
        angular_radius: MOON_SCALE * LUNAR_ANGULAR_RADIUS,
        illuminance: moon.illuminance * MOON_SCALE * MOON_SCALE,
        ..moon
    };
    [sun, second, moon]
}

/// A small texture of the lunar disc's mean albedo.
fn moon_texture(gpu: &Device) -> (Texture, TextureIndex) {
    let desc = TextureDesc::d2(Format::Rgba32Float, 4, 4, TextureUsage::SAMPLED);
    let texture = gpu
        .create_texture(&desc, TextureMemory::Dedicated, "moon albedo")
        .expect("moon texture");
    let mut staging = gpu
        .gpu_malloc::<[f32; 4]>(MemoryKind::Upload, 16, "moon albedo")
        .expect("moon staging");
    let [r, g, b] = lunar_albedo().map(|c| c as f32);
    staging.as_mut_slice().expect("mapped").fill([r, g, b, 1.0]);
    let render = gpu.queue(QueueKind::Render).expect("render queue");
    let mut list = render.command_list().expect("command list");
    list.discard(&texture);
    list.copy_to_texture(
        &texture,
        &TextureRegion::whole_mip(&desc, 0),
        staging.gpu().cast(),
    );
    list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
    let point = render.submit(vec![list], &[]).expect("submit");
    render.wait(point, Duration::from_secs(10)).expect("wait");
    let slot = gpu.alloc_image_slot().expect("moon slot");
    let index = gpu.write_sampled(slot, &texture, &TextureView::default());
    (texture, index)
}

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("sky");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// Runs the example; headless runs and runs with `--out` return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new("orr sky", 1920, 1080));
    let desc = RendererDesc {
        kernels: &KERNELS,
        ..RendererDesc::default()
    };
    let mut display = window.as_ref().map(|w| {
        Display::new(
            gpu,
            w.window_handle(),
            w.display_handle(),
            &DisplayRequest::default(),
            desc.frames_in_flight,
        )
        .expect("display")
    });
    let mut renderer = Renderer::new(gpu, desc).expect("renderer");
    let _workers = Workers::machine(renderer.loading());
    let (mount, mut frame) = mount_at(&mut renderer, PackSource::Memory(cook()), FIRST_FRAME);

    let atmosphere = Atmosphere {
        height_fog: options.has("--fog").then_some(HeightFog {
            density: 1e-4,
            falloff_height: 300.0,
            base_height: 0.0,
            albedo: Vec3::splat(0.9),
            droplet_diameter: 10.0,
        }),
        ..Atmosphere::earth(DVec3::ZERO, Vec3::Y)
    };
    let mut sky = AtmosphereState::new(gpu, &atmosphere, 0).expect("atmosphere");
    sky.set_lights(&[SUN, SECOND_SUN, MOON]);
    let (_moon_texture, moon_albedo) = moon_texture(gpu);

    let scene = renderer
        .create_scene(
            &SceneDesc {
                instance_capacity: 4,
                group_capacity: 2,
                light_capacity: 3,
                material_set_capacity: 4,
                emissive_capacity: 0,
                global_instances: 4,
                ..SceneDesc::default()
            },
            &SceneLightingSettings::default(),
        )
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    options.wait_for_paths(&mut renderer);
    let near = renderer
        .model(mount, AssetId::from_name(NEAR))
        .expect("near model");
    let far = renderer
        .model(mount, AssetId::from_name(FAR))
        .expect("far model");

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
    w.transforms(group, 0, 1).expect("transforms")[0] = Affine3::IDENTITY.into();
    for (model, tag) in [(near, 0), (far, 1)] {
        w.add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: group,
            first: 0,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 200_000.0,
            animation_range: 0.0,
            user_tag: tag,
        })
        .expect("instances");
    }
    w.set_environment(sky.data());
    w.set_medium(sky.data());
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
            "sky target",
        )
        .expect("headless target")
    });
    let capture = options.headless || options.out.is_some();
    let still = options.headless || options.has("--still");
    let mut average = Average::new(&mut renderer, gpu, desc.frames_in_flight, ACCUMULATE_AFTER);
    let frames = options
        .frames
        .or_else(|| options.headless.then_some(HEADLESS_FRAMES));
    let started = Instant::now();
    let mut previous = 0.0;
    let mut last: Option<(u64, TimelinePoint)> = None;
    for i in 0.. {
        if frames.is_some_and(|n| i >= n) {
            break;
        }
        if let Some(w) = window.as_mut()
            && !w.pump_showing_paths(&renderer)
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
        let degrees = if still { 0.0 } else { t * ARC_RATE };

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
        {
            let mut w = f.scene(scene).expect("scene writer");
            w.set_focus(EYE.as_dvec3());
            let lit = lights(degrees, moon_albedo);
            for (id, light) in [SUN, SECOND_SUN, MOON].into_iter().zip(lit) {
                w.set_light(id, Some(&Light::Directional(light)))
                    .expect("light");
            }
        }
        if i == 0 {
            sky.skip_fades();
        }
        let view = LitView {
            scene,
            history,
            position: EYE.as_dvec3(),
            rotation: view::look_at_quat(EYE, TARGET, Vec3::Y).inverse(),
            projection: Projection {
                fov_y: FOV_Y,
                near: 0.05,
            },
            lens: Lens::default(),
            camera_cut: false,
            not_rendering_scene: false,
            screenshot: (capture && final_frame)
                .then_some(ScreenshotRequest::Display { include_ui: false }),
            texture_filter: TextureFilter::Hardware,
            target,
        };
        let output = ViewOutput::default();
        f.prepare_view(&view, &ViewLighting::default(), &output)
            .expect("prepare view");
        f.update_scenes(&mut |c| sky.record(c, &[]).expect("sky passes"))
            .expect("update scenes");
        let lit = f
            .view_lit(&view, &ViewLighting::default(), false)
            .expect("path pipelines");
        let mut average = average.hook(&f, extent, lit);
        let mut hooks = ViewHooks {
            scene_referred: still.then_some(&mut average as &mut dyn FnMut(&mut HookPass<'_, '_>)),
            ..ViewHooks::default()
        };
        f.render_view(&view, &ViewLighting::default(), &output, &mut hooks)
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
    drop(display);
    drop(sky);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}

/// Two models, so each has the position precision of its own size: the objects by the camera,
/// and the ground with the towers.
fn cook() -> Arc<[u8]> {
    let stone = Material {
        base_color: [0.55, 0.52, 0.47],
        specular_roughness: 0.7,
        ..Material::default()
    };
    let metal = Material {
        base_color: [0.9, 0.86, 0.8],
        metalness: 1.0,
        specular_roughness: 0.2,
        ..Material::default()
    };
    let ground = Material {
        base_color: [0.16, 0.2, 0.12],
        specular_roughness: 0.9,
        ..Material::default()
    };
    let tower = Material {
        base_color: [0.5, 0.5, 0.52],
        specular_roughness: 0.8,
        ..Material::default()
    };

    let mut close = MeshBuilder::default();
    close.sphere(0, Vec3::new(-4.0, 1.0, 0.0), 1.0, 24);
    close.sphere(1, Vec3::new(1.5, 0.6, 3.0), 0.6, 24);
    close.cuboid(0, Vec3::new(5.0, 0.0, 2.0), Vec3::new(6.5, 1.5, 3.5), 4);

    let mut plain = MeshBuilder::default();
    plain.quad(
        0,
        [
            Vec3::new(-GROUND, 0.0, GROUND),
            Vec3::new(GROUND, 0.0, GROUND),
            Vec3::new(GROUND, 0.0, -GROUND),
            Vec3::new(-GROUND, 0.0, -GROUND),
        ],
        32,
    );
    for (i, distance) in TOWERS.into_iter().enumerate() {
        // Across the view, with each tower's size in proportion to its distance.
        let x = (i as f32 - 3.0) * 0.14 * distance;
        let half = 0.015 * distance;
        let height = 0.08 * distance;
        plain.cuboid(
            1,
            Vec3::new(x - half, 0.0, EYE.z - distance - half),
            Vec3::new(x + half, height, EYE.z - distance + half),
            2,
        );
    }

    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: NEAR.into(),
                    source: Source::Mesh {
                        mesh: close.finish(),
                        materials: vec![stone, metal],
                        settings: MeshSettings::default(),
                    },
                },
                AssetSpec {
                    name: FAR.into(),
                    source: Source::Mesh {
                        mesh: plain.finish(),
                        materials: vec![ground, tower],
                        settings: MeshSettings::default(),
                    },
                },
            ],
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0)
}
