//! Crytek's Sponza atrium as the Khronos glTF Sample Assets' `Sponza` (Cryengine Limited License
//! Agreement, <https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza>), with
//! Alexandre Pestana's PBR textures: 262 k triangles in 103 primitives, 25 materials, three of
//! them alpha masked, and 69 textures. The scene cooks at startup as one model, with no cook
//! cache, and is lit by a sun through its open roof and `orr_atmosphere`'s sky, linked into
//! orr's kernels as its environment and medium. Frames average in a scene-referred hook while
//! the camera and the sun hold still. Windowed runs denoise with DLSS ray reconstruction where
//! NGX is available, and an overlay shows frame and GPU pass times on Windows.
//!
//! `sponza [--assets DIR] [--fabric] [--camera NAME] [--view x,y,z,yaw,pitch,fov]
//! [--sun ELEVATION,AZIMUTH] [--reference] [--no-vsync] [--frames N] [--headless]
//! [--out file.png]`: `DIR` holds the sample's `glTF` directory; without it the files download
//! from the sample's repository into a temporary directory, removed once they cook. `--fabric`
//! cooks the cloth as fabric, which the sample's opaque materials leave out: translucent and with
//! sheen, both colored by the cloth's own base color, and seen from both sides. `NAME` is the
//! view to start from, `--view` one logged by C. The sun's angles are in degrees, its azimuth
//! from -Z toward +X. `--reference` renders the accumulation reference at 64 bounces and 64
//! dielectric vertices. Keys 1 to 3 jump between the views, WASD moves, Q and E sink and rise,
//! shift moves faster and dragging with the right mouse button looks around. The arrow keys move
//! the sun. C logs the view as a `--view` value and appends it to `sponza-views.txt` beside the
//! executable. Headless runs render 960 x 540, for 120 frames without `--frames`, and write the
//! last frame as a PNG; a windowed run needs `--frames` to write one.

use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use orr::orr_gpu::{Format, QueueKind, TextureDesc, TextureMemory, TextureUsage, TimelinePoint};
use orr::orr_pack::{AssetId, Target};
use orr::{
    AutoExposure, Display, DisplayRequest, Exposure, FrameDesc, FrameTime, GroupWriter, HookPass,
    InstanceFlags, InstancesDesc, Lens, Light, LitView, LocalExposure, MeterWeights, PackSource,
    Projection, ReferenceSettings, Renderer, RendererDesc, SceneDesc, SceneLightingSettings,
    ScreenshotRequest, TextureFilter, TransformGroupDesc, UiPass, ViewHooks, ViewLighting,
    ViewOutput, ViewTarget,
};
use orr_atmosphere::presets::solar_disc;
use orr_atmosphere::{Atmosphere, AtmosphereState};
use orr_cook::{ArchiveSpec, AssetSpec, GltfSettings, Manifest, Source};
use orr_examples::average::Average;
use orr_examples::fetch::{TempDir, download_all};
use orr_examples::fly::{Camera, note_view};
use orr_examples::linked::KERNELS;
use orr_examples::overlay::Overlay;
use orr_examples::perf::Perf;
use orr_examples::window::{KeyCode, Window};
use orr_examples::{
    Capture, Options, Workers, assert_clean, device, enable_reconstruction, fixed_time,
    init_logging, mount_at,
};
use orr_math::{Affine3, DVec3, Vec3};
use serde_json::{Value, json};

const HEADLESS_EXTENT: [u32; 2] = [960, 540];
const HEADLESS_FRAMES: u64 = 120;
/// The frame id the scene is created in, after the pack has mounted.
const FIRST_FRAME: u64 = 500;
/// Frames before averaging starts, while exposure settles.
const ACCUMULATE_AFTER: u64 = 30;
const SPONZA: &str = "sponza";
/// The sample's `glTF` directory at a fixed commit, so the scene does not change under the run.
const FILES_URL: &str = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/\
     4995a638c96dbc94376d3d6164e70fee3ac7f7a2/Models/Sponza/glTF/";
const GLTF_FILE: &str = "Sponza.gltf";
/// `--fabric`'s patched glTF, beside the files the sample names.
const FABRIC_FILE: &str = "Sponza-fabric.gltf";
/// The cloth's materials in the sample's glTF, which names none: the three swags along the
/// upper gallery and the three curtains below them.
const CLOTH: RangeInclusive<usize> = 14..=19;
const DIFFUSE_TRANSMISSION: &str = "KHR_materials_diffuse_transmission";
const SHEEN: &str = "KHR_materials_sheen";
/// The cloth's `diffuseTransmissionFactor`.
const CLOTH_TRANSMISSION: f64 = 0.5;
/// The cloth's `sheenRoughnessFactor`.
const CLOTH_SHEEN_ROUGHNESS: f64 = 0.5;
/// The scene's lowest vertex, m, where the atmosphere's ground plane sits.
const FLOOR: f64 = -1.012;
/// The sun's default elevation and azimuth, degrees: down the court's long axis from +X,
/// so the sun reaches the narrow court's floor.
const SUN_ANGLES: [f32; 2] = [70.0, 70.0];
/// Degrees per second the arrow keys move the sun.
const SUN_RATE: f32 = 30.0;
/// The view's histogram, with a mid-gray card under the sun weighed in, so the sunlit court
/// holds while most of the view is in shade.
const METER: MeterWeights = MeterWeights {
    histogram: 0.75,
    dome: 0.0,
    directional: 0.25,
};
/// The sunlit floor sits about 6 stops above the shaded arcades, more than one exposure holds;
/// local exposure brings them halfway together, as an eye adapting to each would.
const LOCAL_EXPOSURE: LocalExposure = LocalExposure { strength: 0.5 };
const WALK_SPEED: f32 = 2.0;
const NEAR: f32 = 0.05;
const REFERENCE_DEPTH: u32 = 64;
/// The keys that jump between the views, in `views`' order.
const VIEW_KEYS: [KeyCode; 3] = [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3];

/// The named views.
fn views() -> [Camera; 3] {
    [
        Camera::looking(
            "atrium",
            Vec3::new(6.0, 1.0, -0.4),
            Vec3::new(-10.0, 2.1, -0.4),
            60.0,
        ),
        Camera::looking(
            "gallery",
            Vec3::new(9.5, 5.4, 3.9),
            Vec3::new(-9.0, 4.9, 3.6),
            60.0,
        ),
        Camera::looking(
            "roof",
            Vec3::new(-3.0, 0.4, 0.0),
            Vec3::new(2.0, 10.0, 1.0),
            70.0,
        ),
    ]
}

fn main() {
    init_logging();
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("sponza");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// Unit direction to a light at `elevation` and `azimuth` degrees, the azimuth from -Z toward +X.
fn direction([elevation, azimuth]: [f32; 2]) -> Vec3 {
    let (e, a) = (elevation.to_radians(), azimuth.to_radians());
    Vec3::new(e.cos() * a.sin(), e.sin(), -e.cos() * a.cos())
}

/// `--sun`'s value: `elevation,azimuth` in degrees.
fn parse_sun(value: &str) -> [f32; 2] {
    let v: Vec<f32> = value
        .split(',')
        .map(|x| x.trim().parse().expect("a --sun angle"))
        .collect();
    v.try_into().expect("--sun takes elevation,azimuth")
}

/// The sun moved by the held arrow keys over `dt` seconds.
fn steer_sun(window: &Window, [elevation, azimuth]: [f32; 2], dt: f32) -> [f32; 2] {
    let step = SUN_RATE * dt;
    [
        (elevation + step * window.axis(KeyCode::ArrowUp, KeyCode::ArrowDown)).clamp(-90.0, 90.0),
        (azimuth + step * window.axis(KeyCode::ArrowRight, KeyCode::ArrowLeft)).rem_euclid(360.0),
    ]
}

fn read_json(path: &Path) -> Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("glTF JSON")
}

/// The files `gltf` names: its buffers and images.
fn gltf_files(gltf: &Value) -> Vec<String> {
    ["buffers", "images"]
        .iter()
        .flat_map(|key| gltf[key].as_array().expect("glTF buffers and images"))
        .map(|v| v["uri"].as_str().expect("a file uri").to_owned())
        .collect()
}

/// Makes the cloth translucent and gives it sheen, each from the cloth's own base color, and
/// draws both its faces.
fn fabric(gltf: &mut Value) {
    for m in CLOTH {
        let material = &mut gltf["materials"][m];
        let pbr = &material["pbrMetallicRoughness"];
        let texture = pbr["baseColorTexture"].clone();
        let factor = pbr["baseColorFactor"]
            .as_array()
            .expect("a base color factor")[..3]
            .to_vec();
        material["doubleSided"] = true.into();
        material["extensions"] = json!({
            DIFFUSE_TRANSMISSION: {
                "diffuseTransmissionFactor": CLOTH_TRANSMISSION,
                "diffuseTransmissionColorFactor": factor,
                "diffuseTransmissionColorTexture": texture,
            },
            SHEEN: {
                "sheenColorFactor": factor,
                "sheenColorTexture": texture,
                "sheenRoughnessFactor": CLOTH_SHEEN_ROUGHNESS,
            },
        });
    }
    let used = &mut gltf["extensionsUsed"];
    if used.is_null() {
        *used = json!([]);
    }
    let used = used.as_array_mut().expect("extensionsUsed");
    used.extend([DIFFUSE_TRANSMISSION, SHEEN].map(Value::from));
}

/// The scene's glTF, from `--assets` or downloaded into a temporary directory. With `--fabric`
/// it is a patched copy in a temporary directory, beside copies of the files it names.
fn fetch(options: &Options) -> (PathBuf, Option<TempDir>) {
    let assets = options.value("--assets").map(Path::new);
    let fabric_on = options.has("--fabric");
    if let Some(dir) = assets.filter(|_| !fabric_on) {
        return (dir.join(GLTF_FILE), None);
    }
    let temp = TempDir::new("orr-sponza");
    let gltf = temp.0.join(GLTF_FILE);
    match assets {
        Some(dir) => {
            let names = gltf_files(&read_json(&dir.join(GLTF_FILE)));
            for name in names.iter().map(String::as_str).chain([GLTF_FILE]) {
                let (from, to) = (dir.join(name), temp.0.join(name));
                std::fs::copy(&from, &to).unwrap_or_else(|e| panic!("{}: {e}", from.display()));
            }
        }
        None => {
            download_all(FILES_URL, &[GLTF_FILE], &temp.0);
            download_all(FILES_URL, &gltf_files(&read_json(&gltf)), &temp.0);
        }
    }
    if !fabric_on {
        return (gltf, Some(temp));
    }
    let mut json = read_json(&gltf);
    fabric(&mut json);
    let patched = temp.0.join(FABRIC_FILE);
    std::fs::write(&patched, serde_json::to_vec(&json).expect("glTF JSON"))
        .unwrap_or_else(|e| panic!("{}: {e}", patched.display()));
    (patched, Some(temp))
}

/// Cooks the scene into a pack in memory.
fn cook(gltf: PathBuf) -> Arc<[u8]> {
    let started = Instant::now();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: SPONZA.into(),
                source: Source::Gltf {
                    path: gltf,
                    settings: GltfSettings::default(),
                },
            }],
        )],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0);
    log::info!("cooked in {:.1} s", started.elapsed().as_secs_f64());
    pack
}

/// Runs the example; headless runs and windowed runs with `--out` and `--frames` return their
/// last frame.
fn run(options: &Options) -> Option<Capture> {
    let views = views();
    let mut camera = match (options.value("--view"), options.value("--camera")) {
        (Some(view), _) => Camera::parse(view),
        (None, wanted) => {
            let wanted = wanted.unwrap_or(views[0].name);
            *views.iter().find(|c| c.name == wanted).unwrap_or_else(|| {
                let names: Vec<&str> = views.iter().map(|c| c.name).collect();
                panic!("no view {wanted}; the example has {names:?}")
            })
        }
    };
    let mut sun = options.value("--sun").map_or(SUN_ANGLES, parse_sun);
    let reference = options.has("--reference");
    let title = |camera: &Camera| format!("orr sponza: {}", camera.name);
    let (gltf, downloaded) = fetch(options);

    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new(&title(&camera), 1920, 1080));
    let desc = RendererDesc {
        kernels: &KERNELS,
        ..RendererDesc::default()
    };
    let mut display = window.as_ref().map(|w| {
        Display::new(
            gpu,
            w.window_handle(),
            w.display_handle(),
            &DisplayRequest {
                vsync: !options.has("--no-vsync"),
                ..DisplayRequest::default()
            },
            desc.frames_in_flight,
        )
        .expect("display")
    });
    let mut renderer = Renderer::new(gpu, desc).expect("renderer");
    if !options.headless && !reference {
        enable_reconstruction(&mut renderer);
    }
    let _workers = Workers::machine(renderer.loading());
    let (mount, mut frame) = mount_at(&mut renderer, PackSource::Memory(cook(gltf)), FIRST_FRAME);
    drop(downloaded);

    let mut sky = AtmosphereState::new(
        gpu,
        &Atmosphere::earth(DVec3::new(0.0, FLOOR, 0.0), Vec3::Y),
        0,
    )
    .expect("atmosphere");
    // The sun is in the sky from the first frame that records it.
    sky.skip_fades();
    let scene = renderer
        .create_scene(
            &SceneDesc {
                global_instances: 1,
                ..SceneDesc::default()
            },
            &SceneLightingSettings::default(),
        )
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    options.wait_for_paths(&mut renderer);
    let model = renderer
        .model(mount, AssetId::from_name(SPONZA))
        .expect("the scene's model");

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
    let sun_light = w
        .add_light(&Light::Directional(solar_disc(direction(sun))))
        .expect("sun");
    sky.set_lights(&[sun_light]);
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
            "sponza target",
        )
        .expect("headless target")
    });
    let lighting = if reference {
        ViewLighting {
            max_bounces: REFERENCE_DEPTH,
            dielectric_depth: REFERENCE_DEPTH,
            reference: Some(ReferenceSettings::default()),
            ..ViewLighting::default()
        }
    } else {
        ViewLighting::default()
    };
    let capture = options.headless || options.out.is_some();
    let mut average = Average::new(&mut renderer, gpu, desc.frames_in_flight, ACCUMULATE_AFTER);
    let mut overlay = Overlay::new(desc.frames_in_flight);
    let mut perf = Perf::default();
    let frames = options
        .frames
        .or_else(|| options.headless.then_some(HEADLESS_FRAMES));
    let started = Instant::now();
    let mut previous = 0.0;
    let mut last: Option<(u64, TimelinePoint)> = None;
    let mut camera_cut = false;
    for i in 0.. {
        if frames.is_some_and(|n| i >= n) {
            break;
        }
        // Headless runs step a fixed clock, so they repeat exactly.
        let t = if options.headless {
            fixed_time(i).absolute
        } else {
            started.elapsed().as_secs_f64()
        };
        let dt = t - previous;
        previous = t;
        if let Some(w) = window.as_mut() {
            if !w.pump_showing_paths(&renderer) {
                break;
            }
            let (was, sun_was) = (camera, sun);
            for key in w.take_keys() {
                let picked = VIEW_KEYS.iter().position(|&k| k == key);
                if let Some(&picked) = picked.and_then(|p| views.get(p)) {
                    camera = picked;
                    camera_cut = true;
                    w.set_title(&title(&camera));
                }
                if key == KeyCode::KeyC {
                    note_view("sponza", &camera);
                }
            }
            camera.fly(w, WALK_SPEED, dt as f32);
            sun = steer_sun(w, sun, dt as f32);
            if camera_cut || camera != was || sun != sun_was {
                average.restart(ACCUMULATE_AFTER);
            }
        }
        let final_frame = frames.is_some_and(|n| i + 1 == n);
        // A headless run's clock steps by a fixed amount, so its readout's GPU pass times alone
        // measure anything.
        if final_frame && options.headless {
            log::info!("last readout of GPU pass times:\n{}", perf.figures());
        }
        let time = FrameTime {
            absolute: fixed_time(frame + 1).absolute + t,
            delta: dt as f32,
        };
        perf.add(dt, renderer.stats());

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: display.as_mut(),
            })
            .expect("begin frame");
        let id = f.id();
        let telemetry = f.exposure_telemetry(history);
        let (extent, target) = match (&headless_target, f.display()) {
            (Some(t), _) => (HEADLESS_EXTENT, ViewTarget::Texture(t)),
            (None, Some((resolved, extent))) => {
                let label = format!("{}  sun {:.0}, {:.0} deg", camera.name, sun[0], sun[1]);
                let text = perf.readout(
                    &label,
                    extent,
                    &resolved,
                    telemetry.as_ref(),
                    average.frames(),
                );
                overlay.layout(extent, time, &text);
                (extent, ViewTarget::Display)
            }
            (None, None) => {
                last = Some((id, f.end().expect("end frame")));
                continue;
            }
        };
        let position = camera.position.as_dvec3();
        {
            let mut w = f.scene(scene).expect("scene writer");
            w.set_focus(position);
            w.set_light(sun_light, &Light::Directional(solar_disc(direction(sun))))
                .expect("sun");
        }
        let view = LitView {
            scene,
            history,
            position,
            rotation: camera.rotation(),
            projection: Projection {
                fov_y: camera.fov_y,
                near: NEAR,
            },
            lens: Lens::default(),
            camera_cut,
            not_rendering_scene: false,
            screenshot: (capture && final_frame)
                .then_some(ScreenshotRequest::Display { include_ui: false }),
            texture_filter: TextureFilter::Hardware,
            target,
        };
        let output = ViewOutput {
            exposure: Exposure::Auto(AutoExposure {
                references: Some(METER),
                ..AutoExposure::default()
            }),
            local_exposure: Some(LOCAL_EXPOSURE),
            ..ViewOutput::default()
        };
        f.prepare_view(&view, &lighting, &output)
            .expect("prepare view");
        f.update_scenes(&mut |c| sky.record(c, &[]).expect("sky passes"))
            .expect("update scenes");
        let lit = f.view_lit(&view, &lighting, false).expect("path pipelines");
        let mut average = average.hook(&f, extent, lit);
        let mut ui = |pass: &mut UiPass<'_, '_>| overlay.paint(pass, id);
        let mut hooks = ViewHooks {
            scene_referred: Some(&mut average as &mut dyn FnMut(&mut HookPass<'_, '_>)),
            ui: Some(&mut ui),
            ..ViewHooks::default()
        };
        let rendered = f
            .render_view(&view, &lighting, &output, &mut hooks)
            .expect("render view");
        // A headless run waits for its path pipelines, so each frame runs the ones made for
        // exactly what its scene and settings ask for.
        assert!(
            !options.headless || rendered.exact_paths,
            "a frame ran path pipelines that hold more than its view asked for"
        );
        camera_cut = false;
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
    // Off Windows the overlay is a stub that holds nothing.
    #[allow(clippy::drop_non_drop)]
    drop(overlay);
    drop(headless_target);
    drop(display);
    drop(sky);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}
