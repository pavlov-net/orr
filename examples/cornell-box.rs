//! Cornell Box 2026 by Jakub Boksansky (CC0, <https://boksajak.github.io/blog/CornellBox2026>),
//! after the Cornell Box of Cornell University: a pendant behind 48 twisted fins, a glazed
//! ceramic block, an iridescent metal ribbon and a solid glass block in a box open towards +Z,
//! under a striped studio environment. The scene cooks at startup as one model, its glass
//! tracing as a dielectric by its clusters' class, with no cook cache. The keys 1 to 7 switch
//! between the glTF's cameras, and each averages its frames in a scene-referred hook. Windowed
//! runs denoise with DLSS ray reconstruction where NGX is available, and an overlay shows frame
//! and GPU pass times on Windows.
//!
//! `cornell-box [--assets DIR] [--camera NAME] [--reference] [--photons N] [--caustic-history N]
//! [--caustic-jitter F] [--no-vsync] [--frames N] [--headless] [--out file.png]`: `DIR` holds
//! the scene's `.gltf` and `.exr`; without it both download from the scene's page into a
//! temporary directory, removed once they cook. `NAME` is the glTF camera to start from.
//! `--reference` renders the accumulation reference at the scene's published 64 bounces. The
//! caustic options set `CausticSettings::photons`, `max_temporal_samples` and `jitter`. Headless
//! runs render 960 x 540, for 120 frames without `--frames`, and write the last frame as a PNG;
//! a windowed run needs `--frames` to write one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use orr::orr_gpu::{Format, QueueKind, TextureDesc, TextureMemory, TextureUsage, TimelinePoint};
use orr::orr_pack::{AssetId, Target};
use orr::{
    CausticSettings, Display, DisplayRequest, ExposureTelemetry, FrameDesc, FrameStats, FrameTime,
    GroupWriter, HookPass, InstanceFlags, InstancesDesc, Lens, LitView, PackSource, Projection,
    ReferenceSettings, Renderer, RendererDesc, ResolvedDisplay, SceneDesc, SceneLightingSettings,
    ScreenshotRequest, TextureFilter, TransformGroupDesc, UiPass, ViewHooks, ViewLighting,
    ViewOutput, ViewTarget,
};
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, Manifest, MeshSettings, Source, TextureSettings, Usage,
};
use orr_examples::average::Average;
use orr_examples::overlay::Overlay;
use orr_examples::window::{KeyCode, Window};
use orr_examples::{
    Capture, Options, Workers, assert_clean, device, enable_dlss, environment_map, fixed_time,
    mount_at,
};
use orr_math::{Affine3, DVec3, Mat3, Mat4, Quat, Vec3};
use serde_json::Value;

const HEADLESS_EXTENT: [u32; 2] = [960, 540];
const HEADLESS_FRAMES: u64 = 120;
/// The frame id the scene is created in, after the pack has mounted.
const FIRST_FRAME: u64 = 500;
/// Frames before averaging starts, while exposure settles.
const ACCUMULATE_AFTER: u64 = 30;
/// Seconds each figure of the readout averages over.
const READOUT_PERIOD: f64 = 0.5;
/// The costliest GPU passes the readout lists.
const READOUT_PASSES: usize = 8;
const DEFAULT_CAMERA: &str = "camera_front";
/// The keys that pick a camera, in the glTF's node order.
const CAMERA_KEYS: [KeyCode; 7] = [
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
];
const BOX: &str = "box";
const STUDIO: &str = "studio";
/// The scene and its environment as the scene's page links them.
const BOX_URL: &str =
    "https://boksajak.github.io/files/cornell_box_2026/Cornell_Box_2026_Slatted_Pendant.gltf";
const STUDIO_URL: &str =
    "https://boksajak.github.io/files/cornell_box_2026/Cornell_Striped_Studio_4K.exr";
/// The default grid would round the 12 mm emitter and the fins.
const POSITION_STEP_LOG2: i8 = -14;
/// The source map is 4096 x 2048, and its stripes show around the box.
const STUDIO_EDGE: u32 = 2048;
/// The scene's renders read the map at `u = 0.5 - atan2(z, x) / 2 pi`, a mirror image of the
/// cook's `u = 0.5 + atan2(x, -z) / 2 pi`.
const STUDIO_FROM_WORLD: Mat3 = Mat3::from_cols(Vec3::NEG_Z, Vec3::Y, Vec3::NEG_X);
/// A ray through the glass block can reflect inside it many times.
const DIELECTRIC_DEPTH: u32 = 16;
const MAX_BOUNCES: u32 = 8;
const REFERENCE_DEPTH: u32 = 64;

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("cornell-box");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// A directory of this run's own, removed when it drops, however the run ends.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0) {
            log::warn!("could not remove {}: {e}", self.0.display());
        }
    }
}

/// Downloads `url` into `dir`, named by its last path segment.
fn download(url: &str, dir: &Path) -> PathBuf {
    let path = dir.join(url.rsplit('/').next().expect("a file name"));
    let started = Instant::now();
    let mut response = ureq::get(url)
        .call()
        .unwrap_or_else(|e| panic!("{url}: {e}"));
    let mut file =
        std::fs::File::create(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let bytes = std::io::copy(&mut response.body_mut().as_reader(), &mut file)
        .unwrap_or_else(|e| panic!("{url}: {e}"));
    log::info!(
        "downloaded {url}, {:.1} MB in {:.1} s",
        bytes as f64 / 1e6,
        started.elapsed().as_secs_f64()
    );
    path
}

/// The first file in `dir` with extension `ext`.
fn asset(dir: &Path, ext: &str) -> PathBuf {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("directory entry").path())
        .find(|p| p.extension().is_some_and(|e| e == ext))
        .unwrap_or_else(|| panic!("no .{ext} in {}", dir.display()))
}

/// `value` as `N` floats, if it is an array of them.
fn floats<const N: usize>(value: &Value) -> Option<[f32; N]> {
    let v: Vec<f32> = value
        .as_array()?
        .iter()
        .map(|x| x.as_f64().map(|x| x as f32))
        .collect::<Option<_>>()?;
    v.try_into().ok()
}

/// The transform of glTF node `node`. The scene's nodes all sit at its root.
fn pose(node: &Value) -> Affine3 {
    if let Some(m) = floats(&node["matrix"]) {
        return Affine3::from_mat4(Mat4::from_cols_array(&m));
    }
    let scale = floats(&node["scale"]).map_or(Vec3::ONE, Vec3::from_array);
    let rotation = floats(&node["rotation"]).map_or(Quat::IDENTITY, Quat::from_array);
    let translation = floats(&node["translation"]).map_or(Vec3::ZERO, Vec3::from_array);
    Affine3::from_scale_rotation_translation(scale, rotation, translation)
}

/// Frame and pass times, averaged over each `READOUT_PERIOD`.
#[derive(Default)]
struct Perf {
    frames: u32,
    seconds: f64,
    worst: f64,
    stats_frame: u64,
    stats_frames: u32,
    cpu_ms: f64,
    passes: BTreeMap<&'static str, f64>,
    figures: String,
}

impl Perf {
    /// Counts a frame that took `delta` seconds, with `stats` the last completed frame's.
    fn add(&mut self, delta: f64, stats: &FrameStats) {
        self.frames += 1;
        self.seconds += delta;
        self.worst = self.worst.max(delta);
        if stats.frame != self.stats_frame {
            self.stats_frame = stats.frame;
            self.stats_frames += 1;
            self.cpu_ms += stats
                .cpu
                .iter()
                .map(|(_, d)| d.as_secs_f64() * 1000.0)
                .sum::<f64>();
            for p in &stats.gpu {
                *self.passes.entry(p.name).or_default() += p.gpu_ms;
            }
        }
        if self.seconds < READOUT_PERIOD {
            return;
        }
        let n = f64::from(self.stats_frames.max(1));
        let mut passes: Vec<(&str, f64)> = self.passes.iter().map(|(k, v)| (*k, v / n)).collect();
        passes.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut figures = format!(
            "{:.1} fps  {:.2} ms  worst {:.2} ms\nGPU passes {:.2} ms  CPU {:.2} ms",
            f64::from(self.frames) / self.seconds,
            self.seconds * 1000.0 / f64::from(self.frames),
            self.worst * 1000.0,
            passes.iter().map(|p| p.1).sum::<f64>(),
            self.cpu_ms / n,
        );
        for (name, ms) in passes.iter().take(READOUT_PASSES) {
            figures += &format!("\n{ms:6.2} ms  {name}");
        }
        *self = Self {
            stats_frame: self.stats_frame,
            figures,
            ..Self::default()
        };
    }

    /// The overlay's text.
    fn readout(
        &self,
        camera: &str,
        [w, h]: [u32; 2],
        resolved: &ResolvedDisplay,
        telemetry: Option<&ExposureTelemetry>,
        averaged: u32,
    ) -> String {
        let ev = telemetry.map_or_default(|t| format!("  EV100 {:.2}", t.adapted_ev100));
        format!(
            "{camera}  {w} x {h}  {:?}\n{averaged} frames averaged{ev}\n{}",
            resolved.encoding, self.figures,
        )
    }
}

/// A glTF camera.
struct Camera {
    name: String,
    position: Vec3,
    rotation: Quat,
    fov_y: f32,
    near: f32,
}

/// The glTF's cameras in node order.
fn read_gltf(path: &Path) -> Vec<Camera> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let gltf: Value = serde_json::from_slice(&bytes).expect("glTF JSON");
    let nodes = gltf["nodes"].as_array().expect("glTF nodes");
    let name = |n: &Value| n["name"].as_str().expect("node name").to_owned();
    nodes
        .iter()
        .filter_map(|n| {
            let index = n.get("camera")?.as_u64()? as usize;
            let perspective = &gltf["cameras"][index]["perspective"];
            let (_, rotation, position) = pose(n).to_scale_rotation_translation();
            Some(Camera {
                name: name(n),
                position,
                rotation,
                fov_y: perspective["yfov"].as_f64().expect("yfov") as f32,
                near: perspective["znear"].as_f64().expect("znear") as f32,
            })
        })
        .collect()
}

/// Cooks the scene and the environment into a pack in memory.
fn cook(gltf: PathBuf, studio: PathBuf) -> Arc<[u8]> {
    let started = Instant::now();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: BOX.into(),
                    source: Source::Gltf {
                        path: gltf,
                        settings: GltfSettings {
                            // The file's `asset.version` is the scene's own, 6.4.
                            ignore_version: true,
                            mesh: MeshSettings {
                                position_step_log2: POSITION_STEP_LOG2,
                                ..MeshSettings::default()
                            },
                            ..GltfSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: STUDIO.into(),
                    source: Source::Texture {
                        path: studio,
                        settings: TextureSettings {
                            usage: Usage::Environment {
                                light_threshold: None,
                                max_lights: 0,
                            },
                            max_size: Some(STUDIO_EDGE),
                            ..TextureSettings::default()
                        },
                    },
                },
            ],
        )],
    };
    let pack = orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0);
    log::info!("cooked in {:.1} s", started.elapsed().as_secs_f64());
    pack
}

/// The caustics the command line asks for.
fn caustic_settings(options: &Options) -> CausticSettings {
    let defaults = CausticSettings::default();
    CausticSettings {
        photons: number(options, "--photons").unwrap_or(defaults.photons),
        max_temporal_samples: number(options, "--caustic-history")
            .unwrap_or(defaults.max_temporal_samples),
        jitter: number(options, "--caustic-jitter").unwrap_or(defaults.jitter),
        ..defaults
    }
}

/// The number after `flag`, if given.
fn number<T: std::str::FromStr<Err: std::fmt::Display>>(
    options: &Options,
    flag: &str,
) -> Option<T> {
    let v = options.value(flag)?;
    Some(
        v.parse()
            .unwrap_or_else(|e| panic!("{flag} takes a number: {e}")),
    )
}

/// Runs the example; headless runs and windowed runs with `--out` and `--frames` return their
/// last frame.
fn run(options: &Options) -> Option<Capture> {
    let (gltf, studio, downloaded) = match options.value("--assets") {
        Some(dir) => {
            let dir = Path::new(dir);
            (asset(dir, "gltf"), asset(dir, "exr"), None)
        }
        None => {
            let dir = TempDir::new("orr-cornell-box");
            let (gltf, studio) = std::thread::scope(|s| {
                let studio = s.spawn(|| download(STUDIO_URL, &dir.0));
                let gltf = download(BOX_URL, &dir.0);
                (gltf, studio.join().expect("download"))
            });
            (gltf, studio, Some(dir))
        }
    };
    let cameras = read_gltf(&gltf);
    let wanted = options.value("--camera").unwrap_or(DEFAULT_CAMERA);
    let mut current = cameras
        .iter()
        .position(|c| c.name == wanted)
        .unwrap_or_else(|| {
            let names: Vec<&str> = cameras.iter().map(|c| c.name.as_str()).collect();
            panic!("no camera {wanted}; the scene has {names:?}")
        });
    let reference = options.has("--reference");
    let title = |camera: &Camera| format!("orr cornell-box: {}", camera.name);

    let gpu = device();
    let mut window =
        (!options.headless).then(|| Window::new(&title(&cameras[current]), 1920, 1080));
    let desc = RendererDesc::default();
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
        enable_dlss(&mut renderer);
    }
    let _workers = Workers::machine(renderer.loading());
    let (mount, mut frame) = mount_at(
        &mut renderer,
        PackSource::Memory(cook(gltf, studio)),
        FIRST_FRAME,
    );
    drop(downloaded);

    let scene = renderer
        .create_scene(
            &SceneDesc {
                global_instances: 1,
                ..SceneDesc::default()
            },
            &SceneLightingSettings {
                caustics: caustic_settings(options),
                ..SceneLightingSettings::default()
            },
        )
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    options.wait_for_paths(&mut renderer);
    let model = renderer
        .model(mount, AssetId::from_name(BOX))
        .expect("the scene's model");
    let environment = environment_map(
        &renderer,
        gpu,
        mount,
        STUDIO,
        Affine3::from_mat3(STUDIO_FROM_WORLD),
    );

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
    w.set_environment(environment.gpu());
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
            "cornell-box target",
        )
        .expect("headless target")
    });
    let lighting = ViewLighting {
        max_bounces: if reference {
            REFERENCE_DEPTH
        } else {
            MAX_BOUNCES
        },
        dielectric_depth: if reference {
            REFERENCE_DEPTH
        } else {
            DIELECTRIC_DEPTH
        },
        reference: reference.then(ReferenceSettings::default),
        ..ViewLighting::default()
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
        if let Some(w) = window.as_mut() {
            if !w.pump_showing_paths(&renderer) {
                break;
            }
            for key in w.take_keys() {
                let picked = CAMERA_KEYS.iter().position(|&k| k == key);
                if let Some(picked) = picked.filter(|&p| p < cameras.len() && p != current) {
                    current = picked;
                    camera_cut = true;
                    average.restart(ACCUMULATE_AFTER);
                    w.set_title(&title(&cameras[current]));
                }
            }
        }
        let camera = &cameras[current];
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
        perf.add(t - previous, renderer.stats());
        previous = t;

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
                let averaged = average.frames();
                let text = perf.readout(
                    &camera.name,
                    extent,
                    &resolved,
                    telemetry.as_ref(),
                    averaged,
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
        f.scene(scene).expect("scene writer").set_focus(position);
        let view = LitView {
            scene,
            history,
            position,
            rotation: camera.rotation,
            projection: Projection {
                fov_y: camera.fov_y,
                near: camera.near,
            },
            lens: Lens::default(),
            camera_cut,
            not_rendering_scene: false,
            screenshot: (capture && final_frame)
                .then_some(ScreenshotRequest::Display { include_ui: false }),
            texture_filter: TextureFilter::Hardware,
            target,
        };
        let output = ViewOutput::default();
        f.prepare_view(&view, &lighting, &output)
            .expect("prepare view");
        f.update_scenes(&mut |_| {}).expect("update scenes");
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
    drop(environment);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}
