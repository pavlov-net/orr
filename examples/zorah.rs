//! NVIDIA's Zorah sample scene as its public glTF export, `zorah_textured_public.v1` (MIT,
//! <https://developer.nvidia.com/rtx-kit>): 3.31 G unique triangles in 2,812 meshes placed
//! 16,118 times, 1,514 materials and 4,418 KTX2 textures, under the export's
//! `pizzo_pernice_4k.hdr` (Poly Haven, CC0). The scene cooks from the glTF at startup, every
//! mesh a cluster DAG placed by instance pages, and streams under the renderer's default
//! budgets. The run logs how long the cook and the mount took, and how long after the mount, and
//! after each jump between cameras, streaming went quiet and what it read and landed by then.
//!
//! `zorah --scene DIR [--pack FILE] [--cook-only] [--no-vsync] [--frames N] [--headless]
//! [--out file.png]`: `DIR` is the export's directory. Without `--pack` the archive cooks into
//! a temporary directory, removed when the run ends; with it the archive is kept at `FILE`,
//! and a later run with the same `FILE` cooks only what changed. `--cook-only` exits once the
//! archive is written. Keys 1 and 2 jump to the export's two cameras: the RTX MG SDK view in
//! `.scene.json` and the vk_lod_clusters view in `.cfg`. WASD moves, Q and E sink and rise,
//! shift moves faster and dragging with the right mouse button looks around. C logs the view as a
//! `--view` value and appends it to `zorah-views.txt` beside the executable; P probes the pixel
//! under the pointer and logs the report, appending it to `zorah-probes.txt` there. Headless runs
//! render 960 x 540 from the first camera, for 600 frames without `--frames`, and write the
//! last frame as a PNG; a windowed run needs `--frames` to write one. `--probe x,y` probes that
//! output pixel on the last frame and logs the report likewise.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use orr::orr_gpu::{
    Device, Format, QueueKind, TextureDesc, TextureMemory, TextureUsage, TimelinePoint,
};
use orr::orr_pack::texture::TextureInfo;
use orr::orr_pack::{Archive, AssetId, Class, Target};
use orr::{
    Budgets, DirectionalLight, DiscProfile, Display, DisplayRequest, FrameDesc, FrameTime,
    InstanceFlags, InstancePageRef, Lens, Light, LightId, LitView, LoadState, PackSource,
    Projection, Renderer, RendererDesc, RequestSource, ResidencyClass, ResidencyStats, SceneDesc,
    SceneLightingSettings, ScreenshotRequest, TextureFilter, UiPass, ViewHooks, ViewLighting,
    ViewOutput, ViewTarget,
};
use orr_cook::{
    ArchiveSpec, AssetSpec, CookOptions, GltfLayout, GltfSettings, InputChannels, LodMode,
    Manifest, MeshSettings, Outcome, Source, TextureOverride, TextureRules, TextureSettings, Usage,
};
use orr_examples::overlay::Overlay;
use orr_examples::window::{KeyCode, Window};
use orr_examples::{
    Capture, Options, Workers, assert_clean, device, enable_dlss, entry_head, environment_map,
    fixed_time, init_logging,
};
use orr_math::{Affine3, Mat3, Quat, Vec3};
use serde_json::Value;

const HEADLESS_EXTENT: [u32; 2] = [960, 540];
const HEADLESS_FRAMES: u64 = 600;
const SCENE: &str = "zorah";
const SKY: &str = "sky";
/// The export's `_Specular` textures are BC4, so their value is in red, not the alpha
/// `KHR_materials_specular` reads.
const SPECULAR_IMAGES: &str = "_Specular";
/// The source map is 4096 x 2048, whose equator texels an equal-area map of edge 2048 would
/// be coarser than.
const SKY_EDGE: u32 = 4096;
/// The sky averages a luminance of 1.1 over the upper hemisphere, and the sun's 400 source
/// texels above 100 hold 61% of the map's energy.
const SUN_THRESHOLD: f32 = 100.0;
/// The sidecar's `max bounces`.
const MAX_BOUNCES: u32 = 4;
/// The `.cfg`'s `cameraspeed`, in m/s.
const WALK_SPEED: f32 = 2.0;
const RUN_SCALE: f32 = 8.0;
/// Radians per device unit of mouse motion.
const LOOK_SCALE: f32 = 0.0025;
/// The `.cfg` camera's near plane; the sidecar view gives none.
const NEAR: f32 = 0.02;
/// Seconds each figure of the readout averages over.
const READOUT_PERIOD: f64 = 0.5;

fn main() {
    init_logging();
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("zorah");
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

/// A camera of the export.
#[derive(Clone, Copy)]
struct Camera {
    name: &'static str,
    position: Vec3,
    /// Yaw about +Y and pitch, in radians; yaw 0 looks down -Z.
    yaw: f32,
    pitch: f32,
    fov_y: f32,
}

impl Camera {
    fn looking(name: &'static str, eye: Vec3, at: Vec3, fov_y_degrees: f32) -> Self {
        let d = (at - eye).normalize();
        Camera {
            name,
            position: eye,
            yaw: (-d.x).atan2(-d.z),
            pitch: d.y.asin(),
            fov_y: fov_y_degrees.to_radians(),
        }
    }

    fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    /// `--view`'s value: `x,y,z,yaw,pitch,fov`, the angles in degrees.
    fn parse(view: &str) -> Self {
        let v: Vec<f32> = view
            .split(',')
            .map(|x| x.trim().parse().expect("a --view number"))
            .collect();
        let [x, y, z, yaw, pitch, fov] = v[..] else {
            panic!("--view takes x,y,z,yaw,pitch,fov");
        };
        Camera {
            name: "view",
            position: Vec3::new(x, y, z),
            yaw: yaw.to_radians(),
            pitch: pitch.to_radians(),
            fov_y: fov.to_radians(),
        }
    }

    /// This view as a `--view` value.
    fn view(&self) -> String {
        let p = self.position;
        format!(
            "{:.3},{:.3},{:.3},{:.2},{:.2},{:.2}",
            p.x,
            p.y,
            p.z,
            self.yaw.to_degrees(),
            self.pitch.to_degrees(),
            self.fov_y.to_degrees()
        )
    }
}

/// Logs `camera` as a `--view` value and appends it to `zorah-views.txt` beside the executable.
fn note_view(camera: &Camera) {
    use std::io::Write as _;
    let view = camera.view();
    log::info!("--view {view}");
    let path = std::env::current_exe()
        .expect("executable path")
        .with_file_name("zorah-views.txt");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path);
    match file.and_then(|mut f| writeln!(f, "{view}")) {
        Ok(()) => {}
        Err(e) => log::warn!("{}: {e}", path.display()),
    }
}

/// Logs a pixel probe and appends it to `zorah-probes.txt` beside the executable.
fn note_probe(camera: &Camera, probe: &orr::PixelProbe) {
    use std::io::Write as _;
    let report = format!("--view {}\n{probe}", camera.view());
    log::info!("pixel probe\n{report}");
    let path = std::env::current_exe()
        .expect("executable path")
        .with_file_name("zorah-probes.txt");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path);
    match file.and_then(|mut f| writeln!(f, "{report}")) {
        Ok(()) => {}
        Err(e) => log::warn!("{}: {e}", path.display()),
    }
}

/// `--probe`'s value: `x,y` in output pixels.
fn parse_probe(value: &str) -> [u32; 2] {
    let v: Vec<u32> = value
        .split(',')
        .map(|x| x.trim().parse().expect("a --probe pixel"))
        .collect();
    let [x, y] = v[..] else {
        panic!("--probe takes x,y");
    };
    [x, y]
}

/// What the export's files say beyond the glTF.
struct Export {
    gltf: PathBuf,
    sky: PathBuf,
    /// The sidecar's `envmap rotation`, in degrees.
    sky_rotation: f32,
    cameras: Vec<Camera>,
}

/// The one file in `dir` whose name ends with `suffix`.
fn file(dir: &Path, suffix: &str) -> PathBuf {
    let found: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.to_str().is_some_and(|s| s.ends_with(suffix)))
        .collect();
    match <[PathBuf; 1]>::try_from(found) {
        Ok([p]) => p,
        Err(found) => panic!(
            "{} has {} files ending {suffix}, not one",
            dir.display(),
            found.len()
        ),
    }
}

fn vec3(value: &Value) -> Vec3 {
    let v: Vec<f32> = value
        .as_array()
        .expect("a vector")
        .iter()
        .map(|x| x.as_f64().expect("a number") as f32)
        .collect();
    Vec3::from_slice(&v)
}

/// The numbers in `{...}` group `i` of the `.cfg`'s `--camerastring`.
fn camera_group(camera: &str, i: usize) -> Vec<f32> {
    camera
        .split('{')
        .nth(i + 1)
        .and_then(|g| g.split('}').next())
        .unwrap_or_else(|| panic!("camera string group {i}: {camera}"))
        .split(',')
        .map(|x| x.trim().parse().expect("a camera number"))
        .collect()
}

fn read_export(dir: &Path) -> Export {
    let sidecar = file(dir, ".scene.json");
    let json: Value = serde_json::from_slice(
        &std::fs::read(&sidecar).unwrap_or_else(|e| panic!("{}: {e}", sidecar.display())),
    )
    .expect("scene JSON");
    let gltf = dir.join(json["models"][0].as_str().expect("the scene's model"));
    let settings = &json["settings"];
    let sky = dir.join(settings["envmap"].as_str().expect("envmap"));
    let number = |key: &str| settings[key].as_f64().unwrap_or_else(|| panic!("{key}"));
    let sky_rotation = number("envmap rotation") as f32;
    let elevation = number("envmap elevation");
    assert!(elevation == 0.0, "envmap elevation {elevation} is not read");
    let view = &json["view"];
    let mut cameras = vec![Camera::looking(
        "sidecar",
        vec3(&view["position"]),
        vec3(&view["lookat"]),
        view["fov"].as_f64().expect("fov") as f32,
    )];

    let cfg = file(dir, ".cfg");
    let cfg = std::fs::read_to_string(&cfg).unwrap_or_else(|e| panic!("{}: {e}", cfg.display()));
    let camera = cfg
        .lines()
        .find_map(|l| l.strip_prefix("--camerastring "))
        .expect("--camerastring")
        .trim_matches('"');
    cameras.push(Camera::looking(
        "cfg",
        Vec3::from_slice(&camera_group(camera, 0)),
        Vec3::from_slice(&camera_group(camera, 1)),
        camera_group(camera, 3)[0],
    ));
    Export {
        gltf,
        sky,
        sky_rotation,
        cameras,
    }
}

/// Cooks the scene and its sky into the archive `pack`.
fn cook(export: &Export, pack: &Path) {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::File(pack.to_owned()),
            vec![
                AssetSpec {
                    name: SCENE.into(),
                    source: Source::Gltf {
                        path: export.gltf.clone(),
                        settings: GltfSettings {
                            layout: GltfLayout::Meshes,
                            mesh: MeshSettings {
                                lod: LodMode::Dag,
                                ..MeshSettings::default()
                            },
                            textures: TextureRules {
                                overrides: vec![(
                                    SPECULAR_IMAGES.into(),
                                    TextureOverride {
                                        channels: InputChannels {
                                            specular: Some(0),
                                            ..InputChannels::default()
                                        },
                                        ..TextureOverride::default()
                                    },
                                )],
                                ..TextureRules::default()
                            },
                            ..GltfSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: SKY.into(),
                    source: Source::Texture {
                        path: export.sky.clone(),
                        settings: TextureSettings {
                            usage: Usage::Environment {
                                light_threshold: Some(SUN_THRESHOLD),
                                max_lights: 1,
                            },
                            max_size: Some(SKY_EDGE),
                            ..TextureSettings::default()
                        },
                    },
                },
            ],
        )],
    };
    let started = Instant::now();
    let report = orr_cook::cook(&manifest, &CookOptions::default())
        .expect("cook")
        .report;
    let entries = report.assets.iter().flat_map(|a| &a.entries);
    let (mut cooked, mut linked, mut job_seconds) = (0, 0, 0.0);
    for e in entries {
        match e.outcome {
            Outcome::Cooked => cooked += 1,
            Outcome::Cache | Outcome::Archive => linked += 1,
        }
        job_seconds += e.seconds;
    }
    let archive = &report.archives[0];
    log::info!(
        "cooked {cooked} entries and linked {linked} in {:.1} s ({job_seconds:.0} job-s); {} has {} entries, {:.2} GiB{}",
        started.elapsed().as_secs_f64(),
        pack.display(),
        archive.entries,
        archive.bytes as f64 / f64::from(1 << 30),
        if archive.written { "" } else { ", unchanged" },
    );
}

/// The archive's instance pages.
fn page_ids(pack: &Path) -> Vec<AssetId> {
    let archive = Archive::open(pack).unwrap_or_else(|e| panic!("{}: {e}", pack.display()));
    archive
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
        .map(|e| e.id)
        .collect()
}

/// The renderer's default budgets, with the overrides the environment names.
fn budgets() -> Budgets {
    let defaults = Budgets::default();
    let mesh_page_pool =
        std::env::var("ZORAH_MESH_POOL_GIB").map_or(defaults.mesh_page_pool, |g| {
            orr::PoolSize::Bytes(
                g.parse::<u64>()
                    .expect("ZORAH_MESH_POOL_GIB is a count of GiB")
                    << 30,
            )
        });
    let count = |name: &str, default: u32| {
        std::env::var(name).map_or(default, |n| n.parse().expect("a count"))
    };
    Budgets {
        mesh_page_pool,
        initial_rt_cut_clusters: count("ZORAH_RT_CUT_CLUSTERS", defaults.initial_rt_cut_clusters),
        initial_rt_cut_blas: count("ZORAH_RT_CUT_BLAS", defaults.initial_rt_cut_blas),
        rt_cut_memory: std::env::var("ZORAH_RT_CUT_MIB").map_or(defaults.rt_cut_memory, |m| {
            orr::PoolSize::Bytes(
                m.parse::<u64>()
                    .expect("ZORAH_RT_CUT_MIB is a count of MiB")
                    << 20,
            )
        }),
        ..defaults
    }
}

/// The sky's extracted sun, in world space.
fn sun(pack: &Path, from_world: Mat3) -> Option<DirectionalLight> {
    let archive = Archive::open(pack).unwrap_or_else(|e| panic!("{}: {e}", pack.display()));
    let head = entry_head(&archive, AssetId::from_name(SKY)).expect("the sky's header");
    let info = TextureInfo::parse(&head).expect("the sky's header");
    let light = info.lights.first()?;
    Some(DirectionalLight {
        direction_to_light: from_world.transpose() * Vec3::from_array(light.direction_to_light),
        illuminance: Vec3::from_array(light.illuminance),
        angular_radius: light.angular_radius,
        disc: DiscProfile::Uniform,
    })
}

/// Fly-camera motion from the held keys over `dt` seconds.
fn fly(window: &mut Window, camera: &mut Camera, dt: f32) {
    let [x, y] = window.take_look();
    camera.yaw -= x as f32 * LOOK_SCALE;
    camera.pitch = (camera.pitch - y as f32 * LOOK_SCALE)
        .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
    let axis = |plus: KeyCode, minus: KeyCode| {
        f32::from(u8::from(window.held(plus))) - f32::from(u8::from(window.held(minus)))
    };
    let r = camera.rotation();
    let walk = r * Vec3::NEG_Z * axis(KeyCode::KeyW, KeyCode::KeyS)
        + r * Vec3::X * axis(KeyCode::KeyD, KeyCode::KeyA)
        + Vec3::Y * axis(KeyCode::KeyE, KeyCode::KeyQ);
    let speed = if window.held(KeyCode::ShiftLeft) || window.held(KeyCode::ShiftRight) {
        WALK_SPEED * RUN_SCALE
    } else {
        WALK_SPEED
    };
    camera.position += walk.normalize_or_zero() * speed * dt;
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1 << 20)
}

/// Whether the streamed units the view wants have all landed, since the mount or the last
/// camera cut: no loads, nothing in flight, no request dropped, held back or lost to a full
/// feedback.
struct Streaming {
    since: Instant,
    cause: &'static str,
    /// The residency figures at `since`.
    start: ResidencyStats,
    /// Seconds from `since` to the first step with nothing to stream.
    settled: Option<f64>,
    status: String,
}

impl Streaming {
    fn new(since: Instant, cause: &'static str, start: ResidencyStats) -> Self {
        Self {
            since,
            cause,
            start,
            settled: None,
            status: format!("streaming since {cause}"),
        }
    }

    fn step(&mut self, renderer: &Renderer, camera: &str) {
        let r = renderer.residency();
        let cuts = renderer.stats().cuts;
        let detail = renderer.stats().detail;
        let loading = r.classes.iter().any(|c| c.loading_bytes > 0);
        let dropped: u32 = r.dropped.iter().sum();
        let lost = cuts.requests_lost + r.texture_feedback_lost;
        let seconds = self.since.elapsed().as_secs_f64();
        if r.loads > 0 || r.unserved() > 0 || lost > 0 || loading {
            self.status = format!(
                "streaming: {} loads, {dropped} dropped, {} waiting, {lost} lost, {seconds:.1} s since {}",
                r.loads, r.waiting, self.cause
            );
            return;
        }
        let first = self.settled.is_none();
        let settled = *self.settled.get_or_insert(seconds);
        if first {
            let s = &self.start;
            let read = r.bytes_read - s.bytes_read;
            log::info!(
                "{camera}: streaming settled {settled:.1} s after {} at detail x{:.3} (visible x{:.3}, hidden x{:.3}; gathers x{:.3}), mesh pages {:.0} of {:.0} MiB wanted: {} loads landed, {:.0} MiB read at {:.0} MiB/s, {} CLAS retries, CLAS range full {} times, cluster ids full {} times, longest closure since the start {}",
                self.cause,
                detail.memory_scale,
                detail.visible_scale,
                detail.hidden_scale,
                detail.rt_scale,
                mib(detail.mesh_demand),
                mib(detail.mesh_budget),
                r.landed - s.landed,
                mib(read),
                mib(read) / settled,
                r.clas_retries - s.clas_retries,
                r.clas_range_full - s.clas_range_full,
                r.cluster_ids_full - s.cluster_ids_full,
                r.longest_closure,
            );
        }
        self.status = format!("settled {settled:.1} s after {}", self.cause);
    }
}

/// The passes of a scene's DAG traversal and ray-tracing cuts.
const CUT_PASSES: [&str; 4] = ["traverse", "admit cuts", "fill cuts", "build cuts"];

/// Frame rate and residency, averaged over each `READOUT_PERIOD`.
struct Readout {
    frames: u32,
    seconds: f64,
    /// GPU milliseconds of the scenes' passes named in `CUT_PASSES`, summed over the period.
    cut_ms: [f64; 4],
    /// The residency figures at the start of the period.
    start: ResidencyStats,
    text: String,
}

impl Readout {
    fn new(renderer: &Renderer) -> Self {
        Self {
            frames: 0,
            seconds: 0.0,
            cut_ms: [0.0; 4],
            start: *renderer.residency(),
            text: String::new(),
        }
    }

    fn add(&mut self, delta: f64, renderer: &Renderer) {
        self.frames += 1;
        self.seconds += delta;
        for p in &renderer.stats().gpu {
            if let Some(i) = CUT_PASSES.iter().position(|&n| n == p.name) {
                self.cut_ms[i] += p.gpu_ms;
            }
        }
        if self.seconds < READOUT_PERIOD {
            return;
        }
        let r = renderer.residency();
        let cuts = renderer.stats().cuts;
        let detail = renderer.stats().detail;
        let class = |c: ResidencyClass| {
            let s = r.class(c);
            format!(
                "{:.0} / {:.0} MiB, {:.0} loading",
                mib(s.resident_bytes),
                mib(s.budget),
                mib(s.loading_bytes)
            )
        };
        let s = &self.start;
        let rate = |n: f64| n / self.seconds;
        let dropped = |source| r.dropped_from(source);
        self.text = format!(
            "{:.1} fps  {:.2} ms; GPU ms: traversal {:.2}, cut admission {:.2}, fill {:.2}, BLAS build {:.2}\ndetail x{:.3} (visible x{:.3}, hidden x{:.3}), gathers x{:.3}, mesh pages {:.0} of {:.0} MiB wanted, {:.0} MiB more held by ray cuts\nmesh pages {}\ntextures {}\ninstance pages {}\nCLAS {:.0} / {:.0} MiB used, {:.0} retries/s, range full {} times, cluster ids full {} times\nread {:.0} MiB/s, {:.0} loads/s landed\n{} loads  {} evictions  {} failed  {} out of memory  {} pages over budget\ndropped: {} traversal, {} finer cuts, {} textures, {} instance pages, {} hints\n{} waiting  longest closure {}\nfeedback lost: {} requests, {} texture raises\nray cuts: {} gathered ({} cached, {} shared, {} fell back, {} past a full queue, {} empty), {} merged, {} BLAS built of {} CLAS; wanted {} BLAS of {} CLAS, tables hold {} BLAS of {} CLAS in {:.1} MiB\nraster: {} candidate items, {} dropped, {} nodes dropped\nhidden test: {} spheres, {} leaves and {} clusters taken by the hidden thresholds",
            f64::from(self.frames) / self.seconds,
            self.seconds * 1000.0 / f64::from(self.frames),
            self.cut_ms[0] / f64::from(self.frames),
            self.cut_ms[1] / f64::from(self.frames),
            self.cut_ms[2] / f64::from(self.frames),
            self.cut_ms[3] / f64::from(self.frames),
            detail.memory_scale,
            detail.visible_scale,
            detail.hidden_scale,
            detail.rt_scale,
            mib(detail.mesh_demand),
            mib(detail.mesh_budget),
            mib(detail.mesh_held),
            class(ResidencyClass::MeshPages),
            class(ResidencyClass::Textures),
            class(ResidencyClass::InstancePages),
            mib(r.clas_pool_used_bytes),
            mib(r.clas_pool_bytes),
            rate(f64::from(r.clas_retries - s.clas_retries)),
            r.clas_range_full - s.clas_range_full,
            r.cluster_ids_full - s.cluster_ids_full,
            rate(mib(r.bytes_read - s.bytes_read)),
            rate(f64::from(r.landed - s.landed)),
            r.loads,
            r.evictions,
            r.failed,
            r.out_of_memory,
            r.pages_over_budget,
            dropped(RequestSource::Traversal),
            dropped(RequestSource::FinerCuts),
            dropped(RequestSource::Textures),
            dropped(RequestSource::InstancePages),
            dropped(RequestSource::Hints),
            r.waiting,
            r.longest_closure,
            cuts.requests_lost,
            r.texture_feedback_lost,
            cuts.rt_gathered,
            cuts.rt_cached,
            cuts.rt_shared,
            cuts.rt_fallback,
            cuts.rt_lost,
            cuts.rt_empty,
            cuts.rt_merged,
            cuts.rt_built,
            cuts.rt_built_clas,
            cuts.rt_wanted_blas,
            cuts.rt_wanted_clas,
            cuts.rt_blas_capacity,
            cuts.rt_clas_capacity,
            mib(cuts.rt_table_bytes),
            cuts.candidate_items,
            cuts.candidates_dropped,
            cuts.nodes_dropped,
            cuts.hidden_tests,
            cuts.hidden_leaves,
            cuts.hidden_clusters,
        );
        let memory = device().memory_report();
        let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
        for h in memory.heaps.iter().filter(|h| h.device_local) {
            self.text += &format!(
                "
device memory: {:.2} GiB used of {:.2} budget ({:.2} heap), {:.2} in blocks",
                gib(h.usage),
                gib(h.budget),
                gib(h.size),
                gib(h.block_bytes)
            );
        }
        let kinds: Vec<String> = orr::orr_gpu::MemoryKind::ALL
            .iter()
            .zip(&memory.kinds)
            .filter(|(_, u)| u.bytes > 0)
            .map(|(k, u)| format!("{k:?} {:.2} GiB in {}", gib(u.bytes), u.allocations))
            .collect();
        self.text += &format!(
            "
allocated: {}",
            kinds.join(", ")
        );
        if std::env::var_os("ZORAH_LOG_READOUT").is_some() {
            log::info!("readout\n{}{}", self.text, breakdown(renderer));
        }
        self.start = *r;
        self.frames = 0;
        self.seconds = 0.0;
        self.cut_ms = [0.0; 4];
    }
}

/// Labels the breakdown lists, the largest first.
const TOP_LABELS: usize = 16;

/// Where device memory goes: the largest labels, the allocator's pools with what their blocks
/// leave free, the scenes' pools with their fill, and per device-local heap what the process
/// uses outside orr_gpu's blocks.
fn breakdown(renderer: &Renderer) -> String {
    let b = device().memory_breakdown();
    let scene = renderer.scene_memory();
    let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
    let mut s = format!(
        "\ndevice memory allocations: {} of {}, {} made so far",
        b.device_allocations,
        b.max_device_allocations,
        device().memory_report().device_allocations
    );
    for l in b.labels.iter().take(TOP_LABELS) {
        s += &format!(
            "\n  label {:?} {:?}: {:.3} GiB in {}, {:.3} GiB in {} of their own",
            l.label,
            l.kind,
            gib(l.bytes),
            l.allocations,
            gib(l.dedicated_bytes),
            l.dedicated
        );
    }
    let rest: u64 = b.labels.iter().skip(TOP_LABELS).map(|l| l.bytes).sum();
    s += &format!(
        "\n  {} more labels: {:.3} GiB",
        b.labels.len().saturating_sub(TOP_LABELS),
        gib(rest)
    );
    for p in b.pools.iter().filter(|p| p.blocks > 0) {
        s += &format!(
            "\n  pool {:?} heap {}: {} blocks {:.3} GiB, {} allocations {:.3} GiB, {:.3} GiB free in blocks",
            p.name,
            p.heap,
            p.blocks,
            gib(p.block_bytes),
            p.allocations,
            gib(p.allocation_bytes),
            gib(p.free_in_blocks())
        );
    }
    for p in &scene.pools {
        s += &format!(
            "\n  scene pool {:?} {:?}: {} blocks {:.3} GiB, {:.3} GiB used, {} at most half used, {} made, {:.3} GiB spare",
            p.name,
            p.kind,
            p.blocks,
            gib(p.bytes),
            gib(p.used),
            p.half_empty,
            p.made,
            gib(p.spare)
        );
    }
    if let Some(c) = scene.clas {
        s += &format!(
            "\n  CLAS pages of {} KiB: {} bound, {} wholly free, {} reserved, {:.3} GiB used",
            c.page_bytes >> 10,
            c.bound,
            c.wholly_free,
            c.reserved,
            gib(c.used)
        );
    }
    let windows = scene.windows.iter();
    let (blocks, bytes, used) = windows.fold((0, 0, 0), |(n, b, u), w| {
        (n + w.blocks, b + w.bytes, u + w.used)
    });
    s += &format!(
        "\n  upload windows: {} blocks {:.3} GiB, {:.3} GiB used by their last frames",
        blocks,
        gib(bytes),
        gib(used)
    );
    for (i, h) in b.heaps.iter().enumerate().filter(|(_, h)| h.device_local) {
        s += &format!(
            "\n  heap {i}: {:.3} GiB used, {:.3} in blocks, {:.3} outside orr_gpu",
            gib(h.usage),
            gib(h.block_bytes),
            h.outside() as f64 / f64::from(1u32 << 30)
        );
    }
    s
}

/// One line of what frame `frame` holds: per device-local heap its use against its budget and
/// what is waiting to be freed, per scene pool the bytes used, retired and draining against its
/// blocks' bytes, the CLAS pool's bound pages, each class's resident bytes against its budget,
/// and the heaps' excess and shortfall, in MiB.
fn memory_line(frame: u64, renderer: &Renderer) -> String {
    let report = device().memory_report();
    let scene = renderer.scene_memory();
    let r = renderer.residency();
    let detail = renderer.stats().detail;
    let mut s = format!("memory at frame {frame}:");
    for h in report.heaps.iter().filter(|h| h.device_local) {
        s += &format!(
            " heap {:.0}/{:.0} freeing {:.0}",
            mib(h.usage),
            mib(h.budget),
            mib(h.freeing)
        );
    }
    for p in &scene.pools {
        s += &format!(
            "; {} {:.0}/{:.0} in {} (retired {:.0}, draining {:.0})",
            p.name,
            mib(p.used),
            mib(p.bytes),
            p.blocks,
            mib(p.retired),
            mib(p.draining)
        );
    }
    if let Some(c) = scene.clas {
        s += &format!(
            "; CLAS pages {} bound {} free {} held, {:.0} used, {} moved",
            c.bound,
            c.wholly_free,
            c.held,
            mib(c.used),
            r.clas_pages_moved
        );
    }
    for class in [ResidencyClass::MeshPages, ResidencyClass::Textures] {
        let c = r.class(class);
        s += &format!(
            "; {class:?} {:.0}/{:.0}",
            mib(c.resident_bytes),
            mib(c.budget)
        );
    }
    s += &format!(
        "; excess {:.0} short {:.0}; detail x{:.3}; {} loads {} evictions",
        mib(r.heap_excess),
        mib(r.heap_shortfall),
        detail.memory_scale,
        r.loads,
        r.evictions
    );
    s
}

/// Runs the example; headless runs and windowed runs with `--out` and `--frames` return their
/// last frame.
fn run(options: &Options) -> Option<Capture> {
    let dir = options
        .value("--scene")
        .expect("--scene DIR names the export's directory");
    let export = read_export(Path::new(dir));
    let (pack, temp) = match options.value("--pack") {
        Some(p) => {
            let p = PathBuf::from(p);
            if let Some(dir) = p.parent() {
                std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
            }
            (p, None)
        }
        None => {
            let temp = TempDir::new("orr-zorah");
            (temp.0.join("zorah.orr"), Some(temp))
        }
    };
    cook(&export, &pack);
    if options.has("--cook-only") {
        return None;
    }
    let cameras = export.cameras;
    // The view: `--view`'s, else the export's camera last jumped to, as flown since.
    let mut camera = options.value("--view").map_or(cameras[0], Camera::parse);
    let title = |camera: &Camera| format!("orr zorah: {}", camera.name);

    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new(&title(&camera), 1920, 1080));
    let pixels = |name: &str, default: f32| {
        std::env::var(name).map_or(default, |n| n.parse().expect("an error in pixels"))
    };
    let defaults = RendererDesc::default();
    let desc = RendererDesc {
        budgets: budgets(),
        raster_error_pixels: pixels("ZORAH_RASTER_PIXELS", defaults.raster_error_pixels),
        rt_error_pixels: pixels("ZORAH_RT_PIXELS", defaults.rt_error_pixels),
        rt_merge_pixels: pixels("ZORAH_RT_MERGE_PIXELS", defaults.rt_merge_pixels),
        software_raster_pixels: pixels(
            "ZORAH_SOFTWARE_RASTER_PIXELS",
            defaults.software_raster_pixels,
        ),
        culled_error_scale: pixels("ZORAH_CULLED_SCALE", defaults.culled_error_scale),
        hidden_detail_lead: pixels("ZORAH_HIDDEN_LEAD", defaults.hidden_detail_lead),
        ..defaults
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
    if !options.headless {
        enable_dlss(&mut renderer);
    }
    let workers = Workers::machine(renderer.loading());

    let mounting = Instant::now();
    let mount = renderer.mount(PackSource::File(pack.clone()));
    let mut frame = 0;
    while matches!(renderer.mount_state(mount), LoadState::Pending) {
        frame += 1;
        if window.as_mut().is_some_and(|w| !w.pump()) {
            end(gpu, display, renderer, window, workers, temp);
            return None;
        }
        renderer
            .begin_frame(FrameDesc {
                time: fixed_time(frame),
                scenes: &[],
                display: None,
            })
            .expect("begin frame")
            .end()
            .expect("end frame");
        std::thread::sleep(Duration::from_millis(1));
    }
    if let LoadState::Failed(e) = renderer.mount_state(mount) {
        panic!("mount failed: {e}");
    }
    log::info!("mounted in {:.1} s", mounting.elapsed().as_secs_f64());

    let pages: Vec<_> = page_ids(&pack)
        .into_iter()
        .map(|id| renderer.instance_page(mount, id).expect("instance page"))
        .collect();
    let scene = renderer
        .create_scene(&SceneDesc::default(), &SceneLightingSettings::default())
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    options.wait_for_paths(&mut renderer);
    // RTX MG looks the map up at `Ry(-rotation)` of the world direction, with `u` from
    // `atan2(z, x)`, a quarter turn from the cook's `atan2(x, -z) + pi`.
    let from_world = Mat3::from_rotation_y(-(export.sky_rotation + 90.0).to_radians());
    let environment = environment_map(&renderer, gpu, mount, SKY, Affine3::from_mat3(from_world));
    let sun = sun(&pack, from_world);
    if let Some(s) = &sun {
        let d = s.direction_to_light;
        log::info!(
            "sun toward ({:.4}, {:.4}, {:.4}): yaw {:.2}, pitch {:.2} deg",
            d.x,
            d.y,
            d.z,
            (-d.x).atan2(-d.z).to_degrees(),
            d.y.asin().to_degrees()
        );
    }

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
    for &page in &pages {
        w.add_instance_page(&InstancePageRef {
            page,
            frame: None,
            poses: &[],
            material_sets: &[],
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: f32::INFINITY,
            animation_range: 0.0,
            user_tag: 0,
        })
        .expect("instance page");
    }
    if let Some(sun) = sun {
        w.set_light(LightId(0), Some(&Light::Directional(sun)))
            .expect("sun");
    }
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
            "zorah target",
        )
        .expect("headless target")
    });
    use orr::LightingDebugView;
    use orr::orr_lighting::{GuideKind, TracedKind};
    let debug_view = match std::env::var("ZORAH_DEBUG_VIEW").as_deref() {
        Err(_) => LightingDebugView::None,
        Ok("dielectric") => LightingDebugView::DielectricLayer,
        Ok("bias") => LightingDebugView::RayOriginBias,
        Ok("direct") => LightingDebugView::DirectOnly,
        Ok("albedo") => LightingDebugView::Guide(GuideKind::DiffuseAlbedo),
        Ok("specular") => LightingDebugView::Guide(GuideKind::SpecularAlbedo),
        Ok("normal") => LightingDebugView::Guide(GuideKind::Normal),
        Ok("roughness") => LightingDebugView::Guide(GuideKind::Roughness),
        Ok("traced") => LightingDebugView::Traced(TracedKind::Distance),
        Ok("traced-normal") => LightingDebugView::Traced(TracedKind::Normal),
        Ok("traced-disagree") => LightingDebugView::Traced(TracedKind::Disagreement),
        Ok("traced-sun") => LightingDebugView::Traced(TracedKind::Sun),
        Ok(other) => panic!("ZORAH_DEBUG_VIEW {other}"),
    };
    let lighting = ViewLighting {
        max_bounces: MAX_BOUNCES,
        debug_view,
        ..ViewLighting::default()
    };
    let capture = options.headless || options.out.is_some();
    let mut overlay = Overlay::new(desc.frames_in_flight);
    let mut readout = Readout::new(&renderer);
    let frames = options
        .frames
        .or_else(|| options.headless.then_some(HEADLESS_FRAMES));
    let started = Instant::now();
    let mut previous = 0.0;
    let mut last: Option<(u64, TimelinePoint)> = None;
    let mut camera_cut = false;
    let mut streaming = Streaming::new(mounting, "the mount", ResidencyStats::default());
    // `ZORAH_PRESSURE_GIB` GiB of device memory taken at frame `ZORAH_PRESSURE_FRAME` and held
    // to the end, so the budgets shrink under the heap's pressure partway through a run.
    let pressure = std::env::var("ZORAH_PRESSURE_GIB").ok().map(|g| {
        let frame = std::env::var("ZORAH_PRESSURE_FRAME").map_or(1000, |f| {
            f.parse::<u64>()
                .expect("ZORAH_PRESSURE_FRAME is a frame count")
        });
        (
            frame,
            g.parse::<u64>()
                .expect("ZORAH_PRESSURE_GIB is a count of GiB"),
        )
    });
    let mut held = Vec::new();
    // At frame `ZORAH_CUT_FRAME` the mesh page and texture budgets drop to
    // `ZORAH_CUT_MESH_MIB` and `ZORAH_CUT_TEXTURE_MIB`.
    let cut = std::env::var("ZORAH_CUT_FRAME").ok().map(|f| {
        let mib = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|m| m.parse::<u64>().expect("a count of MiB") << 20)
        };
        let frame = f.parse::<u64>().expect("ZORAH_CUT_FRAME is a frame count");
        (
            frame,
            mib("ZORAH_CUT_MESH_MIB"),
            mib("ZORAH_CUT_TEXTURE_MIB"),
        )
    });
    // Every `ZORAH_LOG_MEMORY` frames the run logs what its memory holds.
    let memory_every = std::env::var("ZORAH_LOG_MEMORY")
        .ok()
        .map(|n| n.parse::<u64>().expect("ZORAH_LOG_MEMORY is a frame count"));
    // The output pixel `--probe` probes on the last frame, and the one `P` asked for.
    let probe_at = options.value("--probe").map(parse_probe);
    let mut probe = None;
    for i in 0.. {
        if frames.is_some_and(|n| i >= n) {
            break;
        }
        if let Some((at, mesh, textures)) = cut
            && i == at
        {
            log::info!(
                "lowering the budgets at frame {i}: mesh pages {mesh:?}, textures {textures:?}"
            );
            for (class, bytes) in [
                (ResidencyClass::MeshPages, mesh),
                (ResidencyClass::Textures, textures),
            ] {
                if let Some(b) = bytes {
                    renderer.lower_budget(class, b);
                }
            }
        }
        if let Some((at, gib)) = pressure
            && i == at
        {
            log::info!("taking {gib} GiB of device memory at frame {i}");
            for _ in 0..gib {
                let kind = orr::orr_gpu::MemoryKind::DeviceLocal;
                held.push(
                    gpu.gpu_malloc::<u8>(kind, 1 << 30, "zorah pressure")
                        .expect("pressure"),
                );
            }
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
            for key in w.take_keys() {
                let keys = [KeyCode::Digit1, KeyCode::Digit2];
                let picked = keys.iter().position(|&k| k == key);
                if let Some(&picked) = picked.and_then(|p| cameras.get(p)) {
                    camera = picked;
                    camera_cut = true;
                    streaming = Streaming::new(Instant::now(), "the cut", *renderer.residency());
                    w.set_title(&title(&camera));
                }
                if key == KeyCode::KeyC {
                    note_view(&camera);
                }
                if key == KeyCode::KeyP {
                    probe = w.cursor();
                }
            }
            fly(w, &mut camera, dt as f32);
        }
        let final_frame = frames.is_some_and(|n| i + 1 == n);
        let time = FrameTime {
            absolute: fixed_time(frame + 1).absolute + t,
            delta: dt as f32,
        };
        readout.add(dt, &renderer);

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: display.as_mut(),
            })
            .expect("begin frame");
        let id = f.id();
        let target = match (&headless_target, f.display()) {
            (Some(t), _) => ViewTarget::Texture(t),
            (None, Some((_, extent))) => {
                let text = format!(
                    "{}  {} x {}  view {}\n{}\n{}",
                    camera.name,
                    extent[0],
                    extent[1],
                    camera.view(),
                    streaming.status,
                    readout.text
                );
                overlay.layout(extent, time, &text);
                ViewTarget::Display
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
        let output = ViewOutput::default();
        if final_frame && let Some(at) = probe_at {
            probe = Some(at);
        }
        if let Some(at) = probe.take() {
            f.probe(history, at);
        }
        f.prepare_view(&view, &lighting, &output)
            .expect("prepare view");
        f.update_scenes(&mut |_| {}).expect("update scenes");
        let mut ui = |pass: &mut UiPass<'_, '_>| overlay.paint(pass, id);
        let mut hooks = ViewHooks {
            ui: Some(&mut ui),
            ..ViewHooks::default()
        };
        f.render_view(&view, &lighting, &output, &mut hooks)
            .expect("render view");
        camera_cut = false;
        last = Some((id, f.end().expect("end frame")));
        streaming.step(&renderer, camera.name);
        if memory_every.is_some_and(|n| i.is_multiple_of(n)) {
            log::info!("{}", memory_line(i, &renderer));
        }
        let completed = renderer.stats().frame;
        for p in renderer.probes(history, completed) {
            note_probe(&camera, &p);
        }
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
    if let Some((id, _)) = last.filter(|_| probe_at.is_some()) {
        // A frame of no scene takes the probed frame's feedback to the loading side's plan, which
        // the probe reports.
        frame += 1;
        renderer
            .begin_frame(FrameDesc {
                time: fixed_time(frame),
                scenes: &[],
                display: None,
            })
            .expect("begin frame")
            .end()
            .expect("end frame");
        for p in renderer.probes(history, id) {
            note_probe(&camera, &p);
        }
    }
    let captured = last
        .filter(|_| capture)
        .and_then(|(id, _)| renderer.screenshots(history, id).pop())
        .map(|shot| Capture::from_screenshot(&shot));
    // Off Windows the overlay is a stub that holds nothing.
    #[allow(clippy::drop_non_drop)]
    drop(overlay);
    drop(headless_target);
    drop(held);
    drop(environment);
    end(gpu, display, renderer, window, workers, temp);
    captured
}

/// Drops what every run holds, however it ends.
fn end(
    gpu: &Device,
    display: Option<Display>,
    renderer: Renderer,
    window: Option<Window>,
    workers: Workers,
    temp: Option<TempDir>,
) {
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    // Jobs left on the workers' queue can hold the archive open.
    drop(workers);
    drop(temp);
}
