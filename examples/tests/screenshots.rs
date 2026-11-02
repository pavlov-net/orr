//! Runs each example headless and compares its last frame with its reference in
//! `references/` by FLIP; `ORR_BLESS=1` rewrites the references. hdr-calibration shows what
//! a display does with its patterns and has no headless run. The dlss example's super
//! resolution is compared with the same view rendered at the output size instead.

#[allow(dead_code)]
#[path = "../cluster-rt.rs"]
mod cluster_rt;
#[allow(dead_code)]
#[path = "../dense-vegetation.rs"]
mod dense_vegetation;
#[allow(dead_code)]
#[path = "../dielectrics.rs"]
mod dielectrics;
#[allow(dead_code)]
#[path = "../dlss.rs"]
mod dlss;
#[allow(dead_code)]
#[path = "../gpu-basics.rs"]
mod gpu_basics;
#[allow(dead_code)]
#[path = "../producer-field.rs"]
mod producer_field;
#[allow(dead_code)]
#[path = "../rigid-parts.rs"]
mod rigid_parts;
#[allow(dead_code)]
#[path = "../skinned.rs"]
mod skinned;
#[allow(dead_code)]
#[path = "../sky.rs"]
mod sky;
#[allow(dead_code)]
#[path = "../viewer.rs"]
mod viewer;

use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use examples::average::Average;
use examples::mesh::MeshBuilder;
use examples::scene::{FOV_Y, Showcase, TARGET, orbit_eye};
use examples::{
    Capture, Options, Workers, assert_clean, device, enable_dlss, fixed_time, mount_at,
};
use orr::orr_gpu::{ClearColor, ColorAttachment, ColorLoad, Format, Rendering};
use orr::orr_gpu::{QueueKind, TextureDesc, TextureMemory, TextureUsage};
use orr::orr_pack::material::Material;
use orr::orr_pack::{AssetId, Target};
use orr::{
    Affine3, DlssMode, DlssQuality, FrameDesc, GroupWriter, HookPass, InstanceFlags, InstancesDesc,
    Lens, LitView, PackSource, Projection, Reconstruction, ReferenceSettings, Renderer,
    RendererDesc, SceneId, ScreenshotRequest, TextureFilter, ThinLens, TransformGroupDesc,
    TransparencyPass, ViewHistoryId, ViewHooks, ViewLighting, ViewOutput, ViewTarget,
};
use orr_cook::{ArchiveSpec, AssetSpec, LodMode, Manifest, MeshSettings, Source};
use orr_math::camera::rh::view;
use orr_math::{DVec3, Vec3};
use orr_scene::testing::{Image, Pixels, Tolerance, check, flip};

/// A smaller forest than the bench's, so the debug build renders it in reasonable time.
const DENSE_VEGETATION: [&str; 5] = ["--synthetic", "--trees", "2000", "--grass-density", "50"];

/// One example runs at a time on the shared device.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn image(c: Capture) -> Image {
    Image {
        width: c.width,
        height: c.height,
        pixels: Pixels::Rgba8(c.rgba8),
    }
}

fn compare(name: &str, c: Capture) {
    let image = image(c);
    let reference = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("references")
        .join(name);
    check(
        &reference,
        &image,
        Tolerance::Flip {
            mean: 0.01,
            p99: 0.1,
        },
    );
}

fn headless(rest: &[&str]) -> Options {
    Options::headless(120, rest)
}

#[test]
fn gpu_basics_matches_its_reference() {
    let _serial = serial();
    compare("gpu-basics", gpu_basics::run(&headless(&[])));
    assert_clean();
}

#[test]
fn cluster_rt_matches_its_references() {
    let _serial = serial();
    let cluster_rt = |rest: &[&str]| cluster_rt::run(&headless(rest)).expect("headless capture");
    compare("cluster-rt", cluster_rt(&[]));
    compare("cluster-rt-clusters", cluster_rt(&["--clusters"]));
    assert_clean();
}

#[test]
fn sky_matches_its_reference() {
    let _serial = serial();
    compare("sky", sky::run(&headless(&[])).expect("headless capture"));
    assert_clean();
}

#[test]
fn viewer_matches_its_reference() {
    let _serial = serial();
    compare(
        "viewer",
        viewer::run(&headless(&[])).expect("headless capture"),
    );
    assert_clean();
}

#[test]
fn rigid_parts_matches_its_reference() {
    let _serial = serial();
    compare(
        "rigid-parts",
        rigid_parts::run(&headless(&[])).expect("headless capture"),
    );
    assert_clean();
}

#[test]
fn dielectrics_matches_its_reference() {
    let _serial = serial();
    compare(
        "dielectrics",
        dielectrics::run(&headless(&[])).expect("headless capture"),
    );
    assert_clean();
}

#[test]
fn producer_field_matches_its_reference() {
    let _serial = serial();
    let field = producer_field::run(&headless(&[])).expect("headless capture");
    compare("producer-field", field);
    assert_clean();
}

#[test]
fn skinned_matches_its_reference() {
    let _serial = serial();
    compare(
        "skinned",
        skinned::run(&headless(&[])).expect("headless capture"),
    );
    assert_clean();
}

#[test]
fn dense_vegetation_matches_its_reference() {
    let _serial = serial();
    let forest = dense_vegetation::capture(&headless(&DENSE_VEGETATION)).expect("headless capture");
    compare("dense-vegetation", forest);
    assert_clean();
}

/// The dlss example's raster debug view upscaled by super resolution at Quality, on an
/// orbiting camera, matches the view rendered at the output size, and NGX's log names the
/// preset it applied.
#[test]
fn super_resolution_matches_the_native_render() {
    let _serial = serial();
    let frames = 64;
    let Some(sr) = dlss::run_sr(&Options::headless(frames, &["--sr", "--quality"])) else {
        log::warn!("skipped: NGX offers no super resolution here");
        return;
    };
    let native = dlss::run_sr(&Options::headless(frames, &["--sr", "--native"])).expect("native");
    let preset = sr.preset.expect("NGX's log names the applied SR preset");
    let error = flip(
        &image(native.capture.expect("native capture")),
        &image(sr.capture.expect("SR capture")),
    );
    log::info!("SR Quality, preset {preset}: FLIP {error:?} against the native render");
    // Measured mean 0.0086 and p99 0.141 (preset K); the jitter sign flipped gives 0.0118
    // and 0.189.
    assert!(
        error.mean <= 0.010 && error.p99 <= 0.16,
        "SR against native: {error:?}"
    );
    assert_clean();
}

/// Radians per frame of an orbiting camera.
const ORBIT_STEP: f64 = 0.01;

/// What one run of the showcase's still camera rendered.
struct StillRun {
    capture: Capture,
    render_extent: [u32; 2],
    mode: Option<DlssMode>,
}

/// How [`StillShowcase::run`] renders.
#[derive(Clone, Copy, Default)]
struct StillSettings {
    reconstruction: Reconstruction,
    /// Averages the frames after the first few in a scene-referred hook.
    average: bool,
    /// Premultiplied pre-exposed color over the whole image at the transparency point.
    overlay: Option<[f32; 4]>,
    /// The accumulation reference, with a lens small enough to do no more than jitter each
    /// pixel's first ray.
    reference: bool,
    /// The camera orbits, reaching the still camera's place on the last frame.
    orbit: bool,
}

/// The showcase's camera at `eye`.
fn still_view<'a>(
    scene: SceneId,
    history: ViewHistoryId,
    eye: Vec3,
    reset_history: bool,
    target: ViewTarget<'a>,
) -> LitView<'a> {
    LitView {
        scene,
        history,
        position: eye.as_dvec3(),
        rotation: view::look_at_quat(eye, TARGET, Vec3::Y).inverse(),
        projection: Projection {
            fov_y: FOV_Y,
            near: 0.05,
        },
        lens: Lens::default(),
        reset_history,
        texture_filter: TextureFilter::Hardware,
        target,
    }
}

/// The showcase's scene and a still camera, on one renderer: its pipelines take long to compile
/// in a debug build with the validation layer.
struct StillShowcase {
    showcase: Showcase,
    history: ViewHistoryId,
    frame: u64,
    _workers: Workers,
    renderer: Renderer,
}

impl StillShowcase {
    fn new() -> Self {
        let gpu = device();
        let desc = RendererDesc::default();
        let mut renderer = Renderer::new(gpu, desc).expect("renderer");
        let workers = Workers::new(renderer.loading(), 2);
        let showcase = Showcase::load(&mut renderer, gpu);
        let history = renderer.create_view_history().expect("view history");
        let frame = showcase.frame;
        Self {
            showcase,
            history,
            frame,
            _workers: workers,
            renderer,
        }
    }

    /// Renders `frames` frames into a texture of `extent` and returns the last.
    fn run(&mut self, extent: [u32; 2], frames: u64, settings: StillSettings) -> StillRun {
        let gpu = device();
        let frames_in_flight = RendererDesc::default().frames_in_flight;
        let mut average = settings
            .average
            .then(|| Average::new(&mut self.renderer, gpu, frames_in_flight, 16));
        let target = gpu
            .create_texture(
                &TextureDesc::d2(
                    Format::Rgba8Unorm,
                    extent[0],
                    extent[1],
                    TextureUsage::STORAGE | TextureUsage::SAMPLED,
                ),
                TextureMemory::Dedicated,
                "still showcase target",
            )
            .expect("target");
        let listed = [self.showcase.scene];
        let mut last = None;
        let (mut render_extent, mut mode) = ([0; 2], None);
        for i in 0..frames {
            self.frame += 1;
            let orbit = if settings.orbit { frames - 1 - i } else { 0 };
            let eye = orbit_eye(viewer::STILL_ANGLE + orbit as f64 * ORBIT_STEP);
            let mut f = self
                .renderer
                .begin_frame(FrameDesc {
                    time: fixed_time(self.frame),
                    scenes: &listed,
                    display: None,
                })
                .expect("begin frame");
            let id = f.id();
            f.scene(self.showcase.scene)
                .expect("scene writer")
                .set_focus(eye.as_dvec3());
            let output = ViewOutput {
                reconstruction: settings.reconstruction,
                screenshot: (i + 1 == frames)
                    .then_some(ScreenshotRequest::Display { include_ui: false }),
                ..ViewOutput::default()
            };
            let mut sum = average.as_mut().map(|a| a.hook(&f, i, extent));
            let mut overlay = |pass: &mut TransparencyPass<'_, '_>, _| {
                pass.cmd.begin_rendering(&Rendering {
                    extent: pass.extent,
                    colors: &[ColorAttachment {
                        texture: pass.overlay,
                        load: ColorLoad::Clear(ClearColor::Float(
                            settings.overlay.expect("overlay"),
                        )),
                        store: true,
                    }],
                    depth: None,
                });
                pass.cmd.end_rendering();
            };
            let mut hooks = ViewHooks {
                scene_referred: sum
                    .as_mut()
                    .map(|s| s as &mut dyn FnMut(&mut HookPass<'_, '_>)),
                transparency: settings.overlay.is_some().then_some(&mut overlay as _),
                ..ViewHooks::default()
            };
            let rendered = f
                .render_view(
                    &still_view(
                        self.showcase.scene,
                        self.history,
                        eye,
                        i == 0,
                        ViewTarget::Texture(&target),
                    ),
                    &ViewLighting {
                        reference: settings.reference.then(|| ReferenceSettings {
                            lens: Some(ThinLens {
                                aperture_radius: 1e-4,
                                focus_distance: eye.distance(TARGET),
                            }),
                        }),
                        ..ViewLighting::default()
                    },
                    &output,
                    &mut hooks,
                )
                .expect("render view");
            render_extent = rendered.render_extent;
            mode = rendered.reconstruction;
            last = Some((id, f.end().expect("end frame")));
        }
        let (id, point) = last.expect("a frame");
        gpu.queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(30))
            .expect("wait");
        let shot = self
            .renderer
            .screenshots(self.history, id)
            .pop()
            .expect("screenshot");
        StillRun {
            capture: Capture::from_screenshot(&shot),
            render_extent,
            mode,
        }
    }

    /// Adds a sphere simplified as a cluster DAG, whose cut follows the detail asked for; the
    /// showcase's own meshes have one level.
    fn add_dag_sphere(&mut self) {
        const NAME: &str = "dag sphere";
        let mut sphere = MeshBuilder::default();
        sphere.sphere(0, TARGET + Vec3::new(0.0, 1.0, 3.0), 1.0, 96);
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                Target::Memory,
                vec![AssetSpec {
                    name: NAME.into(),
                    source: Source::Mesh {
                        mesh: sphere.finish(),
                        materials: vec![Material::default()],
                        settings: MeshSettings {
                            lod: LodMode::Dag,
                            ..MeshSettings::default()
                        },
                    },
                }],
            )],
        };
        let pack = orr_cook::cook(&manifest, &Default::default())
            .expect("cook")
            .memory
            .remove(0);
        let (mount, frame) = mount_at(&mut self.renderer, PackSource::Memory(pack), self.frame + 1);
        let model = self
            .renderer
            .model(mount, AssetId::from_name(NAME))
            .expect("model");
        self.frame = frame + 1;
        let listed = [self.showcase.scene];
        let f = self
            .renderer
            .begin_frame(FrameDesc {
                time: fixed_time(self.frame),
                scenes: &listed,
                display: None,
            })
            .expect("begin frame");
        let mut w = f.scene(self.showcase.scene).expect("scene writer");
        let group = w
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .expect("transform group");
        w.transforms(group, 0, 1)
            .expect("transforms")
            .fill(Affine3::IDENTITY);
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
        drop(w);
        f.end().expect("end frame");
    }

    /// The raster candidate clusters the scene's traversal chooses for these views of the still
    /// camera, rendered together with their reconstruction requests, and prepared before the
    /// scene update or not.
    fn detail(&mut self, views: &[([u32; 2], Reconstruction)], prepared: bool) -> u32 {
        const FRAMES: u32 = 8;
        let scene = self.showcase.scene;
        let listed = [scene];
        let histories: Vec<ViewHistoryId> = views
            .iter()
            .map(|_| self.renderer.create_view_history().expect("view history"))
            .collect();
        let eye = orbit_eye(viewer::STILL_ANGLE);
        let lighting = ViewLighting::default();
        let mut last = None;
        for i in 0..FRAMES {
            self.frame += 1;
            let mut f = self
                .renderer
                .begin_frame(FrameDesc {
                    time: fixed_time(self.frame),
                    scenes: &listed,
                    display: None,
                })
                .expect("begin frame");
            f.scene(scene)
                .expect("scene writer")
                .set_focus(eye.as_dvec3());
            let lit: Vec<(LitView<'_>, ViewOutput)> = views
                .iter()
                .zip(&histories)
                .map(|(&(extent, reconstruction), &history)| {
                    let target = ViewTarget::Radiance { extent };
                    let view = still_view(scene, history, eye, i == 0, target);
                    let output = ViewOutput {
                        reconstruction,
                        ..ViewOutput::default()
                    };
                    (view, output)
                })
                .collect();
            if prepared {
                for (view, output) in &lit {
                    f.prepare_view(view, &lighting, output).expect("prepare");
                }
                f.update_scenes(&mut |_| {}).expect("update scenes");
            }
            for (view, output) in &lit {
                f.render_view(view, &lighting, output, &mut ViewHooks::default())
                    .expect("render view");
            }
            last = Some(f.end().expect("end frame"));
        }
        device()
            .queue(QueueKind::Render)
            .expect("render queue")
            .wait(last.expect("a frame"), Duration::from_secs(30))
            .expect("wait");
        let detail = self.renderer.stats().cuts.candidate_clusters;
        for history in histories {
            self.renderer.destroy_view_history(history);
        }
        detail
    }
}

/// A scene chooses its geometry for the render size of its sharpest view: upscaling to any
/// output chooses what a native render at the render size does, with the view prepared before
/// the scene update or not, and a second, smaller view changes nothing.
#[test]
fn scene_detail_follows_the_render_size_of_its_sharpest_view() {
    let _serial = serial();
    let mut still = StillShowcase::new();
    still.add_dag_sphere();
    let off = Reconstruction::Off;
    let (small, large) = ([160, 90], [640, 360]);
    // Loads the pages the larger view's cut takes.
    still.detail(&[(large, off)], false);
    let native = still.detail(&[(large, off)], false);
    let coarse = still.detail(&[(small, off)], false);
    assert!(
        native > coarse,
        "clusters: {native} at {large:?}, {coarse} at {small:?}"
    );
    assert_eq!(still.detail(&[(large, off)], true), native);
    let both = [(small, off), (large, off)];
    assert_eq!(still.detail(&both, true), native);
    assert_eq!(still.detail(&both, false), native);
    if enable_dlss(&mut still.renderer) {
        let upscaled = [([960, 540], Reconstruction::Quality(DlssQuality::Quality))];
        assert_eq!(still.detail(&upscaled, false), native);
        assert_eq!(still.detail(&upscaled, true), native);
    } else {
        log::warn!("no upscaled case: NGX offers no ray reconstruction here");
    }
    assert_clean();
}

/// `render_view` through DLSS ray reconstruction on one renderer: each request renders at the
/// size it resolves to, ray reconstruction at half the output size lands close to the converged
/// image on a still and on a moving camera, the scene-referred hook takes its output, and the
/// transparency overlay reaches the image through NGX's composite.
#[test]
fn render_view_runs_ray_reconstruction() {
    let _serial = serial();
    let mut still = StillShowcase::new();
    let hd = [1920, 1080];
    let plain = StillSettings::default();
    let with = |reconstruction| StillSettings {
        reconstruction,
        ..plain
    };

    let without = still.run(hd, 2, plain);
    assert_eq!((without.render_extent, without.mode), (hd, None));

    if !enable_dlss(&mut still.renderer) {
        log::warn!("skipped: NGX offers no ray reconstruction here");
        return;
    }

    for (extent, request, render, mode) in [
        (
            hd,
            Reconstruction::default(),
            [1280, 720],
            Some(DlssMode::Quality),
        ),
        (
            [1280, 720],
            Reconstruction::default(),
            [853, 480],
            Some(DlssMode::Quality),
        ),
        (
            [64, 32],
            Reconstruction::default(),
            [43, 21],
            Some(DlssMode::Quality),
        ),
        (
            hd,
            Reconstruction::Quality(DlssQuality::Dlaa),
            hd,
            Some(DlssMode::Dlaa),
        ),
        (
            hd,
            Reconstruction::RenderExtent([1100, 619]),
            [1100, 619],
            Some(DlssMode::Balanced),
        ),
        (hd, Reconstruction::Off, hd, None),
    ] {
        let run = still.run(extent, 2, with(request));
        assert_eq!((run.render_extent, run.mode), (render, mode), "{request:?}");
    }

    let extent = [640, 360];
    let off = with(Reconstruction::Off);
    let truth = still.run(
        extent,
        256,
        StillSettings {
            average: true,
            reference: true,
            ..off
        },
    );
    let truth = image(truth.capture);
    let error = |run: &StillRun| flip(&truth, &image(run.capture.clone()));
    let noisy = error(&still.run(extent, 8, off));
    let half = with(Reconstruction::Quality(DlssQuality::Performance));
    let denoised = still.run(extent, 48, half);
    assert_eq!(denoised.render_extent, [320, 180]);
    let moving = error(&still.run(
        extent,
        48,
        StillSettings {
            orbit: true,
            ..half
        },
    ));
    let summed = error(&still.run(
        extent,
        48,
        StillSettings {
            average: true,
            ..half
        },
    ));
    let denoised_error = error(&denoised);
    log::info!(
        "against the converged image: noisy {noisy:?}, still {denoised_error:?}, \
         moving {moving:?}, averaged in the hook {summed:?}"
    );
    // Measured mean 0.068 still and 0.065 moving; the jitter sign flipped gives 0.077 and 0.074,
    // and one path per pixel without ray reconstruction 0.119. The rest of the error is the
    // fireflies in the converged image.
    assert!(
        denoised_error.mean < 0.072,
        "noisy {noisy:?}, still {denoised_error:?}"
    );
    assert!(moving.mean < 0.07, "noisy {noisy:?}, moving {moving:?}");
    assert!(
        summed.mean < noisy.mean * 0.75,
        "noisy {noisy:?}, averaged {summed:?}"
    );

    let red_share = |c: &Capture| {
        let (mut red, mut all) = (0.0, 0.0);
        for p in c.rgba8.chunks_exact(4) {
            red += f64::from(p[0]);
            all += f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2]);
        }
        red / all
    };
    let veiled = still.run(
        extent,
        16,
        StillSettings {
            overlay: Some([0.5, 0.0, 0.0, 0.5]),
            ..half
        },
    );
    let (before, after) = (red_share(&denoised.capture), red_share(&veiled.capture));
    log::info!("red share {before:.3} without the overlay, {after:.3} with it");
    assert!(
        after > before + 0.1,
        "red share {before:.3}, then {after:.3}"
    );

    drop(still);
    device().collect_garbage();
    assert_clean();
}
