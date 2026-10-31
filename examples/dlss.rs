//! The showcase scene path traced in a window with DLSS frame generation: the camera orbits
//! while the display presents generated images between rendered frames, with an egui
//! overlay naming the mode, the images generated per frame and the encoding, and a dot
//! crossing the bottom of the window at constant speed. The overlay is composed over every
//! presented image, so it never moves with the interpolation; with `--ui-per-present` the UI
//! is drawn once per presented image, so the dot advances on every presented image instead
//! of every rendered frame.
//!
//! `dlss [--frames N] [--off | --2x | --3x | --4x] [--cycle] [--ui-per-present]`. The
//! default is NGX's recommended mode (4x Dynamic on RTX 50). `--cycle` steps every few
//! seconds through 2x, 4x, Dynamic and off, with SDR, PQ and scRGB requests, to show frame
//! generation turning on and off at run time. Without frame generation the scene renders
//! without it.
//!
//! `dlss --sr [--dlaa | --quality | --balanced | --performance | --ultra-performance]
//! [--native] [--clusters] [--headless] [--frames N] [--out file.png]` shows DLSS super
//! resolution instead: the same scene as a raster debug view, its G-buffer shaded by the
//! sun and sky alone (or colored by cluster with `--clusters`), rendered at the mode's
//! render size (Quality by default) and upscaled to the window. `--native` renders it at
//! the output size without SR. Headless runs render 960 x 540 for 64 frames unless told
//! otherwise and write the last frame as a PNG.

use std::time::{Duration, Instant};

use examples::overlay::Overlay;
use examples::scene::{FOV_Y, SUN, Showcase, TARGET, orbit_eye};
use examples::shaders::{capture, sun_shadow};
use examples::window::Window;
use examples::{Capture, Options, PROJECT_ID, Workers, assert_clean, device, fixed_time};
use orr::orr_gpu::{
    Format, Hazards, MemoryKind, OutputFormat, QueueKind, SpecData, Stages, TimelinePoint,
};
use orr::orr_output::Image;
use orr::orr_output::orr_dlss::raw::Ngx;
use orr::orr_output::orr_dlss::{Quality, SrInputs, SuperResolution};
use orr::{
    Display, DisplayRequest, DlssSdk, FrameDesc, FrameGenerationMode, FrameTime, Lens, Projection,
    Renderer, RendererDesc, TextureFilter, ViewDesc, ViewHooks, ViewLighting, ViewOutput,
    ViewTarget,
};
use orr_math::camera::rh::view;
use orr_math::{UVec2, Vec2, Vec3};

/// Radians per second.
const ORBIT_RATE: f64 = 0.4;
/// Seconds per step of `--cycle`.
const CYCLE_STEP: f64 = 4.0;
/// The dot's speed in points per second.
const CURSOR_SPEED: f64 = 400.0;

fn main() {
    let options = Options::from_args();
    if options.has("--sr") {
        let run = run_sr(&options);
        if let Some(p) = run.as_ref().and_then(|run| run.preset) {
            log::info!("SR preset {p}");
        }
        if let Some(c) = run.and_then(|run| run.capture) {
            let path = options.out_path("dlss-sr");
            c.save_png(&path);
            log::info!("wrote {}", path.display());
        }
    } else {
        run(&options);
    }
    assert_clean();
}

/// What `--cycle` steps through: a mode and an encoding request.
fn cycle(
    step: usize,
    recommended: FrameGenerationMode,
    most: u32,
) -> (FrameGenerationMode, OutputFormat) {
    use FrameGenerationMode::{Fixed, Off};
    use OutputFormat::{Hdr10Pq, ScRgb, Sdr};
    let steps = [
        (Fixed(1), Sdr),
        (Fixed(most), Hdr10Pq),
        (recommended, ScRgb),
        (Off, ScRgb),
        (Fixed(1), Hdr10Pq),
        (Off, Sdr),
    ];
    steps[step % steps.len()]
}

fn run(options: &Options) {
    let gpu = device();
    let mut window = Window::new("orr dlss", 1600, 900);
    let desc = RendererDesc::default();
    let request = |encoding: Option<OutputFormat>| DisplayRequest {
        encoding,
        vsync: false,
        ..DisplayRequest::default()
    };
    let mut display = Display::new(
        gpu,
        window.window_handle(),
        window.display_handle(),
        &request(None),
        None,
        desc.frames_in_flight,
    )
    .expect("display");
    let sdk = DlssSdk::new(gpu, PROJECT_ID)
        .inspect_err(|e| log::warn!("no DLSS: {e}"))
        .ok();
    let support = sdk.as_ref().map(DlssSdk::support).unwrap_or_default();
    let recommended = FrameGenerationMode::recommended(&support);
    let most = support.max_generated.clamp(1, 3);
    let chosen = if options.has("--off") {
        FrameGenerationMode::Off
    } else if let Some(n) = [("--2x", 1), ("--3x", 2), ("--4x", 3)]
        .into_iter()
        .find(|(flag, _)| options.has(flag))
        .map(|(_, n)| n)
    {
        FrameGenerationMode::Fixed(n.min(support.max_generated.max(1)))
    } else {
        recommended
    };
    let cycling = options.has("--cycle");
    let set_mode = |display: &mut Display, mode| {
        if let Some(sdk) = &sdk
            && let Err(e) = display.set_frame_generation(sdk, mode)
        {
            log::warn!("frame generation {mode:?}: {e}");
        }
    };
    if !cycling {
        set_mode(&mut display, chosen);
    }
    display.set_ui_per_present(options.has("--ui-per-present"));

    let mut renderer = Renderer::new(gpu, desc).expect("renderer");
    let _workers = Workers::new(renderer.loading(), 2);
    let showcase = Showcase::load(&mut renderer, gpu);
    let (scene, history) = (showcase.scene, showcase.history);
    let listed = [scene];
    let mut overlay = Overlay::new(desc.frames_in_flight);
    let started = Instant::now();
    let mut previous = 0.0;
    let mut step = None;
    let mut last: Option<TimelinePoint> = None;
    let mut counted = 0;
    let mut generated = 0;
    let mut frames = 0;
    for i in 0.. {
        if options.frames.is_some_and(|n| i >= n) || !window.pump() {
            break;
        }
        let t = started.elapsed().as_secs_f64();
        if cycling {
            let now = (t / CYCLE_STEP) as usize;
            if step != Some(now) {
                let (mode, encoding) = cycle(now, recommended, most);
                log::info!("frame generation {mode:?}, {encoding:?} requested");
                display.set_request(&request(Some(encoding)));
                set_mode(&mut display, mode);
                step = Some(now);
            }
        }
        let time = FrameTime {
            absolute: fixed_time(showcase.frame + 1).absolute + t,
            delta: (t - previous) as f32,
        };
        previous = t;
        let eye = orbit_eye(0.9 + t * ORBIT_RATE);
        let mode = display.frame_generation_mode();
        let presented = display.last_presented();
        if let Some(p) = presented.filter(|p| p.frame > counted) {
            counted = p.frame;
            generated += p.generated;
        }
        frames = i + 1;

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: Some(&mut display),
            })
            .expect("begin frame");
        let id = f.id();
        let Some((resolved, extent)) = f.display() else {
            last = Some(f.end().expect("end frame"));
            continue;
        };
        let text = format!(
            "{} x {}  {:?}\nframe generation {mode:?}\n{} generated per frame\nframe time {:.1} ms",
            extent[0],
            extent[1],
            resolved.encoding,
            presented.map_or(0, |p| p.generated),
            time.delta * 1000.0,
        );
        overlay.layout(extent, time, &text);
        f.scene(scene)
            .expect("scene writer")
            .set_focus(eye.as_dvec3(), extent[1] as f32 / FOV_Y);
        let width = f64::from(extent[0]);
        let mut ui = |pass: &mut orr::UiPass<'_, '_>| {
            overlay.paint(pass, id);
            let x = ((t + f64::from(pass.time_offset)) * CURSOR_SPEED).rem_euclid(width);
            overlay.paint_cursor(pass, id, x as f32);
        };
        let mut hooks = ViewHooks {
            ui: Some(&mut ui),
            ..ViewHooks::default()
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
                target: ViewTarget::Display,
            },
            &ViewLighting::default(),
            &ViewOutput::default(),
            &mut hooks,
        )
        .expect("render view");
        last = Some(f.end().expect("end frame"));
    }

    display.wait_idle();
    log::info!(
        "{frames} frames rendered; {generated} generated images presented with the frames seen"
    );
    if let Some(point) = last {
        gpu.queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(10))
            .expect("wait");
    }
    drop(overlay);
    drop(showcase);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
}

const SR_HEADLESS_EXTENT: [u32; 2] = [960, 540];
const SR_HEADLESS_FRAMES: u64 = 64;
/// Sun illuminance and sky radiance of the debug shading, relative to paper white.
const SR_SUN_ILLUMINANCE: f32 = 3.0;
const SR_SKY: [f32; 3] = [0.18, 0.24, 0.34];

/// What a `--sr` run showed.
pub(crate) struct SrRun {
    /// The last frame of a headless run.
    pub capture: Option<Capture>,
    /// The SR preset NGX applied, from its log; `None` with `--native`.
    pub preset: Option<char>,
}

/// The SR mode the flags name; Quality without one.
fn sr_quality(options: &Options) -> Quality {
    [
        ("--dlaa", Quality::Dlaa),
        ("--balanced", Quality::Balanced),
        ("--performance", Quality::Performance),
        ("--ultra-performance", Quality::UltraPerformance),
    ]
    .into_iter()
    .find(|(flag, _)| options.has(flag))
    .map_or(Quality::Quality, |(_, q)| q)
}

/// `image`, remade at `extent` when it holds none or one of another size; true when made.
fn sized(image: &mut Option<Image>, extent: [u32; 2], name: &str) -> bool {
    if image
        .as_ref()
        .is_some_and(|im| im.texture().desc().extent[..2] == extent)
    {
        return false;
    }
    *image = Some(Image::new(device(), Format::Rgba16Float, extent, name).expect("image"));
    true
}

/// Runs `--sr`. Returns `None` when SR is wanted and NGX does not offer it.
pub(crate) fn run_sr(options: &Options) -> Option<SrRun> {
    let gpu = device();
    let native = options.has("--native");
    let quality = sr_quality(options);
    let mode = u32::from(options.has("--clusters"));
    let sdk = if native {
        None
    } else {
        match DlssSdk::new(gpu, PROJECT_ID) {
            Ok(sdk) if sdk.support().sr => Some(sdk),
            Ok(_) => {
                log::warn!("NGX offers no super resolution on this device");
                return None;
            }
            Err(e) => {
                log::warn!("no DLSS: {e}");
                return None;
            }
        }
    };
    let mut window = (!options.headless).then(|| Window::new("orr dlss sr", 1600, 900));
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
    let showcase = Showcase::load(&mut renderer, gpu);
    let (scene, history) = (showcase.scene, showcase.history);
    let listed = [scene];

    let shade = renderer
        .compute_pipeline(&sun_shadow::MODULE, "main", SpecData::default())
        .expect("shade pipeline");
    let encode = renderer
        .compute_pipeline(&capture::MODULE, "main", SpecData::default())
        .expect("capture pipeline");
    let slots = u64::from(desc.frames_in_flight);
    let mut roots = gpu
        .gpu_malloc::<sun_shadow::Root>(MemoryKind::Upload, slots, "dlss sr roots")
        .expect("roots");
    let mut capture_root = gpu
        .gpu_malloc::<capture::Root>(MemoryKind::Upload, 1, "dlss sr capture root")
        .expect("capture root");
    let pixels = u64::from(SR_HEADLESS_EXTENT[0] * SR_HEADLESS_EXTENT[1]);
    let readback = options.headless.then(|| {
        gpu.gpu_malloc::<u32>(MemoryKind::Readback, pixels, "dlss sr capture")
            .expect("readback")
    });

    let frames = options
        .frames
        .or(options.headless.then_some(SR_HEADLESS_FRAMES));
    let mut sr: Option<SuperResolution> = None;
    let mut preset = None;
    let mut shaded: Option<Image> = None;
    let mut upscaled: Option<Image> = None;
    let started = Instant::now();
    let mut previous = 0.0;
    let mut last: Option<TimelinePoint> = None;
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
            absolute: fixed_time(showcase.frame + 1).absolute + t,
            delta: (t - previous) as f32,
        };
        previous = t;
        let eye = orbit_eye(0.9 + t * ORBIT_RATE * 0.5);

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: display.as_mut(),
            })
            .expect("begin frame");
        let extent = if options.headless {
            SR_HEADLESS_EXTENT
        } else {
            match f.display() {
                Some((_, extent)) => extent,
                None => {
                    last = Some(f.end().expect("end frame"));
                    continue;
                }
            }
        };
        let mut reset = i == 0;
        if let Some(sdk) = &sdk
            && sr.as_ref().is_none_or(|s| s.output_size() != extent)
        {
            Ngx::take_log();
            let made = SuperResolution::new(sdk, extent, quality, None).expect("SR feature");
            preset = made.applied_preset(&Ngx::take_log());
            log::info!("SR {quality:?}: {:?} -> {extent:?}", made.render_size());
            sr = Some(made);
            reset = true;
        }
        let (render, jitter, mip_bias) = match &sr {
            Some(s) => (s.render_size(), s.jitter(f.id()), s.mip_bias()),
            None => (extent, [0.0; 2], 0.0),
        };
        f.scene(scene)
            .expect("scene writer")
            .set_focus(eye.as_dvec3(), render[1] as f32 / FOV_Y);
        let view = f
            .raster_view(&ViewDesc {
                scene,
                history,
                position: eye.as_dvec3(),
                rotation: view::look_at_quat(eye, TARGET, Vec3::Y).inverse(),
                projection: Projection {
                    fov_y: FOV_Y,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: UVec2::from(render),
                jitter: Vec2::from(jitter),
                reset_history: reset,
                mip_bias,
                texture_filter: TextureFilter::Hardware,
                target: ViewTarget::AfterRr,
            })
            .expect("raster view");

        let fresh = sized(&mut shaded, render, "dlss sr shaded");
        let target = shaded.as_ref().expect("shaded image");
        let slot = f.id() % slots;
        roots.as_mut_slice().expect("roots")[slot as usize] = sun_shadow::Root {
            scene: view.scene,
            view: view.view,
            gbuffer: view.gbuffer,
            target: target.storage,
            mode,
            sun: [SUN.x, SUN.y, SUN.z, SR_SUN_ILLUMINANCE],
            sky: [SR_SKY[0], SR_SKY[1], SR_SKY[2], 1.0],
        };
        let (shade, encode) = (f.pipeline(shade), f.pipeline(encode));
        let list = f.list().expect("frame list");
        if fresh {
            list.discard(target.texture());
        }
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.begin_pass("debug shading", false);
        list.dispatch(
            shade,
            roots.gpu.offset(slot),
            [render[0].div_ceil(8), render[1].div_ceil(8), 1],
        );
        list.end_pass();

        let list = f.output_list().expect("output list");
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        let shown = match sr.as_mut() {
            Some(s) => {
                let fresh = sized(&mut upscaled, extent, "dlss sr output");
                let out = upscaled.as_ref().expect("SR output");
                if fresh {
                    list.discard(out.texture());
                }
                s.record(
                    list,
                    &SrInputs {
                        color: target.texture(),
                        depth: &view.depth,
                        motion: &view.motion,
                        exposure: None,
                        pre_exposure: 1.0,
                        output: out.texture(),
                        reset,
                        jitter,
                    },
                )
                .expect("SR evaluate");
                list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
                out.sampled
            }
            None => target.sampled,
        };
        if let Some(out) = readback.as_ref().filter(|_| final_frame) {
            capture_root.as_mut_slice().expect("capture root")[0] = capture::Root {
                pixels: out.gpu,
                source: shown,
                width: extent[0],
                height: extent[1],
                pad0: 0,
                pad1: 0,
            };
            list.dispatch(
                encode,
                capture_root.gpu,
                [extent[0].div_ceil(8), extent[1].div_ceil(8), 1],
            );
            list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        }
        f.compose(shown).expect("compose");
        last = Some(f.end().expect("end frame"));
    }

    if let Some(d) = display.as_ref() {
        d.wait_idle();
    }
    if let Some(point) = last {
        gpu.queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(10))
            .expect("wait");
    }
    let capture = readback.filter(|_| last.is_some()).map(|out| Capture {
        width: SR_HEADLESS_EXTENT[0],
        height: SR_HEADLESS_EXTENT[1],
        rgba8: out
            .as_slice()
            .expect("readback")
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    });
    drop(sr);
    drop(showcase);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    Some(SrRun { capture, preset })
}
