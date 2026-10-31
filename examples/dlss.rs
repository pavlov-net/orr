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

use std::time::{Duration, Instant};

use examples::overlay::Overlay;
use examples::scene::{FOV_Y, Showcase, TARGET, orbit_eye};
use examples::window::Window;
use examples::{Options, PROJECT_ID, Workers, assert_clean, device, fixed_time};
use orr::orr_gpu::{OutputFormat, QueueKind, TimelinePoint};
use orr::{
    Display, DisplayRequest, DlssSdk, FrameDesc, FrameGenerationMode, FrameTime, Lens, Projection,
    Renderer, RendererDesc, ViewDesc, ViewHooks, ViewLighting, ViewOutput, ViewTarget,
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
    run(&options);
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
