//! A cooked scene path traced by orr: a gold knot, a glossy and a mirror sphere, a block and
//! a glowing panel on a ground, under a cooked sky and a 100 klx sun. Each frame traces one
//! path per pixel, auto exposure meters and adapts to the result, and on Windows an egui
//! overlay shows the exposure. The camera orbits the scene; a still camera averages its
//! frames in a scene-referred hook.
//!
//! `viewer [--still] [--frames N] [--headless] [--out file.png]`. Headless runs keep the
//! camera still, render 480 x 270 for 120 frames unless told otherwise and write the last
//! frame as a PNG; windowed runs with `--out` write their last frame as the window shows
//! it, overlay included.

use std::time::{Duration, Instant};

use examples::overlay::Overlay;
use examples::scene::{FOV_Y, Showcase, TARGET, orbit_eye};
use examples::shaders::accumulate;
use examples::window::Window;
use examples::{Capture, Options, Workers, assert_clean, device, fixed_time};
use orr::orr_gpu::{
    Allocation, Format, MemoryKind, QueueKind, SpecData, TextureDesc, TextureMemory, TextureUsage,
    TimelinePoint,
};
use orr::{
    Display, DisplayRequest, ExposureTelemetry, FrameDesc, FrameTime, HookPass, Lens, Projection,
    Renderer, RendererDesc, ResolvedDisplay, ScreenshotRequest, TextureFilter, ViewDesc, ViewHooks,
    ViewLighting, ViewOutput, ViewTarget,
};
use orr_math::camera::rh::view;
use orr_math::{UVec2, Vec2, Vec3};

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
/// Radians per second.
const ORBIT_RATE: f64 = 0.2;
/// Where a still camera stands on the orbit, radians.
const STILL_ANGLE: f64 = 1.3;
/// Frames a still camera renders before it starts averaging, while exposure settles.
const ACCUMULATE_AFTER: u64 = 30;

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("viewer");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// Runs the example; headless runs and runs with `--out` return their last frame.
pub(crate) fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let mut window = (!options.headless).then(|| Window::new("orr viewer", 1280, 720));
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
    let frame = showcase.frame;
    let listed = [scene];

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
            "viewer target",
        )
        .expect("headless target")
    });
    let mut overlay = Overlay::new(desc.frames_in_flight);
    let capture = options.headless || options.out.is_some();
    let still = options.headless || options.has("--still");
    let accumulate = renderer
        .compute_pipeline(&accumulate::MODULE, "main", SpecData::default())
        .expect("accumulate pipeline");
    let slots = u64::from(desc.frames_in_flight);
    let mut roots = gpu
        .gpu_malloc::<accumulate::Root>(MemoryKind::Upload, slots, "viewer accumulate roots")
        .expect("accumulate roots");
    let mut sum: Option<(Allocation<[f32; 4]>, [u32; 2])> = None;
    let mut accumulated = 0;
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
            0.9 + t * ORBIT_RATE
        };
        let eye = orbit_eye(angle);

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
                let text = readout(extent, time, &resolved, telemetry.as_ref());
                overlay.layout(extent, time, &text);
                (extent, ViewTarget::Display)
            }
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
            screenshot: (capture && final_frame).then_some(ScreenshotRequest::Display {
                include_ui: !options.headless,
            }),
            ..ViewOutput::default()
        };
        if sum.as_ref().is_none_or(|(_, e)| *e != extent) {
            let texels = u64::from(extent[0]) * u64::from(extent[1]);
            let buffer = gpu
                .gpu_malloc(MemoryKind::DeviceLocal, texels, "viewer radiance sum")
                .expect("radiance sum");
            sum = Some((buffer, extent));
            accumulated = 0;
        }
        let sum_ptr = sum.as_ref().expect("radiance sum").0.gpu;
        let frames_in_sum = if i < ACCUMULATE_AFTER { 0 } else { accumulated };
        let pipeline = f.pipeline(accumulate);
        let slot = id % slots;
        let mut average = |pass: &mut HookPass<'_, '_>| {
            roots.as_mut_slice().expect("accumulate roots")[slot as usize] = accumulate::Root {
                view: pass.view,
                sum: sum_ptr,
                image: pass.image,
                width: pass.extent[0],
                height: pass.extent[1],
                frames: frames_in_sum,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            pass.cmd.begin_pass("accumulate", false);
            pass.cmd.dispatch(
                pipeline,
                roots.gpu.offset(slot),
                [pass.extent[0].div_ceil(8), pass.extent[1].div_ceil(8), 1],
            );
            pass.cmd.end_pass();
        };
        let mut ui = |pass: &mut orr::UiPass<'_, '_>| overlay.paint(pass, id);
        let mut hooks = ViewHooks {
            scene_referred: still.then_some(&mut average as &mut dyn FnMut(&mut HookPass<'_, '_>)),
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
                target,
            },
            &ViewLighting::default(),
            &output,
            &mut hooks,
        )
        .expect("render view");
        last = Some((id, f.end().expect("end frame")));
        if i >= ACCUMULATE_AFTER {
            accumulated += 1;
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
    let captured = last
        .filter(|_| capture)
        .and_then(|(id, _)| renderer.screenshots(history, id).pop())
        .map(|shot| Capture::from_screenshot(&shot));
    drop(sum);
    drop(roots);
    drop(overlay);
    drop(headless_target);
    drop(showcase);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}

/// The exposure readout.
fn readout(
    [w, h]: [u32; 2],
    time: FrameTime,
    resolved: &ResolvedDisplay,
    telemetry: Option<&ExposureTelemetry>,
) -> String {
    match telemetry {
        Some(t) => format!(
            "{w} x {h}  {:?}  paper white {:.0} nits\n\
             EV100 {:.2} (target {:.2})  pre-exposure {:.3e}\n\
             frame time {:.1} ms",
            resolved.encoding,
            resolved.paper_white_nits,
            t.adapted_ev100,
            t.target_ev100,
            t.pre_exposure,
            time.delta * 1000.0,
        ),
        None => format!("{w} x {h}  {:?}", resolved.encoding),
    }
}
