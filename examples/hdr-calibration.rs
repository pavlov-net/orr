//! Calibration patterns through orr's frame on a real window, with the display watch's
//! feed: for each display request in turn (follow the OS, SDR, PQ, PQ at a requested
//! calibration, scRGB) it shows the peak, paper white and black patterns at the resolved
//! luminance, with egui text naming the request, the resolved display and what the OS
//! reports. The window changes size with each request. Toggling Windows HDR, moving the SDR
//! content brightness slider or moving the window to another monitor re-resolves on the next
//! frame.
//!
//! `hdr-calibration [frames per pattern] [--ignore-os]`: 120 frames per pattern by default.
//! `--ignore-os` leaves the watch's feed out of the display, so the surface's formats alone
//! decide and PQ and scRGB present even with Windows HDR off.

#[cfg(not(windows))]
fn main() {}

#[cfg(windows)]
fn main() {
    run::main();
}

#[cfg(windows)]
mod run {
    use std::time::{Duration, Instant};

    use egui::{Color32, Context, FullOutput, Pos2, RawInput, Rect, RichText, TexturesDelta, vec2};
    use examples::window::Window;
    use examples::{assert_clean, device};
    use orr::orr_gpu::{OutputFormat, QueueKind};
    use orr::{
        CalibrationPattern, Display, DisplayRequest, DisplayWatch, FrameDesc, FrameTime, Renderer,
        RendererDesc, ResolvedDisplay,
    };
    use orr_egui::Painter;

    /// The requests the script steps through, with their labels.
    fn requests() -> [(&'static str, DisplayRequest); 5] {
        let encoding = |e| DisplayRequest {
            encoding: Some(e),
            ..DisplayRequest::default()
        };
        [
            ("follow the OS", DisplayRequest::default()),
            ("SDR", encoding(OutputFormat::Sdr)),
            ("PQ", encoding(OutputFormat::Hdr10Pq)),
            (
                "PQ at 200 / 600 / 0.01 nits",
                DisplayRequest {
                    paper_white_nits: Some(200.0),
                    peak_nits: Some(600.0),
                    black_nits: Some(0.01),
                    ..encoding(OutputFormat::Hdr10Pq)
                },
            ),
            ("scRGB", encoding(OutputFormat::ScRgb)),
        ]
    }

    /// Pattern `i` of the three at `display`'s resolved luminance. The black patch sits just
    /// above the resolved black, where a display that reaches it shows the patch.
    fn pattern_at(i: usize, display: &ResolvedDisplay) -> CalibrationPattern {
        match i {
            0 => CalibrationPattern::Peak {
                nits: display.peak_nits,
            },
            1 => CalibrationPattern::PaperWhite {
                nits: display.paper_white_nits,
            },
            _ => CalibrationPattern::Black {
                nits: display.black_nits + 0.05,
            },
        }
    }

    pub(super) fn main() {
        let mut args = std::env::args().skip(1);
        let mut frames_per_pattern = 120u64;
        let mut ignore_os = false;
        for arg in args.by_ref() {
            match arg.as_str() {
                "--ignore-os" => ignore_os = true,
                n => frames_per_pattern = n.parse().expect("frames per pattern"),
            }
        }

        let gpu = device();
        let mut window = Window::new("orr hdr calibration", 1280, 720);
        let (_watch, feed) = DisplayWatch::new(window.window_handle()).expect("display watch");
        let desc = RendererDesc::default();
        let requests = requests();
        let mut display = Display::new(
            gpu,
            window.window_handle(),
            window.display_handle(),
            &requests[0].1,
            (!ignore_os).then(|| feed.clone()),
            desc.frames_in_flight,
        )
        .expect("display");
        let mut renderer = Renderer::new(gpu, desc).expect("renderer");
        let mut painter = Painter::new(gpu, desc.frames_in_flight).expect("painter");
        let ctx = Context::default();
        let mut pending = TexturesDelta::default();
        let start = Instant::now();
        let mut previous = 0.0;
        let mut last = None;
        let mut logged = None;

        let steps = requests.len() as u64 * 3 * frames_per_pattern;
        'frames: for i in 0..steps {
            if !window.pump() {
                break;
            }
            let (step, which) = (
                (i / (3 * frames_per_pattern)) as usize,
                (i / frames_per_pattern % 3) as usize,
            );
            let (label, request) = requests[step];
            if *display.request() != request {
                display.set_request(&request);
                let size = if step % 2 == 0 {
                    [1280, 720]
                } else {
                    [960, 600]
                };
                window.resize(size[0], size[1]);
            }
            let absolute = start.elapsed().as_secs_f64();
            let time = FrameTime {
                absolute,
                delta: (absolute - previous) as f32,
            };
            previous = absolute;

            let mut f = renderer
                .begin_frame(FrameDesc {
                    time,
                    scenes: &[],
                    display: Some(&mut display),
                })
                .expect("begin frame");
            let id = f.id();
            let Some((resolved, [w, h])) = f.display() else {
                last = Some(f.end().expect("end frame"));
                continue 'frames;
            };
            let info = feed.latest().1;
            if logged != Some((step, resolved)) {
                log::info!("{label}: {resolved:?}; OS {info:?}");
                logged = Some((step, resolved));
            }
            let pattern = pattern_at(which, &resolved);
            f.pattern(pattern).expect("pattern");

            let FullOutput {
                textures_delta,
                shapes,
                pixels_per_point,
                ..
            } = ctx.run_ui(
                RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(w as f32, h as f32))),
                    time: Some(time.absolute),
                    predicted_dt: time.delta,
                    ..RawInput::default()
                },
                |ui| {
                    let text = format!(
                        "request: {label}{}\n\
                         {:?}  paper white {:.0}  peak {:.0}  black {:.3} nits\n\
                         OS: HDR {:?}, SDR white {:?}, max {:?}, min {:?}, {:?} Hz\n\
                         {pattern:?}",
                        if ignore_os { " (OS state ignored)" } else { "" },
                        resolved.encoding,
                        resolved.paper_white_nits,
                        resolved.peak_nits,
                        resolved.black_nits,
                        info.os_hdr_on,
                        info.sdr_white_nits,
                        info.max_nits,
                        info.min_nits,
                        info.refresh_hz,
                    );
                    egui::Frame::new()
                        .fill(Color32::from_black_alpha(220))
                        .inner_margin(8)
                        .show(ui, |ui| {
                            ui.label(RichText::new(text).size(18.0).color(Color32::WHITE));
                        });
                },
            );
            let primitives = ctx.tessellate(shapes, pixels_per_point);
            pending.append(textures_delta);
            f.ui_only(&mut |pass| {
                painter
                    .update_textures(pass.cmd, id, std::mem::take(&mut pending))
                    .expect("egui textures");
                painter
                    .paint(pass.cmd, id, pass.layer, &primitives, pixels_per_point)
                    .expect("egui paint");
            })
            .expect("ui");
            last = Some(f.end().expect("end frame"));

            // The first frame of each pattern presents at the encoding and extent it resolved.
            if i % frames_per_pattern == 0 {
                display.wait_idle();
                let p = display.last_presented().expect("presented");
                assert_eq!(p.frame, id);
                assert_eq!(p.encoding, resolved.encoding, "{label}");
                assert_eq!(p.extent, [w, h]);
            }
        }

        display.wait_idle();
        if let Some(point) = last {
            gpu.queue(QueueKind::Render)
                .expect("render queue")
                .wait(point, Duration::from_secs(10))
                .expect("wait");
        }
        drop(display);
        drop(painter);
        drop(renderer);
        gpu.collect_garbage();
        assert_clean();
        log::info!("validation clean");
    }
}
