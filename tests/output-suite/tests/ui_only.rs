//! UI-only frames through `orr::Renderer` on a real window: the egui demo windows drawn by
//! `Frame::ui_only` with orr-egui's painter, presented over black by the display's present
//! thread at SDR and then PQ, between frames that draw no UI and present black.
#![cfg(windows)]

use egui::{Context, FullOutput, Pos2, RawInput, Rect, TexturesDelta, vec2};
use gpu_suite::window::Window;
use orr::{Display, DisplayRequest, FrameDesc, FrameTime, Renderer, RendererDesc};
use orr_egui::Painter;
use orr_output::OutputFormat;
use orr_scene::testing::{assert_clean, device, wait};

const FRAMES: u64 = 120;

#[test]
fn ui_only_frames_present_egui_over_black() {
    let gpu = device();
    let mut window = Window::new("orr ui only", 1280, 800);
    let desc = RendererDesc::default();
    let sdr = DisplayRequest {
        encoding: Some(OutputFormat::Sdr),
        ..DisplayRequest::default()
    };
    let mut display = Display::new(
        gpu,
        window.window_handle(),
        window.display_handle(),
        &sdr,
        None,
        desc.frames_in_flight,
    )
    .unwrap();
    let mut renderer = Renderer::new(gpu, desc).unwrap();
    let mut painter = Painter::new(gpu, desc.frames_in_flight).unwrap();
    let ctx = Context::default();
    let mut demo = egui_demo_lib::DemoWindows::default();
    let [w, h] = window.client_size();
    let mut last_present_id = 0;
    let mut last = None;
    let mut pending = TexturesDelta::default();
    for i in 0..FRAMES {
        window.pump();
        if i == FRAMES / 2 {
            display.set_request(&DisplayRequest {
                encoding: Some(OutputFormat::Hdr10Pq),
                ..sdr
            });
        }
        let time = FrameTime {
            absolute: i as f64 / 60.0,
            delta: 1.0 / 60.0,
        };
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
            |ui| demo.ui(ui),
        );
        let primitives = ctx.tessellate(shapes, pixels_per_point);
        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &[],
                display: Some(&mut display),
            })
            .unwrap();
        let id = f.id();
        // Every fourth frame draws no UI and presents black; egui's texture changes wait for
        // the next frame that paints.
        pending.append(textures_delta);
        if i % 4 != 2 {
            f.ui_only(&mut |pass| {
                painter
                    .update_textures(pass.cmd, id, std::mem::take(&mut pending))
                    .unwrap();
                painter
                    .paint(pass.cmd, id, pass.layer, &primitives, pixels_per_point)
                    .unwrap();
            })
            .unwrap();
        }
        last = Some(f.end().unwrap());
        if i % 8 == 7 || i == FRAMES / 2 + 1 {
            display.wait_idle();
            let p = display.last_presented().unwrap();
            assert_eq!(p.frame, id);
            let present_id = p.present_id.expect("presented");
            assert!(present_id > last_present_id, "present ids increase");
            last_present_id = present_id;
            assert_eq!(p.extent, [w, h]);
            let encoding = if i < FRAMES / 2 {
                OutputFormat::Sdr
            } else {
                OutputFormat::Hdr10Pq
            };
            assert_eq!(p.encoding, encoding, "frame {id}");
        }
    }
    display.wait_idle();
    wait(last.unwrap());
    drop(display);
    gpu.collect_garbage();
    assert_clean();
}
