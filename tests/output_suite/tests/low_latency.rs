//! Reflex through `Display::set_low_latency` on a real window: low latency on with boost,
//! then off, then on without boost. In every mode each frame id gets one complete, ordered
//! report, with markers from the render thread and the display's present thread.
#![cfg(windows)]

use std::collections::BTreeMap;
use std::time::Duration;

use gpu_suite::check_latency_reports;
use gpu_suite::window::Window;
use orr::{ConsumerMarker, Display, DisplayRequest, FrameDesc, FrameTime, Renderer, RendererDesc};
use orr_scene::testing::{assert_clean, device, wait};

const FRAMES: u64 = 1200;

#[test]
fn reports_stay_complete_as_low_latency_changes() {
    if !device().caps().low_latency {
        log::warn!("no VK_NV_low_latency2; skipped");
        return;
    }
    let mut window = Window::new("orr low latency", 640, 360);
    let desc = RendererDesc::default();
    let mut display = Display::new(
        device(),
        window.window_handle(),
        window.display_handle(),
        &DisplayRequest {
            vsync: false,
            ..DisplayRequest::default()
        },
        None,
        desc.frames_in_flight,
    )
    .unwrap();

    let render_thread = std::thread::spawn(move || {
        let mut r = Renderer::new(device(), desc).unwrap();
        let mut reports = BTreeMap::new();
        let mut out = Vec::new();
        let mut last = None;
        for i in 0..FRAMES {
            match i {
                0 => display.set_low_latency(true, true),
                400 => display.set_low_latency(false, false),
                800 => display.set_low_latency(true, false),
                _ => {}
            }
            let mut f = r
                .begin_frame(FrameDesc {
                    time: FrameTime {
                        absolute: i as f64 / 60.0,
                        delta: 1.0 / 60.0,
                    },
                    scenes: &[],
                    display: Some(&mut display),
                })
                .unwrap();
            f.latency_marker(ConsumerMarker::SimulationStart);
            f.latency_marker(ConsumerMarker::SimulationEnd);
            f.update_scenes(&mut |_| {}).unwrap();
            let id = f.id();
            last = Some((id, f.end().unwrap()));
            if id.is_multiple_of(16) {
                out.clear();
                display.latency().unwrap().timings(&mut out);
                for t in &out {
                    reports.insert(t.frame_id, *t);
                }
            }
        }
        let (id, point) = last.unwrap();
        display.wait_idle();
        wait(point);
        (id, reports)
    });
    while !render_thread.is_finished() {
        window.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
    let (last, reports) = render_thread.join().unwrap();
    check_latency_reports(&reports, last);
    device().collect_garbage();
    assert_clean();
}
