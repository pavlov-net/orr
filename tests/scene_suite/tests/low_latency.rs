//! `orr::Renderer` frames presented to a window through its `Display`, with low latency on:
//! the render thread sleeps and sets the simulation and submit markers, the display's present
//! thread the present markers.
#![cfg(windows)]

use std::collections::BTreeMap;
use std::time::Duration;

use glam::DVec3;
use gpu_suite::check_latency_reports;
use gpu_suite::window::Window;
use orr::{
    ConsumerMarker, Display, DisplayRequest, FrameDesc, FrameTime, Renderer, RendererDesc,
    SceneDesc,
};
use orr_scene::testing::{assert_clean, device, wait};

const FRAMES: u64 = 1000;

#[test]
fn every_frame_id_gets_a_complete_ordered_report() {
    if !device().caps().low_latency {
        log::warn!("no VK_NV_low_latency2; skipped");
        return;
    }
    let mut window = Window::new("frame latency", 640, 360);
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
    display
        .latency()
        .expect("low latency")
        .set_mode(true, false, 0);

    let render_thread = std::thread::spawn(move || {
        let mut r = Renderer::new(device(), desc).unwrap();
        let s = r.create_scene(&SceneDesc::default()).unwrap();
        let list = [s];
        let mut reports = BTreeMap::new();
        let mut out = Vec::new();
        let mut last = None;
        for i in 0..FRAMES {
            let mut f = r
                .begin_frame(FrameDesc {
                    time: FrameTime {
                        absolute: i as f64 / 60.0,
                        delta: 1.0 / 60.0,
                    },
                    scenes: &list,
                    display: Some(&mut display),
                })
                .unwrap();
            f.latency_marker(ConsumerMarker::SimulationStart);
            f.scene(s)
                .unwrap()
                .set_focus(DVec3::new(i as f64, 0.0, 0.0), 1000.0);
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
