//! `orr::Renderer` frames with a low latency handle on a window, beside a present thread
//! that sets the present markers as the display does.
#![cfg(windows)]

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::Duration;

use glam::DVec3;
use gpu_suite::shaders::swapchain_fill;
use gpu_suite::window::Window;
use gpu_suite::{check_latency_reports, fill_frame};
use orr::orr_gpu::{
    ComputePipeline, LatencyMarker, PresentMode, QueueKind, Swapchain, SwapchainDesc, TimelinePoint,
};
use orr::{ConsumerMarker, FrameDesc, FrameTime, Renderer, RendererDesc, SceneDesc};
use orr_scene::testing::{assert_clean, device, wait};
use scene_suite::pipeline;

const FRAMES: u64 = 1000;

/// Writes the test card into an acquired image and presents it after `point`.
fn present(sc: &mut Swapchain, fill: &ComputePipeline, n: u32, point: TimelinePoint) {
    let frame = sc.acquire().unwrap().expect("window has area");
    let q = device().queue(QueueKind::Present).unwrap();
    let (list, _root) = fill_frame(device(), q, fill, &frame, n);
    q.submit_present(vec![list], &[point], &frame).unwrap();
    sc.present(frame, None).unwrap();
}

#[test]
fn every_frame_id_gets_a_complete_ordered_report() {
    if !device().caps().low_latency {
        log::warn!("no VK_NV_low_latency2; skipped");
        return;
    }
    let mut window = Window::new("frame latency", 640, 360);
    let desc = SwapchainDesc {
        present_mode: PresentMode::Immediate,
        ..SwapchainDesc::default()
    };
    let mut sc = Swapchain::new(
        device(),
        window.window_handle(),
        window.display_handle(),
        &desc,
    )
    .unwrap();
    let latency = sc.latency().expect("low latency");
    latency.set_mode(true, false, 0);
    let fill = pipeline(&swapchain_fill::MODULE, "fill_main");
    let (to_present, frames) = mpsc::sync_channel::<(u64, TimelinePoint)>(1);

    let render_latency = latency.clone();
    let render_thread = std::thread::spawn(move || {
        let mut r = Renderer::new(device(), RendererDesc::default()).unwrap();
        let s = r.create_scene(&SceneDesc::default()).unwrap();
        let list = [s];
        let mut last = None;
        for i in 0..FRAMES {
            let mut f = r
                .begin_frame(&FrameDesc {
                    time: FrameTime {
                        absolute: i as f64 / 60.0,
                        delta: 1.0 / 60.0,
                    },
                    scenes: &list,
                    low_latency: Some(&render_latency),
                })
                .unwrap();
            f.latency_marker(ConsumerMarker::SimulationStart);
            f.scene(s)
                .unwrap()
                .set_focus(DVec3::new(i as f64, 0.0, 0.0), 1000.0);
            f.latency_marker(ConsumerMarker::SimulationEnd);
            f.update_scenes(&mut |_| {}).unwrap();
            let id = f.id();
            let point = f.end().unwrap();
            to_present.send((id, point)).unwrap();
            last = Some((id, point));
        }
        let (id, point) = last.unwrap();
        wait(point);
        id
    });
    let present_thread = std::thread::spawn(move || {
        let mut reports = BTreeMap::new();
        let mut out = Vec::new();
        while let Ok((id, point)) = frames.recv() {
            latency.marker(id, LatencyMarker::PresentStart);
            present(&mut sc, &fill, id as u32, point);
            latency.marker(id, LatencyMarker::PresentEnd);
            if id % 16 == 0 {
                out.clear();
                latency.timings(&mut out);
                for t in &out {
                    reports.insert(t.frame_id, *t);
                }
            }
        }
        reports
    });
    while !present_thread.is_finished() {
        window.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
    let last = render_thread.join().unwrap();
    let reports = present_thread.join().unwrap();
    check_latency_reports(&reports, last);
    device().collect_garbage();
    assert_clean();
}
