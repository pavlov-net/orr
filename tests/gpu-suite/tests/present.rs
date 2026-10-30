//! Swapchains on a window: metered batches, recreation under resizes, and Reflex calls from
//! a render thread beside a present thread.
#![cfg(windows)]

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError, mpsc};
use std::time::Duration;

use gpu_suite::shaders::swapchain_fill;
use gpu_suite::window::Window;
use gpu_suite::{assert_clean, check_latency_reports, device, fill_frame, pipeline, render};
use orr_gpu::{
    ComputePipeline, LatencyMarker, PresentMode, Queue, QueueKind, Swapchain, SwapchainDesc,
    TimelinePoint,
};

/// The tests share the window class and the GPU; one runs at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

fn present_queue() -> &'static Queue {
    device().queue(QueueKind::Present).unwrap()
}

fn swapchain(window: &Window, mode: PresentMode) -> Swapchain {
    let desc = SwapchainDesc {
        present_mode: mode,
        ..SwapchainDesc::default()
    };
    Swapchain::new(
        device(),
        window.window_handle(),
        window.display_handle(),
        &desc,
    )
    .unwrap()
}

/// Acquires an image, writes the test card, submits and presents it. `None` while the
/// window has no area.
fn present_one(
    sc: &mut Swapchain,
    fill: &ComputePipeline,
    n: u32,
    batch: Option<u32>,
    wait: &[TimelinePoint],
) -> Option<u64> {
    let frame = sc.acquire().unwrap()?;
    let q = present_queue();
    let (list, _root) = fill_frame(device(), q, fill, &frame, n);
    q.submit_present(vec![list], wait, &frame).unwrap();
    Some(sc.present(frame, batch).unwrap())
}

fn finish(sc: Swapchain) {
    drop(sc);
    device().collect_garbage();
    assert_clean();
}

#[test]
fn metered_batches_keep_present_ids_increasing() {
    let _serial = serial();
    let mut window = Window::new("metered batches", 960, 540);
    let mut sc = swapchain(&window, PresentMode::Fifo);
    let fill = pipeline(&swapchain_fill::MODULE, "fill_main");
    let mut ids = Vec::new();
    let mut n = 0;
    for _ in 0..20 {
        for batch in [2, 3, 4] {
            for i in 0..batch {
                window.pump();
                let first = (i == 0).then_some(batch);
                ids.push(present_one(&mut sc, &fill, n, first, &[]).expect("window has area"));
                n += 1;
            }
        }
    }
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "present ids not strictly increasing: {ids:?}"
    );
    if device().caps().present_wait {
        let last = *ids.last().unwrap();
        assert!(sc.wait_presented(last, Duration::from_secs(1)).unwrap());
    }
    log::info!("refresh {:?}", sc.refresh());
    finish(sc);
}

#[test]
fn thousand_resizes() {
    let _serial = serial();
    let mut window = Window::new("resizes", 800, 600);
    let mut sc = swapchain(&window, PresentMode::Immediate);
    let fill = pipeline(&swapchain_fill::MODULE, "fill_main");
    for i in 0..1000u32 {
        let w = 200 + (i * 37) % 900;
        let h = 150 + (i * 53) % 600;
        window.resize(w as i32, h as i32);
        let frame = sc.acquire().unwrap().expect("window has area");
        assert_eq!(
            frame.extent(),
            window.client_size(),
            "the swapchain follows the window"
        );
        let q = present_queue();
        let (list, _root) = fill_frame(device(), q, &fill, &frame, i);
        q.submit_present(vec![list], &[], &frame).unwrap();
        sc.present(frame, None).unwrap();
        device().collect_garbage();
    }
    finish(sc);
}

#[test]
fn render_thread_sleeps_and_marks_beside_the_present_thread() {
    const FRAMES: u64 = 10_000;
    let _serial = serial();
    if !device().caps().low_latency {
        log::warn!("no VK_NV_low_latency2; skipped");
        return;
    }
    let mut window = Window::new("low latency", 960, 540);
    let mut sc = swapchain(&window, PresentMode::Immediate);
    let latency = sc.latency().expect("low latency");
    latency.set_mode(true, false, 0);
    let fill = pipeline(&swapchain_fill::MODULE, "fill_main");
    let (to_present, frames) = mpsc::sync_channel::<(u64, TimelinePoint)>(1);

    let render_latency = latency.clone();
    let render_thread = std::thread::spawn(move || {
        let queue = render();
        for id in 1..=FRAMES {
            render_latency.sleep();
            render_latency.marker(id, LatencyMarker::SimulationStart);
            render_latency.marker(id, LatencyMarker::SimulationEnd);
            render_latency.marker(id, LatencyMarker::RenderSubmitStart);
            let list = queue.command_list().unwrap();
            let point = queue.submit(vec![list], &[]).unwrap();
            render_latency.marker(id, LatencyMarker::RenderSubmitEnd);
            to_present.send((id, point)).unwrap();
        }
    });
    let present_thread = std::thread::spawn(move || {
        let mut reports = BTreeMap::new();
        let mut out = Vec::new();
        while let Ok((id, point)) = frames.recv() {
            latency.marker(id, LatencyMarker::PresentStart);
            present_one(&mut sc, &fill, id as u32, None, &[point]).expect("window has area");
            latency.marker(id, LatencyMarker::PresentEnd);
            if id % 16 == 0 {
                out.clear();
                latency.timings(&mut out);
                for t in &out {
                    reports.insert(t.frame_id, *t);
                }
            }
        }
        (sc, reports)
    });
    while !present_thread.is_finished() {
        window.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
    render_thread.join().unwrap();
    let (sc, reports) = present_thread.join().unwrap();
    check_latency_reports(&reports, FRAMES);
    finish(sc);
}
