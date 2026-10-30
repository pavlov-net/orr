//! Presents a test card to a window, cycling the encoding SDR -> PQ -> scRGB every two
//! seconds (with HDR metadata) and the present mode FIFO <-> IMMEDIATE every second. Logs
//! each switch and whether it recreated the swapchain. Runs for the given seconds (default
//! 12) or until the window closes.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use gpu_suite::shaders::swapchain_fill;
use gpu_suite::window::Window;
use gpu_suite::{assert_clean, device, fill_frame, pipeline};
use orr_gpu::{HdrMetadata, OutputFormat, PresentMode, QueueKind, Swapchain, SwapchainDesc};

/// Rec.2020 primaries, D65 white, a 1000-nit peak.
const METADATA: HdrMetadata = HdrMetadata {
    primaries: [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
    white: [0.3127, 0.3290],
    max_luminance: 1000.0,
    min_luminance: 0.005,
    max_cll: 1000.0,
    max_fall: 0.0,
};

fn main() {
    let seconds: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(12);
    let d = device();
    let queue = d.queue(QueueKind::Present).unwrap();
    let mut window = Window::new("swapchain", 1280, 720);
    let mut sc = Swapchain::new(
        d,
        window.window_handle(),
        window.display_handle(),
        &SwapchainDesc::default(),
    )
    .unwrap();
    let formats: Vec<OutputFormat> = sc.supported_formats().to_vec();
    log::info!("surface offers {formats:?}; refresh {:?}", sc.refresh());
    let fill = pipeline(&swapchain_fill::MODULE, "fill_main");

    let start = Instant::now();
    let mut segment = (0, PresentMode::Fifo);
    let mut slots = BTreeSet::new();
    let mut last_slots = BTreeSet::new();
    let mut presents = 0u32;
    let mut segment_start = Instant::now();
    let mut n = 0u32;
    while window.pump() && start.elapsed() < Duration::from_secs(seconds) {
        let second = start.elapsed().as_secs();
        let mode = if second.is_multiple_of(2) {
            PresentMode::Fifo
        } else {
            PresentMode::Immediate
        };
        let wanted = (((second / 2) as usize) % formats.len(), mode);
        if wanted != segment {
            let rate = f64::from(presents) / segment_start.elapsed().as_secs_f64();
            let f = formats[wanted.0];
            if wanted.0 != segment.0 {
                sc.set_format(f, (f != OutputFormat::Sdr).then_some(METADATA))
                    .unwrap();
            }
            sc.set_present_mode(mode);
            log::info!(
                "{:?} {:?}: {rate:.0} presents/s; now {f:?} {mode:?}",
                formats[segment.0],
                segment.1,
            );
            segment = wanted;
            last_slots = std::mem::take(&mut slots);
            presents = 0;
            segment_start = Instant::now();
        }
        let Some(frame) = sc.acquire().unwrap() else {
            std::thread::sleep(Duration::from_millis(10));
            continue;
        };
        if presents == 0 && !last_slots.is_empty() {
            let recreated = !last_slots.contains(&frame.storage().0);
            log::info!("  {:?}, recreated: {recreated}", frame.format());
        }
        slots.insert(frame.storage().0);
        let (list, _root) = fill_frame(device(), queue, &fill, &frame, n);
        queue.submit_present(vec![list], &[], &frame).unwrap();
        sc.present(frame, None).unwrap();
        presents += 1;
        n += 1;
        d.collect_garbage();
    }
    drop(sc);
    d.collect_garbage();
    assert_clean();
    log::info!("{n} presents, validation clean");
}
