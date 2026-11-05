//! Measures the luminance macOS gives EDR 1.0 when it maps PQ content. Two windows open on
//! the screen with the most potential HDR headroom: an SDR window of sRGB white, which macOS
//! shows at the screen's SDR white (EDR 1.0), and a PQ window of gray bars at the luminances
//! the log lists. Where the white window meets the bars, the bar that matches it is the
//! reference; drag a window if they don't touch. A camera judges it better than the eye at an
//! exposure low enough that the bars step visibly.
//!
//! `edr_reference [seconds] [hdr10]`: runs for the given seconds (default 120) or until a
//! window closes; `hdr10` gives the PQ layer HDR10 metadata, as `Display` does.
//!
//! macOS only: elsewhere the example says so and exits.

#[cfg(target_os = "macos")]
mod measure {
    use std::time::{Duration, Instant};

    use gpu_suite::window::Window;
    use gpu_suite::{assert_clean, device};
    use objc2::MainThreadMarker;
    use objc2_foundation::NSPoint;
    use orr_color::{RgbPrimaries, pq_encode};
    use orr_gpu::{
        Allocation, Device, HdrMetadata, MemoryKind, OutputFormat, PresentMode, Queue, QueueKind,
        Swapchain, SwapchainDesc, TextureRegion,
    };

    /// The bars' luminances in nits, left to right.
    const LEVELS: [f64; 10] = [
        60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0, 160.0, 203.0,
    ];
    const WHITE_WIDTH: i32 = 360;
    const BARS_WIDTH: i32 = 1000;
    const HEIGHT: i32 = 500;

    /// Rec.2020 primaries and a 1000-nit peak, above every bar.
    fn metadata() -> HdrMetadata {
        let xy = |c: orr_color::Chromaticity| [c.x as f32, c.y as f32];
        let p = RgbPrimaries::REC2020;
        HdrMetadata {
            primaries: [xy(p.red), xy(p.green), xy(p.blue)],
            white: xy(p.white),
            max_luminance: 1000.0,
            min_luminance: 0.005,
            max_cll: 203.0,
            max_fall: 0.0,
        }
    }

    /// An RGB10A2 gray at `nits`, PQ-encoded.
    fn pq_gray(nits: f64) -> u32 {
        let c = (pq_encode(nits) * 1023.0).round() as u32;
        c | c << 10 | c << 20 | 3 << 30
    }

    /// The bars at `extent`, with a thin black gap before each.
    fn bars(extent: [u32; 2]) -> Vec<u32> {
        let [w, h] = extent.map(|v| v as usize);
        let n = LEVELS.len();
        let row: Vec<u32> = (0..w)
            .map(|x| {
                let gap = (x * n) % w < w / 100;
                if gap {
                    pq_gray(0.0)
                } else {
                    pq_gray(LEVELS[x * n / w])
                }
            })
            .collect();
        row.repeat(h)
    }

    /// One window's swapchain and the texels it shows, uploaded once per size.
    struct Shown {
        swapchain: Swapchain,
        texels: fn([u32; 2]) -> Vec<u32>,
        upload: Option<([u32; 2], Allocation<u32>)>,
    }

    impl Shown {
        fn present(&mut self, d: &Device, queue: &Queue) {
            let Some(frame) = self.swapchain.acquire().unwrap() else {
                return;
            };
            let extent = frame.extent();
            if self.upload.as_ref().is_none_or(|(e, _)| *e != extent) {
                let texels = (self.texels)(extent);
                let mut up = d
                    .gpu_malloc::<u32>(MemoryKind::Upload, texels.len() as u64, "edr texels")
                    .unwrap();
                up.as_mut_slice().unwrap().copy_from_slice(&texels);
                self.upload = Some((extent, up));
            }
            let (_, up) = self.upload.as_ref().unwrap();
            let mut list = queue.command_list().unwrap();
            let region = TextureRegion::whole_mip(frame.texture().desc(), 0);
            list.copy_to_texture(frame.texture(), &region, up.gpu().cast());
            queue.submit_present(vec![list], &[], &frame).unwrap();
            self.swapchain.present(frame, None).unwrap();
        }
    }

    pub(super) fn main() {
        let mut args = std::env::args().skip(1);
        let seconds: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(120);
        let hdr = args.next().as_deref() == Some("hdr10");
        let mtm = MainThreadMarker::new().expect("examples start on the main thread");
        // Made first: it starts the logging.
        let d = device();

        let screens = objc2_app_kit::NSScreen::screens(mtm);
        let screen = screens
            .iter()
            .inspect(|s| {
                log::info!(
                    "screen {}: potential headroom {}, now {}",
                    s.localizedName(),
                    s.maximumPotentialExtendedDynamicRangeColorComponentValue(),
                    s.maximumExtendedDynamicRangeColorComponentValue(),
                );
            })
            .max_by(|a, b| {
                a.maximumPotentialExtendedDynamicRangeColorComponentValue()
                    .total_cmp(&b.maximumPotentialExtendedDynamicRangeColorComponentValue())
            })
            .expect("a screen");
        log::info!("measuring on {}", screen.localizedName());
        let frame = screen.frame();
        let left = frame.origin.x + 60.0;
        let top = frame.origin.y + frame.size.height - 80.0;

        let queue = d.queue(QueueKind::Present).unwrap();
        let mut white_window = Window::new("SDR white", WHITE_WIDTH, HEIGHT);
        white_window.move_to(NSPoint::new(left, top));
        let mut bars_window = Window::new("PQ bars", BARS_WIDTH, HEIGHT);
        bars_window.move_to(NSPoint::new(left + f64::from(WHITE_WIDTH), top));
        let swapchain = |window: &Window, format, hdr| {
            Swapchain::new(
                d,
                window.window_handle(),
                window.display_handle(),
                &SwapchainDesc {
                    format,
                    hdr,
                    present_mode: PresentMode::Fifo,
                    ..SwapchainDesc::default()
                },
            )
            .unwrap()
        };
        let mut white = Shown {
            swapchain: swapchain(&white_window, OutputFormat::Sdr, None),
            texels: |[w, h]| vec![u32::MAX; (w * h) as usize],
            upload: None,
        };
        let mut pq = Shown {
            swapchain: swapchain(&bars_window, OutputFormat::Hdr10Pq, hdr.then(metadata)),
            texels: bars,
            upload: None,
        };
        for (i, nits) in LEVELS.iter().enumerate() {
            log::info!("bar {}: {nits} nits", i + 1);
        }
        log::info!(
            "HDR10 metadata {}; keep the brightness fixed and auto-brightness off",
            if hdr { "on" } else { "off" }
        );

        let start = Instant::now();
        while white_window.pump()
            && bars_window.pump()
            && start.elapsed() < Duration::from_secs(seconds)
        {
            white.present(d, queue);
            pq.present(d, queue);
            d.collect_garbage();
        }
        drop((white, pq));
        d.collect_garbage();
        assert_clean();
    }
}

#[cfg(target_os = "macos")]
fn main() {
    measure::main();
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::print_stderr)]
fn main() {
    eprintln!("the EDR reference example measures macOS's EDR mapping; it runs on macOS only");
}
