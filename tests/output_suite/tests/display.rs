//! `Display` on a real window: frames drawn on the render queue and presented by the present
//! thread, with the encoding following the display feed and the request on the frame they
//! change, the extent following window resizes, and both present modes.
//!
//! AppKit windows live on the main thread, which libtest never runs tests on, so this target
//! has no harness: `main` runs the tests in turn, taking libtest's arguments.

#[cfg(any(windows, target_os = "macos"))]
mod windowed {
    use gpu_suite::window::Window;
    use orr_output::{
        ComposePass, ComposeRoot, ComposeSource, Display, DisplayFeed, DisplayInfo, DisplayRequest,
        OutputFormat, OutputKernels, ResolvedDisplay,
    };
    use orr_scene::orr_gpu::{
        ClearColor, ColorAttachment, ColorLoad, Format, QueueKind, Rendering, Texture, TextureDesc,
        TextureIndex, TextureMemory, TextureUsage, ViewDesc,
    };
    use orr_scene::testing::{assert_clean, device, wait};

    const FRAMES_IN_FLIGHT: u32 = 2;

    /// What the window's monitor reports with HDR on.
    fn hdr_on(sdr_white_nits: f32) -> DisplayInfo {
        DisplayInfo {
            os_hdr_on: Some(true),
            max_nits: Some(580.0),
            min_nits: Some(0.005),
            sdr_white_nits: Some(sdr_white_nits),
            ..DisplayInfo::default()
        }
    }

    struct Renderer {
        kernels: OutputKernels,
        roots: Vec<ComposeRoot>,
        /// A display-linear gray, smaller than the window: compose stretches its edge.
        linear: (Texture, TextureIndex),
        frame: u64,
        last_present_id: u64,
    }

    impl Renderer {
        fn new() -> Self {
            let gpu = device();
            let desc = TextureDesc::d2(
                Format::Rgba16Float,
                256,
                256,
                TextureUsage::SAMPLED | TextureUsage::COLOR,
            );
            let texture = gpu
                .create_texture(&desc, TextureMemory::Dedicated, "display linear")
                .unwrap();
            let index = gpu.write_sampled(
                gpu.alloc_image_slot().unwrap(),
                &texture,
                &ViewDesc::default(),
            );
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            list.discard(&texture);
            list.begin_rendering(&Rendering {
                extent: [256, 256],
                colors: &[ColorAttachment {
                    texture: &texture,
                    load: ColorLoad::Clear(ClearColor::Float([0.18, 0.18, 0.18, 1.0])),
                    store: true,
                }],
                depth: None,
            });
            list.end_rendering();
            wait(queue.submit(vec![list], &[]).unwrap());
            Self {
                kernels: OutputKernels::new(gpu).unwrap(),
                roots: (0..FRAMES_IN_FLIGHT)
                    .map(|_| ComposeRoot::new(gpu).unwrap())
                    .collect(),
                linear: (texture, index),
                frame: 0,
                last_present_id: 0,
            }
        }

        /// Draws one frame into the display as a renderer would: compose into the backbuffer
        /// (unless `ui_only`), the UI layer cleared (when `ui`), one submit waiting the slot's
        /// release, then present. Waits until the present thread is done with it and checks
        /// that it presented this frame at the resolved encoding and the window's extent.
        fn frame(&mut self, display: &mut Display, ui: bool, ui_only: bool) -> ResolvedDisplay {
            self.frame += 1;
            let id = self.frame;
            let queue = device().queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            let mut f = display
                .begin_frame(id)
                .unwrap()
                .expect("the window has an area");
            if !ui_only {
                let target = f.backbuffer(&mut list);
                let root = &mut self.roots[(id % u64::from(FRAMES_IN_FLIGHT)) as usize];
                self.kernels.compose(
                    &mut list,
                    root,
                    &ComposePass {
                        source: ComposeSource::Linear(self.linear.1),
                        ui: None,
                        target,
                        extent: f.extent,
                        display: &f.display,
                    },
                );
            }
            if ui {
                f.ui_layer(&mut list, 0);
            }
            let (resolved, extent, wait_for) = (f.display, f.extent, f.wait);
            let point = queue.submit(vec![list], wait_for.as_slice()).unwrap();
            display.present(&[point], false).unwrap();
            display.wait_idle();
            // The compose root is reused two frames later.
            wait(point);
            let p = display.last_presented().unwrap();
            assert_eq!(p.frame, id);
            let present_id = p.present_id.expect("presented");
            assert!(present_id > self.last_present_id, "present ids increase");
            self.last_present_id = present_id;
            assert_eq!(p.encoding, resolved.encoding, "frame {id}");
            assert_eq!(p.extent, extent, "frame {id}");
            resolved
        }
    }

    /// A window's raw handles on their way to another thread.
    #[cfg(windows)]
    struct Handles<T>(T);

    // SAFETY: a Win32 window handle is a plain id, which any thread may name.
    #[cfg(windows)]
    unsafe impl<T> Send for Handles<T> {}

    #[cfg(windows)]
    impl<T> Handles<T> {
        fn get(self) -> T {
            self.0
        }
    }

    /// Windows only: off the main thread, a Metal swapchain cannot be made at all.
    #[cfg(windows)]
    pub(super) fn a_display_that_cannot_watch_the_os_presents_sdr() {
        let gpu = device();
        let mut window = Window::new("orr display", 640, 400);
        let handles = Handles((window.window_handle(), window.display_handle()));
        // Off the window's thread, where the watch fails.
        let mut display = std::thread::spawn(move || {
            let (window, display) = handles.get();
            Display::new(
                gpu,
                window,
                display,
                &DisplayRequest::default(),
                FRAMES_IN_FLIGHT,
            )
            .unwrap()
        })
        .join()
        .unwrap();
        let mut r = Renderer::new();
        window.pump();
        let d = r.frame(&mut display, true, false);
        assert_eq!(d.encoding, OutputFormat::Sdr);

        // Asked to ignore the OS, the surface's formats decide, and Windows lists PQ.
        display.set_feed(None);
        window.pump();
        let d = r.frame(&mut display, true, false);
        if output_suite::offers_hdr(&display) {
            assert_eq!(d.encoding, OutputFormat::Hdr10Pq);
        }

        drop(display);
        drop(window);
        gpu.collect_garbage();
        assert_clean();
    }

    pub(super) fn display_follows_feed_request_and_window() {
        let gpu = device();
        let mut window = Window::new("orr display", 640, 400);
        let feed = DisplayFeed::new(hdr_on(200.0));
        let mut display = Display::new(
            gpu,
            window.window_handle(),
            window.display_handle(),
            &DisplayRequest::default(),
            FRAMES_IN_FLIGHT,
        )
        .unwrap();
        let mut r = Renderer::new();
        let mut frames = |display: &mut Display, window: &mut Window, n: usize| {
            let mut last = None;
            for i in 0..n {
                window.pump();
                last = Some(r.frame(display, i.is_multiple_of(2), i % 5 == 4));
            }
            last.unwrap()
        };

        // It watches the OS on its own, and follows what the OS says.
        let os = display.feed().expect("watches the OS").latest().1;
        assert!(os.os_hdr_on.is_some());
        assert!(
            os.sdr_white_nits.is_some() || os.headroom.is_some(),
            "a report from the OS: {os:?}"
        );
        let d = frames(&mut display, &mut window, 1);
        if os.os_hdr_on == Some(false) {
            assert_eq!(d.encoding, OutputFormat::Sdr);
        }

        let supported = display.supported_formats();
        if supported.contains(&OutputFormat::Hdr10Pq) {
            display.set_feed(Some(feed.clone()));
            let d = frames(&mut display, &mut window, 6);
            assert_eq!(d.encoding, OutputFormat::Hdr10Pq);
            assert_eq!(d.paper_white_nits, 200.0);

            // Windows HDR off: SDR from the next frame on, though the surface still lists PQ.
            feed.set(DisplayInfo {
                os_hdr_on: Some(false),
                ..hdr_on(200.0)
            });
            assert_eq!(
                frames(&mut display, &mut window, 1).encoding,
                OutputFormat::Sdr
            );
            assert_eq!(
                frames(&mut display, &mut window, 4).encoding,
                OutputFormat::Sdr
            );

            // HDR back on with the SDR content brightness slider moved.
            feed.set(hdr_on(320.0));
            let d = frames(&mut display, &mut window, 1);
            assert_eq!(d.encoding, OutputFormat::Hdr10Pq);
            assert_eq!(d.paper_white_nits, 320.0);

            // An explicit scRGB request.
            if supported.contains(&OutputFormat::ScRgb) {
                display.set_request(&DisplayRequest {
                    encoding: Some(OutputFormat::ScRgb),
                    ..DisplayRequest::default()
                });
                assert_eq!(
                    frames(&mut display, &mut window, 4).encoding,
                    OutputFormat::ScRgb
                );
            }
        } else {
            log::warn!("skipping the HDR checks: the surface offers only {supported:?}");
        }

        // Vsync off on the same swapchain.
        let encoding = frames(&mut display, &mut window, 1).encoding;
        display.set_request(&DisplayRequest {
            vsync: false,
            ..*display.request()
        });
        assert_eq!(frames(&mut display, &mut window, 20).encoding, encoding);

        // Resizes reach the backbuffer on the next frame.
        for size in [(900, 500), (420, 700), (640, 400)] {
            window.resize(size.0, size.1);
            frames(&mut display, &mut window, 3);
        }

        drop(display);
        drop(window);
        gpu.collect_garbage();
        assert_clean();
    }
}

#[cfg(any(windows, target_os = "macos"))]
fn main() {
    let tests: &[(&str, fn())] = &[
        #[cfg(windows)]
        (
            "a_display_that_cannot_watch_the_os_presents_sdr",
            windowed::a_display_that_cannot_watch_the_os_presents_sdr,
        ),
        (
            "display_follows_feed_request_and_window",
            windowed::display_follows_feed_request_and_window,
        ),
    ];
    gpu_suite::run_on_main_thread(tests);
}

#[cfg(not(any(windows, target_os = "macos")))]
fn main() {}
