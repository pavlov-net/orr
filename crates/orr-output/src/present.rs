//! A window's display: the encoded backbuffer, UI layer and generated images per frame in
//! flight on the render thread, frame generation recorded into the frame, and a present
//! thread that owns the swapchain, composes the UI over each presented image and presents
//! it.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use orr_dlss::{DlssSdk, FgInputs, FrameGeneration};
use orr_math::DVec3;
use orr_scene::ViewDesc as SceneView;
use orr_scene::orr_gpu::{
    self, Allocation, CommandList, Device, Format, Hazards, HdrMetadata, LatencyMarker, LowLatency,
    MemoryKind, OutputFormat, PresentMode, QueueKind, Refresh, Stages, StorageImageIndex,
    Swapchain, SwapchainDesc, Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage,
    TimelinePoint, ViewDesc,
};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::frame_generation::{
    DynamicCount, FgCameraState, fg_camera, hardware_scheduling, min_interval_us,
};
use crate::passes::ComposeKernel;
use crate::{
    ComposePass, ComposeRoot, ComposeSource, DisplayFeed, DisplayRequest, DistortionField, Error,
    FrameGenerationMode, ResolvedDisplay, Result, UiLayer,
};

/// How long the present thread waits for a compose root's previous submit, which is a frame
/// slot's worth of frames old, or for a frame's generated images.
const ROOT_TIMEOUT: Duration = Duration::from_secs(10);
/// Swapchain images without frame generation; with it, at least the largest batch plus one.
const MIN_IMAGES: u32 = 5;
/// Frames whose cost the present thread reports and `Dynamic` has not read yet.
const COST_REPORTS: usize = 32;
/// The pass frame generation records under, which `Dynamic` reads its cost from.
const FG_PASS: &str = "frame_generation";

/// A window's display target. The render thread draws each frame into the frame slot
/// [`Display::begin_frame`] returns and hands it over with [`Display::present`]; the present
/// thread then acquires a swapchain image, blends the frame's UI layer over the backbuffer
/// into it on the Present queue and presents it, so the render thread never presents. With
/// frame generation on, the frame's generated images are presented before it, each with
/// the UI composed over it, as one metered batch.
///
/// The encoding and luminance are resolved at every `begin_frame` from the request, the
/// feed's latest `DisplayInfo` and the formats the surface offers; an encoding change takes
/// effect on that frame.
pub struct Display {
    window: isize,
    request: DisplayRequest,
    feed: Option<DisplayFeed>,
    slots: Vec<Slot>,
    begun: Option<Begun>,
    latency: Option<LowLatency>,
    /// The consumer's low-latency request: on, boost.
    low_latency: (bool, bool),
    /// The mode last given to `LowLatency::set_mode`.
    applied_latency: Option<(bool, bool, u32)>,
    field: Option<DistortionField>,
    fg: Option<Fg>,
    costs: VecDeque<Cost>,
    started: Instant,
    shared: Arc<Shared>,
    jobs: Option<Sender<Job>>,
    thread: Option<JoinHandle<()>>,
    device: Device,
    #[cfg(all(windows, feature = "pcl-stats"))]
    _pcl_stats: Option<crate::pcl_stats::Session>,
}

/// Frame generation on this display.
struct Fg {
    sdk: DlssSdk,
    mode: FrameGenerationMode,
    max_generated: u32,
    dynamic: Option<DynamicCount>,
    feature: Option<(FrameGeneration, OutputFormat, [u32; 2])>,
    /// The camera of the last frame that generated images.
    prev: Option<FgCameraState>,
    /// The point camera positions are given relative to, moved at each reset.
    anchor: DVec3,
    /// The next evaluate does not follow the last one.
    reset: bool,
}

impl Fg {
    /// The largest count the mode may ask for.
    fn most(&self) -> u32 {
        match self.mode {
            FrameGenerationMode::Off => 0,
            FrameGenerationMode::Fixed(n) => n,
            FrameGenerationMode::Dynamic { max, .. } => max,
        }
        .min(self.max_generated)
    }
}

/// A rendered frame whose GPU cost `Dynamic` has yet to read.
struct Cost {
    frame: u64,
    /// The frame's main submit.
    render: TimelinePoint,
    generated: u32,
    /// The frame's compose submits, once the present thread reports them.
    compose: Option<Vec<TimelinePoint>>,
}

/// One frame slot: the images a frame draws into, reused `frames_in_flight` frames later.
struct Slot {
    backbuffer: Option<Encoded>,
    ui: Option<UiLayer>,
    backbuffer_written: bool,
    ui_drawn: bool,
    /// Frame generation's outputs, in the backbuffer's encoding and size.
    generated: Vec<Encoded>,
    /// NGX's interpolation-disabled flag for this slot's frame.
    flag: Option<Arc<Allocation<u32>>>,
    /// What the frame's view gave frame generation.
    view: Option<FgView>,
    /// The frame's hooks wrote the display's distortion field.
    field_used: bool,
    /// Images frame generation recorded for the frame.
    recorded: u32,
}

/// The view frame generation reprojects, from [`DisplayFrame::frame_generation`].
struct FgView {
    depth: Arc<Texture>,
    motion: Arc<Texture>,
    camera: FgCameraState,
    jitter: [f32; 2],
    reset: bool,
    not_rendering_scene: bool,
}

/// An image in one of the swapchain's encodings: the UI-free backbuffer compose writes, or
/// a generated image frame generation writes.
struct Encoded {
    texture: Texture,
    storage: StorageImageIndex,
    sampled: TextureIndex,
    encoding: OutputFormat,
    device: Device,
}

impl Encoded {
    fn fits(&self, encoding: OutputFormat, extent: [u32; 2]) -> bool {
        self.encoding == encoding && self.texture.desc().extent[..2] == extent
    }
}

impl Drop for Encoded {
    fn drop(&mut self) {
        self.device.free_image_slot_after_submitted(self.storage.0);
        self.device.free_image_slot_after_submitted(self.sampled.0);
    }
}

struct Begun {
    frame: u64,
    slot: usize,
    display: ResolvedDisplay,
    extent: [u32; 2],
    wait: Option<TimelinePoint>,
    generated: u32,
}

/// A frame handed to the present thread.
struct Job {
    frame: u64,
    slot: usize,
    after: TimelinePoint,
    display: ResolvedDisplay,
    mode: PresentMode,
    source: Option<TextureIndex>,
    ui: Option<TextureIndex>,
    /// Generated images, presented in order before the frame.
    generated: Vec<TextureIndex>,
    flag: Option<Arc<Allocation<u32>>>,
    min_images: u32,
}

/// State the present thread reports back.
struct State {
    /// A slot's frame is with the present thread.
    pending: Vec<bool>,
    /// The Present-queue point after which a slot's images are free again.
    released: Vec<Option<TimelinePoint>>,
    supported: Vec<OutputFormat>,
    refresh: Refresh,
    presented: Option<Presented>,
    /// Each presented frame's compose submits, for frame generation's cost.
    compose: VecDeque<(u64, Vec<TimelinePoint>)>,
    error: Option<orr_gpu::Error>,
}

struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The frame slot a frame draws into on the render thread.
pub struct DisplayFrame<'a> {
    slot: &'a mut Slot,
    field: &'a mut Option<DistortionField>,
    device: &'a Device,
    /// What the frame encodes for; tonemap and compose take it.
    pub display: ResolvedDisplay,
    /// The window's client extent, the size of the backbuffer and UI layer.
    pub extent: [u32; 2],
    /// The frame's main submit waits this point, after which the present thread no longer
    /// reads the slot's images.
    pub wait: Option<TimelinePoint>,
    /// Images frame generation adds after this frame; 0 with it off.
    pub generated: u32,
}

impl DisplayFrame<'_> {
    /// The backbuffer, for compose to write every texel of. The first call records its
    /// discard into `list`; a frame that never calls it presents black under its UI.
    pub fn backbuffer(&mut self, list: &mut CommandList<'_>) -> StorageImageIndex {
        let b = self
            .slot
            .backbuffer
            .as_ref()
            .expect("begin_frame sizes the backbuffer");
        if !self.slot.backbuffer_written {
            list.discard(&b.texture);
            self.slot.backbuffer_written = true;
        }
        b.storage
    }

    /// The UI layer, for UI renderers to draw into. The first call records its clear into
    /// `list`; a frame that never calls it presents no UI.
    pub fn ui_layer(&mut self, list: &mut CommandList<'_>) -> &UiLayer {
        let ui = self
            .slot
            .ui
            .as_ref()
            .expect("begin_frame sizes the UI layer");
        if !self.slot.ui_drawn {
            ui.clear(list);
            self.slot.ui_drawn = true;
        }
        ui
    }

    /// The view the backbuffer shows, for frame generation: its camera, its reverse-Z
    /// `depth` and its `motion` (previous UV minus current UV) at the render extent, which
    /// stay valid until the frame's GPU work completes. A frame without it generates no
    /// images, and the next one that has it starts frame generation afresh.
    pub fn frame_generation(
        &mut self,
        view: &SceneView<'_>,
        depth: &Arc<Texture>,
        motion: &Arc<Texture>,
        not_rendering_scene: bool,
    ) {
        self.slot.view = Some(FgView {
            depth: Arc::clone(depth),
            motion: Arc::clone(motion),
            camera: FgCameraState {
                scene: view.scene,
                position: view.position,
                rotation: view.rotation,
                projection: view.projection,
                extent: view.render_extent.to_array(),
            },
            jitter: view.jitter.to_array(),
            reset: view.reset_history,
            not_rendering_scene,
        });
    }

    /// The display's distortion field at this frame's extent, for the view's warping hooks;
    /// frame generation reads it after them this frame. Pass it to `hook_inputs` and every
    /// hook run, which clear it and write it.
    pub fn distortion_field(&mut self) -> Result<&mut DistortionField> {
        let half = DistortionField::extent_for(self.extent);
        if self.field.as_ref().is_none_or(|f| f.extent() != half) {
            *self.field = Some(DistortionField::new(self.device, self.extent)?);
        }
        self.slot.field_used = true;
        Ok(self.field.as_mut().expect("made above"))
    }
}

/// What the present thread did with a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Presented {
    /// The id passed to [`Display::begin_frame`].
    pub frame: u64,
    /// The swapchain's present id of the frame's own image; `None` when it was not
    /// presented (the window was minimized, or the surface stopped offering its encoding).
    pub present_id: Option<u64>,
    /// Generated images presented before it.
    pub generated: u32,
    /// The swapchain image's encoding and extent.
    pub encoding: OutputFormat,
    pub extent: [u32; 2],
}

impl Display {
    /// Creates the swapchain and starts the present thread. `frames_in_flight` matches the
    /// renderer's. Fails if the device was created without `QueueRequest::present` or the
    /// window is not a Win32 window.
    pub fn new(
        gpu: &Device,
        window: RawWindowHandle,
        display: RawDisplayHandle,
        request: &DisplayRequest,
        feed: Option<DisplayFeed>,
        frames_in_flight: u32,
    ) -> Result<Self> {
        let RawWindowHandle::Win32(handle) = window else {
            return Err(Error::NotWin32Window);
        };
        let swapchain = Swapchain::new(
            gpu,
            window,
            display,
            &SwapchainDesc {
                min_images: MIN_IMAGES,
                present_mode: request.present_mode(),
                ..SwapchainDesc::default()
            },
        )?;
        let latency = swapchain.latency();
        let compose = ComposeKernel::new(gpu)?;
        let n = frames_in_flight.max(1) as usize;
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                pending: vec![false; n],
                released: vec![None; n],
                supported: swapchain.supported_formats().to_vec(),
                refresh: swapchain.refresh(),
                presented: None,
                compose: VecDeque::new(),
                error: None,
            }),
            changed: Condvar::new(),
        });
        let (jobs, rx) = mpsc::channel();
        let thread = {
            let shared = Arc::clone(&shared);
            let presenter = Presenter {
                device: gpu.clone(),
                swapchain,
                latency: latency.clone(),
                compose,
                roots: (0..n).map(|_| Vec::new()).collect(),
                format: None,
            };
            std::thread::Builder::new()
                .name("orr present".into())
                .spawn(move || presenter.run(&rx, &shared))
                .map_err(|e| Error::Thread(e.to_string()))?
        };
        #[cfg(all(windows, feature = "pcl-stats"))]
        let pcl_stats = latency.as_ref().map(crate::pcl_stats::Session::start);
        Ok(Self {
            window: handle.hwnd.get(),
            request: *request,
            feed,
            slots: (0..n)
                .map(|_| Slot {
                    backbuffer: None,
                    ui: None,
                    backbuffer_written: false,
                    ui_drawn: false,
                    generated: Vec::new(),
                    flag: None,
                    view: None,
                    field_used: false,
                    recorded: 0,
                })
                .collect(),
            begun: None,
            latency,
            low_latency: (false, false),
            applied_latency: None,
            field: None,
            fg: None,
            costs: VecDeque::new(),
            started: Instant::now(),
            shared,
            jobs: Some(jobs),
            thread: Some(thread),
            device: gpu.clone(),
            #[cfg(all(windows, feature = "pcl-stats"))]
            _pcl_stats: pcl_stats,
        })
    }

    pub fn request(&self) -> &DisplayRequest {
        &self.request
    }

    /// Replaces the request; the next `begin_frame` resolves with it. Changing only `vsync`
    /// keeps the swapchain.
    pub fn set_request(&mut self, request: &DisplayRequest) {
        self.request = *request;
    }

    /// Turns Reflex low latency on or off (`vkSetLatencySleepModeNV`). Off until called;
    /// kept across swapchain recreation; held on while frame generation is on. Applied at
    /// the next `begin_frame`. Does nothing without `Caps::low_latency`.
    pub fn set_low_latency(&mut self, on: bool, boost: bool) {
        self.low_latency = (on, boost);
    }

    /// Turns frame generation on, changes its mode or turns it off, from the next
    /// `begin_frame`. On, it needs NGX frame generation on `sdk`'s device (the display's),
    /// Windows' hardware-accelerated GPU scheduling, Reflex low latency
    /// (`Caps::low_latency`), present ids and present metering; without them this fails and
    /// leaves frame generation as it was. While it is on, scRGB requests present as PQ and
    /// Reflex stays on with the base rate capped to the refresh rate over n + 1.
    pub fn set_frame_generation(&mut self, sdk: &DlssSdk, mode: FrameGenerationMode) -> Result<()> {
        if mode == FrameGenerationMode::Off {
            self.fg = None;
            return Ok(());
        }
        let support = sdk.support();
        let caps = self.device.caps();
        let missing = if !support.fg || support.max_generated == 0 {
            Some("NGX frame generation")
        } else if self.latency.is_none() {
            Some("VK_NV_low_latency2")
        } else if !caps.present_wait || !caps.present_metering {
            Some("present ids and present metering")
        } else if hardware_scheduling(&self.device) != Some(true) {
            Some("hardware-accelerated GPU scheduling")
        } else {
            None
        };
        if let Some(missing) = missing {
            return Err(Error::FrameGenerationUnavailable(missing));
        }
        let dynamic = match mode {
            FrameGenerationMode::Fixed(n) if (1..=support.max_generated).contains(&n) => None,
            FrameGenerationMode::Dynamic { max, base_hz }
                if max >= 1 && base_hz.is_finite() && base_hz > 0.0 =>
            {
                Some(DynamicCount::new(max, base_hz))
            }
            _ => return Err(Error::FrameGenerationMode(mode)),
        };
        match &mut self.fg {
            Some(fg) => {
                fg.mode = mode;
                fg.dynamic = dynamic;
            }
            None => {
                self.fg = Some(Fg {
                    sdk: sdk.clone(),
                    mode,
                    max_generated: support.max_generated,
                    dynamic,
                    feature: None,
                    prev: None,
                    anchor: DVec3::ZERO,
                    reset: true,
                });
            }
        }
        Ok(())
    }

    /// The frame-generation mode in use.
    pub fn frame_generation_mode(&self) -> FrameGenerationMode {
        self.fg
            .as_ref()
            .map_or(FrameGenerationMode::Off, |f| f.mode)
    }

    /// The swapchain's low-latency handle, for latency reports; `None` without
    /// `Caps::low_latency`.
    pub fn latency(&self) -> Option<&LowLatency> {
        self.latency.as_ref()
    }

    /// Opens frame `frame` (orr's frame id) for drawing: waits until the present thread has
    /// submitted the slot's previous frame, resolves the display, picks the frame's
    /// generated count and sizes the slot's images to the window. `None` while the window
    /// has no area; the frame then draws nothing into the display and `present` does
    /// nothing.
    pub fn begin_frame(&mut self, frame: u64) -> Result<Option<DisplayFrame<'_>>> {
        self.begun = None;
        let slot = (frame % self.slots.len() as u64) as usize;
        let info = self.feed.as_ref().map(|f| f.latest().1);
        let fg_on = self.fg.is_some();
        let (wait, display, refresh, compose) = {
            let mut state = self.shared.lock();
            while state.pending[slot] {
                state = self
                    .shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if let Some(e) = state.error.take() {
                return Err(e.into());
            }
            let display = self.request.resolve(&state.supported, info.as_ref(), fg_on);
            let compose: Vec<_> = state.compose.drain(..).collect();
            (state.released[slot], display, state.refresh, compose)
        };
        let refresh_s = refresh.interval.map(|d| d.as_secs_f64()).or_else(|| {
            info.and_then(|i| i.refresh_hz)
                .map(|hz| 1.0 / f64::from(hz))
        });
        self.read_costs(compose);
        let generated = match &mut self.fg {
            Some(fg) => match (&mut fg.dynamic, fg.mode) {
                (Some(d), _) => {
                    d.set_limits(fg.max_generated, refresh_s.map(|s| 1.0 / s));
                    d.generated()
                }
                (None, FrameGenerationMode::Fixed(n)) => n,
                _ => 0,
            },
            None => 0,
        };
        let cap = if fg_on {
            min_interval_us(generated, refresh_s, refresh.variable)
        } else {
            0
        };
        let mode = (self.low_latency.0 || fg_on, self.low_latency.1, cap);
        if self.applied_latency != Some(mode)
            && let Some(l) = &self.latency
        {
            l.set_mode(mode.0, mode.1, mode.2);
            self.applied_latency = Some(mode);
        }

        let Some(extent) = client_extent(self.window).filter(|e| !e.contains(&0)) else {
            return Ok(None);
        };
        let s = &mut self.slots[slot];
        if !s
            .backbuffer
            .as_ref()
            .is_some_and(|b| b.fits(display.encoding, extent))
        {
            s.backbuffer = Some(encoded(
                &self.device,
                display.encoding,
                extent,
                "backbuffer",
            )?);
        }
        if s.ui.as_ref().is_none_or(|u| u.extent() != extent) {
            s.ui = Some(UiLayer::new(&self.device, extent)?);
        }
        s.backbuffer_written = false;
        s.ui_drawn = false;
        s.view = None;
        s.field_used = false;
        s.recorded = 0;
        self.begun = Some(Begun {
            frame,
            slot,
            display,
            extent,
            wait,
            generated,
        });
        Ok(self.frame())
    }

    /// Feeds `Dynamic` the GPU cost of every frame whose render and compose submits have
    /// completed, in frame order. `compose` holds the present thread's new reports.
    fn read_costs(&mut self, compose: Vec<(u64, Vec<TimelinePoint>)>) {
        let Some(dynamic) = self.fg.as_mut().and_then(|f| f.dynamic.as_mut()) else {
            self.costs.clear();
            return;
        };
        for (frame, points) in compose {
            if let Some(c) = self.costs.iter_mut().find(|c| c.frame == frame) {
                c.compose = Some(points);
            }
        }
        // Reports come in frame order, so a frame before the newest report without one of
        // its own failed to present and never gets one.
        if let Some(newest) = self.costs.iter().rposition(|c| c.compose.is_some()) {
            let mut k = 0;
            self.costs.retain(|c| {
                k += 1;
                k > newest || c.compose.is_some()
            });
        }
        'frames: while let Some(c) = self.costs.front() {
            let Some(compose) = &c.compose else {
                break;
            };
            let Some(render) = self.device.pass_results(c.render) else {
                break;
            };
            let mut gpu_ms = 0.0;
            let mut fg_ms = 0.0;
            for p in &render {
                gpu_ms += p.gpu_ms;
                if p.name == FG_PASS {
                    fg_ms += p.gpu_ms;
                }
            }
            for &point in compose {
                let Some(passes) = self.device.pass_results(point) else {
                    break 'frames;
                };
                gpu_ms += passes.iter().map(|p| p.gpu_ms).sum::<f64>();
            }
            dynamic.measure(
                self.started.elapsed().as_secs_f64(),
                gpu_ms,
                fg_ms,
                c.generated,
            );
            self.costs.pop_front();
        }
        while self.costs.len() > COST_REPORTS {
            self.costs.pop_front();
        }
    }

    /// The frame the last `begin_frame` opened, until `present` hands it over.
    pub fn frame(&mut self) -> Option<DisplayFrame<'_>> {
        let b = self.begun.as_ref()?;
        Some(DisplayFrame {
            slot: &mut self.slots[b.slot],
            field: &mut self.field,
            device: &self.device,
            display: b.display,
            extent: b.extent,
            wait: b.wait,
            generated: b.generated,
        })
    }

    /// Records the open frame's frame generation into `list`, the frame's main list on the
    /// Render queue, after compose has written the backbuffer: one evaluate per generated
    /// image. Does nothing with frame generation off, for a frame whose view did not call
    /// [`DisplayFrame::frame_generation`] or that never wrote the backbuffer, or while the
    /// generated count is 0. `orr`'s `Frame::end` calls it before submitting.
    pub fn record_frame_generation(&mut self, list: &mut CommandList<'_>) -> Result<()> {
        let (Some(b), Some(fg)) = (&self.begun, &mut self.fg) else {
            return Ok(());
        };
        let slot = &mut self.slots[b.slot];
        let view = slot.view.take();
        let Some(view) = view.filter(|_| slot.backbuffer_written && b.generated > 0) else {
            fg.reset = true;
            fg.prev = None;
            return Ok(());
        };
        let n = b.generated;
        let encoding = b.display.encoding;
        let backbuffer = slot.backbuffer.as_ref().expect("sized at begin_frame");
        if fg
            .feature
            .as_ref()
            .is_none_or(|(_, e, x)| (*e, *x) != (encoding, b.extent))
        {
            fg.feature = None;
            let feature = FrameGeneration::new(
                &fg.sdk,
                &backbuffer.texture,
                encoding == OutputFormat::Hdr10Pq,
            )?;
            fg.feature = Some((feature, encoding, b.extent));
            fg.reset = true;
        }
        let prev = fg.prev.filter(|p| view.camera.continues(p));
        let reset = fg.reset || view.reset || prev.is_none();
        if reset {
            fg.anchor = view.camera.position;
        }
        let camera = fg_camera(&view.camera, &prev.unwrap_or(view.camera), fg.anchor);
        fg.prev = Some(view.camera);
        fg.reset = false;

        if slot.generated.iter().any(|g| !g.fits(encoding, b.extent)) {
            slot.generated.clear();
        }
        while slot.generated.len() < n as usize {
            let image = encoded(&self.device, encoding, b.extent, "generated image")?;
            list.discard(&image.texture);
            slot.generated.push(image);
        }
        let flag = match &slot.flag {
            Some(f) => Arc::clone(f),
            None => {
                let f = Arc::new(self.device.gpu_malloc::<u32>(
                    MemoryKind::Readback,
                    1,
                    "interpolation disabled",
                )?);
                slot.flag = Some(Arc::clone(&f));
                f
            }
        };
        let flag_ptr = flag.cpu.expect("readback memory is mapped");
        // SAFETY: the mapping lives as long as `flag`. No GPU work uses it now: the last
        // evaluate that wrote it completed before the present thread read it, which it did
        // before the slot came back to this frame.
        unsafe { flag_ptr.as_ptr().write_volatile(0) };

        let distortion = self
            .field
            .as_ref()
            .filter(|_| slot.field_used)
            .map(DistortionField::current_texture);
        let render = view.camera.extent;
        let (feature, _, _) = fg.feature.as_mut().expect("made above");
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        list.begin_pass(FG_PASS, false);
        for index in 1..=n {
            let inputs = FgInputs {
                backbuffer: &backbuffer.texture,
                depth: &view.depth,
                motion: &view.motion,
                distortion,
                interpolated: &slot.generated[index as usize - 1].texture,
                disable_interpolation: Some(&flag),
                camera,
                jitter: view.jitter,
                mvec_scale: render.map(|v| v as f32),
                reset,
                frame_id: b.frame,
                not_rendering_scene: view.not_rendering_scene,
                depth_scale: 1.0,
            };
            feature.record(list, &inputs, index, n)?;
        }
        list.end_pass();
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        slot.recorded = n;
        Ok(())
    }

    /// Hands the frame opened by the last `begin_frame` to the present thread, whose submits
    /// wait `after` (the frame's main submit), and returns without waiting. Reports an error
    /// the present thread hit since the last call.
    pub fn present(&mut self, after: TimelinePoint) -> Result<()> {
        if let Some(e) = self.shared.lock().error.take() {
            return Err(e.into());
        }
        let Some(b) = self.begun.take() else {
            return Ok(());
        };
        let s = &self.slots[b.slot];
        let recorded = s.recorded as usize;
        let job = Job {
            frame: b.frame,
            slot: b.slot,
            after,
            display: b.display,
            mode: self.request.present_mode(),
            source: s
                .backbuffer_written
                .then(|| s.backbuffer.as_ref().expect("sized").sampled),
            ui: s.ui_drawn.then(|| s.ui.as_ref().expect("sized").index()),
            generated: s.generated[..recorded].iter().map(|g| g.sampled).collect(),
            flag: s.flag.as_ref().filter(|_| recorded > 0).map(Arc::clone),
            min_images: MIN_IMAGES.max(self.fg.as_ref().map_or(0, |f| f.most() + 2)),
        };
        if self.fg.as_ref().is_some_and(|f| f.dynamic.is_some()) {
            self.costs.push_back(Cost {
                frame: b.frame,
                render: after,
                generated: s.recorded,
                compose: None,
            });
        }
        self.shared.lock().pending[b.slot] = true;
        let sent = self.jobs.as_ref().is_some_and(|j| j.send(job).is_ok());
        if !sent {
            self.shared.lock().pending[b.slot] = false;
            return Err(Error::Thread("the present thread has stopped".into()));
        }
        Ok(())
    }

    /// Waits until the present thread has submitted every frame handed to it.
    pub fn wait_idle(&self) {
        let mut state = self.shared.lock();
        while state.pending.contains(&true) {
            state = self
                .shared
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// The newest frame the present thread finished with.
    pub fn last_presented(&self) -> Option<Presented> {
        self.shared.lock().presented
    }
}

impl Drop for Display {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(t) = self.thread.take()
            && t.join().is_err()
        {
            log::error!("the present thread panicked");
        }
    }
}

fn encoded(gpu: &Device, encoding: OutputFormat, extent: [u32; 2], name: &str) -> Result<Encoded> {
    let format = match encoding {
        OutputFormat::Sdr => Format::Bgra8Unorm,
        OutputFormat::Hdr10Pq => Format::A2b10g10r10Unorm,
        OutputFormat::ScRgb => Format::Rgba16Float,
    };
    let desc = TextureDesc::d2(
        format,
        extent[0],
        extent[1],
        TextureUsage::SAMPLED | TextureUsage::STORAGE,
    );
    let texture = gpu.create_texture(&desc, TextureMemory::Dedicated, name)?;
    let view = ViewDesc::default();
    let storage = gpu.write_storage(gpu.alloc_image_slot()?, &texture, &view);
    let sampled = gpu.write_sampled(gpu.alloc_image_slot()?, &texture, &view);
    Ok(Encoded {
        texture,
        storage,
        sampled,
        encoding,
        device: gpu.clone(),
    })
}

/// The window's client area in pixels.
#[cfg(windows)]
fn client_extent(hwnd: isize) -> Option<[u32; 2]> {
    use windows::Win32::Foundation::{HWND, RECT};
    use windows::Win32::UI::WindowsAndMessaging::GetClientRect;
    let mut rc = RECT::default();
    // SAFETY: the display's window outlives it; GetClientRect may be called from any thread.
    unsafe { GetClientRect(HWND(hwnd as *mut std::ffi::c_void), &mut rc) }.ok()?;
    Some([(rc.right - rc.left) as u32, (rc.bottom - rc.top) as u32])
}

#[cfg(not(windows))]
fn client_extent(_hwnd: isize) -> Option<[u32; 2]> {
    None
}

/// The present thread's state.
struct Presenter {
    device: Device,
    swapchain: Swapchain,
    latency: Option<LowLatency>,
    compose: ComposeKernel,
    /// Per frame slot, one compose root per presented image, with the submit that last
    /// read it.
    roots: Vec<Vec<(ComposeRoot, Option<TimelinePoint>)>>,
    /// The encoding and metadata last given to the swapchain.
    format: Option<(OutputFormat, Option<HdrMetadata>)>,
}

impl Presenter {
    fn run(mut self, jobs: &Receiver<Job>, shared: &Shared) {
        while let Ok(job) = jobs.recv() {
            let result = self.present(&job);
            let refresh = self.swapchain.refresh();
            let mut state = shared.lock();
            state.pending[job.slot] = false;
            match result {
                Ok((presented, points)) => {
                    state.presented = Some(presented);
                    if let Some(&last) = points.last() {
                        state.released[job.slot] = Some(last);
                    }
                    if state.compose.len() == COST_REPORTS {
                        state.compose.pop_front();
                    }
                    state.compose.push_back((job.frame, points));
                }
                Err(e) => state.error = Some(e),
            }
            let supported = self.swapchain.supported_formats();
            if state.supported != supported {
                state.supported = supported.to_vec();
            }
            state.refresh = refresh;
            drop(state);
            shared.changed.notify_all();
        }
    }

    /// The present path of one frame, bracketed by `PresentStart` and `PresentEnd`.
    fn present(&mut self, job: &Job) -> Result<(Presented, Vec<TimelinePoint>), orr_gpu::Error> {
        self.mark(job.frame, LatencyMarker::PresentStart);
        let result = self.compose_and_present(job);
        self.mark(job.frame, LatencyMarker::PresentEnd);
        result
    }

    fn mark(&self, frame: u64, marker: LatencyMarker) {
        if let Some(l) = &self.latency {
            l.marker(frame, marker);
        }
    }

    fn compose_and_present(
        &mut self,
        job: &Job,
    ) -> Result<(Presented, Vec<TimelinePoint>), orr_gpu::Error> {
        let encoding = job.display.encoding;
        let mut presented = Presented {
            frame: job.frame,
            present_id: None,
            generated: 0,
            encoding,
            extent: [0; 2],
        };
        let format = (encoding, job.display.hdr_metadata());
        if self.format != Some(format) {
            if !self.swapchain.supported_formats().contains(&encoding) {
                return Ok((presented, Vec::new()));
            }
            self.swapchain.set_format(format.0, format.1)?;
            self.format = Some(format);
        }
        self.swapchain.set_min_images(job.min_images);
        self.swapchain.set_present_mode(job.mode);
        let queue = self
            .device
            .queue(QueueKind::Present)
            .expect("the swapchain needs the Present queue");

        let mut sources = Vec::with_capacity(job.generated.len() + 1);
        if let Some(flag) = &job.flag {
            // NGX sets the flag in the frame's evaluates.
            queue.wait(job.after, ROOT_TIMEOUT)?;
            if !FrameGeneration::interpolation_disabled(flag) {
                sources.extend(job.generated.iter().copied().map(ComposeSource::Encoded));
            }
        }
        let generated = sources.len() as u32;
        sources.push(
            job.source
                .map_or(ComposeSource::Black, ComposeSource::Encoded),
        );
        let batch = sources.len() as u32;
        let roots = &mut self.roots[job.slot];
        while roots.len() < sources.len() {
            roots.push((ComposeRoot::new(&self.device)?, None));
        }

        let mut points = Vec::with_capacity(sources.len());
        for (k, (source, (root, last))) in sources.into_iter().zip(roots.iter_mut()).enumerate() {
            let Some(image) = self.swapchain.acquire()? else {
                break;
            };
            if let Some(p) = last.take() {
                queue.wait(p, ROOT_TIMEOUT)?;
            }
            let mut list = queue.command_list()?;
            self.compose.record(
                &mut list,
                root,
                &ComposePass {
                    source,
                    ui: job.ui,
                    target: image.storage(),
                    extent: image.extent(),
                    display: &job.display,
                },
            );
            let point = queue.submit_present(vec![list], &[job.after], &image)?;
            *last = Some(point);
            points.push(point);
            presented.encoding = image.format();
            presented.extent = image.extent();
            let metered = (k == 0 && batch > 1).then_some(batch);
            let id = self.swapchain.present(image, metered)?;
            if (k as u32) < generated {
                presented.generated += 1;
            } else {
                presented.present_id = Some(id);
            }
        }
        Ok((presented, points))
    }
}
