//! A window's display: the encoded backbuffer and UI layer per frame in flight on the render
//! thread, and a present thread that owns the swapchain, composes the UI over each presented
//! image and presents it.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use orr_scene::orr_gpu::{
    self, CommandList, Device, Format, HdrMetadata, LatencyMarker, LowLatency, OutputFormat,
    PresentMode, QueueKind, StorageImageIndex, Swapchain, SwapchainDesc, Texture, TextureDesc,
    TextureIndex, TextureMemory, TextureUsage, TimelinePoint, ViewDesc,
};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::passes::ComposeKernel;
use crate::{
    ComposePass, ComposeRoot, ComposeSource, DisplayFeed, DisplayRequest, Error, ResolvedDisplay,
    Result, UiLayer,
};

/// How long the present thread waits for a compose root's previous submit, which is a frame
/// slot's worth of frames old.
const ROOT_TIMEOUT: Duration = Duration::from_secs(10);

/// A window's display target. The render thread draws each frame into the frame slot
/// [`Display::begin_frame`] returns and hands it over with [`Display::present`]; the present
/// thread then acquires a swapchain image, blends the frame's UI layer over the backbuffer
/// into it on the Present queue and presents it, so the render thread never presents.
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
    shared: Arc<Shared>,
    jobs: Option<Sender<Job>>,
    thread: Option<JoinHandle<()>>,
    device: Device,
}

/// One frame slot: the images a frame draws into, reused `frames_in_flight` frames later.
struct Slot {
    backbuffer: Option<Backbuffer>,
    ui: Option<UiLayer>,
    backbuffer_written: bool,
    ui_drawn: bool,
}

/// The UI-free encoded image compose writes, in the swapchain's format for its encoding.
struct Backbuffer {
    texture: Texture,
    storage: StorageImageIndex,
    sampled: TextureIndex,
    encoding: OutputFormat,
    device: Device,
}

impl Drop for Backbuffer {
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
}

/// State the present thread reports back.
struct State {
    /// A slot's frame is with the present thread.
    pending: Vec<bool>,
    /// The Present-queue point after which a slot's images are free again.
    released: Vec<Option<TimelinePoint>>,
    supported: Vec<OutputFormat>,
    presented: Option<Presented>,
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
    /// What the frame encodes for; tonemap and compose take it.
    pub display: ResolvedDisplay,
    /// The window's client extent, the size of the backbuffer and UI layer.
    pub extent: [u32; 2],
    /// The frame's main submit waits this point, after which the present thread no longer
    /// reads the slot's images.
    pub wait: Option<TimelinePoint>,
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
}

/// What the present thread did with a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Presented {
    /// The id passed to [`Display::begin_frame`].
    pub frame: u64,
    /// The swapchain's present id; `None` when the frame was not presented (the window was
    /// minimised, or the surface stopped offering its encoding).
    pub present_id: Option<u64>,
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
                present_mode: request.present_mode(),
                ..SwapchainDesc::default()
            },
        )?;
        let latency = swapchain.latency();
        let compose = ComposeKernel::new(gpu)?;
        let n = frames_in_flight.max(1) as usize;
        let roots = (0..n)
            .map(|_| ComposeRoot::new(gpu))
            .collect::<Result<Vec<_>>>()?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                pending: vec![false; n],
                released: vec![None; n],
                supported: swapchain.supported_formats().to_vec(),
                presented: None,
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
                roots: roots.into_iter().map(|r| (r, None)).collect(),
                format: None,
            };
            std::thread::Builder::new()
                .name("orr present".into())
                .spawn(move || presenter.run(&rx, &shared))
                .map_err(|e| Error::Thread(e.to_string()))?
        };
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
                })
                .collect(),
            begun: None,
            latency,
            shared,
            jobs: Some(jobs),
            thread: Some(thread),
            device: gpu.clone(),
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

    /// The swapchain's low-latency handle; `None` without `Caps::low_latency`.
    pub fn latency(&self) -> Option<&LowLatency> {
        self.latency.as_ref()
    }

    /// Opens frame `frame` (orr's frame id) for drawing: waits until the present thread has
    /// submitted the slot's previous frame, resolves the display and sizes the slot's images
    /// to the window. `None` while the window has no area; the frame then draws nothing into
    /// the display and `present` does nothing.
    pub fn begin_frame(&mut self, frame: u64) -> Result<Option<DisplayFrame<'_>>> {
        self.begun = None;
        let slot = (frame % self.slots.len() as u64) as usize;
        let (wait, supported) = {
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
            (state.released[slot], state.supported.clone())
        };
        let Some(extent) = client_extent(self.window).filter(|e| !e.contains(&0)) else {
            return Ok(None);
        };
        let info = self.feed.as_ref().map(|f| f.latest().1);
        let display = self.request.resolve(&supported, info.as_ref(), false);

        let s = &mut self.slots[slot];
        let fits = |b: &Backbuffer| {
            b.encoding == display.encoding && b.texture.desc().extent[..2] == extent
        };
        if !s.backbuffer.as_ref().is_some_and(fits) {
            s.backbuffer = Some(backbuffer(&self.device, display.encoding, extent)?);
        }
        if s.ui.as_ref().is_none_or(|u| u.extent() != extent) {
            s.ui = Some(UiLayer::new(&self.device, extent)?);
        }
        s.backbuffer_written = false;
        s.ui_drawn = false;
        self.begun = Some(Begun {
            frame,
            slot,
            display,
            extent,
            wait,
        });
        Ok(self.frame())
    }

    /// The frame the last `begin_frame` opened, until `present` hands it over.
    pub fn frame(&mut self) -> Option<DisplayFrame<'_>> {
        let b = self.begun.as_ref()?;
        Some(DisplayFrame {
            slot: &mut self.slots[b.slot],
            display: b.display,
            extent: b.extent,
            wait: b.wait,
        })
    }

    /// Hands the frame opened by the last `begin_frame` to the present thread, whose submit
    /// waits `after` (the frame's main submit), and returns without waiting. Reports an error
    /// the present thread hit since the last call.
    pub fn present(&mut self, after: TimelinePoint) -> Result<()> {
        if let Some(e) = self.shared.lock().error.take() {
            return Err(e.into());
        }
        let Some(b) = self.begun.take() else {
            return Ok(());
        };
        let s = &self.slots[b.slot];
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
        };
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

fn backbuffer(gpu: &Device, encoding: OutputFormat, extent: [u32; 2]) -> Result<Backbuffer> {
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
    let texture = gpu.create_texture(&desc, TextureMemory::Dedicated, "backbuffer")?;
    let view = ViewDesc::default();
    let storage = gpu.write_storage(gpu.alloc_image_slot()?, &texture, &view);
    let sampled = gpu.write_sampled(gpu.alloc_image_slot()?, &texture, &view);
    Ok(Backbuffer {
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
    /// One compose root per frame slot, with the submit that last read it.
    roots: Vec<(ComposeRoot, Option<TimelinePoint>)>,
    /// The encoding and metadata last given to the swapchain.
    format: Option<(OutputFormat, Option<HdrMetadata>)>,
}

impl Presenter {
    fn run(mut self, jobs: &Receiver<Job>, shared: &Shared) {
        while let Ok(job) = jobs.recv() {
            let result = self.present(&job);
            let mut state = shared.lock();
            state.pending[job.slot] = false;
            match result {
                Ok((presented, point)) => {
                    state.presented = Some(presented);
                    if point.is_some() {
                        state.released[job.slot] = point;
                    }
                }
                Err(e) => state.error = Some(e),
            }
            state.supported = self.swapchain.supported_formats().to_vec();
            drop(state);
            shared.changed.notify_all();
        }
    }

    /// The present path of one frame, bracketed by `PresentStart` and `PresentEnd`.
    fn present(&mut self, job: &Job) -> Result<(Presented, Option<TimelinePoint>), orr_gpu::Error> {
        let latency = self.latency.clone();
        let mark = |m| {
            if let Some(l) = &latency {
                l.marker(job.frame, m);
            }
        };
        mark(LatencyMarker::PresentStart);
        let result = self.compose_and_present(job);
        mark(LatencyMarker::PresentEnd);
        result
    }

    fn compose_and_present(
        &mut self,
        job: &Job,
    ) -> Result<(Presented, Option<TimelinePoint>), orr_gpu::Error> {
        let encoding = job.display.encoding;
        let skipped = Presented {
            frame: job.frame,
            present_id: None,
            encoding,
            extent: [0; 2],
        };
        let format = (encoding, job.display.hdr_metadata());
        if self.format != Some(format) {
            if !self.swapchain.supported_formats().contains(&encoding) {
                return Ok((skipped, None));
            }
            self.swapchain.set_format(format.0, format.1)?;
            self.format = Some(format);
        }
        self.swapchain.set_present_mode(job.mode);
        let Some(image) = self.swapchain.acquire()? else {
            return Ok((skipped, None));
        };
        let queue = self
            .device
            .queue(QueueKind::Present)
            .expect("the swapchain needs the Present queue");
        let (root, last) = &mut self.roots[job.slot];
        if let Some(p) = last.take() {
            queue.wait(p, ROOT_TIMEOUT)?;
        }
        let mut list = queue.command_list()?;
        let source = job
            .source
            .map_or(ComposeSource::Black, ComposeSource::Encoded);
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
        let (encoding, extent) = (image.format(), image.extent());
        let id = self.swapchain.present(image, None)?;
        let presented = Presented {
            frame: job.frame,
            present_id: Some(id),
            encoding,
            extent,
        };
        Ok((presented, Some(point)))
    }
}
