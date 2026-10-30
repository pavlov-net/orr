//! Swapchains, presents and Reflex low latency. One lock per swapchain covers every call
//! that names it.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use ash::vk;
use ash::vk::TaggedStructure as _;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::debug::name_object;
use super::device::{Device, PresentFns, Shared};
use super::queue::QueueKind;
use super::texture::Texture;
use crate::{
    Error, Format, HdrMetadata, LatencyMarker, LatencyTimings, OutputFormat, PresentMode, Refresh,
    StorageImageIndex, SwapchainDesc, TextureDesc, TextureUsage, ViewDesc,
};

const ACQUIRE_TIMEOUT: Duration = Duration::from_secs(5);
/// How long dropping a swapchain waits for each outstanding present.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
/// How long `LowLatency::sleep` waits for the driver to release the frame.
const SLEEP_TIMEOUT: Duration = Duration::from_secs(1);
/// Reports one `LowLatency::timings` call can return; the driver keeps 64.
const MAX_REPORTS: usize = 64;

/// Surface formats each output format can use, in order of preference.
fn candidates(f: OutputFormat) -> &'static [(vk::Format, vk::ColorSpaceKHR)] {
    match f {
        OutputFormat::Sdr => &[
            (
                vk::Format::B8G8R8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
            (
                vk::Format::R8G8B8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
        ],
        OutputFormat::Hdr10Pq => &[(
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT,
        )],
        OutputFormat::ScRgb => &[(
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
        )],
    }
}

/// Each output format the surface offers, with the surface format it uses.
fn negotiate(surface: &[vk::SurfaceFormatKHR]) -> Vec<(OutputFormat, vk::SurfaceFormatKHR)> {
    [
        OutputFormat::Sdr,
        OutputFormat::Hdr10Pq,
        OutputFormat::ScRgb,
    ]
    .into_iter()
    .filter_map(|f| {
        candidates(f).iter().find_map(|&(format, space)| {
            surface
                .iter()
                .find(|s| s.format == format && s.color_space == space)
                .map(|s| (f, *s))
        })
    })
    .collect()
}

fn texture_format(f: vk::Format) -> Format {
    match f {
        vk::Format::B8G8R8A8_UNORM => Format::Bgra8Unorm,
        vk::Format::R8G8B8A8_UNORM => Format::Rgba8Unorm,
        vk::Format::A2B10G10R10_UNORM_PACK32 => Format::A2b10g10r10Unorm,
        vk::Format::R16G16B16A16_SFLOAT => Format::Rgba16Float,
        other => unreachable!("{other:?} is not a negotiated swapchain format"),
    }
}

fn vk_mode(m: PresentMode) -> vk::PresentModeKHR {
    match m {
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
    }
}

fn vk_marker(m: LatencyMarker) -> vk::LatencyMarkerNV {
    use vk::LatencyMarkerNV as M;
    match m {
        LatencyMarker::SimulationStart => M::SIMULATION_START,
        LatencyMarker::SimulationEnd => M::SIMULATION_END,
        LatencyMarker::RenderSubmitStart => M::RENDERSUBMIT_START,
        LatencyMarker::RenderSubmitEnd => M::RENDERSUBMIT_END,
        LatencyMarker::PresentStart => M::PRESENT_START,
        LatencyMarker::PresentEnd => M::PRESENT_END,
        LatencyMarker::InputSample => M::INPUT_SAMPLE,
    }
}

/// The low-latency sleep mode, restored on every new swapchain.
#[derive(Clone, Copy)]
struct SleepMode {
    low_latency: bool,
    boost: bool,
    min_interval_us: u32,
}

/// What the swapchain's lock guards.
struct Locked {
    /// The current swapchain; null while there is none.
    raw: vk::SwapchainKHR,
    /// The current swapchain was created with low latency on.
    latency: bool,
    mode: Option<SleepMode>,
}

impl Locked {
    /// The swapchain low-latency calls name, if any.
    fn latency_raw(&self) -> Option<vk::SwapchainKHR> {
        self.latency.then_some(self.raw)
    }
}

/// The lock a [`Swapchain`] and its [`LowLatency`] handles share.
struct Lock {
    shared: Arc<Shared>,
    locked: Mutex<Locked>,
    /// Timeline semaphore `vkLatencySleepNV` signals; null without low latency.
    sleep: vk::Semaphore,
    sleep_value: AtomicU64,
}

impl Lock {
    fn lock(&self) -> MutexGuard<'_, Locked> {
        self.locked.lock().unwrap()
    }

    fn fns(&self) -> &PresentFns {
        self.shared
            .present_fn
            .as_ref()
            .expect("a presenting device")
    }

    fn latency_fn(&self) -> &ash::nv::low_latency2::Device {
        self.fns().latency.as_ref().expect("low latency is on")
    }

    /// Applies `mode` to `raw`; the caller holds the lock.
    fn apply_mode(&self, raw: vk::SwapchainKHR, mode: SleepMode) {
        let info = vk::LatencySleepModeInfoNV::default()
            .low_latency_mode(mode.low_latency)
            .low_latency_boost(mode.boost)
            .minimum_interval_us(mode.min_interval_us);
        // SAFETY: `raw` is live and the lock is held.
        if let Err(e) = unsafe { self.latency_fn().set_latency_sleep_mode(raw, Some(&info)) } {
            log::warn!("vkSetLatencySleepModeNV failed: {e}");
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        if self.sleep != vk::Semaphore::null() {
            // SAFETY: every sleep waited its value before returning; nothing else uses it.
            unsafe { self.shared.device.destroy_semaphore(self.sleep, None) };
        }
    }
}

/// Reflex low latency for one swapchain: Send, Sync and cheap to clone. Every call takes
/// the swapchain's lock for the Vulkan call only, and does nothing while the swapchain has
/// no image chain (minimised before its first image).
#[derive(Clone)]
pub struct LowLatency {
    lock: Arc<Lock>,
}

impl LowLatency {
    /// `vkSetLatencySleepModeNV`; kept across swapchain recreation. `min_interval_us` caps
    /// the frame rate (0: no cap).
    pub fn set_mode(&self, low_latency: bool, boost: bool, min_interval_us: u32) {
        let mode = SleepMode {
            low_latency,
            boost,
            min_interval_us,
        };
        let mut locked = self.lock.lock();
        locked.mode = Some(mode);
        if let Some(raw) = locked.latency_raw() {
            self.lock.apply_mode(raw, mode);
        }
    }

    /// Waits until the driver releases the next frame (`vkLatencySleepNV`). Call once per
    /// frame before sampling input; the wait runs outside the lock.
    pub fn sleep(&self) {
        let value = {
            let locked = self.lock.lock();
            let Some(raw) = locked.latency_raw() else {
                return;
            };
            let value = self.lock.sleep_value.fetch_add(1, Ordering::Relaxed) + 1;
            let info = vk::LatencySleepInfoNV::default()
                .signal_semaphore(self.lock.sleep)
                .value(value);
            // SAFETY: the swapchain and semaphore are live and the lock is held.
            if let Err(e) = unsafe { self.lock.latency_fn().latency_sleep(raw, &info) } {
                log::warn!("vkLatencySleepNV failed: {e}");
                return;
            }
            value
        };
        let semaphores = [self.lock.sleep];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the semaphore is live.
        let waited = unsafe {
            self.lock
                .shared
                .device
                .wait_semaphores(&info, SLEEP_TIMEOUT.as_nanos() as u64)
        };
        if let Err(e) = waited {
            log::warn!("low-latency sleep: {e}");
        }
    }

    /// Records `marker` for `frame_id` (`vkSetLatencyMarkerNV`). Every marker of a frame
    /// carries the same strictly increasing id; reports are keyed by it.
    pub fn marker(&self, frame_id: u64, marker: LatencyMarker) {
        let locked = self.lock.lock();
        let Some(raw) = locked.latency_raw() else {
            return;
        };
        let info = vk::SetLatencyMarkerInfoNV::default()
            .present_id(frame_id)
            .marker(vk_marker(marker));
        // SAFETY: the swapchain is live and the lock is held.
        unsafe { self.lock.latency_fn().set_latency_marker(raw, &info) };
    }

    /// Appends the driver's latency reports (`vkGetLatencyTimingsNV`), one per frame id,
    /// oldest first.
    pub fn timings(&self, out: &mut Vec<LatencyTimings>) {
        let mut reports = [vk::LatencyTimingsFrameReportNV::default(); MAX_REPORTS];
        let count = {
            let locked = self.lock.lock();
            let Some(raw) = locked.latency_raw() else {
                return;
            };
            let mut info = vk::GetLatencyMarkerInfoNV::default().timings(&mut reports);
            // SAFETY: the swapchain is live, the lock is held and `reports` has room for
            // `timing_count` entries.
            unsafe { self.lock.latency_fn().get_latency_timings(raw, &mut info) };
            info.timing_count as usize
        };
        let start = out.len();
        out.extend(reports[..count].iter().map(|r| LatencyTimings {
            frame_id: r.present_id,
            input_sample: r.input_sample_time_us,
            simulation_start: r.sim_start_time_us,
            simulation_end: r.sim_end_time_us,
            render_submit_start: r.render_submit_start_time_us,
            render_submit_end: r.render_submit_end_time_us,
            present_start: r.present_start_time_us,
            present_end: r.present_end_time_us,
            driver_start: r.driver_start_time_us,
            driver_end: r.driver_end_time_us,
            os_render_queue_start: r.os_render_queue_start_time_us,
            os_render_queue_end: r.os_render_queue_end_time_us,
            gpu_render_start: r.gpu_render_start_time_us,
            gpu_render_end: r.gpu_render_end_time_us,
        }));
        out[start..].sort_by_key(|t| t.frame_id);
    }
}

/// An acquired swapchain image. Submit the work that writes it with
/// [`crate::Queue::submit_present`], then hand it to [`Swapchain::present`]; every acquired
/// frame is presented.
pub struct SwapchainFrame {
    texture: Arc<Texture>,
    storage: StorageImageIndex,
    format: OutputFormat,
    index: u32,
    raw: vk::SwapchainKHR,
    generation: u64,
    slot: usize,
    pub(crate) acquire: vk::Semaphore,
    pub(crate) present: vk::Semaphore,
    pub(crate) submitted: Cell<bool>,
}

impl SwapchainFrame {
    /// The image, for colour targets and copies. It is in `GENERAL` while the submit's lists
    /// run.
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// The image's storage view.
    pub fn storage(&self) -> StorageImageIndex {
        self.storage
    }

    pub fn format(&self) -> OutputFormat {
        self.format
    }

    pub fn extent(&self) -> [u32; 2] {
        let e = self.texture.desc().extent;
        [e[0], e[1]]
    }

    /// The image's index in its swapchain.
    pub fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    /// Holds an acquired image of this generation.
    Acquired(u64),
    /// Presented on this generation; free once its fence signals.
    Presented(u64),
}

/// The semaphores of one acquire-to-present, and the present fence that frees them.
struct Slot {
    acquire: vk::Semaphore,
    present: vk::Semaphore,
    fence: vk::Fence,
    state: SlotState,
}

/// One `VkSwapchainKHR` and its images.
struct Generation {
    id: u64,
    raw: vk::SwapchainKHR,
    images: Vec<(Arc<Texture>, StorageImageIndex)>,
    format: OutputFormat,
    extent: [u32; 2],
    /// The present modes a present can switch between.
    modes: Vec<vk::PresentModeKHR>,
    present_ids: bool,
}

/// A window's swapchain. Owned by one thread, which acquires and presents; [`LowLatency`]
/// handles reach it from others through its lock. Recreation on resize, format change and
/// out-of-date happens inside [`Swapchain::acquire`] without waiting for the device.
pub struct Swapchain {
    device: Device,
    lock: Arc<Lock>,
    surface: vk::SurfaceKHR,
    current: Option<Generation>,
    /// Replaced swapchains, destroyed once their presents complete.
    retired: Vec<Generation>,
    generations: u64,
    slots: Vec<Slot>,
    next_slot: usize,
    formats: Vec<(OutputFormat, vk::SurfaceFormatKHR)>,
    supported: Vec<OutputFormat>,
    desc: SwapchainDesc,
    /// The format and metadata asked for; applied at the next acquire.
    format: OutputFormat,
    hdr: Option<HdrMetadata>,
    hdr_pending: bool,
    mode: PresentMode,
    /// The last acquire or present reported out of date or suboptimal.
    stale: bool,
    next_present_id: u64,
}

impl Swapchain {
    /// Creates a swapchain for a Win32 window (the display handle is not needed there). The
    /// device needs the Present queue (`QueueRequest::present`).
    pub fn new(
        device: &Device,
        window: RawWindowHandle,
        _display: RawDisplayHandle,
        desc: &SwapchainDesc,
    ) -> Result<Swapchain, Error> {
        let shared = &device.shared;
        let fns = shared.present_fn.as_ref().ok_or_else(|| {
            Error::Invalid("the device was created without QueueRequest::present".into())
        })?;
        let sleep = if shared.caps.low_latency {
            let mut ty = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            // SAFETY: valid create info.
            unsafe {
                shared
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default().push(&mut ty), None)
            }
            .map_err(Error::vk("vkCreateSemaphore"))?
        } else {
            vk::Semaphore::null()
        };
        let lock = Arc::new(Lock {
            shared: Arc::clone(shared),
            locked: Mutex::new(Locked {
                raw: vk::SwapchainKHR::null(),
                latency: false,
                mode: None,
            }),
            sleep,
            sleep_value: AtomicU64::new(0),
        });
        let surface = match window {
            RawWindowHandle::Win32(w) => {
                let hinstance = w.hinstance.ok_or_else(|| {
                    Error::Invalid("the Win32 window handle has no HINSTANCE".into())
                })?;
                let info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hwnd(w.hwnd.get())
                    .hinstance(hinstance.get());
                // SAFETY: the caller's handle names a live window.
                unsafe { fns.win32.create_win32_surface(&info, None) }
                    .map_err(Error::vk("vkCreateWin32SurfaceKHR"))?
            }
            other => {
                return Err(Error::Unsupported {
                    device: shared.caps.device_name.clone(),
                    missing: format!("a surface for {other:?}"),
                });
            }
        };
        // Dropping `s` destroys the surface on the error paths below.
        let mut s = Swapchain {
            device: device.clone(),
            lock,
            surface,
            current: None,
            retired: Vec::new(),
            generations: 0,
            slots: Vec::new(),
            next_slot: 0,
            formats: Vec::new(),
            supported: Vec::new(),
            desc: *desc,
            format: desc.format,
            hdr: desc.hdr,
            hdr_pending: false,
            mode: desc.present_mode,
            stale: false,
            next_present_id: 1,
        };
        let family = device
            .queue(QueueKind::Present)
            .expect("presenting devices have the Present queue")
            .family;
        // SAFETY: the surface and physical device are live.
        let presents = unsafe {
            fns.surface
                .get_physical_device_surface_support(shared.physical, family, surface)
        };
        if presents != Ok(true) {
            return Err(Error::Unsupported {
                device: shared.caps.device_name.clone(),
                missing: "presenting to this window from the Present queue".into(),
            });
        }
        s.query_formats()?;
        s.set_format(desc.format, desc.hdr)?;
        s.recreate()?;
        Ok(s)
    }

    fn fns(&self) -> &PresentFns {
        self.lock.fns()
    }

    fn shared(&self) -> &Shared {
        &self.device.shared
    }

    /// The output formats the surface offers.
    pub fn supported_formats(&self) -> &[OutputFormat] {
        &self.supported
    }

    /// Switches the encoding and HDR metadata at the next acquire. The swapchain is
    /// recreated only when the format changes.
    pub fn set_format(&mut self, f: OutputFormat, hdr: Option<HdrMetadata>) -> Result<(), Error> {
        if !self.supported.contains(&f) {
            return Err(Error::Invalid(format!(
                "the surface does not offer {f:?} (it offers {:?})",
                self.supported
            )));
        }
        self.format = f;
        self.hdr = hdr;
        self.hdr_pending = true;
        Ok(())
    }

    /// Switches between FIFO and IMMEDIATE at the next present, without recreation where the
    /// surface allows it.
    pub fn set_present_mode(&mut self, m: PresentMode) {
        self.mode = m;
    }

    /// Returns a low-latency handle sharing this swapchain's lock; `None` without
    /// `Caps::low_latency`.
    pub fn latency(&self) -> Option<LowLatency> {
        (self.lock.sleep != vk::Semaphore::null()).then(|| LowLatency {
            lock: Arc::clone(&self.lock),
        })
    }

    /// The display's refresh from `vkGetSwapchainTimingPropertiesEXT`; unknown and fixed
    /// without `Caps::present_timing` or before the driver knows it.
    pub fn refresh(&self) -> Refresh {
        let unknown = Refresh {
            interval: None,
            variable: false,
        };
        let Some(timing) = self.fns().timing.as_ref() else {
            return unknown;
        };
        let locked = self.lock.lock();
        if locked.raw == vk::SwapchainKHR::null() {
            return unknown;
        }
        let mut props = vk::SwapchainTimingPropertiesEXT::default();
        let mut counter = 0u64;
        // SAFETY: the swapchain was created with present timing and the lock is held.
        let r = unsafe {
            (timing.fp().get_swapchain_timing_properties_ext)(
                self.shared().device.handle(),
                locked.raw,
                &mut props,
                &mut counter,
            )
        };
        if r != vk::Result::SUCCESS {
            return unknown;
        }
        Refresh {
            interval: (props.refresh_duration != 0)
                .then(|| Duration::from_nanos(props.refresh_duration)),
            variable: props.refresh_interval == u64::MAX,
        }
    }

    /// Waits up to `timeout` for the present `id` (from [`Swapchain::present`]) to reach the
    /// display (`vkWaitForPresent2KHR`). False on timeout. The lock is held while waiting,
    /// so keep the timeout short. Needs `Caps::present_wait`.
    pub fn wait_presented(&self, id: u64, timeout: Duration) -> Result<bool, Error> {
        let wait2 = self
            .fns()
            .wait2
            .as_ref()
            .ok_or_else(|| Error::Invalid("present waits need Caps::present_wait".into()))?;
        let locked = self.lock.lock();
        if locked.raw == vk::SwapchainKHR::null() {
            return Ok(false);
        }
        let info = vk::PresentWait2InfoKHR::default()
            .present_id(id)
            .timeout(timeout.as_nanos().min(u128::from(u64::MAX)) as u64);
        // SAFETY: the swapchain is live and the lock is held.
        match unsafe { wait2.wait_for_present2(locked.raw, &info) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => Ok(true),
            Err(e) => Err(self.shared().vk_error("vkWaitForPresent2KHR", e)),
        }
    }

    /// Acquires the next image. `None` while the window has no area (minimised); the caller
    /// skips this image's work and present. Recreates the swapchain first when the window
    /// size, the format or the surface changed.
    pub fn acquire(&mut self) -> Result<Option<SwapchainFrame>, Error> {
        self.destroy_finished();
        let format_changed = self
            .current
            .as_ref()
            .is_some_and(|g| g.format != self.format);
        let extent = self.surface_extent()?;
        if extent.contains(&0) {
            return Ok(None);
        }
        let resized = self.current.as_ref().is_none_or(|g| g.extent != extent);
        if self.stale || format_changed || resized {
            self.recreate()?;
        }
        for _ in 0..3 {
            let Some(g) = &self.current else {
                return Ok(None);
            };
            let (gen_id, raw) = (g.id, g.raw);
            let slot = self.free_slot()?;
            let acquired = {
                let _locked = self.lock.lock();
                // SAFETY: the swapchain and semaphore are live, the semaphore is unsignaled
                // and the lock is held.
                unsafe {
                    self.fns().swapchain.acquire_next_image(
                        raw,
                        ACQUIRE_TIMEOUT.as_nanos() as u64,
                        self.slots[slot].acquire,
                        vk::Fence::null(),
                    )
                }
            };
            match acquired {
                Ok((index, suboptimal)) => {
                    self.stale |= suboptimal;
                    self.slots[slot].state = SlotState::Acquired(gen_id);
                    if self.hdr_pending {
                        self.apply_hdr();
                    }
                    let g = self.current.as_ref().unwrap();
                    let (texture, storage) = &g.images[index as usize];
                    return Ok(Some(SwapchainFrame {
                        texture: Arc::clone(texture),
                        storage: *storage,
                        format: g.format,
                        index,
                        raw,
                        generation: gen_id,
                        slot,
                        acquire: self.slots[slot].acquire,
                        present: self.slots[slot].present,
                        submitted: Cell::new(false),
                    }));
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate()?,
                Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => return Err(Error::Timeout),
                Err(e) => return Err(self.shared().vk_error("vkAcquireNextImageKHR", e)),
            }
        }
        Ok(None)
    }

    /// Presents `frame` once its submit has run, and returns its present id: strictly
    /// increasing across every present of this swapchain. `batch = Some(n)` makes this the
    /// first present of a metered batch of `n` (with `Caps::present_metering`); the driver
    /// spaces the batch evenly over the frame. An out-of-date present is dropped and the
    /// next acquire recreates the swapchain.
    pub fn present(&mut self, frame: SwapchainFrame, batch: Option<u32>) -> Result<u64, Error> {
        assert!(
            frame.submitted.get(),
            "a swapchain frame is presented after Queue::submit_present"
        );
        let id = self.next_present_id;
        self.next_present_id += 1;
        let owner = self
            .current
            .iter()
            .chain(&self.retired)
            .find(|g| g.id == frame.generation)
            .expect("the frame's swapchain is alive until its present completes");
        let wanted = vk_mode(self.mode);
        let listed = owner.modes.contains(&wanted);
        let mode = if listed { wanted } else { owner.modes[0] };
        let present_ids = owner.present_ids;
        // A mode the swapchain cannot switch to needs a new swapchain.
        self.stale |= !listed;
        let slot = &self.slots[frame.slot];
        let waits = [slot.present];
        let swapchains = [frame.raw];
        let indices = [frame.index];
        let ids = [id];
        let fences = [slot.fence];
        let modes = [mode];
        let mut fence_info = vk::SwapchainPresentFenceInfoKHR::default().fences(&fences);
        let mut mode_info = vk::SwapchainPresentModeInfoKHR::default().present_modes(&modes);
        let mut id_info = vk::PresentId2KHR::default().present_ids(&ids);
        let mut meter = vk::SetPresentConfigNV::default().num_frames_per_batch(batch.unwrap_or(0));
        let mut info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices)
            .push(&mut fence_info)
            .push(&mut mode_info);
        if present_ids {
            info = info.push(&mut id_info);
        }
        if batch.is_some() && self.shared().caps.present_metering {
            info = info.push(&mut meter);
        }
        let queue = self
            .device
            .queue(QueueKind::Present)
            .expect("presenting devices have the Present queue");
        let result = {
            let _locked = self.lock.lock();
            // SAFETY: the swapchain, semaphore and fence are live, the fence is unsignaled,
            // the lock is held and `with_raw` holds the queue.
            queue.with_raw(|q| unsafe { self.fns().swapchain.queue_present(q, &info) })
        };
        self.slots[frame.slot].state = SlotState::Presented(frame.generation);
        match result {
            Ok(suboptimal) => self.stale |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.stale = true,
            Err(e) => return Err(self.shared().vk_error("vkQueuePresentKHR", e)),
        }
        Ok(id)
    }

    /// The window's client extent as the surface reports it.
    fn surface_extent(&self) -> Result<[u32; 2], Error> {
        // SAFETY: the surface and physical device are live.
        let caps = unsafe {
            self.fns()
                .surface
                .get_physical_device_surface_capabilities(self.shared().physical, self.surface)
        }
        .map_err(Error::vk("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))?;
        Ok(extent_of(&caps, self.desc.extent))
    }

    fn query_formats(&mut self) -> Result<(), Error> {
        // SAFETY: the surface and physical device are live.
        let formats = unsafe {
            self.fns()
                .surface
                .get_physical_device_surface_formats(self.shared().physical, self.surface)
        }
        .map_err(Error::vk("vkGetPhysicalDeviceSurfaceFormatsKHR"))?;
        self.formats = negotiate(&formats);
        self.supported = self.formats.iter().map(|f| f.0).collect();
        Ok(())
    }

    /// Replaces the current swapchain with one at the surface's extent, the asked-for format
    /// and present mode. Does nothing while the window has no area.
    fn recreate(&mut self) -> Result<(), Error> {
        self.query_formats()?;
        let shared = Arc::clone(&self.device.shared);
        let lock = Arc::clone(&self.lock);
        let fns = lock.fns();
        let mode = vk_mode(self.mode);
        let mut compatible = [vk::PresentModeKHR::default(); 8];
        let mut compat =
            vk::SurfacePresentModeCompatibilityKHR::default().present_modes(&mut compatible);
        let mut id2 = vk::SurfaceCapabilitiesPresentId2KHR::default();
        let mut wait2 = vk::SurfaceCapabilitiesPresentWait2KHR::default();
        let mut mode_info = vk::SurfacePresentModeKHR::default().present_mode(mode);
        let surface_info = vk::PhysicalDeviceSurfaceInfo2KHR::default()
            .surface(self.surface)
            .push(&mut mode_info);
        let mut caps2 = vk::SurfaceCapabilities2KHR::default()
            .push(&mut compat)
            .push(&mut id2)
            .push(&mut wait2);
        // SAFETY: the chain's structs belong to enabled instance extensions.
        unsafe {
            fns.caps2.get_physical_device_surface_capabilities2(
                shared.physical,
                &surface_info,
                &mut caps2,
            )
        }
        .map_err(Error::vk("vkGetPhysicalDeviceSurfaceCapabilities2KHR"))?;
        let caps = caps2.surface_capabilities;
        let count = compat.present_mode_count as usize;
        let compatible = &compatible[..count];
        let extent = extent_of(&caps, self.desc.extent);
        if extent.contains(&0) {
            return Ok(());
        }
        let Some(&(format, surface_format)) = self.formats.iter().find(|f| f.0 == self.format)
        else {
            return Err(Error::Invalid(format!(
                "the surface no longer offers {:?}",
                self.format
            )));
        };
        let mut modes = vec![mode];
        for m in [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::IMMEDIATE] {
            if m != mode && compatible.contains(&m) {
                modes.push(m);
            }
        }
        let mut images = self.desc.min_images.max(caps.min_image_count);
        if caps.max_image_count != 0 {
            images = images.min(caps.max_image_count);
        }
        let present_ids = shared.caps.present_wait
            && id2.present_id2_supported == vk::TRUE
            && wait2.present_wait2_supported == vk::TRUE;
        let mut flags = vk::SwapchainCreateFlagsKHR::empty();
        if present_ids {
            flags |= vk::SwapchainCreateFlagsKHR::PRESENT_ID_2
                | vk::SwapchainCreateFlagsKHR::PRESENT_WAIT_2;
        }
        if shared.caps.present_timing {
            flags |= vk::SwapchainCreateFlagsKHR::PRESENT_TIMING_EXT;
        }
        let mut usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        usage |= caps.supported_usage_flags & vk::ImageUsageFlags::TRANSFER_SRC;
        let old = self
            .current
            .as_ref()
            .map_or(vk::SwapchainKHR::null(), |g| g.raw);
        let mut modes_info =
            vk::SwapchainPresentModesCreateInfoKHR::default().present_modes(&modes);
        let mut latency = vk::SwapchainLatencyCreateInfoNV::default().latency_mode_enable(true);
        let mut info = vk::SwapchainCreateInfoKHR::default()
            .flags(flags)
            .surface(self.surface)
            .min_image_count(images)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(vk::Extent2D {
                width: extent[0],
                height: extent[1],
            })
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(mode)
            .clipped(true)
            .old_swapchain(old)
            .push(&mut modes_info);
        let low_latency = self.lock.sleep != vk::Semaphore::null() && present_ids;
        if low_latency {
            info = info.push(&mut latency);
        }
        let raw = {
            let mut locked = self.lock.lock();
            // SAFETY: valid create info; the old swapchain is ours and the lock is held.
            let raw = unsafe { fns.swapchain.create_swapchain(&info, None) }
                .map_err(Error::vk("vkCreateSwapchainKHR"))?;
            locked.raw = raw;
            locked.latency = low_latency;
            if let (true, Some(m)) = (low_latency, locked.mode) {
                lock.apply_mode(raw, m);
            }
            raw
        };
        if let Some(g) = self.current.take() {
            self.retired.push(g);
        }
        self.generations += 1;
        let id = self.generations;
        // SAFETY: the swapchain is live.
        let raw_images = unsafe { fns.swapchain.get_swapchain_images(raw) }
            .map_err(Error::vk("vkGetSwapchainImagesKHR"));
        let mut g = Generation {
            id,
            raw,
            images: Vec::new(),
            format,
            extent,
            modes,
            present_ids,
        };
        let filled = raw_images.and_then(|raw_images| {
            let desc = TextureDesc::d2(
                texture_format(surface_format.format),
                extent[0],
                extent[1],
                TextureUsage::STORAGE | TextureUsage::COLOR,
            );
            for (i, &image) in raw_images.iter().enumerate() {
                let name = format!("swapchain {id} image {i}");
                name_object(&shared.debug_fn, image, &name);
                let texture = Texture::swapchain_image(&shared, image, &desc, &name)?;
                let slot = self.device.alloc_image_slot()?;
                let storage = self
                    .device
                    .write_storage(slot, &texture, &ViewDesc::default());
                g.images.push((Arc::new(texture), storage));
            }
            Ok(())
        });
        self.current = Some(g);
        filled?;
        self.stale = false;
        self.hdr_pending = true;
        log::debug!(
            "swapchain {id}: {extent:?} {format:?} {images} images, modes {:?}",
            self.current.as_ref().unwrap().modes
        );
        Ok(())
    }

    /// Hands the HDR metadata to the current swapchain.
    fn apply_hdr(&mut self) {
        self.hdr_pending = false;
        let (Some(hdr), Some(f), Some(g)) = (self.hdr, &self.fns().hdr, &self.current) else {
            return;
        };
        if g.format == OutputFormat::Sdr {
            return;
        }
        let xy = |p: [f32; 2]| vk::XYColorEXT { x: p[0], y: p[1] };
        let metadata = [vk::HdrMetadataEXT::default()
            .display_primary_red(xy(hdr.primaries[0]))
            .display_primary_green(xy(hdr.primaries[1]))
            .display_primary_blue(xy(hdr.primaries[2]))
            .white_point(xy(hdr.white))
            .max_luminance(hdr.max_luminance)
            .min_luminance(hdr.min_luminance)
            .max_content_light_level(hdr.max_cll)
            .max_frame_average_light_level(hdr.max_fall)];
        let _locked = self.lock.lock();
        // SAFETY: the swapchain is live and the lock is held.
        unsafe { f.set_hdr_metadata(&[g.raw], &metadata) };
    }

    /// A slot whose semaphores are unused: a free one, one whose present has completed, or a
    /// new one.
    fn free_slot(&mut self) -> Result<usize, Error> {
        let d = &self.device.shared.device;
        let n = self.slots.len();
        for k in 0..n {
            let i = (self.next_slot + k) % n;
            let s = &mut self.slots[i];
            let free = match s.state {
                SlotState::Free => true,
                SlotState::Presented(_) => signaled(d, s.fence),
                SlotState::Acquired(_) => false,
            };
            if free {
                if s.state != SlotState::Free {
                    // SAFETY: the fence has signaled and nothing waits on it.
                    unsafe { d.reset_fences(&[s.fence]) }.map_err(Error::vk("vkResetFences"))?;
                    s.state = SlotState::Free;
                }
                self.next_slot = (i + 1) % n;
                return Ok(i);
            }
        }
        let semaphore = || {
            // SAFETY: valid create info.
            unsafe { d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(Error::vk("vkCreateSemaphore"))
        };
        let acquire = semaphore()?;
        let present = semaphore()?;
        // SAFETY: valid create info.
        let fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(Error::vk("vkCreateFence"))?;
        let debug = &self.device.shared.debug_fn;
        name_object(debug, acquire, &format!("swapchain acquire {n}"));
        name_object(debug, present, &format!("swapchain present {n}"));
        self.slots.push(Slot {
            acquire,
            present,
            fence,
            state: SlotState::Free,
        });
        Ok(n)
    }

    /// True once no slot holds work of generation `id`.
    fn generation_done(&self, id: u64) -> bool {
        let d = &self.device.shared.device;
        self.slots.iter().all(|s| match s.state {
            SlotState::Acquired(g) => g != id,
            SlotState::Presented(g) => g != id || signaled(d, s.fence),
            SlotState::Free => true,
        })
    }

    /// Destroys retired swapchains whose presents have completed.
    fn destroy_finished(&mut self) {
        let mut i = 0;
        while i < self.retired.len() {
            if self.generation_done(self.retired[i].id) {
                let g = self.retired.swap_remove(i);
                self.destroy_generation(g);
            } else {
                i += 1;
            }
        }
    }

    fn destroy_generation(&self, g: Generation) {
        let heaps = self.device.shared.heaps();
        for (texture, storage) in g.images {
            debug_assert_eq!(
                Arc::strong_count(&texture),
                1,
                "a frame outlived its present"
            );
            drop(texture);
            heaps.release_image_slot(storage.0);
        }
        // SAFETY: no work uses the swapchain or its images any more, and no other call
        // names it: the lock holds only the current swapchain.
        unsafe { self.fns().swapchain.destroy_swapchain(g.raw, None) };
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let d = &self.device.shared.device;
        let pending: Vec<vk::Fence> = self
            .slots
            .iter()
            .filter(|s| matches!(s.state, SlotState::Presented(_)))
            .map(|s| s.fence)
            .collect();
        if !pending.is_empty() {
            // SAFETY: the fences are live.
            let waited =
                unsafe { d.wait_for_fences(&pending, true, DRAIN_TIMEOUT.as_nanos() as u64) };
            if let Err(e) = waited {
                log::warn!("swapchain drop: presents did not complete: {e}");
            }
        }
        {
            let mut locked = self.lock.lock();
            locked.raw = vk::SwapchainKHR::null();
            locked.latency = false;
        }
        let generations: Vec<Generation> = self
            .current
            .take()
            .into_iter()
            .chain(self.retired.drain(..))
            .collect();
        for g in generations {
            self.destroy_generation(g);
        }
        // SAFETY: every present has completed, so nothing waits on these any more.
        unsafe {
            for s in &self.slots {
                d.destroy_semaphore(s.acquire, None);
                d.destroy_semaphore(s.present, None);
                d.destroy_fence(s.fence, None);
            }
            self.fns().surface.destroy_surface(self.surface, None);
        }
    }
}

fn signaled(d: &ash::Device, fence: vk::Fence) -> bool {
    // SAFETY: the caller's fence is live.
    let status = unsafe { d.get_fence_status(fence) };
    status == Ok(true)
}

/// The surface's current extent, or `fallback` within its limits when the surface leaves
/// the extent to the swapchain.
fn extent_of(caps: &vk::SurfaceCapabilitiesKHR, fallback: [u32; 2]) -> [u32; 2] {
    let c = caps.current_extent;
    if c.width != u32::MAX {
        return [c.width, c.height];
    }
    [
        fallback[0].clamp(caps.min_image_extent.width, caps.max_image_extent.width),
        fallback[1].clamp(caps.min_image_extent.height, caps.max_image_extent.height),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(format: vk::Format, color_space: vk::ColorSpaceKHR) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format,
            color_space,
        }
    }

    /// The surface formats of a full-screen window on an RTX 5090 with 621.01 (measured.md
    /// DSP-4).
    fn recorded() -> Vec<vk::SurfaceFormatKHR> {
        use vk::{ColorSpaceKHR as C, Format as F};
        vec![
            f(F::B8G8R8A8_UNORM, C::SRGB_NONLINEAR),
            f(F::B8G8R8A8_SRGB, C::SRGB_NONLINEAR),
            f(F::R8G8B8A8_UNORM, C::SRGB_NONLINEAR),
            f(F::R8G8B8A8_SRGB, C::SRGB_NONLINEAR),
            f(F::R16G16B16A16_SFLOAT, C::EXTENDED_SRGB_LINEAR_EXT),
            f(F::A2B10G10R10_UNORM_PACK32, C::HDR10_ST2084_EXT),
            f(F::A2B10G10R10_UNORM_PACK32, C::SRGB_NONLINEAR),
        ]
    }

    #[test]
    fn negotiation_on_recorded_formats() {
        use vk::{ColorSpaceKHR as C, Format as F};
        let got = negotiate(&recorded());
        assert_eq!(
            got,
            vec![
                (OutputFormat::Sdr, f(F::B8G8R8A8_UNORM, C::SRGB_NONLINEAR)),
                (
                    OutputFormat::Hdr10Pq,
                    f(F::A2B10G10R10_UNORM_PACK32, C::HDR10_ST2084_EXT)
                ),
                (
                    OutputFormat::ScRgb,
                    f(F::R16G16B16A16_SFLOAT, C::EXTENDED_SRGB_LINEAR_EXT)
                ),
            ]
        );
        // Without BGRA, SDR takes RGBA; a 10-bit sRGB surface is not PQ; sRGB-encoded
        // formats are never picked, since compose encodes.
        let sdr_only = vec![
            f(F::B8G8R8A8_SRGB, C::SRGB_NONLINEAR),
            f(F::R8G8B8A8_UNORM, C::SRGB_NONLINEAR),
            f(F::A2B10G10R10_UNORM_PACK32, C::SRGB_NONLINEAR),
        ];
        assert_eq!(
            negotiate(&sdr_only),
            vec![(OutputFormat::Sdr, f(F::R8G8B8A8_UNORM, C::SRGB_NONLINEAR))]
        );
        assert!(negotiate(&[f(F::B8G8R8A8_SRGB, C::SRGB_NONLINEAR)]).is_empty());
    }
}
