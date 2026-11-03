#![allow(non_snake_case)]

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};

use raw_window_handle::RawWindowHandle;
use windows::Foundation::{Point, TypedEventHandler};
use windows::Graphics::Display::{AdvancedColorKind, DisplayInformation};
use windows::System::{DispatcherQueue, DispatcherQueueController};
use windows::Win32::Devices::Display::{
    DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, DISPLAYCONFIG_DEVICE_INFO_HEADER,
    DISPLAYCONFIG_MODE_INFO, DISPLAYCONFIG_PATH_INFO, DISPLAYCONFIG_SOURCE_DEVICE_NAME,
    DisplayConfigGetDeviceInfo, GetDisplayConfigBufferSizes, QDC_ONLY_ACTIVE_PATHS,
    QueryDisplayConfig,
};
use windows::Win32::Foundation::{
    ERROR_INSUFFICIENT_BUFFER, HWND, LPARAM, LRESULT, RPC_E_CHANGED_MODE, WPARAM,
};
use windows::Win32::Graphics::Gdi::{
    GetMonitorInfoW, HMONITOR, MONITOR_DEFAULTTONEAREST, MONITORINFO, MONITORINFOEXW,
    MonitorFromWindow,
};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::System::WinRT::{
    CreateDispatcherQueueController, DQTAT_COM_NONE, DQTYPE_THREAD_CURRENT, DispatcherQueueOptions,
    RO_INIT_SINGLETHREADED, RoInitialize, RoUninitialize,
};
use windows::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
use windows::Win32::UI::WindowsAndMessaging::{
    GetWindowThreadProcessId, KillTimer, RegisterWindowMessageW, SendNotifyMessageW, SetTimer,
    WM_DISPLAYCHANGE, WM_NCDESTROY, WM_SETFOCUS, WM_TIMER, WM_WINDOWPOSCHANGED,
};
use windows_core::{GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface, Ref, w};

use crate::{DisplayFeed, DisplayInfo, Error, Result};

// Windows 11 22621+. windows-rs has no binding; the three IInspectable slots come first.
#[windows_core::interface("7449121C-382B-4705-8DA7-A795BA482013")]
unsafe trait IDisplayInformationStaticsInterop: IUnknown {
    fn GetIids(&self, count: *mut u32, iids: *mut *mut GUID) -> HRESULT;
    fn GetRuntimeClassName(&self, name: *mut *mut c_void) -> HRESULT;
    fn GetTrustLevel(&self, level: *mut i32) -> HRESULT;
    fn GetForWindow(&self, window: HWND, riid: *const GUID, info: *mut *mut c_void) -> HRESULT;
}

/// Turning HDR on or off raises no event, so the watch also reads this often.
const READ_INTERVAL_MS: u32 = 1000;

/// How far the SDR white may move, as a share of its value, and still count as unchanged: each
/// change moves the tone curve, and the reading drifts.
const SDR_WHITE_TOLERANCE: f32 = 0.01;

/// Watches the monitor a window is on and publishes what the OS reports for it to a
/// `DisplayFeed`: once a second, when the window takes the focus, on moves to another monitor,
/// on display mode changes and on the OS's color change event. With the `pcl-stats` feature it
/// also receives the PC latency ping for the frames of every `Display`.
///
/// Create it on the thread that owns the window and pumps its messages; changes arrive
/// there. It uses the thread's dispatcher queue, creating one if the thread has none. It reads
/// through the window's message queue and adds no thread. It may be dropped on any thread, and
/// before or after the window is destroyed; dropped on another thread, it stops when the
/// window's thread next pumps its messages.
pub struct DisplayWatch {
    hwnd: isize,
    /// The address of the `Subclass`, which the window's thread owns.
    id: usize,
}

/// The message that frees a `Subclass`, named by its id in `wparam`; 0 if none could be made.
static STOP: LazyLock<u32> =
    // SAFETY: a plain call with a constant name.
    LazyLock::new(|| unsafe { RegisterWindowMessageW(w!("orr display watch stop")) });

/// The watch's state on the window's thread, freed there by the `stop` message or when the
/// window is destroyed. Its address is the id of the subclass and of the timer, so two watches
/// of a window stay apart.
struct Subclass {
    token: i64,
    reader: Reader,
    monitor: Cell<HMONITOR>,
    info: DisplayInformation,
}

/// Publishes what the OS reports for a window. A failed read keeps the last values, since the
/// OS can refuse for a moment while a window moves between monitors.
#[derive(Clone)]
struct Reader {
    hwnd: isize,
    feed: DisplayFeed,
    /// The last read failed, which is logged once.
    failing: Arc<AtomicBool>,
}

impl Reader {
    fn update(&self, info: &DisplayInformation) {
        match read(info, HWND(self.hwnd as *mut c_void)) {
            Ok(read) => {
                self.failing.store(false, Ordering::Relaxed);
                self.feed.set(steady(self.feed.latest().1, read));
            }
            Err(e) => {
                if !self.failing.swap(true, Ordering::Relaxed) {
                    log::warn!("reading AdvancedColorInfo failed: {e}");
                }
            }
        }
    }
}

/// `new` keeping the last SDR white while it stays within `SDR_WHITE_TOLERANCE` of `new`'s.
fn steady(last: DisplayInfo, mut new: DisplayInfo) -> DisplayInfo {
    if let (Some(old), Some(white)) = (last.sdr_white_nits, new.sdr_white_nits)
        && (white - old).abs() <= SDR_WHITE_TOLERANCE * old
    {
        new.sdr_white_nits = last.sdr_white_nits;
    }
    new
}

struct QueueController(DispatcherQueueController);

struct Apartment;

/// What a thread needs for `GetForWindow`, kept until the thread ends: a queue shut down with
/// the watch that made it would take the thread's other watches down with it.
struct ThreadSetup {
    // Fields drop in order: the queue, then the apartment.
    _queue: Option<QueueController>,
    _apartment: Option<Apartment>,
}

thread_local! {
    static THREAD_SETUP: RefCell<Option<ThreadSetup>> = const { RefCell::new(None) };
}

impl ThreadSetup {
    fn new() -> windows_core::Result<Self> {
        // SAFETY: `Apartment` balances it on this thread. An apartment of another kind already
        // on the thread serves WinRT as well.
        let apartment = match unsafe { RoInitialize(RO_INIT_SINGLETHREADED) } {
            Ok(()) => Some(Apartment),
            Err(e) if e.code() == RPC_E_CHANGED_MODE => None,
            Err(e) => return Err(e),
        };
        let queue = match DispatcherQueue::GetForCurrentThread() {
            Ok(_) => None,
            Err(_) => {
                let options = DispatcherQueueOptions {
                    dwSize: size_of::<DispatcherQueueOptions>() as u32,
                    threadType: DQTYPE_THREAD_CURRENT,
                    apartmentType: DQTAT_COM_NONE,
                };
                // SAFETY: plain call with a filled options struct.
                Some(QueueController(unsafe {
                    CreateDispatcherQueueController(options)
                }?))
            }
        };
        Ok(Self {
            _queue: queue,
            _apartment: apartment,
        })
    }
}

impl DisplayWatch {
    pub fn new(window: RawWindowHandle) -> Result<(Self, DisplayFeed)> {
        let RawWindowHandle::Win32(handle) = window else {
            return Err(Error::NotWin32Window);
        };
        let hwnd = HWND(handle.hwnd.get() as *mut c_void);
        // SAFETY: plain queries; a stale window reads as another thread's.
        if unsafe { GetWindowThreadProcessId(hwnd, None) != GetCurrentThreadId() } {
            return Err(Error::NotWindowThread);
        }

        THREAD_SETUP.with_borrow_mut(|setup| {
            if setup.is_none() {
                *setup = Some(ThreadSetup::new()?);
            }
            windows_core::Result::Ok(())
        })?;
        let info = display_information(hwnd)?;
        let feed = DisplayFeed::hdr_off();
        let reader = Reader {
            hwnd: hwnd.0 as isize,
            feed: feed.clone(),
            failing: Arc::default(),
        };
        reader.update(&info);

        let handler = reader.clone();
        let token = info.AdvancedColorInfoChanged(&TypedEventHandler::new(
            move |sender: Ref<DisplayInformation>, _| {
                if let Some(info) = sender.as_ref() {
                    handler.update(info);
                }
                Ok(())
            },
        ))?;

        let subclass = Box::into_raw(Box::new(Subclass {
            token,
            reader,
            // SAFETY: `hwnd` is a live window.
            monitor: Cell::new(unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) }),
            info,
        }));
        let id = subclass as usize;
        // SAFETY: on the window's thread; `Subclass`'s drop removes the subclass before the box
        // goes away.
        let installed = unsafe { SetWindowSubclass(hwnd, Some(subclass_proc), id, id) };
        if *STOP == 0 || !installed.as_bool() {
            let e = windows_core::Error::from_thread();
            // SAFETY: the box made above, which nothing else holds now.
            drop(unsafe { Box::from_raw(subclass) });
            return Err(e.into());
        }
        // SAFETY: `hwnd` is a live window of this thread; `Subclass`'s drop kills the timer.
        if unsafe { SetTimer(Some(hwnd), id, READ_INTERVAL_MS, None) } == 0 {
            log::warn!(
                "no display read timer, so an OS HDR change goes unseen: {}",
                windows_core::Error::from_thread()
            );
        }
        Ok((
            Self {
                hwnd: hwnd.0 as isize,
                id,
            },
            feed,
        ))
    }
}

impl Drop for DisplayWatch {
    fn drop(&mut self) {
        // This runs at once on the window's thread and does not wait on another: the window's
        // thread may be waiting for this one. A destroyed window has freed the subclass already.
        // SAFETY: a plain message call; `subclass_proc` checks the id before acting on it.
        let _ = unsafe {
            SendNotifyMessageW(
                HWND(self.hwnd as *mut c_void),
                *STOP,
                WPARAM(self.id),
                LPARAM(0),
            )
        };
    }
}

impl Drop for Subclass {
    fn drop(&mut self) {
        let id = std::ptr::from_ref(self) as usize;
        let hwnd = HWND(self.reader.hwnd as *mut c_void);
        // SAFETY: on the window's thread, removing the timer and subclass `DisplayWatch::new`
        // installed.
        unsafe {
            let _ = KillTimer(Some(hwnd), id);
            let _ = RemoveWindowSubclass(hwnd, Some(subclass_proc), id);
        }
        let _ = self.info.RemoveAdvancedColorInfoChanged(self.token);
    }
}

impl Drop for QueueController {
    fn drop(&mut self) {
        let _ = self.0.ShutdownQueueAsync();
    }
}

impl Drop for Apartment {
    fn drop(&mut self) {
        // SAFETY: balances `RoInitialize` in `ThreadSetup::new` on the same thread, after the
        // queue is released.
        unsafe { RoUninitialize() };
    }
}

unsafe extern "system" fn subclass_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    id: usize,
    data: usize,
) -> LRESULT {
    // SAFETY: `data` is the `Subclass` box, alive until the subclass is removed.
    let subclass = unsafe { &*(data as *const Subclass) };
    if msg == *STOP && wparam.0 == id {
        // SAFETY: the box is not used after this, and its drop removes the subclass.
        drop(unsafe { Box::from_raw(data as *mut Subclass) });
        return LRESULT(0);
    }
    if msg == WM_NCDESTROY {
        // SAFETY: as above, and the rest of the window's handlers still run.
        unsafe {
            drop(Box::from_raw(data as *mut Subclass));
            return DefSubclassProc(hwnd, msg, wparam, lparam);
        }
    }
    match msg {
        WM_DISPLAYCHANGE | WM_SETFOCUS => subclass.reader.update(&subclass.info),
        WM_TIMER if wparam.0 == id => {
            subclass.reader.update(&subclass.info);
            return LRESULT(0);
        }
        #[cfg(feature = "pcl-stats")]
        m if crate::pcl_stats::is_ping(m) => crate::pcl_stats::ping(),
        WM_WINDOWPOSCHANGED => {
            // SAFETY: `hwnd` is the live subclassed window.
            let monitor = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
            if subclass.monitor.replace(monitor) != monitor {
                subclass.reader.update(&subclass.info);
            }
        }
        _ => {}
    }
    // SAFETY: forwarding the message unchanged.
    unsafe { DefSubclassProc(hwnd, msg, wparam, lparam) }
}

fn display_information(hwnd: HWND) -> windows_core::Result<DisplayInformation> {
    let interop = windows_core::factory::<DisplayInformation, IDisplayInformationStaticsInterop>()?;
    let mut raw = std::ptr::null_mut();
    // SAFETY: `hwnd` is a live window; on success `raw` holds an owned DisplayInformation.
    unsafe {
        interop
            .GetForWindow(hwnd, &DisplayInformation::IID, &mut raw)
            .ok()?;
        Ok(DisplayInformation::from_raw(raw))
    }
}

/// Reads the window's monitor.
fn read(info: &DisplayInformation, hwnd: HWND) -> windows_core::Result<DisplayInfo> {
    Ok(AdvancedColor::new(info)?.display_info(refresh_hz(hwnd)))
}

/// The fields of an `AdvancedColorInfo`.
#[derive(Clone, Copy, Debug)]
struct AdvancedColor {
    kind: AdvancedColorKind,
    min_nits: f32,
    max_nits: f32,
    max_full_frame_nits: f32,
    sdr_white_nits: f32,
    /// Red, green, blue, white.
    primaries_xy: [[f32; 2]; 4],
}

impl AdvancedColor {
    fn new(info: &DisplayInformation) -> windows_core::Result<Self> {
        let a = info.GetAdvancedColorInfo()?;
        let xy = |p: Point| [p.X, p.Y];
        Ok(Self {
            kind: a.CurrentAdvancedColorKind()?,
            min_nits: a.MinLuminanceInNits()?,
            max_nits: a.MaxLuminanceInNits()?,
            max_full_frame_nits: a.MaxAverageFullFrameLuminanceInNits()?,
            sdr_white_nits: a.SdrWhiteLevelInNits()?,
            primaries_xy: [
                xy(a.RedPrimary()?),
                xy(a.GreenPrimary()?),
                xy(a.BluePrimary()?),
                xy(a.WhitePoint()?),
            ],
        })
    }

    /// Wide colour gamut is SDR with automatic colour management, not HDR. Zero or invalid
    /// luminance and all-zero primaries mean unknown.
    fn display_info(&self, refresh_hz: Option<f32>) -> DisplayInfo {
        let positive = |v: f32| (v.is_finite() && v > 0.0).then_some(v);
        let primaries_known = self.primaries_xy.iter().flatten().any(|&c| c > 0.0)
            && self.primaries_xy.iter().flatten().all(|c| c.is_finite());
        DisplayInfo {
            os_hdr_on: Some(self.kind == AdvancedColorKind::HighDynamicRange),
            max_nits: positive(self.max_nits),
            max_full_frame_nits: positive(self.max_full_frame_nits),
            min_nits: (self.min_nits.is_finite() && self.min_nits >= 0.0).then_some(self.min_nits),
            sdr_white_nits: positive(self.sdr_white_nits),
            refresh_hz: refresh_hz.and_then(positive),
            primaries_xy: primaries_known.then_some(self.primaries_xy),
        }
    }
}

/// The refresh rate of the display path whose source is the window's monitor.
fn refresh_hz(hwnd: HWND) -> Option<f32> {
    let mut monitor_info = MONITORINFOEXW::default();
    monitor_info.monitorInfo.cbSize = size_of::<MONITORINFOEXW>() as u32;
    // SAFETY: `hwnd` is a live window; `monitor_info` is a MONITORINFOEXW with its size set.
    let found = unsafe {
        let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        GetMonitorInfoW(monitor, (&raw mut monitor_info).cast::<MONITORINFO>())
    };
    if !found.as_bool() {
        return None;
    }
    let device = until_nul(&monitor_info.szDevice);

    // The path count can change between sizing and querying; retry then.
    for _ in 0..3 {
        let (mut path_count, mut mode_count) = (0, 0);
        // SAFETY: plain out-parameter calls with buffers of the reported sizes.
        let paths = unsafe {
            GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut path_count, &mut mode_count)
                .ok()
                .ok()?;
            let mut paths = vec![DISPLAYCONFIG_PATH_INFO::default(); path_count as usize];
            let mut modes = vec![DISPLAYCONFIG_MODE_INFO::default(); mode_count as usize];
            let status = QueryDisplayConfig(
                QDC_ONLY_ACTIVE_PATHS,
                &mut path_count,
                paths.as_mut_ptr(),
                &mut mode_count,
                modes.as_mut_ptr(),
                None,
            );
            if status == ERROR_INSUFFICIENT_BUFFER {
                continue;
            }
            status.ok().ok()?;
            paths.truncate(path_count as usize);
            paths
        };
        return paths.iter().find_map(|path| {
            let mut source = DISPLAYCONFIG_SOURCE_DEVICE_NAME {
                header: DISPLAYCONFIG_DEVICE_INFO_HEADER {
                    r#type: DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME,
                    size: size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32,
                    adapterId: path.sourceInfo.adapterId,
                    id: path.sourceInfo.id,
                },
                ..Default::default()
            };
            // SAFETY: the header names a SOURCE_DEVICE_NAME request of the right size.
            if unsafe { DisplayConfigGetDeviceInfo(&mut source.header) } != 0
                || until_nul(&source.viewGdiDeviceName) != device
            {
                return None;
            }
            let rate = path.targetInfo.refreshRate;
            (rate.Denominator != 0).then(|| rate.Numerator as f32 / rate.Denominator as f32)
        });
    }
    None
}

fn until_nul(s: &[u16]) -> &[u16] {
    &s[..s.iter().position(|&c| c == 0).unwrap_or(s.len())]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdr_white_stays_through_small_changes_only() {
        let with = |nits: f32| DisplayInfo {
            os_hdr_on: Some(true),
            sdr_white_nits: Some(nits),
            ..DisplayInfo::default()
        };
        assert_eq!(steady(with(200.0), with(201.0)), with(200.0));
        assert_eq!(steady(with(200.0), with(203.0)), with(203.0));
        assert_eq!(
            steady(DisplayInfo::default(), with(200.0)),
            with(200.0),
            "a first reading stands"
        );
    }

    #[test]
    fn advanced_color_mapping_on_captured_values() {
        // A profiled SDR display under automatic colour management, and an unprofiled one
        // reporting EDID.
        let wcg = AdvancedColor {
            kind: AdvancedColorKind::WideColorGamut,
            min_nits: 0.0,
            max_nits: 120.389,
            max_full_frame_nits: 120.389,
            sdr_white_nits: 80.0,
            primaries_xy: [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.3127, 0.3290]],
        };
        let info = wcg.display_info(Some(144.0));
        assert_eq!(info.os_hdr_on, Some(false));
        assert_eq!((info.min_nits, info.max_nits), (Some(0.0), Some(120.389)));
        assert_eq!(info.primaries_xy, Some(wcg.primaries_xy));
        assert_eq!(info.refresh_hz, Some(144.0));

        let edid = AdvancedColor {
            kind: AdvancedColorKind::StandardDynamicRange,
            min_nits: 0.5,
            max_nits: 270.0,
            max_full_frame_nits: 270.0,
            sdr_white_nits: 80.0,
            primaries_xy: [[0.0; 2]; 4],
        };
        let info = edid.display_info(None);
        assert_eq!(info.os_hdr_on, Some(false));
        assert_eq!((info.min_nits, info.max_nits), (Some(0.5), Some(270.0)));
        assert_eq!(info.primaries_xy, None);

        let hdr = AdvancedColor {
            kind: AdvancedColorKind::HighDynamicRange,
            min_nits: 0.005,
            max_nits: f32::NAN,
            max_full_frame_nits: 0.0,
            sdr_white_nits: 203.0,
            ..wcg
        };
        let info = hdr.display_info(Some(0.0));
        assert_eq!(info.os_hdr_on, Some(true));
        assert_eq!(info.max_nits, None);
        assert_eq!(info.max_full_frame_nits, None);
        assert_eq!(info.refresh_hz, None);
    }
}
