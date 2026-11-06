use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use raw_window_handle::RawWindowHandle;
use windows_core::{EventRevoker, Interface, Ref, w};

use crate::{DisplayFeed, DisplayInfo, Error, OsError, Result, steady};

mod bindings;
use bindings::*;

/// Turning HDR on or off raises no event, so the watch also reads this often.
const READ_INTERVAL_MS: u32 = 1000;

const NEAREST: u32 = MONITOR_DEFAULTTONEAREST as u32;

/// Reads once a second, when the window takes the focus, on moves to another monitor, on
/// display mode changes and on the OS's color change event.
///
/// It uses the thread's dispatcher queue, creating one if the thread has none. It reads
/// through the window's message queue and adds no thread. Dropped on another thread, it stops
/// when the window's thread next pumps its messages.
pub(crate) struct Watch {
    hwnd: isize,
    /// The address of the `Subclass`, which the window's thread owns.
    id: usize,
    /// The message that frees the `Subclass`, named by its id in `wparam`.
    stop: u32,
}

static MESSAGE_HOOK: OnceLock<fn(u32)> = OnceLock::new();

/// Calls `hook` with every message of every watched window, once per watch of the window,
/// before the window handles it, on that window's thread. The first hook set stays.
pub fn set_message_hook(hook: fn(u32)) {
    let _ = MESSAGE_HOOK.set(hook);
}

/// The watch's state on the window's thread, freed there by the `stop` message or when the
/// window is destroyed. Its address is the id of the subclass and of the timer, so two watches
/// of a window stay apart.
struct Subclass {
    _event: EventRevoker,
    reader: Reader,
    monitor: Cell<HMONITOR>,
    info: DisplayInformation,
    stop: u32,
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

windows_core::link!("ntdll.dll" "system" fn RtlDllShutdownInProgress() -> u8);

/// What a thread needs for `GetForWindow`, kept until the thread ends: a queue shut down with
/// the watch that made it would take the thread's other watches down with it.
struct ThreadSetup {
    queue: Option<DispatcherQueueController>,
    /// `RoInitialize` succeeded here, so `drop` balances it.
    apartment: bool,
}

thread_local! {
    static THREAD_SETUP: RefCell<Option<ThreadSetup>> = const { RefCell::new(None) };
}

impl ThreadSetup {
    fn new() -> windows_core::Result<Self> {
        // SAFETY: `drop` balances it on this thread. An apartment of another kind already on the
        // thread serves WinRT as well.
        let apartment = match unsafe { RoInitialize(RO_INIT_SINGLETHREADED) }.ok() {
            Ok(()) => true,
            Err(e) if e.code() == RPC_E_CHANGED_MODE => false,
            Err(e) => return Err(e),
        };
        let mut setup = Self {
            queue: None,
            apartment,
        };
        if DispatcherQueue::GetForCurrentThread().is_err() {
            let options = DispatcherQueueOptions {
                dwSize: size_of::<DispatcherQueueOptions>() as u32,
                threadType: DQTYPE_THREAD_CURRENT,
                apartmentType: DQTAT_COM_NONE,
            };
            // SAFETY: plain call with a filled options struct.
            setup.queue = Some(unsafe { CreateDispatcherQueueController(options) }?.cast()?);
        }
        Ok(setup)
    }
}

impl Watch {
    pub(crate) fn new(window: RawWindowHandle) -> Result<(Self, DisplayFeed)> {
        let RawWindowHandle::Win32(handle) = window else {
            return Err(Error::UnsupportedWindow);
        };
        let hwnd = HWND(handle.hwnd.get() as *mut c_void);
        // SAFETY: plain queries; a stale window reads as another thread's.
        if unsafe { GetWindowThreadProcessId(hwnd, None) != GetCurrentThreadId() } {
            return Err(Error::NotWindowThread);
        }
        // SAFETY: a plain call with a constant name.
        let stop = unsafe { RegisterWindowMessageW(w!("orr display watch stop")) };
        if stop == 0 {
            return Err(windows_core::Error::from_thread().into());
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
        let event = info.AdvancedColorInfoChanged(move |sender: Ref<DisplayInformation>, _| {
            if let Some(info) = sender.as_ref() {
                handler.update(info);
            }
        })?;

        let subclass = Box::into_raw(Box::new(Subclass {
            _event: event,
            reader,
            // SAFETY: `hwnd` is a live window.
            monitor: Cell::new(unsafe { MonitorFromWindow(hwnd, NEAREST) }),
            info,
            stop,
        }));
        let id = subclass as usize;
        // SAFETY: on the window's thread; `Subclass`'s drop removes the subclass before the box
        // goes away.
        let installed = unsafe { SetWindowSubclass(hwnd, Some(subclass_proc), id, id) };
        if !installed.as_bool() {
            // SAFETY: the box made above, which nothing else holds now.
            drop(unsafe { Box::from_raw(subclass) });
            // It sets no last error.
            return Err(windows_core::Error::new(E_FAIL, "SetWindowSubclass failed").into());
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
                stop,
            },
            feed,
        ))
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        // This runs at once on the window's thread and does not wait on another: the window's
        // thread may be waiting for this one. A destroyed window has freed the subclass already.
        // SAFETY: a plain message call; `subclass_proc` checks the id before acting on it.
        let _ = unsafe {
            SendNotifyMessageW(
                HWND(self.hwnd as *mut c_void),
                self.stop,
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
    }
}

impl Drop for ThreadSetup {
    fn drop(&mut self) {
        // The thread that ends the process runs its destructors after combase and CoreMessaging
        // have shut down, and calling into them then faults. The OS frees all of this anyway.
        // SAFETY: a plain query.
        if unsafe { RtlDllShutdownInProgress() } != 0 {
            std::mem::forget(self.queue.take());
            return;
        }
        if let Some(queue) = self.queue.take() {
            let _ = queue.ShutdownQueueAsync();
        }
        if self.apartment {
            // SAFETY: balances `RoInitialize` in `new` on the same thread, after the queue.
            unsafe { RoUninitialize() };
        }
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
    if msg == subclass.stop && wparam.0 == id {
        // SAFETY: the box is not used after this, and its drop removes the subclass.
        drop(unsafe { Box::from_raw(data as *mut Subclass) });
        return LRESULT(0);
    }
    if msg == WM_NCDESTROY as u32 {
        // SAFETY: as above, and the rest of the window's handlers still run.
        unsafe {
            drop(Box::from_raw(data as *mut Subclass));
            return DefSubclassProc(hwnd, msg, wparam, lparam);
        }
    }
    if let Some(hook) = MESSAGE_HOOK.get() {
        hook(msg);
    }
    match msg as i32 {
        WM_DISPLAYCHANGE | WM_SETFOCUS => subclass.reader.update(&subclass.info),
        WM_TIMER if wparam.0 == id => {
            subclass.reader.update(&subclass.info);
            return LRESULT(0);
        }
        WM_WINDOWPOSCHANGED => {
            // SAFETY: `hwnd` is the live subclassed window.
            let monitor = unsafe { MonitorFromWindow(hwnd, NEAREST) };
            if subclass.monitor.replace(monitor) != monitor {
                subclass.reader.update(&subclass.info);
            }
        }
        _ => {}
    }
    // SAFETY: forwarding the message unchanged.
    unsafe { DefSubclassProc(hwnd, msg, wparam, lparam) }
}

/// `GetForWindow` needs Windows 11 22621 or later.
fn display_information(hwnd: HWND) -> windows_core::Result<DisplayInformation> {
    let interop = windows_core::factory::<DisplayInformation, IDisplayInformationStaticsInterop>()?;
    // SAFETY: `hwnd` is a live window.
    unsafe { interop.GetForWindow(hwnd) }
}

impl From<windows_core::Error> for Error {
    fn from(e: windows_core::Error) -> Self {
        Self::Windows(OsError(e))
    }
}

/// Whether Windows schedules the GPU of the adapter with this LUID in hardware, which frame
/// generation needs. `None` when it cannot tell.
pub fn hardware_scheduling(adapter_luid: [u8; 8]) -> Option<bool> {
    let l = adapter_luid;
    let mut open = D3DKMT_OPENADAPTERFROMLUID {
        AdapterLuid: LUID {
            LowPart: u32::from_le_bytes([l[0], l[1], l[2], l[3]]),
            HighPart: i32::from_le_bytes([l[4], l[5], l[6], l[7]]),
        },
        hAdapter: D3DKMT_HANDLE::default(),
    };
    // SAFETY: a plain struct the call fills.
    if unsafe { D3DKMTOpenAdapterFromLuid(&mut open) }.is_err() {
        return None;
    }
    let mut caps = D3DKMT_WDDM_2_7_CAPS::default();
    let mut query = D3DKMT_QUERYADAPTERINFO {
        hAdapter: open.hAdapter,
        Type: KMTQAITYPE_WDDM_2_7_CAPS,
        pPrivateDriverData: (&raw mut caps).cast(),
        PrivateDriverDataSize: size_of::<D3DKMT_WDDM_2_7_CAPS>() as u32,
    };
    // SAFETY: `query` points at a caps struct of the size it states.
    let read = unsafe { D3DKMTQueryAdapterInfo(&mut query) }.is_ok();
    // SAFETY: the adapter was opened above.
    let _ = unsafe {
        D3DKMTCloseAdapter(&D3DKMT_CLOSEADAPTER {
            hAdapter: open.hAdapter,
        })
    };
    // SAFETY: every bit pattern is a valid u32.
    let flags = unsafe { caps.Anonymous.Anonymous };
    read.then(|| flags.HwSchEnabled())
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

    /// Wide color gamut is SDR with automatic color management, not HDR. Zero or invalid
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
            headroom: None,
            refresh_hz: refresh_hz.and_then(positive),
            primaries_xy: primaries_known.then_some(self.primaries_xy),
        }
    }
}

/// The refresh rate of the display path whose source is the window's monitor.
fn refresh_hz(hwnd: HWND) -> Option<f32> {
    let mut monitor_info = MONITORINFOEXW::default();
    monitor_info.Base.cbSize = size_of::<MONITORINFOEXW>() as u32;
    // SAFETY: `hwnd` is a live window; `monitor_info` is a MONITORINFOEXW with its size set.
    let found = unsafe {
        let monitor = MonitorFromWindow(hwnd, NEAREST);
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
            let flags = QDC_ONLY_ACTIVE_PATHS as u32;
            if GetDisplayConfigBufferSizes(flags, &mut path_count, &mut mode_count) != 0 {
                return None;
            }
            let mut paths = vec![DISPLAYCONFIG_PATH_INFO::default(); path_count as usize];
            let mut modes = vec![DISPLAYCONFIG_MODE_INFO::default(); mode_count as usize];
            let status = QueryDisplayConfig(
                flags,
                &mut path_count,
                paths.as_mut_ptr(),
                &mut mode_count,
                modes.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if status == ERROR_INSUFFICIENT_BUFFER {
                continue;
            }
            if status != 0 {
                return None;
            }
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
    fn advanced_color_mapping_on_captured_values() {
        // A profiled SDR display under automatic color management, and an unprofiled one
        // reporting EDID.
        let wcg = AdvancedColor {
            kind: AdvancedColorKind::WideColorGamut,
            min_nits: 0.0,
            max_nits: 120.389,
            max_full_frame_nits: 120.389,
            sdr_white_nits: 80.0,
            primaries_xy: {
                let p = orr_color::RgbPrimaries::REC709;
                [p.red, p.green, p.blue, p.white].map(Into::into)
            },
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
