#![allow(non_snake_case)]

use std::cell::Cell;
use std::ffi::c_void;

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
use windows::Win32::System::WinRT::{
    CreateDispatcherQueueController, DQTAT_COM_NONE, DQTYPE_THREAD_CURRENT, DispatcherQueueOptions,
    RO_INIT_SINGLETHREADED, RoInitialize, RoUninitialize,
};
use windows::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
use windows::Win32::UI::WindowsAndMessaging::{WM_DISPLAYCHANGE, WM_WINDOWPOSCHANGED};
use windows_core::{GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface, Ref};

use crate::{DisplayFeed, DisplayInfo, Error, Result};

// Windows 11 22621+. windows-rs has no binding; the three IInspectable slots come first.
#[windows_core::interface("7449121C-382B-4705-8DA7-A795BA482013")]
unsafe trait IDisplayInformationStaticsInterop: IUnknown {
    fn GetIids(&self, count: *mut u32, iids: *mut *mut GUID) -> HRESULT;
    fn GetRuntimeClassName(&self, name: *mut *mut c_void) -> HRESULT;
    fn GetTrustLevel(&self, level: *mut i32) -> HRESULT;
    fn GetForWindow(&self, window: HWND, riid: *const GUID, info: *mut *mut c_void) -> HRESULT;
}

const SUBCLASS_ID: usize = 0x6f72_725f_6477;

/// Watches the monitor a window is on and publishes what the OS reports for it to a
/// `DisplayFeed`: on HDR, luminance and SDR white changes, on moves to another monitor and on
/// display mode changes.
///
/// Create it on the thread that owns the window and pumps its messages; changes arrive
/// there. Drop it on that thread, before the window is destroyed. It uses the thread's
/// dispatcher queue, creating one if the thread has none.
pub struct DisplayWatch {
    // Fields drop in order: hooks, then the queue, then the apartment.
    _hooks: Hooks,
    _queue: Option<QueueController>,
    _apartment: Option<Apartment>,
}

struct Hooks {
    hwnd: HWND,
    token: i64,
    subclass: Box<Subclass>,
}

struct Subclass {
    info: DisplayInformation,
    feed: DisplayFeed,
    monitor: Cell<HMONITOR>,
}

struct QueueController(DispatcherQueueController);

struct Apartment;

impl DisplayWatch {
    pub fn new(window: RawWindowHandle) -> Result<(Self, DisplayFeed)> {
        let RawWindowHandle::Win32(handle) = window else {
            return Err(Error::NotWin32Window);
        };
        let hwnd = HWND(handle.hwnd.get() as *mut c_void);

        // SAFETY: called on the window's thread; `Apartment` balances it on drop. An
        // apartment of another kind already on the thread serves WinRT as well.
        let apartment = match unsafe { RoInitialize(RO_INIT_SINGLETHREADED) } {
            Ok(()) => Some(Apartment),
            Err(e) if e.code() == RPC_E_CHANGED_MODE => None,
            Err(e) => return Err(e.into()),
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
        let info = display_information(hwnd)?;
        let feed = DisplayFeed::new(read(&info, hwnd));

        let handler_feed = feed.clone();
        let handler_hwnd = hwnd.0 as isize;
        let token = info.AdvancedColorInfoChanged(&TypedEventHandler::new(
            move |sender: Ref<DisplayInformation>, _| {
                if let Some(info) = sender.as_ref() {
                    handler_feed.set(read(info, HWND(handler_hwnd as *mut c_void)));
                }
                Ok(())
            },
        ))?;

        let subclass = Box::new(Subclass {
            info,
            feed: feed.clone(),
            // SAFETY: `hwnd` is a live window.
            monitor: Cell::new(unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) }),
        });
        // SAFETY: on the window's thread; the box outlives the subclass, which `Hooks`
        // removes on drop.
        let installed = unsafe {
            SetWindowSubclass(
                hwnd,
                Some(subclass_proc),
                SUBCLASS_ID,
                &raw const *subclass as usize,
            )
        };
        if !installed.as_bool() {
            let _ = subclass.info.RemoveAdvancedColorInfoChanged(token);
            return Err(windows_core::Error::from_thread().into());
        }

        let hooks = Hooks {
            hwnd,
            token,
            subclass,
        };
        Ok((
            Self {
                _hooks: hooks,
                _queue: queue,
                _apartment: apartment,
            },
            feed,
        ))
    }
}

impl Drop for Hooks {
    fn drop(&mut self) {
        // SAFETY: on the window's thread, removing the subclass `DisplayWatch::new` installed.
        unsafe {
            let _ = RemoveWindowSubclass(self.hwnd, Some(subclass_proc), SUBCLASS_ID);
        }
        let _ = self
            .subclass
            .info
            .RemoveAdvancedColorInfoChanged(self.token);
    }
}

impl Drop for QueueController {
    fn drop(&mut self) {
        let _ = self.0.ShutdownQueueAsync();
    }
}

impl Drop for Apartment {
    fn drop(&mut self) {
        // SAFETY: balances `RoInitialize` in `DisplayWatch::new` on the same thread, after the
        // watch's WinRT objects are released.
        unsafe { RoUninitialize() };
    }
}

unsafe extern "system" fn subclass_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _id: usize,
    data: usize,
) -> LRESULT {
    // SAFETY: `data` is the `Subclass` box, alive until the subclass is removed.
    let subclass = unsafe { &*(data as *const Subclass) };
    match msg {
        WM_DISPLAYCHANGE => subclass.feed.set(read(&subclass.info, hwnd)),
        WM_WINDOWPOSCHANGED => {
            // SAFETY: `hwnd` is the live subclassed window.
            let monitor = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
            if subclass.monitor.replace(monitor) != monitor {
                subclass.feed.set(read(&subclass.info, hwnd));
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

/// Reads the window's monitor. Fields that fail to read are left unknown.
fn read(info: &DisplayInformation, hwnd: HWND) -> DisplayInfo {
    let refresh = refresh_hz(hwnd);
    match AdvancedColor::new(info) {
        Ok(color) => color.display_info(refresh),
        Err(e) => {
            log::warn!("reading AdvancedColorInfo failed: {e}");
            DisplayInfo {
                refresh_hz: refresh,
                ..DisplayInfo::default()
            }
        }
    }
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
            max_nits: 580.0,
            max_full_frame_nits: 0.0,
            sdr_white_nits: 203.0,
            ..wcg
        };
        let info = hdr.display_info(Some(0.0));
        assert_eq!(info.os_hdr_on, Some(true));
        assert_eq!(info.max_full_frame_nits, None);
        assert_eq!(info.refresh_hz, None);
    }
}
