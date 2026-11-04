//! What a window's OS reports for the display the window is on: whether HDR is on, the
//! display's luminance range, primaries and refresh. A [`DisplayWatch`] keeps a
//! [`DisplayFeed`] current with it, with one backend per platform.

use std::sync::{Arc, Mutex, PoisonError};

use raw_window_handle::RawWindowHandle;

#[cfg(windows)]
mod win32;
#[cfg(windows)]
use win32 as imp;

#[cfg(target_os = "macos")]
mod appkit;
#[cfg(target_os = "macos")]
use appkit as imp;

#[cfg(not(any(windows, target_os = "macos")))]
mod none;
#[cfg(not(any(windows, target_os = "macos")))]
use none as imp;

/// What the OS reports for the window's monitor. With HDR on, the luminance fields are the
/// display's default HDR color profile (the Windows HDR Calibration app writes one), else
/// EDID; a consumer's request overrides them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DisplayInfo {
    pub os_hdr_on: Option<bool>,
    pub max_nits: Option<f32>,
    pub max_full_frame_nits: Option<f32>,
    pub min_nits: Option<f32>,
    pub sdr_white_nits: Option<f32>,
    pub refresh_hz: Option<f32>,
    /// Red, green, blue and white, CIE 1931 xy.
    pub primaries_xy: Option<[[f32; 2]; 4]>,
}

/// The latest `DisplayInfo` and a count that rises each time it changes. Clones share one
/// value; any thread may read or write it.
#[derive(Clone, Debug)]
pub struct DisplayFeed(Arc<Mutex<(u64, DisplayInfo)>>);

impl DisplayFeed {
    pub fn new(info: DisplayInfo) -> Self {
        Self(Arc::new(Mutex::new((0, info))))
    }

    /// For a window whose OS report is unknown. Surfaces can list HDR encodings with HDR off
    /// (Windows lists PQ either way), and PQ on an SDR desktop looks dim and washed out, so
    /// it resolves to SDR.
    pub fn hdr_off() -> Self {
        Self::new(DisplayInfo {
            os_hdr_on: Some(false),
            ..DisplayInfo::default()
        })
    }

    /// Stores `info`, raising the count if it differs from the current value.
    pub fn set(&self, info: DisplayInfo) {
        let mut latest = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if latest.1 != info {
            *latest = (latest.0 + 1, info);
        }
    }

    /// The change count and the current value.
    pub fn latest(&self) -> (u64, DisplayInfo) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Errors from orr_window.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no display watch for this kind of window")]
    UnsupportedWindow,
    #[error("a display watch must be made on the thread that owns the window")]
    NotWindowThread,
    #[cfg(windows)]
    #[error("display watch: {0}")]
    Windows(#[from] windows_core::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Watches the monitor a window is on and publishes what the OS reports for it to a
/// [`DisplayFeed`]: on a timer, since some changes raise no event, and on the OS's events for
/// the window changing monitors and the display changing.
///
/// Create it on the thread that owns the window and pumps its events; reads happen there.
/// It may be dropped on any thread, and before or after the window is destroyed.
pub struct DisplayWatch {
    _imp: imp::Watch,
}

impl DisplayWatch {
    /// Watches `window` and returns the feed it keeps current, already holding a first
    /// reading.
    pub fn new(window: RawWindowHandle) -> Result<(Self, DisplayFeed)> {
        let (imp, feed) = imp::Watch::new(window, None)?;
        Ok((Self { _imp: imp }, feed))
    }

    /// As [`Self::new`], and also calls `hook` on the window's thread with every message
    /// the window receives, before the window handles it.
    #[cfg(windows)]
    pub fn with_message_hook(
        window: RawWindowHandle,
        hook: fn(u32),
    ) -> Result<(Self, DisplayFeed)> {
        let (imp, feed) = imp::Watch::new(window, Some(hook))?;
        Ok((Self { _imp: imp }, feed))
    }
}

/// How far the SDR white and the peak may move, as a share of their value, and still count as
/// unchanged: each change moves the tone curve, and the readings drift (on macOS the peak
/// follows the brightness setting).
const STEADY_TOLERANCE: f32 = 0.01;

/// `new` keeping the last SDR white and peak while each stays within `STEADY_TOLERANCE` of
/// `new`'s.
#[cfg_attr(
    not(any(windows, target_os = "macos", test)),
    expect(dead_code, reason = "used by the platform watches")
)]
fn steady(last: DisplayInfo, mut new: DisplayInfo) -> DisplayInfo {
    let hold = |old: Option<f32>, new: &mut Option<f32>| {
        if let (Some(old), Some(value)) = (old, *new)
            && (value - old).abs() <= STEADY_TOLERANCE * old
        {
            *new = Some(old);
        }
    };
    hold(last.sdr_white_nits, &mut new.sdr_white_nits);
    hold(last.max_nits, &mut new.max_nits);
    new
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdr_white_and_peak_stay_through_small_changes_only() {
        let with = |nits: f32| DisplayInfo {
            os_hdr_on: Some(true),
            sdr_white_nits: Some(nits),
            max_nits: Some(nits * 4.0),
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
}
