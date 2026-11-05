//! Platforms without a display watch.

use std::convert::Infallible;

use raw_window_handle::RawWindowHandle;

use crate::{DisplayFeed, Error, Result};

pub(crate) struct Watch(Infallible);

impl Watch {
    pub(crate) fn new(window: RawWindowHandle) -> Result<(Self, DisplayFeed)> {
        let _ = window;
        Err(Error::UnsupportedWindow)
    }
}
