//! Swapchains on Metal: a `CAMetalLayer` in the window's view, whose drawables are the images.
//! Metal 4 queues wait for a drawable before the commits that write it and signal it after;
//! its present follows in order with the queue's submits.

use std::cell::Cell;
use std::convert::Infallible;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use block2::RcBlock;
use dispatch2::DispatchQueue;
use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceITUR_2100_PQ, kCGColorSpaceSRGB};
use objc2_foundation::NSData;
use objc2_metal::{MTL4CommandQueue, MTLDrawable, MTLTexture};
use objc2_quartz_core::{CAEDRMetadata, CAMetalDrawable, CAMetalLayer, CATransaction};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::device::Device;
use super::texture::{Texture, pixel_format};
use super::{Obj, lock};
use crate::{
    Error, Format, HdrMetadata, LatencyMarker, LatencyTimings, OutputFormat, PresentMode,
    QueueKind, Refresh, StorageImageIndex, SwapchainDesc, TextureDesc, TextureUsage, TimelinePoint,
    ViewDesc,
};

/// The most drawables a `CAMetalLayer` holds.
const MAX_DRAWABLES: u32 = 3;

const EDR_REFERENCE_NITS: f32 = orr_color::EDR_REFERENCE_NITS as f32;

/// The encodings a layer takes. scRGB is left out: its 1.0 is 80 nits, but on an EDR layer
/// 1.0 is the screen's SDR white, which moves with the brightness setting.
const SUPPORTED: [OutputFormat; 2] = [OutputFormat::Sdr, OutputFormat::Hdr10Pq];

fn texture_format(f: OutputFormat) -> Format {
    match f {
        OutputFormat::Sdr => Format::Bgra8Unorm,
        OutputFormat::Hdr10Pq => Format::A2b10g10r10Unorm,
        OutputFormat::ScRgb => unreachable!("layers don't take scRGB"),
    }
}

/// Low latency for one swapchain; Metal has none, so [`Swapchain::latency`] returns `None`.
#[derive(Clone)]
pub struct LowLatency {
    never: Infallible,
}

impl LowLatency {
    pub fn set_mode(&self, low_latency: bool, boost: bool, min_interval_us: u32) {
        let _ = (low_latency, boost, min_interval_us);
        match self.never {}
    }

    pub fn sleep(&self) {
        match self.never {}
    }

    pub fn marker(&self, frame_id: u64, marker: LatencyMarker) {
        let _ = (frame_id, marker);
        match self.never {}
    }

    pub fn set_marker_hook(&self, hook: Option<fn(u64, LatencyMarker)>) {
        let _ = hook;
        match self.never {}
    }

    pub fn timings(&self, out: &mut Vec<LatencyTimings>) {
        let _ = out;
        match self.never {}
    }
}

/// An acquired drawable. Submit the work that writes it with
/// [`crate::Queue::submit_present`], then hand it to [`Swapchain::present`]; every acquired
/// frame is presented, or handed to [`Swapchain::discard`] when its submit fails.
pub struct SwapchainFrame {
    texture: Texture,
    storage: StorageImageIndex,
    format: OutputFormat,
    pub(super) drawable: Obj<ProtocolObject<dyn CAMetalDrawable>>,
    /// The point of the submit that writes it.
    pub(super) submitted: Cell<Option<TimelinePoint>>,
}

impl SwapchainFrame {
    /// The drawable's texture, for color targets and copies.
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// The drawable's storage view.
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
}

/// The size of a swapchain's window, for the thread that sizes frames while another acquires
/// and presents: Send, Sync and cheap to clone.
#[derive(Clone)]
pub struct WindowExtent {
    layer: Obj<CAMetalLayer>,
}

impl WindowExtent {
    /// The layer's size in pixels; `None` while it has no area.
    pub fn get(&self) -> Option<[u32; 2]> {
        layer_extent(&self.layer)
    }
}

/// The last present id the display has shown, which presented handlers raise.
#[derive(Default)]
struct Presented {
    last: Mutex<u64>,
    changed: Condvar,
}

impl Presented {
    fn record(&self, id: u64) {
        let mut last = lock(&self.last);
        *last = (*last).max(id);
        self.changed.notify_all();
    }
}

/// What the layer was last set to.
#[derive(Clone, Copy, PartialEq)]
struct Applied {
    format: OutputFormat,
    hdr: Option<HdrMetadata>,
    mode: PresentMode,
    drawables: u32,
    extent: [u32; 2],
}

/// A window's swapchain: a `CAMetalLayer` in its view. Owned by one thread, which acquires
/// and presents. A change of size, encoding or present mode applies at the next acquire.
pub struct Swapchain {
    device: Device,
    layer: Obj<CAMetalLayer>,
    /// The layer was added to the view, so dropping the swapchain takes it out.
    added: bool,
    format: OutputFormat,
    hdr: Option<HdrMetadata>,
    mode: PresentMode,
    drawables: u32,
    applied: Option<Applied>,
    next_present_id: u64,
    presented: Arc<Presented>,
}

impl Swapchain {
    /// Creates a swapchain for an AppKit or UIKit view, on the main thread. The device needs
    /// the Present queue (`QueueRequest::present`).
    pub fn new(
        device: &Device,
        window: RawWindowHandle,
        _display: RawDisplayHandle,
        desc: &SwapchainDesc,
    ) -> Result<Swapchain, Error> {
        let queue = device.queue(QueueKind::Present).ok_or_else(|| {
            Error::Invalid("the device was created without QueueRequest::present".into())
        })?;
        if MainThreadMarker::new().is_none() {
            return Err(Error::Invalid(
                "a Metal swapchain is created on the main thread, which owns the view".into(),
            ));
        }
        let made = match window {
            // SAFETY: the caller's handle names a live view, used here on the main thread.
            RawWindowHandle::AppKit(h) => unsafe {
                raw_window_metal::Layer::from_ns_view(h.ns_view)
            },
            // SAFETY: as above.
            RawWindowHandle::UiKit(h) => unsafe {
                raw_window_metal::Layer::from_ui_view(h.ui_view)
            },
            other => {
                return Err(Error::Unsupported {
                    device: device.caps().device_name.clone(),
                    missing: format!("a layer for {other:?}"),
                });
            }
        };
        let added = !made.pre_existing();
        // SAFETY: `into_raw` hands over its retain of a `CAMetalLayer`.
        let layer: Retained<CAMetalLayer> =
            unsafe { Retained::from_raw(made.into_raw().cast().as_ptr()) }
                .expect("raw-window-metal returns a layer");
        layer.setDevice(Some(&device.shared.device));
        layer.setFramebufferOnly(false);
        queue.raw.addResidencySet(&layer.residencySet());
        let s = Swapchain {
            device: device.clone(),
            layer: Obj(layer),
            added,
            format: if SUPPORTED.contains(&desc.format) {
                desc.format
            } else {
                OutputFormat::Sdr
            },
            hdr: desc.hdr,
            mode: desc.present_mode,
            drawables: desc.min_images.clamp(2, MAX_DRAWABLES),
            applied: None,
            next_present_id: 1,
            presented: Arc::default(),
        };
        Ok(s)
    }

    /// The output formats the layer takes.
    pub fn supported_formats(&self) -> &[OutputFormat] {
        &SUPPORTED
    }

    /// Switches the encoding and HDR metadata at the next acquire.
    pub fn set_format(&mut self, f: OutputFormat, hdr: Option<HdrMetadata>) -> Result<(), Error> {
        if !SUPPORTED.contains(&f) {
            return Err(Error::Invalid(format!(
                "the layer does not take {f:?} (it takes {SUPPORTED:?})"
            )));
        }
        self.format = f;
        self.hdr = hdr;
        Ok(())
    }

    /// Asks for at least `n` drawables at the next acquire; a layer holds two or three.
    pub fn set_min_images(&mut self, n: u32) {
        self.drawables = n.clamp(2, MAX_DRAWABLES);
    }

    /// Switches between FIFO and IMMEDIATE at the next acquire.
    pub fn set_present_mode(&mut self, m: PresentMode) {
        self.mode = m;
    }

    /// The window's size, readable from any thread.
    pub fn window_extent(&self) -> WindowExtent {
        WindowExtent {
            layer: self.layer.clone(),
        }
    }

    /// `None`: Metal has no low-latency mode.
    pub fn latency(&self) -> Option<LowLatency> {
        None
    }

    /// Unknown: Metal reports no refresh timing to swapchains.
    pub fn refresh(&self) -> Refresh {
        Refresh {
            interval: None,
            variable: false,
        }
    }

    /// Waits up to `timeout` for the present `id` (from [`Swapchain::present`]) to reach the
    /// display. False on timeout.
    pub fn wait_presented(&self, id: u64, timeout: Duration) -> Result<bool, Error> {
        let deadline = Instant::now() + timeout;
        let mut last = lock(&self.presented.last);
        while *last < id {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(false);
            }
            last = self
                .presented
                .changed
                .wait_timeout(last, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        Ok(true)
    }

    /// Acquires the next drawable. `None` while the view has no area (minimized) or the layer
    /// has no drawable to give; the caller skips this frame's work and present. Applies a new
    /// size, encoding or present mode to the layer first.
    pub fn acquire(&mut self) -> Result<Option<SwapchainFrame>, Error> {
        let Some(extent) = layer_extent(&self.layer) else {
            return Ok(None);
        };
        let wanted = Applied {
            format: self.format,
            hdr: self.hdr,
            mode: self.mode,
            drawables: self.drawables,
            extent,
        };
        if self.applied != Some(wanted) {
            self.apply(wanted);
            self.applied = Some(wanted);
        }
        // The layer gives none within a second while the screens reconfigure, among other
        // passing states; the frame is skipped, as for a window without area.
        let Some(drawable) = self.layer.nextDrawable() else {
            log::warn!("the layer gave no drawable; skipping a frame");
            return Ok(None);
        };
        let raw = drawable.texture();
        let desc = TextureDesc::d2(
            texture_format(self.format),
            raw.width() as u32,
            raw.height() as u32,
            TextureUsage::STORAGE | TextureUsage::COLOR,
        );
        let texture = Texture::drawable(&self.device.shared, raw, desc);
        let slot = self.device.alloc_image_slot()?;
        let storage = self
            .device
            .write_storage(slot, &texture, &ViewDesc::default());
        Ok(Some(SwapchainFrame {
            texture,
            storage,
            format: self.format,
            drawable: Obj(drawable),
            submitted: Cell::new(None),
        }))
    }

    /// Sets the layer to `a` in one transaction: the present thread has no run loop to
    /// commit an implicit one.
    fn apply(&self, a: Applied) {
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        let layer = &self.layer;
        layer.setDrawableSize(CGSize::new(f64::from(a.extent[0]), f64::from(a.extent[1])));
        layer.setMaximumDrawableCount(a.drawables as usize);
        #[cfg(target_os = "macos")]
        layer.setDisplaySyncEnabled(a.mode == PresentMode::Fifo);
        // SAFETY: CoreGraphics' color space names are constants.
        let (space, edr) = unsafe {
            match a.format {
                OutputFormat::Sdr => (kCGColorSpaceSRGB, false),
                OutputFormat::Hdr10Pq => (kCGColorSpaceITUR_2100_PQ, true),
                OutputFormat::ScRgb => unreachable!("layers don't take scRGB"),
            }
        };
        layer.setPixelFormat(
            pixel_format(texture_format(a.format)).expect("layer formats have pixel formats"),
        );
        let space = CGColorSpace::with_name(Some(space));
        layer.setColorspace(space.as_deref());
        layer.setWantsExtendedDynamicRangeContent(edr);
        let metadata = a.hdr.filter(|_| edr).map(|h| hdr10(&h));
        layer.setEDRMetadata(metadata.as_deref());
        CATransaction::commit();
        log::debug!(
            "swapchain layer: {:?} {:?} {:?}",
            a.extent,
            a.format,
            a.mode
        );
    }

    /// Presents `frame` once its submit has run, and returns its present id: strictly
    /// increasing across every present of this swapchain. `batch` is ignored: Metal has no
    /// metered presents.
    pub fn present(&mut self, frame: SwapchainFrame, batch: Option<u32>) -> Result<u64, Error> {
        let _ = batch;
        let point = frame
            .submitted
            .get()
            .expect("a swapchain frame is presented after Queue::submit_present");
        let id = self.next_present_id;
        self.next_present_id += 1;
        let presented = Arc::clone(&self.presented);
        let handler = RcBlock::new(move |_: NonNull<ProtocolObject<dyn MTLDrawable>>| {
            presented.record(id);
        });
        let drawable = frame.drawable;
        // SAFETY: Metal copies the block, which captures only `Send` values; it is added
        // before the present.
        unsafe { drawable.addPresentedHandler(RcBlock::as_ptr(&handler)) };
        self.queue().in_order(move |_| {
            let drawable: &ProtocolObject<dyn CAMetalDrawable> = &drawable;
            ProtocolObject::<dyn MTLDrawable>::from_ref(drawable).present();
        });
        self.device.free_image_slot(frame.storage.0, point);
        Ok(id)
    }

    /// Hands back `frame` without showing it, for a frame whose own work failed to submit. A
    /// frame that was submitted is presented, so the queue's signal has its present.
    pub fn discard(&mut self, frame: SwapchainFrame) -> Result<(), Error> {
        if frame.submitted.get().is_some() {
            return self.present(frame, None).map(drop);
        }
        self.device.free_image_slot_after_submitted(frame.storage.0);
        Ok(())
    }

    fn queue(&self) -> &super::Queue {
        self.device
            .queue(QueueKind::Present)
            .expect("presenting devices have the Present queue")
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let layer = self.layer.clone();
        let queue = self.queue().raw.clone();
        let added = self.added;
        // The layer's drawables stay resident until the GPU is done with them. The release
        // runs on whichever thread collects garbage, and the view's layers change on the main
        // thread only.
        self.device.shared.defer(Box::new(move || {
            queue.removeResidencySet(&layer.residencySet());
            if added {
                let remove = move || {
                    CATransaction::begin();
                    layer.removeFromSuperlayer();
                    CATransaction::commit();
                };
                if MainThreadMarker::new().is_some() {
                    remove();
                } else {
                    DispatchQueue::main().exec_async(remove);
                }
            }
        }));
    }
}

/// The layer's size in pixels; `None` while either side is 0.
fn layer_extent(layer: &CAMetalLayer) -> Option<[u32; 2]> {
    let size = layer.bounds().size;
    let scale = layer.contentsScale();
    let extent = [
        (size.width * scale).round() as u32,
        (size.height * scale).round() as u32,
    ];
    (!extent.contains(&0)).then_some(extent)
}

/// HDR10 metadata as `CAEDRMetadata` takes it: SMPTE ST 2086 mastering display color volume
/// (primaries in green, blue, red order, in 0.00002 steps; luminance in 0.0001 nit steps) and
/// CTA-861.3 content light levels, big-endian.
fn hdr10(h: &HdrMetadata) -> Retained<CAEDRMetadata> {
    let mut display = Vec::with_capacity(24);
    let xy = |c: [f32; 2], out: &mut Vec<u8>| {
        for v in c {
            out.extend_from_slice(&((v / 0.00002).round() as u16).to_be_bytes());
        }
    };
    for i in [1, 2, 0] {
        xy(h.primaries[i], &mut display);
    }
    xy(h.white, &mut display);
    for nits in [h.max_luminance, h.min_luminance] {
        display.extend_from_slice(&((nits / 0.0001).round() as u32).to_be_bytes());
    }
    let mut content = Vec::with_capacity(4);
    for nits in [h.max_cll, h.max_fall] {
        content.extend_from_slice(&(nits.round() as u16).to_be_bytes());
    }
    CAEDRMetadata::HDR10MetadataWithDisplayInfo_contentInfo_opticalOutputScale(
        Some(&NSData::with_bytes(&display)),
        Some(&NSData::with_bytes(&content)),
        EDR_REFERENCE_NITS,
    )
}
