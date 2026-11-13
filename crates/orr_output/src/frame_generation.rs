//! Frame generation apart from what runs it: the multiplier mode, the Dynamic rule that follows
//! the measured GPU cost, and the two traits a backend implements.

use orr_math::{DMat4, DVec3, Quat};
use orr_scene::orr_gpu::{Allocation, CommandList, Device, Texture};
use orr_scene::{Projection, SceneId};

use crate::Result;

/// How many images frame generation adds per rendered frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrameGenerationMode {
    Off,
    /// `n` generated images per rendered frame, 1 to
    /// [`FrameGenerationBackend::max_generated`].
    Fixed(u32),
    /// Up to `max` generated images per rendered frame, fewer while the rendered frame's
    /// GPU time does not fit the base interval `1 / base_hz`.
    Dynamic {
        max: u32,
        base_hz: f32,
    },
}

impl FrameGenerationMode {
    /// `Dynamic` at a 60 Hz base rate with up to 3 generated images (4x), or `max_generated` if
    /// fewer; `Off` when that is 0.
    pub fn recommended(max_generated: u32) -> Self {
        if max_generated == 0 {
            return Self::Off;
        }
        Self::Dynamic {
            max: max_generated.min(3),
            base_hz: 60.0,
        }
    }
}

/// Reads a frame's flag once its evaluates have completed: whether they asked that its generated
/// images not be shown.
pub type InterpolationCheck = fn(&Allocation<u32>) -> bool;

/// Makes frame generation features on one device.
pub trait FrameGenerationBackend: Send + Sync {
    /// The most images it generates per rendered frame, at least 1, on `gpu`, the display's
    /// device, where `low_latency` tells whether the display has a low-latency handle. Fails with
    /// [`Error::FrameGenerationUnavailable`](crate::Error::FrameGenerationUnavailable) naming what
    /// the device or system lacks.
    fn max_generated(&self, gpu: &Device, low_latency: bool) -> Result<u32>;

    /// A feature for backbuffers like `backbuffer`; `hdr` when they hold PQ. May wait for setup
    /// on the Render queue, so call it on that queue's owner thread.
    fn feature(&self, backbuffer: &Texture, hdr: bool) -> Result<Box<dyn FrameGenerationFeature>>;

    /// How to read the flag in [`FrameGenerationInputs::disable_interpolation`]. `None` when the
    /// backend's images are always shown: frames then carry no flag, and the present thread does
    /// not wait for their evaluates.
    fn interpolation_check(&self) -> Option<InterpolationCheck>;
}

/// Frame generation for one backbuffer size and format. Dropping it releases its resources once
/// everything submitted before the drop has completed.
pub trait FrameGenerationFeature: Send + Sync {
    /// Records the frame's generated images into `list`, on a compute-capable queue of the
    /// feature's device.
    fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &FrameGenerationInputs<'_>,
    ) -> Result<()>;
}

/// A frame's inputs to frame generation: all a backend may use, of which it ignores the rest.
/// Images are 2D with one mip and layer, and have sampled or storage usage.
#[derive(Clone, Copy)]
pub struct FrameGenerationInputs<'a> {
    /// Display-encoded, without UI, in the format and size the feature was made for.
    pub backbuffer: &'a Texture,
    /// The backbuffer `previous_camera` saw, like `backbuffer`; `None` after a reset, or when
    /// `backbuffer` took its place.
    pub previous_backbuffer: Option<&'a Texture>,
    /// Reverse-Z device depth at the render extent, which can change while the feature lives.
    pub depth: &'a Texture,
    /// Unjittered motion at the render extent, previous UV minus current UV.
    pub motion: &'a Texture,
    /// RGBA16F bidirectional distortion field, while a lens or hook warps the image.
    pub distortion: Option<&'a Texture>,
    /// One per generated image, in presented order, in the backbuffer's format and size.
    pub outputs: &'a [&'a Texture],
    /// Host-readable and zeroed before the frame's evaluates, while the backend has an
    /// [`InterpolationCheck`].
    pub disable_interpolation: Option<&'a Allocation<u32>>,
    pub camera: FrameGenerationCamera,
    /// The camera to reproject to: the last generating frame's, or `camera` itself when that
    /// one has another scene or extent. Across a camera cut it is the camera before the cut, with
    /// `reset` set.
    pub previous_camera: FrameGenerationCamera,
    /// The point positions are given relative to, fixed between resets.
    pub anchor: DVec3,
    /// The jitter of `depth` and `motion`, render pixels.
    pub jitter: [f32; 2],
    /// Seconds between the previous rendered frame and this one.
    pub delta_time: f32,
    /// The previous frame is unrelated to this one.
    pub reset: bool,
    /// Increases by one each rendered frame, with frame generation on or off.
    pub frame_id: u64,
    /// Menus and loading screens.
    pub not_rendering_scene: bool,
}

/// Seconds the cost must stay over or under its threshold before `Dynamic` changes n.
const DYNAMIC_HOLD_S: f64 = 0.5;
/// A rise needs the cost at n + 1 under this share of the base interval.
const RISE_SHARE: f64 = 0.95;
/// On a variable-refresh display the base rate is capped this much under the maximum.
const VRR_MARGIN: f64 = 0.03;

/// `Dynamic`'s state: the generated count it chose and how long the cost has held.
#[derive(Clone, Debug)]
pub(crate) struct DynamicCount {
    max: u32,
    base_hz: f32,
    n: u32,
    ceiling: u32,
    over_since: Option<f64>,
    under_since: Option<f64>,
    /// The last measured frame-generation cost of one generated image, ms.
    fg_per_image_ms: f64,
}

impl DynamicCount {
    pub(crate) fn new(max: u32, base_hz: f32) -> Self {
        Self {
            max,
            base_hz,
            n: max,
            ceiling: max,
            over_since: None,
            under_since: None,
            fg_per_image_ms: 0.0,
        }
    }

    /// The ceiling `min(max, max_generated, floor(refresh / base) - 1)`; n is clamped to it.
    pub(crate) fn set_limits(&mut self, max_generated: u32, refresh_hz: Option<f64>) {
        let by_refresh = refresh_hz.map_or(u32::MAX, |r| {
            ((r / f64::from(self.base_hz)).floor() as u32).saturating_sub(1)
        });
        self.ceiling = self.max.min(max_generated).min(by_refresh);
        self.n = self.n.min(self.ceiling);
    }

    pub(crate) fn generated(&self) -> u32 {
        self.n
    }

    /// Feeds one rendered frame's measured GPU time `gpu_ms` (render and present queues,
    /// frame generation, UI paints and compose-UI included), of which `fg_ms` was frame
    /// generation, measured at `t` seconds with `generated` images. `paint_ms` is one UI
    /// paint's cost when the UI is drawn per presented image (each more image adds one), else
    /// 0. Returns the count to use.
    pub(crate) fn measure(
        &mut self,
        t: f64,
        gpu_ms: f64,
        fg_ms: f64,
        generated: u32,
        paint_ms: f64,
    ) -> u32 {
        if generated > 0 {
            self.fg_per_image_ms = fg_ms / f64::from(generated);
        }
        let budget = 1000.0 / f64::from(self.base_hz);
        if gpu_ms > budget {
            self.under_since = None;
            let since = *self.over_since.get_or_insert(t);
            if self.n > 0 && t - since >= DYNAMIC_HOLD_S {
                self.n -= 1;
                self.over_since = None;
            }
            return self.n;
        }
        self.over_since = None;
        if self.n >= self.ceiling {
            self.under_since = None;
            return self.n;
        }
        let at_next = gpu_ms + self.fg_per_image_ms + paint_ms;
        if at_next < RISE_SHARE * budget {
            let since = *self.under_since.get_or_insert(t);
            if t - since >= DYNAMIC_HOLD_S {
                self.n += 1;
                self.under_since = None;
            }
        } else {
            self.under_since = None;
        }
        self.n
    }
}

/// A view's camera as frame generation reprojects it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameGenerationCamera {
    pub scene: SceneId,
    pub position: DVec3,
    /// World from view.
    pub rotation: Quat,
    pub projection: Projection,
    /// The render extent, of the depth and motion.
    pub extent: [u32; 2],
    /// The output extent, whose aspect ratio the projection takes.
    pub output: [u32; 2],
}

impl FrameGenerationCamera {
    pub fn clip_from_view(&self) -> DMat4 {
        self.projection
            .clip_from_view(self.output.into())
            .as_dmat4()
    }

    /// Whether `prev` can be reprojected to this camera.
    pub(crate) fn continues(&self, prev: &Self) -> bool {
        self.scene == prev.scene && self.extent == prev.extent && self.output == prev.output
    }
}

/// The base-rate cap in microseconds for `generated` images per rendered frame: n + 1
/// refresh intervals, of a rate 3% under the maximum on a variable-refresh display.
pub(crate) fn min_interval_us(generated: u32, refresh_s: Option<f64>, variable: bool) -> u32 {
    let Some(interval) = refresh_s else {
        return 0;
    };
    let interval = if variable {
        interval / (1.0 - VRR_MARGIN)
    } else {
        interval
    };
    (interval * 1e6 * f64::from(generated + 1)).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms_at(costs: &[(f64, f64)], t: f64) -> f64 {
        costs
            .iter()
            .rev()
            .find(|(start, _)| t >= *start)
            .map_or_default(|&(_, ms)| ms)
    }

    #[test]
    fn dynamic_steps_down_and_back_after_each_hold() {
        // 60 Hz base on a 240 Hz display: ceiling 3. Frames arrive at 60 per second. The
        // cost per generated image is 1 ms on top of a base load raised in steps.
        let mut d = DynamicCount::new(3, 60.0);
        d.set_limits(5, Some(240.0));
        assert_eq!(d.generated(), 3);
        let budget = 1000.0 / 60.0;
        let base = [
            (0.0, 10.0),
            (1.0, budget - 2.0),
            (3.0, budget + 1.0),
            (6.0, 5.0),
        ];
        let mut changes = Vec::new();
        let mut n = d.generated();
        for frame in 0..600 {
            let t = f64::from(frame) / 60.0;
            let fg = f64::from(n);
            let next = d.measure(t, ms_at(&base, t) + fg, fg, n, 0.0);
            if next != n {
                changes.push((t, n, next));
            }
            n = next;
        }
        // Over budget from 1 s at n = 3 (17.7 ms): down to 2 after the hold; 16.7 ms is not
        // over, so it holds at 2 until the load rises at 3 s; then two holds bring it to 0.
        // From 6 s each rise follows its own hold until the ceiling.
        let expected = [
            (1.5, 3, 2),
            (3.5, 2, 1),
            (4.0, 1, 0),
            (6.5, 0, 1),
            (7.0, 1, 2),
            (7.5, 2, 3),
        ];
        assert_eq!(changes.len(), expected.len(), "{changes:?}");
        for ((t, from, to), (et, ef, eto)) in changes.iter().zip(expected) {
            assert_eq!((*from, *to), (ef, eto), "{changes:?}");
            assert!((t - et).abs() < 2.5 / 60.0, "{changes:?}");
        }
    }

    #[test]
    fn dynamic_rise_counts_one_more_ui_paint() {
        let budget = 1000.0 / 60.0;
        let rises = |paint_ms: f64| {
            let mut d = DynamicCount::new(2, 60.0);
            d.set_limits(5, Some(240.0));
            for frame in 0..60 {
                d.measure(f64::from(frame) / 60.0, budget + 1.0, 2.0, 2, paint_ms);
            }
            assert_eq!(d.generated(), 1);
            // 1 ms of frame generation per image: the next image fits by 0.5 ms alone.
            let cost = 0.95 * budget - 1.5;
            for frame in 60..120 {
                d.measure(f64::from(frame) / 60.0, cost, 1.0, 1, paint_ms);
            }
            d.generated() == 2
        };
        assert!(rises(0.0));
        assert!(!rises(1.0));
    }

    #[test]
    fn dynamic_ceiling_follows_refresh() {
        let mut d = DynamicCount::new(3, 60.0);
        d.set_limits(5, Some(165.0));
        assert_eq!(d.generated(), 1);
        d.set_limits(5, Some(60.0));
        assert_eq!(d.generated(), 0);
        let mut d = DynamicCount::new(3, 60.0);
        d.set_limits(1, None);
        assert_eq!(d.generated(), 1);
    }
}
