//! Frame generation's policy: the multiplier mode, the Dynamic rule that follows the
//! measured GPU cost, the camera NGX reprojects with and the system requirements.

use orr_dlss::{FgCamera, Support};
use orr_math::{DMat4, DVec3, Quat, Vec3};
use orr_scene::orr_gpu::Device;
use orr_scene::{Projection, SceneId};

/// How many images frame generation adds per rendered frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrameGenerationMode {
    Off,
    /// `n` generated images per rendered frame, 1 to `Support::max_generated`.
    Fixed(u32),
    /// Up to `max` generated images per rendered frame, fewer while the rendered frame's
    /// GPU time does not fit the base interval `1 / base_hz`.
    Dynamic {
        max: u32,
        base_hz: f32,
    },
}

impl FrameGenerationMode {
    /// `Dynamic` at a 60 Hz base rate with up to 3 generated images (4x), or as many as NGX
    /// can generate if fewer; `Off` without frame generation.
    pub fn recommended(support: &Support) -> Self {
        if !support.fg || support.max_generated == 0 {
            return Self::Off;
        }
        Self::Dynamic {
            max: support.max_generated.min(3),
            base_hz: 60.0,
        }
    }
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
pub(crate) struct FgCameraState {
    pub(crate) scene: SceneId,
    pub(crate) position: DVec3,
    /// World from view.
    pub(crate) rotation: Quat,
    pub(crate) projection: Projection,
    pub(crate) extent: [u32; 2],
}

impl FgCameraState {
    fn clip_from_view(&self) -> DMat4 {
        self.projection
            .clip_from_view(self.extent.into())
            .as_dmat4()
    }

    /// Whether `prev` can be reprojected to this camera.
    pub(crate) fn continues(&self, prev: &Self) -> bool {
        self.scene == prev.scene && self.extent == prev.extent
    }
}

/// The far plane frame generation is told; orr's projections have none.
const FG_FAR_M: f32 = 100_000.0;

/// NGX's camera for `cur`, reprojecting to `prev`, with the position relative to `anchor`.
pub(crate) fn fg_camera(cur: &FgCameraState, prev: &FgCameraState, anchor: DVec3) -> FgCamera {
    let clip = cur.clip_from_view();
    let view_from_clip = clip.inverse();
    let world_from_view = DMat4::from_quat(cur.rotation.as_dquat());
    let prev_view_from_world = DMat4::from_quat(prev.rotation.as_dquat().inverse());
    // Positions relative to each camera: the current one sits at the previous one's
    // position plus the motion.
    let motion = DMat4::from_translation(cur.position - prev.position);
    let clip_to_prev =
        prev.clip_from_view() * prev_view_from_world * motion * world_from_view * view_from_clip;
    let cols = |m: DMat4| m.as_mat4().to_cols_array_2d();
    let dir = |v: Vec3| (cur.rotation * v).to_array();
    FgCamera {
        view_to_clip: cols(clip),
        clip_to_view: cols(view_from_clip),
        clip_to_prev_clip: cols(clip_to_prev),
        prev_clip_to_clip: cols(clip_to_prev.inverse()),
        position: (cur.position - anchor).as_vec3().to_array(),
        up: dir(Vec3::Y),
        right: dir(Vec3::X),
        forward: dir(Vec3::NEG_Z),
        near: cur.projection.near,
        far: FG_FAR_M,
        fov_y: cur.projection.fov_y,
        aspect: cur.extent[0] as f32 / cur.extent[1].max(1) as f32,
    }
}

/// Whether Windows schedules the device's GPU in hardware, which frame generation needs.
/// `None` when it cannot tell; always true off Windows.
#[cfg(windows)]
pub(crate) fn hardware_scheduling(gpu: &Device) -> Option<bool> {
    use orr_scene::orr_gpu::vulkan::{self, ash::vk};
    use vk::TaggedStructure as _;
    use windows::Wdk::Graphics::Direct3D::{
        D3DKMT_CLOSEADAPTER, D3DKMT_OPENADAPTERFROMLUID, D3DKMT_QUERYADAPTERINFO,
        D3DKMT_WDDM_2_7_CAPS, D3DKMTCloseAdapter, D3DKMTOpenAdapterFromLuid,
        D3DKMTQueryAdapterInfo, KMTQAITYPE_WDDM_2_7_CAPS,
    };
    use windows::Win32::Foundation::LUID;

    let raw = vulkan::raw_device(gpu);
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push(&mut id);
    // SAFETY: a query on the device's live physical device.
    unsafe {
        raw.instance
            .get_physical_device_properties2(raw.physical_device, &mut props);
    }
    if id.device_luid_valid == vk::FALSE {
        return None;
    }
    let l = id.device_luid;
    let mut open = D3DKMT_OPENADAPTERFROMLUID {
        AdapterLuid: LUID {
            LowPart: u32::from_le_bytes([l[0], l[1], l[2], l[3]]),
            HighPart: i32::from_le_bytes([l[4], l[5], l[6], l[7]]),
        },
        hAdapter: 0,
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
    // SAFETY: every bit pattern is a valid u32. HwSchEnabled is bit 1.
    let value = unsafe { caps.Anonymous.Value };
    read.then_some(value & 0b10 != 0)
}

#[cfg(not(windows))]
pub(crate) fn hardware_scheduling(_gpu: &Device) -> Option<bool> {
    Some(true)
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
    use orr_math::{DVec4, Mat4};

    /// `m`, a column-vector matrix given as columns, applied to `p`.
    fn apply(m: [[f32; 4]; 4], p: DVec4) -> DVec4 {
        Mat4::from_cols_array_2d(&m).as_dmat4() * p
    }

    fn ms_at(costs: &[(f64, f64)], t: f64) -> f64 {
        costs
            .iter()
            .rev()
            .find(|(start, _)| t >= *start)
            .map_or(0.0, |&(_, ms)| ms)
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

    #[test]
    fn clip_to_prev_clip_reprojects_a_world_point() {
        let projection = Projection {
            fov_y: 0.9,
            near: 0.1,
        };
        let prev = FgCameraState {
            scene: SceneId(0),
            position: DVec3::new(1000.0, 2.0, -3.0),
            rotation: Quat::from_rotation_y(0.3),
            projection,
            extent: [1600, 900],
        };
        let cur = FgCameraState {
            position: prev.position + DVec3::new(0.4, 0.1, -0.7),
            rotation: Quat::from_rotation_y(0.34) * Quat::from_rotation_x(-0.05),
            ..prev
        };
        let c = fg_camera(&cur, &prev, prev.position);
        let world = DVec3::new(1003.0, 2.5, -12.0);
        let clip_of = |cam: &FgCameraState| {
            let view = cam.rotation.as_dquat().inverse() * (world - cam.position);
            let clip = cam.clip_from_view() * view.extend(1.0);
            clip / clip.w
        };
        let here = clip_of(&cur);
        let there = apply(c.clip_to_prev_clip, here);
        let there = there / there.w;
        assert!(
            (there - clip_of(&prev)).abs().max_element() < 1e-5,
            "{there} vs {}",
            clip_of(&prev)
        );
        let back = apply(c.prev_clip_to_clip, clip_of(&prev));
        assert!((back / back.w - here).abs().max_element() < 1e-5);
        let moved = Vec3::from_array(c.position) - Vec3::new(0.4, 0.1, -0.7);
        assert!(moved.abs().max_element() < 1e-6);
    }
}
