/// A frame's time on the consumer's clock. Kernels never see `absolute`: scene and view
/// roots carry `delta` and the frame index only.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameTime {
    /// Seconds.
    pub absolute: f64,
    /// Seconds since the previous frame.
    pub delta: f32,
}

impl FrameTime {
    /// Where `absolute` sits in a cycle of `period` seconds, in [0, 1). Computed in f64, so
    /// periodic animation stays smooth however long the clock has run.
    pub fn phase(&self, period: f64) -> f32 {
        (self.absolute.rem_euclid(period) / period) as f32
    }
}
