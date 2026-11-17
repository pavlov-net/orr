//! How orr's reconstruction inputs map onto MetalFX's, as the probe tests measure them.

/// MetalFX's jitter offset for orr's `jitter`, which samples each pixel at its center minus the
/// value, in render pixels with y down.
pub fn jitter(orr: [f32; 2]) -> [f32; 2] {
    orr
}

/// MetalFX's motion vector scale for orr's motion, previous UV minus current UV at render size
/// `render`: MetalFX takes motion in pixels pointing to where each pixel was.
pub fn motion_scale(render: [u32; 2]) -> [f32; 2] {
    [render[0] as f32, render[1] as f32]
}
