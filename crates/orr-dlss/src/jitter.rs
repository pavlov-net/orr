/// Number of jitter phases before the sequence repeats: 32, or `ceil(8 * scale^2)` when that
/// is larger, where `scale` is output width over render width.
pub fn jitter_phases(render: [u32; 2], output: [u32; 2]) -> u32 {
    let scale = f64::from(output[0]) / f64::from(render[0]);
    32u32.max((8.0 * scale * scale).ceil() as u32)
}

/// NGX's jitter offset for `frame`, in render pixels, each axis in [-0.5, 0.5): Halton(2, 3)
/// at index `frame % phases + 1`, minus 0.5.
///
/// NGX's offset is the negative of the sample offset: the projection places each pixel's
/// sample at the pixel centre minus this value on both axes, with y pointing down.
pub fn jitter(frame: u64, phases: u32) -> [f32; 2] {
    let i = (frame % u64::from(phases)) as u32 + 1;
    [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
}

/// Texture LOD bias for rendering at `render` into `output`: `log2(render / output) - 1`.
pub fn mip_bias(render: [u32; 2], output: [u32; 2]) -> f32 {
    (f64::from(render[0]) / f64::from(output[0])).log2() as f32 - 1.0
}

fn halton(mut i: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_cycle_covers_the_pixel_evenly() {
        let phases = super::jitter_phases([1920, 1080], [3840, 2160]);
        let cycle: Vec<[f32; 2]> = (0..u64::from(phases))
            .map(|f| super::jitter(f, phases))
            .collect();
        assert!(cycle.iter().flatten().all(|v| (-0.5..0.5).contains(v)));
        for axis in 0..2 {
            let mean = cycle.iter().map(|j| j[axis]).sum::<f32>() / phases as f32;
            assert!(mean.abs() < 0.05, "axis {axis} mean {mean}");
        }
        assert_eq!(super::jitter(u64::from(phases), phases), cycle[0]);
    }
}
