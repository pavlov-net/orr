//! Image quality measures over the frames of a run: noise, flicker, error and bias against a
//! reference, how correlated the noise is, and how many frames a change of lighting takes to
//! reach the image.
//!
//! All of them work on luminance. Ratios take a floor of 1% of the mean luminance of the image,
//! so dark pixels do not dominate them.

/// Pixels on a side of the tiles that bias and step responses are measured over.
pub const TILE: u32 = 16;
/// Share of the mean luminance added to a ratio's denominator.
const FLOOR: f64 = 0.01;
/// Relative noise under which a pixel takes no part in the correlation measures.
const QUIET: f64 = 0.02;

/// Luminance of Rec.709 RGBA pixels.
pub fn luminance(rgba: &[f32]) -> Vec<f32> {
    rgba.chunks_exact(4)
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect()
}

/// The frames of a run of a still camera, or any frames of one size.
pub struct Series {
    width: u32,
    count: u32,
    sum: Vec<f64>,
    squares: Vec<f64>,
    /// Per tile, its mean luminance in each frame.
    tiles: Vec<Vec<f64>>,
    /// The first `keep` frames.
    kept: Vec<Vec<f32>>,
    keep: usize,
    previous: Option<Vec<f32>>,
    /// Sum over frames after the first of the mean ratio of change to luminance, with the
    /// floor at each frame's own mean.
    flicker: f64,
}

impl Series {
    /// An empty series of `width` x `height` frames that keeps the first `keep` whole, which
    /// the correlation measures read.
    pub fn new(width: u32, height: u32, keep: usize) -> Self {
        let tiles = (width.div_ceil(TILE) * height.div_ceil(TILE)) as usize;
        let pixels = (width * height) as usize;
        Self {
            width,
            count: 0,
            sum: vec![0.0; pixels],
            squares: vec![0.0; pixels],
            tiles: vec![Vec::new(); tiles],
            kept: Vec::new(),
            keep,
            previous: None,
            flicker: 0.0,
        }
    }

    /// Adds a frame of luminance values.
    pub fn add(&mut self, frame: Vec<f32>) {
        assert_eq!(frame.len(), self.sum.len(), "frame size");
        for ((s, q), &v) in self.sum.iter_mut().zip(&mut self.squares).zip(&frame) {
            let v = f64::from(v);
            *s += v;
            *q += v * v;
        }
        let means = tile_means(self.width, self.tiles.len(), &frame);
        for (tile, mean) in self.tiles.iter_mut().zip(means) {
            tile.push(mean);
        }
        if let Some(previous) = &self.previous {
            let floor = FLOOR * mean_of(&frame);
            let change: f64 = frame
                .iter()
                .zip(previous)
                .map(|(&a, &b)| f64::from((a - b).abs()) / (0.5 * f64::from(a + b) + floor))
                .sum();
            self.flicker += change / frame.len() as f64;
        }
        self.count += 1;
        if self.kept.len() < self.keep {
            self.kept.push(frame.clone());
        }
        self.previous = Some(frame);
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// Each pixel's mean.
    pub fn mean(&self) -> Vec<f32> {
        let n = f64::from(self.count);
        self.sum.iter().map(|s| (s / n) as f32).collect()
    }

    /// Each pixel's standard deviation over the frames, over its mean.
    pub fn noise_map(&self) -> Vec<f32> {
        let n = f64::from(self.count);
        let floor = FLOOR * self.sum.iter().sum::<f64>() / (n * self.sum.len() as f64);
        self.sum
            .iter()
            .zip(&self.squares)
            .map(|(&s, &q)| {
                let mean = s / n;
                let variance = ((q - s * s / n) / (n - 1.0)).max(0.0);
                (variance.sqrt() / (mean + floor)) as f32
            })
            .collect()
    }

    /// The mean of [`Self::noise_map`].
    pub fn noise(&self) -> f64 {
        mean_of(&self.noise_map())
    }

    /// The mean frame-to-frame change of a pixel, over its luminance.
    pub fn flicker(&self) -> f64 {
        self.flicker / f64::from(self.count - 1)
    }

    /// The last frame.
    pub fn last(&self) -> &[f32] {
        self.previous.as_ref().expect("a frame")
    }

    /// Mean of [`relative_error`] of the kept frames against `reference`.
    pub fn mean_relative_error(&self, reference: &[f32]) -> f64 {
        let total: f64 = self.kept.iter().map(|f| relative_error(f, reference)).sum();
        total / self.kept.len() as f64
    }

    /// Correlation of a pixel's noise with the noise of the pixel `lag` to its right, over the
    /// kept frames. Noise here is the deviation from the pixel's mean over its deviation over
    /// the frames, so every noisy pixel counts alike; pixels that hardly change, such as the sky
    /// behind a smooth medium, count for nothing. 0 for independent pixels.
    pub fn spatial_correlation(&self, lag: u32) -> f64 {
        let z = self.scores();
        let w = self.width as usize;
        let lag = lag as usize;
        let (mut product, mut square) = (0.0, 0.0);
        for frame in &z {
            for row in frame.chunks_exact(w) {
                for (a, b) in row.iter().zip(&row[lag..]) {
                    product += a * b;
                    square += a * a;
                }
            }
        }
        product / square
    }

    /// Correlation of a pixel's noise with its own `lag` kept frames later.
    pub fn temporal_correlation(&self, lag: usize) -> f64 {
        let z = self.scores();
        let (mut product, mut square) = (0.0, 0.0);
        for (a, b) in z.iter().zip(&z[lag..]) {
            for (x, y) in a.iter().zip(b) {
                product += x * y;
                square += x * x;
            }
        }
        product / square
    }

    /// The kept frames as deviations from each pixel's mean over them, in standard deviations.
    fn scores(&self) -> Vec<Vec<f64>> {
        let n = self.kept.len() as f64;
        assert!(n >= 2.0, "correlation needs kept frames");
        let pixels = self.sum.len();
        let mut mean = vec![0.0; pixels];
        let mut spread = vec![0.0; pixels];
        for frame in &self.kept {
            for (m, &v) in mean.iter_mut().zip(frame) {
                *m += f64::from(v) / n;
            }
        }
        for frame in &self.kept {
            for ((s, &m), &v) in spread.iter_mut().zip(&mean).zip(frame) {
                let d = f64::from(v) - m;
                *s += d * d / (n - 1.0);
            }
        }
        let floor = FLOOR * mean.iter().sum::<f64>() / pixels as f64;
        self.kept
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .zip(&mean)
                    .zip(&spread)
                    .map(|((&v, &m), &s)| {
                        if s.sqrt() > QUIET * (m + floor) {
                            (f64::from(v) - m) / s.sqrt()
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// How many frames after `step` each tile that a change in that frame moved by at least 5%
    /// takes for the mean of the next eight to come within 10% of the change of where it settles.
    /// Eight frames keep noise from deciding it. The last eight frames give where a tile settles
    /// and the eight before `step` where it began.
    pub fn step_response(&self, step: usize) -> Vec<u32> {
        let mut frames = Vec::new();
        for tile in &self.tiles {
            assert!(
                step >= 8 && tile.len() >= step + 16,
                "frames around the step"
            );
            let before = tile[step - 8..step].iter().sum::<f64>() / 8.0;
            let after = tile[tile.len() - 8..].iter().sum::<f64>() / 8.0;
            let change = (after - before).abs();
            if change < 0.05 * before.abs().max(after.abs()) {
                continue;
            }
            let settled = tile[step..]
                .windows(8)
                .position(|w| (w.iter().sum::<f64>() / 8.0 - after).abs() <= 0.1 * change);
            frames.push(settled.map_or(tile.len() - step, |k| k) as u32);
        }
        frames.sort_unstable();
        frames
    }
}

fn mean_of(values: &[f32]) -> f64 {
    values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64
}

/// The mean of each of a frame's `tiles` tiles, in rows, for a frame `width` pixels wide.
fn tile_means(width: u32, tiles: usize, frame: &[f32]) -> Vec<f64> {
    let columns = width.div_ceil(TILE) as usize;
    let mut sums = vec![(0.0, 0u32); tiles];
    for (y, row) in frame.chunks_exact(width as usize).enumerate() {
        let first = (y / TILE as usize) * columns;
        for (x, &v) in row.iter().enumerate() {
            let tile = &mut sums[first + x / TILE as usize];
            tile.0 += f64::from(v);
            tile.1 += 1;
        }
    }
    sums.iter().map(|&(s, n)| s / f64::from(n)).collect()
}

/// The mean of each pixel's difference from `reference` over the reference, each capped at 1 so
/// a few fireflies in either image do not decide it.
pub fn relative_error(frame: &[f32], reference: &[f32]) -> f64 {
    mean_of(&error_map(frame, reference))
}

/// Each pixel's term of [`relative_error`].
pub fn error_map(frame: &[f32], reference: &[f32]) -> Vec<f32> {
    let floor = FLOOR as f32 * mean_of(reference) as f32;
    frame
        .iter()
        .zip(reference)
        .map(|(&a, &b)| ((a - b).abs() / (b + floor)).min(1.0))
        .collect()
}

/// How two series' tile means compare.
#[derive(Clone, Debug)]
pub struct Bias {
    pub tiles: usize,
    /// Tiles whose means differ at the test's significance. About `tiles` x the significance
    /// when the series converge to the same image.
    pub flagged: usize,
    /// Mean over the tiles of the difference of means over the reference's mean.
    pub mean_difference: f64,
    /// Per tile, the difference of means over the reference's mean where flagged, else 0.
    pub map: Vec<f32>,
}

/// Welch's t-test (Jung, Hanika and Dachsbacher, JCGT 2020) of each tile of `series` against
/// `reference` at two-sided `significance`. Frames that remember earlier ones count for less:
/// each series' sample count falls by its lag 1 autocorrelation. With at least some tens of
/// frames a normal distribution stands in for Student's t.
pub fn welch(series: &Series, reference: &Series, significance: f64) -> Bias {
    assert_eq!(series.tiles.len(), reference.tiles.len(), "series size");
    let mut flagged = 0;
    let mut map = Vec::with_capacity(series.tiles.len());
    let mut total = 0.0;
    for (a, b) in series.tiles.iter().zip(&reference.tiles) {
        let (mean_a, error_a) = mean_and_variance(a);
        let (mean_b, error_b) = mean_and_variance(b);
        let difference = (mean_a - mean_b) / mean_b.abs().max(f64::MIN_POSITIVE);
        total += difference.abs();
        let t = (mean_a - mean_b).abs() / (error_a + error_b).sqrt();
        let significant = two_sided_p(t) < significance;
        flagged += usize::from(significant);
        map.push(if significant { difference as f32 } else { 0.0 });
    }
    Bias {
        tiles: map.len(),
        flagged,
        mean_difference: total / map.len() as f64,
        map,
    }
}

/// The mean of `samples` and the variance of that mean, allowing for lag 1 correlation.
fn mean_and_variance(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let square: f64 = samples.iter().map(|v| (v - mean) * (v - mean)).sum();
    let product: f64 = samples
        .iter()
        .zip(&samples[1..])
        .map(|(a, b)| (a - mean) * (b - mean))
        .sum();
    let rho = if square > 0.0 {
        (product / square).clamp(0.0, 0.95)
    } else {
        0.0
    };
    let effective = (n * (1.0 - rho) / (1.0 + rho)).max(2.0);
    (mean, square / (n - 1.0) / effective)
}

/// Probability that a standard normal value lies further from 0 than `t`.
fn two_sided_p(t: f64) -> f64 {
    // erfc by Abramowitz and Stegun 7.1.26.
    let x = t / std::f64::consts::SQRT_2;
    let k = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = k
        * (0.254_829_592
            + k * (-0.284_496_736
                + k * (1.421_413_741 + k * (-1.453_152_027 + k * 1.061_405_429))));
    poly * (-x * x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 and normal values by Box-Muller; fixed seeds keep the tests repeatable.
    struct Rng(u64);

    impl Rng {
        fn unit(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (a, b) = (self.unit().max(1e-300), self.unit());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    const SIZE: u32 = 128;

    /// `frames` frames of mean `mean` and relative noise `noise`, independent per pixel and frame.
    fn white(frames: u32, mean: f64, noise: f64, keep: usize, seed: u64) -> Series {
        let mut rng = Rng(seed);
        let mut s = Series::new(SIZE, SIZE, keep);
        for _ in 0..frames {
            s.add(
                (0..SIZE * SIZE)
                    .map(|_| (mean * (1.0 + noise * rng.normal())) as f32)
                    .collect(),
            );
        }
        s
    }

    #[test]
    fn noise_and_flicker_read_a_known_deviation() {
        let s = white(64, 2.0, 0.1, 0, 1);
        // Noise over the pixel's mean with the floor of 1% added.
        assert!((s.noise() - 0.1 / 1.01).abs() < 0.002, "{}", s.noise());
        // The mean absolute difference of two normal values is 2 sigma / sqrt(pi).
        let expected = 2.0 * 0.1 / std::f64::consts::PI.sqrt();
        assert!((s.flicker() - expected).abs() < 0.005, "{}", s.flicker());
    }

    #[test]
    fn welch_finds_a_small_bias_and_passes_equal_images() {
        let reference = white(64, 1.0, 0.5, 0, 2);
        let same = welch(&white(64, 1.0, 0.5, 0, 3), &reference, 1e-3);
        assert!(same.flagged <= 1, "{same:?}");
        // 4% is seven standard errors of the difference of two tile means here.
        let biased = welch(&white(64, 1.04, 0.5, 0, 4), &reference, 1e-3);
        assert!(
            biased.flagged * 10 > biased.tiles * 9,
            "{} of {}",
            biased.flagged,
            biased.tiles
        );
        assert!((biased.mean_difference - 0.04).abs() < 0.005);
    }

    /// Frames that blend in earlier ones have the same mean but far less spread between frames
    /// than their spread over time suggests: without the correction they fail the test.
    #[test]
    fn welch_allows_for_frames_that_remember_earlier_ones() {
        let series = |seed| {
            let mut rng = Rng(seed);
            let mut s = Series::new(SIZE, SIZE, 0);
            let mut state = vec![0.0f64; (SIZE * SIZE) as usize];
            for frame in 0..256 {
                let tiles = (SIZE / TILE) * (SIZE / TILE);
                let shared: Vec<f64> = (0..tiles).map(|_| rng.normal()).collect();
                let blend = if frame == 0 { 1.0 } else { 0.1 };
                let mut values = Vec::with_capacity(state.len());
                for (i, v) in state.iter_mut().enumerate() {
                    let (x, y) = (i as u32 % SIZE, i as u32 / SIZE);
                    let tile = (y / TILE) * (SIZE / TILE) + x / TILE;
                    *v += blend * (0.2 * shared[tile as usize] - *v);
                    values.push((1.0 + *v) as f32);
                }
                s.add(values);
            }
            s
        };
        let bias = welch(&series(5), &series(6), 1e-3);
        assert!(bias.flagged <= 1, "{} of {}", bias.flagged, bias.tiles);
    }

    #[test]
    fn correlation_finds_noise_shared_by_neighbors_and_frames() {
        let white_noise = white(32, 1.0, 0.2, 32, 7);
        assert!(white_noise.spatial_correlation(1).abs() < 0.01);
        assert!(white_noise.temporal_correlation(1).abs() < 0.05);

        // Rows of blocks of eight pixels sharing their noise.
        let mut rng = Rng(8);
        let mut blocks = Series::new(SIZE, SIZE, 32);
        for _ in 0..32 {
            let shared: Vec<f64> = (0..SIZE * SIZE / 8).map(|_| rng.normal()).collect();
            blocks.add(
                (0..SIZE * SIZE)
                    .map(|i| (1.0 + 0.2 * shared[(i / 8) as usize]) as f32)
                    .collect(),
            );
        }
        // Of eight pixels, seven have a right neighbor in their own block.
        assert!((blocks.spatial_correlation(1) - 7.0 / 8.0).abs() < 0.02);
        assert!(blocks.spatial_correlation(8).abs() < 0.02);

        let mut rng = Rng(9);
        let mut memory = Series::new(SIZE, SIZE, 64);
        let mut state: Vec<f64> = (0..SIZE * SIZE).map(|_| rng.normal()).collect();
        for _ in 0..64 {
            for v in &mut state {
                *v = 0.8 * *v + 0.6 * rng.normal();
            }
            memory.add(state.iter().map(|&v| (5.0 + v) as f32).collect());
        }
        // Subtracting the sample mean of 64 frames lowers it slightly.
        assert!((memory.temporal_correlation(1) - 0.8).abs() < 0.08);
    }

    #[test]
    fn step_response_counts_the_frames_a_blend_takes() {
        // After k frames a blend of 0.2 per frame leaves 2 x 0.8^k of the change of 2, and the
        // mean of eight frames from then on 0.53 of that: within 0.2 from k = 7 on.
        let mut s = Series::new(SIZE, SIZE, 0);
        let step = 16;
        let mut v = 1.0f32;
        for frame in 0..96 {
            if frame >= step {
                v += 0.2 * (3.0 - v);
            }
            s.add(vec![v; (SIZE * SIZE) as usize]);
        }
        let frames = s.step_response(step);
        assert_eq!(frames.len(), 64);
        assert!(frames.iter().all(|&f| f == 7), "{frames:?}");
    }

    #[test]
    fn relative_error_reads_a_difference_and_caps_fireflies() {
        let reference = vec![2.0; 1000];
        let mut frame = vec![2.2; 1000];
        assert!((relative_error(&frame, &reference) - 0.1 / 1.01).abs() < 1e-4);
        frame[0] = 1e6;
        assert!(relative_error(&frame, &reference) < 0.1);
    }
}
