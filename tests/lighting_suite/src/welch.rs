//! Welch's t-test on independent estimates of a mean, such as the frames of two renders.

/// |t| past which two means differ: about 1 in 10^4 by chance at 30 degrees of freedom.
const T_LIMIT: f64 = 4.0;

/// The mean of `samples` and the variance of that mean.
fn mean(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    assert!(samples.len() > 1, "a variance needs two samples");
    let mean = samples.iter().sum::<f64>() / n;
    let square = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>();
    (mean, square / (n - 1.0) / n)
}

/// Welch's t of samples `a` and `b`, which need not share a variance or a count.
pub fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    let (a, b) = (mean(a), mean(b));
    (a.0 - b.0) / (a.1 + b.1).sqrt()
}

/// Panics unless samples `a` and `b` have the same mean by Welch's test.
pub fn assert_same_mean(what: &str, a: &[f64], b: &[f64]) {
    let t = welch_t(a, b);
    assert!(
        t.abs() < T_LIMIT,
        "{what}: t {t}, means {} and {}",
        mean(a).0,
        mean(b).0
    );
}

/// Panics unless `samples` have mean `expected`, known to a relative `tolerance` that also covers
/// samples with no noise of their own.
pub fn assert_mean(what: &str, samples: &[f64], expected: f64, tolerance: f64) {
    let (m, variance) = mean(samples);
    let t = (m - expected) / (variance + (tolerance * expected).powi(2)).sqrt();
    assert!(
        t.abs() < T_LIMIT,
        "{what}: t {t}, mean {m}, expected {expected}"
    );
}
