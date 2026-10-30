use shader_maths::stats::{Rng, chi_square_p};
use shader_maths::*;

/// Sobol direction numbers from Joe and Kuo's primitive polynomials (new-joe-kuo-6.21201):
/// (degree s, coefficients a, initial m).
fn directions(dim: usize) -> [u32; 32] {
    let params: [(usize, u32, &[u32]); 3] = [(1, 0, &[1]), (2, 1, &[1, 3]), (3, 1, &[1, 3, 1])];
    if dim == 0 {
        return std::array::from_fn(|i| 1 << (31 - i));
    }
    let (s, a, init) = params[dim - 1];
    let mut m = init.to_vec();
    for i in s..32 {
        let mut v = m[i - s] ^ (m[i - s] << s);
        for k in 1..s {
            v ^= ((a >> (s - 1 - k)) & 1) * (m[i - k] << k);
        }
        m.push(v);
    }
    std::array::from_fn(|i| m[i] << (31 - i))
}

#[test]
fn sobol_matches_joe_kuo() {
    let mut rng = Rng::new(20);
    for dim in 0..4 {
        let v = directions(dim);
        for index in (0..4096).chain((0..10_000).map(|_| rng.next_u32())) {
            let want = (0..32)
                .filter(|b| index >> b & 1 != 0)
                .fold(0, |x, b| x ^ v[b]);
            assert_eq!(sobol(index, dim as u32), want, "dim {dim} index {index}");
        }
    }
}

/// Whether `points` (2^m of them) put exactly one point in every elementary interval of
/// area 2^-m: every 2^k x 2^(m-k) grid of cells.
fn is_0m2_net(points: &[[f32; 2]], m: u32) -> bool {
    (0..=m).all(|k| {
        let (nx, ny) = (1usize << k, 1usize << (m - k));
        let mut seen = vec![false; nx * ny];
        points.iter().all(|p| {
            let cell = (p[0] * nx as f32) as usize + nx * (p[1] * ny as f32) as usize;
            !std::mem::replace(&mut seen[cell], true)
        })
    })
}

#[test]
fn owen_scrambled_sobol_keeps_its_stratification() {
    let m = 10;
    for seed in [0u32, 1, 0xDEAD_BEEF, 12345] {
        let points: Vec<[f32; 4]> = (0..1u32 << m).map(|i| owen_sobol4(i, seed)).collect();
        for d in 0..4 {
            let mut strata = vec![false; 1 << m];
            for p in &points {
                let s = (p[d] * (1 << m) as f32) as usize;
                assert!(
                    !std::mem::replace(&mut strata[s], true),
                    "seed {seed} dim {d}"
                );
            }
        }
        let pairs: Vec<[f32; 2]> = points.iter().map(|p| [p[0], p[1]]).collect();
        assert!(is_0m2_net(&pairs, m), "seed {seed}");
    }
}

#[test]
fn sampler_dimensions_are_nets_over_frames_and_independent() {
    let frames = 4096u32;
    let dims = 6;
    for pixel in [[0u32, 0], [17, 3], [1919, 1079]] {
        let per_frame: Vec<Vec<[f32; 2]>> =
            (0..frames).map(|f| sampler_next2(pixel, f, dims)).collect();
        for d in 0..dims {
            let pts: Vec<[f32; 2]> = per_frame.iter().map(|s| s[d]).collect();
            assert!(is_0m2_net(&pts, 12), "pixel {pixel:?} dimension {d}");
        }
        // Consecutive dimensions are scrambled independently.
        for d in 0..dims - 1 {
            let mut observed = vec![0.0; 256];
            for s in &per_frame {
                let (a, b) = (s[d][0], s[d + 1][0]);
                observed[(a * 16.0) as usize * 16 + (b * 16.0) as usize] += 1.0;
            }
            let expected = vec![f64::from(frames) / 256.0; 256];
            let p = chi_square_p(&observed, &expected);
            assert!(
                p > 0.01,
                "pixel {pixel:?} dimensions {d}, {}: p = {p}",
                d + 1
            );
        }
    }
    // One frame over many pixels is uniform.
    let mut observed = vec![0.0; 64];
    let n = 256 * 256;
    for y in 0..256 {
        for x in 0..256 {
            let u = sampler_next2([x, y], 7, 1)[0];
            observed[(u[0] * 8.0) as usize * 8 + (u[1] * 8.0) as usize] += 1.0;
        }
    }
    let p = chi_square_p(&observed, &vec![f64::from(n) / 64.0; 64]);
    assert!(p > 0.01, "pixels: p = {p}");
}
