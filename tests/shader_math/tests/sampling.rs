use orr_math::Vec3;
use shader_math::stats::{Rng, SphereGrid, chi_square_p};
use shader_math::*;

const SAMPLES: usize = 1_000_000;

/// Checks a direction sampler against its density over the sphere: the density integrates
/// to `mass` (1, or less when samples can fall where it is not defined), and the histogram
/// of samples matches it (chi-square p > 0.01). Samples outside `defined` are counted in
/// one cell that expects the rest of the mass.
fn check_directions(
    name: &str,
    mass: Option<f64>,
    defined: impl Fn([f32; 3]) -> bool,
    sample: impl Fn([f32; 2]) -> [f32; 3],
    pdf: impl Fn([f32; 3]) -> f64,
) {
    let grid = SphereGrid { nz: 64, nphi: 128 };
    let probs = grid.integrate(8, |d| if defined(d) { pdf(d) } else { 0.0 });
    let total: f64 = probs.iter().sum();
    if let Some(mass) = mass {
        assert!(
            (total - mass).abs() < 2e-3,
            "{name}: density integrates to {total}"
        );
    }
    assert!(total < 1.0 + 2e-3, "{name}: density integrates to {total}");
    let mut rng = Rng::new(30);
    let mut observed = vec![0.0; grid.cells() + 1];
    for _ in 0..SAMPLES {
        let d = sample([rng.f32(), rng.f32()]);
        let len = Vec3::from(d).length();
        assert!(
            (len - 1.0).abs() < 1e-4,
            "{name}: sample {d:?} is not unit length"
        );
        if defined(d) {
            observed[grid.cell(d)] += 1.0;
        } else {
            observed[grid.cells()] += 1.0;
        }
    }
    let mut expected: Vec<f64> = probs.iter().map(|p| p * SAMPLES as f64).collect();
    expected.push((1.0 - total).max(0.0) * SAMPLES as f64);
    let p = chi_square_p(&observed, &expected);
    assert!(p > 0.01, "{name}: chi-square p = {p}");
}

#[test]
fn hemisphere_sphere_and_cone_match_their_densities() {
    check_directions(
        "cosine hemisphere",
        Some(1.0),
        |_| true,
        sample_cosine_hemisphere,
        |d| f64::from(cosine_hemisphere_pdf(d[2])),
    );
    check_directions(
        "uniform hemisphere",
        Some(1.0),
        |_| true,
        sample_uniform_hemisphere,
        |d| {
            if d[2] > 0.0 {
                1.0 / std::f64::consts::TAU
            } else {
                0.0
            }
        },
    );
    check_directions(
        "uniform sphere",
        Some(1.0),
        |_| true,
        sample_uniform_sphere,
        |_| 1.0 / (2.0 * std::f64::consts::TAU),
    );
    // 0.75 is a band edge of the grid, so the cone's rim does not cut a cell.
    check_directions(
        "cone",
        Some(1.0),
        |_| true,
        |u| sample_cone(u, 0.25),
        |d| {
            if d[2] >= 0.75 {
                f64::from(cone_pdf(0.25))
            } else {
                0.0
            }
        },
    );
}

#[test]
fn bounded_ggx_vndf_matches_its_density() {
    for (v, alpha) in [
        ([0.0, 0.0, 1.0], [0.3, 0.3]),
        ([0.6, 0.0, 0.8], [0.6, 0.6]),
        ([0.3, -0.4, 0.866_025_4], [0.1, 0.5]),
        ([0.998_749_2, 0.0, 0.05], [0.4, 0.4]),
        ([0.0, 0.6, 0.8], [1.0, 1.0]),
    ] {
        let name = format!("ggx v {v:?} alpha {alpha:?}");
        check_directions(
            &name,
            None,
            |l| l[2] > 0.0,
            |u| sample_ggx_vndf_bounded(v, alpha, u),
            |l| f64::from(ggx_vndf_bounded_pdf(v, l, alpha)),
        );
    }
}

#[test]
fn balance_heuristic_is_zero_where_its_technique_has_no_density() {
    assert_eq!(mis_balance(0.0, 0.0), 0.0);
    assert_eq!(mis_balance(0.0, 3.0), 0.0);
    assert_eq!(mis_balance(3.0, 0.0), 1.0);
    assert_eq!(mis_balance(3.0, 1.0), 0.75);
}
