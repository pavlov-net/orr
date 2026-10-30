use glam::DVec3;
use orr_shaders::AliasEntry;
use shader_maths::stats::{Rng, SphereGrid, chi_square_p};
use shader_maths::*;

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
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
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
                f64::from(uniform_hemisphere_pdf())
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
        |_| f64::from(uniform_sphere_pdf()),
    );
    // 0.75 is a band edge of the grid, so the cone's rim does not cut a cell.
    check_directions(
        "cone",
        Some(1.0),
        |_| true,
        |u| sample_cone(u, 0.75),
        |d| {
            if d[2] >= 0.75 {
                f64::from(cone_pdf(0.75))
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

/// Checks samples on a planar shape parameterised by (s, t) in [0, 1]^2 against a solid
/// angle density of `1 / solid_angle`: maps each sample to (s, t), and compares the
/// histogram with the density carried to (s, t) by `jacobian`, the area per unit (s, t) at
/// a point. Returns the solid angle measured by integrating over the shape.
fn check_planar(
    name: &str,
    origin: DVec3,
    normal: DVec3,
    point: impl Fn(f64, f64) -> DVec3,
    jacobian: impl Fn(f64, f64) -> f64,
    sample_st: impl Fn([f32; 2]) -> (f64, f64),
    solid_angle: f64,
) {
    let res = 32;
    let sub = 16;
    let mut probs = vec![0.0; res * res];
    let mut measured = 0.0;
    for i in 0..res * sub {
        for j in 0..res * sub {
            let (s, t) = (
                (i as f64 + 0.5) / (res * sub) as f64,
                (j as f64 + 0.5) / (res * sub) as f64,
            );
            let p = point(s, t);
            let to = origin - p;
            let d2 = to.length_squared();
            let w = normal.dot(to).abs() / (d2 * d2.sqrt()) * jacobian(s, t)
                / ((res * sub) as f64).powi(2);
            measured += w;
            probs[(i / sub) * res + j / sub] += w;
        }
    }
    // f32 solid angles of small shapes lose about 1e-6 to cancellation.
    assert!(
        (measured - solid_angle).abs() < 1e-4 * solid_angle + 1e-6,
        "{name}: solid angle {solid_angle}, measured {measured}"
    );
    let mut rng = Rng::new(31);
    let mut observed = vec![0.0; res * res];
    for _ in 0..SAMPLES {
        let (s, t) = sample_st([rng.f32(), rng.f32()]);
        assert!(
            (-1e-4..=1.0 + 1e-4).contains(&s) && (-1e-4..=1.0 + 1e-4).contains(&t),
            "{name}: sample off the shape at {s}, {t}"
        );
        let cell = |x: f64| ((x * res as f64) as usize).min(res - 1);
        observed[cell(s) * res + cell(t)] += 1.0;
    }
    let expected: Vec<f64> = probs
        .iter()
        .map(|p| p / measured * SAMPLES as f64)
        .collect();
    let p = chi_square_p(&observed, &expected);
    assert!(p > 0.01, "{name}: chi-square p = {p}");
}

#[test]
fn spherical_rectangle_is_uniform_in_solid_angle() {
    for (o, corner, ex, ey) in [
        (
            [0.0, 0.0, 0.0],
            [-0.3, 0.2, 1.0],
            [1.5, 0.0, 0.4],
            [0.0, 0.8, 0.0],
        ),
        (
            [0.2, -0.3, 0.1],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ),
        (
            [3.0, 1.0, -2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.3, 0.0, 0.0],
        ),
    ] {
        let (od, cd, exd, eyd) = (
            DVec3::from(o.map(f64::from)),
            DVec3::from(corner.map(f64::from)),
            DVec3::from(ex.map(f64::from)),
            DVec3::from(ey.map(f64::from)),
        );
        let solid_angle = f64::from(sample_spherical_rectangle(o, corner, ex, ey, [0.5, 0.5]).1);
        check_planar(
            &format!("rectangle {o:?} {corner:?}"),
            od,
            exd.cross(eyd).normalize(),
            |s, t| cd + s * exd + t * eyd,
            |_, _| exd.cross(eyd).length(),
            |u| {
                let p = DVec3::from(
                    sample_spherical_rectangle(o, corner, ex, ey, u)
                        .0
                        .map(f64::from),
                ) - cd;
                (
                    p.dot(exd) / exd.length_squared(),
                    p.dot(eyd) / eyd.length_squared(),
                )
            },
            solid_angle,
        );
    }
}

#[test]
fn spherical_triangle_is_uniform_in_solid_angle() {
    for (o, v) in [
        (
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, -0.5, 1.5]],
        ),
        (
            [0.1, 0.1, 0.05],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ),
        (
            [-2.0, 0.5, 1.0],
            [[0.0, -1.0, 0.0], [0.2, 1.0, 0.1], [0.0, 0.0, 0.7]],
        ),
    ] {
        let od = DVec3::from(o.map(f64::from));
        let [a, b, c] = v.map(|p| DVec3::from(p.map(f64::from)));
        let n = (b - a).cross(c - a);
        let solid_angle = f64::from(sample_spherical_triangle(o, v, [0.5, 0.5]).1);
        check_planar(
            &format!("triangle {o:?}"),
            od,
            n.normalize(),
            // (1 - s) a + s ((1 - t) b + t c): area per unit (s, t) is 2 x area x s.
            |s, t| (1.0 - s) * a + s * ((1.0 - t) * b + t * c),
            |s, _| n.length() * s,
            |u| {
                let d = DVec3::from(sample_spherical_triangle(o, v, u).0.map(f64::from));
                // Ray-plane hit, then barycentrics.
                let p = od + d * (a - od).dot(n) / d.dot(n);
                let (e1, e2, e) = (b - a, c - a, p - a);
                let (d11, d12, d22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
                let (d1, d2) = (e.dot(e1), e.dot(e2));
                let den = d11 * d22 - d12 * d12;
                let (b1, b2) = ((d22 * d1 - d12 * d2) / den, (d11 * d2 - d12 * d1) / den);
                let s = b1 + b2;
                (s, if s > 0.0 { b2 / s } else { 0.0 })
            },
            solid_angle,
        );
    }
}

#[test]
fn alias_table_draws_with_its_weights() {
    let mut rng = Rng::new(32);
    let mut weights: Vec<f32> = (0..37).map(|_| rng.f32()).collect();
    weights[3] = 0.0;
    weights[20] = 0.0;
    weights[7] = 30.0;
    for weights in [weights, vec![0.0; 5], vec![2.0]] {
        let table = AliasEntry::table(&weights);
        let mut observed = vec![0.0; weights.len()];
        for _ in 0..SAMPLES {
            observed[sample_alias(&table, rng.f32()) as usize] += 1.0;
        }
        let sum: f32 = weights.iter().sum();
        let expected: Vec<f64> = weights
            .iter()
            .map(|&w| if sum > 0.0 { f64::from(w / sum) } else { 1.0 / weights.len() as f64 } * SAMPLES as f64)
            .collect();
        for (i, (&o, &e)) in observed.iter().zip(&expected).enumerate() {
            assert!(e > 0.0 || o == 0.0, "zero-weight index {i} drawn {o} times");
        }
        if weights.len() > 1 {
            let p = chi_square_p(&observed, &expected);
            assert!(p > 0.01, "{weights:?}: chi-square p = {p}");
        }
    }
}
