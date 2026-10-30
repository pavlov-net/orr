//! orr-lighting's real-time BSDF on the GPU, reading the tables from `BsdfTables`: it agrees
//! with the same Slang on the CPU within 1e-3 on every call, passes the harness gates against
//! adobe/openpbr-bsdf, and stays finite on edge cases. Runs with validation on and fails if
//! the layer reports anything.

use lighting_maths::gpu::bsdf_cases::{BsdfCase, CaseSample};
use lighting_maths::{
    MAX_MATERIALS, RealTime, RealTimeGpu, edge_case, edge_direction, finite_non_negative,
    subsurface_volume,
};
use openpbr_ref::harness::{Errors, Estimate};
use openpbr_ref::materials::{VIEW_COSINES, grid, view};
use openpbr_ref::report::{Counts, SAME_MODEL, evaluate};
use openpbr_ref::{Bsdf, Inputs, Lobe, Reference, Sample};
use shader_maths::stats::Rng;

/// Directional albedo accuracy the plan asks of the approximation, absolute.
const ALBEDO: f64 = 0.02;

fn assert_clean() {
    assert_eq!(
        RealTimeGpu::device().validation_messages(),
        0,
        "validation messages (see the log)"
    );
}

/// Relative error of `a` against `b` in every channel, as the harness measures it.
fn errors(e: &mut Errors, a: [f32; 3], b: [f32; 3]) {
    for c in 0..3 {
        e.add(f64::from(a[c]), f64::from(b[c]));
    }
}

/// Cosines below which a sampled direction's z is too ill-conditioned in f32 for its weight
/// to be compared across two builds: 1e-5 apart in z is 1% apart in weight at 1e-3.
const GRAZING: f32 = 0.01;

/// Samples whose lobe (or presence) differs, and samples of the same lobe whose directions
/// differ by more than 1e-3: deep in a lobe's tail the inverse CDF is ill-conditioned, so
/// rounding moves a few directions further (1 in 100,000 allowed).
#[derive(Default)]
struct Mismatches {
    lobe: usize,
    direction: usize,
}

/// Compares a GPU and a CPU sample for the same random numbers. A glossy or diffuse
/// sample's weight and density must equal `cpu`'s eval over pdf at the GPU's direction, so
/// the comparison does not depend on how two builds round a direction near the plane; delta
/// and walk samples compare weights directly, walk samples only away from the plane.
fn compare_sample(
    g: Option<Sample>,
    c: Option<Sample>,
    cpu: &RealTime,
    weight: &mut Errors,
    mismatches: &mut Mismatches,
) {
    match (g, c) {
        (None, None) => {}
        (Some(g), Some(c)) if g.lobe == c.lobe => {
            if g.lobe.is(Lobe::SPECULAR) {
                errors(weight, g.weight, c.weight);
                weight.add(f64::from(g.pdf), f64::from(c.pdf));
            } else if g.lobe.is(Lobe::VOLUME) {
                if g.wi[2].abs() >= GRAZING {
                    errors(weight, g.weight, c.weight);
                    weight.add(f64::from(g.pdf), f64::from(c.pdf));
                }
            } else {
                let (e, p) = (cpu.eval(g.wi).total(), cpu.pdf(g.wi));
                errors(weight, g.weight, e.map(|x| x / p));
                weight.add(f64::from(g.pdf), f64::from(p));
            }
            if (0..3).any(|k| (g.wi[k] - c.wi[k]).abs() >= 1e-3) {
                mismatches.direction += 1;
            }
        }
        _ => mismatches.lobe += 1,
    }
}

#[test]
fn gpu_matches_cpu() {
    const CASES: usize = 4_000;
    let materials: Vec<(&str, Inputs)> = grid()
        .iter()
        .flat_map(|m| {
            let volume_and_emission = Inputs {
                emission_luminance: 100.0,
                transmission_depth: 0.5,
                transmission_scatter: [0.2, 0.3, 0.4],
                ..m.inputs
            };
            [(m.name, m.inputs), (m.name, volume_and_emission)]
        })
        .collect();
    let inputs: Vec<Inputs> = materials.iter().map(|m| m.1).collect();
    let mut rng = Rng::new(5);
    let mut cases = Vec::new();
    for material in 0..materials.len() as u32 {
        for collapse in 0..2 {
            for mu in VIEW_COSINES {
                for _ in 0..CASES {
                    cases.push(BsdfCase {
                        material,
                        collapse,
                        view: view(mu),
                        wi: rng.unit(),
                        u: [rng.f32(), rng.f32(), rng.f32()],
                    });
                }
            }
        }
    }
    let results = RealTimeGpu::run(&inputs, &cases);

    let mut failures = Vec::new();
    let mut mismatches = Mismatches::default();
    let (mut cpu, mut collapsed) = (RealTime::new(), RealTime::collapsed());
    for (group, (cases, results)) in cases.chunks(CASES).zip(results.chunks(CASES)).enumerate() {
        let c0 = cases[0];
        let (name, m) = materials[c0.material as usize];
        let bsdf = if c0.collapse == 0 {
            &mut cpu
        } else {
            &mut collapsed
        };
        bsdf.prepare(&m, c0.view);
        let (mut e, mut p, mut s) = (Errors::default(), Errors::default(), Errors::default());
        for (c, r) in cases.iter().zip(results) {
            let (g, k) = (r.split_eval(), bsdf.eval(c.wi));
            errors(&mut e, g.diffuse, k.diffuse);
            errors(&mut e, g.total(), k.total());
            p.add(f64::from(r.pdf), f64::from(bsdf.pdf(c.wi)));
            let (gs, cs) = (r.sample.to_sample(), bsdf.sample(c.u));
            compare_sample(gs, cs, bsdf, &mut s, &mut mismatches);
            let (ge, ce) = (r.entry.to_sample(), bsdf.subsurface_entry(c.u));
            compare_sample(ge, ce, bsdf, &mut s, &mut mismatches);
        }
        let r = results[0];
        let mut once = Errors::default();
        let (ga, ca) = (r.albedo(), bsdf.albedo());
        for (g, k) in [
            (ga.diffuse, ca.diffuse),
            (ga.specular, ca.specular),
            (ga.transmission, ca.transmission),
            (ga.subsurface, ca.subsurface),
            (r.emission, bsdf.emission()),
        ] {
            errors(&mut once, g, k);
        }
        let (vg, vc) = (r.volume(), subsurface_volume(&m));
        errors(&mut once, vg.extinction, vc.extinction);
        errors(&mut once, vg.albedo, vc.albedo);
        once.add(f64::from(vg.anisotropy), f64::from(vc.anisotropy));
        let worst = [e.max, p.max, s.max, once.max];
        if worst.iter().any(|&x| x > SAME_MODEL) {
            failures.push(format!(
                "{name} ({group}), collapse {}, view {:?}: eval {:.2e}, pdf {:.2e}, samples {:.2e}, albedo, emission and volume {:.2e}",
                c0.collapse, c0.view, e.max, p.max, s.max, once.max
            ));
        }
    }
    let samples = 2 * cases.len();
    if mismatches.lobe > 0 || mismatches.direction * 100_000 > samples {
        failures.push(format!(
            "of {samples} samples, {} differ in lobe and {} in direction",
            mismatches.lobe, mismatches.direction
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_clean();
}

/// `a` and `b` agree within `tolerance` plus 3 combined standard errors in every channel.
fn agree(a: &Estimate, b: &Estimate, tolerance: f64) -> bool {
    (0..3).all(|c| {
        let se = (a.se[c] * a.se[c] + b.se[c] * b.se[c]).sqrt();
        (a.mean[c] - b.mean[c]).abs() <= tolerance + 3.0 * se
    })
}

#[test]
fn gpu_passes_the_harness_gates_with_the_reference_albedo() {
    let results = evaluate(
        "orr-lighting bsdf (GPU)",
        RealTimeGpu::new,
        Reference::new,
        false,
        Counts::default(),
    );
    let mut failures = results.failures();
    for m in &results.materials {
        for ((a, r), mu) in m.albedo.iter().zip(&m.reference_albedo).zip(VIEW_COSINES) {
            if !agree(&a.total, &r.total, ALBEDO) {
                failures.push(format!(
                    "{}, mu {mu}: albedo {:?} against {:?}",
                    m.material.name, a.total.mean, r.total.mean
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}\n\n{}",
        failures.join("\n"),
        results.markdown()
    );
    assert_clean();
}

/// As the CPU suite's sweep: random parameters at their limits, views down to the surface
/// plane and below it, light directions in it.
#[test]
fn gpu_finite_for_any_parameters_and_directions() {
    let mut rng = Rng::new(11);
    let (mut materials, mut cases) = (Vec::new(), Vec::new());
    for i in 0..MAX_MATERIALS {
        let (inputs, view) = edge_case(&mut rng, i);
        materials.push(inputs);
        for _ in 0..8 {
            cases.push(BsdfCase {
                material: i as u32,
                collapse: u32::from(rng.next_u32().is_multiple_of(4)),
                view,
                wi: edge_direction(&mut rng),
                u: [rng.f32(), rng.f32(), rng.f32()],
            });
        }
    }
    let ok3 = finite_non_negative;
    let results = RealTimeGpu::run(&materials, &cases);
    for (c, r) in cases.iter().zip(&results) {
        let finite_sample = |s: &CaseSample| {
            s.pdf == 0.0 || (ok3(s.weight) && s.pdf.is_finite() && ok3(s.wi.map(f32::abs)))
        };
        let e = r.split_eval();
        let a = r.albedo();
        assert!(
            ok3(e.diffuse)
                && ok3(e.specular)
                && r.pdf.is_finite()
                && r.pdf >= 0.0
                && finite_sample(&r.sample)
                && finite_sample(&r.entry)
                && ok3(a.diffuse)
                && ok3(a.specular)
                && ok3(a.transmission)
                && ok3(a.subsurface)
                && ok3(r.emission),
            "{c:?} of {:?}: {r:?}",
            materials[c.material as usize]
        );
    }
    assert_clean();
}
