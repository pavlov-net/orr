//! orr-lighting's real-time BSDF against adobe/openpbr-bsdf (lighting.md L1): the harness
//! gates, directional albedo by kind within 2% over a parameter grid, sampling consistent
//! with eval, the white furnace, collapse, the mirror threshold, the subsurface entry, and
//! finite results everywhere.

use std::thread;

use lighting_maths::{RealTime, edge_case, edge_direction, finite_non_negative, subsurface_volume};
use openpbr_ref::harness::{self, Estimate, integrate_cells};
use openpbr_ref::materials::{VIEW_COSINES, grid, view};
use openpbr_ref::report::{Counts, TABLE_TOLERANCE, evaluate};
use openpbr_ref::{Bsdf, Inputs, Lobe, Reference};
use shader_maths::stats::{Rng, SphereGrid};

/// Directional albedo accuracy the plan asks of the approximation, absolute.
const ALBEDO: f64 = 0.02;

/// `a` and `b` agree within `tolerance` plus 3 combined standard errors in every channel.
fn agree(a: &Estimate, b: &Estimate, tolerance: f64) -> bool {
    (0..3).all(|c| {
        let se = (a.se[c] * a.se[c] + b.se[c] * b.se[c]).sqrt();
        (a.mean[c] - b.mean[c]).abs() <= tolerance + 3.0 * se
    })
}

/// Directional albedo of the diffuse lobes and of the rest: each draw adds eval / pdf split
/// by kind, a delta draw its weight to the rest.
fn albedo_by_kind(
    bsdf: &mut dyn Bsdf,
    inputs: &Inputs,
    view: [f32; 3],
    samples: usize,
    seed: u64,
) -> [Estimate; 2] {
    bsdf.prepare(inputs, view);
    let mut rng = Rng::new(seed);
    let mut sum = [[0.0f64; 3]; 2];
    let mut sq = [[0.0f64; 3]; 2];
    for _ in 0..samples {
        let u = [rng.f32(), rng.f32(), rng.f32()];
        let Some(s) = bsdf.sample(u) else { continue };
        let parts = if s.lobe.is(Lobe::SPECULAR) {
            [[0.0; 3], s.weight]
        } else {
            let e = bsdf.eval(s.wi);
            [e.diffuse, e.specular].map(|v| v.map(|x| x / s.pdf))
        };
        for (k, part) in parts.iter().enumerate() {
            for c in 0..3 {
                let w = f64::from(part[c]);
                sum[k][c] += w;
                sq[k][c] += w * w;
            }
        }
    }
    let n = samples as f64;
    [0, 1].map(|k| {
        let mut e = Estimate::default();
        for c in 0..3 {
            e.mean[c] = sum[k][c] / n;
            e.se[c] = ((sq[k][c] / n - e.mean[c] * e.mean[c]).max(0.0) / (n - 1.0)).sqrt();
        }
        e
    })
}

#[test]
fn passes_the_harness_gates_with_the_reference_albedo() {
    let results = evaluate(
        "orr-lighting bsdf",
        RealTime::new,
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
}

/// Roughness 0..1, IOR 1..2, specular weight 0..1, metal and dielectric, each optional lobe
/// on its own, at every view of the harness: diffuse and non-diffuse albedo within 2%.
#[test]
fn albedo_by_kind_matches_the_reference_over_the_parameter_grid() {
    let base = Inputs {
        base_color: [0.9, 0.6, 0.3],
        ..Inputs::default()
    };
    let features: [(&str, Inputs); 5] = [
        ("plain", base),
        (
            "coat",
            Inputs {
                coat_weight: 1.0,
                coat_roughness: 0.25,
                coat_color: [0.9, 0.8, 0.7],
                ..base
            },
        ),
        (
            "fuzz",
            Inputs {
                fuzz_weight: 1.0,
                fuzz_roughness: 0.6,
                ..base
            },
        ),
        (
            "anisotropic",
            Inputs {
                specular_roughness_anisotropy: 0.6,
                specular_rotation: [0.6, 0.8],
                ..base
            },
        ),
        (
            "thin film",
            Inputs {
                thin_film_weight: 1.0,
                thin_film_thickness: 0.4,
                ..base
            },
        ),
    ];
    let mut cases = Vec::new();
    for (name, f) in features {
        for metal in [0.0, 1.0] {
            let iors: &[f32] = if metal > 0.0 {
                &[1.5]
            } else {
                &[1.0, 1.25, 1.5, 2.0]
            };
            for &ior in iors {
                for weight in [0.0, 0.5, 1.0] {
                    for roughness in [0.0, 0.05, 0.3, 0.6, 1.0] {
                        let inputs = Inputs {
                            base_metalness: metal,
                            specular_ior: ior,
                            specular_weight: weight,
                            specular_roughness: roughness,
                            ..f
                        };
                        for mu in VIEW_COSINES {
                            cases.push((name, inputs, mu));
                        }
                    }
                }
            }
        }
    }
    let threads = thread::available_parallelism().map_or(4, usize::from);
    let failures: Vec<String> = thread::scope(|scope| {
        let handles: Vec<_> = cases
            .chunks(cases.len().div_ceil(threads))
            .enumerate()
            .map(|(t, chunk)| {
                scope.spawn(move || {
                    let (mut c, mut r) = (RealTime::new(), Reference::new());
                    let mut out = Vec::new();
                    for (i, (name, inputs, mu)) in chunk.iter().enumerate() {
                        let seed = ((t as u64) << 32) | i as u64;
                        let a = albedo_by_kind(&mut c, inputs, view(*mu), 20_000, seed);
                        let b = albedo_by_kind(&mut r, inputs, view(*mu), 20_000, seed);
                        for (kind, (a, b)) in ["diffuse", "other"].iter().zip(a.iter().zip(&b)) {
                            if !agree(a, b, ALBEDO) {
                                out.push(format!(
                                    "{name}, metal {}, ior {}, weight {}, roughness {}, mu {mu}: {kind} {:?} against {:?}",
                                    inputs.base_metalness,
                                    inputs.specular_ior,
                                    inputs.specular_weight,
                                    inputs.specular_roughness,
                                    a.mean,
                                    b.mean
                                ));
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("thread panicked"))
            .collect()
    });
    assert!(
        failures.is_empty(),
        "{} of {} cases:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// Albedo from sample / pdf equals eval integrated over the sphere within 1%: every direction
/// eval reaches, sampling reaches with the density pdf reports.
#[test]
fn sampled_albedo_equals_integrated_eval() {
    let mut bsdf = RealTime::new();
    let cells = SphereGrid { nz: 32, nphi: 64 };
    let mut failures = Vec::new();
    for (i, m) in grid().iter().enumerate().filter(|(_, m)| !m.delta) {
        for (v, &mu) in VIEW_COSINES.iter().enumerate() {
            let a = harness::albedo(&mut bsdf, &m.inputs, view(mu), 200_000, (i * 8 + v) as u64);
            let sampled: f64 = a.total.mean.iter().sum();
            let se: f64 = a.total.se.iter().sum();
            let integrated: f64 = integrate_cells(&cells, |d| {
                f64::from(bsdf.eval(d).total().iter().sum::<f32>())
            })
            .iter()
            .sum();
            if (sampled - integrated).abs() > 0.01 * integrated + 3.0 * se {
                failures.push(format!(
                    "{}, mu {mu}: sampled {sampled:.4}, integrated {integrated:.4}",
                    m.name
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// White non-absorbing configurations, metal with F0 = F82 = 1 and a dielectric of IOR 1.5
/// over a white base, reflect between 0.98 and 1, widened to the reference's own albedo
/// where it falls outside (REF-9, REF-11).
#[test]
fn white_furnace() {
    let white = Inputs {
        base_color: [1.0; 3],
        specular_color: [1.0; 3],
        ..Inputs::default()
    };
    let (mut bsdf, mut reference) = (RealTime::new(), Reference::new());
    let mut failures = Vec::new();
    for (name, metal) in [("metal", 1.0), ("dielectric", 0.0)] {
        for roughness in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let inputs = Inputs {
                base_metalness: metal,
                specular_roughness: roughness,
                ..white
            };
            for mu in [0.02, 0.05, 0.3, 0.7, 1.0] {
                let a = harness::albedo(&mut bsdf, &inputs, view(mu), 100_000, 7);
                let r = harness::albedo(&mut reference, &inputs, view(mu), 100_000, 8);
                for c in 0..3 {
                    let (e, ref_e) = (a.total.mean[c], r.total.mean[c]);
                    let se = (a.total.se[c].powi(2) + r.total.se[c].powi(2)).sqrt();
                    let low = 0.98f64.min(ref_e) - 3.0 * se;
                    let high = 1.0f64.max(ref_e) + 3.0 * se + TABLE_TOLERANCE;
                    if e < low || e > high {
                        failures.push(format!(
                            "{name}, roughness {roughness}, mu {mu}: {e:.4} (reference {ref_e:.4})"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}",
        failures.join(
            "
"
        )
    );
}

/// `bsdf_collapse` keeps the directional albedo of the full lobes within 2%.
#[test]
fn collapse_keeps_the_directional_albedo() {
    let coloured = Inputs {
        base_color: [0.5, 0.2, 0.1],
        coat_weight: 1.0,
        coat_roughness: 0.1,
        coat_color: [0.6, 0.4, 0.2],
        fuzz_weight: 0.7,
        fuzz_color: [0.3, 0.5, 0.9],
        ..Inputs::default()
    };
    let materials: Vec<(&str, Inputs)> = grid()
        .into_iter()
        .filter(|m| m.inputs.coat_weight > 0.0 || m.inputs.fuzz_weight > 0.0)
        .map(|m| (m.name, m.inputs))
        .chain([("coloured coat and fuzz", coloured)])
        .collect();
    let (mut full, mut collapsed) = (RealTime::new(), RealTime::collapsed());
    let mut failures = Vec::new();
    for (name, inputs) in &materials {
        for mu in VIEW_COSINES {
            let a = harness::albedo(&mut full, inputs, view(mu), 200_000, 3);
            let b = harness::albedo(&mut collapsed, inputs, view(mu), 200_000, 4);
            if !agree(&a.total, &b.total, ALBEDO) {
                failures.push(format!(
                    "{name}, mu {mu}: full {:?}, collapsed {:?}",
                    a.total.mean, b.total.mean
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Either side of `MIRROR_ROUGHNESS` the albedo is the same; below it the lobes are delta.
#[test]
fn mirror_threshold_is_continuous() {
    let mut bsdf = RealTime::new();
    for (name, inputs) in [
        (
            "metal",
            Inputs {
                base_metalness: 1.0,
                base_color: [0.9, 0.7, 0.5],
                ..Inputs::default()
            },
        ),
        (
            "glass",
            Inputs {
                transmission_weight: 1.0,
                ..Inputs::default()
            },
        ),
    ] {
        for mu in VIEW_COSINES {
            let below = Inputs {
                specular_roughness: 0.0099,
                ..inputs
            };
            let above = Inputs {
                specular_roughness: 0.0101,
                ..inputs
            };
            let a = harness::albedo(&mut bsdf, &below, view(mu), 100_000, 1);
            let b = harness::albedo(&mut bsdf, &above, view(mu), 100_000, 2);
            assert!(
                a.lobes.keys().all(|l| l.is(Lobe::SPECULAR)),
                "{name}, mu {mu}: {:?}",
                a.lobes.keys()
            );
            assert!(
                b.lobes.keys().all(|l| !l.is(Lobe::SPECULAR)),
                "{name}, mu {mu}: {:?}",
                b.lobes.keys()
            );
            assert!(
                agree(&a.total, &b.total, 0.005),
                "{name}, mu {mu}: {:?} against {:?}",
                a.total.mean,
                b.total.mean
            );
        }
    }
}

/// The walk lobe alone carries what the reference transmits into a subsurface material.
#[test]
fn subsurface_entry_carries_the_reference_transmission() {
    let material = grid().into_iter().find(|m| m.name == "subsurface").unwrap();
    let mut bsdf = RealTime::new();
    let mut reference = Reference::new();
    for (v, mu) in VIEW_COSINES.into_iter().enumerate() {
        let r = harness::albedo(
            &mut reference,
            &material.inputs,
            view(mu),
            200_000,
            v as u64,
        );
        let transmitted = r
            .lobes
            .iter()
            .filter(|(l, _)| l.is(Lobe::TRANSMISSION))
            .fold(Estimate::default(), |mut acc, (_, e)| {
                for c in 0..3 {
                    acc.mean[c] += e.mean[c];
                    acc.se[c] += e.se[c];
                }
                acc
            });
        bsdf.prepare(&material.inputs, view(mu));
        let mut rng = Rng::new(v as u64);
        let n = 200_000u32;
        let mut sum = [0.0f64; 3];
        for _ in 0..n {
            let Some(s) = bsdf.subsurface_entry([rng.f32(), rng.f32(), rng.f32()]) else {
                continue;
            };
            assert!(s.lobe.is(Lobe::VOLUME) && s.wi[2] < 0.0, "{s:?}");
            for (total, w) in sum.iter_mut().zip(s.weight) {
                *total += f64::from(w);
            }
        }
        for (c, total) in sum.iter().enumerate() {
            let entry = total / f64::from(n);
            assert!(
                (entry - transmitted.mean[c]).abs() <= ALBEDO + 3.0 * transmitted.se[c],
                "mu {mu}: entry {entry:.4}, reference {:.4}",
                transmitted.mean[c]
            );
        }
    }
}

#[test]
fn emission_and_subsurface_volume_match_the_reference() {
    let (mut bsdf, mut reference) = (RealTime::new(), Reference::new());
    let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * b.abs().max(1e-3);
    for m in grid() {
        let inputs = Inputs {
            emission_luminance: 100.0,
            ..m.inputs
        };
        for mu in VIEW_COSINES {
            bsdf.prepare(&inputs, view(mu));
            reference.prepare(&inputs, view(mu));
            let (a, b) = (bsdf.emission(), reference.emission());
            assert!(
                (0..3).all(|c| close(a[c], b[c])),
                "{}: {a:?} vs {b:?}",
                m.name
            );
        }
        if inputs.subsurface_weight > 0.0 && inputs.geometry_thin_walled == 0 {
            let (a, b) = (subsurface_volume(&inputs), reference.volume());
            for c in 0..3 {
                assert!(close(a.extinction[c], b.extinction[c]), "{a:?} vs {b:?}");
                assert!(close(a.albedo[c], b.albedo[c]), "{a:?} vs {b:?}");
            }
            assert!(close(a.anisotropy, b.anisotropy), "{a:?} vs {b:?}");
        }
    }
}

/// Random parameters at their limits, views down to the surface plane and below it: eval,
/// pdf, samples and estimates stay finite and non-negative.
#[test]
fn finite_for_any_parameters_and_directions() {
    let ok3 = finite_non_negative;
    let mut rng = Rng::new(11);
    let mut bsdf = RealTime::new();
    for case in 0..100_000 {
        let (inputs, view_dir) = edge_case(&mut rng, case);
        bsdf.prepare(&inputs, view_dir);
        let a = bsdf.albedo();
        assert!(
            ok3(a.diffuse) && ok3(a.specular) && ok3(a.transmission) && ok3(a.subsurface),
            "case {case}: {a:?} for {inputs:?} from {view_dir:?}"
        );
        assert!(ok3(bsdf.emission()), "case {case}");
        for _ in 0..8 {
            let wi = edge_direction(&mut rng);
            let e = bsdf.eval(wi);
            let p = bsdf.pdf(wi);
            assert!(
                ok3(e.diffuse) && ok3(e.specular) && p.is_finite() && p >= 0.0,
                "case {case}: eval {e:?} pdf {p} toward {wi:?} for {inputs:?} from {view_dir:?}"
            );
            let u = [rng.f32(), rng.f32(), rng.f32()];
            for s in [bsdf.sample(u), bsdf.subsurface_entry(u)]
                .into_iter()
                .flatten()
            {
                assert!(
                    ok3(s.weight) && s.pdf.is_finite() && ok3(s.wi.map(f32::abs)),
                    "case {case}: {s:?} for {inputs:?} from {view_dir:?}"
                );
            }
        }
    }
}
