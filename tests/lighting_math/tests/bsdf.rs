//! orr_lighting's real-time BSDF against adobe/openpbr-bsdf (lighting.md L1): the harness
//! gates, directional albedo by kind within 2% over a parameter grid, sampling consistent
//! with eval, the white furnace, collapse, the mirror threshold, the subsurface entry, and
//! finite results everywhere.

use std::thread;

use lighting_math::{
    RealTime, camera_weight, dispersed_ior, edge_case, edge_direction, finite_non_negative,
    spectral_transfer, subsurface_volume,
};
use openpbr_ref::harness::{self, Estimate, integrate_cells};
use openpbr_ref::materials::{Material, VIEW_COSINES, grid, view};
use openpbr_ref::report::{Counts, TABLE_TOLERANCE, evaluate};
use openpbr_ref::walk::fresnel;
use openpbr_ref::{Bsdf, Inputs, Lobe, Reference};
use orr_math::Vec3;
use shader_math::stats::{Rng, SphereGrid};

/// Directional albedo accuracy the plan asks of the approximation, absolute.
const ALBEDO: f64 = 0.02;

/// `BSDF_FRESNEL_MIX` and `BSDF_METAL_FLAT`.
const FRESNEL_MIX: u32 = 1 << 7;
const METAL_FLAT: u32 = 1 << 8;

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
        "orr_lighting bsdf",
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
/// eval reaches, sampling reaches with the density pdf reports. Collapsed too, where the
/// coat and fuzz fold onto the specular and cosine lobes.
#[test]
fn sampled_albedo_equals_integrated_eval() {
    let cells = SphereGrid { nz: 32, nphi: 64 };
    let mut failures = Vec::new();
    let materials = grid();
    let layered = |m: &&Material| m.inputs.coat_weight > 0.0 || m.inputs.fuzz_weight > 0.0;
    let cases = materials.iter().enumerate().filter(|(_, m)| !m.delta);
    for (collapse, (i, m)) in cases
        .clone()
        .map(|c| (false, c))
        .chain(cases.filter(|(_, m)| layered(m)).map(|c| (true, c)))
    {
        let (mut bsdf, label) = if collapse {
            (RealTime::collapsed(), " collapsed")
        } else {
            (RealTime::new(), "")
        };
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
                    "{}{label}, mu {mu}: sampled {sampled:.4}, integrated {integrated:.4}",
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
    let colored = Inputs {
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
        .chain([("colored coat and fuzz", colored)])
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

/// OpenPBR's metal table ranges `specular_weight` over [0, 1], where the dielectric's goes past
/// it: a metal at 1.5 is the metal at 1.
/// A coat on a normal and anisotropy of its own, around its highlight, against the reference's
/// second basis. One frame for every layer misses a tilted coat's peak by a factor of 5.
#[test]
fn a_coat_keeps_its_own_frame() {
    let normal = Vec3::new(0.35, 0.2, 1.0).normalize();
    let view = Vec3::new(0.5, -0.3, 0.8).normalize();
    let coat = Inputs {
        coat_weight: 1.0,
        coat_roughness: 0.3,
        specular_roughness: 0.6,
        ..Inputs::default()
    };
    let cases = [
        Inputs {
            coat_normal: normal.to_array(),
            ..coat
        },
        Inputs {
            coat_roughness_anisotropy: 0.8,
            coat_rotation: [0.6, 0.8],
            ..coat
        },
        Inputs {
            coat_normal: normal.to_array(),
            fuzz_weight: 1.0,
            fuzz_roughness: 0.4,
            ..coat
        },
    ];
    let mirror = 2.0 * normal.dot(view) * normal - view;
    let lights = [
        mirror,
        (mirror + Vec3::new(0.2, 0.0, 0.0)).normalize(),
        (mirror + Vec3::new(0.0, 0.25, 0.0)).normalize(),
        (mirror + Vec3::new(-0.3, -0.1, 0.0)).normalize(),
    ];
    let (mut ours, mut reference) = (RealTime::new(), Reference::new());
    for (i, inputs) in cases.iter().enumerate() {
        ours.prepare(inputs, view.to_array());
        reference.prepare(inputs, view.to_array());
        for light in lights {
            let (a, b) = (
                ours.eval(light.to_array()).total(),
                reference.eval(light.to_array()).total(),
            );
            for k in 0..3 {
                let near = (a[k] - b[k]).abs() <= 0.03 * b[k] + 2e-3;
                assert!(near, "{i} {light}: {a:?} vs {b:?}");
            }
        }
    }
}

/// glTF's `mix(dielectric_f0, dielectric_f90, fresnel_w)` and PS "Edge color is assumed white":
/// a colored F0 falls to a white edge, scaled by the weight, and the IOR plays no part.
#[test]
fn a_fresnel_mix_runs_from_f0_to_a_white_edge() {
    let f0 = [0.5, 0.25, 0.1];
    let inputs = Inputs {
        specular_color: f0,
        specular_weight: 0.5,
        specular_roughness: 0.0,
        specular_ior: 1.1,
        base_color: [0.0; 3],
        bsdf_flags: FRESNEL_MIX,
        ..Inputs::default()
    };
    let delta = |bsdf: &mut RealTime, c: f32| {
        let view = [(1.0 - c * c).sqrt(), 0.0, c];
        bsdf.prepare(&inputs, view);
        let wi = [-view[0], 0.0, c];
        let (value, chance) = bsdf.delta_value(2, wi);
        assert!(chance > 0.0);
        value
    };
    let mut bsdf = RealTime::new();
    for c in [1.0, 0.7, 0.3, 0.05] {
        let got = delta(&mut bsdf, c);
        let x = (1.0f32 - c).powi(5);
        for k in 0..3 {
            let want = 0.5 * (f0[k] + (1.0 - f0[k]) * x);
            assert!((got[k] - want).abs() < 1e-4, "{c}: {got:?}");
        }
    }
}

/// glTF: "The metal BRDF is not affected by the parameter", against OpenPBR's weight and tint.
#[test]
fn a_fresnel_mix_leaves_metal_alone() {
    let view = [0.6, 0.0, 0.8];
    let metal = Inputs {
        base_metalness: 1.0,
        base_color: [0.9, 0.6, 0.3],
        specular_roughness: 0.4,
        ..Inputs::default()
    };
    let tinted = Inputs {
        specular_color: [0.1, 0.2, 0.3],
        specular_weight: 0.25,
        bsdf_flags: FRESNEL_MIX,
        ..metal
    };
    let [_, want] = albedo_by_kind(&mut RealTime::new(), &metal, view, 1 << 16, 3);
    let [_, got] = albedo_by_kind(&mut RealTime::new(), &tinted, view, 1 << 16, 3);
    assert!(agree(&got, &want, 1e-3), "{got:?} vs {want:?}");
}

/// PS: a metal's "F0 ... and edge F90 reflectivity will simply be the Albedo".
#[test]
fn a_flat_metal_reflects_its_color_at_every_angle() {
    let color = [0.9, 0.6, 0.3];
    let inputs = Inputs {
        base_metalness: 1.0,
        base_color: color,
        specular_roughness: 0.0,
        bsdf_flags: FRESNEL_MIX | METAL_FLAT,
        ..Inputs::default()
    };
    let mut bsdf = RealTime::new();
    for c in [1.0f32, 0.5, 0.05] {
        let view = [(1.0 - c * c).sqrt(), 0.0, c];
        bsdf.prepare(&inputs, view);
        let (value, _) = bsdf.delta_value(2, [-view[0], 0.0, c]);
        for k in 0..3 {
            assert!((value[k] - color[k]).abs() < 1e-4, "{c}: {value:?}");
        }
    }
}

#[test]
fn a_metals_specular_weight_stops_at_1() {
    let mut bsdf = RealTime::new();
    let metal = Inputs {
        base_metalness: 1.0,
        base_color: [0.9, 0.7, 0.5],
        specular_roughness: 0.3,
        ..Inputs::default()
    };
    let past = Inputs {
        specular_weight: 1.5,
        ..metal
    };
    for mu in VIEW_COSINES {
        let a = harness::albedo(&mut bsdf, &metal, view(mu), 20_000, 1);
        let b = harness::albedo(&mut bsdf, &past, view(mu), 20_000, 1);
        assert_eq!(a.total.mean, b.total.mean, "mu {mu}");
    }
}

/// KHR_materials_ior: at `ior` 0 "the Fresnel term **MUST** evaluate to `1.0` independently of the
/// view or light direction".
#[test]
fn an_ior_of_0_reflects_everything_from_every_view() {
    let mut bsdf = RealTime::new();
    let inputs = Inputs {
        base_color: [0.0; 3],
        specular_ior: 0.0,
        specular_roughness: 0.0,
        ..Inputs::default()
    };
    for mu in VIEW_COSINES {
        let a = harness::albedo(&mut bsdf, &inputs, view(mu), 2_000, 1);
        for c in a.total.mean {
            assert!((c - 1.0).abs() < 1e-3, "mu {mu}: {:?}", a.total.mean);
        }
    }
}

/// A coat at IOR 0, which the G-buffer's IOR code keeps, is a layer the base divides by.
#[test]
fn a_coat_of_ior_0_stays_finite() {
    let mut bsdf = RealTime::new();
    let inputs = Inputs {
        coat_weight: 1.0,
        coat_ior: 0.0,
        specular_roughness: 0.3,
        ..Inputs::default()
    };
    for mu in VIEW_COSINES {
        bsdf.prepare(&inputs, view(mu));
        let e = bsdf.eval(view(0.7));
        assert!(finite_non_negative(e.total()), "mu {mu}: {e:?}");
        assert!(bsdf.pdf(view(0.7)).is_finite(), "mu {mu}");
    }
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

/// Refraction scales a camera path's radiance by the squared IOR ratio, against a light path's
/// flux, on rough and ideal glass from either side; reflection does not. A path that enters and
/// leaves multiplies to 1.
#[test]
fn camera_paths_scale_refraction_by_the_squared_ior_ratio() {
    let mut bsdf = RealTime::new();
    for (roughness, exterior) in [(0.0f32, 1.0f32), (0.0, 1.33), (0.3, 1.0), (0.3, 1.33)] {
        let glass = Inputs {
            base_color: [1.0; 3],
            specular_roughness: roughness,
            specular_ior: 1.5,
            transmission_weight: 1.0,
            exterior_ior: exterior,
            ..Inputs::default()
        };
        let mut through = [1.0f64; 2];
        for (side, view) in [[0.3f32, 0.0, 0.954], [0.3, 0.0, -0.954]]
            .iter()
            .enumerate()
        {
            let (from, to) = if side == 1 {
                (1.5, f64::from(exterior))
            } else {
                (f64::from(exterior), 1.5)
            };
            bsdf.prepare(&glass, *view);
            for u in [[0.5f32, 0.5, 0.0], [0.5, 0.5, 0.999_99]] {
                let Some(s) = bsdf.sample(u) else { continue };
                let wanted = if s.lobe.is(Lobe::TRANSMISSION) {
                    (from / to).powi(2)
                } else {
                    1.0
                };
                let got = f64::from(camera_weight(&glass, *view, u));
                assert!(
                    (got - wanted).abs() < 1e-3,
                    "{got} against {wanted}: roughness {roughness}, {from} to {to}, {s:?}"
                );
                if s.lobe.is(Lobe::TRANSMISSION) {
                    through[side] = got;
                }
            }
        }
        assert!((through[0] * through[1] - 1.0).abs() < 2e-3, "{through:?}");
    }
}

/// A thin wall refracts nothing, so a camera path's weight is a light path's for every lobe of
/// rough glass and of a rough subsurface leaf, from either side.
#[test]
fn camera_paths_through_thin_walls_keep_their_radiance() {
    let mut bsdf = RealTime::new();
    let glass = Inputs {
        base_color: [1.0; 3],
        specular_roughness: 0.5,
        specular_ior: 1.5,
        transmission_weight: 1.0,
        geometry_thin_walled: 1,
        ..Inputs::default()
    };
    let leaf = Inputs {
        transmission_weight: 0.0,
        subsurface_weight: 1.0,
        ..glass
    };
    let mut rng = Rng::new(7);
    for inputs in [glass, leaf] {
        let mut through = 0;
        for view in [[0.3f32, 0.0, 0.954], [0.3, 0.0, -0.954]] {
            bsdf.prepare(&inputs, view);
            for _ in 0..4096 {
                let u = [rng.f32(), rng.f32(), rng.f32()];
                let Some(s) = bsdf.sample(u) else { continue };
                through += usize::from(s.lobe.is(Lobe::TRANSMISSION));
                let got = camera_weight(&inputs, view, u);
                assert!(
                    (got - 1.0).abs() < 1e-3,
                    "{got}: {inputs:?}, {view:?}, {s:?}"
                );
            }
        }
        assert!(through > 0, "{inputs:?}");
    }
}

/// A smooth dielectric seen from either side at several angles, in air and in water:
/// reflection mirrors the view, refraction follows Snell's law, and the reflection and
/// transmission albedos are the Fresnel reflectance and its complement, transmission 0
/// past the critical angle.
#[test]
fn smooth_dielectrics_follow_snell_and_fresnel() {
    let mut bsdf = RealTime::new();
    for exterior in [1.0f32, 1.33] {
        let glass = Inputs {
            base_color: [1.0; 3],
            specular_roughness: 0.0,
            specular_ior: 1.5,
            transmission_weight: 1.0,
            exterior_ior: exterior,
            ..Inputs::default()
        };
        for inside in [false, true] {
            for cos in [1.0f64, 0.9, 0.6, 0.3] {
                let sin = (1.0 - cos * cos).sqrt();
                let side = if inside { -1.0 } else { 1.0 };
                bsdf.prepare(&glass, [sin as f32, 0.0, (side * cos) as f32]);
                let (from, to) = if inside {
                    (1.5, f64::from(exterior))
                } else {
                    (f64::from(exterior), 1.5)
                };
                let f = fresnel(to / from, cos);
                let what = format!("ior {to} from {from} at cos {cos}");
                let a = bsdf.albedo();
                assert!(
                    (f64::from(a.specular[0]) - f).abs() < 2e-3,
                    "{what}: {a:?}, F {f}"
                );
                assert!(
                    (f64::from(a.transmission[0]) - (1.0 - f)).abs() < 2e-3,
                    "{what}: {a:?}, F {f}"
                );
                let r = bsdf.sample([0.5, 0.5, 0.0]).expect("reflection");
                assert!(r.lobe.is(Lobe::REFLECTION), "{what}: {r:?}");
                let mirror = [-sin, 0.0, side * cos];
                assert!(
                    (0..3).all(|k| (f64::from(r.wi[k]) - mirror[k]).abs() < 1e-4),
                    "{what}: {r:?}"
                );
                let t = bsdf.sample([0.5, 0.5, 0.999_99]).expect("sample");
                let sin_t = sin * from / to;
                if sin_t >= 1.0 {
                    assert!(t.lobe.is(Lobe::REFLECTION), "{what}: {t:?}");
                    continue;
                }
                assert!(t.lobe.is(Lobe::TRANSMISSION), "{what}: {t:?}");
                let snell = [-sin_t, 0.0, -side * (1.0 - sin_t * sin_t).sqrt()];
                assert!(
                    (0..3).all(|k| (f64::from(t.wi[k]) - snell[k]).abs() < 1e-4),
                    "{what}: {t:?} against {snell:?}"
                );
            }
        }
    }
}

#[test]
fn delta_values_match_the_samples_that_drew_them() {
    let glass = Inputs {
        base_color: [1.0; 3],
        specular_roughness: 0.0,
        specular_ior: 1.5,
        transmission_weight: 1.0,
        ..Inputs::default()
    };
    // Each with the sides it is seen from.
    let materials = [
        (glass, [false, true]),
        (
            Inputs {
                geometry_thin_walled: 1,
                ..glass
            },
            [false, true],
        ),
        (
            Inputs {
                coat_weight: 1.0,
                coat_roughness: 0.0,
                specular_roughness: 0.4,
                ..Inputs::default()
            },
            [false, false],
        ),
    ];
    let mut bsdf = RealTime::new();
    let mut rng = Rng::new(7);
    for (i, (m, sides)) in materials.iter().enumerate() {
        for inside in *sides {
            let z = if inside { -0.6f32 } else { 0.6 };
            bsdf.prepare(m, [0.8, 0.0, z]);
            let mut deltas = 0;
            for _ in 0..64 {
                let u = [rng.f32(), rng.f32(), rng.f32()];
                let Some(s) = bsdf.sample(u) else { continue };
                if !s.lobe.is(Lobe::SPECULAR) {
                    continue;
                }
                deltas += 1;
                let component = bsdf.sample_component(u);
                let (value, chance) = bsdf.delta_value(component, s.wi);
                let what = format!("material {i}, inside {inside}: {s:?}");
                assert!((chance - s.pdf).abs() <= 1e-5 * s.pdf, "{what}: {chance}");
                for c in 0..3 {
                    let wanted = s.weight[c] * s.pdf;
                    assert!(
                        (value[c] - wanted).abs() <= 1e-4 * wanted.max(1e-3),
                        "{what}: {value:?}"
                    );
                }
                let off = Vec3::new(s.wi[0] + 0.01, s.wi[1], s.wi[2]).normalize();
                let (value, chance) = bsdf.delta_value(component, off.to_array());
                assert!(
                    chance == 0.0 && value == [0.0; 3],
                    "{what}: off the direction"
                );
            }
            assert!(deltas > 0, "material {i} drew no delta lobe");
        }
    }
}

#[test]
fn dispersed_iors_follow_the_abbe_number() {
    for (ior, scale, abbe) in [
        (1.5f32, 1.0f32, 20.0f32),
        (1.7, 0.5, 40.0),
        (1.33, 1.0, 55.0),
    ] {
        let at = |nm| f64::from(dispersed_ior(ior, scale, abbe, nm));
        assert!((at(587.6) - f64::from(ior)).abs() < 1e-5, "ior {ior}");
        let v = (at(587.6) - 1.0) / (at(486.1) - at(656.3));
        let wanted = f64::from(abbe / scale);
        assert!((v - wanted).abs() < 1e-3 * wanted, "ior {ior}: Abbe {v}");
        assert!(at(400.0) > at(500.0) && at(500.0) > at(700.0));
    }
    assert_eq!(dispersed_ior(1.5, 0.0, 20.0, 450.0), 1.5);
    // Below 1 (a dense medium against a lighter one) the reciprocal disperses.
    let inverse = f64::from(dispersed_ior(1.0 / 1.5, 1.0, 20.0, 486.1));
    assert!((1.0 / inverse - f64::from(dispersed_ior(1.5, 1.0, 20.0, 486.1))).abs() < 1e-5);
}

/// Colors sent through the sampled wavelengths and back keep their value on average.
#[test]
fn sampled_wavelengths_return_colors_unchanged() {
    let n = 4096;
    let mut mean = [[0.0f64; 3]; 3];
    for i in 0..n {
        let m = spectral_transfer((i as f32 + 0.5) / n as f32, [1.0; 4]);
        for r in 0..3 {
            for c in 0..3 {
                mean[r][c] += f64::from(m[r][c]) / f64::from(n);
            }
        }
    }
    for r in 0..3 {
        for c in 0..3 {
            let wanted = if r == c { 1.0 } else { 0.0 };
            assert!((mean[r][c] - wanted).abs() < 2e-3, "{mean:?}");
        }
    }
}
