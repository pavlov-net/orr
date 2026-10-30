//! `orr.openpbr` on the GPU (heap LUTs) passes every harness gate against adobe's C++, and
//! agrees with the same Slang on the CPU (array LUTs) within 1e-3. Runs with validation on
//! and fails if the layer reports anything.

use openpbr_ref::harness::compare;
use openpbr_ref::materials::{VIEW_COSINES, grid, view};
use openpbr_ref::report::{Counts, SAME_MODEL, evaluate};
use openpbr_ref::{Bsdf, Inputs, Reference, SlangCpu, SlangGpu};

fn assert_clean() {
    assert_eq!(
        SlangGpu::device().validation_messages(),
        0,
        "validation messages (see the log)"
    );
}

#[test]
fn gpu_passes_the_gates() {
    let results = evaluate(
        "orr.openpbr (GPU)",
        SlangGpu::new,
        Reference::new,
        true,
        Counts::default(),
    );
    let failures = results.failures();
    assert!(
        failures.is_empty(),
        "{}\n\n{}",
        failures.join("\n"),
        results.markdown()
    );
    assert_clean();
}

#[test]
fn gpu_matches_cpu() {
    let (mut g, mut c) = (SlangGpu::new(), SlangCpu::new());
    let mut failures = Vec::new();
    for (i, m) in grid().iter().enumerate() {
        for (v, &mu) in VIEW_COSINES.iter().enumerate() {
            let seed = (i * 8 + v) as u64;
            let cmp = compare(&mut g, &mut c, &m.inputs, view(mu), 4_000, seed);
            if cmp.eval.max > SAME_MODEL || cmp.pdf.max > SAME_MODEL {
                failures.push(format!(
                    "{}, mu {mu}: eval {:.2e}, pdf {:.2e}, {} of {} samples differ",
                    m.name, cmp.eval.max, cmp.pdf.max, cmp.sample_mismatches, cmp.samples
                ));
            }
        }

        let inputs = Inputs {
            emission_luminance: 100.0,
            transmission_depth: 0.5,
            transmission_scatter: [0.2, 0.3, 0.4],
            ..m.inputs
        };
        g.prepare(&inputs, [0.6, 0.0, 0.8]);
        c.prepare(&inputs, [0.6, 0.0, 0.8]);
        let (vg, vc) = (g.volume(), c.volume());
        let (eg, ec) = (g.emission(), c.emission());
        let close = |a: f32, b: f32| (a - b).abs() <= SAME_MODEL as f32 * b.abs().max(1e-3);
        let same = (0..3).all(|k| {
            close(vg.extinction[k], vc.extinction[k])
                && close(vg.albedo[k], vc.albedo[k])
                && close(eg[k], ec[k])
        }) && close(vg.anisotropy, vc.anisotropy);
        if !same {
            failures.push(format!(
                "{}: volume {vg:?} vs {vc:?}, emission {eg:?} vs {ec:?}",
                m.name
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_clean();
}
