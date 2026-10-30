//! The reference wrapper `orr.openpbr` (Slang, CPU) passes every harness gate: energy,
//! chi-square, reciprocity, and agreement with adobe/openpbr-bsdf's C++ within 1e-3.

use openpbr_ref::report::{Counts, evaluate};
use openpbr_ref::{Bsdf, Reference, SlangCpu};

#[test]
fn reference_wrapper_passes_the_gates() {
    let results = evaluate(
        "orr.openpbr",
        SlangCpu::new,
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
}

#[test]
fn volume_and_emission_match_the_reference() {
    let (mut s, mut r) = (SlangCpu::new(), Reference::new());
    for m in openpbr_ref::materials::grid() {
        let inputs = openpbr_ref::Inputs {
            emission_luminance: 100.0,
            transmission_depth: 0.5,
            transmission_scatter: [0.2, 0.3, 0.4],
            ..m.inputs
        };
        s.prepare(&inputs, [0.6, 0.0, 0.8]);
        r.prepare(&inputs, [0.6, 0.0, 0.8]);
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * b.abs().max(1e-3);
        let (vs, vr) = (s.volume(), r.volume());
        let (es, er) = (s.emission(), r.emission());
        for c in 0..3 {
            assert!(
                close(vs.extinction[c], vr.extinction[c]),
                "{}: {vs:?} vs {vr:?}",
                m.name
            );
            assert!(
                close(vs.albedo[c], vr.albedo[c]),
                "{}: {vs:?} vs {vr:?}",
                m.name
            );
            assert!(close(es[c], er[c]), "{}: {es:?} vs {er:?}", m.name);
        }
        assert!(close(vs.anisotropy, vr.anisotropy), "{}", m.name);
    }
}
