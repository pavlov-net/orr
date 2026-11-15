//! The shader bodies of the pipeline path tracer, as it ships them, against their kinds' register
//! budgets, compiled as compute stand-ins for the pipeline that may meet everything and for the
//! bench walk's view pipeline, with orr_lighting's own kernels and with the suite's linked
//! surfaces.

use lighting_suite::{device, linked, require};
use orr_gpu::KernelSet;
use orr_lighting::standins::{Body, compile_body};

/// `path_may` and `path_may2` of the pipeline of the bench's synthetic walk that runs a view's
/// paths, the widest of its sources': over cooked clusters and producer entries, with interiors,
/// layers, particles, coverage, caustics, a plane stack, local light samplers, froxels, the
/// cache, directional and emissive lights.
const BENCH_MAY: u64 = 0xffff_fc30_0378_700d;

/// The bodies that do not fit yet, with what holds their registers: each must exceed its budget
/// somewhere, so the list cannot outlive the redesign that takes a body off it.
const OVER: &[(Body, &str)] = &[
    (
        Body::Resolve,
        "the standard surface's evaluation, and its decals with every path_may2 bit",
    ),
    (
        Body::Vertex,
        "the surface's evaluation, then the path with the BSDF across its light candidates",
    ),
    (
        Body::VertexLayer,
        "the surface's evaluation, then the path with the BSDF across its light candidates",
    ),
    (
        Body::Emission,
        "the emitter's triangle, rebuilt for the light set's density at the hit",
    ),
    (
        Body::CoverPath,
        "the hit reconstruction its presence resolve shares with the inline handlers",
    ),
    (
        Body::CoverShadow,
        "the hit reconstruction its presence resolve shares with the inline handlers",
    ),
];

#[test]
fn shader_bodies_fit_their_budgets() {
    let d = device();
    if !require(
        d.caps().pipeline_executable_info,
        "the device reports no pipeline statistics",
    ) {
        return;
    }
    let sets: [(&str, &KernelSet); 2] = [("own", &KernelSet::NONE), ("linked", &linked::KERNELS)];
    let mut failures = Vec::new();
    let mut table = Vec::new();
    for body in Body::ALL {
        let mut fits = true;
        for (set, kernels) in sets {
            for may in [u64::MAX, BENCH_MAY] {
                let s = compile_body(d, kernels, body, may)
                    .unwrap()
                    .expect("the device reports statistics");
                let line = format!(
                    "{} ({set}, {may:#018x}): {} registers of {}, {} B local, {} B shared, {} B of code, {:.2} s",
                    body.name(),
                    s.code.registers,
                    body.kind().budget(),
                    s.code.local_bytes,
                    s.code.shared_bytes,
                    s.code.binary_bytes,
                    s.seconds,
                );
                table.push(line.clone());
                if !s.fits(body.kind()) {
                    fits = false;
                    if !OVER.iter().any(|(b, _)| *b == body) {
                        failures.push(format!("over budget: {line}"));
                    }
                }
            }
        }
        if let Some((_, why)) = OVER.iter().find(|(b, _)| *b == body) {
            if fits {
                failures.push(format!("{} fits now: take it off OVER", body.name()));
            } else {
                table.push(format!("{} waits on {why}", body.name()));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}\n\n{}",
        failures.join("\n"),
        table.join("\n")
    );
}
