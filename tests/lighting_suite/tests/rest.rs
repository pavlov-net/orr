//! The rest stage, which runs a path's later rounds in one lane, against the stages running every
//! round. It is a target of its own: the stage's pipeline holds the trace stage and every shade
//! stage, and the frames of other tests in its process would wait out its compile.

use lighting_suite::assert_clean;
use lighting_suite::bed::{Accumulated, Bed, FURNACE_RADIANCE, GLASS_SPHERE, Item, Paths, Spec};
use orr_gpu::KernelSet;
use orr_lighting::{PathEntry, ViewLighting};
use orr_math::{Mat3, Vec3};

fn render_paths(bed: &mut Bed, paths: &Paths, frames: u32) -> Accumulated {
    bed.accumulate(frames, 1, &mut |list, f| paths.record(list, f))
}

/// The rest stage runs a path's later rounds in one lane. Light inside a clear glass sphere
/// reflects on past the round it starts at, and the sphere renders bit for bit as it does when
/// the stages run every round.
#[test]
fn the_rest_stage_traces_the_paths_the_stages_do() {
    let spec = Spec {
        copies: vec![Item {
            model: GLASS_SPHERE,
            linear: Mat3::IDENTITY,
            at: Vec3::ZERO,
        }],
        lights: Vec::new(),
        environment: Some(FURNACE_RADIANCE),
        eye: Vec3::new(0.0, 0.0, 4.0),
        target: Vec3::ZERO,
        fov_y: 0.6,
        focus: None,
    };
    let bits = |a: &Accumulated| {
        a.mean
            .iter()
            .map(|p| p.map(f32::to_bits))
            .collect::<Vec<_>>()
    };
    let settings = ViewLighting {
        rest_stage: false,
        ..ViewLighting::default()
    };
    let mut paths = Paths::new(&KernelSet::default(), PathEntry::Compute, settings);
    paths.specialize = true;
    let stages = render_paths(&mut Bed::with(spec.clone()), &paths, 4);
    paths.settings.rest_stage = true;
    paths.rest = true;
    let rest = render_paths(&mut Bed::with(spec), &paths, 4);
    assert!(
        bits(&stages) == bits(&rest),
        "the rest stage changed the image"
    );
    assert_clean();
}
