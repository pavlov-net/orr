//! Runs each example headless and compares its last frame with its reference in
//! `references/` by FLIP; `ORR_BLESS=1` rewrites the references. hdr-calibration shows what
//! a display does with its patterns and has no headless run.

#[allow(dead_code)]
#[path = "../cluster-rt.rs"]
mod cluster_rt;
#[allow(dead_code)]
#[path = "../gpu-basics.rs"]
mod gpu_basics;
#[allow(dead_code)]
#[path = "../producer-field.rs"]
mod producer_field;
#[allow(dead_code)]
#[path = "../rigid-parts.rs"]
mod rigid_parts;
#[allow(dead_code)]
#[path = "../viewer.rs"]
mod viewer;

use std::path::Path;

use examples::{Capture, Options, assert_clean};
use orr_scene::testing::{Image, Pixels, Tolerance, check};

fn compare(name: &str, c: Capture) {
    let image = Image {
        width: c.width,
        height: c.height,
        pixels: Pixels::Rgba8(c.rgba8),
    };
    let reference = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("references")
        .join(name);
    check(
        &reference,
        &image,
        Tolerance::Flip {
            mean: 0.01,
            p99: 0.1,
        },
    );
}

#[test]
fn examples_match_their_references() {
    compare("gpu-basics", gpu_basics::run(&Options::headless(120, &[])));
    let cluster_rt =
        |rest: &[&str]| cluster_rt::run(&Options::headless(120, rest)).expect("headless capture");
    compare("cluster-rt", cluster_rt(&[]));
    compare("cluster-rt-clusters", cluster_rt(&["--clusters"]));
    let viewer = viewer::run(&Options::headless(120, &[])).expect("headless capture");
    compare("viewer", viewer);
    let rigid = rigid_parts::run(&Options::headless(120, &[])).expect("headless capture");
    compare("rigid-parts", rigid);
    let field = producer_field::run(&Options::headless(120, &[])).expect("headless capture");
    compare("producer-field", field);
    assert_clean();
}
