//! Runs each example headless and compares its last frame with its reference in
//! `references/` by FLIP; `ORR_BLESS=1` rewrites the references. hdr-calibration shows what
//! a display does with its patterns and has no headless run. The dlss example's super
//! resolution is compared with the same view rendered at the output size instead.

#[allow(dead_code)]
#[path = "../cluster-rt.rs"]
mod cluster_rt;
#[allow(dead_code)]
#[path = "../dense-vegetation.rs"]
mod dense_vegetation;
#[allow(dead_code)]
#[path = "../dielectrics.rs"]
mod dielectrics;
#[allow(dead_code)]
#[path = "../dlss.rs"]
mod dlss;
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
#[path = "../skinned.rs"]
mod skinned;
#[allow(dead_code)]
#[path = "../viewer.rs"]
mod viewer;

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use examples::{Capture, Options, assert_clean};
use orr_scene::testing::{Image, Pixels, Tolerance, check, flip};

/// A smaller forest than the bench's, so the debug build renders it in reasonable time.
const DENSE_VEGETATION: [&str; 5] = ["--synthetic", "--trees", "2000", "--grass-density", "50"];

/// One example runs at a time on the shared device.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn image(c: Capture) -> Image {
    Image {
        width: c.width,
        height: c.height,
        pixels: Pixels::Rgba8(c.rgba8),
    }
}

fn compare(name: &str, c: Capture) {
    let image = image(c);
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

fn headless(rest: &[&str]) -> Options {
    Options::headless(120, rest)
}

#[test]
fn gpu_basics_matches_its_reference() {
    let _serial = serial();
    compare("gpu-basics", gpu_basics::run(&headless(&[])));
    assert_clean();
}

#[test]
fn cluster_rt_matches_its_references() {
    let _serial = serial();
    let cluster_rt = |rest: &[&str]| cluster_rt::run(&headless(rest)).expect("headless capture");
    compare("cluster-rt", cluster_rt(&[]));
    compare("cluster-rt-clusters", cluster_rt(&["--clusters"]));
    assert_clean();
}

#[test]
fn viewer_matches_its_reference() {
    let _serial = serial();
    compare(
        "viewer",
        viewer::run(&headless(&[])).expect("headless capture"),
    );
    assert_clean();
}

#[test]
fn rigid_parts_matches_its_reference() {
    let _serial = serial();
    compare(
        "rigid-parts",
        rigid_parts::run(&headless(&[])).expect("headless capture"),
    );
    assert_clean();
}

#[test]
fn dielectrics_matches_its_reference() {
    let _serial = serial();
    compare(
        "dielectrics",
        dielectrics::run(&headless(&[])).expect("headless capture"),
    );
    assert_clean();
}

#[test]
fn producer_field_matches_its_reference() {
    let _serial = serial();
    let field = producer_field::run(&headless(&[])).expect("headless capture");
    compare("producer-field", field);
    assert_clean();
}

#[test]
fn skinned_matches_its_reference() {
    let _serial = serial();
    compare(
        "skinned",
        skinned::run(&headless(&[])).expect("headless capture"),
    );
    assert_clean();
}

#[test]
fn dense_vegetation_matches_its_reference() {
    let _serial = serial();
    let forest = dense_vegetation::capture(&headless(&DENSE_VEGETATION)).expect("headless capture");
    compare("dense-vegetation", forest);
    assert_clean();
}

/// The dlss example's raster debug view upscaled by super resolution at Quality, on an
/// orbiting camera, matches the view rendered at the output size, and NGX's log names the
/// preset it applied.
#[test]
fn super_resolution_matches_the_native_render() {
    let _serial = serial();
    let frames = 64;
    let Some(sr) = dlss::run_sr(&Options::headless(frames, &["--sr", "--quality"])) else {
        log::warn!("skipped: NGX offers no super resolution here");
        return;
    };
    let native = dlss::run_sr(&Options::headless(frames, &["--sr", "--native"])).expect("native");
    let preset = sr.preset.expect("NGX's log names the applied SR preset");
    let error = flip(
        &image(native.capture.expect("native capture")),
        &image(sr.capture.expect("SR capture")),
    );
    log::info!("SR Quality, preset {preset}: FLIP {error:?} against the native render");
    // Measured mean 0.0086 and p99 0.141 (preset K); the jitter sign flipped gives 0.0118
    // and 0.189.
    assert!(
        error.mean <= 0.010 && error.p99 <= 0.16,
        "SR against native: {error:?}"
    );
    assert_clean();
}
