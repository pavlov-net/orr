//! Compiles orr_output's Slang kernels and publishes its shader root, and builds the FFT glare's
//! kernel table (`build/fft.rs`) into `OUT_DIR/glare_fft_table.bin` and its far-field terms into
//! `OUT_DIR/glare_fft.rs`, publishing that directory as `glare_fft`. With the `pcl-stats` feature
//! on Windows it also builds the PCL Stats shim against `pclstats.h` from the Reflex SDK at
//! `REFLEX_SDK`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use orr_shaders::build::Build;

#[allow(unreachable_pub)]
#[path = "build/fft.rs"]
mod fft;
#[allow(unreachable_pub)]
#[path = "build/psf.rs"]
mod psf;
#[allow(unreachable_pub)]
#[path = "build/spectral_weights.rs"]
mod spectral_weights;

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let out = Build::new().and_then(|b| {
        b.kernels(manifest.join("src"))
            .dep("orr_scene")
            .extern_type("Gt7Curve", "crate::Gt7Curve")
            .publish_types("::orr_output")
            .compile()
    });
    if let Err(e) = out {
        panic!("{e}");
    }
    glare_fft(&out_dir);
    #[cfg(feature = "pcl-stats")]
    pcl_stats(&manifest);
}

#[cfg(feature = "pcl-stats")]
fn pcl_stats(manifest: &Path) {
    println!("cargo::rerun-if-env-changed=REFLEX_SDK");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let sdk = env::var_os("REFLEX_SDK")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .expect("the pcl-stats feature needs REFLEX_SDK set to the Reflex SDK");
    let include = sdk.join("Reflex_Stats");
    let header = include.join("pclstats.h");
    assert!(header.is_file(), "{} not found", header.display());
    let shim = manifest.join("src/pcl_stats.cpp");
    println!("cargo::rerun-if-changed={}", header.display());
    println!("cargo::rerun-if-changed={}", shim.display());
    cc::Build::new()
        .cpp(true)
        .file(shim)
        .include(include)
        .compile("orr_pcl_stats");
    println!("cargo::rustc-link-lib=advapi32");
    println!("cargo::rustc-link-lib=user32");
}

/// The FFT glare's kernel table and far-field terms, published for the output suite's checks.
fn glare_fft(out_dir: &Path) {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=build");
    let kernel = fft::Kernel::compute(
        &psf::Diaphragm::new(psf::BLADES),
        &spectral_weights::SPECTRAL_WEIGHTS,
        psf::REFERENCE_PITCH_UM,
    );
    let bytes: Vec<u8> = kernel
        .levels
        .iter()
        .flatten()
        .flat_map(|e| [e[0] as f32, e[1] as f32, e[2] as f32, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    fs::write(out_dir.join("glare_fft_table.bin"), bytes).unwrap();
    let kappa = kernel.kappa.map(|k| k as f32);
    let spikes: Vec<[f32; 2]> = kernel.spikes.iter().map(|s| s.map(|c| c as f32)).collect();
    let terms = format!(
        "pub(crate) const GLARE_FFT_KAPPA: [f32; 3] = {kappa:?};\n\
         pub(crate) const GLARE_FFT_SPIKES: [[f32; 2]; {}] = {spikes:?};\n\
         pub(crate) const GLARE_FFT_TABLE: usize = {};\n\
         pub(crate) const GLARE_FFT_LEVELS: usize = {};\n\
         pub(crate) const GLARE_FFT_BOX_TEXELS: f64 = {:?};\n\
         pub(crate) const GLARE_FFT_REFERENCE_F: f64 = {:?};\n\
         pub(crate) const GLARE_FFT_TEXELS_PER_PIXEL: f64 = {:?};\n",
        spikes.len(),
        fft::TABLE,
        fft::LEVELS,
        fft::BOX_TEXELS,
        fft::REFERENCE_F,
        fft::TEXELS_PER_PIXEL,
    );
    fs::write(out_dir.join("glare_fft.rs"), terms).unwrap();
    println!("cargo::metadata=glare_fft={}", out_dir.display());
}
