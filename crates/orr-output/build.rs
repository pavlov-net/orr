//! Compiles orr-output's Slang kernels and publishes its shader root, and fits the glare
//! band weights: the diaphragm's wavelength-integrated Fraunhofer PSF (CIE 1931 2-degree
//! matching functions, D65 weighted, in Rec.2020) against the pyramid's kernels, into
//! `OUT_DIR/glare_tables.rs`. The PSF data the output suite checks the fit with goes to
//! `OUT_DIR/glare_psf.rs`, published with the CPU model's directory as `glare_psf` and
//! `glare_model`. The FFT glare's kernel table (`build/fft.rs`) goes to
//! `OUT_DIR/glare_fft_table.bin`, its far-field terms to `OUT_DIR/glare_fft.rs`. With the
//! `pcl-stats` feature on Windows it also builds the PCL Stats shim against `pclstats.h` from
//! the Reflex SDK at `REFLEX_SDK`.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use orr_color::RgbPrimaries;
use orr_shaders::build::{Build, fetched};

#[allow(dead_code, unreachable_pub)]
#[path = "build/fft.rs"]
mod fft;
#[allow(unreachable_pub)]
#[path = "build/fit.rs"]
mod fit;
#[allow(unreachable_pub)]
#[path = "build/psf.rs"]
mod psf;
#[allow(dead_code, unreachable_pub)]
#[path = "build/pyramid.rs"]
mod pyramid;

/// Blades of the diaphragm the glare PSF is fitted for.
const GLARE_BLADES: u32 = 6;

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let scene = env::var("DEP_ORR_SCENE_SHADER_DIRS").unwrap_or_default();
    let out = Build::new().and_then(|b| {
        b.kernels(manifest.join("src"))
            .roots(&scene)
            .extern_type("Gt7Curve", "crate::Gt7Curve")
            .extern_type("ViewConstants", "::orr_scene::ViewConstants")
            .extern_type("ExposureState", "::orr_scene::ExposureState")
            .extern_type("SkyStats", "::orr_scene::SkyStats")
            .compile()
    });
    match out {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
    glare(&manifest, &out_dir);
    #[cfg(feature = "pcl-stats")]
    pcl_stats(&manifest);
}

#[cfg(feature = "pcl-stats")]
fn pcl_stats(manifest: &Path) {
    println!("cargo:rerun-if-env-changed=REFLEX_SDK");
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
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", shim.display());
    cc::Build::new()
        .cpp(true)
        .file(shim)
        .include(include)
        .compile("orr_pcl_stats");
    println!("cargo:rustc-link-lib=advapi32");
    println!("cargo:rustc-link-lib=user32");
}

fn glare(manifest: &Path, out_dir: &Path) {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build");
    let sky = fetched("ORR_SKY_DATA", "precomputed_atmospheric_scattering")
        .unwrap_or_else(|e| panic!("{e}"));
    let d65 = fetched("ORR_CIE_D65", "cie/CIE_std_illum_D65.csv").unwrap_or_else(|e| panic!("{e}"));
    let spectral = spectral(&sky.join("atmosphere/constants.h"), &d65);

    let diaphragm = psf::Diaphragm::new(GLARE_BLADES);
    let (ee, tail) = diaphragm.encircled();
    let tables = fit::tables(&ee, tail, &spectral);
    let entries = pyramid::LAST_LEVEL as usize + 1;

    let mut rs = String::new();
    let n = fit::F_NUMBERS;
    writeln!(rs, "pub(crate) const GLARE_F_NUMBERS: usize = {n};").unwrap();
    writeln!(rs, "/// Pixel pitch the tables are fitted at, um.").unwrap();
    writeln!(
        rs,
        "pub(crate) const GLARE_REFERENCE_PITCH_UM: f32 = {:?};",
        fit::REFERENCE_PITCH_UM as f32
    )
    .unwrap();
    writeln!(rs, "pub(crate) const GLARE_ENTRIES: usize = {entries};").unwrap();
    writeln!(
        rs,
        "/// Weights per first level (1, 2), F-number and entry (w0, then each mip's band);          xyz per Rec.2020 channel."
    )
    .unwrap();
    writeln!(
        rs,
        "pub(crate) static GLARE_TABLES: [[f32; 4]; {}] = [",
        2 * n * entries
    )
    .unwrap();
    for w in tables.iter().flatten().flatten() {
        writeln!(
            rs,
            "    [{:?}, {:?}, {:?}, 0.0],",
            w[0] as f32, w[1] as f32, w[2] as f32
        )
        .unwrap();
    }
    writeln!(rs, "];").unwrap();
    write(&out_dir.join("glare_tables.rs"), &rs);

    let f_numbers: Vec<f64> = (0..n).map(fit::f_number).collect();
    let mut rs = String::new();
    writeln!(rs, "pub const BLADES: u32 = {GLARE_BLADES};").unwrap();
    writeln!(
        rs,
        "pub const REFERENCE_PITCH_UM: f64 = {:?};",
        fit::REFERENCE_PITCH_UM
    )
    .unwrap();
    writeln!(rs, "pub const F_NUMBERS: [f64; {n}] = {f_numbers:?};").unwrap();
    writeln!(rs, "pub const EE_TAIL: f64 = {tail:?};").unwrap();
    writeln!(rs, "pub static EE: [f64; {}] = {ee:?};", ee.len()).unwrap();
    writeln!(
        rs,
        "pub static SPECTRAL: [(f64, [f64; 3]); {}] = {spectral:?};",
        spectral.len()
    )
    .unwrap();
    writeln!(
        rs,
        "pub static TABLES: [[[[f64; 3]; {entries}]; {n}]; 2] = {tables:?};"
    )
    .unwrap();
    let kernel = fft::Kernel::compute(&diaphragm, &spectral, fit::REFERENCE_PITCH_UM);
    let kappa = kernel.kappa.map(|k| k as f32);
    let spikes: Vec<[f32; 2]> = kernel.spikes.iter().map(|s| s.map(|c| c as f32)).collect();
    let table_path = out_dir.join("glare_fft_table.bin");
    let bytes: Vec<u8> = kernel
        .levels
        .iter()
        .flatten()
        .flat_map(|e| [e[0] as f32, e[1] as f32, e[2] as f32, 0.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    fs::write(&table_path, bytes).unwrap_or_else(|e| panic!("{}: {e}", table_path.display()));
    let fft_rs = format!(
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
    write(&out_dir.join("glare_fft.rs"), &fft_rs);
    let kappa64 = kappa.map(f64::from);
    let spikes64: Vec<[f64; 2]> = spikes.iter().map(|s| s.map(f64::from)).collect();
    writeln!(
        rs,
        "pub const FFT_KAPPA: [f64; 3] = {kappa64:?};\n\
         pub static FFT_SPIKES: [[f64; 2]; {}] = {spikes64:?};\n\
         pub const FFT_TABLE_BIN: &str = {:?};",
        spikes64.len(),
        table_path.display().to_string()
    )
    .unwrap();
    let psf_path = out_dir.join("glare_psf.rs");
    write(&psf_path, &rs);
    println!("cargo:glare_psf={}", psf_path.display());
    println!("cargo:glare_model={}", manifest.join("build").display());
}

/// Per wavelength (um), each Rec.2020 channel's share of a D65 white: D65 times the
/// channel's color matching function, normalized per channel over the 5 nm samples.
fn spectral(constants: &Path, d65: &Path) -> Vec<(f64, [f64; 3])> {
    // Rows of (wavelength, x, y, z) every 5 nm from 360 nm.
    let cie = parse_array(&read(constants), "CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[380]");
    let illuminant: Vec<(f64, f64)> = read(d65)
        .lines()
        .filter_map(|l| {
            let (nm, v) = l.split_once(',')?;
            Some((nm.trim().parse().ok()?, v.trim().parse().ok()?))
        })
        .collect();
    let to_rec2020 = RgbPrimaries::REC2020.xyz_to_rgb();
    let rows: Vec<(f64, [f64; 3])> = cie
        .chunks_exact(4)
        .map(|r| {
            let &(_, s) = illuminant
                .iter()
                .find(|&&(nm, _)| nm == r[0])
                .unwrap_or_else(|| panic!("D65 has no {} nm", r[0]));
            let rgb = orr_color::mul(&to_rec2020, [r[1], r[2], r[3]]);
            (r[0] / 1000.0, rgb.map(|c| c * s))
        })
        .collect();
    let sums = [0, 1, 2].map(|c| rows.iter().map(|r| r.1[c]).sum::<f64>());
    rows.into_iter()
        .map(|(l, w)| (l, [0, 1, 2].map(|c| w[c] / sums[c])))
        .collect()
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn write(path: &Path, text: &str) {
    fs::write(path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The numbers inside the braces after `name` in C++ source.
fn parse_array(src: &str, name: &str) -> Vec<f64> {
    let start = src.find(name).unwrap_or_else(|| panic!("{name} not found"));
    let open = start + src[start..].find('{').unwrap();
    let close = open + src[open..].find('}').unwrap();
    src[open + 1..close]
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or_else(|e| panic!("{name}: {s}: {e}")))
        .collect()
}
