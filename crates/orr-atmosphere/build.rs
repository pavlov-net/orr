//! Derives the spectral, group and Rec.2020 band coefficients from the pinned ASTM G173
//! solar spectrum, ozone cross sections and CIE 1931 2-deg tables, into
//! `OUT_DIR/spectrum.rs`; compiles the crate's kernels and publishes its shader root.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

#[allow(dead_code, unreachable_pub)]
#[path = "src/colour.rs"]
mod colour;

use colour::{LUMINOUS_EFFICACY, XYZ_TO_REC2020};
use orr_shaders::build::Build;

/// Wavelength groups in the LUT passes; each is `BINS / SPECTRAL_GROUPS` adjacent 10 nm bins.
/// 24 is the finer alternative.
const SPECTRAL_GROUPS: usize = 12;
const BINS: usize = 48;
const FIRST_NM: f64 = 360.0;
const BIN_NM: f64 = 10.0;
const ANGSTROM_STEPS: usize = 26;
const ANGSTROM_STEP: f64 = 0.1;
/// Rayleigh scattering of Earth's sea-level air at 1 um, 1/m (scales as lambda^-4).
const RAYLEIGH_1UM: f64 = 1.24062e-6;
const DATA_DIR: &str = "precomputed_atmospheric_scattering";

fn main() {
    println!("cargo:rerun-if-env-changed=ORR_SKY_DATA");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/colour.rs");
    assert_eq!(
        BINS % SPECTRAL_GROUPS,
        0,
        "groups must split the bins evenly"
    );
    assert_eq!(
        SPECTRAL_GROUPS % 3,
        0,
        "the LUT passes run three groups at a time"
    );

    let root = data_dir();
    let demo_path = root.join("atmosphere/demo/demo.cc");
    let constants_path = root.join("atmosphere/constants.h");
    println!("cargo:rerun-if-changed={}", demo_path.display());
    println!("cargo:rerun-if-changed={}", constants_path.display());
    let demo = read(&demo_path);
    let constants = read(&constants_path);

    // G173 ETR averaged over [l, l + 10 nm), W m^-2 nm^-1, and ozone cross sections at
    // 233 K averaged the same way, m^2; both labelled with the bin's first wavelength.
    let solar = parse_array(&demo, "kSolarIrradiance[48]");
    let ozone = parse_array(&demo, "kOzoneCrossSection[48]");
    // Rows of (wavelength, x, y, z) every 5 nm from 360 nm.
    let cie = parse_array(&constants, "CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[380]");
    assert_eq!((solar.len(), ozone.len(), cie.len()), (BINS, BINS, 380));

    let lambda: Vec<f64> = (0..BINS).map(|i| FIRST_NM + BIN_NM * i as f64).collect();
    // Each bin's colour matching functions are sampled at its labelled wavelength.
    let cmf: Vec<[f64; 3]> = (0..BINS)
        .map(|i| {
            let row = 2 * i;
            assert_eq!(cie[4 * row], lambda[i]);
            [cie[4 * row + 1], cie[4 * row + 2], cie[4 * row + 3]]
        })
        .collect();
    // Rec.2020 colour matching functions.
    let rbar: Vec<[f64; 3]> = cmf
        .iter()
        .map(|&c| colour::mul(&XYZ_TO_REC2020, c))
        .collect();

    let illuminance: f64 = (0..BINS)
        .map(|i| LUMINOUS_EFFICACY * BIN_NM * solar[i] * cmf[i][1])
        .sum();
    // Each bin's Rec.2020 colour per lux of top-of-atmosphere solar illuminance.
    let bin_rec2020: Vec<[f64; 3]> = (0..BINS)
        .map(|i| rbar[i].map(|r| LUMINOUS_EFFICACY * BIN_NM * solar[i] * r / illuminance))
        .collect();
    let solar_colour = [0, 1, 2].map(|c| bin_rec2020.iter().map(|w| w[c]).sum::<f64>());

    let rayleigh: Vec<f64> = lambda
        .iter()
        .map(|l| RAYLEIGH_1UM / (l / 1000.0).powi(4))
        .collect();
    let aerosol = |i: usize, a: usize| (lambda[i] / 550.0).powf(-(a as f64 * ANGSTROM_STEP));

    let spectral = Bands {
        wavelength: lambda.clone(),
        rayleigh: rayleigh.clone(),
        ozone_cross_section: ozone.clone(),
        aerosol: (0..ANGSTROM_STEPS)
            .map(|a| (0..BINS).map(|i| aerosol(i, a)).collect())
            .collect(),
        rec2020: bin_rec2020.clone(),
    };

    // A band is a weighted mean over bins.
    let band = |w: &dyn Fn(usize) -> f64, rec2020: [f64; 3]| {
        let ws: f64 = (0..BINS).map(w).sum();
        let mean = |f: &dyn Fn(usize) -> f64| (0..BINS).map(|i| w(i) * f(i)).sum::<f64>() / ws;
        let wpos: f64 = (0..BINS).map(|i| w(i).max(0.0)).sum();
        BandRow {
            wavelength: (0..BINS).map(|i| w(i).max(0.0) * lambda[i]).sum::<f64>() / wpos,
            rayleigh: mean(&|i| rayleigh[i]),
            ozone_cross_section: mean(&|i| ozone[i]),
            aerosol: (0..ANGSTROM_STEPS)
                .map(|a| mean(&|i| aerosol(i, a)))
                .collect(),
            rec2020,
        }
    };

    let per_group = BINS / SPECTRAL_GROUPS;
    let groups = Bands::from_rows(
        (0..SPECTRAL_GROUPS)
            .map(|g| {
                let bins = g * per_group..(g + 1) * per_group;
                let rec2020 = [0, 1, 2].map(|c| bins.clone().map(|i| bin_rec2020[i][c]).sum());
                band(&|i| if bins.contains(&i) { solar[i] } else { 0.0 }, rec2020)
            })
            .collect(),
    );
    let rgb = Bands::from_rows(
        (0..3)
            .map(|c| {
                let mut rec2020 = [0.0; 3];
                rec2020[c] = solar_colour[c];
                band(&|i| solar[i] * rbar[i][c], rec2020)
            })
            .collect(),
    );

    let mut out = String::new();
    writeln!(out, "/// Wavelength groups in the LUT passes.").unwrap();
    writeln!(out, "pub const SPECTRAL_GROUPS: usize = {SPECTRAL_GROUPS};").unwrap();
    writeln!(out, "/// 10 nm bins of the spectral reference, 360-830 nm.").unwrap();
    writeln!(out, "pub const BINS: usize = {BINS};").unwrap();
    writeln!(out, "/// Entries in each band's aerosol table.").unwrap();
    writeln!(out, "pub const ANGSTROM_STEPS: usize = {ANGSTROM_STEPS};").unwrap();
    writeln!(out, "/// Angstrom exponent between aerosol table entries.").unwrap();
    writeln!(out, "pub const ANGSTROM_STEP: f64 = {ANGSTROM_STEP:?};").unwrap();
    writeln!(
        out,
        "/// Top-of-atmosphere solar illuminance from the spectrum, lux."
    )
    .unwrap();
    writeln!(out, "pub const SOLAR_ILLUMINANCE: f64 = {illuminance:?};").unwrap();
    writeln!(
        out,
        "/// Linear Rec.2020 colour of sunlight above the atmosphere, luminance 1."
    )
    .unwrap();
    writeln!(out, "pub const SOLAR_COLOUR: [f64; 3] = {solar_colour:?};").unwrap();
    spectral.write(&mut out, "SPECTRAL", "BINS", "The 10 nm bins.");
    groups.write(
        &mut out,
        "GROUPS",
        "SPECTRAL_GROUPS",
        "The wavelength groups of the LUT passes.",
    );
    rgb.write(
        &mut out,
        "RGB",
        "3",
        "The three solar-weighted Rec.2020 bands (red, green, blue).",
    );

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(out_dir.join("spectrum.rs"), out).unwrap();

    compile_kernels();
}

fn compile_kernels() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let roots = env::var("DEP_ORR_SHADER_DIRS").expect("orr's shader roots");
    let out = Build::new().and_then(|b| b.kernels(&src).roots(&roots).compile());
    match out {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
}

struct BandRow {
    wavelength: f64,
    rayleigh: f64,
    ozone_cross_section: f64,
    aerosol: Vec<f64>,
    rec2020: [f64; 3],
}

struct Bands {
    wavelength: Vec<f64>,
    rayleigh: Vec<f64>,
    ozone_cross_section: Vec<f64>,
    aerosol: Vec<Vec<f64>>,
    rec2020: Vec<[f64; 3]>,
}

impl Bands {
    fn from_rows(rows: Vec<BandRow>) -> Self {
        Bands {
            wavelength: rows.iter().map(|r| r.wavelength).collect(),
            rayleigh: rows.iter().map(|r| r.rayleigh).collect(),
            ozone_cross_section: rows.iter().map(|r| r.ozone_cross_section).collect(),
            aerosol: (0..ANGSTROM_STEPS)
                .map(|a| rows.iter().map(|r| r.aerosol[a]).collect())
                .collect(),
            rec2020: rows.iter().map(|r| r.rec2020).collect(),
        }
    }

    fn write(&self, out: &mut String, name: &str, len: &str, doc: &str) {
        writeln!(out, "/// {doc}").unwrap();
        writeln!(out, "pub const {name}: Bands<{len}> = Bands {{").unwrap();
        writeln!(out, "    wavelength: {:?},", self.wavelength).unwrap();
        writeln!(out, "    rayleigh: {:?},", self.rayleigh).unwrap();
        writeln!(
            out,
            "    ozone_cross_section: {:?},",
            self.ozone_cross_section
        )
        .unwrap();
        writeln!(out, "    aerosol: {:?},", self.aerosol).unwrap();
        writeln!(out, "    rec2020: {:?},", self.rec2020).unwrap();
        writeln!(out, "}};").unwrap();
    }
}

/// `ORR_SKY_DATA`, or `.tools/precomputed_atmospheric_scattering` in the nearest directory
/// above the crate that has it (where `tools/fetch-slang.sh` puts it).
fn data_dir() -> PathBuf {
    if let Some(dir) = env::var_os("ORR_SKY_DATA").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    manifest
        .ancestors()
        .map(|d| d.join(".tools").join(DATA_DIR))
        .find(|d| d.is_dir())
        .unwrap_or_else(|| {
            panic!(
                "spectral data not found: run tools/fetch-slang.sh, or set ORR_SKY_DATA to a \
                 checkout of ebruneton/precomputed_atmospheric_scattering"
            )
        })
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
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
