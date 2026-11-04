//! CPU models of orr_output's glare: the PSF and FFT kernel its build uses, the band pyramid, the
//! fit behind its committed tables, and the tables themselves.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

pub use encircled::encircled_rgb;
use orr_color::RgbPrimaries;

mod encircled;
#[path = "../../../crates/orr_output/build/fft.rs"]
pub mod fft;
pub mod fit;
#[allow(dead_code)]
#[path = "../../../crates/orr_output/src/glare_tables.rs"]
mod glare_tables;
mod kernel_energy;
#[path = "../../../crates/orr_output/build/psf.rs"]
pub mod psf;
pub mod pyramid;
#[path = "../../../crates/orr_output/build/spectral_weights.rs"]
pub mod spectral_weights;

mod fft_terms {
    #![allow(dead_code)]
    include!(concat!(env!("ORR_GLARE_FFT"), "/glare_fft.rs"));
}

/// The F-numbers of the tables.
pub static F_NUMBERS: LazyLock<[f64; fit::F_NUMBERS]> =
    LazyLock::new(|| std::array::from_fn(fit::f_number));

/// orr_output's committed band weights.
pub static TABLES: [[[[f64; 3]; pyramid::LAST_LEVEL as usize + 1]; fit::F_NUMBERS]; 2] =
    glare_tables::GLARE_TABLES;

/// The FFT kernel's table and spikes as the GPU reads them.
pub fn fft_kernel() -> fft::Kernel {
    let bytes = include_bytes!(concat!(env!("ORR_GLARE_FFT"), "/glare_fft_table.bin"));
    let floats: Vec<f64> = bytes
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
        .collect();
    let mut levels = Vec::new();
    let mut at = 0;
    for k in 0..fft::LEVELS {
        let n = fft::TABLE >> k;
        levels.push(
            (at..at + n * n)
                .map(|i| [floats[4 * i], floats[4 * i + 1], floats[4 * i + 2]])
                .collect(),
        );
        at += n * n;
    }
    assert_eq!(4 * at, floats.len());
    fft::Kernel {
        levels,
        kappa: fft_terms::GLARE_FFT_KAPPA.map(f64::from),
        spikes: fft_terms::GLARE_FFT_SPIKES
            .iter()
            .map(|s| s.map(f64::from))
            .collect(),
    }
}

/// Checks the committed file `path` (from this package's directory) against `generated`, or with
/// `ORR_BLESS=1` rewrites it.
pub fn check_or_bless(path: &str, generated: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    if std::env::var_os("ORR_BLESS").is_some_and(|v| v == "1") {
        std::fs::write(path, generated).unwrap();
        return;
    }
    let committed = std::fs::read_to_string(&path).unwrap();
    let different = generated
        .lines()
        .zip(committed.lines())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| generated.lines().count().min(committed.lines().count()));
    assert!(
        generated == committed,
        "{} differs (first at line {different}); ORR_BLESS=1 rewrites it",
        path.display()
    );
}

/// The text of orr_output's `build/spectral_weights.rs` with the header `committed` has, when
/// `tools/fetch-slang.sh` has fetched the files it comes from (`ORR_VALIDATION_DATA=1`).
pub fn spectral_weights_source(committed: &str) -> Option<String> {
    let tools = std::env::var_os("ORR_TOOLS")
        .filter(|t| !t.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .map(|d| d.join(".tools"))
                .find(|d| d.is_dir())
        })?;
    let constants = std::fs::read_to_string(
        tools.join("precomputed_atmospheric_scattering/atmosphere/constants.h"),
    )
    .ok()?;
    let d65 = std::fs::read_to_string(tools.join("cie/CIE_std_illum_D65.csv")).ok()?;
    let start = constants
        .find("CIE_2_DEG_COLOR_MATCHING_FUNCTIONS[380]")
        .unwrap();
    let open = start + constants[start..].find('{').unwrap();
    let close = open + constants[open..].find('}').unwrap();
    // Rows of (wavelength, x, y, z) every 5 nm from 360 nm.
    let cie: Vec<f64> = constants[open + 1..close]
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let illuminant: Vec<(f64, f64)> = d65
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
            let rgb = to_rec2020.mul_vec3([r[1], r[2], r[3]].into()).to_array();
            (r[0] / 1000.0, rgb.map(|c| c * s))
        })
        .collect();
    let sums = [0, 1, 2].map(|c| rows.iter().map(|r| r.1[c]).sum::<f64>());
    let table = "#[rustfmt::skip]\n";
    let mut s = committed[..committed.find(table).unwrap()].to_owned();
    s.push_str(table);
    s.push_str(&format!(
        "pub const SPECTRAL_WEIGHTS: [(f64, [f64; 3]); {}] = [\n",
        rows.len()
    ));
    for (lambda, w) in rows {
        let w = [0, 1, 2].map(|c| w[c] / sums[c]);
        s.push_str(&format!("    ({lambda:?}, {w:?}),\n"));
    }
    s.push_str("];\n");
    Some(s)
}
