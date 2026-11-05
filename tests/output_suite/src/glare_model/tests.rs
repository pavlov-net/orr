use std::path::{Path, PathBuf};

use orr_color::RgbPrimaries;

use super::pyramid::{self, Plane};
use super::{TABLES, encircled_rgb, fft, fft_kernel, fit, psf};

/// `ORR_BLESS=1` rewrites orr_output's `src/glare_tables.rs`.
#[test]
fn the_committed_glare_tables_match_the_fit() {
    check_or_bless(
        "../../crates/orr_output/src/glare_tables.rs",
        &fit::source(),
    );
}

/// Skipped when `tools/fetch-slang.sh` has not fetched the files (`ORR_VALIDATION_DATA=1`).
#[test]
#[allow(clippy::print_stderr)]
fn the_spectral_weights_match_the_fetched_files() {
    let path = "../../crates/orr_output/build/spectral_weights.rs";
    let committed = include_str!("../../../../crates/orr_output/build/spectral_weights.rs");
    match spectral_weights_source(committed) {
        Some(generated) => check_or_bless(path, &generated),
        None => eprintln!("skipped: D65 or the color matching functions not fetched"),
    }
}

#[test]
fn tables_are_non_negative_and_keep_energy() {
    for first in &TABLES {
        for row in first {
            for c in 0..3 {
                assert!(row.iter().all(|w| w[c] >= 0.0), "{row:?}");
                let sum: f64 = row.iter().map(|w| w[c]).sum();
                assert!((sum - 1.0).abs() < 1e-9, "{sum}");
            }
        }
    }
}

#[test]
fn cpu_fit_keeps_uniform_fields_and_point_energy() {
    let w: Vec<f64> = TABLES[0][4].iter().map(|w| w[1]).collect();
    let size = [96, 64];
    let input = Plane::dense(size, vec![1.5; 96 * 64]);
    let downs = pyramid::downsamples(&input, 1);
    let out = pyramid::compose(&input, &downs, &pyramid::fold(&w, downs.len()));
    for v in &out.data {
        assert!((v / 1.5 - 1.0).abs() < 1e-9, "{v}");
    }
    let size = [4096; 2];
    let input = Plane::impulse(size, [2048, 2048], 1.0);
    let out = pyramid::compose(&input, &pyramid::downsamples(&input, 1), &w);
    assert!((out.sum() - 1.0).abs() < 0.01, "{}", out.sum());
}

/// The kernel's energy outside each radius against the PSF's, from 2 pixels out to 512,
/// at F-numbers across the range and both resolutions. Inside 8 pixels the lattice disc
/// the kernel is summed over differs in shape from the PSF's circle, which shows for the
/// spikes: up to 0.10 at f/22, where the kernel is the table itself.
#[test]
fn fft_kernel_matches_psf_encircled_energy() {
    let kernel = fft_kernel();
    let reach: i64 = 600;
    let (mut near, mut far): (f64, f64) = (0.0, 0.0);
    for first_level in [1, 2] {
        for n in [4.0, 5.6, 8.0, 11.0, 16.0, 22.0] {
            let u = fft::Kernel::texels(n, first_level);
            let pitch = f64::from(1u32 << first_level) * psf::REFERENCE_PITCH_UM;
            // Energy and pixel count per squared distance.
            let mut shells = std::collections::BTreeMap::<i64, ([f64; 3], u64)>::new();
            let mut total = [0.0; 3];
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let d2 = dx * dx + dy * dy;
                    if d2 > reach * reach {
                        continue;
                    }
                    let e = kernel.energy(dx, dy, u);
                    let s = shells.entry(d2).or_default();
                    for c in 0..3 {
                        s.0[c] += e[c];
                        total[c] += e[c];
                    }
                    s.1 += 1;
                }
            }
            let mut inside = [0.0; 3];
            let mut count = 0;
            let mut next = 2.0;
            for (&d2, &(e, k)) in &shells {
                for c in 0..3 {
                    inside[c] += e[c];
                }
                count += k;
                let r = (d2 as f64).sqrt();
                if r < next || r > 512.0 {
                    continue;
                }
                next = r * 2.0;
                let area_radius = (count as f64 / std::f64::consts::PI).sqrt();
                let psf = encircled_rgb(area_radius, n, pitch);
                for c in 0..3 {
                    let error = ((1.0 - inside[c]) / (1.0 - psf[c]) - 1.0).abs();
                    let worst = if r < 8.0 { &mut near } else { &mut far };
                    *worst = worst.max(error);
                }
            }
            for (total, kappa) in total.iter().zip(kernel.kappa) {
                // Each spike past the reach carries kappa / distance.
                let past = kappa * kernel.spikes.len() as f64 / (reach as f64 * u);
                assert!(
                    (total + past - 1.0).abs() < 1e-4,
                    "level {first_level} f/{n}: {total} inside, {past} past"
                );
            }
        }
    }
    log::info!(
        "fft kernel: energy outside a radius within {near:.3} of the PSF's inside 8 pixels, {far:.3} past"
    );
    assert!(near < 0.12 && far < 0.02, "{near}, {far}");
}

/// Checks the committed file `path` (from this package's directory) against `generated`, or with
/// `ORR_BLESS=1` rewrites it.
fn check_or_bless(path: &str, generated: &str) {
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
fn spectral_weights_source(committed: &str) -> Option<String> {
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
