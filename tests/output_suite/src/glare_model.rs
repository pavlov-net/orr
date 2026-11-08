//! CPU models of orr_output's glare: the PSF and FFT kernel its build uses, the band pyramid, the
//! fit behind its committed tables, and the tables themselves.

use std::sync::LazyLock;

pub use encircled::encircled_rgb;

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
#[cfg(test)]
mod tests;

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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f64::from(f32::from_le_bytes(b)))
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
