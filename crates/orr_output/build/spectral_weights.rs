// Per wavelength (um, every 5 nm from 360 to 830 nm), each Rec.2020 channel's share of a D65
// white: the channel's color matching function (`orr_color::CIE_1931_CMF`) times D65
// (`d65.in`), normalized per channel over these samples. Negative where the channel's color
// matching function is.

use std::sync::LazyLock;

use orr_color::{CIE_1931_CMF, CIE_1931_CMF_FIRST_NM, CIE_1931_CMF_STEP_NM, RgbPrimaries};

/// CIE standard illuminant D65 at the wavelengths of [`CIE_1931_CMF`].
pub const D65: [f64; 95] = include!("d65.in");

pub static SPECTRAL_WEIGHTS: LazyLock<[(f64, [f64; 3]); 95]> = LazyLock::new(|| {
    let to_rec2020 = RgbPrimaries::REC2020.xyz_to_rgb();
    let rgb: [[f64; 3]; 95] = std::array::from_fn(|i| {
        let rgb = to_rec2020.mul_vec3(CIE_1931_CMF[i].into()).to_array();
        rgb.map(|c| c * D65[i])
    });
    let sums = [0, 1, 2].map(|c| rgb.iter().map(|r| r[c]).sum::<f64>());
    std::array::from_fn(|i| {
        let nm = CIE_1931_CMF_FIRST_NM + CIE_1931_CMF_STEP_NM * i as f64;
        (nm / 1000.0, [0, 1, 2].map(|c| rgb[i][c] / sums[c]))
    })
});
