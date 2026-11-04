// The measured inputs of `crate::spectrum`: per 10 nm bin from 360 to 830 nm, solar irradiance,
// ozone absorption cross section and the CIE 1931 2-degree color matching functions.
//
// Copied unchanged from Eric Bruneton's Precomputed Atmospheric Scattering,
// https://github.com/ebruneton/precomputed_atmospheric_scattering at commit
// 34f14e745cff948f4ca3157d1b62a445ffa7286f, `atmosphere/demo/demo.cc` and
// `atmosphere/constants.h`, which cite these sources:
//
// - Solar irradiance: "Reference Solar Spectral Irradiance: ASTM G-173", ETR column,
//   http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html, averaged over each bin.
// - Ozone: http://www.iup.uni-bremen.de/gruppen/molspec/databases/referencespectra/o3spectra2011/index.html
//   at 233 K, averaged over each bin.
// - Color matching functions: "CIE (1931) 2-deg color matching functions",
//   http://web.archive.org/web/20081228084047/http://www.cvrl.org/database/data/cmfs/ciexyz31.txt,
//   sampled at each bin's first wavelength.
//
// `crate::spectrum` derives everything else from these, for bin i at wavelength l[i] with
// irradiance S[i] and color matching functions (x[i], y[i], z[i]):
//
// - Solar illuminance E = sum(683 lm/W * 10 nm * S[i] * y[i]).
// - Bin color rec2020[i] = 683 lm/W * 10 nm * S[i] * r[i] / E, where
//   r[i] = (XYZ to Rec.2020) * (x[i], y[i], z[i]); the solar color is their sum.
// - Rayleigh scattering = 1.24062e-6 / m / (l[i] / 1 um)^4.
// - Aerosol factor = (l[i] / 550 nm)^-a, for a = 0, 0.1, ... 2.5.
// - A band's value is the weighted mean over its bins: by S[i] for a wavelength group, by
//   S[i] * r[i][c] for Rec.2020 channel c. Its wavelength is the mean of l[i] by the weights'
//   positive parts.
//
// That project's license:
//
// Copyright (c) 2017 Eric Bruneton
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/// Solar irradiance averaged over each bin, W m^-2 nm^-1.
pub(crate) const SOLAR_IRRADIANCE: [f64; 48] = [
    1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253, 1.91198, 2.03474,
    2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298, 1.8685, 1.8931, 1.85149, 1.8504, 1.8341,
    1.8345, 1.8147, 1.78158, 1.7533, 1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113,
    1.474, 1.4482, 1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082, 1.18737,
    1.14683, 1.12362, 1.1058, 1.07124, 1.04992,
];

/// Ozone absorption cross section at 233 K averaged over each bin, m^2.
pub(crate) const OZONE_CROSS_SECTION: [f64; 48] = [
    1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27, 8.451e-27,
    1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26, 1.48e-25, 1.602e-25,
    2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25, 4.672e-25, 4.398e-25, 4.701e-25,
    5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25, 2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25,
    1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26,
    2.451e-26, 2.801e-26, 2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27,
];

/// CIE 1931 2-degree x, y and z at each bin's first wavelength.
pub(crate) const COLOR_MATCHING: [[f64; 3]; 48] = [
    [0.000129900000, 0.000003917000, 0.000606100000], // 360 nm
    [0.000414900000, 0.000012390000, 0.001946000000], // 370 nm
    [0.001368000000, 0.000039000000, 0.006450001000], // 380 nm
    [0.004243000000, 0.000120000000, 0.020050010000], // 390 nm
    [0.014310000000, 0.000396000000, 0.067850010000], // 400 nm
    [0.043510000000, 0.001210000000, 0.207400000000], // 410 nm
    [0.134380000000, 0.004000000000, 0.645600000000], // 420 nm
    [0.283900000000, 0.011600000000, 1.385600000000], // 430 nm
    [0.348280000000, 0.023000000000, 1.747060000000], // 440 nm
    [0.336200000000, 0.038000000000, 1.772110000000], // 450 nm
    [0.290800000000, 0.060000000000, 1.669200000000], // 460 nm
    [0.195360000000, 0.090980000000, 1.287640000000], // 470 nm
    [0.095640000000, 0.139020000000, 0.812950100000], // 480 nm
    [0.032010000000, 0.208020000000, 0.465180000000], // 490 nm
    [0.004900000000, 0.323000000000, 0.272000000000], // 500 nm
    [0.009300000000, 0.503000000000, 0.158200000000], // 510 nm
    [0.063270000000, 0.710000000000, 0.078249990000], // 520 nm
    [0.165500000000, 0.862000000000, 0.042160000000], // 530 nm
    [0.290400000000, 0.954000000000, 0.020300000000], // 540 nm
    [0.433449900000, 0.994950100000, 0.008749999000], // 550 nm
    [0.594500000000, 0.995000000000, 0.003900000000], // 560 nm
    [0.762100000000, 0.952000000000, 0.002100000000], // 570 nm
    [0.916300000000, 0.870000000000, 0.001650001000], // 580 nm
    [1.026300000000, 0.757000000000, 0.001100000000], // 590 nm
    [1.062200000000, 0.631000000000, 0.000800000000], // 600 nm
    [1.002600000000, 0.503000000000, 0.000340000000], // 610 nm
    [0.854449900000, 0.381000000000, 0.000190000000], // 620 nm
    [0.642400000000, 0.265000000000, 0.000049999990], // 630 nm
    [0.447900000000, 0.175000000000, 0.000020000000], // 640 nm
    [0.283500000000, 0.107000000000, 0.000000000000], // 650 nm
    [0.164900000000, 0.061000000000, 0.000000000000], // 660 nm
    [0.087400000000, 0.032000000000, 0.000000000000], // 670 nm
    [0.046770000000, 0.017000000000, 0.000000000000], // 680 nm
    [0.022700000000, 0.008210000000, 0.000000000000], // 690 nm
    [0.011359160000, 0.004102000000, 0.000000000000], // 700 nm
    [0.005790346000, 0.002091000000, 0.000000000000], // 710 nm
    [0.002899327000, 0.001047000000, 0.000000000000], // 720 nm
    [0.001439971000, 0.000520000000, 0.000000000000], // 730 nm
    [0.000690078600, 0.000249200000, 0.000000000000], // 740 nm
    [0.000332301100, 0.000120000000, 0.000000000000], // 750 nm
    [0.000166150500, 0.000060000000, 0.000000000000], // 760 nm
    [0.000083075270, 0.000030000000, 0.000000000000], // 770 nm
    [0.000041509940, 0.000014990000, 0.000000000000], // 780 nm
    [0.000020673830, 0.000007465700, 0.000000000000], // 790 nm
    [0.000010253980, 0.000003702900, 0.000000000000], // 800 nm
    [0.000005085868, 0.000001836600, 0.000000000000], // 810 nm
    [0.000002522525, 0.000000910930, 0.000000000000], // 820 nm
    [0.000001251141, 0.000000451810, 0.000000000000], // 830 nm
];
