# Third-party licenses

orr is licensed under MIT OR Apache-2.0. The material below is not, or carries its own
notice, and keeps its own terms.

## CIE standard illuminant D65

CIE 2019, CIE standard illuminant D65, International Commission on Illumination (CIE),
Vienna, Austria. DOI: [10.25039/CIE.DS.hjfjmt59](https://doi.org/10.25039/CIE.DS.hjfjmt59).
Licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

`crates/orr_output/build/spectral_weights.rs` is adapted from this data and is offered
under the same license: sampled every 5 nm from 360 to 830 nm, multiplied by the CIE 1931
color matching functions, converted to Rec.2020 and normalized per channel.

`crates/orr_output/src/glare_tables.rs` and the glare FFT table generated at build time
are computed from that file.

## Precomputed Atmospheric Scattering

`crates/orr_atmosphere/src/spectrum_data.rs` holds measured spectra and the CIE 1931 color
matching functions, and `crates/orr_output/build/spectral_weights.rs` the color matching
functions, from <https://github.com/ebruneton/precomputed_atmospheric_scattering> at commit
`34f14e745cff948f4ca3157d1b62a445ffa7286f`.

> Copyright (c) 2017 Eric Bruneton
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this
> list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright notice,
> this list of conditions and the following disclaimer in the documentation
> and/or other materials provided with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its contributors
> may be used to endorse or promote products derived from this software without
> specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
