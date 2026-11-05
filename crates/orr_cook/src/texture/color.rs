//! Color in image files: what a file states ([`FileColor`]), what an importer's rules make of
//! that ([`ImageColor`]), and how a stored texel then reads ([`Read`]).
//!
//! The rules differ by format, so each importer states its own:
//! - glTF ignores what a file states: color textures are sRGB Rec.709, and other textures are
//!   numbers. Floats and signed values stay linear, because the sRGB curve doesn't encode them.
//! - USD reads `sourceColorSpace`: `sRGB`, `raw`, or `auto`, which is what the file states, else
//!   sRGB for 8 bits in 3 or 4 channels and linear for the rest. The rule is per file, so it holds
//!   for every input reading the file.
//! - MaterialX names a space for color types, and ignores what the file states.
//! - A plain file is what it states, else sRGB unless the curve can't encode what it stores.
//!
//! Primaries with a cooked id stay as stored, tagged with the id. Any others convert to Rec.2020,
//! the working space, whose gamut is the widest of the ids.

use std::fmt;
use std::sync::Arc;

use moxcms::{ColorProfile, DataColorSpace, Layout, ToneReprCurve, TransformOptions};
use orr_color::{
    Chromaticity, Primaries, RgbPrimaries, hlg_decode, pq_decode, rec709_to_rec2020, srgb_decode,
};
use orr_math::{DMat3, DVec3};

use super::mips::settled;

/// A stored signal to linear light.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Transfer {
    Linear,
    Srgb,
    /// `signal ^ exponent`.
    Gamma(f32),
    /// The inverse of the ITU-R BT.709 OETF.
    Itu,
    St240,
    /// H.273 logarithmic transfer over 2 decades.
    Log100,
    /// H.273 logarithmic transfer over 2.5 decades.
    Log316,
    /// SMPTE ST 428 and DCI, 1 at the 48 nit reference white.
    Dci,
    Pq,
    Hlg,
    SLog,
    SLog2,
    AcesCc,
    AcesCct,
}

impl Transfer {
    pub(crate) fn decode(self, signal: f32) -> f32 {
        // ITU-R BT.2408 puts HDR reference white at 203 nits and 75% HLG, where SDR has 1.
        const PQ_WHITE_NITS: f64 = 203.0;
        const HLG_WHITE_SIGNAL: f64 = 0.75;
        let v = f64::from(signal);
        let log2 = |v: f64| (v * 17.52 - 9.72).exp2();
        let slog = |v: f64| 10f64.powf((v - 0.646596) / 0.432699) - 0.037584;
        let linear = match self {
            Self::Linear => return signal,
            Self::Srgb => srgb_decode(v),
            Self::Gamma(g) => v.max(0.0).powf(f64::from(g)),
            Self::Itu if v < 0.081 => v / 4.5,
            Self::Itu => ((v + 0.099) / 1.099).powf(1.0 / 0.45),
            Self::St240 if v < 0.0913 => v / 4.0,
            Self::St240 => ((v + 0.1115) / 1.1115).powf(1.0 / 0.45),
            Self::Log100 | Self::Log316 if v <= 0.0 => 0.0,
            Self::Log100 => 10f64.powf(2.0 * (v - 1.0)),
            Self::Log316 => 10f64.powf(2.5 * (v - 1.0)),
            Self::Dci => v.max(0.0).powf(2.6) * 52.37 / 48.0,
            Self::Pq => pq_decode(v) / PQ_WHITE_NITS,
            Self::Hlg => hlg_decode(v) / hlg_decode(HLG_WHITE_SIGNAL),
            Self::SLog => slog(v),
            Self::SLog2 if v < 0.030_001_222_851_889_303 => {
                (v - 0.030_001_222_851_889_303) / 3.538_812_785_388_13
            }
            Self::SLog2 => slog(v) * 219.0 / 155.0,
            Self::AcesCc if v <= (9.72 - 15.0) / 17.52 => (log2(v) - 2f64.powi(-16)) * 2.0,
            Self::AcesCc => log2(v).min(65504.0),
            Self::AcesCct if v <= 0.155_251_141_552_511 => {
                (v - 0.072_905_534_195_835_5) / 10.540_237_741_654_5
            }
            Self::AcesCct => log2(v),
        };
        linear as f32
    }
}

/// The gamut of a file's linear values.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Gamut {
    Rgb(RgbPrimaries),
    /// CIE XYZ, which needs no adaptation: its values state a color, not a mix of primaries.
    Xyz,
}

impl Gamut {
    pub(crate) const REC709: Self = Self::Rgb(RgbPrimaries::REC709);

    fn matrix_to(self, dst: RgbPrimaries) -> DMat3 {
        match self {
            Self::Rgb(p) => p.matrix_to(dst),
            Self::Xyz => dst.xyz_to_rgb(),
        }
    }

    /// The id of these primaries, if they have one.
    fn id(self) -> Option<Primaries> {
        match self {
            Self::Rgb(p) => Primaries::matching(p),
            Self::Xyz => None,
        }
    }

    /// Their id, with the matrix into that id's primaries where texels must convert.
    fn cooked(self) -> (Primaries, Option<DMat3>) {
        match self.id() {
            Some(p) => (p, None),
            None => (
                Primaries::Rec2020,
                Some(self.matrix_to(RgbPrimaries::REC2020)),
            ),
        }
    }
}

/// `m * rgb`. Infinite channels enter as the largest finite values: under a row of mixed signs
/// they would cancel to NaN, and the brightest texel would read as nothing.
pub(crate) fn multiply(m: &DMat3, rgb: [f32; 3]) -> [f32; 3] {
    let rgb = rgb.map(|v| f64::from(settled(v, f32::MIN, f32::MAX)));
    (*m * DVec3::from(rgb)).as_vec3().to_array()
}

/// A named color space.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Space {
    pub(crate) transfer: Transfer,
    pub(crate) gamut: Gamut,
}

impl Space {
    pub(crate) const SRGB: Self = Self {
        transfer: Transfer::Srgb,
        gamut: Gamut::REC709,
    };

    /// `rgb` in this space as the linear Rec.709 of material fields.
    pub(crate) fn to_rec709(self, rgb: [f32; 3]) -> [f32; 3] {
        let linear = DVec3::from(rgb.map(|v| f64::from(self.transfer.decode(v))));
        (self.gamut.matrix_to(RgbPrimaries::REC709) * linear)
            .as_vec3()
            .to_array()
    }
}

/// What an importer's rules say an image's texels are.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum ImageColor {
    /// Numbers, read as stored whatever the file states.
    Data,
    /// A space, whatever the file states.
    Named(Space),
    /// sRGB Rec.709 whatever the file states, where the curve can encode what it stores.
    Gltf,
    /// `auto` of USD's `sourceColorSpace`.
    UsdAuto,
    /// What a plain file states.
    File,
}

impl ImageColor {
    /// The space of a name in the ASWF Color Interop Forum's recommendation, which MaterialX, USD
    /// and OpenEXR's `colorInteropID` share, or of the names MaterialX deprecated.
    pub(crate) fn named(name: &str) -> Option<Self> {
        use Transfer::{Gamma, Linear, Srgb};
        const ADOBE_GAMMA: f32 = 563.0 / 256.0;
        let space = |transfer, primaries| {
            Some(Self::Named(Space {
                transfer,
                gamut: Gamut::Rgb(primaries),
            }))
        };
        let (rec709, p3) = (RgbPrimaries::REC709, RgbPrimaries::DISPLAY_P3);
        let (ap1, adobe) = (RgbPrimaries::ACES_CG, RgbPrimaries::ADOBE_RGB);
        match name {
            "data" | "none" | "raw" | "identity" => Some(Self::Data),
            "lin_ap1_scene" | "acescg" | "lin_ap1" => space(Linear, ap1),
            "lin_ap0_scene" => space(Linear, RgbPrimaries::ACES_AP0),
            "lin_rec709_scene" | "lin_rec709" => space(Linear, rec709),
            "lin_p3d65_scene" | "lin_displayp3" => space(Linear, p3),
            "lin_rec2020_scene" => space(Linear, RgbPrimaries::REC2020),
            "lin_adobergb_scene" | "lin_adobergb" => space(Linear, adobe),
            "lin_ciexyzd65_scene" => Some(Self::Named(Space {
                transfer: Linear,
                gamut: Gamut::Xyz,
            })),
            "srgb_rec709_scene" | "srgb_texture" => space(Srgb, rec709),
            "g24_rec709_scene" | "rec709_display" => space(Gamma(2.4), rec709),
            "g22_rec709_scene" | "g22_rec709" => space(Gamma(2.2), rec709),
            "g18_rec709_scene" | "g18_rec709" => space(Gamma(1.8), rec709),
            "srgb_ap1_scene" => space(Srgb, ap1),
            "g22_ap1_scene" | "g22_ap1" => space(Gamma(2.2), ap1),
            "srgb_p3d65_scene" | "srgb_displayp3" => space(Srgb, p3),
            "g22_adobergb_scene" | "adobergb" => space(Gamma(ADOBE_GAMMA), adobe),
            _ => None,
        }
    }

    fn transfer(self, f: &FileColor) -> Transfer {
        let unstated = if f.numeric {
            Transfer::Linear
        } else {
            Transfer::Srgb
        };
        match self {
            Self::Data => Transfer::Linear,
            Self::Named(s) => s.transfer,
            Self::Gltf => unstated,
            Self::File => f.transfer.unwrap_or(unstated),
            Self::UsdAuto => match f.transfer {
                // Hio's reading of a gamma hint, as a file's gamma: near 1/2.2 is sRGB and near 1
                // is linear.
                Some(Transfer::Gamma(g)) if (1.0 / g - 0.45455).abs() < 0.1 => Transfer::Srgb,
                Some(Transfer::Gamma(g)) if (1.0 / g - 1.0).abs() < 0.1 => Transfer::Linear,
                Some(t) => t,
                None if f.rgb8 => Transfer::Srgb,
                None => Transfer::Linear,
            },
        }
    }

    fn honors_file(self) -> bool {
        matches!(self, Self::File | Self::UsdAuto)
    }

    fn gamut(self, f: &FileColor) -> Gamut {
        match self {
            Self::Named(s) => s.gamut,
            Self::File | Self::UsdAuto => f.gamut.unwrap_or(Gamut::REC709),
            Self::Data | Self::Gltf => Gamut::REC709,
        }
    }

    /// How a number in file `f` reads: an input that isn't a color takes the curve the rules
    /// state, and no primaries.
    pub(crate) fn decode(self, f: &FileColor) -> Decode {
        Decode {
            narrow: f.narrow.filter(|_| self.honors_file()),
            transfer: self.transfer(f),
        }
    }

    /// How a color in file `f` reads, and the id and matrix of what it reads as. Primaries that
    /// `over` sets stand in for what the rules say, and take the file's texels as they are.
    pub(crate) fn color(
        self,
        f: &FileColor,
        over: Option<Primaries>,
    ) -> (Read, Primaries, Option<DMat3>) {
        let icc = f
            .icc
            .clone()
            .filter(|_| self.honors_file() && over.is_none());
        let (primaries, matrix) = match (over, &icc) {
            (Some(p), _) => (p, None),
            (None, Some(_)) => (Primaries::Rec2020, None),
            (None, None) => self.gamut(f).cooked(),
        };
        let read = Read {
            decode: self.decode(f),
            icc,
        };
        (read, primaries, matrix)
    }
}

/// How one stored value reads, with no other channel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Decode {
    /// `[a, b]` of the `a + b * stored` that takes a narrow-range signal to 0..1.
    pub(crate) narrow: Option<[f32; 2]>,
    pub(crate) transfer: Transfer,
}

impl Decode {
    pub(crate) const STORED: Self = Self {
        narrow: None,
        transfer: Transfer::Linear,
    };

    pub(crate) const SRGB: Self = Self {
        narrow: None,
        transfer: Transfer::Srgb,
    };

    fn signal(self, stored: f32) -> f32 {
        self.narrow.map_or(stored, |[a, b]| a + b * stored)
    }

    pub(crate) fn value(self, stored: f32) -> f32 {
        self.transfer.decode(self.signal(stored))
    }
}

/// How a texel's channels read.
#[derive(Clone, Debug)]
pub(crate) struct Read {
    pub(crate) decode: Decode,
    /// Stands in for `decode`'s transfer on red, green and blue.
    pub(crate) icc: Option<Icc>,
}

impl Read {
    pub(crate) fn of(decode: Decode) -> Self {
        Self { decode, icc: None }
    }

    /// Whether this is `decode` alone, so that blocks stored under `decode` hold what it reads.
    pub(crate) fn is(&self, decode: Decode) -> bool {
        self.decode == decode && self.icc.is_none()
    }

    /// The color of stored texel `t`, through `gamut`, with its alpha.
    pub(crate) fn color(&self, t: [f32; 4], gamut: Option<&DMat3>) -> [f32; 4] {
        let rgb = match &self.icc {
            Some(icc) => icc.linear([0, 1, 2].map(|c| self.decode.signal(t[c]))),
            None => [0, 1, 2].map(|c| self.decode.value(t[c])),
        };
        let [r, g, b] = gamut.map_or(rgb, |m| multiply(m, rgb));
        [r, g, b, t[3]]
    }

    /// Channel `c` of stored texel `t`, as a channel of color.
    pub(crate) fn value(&self, t: [f32; 4], c: usize) -> f32 {
        if self.icc.is_some() && c < 3 {
            return self.color(t, None)[c];
        }
        self.decode.value(t[c])
    }
}

/// What an image file states of its color, with what its format says where that bears on it.
#[derive(Clone, Debug, Default)]
pub(crate) struct FileColor {
    pub(crate) transfer: Option<Transfer>,
    pub(crate) gamut: Option<Gamut>,
    /// See [`Decode::narrow`].
    pub(crate) narrow: Option<[f32; 2]>,
    pub(crate) icc: Option<Icc>,
    /// It stores floats or signed values, which the sRGB curve doesn't encode.
    pub(crate) numeric: bool,
    /// It stores 8 bits in each of 3 or 4 channels.
    pub(crate) rgb8: bool,
}

/// An ICC profile's transform to linear Rec.2020.
#[derive(Clone)]
pub(crate) struct Icc {
    transform: Arc<moxcms::TransformF32Executor>,
    gray: bool,
}

impl fmt::Debug for Icc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Icc")
    }
}

impl Icc {
    /// `rgb` in the profile as linear Rec.2020. A gray profile reads red, which gray files
    /// repeat over RGB.
    fn linear(&self, rgb: [f32; 3]) -> [f32; 3] {
        let n = if self.gray { 1 } else { 3 };
        let mut out = [0.0; 3];
        self.transform
            .transform(&rgb[..n], &mut out)
            .expect("one pixel of each layout");
        out
    }
}

/// States on `f` what ICC profile `bytes` says of an image of RGB or `gray`. A profile for
/// neither, or one that doesn't parse, states nothing, and `false` says so. A profile that reads
/// as sRGB states that, with no transform: most profiles are sRGB, and their blocks and ids stay.
pub(crate) fn read_icc(f: &mut FileColor, bytes: &[u8], gray: bool) -> bool {
    let Some(icc) = icc_transform(bytes, gray) else {
        return false;
    };
    const STEPS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
    let srgb = STEPS.iter().all(|&r| {
        STEPS.iter().all(|&g| {
            STEPS.iter().all(|&b| {
                let (r, g, b) = if gray { (r, r, r) } else { (r, g, b) };
                let want = rec709_to_rec2020([r, g, b].map(|v| srgb_decode(f64::from(v))));
                let got = DVec3::from(icc.linear([r, g, b]).map(f64::from));
                (got - want).abs().max_element() < 2e-3
            })
        })
    });
    if srgb {
        f.transfer = Some(Transfer::Srgb);
        f.gamut = Some(Gamut::REC709);
        return true;
    }
    // Inputs that aren't colors take no primaries, so they read the curve at middle gray.
    let weights = RgbPrimaries::REC2020.luminance_weights();
    let y = weights.dot(DVec3::from(icc.linear([0.5; 3]).map(f64::from)));
    let gamma = y.ln() / 0.5f64.ln();
    f.transfer = (gamma.is_finite() && gamma > 0.0).then_some(Transfer::Gamma(gamma as f32));
    f.icc = Some(icc);
    true
}

fn icc_transform(bytes: &[u8], gray: bool) -> Option<Icc> {
    let profile = ColorProfile::new_from_slice(bytes).ok()?;
    let stated = match profile.color_space {
        DataColorSpace::Gray => true,
        DataColorSpace::Rgb => false,
        _ => return None,
    };
    if stated != gray {
        return None;
    }
    let linear = Some(ToneReprCurve::Lut(Vec::new()));
    let mut target = ColorProfile::new_bt2020();
    target.red_trc.clone_from(&linear);
    target.green_trc.clone_from(&linear);
    target.blue_trc = linear;
    let layout = if gray { Layout::Gray } else { Layout::Rgb };
    let transform = profile
        .create_transform_f32(layout, &target, Layout::Rgb, TransformOptions::default())
        .ok()?;
    Some(Icc { transform, gray })
}

/// RGB primaries at `[x, y]` of red, green, blue and white.
fn chromaticities(rgbw: [[f64; 2]; 4]) -> Gamut {
    let [red, green, blue, white] = rgbw.map(|[x, y]| Chromaticity::new(x, y));
    Gamut::Rgb(RgbPrimaries {
        red,
        green,
        blue,
        white,
    })
}

/// [`chromaticities`] a file writes out, which say nothing where no RGB space has them.
pub(crate) fn stated_gamut(rgbw: [[f64; 2]; 4]) -> Option<Gamut> {
    let gamut = chromaticities(rgbw);
    let finite = gamut.matrix_to(RgbPrimaries::REC2020).is_finite();
    finite.then_some(gamut)
}

/// CIE illuminant C, of the older television systems.
const WHITE_C: [f64; 2] = [0.310, 0.316];
const D65: [f64; 2] = [0.3127, 0.3290];

/// BT.470 M, the NTSC of 1953.
fn ntsc_1953() -> Gamut {
    chromaticities([[0.67, 0.33], [0.21, 0.71], [0.14, 0.08], WHITE_C])
}

/// BT.601 for 625 lines.
fn ebu() -> Gamut {
    chromaticities([[0.64, 0.33], [0.29, 0.60], [0.15, 0.06], D65])
}

/// BT.601 for 525 lines, under `white`.
fn smpte(white: [f64; 2]) -> Gamut {
    chromaticities([[0.63, 0.34], [0.31, 0.595], [0.155, 0.07], white])
}

/// The gamut of a KTX2 DFD `colorPrimaries`.
pub(crate) fn ktx2_gamut(p: ktx2::ColorPrimaries) -> Option<Gamut> {
    use ktx2::ColorPrimaries as P;
    Some(match p {
        P::BT709 => Gamut::REC709,
        P::BT601EBU => ebu(),
        P::BT601SMPTE => smpte(D65),
        P::BT2020 => Gamut::Rgb(RgbPrimaries::REC2020),
        P::CIEXYZ => Gamut::Xyz,
        P::ACES => Gamut::Rgb(RgbPrimaries::ACES_AP0),
        P::ACESCC => Gamut::Rgb(RgbPrimaries::ACES_CG),
        P::NTSC1953 => ntsc_1953(),
        P::PAL525 => smpte([0.3101, 0.3162]),
        P::DISPLAYP3 => Gamut::Rgb(RgbPrimaries::DISPLAY_P3),
        P::AdobeRGB => Gamut::Rgb(RgbPrimaries::ADOBE_RGB),
        _ => return None,
    })
}

/// The transfer of a KTX2 DFD `transferFunction`, as the Khronos Data Format Specification
/// defines each. A signal reads to scene light through an OETF's inverse and display light
/// through an EOTF; both PQ and both HLG values read as one curve, which keeps white at 1.
pub(crate) fn ktx2_transfer(t: ktx2::TransferFunction) -> Option<Transfer> {
    use ktx2::TransferFunction as T;
    Some(match t {
        T::Linear => Transfer::Linear,
        T::SRGB => Transfer::Srgb,
        T::ITU => Transfer::Itu,
        T::NTSC => Transfer::Gamma(2.2),
        T::SLOG => Transfer::SLog,
        T::SLOG2 => Transfer::SLog2,
        T::BT1886 => Transfer::Gamma(2.4),
        T::HLGOETF | T::HLGEOTF => Transfer::Hlg,
        T::PQEOTF | T::PQOETF => Transfer::Pq,
        T::DCIP3 => Transfer::Dci,
        T::PALOETF => Transfer::Gamma(2.5),
        T::PAL625EOTF => Transfer::Gamma(2.8),
        T::ST240 => Transfer::St240,
        T::ACESCC => Transfer::AcesCc,
        T::ACESCCT => Transfer::AcesCct,
        T::AdobeRGB => Transfer::Gamma(563.0 / 256.0),
        _ => return None,
    })
}

/// The gamut of an H.273 primaries code; values it leaves unspecified or reserved say nothing.
pub(crate) fn cicp_gamut(p: u8) -> Option<Gamut> {
    Some(match p {
        1 => Gamut::REC709,
        4 => ntsc_1953(),
        5 => ebu(),
        6 | 7 => smpte(D65),
        8 => chromaticities([
            [0.681, 0.319],
            [0.243, 0.692],
            [0.145, 0.049],
            [0.310, 0.316],
        ]),
        9 => Gamut::Rgb(RgbPrimaries::REC2020),
        10 => Gamut::Xyz,
        11 => chromaticities([
            [0.680, 0.320],
            [0.265, 0.690],
            [0.150, 0.060],
            [0.314, 0.351],
        ]),
        12 => Gamut::Rgb(RgbPrimaries::DISPLAY_P3),
        22 => chromaticities([[0.630, 0.340], [0.295, 0.605], [0.155, 0.077], D65]),
        _ => return None,
    })
}

/// The transfer of an H.273 `TransferCharacteristics`; values it leaves unspecified or reserved
/// say nothing. The extended ranges of 11 and 12 read as 1 does over the 0..1 a file stores.
pub(crate) fn cicp_transfer(t: u8) -> Option<Transfer> {
    Some(match t {
        1 | 6 | 11 | 12 | 14 | 15 => Transfer::Itu,
        4 => Transfer::Gamma(2.2),
        5 => Transfer::Gamma(2.8),
        7 => Transfer::St240,
        8 => Transfer::Linear,
        9 => Transfer::Log100,
        10 => Transfer::Log316,
        13 => Transfer::Srgb,
        16 => Transfer::Pq,
        17 => Transfer::Dci,
        18 => Transfer::Hlg,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piecewise_transfers_meet_at_their_joins() {
        let at = |t: Transfer, v: f32| (t.decode(v - 1e-6), t.decode(v + 1e-6));
        for (t, join) in [
            (Transfer::Itu, 0.081),
            (Transfer::St240, 0.0913),
            (Transfer::SLog2, 0.030_001_22),
            (Transfer::AcesCc, (9.72 - 15.0) / 17.52),
            (Transfer::AcesCct, 0.155_251_14),
        ] {
            let (below, above) = at(t, join);
            assert!((below - above).abs() < 1e-4, "{t:?}: {below} {above}");
        }
    }

    #[test]
    fn hdr_transfers_put_reference_white_at_1() {
        assert!((Transfer::Hlg.decode(0.75) - 1.0).abs() < 1e-5);
        let pq_203 = orr_color::pq_encode(203.0) as f32;
        assert!((Transfer::Pq.decode(pq_203) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn unlisted_primaries_cook_to_rec2020_and_keep_neutrals() {
        let (id, m) = Gamut::Rgb(RgbPrimaries::ACES_AP0).cooked();
        assert_eq!(id, Primaries::Rec2020);
        let white = multiply(&m.unwrap(), [1.0; 3]);
        assert!(white.iter().all(|c| (c - 1.0).abs() < 1e-5), "{white:?}");
        let hottest = multiply(&m.unwrap(), [f32::INFINITY; 3]);
        assert!(hottest.iter().all(|&c| c > 1e30), "{hottest:?}");
    }
}
