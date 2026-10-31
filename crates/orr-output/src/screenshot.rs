//! Screenshots: what a capture requests, the captured floats and the file writers.

use std::fs::File;
use std::io::{self, BufWriter};
use std::path::Path;

use crate::encode::{pq_encode, srgb_encode};

/// A capture of one view, made in the frame that sets it in `ViewOutput::screenshot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenshotRequest {
    /// What the display shows: display-linear Rec.709, 1.0 = paper white.
    Display { include_ui: bool },
    /// After `hooks.scene_referred`, before camera effects: absolute luminance in cd/m2,
    /// linear Rec.2020, or Rec.709 with `rec709`. EXR only.
    SceneReferred { rec709: bool },
}

/// The primaries of a capture's linear RGB; both use the D65 white.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Primaries {
    Rec709,
    Rec2020,
}

/// A captured image, read back one to three frames after the frame it holds.
#[derive(Clone, Debug)]
pub struct Screenshot {
    /// orr's frame id of the captured frame.
    pub frame: u64,
    pub size: [u32; 2],
    /// Row-major linear RGBA, top row first; alpha is 1. Display captures are relative to
    /// paper white, scene-referred ones in cd/m2.
    pub rgba: Vec<f32>,
    pub scene_referred: bool,
    pub primaries: Primaries,
    /// The display's paper white when captured.
    pub paper_white_nits: f32,
    /// The display was HDR (PQ or scRGB).
    pub hdr: bool,
}

/// MaxCLL is taken at this percentile of the pixels' brightest channel, so a few hot pixels
/// do not set it.
const MAX_CLL_PERCENTILE: f64 = 0.9999;

impl Screenshot {
    fn pixels(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        self.rgba.chunks_exact(4).map(|p| [p[0], p[1], p[2]])
    }

    fn display_only(&self, what: &str) -> io::Result<()> {
        if self.scene_referred {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("a scene-referred capture has no {what}; save it as EXR"),
            ));
        }
        Ok(())
    }

    /// Writes an 8-bit sRGB PNG. HDR captures clip at paper white. Errors on a
    /// scene-referred capture.
    pub fn save_png_srgb(&self, path: &Path) -> io::Result<()> {
        self.display_only("sRGB PNG")?;
        let data: Vec<u8> = self
            .pixels()
            .flat_map(|p| {
                p.map(|c| quantise(srgb_encode(f64::from(c).clamp(0.0, 1.0)), 255.0) as u8)
            })
            .collect();
        let mut encoder = self.png_encoder(path, png::BitDepth::Eight, None)?;
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let mut writer = encoder.write_header().map_err(io::Error::other)?;
        writer.write_image_data(&data).map_err(io::Error::other)?;
        writer.finish().map_err(io::Error::other)
    }

    /// Writes a 16-bit Rec.2100 PQ PNG with `cICP`, a Rec.2100 PQ ICC profile in `iCCP` and
    /// `cLLI` (MaxCLL at the 99.99th percentile of each pixel's brightest channel, MaxFALL
    /// the mean). Errors on a scene-referred capture.
    pub fn save_png_pq(&self, path: &Path) -> io::Result<()> {
        self.display_only("PQ PNG")?;
        let pw = f64::from(self.paper_white_nits);
        let nits: Vec<[f64; 3]> = self
            .pixels()
            .map(|p| {
                mul(&REC709_TO_REC2020, p.map(f64::from)).map(|c| (c * pw).clamp(0.0, 10000.0))
            })
            .collect();
        let data: Vec<u8> = nits
            .iter()
            .flat_map(|p| p.map(|c| quantise(pq_encode(c), 65535.0) as u16))
            .flat_map(u16::to_be_bytes)
            .collect();
        let (max_cll, max_fall) = content_light_level(&nits);

        let encoder = self.png_encoder(path, png::BitDepth::Sixteen, Some(pq_icc_profile()?))?;
        let mut writer = encoder.write_header().map_err(io::Error::other)?;
        // Rec.2020 primaries, PQ, RGB, full range.
        writer
            .write_chunk(png::chunk::cICP, &[9, 16, 0, 1])
            .map_err(io::Error::other)?;
        let units = |nits: f64| ((nits * 10000.0).round() as u32).to_be_bytes();
        let clli = [units(max_cll), units(max_fall)].concat();
        writer
            .write_chunk(png::chunk::cLLI, &clli)
            .map_err(io::Error::other)?;
        writer.write_image_data(&data).map_err(io::Error::other)?;
        writer.finish().map_err(io::Error::other)
    }

    /// Writes a float RGB OpenEXR with its primaries as the `chromaticities` attribute and
    /// `whiteLuminance` the nits of 1.0: paper white for a display capture, 1 for a
    /// scene-referred one, which holds cd/m2.
    pub fn save_exr(&self, path: &Path) -> io::Result<()> {
        use exr::prelude::{Image, SpecificChannels, Vec2, WritableImage};
        let [w, h] = self.size.map(|v| v as usize);
        let rgba = &self.rgba;
        let channels = SpecificChannels::rgb(|Vec2(x, y): Vec2<usize>| {
            let p = &rgba[4 * (y * w + x)..];
            (p[0], p[1], p[2])
        });
        let mut image = Image::from_channels((w, h), channels);
        let (r, g, b) = match self.primaries {
            Primaries::Rec709 => ([0.64, 0.33], [0.30, 0.60], [0.15, 0.06]),
            Primaries::Rec2020 => ([0.708, 0.292], [0.170, 0.797], [0.131, 0.046]),
        };
        let xy = |c: [f32; 2]| Vec2(c[0], c[1]);
        image.attributes.chromaticities = Some(exr::meta::attribute::Chromaticities {
            red: xy(r),
            green: xy(g),
            blue: xy(b),
            white: xy([0.3127, 0.3290]),
        });
        image.layer_data.attributes.white_luminance = Some(if self.scene_referred {
            1.0
        } else {
            self.paper_white_nits
        });
        image
            .write()
            .non_parallel()
            .to_file(path)
            .map_err(io::Error::other)
    }

    fn png_encoder(
        &self,
        path: &Path,
        depth: png::BitDepth,
        icc_profile: Option<Vec<u8>>,
    ) -> io::Result<png::Encoder<'static, BufWriter<File>>> {
        let file = BufWriter::new(File::create(path)?);
        let mut info = png::Info::with_size(self.size[0], self.size[1]);
        info.color_type = png::ColorType::Rgb;
        info.bit_depth = depth;
        info.icc_profile = icc_profile.map(Into::into);
        png::Encoder::with_info(file, info).map_err(io::Error::other)
    }
}

fn quantise(signal: f64, max: f64) -> f64 {
    (signal * max).round().clamp(0.0, max)
}

/// MaxCLL and MaxFALL of Rec.2020 pixels in nits.
fn content_light_level(nits: &[[f64; 3]]) -> (f64, f64) {
    if nits.is_empty() {
        return (0.0, 0.0);
    }
    let mut brightest: Vec<f64> = nits.iter().map(|p| p[0].max(p[1]).max(p[2])).collect();
    let fall = brightest.iter().sum::<f64>() / brightest.len() as f64;
    let rank = ((MAX_CLL_PERCENTILE * brightest.len() as f64).ceil() as usize)
        .clamp(1, brightest.len())
        - 1;
    let (_, cll, _) = brightest.select_nth_unstable_by(rank, f64::total_cmp);
    (*cll, fall)
}

/// An ICC v4 Rec.2100 PQ profile whose `cicp` tag gives the same code points as `cICP`.
fn pq_icc_profile() -> io::Result<Vec<u8>> {
    use moxcms::{
        CicpColorPrimaries, CicpProfile, ColorProfile, MatrixCoefficients, TransferCharacteristics,
    };
    let mut profile = ColorProfile::new_bt2020_pq();
    profile.cicp = Some(CicpProfile {
        color_primaries: CicpColorPrimaries::Bt2020,
        transfer_characteristics: TransferCharacteristics::Smpte2084,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    });
    profile.encode().map_err(io::Error::other)
}

const REC709_TO_REC2020: [[f64; 3]; 3] = [
    [
        0.627_403_895_934_699,
        0.329_283_038_377_884,
        0.043_313_065_687_417,
    ],
    [
        0.069_097_289_358_232,
        0.919_540_395_075_458,
        0.011_362_315_566_310,
    ],
    [
        0.016_391_438_875_150,
        0.088_013_307_877_226,
        0.895_595_253_247_624,
    ],
];

fn mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    m.map(|r| r[0] * v[0] + r[1] * v[1] + r[2] * v[2])
}
