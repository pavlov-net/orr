use std::sync::{Arc, Mutex, PoisonError};

use orr_scene::orr_gpu::{HdrMetadata, OutputFormat, PresentMode};

/// GT7's SDR calibration: SDR paper white and peak, and the default HDR paper white.
const SDR_WHITE_NITS: f32 = 250.0;
const DEFAULT_PEAK_NITS: f32 = 1000.0;
const MAX_NITS: f32 = 10000.0;

const REC2020_PRIMARIES: [[f32; 2]; 3] = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]];
const D65: [f32; 2] = [0.3127, 0.3290];

/// What the consumer asks of a display. Luminance requests (from the consumer's calibration)
/// override what the OS reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayRequest {
    /// Use an HDR encoding when the surface offers one and the OS has HDR on. Without a
    /// `DisplayInfo` the surface's list alone decides; Windows lists PQ with HDR off, so pass
    /// a `DisplayWatch` feed there.
    pub hdr: bool,
    /// Overrides `hdr` when the surface supports it.
    pub encoding: Option<OutputFormat>,
    /// FIFO when on, IMMEDIATE when off.
    pub vsync: bool,
    /// PQ and scRGB only.
    pub paper_white_nits: Option<f32>,
    pub peak_nits: Option<f32>,
    pub black_nits: Option<f32>,
}

impl Default for DisplayRequest {
    fn default() -> Self {
        Self {
            hdr: true,
            encoding: None,
            vsync: true,
            paper_white_nits: None,
            peak_nits: None,
            black_nits: None,
        }
    }
}

/// What the OS reports for the window's monitor. With HDR on, the luminance fields are the
/// display's default HDR colour profile (the Windows HDR Calibration app writes one), else
/// EDID; the consumer's request overrides them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DisplayInfo {
    pub os_hdr_on: Option<bool>,
    pub max_nits: Option<f32>,
    pub max_full_frame_nits: Option<f32>,
    pub min_nits: Option<f32>,
    pub sdr_white_nits: Option<f32>,
    pub refresh_hz: Option<f32>,
    /// Red, green, blue and white, CIE 1931 xy.
    pub primaries_xy: Option<[[f32; 2]; 4]>,
}

/// The latest `DisplayInfo` and a count that rises each time it changes. Clones share one
/// value; any thread may read or write it.
#[derive(Clone, Debug, Default)]
pub struct DisplayFeed(Arc<Mutex<(u64, DisplayInfo)>>);

impl DisplayFeed {
    pub fn new(info: DisplayInfo) -> Self {
        Self(Arc::new(Mutex::new((0, info))))
    }

    /// Stores `info`, raising the count if it differs from the current value.
    pub fn set(&self, info: DisplayInfo) {
        let mut latest = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if latest.1 != info {
            *latest = (latest.0 + 1, info);
        }
    }

    /// The change count and the current value.
    pub fn latest(&self) -> (u64, DisplayInfo) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The encoding and luminance a `Display` presents with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedDisplay {
    pub encoding: OutputFormat,
    pub paper_white_nits: f32,
    pub peak_nits: f32,
    pub black_nits: f32,
}

impl DisplayRequest {
    /// Picks the encoding and luminance for a surface supporting `supported`, given what the
    /// OS reports. Pure.
    ///
    /// - Encoding: `encoding` if supported, else with `hdr` PQ then scRGB, else SDR. With
    ///   `info.os_hdr_on == Some(false)` every choice resolves to SDR, since Windows
    ///   advertises PQ either way. With frame generation on, scRGB becomes PQ.
    /// - SDR: paper white = peak = 250 nits, black 0; requests and OS values are ignored.
    /// - PQ and scRGB: paper white from the request, else the OS SDR white, else 250; peak
    ///   from the request, else the display maximum, else 1000, at least 250; black from the
    ///   request, else the display minimum, else 0. Then black <= paper white <= peak <=
    ///   10000.
    pub fn resolve(
        &self,
        supported: &[OutputFormat],
        info: Option<&DisplayInfo>,
        frame_generation: bool,
    ) -> ResolvedDisplay {
        let info = info.copied().unwrap_or_default();
        let available = |f: OutputFormat| {
            supported.contains(&f) && (f == OutputFormat::Sdr || info.os_hdr_on != Some(false))
        };
        let mut encoding = match self.encoding {
            Some(e) if available(e) => e,
            _ if self.hdr => [OutputFormat::Hdr10Pq, OutputFormat::ScRgb]
                .into_iter()
                .find(|&f| available(f))
                .unwrap_or(OutputFormat::Sdr),
            _ => OutputFormat::Sdr,
        };
        if frame_generation && encoding == OutputFormat::ScRgb {
            encoding = if available(OutputFormat::Hdr10Pq) {
                OutputFormat::Hdr10Pq
            } else {
                OutputFormat::Sdr
            };
        }
        if encoding == OutputFormat::Sdr {
            return ResolvedDisplay {
                encoding,
                paper_white_nits: SDR_WHITE_NITS,
                peak_nits: SDR_WHITE_NITS,
                black_nits: 0.0,
            };
        }

        let positive = |v: Option<f32>| v.filter(|v| v.is_finite() && *v > 0.0);
        let peak = positive(self.peak_nits)
            .or(positive(info.max_nits))
            .unwrap_or(DEFAULT_PEAK_NITS)
            .clamp(SDR_WHITE_NITS, MAX_NITS);
        let paper_white = positive(self.paper_white_nits)
            .or(positive(info.sdr_white_nits))
            .unwrap_or(SDR_WHITE_NITS)
            .min(peak);
        let non_negative = |v: Option<f32>| v.filter(|v| v.is_finite() && *v >= 0.0);
        let black = non_negative(self.black_nits)
            .or(non_negative(info.min_nits))
            .unwrap_or(0.0)
            .min(paper_white);
        ResolvedDisplay {
            encoding,
            paper_white_nits: paper_white,
            peak_nits: peak,
            black_nits: black,
        }
    }

    pub fn present_mode(&self) -> PresentMode {
        if self.vsync {
            PresentMode::Fifo
        } else {
            PresentMode::Immediate
        }
    }
}

impl ResolvedDisplay {
    /// Mastering metadata for HDR encodings: Rec.2020 primaries, D65, the resolved peak and
    /// black, MaxCLL at the resolved peak, MaxFALL unknown. `None` for SDR.
    pub fn hdr_metadata(&self) -> Option<HdrMetadata> {
        (self.encoding != OutputFormat::Sdr).then_some(HdrMetadata {
            primaries: REC2020_PRIMARIES,
            white: D65,
            max_luminance: self.peak_nits,
            min_luminance: self.black_nits,
            max_cll: self.peak_nits,
            max_fall: 0.0,
        })
    }
}

/// A full-screen calibration pattern: a centre patch on a surround, shown in place of the
/// view and encoded without tonemapping, so each level reaches the display at its stated
/// luminance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationPattern {
    /// A patch at `nits` on a 10,000-nit surround. The patch disappears once `nits` reaches
    /// the luminance at which the display clips.
    Peak { nits: f32 },
    /// A patch at `nits` on a surround at 18% of it: white paper and a mid-grey card under
    /// that paper white.
    PaperWhite { nits: f32 },
    /// A patch at `nits` on a 0-nit surround. The patch appears once `nits` rises above the
    /// display's black.
    Black { nits: f32 },
}

/// Pattern levels in display-linear units: linear Rec.709, 1.0 = paper white.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatternLevels {
    pub patch: f32,
    pub surround: f32,
}

impl CalibrationPattern {
    /// The pattern's levels on `display`. On SDR, levels above paper white clip.
    pub fn levels(&self, display: &ResolvedDisplay) -> PatternLevels {
        let (patch, surround) = match *self {
            Self::Peak { nits } => (nits, MAX_NITS),
            Self::PaperWhite { nits } => (nits, 0.18 * nits),
            Self::Black { nits } => (nits, 0.0),
        };
        PatternLevels {
            patch: patch / display.paper_white_nits,
            surround: surround / display.paper_white_nits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [OutputFormat; 3] = [
        OutputFormat::Sdr,
        OutputFormat::Hdr10Pq,
        OutputFormat::ScRgb,
    ];

    fn hdr_info() -> DisplayInfo {
        DisplayInfo {
            os_hdr_on: Some(true),
            max_nits: Some(580.0),
            min_nits: Some(0.005),
            sdr_white_nits: Some(200.0),
            ..DisplayInfo::default()
        }
    }

    #[test]
    fn os_hdr_off_resolves_sdr_though_pq_is_advertised() {
        let info = DisplayInfo {
            os_hdr_on: Some(false),
            ..hdr_info()
        };
        for encoding in [None, Some(OutputFormat::Hdr10Pq), Some(OutputFormat::ScRgb)] {
            let request = DisplayRequest {
                encoding,
                ..DisplayRequest::default()
            };
            assert_eq!(
                request.resolve(&ALL, Some(&info), false).encoding,
                OutputFormat::Sdr
            );
        }
        let on = DisplayRequest::default().resolve(&ALL, Some(&hdr_info()), false);
        assert_eq!(on.encoding, OutputFormat::Hdr10Pq);
    }

    #[test]
    fn frame_generation_turns_scrgb_into_pq() {
        let request = DisplayRequest {
            encoding: Some(OutputFormat::ScRgb),
            ..DisplayRequest::default()
        };
        assert_eq!(
            request.resolve(&ALL, Some(&hdr_info()), false).encoding,
            OutputFormat::ScRgb
        );
        let fg = request.resolve(&ALL, Some(&hdr_info()), true);
        assert_eq!(fg.encoding, OutputFormat::Hdr10Pq);
        assert_eq!(fg.paper_white_nits, 200.0);
        let scrgb_only = [OutputFormat::Sdr, OutputFormat::ScRgb];
        assert_eq!(
            request
                .resolve(&scrgb_only, Some(&hdr_info()), true)
                .encoding,
            OutputFormat::Sdr
        );
        let hdr = DisplayRequest::default();
        assert_eq!(
            hdr.resolve(&scrgb_only, Some(&hdr_info()), false).encoding,
            OutputFormat::ScRgb
        );
    }

    #[test]
    fn sdr_ignores_os_white_and_requests() {
        let request = DisplayRequest {
            hdr: false,
            paper_white_nits: Some(400.0),
            peak_nits: Some(800.0),
            black_nits: Some(0.1),
            ..DisplayRequest::default()
        };
        let sdr = request.resolve(&ALL, Some(&hdr_info()), false);
        assert_eq!(
            sdr,
            ResolvedDisplay {
                encoding: OutputFormat::Sdr,
                paper_white_nits: 250.0,
                peak_nits: 250.0,
                black_nits: 0.0
            }
        );
        assert_eq!(sdr.hdr_metadata(), None);
    }

    #[test]
    fn luminance_defaults_overrides_and_clamps() {
        let r = DisplayRequest::default().resolve(&ALL, Some(&hdr_info()), false);
        assert_eq!(
            (r.paper_white_nits, r.peak_nits, r.black_nits),
            (200.0, 580.0, 0.005)
        );

        let none = DisplayRequest::default().resolve(&ALL, None, false);
        assert_eq!(none.encoding, OutputFormat::Hdr10Pq);
        assert_eq!(
            (none.paper_white_nits, none.peak_nits, none.black_nits),
            (250.0, 1000.0, 0.0)
        );

        let request = DisplayRequest {
            paper_white_nits: Some(300.0),
            peak_nits: Some(1500.0),
            black_nits: Some(0.0),
            ..DisplayRequest::default()
        };
        let r = request.resolve(&ALL, Some(&hdr_info()), false);
        assert_eq!(
            (r.paper_white_nits, r.peak_nits, r.black_nits),
            (300.0, 1500.0, 0.0)
        );

        let dim = DisplayInfo {
            max_nits: Some(120.0),
            sdr_white_nits: Some(240.0),
            ..hdr_info()
        };
        let r = DisplayRequest::default().resolve(&ALL, Some(&dim), false);
        assert_eq!((r.paper_white_nits, r.peak_nits), (240.0, 250.0));

        let request = DisplayRequest {
            paper_white_nits: Some(20000.0),
            peak_nits: Some(20000.0),
            black_nits: Some(50000.0),
            ..DisplayRequest::default()
        };
        let r = request.resolve(&ALL, Some(&hdr_info()), false);
        assert_eq!(
            (r.paper_white_nits, r.peak_nits, r.black_nits),
            (10000.0, 10000.0, 10000.0)
        );

        let request = DisplayRequest {
            paper_white_nits: Some(f32::NAN),
            peak_nits: Some(-5.0),
            black_nits: Some(-1.0),
            ..DisplayRequest::default()
        };
        let r = request.resolve(&ALL, Some(&hdr_info()), false);
        assert_eq!(
            (r.paper_white_nits, r.peak_nits, r.black_nits),
            (200.0, 580.0, 0.005)
        );
    }
}
