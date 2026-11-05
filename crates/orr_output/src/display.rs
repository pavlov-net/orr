use orr_color::{EDR_REFERENCE_NITS, PQ_MAX_NITS, RgbPrimaries, SCRGB_WHITE_NITS};
use orr_scene::orr_gpu::{HdrMetadata, OutputFormat, PresentMode};
use orr_window::DisplayInfo;

/// GT7's SDR calibration: SDR paper white and peak, and the default HDR paper white.
const SDR_WHITE_NITS: f32 = 250.0;
const DEFAULT_PEAK_NITS: f32 = 1000.0;
const MAX_NITS: f32 = PQ_MAX_NITS as f32;

/// What the consumer asks of a display. Luminance requests (from the consumer's calibration)
/// override what the OS reports. On a display that reports headroom, nits are the tone curve's,
/// with the OS's SDR white at 250, and the headroom limits the peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayRequest {
    /// Use an HDR encoding when the surface offers one and the OS has HDR on. Without a
    /// `DisplayInfo` the surface's list alone decides, and Windows lists PQ with HDR off; a
    /// `Display` watches the OS for it.
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

/// The encoding and luminance a `Display` presents with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedDisplay {
    pub encoding: OutputFormat,
    /// Paper white in the tone curve's nits.
    pub paper_white_nits: f32,
    /// Paper white's level in the signal, in the encoding's nits. It equals `paper_white_nits`
    /// on displays that report nits; on one that reports headroom, the curve's 250 lands on
    /// the OS's SDR white.
    pub signal_white_nits: f32,
    pub peak_nits: f32,
    pub black_nits: f32,
}

impl DisplayRequest {
    /// Picks the encoding and luminance for a surface supporting `supported`, given what the
    /// OS reports. Pure.
    ///
    /// - Encoding: `encoding` if supported, else with `hdr` PQ then scRGB, or scRGB then PQ on
    ///   a display that reports headroom, else SDR. With `info.os_hdr_on == Some(false)` every
    ///   choice resolves to SDR, since Windows advertises PQ either way. With frame generation
    ///   on, scRGB becomes PQ.
    /// - SDR: paper white = peak = 250 nits, black 0; requests and OS values are ignored.
    /// - PQ and scRGB: paper white from the request, else the OS SDR white, else 250; peak
    ///   from the request, else the display maximum, else 1000, at least 250; black from the
    ///   request, else the display minimum, else 0. Then black <= paper white <= peak <=
    ///   10000.
    /// - With headroom: as above, with the OS's SDR white at 250 and the peak 250 x the current
    ///   headroom, or the request if that's lower.
    pub fn resolve(
        &self,
        supported: &[OutputFormat],
        info: Option<&DisplayInfo>,
        frame_generation: bool,
    ) -> ResolvedDisplay {
        let info = info.copied().unwrap_or_default();
        let headroom = info.headroom.map(|h| h.current.max(1.0));
        let available = |f: OutputFormat| {
            supported.contains(&f) && (f == OutputFormat::Sdr || info.os_hdr_on != Some(false))
        };
        let mut encoding = match self.encoding {
            Some(e) if available(e) => e,
            _ if self.hdr => {
                let order = if headroom.is_some() {
                    [OutputFormat::ScRgb, OutputFormat::Hdr10Pq]
                } else {
                    [OutputFormat::Hdr10Pq, OutputFormat::ScRgb]
                };
                order
                    .into_iter()
                    .find(|&f| available(f))
                    .unwrap_or(OutputFormat::Sdr)
            }
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
                signal_white_nits: SDR_WHITE_NITS,
                peak_nits: SDR_WHITE_NITS,
                black_nits: 0.0,
            };
        }

        let positive = |v: Option<f32>| v.filter(|v| v.is_finite() && *v > 0.0);
        let requested_peak = positive(self.peak_nits);
        let peak = match headroom {
            Some(h) => {
                let room = SDR_WHITE_NITS * h;
                requested_peak.map_or(room, |r| r.min(room))
            }
            None => requested_peak
                .or(positive(info.max_nits))
                .unwrap_or(DEFAULT_PEAK_NITS),
        }
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
        let signal_white = match headroom {
            Some(_) => {
                let os_white = match encoding {
                    OutputFormat::ScRgb => SCRGB_WHITE_NITS,
                    _ => EDR_REFERENCE_NITS,
                };
                paper_white / SDR_WHITE_NITS * os_white as f32
            }
            None => paper_white,
        };
        ResolvedDisplay {
            encoding,
            paper_white_nits: paper_white,
            signal_white_nits: signal_white,
            peak_nits: peak,
            black_nits: black,
        }
    }

    pub(crate) fn present_mode(&self) -> PresentMode {
        if self.vsync {
            PresentMode::Fifo
        } else {
            PresentMode::Immediate
        }
    }
}

impl ResolvedDisplay {
    /// Mastering metadata for HDR encodings: Rec.2020 primaries, D65, the resolved peak and
    /// black at their signal levels, MaxCLL at the peak, MaxFALL unknown. `None` for SDR.
    pub(crate) fn hdr_metadata(&self) -> Option<HdrMetadata> {
        let p = RgbPrimaries::REC2020;
        let signal = self.signal_white_nits / self.paper_white_nits;
        (self.encoding != OutputFormat::Sdr).then_some(HdrMetadata {
            primaries: [p.red, p.green, p.blue].map(Into::into),
            white: p.white.into(),
            max_luminance: self.peak_nits * signal,
            min_luminance: self.black_nits * signal,
            max_cll: self.peak_nits * signal,
            max_fall: 0.0,
        })
    }
}

/// A full-screen calibration pattern: a center patch on a surround, shown in place of the
/// view and encoded without tonemapping, so each level reaches the display at its stated
/// luminance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationPattern {
    /// A patch at `nits` on a 10,000-nit surround. The patch disappears once `nits` reaches
    /// the luminance at which the display clips.
    Peak { nits: f32 },
    /// A patch at `nits` on a surround at 18% of it: white paper and a mid-gray card under
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
    use orr_window::DisplayHeadroom;

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

    fn headroom(current: f32) -> DisplayInfo {
        DisplayInfo {
            os_hdr_on: Some(true),
            headroom: Some(DisplayHeadroom {
                current,
                potential: 8.0,
                reference: None,
            }),
            ..DisplayInfo::default()
        }
    }

    #[test]
    fn headroom_puts_os_white_at_the_curves_sdr_white() {
        let hdr = DisplayRequest::default();
        // No headroom yet: the SDR curve, with OS white at scRGB's 1.0.
        let r = hdr.resolve(&ALL, Some(&headroom(1.0)), false);
        assert_eq!(r.encoding, OutputFormat::ScRgb);
        assert_eq!((r.paper_white_nits, r.peak_nits), (250.0, 250.0));
        assert_eq!(r.signal_white_nits, SCRGB_WHITE_NITS as f32);

        let r = hdr.resolve(&ALL, Some(&headroom(1.5)), false);
        assert_eq!((r.paper_white_nits, r.peak_nits), (250.0, 375.0));
        // Anything below 1 is no headroom, on a screen that still reports it.
        for current in [0.5, 0.0, f32::NAN] {
            let r = hdr.resolve(&ALL, Some(&headroom(current)), false);
            assert_eq!(r.peak_nits, 250.0);
            assert_eq!(r.signal_white_nits, SCRGB_WHITE_NITS as f32);
        }

        // PQ puts OS white at the level macOS maps to it, and its metadata follows.
        let pq = DisplayRequest {
            encoding: Some(OutputFormat::Hdr10Pq),
            ..hdr
        };
        let r = pq.resolve(&ALL, Some(&headroom(4.0)), false);
        assert_eq!(r.signal_white_nits, EDR_REFERENCE_NITS as f32);
        let metadata = r.hdr_metadata().unwrap();
        assert_eq!(metadata.max_luminance, 4.0 * EDR_REFERENCE_NITS as f32);

        // A paper white above OS white raises the signal with it; headroom limits a peak.
        let calibrated = DisplayRequest {
            paper_white_nits: Some(500.0),
            peak_nits: Some(2000.0),
            ..hdr
        };
        let r = calibrated.resolve(&ALL, Some(&headroom(4.0)), false);
        assert_eq!((r.paper_white_nits, r.peak_nits), (500.0, 1000.0));
        assert_eq!(r.signal_white_nits, 2.0 * SCRGB_WHITE_NITS as f32);
        let low = DisplayRequest {
            peak_nits: Some(600.0),
            ..hdr
        };
        let r = low.resolve(&ALL, Some(&headroom(4.0)), false);
        assert_eq!(r.peak_nits, 600.0);

        // Frame generation still leaves scRGB, and the signal follows the encoding.
        let r = hdr.resolve(&ALL, Some(&headroom(4.0)), true);
        assert_eq!(r.encoding, OutputFormat::Hdr10Pq);
        assert_eq!(r.signal_white_nits, EDR_REFERENCE_NITS as f32);
    }

    #[test]
    fn nits_keep_the_signal_at_paper_white() {
        let r = DisplayRequest::default().resolve(&ALL, Some(&hdr_info()), false);
        assert_eq!(r.signal_white_nits, r.paper_white_nits);
        assert_eq!(r.hdr_metadata().unwrap().max_luminance, r.peak_nits);
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
                signal_white_nits: 250.0,
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
