//! Tonemap and compose on the GPU against the same Slang on the CPU target, for SDR, PQ and
//! scRGB backbuffers, and compose against BT.2111's PQ code values.

use half::f16;
use orr_color::{Primaries, pq_decode, pq_encode, srgb_decode};
use orr_output::{
    ComposePass, ComposeRoot, ComposeSource, DisplayHeadroom, DisplayInfo, DisplayRequest,
    Gt7Curve, Gt7Params, OutputFormat, OutputKernels, OutputRoots, ResolvedDisplay, TonemapPass,
    ViewOutput, exposure_from_ev100,
};
use orr_scene::orr_gpu::{
    Format, Hazards, MemoryKind, QueueKind, Stages, TextureDesc, TextureMemory, TextureRegion,
    TextureUsage, ViewDesc,
};
use orr_scene::testing::{assert_clean, device, read_texture, wait};
use orr_scene::{ExposureState, ViewConstants};
use output_math::{encode_display, gt7_display_linear};

const WIDTH: u32 = 1024;
const EV100: f32 = 12.0;

/// Linear Rec.2020 hues, one per row: gray, the Rec.2020 primaries and secondaries, the
/// Rec.709 red and a warm tone.
fn hues() -> [[f32; 3]; 8] {
    let rec709_red = orr_color::rec709_to_rec2020([1.0, 0.0, 0.0])
        .as_vec3()
        .to_array();
    [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        rec709_red,
        [1.0, 0.6, 0.4],
    ]
}

/// Exposed values, row by row: 0, then log-spaced from 1e-4 to 1e3 times paper white.
fn ramp() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    for hue in hues() {
        for i in 0..WIDTH {
            let x = if i == 0 {
                0.0
            } else {
                10f32.powf(-4.0 + 7.0 * (i - 1) as f32 / (WIDTH - 2) as f32)
            };
            out.push(hue.map(|h| h * x));
        }
    }
    out
}

/// Runs tonemap and compose over the ramp into a backbuffer of `format`; returns its texels.
fn run(display: &ResolvedDisplay, format: Format) -> Vec<u8> {
    let gpu = device();
    let height = hues().len() as u32;
    // Lit two stops brighter than the manual exposure, so tonemap scales by a quarter.
    let pre_exposure = 4.0 * exposure_from_ev100(EV100);
    let radiance: Vec<[f32; 4]> = ramp()
        .iter()
        .map(|c| [c[0] * 4.0, c[1] * 4.0, c[2] * 4.0, 1.0])
        .collect();

    let texture = |format, usage, name| {
        let desc = TextureDesc::d2(format, WIDTH, height, usage);
        gpu.create_texture(&desc, TextureMemory::Dedicated, name)
            .unwrap()
    };
    let source = texture(Format::Rgba32Float, TextureUsage::SAMPLED, "radiance");
    let linear = texture(
        Format::Rgba16Float,
        TextureUsage::SAMPLED | TextureUsage::STORAGE,
        "display linear",
    );
    let backbuffer = texture(format, TextureUsage::STORAGE, "backbuffer");
    let slots: Vec<u32> = Vec::from_fn(4, |_| gpu.alloc_image_slot().unwrap());
    let view = ViewDesc::default();
    let source_index = gpu.write_sampled(slots[0], &source, &view);
    let linear_store = gpu.write_storage(slots[1], &linear, &view);
    let linear_index = gpu.write_sampled(slots[2], &linear, &view);
    let backbuffer_store = gpu.write_storage(slots[3], &backbuffer, &view);

    let mut upload = gpu
        .gpu_malloc::<[f32; 4]>(MemoryKind::Upload, radiance.len() as u64, "radiance upload")
        .unwrap();
    upload.as_mut_slice().unwrap().copy_from_slice(&radiance);

    // The view's current slot holds the pre-exposure and the next one the adapted exposure,
    // as adapt leaves them.
    let mut exposure = gpu
        .gpu_malloc::<ExposureState>(MemoryKind::Upload, 2, "exposure slots")
        .unwrap();
    exposure.as_mut_slice().unwrap().copy_from_slice(&[
        ExposureState {
            pre_exposure,
            ..ExposureState::default()
        },
        ExposureState {
            adapted_ev100: EV100,
            ..ExposureState::default()
        },
    ]);
    let mut constants = gpu
        .gpu_malloc::<ViewConstants>(MemoryKind::Upload, 1, "view constants")
        .unwrap();
    let mut view: ViewConstants = bytemuck::Zeroable::zeroed();
    view.exposure = exposure.gpu();
    constants.as_mut_slice().unwrap()[0] = view;

    let kernels = OutputKernels::new(gpu).unwrap();
    let mut roots = OutputRoots::new(gpu).unwrap();
    let mut compose_root = ComposeRoot::new(gpu).unwrap();
    let extent = [WIDTH, height];
    let queue = gpu.queue(QueueKind::Render).unwrap();
    let mut list = queue.command_list().unwrap();
    for t in [&source, &linear, &backbuffer] {
        list.discard(t);
    }
    let whole = TextureRegion::whole_mip(source.desc(), 0);
    list.copy_to_texture(&source, &whole, upload.gpu().cast());
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    kernels.tonemap(
        &mut list,
        &mut roots,
        &TonemapPass {
            source: source_index,
            target: linear_store,
            extent,
            view: constants.gpu(),
            display,
            output: &ViewOutput::default(),
            hook_source: None,
            glare: None,
            distortion: None,
            frame: 0,
        },
        None,
    );
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    kernels.compose(
        &mut list,
        &mut compose_root,
        &ComposePass {
            source: ComposeSource::Linear(linear_index, display.primaries()),
            ui: None,
            target: backbuffer_store,
            extent,
            display,
        },
    );
    let point = queue.submit(vec![list], &[]).unwrap();
    wait(point);
    for slot in slots {
        gpu.free_image_slot(slot, point);
    }
    read_texture(&backbuffer, &whole)
}

/// The CPU target's encoded signal for every texel, through an RGBA16F display-linear image
/// like the GPU's.
fn reference(display: &ResolvedDisplay) -> Vec<[f32; 3]> {
    let curve = Gt7Curve::new(
        &Gt7Params::default(),
        display.peak_nits,
        display.paper_white_nits,
    );
    let linear: Vec<[f32; 3]> = gt7_display_linear(&curve, &ramp(), display.encoding)
        .into_iter()
        .map(|c| c.map(|v| f16::from_f32(v).to_f32()))
        .collect();
    encode_display(&linear, display.encoding, display.signal_white_nits)
}

/// How far a channel may differ in nits, as a fraction of the texel's brightest channel,
/// where code values do not bound it. Channels near zero in saturated colors come out of
/// cancellations in GT7's ICtCp step in f32, and for SDR and scRGB in its Rec.2020 to Rec.709
/// conversion.
const NITS_TOLERANCE: f64 = 1e-3;

/// Checks every channel of `got` (signal times `scale`) against `reference` (signal):
/// within `tolerance(reference)` of it, or within `NITS_TOLERANCE` once decoded to nits.
fn check(
    name: &str,
    got: &[[f32; 3]],
    reference: &[[f32; 3]],
    scale: f32,
    tolerance: impl Fn(f32) -> f32,
    nits: impl Fn(f32) -> f64,
) {
    for (i, (g, r)) in got.iter().zip(reference).enumerate() {
        let brightest = r.iter().map(|&v| nits(v)).fold(0.0, f64::max);
        for c in 0..3 {
            let close = (g[c] - r[c] * scale).abs() < tolerance(r[c]);
            let nits_error = (nits(g[c] / scale) - nits(r[c])).abs();
            assert!(
                close || nits_error <= NITS_TOLERANCE * brightest,
                "{name} texel {i}: {g:?} against {:?}",
                r.map(|v| v * scale)
            );
        }
    }
}

const HDR_MONITOR: DisplayInfo = DisplayInfo {
    os_hdr_on: Some(true),
    max_nits: Some(1000.0),
    max_full_frame_nits: Some(600.0),
    min_nits: Some(0.05),
    sdr_white_nits: Some(240.0),
    headroom: None,
    refresh_hz: Some(144.0),
    primaries_xy: None,
};

const EDR_HEADROOM: f32 = 4.0;

/// A screen that reports headroom and no nits.
const EDR_SCREEN: DisplayInfo = DisplayInfo {
    os_hdr_on: Some(true),
    max_nits: None,
    max_full_frame_nits: None,
    min_nits: None,
    sdr_white_nits: None,
    headroom: Some(DisplayHeadroom {
        current: EDR_HEADROOM,
        potential: 8.0,
        reference: None,
    }),
    refresh_hz: Some(120.0),
    primaries_xy: None,
};

#[test]
fn sdr_pq_and_scrgb_match_cpu_target() {
    let supported = [
        OutputFormat::Sdr,
        OutputFormat::Hdr10Pq,
        OutputFormat::ScRgb,
    ];

    let sdr = DisplayRequest {
        hdr: false,
        ..DisplayRequest::default()
    }
    .resolve(&supported, Some(&HDR_MONITOR), false);
    assert_eq!(sdr.encoding, OutputFormat::Sdr);
    let got: Vec<[f32; 3]> = run(&sdr, Format::Bgra8Unorm)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|t| [f32::from(t[2]), f32::from(t[1]), f32::from(t[0])])
        .collect();
    let sdr_nits = |v: f32| srgb_decode(f64::from(v)) * f64::from(sdr.paper_white_nits);
    check("SDR", &got, &reference(&sdr), 255.0, |_| 1.0, sdr_nits);

    let pq = DisplayRequest::default().resolve(&supported, Some(&HDR_MONITOR), false);
    assert_eq!(pq.encoding, OutputFormat::Hdr10Pq);
    let got: Vec<[f32; 3]> = run(&pq, Format::A2b10g10r10Unorm)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|t| {
            let v = u32::from_le_bytes(*t);
            [0, 10, 20].map(|shift| ((v >> shift) & 0x3ff) as f32)
        })
        .collect();
    let pq_nits = |v: f32| pq_decode(f64::from(v));
    check("PQ", &got, &reference(&pq), 1023.0, |_| 1.0, pq_nits);
    // The brightest gray sits at the display's peak.
    let peak = pq_encode(1000.0) as f32 * 1023.0;
    let white = got[WIDTH as usize - 1];
    assert!(
        white.iter().all(|&c| (c - peak).abs() < 1.0),
        "PQ gray at 1e3 x paper white: {white:?}, peak code {peak}"
    );

    let scrgb = DisplayRequest {
        encoding: Some(OutputFormat::ScRgb),
        ..DisplayRequest::default()
    }
    .resolve(&supported, Some(&HDR_MONITOR), false);
    let run_scrgb = |display| -> Vec<[f32; 3]> {
        run(display, Format::Rgba16Float)
            .as_chunks::<2>()
            .0
            .as_chunks::<4>()
            .0
            .iter()
            .map(|t| [t[0], t[1], t[2]].map(|c| f16::from_le_bytes(c).to_f32()))
            .collect()
    };
    let got = run_scrgb(&scrgb);
    let scrgb_nits = |v: f32| f64::from(v) * orr_color::SCRGB_WHITE_NITS;
    // One RGBA16F step in the display-linear image and one in the backbuffer.
    let two_steps = |r: f32| r.abs() / 512.0 + 1e-6;
    check(
        "scRGB",
        &got,
        &reference(&scrgb),
        1.0,
        two_steps,
        scrgb_nits,
    );

    let edr = DisplayRequest::default().resolve(&supported, Some(&EDR_SCREEN), false);
    assert_eq!(edr.encoding, OutputFormat::ScRgb);
    let got = run_scrgb(&edr);
    check("EDR", &got, &reference(&edr), 1.0, two_steps, scrgb_nits);
    // The brightest gray sits at the screen's headroom.
    let top = got[WIDTH as usize - 1];
    assert!(
        top.iter()
            .all(|&c| (c - EDR_HEADROOM).abs() < two_steps(EDR_HEADROOM)),
        "EDR gray at 1e3 x paper white: {top:?}"
    );

    assert_clean();
}

/// Paper white of BT.2111's 58% bars: 75% HLG at the 1000 cd/m2 reference.
const BT2111_WHITE_NITS: f32 = 203.15;

/// ITU-R BT.2111-3, Table 4 (PQ full range, 10 bits): each bar as display-linear color in its
/// primaries, 1.0 at `BT2111_WHITE_NITS`, and its code values. The 100% bar is at 10000 cd/m2.
const BT2111_PQ_BARS: [(Primaries, [f32; 3], [u32; 3]); 15] = [
    (
        Primaries::Rec2020,
        [10000.0 / BT2111_WHITE_NITS; 3],
        [1023; 3],
    ),
    (Primaries::Rec2020, [1.0, 1.0, 1.0], [594, 594, 594]),
    (Primaries::Rec2020, [1.0, 1.0, 0.0], [594, 594, 0]),
    (Primaries::Rec2020, [0.0, 1.0, 1.0], [0, 594, 594]),
    (Primaries::Rec2020, [0.0, 1.0, 0.0], [0, 594, 0]),
    (Primaries::Rec2020, [1.0, 0.0, 1.0], [594, 0, 594]),
    (Primaries::Rec2020, [1.0, 0.0, 0.0], [594, 0, 0]),
    (Primaries::Rec2020, [0.0, 0.0, 1.0], [0, 0, 594]),
    (Primaries::Rec2020, [0.0; 3], [0; 3]),
    (Primaries::Rec709, [1.0, 1.0, 0.0], [589, 593, 370]),
    (Primaries::Rec709, [0.0, 1.0, 1.0], [491, 586, 592]),
    (Primaries::Rec709, [0.0, 1.0, 0.0], [479, 585, 355]),
    (Primaries::Rec709, [1.0, 0.0, 1.0], [552, 348, 584]),
    (Primaries::Rec709, [1.0, 0.0, 0.0], [545, 335, 225]),
    (Primaries::Rec709, [0.0, 0.0, 1.0], [296, 201, 582]),
];

/// Compose writes BT.2111's PQ bars, Rec.2020 and BT.709 ones, as its code values: the PQ
/// curve, the full range and the primaries a display-linear source states. f32 PQ math puts
/// one code within 0.02 of a rounding boundary, so codes may differ by one.
#[test]
fn bt2111_pq_bars_compose_to_their_code_values() {
    let gpu = device();
    let display = ResolvedDisplay {
        encoding: OutputFormat::Hdr10Pq,
        paper_white_nits: BT2111_WHITE_NITS,
        signal_white_nits: BT2111_WHITE_NITS,
        peak_nits: 10000.0,
        black_nits: 0.0,
    };
    let kernels = OutputKernels::new(gpu).unwrap();
    for primaries in [Primaries::Rec2020, Primaries::Rec709] {
        let bars: Vec<_> = BT2111_PQ_BARS.iter().filter(|b| b.0 == primaries).collect();
        let extent = [bars.len() as u32, 1];
        let texture = |format, usage, name| {
            let desc = TextureDesc::d2(format, extent[0], 1, usage);
            gpu.create_texture(&desc, TextureMemory::Dedicated, name)
                .unwrap()
        };
        let source = texture(Format::Rgba32Float, TextureUsage::SAMPLED, "bars");
        let backbuffer = texture(
            Format::A2b10g10r10Unorm,
            TextureUsage::STORAGE,
            "backbuffer",
        );
        let mut upload = gpu
            .gpu_malloc::<[f32; 4]>(MemoryKind::Upload, bars.len() as u64, "bars upload")
            .unwrap();
        for (t, b) in upload.as_mut_slice().unwrap().iter_mut().zip(&bars) {
            *t = [b.1[0], b.1[1], b.1[2], 1.0];
        }
        let slots = [
            gpu.alloc_image_slot().unwrap(),
            gpu.alloc_image_slot().unwrap(),
        ];
        let view = ViewDesc::default();
        let source_index = gpu.write_sampled(slots[0], &source, &view);
        let target = gpu.write_storage(slots[1], &backbuffer, &view);
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.discard(&source);
        list.discard(&backbuffer);
        let whole = TextureRegion::whole_mip(source.desc(), 0);
        list.copy_to_texture(&source, &whole, upload.gpu().cast());
        list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        let mut root = ComposeRoot::new(gpu).unwrap();
        kernels.compose(
            &mut list,
            &mut root,
            &ComposePass {
                source: ComposeSource::Linear(source_index, primaries),
                ui: None,
                target,
                extent,
                display: &display,
            },
        );
        let point = queue.submit(vec![list], &[]).unwrap();
        wait(point);
        for slot in slots {
            gpu.free_image_slot(slot, point);
        }
        let texels = read_texture(&backbuffer, &whole);
        for (t, b) in texels.as_chunks::<4>().0.iter().zip(&bars) {
            let v = u32::from_le_bytes(*t);
            let got = [0, 10, 20].map(|shift| (v >> shift) & 0x3ff);
            assert!(
                (0..3).all(|c| got[c].abs_diff(b.2[c]) <= 1),
                "{primaries:?} bar {:?}: codes {got:?}, BT.2111 {:?}",
                b.1,
                b.2
            );
        }
    }
    assert_clean();
}
