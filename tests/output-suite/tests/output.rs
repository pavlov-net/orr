//! Tonemap and compose on the GPU against the same Slang on the CPU target, for SDR, PQ and
//! scRGB backbuffers.

use half::f16;
use orr_output::{
    ComposePass, ComposeRoot, ComposeSource, DisplayInfo, DisplayRequest, Gt7Curve, Gt7Params,
    OutputFormat, OutputKernels, OutputRoots, ResolvedDisplay, TonemapPass, ViewOutput,
    exposure_from_ev100, pq_decode, pq_encode, srgb_decode,
};
use orr_scene::orr_gpu::{
    Format, Hazards, MemoryKind, QueueKind, Stages, TextureDesc, TextureMemory, TextureRegion,
    TextureUsage, ViewDesc,
};
use orr_scene::testing::{assert_clean, device, read_texture, wait};
use orr_scene::{ExposureState, ViewConstants};
use output_maths::{encode_display, gt7_display_linear};

const WIDTH: u32 = 1024;
const EV100: f32 = 12.0;

/// Linear Rec.2020 hues, one per row: grey, the Rec.2020 primaries and secondaries, the
/// Rec.709 red and a warm tone.
const HUES: [[f32; 3]; 8] = [
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.627, 0.069, 0.016],
    [1.0, 0.6, 0.4],
];

/// Exposed values, row by row: 0, then log-spaced from 1e-4 to 1e3 times paper white.
fn ramp() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    for hue in HUES {
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
    let height = HUES.len() as u32;
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
    let slots: Vec<u32> = (0..4).map(|_| gpu.alloc_image_slot().unwrap()).collect();
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
    view.exposure = exposure.gpu;
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
    list.copy_to_texture(&source, &whole, upload.gpu.cast());
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    kernels.tonemap(
        &mut list,
        &mut roots,
        &TonemapPass {
            source: source_index,
            target: linear_store,
            extent,
            view: constants.gpu,
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
            source: ComposeSource::Linear(linear_index),
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
    let sdr = display.encoding == OutputFormat::Sdr;
    let linear: Vec<[f32; 3]> = gt7_display_linear(&curve, &ramp(), sdr)
        .into_iter()
        .map(|c| c.map(|v| f16::from_f32(v).to_f32()))
        .collect();
    encode_display(&linear, display.encoding, display.paper_white_nits)
}

/// How far a channel may differ in nits, as a fraction of the texel's brightest channel,
/// where code values do not bound it. Channels near zero in saturated colours come out of
/// cancellations: GT7's ICtCp step in f32 and Rec.709 to Rec.2020 in compose, which turns
/// one RGBA16F step of the brightest Rec.709 channel (2^-11 of it) into up to 1e-3 of it.
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
    refresh_hz: Some(144.0),
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
        .chunks_exact(4)
        .map(|t| [f32::from(t[2]), f32::from(t[1]), f32::from(t[0])])
        .collect();
    let sdr_nits = |v: f32| srgb_decode(f64::from(v)) * f64::from(sdr.paper_white_nits);
    check("SDR", &got, &reference(&sdr), 255.0, |_| 1.0, sdr_nits);

    let pq = DisplayRequest::default().resolve(&supported, Some(&HDR_MONITOR), false);
    assert_eq!(pq.encoding, OutputFormat::Hdr10Pq);
    let got: Vec<[f32; 3]> = run(&pq, Format::A2b10g10r10Unorm)
        .chunks_exact(4)
        .map(|t| {
            let v = u32::from_le_bytes(t.try_into().unwrap());
            [0, 10, 20].map(|shift| ((v >> shift) & 0x3ff) as f32)
        })
        .collect();
    let pq_nits = |v: f32| pq_decode(f64::from(v));
    check("PQ", &got, &reference(&pq), 1023.0, |_| 1.0, pq_nits);
    // The brightest grey sits at the display's peak.
    let peak = pq_encode(1000.0) as f32 * 1023.0;
    let white = got[WIDTH as usize - 1];
    assert!(
        white.iter().all(|&c| (c - peak).abs() < 1.0),
        "PQ grey at 1e3 x paper white: {white:?}, peak code {peak}"
    );

    let scrgb = DisplayRequest {
        encoding: Some(OutputFormat::ScRgb),
        ..DisplayRequest::default()
    }
    .resolve(&supported, Some(&HDR_MONITOR), false);
    let got: Vec<[f32; 3]> = run(&scrgb, Format::Rgba16Float)
        .chunks_exact(8)
        .map(|t| [0, 2, 4].map(|o| f16::from_le_bytes([t[o], t[o + 1]]).to_f32()))
        .collect();
    let scrgb_nits = |v: f32| f64::from(v) * 80.0;
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

    assert_clean();
}
