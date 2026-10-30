//! Screenshot writers: each file decodes back to the captured floats.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use orr_output::{Primaries, Screenshot, pq_decode, pq_encode, srgb_encode};

const W: u32 = 100;
const H: u32 = 100;

const REC709_TO_REC2020: [[f64; 3]; 3] = [
    [0.627_403_9, 0.329_283_0, 0.043_313_1],
    [0.069_097_3, 0.919_540_4, 0.011_362_3],
    [0.016_391_4, 0.088_013_3, 0.895_595_3],
];

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// An HDR display capture: hues ramping to 4 x paper white, some outside Rec.709, one
/// out-of-gamut texel past PQ's range and one hot texel at the top-left.
fn hdr_capture() -> Screenshot {
    let hues = [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.6, 0.2],
        [1.1, -0.05, 0.0],
    ];
    let mut rgba = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let h = hues[y as usize % hues.len()];
            let s = 4.0 * (y * W + x) as f32 / (W * H) as f32;
            rgba.extend_from_slice(&[h[0] * s, h[1] * s, h[2] * s, 1.0]);
        }
    }
    rgba[..3].copy_from_slice(&[25.0; 3]);
    Screenshot {
        frame: 7,
        size: [W, H],
        rgba,
        scene_referred: false,
        primaries: Primaries::Rec709,
        paper_white_nits: 200.0,
        hdr: true,
    }
}

fn read_png(name: &str) -> (png::Info<'static>, Vec<u8>) {
    let file = BufReader::new(File::open(path(name)).unwrap());
    let mut reader = png::Decoder::new(file).read_info().unwrap();
    let mut data = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut data).unwrap();
    let info = reader.info().clone();
    (info, data)
}

fn rec2020_nits(shot: &Screenshot) -> Vec<[f64; 3]> {
    let pw = f64::from(shot.paper_white_nits);
    shot.rgba
        .chunks_exact(4)
        .map(|p| {
            REC709_TO_REC2020.map(|r| {
                let c = r[0] * f64::from(p[0]) + r[1] * f64::from(p[1]) + r[2] * f64::from(p[2]);
                (c * pw).clamp(0.0, 10000.0)
            })
        })
        .collect()
}

#[test]
fn pq_png_decodes_to_the_capture() {
    let shot = hdr_capture();
    shot.save_png_pq(&path("pq.png")).unwrap();
    let (info, data) = read_png("pq.png");
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
    assert_eq!(info.color_type, png::ColorType::Rgb);
    let cicp = info.coding_independent_code_points.unwrap();
    assert_eq!(
        (
            cicp.color_primaries,
            cicp.transfer_function,
            cicp.matrix_coefficients,
            cicp.is_video_full_range_image
        ),
        (9, 16, 0, true)
    );

    let icc = moxcms::ColorProfile::new_from_slice(info.icc_profile.as_deref().unwrap()).unwrap();
    let icc_cicp = icc.cicp.unwrap();
    assert_eq!(icc_cicp.color_primaries, moxcms::CicpColorPrimaries::Bt2020);
    assert_eq!(
        icc_cicp.transfer_characteristics,
        moxcms::TransferCharacteristics::Smpte2084
    );

    let nits = rec2020_nits(&shot);
    let samples: Vec<f64> = data
        .chunks_exact(2)
        .map(|s| f64::from(u16::from_be_bytes([s[0], s[1]])) / 65535.0)
        .collect();
    for (i, want) in nits.iter().enumerate() {
        for c in 0..3 {
            let got = samples[3 * i + c];
            // Within 10-bit PQ quantisation, as a 10-bit swapchain would show it.
            assert!(
                (got - pq_encode(want[c])).abs() <= 0.5 / 1023.0,
                "texel {i} channel {c}: {} nits against {}",
                pq_decode(got),
                want[c]
            );
        }
    }

    // MaxCLL skips the hot texel: the 99.99th percentile of 10,000 texels is the second
    // brightest, the bright end of the ramp.
    let mut brightest: Vec<f64> = nits.iter().map(|p| p[0].max(p[1]).max(p[2])).collect();
    let fall = brightest.iter().sum::<f64>() / brightest.len() as f64;
    brightest.sort_by(f64::total_cmp);
    assert!(brightest[brightest.len() - 1] >= 5000.0);
    let cll = brightest[brightest.len() - 2];
    let light = info.content_light_level.unwrap();
    assert_eq!(
        light.max_content_light_level,
        (cll * 10000.0).round() as u32
    );
    assert_eq!(
        light.max_frame_average_light_level,
        (fall * 10000.0).round() as u32
    );
}

#[test]
fn srgb_png_clips_at_paper_white() {
    let shot = hdr_capture();
    shot.save_png_srgb(&path("srgb.png")).unwrap();
    let (info, data) = read_png("srgb.png");
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    assert!(info.srgb.is_some());
    for (i, p) in shot.rgba.chunks_exact(4).enumerate() {
        for c in 0..3 {
            let want = srgb_encode(f64::from(p[c]).clamp(0.0, 1.0)) * 255.0;
            let got = f64::from(data[3 * i + c]);
            assert!((got - want).abs() <= 0.5, "texel {i} channel {c}");
        }
    }
}

#[test]
fn exr_holds_scene_referred_nits() {
    let rgba: Vec<f32> = (0..W * H * 4)
        .map(|i| if i % 4 == 3 { 1.0 } else { 0.01 * i as f32 })
        .collect();
    let shot = Screenshot {
        frame: 3,
        size: [W, H],
        rgba,
        scene_referred: true,
        primaries: Primaries::Rec2020,
        paper_white_nits: 250.0,
        hdr: false,
    };
    assert!(shot.save_png_srgb(&path("scene.png")).is_err());
    assert!(shot.save_png_pq(&path("scene.png")).is_err());
    shot.save_exr(&path("scene.exr")).unwrap();

    let image = exr::prelude::read_first_rgba_layer_from_file(
        path("scene.exr"),
        |size, _| vec![[0f32; 4]; size.0 * size.1],
        |pixels: &mut Vec<[f32; 4]>, at, (r, g, b, a): (f32, f32, f32, f32)| {
            pixels[at.1 * W as usize + at.0] = [r, g, b, a];
        },
    )
    .unwrap();
    assert_eq!(
        image.layer_data.channel_data.pixels.as_flattened(),
        shot.rgba
    );
    let chroma = image.attributes.chromaticities.unwrap();
    assert_eq!((chroma.red.0, chroma.red.1), (0.708, 0.292));
    assert_eq!((chroma.white.0, chroma.white.1), (0.3127, 0.3290));
    assert_eq!(image.layer_data.attributes.white_luminance, Some(1.0));
}
