//! Frame generation on an orr device whose Render queue binds heaps in every list: a bar
//! moving at a known speed lands at each generated image's share of the step, at 2x and at
//! 4x, and with a distortion field bound, a uniformly shifted presented image keeps its
//! generated bar shifted with it. NGX aligns a uniform shift without the field too, so the
//! second test checks that binding a field works, not its encoding.
#![cfg(ngx)]
#![allow(clippy::print_stderr)]

mod harness;

use std::time::Duration;

use orr_dlss::{DlssSdk, FgCamera, FgInputs, FrameGeneration};
use orr_gpu::{
    Allocation, CommandList, Device, DeviceDesc, Format, Hazards, MemoryKind, QueueKind, Stages,
    Texture, TextureDesc, TextureMemory, TextureRegion, TextureUsage,
};

const OUTPUT: [u32; 2] = [1280, 720];
const BAR_WIDTH: u32 = 48;
/// Pixels per rendered frame.
const SPEED: u32 = 24;
const START: u32 = 160;
const FRAMES: u32 = 12;
/// How far a generated bar may sit from its share of the step, pixels.
const TOLERANCE: f64 = 2.0;
const BACKGROUND: u8 = 40;
const BAR: u8 = 230;

fn device() -> Device {
    let hooks = DlssSdk::device_hooks(harness::PROJECT_ID);
    Device::new(&DeviceDesc {
        name: "orr-dlss fg tests",
        hooks: Some(&hooks),
        validation: std::env::var("VK_INSTANCE_LAYERS")
            .is_ok_and(|v| v.contains("KHRONOS_validation")),
        ..DeviceDesc::default()
    })
    .expect("orr device with NGX hooks")
}

/// Left edge of the bar, as rendered, in frame `f`.
fn bar_x(f: u32) -> u32 {
    START + SPEED * f
}

/// BGRA8: the bar over a flat background, `shift` pixels right of where it was rendered.
fn color(f: u32, shift: u32) -> Vec<u8> {
    let [w, h] = OUTPUT;
    let left = bar_x(f) + shift;
    (0..h)
        .flat_map(|_| {
            (0..w).flat_map(move |x| {
                let v = if (left..left + BAR_WIDTH).contains(&x) {
                    BAR
                } else {
                    BACKGROUND
                };
                [v, v, v, 255]
            })
        })
        .collect()
}

/// RG16F motion, previous UV minus current UV, where the bar was rendered.
fn motion(f: u32) -> Vec<u8> {
    let [w, h] = OUTPUT;
    let left = bar_x(f);
    let step = -(SPEED as f32) / w as f32;
    let values: Vec<f32> = (0..h)
        .flat_map(|_| {
            (0..w).flat_map(move |x| {
                let u = if (left..left + BAR_WIDTH).contains(&x) {
                    step
                } else {
                    0.0
                };
                [u, 0.0]
            })
        })
        .collect();
    harness::f16_bytes(&values)
}

/// R32F reverse-Z depth: the bar nearer than the background.
fn depth(f: u32) -> Vec<u8> {
    let [w, h] = OUTPUT;
    let left = bar_x(f);
    (0..h)
        .flat_map(|_| {
            (0..w).flat_map(move |x| {
                let d: f32 = if (left..left + BAR_WIDTH).contains(&x) {
                    0.02
                } else {
                    0.005
                };
                d.to_le_bytes()
            })
        })
        .collect()
}

/// The field for a presented image shifted `shift` pixels right: at each presented pixel
/// the UV offset back to its rendered position, and the reverse.
fn field(shift: u32) -> Vec<u8> {
    let [w, h] = OUTPUT.map(|v| v.div_ceil(2));
    let du = shift as f32 / OUTPUT[0] as f32;
    let values: Vec<f32> = (0..w * h).flat_map(|_| [-du, 0.0, du, 0.0]).collect();
    harness::f16_bytes(&values)
}

/// A static camera: 60 degrees vertical, reverse Z with no far plane, 0.1 m near.
fn camera() -> FgCamera {
    let f = 1.0 / (30f32.to_radians()).tan();
    let aspect = OUTPUT[0] as f32 / OUTPUT[1] as f32;
    let near = 0.1;
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    FgCamera {
        view_to_clip: [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, -f, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, near, 0.0],
        ],
        clip_to_view: [
            [aspect / f, 0.0, 0.0, 0.0],
            [0.0, -1.0 / f, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / near],
            [0.0, 0.0, -1.0, 0.0],
        ],
        clip_to_prev_clip: identity,
        prev_clip_to_clip: identity,
        position: [0.0; 3],
        up: [0.0, 1.0, 0.0],
        right: [1.0, 0.0, 0.0],
        forward: [0.0, 0.0, -1.0],
        near,
        far: 100_000.0,
        fov_y: 60f32.to_radians(),
        aspect,
    }
}

/// The middle row's bright span center, pixels, in a BGRA8 image.
fn bar_center(bgra: &[u8]) -> f64 {
    let [w, h] = OUTPUT;
    let row = (h / 2 * w * 4) as usize;
    let threshold = (u32::from(BAR) + u32::from(BACKGROUND)) / 2;
    let (mut sum, mut count) = (0.0, 0.0);
    for x in 0..w as usize {
        let v = u32::from(bgra[row + 4 * x + 1]);
        if v > threshold {
            sum += x as f64 + 0.5;
            count += 1.0;
        }
    }
    assert!(count > 0.0, "no bar in the generated image");
    sum / count
}

/// Runs `FRAMES` frames generating `count` images each, the presented image shifted `shift`
/// pixels with the field passed when `with_field`. Returns the bar centers in the last
/// frame's generated images, in order.
fn run(device: &Device, sdk: &DlssSdk, count: u32, shift: u32, with_field: bool) -> Vec<f64> {
    let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
    let texture = |format, size: [u32; 2], name| {
        let desc = TextureDesc::d2(format, size[0], size[1], usage);
        device
            .create_texture(&desc, TextureMemory::Dedicated, name)
            .expect("texture")
    };
    let backbuffer = texture(Format::Bgra8Unorm, OUTPUT, "fg backbuffer");
    let depth_t = texture(Format::R32Float, OUTPUT, "fg depth");
    let motion_t = texture(Format::Rg16Float, OUTPUT, "fg motion");
    let field_t = texture(
        Format::Rgba16Float,
        OUTPUT.map(|v| v.div_ceil(2)),
        "fg distortion",
    );
    let generated: Vec<Texture> = (0..count)
        .map(|_| texture(Format::Bgra8Unorm, OUTPUT, "fg generated"))
        .collect();
    let mut fg = FrameGeneration::new(sdk, &backbuffer, false).expect("FG create");
    let mut flag: Allocation<u32> = device
        .gpu_malloc(MemoryKind::Readback, 1, "fg flag")
        .expect("flag");
    let pixels = u64::from(OUTPUT[0] * OUTPUT[1]);
    let readbacks: Vec<Allocation<u8>> = (0..count)
        .map(|_| {
            device
                .gpu_malloc(MemoryKind::Readback, pixels * 4, "fg readback")
                .expect("readback")
        })
        .collect();
    let mut uploads = Vec::new();
    let mut upload = |list: &mut CommandList<'_>, t: &Texture, bytes: &[u8]| {
        let mut a: Allocation<u8> = device
            .gpu_malloc(MemoryKind::Upload, bytes.len() as u64, "fg upload")
            .expect("upload");
        a.as_mut_slice().unwrap().copy_from_slice(bytes);
        list.copy_to_texture(t, &TextureRegion::whole_mip(t.desc(), 0), a.gpu);
        uploads.push(a);
    };

    let queue = device.queue(QueueKind::Render).unwrap();
    for f in 0..FRAMES {
        let mut list = queue.command_list().expect("list");
        if f == 0 {
            for t in [&backbuffer, &depth_t, &motion_t, &field_t]
                .into_iter()
                .chain(&generated)
            {
                list.discard(t);
            }
            upload(&mut list, &field_t, &field(shift));
        }
        upload(&mut list, &backbuffer, &color(f, shift));
        upload(&mut list, &motion_t, &motion(f));
        upload(&mut list, &depth_t, &depth(f));
        flag.as_mut_slice().unwrap()[0] = 0;
        list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
        for (k, out) in generated.iter().enumerate() {
            let inputs = FgInputs {
                backbuffer: &backbuffer,
                depth: &depth_t,
                motion: &motion_t,
                distortion: with_field.then_some(&field_t),
                interpolated: out,
                disable_interpolation: Some(&flag),
                camera: camera(),
                jitter: [0.0; 2],
                mvec_scale: OUTPUT.map(|v| v as f32),
                reset: f == 0,
                frame_id: u64::from(f) + 1,
                not_rendering_scene: false,
                depth_scale: 1.0,
            };
            fg.record(&mut list, &inputs, k as u32 + 1, count)
                .expect("FG evaluate");
        }
        list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
        if f + 1 == FRAMES {
            for (t, r) in generated.iter().zip(&readbacks) {
                list.copy_from_texture(r.gpu, t, &TextureRegion::whole_mip(t.desc(), 0));
            }
        }
        let done = queue.submit(vec![list], &[]).expect("submit");
        queue.wait(done, Duration::from_secs(30)).expect("wait");
        if f + 1 == FRAMES {
            assert!(
                !FrameGeneration::interpolation_disabled(&flag),
                "NGX disabled interpolation on a steady frame"
            );
        }
    }
    device.collect_garbage();
    readbacks
        .iter()
        .map(|r| bar_center(r.as_slice().unwrap()))
        .collect()
}

/// Where generated image `k` (1-based) of `count` puts the bar's center in the last frame.
fn expected(k: u32, count: u32, shift: u32) -> f64 {
    let previous = f64::from(bar_x(FRAMES - 2) + shift) + f64::from(BAR_WIDTH) / 2.0;
    previous + f64::from(SPEED) * f64::from(k) / f64::from(count + 1)
}

fn fg_sdk(device: &Device) -> Option<DlssSdk> {
    let sdk = DlssSdk::new(device, harness::PROJECT_ID).expect("NGX init on an orr device");
    let support = sdk.support();
    if !support.fg {
        eprintln!("skipped: NGX offers no frame generation on this device");
        return None;
    }
    Some(sdk)
}

#[test]
fn generated_images_split_the_step() {
    let device = device();
    let Some(sdk) = fg_sdk(&device) else {
        return;
    };
    let most = sdk.support().max_generated.min(3);
    let mut counts = vec![1];
    if most > 1 {
        counts.push(most);
    }
    for count in counts {
        let centers = run(&device, &sdk, count, 0, false);
        eprintln!("{}x: bar centers {centers:?}", count + 1);
        for (k, c) in (1..=count).zip(&centers) {
            let want = expected(k, count, 0);
            assert!(
                (c - want).abs() <= TOLERANCE,
                "{}x image {k}: bar at {c:.1}, expected {want:.1}",
                count + 1
            );
        }
    }
    assert_eq!(device.validation_messages(), 0, "validation messages");
}

#[test]
fn bound_distortion_field_keeps_a_shifted_bar_in_place() {
    let device = device();
    let Some(sdk) = fg_sdk(&device) else {
        return;
    };
    let shift = 96;
    let with = run(&device, &sdk, 1, shift, true)[0];
    let want = expected(1, 1, shift);
    assert!(
        (with - want).abs() <= TOLERANCE,
        "with the field the bar is at {with:.1}, expected {want:.1}"
    );
    assert_eq!(device.validation_messages(), 0, "validation messages");
}
