//! Ray reconstruction on a device that binds descriptor heaps, with synthetic noisy inputs.
#![cfg(ngx)]
#![allow(clippy::print_stderr)]

mod harness;

use ash::vk;
use harness::{Gpu, f16_bytes, f16_values};
use orr_dlss::Quality;
use orr_dlss::raw::{ClassicBind, Image, Ngx, RayReconstruction, RrInputs};

/// A value RR never writes: negative alpha, and channel values no denoised pixel matches.
const SENTINEL: [f32; 4] = [-3.0, 0.125, -5.0, -7.0];

const PALETTE: [[f32; 3]; 4] = [
    [0.8, 0.3, 0.2],
    [0.2, 0.7, 0.3],
    [0.25, 0.35, 0.9],
    [0.75, 0.75, 0.7],
];

/// Noise-free radiance: 64-pixel tiles of four albedos lit by a vertical ramp.
fn base(x: u32, y: u32, render: [u32; 2]) -> [f32; 3] {
    let albedo = PALETTE[((x / 64 + y / 64) % 4) as usize];
    let light = 0.25 + 0.75 * (y as f32 + 0.5) / render[1] as f32;
    albedo.map(|a| a * light)
}

fn hash(mut v: u32) -> u32 {
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb_352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846c_a68b);
    v ^ (v >> 16)
}

/// One sample per pixel: the base times a uniform [0, 2) factor, so its mean is the base.
fn noisy_color(render: [u32; 2], frame: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity((render[0] * render[1] * 4) as usize);
    for y in 0..render[1] {
        for x in 0..render[0] {
            let n = hash(x ^ hash(y ^ hash(frame as u32 ^ 0x9e37_79b9)));
            let u = (n >> 8) as f32 / (1u32 << 24) as f32 * 2.0;
            let b = base(x, y, render);
            out.extend_from_slice(&[b[0] * u, b[1] * u, b[2] * u, 1.0]);
        }
    }
    out
}

fn mean_base(render: [u32; 2]) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    for y in 0..render[1] {
        for x in 0..render[0] {
            let b = base(x, y, render);
            (0..3).for_each(|c| sum[c] += f64::from(b[c]));
        }
    }
    sum.map(|s| s / f64::from(render[0] * render[1]))
}

struct Targets {
    color: Image,
    depth: Image,
    motion: Image,
    diffuse: Image,
    specular: Image,
    normals: Image,
    specular_motion: Image,
    overlay: Image,
}

/// Uploads the static inputs and returns the targets and the colour staging buffer.
fn targets(
    gpu: &mut Gpu,
    render: [u32; 2],
    overlay: Option<([u32; 4], [f32; 4])>,
) -> (Targets, usize) {
    let px = (render[0] * render[1]) as usize;
    let t = Targets {
        color: gpu.image(render, vk::Format::R16G16B16A16_SFLOAT),
        depth: gpu.image(render, vk::Format::R32_SFLOAT),
        motion: gpu.image(render, vk::Format::R16G16_SFLOAT),
        diffuse: gpu.image(render, vk::Format::R8G8B8A8_UNORM),
        specular: gpu.image(render, vk::Format::R8G8B8A8_UNORM),
        normals: gpu.image(render, vk::Format::R16G16B16A16_SFLOAT),
        specular_motion: gpu.image(render, vk::Format::R16G16_SFLOAT),
        overlay: gpu.image(render, vk::Format::R16G16B16A16_SFLOAT),
    };
    let mut diffuse = Vec::with_capacity(px * 4);
    for y in 0..render[1] {
        for x in 0..render[0] {
            let a = PALETTE[((x / 64 + y / 64) % 4) as usize];
            diffuse.extend(a.map(|v| (v * 255.0).round() as u8));
            diffuse.push(255);
        }
    }
    let mut over = vec![0.0f32; px * 4];
    if let Some((rect, value)) = overlay {
        for y in rect[1]..rect[3] {
            for x in rect[0]..rect[2] {
                let i = ((y * render[0] + x) * 4) as usize;
                over[i..i + 4].copy_from_slice(&value);
            }
        }
    }
    let uploads: Vec<(&Image, Vec<u8>)> = vec![
        (&t.depth, 0.5f32.to_le_bytes().repeat(px)),
        (&t.motion, vec![0; px * 4]),
        (&t.specular_motion, vec![0; px * 4]),
        (&t.diffuse, diffuse),
        (&t.specular, [10u8, 10, 10, 255].repeat(px)),
        (&t.normals, f16_bytes(&[0.0, 0.0, 1.0, 0.5]).repeat(px)),
        (&t.overlay, f16_bytes(&over)),
    ];
    let staging: Vec<usize> = uploads
        .iter()
        .map(|(_, b)| gpu.buffer(b.len() as u64))
        .collect();
    let color_staging = gpu.buffer(px as u64 * 8);
    gpu.begin();
    let all = [
        &t.color,
        &t.depth,
        &t.motion,
        &t.diffuse,
        &t.specular,
        &t.normals,
        &t.specular_motion,
        &t.overlay,
    ];
    gpu.to_general(&all);
    for ((image, bytes), &s) in uploads.iter().zip(&staging) {
        gpu.upload(s, image, bytes);
    }
    gpu.submit_wait();
    (t, color_staging)
}

fn inputs<'a>(
    t: &'a Targets,
    output: &'a Image,
    overlay: bool,
    jitter: [f32; 2],
    reset: bool,
) -> RrInputs<'a> {
    RrInputs {
        color: &t.color,
        color_before_transparency: None,
        depth: &t.depth,
        motion: &t.motion,
        diffuse_albedo: &t.diffuse,
        specular_albedo: &t.specular,
        normal_roughness: &t.normals,
        specular_motion: &t.specular_motion,
        transparency: overlay.then_some(&t.overlay),
        depth_of_field_guide: None,
        output,
        reset,
        jitter,
    }
}

fn create(gpu: &Gpu, ngx: &Ngx, output: [u32; 2], quality: Quality) -> RayReconstruction {
    gpu.begin();
    // SAFETY: the command buffer is recording on NGX's device; the feature is dropped before
    // NGX and after the queue is idle.
    let rr = unsafe { RayReconstruction::new(ngx, gpu.cmd, output, quality, None) }.unwrap_or_else(
        |e| {
            for l in Ngx::take_log() {
                eprintln!("NGX: {l}");
            }
            panic!("RR create: {e} {:?}", ngx.support())
        },
    );
    gpu.submit_wait();
    rr
}

/// Runs `frames` frames of every feature on the same noisy inputs, each frame with heaps
/// bound. The last frame writes a fresh output cleared to the sentinel; returns those outputs.
fn run(
    gpu: &mut Gpu,
    features: &mut [(RayReconstruction, bool)],
    t: &Targets,
    color_staging: usize,
    frames: u64,
    (warm, last): &(Vec<Image>, Vec<Image>),
) -> Vec<Vec<f32>> {
    let classic = ClassicBind::new(&gpu.device).expect("classic bind");
    let readbacks: Vec<usize> = last
        .iter()
        .map(|o| gpu.buffer(u64::from(o.extent[0] * o.extent[1]) * 8))
        .collect();
    let render = features[0].0.render_size();

    // A heap bound in an earlier submission on the queue.
    gpu.begin();
    gpu.bind_heaps();
    gpu.to_general(&warm.iter().chain(last).collect::<Vec<_>>());
    for o in last {
        gpu.clear(o, SENTINEL);
    }
    gpu.submit_wait();

    for frame in 0..frames {
        let outputs = if frame + 1 == frames { last } else { warm };
        gpu.begin();
        gpu.bind_heaps();
        gpu.upload(
            color_staging,
            &t.color,
            &f16_bytes(&noisy_color(render, frame)),
        );
        gpu.barrier();
        // SAFETY: recording on the device the bind was created on.
        unsafe { classic.record(&gpu.device, gpu.cmd) };
        for ((rr, overlay), output) in features.iter_mut().zip(outputs) {
            let jitter = rr.jitter(frame);
            // SAFETY: recording on NGX's device; every image lives as long as `gpu`.
            unsafe { rr.record(gpu.cmd, &inputs(t, output, *overlay, jitter, frame == 0)) }
                .expect("RR evaluate");
        }
        gpu.barrier();
        if frame + 1 == frames {
            for (o, &r) in last.iter().zip(&readbacks) {
                gpu.download(o, r);
            }
        }
        gpu.submit_wait();
    }
    // SAFETY: the queue is idle.
    unsafe { classic.destroy(&gpu.device) };
    last.iter()
        .zip(&readbacks)
        .map(|(o, &r)| f16_values(gpu.bytes(r), (o.extent[0] * o.extent[1] * 4) as usize))
        .collect()
}

/// Asserts RR wrote every pixel with a finite value and kept the input's mean radiance.
fn check_output(out: &[f32], render: [u32; 2]) {
    let pixels: Vec<&[f32]> = out.chunks_exact(4).collect();
    let untouched = pixels.iter().filter(|p| **p == SENTINEL).count();
    assert_eq!(
        untouched,
        0,
        "{untouched} of {} pixels kept the sentinel",
        pixels.len()
    );
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    let expect = mean_base(render);
    for c in 0..3 {
        let mean = pixels.iter().map(|p| f64::from(p[c])).sum::<f64>() / pixels.len() as f64;
        let rel = (mean - expect[c]).abs() / expect[c];
        assert!(
            rel < 0.1,
            "channel {c} mean {mean:.4}, input mean {:.4}",
            expect[c]
        );
    }
}

/// Output images for `count` features: one for the frames before the last, one for the last.
fn outputs(gpu: &mut Gpu, size: [u32; 2], count: usize) -> (Vec<Image>, Vec<Image>) {
    let mut make = || {
        (0..count)
            .map(|_| gpu.image(size, vk::Format::R16G16B16A16_SFLOAT))
            .collect()
    };
    (make(), make())
}

fn validation_clean(gpu: &Gpu) {
    if gpu.validation {
        assert_eq!(Gpu::validation_messages(), 0, "validation messages");
        eprintln!("NGX-internal sync hazards: {}", Gpu::ngx_hazards());
    }
}

#[test]
fn rr_performance_and_dlaa_overwrite_every_pixel_after_heap_binds() {
    let mut gpu = Gpu::new(true);
    // SAFETY: the device has NGX's extensions and outlives `ngx`.
    let ngx = unsafe { Ngx::new(gpu.raw(), harness::PROJECT_ID) }.expect("NGX init");
    assert!(ngx.support().rr, "ray reconstruction unavailable");

    let uhd = outputs(&mut gpu, [3840, 2160], 1);
    let hd = outputs(&mut gpu, [1920, 1080], 1);
    let rr = create(&gpu, &ngx, [3840, 2160], Quality::Performance);
    assert_eq!(rr.render_size(), [1920, 1080]);
    let (t, staging) = targets(&mut gpu, rr.render_size(), None);
    let out = run(&mut gpu, &mut [(rr, false)], &t, staging, 8, &uhd);
    check_output(&out[0], [1920, 1080]);
    let log = Ngx::take_log();
    eprintln!(
        "applied preset {:?}, vram {}",
        Ngx::applied_preset(&log),
        ngx.vram_bytes()
    );

    let dlaa = create(&gpu, &ngx, [1920, 1080], Quality::Dlaa);
    assert_eq!(dlaa.render_size(), [1920, 1080]);
    let out = run(&mut gpu, &mut [(dlaa, false)], &t, staging, 8, &hd);
    check_output(&out[0], [1920, 1080]);
    validation_clean(&gpu);
}

#[test]
fn transparency_layer_is_composited_after_denoising() {
    let mut gpu = Gpu::new(true);
    // SAFETY: as above.
    let ngx = unsafe { Ngx::new(gpu.raw(), harness::PROJECT_ID) }.expect("NGX init");
    let output = [1920, 1080];
    let images = outputs(&mut gpu, output, 2);
    let plain = create(&gpu, &ngx, output, Quality::Performance);
    let layered = create(&gpu, &ngx, output, Quality::Performance);
    let render = plain.render_size();
    assert_eq!(render, [960, 540]);
    // Premultiplied colour, coverage 0.5.
    let overlay = [0.3, 0.1, 0.05, 0.5];
    let rect = [300, 150, 600, 400];
    let (t, staging) = targets(&mut gpu, render, Some((rect, overlay)));
    let out = run(
        &mut gpu,
        &mut [(plain, false), (layered, true)],
        &t,
        staging,
        16,
        &images,
    );

    let (mut sum, mut worst, mut n) = (0.0f64, 0.0f64, 0);
    let inset = 8;
    for y in rect[1] * 2 + inset..rect[3] * 2 - inset {
        for x in rect[0] * 2 + inset..rect[2] * 2 - inset {
            let i = ((y * output[0] + x) * 4) as usize;
            for c in 0..3 {
                let composite = f64::from(overlay[c])
                    + (1.0 - f64::from(overlay[3])) * f64::from(out[0][i + c]);
                let rel = (f64::from(out[1][i + c]) - composite).abs() / composite;
                sum += rel;
                worst = worst.max(rel);
                n += 1;
            }
        }
    }
    eprintln!(
        "overlay: mean rel {:.5}, worst {:.5}",
        sum / f64::from(n),
        worst
    );
    assert!(
        sum / f64::from(n) < 0.02,
        "mean relative error {}",
        sum / f64::from(n)
    );
    validation_clean(&gpu);
}
