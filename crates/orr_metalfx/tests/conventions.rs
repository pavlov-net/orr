//! The input conventions in `orr_metalfx::conventions`, measured: a pattern above the render
//! resolution's Nyquist limit and below the output's, sampled the way orr's projection samples it,
//! comes back from MetalFX far closer to the pattern with orr's jitter and motion mapped as the
//! crate maps them than with either one flipped.

#![cfg(target_vendor = "apple")]

use std::f32::consts::TAU;
use std::time::Duration;

use half::f16;
use orr_gpu::{
    Device, DeviceDesc, Format, Hazards, MemoryKind, QueueKind, Stages, Texture, TextureDesc,
    TextureMemory, TextureRegion, TextureUsage,
};
use orr_math::jitter::{jitter, jitter_phases};
use orr_metalfx::{TemporalInputs, TemporalScaler};

const RENDER: [u32; 2] = [64, 64];
const OUTPUT: [u32; 2] = [128, 128];
const FRAMES: u64 = 64;
/// Cycles across the image on each axis: above the render's 32, below the output's 64.
const CYCLES: [f32; 2] = [44.0, 40.0];

/// The pattern at `[u, v]` in [0, 1], moved right by `shift` of the image width.
fn pattern([u, v]: [f32; 2], shift: f32) -> f32 {
    0.5 + 0.4 * (TAU * CYCLES[0] * (u - shift)).sin() * (TAU * CYCLES[1] * v).sin()
}

struct Probe {
    device: Device,
    color: Texture,
    depth: Texture,
    motion: Texture,
    output: Texture,
}

impl Probe {
    fn new() -> Self {
        let device = Device::new(&DeviceDesc {
            name: "orr_metalfx conventions",
            ..DeviceDesc::default()
        })
        .expect("a device");
        let texture = |format, [w, h]: [u32; 2], name| {
            // MetalFX writes its output as a render target.
            let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE | TextureUsage::COLOR;
            device
                .create_texture(
                    &TextureDesc::d2(format, w, h, usage),
                    TextureMemory::Dedicated,
                    name,
                )
                .expect("a texture")
        };
        Self {
            color: texture(Format::Rgba16Float, RENDER, "probe color"),
            depth: texture(Format::R32Float, RENDER, "probe depth"),
            motion: texture(Format::Rg16Float, RENDER, "probe motion"),
            output: texture(Format::Rgba16Float, OUTPUT, "probe output"),
            device,
        }
    }

    /// The RMS difference between MetalFX's last output and the pattern at its final position,
    /// for content moving `velocity` render pixels right per frame. The scaler gets
    /// `jitter_sign` times orr's jitter and `motion_sign` times the true motion.
    fn error(&self, velocity: f32, jitter_sign: f32, motion_sign: f32) -> f32 {
        let mut scaler = TemporalScaler::new(&self.device, RENDER, OUTPUT).expect("a scaler");
        let queue = self
            .device
            .queue(QueueKind::Render)
            .expect("the Render queue");
        let phases = jitter_phases(RENDER, OUTPUT);
        let [w, h] = RENDER.map(|e| e as usize);
        let shift = |frame: u64| frame as f32 * velocity / RENDER[0] as f32;
        let mut last = None;
        for frame in 0..FRAMES {
            let j = jitter(frame, phases);
            let mut color = Vec::with_capacity(w * h * 4);
            for y in 0..h {
                for x in 0..w {
                    let at = [
                        (x as f32 + 0.5 - j[0]) / w as f32,
                        (y as f32 + 0.5 - j[1]) / h as f32,
                    ];
                    let c = f16::from_f32(pattern(at, shift(frame)));
                    color.extend([c, c, c, f16::ONE]);
                }
            }
            // Previous UV minus current UV of content moving right.
            let mx = f16::from_f32(motion_sign * -velocity / RENDER[0] as f32);
            let motion: Vec<f16> = (0..w * h).flat_map(|_| [mx, f16::ZERO]).collect();
            let depth = vec![0.5f32; w * h];
            let mut list = queue.command_list().expect("a list");
            let mut uploads = Vec::new();
            for (t, bytes) in [
                (&self.color, bytemuck::cast_slice::<f16, u8>(&color)),
                (&self.motion, bytemuck::cast_slice(&motion)),
                (&self.depth, bytemuck::cast_slice(&depth)),
            ] {
                let mut up = self
                    .device
                    .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, "probe upload")
                    .expect("upload memory");
                up.as_mut_slice().expect("mapped").copy_from_slice(bytes);
                list.copy_to_texture(t, &TextureRegion::whole_mip(t.desc(), 0), up.gpu());
                uploads.push(up);
            }
            list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
            scaler
                .record(
                    &mut list,
                    &TemporalInputs {
                        color: &self.color,
                        depth: &self.depth,
                        motion: &self.motion,
                        exposure: None,
                        pre_exposure: 1.0,
                        reactive_mask: None,
                        output: &self.output,
                        reset: frame == 0,
                        jitter: j.map(|c| jitter_sign * c),
                    },
                )
                .expect("a record");
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            let point = queue.submit(vec![list], &[]).expect("a submit");
            queue
                .wait(point, Duration::from_secs(10))
                .expect("the frame");
            last = Some(uploads);
        }
        drop(last);
        let [ow, oh] = OUTPUT.map(|e| e as usize);
        let mut back = self
            .device
            .gpu_malloc::<f16>(MemoryKind::Readback, (ow * oh * 4) as u64, "probe readback")
            .expect("readback memory");
        let mut list = queue.command_list().expect("a list");
        list.copy_from_texture(
            back.gpu().cast(),
            &self.output,
            &TextureRegion::whole_mip(self.output.desc(), 0),
        );
        let point = queue.submit(vec![list], &[]).expect("a submit");
        queue
            .wait(point, Duration::from_secs(10))
            .expect("the readback");
        let out = back.as_mut_slice().expect("mapped");
        // The border, where MetalFX clamps, stays out of the measure.
        let (mut sum, mut n) = (0.0, 0);
        for y in 8..oh - 8 {
            for x in 8..ow - 8 {
                let at = [(x as f32 + 0.5) / ow as f32, (y as f32 + 0.5) / oh as f32];
                let d = out[(y * ow + x) * 4].to_f32() - pattern(at, shift(FRAMES - 1));
                sum += d * d;
                n += 1;
            }
        }
        (sum / n as f32).sqrt()
    }
}

#[test]
fn jitter_and_motion_map_as_measured() {
    let probe = Probe::new();
    let still = probe.error(0.0, 1.0, 1.0);
    let still_flipped = probe.error(0.0, -1.0, 1.0);
    // A whole render pixel per frame: at sub-pixel speeds the aliased pattern changes from frame
    // to frame, and MetalFX drops the history whatever the motion says.
    let moving = probe.error(1.0, 1.0, 1.0);
    let moving_flipped = probe.error(1.0, 1.0, -1.0);
    assert!(
        still < 0.5 * still_flipped,
        "jitter: {still} vs flipped {still_flipped}"
    );
    assert!(
        moving < 0.8 * moving_flipped,
        "motion: {moving} vs flipped {moving_flipped}"
    );
}
