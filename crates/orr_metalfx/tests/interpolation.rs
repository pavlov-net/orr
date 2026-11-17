//! Frame interpolation lands between its two frames: a pattern moving right at constant speed,
//! interpolated between frame `t` and `t + 1`, comes out close to the pattern at `t + 1/2`, far
//! closer than either frame it was made from, and closer than with the motion flipped.

#![cfg(target_vendor = "apple")]

use std::f32::consts::TAU;
use std::time::Duration;

use half::f16;
use orr_gpu::{
    Device, DeviceDesc, Format, Hazards, MemoryKind, QueueKind, Stages, Texture, TextureDesc,
    TextureMemory, TextureRegion, TextureUsage,
};
use orr_metalfx::{Float4x4, FrameInterpolator, InterpolatorInputs};

const EXTENT: [u32; 2] = [128, 128];
/// Render pixels per frame.
const SPEED: f32 = 8.0;
const FRAMES: u32 = 8;

/// The pattern at pixel `[x, y]` (pixel centers at .5) on frame `t`.
fn pattern([x, y]: [f32; 2], t: f32) -> f32 {
    let [w, h] = EXTENT.map(|e| e as f32);
    let u = (x - SPEED * t) / w;
    0.5 + 0.3 * (TAU * 3.0 * u).sin() * (TAU * 2.0 * y / h).sin()
}

struct Bed {
    device: Device,
    frames: [Texture; 2],
    depth: Texture,
    motion: Texture,
    output: Texture,
}

impl Bed {
    fn new() -> Self {
        let device = Device::new(&DeviceDesc {
            name: "orr_metalfx interpolation",
            ..DeviceDesc::default()
        })
        .expect("a device");
        let texture = |format, name| {
            // MetalFX writes its output as a render target.
            let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE | TextureUsage::COLOR;
            device
                .create_texture(
                    &TextureDesc::d2(format, EXTENT[0], EXTENT[1], usage),
                    TextureMemory::Dedicated,
                    name,
                )
                .expect("a texture")
        };
        Self {
            frames: [
                texture(Format::Rgba16Float, "frame a"),
                texture(Format::Rgba16Float, "frame b"),
            ],
            depth: texture(Format::R32Float, "depth"),
            motion: texture(Format::Rg16Float, "motion"),
            output: texture(Format::Rgba16Float, "interpolated"),
            device,
        }
    }

    fn upload(
        &self,
        list: &mut orr_gpu::CommandList<'_>,
        t: &Texture,
        bytes: &[u8],
    ) -> orr_gpu::Allocation<u8> {
        let mut up = self
            .device
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, "upload")
            .expect("upload memory");
        up.as_mut_slice().expect("mapped").copy_from_slice(bytes);
        list.copy_to_texture(t, &TextureRegion::whole_mip(t.desc(), 0), up.gpu());
        up
    }

    /// The RMS difference from the pattern at the frame halfway between the last two, of the
    /// interpolated image with the true motion times `motion_sign`, and of the last frame.
    fn errors(&self, motion_sign: f32) -> (f32, f32) {
        let mut fi = FrameInterpolator::new(&self.device).expect("an interpolator");
        let queue = self
            .device
            .queue(QueueKind::Render)
            .expect("the Render queue");
        let [w, h] = EXTENT.map(|e| e as usize);
        let identity = Float4x4::from_cols_array_2d(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let image = |t: f32| -> Vec<f16> {
            (0..w * h)
                .flat_map(|i| {
                    let c = f16::from_f32(pattern([(i % w) as f32 + 0.5, (i / w) as f32 + 0.5], t));
                    [c, c, c, f16::ONE]
                })
                .collect()
        };
        // Previous UV minus current UV of content moving right.
        let mx = f16::from_f32(motion_sign * -SPEED / EXTENT[0] as f32);
        let motion: Vec<f16> = (0..w * h).flat_map(|_| [mx, f16::ZERO]).collect();
        let depth = vec![0.5f32; w * h];
        for frame in 0..FRAMES {
            let (cur, prev) = (
                &self.frames[frame as usize % 2],
                &self.frames[(frame as usize + 1) % 2],
            );
            let mut list = queue.command_list().expect("a list");
            let color = image(frame as f32);
            let _held = [
                self.upload(&mut list, cur, bytemuck::cast_slice(&color)),
                self.upload(&mut list, &self.motion, bytemuck::cast_slice(&motion)),
                self.upload(&mut list, &self.depth, bytemuck::cast_slice(&depth)),
            ];
            list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
            fi.record(
                &mut list,
                &InterpolatorInputs {
                    color: cur,
                    previous_color: if frame == 0 { cur } else { prev },
                    depth: &self.depth,
                    motion: &self.motion,
                    distortion: None,
                    output: &self.output,
                    reset: frame == 0,
                    jitter: [0.0; 2],
                    delta_time: 1.0 / 60.0,
                    near: 0.1,
                    far: 100_000.0,
                    fov_y: 1.0,
                    aspect: 1.0,
                    world_to_view: identity,
                    view_to_clip: identity,
                },
            )
            .expect("a record");
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            let point = queue.submit(vec![list], &[]).expect("a submit");
            queue
                .wait(point, Duration::from_secs(10))
                .expect("the frame");
        }
        let mut back = self
            .device
            .gpu_malloc::<f16>(MemoryKind::Readback, (w * h * 4) as u64, "readback")
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
        let mid = FRAMES as f32 - 1.5;
        let (mut interpolated, mut last, mut n) = (0.0, 0.0, 0);
        // Away from the left and right edges, where content enters and leaves.
        for y in 0..h {
            for x in 16..w - 16 {
                let at = [x as f32 + 0.5, y as f32 + 0.5];
                let want = pattern(at, mid);
                let d = out[(y * w + x) * 4].to_f32() - want;
                let e = pattern(at, FRAMES as f32 - 1.0) - want;
                interpolated += d * d;
                last += e * e;
                n += 1;
            }
        }
        ((interpolated / n as f32).sqrt(), (last / n as f32).sqrt())
    }
}

#[test]
fn interpolation_lands_halfway() {
    let bed = Bed::new();
    let (interpolated, last) = bed.errors(1.0);
    let (flipped, _) = bed.errors(-1.0);
    assert!(
        interpolated < 0.5 * last,
        "interpolated {interpolated}, last frame {last}, motion flipped {flipped}"
    );
    assert!(
        interpolated < 0.5 * flipped,
        "interpolated {interpolated}, last frame {last}, motion flipped {flipped}"
    );
}
