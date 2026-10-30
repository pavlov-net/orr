//! The `mock` backend: no features supported, and ray reconstruction upscales colour.
#![cfg(feature = "mock")]

mod harness;

use ash::vk;
use harness::{Gpu, f16_bytes, f16_values};
use orr_dlss::Quality;
use orr_dlss::raw::{Ngx, RayReconstruction, RrInputs};

#[test]
fn mock_rr_upscales_color_bilinearly() {
    let mut gpu = Gpu::new(false);
    // SAFETY: the device outlives `ngx`.
    let ngx = unsafe { Ngx::new(gpu.raw(), harness::PROJECT_ID) }.expect("mock init");
    assert_eq!(ngx.support(), orr_dlss::Support::default());

    let output = [512, 256];
    gpu.begin();
    // SAFETY: recording on the device; dropped after the queue is idle.
    let mut rr =
        unsafe { RayReconstruction::new(&ngx, gpu.cmd, output, Quality::Performance, None) }
            .unwrap();
    gpu.submit_wait();
    let render = rr.render_size();
    assert_eq!(render, [256, 128]);

    let f16 = vk::Format::R16G16B16A16_SFLOAT;
    let color = gpu.image(render, f16);
    let other = gpu.image(render, f16);
    let out = gpu.image(output, f16);
    let ramp: Vec<f32> = (0..render[1])
        .flat_map(|_| (0..render[0]).flat_map(|x| [x as f32 / 16.0, 1.0, 0.0, 1.0]))
        .collect();
    let staging = gpu.buffer(u64::from(render[0] * render[1]) * 8);
    let readback = gpu.buffer(u64::from(output[0] * output[1]) * 8);
    gpu.begin();
    gpu.to_general(&[&color, &other, &out]);
    gpu.upload(staging, &color, &f16_bytes(&ramp));
    gpu.barrier();
    let inputs = RrInputs {
        color: &color,
        color_before_transparency: None,
        depth: &other,
        motion: &other,
        diffuse_albedo: &other,
        specular_albedo: &other,
        normal_roughness: &other,
        specular_motion: &other,
        transparency: None,
        depth_of_field_guide: None,
        output: &out,
        reset: true,
        jitter: rr.jitter(0),
    };
    // SAFETY: recording on the device; the images live as long as `gpu`.
    unsafe { rr.record(gpu.cmd, &inputs) }.unwrap();
    gpu.barrier();
    gpu.download(&out, readback);
    gpu.submit_wait();

    let got = f16_values(gpu.bytes(readback), (output[0] * output[1] * 4) as usize);
    for x in 0..output[0] {
        // Pixel centres map through the 2x scale; the edges clamp.
        let src = ((x as f32 + 0.5) * 0.5 - 0.5).clamp(0.0, (render[0] - 1) as f32);
        let want = src / 16.0;
        let have = got[(x * 4) as usize];
        assert!((have - want).abs() < 0.02, "x {x}: {have} vs {want}");
    }
    assert!(got.chunks_exact(4).all(|p| p[1] == 1.0));
    if gpu.validation {
        assert_eq!(Gpu::validation_messages(), 0, "validation messages");
    }
}
