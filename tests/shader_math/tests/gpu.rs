//! `pack_half` and `unpack_half` on the device, bit for bit against `half`, which the host and
//! the cook write and read halves with: they use integer operations alone, so the device's
//! denormal mode must not change a subnormal half. Fails if the validation layer reports
//! anything.

use std::time::Duration;

use half::f16;
use orr_gpu::{
    Allocation, Device, DeviceDesc, Hazards, MemoryKind, QueueKind, ShaderStage, SpecData, Stages,
};
use orr_shaders::bytemuck::Pod;
use shader_math::gpu::halves::{MODULE, Root};
use shader_math::stats::Rng;

/// Every finite half's value; each midpoint between neighboring halves, where rounding ties to
/// even, with the f32s on either side of it, up to 65520 where rounding reaches infinity; f32
/// zeros, subnormals, infinities and NaNs with payloads; then random bit patterns.
fn inputs() -> Vec<f32> {
    let value = |c: u16| f64::from(f16::from_bits(c).to_f32());
    let mut out = Vec::new();
    for c in 0..0x7C00u16 {
        let next = if c == 0x7BFF { 65536.0 } else { value(c + 1) };
        let mid = ((value(c) + next) / 2.0) as f32;
        let b = mid.to_bits();
        for v in [
            value(c) as f32,
            mid,
            f32::from_bits(b - 1),
            f32::from_bits(b + 1),
        ] {
            out.extend([v, -v]);
        }
    }
    out.extend(
        [
            0x0000_0001u32,
            0x0000_1000,
            0x007F_FFFF,
            0x0080_0000,
            0x7F7F_FFFF,
            0x7F80_0000,
            0x7F80_0001,
            0x7F80_2000,
            0x7FC0_0000,
            0x7FFF_FFFF,
            0x7FA0_E000,
        ]
        .into_iter()
        .flat_map(|b| [f32::from_bits(b), f32::from_bits(b | 0x8000_0000)]),
    );
    let mut rng = Rng::new(7);
    out.extend((0..1 << 16).map(|_| f32::from_bits(rng.next_u32())));
    out
}

fn malloc<T: Pod>(device: &Device, kind: MemoryKind, count: usize, name: &str) -> Allocation<T> {
    device
        .gpu_malloc(kind, count as u64, name)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn half_helpers_match_the_host_bit_for_bit_on_the_device() {
    // Validation messages among the warnings.
    let _ = env_logger::Builder::new()
        .parse_filters("warn")
        .parse_default_env()
        .is_test(true)
        .try_init();
    let device = Device::new(&DeviceDesc {
        name: "shader_math",
        ..DeviceDesc::default()
    })
    .expect("Vulkan device");
    let stage = ShaderStage::new(&MODULE, "main", SpecData::default()).expect("halves stage");
    let pipeline = device
        .create_compute_pipeline(&stage, "halves")
        .expect("halves pipeline");

    let values = inputs();
    let mut inputs = malloc::<u32>(&device, MemoryKind::Upload, values.len(), "half inputs");
    for (dst, v) in inputs
        .as_mut_slice()
        .expect("mapped")
        .iter_mut()
        .zip(&values)
    {
        *dst = v.to_bits();
    }
    let packed = malloc::<u32>(&device, MemoryKind::Readback, values.len(), "packed halves");
    let decoded = malloc::<u32>(&device, MemoryKind::Readback, 1 << 16, "decoded halves");
    let mut root = malloc::<Root>(&device, MemoryKind::Upload, 1, "halves root");
    root.as_mut_slice().expect("mapped")[0] = Root {
        inputs: inputs.gpu(),
        packed: packed.gpu(),
        decoded: decoded.gpu(),
        count: values.len() as u32,
        _pad0: 0,
    };
    let queue = device.queue(QueueKind::Render).expect("render queue");
    let mut list = queue.command_list().expect("command list");
    let threads = values.len().max(1 << 16) as u32;
    list.dispatch(&pipeline, root.gpu(), [threads.div_ceil(64), 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let point = queue.submit(vec![list], &[]).expect("submit");
    queue.wait(point, Duration::from_secs(60)).expect("halves");

    for (c, &got) in decoded.as_slice().expect("mapped").iter().enumerate() {
        let want = f16::from_bits(c as u16).to_f32().to_bits();
        assert_eq!(
            got, want,
            "unpack_half({c:#06x}): {got:#010x}, want {want:#010x}"
        );
    }
    for (&v, &got) in values.iter().zip(packed.as_slice().expect("mapped")) {
        let want = u32::from(f16::from_f32(v).to_bits());
        assert_eq!(
            got,
            want,
            "pack_half({:#010x}): {got:#06x}, want {want:#06x}",
            v.to_bits()
        );
    }
    assert_eq!(
        device.validation_messages(),
        0,
        "validation messages (see the log)"
    );
}
