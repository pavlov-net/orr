//! Times `orr.openpbr` on the GPU (`gpu/openpbr_bench.slang`, heap LUTs) and prints a
//! Markdown table of nanoseconds per call: prepare, eval, sample, and eval plus sample, for
//! each harness material and for all of them mixed across neighbouring threads.
//!
//! `openpbr-bench [--validation]`. Timings are GPU timestamps around each dispatch, the
//! median of five after one warm-up; run without the validation layer for numbers.

use std::time::Duration;

use openpbr_ref::gpu::openpbr_bench::{MODULE, Root, Spec};
use openpbr_ref::materials::grid;
use openpbr_ref::{Inputs, Luts};
use orr_gpu::{
    ComputePipeline, Device, DeviceDesc, GpuPtr, MemoryKind, QueueKind, ShaderStage, SpecData,
};

const THREADS: u32 = 1 << 20;
const CALLS: u32 = 16;
const EVAL: u32 = 1;
const SAMPLE: u32 = 2;
const RUNS: usize = 5;

fn main() {
    openpbr_ref::log_to_stderr();
    let validation = std::env::args().any(|a| a == "--validation");
    let device = Device::new(&DeviceDesc {
        name: "openpbr-bench",
        validation,
        ..DeviceDesc::default()
    })
    .expect("Vulkan device");
    let luts = Luts::upload(&device);
    let spec = Spec {
        openpbr_lut_slot: luts.first_slot,
        ..Spec::default()
    };
    let bytes = spec.bytes();
    let stage = ShaderStage::new(
        &MODULE,
        "main",
        SpecData {
            entries: Spec::ENTRIES,
            bytes: &bytes,
        },
    )
    .expect("openpbr_bench stage");
    let pipeline = device
        .create_compute_pipeline(&stage, "openpbr_bench")
        .expect("openpbr_bench pipeline");

    let materials = grid();
    let inputs: Vec<Inputs> = materials.iter().map(|m| m.inputs).collect();
    let mut table = device
        .gpu_malloc::<Inputs>(MemoryKind::Upload, inputs.len() as u64, "bench materials")
        .expect("materials");
    table
        .as_mut_slice()
        .expect("mapped")
        .copy_from_slice(&inputs);
    let sink = device
        .gpu_malloc::<f32>(MemoryKind::DeviceLocal, u64::from(THREADS), "bench sink")
        .expect("sink");
    let mut roots = device
        .gpu_malloc::<Root>(MemoryKind::Upload, 4, "bench roots")
        .expect("roots");
    let mut out = String::from(
        "| Material | prepare | eval | sample | eval + sample |\n| --- | --- | --- | --- | --- |\n",
    );
    let rows = materials
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name, table.gpu.offset(i as u64), 1, 1))
        .chain([("all, mixed per thread", table.gpu, inputs.len() as u32, 0)]);
    for (name, first, count, coherent) in rows {
        let modes = [0, EVAL, SAMPLE, EVAL | SAMPLE];
        let slots = roots.as_mut_slice().expect("mapped");
        for (slot, mode) in slots.iter_mut().zip(modes) {
            *slot = Root {
                materials: first,
                sink: sink.gpu,
                material_count: count,
                coherent,
                calls: CALLS,
                mode,
                threads: THREADS,
                seed: 1,
                _pad0: 0,
                _pad1: 0,
            };
        }
        let ms: Vec<f64> = (0..modes.len())
            .map(|k| median_ms(&device, &pipeline, roots.gpu.offset(k as u64)))
            .collect();
        let per = |t: f64| t * 1e6 / f64::from(THREADS);
        let per_call = |t: f64| (t - ms[0]).max(0.0) * 1e6 / f64::from(THREADS * CALLS);
        out += &format!(
            "| {name} | {:.2} | {:.3} | {:.3} | {:.3} |\n",
            per(ms[0]),
            per_call(ms[1]),
            per_call(ms[2]),
            per_call(ms[3])
        );
    }
    #[allow(clippy::print_stdout)]
    {
        println!(
            "orr.openpbr on {THREADS} threads, {CALLS} calls each: ns per thread for prepare, per call otherwise.\n"
        );
        print!("{out}");
    }
    if validation {
        assert_eq!(
            device.validation_messages(),
            0,
            "validation messages (see the log)"
        );
    }
}

/// The median GPU time of `RUNS` dispatches after one warm-up.
fn median_ms(device: &Device, pipeline: &ComputePipeline, root: GpuPtr<Root>) -> f64 {
    let queue = device.queue(QueueKind::Render).expect("render queue");
    let mut times: Vec<f64> = (0..=RUNS)
        .map(|_| {
            let mut list = queue.command_list().expect("command list");
            list.begin_pass("openpbr bench", false);
            list.dispatch(pipeline, root, [THREADS.div_ceil(64), 1, 1]);
            list.end_pass();
            let point = queue.submit(vec![list], &[]).expect("submit");
            queue
                .wait(point, Duration::from_secs(60))
                .expect("openpbr_bench");
            device.pass_results(point).expect("completed")[0].gpu_ms
        })
        .skip(1)
        .collect();
    times.sort_by(f64::total_cmp);
    times[RUNS / 2]
}
