//! Nsight Perf SDK counters land on the pass they measured, with each metric in its place, in
//! one frame and across replays, beside exact timestamps. Runs with the `nsight-perf` feature
//! on an NVIDIA GPU with profiling permissions:
//! `cargo test -p gpu_suite --features nsight-perf --test counters`.

use std::time::Duration;

use gpu_suite::shaders::counters;
use orr_gpu::{
    Allocation, ComputePipeline, CounterDesc, CounterFrame, CounterSession,
    DEFAULT_COUNTER_METRICS, Device, DeviceDesc, MemoryKind, PassCounters, PassResult, Queue,
    QueueKind, ShaderStage, SpecData, TimelinePoint,
};

const TIME: &str = "gpu__time_duration.sum";
const ACTIVE: &str = "sm__cycles_active.avg.pct_of_peak_sustained_elapsed";
const ISSUE: &str = "sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed";
const DRAM: &str = "dram__throughput.avg.pct_of_peak_sustained_elapsed";
/// Doesn't fit in one frame with the defaults.
const L1_HITS: &str = "l1tex__t_sector_hit_rate.pct";

/// A short and a long `alu` pass around a `stream` pass, and an empty pass.
struct Bench<'d> {
    queue: &'d Queue,
    alu: ComputePipeline,
    stream: ComputePipeline,
    roots: [Allocation<counters::Root>; 3],
    _data: Allocation<f32>,
}

const N: u32 = 1 << 26;

impl<'d> Bench<'d> {
    fn new(d: &'d Device) -> Self {
        let data = d
            .gpu_malloc::<f32>(MemoryKind::DeviceLocal, 2 * u64::from(N), "counters data")
            .unwrap();
        let root = |iterations: u32| {
            let mut r = d
                .gpu_malloc::<counters::Root>(MemoryKind::Upload, 1, "counters root")
                .unwrap();
            r.as_mut_slice().unwrap()[0] = counters::Root {
                data: data.gpu(),
                n: N,
                iterations,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                pad3: 0,
            };
            r
        };
        let pipeline = |entry| {
            let stage = ShaderStage::new(&counters::MODULE, entry, SpecData::default()).unwrap();
            d.create_compute_pipeline(&stage, entry).unwrap()
        };
        Bench {
            queue: d.queue(QueueKind::Render).unwrap(),
            alu: pipeline("alu"),
            stream: pipeline("stream"),
            roots: [root(50_000), root(200_000), root(0)],
            _data: data,
        }
    }

    /// Submits the frame, without its empty pass when `whole` is false.
    fn frame(&self, whole: bool) -> TimelinePoint {
        let [short, long, copy] = &self.roots;
        let mut list = self.queue.command_list().unwrap();
        list.begin_pass("alu", false);
        list.dispatch(&self.alu, short.gpu(), [4096, 1, 1]);
        list.end_pass();
        list.begin_pass("stream", false);
        list.dispatch(&self.stream, copy.gpu(), [N / 64, 1, 1]);
        list.end_pass();
        list.begin_pass("alu", false);
        list.dispatch(&self.alu, long.gpu(), [4096, 1, 1]);
        list.end_pass();
        if whole {
            list.begin_pass("empty", false);
            list.end_pass();
        }
        self.queue.submit(vec![list], &[]).unwrap()
    }

    /// Counts frames until a collection completes; returns its last frame.
    fn count(&self, s: &mut CounterSession<'_>) -> TimelinePoint {
        for replay in 1.. {
            s.begin_frame().unwrap();
            let point = self.frame(true);
            match s.end_frame().unwrap() {
                CounterFrame::Replay => assert!(replay < s.replays()),
                CounterFrame::Complete => {
                    assert_eq!(replay, s.replays());
                    return point;
                }
            }
        }
        unreachable!()
    }
}

/// Each pass's counted GPU time is its own and agrees with its exact timestamps, `alu` keeps
/// the SMs busy and DRAM idle and `stream` the reverse, in one frame with the default metrics,
/// twice in a row in one session, and across the replays a metric set needs. A replay whose
/// passes differ from its collection's first frame fails it, and a session dropped inside a
/// frame leaves the next session free to count.
#[test]
fn counters_land_on_their_pass_and_metric() {
    gpu_suite::log_to_stderr();
    let sdk = std::env::var_os("NVPERF_SDK_PATH").map(std::path::PathBuf::from);
    let d = Device::new(&DeviceDesc {
        name: "gpu_suite counters",
        perf_counters: Some(CounterDesc {
            sdk: sdk.as_deref(),
        }),
        ..DeviceDesc::default()
    })
    .expect("device");
    assert!(
        d.caps().perf_counters,
        "the device collects no counters; the log says why"
    );
    let bench = Bench::new(&d);
    let queue = bench.queue;
    let point = bench.frame(true);
    queue.wait(point, Duration::from_secs(10)).unwrap();
    let plain = d.pass_results(point).unwrap();
    assert!(
        plain.iter().all(|p| p.counters.is_none() && !p.exact),
        "{plain:?}"
    );

    let mut s = queue.start_counters(DEFAULT_COUNTER_METRICS).unwrap();
    assert_eq!(s.replays(), 1);
    let metrics = s.metrics().to_vec();
    assert!(metrics.iter().all(|m| m.available), "{metrics:?}");
    let names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, DEFAULT_COUNTER_METRICS);
    let at = |name: &str| names.iter().position(|n| *n == name).unwrap();
    assert_eq!(metrics[at(TIME)].unit, "nanosecond");
    assert_eq!(metrics[at(DRAM)].unit, "percent");
    for _ in 0..2 {
        let point = bench.count(&mut s);
        check(&d.pass_results(point).unwrap(), &at);
    }
    drop(s);

    let mut metrics = DEFAULT_COUNTER_METRICS.to_vec();
    metrics.push(L1_HITS);
    let mut s = queue.start_counters(&metrics).unwrap();
    assert!(s.replays() > 1, "{metrics:?} fit in one frame");
    s.begin_frame().unwrap();
    bench.frame(true);
    assert_eq!(s.end_frame().unwrap(), CounterFrame::Replay);
    s.begin_frame().unwrap();
    let point = bench.frame(false);
    assert!(s.end_frame().is_err());
    let missing = d.pass_results(point).unwrap();
    assert!(missing.iter().all(|p| p.counters.is_none()), "{missing:?}");
    let point = bench.count(&mut s);
    let passes = d.pass_results(point).unwrap();
    check(&passes, &|m| metrics.iter().position(|n| *n == m).unwrap());
    let l1 = metrics.len() - 1;
    for p in &passes[..3] {
        assert!(values(p)[l1].is_some(), "{p:?}");
    }

    s.begin_frame().unwrap();
    bench.frame(true);
    drop(s);
    let mut s = queue.start_counters(&[TIME]).unwrap();
    let point = bench.count(&mut s);
    assert!(
        d.pass_results(point)
            .unwrap()
            .iter()
            .all(|p| values(p)[0].is_some())
    );
    drop(s);
    assert_eq!(d.validation_messages(), 0);
}

fn values(p: &PassResult) -> Vec<Option<f64>> {
    match &p.counters {
        Some(PassCounters::Values(v)) => v.clone(),
        other => panic!("pass {}: {other:?}", p.name),
    }
}

/// The frame's passes, with `at` giving a metric's index.
fn check(passes: &[PassResult], at: &dyn Fn(&str) -> usize) {
    let [short, stream, long, empty] = passes else {
        panic!("{passes:?}");
    };
    let [short, stream, long, empty] = [short, stream, long, empty].map(values);
    let get = |v: &[Option<f64>], m| v[at(m)].unwrap();
    assert!(empty[at(TIME)].is_some(), "{empty:?}");
    assert!(passes.iter().all(|p| p.exact), "{passes:?}");
    let ratio = get(&long, TIME) / get(&short, TIME);
    assert!(
        (3.5..4.5).contains(&ratio),
        "long alu / short alu time {ratio}"
    );
    for (p, v) in passes.iter().zip([&short, &stream, &long]) {
        let counted = get(v, TIME) * 1e-6;
        assert!(
            (counted - p.gpu_ms).abs() < 0.05 * p.gpu_ms,
            "{}: counted {counted} ms, timestamps {} ms",
            p.name,
            p.gpu_ms
        );
    }
    let (active, issue, dram) = (get(&long, ACTIVE), get(&long, ISSUE), get(&long, DRAM));
    assert!(
        active > 70.0 && issue > 40.0 && dram < 1.0,
        "alu: active {active}%, issue {issue}%, DRAM {dram}%"
    );
    let (issue, dram) = (get(&stream, ISSUE), get(&stream, DRAM));
    assert!(
        issue < 15.0 && dram > 40.0,
        "stream: issue {issue}%, DRAM {dram}%"
    );
}
