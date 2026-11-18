//! Exactly timed passes read their own work's time, against CPU wall time, for compute, raster
//! and ray tracing work and whatever runs before them; passes not asked for stay unexact.

use std::time::{Duration, Instant};

use gpu_suite::shaders::{timing, timing_rays};
use gpu_suite::{assert_clean, device, raygen_pipeline, render, require};
use orr_gpu::{
    Allocation, ColorAttachment, ColorLoad, ColorTarget, CommandList, ComputePipeline, Format,
    HitTable, MemoryKind, MeshPipeline, MeshPipelineDesc, PassResult, RayPipeline, Rendering,
    ShaderStage, SpecData, Texture, TextureDesc, TextureMemory, TextureUsage,
};

/// Workgroups of 64 compute threads, and the raster and ray tracing extent's side, each about
/// as many threads as the GPU holds at once, so a pass has a tail of late workgroups.
const GROUPS: u32 = 4096;
const SIDE: u32 = 512;
/// Steps of the chain in a short pass. A pass's time takes in whatever work of other processes
/// the GPU runs while it does, so the shorter the pass, the more of its readings miss that work.
const STEPS: u32 = 4_000;
/// Submits each comparison takes its least readings from, so that they miss other processes'
/// work on a shared GPU.
const SAMPLES: usize = 32;
/// Steps of the chain in a pass checked against the CPU's clock, in multiples of a short
/// pass's: tens of milliseconds of work, so an end timestamp written early cuts off
/// milliseconds.
const WALL_STEPS: u32 = 500;
/// Submits of each pass checked against the CPU's clock.
const WALL_SAMPLES: usize = 8;
/// The most the wall time of a submit may be over its one exact pass's time. The submit call,
/// the GPU's start and the CPU's wake take under a millisecond, but other processes' work the
/// GPU runs ahead of the pass can take tens, so only a pass read short by half fails, as a
/// wrong unit or timestamp period reads it.
const WALL_OVER_PASS: f64 = 2.0;
/// How long after its end timestamp the CPU may wake to the completion of a submit: the time
/// the GPU takes to signal it and the CPU to wake, a tenth of a millisecond on an idle CPU
/// and more on a busy one, but well short of the tail an early end timestamp cuts off.
const WAKE_MARGIN_MS: f64 = 5.0;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Compute,
    Raster,
    Rays,
}

struct Bench {
    alu: ComputePipeline,
    raster: MeshPipeline,
    rays: Option<RayPipeline>,
    target: Texture,
    out: Allocation<f32>,
    roots: Vec<(u32, Allocation<timing::Root>)>,
}

impl Bench {
    fn new() -> Self {
        let d = device();
        let stage = |e| ShaderStage::new(&timing::MODULE, e, SpecData::default()).unwrap();
        let rays = d.caps().ray_dispatch.then(|| {
            let stage =
                ShaderStage::new(&timing_rays::MODULE, "alu_rays", SpecData::default()).unwrap();
            raygen_pipeline(d, &stage, "timing rays").unwrap()
        });
        Bench {
            alu: d
                .create_compute_pipeline(&stage("alu"), "timing alu")
                .unwrap(),
            raster: d
                .create_mesh_pipeline(
                    &MeshPipelineDesc {
                        pixel: Some(stage("alu_pixel")),
                        colors: &[ColorTarget::new(Format::R32Float)],
                        ..MeshPipelineDesc::new(stage("cover"))
                    },
                    "timing raster",
                )
                .unwrap(),
            rays,
            target: d
                .create_texture(
                    &TextureDesc::d2(Format::R32Float, SIDE, SIDE, TextureUsage::COLOR),
                    TextureMemory::Dedicated,
                    "timing target",
                )
                .unwrap(),
            out: d
                .gpu_malloc(MemoryKind::DeviceLocal, 1, "timing out")
                .unwrap(),
            roots: Vec::new(),
        }
    }

    fn root(&mut self, steps: u32) -> orr_gpu::GpuPtr<timing::Root> {
        if let Some((_, r)) = self.roots.iter().find(|(s, _)| *s == steps) {
            return r.gpu();
        }
        let mut r = device()
            .gpu_malloc::<timing::Root>(MemoryKind::Upload, 1, "timing root")
            .unwrap();
        r.as_mut_slice().unwrap()[0] = timing::Root {
            out: self.out.gpu(),
            iterations: steps,
            pad: 0,
        };
        let ptr = r.gpu();
        self.roots.push((steps, r));
        ptr
    }

    /// Records `kind`'s work of `steps` steps a thread as the pass `name`.
    fn pass(&mut self, list: &mut CommandList<'_>, name: &'static str, kind: Kind, steps: u32) {
        let root = self.root(steps);
        list.begin_pass(name, false);
        match kind {
            Kind::Compute => list.dispatch(&self.alu, root, [GROUPS, 1, 1]),
            Kind::Raster => {
                list.begin_rendering(&Rendering {
                    extent: [SIDE; 2],
                    colors: &[ColorAttachment {
                        texture: &self.target,
                        load: ColorLoad::Discard,
                        store: true,
                    }],
                    depth: None,
                });
                list.draw_mesh_tasks(&self.raster, root, [1, 1, 1]);
                list.end_rendering();
            }
            Kind::Rays => {
                let root = timing_rays::Root {
                    out: self.out.gpu(),
                    iterations: steps,
                    pad: 0,
                };
                let rays = self.rays.as_ref().unwrap();
                list.trace_rays(rays, 0, &HitTable::NONE, &root, [SIDE, SIDE, 1]);
            }
        }
        list.end_pass();
    }

    /// Submits `list` to the idle queue: its passes, and the instants of the submit and of the
    /// CPU's wake to its completion.
    fn run(&mut self, list: CommandList<'_>) -> (Vec<PassResult>, Instant, Instant) {
        let queue = render();
        let start = Instant::now();
        let point = queue.submit(vec![list], &[]).unwrap();
        queue.wait(point, Duration::from_secs(10)).unwrap();
        let woke = Instant::now();
        (device().pass_results(point).unwrap(), start, woke)
    }

    /// The least time of each of `kind`'s exact passes of `steps` steps, all in every submit,
    /// after checking that each submit's passes fit its wall time.
    fn alone<const N: usize>(&mut self, kind: Kind, steps: [u32; N]) -> [f64; N] {
        let mut least = [f64::MAX; N];
        for _ in 0..SAMPLES {
            let mut list = render().command_list().unwrap();
            list.discard(&self.target);
            list.set_exact_timing(true);
            for s in steps {
                self.pass(&mut list, "timed", kind, s);
            }
            let (passes, start, woke) = self.run(list);
            let wall = ms(woke - start);
            assert_eq!(passes.len(), N, "{passes:?}");
            assert!(passes.iter().all(|p| p.exact), "{passes:?}");
            let sum: f64 = passes.iter().map(|p| p.gpu_ms).sum();
            assert!(sum <= wall, "{kind:?}: {passes:?} in a {wall} ms submit");
            for (l, p) in least.iter_mut().zip(&passes) {
                *l = l.min(p.gpu_ms);
            }
        }
        least
    }

    /// `kind`'s exact pass of `steps` steps alone in a submit, against the CPU's clock: the
    /// least ratio of the submit's wall time to the pass's time, and with calibrated
    /// timestamps the least time from the pass's end to the CPU's wake.
    fn against_wall(&mut self, kind: Kind, steps: u32) -> (f64, f64) {
        let (mut over, mut wake) = (f64::MAX, f64::MAX);
        for _ in 0..WALL_SAMPLES {
            let mut list = render().command_list().unwrap();
            list.discard(&self.target);
            list.set_exact_timing(true);
            self.pass(&mut list, "timed", kind, steps);
            let (passes, start, woke) = self.run(list);
            let wall = ms(woke - start);
            let [p] = passes.as_slice() else {
                panic!("{passes:?}")
            };
            assert!(p.exact && p.gpu_ms <= wall, "{p:?} in a {wall} ms submit");
            over = over.min(wall / p.gpu_ms);
            if let Some((_, end)) = p.cpu {
                // The calibration can place the end a little past the wake.
                wake = wake.min(ms(woke.saturating_duration_since(end)));
            }
        }
        (over, wake)
    }

    /// The least times of `kind`'s exact pass of `steps` steps alone, and between a long compute
    /// pass and another `kind` pass, both unexact, read in the same submits.
    fn between(&mut self, kind: Kind, steps: u32) -> (f64, f64) {
        let (mut alone, mut between) = (f64::MAX, f64::MAX);
        for _ in 0..SAMPLES {
            let mut list = render().command_list().unwrap();
            list.discard(&self.target);
            list.set_exact_timing(true);
            self.pass(&mut list, "alone", kind, steps);
            list.set_exact_timing(false);
            self.pass(&mut list, "before", Kind::Compute, 4 * STEPS);
            list.set_exact_timing(true);
            self.pass(&mut list, "timed", kind, steps);
            list.set_exact_timing(false);
            self.pass(&mut list, "after", kind, steps);
            let (passes, ..) = self.run(list);
            let exact: Vec<(&str, bool)> = passes.iter().map(|p| (p.name, p.exact)).collect();
            assert_eq!(
                exact,
                [
                    ("alone", true),
                    ("before", false),
                    ("timed", true),
                    ("after", false)
                ]
            );
            alone = alone.min(passes[0].gpu_ms);
            between = between.min(passes[2].gpu_ms);
        }
        (alone, between)
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Every exact pass agrees with the CPU's clock, four times the steps read four times the
/// time, and an exact pass reads the same whatever unexact work comes before and after it, for
/// every kind of work.
#[test]
fn exact_passes_read_their_own_work() {
    let mut bench = Bench::new();
    let calibrated = require(
        device().caps().calibrated_timestamps,
        "no calibrated timestamps to place passes' ends",
    );
    let mut kinds = vec![(Kind::Compute, STEPS), (Kind::Raster, STEPS)];
    if require(bench.rays.is_some(), "no ray tracing pipelines") {
        // A raygen thread takes the chain's steps faster than a compute or pixel thread.
        kinds.push((Kind::Rays, 4 * STEPS));
    }
    for (kind, steps) in kinds {
        // The long passes also let the GPU's clocks settle under load before the passes
        // compared.
        let (over, wake) = bench.against_wall(kind, WALL_STEPS * steps);
        assert!(
            over < WALL_OVER_PASS,
            "{kind:?}: submits took at least {over} times their exact pass's time"
        );
        if calibrated {
            assert!(
                wake < WAKE_MARGIN_MS,
                "{kind:?}: the CPU woke at least {wake} ms after an exact pass's end"
            );
        }
        let [short, long] = bench.alone(kind, [steps, 4 * steps]);
        assert!(
            (3.2..4.8).contains(&(long / short)),
            "{kind:?}: {long} ms for 4 times the steps of {short} ms"
        );
        let (alone, between) = bench.between(kind, steps);
        assert!(
            (0.8..1.25).contains(&(between / alone)),
            "{kind:?}: {between} ms between unexact passes, {alone} ms alone"
        );
    }
    assert_clean();
}
