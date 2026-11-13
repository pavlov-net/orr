//! Exactly timed passes read their own work's time, against CPU wall time, for compute, raster
//! and ray-dispatch work and whatever runs before them; passes not asked for stay unexact.

use std::time::{Duration, Instant};

use gpu_suite::shaders::{timing, timing_rays};
use gpu_suite::{assert_clean, device, render, require};
use orr_gpu::{
    Allocation, ColorAttachment, ColorLoad, ColorTarget, CommandList, ComputePipeline, Format,
    MemoryKind, MeshPipeline, MeshPipelineDesc, PassResult, RayDispatchPipeline, Rendering,
    ShaderStage, SpecData, Texture, TextureDesc, TextureMemory, TextureUsage,
};

/// Workgroups of 64 compute threads, and the raster and ray-dispatch extent's side, each about
/// as many threads as the GPU holds at once, so a pass has a tail of late workgroups.
const GROUPS: u32 = 4096;
const SIDE: u32 = 512;
/// Steps of the chain in a short pass.
const STEPS: u32 = 50_000;
/// How far the wall time of a submit may exceed its one pass's time: the time the submit call
/// and the GPU take to start it and the CPU takes to wake once it completes.
const WALL_MARGIN_MS: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Compute,
    Raster,
    Rays,
}

struct Bench {
    alu: ComputePipeline,
    raster: MeshPipeline,
    rays: Option<RayDispatchPipeline>,
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
            d.create_ray_dispatch_pipeline(&stage, "timing rays")
                .unwrap()
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
                list.dispatch_rays(self.rays.as_ref().unwrap(), &root, [SIDE, SIDE, 1]);
            }
        }
        list.end_pass();
    }

    /// Submits `list` to the idle queue: its passes, and the wall time from the submit to its
    /// completion.
    fn run(&mut self, list: CommandList<'_>) -> (Vec<PassResult>, f64) {
        let queue = render();
        let start = Instant::now();
        let point = queue.submit(vec![list], &[]).unwrap();
        queue.wait(point, Duration::from_secs(10)).unwrap();
        let wall = start.elapsed().as_secs_f64() * 1e3;
        (device().pass_results(point).unwrap(), wall)
    }

    /// The least time of `kind`'s exact pass of `steps` steps alone in a submit, after
    /// checking each against the submit's wall time.
    fn alone(&mut self, kind: Kind, steps: u32) -> f64 {
        let mut least = f64::MAX;
        let mut gap = f64::MAX;
        for _ in 0..5 {
            let mut list = render().command_list().unwrap();
            list.discard(&self.target);
            list.set_exact_timing(true);
            self.pass(&mut list, "timed", kind, steps);
            let (passes, wall) = self.run(list);
            let [p] = passes.as_slice() else {
                panic!("{passes:?}")
            };
            assert!(p.exact, "{p:?}");
            assert!(
                p.gpu_ms <= wall,
                "{kind:?}: {} ms in a {wall} ms submit",
                p.gpu_ms
            );
            least = least.min(p.gpu_ms);
            gap = gap.min(wall - p.gpu_ms);
        }
        assert!(
            gap < WALL_MARGIN_MS,
            "{kind:?} of {steps} steps: {least} ms, at least {gap} ms short of its wall time"
        );
        least
    }

    /// The least time of `kind`'s exact pass of `steps` steps between a long compute pass
    /// and another `kind` pass, both unexact.
    fn between(&mut self, kind: Kind, steps: u32) -> f64 {
        let mut least = f64::MAX;
        for _ in 0..3 {
            let mut list = render().command_list().unwrap();
            list.discard(&self.target);
            self.pass(&mut list, "before", Kind::Compute, 4 * STEPS);
            list.set_exact_timing(true);
            self.pass(&mut list, "timed", kind, steps);
            list.set_exact_timing(false);
            self.pass(&mut list, "after", kind, steps);
            let (passes, _) = self.run(list);
            let exact: Vec<(&str, bool)> = passes.iter().map(|p| (p.name, p.exact)).collect();
            assert_eq!(
                exact,
                [("before", false), ("timed", true), ("after", false)]
            );
            least = least.min(passes[1].gpu_ms);
        }
        least
    }
}

/// Four times the steps read four times the time, every pass agrees with the wall time of
/// its submit, and an exact pass reads the same whatever unexact work comes before and after
/// it.
#[test]
fn exact_passes_read_their_own_work() {
    let mut bench = Bench::new();
    // The GPU's clocks settle under load before the passes compared.
    bench.alone(Kind::Compute, 4 * STEPS);
    let short = bench.alone(Kind::Compute, STEPS);
    let long = bench.alone(Kind::Compute, 4 * STEPS);
    let ratio = long / short;
    assert!(
        (3.2..4.8).contains(&ratio),
        "{long} ms for 4 times the steps of {short} ms"
    );
    let mut kinds = vec![(Kind::Compute, STEPS), (Kind::Raster, STEPS)];
    if require(bench.rays.is_some(), "no ray-dispatch pipelines") {
        // A raygen thread takes the chain's steps faster than a compute or pixel thread.
        kinds.push((Kind::Rays, 4 * STEPS));
    }
    for (kind, steps) in kinds {
        let alone = bench.alone(kind, steps);
        let between = bench.between(kind, steps);
        assert!(
            (0.8..1.25).contains(&(between / alone)),
            "{kind:?}: {between} ms between unexact passes, {alone} ms alone"
        );
    }
    assert_clean();
}
