//! Averaging a still camera's frames in a scene-referred hook.

use orr::orr_gpu::{Allocation, Device, MemoryKind, SpecData};
use orr::{Frame, HookPass, PipelineId, Renderer};

use crate::shaders::accumulate;

/// Averages a still view's radiance over frames. The sum starts after the first `skip` lit frames
/// of the run, while exposure settles, and starts over when the extent changes. Frames that
/// render black while the path tracing kernels compile do not count, as the hook does not run in
/// them.
pub struct Average {
    gpu: &'static Device,
    pipeline: PipelineId,
    roots: Allocation<accumulate::Root>,
    slots: u64,
    sum: Option<(Allocation<[f32; 4]>, [u32; 2])>,
    frames: u32,
    skip: u64,
}

impl Average {
    pub fn new(
        renderer: &mut Renderer,
        gpu: &'static Device,
        frames_in_flight: u32,
        skip: u64,
    ) -> Self {
        let pipeline = renderer
            .compute_pipeline(&accumulate::MODULE, "main", SpecData::default())
            .expect("accumulate pipeline");
        let slots = u64::from(frames_in_flight);
        let roots = gpu
            .gpu_malloc(MemoryKind::Upload, slots, "accumulate roots")
            .expect("accumulate roots");
        Self {
            gpu,
            pipeline,
            roots,
            slots,
            sum: None,
            frames: 0,
            skip,
        }
    }

    /// Frames in the sum.
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// Starts the sum over after `skip` more lit frames, for a view that has changed.
    pub fn restart(&mut self, skip: u64) {
        self.frames = 0;
        self.skip = skip;
    }

    /// The scene-referred hook of frame `f` at `extent`. Call once per frame, whether the hook is
    /// used or not.
    pub fn hook<'s, 'a>(
        &'s mut self,
        f: &Frame<'a>,
        extent: [u32; 2],
    ) -> impl FnMut(&mut HookPass<'_, '_>) + use<'s, 'a> {
        if self.sum.as_ref().is_none_or(|(_, e)| *e != extent) {
            let texels = u64::from(extent[0]) * u64::from(extent[1]);
            let buffer = self
                .gpu
                .gpu_malloc(MemoryKind::DeviceLocal, texels, "radiance sum")
                .expect("radiance sum");
            self.sum = Some((buffer, extent));
            self.frames = 0;
        }
        let in_sum = self.frames;
        if f.paths_ready() {
            if self.skip > 0 {
                self.skip -= 1;
            } else {
                self.frames += 1;
            }
        }
        let sum = self.sum.as_ref().expect("radiance sum").0.gpu();
        let pipeline = f.pipeline(self.pipeline);
        let slot = f.id() % self.slots;
        let roots = &mut self.roots;
        move |pass: &mut HookPass<'_, '_>| {
            roots.as_mut_slice().expect("accumulate roots")[slot as usize] = accumulate::Root {
                view: pass.view,
                sum,
                image: pass.image,
                width: pass.extent[0],
                height: pass.extent[1],
                frames: in_sum,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            pass.cmd.begin_pass("accumulate", false);
            pass.cmd.dispatch(
                pipeline,
                roots.gpu().offset(slot),
                [pass.extent[0].div_ceil(8), pass.extent[1].div_ceil(8), 1],
            );
            pass.cmd.end_pass();
        }
    }
}
