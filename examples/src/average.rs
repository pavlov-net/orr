//! Averaging a still camera's frames in a scene-referred hook.

use orr::orr_gpu::{Allocation, Device, MemoryKind, SpecData};
use orr::{Frame, HookPass, PipelineId, Renderer};

use crate::shaders::accumulate;

/// Averages a still view's radiance over frames. The sum starts after the first `skip` frames
/// of the run, while exposure settles, and starts over when the extent changes.
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

    /// The scene-referred hook of the run's frame `i`, frame `f`, at `extent`. Call once per
    /// frame, whether the hook is used or not.
    pub fn hook<'s, 'a>(
        &'s mut self,
        f: &Frame<'a>,
        i: u64,
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
        let in_sum = if i < self.skip { 0 } else { self.frames };
        if i >= self.skip {
            self.frames += 1;
        }
        let sum = self.sum.as_ref().expect("radiance sum").0.gpu;
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
                roots.gpu.offset(slot),
                [pass.extent[0].div_ceil(8), pass.extent[1].div_ceil(8), 1],
            );
            pass.cmd.end_pass();
        }
    }
}
