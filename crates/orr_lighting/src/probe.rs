//! The lighting side of a pixel probe: the first path vertex at one G-buffer pixel and the
//! origin its rays leave from, read back once the frame completes.

use std::fmt;

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, KernelSet, MemoryKind,
    Stages,
};
use orr_scene::{GBufferGpu, SceneRoot, ViewConstants};

use crate::Error;
use crate::shaders::probe::{MODULE, OriginProbeGpu, OriginProbeRoot};

/// The first vertex of a probed pixel's paths.
#[derive(Clone, Copy, Debug)]
pub struct OriginProbe {
    pub frame: u64,
    pub gpu: OriginProbeGpu,
}

/// A view's lighting probes: recorded into frames, read once each frame completes.
#[derive(Default)]
pub struct OriginProbes {
    pipeline: Option<ComputePipeline>,
    pending: Vec<(u64, Allocation<OriginProbeRoot>, Allocation<OriginProbeGpu>)>,
}

impl OriginProbes {
    /// Records the probe of render pixel `pixel` of the G-buffer `gbuffer` into `list`, while
    /// the G-buffer holds the frame's resolve.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &Device,
        kernels: &KernelSet,
        list: &mut CommandList<'_>,
        frame: u64,
        scene: GpuPtr<SceneRoot>,
        view: GpuPtr<ViewConstants>,
        gbuffer: GpuPtr<GBufferGpu>,
        pixel: [u32; 2],
    ) -> Result<(), Error> {
        if self.pipeline.is_none() {
            self.pipeline = Some(crate::pipeline(device, kernels, &MODULE, "probe")?);
        }
        let out = device.gpu_malloc::<OriginProbeGpu>(MemoryKind::Readback, 1, "origin probe")?;
        let mut root =
            device.gpu_malloc::<OriginProbeRoot>(MemoryKind::Upload, 1, "origin probe root")?;
        root.as_mut_slice().expect("upload memory")[0] = OriginProbeRoot {
            scene,
            view,
            gbuffer,
            out: out.gpu(),
            pixel_x: pixel[0],
            pixel_y: pixel[1],
            _pad: 0,
        };
        list.begin_pass("origin probe", false);
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.dispatch(
            self.pipeline.as_ref().expect("made above"),
            root.gpu(),
            [1, 1, 1],
        );
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        list.end_pass();
        self.pending.push((frame, root, out));
        Ok(())
    }

    /// Takes the probes of frames at or before `completed`, a frame whose work has completed.
    pub fn take(&mut self, completed: u64) -> Vec<OriginProbe> {
        let (done, pending): (Vec<_>, _) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|(frame, _, _)| *frame <= completed);
        self.pending = pending;
        done.into_iter()
            .map(|(frame, _, out)| OriginProbe {
                frame,
                gpu: out.as_slice().expect("readback memory")[0],
            })
            .collect()
    }
}

impl fmt::Display for OriginProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = &self.gpu;
        if g.hit == 0 {
            return writeln!(f, "lighting: no first vertex (sky)");
        }
        let policy = ["default", "surface", "triangle", "?"][g.policy.min(3) as usize];
        writeln!(
            f,
            "lighting: first vertex {:.3} m from the eye; origin error max(raster {:.6}, traced {:.6}) = {:.6} m, policy {policy}, floor {:.6} m: its rays leave {:.6} m off the surface",
            g.camera_distance, g.raster_error, g.traced_error, g.origin_error, g.floor, g.offset
        )
    }
}
