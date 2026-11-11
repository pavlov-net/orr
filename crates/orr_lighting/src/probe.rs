//! The lighting side of a pixel probe: the first path vertex at one G-buffer pixel and the
//! origin its rays leave from, read back once the frame completes.

use std::fmt;

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, KernelSet, MemoryKind,
    Stages, StorageImageIndex,
};
use orr_scene::{GBufferGpu, SceneRoot, ViewConstants};

use crate::Error;
use crate::TracedKind;
use crate::scene::{FRAME_SLOTS, Ring};
use crate::shaders::probe::{MODULE, OriginProbeGpu, OriginProbeRoot};
use crate::shaders::traced::{self, TracedRoot};
use crate::shaders::types::TracedHit;

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

/// `TRACED_NONE` in `traced_debug.slang`.
const NONE: u32 = u32::MAX;
/// `RT_CUT_BUILT` in `cut.slang`.
const RT_CUT_BUILT: u32 = 255;
/// `CLASS_*` in `orr_shaders.trace`.
const CLASSES: [&str; 6] = [
    "opaque",
    "dielectric",
    "presence",
    "particle",
    "medium",
    "light",
];

impl fmt::Display for TracedHit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.hit == 0 {
            return write!(f, "nothing: it leaves the scene");
        }
        let class = CLASSES.get(self.class_id as usize).unwrap_or(&"?");
        write!(
            f,
            "{class} at {:.4} m, instance {}",
            self.t, self.instance_index
        )?;
        if self.copy == NONE {
            return write!(
                f,
                " (a producer entry or a proxy), cluster id {}",
                self.cluster_id
            );
        }
        let cut = match self.rt_lod {
            RT_CUT_BUILT => "a BLAS built that frame".to_owned(),
            k => format!("cached cut {k}"),
        };
        let n = self.normal;
        write!(
            f,
            ": copy {}{}, model {}, mesh {}, mesh cluster {} (cluster id {}), triangle {}; its instance holds {cut}; geometric normal ({:.3}, {:.3}, {:.3})",
            self.copy & 0x7fff_ffff,
            if self.copy & 0x8000_0000 != 0 {
                " (far step)"
            } else {
                ""
            },
            self.model,
            self.mesh,
            self.mesh_cluster,
            self.cluster_id,
            self.primitive,
            n[0],
            n[1],
            n[2]
        )
    }
}

impl fmt::Display for OriginProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = &self.gpu;
        let o = g.camera_origin;
        let d = g.camera_direction;
        writeln!(
            f,
            "traced camera ray from render ({:.3}, {:.3}, {:.3}) along ({:.4}, {:.4}, {:.4}) meets {}",
            o[0], o[1], o[2], d[0], d[1], d[2], g.traced
        )?;
        if g.hit == 0 {
            return writeln!(f, "lighting: no first vertex (sky)");
        }
        let policy = ["default", "surface", "triangle", "?"][g.policy.min(3) as usize];
        writeln!(
            f,
            "lighting: first vertex {:.3} m from the eye; origin error max(raster {:.6}, traced {:.6}) = {:.6} m, policy {policy}, floor {:.6} m: its rays leave {:.6} m off the surface",
            g.camera_distance, g.raster_error, g.traced_error, g.origin_error, g.floor, g.offset
        )?;
        if g.sun == 0 {
            return writeln!(f, "sun: the scene has no directional light");
        }
        let s = g.sun_direction;
        let n = g.geometric_normal;
        writeln!(
            f,
            "sun ray along ({:.4}, {:.4}, {:.4}), cosine {:.4} with the geometric normal, from {:.6} m off the surface, meets {}",
            s[0],
            s[1],
            s[2],
            s[0] * n[0] + s[1] * n[1] + s[2] * n[2],
            g.sun_offset,
            g.blocker
        )
    }
}

/// A view's [`crate::LightingDebugView::Traced`] pass: its pipeline and its roots, one per
/// frame slot, made at its first frame.
#[derive(Default)]
pub struct TracedViews {
    pass: Option<(ComputePipeline, Ring<TracedRoot>)>,
}

impl TracedViews {
    /// Records the traced view `kind` of the G-buffer `gbuffer` over `radiance`, a storage
    /// image of `extent`, into `list` for frame `frame`, after the paths wrote it.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &Device,
        kernels: &KernelSet,
        list: &mut CommandList<'_>,
        frame: u64,
        kind: TracedKind,
        scene: GpuPtr<SceneRoot>,
        view: GpuPtr<ViewConstants>,
        gbuffer: GpuPtr<GBufferGpu>,
        radiance: StorageImageIndex,
        extent: [u32; 2],
    ) -> Result<(), Error> {
        if self.pass.is_none() {
            self.pass = Some((
                crate::pipeline(device, kernels, &traced::MODULE, "traced")?,
                Ring::new(device, "traced view roots")?,
            ));
        }
        let (pipeline, roots) = self.pass.as_mut().expect("made above");
        // `TRACED_*` in `traced.slang`.
        let mode = match kind {
            TracedKind::Distance => 0,
            TracedKind::Normal => 1,
            TracedKind::Disagreement => 2,
            TracedKind::Sun => 3,
        };
        let root = roots.write(
            (frame % FRAME_SLOTS as u64) as usize,
            TracedRoot {
                scene,
                view,
                gbuffer,
                radiance,
                mode,
                width: extent[0],
                height: extent[1],
                _pad: 0,
            },
        );
        list.begin_pass("traced view", false);
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.dispatch(
            pipeline,
            root,
            [extent[0].div_ceil(8), extent[1].div_ceil(8), 1],
        );
        list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        list.end_pass();
        Ok(())
    }
}
