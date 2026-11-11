//! Recording pixel probes into a frame and reading them back.

use orr_gpu::{Hazards, MemoryKind, ShaderStage, SpecData, Stages};

use super::{Core, Frame, PoolBytes, Scenes};
use crate::probe::{self, PendingProbe, PixelProbe, ProbeContext, ProbeSettings, ProbeView};
use crate::shaders::probe::{MODULE, ProbeGpu, ProbeRoot};
use crate::view::{RasterView, ViewHistoryId};
use crate::world::WorldPos;
use crate::{DetailStats, Error, Result};

/// Workgroups per row of a workgroup grid (`GRID_X` in `visibility.slang`).
const GRID_X: u32 = 65535;

impl Core {
    /// The probe's two pipelines, made at the first probe.
    fn probe_pipelines(
        &mut self,
        device: &orr_gpu::Device,
    ) -> Result<&[orr_gpu::ComputePipeline; 2]> {
        if self.probe.is_none() {
            let make = |entry: &str| -> Result<orr_gpu::ComputePipeline> {
                let stage = ShaderStage::new(&MODULE, entry, SpecData::default())?;
                Ok(device.create_compute_pipeline(&stage, MODULE.name)?)
            };
            self.probe = Some([make("probe")?, make("lists")?]);
        }
        Ok(self.probe.as_ref().expect("made above"))
    }
}

impl Frame<'_> {
    /// Records a probe of render pixel `pixel` of `view`, a view this frame's
    /// [`Frame::raster_view`] rendered, before its render enters a step past
    /// [`crate::ViewStep::Transparency`], which may take the memory of the visibility buffer and
    /// the G-buffer. [`Scenes::probes`] returns it once the frame has completed.
    pub fn probe(&mut self, view: &RasterView, pixel: [u32; 2]) -> Result<()> {
        self.ensure_updated()?;
        let device = self.device;
        let frame = self.id;
        let core = &mut *self.core;
        let desc = core
            .histories
            .get(view.history.0 as usize)
            .and_then(Option::as_ref)
            .and_then(|h| h.previous)
            .ok_or(Error::UnknownViewHistory(view.history))?;
        let slot = (frame % u64::from(core.fif)) as usize;
        let (focus, rt_scale) = {
            let cpu = core.scene(desc.scene)?.cpu.lock().unwrap();
            let focus = cpu.focus.map(|f| (f, cpu.pixels_per_radian));
            (focus, cpu.cuts.as_ref().map_or(1.0, |c| c.rt_scale(slot)))
        };
        let memory_scale = core.memory.scale();
        let (focus_cell, focus_offset) = focus.map_or(([0; 4], [0.0; 4]), |(f, ppr)| {
            let w = WorldPos::new(f, core.cell_log2);
            (w.cell.extend(1).to_array(), w.offset.extend(ppr).to_array())
        });
        let pools = PoolBytes::new(device, &core.budgets);
        let context = ProbeContext {
            frame,
            pixel,
            view: ProbeView {
                position: desc.position,
                rotation: desc.rotation,
                fov_y: desc.projection.fov_y,
                render_extent: desc.render_extent.to_array(),
                output_extent: desc.output_extent.to_array(),
                jitter: desc.jitter.to_array(),
                mip_bias: desc.mip_bias,
            },
            detail: DetailStats {
                memory_scale,
                rt_scale,
                ..core.stats.detail
            },
            residency: *core.mounts.streaming.stats(),
            settings: ProbeSettings {
                raster_error_pixels: core.raster_error_pixels,
                rt_error_pixels: core.rt_error_pixels,
                culled_error_scale: core.culled_error_scale,
                rt_merge_pixels: core.rt_merge_pixels,
                software_raster_pixels: core.software_raster_pixels,
                mesh_pool: pools.mesh,
                texture_pool: pools.textures,
            },
            focus,
        };
        let readback = device.gpu_malloc::<ProbeGpu>(MemoryKind::Readback, 1, "pixel probe")?;
        let root = core.windows[self.window].put_root(
            device,
            &ProbeRoot {
                scene: view.scene,
                view: view.view,
                raster: view.raster,
                out: readback.gpu(),
                focus_cell,
                focus_offset,
                pixel_x: pixel[0],
                pixel_y: pixel[1],
                raster_pixels: core.raster_error_pixels * memory_scale,
                rt_pixels: core.rt_error_pixels * memory_scale,
                culled_scale: core.culled_error_scale,
                gather_scale: rt_scale,
                merge_pixels: core.rt_merge_pixels,
                _pad: 0,
            },
        )?;
        let entries = core
            .histories
            .get(view.history.0 as usize)
            .and_then(Option::as_ref)
            .and_then(|h| h.lists.as_ref())
            .map_or(0, |l| l.sizes.visible_clusters);
        let [probe, lists] = core.probe_pipelines(device)?;
        let list = self.list.as_mut().expect("frame list");
        list.begin_pass("pixel probe", false);
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.dispatch(probe, root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        let groups = entries.div_ceil(64).max(1);
        list.dispatch(
            lists,
            root,
            [groups.min(GRID_X), groups.div_ceil(GRID_X), 1],
        );
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        list.end_pass();
        core.mounts.streaming.note_outcomes(true);
        let history = core.histories[view.history.0 as usize]
            .as_mut()
            .expect("found above");
        history.probes.push(PendingProbe { readback, context });
        Ok(())
    }
}

impl Scenes {
    /// Takes the probes of view `history` of frames at or before `completed`, a frame whose
    /// work has completed ([`crate::FrameStats::frame`]), oldest first.
    pub fn probes(&mut self, history: ViewHistoryId, completed: u64) -> Vec<PixelProbe> {
        let core = &mut self.core;
        let Some(h) = core
            .histories
            .get_mut(history.0 as usize)
            .and_then(Option::as_mut)
        else {
            return Vec::new();
        };
        let (done, pending): (Vec<PendingProbe>, _) = std::mem::take(&mut h.probes)
            .into_iter()
            .partition(|p| p.context.frame <= completed);
        h.probes = pending;
        let read_at = core.next_frame - 1;
        let out = done
            .iter()
            .map(|p| probe::resolve(p, &core.mounts, read_at))
            .collect();
        let waiting = core
            .histories
            .iter()
            .flatten()
            .any(|h| !h.probes.is_empty());
        core.mounts.streaming.note_outcomes(waiting);
        out
    }
}
