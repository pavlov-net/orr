//! Pixel probes of lit views: the scene's probe of the render pixel under an output pixel, and
//! the lighting's first vertex there.

use std::fmt;

use orr_lighting::OriginProbe;
use orr_output::UpscaleMode;
use orr_scene::orr_gpu::{Device, KernelSet};
use orr_scene::{RasterView, ViewHistoryId};

use crate::{Result, ViewState};

/// One probed pixel of a view's render ([`crate::Frame::probe`]).
#[derive(Clone, Debug)]
pub struct PixelProbe {
    /// The output pixel asked for, and the output extent.
    pub output_pixel: [u32; 2],
    pub output_extent: [u32; 2],
    /// Ray reconstruction ran, and the named mode it ran in.
    pub reconstructed: bool,
    pub upscale_mode: Option<UpscaleMode>,
    /// What the raster drew at the render pixel under it and why.
    pub scene: orr_scene::PixelProbe,
    /// The first vertex of its paths; `None` for a view only rasterized.
    pub lighting: Option<OriginProbe>,
}

/// A lit view's probe recorded into a frame.
pub(crate) struct ProbeAt {
    pub(crate) frame: u64,
    pub(crate) output: [u32; 2],
    pub(crate) output_extent: [u32; 2],
    pub(crate) upscale_mode: Option<UpscaleMode>,
    pub(crate) reconstructed: bool,
}

/// The output pixel `probes` asks of `history`'s render, taken from it.
pub(crate) fn take_probe(
    probes: &mut Vec<(ViewHistoryId, [u32; 2])>,
    history: ViewHistoryId,
) -> Option<[u32; 2]> {
    let i = probes.iter().position(|(h, _)| *h == history)?;
    Some(probes.swap_remove(i).1)
}

/// The render pixel whose center lies under output pixel `pixel`'s.
fn render_pixel(pixel: [u32; 2], output: [u32; 2], render: [u32; 2]) -> [u32; 2] {
    [0, 1].map(|a| {
        let x = (f64::from(pixel[a]) + 0.5) * f64::from(render[a]) / f64::from(output[a].max(1));
        (x as u32).min(render[a].saturating_sub(1))
    })
}

/// Records the scene's and the lighting's probes of `probe`'s pixel of `raster`.
pub(crate) fn record_probe(
    scenes: &mut orr_scene::Frame<'_>,
    kernels: &KernelSet,
    gpu: &Device,
    state: &mut ViewState,
    raster: &RasterView,
    probe: ProbeAt,
) -> Result<()> {
    let render = [raster.targets.extent[0], raster.targets.extent[1]];
    let pixel = render_pixel(probe.output, probe.output_extent, render);
    scenes.probe(raster, pixel)?;
    let list = scenes.list()?;
    state.origin_probes.record(
        gpu,
        kernels,
        list,
        probe.frame,
        raster.scene,
        raster.view,
        raster.gbuffer,
        pixel,
    )?;
    state.probes.push(probe);
    Ok(())
}

impl fmt::Display for PixelProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "probe of output pixel ({}, {}) of {} x {}; {}",
            self.output_pixel[0],
            self.output_pixel[1],
            self.output_extent[0],
            self.output_extent[1],
            match (self.reconstructed, self.upscale_mode) {
                (false, _) => "no ray reconstruction".to_owned(),
                (true, Some(m)) => format!("ray reconstruction, upscale mode {m:?}"),
                (true, None) => "ray reconstruction at an explicit render extent".to_owned(),
            }
        )?;
        write!(f, "{}", self.scene)?;
        match &self.lighting {
            Some(l) => write!(f, "{l}"),
            None => writeln!(f, "lighting: the view is only rasterized"),
        }
    }
}
