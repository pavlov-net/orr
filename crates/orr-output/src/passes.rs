//! Output's compute passes, recorded into a command list: `tonemap` and `compose`.

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, MemoryKind, OutputFormat, ShaderStage,
    SpecData, StorageImageIndex, TextureIndex,
};

use crate::shaders::{compose, tonemap};
use crate::{Gt7Curve, Gt7Params, ResolvedDisplay, Result};

/// Output's compute pipelines on one device.
pub struct OutputKernels {
    tonemap: ComputePipeline,
    compose: ComputePipeline,
}

/// Upload memory for the roots of one view's output passes in one frame. Recording a pass
/// writes its root here, so reuse it only once the list that last recorded with it has
/// completed: keep one per view and frame in flight.
pub struct OutputRoots {
    tonemap: Allocation<tonemap::Root>,
    curve: Allocation<Gt7Curve>,
    compose: Allocation<compose::Root>,
}

impl OutputRoots {
    pub fn new(gpu: &Device) -> Result<Self> {
        Ok(Self {
            tonemap: gpu.gpu_malloc(MemoryKind::Upload, 1, "tonemap root")?,
            curve: gpu.gpu_malloc(MemoryKind::Upload, 1, "tonemap curve")?,
            compose: gpu.gpu_malloc(MemoryKind::Upload, 1, "compose root")?,
        })
    }
}

/// Tonemapping one image: pre-exposed linear Rec.2020 radiance in, display-linear Rec.709
/// (1.0 = paper white) out.
#[derive(Clone, Copy, Debug)]
pub struct TonemapPass<'a> {
    /// Radiance, read at each output texel.
    pub source: TextureIndex,
    /// An RGBA16F (or wider) texture of `extent`.
    pub target: StorageImageIndex,
    pub extent: [u32; 2],
    /// Scales source values to exposed values: the exposure (`exposure_from_ev100` for
    /// manual exposure) over the pre-exposure the source was lit with.
    pub exposure: f32,
    pub gt7: &'a Gt7Params,
    pub display: &'a ResolvedDisplay,
}

/// Encoding display-linear Rec.709 for the display.
#[derive(Clone, Copy, Debug)]
pub struct ComposePass<'a> {
    /// The tonemapped image.
    pub source: TextureIndex,
    /// The backbuffer, in the format of `display.encoding`.
    pub target: StorageImageIndex,
    pub extent: [u32; 2],
    pub display: &'a ResolvedDisplay,
}

impl OutputKernels {
    pub fn new(gpu: &Device) -> Result<Self> {
        let pipeline = |module, name| -> Result<ComputePipeline> {
            let stage = ShaderStage::new(module, "main", SpecData::default())?;
            Ok(gpu.create_compute_pipeline(&stage, name)?)
        };
        Ok(Self {
            tonemap: pipeline(&tonemap::MODULE, "tonemap")?,
            compose: pipeline(&compose::MODULE, "compose")?,
        })
    }

    /// Records the tonemap pass. Order its source's writes before it with a barrier.
    pub fn tonemap(
        &self,
        list: &mut CommandList<'_>,
        roots: &mut OutputRoots,
        pass: &TonemapPass<'_>,
    ) {
        let d = pass.display;
        roots.curve.as_mut_slice().expect("upload memory")[0] =
            Gt7Curve::new(pass.gt7, d.peak_nits, d.paper_white_nits);
        roots.tonemap.as_mut_slice().expect("upload memory")[0] = tonemap::Root {
            curve: roots.curve.gpu,
            source: pass.source,
            target: pass.target,
            width: pass.extent[0],
            height: pass.extent[1],
            exposure: pass.exposure,
            sdr: u32::from(d.encoding == OutputFormat::Sdr),
        };
        list.begin_pass("tonemap", false);
        list.dispatch(&self.tonemap, roots.tonemap.gpu, groups(pass.extent));
        list.end_pass();
    }

    /// Records the compose pass. Order its source's writes before it with a barrier.
    pub fn compose(
        &self,
        list: &mut CommandList<'_>,
        roots: &mut OutputRoots,
        pass: &ComposePass<'_>,
    ) {
        // ENCODING_* in display.slang.
        let encoding = match pass.display.encoding {
            OutputFormat::Sdr => 0,
            OutputFormat::Hdr10Pq => 1,
            OutputFormat::ScRgb => 2,
        };
        roots.compose.as_mut_slice().expect("upload memory")[0] = compose::Root {
            source: pass.source,
            target: pass.target,
            width: pass.extent[0],
            height: pass.extent[1],
            encoding,
            paper_white_nits: pass.display.paper_white_nits,
            _pad0: 0,
            _pad1: 0,
        };
        list.begin_pass("compose", false);
        list.dispatch(&self.compose, roots.compose.gpu, groups(pass.extent));
        list.end_pass();
    }
}

/// 8x8 workgroups covering `extent`.
fn groups(extent: [u32; 2]) -> [u32; 3] {
    [extent[0].div_ceil(8), extent[1].div_ceil(8), 1]
}
