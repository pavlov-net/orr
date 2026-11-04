use super::command::CommandList;
use super::device::Device;
use super::not_implemented;
use super::pipeline::ShaderStage;
use super::texture::Texture;
use crate::{ColorLoad, ColorTarget, DepthLoad, DepthTarget, Error, RasterState};

/// How to build a mesh pipeline. Every stage reads the root pointer passed to the draw.
#[derive(Clone, Copy, Debug)]
pub struct MeshPipelineDesc<'a> {
    pub task: Option<ShaderStage<'a>>,
    pub mesh: ShaderStage<'a>,
    /// `None`: no pixel stage; only depth is written.
    pub pixel: Option<ShaderStage<'a>>,
    /// Color targets in attachment order.
    pub colors: &'a [ColorTarget],
    pub depth: Option<DepthTarget>,
    pub raster: RasterState,
    /// Rasterization samples per pixel: 1, or more for pipelines with no targets.
    pub samples: u32,
}

impl<'a> MeshPipelineDesc<'a> {
    /// A mesh stage alone: no targets, default raster state, one sample.
    pub fn new(mesh: ShaderStage<'a>) -> Self {
        Self {
            task: None,
            mesh,
            pixel: None,
            colors: &[],
            depth: None,
            raster: RasterState::default(),
            samples: 1,
        }
    }
}

/// A task/mesh/pixel pipeline. Not implemented on Metal yet:
/// [`Device::create_mesh_pipeline`] fails.
pub struct MeshPipeline {
    never: std::convert::Infallible,
}

/// A color attachment of [`Rendering`]: mip 0, layer 0 of a texture with `COLOR` usage.
#[derive(Clone, Copy)]
pub struct ColorAttachment<'a> {
    pub texture: &'a Texture,
    pub load: ColorLoad,
    /// Keep what rendering wrote; otherwise the contents are undefined afterwards.
    pub store: bool,
}

/// The depth attachment of [`Rendering`]: mip 0, layer 0 of a texture with `DEPTH` usage.
#[derive(Clone, Copy)]
pub struct DepthAttachment<'a> {
    pub texture: &'a Texture,
    pub load: DepthLoad,
    pub store: bool,
}

/// What [`CommandList::begin_rendering`] draws into. With no attachments the extent alone
/// sets the raster grid.
#[derive(Clone, Copy)]
pub struct Rendering<'a> {
    /// Width and height in pixels, from the origin.
    pub extent: [u32; 2],
    pub colors: &'a [ColorAttachment<'a>],
    pub depth: Option<DepthAttachment<'a>>,
}

impl Device {
    pub fn create_mesh_pipeline(
        &self,
        d: &MeshPipelineDesc<'_>,
        name: &str,
    ) -> Result<MeshPipeline, Error> {
        let _ = (d, name);
        Err(Error::NotImplemented("Device::create_mesh_pipeline"))
    }
}

impl MeshPipeline {
    pub(super) fn unreachable(&self) -> ! {
        match self.never {}
    }
}

impl CommandList<'_> {
    pub fn begin_rendering(&mut self, r: &Rendering<'_>) {
        let _ = r;
        not_implemented("CommandList::begin_rendering")
    }

    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        let _ = (offset, extent);
        not_implemented("CommandList::set_scissor")
    }

    pub fn end_rendering(&mut self) {
        not_implemented("CommandList::end_rendering")
    }
}
