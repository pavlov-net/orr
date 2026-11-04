//! Subsurface walk settings and the subsurface table on the GPU.

use orr_scene::orr_gpu::{Allocation, Device, GpuPtr, MemoryKind};

use crate::Error;
use crate::shaders::types::SubsurfaceTableGpu;
use crate::subsurface_table as table;

/// Which subsurface vertices of a path walk; the others shade their subsurface share as
/// diffuse reflection in the subsurface color.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SubsurfaceDepth {
    /// The first subsurface vertex of each path.
    #[default]
    FirstEvent,
    /// A path's first vertex alone.
    PrimaryOnly,
}

/// The range of [`ViewLighting::sss_max_steps`](crate::ViewLighting::sss_max_steps): the
/// step caps the subsurface table covers.
pub const SSS_STEPS: std::ops::RangeInclusive<u32> =
    table::CAPS[0]..=table::CAPS[table::CAPS.len() - 1];

/// The subsurface table on the GPU, which every walk reads through
/// [`PathRoot::subsurface_table`](crate::types::PathRoot::subsurface_table): what a walk stopped
/// at its step cap would yet carry out, by single-scattering albedo, anisotropy, cap and the IOR
/// of the surface it meets.
pub struct SubsurfaceTable {
    header: Allocation<SubsurfaceTableGpu>,
    _values: Allocation<f32>,
}

impl SubsurfaceTable {
    /// Uploads the table.
    pub fn new(device: &Device) -> Result<Self, Error> {
        let mut values = device.gpu_malloc::<f32>(
            MemoryKind::Upload,
            table::VALUES.len() as u64,
            "subsurface table",
        )?;
        values
            .as_mut_slice()
            .expect("upload memory is mapped")
            .copy_from_slice(&table::VALUES);
        let mut header = device.gpu_malloc::<SubsurfaceTableGpu>(
            MemoryKind::Upload,
            1,
            "subsurface table header",
        )?;
        header.as_mut_slice().expect("upload memory is mapped")[0] = SubsurfaceTableGpu {
            values: values.gpu(),
            albedos: table::ALBEDOS,
            anisotropies: table::ANISOTROPIES,
            caps: table::CAPS.len() as u32,
            first_cap: table::CAPS[0],
            g_max: table::G_MAX,
            iors: table::IORS,
            ior_share_max: table::IOR_SHARE_MAX,
            _pad0: 0,
        };
        Ok(Self {
            header,
            _values: values,
        })
    }

    /// The table's header, for
    /// [`PathRoot::subsurface_table`](crate::types::PathRoot::subsurface_table).
    pub fn gpu(&self) -> GpuPtr<SubsurfaceTableGpu> {
        self.header.gpu()
    }
}
