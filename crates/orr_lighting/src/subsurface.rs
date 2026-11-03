//! Subsurface walk settings and the walk's cap tail table on the GPU.

use orr_scene::orr_gpu::{Allocation, Device, GpuPtr, MemoryKind};

use crate::Error;
use crate::shaders::types::CapTail;

include!(concat!(env!("OUT_DIR"), "/cap_tail.rs"));

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
/// step caps the cap tail table covers.
pub const SSS_STEPS: std::ops::RangeInclusive<u32> =
    CAP_TAIL_CAPS[0]..=CAP_TAIL_CAPS[CAP_TAIL_CAPS.len() - 1];

/// The subsurface walk's cap tail table on the GPU, which every walk reads through
/// [`PathRoot::cap_tail`](crate::types::PathRoot::cap_tail): what a walk stopped at its step cap
/// would yet carry out, by single-scattering albedo, anisotropy, cap and the IOR of the
/// surface it meets.
pub struct CapTailTable {
    header: Allocation<CapTail>,
    _values: Allocation<f32>,
}

impl CapTailTable {
    /// Uploads the table, which the crate generates at build time.
    pub fn new(device: &Device) -> Result<Self, Error> {
        let mut values =
            device.gpu_malloc::<f32>(MemoryKind::Upload, CAP_TAIL.len() as u64, "cap tail")?;
        values
            .as_mut_slice()
            .expect("upload memory is mapped")
            .copy_from_slice(&CAP_TAIL);
        let mut header = device.gpu_malloc::<CapTail>(MemoryKind::Upload, 1, "cap tail header")?;
        header.as_mut_slice().expect("upload memory is mapped")[0] = CapTail {
            values: values.gpu(),
            albedos: CAP_TAIL_ALBEDOS,
            anisotropies: CAP_TAIL_ANISOTROPIES,
            caps: CAP_TAIL_CAPS.len() as u32,
            first_cap: CAP_TAIL_CAPS[0],
            g_max: CAP_TAIL_G_MAX,
            iors: CAP_TAIL_IORS,
            ior_share_max: CAP_TAIL_IOR_SHARE_MAX,
            _pad0: 0,
        };
        Ok(Self {
            header,
            _values: values,
        })
    }

    /// The table's header, for [`PathRoot::cap_tail`](crate::types::PathRoot::cap_tail).
    pub fn gpu(&self) -> GpuPtr<CapTail> {
        self.header.gpu()
    }
}
