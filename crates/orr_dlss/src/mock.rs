//! The backend without NGX. With the `mock` feature it reports every feature unsupported and
//! ray reconstruction and super resolution blit color to the output; frame generation cannot
//! be created. Without it, initialization fails.

use crate::raw::{FgInputs, Image, RrInputs, SrInputs, Upscaler, Vulkan};
use crate::{DlssError, Quality, Support, Uuid};
use ash::vk;
use std::ffi::CString;

pub(crate) fn take_log() -> Vec<String> {
    Vec::new()
}

pub(crate) fn instance_extensions(_project_id: Uuid) -> Result<Vec<CString>, DlssError> {
    Ok(Vec::new())
}

pub(crate) unsafe fn device_extensions(
    _instance: &ash::Instance,
    _physical_device: vk::PhysicalDevice,
    _project_id: Uuid,
) -> Result<Vec<CString>, DlssError> {
    Ok(Vec::new())
}

pub(crate) struct Ngx {
    device: ash::Device,
}

impl Ngx {
    pub(crate) unsafe fn new(vulkan: Vulkan<'_>, _project_id: Uuid) -> Result<Self, DlssError> {
        if cfg!(feature = "mock") {
            Ok(Self {
                device: vulkan.device.clone(),
            })
        } else {
            Err(DlssError::NoSdk)
        }
    }

    pub(crate) fn support(&self) -> Support {
        Support::default()
    }

    pub(crate) fn vram_bytes(&self) -> u64 {
        0
    }

    /// NGX's published scale factors.
    pub(crate) fn render_size(
        &self,
        output: [u32; 2],
        quality: Quality,
        _upscaler: Upscaler,
    ) -> Result<[u32; 2], DlssError> {
        let scale = match quality {
            Quality::Dlaa => 1.0,
            Quality::Quality => 2.0 / 3.0,
            Quality::Balanced => 0.58,
            Quality::Performance => 0.5,
            Quality::UltraPerformance => 1.0 / 3.0,
        };
        Ok(output.map(|o| ((f64::from(o) * scale).round() as u32).max(1)))
    }
}

pub(crate) struct Upscale {
    device: ash::Device,
}

impl Upscale {
    pub(crate) unsafe fn new(
        ngx: &Ngx,
        _cmd: vk::CommandBuffer,
        _upscaler: Upscaler,
        _render: [u32; 2],
        _output: [u32; 2],
        _quality: Quality,
        _preset: Option<u32>,
    ) -> Result<Self, DlssError> {
        Ok(Self {
            device: ngx.device.clone(),
        })
    }

    pub(crate) unsafe fn record_rr(
        &mut self,
        cmd: vk::CommandBuffer,
        inputs: &RrInputs<'_>,
        render: [u32; 2],
    ) -> Result<(), DlssError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.blit(cmd, inputs.color, inputs.output, render) };
        Ok(())
    }

    pub(crate) unsafe fn record_sr(
        &mut self,
        cmd: vk::CommandBuffer,
        inputs: &SrInputs<'_>,
        render: [u32; 2],
    ) -> Result<(), DlssError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.blit(cmd, inputs.color, inputs.output, render) };
        Ok(())
    }

    /// A bilinear blit of `color`'s `render` corner over all of `output`.
    unsafe fn blit(&self, cmd: vk::CommandBuffer, color: &Image, output: &Image, render: [u32; 2]) {
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let corner = |e: [u32; 2]| vk::Offset3D {
            x: e[0] as i32,
            y: e[1] as i32,
            z: 1,
        };
        let region = vk::ImageBlit::default()
            .src_subresource(layers)
            .src_offsets([vk::Offset3D::default(), corner(render)])
            .dst_subresource(layers)
            .dst_offsets([vk::Offset3D::default(), corner(output.extent)]);
        // SAFETY: `cmd` is recording on this device and both images are in GENERAL (caller).
        unsafe {
            self.device.cmd_blit_image(
                cmd,
                color.image,
                vk::ImageLayout::GENERAL,
                output.image,
                vk::ImageLayout::GENERAL,
                &[region],
                vk::Filter::LINEAR,
            );
        }
    }
}

pub(crate) struct Fg;

impl Fg {
    pub(crate) unsafe fn new(
        _ngx: &Ngx,
        _cmd: vk::CommandBuffer,
        _output: [u32; 2],
        _format: vk::Format,
    ) -> Result<Self, DlssError> {
        Err(DlssError::Unsupported("frame generation"))
    }

    pub(crate) unsafe fn record(
        &mut self,
        _cmd: vk::CommandBuffer,
        _inputs: &FgInputs<'_>,
        _index: u32,
        _count: u32,
    ) -> Result<(), DlssError> {
        Err(DlssError::Unsupported("frame generation"))
    }
}
