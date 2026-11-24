//! Raw Metal access for libraries that encode their own work into orr's command buffers:
//! the device, a texture's raw object, the command buffer and deferred release.

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4CommandBuffer, MTL4CommandEncoder, MTL4VisibilityOptions, MTLDevice, MTLStages, MTLTexture,
};

use super::command::CommandList;
use super::device::Device;
use super::texture::Texture;

/// The raw Metal objects of a device. A library that builds pipelines makes a compiler of its
/// own: orr's records what it builds into orr's pipeline cache.
pub struct RawDevice<'a> {
    pub device: &'a ProtocolObject<dyn MTLDevice>,
}

/// The raw objects of `device`.
pub fn raw_device(device: &Device) -> RawDevice<'_> {
    RawDevice {
        device: &device.shared.device,
    }
}

/// The raw texture behind `t`. orr's textures are private and untracked; the usage it was
/// created with is [`MTLTexture::usage`].
pub fn texture_handle(t: &Texture) -> &ProtocolObject<dyn MTLTexture> {
    &t.raw
}

/// Runs `release` once the GPU is done with the work submitted to `device` so far and with
/// the command lists open now, from [`Device::collect_garbage`] or when the device is
/// destroyed. For objects a library made on the device and its encoders use.
pub fn defer_after_submitted(device: &Device, release: impl FnOnce() + Send + 'static) {
    device.shared.defer(Box::new(release));
}

impl CommandList<'_> {
    /// Has `encode` encode a library's own encoders into the list's command buffer, with no
    /// encoder open, when the list is encoded at submit. Their work starts after everything
    /// recorded on the queue before, whatever barriers were pending: orr's barriers wait only
    /// in encoders orr opens, so this ends the list's work with an encoder whose producer
    /// barrier holds back every later encoder. Work recorded on the list afterwards waits for
    /// the library's only through a [`CommandList::barrier`], as for orr's own work.
    pub fn metal_raw(
        &mut self,
        encode: impl FnOnce(&ProtocolObject<dyn MTL4CommandBuffer>) + Send + 'static,
    ) {
        assert!(
            self.pass.is_none(),
            "raw encoders are recorded outside rendering"
        );
        let scratch = self.scratch();
        self.record(move |e| {
            let c = e.carrier(&scratch);
            c.barrierAfterStages_beforeQueueStages_visibilityOptions(
                MTLStages::All,
                MTLStages::All,
                MTL4VisibilityOptions::Device,
            );
            e.end_encoder();
            encode(&e.buffer);
        });
    }
}
