//! The UI layer UI renderers draw into and compose blends over the image.

use orr_scene::orr_gpu::{
    ClearColor, ColorAttachment, ColorLoad, CommandList, Device, Format, Hazards, Rendering,
    Stages, Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage, ViewDesc,
};

use crate::Result;

/// What a UI hook draws with: the frame's command list and the UI layer, already cleared,
/// to draw into as a colour attachment.
pub struct UiPass<'a, 'q> {
    pub cmd: &'a mut CommandList<'q>,
    /// RGBA8 UNORM at the display's extent, premultiplied, sRGB-encoded.
    pub layer: &'a Texture,
    /// The presented image within the frame the UI is for: 0 unless the UI is drawn per
    /// presented image.
    pub presented: u32,
    /// Seconds from the frame's own image to that presented image; 0 unless the UI is drawn
    /// per presented image.
    pub time_offset: f32,
}

/// A UI layer: RGBA8 UNORM at output resolution, holding premultiplied, sRGB-encoded colour
/// (egui's reference blending). A display keeps one per frame in flight.
pub struct UiLayer {
    texture: Texture,
    index: TextureIndex,
    device: Device,
}

impl UiLayer {
    /// Creates the layer; [`UiLayer::clear`] it before each frame's UI draws into it.
    pub fn new(gpu: &Device, extent: [u32; 2]) -> Result<Self> {
        let desc = TextureDesc::d2(
            Format::Rgba8Unorm,
            extent[0],
            extent[1],
            TextureUsage::SAMPLED | TextureUsage::COLOR,
        );
        let texture = gpu.create_texture(&desc, TextureMemory::Dedicated, "ui layer")?;
        let index = gpu.write_sampled(gpu.alloc_image_slot()?, &texture, &ViewDesc::default());
        Ok(Self {
            texture,
            index,
            device: gpu.clone(),
        })
    }

    /// The texture UI renderers draw into, as a colour attachment.
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// The sampled view compose reads.
    pub fn index(&self) -> TextureIndex {
        self.index
    }

    pub fn extent(&self) -> [u32; 2] {
        let [w, h, _] = self.texture.desc().extent;
        [w, h]
    }

    /// Records clearing the layer to transparent. Order earlier reads of the layer before
    /// it; it ends with a barrier, so UI draws may follow directly.
    pub fn clear(&self, list: &mut CommandList<'_>) {
        list.discard(&self.texture);
        list.begin_rendering(&Rendering {
            extent: self.extent(),
            colors: &[ColorAttachment {
                texture: &self.texture,
                load: ColorLoad::Clear(ClearColor::Float([0.0; 4])),
                store: true,
            }],
            depth: None,
        });
        list.end_rendering();
        list.barrier(Stages::COLOR_OUT, Stages::COLOR_OUT, Hazards::empty());
    }
}

impl Drop for UiLayer {
    fn drop(&mut self) {
        self.device.free_image_slot_after_submitted(self.index.0);
    }
}
