//! Hook points: consumer passes on a view's image after RR, at output resolution, with depth,
//! motion, a read-only source and the distortion field.

use orr_scene::ViewConstants;
use orr_scene::orr_gpu::{
    CommandList, Device, Format, GpuPtr, Hazards, Stages, StorageImageIndex, Texture, TextureDesc,
    TextureIndex, TextureMemory, TextureUsage, ViewDesc,
};

use crate::Result;

/// One run of a consumer hook. A screen warp reads `source` and writes `image`; an overlay
/// depth-tests against `depth` in its own shader.
pub struct HookPass<'a, 'q> {
    pub cmd: &'a mut CommandList<'q>,
    /// The chain's image at this point; the hook writes it.
    pub image: StorageImageIndex,
    /// `image` as it was before the hook, read-only.
    pub source: TextureIndex,
    /// R32F reverse-Z depth at `extent`.
    pub depth: TextureIndex,
    /// RG16F previous minus current UV at `extent`.
    pub motion: TextureIndex,
    /// Frame generation's distortion field at half of `extent`, rounded up; `NONE` unless
    /// the view has the field on. A hook that moves pixels reads `distortion_source`,
    /// composes its warp into it, writes every texel of `distortion` and sets
    /// `distortion_written`.
    pub distortion: StorageImageIndex,
    /// The field before this hook, read-only; `NONE` unless the field is on.
    pub distortion_source: TextureIndex,
    pub distortion_written: bool,
    pub extent: [u32; 2],
    pub view: GpuPtr<ViewConstants>,
    /// Render pixels: each render sample sits at its pixel centre minus the jitter.
    pub jitter: [f32; 2],
}

/// Where a hook runs in a view's output chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookPoint {
    /// On RR's pre-exposed radiance, before metering.
    SceneReferred,
    /// On tonemap's display-linear image, before UI and compose.
    DisplayLinear,
}

/// The hooks a view sets this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hooks {
    pub scene_referred: bool,
    pub display_linear: bool,
}

impl Hooks {
    pub fn any(self) -> bool {
        self.scene_referred || self.display_linear
    }
}

/// The two output-resolution RGBA16F images a view's output chain alternates between.
/// `rr_output` holds the radiance the next pass reads, RR's output at first; `linear` is the
/// free image tonemap writes as the display-linear target. A camera effect writes `linear`
/// and swaps the two. Each hook's source is the other image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainImages {
    pub rr_output: TextureIndex,
    pub rr_output_store: StorageImageIndex,
    pub linear: TextureIndex,
    pub linear_store: StorageImageIndex,
}

impl ChainImages {
    /// The chain after a pass that read `rr_output` and wrote `linear`.
    pub fn swapped(self) -> Self {
        Self {
            rr_output: self.linear,
            rr_output_store: self.linear_store,
            linear: self.rr_output,
            linear_store: self.rr_output_store,
        }
    }
}

/// A texture with a sampled and a storage view, both freed with it.
pub struct Image {
    texture: Texture,
    pub sampled: TextureIndex,
    pub storage: StorageImageIndex,
    bytes: u64,
    device: Device,
}

impl Image {
    pub fn new(gpu: &Device, format: Format, extent: [u32; 2], name: &str) -> Result<Self> {
        let desc = TextureDesc::d2(
            format,
            extent[0],
            extent[1],
            TextureUsage::SAMPLED | TextureUsage::STORAGE,
        );
        let texture = gpu.create_texture(&desc, TextureMemory::Dedicated, name)?;
        let view = ViewDesc::default();
        let sampled = gpu.write_sampled(gpu.alloc_image_slot()?, &texture, &view);
        let storage = gpu.write_storage(gpu.alloc_image_slot()?, &texture, &view);
        Ok(Self {
            bytes: gpu.texture_requirements(&desc).0,
            texture,
            sampled,
            storage,
            device: gpu.clone(),
        })
    }

    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        self.device.free_image_slot_after_submitted(self.sampled.0);
        self.device.free_image_slot_after_submitted(self.storage.0);
    }
}

/// Frame generation's bidirectional distortion field (DLSS-FG's
/// `DLSSG.BidirectionalDistortionField`): RGBA16F at half the output resolution, RG the UV
/// offset from a presented pixel to its undistorted position, BA the reverse, zero for
/// identity. Two images ping-pong so each warping stage reads the previous field and writes
/// the next. A display keeps one.
pub struct DistortionField {
    images: [Image; 2],
    current: usize,
    fresh: bool,
}

impl DistortionField {
    /// A field for an output of `extent`.
    pub fn new(gpu: &Device, extent: [u32; 2]) -> Result<Self> {
        let half = Self::extent_for(extent);
        let image = || Image::new(gpu, Format::Rgba16Float, half, "distortion field");
        Ok(Self {
            images: [image()?, image()?],
            current: 0,
            fresh: true,
        })
    }

    pub(crate) fn extent_for(output: [u32; 2]) -> [u32; 2] {
        output.map(|n| n.div_ceil(2))
    }

    pub fn extent(&self) -> [u32; 2] {
        let [w, h, _] = self.images[0].texture.desc().extent;
        [w, h]
    }

    /// The field as the last stage left it: what frame generation reads.
    pub fn current(&self) -> TextureIndex {
        self.images[self.current].sampled
    }

    pub fn bytes(&self) -> u64 {
        self.images.iter().map(|i| i.bytes).sum()
    }

    pub(crate) fn current_storage(&self) -> StorageImageIndex {
        self.images[self.current].storage
    }

    /// Records the discard each image needs before its first write.
    pub(crate) fn prepare(&mut self, list: &mut CommandList<'_>) {
        if std::mem::take(&mut self.fresh) {
            for image in &self.images {
                list.discard(image.texture());
            }
        }
    }
}

/// Depth and motion at output resolution, allocated while the view sets a hook.
pub(crate) struct HookTargets {
    pub(crate) depth: Image,
    pub(crate) motion: Image,
    pub(crate) fresh: bool,
}

impl HookTargets {
    pub(crate) fn new(gpu: &Device, extent: [u32; 2]) -> Result<Self> {
        Ok(Self {
            depth: Image::new(gpu, Format::R32Float, extent, "hook depth")?,
            motion: Image::new(gpu, Format::Rg16Float, extent, "hook motion")?,
            fresh: true,
        })
    }

    pub(crate) fn extent(&self) -> [u32; 2] {
        let [w, h, _] = self.depth.texture().desc().extent;
        [w, h]
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.depth.bytes + self.motion.bytes
    }
}

/// What a frame's hooks run with, from [`crate::OutputKernels::hook_inputs`].
#[derive(Clone, Copy, Debug)]
pub struct HookInputs {
    pub(crate) hooks: Hooks,
    pub(crate) depth: TextureIndex,
    pub(crate) motion: TextureIndex,
    pub(crate) extent: [u32; 2],
    pub(crate) view: GpuPtr<ViewConstants>,
    pub(crate) jitter: [f32; 2],
}

impl HookInputs {
    /// Records the hook at `point`, between barriers that order everything before it and
    /// make its writes visible to every later pass. `chain` is the chain as it stands at the
    /// point: for `SceneReferred` the one given to `hook_inputs`, for `DisplayLinear` the one
    /// tonemap wrote. Pass the field given to `hook_inputs`.
    pub fn run(
        &self,
        list: &mut CommandList<'_>,
        point: HookPoint,
        chain: &ChainImages,
        field: Option<&mut DistortionField>,
        hook: &mut dyn FnMut(&mut HookPass<'_, '_>),
    ) {
        let (image, source) = match point {
            HookPoint::SceneReferred => {
                debug_assert!(self.hooks.scene_referred, "no scene_referred source copied");
                (chain.rr_output_store, chain.linear)
            }
            HookPoint::DisplayLinear => {
                debug_assert!(
                    self.hooks.display_linear,
                    "no display_linear source written"
                );
                (chain.linear_store, chain.rr_output)
            }
        };
        let (distortion, distortion_source) = match &field {
            Some(f) => (f.images[f.current ^ 1].storage, f.current()),
            None => (StorageImageIndex::NONE, TextureIndex::NONE),
        };
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        let mut pass = HookPass {
            cmd: list,
            image,
            source,
            depth: self.depth,
            motion: self.motion,
            distortion,
            distortion_source,
            distortion_written: false,
            extent: self.extent,
            view: self.view,
            jitter: self.jitter,
        };
        hook(&mut pass);
        let written = pass.distortion_written;
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        if let Some(f) = field
            && written
        {
            f.current ^= 1;
        }
    }
}
