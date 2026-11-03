//! The layer in front of a pixel's dominant surface, which the blurs take into account.

use orr_scene::orr_gpu::TextureIndex;

/// A view's nearest overlay or see-through layer per pixel, at the render extent, as lighting
/// resolves it. Motion blur and depth of field read depth and motion blended from the dominant
/// surface's toward the layer's by `weight`, so that share of the pixel does not blur like what
/// lies behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayFront {
    /// R32F reverse-Z.
    pub depth: TextureIndex,
    /// RG16F previous minus current UV.
    pub motion: TextureIndex,
    /// R16F, the layer's share of the pixel; 0 on pixels without a layer.
    pub weight: TextureIndex,
}

impl OverlayFront {
    /// The front layer's depth, motion and weight, each NONE for no layer.
    pub(crate) fn indices(overlay: Option<Self>) -> [TextureIndex; 3] {
        overlay.map_or([TextureIndex::NONE; 3], |o| [o.depth, o.motion, o.weight])
    }
}
