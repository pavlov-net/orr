use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::Zeroable;
use gpui::{
    AtlasTile, Background, BorderStyle, Bounds, Corners, Hsla, PathCurve, Point, PrimitiveBatch,
    Rgba, ScaledPixels, Scene,
};
use orr_gpu::Error;

use crate::atlas::{Pages, Tile};
use crate::shaders::painter::{Curve, Fill, Instance};

pub(crate) const QUAD: u32 = 0;
pub(crate) const SHADOW: u32 = 1;
pub(crate) const UNDERLINE: u32 = 2;
pub(crate) const PATH: u32 = 3;
pub(crate) const MASK_SPRITE: u32 = 4;
pub(crate) const COLOR_SPRITE: u32 = 5;

pub(crate) struct Batch {
    pub(crate) first: u32,
    pub(crate) len: u32,
    pub(crate) kind: u32,
    pub(crate) page: Option<u32>,
}

pub(crate) struct SceneData {
    pub(crate) instances: Vec<Instance>,
    pub(crate) curves: Vec<Curve>,
    pub(crate) batches: Vec<Batch>,
    pub(crate) tiles: HashMap<(u32, u32), Arc<Tile>>,
}

struct SpriteUv {
    rect: [f32; 4],
    allocation: [f32; 4],
}

impl SceneData {
    pub(crate) fn from_gpui(scene: &Scene, atlas: &Pages) -> Result<Self, Error> {
        let mut frame = Self {
            instances: Vec::new(),
            curves: Vec::new(),
            batches: Vec::new(),
            tiles: HashMap::new(),
        };
        let mut paints = Vec::new();
        let mut curves = Vec::new();
        let mut tiles = Vec::new();
        for batch in scene.batches() {
            let first = count(frame.instances.len())?;
            let mut page = None;
            let kind = match batch {
                PrimitiveBatch::Quads(range) => {
                    frame
                        .instances
                        .extend(scene.quads[range].iter().map(|quad| {
                            let bounds = rect(quad.bounds);
                            Instance {
                                bounds,
                                clip: rect(quad.content_mask.bounds),
                                shape: bounds,
                                radii: radii(quad.corner_radii),
                                edges: [
                                    quad.border_widths.left.0,
                                    quad.border_widths.top.0,
                                    quad.border_widths.right.0,
                                    quad.border_widths.bottom.0,
                                ],
                                color: rgba(quad.border_color),
                                fill: fill(quad.background),
                                flags: u32::from(quad.border_style == BorderStyle::Dashed),
                                ..Instance::zeroed()
                            }
                        }));
                    QUAD
                }
                PrimitiveBatch::Shadows(range) => {
                    frame
                        .instances
                        .extend(scene.shadows[range].iter().map(|shadow| {
                            Instance {
                                bounds: rect(shadow.bounds),
                                clip: rect(shadow.content_mask.bounds),
                                shape: rect(shadow.shape_bounds),
                                radii: radii(
                                    shadow
                                        .shape_corner_radii
                                        .clamp_radii_for_quad_size(shadow.shape_bounds.size),
                                ),
                                edges: radii(shadow.corner_radii),
                                color: rgba(shadow.color),
                                parameter: shadow.blur_radius.0,
                                flags: u32::from(shadow.inset.as_bool()),
                                ..Instance::zeroed()
                            }
                        }));
                    SHADOW
                }
                PrimitiveBatch::Underlines(range) => {
                    frame
                        .instances
                        .extend(scene.underlines[range].iter().map(|line| Instance {
                            bounds: rect(line.bounds),
                            clip: rect(line.content_mask.bounds),
                            color: rgba(line.color),
                            parameter: line.thickness.0,
                            flags: u32::from(line.wavy.as_bool()),
                            ..Instance::zeroed()
                        }));
                    UNDERLINE
                }
                PrimitiveBatch::Paths(range) => {
                    paints.clear();
                    curves.clear();
                    tiles.clear();
                    for path in &scene.paths[range] {
                        path.append_to(&mut paints, &mut curves, &mut tiles);
                    }
                    let curve_base = count(frame.curves.len())?;
                    count(frame.curves.len() + curves.len())?;
                    frame.curves.extend(curves.iter().map(curve));
                    let fills: Vec<_> = paints.iter().map(|paint| fill(paint.color)).collect();
                    frame.instances.extend(tiles.iter().map(|tile| {
                        let paint = &paints[tile.paint as usize];
                        Instance {
                            bounds: [
                                tile.top_left.x,
                                tile.top_left.y,
                                tile.bottom_right.x,
                                tile.bottom_right.y,
                            ],
                            clip: rect(paint.content_mask.bounds),
                            shape: rect(paint.bounds),
                            fill: fills[tile.paint as usize],
                            offset: xy(tile.reference),
                            curve_start: curve_base + tile.curve_start,
                            curve_count: tile.curve_count,
                            backdrop: tile.backdrop,
                            flags: u32::from(paint.even_odd.as_bool()),
                            ..Instance::zeroed()
                        }
                    }));
                    PATH
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                    page = Some(texture_id.index);
                    for sprite in &scene.monochrome_sprites[range] {
                        if let Some(uv) = frame.sprite(atlas, sprite.tile)? {
                            let transform = sprite.transformation;
                            frame.instances.push(Instance {
                                bounds: rect(sprite.bounds),
                                clip: rect(sprite.content_mask.bounds),
                                shape: uv.allocation,
                                color: rgba(sprite.color),
                                uv: uv.rect,
                                transform_x: transform.rotation_scale[0],
                                transform_y: transform.rotation_scale[1],
                                offset: transform.translation,
                                ..Instance::zeroed()
                            });
                        }
                    }
                    MASK_SPRITE
                }
                PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                    page = Some(texture_id.index);
                    for sprite in &scene.polychrome_sprites[range] {
                        if let Some(uv) = frame.sprite(atlas, sprite.tile)? {
                            frame.instances.push(Instance {
                                bounds: rect(sprite.bounds),
                                clip: rect(sprite.content_mask.bounds),
                                shape: uv.allocation,
                                radii: radii(sprite.corner_radii),
                                uv: uv.rect,
                                parameter: sprite.opacity,
                                flags: u32::from(sprite.grayscale.as_bool()),
                                ..Instance::zeroed()
                            });
                        }
                    }
                    COLOR_SPRITE
                }
                PrimitiveBatch::SubpixelSprites { .. } => {
                    return Err(Error::Invalid(
                        "a transparent UI layer needs grayscale text coverage".into(),
                    ));
                }
                PrimitiveBatch::Surfaces(_) | PrimitiveBatch::NativeViews(_) => {
                    return Err(Error::Invalid(
                        "external surfaces and native views need host composition".into(),
                    ));
                }
            };
            let len = count(frame.instances.len())? - first;
            if len != 0 {
                frame.batches.push(Batch {
                    first,
                    len,
                    kind,
                    page,
                });
            }
        }
        Ok(frame)
    }

    fn sprite(&mut self, atlas: &Pages, tile: AtlasTile) -> Result<Option<SpriteUv>, Error> {
        let page = atlas
            .pages
            .get(tile.texture_id.index as usize)
            .filter(|page| page.id == tile.texture_id)
            .ok_or_else(|| Error::Invalid("GPUI must use the painter's sprite atlas".into()))?;
        // A retained scene can refer to an image removed from the atlas before the next draw.
        let Some(pixels) = page.tile(tile) else {
            return Ok(None);
        };
        let [x, y, w, h] = pixels.bounds;
        let [width, height] = page.extent.map(|n| n as f32);
        let normalize =
            |[x0, y0, x1, y1]: [f32; 4]| [x0 / width, y0 / height, x1 / width, y1 / height];
        let left = tile.bounds.origin.x.0 as f32;
        let top = tile.bounds.origin.y.0 as f32;
        let right = left + tile.bounds.size.width.0 as f32;
        let bottom = top + tile.bounds.size.height.0 as f32;
        self.tiles.insert((page.id.index, tile.tile_id.0), pixels);
        Ok(Some(SpriteUv {
            rect: normalize([left, top, right, bottom]),
            allocation: normalize([x as f32, y as f32, (x + w) as f32, (y + h) as f32]),
        }))
    }
}

fn count(n: usize) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| Error::Invalid("GPUI scene exceeds 32-bit indexing".into()))
}

fn rect(bounds: Bounds<ScaledPixels>) -> [f32; 4] {
    let x = bounds.origin.x.0;
    let y = bounds.origin.y.0;
    [x, y, x + bounds.size.width.0, y + bounds.size.height.0]
}

fn radii(corners: Corners<ScaledPixels>) -> [f32; 4] {
    [
        corners.top_left.0,
        corners.top_right.0,
        corners.bottom_right.0,
        corners.bottom_left.0,
    ]
}

fn rgba(hsla: Hsla) -> [f32; 4] {
    let Rgba { r, g, b, a } = hsla.into();
    [r, g, b, a]
}

fn xy(point: Point<f32>) -> [f32; 2] {
    [point.x, point.y]
}

fn curve(curve: &PathCurve) -> Curve {
    Curve {
        start: xy(curve.p0),
        end: xy(curve.p1),
        quadratic: xy(curve.a),
        linear: xy(curve.b),
        winding: curve.sign,
        _pad0: 0,
    }
}

// The pinned GPUI background is a C-layout shader record with private fields.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BackgroundLayout {
    kind: u32,
    color_space: u32,
    solid: [f32; 4],
    parameter: f32,
    stops: [[f32; 5]; 2],
    padding: u32,
}

const _: () = {
    assert!(size_of::<Background>() == size_of::<BackgroundLayout>());
    assert!(align_of::<Background>() == align_of::<BackgroundLayout>());
};

fn fill(background: Background) -> Fill {
    if let Some(color) = background.as_solid() {
        return Fill {
            color0: rgba(color),
            ..Fill::zeroed()
        };
    }
    // SAFETY: GPUI 0.3.14 defines Background, Hsla and LinearColorStop with C layout.
    // The mirror represents the two enum tags as u32 and contains every initialized field.
    let raw = unsafe { std::ptr::read((&raw const background).cast::<BackgroundLayout>()) };
    let color = |[h, s, l, a]: [f32; 4]| rgba(Hsla { h, s, l, a });
    let [a, b] = raw.stops;
    Fill {
        color0: if raw.kind == 1 {
            color([a[0], a[1], a[2], a[3]])
        } else {
            color(raw.solid)
        },
        color1: color([b[0], b[1], b[2], b[3]]),
        stops: [a[4], b[4]],
        parameter: raw.parameter,
        kind: raw.kind,
        color_space: raw.color_space,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}

#[cfg(test)]
mod tests {
    use gpui::{ColorSpace, checkerboard, linear_color_stop, linear_gradient, pattern_slash, rgb};

    use super::fill;

    #[test]
    fn background_layout_decodes_the_pinned_gpui_colors() {
        let red = rgb(0xff0000);
        let blue = rgb(0x0000ff);
        let gradient = fill(
            linear_gradient(
                31.0,
                linear_color_stop(red, 0.125),
                linear_color_stop(blue, 0.875),
            )
            .color_space(ColorSpace::Oklab),
        );
        assert_eq!(gradient.kind, 1);
        assert_eq!(gradient.color_space, 1);
        assert_eq!(gradient.parameter, 31.0);
        assert_eq!(gradient.stops, [0.125, 0.875]);
        assert_eq!(gradient.color0, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(gradient.color1, [0.0, 0.0, 1.0, 1.0]);

        let slash = fill(pattern_slash(blue, 1.0, 2.0));
        assert_eq!(slash.kind, 2);
        assert_eq!(slash.parameter, (255 * 65535 + 510) as f32);
        assert_eq!(slash.color0, [0.0, 0.0, 1.0, 1.0]);

        let checks = fill(checkerboard(red, 4.0));
        assert_eq!(checks.kind, 3);
        assert_eq!(checks.parameter, 4.0);
        assert_eq!(checks.color0, [1.0, 0.0, 0.0, 1.0]);
    }
}
