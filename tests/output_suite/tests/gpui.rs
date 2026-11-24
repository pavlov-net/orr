use gpui::{
    AtlasKey, Bounds, ContentMask, DevicePixels, ImageId, PaintedPath, PathBuilder,
    PolychromeSprite, Quad, RenderImageParams, ScaledPixels, Scene, point, px, rgba, size,
};
use orr_gpui::Painter;
use orr_output::UiLayer;
use orr_scene::orr_gpu::{Hazards, QueueKind, Stages, TextureRegion};
use orr_scene::testing::{assert_clean, device, read_texture, wait};

#[test]
fn gpui_layer_preserves_order_clip_and_atlas_pixels() {
    let gpu = device();
    let mut painter = Painter::new(gpu).unwrap();
    let atlas = painter.sprite_atlas();
    let image = AtlasKey::Image(RenderImageParams {
        image_id: ImageId(0),
        frame_index: 0,
    });
    let tile = atlas
        .get_or_insert_with(image.clone(), &mut || {
            Ok(Some((
                size(DevicePixels(2), DevicePixels(1)),
                [255, 0, 0, 128, 0, 255, 0, 255].as_slice().into(),
            )))
        })
        .unwrap()
        .unwrap();

    let bounds = |x, y, w, h| Bounds {
        origin: point(ScaledPixels(x), ScaledPixels(y)),
        size: size(ScaledPixels(w), ScaledPixels(h)),
    };
    let clip = ContentMask {
        bounds: bounds(0.0, 0.0, 18.0, 32.0),
    };
    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        bounds: bounds(2.0, 2.0, 20.0, 20.0),
        content_mask: clip,
        background: rgba(0xff000080).into(),
        ..Quad::default()
    });
    let mut left = tile;
    left.bounds.size.width = DevicePixels(1);
    let mut right = left;
    right.bounds.origin.x.0 += 1;
    let sprite = PolychromeSprite {
        order: 0,
        pad: 0,
        tile: left,
        grayscale: false.into(),
        opacity: 1.0,
        bounds: bounds(8.0, 5.0, 4.0, 4.0),
        content_mask: clip,
        corner_radii: Default::default(),
    };
    scene.insert_primitive(sprite);
    scene.insert_primitive(PolychromeSprite {
        tile: right,
        bounds: bounds(13.0, 5.0, 4.0, 4.0),
        ..sprite
    });
    let mut path = PathBuilder::fill();
    path.move_to(point(px(3.0), px(22.0)));
    path.line_to(point(px(13.0), px(22.0)));
    path.line_to(point(px(3.0), px(30.0)));
    path.close();
    scene.insert_primitive(PaintedPath::new(
        path.build().unwrap(),
        1.0,
        ContentMask {
            bounds: Bounds {
                origin: point(px(0.0), px(0.0)),
                size: size(px(32.0), px(32.0)),
            },
        },
        rgba(0x00ff00ff).into(),
    ));
    scene.finish();
    let prepared = painter.prepare(&scene).unwrap();

    // Replacing an image must keep the pixels of an older prepared scene alive.
    atlas.remove(&image);
    let replacement = atlas
        .get_or_insert_with(image, &mut || {
            Ok(Some((
                size(DevicePixels(1), DevicePixels(1)),
                [0, 255, 0, 255].as_slice().into(),
            )))
        })
        .unwrap()
        .unwrap();
    let mut next = Scene::default();
    next.insert_primitive(PolychromeSprite {
        tile: replacement,
        ..scene.polychrome_sprites[0]
    });
    next.finish();
    let _next = painter.prepare(&next).unwrap();

    let layer = UiLayer::new(gpu, [32, 32]).unwrap();
    let queue = gpu.queue(QueueKind::Render).unwrap();
    let mut cmd = queue.command_list().unwrap();
    layer.clear(&mut cmd);
    painter.paint(&mut cmd, layer.texture(), &prepared).unwrap();
    cmd.barrier(Stages::COLOR_OUT, Stages::TRANSFER, Hazards::empty());
    wait(queue.submit(vec![cmd], &[]).unwrap());
    let bytes = read_texture(
        layer.texture(),
        &TextureRegion::whole_mip(layer.texture().desc(), 0),
    );

    let pixel = |x: usize, y: usize| &bytes[(y * 32 + x) * 4..][..4];
    assert_eq!(pixel(0, 0), [0, 0, 0, 0]);
    assert_eq!(pixel(5, 5), [128, 0, 0, 128]);
    assert_eq!(pixel(20, 5), [0, 0, 0, 0]);
    assert_eq!(pixel(9, 6), [64, 0, 128, 192]);
    assert_eq!(pixel(15, 6), [0, 255, 0, 255]);
    assert_eq!(pixel(4, 24), [0, 255, 0, 255]);
    assert_clean();
}
