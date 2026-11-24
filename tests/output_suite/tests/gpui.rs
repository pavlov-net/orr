use gpui::{
    AtlasKey, Bounds, ColorSpace, ContentMask, Corners, DevicePixels, ImageId, PaintedPath,
    PathBuilder, PolychromeSprite, Quad, RenderImageParams, ScaledPixels, Scene, Shadow,
    linear_color_stop, linear_gradient, point, px, rgba, size,
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
                size(DevicePixels(4), DevicePixels(1)),
                [255, 0, 0, 128, 0, 0, 255, 255, 0, 255, 0, 255, 0, 0, 255, 0]
                    .as_slice()
                    .into(),
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
    let mut right = tile;
    right.bounds.origin.x.0 += 2;
    right.bounds.size.width = DevicePixels(2);
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
        bounds: bounds(13.0, 5.0, 5.0, 4.0),
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

    let full = ContentMask {
        bounds: bounds(0.0, 0.0, 32.0, 32.0),
    };
    scene.insert_primitive(Quad {
        bounds: bounds(21.0, 3.0, 9.0, 3.0),
        content_mask: full,
        background: linear_gradient(
            90.0,
            linear_color_stop(rgba(0xff000080), 0.0),
            linear_color_stop(rgba(0xffff0080), 1.0),
        )
        .color_space(ColorSpace::Oklab),
        ..Quad::default()
    });
    // A 100 px circle with a spread of -45 px keeps its original radius.
    let shape = bounds(20.0, 19.0, 10.0, 10.0);
    let blur_radius = ScaledPixels(2.0);
    scene.insert_primitive(Shadow {
        order: 0,
        blur_radius,
        bounds: shape.dilate(blur_radius * 3.0),
        corner_radii: Corners::default(),
        content_mask: full,
        color: rgba(0xffffffff).into(),
        shape_bounds: shape,
        shape_corner_radii: Corners::all(ScaledPixels(50.0)),
        inset: false.into(),
        pad: 0,
    });
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
    // The cropped sprite mixes opaque green and transparent red, over the red quad.
    let filtered = pixel(15, 6);
    assert!(
        filtered
            .iter()
            .zip([64, 128, 0, 192])
            .all(|(&got, want)| got.abs_diff(want) <= 1),
        "{filtered:?}"
    );
    assert_eq!(pixel(4, 24), [0, 255, 0, 255]);
    let gradient = pixel(25, 4);
    assert_eq!([gradient[0], gradient[3]], [128, 128]);
    assert!(pixel(24, 23)[3] > 220, "{:?}", pixel(24, 23));
    assert_clean();
}
