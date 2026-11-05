//! KTX2 files of values PNG can't hold.

/// A KTX2 of `w` x `h` float texels `f(x, y)` stored as `format`, with every mip if `mipmap`.
pub(crate) fn float_ktx2(
    w: u32,
    h: u32,
    format: ctt::TargetFormat,
    mipmap: bool,
    f: impl Fn(u32, u32) -> [f32; 4],
) -> Vec<u8> {
    let texels: Vec<u8> = (0..w * h)
        .flat_map(|i| f(i % w, i / w))
        .flat_map(f32::to_le_bytes)
        .collect();
    let image = ctt::ImageRef {
        surfaces: vec![vec![ctt::SurfaceRef {
            data: &texels,
            width: w,
            height: h,
            depth: 1,
            stride: w * 16,
            slice_stride: 0,
        }]],
        kind: ctt::TextureKind::Texture2D,
        desc: ctt::FormatDesc {
            format: ctt::Format::R32G32B32A32_SFLOAT,
            color_space: ctt::ColorSpace::Linear,
            alpha: ctt::AlphaMode::Opaque,
        },
    };
    let ctt::PipelineOutput::Encoded(bytes) = ctt::convert(
        image,
        ctt::ConvertSettings {
            format: Some(format),
            container: ctt::Container::ktx2(),
            mipmap,
            ..Default::default()
        },
    )
    .unwrap() else {
        panic!("container output");
    };
    bytes
}
