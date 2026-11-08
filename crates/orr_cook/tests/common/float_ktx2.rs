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

/// A KTX2 whose key/value data starts with `KTXswizzle` set to `swizzle`, with a padding entry
/// that keeps the levels' alignment.
pub(crate) fn with_swizzle(file: &[u8], swizzle: &str) -> Vec<u8> {
    let entry = |key: &str, value: &[u8]| {
        let mut e = (key.len() as u32 + 1 + value.len() as u32)
            .to_le_bytes()
            .to_vec();
        e.extend_from_slice(key.as_bytes());
        e.push(0);
        e.extend_from_slice(value);
        e.resize(e.len().next_multiple_of(4), 0);
        e
    };
    let mut added = entry("KTXswizzle", format!("{swizzle}\0").as_bytes());
    let padding = added.len().next_multiple_of(16) - added.len();
    if padding >= 8 {
        added.extend(entry("pad", &vec![0; padding - 8]));
    } else if padding > 0 {
        added.extend(entry("padding", &vec![0; padding + 8]));
    }
    let word = |at: usize| u32::from_le_bytes(file[at..at + 4].try_into().unwrap());
    let long = |at: usize| u64::from_le_bytes(file[at..at + 8].try_into().unwrap());
    let (dfd, dfd_len) = (word(48), word(52));
    let (kvd, kvd_len) = (word(56), word(60));
    let at = if kvd_len > 0 { kvd } else { dfd + dfd_len } as usize;
    let levels = word(40).max(1) as usize;
    let mut out = file[..at].to_vec();
    out.extend_from_slice(&added);
    out.extend_from_slice(&file[at..]);
    out[56..60].copy_from_slice(&(at as u32).to_le_bytes());
    out[60..64].copy_from_slice(&(kvd_len + added.len() as u32).to_le_bytes());
    for l in 0..levels {
        let offset = 80 + 24 * l;
        let moved = long(offset) + added.len() as u64;
        out[offset..offset + 8].copy_from_slice(&moved.to_le_bytes());
    }
    let sgd_len = long(72);
    if sgd_len > 0 {
        let moved = long(64) + added.len() as u64;
        out[64..72].copy_from_slice(&moved.to_le_bytes());
    }
    out
}
