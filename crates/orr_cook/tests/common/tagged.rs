//! PNG files that state their color.

use std::borrow::Cow;

/// A PNG of `w` x `h` `texels`, with `icc` and `chunks` ahead of its image data.
pub(crate) fn png(
    (color, depth): (png::ColorType, png::BitDepth),
    [w, h]: [u32; 2],
    texels: &[u8],
    icc: Option<&[u8]>,
    chunks: &[(&[u8; 4], &[u8])],
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut info = png::Info::with_size(w, h);
    info.color_type = color;
    info.bit_depth = depth;
    info.icc_profile = icc.map(|p| Cow::Owned(p.to_vec()));
    let mut writer = png::Encoder::with_info(&mut out, info)
        .unwrap()
        .write_header()
        .unwrap();
    for (name, data) in chunks {
        writer
            .write_chunk(png::chunk::ChunkType(**name), data)
            .unwrap();
    }
    writer.write_image_data(texels).unwrap();
    writer.finish().unwrap();
    out
}

pub(crate) const RGB8: (png::ColorType, png::BitDepth) =
    (png::ColorType::Rgb, png::BitDepth::Eight);

/// A flat 8 x 8 8-bit RGB PNG.
pub(crate) fn flat_rgb(rgb: [u8; 3], chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    png(RGB8, [8, 8], &rgb.repeat(64), None, chunks)
}

/// An 8 x 8 8-bit RGB PNG whose texels all differ.
pub(crate) fn varying_rgb(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let texels: Vec<u8> = (0..64u8).flat_map(|i| [i * 4, 255 - i * 3, i]).collect();
    png(RGB8, [8, 8], &texels, None, chunks)
}

/// The data of a `gAMA` chunk for file gamma `gamma`.
pub(crate) fn gama(gamma: f64) -> [u8; 4] {
    ((gamma * 100_000.0).round() as u32).to_be_bytes()
}
