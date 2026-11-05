//! Reading cooked textures back, and writing the GLB files that make them.

use std::path::Path;

use orr_pack::texture::{FormatInfo, TILE_BYTES, TextureInfo, tile_grid, vk};
use orr_pack::{Archive, AssetId, Class};

pub(crate) fn read_blob(a: &Archive, e: &orr_pack::Entry, i: usize) -> Vec<u8> {
    let b = &a.blobs(e)[i];
    let mut out = vec![0; b.raw_len as usize];
    a.read(b, &mut out).unwrap();
    out
}

/// Header and every mip of every layer, tight, from the archive.
pub(crate) fn mips(a: &Archive, name: &str) -> (TextureInfo, Vec<Vec<Vec<u8>>>) {
    let e = *a
        .expect(
            AssetId::from_name(name),
            Class::Texture as u16,
            orr_pack::texture::SCHEMA,
        )
        .unwrap();
    a.verify(&e).unwrap();
    let info = TextureInfo::parse(&read_blob(a, &e, 0)).unwrap();
    let h = info.header;
    assert_eq!(a.blobs(&e).len(), h.blob_count());
    for (i, len) in info.blob_lens().into_iter().enumerate() {
        let b = &a.blobs(&e)[i + 1];
        assert_eq!(b.raw_len as usize, len);
        let tail = i == h.blob_count() - 2 && h.tail_first_mip < h.mip_count;
        if tail {
            assert!(b.is_mount(), "the mip tail is resident from mount");
        } else {
            // One 64 KiB tile per chunk.
            assert!(!b.is_mount());
            let [gx, gy] = tile_grid(&h, i as u32);
            assert_eq!(u32::from(b.chunk_count), gx * gy * u32::from(h.layers));
            assert_eq!(b.chunk_raw_len() as usize, TILE_BYTES);
        }
    }
    let levels = (0..u32::from(h.layers))
        .map(|l| {
            (0..u32::from(h.mip_count))
                .map(|m| {
                    let blob = read_blob(a, &e, h.mip_blob(m));
                    info.mip_texels(&blob, m, l).unwrap()
                })
                .collect()
        })
        .collect();
    (info, levels)
}

pub(crate) fn decode(vk_format: u32, w: u32, h: u32, data: &[u8]) -> Vec<[f32; 4]> {
    let snorm = |v: u8| (f32::from(v.cast_signed()) / 127.0).max(-1.0);
    let f = FormatInfo::of(vk_format).unwrap();
    let bw = w.div_ceil(4) as usize;
    let mut out = vec![[0.0; 4]; (w * h) as usize];
    for (bi, block) in data.chunks_exact(f.bytes as usize).enumerate() {
        let mut t = [[0.0f32; 4]; 16];
        let mut b8 = [0u8; 64];
        match vk_format {
            vk::BC7_SRGB | vk::BC7_UNORM => {
                bcdec_rs::bc7(block, &mut b8, 16);
                for i in 0..16 {
                    t[i] = std::array::from_fn(|c| f32::from(b8[i * 4 + c]) / 255.0);
                }
            }
            vk::BC4_UNORM => {
                bcdec_rs::bc4(block, &mut b8[..16], 4, false);
                for i in 0..16 {
                    t[i][0] = f32::from(b8[i]) / 255.0;
                }
            }
            vk::BC5_UNORM => {
                bcdec_rs::bc5(block, &mut b8[..32], 8, false);
                for i in 0..16 {
                    t[i][0] = f32::from(b8[i * 2]) / 255.0;
                    t[i][1] = f32::from(b8[i * 2 + 1]) / 255.0;
                }
            }
            vk::BC4_SNORM => {
                bcdec_rs::bc4(block, &mut b8[..16], 4, true);
                for i in 0..16 {
                    t[i][0] = snorm(b8[i]);
                }
            }
            vk::BC5_SNORM => {
                bcdec_rs::bc5(block, &mut b8[..32], 8, true);
                for i in 0..16 {
                    t[i][0] = snorm(b8[i * 2]);
                    t[i][1] = snorm(b8[i * 2 + 1]);
                }
            }
            vk::BC6H_UFLOAT => {
                let mut f = [0.0f32; 48];
                bcdec_rs::bc6h_float(block, &mut f, 12, false);
                for i in 0..16 {
                    t[i] = [f[i * 3], f[i * 3 + 1], f[i * 3 + 2], 1.0];
                }
            }
            _ => panic!("format {vk_format}"),
        }
        let (bx, by) = (bi % bw, bi / bw);
        for (i, p) in t.iter().enumerate() {
            let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
            if x < w as usize && y < h as usize {
                out[y * w as usize + x] = *p;
            }
        }
    }
    out
}

/// A GLB holding a quad, the given materials and PNG images.
pub(crate) fn write_glb(path: &Path, materials: &[String], images: &[Vec<u8>]) {
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut view = |bin: &mut Vec<u8>, bytes: &[u8]| {
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            bin.len(),
            bytes.len()
        ));
        bin.extend_from_slice(bytes);
        bin.resize(bin.len().next_multiple_of(4), 0);
        views.len() - 1
    };
    let pos: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let nrm = [[0.0f32, 0.0, 1.0]; 4];
    let uv: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let idx: [u32; 6] = [0, 1, 2, 0, 2, 3];
    let vp = view(&mut bin, bytemuck::cast_slice(&pos));
    let vn = view(&mut bin, bytemuck::cast_slice(&nrm));
    let vu = view(&mut bin, bytemuck::cast_slice(&uv));
    let vi = view(&mut bin, bytemuck::cast_slice(&idx));
    let image_views: Vec<usize> = images.iter().map(|i| view(&mut bin, i)).collect();
    let accessors = format!(
        r#"[{{"bufferView":{vp},"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
            {{"bufferView":{vn},"componentType":5126,"count":4,"type":"VEC3"}},
            {{"bufferView":{vu},"componentType":5126,"count":4,"type":"VEC2"}},
            {{"bufferView":{vi},"componentType":5125,"count":6,"type":"SCALAR"}}]"#
    );
    let prims: Vec<String> = (0..materials.len())
        .map(|m| {
            format!(
                r#"{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"material":{m}}}"#
            )
        })
        .collect();
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
            "meshes":[{{"primitives":[{}]}}],"materials":[{}],
            "textures":[{}],"images":[{}],"samplers":[{{}}],
            "accessors":{accessors},"bufferViews":[{}],"buffers":[{{"byteLength":{}}}]}}"#,
        prims.join(","),
        materials.join(","),
        (0..images.len())
            .map(|i| format!(r#"{{"source":{i},"sampler":0}}"#))
            .collect::<Vec<_>>()
            .join(","),
        image_views
            .iter()
            .map(|v| format!(r#"{{"bufferView":{v},"mimeType":"image/png"}}"#))
            .collect::<Vec<_>>()
            .join(","),
        views.join(","),
        bin.len()
    );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);
    std::fs::write(path, glb).unwrap();
}
