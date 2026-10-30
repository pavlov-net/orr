use orr_pack::Span;
use orr_pack::texture::{
    FormatInfo, TILE_BYTES, TextureHeader, TextureInfo, octahedral, tail_first_mip, tile,
    tile_shape, untile, vk,
};

/// Standard 2D sparse block shapes reported by the device (measured.md SP-8).
#[test]
fn tile_shapes_and_tails_match_the_device() {
    let shapes = [
        (vk::BC7_SRGB, [256, 256]),
        (vk::BC7_UNORM, [256, 256]),
        (vk::BC6H_UFLOAT, [256, 256]),
        (vk::BC5_UNORM, [256, 256]),
        (vk::BC4_UNORM, [512, 256]),
        (vk::R8_UNORM, [256, 256]),
        (vk::R16_UNORM, [256, 128]),
        (vk::R8G8B8A8_UNORM, [128, 128]),
        (vk::R16G16B16A16_SFLOAT, [128, 64]),
        (vk::R32G32B32A32_SFLOAT, [64, 64]),
    ];
    for (format, shape) in shapes {
        let f = FormatInfo::of(format).unwrap();
        assert_eq!(tile_shape(f), shape, "format {format}");
        let [w, h] = shape;
        assert_eq!(f.mip_bytes(w, h), TILE_BYTES, "format {format}");
    }
    // Tail: the first mip smaller than one tile in either axis.
    let bc7 = FormatInfo::of(vk::BC7_UNORM).unwrap();
    let bc4 = FormatInfo::of(vk::BC4_UNORM).unwrap();
    let r16 = FormatInfo::of(vk::R16_UNORM).unwrap();
    assert_eq!(tail_first_mip(bc7, 4096, 4096, 13), 5);
    assert_eq!(tail_first_mip(bc7, 1300, 700, 11), 2);
    assert_eq!(tail_first_mip(bc7, 2600, 300, 12), 1);
    assert_eq!(tail_first_mip(bc4, 4096, 4096, 13), 4);
    assert_eq!(tail_first_mip(r16, 300, 5000, 13), 1);
    assert_eq!(tail_first_mip(bc7, 200, 200, 8), 0);
}

#[test]
fn tiles_round_trip_with_padded_edges() {
    for (format, w, h) in [
        (vk::BC7_UNORM, 1300, 700),
        (vk::BC4_UNORM, 700, 1300),
        (vk::R16_UNORM, 300, 900),
        (vk::R8G8B8A8_SRGB, 129, 257),
    ] {
        let f = FormatInfo::of(format).unwrap();
        let tight: Vec<u8> = (0..f.mip_bytes(w, h))
            .map(|i| (i * 7 + i / 251) as u8)
            .collect();
        let mut tiles = Vec::new();
        tile(f, w, h, &tight, &mut tiles);
        let [tw, th] = tile_shape(f);
        let grid = (w.div_ceil(tw) * h.div_ceil(th)) as usize;
        assert_eq!(tiles.len(), grid * TILE_BYTES);
        assert_eq!(untile(f, w, h, &tiles), tight, "format {format} {w}x{h}");
        // The bottom-right tile is zero past the mip's last block row and column.
        let last = &tiles[(grid - 1) * TILE_BYTES..];
        let (bw, bh) = ((tw / f.block) as usize, (th / f.block) as usize);
        let used_x = (w.div_ceil(f.block) as usize - 1) % bw + 1;
        let used_y = (h.div_ceil(f.block) as usize - 1) % bh + 1;
        let bpb = f.bytes as usize;
        for y in 0..bh {
            let row = &last[y * bw * bpb..][..bw * bpb];
            let pad = if y < used_y {
                &row[used_x * bpb..]
            } else {
                row
            };
            assert!(pad.iter().all(|&b| b == 0), "format {format} row {y}");
        }
    }
}

#[test]
fn header_rejects_a_wrong_tail() {
    let f = FormatInfo::of(vk::BC7_SRGB).unwrap();
    let mut header = TextureHeader {
        vk_format: vk::BC7_SRGB,
        width: 1024,
        height: 512,
        layers: 1,
        kind: TextureHeader::D2,
        mip_count: 11,
        tail_first_mip: tail_first_mip(f, 1024, 512, 11) as u8,
        primaries: TextureHeader::REC709,
        _pad: [0; 2],
        alpha_cutoff: f32::NAN,
        lights: Span::default(),
    };
    let info = TextureInfo {
        header,
        lights: vec![],
    };
    let parsed = TextureInfo::parse(&info.to_bytes()).unwrap();
    assert_eq!(parsed.header.tail_first_mip, 2);
    assert_eq!(parsed.blob_lens().len(), 3);
    header.tail_first_mip = 3;
    let bad = TextureInfo {
        header,
        lights: vec![],
    };
    assert!(TextureInfo::parse(&bad.to_bytes()).is_err());
}

#[test]
fn octahedral_map_is_equal_area_and_invertible() {
    let n = 64;
    // Every texel's solid angle, summed over flat triangles between mapped sub-texel corners,
    // equals 4 pi / n^2.
    let sub = 16;
    let d = |s: f64, t: f64| octahedral::to_dir([s as f32, t as f32]).map(f64::from);
    let tri = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
        let u = [0, 1, 2].map(|k| b[k] - a[k]);
        let v = [0, 1, 2].map(|k| c[k] - a[k]);
        let x = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
    };
    let mut areas = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let mut a = 0.0f64;
            let at = |i: u32, j: u32| {
                d(
                    (f64::from(x) + f64::from(i) / f64::from(sub)) / f64::from(n),
                    (f64::from(y) + f64::from(j) / f64::from(sub)) / f64::from(n),
                )
            };
            for j in 0..sub {
                for i in 0..sub {
                    let (p00, p10, p01, p11) =
                        (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
                    a += tri(p00, p10, p11) + tri(p00, p11, p01);
                }
            }
            areas.push(a);
        }
    }
    let want = 4.0 * std::f64::consts::PI / f64::from(n * n);
    for (i, a) in areas.iter().enumerate() {
        assert!(
            (a / want - 1.0).abs() < 0.02,
            "texel {i}: {a} sr against {want}"
        );
    }
    for y in 0..n {
        for x in 0..n {
            let st = [(x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32];
            let d = octahedral::to_dir(st);
            let back = octahedral::from_dir(d);
            assert!((back[0] - st[0]).abs() < 1e-5 && (back[1] - st[1]).abs() < 1e-5);
        }
    }
    assert_eq!(octahedral::to_dir([0.5, 0.5]), [0.0, 1.0, 0.0]);
}
