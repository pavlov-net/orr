//! Raster sets: exact round trips, the border and LOD rules, reductions, producer tiles
//! against the cook's own tiling, BCn layers and named errors.

use std::path::{Path, PathBuf};

use orr_cook::{
    ArchiveSpec, AssetSpec, CookOptions, Cooked, Error, Manifest, Outcome, RasterCompression,
    RasterInput, RasterLayerSpec, RasterSet, Reduction, Source,
};
use orr_pack::raster::RasterSetInfo;
use orr_pack::texture::vk;
use orr_pack::{Archive, AssetId, Class, Filter, Target};
use tiff::encoder::{TiffEncoder, colortype};

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A raw raster and its sidecar.
fn write_raw(path: &Path, size: u32, format: &str, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap();
    std::fs::write(
        path.with_extension("json"),
        format!(r#"{{"width": {size}, "height": {size}, "format": "{format}"}}"#),
    )
    .unwrap();
}

fn write_tiff_f32(path: &Path, size: u32, v: &[f32], rows_per_strip: u32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    let mut e = TiffEncoder::new(&mut f).unwrap();
    let mut img = e.new_image::<colortype::Gray32Float>(size, size).unwrap();
    img.rows_per_strip(rows_per_strip).unwrap();
    img.write_data(v).unwrap();
}

fn write_tiff_rgba8(path: &Path, size: u32, v: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    TiffEncoder::new(&mut f)
        .unwrap()
        .write_image::<colortype::RGBA8>(size, size, v)
        .unwrap();
}

fn write_tiff_rgb8(path: &Path, size: u32, v: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    TiffEncoder::new(&mut f)
        .unwrap()
        .write_image::<colortype::RGB8>(size, size, v)
        .unwrap();
}

fn layer(name: &str, input: RasterInput) -> RasterLayerSpec {
    RasterLayerSpec {
        name: name.into(),
        input,
        border: 1,
        reduction: Reduction::Mean,
        srgb: false,
        compress: None,
        quality: Default::default(),
    }
}

fn whole(path: &Path, tile_size: u32) -> RasterInput {
    RasterInput::Whole {
        path: path.to_owned(),
        tile_size,
    }
}

fn tiles(root: &Path) -> RasterInput {
    RasterInput::Tiles {
        root: root.to_owned(),
        pattern: "{lod}/*/0_{lod}_{x}_{y}.tif".into(),
    }
}

fn set(layers: Vec<RasterLayerSpec>) -> RasterSet {
    RasterSet {
        world_min: [-100.5, 250.0],
        world_size: 64.0,
        layers,
    }
}

fn cook_to(target: Target, set: RasterSet) -> Result<Cooked, Error> {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            target,
            vec![AssetSpec {
                name: "terrain".into(),
                source: Source::Raster(set),
            }],
        )],
    };
    orr_cook::cook(&manifest, &CookOptions::default())
}

fn cook(set: RasterSet) -> Archive {
    let mut c = cook_to(Target::Memory, set).unwrap();
    Archive::from_memory(c.memory.remove(0)).unwrap()
}

struct Set {
    a: Archive,
    info: RasterSetInfo,
}

impl Set {
    fn new(a: Archive) -> Self {
        let e = *a
            .expect(
                AssetId::from_name("terrain"),
                Class::RasterSet as u16,
                orr_pack::raster::SCHEMA,
            )
            .unwrap();
        a.verify(&e).unwrap();
        let b = a.blobs(&e)[0];
        let mut head = vec![0; b.raw_len as usize];
        a.read(&b, &mut head).unwrap();
        let info = RasterSetInfo::parse(&head).unwrap();
        Self { a, info }
    }

    fn tile(&self, layer: u16, lod: u8, x: u32, y: u32) -> Vec<u8> {
        let t = self.info.tile(layer, lod, x, y).expect("tile in the set");
        let e = self.a.find(AssetId::from_name("terrain")).unwrap();
        let b = self.a.blobs(e)[t.blob as usize];
        let mut out = vec![0; b.raw_len as usize];
        self.a.read(&b, &mut out).unwrap();
        out
    }

    fn tile_f32(&self, layer: u16, lod: u8, x: u32, y: u32) -> Vec<u32> {
        bytemuck::pod_collect_to_vec(&self.tile(layer, lod, x, y))
    }
}

/// The reference pyramid: finest first, each coarser level by `reduce` over 2x2 blocks.
fn levels<T: Copy>(
    finest: Vec<T>,
    size: u32,
    lods: u32,
    reduce: impl Fn([T; 4]) -> T,
) -> Vec<Vec<T>> {
    let mut out = vec![finest];
    let mut n = size as usize;
    for _ in 1..lods {
        let src = out.last().unwrap();
        let m = n / 2;
        let next = (0..m * m)
            .map(|i| {
                let (x, y) = (2 * (i % m), 2 * (i / m));
                reduce([
                    src[y * n + x],
                    src[y * n + x + 1],
                    src[(y + 1) * n + x],
                    src[(y + 1) * n + x + 1],
                ])
            })
            .collect();
        out.push(next);
        n = m;
    }
    out.reverse();
    out
}

/// The reference tile cut: level texels from `x * tile - border` on, clamped to the level.
fn cut<T: Copy>(level: &[T], size: u32, tile: u32, border: u32, x: u32, y: u32) -> Vec<T> {
    let n = tile + 2 * border;
    let s = i64::from(size);
    let mut out = Vec::with_capacity((n * n) as usize);
    for r in 0..n {
        for c in 0..n {
            let ly = (i64::from(y * tile + r) - i64::from(border)).clamp(0, s - 1);
            let lx = (i64::from(x * tile + c) - i64::from(border)).clamp(0, s - 1);
            out.push(level[(ly * s + lx) as usize]);
        }
    }
    out
}

fn mean_f32(v: [f32; 4]) -> f32 {
    (v[0] + v[1] + v[2] + v[3]) * 0.25
}

fn test_values(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 100_000) as f32 * 0.0137 - 300.0
        })
        .collect()
}

#[test]
fn f32_texels_round_trip_bit_exact() {
    let d = dir("raster-exact");
    let specials: [u32; 8] = [
        0x7fc0_1234, // quiet NaN with a payload
        0x7f80_0001, // signalling NaN
        0xffc0_0042, // negative NaN with a payload
        0x8000_0000, // negative zero
        0x0000_0001, // smallest subnormal
        0x7f80_0000, // infinity
        0xff7f_ffff, // most negative finite
        0x3f80_0000,
    ];
    let mut bits: Vec<u32> = test_values(16 * 16, 7)
        .iter()
        .map(|v| v.to_bits())
        .collect();
    for (i, s) in specials.iter().enumerate() {
        bits[i * 37 % 256] = *s;
    }
    let raw = d.join("height.r32");
    write_raw(&raw, 16, "r32f", bytemuck::cast_slice(&bits));
    let floats: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
    let tif = d.join("height.tif");
    write_tiff_f32(&tif, 16, &floats, 3);
    let tile_tif = d.join("tiles/0/0_0/0_0_0_0.tif");
    write_tiff_f32(&tile_tif, 16, &floats, 16);
    let s = Set::new(cook(set(vec![
        RasterLayerSpec {
            border: 0,
            ..layer("raw", whole(&raw, 8))
        },
        RasterLayerSpec {
            border: 0,
            ..layer("tiff", whole(&tif, 8))
        },
        RasterLayerSpec {
            border: 0,
            ..layer("tile", tiles(&d.join("tiles")))
        },
    ])));
    for layer in [0, 1] {
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let want = cut(&bits, 16, 8, 0, x, y);
            assert_eq!(
                s.tile_f32(layer, 1, x, y),
                want,
                "layer {layer} tile {x},{y}"
            );
        }
    }
    assert_eq!(s.tile_f32(2, 0, 0, 0), bits);
    let range = s.info.layers[0].value_range;
    let finite = floats.iter().filter(|v| v.is_finite());
    assert_eq!(
        range[0],
        finite.clone().fold(f32::INFINITY, |a, &b| a.min(b))
    );
    assert_eq!(range[1], finite.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    assert_eq!(s.info.layers[0].value_range, s.info.layers[2].value_range);
}

#[test]
fn border_rules_and_tiles_match_the_reference_pyramid() {
    let d = dir("raster-border");
    // 3 LODs of 4-texel tiles: 16x16 finest.
    let finest = test_values(16 * 16, 3);
    let raw = d.join("h.r32");
    write_raw(&raw, 16, "r32f", bytemuck::cast_slice(&finest));
    let reference = levels(finest, 16, 3, mean_f32);
    for border in [1u16, 2, 5] {
        let s = Set::new(cook(set(vec![RasterLayerSpec {
            border,
            ..layer("height", whole(&raw, 4))
        }])));
        let l = s.info.layers[0];
        assert_eq!((l.tile_size, l.lod_count, l.border), (4, 3, border));
        assert_eq!(l.vk_format, vk::R32_SFLOAT);
        assert_eq!(s.info.tiles.len(), 1 + 4 + 16);
        for lod in 0..3u8 {
            let size = 4u32 << lod;
            for y in 0..1u32 << lod {
                for x in 0..1u32 << lod {
                    let want: Vec<u32> =
                        cut(&reference[lod as usize], size, 4, border.into(), x, y)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect();
                    assert_eq!(
                        s.tile_f32(0, lod, x, y),
                        want,
                        "border {border} lod {lod} ({x},{y})"
                    );
                }
            }
        }
    }
}

#[test]
fn producer_tiles_match_the_cooks_tiling() {
    let d = dir("raster-producer");
    let finest = test_values(8 * 8, 11);
    let raw = d.join("h.r32");
    write_raw(&raw, 8, "r32f", bytemuck::cast_slice(&finest));
    let colour: Vec<u8> = (0..8 * 8 * 4).map(|i| (i * 37 % 251) as u8).collect();
    let craw = d.join("c.rgba");
    write_raw(&craw, 8, "rgba8", &colour);
    // The producer side: the same border and LOD rules, written as TIFF tiles.
    let h = levels(finest, 8, 2, mean_f32);
    let px: Vec<[u8; 4]> = colour
        .chunks_exact(4)
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    let c = levels(px, 8, 2, |v| v[0]);
    for lod in 0..2u32 {
        for y in 0..1 << lod {
            for x in 0..1 << lod {
                let name = format!("{lod}/{}_{}/0_{lod}_{x}_{y}.tif", x / 8, y / 8);
                let t = cut(&h[lod as usize], 4 << lod, 4, 1, x, y);
                write_tiff_f32(&d.join("height").join(&name), 6, &t, 2);
                let t = cut(&c[lod as usize], 4 << lod, 4, 1, x, y);
                write_tiff_rgba8(&d.join("colour").join(&name), 6, t.as_flattened());
            }
        }
    }
    let cooked = Set::new(cook(set(vec![
        layer("height", whole(&raw, 4)),
        RasterLayerSpec {
            reduction: Reduction::Nearest,
            ..layer("colour", whole(&craw, 4))
        },
    ])));
    let imported = Set::new(cook(set(vec![
        layer("height", tiles(&d.join("height"))),
        RasterLayerSpec {
            reduction: Reduction::Nearest,
            ..layer("colour", tiles(&d.join("colour")))
        },
    ])));
    assert_eq!(cooked.info.layers, imported.info.layers);
    assert_eq!(cooked.info.tiles, imported.info.tiles);
    for t in &cooked.info.tiles {
        assert_eq!(
            cooked.tile(t.layer, t.lod, t.x, t.y),
            imported.tile(t.layer, t.lod, t.x, t.y),
            "layer {} lod {} ({}, {})",
            t.layer,
            t.lod,
            t.x,
            t.y
        );
    }
}

#[test]
fn rgb_tiles_store_rgba_with_opaque_alpha() {
    let d = dir("raster-rgb");
    let rgb: Vec<u8> = (0..4 * 4 * 3).map(|i| (i * 13) as u8).collect();
    write_tiff_rgb8(&d.join("c/0/0_0/0_0_0_0.tif"), 4, &rgb);
    let s = Set::new(cook(set(vec![RasterLayerSpec {
        border: 0,
        srgb: true,
        ..layer("colour", tiles(&d.join("c")))
    }])));
    assert_eq!(s.info.layers[0].vk_format, vk::R8G8B8A8_SRGB);
    let want: Vec<u8> = rgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    assert_eq!(s.tile(0, 0, 0, 0), want);
    let (lo, hi) = (rgb.iter().min().unwrap(), rgb.iter().max().unwrap());
    assert_eq!(
        s.info.layers[0].value_range,
        [f32::from(*lo) / 255.0, f32::from(*hi) / 255.0]
    );
}

#[test]
fn each_reduction_on_a_4x4_raster() {
    let d = dir("raster-reduce");
    #[rustfmt::skip]
    let v: [f32; 16] = [
        1.0, 2.0,   5.0, -0.0,
        3.0, 10.0,  f32::NAN, 2.0,
        -4.0, 8.0,  0.5, 0.25,
        6.0, 2.0,   0.75, 1.5,
    ];
    let raw = d.join("f.r32");
    write_raw(&raw, 4, "r32f", bytemuck::cast_slice(&v));
    let u: [u8; 16] = [1, 2, 200, 0, 4, 10, 255, 255, 0, 0, 7, 9, 1, 0, 3, 3];
    let uraw = d.join("u.r8");
    write_raw(&uraw, 4, "r8", &u);
    let cases: [(Reduction, [f32; 4], [u8; 4]); 4] = [
        (Reduction::Mean, [4.0, f32::NAN, 3.0, 0.75], [4, 178, 0, 6]),
        (Reduction::Min, [1.0, -0.0, -4.0, 0.25], [1, 0, 0, 3]),
        (Reduction::Max, [10.0, 5.0, 8.0, 1.5], [10, 255, 1, 9]),
        (Reduction::Nearest, [1.0, 5.0, -4.0, 0.5], [1, 200, 0, 7]),
    ];
    for (r, want_f, want_u) in cases {
        let s = Set::new(cook(set(vec![
            RasterLayerSpec {
                border: 0,
                reduction: r,
                ..layer("f", whole(&raw, 2))
            },
            RasterLayerSpec {
                border: 0,
                reduction: r,
                ..layer("u", whole(&uraw, 2))
            },
        ])));
        let got: Vec<u32> = s.tile_f32(0, 0, 0, 0);
        for (g, w) in got.iter().zip(want_f) {
            let g = f32::from_bits(*g);
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "{r:?}: {g} vs {w}"
            );
        }
        assert_eq!(s.tile(1, 0, 0, 0), want_u, "{r:?}");
    }
}

#[test]
fn srgb_mean_averages_linear_light() {
    let d = dir("raster-srgb");
    let px: [[u8; 4]; 4] = [
        [0, 0, 0, 0],
        [255, 255, 255, 255],
        [0, 0, 0, 0],
        [255, 255, 255, 255],
    ];
    let raw = d.join("c.rgba");
    write_raw(&raw, 2, "rgba8", px.as_flattened());
    let s = Set::new(cook(set(vec![RasterLayerSpec {
        border: 0,
        srgb: true,
        ..layer("c", whole(&raw, 1))
    }])));
    // Half of white in linear light is sRGB 188; alpha averages to 128.
    assert_eq!(s.tile(0, 0, 0, 0), [188, 188, 188, 128]);
}

#[test]
fn lossless_tiles_take_byte_planes_where_they_shrink() {
    let d = dir("raster-planes");
    // A smooth slope: the high bytes of neighbouring floats repeat.
    let v: Vec<f32> = (0..256 * 256)
        .map(|i| 100.0 + (i % 256) as f32 * 0.37 + (i / 256) as f32 * 0.11)
        .collect();
    let raw = d.join("h.r32");
    write_raw(&raw, 256, "r32f", bytemuck::cast_slice(&v));
    let c = cook_to(Target::Memory, set(vec![layer("height", whole(&raw, 128))])).unwrap();
    let r = c.report.assets[0].entries[0].raster.clone().unwrap();
    let l = &r.layers[0];
    assert_eq!(l.tiles, 5);
    let planes = l.byte_planes.unwrap();
    assert_eq!(l.stored, l.zstd.min(planes).min(l.stored));
    assert!(l.byte_plane_tiles > 0 && planes < l.zstd, "{l:?}");
    let a = Archive::from_memory(std::sync::Arc::clone(&c.memory[0])).unwrap();
    let e = a.find(AssetId::from_name("terrain")).unwrap();
    let planed = a.blobs(e)[1..]
        .iter()
        .filter(|b| b.filter() == Filter::BytePlanes4)
        .count();
    assert_eq!(planed, l.byte_plane_tiles);
    let s = Set::new(a);
    let want: Vec<u32> = cut(&v, 256, 128, 1, 1, 0)
        .iter()
        .map(|f| f.to_bits())
        .collect();
    assert_eq!(s.tile_f32(0, 1, 1, 0), want);
}

#[test]
fn bc4_layer_encodes_through_the_texture_path() {
    let d = dir("raster-bc");
    let v: Vec<u8> = (0..16 * 16).map(|i| ((i % 16) * 16) as u8).collect();
    let raw = d.join("g.r8");
    write_raw(&raw, 16, "r8", &v);
    let s = Set::new(cook(set(vec![RasterLayerSpec {
        border: 2,
        compress: Some(RasterCompression::Bc4),
        ..layer("grass", whole(&raw, 4))
    }])));
    let l = s.info.layers[0];
    assert_eq!((l.vk_format, l.lossless), (vk::BC4_UNORM, 0));
    let blocks = s.tile(0, 2, 1, 1);
    assert_eq!(blocks.len(), 2 * 2 * 8);
    let want = cut(&v, 16, 4, 2, 1, 1);
    let mut texels = [0u8; 16];
    for (b, block) in blocks.chunks_exact(8).enumerate() {
        bcdec_rs::bc4(block, &mut texels, 4, false);
        for (i, t) in texels.iter().enumerate() {
            let (x, y) = ((b % 2) * 4 + i % 4, (b / 2) * 4 + i / 4);
            let w = want[y * 8 + x];
            assert!(t.abs_diff(w) <= 8, "texel ({x},{y}): {t} vs {w}");
        }
    }
}

#[test]
fn header_places_the_set_and_tiles_sort_by_morton_code() {
    let d = dir("raster-header");
    let raw = d.join("h.r32");
    write_raw(
        &raw,
        32,
        "r32f",
        bytemuck::cast_slice(&test_values(32 * 32, 5)),
    );
    let s = Set::new(cook(set(vec![layer("height", whole(&raw, 8))])));
    assert_eq!(s.info.header.world_min, [-100.5, 250.0]);
    assert_eq!(s.info.header.world_size, 64.0);
    assert_eq!(s.info.names, ["height"]);
    let lod2: Vec<(u32, u32)> = s
        .info
        .tiles
        .iter()
        .filter(|t| t.lod == 2)
        .map(|t| (t.x, t.y))
        .collect();
    assert_eq!(
        &lod2[..6],
        &[(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0)]
    );
    for (i, t) in s.info.tiles.iter().enumerate() {
        assert_eq!(t.blob as usize, i + 1);
    }
}

#[test]
fn unchanged_rasters_link_and_a_changed_tile_recooks() {
    let d = dir("raster-incremental");
    let v = test_values(8 * 8, 1);
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        let t = cut(&v, 8, 4, 0, x, y);
        write_tiff_f32(&d.join(format!("t/1/0_0/0_1_{x}_{y}.tif")), 4, &t, 4);
    }
    write_tiff_f32(&d.join("t/0/0_0/0_0_0_0.tif"), 4, &v[..16], 4);
    let out = d.join("terrain.orr");
    let spec = || {
        set(vec![RasterLayerSpec {
            border: 0,
            ..layer("t", tiles(&d.join("t")))
        }])
    };
    let outcome = |c: &Cooked| c.report.assets[0].entries[0].outcome;
    let first = cook_to(Target::File(out.clone()), spec()).unwrap();
    assert_eq!(outcome(&first), Outcome::Cooked);
    let again = cook_to(Target::File(out.clone()), spec()).unwrap();
    assert_eq!(outcome(&again), Outcome::Archive);
    assert!(!again.report.archives[0].written);
    let mut t = cut(&v, 8, 4, 0, 1, 1);
    t[3] += 1.0;
    write_tiff_f32(&d.join("t/1/0_0/0_1_1_1.tif"), 4, &t, 4);
    let changed = cook_to(Target::File(out.clone()), spec()).unwrap();
    assert_eq!(outcome(&changed), Outcome::Cooked);
    let s = Set::new(Archive::open(&out).unwrap());
    assert_eq!(
        s.tile_f32(0, 1, 1, 1),
        t.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn rasters_that_do_not_fit_fail_with_named_errors() {
    let d = dir("raster-errors");
    let raw = d.join("h.r32");
    write_raw(&raw, 12, "r32f", bytemuck::cast_slice(&test_values(144, 2)));
    let err = |set: RasterSet| match cook_to(Target::Memory, set) {
        Err(Error::Raster { message, .. }) => message,
        other => panic!("expected a raster error, got {other:?}"),
    };
    assert!(err(set(vec![layer("h", whole(&raw, 5))])).contains("power of two"));
    assert!(
        err(set(vec![
            layer("h", whole(&raw, 3)),
            layer("h", whole(&raw, 3))
        ]))
        .contains("two layers")
    );
    let m = err(set(vec![RasterLayerSpec {
        compress: Some(RasterCompression::Bc4),
        ..layer("h", whole(&raw, 3))
    }]));
    assert!(m.contains("multiple of 4"), "{m}");
    let m = err(set(vec![RasterLayerSpec {
        srgb: true,
        ..layer("h", whole(&raw, 3))
    }]));
    assert!(m.contains("sRGB"), "{m}");
    let m = err(set(vec![layer("h", tiles(&d))]));
    assert!(m.contains("no tile"), "{m}");
}

#[test]
fn manifest_reads_raster_sources() {
    let d = dir("raster-manifest");
    write_raw(
        &d.join("h.r32"),
        4,
        "r32f",
        bytemuck::cast_slice(&test_values(16, 9)),
    );
    let toml = r#"
[[archive]]
output = "out.orr"

[[archive.asset]]
name = "terrain"
raster = { world_min = [0.0, 0.0], world_size = 4.0, layer = [
    { name = "height", border = 0, input = { whole = { path = "h.r32", tile_size = 2 } } },
] }
"#;
    std::fs::write(d.join("m.toml"), toml).unwrap();
    let m = Manifest::load(&d.join("m.toml")).unwrap();
    orr_cook::cook(&m, &CookOptions::default()).unwrap();
    let s = Set::new(Archive::open(&d.join("out.orr")).unwrap());
    assert_eq!(s.info.tiles.len(), 5);
}
