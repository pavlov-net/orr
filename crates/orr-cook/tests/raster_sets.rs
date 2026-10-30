#![allow(clippy::print_stdout)] // the test reports measured figures

//! Cooks an external tiled pyramid; ignored unless asked for, since the set lives outside
//! the repository.
//!
//! `ORR_RASTER_SET=<dir>`: one subdirectory per layer, its TIFF tiles matching
//! `ORR_RASTER_PATTERN` (default `{lod}/*/0_{lod}_{x}_{y}.tif`; file names end in
//! `_<lod>_<x>_<y>.tif`) with `ORR_RASTER_BORDER` border texels (default 1). Every layer is imported into one archive, and every tile
//! must decode to its TIFF's texels bit for bit. Each `layer:reduction` of
//! `ORR_RASTER_RETILE` (e.g. `height:mean,surface:nearest`) is then rebuilt by the cook from
//! the whole raster its finest tiles make, and every tile must equal the imported one.
//! Prints stored bytes per layer with plain zstd and with byte planes.
//!
//! `cargo test --release -p orr-cook --test raster_sets -- --ignored --nocapture`

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use orr_cook::{
    ArchiveSpec, AssetSpec, CookOptions, Manifest, RasterInput, RasterLayerSpec, RasterReport,
    RasterSet, Reduction, Source,
};
use orr_pack::raster::RasterSetInfo;
use orr_pack::{Archive, AssetId, Class, Target};
use rayon::prelude::*;
use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};

const PATTERN: &str = "{lod}/*/0_{lod}_{x}_{y}.tif";

struct Set {
    a: Archive,
    info: RasterSetInfo,
    blobs: Vec<orr_pack::Blob>,
}

impl Set {
    fn open(path: &Path, name: &str) -> Self {
        let a = Archive::open(path).unwrap();
        let e = *a
            .expect(
                AssetId::from_name(name),
                Class::RasterSet as u16,
                orr_pack::raster::SCHEMA,
            )
            .unwrap();
        let blobs = a.blobs(&e).to_vec();
        let mut head = vec![0; blobs[0].raw_len as usize];
        a.read(&blobs[0], &mut head).unwrap();
        let info = RasterSetInfo::parse(&head).unwrap();
        Self { a, info, blobs }
    }

    fn tile(&self, layer: u16, lod: u8, x: u32, y: u32) -> Vec<u8> {
        let t = self.info.tile(layer, lod, x, y).unwrap();
        let b = &self.blobs[t.blob as usize];
        let mut out = vec![0; b.raw_len as usize];
        self.a.read(b, &mut out).unwrap();
        out
    }
}

/// Texels of a TIFF tile as the cook stores them (RGB gains an opaque alpha), its size and
/// the sidecar name of its format.
fn tiff_texels(path: &Path) -> (Vec<u8>, u32, &'static str) {
    let mut d = Decoder::new(std::fs::File::open(path).unwrap()).unwrap();
    let (w, _) = d.dimensions().unwrap();
    let ct = d.colortype().unwrap();
    let bytes = match d.read_image().unwrap() {
        DecodingResult::U8(v) => v,
        DecodingResult::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DecodingResult::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        _ => panic!("{}: sample type", path.display()),
    };
    match ct {
        ColorType::RGB(8) => (
            bytes
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
            w,
            "rgb8",
        ),
        ColorType::RGBA(8) => (bytes, w, "rgba8"),
        ColorType::GrayA(8) => (bytes, w, "rg8"),
        ColorType::Gray(8) => (bytes, w, "r8"),
        ColorType::Gray(16) => (bytes, w, "r16"),
        ColorType::Gray(32) => (bytes, w, "r32f"),
        c => panic!("{}: colour type {c:?}", path.display()),
    }
}

/// Every TIFF under `root` by the `<lod>_<x>_<y>` that ends its file name.
fn tile_index(root: &Path) -> HashMap<(u32, u32, u32), PathBuf> {
    let mut out = HashMap::new();
    let mut stack = vec![root.to_owned()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
            let n: Vec<u32> = stem
                .rsplit('_')
                .take(3)
                .filter_map(|v| v.parse().ok())
                .collect();
            if let [y, x, lod] = n[..] {
                out.insert((lod, x, y), p);
            }
        }
    }
    out
}

fn cook(out: &Path, cache: &Path, name: &str, layers: Vec<RasterLayerSpec>) -> RasterReport {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::File(out.to_owned()),
            vec![AssetSpec {
                name: name.into(),
                source: Source::Raster(RasterSet {
                    world_min: [0.0, 0.0],
                    world_size: 1.0,
                    layers,
                }),
            }],
        )],
    };
    let start = std::time::Instant::now();
    let cooked = orr_cook::cook(
        &manifest,
        &CookOptions {
            force: true,
            cache: Some(cache.to_owned()),
            ..CookOptions::default()
        },
    )
    .unwrap();
    println!(
        "{name}: cooked in {:.1} s, {} B",
        start.elapsed().as_secs_f64(),
        cooked.report.archives[0].bytes
    );
    cooked.report.assets[0].entries[0].raster.clone().unwrap()
}

#[test]
#[ignore = "needs ORR_RASTER_SET"]
fn external_pyramid_cooks_exactly() {
    let root = PathBuf::from(std::env::var("ORR_RASTER_SET").expect("ORR_RASTER_SET"));
    let pattern = std::env::var("ORR_RASTER_PATTERN").unwrap_or_else(|_| PATTERN.into());
    let border: u16 = std::env::var("ORR_RASTER_BORDER").map_or(1, |b| b.parse().unwrap());
    let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("raster-set");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let cache = work.join("cache");
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(|e| {
            let e = e.unwrap();
            e.file_type()
                .unwrap()
                .is_dir()
                .then(|| e.file_name().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    let spec = |name: &str, input: RasterInput, reduction: Reduction| RasterLayerSpec {
        name: name.into(),
        input,
        border,
        reduction,
        srgb: false,
        compress: None,
        quality: Default::default(),
    };
    let imported_path = work.join("imported.orr");
    let report = cook(
        &imported_path,
        &cache,
        "imported",
        names
            .iter()
            .map(|n| {
                spec(
                    n,
                    RasterInput::Tiles {
                        root: root.join(n),
                        pattern: pattern.clone(),
                    },
                    Reduction::Mean,
                )
            })
            .collect(),
    );
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "{:<16} {:>6} {:>14} {:>14} {:>14} {:>14} {:>7}",
        "layer", "tiles", "raw", "stored", "zstd", "byte planes", "planed"
    )
    .unwrap();
    for l in &report.layers {
        writeln!(
            out,
            "{:<16} {:>6} {:>14} {:>14} {:>14} {:>14} {:>7}",
            l.name,
            l.tiles,
            l.raw,
            l.stored,
            l.zstd,
            l.byte_planes.map_or("-".into(), |b| b.to_string()),
            l.byte_plane_tiles
        )
        .unwrap();
    }
    drop(out);

    let imported = Set::open(&imported_path, "imported");
    let index: Vec<_> = names.iter().map(|n| tile_index(&root.join(n))).collect();
    let bad: usize = imported
        .info
        .tiles
        .par_iter()
        .filter(|t| {
            let path = &index[usize::from(t.layer)][&(u32::from(t.lod), t.x, t.y)];
            let (want, ..) = tiff_texels(path);
            imported.tile(t.layer, t.lod, t.x, t.y) != want
        })
        .count();
    println!(
        "{} tiles imported, {bad} differ from their TIFFs",
        imported.info.tiles.len()
    );
    assert_eq!(bad, 0);

    let retile = std::env::var("ORR_RASTER_RETILE").unwrap_or_default();
    for item in retile.split(',').filter(|s| !s.is_empty()) {
        let (name, reduction) = item.split_once(':').expect("layer:reduction");
        let reduction = match reduction {
            "mean" => Reduction::Mean,
            "min" => Reduction::Min,
            "max" => Reduction::Max,
            "nearest" => Reduction::Nearest,
            r => panic!("unknown reduction {r}"),
        };
        let li = imported.info.layer(name).expect("layer in the set");
        let l = imported.info.layers[li];
        let finest = u32::from(l.lod_count) - 1;
        let n = 1u32 << finest;
        let ts = l.tile_size as usize;
        let b = usize::from(l.border);
        // The whole raster from the centres of the finest tiles, one tile row at a time.
        let raw = work.join(format!("{name}.raw"));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&raw).unwrap());
        let mut format = "";
        for ty in 0..n {
            let row: Vec<(Vec<u8>, u32, &str)> = (0..n)
                .into_par_iter()
                .map(|tx| tiff_texels(&index[li][&(finest, tx, ty)]))
                .collect();
            format = row[0].2;
            let bpp = row[0].0.len() / (row[0].1 as usize).pow(2);
            let stride = row[0].1 as usize * bpp;
            for r in 0..ts {
                for (t, ..) in &row {
                    file.write_all(&t[(r + b) * stride + b * bpp..][..ts * bpp])
                        .unwrap();
                }
            }
        }
        file.flush().unwrap();
        drop(file);
        let size = n as usize * ts;
        let format = if format == "rgb8" { "rgba8" } else { format };
        std::fs::write(
            raw.with_extension("json"),
            format!(r#"{{"width": {size}, "height": {size}, "format": "{format}"}}"#),
        )
        .unwrap();
        let whole_path = work.join(format!("{name}.orr"));
        cook(
            &whole_path,
            &cache,
            name,
            vec![spec(
                name,
                RasterInput::Whole {
                    path: raw.clone(),
                    tile_size: l.tile_size,
                },
                reduction,
            )],
        );
        let whole = Set::open(&whole_path, name);
        let mut per_lod = vec![0usize; usize::from(l.lod_count)];
        for t in &whole.info.tiles {
            if whole.tile(0, t.lod, t.x, t.y) != imported.tile(li as u16, t.lod, t.x, t.y) {
                per_lod[usize::from(t.lod)] += 1;
            }
        }
        println!(
            "{name} ({reduction:?}) tiled from its {size}^2 raster: tiles differing per LOD {per_lod:?}"
        );
        assert_eq!(per_lod.iter().sum::<usize>(), 0);
        let _ = std::fs::remove_file(&raw);
    }
}
