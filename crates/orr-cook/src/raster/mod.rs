//! Raster set cook: whole rasters tiled into bordered LOD pyramids by the cook, or tiles a
//! producer already cut, into one raster set entry (`orr_pack::raster`).
//!
//! A whole raster is read once in row bands; each coarser level is reduced from the finer
//! one as rows arrive and written to a scratch file (in the cache directory, or the system
//! temporary directory without a cache), so memory stays bounded by the band and the
//! tiles in flight. Tiles are then cut from the level files in table order and encoded in
//! parallel on the cook's pool.

mod input;
mod texel;

use std::path::{Path, PathBuf};

use orr_pack::raster::{self, RasterLayer, RasterSetHeader, RasterSetInfo, TileKey, morton};
use orr_pack::{ArchiveWriter, BlobIn, Class, EncodedBlob, Filter, Hash32, Span};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::key::{Key, ZSTD_LEVEL};
use crate::state::{Memo, read_checked};
use crate::texture::{Encoding, Quality, encode_chains};
use crate::{Result, mount};
use input::{Level, Scratch, Sidecar, TilePattern, WholeRaster, raster_error};
pub use texel::{RasterFormat, Reduction};

/// Raw tile bytes encoded per batch; a batch is cut, then encoded in parallel.
const BATCH_BYTES: usize = 64 << 20;

/// The scheduler's estimate of a raster job's peak bytes: the batch, its encodings and the
/// whole-raster band.
pub(crate) const WORKING_SET: u64 = 4 * BATCH_BYTES as u64 + (16 << 20);

/// A raster set source: layers over one square of the world.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RasterSet {
    /// World x and z of the corner at texel `(0, 0)`, in m.
    pub world_min: [f64; 2],
    /// Edge of the square every layer covers, in m.
    pub world_size: f32,
    #[serde(rename = "layer")]
    pub layers: Vec<RasterLayerSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RasterLayerSpec {
    pub name: String,
    pub input: RasterInput,
    /// Texels each tile repeats from its neighbours on every side.
    #[serde(default = "default_border")]
    pub border: u16,
    /// How the cook builds coarser LODs from a whole raster; for producer tiles, how the
    /// producer built them (recorded only).
    #[serde(default)]
    pub reduction: Reduction,
    /// RGB of 8-bit colour layers is sRGB-encoded: stored with an sRGB format, and averaged
    /// in linear light.
    #[serde(default)]
    pub srgb: bool,
    /// Stores tiles as BCn through the texture encoders instead of the exact texels.
    #[serde(default)]
    pub compress: Option<RasterCompression>,
    /// BCn encoder effort.
    #[serde(default)]
    pub quality: Quality,
}

fn default_border() -> u16 {
    1
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RasterInput {
    /// One square raster for the whole layer, `tile_size << (lods - 1)` texels per axis;
    /// the cook builds the coarser LODs. A `.tif`/`.tiff` path is a TIFF; any other path is
    /// raw texels described by the JSON sidecar at the path with extension `json`
    /// (`{"width", "height", "format", "offset"}`, `format` one of the [`RasterFormat`]
    /// names).
    Whole { path: PathBuf, tile_size: u32 },
    /// One TIFF per tile, already bordered: every file under `root` whose relative path
    /// matches `pattern`, `/`-separated components with `{lod}`, `{x}`, `{y}` and `*`
    /// (e.g. `"{lod}/*/0_{lod}_{x}_{y}.tif"`).
    Tiles { root: PathBuf, pattern: String },
}

/// BCn formats for raster layers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RasterCompression {
    /// R.
    Bc4,
    /// RG.
    Bc5,
    /// Unsigned float RGB.
    Bc6h,
    /// Linear RGBA.
    Bc7,
    /// sRGB RGB with linear alpha; needs an `srgb` layer.
    Bc7Srgb,
}

impl RasterCompression {
    fn encoding(self) -> Encoding {
        match self {
            Self::Bc4 => Encoding::Bc4,
            Self::Bc5 => Encoding::Bc5,
            Self::Bc6h => Encoding::Bc6h,
            Self::Bc7 => Encoding::Bc7,
            Self::Bc7Srgb => Encoding::Bc7Srgb,
        }
    }
}

/// What the cook made of one layer.
#[derive(Clone, Debug, Serialize)]
pub struct RasterLayerReport {
    pub name: String,
    pub vk_format: u32,
    pub tile_size: u32,
    pub lods: u8,
    pub tiles: usize,
    /// Decoded bytes of its tiles.
    pub raw: u64,
    /// Stored bytes of its tiles as written.
    pub stored: u64,
    /// Stored bytes with plain zstd and with zstd over byte planes, for lossless layers.
    pub zstd: u64,
    pub byte_planes: Option<u64>,
    /// Tiles stored with byte planes.
    pub byte_plane_tiles: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RasterReport {
    pub layers: Vec<RasterLayerReport>,
}

/// A planned raster set: its inputs resolved and hashed.
#[derive(Debug)]
pub(crate) struct Plan {
    set: RasterSet,
    layers: Vec<LayerInput>,
}

#[derive(Debug)]
enum LayerInput {
    Whole {
        path: PathBuf,
        hash: Hash32,
        sidecar: Option<Sidecar>,
        tile_size: u32,
    },
    /// `(lod, x, y)`, path and hash, in table order.
    Tiles(Vec<([u32; 3], PathBuf, Hash32)>),
}

pub(crate) struct Planned {
    pub plan: Plan,
    pub key: Hash32,
    pub files: Vec<PathBuf>,
}

/// Resolves and hashes the inputs of a raster set and computes its key.
pub(crate) fn plan(asset: &str, set: &RasterSet, memo: &Memo) -> Result<Planned> {
    let fail = |m: String| raster_error(asset, m);
    if set.layers.is_empty() {
        return Err(fail("a raster set needs a layer".into()));
    }
    if set.world_size.is_nan() || set.world_size <= 0.0 {
        return Err(fail("world_size must be positive".into()));
    }
    let mut key = Key::new(Class::RasterSet, asset).debug(&(set.world_min, set.world_size));
    let mut files = Vec::new();
    let mut layers = Vec::new();
    for (i, l) in set.layers.iter().enumerate() {
        if set.layers[..i].iter().any(|o| o.name == l.name) {
            return Err(fail(format!("two layers are named {}", l.name)));
        }
        if let Some(c) = l.compress
            && l.srgb != (c == RasterCompression::Bc7Srgb)
        {
            return Err(fail(format!(
                "layer {}: sRGB layers compress to bc7_srgb, and only they do",
                l.name
            )));
        }
        key = key.debug(&(
            &l.name,
            l.border,
            l.reduction,
            l.srgb,
            l.compress,
            l.quality,
        ));
        let input = match &l.input {
            RasterInput::Whole { path, tile_size } => {
                if *tile_size == 0 {
                    return Err(fail(format!("layer {}: tile_size is 0", l.name)));
                }
                let hash = memo.hash(path)?;
                files.push(path.clone());
                key = key.str("whole").bytes(&tile_size.to_le_bytes()).hash(hash);
                let sidecar = if input::is_tiff(path) {
                    None
                } else {
                    let side = path.with_extension("json");
                    let bytes = std::fs::read(&side).map_err(|source| crate::Error::Read {
                        path: side.clone(),
                        source,
                    })?;
                    let s: Sidecar = serde_json::from_slice(&bytes)
                        .map_err(|e| fail(format!("{}: {e}", side.display())))?;
                    key = key.hash(memo.hash(&side)?);
                    files.push(side);
                    Some(s)
                };
                LayerInput::Whole {
                    path: path.clone(),
                    hash,
                    sidecar,
                    tile_size: *tile_size,
                }
            }
            RasterInput::Tiles { root, pattern } => {
                let pattern = TilePattern::parse(pattern)
                    .map_err(|m| fail(format!("layer {}: {m}", l.name)))?;
                let mut tiles = Vec::new();
                for (path, rel) in input::walk(root)? {
                    let Some([lod, x, y]) = pattern.matches(&rel) else {
                        continue;
                    };
                    if lod >= 24 || x >> lod != 0 || y >> lod != 0 {
                        return Err(fail(format!(
                            "{}: tile ({x}, {y}) is outside LOD {lod}",
                            path.display()
                        )));
                    }
                    tiles.push(([lod, x, y], path));
                }
                if tiles.is_empty() {
                    return Err(fail(format!(
                        "layer {}: no tile under {} matches",
                        l.name,
                        root.display()
                    )));
                }
                tiles.sort_by_key(|(c, _)| (c[0], morton(c[1], c[2])));
                if let Some(w) = tiles.windows(2).find(|w| w[0].0 == w[1].0) {
                    return Err(fail(format!(
                        "{} and {} are the same tile",
                        w[0].1.display(),
                        w[1].1.display()
                    )));
                }
                key = key.str("tiles");
                let mut hashed = Vec::with_capacity(tiles.len());
                for (c, path) in tiles {
                    let h = memo.hash(&path)?;
                    key = key.bytes(bytemuck::cast_slice(&c)).hash(h);
                    files.push(path.clone());
                    hashed.push((c, path, h));
                }
                LayerInput::Tiles(hashed)
            }
        };
        layers.push(input);
    }
    Ok(Planned {
        plan: Plan {
            set: set.clone(),
            layers,
        },
        key: key.finish(),
        files,
    })
}

/// Where one tile's texels come from.
enum TileSource<'p> {
    /// LOD, x and y in the layer's level files.
    Level(usize, u32, u32),
    File(&'p Path, Hash32),
}

struct Layer<'p> {
    record: RasterLayer,
    spec: &'p RasterLayerSpec,
    format: RasterFormat,
    levels: Vec<Level>,
}

impl Layer<'_> {
    /// Lossless layers of multi-byte texels try `Filter::BytePlanes4` per tile.
    fn tries_byte_planes(&self) -> bool {
        self.record.lossless == 1 && self.format.stored().bytes() > 1
    }
}

/// Cooks a planned raster set into `w` as one entry.
pub(crate) fn cook(
    w: &mut ArchiveWriter,
    asset: &str,
    key: Hash32,
    plan: &Plan,
    scratch_dir: &Path,
) -> Result<RasterReport> {
    let fail = |m: String| raster_error(asset, m);
    let mut scratch = Scratch::new(scratch_dir, key);
    let mut layers = Vec::new();
    let mut tiles: Vec<(TileKey, TileSource)> = Vec::new();
    for (li, (spec, input)) in plan.set.layers.iter().zip(&plan.layers).enumerate() {
        let border = u32::from(spec.border);
        let check = |format: RasterFormat, tile_size: u32| {
            let texels = tile_size + 2 * border;
            if spec.compress.is_some() && !texels.is_multiple_of(4) {
                return Err(fail(format!(
                    "layer {}: BCn tiles need a multiple of 4 texels, not {texels}",
                    spec.name
                )));
            }
            if spec.srgb && format.stored() != texel::Texel::Rgba8 {
                return Err(fail(format!(
                    "layer {}: only RGB and RGBA layers are sRGB",
                    spec.name
                )));
            }
            Ok(())
        };
        let (format, tile_size, lod_count, range, levels) = match input {
            LayerInput::Whole {
                path,
                hash,
                sidecar,
                tile_size,
            } => {
                input::check_hash(path, *hash)?;
                let src = WholeRaster::open(path, sidecar.as_ref())?;
                let ratio = src.size / tile_size;
                if src.size % tile_size != 0 || !ratio.is_power_of_two() || ratio >= 1 << 24 {
                    return Err(fail(format!(
                        "layer {}: {} texels are not tile_size {tile_size} times a power of two",
                        spec.name, src.size
                    )));
                }
                let lods = ratio.trailing_zeros() + 1;
                let format = src.format;
                check(format, *tile_size)?;
                let raw = sidecar.as_ref().map(|s| (path.as_path(), s.offset));
                let built =
                    input::build_levels(src, raw, lods, spec.reduction, spec.srgb, &mut scratch)?;
                for lod in 0..lods {
                    let n = 1 << lod;
                    let mut keys: Vec<(u32, u32)> =
                        (0..n).flat_map(|y| (0..n).map(move |x| (x, y))).collect();
                    keys.sort_by_key(|&(x, y)| morton(x, y));
                    for (x, y) in keys {
                        tiles.push((
                            tile_key(li, lod, x, y),
                            TileSource::Level(lod as usize, x, y),
                        ));
                    }
                }
                (format, *tile_size, lods, built.range, built.levels)
            }
            LayerInput::Tiles(list) => {
                let finest = list
                    .iter()
                    .map(|(c, ..)| c[0])
                    .max()
                    .expect("planned tiles");
                let facts: Vec<([u32; 2], RasterFormat, [f32; 2])> = list
                    .par_iter()
                    .map(|(_, path, hash)| {
                        let bytes = read_checked(path, *hash)?;
                        let t = input::decode_tiff(path, &bytes)?;
                        let mut range = [f32::INFINITY, f32::NEG_INFINITY];
                        t.format.range(&t.texels, &mut range);
                        Ok((t.size, t.format, range))
                    })
                    .collect::<Result<_>>()?;
                let (size, format, _) = facts[0];
                if let Some(((_, path, _), f)) = list
                    .iter()
                    .zip(&facts)
                    .find(|(_, f)| f.0 != size || f.1 != format)
                {
                    return Err(fail(format!(
                        "{}: a {}x{} {:?} tile in a layer of {}x{} {format:?} tiles",
                        path.display(),
                        f.0[0],
                        f.0[1],
                        f.1,
                        size[0],
                        size[1]
                    )));
                }
                if size[0] != size[1] || size[0] <= 2 * border {
                    return Err(fail(format!(
                        "layer {}: {}x{} tiles do not fit border {border}",
                        spec.name, size[0], size[1]
                    )));
                }
                check(format, size[0] - 2 * border)?;
                let range = facts
                    .iter()
                    .fold([f32::INFINITY, f32::NEG_INFINITY], |r, f| {
                        [r[0].min(f.2[0]), r[1].max(f.2[1])]
                    });
                for (c, path, hash) in list {
                    tiles.push((
                        tile_key(li, c[0], c[1], c[2]),
                        TileSource::File(path, *hash),
                    ));
                }
                (format, size[0] - 2 * border, finest + 1, range, Vec::new())
            }
        };
        let vk_format = match spec.compress {
            Some(c) => c.encoding().vk_format(),
            None => format.stored().vk_format(spec.srgb),
        };
        layers.push(Layer {
            record: RasterLayer {
                name: Span::default(),
                vk_format,
                tile_size,
                border: spec.border,
                lod_count: lod_count as u8,
                lossless: u8::from(spec.compress.is_none()),
                reduction: spec.reduction.code(),
                _pad: [0; 3],
                value_range: range,
            },
            spec,
            format,
            levels,
        });
    }
    for (i, (t, _)) in tiles.iter_mut().enumerate() {
        t.blob = i as u32 + 1;
    }
    let info = RasterSetInfo {
        header: RasterSetHeader {
            world_min: plan.set.world_min,
            world_size: plan.set.world_size,
            _pad: 0,
            layers: Span::default(),
            names: Span::default(),
            tiles: Span::default(),
        },
        layers: layers.iter().map(|l| l.record).collect(),
        names: layers.iter().map(|l| l.spec.name.clone()).collect(),
        tiles: tiles.iter().map(|(t, _)| *t).collect(),
    };
    let head = info.to_bytes();
    let mut report: Vec<RasterLayerReport> = layers
        .iter()
        .map(|l| RasterLayerReport {
            name: l.spec.name.clone(),
            vk_format: l.record.vk_format,
            tile_size: l.record.tile_size,
            lods: l.record.lod_count,
            tiles: 0,
            raw: 0,
            stored: 0,
            zstd: 0,
            byte_planes: l.tries_byte_planes().then_some(0),
            byte_plane_tiles: 0,
        })
        .collect();
    w.add_streamed(
        asset,
        Class::RasterSet as u16,
        raster::SCHEMA,
        key,
        |e| -> Result<()> {
            e.push(&mount(&head))?;
            let mut at = 0;
            while at < tiles.len() {
                let mut end = at;
                let mut bytes = 0;
                while end < tiles.len() && (end == at || bytes < BATCH_BYTES) {
                    bytes += layers[usize::from(tiles[end].0.layer)].record.tile_bytes();
                    end += 1;
                }
                let batch = &tiles[at..end];
                let raws: Vec<Vec<u8>> = batch
                    .par_iter()
                    .map(|(t, src)| tile_bytes(asset, &layers[usize::from(t.layer)], src))
                    .collect::<Result<_>>()?;
                let encoded: Vec<Encoded> = raws
                    .par_iter()
                    .zip(batch)
                    .map(|(raw, (t, _))| encode(raw, &layers[usize::from(t.layer)]))
                    .collect::<Result<_>>()?;
                for ((enc, raw), (t, _)) in encoded.iter().zip(&raws).zip(batch) {
                    let r = &mut report[usize::from(t.layer)];
                    r.tiles += 1;
                    r.raw += raw.len() as u64;
                    r.stored += enc.best.stored_len() as u64;
                    r.zstd += enc.zstd as u64;
                    if let (Some(total), Some(p)) = (&mut r.byte_planes, enc.planes) {
                        *total += p as u64;
                    }
                    if enc.best.filter() == Filter::BytePlanes4 {
                        r.byte_plane_tiles += 1;
                    }
                    e.push_encoded(&enc.best)?;
                }
                at = end;
            }
            Ok(())
        },
    )?;
    Ok(RasterReport { layers: report })
}

fn tile_key(layer: usize, lod: u32, x: u32, y: u32) -> TileKey {
    TileKey {
        layer: layer as u16,
        lod: lod as u8,
        _pad: 0,
        x,
        y,
        blob: 0,
    }
}

/// A tile's stored bytes: its texels, or BC blocks for a compressed layer.
fn tile_bytes(asset: &str, l: &Layer, src: &TileSource) -> Result<Vec<u8>> {
    let r = &l.record;
    let texels = match src {
        TileSource::Level(lod, x, y) => {
            l.levels[*lod].tile(r.tile_size, u32::from(r.border), *x, *y)?
        }
        TileSource::File(path, hash) => {
            let bytes = read_checked(path, *hash)?;
            let t = input::decode_tiff(path, &bytes)?;
            let mut out = Vec::new();
            t.format.expand(&t.texels, &mut out);
            out
        }
    };
    let Some(c) = l.spec.compress else {
        return Ok(texels);
    };
    let img = l
        .format
        .stored()
        .to_img(l.spec.srgb, r.tile_texels(), &texels);
    let mut out = encode_chains(c.encoding(), l.spec.quality, &[vec![img]])
        .map_err(|m| raster_error(asset, format!("layer {}: {m}", l.spec.name)))?;
    Ok(out.remove(0).remove(0))
}

struct Encoded<'a> {
    best: EncodedBlob<'a>,
    zstd: usize,
    planes: Option<usize>,
}

/// zstd, and for lossless layers of multi-byte texels also zstd over byte planes; the
/// smaller one is kept.
fn encode<'a>(raw: &'a [u8], l: &Layer) -> Result<Encoded<'a>> {
    let plain = BlobIn::new(raw).encode(ZSTD_LEVEL)?;
    let zstd = plain.stored_len();
    if !l.tries_byte_planes() {
        return Ok(Encoded {
            best: plain,
            zstd,
            planes: None,
        });
    }
    let planes = BlobIn {
        filter: Filter::BytePlanes4,
        ..BlobIn::new(raw)
    }
    .encode(ZSTD_LEVEL)?;
    let p = planes.stored_len();
    Ok(Encoded {
        best: if p < zstd { planes } else { plain },
        zstd,
        planes: Some(p),
    })
}

/// Scratch directory of raster jobs.
pub(crate) fn scratch_dir(cache: Option<&Path>) -> PathBuf {
    cache.map_or_else(std::env::temp_dir, Path::to_owned)
}
