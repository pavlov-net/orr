//! Raster inputs: raw rasters with a JSON sidecar, TIFF rasters and tiles, the level files
//! the cook builds from a whole raster, and tile reads from them.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use orr_pack::Hash32;
use serde::Deserialize;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;
use tiff::{ColorType, TiffError};

use super::texel::{RasterFormat, Reduction, reduce_rows};
use crate::state::hash_file;
use crate::{Error, Result};

/// Rows a whole-raster band holds at most, by bytes.
const BAND_BYTES: usize = 8 << 20;

/// The JSON sidecar of a raw raster: little-endian texels, row-major, rows tightly packed,
/// starting `offset` bytes into the file.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Sidecar {
    pub width: u32,
    pub height: u32,
    pub format: RasterFormat,
    #[serde(default)]
    pub offset: u64,
}

pub(crate) fn raster_error(asset: &str, message: impl Into<String>) -> Error {
    Error::Raster {
        asset: asset.into(),
        message: message.into(),
    }
}

fn read_error(path: &Path, source: io::Error) -> Error {
    Error::Read {
        path: path.to_owned(),
        source,
    }
}

/// Fails if the file's hash is not `hash`.
pub(crate) fn check_hash(path: &Path, hash: Hash32) -> Result<()> {
    if hash_file(path)? != hash {
        return Err(Error::Changed {
            path: path.to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn is_tiff(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
}

fn import_error(path: &Path, message: String) -> Error {
    Error::Import {
        asset: path.display().to_string(),
        message,
    }
}

fn tiff_error(path: &Path, e: TiffError) -> Error {
    import_error(path, format!("TIFF: {e}"))
}

/// Input format of a TIFF image: 8-bit gray, gray-alpha, RGB and RGBA, 16-bit gray and
/// 32-bit float gray, chunky.
fn tiff_format<R: Read + Seek>(d: &mut Decoder<R>) -> std::result::Result<RasterFormat, String> {
    let float = d
        .find_tag_unsigned_vec::<u16>(Tag::SampleFormat)
        .map_err(|e| e.to_string())?
        .is_some_and(|v| v.first() == Some(&3));
    let planar = d
        .find_tag_unsigned::<u16>(Tag::PlanarConfiguration)
        .map_err(|e| e.to_string())?
        == Some(2);
    let ct = d.colortype().map_err(|e| e.to_string())?;
    let f = match (ct, float) {
        (ColorType::Gray(32), true) => RasterFormat::R32f,
        (ColorType::Gray(16), false) => RasterFormat::R16,
        (ColorType::Gray(8), false) => RasterFormat::R8,
        (ColorType::GrayA(8), false) => RasterFormat::Rg8,
        (ColorType::RGB(8), false) => RasterFormat::Rgb8,
        (ColorType::RGBA(8), false) => RasterFormat::Rgba8,
        _ => return Err(format!("unsupported TIFF colour type {ct:?}")),
    };
    if planar
        && matches!(
            f,
            RasterFormat::Rg8 | RasterFormat::Rgb8 | RasterFormat::Rgba8
        )
    {
        return Err("planar TIFF channels are not supported".into());
    }
    Ok(f)
}

fn result_bytes(r: DecodingResult) -> std::result::Result<Vec<u8>, String> {
    Ok(match r {
        DecodingResult::U8(v) => v,
        DecodingResult::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DecodingResult::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        _ => return Err("unsupported TIFF sample type".into()),
    })
}

/// A decoded TIFF tile: its size, input format and input texels.
pub(crate) struct TiffTile {
    pub size: [u32; 2],
    pub format: RasterFormat,
    pub texels: Vec<u8>,
}

pub(crate) fn decode_tiff(path: &Path, bytes: &[u8]) -> Result<TiffTile> {
    let fail = |m| import_error(path, m);
    let mut d = Decoder::new(Cursor::new(bytes))
        .map_err(|e| tiff_error(path, e))?
        .with_limits(Limits::unlimited());
    let (w, h) = d.dimensions().map_err(|e| tiff_error(path, e))?;
    let format = tiff_format(&mut d).map_err(fail)?;
    let texels = result_bytes(d.read_image().map_err(|e| tiff_error(path, e))?).map_err(fail)?;
    if texels.len() != w as usize * h as usize * format.bytes() {
        return Err(fail(format!(
            "{} bytes for a {w}x{h} {format:?} image",
            texels.len()
        )));
    }
    Ok(TiffTile {
        size: [w, h],
        format,
        texels,
    })
}

/// A whole raster read in bands of full rows.
pub(crate) struct WholeRaster {
    pub size: u32,
    pub format: RasterFormat,
    source: Rows,
}

enum Rows {
    Raw {
        file: BufReader<File>,
        path: PathBuf,
        offset: u64,
        next: u32,
    },
    Tiff {
        decoder: Box<Decoder<BufReader<File>>>,
        path: PathBuf,
        chunk: [u32; 2],
        next: u32,
    },
}

impl WholeRaster {
    pub(crate) fn open(path: &Path, sidecar: Option<&Sidecar>) -> Result<Self> {
        let file = File::open(path).map_err(|e| read_error(path, e))?;
        let file = BufReader::with_capacity(1 << 20, file);
        let fail = |m| import_error(path, m);
        let (w, h, format, source) = match sidecar {
            Some(s) => (
                s.width,
                s.height,
                s.format,
                Rows::Raw {
                    file,
                    path: path.to_owned(),
                    offset: s.offset,
                    next: 0,
                },
            ),
            None => {
                let mut d = Decoder::new(file)
                    .map_err(|e| tiff_error(path, e))?
                    .with_limits(Limits::unlimited());
                let (w, h) = d.dimensions().map_err(|e| tiff_error(path, e))?;
                let format = tiff_format(&mut d).map_err(fail)?;
                let (cw, ch) = d.chunk_dimensions();
                (
                    w,
                    h,
                    format,
                    Rows::Tiff {
                        decoder: Box::new(d),
                        path: path.to_owned(),
                        chunk: [cw, ch],
                        next: 0,
                    },
                )
            }
        };
        if w != h || w == 0 {
            return Err(fail(format!("raster is {w}x{h}; rasters are square")));
        }
        if let Rows::Raw { file, offset, .. } = &source {
            let need = *offset + u64::from(w) * u64::from(h) * format.bytes() as u64;
            let have = file
                .get_ref()
                .metadata()
                .map_err(|e| read_error(path, e))?
                .len();
            if have < need {
                return Err(fail(format!("raw raster holds {have} bytes, needs {need}")));
            }
        }
        Ok(Self {
            size: w,
            format,
            source,
        })
    }

    /// The next band of input rows, or `None` after the last row.
    pub(crate) fn band(&mut self) -> Result<Option<Vec<u8>>> {
        let (size, bpp) = (self.size, self.format.bytes());
        let row = size as usize * bpp;
        match &mut self.source {
            Rows::Raw {
                file,
                path,
                offset,
                next,
            } => {
                if *next == size {
                    return Ok(None);
                }
                let rows = (BAND_BYTES / row).clamp(1, (size - *next) as usize);
                if *next == 0 {
                    file.seek(SeekFrom::Start(*offset))
                        .map_err(|e| read_error(path, e))?;
                }
                let mut band = vec![0; rows * row];
                file.read_exact(&mut band)
                    .map_err(|e| read_error(path, e))?;
                *next += rows as u32;
                Ok(Some(band))
            }
            Rows::Tiff {
                decoder,
                path,
                chunk: [cw, ch],
                next,
            } => {
                if *next == size {
                    return Ok(None);
                }
                let across = size.div_ceil(*cw);
                let chunk_row = *next / *ch;
                let rows = (*ch).min(size - *next) as usize;
                let mut band = vec![0; rows * row];
                for col in 0..across {
                    let index = chunk_row * across + col;
                    let (dw, dh) = decoder.chunk_data_dimensions(index);
                    let (dw, dh) = (dw as usize, dh as usize);
                    let bytes =
                        result_bytes(decoder.read_chunk(index).map_err(|e| tiff_error(path, e))?)
                            .map_err(|m| import_error(path, m))?;
                    let stride = bytes.len() / dh.max(1);
                    let x0 = (col * *cw) as usize * bpp;
                    for r in 0..dh.min(rows) {
                        band[r * row + x0..][..dw * bpp]
                            .copy_from_slice(&bytes[r * stride..][..dw * bpp]);
                    }
                }
                *next += rows as u32;
                Ok(Some(band))
            }
        }
    }
}

/// Files the cook writes while it runs, removed when dropped.
pub(crate) struct Scratch {
    dir: PathBuf,
    stem: String,
    files: Vec<PathBuf>,
}

impl Scratch {
    pub(crate) fn new(dir: &Path, key: Hash32) -> Self {
        Self {
            dir: dir.to_owned(),
            stem: format!("orr-cook-{}-{}", crate::key::hex(key), std::process::id()),
            files: Vec::new(),
        }
    }

    fn create(&mut self) -> Result<(File, PathBuf)> {
        let path = self
            .dir
            .join(format!("{}-{}.tmp", self.stem, self.files.len()));
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|source| Error::Write {
                path: path.clone(),
                source,
            })?;
        self.files.push(path.clone());
        Ok((file, path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for f in &self.files {
            let _ = fs::remove_file(f);
        }
    }
}

/// One level of a layer as row-major stored texels in a file.
pub(crate) struct Level {
    file: File,
    path: PathBuf,
    offset: u64,
    pub size: u32,
    bpp: usize,
}

impl Level {
    /// The bordered tile `(x, y)`: level texels `x * tile - border` on, edges repeated.
    pub(crate) fn tile(&self, tile: u32, border: u32, x: u32, y: u32) -> Result<Vec<u8>> {
        let n = (tile + 2 * border) as usize;
        let bpp = self.bpp;
        let size = i64::from(self.size);
        let x0 = i64::from(x) * i64::from(tile) - i64::from(border);
        let y0 = i64::from(y) * i64::from(tile) - i64::from(border);
        let (lo, hi) = (x0.max(0), (x0 + n as i64).min(size));
        let (left, span) = ((lo - x0) as usize, (hi - lo) as usize);
        let mut out = vec![0; n * n * bpp];
        for (r, row) in out.chunks_exact_mut(n * bpp).enumerate() {
            let ly = (y0 + r as i64).clamp(0, size - 1) as u64;
            let at = self.offset + (ly * u64::from(self.size) + lo as u64) * bpp as u64;
            read_exact_at(&self.file, &mut row[left * bpp..][..span * bpp], at)
                .map_err(|e| read_error(&self.path, e))?;
            let first = left * bpp;
            for p in 0..left {
                row.copy_within(first..first + bpp, p * bpp);
            }
            let last = (left + span - 1) * bpp;
            for p in left + span..n {
                row.copy_within(last..last + bpp, p * bpp);
            }
        }
        Ok(out)
    }
}

/// The levels of a whole raster, coarsest first, and the value range of its texels.
pub(crate) struct Levels {
    pub levels: Vec<Level>,
    pub range: [f32; 2],
}

/// Streams a whole raster once in row bands, writing each coarser level by `reduction`
/// to a scratch file. The finest level reads from the input when it holds stored texels
/// already (a raw raster of a stored format), else it is written to scratch too.
pub(crate) fn build_levels(
    mut src: WholeRaster,
    raw: Option<(&Path, u64)>,
    lod_count: u32,
    reduction: Reduction,
    srgb: bool,
    scratch: &mut Scratch,
) -> Result<Levels> {
    let format = src.format;
    let t = format.stored();
    let bpp = t.bytes();
    let size = src.size;
    let direct = raw.filter(|_| format.bytes() == bpp);
    let mut finest = match direct {
        Some(_) => None,
        None => Some(scratch.create()?),
    };
    let mut finest_out = finest
        .as_ref()
        .map(|(f, _)| BufWriter::with_capacity(1 << 20, f));
    // Coarser levels from the finest - 1 down to 0, each with its pending upper row.
    let mut coarse = Vec::new();
    for l in (0..lod_count - 1).rev() {
        let (file, path) = scratch.create()?;
        coarse.push((file, path, size >> (lod_count - 1 - l)));
    }
    let mut pending: Vec<Option<Vec<u8>>> = vec![None; coarse.len()];
    let mut writers: Vec<BufWriter<&File>> = coarse
        .iter()
        .map(|(f, ..)| BufWriter::with_capacity(1 << 20, f))
        .collect();
    let mut range = [f32::INFINITY, f32::NEG_INFINITY];
    let mut stored = Vec::new();
    let row_bytes = size as usize * bpp;
    while let Some(band) = src.band()? {
        format.range(&band, &mut range);
        stored.clear();
        format.expand(&band, &mut stored);
        if let Some(w) = &mut finest_out {
            w.write_all(&stored).map_err(|e| write_error(scratch, e))?;
        }
        for row in stored.chunks_exact(row_bytes) {
            let mut row = row.to_vec();
            for (k, p) in pending.iter_mut().enumerate() {
                let Some(top) = p.take() else {
                    *p = Some(row);
                    break;
                };
                let mut out = vec![0; row.len() / 2];
                reduce_rows(t, reduction, srgb, &top, &row, &mut out);
                writers[k]
                    .write_all(&out)
                    .map_err(|e| write_error(scratch, e))?;
                row = out;
            }
        }
    }
    for w in writers {
        w.into_inner()
            .map_err(|e| write_error(scratch, e.into_error()))?;
    }
    finest_out
        .map(BufWriter::into_inner)
        .transpose()
        .map_err(|e| write_error(scratch, e.into_error()))?;
    let finest = match (finest.take(), direct) {
        (Some((file, path)), _) => Level {
            file,
            path,
            offset: 0,
            size,
            bpp,
        },
        (None, Some((path, offset))) => Level {
            file: File::open(path).map_err(|e| read_error(path, e))?,
            path: path.to_owned(),
            offset,
            size,
            bpp,
        },
        (None, None) => unreachable!("the finest level is direct or written"),
    };
    let mut levels: Vec<Level> = coarse
        .into_iter()
        .rev()
        .map(|(file, path, size)| Level {
            file,
            path,
            offset: 0,
            size,
            bpp,
        })
        .collect();
    levels.push(finest);
    Ok(Levels { levels, range })
}

fn write_error(scratch: &Scratch, source: io::Error) -> Error {
    Error::Write {
        path: scratch.dir.clone(),
        source,
    }
}

fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    if orr_io::read_at(f, offset, buf)? < buf.len() {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(())
}

/// A pattern over relative tile paths: `/`-separated components with `{lod}`, `{x}` and
/// `{y}` (decimal numbers; a name used twice must match the same number) and `*` (any run
/// of characters within a component).
pub(crate) struct TilePattern {
    components: Vec<Vec<Token>>,
}

#[derive(Clone, PartialEq)]
enum Token {
    Lit(char),
    Star,
    Var(usize),
}

impl TilePattern {
    pub(crate) fn parse(s: &str) -> std::result::Result<Self, String> {
        let mut components = Vec::new();
        let mut seen = [false; 3];
        for c in s.split('/') {
            let mut tokens = Vec::new();
            let mut rest = c;
            while let Some(ch) = rest.chars().next() {
                if ch == '{' {
                    let end = rest.find('}').ok_or("unclosed { in the tile pattern")?;
                    let var = match &rest[1..end] {
                        "lod" => 0,
                        "x" => 1,
                        "y" => 2,
                        v => return Err(format!("unknown tile pattern field {{{v}}}")),
                    };
                    seen[var] = true;
                    tokens.push(Token::Var(var));
                    rest = &rest[end + 1..];
                } else {
                    tokens.push(if ch == '*' {
                        Token::Star
                    } else {
                        Token::Lit(ch)
                    });
                    rest = &rest[ch.len_utf8()..];
                }
            }
            components.push(tokens);
        }
        if seen != [true; 3] {
            return Err("the tile pattern needs {lod}, {x} and {y}".into());
        }
        Ok(Self { components })
    }

    /// `[lod, x, y]` of a relative path, if it matches.
    pub(crate) fn matches(&self, rel: &[String]) -> Option<[u32; 3]> {
        if rel.len() != self.components.len() {
            return None;
        }
        let mut vars = [None; 3];
        for (tokens, name) in self.components.iter().zip(rel) {
            let chars: Vec<char> = name.chars().collect();
            if !match_tokens(tokens, &chars, &mut vars) {
                return None;
            }
        }
        Some(vars.map(|v| v.expect("every field appears")))
    }
}

fn match_tokens(t: &[Token], s: &[char], vars: &mut [Option<u32>; 3]) -> bool {
    match t.first() {
        None => s.is_empty(),
        Some(Token::Lit(c)) => s.first() == Some(c) && match_tokens(&t[1..], &s[1..], vars),
        Some(Token::Star) => (0..=s.len()).any(|i| {
            let mut v = *vars;
            let ok = match_tokens(&t[1..], &s[i..], &mut v);
            if ok {
                *vars = v;
            }
            ok
        }),
        Some(Token::Var(i)) => {
            let digits = s.iter().take_while(|c| c.is_ascii_digit()).count();
            (1..=digits.min(9)).any(|n| {
                let value: u32 = s[..n].iter().collect::<String>().parse().expect("digits");
                if vars[*i].is_some_and(|v| v != value) {
                    return false;
                }
                let mut v = *vars;
                v[*i] = Some(value);
                let ok = match_tokens(&t[1..], &s[n..], &mut v);
                if ok {
                    *vars = v;
                }
                ok
            })
        }
    }
}

/// Every file under `root` with its path components relative to it.
pub(crate) fn walk(root: &Path) -> Result<Vec<(PathBuf, Vec<String>)>> {
    let mut out = Vec::new();
    let mut stack = vec![(root.to_owned(), Vec::new())];
    while let Some((dir, rel)) = stack.pop() {
        let mut items: Vec<_> = fs::read_dir(&dir)
            .map_err(|e| read_error(&dir, e))?
            .collect::<io::Result<_>>()
            .map_err(|e| read_error(&dir, e))?;
        items.sort_by_key(fs::DirEntry::file_name);
        for item in items {
            let name = item.file_name().to_string_lossy().into_owned();
            let mut r: Vec<String> = rel.clone();
            r.push(name);
            let ty = item.file_type().map_err(|e| read_error(&item.path(), e))?;
            if ty.is_dir() {
                stack.push((item.path(), r));
            } else {
                out.push((item.path(), r));
            }
        }
    }
    Ok(out)
}
