use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;

use orr_cook::{CookOptions, Manifest, Outcome};
use orr_pack::clip::ClipHeader;
use orr_pack::instance::{InstancePageInfo, InstanceRecord};
use orr_pack::mesh::{MeshHeader, MeshInfo, PageHeader};
use orr_pack::raster::{RasterLayer, RasterSetInfo};
use orr_pack::skeleton::{NO_PARENT, SkeletonInfo};
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, Class, Codec, Entry};

const USAGE: &str = "usage:
  orr-cook build <manifest.toml> [--jobs N] [--force] [--cache DIR] [--cache-cap BYTES]
                 [--working-set BYTES] [--report r.json]
  orr-cook inspect <archive> [--class C] [--name N]
  orr-cook verify <archive>
  orr-cook bench-decode <archive>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("build") => build(&args[1..]),
        Some("inspect") => inspect(&args[1..]),
        Some("verify") => verify(&args[1..]),
        Some("bench-decode") => bench_decode(&args[1..]),
        _ => Err(USAGE.into()),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            let _ = writeln!(io::stderr(), "{e}");
            ExitCode::from(2)
        }
    }
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn build(args: &[String]) -> Result<bool> {
    let mut opts = CookOptions::default();
    let mut manifest = None;
    let mut it = args.iter();
    let number =
        |v: Option<&String>| -> Result<u64> { Ok(v.ok_or(USAGE)?.parse().map_err(|_| USAGE)?) };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--jobs" => opts.jobs = number(it.next())? as usize,
            "--force" => opts.force = true,
            "--cache" => opts.cache = Some(it.next().ok_or(USAGE)?.into()),
            "--cache-cap" => opts.cache_cap = Some(number(it.next())?),
            "--working-set" => opts.working_set = number(it.next())?,
            "--report" => opts.report = Some(it.next().ok_or(USAGE)?.into()),
            _ if manifest.is_none() => manifest = Some(Path::new(a)),
            _ => return Err(USAGE.into()),
        }
    }
    let manifest = Manifest::load(manifest.ok_or(USAGE)?)?;
    let cooked = orr_cook::cook(&manifest, &opts)?;
    let mut out = io::stdout().lock();
    let (mut cooked_n, mut linked) = (0, 0);
    for e in cooked.report.assets.iter().flat_map(|a| &a.entries) {
        match e.outcome {
            Outcome::Cooked => cooked_n += 1,
            Outcome::Cache | Outcome::Archive => linked += 1,
        }
    }
    for a in &cooked.report.archives {
        writeln!(
            out,
            "{}: {} entries, {} B{}",
            a.output
                .as_deref()
                .unwrap_or(Path::new("(memory)"))
                .display(),
            a.entries,
            a.bytes,
            if a.written { "" } else { ", unchanged" }
        )?;
    }
    writeln!(
        out,
        "{cooked_n} entries cooked, {linked} linked in {:.2} s",
        cooked.report.seconds
    )?;
    Ok(true)
}

fn class_name(class: u16) -> String {
    match Class::from_u16(class) {
        Some(c) => c.name().into(),
        None if class >= Class::USER_FIRST => format!("user:{:#06x}", class),
        None => format!("unknown:{class}"),
    }
}

fn parse_class(s: &str) -> Result<u16> {
    if let Some(c) = Class::from_name(s) {
        return Ok(c as u16);
    }
    let n = s
        .strip_prefix("0x")
        .map_or_else(|| s.parse(), |h| u16::from_str_radix(h, 16));
    n.map_err(|_| format!("unknown class {s}").into())
}

fn inspect(args: &[String]) -> Result<bool> {
    let (mut path, mut class, mut name) = (None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--class" => class = Some(parse_class(it.next().ok_or(USAGE)?)?),
            "--name" => name = Some(it.next().ok_or(USAGE)?.as_str()),
            _ if path.is_none() => path = Some(Path::new(a)),
            _ => return Err(USAGE.into()),
        }
    }
    let a = Archive::open(path.ok_or(USAGE)?)?;
    let h = a.header();
    let mut out = io::stdout().lock();
    writeln!(
        out,
        "{} entries, {} blobs, {} chunks; index {} B (mount {} B), payload {} B",
        h.entry_count,
        h.blob_count,
        h.chunk_count,
        h.index_len,
        h.mount_len,
        h.file_len - u64::from(h.index_len)
    )?;
    writeln!(
        out,
        "{:<32} {:<20} {:>6} {:>5} {:>12} {:>12}  name",
        "id", "class", "schema", "blobs", "raw", "stored"
    )?;
    let selected = |e: &&Entry| {
        class.is_none_or(|c| e.class == c) && name.is_none_or(|n| a.name(e).contains(n))
    };
    for e in a.entries().iter().filter(selected) {
        let blobs = a.blobs(e);
        let raw: u64 = blobs.iter().map(|b| u64::from(b.raw_len)).sum();
        let stored: u64 = blobs.iter().map(|b| u64::from(b.stored_len)).sum();
        writeln!(
            out,
            "{} {:<20} {:>6} {:>5} {:>12} {:>12}  {}",
            e.id,
            class_name(e.class),
            e.schema,
            blobs.len(),
            raw,
            stored,
            a.name(e)
        )?;
        if name.is_some() || class.is_some() {
            if e.class == Class::Texture as u16 {
                texture_line(&a, e, &mut out)?;
            } else if e.class == Class::InstancePage as u16 {
                instance_line(&a, e, &mut out)?;
            } else if e.class == Class::RasterSet as u16 {
                raster_lines(&a, e, &mut out)?;
            } else if e.class == Class::Skeleton as u16 {
                skeleton_line(&a, e, &mut out)?;
            } else if e.class == Class::Clip as u16 {
                clip_line(&a, e, &mut out)?;
            } else if e.class == Class::Mesh as u16 {
                mesh_lines(&a, e, &mut out)?;
            }
        }
        if name.is_some() {
            for (i, b) in blobs.iter().enumerate() {
                writeln!(
                    out,
                    "    blob {i}: {:?}/{:?}{} offset {} stored {} raw {} chunks {}",
                    b.codec(),
                    b.filter(),
                    if b.is_mount() { " mount" } else { "" },
                    b.offset,
                    b.stored_len,
                    b.raw_len,
                    b.chunk_count
                )?;
            }
        }
    }
    Ok(true)
}

fn mount_blob(a: &Archive, e: &Entry) -> Result<Vec<u8>> {
    let b = &a.blobs(e)[0];
    let mut head = vec![0; b.raw_len as usize];
    a.read(b, &mut head)?;
    Ok(head)
}

/// Format, size, mips and the bytes of the mip tail, which stays resident from mount.
fn texture_line(a: &Archive, e: &Entry, out: &mut impl Write) -> Result<()> {
    let head = mount_blob(a, e)?;
    let info = TextureInfo::parse(&head)?;
    let h = info.header;
    let tail = if h.tail_first_mip < h.mip_count {
        a.blobs(e).last().map_or(0, |b| b.raw_len)
    } else {
        0
    };
    writeln!(
        out,
        "    vk format {} {}x{} x{} layers, {} mips, tail from mip {} ({tail} B), {} lights",
        h.vk_format,
        h.width,
        h.height,
        h.layers,
        h.mip_count,
        h.tail_first_mip,
        info.lights.len()
    )?;
    Ok(())
}

/// Cell, records, prototypes, partitions and the scale range of an instance page.
fn instance_line(a: &Archive, e: &Entry, out: &mut impl Write) -> Result<()> {
    let head = mount_blob(a, e)?;
    let info = InstancePageInfo::parse(&head)?;
    let h = info.header;
    writeln!(
        out,
        "    cell {:?}, {} records, {} prototypes, {} partitions, step {:?} m, scale {:.3}..{:.3}, user stride {}",
        h.cell,
        a.blobs(e)[1].raw_len as usize / size_of::<InstanceRecord>(),
        info.prototypes.len(),
        info.partitions.len(),
        h.extent.map(|x| x / 65535.0),
        h.log2_scale_min.exp2(),
        h.log2_scale_max.exp2(),
        h.user_stride
    )?;
    Ok(())
}

/// Placement of a raster set, then per layer its format, tiles, LODs, reduction, value
/// range and stored bytes.
fn raster_lines(a: &Archive, e: &Entry, out: &mut impl Write) -> Result<()> {
    let blobs = a.blobs(e);
    let head = mount_blob(a, e)?;
    let info = RasterSetInfo::parse(&head)?;
    let h = info.header;
    writeln!(
        out,
        "    world min {:?} m, size {} m, {} layers, {} tiles",
        h.world_min,
        h.world_size,
        info.layers.len(),
        info.tiles.len()
    )?;
    for (i, (l, name)) in info.layers.iter().zip(&info.names).enumerate() {
        let tiles: Vec<_> = info
            .tiles
            .iter()
            .filter(|t| usize::from(t.layer) == i)
            .collect();
        let stored: u64 = tiles
            .iter()
            .map(|t| u64::from(blobs[t.blob as usize].stored_len))
            .sum();
        let reduction = match l.reduction {
            RasterLayer::MEAN => "mean",
            RasterLayer::MIN => "min",
            RasterLayer::MAX => "max",
            _ => "nearest",
        };
        writeln!(
            out,
            "    {name}: vk format {} tile {}+2x{}, {} lods, {} tiles, {reduction}, {}, range {:?}, stored {stored} B",
            l.vk_format,
            l.tile_size,
            l.border,
            l.lod_count,
            tiles.len(),
            if l.lossless == 1 { "lossless" } else { "BCn" },
            l.value_range,
        )?;
    }
    Ok(())
}

/// Joints and the names of the roots.
fn skeleton_line(a: &Archive, e: &Entry, out: &mut impl Write) -> Result<()> {
    let head = mount_blob(a, e)?;
    let info = SkeletonInfo::parse(&head)?;
    let roots: Vec<&str> = (0..info.joint_count())
        .filter(|&j| info.parents[j] == NO_PARENT)
        .map(|j| info.names[j].as_str())
        .collect();
    writeln!(out, "    {} joints, roots {roots:?}", info.joint_count())?;
    Ok(())
}

/// Joints, frames, rate, duration and the skeleton.
fn clip_line(a: &Archive, e: &Entry, out: &mut impl Write) -> Result<()> {
    let head = mount_blob(a, e)?;
    let h: ClipHeader = bytemuck::pod_read_unaligned(
        head.get(..size_of::<ClipHeader>())
            .ok_or("clip header is short")?,
    );
    writeln!(
        out,
        "    {} joints, {} frames at {:.3} Hz ({:.3} s), skeleton {}",
        h.joint_count, h.frame_count, h.sample_rate, h.duration, h.skeleton
    )?;
    Ok(())
}

/// Grid, sections, parts and pages of a mesh, then per level its error, clusters, bytes and
/// micromap bytes.
fn mesh_lines(a: &Archive, e: &Entry, out: &mut impl Write) -> Result<()> {
    let head = mount_blob(a, e)?;
    let info = MeshInfo::parse(&head)?;
    let h = info.header;
    writeln!(
        out,
        "    grid 2^{} m, clas truncate {} bits, {} sections, {} parts, {} pages ({} pinned){}",
        h.position_exponent,
        h.clas_truncate_bits,
        info.sections.len(),
        info.parts.len(),
        info.pages.len(),
        h.pinned_pages,
        if h.merge_level == MeshHeader::NO_MERGE_LEVEL {
            String::new()
        } else {
            format!(", merged from level {}", h.merge_level)
        }
    )?;
    let blobs = a.blobs(e);
    for (i, l) in info.levels.iter().enumerate() {
        let range = l.clusters.first..l.clusters.first + l.clusters.count;
        let (mut first, mut stored, mut resident, mut omm) = (0, 0u64, 0u64, 0u64);
        for p in &info.pages {
            if range.contains(&first) {
                let b = &blobs[p.blob as usize];
                stored += u64::from(b.stored_len);
                resident += u64::from(p.resident_len);
                // A stored page starts with its resident header.
                let mut page = vec![0; b.raw_len as usize];
                a.read(b, &mut page)?;
                let h: PageHeader = bytemuck::pod_read_unaligned(
                    page.get(..size_of::<PageHeader>())
                        .ok_or("page header is short")?,
                );
                omm += u64::from(h.len.saturating_sub(h.omm_offset));
            }
            first += u32::from(p.cluster_count);
        }
        writeln!(
            out,
            "    level {i}: error {} m, {} clusters, stored {stored} B, resident {resident} B, micromaps {omm} B",
            l.error, l.clusters.count
        )?;
    }
    Ok(())
}

/// Decode throughput per class on one thread: stored bytes to resident output, mesh pages
/// through `decode_page`. Blobs already in memory are read first, so only decode is timed.
fn bench_decode(args: &[String]) -> Result<bool> {
    let [path] = args else {
        return Err(USAGE.into());
    };
    let a = Archive::open(Path::new(path))?;
    let mut by_class: std::collections::BTreeMap<u16, (u64, f64)> = Default::default();
    let mut scratch = Vec::new();
    let mut page = Vec::new();
    for e in a.entries() {
        let blobs = a.blobs(e);
        if blobs.iter().any(|b| b.codec() == Codec::GDeflate) {
            continue;
        }
        let mesh = (e.class == Class::Mesh as u16)
            .then(|| MeshInfo::parse(&mount_blob(&a, e).ok()?).ok())
            .flatten();
        for (i, b) in blobs.iter().enumerate() {
            let mut stored = vec![0; b.stored_len as usize];
            a.read_stored(b, &mut stored)?;
            scratch.resize(b.raw_len as usize, 0);
            let resident = mesh
                .as_ref()
                .and_then(|m| m.pages.iter().find(|p| p.blob as usize == i))
                .map(|p| p.resident_len as usize);
            let start = std::time::Instant::now();
            a.decode(b, None, &stored, &mut scratch)?;
            let bytes = match resident {
                Some(len) => {
                    page.resize(len, 0);
                    orr_pack::mesh::decode_page(&scratch, &mut page)?;
                    len
                }
                None => scratch.len(),
            };
            let t = start.elapsed().as_secs_f64();
            let c = by_class.entry(e.class).or_default();
            c.0 += bytes as u64;
            c.1 += t;
        }
    }
    let mut out = io::stdout().lock();
    for (class, (bytes, seconds)) in by_class {
        writeln!(
            out,
            "{:<20} {:>14} B  {:>8.3} GB/s",
            class_name(class),
            bytes,
            bytes as f64 / seconds.max(1e-9) / 1e9
        )?;
    }
    Ok(true)
}

fn verify(args: &[String]) -> Result<bool> {
    let [path] = args else {
        return Err(USAGE.into());
    };
    let a = Archive::open(Path::new(path))?;
    let mut out = io::stdout().lock();
    let (mut bad, mut gpu_only) = (0, 0);
    for e in a.entries() {
        if a.blobs(e).iter().any(|b| b.codec() == Codec::GDeflate) {
            gpu_only += 1;
            continue;
        }
        if let Err(err) = a.verify(e) {
            writeln!(out, "FAIL {} ({}): {err}", a.name(e), e.id)?;
            bad += 1;
        }
    }
    writeln!(
        out,
        "{} entries: {} ok, {bad} failed, {gpu_only} not checked (GDeflate)",
        a.entries().len(),
        a.entries().len() - bad - gpu_only
    )?;
    Ok(bad == 0)
}
