use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;

use orr_cook::{CookOptions, Manifest, Outcome};
use orr_pack::instance::{InstancePageInfo, InstanceRecord};
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, Class, Codec, Entry};

const USAGE: &str = "usage:
  orr-cook build <manifest.toml> [--jobs N] [--force] [--cache DIR] [--cache-cap BYTES]
                 [--working-set BYTES] [--report r.json]
  orr-cook inspect <archive> [--class C] [--name N]
  orr-cook verify <archive>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("build") => build(&args[1..]),
        Some("inspect") => inspect(&args[1..]),
        Some("verify") => verify(&args[1..]),
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

/// Format, size, mips and the bytes of the mip tail, which stays resident from mount.
fn texture_line(a: &Archive, e: &Entry, out: &mut impl Write) -> Result<()> {
    let b = &a.blobs(e)[0];
    let mut head = vec![0; b.raw_len as usize];
    a.read(b, &mut head)?;
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
    let b = &a.blobs(e)[0];
    let mut head = vec![0; b.raw_len as usize];
    a.read(b, &mut head)?;
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
