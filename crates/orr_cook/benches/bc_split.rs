#![allow(clippy::print_stdout)] // the bench reports measured figures

//! zstd bytes and decode rate of BCn textures with and without the endpoint and index split.
//! Every BCn texture of the `.orr` archives under `ORR_COOKED_SET` is stored again both ways,
//! with the cook's zstd level and chunking, and decoded on one thread. Prints, per format,
//! the stored bytes each way, the split's gain, and the decode rate each way.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, ArchiveWriter, BcLayout, BlobIn, Class, Filter, Hash32, Target};
use rayon::prelude::*;

const ZSTD_LEVEL: i32 = 19;

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            files(&p, out);
        } else if p.extension().is_some_and(|x| x == "orr") {
            out.push(p);
        }
    }
}

/// A texture's decoded blobs, whether each is a mount blob, and its chunk length.
type Texture = Vec<(Vec<u8>, bool, u32)>;

/// The textures stored with `filter` (blob 0 never takes it), and their stored payload bytes.
fn store(textures: &[Texture], filter: Filter) -> (Archive, u64) {
    let mut w = ArchiveWriter::create(Target::Memory).unwrap();
    w.set_zstd_level(ZSTD_LEVEL);
    let mut stored = 0;
    for (i, t) in textures.iter().enumerate() {
        let encoded: Vec<_> = t
            .par_iter()
            .enumerate()
            .map(|(b, (bytes, mount, chunk_len))| {
                BlobIn {
                    filter: if b == 0 { Filter::None } else { filter },
                    mount: *mount,
                    chunk_len: *chunk_len,
                    ..BlobIn::new(bytes)
                }
                .encode(ZSTD_LEVEL, None)
                .unwrap()
            })
            .collect();
        stored += encoded[1..]
            .iter()
            .map(|e| e.stored_len() as u64)
            .sum::<u64>();
        w.add_streamed::<orr_pack::Error>(
            &i.to_string(),
            Class::Texture as u16,
            orr_pack::texture::SCHEMA,
            Hash32::of(&[]),
            |blobs| encoded.iter().try_for_each(|e| blobs.push_encoded(e)),
        )
        .unwrap();
    }
    let a = Archive::from_memory(w.finish().unwrap().unwrap()).unwrap();
    (a, stored)
}

/// Decoded bytes per second of every blob after blob 0, on this thread.
fn decode_rate(a: &Archive) -> f64 {
    let mut buf = Vec::new();
    let (mut bytes, t) = (0u64, Instant::now());
    for e in a.entries() {
        for b in &a.blobs(e)[1..] {
            buf.resize(b.raw_len as usize, 0);
            a.read(b, &mut buf).unwrap();
            bytes += u64::from(b.raw_len);
        }
    }
    bytes as f64 / t.elapsed().as_secs_f64()
}

fn main() {
    let Ok(dir) = std::env::var("ORR_COOKED_SET") else {
        println!("set ORR_COOKED_SET to a directory of cooked archives");
        return;
    };
    let mut paths = Vec::new();
    files(dir.as_ref(), &mut paths);
    let mut sets: BTreeMap<BcLayout, Vec<Texture>> = BTreeMap::new();
    for p in paths {
        let a = Archive::open(&p).unwrap();
        for e in a.entries() {
            if e.class != Class::Texture as u16 {
                continue;
            }
            let blobs: Vec<_> = a
                .blobs(e)
                .iter()
                .map(|b| {
                    let mut bytes = vec![0; b.raw_len as usize];
                    a.read(b, &mut bytes).unwrap();
                    let chunk_len = if b.chunk_count > 1 {
                        b.chunk_raw_len()
                    } else {
                        0
                    };
                    (bytes, b.is_mount(), chunk_len)
                })
                .collect();
            let info = TextureInfo::parse(&blobs[0].0).unwrap();
            if let Some(l) = BcLayout::of(info.header.vk_format) {
                sets.entry(l).or_default().push(blobs);
            }
        }
    }
    for (layout, textures) in &sets {
        let raw: u64 = textures
            .iter()
            .flat_map(|t| &t[1..])
            .map(|b| b.0.len() as u64)
            .sum();
        let (plain, plain_bytes) = store(textures, Filter::None);
        let (split, split_bytes) = store(textures, Filter::BcSplit(*layout));
        let gain = 100.0 * (1.0 - split_bytes as f64 / plain_bytes as f64);
        println!(
            "{layout:?}: {} textures, {:.1} MB raw, zstd {:.1} MB, split {:.1} MB ({gain:+.1}%), decode {:.0} MB/s, split {:.0} MB/s",
            textures.len(),
            raw as f64 / 1e6,
            plain_bytes as f64 / 1e6,
            split_bytes as f64 / 1e6,
            decode_rate(&plain) / 1e6,
            decode_rate(&split) / 1e6,
        );
    }
}
