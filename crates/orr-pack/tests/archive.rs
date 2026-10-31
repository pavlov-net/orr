use std::path::PathBuf;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use orr_pack::{
    Archive, ArchiveWriter, AssetId, BlobIn, Codec, Error, Filter, Hash32, StoredRange, Target,
};

const USER: u16 = 0x8001;

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Compressible bytes: a slow ramp of f32s with noise in the low byte.
fn data(len: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..len)
        .map(|i| {
            if i % 4 == 0 {
                r.next() as u8 & 3
            } else {
                (i / 4096 + i % 4) as u8
            }
        })
        .collect()
}

fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed | 1);
    (0..len).map(|_| r.next() as u8).collect()
}

struct Case {
    name: String,
    blobs: Vec<(Vec<u8>, Codec, Filter, bool, u32)>,
}

impl Case {
    fn ins(&self) -> Vec<BlobIn<'_>> {
        self.blobs
            .iter()
            .map(|(bytes, codec, filter, mount, chunk_len)| BlobIn {
                bytes,
                codec: *codec,
                filter: *filter,
                mount: *mount,
                chunk_len: *chunk_len,
            })
            .collect()
    }
}

fn cases() -> Vec<Case> {
    const K: usize = 1024;
    let mut out = Vec::new();
    let mut add = |name: &str, blobs: Vec<(Vec<u8>, Codec, Filter, bool, u32)>| {
        out.push(Case {
            name: name.into(),
            blobs,
        });
    };
    for (codec, filter, tag) in [
        (Codec::None, Filter::None, "none"),
        (Codec::Zstd, Filter::None, "zstd"),
        (Codec::Zstd, Filter::BytePlanes4, "zstd-planes"),
        (Codec::None, Filter::BytePlanes4, "none-planes"),
    ] {
        let b = |v: Vec<u8>, mount, chunk| (v, codec, filter, mount, chunk);
        add(&format!("{tag}/empty"), vec![b(Vec::new(), false, 0)]);
        add(
            &format!("{tag}/small"),
            vec![b(data(1000, 1), false, 0), b(data(37, 2), true, 0)],
        );
        add(
            &format!("{tag}/one-chunk"),
            vec![
                b(data(256 * K, 3), false, 0),
                b(data(64 * K, 4), false, 64 * K as u32),
            ],
        );
        add(
            &format!("{tag}/short-last"),
            vec![
                b(data(256 * K * 2 + 103, 5), false, 0),
                b(data(64 * K * 3 + 1, 6), false, 64 * K as u32),
            ],
        );
        add(
            &format!("{tag}/incompressible"),
            vec![b(noise(300 * K, 7), false, 0), b(noise(5000, 8), true, 0)],
        );
        add(
            &format!("{tag}/mount-large"),
            vec![b(data(600 * K, 9), true, 0)],
        );
    }
    // Same bytes and settings as an earlier blob: stored once.
    add(
        "dup",
        vec![(data(1000, 1), Codec::Zstd, Filter::None, false, 0)],
    );
    out
}

fn write(target: Target, cases: &[Case]) -> Option<Arc<[u8]>> {
    let mut w = ArchiveWriter::create(target).unwrap();
    w.set_zstd_level(3);
    for c in cases {
        w.add(&c.name, USER, 1, Hash32::of(c.name.as_bytes()), &c.ins())
            .unwrap();
    }
    w.finish().unwrap()
}

fn check(a: &Archive, cases: &[Case]) {
    assert_eq!(a.entries().len(), cases.len());
    for c in cases {
        let e = a.expect(AssetId::from_name(&c.name), USER, 1).unwrap();
        assert_eq!(a.name(e), c.name);
        a.verify(e).unwrap();
        for (b, (want, ..)) in a.blobs(e).iter().zip(&c.blobs) {
            let mut got = vec![0; b.raw_len as usize];
            a.read(b, &mut got).unwrap();
            assert!(got == *want, "{}", c.name);
            let mut at = 0;
            for i in 0..u32::from(b.chunk_count) {
                let mut chunk = vec![0; b.chunk_len(i) as usize];
                a.read_chunk(b, i, &mut chunk).unwrap();
                assert!(chunk == want[at..at + chunk.len()], "{} chunk {i}", c.name);
                at += chunk.len();
            }
            assert_eq!(at, want.len());
            if !b.is_mount() {
                assert_eq!(b.offset % 4096, 0);
            }
        }
    }
}

#[repr(C, align(4096))]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Page([u8; 4096]);

/// Reads every blob and chunk at its stored range into a 4096-aligned buffer through
/// `read_range` and checks that it decodes to the bytes `Archive::read` gives.
fn check_stored_ranges(a: &Archive, mut read_range: impl FnMut(StoredRange, &mut [u8])) {
    for e in a.entries() {
        for b in a.blobs(e) {
            let mut want = vec![0; b.raw_len as usize];
            a.read(b, &mut want).unwrap();
            for chunk in std::iter::once(None).chain((0..u32::from(b.chunk_count)).map(Some)) {
                let r = a.stored_range(b, chunk);
                assert!(r.offset.is_multiple_of(4096) && r.len.is_multiple_of(4096));
                let mut pages = vec![Page([0; 4096]); r.len as usize / 4096];
                let buf: &mut [u8] = bytemuck::cast_slice_mut(&mut pages);
                read_range(r, buf);
                let (lo, n) = match chunk {
                    None => (0, b.raw_len as usize),
                    Some(i) => (
                        i as usize * b.chunk_raw_len() as usize,
                        b.chunk_len(i) as usize,
                    ),
                };
                let mut got = vec![0; n];
                a.decode(b, chunk, &buf[r.skip as usize..], &mut got)
                    .unwrap();
                assert!(got == want[lo..lo + n]);
            }
        }
    }
}

#[test]
fn round_trip_file_and_memory() {
    let cases = cases();
    let path = dir("round_trip").join("a.orrpack");
    assert!(write(Target::File(path.clone()), &cases).is_none());
    let memory = write(Target::Memory, &cases).unwrap();
    let file_bytes = std::fs::read(&path).unwrap();
    assert!(
        file_bytes[..] == memory[..],
        "memory and file archives differ"
    );
    assert_eq!(file_bytes.len() % 4096, 0);

    let a = Archive::open(&path).unwrap();
    check(&a, &cases);
    let m = Archive::from_memory(memory).unwrap();
    check(&m, &cases);

    let dup = |a: &Archive, n: &str| a.blobs(a.find(AssetId::from_name(n)).unwrap())[0];
    assert_eq!(dup(&a, "dup"), dup(&a, "zstd/small"));
    let big = dup(&a, "zstd/short-last");
    assert!(big.chunk_count == 3 && big.codec() == Codec::Zstd);
    let raw = dup(&a, "zstd/incompressible");
    assert_eq!(
        raw.codec(),
        Codec::None,
        "incompressible blobs are stored raw"
    );

    check_stored_ranges(&a, |r, buf| {
        buf.copy_from_slice(&file_bytes[r.offset as usize..][..buf.len()]);
    });
}

#[cfg(windows)]
#[test]
fn unbuffered_reads_at_stored_ranges() {
    use std::os::windows::fs::OpenOptionsExt;
    const NO_BUFFERING: u32 = 0x2000_0000;

    let path = dir("unbuffered").join("a.orrpack");
    write(Target::File(path.clone()), &cases());
    let a = Archive::open(&path).unwrap();
    let f = std::fs::File::options()
        .read(true)
        .custom_flags(NO_BUFFERING)
        .open(&path)
        .unwrap();
    check_stored_ranges(&a, |r, buf| {
        assert_eq!(orr_io::read_at(&f, r.offset, buf).unwrap(), buf.len());
    });
}

#[test]
fn concurrent_reads() {
    let cases = cases();
    let path = dir("concurrent").join("a.orrpack");
    write(Target::File(path.clone()), &cases);
    let a = Archive::open(&path).unwrap();
    let want: Vec<Vec<Vec<u8>>> = cases
        .iter()
        .map(|c| c.blobs.iter().map(|b| b.0.clone()).collect())
        .collect();
    std::thread::scope(|s| {
        for t in 0..8 {
            let (a, cases, want) = (&a, &cases, &want);
            s.spawn(move || {
                let mut r = Rng(0x9E37_79B9 + t);
                for _ in 0..400 {
                    let ci = r.next() as usize % cases.len();
                    let e = a.find(AssetId::from_name(&cases[ci].name)).unwrap();
                    let bi = r.next() as usize % a.blobs(e).len();
                    let b = &a.blobs(e)[bi];
                    let w = &want[ci][bi];
                    if r.next().is_multiple_of(2) {
                        let mut got = vec![0; b.raw_len as usize];
                        a.read(b, &mut got).unwrap();
                        assert!(got == *w);
                    } else {
                        let i = (r.next() % u64::from(b.chunk_count)) as u32;
                        let mut got = vec![0; b.chunk_len(i) as usize];
                        a.read_chunk(b, i, &mut got).unwrap();
                        let lo = i as usize * b.chunk_raw_len() as usize;
                        assert!(got == w[lo..lo + got.len()]);
                    }
                }
            });
        }
    });
}

#[test]
fn copy_entry_keeps_stored_bytes() {
    let cases = cases();
    let src = Archive::from_memory(write(Target::Memory, &cases).unwrap()).unwrap();
    let mut w = ArchiveWriter::create(Target::Memory).unwrap();
    for e in src.entries() {
        w.copy_entry(&src, e).unwrap();
    }
    assert!(matches!(
        w.copy_entry(&src, &src.entries()[0]),
        Err(Error::Duplicate(_))
    ));
    let copy = Archive::from_memory(w.finish().unwrap().unwrap()).unwrap();
    check(&copy, &cases);
    for (a, b) in src.entries().iter().zip(copy.entries()) {
        assert_eq!(
            (a.id, a.content_hash, a.cook_key),
            (b.id, b.content_hash, b.cook_key)
        );
        for (x, y) in src.blobs(a).iter().zip(copy.blobs(b)) {
            let (mut sx, mut sy) = (
                vec![0; x.stored_len as usize],
                vec![0; y.stored_len as usize],
            );
            src.read_stored(x, &mut sx).unwrap();
            copy.read_stored(y, &mut sy).unwrap();
            assert!(sx == sy && x.codec == y.codec && x.raw_len == y.raw_len);
        }
    }
}

#[test]
fn damaged_archives_fail_by_name() {
    let cases = cases();
    let good = write(Target::Memory, &cases).unwrap();
    let from = |v: Vec<u8>| Archive::from_memory(v.into());

    let mut toc = good.to_vec();
    toc[200] ^= 1;
    assert!(matches!(from(toc), Err(Error::CorruptToc)));

    let short = good[..good.len() - 4096].to_vec();
    assert!(matches!(from(short), Err(Error::Truncated { .. })));
    assert!(matches!(
        from(good[..64].to_vec()),
        Err(Error::Truncated { .. })
    ));

    let mut version = good.to_vec();
    version[8] = 99;
    assert!(matches!(
        from(version),
        Err(Error::Version { found: 99, .. })
    ));
    assert!(matches!(from(vec![0; 8192]), Err(Error::NotArchive)));

    let a = Archive::from_memory(Arc::clone(&good)).unwrap();
    let id = AssetId::from_name(&cases[0].name);
    assert!(matches!(
        a.expect(id, USER, 2),
        Err(Error::Schema {
            found: 1,
            expected: 2,
            ..
        })
    ));
    assert!(matches!(
        a.expect(id, USER + 1, 1),
        Err(Error::Class { .. })
    ));
    assert!(matches!(
        a.expect(AssetId::from_name("absent"), USER, 1),
        Err(Error::Missing(_))
    ));

    // A payload byte flipped: the archive opens, and verify catches the entry.
    let e = a.find(AssetId::from_name("none/small")).unwrap();
    let b = a.blobs(e)[0];
    let mut payload = good.to_vec();
    payload[b.offset as usize] ^= 1;
    let damaged = from(payload).unwrap();
    let e = damaged.find(e.id).unwrap();
    assert!(matches!(damaged.verify(e), Err(Error::ContentHash { .. })));

    let path = dir("truncated").join("a.orrpack");
    std::fs::write(&path, &good[..good.len() - 4096]).unwrap();
    assert!(matches!(Archive::open(&path), Err(Error::Truncated { .. })));
}

#[test]
fn many_entries_open_from_the_index_range() {
    let path = dir("many").join("a.orrpack");
    let mut w = ArchiveWriter::create(Target::File(path.clone())).unwrap();
    for i in 0u32..100_000 {
        let bytes = i.to_le_bytes();
        let blob = BlobIn {
            mount: i % 2 == 0,
            codec: Codec::None,
            ..BlobIn::new(&bytes)
        };
        w.add(&format!("asset/{i}"), USER, 1, Hash32([0; 32]), &[blob])
            .unwrap();
    }
    w.finish().unwrap();

    let file = std::fs::read(&path).unwrap();
    let range = Archive::index_range(&file[..Archive::HEAD_LEN]).unwrap();
    let index = file[..range.len as usize].to_vec().into_boxed_slice();
    let a = Archive::from_index(&path, index).unwrap();
    assert_eq!(a.entries().len(), 100_000);
    for i in [0u32, 1, 4242, 99_999] {
        let e = a.find(AssetId::from_name(&format!("asset/{i}"))).unwrap();
        let mut v = [0; 4];
        a.read(&a.blobs(e)[0], &mut v).unwrap();
        assert_eq!(u32::from_le_bytes(v), i);
    }
}

#[cfg(windows)]
#[test]
fn finish_replaces_an_open_archive() {
    use std::os::windows::fs::OpenOptionsExt;
    const SHARE_READ: u32 = 1;

    let d = dir("replace");
    let path = d.join("a.orrpack");
    let one = |tag: &str| Case {
        name: tag.into(),
        blobs: vec![(
            data(70_000, tag.len() as u64),
            Codec::Zstd,
            Filter::None,
            false,
            0,
        )],
    };
    write(Target::File(path.clone()), &[one("old")]);

    let old = Archive::open(&path).unwrap();
    write(Target::File(path.clone()), &[one("newer")]);
    check(&old, &[one("old")]);
    check(&Archive::open(&path).unwrap(), &[one("newer")]);

    let _locked = std::fs::File::options()
        .read(true)
        .share_mode(SHARE_READ)
        .open(&path)
        .unwrap();
    let mut w = ArchiveWriter::create(Target::File(path.clone())).unwrap();
    w.add("newest", USER, 1, Hash32([0; 32]), &[BlobIn::new(b"x")])
        .unwrap();
    assert!(matches!(w.finish(), Err(Error::Replace { .. })));
    check(&Archive::open(&path).unwrap(), &[one("newer")]);
    assert!(d.join("a.orrpack.tmp").exists());
    assert!(!d.join("a.orrpack.payload.tmp").exists());
}

#[test]
fn streamed_and_pre_encoded_blobs_equal_plain_adds() {
    let cases = cases();
    let plain = write(Target::Memory, &cases).unwrap();
    let mut w = ArchiveWriter::create(Target::Memory).unwrap();
    w.set_zstd_level(3);
    for c in &cases {
        let ins = c.ins();
        let pre: Vec<_> = ins.iter().map(|b| b.encode(3)).collect();
        w.add_streamed(&c.name, USER, 1, Hash32::of(c.name.as_bytes()), |e| {
            for (j, (b, p)) in ins.iter().zip(&pre).enumerate() {
                match p {
                    Ok(p) if j % 2 == 1 => e.push_encoded(p)?,
                    _ => e.push(b)?,
                }
            }
            Ok::<_, Error>(())
        })
        .unwrap();
    }
    assert!(plain[..] == w.finish().unwrap().unwrap()[..]);

    let mut w = ArchiveWriter::create(Target::Memory).unwrap();
    let bytes = data(1 << 20, 3);
    let failed = w.add_streamed("broken", USER, 1, Hash32([0; 32]), |e| {
        e.push(&BlobIn::new(&bytes))?;
        Err(Error::InvalidBlob("stop"))
    });
    assert!(matches!(failed, Err(Error::InvalidBlob("stop"))));
    w.add("kept", USER, 1, Hash32([1; 32]), &[BlobIn::new(&bytes)])
        .unwrap();
    let a = Archive::from_memory(w.finish().unwrap().unwrap()).unwrap();
    assert_eq!(a.entries().len(), 1);
    let e = a.find(AssetId::from_name("kept")).unwrap();
    a.verify(e).unwrap();
}
