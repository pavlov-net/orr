use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use orr_io::{AlignedBuf, Read, ReadFile, Ring, SECTOR, Wake, open_shared, read_at};

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn bytes(len: usize) -> Vec<u8> {
    let mut x = 0x9e37_79b9_7f4a_7c15_u64;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

type Results = Arc<Mutex<Vec<Option<(Vec<u8>, usize)>>>>;

fn run_ring(mut ring: Ring, file: &Arc<ReadFile>, reads: &[(u64, usize)]) -> Vec<Vec<u8>> {
    let results: Results = Arc::new(Mutex::new(vec![None; reads.len()]));
    for (i, &(offset, len)) in reads.iter().enumerate() {
        let results = Arc::clone(&results);
        ring.push(Read {
            file: Arc::clone(file),
            offset,
            len,
            done: Box::new(move |r| {
                let (buf, n) = r.expect("ring read");
                results.lock().unwrap()[i] = Some((buf.as_slice().to_vec(), n));
            }),
        });
    }
    let wake = Wake::default();
    loop {
        ring.pump();
        if !ring.busy() {
            break;
        }
        wake.wait(Some(&ring));
    }
    let results = results.lock().unwrap();
    results
        .iter()
        .map(|r| {
            let (buf, n) = r.as_ref().expect("every read completes");
            buf[..*n].to_vec()
        })
        .collect()
}

#[test]
fn ring_matches_positioned_reads() {
    let len = 37 * SECTOR + 123;
    let data = bytes(len);
    let p = path("ring.bin");
    fs::write(&p, &data).unwrap();
    let file = Arc::new(ReadFile::open(&p).unwrap());
    let plain = fs::File::open(&p).unwrap();

    // More reads than the ring keeps in flight, with lengths that are not whole sectors and
    // reads that cross, end at and start past the end of the file.
    let mut reads: Vec<(u64, usize)> = (0..150)
        .map(|i| {
            (
                (i * 7 % 36) as u64 * SECTOR as u64,
                1 + i * 997 % (3 * SECTOR),
            )
        })
        .collect();
    reads.extend([
        (36 * SECTOR as u64, 2 * SECTOR),
        (37 * SECTOR as u64, SECTOR),
        (37 * SECTOR as u64, 123),
        (38 * SECTOR as u64, SECTOR),
    ]);
    let expected: Vec<Vec<u8>> = reads
        .iter()
        .map(|&(offset, n)| {
            let mut buf = vec![0; n];
            let got = read_at(&plain, offset, &mut buf).unwrap();
            buf.truncate(got);
            buf
        })
        .collect();

    let mut rings = vec![("positioned", Ring::positioned())];
    let native = Ring::native();
    #[cfg(windows)]
    assert!(native.is_some(), "IoRing is missing");
    rings.extend(native.map(|r| ("native", r)));
    for (name, ring) in rings {
        let got = run_ring(ring, &file, &reads);
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(g == e, "{name} ring, read {i} {:?}", reads[i]);
        }
    }
}

#[test]
fn read_at_is_short_only_at_end_of_file() {
    let p = path("short.bin");
    let data = bytes(SECTOR + 10);
    fs::write(&p, &data).unwrap();
    let f = fs::File::open(&p).unwrap();
    for (offset, want) in [
        (0, 64),
        (SECTOR as u64 - 6, 16),
        (SECTOR as u64 + 10, 0),
        (2 * SECTOR as u64, 0),
    ] {
        let mut buf = vec![0; 64];
        let n = read_at(&f, offset, &mut buf).unwrap();
        assert_eq!(n, want, "offset {offset}");
        assert_eq!(&buf[..n], &data[data.len().min(offset as usize)..][..n]);
    }
}

#[test]
fn open_file_can_be_replaced_by_rename() {
    let p = path("replace.bin");
    let new = path("replace.new");
    fs::write(&p, b"old contents").unwrap();
    let shared = open_shared(&p).unwrap();
    let file = Arc::new(ReadFile::open(&p).unwrap());

    fs::write(&new, b"new").unwrap();
    fs::rename(&new, &p).unwrap();
    assert_eq!(fs::read(&p).unwrap(), b"new");

    let mut buf = [0; 32];
    let n = read_at(&shared, 0, &mut buf).unwrap();
    assert_eq!(&buf[..n], b"old contents");
    // The per-thread handle is opened after the rename and still names the old file.
    let mut sector = AlignedBuf::new(SECTOR);
    let n = file.read(0, &mut sector).unwrap();
    assert_eq!(&sector.as_slice()[..n], b"old contents");
}
