use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    // A file that ends on a sector takes one more read to find its end.
    for len in [37 * SECTOR + 123, 37 * SECTOR] {
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
        match Ring::native() {
            Some(r) => rings.push(("native", r)),
            #[allow(clippy::print_stderr)]
            None => eprintln!("skipping the native ring: the system has none"),
        }
        for (name, ring) in rings {
            let got = run_ring(ring, &file, &reads);
            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert!(g == e, "{name} ring, read {i} {:?}", reads[i]);
            }
        }
    }
}

#[test]
fn ring_finishes_when_every_read_completes_before_the_wait() {
    let p = path("drain.bin");
    fs::write(&p, bytes(SECTOR)).unwrap();
    let file = Arc::new(ReadFile::open(&p).unwrap());
    let Some(mut ring) = Ring::native() else {
        return;
    };
    // More reads than the ring keeps in flight.
    for _ in 0..200 {
        ring.push(Read {
            file: Arc::clone(&file),
            offset: 0,
            len: SECTOR,
            done: Box::new(|r| drop(r.expect("ring read"))),
        });
    }
    let wake = Wake::default();
    let finished = AtomicBool::new(false);
    let limit = Duration::from_secs(10);
    let start = Instant::now();
    std::thread::scope(|s| {
        // Ends a wait that nothing else would, so that a stuck ring fails the test.
        s.spawn(|| {
            while !finished.load(Ordering::Relaxed) && start.elapsed() < limit {
                std::thread::sleep(Duration::from_millis(10));
            }
            wake.set();
        });
        loop {
            ring.pump();
            if !ring.busy() {
                break;
            }
            // Everything in flight finishes here, so the wait takes the event for all of it.
            std::thread::sleep(Duration::from_millis(20));
            wake.wait(Some(&ring));
            assert!(start.elapsed() < limit, "the ring stopped reading");
        }
        finished.store(true, Ordering::Relaxed);
    });
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
fn reads_on_many_threads_match_the_file() {
    // A file that ends on a sector takes one more read to find its end.
    for len in [64 * SECTOR + 123, 64 * SECTOR] {
        let data = bytes(len);
        let p = path("threads.bin");
        fs::write(&p, &data).unwrap();
        let file = ReadFile::open(&p).unwrap();
        std::thread::scope(|s| {
            for t in 0..8 {
                let (file, data) = (&file, &data);
                s.spawn(move || {
                    let mut buf = AlignedBuf::new(3 * SECTOR);
                    for i in 0..200 {
                        // Every thread reaches the last sectors and the sector past the end.
                        let offset = (i * 7 + t) % 66 * SECTOR;
                        let n = file.read(offset as u64, &mut buf).unwrap();
                        let start = offset.min(len);
                        let end = (offset + 3 * SECTOR).min(len);
                        assert_eq!(n, end - start, "offset {offset}");
                        assert!(buf.as_slice()[..n] == data[start..end], "offset {offset}");
                    }
                });
            }
        });
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
    let mut sector = AlignedBuf::new(SECTOR);
    let n = file.read(0, &mut sector).unwrap();
    assert_eq!(&sector.as_slice()[..n], b"old contents");
    for ring in [Some(Ring::positioned()), Ring::native()]
        .into_iter()
        .flatten()
    {
        assert_eq!(run_ring(ring, &file, &[(0, SECTOR)]), [b"old contents"]);
    }
}

#[test]
fn handles_opened_together_name_one_file_while_the_path_is_replaced() {
    let p = path("together.bin");
    let new = path("together.new");
    fs::write(&p, [0; SECTOR]).unwrap();
    std::thread::scope(|s| {
        let reader = s.spawn(|| {
            for _ in 0..2000 {
                let (file, shared) = ReadFile::open_with_shared(&p).unwrap();
                let mut sector = AlignedBuf::new(SECTOR);
                file.read(0, &mut sector).unwrap();
                let mut byte = [0];
                read_at(&shared, 0, &mut byte).unwrap();
                assert_eq!(sector.as_slice()[0], byte[0]);
            }
        });
        let mut i = 0_u8;
        while !reader.is_finished() {
            i = i.wrapping_add(1);
            fs::write(&new, [i; SECTOR]).unwrap();
            fs::rename(&new, &p).unwrap();
        }
    });
}
