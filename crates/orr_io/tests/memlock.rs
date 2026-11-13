//! The io_uring ring with its buffers registered, then naming them by address once the memory
//! lock limit refuses their registration. A binary of its own, as the limit is the process's.
#![cfg(target_os = "linux")]

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use orr_io::{Read, ReadBuffers, ReadFile, Ring, SECTOR, Wake};

fn bytes(len: usize) -> Vec<u8> {
    let mut x = 0x2545_f491_4f6c_dd1d_u64;
    Vec::from_fn(len, |_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x as u8
    })
}

/// Reads into the arena, into memory of their own once it is full and past the end of the
/// file match the file.
fn reads_match(mut ring: Ring, file: &Arc<ReadFile>, data: &[u8]) {
    let reads: Vec<(usize, usize)> =
        Vec::from_fn(100, |i| (i * 3 % 42 * SECTOR, 1 + i * 1777 % (6 * SECTOR)));
    let got = Arc::new(Mutex::new(vec![None; reads.len()]));
    for (i, &(offset, len)) in reads.iter().enumerate() {
        let got = Arc::clone(&got);
        ring.push(Read {
            file: Arc::clone(file),
            offset: offset as u64,
            len,
            into: None,
            done: Box::new(move |r| {
                got.lock().unwrap()[i] = Some(r.expect("ring read").as_slice().to_vec());
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
    for (i, (&(offset, len), got)) in reads.iter().zip(got.lock().unwrap().iter()).enumerate() {
        let start = offset.min(data.len());
        let want = &data[start..(offset + len).min(data.len())];
        assert!(
            got.as_deref() == Some(want),
            "read {i} at {offset} of {len}"
        );
    }
}

#[test]
fn a_ring_reads_into_registered_buffers_and_by_address_past_the_memory_lock_limit() {
    let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("memlock.bin");
    let data = bytes(40 * SECTOR + 77);
    fs::write(&p, &data).unwrap();
    let file = Arc::new(ReadFile::open(&p).unwrap());
    // Past the common limit of 8 MiB, so a limit below it registers a prefix.
    let buffers = ReadBuffers::new(48 << 20).unwrap();
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is written only during the call.
    let got = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &raw mut limit) };
    assert_eq!(got, 0);
    let Some(ring) = Ring::native(&buffers) else {
        #[allow(clippy::print_stderr)]
        {
            eprintln!("skipping: the system has no io_uring");
        }
        return;
    };
    // The limit counts every process of the user, so only one well above the buffers is sure
    // to take them all; reads stay in the registered prefix, where the arena's first ranges
    // are.
    let registered = ring.registered_bytes();
    if limit.rlim_cur >= 8 * buffers.bytes() as u64 {
        assert_eq!(registered, buffers.bytes(), "registered under {limit:?}");
    } else if limit.rlim_cur >= 16 << 20 {
        assert!(registered > 0, "nothing registered under {limit:?}");
    }
    if limit.rlim_cur < buffers.bytes() as u64 && !locks_past_the_limit() {
        assert!(
            registered < buffers.bytes(),
            "all registered under {limit:?}"
        );
    }
    reads_match(ring, &file, &data);

    limit.rlim_cur = 0;
    // SAFETY: lowers the soft limit only, which needs no privilege.
    let set = unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &raw const limit) };
    assert_eq!(set, 0);
    // Kernels before 5.12 charge the ring itself to the limit.
    let Some(ring) = Ring::native(&buffers) else {
        #[allow(clippy::print_stderr)]
        {
            eprintln!("skipping the by-address reads: the kernel makes no ring under the limit");
        }
        return;
    };
    if !locks_past_the_limit() {
        assert_eq!(ring.registered_bytes(), 0, "registered past the limit");
    }
    reads_match(ring, &file, &data);
}

/// Whether the process has `CAP_IPC_LOCK`, which locks memory past the limit.
fn locks_past_the_limit() -> bool {
    const CAP_IPC_LOCK: u32 = 14;
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let caps = status
        .lines()
        .find_map(|l| l.strip_prefix("CapEff:"))
        .expect("effective capabilities");
    u64::from_str_radix(caps.trim(), 16).unwrap() & 1 << CAP_IPC_LOCK != 0
}
