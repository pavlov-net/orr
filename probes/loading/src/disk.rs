//! Case 5: disk. Unbuffered reads of 64-256 KiB blobs at 4 KiB-aligned ranges from a
//! multi-GB archive in random order, IoRing (one thread, registered file and buffers) against
//! positioned reads (`FileExt::seek_read`) from worker threads, while the frame renders.

use crate::gpu::Gpu;
use crate::{Rng, Work, row, with_frame};
use std::alloc::{Layout, alloc_zeroed};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::windows::fs::{FileExt, OpenOptionsExt};
use std::os::windows::io::AsRawHandle;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use windows_sys::Win32::Storage::FileSystem as fs;

const MAGIC: u64 = 0x6f72_725f_7072_6231;
const PAGE: usize = 4096;
const MAX_BLOB: usize = 256 << 10;

fn archive_path() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("../../.tools/prb1/archive.bin")
}

/// Every 4 KiB page starts with its file offset xor MAGIC; the rest is PRNG data.
fn make_archive(path: &std::path::Path, bytes: u64) {
    if std::fs::metadata(path).is_ok_and(|m| m.len() == bytes) {
        return;
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let t0 = Instant::now();
    let mut f = std::io::BufWriter::with_capacity(8 << 20, File::create(path).unwrap());
    let mut r = Rng(1);
    let mut page = vec![0u8; PAGE];
    for p in 0..bytes / PAGE as u64 {
        for w in page.chunks_mut(8) {
            w.copy_from_slice(&r.next().to_le_bytes());
        }
        page[..8].copy_from_slice(&((p * PAGE as u64) ^ MAGIC).to_le_bytes());
        f.write_all(&page).unwrap();
    }
    f.into_inner().unwrap().sync_all().unwrap();
    row("archive written", format!("{} GiB in {:.1} s", bytes >> 30, t0.elapsed().as_secs_f64()));
}

/// Blobs of 64-256 KiB (4 KiB multiples) packed at 4 KiB alignment, shuffled.
fn blobs(bytes: u64) -> Vec<(u64, u32)> {
    let mut r = Rng(7);
    let mut v = Vec::new();
    let mut off = 0u64;
    loop {
        let len = (16 + r.below(49)) as u32 * PAGE as u32;
        if off + u64::from(len) > bytes {
            break;
        }
        v.push((off, len));
        off += u64::from(len);
    }
    for i in (1..v.len()).rev() {
        v.swap(i, r.below(i as u64 + 1) as usize);
    }
    v
}

fn check(buf: &[u8], off: u64) -> bool {
    buf.chunks(PAGE).enumerate().all(|(i, p)| u64::from_le_bytes(p[..8].try_into().unwrap()) == (off + (i * PAGE) as u64) ^ MAGIC)
}

fn aligned(len: usize) -> &'static mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(alloc_zeroed(Layout::from_size_align(len, PAGE).unwrap()), len) }
}

fn open(overlapped: bool) -> File {
    let mut flags = fs::FILE_FLAG_NO_BUFFERING;
    if overlapped {
        flags |= fs::FILE_FLAG_OVERLAPPED;
    }
    OpenOptions::new().read(true).custom_flags(flags).open(archive_path()).unwrap()
}

fn process_cpu() -> f64 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let mut t = [FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 }; 4];
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut t[0], &mut t[1], &mut t[2], &mut t[3]) };
    let f = |x: FILETIME| ((u64::from(x.dwHighDateTime) << 32) | u64::from(x.dwLowDateTime)) as f64 * 1e-7;
    f(t[2]) + f(t[3])
}

struct Reads {
    work: Work,
    process_cpu: f64,
    reads: u64,
    bad: u64,
    lat_us: Vec<f64>,
}

/// Positioned reads: `workers` threads, one blocking read each in flight, each on its own
/// unbuffered handle (`shared` puts every worker on one synchronous handle, which the I/O
/// manager serialises).
fn positioned(shared_file: &File, list: &[(u64, u32)], next: &AtomicUsize, workers: usize, shared: bool, dur: Duration) -> Reads {
    let p0 = process_cpu();
    let t0 = Instant::now();
    let parts: Vec<(f64, f64, u64, u64, Vec<f64>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| {
                    let own = (!shared).then(|| open(false));
                    let file = own.as_ref().unwrap_or(shared_file);
                    let buf = aligned(MAX_BLOB);
                    let c0 = crate::thread_cpu();
                    let (mut bytes, mut n, mut bad) = (0u64, 0u64, 0u64);
                    let mut lat = Vec::new();
                    while t0.elapsed() < dur {
                        let (off, len) = list[next.fetch_add(1, Ordering::Relaxed) % list.len()];
                        let t = Instant::now();
                        let got = file.seek_read(&mut buf[..len as usize], off).unwrap();
                        lat.push(t.elapsed().as_secs_f64() * 1e6);
                        if got != len as usize || !check(&buf[..len as usize], off) {
                            bad += 1;
                        }
                        bytes += got as u64;
                        n += 1;
                    }
                    (bytes as f64, crate::thread_cpu() - c0, n, bad, lat)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let wall = t0.elapsed().as_secs_f64();
    let mut r = Reads { work: Work { wall, ..Work::default() }, process_cpu: process_cpu() - p0, reads: 0, bad: 0, lat_us: Vec::new() };
    for (b, c, n, bad, lat) in parts {
        r.work.bytes += b;
        r.work.cpu += c;
        r.reads += n;
        r.bad += bad;
        r.lat_us.extend(lat);
    }
    r
}

fn hr(what: &str, h: i32) {
    if h < 0 {
        crate::fail(&format!("{what} failed: HRESULT {h:#x}"));
    }
}

/// IoRing reads from one thread with `depth` in flight. `registered` uses a registered file
/// handle and registered buffers; otherwise raw handle and buffer addresses.
fn ioring(file: &File, list: &[(u64, u32)], next: &AtomicUsize, depth: usize, registered: bool, dur: Duration) -> Reads {
    unsafe {
        let mut ring: fs::HIORING = std::ptr::null_mut();
        let flags = fs::IORING_CREATE_FLAGS { Required: fs::IORING_CREATE_REQUIRED_FLAGS_NONE, Advisory: fs::IORING_CREATE_ADVISORY_FLAGS_NONE };
        hr("CreateIoRing", fs::CreateIoRing(fs::IORING_VERSION_3, flags, 256, 512, &mut ring));
        let handle = file.as_raw_handle();
        let bufs: Vec<&'static mut [u8]> = (0..depth).map(|_| aligned(MAX_BLOB)).collect();
        let mut cqe = fs::IORING_CQE::default();
        if registered {
            // Registration runs at submit: the arrays must outlive it.
            let handles = [handle as windows_sys::Win32::Foundation::HANDLE];
            hr("register files", fs::BuildIoRingRegisterFileHandles(ring, 1, handles.as_ptr(), 0));
            let info: Vec<fs::IORING_BUFFER_INFO> = bufs.iter().map(|b| fs::IORING_BUFFER_INFO { Address: b.as_ptr() as *mut _, Length: MAX_BLOB as u32 }).collect();
            hr("register buffers", fs::BuildIoRingRegisterBuffers(ring, depth as u32, info.as_ptr(), 0));
            let mut submitted = 0;
            hr("submit registration", fs::SubmitIoRing(ring, 2, u32::MAX, &mut submitted));
            while fs::PopIoRingCompletion(ring, &mut cqe) == 0 {
                hr("registration completion", cqe.ResultCode);
            }
        }
        let file_ref = if registered {
            fs::IORING_HANDLE_REF { Kind: fs::IORING_REF_REGISTERED, Handle: fs::IORING_HANDLE_REF_0 { Index: 0 } }
        } else {
            fs::IORING_HANDLE_REF { Kind: fs::IORING_REF_RAW, Handle: fs::IORING_HANDLE_REF_0 { Handle: handle as _ } }
        };
        let mut inflight: Vec<(u64, u32, Instant)> = vec![(0, 0, Instant::now()); depth];
        let p0 = process_cpu();
        let c0 = crate::thread_cpu();
        let t0 = Instant::now();
        let issue = |slot: usize, inflight: &mut Vec<(u64, u32, Instant)>| {
            let (off, len) = list[next.fetch_add(1, Ordering::Relaxed) % list.len()];
            let buf_ref = if registered {
                fs::IORING_BUFFER_REF { Kind: fs::IORING_REF_REGISTERED, Buffer: fs::IORING_BUFFER_REF_0 { IndexAndOffset: fs::IORING_REGISTERED_BUFFER { BufferIndex: slot as u32, Offset: 0 } } }
            } else {
                fs::IORING_BUFFER_REF { Kind: fs::IORING_REF_RAW, Buffer: fs::IORING_BUFFER_REF_0 { Address: bufs[slot].as_ptr() as *mut _ } }
            };
            hr("BuildIoRingReadFile", fs::BuildIoRingReadFile(ring, file_ref, buf_ref, len, off, slot, fs::IOSQE_FLAGS_NONE));
            inflight[slot] = (off, len, Instant::now());
        };
        for s in 0..depth {
            issue(s, &mut inflight);
        }
        let (mut bytes, mut n, mut bad, mut live) = (0u64, 0u64, 0u64, depth);
        let mut lat = Vec::new();
        let mut submitted = 0;
        while live > 0 {
            hr("SubmitIoRing", fs::SubmitIoRing(ring, 1, u32::MAX, &mut submitted));
            while fs::PopIoRingCompletion(ring, &mut cqe) == 0 {
                let slot = cqe.UserData;
                let (off, len, t) = inflight[slot];
                lat.push(t.elapsed().as_secs_f64() * 1e6);
                if cqe.ResultCode < 0 || cqe.Information != len as usize || !check(&bufs[slot][..len as usize], off) {
                    if bad == 0 {
                        eprintln!("first bad IoRing read: hr {:#x} info {} len {len} off {off}", cqe.ResultCode, cqe.Information);
                    }
                    bad += 1;
                }
                bytes += cqe.Information as u64;
                n += 1;
                if t0.elapsed() < dur {
                    issue(slot, &mut inflight);
                } else {
                    live -= 1;
                }
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        let r = Reads {
            work: Work { bytes: bytes as f64, cpu: crate::thread_cpu() - c0, wall },
            process_cpu: process_cpu() - p0,
            reads: n,
            bad,
            lat_us: lat,
        };
        fs::CloseIoRing(ring);
        r
    }
}

/// BypassIO state of the archive handle: query, then enable for this handle.
fn bypass_io(file: &File) -> String {
    use windows_sys::Win32::System::IO::DeviceIoControl;
    use windows_sys::Win32::System::Ioctl as io;
    /// FS_BPIO_INPUT (ntifs.h).
    #[repr(C)]
    struct BpioInput {
        operation: i32,
        in_flags: i32,
        reserved1: u64,
        reserved2: u64,
    }
    let mut out = String::new();
    for (name, op) in [("query", io::FS_BPIO_OP_QUERY), ("enable", io::FS_BPIO_OP_ENABLE)] {
        let input = BpioInput { operation: op, in_flags: io::FSBPIO_INFL_None, reserved1: 0, reserved2: 0 };
        let mut o: io::FS_BPIO_OUTPUT = unsafe { std::mem::zeroed() };
        let mut ret = 0u32;
        let ok = unsafe {
            DeviceIoControl(
                file.as_raw_handle() as _,
                io::FSCTL_MANAGE_BYPASS_IO,
                (&raw const input).cast(),
                std::mem::size_of_val(&input) as u32,
                (&raw mut o).cast(),
                std::mem::size_of_val(&o) as u32,
                &mut ret,
                std::ptr::null_mut(),
            )
        };
        let res = unsafe { o.Anonymous.Query };
        let reason = String::from_utf16_lossy(&res.FailureReason[..(res.FailureReasonLen as usize / 2).min(128)]);
        let driver = String::from_utf16_lossy(&res.FailingDriverName[..(res.FailingDriverNameLen as usize / 2).min(32)]);
        out += &format!(
            "{name}: ioctl {} status {:#x}{}{}; ",
            if ok != 0 { "ok" } else { "failed" },
            res.OpStatus,
            if driver.is_empty() { String::new() } else { format!(" driver {driver}") },
            if reason.is_empty() { String::new() } else { format!(" reason \"{reason}\"") }
        );
    }
    out
}

fn lat_text(l: &mut [f64]) -> String {
    format!("latency p50 {:.0} us p99 {:.0} us", crate::pct(l, 0.5), crate::pct(l, 0.99))
}

pub unsafe fn run(gpu: &'static Gpu, args: &[String]) {
    let gib: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(16);
    let bytes = gib << 30;
    let path = archive_path();
    make_archive(&path, bytes);
    let list = blobs(bytes);
    row("archive", format!("{} ({gib} GiB), {} blobs of 64-256 KiB, random order", path.display(), list.len()));
    let mut caps = fs::IORING_CAPABILITIES::default();
    fs::QueryIoRingCapabilities(&mut caps);
    row(
        "IoRing capabilities",
        format!(
            "max version {}, sq {}, cq {}, feature flags {:#x} (user-mode emulation: {})",
            caps.MaxVersion,
            caps.MaxSubmissionQueueSize,
            caps.MaxCompletionQueueSize,
            caps.FeatureFlags,
            caps.FeatureFlags & fs::IORING_FEATURE_UM_EMULATION != 0
        ),
    );
    let sync = open(false);
    let ovl = open(true);
    row("BypassIO (positioned handle)", bypass_io(&sync));
    row("BypassIO (IoRing handle)", bypass_io(&ovl));
    let next = AtomicUsize::new(0);
    let list = &list;
    let next = &next;
    let sync = &sync;
    let ovl = &ovl;
    with_frame(gpu, |frame| {
        type Reader<'a> = Box<dyn Fn(Duration) -> Reads + 'a>;
        let readers: Vec<(String, Reader)> = vec![
            ("IoRing registered, 64 in flight".into(), Box::new(move |d| ioring(ovl, list, next, 64, true, d))),
            ("IoRing raw, 64 in flight".into(), Box::new(move |d| ioring(ovl, list, next, 64, false, d))),
            ("IoRing registered, 8 in flight".into(), Box::new(move |d| ioring(ovl, list, next, 8, true, d))),
            ("positioned reads, 64 workers, handle per worker".into(), Box::new(move |d| positioned(sync, list, next, 64, false, d))),
            ("positioned reads, 8 workers, handle per worker".into(), Box::new(move |d| positioned(sync, list, next, 8, false, d))),
            ("positioned reads, 64 workers, one shared handle".into(), Box::new(move |d| positioned(sync, list, next, 64, true, d))),
            ("IoRing registered, 64 in flight, synchronous handle".into(), Box::new(move |d| ioring(sync, list, next, 64, true, d))),
        ];
        for (name, read) in &readers {
            let (delta, rs) = frame.ab(3, Duration::from_millis(2000), |d| read(d));
            let w = Work::sum(&rs.iter().map(|r| r.work).collect::<Vec<_>>());
            let pc: f64 = rs.iter().map(|r| r.process_cpu).sum();
            let n: u64 = rs.iter().map(|r| r.reads).sum();
            let bad: u64 = rs.iter().map(|r| r.bad).sum();
            let mut lat: Vec<f64> = rs.iter().flat_map(|r| r.lat_us.iter().copied()).collect();
            row(
                name,
                format!(
                    "{:.2} GB/s, reader threads {:.2} cores, process {:.2} cores; {n} reads, {bad} bad; {}; {}",
                    w.gbps(),
                    w.cores(),
                    pc / w.wall,
                    lat_text(&mut lat),
                    delta.text()
                ),
            );
            if bad > 0 {
                crate::fail("a read returned wrong bytes");
            }
        }
    });
}
