//! P13: open an archive, rename a new file over it with `std::fs::rename`, check the old handle
//! still reads the old bytes and a fresh open reads the new ones. Run per share mode and for an
//! unbuffered handle (the loading side's reader).

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fs::{File, OpenOptions};
use std::os::windows::fs::{FileExt, OpenOptionsExt};
use std::path::Path;

const SHARE_READ: u32 = 1;
const SHARE_WRITE: u32 = 2;
const SHARE_DELETE: u32 = 4;
const NO_BUFFERING: u32 = 0x2000_0000;
const LEN: usize = 1 << 20;

fn fill(tag: u8) -> Vec<u8> {
    (0..LEN).map(|i| (i as u8).wrapping_mul(31) ^ tag).collect()
}

struct Aligned(*mut u8, Layout);
impl Aligned {
    fn new(n: usize) -> Self {
        let l = Layout::from_size_align(n, 4096).unwrap();
        // SAFETY: non-zero size, valid layout.
        Aligned(unsafe { alloc_zeroed(l) }, l)
    }
    fn s(&mut self) -> &mut [u8] {
        // SAFETY: allocated with this size.
        unsafe { std::slice::from_raw_parts_mut(self.0, self.1.size()) }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout.
        unsafe { dealloc(self.0, self.1) }
    }
}

fn read_at(f: &File, off: u64, n: usize) -> std::io::Result<Vec<u8>> {
    let mut b = Aligned::new(n);
    let got = f.seek_read(b.s(), off)?;
    Ok(b.s()[..got].to_vec())
}

fn case(dir: &Path, name: &str, share: u32, flags: u32) {
    let path = dir.join(format!("{name}.orrpack"));
    let tmp = dir.join(format!("{name}.orrpack.tmp"));
    let _ = std::fs::remove_file(&path);
    std::fs::write(&path, fill(0xA0)).unwrap();
    let old = OpenOptions::new().read(true).share_mode(share).custom_flags(flags).open(&path).unwrap();
    let before = read_at(&old, 0, 4096).unwrap();
    assert_eq!(before, fill(0xA0)[..4096]);
    std::fs::write(&tmp, fill(0xB0)).unwrap();
    let r1 = std::fs::rename(&tmp, &path);
    let old_reads = read_at(&old, 512 * 1024, 4096).map(|v| v == fill(0xA0)[512 * 1024..512 * 1024 + 4096]);
    let fresh = std::fs::read(&path).map(|v| v == fill(0xB0)).unwrap_or(false);
    // A second cook while the first renderer handle and a handle on the new file stay open.
    let mid = OpenOptions::new().read(true).share_mode(share).custom_flags(flags).open(&path).ok();
    std::fs::write(&tmp, fill(0xC0)).unwrap();
    let r2 = std::fs::rename(&tmp, &path);
    let old_after2 = read_at(&old, 0, 4096).map(|v| v == fill(0xA0)[..4096]);
    let mid_after2 = mid.as_ref().map(|m| read_at(m, 0, 4096).map(|v| v == fill(0xB0)[..4096]).unwrap_or(false));
    let fresh2 = std::fs::read(&path).map(|v| v == fill(0xC0)).unwrap_or(false);
    drop(old);
    drop(mid);
    let leftovers: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().to_string_lossy().to_string();
            (n.starts_with(name) && n != format!("{name}.orrpack") && n != format!("{name}.orrpack.tmp")).then_some(n)
        })
        .collect();
    println!(
        "{name}: rename1 {:?}, old handle reads old bytes {:?}, fresh open new bytes {fresh}; rename2 {:?}, old handle {:?}, second handle {:?}, fresh {fresh2}; leftovers {leftovers:?}",
        r1.as_ref().map(|_| "ok").map_err(|e| e.to_string()),
        old_reads.map_err(|e| e.to_string()),
        r2.as_ref().map(|_| "ok").map_err(|e| e.to_string()),
        old_after2.map_err(|e| e.to_string()),
        mid_after2,
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&tmp);
}

pub fn run(a: &[String]) {
    let dir = Path::new(&a[0]);
    std::fs::create_dir_all(dir).unwrap();
    case(dir, "std_default", SHARE_READ | SHARE_WRITE | SHARE_DELETE, 0);
    case(dir, "read_delete", SHARE_READ | SHARE_DELETE, 0);
    case(dir, "unbuffered_read_delete", SHARE_READ | SHARE_DELETE, NO_BUFFERING);
    case(dir, "read_only_share", SHARE_READ, 0);
    case(dir, "unbuffered_read_only_share", SHARE_READ, NO_BUFFERING);
}
