//! GDeflate 1.0 pages through NVIDIA's libdeflate fork (`gdeflate` branch), compiled from the
//! checkout found at build time.
//!
//! Each page is one raw `libdeflate_gdeflate_compress_ex` page with no stream framing,
//! written into a buffer of [`page_bound`] bytes: the fork's Huffman path does not clamp its
//! writes to a shorter buffer.

use std::cell::RefCell;
use std::ffi::{c_int, c_void};
use std::ptr::NonNull;

use orr_pack::{Error, GDEFLATE_PAGE, GDeflate, Result};

/// The libdeflate commit the codec is built from.
pub const COMMIT: &str = env!("ORR_COOK_GDEFLATE");

#[repr(C)]
struct InPage {
    data: *const c_void,
    nbytes: usize,
}

#[repr(C)]
struct OutPage {
    data: *mut c_void,
    nbytes: usize,
}

unsafe extern "system" {
    fn libdeflate_alloc_gdeflate_compressor(level: c_int) -> *mut c_void;
    fn libdeflate_free_gdeflate_compressor(c: *mut c_void);
    fn libdeflate_gdeflate_compress_bound(
        c: *mut c_void,
        in_nbytes: usize,
        out_npages: *mut usize,
    ) -> usize;
    fn libdeflate_gdeflate_compress_ex(
        c: *mut c_void,
        in_pages: *const InPage,
        out_pages: *mut OutPage,
        npages: usize,
    ) -> usize;
    fn libdeflate_alloc_gdeflate_decompressor() -> *mut c_void;
    fn libdeflate_free_gdeflate_decompressor(d: *mut c_void);
    fn libdeflate_gdeflate_decompress(
        d: *mut c_void,
        in_pages: *mut InPage,
        in_npages: usize,
        out: *mut c_void,
        out_nbytes_avail: usize,
        actual_out_nbytes_ret: *mut usize,
    ) -> c_int;
}

/// Largest compressed page of at most 64 KiB, at any level: 65,964 bytes.
///
/// The fork's page bound adds its sub-stream packets to the DEFLATE bound, which the fork
/// raises by two more packets per sub-stream only where `deflate_compress.c` is built as
/// part of the GDeflate compressor. That copy is hidden, so the library links the plain
/// DEFLATE bound and reports 256 bytes less; the output buffer takes the larger figure.
pub fn page_bound() -> usize {
    // SAFETY: a null compressor asks for the bound across every level.
    let linked = unsafe {
        libdeflate_gdeflate_compress_bound(
            std::ptr::null_mut(),
            GDEFLATE_PAGE as usize,
            std::ptr::null_mut(),
        )
    };
    linked + 2 * SUB_STREAMS * PACKET_BYTES
}

/// GDeflate sub-streams and the bytes of one packet of each.
const SUB_STREAMS: usize = 32;
const PACKET_BYTES: usize = 4;

/// The fork's codec at one compression level (1 fastest to 12 smallest).
#[derive(Copy, Clone, Debug)]
pub struct Libdeflate {
    level: i32,
}

impl Libdeflate {
    pub fn new(level: i32) -> Result<Self> {
        if !(1..=12).contains(&level) {
            return Err(Error::Encode(format!(
                "GDeflate level {level} is outside 1 to 12"
            )));
        }
        Ok(Self { level })
    }

    pub fn level(&self) -> i32 {
        self.level
    }
}

/// A compressor or decompressor, freed on drop. Neither is safe to share between threads,
/// so each thread keeps its own.
struct State {
    ptr: NonNull<c_void>,
    free: unsafe extern "system" fn(*mut c_void),
}

impl Drop for State {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from the allocator `free` belongs to and is freed once.
        unsafe { (self.free)(self.ptr.as_ptr()) }
    }
}

thread_local! {
    /// Compressors by level (index 0 unused) and the decompressor.
    static STATES: RefCell<(Vec<Option<State>>, Option<State>)> =
        const { RefCell::new((Vec::new(), None)) };
}

fn out_of_memory() -> Error {
    Error::Encode("cannot allocate a GDeflate codec".into())
}

impl GDeflate for Libdeflate {
    fn compress(&self, page: &[u8], out: &mut Vec<u8>) -> Result<()> {
        if page.len() > GDEFLATE_PAGE as usize {
            return Err(Error::Encode(format!(
                "GDeflate page of {} bytes",
                page.len()
            )));
        }
        STATES.with_borrow_mut(|(compressors, _)| {
            let level = self.level as usize;
            if compressors.len() <= level {
                compressors.resize_with(level + 1, || None);
            }
            let c = match &mut compressors[level] {
                Some(c) => c,
                slot => {
                    // SAFETY: the level was checked to be in range.
                    let ptr = unsafe { libdeflate_alloc_gdeflate_compressor(self.level) };
                    slot.insert(State {
                        ptr: NonNull::new(ptr).ok_or_else(out_of_memory)?,
                        free: libdeflate_free_gdeflate_compressor,
                    })
                }
            };
            let bound = page_bound();
            let at = out.len();
            out.resize(at + bound, 0);
            let input = InPage {
                data: page.as_ptr().cast(),
                nbytes: page.len(),
            };
            let mut output = OutPage {
                data: out[at..].as_mut_ptr().cast(),
                nbytes: bound,
            };
            // SAFETY: one page of at most 64 KiB in, into a buffer of the bound's length; the
            // compressor is this thread's.
            let done =
                unsafe { libdeflate_gdeflate_compress_ex(c.ptr.as_ptr(), &input, &mut output, 1) };
            if done != 1 || output.nbytes == 0 || output.nbytes > bound {
                out.truncate(at);
                return Err(Error::Encode(format!(
                    "GDeflate level {} failed on a {} byte page",
                    self.level,
                    page.len()
                )));
            }
            out.truncate(at + output.nbytes);
            Ok(())
        })
    }

    fn decompress(&self, page: &[u8], dst: &mut [u8]) -> Result<()> {
        STATES.with_borrow_mut(|(_, decompressor)| {
            let d = match decompressor {
                Some(d) => d,
                slot => {
                    // SAFETY: takes no arguments.
                    let ptr = unsafe { libdeflate_alloc_gdeflate_decompressor() };
                    slot.insert(State {
                        ptr: NonNull::new(ptr).ok_or_else(out_of_memory)?,
                        free: libdeflate_free_gdeflate_decompressor,
                    })
                }
            };
            let mut input = InPage {
                data: page.as_ptr().cast(),
                nbytes: page.len(),
            };
            let mut written = 0;
            // SAFETY: one page in, `dst` as the output with its length; the decompressor is
            // this thread's.
            let r = unsafe {
                libdeflate_gdeflate_decompress(
                    d.ptr.as_ptr(),
                    &mut input,
                    1,
                    dst.as_mut_ptr().cast(),
                    dst.len(),
                    &mut written,
                )
            };
            if r != 0 || written != dst.len() {
                return Err(Error::Decode(format!(
                    "GDeflate page gave {written} of {} bytes (result {r})",
                    dst.len()
                )));
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(seed: u64, n: usize) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    fn round_trip(g: &Libdeflate, page: &[u8]) -> usize {
        let mut out = vec![7];
        g.compress(page, &mut out).unwrap();
        let stored = &out[1..];
        assert!(
            stored.len().is_multiple_of(4) && stored.len() <= page_bound(),
            "level {}: {} byte page stored in {} bytes (bound {})",
            g.level(),
            page.len(),
            stored.len(),
            page_bound()
        );
        let mut back = vec![0; page.len()];
        g.decompress(stored, &mut back).unwrap();
        assert!(back == page, "level {} page differs", g.level());
        stored.len()
    }

    #[test]
    fn pages_fit_the_bound_in_whole_packets_at_every_level() {
        let full = GDEFLATE_PAGE as usize;
        let text: Vec<u8> = b"the loading queue decompresses pages of 64 KiB. "
            .iter()
            .copied()
            .cycle()
            .take(full)
            .collect();
        for level in 1..=12 {
            let g = Libdeflate::new(level).unwrap();
            for seed in 0..4 {
                let stored = round_trip(&g, &random(seed + 1, full));
                assert!(stored > full, "random page compressed to {stored} bytes");
            }
            assert!(round_trip(&g, &vec![0; full]) < 1024);
            assert!(round_trip(&g, &text) < full / 8);
            round_trip(&g, &random(99, 4099));
            round_trip(&g, &text[..1]);
        }
    }

    #[test]
    fn wrong_lengths_fail() {
        let g = Libdeflate::new(6).unwrap();
        assert!(Libdeflate::new(0).is_err() && Libdeflate::new(13).is_err());
        assert!(g.compress(&vec![0; 65537], &mut Vec::new()).is_err());
        let mut out = Vec::new();
        g.compress(&[5; 1000], &mut out).unwrap();
        assert!(g.decompress(&out, &mut [0; 999]).is_err());
        assert!(g.decompress(&out, &mut [0; 1001]).is_err());
    }
}
