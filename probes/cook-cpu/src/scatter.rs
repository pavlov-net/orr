//! P9: count placement records in the scatter chunk files (postcard `Vec` of id u64, prototype u32,
//! layer u32 as varints, then position, rotation, scale as f32). Every file is decoded to its end,
//! so a count from the length prefix alone cannot pass on a misread layout. Far-field files
//! (derived per-chunk impostor batches) are counted separately.

use std::path::Path;

struct Rd<'a> {
    d: &'a [u8],
    p: usize,
}

impl Rd<'_> {
    fn varint(&mut self) -> u64 {
        let mut v = 0u64;
        let mut s = 0;
        loop {
            let b = self.d[self.p];
            self.p += 1;
            v |= u64::from(b & 0x7f) << s;
            if b & 0x80 == 0 {
                return v;
            }
            s += 7;
        }
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        let o = (0..n)
            .map(|i| f32::from_le_bytes(self.d[self.p + i * 4..self.p + i * 4 + 4].try_into().unwrap()))
            .collect();
        self.p += n * 4;
        o
    }
    fn byte(&mut self) -> u8 {
        self.p += 1;
        self.d[self.p - 1]
    }
}

pub fn run(a: &[String]) {
    let dir = Path::new(&a[0]);
    let (mut files, mut total, mut max, mut over_16k, mut bytes) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut far_files, mut far_total) = (0u64, 0u64);
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut protos = std::collections::BTreeMap::<u64, u64>::new();
    let mut non_unit_quat = 0u64;
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        let n = p.file_name().unwrap().to_string_lossy().to_string();
        if !n.ends_with(".bin") {
            continue;
        }
        let d = std::fs::read(&p).unwrap();
        let mut r = Rd { d: &d, p: 0 };
        if n.starts_with("chunk_") {
            let count = r.varint();
            for _ in 0..count {
                r.varint();
                *protos.entry(r.varint()).or_default() += 1;
                r.varint();
                let f = r.f32s(10);
                for k in 0..3 {
                    lo[k] = lo[k].min(f[k]);
                    hi[k] = hi[k].max(f[k]);
                }
                let q = f[3] * f[3] + f[4] * f[4] + f[5] * f[5] + f[6] * f[6];
                if (q - 1.0).abs() > 1e-3 {
                    non_unit_quat += 1;
                }
            }
            assert_eq!(r.p, d.len(), "{n}: trailing bytes");
            files += 1;
            total += count;
            max = max.max(count);
            over_16k += u64::from(count > 16384);
            bytes += d.len() as u64;
        } else if n.starts_with("far_") {
            for _ in 0..2 {
                let count = r.varint();
                for _ in 0..count {
                    r.f32s(5);
                    r.varint();
                    if r.byte() == 1 {
                        r.byte();
                        r.f32s(1);
                    }
                }
                r.f32s(6);
                far_total += count;
            }
            assert_eq!(r.p, d.len(), "{n}: trailing bytes");
            far_files += 1;
        }
    }
    println!("chunk files {files}, records {total}, max per file {max}, files over 16384 {over_16k}");
    println!("chunk bytes {bytes} ({:.1} B per record)", bytes as f64 / total as f64);
    println!("position range {lo:?} .. {hi:?} (world metres)");
    println!("prototypes {} distinct; non-unit quaternions {non_unit_quat}", protos.len());
    println!("far files {far_files}, far plants {far_total} (derived from chunk placements)");
}
