//! Hash maps and sets of integer keys with a multiply-rotate hasher: the mesh cook's
//! lookups by vertex, grid cell and edge, where SipHash's cost shows.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub(crate) struct Mix(u64);

impl Mix {
    fn add(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(26) ^ v).wrapping_mul(0xf135_7aea_2e62_a9c5);
    }
}

impl Hasher for Mix {
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }

    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            self.add(u64::from_le_bytes(b));
        }
    }

    fn write_u8(&mut self, v: u8) {
        self.add(u64::from(v));
    }

    fn write_u32(&mut self, v: u32) {
        self.add(u64::from(v));
    }

    fn write_i32(&mut self, v: i32) {
        self.add(u64::from(v as u32));
    }

    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }

    fn write_usize(&mut self, v: usize) {
        self.add(v as u64);
    }
}

pub(crate) type Map<K, V> = HashMap<K, V, BuildHasherDefault<Mix>>;
pub(crate) type Set<K> = HashSet<K, BuildHasherDefault<Mix>>;
