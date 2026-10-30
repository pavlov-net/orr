//! P10: repeated geometry across the tree set. Two units are tested:
//! - connected components of each section (triangles joined through shared positions);
//! - rigid parts of the wind builds (triangles by `_ID` joint).
//!
//! A unit's signature hashes its local topology (vertices renumbered by first use), its UVs
//! (1/4096) and a shape descriptor invariant to rotation, translation and uniform scale
//! (distances to the centroid and to the first vertex over the RMS radius, 1/1024). Repeats
//! are every copy after the first of a signature, across the whole set. Bytes use the
//! resident rate: 18 B per vertex + 3 B per triangle.

use crate::gl::{self, Prim};
use meshopt::ffi;
use std::collections::HashMap;
use std::path::Path;

fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

fn signature(p: &Prim, tris: &[u32]) -> ([u8; 32], usize, usize) {
    let mut local: HashMap<u32, u32> = HashMap::new();
    let mut order = Vec::new();
    let mut topo = Vec::with_capacity(tris.len());
    for &t in tris {
        for &v in &p.idx[t as usize * 3..t as usize * 3 + 3] {
            let n = *local.entry(v).or_insert_with(|| {
                order.push(v);
                (order.len() - 1) as u32
            });
            topo.push(n);
        }
    }
    let pos: Vec<[f64; 3]> = order.iter().map(|&v| p.pos[v as usize].map(f64::from)).collect();
    let n = pos.len() as f64;
    let c = [0, 1, 2].map(|k| pos.iter().map(|q| q[k]).sum::<f64>() / n);
    let d = |a: &[f64; 3], b: &[f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let r = (pos.iter().map(|q| d(q, &c).powi(2)).sum::<f64>() / n).sqrt().max(1e-12);
    let mut h = blake3::Hasher::new();
    for t in &topo {
        h.update(&t.to_le_bytes());
    }
    for (i, q) in pos.iter().enumerate() {
        let uv = p.uv.get(order[i] as usize).copied().unwrap_or([0.0; 2]);
        for x in [uv[0] as f64 * 4096.0, uv[1] as f64 * 4096.0, d(q, &c) / r * 1024.0, d(q, &pos[0]) / r * 1024.0] {
            h.update(&(x.round() as i64).to_le_bytes());
        }
    }
    (*h.finalize().as_bytes(), order.len(), tris.len())
}

#[derive(Default)]
struct Acc {
    seen: HashMap<[u8; 32], usize>,
    total: usize,
    total_tris: usize,
    dup: [usize; 3], // bytes repeated in units of <32, 32..127, >=128 triangles
    dup_tris: [usize; 3],
    units: [usize; 3],
}

impl Acc {
    fn add(&mut self, sig: ([u8; 32], usize, usize)) {
        let bytes = sig.1 * 18 + sig.2 * 3;
        let b = if sig.2 < 32 { 0 } else if sig.2 < 128 { 1 } else { 2 };
        self.total += bytes;
        self.total_tris += sig.2;
        self.units[b] += 1;
        let c = self.seen.entry(sig.0).or_default();
        if *c > 0 {
            self.dup[b] += bytes;
            self.dup_tris[b] += sig.2;
        }
        *c += 1;
    }
    fn report(&self, what: &str) {
        let pct = |x: usize| x as f64 / self.total as f64 * 100.0;
        println!(
            "{what}: {} units ({} under 32 tris, {} 32-127, {} 128+), {} distinct; {:.1} MB, {} tris",
            self.units.iter().sum::<usize>(),
            self.units[0],
            self.units[1],
            self.units[2],
            self.seen.len(),
            self.total as f64 / 1e6,
            self.total_tris
        );
        println!(
            "  repeated bytes: {:.2}% total; units under 32 tris {:.2}%, 32-127 {:.2}%, 128+ {:.2}%",
            pct(self.dup.iter().sum()),
            pct(self.dup[0]),
            pct(self.dup[1]),
            pct(self.dup[2])
        );
    }
}

pub fn run(a: &[String]) {
    let mut comp = Acc::default();
    for g in crate::codec::tree_glbs(Path::new(&a[0])) {
        let d = gl::open(&g);
        for (node, mesh, m) in d.mesh_nodes() {
            for p in d.prims(mesh, &m, d.node_name(node)) {
                let mut canon = vec![0u32; p.pos.len()];
                // SAFETY: one entry per vertex; 3 floats per position.
                unsafe { ffi::meshopt_generatePositionRemap(canon.as_mut_ptr(), p.pos.as_ptr().cast(), p.pos.len(), 12) };
                let mut parent: Vec<u32> = (0..p.pos.len() as u32).collect();
                for t in p.idx.chunks(3) {
                    let a0 = find(&mut parent, canon[t[0] as usize]);
                    for &v in &t[1..] {
                        let b = find(&mut parent, canon[v as usize]);
                        parent[b as usize] = a0;
                    }
                }
                let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
                for (ti, t) in p.idx.chunks(3).enumerate() {
                    let r = find(&mut parent, canon[t[0] as usize]);
                    groups.entry(r).or_default().push(ti as u32);
                }
                for tris in groups.values() {
                    comp.add(signature(&p, tris));
                }
            }
        }
        d.clear_cache();
    }
    comp.report("connected components, tree set (all variants and glTF LODs)");
    if let Some(wind) = a.get(1) {
        let mut parts = Acc::default();
        for g in crate::codec::tree_glbs(Path::new(wind)) {
            let d = gl::open(&g);
            for (node, mesh, m) in d.mesh_nodes() {
                for p in d.prims(mesh, &m, d.node_name(node)) {
                    let mut by: HashMap<u32, Vec<u32>> = HashMap::new();
                    for (ti, t) in p.idx.chunks(3).enumerate() {
                        let ids: Vec<u32> = t.iter().map(|&v| p.id[v as usize]).collect();
                        by.entry(if ids[1] == ids[2] { ids[1] } else { ids[0] }).or_default().push(ti as u32);
                    }
                    for tris in by.values() {
                        parts.add(signature(&p, tris));
                    }
                }
            }
        }
        parts.report("rigid parts (joint, section), wind builds");
    }
}
