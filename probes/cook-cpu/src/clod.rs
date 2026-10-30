//! P2 and the hierarchy-bytes question: `meshopt::ffi::clodBuild` per section, dilation on
//! double-sided sections, positions copied per cluster in the output callback.
//!
//! Checks: source clusters' copies equal the input bit for bit; how many copies differ from
//! the final (dilated) position buffer; cuts at many error thresholds selected by the
//! clusterlod rule never give one source position two different copies (a crack).

use crate::gl::{self, Prim};
use meshopt::ffi;
use std::collections::HashMap;
use std::os::raw::{c_int, c_void};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

pub const NODE_BYTES: usize = 32; // orr_pack Node
const CLUSTER_BYTES: usize = 128;
const VERTEX_BYTES: usize = 18; // u16x3 position + oct normal + oct tangent + uv16x2
const TRI_BYTES: usize = 3;

#[derive(Clone, Copy)]
struct Cl {
    group: u32,
    refined: i32,
    first: u32, // into verts / copies
    nverts: u32,
    first_tri: u32, // into tris (bytes, 3 per triangle)
    ntris: u32,
}

#[derive(Default)]
struct Out {
    groups: Vec<ffi::clodGroup>,
    clusters: Vec<Cl>,
    verts: Vec<u32>,
    copies: Vec<[f32; 3]>,
    tris: Vec<u8>,
    pos: *const f32,
    stride: usize,
}

unsafe extern "C" fn emit(ctx: *mut c_void, group: ffi::clodGroup, cl: *const ffi::clodCluster, n: usize) -> c_int {
    // SAFETY: ctx is the Out passed to clodBuild; cl points at n clusters for this call.
    let o = unsafe { &mut *ctx.cast::<Out>() };
    let cl = unsafe { std::slice::from_raw_parts(cl, n) };
    let gid = o.groups.len() as u32;
    o.groups.push(group);
    let mut lv = [0u32; 256];
    let mut lt = [0u8; 256 * 3];
    for c in cl {
        // SAFETY: indices holds index_count entries; local buffers fit 256 vertices / triangles.
        let nv = unsafe { ffi::clodLocalIndices(lv.as_mut_ptr(), lt.as_mut_ptr(), c.indices, c.index_count) };
        let first = o.verts.len() as u32;
        let first_tri = o.tris.len() as u32;
        for &v in &lv[..nv] {
            o.verts.push(v);
            // SAFETY: v < vertex_count; pos is the live (possibly dilated) position buffer.
            let p = unsafe { o.pos.add(v as usize * o.stride) };
            o.copies.push(unsafe { [*p, *p.add(1), *p.add(2)] });
        }
        o.tris.extend_from_slice(&lt[..c.index_count]);
        o.clusters.push(Cl {
            group: gid,
            refined: c.refined,
            first,
            nverts: nv as u32,
            first_tri,
            ntris: (c.index_count / 3) as u32,
        });
    }
    gid as c_int
}

pub struct Built {
    pub secs: f64,
    pub tris: usize,
    pub groups: usize,
    pub clusters: usize,
    pub levels: usize,
    pub tris_per_level: Vec<usize>,
    pub nodes: usize,
    pub root_bytes: usize,
    pub dag_bytes: usize,
    pub root_tris: usize,
    pub lines: Vec<String>,
}

pub fn config(double_sided: bool) -> ffi::clodConfig {
    // SAFETY: plain value constructor.
    let mut c = unsafe { ffi::clodDefaultConfigRT(128) };
    c.max_vertices = if double_sided { 256 } else { 128 };
    c.partition_sort = true;
    c.simplify_error_clamped = true;
    c.simplify_dilate_borders = double_sided;
    c
}

/// Attributes for permissive simplification: normal (weight 0.5) and uv (weight 1, protected).
fn attributes(p: &Prim) -> Vec<f32> {
    let mut a = Vec::with_capacity(p.pos.len() * 5);
    for i in 0..p.pos.len() {
        let n = p.nrm.get(i).copied().unwrap_or([0.0, 0.0, 1.0]);
        let t = p.uv.get(i).copied().unwrap_or([0.0, 0.0]);
        a.extend_from_slice(&[n[0], n[1], n[2], t[0], t[1]]);
    }
    a
}

pub fn build(p: &Prim, cfg: ffi::clodConfig, lock: Option<&[u8]>, check: bool) -> Built {
    let original: Vec<f32> = p.pos.iter().flatten().copied().collect();
    let mut live = original.clone();
    let attrs = attributes(p);
    let weights = [0.5f32, 0.5, 0.5, 1.0, 1.0];
    let mesh = ffi::clodMesh {
        indices: p.idx.as_ptr(),
        index_count: p.idx.len(),
        vertex_count: p.pos.len(),
        vertex_positions: live.as_mut_ptr(),
        vertex_positions_stride: 12,
        vertex_attributes: attrs.as_ptr(),
        vertex_attributes_stride: 20,
        vertex_lock: lock.map_or(std::ptr::null(), <[u8]>::as_ptr),
        attribute_weights: weights.as_ptr(),
        attribute_count: 5,
        attribute_protect_mask: (1 << 3) | (1 << 4),
    };
    let mut o = Out { pos: live.as_ptr(), stride: 3, ..Default::default() };
    let t = Instant::now();
    // SAFETY: mesh points at live buffers that outlive the call; emit matches clodOutput.
    // clodBuild writes dilated positions into `live` through vertex_positions.
    unsafe { ffi::clodBuild(cfg, mesh, (&raw mut o).cast(), Some(emit)) };
    let secs = crate::secs(t);
    let levels = o.groups.iter().map(|g| g.depth as usize + 1).max().unwrap_or(0);
    let mut tris_per_level = vec![0usize; levels];
    for c in &o.clusters {
        tris_per_level[o.groups[c.group as usize].depth as usize] += c.ntris as usize;
    }
    // SAFETY: bound and build over the groups clodBuild emitted.
    let bound = unsafe { ffi::clodBuildHierarchyBound(o.groups.len(), 8, levels) };
    let mut nodes: Vec<ffi::clodNode> = vec![unsafe { std::mem::zeroed() }; bound];
    let nodes_n = unsafe { ffi::clodBuildHierarchy(nodes.as_mut_ptr(), o.groups.as_ptr(), o.groups.len(), 8, levels) };
    let terminal = |c: &Cl| o.groups[c.group as usize].simplified.error == f32::MAX;
    let bytes = |c: &Cl| CLUSTER_BYTES + c.nverts as usize * VERTEX_BYTES + (c.ntris as usize * TRI_BYTES).next_multiple_of(4);
    let root_bytes = o.clusters.iter().filter(|c| terminal(c)).map(bytes).sum();
    let root_tris = o.clusters.iter().filter(|c| terminal(c)).map(|c| c.ntris as usize).sum();
    let dag_bytes = o.clusters.iter().map(bytes).sum();
    let mut lines = Vec::new();
    if check {
        lines = checks(p, &original, &live, &o);
    }
    Built {
        secs,
        tris: p.tris(),
        groups: o.groups.len(),
        clusters: o.clusters.len(),
        levels,
        tris_per_level,
        nodes: nodes_n,
        root_bytes,
        dag_bytes,
        root_tris,
        lines,
    }
}

fn checks(p: &Prim, original: &[f32], live: &[f32], o: &Out) -> Vec<String> {
    let mut out = Vec::new();
    let nan = o.copies.iter().flatten().filter(|x| !x.is_finite()).count();
    let (mut src_diff, mut src_n, mut final_diff) = (0usize, 0usize, 0usize);
    for c in &o.clusters {
        for k in c.first..c.first + c.nverts {
            let v = o.verts[k as usize] as usize;
            let cp = o.copies[k as usize];
            let orig = [original[v * 3], original[v * 3 + 1], original[v * 3 + 2]];
            let fin = [live[v * 3], live[v * 3 + 1], live[v * 3 + 2]];
            if c.refined == -1 {
                src_n += 1;
                src_diff += usize::from(cp.map(f32::to_bits) != orig.map(f32::to_bits));
            }
            final_diff += usize::from(cp.map(f32::to_bits) != fin.map(f32::to_bits));
        }
    }
    let moved = (0..p.pos.len())
        .filter(|&v| (0..3).any(|k| original[v * 3 + k].to_bits() != live[v * 3 + k].to_bits()))
        .count();
    out.push(format!(
        "  copies: {} vertex copies, non-finite {nan}; source-cluster copies differing from input {src_diff}/{src_n}; copies differing from the final buffer {final_diff}; vertices the final buffer moved {moved}/{}",
        o.copies.len(),
        p.pos.len()
    ));
    // Position remap: wedges at one source position share a canonical id.
    let mut canon = vec![0u32; p.pos.len()];
    // SAFETY: canon has vertex_count entries; positions are 3 floats per vertex.
    unsafe { ffi::meshopt_generatePositionRemap(canon.as_mut_ptr(), original.as_ptr(), p.pos.len(), 12) };
    let errs: Vec<f32> = o.groups.iter().map(|g| g.simplified.error).filter(|&e| e < f32::MAX).collect();
    let emax = errs.iter().copied().fold(0.0f32, f32::max).max(1e-9);
    let emin = errs.iter().copied().filter(|&e| e > 0.0).fold(f32::MAX, f32::min).min(emax);
    let mut cracks_total = 0;
    let mut cut_lines = Vec::new();
    for s in 0..=12 {
        let t = if s == 0 { 0.0 } else { emin * (emax / emin).powf((s - 1) as f32 / 11.0) * 1.0001 };
        let mut seen: HashMap<u32, [u32; 3]> = HashMap::new();
        let (mut sel_tris, mut cracks) = (0usize, 0usize);
        for c in &o.clusters {
            let own = o.groups[c.group as usize].simplified.error;
            let refined_ok = c.refined == -1 || o.groups[c.refined as usize].simplified.error <= t;
            if !(own > t && refined_ok) {
                continue;
            }
            sel_tris += c.ntris as usize;
            for k in c.first..c.first + c.nverts {
                let id = canon[o.verts[k as usize] as usize];
                let cp = o.copies[k as usize].map(f32::to_bits);
                match seen.get(&id) {
                    Some(prev) if *prev != cp => cracks += 1,
                    Some(_) => {}
                    None => {
                        seen.insert(id, cp);
                    }
                }
            }
        }
        cracks_total += cracks;
        cut_lines.push(format!("{t:.2e}:{sel_tris}t/{cracks}c"));
    }
    out.push(format!("  cuts (threshold:triangles/cracks): {}", cut_lines.join(" ")));
    out.push(format!("  crack total over 13 cuts: {cracks_total}"));
    out
}

pub fn summary(b: &Built) -> String {
    format!(
        "{} tris, {:.2} s, {} clusters, {} groups, {} levels {:?}, hierarchy {} nodes = {} B ({:.4} B/tri), roots {} tris {} B, DAG {} B ({:.2} B/tri)",
        b.tris,
        b.secs,
        b.clusters,
        b.groups,
        b.levels,
        b.tris_per_level,
        b.nodes,
        b.nodes * NODE_BYTES,
        (b.nodes * NODE_BYTES) as f64 / b.tris as f64,
        b.root_tris,
        b.root_bytes,
        b.dag_bytes,
        b.dag_bytes as f64 / b.tris as f64,
    )
}

fn sections(path: &str, node: &str) -> Vec<Prim> {
    let d = gl::open(Path::new(path));
    let mut out = Vec::new();
    for (n, mesh, m) in d.mesh_nodes() {
        let name = d.node_name(n).unwrap_or_default();
        if name == node {
            out.extend(d.prims(mesh, &m, Some(name)));
        }
    }
    out
}

fn dump(p: &Prim, path: &Path) {
    let mut b = Vec::new();
    for x in [p.pos.len() as u32, p.idx.len() as u32, u32::from(p.double_sided), p.nrm.len() as u32, p.uv.len() as u32] {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for v in p.pos.iter().flatten().chain(p.nrm.iter().flatten()).chain(p.uv.iter().flatten()) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for i in &p.idx {
        b.extend_from_slice(&i.to_le_bytes());
    }
    b.extend_from_slice(p.material.as_bytes());
    std::fs::write(path, b).unwrap();
}

fn undump(path: &Path) -> Prim {
    let b = std::fs::read(path).unwrap();
    let w = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    let f = |i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    let (nv, ni, ds, nn, nu) = (w(0) as usize, w(1) as usize, w(2) != 0, w(3) as usize, w(4) as usize);
    let mut o = 5;
    let pos = (0..nv).map(|i| [f(o + i * 3), f(o + i * 3 + 1), f(o + i * 3 + 2)]).collect();
    o += nv * 3;
    let nrm = (0..nn).map(|i| [f(o + i * 3), f(o + i * 3 + 1), f(o + i * 3 + 2)]).collect();
    o += nn * 3;
    let uv = (0..nu).map(|i| [f(o + i * 2), f(o + i * 2 + 1)]).collect();
    o += nu * 2;
    let idx = (0..ni).map(|i| w(o + i)).collect();
    o += ni;
    let material = String::from_utf8_lossy(&b[o * 4..]).to_string();
    Prim { pos, nrm, uv, idx, double_sided: ds, material, ..Default::default() }
}

/// `clod-one <dump>`: one section in its own process, loaded from a raw dump so the process
/// peak is the build's.
pub fn run_one(a: &[String]) {
    let p = undump(Path::new(&a[0]));
    let (ws0, _) = crate::peak_mib();
    let b = build(&p, config(p.double_sided), None, true);
    let (ws, pf) = crate::peak_mib();
    let mut canon = vec![0u32; p.pos.len()];
    // SAFETY: canon has one entry per vertex; positions are 3 floats per vertex.
    unsafe { ffi::meshopt_generatePositionRemap(canon.as_mut_ptr(), p.pos.as_ptr().cast(), p.pos.len(), 12) };
    let distinct = canon.iter().enumerate().filter(|&(i, &c)| c as usize == i).count();
    println!(
        "section '{}' double_sided {} verts {} (distinct positions {distinct}): {}",
        p.material,
        p.double_sided,
        p.pos.len(),
        summary(&b)
    );
    println!("  peak working set {ws:.0} MiB (input loaded {ws0:.0} MiB), peak private {pf:.0} MiB");
    for l in &b.lines {
        println!("{l}");
    }
}

/// `clod <glb> <node> <dump dir>`: every section of the node, each in a child process.
pub fn run(a: &[String]) {
    let secs = sections(&a[0], &a[1]);
    let dir = Path::new(&a[2]);
    std::fs::create_dir_all(dir).unwrap();
    let exe = std::env::current_exe().unwrap();
    for (i, p) in secs.iter().enumerate() {
        let path = dir.join(format!("section{i}.bin"));
        dump(p, &path);
        let t = Instant::now();
        let out = Command::new(&exe).arg("clod-one").arg(&path).output().unwrap();
        print!("{}", String::from_utf8_lossy(&out.stdout));
        println!("  child wall {:.2} s", crate::secs(t));
        if !out.status.success() {
            println!("section {i}: FAILED status {:?}
{}", out.status, String::from_utf8_lossy(&out.stderr));
        }
        std::fs::remove_file(&path).unwrap();
    }
}

/// `clod-zorah <gltf> <samples>`: clodBuild on primitives spread log-uniformly over Zorah's
/// size range; nodes per triangle fit, then hierarchy bytes per mesh over every Zorah mesh.
pub fn run_zorah(a: &[String]) {
    let d = gl::open(Path::new(&a[0]));
    let samples: usize = a[1].parse().unwrap();
    let max_tris: usize = a.get(2).map_or(40_000_000, |s| s.parse().unwrap());
    let acc = |i: u64| d.json["accessors"][i as usize]["count"].as_u64().unwrap() as usize;
    let mut prims = Vec::new(); // (tris, mesh, prim)
    for (mi, m) in d.json["meshes"].as_array().unwrap().iter().enumerate() {
        for (pi, p) in m["primitives"].as_array().unwrap().iter().enumerate() {
            prims.push((acc(p["indices"].as_u64().unwrap()) / 3, mi, pi));
        }
    }
    prims.sort();
    let lo = (prims[0].0.max(100)) as f64;
    let hi = prims.iter().map(|p| p.0).filter(|&t| t <= max_tris).max().unwrap() as f64;
    let mut picked = Vec::new();
    for s in 0..samples {
        let target = lo * (hi / lo).powf(s as f64 / (samples - 1).max(1) as f64);
        let i = prims.partition_point(|p| (p.0 as f64) < target).min(prims.len() - 1);
        if !picked.contains(&i) {
            picked.push(i);
        }
    }
    let ident = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut fit = Vec::new();
    for i in picked {
        let (tris, mesh, prim) = prims[i];
        let p = d.prims(mesh, &ident, None).swap_remove(prim);
        let b = build(&p, config(false), None, false);
        let (ws, _) = crate::peak_mib();
        println!("zorah mesh {mesh} prim {prim}: {} (peak ws so far {ws:.0} MiB)", summary(&b));
        fit.push((tris as f64, b.nodes as f64));
        drop(p);
        d.clear_cache();
    }
    // Nodes per section from log-log interpolation between the measured samples.
    fit.sort_by(|a, b| a.0.total_cmp(&b.0));
    let est = |t: f64| -> f64 {
        let (first, last) = (fit[0], fit[fit.len() - 1]);
        if t <= first.0 {
            return (first.1 * t / first.0).max(1.0);
        }
        if t >= last.0 {
            return last.1 * t / last.0;
        }
        let i = fit.partition_point(|f| f.0 < t);
        let (a, b) = (fit[i - 1], fit[i]);
        let f = (t.ln() - a.0.ln()) / (b.0.ln() - a.0.ln());
        (a.1.ln() + f * (b.1.ln() - a.1.ln())).exp()
    };
    let mut per_mesh = vec![0f64; d.json["meshes"].as_array().unwrap().len()];
    for &(t, m, _) in &prims {
        per_mesh[m] += est(t as f64).round().max(1.0) * NODE_BYTES as f64;
    }
    let total: f64 = per_mesh.iter().sum();
    let tris: usize = prims.iter().map(|p| p.0).sum();
    println!(
        "zorah: {} meshes, {} sections, {tris} tris, hierarchy all resident {:.1} MiB ({:.4} B/tri)",
        per_mesh.len(),
        prims.len(),
        total / 1048576.0,
        total / tris as f64
    );
    let mut sorted = per_mesh.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "per-mesh hierarchy: median {:.0} KiB, p90 {:.0} KiB, p99 {:.0} KiB, max {:.0} KiB",
        sorted[sorted.len() / 2] / 1024.0,
        sorted[sorted.len() * 9 / 10] / 1024.0,
        sorted[sorted.len() * 99 / 100] / 1024.0,
        sorted[sorted.len() - 1] / 1024.0
    );
    for log2 in 10..=27 {
        let thr = (1u64 << log2) as f64;
        let mut mount = 0.0;
        let mut paged = 0;
        for &h in &per_mesh {
            if h <= thr {
                mount += h;
            } else {
                mount += h.min(65536.0);
                paged += 1;
            }
        }
        println!(
            "threshold {:>8} KiB: mount {:>7.1} MiB, meshes paged {paged}",
            thr / 1024.0,
            mount / 1048576.0
        );
    }
}
