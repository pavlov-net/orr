//! P6: part-locked simplification on a rigid-part tree. Parts come from the per-vertex `_ID`
//! joint attribute (each triangle goes to the joint most of its corners carry), checked against
//! the per-joint triangle counts in the build's wind JSON.
//!
//! Per section, the coarsest result of:
//! - discrete, M2b form: `simplify` with LockBorder + ErrorAbsolute, no target, on the whole
//!   section vs on each part alone (part boundaries become locked borders);
//! - discrete, locks only: `simplifyWithAttributes` without LockBorder, with and without
//!   `SimplifyVertex::Lock` on positions shared by two parts;
//! - DAG: clodBuild on the whole section vs per part, terminal-group triangles.

use crate::clod;
use crate::gl::{self, Prim};
use meshopt::ffi;
use std::collections::{HashMap, HashSet};
use std::path::Path;

const LOCK_BORDER: u32 = 1;
const SPARSE: u32 = 2;
const ERROR_ABSOLUTE: u32 = 4;

fn simplify(p: &Prim, idx: &[u32], options: u32, lock: Option<&[u8]>) -> usize {
    let mut dst = vec![0u32; idx.len()];
    let mut err = 0f32;
    // SAFETY: dst holds idx.len() entries; positions are 3 floats per vertex; lock has one
    // entry per vertex when present.
    let n = unsafe {
        match lock {
            None => ffi::meshopt_simplify(
                dst.as_mut_ptr(),
                idx.as_ptr(),
                idx.len(),
                p.pos.as_ptr().cast(),
                p.pos.len(),
                12,
                0,
                f32::MAX,
                options,
                &mut err,
            ),
            Some(l) => ffi::meshopt_simplifyWithAttributes(
                dst.as_mut_ptr(),
                idx.as_ptr(),
                idx.len(),
                p.pos.as_ptr().cast(),
                p.pos.len(),
                12,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                l.as_ptr(),
                0,
                f32::MAX,
                options,
                &mut err,
            ),
        }
    };
    n / 3
}

/// A compact sub-mesh holding only the triangles in `idx`.
fn sub(p: &Prim, idx: &[u32]) -> Prim {
    let mut map = HashMap::new();
    let mut o = Prim { double_sided: p.double_sided, material: p.material.clone(), ..Default::default() };
    for &v in idx {
        let n = *map.entry(v).or_insert_with(|| {
            o.pos.push(p.pos[v as usize]);
            if !p.nrm.is_empty() {
                o.nrm.push(p.nrm[v as usize]);
            }
            if !p.uv.is_empty() {
                o.uv.push(p.uv[v as usize]);
            }
            (o.pos.len() - 1) as u32
        });
        o.idx.push(n);
    }
    o
}

pub fn run(a: &[String]) {
    let path = Path::new(&a[0]);
    let variant = &a[1];
    let d = gl::open(path);
    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
    let wind: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.with_file_name(format!("{stem}_wind.json"))).unwrap()).unwrap();
    let wv = wind["variants"].as_array().unwrap().iter().find(|v| v["name"] == variant.as_str()).unwrap();
    let joints = wv["joints"].as_array().unwrap().len();
    let node_name = wv["node"].as_str().unwrap();
    let (node, mesh, m) = d.mesh_nodes().into_iter().find(|n| d.node_name(n.0).as_deref() == Some(node_name)).unwrap();
    let prims = d.prims(mesh, &m, d.node_name(node));
    println!("{stem} {variant}: {joints} joints, node {node_name}, {} sections", prims.len());
    let (mut tot_whole, mut tot_parts) = ([0usize; 3], [0usize; 3]);
    let mut tot_tris = 0;
    for (si, p) in prims.iter().enumerate() {
        let tri_part: Vec<u32> = p
            .idx
            .chunks(3)
            .map(|t| {
                let ids: Vec<u32> = t.iter().map(|&v| p.id[v as usize]).collect();
                if ids[1] == ids[2] { ids[1] } else { ids[0] }
            })
            .collect();
        let mut per = vec![0usize; joints];
        for &j in &tri_part {
            per[j as usize] += 1;
        }
        let json_per: Vec<usize> =
            wv["primitives"][si]["joint_triangles"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let used = per.iter().filter(|&&c| c > 0).count();
        let small = per.iter().filter(|&&c| c > 0 && c < 64).count();
        println!(
            "section {si} '{}' double_sided {}: {} tris, {} verts, parts used {used}, under 64 tris {small}, min {}, max {}; per-joint counts equal wind JSON: {}",
            p.material,
            p.double_sided,
            p.tris(),
            p.pos.len(),
            per.iter().filter(|&&c| c > 0).min().unwrap(),
            per.iter().max().unwrap(),
            json_per == per
        );
        let mut part_idx: Vec<Vec<u32>> = vec![Vec::new(); joints];
        for (t, &j) in tri_part.iter().enumerate() {
            part_idx[j as usize].extend_from_slice(&p.idx[t * 3..t * 3 + 3]);
        }
        // Positions used by triangles of two or more parts.
        let mut canon = vec![0u32; p.pos.len()];
        // SAFETY: one entry per vertex; 3 floats per position.
        unsafe { ffi::meshopt_generatePositionRemap(canon.as_mut_ptr(), p.pos.as_ptr().cast(), p.pos.len(), 12) };
        let mut owner: HashMap<u32, u32> = HashMap::new();
        let mut shared = HashSet::new();
        for (t, &j) in tri_part.iter().enumerate() {
            for &v in &p.idx[t * 3..t * 3 + 3] {
                let c = canon[v as usize];
                match owner.get(&c) {
                    Some(&o) if o != j => {
                        shared.insert(c);
                    }
                    Some(_) => {}
                    None => {
                        owner.insert(c, j);
                    }
                }
            }
        }
        let lock: Vec<u8> = (0..p.pos.len()).map(|v| u8::from(shared.contains(&canon[v]))).collect();
        let none = vec![0u8; p.pos.len()];
        let opt = LOCK_BORDER | ERROR_ABSOLUTE;
        let a_whole = simplify(p, &p.idx, opt, None);
        let a_parts: usize = part_idx.iter().filter(|i| !i.is_empty()).map(|i| simplify(p, i, opt | SPARSE, None)).sum();
        let b_free = simplify(p, &p.idx, ERROR_ABSOLUTE, Some(&none));
        let b_lock = simplify(p, &p.idx, ERROR_ABSOLUTE, Some(&lock));
        let dag_whole = clod::build(p, clod::config(p.double_sided), None, false).root_tris;
        let dag_parts: usize = part_idx
            .iter()
            .filter(|i| !i.is_empty())
            .map(|i| clod::build(&sub(p, i), clod::config(p.double_sided), None, false).root_tris)
            .sum();
        let r = |x: usize, y: usize| x as f64 / y.max(1) as f64;
        println!(
            "  shared positions {}; coarsest tris: M2b whole {a_whole} vs per part {a_parts} ({:.2}x); locks only: free {b_free} vs part locks {b_lock} ({:.2}x); DAG roots whole {dag_whole} vs per part {dag_parts} ({:.2}x)",
            shared.len(),
            r(a_parts, a_whole),
            r(b_lock, b_free),
            r(dag_parts, dag_whole)
        );
        tot_tris += p.tris();
        for (i, (w, q)) in [(a_whole, a_parts), (b_free, b_lock), (dag_whole, dag_parts)].into_iter().enumerate() {
            tot_whole[i] += w;
            tot_parts[i] += q;
        }
    }
    let r = |i: usize| tot_parts[i] as f64 / tot_whole[i].max(1) as f64;
    println!(
        "mesh {tot_tris} tris; coarsest whole vs parts: M2b {} vs {} ({:.2}x); locks only {} vs {} ({:.2}x); DAG {} vs {} ({:.2}x)",
        tot_whole[0],
        tot_parts[0],
        r(0),
        tot_whole[1],
        tot_parts[1],
        r(1),
        tot_whole[2],
        tot_parts[2],
        r(2)
    );
}
