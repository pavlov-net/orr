//! P4 and the mesh half of P5. Every mesh node of every tree (all variants, all glTF LODs) is
//! cut into level-0 clusters (`buildMeshletsSpatial`, 128 triangles, 128 vertices or 256 on
//! double-sided sections, `optimizeMeshletLevel` 3), quantised to the mesh grid and packed
//! into resident pages of at most 128 KiB in the contracts.md 17 layout.
//!
//! Disk forms per page: (a) zstd of the resident bytes; (b) header and cluster records as is,
//! positions and attributes through the vertex codec (one stream each per page), triangles
//! through the meshlet codec per cluster, then zstd. Both decode back to the resident bytes
//! bit for bit; decode GB/s is resident output per second on one core.
//!
//! Chunking (P5): each mesh's pages as one blob, compressed whole, in 1 MiB and 256 KiB chunks,
//! and per page.

use crate::gl::{self, Prim};
use meshopt::ffi;
use std::path::Path;
use std::time::Instant;

const PAGE_MAX: usize = 128 * 1024;
const HEADER: usize = 128;
const CLUSTER: usize = 128;

pub struct Cluster {
    anchor: [i32; 3],
    local: Vec<[u16; 3]>,
    attrs: Vec<[u32; 3]>, // normal oct, tangent oct (+sign), uv
    tris: Vec<u8>,
    uv_off: [f32; 2],
    uv_scale: [f32; 2],
    section: u8,
}

pub struct Page {
    pub resident: Vec<u8>,
    cl_verts: Vec<u32>,
    cl_tris: Vec<u32>,
    pos_off: usize,
    attr_off: usize,
    tri_off: usize,
}

fn oct(n: [f32; 3]) -> [f32; 2] {
    let l = n[0].abs() + n[1].abs() + n[2].abs();
    let (mut x, mut y) = (n[0] / l, n[1] / l);
    if n[2] < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    [x * 0.5 + 0.5, y * 0.5 + 0.5]
}

fn unorm16(v: f32) -> u32 {
    (v.clamp(0.0, 1.0) * 65535.0).round() as u32
}

/// Mesh grid exponent: k = max(-11, ceil(log2(extent / 65535))).
pub fn grid_k(prims: &[Prim]) -> i32 {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in prims {
        for v in &p.pos {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    }
    let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max);
    (-11).max((extent as f64 / 65535.0).log2().ceil() as i32)
}

/// Level-0 clusters of one section on the mesh grid 2^k.
pub fn clusters(p: &Prim, k: i32, section: u8) -> Vec<Cluster> {
    let maxv = if p.double_sided { 256 } else { 128 };
    let maxt = 128;
    // SAFETY: buffers sized by meshopt's bound (spatial clustering bounds with min_triangles);
    // positions are 3 floats per vertex.
    let bound = unsafe { ffi::meshopt_buildMeshletsBound(p.idx.len(), maxv, maxt / 4) };
    let mut ml = vec![ffi::meshopt_Meshlet { vertex_offset: 0, triangle_offset: 0, vertex_count: 0, triangle_count: 0 }; bound];
    let mut mv = vec![0u32; bound * maxv];
    let mut mt = vec![0u8; bound * maxt * 3];
    // SAFETY: as above.
    let n = unsafe {
        ffi::meshopt_buildMeshletsSpatial(
            ml.as_mut_ptr(),
            mv.as_mut_ptr(),
            mt.as_mut_ptr(),
            p.idx.as_ptr(),
            p.idx.len(),
            p.pos.as_ptr().cast(),
            p.pos.len(),
            12,
            maxv,
            maxt / 4,
            maxt,
            0.5,
        )
    };
    let step = 2f64.powi(k);
    let mut out = Vec::with_capacity(n);
    for m in &ml[..n] {
        let (vo, to) = (m.vertex_offset as usize, m.triangle_offset as usize);
        let (vc, tc) = (m.vertex_count as usize, m.triangle_count as usize);
        // SAFETY: the meshlet's ranges lie inside mv and mt.
        unsafe { ffi::meshopt_optimizeMeshletLevel(mv.as_mut_ptr().add(vo), vc, mt.as_mut_ptr().add(to), tc, 3) };
        let verts = &mv[vo..vo + vc];
        let grid: Vec<[i32; 3]> = verts
            .iter()
            .map(|&v| p.pos[v as usize].map(|x| (x as f64 / step).round() as i32))
            .collect();
        let anchor = [0, 1, 2].map(|a| grid.iter().map(|g| g[a]).min().unwrap());
        let local = grid
            .iter()
            .map(|g| [0, 1, 2].map(|a| u16::try_from(g[a] - anchor[a]).expect("cluster over 16 bits")))
            .collect();
        let uvs: Vec<[f32; 2]> = verts.iter().map(|&v| p.uv.get(v as usize).copied().unwrap_or([0.0; 2])).collect();
        let uv_lo = [0, 1].map(|a| uvs.iter().map(|u| u[a]).fold(f32::MAX, f32::min));
        let uv_hi = [0, 1].map(|a| uvs.iter().map(|u| u[a]).fold(f32::MIN, f32::max));
        let uv_scale = [0, 1].map(|a| (uv_hi[a] - uv_lo[a]).max(1e-12));
        let attrs = verts
            .iter()
            .zip(&uvs)
            .map(|(&v, uv)| {
                let n = oct(p.nrm.get(v as usize).copied().unwrap_or([0.0, 0.0, 1.0]));
                let t4 = p.tan.get(v as usize).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]);
                let t = oct([t4[0], t4[1], t4[2]]);
                let ty = (unorm16(t[1]) & !1) | u32::from(t4[3] < 0.0);
                [
                    unorm16(n[0]) | unorm16(n[1]) << 16,
                    unorm16(t[0]) | ty << 16,
                    unorm16((uv[0] - uv_lo[0]) / uv_scale[0]) | unorm16((uv[1] - uv_lo[1]) / uv_scale[1]) << 16,
                ]
            })
            .collect();
        out.push(Cluster { anchor, local, attrs, tris: codec_order(&mt[to..to + tc * 3]), uv_off: uv_lo, uv_scale, section });
    }
    out
}

/// The meshlet codec rotates triangle corners; the cook fixes corner order by one round trip,
/// so the resident page holds what the codec decodes (OMM generation follows it).
fn codec_order(tris: &[u8]) -> Vec<u8> {
    let t = tris.len() / 3;
    // SAFETY: plain bound query, then a triangles-only encode and decode of t triangles.
    let mut buf = vec![0u8; unsafe { ffi::meshopt_encodeMeshletBound(256, 256) }];
    let len = unsafe { ffi::meshopt_encodeMeshlet(buf.as_mut_ptr(), buf.len(), std::ptr::null(), 0, tris.as_ptr(), t) };
    let mut out = vec![0u8; (t * 3).next_multiple_of(4)];
    let rc = unsafe { ffi::meshopt_decodeMeshlet(std::ptr::null_mut(), 0, 4, out.as_mut_ptr().cast(), t, 3, buf.as_ptr(), len) };
    assert_eq!(rc, 0);
    out.truncate(t * 3);
    let key = |s: &[u8]| {
        let mut v: Vec<[u8; 3]> = s.chunks(3).map(|c| {
            let m = (0..3).min_by_key(|&i| c[i]).unwrap();
            [c[m], c[(m + 1) % 3], c[(m + 2) % 3]]
        }).collect();
        v.sort();
        v
    };
    assert_eq!(key(tris), key(&out), "codec changed the triangle set or winding");
    out
}

fn cluster_bytes(c: &Cluster) -> usize {
    CLUSTER + (c.local.len() * 6).next_multiple_of(4) + c.attrs.len() * 12 + c.tris.len().next_multiple_of(4)
}

fn put(b: &mut [u8], o: usize, v: &[u8]) {
    b[o..o + v.len()].copy_from_slice(v);
}

/// Pages of whole clusters, at most 128 KiB each, in the contracts.md 17 layout.
pub fn pages(cl: &[Cluster], k: i32) -> Vec<Page> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < cl.len() {
        let mut size = HEADER;
        let mut j = i;
        while j < cl.len() && size + cluster_bytes(&cl[j]) <= PAGE_MAX {
            size += cluster_bytes(&cl[j]);
            j += 1;
        }
        let group = &cl[i..j];
        let n = group.len();
        let pos_bytes: usize = group.iter().map(|c| (c.local.len() * 6).next_multiple_of(4)).sum();
        let attr_bytes: usize = group.iter().map(|c| c.attrs.len() * 12).sum();
        let pos_off = HEADER + n * CLUSTER;
        let attr_off = pos_off + pos_bytes;
        let tri_off = attr_off + attr_bytes;
        let mut b = vec![0u8; size];
        for (o, v) in [n as u32, pos_off as u32, attr_off as u32, tri_off as u32, size as u32].iter().enumerate() {
            put(&mut b, o * 4, &v.to_le_bytes());
        }
        let (mut po, mut ao, mut to) = (pos_off, attr_off, tri_off);
        let step = 2f32.powi(k);
        let mut cl_verts = Vec::new();
        let mut cl_tris = Vec::new();
        for (ci, c) in group.iter().enumerate() {
            let r = HEADER + ci * CLUSTER;
            let mx = [0, 1, 2].map(|a| c.local.iter().map(|l| l[a]).max().unwrap());
            let centre = [0, 1, 2].map(|a| (c.anchor[a] as f32 + mx[a] as f32 * 0.5) * step);
            let radius = (mx.iter().map(|&m| (m as f32 * 0.5 * step).powi(2)).sum::<f32>()).sqrt();
            // Record: spheres and errors 0..56, cone 56, anchor 60, firsts 72, counts 80.
            let mut rec = Vec::with_capacity(CLUSTER);
            for s in 0..3 {
                for v in centre {
                    rec.extend_from_slice(&v.to_le_bytes());
                }
                rec.extend_from_slice(&radius.to_le_bytes());
                if s < 2 {
                    rec.extend_from_slice(&0f32.to_le_bytes());
                }
            }
            rec.extend_from_slice(&[0u8; 4]);
            for v in c.anchor {
                rec.extend_from_slice(&v.to_le_bytes());
            }
            rec.extend_from_slice(&((po - pos_off) as u32 / 2).to_le_bytes());
            rec.extend_from_slice(&((to - tri_off) as u32).to_le_bytes());
            rec.extend_from_slice(&(c.local.len() as u16).to_le_bytes()); // 80
            rec.push((c.tris.len() / 3) as u8); // 82
            rec.push((c.tris.len() / 3) as u8);
            rec.extend_from_slice(&[c.section, 0, 0, 0]);
            for v in mx {
                rec.extend_from_slice(&v.to_le_bytes());
            }
            rec.extend_from_slice(&[0u8; 6]);
            for v in c.uv_off.iter().chain(&c.uv_scale) {
                rec.extend_from_slice(&v.to_le_bytes());
            }
            rec.extend_from_slice(&u32::MAX.to_le_bytes());
            rec.extend_from_slice(&0f32.to_le_bytes());
            assert!(rec.len() <= CLUSTER, "record {}", rec.len());
            put(&mut b, r, &rec);
            for l in &c.local {
                for v in l {
                    put(&mut b, po, &v.to_le_bytes());
                    po += 2;
                }
            }
            po = po.next_multiple_of(4);
            for a in &c.attrs {
                for v in a {
                    put(&mut b, ao, &v.to_le_bytes());
                    ao += 4;
                }
            }
            put(&mut b, to, &c.tris);
            to += c.tris.len().next_multiple_of(4);
            cl_verts.push(c.local.len() as u32);
            cl_tris.push((c.tris.len() / 3) as u32);
        }
        out.push(Page { resident: b, cl_verts, cl_tris, pos_off, attr_off, tri_off });
        i = j;
    }
    out
}

/// Codec form of a page (before zstd): header + records, then u32 sizes of the position stream,
/// the attribute stream and each cluster's meshlet stream, then the streams.
fn encode(p: &Page) -> Vec<u8> {
    let r = &p.resident;
    let nv: usize = p.cl_verts.iter().map(|&v| v as usize).sum();
    // Positions widened to 8 bytes (the vertex codec needs a multiple of 4).
    let mut pos8 = Vec::with_capacity(nv * 8);
    let mut po = p.pos_off;
    for &v in &p.cl_verts {
        for _ in 0..v {
            pos8.extend_from_slice(&r[po..po + 6]);
            pos8.extend_from_slice(&[0, 0]);
            po += 6;
        }
        po = po.next_multiple_of(4);
    }
    let attrs = &r[p.attr_off..p.attr_off + nv * 12];
    let venc = |data: &[u8], stride: usize| -> Vec<u8> {
        let n = data.len() / stride;
        // SAFETY: buffer sized by the bound; data holds n vertices of stride bytes.
        let mut out = vec![0u8; unsafe { ffi::meshopt_encodeVertexBufferBound(n, stride) }];
        // SAFETY: as above.
        let len = unsafe { ffi::meshopt_encodeVertexBufferLevel(out.as_mut_ptr(), out.len(), data.as_ptr().cast(), n, stride, 3, 1) };
        out.truncate(len);
        out
    };
    let ep = venc(&pos8, 8);
    let ea = venc(attrs, 12);
    let mut em = Vec::new();
    let mut sizes = Vec::new();
    let mut to = p.tri_off;
    // SAFETY: plain bound query.
    let mut buf = vec![0u8; unsafe { ffi::meshopt_encodeMeshletBound(256, 256) }];
    for &t in &p.cl_tris {
        let t = t as usize;
        // SAFETY: triangles only (no vertex references); buf sized by the bound.
        let len = unsafe { ffi::meshopt_encodeMeshlet(buf.as_mut_ptr(), buf.len(), std::ptr::null(), 0, r.as_ptr().add(to), t) };
        assert!(len > 0);
        sizes.push(len as u32);
        em.extend_from_slice(&buf[..len]);
        to += (t * 3).next_multiple_of(4);
    }
    let mut out = r[..p.pos_off].to_vec();
    out.extend_from_slice(&(ep.len() as u32).to_le_bytes());
    out.extend_from_slice(&(ea.len() as u32).to_le_bytes());
    for s in &sizes {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out.extend_from_slice(&ep);
    out.extend_from_slice(&ea);
    out.extend_from_slice(&em);
    out
}

struct Scratch {
    pos8: Vec<u8>,
    tri: Vec<u8>,
}

/// Decodes the codec form into `dst` (resident layout). Cluster counts come from the records.
fn decode(enc: &[u8], dst: &mut [u8], s: &mut Scratch) {
    let w = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
    let n = w(enc, 0);
    let (pos_off, attr_off, tri_off, size) = (w(enc, 4), w(enc, 8), w(enc, 12), w(enc, 16));
    assert_eq!(size, dst.len());
    dst[..pos_off].copy_from_slice(&enc[..pos_off]);
    let mut verts = Vec::with_capacity(n);
    let mut tris = Vec::with_capacity(n);
    for ci in 0..n {
        let r = HEADER + ci * CLUSTER;
        verts.push(u16::from_le_bytes([enc[r + 80], enc[r + 81]]) as usize);
        tris.push(enc[r + 82] as usize);
    }
    let nv: usize = verts.iter().sum();
    let ep_len = w(enc, pos_off);
    let ea_len = w(enc, pos_off + 4);
    let mut o = pos_off + 8 + n * 4;
    s.pos8.resize(nv * 8, 0);
    // SAFETY: destinations hold nv vertices; sources are the encoded streams.
    unsafe {
        assert_eq!(ffi::meshopt_decodeVertexBuffer(s.pos8.as_mut_ptr().cast(), nv, 8, enc.as_ptr().add(o), ep_len), 0);
        o += ep_len;
        assert_eq!(
            ffi::meshopt_decodeVertexBuffer(dst.as_mut_ptr().add(attr_off).cast(), nv, 12, enc.as_ptr().add(o), ea_len),
            0
        );
        o += ea_len;
    }
    let mut po = pos_off;
    let mut src = 0;
    for &v in &verts {
        for _ in 0..v {
            dst[po..po + 6].copy_from_slice(&s.pos8[src..src + 6]);
            po += 6;
            src += 8;
        }
        let pad = po.next_multiple_of(4);
        dst[po..pad].fill(0);
        po = pad;
    }
    let mut to = tri_off;
    for (ci, &t) in tris.iter().enumerate() {
        let len = w(enc, pos_off + 8 + ci * 4);
        let tb = (t * 3).next_multiple_of(4);
        s.tri.resize(tb, 0);
        // SAFETY: triangles-only decode of len bytes into a 4-aligned buffer of t triangles.
        let rc = unsafe {
            ffi::meshopt_decodeMeshlet(std::ptr::null_mut(), 0, 4, s.tri.as_mut_ptr().cast(), t, 3, enc.as_ptr().add(o), len)
        };
        assert_eq!(rc, 0);
        dst[to..to + t * 3].copy_from_slice(&s.tri[..t * 3]);
        dst[to + t * 3..to + tb].fill(0);
        to += tb;
        o += len;
    }
}

fn zstd(b: &[u8], level: i32) -> Vec<u8> {
    zstd::bulk::compress(b, level).unwrap()
}

fn chunked(b: &[u8], chunk: usize, level: i32) -> usize {
    b.chunks(chunk).map(|c| zstd(c, level).len()).sum()
}

pub fn tree_glbs(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut g: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir())
        .map(|p| p.join(format!("{}.glb", p.file_name().unwrap().to_string_lossy())))
        .filter(|p| p.exists())
        .collect();
    g.sort();
    g
}

struct MeshOut {
    tris: usize,
    pages: Vec<(Vec<u8>, Vec<u8>, usize, Vec<u8>)>, // resident, zstd(resident), codec len, zstd(codec)
    chunk: [usize; 6],                              // whole/1M/256K raw, whole/1M/256K codec
}

fn mesh(prims: &[Prim], level: i32) -> MeshOut {
    let k = grid_k(prims);
    let mut cl = Vec::new();
    let mut tris = 0;
    for (si, p) in prims.iter().enumerate() {
        tris += p.tris();
        cl.extend(clusters(p, k, si as u8));
    }
    let pg = pages(&cl, k);
    let mut blob_raw = Vec::new();
    let mut blob_cod = Vec::new();
    let mut pages_out = Vec::new();
    for p in &pg {
        let enc = encode(p);
        blob_raw.extend_from_slice(&p.resident);
        blob_cod.extend_from_slice(&enc);
        pages_out.push((p.resident.clone(), zstd(&p.resident, level), enc.len(), zstd(&enc, level)));
    }
    let chunk = [
        zstd(&blob_raw, level).len(),
        chunked(&blob_raw, 1 << 20, level),
        chunked(&blob_raw, 256 << 10, level),
        zstd(&blob_cod, level).len(),
        chunked(&blob_cod, 1 << 20, level),
        chunked(&blob_cod, 256 << 10, level),
    ];
    MeshOut { tris, pages: pages_out, chunk }
}

pub fn run(a: &[String]) {
    use rayon::prelude::*;
    let level: i32 = a.get(1).map_or(19, |s| s.parse().unwrap());
    let mut all: Vec<MeshOut> = Vec::new();
    for g in tree_glbs(Path::new(&a[0])) {
        let d = gl::open(&g);
        let nodes: Vec<Vec<Prim>> =
            d.mesh_nodes().into_iter().map(|(node, mesh, m)| d.prims(mesh, &m, d.node_name(node))).collect();
        drop(d);
        let outs: Vec<MeshOut> = nodes.par_iter().map(|p| mesh(p, level)).collect();
        let tris: usize = outs.iter().map(|o| o.tris).sum();
        let sum = |f: &dyn Fn(&(Vec<u8>, Vec<u8>, usize, Vec<u8>)) -> usize| -> usize {
            outs.iter().flat_map(|o| &o.pages).map(f).sum()
        };
        let (res, zr, ze) = (sum(&|p| p.0.len()), sum(&|p| p.1.len()), sum(&|p| p.3.len()));
        println!(
            "{}: {tris} tris, resident {:.1} MB ({:.1} B/tri), zstd {:.2} B/tri, codec+zstd {:.2} B/tri ({:+.1}%)",
            g.file_stem().unwrap().to_string_lossy(),
            res as f64 / 1e6,
            res as f64 / tris as f64,
            zr as f64 / tris as f64,
            ze as f64 / tris as f64,
            (ze as f64 / zr as f64 - 1.0) * 100.0
        );
        all.extend(outs);
    }
    // Decode timing, one thread, nothing else running: best of 5 per page, output checked.
    let mut dctx = zstd::bulk::Decompressor::new().unwrap();
    let mut scratch = Scratch { pos8: Vec::new(), tri: Vec::new() };
    let mut zbuf = vec![0u8; PAGE_MAX * 2];
    let mut dst = vec![0u8; PAGE_MAX];
    let (mut t_zstd, mut t_codec) = (0f64, 0f64);
    for (res, zr, _, ze) in all.iter().flat_map(|o| &o.pages) {
        let out = &mut dst[..res.len()];
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            let n = dctx.decompress_to_buffer(zr, &mut zbuf).unwrap();
            out.copy_from_slice(&zbuf[..n]);
            best = best.min(crate::secs(t0));
        }
        assert_eq!(out, &res[..]);
        t_zstd += best;
        let mut best = f64::MAX;
        for _ in 0..5 {
            out.fill(0xCD);
            let t0 = Instant::now();
            let n = dctx.decompress_to_buffer(ze, &mut zbuf).unwrap();
            decode(&zbuf[..n], out, &mut scratch);
            best = best.min(crate::secs(t0));
        }
        if out != &res[..] {
            let i = out.iter().zip(res.iter()).position(|(a, b)| a != b).unwrap();
            let w = |o: usize| u32::from_le_bytes(res[o..o + 4].try_into().unwrap());
            panic!("codec round trip differs at byte {i} of {}; pos {} attr {} tri {}", res.len(), w(4), w(8), w(12));
        }
        t_codec += best;
    }
    let pages = || all.iter().flat_map(|o| &o.pages);
    let tris: usize = all.iter().map(|o| o.tris).sum();
    let resident: usize = pages().map(|p| p.0.len()).sum();
    let raw_z: usize = pages().map(|p| p.1.len()).sum();
    let codec_pre: usize = pages().map(|p| p.2).sum();
    let codec_z: usize = pages().map(|p| p.3.len()).sum();
    let ch = |i: usize| -> usize { all.iter().map(|o| o.chunk[i]).sum() };
    let gb = |s: f64| resident as f64 / s / 1e9;
    println!("zstd level {level}: {} meshes, {} pages, {tris} source triangles", all.len(), pages().count());
    println!("resident {resident} B ({:.2} B/tri)", resident as f64 / tris as f64);
    println!(
        "zstd alone {raw_z} B ({:.2} B/tri), decode {:.2} GB/s resident output per core",
        raw_z as f64 / tris as f64,
        gb(t_zstd)
    );
    println!(
        "codec {codec_pre} B before zstd, codec+zstd {codec_z} B ({:.2} B/tri, {:.1}% smaller than zstd alone), decode {:.2} GB/s",
        codec_z as f64 / tris as f64,
        (1.0 - codec_z as f64 / raw_z as f64) * 100.0,
        gb(t_codec)
    );
    let loss = |c: usize, w: usize| (c as f64 / w as f64 - 1.0) * 100.0;
    println!(
        "chunking, resident pages: whole {}, 1 MiB {} ({:+.2}%), 256 KiB {} ({:+.2}%), per page {raw_z} ({:+.2}%)",
        ch(0),
        ch(1),
        loss(ch(1), ch(0)),
        ch(2),
        loss(ch(2), ch(0)),
        loss(raw_z, ch(0))
    );
    println!(
        "chunking, codec pages: whole {}, 1 MiB {} ({:+.2}%), 256 KiB {} ({:+.2}%), per page {codec_z} ({:+.2}%)",
        ch(3),
        ch(4),
        loss(ch(4), ch(3)),
        ch(5),
        loss(ch(5), ch(3)),
        loss(codec_z, ch(3))
    );
}

/// `dict <dir> <level> <dict KiB>`: per-page zstd with one dictionary trained on every 4th
/// page of the set, against per-page zstd without one; decode GB/s with the dictionary.
pub fn run_dict(a: &[String]) {
    use rayon::prelude::*;
    let level: i32 = a[1].parse().unwrap();
    let dict_size: usize = a[2].parse::<usize>().unwrap() * 1024;
    let mut pages_all: Vec<Vec<u8>> = Vec::new();
    for g in tree_glbs(Path::new(&a[0])) {
        let d = gl::open(&g);
        let nodes: Vec<Vec<Prim>> =
            d.mesh_nodes().into_iter().map(|(node, mesh, m)| d.prims(mesh, &m, d.node_name(node))).collect();
        drop(d);
        let pg: Vec<Vec<Vec<u8>>> = nodes
            .par_iter()
            .map(|prims| {
                let k = grid_k(prims);
                let mut cl = Vec::new();
                for (si, p) in prims.iter().enumerate() {
                    cl.extend(clusters(p, k, si as u8));
                }
                pages(&cl, k).into_iter().map(|p| p.resident).collect()
            })
            .collect();
        pages_all.extend(pg.into_iter().flatten());
    }
    let samples: Vec<&Vec<u8>> = pages_all.iter().step_by(4).collect();
    let t = Instant::now();
    let dict = zstd::dict::from_samples(&samples, dict_size).unwrap();
    println!("{} pages; dictionary {} B trained on {} pages in {:.1} s", pages_all.len(), dict.len(), samples.len(), crate::secs(t));
    let plain: usize = pages_all.par_iter().map(|p| zstd(p, level).len()).sum();
    let with: Vec<Vec<u8>> = pages_all
        .par_iter()
        .map(|p| zstd::bulk::Compressor::with_dictionary(level, &dict).unwrap().compress(p).unwrap())
        .collect();
    let with_n: usize = with.iter().map(Vec::len).sum();
    let mut dctx = zstd::bulk::Decompressor::with_dictionary(&dict).unwrap();
    let mut buf = vec![0u8; PAGE_MAX];
    let mut tsum = 0f64;
    let mut out = 0usize;
    for (p, z) in pages_all.iter().zip(&with) {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            let n = dctx.decompress_to_buffer(z, &mut buf).unwrap();
            best = best.min(crate::secs(t0));
            assert_eq!(&buf[..n], &p[..]);
        }
        tsum += best;
        out += p.len();
    }
    println!(
        "level {level}: per page {plain} B, per page with dictionary {with_n} B ({:+.2}%, dictionary included once {:+.2}%); decode with dictionary {:.2} GB/s",
        (with_n as f64 / plain as f64 - 1.0) * 100.0,
        ((with_n + dict.len()) as f64 / plain as f64 - 1.0) * 100.0,
        out as f64 / tsum / 1e9
    );
}
