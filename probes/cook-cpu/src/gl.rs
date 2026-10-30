//! Minimal glTF reader for the probes: JSON through serde_json, buffers from the GLB chunk or
//! from files by positioned reads, meshopt-compressed buffer views (EXT/KHR) decoded through
//! meshopt's FFI, accessors dequantised to f32. Node transforms are applied to positions,
//! normals and tangents.

use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::rc::Rc;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub struct Doc {
    pub json: Value,
    pub raw_json: Vec<u8>,
    bin: Vec<u8>,
    dir: PathBuf,
    cache: RefCell<HashMap<usize, Rc<(Vec<u8>, usize)>>>,
}

#[derive(Default, Clone)]
pub struct Prim {
    pub mesh: usize,
    pub prim: usize,
    pub node: Option<String>,
    pub material: String,
    pub double_sided: bool,
    pub alpha_mask: bool,
    pub pos: Vec<[f32; 3]>,
    pub nrm: Vec<[f32; 3]>,
    pub tan: Vec<[f32; 4]>,
    pub uv: Vec<[f32; 2]>,
    pub id: Vec<u32>,
    pub idx: Vec<u32>,
}

impl Prim {
    pub fn tris(&self) -> usize {
        self.idx.len() / 3
    }
}

pub fn open(path: &Path) -> Doc {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut head = [0u8; 4];
    f.read_exact(&mut head).unwrap();
    f.seek(SeekFrom::Start(0)).unwrap();
    let dir = path.parent().unwrap().to_path_buf();
    if &head == b"glTF" {
        let mut d = Vec::new();
        f.read_to_end(&mut d).unwrap();
        let jl = u32::from_le_bytes(d[12..16].try_into().unwrap()) as usize;
        let raw_json = d[20..20 + jl].to_vec();
        let mut bin = Vec::new();
        let o = 20 + jl;
        if o + 8 <= d.len() {
            let bl = u32::from_le_bytes(d[o..o + 4].try_into().unwrap()) as usize;
            bin = d[o + 8..o + 8 + bl].to_vec();
        }
        let json = serde_json::from_slice(&raw_json).unwrap();
        Doc { json, raw_json, bin, dir, cache: Default::default() }
    } else {
        let mut raw_json = Vec::new();
        f.read_to_end(&mut raw_json).unwrap();
        let json = serde_json::from_slice(&raw_json).unwrap();
        Doc { json, raw_json, bin: Vec::new(), dir, cache: Default::default() }
    }
}

fn u(v: &Value, k: &str) -> usize {
    v[k].as_u64().unwrap_or(0) as usize
}

impl Doc {
    fn buffer_range(&self, buffer: usize, off: usize, len: usize) -> Vec<u8> {
        let b = &self.json["buffers"][buffer];
        match b["uri"].as_str() {
            None => self.bin[off..off + len].to_vec(),
            Some(uri) => {
                let mut f = File::open(self.dir.join(uri)).unwrap();
                f.seek(SeekFrom::Start(off as u64)).unwrap();
                let mut v = vec![0u8; len];
                f.read_exact(&mut v).unwrap();
                v
            }
        }
    }

    /// Bytes of a buffer view and its stride (0 = tightly packed), decoded once per document.
    pub fn view(&self, i: usize) -> Rc<(Vec<u8>, usize)> {
        if let Some(v) = self.cache.borrow().get(&i) {
            return v.clone();
        }
        let v = Rc::new(self.view_uncached(i));
        self.cache.borrow_mut().insert(i, v.clone());
        v
    }

    pub fn clear_cache(&self) {
        self.cache.borrow_mut().clear();
    }

    fn view_uncached(&self, i: usize) -> (Vec<u8>, usize) {
        let bv = &self.json["bufferViews"][i];
        let ext = bv["extensions"]
            .get("EXT_meshopt_compression")
            .or_else(|| bv["extensions"].get("KHR_meshopt_compression"));
        if let Some(e) = ext {
            let src = self.buffer_range(u(e, "buffer"), u(e, "byteOffset"), u(e, "byteLength"));
            let stride = u(e, "byteStride");
            let count = u(e, "count");
            let mut dst = vec![0u8; stride * count];
            let mode = e["mode"].as_str().unwrap();
            // SAFETY: dst holds count * stride bytes; src is the encoded stream.
            let rc = unsafe {
                match mode {
                    "ATTRIBUTES" => meshopt::ffi::meshopt_decodeVertexBuffer(
                        dst.as_mut_ptr().cast(),
                        count,
                        stride,
                        src.as_ptr(),
                        src.len(),
                    ),
                    "TRIANGLES" => meshopt::ffi::meshopt_decodeIndexBuffer(
                        dst.as_mut_ptr().cast(),
                        count,
                        stride,
                        src.as_ptr(),
                        src.len(),
                    ),
                    "INDICES" => meshopt::ffi::meshopt_decodeIndexSequence(
                        dst.as_mut_ptr().cast(),
                        count,
                        stride,
                        src.as_ptr(),
                        src.len(),
                    ),
                    m => panic!("meshopt mode {m}"),
                }
            };
            assert_eq!(rc, 0, "meshopt decode failed on view {i}");
            let filter = e["filter"].as_str().unwrap_or("NONE");
            // SAFETY: filters run in place over count elements of stride bytes.
            unsafe {
                let p = dst.as_mut_ptr().cast();
                match filter {
                    "NONE" => {}
                    "OCTAHEDRAL" => meshopt::ffi::meshopt_decodeFilterOct(p, count, stride),
                    "QUATERNION" => meshopt::ffi::meshopt_decodeFilterQuat(p, count, stride),
                    "EXPONENTIAL" => meshopt::ffi::meshopt_decodeFilterExp(p, count, stride),
                    "COLOR" => meshopt::ffi::meshopt_decodeFilterColor(p, count, stride),
                    f => panic!("meshopt filter {f}"),
                }
            }
            return (dst, stride);
        }
        let d = self.buffer_range(u(bv, "buffer"), u(bv, "byteOffset"), u(bv, "byteLength"));
        (d, u(bv, "byteStride"))
    }

    /// Accessor as f32 rows of `n` components (normalised integers mapped to [0,1] / [-1,1]).
    pub fn floats(&self, a: usize, n: usize) -> Vec<f32> {
        let acc = &self.json["accessors"][a];
        assert!(acc.get("sparse").is_none(), "sparse accessor");
        let count = u(acc, "count");
        let ct = u(acc, "componentType");
        let norm = acc["normalized"].as_bool().unwrap_or(false);
        let comps = match acc["type"].as_str().unwrap() {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            t => panic!("type {t}"),
        };
        let cs = match ct {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            _ => 4,
        };
        let v = self.view(u(acc, "bufferView"));
        let (d, stride) = (&v.0, v.1);
        let stride = if stride == 0 { comps * cs } else { stride };
        let off = u(acc, "byteOffset");
        let mut out = Vec::with_capacity(count * n);
        for i in 0..count {
            let base = off + i * stride;
            for c in 0..n {
                let p = base + c * cs;
                let v = if c >= comps {
                    0.0
                } else {
                    match ct {
                        5126 => f32::from_le_bytes(d[p..p + 4].try_into().unwrap()),
                        5121 => {
                            let x = d[p] as f32;
                            if norm { x / 255.0 } else { x }
                        }
                        5120 => {
                            let x = d[p] as i8 as f32;
                            if norm { (x / 127.0).max(-1.0) } else { x }
                        }
                        5123 => {
                            let x = u16::from_le_bytes([d[p], d[p + 1]]) as f32;
                            if norm { x / 65535.0 } else { x }
                        }
                        5122 => {
                            let x = i16::from_le_bytes([d[p], d[p + 1]]) as f32;
                            if norm { (x / 32767.0).max(-1.0) } else { x }
                        }
                        5125 => u32::from_le_bytes(d[p..p + 4].try_into().unwrap()) as f32,
                        _ => panic!("componentType {ct}"),
                    }
                };
                out.push(v);
            }
        }
        out
    }

    pub fn indices(&self, a: usize) -> Vec<u32> {
        let acc = &self.json["accessors"][a];
        let count = u(acc, "count");
        let ct = u(acc, "componentType");
        let v = self.view(u(acc, "bufferView"));
        let d = &v.0;
        let off = u(acc, "byteOffset");
        (0..count)
            .map(|i| match ct {
                5125 => u32::from_le_bytes(d[off + i * 4..off + i * 4 + 4].try_into().unwrap()),
                5123 => u16::from_le_bytes([d[off + i * 2], d[off + i * 2 + 1]]) as u32,
                5121 => d[off + i] as u32,
                _ => panic!("index type {ct}"),
            })
            .collect()
    }

    pub fn index_count(&self, mesh: usize) -> usize {
        self.json["meshes"][mesh]["primitives"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| u(&self.json["accessors"][u(p, "indices")], "count"))
            .sum()
    }

    /// Node world matrices (column-major 4x4) for every node that holds a mesh, as (node, mesh).
    pub fn mesh_nodes(&self) -> Vec<(usize, usize, [f32; 16])> {
        let nodes = self.json["nodes"].as_array().cloned().unwrap_or_default();
        let mut parent = vec![usize::MAX; nodes.len()];
        for (i, n) in nodes.iter().enumerate() {
            for c in n["children"].as_array().into_iter().flatten() {
                parent[c.as_u64().unwrap() as usize] = i;
            }
        }
        let local = |n: &Value| -> [f32; 16] {
            if let Some(m) = n["matrix"].as_array() {
                let mut o = [0f32; 16];
                for (k, v) in m.iter().enumerate() {
                    o[k] = v.as_f64().unwrap() as f32;
                }
                return o;
            }
            let g = |k: &str, d: &[f32]| -> Vec<f32> {
                n[k].as_array()
                    .map(|a| a.iter().map(|v| v.as_f64().unwrap() as f32).collect())
                    .unwrap_or_else(|| d.to_vec())
            };
            let t = g("translation", &[0.0, 0.0, 0.0]);
            let r = g("rotation", &[0.0, 0.0, 0.0, 1.0]);
            let s = g("scale", &[1.0, 1.0, 1.0]);
            let (x, y, z, w) = (r[0], r[1], r[2], r[3]);
            let m3 = [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + z * w),
                2.0 * (x * z - y * w),
                2.0 * (x * y - z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + x * w),
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ];
            [
                m3[0] * s[0], m3[1] * s[0], m3[2] * s[0], 0.0,
                m3[3] * s[1], m3[4] * s[1], m3[5] * s[1], 0.0,
                m3[6] * s[2], m3[7] * s[2], m3[8] * s[2], 0.0,
                t[0], t[1], t[2], 1.0,
            ]
        };
        let mul = |a: &[f32; 16], b: &[f32; 16]| -> [f32; 16] {
            let mut o = [0f32; 16];
            for c in 0..4 {
                for r in 0..4 {
                    o[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
                }
            }
            o
        };
        let mut out = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            let Some(mesh) = n["mesh"].as_u64() else { continue };
            let mut m = local(n);
            let mut p = parent[i];
            while p != usize::MAX {
                m = mul(&local(&nodes[p]), &m);
                p = parent[p];
            }
            out.push((i, mesh as usize, m));
        }
        out
    }

    pub fn node_name(&self, node: usize) -> Option<String> {
        self.json["nodes"][node]["name"].as_str().map(str::to_owned)
    }

    /// Every primitive of a mesh, transformed by `m`.
    pub fn prims(&self, mesh: usize, m: &[f32; 16], node: Option<String>) -> Vec<Prim> {
        let mut out = Vec::new();
        for (pi, p) in self.json["meshes"][mesh]["primitives"].as_array().unwrap().iter().enumerate() {
            let attr = &p["attributes"];
            let get = |k: &str, n: usize| attr.get(k).map(|a| self.floats(a.as_u64().unwrap() as usize, n));
            let pos = get("POSITION", 3).unwrap();
            let mat = p.get("material").and_then(Value::as_u64).map(|i| &self.json["materials"][i as usize]);
            let mut pr = Prim {
                mesh,
                prim: pi,
                node: node.clone(),
                material: mat.and_then(|m| m["name"].as_str()).unwrap_or("").to_owned(),
                double_sided: mat.and_then(|m| m["doubleSided"].as_bool()).unwrap_or(false),
                alpha_mask: mat.and_then(|m| m["alphaMode"].as_str()) == Some("MASK"),
                ..Default::default()
            };
            let xf = |v: [f32; 3], w: f32| -> [f32; 3] {
                [
                    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * w,
                    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * w,
                    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * w,
                ]
            };
            let norm = |v: [f32; 3]| -> [f32; 3] {
                let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-20);
                [v[0] / l, v[1] / l, v[2] / l]
            };
            pr.pos = pos.chunks(3).map(|c| xf([c[0], c[1], c[2]], 1.0)).collect();
            if let Some(n) = get("NORMAL", 3) {
                // Normal matrix ignored: the probe content uses uniform scales.
                pr.nrm = n.chunks(3).map(|c| norm(xf([c[0], c[1], c[2]], 0.0))).collect();
            }
            if let Some(t) = get("TANGENT", 4) {
                pr.tan = t
                    .chunks(4)
                    .map(|c| {
                        let v = norm(xf([c[0], c[1], c[2]], 0.0));
                        [v[0], v[1], v[2], c[3]]
                    })
                    .collect();
            }
            if let Some(t) = get("TEXCOORD_0", 2) {
                pr.uv = t.chunks(2).map(|c| [c[0], c[1]]).collect();
            }
            if let Some(t) = get("_ID", 1) {
                pr.id = t.iter().map(|&x| x.round() as u32).collect();
            }
            pr.idx = match p.get("indices") {
                Some(a) => self.indices(a.as_u64().unwrap() as usize),
                None => (0..pr.pos.len() as u32).collect(),
            };
            out.push(pr);
        }
        out
    }
}
