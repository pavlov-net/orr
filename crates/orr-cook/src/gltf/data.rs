//! A glTF document with its buffers ready to read.
//!
//! The JSON is parsed first, and the extensions the cook implements itself are removed from
//! `extensionsRequired`, so the `gltf` crate validates the rest of the document. Buffer views
//! compressed with `EXT_meshopt_compression` or `KHR_meshopt_compression` are decoded with
//! meshoptimizer's decoders and filters into their fallback buffers, so every view reads as
//! if it were stored plain. Accessors are read here, including the normalised and
//! unnormalised integer forms `KHR_mesh_quantization` allows and sparse accessors.

use gltf::accessor::{DataType, Dimensions};
use gltf::buffer::Source as BufferSource;
use meshopt::ffi;
use serde_json::Value;

/// Extensions the cook reads itself, removed from `extensionsRequired` before validation.
const COOK_EXTENSIONS: &[&str] = &[
    "KHR_mesh_quantization",
    "KHR_meshopt_compression",
    "EXT_meshopt_compression",
    "EXT_mesh_gpu_instancing",
    "KHR_materials_anisotropy",
    "KHR_materials_clearcoat",
    "KHR_materials_diffuse_transmission",
    "KHR_materials_dispersion",
    "KHR_materials_displacement",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_iridescence",
    "KHR_materials_sheen",
    "KHR_materials_specular",
    "KHR_materials_transmission",
    "KHR_materials_volume",
];

const MESHOPT: [&str; 2] = ["EXT_meshopt_compression", "KHR_meshopt_compression"];

pub(crate) struct Document {
    pub doc: gltf::Document,
    /// The parsed JSON, for extensions the `gltf` crate does not model.
    pub json: Value,
    pub buffers: Vec<Vec<u8>>,
}

/// Opens a `.gltf` or `.glb` file's bytes; `load` reads a URI (a data URI or a file next to
/// the document) and names what it loads in errors.
pub(crate) fn open(
    bytes: &[u8],
    load: &mut dyn FnMut(&str, &str) -> crate::Result<Vec<u8>>,
    fail: &dyn Fn(String) -> crate::Error,
) -> crate::Result<Document> {
    let (json_bytes, bin) = if bytes.starts_with(b"glTF") {
        let glb = gltf::Glb::from_slice(bytes).map_err(|e| fail(e.to_string()))?;
        (
            glb.json.into_owned(),
            glb.bin.map(std::borrow::Cow::into_owned),
        )
    } else {
        (bytes.to_vec(), None)
    };
    let mut json: Value = serde_json::from_slice(&json_bytes).map_err(|e| fail(e.to_string()))?;
    if let Some(Value::Array(required)) = json.get_mut("extensionsRequired") {
        required.retain(|e| !e.as_str().is_some_and(|n| COOK_EXTENSIONS.contains(&n)));
    }
    let root: gltf::json::Root =
        serde_json::from_value(json.clone()).map_err(|e| fail(e.to_string()))?;
    let doc = gltf::Document::from_json(root).map_err(|e| fail(e.to_string()))?;

    let mut buffers = Vec::new();
    for b in doc.buffers() {
        let j = &json["buffers"][b.index()];
        let fallback = MESHOPT
            .iter()
            .any(|e| j["extensions"][e]["fallback"].as_bool() == Some(true));
        let data = if fallback {
            vec![0; b.length()]
        } else {
            match b.source() {
                BufferSource::Bin if b.index() == 0 => bin
                    .clone()
                    .ok_or_else(|| fail("GLB binary chunk missing".into()))?,
                BufferSource::Bin => {
                    return Err(fail(format!("buffer {} has no data", b.index())));
                }
                BufferSource::Uri(uri) => load(uri, &format!("buffer {}", b.index()))?,
            }
        };
        if data.len() < b.length() {
            return Err(fail(format!(
                "buffer {} holds {} bytes, declares {}",
                b.index(),
                data.len(),
                b.length()
            )));
        }
        buffers.push(data);
    }
    let mut d = Document { doc, json, buffers };
    d.check_views().map_err(&fail)?;
    d.decode_views().map_err(&fail)?;
    Ok(d)
}

fn field(v: &Value, k: &str) -> usize {
    v[k].as_u64().unwrap_or(0) as usize
}

impl Document {
    /// Every buffer view lies inside its buffer, and every accessor's stride holds its
    /// element and its last element ends inside its view.
    fn check_views(&self) -> Result<(), String> {
        for v in self.doc.views() {
            if v.offset() + v.length() > v.buffer().length() {
                return Err(format!("buffer view {} runs past its buffer", v.index()));
            }
        }
        for a in self.doc.accessors() {
            let Some(v) = a.view() else { continue };
            let size = a.size();
            let stride = v.stride().unwrap_or(size);
            if stride < size {
                return Err(format!(
                    "accessor {}: stride {stride} is below its {size}-byte element",
                    a.index()
                ));
            }
            if a.count() > 0 && a.offset() + stride * (a.count() - 1) + size > v.length() {
                return Err(format!("accessor {} runs past its buffer view", a.index()));
            }
        }
        Ok(())
    }

    /// Decodes every meshopt-compressed view into its place in its fallback buffer.
    fn decode_views(&mut self) -> Result<(), String> {
        for v in self.doc.views() {
            let j = &self.json["bufferViews"][v.index()]["extensions"];
            let Some(e) = MESHOPT.iter().find_map(|n| j.get(*n)) else {
                continue;
            };
            let what = format!("buffer view {}", v.index());
            let (count, stride) = (field(e, "count"), field(e, "byteStride"));
            let (src_buffer, at, len) = (
                field(e, "buffer"),
                field(e, "byteOffset"),
                field(e, "byteLength"),
            );
            let src = self
                .buffers
                .get(src_buffer)
                .and_then(|b| b.get(at..at.checked_add(len)?))
                .ok_or_else(|| format!("{what}: compressed range outside its buffer"))?
                .to_vec();
            let size = count
                .checked_mul(stride)
                .filter(|&s| s <= v.length())
                .ok_or_else(|| format!("{what}: {count} x {stride} B exceed the view"))?;
            let mode = e["mode"].as_str().unwrap_or_default();
            let filter = e["filter"].as_str().unwrap_or("NONE");
            let valid = match mode {
                "ATTRIBUTES" => stride % 4 == 0 && (4..=256).contains(&stride),
                "TRIANGLES" => (stride == 2 || stride == 4) && count % 3 == 0,
                "INDICES" => stride == 2 || stride == 4,
                _ => false,
            } && match filter {
                "NONE" => true,
                "OCTAHEDRAL" | "COLOR" => mode == "ATTRIBUTES" && (stride == 4 || stride == 8),
                "QUATERNION" => mode == "ATTRIBUTES" && stride == 8,
                "EXPONENTIAL" => mode == "ATTRIBUTES",
                _ => false,
            };
            if !valid {
                return Err(format!(
                    "{what}: meshopt mode {mode} filter {filter} with stride {stride}"
                ));
            }
            let mut dst = vec![0u8; size];
            let (dp, sp) = (dst.as_mut_ptr().cast(), src.as_ptr());
            // SAFETY: `dst` holds count x stride bytes and the parameters were checked
            // against each decoder's requirements; the decoders are safe on any input.
            let rc = unsafe {
                match mode {
                    "ATTRIBUTES" => {
                        ffi::meshopt_decodeVertexBuffer(dp, count, stride, sp, src.len())
                    }
                    "TRIANGLES" => ffi::meshopt_decodeIndexBuffer(dp, count, stride, sp, src.len()),
                    _ => ffi::meshopt_decodeIndexSequence(dp, count, stride, sp, src.len()),
                }
            };
            if rc != 0 {
                return Err(format!("{what}: meshopt stream does not decode"));
            }
            // SAFETY: the filters run in place over the `count` decoded elements.
            unsafe {
                match filter {
                    "OCTAHEDRAL" => ffi::meshopt_decodeFilterOct(dp, count, stride),
                    "QUATERNION" => ffi::meshopt_decodeFilterQuat(dp, count, stride),
                    "EXPONENTIAL" => ffi::meshopt_decodeFilterExp(dp, count, stride),
                    "COLOR" => ffi::meshopt_decodeFilterColor(dp, count, stride),
                    _ => {}
                }
            }
            let out = &mut self.buffers[v.buffer().index()];
            out[v.offset()..v.offset() + size].copy_from_slice(&dst);
        }
        Ok(())
    }

    /// An accessor as `count` rows of `n` floats: normalised integers map to 0..1 or -1..1,
    /// other integers to their value; components past the accessor's read 0.
    pub(crate) fn floats(&self, a: &gltf::Accessor<'_>, n: usize) -> Result<Vec<f32>, String> {
        let comps = match a.dimensions() {
            Dimensions::Scalar => 1,
            Dimensions::Vec2 => 2,
            Dimensions::Vec3 => 3,
            Dimensions::Vec4 => 4,
            d => return Err(format!("accessor {}: {d:?} is not a vector", a.index())),
        };
        let ty = a.data_type();
        let normalized = a.normalized();
        let read = |b: &[u8], out: &mut [f32]| {
            for (c, o) in out.iter_mut().enumerate().take(comps.min(n)) {
                *o = component(b, ty, c, normalized);
            }
        };
        let mut out = vec![0.0; a.count() * n];
        if let Some(v) = a.view() {
            let (data, stride) = self.view_elements(&v, a.size());
            let base = &data[a.offset()..];
            for (i, row) in out.chunks_exact_mut(n).enumerate() {
                read(&base[i * stride..], row);
            }
        }
        if let Some(s) = a.sparse() {
            let indices = self.sparse_indices(&s, a.count())?;
            let values = s.values();
            let (data, stride) = self.view_elements(&values.view(), a.size());
            let base = data
                .get(values.offset()..)
                .filter(|b| b.len() >= stride * s.count().saturating_sub(1) + a.size())
                .ok_or_else(|| format!("accessor {}: sparse values outside", a.index()))?;
            for (k, &i) in indices.iter().enumerate() {
                read(&base[k * stride..], &mut out[i * n..(i + 1) * n]);
            }
        }
        Ok(out)
    }

    /// An index or other unsigned integer scalar accessor.
    pub(crate) fn indices(&self, a: &gltf::Accessor<'_>) -> Result<Vec<u32>, String> {
        if a.dimensions() != Dimensions::Scalar
            || !matches!(a.data_type(), DataType::U8 | DataType::U16 | DataType::U32)
        {
            return Err(format!(
                "accessor {}: not unsigned integer scalars",
                a.index()
            ));
        }
        if a.sparse().is_some() {
            return Ok(self.floats(a, 1)?.into_iter().map(|v| v as u32).collect());
        }
        let Some(v) = a.view() else {
            return Ok(vec![0; a.count()]);
        };
        let (data, stride) = self.view_elements(&v, a.size());
        let base = &data[a.offset()..];
        Ok((0..a.count())
            .map(|i| {
                let b = &base[i * stride..];
                match a.data_type() {
                    DataType::U8 => u32::from(b[0]),
                    DataType::U16 => u32::from(u16::from_le_bytes([b[0], b[1]])),
                    _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                }
            })
            .collect())
    }

    fn view_elements(&self, v: &gltf::buffer::View<'_>, size: usize) -> (&[u8], usize) {
        let data = &self.buffers[v.buffer().index()][v.offset()..v.offset() + v.length()];
        (data, v.stride().unwrap_or(size))
    }

    fn sparse_indices(
        &self,
        s: &gltf::accessor::sparse::Sparse<'_>,
        count: usize,
    ) -> Result<Vec<usize>, String> {
        use gltf::accessor::sparse::IndexType;
        let ix = s.indices();
        let size = match ix.index_type() {
            IndexType::U8 => 1,
            IndexType::U16 => 2,
            IndexType::U32 => 4,
        };
        let (data, _) = self.view_elements(&ix.view(), size);
        let b = data
            .get(ix.offset()..ix.offset() + size * s.count())
            .ok_or("sparse indices outside their view")?;
        let out: Vec<usize> = b
            .chunks_exact(size)
            .map(|c| match size {
                1 => usize::from(c[0]),
                2 => usize::from(u16::from_le_bytes([c[0], c[1]])),
                _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize,
            })
            .collect();
        if out.iter().any(|&i| i >= count) {
            return Err("sparse index past the accessor".into());
        }
        Ok(out)
    }
}

fn component(b: &[u8], ty: DataType, c: usize, normalized: bool) -> f32 {
    match ty {
        DataType::I8 => {
            let x = f32::from(b[c] as i8);
            if normalized { (x / 127.0).max(-1.0) } else { x }
        }
        DataType::U8 => {
            let x = f32::from(b[c]);
            if normalized { x / 255.0 } else { x }
        }
        DataType::I16 => {
            let x = f32::from(i16::from_le_bytes([b[2 * c], b[2 * c + 1]]));
            if normalized {
                (x / 32767.0).max(-1.0)
            } else {
                x
            }
        }
        DataType::U16 => {
            let x = f32::from(u16::from_le_bytes([b[2 * c], b[2 * c + 1]]));
            if normalized { x / 65535.0 } else { x }
        }
        DataType::U32 => u32::from_le_bytes(b[4 * c..4 * c + 4].try_into().unwrap()) as f32,
        DataType::F32 => f32::from_le_bytes(b[4 * c..4 * c + 4].try_into().unwrap()),
    }
}
