//! A glTF document and the bytes of the buffer views a caller reads.
//!
//! The JSON is parsed first, and the extensions the cook implements itself are removed from
//! `extensionsRequired`, so the `gltf` crate validates the rest of the document, but for two
//! things it refuses that glTF allows: an accessor with no element source reads as zeros and
//! a primitive without positions is left to the importer to skip. Buffers stay
//! where they are: [`Data::load`] reads the byte ranges of the views a caller needs, and
//! decodes views compressed with `EXT_meshopt_compression` or `KHR_meshopt_compression` with
//! meshoptimizer's decoders and filters, so every view reads as if it were stored plain.
//! Accessors are read here, including the normalized and unnormalized integer forms
//! `KHR_mesh_quantization` allows and sparse accessors. A texture's `EXT_texture_webp` image
//! replaces its core `source`, so a texture may carry only the extension's source. An asset
//! whose `minVersion` is not glTF 2.0 or 2.1, or without one whose `version` has a major other
//! than 2, is refused, as is a document whose `weights` arrays do not match its morph targets.
//! A morph target reads from the JSON, as it may name any attribute and the `gltf` crate models
//! three.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use base64::Engine;
use gltf::accessor::{DataType, Dimensions};
use gltf::buffer::Source as BufferSource;
use meshopt::ffi;
use orr_pack::Hash32;
use percent_encoding::percent_decode_str;
use serde_json::Value;

/// Extensions the cook reads itself, removed from `extensionsRequired` before validation.
const COOK_EXTENSIONS: &[&str] = &[
    "KHR_mesh_quantization",
    "KHR_meshopt_compression",
    "EXT_meshopt_compression",
    "EXT_mesh_gpu_instancing",
    "EXT_texture_webp",
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
    "KHR_node_visibility",
    "KHR_texture_transform",
];

const MESHOPT: [&str; 2] = ["EXT_meshopt_compression", "KHR_meshopt_compression"];

/// Where a buffer's bytes are.
pub(super) enum Buffer {
    /// A meshopt fallback buffer: its views decode from other buffers.
    Fallback,
    /// The GLB binary chunk or a data URI, and its hash.
    Bytes(Vec<u8>, Hash32),
    /// A file next to the document.
    File(PathBuf),
}

/// A parsed glTF document. Buffer bytes are read by [`Data::load`].
pub(crate) struct Document {
    pub(crate) doc: gltf::Document,
    /// The parsed JSON, for extensions the `gltf` crate does not model.
    pub(super) json: Value,
    pub(super) buffers: Vec<Buffer>,
}

/// Whether the cook reads this asset: its `minVersion` when it has one, else its `version`.
fn check_version(asset: &Value) -> Result<(), String> {
    let parse = |v: &Value| {
        let (major, minor) = v.as_str()?.split_once('.')?;
        Some((major.parse::<u32>().ok()?, minor.parse::<u32>().ok()?))
    };
    if let Some(v) = asset.get("minVersion") {
        if parse(v).is_none_or(|m| m.0 != 2 || m.1 > 1) {
            return Err(format!("asset minVersion {v} is not glTF 2.0 or 2.1"));
        }
    } else if parse(&asset["version"]).is_none_or(|v| v.0 != 2) {
        return Err(format!(
            "asset version {} is not a glTF 2 version",
            asset["version"]
        ));
    }
    Ok(())
}

/// Each morph target of the document holds accessors of it.
fn check_morph_targets(json: &Value, accessors: usize) -> Result<(), String> {
    let primitives = json["meshes"].as_array().into_iter().flatten();
    let primitives = primitives.flat_map(|m| m["primitives"].as_array().into_iter().flatten());
    let targets = primitives.flat_map(|p| p["targets"].as_array().into_iter().flatten());
    for (name, accessor) in targets.flat_map(|t| t.as_object().into_iter().flatten()) {
        if accessor.as_u64().is_none_or(|a| a >= accessors as u64) {
            return Err(format!("morph target {name}: no accessor {accessor}"));
        }
    }
    Ok(())
}

/// The morph targets of a mesh: as many as its first primitive that has any. The other
/// primitives that have any must have as many, and a `weights` array holds one per target.
fn check_morph_weights(doc: &gltf::Document) -> Result<(), String> {
    let targets = |mesh: &gltf::Mesh<'_>| {
        let counts = mesh.primitives().map(|p| p.morph_targets().len());
        counts.max().unwrap_or(0)
    };
    let check = |what: &str, index: usize, weights: Option<&[f32]>, n: usize| match weights {
        Some(w) if w.len() != n => Err(format!(
            "{what} {index}: {} weights for {n} morph targets",
            w.len()
        )),
        _ => Ok(()),
    };
    for mesh in doc.meshes() {
        let n = targets(&mesh);
        let mut counts = mesh.primitives().map(|p| p.morph_targets().len());
        if counts.any(|c| c != 0 && c != n) {
            return Err(format!(
                "mesh {}: primitives differ in their morph target counts",
                mesh.index()
            ));
        }
        check("mesh", mesh.index(), mesh.weights(), n)?;
    }
    for node in doc.nodes() {
        let n = node.mesh().map_or(0, |m| targets(&m));
        check("node", node.index(), node.weights(), n)?;
    }
    Ok(())
}

/// Readies the images for the `gltf` crate, which panics on a buffer view's image without a
/// `mimeType` and on an image with no source. Images are recognized by their bytes, so the
/// former reads as any other; the latter has nothing to read.
fn check_images(json: &mut Value) -> Result<(), String> {
    let Some(Value::Array(images)) = json.get_mut("images") else {
        return Ok(());
    };
    for (i, image) in images.iter_mut().enumerate() {
        if image["bufferView"].is_null() {
            if image["uri"].is_null() {
                return Err(format!("image {i} has neither a uri nor a buffer view"));
            }
        } else if image["mimeType"].is_null() {
            image["mimeType"] = "application/octet-stream".into();
        }
    }
    Ok(())
}

/// Validates as `gltf::Document::from_json` does, less what the module documentation lists.
fn validate(root: &gltf::json::Root) -> Result<(), String> {
    use gltf::json::validation::{Error, Validate};
    let mut errors = Vec::new();
    root.validate(root, gltf::json::Path::new, &mut |path, error| {
        let path = path();
        let at = path.as_str();
        let no_positions = at.ends_with(r#".attributes["POSITION"]"#);
        let no_source = at.starts_with("accessors[") && at.ends_with(".bufferView");
        if error != Error::Missing || !(no_positions || no_source) {
            errors.push(format!("{path}: {error}"));
        }
    });
    match errors.is_empty() {
        true => Ok(()),
        false => Err(errors.join("; ")),
    }
}

/// Opens a `.gltf` or `.glb` file's bytes; buffer URIs resolve against `dir`.
pub(super) fn open(bytes: &[u8], dir: &Path) -> Result<Document, String> {
    let (json_bytes, mut bin) = if bytes.starts_with(b"glTF") {
        let glb = gltf::Glb::from_slice(bytes).map_err(|e| e.to_string())?;
        (glb.json.into_owned(), glb.bin.map(Cow::into_owned))
    } else {
        (bytes.to_vec(), None)
    };
    let mut json: Value = serde_json::from_slice(&json_bytes).map_err(|e| e.to_string())?;
    check_version(&json["asset"])?;
    if let Some(Value::Array(required)) = json.get_mut("extensionsRequired") {
        required.retain(|e| !e.as_str().is_some_and(|n| COOK_EXTENSIONS.contains(&n)));
    }
    if let Some(Value::Array(textures)) = json.get_mut("textures") {
        for t in textures {
            if let Some(webp) = t.pointer("/extensions/EXT_texture_webp/source").cloned() {
                t["source"] = webp;
            }
        }
    }
    check_images(&mut json)?;
    let root: gltf::json::Root = serde_json::from_value(json.clone()).map_err(|e| e.to_string())?;
    validate(&root)?;
    let doc = gltf::Document::from_json_without_validation(root);
    check_morph_weights(&doc)?;
    check_morph_targets(&json, doc.accessors().len())?;
    let mut buffers = Vec::new();
    for b in doc.buffers() {
        let j = &json["buffers"][b.index()];
        let fallback = MESHOPT
            .iter()
            .any(|e| j["extensions"][e]["fallback"].as_bool() == Some(true));
        let data = match b.source() {
            _ if fallback => {
                buffers.push(Buffer::Fallback);
                continue;
            }
            BufferSource::Bin if b.index() == 0 => bin.take().ok_or("GLB binary chunk missing")?,
            BufferSource::Bin => return Err(format!("buffer {} has no data", b.index())),
            BufferSource::Uri(uri) => match data_uri(uri) {
                Some(data) => data.map_err(|e| format!("buffer {}: {e}", b.index()))?,
                None => {
                    buffers.push(Buffer::File(uri_path(dir, uri)));
                    continue;
                }
            },
        };
        if data.len() < b.length() {
            return Err(format!(
                "buffer {} holds {} bytes, declares {}",
                b.index(),
                data.len(),
                b.length()
            ));
        }
        let hash = Hash32::of(&data);
        buffers.push(Buffer::Bytes(data, hash));
    }
    let d = Document { doc, json, buffers };
    d.check_views()?;
    Ok(d)
}

/// `uri` without the stray whitespace and control characters glTF recommends removing.
fn clean_uri(uri: &str) -> Cow<'_, str> {
    let stray = ['\t', '\n', '\r'];
    let uri = uri.trim_matches(|c| c <= ' ');
    match uri.contains(stray) {
        true => uri.replace(stray, "").into(),
        false => uri.into(),
    }
}

/// The file that relative path `uri` names under `dir`.
pub(super) fn uri_path(dir: &Path, uri: &str) -> PathBuf {
    let path = clean_uri(uri).replace('\\', "/");
    dir.join(&*percent_decode_str(&path).decode_utf8_lossy())
}

/// The bytes of a data URL, base64 or percent-encoded; `None` for any other URI.
pub(super) fn data_uri(uri: &str) -> Option<Result<Vec<u8>, String>> {
    let uri = clean_uri(uri);
    let (scheme, data) = uri.split_at_checked("data:".len())?;
    if !scheme.eq_ignore_ascii_case("data:") {
        return None;
    }
    let Some((header, payload)) = data.split_once(',') else {
        return Some(Err("data URL has no comma before its data".into()));
    };
    let bytes = Cow::<[u8]>::from(percent_decode_str(payload));
    Some(if header.to_ascii_lowercase().ends_with(";base64") {
        base64::engine::general_purpose::STANDARD
            .decode(bytes)
            .map_err(|e| e.to_string())
    } else {
        Ok(bytes.into_owned())
    })
}

fn field(v: &Value, k: &str) -> usize {
    v[k].as_u64().unwrap_or(0) as usize
}

impl Document {
    /// The morph targets of primitive `prim` of mesh `mesh`.
    fn morph_targets(&self, mesh: usize, prim: usize) -> &[Value] {
        let targets = self.json["meshes"][mesh]["primitives"][prim]["targets"].as_array();
        targets.map_or(&[], Vec::as_slice)
    }

    fn accessor(&self, json: &Value) -> Option<gltf::Accessor<'_>> {
        self.doc.accessors().nth(json.as_u64()? as usize)
    }

    /// The deltas morph target `target` of primitive `prim` of mesh `mesh` holds for attribute
    /// `semantic`.
    pub(super) fn morph_deltas(
        &self,
        mesh: usize,
        prim: usize,
        target: usize,
        semantic: &str,
    ) -> Option<gltf::Accessor<'_>> {
        self.accessor(self.morph_targets(mesh, prim).get(target)?.get(semantic)?)
    }

    /// The accessors of the morph targets of primitive `prim` of mesh `mesh`.
    pub(super) fn morph_accessors(&self, mesh: usize, prim: usize) -> Vec<gltf::Accessor<'_>> {
        let targets = self.morph_targets(mesh, prim).iter();
        let attributes = targets.flat_map(|t| t.as_object().into_iter().flatten());
        attributes.filter_map(|(_, a)| self.accessor(a)).collect()
    }

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

    /// The meshopt compression object of buffer view `v`, if it has one.
    fn meshopt(&self, v: usize) -> Option<&Value> {
        let j = &self.json["bufferViews"][v]["extensions"];
        MESHOPT.iter().find_map(|n| j.get(*n))
    }

    /// The buffer and byte range buffer view `v`'s stored bytes are read from: its own, or
    /// with meshopt compression the compressed stream's.
    pub(super) fn view_source(&self, v: usize) -> (usize, usize, usize) {
        match self.meshopt(v) {
            Some(e) => (
                field(e, "buffer"),
                field(e, "byteOffset"),
                field(e, "byteLength"),
            ),
            None => {
                let view = self.doc.views().nth(v).expect("view index");
                (view.buffer().index(), view.offset(), view.length())
            }
        }
    }

    /// Adds the buffer views accessor `a` reads, sparse ones included.
    pub(super) fn accessor_views(a: &gltf::Accessor<'_>, out: &mut Vec<usize>) {
        out.extend(a.view().map(|v| v.index()));
        if let Some(s) = a.sparse() {
            out.push(s.indices().view().index());
            out.push(s.values().view().index());
        }
    }
}

/// The bytes of some buffer views of a document, meshopt-compressed views decoded.
pub(crate) struct Data<'d> {
    pub(crate) doc: &'d Document,
    views: HashMap<usize, Vec<u8>>,
}

impl<'d> Data<'d> {
    /// Reads and decodes `views`, naming `asset` in errors.
    pub(super) fn load(
        doc: &'d Document,
        views: impl IntoIterator<Item = usize>,
        asset: &str,
    ) -> crate::Result<Self> {
        let mut out = HashMap::new();
        let mut files = HashMap::new();
        for v in views {
            if out.contains_key(&v) {
                continue;
            }
            let fail = |m: String| crate::Error::Import {
                asset: asset.into(),
                message: format!("buffer view {v}: {m}"),
            };
            let (b, at, len) = doc.view_source(v);
            let end = at
                .checked_add(len)
                .ok_or_else(|| fail("range overflows".into()))?;
            let bytes = match doc.buffers.get(b) {
                None => return Err(fail(format!("buffer {b} does not exist"))),
                Some(Buffer::Fallback) => {
                    return Err(fail(format!(
                        "reads fallback buffer {b}, which holds no data"
                    )));
                }
                Some(Buffer::Bytes(data, _)) => data
                    .get(at..end)
                    .ok_or_else(|| fail("range outside its buffer".into()))?
                    .to_vec(),
                Some(Buffer::File(path)) => {
                    let read = |source| crate::Error::Read {
                        path: path.clone(),
                        source,
                    };
                    let file = match files.entry(path) {
                        std::collections::hash_map::Entry::Occupied(f) => f.into_mut(),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(orr_io::open_shared(path).map_err(read)?)
                        }
                    };
                    let mut bytes = vec![0; len];
                    orr_io::read_exact_at(file, at as u64, &mut bytes).map_err(read)?;
                    bytes
                }
            };
            let bytes = match doc.meshopt(v) {
                Some(e) => {
                    let length = doc.doc.views().nth(v).expect("view index").length();
                    decode_view(e, &bytes, length).map_err(fail)?
                }
                None => bytes,
            };
            out.insert(v, bytes);
        }
        Ok(Self { doc, views: out })
    }

    /// The bytes of a loaded buffer view.
    pub(super) fn view(&self, v: &gltf::buffer::View<'_>) -> Result<&[u8], String> {
        self.views
            .get(&v.index())
            .map(Vec::as_slice)
            .ok_or_else(|| format!("buffer view {} is not loaded", v.index()))
    }

    /// An accessor as `count` rows of `n` floats: normalized integers map to 0..1 or -1..1,
    /// other integers to their value; components past the accessor's read 0.
    pub(crate) fn floats(&self, a: &gltf::Accessor<'_>, n: usize) -> Result<Vec<f32>, String> {
        let comps = match a.dimensions() {
            Dimensions::Scalar => 1,
            Dimensions::Vec2 => 2,
            Dimensions::Vec3 => 3,
            Dimensions::Vec4 => 4,
            Dimensions::Mat4 => 16,
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
            let (data, stride) = self.view_elements(&v, a.size())?;
            let base = &data[a.offset()..];
            for (i, row) in out.chunks_exact_mut(n).enumerate() {
                read(&base[i * stride..], row);
            }
        }
        self.sparse_elements(a, |i, b| read(b, &mut out[i * n..(i + 1) * n]))?;
        Ok(out)
    }

    /// Calls `replace` with the index and bytes of each sparse element of `a`.
    fn sparse_elements(
        &self,
        a: &gltf::Accessor<'_>,
        mut replace: impl FnMut(usize, &[u8]),
    ) -> Result<(), String> {
        let Some(s) = a.sparse() else {
            return Ok(());
        };
        let indices = self.sparse_indices(&s, a.count())?;
        let values = s.values();
        let (data, stride) = self.view_elements(&values.view(), a.size())?;
        let base = data
            .get(values.offset()..)
            .filter(|b| b.len() >= stride * s.count().saturating_sub(1) + a.size())
            .ok_or_else(|| format!("accessor {}: sparse values outside", a.index()))?;
        for (k, &i) in indices.iter().enumerate() {
            replace(i, &base[k * stride..]);
        }
        Ok(())
    }

    /// An index or other unsigned integer scalar accessor.
    pub(super) fn indices(&self, a: &gltf::Accessor<'_>) -> Result<Vec<u32>, String> {
        if a.dimensions() != Dimensions::Scalar
            || !matches!(a.data_type(), DataType::U8 | DataType::U16 | DataType::U32)
        {
            return Err(format!(
                "accessor {}: not unsigned integer scalars",
                a.index()
            ));
        }
        let read = |b: &[u8]| match a.data_type() {
            DataType::U8 => u32::from(b[0]),
            DataType::U16 => u32::from(u16::from_le_bytes([b[0], b[1]])),
            _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        };
        let mut out = match a.view() {
            Some(v) => {
                let (data, stride) = self.view_elements(&v, a.size())?;
                let base = &data[a.offset()..];
                (0..a.count()).map(|i| read(&base[i * stride..])).collect()
            }
            None => vec![0; a.count()],
        };
        self.sparse_elements(a, |i, b| out[i] = read(b))?;
        Ok(out)
    }

    fn view_elements(
        &self,
        v: &gltf::buffer::View<'_>,
        size: usize,
    ) -> Result<(&[u8], usize), String> {
        Ok((self.view(v)?, v.stride().unwrap_or(size)))
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
        let (data, _) = self.view_elements(&ix.view(), size)?;
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

/// Decodes one meshopt-compressed view from its compressed stream into `length` bytes.
fn decode_view(e: &Value, src: &[u8], length: usize) -> Result<Vec<u8>, String> {
    let (count, stride) = (field(e, "count"), field(e, "byteStride"));
    count
        .checked_mul(stride)
        .filter(|&s| s <= length)
        .ok_or_else(|| format!("{count} x {stride} B exceed the view"))?;
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
            "meshopt mode {mode} filter {filter} with stride {stride}"
        ));
    }
    let mut dst = vec![0u8; length];
    let (dp, sp) = (dst.as_mut_ptr().cast(), src.as_ptr());
    // SAFETY: `dst` holds at least count x stride bytes and the parameters were checked
    // against each decoder's requirements; the decoders are safe on any input.
    let rc = unsafe {
        match mode {
            "ATTRIBUTES" => ffi::meshopt_decodeVertexBuffer(dp, count, stride, sp, src.len()),
            "TRIANGLES" => ffi::meshopt_decodeIndexBuffer(dp, count, stride, sp, src.len()),
            _ => ffi::meshopt_decodeIndexSequence(dp, count, stride, sp, src.len()),
        }
    };
    if rc != 0 {
        return Err("meshopt stream does not decode".into());
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
    Ok(dst)
}

#[cfg(test)]
mod tests {
    /// L1738 allows uint32 indices, and 2^24 + 1 has no f32.
    #[test]
    fn sparse_u32_indices_stay_exact() {
        let json = r#"{"asset":{"version":"2.0"},
"buffers":[{"byteLength":8,"uri":"data:,%01%00%00%00%01%00%00%01"}],
"bufferViews":[{"buffer":0,"byteLength":4},{"buffer":0,"byteOffset":4,"byteLength":4}],
"accessors":[{"componentType":5125,"count":2,"type":"SCALAR","sparse":{"count":1,
"indices":{"bufferView":0,"componentType":5125},"values":{"bufferView":1}}}]}"#;
        let doc = super::open(json.as_bytes(), std::path::Path::new("")).unwrap();
        let data = super::Data::load(&doc, [0, 1], "g").unwrap();
        let accessor = doc.doc.accessors().next().unwrap();
        assert_eq!(data.indices(&accessor).unwrap(), [0, (1 << 24) + 1]);
    }

    #[test]
    fn min_version_other_than_2_0_or_2_1_is_refused() {
        let opens = |min: &str| {
            let json = format!(r#"{{"asset":{{"version":"2.1","minVersion":"{min}"}}}}"#);
            super::open(json.as_bytes(), std::path::Path::new("")).is_ok()
        };
        assert!(opens("2.0") && opens("2.1"));
        assert!(!opens("2.2") && !opens("2.10") && !opens("3.0") && !opens("2") && !opens("1.0"));
    }

    #[test]
    fn version_major_other_than_2_is_refused_without_min_version() {
        let opens = |asset: &str| {
            let json = format!(r#"{{"asset":{asset}}}"#);
            super::open(json.as_bytes(), std::path::Path::new("")).is_ok()
        };
        assert!(opens(r#"{"version":"2.0"}"#) && opens(r#"{"version":"2.9"}"#));
        assert!(!opens(r#"{"version":"1.0"}"#) && !opens(r#"{"version":"3.0"}"#));
        assert!(!opens(r#"{"version":"2"}"#) && !opens(r#"{"version":2.0}"#) && !opens("{}"));
        // A newer asset that a reader of 2.0 can load.
        assert!(opens(r#"{"version":"3.0","minVersion":"2.0"}"#));
    }
}
