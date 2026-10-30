//! A GLB writer for test fixtures: the caller adds accessors and writes the rest of the
//! document as JSON.

use std::path::Path;

#[derive(Default)]
pub(crate) struct Glb {
    bin: Vec<u8>,
    views: Vec<String>,
    accessors: Vec<String>,
    /// Byte length of the meshopt fallback buffer (buffer 1), when views are compressed.
    fallback: usize,
}

const F32: u32 = 5126;
pub(crate) const U32: u32 = 5125;
pub(crate) const U16: u32 = 5123;
pub(crate) const I8: u32 = 5120;

impl Glb {
    fn push(&mut self, bytes: &[u8]) -> (usize, usize) {
        let at = self.bin.len();
        self.bin.extend_from_slice(bytes);
        self.bin.resize(self.bin.len().next_multiple_of(4), 0);
        (at, bytes.len())
    }

    /// A buffer view of `bytes`, with a stride when given.
    pub(crate) fn view(&mut self, bytes: &[u8], stride: Option<usize>) -> usize {
        let (at, len) = self.push(bytes);
        let stride = stride.map_or(String::new(), |s| format!(r#","byteStride":{s}"#));
        self.views.push(format!(
            r#"{{"buffer":0,"byteOffset":{at},"byteLength":{len}{stride}}}"#
        ));
        self.views.len() - 1
    }

    /// A view compressed with EXT_meshopt_compression into the fallback buffer: `mode` is
    /// ATTRIBUTES (vertex codec), TRIANGLES or INDICES.
    pub(crate) fn compressed_view(
        &mut self,
        raw: &[u8],
        stride: usize,
        mode: &str,
        byte_stride: bool,
    ) -> usize {
        let count = raw.len() / stride;
        let encoded = match mode {
            "ATTRIBUTES" => {
                // SAFETY: bound query, then an encode of `count` vertices of `stride` bytes.
                let bound = unsafe { meshopt::ffi::meshopt_encodeVertexBufferBound(count, stride) };
                let mut out = vec![0u8; bound];
                // SAFETY: `out` holds the bound and `raw` holds `count` vertices.
                let len = unsafe {
                    meshopt::ffi::meshopt_encodeVertexBuffer(
                        out.as_mut_ptr(),
                        bound,
                        raw.as_ptr().cast(),
                        count,
                        stride,
                    )
                };
                out.truncate(len);
                out
            }
            "TRIANGLES" => {
                let indices: Vec<u32> = bytemuck::cast_slice(raw).to_vec();
                let vertices = indices.iter().max().map_or(0, |&m| m as usize + 1);
                meshopt::encode_index_buffer(&indices, vertices).unwrap()
            }
            _ => panic!("mode {mode}"),
        };
        let (at, len) = self.push(&encoded);
        let target = self.fallback;
        self.fallback = (self.fallback + raw.len()).next_multiple_of(4);
        let stride_field = if byte_stride {
            format!(r#","byteStride":{stride}"#)
        } else {
            String::new()
        };
        self.views.push(format!(
            r#"{{"buffer":1,"byteOffset":{target},"byteLength":{}{stride_field},"extensions":{{"EXT_meshopt_compression":{{"buffer":0,"byteOffset":{at},"byteLength":{len},"byteStride":{stride},"count":{count},"mode":"{mode}"}}}}}}"#,
            raw.len()
        ));
        self.views.len() - 1
    }

    /// An accessor over `view`; `extra` adds fields such as `"normalized":true` or min/max.
    pub(crate) fn accessor(
        &mut self,
        view: usize,
        component: u32,
        count: usize,
        ty: &str,
        extra: &str,
    ) -> usize {
        self.accessors.push(format!(
            r#"{{"bufferView":{view},"componentType":{component},"count":{count},"type":"{ty}"{extra}}}"#
        ));
        self.accessors.len() - 1
    }

    /// Plain f32 rows; positions get their min and max.
    pub(crate) fn floats<const N: usize>(&mut self, rows: &[[f32; N]], ty: &str) -> usize {
        let view = self.view(bytemuck::cast_slice(rows), None);
        let extra = if ty == "VEC3" {
            let lo: [f32; N] =
                std::array::from_fn(|a| rows.iter().map(|v| v[a]).fold(f32::MAX, f32::min));
            let hi: [f32; N] =
                std::array::from_fn(|a| rows.iter().map(|v| v[a]).fold(f32::MIN, f32::max));
            format!(r#","min":{lo:?},"max":{hi:?}"#)
        } else {
            String::new()
        };
        self.accessor(view, F32, rows.len(), ty, &extra)
    }

    pub(crate) fn indices(&mut self, idx: &[u32]) -> usize {
        let view = self.view(bytemuck::cast_slice(idx), None);
        self.accessor(view, U32, idx.len(), "SCALAR", "")
    }

    /// Embeds an image file's bytes as a buffer view; returns the view.
    pub(crate) fn image(&mut self, bytes: &[u8]) -> usize {
        self.view(bytes, None)
    }

    /// Writes the GLB; `rest` is the document's other top-level members, without braces.
    pub(crate) fn write(mut self, path: &Path, rest: &str) {
        let mut buffers = format!(r#"{{"byteLength":{}}}"#, self.bin.len());
        if self.fallback > 0 {
            buffers += &format!(
                r#",{{"byteLength":{},"extensions":{{"EXT_meshopt_compression":{{"fallback":true}}}}}}"#,
                self.fallback
            );
        }
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},{rest},"buffers":[{buffers}],"bufferViews":[{}],"accessors":[{}]}}"#,
            self.views.join(","),
            self.accessors.join(",")
        );
        let mut json = json.into_bytes();
        json.resize(json.len().next_multiple_of(4), b' ');
        self.bin.resize(self.bin.len().next_multiple_of(4), 0);
        let mut glb = Vec::new();
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&((12 + 8 + json.len() + 8 + self.bin.len()) as u32).to_le_bytes());
        glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json);
        glb.extend_from_slice(&(self.bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&self.bin);
        std::fs::write(path, glb).unwrap();
    }
}
