//! OpenSubdiv Far through the C shim in `osd.cpp`.

use std::ffi::{CStr, c_char, c_int, c_void};

use super::{Boundary, CageInput, SubdSettings};

#[repr(C)]
struct Cage {
    vertex_count: c_int,
    face_count: c_int,
    face_counts: *const c_int,
    face_indices: *const c_int,
    crease_count: c_int,
    crease_pairs: *const c_int,
    crease_sharpness: *const f32,
    corner_count: c_int,
    corners: *const c_int,
    corner_sharpness: *const f32,
    hole_count: c_int,
    holes: *const c_int,
    left_handed: c_int,
    boundary: c_int,
    uv_linear: c_int,
    triangles: c_int,
    uv_count: c_int,
    uv_indices: *const c_int,
    isolation: c_int,
    smooth_isolation: c_int,
}

#[repr(C)]
struct Osd {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    fn orr_osd_build(cage: *const Cage, error: *mut c_char, error_len: c_int) -> *mut Osd;
    fn orr_osd_free(osd: *mut Osd);
    fn orr_osd_table(osd: *const Osd, table: c_int, data: *mut *const c_void) -> c_int;
    #[cfg(test)]
    fn orr_osd_limit(
        cage: *const Cage,
        points: *const f32,
        uvs: *const f32,
        count: c_int,
        ptex: *const c_int,
        s: *const f32,
        t: *const f32,
        out_points: *mut f32,
        out_uvs: *mut f32,
        error: *mut c_char,
        error_len: c_int,
    ) -> c_int;
}

const STENCIL_SIZES: c_int = 0;
const PATCH_TYPES: c_int = 3;
const UV: c_int = 6;

/// Points as stencils over the cage: point `i` is the sum of `weights[t] x cage[indices[t]]`
/// for `t` in `offsets[i]..offsets[i + 1]`. The first points are the cage's own.
pub(super) struct Stencils {
    pub(super) offsets: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) weights: Vec<f32>,
}

impl Stencils {
    pub(super) fn terms(&self, point: usize) -> std::ops::Range<usize> {
        self.offsets[point] as usize..self.offsets[point + 1] as usize
    }

    pub(super) fn apply<const N: usize>(&self, cage: &[[f32; N]]) -> Vec<[f32; N]> {
        orr_pack::subd::apply_stencils(&self.offsets, &self.indices, &self.weights, cage)
    }
}

/// One patch table side (positions or uvs): per patch its type, points and Far patch param.
pub(super) struct Patches {
    pub(super) types: Vec<u8>,
    /// Points of patch `p` are `cvs[first[p]..first[p + 1]]`, indices into the stencils.
    pub(super) first: Vec<usize>,
    pub(super) cvs: Vec<u32>,
    /// Far's two packed `PatchParam` words.
    pub(super) params: Vec<[u32; 2]>,
}

pub(super) struct Refined {
    pub(super) stencils: Stencils,
    pub(super) patches: Patches,
    pub(super) uv: Option<(Stencils, Patches)>,
}

/// The topology arrays the shim reads, kept alive while it runs.
struct Arrays {
    counts: Vec<c_int>,
    indices: Vec<c_int>,
    creases: Vec<c_int>,
    crease_sharpness: Vec<f32>,
    corners: Vec<c_int>,
    corner_sharpness: Vec<f32>,
    holes: Vec<c_int>,
    uv_indices: Vec<c_int>,
}

fn int(v: u32) -> c_int {
    v as c_int
}

impl Arrays {
    fn new(cage: &CageInput) -> Self {
        Self {
            counts: cage.face_counts.iter().copied().map(int).collect(),
            indices: cage.face_indices.iter().copied().map(int).collect(),
            creases: cage.creases.iter().flat_map(|c| c.edge.map(int)).collect(),
            crease_sharpness: cage.creases.iter().map(|c| c.sharpness).collect(),
            corners: cage.corners.iter().map(|c| int(c.point)).collect(),
            corner_sharpness: cage.corners.iter().map(|c| c.sharpness).collect(),
            holes: cage.holes.iter().copied().map(int).collect(),
            uv_indices: cage
                .uvs
                .as_ref()
                .map(|u| u.indices.iter().copied().map(int).collect())
                .unwrap_or_default(),
        }
    }

    fn cage(&self, cage: &CageInput, settings: &SubdSettings) -> Cage {
        Cage {
            vertex_count: cage.positions.len() as c_int,
            face_count: self.counts.len() as c_int,
            face_counts: self.counts.as_ptr(),
            face_indices: self.indices.as_ptr(),
            crease_count: self.crease_sharpness.len() as c_int,
            crease_pairs: self.creases.as_ptr(),
            crease_sharpness: self.crease_sharpness.as_ptr(),
            corner_count: self.corners.len() as c_int,
            corners: self.corners.as_ptr(),
            corner_sharpness: self.corner_sharpness.as_ptr(),
            hole_count: self.holes.len() as c_int,
            holes: self.holes.as_ptr(),
            left_handed: c_int::from(cage.left_handed),
            boundary: match cage.boundary {
                Boundary::None => 0,
                Boundary::Edges => 1,
                Boundary::EdgesAndCorners => 2,
            },
            uv_linear: cage.uv_linear as c_int,
            triangles: c_int::from(cage.smooth_triangles),
            uv_count: cage.uvs.as_ref().map_or(0, |u| u.values.len() as c_int),
            uv_indices: self.uv_indices.as_ptr(),
            isolation: c_int::from(settings.isolation),
            smooth_isolation: c_int::from(settings.smooth_isolation),
        }
    }
}

struct Handle(*mut Osd);

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the handle came from orr_osd_build and is freed once.
        unsafe { orr_osd_free(self.0) }
    }
}

impl Handle {
    /// Table `table` of the shim as `T`: its 4-byte ints, or floats for weight tables.
    fn table<T: bytemuck::Pod>(&self, table: c_int) -> &[T] {
        let mut data = std::ptr::null();
        // SAFETY: the shim returns a pointer to `n` 4-byte values that live as long as the
        // handle.
        let words: &[u32] = unsafe {
            let n = orr_osd_table(self.0, table, &mut data);
            if n == 0 {
                return &[];
            }
            std::slice::from_raw_parts(data.cast(), n as usize)
        };
        bytemuck::cast_slice(words)
    }

    fn stencils(&self, base: c_int) -> Stencils {
        let mut offsets = vec![0u32];
        for &s in self.table::<u32>(base) {
            offsets.push(offsets.last().unwrap() + s);
        }
        Stencils {
            offsets,
            indices: self.table::<u32>(base + 1).to_vec(),
            weights: self.table::<f32>(base + 2).to_vec(),
        }
    }

    fn patches(&self, base: c_int) -> Patches {
        let types: Vec<u8> = self.table::<i32>(base).iter().map(|&t| t as u8).collect();
        let raw = self.table::<i32>(base + 1);
        let mut first = Vec::with_capacity(types.len() + 1);
        let mut cvs = Vec::with_capacity(raw.len());
        let mut at = 0;
        for &t in &types {
            first.push(cvs.len());
            let n = match t {
                0 => 16,
                1 => 20,
                _ => 4,
            };
            let own = &raw[at..at + n];
            // Points beyond a boundary have no weight; keep their index valid.
            let fill = own.iter().copied().find(|&i| i >= 0).unwrap_or(0) as u32;
            cvs.extend(own.iter().map(|&i| if i >= 0 { i as u32 } else { fill }));
            at += n;
        }
        first.push(cvs.len());
        let params = self.table::<[u32; 2]>(base + 2).to_vec();
        Patches {
            types,
            first,
            cvs,
            params,
        }
    }
}

fn message(buf: &[c_char]) -> String {
    // SAFETY: the shim writes a nul-terminated string into the buffer.
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub(super) fn refine(cage: &CageInput, settings: &SubdSettings) -> Result<Refined, String> {
    let arrays = Arrays::new(cage);
    let desc = arrays.cage(cage, settings);
    let mut error = [0 as c_char; 512];
    // SAFETY: `desc` points into `arrays`, which outlives the call.
    let osd = unsafe { orr_osd_build(&desc, error.as_mut_ptr(), error.len() as c_int) };
    if osd.is_null() {
        return Err(format!("OpenSubdiv: {}", message(&error)));
    }
    let h = Handle(osd);
    let uv = cage
        .uvs
        .is_some()
        .then(|| (h.stencils(UV), h.patches(UV + 3)));
    Ok(Refined {
        stencils: h.stencils(STENCIL_SIZES),
        patches: h.patches(PATCH_TYPES),
        uv,
    })
}

/// Limit positions and uvs.
#[cfg(test)]
pub(super) type Samples = (Vec<[f32; 3]>, Vec<[f32; 2]>);

/// OpenSubdiv's own limit positions and uvs at ptex locations `(face, s, t)`.
#[cfg(test)]
pub(super) fn limit(
    cage: &CageInput,
    settings: &SubdSettings,
    at: &[(u32, f32, f32)],
) -> Result<Samples, String> {
    let arrays = Arrays::new(cage);
    let desc = arrays.cage(cage, settings);
    let points: Vec<[f32; 3]> = cage
        .positions
        .iter()
        .map(orr_math::Vec3::to_array)
        .collect();
    let uvs: Vec<[f32; 2]> = cage
        .uvs
        .as_ref()
        .map(|u| u.values.iter().map(orr_math::Vec2::to_array).collect())
        .unwrap_or_default();
    let ptex: Vec<c_int> = at.iter().map(|a| a.0 as c_int).collect();
    let s: Vec<f32> = at.iter().map(|a| a.1).collect();
    let t: Vec<f32> = at.iter().map(|a| a.2).collect();
    let mut out = vec![[0.0f32; 3]; at.len()];
    let mut out_uv = vec![[0.0f32; 2]; at.len()];
    let mut error = [0 as c_char; 512];
    // SAFETY: every array holds what the shim reads or writes for `at.len()` locations.
    let ok = unsafe {
        orr_osd_limit(
            &desc,
            points.as_ptr().cast(),
            if uvs.is_empty() {
                std::ptr::null()
            } else {
                uvs.as_ptr().cast()
            },
            at.len() as c_int,
            ptex.as_ptr(),
            s.as_ptr(),
            t.as_ptr(),
            out.as_mut_ptr().cast(),
            out_uv.as_mut_ptr().cast(),
            error.as_mut_ptr(),
            error.len() as c_int,
        )
    };
    if ok == 0 {
        return Err(message(&error));
    }
    Ok((out, out_uv))
}
