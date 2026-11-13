//! OpenSubdiv Far through the C shim in `osd.cpp`.

use std::ffi::{CStr, c_char, c_int, c_void};

use orr_pack::subd::PatchKind;

use super::{Boundary, CageInput, CageValues, SubdSettings};

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
    fvar_count: c_int,
    fvar_values: *const c_int,
    fvar_indices: *const c_int,
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
        fvar: *const f32,
        fvar_width: c_int,
        count: c_int,
        ptex: *const c_int,
        s: *const f32,
        t: *const f32,
        out_points: *mut f32,
        out_fvar: *mut f32,
        error: *mut c_char,
        error_len: c_int,
    ) -> c_int;
}

const STENCIL_SIZES: c_int = 0;
const PATCH_TYPES: c_int = 3;
/// A channel's stencil and patch tables, which channel `k` has from `TABLES * (k + 1)` on.
const TABLES: c_int = 6;

/// Points as stencils over the cage: point `i` is the sum of `weights[t] x cage[indices[t]]`
/// for `t` in `offsets[i]..offsets[i + 1]`. The first points are the cage's own.
pub(super) struct Stencils {
    offsets: Vec<u32>,
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

/// One patch table side (positions or a face-varying channel): per patch its type, points and
/// Far patch param.
pub(super) struct Patches {
    pub(super) types: Vec<PatchKind>,
    /// Points of patch `p` are `cvs[first[p]..first[p + 1]]`, indices into the stencils.
    pub(super) first: Vec<usize>,
    pub(super) cvs: Vec<u32>,
    /// Far's two packed `PatchParam` words.
    pub(super) params: Vec<[u32; 2]>,
}

pub(super) struct Refined {
    pub(super) stencils: Stencils,
    pub(super) patches: Patches,
    /// Those of each [`CageValues::FaceVarying`] of the cage, uvs then colors, in order.
    pub(super) fvar: Vec<(Stencils, Patches)>,
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
    /// Value count of each face-varying channel, uvs then colors, and one after another their
    /// value index per face corner.
    fvar_values: Vec<c_int>,
    fvar_indices: Vec<c_int>,
}

fn int(v: u32) -> c_int {
    v as c_int
}

/// The value count and indices of a face-varying channel.
fn channel<T>(values: &CageValues<T>) -> Option<(usize, &Vec<u32>)> {
    match values {
        CageValues::FaceVarying(f) => Some((f.values.len(), &f.indices)),
        _ => None,
    }
}

impl Arrays {
    fn new(cage: &CageInput) -> Self {
        let uvs = cage.uvs.iter().filter_map(|u| channel(&u.values));
        let colors = cage.colors.iter().filter_map(|c| channel(&c.values));
        let (mut fvar_values, mut fvar_indices) = (Vec::new(), Vec::new());
        for (values, indices) in uvs.chain(colors) {
            fvar_values.push(values as c_int);
            fvar_indices.extend(indices.iter().copied().map(int));
        }
        Self {
            counts: cage.face_counts.iter().copied().map(int).collect(),
            indices: cage.face_indices.iter().copied().map(int).collect(),
            creases: cage.creases.iter().flat_map(|c| c.edge.map(int)).collect(),
            crease_sharpness: cage.creases.iter().map(|c| c.sharpness).collect(),
            corners: cage.corners.iter().map(|c| int(c.point)).collect(),
            corner_sharpness: cage.corners.iter().map(|c| c.sharpness).collect(),
            holes: cage.holes.iter().copied().map(int).collect(),
            fvar_values,
            fvar_indices,
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
            fvar_count: self.fvar_values.len() as c_int,
            fvar_values: self.fvar_values.as_ptr(),
            fvar_indices: self.fvar_indices.as_ptr(),
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
            let n = orr_osd_table(self.0, table, &raw mut data);
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
        let types: Vec<PatchKind> = self
            .table::<i32>(base)
            .iter()
            .map(|&t| {
                u8::try_from(t)
                    .ok()
                    .and_then(|t| PatchKind::try_from(t).ok())
                    .expect("the shim stores only the patch kinds it names")
            })
            .collect();
        let raw = self.table::<i32>(base + 1);
        let mut first = Vec::with_capacity(types.len() + 1);
        let mut cvs = Vec::with_capacity(raw.len());
        let mut at = 0;
        for &t in &types {
            first.push(cvs.len());
            let n = t.cv_count();
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
    let osd = unsafe { orr_osd_build(&raw const desc, error.as_mut_ptr(), error.len() as c_int) };
    if osd.is_null() {
        return Err(format!("OpenSubdiv: {}", message(&error)));
    }
    let h = Handle(osd);
    let fvar = (1..=desc.fvar_count).map(|k| {
        let base = TABLES * k;
        (h.stencils(base), h.patches(base + PATCH_TYPES))
    });
    Ok(Refined {
        fvar: fvar.collect(),
        stencils: h.stencils(STENCIL_SIZES),
        patches: h.patches(PATCH_TYPES),
    })
}

/// Limit positions and values `N` wide.
#[cfg(test)]
type Samples<const N: usize> = (Vec<[f32; 3]>, Vec<[f32; N]>);

/// OpenSubdiv's own limit positions, and of the cage's first face-varying uv, at ptex locations
/// `(face, s, t)`.
#[cfg(test)]
pub(super) fn limit(
    cage: &CageInput,
    settings: &SubdSettings,
    at: &[(u32, f32, f32)],
) -> Result<Samples<2>, String> {
    let uvs: Vec<[f32; 2]> = cage
        .uvs
        .iter()
        .find_map(|u| match &u.values {
            CageValues::FaceVarying(f) => Some(&f.values),
            _ => None,
        })
        .map_or_default(|v| v.iter().map(orr_math::Vec2::to_array).collect());
    limit_of(cage, settings, at, &uvs)
}

/// OpenSubdiv's own limit of the cage's first face-varying color at ptex locations
/// `(face, s, t)`, of the cage without its uvs so that the color is its first channel.
#[cfg(test)]
pub(super) fn limit_color(
    cage: &CageInput,
    settings: &SubdSettings,
    at: &[(u32, f32, f32)],
) -> Result<Vec<[f32; 4]>, String> {
    let colors: Vec<[f32; 4]> = cage
        .colors
        .iter()
        .find_map(|c| match &c.values {
            CageValues::FaceVarying(f) => Some(&f.values),
            _ => None,
        })
        .ok_or("the cage has no face-varying color")?
        .iter()
        .map(orr_math::Vec4::to_array)
        .collect();
    let alone = CageInput {
        uvs: Vec::new(),
        ..cage.clone()
    };
    Ok(limit_of(&alone, settings, at, &colors)?.1)
}

/// Limit positions and values of the cage's first face-varying channel.
#[cfg(test)]
fn limit_of<const N: usize>(
    cage: &CageInput,
    settings: &SubdSettings,
    at: &[(u32, f32, f32)],
    fvar: &[[f32; N]],
) -> Result<Samples<N>, String> {
    let arrays = Arrays::new(cage);
    let desc = arrays.cage(cage, settings);
    let points: Vec<[f32; 3]> = cage
        .positions
        .iter()
        .map(orr_math::Vec3::to_array)
        .collect();
    let ptex: Vec<c_int> = at.iter().map(|a| a.0 as c_int).collect();
    let s: Vec<f32> = at.iter().map(|a| a.1).collect();
    let t: Vec<f32> = at.iter().map(|a| a.2).collect();
    let mut out = vec![[0.0f32; 3]; at.len()];
    let mut out_fvar = vec![[0.0f32; N]; at.len()];
    let mut error = [0 as c_char; 512];
    // SAFETY: every array holds what the shim reads or writes for `at.len()` locations.
    let ok = unsafe {
        orr_osd_limit(
            &raw const desc,
            points.as_ptr().cast(),
            if fvar.is_empty() {
                std::ptr::null()
            } else {
                fvar.as_ptr().cast()
            },
            N as c_int,
            at.len() as c_int,
            ptex.as_ptr(),
            s.as_ptr(),
            t.as_ptr(),
            out.as_mut_ptr().cast(),
            out_fvar.as_mut_ptr().cast(),
            error.as_mut_ptr(),
            error.len() as c_int,
        )
    };
    if ok == 0 {
        return Err(message(&error));
    }
    Ok((out, out_fvar))
}
