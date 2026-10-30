//! glTF 2.0 import: the default scene's triangle primitives as one mesh, and the materials
//! they use.
//!
//! Material mapping (glTF -> OpenPBR):
//! - `baseColorFactor` rgb -> `base_color`; `metallicFactor` -> `metalness`;
//!   `roughnessFactor` -> `specular_roughness`; `KHR_materials_ior` -> `specular_ior`.
//! - `emissiveFactor` -> `emission_color`; `KHR_materials_emissive_strength` (1 without it)
//!   -> `emission_luminance`, read as nits.
//! - `alphaMode`: `OPAQUE`; `MASK` with `alphaCutoff`; `BLEND`. Base alpha -> `geometry_opacity`
//!   under `MASK` and `BLEND`.
//! - `doubleSided` -> `double_sided`.
//! - Texture references are not bound.

use std::collections::BTreeMap;
use std::path::Path;

use base64::Engine;
use glam::{Mat3, Mat4, Vec3};
use gltf::buffer::Source as BufferSource;
use gltf::material::AlphaMode;
use gltf::mesh::Mode;
use orr_pack::Hash32;
use orr_pack::material::Material;
use orr_pack::mesh::Section;

use crate::mesh::{MeshData, SectionData};
use crate::{Error, Result};

pub(crate) struct Imported {
    pub mesh: MeshData,
    /// Name suffix and material per mesh section.
    pub materials: Vec<(String, Material)>,
    /// blake3 of the document and every buffer it reads.
    pub input_hash: Hash32,
}

#[derive(Default)]
struct Vertices {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
}

pub(crate) fn import(asset: &str, path: &Path) -> Result<Imported> {
    let fail = |message: String| Error::Import {
        asset: asset.into(),
        message,
    };
    let read = |p: &Path| {
        std::fs::read(p).map_err(|source| Error::Read {
            path: p.into(),
            source,
        })
    };
    let bytes = read(path)?;
    let doc =
        gltf::Gltf::from_slice(&bytes).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    let mut hash = blake3::Hasher::new();
    hash.update(&bytes);

    let mut buffers = Vec::new();
    for b in doc.buffers() {
        let data = match b.source() {
            BufferSource::Bin => doc
                .blob
                .clone()
                .ok_or_else(|| fail("GLB binary chunk missing".into()))?,
            BufferSource::Uri(uri) => match uri.strip_prefix("data:") {
                Some(data) => {
                    let (_, b64) = data.split_once(";base64,").ok_or_else(|| {
                        fail(format!("buffer {}: data URI is not base64", b.index()))
                    })?;
                    base64::engine::general_purpose::STANDARD
                        .decode(b64)
                        .map_err(|e| fail(format!("buffer {}: {e}", b.index())))?
                }
                None => {
                    let data = read(&path.parent().unwrap_or(Path::new("")).join(uri))?;
                    hash.update(&data);
                    data
                }
            },
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

    let scene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .ok_or_else(|| fail("no scene".into()))?;
    let mut v = Vertices::default();
    let mut all_tangents = true;
    // Material and triangles per glTF material index; usize::MAX is the default material.
    let mut groups: BTreeMap<usize, (Material, Vec<u32>)> = BTreeMap::new();
    let mut stack: Vec<_> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        stack.extend(node.children().map(|c| (c, world)));
        let Some(mesh) = node.mesh() else { continue };
        let linear = Mat3::from_mat4(world);
        let flip = linear.determinant() < 0.0;
        let normal_m = linear.inverse().transpose();
        for prim in mesh.primitives() {
            let what = || format!("mesh {} primitive {}", mesh.index(), prim.index());
            if prim.mode() != Mode::Triangles {
                return Err(fail(format!(
                    "{}: mode {:?} is not triangles",
                    what(),
                    prim.mode()
                )));
            }
            let r = prim.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
            let positions: Vec<[f32; 3]> = r
                .read_positions()
                .ok_or_else(|| fail(format!("{}: no positions", what())))?
                .map(|p| world.transform_point3(Vec3::from(p)).to_array())
                .collect();
            let n = positions.len();
            let mut indices: Vec<u32> = match r.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..n as u32).collect(),
            };
            if !indices.len().is_multiple_of(3) || indices.iter().any(|&i| i as usize >= n) {
                return Err(fail(format!("{}: bad index list", what())));
            }
            if flip {
                indices.chunks_exact_mut(3).for_each(|t| t.swap(1, 2));
            }
            let uvs: Vec<[f32; 2]> = match r.read_tex_coords(0) {
                Some(t) => t.into_f32().collect(),
                None => vec![[0.0; 2]; n],
            };
            let tangents: Option<Vec<[f32; 4]>> = r.read_tangents().map(|t| {
                t.map(|[x, y, z, w]| {
                    let t = (linear * Vec3::new(x, y, z)).to_array();
                    [t[0], t[1], t[2], if flip { -w } else { w }]
                })
                .collect()
            });
            all_tangents &= tangents.is_some();
            let tangents = tangents.unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; n]);
            let normals: Option<Vec<[f32; 3]>> = r.read_normals().map(|ns| {
                ns.map(|nv| (normal_m * Vec3::from(nv)).to_array())
                    .collect()
            });
            if uvs.len() != n
                || tangents.len() != n
                || normals.as_ref().is_some_and(|x| x.len() != n)
            {
                return Err(fail(format!("{}: attribute counts differ", what())));
            }

            let base = v.positions.len() as u32;
            let mat = prim.material();
            let (_, tris) = groups
                .entry(mat.index().unwrap_or(usize::MAX))
                .or_insert_with(|| (material(&mat), Vec::new()));
            match normals {
                Some(normals) => {
                    v.positions.extend(positions);
                    v.normals.extend(normals);
                    v.tangents.extend(tangents);
                    v.uvs.extend(uvs);
                    tris.extend(indices.iter().map(|i| base + i));
                }
                // Without normals, glTF asks for flat shading: one vertex per corner.
                None => {
                    for t in indices.as_chunks::<3>().0 {
                        let p = t.map(|i| Vec3::from(positions[i as usize]));
                        let face = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
                        for &i in t {
                            tris.push(v.positions.len() as u32);
                            v.positions.push(positions[i as usize]);
                            v.normals.push(face.to_array());
                            v.tangents.push(tangents[i as usize]);
                            v.uvs.push(uvs[i as usize]);
                        }
                    }
                }
            }
        }
    }
    if groups.values().all(|(_, t)| t.is_empty()) {
        return Err(fail("the scene has no triangles".into()));
    }

    let mut indices = Vec::new();
    let mut sections = Vec::new();
    let mut materials = Vec::new();
    for (index, (m, tris)) in groups {
        if tris.is_empty() {
            continue;
        }
        let suffix = match index {
            usize::MAX => "default".to_string(),
            i => i.to_string(),
        };
        let flags = if m.alpha_mode == Material::MASK {
            Section::ALPHA_TESTED
        } else {
            0
        } | if m.double_sided != 0 {
            Section::DOUBLE_SIDED
        } else {
            0
        } | if m.emits() { Section::EMISSIVE } else { 0 };
        sections.push(SectionData {
            indices: indices.len()..indices.len() + tris.len(),
            flags,
        });
        indices.extend(tris);
        materials.push((suffix, m));
    }

    Ok(Imported {
        mesh: MeshData {
            positions: v.positions,
            normals: v.normals,
            tangents: all_tangents.then_some(v.tangents),
            uvs: v.uvs,
            indices,
            sections,
        },
        materials,
        input_hash: Hash32(*hash.finalize().as_bytes()),
    })
}

fn material(m: &gltf::Material<'_>) -> Material {
    let pbr = m.pbr_metallic_roughness();
    let [r, g, b, a] = pbr.base_color_factor();
    let emissive = m.emissive_factor();
    let (alpha_mode, opacity) = match m.alpha_mode() {
        AlphaMode::Opaque => (Material::OPAQUE, 1.0),
        AlphaMode::Mask => (Material::MASK, a),
        AlphaMode::Blend => (Material::BLEND, a),
    };
    Material {
        base_color: [r, g, b],
        metalness: pbr.metallic_factor(),
        specular_roughness: pbr.roughness_factor(),
        specular_ior: m.ior().unwrap_or(1.5),
        emission_color: emissive,
        emission_luminance: if emissive.iter().any(|&c| c > 0.0) {
            m.emissive_strength().unwrap_or(1.0)
        } else {
            0.0
        },
        geometry_opacity: opacity,
        alpha_mode,
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        double_sided: m.double_sided().into(),
        ..Material::default()
    }
}
