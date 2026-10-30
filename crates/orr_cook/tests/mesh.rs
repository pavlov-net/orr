use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glam::{Mat4, Vec3};
use orr_cook::{ArchiveSpec, AssetSpec, GltfSettings, Manifest, Source};
use orr_pack::material::Material;
use orr_pack::mesh::{self, MeshInfo, PageView, Section};
use orr_pack::model::ModelInfo;
use orr_pack::{Archive, AssetId, Class, Target};

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[derive(Default)]
struct Prim {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    tan: Vec<[f32; 4]>,
    uv: Vec<[f32; 2]>,
    idx: Vec<u32>,
}

/// Torus with a uv seam (duplicated seam vertices), radius `scale`.
fn torus(scale: f32) -> Prim {
    let (nu, nv) = (96, 48);
    let (big, small) = (scale, 0.35 * scale);
    let mut p = Prim::default();
    for j in 0..=nv {
        for i in 0..=nu {
            let (u, v) = (i as f32 / nu as f32, j as f32 / nv as f32);
            let (a, b) = (u * std::f32::consts::TAU, v * std::f32::consts::TAU);
            let n = [a.cos() * b.cos(), b.sin(), a.sin() * b.cos()];
            p.pos.push([
                a.cos() * big + small * n[0],
                small * n[1],
                a.sin() * big + small * n[2],
            ]);
            p.nrm.push(n);
            p.tan
                .push([-a.sin(), 0.0, a.cos(), if i % 2 == 0 { 1.0 } else { -1.0 }]);
            p.uv.push([u * 3.0, v]);
        }
    }
    let w = nu + 1;
    for j in 0..nv {
        for i in 0..nu {
            let c = j * w + i;
            p.idx.extend([c, c + w, c + 1, c + 1, c + w, c + w + 1]);
        }
    }
    p
}

/// A wavy grid of quads above the torus.
fn sheet(scale: f32) -> Prim {
    let n = 40;
    let mut p = Prim::default();
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            p.pos.push([
                (u - 0.5) * 2.0 * scale,
                0.6 * scale + 0.05 * scale * (u * 9.0).sin(),
                (v - 0.5) * 2.0 * scale,
            ]);
            p.nrm.push([0.0, 1.0, 0.0]);
            p.tan.push([1.0, 0.0, 0.0, 1.0]);
            p.uv.push([u, 1.0 - v]);
        }
    }
    for j in 0..n {
        for i in 0..n {
            let c = j * (n + 1) + i;
            p.idx
                .extend([c, c + n + 1, c + 1, c + 1, c + n + 1, c + n + 2]);
        }
    }
    p
}

/// A GLB with one node at `translation` holding the two primitives.
fn write_glb(path: &Path, prims: &[&Prim], translation: [f32; 3]) {
    let mut bin = Vec::new();
    let mut views = Vec::new();
    let mut accessors = Vec::new();
    let mut add = |bytes: &[u8], count: usize, ty: &str, comp: u32, minmax: Option<String>| {
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            bin.len(),
            bytes.len()
        ));
        bin.extend_from_slice(bytes);
        bin.resize(bin.len().next_multiple_of(4), 0);
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{comp},"count":{count},"type":"{ty}"{}}}"#,
            views.len() - 1,
            minmax.unwrap_or_default()
        ));
        accessors.len() - 1
    };
    let mut prims_json = Vec::new();
    for (m, p) in prims.iter().enumerate() {
        let lo = [0, 1, 2].map(|a| p.pos.iter().map(|v| v[a]).fold(f32::MAX, f32::min));
        let hi = [0, 1, 2].map(|a| p.pos.iter().map(|v| v[a]).fold(f32::MIN, f32::max));
        let n = p.pos.len();
        let pos = add(
            bytemuck::cast_slice(&p.pos),
            n,
            "VEC3",
            5126,
            Some(format!(r#","min":{lo:?},"max":{hi:?}"#)),
        );
        let nrm = add(bytemuck::cast_slice(&p.nrm), n, "VEC3", 5126, None);
        let tan = add(bytemuck::cast_slice(&p.tan), n, "VEC4", 5126, None);
        let uv = add(bytemuck::cast_slice(&p.uv), n, "VEC2", 5126, None);
        let idx = add(
            bytemuck::cast_slice(&p.idx),
            p.idx.len(),
            "SCALAR",
            5125,
            None,
        );
        prims_json.push(format!(
            r#"{{"attributes":{{"POSITION":{pos},"NORMAL":{nrm},"TANGENT":{tan},"TEXCOORD_0":{uv}}},"indices":{idx},"material":{m}}}"#
        ));
    }
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"extensionsUsed":["KHR_materials_emissive_strength"],
"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0,"translation":{translation:?}}}],
"meshes":[{{"primitives":[{}]}}],
"materials":[
 {{"pbrMetallicRoughness":{{"baseColorFactor":[0.8,0.2,0.2,1.0],"metallicFactor":0.25,"roughnessFactor":0.6}},
   "emissiveFactor":[1.0,0.5,0.0],"extensions":{{"KHR_materials_emissive_strength":{{"emissiveStrength":4.0}}}}}},
 {{"pbrMetallicRoughness":{{"baseColorFactor":[0.2,0.6,0.1,0.8]}},"alphaMode":"MASK","alphaCutoff":0.3,"doubleSided":true}}],
"buffers":[{{"byteLength":{}}}],"bufferViews":[{}],"accessors":[{}]}}"#,
        prims_json.join(","),
        bin.len(),
        views.join(","),
        accessors.join(",")
    );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);
    std::fs::write(path, glb).unwrap();
}

fn cook_memory(path: &Path) -> orr_cook::Result<Archive> {
    let manifest = Manifest {
        archives: vec![ArchiveSpec {
            output: Target::Memory,
            cell_log2: ArchiveSpec::DEFAULT_CELL_LOG2,
            assets: vec![AssetSpec {
                name: "fixture".into(),
                source: Source::Gltf {
                    path: path.into(),
                    settings: GltfSettings::default(),
                },
            }],
        }],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())?
        .memory
        .remove(0);
    Ok(Archive::from_memory(bytes).unwrap())
}

fn read_blob(a: &Archive, b: &orr_pack::Blob) -> Vec<u8> {
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

type TriKey = (u8, [[u32; 3]; 3]);

/// A triangle keyed by its corner positions' bits, rotated so the smallest corner comes
/// first (winding kept), and the rotation applied.
fn key(section: u8, p: [[f32; 3]; 3]) -> (TriKey, usize) {
    let bits = p.map(|v| v.map(f32::to_bits));
    let r = (0..3).min_by_key(|&i| bits[i]).unwrap();
    (
        (section, [bits[r], bits[(r + 1) % 3], bits[(r + 2) % 3]]),
        r,
    )
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    Vec3::from(a).dot(Vec3::from(b))
}

/// Cooks the fixture at a scale and checks the grid, clusters, attributes and materials
/// against values computed here from the source.
fn check_fixture(scale: f32, translation: [f32; 3]) {
    let d = dir(&format!("mesh-{scale}"));
    let path = d.join("fixture.glb");
    let prims = [torus(scale), sheet(scale)];
    write_glb(&path, &[&prims[0], &prims[1]], translation);
    let a = cook_memory(&path).unwrap();
    for e in a.entries() {
        a.verify(e).unwrap();
    }

    let model = a
        .expect(
            AssetId::from_name("fixture"),
            Class::Model as u16,
            orr_pack::model::SCHEMA,
        )
        .unwrap();
    let model = ModelInfo::parse(&read_blob(&a, &a.blobs(model)[0])).unwrap();
    assert_eq!(model.materials.len(), 2);
    let mats: Vec<Material> = model
        .materials
        .iter()
        .map(|&id| {
            let e = a
                .expect(id, Class::Material as u16, orr_pack::material::SCHEMA)
                .unwrap();
            Material::from_bytes(&read_blob(&a, &a.blobs(e)[0])).unwrap()
        })
        .collect();
    assert_eq!(mats[0].base_color, [0.8, 0.2, 0.2]);
    assert_eq!((mats[0].metalness, mats[0].specular_roughness), (0.25, 0.6));
    assert_eq!(
        (mats[0].emission_color, mats[0].emission_luminance),
        ([1.0, 0.5, 0.0], 4.0)
    );
    assert_eq!(
        (
            mats[1].alpha_mode,
            mats[1].alpha_cutoff,
            mats[1].double_sided
        ),
        (Material::MASK, 0.3, 1)
    );
    assert_eq!(mats[1].geometry_opacity, 0.8);

    let me = a
        .expect(model.header.mesh, Class::Mesh as u16, mesh::SCHEMA)
        .unwrap();
    let blobs = a.blobs(me);
    let info = MeshInfo::parse(&read_blob(&a, &blobs[0])).unwrap();
    let h = info.header;
    let flags: Vec<u8> = info.sections.iter().map(|s| s.flags).collect();
    assert_eq!(
        flags,
        [
            Section::EMISSIVE,
            Section::ALPHA_TESTED | Section::DOUBLE_SIDED
        ]
    );

    // The grid, from the plan's formula over the transformed source positions.
    let world = Mat4::from_translation(Vec3::from(translation));
    let src: Vec<Vec<[f32; 3]>> = prims
        .iter()
        .map(|p| {
            p.pos
                .iter()
                .map(|&v| world.transform_point3(Vec3::from(v)).to_array())
                .collect()
        })
        .collect();
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for v in src.iter().flatten() {
        lo = [0, 1, 2].map(|a| lo[a].min(v[a]));
        hi = [0, 1, 2].map(|a| hi[a].max(v[a]));
    }
    let extent = (0..3).map(|a| f64::from(hi[a] - lo[a])).fold(0.0, f64::max);
    let k = (-11).max((extent / 65535.0).log2().ceil() as i32);
    assert_eq!(i32::from(h.position_exponent), k);
    let step = 2f64.powi(k);
    let cell = |x: f32| (f64::from(x) / step + 0.5).floor() as i64;
    let snap = |v: [f32; 3]| v.map(|x| (cell(x) as f64 * step) as f32);
    let max_cell = src
        .iter()
        .flatten()
        .flatten()
        .map(|&x| cell(x).abs())
        .max()
        .unwrap();
    assert_eq!(
        u32::from(h.clas_truncate_bits),
        24 - (64 - max_cell.leading_zeros())
    );

    // Source triangles on the grid, with their corner vertices.
    let mut expected: HashMap<TriKey, (usize, [u32; 3])> = HashMap::new();
    let mut area = [0f64; 2];
    for (s, p) in prims.iter().enumerate() {
        for t in p.idx.as_chunks::<3>().0 {
            let q = t.map(|i| snap(src[s][i as usize]));
            let (k, r) = key(s as u8, q);
            let rotated = [t[r], t[(r + 1) % 3], t[(r + 2) % 3]];
            assert!(expected.insert(k, (s, rotated)).is_none());
            let q = q.map(|v| Vec3::from(v).as_dvec3());
            area[s] += 0.5 * (q[1] - q[0]).cross(q[2] - q[0]).length();
        }
    }

    let mut cluster_area = [0f64; 2];
    let mut clusters = 0;
    let truncate = u32::from(h.clas_truncate_bits);
    for (pi, page) in info.pages.iter().enumerate() {
        let b = &blobs[page.blob as usize];
        assert_eq!(b.is_mount(), (pi as u32) < h.pinned_pages);
        let stored = read_blob(&a, b);
        let mut resident = vec![0; page.resident_len as usize];
        let layout = mesh::decode_page(&stored, &mut resident).unwrap();
        assert_eq!(layout.len, page.resident_len);
        let view = PageView::new(&resident, &h).unwrap();
        assert_eq!(view.cluster_count(), u32::from(page.cluster_count));
        for ci in 0..view.cluster_count() {
            clusters += 1;
            let c = view.cluster(ci);
            let s = usize::from(c.section);
            let foliage = info.sections[s].flags & Section::ALPHA_TESTED != 0;
            assert!(c.tri_count <= 128);
            assert!(c.vertex_count <= if foliage { 256 } else { 128 });
            // The sheet's constant opacity passes its cutoff, so every triangle is opaque.
            assert_eq!(c.alpha_first, c.tri_count);
            assert!(c.anchor.iter().all(|&v| v.abs() < 1 << 23));
            let mut local_max = [0; 3];
            for v in 0..u32::from(c.vertex_count) {
                let g = view.grid(&c, v);
                assert!(g.iter().all(|&x| x.abs() < 1 << 23));
                local_max = [0, 1, 2].map(|a| local_max[a].max(g[a] - c.anchor[a]));
                for x in view.position(&c, v) {
                    let bits = x.to_bits();
                    assert_eq!(bits >> truncate << truncate, bits);
                }
            }
            assert_eq!(local_max, c.local_max.map(i32::from));
            cluster_area[s] += f64::from(c.area);

            for t in 0..u32::from(c.tri_count) {
                let corners = view.triangle(&c, t).map(u32::from);
                let pos = corners.map(|v| view.position(&c, v));
                let (k, r) = key(c.section, pos);
                let (sec, verts) = expected
                    .remove(&k)
                    .expect("cooked triangle is a source triangle on the grid");
                let corners = [corners[r], corners[(r + 1) % 3], corners[(r + 2) % 3]];
                for (&v, &src_v) in corners.iter().zip(&verts) {
                    let (p, sv) = (&prims[sec], src_v as usize);
                    assert!(dot(view.normal(&c, v), p.nrm[sv]) > 0.99999);
                    let tan = view.tangent(&c, v).unwrap();
                    let st = p.tan[sv];
                    assert!(dot([tan[0], tan[1], tan[2]], [st[0], st[1], st[2]]) > 0.99999);
                    assert_eq!(tan[3], st[3]);
                    let uv = view.uv(&c, v);
                    for ((got, want), scale) in uv.iter().zip(p.uv[sv]).zip(c.uv_scale) {
                        assert!((got - want).abs() <= scale / 65535.0 * 0.5 + 1e-6);
                    }
                }
            }
        }
    }
    assert!(
        expected.is_empty(),
        "{} source triangles missing",
        expected.len()
    );
    assert_eq!(info.levels[0].clusters.count, clusters);
    for s in 0..2 {
        assert!((cluster_area[s] - area[s]).abs() <= 1e-5 * area[s]);
    }
}

#[test]
fn fixture_on_the_finest_grid() {
    check_fixture(1.0, [1000.25, -3.0, 40.0]);
}

#[test]
fn fixture_on_a_coarser_grid() {
    check_fixture(150.0, [-2000.0, 10.0, 0.5]);
}

#[test]
fn positions_past_24_bits_fail() {
    let d = dir("mesh-far");
    let path = d.join("far.glb");
    let prims = [torus(1.0), sheet(1.0)];
    // 2^23 steps of 2^-11 m is 4096 m.
    write_glb(&path, &[&prims[0], &prims[1]], [5000.0, 0.0, 0.0]);
    let err = cook_memory(&path).err().unwrap();
    assert!(matches!(err, orr_cook::Error::Mesh { .. }), "{err}");
}

#[test]
fn corrupt_pages_are_rejected() {
    let d = dir("mesh-corrupt");
    let path = d.join("fixture.glb");
    let prims = [torus(1.0), sheet(1.0)];
    write_glb(&path, &[&prims[0], &prims[1]], [0.0; 3]);
    let a = cook_memory(&path).unwrap();
    let me = a.find(AssetId::from_name("fixture/mesh")).unwrap();
    let info = MeshInfo::parse(&read_blob(&a, &a.blobs(me)[0])).unwrap();
    let page = read_blob(&a, &a.blobs(me)[1]);
    let mut dst = vec![0; info.pages[0].resident_len as usize];
    mesh::decode_page(&page, &mut dst).unwrap();

    let short = &page[..page.len() - 4];
    assert!(mesh::decode_page(short, &mut dst).is_err());

    // Cluster 0's first vertex moved past the vertex arrays.
    let mut bad = page.clone();
    let at = size_of::<mesh::PageHeader>() + std::mem::offset_of!(mesh::Cluster, vertex_first);
    bad[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(mesh::decode_page(&bad, &mut dst).is_err());

    // Cluster 0 claims more vertices than its stream holds.
    let mut bad = page.clone();
    let at = size_of::<mesh::PageHeader>() + std::mem::offset_of!(mesh::Cluster, vertex_count);
    bad[at..at + 2].copy_from_slice(&300u16.to_le_bytes());
    assert!(mesh::decode_page(&bad, &mut dst).is_err());
}
