#[path = "common/tangent_quads.rs"]
mod tangent_quads;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, Manifest, MeshInput, MeshSettings, Source,
    TextureSettings, Usage,
};
use orr_math::{Mat4, Vec2, Vec3};
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{self, MeshInfo, PageView, Section};
use orr_pack::model::ModelInfo;
use orr_pack::{Archive, AssetId, Class, Span, Target};

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
            gdeflate: false,
            assets: vec![AssetSpec {
                name: "fixture".into(),
                source: Source::Gltf {
                    path: path.into(),
                    // The fixture checks how stored tangents are encoded.
                    settings: GltfSettings {
                        mesh: MeshSettings {
                            tangent_error_max: None,
                            ..MeshSettings::default()
                        },
                        ..GltfSettings::default()
                    },
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

type TriKey = (u16, [[u32; 3]; 3]);

/// A triangle keyed by its corner positions' bits, rotated so the smallest corner comes
/// first (winding kept), and the rotation applied.
fn key(section: u16, p: [[f32; 3]; 3]) -> (TriKey, usize) {
    let bits = p.map(|v| v.map(f32::to_bits));
    let r = (0..3).min_by_key(|&i| bits[i]).unwrap();
    (
        (section, [bits[r], bits[(r + 1) % 3], bits[(r + 2) % 3]]),
        r,
    )
}

/// Cooks the fixture at a scale and checks the grid, which is `2^k` m, clusters, attributes
/// and materials against values computed here from the source; returns how many clusters
/// span more than 16 bits.
fn check_fixture(scale: f32, translation: [f32; 3], k: i32) -> usize {
    let d = dir(&format!("mesh-{scale}-{}", translation[0]));
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
            Section::EMISSIVE | Section::CLOSED,
            Section::ALPHA_TESTED | Section::DOUBLE_SIDED
        ]
    );

    // The grid: the finest step on which every transformed source position's cell is
    // within 2^23 of the origin.
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
    let cell = |x: f32, k: i32| (f64::from(x) / 2f64.powi(k) + 0.5).floor() as i64;
    let max_cell = |k: i32| {
        src.iter()
            .flatten()
            .flatten()
            .map(|&x| cell(x, k).abs())
            .max()
            .unwrap()
    };
    assert_eq!((-11..).find(|&k| max_cell(k) < 1 << 23), Some(k));
    assert_eq!(i32::from(h.position_exponent), k);
    let step = 2f64.powi(k);
    let snap = |v: [f32; 3]| v.map(|x| (cell(x, k) as f64 * step) as f32);
    let max_cell = max_cell(k);
    assert_eq!(
        u32::from(h.clas_truncate_bits),
        24 - (64 - max_cell.leading_zeros())
    );
    // An edit without bounds of its own keeps the bits of the model's box.
    assert_eq!(
        h.clas_truncate_bits_within(&model.header.bounds),
        h.clas_truncate_bits
    );

    // Source triangles on the grid, with their corner vertices.
    let mut expected: HashMap<TriKey, (usize, [u32; 3])> = HashMap::new();
    let mut area = [0f64; 2];
    for (s, p) in prims.iter().enumerate() {
        for t in p.idx.as_chunks::<3>().0 {
            let q = t.map(|i| snap(src[s][i as usize]));
            let (k, r) = key(s as u16, q);
            let rotated = [t[r], t[(r + 1) % 3], t[(r + 2) % 3]];
            assert!(expected.insert(k, (s, rotated)).is_none());
            let q = q.map(|v| Vec3::from(v).as_dvec3());
            area[s] += 0.5 * (q[1] - q[0]).cross(q[2] - q[0]).length();
        }
    }

    let mut cluster_area = [0f64; 2];
    let (mut clusters, mut wide) = (0, 0);
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
        let tris = (0..view.cluster_count()).map(|ci| u32::from(view.cluster(ci).tri_count));
        assert_eq!(tris.sum::<u32>(), page.tri_count, "page {pi}");
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
            assert_eq!(local_max, c.local_max.map(|m| m as i32));
            wide += usize::from(c.wide());
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
                    assert!(Vec3::from(view.normal(&c, v)).dot(Vec3::from(p.nrm[sv])) > 0.99999);
                    let tan = view.tangent(&c, v).unwrap();
                    let st = p.tan[sv];
                    assert!(Vec3::from_slice(&tan).dot(Vec3::from_slice(&st)) > 0.99999);
                    assert_eq!(tan[3], st[3]);
                    let uv = view.uv(&c, v, 0);
                    for ((got, want), step) in uv.iter().zip(p.uv[sv]).zip(c.uv_step[0]) {
                        assert!((got - want).abs() <= 0.5 * step);
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
    wide
}

#[test]
fn fixture_on_the_finest_grid() {
    assert_eq!(check_fixture(1.0, [1000.25, -3.0, 40.0], -11), 0);
}

/// A 400 m fixture keeps the finest step, its clusters spanning more than 16 bits of it.
#[test]
fn large_fixture_on_the_finest_grid() {
    assert!(check_fixture(150.0, [-2000.0, 10.0, 0.5], -11) > 0);
}

/// 2^23 steps of 2^-11 m is 4096 m: past it the step coarsens to keep every position in 24
/// bits.
#[test]
fn fixture_past_24_bits_on_a_coarser_grid() {
    assert_eq!(check_fixture(1.0, [5000.0, 0.0, 0.0], -10), 0);
}

/// A strip of `n` columns of quads from `x0` to `x1`, its rows from z = 0.03 to 5.03 rising
/// in y with z alone, so two strips that meet in x share their edge's points.
fn strip(x0: f32, x1: f32, n: u32) -> MeshInput {
    let mut m = MeshInput::default();
    let mut uvs = Vec::new();
    for j in 0..=20 {
        for i in 0..=n {
            let x = if i == n {
                x1
            } else {
                x0 + (x1 - x0) * i as f32 / n as f32
            };
            let z = 0.03 + 0.25 * j as f32;
            m.positions.push(Vec3::new(x, 0.017 * z, z));
            m.normals.push(Vec3::Y);
            uvs.push(Vec2::new(i as f32 / n as f32, j as f32 / 20.0));
        }
    }
    m.uvs = Some(uvs);
    let w = n + 1;
    for j in 0..20 {
        for i in 0..n {
            let c = j * w + i;
            m.indices.extend([c, c + w, c + 1, c + 1, c + w, c + w + 1]);
        }
    }
    m.sections = vec![Span {
        first: 0,
        count: m.indices.len() as u32,
    }];
    m
}

/// A 1 km strip and a 10 m one that meet at x = 0.1 m, cooked as they are: the points of
/// their shared edge read back bit for bit alike from both meshes.
#[test]
fn meshes_of_any_extent_store_a_shared_edge_alike() {
    let meshes = [
        ("large", strip(0.1 - 1000.0, 0.1, 20)),
        ("small", strip(0.1, 10.1, 20)),
    ];
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            meshes
                .into_iter()
                .map(|(name, mesh)| AssetSpec {
                    name: name.into(),
                    source: Source::Mesh {
                        mesh,
                        materials: vec![Material::default()],
                        settings: MeshSettings::default(),
                    },
                })
                .collect(),
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let a = Archive::from_memory(bytes).unwrap();
    // The points of a mesh at its least (`side` -1) or greatest (1) x, as bits.
    let edge = |name: &str, side: f32| {
        let e = a.find(AssetId::from_name(&format!("{name}/mesh"))).unwrap();
        let blobs = a.blobs(e);
        let info = MeshInfo::parse(&read_blob(&a, &blobs[0])).unwrap();
        let mut points = Vec::new();
        for page in &info.pages {
            let mut resident = vec![0; page.resident_len as usize];
            mesh::decode_page(&read_blob(&a, &blobs[page.blob as usize]), &mut resident).unwrap();
            let view = PageView::new(&resident, &info.header).unwrap();
            for i in 0..view.cluster_count() {
                let c = view.cluster(i);
                points.extend((0..u32::from(c.vertex_count)).map(|v| view.position(&c, v)));
            }
        }
        let x = points.iter().map(|p| side * p[0]).fold(f32::MIN, f32::max);
        let mut edge: Vec<[u32; 3]> = points
            .iter()
            .filter(|p| side * p[0] == x)
            .map(|p| p.map(f32::to_bits))
            .collect();
        edge.sort_unstable();
        edge.dedup();
        edge
    };
    let large = edge("large", 1.0);
    assert_eq!(large.len(), 21);
    assert_eq!(large, edge("small", -1.0));
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

/// The normal texture a library mesh's materials bind, and the model `input` cooked keeping
/// its tangents with `materials`, and the cook's warnings.
fn library_cook(name: &str, input: MeshInput, materials: Vec<Material>) -> (Archive, Vec<String>) {
    let normals = image::RgbaImage::from_fn(16, 16, |x, y| {
        image::Rgba([96 + (x * 4) as u8, 160 - (y * 4) as u8, 255, 255])
    });
    let settings = MeshSettings {
        tangent_error_max: None,
        ..MeshSettings::default()
    };
    let (a, report) = library_cook_with(name, input, materials, settings, normals);
    let warnings = report.assets.iter().flat_map(|a| &a.entries);
    let warnings = warnings.flat_map(|e| e.warnings.clone()).collect();
    (a, warnings)
}

/// `input` cooked with `materials` and `settings` beside the normal texture `normals` they bind,
/// and the cook's report.
fn library_cook_with(
    name: &str,
    input: MeshInput,
    materials: Vec<Material>,
    settings: MeshSettings,
    normals: image::RgbaImage,
) -> (Archive, orr_cook::CookReport) {
    let path = dir(name).join("normal.png");
    normals.save(&path).unwrap();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: "tex".into(),
                    source: Source::Texture {
                        path,
                        settings: TextureSettings {
                            usage: Usage::Normal,
                            ..TextureSettings::default()
                        },
                    },
                },
                AssetSpec {
                    name: "m".into(),
                    source: Source::Mesh {
                        mesh: input,
                        materials,
                        settings,
                    },
                },
            ],
        )],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
    let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
    (a, cooked.report)
}

/// Section, position and tangent of every vertex of the library model.
fn library_tangents(a: &Archive) -> Vec<(u16, Vec3, [f32; 4])> {
    let e = a.find(AssetId::from_name("m/mesh")).unwrap();
    let blobs = a.blobs(e);
    let read = |b: &orr_pack::Blob| {
        let mut v = vec![0; b.raw_len as usize];
        a.read(b, &mut v).unwrap();
        v
    };
    let info = MeshInfo::parse(&read(&blobs[0])).unwrap();
    let mut out = Vec::new();
    for page in &info.pages {
        let mut resident = vec![0; page.resident_len as usize];
        mesh::decode_page(&read(&blobs[page.blob as usize]), &mut resident).unwrap();
        let view = PageView::new(&resident, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            for v in 0..u32::from(c.vertex_count) {
                let t = view.tangent(&c, v).unwrap();
                out.push((c.section, Vec3::from(view.position(&c, v)), t));
            }
        }
    }
    out
}

fn normal_mapped(uv_set: u8) -> Material {
    let mut m = Material::default();
    m.textures[slot::NORMAL].texture = AssetId::from_name("tex");
    m.textures[slot::NORMAL].uv_set = uv_set;
    m
}

/// A library mesh without tangents gets MikkTSpace's where its material has a normal map, and
/// `NormalTangentMirrorTest`'s stored tangents are what they come to.
#[test]
fn library_meshes_without_tangents_compute_the_published_ones() {
    let (p, n, uv, t, idx) = tangent_quads::arrays();
    let input = MeshInput {
        positions: p.iter().map(|&p| p.into()).collect(),
        normals: n.iter().map(|&n| n.into()).collect(),
        uvs: Some(uv.iter().map(|&u| u.into()).collect()),
        sections: vec![Span {
            first: 0,
            count: idx.len() as u32,
        }],
        indices: idx,
        ..MeshInput::default()
    };
    let (a, _) = library_cook("library-tangents", input, vec![normal_mapped(0)]);
    let got = library_tangents(&a);
    assert_eq!(got.len(), p.len());
    for (_, at, tangent) in got {
        let k = p
            .iter()
            .position(|&q| Vec3::from(q).abs_diff_eq(at, 1e-3))
            .unwrap();
        let want = Vec3::from_slice(&t[k]);
        assert!(Vec3::from_slice(&tangent).dot(want) > 0.9999 && tangent[3] == t[k][3]);
    }
}

/// Maps of mesh space directions read no tangent frame, and neither map's uv set takes one.
#[test]
fn object_space_normal_maps_read_no_tangents() {
    let (p, n, uv, _, idx) = tangent_quads::arrays();
    let input = MeshInput {
        positions: p.iter().map(|&p| p.into()).collect(),
        normals: n.iter().map(|&n| n.into()).collect(),
        uvs: Some(uv.iter().map(|&u| u.into()).collect()),
        sections: vec![Span {
            first: 0,
            count: idx.len() as u32,
        }],
        indices: idx,
        ..MeshInput::default()
    };
    let mut m = normal_mapped(0);
    m.textures[slot::COAT_NORMAL].texture = AssetId::from_name("tex");
    m.object_normals = Material::OBJECT_NORMAL | Material::OBJECT_COAT_NORMAL;
    let (a, _) = library_cook("library-object-normals", input, vec![m]);
    let blob = a.blobs(a.find(AssetId::from_name("m/mesh")).unwrap())[0];
    let mut head = vec![0; blob.raw_len as usize];
    a.read(&blob, &mut head).unwrap();
    let info = MeshInfo::parse(&head).unwrap();
    assert_eq!(info.header.attributes & mesh::MeshHeader::TANGENT, 0);
}

/// A plane with u along +x and v down y, and its second uv set turned to run along y.
fn turned_sets() -> MeshInput {
    let z = Vec3::Z;
    MeshInput {
        positions: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        normals: vec![z; 4],
        uvs: Some(vec![
            [0.0, 1.0].into(),
            [1.0, 1.0].into(),
            [1.0, 0.0].into(),
            [0.0, 0.0].into(),
        ]),
        indices: vec![0, 1, 2, 0, 2, 3],
        ..MeshInput::default()
    }
}

/// A frame on uvs the mesh doesn't carry follows set 0, as the shaders do without a tangent
/// stream. Set 0 runs along y here, which neither set 1 nor the tangent of no uvs does.
#[test]
fn a_frame_outside_the_uv_sets_follows_set_0() {
    let along_x = turned_sets();
    let input = MeshInput {
        uvs: Some(vec![
            [1.0, 1.0].into(),
            [1.0, 0.0].into(),
            [0.0, 0.0].into(),
            [0.0, 1.0].into(),
        ]),
        uvs1: along_x.uvs.clone(),
        sections: vec![Span { first: 0, count: 6 }],
        ..along_x
    };
    let mut material = normal_mapped(0);
    material.frame_uv = Material::FRAME_OTHER;
    let (a, _) = library_cook("other-frame-tangents", input, vec![material]);
    let got = library_tangents(&a);
    assert!(!got.is_empty());
    for (_, _, t) in got {
        assert!(Vec3::from_slice(&t).dot(-Vec3::Y) > 0.999, "{t:?}");
    }
}

/// Sections over the same triangles, on different uv sets, each hold their own tangents.
#[test]
fn overlapping_sections_compute_their_own_tangents() {
    let span = Span { first: 0, count: 6 };
    let input = MeshInput {
        uvs1: Some(vec![
            [1.0, 1.0].into(),
            [1.0, 0.0].into(),
            [0.0, 0.0].into(),
            [0.0, 1.0].into(),
        ]),
        sections: vec![span, span],
        ..turned_sets()
    };
    let materials = vec![normal_mapped(0), normal_mapped(1)];
    let (a, _) = library_cook("overlap-tangents", input, materials);
    let got = library_tangents(&a);
    assert!(got.iter().any(|t| t.0 == 0) && got.iter().any(|t| t.0 == 1));
    for (section, _, t) in got {
        let want = if section == 0 { Vec3::X } else { -Vec3::Y };
        assert!(Vec3::from_slice(&t).dot(want) > 0.9999, "{section}: {t:?}");
    }
}

/// A copy can bind a material the cook never sees, so a uv set no section reads stays.
#[test]
fn a_uv_set_no_section_reads_stays_for_copies() {
    let input = MeshInput {
        uvs1: Some(vec![[0.25, 0.75].into(); 4]),
        sections: vec![Span { first: 0, count: 6 }],
        ..turned_sets()
    };
    let (a, _) = library_cook("unread-uv1", input, vec![normal_mapped(0)]);
    let e = a.find(AssetId::from_name("m/mesh")).unwrap();
    let blobs = a.blobs(e);
    let read = |b: &orr_pack::Blob| {
        let mut v = vec![0; b.raw_len as usize];
        a.read(b, &mut v).unwrap();
        v
    };
    let info = MeshInfo::parse(&read(&blobs[0])).unwrap();
    let page = &info.pages[0];
    let mut resident = vec![0; page.resident_len as usize];
    mesh::decode_page(&read(&blobs[page.blob as usize]), &mut resident).unwrap();
    let view = PageView::new(&resident, &info.header).unwrap();
    let c = view.cluster(0);
    let uv = view.uv(&c, 0, 1);
    assert!(
        (uv[0] - 0.25).abs() < 1e-3 && (uv[1] - 0.75).abs() < 1e-3,
        "{uv:?}"
    );
}

/// The shaders read a uv of zero from a set a mesh lacks, and so does the cook, with a warning.
#[test]
fn a_uv_set_the_mesh_lacks_reads_zero() {
    let span = Span { first: 0, count: 6 };
    let input = MeshInput {
        sections: vec![span],
        ..turned_sets()
    };
    let (a, warnings) = library_cook("missing-uv1", input, vec![normal_mapped(1)]);
    assert!(
        warnings.iter().any(|w| w.contains("uv set 1")),
        "{warnings:?}"
    );
    assert!(!library_tangents(&a).is_empty());
}

/// Uvs on whole numbers stay on them, whatever clusters they fall in: a strip across u = 1 to
/// u = 6, its other uvs off any power-of-two grid, read back each whole u exactly and every
/// other uv within half its cluster's step.
#[test]
fn whole_uvs_inside_a_cluster_are_exact() {
    let columns = 61;
    let u = |i: u32| {
        let u = 0.3 + 0.1 * f64::from(i);
        (if (u - u.round()).abs() < 1e-9 {
            u.round()
        } else {
            u
        }) as f32
    };
    let v = [0.15f32, 0.85];
    let mut input = MeshInput::default();
    let mut uvs = Vec::new();
    for i in 0..columns {
        for (row, &v) in v.iter().enumerate() {
            input.positions.push(Vec3::new(u(i), row as f32, 0.0));
            input.normals.push(Vec3::Z);
            uvs.push([u(i), v].into());
        }
    }
    input.uvs = Some(uvs);
    for i in 0..columns - 1 {
        let c = 2 * i;
        input.indices.extend([c, c + 2, c + 1, c + 1, c + 2, c + 3]);
    }
    input.sections = vec![Span {
        first: 0,
        count: input.indices.len() as u32,
    }];
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "m".into(),
                source: Source::Mesh {
                    mesh: input,
                    materials: vec![Material::default()],
                    settings: MeshSettings {
                        max_triangles: 16,
                        ..MeshSettings::default()
                    },
                },
            }],
        )],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
    let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
    let blobs = a.blobs(a.find(AssetId::from_name("m/mesh")).unwrap());
    let info = MeshInfo::parse(&read_blob(&a, &blobs[0])).unwrap();
    let (mut whole, mut clusters) = (0, 0);
    for page in &info.pages {
        let mut resident = vec![0; page.resident_len as usize];
        mesh::decode_page(&read_blob(&a, &blobs[page.blob as usize]), &mut resident).unwrap();
        let view = PageView::new(&resident, &info.header).unwrap();
        for ci in 0..view.cluster_count() {
            let c = view.cluster(ci);
            clusters += 1;
            for k in 0..u32::from(c.vertex_count) {
                let [x, y, _] = view.position(&c, k);
                let i = (0..columns).find(|&i| (u(i) - x).abs() < 1e-3).unwrap();
                let want = [u(i), v[y.round() as usize]];
                let got = view.uv(&c, k, 0);
                if want[0] == want[0].round() {
                    assert_eq!(got[0], want[0], "cluster {ci} vertex {k}");
                    whole += 1;
                }
                for a in 0..2 {
                    let step = c.uv_step[0][a];
                    assert!(step.log2().fract() == 0.0, "{step}");
                    assert!(
                        (got[a] - want[a]).abs() <= 0.5 * step,
                        "cluster {ci} vertex {k}: {got:?} for {want:?}"
                    );
                }
            }
        }
    }
    assert!(
        clusters > 1 && whole >= 12,
        "{clusters} clusters, {whole} whole uvs"
    );
}

/// The 99th percentile over the normal map's texel centers, `texels` a side, of the degrees
/// between `tangent` and the tangent the runtime derives from each triangle's grid positions and
/// stored uvs (`orr_scene.hit`'s `triangle_tangent` without `MESH_TANGENT`).
fn derived_angle_p99(a: &Archive, tangent: Vec3, texels: u32) -> f32 {
    let blobs = a.blobs(a.find(AssetId::from_name("m/mesh")).unwrap());
    let info = MeshInfo::parse(&read_blob(a, &blobs[0])).unwrap();
    let mut angles = Vec::new();
    for page in &info.pages {
        let mut resident = vec![0; page.resident_len as usize];
        mesh::decode_page(&read_blob(a, &blobs[page.blob as usize]), &mut resident).unwrap();
        let view = PageView::new(&resident, &info.header).unwrap();
        for ci in 0..view.cluster_count() {
            let c = view.cluster(ci);
            for t in 0..u32::from(c.tri_count) {
                let k = view.triangle(&c, t).map(u32::from);
                let p = k.map(|v| Vec3::from(view.position(&c, v)));
                let uv = k.map(|v| Vec2::from(view.uv(&c, v, 0)));
                let n = k
                    .iter()
                    .map(|&v| Vec3::from(view.normal(&c, v)))
                    .sum::<Vec3>()
                    .normalize();
                let (e1, e2, uv1, uv2) = (p[1] - p[0], p[2] - p[0], uv[1] - uv[0], uv[2] - uv[0]);
                let r = uv1.x * uv2.y - uv2.x * uv1.y;
                let along = (e1 * uv2.y - e2 * uv1.y) / r;
                let derived = (along - n * n.dot(along)).normalize();
                let degrees = derived.dot(tangent).clamp(-1.0, 1.0).acos().to_degrees();
                angles.push((degrees, texel_centers(uv, texels)));
            }
        }
    }
    angles.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: u32 = angles.iter().map(|a| a.1).sum();
    let mut sum = 0;
    angles
        .iter()
        .find(|a| {
            sum += a.1;
            f64::from(sum) >= 0.99 * f64::from(total)
        })
        .unwrap()
        .0
}

/// Texel centers of a map `texels` a side inside the uv triangle `uv`.
fn texel_centers(uv: [Vec2; 3], texels: u32) -> u32 {
    let p = uv.map(|c| c * texels as f32);
    let cross = |a: Vec2, b: Vec2, q: Vec2| (b - a).perp_dot(q - a);
    let s = cross(p[0], p[1], p[2]).signum();
    let lo = p[0].min(p[1]).min(p[2]).floor().as_ivec2();
    let hi = p[0].max(p[1]).max(p[2]).ceil().as_ivec2();
    let mut n = 0;
    for y in lo.y..hi.y {
        for x in lo.x..hi.x {
            let q = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            if (0..3).all(|i| s * cross(p[i], p[(i + 1) % 3], q) >= 0.0) {
                n += 1;
            }
        }
    }
    n
}

/// A mesh drops its tangents only where those the runtime derives from the stored grid
/// positions and uvs shade its normal map within the budget, as the check measures them. A
/// flat grid turned off the grid's axes, whose stored tangents are its uvs' own, its vertices
/// 3.3, 26.4 and 211 grid steps apart: the grid turns the tangents its triangles derive by
/// degrees at the coarsest step.
#[test]
fn the_tangent_check_measures_the_tangents_the_runtime_derives() {
    const QUADS: u32 = 31;
    const TEXELS: u32 = 256;
    let spacing = 3.3 * 2f32.powi(-8);
    let (along, across) = (Vec3::new(0.8, 0.6, 0.0), Vec3::new(-0.6, 0.8, 0.0));
    let mut input = MeshInput::default();
    let mut uvs = Vec::new();
    for j in 0..=QUADS {
        for i in 0..=QUADS {
            let at = [i, j].map(|k| spacing * k as f32);
            input.positions.push(along * at[0] + across * at[1]);
            input.normals.push(Vec3::Z);
            // 8 x 8 texels a quad, a quarter texel off the texel grid along u, so that no texel
            // center lies on an edge.
            let texel = [8.0 * i as f32 + 0.25, 8.0 * j as f32];
            uvs.push(Vec2::from(texel.map(|t| t / TEXELS as f32)));
        }
    }
    input.uvs = Some(uvs);
    // v runs along `across`, which with v down the image is a bitangent sign of -1.
    input.tangents = Some(vec![along.extend(-1.0); input.positions.len()]);
    let w = QUADS + 1;
    for j in 0..QUADS {
        for i in 0..QUADS {
            let c = j * w + i;
            input
                .indices
                .extend([c, c + 1, c + w, c + 1, c + w + 1, c + w]);
        }
    }
    input.sections = vec![Span {
        first: 0,
        count: input.indices.len() as u32,
    }];
    // Tilted all the way to the tangent, so a shading normal turns as far as its tangent.
    let normals = image::RgbaImage::from_pixel(TEXELS, TEXELS, image::Rgba([255, 128, 128, 255]));
    let budget = 1.0;
    let (mut kept, mut dropped) = (false, false);
    for step_log2 in [-8, -11, -14] {
        let settings = MeshSettings {
            position_step_log2: step_log2,
            tangent_error_max: Some(budget),
            ..MeshSettings::default()
        };
        let name = format!("tangent-check-{}", -step_log2);
        let materials = vec![normal_mapped(0)];
        let (a, report) =
            library_cook_with(&name, input.clone(), materials, settings, normals.clone());
        let mut entries = report.assets.iter().flat_map(|a| &a.entries);
        let check = entries
            .find_map(|e| e.mesh.as_ref()?.tangents.clone())
            .unwrap();
        let measured = check.angle_p99.unwrap();
        let runtime = derived_angle_p99(&a, along, TEXELS);
        // The check reports the upper edge of its 0.01 degree bin.
        assert!(
            (measured - 0.01 - 0.02..=measured + 0.02).contains(&runtime),
            "step 2^{step_log2}: the check measures {measured} deg, the runtime derives {runtime}"
        );
        if check.derived {
            assert!(runtime <= budget, "step 2^{step_log2}: {runtime} deg");
        }
        kept |= !check.derived;
        dropped |= check.derived;
    }
    assert!(kept && dropped, "kept {kept}, dropped {dropped}");
}
