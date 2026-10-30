//! glTF import beyond plain float meshes: quantised and meshopt-compressed buffers, part
//! attributes, material extensions and overrides, model selection with joined grids, and
//! the textures a mesh cook reads (opacity for opaque-first triangles, displacement bounds).

#[path = "common/glb.rs"]
mod glb;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glb::{Glb, I8, U16};
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, Manifest, MaterialPatch, MeshPatch, NodeSelect,
    PartAttribute, Source,
};
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{self, Cluster, MeshInfo, PageView};
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, AssetId, Class, Target};

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cook(path: &Path, settings: GltfSettings) -> orr_cook::Result<(Archive, orr_cook::CookReport)> {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "g".into(),
                source: Source::Gltf {
                    path: path.into(),
                    settings,
                },
            }],
        )],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default())?;
    let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
    for e in a.entries() {
        a.verify(e).unwrap();
    }
    Ok((a, cooked.report))
}

fn read(a: &Archive, b: &orr_pack::Blob) -> Vec<u8> {
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

/// A cluster with its triangles' corner uvs and positions and its vertex normals.
type Corners = (
    Cluster,
    Vec<[[f32; 2]; 3]>,
    Vec<[[f32; 3]; 3]>,
    Vec<[f32; 3]>,
);

/// The mesh entry `name`: its info and every cluster.
struct Cooked {
    info: MeshInfo,
    clusters: Vec<Corners>,
}

fn mesh_of(a: &Archive, name: &str) -> Cooked {
    let e = a.find(AssetId::from_name(name)).unwrap();
    let blobs = a.blobs(e);
    let info = MeshInfo::parse(&read(a, &blobs[0])).unwrap();
    let mut clusters = Vec::new();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        mesh::decode_page(&read(a, &blobs[p.blob as usize]), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            let tris = (0..u32::from(c.tri_count)).map(|t| view.triangle(&c, t).map(u32::from));
            let uvs = tris.clone().map(|t| t.map(|v| view.uv(&c, v))).collect();
            let pos = tris.map(|t| t.map(|v| view.position(&c, v))).collect();
            let normals = (0..u32::from(c.vertex_count))
                .map(|v| view.normal(&c, v))
                .collect();
            clusters.push((c, uvs, pos, normals));
        }
    }
    Cooked { info, clusters }
}

/// Triangle positions as bits, corners rotated so the smallest comes first.
fn triangle_set(c: &Cooked) -> Vec<[[u32; 3]; 3]> {
    let mut out: Vec<_> = c
        .clusters
        .iter()
        .flat_map(|(_, _, p, _)| p.iter())
        .map(|t| {
            let b = t.map(|v| v.map(f32::to_bits));
            let r = (0..3).min_by_key(|&i| b[i]).unwrap();
            [b[r], b[(r + 1) % 3], b[(r + 2) % 3]]
        })
        .collect();
    out.sort_unstable();
    out
}

/// A `n` x `n` grid of quads on [0, 1]^2 in xz, uv = xz, height from `h`.
fn grid(n: u32, h: impl Fn(f32, f32) -> f32) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
    let (mut p, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            p.push([u, h(u, v), v]);
            uv.push([u, v]);
        }
    }
    let w = n + 1;
    for j in 0..n {
        for i in 0..n {
            let c = j * w + i;
            idx.extend([c, c + w, c + 1, c + 1, c + w, c + w + 1]);
        }
    }
    (p, uv, idx)
}

fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(w, h, |x, y| image::Rgba(f(x, y)));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

#[test]
fn quantised_compressed_files_cook_like_float_ones() {
    let d = dir("gltf-quantised");
    let (pos, uv, idx) = grid(24, |u, v| 0.2 * (u * 7.0).sin() * (v * 5.0).cos());
    // Quantised: positions u16 on [0, 65535] with the node's scale and translation.
    let q: Vec<[u16; 4]> = pos
        .iter()
        .map(|p| {
            let c = p.map(|x| ((x + 0.5) * 32000.0).round() as u16);
            [c[0], c[1], c[2], 0]
        })
        .collect();
    let normals: Vec<[i8; 4]> = pos.iter().map(|_| [0, 127, 0, 0]).collect();
    let uvq: Vec<[u16; 2]> = uv
        .iter()
        .map(|t| t.map(|x| (x * 65535.0).round() as u16))
        .collect();
    let node =
        r#"{"mesh":0,"scale":[0.00003125,0.00003125,0.00003125],"translation":[-0.25,1.0,3.5]}"#;

    let mut g = Glb::default();
    let pv = g.compressed_view(bytemuck::cast_slice(&q), 8, "ATTRIBUTES", true);
    let pa = g.accessor(
        pv,
        U16,
        q.len(),
        "VEC3",
        r#","min":[0,0,0],"max":[65535,65535,65535]"#,
    );
    let nv = g.compressed_view(bytemuck::cast_slice(&normals), 4, "ATTRIBUTES", true);
    let na = g.accessor(nv, I8, normals.len(), "VEC3", r#","normalized":true"#);
    let tv = g.view(bytemuck::cast_slice(&uvq), None);
    let ta = g.accessor(tv, U16, uvq.len(), "VEC2", r#","normalized":true"#);
    let iv = g.compressed_view(bytemuck::cast_slice(&idx), 4, "TRIANGLES", false);
    let ia = g.accessor(iv, glb::U32, idx.len(), "SCALAR", "");
    let compressed = d.join("compressed.glb");
    g.write(
        &compressed,
        &format!(
            r#""extensionsUsed":["KHR_mesh_quantization","EXT_meshopt_compression"],
"extensionsRequired":["KHR_mesh_quantization","EXT_meshopt_compression"],
"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{node}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"NORMAL":{na},"TEXCOORD_0":{ta}}},"indices":{ia}}}]}}]"#
        ),
    );

    // The same values as floats.
    let mut g = Glb::default();
    let qf: Vec<[f32; 3]> = q
        .iter()
        .map(|c| [c[0], c[1], c[2]].map(f32::from))
        .collect();
    let pa = g.floats(&qf, "VEC3");
    let na = g.floats(&vec![[0.0f32, 1.0, 0.0]; qf.len()], "VEC3");
    let uvf: Vec<[f32; 2]> = uvq
        .iter()
        .map(|t| t.map(|x| f32::from(x) / 65535.0))
        .collect();
    let ta = g.floats(&uvf, "VEC2");
    let ia = g.indices(&idx);
    let plain = d.join("plain.glb");
    g.write(
        &plain,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{node}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"NORMAL":{na},"TEXCOORD_0":{ta}}},"indices":{ia}}}]}}]"#
        ),
    );
    let (a, _) = cook(&compressed, GltfSettings::default()).unwrap();
    let (b, _) = cook(&plain, GltfSettings::default()).unwrap();
    let (ma, mb) = (mesh_of(&a, "g/mesh"), mesh_of(&b, "g/mesh"));
    assert_eq!(triangle_set(&ma), triangle_set(&mb));
    assert_eq!(triangle_set(&ma).len(), idx.len() / 3);
    for (_, _, _, n) in &ma.clusters {
        assert!(n.iter().all(|n| (n[1] - 1.0).abs() < 1e-4));
    }
}

#[test]
fn float_part_attributes_round_to_parts() {
    let d = dir("gltf-parts");
    let (pos, _, idx) = grid(20, |_, _| 0.0);
    // Joint by x band, with float noise that rounds back to the band.
    let ids: Vec<[f32; 1]> = pos
        .iter()
        .enumerate()
        .map(|(i, p)| [(p[0] * 6.99).floor() + if i % 3 == 0 { 0.4 } else { -0.3 }])
        .collect();
    let mut g = Glb::default();
    let pa = g.floats(&pos, "VEC3");
    let id = g.floats(&ids, "SCALAR");
    let ia = g.indices(&idx);
    let path = d.join("parts.glb");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"_ID":{id}}},"indices":{ia}}}]}}]"#
        ),
    );
    let settings = GltfSettings {
        parts_from_attribute: Some(PartAttribute {
            name: "_ID".into(),
            parts: 9,
            ..Default::default()
        }),
        ..Default::default()
    };
    let (a, _) = cook(&path, settings).unwrap();
    let m = mesh_of(&a, "g/mesh");
    assert_eq!(m.info.parts.len(), 9);
    let mut cooked = [0u32; 9];
    for (c, ..) in &m.clusters {
        cooked[usize::from(c.part)] += u32::from(c.tri_count);
    }
    let mut expected = [0u32; 9];
    for t in idx.chunks(3) {
        let v = [0, 1, 2].map(|k| ids[t[k] as usize][0].round() as usize);
        let value = if v[0] == v[1] || v[0] == v[2] {
            v[0]
        } else if v[1] == v[2] {
            v[1]
        } else {
            v[0].min(v[1]).min(v[2])
        };
        expected[value] += 1;
    }
    assert_eq!(cooked, expected);
    assert_eq!(cooked[7..], [0, 0]);
}

/// One triangle per material, side by side.
fn per_material(g: &mut Glb, materials: usize) -> String {
    let mut prims = Vec::new();
    for m in 0..materials {
        let x = m as f32 * 2.0;
        let pos = [[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 0.0, 1.0]];
        let pa = g.floats(&pos, "VEC3");
        let na = g.floats(&[[0.0f32, 1.0, 0.0]; 3], "VEC3");
        let ia = g.indices(&[0, 2, 1]);
        prims.push(format!(
            r#"{{"attributes":{{"POSITION":{pa},"NORMAL":{na}}},"indices":{ia},"material":{m}}}"#
        ));
    }
    prims.join(",")
}

#[test]
fn material_extensions_map_to_openpbr() {
    let d = dir("gltf-materials");
    let mut g = Glb::default();
    let prims = per_material(&mut g, 4);
    let path = d.join("materials.glb");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{prims}]}}],
"extensionsUsed":["KHR_materials_clearcoat","KHR_materials_sheen","KHR_materials_specular","KHR_materials_ior","KHR_materials_anisotropy","KHR_materials_iridescence","KHR_materials_transmission","KHR_materials_volume","KHR_materials_dispersion","KHR_materials_diffuse_transmission"],
"extensionsRequired":["KHR_materials_clearcoat"],
"materials":[
 {{"name":"lacquer","extensions":{{
   "KHR_materials_clearcoat":{{"clearcoatFactor":0.7,"clearcoatRoughnessFactor":0.2}},
   "KHR_materials_sheen":{{"sheenColorFactor":[0.5,0.25,0.1],"sheenRoughnessFactor":0.4}},
   "KHR_materials_specular":{{"specularFactor":0.6,"specularColorFactor":[1.0,0.9,0.8]}},
   "KHR_materials_ior":{{"ior":1.7}},
   "KHR_materials_anisotropy":{{"anisotropyStrength":0.3,"anisotropyRotation":0.5}},
   "KHR_materials_iridescence":{{"iridescenceFactor":1.0,"iridescenceIor":1.25,"iridescenceThicknessMaximum":350.0}}}}}},
 {{"name":"glass","extensions":{{
   "KHR_materials_transmission":{{"transmissionFactor":0.9}},
   "KHR_materials_volume":{{"thicknessFactor":0.02,"attenuationDistance":0.5,"attenuationColor":[0.9,0.5,0.2]}},
   "KHR_materials_dispersion":{{"dispersion":0.5}}}}}},
 {{"name":"film","extensions":{{"KHR_materials_transmission":{{"transmissionFactor":1.0}}}}}},
 {{"name":"skin leaf","doubleSided":true,"extensions":{{
   "KHR_materials_diffuse_transmission":{{"diffuseTransmissionFactor":0.4,"diffuseTransmissionColorFactor":[0.3,0.8,0.2]}}}}}}]"#
        ),
    );
    let settings = GltfSettings {
        material_overrides: vec![
            (
                "skin".into(),
                MaterialPatch {
                    surface: Some(3),
                    subsurface_radius: Some(0.004),
                    thin_walled: Some(0),
                    ..Default::default()
                },
            ),
            (
                "glass".into(),
                MaterialPatch {
                    transmission_scatter_anisotropy: Some(0.25),
                    ..Default::default()
                },
            ),
        ],
        ..Default::default()
    };
    let (a, report) = cook(&path, settings).unwrap();
    let m = |i: usize| {
        let e = a
            .expect(
                AssetId::from_name(&format!("g/material/{i}")),
                Class::Material as u16,
                orr_pack::material::SCHEMA,
            )
            .unwrap();
        Material::from_bytes(&read(&a, &a.blobs(e)[0])).unwrap()
    };
    let lacquer = m(0);
    assert_eq!(
        (
            lacquer.coat_weight,
            lacquer.coat_roughness,
            lacquer.coat_ior
        ),
        (0.7, 0.2, 1.5)
    );
    assert_eq!(
        (
            lacquer.fuzz_weight,
            lacquer.fuzz_color,
            lacquer.fuzz_roughness
        ),
        (1.0, [0.5, 0.25, 0.1], 0.4)
    );
    assert_eq!(
        (lacquer.specular_weight, lacquer.specular_color),
        (0.6, [1.0, 0.9, 0.8])
    );
    assert_eq!(lacquer.specular_ior, 1.7);
    assert_eq!(
        (lacquer.specular_anisotropy, lacquer.specular_rotation),
        (0.3, 0.5)
    );
    assert_eq!(
        (lacquer.thin_film_thickness, lacquer.thin_film_ior),
        (350.0, 1.25)
    );
    let glass = m(1);
    assert_eq!(glass.transmission_weight, 0.9);
    assert_eq!(
        (glass.transmission_color, glass.transmission_depth),
        ([0.9, 0.5, 0.2], 0.5)
    );
    assert_eq!(glass.thin_walled, 0);
    assert_eq!(
        (
            glass.transmission_dispersion_scale,
            glass.transmission_dispersion_abbe_number
        ),
        (1.0, 40.0)
    );
    assert_eq!(glass.transmission_scatter_anisotropy, 0.25);
    let film = m(2);
    assert_eq!((film.transmission_weight, film.thin_walled), (1.0, 1));
    let leaf = m(3);
    assert_eq!(
        (leaf.subsurface_weight, leaf.subsurface_color),
        (0.4, [0.3, 0.8, 0.2])
    );
    assert_eq!(
        (leaf.surface, leaf.subsurface_radius, leaf.thin_walled),
        (3, 0.004, 0)
    );

    // The patched leaf scatters below the surface of an open triangle: the cook warns.
    let warnings: Vec<&String> = report
        .assets
        .iter()
        .flat_map(|a| &a.entries)
        .flat_map(|e| &e.warnings)
        .collect();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("section 3") && warnings[0].contains("border edges"));
}

/// Two models from one scene: a 1 m box and a 40 m slab, with their nodes' transforms.
fn rocks(path: &Path, a: &str, b: &str) {
    let mut g = Glb::default();
    let mut meshes = Vec::new();
    for size in [1.0f32, 40.0] {
        let pos = [
            [0.0, 0.0, 0.0],
            [size, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [size, 0.3, 0.5],
        ];
        let pa = g.floats(&pos, "VEC3");
        let ia = g.indices(&[0, 2, 1, 1, 2, 3]);
        meshes.push(format!(
            r#"{{"primitives":[{{"attributes":{{"POSITION":{pa}}},"indices":{ia}}}]}}"#
        ));
    }
    g.write(
        path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1,2]}}],
"nodes":[{{"name":"rock_a","mesh":0,{a}}},{{"name":"rock_b","mesh":1,{b}}},{{"name":"stray","mesh":0}}],
"meshes":[{}]"#,
            meshes.join(",")
        ),
    );
}

#[test]
fn selected_models_join_on_one_grid() {
    let d = dir("gltf-joined");
    let path = d.join("rocks.glb");
    rocks(
        &path,
        r#""translation":[3.0,0.5,-1.0]"#,
        r#""translation":[-2.0,0.0,0.25],"rotation":[0.0,0.70710677,0.0,0.70710677]"#,
    );
    let settings = GltfSettings {
        models: vec![NodeSelect {
            node: "rock".into(),
        }],
        joined: vec!["rock".into()],
        ..Default::default()
    };
    let (a, _) = cook(&path, settings.clone()).unwrap();
    for name in ["g/rock_a", "g/rock_b"] {
        assert!(a.find(AssetId::from_name(name)).is_some());
        let m = mesh_of(&a, &format!("{name}/mesh"));
        // The slab alone needs 2^-10 m; the box takes it too.
        assert_eq!(m.info.header.position_exponent, -10);
    }
    assert!(a.find(AssetId::from_name("g/mesh")).is_none());
    // Not joined, the box keeps the finest step; a mesh override reaches only its model.
    let (a, _) = cook(
        &path,
        GltfSettings {
            joined: Vec::new(),
            mesh_overrides: vec![(
                "rock_b".into(),
                MeshPatch {
                    position_step_log2: Some(-8),
                    ..Default::default()
                },
            )],
            ..settings.clone()
        },
    )
    .unwrap();
    let k = |name: &str| mesh_of(&a, name).info.header.position_exponent;
    assert_eq!((k("g/rock_a/mesh"), k("g/rock_b/mesh")), (-11, -8));

    for (ta, tb) in [
        (
            r#""translation":[3.0001,0.5,-1.0]"#,
            r#""translation":[0.0,0.0,0.0]"#,
        ),
        (
            r#""translation":[3.0,0.5,-1.0]"#,
            r#""rotation":[0.0,0.38268343,0.0,0.9238795]"#,
        ),
        (
            r#""translation":[3.0,0.5,-1.0]"#,
            r#""scale":[2.0,2.0,2.0]"#,
        ),
    ] {
        rocks(&path, ta, tb);
        let err = cook(&path, settings.clone()).err().unwrap();
        assert!(matches!(err, orr_cook::Error::Import { .. }), "{err}");
    }
}

#[test]
fn opaque_triangles_never_cover_cut_out_texels() {
    let d = dir("gltf-alpha");
    let (pos, uv, idx) = grid(32, |_, _| 0.0);
    // Opaque disc with a ragged edge, cut out elsewhere.
    let tex = png(128, 128, |x, y| {
        let (dx, dy) = (x as f32 - 60.0, y as f32 - 70.0);
        let r = (dx * dx + dy * dy).sqrt() + 4.0 * ((x * 7 + y * 3) % 5) as f32;
        let a = if r < 45.0 { 255 } else { 0 };
        [200, 180, 40, a]
    });
    let mut g = Glb::default();
    let pa = g.floats(&pos, "VEC3");
    let ta = g.floats(&uv, "VEC2");
    let ia = g.indices(&idx);
    let img = g.image(&tex);
    let path = d.join("alpha.glb");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"TEXCOORD_0":{ta}}},"indices":{ia},"material":0}}]}}],
"materials":[{{"alphaMode":"MASK","alphaCutoff":0.5,"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}}],
"textures":[{{"source":0}}],"images":[{{"bufferView":{img},"mimeType":"image/png"}}]"#
        ),
    );
    let (a, _) = cook(&path, GltfSettings::default()).unwrap();
    let opacity = texture_mip(&a, "g/texture/0/opacity", 0);
    let m = mesh_of(&a, "g/mesh");
    let (mut opaque, mut tested) = (0, 0);
    for (c, uvs, ..) in &m.clusters {
        for (t, uv) in uvs.iter().enumerate() {
            if t < usize::from(c.alpha_first) {
                opaque += 1;
                // Bilinear samples over the triangle all pass the cutoff.
                for i in 0..=8 {
                    for j in 0..=8 - i {
                        let (b0, b1) = (i as f32 / 8.0, j as f32 / 8.0);
                        let p = [0, 1]
                            .map(|k| uv[0][k] * (1.0 - b0 - b1) + uv[1][k] * b0 + uv[2][k] * b1);
                        assert!(opacity.bilinear(p) >= 0.5, "opaque triangle samples {p:?}");
                    }
                }
            } else {
                tested += 1;
            }
        }
    }
    assert!(
        opaque > 100 && tested > 100,
        "{opaque} opaque, {tested} alpha-tested"
    );
}

/// One mip of a cooked single-channel texture: BC4 or R16.
struct Mip {
    w: u32,
    h: u32,
    texels: Vec<f32>,
}

impl Mip {
    fn texel(&self, x: i64, y: i64) -> f32 {
        let x = x.rem_euclid(i64::from(self.w)) as usize;
        let y = y.rem_euclid(i64::from(self.h)) as usize;
        self.texels[y * self.w as usize + x]
    }

    fn bilinear(&self, uv: [f32; 2]) -> f32 {
        let x = f64::from(uv[0]) * f64::from(self.w) - 0.5;
        let y = f64::from(uv[1]) * f64::from(self.h) - 0.5;
        let (fx, fy) = ((x - x.floor()) as f32, (y - y.floor()) as f32);
        let (x0, y0) = (x.floor() as i64, y.floor() as i64);
        let t = |dx, dy| self.texel(x0 + dx, y0 + dy);
        let top = t(0, 0) + (t(1, 0) - t(0, 0)) * fx;
        let bottom = t(0, 1) + (t(1, 1) - t(0, 1)) * fx;
        top + (bottom - top) * fy
    }
}

fn texture_mip(a: &Archive, name: &str, mip: u32) -> Mip {
    let e = a.find(AssetId::from_name(name)).unwrap();
    let blobs = a.blobs(e);
    let info = TextureInfo::parse(&read(a, &blobs[0])).unwrap();
    let h = info.header;
    let bytes = info
        .mip_texels(&read(a, &blobs[h.mip_blob(mip)]), mip, 0)
        .unwrap();
    let [w, hh] = h.mip_size(mip);
    let texels = if h.vk_format == orr_pack::texture::vk::R16_UNORM {
        bytes
            .chunks(2)
            .map(|c| f32::from(u16::from_le_bytes([c[0], c[1]])) / 65535.0)
            .collect()
    } else {
        let (bw, bh) = (w.div_ceil(4) as usize, hh.div_ceil(4) as usize);
        let mut out = vec![0.0; (w * hh) as usize];
        for by in 0..bh {
            for bx in 0..bw {
                let mut px = [0u8; 16];
                bcdec_rs::bc4(&bytes[(by * bw + bx) * 8..][..8], &mut px, 4, false);
                for (i, &v) in px.iter().enumerate() {
                    let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
                    if x < w as usize && y < hh as usize {
                        out[y * w as usize + x] = f32::from(v) / 255.0;
                    }
                }
            }
        }
        out
    };
    Mip { w, h: hh, texels }
}

/// A displaced plane in two uv islands that meet at x = 0.5 with different uvs there.
fn displaced(path: &Path, tex: &[u8]) {
    let (mut pos, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    for island in 0..2u32 {
        let base = pos.len() as u32;
        let n = 12;
        for j in 0..=n {
            for i in 0..=n {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                pos.push([0.5 * (island as f32 + u), 0.0, v]);
                uv.push([0.45 * u + 0.55 * island as f32, v]);
            }
        }
        let w = n + 1;
        for j in 0..n {
            for i in 0..n {
                let c = base + j * w + i;
                idx.extend([c, c + w, c + 1, c + 1, c + w, c + w + 1]);
            }
        }
    }
    let mut g = Glb::default();
    let pa = g.floats(&pos, "VEC3");
    let na = g.floats(&vec![[0.0f32, 1.0, 0.0]; pos.len()], "VEC3");
    let ta = g.floats(&uv, "VEC2");
    let ia = g.indices(&idx);
    let img = g.image(tex);
    g.write(
        path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"NORMAL":{na},"TEXCOORD_0":{ta}}},"indices":{ia},"material":0}}]}}],
"extensionsUsed":["KHR_materials_displacement"],
"materials":[{{"extensions":{{"KHR_materials_displacement":{{"displacementGeometryFactor":0.4,"displacementGeometryOffset":-0.1,"displacementGeometryTexture":{{"index":0}}}}}}}}],
"textures":[{{"source":0}}],"images":[{{"bufferView":{img},"mimeType":"image/png"}}]"#
        ),
    );
}

#[test]
fn displacement_bounds_hold_every_mip() {
    let d = dir("gltf-displacement");
    let tex = png(64, 64, |x, y| {
        let v = (128.0 + 100.0 * ((x as f32 * 0.4).sin() * (y as f32 * 0.25).cos())) as u8;
        [v, v, v, 255]
    });
    let path = d.join("displaced.glb");
    displaced(&path, &tex);
    let (a, report) = cook(&path, GltfSettings::default()).unwrap();
    let e = a
        .find(AssetId::from_name("g/texture/0/displacement"))
        .unwrap();
    let info = TextureInfo::parse(&read(&a, &a.blobs(e)[0])).unwrap();
    let mips: Vec<Mip> = (0..u32::from(info.header.mip_count))
        .map(|m| texture_mip(&a, "g/texture/0/displacement", m))
        .collect();
    assert!(mips.len() > 3);
    let m = mesh_of(&a, "g/mesh");
    let half = |b: u16| half::f16::from_bits(b).to_f32();
    let mut checked = 0;
    for (c, uvs, pos, _) in &m.clusters {
        let (lo, hi) = (half(c.displacement_min), half(c.displacement_max));
        assert!(-0.1 <= lo && lo <= hi && hi <= 0.3 + 1e-3);
        let step = 2f32.powi(i32::from(m.info.header.position_exponent));
        let aabb_lo = c.anchor.map(|g| g as f32 * step);
        for (uv, p) in uvs.iter().zip(pos) {
            for (uv, p) in uv.iter().zip(p) {
                for mip in &mips {
                    let disp = 0.4 * mip.bilinear(*uv) - 0.1;
                    assert!(lo <= disp && disp <= hi, "{disp} outside {lo}..{hi}");
                    // Displaced along +y, the vertex stays in the rest AABB grown by the bound.
                    let y = p[1] + disp;
                    let top = aabb_lo[1] + f32::from(c.local_max[1]) * step;
                    assert!(aabb_lo[1] - lo.abs().max(hi.abs()) <= y);
                    assert!(y <= top + lo.abs().max(hi.abs()));
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 1000);
    // The islands' uvs at x = 0.5 sample different displacement.
    let warnings: Vec<&String> = report
        .assets
        .iter()
        .flat_map(|a| &a.entries)
        .flat_map(|e| &e.warnings)
        .collect();
    assert!(
        warnings.iter().any(|w| w.contains("uv seam")),
        "{warnings:?}"
    );

    // A declared range narrower than the texture fails.
    let settings = GltfSettings {
        material_overrides: vec![(
            String::new(),
            MaterialPatch {
                displacement_max: Some(0.1),
                ..Default::default()
            },
        )],
        ..Default::default()
    };
    let err = cook(&path, settings).err().unwrap();
    assert!(matches!(err, orr_cook::Error::Mesh { .. }), "{err}");
}

#[test]
fn material_textures_bind_their_slots() {
    let d = dir("gltf-slots");
    let noise = |seed: u32| {
        png(16, 16, move |x, y| {
            let v = ((x * 37 + y * 11 + seed * 5) % 251) as u8;
            [v, v / 2, 255 - v, v]
        })
    };
    let mut g = Glb::default();
    let prims = per_material(&mut g, 1);
    let images: Vec<usize> = (0..4).map(|i| g.image(&noise(i))).collect();
    let path = d.join("slots.glb");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{prims}]}}],
"materials":[{{"extensions":{{
  "KHR_materials_specular":{{"specularTexture":{{"index":0}}}},
  "KHR_materials_clearcoat":{{"clearcoatFactor":0.5,"clearcoatTexture":{{"index":1}}}},
  "KHR_materials_transmission":{{"transmissionFactor":0.8,"transmissionTexture":{{"index":2}}}},
  "KHR_materials_diffuse_transmission":{{"diffuseTransmissionFactor":1.0,"diffuseTransmissionColorTexture":{{"index":3}}}}}}}}],
"textures":[{{"source":0}},{{"source":1}},{{"source":2}},{{"source":3}}],
"images":[{}]"#,
            images
                .iter()
                .map(|v| format!(r#"{{"bufferView":{v},"mimeType":"image/png"}}"#))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
    let (a, _) = cook(&path, GltfSettings::default()).unwrap();
    let e = a.find(AssetId::from_name("g/material/0")).unwrap();
    let m = Material::from_bytes(&read(&a, &a.blobs(e)[0])).unwrap();
    let roles: HashMap<usize, &str> = [
        (slot::SPECULAR_WEIGHT, "specular_weight"),
        (slot::COAT, "coat"),
        (slot::TRANSMISSION, "transmission"),
        (slot::SUBSURFACE_COLOR, "subsurface_color"),
    ]
    .into();
    for (s, role) in roles {
        assert_eq!(
            m.textures[s].texture,
            AssetId::from_name(&format!("g/texture/0/{role}")),
            "{role}"
        );
    }
    // Weight factors are folded into their textures.
    assert_eq!((m.coat_weight, m.transmission_weight), (1.0, 1.0));
    let coat = texture_mip(&a, "g/texture/0/coat", 0);
    assert!(coat.texels.iter().all(|&t| t <= 0.5 + 1.0 / 255.0));
}
