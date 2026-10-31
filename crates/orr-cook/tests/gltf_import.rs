//! glTF import beyond plain float meshes: quantised and meshopt-compressed buffers, part
//! attributes, material extensions and overrides, model selection with joined grids, and
//! the textures a mesh cook reads (opacity for opaque-first triangles and micromaps,
//! displacement bounds).

#[path = "common/glb.rs"]
mod glb;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glb::{Glb, I8, U16};
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, LodMode, Manifest, MaterialPatch, MeshPatch,
    MeshSettings, NodeSelect, OmmSettings, PartAttribute, Source,
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

/// Micromap index of barycentrics (u, v) at `level`, for the point `c0 w + c1 u + c2 v`
/// (the Vulkan specification's reference code).
fn bary_to_index(u: f32, v: f32, level: u32) -> u32 {
    let n = 1u32 << level;
    let (fu, fv) = (u.clamp(0.0, 1.0) * n as f32, v.clamp(0.0, 1.0) * n as f32);
    let (mut iu, iv) = ((fu as u32).min(n - 1), (fv as u32).min(n - 1));
    let (uf, vf) = (fu - iu as f32, fv - iv as f32);
    let iuv = iu + iv;
    if iuv >= n {
        iu -= iuv - n + 1;
    }
    let mut iw = !(iu + iv);
    if uf + vf >= 1.0 && iuv < n - 1 {
        iw = iw.wrapping_sub(1);
    }
    let mut b0 = !(iu ^ iw) & (n - 1);
    let t = (iu ^ iv) & b0;
    let mut f = t;
    f ^= f >> 1;
    f ^= f >> 2;
    f ^= f >> 4;
    f ^= f >> 8;
    let mut b1 = ((f ^ iu) & !b0) | t;
    for b in [&mut b0, &mut b1] {
        *b = (*b | (*b << 8)) & 0x00ff00ff;
        *b = (*b | (*b << 4)) & 0x0f0f0f0f;
        *b = (*b | (*b << 2)) & 0x33333333;
        *b = (*b | (*b << 1)) & 0x55555555;
    }
    b0 | (b1 << 1)
}

/// A 4x4 card atlas of 64-texel leaves with soft, ragged edges; odd cells have a one-texel
/// transparent vein through the leaf.
fn card_atlas() -> Vec<u8> {
    png(256, 256, |x, y| {
        let cell = (y / 64) * 4 + x / 64;
        let (lx, ly) = ((x % 64) as f32 + 0.5 - 32.0, (y % 64) as f32 + 0.5 - 32.0);
        let (rx, ry) = (
            14.0 + (cell % 4) as f32 * 4.0,
            26.0 - (cell / 4) as f32 * 2.0,
        );
        let d = ((lx / rx).powi(2) + (ly / ry).powi(2)).sqrt();
        let ragged = 0.08 * (((x * 13 + y * 7) % 11) as f32 / 11.0);
        let mut a = ((1.0 - d - ragged) * 6.0 + 0.3).clamp(0.0, 1.0);
        if cell % 2 == 1 && (lx - ly * 0.3).abs() < 0.5 {
            a = 0.0;
        }
        [60, 140, 40, (a * 255.0).round() as u8]
    })
}

/// `count` cards of 2x2 quads at scattered places and turns, each over one atlas cell.
fn cards(count: u32) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
    let (mut pos, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    for c in 0..count {
        let origin = [rand() * 4.0, rand() * 4.0, rand() * 4.0];
        let (yaw, pitch) = (rand() * std::f32::consts::TAU, rand() * 3.0 - 1.5);
        let right = [yaw.cos(), 0.0, yaw.sin()];
        let up = [
            -yaw.sin() * pitch.sin(),
            pitch.cos(),
            yaw.cos() * pitch.sin(),
        ];
        let cell = c % 16;
        let (u0, v0) = ((cell % 4) as f32 * 0.25, (cell / 4) as f32 * 0.25);
        let flip = c % 3 == 0;
        let first = pos.len() as u32;
        for j in 0..3 {
            for i in 0..3 {
                let (s, t) = (i as f32 / 2.0, j as f32 / 2.0);
                pos.push([0, 1, 2].map(|a| origin[a] + 0.3 * (s * right[a] + t * up[a])));
                let s = if flip { 1.0 - s } else { s };
                uv.push([u0 + (s * 62.0 + 1.0) / 256.0, v0 + (t * 62.0 + 1.0) / 256.0]);
            }
        }
        for j in 0..2 {
            for i in 0..2 {
                let k = first + j * 3 + i;
                idx.extend([k, k + 3, k + 1, k + 1, k + 3, k + 4]);
            }
        }
    }
    (pos, uv, idx)
}

#[test]
fn micromap_states_never_contradict_the_alpha_test() {
    let d = dir("gltf-omm");
    let (pos, uv, idx) = cards(160);
    let mut g = Glb::default();
    let pa = g.floats(&pos, "VEC3");
    let ta = g.floats(&uv, "VEC2");
    let ia = g.indices(&idx);
    let img = g.image(&card_atlas());
    let path = d.join("cards.glb");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"TEXCOORD_0":{ta}}},"indices":{ia},"material":0}}]}}],
"materials":[{{"alphaMode":"MASK","alphaCutoff":0.3,"doubleSided":true,"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}}],
"textures":[{{"source":0}}],"images":[{{"bufferView":{img},"mimeType":"image/png"}}]"#
        ),
    );
    let defaults = OmmSettings::default();
    let capped = OmmSettings {
        max_level: 4,
        target_edge: 0.0,
        cluster_bytes_max: 1024,
    };
    let discrete = LodMode::Discrete {
        errors: vec![0.01, 0.05],
    };
    for (lod, omm) in [
        (discrete.clone(), defaults.clone()),
        (discrete, capped.clone()),
        (LodMode::Dag, defaults),
    ] {
        let settings = GltfSettings {
            mesh: MeshSettings {
                lod,
                omm: Some(omm.clone()),
                ..Default::default()
            },
            ..Default::default()
        };
        let (a, report) = cook(&path, settings).unwrap();
        let opacity = texture_mip(&a, "g/texture/0/opacity", 0);
        let e = a.find(AssetId::from_name("g/mesh")).unwrap();
        let blobs = a.blobs(e);
        let info = MeshInfo::parse(&read(&a, &blobs[0])).unwrap();
        let (mut known, mut samples, mut top_level) = (0u64, 0u64, 0);
        for p in &info.pages {
            let mut page = vec![0; p.resident_len as usize];
            mesh::decode_page(&read(&a, &blobs[p.blob as usize]), &mut page).unwrap();
            let view = PageView::new(&page, &info.header).unwrap();
            let maps = view.micromaps().unwrap();
            corrupt_micromaps_are_rejected(&page, &info.header);
            for i in 0..maps.entry_count() {
                top_level = top_level.max(maps.entry(i).level);
            }
            for ci in 0..view.cluster_count() {
                let c = view.cluster(ci);
                for t in u32::from(c.alpha_first)..u32::from(c.tri_count) {
                    let index = view
                        .omm_index(&c, t)
                        .expect("alpha-tested triangles have micromaps");
                    let uv = view.triangle(&c, t).map(|v| view.uv(&c, u32::from(v)));
                    let level = u32::try_from(index).map_or(0, |e| u32::from(maps.entry(e).level));
                    let n = (2 << level).min(32);
                    for j in 0..n {
                        for i in 0..n - j {
                            for (du, dv) in [(0.2, 0.3), (0.65, 0.25)] {
                                let (bu, bv) =
                                    ((i as f32 + du) / n as f32, (j as f32 + dv) / n as f32);
                                if bu + bv > 1.0 {
                                    continue;
                                }
                                let state = match u32::try_from(index) {
                                    Ok(e) => maps.state(e, bary_to_index(bu, bv, level)),
                                    Err(_) => (-1 - index) as u8,
                                };
                                samples += 1;
                                if state >= 2 {
                                    continue;
                                }
                                known += 1;
                                let bw = 1.0 - bu - bv;
                                let p =
                                    [0, 1].map(|k| uv[0][k] * bw + uv[1][k] * bu + uv[2][k] * bv);
                                let pass = opacity.bilinear(p) >= 0.3;
                                assert_eq!(
                                    pass,
                                    state == 1,
                                    "state {state} at {p:?}, triangle {t}"
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(
            known * 2 > samples,
            "{known} of {samples} samples in known states"
        );
        if omm == capped {
            // Unadaptive level 4 is 64 B per triangle, over the cap.
            assert!(
                top_level < omm.max_level.into(),
                "level {top_level} under the cap"
            );
        }
        let entry = report.assets[0]
            .entries
            .iter()
            .find(|e| e.name == "g/mesh")
            .unwrap();
        for l in &entry.mesh.as_ref().unwrap().levels {
            assert!(l.omm_triangles > 0 && l.omm_bytes > 0);
            assert!(
                l.omm_unknown > 0.0 && l.omm_unknown < 0.5,
                "unknown share {}",
                l.omm_unknown
            );
        }
    }
}

/// A usage count off by one, an entry past the page and an index past the entries each fail
/// [`PageView::micromaps`].
fn corrupt_micromaps_are_rejected(page: &[u8], mesh: &mesh::MeshHeader) {
    let h: mesh::PageHeader = bytemuck::pod_read_unaligned(&page[..size_of::<mesh::PageHeader>()]);
    let rejects = |edit: &dyn Fn(&mut Vec<u8>)| {
        let mut p = page.to_vec();
        edit(&mut p);
        PageView::new(&p, mesh).unwrap().micromaps().is_err()
    };
    let usage = std::mem::offset_of!(mesh::PageHeader, omm_usage);
    assert!(rejects(&|p| p[usage] ^= 1));
    let entry = h.omm_offset as usize;
    assert!(rejects(
        &|p| p[entry..entry + 4].copy_from_slice(&h.len.to_le_bytes())
    ));
    let view = PageView::new(page, mesh).unwrap();
    let c = (0..view.cluster_count())
        .map(|i| view.cluster(i))
        .find(|c| c.omm_first != Cluster::NO_OMM)
        .unwrap();
    let at = (h.omm_offset + c.omm_first) as usize;
    let past = view.micromaps().unwrap().entry_count() as i16;
    assert!(rejects(
        &|p| p[at..at + 2].copy_from_slice(&past.to_le_bytes())
    ));
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

/// A scene with a plain node and an instanced one: node `tree` carries
/// `EXT_mesh_gpu_instancing` (required) and a translation of its own.
fn instanced(path: &Path, instances: &[([f32; 3], [f32; 4], [f32; 3])]) {
    let mut g = Glb::default();
    let quad = [
        [0.0f32, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ];
    let pa = g.floats(&quad, "VEC3");
    let ia = g.indices(&[0, 2, 1, 1, 2, 3]);
    let t = g.floats(&instances.iter().map(|i| i.0).collect::<Vec<_>>(), "VEC3");
    let r = g.floats(&instances.iter().map(|i| i.1).collect::<Vec<_>>(), "VEC4");
    let s = g.floats(&instances.iter().map(|i| i.2).collect::<Vec<_>>(), "VEC3");
    g.write(
        path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1]}}],
"extensionsUsed":["EXT_mesh_gpu_instancing"],"extensionsRequired":["EXT_mesh_gpu_instancing"],
"nodes":[{{"name":"ground","mesh":0,"scale":[8,1,8]}},
 {{"name":"tree","mesh":0,"translation":[10,0,0],"extensions":{{"EXT_mesh_gpu_instancing":{{"attributes":{{"TRANSLATION":{t},"ROTATION":{r},"SCALE":{s}}}}}}}}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa}}},"indices":{ia}}}]}}]"#
        ),
    );
}

#[test]
fn gpu_instancing_becomes_instance_pages() {
    let d = dir("gltf-instancing");
    let path = d.join("forest.glb");
    let q = glam::Quat::from_rotation_y(0.7).to_array();
    let instances = [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
        ([3.0, 0.5, -2.0], q, [2.0, 2.0, 2.0]),
        ([300.0, 0.0, 40.0], q, [1.0, 1.5, 1.0]),
    ];
    instanced(&path, &instances);
    let (a, _) = cook(&path, GltfSettings::default()).unwrap();

    // The instanced node is a model in its mesh's space; the scene model keeps the rest.
    let mesh = mesh_of(&a, "g/tree/mesh");
    assert_eq!(mesh.info.header.bounds.max, [1.0, 0.0, 1.0]);
    assert_eq!(
        mesh_of(&a, "g/mesh").info.header.bounds.max,
        [8.0, 0.0, 8.0]
    );
    let model = AssetId::from_name("g/tree");
    let pages: Vec<&orr_pack::Entry> = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
        .collect();
    assert_eq!(pages.len(), 3, "one page per cell");
    let mut placed = Vec::new();
    for e in pages {
        assert!(a.name(e).starts_with("g/instances/"));
        let blobs = a.blobs(e);
        let head = read(&a, &blobs[0]);
        let info = orr_pack::instance::InstancePageInfo::parse(&head).unwrap();
        assert_eq!(info.prototypes[0].model, model);
        let bytes = read(&a, &blobs[1]);
        let (cell, records) = orr_pack::instance::decode(&head, &bytes).unwrap();
        let edge = 256.0;
        for (_, m) in records {
            let p = [0, 1, 2].map(|k| m.rows[k][3] + cell[k] as f32 * edge);
            let x = [0, 1, 2].map(|k| m.rows[k][0]);
            placed.push((p, x));
        }
    }
    placed.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]));
    for ((p, x), (t, r, s)) in placed.iter().zip(&instances) {
        let r = glam::Quat::from_array(*r);
        let want_p = glam::Vec3::from(*t) + glam::Vec3::new(10.0, 0.0, 0.0);
        let want_x = r * glam::Vec3::X * s[0];
        assert!(
            (glam::Vec3::from(*p) - want_p).abs().max_element() < 0.01,
            "{p:?} {want_p}"
        );
        assert!(
            (glam::Vec3::from(*x) - want_x).abs().max_element() < 0.01,
            "{x:?} {want_x}"
        );
    }

    // An unchanged rebuild links the pages with the rest.
    let file = d.join("forest.orr");
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::File(file.clone()),
            vec![AssetSpec {
                name: "g".into(),
                source: Source::Gltf {
                    path: path.clone(),
                    settings: GltfSettings::default(),
                },
            }],
        )],
    };
    let first = orr_cook::cook(&manifest, &Default::default()).unwrap();
    let bytes = std::fs::read(&file).unwrap();
    let again = orr_cook::cook(&manifest, &Default::default()).unwrap();
    assert!(!again.report.archives[0].written);
    assert_eq!(std::fs::read(&file).unwrap(), bytes);
    let count = |c: &orr_cook::Cooked| c.report.assets[0].entries.len();
    assert_eq!(count(&again), count(&first));
}

/// A unit quad facing +z as a library mesh with `material`, then texture asset `texture`
/// from `pixels` (64 x 64, one channel), cooked.
fn library_quad(
    material: Material,
    settings: MeshSettings,
    texture: &str,
    pixels: impl Fn(u32, u32) -> u8,
) -> Archive {
    use glam::{Vec2, Vec3};
    use orr_cook::{MeshInput, TextureSettings, Usage};
    let path = dir(&format!("library-{texture}")).join("t.png");
    std::fs::write(&path, png(64, 64, |x, y| [pixels(x, y); 4])).unwrap();
    let quad = MeshInput {
        positions: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        normals: vec![Vec3::Z; 4],
        uvs: Some(vec![
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ]),
        indices: vec![0, 1, 2, 0, 2, 3],
        sections: vec![orr_pack::Span { first: 0, count: 6 }],
        ..MeshInput::default()
    };
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: "quad".into(),
                    source: Source::Mesh {
                        mesh: quad,
                        materials: vec![material],
                        settings,
                    },
                },
                AssetSpec {
                    name: texture.into(),
                    source: Source::Texture {
                        path,
                        settings: TextureSettings {
                            usage: Usage::Linear1,
                            ..TextureSettings::default()
                        },
                    },
                },
            ],
        )],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
    assert_eq!(cooked.report.assets[0].name, "quad");
    Archive::from_memory(cooked.memory.remove(0)).unwrap()
}

#[test]
fn a_library_mesh_takes_micromaps_from_a_texture_asset_listed_after_it() {
    let mut leaf = Material {
        alpha_mode: Material::MASK,
        alpha_cutoff: 0.5,
        ..Material::default()
    };
    leaf.textures[slot::OPACITY].texture = AssetId::from_name("half");
    let a = library_quad(leaf, MeshSettings::default(), "half", |x, _| {
        if x < 40 { 255 } else { 0 }
    });
    let e = a.find(AssetId::from_name("quad/mesh")).unwrap();
    let blobs = a.blobs(e);
    let info = MeshInfo::parse(&read(&a, &blobs[0])).unwrap();
    let mut alpha = 0;
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        mesh::decode_page(&read(&a, &blobs[p.blob as usize]), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for ci in 0..view.cluster_count() {
            let c = view.cluster(ci);
            for t in u32::from(c.alpha_first)..u32::from(c.tri_count) {
                alpha += 1;
                assert!(
                    view.omm_index(&c, t).is_some(),
                    "alpha-tested triangle {t} has no micromap"
                );
            }
        }
    }
    assert!(alpha > 0, "no alpha-tested triangles");
}

#[test]
fn a_statically_displaced_library_mesh_binds_its_material_without_displacement() {
    let mut bumpy = Material {
        displacement_scale: 0.05,
        displacement_max: 0.05,
        ..Material::default()
    };
    bumpy.textures[slot::DISPLACEMENT].texture = AssetId::from_name("bumps");
    let settings = MeshSettings {
        displace_static: Some(orr_cook::StaticDisplacement {
            max_edge: 0.05,
            target_error: 0.001,
        }),
        ..MeshSettings::default()
    };
    let a = library_quad(bumpy, settings, "bumps", |x, y| {
        ((x * 7 + y * 3) % 64 * 4) as u8
    });
    let e = a.find(AssetId::from_name("quad")).unwrap();
    let model = orr_pack::model::ModelInfo::parse(&read(&a, &a.blobs(e)[0])).unwrap();
    assert_eq!(
        model.materials,
        vec![AssetId::from_name("quad/material/0/static")]
    );
    let e = a.find(model.materials[0]).unwrap();
    let m: Material = bytemuck::pod_read_unaligned(&read(&a, &a.blobs(e)[0]));
    assert_eq!(m.textures[slot::DISPLACEMENT].texture, AssetId::NONE);
    assert_eq!(m.displacement_scale, 0.0);
}
