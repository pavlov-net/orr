//! The glTF 2.0 rules the cook follows, a table row or test per clause. `L…` are lines of
//! `specification/2.0/Specification.adoc` on the `main` branch of KhronosGroup/glTF; other
//! citations are the extension's README or an RFC.

#[allow(dead_code)]
#[path = "common/glb.rs"]
mod glb;
#[allow(dead_code)]
#[path = "common/tagged.rs"]
mod tagged;
#[path = "common/tangent_quads.rs"]
mod tangent_quads;

use std::path::{Path, PathBuf};

use glb::{Glb, U16};
use orr_cook::{
    ArchiveSpec, AssetSpec, CookReport, GltfLayout, GltfSettings, Manifest, NodeSelect, Source,
};
use orr_math::{Affine3, Mat4, Vec3};
use orr_pack::clip::Clip;
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{self, MeshHeader, MeshInfo, PageView};
use orr_pack::skeleton::{SkeletonInfo, Trs};
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, AssetId, Class, Target};

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("spec_gltf")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cook(path: &Path, settings: GltfSettings) -> orr_cook::Result<(Archive, CookReport)> {
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
    Ok((a, cooked.report))
}

fn cooked(path: &Path) -> (Archive, CookReport) {
    cook(path, GltfSettings::default()).unwrap()
}

fn meshes_layout() -> GltfSettings {
    GltfSettings {
        layout: GltfLayout::Meshes,
        ..GltfSettings::default()
    }
}

fn read(a: &Archive, b: &orr_pack::Blob) -> Vec<u8> {
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

fn material(a: &Archive, name: &str) -> Material {
    let e = a.find(AssetId::from_name(name)).unwrap();
    Material::from_bytes(&read(a, &a.blobs(e)[0])).unwrap()
}

fn material_of(a: &Archive) -> Material {
    material(a, "g/material/0")
}

fn is_bound(m: &Material, s: usize) -> bool {
    m.textures[s].texture != AssetId::NONE
}

fn skeleton(a: &Archive) -> SkeletonInfo {
    skeleton_of(a, "g")
}

/// The skeleton of model `model`.
fn skeleton_of(a: &Archive, model: &str) -> SkeletonInfo {
    let name = format!("{model}/skeleton");
    let e = a
        .find(AssetId::from_name(&name))
        .unwrap_or_else(|| panic!("no {name}"));
    SkeletonInfo::parse(&read(a, &a.blobs(e)[0])).unwrap()
}

/// Clip `name` of model `model`.
fn clip_of(a: &Archive, model: &str, name: &str) -> Clip {
    let name = format!("{model}/clip/{name}");
    let e = a
        .find(AssetId::from_name(&name))
        .unwrap_or_else(|| panic!("no {name}"));
    let blobs = a.blobs(e);
    Clip::parse(&read(a, &blobs[0]), &read(a, &blobs[1])).unwrap()
}

/// The pose of `clip` `t` seconds in, as skinning matrices of `sk`.
fn posed_at(sk: &SkeletonInfo, clip: &Clip, t: f32) -> Vec<Affine3> {
    let mut pose = sk.rest.clone();
    clip.sample(t, &mut pose);
    palette(sk, &pose)
}

/// The mesh of `name` as its header and each triangle's corners with their cluster's part.
fn part_corners(a: &Archive, name: &str) -> (MeshInfo, Vec<(u16, Vec3)>) {
    let e = a.find(AssetId::from_name(name)).unwrap();
    let info = MeshInfo::parse(&read(a, &a.blobs(e)[0])).unwrap();
    let tris = per_triangle(a, name, |view, c, t| {
        t.map(|v| (c.part, Vec3::from(view.position(c, v))))
    });
    (info, tris.into_iter().flatten().collect())
}

/// What `f` makes of every triangle of mesh entry `name`.
fn per_triangle<T>(
    a: &Archive,
    name: &str,
    f: impl Fn(&PageView<'_>, &mesh::Cluster, [u32; 3]) -> T,
) -> Vec<T> {
    let e = a.find(AssetId::from_name(name)).unwrap();
    let blobs = a.blobs(e);
    let info = MeshInfo::parse(&read(a, &blobs[0])).unwrap();
    let mut out = Vec::new();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        mesh::decode_page(&read(a, &blobs[p.blob as usize]), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            for t in 0..u32::from(c.tri_count) {
                out.push(f(&view, &c, view.triangle(&c, t).map(u32::from)));
            }
        }
    }
    out
}

/// The corners of every triangle of mesh entry `name`, in winding order.
fn triangles(a: &Archive, name: &str) -> Vec<[Vec3; 3]> {
    per_triangle(a, name, |view, c, t| {
        t.map(|v| Vec3::from(view.position(c, v)))
    })
}

fn near(a: Vec3, b: Vec3) -> bool {
    a.abs_diff_eq(b, 1e-3)
}

fn assert_has(got: &[[Vec3; 3]], want: Vec3) {
    assert!(
        got.iter().flatten().any(|&p| near(p, want)),
        "{want} not among {got:?}"
    );
}

/// `n` x `n` PNG bytes.
fn png(n: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    image::RgbaImage::from_fn(n, n, |x, y| image::Rgba(f(x, y)))
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}

fn gradient() -> Vec<u8> {
    png(8, |x, y| [(x * 30) as u8, (y * 30) as u8, 0, 255])
}

/// Mip 0 of a cooked BC4 texture, row by row, and its width.
fn bc4_mip0(a: &Archive, name: &str) -> (u32, Vec<f32>) {
    let e = a.find(AssetId::from_name(name)).unwrap();
    let blobs = a.blobs(e);
    let info = TextureInfo::parse(&read(a, &blobs[0])).unwrap();
    let bytes = info
        .mip_texels(&read(a, &blobs[info.header.mip_blob(0)]), 0, 0)
        .unwrap();
    let [w, h] = info.header.mip_size(0);
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    let mut out = vec![0.0; (w * h) as usize];
    for by in 0..bh {
        for bx in 0..bw {
            let mut px = [0u8; 16];
            bcdec_rs::bc4(&bytes[(by * bw + bx) * 8..][..8], &mut px, 4, false);
            for (i, &v) in px.iter().enumerate() {
                let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
                if x < w as usize && y < h as usize {
                    out[y * w as usize + x] = f32::from(v) / 255.0;
                }
            }
        }
    }
    (w, out)
}

const QUAD: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
];
const QUAD_UV: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];
const TRIANGLE: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

/// A quad with uvs, material 0 being `material`, texture 0 being `image`, and the other
/// top-level members `top`, ending in a comma.
fn textured_quad_json(g: &mut Glb, material: &str, image: &str, top: &str) -> String {
    let pa = g.floats(&QUAD, "VEC3");
    let ua = g.floats(&QUAD_UV, "VEC2");
    let ia = g.indices(&QUAD_INDICES);
    format!(
        r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],{top}
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"TEXCOORD_0":{ua}}},"indices":{ia},"material":0}}]}}],
"materials":[{material}],
"textures":[{{"source":0}}],"images":[{image}]"#
    )
}

fn textured_quad(path: &Path, tex: &[u8], material: &str, top: &str) {
    let mut g = Glb::default();
    let img = g.image(tex);
    let image = format!(r#"{{"bufferView":{img},"mimeType":"image/png"}}"#);
    let rest = textured_quad_json(&mut g, material, &image, top);
    g.write(path, &rest);
}

/// L2516 "Client implementations **MUST** follow specified wrapping modes." and L2514 filters
/// **SHOULD**: a sampler's modes reach the binding of the slot reading its texture, and an
/// unset filter leaves the renderer's own.
#[test]
fn samplers_reach_the_bindings_that_read_them() {
    use orr_pack::material::{filter, wrap};
    let nearest_min = filter::MIN_NEAREST | filter::MIP_NEAREST | filter::ONE_MIP;
    let cases = [
        ("{}", [wrap::REPEAT; 2], 0),
        (
            r#"{"wrapS":33071,"wrapT":33648}"#,
            [wrap::CLAMP, wrap::MIRROR],
            0,
        ),
        (
            r#"{"wrapS":33648,"wrapT":10497}"#,
            [wrap::MIRROR, wrap::REPEAT],
            0,
        ),
        (
            r#"{"magFilter":9728}"#,
            [wrap::REPEAT; 2],
            filter::MAG_NEAREST,
        ),
        (r#"{"magFilter":9729}"#, [wrap::REPEAT; 2], 0),
        (r#"{"minFilter":9728}"#, [wrap::REPEAT; 2], nearest_min),
        (
            r#"{"minFilter":9729}"#,
            [wrap::REPEAT; 2],
            filter::MIP_NEAREST | filter::ONE_MIP,
        ),
        (
            r#"{"minFilter":9984}"#,
            [wrap::REPEAT; 2],
            filter::MIN_NEAREST | filter::MIP_NEAREST,
        ),
        (
            r#"{"minFilter":9985}"#,
            [wrap::REPEAT; 2],
            filter::MIP_NEAREST,
        ),
        (
            r#"{"minFilter":9986}"#,
            [wrap::REPEAT; 2],
            filter::MIN_NEAREST,
        ),
        (r#"{"minFilter":9987}"#, [wrap::REPEAT; 2], 0),
    ];
    let d = dir("samplers");
    for (i, (sampler, modes, bits)) in cases.into_iter().enumerate() {
        let path = d.join(format!("{i}.glb"));
        let mut g = Glb::default();
        let img = g.image(&gradient());
        let image = format!(r#"{{"bufferView":{img},"mimeType":"image/png"}}"#);
        let material = r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}"#;
        let rest = textured_quad_json(&mut g, material, &image, "").replace(
            r#""textures":[{"source":0}]"#,
            &format!(r#""samplers":[{sampler}],"textures":[{{"source":0,"sampler":0}}]"#),
        );
        g.write(&path, &rest);
        let (a, _) = cooked(&path);
        let b = material_of(&a).textures[slot::BASE_COLOR];
        assert_eq!((b.wrap, b.filter), (modes, bits), "{sampler}");
    }
}

/// `KHR_texture_transform`: "Rotate the UVs by this many radians counter-clockwise around the
/// origin. This is equivalent to a similar rotation of the image clockwise", with the uv origin
/// at the upper left. Each row maps uvs to where the transform takes them.
#[test]
fn texture_transforms_turn_the_uvs() {
    let (s, c) = 0.4f32.sin_cos();
    type Case<'a> = (&'a str, &'a str, Vec<([f32; 2], [f32; 2])>);
    let cases: [Case<'_>; 3] = [
        (
            "rotation alone: +u turns toward -v, up the image",
            r#"{"rotation":1.5707963267948966}"#,
            vec![([1.0, 0.0], [0.0, -1.0]), ([0.0, 1.0], [1.0, 0.0])],
        ),
        (
            "the README's example, \"only the lower left quadrant of the source image, rotated \
             clockwise 90\u{b0}\": the unit square lands in u 0..0.5, v 0.5..1",
            r#"{"offset":[0,1],"rotation":1.5707963267948966,"scale":[0.5,0.5]}"#,
            vec![
                ([0.0, 0.0], [0.0, 1.0]),
                ([1.0, 0.0], [0.0, 0.5]),
                ([1.0, 1.0], [0.5, 0.5]),
                ([0.0, 1.0], [0.5, 1.0]),
            ],
        ),
        (
            "scale, then rotation, then offset",
            r#"{"offset":[0.2,-0.1],"rotation":0.4,"scale":[0.8,1.2]}"#,
            vec![
                ([1.0, 0.0], [0.8 * c + 0.2, -0.8 * s - 0.1]),
                ([0.0, 1.0], [1.2 * s + 0.2, 1.2 * c - 0.1]),
            ],
        ),
    ];
    let d = dir("transform");
    for (i, (what, t, uvs)) in cases.iter().enumerate() {
        let path = d.join(format!("{i}.glb"));
        textured_quad(
            &path,
            &gradient(),
            &format!(
                r#"{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0,"extensions":{{"KHR_texture_transform":{t}}}}}}}}}"#
            ),
            r#""extensionsUsed":["KHR_texture_transform"],"#,
        );
        let (a, _) = cooked(&path);
        let t = material(&a, "g/material/0").uv_transforms[slot::BASE_COLOR];
        for (uv, want) in uvs {
            let got = [
                (1.0 + t[0]) * uv[0] + t[1] * uv[1] + t[2],
                t[3] * uv[0] + (1.0 + t[4]) * uv[1] + t[5],
            ];
            assert!(
                (got[0] - want[0]).abs() < 1e-5 && (got[1] - want[1]).abs() < 1e-5,
                "{what}: {uv:?} reaches {got:?}, not {want:?}"
            );
        }
    }
}

/// L1790-1802: a strip's `P_i = { v_i, v_{i+(1+i%2)}, v_{i+(2-i%2)} }` and a fan's
/// `P_i = { v_{i+1}, v_{i+2}, v_0 }`, winding included, in both layouts.
#[test]
fn strips_and_fans_are_the_spec_triangles() {
    const V: [[f32; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    type Case<'a> = (&'a str, u32, Option<&'a [u32]>, f32, &'a [[usize; 3]]);
    let cases: [Case<'_>; 5] = [
        ("strip", 5, None, 1.0, &[[0, 1, 2], [1, 3, 2], [2, 3, 4]]),
        (
            "strip through indices",
            5,
            Some(&[4, 3, 2, 1]),
            1.0,
            &[[4, 3, 2], [3, 1, 2]],
        ),
        ("fan", 6, None, 1.0, &[[1, 2, 0], [2, 3, 0], [3, 4, 0]]),
        (
            "fan through indices",
            6,
            Some(&[3, 1, 0, 2]),
            1.0,
            &[[1, 0, 3], [0, 2, 3]],
        ),
        (
            "a mirroring node turns a strip's winding as it does a list's",
            5,
            None,
            -1.0,
            &[[0, 2, 1], [1, 2, 3], [2, 4, 3]],
        ),
    ];
    // A triangle's rotation with its smallest vertex first.
    let canon = |t: [usize; 3]| {
        let r = (0..3).min_by_key(|&i| t[i]).unwrap();
        [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
    };
    let sorted = |mut t: Vec<[usize; 3]>| {
        t.sort_unstable();
        t
    };
    let d = dir("topology");
    for (i, (what, mode, indices, x_scale, want)) in cases.into_iter().enumerate() {
        let path = d.join(format!("{i}.glb"));
        let mut g = Glb::default();
        let pa = g.floats(&V, "VEC3");
        let ix = indices.map_or(String::new(), |ix| {
            format!(r#","indices":{}"#, g.indices(ix))
        });
        g.write(
            &path,
            &format!(
                r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0,"scale":[{x_scale},1,1]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa}}},"mode":{mode}{ix}}}]}}]"#
            ),
        );
        let vertex = |p: Vec3| {
            V.iter()
                .position(|&v| near(p, Vec3::from(v) * Vec3::new(x_scale, 1.0, 1.0)))
                .unwrap_or_else(|| panic!("{what}: {p} is no vertex"))
        };
        let want = sorted(want.iter().map(|&t| canon(t)).collect());
        let vertices = |a: &Archive, name| {
            let got = triangles(a, name).into_iter();
            sorted(got.map(|t| canon(t.map(vertex))).collect())
        };
        assert_eq!(vertices(&cooked(&path).0, "g/mesh"), want, "{what}");
        if x_scale > 0.0 {
            let (a, _) = cook(&path, meshes_layout()).unwrap();
            assert_eq!(vertices(&a, "g/mesh/0/mesh"), want, "{what}, meshes layout");
        }
    }
}

/// L1752 lists points and lines beside triangles, and L1811 says of a primitive without
/// positions that clients "**SHOULD** skip primitive's rendering". Neither takes down the
/// triangles beside it, in its mesh or another, and each one skipped is reported.
#[test]
fn primitives_without_a_surface_leave_the_rest_to_cook() {
    let d = dir("mixed");
    for (i, (what, attributes, mode)) in [
        ("lines", r#""POSITION":0"#, 1),
        ("points", r#""POSITION":0"#, 0),
        ("no POSITION", r#""TEXCOORD_0":1"#, 4),
    ]
    .into_iter()
    .enumerate()
    {
        let path = d.join(format!("{i}.glb"));
        let mut g = Glb::default();
        g.floats(&QUAD, "VEC3");
        g.floats(&QUAD_UV, "VEC2");
        let ia = g.indices(&QUAD_INDICES);
        let other = format!(r#"{{"attributes":{{{attributes}}},"mode":{mode}}}"#);
        g.write(
            &path,
            &format!(
                r#""scene":0,"scenes":[{{"nodes":[0,1]}}],"nodes":[{{"mesh":0}},{{"mesh":1}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":{ia}}},{other}]}},
{{"primitives":[{other}]}}]"#
            ),
        );
        let (a, report) = cooked(&path);
        assert_eq!(triangles(&a, "g/mesh").len(), 2, "{what}");
        let warnings = &report.assets[0].warnings;
        assert_eq!(warnings.len(), 2, "{what}: {warnings:?}");
        let (a, _) = cook(&path, meshes_layout()).unwrap();
        assert_eq!(
            triangles(&a, "g/mesh/0/mesh").len(),
            2,
            "{what}, meshes layout"
        );
        assert!(a.find(AssetId::from_name("g/mesh/1/mesh")).is_none());
    }

    // A scene with nothing else still cooks, in both layouts, and says what it skipped.
    let path = d.join("only-lines.glb");
    let mut g = Glb::default();
    let pa = g.floats(&QUAD, "VEC3");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa}}},"mode":1}}]}}]"#
        ),
    );
    for settings in [GltfSettings::default(), meshes_layout()] {
        let (_, report) = cook(&path, settings).unwrap();
        assert_eq!(report.assets[0].warnings.len(), 1);
    }
}

/// L1813: "When normals are not specified, the mesh primitive has implicit flat normals. The
/// provided tangents are ignored in this case".
#[test]
fn tangents_without_normals_are_ignored() {
    let path = dir("flat").join("flat.glb");
    let tex = png(16, |x, y| {
        [128 + (x * 4) as u8, 128 + (y * 3) as u8, 255, 255]
    });
    let mut g = Glb::default();
    let pa = g.floats(&QUAD, "VEC3");
    let ua = g.floats(&QUAD_UV, "VEC2");
    let ta = g.floats(&[[0.0f32, 1.0, 0.0, 1.0]; 4], "VEC4");
    let ia = g.indices(&QUAD_INDICES);
    let img = g.image(&tex);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"TEXCOORD_0":{ua},"TANGENT":{ta}}},"indices":{ia},"material":0}}]}}],
"materials":[{{"normalTexture":{{"index":0}}}}],
"textures":[{{"source":0}}],"images":[{{"bufferView":{img},"mimeType":"image/png"}}]"#
        ),
    );
    let (a, _) = cooked(&path);
    let e = a.find(AssetId::from_name("g/mesh")).unwrap();
    let info = MeshInfo::parse(&read(&a, &a.blobs(e)[0])).unwrap();
    assert_eq!(info.header.attributes & MeshHeader::TANGENT, 0);
    let normals = per_triangle(&a, "g/mesh", |view, c, t| view.normal(c, t[0]));
    assert!(
        normals.iter().all(|&n| near(n.into(), Vec3::Z)),
        "{normals:?}"
    );
}

/// A normal texture whose texels differ in x and y, for tangents to turn.
fn textured_normals() -> Vec<u8> {
    png(16, |x, y| {
        [96 + (x * 4) as u8, 160 - (y * 4) as u8, 255, 255]
    })
}

/// L1825, L2832: the bitangent is `cross(normal, tangent) * w`, and the texture's y turns along
/// it. `NormalTangentMirrorTest`'s quads, plain and mirrored, shade the same from their stored
/// tangents as from the triangle's uvs: the cook drops the stored ones.
#[test]
fn uv_tangents_share_the_handedness_of_published_ones() {
    let path = dir("handed").join("quads.glb");
    let (p, n, uv, t, idx) = tangent_quads::arrays();
    let mut g = Glb::default();
    let (pa, na, ta, ua) = (
        g.floats(&p, "VEC3"),
        g.floats(&n, "VEC3"),
        g.floats(&t, "VEC4"),
        g.floats(&uv, "VEC2"),
    );
    let ia = g.indices(&idx);
    let img = g.image(&textured_normals());
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"NORMAL":{na},"TANGENT":{ta},"TEXCOORD_0":{ua}}},"indices":{ia},"material":0}}]}}],
"materials":[{{"normalTexture":{{"index":0}}}}],
"textures":[{{"source":0}}],"images":[{{"bufferView":{img},"mimeType":"image/png"}}]"#
        ),
    );
    let (_, report) = cooked(&path);
    let mesh = report
        .assets
        .iter()
        .flat_map(|a| &a.entries)
        .find_map(|e| e.mesh.clone())
        .unwrap();
    let t = mesh.tangents.unwrap();
    assert!(t.derived && t.angle_p99.unwrap() < 0.1, "{t:?}");
}

/// Cooks keeping tangents in the mesh, for reading them back.
fn keeping_tangents(layout: GltfLayout) -> GltfSettings {
    let mut s = GltfSettings {
        layout,
        ..GltfSettings::default()
    };
    s.mesh.tangent_error_max = None;
    s
}

const NORMAL_MAP: &str = r#"{"normalTexture":{"index":0}}"#;

/// Positions, and normals and tangents where stated.
type Attributes<'a> = (
    &'a [[f32; 3]],
    Option<&'a [[f32; 3]]>,
    Option<&'a [[f32; 4]]>,
);

/// A primitive's JSON over its attributes, after `extra` members.
fn primitive(
    g: &mut Glb,
    (p, n, t): Attributes<'_>,
    uv: &[[f32; 2]],
    idx: &[u32],
    extra: &str,
) -> String {
    let mut attributes = format!(
        r#""POSITION":{},"TEXCOORD_0":{}"#,
        g.floats(p, "VEC3"),
        g.floats(uv, "VEC2")
    );
    if let Some(n) = n {
        attributes += &format!(r#","NORMAL":{}"#, g.floats(n, "VEC3"));
    }
    if let Some(t) = t {
        attributes += &format!(r#","TANGENT":{}"#, g.floats(t, "VEC4"));
    }
    let ia = g.indices(idx);
    format!(r#"{{"attributes":{{{attributes}}},"indices":{ia},"material":0{extra}}}"#)
}

/// One node holding `primitives`, with `node` members and material 0 being `material`.
fn write_mesh(mut g: Glb, path: &Path, primitives: &[String], node: &str, material: &str) {
    let img = g.image(&textured_normals());
    let primitives = primitives.join(",");
    g.write(
        path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0{node}}}],
"meshes":[{{"primitives":[{primitives}]}}],"materials":[{material}],
"textures":[{{"source":0}}],"images":[{{"bufferView":{img},"mimeType":"image/png"}}]"#
        ),
    );
}

/// The position and tangent of every corner of mesh entry `name`.
fn corner_tangents(a: &Archive, name: &str) -> Vec<(Vec3, [f32; 4])> {
    per_triangle(a, name, |view, c, t| {
        t.map(|v| (Vec3::from(view.position(c, v)), view.tangent(c, v).unwrap()))
    })
    .into_iter()
    .flatten()
    .collect()
}

/// Asserts that the corners at `at` have tangent `want`, to the precision of its encoding.
fn assert_tangent(got: &[(Vec3, [f32; 4])], at: Vec3, want: [f32; 4]) {
    let dir = Vec3::from_slice(&want).normalize();
    let mut seen = false;
    for (p, t) in got {
        if near(*p, at) {
            seen = true;
            assert!(
                Vec3::from_slice(t).dot(dir) > 0.9999 && t[3] == want[3],
                "at {at}: {t:?} want {want:?}"
            );
        }
    }
    assert!(seen, "no corner at {at}");
}

fn assert_published(got: &[(Vec3, [f32; 4])]) {
    let (p, _, _, t, _) = tangent_quads::arrays();
    for (p, t) in p.iter().zip(t) {
        assert_tangent(got, Vec3::from(*p), t);
    }
}

/// L1821: "When tangents are not specified and the material has a normal texture, the tangent
/// vectors are calculated using default MikkTSpace algorithms with the vertex positions,
/// normals [...] and texture coordinates associated with the normal texture." The tangents
/// `NormalTangentMirrorTest` publishes are what its geometry computes, plain, mirrored and facing
/// away, in either layout.
#[test]
fn unstated_tangents_are_the_published_mikktspace_ones() {
    let path = dir("mikk").join("quads.glb");
    let (p, n, uv, _, idx) = tangent_quads::arrays();
    let mut g = Glb::default();
    let prim = primitive(&mut g, (&p, Some(&n), None), &uv, &idx, "");
    write_mesh(g, &path, &[prim], "", NORMAL_MAP);
    let (a, _) = cook(&path, keeping_tangents(GltfLayout::Scene)).unwrap();
    assert_published(&corner_tangents(&a, "g/mesh"));
    let (a, _) = cook(&path, keeping_tangents(GltfLayout::Meshes)).unwrap();
    assert_published(&corner_tangents(&a, "g/mesh/0/mesh"));
}

/// L1821, on the texture coordinates "associated with the normal texture".
#[test]
fn tangents_follow_the_normal_textures_uv_set() {
    let path = dir("mikk-set").join("quads.glb");
    let (p, n, uv, _, idx) = tangent_quads::arrays();
    let turned: Vec<[f32; 2]> = uv.iter().map(|&[u, v]| [v, -u]).collect();
    let mut g = Glb::default();
    let set1 = g.floats(&uv, "VEC2");
    let prim = primitive(&mut g, (&p, Some(&n), None), &turned, &idx, "").replace(
        r#""TEXCOORD_0""#,
        &format!(r#""TEXCOORD_1":{set1},"TEXCOORD_0""#),
    );
    write_mesh(
        g,
        &path,
        &[prim],
        "",
        r#"{"normalTexture":{"index":0,"texCoord":1}}"#,
    );
    let (a, _) = cook(&path, keeping_tangents(GltfLayout::Scene)).unwrap();
    assert_published(&corner_tangents(&a, "g/mesh"));
}

/// L1813: without normals "the provided tangents are ignored", and L1821 computes them from
/// normals "specified or generated".
#[test]
fn flat_primitives_compute_tangents_from_generated_normals() {
    let path = dir("mikk-flat").join("quads.glb");
    let (p, _, uv, _, idx) = tangent_quads::arrays();
    let mut g = Glb::default();
    let wrong = vec![[0.0, 0.0, 1.0, -1.0]; p.len()];
    let prim = primitive(&mut g, (&p, None, Some(&wrong)), &uv, &idx, "");
    write_mesh(g, &path, &[prim], "", NORMAL_MAP);
    let (a, _) = cook(&path, keeping_tangents(GltfLayout::Scene)).unwrap();
    assert_published(&corner_tangents(&a, "g/mesh"));
}

/// L1821 and L1660's table are per primitive: a primitive stating `TANGENT` keeps it beside one
/// that doesn't.
#[test]
fn stated_tangents_stay_beside_a_primitive_without_them() {
    let path = dir("mikk-mixed").join("quads.glb");
    let (p, n, uv, t, idx) = tangent_quads::arrays();
    let stated = [0.0, 1.0, 0.0, 1.0];
    let mut g = Glb::default();
    let first = primitive(
        &mut g,
        (&p[..4], Some(&n[..4]), Some(&[stated; 4])),
        &uv[..4],
        &idx[..6],
        "",
    );
    let second = primitive(
        &mut g,
        (&p[4..8], Some(&n[4..8]), None),
        &uv[4..8],
        &idx[..6],
        "",
    );
    write_mesh(g, &path, &[first, second], "", NORMAL_MAP);
    let (a, _) = cook(&path, keeping_tangents(GltfLayout::Scene)).unwrap();
    let got = corner_tangents(&a, "g/mesh");
    assert_tangent(&got, Vec3::from(p[0]), stated);
    assert_tangent(&got, Vec3::from(p[4]), t[4]);
}

/// A prototype can bind a material that reads tangents to any section, so a primitive whose
/// material reads none still has its own beside one that does.
#[test]
fn tangents_reach_a_primitive_whose_material_reads_none() {
    let path = dir("mikk-plain").join("quads.glb");
    let (p, n, uv, t, idx) = tangent_quads::arrays();
    let mut g = Glb::default();
    let quad = |g: &mut Glb, at: std::ops::Range<usize>| {
        let v = (&p[at.clone()], Some(&n[at.clone()]), None);
        primitive(g, v, &uv[at], &idx[..6], "")
    };
    let mapped = quad(&mut g, 0..4);
    let plain = quad(&mut g, 4..8).replace(r#""material":0"#, r#""material":1"#);
    write_mesh(
        g,
        &path,
        &[mapped, plain],
        "",
        &format!("{NORMAL_MAP},{{}}"),
    );
    let (a, _) = cook(&path, keeping_tangents(GltfLayout::Scene)).unwrap();
    assert_tangent(&corner_tangents(&a, "g/mesh"), Vec3::from(p[4]), t[4]);
}

/// L1998 applies a transform after the vertex attributes: tangents come from the primitive's
/// own attributes, and a node's mirror and uneven scale reach them as they reach stated ones.
#[test]
fn tangents_are_computed_before_node_transforms() {
    let path = dir("mikk-node").join("strip.glb");
    let d = dir("mikk-node");
    // A bent strip, its middle column's tangent an average that scale would change.
    let column = |x: f32, z: f32, u: f32, nx: f32| {
        let n = Vec3::new(nx, 0.0, 1.0).normalize().to_array();
        [([x, 0.0, z], n, [u, 0.0]), ([x, 1.0, z], n, [u, 1.0])]
    };
    let v = [
        column(0.0, 0.0, 0.0, -0.6),
        column(1.0, 0.7, 0.4, -0.1),
        column(1.5, 0.3, 1.0, 0.8),
    ]
    .concat();
    let p: Vec<[f32; 3]> = v.iter().map(|c| c.0).collect();
    let n: Vec<[f32; 3]> = v.iter().map(|c| c.1).collect();
    let uv: Vec<[f32; 2]> = v.iter().map(|c| c.2).collect();
    let idx = [0, 2, 1, 1, 2, 3, 2, 4, 3, 3, 4, 5];
    let scale = Vec3::new(-2.0, 1.0, 5.0);
    let write = |path: &Path, node: &str| {
        let mut g = Glb::default();
        let prim = primitive(&mut g, (&p, Some(&n), None), &uv, &idx, "");
        write_mesh(g, path, &[prim], node, NORMAL_MAP);
    };
    let plain_path = d.join("plain.glb");
    write(&plain_path, "");
    write(&path, r#","scale":[-2,1,5]"#);
    let (a, _) = cook(&plain_path, keeping_tangents(GltfLayout::Scene)).unwrap();
    let plain = corner_tangents(&a, "g/mesh");
    let (a, _) = cook(&path, keeping_tangents(GltfLayout::Scene)).unwrap();
    let scaled = corner_tangents(&a, "g/mesh");
    for (at, t) in plain {
        let dir = Vec3::from_slice(&t) * scale;
        assert_tangent(&scaled, at * scale, [dir.x, dir.y, dir.z, -t[3]]);
    }
}

/// L2004: "the tangent vector deltas are also calculated for each morph target using the same
/// algorithms and vertex positions, normals, and texture coordinates with the corresponding
/// deltas applied. The provided tangent deltas are ignored in this case".
#[test]
fn computed_tangents_follow_morphed_positions() {
    let path = dir("mikk-morph").join("quad.glb");
    let mut g = Glb::default();
    let shear: Vec<[f32; 3]> = QUAD.iter().map(|p| [0.0, p[0], 0.0]).collect();
    let shear = g.floats(&shear, "VEC3");
    let ignored = g.floats(&[[0.0f32, 0.0, 1.0]; 4], "VEC3");
    let targets = format!(r#","targets":[{{"POSITION":{shear},"TANGENT":{ignored}}}]"#);
    let prim = primitive(
        &mut g,
        (&QUAD, None, None),
        &QUAD_UV,
        &QUAD_INDICES,
        &targets,
    );
    write_mesh(g, &path, &[prim], r#","weights":[1]"#, NORMAL_MAP);
    let (a, _) = cook(&path, keeping_tangents(GltfLayout::Scene)).unwrap();
    let got = corner_tangents(&a, "g/mesh");
    assert_tangent(&got, Vec3::new(1.0, 1.0, 0.0), [1.0, 1.0, 0.0, -1.0]);
}

/// L1821 at a uv fold: one vertex's corners on either side take tangents of opposite handedness,
/// and L1823 wants one W per triangle.
#[test]
fn tangents_split_a_vertex_where_uvs_fold() {
    let path = dir("mikk-fold").join("fold.glb");
    let p = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.5, 0.0],
        [-1.0, 0.5, 0.0],
    ];
    let uv = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.5], [1.0, 0.5]];
    let mut g = Glb::default();
    let prim = primitive(
        &mut g,
        (&p, Some(&[[0.0, 0.0, 1.0]; 4]), None),
        &uv,
        &[0, 2, 1, 0, 1, 3],
        "",
    );
    write_mesh(g, &path, &[prim], "", NORMAL_MAP);
    let (a, _) = cook(&path, keeping_tangents(GltfLayout::Scene)).unwrap();
    let triangles = per_triangle(&a, "g/mesh", |view, c, t| {
        t.map(|v| (Vec3::from(view.position(c, v)), view.tangent(c, v).unwrap()))
    });
    assert_eq!(triangles.len(), 2);
    for t in triangles {
        let right = t.iter().any(|c| c.0.x > 0.5);
        let want = if right {
            [1.0, 0.0, 0.0, -1.0]
        } else {
            [-1.0, 0.0, 0.0, 1.0]
        };
        for (p, _) in t {
            assert_tangent(&t, p, want);
        }
    }
}

/// A mesh read at plan time for assemblies keeps the uv set its normal map reads, which its
/// sections only learn of later.
#[test]
fn assemblies_keep_the_uv_set_a_material_reads() {
    let path = dir("assembly-uv1").join("quads.glb");
    let (p, n, uv, _, idx) = tangent_quads::arrays();
    let mut g = Glb::default();
    let set1 = g.floats(&uv, "VEC2");
    let zeros = vec![[0.0f32; 2]; uv.len()];
    let prim = primitive(&mut g, (&p, Some(&n), None), &zeros, &idx, "").replace(
        r#""TEXCOORD_0""#,
        &format!(r#""TEXCOORD_1":{set1},"TEXCOORD_0""#),
    );
    write_mesh(
        g,
        &path,
        &[prim],
        "",
        r#"{"normalTexture":{"index":0,"texCoord":1}}"#,
    );
    let mut settings = keeping_tangents(GltfLayout::Meshes);
    settings.mesh.assemblies = Some(orr_cook::AssemblySettings {
        min_triangles: 1000,
    });
    let (a, report) = cook(&path, settings).unwrap();
    assert!(report.assets[0].warnings.is_empty());
    assert_published(&corner_tangents(&a, "g/mesh/0/mesh"));
    let got = per_triangle(&a, "g/mesh/0/mesh", |view, c, t| view.uv(c, t[0], 1));
    assert!(
        got.iter()
            .all(|u| uv.iter().any(|w| (u[0] - w[0]).abs() < 1e-3))
    );
}

/// `path` without its primitives' `TANGENT`, written to `out`.
fn without_tangents(path: &Path, out: &Path) {
    let glb = std::fs::read(path).unwrap();
    let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    let mut json: serde_json::Value = serde_json::from_slice(&glb[20..20 + len]).unwrap();
    for mesh in json["meshes"].as_array_mut().unwrap() {
        for prim in mesh["primitives"].as_array_mut().unwrap() {
            prim["attributes"]
                .as_object_mut()
                .unwrap()
                .remove("TANGENT");
        }
    }
    let mut text = serde_json::to_vec(&json).unwrap();
    text.resize(text.len().next_multiple_of(4), b' ');
    let bin = &glb[20 + len..];
    let mut bytes = glb[..8].to_vec();
    bytes.extend((20 + text.len() + bin.len()).to_le_bytes()[..4].iter());
    bytes.extend((text.len() as u32).to_le_bytes());
    bytes.extend(b"JSON");
    bytes.extend(text);
    bytes.extend(bin);
    std::fs::write(out, bytes).unwrap();
}

/// L1821 over a whole published model: without its `TANGENT`, `NormalTangentMirrorTest`'s bowls
/// and quads compute the tangents the file stores.
#[test]
#[ignore = "needs the Khronos sample: ORR_VALIDATION_DATA=1 tools/fetch-slang.sh"]
fn a_published_model_computes_the_tangents_it_stores() {
    let rel = "gltf/NormalTangentMirrorTest.glb";
    let path = match std::env::var_os("ORR_TOOLS").filter(|p| !p.is_empty()) {
        Some(t) => PathBuf::from(t).join(rel),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .map(|d| d.join(".tools").join(rel))
            .find(|p| p.exists())
            .expect("sample not fetched: ORR_VALIDATION_DATA=1 tools/fetch-slang.sh"),
    };
    let stripped = dir("mikk-published").join("stripped.glb");
    without_tangents(&path, &stripped);
    let settings = || keeping_tangents(GltfLayout::Scene);
    let (stated, _) = cook(&path, settings()).unwrap();
    let (computed, _) = cook(&stripped, settings()).unwrap();
    let corners = |a: &Archive| {
        let mut c = per_triangle(a, "g/mesh", |view, cl, t| {
            t.map(|v| {
                let [u, w] = view.uv(cl, v, 0).map(|x| (x * 4096.0).round() as i32);
                let [x, y, z] = view.position(cl, v).map(|x| (x * 4096.0).round() as i32);
                let t = view.tangent(cl, v).unwrap();
                ([x, y, z, u, w, t[3] as i32], t)
            })
        })
        .concat();
        c.sort_unstable_by_key(|c| c.0);
        c.dedup_by_key(|c| c.0);
        c
    };
    let (stated, computed) = (corners(&stated), corners(&computed));
    assert_eq!(stated.len(), computed.len());
    for (a, b) in stated.iter().zip(&computed) {
        assert_eq!(a.0, b.0);
        let dot = Vec3::from_slice(&a.1).dot(Vec3::from_slice(&b.1));
        assert!(dot > 0.9999 && a.1[3] == b.1[3], "{a:?} {b:?}");
    }
}

/// `KHR_materials_anisotropy`: "When the mesh primitive does not have `NORMAL` or `TANGENT`
/// vectors, they are computed as defined in the glTF 2.0 specification."
#[test]
fn anisotropy_computes_tangents_without_a_normal_texture() {
    let path = dir("mikk-aniso").join("quads.glb");
    let (p, n, uv, _, idx) = tangent_quads::arrays();
    let mut g = Glb::default();
    let prim = primitive(&mut g, (&p, Some(&n), None), &uv, &idx, "");
    write_mesh(
        g,
        &path,
        &[prim],
        "",
        r#"{"extensions":{"KHR_materials_anisotropy":{"anisotropyStrength":0.5}}}"#,
    );
    let (a, _) = cooked(&path);
    assert_published(&corner_tangents(&a, "g/mesh"));
}

/// L2899: "If the alpha value is greater than or equal to the alpha cutoff value, the material
/// is rendered as fully opaque, otherwise, it is rendered as fully transparent", per texel and
/// with the base color factor's alpha. Texels within one UNORM8 step of each other fold into
/// the factor only when they fall on one side of the cutoff.
#[test]
fn the_alpha_test_holds_per_texel() {
    type Case<'a> = (&'a str, [u8; 2], f32, f32, bool);
    let cases: [Case<'_>; 6] = [
        ("127 and 128 under 0.5", [127, 128], 1.0, 0.5, true),
        ("127 and 128 under 0.501", [127, 128], 1.0, 0.501, true),
        (
            "a texel equal to the cutoff passes",
            [127, 128],
            1.0,
            128.0 / 255.0,
            true,
        ),
        ("the factor counts", [254, 255], 0.5, 0.5, true),
        ("both texels pass", [200, 201], 1.0, 0.5, false),
        ("both texels fail", [100, 101], 1.0, 0.5, false),
    ];
    let d = dir("alpha");
    for (i, (what, alphas, factor, cutoff, texture)) in cases.into_iter().enumerate() {
        let alpha = |x: u32, y: u32| alphas[((x + y) % 2) as usize];
        let tex = png(8, |x, y| [200, 180, 40, alpha(x, y)]);
        let path = d.join(format!("{i}.glb"));
        textured_quad(
            &path,
            &tex,
            &format!(
                r#"{{"alphaMode":"MASK","alphaCutoff":{cutoff},"pbrMetallicRoughness":{{"baseColorFactor":[1,1,1,{factor}],"baseColorTexture":{{"index":0}}}}}}"#
            ),
            "",
        );
        let (a, _) = cooked(&path);
        let m = material(&a, "g/material/0");
        let passes = |x, y| f32::from(alpha(x, y)) / 255.0 * factor >= cutoff;
        assert_eq!(is_bound(&m, slot::OPACITY), texture, "{what}");
        if !texture {
            assert_eq!(m.geometry_opacity >= cutoff, passes(0, 0), "{what}");
            continue;
        }
        let (w, mip) = bc4_mip0(&a, "g/texture/0/opacity");
        for (i, t) in mip.iter().enumerate() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            assert_eq!(
                t * m.geometry_opacity >= cutoff,
                passes(x, y),
                "{what}: texel {x},{y} is {t}"
            );
        }
    }
}

/// KHR_materials_specular: "The specular color factor is allowed to be set to values greater than
/// [1, 1, 1]", and clearcoat's `fresnel_coat` is a mix: the model says so, darkening nothing.
#[test]
fn specular_and_coat_keep_the_gltf_model() {
    let path = dir("specular-model").join("specular.glb");
    textured_quad(
        &path,
        &gradient(),
        r#"{"extensions":{"KHR_materials_specular":{"specularFactor":0.5,"specularColorFactor":[4.0,2.0,0.5]},
            "KHR_materials_clearcoat":{"clearcoatFactor":1.0}}}"#,
        "",
    );
    let m = material_of(&cooked(&path).0);
    assert_eq!(m.model, Material::MODEL_GLTF);
    assert_eq!(m.specular_weight, 0.5);
    assert_eq!(m.specular_color, [4.0, 2.0, 0.5]);
    assert_eq!((m.coat_ior, m.coat_darkening), (1.5, 0.0));
}

/// Each extension texture reaches a slot of its own, on the channel its extension states.
/// `thicknessTexture` stays unread: "Ray-tracers should ignore the thickness texture".
#[test]
fn extension_textures_reach_their_slots() {
    let path = dir("extension-textures").join("all.glb");
    let image = png(8, |x, y| {
        [
            (x * 30) as u8,
            (y * 30) as u8,
            ((x + y) * 15) as u8,
            255 - (x * 20) as u8,
        ]
    });
    textured_quad(
        &path,
        &image,
        r#"{"extensions":{
            "KHR_materials_specular":{"specularColorTexture":{"index":0}},
            "KHR_materials_clearcoat":{"clearcoatFactor":1.0,"clearcoatRoughnessFactor":0.5,
              "clearcoatRoughnessTexture":{"index":0},"clearcoatNormalTexture":{"index":0,"scale":0.5}},
            "KHR_materials_sheen":{"sheenColorFactor":[1,1,1],"sheenRoughnessFactor":1.0,
              "sheenColorTexture":{"index":0},"sheenRoughnessTexture":{"index":0}},
            "KHR_materials_anisotropy":{"anisotropyStrength":1.0,"anisotropyTexture":{"index":0}},
            "KHR_materials_iridescence":{"iridescenceFactor":1.0,"iridescenceTexture":{"index":0},
              "iridescenceThicknessMinimum":200.0,"iridescenceThicknessMaximum":100.0,
              "iridescenceThicknessTexture":{"index":0}},
            "KHR_materials_diffuse_transmission":{"diffuseTransmissionFactor":1.0,
              "diffuseTransmissionTexture":{"index":0}}}}"#,
        "",
    );
    let (a, _) = cooked(&path);
    let m = material_of(&a);
    for s in [
        slot::SPECULAR_COLOR,
        slot::COAT_ROUGHNESS,
        slot::COAT_NORMAL,
        slot::FUZZ_COLOR,
        slot::FUZZ_ROUGHNESS,
        slot::ANISOTROPY,
        slot::TANGENT,
        slot::THIN_FILM_WEIGHT,
        slot::THIN_FILM_THICKNESS,
        slot::SUBSURFACE_WEIGHT,
    ] {
        assert!(is_bound(&m, s), "slot {s}");
    }
    // `mix(minimum, maximum, g)` as a share of the larger: green runs 0 to 210 down the image.
    assert_eq!(m.thin_film_thickness, 200.0);
    let name = "g/texture/0/thin_film_thickness";
    let (w, thickness) = bc4_mip0(&a, name);
    let share = |g: f32| 1.0 - 0.5 * g;
    assert!((thickness[0] - share(0.0)).abs() < 0.02, "{}", thickness[0]);
    let last = thickness[(7 * w) as usize];
    assert!((last - share(210.0 / 255.0)).abs() < 0.02, "{last}");
}

/// KHR_materials_volume: thickness 0 "the material is thin-walled", so attenuation has no volume
/// to act in; KHR_materials_transmission: `baseColor` tints.
#[test]
fn a_thin_volume_attenuates_nothing() {
    let path = dir("thin-volume").join("thin.glb");
    textured_quad(
        &path,
        &gradient(),
        r#"{"extensions":{"KHR_materials_transmission":{"transmissionFactor":1.0},
            "KHR_materials_volume":{"attenuationColor":[0.5,0.5,0.5],"attenuationDistance":1.0}}}"#,
        "",
    );
    let m = material_of(&cooked(&path).0);
    assert_eq!(m.thin_walled, 1);
    assert_eq!(
        (m.transmission_color, m.transmission_depth),
        ([1.0; 3], 0.0)
    );
}

/// L2666 "used as an extra multiplier for the base color value", with the example below it for
/// alpha, and a primitive beside it without `COLOR_0` multiplying by nothing.
#[test]
fn color_0_multiplies_the_base_color() {
    let path = dir("color-0").join("colors.glb");
    let mut g = Glb::default();
    let pa = g.floats(&QUAD, "VEC3");
    let colors: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.5],
        [0.25, 0.5, 0.75, 0.5],
    ];
    let ca = g.floats(&colors, "VEC4");
    let ia = g.indices(&QUAD_INDICES);
    let tri = g.floats(&TRIANGLE, "VEC3");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"COLOR_0":{ca}}},"indices":{ia},"material":0}},
{{"attributes":{{"POSITION":{tri}}},"material":0}}]}}],
"materials":[{{"alphaMode":"MASK"}}]"#
        ),
    );
    let a = cooked(&path).0;
    let m = material_of(&a);
    assert_eq!(m.textures[slot::BASE_COLOR].vertex, 1);
    assert_eq!(m.textures[slot::OPACITY].vertex, 4);
    let corners = per_triangle(&a, "g/mesh", |view, c, t| {
        t.map(|v| (view.position(c, v), view.color(c, v)))
    });
    let corners: Vec<_> = corners.into_iter().flatten().collect();
    let at = |i: usize| {
        let want = Vec3::from(QUAD[i]);
        corners
            .iter()
            .find(|c| near(Vec3::from(c.0), want) && c.1[3] == 0.5)
            .expect("a corner of the quad")
            .1
    };
    assert_eq!(at(0), colors[0]);
    assert_eq!(at(3), colors[3]);
    assert!(corners.iter().any(|c| c.1 == [1.0; 4]), "no white corner");
}

/// L1844 a target maps any attribute to its deltas, L1951 `TEXCOORD_n` and `COLOR_n` among them,
/// and L2008 "clamp all `COLOR_0` attribute components to the [0, 1] range after applying morph
/// deltas". A `VEC3` color's deltas leave its implied alpha.
#[test]
fn morph_targets_move_colors_and_texture_coordinates() {
    let path = dir("morph-color").join("morph.glb");
    let mut g = Glb::default();
    let pa = g.floats(&QUAD, "VEC3");
    let colors = [[0.5, 0.25, 0.0]; 4];
    let ca = g.floats(&colors, "VEC3");
    let deltas = [[0.25, -0.5, 0.5]; 4];
    let da = g.floats(&deltas, "VEC3");
    let uvs = [[0.0, 0.0]; 4];
    let ua = g.floats(&uvs, "VEC2");
    let moves = [[0.5, 0.25]; 4];
    let ma = g.floats(&moves, "VEC2");
    let ia = g.indices(&QUAD_INDICES);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"weights":[1.0,0.5],"primitives":[{{"attributes":{{"POSITION":{pa},"COLOR_0":{ca},"TEXCOORD_0":{ua}}},
"indices":{ia},"material":0,"targets":[{{"COLOR_0":{da}}},{{"TEXCOORD_0":{ma}}}]}}]}}],
"materials":[{{}}]"#
        ),
    );
    let a = cooked(&path).0;
    let corners = per_triangle(&a, "g/mesh", |view, c, t| {
        t.map(|v| (view.color(c, v), view.uv(c, v, 0)))
    });
    for (color, uv) in corners.into_iter().flatten() {
        assert_eq!(color, [0.75, 0.0, 0.5, 1.0], "the sum, clamped");
        assert_eq!(uv, [0.25, 0.125]);
    }
}

/// L1926 lists what morph targets move and joints are not among it: moved, this skin's vertices
/// would follow a joint past it, with weights summing to 2.
#[test]
fn morph_targets_leave_joints_and_weights() {
    let path = dir("morph-skin").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    let deltas = g.floats(&[[5.0f32, 0.0, 0.0, 0.0]; 3], "VEC4");
    let more = g.floats(&[[1.0f32, 0.0, 0.0, 0.0]; 3], "VEC4");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1]}}],
"nodes":[{{"name":"hip"}},{{"mesh":0,"skin":0}}],"skins":[{{"joints":[0]}}],
"meshes":[{{"weights":[1.0],"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}},
"targets":[{{"JOINTS_0":{deltas},"WEIGHTS_0":{more}}}]}}]}}]"#
        ),
    );
    cooked(&path);
}

/// A quad displaced 0.5 under a node of `scale`, with vertex `normals` if any.
fn displaced_quad(name: &str, scale: [f32; 3], normals: Option<&[[f32; 3]]>) -> PathBuf {
    let path = dir(name).join("displaced.glb");
    let mut g = Glb::default();
    let pa = g.floats(&QUAD, "VEC3");
    let ua = g.floats(&QUAD_UV, "VEC2");
    let ia = g.indices(&QUAD_INDICES);
    let img = g.image(&png(4, |_, _| [255, 255, 255, 255]));
    let normal = normals.map_or(String::new(), |n| {
        format!(r#","NORMAL":{}"#, g.floats(n, "VEC3"))
    });
    g.write(
        &path,
        &format!(
            r#""extensionsUsed":["KHR_materials_displacement"],"scene":0,"scenes":[{{"nodes":[0]}}],
"nodes":[{{"mesh":0,"scale":{scale:?}}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"TEXCOORD_0":{ua}{normal}}},"indices":{ia},"material":0}}]}}],
"materials":[{{"extensions":{{"KHR_materials_displacement":{{"displacementGeometryTexture":{{"index":0}},"displacementGeometryFactor":0.5}}}}}}],
"textures":[{{"source":0}}],"images":[{{"bufferView":{img},"mimeType":"image/png"}}]"#
        ),
    );
    path
}

fn baking(layout: GltfLayout, max_edge: f32) -> GltfSettings {
    let mut s = GltfSettings {
        layout,
        ..GltfSettings::default()
    };
    s.mesh.displace_static = Some(orr_cook::StaticDisplacement {
        max_edge,
        target_error: 0.001,
    });
    s
}

fn heights(a: &Archive, name: &str) -> Vec<f32> {
    let corners = per_triangle(a, name, |view, c, t| t.map(|v| view.position(c, v)[2]));
    corners.into_iter().flatten().collect()
}

/// KHR_materials_volume's rule for a material's length: "given in the coordinate space of the
/// mesh", so "any transformations applied to the mesh's node will also be applied". A displacement
/// of 0.5 under a z scale of 3 moves the mesh 1.5, whichever layout places it, and under a z scale
/// of 0 stays in the plane the node flattens the mesh to.
#[test]
fn a_node_scales_a_baked_displacement_in_either_layout() {
    let path = displaced_quad("displaced-node", [2.0, 1.0, 3.0], None);
    let (a, _) = cook(&path, baking(GltfLayout::Scene, 2.0)).unwrap();
    let scene = heights(&a, "g/mesh");
    assert!(scene.iter().all(|z| (z - 1.5).abs() < 1e-3), "{scene:?}");
    let (a, _) = cook(&path, baking(GltfLayout::Meshes, 2.0)).unwrap();
    let own = heights(&a, "g/mesh/0/mesh");
    assert!(own.iter().all(|z| (z - 0.5).abs() < 1e-3), "{own:?}");

    let path = displaced_quad("displaced-flat", [2.0, 1.0, 0.0], None);
    let (a, _) = cook(&path, baking(GltfLayout::Meshes, 2.0)).unwrap();
    let flat = heights(&a, "g/mesh/0/baked/0/mesh");
    assert!(flat.iter().all(|z| z.abs() < 1e-3), "{flat:?}");
}

/// Normals turning 90 degrees across the quad are straight up halfway, where a split puts points:
/// a height of 0.5 has its whole length there, 1.5 under a z scale of 3, against 1.06 at the
/// corners.
#[test]
fn a_split_point_takes_the_whole_length_of_its_normal() {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let normals = [[-s, 0.0, s], [s, 0.0, s], [s, 0.0, s], [-s, 0.0, s]];
    let path = displaced_quad("displaced-curve", [1.0, 1.0, 3.0], Some(&normals));
    let (a, _) = cook(&path, baking(GltfLayout::Scene, 0.2)).unwrap();
    let top = heights(&a, "g/mesh").into_iter().fold(f32::MIN, f32::max);
    assert!((top - 1.5).abs() < 5e-3, "{top}");
}

/// KHR_materials_iridescence: "`iridescenceFactor` is multiplied with the red channel of
/// `iridescenceTexture` to control the overall strength".
#[test]
fn iridescence_keeps_its_strength() {
    let path = dir("iridescence").join("film.glb");
    textured_quad(
        &path,
        &gradient(),
        r#"{"extensions":{"KHR_materials_iridescence":{"iridescenceFactor":0.5}}}"#,
        "",
    );
    let m = material_of(&cooked(&path).0);
    assert_eq!(m.thin_film_weight, 0.5);
    assert_eq!(m.thin_film_thickness, 400.0);
    assert_eq!(m.thin_film_ior, 1.3);
}

/// L2846 "Client implementations that are capable of computing indirect lighting themselves,
/// e.g., via path tracing, **SHOULD** ignore the occlusion texture.": its image isn't opened, so
/// one that doesn't decode cooks, and nothing of it reaches the archive.
#[test]
fn the_occlusion_texture_is_ignored() {
    let path = dir("occlusion").join("occlusion.glb");
    textured_quad(
        &path,
        b"not an image",
        r#"{"occlusionTexture":{"index":0,"strength":0.5}}"#,
        "",
    );
    let (a, _) = cooked(&path);
    let texture = a.entries().iter().any(|e| e.class == Class::Texture as u16);
    assert!(!texture);
}

/// The accessors of a triangle skinned to its skin's joint 0: positions, joints, weights.
fn skinned_triangle(g: &mut Glb) -> (usize, usize, usize) {
    let pa = g.floats(&TRIANGLE, "VEC3");
    let jv = g.view(bytemuck::cast_slice(&[[0u8; 4]; 3]), None);
    let ja = g.accessor(jv, 5121, 3, "VEC4", "");
    let wa = g.floats(&[[1.0f32, 0.0, 0.0, 0.0]; 3], "VEC4");
    (pa, ja, wa)
}

/// Skinning matrices of `sk` at `pose`.
fn palette(sk: &SkeletonInfo, pose: &[Trs]) -> Vec<Affine3> {
    let n = sk.joint_count();
    let (mut model, mut palette) = (vec![Affine3::IDENTITY; n], vec![Affine3::IDENTITY; n]);
    sk.model_pose(pose, &mut model);
    sk.palette(&model, &mut palette);
    palette
}

/// Each triangle corner of `g/mesh` and its one joint.
fn skinned_corners(a: &Archive) -> Vec<(Vec3, usize)> {
    skinned_corners_of(a, "g/mesh")
}

/// Each triangle corner of mesh `name` and its one joint.
fn skinned_corners_of(a: &Archive, name: &str) -> Vec<(Vec3, usize)> {
    let tris = per_triangle(a, name, |view, c, t| {
        t.map(|v| {
            let joint = view.joints(c, v).unwrap()[0];
            (Vec3::from(view.position(c, v)), usize::from(joint))
        })
    });
    tris.into_iter().flatten().collect()
}

/// L2124: `inverseBindMatrices` has a "number of elements that is greater than or equal to the
/// number of elements of the `joints` array", and only those up to that number are "associated
/// with the joints", so the rest need not even be finite.
#[test]
fn inverse_binds_past_the_joints_are_unused() {
    let path = dir("extra-binds").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    let bind = Mat4::from_translation(Vec3::NEG_Y).to_cols_array();
    let binds = [bind, Mat4::IDENTITY.to_cols_array(), [f32::NAN; 16]];
    let bv = g.view(bytemuck::cast_slice(&binds), None);
    let ba = g.accessor(bv, 5126, 3, "MAT4", "");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1]}}],
"nodes":[{{"translation":[0,1,0]}},{{"mesh":0,"skin":0}}],
"skins":[{{"joints":[0],"inverseBindMatrices":{ba}}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}}}}]}}]"#
        ),
    );
    let (a, _) = cooked(&path);
    let bind = skeleton(&a).inverse_binds[0].translation;
    assert!(near(bind, Vec3::NEG_Y), "{bind}");
}

/// L1580, L1584: "Any node **MAY** contain one mesh", and "The mesh **MAY** be skinned", so a
/// static mesh sits beside a skin at its node's transform, and stays there as the joints move:
/// in the whole-scene model, with the skin in a model of its own.
#[test]
fn a_static_mesh_beside_a_skin_keeps_its_place() {
    let path = dir("static").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1,2]}}],
"nodes":[{{"name":"hip"}},{{"mesh":0,"skin":0}},{{"mesh":1,"translation":[5,0,0],"scale":[1,2,1]}}],
"skins":[{{"joints":[0]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}}}}]}},
{{"primitives":[{{"attributes":{{"POSITION":{pa}}}}}]}}]"#
        ),
    );
    let (a, _) = cooked(&path);
    let still = triangles(&a, "g/mesh");
    assert_has(&still, Vec3::new(5.0, 2.0, 0.0));
    assert_eq!(still.len(), 1);
    assert!(a.find(AssetId::from_name("g/skeleton")).is_none());
    let sk = skeleton_of(&a, "g/hip");
    let mut pose = sk.rest.clone();
    pose[0].translation = [1.0, 0.0, 0.0];
    let posed = palette(&sk, &pose);
    for (p, j) in skinned_corners_of(&a, "g/hip/mesh") {
        let got = posed[j].transform_point3(p);
        assert!(near(got, p + Vec3::X), "{p} on joint {j} reaches {got}");
    }
}

/// A scene of a static mesh, and `spin` holding a mesh with `arm`, translated, holding another,
/// each animated; a static wrapper stands above `spin`. L3237: an animation channel targets a
/// node's TRS property. `spin` moves 2 m along x over its second, and `arm` turns.
fn animated_scene(path: &Path) {
    let mut g = Glb::default();
    let pa = g.floats(&TRIANGLE, "VEC3");
    let ta = g.floats(&[[1.0f32], [2.0]], "SCALAR");
    let moved = g.floats(&[[0.0f32, 0.0, 0.0], [2.0, 0.0, 0.0]], "VEC3");
    let half_turn = [0.0f32, 0.0, 0.707_106_77, 0.707_106_77];
    let turn = g.floats(&[[0.0f32, 0.0, 0.0, 1.0], half_turn], "VEC4");
    let early = g.floats(&[[0.0f32], [3.0]], "SCALAR");
    g.write(
        path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1,5]}}],
"nodes":[{{"name":"ground","mesh":0,"translation":[0,-1,0]}},
{{"name":"wrapper","translation":[0,0,10],"scale":[2,2,2],"children":[2]}},
{{"name":"spin","mesh":0,"children":[3]}},
{{"name":"arm","mesh":0,"translation":[0,1,0],"children":[4]}},
{{"name":"child","mesh":0,"translation":[1,0,0]}},
{{"name":"far","mesh":0,"translation":[100,0,0]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa}}}}}]}}],
"animations":[{{"name":"go","samplers":[
{{"input":{ta},"output":{moved}}},{{"input":{ta},"output":{turn}}},{{"input":{early},"output":{moved}}}],
"channels":[{{"sampler":0,"target":{{"node":2,"path":"translation"}}}},
{{"sampler":1,"target":{{"node":3,"path":"rotation"}}}},
{{"sampler":2,"target":{{"node":5,"path":"translation"}}}}]}}]"#
        ),
    );
}

/// L3237: an animated node moves its subtree, skin or no skin. Each rig is a model with a
/// skeleton of rigid parts, the static scene stays out of it, and a static node above folds into
/// the skeleton's root without spending a joint. Part 0 takes no pose, so no triangle is on it.
#[test]
fn an_animated_node_moves_its_subtree_as_rigid_parts() {
    let path = dir("animated").join("scene.glb");
    animated_scene(&path);
    let (a, _) = cooked(&path);
    assert!(a.find(AssetId::from_name("g/skeleton")).is_none());
    assert_eq!(triangles(&a, "g/mesh").len(), 1);

    let sk = skeleton_of(&a, "g/spin");
    assert_eq!(sk.names, ["", "spin", "arm"]);
    let (info, corners) = part_corners(&a, "g/spin/mesh");
    assert_eq!(info.header.attributes & MeshHeader::SKIN, 0);
    assert_eq!(
        info.parts.iter().map(|p| p.joint).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert!(corners.iter().all(|c| c.0 != 0), "a triangle on part 0");

    let clip = clip_of(&a, "g/spin", "go");
    // A joint with no clip channel holds still, and its rest pose places it.
    let rest = palette(&sk, &sk.rest);
    for (part, p) in &corners {
        let at = rest[usize::from(*part)].transform_point3(*p);
        assert!(near(at, *p), "rest pose moved {p} to {at}");
    }
    // At the end `spin` has moved 4 m of scene (2 m under the wrapper's scale of 2), and `arm`
    // has turned 90 degrees about z above `child`, 1 m along x of it, so that lands on y.
    let end = posed_at(&sk, &clip, clip.header.duration);
    let want = |local: Vec3| Vec3::new(4.0, 2.0, 10.0) + local * 2.0;
    let child = TRIANGLE.map(|t| {
        let t = Vec3::from(t);
        want(Vec3::new(-t.y, t.x + 1.0, t.z))
    });
    let reached: Vec<Vec3> = corners
        .iter()
        .map(|(part, p)| end[usize::from(*part)].transform_point3(*p))
        .collect();
    for w in child {
        assert!(
            reached.iter().any(|&r| near(r, w)),
            "{w} not in {reached:?}"
        );
    }
}

/// L3081 leaves timelines to the client, which plays an animation as one: the clips it leaves on
/// separate skeletons share its span, whichever channels each one holds.
#[test]
fn one_animation_keeps_time_across_skeletons() {
    let path = dir("animated-time").join("scene.glb");
    animated_scene(&path);
    let (a, _) = cooked(&path);
    let (spin, far) = (clip_of(&a, "g/spin", "go"), clip_of(&a, "g/far", "go"));
    assert!(
        (spin.header.duration - 3.0).abs() < 1e-5,
        "{:?}",
        spin.header
    );
    assert!((far.header.duration - 3.0).abs() < 1e-5, "{:?}", far.header);
    // `spin` keys from 1 s to 2 s of the animation's 0 to 3: still at 1 s, and home by 2 s.
    let sk = skeleton_of(&a, "g/spin");
    let x = |t: f32| posed_at(&sk, &spin, t)[1].translation.x;
    assert!(
        x(0.9).abs() < 1e-4 && (x(2.1) - 4.0).abs() < 1e-3,
        "{} {}",
        x(0.9),
        x(2.1)
    );
}

/// A chain of `count` nodes with one mesh each, every one moved by an animation, 1 m along x by
/// its end on top of what the nodes above it add. `skinned` puts each vertex on one joint of a
/// skin instead.
fn long_chain(path: &Path, count: usize, skinned: bool) {
    let mut g = Glb::default();
    let positions: Vec<[f32; 3]> = (0..count)
        .flat_map(|i| TRIANGLE.map(|p| [p[0], p[1] + i as f32, p[2]]))
        .collect();
    let joints: Vec<[u16; 4]> = (0..3 * count as u16).map(|v| [v / 3, 0, 0, 0]).collect();
    let pa = g.floats(&positions, "VEC3");
    let tri = g.floats(&TRIANGLE, "VEC3");
    let jv = g.view(bytemuck::cast_slice(&joints), None);
    let ja = g.accessor(jv, 5123, joints.len(), "VEC4", "");
    let wa = g.floats(&vec![[1.0f32, 0.0, 0.0, 0.0]; joints.len()], "VEC4");
    let ta = g.floats(&[[0.0f32], [1.0]], "SCALAR");
    let va = g.floats(&[[0.0f32, 1.0, 0.0], [1.0, 1.0, 0.0]], "VEC3");
    let channels: Vec<String> = (0..count)
        .map(|i| format!(r#"{{"sampler":0,"target":{{"node":{i},"path":"translation"}}}}"#))
        .collect();
    let chain: Vec<String> = (0..count)
        .map(|i| {
            let mesh = if skinned { "" } else { r#""mesh":0,"# };
            let next = if i + 1 < count {
                format!(r#","children":[{}]"#, i + 1)
            } else {
                String::new()
            };
            format!(r#"{{"name":"n{i}",{mesh}"translation":[0,1,0]{next}}}"#)
        })
        .collect();
    let list: Vec<String> = (0..count).map(|i| i.to_string()).collect();
    let (extra, roots, attributes) = if skinned {
        (
            r#",{"mesh":0,"skin":0}"#.to_string(),
            format!("0,{count}"),
            format!(r#""POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}"#),
        )
    } else {
        (
            String::new(),
            "0".to_string(),
            format!(r#""POSITION":{tri}"#),
        )
    };
    g.write(
        path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[{roots}]}}],
"nodes":[{}{extra}],
"skins":[{{"joints":[{}]}}],
"meshes":[{{"primitives":[{{"attributes":{{{attributes}}}}}]}}],
"animations":[{{"name":"go","samplers":[{{"input":{ta},"output":{va}}}],
"channels":[{}]}}]"#,
            chain.join(","),
            list.join(","),
            channels.join(",")
        ),
    );
}

/// Vertices name joints in a u8, parts do not, so a rigid skeleton has no skin's bound: neither
/// 256 joints with the still joint, nor a chain of 300 animated nodes, past what a skin holds.
#[test]
fn rigid_skeletons_pass_the_joints_a_skin_can_name() {
    let d = dir("long-chain");
    let settings = GltfSettings {
        parts_from_skin: true,
        ..GltfSettings::default()
    };
    let path = d.join("skin.glb");
    long_chain(&path, 256, true);
    let (a, _) = cook(&path, settings).unwrap();
    assert_eq!(skeleton(&a).joint_count(), 257);
    let (info, corners) = part_corners(&a, "g/mesh");
    assert_eq!(info.parts.len(), 257);
    assert!(corners.iter().any(|c| c.0 == 256));
    assert!(corners.iter().all(|c| c.0 != 0));

    let path = d.join("nodes.glb");
    long_chain(&path, 300, false);
    let (a, _) = cooked(&path);
    let sk = skeleton(&a);
    assert_eq!(sk.joint_count(), 301);
    let (_, corners) = part_corners(&a, "g/mesh");
    assert!(corners.iter().any(|c| c.0 == 300));
    let end = posed_at(&sk, &clip_of(&a, "g", "go"), 1.0);
    let (part, p) = corners.iter().find(|c| c.0 == 300).unwrap();
    let got = end[usize::from(*part)].transform_point3(*p);
    assert!(near(got, *p + Vec3::X * 300.0), "{p} reaches {got}");
}

/// The selection frames a model, whatever moves in it: one skeleton in the node's space holds
/// its skin, whose joints may stand outside the node, and a mesh there with no joint above it
/// keeps its place. L2153: a skinned mesh's node transform is ignored.
#[test]
fn a_selected_model_holds_a_skeleton_of_its_own() {
    let path = dir("selected").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1,4]}}],
"nodes":[{{"name":"hip","translation":[1,0,0]}},
{{"name":"body","translation":[0,0,5],"children":[2,3]}},
{{"mesh":0,"skin":0}},{{"mesh":1,"translation":[0,3,0]}},
{{"name":"other","mesh":1}}],
"skins":[{{"joints":[0]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}}}}]}},
{{"primitives":[{{"attributes":{{"POSITION":{pa}}}}}]}}]"#
        ),
    );
    let settings = GltfSettings {
        models: vec![
            NodeSelect {
                node: "body".into(),
            },
            NodeSelect {
                node: "other".into(),
            },
        ],
        ..GltfSettings::default()
    };
    let (a, _) = cook(&path, settings).unwrap();
    assert!(a.find(AssetId::from_name("g/other/skeleton")).is_none());
    let sk = skeleton_of(&a, "g/body");
    assert_eq!(sk.names, ["", "hip"]);
    let rest = palette(&sk, &sk.rest);
    let corners = skinned_corners_of(&a, "g/body/mesh");
    let still = |p: Vec3| p.y >= 3.0 - 1e-4;
    for (p, j) in corners {
        let got = rest[j].transform_point3(p);
        // The hip stands at (1, 0, -5) in the body's space, and bind space is the file's.
        let want = if still(p) {
            p
        } else {
            p + Vec3::new(1.0, 0.0, -5.0)
        };
        assert!(near(got, want), "{p} on joint {j} reaches {got}");
    }
}

/// A selected node may be a joint of the skin it holds. The model is in that node's space, so the
/// joint rests at the model's origin, and the root takes the node's place back out.
#[test]
fn a_selected_node_can_be_a_joint_of_its_skin() {
    let path = dir("selected-joint").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],
"nodes":[{{"name":"body","translation":[0,0,5],"children":[1,2]}},
{{"name":"hip","translation":[1,0,0]}},{{"mesh":0,"skin":0}}],
"skins":[{{"joints":[0,1]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}}}}]}}]"#
        ),
    );
    let settings = GltfSettings {
        models: vec![NodeSelect {
            node: "body".into(),
        }],
        ..GltfSettings::default()
    };
    let (a, _) = cook(&path, settings).unwrap();
    let sk = skeleton_of(&a, "g/body");
    assert_eq!(sk.names, ["body", "hip"]);
    let rest = palette(&sk, &sk.rest);
    for (p, j) in skinned_corners_of(&a, "g/body/mesh") {
        let got = rest[j].transform_point3(p);
        assert!(near(got, p), "{p} on joint {j} reaches {got}");
    }
}

/// A skin with no joint in the scene has nothing to pose it: an error, not a mesh left at its node.
#[test]
fn a_skin_of_joints_out_of_the_scene_does_not_cook() {
    let path = dir("joints-elsewhere").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[1]}}],
"nodes":[{{"name":"hip"}},{{"mesh":0,"skin":0}}],
"skins":[{{"joints":[0]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}}}}]}}]"#
        ),
    );
    let e = cook(&path, GltfSettings::default()).err().unwrap();
    assert!(e.to_string().contains("not in the scene"), "{e}");
}

/// A rig takes its first joint's name, unless that is the whole scene's mesh entry.
#[test]
fn a_rig_named_mesh_leaves_the_scene_its_mesh() {
    let path = dir("rig-name").join("scene.glb");
    let mut g = Glb::default();
    let pa = g.floats(&TRIANGLE, "VEC3");
    let ta = g.floats(&[[0.0f32], [1.0]], "SCALAR");
    let va = g.floats(&[[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]], "VEC3");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1]}}],
"nodes":[{{"name":"ground","mesh":0}},{{"name":"mesh","mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa}}}}}]}}],
"animations":[{{"name":"go","samplers":[{{"input":{ta},"output":{va}}}],
"channels":[{{"sampler":0,"target":{{"node":1,"path":"translation"}}}}]}}]"#
        ),
    );
    let (a, _) = cooked(&path);
    assert_eq!(triangles(&a, "g/mesh").len(), 1);
    assert_eq!(skeleton_of(&a, "g/node1").names, ["", "mesh"]);
}

/// `layout` on a skinned scene: the model-per-mesh layout places what stands still, and a skin
/// goes to a model of its own there too.
#[test]
fn the_meshes_layout_keeps_what_moves_in_rigs() {
    let path = dir("meshes-skin").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1,2]}}],
"nodes":[{{"name":"hip"}},{{"mesh":0,"skin":0}},{{"mesh":1,"translation":[5,0,0]}}],
"skins":[{{"joints":[0]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}}}}]}},
{{"primitives":[{{"attributes":{{"POSITION":{pa}}}}}]}}]"#
        ),
    );
    let settings = GltfSettings {
        layout: GltfLayout::Meshes,
        ..GltfSettings::default()
    };
    let (a, _) = cook(&path, settings).unwrap();
    assert!(a.find(AssetId::from_name("g/mesh/1")).is_some());
    assert!(a.find(AssetId::from_name("g/mesh/0")).is_none());
    assert_eq!(skeleton_of(&a, "g/hip").names, ["hip"]);
}

/// `EXT_mesh_gpu_instancing`: an object space transform on the node's world transform, "with
/// instance transformation applied first". Under an animated node the copies move with it,
/// baked into its rig. A skinned instanced node is a model placed per instance, posed by one
/// skeleton, beside a skin of the scene; a scale of zero is no placement, and goes to a skeleton
/// of its own while the mesh keeps its bind pose.
#[test]
fn instanced_nodes_move_with_rigs_and_skins() {
    let path = dir("instanced").join("scene.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    let ta = g.floats(&[[0.0f32], [1.0]], "SCALAR");
    let va = g.floats(&[[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]], "VEC3");
    let at = g.floats(&[[0.0f32, 2.0, 0.0], [0.0, 4.0, 0.0]], "VEC3");
    let bt = g.floats(&[[7.0f32, 0.0, 0.0], [9.0, 0.0, 0.0]], "VEC3");
    let bs = g.floats(&[[1.0f32, 1.0, 1.0], [1.0, 0.0, 1.0]], "VEC3");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,2,3]}}],
"extensionsUsed":["EXT_mesh_gpu_instancing"],
"nodes":[{{"name":"mover","children":[1]}},
{{"name":"copies","mesh":1,"extensions":{{"EXT_mesh_gpu_instancing":{{"attributes":{{"TRANSLATION":{at}}}}}}}}},
{{"name":"hip"}},
{{"name":"crowd","mesh":0,"skin":0,"extensions":{{"EXT_mesh_gpu_instancing":{{"attributes":{{"TRANSLATION":{bt},"SCALE":{bs}}}}}}}}}],
"skins":[{{"joints":[2]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}}}}]}},
{{"primitives":[{{"attributes":{{"POSITION":{pa}}}}}]}}],
"animations":[{{"name":"go","samplers":[{{"input":{ta},"output":{va}}}],
"channels":[{{"sampler":0,"target":{{"node":0,"path":"translation"}}}}]}}]"#
        ),
    );
    let (a, _) = cooked(&path);
    let sk = skeleton_of(&a, "g");
    assert_eq!(sk.names, ["", "mover"]);
    let (_, corners) = part_corners(&a, "g/mesh");
    assert_eq!(corners.len(), 6, "a copy of the mesh per instance");
    let end = posed_at(&sk, &clip_of(&a, "g", "go"), 1.0);
    let top = corners
        .iter()
        .map(|(j, p)| end[usize::from(*j)].transform_point3(*p))
        .fold(Vec3::ZERO, Vec3::max);
    assert!(top.x >= 1.99 && top.y >= 4.0, "{top}");

    let crowd = skeleton_of(&a, "g/crowd");
    assert_eq!(crowd.names, ["hip"]);
    assert!(a.find(AssetId::from_name("g/crowd/clip/go")).is_none());
    let flat = skeleton_of(&a, "g/crowd/baked/0");
    assert_eq!(flat.root.matrix3.y_axis.y, 0.0);
    let bind = skinned_corners_of(&a, "g/crowd/baked/0/mesh");
    assert!(bind.iter().any(|c| c.0.y > 0.5), "{bind:?}");
    let pages = a
        .entries()
        .iter()
        .filter(|e| a.name(e).contains("/instances"));
    assert_eq!(pages.count(), 1);
}

/// L2098, L2100: inverse binds are per skin, and L2167 lets the nodes of one mesh "specify
/// different ... skins", so two skins may bind one joint differently. Each keeps its bind, and
/// a clip on the joint moves both.
#[test]
fn skins_sharing_a_joint_keep_their_inverse_binds() {
    let path = dir("shared").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    let binds = [
        Mat4::IDENTITY.to_cols_array(),
        Mat4::from_translation(Vec3::NEG_Z * 3.0).to_cols_array(),
    ];
    let bv = g.view(bytemuck::cast_slice(&binds), None);
    let ba = g.accessor(bv, 5126, 1, "MAT4", "");
    let bb = g.accessor(bv, 5126, 1, "MAT4", r#","byteOffset":64"#);
    let times = g.floats(&[[0.0f32], [1.0]], "SCALAR");
    let moves = g.floats(&[[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]], "VEC3");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1,2]}}],
"nodes":[{{"name":"hip"}},{{"mesh":0,"skin":0}},{{"mesh":0,"skin":1}}],
"skins":[{{"joints":[0],"inverseBindMatrices":{ba}}},{{"joints":[0],"inverseBindMatrices":{bb}}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}}}}]}}],
"animations":[{{"name":"walk","channels":[{{"sampler":0,"target":{{"node":0,"path":"translation"}}}}],
"samplers":[{{"input":{times},"output":{moves}}}]}}]"#
        ),
    );
    let (a, _) = cooked(&path);
    let sk = skeleton(&a);
    let e = a.find(AssetId::from_name("g/clip/walk")).unwrap();
    let blobs = a.blobs(e);
    let clip = Clip::parse(&read(&a, &blobs[0]), &read(&a, &blobs[1])).unwrap();
    let mut pose = sk.rest.clone();
    clip.sample(1.0, &mut pose);
    let (rest, moved) = (palette(&sk, &sk.rest), palette(&sk, &pose));
    let corners = skinned_corners(&a);
    assert_eq!(corners.len(), 6);
    let mut depths: Vec<f32> = corners
        .iter()
        .map(|&(p, j)| rest[j].transform_point3(p).z)
        .collect();
    depths.sort_by(f32::total_cmp);
    assert!(
        (depths[0] + 3.0).abs() < 1e-3 && depths[5].abs() < 1e-3,
        "{depths:?}"
    );
    for (p, j) in corners {
        let (at, to) = (rest[j].transform_point3(p), moved[j].transform_point3(p));
        assert!(near(to, at + Vec3::X), "{p} on joint {j} reaches {to}");
    }
}

/// L578, L629 and RFC 3986 section 2.1: a relative path is percent-decoded, after the
/// recommended steps of L537 and L629 remove stray whitespace and turn reverse solidus into
/// solidus. L554 only asks data URLs for base64 with a **SHOULD**: RFC 2397 has a plain,
/// percent-encoded form too.
#[test]
fn uris_decode_to_the_resource_they_name() {
    let d = dir("uri");
    let fixture = |path: &Path, uri: &str| {
        let mut g = Glb::default();
        let pa = g.floats(&QUAD, "VEC3");
        let ia = g.indices(&QUAD_INDICES);
        g.write_uri(
            path,
            &format!(
                r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa}}},"indices":{ia}}}]}}]"#
            ),
            uri,
        )
    };
    let bin = fixture(&d.join("bytes.gltf"), "");
    let percent: String = bin.iter().map(|b| format!("%{b:02X}")).collect();
    let b64 = {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(&bin)
    };
    std::fs::create_dir_all(d.join("sub dir")).unwrap();
    let cases: [(&str, String, Option<&str>); 8] = [
        ("%20 in a path", "my%20mesh.bin".into(), Some("my mesh.bin")),
        (
            "%C3%A8 in a path",
            "grande_sph%C3%A8re.bin".into(),
            Some("grande_sphère.bin"),
        ),
        (
            "an IRI needs no encoding",
            "grande_sphère.bin".into(),
            Some("grande_sphère.bin"),
        ),
        (
            "a reverse solidus",
            r"sub%20dir\mesh.bin".into(),
            Some("sub dir/mesh.bin"),
        ),
        (
            "stray spaces and a tab",
            " my\tmesh2.bin\n".into(),
            Some("mymesh2.bin"),
        ),
        (
            "a data URL without base64",
            format!("data:application/gltf-buffer,{percent}"),
            None,
        ),
        (
            "a base64 data URL with percent-encoded characters",
            format!(
                "data:application/gltf-buffer;base64,{}",
                b64.replace('A', "%41")
            ),
            None,
        ),
        (
            "the data scheme and base64 in capitals",
            format!("DATA:;BASE64,{b64}"),
            None,
        ),
    ];
    for (i, (what, uri, file)) in cases.into_iter().enumerate() {
        let path = d.join(format!("{i}.gltf"));
        let bin = fixture(&path, &uri);
        if let Some(f) = file {
            std::fs::write(d.join(f), bin).unwrap();
        }
        let (a, _) = cook(&path, GltfSettings::default()).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(triangles(&a, "g/mesh").len(), 2, "{what}");
    }

    // An image's URI resolves the same way.
    std::fs::write(d.join("a b.png"), gradient()).unwrap();
    let path = d.join("image.gltf");
    let mut g = Glb::default();
    let rest = textured_quad_json(
        &mut g,
        r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}"#,
        r#"{"uri":"a%20b.png"}"#,
        "",
    );
    let bin = g.write_uri(&path, &rest, "image.bin");
    std::fs::write(d.join("image.bin"), bin).unwrap();
    let (a, _) = cooked(&path);
    assert!(is_bound(&material(&a, "g/material/0"), slot::BASE_COLOR));
}

/// L1147: "In absence of element sources, accessor elements are sourced from an infinite
/// sequence of zero bytes", and L1149: sparse replacements "happen after resolving the element
/// source"; their indices may be 8, 16 or 32 bits.
#[test]
fn accessors_without_a_source_read_zeros_under_their_sparse_values() {
    let d = dir("accessors");
    let rows: [(&str, u32, Vec<u8>); 3] = [
        ("8-bit indices", 5121, vec![1, 3]),
        (
            "16-bit indices",
            U16,
            bytemuck::cast_slice(&[1u16, 3]).to_vec(),
        ),
        (
            "32-bit indices",
            5125,
            bytemuck::cast_slice(&[1u32, 3]).to_vec(),
        ),
    ];
    for (i, (what, component, index_bytes)) in rows.into_iter().enumerate() {
        let path = d.join(format!("{i}.glb"));
        let mut g = Glb::default();
        let pa = g.floats(&QUAD, "VEC3");
        let ia = g.indices(&QUAD_INDICES);
        let iv = g.view(&index_bytes, None);
        let values = [[0.0f32, 2.0, 0.0], [0.0, 3.0, 0.0]];
        let vv = g.view(bytemuck::cast_slice(&values), None);
        // A morph target's deltas: zeros but for vertices 1 and 3.
        let morph = g.accessor_json(format!(
            r#"{{"componentType":5126,"count":4,"type":"VEC3","sparse":{{"count":2,"indices":{{"bufferView":{iv},"componentType":{component}}},"values":{{"bufferView":{vv}}}}}}}"#
        ));
        let uvs = g.accessor_json(r#"{"componentType":5126,"count":4,"type":"VEC2"}"#.into());
        g.write(
            &path,
            &format!(
                r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"TEXCOORD_0":{uvs}}},"indices":{ia},"targets":[{{"POSITION":{morph}}}]}}],"weights":[1.0]}}]"#
            ),
        );
        let (a, _) = cook(&path, GltfSettings::default()).unwrap_or_else(|e| panic!("{what}: {e}"));
        let got = triangles(&a, "g/mesh");
        assert_has(&got, Vec3::new(0.0, 0.0, 0.0));
        assert_has(&got, Vec3::new(1.0, 2.0, 0.0));
        assert_has(&got, Vec3::new(1.0, 1.0, 0.0));
        assert_has(&got, Vec3::new(0.0, 4.0, 0.0));
        let uv = per_triangle(&a, "g/mesh", |view, c, t| view.uv(c, t[0], 0));
        assert!(uv.iter().all(|&u| u == [0.0, 0.0]), "{what}: {uv:?}");
    }
}

/// L2386 has an image in a buffer view define `mimeType`, but images are recognized by their
/// bytes, so one without it reads anyway. An image with no source has nothing to read: an
/// error, not a panic.
#[test]
fn malformed_images_do_not_panic() {
    const MATERIAL: &str = r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}"#;
    let d = dir("images");
    let path = d.join("no-mime.glb");
    let mut g = Glb::default();
    let img = g.image(&gradient());
    let image = format!(r#"{{"bufferView":{img}}}"#);
    let rest = textured_quad_json(&mut g, MATERIAL, &image, "");
    g.write(&path, &rest);
    let (a, _) = cooked(&path);
    assert!(is_bound(&material(&a, "g/material/0"), slot::BASE_COLOR));

    let path = d.join("no-source.glb");
    let mut g = Glb::default();
    let rest = textured_quad_json(&mut g, MATERIAL, "{}", "");
    g.write(&path, &rest);
    let e = cook(&path, GltfSettings::default()).err().unwrap();
    assert!(e.to_string().contains("image 0"), "{e}");
}

/// L979: "Computing local and global transformation matrices **SHOULD** be done in double
/// precision". 2^24 + 1 has no f32, so a world transform in f32 loses the middle node.
#[test]
fn node_transforms_compose_in_double_precision() {
    let path = dir("double").join("double.glb");
    let mut g = Glb::default();
    let pa = g.floats(&QUAD, "VEC3");
    let ia = g.indices(&QUAD_INDICES);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],
"nodes":[{{"translation":[16777216,0,0],"children":[1]}},{{"translation":[1,0,0],"children":[2]}},
{{"mesh":0,"translation":[-16777216,0,0]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa}}},"indices":{ia}}}]}}]"#
        ),
    );
    let (a, _) = cooked(&path);
    let got = triangles(&a, "g/mesh");
    assert_has(&got, Vec3::new(1.0, 0.0, 0.0));
    assert_has(&got, Vec3::new(2.0, 1.0, 0.0));
}

/// The f32s nearest 0.3 and 0.2 sum to just over 0.5 in f64, but the node is on the grid at the
/// precision glTF stores.
#[test]
fn chained_decimal_translations_meet_the_join_grid() {
    let path = dir("join").join("join.glb");
    let mut g = Glb::default();
    let pa = g.floats(&QUAD, "VEC3");
    let ia = g.indices(&QUAD_INDICES);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],
"nodes":[{{"translation":[0.3,0,0],"children":[1]}},{{"name":"tile","mesh":0,"translation":[0.2,0,0]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa}}},"indices":{ia}}}]}}]"#
        ),
    );
    let settings = GltfSettings {
        models: vec![NodeSelect {
            node: "tile".into(),
        }],
        joined: vec!["tile".into()],
        ..GltfSettings::default()
    };
    let (a, _) = cook(&path, settings).unwrap();
    assert!(a.find(AssetId::from_name("g/tile/mesh")).is_some());
}

/// Only a surface takes a skin: a skinned mesh of lines leaves the meshes layout open, and asks
/// for no skeleton in the scene layout, whose skin's joint is not even in the scene.
#[test]
fn a_skin_without_a_surface_binds_nothing() {
    let path = dir("skinned-lines").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    let ia = g.indices(&QUAD_INDICES);
    let qa = g.floats(&QUAD, "VEC3");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[1,2]}}],
"nodes":[{{"name":"hip"}},{{"mesh":0,"skin":0}},{{"mesh":1}}],
"skins":[{{"joints":[0]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}},"mode":1}}]}},
{{"primitives":[{{"attributes":{{"POSITION":{qa}}},"indices":{ia}}}]}}]"#
        ),
    );
    let (a, _) = cooked(&path);
    assert!(a.find(AssetId::from_name("g/skeleton")).is_none());
    assert_eq!(triangles(&a, "g/mesh").len(), 2);
    let (a, _) = cook(&path, meshes_layout()).unwrap();
    assert_eq!(triangles(&a, "g/mesh/1/mesh").len(), 2);
}

/// A joint's rotation without length reads as no rotation, as it does in the node's transform.
#[test]
fn a_joint_rotation_without_length_is_none() {
    let path = dir("zero-rotation").join("skin.glb");
    let mut g = Glb::default();
    let (pa, ja, wa) = skinned_triangle(&mut g);
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1]}}],
"nodes":[{{"rotation":[0,0,0,0]}},{{"mesh":0,"skin":0}}],
"skins":[{{"joints":[0]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}}}}]}}]"#
        ),
    );
    let (a, _) = cooked(&path);
    assert_eq!(skeleton(&a).rest[0].rotation, [0.0, 0.0, 0.0, 1.0]);
}

/// L2418 "Any colorspace information (such as ICC profiles, intents, gamma values, etc.) found in
/// images **MUST** be ignored. ... The effective color gamut for all images that are used as color
/// textures is ITU-R BT.709." L2441 lists the chunks. A varying and a constant image stating
/// Display P3, linear light, narrow range and a gamma cook as the untagged one does.
#[test]
fn image_color_metadata_is_ignored() {
    let material_json = r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}"#;
    let gamma = tagged::gama(1.0);
    let hints: &[(&[u8; 4], &[u8])] = &[(b"gAMA", &gamma), (b"cICP", &[12, 8, 0, 0])];
    let texture = |a: &Archive| {
        let m = material(a, "g/material/0");
        let e = a.find(m.textures[slot::BASE_COLOR].texture).unwrap();
        let blobs = a.blobs(e);
        (read(a, &blobs[0]), read(a, &blobs[1]))
    };
    let d = dir("image_color_metadata");
    let path = d.join("m.glb");
    let cook_image = |image: &[u8]| {
        textured_quad(&path, image, material_json, "");
        cooked(&path).0
    };
    let tagged = cook_image(&tagged::varying_rgb(hints));
    let plain = cook_image(&tagged::varying_rgb(&[]));
    assert_eq!(texture(&tagged), texture(&plain));

    let tagged = cook_image(&tagged::flat_rgb([128, 64, 32], hints));
    let plain = cook_image(&tagged::flat_rgb([128, 64, 32], &[]));
    assert_eq!(
        material(&tagged, "g/material/0").base_color,
        material(&plain, "g/material/0").base_color
    );
}
