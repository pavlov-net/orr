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
    let e = a.find(AssetId::from_name("g/skeleton")).unwrap();
    SkeletonInfo::parse(&read(a, &a.blobs(e)[0])).unwrap()
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
    let tris = per_triangle(a, "g/mesh", |view, c, t| {
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
/// static mesh sits beside a skin at its node's transform, and stays there as the joints move.
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
    assert_has(&triangles(&a, "g/mesh"), Vec3::new(5.0, 2.0, 0.0));
    let sk = skeleton(&a);
    let mut pose = sk.rest.clone();
    pose[0].translation = [1.0, 0.0, 0.0];
    let posed = palette(&sk, &pose);
    for (p, j) in skinned_corners(&a) {
        let want = if p.x >= 4.0 { p } else { p + Vec3::X };
        let got = posed[j].transform_point3(p);
        assert!(near(got, want), "{p} on joint {j} reaches {got}");
    }
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
