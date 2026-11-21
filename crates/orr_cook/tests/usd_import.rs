//! USD and MaterialX import: a composed stage with a point instancer, a subdivision cage, a
//! skinned mesh and OpenPBR materials; a tree that cooks alike from glTF and USD; MaterialX
//! documents and the nodes they may not use.

#![allow(clippy::print_stdout)] // the set test reports what each stage cooked

#[allow(dead_code)]
#[path = "common/float_ktx2.rs"]
mod float_ktx2;
#[allow(dead_code)]
#[path = "common/glb.rs"]
mod glb;

use std::path::{Path, PathBuf};

use float_ktx2::float_ktx2;
use glb::Glb;
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, LodMode, Manifest, MeshSettings, Source, TextureRules,
    UsdSettings, VariantSelection,
};
use orr_math::{DMat3, DMat4, DQuat, DVec3, DVec4, Quat};
use orr_pack::clip::Clip;
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{MeshInfo, PageView};
use orr_pack::model::ModelInfo;
use orr_pack::names::{Derived, MaterialBase, MaterialName, ModelName};
use orr_pack::skeleton::{SkeletonInfo, Trs};
use orr_pack::{Archive, AssetId, Class, Entry, Target};

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cook_one(source: Source) -> orr_cook::Result<(Archive, orr_cook::CookReport)> {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "u".into(),
                source,
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

fn usd(path: &Path, settings: UsdSettings) -> orr_cook::Result<(Archive, orr_cook::CookReport)> {
    cook_one(Source::Usd {
        path: path.into(),
        settings,
    })
}

fn blob(a: &Archive, name: &str, i: usize) -> Vec<u8> {
    let e = a
        .find(AssetId::from_name(name))
        .unwrap_or_else(|| panic!("no entry {name}"));
    entry_blob(a, e, i)
}

fn entry_blob(a: &Archive, e: &Entry, i: usize) -> Vec<u8> {
    let b = &a.blobs(e)[i];
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

fn material(a: &Archive, name: &str) -> Material {
    Material::from_bytes(&blob(a, name, 0)).unwrap()
}

/// Corner positions and uvs of one triangle.
type Triangle = ([[f32; 3]; 3], [[f32; 2]; 3]);

/// Every triangle of a mesh entry.
fn triangles(a: &Archive, name: &str) -> (MeshInfo, Vec<Triangle>) {
    let info = MeshInfo::parse(&blob(a, name, 0)).unwrap();
    let mut out = Vec::new();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        orr_pack::mesh::decode_page(&blob(a, name, p.blob as usize), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            for t in 0..u32::from(c.tri_count) {
                let tri = view.triangle(&c, t).map(u32::from);
                out.push((
                    tri.map(|v| view.position(&c, v)),
                    tri.map(|v| view.uv(&c, v, 0)),
                ));
            }
        }
    }
    (info, out)
}

/// Triangle positions as bits, corners rotated so the smallest comes first, sorted.
fn triangle_set(a: &Archive, name: &str) -> Vec<[[u32; 3]; 3]> {
    let mut out: Vec<_> = triangles(a, name)
        .1
        .iter()
        .map(|(p, _)| {
            let b = p.map(|v| v.map(f32::to_bits));
            let r = (0..3).min_by_key(|&i| b[i]).unwrap();
            [b[r], b[(r + 1) % 3], b[(r + 2) % 3]]
        })
        .collect();
    out.sort_unstable();
    out
}

fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(w, h, |x, y| image::Rgba(f(x, y)));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

const STAGE: &str = r#"#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 0.01
    startTimeCode = 0
    endTimeCode = 10
    timeCodesPerSecond = 10
)

def Xform "World"
{
    def Mesh "Cage" (
        variants = {
            string size = "small"
        }
        prepend variantSets = "size"
    )
    {
        uniform token subdivisionScheme = "catmullClark"
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </World/Looks/Bark>
        variantSet "size" = {
            "small" {
                point3f[] points = [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)]
            }
            "large" {
                point3f[] points = [(0, 0, 0), (200, 0, 0), (200, 200, 0), (0, 200, 0)]
            }
        }
    }

    def PointInstancer "Forest"
    {
        double3 xformOp:translate = (500, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel prototypes = [</World/Forest/Protos/Tree>]
        int[] protoIndices = [0, 0, 0]
        point3f[] positions = [(0, 0, 0), (100, 200, 0), (-300, 50, 0)]
        quath[] orientations = [(1, 0, 0, 0), (0.7071068, 0, 0, 0.7071068), (0.9238795, 0, 0.3826834, 0)]
        float3[] scales = [(1, 1, 1), (2, 2, 2), (1, 1, 1.5)]
        int64[] ids = [10, 11, 12]
        int64[] invisibleIds = [11]

        def Scope "Protos"
        {
            def Xform "Tree" (
                prepend payload = @tree.usda@</Tree>
            )
            {
            }
        }
    }

    def SkelRoot "Hero"
    {
        def Skeleton "Rig"
        {
            uniform token[] joints = ["Root", "Root/Arm"]
            uniform matrix4d[] bindTransforms = [((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 100, 1))]
            uniform matrix4d[] restTransforms = [((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 100, 1))]

            def SkelAnimation "Wave"
            {
                uniform token[] joints = ["Root/Arm"]
                float3[] translations.timeSamples = {
                    0: [(0, 0, 100)],
                    10: [(0, 0, 100)],
                }
                quatf[] rotations.timeSamples = {
                    0: [(1, 0, 0, 0)],
                    10: [(0.7071068, 0.7071068, 0, 0)],
                }
                half3[] scales.timeSamples = {
                    0: [(1, 1, 1)],
                    10: [(1, 1, 1)],
                }
            }
        }

        def Mesh "Body" (
            prepend apiSchemas = ["SkelBindingAPI"]
        )
        {
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (10, 0, 0), (0, 0, 200)]
            int[] primvars:skel:jointIndices = [0, 0, 1] (
                elementSize = 1
                interpolation = "vertex"
            )
            float[] primvars:skel:jointWeights = [1, 1, 1] (
                elementSize = 1
                interpolation = "vertex"
            )
            rel skel:skeleton = </World/Hero/Rig>
            rel material:binding = </World/Looks/Skin>
        }
    }

    def Scope "Looks"
    {
        def Material "Bark"
        {
            token outputs:mtlx:surface.connect = </World/Looks/Bark/Surface.outputs:out>

            def Shader "Surface"
            {
                uniform token info:id = "ND_open_pbr_surface_surfaceshader"
                color3f inputs:base_color.connect = </World/Looks/Bark/Graph.outputs:color>
                float inputs:specular_roughness = 0.7
                float inputs:thin_film_weight = 1
                float inputs:thin_film_thickness = 0.4
                token outputs:out
            }

            def NodeGraph "Graph"
            {
                color3f outputs:color.connect = </World/Looks/Bark/Graph/Image.outputs:out>

                def Shader "Image"
                {
                    uniform token info:id = "ND_image_color3"
                    asset inputs:file = @bark.png@ (
                        colorSpace = "srgb_texture"
                    )
                    string inputs:uaddressmode = "clamp"
                    float2 inputs:texcoord.connect = </World/Looks/Bark/Graph/Place.outputs:out>
                    color3f outputs:out
                }

                def Shader "Place"
                {
                    uniform token info:id = "ND_place2d_vector2"
                    float2 inputs:texcoord.connect = </World/Looks/Bark/Graph/St.outputs:out>
                    float2 inputs:scale = (2, 2)
                    float2 outputs:out
                }

                def Shader "St"
                {
                    uniform token info:id = "ND_geompropvalue_vector2"
                    string inputs:geomprop = "st"
                    float2 outputs:out
                }
            }
        }

        def Material "Skin"
        {
            token outputs:surface.connect = </World/Looks/Skin/Surface.outputs:surface>

            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.5, 0.25, 0.125)
                float inputs:roughness = 0.4
                token outputs:surface
            }
        }
    }
}
"#;

const TREE: &str = r#"#usda 1.0
(
    defaultPrim = "Tree"
)

def Xform "Tree"
{
    def Mesh "Trunk"
    {
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (10, 0, 0), (0, 0, 300)]
        normal3f[] normals = [(0, -1, 0), (0, -1, 0), (0, -1, 0)] (
            interpolation = "vertex"
        )
    }
}
"#;

/// The scene-space placements (translation, rotation, scale) of the fixture's instancer,
/// from the values openusd reads: `/World/Forest` at stage translation (500, 0, 0), its
/// visible instances' positions, orientations and scales.
fn openusd_placements(stage: &Path) -> Vec<(DVec3, DQuat, DVec3)> {
    use openusd::{gf, usd};
    let stage = usd::Stage::open(stage.to_str().unwrap()).unwrap();
    let prim = stage.prim("/World/Forest").unwrap();
    let get = |name: &str| prim.attribute(name);
    let positions: Vec<gf::Vec3f> = get("positions").get().unwrap().unwrap();
    let orientations: Vec<gf::Quath> = get("orientations").get().unwrap().unwrap();
    let scales: Vec<gf::Vec3f> = get("scales").get().unwrap().unwrap();
    let ids: Vec<i64> = get("ids").get().unwrap().unwrap();
    let invisible: Vec<i64> = get("invisibleIds").get().unwrap().unwrap();
    let r = stage_to_scene();
    // Rotations conjugated by the axis change; the unit scale cancels.
    let axis = DQuat::from_mat4(&DMat4::from_cols(
        DVec4::X,
        DVec4::NEG_Z,
        DVec4::Y,
        DVec4::W,
    ));
    (0..positions.len())
        .filter(|&i| !invisible.contains(&ids[i]))
        .map(|i| {
            let p = positions[i];
            let t = r.transform_point3(
                DVec3::new(500.0, 0.0, 0.0) + DVec3::new(p.x.into(), p.y.into(), p.z.into()),
            );
            let o = orientations[i];
            let q = DQuat::from_xyzw(o.x.to_f64(), o.y.to_f64(), o.z.to_f64(), o.w.to_f64())
                .normalize();
            let s = scales[i];
            let s = DVec3::new(s.x.into(), s.z.into(), s.y.into());
            (t, axis * q * axis.inverse(), s)
        })
        .collect()
}

/// Scene space from the fixture stage's space: Z up to Y up, centimeters to meters.
fn stage_to_scene() -> DMat4 {
    DMat4::from_scale(DVec3::splat(0.01))
        * DMat4::from_cols(DVec4::X, DVec4::NEG_Z, DVec4::Y, DVec4::W)
}

#[test]
fn composed_stage_cooks_instancer_cage_skin_and_openpbr() {
    let d = dir("usd-stage");
    std::fs::write(d.join("stage.usda"), STAGE).unwrap();
    std::fs::write(d.join("tree.usda"), TREE).unwrap();
    std::fs::write(
        d.join("bark.png"),
        png(8, 8, |x, y| [(x * 32) as u8, (y * 32) as u8, 128, 255]),
    )
    .unwrap();
    let settings = UsdSettings {
        variants: vec![VariantSelection {
            prim: "/World/Cage".into(),
            set: "size".into(),
            selection: "large".into(),
        }],
        ..UsdSettings::default()
    };
    let (a, report) = usd(&d.join("stage.usda"), settings).unwrap();

    // The payload and the texture are inputs of the source.
    let inputs = &report.assets[0].inputs;
    for f in ["stage.usda", "tree.usda", "bark.png"] {
        assert!(
            inputs.iter().any(|p| p.ends_with(f)),
            "{f} is not among {inputs:?}"
        );
    }

    // The cage cooks as its hull, at the selected variant's size, in meters and Y up, with v
    // flipped to a top-left origin.
    let (info, tris) = triangles(&a, "u/mesh");
    assert_eq!(tris.len(), 2);
    assert_eq!(info.header.bounds.min, [0.0, 0.0, -2.0]);
    assert_eq!(info.header.bounds.max, [2.0, 0.0, 0.0]);
    let mut uvs: Vec<[u32; 2]> = tris
        .iter()
        .flat_map(|t| t.1)
        .map(|uv| uv.map(|x| (x * 1000.0).round() as u32))
        .collect();
    uvs.sort_unstable();
    uvs.dedup();
    assert_eq!(uvs, [[0, 0], [0, 1000], [1000, 0], [1000, 1000]]);
    let bark = material(
        &a,
        &MaterialName::new(MaterialBase::Path("/World/Looks/Bark".into())).entry("u"),
    );
    // The place2d scale halves st about the bottom-left origin, which is v = 1 here.
    assert_eq!(
        bark.uv_transforms[slot::BASE_COLOR],
        [-0.5, 0.0, 0.0, 0.0, -0.5, 0.5]
    );
    assert_eq!(bark.specular_roughness, 0.7);
    assert_eq!(bark.base_color, [1.0; 3]);
    assert!((bark.thin_film_thickness - 400.0).abs() < 1e-3);
    let bound = bark.textures[slot::BASE_COLOR].texture;
    assert_eq!(
        bound,
        AssetId::from_name(
            &MaterialName::new(MaterialBase::Path("/World/Looks/Bark".into()))
                .texture("u", "base_color")
        )
    );
    let model = ModelInfo::parse(&blob(&a, "u", 0)).unwrap();
    assert_eq!(
        model.materials,
        [AssetId::from_name(
            &MaterialName::new(MaterialBase::Path("/World/Looks/Bark".into())).entry("u")
        )]
    );

    // The prototype is a model of its own; the visible instances are its placements.
    let (tree, _) = triangles(&a, &ModelName::source("Tree").mesh("u"));
    let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-3);
    assert!(
        close(tree.header.bounds.max, [0.1, 3.0, 0.0]),
        "{:?}",
        tree.header.bounds
    );
    let want = openusd_placements(&d.join("stage.usda"));
    let tree = AssetId::from_name(&ModelName::source("Tree").entry("u"));
    let whole = AssetId::from_name(&ModelName::WHOLE.entry("u"));
    let mut placed = Vec::new();
    let mut stage = Vec::new();
    for e in a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
    {
        assert!(a.name(e).starts_with("u/instances/"));
        let (head, bytes) = (entry_blob(&a, e, 0), entry_blob(&a, e, 1));
        let info = orr_pack::instance::InstancePageInfo::parse(&head).unwrap();
        let h = info.header;
        let half_step = DVec3::from(h.step.map(|s| f64::from(s) / 2.0));
        let (cell, records) = orr_pack::instance::decode(&head, &bytes).unwrap();
        for (proto, m) in records {
            let p = m.translation.as_dvec3() + DVec3::from(cell.map(f64::from)) * 256.0;
            let model = info.prototypes[usize::from(proto)].model;
            if model == whole {
                stage.push((p, m.matrix3));
                continue;
            }
            assert_eq!(model, tree);
            let axes = [0, 1, 2].map(|c| m.matrix3.col(c).as_dvec3());
            placed.push((p, axes, half_step));
        }
    }
    // What no instancer places stands once where the stage puts it, at the origin.
    assert_eq!(stage, [(DVec3::ZERO, orr_math::Mat3::IDENTITY)]);
    assert_eq!(placed.len(), 2, "the invisible id is dropped");
    assert_eq!(want.len(), 2);
    for (t, q, s) in want {
        let (p, axes, half_step) = placed
            .iter()
            .min_by(|a, b| a.0.distance(t).total_cmp(&b.0.distance(t)))
            .unwrap();
        let slack = DVec3::splat(1e-6);
        assert!(
            ((*p - t).abs() - *half_step - slack).max_element() <= 0.0,
            "{p} {t}"
        );
        let got = DQuat::from_mat3(&DMat3::from_cols(
            axes[0].normalize(),
            axes[1].normalize(),
            axes[2].normalize(),
        ));
        // The instance encoder's bound: smallest-three codes of 10 bits per component.
        assert!(got.angle_between(q).to_degrees() <= 0.17, "{got} {q}");
        for (axis, want) in axes.iter().zip(s.to_array()) {
            assert!((axis.length() / want - 1.0).abs() < 0.002, "{axis} {s}");
        }
    }

    // The skinned mesh is its SkelRoot's model, in bind space, with the skeleton and clip.
    let hero = ModelInfo::parse(&blob(&a, &ModelName::source("Hero").entry("u"), 0)).unwrap();
    assert_eq!(
        hero.header.skeleton,
        AssetId::from_name(&ModelName::source("Hero").skeleton("u"))
    );
    let (_, body) = triangles(&a, &ModelName::source("Hero").mesh("u"));
    let mut corners: Vec<[i32; 3]> = body
        .iter()
        .flat_map(|t| t.0)
        .map(|p| p.map(|x| (x * 1000.0).round() as i32))
        .collect();
    corners.sort_unstable();
    assert_eq!(corners, [[0, 0, 0], [0, 2000, 0], [100, 0, 0]]);
    let skin = material(
        &a,
        &MaterialName::new(MaterialBase::Path("/World/Looks/Skin".into())).entry("u"),
    );
    assert_eq!(skin.base_color, [0.5, 0.25, 0.125]);
    assert_eq!(skin.specular_roughness, 0.4);
    let skeleton =
        SkeletonInfo::parse(&blob(&a, &ModelName::source("Hero").skeleton("u"), 0)).unwrap();
    assert_eq!(skeleton.names, ["Root", "Arm"]);
    let clip = Clip::parse(
        &blob(&a, &ModelName::source("Hero").clip("u", "Wave"), 0),
        &blob(&a, &ModelName::source("Hero").clip("u", "Wave"), 1),
    )
    .unwrap();
    assert!((clip.header.duration - 1.0).abs() < 1e-6);
    let mut pose = vec![Trs::IDENTITY; 2];
    clip.sample(1.0, &mut pose);
    // The arm turns 90 degrees about stage x, which is scene x, and sits 1 m up scene y.
    let q = Quat::from_array(pose[1].rotation);
    let want = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    assert!(q.angle_between(want).to_degrees() < 0.1, "{q}");
    assert!(
        close(pose[1].translation, [0.0, 0.0, 100.0]),
        "{:?}",
        pose[1].translation
    );
    let mut model = vec![orr_math::Affine3::IDENTITY; 2];
    skeleton.model_pose(&pose, &mut model);
    let root = orr_math::Mat4::from(model[0])
        .to_scale_rotation_translation()
        .1;
    assert!(
        root.angle_between(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .to_degrees()
            < 0.1
    );
}

/// A prototype at the stage root has no parent to frame it, and scene space is that frame:
/// the stage's units and axes must enter once. Z up in centimeters shows them.
#[test]
fn a_prototype_at_the_stage_root_lands_where_the_stage_says() {
    use orr_math::{IVec3, Mat4, Vec3};
    let d = dir("usd-root-prototype");
    let path = d.join("stage.usda");
    std::fs::write(
        &path,
        r#"#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Z"
)

def Mesh "Proto"
{
    double3 xformOp:translate = (100, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (100, 0, 0), (0, 100, 0)]
    uniform token subdivisionScheme = "none"
}

def Xform "World"
{
    double3 xformOp:translate = (0, 200, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def PointInstancer "Copies"
    {
        rel prototypes = [</Proto>]
        int[] protoIndices = [0]
        point3f[] positions = [(0, 0, 300)]
    }
}
"#,
    )
    .unwrap();
    let (a, _) = usd(&path, UsdSettings::default()).unwrap();
    let proto = AssetId::from_name(&ModelName::source("Proto").entry("u"));
    let mut copies = Vec::new();
    for page in a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
    {
        let (head, body) = (entry_blob(&a, page, 0), entry_blob(&a, page, 1));
        let info = orr_pack::instance::InstancePageInfo::parse(&head).unwrap();
        let (cell, records) = orr_pack::instance::decode(&head, &body).unwrap();
        for (p, mut m) in records {
            if info.prototypes[usize::from(p)].model == proto {
                m.translation += IVec3::from(cell).as_vec3() * 256.0;
                copies.push(m);
            }
        }
    }
    let [placed] = copies[..] else {
        panic!("{copies:?}");
    };
    let placed = Mat4::from(placed);
    let corners = triangles(&a, &ModelName::source("Proto").mesh("u")).1[0].0;
    for p in [Vec3::ZERO, Vec3::X * 100.0, Vec3::Y * 100.0] {
        // Stage offsets of the mesh, the instancer's parent and the copy; then meters, Y up.
        let s = (p + Vec3::new(100.0, 200.0, 300.0)) / 100.0;
        let want = Vec3::new(s.x, s.z, -s.y);
        assert!(
            corners
                .iter()
                .any(|&c| placed.transform_point3(c.into()).abs_diff_eq(want, 1e-3)),
            "{want} not reached"
        );
    }
}

const BIND_IDENTITY: &str = "((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))";
const BIND_FLAT: &str = "((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1))";
const BIND_ARM: &str = "((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 0, 1))";

/// Joints `Root` and `Root/Arm`, with `Root` animated; `rest` is the `restTransforms` line.
fn rig(binds: [&str; 2], rest: &str) -> String {
    let [root, arm] = binds;
    format!(
        r#"uniform token[] joints = ["Root", "Root/Arm"]
        uniform matrix4d[] bindTransforms = [{root}, {arm}]
        {rest}

        def SkelAnimation "Slide"
        {{
            uniform token[] joints = ["Root"]
            float3[] translations.timeSamples = {{
                0: [(0, 0, 0)],
                10: [(1, 0, 0)],
            }}
            quatf[] rotations = [(1, 0, 0, 0)]
            half3[] scales = [(1, 1, 1)]
        }}"#
    )
}

const SECOND_JOINT: &str = "[1, 1, 1]";

/// Cooks a triangle whose points take joint indices `indices`, weighted 1: `xform` holds the
/// `SkelRoot`'s transform attributes, `skeleton` the skeleton's and `mesh` more of the mesh's.
fn bound_triangle(
    name: &str,
    xform: &str,
    skeleton: &str,
    indices: &str,
    mesh: &str,
) -> orr_cook::Result<(Archive, orr_cook::CookReport)> {
    let path = dir(name).join("stage.usda");
    std::fs::write(
        &path,
        format!(
            r#"#usda 1.0
(
    metersPerUnit = 1
    timeCodesPerSecond = 10
    upAxis = "Y"
)

def SkelRoot "Hero"
{{
    {xform}

    def Skeleton "Rig"
    {{
        {skeleton}
    }}

    def Mesh "Body" (
        prepend apiSchemas = ["SkelBindingAPI"]
    )
    {{
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 1)]
        int[] primvars:skel:jointIndices = {indices} (
            elementSize = 1
            interpolation = "vertex"
        )
        float[] primvars:skel:jointWeights = [1, 1, 1] (
            elementSize = 1
            interpolation = "vertex"
        )
        rel skel:skeleton = </Hero/Rig>
        {mesh}
    }}
}}
"#
        ),
    )
    .unwrap();
    usd(&path, UsdSettings::default())
}

/// Skinned points land where UsdSkel puts them, `world * joint * inverse bind * point`, with
/// the pseudo-inverse of a bind that has no inverse, and a skeleton world transform that no
/// joint could hold.
#[test]
fn bound_vertices_land_where_the_stage_puts_them() {
    use orr_math::{Affine3, Mat3, Mat4, Vec3};
    let rest =
        |root: &str, arm: &str| format!("uniform matrix4d[] restTransforms = [{root}, {arm}]");
    let authored = rest(BIND_IDENTITY, BIND_ARM);
    let landed = "((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 0), (1, 0, 0, 1))";
    let sheared = "((1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (0, 0, 0, 1))";
    let turned = "((0.70710678, 0.70710678, 0, 0), (-0.70710678, 0.70710678, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))";
    let turned_arm = "((0.70710678, 0.70710678, 0, 0), (-0.70710678, 0.70710678, 0, 0), (0, 0, 1, 0), (-0.70710678, 0.70710678, 0, 1))";
    let scale = |x, y, z| Mat4::from_scale(Vec3::new(x, y, z));
    let up = Mat4::from_translation(Vec3::Y);
    let half = scale(0.5, 0.5, 0.0);
    let moved = r#"double3 xformOp:translate = (0, 0, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]"#;
    let squashed = r#"double3 xformOp:scale = (2, 2, 0)
    uniform token[] xformOpOrder = ["xformOp:scale"]"#;
    let stretched = r#"double3 xformOp:scale = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:scale"]"#;
    let third = Vec3::splat(1.0 / 3.0);
    let sheared_inverse = Mat4::from_mat3(Mat3::from_cols(
        third * Vec3::new(2.0, -1.0, 1.0),
        third * Vec3::new(-1.0, 2.0, 1.0),
        Vec3::ZERO,
    ));
    for (name, binds, rest, xform, want) in [
        (
            "usd-bind-landed",
            [BIND_IDENTITY, landed],
            rest(BIND_IDENTITY, landed),
            "",
            scale(1.0, 1.0, 0.0),
        ),
        (
            "usd-bind-arm",
            [BIND_IDENTITY, BIND_FLAT],
            authored.clone(),
            "",
            up * half,
        ),
        (
            "usd-bind-parent",
            [BIND_FLAT, BIND_ARM],
            String::new(),
            "",
            scale(1.0, 1.0, 0.0),
        ),
        (
            "usd-bind-sheared",
            [BIND_IDENTITY, sheared],
            rest(BIND_IDENTITY, BIND_IDENTITY),
            "",
            sheared_inverse,
        ),
        (
            "usd-bind-moved",
            [BIND_IDENTITY, BIND_FLAT],
            authored.clone(),
            moved,
            Mat4::from_translation(Vec3::Z * 3.0) * up * half,
        ),
        (
            "usd-bind-squashed",
            [BIND_IDENTITY, BIND_ARM],
            authored,
            squashed,
            scale(2.0, 2.0, 0.0),
        ),
        (
            "usd-bind-stretched",
            [turned, turned_arm],
            rest(turned, BIND_ARM),
            stretched,
            scale(1.0, 2.0, 3.0),
        ),
    ] {
        let (a, _) = bound_triangle(name, xform, &rig(binds, &rest), SECOND_JOINT, "").unwrap();
        // One second of linear motion needs nothing past the starting rate.
        let slide = Clip::parse(
            &blob(&a, &ModelName::source("Hero").clip("u", "Slide"), 0),
            &blob(&a, &ModelName::source("Hero").clip("u", "Slide"), 1),
        )
        .unwrap();
        assert_eq!(slide.header.frame_count, 31, "{name}");
        let sk =
            SkeletonInfo::parse(&blob(&a, &ModelName::source("Hero").skeleton("u"), 0)).unwrap();
        let (mut model, mut palette) = (vec![Affine3::IDENTITY; 2], vec![Affine3::IDENTITY; 2]);
        sk.model_pose(&sk.rest, &mut model);
        sk.palette(&model, &mut palette);
        // The mesh and its palettes stand in the skeleton's space, at its origin.
        let posed = triangles(&a, &ModelName::source("Hero").mesh("u")).1[0]
            .0
            .map(|p| palette[1].transform_point3(p.into()) + sk.origin.as_vec3());
        for p in [Vec3::ZERO, Vec3::X, Vec3::new(0.0, 1.0, 1.0)] {
            let w = want.transform_point3(p);
            assert!(
                posed.iter().any(|g| g.abs_diff_eq(w, 1e-3)),
                "{name}: {w} not among {posed:?}"
            );
        }
    }
}

/// A joint's parent is its nearest listed ancestor, an animated scale of zero leaves the
/// authored rotation in place, and an authored mirror stays on its axis.
#[test]
fn a_clip_keeps_a_zero_scale_keys_rotation_on_a_joint_past_a_gap() {
    use orr_math::Vec3;
    let skeleton = format!(
        r#"uniform token[] joints = ["Root", "Root/Gap/Arm"]
        uniform matrix4d[] bindTransforms = [{BIND_IDENTITY}, {BIND_ARM}]
        uniform matrix4d[] restTransforms = [{BIND_IDENTITY}, ((-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 0, 1))]

        def SkelAnimation "Vanish"
        {{
            uniform token[] joints = ["Root/Gap/Arm"]
            float3[] translations = [(0, 1, 0)]
            quatf[] rotations.timeSamples = {{
                0: [(0.7071068, 0.7071068, 0, 0)],
                10: [(0.7071068, 0.7071068, 0, 0)],
            }}
            float3[] scales.timeSamples = {{
                0: [(-1, 1, 1)],
                10: [(0, 0, 0)],
            }}
        }}"#
    );
    let (a, _) = bound_triangle("usd-zero-scale", "", &skeleton, SECOND_JOINT, "").unwrap();
    let sk = SkeletonInfo::parse(&blob(&a, &ModelName::source("Hero").skeleton("u"), 0)).unwrap();
    assert_eq!(sk.parents[1], 0);
    assert_eq!(sk.rest[1].scale, [-1.0, 1.0, 1.0]);
    let clip = Clip::parse(
        &blob(&a, &ModelName::source("Hero").clip("u", "Vanish"), 0),
        &blob(&a, &ModelName::source("Hero").clip("u", "Vanish"), 1),
    )
    .unwrap();
    let want = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let mut pose = vec![Trs::IDENTITY; 2];
    for (t, scale) in [(0.0, [-1.0, 1.0, 1.0]), (1.0, [0.0; 3])] {
        clip.sample(t, &mut pose);
        let q = Quat::from_array(pose[1].rotation);
        assert!(q.angle_between(want).to_degrees() < 0.1, "{t}: {q}");
        assert!(
            Vec3::from(pose[1].scale).abs_diff_eq(scale.into(), 1e-5),
            "{t}: {:?}",
            pose[1].scale
        );
    }
}

/// OpenUSD fails a skin on a joint index outside the skeleton, and reads a
/// `geomBindTransform` that is not a matrix as unset. An xform op of the wrong size and a
/// skeleton with no finite world transform fail.
#[test]
fn bad_skin_attributes_read_as_openusd_reads_them() {
    use orr_math::Vec3;
    let skeleton = rig([BIND_IDENTITY, BIND_ARM], "");
    let e = bound_triangle("usd-negative-joint", "", &skeleton, "[-1, 1, 1]", "")
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("joint index -1"), "{e}");
    for attribute in [
        "double[] primvars:skel:geomBindTransform = [1, 2, 3]",
        "token primvars:skel:geomBindTransform = \"none\"",
    ] {
        let (a, _) =
            bound_triangle("usd-typed-bind", "", &skeleton, SECOND_JOINT, attribute).unwrap();
        let corners = triangles(&a, &ModelName::source("Hero").mesh("u")).1[0].0;
        assert!(
            corners
                .iter()
                .any(|&c| Vec3::from(c).abs_diff_eq(Vec3::X, 1e-3))
        );
    }
    for (xform, want) in [
        (
            r#"double xformOp:translate = 1
    uniform token[] xformOpOrder = ["xformOp:translate"]"#,
            "xform op translate",
        ),
        (
            r#"double3 xformOp:scale = (0, 1, 1)
    uniform token[] xformOpOrder = ["!invert!xformOp:scale"]"#,
            "not finite",
        ),
    ] {
        let e = bound_triangle("usd-bad-xform", xform, &skeleton, SECOND_JOINT, "")
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains(want), "{e}");
    }
}

/// Sheared copies are instances of their prototype: the instancer's uneven scale over each
/// copy's rotation shears it, differently per copy, and nothing bakes.
#[test]
fn an_instancer_keeps_sheared_copies_on_one_model() {
    use orr_math::{Mat4, Vec3};
    let d = dir("usd-sheared");
    let path = d.join("stage.usda");
    std::fs::write(
        &path,
        r#"#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "World"
{
    def PointInstancer "Sheared"
    {
        double3 xformOp:scale = (2, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:scale"]
        rel prototypes = [</World/Sheared/Protos/Tile>]
        int[] protoIndices = [0, 0]
        point3f[] positions = [(0, 0, 0), (4, 0, 0)]
        quath[] orientations = [(0.9238795, 0, 0, 0.3826834), (0.9238795, 0, 0.3826834, 0)]
        def Scope "Protos"
        {
            def Mesh "Tile"
            {
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
                point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 1)]
                uniform token subdivisionScheme = "none"
            }
        }
    }
}
"#,
    )
    .unwrap();
    let (a, _) = usd(&path, UsdSettings::default()).unwrap();
    assert!(
        a.find(AssetId::from_name(&ModelName::source("Tile").mesh("u")))
            .is_some()
    );
    assert!(
        a.find(AssetId::from_name(
            &ModelName::source("Tile").with(Derived::Baked(0)).entry("u")
        ))
        .is_none()
    );
    let page = a
        .entries()
        .iter()
        .find(|e| e.class == Class::InstancePage as u16)
        .unwrap();
    let (head, body) = (entry_blob(&a, page, 0), entry_blob(&a, page, 1));
    let (_, records) = orr_pack::instance::decode(&head, &body).unwrap();
    let got: Vec<Mat4> = records.map(|(_, m)| m.into()).collect();
    let stretch = Vec3::new(2.0, 1.0, 1.0);
    let turn = |axis: Vec3| Quat::from_axis_angle(axis, std::f32::consts::FRAC_PI_4);
    for (q, x) in [(turn(Vec3::Z), 0.0), (turn(Vec3::Y), 8.0)] {
        let want =
            Mat4::from_translation(Vec3::X * x) * Mat4::from_scale(stretch) * Mat4::from_quat(q);
        assert!(
            got.iter().any(|m| m.abs_diff_eq(want, 1e-3)),
            "{want:?} not among {got:?}"
        );
    }
}

/// Copies an instance transform cannot hold bake their surface, whether the instancer's
/// scales or the instancer itself flatten them, and keep the surface's normal.
#[test]
fn an_instancer_bakes_collapsed_copies() {
    let d = dir("usd-collapsed");
    let path = d.join("stage.usda");
    let tile = |name: &str| {
        format!(
            r#"def Scope "Protos"
        {{
            def Mesh "{name}"
            {{
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
                point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 1)]
                normal3f[] normals = [(0, 0, -1), (0, 0, -1), (0, 0, -1)] (interpolation = "vertex")
                uniform token subdivisionScheme = "none"
            }}
        }}"#
        )
    };
    let (a, b) = (tile("Kept"), tile("Flat"));
    std::fs::write(
        &path,
        format!(
            r#"#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "World"
{{
    def PointInstancer "Scaled"
    {{
        rel prototypes = [</World/Scaled/Protos/Kept>]
        int[] protoIndices = [0, 0, 0]
        point3f[] positions = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        float3[] scales = [(1, 1, 1), (2, 2, 0), (0, 0, 0)]
        {a}
    }}

    def PointInstancer "Squashed"
    {{
        double3 xformOp:scale = (2, 2, 0)
        uniform token[] xformOpOrder = ["xformOp:scale"]
        rel prototypes = [</World/Squashed/Protos/Flat>]
        int[] protoIndices = [0]
        point3f[] positions = [(0, 0, 0)]
        {b}
    }}
}}
"#
        ),
    )
    .unwrap();
    let (a, _) = usd(&path, UsdSettings::default()).unwrap();
    assert!(
        a.find(AssetId::from_name(&ModelName::source("Kept").mesh("u")))
            .is_some()
    );
    assert!(
        a.find(AssetId::from_name(
            &ModelName::source("Kept").with(Derived::Baked(1)).mesh("u")
        ))
        .is_some()
    );
    assert!(
        a.find(AssetId::from_name(&ModelName::source("Flat").mesh("u")))
            .is_none()
    );
    let two = 2.0f32.to_bits();
    let flat = [[0, 0, 0], [two, 0, 0], [0, two, 0]];
    for mesh in [
        &ModelName::source("Kept").with(Derived::Baked(0)).mesh("u"),
        &ModelName::source("Flat").with(Derived::Baked(0)).mesh("u"),
    ] {
        assert_eq!(triangle_set(&a, mesh), [flat], "{mesh}");
        let info = MeshInfo::parse(&blob(&a, mesh, 0)).unwrap();
        let p = &info.pages[0];
        let mut page = vec![0; p.resident_len as usize];
        orr_pack::mesh::decode_page(&blob(&a, mesh, p.blob as usize), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        let n = view.normal(&view.cluster(0), 0);
        assert!(n[2] < -0.99, "{mesh}: {n:?}");
    }
}

/// One tree written as glTF and as USD: a square trunk of four quads and two crossed leaf
/// cards, a bark material with a normal map and an alpha-tested leaf material.
fn tree_pair(d: &Path) -> (PathBuf, PathBuf) {
    let (mut p, mut n, mut uv) = (Vec::new(), Vec::new(), Vec::new());
    let (mut trunk, mut leaves) = (Vec::new(), Vec::new());
    let mut quad = |corners: [[f32; 3]; 4], normal: [f32; 3], out: &mut Vec<u32>| {
        let base = p.len() as u32;
        for (c, t) in corners
            .iter()
            .zip([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        {
            p.push(*c);
            n.push(normal);
            uv.push(t);
        }
        out.extend([base, base + 1, base + 2, base + 3]);
    };
    let r = 0.125f32;
    let sides = [
        (
            [[-r, 0.0, r], [r, 0.0, r], [r, 2.0, r], [-r, 2.0, r]],
            [0.0, 0.0, 1.0],
        ),
        (
            [[r, 0.0, r], [r, 0.0, -r], [r, 2.0, -r], [r, 2.0, r]],
            [1.0, 0.0, 0.0],
        ),
        (
            [[r, 0.0, -r], [-r, 0.0, -r], [-r, 2.0, -r], [r, 2.0, -r]],
            [0.0, 0.0, -1.0],
        ),
        (
            [[-r, 0.0, -r], [-r, 0.0, r], [-r, 2.0, r], [-r, 2.0, -r]],
            [-1.0, 0.0, 0.0],
        ),
    ];
    for (c, nrm) in sides {
        quad(c, nrm, &mut trunk);
    }
    quad(
        [
            [-1.0, 1.5, 0.0],
            [1.0, 1.5, 0.0],
            [1.0, 3.0, 0.0],
            [-1.0, 3.0, 0.0],
        ],
        [0.0, 0.0, 1.0],
        &mut leaves,
    );
    quad(
        [
            [0.0, 1.5, 1.0],
            [0.0, 1.5, -1.0],
            [0.0, 3.0, -1.0],
            [0.0, 3.0, 1.0],
        ],
        [1.0, 0.0, 0.0],
        &mut leaves,
    );
    std::fs::write(
        d.join("bark.png"),
        png(16, 16, |x, y| [(x * 16) as u8, 90, (y * 16) as u8, 255]),
    )
    .unwrap();
    std::fs::write(
        d.join("bark_n.png"),
        png(16, 16, |x, y| {
            [128 + (x * 4) as u8, 128 - (y * 4) as u8, 240, 255]
        }),
    )
    .unwrap();
    std::fs::write(
        d.join("leaf.png"),
        png(16, 16, |x, y| {
            let inside = (x as i32 - 8).pow(2) + (y as i32 - 8).pow(2) < 40;
            [40, 120 + (y * 4) as u8, 30, if inside { 255 } else { 0 }]
        }),
    )
    .unwrap();

    // glTF: triangles as the USD importer cuts each quad (a b c d into d a b, b c d).
    let tris = |q: &[u32]| -> Vec<u32> {
        q.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|c| [c[3], c[0], c[1], c[1], c[2], c[3]])
            .collect()
    };
    let mut g = Glb::default();
    let pa = g.floats(&p, "VEC3");
    let na = g.floats(&n, "VEC3");
    let ta = g.floats(&uv, "VEC2");
    let ia = g.indices(&tris(&trunk));
    let la = g.indices(&tris(&leaves));
    let gltf = d.join("tree.glb");
    g.write(
        &gltf,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[
 {{"attributes":{{"POSITION":{pa},"NORMAL":{na},"TEXCOORD_0":{ta}}},"indices":{ia},"material":0}},
 {{"attributes":{{"POSITION":{pa},"NORMAL":{na},"TEXCOORD_0":{ta}}},"indices":{la},"material":1}}]}}],
"materials":[
 {{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}},"metallicFactor":0,"roughnessFactor":0.875}},"normalTexture":{{"index":1}}}},
 {{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":2}},"metallicFactor":0,"roughnessFactor":0.625}},"alphaMode":"MASK","alphaCutoff":0.5,"doubleSided":true}}],
"textures":[{{"source":0}},{{"source":1}},{{"source":2}}],
"images":[{{"uri":"bark.png"}},{{"uri":"bark_n.png"}},{{"uri":"leaf.png"}}]"#
        ),
    );

    // USD: the same points and normals, st with v from the bottom.
    let list = |v: &[[f32; 3]]| {
        v.iter()
            .map(|a| format!("({:?}, {:?}, {:?})", a[0], a[1], a[2]))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let st = uv
        .iter()
        .map(|t| format!("({:?}, {:?})", t[0], 1.0 - t[1]))
        .collect::<Vec<_>>()
        .join(", ");
    let ints = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");
    let mesh = |name: &str, idx: &[u32], material: &str, double: bool| {
        format!(
            r#"    def Mesh "{name}"
    {{
        uniform token subdivisionScheme = "none"
        uniform bool doubleSided = {double}
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{idx}]
        point3f[] points = [{points}]
        normal3f[] normals = [{normals}] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [{st}] (
            interpolation = "vertex"
        )
        rel material:binding = </Tree/Looks/{material}>
    }}
"#,
            counts = vec!["4"; idx.len() / 4].join(", "),
            idx = ints(idx),
            points = list(&p),
            normals = list(&n),
        )
    };
    let texture = |material: &str, name: &str, file: &str, extra: &str| {
        format!(
            r#"            def Shader "{name}"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{file}@
                float2 inputs:st.connect = </Tree/Looks/{material}/St.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
{extra}                float3 outputs:rgb
                float outputs:a
            }}
"#
        )
    };
    let st_reader = r#"            def Shader "St"
            {
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }
"#;
    let usd_text = format!(
        r#"#usda 1.0
(
    defaultPrim = "Tree"
    upAxis = "Y"
    metersPerUnit = 1
)

def Xform "Tree"
{{
{trunk}{leaves}
    def Scope "Looks"
    {{
        def Material "Bark"
        {{
            token outputs:surface.connect = </Tree/Looks/Bark/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Tree/Looks/Bark/Color.outputs:rgb>
                normal3f inputs:normal.connect = </Tree/Looks/Bark/Normal.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness = 0.875
                token outputs:surface
            }}
{color}{normal}{st_reader}        }}

        def Material "Leaf"
        {{
            token outputs:surface.connect = </Tree/Looks/Leaf/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Tree/Looks/Leaf/Color.outputs:rgb>
                float inputs:opacity.connect = </Tree/Looks/Leaf/Color.outputs:a>
                float inputs:opacityThreshold = 0.5
                float inputs:metallic = 0
                float inputs:roughness = 0.625
                token outputs:surface
            }}
{leaf}{st_reader}        }}
    }}
}}
"#,
        trunk = mesh("Trunk", &trunk, "Bark", false),
        leaves = mesh("Leaves", &leaves, "Leaf", true),
        color = texture("Bark", "Color", "bark.png", ""),
        normal = texture(
            "Bark",
            "Normal",
            "bark_n.png",
            "                token inputs:sourceColorSpace = \"raw\"\n                float4 inputs:scale = (2, 2, 2, 1)\n                float4 inputs:bias = (-1, -1, -1, 0)\n",
        ),
        leaf = texture("Leaf", "Color", "leaf.png", ""),
    );
    let usd = d.join("tree.usda");
    std::fs::write(&usd, usd_text).unwrap();
    (gltf, usd)
}

/// Fields of two materials match, texture bindings by slot use and cooked contents.
fn assert_same_materials(ga: &Archive, g: &Material, ua: &Archive, u: &Material) {
    let content = |a: &Archive, id: AssetId| a.find(id).map(|e| e.content_hash);
    for s in 0..g.textures.len() {
        let (gb, ub) = (g.textures[s], u.textures[s]);
        assert_eq!(
            gb.texture == AssetId::NONE,
            ub.texture == AssetId::NONE,
            "slot {s}"
        );
        assert_eq!(
            (gb.channels, gb.uv_set),
            (ub.channels, ub.uv_set),
            "slot {s}"
        );
        assert_eq!(
            content(ga, gb.texture),
            content(ua, ub.texture),
            "slot {s} texels"
        );
    }
    assert_eq!(g.tangent_set(), u.tangent_set());
    let strip = |m: &Material| Material {
        textures: bytemuck::Zeroable::zeroed(),
        frame_uv: 0,
        ..*m
    };
    assert_eq!(strip(g), strip(u));
}

/// `UsdUVTexture` scale and bias state how a normal map is encoded, for any image: the same
/// vectors cook to the same texture from values as stored, at `0.5 v + 0.5`, or flipped. They
/// repeat: black past the edges reads as a different vector under each.
#[test]
fn usd_normals_decode_by_their_scale_and_bias() {
    let d = dir("usd-normal-decode");
    let x = |px: u32| if px < 8 { 0.6f32 } else { -0.6 };
    let cases = [
        ("(1, 1, 1, 1)", "(0, 0, 0, 0)", 1.0, 0.0),
        ("(2, 2, 2, 1)", "(-1, -1, -1, 0)", 0.5, 0.5),
        ("(-1, 1, 1, 1)", "(0, 0, 0, 0)", -1.0, 0.0),
    ];
    let cooked = cases.map(|(scale, bias, k, c)| {
        let format = ctt::TargetFormat::Uncompressed(ctt::Format::R32G32B32A32_SFLOAT);
        let file = float_ktx2(16, 16, format, false, |px, _| {
            [k * x(px) + c, c, 0.8 * k.abs() + c, 1.0]
        });
        std::fs::write(d.join("n.ktx2"), file).unwrap();
        let path = d.join("n.usda");
        std::fs::write(
            &path,
            format!(
                r#"#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)

def Xform "World"
{{
    def Mesh "Quad"
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </World/Looks/M>
    }}

    def Scope "Looks"
    {{
        def Material "M"
        {{
            token outputs:surface.connect = </World/Looks/M/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                normal3f inputs:normal.connect = </World/Looks/M/Normal.outputs:rgb>
                token outputs:surface
            }}
            def Shader "Normal"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @n.ktx2@
                float2 inputs:st.connect = </World/Looks/M/St.outputs:result>
                float4 inputs:scale = {scale}
                float4 inputs:bias = {bias}
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float3 outputs:rgb
            }}
            def Shader "St"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}
}}
"#
            ),
        )
        .unwrap();
        let (a, _) = usd(&path, UsdSettings::default()).unwrap();
        let m = material(
            &a,
            &MaterialName::new(MaterialBase::Path("/World/Looks/M".into())).entry("u"),
        );
        a.find(m.textures[slot::NORMAL].texture)
            .unwrap()
            .content_hash
    });
    assert_eq!(cooked[0], cooked[1]);
    assert_eq!(cooked[0], cooked[2]);
}

/// USD states no tangents, so a mesh reading a normal map gets MikkTSpace's, here along u.
#[test]
fn usd_meshes_with_normal_maps_compute_tangents() {
    let d = dir("usd-tangents");
    std::fs::write(d.join("n.png"), png(4, 4, |_, _| [128, 128, 255, 255])).unwrap();
    let path = d.join("quad.usda");
    std::fs::write(
        &path,
        r#"#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)

def Xform "World"
{
    def Mesh "Quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </World/Looks/M>
    }

    def Scope "Looks"
    {
        def Material "M"
        {
            token outputs:surface.connect = </World/Looks/M/Surface.outputs:surface>
            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                normal3f inputs:normal.connect = </World/Looks/M/Normal.outputs:rgb>
                token outputs:surface
            }
            def Shader "Normal"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @n.png@
                float2 inputs:st.connect = </World/Looks/M/St.outputs:result>
                float4 inputs:scale = (2, 2, 2, 1)
                float4 inputs:bias = (-1, -1, -1, 0)
                float3 outputs:rgb
            }
            def Shader "St"
            {
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }
        }
    }
}
"#,
    )
    .unwrap();
    let settings = UsdSettings {
        mesh: MeshSettings {
            tangent_error_max: None,
            ..MeshSettings::default()
        },
        ..UsdSettings::default()
    };
    let (a, _) = usd(&path, settings).unwrap();
    let info = MeshInfo::parse(&blob(&a, "u/mesh", 0)).unwrap();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        orr_pack::mesh::decode_page(&blob(&a, "u/mesh", p.blob as usize), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            for v in 0..u32::from(c.vertex_count) {
                assert!(view.tangent(&c, v).unwrap()[0] > 0.999);
            }
        }
    }
}

fn level_zero() -> MeshSettings {
    MeshSettings {
        lod: LodMode::Discrete { errors: Vec::new() },
        ..MeshSettings::default()
    }
}

/// Cooks a glTF tree and its USD export; their meshes and materials match.
fn assert_cooks_alike(gltf: &Path, usd_path: &Path) {
    let (ga, _) = cook_one(Source::Gltf {
        path: gltf.into(),
        settings: GltfSettings {
            mesh: level_zero(),
            ..GltfSettings::default()
        },
    })
    .unwrap();
    let (ua, _) = usd(
        usd_path,
        UsdSettings {
            mesh: level_zero(),
            ..UsdSettings::default()
        },
    )
    .unwrap();
    assert_eq!(triangle_set(&ga, "u/mesh"), triangle_set(&ua, "u/mesh"));
    let materials = |a: &Archive| {
        let m = ModelInfo::parse(&blob(a, "u", 0)).unwrap();
        m.materials
            .iter()
            .map(|id| {
                Material::from_bytes(&{
                    let e = a.find(*id).unwrap();
                    let b = &a.blobs(e)[0];
                    let mut v = vec![0; b.raw_len as usize];
                    a.read(b, &mut v).unwrap();
                    v
                })
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let (gm, um) = (materials(&ga), materials(&ua));
    assert_eq!(gm.len(), um.len());
    for (g, u) in gm.iter().zip(&um) {
        assert_same_materials(&ga, g, &ua, u);
    }
}

#[test]
fn a_tree_cooks_alike_from_gltf_and_usd() {
    let d = dir("usd-tree");
    let (gltf, usd_path) = tree_pair(&d);
    assert_cooks_alike(&gltf, &usd_path);

    // Packaged with its textures, the stage cooks the same.
    let usdz = d.join("tree.usdz");
    let mut w = openusd::usdz::ArchiveWriter::create(&usdz).unwrap();
    for f in ["tree.usda", "bark.png", "bark_n.png", "leaf.png"] {
        w.add_layer(f, &std::fs::read(d.join(f)).unwrap()).unwrap();
    }
    w.finish().unwrap();
    assert_cooks_alike(&gltf, &usdz);
}

#[test]
fn unchanged_stages_link_and_changed_textures_recook() {
    let d = dir("usd-incremental");
    let (_, usd_path) = tree_pair(&d);
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::File(d.join("tree.orr")),
            vec![AssetSpec {
                name: "u".into(),
                source: Source::Usd {
                    path: usd_path,
                    settings: UsdSettings::default(),
                },
            }],
        )],
    };
    let cook = || {
        orr_cook::cook(&manifest, &Default::default())
            .unwrap()
            .report
    };
    assert!(cook().archives[0].written);
    assert!(!cook().archives[0].written);
    std::fs::write(
        d.join("leaf.png"),
        png(16, 16, |x, _| [40, 200, 30, if x < 8 { 255 } else { 0 }]),
    )
    .unwrap();
    assert!(cook().archives[0].written);
}

#[test]
#[ignore = "needs ORR_USD_TREE, a USD tree with its glTF export (same stem, .glb) beside it"]
fn exported_tree_cooks_alike_from_gltf_and_usd() {
    let usd_path = PathBuf::from(std::env::var("ORR_USD_TREE").expect("ORR_USD_TREE"));
    assert_cooks_alike(&usd_path.with_extension("glb"), &usd_path);
}

#[test]
#[ignore = "needs ORR_USD_SET, a directory of USD stages"]
fn usd_set_cooks() {
    let set = PathBuf::from(std::env::var("ORR_USD_SET").expect("ORR_USD_SET"));
    let mut stages: Vec<PathBuf> = std::fs::read_dir(&set)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension()
                .is_some_and(|e| ["usd", "usda", "usdc", "usdz"].contains(&e.to_str().unwrap()))
        })
        .collect();
    stages.sort();
    assert!(!stages.is_empty(), "no stages in {}", set.display());
    for s in stages {
        let (a, _) =
            usd(&s, UsdSettings::default()).unwrap_or_else(|e| panic!("{}: {e}", s.display()));
        let count = |c: Class| a.entries().iter().filter(|e| e.class == c as u16).count();
        println!(
            "{}: {} models, {} materials, {} textures, {} instance pages, {} clips",
            s.display(),
            count(Class::Model),
            count(Class::Material),
            count(Class::Texture),
            count(Class::InstancePage),
            count(Class::Clip),
        );
        #[cfg(feature = "subd")]
        {
            let cages = UsdSettings {
                subdivision: orr_cook::Subdivision::Cage,
                ..UsdSettings::default()
            };
            let (a, _) = usd(&s, cages).unwrap_or_else(|e| panic!("{} as cages: {e}", s.display()));
            let n = a
                .entries()
                .iter()
                .filter(|e| e.class == Class::Subd as u16)
                .count();
            println!("{}: {n} subdivision cages", s.display());
        }
    }
}

const MTLX: &str = r#"<?xml version="1.0"?>
<materialx version="1.39" colorspace="lin_rec709" fileprefix="tex/">
  <nodegraph name="NG_wood">
    <input name="tint" type="color3" value="0.5, 0.25, 0.125"/>
    <constant name="tinted" type="color3">
      <input name="value" type="color3" interfacename="tint"/>
    </constant>
    <image name="rough" type="float">
      <input name="file" type="filename" value="orm.png"/>
    </image>
    <image name="orm" type="color3">
      <input name="file" type="filename" value="orm.png"/>
    </image>
    <extract name="metal" type="float">
      <input name="in" type="color3" nodename="orm"/>
      <input name="index" type="integer" value="2"/>
    </extract>
    <output name="color_out" type="color3" nodename="tinted"/>
    <output name="rough_out" type="float" nodename="rough"/>
    <output name="metal_out" type="float" nodename="metal"/>
  </nodegraph>
  <open_pbr_surface name="SR_wood" type="surfaceshader">
    <input name="base_color" type="color3" nodegraph="NG_wood" output="color_out"/>
    <input name="specular_roughness" type="float" nodegraph="NG_wood" output="rough_out"/>
    <input name="base_metalness" type="float" nodegraph="NG_wood" output="metal_out"/>
    <input name="coat_weight" type="float" value="0.25"/>
    <input name="geometry_thin_walled" type="boolean" value="true"/>
  </open_pbr_surface>
  <surfacematerial name="M_wood" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_wood"/>
  </surfacematerial>
  <noise3d name="noise" type="color3"/>
  <open_pbr_surface name="SR_noisy" type="surfaceshader">
    <input name="base_color" type="color3" nodename="noise"/>
  </open_pbr_surface>
  <surfacematerial name="M_noisy" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_noisy"/>
  </surfacematerial>
</materialx>
"#;

fn materialx(path: &Path, material: &str) -> orr_cook::Result<(Archive, orr_cook::CookReport)> {
    cook_one(Source::MaterialX {
        path: path.into(),
        material: material.into(),
        textures: TextureRules::default(),
    })
}

#[test]
fn materialx_documents_map_one_material() {
    let d = dir("mtlx");
    std::fs::create_dir_all(d.join("tex")).unwrap();
    std::fs::write(
        d.join("tex/orm.png"),
        png(8, 8, |x, y| [(x * 32) as u8, 255, (y * 32) as u8, 255]),
    )
    .unwrap();
    let doc = d.join("wood.mtlx");
    std::fs::write(&doc, MTLX).unwrap();
    let (a, _) = materialx(&doc, "M_wood").unwrap();
    let m = material(&a, "u");
    assert_eq!(m.base_color, [0.5, 0.25, 0.125]);
    assert_eq!(m.coat_weight, 0.25);
    assert_eq!(m.thin_walled, 1);
    // Roughness (red) and metalness (blue) of one image share one two-channel texture.
    let packed = AssetId::from_name("u/texture/roughness_metalness");
    assert_eq!(m.textures[slot::ROUGHNESS].texture, packed);
    assert_eq!(m.textures[slot::METALNESS].texture, packed);
    assert_eq!(m.textures[slot::ROUGHNESS].channels, 0);
    assert_eq!(m.textures[slot::METALNESS].channels, 1);
    assert_eq!((m.specular_roughness, m.metalness), (1.0, 1.0));

    let e = match materialx(&doc, "M_noisy") {
        Ok(_) => panic!("a noise node cooks"),
        Err(e) => e.to_string(),
    };
    assert!(e.contains("noise (noise3d)"), "{e}");
}

#[cfg(feature = "subd")]
const CAGE: &str = r#"#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 0.01
)

def Xform "World"
{
    def Mesh "Box"
    {
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5]
        point3f[] points = [(-100, -100, -100), (100, -100, -100), (-100, 100, -100), (100, 100, -100), (-100, -100, 100), (100, -100, 100), (-100, 100, 100), (100, 100, 100)]
        int[] creaseIndices = [0, 1, 3]
        int[] creaseLengths = [3]
        float[] creaseSharpnesses = [2]
        int[] cornerIndices = [7]
        float[] cornerSharpnesses = [10]
        int[] holeIndices = [5]
        token interpolateBoundary = "edgeOnly"
        token faceVaryingLinearInterpolation = "boundaries"
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)] (
            interpolation = "faceVarying"
        )
        int[] primvars:st:indices = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2, 3]
        rel material:binding = </World/Looks/A>

        def GeomSubset "Front"
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [0, 1]
            rel material:binding = </World/Looks/B>
        }
    }

    def Mesh "Floor"
    {
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-300, -300, -200), (300, -300, -200), (300, 300, -200), (-300, 300, -200)]
        rel material:binding = </World/Looks/A>
    }

    def Scope "Looks"
    {
        def Material "A"
        {
            token outputs:surface.connect = </World/Looks/A/S.outputs:surface>
            def Shader "S"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1, 0, 0)
                token outputs:surface
            }
        }
        def Material "B"
        {
            token outputs:surface.connect = </World/Looks/B/S.outputs:surface>
            def Shader "S"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0, 0, 1)
                token outputs:surface
            }
        }
    }
}
"#;

/// A left-handed 2 by 2 grid whose faces read primvars of each interpolation through the vertex
/// color stream: `M` a vertex color and a face-varying fade, `N` a varying color.
#[cfg(feature = "subd")]
const PAINTED_CAGE: &str = r#"#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1
)

def Mesh "Grid" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    uniform token orientation = "leftHanded"
    int[] faceVertexCounts = [4, 4, 4, 4]
    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 0, 1), (1, 0.5, 1), (2, 0, 1), (0, 0, 2), (1, 0, 2), (2, 0, 2)]
    color3f[] primvars:paint = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)] (
        interpolation = "vertex"
    )
    float[] primvars:fade = [0.25, 0.5, 0.75, 1] (
        interpolation = "faceVarying"
    )
    int[] primvars:fade:indices = [0, 1, 1, 0, 1, 2, 2, 1, 3, 3, 2, 2, 3, 2, 2, 3]
    color3f[] primvars:tint = [(1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 1)] (
        interpolation = "varying"
    )
    rel material:binding = </Looks/M>

    def GeomSubset "Right" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        uniform token elementType = "face"
        uniform token familyName = "materialBind"
        int[] indices = [1, 3]
        rel material:binding = </Looks/N>
    }
}

def Scope "Looks"
{
    def Material "M"
    {
        token outputs:surface.connect = </Looks/M/S.outputs:surface>
        def Shader "S"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Looks/M/Paint.outputs:result>
            float inputs:opacity.connect = </Looks/M/Fade.outputs:result>
            token outputs:surface
        }
        def Shader "Paint"
        {
            uniform token info:id = "UsdPrimvarReader_float3"
            string inputs:varname = "paint"
            float3 outputs:result
        }
        def Shader "Fade"
        {
            uniform token info:id = "UsdPrimvarReader_float"
            string inputs:varname = "fade"
            float outputs:result
        }
    }
    def Material "N"
    {
        token outputs:surface.connect = </Looks/N/S.outputs:surface>
        def Shader "S"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Looks/N/Tint.outputs:result>
            token outputs:surface
        }
        def Shader "Tint"
        {
            uniform token info:id = "UsdPrimvarReader_float3"
            string inputs:varname = "tint"
            float3 outputs:result
        }
    }
}
"#;

/// Three meshes of a model, two quads each, whose uv sets each state their own interpolation.
#[cfg(feature = "subd")]
const SET_CAGE: &str = r#"#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1
)

def Mesh "Curved"
{
    int[] faceVertexCounts = [4, 4]
    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1)]
    texCoord2f[] primvars:st = [(0, 0), (0.5, 0.25), (1, 0), (0, 1), (0.5, 0.75), (1, 1)] (
        interpolation = "vertex"
    )
    texCoord2f[] primvars:st1 = [(0.25, 0.25), (0.75, 0.75)] (
        interpolation = "uniform"
    )
}

def Mesh "Flat"
{
    int[] faceVertexCounts = [4, 4]
    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]
    point3f[] points = [(0, 0, 3), (1, 0, 3), (2, 0, 3), (0, 0, 4), (1, 0, 4), (2, 0, 4)]
    texCoord2f[] primvars:st = [(0, 0), (0.5, 0.25), (1, 0), (0, 1), (0.5, 0.75), (1, 1)] (
        interpolation = "varying"
    )
    texCoord2f[] primvars:st1 = [(0.5, 0.25)] (
        interpolation = "constant"
    )
}

def Mesh "Bare"
{
    int[] faceVertexCounts = [4, 4]
    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]
    point3f[] points = [(0, 0, 6), (1, 0, 6), (2, 0, 6), (0, 0, 7), (1, 0, 7), (2, 0, 7)]
}
"#;

/// A ptex face and its limit position, uv and color at one point.
#[cfg(feature = "subd")]
type LimitSample = (u32, [f32; 3], Option<[f32; 2]>, [f32; 4]);

/// Both uv sets of every face of a subdivision cage entry on a 3x3 grid.
#[cfg(feature = "subd")]
fn uv_samples(a: &Archive, name: &str) -> Vec<(u32, [[f32; 2]; 2])> {
    use orr_pack::subd::{Page, SubdInfo};
    let info = SubdInfo::parse(&blob(a, name, 0)).unwrap();
    let mut out = Vec::new();
    for p in &info.pages {
        let page = Page::parse(&blob(a, name, p.blob as usize)).unwrap();
        let values = page.value_points(&info.values);
        for (f, face) in page.faces.iter().enumerate() {
            for u in [0.0, 0.5, 1.0] {
                for v in [0.0, 0.5, 1.0] {
                    let uv = |set| page.limit_uv(f, u, v, set, &values).unwrap();
                    out.push((face.ptex, [uv(0), uv(1)]));
                }
            }
        }
    }
    out.sort_by_key(|s| s.0);
    out
}

#[cfg(feature = "subd")]
#[test]
fn a_cages_uv_sets_cook_by_each_meshs_interpolation() {
    use orr_cook::{CageInput, CageUv, CageValues, FaceVarying, Subdivision};
    use orr_math::{Vec2, Vec3};
    let d = dir("usd-cage-uvs");
    std::fs::write(d.join("cage.usda"), SET_CAGE).unwrap();
    let settings = UsdSettings {
        subdivision: Subdivision::Cage,
        uv1: Some("st1".into()),
        ..UsdSettings::default()
    };
    let (a, report) = usd(&d.join("cage.usda"), settings).unwrap();
    assert!(report.assets[0].warnings.is_empty(), "{report:?}");

    let top_left = |[u, v]: [f32; 2]| Vec2::new(u, 1.0 - v);
    let st = [
        [0.0, 0.0],
        [0.5, 0.25],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.75],
        [1.0, 1.0],
    ]
    .map(top_left);
    let mut cage = CageInput::default();
    for z in [0.0, 3.0, 6.0] {
        let first = cage.positions.len() as u32;
        for (dz, x) in [
            (0.0, 0.0),
            (0.0, 1.0),
            (0.0, 2.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (1.0, 2.0),
        ] {
            cage.positions.push(Vec3::new(x, 0.0, z + dz));
        }
        cage.face_counts.extend([4, 4]);
        cage.face_indices
            .extend([0, 1, 4, 3, 1, 2, 5, 4].map(|k| first + k));
    }
    let of_second = |values: Vec<Vec2>, indices: Vec<u32>| FaceVarying {
        values: [vec![Vec2::ZERO], values].concat(),
        indices: [vec![0; 8], indices, vec![0; 8]].concat(),
    };
    let mut on_points = st.to_vec();
    on_points.resize(18, Vec2::ZERO);
    let corners = vec![1, 2, 5, 4, 2, 3, 6, 5];
    let uniform = [[0.25, 0.25], [0.75, 0.75]].map(top_left).to_vec();
    cage.uvs = vec![
        CageUv {
            values: CageValues::Vertex(on_points),
            set: 0,
            faces: vec![0, 1],
        },
        CageUv {
            values: CageValues::Linear(of_second(st.to_vec(), corners)),
            set: 0,
            faces: vec![2, 3],
        },
        CageUv {
            values: CageValues::Linear(FaceVarying {
                values: [vec![Vec2::ZERO], uniform].concat(),
                indices: [vec![1; 4], vec![2; 4], vec![0; 16]].concat(),
            }),
            set: 1,
            faces: vec![0, 1],
        },
        CageUv {
            values: CageValues::Linear(of_second(vec![top_left([0.5, 0.25])], vec![1; 8])),
            set: 1,
            faces: vec![2, 3],
        },
    ];
    let (b, _) = cook_one(Source::Subd {
        cage,
        materials: vec![material(&a, "u/material/default")],
        settings: Default::default(),
    })
    .unwrap();
    let (usd, built) = (uv_samples(&a, "u/subd"), uv_samples(&b, "u/subd"));
    assert_eq!(usd.len(), 6 * 9);
    for (x, y) in usd.iter().zip(&built) {
        assert_eq!(x.0, y.0);
        let e = (0..2).flat_map(|s| (0..2).map(move |k| (s, k)));
        let e = e.fold(0.0f32, |m, (s, k)| m.max((x.1[s][k] - y.1[s][k]).abs()));
        assert!(e < 1e-6, "{x:?} {y:?}");
    }
    // The curved mesh's points smooth off their control values; the flat mesh's are bilinear.
    let middle = |face: u32| usd.iter().filter(|s| s.0 == face).nth(4).unwrap().1[0];
    let (curved, flat) = (middle(0), middle(2));
    assert_ne!(curved, flat);
    assert!(
        usd.iter()
            .filter(|s| s.0 >= 4)
            .all(|s| s.1 == [[0.0; 2]; 2])
    );
}

/// The limit positions, uvs and colors of every face of a subdivision cage entry on a 3x3 grid.
#[cfg(feature = "subd")]
fn limit_samples(a: &Archive, name: &str) -> Vec<LimitSample> {
    use orr_pack::subd::{Page, SubdInfo};
    let info = SubdInfo::parse(&blob(a, name, 0)).unwrap();
    let mut out = Vec::new();
    for p in &info.pages {
        let page = Page::parse(&blob(a, name, p.blob as usize)).unwrap();
        let (points, values) = (page.points(&info.points), page.value_points(&info.values));
        for (f, face) in page.faces.iter().enumerate() {
            for u in [0.0, 0.5, 1.0] {
                for v in [0.0, 0.5, 1.0] {
                    let l = page.limit(f, u, v, &points).unwrap();
                    let uv = page.limit_uv(f, u, v, 0, &values);
                    let color = page.limit_color(f, u, v, &values).unwrap();
                    out.push((face.ptex, l.position, uv, color));
                }
            }
        }
    }
    out.sort_by_key(|s| s.0);
    out
}

#[cfg(feature = "subd")]
#[test]
fn usd_cages_cook_the_surface_of_the_same_cage_built_in_memory() {
    use orr_cook::{
        Boundary, CageInput, CageUv, CageValues, Corner, Crease, FaceVarying, Subdivision, UvLinear,
    };
    use orr_math::{Vec2, Vec3};
    let d = dir("usd-cage");
    std::fs::write(d.join("cage.usda"), CAGE).unwrap();
    let settings = UsdSettings {
        subdivision: Subdivision::Cage,
        ..UsdSettings::default()
    };
    let (a, _) = usd(&d.join("cage.usda"), settings).unwrap();

    // One model: the floor's mesh, then the box's cage, their sections in that order.
    let model = ModelInfo::parse(&blob(&a, "u", 0)).unwrap();
    assert_eq!(model.header.mesh, AssetId::from_name("u/mesh"));
    assert_eq!(model.header.subd, AssetId::from_name("u/subd"));
    let (la, lb) = (
        AssetId::from_name(
            &MaterialName::new(MaterialBase::Path("/World/Looks/A".into())).entry("u"),
        ),
        AssetId::from_name(
            &MaterialName::new(MaterialBase::Path("/World/Looks/B".into())).entry("u"),
        ),
    );
    assert_eq!(model.materials, [la, la, lb]);

    // The same cage in scene space (Y up, meters), uvs with v flipped.
    let points = [-1.0f32, 1.0];
    let positions = Vec::from_fn(8, |i| {
        let p = [points[i & 1], points[(i >> 1) & 1], points[(i >> 2) & 1]];
        Vec3::new(p[0], p[2], -p[1])
    });
    let st = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
    let corners: [u32; 24] = [
        0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5,
    ];
    let mut uv_indices = [0, 1, 2, 3].repeat(6);
    uv_indices[19] = 4;
    let cage = CageInput {
        positions,
        face_counts: vec![4; 6],
        face_indices: corners.to_vec(),
        face_sections: vec![1, 1, 0, 0, 0, 0],
        creases: vec![
            Crease {
                edge: [0, 1],
                sharpness: 2.0,
            },
            Crease {
                edge: [1, 3],
                sharpness: 2.0,
            },
        ],
        corners: vec![Corner {
            point: 7,
            sharpness: 10.0,
        }],
        holes: vec![5],
        uvs: vec![CageUv {
            values: CageValues::FaceVarying(FaceVarying {
                values: st.iter().map(|&[u, v]| Vec2::new(u, 1.0 - v)).collect(),
                indices: uv_indices,
            }),
            set: 0,
            faces: Vec::new(),
        }],
        colors: Vec::new(),
        left_handed: false,
        boundary: Boundary::Edges,
        uv_linear: UvLinear::Boundaries,
        smooth_triangles: false,
    };
    let (b, _) = cook_one(Source::Subd {
        cage,
        materials: vec![
            material(
                &a,
                &MaterialName::new(MaterialBase::Path("/World/Looks/A".into())).entry("u"),
            ),
            material(
                &a,
                &MaterialName::new(MaterialBase::Path("/World/Looks/B".into())).entry("u"),
            ),
        ],
        settings: Default::default(),
    })
    .unwrap();
    let (usd, built) = (limit_samples(&a, "u/subd"), limit_samples(&b, "u/subd"));
    assert_eq!(usd.len(), 5 * 9, "the hole cooks no face");
    for (x, y) in usd.iter().zip(&built) {
        assert_eq!(x.0, y.0);
        assert!(
            (0..3).all(|k| (x.1[k] - y.1[k]).abs() < 1e-6),
            "{x:?} {y:?}"
        );
        let (p, q) = (x.2.unwrap(), y.2.unwrap());
        assert!((0..2).all(|k| (p[k] - q[k]).abs() < 1e-6), "{x:?} {y:?}");
    }
}

/// Each primvar lands in the channels its reader took, under the rule its interpolation asks
/// for, on the faces that read it, in the order the flipped faces hold their corners.
#[cfg(feature = "subd")]
#[test]
fn usd_cages_carry_primvars_by_their_interpolation_as_the_same_cage_built_in_memory() {
    use orr_cook::{CageColor, CageInput, CageValues, FaceVarying, Subdivision};
    use orr_math::{Vec3, Vec4};
    let d = dir("usd-cage-colors");
    std::fs::write(d.join("cage.usda"), PAINTED_CAGE).unwrap();
    let settings = UsdSettings {
        subdivision: Subdivision::Cage,
        ..UsdSettings::default()
    };
    let (a, report) = usd(&d.join("cage.usda"), settings).unwrap();
    assert!(report.assets[0].warnings.is_empty(), "{report:?}");

    let rgb = |c: [f32; 3]| Vec4::new(c[0], c[1], c[2], 0.0);
    let paint = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ];
    let black = [0.0; 3];
    let tint = [
        [1.0, 0.0, 0.0],
        black,
        [0.0, 1.0, 0.0],
        black,
        [0.0, 0.0, 1.0],
        black,
        [1.0, 1.0, 0.0],
        black,
        [0.0, 1.0, 1.0],
    ];
    let corners = vec![0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7];
    let fade = [0.25, 0.5, 0.75, 1.0].map(|f| Vec4::new(0.0, 0.0, 0.0, f));
    let cage = CageInput {
        positions: Vec::from_fn(9, |i| {
            let y = if i == 4 { 0.5 } else { 0.0 };
            Vec3::new((i % 3) as f32, y, (i / 3) as f32)
        }),
        face_counts: vec![4; 4],
        face_indices: corners.clone(),
        face_sections: vec![0, 1, 0, 1],
        colors: vec![
            CageColor {
                values: CageValues::Vertex(paint.map(rgb).to_vec()),
                channels: 0b0111,
                faces: vec![0, 2],
            },
            CageColor {
                values: CageValues::FaceVarying(FaceVarying {
                    values: fade.to_vec(),
                    indices: vec![0, 1, 1, 0, 1, 2, 2, 1, 3, 3, 2, 2, 3, 2, 2, 3],
                }),
                channels: 0b1000,
                faces: vec![0, 2],
            },
            CageColor {
                values: CageValues::Linear(FaceVarying {
                    values: tint.map(rgb).to_vec(),
                    indices: corners,
                }),
                channels: 0b0111,
                faces: vec![1, 3],
            },
        ],
        left_handed: true,
        ..CageInput::default()
    };
    let (b, _) = cook_one(Source::Subd {
        cage,
        materials: vec![
            material(
                &a,
                &MaterialName::new(MaterialBase::Path("/Looks/M".into())).entry("u"),
            ),
            material(
                &a,
                &MaterialName::new(MaterialBase::Path("/Looks/N".into())).entry("u"),
            ),
        ],
        settings: Default::default(),
    })
    .unwrap();
    let (usd, built) = (limit_samples(&a, "u/subd"), limit_samples(&b, "u/subd"));
    assert_eq!(usd.len(), 4 * 9);
    for (x, y) in usd.iter().zip(&built) {
        assert_eq!(x.0, y.0);
        assert!(
            (0..4).all(|k| (x.3[k] - y.3[k]).abs() < 1e-6),
            "{x:?} {y:?}"
        );
    }
    // The faces of `M` read all four channels and those of `N` read 1 in the fourth.
    assert!(usd.iter().any(|s| s.3[3] == 1.0));
    assert!(usd.iter().any(|s| s.3[3] < 0.9));
}

/// Two collections that overlap make four partitions, each on pages of its own: what neither
/// holds, what one holds, the other, and both. An instancer's copies follow the instancer's
/// membership, and a group carries its prim's attributes with their types. A moving prim heads
/// a rig in each partition whose meshes it moves, each listed as posed under a name of its
/// own, and copies hung from it are on pages the consumer places. With a selection, each
/// selected model stands once where its prim does, on its prim's partition's pages.
#[test]
fn collections_split_a_stage_into_partitions() {
    use orr_cook::{GroupSpec, NodeSelect};
    use orr_math::{IVec3, Mat4, Vec3};
    use orr_pack::instance::InstancePageInfo;
    use orr_pack::layout::{AttributeValue, LayoutInfo};
    use orr_pack::names::{self, ModelBase};

    let d = dir("usd-collections");
    let path = d.join("stage.usda");
    let tri = |name: &str, x: i32| {
        format!(
            r#"    def Mesh "{name}"
    {{
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [({x}, 0, 0), ({x}, 1, 0), ({x}, 0, 1)]
        uniform token subdivisionScheme = "none"
    }}
"#
        )
    };
    let stage = format!(
        r#"#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Z"
)

class "Protos"
{{
{flame}}}

def Xform "World"
{{
{base}{night}{candle}
    def Xform "Both"
    {{
        double3 xformOp:translate = (0, 20, 7)
        uniform token[] xformOpOrder = ["xformOp:translate"]
{both}    }}

    def PointInstancer "Flames"
    {{
        rel prototypes = [</Protos/Flame>]
        int[] protoIndices = [0, 0]
        point3f[] positions = [(10, 0, 0), (20, 0, 0)]
    }}

    def Xform "Spinner"
    {{
        double3 xformOp:translate.timeSamples = {{ 0: (0, 0, 0), 10: (0, 5, 0) }}
        uniform token[] xformOpOrder = ["xformOp:translate"]
{day}{lamp}
        def PointInstancer "Sparks"
        {{
            rel prototypes = [</Protos/Flame>]
            int[] protoIndices = [0, 0, 0]
            point3f[] positions = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]
        }}
    }}
}}

def Scope "Layers"
{{
    def Scope "Night" (
        prepend apiSchemas = ["CollectionAPI:members"]
    )
    {{
        rel collection:members:includes = [</World/Night>, </World/Both>, </World/Spinner/Lamp>]
        token dataLayer:initialState = "activated"
        int dataLayer:order = 2
        token[] dataLayer:tags = ["street", "lamp"]
        int2 dataLayer:cell = (3, -4)
        float other = 1
    }}

    def Scope "Candles" (
        prepend apiSchemas = ["CollectionAPI:members"]
    )
    {{
        rel collection:members:includes = [
            </World/Candle>, </World/Both>, </World/Flames>, </World/Spinner/Sparks>
        ]
    }}
}}
"#,
        flame = tri("Flame", 0),
        base = tri("Base", 1),
        night = tri("Night", 2),
        candle = tri("Candle", 3),
        both = tri("Shape", 4),
        day = tri("Day", 5),
        lamp = tri("Lamp", 6),
    );
    std::fs::write(&path, stage).unwrap();
    let group = |prim: &str, attributes: &[&str]| GroupSpec {
        prim: prim.into(),
        collection: "members".into(),
        attributes: attributes.iter().map(|&a| a.into()).collect(),
    };
    let settings = UsdSettings {
        groups: vec![
            group("/Layers/Night", &["dataLayer"]),
            group("/Layers/Candles", &[]),
        ],
        ..UsdSettings::default()
    };
    let (a, _) = usd(&path, settings.clone()).unwrap();
    let info = LayoutInfo::parse(&blob(&a, &names::layout("u"), 0)).unwrap();

    let names: Vec<&str> = info.groups.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "/Layers/Night.collection:members",
            "/Layers/Candles.collection:members"
        ]
    );
    let night: Vec<(&str, &AttributeValue)> = info.groups[0]
        .attributes
        .iter()
        .map(|at| (at.name.as_str(), &at.value))
        .collect();
    assert_eq!(night.len(), 4, "only the named namespace: {night:?}");
    assert!(night.contains(&(
        "dataLayer:initialState",
        &AttributeValue::Token("activated".into())
    )));
    assert!(night.contains(&("dataLayer:order", &AttributeValue::Int(2))));
    assert!(night.contains(&(
        "dataLayer:tags",
        &AttributeValue::TokenArray(vec!["street".into(), "lamp".into()])
    )));
    assert!(night.contains(&("dataLayer:cell", &AttributeValue::Int2([3, -4]))));
    assert!(info.groups[1].attributes.is_empty());

    // Per partition, by its groups: the models its sources' pages place, with their copies.
    let model = |p: Option<u32>| match p {
        None => AssetId::from_name(&ModelName::WHOLE.entry("u")),
        Some(p) => AssetId::from_name(&ModelName::new(ModelBase::Partition(p)).entry("u")),
    };
    let flame = AssetId::from_name(&ModelName::source("Flame").entry("u"));
    let pages: Vec<(AssetId, InstancePageInfo)> = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
        .map(|e| {
            (
                e.id,
                InstancePageInfo::parse(&entry_blob(&a, e, 0)).unwrap(),
            )
        })
        .collect();
    let mut seen_pages = Vec::new();
    let mut by_groups = std::collections::BTreeMap::new();
    let mut hung = Vec::new();
    let spinner = ModelName::source("Spinner");
    for (p, part) in info.partitions.iter().enumerate() {
        let mut placed: Vec<(AssetId, u16)> = Vec::new();
        for (id, info) in &pages {
            let plain = part.sources.contains(&info.header.source);
            if plain || part.posed_sources.contains(&info.header.source) {
                seen_pages.push(*id);
            }
            if plain {
                assert_eq!(info.header.frame_skeleton, AssetId::NONE);
                placed.extend(info.prototypes.iter().map(|q| (q.model, q.records)));
            } else if part.posed_sources.contains(&info.header.source) {
                let rig = spinner.with(Derived::Partition(p as u32)).skeleton("u");
                assert_eq!(info.header.frame_skeleton, AssetId::from_name(&rig));
                hung.extend(
                    info.prototypes
                        .iter()
                        .map(|q| (part.groups.clone(), q.model, q.records)),
                );
            }
        }
        placed.sort();
        let whole = model((p != 0).then_some(p as u32));
        let rig = match p {
            0 => spinner.clone(),
            p => spinner.with(Derived::Partition(p as u32)),
        };
        let posed = vec![AssetId::from_name(&rig.entry("u"))];
        let moved = matches!(part.groups.as_slice(), [] | [0]);
        assert_eq!(
            part.posed,
            if moved { posed } else { Vec::new() },
            "{:?}",
            part.groups
        );
        by_groups.insert(part.groups.clone(), (whole, placed));
    }
    assert_eq!(
        hung,
        [(vec![1], flame, 3)],
        "copies hung from the rig are posed"
    );
    let only = |m: AssetId| vec![(m, 1)];
    let (whole, placed) = &by_groups[&vec![]];
    assert_eq!(placed, &only(*whole));
    assert_eq!(*whole, model(None));
    assert_eq!(by_groups[&vec![0]].1, only(by_groups[&vec![0]].0));
    assert_eq!(by_groups[&vec![0, 1]].1, only(by_groups[&vec![0, 1]].0));
    let (candles, placed) = &by_groups[&vec![1]];
    let mut want = vec![(*candles, 1), (flame, 2)];
    want.sort();
    assert_eq!(
        placed, &want,
        "the instancer's copies follow its membership"
    );
    assert_eq!(by_groups.len(), 4);

    let mut all: Vec<AssetId> = pages.iter().map(|p| p.0).collect();
    all.sort();
    seen_pages.sort();
    assert_eq!(seen_pages, all, "every page in exactly one partition");

    let select = |node: &str| NodeSelect { node: node.into() };
    let selecting = UsdSettings {
        models: vec![select("Both"), select("Spinner")],
        ..settings
    };
    let (a, _) = usd(&path, selecting).unwrap();
    let info = LayoutInfo::parse(&blob(&a, &names::layout("u"), 0)).unwrap();
    let both = ModelName::source("Both");
    let ids = [both.clone(), spinner].map(|m| AssetId::from_name(&m.entry("u")));
    // Per copy of a selected model: its partition's groups, whether its page is posed, the
    // skeleton it hangs from and where it stands.
    let mut copies = Vec::new();
    for page in a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
    {
        let (head, body) = (entry_blob(&a, page, 0), entry_blob(&a, page, 1));
        let page = InstancePageInfo::parse(&head).unwrap();
        let source = page.header.source;
        let part = info
            .partitions
            .iter()
            .find(|p| p.sources.contains(&source) || p.posed_sources.contains(&source));
        let part = part.expect("every page in a partition");
        let (cell, records) = orr_pack::instance::decode(&head, &body).unwrap();
        for (k, mut m) in records {
            let model = page.prototypes[usize::from(k)].model;
            if let Some(i) = ids.iter().position(|&id| id == model) {
                m.translation += IVec3::from(cell).as_vec3() * 256.0;
                let posed = part.posed_sources.contains(&source);
                let frame = page.header.frame_skeleton;
                copies.push((i, part.groups.clone(), posed, frame, Mat4::from(m)));
            }
        }
    }
    copies.sort_by_key(|c| c.0);
    let [
        (0, groups, false, AssetId::NONE, placed),
        (1, hung, true, frame, _),
    ] = &copies[..]
    else {
        panic!("one copy of each selected model: {copies:?}");
    };
    assert_eq!(groups, &[0, 1], "Both stands in what both groups hold");
    assert_eq!(hung, &Vec::<u32>::new());
    let rig = ModelName::source("/World/Spinner").skeleton("u");
    assert_eq!(
        *frame,
        AssetId::from_name(&rig),
        "Spinner hangs from its own xform's rig"
    );
    let corners = triangles(&a, &both.mesh("u")).1[0].0;
    for p in [Vec3::ZERO, Vec3::Y, Vec3::Z] {
        // The mesh's points, then Both's offset; then meters, Y up.
        let s = (p + Vec3::new(4.0, 0.0, 0.0) + Vec3::new(0.0, 20.0, 7.0)) / 100.0;
        let want = Vec3::new(s.x, s.z, -s.y);
        assert!(
            corners
                .iter()
                .any(|&c| placed.transform_point3(c.into()).abs_diff_eq(want, 1e-3)),
            "{want} not reached: {:?}",
            corners.map(|c| placed.transform_point3(c.into()))
        );
    }
}

/// UsdLux lights cook into their partitions' layout records in scene space, each value as the
/// UsdLux schema (OpenUSD v26.05 `usdLux/schema.usda`) defines it: luminance `intensity ·
/// 2^exposure · color`, divided under `normalize` by the world-space area (a sphere's 4π r², a
/// rect's width x height) or a distant light's π sin²θmax; a sphere emits that luminance over
/// its disc π r²; lights emit along their −Z axis; a DistantLight's `angle` is a diameter in
/// degrees; `ShapingAPI` takes its fallbacks where applied; a normalized sphere of radius 0 is
/// the point every larger one is from afar; `colorTemperature` is white at 6500 K; ShadowAPI's
/// `shadow:enable` turns shadows off. Invisible lights are not cooked,
/// a light in a group lands in its partition, and a prototype's light lands once per copy.
#[test]
fn usd_lights_cook_into_layout_records_as_usdlux_defines_them() {
    use orr_cook::GroupSpec;
    use orr_math::Vec3;
    use orr_pack::layout::{LayoutInfo, LightSource, ShapingSource};
    use orr_pack::names;
    use std::f64::consts::PI;

    let d = dir("usd-lights");
    let path = d.join("stage.usda");
    let light = |ty: &str, name: &str, at: [i32; 3], api: &str, body: &str| {
        format!(
            r#"        def {ty} "{name}" (
            prepend apiSchemas = [{api}]
        )
        {{
            double3 xformOp:translate = ({}, {}, {})
            uniform token[] xformOpOrder = ["xformOp:translate"]
{body}        }}
"#,
            at[0], at[1], at[2]
        )
    };
    let lights = [
        light(
            "DistantLight",
            "Sun",
            [0, 0, 0],
            "",
            "            float inputs:angle = 2\n            float inputs:intensity = 1000\n            \
             float inputs:exposure = 1\n            bool inputs:normalize = 1\n            \
             color3f inputs:color = (1, 0.9, 0.8)\n",
        ),
        light(
            "DistantLight",
            "Moon",
            [0, 0, 0],
            "",
            "            float inputs:angle = 1\n            float inputs:intensity = 3\n",
        ),
        light(
            "SphereLight",
            "Spot",
            [10, 20, 30],
            r#""ShapingAPI""#,
            "            float inputs:radius = 10\n            float inputs:intensity = 400\n            \
             bool inputs:normalize = 1\n            color3f inputs:color = (1, 0.5, 0.25)\n            \
             float inputs:shaping:cone:angle = 40\n            float inputs:shaping:cone:softness = 1.5\n            \
             float inputs:shaping:focus = 2\n            color3f inputs:shaping:focusTint = (0.1, 0.2, 0.3)\n",
        ),
        light(
            "SphereLight",
            "Bulb",
            [-40, 0, 0],
            r#""ShadowAPI""#,
            "            float inputs:radius = 20\n            float inputs:intensity = 5\n            \
             float inputs:exposure = -1\n            bool inputs:shadow:enable = 0\n            \
             bool inputs:enableColorTemperature = 1\n",
        ),
        light(
            "SphereLight",
            "Pin",
            [0, 60, 0],
            r#""ShapingAPI""#,
            "            float inputs:radius = 5\n            float inputs:intensity = 12\n            \
             bool inputs:normalize = 1\n            bool treatAsPoint = 1\n            \
             float inputs:shaping:focus = -3\n",
        ),
        light(
            "SphereLight",
            "Dot",
            [0, 0, 90],
            "",
            "            float inputs:radius = 0\n            float inputs:intensity = 7\n",
        ),
        light(
            "SphereLight",
            "Spark",
            [0, 0, 95],
            "",
            "            float inputs:radius = 0\n            float inputs:intensity = 8\n            \
             bool inputs:normalize = 1\n",
        ),
        light(
            "RectLight",
            "Panel",
            [5, 5, 5],
            "",
            "            float inputs:width = 40\n            float inputs:height = 20\n            \
             float inputs:intensity = 50\n            bool inputs:normalize = 1\n",
        ),
        light(
            "SphereLight",
            "Hidden",
            [0, 0, 0],
            "",
            "            token visibility = \"invisible\"\n",
        ),
        light("SphereLight", "Lamp", [1, 2, 3], "", ""),
    ]
    .concat();
    let stage = format!(
        r#"#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Z"
)

class "Protos"
{{
    def Xform "Lantern"
    {{
        def Mesh "Body"
        {{
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (0, 1, 0), (0, 0, 1)]
            uniform token subdivisionScheme = "none"
        }}

        def SphereLight "Flame"
        {{
            double3 xformOp:translate = (0, 0, 10)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            float inputs:radius = 1
            float inputs:intensity = 4
        }}
    }}
}}

def Xform "World"
{{
    def Mesh "Floor"
    {{
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        uniform token subdivisionScheme = "none"
    }}

    def Xform "Rig"
    {{
        double3 xformOp:translate = (100, -50, 200)
        double3 xformOp:rotateXYZ = (30, 0, 90)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
{lights}    }}

    def PointInstancer "Lanterns"
    {{
        rel prototypes = [</Protos/Lantern>]
        int[] protoIndices = [0, 0]
        point3f[] positions = [(0, 0, 0), (100, 0, 0)]
    }}
}}

def Scope "Night" (
    prepend apiSchemas = ["CollectionAPI:members"]
)
{{
    rel collection:members:includes = [</World/Rig/Lamp>]
}}
"#
    );
    std::fs::write(&path, stage).unwrap();
    let settings = UsdSettings {
        groups: vec![GroupSpec {
            prim: "/Night".into(),
            collection: "members".into(),
            attributes: Vec::new(),
        }],
        ..UsdSettings::default()
    };
    let (a, _) = usd(&path, settings).unwrap();
    let info = LayoutInfo::parse(&blob(&a, &names::layout("u"), 0)).unwrap();

    // Stage space to scene space, with the rig's transform for its lights.
    let s = stage_to_scene();
    let rig = s
        * DMat4::from_translation(DVec3::new(100.0, -50.0, 200.0))
        * DMat4::from_rotation_z(90f64.to_radians())
        * DMat4::from_rotation_x(30f64.to_radians());
    let at = |p: [f64; 3]| rig.transform_point3(DVec3::from_array(p));
    let axis = rig.transform_vector3(DVec3::NEG_Z).normalize().as_vec3();
    let rgb = |c: [f64; 3], k: f64| (DVec3::from_array(c) * k).as_vec3();
    let deg = |x: f64| x.to_radians() as f32;
    let shaped = |cone: f64, softness: f32, focus: f32, tint: Vec3| ShapingSource {
        axis,
        cone_angle: deg(cone),
        cone_softness: softness,
        focus,
        focus_tint: tint,
    };
    let lantern = |x: f64| LightSource::Sphere {
        position: s.transform_point3(DVec3::new(x, 0.0, 10.0)),
        radius: 0.01,
        intensity: rgb([1.0; 3], 4.0 * PI * 0.01f64.powi(2)),
        shaping: None,
        shadows: true,
    };
    let want = vec![
        // 2^1 x 1000 lux with normalize, over a 1 deg angular radius.
        LightSource::Directional {
            direction_to_light: -axis,
            illuminance: rgb([1.0, 0.9, 0.8], 2000.0),
            angular_radius: deg(1.0),
        },
        // Without normalize, 3 nits over the disc: 3 π sin²(0.5 deg) lux.
        LightSource::Directional {
            direction_to_light: -axis,
            illuminance: rgb([1.0; 3], 3.0 * PI * 0.5f64.to_radians().sin().powi(2)),
            angular_radius: deg(0.5),
        },
        // 400 nits over its area 4π (0.1 m)², seen over its disc π (0.1 m)²; softness clamps.
        LightSource::Sphere {
            position: at([10.0, 20.0, 30.0]),
            radius: 0.1,
            intensity: rgb([1.0, 0.5, 0.25], 400.0 / (4.0 * PI * 0.01) * PI * 0.01),
            shaping: Some(shaped(40.0, 1.0, 2.0, Vec3::new(0.1, 0.2, 0.3))),
            shadows: true,
        },
        // Surfaces cast no shadow from it, and its fallback colorTemperature, 6500 K, is white.
        LightSource::Sphere {
            position: at([-40.0, 0.0, 0.0]),
            radius: 0.2,
            intensity: rgb([1.0; 3], 5.0 * 0.5 * PI * 0.04),
            shaping: None,
            shadows: false,
        },
        // The sphere it stands for, 12 nits over 4π (0.05 m)² seen over π (0.05 m)²; the
        // fallback cone, and a negative focus ignored.
        LightSource::Point {
            position: at([0.0, 60.0, 0.0]),
            intensity: rgb([1.0; 3], 12.0 / 4.0),
            shaping: Some(shaped(90.0, 0.0, 0.0, Vec3::ZERO)),
            shadows: true,
        },
        // A sphere of no area emits nothing without normalize.
        LightSource::Point {
            position: at([0.0, 0.0, 90.0]),
            intensity: Vec3::ZERO,
            shaping: None,
            shadows: true,
        },
        // Normalized, the quarter of its emission every larger sphere gives.
        LightSource::Point {
            position: at([0.0, 0.0, 95.0]),
            intensity: rgb([1.0; 3], 8.0 / 4.0),
            shaping: None,
            shadows: true,
        },
    ];
    let lights = &info.partitions[0].lights;
    assert_eq!(lights.len(), want.len() + 3, "{lights:#?}");
    let close = |a: Vec3, b: Vec3| close(a, b);
    for (got, want) in lights.iter().zip(&want) {
        assert!(
            same_light(got, want),
            "{got:#?}
want {want:#?}"
        );
    }
    // A 40 x 20 cm rect, 50 nits over its 0.08 m², emitting along its −Z axis.
    let LightSource::Rect {
        center,
        edge_x,
        edge_y,
        luminance,
        shaping: None,
        shadows: true,
    } = lights[want.len()]
    else {
        panic!("a rect: {:?}", lights[want.len()]);
    };
    assert!(center.abs_diff_eq(at([5.0, 5.0, 5.0]), 1e-9));
    let x = rig.transform_vector3(DVec3::new(40.0, 0.0, 0.0)).as_vec3();
    assert!(close(edge_x, x), "{edge_x} {x}");
    assert!((edge_y.length() - 0.2).abs() < 1e-6);
    assert!(edge_y.dot(x).abs() < 1e-6);
    assert!(close(edge_x.cross(edge_y).normalize(), axis));
    assert!(close(luminance, Vec3::splat(50.0 / 0.08)));
    // Each copy of the lantern places its flame.
    for (got, x) in lights[want.len() + 1..].iter().zip([0.0, 100.0]) {
        let want = lantern(x);
        assert!(
            same_light(got, &want),
            "{got:#?}
want {want:#?}"
        );
    }
    // The lamp is in the group's partition, at its fallback intensity of 1 nit over a radius of
    // 0.5 cm.
    let night = info.partitions.iter().find(|p| p.groups == [0]).unwrap();
    let [
        LightSource::Sphere {
            position,
            radius,
            intensity,
            shaping: None,
            shadows: true,
        },
    ] = night.lights[..]
    else {
        panic!("one lamp in the group: {:?}", night.lights);
    };
    assert!(position.abs_diff_eq(at([1.0, 2.0, 3.0]), 1e-9));
    assert!((radius - 0.005).abs() < 1e-9);
    assert!(close(
        intensity,
        Vec3::splat((PI * 0.005f64.powi(2)) as f32)
    ));
}

fn close(a: orr_math::Vec3, b: orr_math::Vec3) -> bool {
    a.abs_diff_eq(b, 1e-5 * b.length().max(1.0))
}

fn same_light(got: &orr_pack::layout::LightSource, want: &orr_pack::layout::LightSource) -> bool {
    use orr_pack::layout::LightSource as L;
    match (got, want) {
        (
            L::Directional {
                direction_to_light: d,
                illuminance: e,
                angular_radius: r,
            },
            L::Directional {
                direction_to_light: wd,
                illuminance: we,
                angular_radius: wr,
            },
        ) => close(*d, *wd) && close(*e, *we) && (r - wr).abs() < 1e-6,
        (
            L::Sphere {
                position: p,
                radius: r,
                intensity: i,
                shaping: s,
                shadows: h,
            },
            L::Sphere {
                position: wp,
                radius: wr,
                intensity: wi,
                shaping: ws,
                shadows: wh,
            },
        ) => {
            p.abs_diff_eq(*wp, 1e-9)
                && (r - wr).abs() < 1e-6
                && close(*i, *wi)
                && same_shaping(s, ws)
                && h == wh
        }
        (
            L::Point {
                position: p,
                intensity: i,
                shaping: s,
                shadows: h,
            },
            L::Point {
                position: wp,
                intensity: wi,
                shaping: ws,
                shadows: wh,
            },
        ) => p.abs_diff_eq(*wp, 1e-9) && close(*i, *wi) && same_shaping(s, ws) && h == wh,
        _ => false,
    }
}

fn same_shaping(
    a: &Option<orr_pack::layout::ShapingSource>,
    b: &Option<orr_pack::layout::ShapingSource>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.axis.abs_diff_eq(b.axis, 1e-6)
                && (a.cone_angle - b.cone_angle).abs() < 1e-6
                && a.cone_softness == b.cone_softness
                && a.focus == b.focus
                && a.focus_tint.abs_diff_eq(b.focus_tint, 1e-7)
        }
        _ => false,
    }
}

/// A light input whose samples change over time fails the import by name, as a light's record
/// holds one value; samples that all hold one value cook.
#[test]
fn time_sampled_light_inputs_fail_by_name() {
    let d = dir("usd-light-samples");
    let path = d.join("stage.usda");
    let stage = |samples: &str| {
        format!(
            r#"#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Z"
)

def Xform "World"
{{
    def Mesh "Floor"
    {{
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        uniform token subdivisionScheme = "none"
    }}

    def SphereLight "Lamp"
    {{
        float inputs:intensity.timeSamples = {{ {samples} }}
    }}
}}
"#
        )
    };
    std::fs::write(&path, stage("0: 5, 10: 5")).unwrap();
    usd(&path, UsdSettings::default()).unwrap();
    std::fs::write(&path, stage("0: 5, 10: 50")).unwrap();
    let Err(error) = usd(&path, UsdSettings::default()) else {
        panic!("an intensity that changes over time cooked");
    };
    let error = error.to_string();
    assert!(
        error.contains("/World/Lamp inputs:intensity") && error.contains("sampled over time"),
        "{error}"
    );
}
