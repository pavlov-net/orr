//! USD behavior clause by clause. Each row cites the text it holds the importer to:
//! `xformOp.h`, `xformable.h`, `mesh.h`, `primvar.h`, `pointBased.h`, `pointInstancer.h` (PI),
//! `overview.dox` in usdGeom, `bindingAPI.h` and `schemas.dox` in usdSkel, and
//! `spec_usdpreviewsurface.rst` (PS), all as OpenUSD's `release` branch has them.

#[allow(dead_code)]
#[path = "common/tagged.rs"]
mod tagged;

use std::path::PathBuf;

use orr_cook::{ArchiveSpec, AssetSpec, Manifest, NodeSelect, Source, Subdivision, UsdSettings};
use orr_math::{Mat4, Quat, Vec3};
use orr_pack::clip::Clip;
use orr_pack::instance::{InstancePageInfo, decode};
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{MeshInfo, PageView};
use orr_pack::model::ModelInfo;
use orr_pack::skeleton::{SkeletonInfo, Trs};
use orr_pack::texture::TextureInfo;
use orr_pack::{Archive, AssetId, Blob, Class, Target};

const HEAD: &str = r#"#usda 1.0
(
    metersPerUnit = 1
    timeCodesPerSecond = 10
    upAxis = "Y"
)
"#;

/// The mesh attributes of a triangle at (0, 0, 0), (1, 0, 0) and (0, 1, 0).
const TRIANGLE: &str = r#"uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]"#;

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("spec_usd")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cook_at(d: &std::path::Path, body: &str, settings: UsdSettings) -> orr_cook::Result<Archive> {
    cook_stage(d, HEAD, body, settings)
}

fn cook_stage(
    d: &std::path::Path,
    head: &str,
    body: &str,
    settings: UsdSettings,
) -> orr_cook::Result<Archive> {
    let path = d.join("stage.usda");
    std::fs::write(&path, format!("{head}\n{body}")).unwrap();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "u".into(),
                source: Source::Usd { path, settings },
            }],
        )],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default())?;
    Ok(Archive::from_memory(cooked.memory.remove(0)).unwrap())
}

fn cook(name: &str, body: &str) -> Archive {
    cook_at(&dir(name), body, UsdSettings::default()).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn read(a: &Archive, b: &Blob) -> Vec<u8> {
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

fn blob(a: &Archive, name: &str, i: usize) -> Vec<u8> {
    let e = a
        .find(AssetId::from_name(name))
        .unwrap_or_else(|| panic!("no entry {name}"));
    read(a, &a.blobs(e)[i])
}

fn has(a: &Archive, name: &str) -> bool {
    a.find(AssetId::from_name(name)).is_some()
}

/// Position, normal and uv of every corner of a mesh entry.
fn corners(a: &Archive, name: &str) -> Vec<(Vec3, Vec3, [f32; 2])> {
    let info = MeshInfo::parse(&blob(a, name, 0)).unwrap();
    let mut out = Vec::new();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        orr_pack::mesh::decode_page(&blob(a, name, p.blob as usize), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            for t in 0..u32::from(c.tri_count) {
                for v in view.triangle(&c, t).map(u32::from) {
                    out.push((
                        Vec3::from(view.position(&c, v)),
                        Vec3::from(view.normal(&c, v)),
                        view.uv(&c, v, 0),
                    ));
                }
            }
        }
    }
    out
}

/// The transforms that place model `model`, in scene space.
fn copies(a: &Archive, model: &str) -> Vec<Mat4> {
    let want = AssetId::from_name(model);
    let mut out = Vec::new();
    for e in a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
    {
        let blobs = a.blobs(e);
        let (head, body) = (read(a, &blobs[0]), read(a, &blobs[1]));
        let info = InstancePageInfo::parse(&head).unwrap();
        let (cell, records) = decode(&head, &body).unwrap();
        for (p, mut m) in records {
            if info.prototypes[usize::from(p)].model == want {
                m.translation += Vec3::from(cell.map(|c| c as f32)) * 256.0;
                out.push(Mat4::from(m));
            }
        }
    }
    out
}

/// Sorted scene positions, rounded to millimeters, that the copies of `model` put its origin at.
fn origins(a: &Archive, model: &str) -> Vec<[i32; 3]> {
    let mut out: Vec<[i32; 3]> = copies(a, model)
        .iter()
        .map(|m| {
            let p = m.transform_point3(Vec3::ZERO);
            p.to_array().map(|x| (x * 1000.0).round() as i32)
        })
        .collect();
    out.sort_unstable();
    out
}

/// `xformOp.h`: "vec[0] = X, vec[1] = Y, vec[2] = Z. The A, B, C in the op name dictate the order
/// in which their corresponding elements are consumed". A `rotateABC` applies A first, so it is
/// the single-axis ops listed C, B, A.
#[test]
fn rotate_abc_reads_angles_by_axis_and_applies_them_in_name_order() {
    for order in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"] {
        let triple = format!(
            r#"float3 xformOp:rotate{order} = (10, 20, 30)
    uniform token[] xformOpOrder = ["xformOp:rotate{order}"]"#
        );
        let angle = |a: char| 10.0 * ("XYZ".find(a).unwrap() + 1) as f32;
        let axes: Vec<char> = order.chars().rev().collect();
        let singles = format!(
            r#"float xformOp:rotate{a} = {x}
    float xformOp:rotate{b} = {y}
    float xformOp:rotate{c} = {z}
    uniform token[] xformOpOrder = ["xformOp:rotate{a}", "xformOp:rotate{b}", "xformOp:rotate{c}"]"#,
            a = axes[0],
            b = axes[1],
            c = axes[2],
            x = angle(axes[0]),
            y = angle(axes[1]),
            z = angle(axes[2]),
        );
        let got = cook_points(&format!("rotate-{order}"), &triple);
        let want = cook_points(&format!("rotate-{order}-singles"), &singles);
        for (g, w) in got.iter().zip(&want) {
            assert!(g.abs_diff_eq(*w, 1e-4), "{order}: {g} {w}");
        }
    }
}

/// Sorted points of a triangle under `ops`.
fn cook_points(name: &str, ops: &str) -> Vec<Vec3> {
    let a = cook(
        name,
        &format!(
            r#"def Xform "X"
{{
    {ops}
    def Mesh "M"
    {{
        {TRIANGLE}
    }}
}}"#
        ),
    );
    let mut p: Vec<Vec3> = corners(&a, "u/mesh").iter().map(|c| c.0).collect();
    p.sort_by(|a, b| a.to_array().partial_cmp(&b.to_array()).unwrap());
    p
}

/// `xformOp.h` lists the single-axis translate and scale ops; `xformable.h`: "Any entries in
/// xformOpOrder that do not correspond to valid attributes on the xformable prim are skipped".
/// The order puts the last op nearest the geometry, and `!invert!` inverts its op.
#[test]
fn xform_ops_compose_as_listed() {
    let rows: &[(&str, &str, [f32; 3])] = &[
        (
            "translate-x",
            r#"double xformOp:translateX = 2
    uniform token[] xformOpOrder = ["xformOp:translateX"]"#,
            [3.0, 0.0, 0.0],
        ),
        (
            "translate-y",
            r#"double xformOp:translateY = 2
    uniform token[] xformOpOrder = ["xformOp:translateY"]"#,
            [1.0, 2.0, 0.0],
        ),
        (
            "translate-z",
            r#"double xformOp:translateZ = 2
    uniform token[] xformOpOrder = ["xformOp:translateZ"]"#,
            [1.0, 0.0, 2.0],
        ),
        (
            "scale-x",
            r#"float xformOp:scaleX = 3
    uniform token[] xformOpOrder = ["xformOp:scaleX"]"#,
            [3.0, 0.0, 0.0],
        ),
        (
            "scale-y-keeps-x",
            r#"float xformOp:scaleY = 3
    uniform token[] xformOpOrder = ["xformOp:scaleY"]"#,
            [1.0, 0.0, 0.0],
        ),
        (
            "unvalued-op-skipped",
            r#"double3 xformOp:translate = (0, 0, 5)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateX:gone"]"#,
            [1.0, 0.0, 5.0],
        ),
        (
            "pivot",
            r#"double3 xformOp:translate:pivot = (1, 0, 0)
    float xformOp:rotateZ = 90
    uniform token[] xformOpOrder = ["xformOp:translate:pivot", "xformOp:rotateZ", "!invert!xformOp:translate:pivot"]"#,
            [1.0, 0.0, 0.0],
        ),
        (
            "ops-before-reset-dropped",
            r#"double3 xformOp:translate = (9, 9, 9)
    float xformOp:scale:after = 2
    uniform token[] xformOpOrder = ["xformOp:translate", "!resetXformStack!", "xformOp:scale:after"]"#,
            [2.0, 0.0, 0.0],
        ),
    ];
    for &(name, ops, want) in rows {
        let want = Vec3::from(want);
        let got = cook_points(&format!("op-{name}"), ops);
        assert!(
            got.iter().any(|p| p.abs_diff_eq(want, 1e-4)),
            "{name}: {want} not in {got:?}"
        );
    }
}

/// `mesh.h`: "If a polygonal mesh specifies neither normals nor primvars:normals, then it should
/// be treated and rendered as faceted". Storm's `_UseSmoothNormals` is false for `none` and
/// `bilinear`; the other schemes smooth their hulls.
#[test]
fn polygonal_meshes_without_normals_are_faceted() {
    for (scheme, faceted) in [("none", true), ("bilinear", true), ("catmullClark", false)] {
        let a = cook(
            &format!("faceted-{scheme}"),
            &format!(
                r#"def Mesh "M"
{{
    uniform token subdivisionScheme = "{scheme}"
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 0, 3, 1]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
}}"#
            ),
        );
        let mesh = corners(&a, "u/mesh");
        let face = |n: Vec3| n.abs_diff_eq(Vec3::Z, 1e-2) || n.abs_diff_eq(Vec3::Y, 1e-2);
        assert_eq!(
            mesh.iter().all(|c| face(c.1)),
            faceted,
            "{scheme}: {mesh:?}"
        );
    }
}

/// `primvar.h`: "Return the Primvar's interpolation, which is UsdGeomTokens->constant if
/// unauthored". `pointBased.h`: the plain `normals` attribute falls back to `vertex`.
#[test]
fn unauthored_interpolation_is_constant_for_primvars_and_vertex_for_normals() {
    let a = cook(
        "interpolation-primvar",
        &format!(
            r#"def Mesh "M"
{{
    {TRIANGLE}
    texCoord2f[] primvars:st = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
    normal3f[] primvars:normals = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]
}}"#
        ),
    );
    for c in corners(&a, "u/mesh") {
        assert_eq!(c.2, [0.25, 0.25], "st reads its first element, t flipped");
        assert!(c.1.abs_diff_eq(Vec3::Y, 1e-2), "{}", c.1);
    }
    let a = cook(
        "interpolation-normals",
        &format!(
            r#"def Mesh "M"
{{
    {TRIANGLE}
    normal3f[] normals = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]
}}"#
        ),
    );
    let by_point = [
        (Vec3::ZERO, Vec3::Y),
        (Vec3::X, Vec3::X),
        (Vec3::Y, Vec3::Z),
    ];
    for c in corners(&a, "u/mesh") {
        let (_, n) = by_point.iter().find(|p| p.0 == c.0).unwrap();
        assert!(c.1.abs_diff_eq(*n, 1e-2), "{}", c.1);
    }

    // A primvar of no values holds nothing, whatever it would have applied to.
    let a = cook(
        "interpolation-empty",
        &format!(
            r#"def Mesh "M"
{{
    {TRIANGLE}
    texCoord2f[] primvars:st = []
    normal3f[] primvars:normals = []
    normal3f[] normals = [(0, 1, 0), (0, 1, 0), (0, 1, 0)]
}}"#
        ),
    );
    assert!(
        corners(&a, "u/mesh")
            .iter()
            .all(|c| c.1.abs_diff_eq(Vec3::Y, 1e-2))
    );
}

/// A `PointInstancer` at the given attributes, its prototype a triangle child `Protos/P`.
fn instancer(attributes: &str) -> String {
    format!(
        r#"def PointInstancer "I"
{{
    rel prototypes = [</I/Protos/P>]
    {attributes}
    def Scope "Protos"
    {{
        def Mesh "P"
        {{
            {TRIANGLE}
        }}
    }}
}}"#
    )
}

/// A prototype goes by its path where a selected prim has its name, and a bilinear mesh, whose
/// surface is its faces, needs no cage.
#[test]
fn a_prototype_gives_way_to_a_selected_model_of_its_name() {
    let body = format!(
        r#"def Mesh "P"
{{
    uniform token subdivisionScheme = "bilinear"
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
}}

{}"#,
        instancer("int[] protoIndices = [0]\n point3f[] positions = [(0, 0, 0)]")
    );
    let settings = UsdSettings {
        models: vec![NodeSelect { node: "P".into() }],
        subdivision: Subdivision::Cage,
        ..Default::default()
    };
    let a = cook_at(&dir("prototype-name"), &body, settings).unwrap();
    assert!(has(&a, "u/P/mesh") && has(&a, "u/I/Protos/P/mesh"));
}

/// PI, below: where a copy at the origin turns +x, by each row's orientation attributes.
#[test]
fn orientationsf_wins_unless_its_first_value_is_empty() {
    let turned = "(0.70710678, 0, 0.70710678, 0)";
    let flat = "(1, 0, 0, 0)";
    let start = "int[] protoIndices = [0]\n    point3f[] positions = [(0, 0, 0)]";
    let rows = [
        // PI: "orientationsf will be preferred over orientations if both are authored".
        (
            "both",
            format!("quath[] orientations = [{flat}]\n    quatf[] orientationsf = [{turned}]"),
            Vec3::NEG_Z,
        ),
        // PI: only "if the earliest time sample ... of orientationsf is not empty".
        (
            "empty",
            format!("quath[] orientations = [{turned}]\n    quatf[] orientationsf = []"),
            Vec3::NEG_Z,
        ),
        (
            "empty-first-sample",
            format!(
                "quath[] orientations = [{turned}]
    quatf[] orientationsf.timeSamples = {{ 0: [], 5: [{flat}] }}"
            ),
            Vec3::NEG_Z,
        ),
        (
            "only-f",
            format!("quatf[] orientationsf = [{turned}]"),
            Vec3::NEG_Z,
        ),
    ];
    for (name, attrs, want) in rows {
        let a = cook(
            &format!("orient-{name}"),
            &instancer(&format!("{start}\n    {attrs}")),
        );
        let m = copies(&a, "u/P")[0];
        let got = m.transform_vector3(Vec3::X);
        assert!(got.abs_diff_eq(want, 2e-2), "{name}: {got}");
    }
}

/// PI: "Both pruning features' lists of ID's are combined to produce the mask", the ids being
/// `ids` where authored and the index otherwise.
#[test]
fn inactive_and_invisible_ids_both_mask() {
    let positions = "int[] protoIndices = [0, 0, 0]
    point3f[] positions = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]";
    for (name, meta, attrs, want) in [
        ("inactive", "inactiveIds = [ 0 ]", "", vec![2000, 3000]),
        (
            "union",
            "inactiveIds = [ 0 ]",
            "int64[] invisibleIds = [2]",
            vec![2000],
        ),
        (
            "by-ids",
            "inactiveIds = [ 7 ]",
            "int64[] ids = [5, 6, 7]\n    int64[] invisibleIds = [5]",
            vec![2000],
        ),
    ] {
        let body = instancer(&format!("{positions}\n    {attrs}")).replace(
            r#"def PointInstancer "I""#,
            &format!("def PointInstancer \"I\" (\n    {meta}\n)"),
        );
        let a = cook(&format!("mask-{name}"), &body);
        let got: Vec<i32> = origins(&a, "u/P").iter().map(|o| o[0]).collect();
        assert_eq!(got, want, "{name}");
    }
}

/// PI: "If instance transforms are requested at a time between samples and either of the
/// velocity attributes is authored, then we will not attempt to interpolate samples of positions
/// or orientations". `overview.dox`: velocities that don't line up with positions fall back to
/// standard position computation logic. Ten time codes make a second.
#[test]
fn velocities_extrapolate_from_the_lower_sample_where_they_line_up() {
    let samples = "int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0)], 10: [(10, 0, 0)] }";
    let at = |time: f64| UsdSettings {
        time: Some(time),
        ..UsdSettings::default()
    };
    for (name, attrs, time, want) in [
        (
            "aligned",
            "vector3f[] velocities.timeSamples = { 0: [(2, 0, 0)], 10: [(2, 0, 0)] }",
            5.0,
            [1000, 0, 0],
        ),
        (
            "accelerated",
            "vector3f[] velocities.timeSamples = { 0: [(2, 0, 0)], 10: [(2, 0, 0)] }
    vector3f[] accelerations.timeSamples = { 0: [(0, 8, 0)], 10: [(0, 8, 0)] }",
            5.0,
            [1000, 1000, 0],
        ),
        (
            "on-a-sample",
            "vector3f[] velocities.timeSamples = { 0: [(2, 0, 0)], 10: [(2, 0, 0)] }",
            10.0,
            [10000, 0, 0],
        ),
        (
            "misaligned",
            "vector3f[] velocities.timeSamples = { 2: [(2, 0, 0)], 10: [(2, 0, 0)] }",
            5.0,
            [5000, 0, 0],
        ),
        ("unauthored", "", 5.0, [5000, 0, 0]),
    ] {
        let a = cook_at(
            &dir(&format!("velocity-{name}")),
            &instancer(&format!("{samples}\n    {attrs}")),
            at(time),
        )
        .unwrap();
        assert_eq!(origins(&a, "u/P"), [want], "{name}");
    }

    // Angular velocities turn the lower sample's orientation, in degrees per second, after it.
    let a = cook_at(
        &dir("velocity-angular"),
        &instancer(
            "int[] protoIndices = [0]
    point3f[] positions = [(0, 0, 0)]
    quath[] orientations.timeSamples = { 0: [(1, 0, 0, 0)], 10: [(0, 0, 1, 0)] }
    vector3f[] angularVelocities.timeSamples = { 0: [(0, 180, 0)], 10: [(0, 180, 0)] }",
        ),
        at(5.0),
    )
    .unwrap();
    let got = copies(&a, "u/P")[0].transform_vector3(Vec3::X);
    assert!(got.abs_diff_eq(Vec3::NEG_Z, 2e-2), "{got}");
}

/// PI: "We do not ... provide a specific mechanism for identifying prototypes as geometry that
/// should not be drawn ... in their own, local spaces"; only the instancer prunes traversal.
#[test]
fn a_prototype_outside_the_instancer_also_draws_where_it_stands() {
    let a = cook(
        "proto-in-place",
        &format!(
            r#"def Mesh "P"
{{
    {TRIANGLE}
}}

def PointInstancer "I"
{{
    rel prototypes = [</P>]
    int[] protoIndices = [0]
    point3f[] positions = [(0, 5, 0)]
}}"#
        ),
    );
    assert_eq!(origins(&a, "u/P"), [[0, 5000, 0]]);
    assert_eq!(corners(&a, "u/mesh").len(), 3);
    assert_eq!(corners(&a, "u/P/mesh").len(), 3);
}

/// PI: place prototypes "under a prim that is a "class" or "over". "class" is recommended": they
/// aren't drawn in place, but instance.
#[test]
fn prototypes_under_a_class_or_over_cook_only_as_copies() {
    for specifier in ["class", "over"] {
        let a = cook(
            &format!("proto-{specifier}"),
            &format!(
                r#"{specifier} "Protos"
{{
    def Mesh "P"
    {{
        {TRIANGLE}
    }}
}}

def PointInstancer "I"
{{
    rel prototypes = [</Protos/P>]
    int[] protoIndices = [0]
    point3f[] positions = [(0, 5, 0)]
}}"#
            ),
        );
        assert_eq!(origins(&a, "u/P"), [[0, 5000, 0]], "{specifier}");
        assert!(!has(&a, "u/mesh"), "{specifier} draws in place");
    }
}

/// PI step 1: "Apply (most locally) the authored local to parent transformation for
/// prototypes[protoIndices[i]]". `pointInstancer.cpp` takes `GetLocalTransformation` and ignores
/// its reset flag, and Hydra 2.0's `piPrototypeSceneIndex` keeps the local matrix too; the legacy
/// `UsdImagingDelegate` adapter differs, working from the prim's world transform.
#[test]
fn a_prototype_root_that_resets_the_stack_keeps_only_its_local_transform() {
    let a = cook(
        "proto-reset",
        &format!(
            r#"def Xform "Parent"
{{
    double3 xformOp:translate = (10, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def PointInstancer "I"
    {{
        rel prototypes = [</Parent/I/P>]
        int[] protoIndices = [0]
        point3f[] positions = [(1, 0, 0)]

        def Mesh "P"
        {{
            double3 xformOp:translate = (1, 0, 0)
            uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:translate"]
            {TRIANGLE}
        }}
    }}
}}"#
        ),
    );
    let m = copies(&a, "u/P")[0];
    let got: Vec<Vec3> = corners(&a, "u/P/mesh")
        .iter()
        .map(|c| m.transform_point3(c.0))
        .collect();
    // The instancer at 10, its copy at 1, the root's own 1.
    assert!(
        got.iter()
            .any(|p| p.abs_diff_eq(Vec3::new(12.0, 0.0, 0.0), 1e-3)),
        "{got:?}"
    );
}

/// PI: "Any prim in the scenegraph can be targeted as a prototype";
/// `pointInstancerAdapter.cpp` handles "another nested PointInstancer".
#[test]
fn nested_instancers_multiply_their_copies() {
    let a = cook(
        "proto-nested",
        &format!(
            r#"def PointInstancer "Outer"
{{
    rel prototypes = [</Outer/Inner>]
    int[] protoIndices = [0, 0]
    point3f[] positions = [(100, 0, 0), (200, 0, 0)]

    def PointInstancer "Inner"
    {{
        rel prototypes = [</Outer/Inner/P>]
        int[] protoIndices = [0, 0]
        point3f[] positions = [(0, 1, 0), (0, 2, 0)]

        def Mesh "P"
        {{
            {TRIANGLE}
        }}
    }}
}}"#
        ),
    );
    assert_eq!(
        origins(&a, "u/P"),
        [
            [100000, 1000, 0],
            [100000, 2000, 0],
            [200000, 1000, 0],
            [200000, 2000, 0]
        ]
    );
}

/// A prototype's own subtree decides what it draws, whatever is above it; one with nothing to
/// draw places nothing.
#[test]
fn prototypes_draw_by_their_own_subtree() {
    let a = cook(
        "proto-hidden-scope",
        &format!(
            r#"def Xform "Protos"
{{
    token visibility = "invisible"

    def Mesh "P"
    {{
        {TRIANGLE}
    }}
    def Xform "Empty"
    {{
    }}
}}

def PointInstancer "I"
{{
    rel prototypes = [</Protos/P>, </Protos/Empty>]
    int[] protoIndices = [0, 1]
    point3f[] positions = [(0, 5, 0), (0, 6, 0)]
}}"#
        ),
    );
    assert_eq!(origins(&a, "u/P"), [[0, 5000, 0]]);
    assert!(copies(&a, "u/Empty").is_empty());
}

/// `overview.dox`: "A prim that is Imageable with an authored opinion about its purpose will
/// always have the same effective purpose as its authored value"; a non-Imageable prim's opinion
/// is ignored, and what isn't authored comes from the closest Imageable ancestor.
#[test]
fn purpose_is_the_nearest_opinion_on_an_imageable_prim() {
    let mesh = |purpose: &str| format!("def Mesh \"M\" {{\n{TRIANGLE}\n{purpose}\n}}");
    let rows = [
        (
            "render-under-guide",
            format!(
                "def Xform \"A\" {{\n token purpose = \"guide\"\n{}\n}}",
                mesh("uniform token purpose = \"render\"")
            ),
            true,
        ),
        (
            "typeless-parent",
            format!("def \"A\" {{\n token purpose = \"proxy\"\n{}\n}}", mesh("")),
            true,
        ),
        (
            "through-typeless",
            format!(
                "def Xform \"A\" {{\n token purpose = \"guide\"\n def \"B\" {{\n{}\n}}\n}}",
                mesh("")
            ),
            false,
        ),
        (
            "typeless-render-under-guide",
            format!(
                "def Xform \"A\" {{\n token purpose = \"guide\"\n def \"B\" {{\n token purpose = \"render\"\n{}\n}}\n}}",
                mesh("")
            ),
            false,
        ),
    ];
    for (name, body, drawn) in rows {
        // Something else draws, so that a stage with its mesh hidden still cooks.
        let body = format!("{body}\ndef Mesh \"Other\" {{\n{TRIANGLE}\n}}");
        let a = cook(&format!("purpose-{name}"), &body);
        let want = if drawn { 6 } else { 3 };
        assert_eq!(corners(&a, "u/mesh").len(), want, "{name}");
    }
}

/// Joints `Root` and `Root/Arm`, the arm bound at y = 1.
const RIG: &str = r#"def Skeleton "Rig"
    {
        uniform token[] joints = ["Root", "Root/Arm"]
        uniform matrix4d[] bindTransforms = [((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 1, 0, 1))]
    }"#;

const INFLUENCES: &str = r#"int[] primvars:skel:jointIndices = [1] (
            elementSize = 1
            interpolation = "constant"
        )
        float[] primvars:skel:jointWeights = [1] (
            elementSize = 1
            interpolation = "constant"
        )"#;

/// The joint each vertex of `name` binds, by its largest weight.
fn joints_of(a: &Archive, name: &str) -> Vec<u8> {
    let info = MeshInfo::parse(&blob(a, name, 0)).unwrap();
    let mut out = Vec::new();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        orr_pack::mesh::decode_page(&blob(a, name, p.blob as usize), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            for t in 0..u32::from(c.tri_count) {
                for v in view.triangle(&c, t).map(u32::from) {
                    let (j, w) = (view.joints(&c, v).unwrap(), view.weights(&c, v).unwrap());
                    let k = (0..4).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap();
                    out.push(j[k]);
                }
            }
        }
    }
    out
}

/// `bindingAPI.h`: `skel:joints` "If undefined on a primitive, the primitive inherits the value of
/// the nearest ancestor prim"; `geomBindTransform` "may be authored on the ancestor, to "inherit"
/// down to all the leaf gprims". `cacheImpl.cpp` carries the joint influences down the same way,
/// from prims with `SkelBindingAPI`.
#[test]
fn skinning_properties_inherit_from_binding_ancestors() {
    let group = |api: &str, attributes: &str, mesh: &str| {
        format!(
            r#"def SkelRoot "Hero"
{{
    {RIG}

    def Xform "Group" (
        {api}
    )
    {{
        {attributes}

        def Mesh "Body"
        {{
            {TRIANGLE}
            rel skel:skeleton = </Hero/Rig>
            {mesh}
        }}
    }}
}}"#
        )
    };
    let api = "prepend apiSchemas = [\"SkelBindingAPI\"]";

    // Index 1 of the parent's list is the skeleton's joint 0.
    let a = cook(
        "skel-joints",
        &group(
            api,
            r#"uniform token[] skel:joints = ["Root/Arm", "Root"]"#,
            INFLUENCES,
        ),
    );
    assert!(joints_of(&a, "u/Hero/mesh").iter().all(|&j| j == 0));

    let a = cook(
        "skel-geom-bind",
        &group(
            api,
            "matrix4d primvars:skel:geomBindTransform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (7, 0, 0, 1))",
            INFLUENCES,
        ),
    );
    assert!(
        corners(&a, "u/Hero/mesh")
            .iter()
            .any(|c| c.0.abs_diff_eq(Vec3::new(8.0, 0.0, 0.0), 1e-3))
    );

    let a = cook("skel-influences", &group(api, INFLUENCES, ""));
    assert!(has(&a, "u/Hero/skeleton"), "the mesh cooks skinned");
    assert!(joints_of(&a, "u/Hero/mesh").iter().all(|&j| j == 1));

    // Without the schema the parent's attributes are not bindings.
    let a = cook(
        "skel-joints-no-api",
        &group(
            "",
            r#"uniform token[] skel:joints = ["Root/Arm", "Root"]"#,
            INFLUENCES,
        ),
    );
    assert!(joints_of(&a, "u/Hero/mesh").iter().all(|&j| j == 1));
}

/// A skinned triangle under `SkelRoot` `Hero` at `root`, with `prims` beside the skeleton.
fn skinned(root: &str, prims: &str) -> String {
    format!(
        r#"def SkelRoot "Hero"
{{
    {root}
    {RIG}
    {prims}

    def Mesh "Body"
    {{
        {TRIANGLE}
        rel skel:skeleton = </Hero/Rig>
        {INFLUENCES}
    }}
}}"#
    )
}

const TRANSLATIONS: &str =
    "float3[] translations.timeSamples = { 0: [(0, 0, 0)], 10: [(1, 0, 0)] }";
const ROTATIONS: &str = "quatf[] rotations = [(1, 0, 0, 0)]";
const SCALES: &str = "half3[] scales = [(1, 1, 1)]";

/// `primvar.h`: unauthored interpolation is constant, so one element binds every point.
#[test]
fn joint_influences_without_interpolation_are_constant() {
    let a = cook(
        "skel-unauthored-interpolation",
        &skinned("", "").replace("interpolation = \"constant\"", ""),
    );
    let joints = joints_of(&a, "u/Hero/mesh");
    assert_eq!(joints, [1, 1, 1]);
}

/// `schemas.dox`: "An animation source is only valid if its translation, rotation, and scale
/// components are all authored, storing arrays size to the same size as the authored joints
/// array."
#[test]
fn an_animation_needs_all_three_components_at_the_joint_count() {
    let wave = |attrs: &str| {
        format!("def SkelAnimation \"Wave\" {{\n uniform token[] joints = [\"Root\"]\n {attrs}\n}}")
    };
    for (name, attrs, clip) in [
        (
            "whole",
            format!("{TRANSLATIONS}\n{ROTATIONS}\n{SCALES}"),
            true,
        ),
        ("no-scales", format!("{TRANSLATIONS}\n{ROTATIONS}"), false),
        ("no-rotations", format!("{TRANSLATIONS}\n{SCALES}"), false),
        (
            "blocked",
            format!("{TRANSLATIONS}\n{ROTATIONS}\nhalf3[] scales = None"),
            false,
        ),
        (
            "short",
            format!("{TRANSLATIONS}\n{ROTATIONS}\nhalf3[] scales = []"),
            false,
        ),
    ] {
        let a = cook(&format!("anim-{name}"), &skinned("", &wave(&attrs)));
        assert_eq!(has(&a, "u/Hero/clip/Wave"), clip, "{name}");
    }
}

/// `schemas.dox`: "When a skel:animationSource property is defined, it is "inherited" down
/// namespace, onto any Skeleton primitive beneath it", and a skeleton animates "according to
/// whichever inherited skel:animationSource is defined at the location at which the skel:skeleton
/// is bound". An animation outside the `SkelRoot` shows that the binding was followed.
#[test]
fn an_animation_source_inherits_down_to_the_binding_site() {
    let outside = format!(
        "def SkelAnimation \"Wave\"\n{{\n uniform token[] joints = [\"Root\"]\n {TRANSLATIONS}\n {ROTATIONS}\n {SCALES}\n}}"
    );
    let source = "rel skel:animationSource = </Wave>";
    let above = format!("def Xform \"Above\"\n{{\n{source}\n{}\n}}", skinned("", ""));
    for (name, body) in [
        (
            "above-the-root",
            above.replace("</Hero/Rig>", "</Above/Hero/Rig>"),
        ),
        (
            "at-the-binding-parent",
            format!(
                r#"def SkelRoot "Hero"
{{
    {RIG}

    def Xform "Site"
    {{
        {source}
        rel skel:skeleton = </Hero/Rig>

        def Mesh "Body"
        {{
            {TRIANGLE}
            {INFLUENCES}
        }}
    }}
}}"#
            ),
        ),
    ] {
        let a = cook(&format!("source-{name}"), &format!("{outside}\n{body}"));
        assert!(has(&a, "u/Hero/clip/Wave"), "{name}");
    }
}

/// `schemas.dox`: an animation source "may also be directly nullified by either deactivating the
/// primitive, or by blocking the component attributes".
#[test]
fn a_deactivated_animation_source_nullifies_the_binding() {
    let wave = format!(
        "def SkelAnimation \"Wave\" (\n active = false\n)\n{{\n uniform token[] joints = [\"Root\"]\n {TRANSLATIONS}\n {ROTATIONS}\n {SCALES}\n}}"
    );
    let a = cook(
        "source-inactive",
        &skinned("rel skel:animationSource = </Hero/Wave>", &wave),
    );
    assert!(!has(&a, "u/Hero/clip/Wave"));
    assert!(has(&a, "u/Hero/skeleton"));
}

/// `schemas.dox`: "bindings defined on any primitives that do not have a UsdSkelRoot primitive as
/// one of their ancestors have no meaning, and should be ignored".
#[test]
fn a_skinned_mesh_outside_a_skel_root_is_static() {
    let a = cook(
        "skel-no-root",
        &format!(
            r#"def Xform "Rigless"
{{
    {RIG}

    def Mesh "Body"
    {{
        double3 xformOp:translate = (0, 3, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        {TRIANGLE}
        rel skel:skeleton = </Rigless/Rig>
        {INFLUENCES}
    }}
}}"#
        ),
    );
    assert!(
        corners(&a, "u/mesh")
            .iter()
            .any(|c| c.0.abs_diff_eq(Vec3::new(0.0, 4.0, 0.0), 1e-3))
    );
}

/// A triangle bound to `Material` `M`, whose `UsdPreviewSurface` holds `inputs`, beside `prims`.
fn previewed(inputs: &str, prims: &str, mesh: &str) -> String {
    format!(
        r#"def Mesh "Tri"
{{
    {TRIANGLE}
    rel material:binding = </M>
    {mesh}
}}

def Material "M"
{{
    token outputs:surface.connect = </M/S.outputs:surface>

    def Shader "S"
    {{
        uniform token info:id = "UsdPreviewSurface"
        {inputs}
        token outputs:surface
    }}
    {prims}
}}"#
    )
}

const ST: &str = r#"texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
        interpolation = "vertex"
    )"#;

/// A `UsdUVTexture` `T` reading `tex.png` through a primvar reader `R` of `st`, with `inputs`.
fn texture(name: &str, inputs: &str) -> String {
    format!(
        r#"def Shader "{name}"
    {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @tex.png@
        float2 inputs:st.connect = </M/R.outputs:result>
        {inputs}
        float4 outputs:rgba
        float outputs:r
        float outputs:g
        float3 outputs:rgb
    }}"#
    )
}

const RAW: &str = r#"token inputs:sourceColorSpace = "raw""#;

const READER: &str = r#"def Shader "R"
    {
        uniform token info:id = "UsdPrimvarReader_float2"
        string inputs:varname = "st"
        float2 outputs:result
    }"#;

fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(w, h, |x, y| image::Rgba(f(x, y)));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn shaded_with(name: &str, body: &str, image: &[u8]) -> orr_cook::Result<Archive> {
    let d = dir(name);
    std::fs::write(d.join("tex.png"), image).unwrap();
    cook_at(&d, body, UsdSettings::default())
}

fn shaded(name: &str, body: &str, image: &[u8]) -> Archive {
    shaded_with(name, body, image).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn material(a: &Archive) -> Material {
    Material::from_bytes(&blob(a, "u/material/M", 0)).unwrap()
}

/// Red texels of mip 0 of a bound BC4 texture.
fn red(a: &Archive, m: &Material, s: usize) -> Vec<f32> {
    let e = a.find(m.textures[s].texture).expect("bound");
    let blobs = a.blobs(e);
    let info = TextureInfo::parse(&read(a, &blobs[0])).unwrap();
    let h = info.header;
    let bytes = info
        .mip_texels(&read(a, &blobs[h.mip_blob(0)]), 0, 0)
        .unwrap();
    let mut out = Vec::new();
    for block in bytes.chunks_exact(8) {
        let mut px = [0u8; 16];
        bcdec_rs::bc4(block, &mut px, 4, false);
        out.extend(px.iter().map(|&v| f32::from(v) / 255.0));
    }
    out
}

fn gradient() -> Vec<u8> {
    png(8, 8, |x, _| [(x * 32) as u8, (x * 32) as u8, 0, 255])
}

/// The binding of `diffuseColor` reading texture `T`, with `inputs` on the texture.
fn diffuse_binding(name: &str, inputs: &str, image: &[u8]) -> orr_pack::material::TextureBinding {
    let body = previewed(
        "color3f inputs:diffuseColor.connect = </M/T.outputs:rgb>",
        &format!("{}\n{READER}", texture("T", inputs)),
        ST,
    );
    material(&shaded(name, &body, image)).textures[slot::BASE_COLOR]
}

/// PS `wrapS`, `wrapT`: `black`, `clamp`, `repeat`, `mirror` and the default `useMetadata`: "If
/// the texture contains no such metadata, then fall back to *black*".
#[test]
fn wrap_modes_reach_the_binding_and_metadata_falls_back_to_black() {
    use orr_pack::material::wrap;
    let cases = [
        ("", [wrap::BORDER; 2]),
        (r#"token inputs:wrapS = "useMetadata""#, [wrap::BORDER; 2]),
        (r#"token inputs:wrapT = "black""#, [wrap::BORDER; 2]),
        (
            r#"token inputs:wrapS = "repeat"
            token inputs:wrapT = "clamp""#,
            [wrap::REPEAT, wrap::CLAMP],
        ),
        (
            r#"token inputs:wrapS = "mirror"
            token inputs:wrapT = "repeat""#,
            [wrap::MIRROR, wrap::REPEAT],
        ),
    ];
    for (i, (inputs, want)) in cases.into_iter().enumerate() {
        let b = diffuse_binding(&format!("wrap-{i}"), inputs, &gradient());
        assert_eq!(b.wrap, want, "{inputs:?}");
    }
}

/// PS `wrapS`: `useMetadata` reads what the texture's metadata states, here the `wrapmodes`
/// attribute the OpenEXR standard defines; a stated mode stands.
#[test]
fn use_metadata_reads_an_openexr_wrapmodes_attribute() {
    use exr::prelude::*;
    use orr_pack::material::wrap;
    let attributes = LayerAttributes {
        wrap_mode_name: Some(Text::from("periodic, mirror")),
        ..LayerAttributes::default()
    };
    let layer = Layer::new(
        (8, 8),
        attributes,
        Encoding::FAST_LOSSLESS,
        SpecificChannels::rgba(|Vec2(x, _): Vec2<usize>| (x as f32 / 8.0, 0.5, 0.25, 1.0)),
    );
    let mut bytes = std::io::Cursor::new(Vec::new());
    Image::from_layer(layer)
        .write()
        .to_buffered(&mut bytes)
        .unwrap();
    let bytes = bytes.into_inner();

    let b = diffuse_binding("wrap-exr", "", &bytes);
    assert_eq!(b.wrap, [wrap::REPEAT, wrap::MIRROR]);
    let b = diffuse_binding("wrap-exr-s", r#"token inputs:wrapS = "clamp""#, &bytes);
    assert_eq!(b.wrap, [wrap::CLAMP, wrap::MIRROR]);
}

/// PS `black`: "transparent black (0.0, 0.0, 0.0, 0.0) outside unit square". Scale and bias
/// follow the read, so a bias is what reads there; an image of one value is that value only
/// inside the square, so it stays a texture; and an alpha the material doesn't read as opacity
/// doesn't cut the surface away there.
#[test]
fn black_reads_transparent_black_through_the_reader() {
    use orr_pack::material::wrap;
    let flat = png(8, 8, |_, _| [64, 64, 64, 255]);
    let b = diffuse_binding("black-flat", "", &flat);
    assert_ne!(b.texture, AssetId::NONE, "the image alone is not the input");
    assert_eq!((b.wrap, b.outside), ([wrap::BORDER; 2], [0.0; 4]));

    let bias = previewed(
        "color3f inputs:diffuseColor.connect = </M/T.outputs:rgb>",
        &format!(
            "{}\n{READER}",
            texture("T", "float4 inputs:bias = (0.25, 0.5, 0.75, 0)")
        ),
        ST,
    );
    let m = material(&shaded("black-bias", &bias, &gradient()));
    let b = m.textures[slot::BASE_COLOR];
    assert_eq!(
        b.wrap,
        [wrap::CONSTANT; 2],
        "no hardware border reads a bias"
    );
    // The field holds what the reader reaches past 1.
    for (c, want) in [0.25, 0.5, 0.75].into_iter().enumerate() {
        let got = m.base_color[c] * b.outside[c];
        assert!((got - want).abs() < 1e-5, "channel {c}: {got}");
    }

    let rough = previewed(
        "float inputs:roughness.connect = </M/T.outputs:r>",
        &format!(
            "{}\n{READER}",
            texture(
                "T",
                r#"float4 inputs:scale = (-1, -1, -1, 1)
                float4 inputs:bias = (1, 1, 1, 0)"#
            )
        ),
        ST,
    );
    let m = material(&shaded("black-gloss", &rough, &gradient()));
    let b = m.textures[slot::ROUGHNESS];
    assert_eq!(b.wrap, [wrap::CONSTANT; 2]);
    assert_eq!(b.outside[0], 1.0, "1 - t of a black texel");

    let cut = previewed(
        r#"color3f inputs:diffuseColor.connect = </M/T.outputs:rgb>
        float inputs:opacity = 0.75
        float inputs:opacityThreshold = 0.5"#,
        &format!("{}\n{READER}", texture("T", "")),
        ST,
    );
    let m = material(&shaded("black-opaque", &cut, &gradient()));
    let b = m.textures[slot::BASE_COLOR];
    assert_eq!(b.wrap, [wrap::CONSTANT; 2]);
    assert_eq!(b.outside, [0.0, 0.0, 0.0, 1.0]);
}

/// PS defaults and parameters: `ior` "including the clearcoat if clearcoat > 0";
/// `clearcoatRoughness` 0.01; `specularColor` (0, 0, 0): "This is the color at 0 incidence. Edge
/// color is assumed white."
#[test]
fn preview_parameters_carry_their_spec_values() {
    let close = |a: f32, b: f32| (a - b).abs() < 1e-4;

    let m = material(&cook("preview-empty", &previewed("", "", "")));
    assert_eq!(m.base_color, [0.18; 3]);
    assert_eq!(m.emission_luminance, 0.0);
    assert_eq!((m.metalness, m.specular_roughness), (0.0, 0.5));
    assert_eq!((m.specular_ior, m.geometry_opacity), (1.5, 1.0));
    assert_eq!((m.alpha_mode, m.coat_weight), (Material::OPAQUE, 0.0));

    let a = cook(
        "preview-coat",
        &previewed(
            "float inputs:clearcoat = 0.5\n float inputs:ior = 1.33",
            "",
            "",
        ),
    );
    let m = material(&a);
    assert!(close(m.coat_ior, 1.33), "{}", m.coat_ior);
    assert!(close(m.coat_roughness, 0.01), "{}", m.coat_roughness);

    let f0 = |m: &Material, c: usize| {
        let r = (m.specular_ior - 1.0) / (m.specular_ior + 1.0);
        m.specular_color[c] * r * r
    };
    let specular = "int inputs:useSpecularWorkflow = 1";
    let a = cook("preview-specular-default", &previewed(specular, "", ""));
    assert!(close(f0(&material(&a), 0), 0.0));

    let a = cook(
        "preview-specular-gray",
        &previewed(
            &format!("{specular}\n color3f inputs:specularColor = (0.5, 0.5, 0.5)"),
            "",
            "",
        ),
    );
    let m = material(&a);
    assert!(close(f0(&m, 1), 0.5), "{}", f0(&m, 1));
    assert_eq!(m.specular_color, [1.0; 3], "a gray F0 needs no tint");

    let a = cook(
        "preview-specular-color",
        &previewed(
            &format!("{specular}\n color3f inputs:specularColor = (0.4, 0.2, 0.1)"),
            "",
            "",
        ),
    );
    let m = material(&a);
    for (c, want) in [0.4, 0.2, 0.1].into_iter().enumerate() {
        assert!(close(f0(&m, c), want), "{c}: {}", f0(&m, c));
    }
}

/// PS `opacityMode`: under the default *transparent* "opacity reduces the contribution of the
/// diffuse lighting component in favor of transparency, while the specular response and emissive
/// component remain at full weight"; *presence* "scales the entire lighting response".
#[test]
fn opacity_mode_picks_transmission_or_coverage() {
    let a = cook(
        "opacity-transparent",
        &previewed("float inputs:opacity = 0.25", "", ""),
    );
    let m = material(&a);
    assert_eq!(m.transmission_weight, 0.75);
    assert_eq!(m.geometry_opacity, 1.0);
    assert_eq!(m.alpha_mode, Material::OPAQUE);
    assert_eq!(m.thin_walled, 1);

    let a = cook(
        "opacity-presence",
        &previewed(
            "float inputs:opacity = 0.25\n token inputs:opacityMode = \"presence\"",
            "",
            "",
        ),
    );
    let m = material(&a);
    assert_eq!(m.transmission_weight, 0.0);
    assert_eq!(m.geometry_opacity, 0.25);
    assert_eq!(m.alpha_mode, Material::BLEND);

    // A texture's opacity inverts into the transmission slot.
    let body = previewed(
        "float inputs:opacity.connect = </M/T.outputs:r>",
        &format!("{}\n{READER}", texture("T", RAW)),
        ST,
    );
    let a = shaded("opacity-textured", &body, &gradient());
    let m = material(&a);
    assert_eq!(m.alpha_mode, Material::OPAQUE);
    assert_ne!(m.textures[slot::TRANSMISSION].texture, AssetId::NONE);
    let t = red(&a, &m, slot::TRANSMISSION);
    assert!((t[0] - 1.0).abs() < 0.02, "texel 0 is opacity 0: {}", t[0]);
    assert!((t[3] - (1.0 - 96.0 / 255.0)).abs() < 0.02, "{}", t[3]);
}

/// PS `occlusion`: "pathtracers will likely choose to ignore it". Its reader's image isn't
/// opened, so one that doesn't decode cooks, and its `scale` means nothing either.
#[test]
fn the_occlusion_input_is_ignored() {
    let body = previewed(
        "float inputs:occlusion.connect = </M/T.outputs:r>",
        &format!(
            "{}\n{READER}",
            texture("T", "float4 inputs:scale = (0.5, 0.5, 0.5, 1)")
        ),
        ST,
    );
    let a = shaded("occlusion", &body, b"not an image");
    let bound = material(&a).textures.map(|t| t.texture);
    assert_eq!(bound, [AssetId::NONE; slot::COUNT]);
}

/// PS `scale` and `bias`: "output = textureValue * scale + bias", on any input.
#[test]
fn a_texture_bias_applies_to_scalar_and_color_inputs() {
    let gloss = r#"token inputs:sourceColorSpace = "raw"
        float4 inputs:scale = (-1, -1, -1, 1)
        float4 inputs:bias = (1, 1, 1, 0)
        token inputs:wrapS = "repeat"
        token inputs:wrapT = "repeat""#;
    let body = previewed(
        "float inputs:roughness.connect = </M/T.outputs:r>",
        &format!("{}\n{READER}", texture("T", gloss)),
        ST,
    );

    let a = shaded("bias-varying", &body, &gradient());
    let m = material(&a);
    assert_eq!(m.specular_roughness, 1.0);
    let t = red(&a, &m, slot::ROUGHNESS);
    assert!((t[0] - 1.0).abs() < 0.02, "{}", t[0]);
    assert!((t[3] - (1.0 - 96.0 / 255.0)).abs() < 0.02, "{}", t[3]);

    let a = shaded("bias-constant", &body, &png(8, 8, |_, _| [64, 64, 64, 255]));
    let m = material(&a);
    assert!((m.specular_roughness - (1.0 - 64.0 / 255.0)).abs() < 1e-3);
    assert_eq!(m.textures[slot::ROUGHNESS].texture, AssetId::NONE);

    let body = previewed(
        "color3f inputs:diffuseColor.connect = </M/T.outputs:rgb>",
        &format!(
            "{}\n{READER}",
            texture(
                "T",
                "token inputs:sourceColorSpace = \"raw\"\n float4 inputs:scale = (0.5, 0.5, 0.5, 1)\n float4 inputs:bias = (0.25, 0, 0.5, 0)\n token inputs:wrapS = \"repeat\"\n token inputs:wrapT = \"repeat\""
            )
        ),
        ST,
    );
    let a = shaded("bias-color", &body, &png(8, 8, |_, _| [255, 255, 255, 255]));
    let m = material(&a);
    for (got, want) in m.base_color.iter().zip([0.75, 0.5, 1.0]) {
        assert!((got - want).abs() < 1e-3, "{:?}", m.base_color);
    }

    // Emission reaches past what a texel stores.
    let body = previewed(
        "color3f inputs:emissiveColor.connect = </M/T.outputs:rgb>",
        &format!(
            "{}\n{READER}",
            texture(
                "T",
                "token inputs:sourceColorSpace = \"raw\"\n float4 inputs:scale = (4, 4, 4, 1)\n float4 inputs:bias = (1, 1, 1, 0)"
            )
        ),
        ST,
    );
    let a = shaded("bias-past-one", &body, &png(8, 8, |_, _| [255; 4]));
    assert_eq!(material(&a).emission_color, [5.0; 3]);
}

/// PS `normal`: "properly scaled and ready to be consumed as a tangent space normal"; the (2, -1)
/// pair is a must only for 8-bit textures, so half strength and other pairs are authored meaning.
#[test]
fn a_normal_texture_reads_through_any_scale_and_bias() {
    // A red and green of 0.75 and blue of 1.
    let tilted = png(8, 8, |x, _| [191 + (x % 2) as u8, 191, 255, 255]);
    for (name, inputs, want) in [
        (
            "encoded",
            "float4 inputs:scale = (2, 2, 2, 1)\n float4 inputs:bias = (-1, -1, -1, 0)",
            0.5,
        ),
        (
            "half-strength",
            "float4 inputs:scale = (1, 1, 2, 1)\n float4 inputs:bias = (-0.5, -0.5, -1, 0)",
            0.25,
        ),
        (
            "as-stored",
            "float4 inputs:scale = (1, 1, 1, 1)\n float4 inputs:bias = (-0.5, -0.5, 0, 0)",
            0.25,
        ),
    ] {
        let body = previewed(
            "normal3f inputs:normal.connect = </M/T.outputs:rgb>",
            &format!(
                "{}\n{READER}",
                texture(
                    "T",
                    &format!("token inputs:sourceColorSpace = \"raw\"\n {inputs}")
                )
            ),
            ST,
        );
        let a = shaded(&format!("normal-{name}"), &body, &tilted);
        let m = material(&a);
        let x = normal_x(&a, &m);
        // The cook stores unit normals, so the tilt reads against the vector's length.
        let n = Vec3::new(want, want, 1.0).normalize();
        assert!((x - n.x).abs() < 0.03, "{name}: {x} {}", n.x);
    }
}

/// The mean x of the first texels of a bound normal map.
fn normal_x(a: &Archive, m: &Material) -> f32 {
    let e = a.find(m.textures[slot::NORMAL].texture).expect("bound");
    let blobs = a.blobs(e);
    let info = TextureInfo::parse(&read(a, &blobs[0])).unwrap();
    let h = info.header;
    let bytes = info
        .mip_texels(&read(a, &blobs[h.mip_blob(0)]), 0, 0)
        .unwrap();
    let mut px = [0u8; 32];
    bcdec_rs::bc5(&bytes[..16], &mut px, 8, false);
    let xy = px.as_chunks::<2>().0;
    let sum: f32 = xy
        .iter()
        .map(|t| orr_pack::texture::normal::decode(t.map(|v| f32::from(v) / 255.0))[0])
        .sum();
    sum / xy.len() as f32
}

/// PS `normal` and `displacement` are ordinary inputs: a constant one holds across the surface.
#[test]
fn constant_normal_and_displacement_apply() {
    let a = cook(
        "constant-normal",
        &previewed("normal3f inputs:normal = (0.6, 0, 0.8)", "", ""),
    );
    let m = material(&a);
    assert!((normal_x(&a, &m) - 0.6).abs() < 0.02);

    let a = cook(
        "default-normal",
        &previewed("normal3f inputs:normal = (0, 0, 1)", "", ""),
    );
    assert_eq!(material(&a).textures[slot::NORMAL].texture, AssetId::NONE);

    let a = cook(
        "constant-displacement",
        &previewed("float inputs:displacement = 0.25", "", ""),
    );
    let m = material(&a);
    assert_ne!(m.textures[slot::DISPLACEMENT].texture, AssetId::NONE);
    assert_eq!(m.displacement_offset, 0.25);
    assert_eq!(m.displacement_scale, 0.0);
    assert_eq!((m.displacement_min, m.displacement_max), (0.25, 0.25));
}

/// `body` on a stage in centimeters, with `image` as `tex.png`.
fn cook_centimeters(name: &str, body: &str, image: &[u8]) -> Archive {
    let d = dir(name);
    std::fs::write(d.join("tex.png"), image).unwrap();
    let head = "#usda 1.0\n(\n    metersPerUnit = 0.01\n    upAxis = \"Y\"\n)\n";
    cook_stage(&d, head, body, UsdSettings::default()).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A triangle bound to a Material `M` of `outputs` and `prims`.
fn mtlx_material(outputs: &str, prims: &str) -> String {
    format!(
        r#"def Mesh "Tri"
{{
    {TRIANGLE}
    {ST}
    rel material:binding = </M>
}}

def Material "M"
{{
    {outputs}
    {prims}
}}"#
    )
}

fn open_pbr(inputs: &str) -> String {
    format!(
        r#"def Shader "S"
    {{
        uniform token info:id = "ND_open_pbr_surface_surfaceshader"
        {inputs}
        token outputs:out
    }}"#
    )
}

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1.0)
}

/// OpenPBR Metadata asks for "the floating-point conversion factor from the parameters given in
/// world space length units to meters", which `metersPerUnit` is on a stage. OpenPBR states
/// `thin_film_thickness` in micrometers, which no stage changes.
#[test]
fn open_pbr_lengths_read_in_the_stage_metric() {
    let surface = "token outputs:mtlx:surface.connect = </M/S.outputs:out>";
    let inputs = "float inputs:transmission_depth = 2
        float inputs:thin_film_weight = 1
        float inputs:thin_film_thickness = 0.5";
    let body = mtlx_material(surface, &open_pbr(inputs));
    let m = material(&cook_centimeters("stage-lengths", &body, &[]));
    assert!(near(m.transmission_depth, 0.02), "{}", m.transmission_depth);
    assert_eq!(m.thin_film_thickness, 500.0);
    // The nodedef's default radius of 1 is a length like any other.
    assert!(near(m.subsurface_radius, 0.01), "{}", m.subsurface_radius);

    let body = mtlx_material(surface, &open_pbr("float inputs:subsurface_radius = 3"));
    let m = material(&cook_centimeters("stage-radius", &body, &[]));
    assert!(near(m.subsurface_radius, 0.03), "{}", m.subsurface_radius);
}

/// PS `displacement` under the `metrics.h` metric: Storm's `displacementShader` adds
/// `Neye * displacement` to the eye space point, so the input is a world space length in stage
/// units. The range the cook measures and declares follows the converted scale and offset.
#[test]
fn displacement_reads_in_the_stage_metric() {
    let a = cook_centimeters(
        "stage-constant-displacement",
        &previewed("float inputs:displacement = 25", "", ""),
        &[],
    );
    let m = material(&a);
    assert!(near(m.displacement_offset, 0.25), "{m:?}");
    assert!(near(m.displacement_min, 0.25) && near(m.displacement_max, 0.25));

    let gradient = png(4, 1, |x, _| {
        let v = [0, 85, 170, 255][x as usize];
        [v, v, v, 255]
    });
    let remap = "float4 inputs:scale = (40, 40, 40, 1)
        float4 inputs:bias = (-10, -10, -10, 0)";
    let body = previewed(
        "float inputs:displacement.connect = </M/T.outputs:r>",
        &format!("{}\n{READER}", texture("T", remap)),
        ST,
    );
    let m = material(&cook_centimeters("stage-displacement", &body, &gradient));
    assert!(near(m.displacement_scale, 0.4), "{m:?}");
    assert!(near(m.displacement_offset, -0.1), "{m:?}");
    assert!(
        near(m.displacement_min, -0.1) && near(m.displacement_max, 0.3),
        "{m:?}"
    );

    let remap = "float4 inputs:scale = (2, 2, 2, 1)
        float4 inputs:bias = (1, 1, 1, 0)";
    let body = mtlx_material(
        "token outputs:mtlx:surface.connect = </M/S.outputs:out>
    token outputs:mtlx:displacement.connect = </M/D.outputs:out>",
        &format!(
            r#"{}
    def Shader "D"
    {{
        uniform token info:id = "ND_displacement_float"
        float inputs:displacement.connect = </M/T.outputs:r>
        float inputs:scale = 10
        token outputs:out
    }}
    {}
    {READER}"#,
            open_pbr(""),
            texture("T", remap),
        ),
    );
    let m = material(&cook_centimeters(
        "stage-mtlx-displacement",
        &body,
        &gradient,
    ));
    assert!(near(m.displacement_scale, 0.2), "{m:?}");
    assert!(near(m.displacement_offset, 0.1), "{m:?}");
}

/// Where a slot's uv transform sends mesh uv `p`.
fn sampled(m: &Material, s: usize, p: [f32; 2]) -> [f32; 2] {
    let [a, b, c, d, e, f] = m.uv_transforms[s];
    [
        (1.0 + a) * p[0] + b * p[1] + c,
        d * p[0] + (1.0 + e) * p[1] + f,
    ]
}

/// PS `st`: an input with the default (0, 0), usually connected; `UsdPrimvarReader_float2`
/// `fallback`: "fallback value to be returned if geometry fetch failed"; per-reader transforms.
/// Mesh uvs and sampled uvs both count v from the top, st's t from the bottom.
#[test]
fn texture_coordinates_are_per_reader() {
    let base = "color3f inputs:diffuseColor.connect = </M/T.outputs:rgb>";
    let image = gradient();
    let near = |got: [f32; 2], want: [f32; 2]| {
        assert!(
            (got[0] - want[0]).abs() < 1e-5 && (got[1] - want[1]).abs() < 1e-5,
            "{got:?} {want:?}"
        );
    };
    let any = [0.3, 0.7];

    // Unconnected: st (0, 0), the image's bottom left.
    let body = previewed(
        base,
        r#"def Shader "T"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @tex.png@
        float3 outputs:rgb
    }"#,
        ST,
    );
    let m = material(&shaded("st-unconnected", &body, &image));
    near(sampled(&m, slot::BASE_COLOR, any), [0.0, 1.0]);

    // An authored value.
    let body = previewed(
        base,
        r#"def Shader "T"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @tex.png@
        float2 inputs:st = (0.25, 0.75)
        float3 outputs:rgb
    }"#,
        ST,
    );
    let m = material(&shaded("st-value", &body, &image));
    near(sampled(&m, slot::BASE_COLOR, any), [0.25, 0.25]);

    // A mesh without the primvar reads the reader's fallback, and one with it reads it.
    let reader = r#"def Shader "R"
    {
        uniform token info:id = "UsdPrimvarReader_float2"
        string inputs:varname = "st"
        float2 inputs:fallback = (0.5, 0.25)
        float2 outputs:result
    }"#;
    let prims = format!("{}\n{reader}", texture("T", ""));
    let body = previewed(base, &prims, "");
    let m = material(&shaded("st-fallback", &body, &image));
    near(sampled(&m, slot::BASE_COLOR, any), [0.5, 0.75]);
    let body = previewed(base, &prims, ST);
    let m = material(&shaded("st-primvar", &body, &image));
    near(sampled(&m, slot::BASE_COLOR, any), any);

    // An unset `varname` fetches nothing.
    let unnamed = reader.replace("string inputs:varname = \"st\"\n", "");
    let body = previewed(base, &format!("{}\n{unnamed}", texture("T", "")), ST);
    let m = material(&shaded("st-unnamed", &body, &image));
    near(sampled(&m, slot::BASE_COLOR, any), [0.5, 0.75]);

    // A point's normal map cooks, with no uv frame to turn its normal into.
    let body = previewed(
        "normal3f inputs:normal.connect = </M/T.outputs:rgb>",
        r#"def Shader "T"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @tex.png@
        float4 inputs:scale = (2, 2, 2, 1)
        float4 inputs:bias = (-1, -1, -1, 0)
        float3 outputs:rgb
    }"#,
        ST,
    );
    let m = material(&shaded("st-point-normal", &body, &image));
    near(sampled(&m, slot::NORMAL, any), [0.0, 1.0]);

    // Two readers transform independently; t's flip conjugates each one.
    let transform = r#"def Shader "X"
    {
        uniform token info:id = "UsdTransform2d"
        float2 inputs:in.connect = </M/R.outputs:result>
        float2 inputs:scale = (2, 2)
        float inputs:rotation = 90
        float2 inputs:translation = (0.25, 0.5)
        float2 outputs:result
    }"#;
    let second = texture("U", "").replace("</M/R.outputs:result>", "</M/X.outputs:result>");
    let body = previewed(
        &format!("{base}\n float inputs:roughness.connect = </M/U.outputs:r>"),
        &format!("{}\n{second}\n{READER}\n{transform}", texture("T", "")),
        ST,
    );
    let a = shaded("st-two-transforms", &body, &image);
    let m = material(&a);
    near(sampled(&m, slot::BASE_COLOR, any), any);
    // st (0.3, 0.3) scales to (0.6, 0.6), turns to (-0.6, 0.6) and moves to (-0.35, 1.1).
    near(sampled(&m, slot::ROUGHNESS, any), [-0.35, -0.1]);
    for c in corners(&a, "u/mesh") {
        assert!(
            [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]].contains(&c.2),
            "the mesh keeps its own uvs: {:?}",
            c.2
        );
    }
}

/// PS lists float, float3, float4 and others among reader instantiations. A primvar that holds
/// one value for the mesh gives that value; a missing one the reader's `fallback`.
#[test]
fn constant_primvar_readers_feed_inputs() {
    let reader = |fallback: &str| {
        format!(
            r#"def Shader "C"
    {{
        uniform token info:id = "UsdPrimvarReader_float3"
        string inputs:varname = "displayColor"
        {fallback}
        float3 outputs:result
    }}"#
        )
    };
    let base = "color3f inputs:diffuseColor.connect = </M/C.outputs:result>";
    let a = cook(
        "reader-constant",
        &previewed(
            base,
            &reader(""),
            "color3f[] primvars:displayColor = [(0.25, 0.5, 0.75)]",
        ),
    );
    assert_eq!(material(&a).base_color, [0.25, 0.5, 0.75]);

    let a = cook(
        "reader-fallback",
        &previewed(
            base,
            &reader("float3 inputs:fallback = (0.5, 0.5, 0.25)"),
            "",
        ),
    );
    assert_eq!(material(&a).base_color, [0.5, 0.5, 0.25]);

    // An index picks its element whichever values it leaves out; one that misses them, or a
    // primvar of no numbers, is the fallback.
    let colors = "color3f[] primvars:displayColor = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]";
    for (name, primvar, want) in [
        (
            "indexed",
            format!("{colors}\n int[] primvars:displayColor:indices = [1]"),
            [0.0, 1.0, 0.0],
        ),
        (
            "index-past",
            format!("{colors}\n int[] primvars:displayColor:indices = [7]"),
            [0.5, 0.5, 0.25],
        ),
        (
            "string",
            "string primvars:displayColor = \"red\"".to_string(),
            [0.5, 0.5, 0.25],
        ),
    ] {
        let fallback = reader("float3 inputs:fallback = (0.5, 0.5, 0.25)");
        let a = cook(
            &format!("reader-{name}"),
            &previewed(base, &fallback, &primvar),
        );
        assert_eq!(material(&a).base_color, want, "{name}");
    }

    // Two meshes of one material with different values take a material each.
    let body = format!(
        r#"{}

def Mesh "Other"
{{
    {TRIANGLE}
    rel material:binding = </M>
    color3f[] primvars:displayColor = [(1, 0, 0)]
}}"#,
        previewed(
            base,
            &reader(""),
            "color3f[] primvars:displayColor = [(0, 1, 0)]"
        ),
    );
    let a = cook("reader-two-meshes", &body);
    let model = ModelInfo::parse(&blob(&a, "u", 0)).unwrap();
    let colors: Vec<[f32; 3]> = model
        .materials
        .iter()
        .map(|&id| {
            let e = a.find(id).unwrap();
            Material::from_bytes(&read(&a, &a.blobs(e)[0]))
                .unwrap()
                .base_color
        })
        .collect();
    assert!(colors.contains(&[1.0, 0.0, 0.0]) && colors.contains(&[0.0, 1.0, 0.0]));
}

/// A joint an animation does not list holds its rest transform.
#[test]
fn a_clip_animates_the_joints_it_names() {
    let wave = format!(
        "def SkelAnimation \"Wave\" {{\n uniform token[] joints = [\"Root\"]\n {TRANSLATIONS}\n {ROTATIONS}\n {SCALES}\n}}"
    );
    let a = cook("clip-sparse", &skinned("", &wave));
    let clip = Clip::parse(
        &blob(&a, "u/Hero/clip/Wave", 0),
        &blob(&a, "u/Hero/clip/Wave", 1),
    )
    .unwrap();
    let skeleton = SkeletonInfo::parse(&blob(&a, "u/Hero/skeleton", 0)).unwrap();
    let mut pose = vec![Trs::IDENTITY; 2];
    clip.sample(clip.header.duration, &mut pose);
    assert!(Vec3::from(pose[0].translation).abs_diff_eq(Vec3::X, 1e-3));
    assert_eq!(pose[1].translation, skeleton.rest[1].translation);
    let q = Quat::from_array(pose[0].rotation);
    assert!(q.angle_between(Quat::IDENTITY) < 1e-3);
}

/// A `UsdPreviewSurface` whose `input` of `kind` reads `output` of `tex.png` as `color_space`,
/// repeating so that a flat image folds into its factor.
fn read_through(input: &str, kind: &str, output: &str, color_space: &str) -> String {
    previewed(
        &format!("{kind} inputs:{input}.connect = </M/T.outputs:{output}>"),
        &format!(
            "{}\n{READER}",
            texture(
                "T",
                &format!(
                    "token inputs:sourceColorSpace = \"{color_space}\"
 token inputs:wrapS = \"repeat\"
 token inputs:wrapT = \"repeat\""
                )
            )
        ),
        ST,
    )
}

/// PS `sourceColorSpace` `auto`: "Check for gamma/color space metadata in the texture file
/// itself; if metadata is indicative of sRGB, mark texture as sRGB. If no relevant metadata is
/// found, mark texture as sRGB if it is either 8-bit and has 3 channels or if it is 8-bit and has
/// 4 channels. Otherwise, do not mark texture as sRGB". The row of 8-bit RGB at gamma 1 is Hio's
/// reading (`stbImage.cpp`), which the text doesn't settle.
#[test]
fn auto_reads_file_metadata_then_channels_and_depth() {
    use png::BitDepth::{Eight, Sixteen};
    use png::ColorType::{Grayscale, Rgb};
    let (linear, srgb) = (128.0 / 255.0, orr_color::srgb_decode(128.0 / 255.0) as f32);
    let wide = [0x80u8, 0x80].repeat(3 * 64);
    let gray = [128u8; 64];
    let hint = tagged::gama(0.45455);
    let hint: &[(&[u8; 4], &[u8])] = &[(b"gAMA", &hint)];
    let one = tagged::gama(1.0);
    let rows: [(&str, Vec<u8>, f32); 6] = [
        ("rgb8", tagged::flat_rgb([128; 3], &[]), srgb),
        (
            "gray8",
            tagged::png((Grayscale, Eight), [8, 8], &gray, None, &[]),
            linear,
        ),
        (
            "rgb16",
            tagged::png((Rgb, Sixteen), [8, 8], &wide, None, &[]),
            linear,
        ),
        (
            "gray8_gamma",
            tagged::png((Grayscale, Eight), [8, 8], &gray, None, hint),
            srgb,
        ),
        (
            "rgb16_gamma",
            tagged::png((Rgb, Sixteen), [8, 8], &wide, None, hint),
            srgb,
        ),
        (
            "rgb8_gamma_1",
            tagged::flat_rgb([128; 3], &[(b"gAMA", &one)]),
            linear,
        ),
    ];
    let body = read_through("diffuseColor", "color3f", "rgb", "auto");
    for (name, image, want) in rows {
        let got = material(&shaded(&format!("auto_{name}"), &body, &image)).base_color[0];
        assert!((got - want).abs() < 2e-3, "{name}: {got}, not {want}");
    }
}

/// PS `sourceColorSpace` marks the file, whatever reads it, which is why its `normal` input asks
/// for `raw`: 188 into `roughness` is 0.503 under `auto` and `sRGB`, 0.737 under `raw`.
#[test]
fn the_color_space_holds_for_every_input_of_the_file() {
    let image = tagged::flat_rgb([188; 3], &[]);
    for (space, want) in [("auto", 0.503), ("sRGB", 0.503), ("raw", 188.0 / 255.0)] {
        let body = read_through("roughness", "float", "r", space);
        let name = format!("scalar_{}", space.to_lowercase());
        let got = material(&shaded(&name, &body, &image)).specular_roughness;
        assert!((got - want).abs() < 2e-3, "{space}: {got}, not {want}");
    }
}

/// `colorSpace` metadata on the file names its space where `sourceColorSpace` leaves it to `auto`,
/// and a `sourceColorSpace` authored over `auto` stands. This project's rule: PS doesn't mention
/// the metadata, and `UsdAttribute::GetColorSpace` states what it means.
#[test]
fn file_metadata_stands_in_for_auto() {
    let image = tagged::flat_rgb([128; 3], &[]);
    let srgb = orr_color::srgb_decode(128.0 / 255.0) as f32;
    for (space, want) in [("auto", 128.0 / 255.0), ("sRGB", srgb)] {
        let body = read_through("diffuseColor", "color3f", "rgb", space).replace(
            "@tex.png@",
            "@tex.png@ (
 colorSpace = \"lin_rec709_scene\"
)",
        );
        let name = format!("metadata_{}", space.to_lowercase());
        let got = material(&shaded(&name, &body, &image)).base_color[0];
        assert!((got - want).abs() < 2e-3, "{space}: {got}, not {want}");
    }
}

/// PS `raw`: "Use texture data as it was read from the texture and do not mark it as using a
/// specific color space." Display P3 stated in a file stands under `auto` alone, `sRGB` naming the
/// space itself.
#[test]
fn raw_takes_nothing_from_the_file() {
    let image = tagged::varying_rgb(&[(b"cICP", &[12, 13, 0, 1])]);
    for (space, want) in [("raw", 0), ("sRGB", 0), ("auto", 1)] {
        let body = read_through("diffuseColor", "color3f", "rgb", space);
        let a = shaded(
            &format!("primaries_{}", space.to_lowercase()),
            &body,
            &image,
        );
        let e = a
            .find(material(&a).textures[slot::BASE_COLOR].texture)
            .unwrap();
        let info = TextureInfo::parse(&read(&a, &a.blobs(e)[0])).unwrap();
        assert_eq!(info.header.primaries, want, "{space}");
    }
}
