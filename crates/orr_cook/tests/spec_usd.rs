//! USD behavior clause by clause. Each row cites the text it holds the importer to:
//! `xformOp.h`, `xformable.h`, `mesh.h`, `primvar.h`, `pointBased.h`, `pointInstancer.h` (PI),
//! `overview.dox` in usdGeom, `bindingAPI.h` and `schemas.dox` in usdSkel, and
//! `spec_usdpreviewsurface.rst` (PS), all as OpenUSD's `release` branch has them.

#[allow(dead_code)]
#[path = "common/tagged.rs"]
mod tagged;
#[path = "common/tangent_quads.rs"]
mod tangent_quads;

use std::path::PathBuf;

use orr_cook::{
    ArchiveSpec, AssetSpec, Manifest, MeshPatch, NodeSelect, Source, StaticDisplacement,
    Subdivision, UsdSettings,
};
use orr_math::{Affine3, Mat4, Quat, Vec3};
use orr_pack::clip::Clip;
use orr_pack::instance::{
    InstancePageInfo, decode, motion_interval, motion_rows, motion_transform,
};
use orr_pack::material::{Material, slot, wrap};
use orr_pack::mesh::{MeshInfo, PageView};
use orr_pack::model::ModelInfo;
use orr_pack::skeleton::{SkeletonInfo, Trs};
use orr_pack::texture::TextureInfo;
use orr_pack::tiles::TextureTiles;
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

/// `xformable.h`: `!resetXformStack!` makes a prim "not inherit the parent transformation", so a
/// child of a selected model stands in the world whatever frames the model.
#[test]
fn a_reset_prim_stands_in_the_world_under_a_framed_model() {
    let body = format!(
        r#"def Xform "Frame"
{{
    double3 xformOp:translate = (10, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Child"
    {{
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:translate"]
        {TRIANGLE}
    }}
}}"#
    );
    let settings = UsdSettings {
        models: vec![NodeSelect {
            node: "Frame".into(),
        }],
        ..Default::default()
    };
    let a = cook_at(&dir("reset-under-frame"), &body, settings).unwrap();
    // In the frame's space, which stands at 10.
    let lowest = corners(&a, "u/Frame/mesh")
        .iter()
        .map(|c| c.0.x)
        .fold(f32::MAX, f32::min);
    assert!((lowest + 9.0).abs() < 1e-3, "{lowest}");
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

    // `samplingUtils.cpp` `UsdGeom_GetScales`: "we lock the scale to the last authored value
    // without performing any interpolation", and with a rate nothing interpolates.
    let a = cook_at(
        &dir("velocity-scales"),
        &instancer(&format!(
            "{samples}
    vector3f[] velocities.timeSamples = {{ 0: [(2, 0, 0)], 10: [(2, 0, 0)] }}
    float3[] scales.timeSamples = {{ 0: [(1, 1, 1)], 10: [(2, 2, 2)] }}"
        )),
        at(5.0),
    )
    .unwrap();
    let scale = copies(&a, "u/P")[0].transform_vector3(Vec3::X).length();
    assert!((scale - 1.0).abs() < 1e-3, "{scale}");

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
                    let codes: u32 = w.iter().map(|w| (w * 65535.0).round() as u32).sum();
                    assert_eq!(codes, 65535, "weights {w:?}");
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
    for block in bytes.as_chunks::<8>().0 {
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

    let specular = "int inputs:useSpecularWorkflow = 1";
    let a = cook("preview-specular-default", &previewed(specular, "", ""));
    let m = material(&a);
    assert_eq!(m.model, Material::MODEL_PREVIEW_SPECULAR);
    assert_eq!(m.specular_color, [0.0; 3]);
    assert_eq!(m.specular_ior, 1.5, "ior stays for refraction");

    let a = cook(
        "preview-specular-gray",
        &previewed(
            &format!("{specular}\n color3f inputs:specularColor = (0.5, 0.5, 0.5)"),
            "",
            "",
        ),
    );
    assert_eq!(material(&a).specular_color, [0.5; 3]);
}

/// The vertex color stream of the stage's mesh, one entry per vertex.
fn vertex_colors(a: &Archive) -> Vec<[f32; 4]> {
    let e = a.find(AssetId::from_name("u/mesh")).unwrap();
    let blobs = a.blobs(e);
    let read = |b: &Blob| {
        let mut v = vec![0; b.raw_len as usize];
        a.read(b, &mut v).unwrap();
        v
    };
    let info = MeshInfo::parse(&read(&blobs[0])).unwrap();
    let mut out = Vec::new();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        orr_pack::mesh::decode_page(&read(&blobs[p.blob as usize]), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            out.extend((0..u32::from(c.vertex_count)).map(|v| view.color(&c, v)));
        }
    }
    out
}

/// PS primvar reader: "(potentially) surface-varying data". A varying primvar takes channels of
/// the vertex stream, by its interpolation, and `opacity` under `transparent` reads its complement.
#[test]
fn varying_primvars_read_through_the_vertex_stream() {
    let reader = |name: &str, kind: &str, primvar: &str, output: &str| {
        format!(
            r#"def Shader "{name}"
    {{
        uniform token info:id = "UsdPrimvarReader_{kind}"
        string inputs:varname = "{primvar}"
        {output} outputs:result
    }}"#
        )
    };
    let inputs = "color3f inputs:diffuseColor.connect = </M/C.outputs:result>
        float inputs:opacity.connect = </M/O.outputs:result>";
    let prims = format!(
        "{}\n{}",
        reader("C", "float3", "displayColor", "float3"),
        reader("O", "float", "displayOpacity", "float")
    );
    let mesh = r#"color3f[] primvars:displayColor = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] (
        interpolation = "vertex"
    )
    float[] primvars:displayOpacity = [0.25] (
        interpolation = "uniform"
    )"#;
    let a = cook("reader-varying", &previewed(inputs, &prims, mesh));
    let m = material(&a);
    assert_eq!(m.base_color, [1.0; 3]);
    assert_eq!(m.textures[slot::BASE_COLOR].vertex, 1);
    assert_eq!(m.textures[slot::TRANSMISSION].vertex, 4);
    assert_ne!(m.textures[slot::TRANSMISSION].vertex_complement, 0);
    let colors = vertex_colors(&a);
    assert_eq!(colors.len(), 3);
    assert!(colors.contains(&[1.0, 0.0, 0.0, 0.25]), "{colors:?}");
    assert!(colors.contains(&[0.0, 0.0, 1.0, 0.25]), "{colors:?}");
}

/// PS `UsdPrimvarReader`: "fallback value to be returned if geometry fetch failed", which holds
/// for an element a short array doesn't reach as for a primvar the mesh doesn't have, and opacity
/// reads it as it reads any other value.
#[test]
fn a_short_primvar_reads_the_readers_fallback() {
    let d = dir("reader-short");
    let inputs = "color3f inputs:diffuseColor.connect = </M/C.outputs:result>
        float inputs:opacity.connect = </M/O.outputs:result>";
    let prims = r#"def Shader "C"
    {
        uniform token info:id = "UsdPrimvarReader_float3"
        string inputs:varname = "tint"
        float3 inputs:fallback = (0.5, 0.25, 0.125)
        float3 outputs:result
    }
    def Shader "O"
    {
        uniform token info:id = "UsdPrimvarReader_float"
        string inputs:varname = "see"
        float inputs:fallback = 0.75
        float outputs:result
    }"#;
    let mesh = r#"color3f[] primvars:tint = [(1, 0, 0)] (
        interpolation = "vertex"
    )
    float[] primvars:see = [0.25] (
        interpolation = "vertex"
    )"#;
    let stage = previewed(inputs, prims, mesh);
    let (a, warnings) = cook_with_warnings(&d, &stage, Subdivision::Hull);
    let colors = vertex_colors(&a);
    assert!(colors.contains(&[1.0, 0.0, 0.0, 0.25]), "{colors:?}");
    assert!(colors.contains(&[0.5, 0.25, 0.125, 0.75]), "{colors:?}");
    let says = |name: &str| {
        warnings
            .iter()
            .any(|w| w.contains(name) && w.contains("2 elements"))
    };
    assert!(says("primvar tint") && says("primvar see"), "{warnings:?}");
}

/// `primvar.h` `elementSize`: "how many consecutive elements in the value array should be taken
/// as an atomic element", so a vertex's value starts a whole element on, however wide, and a
/// reader takes the first of it.
#[test]
fn wide_primvars_read_at_their_own_stride() {
    let prims = r#"def Shader "C"
    {
        uniform token info:id = "UsdPrimvarReader_float4"
        string inputs:varname = "wide"
        float4 outputs:result
    }"#;
    let inputs = "color3f inputs:diffuseColor.connect = </M/C.outputs:result>";
    let colors =
        |name: &str, primvar: &str| vertex_colors(&cook(name, &previewed(inputs, prims, primvar)));
    let held = |c: &[[f32; 4]], first: f32| {
        assert!(
            c.contains(&[first, first + 1.0, first + 2.0, first + 3.0]),
            "{c:?}"
        );
    };
    let eight = colors(
        "primvar-eight",
        r#"float[] primvars:wide = [
        0, 1, 2, 3, 4, 5, 6, 7,
        10, 11, 12, 13, 14, 15, 16, 17,
        20, 21, 22, 23, 24, 25, 26, 27
    ] (
        elementSize = 8
        interpolation = "vertex"
    )"#,
    );
    held(&eight, 10.0);
    held(&eight, 20.0);
    // 18 numbers are no whole number of four-wide values.
    let six = colors(
        "primvar-six",
        r#"float[] primvars:wide = [
        0, 1, 2, 3, 4, 5,
        10, 11, 12, 13, 14, 15,
        20, 21, 22, 23, 24, 25
    ] (
        elementSize = 6
        interpolation = "vertex"
    )"#,
    );
    held(&six, 10.0);
    held(&six, 20.0);
    let matrices = colors(
        "primvar-matrix",
        r#"matrix4d[] primvars:wide = [
        ((0, 1, 2, 3), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        ((10, 11, 12, 13), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        ((20, 21, 22, 23), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    ] (
        interpolation = "vertex"
    )"#,
    );
    held(&matrices, 10.0);
    held(&matrices, 20.0);
}

/// `_ComputeFlattenedHelper`: an index is invalid where `(indices[i] + 1) * elementSize` passes
/// the array, so an array that stops within an element holds the elements before it.
#[test]
fn a_partial_last_element_reads_the_fallback() {
    let d = dir("primvar-ragged");
    let prims = r#"def Shader "C"
    {
        uniform token info:id = "UsdPrimvarReader_float2"
        string inputs:varname = "pair"
        float2 inputs:fallback = (0.5, 0.25)
        float2 outputs:result
    }"#;
    let inputs = "float inputs:roughness.connect = </M/C.outputs:result>";
    let mesh = r#"float[] primvars:pair = [1, 2, 3] (
        elementSize = 2
        interpolation = "vertex"
    )"#;
    let stage = previewed(inputs, prims, mesh);
    let (a, warnings) = cook_with_warnings(&d, &stage, Subdivision::Hull);
    let colors = vertex_colors(&a);
    assert!(colors.contains(&[1.0, 2.0, 1.0, 1.0]), "{colors:?}");
    assert!(colors.contains(&[0.5, 0.25, 1.0, 1.0]), "{colors:?}");
    let says = |w: &&String| w.contains("primvar pair") && w.contains("2 elements");
    assert!(warnings.iter().any(|w| says(&w)), "{warnings:?}");
}

/// OpenPBR `emission_color`: "permitted to have arbitrarily large (positive) components", in a
/// range of `[0, inf)` past what the stream's half floats hold, which then hold them over a power
/// of two the field gives back. Opacity past it stays exact below 1, its range, by a scale of the
/// complement's own.
#[test]
fn primvars_past_half_floats_fold_into_the_field() {
    let reader = |name: &str, kind: &str, primvar: &str, ty: &str| {
        format!(
            r#"def Shader "{name}"
    {{
        uniform token info:id = "UsdPrimvarReader_{kind}"
        string inputs:varname = "{primvar}"
        {ty} outputs:result
    }}"#
        )
    };
    let glow = reader("C", "float3", "glow", "float3");
    let inputs = "color3f inputs:emissiveColor.connect = </M/C.outputs:result>";
    let mesh = |peak: &str| {
        format!(
            r#"color3f[] primvars:glow = [({peak}, 3, 0), (1, 0.5, 0), (0.25, 0.125, 0)] (
        interpolation = "vertex"
    )"#
        )
    };
    let a = cook(
        "primvar-past-half",
        &previewed(inputs, &glow, &mesh("100032")),
    );
    let m = material(&a);
    assert_eq!(m.emission_color, [2.0; 3]);
    let colors = vertex_colors(&a);
    assert!(colors.contains(&[50016.0, 1.5, 0.0, 1.0]), "{colors:?}");
    assert!(colors.contains(&[0.5, 0.25, 0.0, 1.0]), "{colors:?}");
    assert!(colors.contains(&[0.125, 0.0625, 0.0, 1.0]), "{colors:?}");

    let a = cook("primvar-in-half", &previewed(inputs, &glow, &mesh("60000")));
    assert_eq!(material(&a).emission_color, [1.0; 3]);

    let see = reader("O", "float", "see", "float");
    let inputs = "float inputs:opacity.connect = </M/O.outputs:result>";
    let mesh = r#"float[] primvars:see = [0.5, 1, 1000000] (
        interpolation = "vertex"
    )"#;
    let a = cook("primvar-opacity", &previewed(inputs, &see, mesh));
    let m = material(&a);
    assert_eq!(m.transmission_weight, 1.0);
    assert_eq!(m.textures[slot::TRANSMISSION].vertex_complement, 5);
    let colors = vertex_colors(&a);
    assert!(colors.contains(&[0.03125, 1.0, 1.0, 1.0]), "{colors:?}");
}

/// The stream's values are half floats, so a value no power of two brings within them, past an
/// `f32` field's reach, fails the cook rather than becoming infinity.
#[test]
fn a_primvar_past_f32_fails_the_cook() {
    let d = dir("primvar-past-f32");
    let prims = r#"def Shader "C"
    {
        uniform token info:id = "UsdPrimvarReader_float"
        string inputs:varname = "huge"
        float outputs:result
    }"#;
    let inputs = "float inputs:roughness.connect = </M/C.outputs:result>";
    let mesh = r#"double[] primvars:huge = [1e300, 1, 1] (
        interpolation = "vertex"
    )"#;
    let body = previewed(inputs, prims, mesh);
    let err = cook_at(&d, &body, UsdSettings::default()).err();
    assert!(err.is_some_and(|e| e.to_string().contains("vertex")));
}

/// Two readers of one primvar each keep their own `fallback`.
#[test]
fn readers_of_a_primvar_keep_their_fallbacks() {
    let reader = |name: &str, fallback: f32| {
        format!(
            r#"def Shader "{name}"
    {{
        uniform token info:id = "UsdPrimvarReader_float"
        string inputs:varname = "w"
        float inputs:fallback = {fallback}
        float outputs:result
    }}"#
        )
    };
    let inputs = "float inputs:roughness.connect = </M/R.outputs:result>
        float inputs:metallic.connect = </M/N.outputs:result>";
    let prims = format!("{}\n{}", reader("R", 0.5), reader("N", 0.25));
    let mesh = r#"float[] primvars:w = [1] (
        interpolation = "vertex"
    )"#;
    let a = cook("reader-fallbacks", &previewed(inputs, &prims, mesh));
    let m = material(&a);
    assert_eq!(m.textures[slot::METALNESS].vertex, 1);
    assert_eq!(m.textures[slot::ROUGHNESS].vertex, 2);
    let colors = vertex_colors(&a);
    assert!(colors.contains(&[0.25, 0.5, 1.0, 1.0]), "{colors:?}");
}

/// MaterialX `geomcolor`: `index` is the "index of the geometric color", `HwGeomColorNode` reads
/// each from a stream of its own, and `out` defaults to zero. A mesh holds the one.
#[test]
fn geomcolor_past_the_first_reads_zero() {
    let m = |name: &str, index: u8| {
        let prims = format!(
            r#"def Shader "C"
    {{
        uniform token info:id = "ND_geomcolor_color3"
        int inputs:index = {index}
        color3f outputs:out
    }}"#
        );
        let inputs = "color3f inputs:diffuseColor.connect = </M/C.outputs:out>";
        let mesh = r#"color3f[] primvars:displayColor = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] (
        interpolation = "vertex"
    )"#;
        material(&cook(name, &previewed(inputs, &prims, mesh)))
    };
    let first = m("geomcolor-0", 0);
    assert_eq!(first.textures[slot::BASE_COLOR].vertex, 1);
    assert_eq!(first.base_color, [1.0; 3]);
    let other = m("geomcolor-7", 7);
    assert_eq!(other.textures[slot::BASE_COLOR].vertex, 0);
    assert_eq!(other.base_color, [0.0; 3]);
}

/// PS inputs are connectable: `specularColor`, `ior` (for the clearcoat too) and
/// `clearcoatRoughness` read textures.
#[test]
fn preview_inputs_read_textures_with_slots_of_their_own() {
    let inputs = "int inputs:useSpecularWorkflow = 1
        color3f inputs:specularColor.connect = </M/T.outputs:rgb>
        float inputs:ior.connect = </M/I.outputs:r>
        float inputs:clearcoat = 1
        float inputs:clearcoatRoughness.connect = </M/T.outputs:g>";
    let prims = format!(
        "{}\n{}\n{READER}",
        texture("T", RAW),
        texture(
            "I",
            &format!(
                "{RAW}\n float4 inputs:scale = (2, 2, 2, 1)\n float4 inputs:bias = (1, 1, 1, 0)"
            )
        )
    );
    let m = material(&shaded(
        "preview-textured",
        &previewed(inputs, &prims, ST),
        &gradient(),
    ));
    let bound = |s: usize| m.textures[s].texture != AssetId::NONE;
    assert!(bound(slot::SPECULAR_COLOR));
    assert!(bound(slot::COAT_ROUGHNESS));
    assert!(bound(slot::SPECULAR_IOR) && bound(slot::COAT_IOR));
    assert_eq!(m.model, Material::MODEL_PREVIEW_SPECULAR);

    // PS `ior`: "including the clearcoat if clearcoat > 0", as it varies over the mesh too.
    let reader = r#"def Shader "I"
    {
        uniform token info:id = "UsdPrimvarReader_float"
        string inputs:varname = "eta"
        float outputs:result
    }"#;
    let inputs = "float inputs:ior.connect = </M/I.outputs:result>
        float inputs:clearcoat = 1";
    let mesh = r#"float[] primvars:eta = [1.25, 1.5, 2] (
        interpolation = "vertex"
    )"#;
    let m = material(&cook(
        "preview-varying-ior",
        &previewed(inputs, reader, mesh),
    ));
    assert_eq!((m.specular_ior, m.coat_ior), (1.0, 1.0));
    assert_eq!(m.textures[slot::SPECULAR_IOR].vertex, 1);
    assert_eq!(m.textures[slot::COAT_IOR].vertex, 1);
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

/// USD's text of `v`, one tuple per element.
fn usd_list<const N: usize>(v: impl IntoIterator<Item = [f32; N]>) -> String {
    let tuples: Vec<String> = v
        .into_iter()
        .map(|t| format!("({})", t.map(|x| x.to_string()).join(", ")))
        .collect();
    format!("[{}]", tuples.join(", "))
}

/// `NormalTangentMirrorTest`'s quads as mesh `Q`, its uvs in `st` and `reversed` winding them
/// left-handed, with `mesh` attributes, bound to a normal map whose `Material` holds `material`.
fn framed(name: &str, mesh: &str, material: &str, reversed: bool) -> Vec<(Vec3, [f32; 4])> {
    let (p, n, uv, _, idx) = tangent_quads::arrays();
    let idx: Vec<u32> = idx
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|t| match reversed {
            true => [t[0], t[2], t[1]],
            false => *t,
        })
        .collect();
    let counts = vec!["3"; idx.len() / 3].join(", ");
    let indices = idx
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let (points, normals) = (usd_list(p), usd_list(n));
    let st = usd_list(uv.iter().map(|&[u, v]| [u, 1.0 - v]));
    let orientation = if reversed {
        r#"uniform token orientation = "leftHanded""#
    } else {
        ""
    };
    let normal_map = texture(
        "T",
        &format!(
            "{RAW}\n float4 inputs:scale = (2, 2, 2, 1)\n float4 inputs:bias = (-1, -1, -1, 0)"
        ),
    );
    let body = format!(
        r#"def Mesh "Q"
{{
    uniform token subdivisionScheme = "none"
    {orientation}
    int[] faceVertexCounts = [{counts}]
    int[] faceVertexIndices = [{indices}]
    point3f[] points = {points}
    normal3f[] normals = {normals} (
        interpolation = "vertex"
    )
    texCoord2f[] primvars:st = {st} (
        interpolation = "vertex"
    )
    rel material:binding = </M>
    {mesh}
}}

def Material "M"
{{
    token outputs:surface.connect = </M/S.outputs:surface>
    {material}

    def Shader "S"
    {{
        uniform token info:id = "UsdPreviewSurface"
        normal3f inputs:normal.connect = </M/T.outputs:rgb>
        token outputs:surface
    }}
    {normal_map}
    {READER}
}}"#
    );
    let d = dir(name);
    let tilted = png(16, 16, |x, y| {
        [96 + (x * 4) as u8, 160 - (y * 4) as u8, 255, 255]
    });
    std::fs::write(d.join("tex.png"), tilted).unwrap();
    let mut settings = UsdSettings::default();
    settings.mesh.tangent_error_max = None;
    let a = cook_at(&d, &body, settings).unwrap_or_else(|e| panic!("{name}: {e}"));
    corner_tangents(&a, "u/mesh")
}

/// Position and tangent of every corner of a mesh entry.
fn corner_tangents(a: &Archive, name: &str) -> Vec<(Vec3, [f32; 4])> {
    let info = MeshInfo::parse(&blob(a, name, 0)).unwrap();
    let mut out = Vec::new();
    for p in &info.pages {
        let mut page = vec![0; p.resident_len as usize];
        orr_pack::mesh::decode_page(&blob(a, name, p.blob as usize), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            for v in 0..u32::from(c.vertex_count) {
                let at = Vec3::from(view.position(&c, v));
                out.push((at, view.tangent(&c, v).expect("tangents")));
            }
        }
    }
    out
}

fn assert_tangents(got: &[(Vec3, [f32; 4])], want: impl Fn(Vec3) -> Option<[f32; 4]>) {
    assert!(!got.is_empty());
    for (at, t) in got {
        let want = want(*at).unwrap_or_else(|| panic!("no tangent wanted at {at}"));
        let dir = Vec3::from_slice(&want).normalize();
        assert!(
            Vec3::from_slice(t).dot(dir) > 0.9999 && t[3] == want[3],
            "at {at}: {t:?} want {want:?}"
        );
    }
}

/// The tangent `NormalTangentMirrorTest` stores at `at`, with `place` taking its positions and
/// tangents where the model puts them.
fn published(at: Vec3, place: impl Fn(Vec3) -> Vec3) -> Option<[f32; 4]> {
    let (p, _, _, t, _) = tangent_quads::arrays();
    let k = p
        .iter()
        .position(|&q| place(q.into()).abs_diff_eq(at, 2e-3))?;
    let v = place(Vec3::from_slice(&t[k]));
    Some([v.x, v.y, v.z, t[k][3]])
}

/// PS `frame:stPrimvarName`: "Mikktspace or similar is recommended for consistency". The
/// tangents `NormalTangentMirrorTest` publishes, from uvs USD flips, left-handed winding included.
#[test]
fn unstated_tangents_are_mikktspaces_on_either_winding() {
    let want = |at| published(at, |p| p);
    assert_tangents(&framed("frame-mikk", "", "", false), want);
    assert_tangents(&framed("frame-left", "", "", true), want);
}

/// A mesh's own space holds its tangents, and a transform reaches them as it reaches `points`:
/// a mirror turns them with the surface and negates their sign.
#[test]
fn a_mirroring_transform_negates_the_tangent_sign() {
    let mirrored = r#"double3 xformOp:scale = (-1, 1, 1)
    uniform token[] xformOpOrder = ["xformOp:scale"]"#;
    let flip = |v: Vec3| v * Vec3::new(-1.0, 1.0, 1.0);
    let got = framed("frame-mirror", mirrored, "", false);
    assert_tangents(&got, |at| {
        published(at, flip).map(|[x, y, z, w]| [x, y, z, -w])
    });
}

/// PS `frame:tangentsPrimvarName`, default "tangents", and `frame:binormalsPrimvarName`, default
/// "binormals": the file's frame stands over MikkTSpace, and the binormal gives its handedness.
/// A stage of a two-triangle mesh on `Plain`, whose subset binds faces `faces` to normal-mapped
/// `M`, and a plain triangle `Other` beside it; `M` says `frame` and the mesh holds `mesh`.
fn subset_stage(faces: &str, frame: &str, mesh: &str) -> String {
    let normal_map = texture("T", RAW);
    format!(
        r#"def Mesh "Two"
{{
    uniform token subdivisionScheme = "{mesh}"
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 2, 1, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1), (1, 1)] (
        interpolation = "vertex"
    )
    rel material:binding = </Plain>

    def GeomSubset "Mapped"
    {{
        uniform token elementType = "face"
        uniform token familyName = "materialBind"
        int[] indices = [{faces}]
        rel material:binding = </M>
    }}
}}

def Mesh "Other"
{{
    {TRIANGLE}
    rel material:binding = </Plain>
}}

def Material "Plain"
{{
    token outputs:surface.connect = </Plain/S.outputs:surface>

    def Shader "S"
    {{
        uniform token info:id = "UsdPreviewSurface"
        token outputs:surface
    }}
}}

def Material "M"
{{
    token outputs:surface.connect = </M/S.outputs:surface>
    {frame}

    def Shader "S"
    {{
        uniform token info:id = "UsdPreviewSurface"
        normal3f inputs:normal.connect = </M/T.outputs:rgb>
        token outputs:surface
    }}
    {normal_map}
    {READER}
}}"#
    )
}

/// The cook of `stage`, keeping its tangents, and its warnings.
fn cook_with_warnings(
    d: &std::path::Path,
    stage: &str,
    subdivision: Subdivision,
) -> (Archive, Vec<String>) {
    let path = d.join("stage.usda");
    std::fs::write(&path, format!("{HEAD}\n{stage}")).unwrap();
    std::fs::write(d.join("tex.png"), png(4, 4, |_, _| [96, 160, 255, 255])).unwrap();
    let mut settings = UsdSettings::default();
    settings.mesh.tangent_error_max = None;
    settings.subdivision = subdivision;
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "u".into(),
                source: Source::Usd { path, settings },
            }],
        )],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
    let warnings = cooked
        .report
        .assets
        .iter()
        .flat_map(|a| {
            a.warnings
                .iter()
                .chain(a.entries.iter().flat_map(|e| &e.warnings))
        })
        .cloned();
    let warnings = warnings.collect();
    (
        Archive::from_memory(cooked.memory.remove(0)).unwrap(),
        warnings,
    )
}

/// A material only a subset binds reads tangents, which every mesh of its model then carries.
#[test]
fn a_subsets_material_gives_its_model_tangents() {
    let d = dir("subset-tangents");
    let (a, _) = cook_with_warnings(&d, &subset_stage("1", "", "none"), Subdivision::Hull);
    assert!(!corner_tangents(&a, "u/mesh").is_empty());
}

/// A frame on uvs a mesh doesn't author warns for a material a face draws with, and only then.
#[test]
fn a_frame_on_unauthored_uvs_warns_for_a_material_a_face_uses() {
    let d = dir("subset-frame-warning");
    let frame = r#"token inputs:frame:stPrimvarName = "elsewhere""#;
    let warned = |faces: &str| {
        let stage = subset_stage(faces, frame, "none");
        let (_, warnings) = cook_with_warnings(&d, &stage, Subdivision::Hull);
        let says = |w: &&String| w.contains("tangents follow elsewhere");
        warnings.iter().filter(says).count()
    };
    assert_eq!(warned("1"), 1);
    assert_eq!(warned(""), 0, "no face draws with it");
}

/// Stated tangents on a cage warn, whatever else the model holds.
#[test]
fn tangents_on_a_cage_warn_that_they_are_not_read() {
    let d = dir("cage-tangents-warning");
    let stage = subset_stage("1", "", "catmullClark").replace(
        "rel material:binding = </Plain>\n\n    def GeomSubset",
        "vector3f[] primvars:tangents = [(0, 1, 0)] (\n        interpolation = \"constant\"\n    \
         )\n    rel material:binding = </Plain>\n\n    def GeomSubset",
    );
    let (_, warnings) = cook_with_warnings(&d, &stage, Subdivision::Cage);
    let says = |w: &&String| w.contains("on a subdivision cage is not read");
    assert_eq!(warnings.iter().filter(says).count(), 2, "{warnings:?}");
}

/// A tangent's declared type says how wide it is: four components hold a bitangent sign last,
/// also where their count divides by three.
#[test]
fn four_component_tangents_state_their_sign() {
    for w in [1.0f32, -1.0] {
        let want = |_: Vec3| Some([0.0, 1.0, 0.0, w]);
        let each = vec![format!("(0, 1, 0, {w})"); 12].join(", ");
        for tangents in [
            format!(
                "float4[] primvars:tangents = [(0, 1, 0, {w})] (\n interpolation = \"constant\"\n)"
            ),
            format!("float4[] primvars:tangents = [{each}] (\n interpolation = \"vertex\"\n)"),
            format!(
                "float[] primvars:tangents = [0, 1, 0, {w}] (\n elementSize = 4\n interpolation = \"constant\"\n)"
            ),
        ] {
            assert_tangents(&framed("frame-four", &tangents, "", false), want);
        }
    }
}

#[test]
fn frame_primvars_state_tangents_and_their_handedness() {
    let primvars = |tangents: &str, binormals: &str, y: f32| {
        format!(
            r#"vector3f[] primvars:{tangents} = [(0, 1, 0)] (
        interpolation = "constant"
    )
    vector3f[] primvars:{binormals} = [({y}, 0, 0)] (
        interpolation = "constant"
    )"#
        )
    };
    // The third quad faces -z, where cross(n, t) runs +x.
    let z = |at: Vec3| at.z < -0.005;
    let want = |w: f32| move |at: Vec3| Some([0.0, 1.0, 0.0, if z(at) { -w } else { w }]);
    let got = framed(
        "frame-stated",
        &primvars("tangents", "binormals", -1.0),
        "",
        false,
    );
    assert_tangents(&got, want(1.0));
    let got = framed(
        "frame-left-stated",
        &primvars("tangents", "binormals", 1.0),
        "",
        false,
    );
    assert_tangents(&got, want(-1.0));

    let names = r#"string inputs:frame:tangentsPrimvarName = "t2"
    string inputs:frame:binormalsPrimvarName = "b2""#;
    assert_tangents(
        &framed("frame-named", &primvars("t2", "b2", -1.0), names, false),
        want(1.0),
    );
}

/// PS `frame:stPrimvarName`: "Name of the texture coordinate to be use to calculate the tangent
/// frame".
#[test]
fn the_frames_st_primvar_can_differ_from_the_textures() {
    let (_, _, uv, _, _) = tangent_quads::arrays();
    let st = usd_list(uv.iter().map(|&[u, v]| [1.0 - u, 1.0 - v]));
    let mirrored = format!(
        r#"texCoord2f[] primvars:frame_st = {st} (
        interpolation = "vertex"
    )"#
    );
    let material = r#"string inputs:frame:stPrimvarName = "frame_st""#;
    let got = framed("frame-st", &mirrored, material, false);
    assert_tangents(&got, |at| {
        published(at, |p| p).map(|[x, y, z, w]| [-x, -y, -z, -w])
    });
    // A mesh without the named primvar falls back on the first set.
    let got = framed("frame-st-absent", "", material, false);
    assert_tangents(&got, |at| published(at, |p| p));
}

/// MaterialX `tangent`: "The index of the texcoord space to use to compute the tangent vector",
/// through a `normalmap`'s `tangent` input; by default `Tworld`, index 0, whichever set the image
/// reads.
#[test]
fn a_normalmaps_tangent_node_picks_the_uv_set() {
    let (p, n, uv, _, idx) = tangent_quads::arrays();
    let counts = vec!["3"; idx.len() / 3].join(", ");
    let indices = idx
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let (points, normals) = (usd_list(p), usd_list(n));
    let st = usd_list(uv.iter().map(|&[u, v]| [u, 1.0 - v]));
    let other = usd_list(uv.iter().map(|&[u, v]| [1.0 - u, 1.0 - v]));
    let stage = |tangent: &str, input: &str| {
        format!(
            r#"def Mesh "Q"
{{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [{counts}]
    int[] faceVertexIndices = [{indices}]
    point3f[] points = {points}
    normal3f[] normals = {normals} (
        interpolation = "vertex"
    )
    texCoord2f[] primvars:st = {st} (
        interpolation = "vertex"
    )
    texCoord2f[] primvars:st1 = {other} (
        interpolation = "vertex"
    )
    rel material:binding = </M>
}}

def Material "M"
{{
    token outputs:mtlx:surface.connect = </M/S.outputs:out>

    def Shader "S"
    {{
        uniform token info:id = "ND_open_pbr_surface_surfaceshader"
        vector3f inputs:geometry_normal.connect = </M/N.outputs:out>
        token outputs:out
    }}
    def Shader "N"
    {{
        uniform token info:id = "ND_normalmap"
        vector3f inputs:in.connect = </M/I.outputs:out>
        {input}
        vector3f outputs:out
    }}
    def Shader "I"
    {{
        uniform token info:id = "ND_image_vector3"
        asset inputs:file = @tex.png@
        vector3f outputs:out
    }}
    {tangent}
}}"#
        )
    };
    let tilted = png(16, 16, |x, y| {
        [96 + (x * 4) as u8, 160 - (y * 4) as u8, 255, 255]
    });
    let run = |name: &str, body: &str| {
        let d = dir(name);
        std::fs::write(d.join("tex.png"), &tilted).unwrap();
        let mut settings = UsdSettings {
            uv1: Some("st1".into()),
            ..UsdSettings::default()
        };
        settings.mesh.tangent_error_max = None;
        let a = cook_at(&d, body, settings).unwrap_or_else(|e| panic!("{name}: {e}"));
        corner_tangents(&a, "u/mesh")
    };
    assert_tangents(&run("mtlx-tworld", &stage("", "")), |at| {
        published(at, |p| p)
    });
    let node = r#"def Shader "T"
    {
        uniform token info:id = "ND_tangent_vector3"
        int inputs:index = 1
        vector3f outputs:out
    }"#;
    let input = "vector3f inputs:tangent.connect = </M/T.outputs:out>";
    assert_tangents(&run("mtlx-tangent", &stage(node, input)), |at| {
        published(at, |p| p).map(|[x, y, z, w]| [-x, -y, -z, -w])
    });
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

/// `ND_extract_matrix33` reads a `matrix33` and gives a `vector3`, so its id names its input where
/// other ids name their output, and a `matrix3d` value holds the matrix.
#[test]
fn a_matrix_row_extracts_through_node_definition_ids() {
    let inputs = "float inputs:specular_roughness.connect = </M/E.outputs:out>";
    let prims = format!(
        r#"{}
    def Shader "R"
    {{
        uniform token info:id = "ND_extract_matrix33"
        matrix3d inputs:in = ((1, 2, 3), (0.4, 0.5, 0.6), (7, 8, 9))
        int inputs:index = 1
        vector3f outputs:out
    }}
    def Shader "E"
    {{
        uniform token info:id = "ND_extract_vector3"
        vector3f inputs:in.connect = </M/R.outputs:out>
        int inputs:index = 2
        float outputs:out
    }}"#,
        open_pbr(inputs)
    );
    let surface = "token outputs:mtlx:surface.connect = </M/S.outputs:out>";
    let body = mtlx_material(surface, &prims);
    let m = material(&cook_centimeters("matrix-row", &body, &[]));
    assert!(near(m.specular_roughness, 0.6), "{}", m.specular_roughness);
}

/// OpenPBR Metadata: "If no unit is given, the length is assumed to be in centimeters", and
/// `thin_film_thickness` "is always in micrometers". A stage's `metersPerUnit` is the unit of its
/// geometry, not of these parameters.
#[test]
fn open_pbr_lengths_read_in_centimeters_on_any_stage() {
    let surface = "token outputs:mtlx:surface.connect = </M/S.outputs:out>";
    let inputs = "float inputs:transmission_depth = 2
        float inputs:thin_film_weight = 1
        float inputs:thin_film_thickness = 0.5";
    let body = mtlx_material(surface, &open_pbr(inputs));
    for m in [
        material(&cook("meter-stage-lengths", &body)),
        material(&cook_centimeters("stage-lengths", &body, &[])),
    ] {
        assert!(near(m.transmission_depth, 0.02), "{}", m.transmission_depth);
        assert_eq!(m.thin_film_thickness, 500.0);
        assert!(near(m.subsurface_radius, 0.01), "{}", m.subsurface_radius);
    }

    let body = mtlx_material(surface, &open_pbr("float inputs:subsurface_radius = 3"));
    let m = material(&cook("meter-stage-radius", &body));
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

/// PS `fallback`: "Fallback value used when texture can not be read", default (0, 0, 0, 1), with
/// `output = textureValue * scale + bias`; Storm's accessor applies scale and bias to it as well.
#[test]
fn a_texture_that_can_not_be_read_reads_its_fallback() {
    let unread = |inputs: &str, wired: &str, image: Option<&[u8]>| {
        let d = dir("fallback");
        if let Some(bytes) = image {
            std::fs::write(d.join("tex.png"), bytes).unwrap();
        }
        let prims = format!("{}\n{READER}", texture("T", inputs));
        let body = previewed(wired, &prims, ST);
        material(&cook_at(&d, &body, UsdSettings::default()).unwrap())
    };
    const FALLBACK: &str = "float4 inputs:fallback = (0.25, 0.5, 0.75, 0.125)";
    const DIFFUSE: &str = "color3f inputs:diffuseColor.connect = </M/T.outputs:rgb>";
    const ROUGH: &str = "float inputs:roughness.connect = </M/T.outputs:g>";

    let m = unread(FALLBACK, DIFFUSE, None);
    assert_eq!(m.textures[slot::BASE_COLOR].texture, AssetId::NONE);
    assert!(
        near(m.base_color[0], 0.25) && near(m.base_color[2], 0.75),
        "{m:?}"
    );

    // A file that resolves and doesn't decode can't be read either.
    let m = unread(FALLBACK, ROUGH, Some(b"not an image"));
    assert_eq!(m.textures[slot::ROUGHNESS].texture, AssetId::NONE);
    assert!(near(m.specular_roughness, 0.5), "{m:?}");

    let scaled = format!(
        "{FALLBACK}\nfloat4 inputs:scale = (1, 0.5, 1, 1)\nfloat4 inputs:bias = (0, 0.25, 0, 0)"
    );
    let m = unread(&scaled, ROUGH, None);
    assert!(near(m.specular_roughness, 0.5), "{m:?}");

    // A stated value past 0..1 is no texel, and reaches the scale whole.
    let past = "float4 inputs:fallback = (2, 2, 2, 1)\nfloat4 inputs:scale = (0.25, 0.25, 0.25, 1)";
    let m = unread(past, ROUGH, None);
    assert!(near(m.specular_roughness, 0.5), "{m:?}");
    assert!(near(unread(past, DIFFUSE, None).base_color[0], 0.5));

    // Stated colors aren't encoded, so a source color space leaves them be.
    let srgb = format!(
        r#"{FALLBACK}{}token inputs:sourceColorSpace = "sRGB""#,
        "\n"
    );
    let m = unread(&srgb, DIFFUSE, None);
    assert!(near(m.base_color[0], 0.25), "{m:?}");

    let m = unread("", DIFFUSE, None);
    assert_eq!(m.base_color, [0.0; 3], "the fallback's default");
    let m = unread("", "float inputs:metallic.connect = </M/T.outputs:a>", None);
    assert!(near(m.metalness, 1.0), "the default's alpha is 1: {m:?}");

    // `file`'s own default is the empty string.
    let prims = format!(
        r#"def Shader "T"
    {{
        uniform token info:id = "UsdUVTexture"
        {FALLBACK}
        float outputs:g
    }}"#
    );
    let m = material(&cook("fallback-no-file", &previewed(ROUGH, &prims, ST)));
    assert!(near(m.specular_roughness, 0.5), "{m:?}");
}

/// PS `fallback` reads only while the texture can't: the file appearing later cooks it.
#[test]
fn a_fallback_stands_only_until_the_file_appears() {
    let d = dir("fallback-late");
    let prims = format!(
        "{}\n{READER}",
        texture("T", "float4 inputs:fallback = (0, 0.5, 0, 1)")
    );
    let body = previewed(
        "float inputs:roughness.connect = </M/T.outputs:g>",
        &prims,
        ST,
    );
    let path = d.join("stage.usda");
    std::fs::write(&path, format!("{HEAD}\n{body}")).unwrap();
    let out = d.join("out.orr");
    let cook_file = || {
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                Target::File(out.clone()),
                vec![AssetSpec {
                    name: "u".into(),
                    source: Source::Usd {
                        path: path.clone(),
                        settings: UsdSettings::default(),
                    },
                }],
            )],
        };
        orr_cook::cook(&manifest, &Default::default()).unwrap();
        material(&Archive::open(&out).unwrap())
    };
    assert!(near(cook_file().specular_roughness, 0.5));
    let varying = png(8, 8, |x, y| [(x * 32) as u8, (y * 32) as u8, 0, 255]);
    std::fs::write(d.join("tex.png"), varying).unwrap();
    assert_ne!(cook_file().textures[slot::ROUGHNESS].texture, AssetId::NONE);
}

/// A preview surface reading `tex.<UDIM>.png` in `d`, its `inputs` on the texture and `wired` on
/// the surface.
fn udim_stage(inputs: &str, wired: &str) -> String {
    let texture = texture("T", inputs).replace("@tex.png@", "@tex.<UDIM>.png@");
    previewed(wired, &format!("{texture}\n{READER}"), ST)
}

/// The tiles of the set slot `s` binds, as cell and mip 0 size.
fn tiles_of(a: &Archive, m: &Material, s: usize) -> Vec<([u16; 2], [u32; 2])> {
    let e = a.find(m.textures[s].texture).expect("bound");
    assert_eq!(e.class, Class::TextureTiles as u16);
    let set = TextureTiles::parse(&read(a, &a.blobs(e)[0])).unwrap();
    set.tiles
        .iter()
        .map(|t| {
            let e = a.find(t.texture).expect("a tile cooks");
            let h = TextureInfo::parse(&read(a, &a.blobs(e)[0])).unwrap().header;
            ([t.column, t.row], [h.width, h.height])
        })
        .collect()
}

/// PS `file`: "Mari UDIM substitution in *file* values uses the "<UDIM>" token"; the UDIM Tiling
/// Constraints: "a maximum of ten tiles in the U direction", "within the range [1001, 1100]".
/// Tile 1012 is u 1..2, v 1..2, and tiles keep their own sizes. A point on no tile names a file
/// that isn't there, so it reads `fallback`, through `scale` and `bias`.
#[test]
fn a_udim_file_reads_as_tiles_across_st() {
    let d = dir("udim");
    let varying = |w: u32| png(w, w, move |x, y| [(x * 8) as u8, (y * 8) as u8, 0, 255]);
    for (name, size) in [
        ("1001", 8),
        ("1002", 16),
        ("1012", 4),
        ("1101", 8),
        ("1000", 8),
    ] {
        std::fs::write(d.join(format!("tex.{name}.png")), varying(size)).unwrap();
    }
    let inputs = format!(
        "{RAW}\nfloat4 inputs:fallback = (0.25, 0.5, 0.75, 1)\n\
         float4 inputs:scale = (0.5, 0.5, 0.5, 1)\nfloat4 inputs:bias = (0.25, 0, 0, 0)"
    );
    let wired = "float inputs:roughness.connect = </M/T.outputs:r>";
    let body = udim_stage(&inputs, wired);
    let a = cook_at(&d, &body, UsdSettings::default()).unwrap();
    let m = material(&a);
    let got = tiles_of(&a, &m, slot::ROUGHNESS);
    let want = [([0, 0], [8; 2]), ([1, 0], [16; 2]), ([1, 1], [4; 2])];
    assert_eq!(got, want);
    let b = m.textures[slot::ROUGHNESS];
    assert_eq!(b.wrap, [wrap::CLAMP; 2], "no point lies past a tile's edge");
    assert!(near(b.outside[0], 0.375), "{b:?}");

    // A channel of one value on every tile still reads otherwise past them.
    std::fs::remove_file(d.join("tex.1002.png")).unwrap();
    std::fs::remove_file(d.join("tex.1012.png")).unwrap();
    let flat = png(8, 8, |_, _| [128, 128, 128, 255]);
    std::fs::write(d.join("tex.1001.png"), flat).unwrap();
    let a = cook_at(&d, &body, UsdSettings::default()).unwrap();
    assert_eq!(tiles_of(&a, &material(&a), slot::ROUGHNESS).len(), 1);
}

/// PS `fallback`, per tile: a tile that can't be read is a point's file that can't be, and with
/// none left the texture itself can't be.
#[test]
fn a_udim_set_reads_its_fallback_where_tiles_can_not_be_read() {
    let d = dir("udim-unread");
    let good = png(8, 8, |x, y| [(x * 32) as u8, (y * 32) as u8, 0, 255]);
    std::fs::write(d.join("tex.1001.png"), good).unwrap();
    std::fs::write(d.join("tex.1002.png"), b"not an image").unwrap();
    let inputs = format!("{RAW}\nfloat4 inputs:fallback = (0.25, 0.5, 0.75, 1)");
    let wired = "float inputs:roughness.connect = </M/T.outputs:g>";
    let body = udim_stage(&inputs, wired);
    let a = cook_at(&d, &body, UsdSettings::default()).unwrap();
    let m = material(&a);
    assert_eq!(tiles_of(&a, &m, slot::ROUGHNESS), [([0, 0], [8; 2])]);
    assert!(near(m.textures[slot::ROUGHNESS].outside[0], 0.5));

    std::fs::remove_file(d.join("tex.1001.png")).unwrap();
    let m = material(&cook_at(&d, &body, UsdSettings::default()).unwrap());
    assert_eq!(m.textures[slot::ROUGHNESS].texture, AssetId::NONE);
    assert!(near(m.specular_roughness, 0.5), "{m:?}");

    std::fs::remove_file(d.join("tex.1002.png")).unwrap();
    let m = material(&cook_at(&d, &body, UsdSettings::default()).unwrap());
    assert!(near(m.specular_roughness, 0.5), "no tile resolves: {m:?}");
}

/// A tile that appears later is a point's file that now resolves.
#[test]
fn a_udim_tile_appearing_cooks_the_set_again() {
    let d = dir("udim-late");
    let tile = png(8, 8, |x, y| [(x * 32) as u8, (y * 32) as u8, 0, 255]);
    std::fs::write(d.join("tex.1001.png"), &tile).unwrap();
    let body = udim_stage(RAW, "float inputs:roughness.connect = </M/T.outputs:r>");
    let path = d.join("stage.usda");
    std::fs::write(&path, format!("{HEAD}\n{body}")).unwrap();
    let out = d.join("out.orr");
    let tiles = || {
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                Target::File(out.clone()),
                vec![AssetSpec {
                    name: "u".into(),
                    source: Source::Usd {
                        path: path.clone(),
                        settings: UsdSettings::default(),
                    },
                }],
            )],
        };
        orr_cook::cook(&manifest, &Default::default()).unwrap();
        let a = Archive::open(&out).unwrap();
        tiles_of(&a, &material(&a), slot::ROUGHNESS).len()
    };
    assert_eq!(tiles(), 1);
    std::fs::write(d.join("tex.1002.png"), &tile).unwrap();
    assert_eq!(tiles(), 2);
}

/// A 16 x 16 grid of quads with `st` running 1/8 to 7/8 from `(s, t)`, under a preview surface
/// with `wired`, whose `T` reads `file` clamped.
fn grid_stage(file: &str, at: [f32; 2], wired: &str) -> String {
    grid_over(file, at, [0.125, 0.75], wired)
}

/// [`grid_stage`] with `st` running `span` from its start, for `span[1]` in all.
fn grid_over(file: &str, [s, t]: [f32; 2], span: [f32; 2], wired: &str) -> String {
    const N: usize = 16;
    let list = |f: &dyn Fn(usize, usize) -> String| {
        let at = (0..=N).flat_map(|y| (0..=N).map(move |x| (x, y)));
        at.map(|(x, y)| f(x, y)).collect::<Vec<_>>().join(", ")
    };
    let points = list(&|x, y| format!("({}, {}, 0)", x as f32 / 16.0, y as f32 / 16.0));
    let st = list(&|x, y| {
        let at = |o: f32, i: usize| o + span[0] + span[1] * i as f32 / N as f32;
        format!("({}, {})", at(s, x), at(t, y))
    });
    let quads = (0..N).flat_map(|y| (0..N).map(move |x| (x, y)));
    let indices = quads.map(|(x, y)| {
        let c = y * (N + 1) + x;
        format!("{}, {}, {}, {}", c, c + 1, c + N + 2, c + N + 1)
    });
    let indices = indices.collect::<Vec<_>>().join(", ");
    let counts = vec!["4"; N * N].join(", ");
    let texture = texture(
        "T",
        &format!("{RAW}\ntoken inputs:wrapS = \"clamp\"\ntoken inputs:wrapT = \"clamp\""),
    )
    .replace("@tex.png@", &format!("@{file}@"));
    format!(
        r#"def Mesh "Grid"
{{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [{counts}]
    int[] faceVertexIndices = [{indices}]
    point3f[] points = [{points}]
    texCoord2f[] primvars:st = [{st}] (
        interpolation = "vertex"
    )
    rel material:binding = </M>
}}

def Material "M"
{{
    token outputs:surface.connect = </M/S.outputs:surface>

    def Shader "S"
    {{
        uniform token info:id = "UsdPreviewSurface"
        {wired}
        token outputs:surface
    }}
    {texture}
    {READER}
}}
"#
    )
}

/// Every blob of the mesh `stage` cooks.
fn mesh_blobs(d: &std::path::Path, stage: &str) -> Vec<Vec<u8>> {
    let a = cook_at(d, stage, UsdSettings::default()).unwrap();
    let e = a.find(AssetId::from_name("u/mesh")).expect("a mesh");
    a.blobs(e).iter().map(|b| read(&a, b)).collect()
}

/// Rings of alpha, so a grid cell can lie on either side of a cutoff or across it.
fn rings() -> Vec<u8> {
    png(64, 64, |x, y| {
        let (x, y) = (x as f32 - 31.5, y as f32 - 31.5);
        let v = if (x * x + y * y).sqrt() % 16.0 < 8.0 {
            255
        } else {
            0
        };
        [v, v, v, 255]
    })
}

/// A tile reads clamped, so inside it the mesh cook reads what it reads of the same image
/// clamped: micromaps, opaque triangles and all.
#[test]
fn a_mesh_on_one_tile_cooks_as_on_the_same_image() {
    let d = dir("udim-mesh-opacity");
    for file in ["tex.png", "tex.1001.png"] {
        std::fs::write(d.join(file), rings()).unwrap();
    }
    let wired = r#"float inputs:opacity.connect = </M/T.outputs:r>
        float inputs:opacityThreshold = 0.5"#;
    let image = mesh_blobs(&d, &grid_stage("tex.png", [0.0, 0.0], wired));
    let tiled = mesh_blobs(&d, &grid_stage("tex.<UDIM>.png", [0.0, 0.0], wired));
    assert_eq!(image, tiled);
}

const NORMAL: &str = "normal3f inputs:normal.connect = </M/T.outputs:rgb>";

/// The tangent check of a mesh in `stage`.
fn tangent_check(d: &std::path::Path, stage: &str) -> orr_cook::TangentReport {
    let path = d.join("stage.usda");
    std::fs::write(&path, format!("{HEAD}\n{stage}")).unwrap();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "u".into(),
                source: Source::Usd {
                    path,
                    settings: UsdSettings::default(),
                },
            }],
        )],
    };
    let cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
    let mut entries = cooked.report.assets.iter().flat_map(|a| &a.entries);
    let mesh = entries.find_map(|e| e.mesh.clone()).expect("a mesh");
    mesh.tangents.expect("a mesh with tangents")
}

/// A tile's normals weigh in the tangent check as the same image's do.
#[test]
fn the_tangent_check_reads_one_tile_as_the_same_image() {
    let d = dir("udim-mesh-normals");
    let normals = png(32, 32, |x, y| {
        [64 + (x * 4) as u8, 192 - (y * 4) as u8, 255, 255]
    });
    for file in ["tex.png", "tex.1001.png"] {
        std::fs::write(d.join(file), &normals).unwrap();
    }
    let check = |file: &str| tangent_check(&d, &grid_stage(file, [0.0, 0.0], NORMAL));
    let (image, tile) = (check("tex.png"), check("tex.<UDIM>.png"));
    assert!(image.angle_p99.is_some() && image.texels > 0, "{image:?}");
    assert_eq!(
        (image.angle_p99, image.texels, image.derived),
        (tile.angle_p99, tile.texels, tile.derived)
    );
}

/// Tiles weigh by the surface they cover, not their texels: tangents turned from the uvs show
/// only under a tilted normal, here on a small tile that covers half the mesh.
#[test]
fn the_tangent_check_weighs_tiles_by_the_surface_they_cover() {
    let d = dir("udim-mesh-normal-weights");
    let flat = png(1024, 1024, |_, _| [128, 128, 255, 255]);
    let tilted = png(4, 4, |_, _| [224, 128, 200, 255]);
    std::fs::write(d.join("tex.1001.png"), flat).unwrap();
    std::fs::write(d.join("tex.1002.png"), tilted).unwrap();
    let stage = grid_stage("tex.<UDIM>.png", [0.5, 0.0], NORMAL).replace(
        "rel material:binding",
        "float3[] primvars:tangents = [(0, 1, 0)] (\n        interpolation = \"constant\"\n    )\n    \
         rel material:binding",
    );
    let check = tangent_check(&d, &stage);
    assert!(
        !check.derived && check.angle_p99.unwrap() > 1.0,
        "{check:?}"
    );
}

const HEIGHT: &str = "float inputs:displacement.connect = </M/T.outputs:r>";

/// Settings that bake heights into the mesh.
fn baking() -> UsdSettings {
    let mut s = UsdSettings::default();
    s.mesh.displace_static = Some(StaticDisplacement {
        max_edge: 0.02,
        target_error: 0.0001,
    });
    s
}

/// PS `displacement` is a world length, so a copy that scales keeps the height it states: it cooks
/// with its scale applied, and a copy that doesn't stays a placement of the model.
#[test]
fn a_baked_height_holds_under_a_placement_scale() {
    let d = dir("displaced-copies");
    let surface = previewed("float inputs:displacement = 0.25", "", "");
    let material = &surface[surface.find("def Material").unwrap()..];
    let stage = format!(
        r#"def PointInstancer "I"
{{
    rel prototypes = [</I/Protos/P>]
    int[] protoIndices = [0, 0]
    point3f[] positions = [(0, 0, 0), (5, 0, 0)]
    float3[] scales = [(1, 1, 1), (2, 2, 2)]
    def Scope "Protos"
    {{
        def Mesh "P"
        {{
            {TRIANGLE}
            rel material:binding = </M>
        }}
    }}
}}

{material}"#
    );
    // Baking for every model, or for this one alone.
    let overridden = UsdSettings {
        mesh_overrides: vec![(
            "P".into(),
            MeshPatch {
                displace_static: Some(baking().mesh.displace_static),
                ..Default::default()
            },
        )],
        ..Default::default()
    };
    for settings in [baking(), overridden] {
        let a = cook_at(&d, &stage, settings).unwrap();
        let height = |name: &str| {
            corners(&a, name)
                .iter()
                .map(|c| c.0.z)
                .fold(f32::MIN, f32::max)
        };
        assert!((height("u/P/mesh") - 0.25).abs() < 1e-3);
        assert!((height("u/P/baked/0/mesh") - 0.25).abs() < 1e-3);
    }
}

/// Nodes `displacement`: "the vector signature allows displacement in tangent/normal space using
/// (dPdu, dPdv, N) coordinates": x along u, y along t, and z along the normal.
#[test]
fn a_vector_displacement_bakes_along_the_uvs_and_the_normal() {
    let d = dir("vector-displacement");
    let outputs = "token outputs:mtlx:surface.connect = </M/S.outputs:out>
    token outputs:mtlx:displacement.connect = </M/D.outputs:out>";
    let prims = format!(
        r#"{}
    def Shader "D"
    {{
        uniform token info:id = "ND_displacement_vector3"
        vector3f inputs:displacement = (0.25, 0.5, 0.125)
        token outputs:out
    }}"#,
        open_pbr("")
    );
    let a = cook_at(&d, &mtlx_material(outputs, &prims), baking()).unwrap();
    let lowest = |axis: usize| {
        let points = corners(&a, "u/mesh");
        points.iter().map(|c| c.0[axis]).fold(f32::MAX, f32::min)
    };
    let moved = [lowest(0), lowest(1), lowest(2)];
    assert!(
        Vec3::from(moved).abs_diff_eq(Vec3::new(0.25, 0.5, 0.125), 1e-3),
        "{moved:?}"
    );
}

/// `Version.cpp` reads a 1.38 `space` of `object`, so directions in the prim's own space, and a
/// default import bakes the prim's transform into one model: the map has to take the turn too, in
/// a copy of the material that a prim standing as it was doesn't need.
#[test]
fn an_object_space_map_turns_with_the_transform_baked_in() {
    let d = dir("object-space-baked");
    std::fs::write(
        d.join("tex.png"),
        png(2, 2, |x, y| [(x * 200) as u8, (y * 200) as u8, 200, 255]),
    )
    .unwrap();
    let surface = "token outputs:mtlx:surface.connect = </M/S.outputs:out>";
    let shader = open_pbr("float3 inputs:geometry_normal.connect = </M/N.outputs:out>");
    let prims = format!(
        r#"{shader}
    def Shader "I"
    {{
        uniform token info:id = "ND_image_vector3"
        asset inputs:file = @tex.png@
        float3 outputs:out
    }}
    def Shader "N"
    {{
        uniform token info:id = "ND_normalmap"
        float3 inputs:in.connect = </M/I.outputs:out>
        string inputs:space = "object"
        float3 outputs:out
    }}"#
    );
    let material = mtlx_material(surface, &prims);
    let material = &material[material.find("def Material").unwrap()..];
    let stage = format!(
        r#"def Xform "Turned"
{{
    double3 xformOp:rotateXYZ = (0, 0, 90)
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]

    def Mesh "A"
    {{
        {TRIANGLE}
        {ST}
        rel material:binding = </M>
    }}
}}

def Mesh "B"
{{
    {TRIANGLE}
    {ST}
    rel material:binding = </M>
}}

{material}"#
    );
    let a = cook_at(&d, &stage, UsdSettings::default()).unwrap();
    let model = ModelInfo::parse(&blob(&a, "u", 0)).unwrap();
    let frames: Vec<[[f32; 3]; 3]> = model
        .materials
        .iter()
        .map(|&m| {
            let e = a.find(m).unwrap();
            Material::from_bytes(&read(&a, &a.blobs(e)[0]))
                .unwrap()
                .object_frame
        })
        .collect();
    assert_eq!(frames.len(), 2, "one material as it was, one turned");
    assert!(frames.contains(&[[0.0; 3]; 3]));
    let turned = frames.iter().find(|f| **f != [[0.0; 3]; 3]).unwrap();
    // A quarter turn about z takes x to y: rows (0, -1, 0), (1, 0, 0), (0, 0, 1).
    let near_row = |r: usize, want: [f32; 3]| {
        (0..3).all(|c| (turned[r][c] + f32::from(r == c) - want[c]).abs() < 1e-5)
    };
    assert!(
        near_row(0, [0.0, -1.0, 0.0]) && near_row(1, [1.0, 0.0, 0.0]),
        "{turned:?}"
    );
}

/// Cluster bounds and baked heights read a tile as they read the same image clamped.
#[test]
fn heights_on_one_tile_cook_as_on_the_same_image() {
    let d = dir("udim-mesh-heights");
    for file in ["tex.png", "tex.1001.png"] {
        std::fs::write(d.join(file), rings()).unwrap();
    }
    let cooked = |file: &str, s: UsdSettings| {
        let a = cook_at(&d, &grid_stage(file, [0.0, 0.0], HEIGHT), s).unwrap();
        let e = a.find(AssetId::from_name("u/mesh")).expect("a mesh");
        a.blobs(e).iter().map(|b| read(&a, b)).collect::<Vec<_>>()
    };
    for settings in [UsdSettings::default, baking] {
        assert_eq!(
            cooked("tex.png", settings()),
            cooked("tex.<UDIM>.png", settings())
        );
    }
}

/// A quad over all of a tile has vertices on its edges, which read the tile their triangles lie
/// on and not the cells past them: heights bright at the edges, against a fallback of 0, bake and
/// bound as the same image's do.
#[test]
fn a_quad_over_a_whole_tile_cooks_as_on_the_same_image() {
    let d = dir("udim-mesh-whole-tile");
    let bright = png(64, 64, |x, y| {
        let (x, y) = (x as f32 - 31.5, y as f32 - 31.5);
        let v = if (x * x + y * y).sqrt() < 8.0 { 0 } else { 255 };
        [v, v, v, 255]
    });
    for file in ["tex.png", "tex.1001.png"] {
        std::fs::write(d.join(file), &bright).unwrap();
    }
    let cooked = |file: &str, s: UsdSettings| {
        let stage = grid_over(file, [0.0, 0.0], [0.0, 1.0], HEIGHT);
        let a = cook_at(&d, &stage, s).unwrap();
        let e = a.find(AssetId::from_name("u/mesh")).expect("a mesh");
        a.blobs(e).iter().map(|b| read(&a, b)).collect::<Vec<_>>()
    };
    for settings in [UsdSettings::default, baking] {
        assert!(cooked("tex.png", settings()) == cooked("tex.<UDIM>.png", settings()));
    }
}

/// Tiles of heights 64/255 and 191/255 side by side, past which the fallback reads 0.
fn stepped(name: &str) -> PathBuf {
    let d = dir(name);
    for (file, v) in [("tex.1001.png", 64), ("tex.1002.png", 191)] {
        std::fs::write(d.join(file), png(8, 8, |_, _| [v, v, v, 255])).unwrap();
    }
    d
}

const LOW: f32 = 64.0 / 255.0;
const HIGH: f32 = 191.0 / 255.0;

/// Clusters bound what the tiles under them read, and the fallback past them: a cluster on one
/// tile holds that tile's height alone, where the declared range would hold them all.
#[test]
fn cluster_bounds_follow_the_tiles_under_them() {
    let d = stepped("udim-mesh-bounds");
    let bounds = |at: [f32; 2]| {
        let stage = grid_stage("tex.<UDIM>.png", at, HEIGHT);
        let a = cook_at(&d, &stage, UsdSettings::default()).unwrap();
        let info = MeshInfo::parse(&blob(&a, "u/mesh", 0)).unwrap();
        let half = |b: u16| half::f16::from_bits(b).to_f32();
        let mut out = Vec::new();
        for p in &info.pages {
            let mut page = vec![0; p.resident_len as usize];
            orr_pack::mesh::decode_page(&blob(&a, "u/mesh", p.blob as usize), &mut page).unwrap();
            let view = PageView::new(&page, &info.header).unwrap();
            for i in 0..view.cluster_count() {
                let c = view.cluster(i);
                out.push((half(c.displacement_min), half(c.displacement_max)));
            }
        }
        out
    };
    let near = |a: f32, b: f32| (a - b).abs() < 0.01;
    let across = bounds([0.5, 0.0]);
    assert!(
        across.iter().any(|b| near(b.0, LOW) && near(b.1, LOW)),
        "{across:?}"
    );
    assert!(
        across.iter().any(|b| near(b.0, HIGH) && near(b.1, HIGH)),
        "{across:?}"
    );
    assert!(
        across.iter().any(|b| near(b.0, LOW) && near(b.1, HIGH)),
        "{across:?}"
    );
    assert!(
        across
            .iter()
            .all(|b| b.0 >= LOW - 0.01 && b.1 <= HIGH + 0.01)
    );
    let past = bounds([1.5, 0.0]);
    assert!(
        past.iter().any(|b| near(b.0, 0.0) && near(b.1, HIGH)),
        "{past:?}"
    );
}

/// One vertex on two tiles reads each tile's height from its triangles on that tile. The surface
/// stays closed, baked at their mean and welded at runtime, but it takes a step the tiles disagree
/// on, as across a uv seam.
#[test]
fn heights_differing_across_a_tile_edge_warn() {
    let d = stepped("udim-mesh-seam");
    let stage = grid_stage("tex.<UDIM>.png", [0.5, 0.0], HEIGHT);
    let (_, warnings) = cook_with_warnings(&d, &stage, Subdivision::Hull);
    assert!(
        warnings.iter().any(|w| w.contains("tile edge")),
        "{warnings:?}"
    );
}

/// Baked vertices take the height of the tile under them, and the fallback past the tiles.
#[test]
fn baked_heights_follow_the_tile_under_each_vertex() {
    let d = stepped("udim-mesh-bake");
    let heights = |at: [f32; 2]| {
        let stage = grid_stage("tex.<UDIM>.png", at, HEIGHT);
        let a = cook_at(&d, &stage, baking()).unwrap();
        corners(&a, "u/mesh")
    };
    for (p, _, uv) in heights([0.5, 0.0]) {
        if (uv[0] - 1.0).abs() > 1e-3 {
            let want = if uv[0] < 1.0 { LOW } else { HIGH };
            assert!((p.z - want).abs() < 0.01, "{p} at {uv:?}");
        }
    }
    for (p, _, uv) in heights([1.5, 0.0]) {
        if (uv[0] - 2.0).abs() > 1e-3 {
            let want = if uv[0] < 2.0 { HIGH } else { 0.0 };
            assert!((p.z - want).abs() < 0.01, "{p} at {uv:?}");
        }
    }
}

/// A mesh past every tile reads the fallback, which decides its triangles as one texel would.
#[test]
fn a_mesh_past_a_tile_reads_what_the_set_holds_there() {
    let d = dir("udim-mesh-missing");
    std::fs::write(d.join("tex.1001.png"), rings()).unwrap();
    let alpha_tested = |fallback: f32, at: [f32; 2]| {
        let wired = r#"float inputs:opacity.connect = </M/T.outputs:r>
        float inputs:opacityThreshold = 0.5"#;
        let stage = grid_stage("tex.<UDIM>.png", at, wired).replace(
            r#"token inputs:sourceColorSpace = "raw""#,
            &format!(
                "token inputs:sourceColorSpace = \"raw\"\nfloat4 inputs:fallback = ({fallback}, 0, 0, 1)"
            ),
        );
        let a = cook_at(&d, &stage, UsdSettings::default()).unwrap();
        let info = MeshInfo::parse(&blob(&a, "u/mesh", 0)).unwrap();
        let mut tested = 0;
        for p in &info.pages {
            let mut page = vec![0; p.resident_len as usize];
            orr_pack::mesh::decode_page(&blob(&a, "u/mesh", p.blob as usize), &mut page).unwrap();
            let view = PageView::new(&page, &info.header).unwrap();
            for i in 0..view.cluster_count() {
                let c = view.cluster(i);
                tested += u32::from(c.tri_count - c.alpha_first);
            }
        }
        tested
    };
    assert!(alpha_tested(1.0, [0.0, 0.0]) > 0, "rings cross the cutoff");
    // Tile 1002 has no file: past it the whole mesh reads the fallback.
    assert_eq!(
        alpha_tested(1.0, [1.0, 0.0]),
        0,
        "a passing fallback is opaque"
    );
    assert_eq!(alpha_tested(0.0, [1.0, 0.0]), 512, "a failing one is not");
}

/// Inputs packed into one texture share it tile by tile when they read the same cells, and a set
/// beside an image, or a set of other cells, keeps a texture of its own.
#[test]
fn inputs_share_tiles_only_at_the_same_cells() {
    let d = dir("udim-pack");
    let tile = png(8, 8, |x, y| [(x * 32) as u8, (y * 32) as u8, 0, 255]);
    for name in ["tex.1001", "tex.1002", "other.1001", "plain"] {
        std::fs::write(d.join(format!("{name}.png")), &tile).unwrap();
    }
    let reader =
        |name: &str, file: &str| texture(name, RAW).replace("@tex.png@", &format!("@{file}@"));
    let stage = |rough: &str, metal: &str| {
        let prims = format!("{}\n{}\n{READER}", reader("T", rough), reader("U", metal));
        previewed(
            "float inputs:roughness.connect = </M/T.outputs:r>\n\
             float inputs:metallic.connect = </M/U.outputs:g>",
            &prims,
            ST,
        )
    };
    let cooked = |rough: &str, metal: &str| {
        let body = stage(rough, metal);
        let a = cook_at(&d, &body, UsdSettings::default()).unwrap();
        let m = material(&a);
        let same = m.textures[slot::ROUGHNESS].texture == m.textures[slot::METALNESS].texture;
        (a, m, same)
    };
    let (a, m, same) = cooked("tex.<UDIM>.png", "tex.<UDIM>.png");
    assert!(same, "one set of two channels");
    assert_eq!(tiles_of(&a, &m, slot::ROUGHNESS).len(), 2);
    assert_eq!(m.textures[slot::METALNESS].channels, 1);
    let (a, m, same) = cooked("tex.<UDIM>.png", "other.<UDIM>.png");
    assert!(!same, "other cells");
    assert_eq!(tiles_of(&a, &m, slot::METALNESS).len(), 1);
    let (.., same) = cooked("tex.<UDIM>.png", "plain.png");
    assert!(!same, "a set and an image");
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

/// Each corner of mesh entry `name` with its cluster's part.
fn part_corners(a: &Archive, name: &str) -> Vec<(u16, Vec3)> {
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
                    out.push((c.part, Vec3::from(view.position(&c, v))));
                }
            }
        }
    }
    out
}

/// Clip `clip` of skeleton `model`, and the skeleton's skinning matrices `t` seconds into it.
fn posed(a: &Archive, model: &str, clip: &str, t: f32) -> (Clip, Vec<Affine3>) {
    let sk = SkeletonInfo::parse(&blob(a, &format!("{model}/skeleton"), 0)).unwrap();
    let name = format!("{model}/clip/{clip}");
    let clip = Clip::parse(&blob(a, &name, 0), &blob(a, &name, 1)).unwrap();
    let mut pose = sk.rest.clone();
    clip.sample(t, &mut pose);
    let n = sk.joint_count();
    let (mut model, mut palette) = (vec![Affine3::IDENTITY; n], vec![Affine3::IDENTITY; n]);
    sk.model_pose(&pose, &mut model);
    sk.palette(&model, &mut palette);
    (clip, palette)
}

/// `xformOp.h`: time samples of an op interpolate as the op's own value, so a full turn between
/// two samples turns all the way, though the matrices at those samples are equal. What moves
/// leaves the stage's model for one of its own, and a prim sampled at one value holds still.
#[test]
fn a_time_sampled_xform_moves_the_meshes_under_it() {
    let a = cook(
        "xform-samples",
        &format!(
            r#"def Xform "Spin"
{{
    double xformOp:rotateY.timeSamples = {{ 0: 0, 10: 360 }}
    uniform token[] xformOpOrder = ["xformOp:rotateY"]
    def Mesh "M"
    {{
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        {TRIANGLE}
    }}
}}
def Xform "Held"
{{
    double3 xformOp:translate.timeSamples = {{ 0: (0, 5, 0), 10: (0, 5, 0) }}
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "M"
    {{
        {TRIANGLE}
    }}
}}"#
        ),
    );
    assert!(!has(&a, "u/skeleton") && !has(&a, "u/Held/skeleton"));
    let still = corners(&a, "u/mesh");
    assert!(still.len() == 3 && still.iter().all(|c| c.0.y >= 5.0));
    let sk = SkeletonInfo::parse(&blob(&a, "u/Spin/skeleton", 0)).unwrap();
    assert_eq!(sk.names, ["", "Spin"]);
    let moved = part_corners(&a, "u/Spin/mesh");
    assert!(
        moved.iter().all(|c| c.0 == 1),
        "the renderer poses no part 0"
    );
    let far = moved.iter().find(|c| c.1.x > 1.5).unwrap().1;
    let at = |t: f32| posed(&a, "u/Spin", "xform", t).1[1].transform_point3(far);
    let (clip, _) = posed(&a, "u/Spin", "xform", 0.0);
    assert!((clip.header.duration - 1.0).abs() < 1e-5);
    assert!(
        at(0.25).abs_diff_eq(Vec3::new(0.0, 0.0, -2.0), 1e-2),
        "{}",
        at(0.25)
    );
    assert!(
        at(0.5).abs_diff_eq(Vec3::new(-2.0, 0.0, 0.0), 1e-2),
        "{}",
        at(0.5)
    );
}

/// The stage holds one timeline, so a rig whose samples start late keeps its place on it, and
/// two rigs of one name part: a top-level prim's path is that name too.
#[test]
fn xform_clips_keep_time_and_rigs_of_one_name_part() {
    let a = cook(
        "xform-timeline",
        &format!(
            r#"def Xform "Group"
{{
    def Xform "Door"
    {{
        double3 xformOp:translate.timeSamples = {{ 0: (0, 0, 0), 10: (3, 0, 0) }}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "M"
        {{
            {TRIANGLE}
        }}
    }}
}}
def Xform "Door"
{{
    double3 xformOp:translate.timeSamples = {{ 5: (0, 0, 0), 10: (3, 0, 0) }}
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "M"
    {{
        {TRIANGLE}
    }}
}}"#
        ),
    );
    let (clip, half) = posed(&a, "u/node3", "xform", 0.5);
    assert!((clip.header.duration - 1.0).abs() < 1e-5);
    let rest = posed(&a, "u/node3", "xform", 0.0).1[1].translation;
    assert!(half[1].translation.abs_diff_eq(rest, 1e-4), "still at half");
    assert!(has(&a, "u/Door/skeleton"));
}

/// Every clip and animated page of a stage runs on its one timeline, from the first sample of
/// anything that moves: an animation that starts late holds its first pose until its first
/// sample, as USD holds a value, and a page's clock reads the same time as the clips.
#[test]
fn clips_and_pages_run_on_the_stage_timeline() {
    let anim = |name: &str, samples: &str| {
        format!(
            "def SkelAnimation \"{name}\" {{\n uniform token[] joints = [\"Root\"]\n \
             float3[] translations.timeSamples = {{ {samples} }}\n {ROTATIONS}\n {SCALES}\n}}"
        )
    };
    let late = anim("Late", "5: [(0, 0, 0)], 10: [(1, 0, 0)]");
    let early = anim("Early", "-10: [(0, 0, 0)], 0: [(2, 0, 0)]");
    let body = format!(
        "{}\n{}",
        skinned("", &format!("{late}\n{early}")),
        instancing(
            "int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0)], 10: [(4, 0, 0)] }"
        )
    );
    let a = cook("stage-timeline", &body);
    // Time codes -10 to 10 at 10 a second.
    let root_at = |clip: &str, t: f32| {
        let name = format!("u/Hero/clip/{clip}");
        let clip = Clip::parse(&blob(&a, &name, 0), &blob(&a, &name, 1)).unwrap();
        assert!((clip.header.duration - 2.0).abs() < 1e-5, "{name}");
        let mut pose = vec![Trs::IDENTITY; 2];
        clip.sample(t, &mut pose);
        pose[0].translation[0]
    };
    for (clip, t, want) in [
        ("Late", 0.0, 0.0),
        ("Late", 1.25, 0.0),
        ("Late", 1.75, 0.5),
        ("Late", 2.0, 1.0),
        ("Early", 0.5, 1.0),
        ("Early", 1.5, 2.0),
    ] {
        let x = root_at(clip, t);
        assert!((x - want).abs() < 1e-3, "{clip} at {t} s: {x}, not {want}");
    }
    assert_copies(&a, 0.0, &[[0.0, 0.0, 0.0]]);
    assert_copies(&a, 1.75, &[[3.0, 0.0, 0.0]]);
}

/// `schemas.dox` poses a skeleton in its prim's space, so a prim above it moves the skin: it
/// takes that prim as a joint, and every prim down to the skeleton.
#[test]
fn an_xform_above_a_skeleton_moves_its_skin() {
    let a = cook(
        "xform-above-skel",
        &format!(
            r#"def Xform "Mover"
{{
    double3 xformOp:translate.timeSamples = {{ 0: (0, 0, 0), 10: (3, 0, 0) }}
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def SkelRoot "Hero"
    {{
        {RIG}
        def Mesh "Body"
        {{
            {TRIANGLE}
            rel skel:skeleton = </Mover/Hero/Rig>
            {INFLUENCES}
        }}
    }}
}}"#
        ),
    );
    let sk = SkeletonInfo::parse(&blob(&a, "u/Hero/skeleton", 0)).unwrap();
    assert_eq!(sk.names, ["Mover", "Hero", "Rig", "Root", "Arm"]);
    assert_eq!(
        sk.parents[3], 2,
        "a root joint hangs on its skeleton's prim"
    );
    let joint = usize::from(joints_of(&a, "u/Hero/mesh")[0]);
    assert_eq!(joint, 4, "joint indices follow the prims above");
    let (_, start) = posed(&a, "u/Hero", "xform", 0.0);
    let (_, end) = posed(&a, "u/Hero", "xform", 1.0);
    for (p, ..) in corners(&a, "u/Hero/mesh") {
        let (from, to) = (
            start[joint].transform_point3(p),
            end[joint].transform_point3(p),
        );
        assert!(to.abs_diff_eq(from + Vec3::X * 3.0, 1e-4), "{from} {to}");
    }
}

/// A selection frames a model, and a prim under it that moves leaves it one model with a
/// skeleton: what holds still sits on the still joint, in part 0, which takes no pose.
#[test]
fn a_selected_model_keeps_what_moves_in_it() {
    let settings = UsdSettings {
        models: vec![NodeSelect {
            node: "Frame".into(),
        }],
        ..UsdSettings::default()
    };
    let d = dir("xform-selected");
    let body = format!(
        r#"def Xform "Frame"
{{
    double3 xformOp:translate = (0, 0, 9)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "Still"
    {{
        {TRIANGLE}
    }}
    def Xform "Door"
    {{
        double3 xformOp:translate.timeSamples = {{ 0: (2, 0, 0), 10: (4, 0, 0) }}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "M"
        {{
            {TRIANGLE}
        }}
    }}
}}"#
    );
    let a = cook_at(&d, &body, settings).unwrap();
    let sk = SkeletonInfo::parse(&blob(&a, "u/Frame/skeleton", 0)).unwrap();
    assert_eq!(sk.names, ["", "Door"]);
    let parts = part_corners(&a, "u/Frame/mesh");
    assert!(parts.iter().any(|c| c.0 == 0) && parts.iter().any(|c| c.0 == 1));
    assert!(
        parts.iter().all(|c| (c.0 == 1) == (c.1.x >= 2.0)),
        "{parts:?}"
    );
    let (_, end) = posed(&a, "u/Frame", "xform", 1.0);
    let door = parts.iter().find(|c| c.0 == 1).unwrap().1;
    assert!(
        end[1]
            .transform_point3(door)
            .abs_diff_eq(door + Vec3::X * 2.0, 1e-4)
    );
}

/// `bindTransforms` take any invertible transform, whatever its orientation. A bind thin in a
/// turned direction inverts in the cook's double precision, so a point off its plane comes back
/// at rest instead of dropping to the plane.
#[test]
fn a_thin_turned_bind_keeps_the_points_off_its_plane() {
    // 1e-6 along (1, -1, 0), turned on one side or on both, where it shears.
    let (c, thin) = (std::f64::consts::FRAC_1_SQRT_2, 1e-6);
    let (p, q) = ((1.0 + thin) / 2.0, (1.0 - thin) / 2.0);
    let one = format!(
        "(({c}, {c}, 0, 0), ({}, {}, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))",
        -c * thin,
        c * thin
    );
    let both = format!("(({p}, {q}, 0, 0), ({q}, {p}, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))");
    for (name, bind) in [("one", one), ("both", both)] {
        let a = cook(
            &format!("skel-thin-bind-{name}"),
            &format!(
                r#"def SkelRoot "Hero"
{{
    def Skeleton "Rig"
    {{
        uniform token[] joints = ["Root"]
        uniform matrix4d[] bindTransforms = [{bind}]
    }}
    def Mesh "Body"
    {{
        {TRIANGLE}
        rel skel:skeleton = </Hero/Rig>
        int[] primvars:skel:jointIndices = [0] (
            elementSize = 1
            interpolation = "constant"
        )
        float[] primvars:skel:jointWeights = [1] (
            elementSize = 1
            interpolation = "constant"
        )
    }}
}}"#
            ),
        );
        let sk = SkeletonInfo::parse(&blob(&a, "u/Hero/skeleton", 0)).unwrap();
        let (mut model, mut palette) = (vec![Affine3::IDENTITY; 1], vec![Affine3::IDENTITY; 1]);
        sk.model_pose(&sk.rest, &mut model);
        sk.palette(&model, &mut palette);
        let worst = corners(&a, "u/Hero/mesh")
            .iter()
            .map(|c| palette[0].transform_point3(c.0).distance(c.0))
            .fold(0.0, f32::max);
        assert!(
            worst < 0.2,
            "{name}: a point lands {worst} m from where it is bound"
        );
    }
}

/// `instancing.dox`: copies of a model "will typically have a different root transform, and
/// different set of joint transforms", sharing all the rest: the "bind state". A skinned prototype
/// is one model and skeleton however many copies place it, with its other meshes on a joint of
/// their own, and a scale that collapses an axis goes to a skeleton while the mesh stays bound.
#[test]
fn a_skinned_prototype_is_a_model_placed_per_copy() {
    let a = cook(
        "skel-prototype",
        &format!(
            r#"def PointInstancer "Crowd"
{{
    rel prototypes = [</Crowd/Hero>]
    int[] protoIndices = [0, 0, 0]
    point3f[] positions = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
    float3[] scales = [(1, 1, 1), (1, 1, 1), (1, 0, 1)]
    def SkelRoot "Hero"
    {{
        {RIG}
        def Mesh "Body"
        {{
            {TRIANGLE}
            rel skel:skeleton = </Crowd/Hero/Rig>
            {INFLUENCES}
        }}
        def Mesh "Prop"
        {{
            double3 xformOp:translate = (5, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            {TRIANGLE}
        }}
    }}
}}"#
        ),
    );
    let sk = SkeletonInfo::parse(&blob(&a, "u/Hero/skeleton", 0)).unwrap();
    assert_eq!(sk.names, ["Root", "Arm", ""]);
    assert_eq!(copies(&a, "u/Hero").len(), 2);
    let model = ModelInfo::parse(&blob(&a, "u/Hero", 0)).unwrap();
    assert_eq!(model.header.skeleton, AssetId::from_name("u/Hero/skeleton"));
    let (mut pose, mut palette) = (vec![Affine3::IDENTITY; 3], vec![Affine3::IDENTITY; 3]);
    sk.model_pose(&sk.rest, &mut pose);
    sk.palette(&pose, &mut palette);
    let mesh = corners(&a, "u/Hero/mesh");
    assert_eq!(mesh.len(), 6);
    let joints = joints_of(&a, "u/Hero/mesh");
    assert!(joints.contains(&1) && joints.contains(&2), "{joints:?}");
    let prop = mesh.iter().find(|c| c.0.x >= 5.0).expect("the prop").0;
    assert!(palette[2].transform_point3(prop).abs_diff_eq(prop, 1e-5));

    let flat = SkeletonInfo::parse(&blob(&a, "u/Hero/baked/0/skeleton", 0)).unwrap();
    assert_eq!(flat.root.matrix3.y_axis.y, 0.0);
    assert_eq!(copies(&a, "u/Hero/baked/0").len(), 1);
    let bound = corners(&a, "u/Hero/baked/0/mesh");
    assert!(
        bound.iter().any(|c| c.0.y > 0.5),
        "the mesh keeps its bind pose"
    );
}

/// `schemas.dox`: "Each skel:skeleton binding may be thought of as identifying a Skeleton
/// Instance, animated according to whichever inherited skel:animationSource is defined at the
/// location at which the skel:skeleton is bound", so one root holds as many as it has bindings,
/// of one skeleton or of several.
#[test]
fn each_skeleton_binding_under_a_root_is_a_model() {
    let binding = |prim: &str, skeleton: &str, anim: &str| {
        format!(
            r#"def Xform "{prim}" (
        prepend apiSchemas = ["SkelBindingAPI"]
    )
    {{
        rel skel:skeleton = </Crowd/{skeleton}>
        rel skel:animationSource = </Crowd/{anim}>
        def Mesh "M"
        {{
            {TRIANGLE}
            {INFLUENCES}
        }}
    }}"#
        )
    };
    let anim = |name: &str, x: f32| {
        format!(
            r#"def SkelAnimation "{name}"
    {{
        uniform token[] joints = ["Root"]
        float3[] translations.timeSamples = {{ 0: [(0, 0, 0)], 10: [({x}, 0, 0)] }}
        {ROTATIONS}
        half3[] scales = [(1, 1, 1)]
    }}"#
        )
    };
    let a = cook(
        "skel-bindings",
        &format!(
            r#"def SkelRoot "Crowd"
{{
    {RIG}
    {}
    {}
    {}
    {}
    {}
}}"#,
            RIG.replace("\"Rig\"", "\"Other\""),
            anim("Walk", 1.0),
            binding("One", "Rig", "Walk"),
            binding("Two", "Rig", "Walk"),
            binding("Three", "Other", "Walk"),
        ),
    );
    for model in ["One", "Two", "Three"] {
        let name = format!("u/Crowd/{model}");
        assert!(has(&a, &format!("{name}/skeleton")), "{model}");
        assert!(has(&a, &format!("{name}/clip/Walk")), "{model}");
        assert_eq!(corners(&a, &format!("{name}/mesh")).len(), 3, "{model}");
    }
    assert!(!has(&a, "u/Crowd/skeleton"));
}

/// `instancing.dox` asks one bind state of a prototype's copies however they are placed: with
/// every copy flat its mesh still binds its skeleton, and a clip under the root comes with no
/// `skel:animationSource`.
#[test]
fn a_prototype_whose_copies_all_flatten_stays_skinned() {
    let a = cook(
        "skel-prototype-flat",
        &format!(
            r#"def PointInstancer "Crowd"
{{
    rel prototypes = [</Crowd/Hero>]
    int[] protoIndices = [0]
    point3f[] positions = [(1, 0, 0)]
    float3[] scales = [(1, 0, 1)]
    def SkelRoot "Hero"
    {{
        {RIG}
        def SkelAnimation "Walk"
        {{
            uniform token[] joints = ["Root"]
            float3[] translations.timeSamples = {{ 0: [(0, 0, 0)], 10: [(1, 0, 0)] }}
            {ROTATIONS}
            half3[] scales = [(1, 1, 1)]
        }}
        def Mesh "Body"
        {{
            {TRIANGLE}
            rel skel:skeleton = </Crowd/Hero/Rig>
            {INFLUENCES}
        }}
    }}
}}"#
        ),
    );
    assert!(has(&a, "u/Hero/baked/0/clip/Walk"));
    let joints = joints_of(&a, "u/Hero/baked/0/mesh");
    assert!(joints.contains(&1), "{joints:?}");
}

/// Sampled xforms move a prim whichever skin stands beside it. Above a prototype's
/// skeleton the still joint undoes the root the prims take, and a mesh under a moving prim beside
/// the skin rides a rig of its own.
#[test]
fn what_moves_in_a_skinned_prototype_moves() {
    let a = cook(
        "skel-prototype-moving",
        &format!(
            r#"def PointInstancer "Crowd"
{{
    rel prototypes = [</Crowd/Hero>]
    int[] protoIndices = [0]
    point3f[] positions = [(1, 0, 0)]
    def SkelRoot "Hero"
    {{
        double3 xformOp:translate = (0, 3, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Xform "Lift"
        {{
            double3 xformOp:translate.timeSamples = {{ 0: (0, 1, 0), 24: (0, 2, 0) }}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            {RIG}
        }}
        def Mesh "Body"
        {{
            {TRIANGLE}
            rel skel:skeleton = </Crowd/Hero/Lift/Rig>
            {INFLUENCES}
        }}
        def Mesh "Prop"
        {{
            double3 xformOp:translate = (5, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            {TRIANGLE}
        }}
        def Xform "Door"
        {{
            double3 xformOp:translate.timeSamples = {{ 0: (0, 0, 0), 24: (1, 0, 0) }}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Mesh "Leaf"
            {{
                {TRIANGLE}
            }}
        }}
    }}
}}"#
        ),
    );
    let sk = SkeletonInfo::parse(&blob(&a, "u/Hero/skeleton", 0)).unwrap();
    let n = sk.joint_count();
    let (mut pose, mut palette) = (vec![Affine3::IDENTITY; n], vec![Affine3::IDENTITY; n]);
    sk.model_pose(&sk.rest, &mut pose);
    sk.palette(&pose, &mut palette);
    let prop = Vec3::new(5.0, 3.0, 0.0);
    let held = palette[n - 1].transform_point3(prop);
    assert!(held.abs_diff_eq(prop, 1e-5), "{held}");
    assert_eq!(corners(&a, "u/Hero/mesh").len(), 6);
    assert!(has(&a, "u/Hero/moved/skeleton"));
    assert!(has(&a, "u/Hero/moved/clip/xform"));
    assert_eq!(copies(&a, "u/Hero/moved").len(), 1);
}

/// Binding prims that share a name are still instances of their own, and a root drawn in place
/// that is also a prototype keeps both.
#[test]
fn binding_prims_of_one_name_and_a_drawn_prototype_each_keep_a_model() {
    let binding = |prim: &str| {
        format!(
            r#"def Xform "{prim}"
    {{
        def Xform "Geo" (
            prepend apiSchemas = ["SkelBindingAPI"]
        )
        {{
            rel skel:skeleton = </Hero/Rig>
            def Mesh "M"
            {{
                {TRIANGLE}
                {INFLUENCES}
            }}
        }}
    }}"#
        )
    };
    let a = cook(
        "skel-binding-names",
        &format!(
            r#"def SkelRoot "Hero"
{{
    {RIG}
    {}
    {}
}}
def PointInstancer "Crowd"
{{
    rel prototypes = [</Hero>]
    int[] protoIndices = [0]
    point3f[] positions = [(1, 0, 0)]
}}"#,
            binding("A"),
            binding("B"),
        ),
    );
    for model in ["u/Hero/A/Geo", "u/Hero/B/Geo", "u/Hero/prototype"] {
        assert!(has(&a, &format!("{model}/skeleton")), "{model}");
    }
    assert_eq!(copies(&a, "u/Hero/prototype").len(), 1);
}

/// `xformOp.h` takes any matrix for `xformOp:transform`, so a shear moves like the rest: a joint
/// holds the matrix at rest and at every key, and its stretch interpolates.
#[test]
fn a_shearing_xform_is_a_joint_local() {
    let a = cook(
        "xform-shear",
        &format!(
            r#"def Xform "Lean"
{{
    matrix4d xformOp:transform.timeSamples = {{
        0: ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        10: ((1, 0, 0, 0), (2, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    }}
    uniform token[] xformOpOrder = ["xformOp:transform"]
    def Mesh "M"
    {{
        {TRIANGLE}
    }}
}}"#
        ),
    );
    let (_, end) = posed(&a, "u", "xform", 1.0);
    let (_, middle) = posed(&a, "u", "xform", 0.5);
    let top = Vec3::Y;
    assert!(
        end[1]
            .transform_point3(top)
            .abs_diff_eq(Vec3::new(2.0, 1.0, 0.0), 1e-3)
    );
    assert!(
        middle[1]
            .transform_point3(top)
            .abs_diff_eq(Vec3::new(1.0, 1.0, 0.0), 1e-3)
    );
}

/// A clip's rate answers to a shear as to a turn: a lean that lands between two frames is an
/// error the rate doubles for, though no joint moves and no frame of a joint turns.
#[test]
fn a_sudden_shear_raises_the_clip_rate() {
    let a = cook(
        "xform-sudden-shear",
        &format!(
            r#"def Xform "Lean"
{{
    matrix4d xformOp:transform.timeSamples = {{
        0: ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        5: ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        5.2: ((1, 0, 0, 0), (2, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        10: ((1, 0, 0, 0), (2, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    }}
    uniform token[] xformOpOrder = ["xformOp:transform"]
    def Mesh "M"
    {{
        {TRIANGLE}
    }}
}}"#
        ),
    );
    let (clip, _) = posed(&a, "u", "xform", 0.0);
    let rate = clip.header.sample_rate;
    assert!(rate > 200.0, "{rate} frames a second");
}

/// `xformOp.h` takes any matrix, but a joint has no place for a projective column.
#[test]
fn a_projective_xform_is_no_joint_local() {
    let body = format!(
        r#"def Xform "Lean"
{{
    matrix4d xformOp:transform.timeSamples = {{
        0: ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        10: ((1, 0, 0, 0.5), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    }}
    uniform token[] xformOpOrder = ["xformOp:transform"]
    def Mesh "M"
    {{
        {TRIANGLE}
    }}
}}"#
    );
    let Err(e) = cook_at(&dir("xform-projective"), &body, UsdSettings::default()) else {
        panic!("a projective key cooked");
    };
    assert!(e.to_string().contains("affine"), "{e}");
}

/// `xformResolver.cpp` `GetPrimLocalToCommonSpace` takes "the first" of an instancer's instance
/// transforms: a skinned mesh in a prototype is posed in the space of its first copy, which stands
/// exactly at the skeleton's world pose. What stands above the prototype's root plays no part.
#[test]
fn a_skeleton_outside_a_prototype_is_seen_from_its_first_copy() {
    let a = cook(
        "skel-outside-prototype",
        &format!(
            r#"def SkelRoot "Root"
{{
    {RIG}
    def Xform "Group"
    {{
        double3 xformOp:translate = (0, 0, 7)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Xform "Proto"
        {{
            double3 xformOp:translate = (0, 9, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Mesh "Body"
            {{
                {TRIANGLE}
                rel skel:skeleton = </Root/Rig>
                {INFLUENCES}
            }}
        }}
    }}
    def PointInstancer "Crowd"
    {{
        rel prototypes = [</Root/Group/Proto>]
        int[] protoIndices = [0]
        point3f[] positions = [(1, 0, 0)]
    }}
}}"#
        ),
    );
    let sk = SkeletonInfo::parse(&blob(&a, "u/Proto/skeleton", 0)).unwrap();
    let (mut pose, mut palette) = (vec![Affine3::IDENTITY; 2], vec![Affine3::IDENTITY; 2]);
    sk.model_pose(&sk.rest, &mut pose);
    sk.palette(&pose, &mut palette);
    let joint = usize::from(joints_of(&a, "u/Proto/mesh")[0]);
    // The copy stands at x = 1, so the model holds the skeleton's pose 1 m the other way.
    let source = [Vec3::ZERO, Vec3::X, Vec3::Y].map(|p| p - Vec3::X);
    for (p, ..) in corners(&a, "u/Proto/mesh") {
        let got = palette[joint].transform_point3(p);
        let hit = source.iter().any(|w| got.abs_diff_eq(*w, 1e-4));
        assert!(hit, "{p} reaches {got}");
    }
    assert_eq!(copies(&a, "u/Proto").len(), 1);
}

/// Per skeleton of `a`, where the first corner of its mesh stands at the start and end of its
/// `xform` clip.
fn skinned_travel(a: &Archive) -> Vec<(Vec3, Vec3)> {
    let skeletons = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::Skeleton as u16);
    let models = skeletons.map(|e| a.name(e).trim_end_matches("/skeleton").to_string());
    models
        .map(|model| {
            let joint = usize::from(joints_of(a, &format!("{model}/mesh"))[0]);
            let (_, start) = posed(a, &model, "xform", 0.0);
            let (_, end) = posed(a, &model, "xform", 1.0);
            let bound = corners(a, &format!("{model}/mesh"))[0].0;
            (
                start[joint].transform_point3(bound),
                end[joint].transform_point3(bound),
            )
        })
        .collect()
}

/// A skeleton hung from a prototype's moving root stays in the prototype's model, which holds
/// nothing of what stands above the root.
#[test]
fn a_moving_prototype_root_hangs_its_skeleton_in_the_model() {
    let a = cook(
        "prototype-root-moves",
        &format!(
            r#"def Xform "Group"
{{
    double3 xformOp:translate = (0, 0, 7)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def PointInstancer "Crowd"
    {{
        rel prototypes = [</Group/Crowd/Hero>]
        int[] protoIndices = [0]
        point3f[] positions = [(0, 0, 0)]
        def SkelRoot "Hero"
        {{
            double3 xformOp:translate.timeSamples = {{ 0: (0, 0, 0), 10: (0, 3, 0) }}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            {RIG}
            def Mesh "Body"
            {{
                {TRIANGLE}
                rel skel:skeleton = </Group/Crowd/Hero/Rig>
                {INFLUENCES}
            }}
        }}
    }}
}}"#
        ),
    );
    let [(from, to)] = skinned_travel(&a)[..] else {
        panic!("one skeleton");
    };
    assert!(from.z.abs() < 1e-4, "{from}");
    assert!(to.abs_diff_eq(from + Vec3::Y * 3.0, 1e-4), "{from} {to}");
}

/// A prim that moves, with an instancer under it placing `body` at x = 1 and 5 from `prototype`.
fn moving_crowd(prototype: &str, body: &str) -> String {
    format!(
        r#"def Xform "Mover"
{{
    double3 xformOp:translate.timeSamples = {{ 0: (0, 0, 0), 10: (0, 3, 0) }}
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def PointInstancer "Crowd"
    {{
        rel prototypes = [<{prototype}>]
        int[] protoIndices = [0, 0]
        point3f[] positions = [(1, 0, 0), (5, 0, 0)]
        {body}
    }}
}}"#
    )
}

/// The skeleton and joint model `model`'s page hangs from, and where its records stand.
fn hung(a: &Archive, model: &str) -> (AssetId, usize, Vec<Vec3>) {
    let want = AssetId::from_name(model);
    let mut found = None;
    let mut at = Vec::new();
    for e in a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
    {
        let blobs = a.blobs(e);
        let (head, body) = (read(a, &blobs[0]), read(a, &blobs[1]));
        let info = InstancePageInfo::parse(&head).unwrap();
        if info.prototypes.iter().all(|p| p.model != want) {
            continue;
        }
        found = Some((info.header.frame_skeleton, info.header.frame_joint as usize));
        let (cell, records) = decode(&head, &body).unwrap();
        for (_, m) in records {
            at.push(m.translation + Vec3::from(cell.map(|c| c as f32)) * 256.0);
        }
    }
    let (skeleton, joint) = found.unwrap_or_else(|| panic!("no page of {model}"));
    at.sort_by(|a, b| a.x.total_cmp(&b.x));
    (skeleton, joint, at)
}

/// `pointInstancer.h`: instances are placed in the instancer's space, so whatever moves it moves
/// them. They stay copies of one model, on a page that hangs from the joint of the prim above: a
/// palette entry poses them all, and what moves inside the prototype stays the prototype's.
#[test]
fn an_instancer_under_a_moving_prim_hangs_from_its_joint() {
    let body = format!(
        r#"def Xform "Proto"
        {{
            def Mesh "Still"
            {{
                {TRIANGLE}
            }}
            def Xform "Arm"
            {{
                double3 xformOp:translate.timeSamples = {{ 0: (0, 0, 0), 10: (0, 0, 2) }}
                uniform token[] xformOpOrder = ["xformOp:translate"]
                def Mesh "M"
                {{
                    {TRIANGLE}
                }}
            }}
        }}"#
    );
    let stage = format!(
        r#"{}
def Mesh "Ground"
{{
    double3 xformOp:translate = (0, -9, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    {TRIANGLE}
}}"#,
        moving_crowd("/Mover/Crowd/Proto", &body)
    );
    let a = cook("instancer-moved", &stage);
    assert_eq!(
        corners(&a, "u/mesh").len(),
        3,
        "the ground stays where it is"
    );
    let sk = SkeletonInfo::parse(&blob(&a, "u/Mover/skeleton", 0)).unwrap();
    assert_eq!(sk.names, ["", "Mover"]);
    let (skeleton, joint, at) = hung(&a, "u/Proto");
    assert_eq!(skeleton, AssetId::from_name("u/Mover/skeleton"));
    assert_eq!(joint, 1);
    assert_eq!(at.len(), 2);
    let (_, end) = posed(&a, "u/Mover", "xform", 1.0);
    for (x, p) in [1.0, 5.0].into_iter().zip(at) {
        let got = end[joint].transform_point3(p);
        assert!(got.abs_diff_eq(Vec3::new(x, 3.0, 0.0), 1e-3), "{got}");
    }
    // One mesh for both copies, with its own rig.
    assert!(has(&a, "u/Proto/skeleton"));
    assert_eq!(part_corners(&a, "u/Proto/mesh").len(), 6);
}

/// `instancing.dox`: copies share the bind state and take "different set of joint transforms". A
/// moving instancer changes only where they stand: one skinned model and skeleton for all, on a
/// page hung from the prim above, with a palette per copy.
#[test]
fn skinned_copies_under_a_moving_prim_share_their_skeleton() {
    let body = format!(
        r#"def SkelRoot "Hero"
        {{
            {RIG}
            def Mesh "Body"
            {{
                {TRIANGLE}
                rel skel:skeleton = </Mover/Crowd/Hero/Rig>
                {INFLUENCES}
            }}
        }}"#
    );
    let a = cook(
        "instancer-moved-skin",
        &moving_crowd("/Mover/Crowd/Hero", &body),
    );
    let skeletons: Vec<&str> = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::Skeleton as u16)
        .map(|e| a.name(e))
        .collect();
    assert_eq!(skeletons.len(), 2, "{skeletons:?}");
    let hero = SkeletonInfo::parse(&blob(&a, "u/Hero/skeleton", 0)).unwrap();
    assert_eq!(hero.names, ["Root", "Arm"]);
    let (skeleton, joint, at) = hung(&a, "u/Hero");
    assert_eq!(skeleton, AssetId::from_name("u/skeleton"));
    assert_eq!(at.len(), 2);
    let (_, end) = posed(&a, "u", "xform", 1.0);
    let got = end[joint].transform_point3(at[0]);
    assert!(got.abs_diff_eq(at[0] + Vec3::Y * 3.0, 1e-3), "{got}");
}

/// A selection holds no stage, so what moves an instancer outside it has a rig of its own, as the
/// instancer's prototypes have a model whether selected or not.
#[test]
fn a_selection_leaves_a_moving_instancer_its_rig() {
    let body = format!(
        r#"def Xform "Proto"
        {{
            def Mesh "M"
            {{
                {TRIANGLE}
            }}
        }}"#
    );
    let stage = format!(
        r#"{}
def Mesh "Picked"
{{
    {TRIANGLE}
}}"#,
        moving_crowd("/Mover/Crowd/Proto", &body)
    );
    let settings = UsdSettings {
        models: vec![NodeSelect {
            node: "Picked".into(),
        }],
        ..UsdSettings::default()
    };
    let a = cook_at(&dir("instancer-selected"), &stage, settings).unwrap();
    let (skeleton, joint, _) = hung(&a, "u/Proto");
    assert_eq!(skeleton, AssetId::from_name("u/Mover/Crowd/skeleton"));
    assert_eq!(joint, 1);
}

/// `pointInstancer.h`: "the 64 bit integer ID of the particle". Ids a double cannot tell apart are
/// two instances, and a mask names one of them.
#[test]
fn ids_past_a_double_stay_apart() {
    for (name, mask) in [
        ("attribute", "int64[] invisibleIds = [9007199254740993]"),
        ("metadata", ""),
    ] {
        let stage = instancer(&format!(
            "int[] protoIndices = [0, 0]
    point3f[] positions = [(0, 0, 0), (1, 0, 0)]
    int64[] ids = [9007199254740992, 9007199254740993]
    {mask}"
        ));
        let stage = match name {
            "metadata" => stage.replacen(
                "def PointInstancer \"I\"",
                "def PointInstancer \"I\" (\n    inactiveIds = [ 9007199254740993 ]\n)",
                1,
            ),
            _ => stage,
        };
        let a = cook(&format!("ids-wide-{name}"), &stage);
        assert_eq!(copies(&a, "u/P").len(), 1, "{name}");
    }
}

/// `samplingUtils.cpp` `_GetAttrForTransforms`: on a sample, alignment is judged on the interval
/// ahead. Positions at 0, 5 and 20 with velocities at 0, 5 and 10 line up behind time 5 but not
/// ahead of it, so at 5 the velocities drop and orientations, with samples at 0 and 10, slerp.
#[test]
fn alignment_on_a_sample_looks_at_the_interval_ahead() {
    let at = UsdSettings {
        time: Some(5.0),
        ..UsdSettings::default()
    };
    let a = cook_at(
        &dir("align-ahead"),
        &instancer(
            "int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0)], 5: [(0, 0, 0)], 20: [(0, 0, 0)] }
    vector3f[] velocities.timeSamples = { 0: [(1, 0, 0)], 5: [(1, 0, 0)], 10: [(1, 0, 0)] }
    quath[] orientations.timeSamples = { 0: [(1, 0, 0, 0)], 10: [(0, 0, 0, 1)] }",
        ),
        at,
    )
    .unwrap();
    // Half of a half turn about z: x goes to y.
    let got = copies(&a, "u/P")[0].transform_vector3(Vec3::X);
    assert!(got.abs_diff_eq(Vec3::Y, 2e-2), "{got}");
}

/// `pointInstancer.cpp` warns and places nothing where an attribute's length is not the number of
/// instances, and `samplingUtils.cpp` drops a rate of the wrong length with a warning: neither
/// refuses the stage.
#[test]
fn lengths_that_disagree_place_nothing_and_warn() {
    let body = |attributes: &str| instancer(&format!("int[] protoIndices = [0, 0]\n{attributes}"));
    for (name, attributes) in [
        ("positions", "point3f[] positions = [(0, 0, 0)]"),
        (
            "orientations",
            "point3f[] positions = [(0, 0, 0), (1, 0, 0)]
    quath[] orientations = [(1, 0, 0, 0)]",
        ),
        (
            "scales",
            "point3f[] positions = [(0, 0, 0), (1, 0, 0)]
    float3[] scales = [(1, 1, 1)]",
        ),
        (
            "ids",
            "point3f[] positions = [(0, 0, 0), (1, 0, 0)]
    int64[] ids = [7]",
        ),
    ] {
        let d = dir(&format!("length-{name}"));
        let stage = format!(
            "{}\ndef Mesh \"Ground\"\n{{\n    {TRIANGLE}\n}}",
            body(attributes)
        );
        let (a, warnings) = cook_with_warnings(&d, &stage, Subdivision::Hull);
        assert!(copies(&a, "u/P").is_empty(), "{name}");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains(name) && w.contains("1 for 2")),
            "{name}: {warnings:?}"
        );
    }

    // A rate of the wrong length goes, and what it would move stays.
    let d = dir("length-velocities");
    let (a, warnings) = cook_with_warnings(
        &d,
        &body(
            "point3f[] positions.timeSamples = { 0: [(0, 0, 0), (1, 0, 0)], 10: [(0, 0, 0), (1, 0, 0)] }
    vector3f[] velocities.timeSamples = { 0: [(1, 0, 0)], 10: [(1, 0, 0)] }",
        ),
        Subdivision::Hull,
    );
    assert_eq!(copies(&a, "u/P").len(), 2);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("velocities") && w.contains("dropped")),
        "{warnings:?}"
    );
}

/// `overview.dox`: without velocities, "if the timeSamples bracketing a requested sample have the
/// same number of elements, apply linear interpolation between the two samples; otherwise, use the
/// value of the sample with the lower/earlier ordinate".
#[test]
fn samples_of_different_lengths_hold_the_lower() {
    let at = UsdSettings {
        time: Some(5.0),
        ..UsdSettings::default()
    };
    let a = cook_at(
        &dir("length-held"),
        &instancer(
            "int[] protoIndices.timeSamples = { 0: [0], 10: [0, 0] }
    point3f[] positions.timeSamples = { 0: [(1, 0, 0)], 10: [(3, 0, 0), (9, 0, 0)] }",
        ),
        at,
    )
    .unwrap();
    let placed = transforms_at(&a, "u/P", 0.5);
    assert_eq!(placed.len(), 1);
    let x = placed[0].transform_point3(Vec3::ZERO).x;
    assert!((x - 1.0).abs() < 1e-3, "{x}");
}

fn instancing(attributes: &str) -> String {
    format!(
        r#"def PointInstancer "Crowd"
{{
    rel prototypes = [</Crowd/Proto>, </Crowd/Other>]
    {attributes}
    def Xform "Proto"
    {{
        def Mesh "M"
        {{
            {TRIANGLE}
        }}
    }}
    def Xform "Other"
    {{
        def Mesh "M"
        {{
            {TRIANGLE}
        }}
    }}
}}"#
    )
}

/// The cells of the animated pages of `model` and how they place its copies `t` seconds in.
fn placed_at(a: &Archive, model: &str, t: f32) -> Vec<([i32; 3], Affine3)> {
    let want = AssetId::from_name(model);
    let mut out = Vec::new();
    for e in a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
    {
        let blobs = a.blobs(e);
        let head = read(a, &blobs[0]);
        let info = InstancePageInfo::parse(&head).unwrap();
        if info.prototypes.iter().all(|p| p.model != want) {
            continue;
        }
        assert!(!info.motion_times.is_empty(), "{model} holds still");
        let h = info.header;
        let windows = (0..h.motion_windows()).map(|w| &blobs[h.motion_blob(w) as usize]);
        let rows: Vec<_> = windows.flat_map(|b| motion_rows(&read(a, b))).collect();
        let (k, dt) = motion_interval(&info.motion_times, t);
        let n = usize::from(h.record_count);
        for row in &rows[k * n..(k + 1) * n] {
            out.extend(motion_transform(&h, row, dt).map(|m| (h.cell, m)));
        }
    }
    out
}

/// How the animated pages of `model` place its copies `t` seconds in, in scene space, sorted.
fn transforms_at(a: &Archive, model: &str, t: f32) -> Vec<Mat4> {
    let in_scene = |(cell, mut m): ([i32; 3], Affine3)| {
        m.translation += Vec3::from(cell.map(|c| c as f32)) * 256.0;
        Mat4::from(m)
    };
    let mut out: Vec<Mat4> = placed_at(a, model, t).into_iter().map(in_scene).collect();
    let at = |m: &Mat4| m.w_axis.to_array();
    out.sort_by(|a, b| at(a).partial_cmp(&at(b)).unwrap());
    out
}

/// The lowest corner of each copy of the triangle in `model`, `t` seconds in.
fn copies_at(a: &Archive, model: &str, t: f32) -> Vec<Vec3> {
    let lowest = |m: &Mat4| {
        [Vec3::ZERO, Vec3::X, Vec3::Y]
            .map(|c| m.transform_point3(c))
            .into_iter()
            .reduce(Vec3::min)
            .unwrap()
    };
    let mut out: Vec<Vec3> = transforms_at(a, model, t).iter().map(lowest).collect();
    out.sort_by(|a, b| a.to_array().partial_cmp(&b.to_array()).unwrap());
    out
}

#[track_caller]
fn assert_copies(a: &Archive, t: f32, want: &[[f32; 3]]) {
    let got = copies_at(a, "u/Proto", t);
    assert_eq!(got.len(), want.len(), "at {t} s: {got:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(g.abs_diff_eq(Vec3::from(*w), 2e-3), "at {t} s: {got:?}");
    }
}

/// `overview.dox`: without velocities, "if the timeSamples bracketing a requested sample have the
/// same number of elements, apply linear interpolation". Copies move on a page of one model.
#[test]
fn sampled_positions_orientations_and_scales_move_their_copies() {
    let a = cook(
        "instances-sampled",
        &instancing(
            r#"int[] protoIndices = [0, 0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0), (4, 0, 0)], 10: [(0, 2, 0), (4, 0, 2)] }
    quath[] orientations.timeSamples = { 0: [(1, 0, 0, 0), (1, 0, 0, 0)], 10: [(0.70710678, 0, 0, 0.70710678), (1, 0, 0, 0)] }
    float3[] scales.timeSamples = { 0: [(1, 1, 1), (1, 1, 1)], 10: [(1, 1, 1), (2, 2, 2)] }"#,
        ),
    );
    assert_copies(&a, 0.0, &[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]);
    let turned = -std::f32::consts::FRAC_1_SQRT_2;
    assert_copies(&a, 0.5, &[[turned, 1.0, 0.0], [4.0, 0.0, 1.0]]);
    let end = transforms_at(&a, "u/Proto", 1.0);
    let scaled = end[1].determinant();
    assert!((scaled - 8.0).abs() < 1e-3, "{scaled}");
    assert!(!has(&a, "u/skeleton"));
}

/// Far from the instancer's origin, what a copy moves by stays as fine as near it: an f32 of the
/// position there steps by 0.25.
#[test]
fn moving_copies_keep_their_precision_far_from_the_instancer() {
    let a = cook(
        "instances-far",
        &instancing(
            r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(3000000, 0, 0)], 10: [(3000000.5, 0, 0)] }"#,
        ),
    );
    let (cell, m) = placed_at(&a, "u/Proto", 0.25)[0];
    let x = f64::from(cell[0]) * 256.0 + f64::from(m.translation.x);
    assert!((x - 3000000.125).abs() < 1e-3, "{x}");
}

/// `pointInstancer.h` leaves scales free, and a scale of 0 flattens a copy or takes it away. An
/// instance needs a transform that inverts, so a flat one stays where it can, at no visible size.
#[test]
fn scales_through_zero_flatten_and_hide_their_copies() {
    let a = cook(
        "instances-flat",
        &instancing(
            r#"int[] protoIndices = [0, 0]
    point3f[] positions = [(0, 0, 0), (5, 0, 0)]
    float3[] scales.timeSamples = { 0: [(1, 1, 1), (1, 1, 1)], 10: [(1, 0, 1), (0, 0, 0)] }"#,
        ),
    );
    assert_eq!(transforms_at(&a, "u/Proto", 0.5).len(), 2);
    let at = transforms_at(&a, "u/Proto", 1.0);
    assert_eq!(at.len(), 1);
    assert!(at[0].determinant() != 0.0);
    assert!(at[0].inverse().is_finite());
}

/// A sample that holds the same value throughout moves nothing, so the copies stay as they were.
#[test]
fn samples_that_hold_still_stay_on_pages() {
    let a = cook(
        "instances-held",
        &instancing(
            r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(1, 0, 0)], 10: [(1, 0, 0)] }"#,
        ),
    );
    assert_eq!(origins(&a, "u/Proto"), [[1000, 0, 0]]);
    let page = a
        .entries()
        .iter()
        .find(|e| e.class == Class::InstancePage as u16);
    let head = read(&a, &a.blobs(page.unwrap())[0]);
    assert!(
        InstancePageInfo::parse(&head)
            .unwrap()
            .motion_times
            .is_empty()
    );
}

/// `overview.dox`: with velocities whose samples line up with those of positions, "use the lower
/// velocities timeSample and the lower points timeSample", and `pointInstancer.h`: "we will not
/// attempt to interpolate samples of positions". Their sampling apart, "fall back to standard
/// position computation logic".
#[test]
fn velocities_carry_positions_from_the_lower_sample() {
    let carried = cook(
        "instances-velocities",
        &instancing(
            r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0)], 10: [(9, 0, 0)] }
    vector3f[] velocities.timeSamples = { 0: [(1, 0, 0)], 10: [(1, 0, 0)] }"#,
        ),
    );
    assert_copies(&carried, 0.5, &[[0.5, 0.0, 0.0]]);
    assert_copies(&carried, 1.0, &[[9.0, 0.0, 0.0]]);
    let apart = cook(
        "instances-velocities-apart",
        &instancing(
            r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0)], 10: [(9, 0, 0)] }
    vector3f[] velocities.timeSamples = { 0: [(1, 0, 0)], 5: [(1, 0, 0)] }"#,
        ),
    );
    assert_copies(&apart, 0.5, &[[4.5, 0.0, 0.0]]);
    let spun = cook(
        "instances-spin",
        &instancing(
            r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0)], 10: [(0, 0, 0)] }
    quath[] orientations.timeSamples = { 0: [(1, 0, 0, 0)], 10: [(1, 0, 0, 0)] }
    vector3f[] angularVelocities.timeSamples = { 0: [(0, 0, 270)], 10: [(0, 0, 270)] }"#,
        ),
    );
    let x = transforms_at(&spun, "u/Proto", 0.5)[0].transform_vector3(Vec3::X);
    let want = Vec3::new(135f32.to_radians().cos(), 135f32.to_radians().sin(), 0.0);
    assert!(x.abs_diff_eq(want, 1e-2), "{x}");
}

/// A sample of something else splits a rate's interval, and the quadratic and the spin go on
/// through it. `pointInstancer.h`: with a rate, values come from the lower sample, so scales step
/// though they would interpolate alone, and so do orientations whose own rate was dropped.
#[test]
fn a_rate_runs_through_samples_and_holds_the_rest() {
    let a = cook(
        "instances-rekeyed",
        &instancing(
            r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0)], 10: [(9, 0, 0)] }
    vector3f[] velocities.timeSamples = { 0: [(1, 0, 0)], 10: [(0, 0, 0)] }
    vector3f[] accelerations.timeSamples = { 0: [(2, 0, 0)], 10: [(0, 0, 0)] }
    quath[] orientations.timeSamples = { 0: [(1, 0, 0, 0)], 10: [(0, 0, 0, 1)] }
    vector3f[] angularVelocities.timeSamples = { 0: [(0, 0, 90)], 5: [(0, 0, 90)] }
    float3[] scales.timeSamples = { 0: [(1, 1, 1)], 4: [(2, 2, 2)], 10: [(3, 3, 3)] }"#,
        ),
    );
    let at = |t: f32| transforms_at(&a, "u/Proto", t)[0];
    for t in [0.2f32, 0.39, 0.4, 0.75] {
        let x = at(t).transform_point3(Vec3::ZERO).x;
        let want = t + t * t;
        assert!((x - want).abs() < 1e-4, "at {t} s: {x}, not {want}");
    }
    assert!(
        at(0.39)
            .transform_vector3(Vec3::X)
            .abs_diff_eq(Vec3::X, 1e-5)
    );
    assert!(
        at(0.4)
            .transform_vector3(Vec3::X)
            .abs_diff_eq(Vec3::X * 2.0, 1e-5)
    );
    assert!(
        at(0.75)
            .transform_vector3(Vec3::X)
            .abs_diff_eq(Vec3::X * 2.0, 1e-5)
    );
}

/// `pointInstancer.h`: positions before and after their samples come from the first and last, and
/// a rate there "extrapolates". Without one they stand still at the end.
#[test]
fn past_their_samples_rates_extrapolate_and_values_clamp() {
    let a = cook(
        "instances-open-ends",
        &instancing(
            r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 10: [(0, 0, 0)] }
    vector3f[] velocities.timeSamples = { 10: [(1, 0, 0)] }"#,
        ),
    );
    assert_copies(&a, -2.0, &[[-2.0, 0.0, 0.0]]);
    assert_copies(&a, 3.0, &[[3.0, 0.0, 0.0]]);
    let b = cook(
        "instances-clamped",
        &instancing(
            r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 10: [(0, 0, 0)], 20: [(4, 0, 0)] }"#,
        ),
    );
    assert_copies(&b, -2.0, &[[0.0, 0.0, 0.0]]);
    assert_copies(&b, 0.5, &[[2.0, 0.0, 0.0]]);
    assert_copies(&b, 7.0, &[[4.0, 0.0, 0.0]]);
}

/// `pointInstancer.h`: `ids` give "the 64 bit integer ID of the particle at each index, at each
/// point in time", and an index "will be taken over by a new particle": it is the index that
/// interpolates. `overview.dox`: samples of different lengths hold the lower one.
#[test]
fn ids_follow_instances_that_are_born_and_die() {
    let a = cook(
        "instances-ids",
        &instancing(
            r#"int[] protoIndices.timeSamples = { 0: [0, 0], 10: [0], 20: [0, 0] }
    int64[] ids.timeSamples = { 0: [1, 2], 10: [2], 20: [2, 3] }
    point3f[] positions.timeSamples = { 0: [(0, 0, 0), (5, 0, 0)], 10: [(5, 2, 0)], 20: [(5, 4, 0), (9, 0, 0)] }"#,
        ),
    );
    assert_copies(&a, 0.0, &[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
    assert_copies(&a, 0.95, &[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
    assert_copies(&a, 1.0, &[[5.0, 2.0, 0.0]]);
    assert_copies(&a, 1.95, &[[5.0, 2.0, 0.0]]);
    assert_copies(&a, 2.0, &[[5.0, 4.0, 0.0], [9.0, 0.0, 0.0]]);
}

/// `pointInstancer.h`: "The second masking feature encodes a list of IDs in a time-varying
/// Int64Array-valued UsdAttribute", so an instance hides by the sample as well. `protoIndices`
/// steps too, and a sample that does not match in length holds no instances.
#[test]
fn what_steps_changes_a_copy_at_its_sample() {
    let a = cook(
        "instances-invisible",
        &instancing(
            r#"int[] protoIndices = [0, 0]
    point3f[] positions = [(0, 0, 0), (5, 0, 0)]
    int64[] invisibleIds.timeSamples = { 0: [], 10: [1] }"#,
        ),
    );
    assert_copies(&a, 0.0, &[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
    assert_copies(&a, 1.0, &[[0.0, 0.0, 0.0]]);
    let b = cook(
        "instances-prototypes",
        &instancing(
            r#"int[] protoIndices.timeSamples = { 0: [0, 0], 10: [1, 0], 20: [1, 0, 0] }
    point3f[] positions.timeSamples = { 0: [(0, 0, 0), (5, 0, 0)], 10: [(0, 2, 0), (5, 2, 0)], 20: [(0, 2, 0), (5, 2, 0)], 30: [(0, 2, 0), (5, 2, 0), (7, 0, 0)] }"#,
        ),
    );
    assert_copies(&b, 0.5, &[[0.0, 1.0, 0.0], [5.0, 1.0, 0.0]]);
    assert_copies(&b, 1.0, &[[5.0, 2.0, 0.0]]);
    assert_eq!(copies_at(&b, "u/Other", 0.5), []);
    assert_eq!(copies_at(&b, "u/Other", 1.5), [Vec3::new(0.0, 2.0, 0.0)]);
    assert_copies(&b, 2.5, &[]);
    assert_eq!(copies_at(&b, "u/Other", 2.5), []);
    assert_copies(&b, 3.0, &[[5.0, 2.0, 0.0], [7.0, 0.0, 0.0]]);
}

/// `pointInstancer.h`: "instances are placed in the instancer's space": a prim moving above and
/// the instances' own samples compose, the one as the page's joint and the other as its rows. An
/// instancer in a prototype of moving instances stands in the copies the time read found.
#[test]
fn moving_instances_hang_from_what_moves_them() {
    let body = format!(
        r#"point3f[] positions.timeSamples = {{ 0: [(1, 0, 0), (5, 0, 0)], 10: [(1, 2, 0), (5, 2, 0)] }}
        def Xform "Proto"
        {{
            def Mesh "M"
            {{
                {TRIANGLE}
            }}
        }}"#
    );
    let stage = moving_crowd("/Mover/Crowd/Proto", &body)
        .replace("point3f[] positions = [(1, 0, 0), (5, 0, 0)]\n", "\n");
    let a = cook("instances-moving-hung", &stage);
    let (skeleton, joint, _) = hung(&a, "u/Proto");
    assert_eq!(skeleton, AssetId::from_name("u/skeleton"));
    assert_eq!(joint, 1);
    // The rows stand in the joint's bind space: the joint adds what the prim above moved.
    let (_, end) = posed(&a, "u", "xform", 1.0);
    let at = transforms_at(&a, "u/Proto", 1.0);
    let got = end[joint].transform_point3(at[0].transform_point3(Vec3::ZERO));
    assert!(got.abs_diff_eq(Vec3::new(1.0, 5.0, 0.0), 1e-3), "{got}");
}

/// 20 000 instances are that many records of one shared model, however a page splits them, and
/// each record keeps its own rows.
#[test]
fn many_moving_copies_stay_records_of_one_model() {
    let n = 20_000;
    let spot = |i: usize, y: usize| format!("({}, {y}, {})", i % 200 * 3, i / 200 * 3);
    let list = |y: usize| (0..n).map(|i| spot(i, y)).collect::<Vec<_>>().join(", ");
    let a = cook(
        "instances-many",
        &instancing(&format!(
            r#"int[] protoIndices = [{}]
    point3f[] positions.timeSamples = {{ 0: [{}], 10: [{}] }}"#,
            vec!["0"; n].join(", "),
            list(0),
            list(4),
        )),
    );
    assert!(!has(&a, "u/skeleton"));
    let models = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::Model as u16);
    assert_eq!(models.count(), 2, "one per prototype");
    let at = transforms_at(&a, "u/Proto", 0.5);
    assert_eq!(at.len(), n);
    let mut got: Vec<[i32; 3]> = at
        .iter()
        .map(|m| m.w_axis.truncate().to_array().map(|x| x.round() as i32))
        .collect();
    let mut want: Vec<[i32; 3]> = (0..n as i32)
        .map(|i| [i % 200 * 3, 2, i / 200 * 3])
        .collect();
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want);
}

/// `pointInstancer.cpp`, `_GetPrototypePathsForInstanceTransforms`: an index outside the prototypes
/// fails that call alone, at that time. USD defines no non-finite value, and a copy can't hold
/// one, so it is that instance alone that is skipped. Neither stops the cook.
#[test]
fn a_bad_sample_loses_its_instances_there_and_nothing_else() {
    let d = dir("instances-bad-samples");
    let (moving, warnings) = cook_with_warnings(
        &d,
        &instancing(
            r#"int[] protoIndices.timeSamples = { 0: [0, 0], 10: [0, 5], 20: [0, 0] }
    point3f[] positions = [(0, 0, 0), (5, 0, 0)]"#,
        ),
        Subdivision::Hull,
    );
    assert_copies(&moving, 0.5, &[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
    assert_copies(&moving, 1.5, &[]);
    assert_copies(&moving, 2.5, &[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
    let told = warnings.iter().filter(|w| w.contains("protoIndex 5"));
    assert_eq!(told.count(), 1, "{warnings:?}");
    let (still, warnings) = cook_with_warnings(
        &d,
        &instancing(
            r#"int[] protoIndices = [0, 5]
    point3f[] positions = [(0, 0, 0), (5, 0, 0)]"#,
        ),
        Subdivision::Hull,
    );
    assert_eq!(origins(&still, "u/Proto"), Vec::<[i32; 3]>::new());
    assert!(warnings.iter().any(|w| w.contains("protoIndex 5")));
    let (spent, warnings) = cook_with_warnings(
        &d,
        &instancing(
            r#"int[] protoIndices = [0, 0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0), (5, 0, 0)], 10: [(0, 2, 0), (inf, 0, 0)], 20: [(0, 2, 0), (5, 0, 0)] }"#,
        ),
        Subdivision::Hull,
    );
    assert_copies(&spent, 0.5, &[[0.0, 1.0, 0.0]]);
    assert_copies(&spent, 2.0, &[[0.0, 2.0, 0.0], [5.0, 0.0, 0.0]]);
    let told = warnings
        .iter()
        .filter(|w| w.contains("instance 1 is not finite"));
    assert_eq!(told.count(), 1, "{warnings:?}");
    let (held, warnings) = cook_with_warnings(
        &d,
        &instancing(
            r#"int[] protoIndices = [0, 0]
    point3f[] positions = [(1, 0, 0), (inf, 0, 0)]"#,
        ),
        Subdivision::Hull,
    );
    assert_eq!(origins(&held, "u/Proto"), [[1000, 0, 0]]);
    assert!(warnings.iter().any(|w| w.contains("not finite")));
}

/// As `a_baked_height_holds_under_a_placement_scale`, for copies that move: a scale that stands
/// still between samples cooks into a mesh and the rows move it rigidly, so the height keeps its
/// world length. A scale that grows has no mesh to go into, and says so.
#[test]
fn a_baked_height_holds_under_a_moving_copy() {
    let d = dir("displaced-moving-copies");
    let surface = previewed("float inputs:displacement = 0.25", "", "");
    let material = &surface[surface.find("def Material").unwrap()..];
    let stage = |attributes: &str| {
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
            rel material:binding = </M>
        }}
    }}
}}

{material}"#
        )
    };
    let height = |a: &Archive, name: &str| {
        corners(a, name)
            .iter()
            .map(|c| c.0.z)
            .fold(f32::MIN, f32::max)
    };
    let a = cook_at(
        &d,
        &stage(
            r#"int[] protoIndices = [0, 0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0), (5, 0, 0)], 10: [(0, 2, 0), (5, 2, 0)] }
    quath[] orientations.timeSamples = { 0: [(1, 0, 0, 0), (1, 0, 0, 0)], 10: [(0.70710678, 0, 0, 0.70710678), (1, 0, 0, 0)] }
    float3[] scales = [(1, 1, 1), (2, 2, 2)]"#,
        ),
        baking(),
    )
    .unwrap();
    assert!((height(&a, "u/P/mesh") - 0.25).abs() < 1e-3);
    assert!((height(&a, "u/P/baked/0/mesh") - 0.25).abs() < 1e-3);
    let plain = transforms_at(&a, "u/P", 0.5);
    let scaled = transforms_at(&a, "u/P/baked/0", 0.5);
    assert_eq!((plain.len(), scaled.len()), (1, 1));
    let at = scaled[0].transform_point3(Vec3::ZERO);
    assert!(at.abs_diff_eq(Vec3::new(5.0, 1.0, 0.0), 1e-3), "{at}");
    let det = scaled[0].determinant();
    assert!((det - 1.0).abs() < 1e-3, "the rows stretch by {det}");
    // The stretch is in the mesh: a corner stands twice as far out.
    let far = corners(&a, "u/P/baked/0/mesh")
        .iter()
        .map(|c| c.0.x)
        .fold(f32::MIN, f32::max);
    assert!((far - 2.0).abs() < 1e-3, "{far}");

    // A scale that steps at a sample stands in one mesh, then another.
    let stepped = cook_at(
        &d,
        &stage(
            r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(0, 0, 0)], 10: [(0, 2, 0)], 20: [(0, 4, 0)] }
    vector3f[] velocities.timeSamples = { 0: [(0, 2, 0)], 10: [(0, 2, 0)], 20: [(0, 2, 0)] }
    float3[] scales.timeSamples = { 0: [(2, 2, 2)], 10: [(3, 3, 3)] }"#,
        ),
        baking(),
    )
    .unwrap();
    assert_eq!(transforms_at(&stepped, "u/P/baked/0", 0.5).len(), 1);
    assert_eq!(transforms_at(&stepped, "u/P/baked/1", 0.5).len(), 0);
    assert_eq!(transforms_at(&stepped, "u/P/baked/0", 1.5).len(), 0);
    assert_eq!(transforms_at(&stepped, "u/P/baked/1", 1.5).len(), 1);
    assert!((height(&stepped, "u/P/baked/1/mesh") - 0.25).abs() < 1e-3);

    // A stretch above the instancer doesn't commute with a turn, so what holds it holds the turn
    // too, and the rows only carry the copy.
    let instancer = stage(
        r#"int[] protoIndices = [0]
    point3f[] positions.timeSamples = { 0: [(1, 0, 0)], 10: [(1, 2, 0)] }"#,
    )
    .replace("</I/Protos/P>", "</W/I/Protos/P>");
    let (instancer, material) = instancer.split_at(instancer.find("def Material").unwrap());
    let under = format!(
        r#"def Xform "W"
{{
    double3 xformOp:scale = (2, 1, 1)
    uniform token[] xformOpOrder = ["xformOp:scale"]
    {instancer}
}}

{material}"#
    );
    let stretched = cook_at(&d, &under, baking()).unwrap();
    assert!((height(&stretched, "u/P/baked/0/mesh") - 0.25).abs() < 1e-3);
    let at = transforms_at(&stretched, "u/P/baked/0", 0.5);
    let origin = at[0].transform_point3(Vec3::ZERO);
    assert!(
        origin.abs_diff_eq(Vec3::new(2.0, 1.0, 0.0), 1e-3),
        "{origin}"
    );
    assert!((at[0].determinant() - 1.0).abs() < 1e-3);

    let path = d.join("stage.usda");
    let grows = stage(
        r#"int[] protoIndices = [0]
    point3f[] positions = [(0, 0, 0)]
    float3[] scales.timeSamples = { 0: [(1, 1, 1)], 10: [(2, 2, 2)] }"#,
    );
    std::fs::write(&path, format!("{HEAD}\n{grows}")).unwrap();
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "u".into(),
                source: Source::Usd {
                    path,
                    settings: baking(),
                },
            }],
        )],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
    let warnings: Vec<&String> = cooked
        .report
        .assets
        .iter()
        .flat_map(|a| &a.warnings)
        .collect();
    let told = |w: &&String| w.contains("/I") && w.contains("stretches");
    assert!(warnings.iter().any(told), "{warnings:?}");
    let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
    assert_eq!(transforms_at(&a, "u/P", 0.5).len(), 1);
    assert!(!has(&a, "u/P/baked/0/mesh"));
}
