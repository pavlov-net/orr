//! USD behavior clause by clause. Each row cites the text it holds the importer to:
//! `xformOp.h`, `xformable.h`, `mesh.h`, `primvar.h`, `pointBased.h`, `pointInstancer.h` (PI),
//! `overview.dox` in usdGeom, `bindingAPI.h` and `schemas.dox` in usdSkel, and
//! `spec_usdpreviewsurface.rst` (PS), all as OpenUSD's `release` branch has them.

use std::path::PathBuf;

use orr_cook::{ArchiveSpec, AssetSpec, Manifest, Source, UsdSettings};
use orr_math::Vec3;
use orr_pack::mesh::{MeshInfo, PageView};
use orr_pack::{Archive, AssetId, Blob, Target};

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
    let path = d.join("stage.usda");
    std::fs::write(&path, format!("{HEAD}\n{body}")).unwrap();
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
