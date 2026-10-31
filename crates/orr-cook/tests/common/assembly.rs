//! A GLB whose mesh repeats one rigid piece: rotated copies of a bumpy, partial disc of 240
//! triangles in material 0, and with `sheet` a grid of 128 triangles in material 1.

use std::path::Path;

use glam::{EulerRot, Quat, Vec3};

pub(crate) const UNIT_TRIANGLES: u32 = 240;

/// Writes the GLB and returns each copy's area-weighted centroid.
pub(crate) fn write(path: &Path, copies: usize, sheet: bool) -> Vec<Vec3> {
    let n = 120;
    let mut disc = vec![([0.0, 0.0, 0.02], [0.5, 0.5])];
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let r = 0.2 + 0.03 * (3.0 * a).sin();
        disc.push((
            [0.1 * a.cos(), 0.1 * a.sin(), 0.01 * (5.0 * a).cos()],
            [a, 0.3],
        ));
        disc.push(([r * a.cos(), r * a.sin(), 0.0], [a, 0.9]));
    }
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for i in 0..n * 2 / 3 {
        let (a, b) = (1 + 2 * i, 1 + 2 * (i + 1));
        tris.extend([[0, a, b], [a, a + 1, b + 1], [a, b + 1, b]]);
    }
    assert_eq!(tris.len() as u32, UNIT_TRIANGLES);
    let (mut pos, mut nrm, mut uv) = (Vec::new(), Vec::new(), Vec::new());
    let (mut idx0, mut idx1) = (Vec::new(), Vec::new());
    let mut centroids = Vec::new();
    for c in 0..copies {
        let c = c as f32;
        let q = Quat::from_euler(EulerRot::XYZ, 0.37 * c, 1.1 * c, -0.23 * c);
        let t = Vec3::new(c.sin() * 3.0, 0.4 * c, (0.7 * c).cos() * 2.0);
        let base = pos.len() as u32;
        for (p, u) in &disc {
            pos.push((q * Vec3::from(*p) + t).to_array());
            nrm.push((q * Vec3::Z).to_array());
            uv.push(*u);
        }
        let (mut sum, mut area) = (glam::DVec3::ZERO, 0.0);
        for t in &tris {
            let p = t.map(|v| Vec3::from(pos[(base + v) as usize]).as_dvec3());
            let a = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length();
            sum += a * (p[0] + p[1] + p[2]) / 3.0;
            area += a;
        }
        centroids.push((sum / area).as_vec3());
        idx0.extend(tris.iter().flat_map(|t| t.map(|v| v + base)));
    }
    if sheet {
        let base = pos.len() as u32;
        for j in 0..9 {
            for i in 0..9 {
                pos.push([i as f32 * 0.5, -1.0 + 0.01 * (i * j) as f32, j as f32 * 0.5]);
                nrm.push([0.0, 1.0, 0.0]);
                uv.push([i as f32, j as f32]);
            }
        }
        for j in 0..8 {
            for i in 0..8 {
                let c = base + j * 9 + i;
                idx1.extend([c, c + 9, c + 1, c + 1, c + 9, c + 10]);
            }
        }
    }

    let mut bin = Vec::new();
    let mut views = Vec::new();
    let mut view = |bytes: &[u8]| {
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            bin.len(),
            bytes.len()
        ));
        bin.extend_from_slice(bytes);
        views.len() - 1
    };
    let (lo, hi) = pos.iter().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(lo, hi), p| (lo.min(Vec3::from(*p)), hi.max(Vec3::from(*p))),
    );
    let vp = view(bytemuck::cast_slice(&pos));
    let vn = view(bytemuck::cast_slice(&nrm));
    let vu = view(bytemuck::cast_slice(&uv));
    let v0 = view(bytemuck::cast_slice(&idx0));
    let v1 = (!idx1.is_empty()).then(|| view(bytemuck::cast_slice(&idx1)));
    let count = pos.len();
    let mut accessors = vec![
        format!(
            r#"{{"bufferView":{vp},"componentType":5126,"count":{count},"type":"VEC3","min":{:?},"max":{:?}}}"#,
            lo.to_array(),
            hi.to_array()
        ),
        format!(r#"{{"bufferView":{vn},"componentType":5126,"count":{count},"type":"VEC3"}}"#),
        format!(r#"{{"bufferView":{vu},"componentType":5126,"count":{count},"type":"VEC2"}}"#),
        format!(
            r#"{{"bufferView":{v0},"componentType":5125,"count":{},"type":"SCALAR"}}"#,
            idx0.len()
        ),
    ];
    let prim = |i: usize, m: usize| {
        format!(
            r#"{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":{i},"material":{m}}}"#
        )
    };
    let mut prims = vec![prim(3, 0)];
    if let Some(v1) = v1 {
        accessors.push(format!(
            r#"{{"bufferView":{v1},"componentType":5125,"count":{},"type":"SCALAR"}}"#,
            idx1.len()
        ));
        prims.push(prim(4, 1));
    }
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{}]}}],"materials":[{{}},{{"doubleSided":true}}],
"buffers":[{{"byteLength":{}}}],"bufferViews":[{}],"accessors":[{}]}}"#,
        prims.join(","),
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
    centroids
}
