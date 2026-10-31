//! A skinned creature the examples cook from glTF: a tapered tentacle over a chain of joints
//! up its axis, with a looping `wave` clip that sends a swirl from its base to its tip.

use std::path::Path;

use orr_math::{Quat, Vec3};

/// Joints in the chain, the root at the base.
pub const JOINTS: usize = 7;
/// m between consecutive joints.
pub const SEGMENT: f32 = 0.4;
/// m from the base to the tip.
pub const HEIGHT: f32 = SEGMENT * (JOINTS - 1) as f32 + 0.2;
/// s of one loop of the `wave` clip.
pub const PERIOD: f32 = 2.0;
/// The farthest any vertex moves from its rest position in the clip, rounded up, m.
pub const REACH: f32 = 2.5;

/// The little-endian bytes of 4-byte values.
fn le(values: impl Iterator<Item = [u8; 4]>) -> Vec<u8> {
    values.flatten().collect()
}

/// The rotation of joint `j` (from 1 up) relative to its parent `t` seconds into the wave.
fn bend(j: usize, t: f32) -> Quat {
    let phase = std::f32::consts::TAU * t / PERIOD - 0.9 * j as f32;
    Quat::from_rotation_z(0.32 * phase.sin()) * Quat::from_rotation_x(0.22 * phase.cos())
}

/// Writes the creature as a GLB to `path`: `rings` rings of `sides` vertices, radius
/// tapering from `radius` at the base, weights blending from joint to joint with height, a
/// base color, and the `wave` clip sampled 16 times a loop.
pub fn write_glb(path: &Path, rings: u32, sides: u32, radius: f32, color: [f32; 3]) {
    let mut positions: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut joints: Vec<u8> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let mut vertex = |p: Vec3, n: Vec3| {
        positions.extend(p.to_array());
        normals.extend(n.to_array());
        let s = (p.y / SEGMENT).clamp(0.0, (JOINTS - 1) as f32);
        let j = (s as usize).min(JOINTS - 2);
        let u = (s - j as f32).clamp(0.0, 1.0);
        joints.extend([j as u8, j as u8 + 1, 0, 0]);
        weights.extend([1.0 - u, u, 0.0, 0.0]);
    };
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let y = HEIGHT * v;
        let width = radius * (1.0 - 0.75 * v);
        for k in 0..sides {
            let a = std::f32::consts::TAU * k as f32 / sides as f32;
            let n = Vec3::new(a.cos(), 0.0, -a.sin());
            vertex(Vec3::new(0.0, y, 0.0) + n * width, n);
        }
    }
    vertex(Vec3::ZERO, Vec3::NEG_Y);
    vertex(Vec3::new(0.0, HEIGHT, 0.0), Vec3::Y);
    let bottom = (rings + 1) * sides;
    let top = bottom + 1;
    let at = |r: u32, k: u32| r * sides + k % sides;
    let mut indices: Vec<u32> = Vec::new();
    for r in 0..rings {
        for k in 0..sides {
            let (a, b, c, d) = (at(r, k), at(r, k + 1), at(r + 1, k + 1), at(r + 1, k));
            indices.extend([a, b, c, a, c, d]);
        }
    }
    for k in 0..sides {
        indices.extend([bottom, at(0, k + 1), at(0, k)]);
        indices.extend([top, at(rings, k), at(rings, k + 1)]);
    }
    let vertices = positions.len() / 3;
    // Joint j sits at height j x SEGMENT in the bind pose.
    let inverse_binds: Vec<f32> = (0..JOINTS)
        .flat_map(|j| {
            let y = -(j as f32) * SEGMENT;
            [
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., y, 0., 1.,
            ]
        })
        .collect();
    let keys = 17;
    let times: Vec<f32> = (0..keys)
        .map(|i| PERIOD * i as f32 / (keys - 1) as f32)
        .collect();

    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut add = |bytes: &[u8]| {
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            bin.len(),
            bytes.len()
        ));
        bin.extend_from_slice(bytes);
        bin.resize(bin.len().next_multiple_of(4), 0);
        views.len() - 1
    };
    let mut accessors = vec![
        format!(
            r#"{{"bufferView":{},"componentType":5126,"count":{vertices},"type":"VEC3","min":[{lo},0,{lo}],"max":[{radius},{HEIGHT},{radius}]}}"#,
            add(&le(positions.iter().map(|f| f.to_le_bytes()))),
            lo = -radius,
        ),
        format!(
            r#"{{"bufferView":{},"componentType":5126,"count":{vertices},"type":"VEC3"}}"#,
            add(&le(normals.iter().map(|f| f.to_le_bytes())))
        ),
        format!(
            r#"{{"bufferView":{},"componentType":5121,"count":{vertices},"type":"VEC4"}}"#,
            add(&joints)
        ),
        format!(
            r#"{{"bufferView":{},"componentType":5126,"count":{vertices},"type":"VEC4"}}"#,
            add(&le(weights.iter().map(|f| f.to_le_bytes())))
        ),
        format!(
            r#"{{"bufferView":{},"componentType":5125,"count":{},"type":"SCALAR"}}"#,
            add(&le(indices.iter().map(|i| i.to_le_bytes()))),
            indices.len()
        ),
        format!(
            r#"{{"bufferView":{},"componentType":5126,"count":{JOINTS},"type":"MAT4"}}"#,
            add(&le(inverse_binds.iter().map(|f| f.to_le_bytes())))
        ),
        format!(
            r#"{{"bufferView":{},"componentType":5126,"count":{keys},"type":"SCALAR","min":[0],"max":[{PERIOD}]}}"#,
            add(&le(times.iter().map(|f| f.to_le_bytes())))
        ),
    ];
    let mut samplers = Vec::new();
    let mut channels = Vec::new();
    for j in 1..JOINTS {
        let rotations: Vec<f32> = times.iter().flat_map(|&t| bend(j, t).to_array()).collect();
        accessors.push(format!(
            r#"{{"bufferView":{},"componentType":5126,"count":{keys},"type":"VEC4"}}"#,
            add(&le(rotations.iter().map(|f| f.to_le_bytes())))
        ));
        samplers.push(format!(
            r#"{{"input":6,"output":{},"interpolation":"LINEAR"}}"#,
            accessors.len() - 1
        ));
        channels.push(format!(
            r#"{{"sampler":{},"target":{{"node":{j},"path":"rotation"}}}}"#,
            samplers.len() - 1
        ));
    }
    let nodes: Vec<String> = (0..JOINTS)
        .map(|j| {
            let children = if j + 1 < JOINTS {
                format!(r#","children":[{}]"#, j + 1)
            } else {
                String::new()
            };
            let y = if j == 0 { 0.0 } else { SEGMENT };
            format!(r#"{{"name":"joint {j}","translation":[0,{y},0]{children}}}"#)
        })
        .chain([r#"{"name":"body","mesh":0,"skin":0}"#.to_string()])
        .collect();
    let joint_list: Vec<String> = (0..JOINTS).map(|j| j.to_string()).collect();
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0,{JOINTS}]}}],
"nodes":[{nodes}],
"skins":[{{"joints":[{joints}],"inverseBindMatrices":5}}],
"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[{r},{g},{b},1],"metallicFactor":0,"roughnessFactor":0.55}}}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"JOINTS_0":2,"WEIGHTS_0":3}},"indices":4,"material":0}}]}}],
"animations":[{{"name":"wave","samplers":[{samplers}],"channels":[{channels}]}}],
"accessors":[{accessors}],
"bufferViews":[{views}],
"buffers":[{{"byteLength":{len}}}]}}"#,
        nodes = nodes.join(","),
        joints = joint_list.join(","),
        r = color[0],
        g = color[1],
        b = color[2],
        samplers = samplers.join(","),
        channels = channels.join(","),
        accessors = accessors.join(","),
        views = views.join(","),
        len = bin.len(),
    );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend(b"glTF");
    glb.extend(2u32.to_le_bytes());
    glb.extend(((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    glb.extend((json.len() as u32).to_le_bytes());
    glb.extend(b"JSON");
    glb.extend(&json);
    glb.extend((bin.len() as u32).to_le_bytes());
    glb.extend(b"BIN\0");
    glb.extend(&bin);
    std::fs::write(path, glb).expect("write the creature");
}
