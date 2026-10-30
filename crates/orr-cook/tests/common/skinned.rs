//! A skinned GLB: a square tube over a three-joint chain under a rotated, scaled root, a
//! rigid prop under the middle joint, and three animations. Only arithmetic builds it, so
//! its bytes are the same on every platform.
//!
//! - Nodes: `root` (translated, 90 degrees about x, scale 0.5) with children `hip` and `body`;
//!   `hip` > `knee` > `foot`, and `prop` (a mesh without a skin) under `knee`. `body` holds
//!   the skinned tube; its own transform is ignored, as glTF asks.
//! - Skin joints in the order foot, hip, knee; the bind pose has the knee straight and the
//!   rest pose bends it 20 degrees. Weights blend hip to knee along the tube and put 0.1 on
//!   the foot in a second `JOINTS_1`/`WEIGHTS_1` set.
//! - Animations: `bend` (with `bend` set: knee rotation `LINEAR`, hip translation
//!   `CUBICSPLINE`, foot scale `STEP`; else knee translation `LINEAR`), an unnamed one moving
//!   the root from 0.25 s (rotating it too with `bend`), and `drift`, which only moves
//!   `body` and so makes no clip.

use std::path::{Path, PathBuf};

use glam::{Mat4, Quat, Vec3};

const ROOT_T: [f32; 3] = [1.0, 0.0, 0.0];
const ROOT_R: [f32; 4] = [0.707_106_77, 0.0, 0.0, 0.707_106_77];
const ROOT_S: f32 = 0.5;
/// 20 and 70 degrees about z; 45 degrees about y.
const Z20: [f32; 4] = [0.0, 0.0, 0.173_648_18, 0.984_807_7];
const Z70: [f32; 4] = [0.0, 0.0, 0.573_576_44, 0.819_152_06];
const Y45: [f32; 4] = [0.0, 0.382_683_43, 0.0, 0.923_879_5];

pub(crate) fn skinned_glb(name: &str, bend: bool) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.glb"));
    write(&path, bend);
    path
}

#[derive(Default)]
struct Bin {
    bin: Vec<u8>,
    views: Vec<String>,
    accessors: Vec<String>,
}

impl Bin {
    fn add(&mut self, bytes: &[u8], count: usize, ty: &str, comp: u32, extra: &str) -> usize {
        self.views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            self.bin.len(),
            bytes.len()
        ));
        self.bin.extend_from_slice(bytes);
        self.bin.resize(self.bin.len().next_multiple_of(4), 0);
        self.accessors.push(format!(
            r#"{{"bufferView":{},"componentType":{comp},"count":{count},"type":"{ty}"{extra}}}"#,
            self.views.len() - 1
        ));
        self.accessors.len() - 1
    }

    fn f32s(&mut self, v: &[f32], ty: &str, per: usize) -> usize {
        self.add(bytemuck::cast_slice(v), v.len() / per, ty, 5126, "")
    }

    fn positions(&mut self, p: &[[f32; 3]]) -> usize {
        let lo = [0, 1, 2].map(|a| p.iter().map(|v| v[a]).fold(f32::MAX, f32::min));
        let hi = [0, 1, 2].map(|a| p.iter().map(|v| v[a]).fold(f32::MIN, f32::max));
        let extra = format!(r#","min":{lo:?},"max":{hi:?}"#);
        self.add(bytemuck::cast_slice(p), p.len(), "VEC3", 5126, &extra)
    }
}

fn write(path: &Path, bend: bool) {
    let root = Mat4::from_scale_rotation_translation(
        Vec3::splat(ROOT_S),
        Quat::from_array(ROOT_R),
        ROOT_T.into(),
    );
    // Bind pose: the chain straight up root-local y.
    let bind = [
        root,
        root * Mat4::from_translation(Vec3::Y),
        root * Mat4::from_translation(Vec3::Y * 2.0),
    ];
    // Skin order foot, hip, knee.
    let ibm: Vec<[[f32; 4]; 4]> = [bind[2], bind[0], bind[1]]
        .iter()
        .map(|m| m.inverse().to_cols_array_2d())
        .collect();

    let mut b = Bin::default();
    let (mut pos, mut nrm, mut j0, mut w0, mut j1, mut w1, mut idx) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let corners = [[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1]];
    let rings = 9;
    for r in 0..rings {
        let y = r as f32 * 0.25;
        let (hip, knee, foot): (f32, f32, f32) = if y <= 0.5 {
            (1.0, 0.0, 0.0)
        } else if y < 1.5 {
            (1.5 - y, y - 0.5, 0.0)
        } else if y < 1.75 {
            (0.0, 1.0, 0.0)
        } else {
            (0.0, 0.9, 0.1)
        };
        for [x, z] in corners {
            pos.push(root.transform_point3(Vec3::new(x, y, z)).to_array());
            let n = root.transform_vector3(Vec3::new(x, 0.0, z)).to_array();
            nrm.push(n);
            j0.push([1u16, 2, 0, 0]);
            w0.push([hip, knee, 0.0, 0.0]);
            j1.push([0u16, 0, 0, 0]);
            w1.push([foot, 0.0, 0.0, 0.0]);
        }
    }
    for r in 0..rings - 1 {
        for c in 0..4u32 {
            let (a, d) = (r * 4 + c, r * 4 + (c + 1) % 4);
            idx.extend([a, d, a + 4, d, d + 4, a + 4]);
        }
    }
    let tube_pos = b.positions(&pos);
    let tube_nrm = b.add(bytemuck::cast_slice(&nrm), nrm.len(), "VEC3", 5126, "");
    let j0a = b.add(bytemuck::cast_slice(&j0), j0.len(), "VEC4", 5123, "");
    let w0a = b.add(bytemuck::cast_slice(&w0), w0.len(), "VEC4", 5126, "");
    let j1a = b.add(bytemuck::cast_slice(&j1), j1.len(), "VEC4", 5123, "");
    let w1a = b.add(bytemuck::cast_slice(&w1), w1.len(), "VEC4", 5126, "");
    let tube_idx = b.add(bytemuck::cast_slice(&idx), idx.len(), "SCALAR", 5125, "");

    // A box without normals, so the importer splits it into flat corners.
    let prop: Vec<[f32; 3]> = (0..8)
        .map(|i| [0, 1, 2].map(|a| if i >> a & 1 == 1 { 0.05 } else { -0.05 }))
        .collect();
    let prop_idx: [u32; 36] = [
        0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6,
        1, 3, 5, 3, 7, 5,
    ];
    let prop_pos = b.positions(&prop);
    let prop_ix = b.add(bytemuck::cast_slice(&prop_idx), 36, "SCALAR", 5125, "");
    let ibm_a = b.add(bytemuck::cast_slice(&ibm), 3, "MAT4", 5126, "");

    // Animations: (sampler json, channel json) lists.
    let mut anims = Vec::new();
    let sampler =
        |b: &mut Bin, times: &[f32], values: &[f32], ty: &str, per: usize, interp: &str| {
            let t = b.f32s(times, "SCALAR", 1);
            let v = b.f32s(values, ty, per);
            format!(r#"{{"input":{t},"output":{v},"interpolation":"{interp}"}}"#)
        };
    let channel = |s: usize, node: usize, path: &str| {
        format!(r#"{{"sampler":{s},"target":{{"node":{node},"path":"{path}"}}}}"#)
    };
    let knee_rotation = if bend {
        format!(r#","rotation":{Z20:?}"#)
    } else {
        String::new()
    };
    if bend {
        let s0 = sampler(
            &mut b,
            &[0.0, 0.4, 1.0],
            &[Z20, Z70, Z20].concat(),
            "VEC4",
            4,
            "LINEAR",
        );
        let hip = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0,
            0.0,
        ];
        let s1 = sampler(&mut b, &[0.0, 1.0], &hip, "VEC3", 3, "CUBICSPLINE");
        let s2 = sampler(&mut b, &[0.0, 0.5], &[1.0; 6], "VEC3", 3, "STEP");
        anims.push(format!(
            r#"{{"name":"bend","samplers":[{s0},{s1},{s2}],"channels":[{},{},{}]}}"#,
            channel(0, 3, "rotation"),
            channel(1, 2, "translation"),
            channel(2, 4, "scale")
        ));
    } else {
        let s0 = sampler(
            &mut b,
            &[0.0, 0.5, 1.0],
            &[0.0, 1.0, 0.0, 0.2, 1.0, 0.0, 0.0, 1.0, 0.1],
            "VEC3",
            3,
            "LINEAR",
        );
        anims.push(format!(
            r#"{{"name":"bend","samplers":[{s0}],"channels":[{}]}}"#,
            channel(0, 3, "translation")
        ));
    }
    let s0 = sampler(
        &mut b,
        &[0.25, 0.75],
        &[1.0, 0.0, 0.0, 1.0, 0.5, 0.0],
        "VEC3",
        3,
        "LINEAR",
    );
    if bend {
        let turned = (Quat::from_array(Y45) * Quat::from_array(ROOT_R)).to_array();
        let s1 = sampler(
            &mut b,
            &[0.25, 0.75],
            &[ROOT_R, turned].concat(),
            "VEC4",
            4,
            "LINEAR",
        );
        anims.push(format!(
            r#"{{"samplers":[{s0},{s1}],"channels":[{},{}]}}"#,
            channel(0, 0, "translation"),
            channel(1, 0, "rotation")
        ));
    } else {
        anims.push(format!(
            r#"{{"samplers":[{s0}],"channels":[{}]}}"#,
            channel(0, 0, "translation")
        ));
    }
    let s0 = sampler(
        &mut b,
        &[0.0, 1.0],
        &[0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        "VEC3",
        3,
        "LINEAR",
    );
    anims.push(format!(
        r#"{{"name":"drift","samplers":[{s0}],"channels":[{}]}}"#,
        channel(0, 1, "translation")
    ));

    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],
"nodes":[
 {{"name":"root","translation":{ROOT_T:?},"rotation":{ROOT_R:?},"scale":[{ROOT_S},{ROOT_S},{ROOT_S}],"children":[2,1]}},
 {{"name":"body","mesh":0,"skin":0,"translation":[5,5,5]}},
 {{"name":"hip","children":[3]}},
 {{"name":"knee","translation":[0,1,0]{knee_rotation},"children":[4,5]}},
 {{"name":"foot","translation":[0,1,0]}},
 {{"name":"prop","mesh":1,"translation":[0.2,0.5,0]}}],
"skins":[{{"joints":[4,2,3],"inverseBindMatrices":{ibm_a}}}],
"meshes":[
 {{"primitives":[{{"attributes":{{"POSITION":{tube_pos},"NORMAL":{tube_nrm},"JOINTS_0":{j0a},"WEIGHTS_0":{w0a},"JOINTS_1":{j1a},"WEIGHTS_1":{w1a}}},"indices":{tube_idx}}}]}},
 {{"primitives":[{{"attributes":{{"POSITION":{prop_pos}}},"indices":{prop_ix}}}]}}],
"animations":[{}],
"buffers":[{{"byteLength":{}}}],"bufferViews":[{}],"accessors":[{}]}}"#,
        anims.join(","),
        b.bin.len(),
        b.views.join(","),
        b.accessors.join(",")
    );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&((12 + 8 + json.len() + 8 + b.bin.len()) as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(b.bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&b.bin);
    std::fs::write(path, glb).unwrap();
}
