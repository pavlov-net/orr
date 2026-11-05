//! Three quads of `NormalTangentMirrorTest.glb` with the tangents it stores, which are
//! MikkTSpace's: one plain, one with mirrored uvs, one facing -z.
//!
//! Source: KhronosGroup/glTF-Sample-Assets, commit 8f492390ddc7a2ae221a1369ef560589cf8c70fb,
//! `Models/NormalTangentMirrorTest`. (c) 2018 Analytical Graphics, Inc., Ed Mackey, licensed
//! under CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/legalcode>. Values are the
//! model's, rounded by printing; only the choice of quads is ours.

struct Quad {
    tangent: [f32; 4],
    normal: [f32; 3],
    positions: [[f32; 3]; 4],
    uvs: [[f32; 2]; 4],
}

/// Corners of each quad's two triangles, as the model winds them.
const INDICES: [u32; 6] = [0, 1, 2, 0, 3, 1];

const QUADS: [Quad; 3] = [
    Quad {
        tangent: [0.48970193, -0.87188995, 0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
        positions: [
            [-0.80462575, 0.55, -4.1523847e-8],
            [-1.1046257, 0.25, -1.8874475e-8],
            [-0.80462575, 0.25, -1.8874475e-8],
            [-1.1046257, 0.55, -4.1523847e-8],
        ],
        uvs: [
            [0.4506911, 0.17217863],
            [0.49390012, 0.32611632],
            [0.5492645, 0.227543],
            [0.39532673, 0.27075207],
        ],
    },
    Quad {
        tangent: [-0.52316403, 0.852232, 0.0, -1.0],
        normal: [0.0, 0.0, 1.0],
        positions: [
            [-0.47931322, 0.15000004, 1.1324688e-8],
            [-0.7793132, -0.15000004, -1.1324688e-8],
            [-0.47931322, -0.15000004, -1.1324688e-8],
            [-0.7793132, 0.15000004, 1.1324688e-8],
        ],
        uvs: [
            [0.5868253, 0.49513894],
            [0.5496218, 0.6506374],
            [0.49047434, 0.5542864],
            [0.6459728, 0.5914899],
        ],
    },
    Quad {
        tangent: [-1.0, 0.0, 0.0, 1.0],
        normal: [0.0, 0.0, -1.0],
        positions: [
            [-0.31, -1.0500001, -0.009999894],
            [0.31000003, -1.2, -0.009999936],
            [-0.31, -1.2, -0.009999882],
            [0.31000003, -1.0500001, -0.009999948],
        ],
        uvs: [
            [0.95801836, 0.8869011],
            [0.5905804, 0.97579736],
            [0.95801836, 0.97579736],
            [0.5905804, 0.8869011],
        ],
    },
];

/// The quads as one vertex array: positions, normals, uvs, tangents and indices.
#[allow(clippy::type_complexity)]
pub(crate) fn arrays() -> (
    Vec<[f32; 3]>,
    Vec<[f32; 3]>,
    Vec<[f32; 2]>,
    Vec<[f32; 4]>,
    Vec<u32>,
) {
    let (mut p, mut n, mut uv, mut t, mut idx) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for q in &QUADS {
        let first = p.len() as u32;
        p.extend(q.positions);
        n.extend([q.normal; 4]);
        uv.extend(q.uvs);
        t.extend([q.tangent; 4]);
        idx.extend(INDICES.map(|i| first + i));
    }
    (p, n, uv, t, idx)
}
