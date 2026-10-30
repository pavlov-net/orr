//! Fixtures shared by the cook tests.

use std::path::PathBuf;

/// A GLB with two sections (two quads, materials 0 and 1), written once per test run.
pub(crate) fn block_glb() -> PathBuf {
    static PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    PATH.get_or_init(write_block_glb).clone()
}

fn write_block_glb() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("block.glb");
    let pos: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let idx: [u32; 12] = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6];
    let mut bin = Vec::new();
    bin.extend_from_slice(bytemuck::cast_slice(&pos));
    bin.extend_from_slice(bytemuck::cast_slice(&idx));
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}},{{"attributes":{{"POSITION":0}},"indices":2,"material":1}}]}}],
"materials":[{{}},{{"doubleSided":true}}],
"buffers":[{{"byteLength":{}}}],
"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":96}},{{"buffer":0,"byteOffset":96,"byteLength":48}}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":8,"type":"VEC3","min":[0,0,0],"max":[1,1,1]}},
 {{"bufferView":1,"byteOffset":0,"componentType":5125,"count":6,"type":"SCALAR"}},
 {{"bufferView":1,"byteOffset":24,"componentType":5125,"count":6,"type":"SCALAR"}}]}}"#,
        bin.len()
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
    std::fs::write(&path, glb).unwrap();
    path
}
