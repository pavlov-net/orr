//! P8: `gltf::Gltf::from_slice` on each file; on rejection, the pre-parse workaround (drop the
//! extensions the cook decodes itself from `extensionsRequired`) and `from_slice_without_validation`.

use std::path::Path;
use std::time::Instant;

const COOK_DECODED: &[&str] = &["EXT_meshopt_compression", "KHR_meshopt_compression", "KHR_mesh_quantization"];

fn rebuild(bytes: &[u8], json: &[u8]) -> Vec<u8> {
    if &bytes[0..4] != b"glTF" {
        return json.to_vec();
    }
    let jl = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let rest = &bytes[20 + jl..];
    let mut j = json.to_vec();
    while j.len() % 4 != 0 {
        j.push(b' ');
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&((12 + 8 + j.len() + rest.len()) as u32).to_le_bytes());
    out.extend_from_slice(&(j.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&j);
    out.extend_from_slice(rest);
    out
}

pub fn run(files: &[String]) {
    let (mut ok, mut rejected, mut fixed) = (0, 0, 0);
    for f in files {
        let bytes = std::fs::read(f).unwrap();
        let name = Path::new(f).file_name().unwrap().to_string_lossy().to_string();
        let doc = crate::gl::open(Path::new(f));
        let req: Vec<String> = doc.json["extensionsRequired"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_str().unwrap().to_owned()).collect())
            .unwrap_or_default();
        let t = Instant::now();
        match gltf::Gltf::from_slice(&bytes) {
            Ok(g) => {
                ok += 1;
                println!("{name}: ok ({} meshes, {:.3} s) required {req:?}", g.meshes().count(), crate::secs(t));
            }
            Err(e) => {
                rejected += 1;
                let mut j = doc.json.clone();
                if let Some(a) = j["extensionsRequired"].as_array_mut() {
                    a.retain(|v| !COOK_DECODED.contains(&v.as_str().unwrap()));
                }
                let patched = rebuild(&bytes, &serde_json::to_vec(&j).unwrap());
                let t2 = Instant::now();
                let after = gltf::Gltf::from_slice(&patched);
                let nv = gltf::Gltf::from_slice_without_validation(&bytes).is_ok();
                if after.is_ok() {
                    fixed += 1;
                }
                println!(
                    "{name}: REJECTED ({e}) required {req:?}; after dropping cook-decoded required: {} ({:.3} s); without_validation: {}",
                    match &after {
                        Ok(g) => format!("ok, {} meshes", g.meshes().count()),
                        Err(e) => format!("still rejected: {e}"),
                    },
                    crate::secs(t2),
                    if nv { "ok" } else { "rejected" }
                );
            }
        }
    }
    println!("total {}: ok {ok}, rejected {rejected}, fixed by pre-parse {fixed}", files.len());
}
