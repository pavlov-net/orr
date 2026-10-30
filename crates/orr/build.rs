//! Publishes the shader roots of orr's crates, so consumers' builds read
//! `DEP_ORR_SHADER_DIRS`.

fn main() {
    let roots = std::env::var("DEP_ORR_SCENE_SHADER_DIRS").expect("orr-scene shader roots");
    println!("cargo:shader_dirs={roots}");
}
