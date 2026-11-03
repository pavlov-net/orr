//! Publishes the shader roots, link points and Rust types of orr's crates, so consumers'
//! builds read `DEP_ORR_SHADER_DIRS`, `DEP_ORR_LINK_POINTS` and `DEP_ORR_TYPES`.

fn main() {
    let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name}"));
    let crates = ["DEP_ORR_SCENE", "DEP_ORR_LIGHTING"];
    let joined = |what: &str| {
        crates
            .iter()
            .map(|c| var(&format!("{c}_{what}")))
            .collect::<Vec<_>>()
            .join(";")
    };
    println!("cargo:shader_dirs={}", joined("SHADER_DIRS"));
    println!("cargo:link_points={}", joined("LINK_POINTS"));
    let moves = [
        ("::orr_scene", "::orr::orr_scene"),
        ("::orr_lighting", "::orr::orr_lighting"),
        ("::orr_pack", "::orr::orr_pack"),
    ];
    if let Err(e) = orr_shaders::build::reexport_types(&joined("TYPES"), &moves) {
        panic!("{e}");
    }
}
