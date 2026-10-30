//! Publishes the shader roots and link points of orr's crates, so consumers' builds read
//! `DEP_ORR_SHADER_DIRS` and `DEP_ORR_LINK_POINTS`.

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
}
