//! Publishes the shader roots, link points and Rust types of orr's crates, so consumers'
//! builds read `DEP_ORR_SHADER_DIRS`, `DEP_ORR_LINK_POINTS` and `DEP_ORR_TYPES`.

fn main() {
    let moved = [
        ("::orr_scene", "::orr::orr_scene"),
        ("::orr_lighting", "::orr::orr_lighting"),
        ("::orr_gpu", "::orr::orr_scene::orr_gpu"),
        ("::orr_pack", "::orr::orr_scene::orr_pack"),
    ];
    if let Err(e) = orr_shaders::build::publish_union(&["orr_scene", "orr_lighting"], &moved) {
        panic!("{e}");
    }
}
