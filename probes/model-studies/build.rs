// Builds the OpenPBR reference shim against an external openpbr-bsdf checkout and GLM.
// Also links Polyphony's GT7 sample for the tonemapper port check. OPENPBR_BSDF, GLM and
// GT7_SAMPLE locate them; all default to the fetched copies in .tools/.
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.tools");
    let openpbr = std::env::var("OPENPBR_BSDF").map(PathBuf::from).unwrap_or(root.join("openpbr-bsdf"));
    let glm = std::env::var("GLM").map(PathBuf::from).unwrap_or(root.join("glm"));
    let gt7 = std::env::var("GT7_SAMPLE").map(PathBuf::from).unwrap_or(root.join("display/gt7"));
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-changed=cpp/gt7_ref.cpp");
    println!("cargo:rerun-if-env-changed=OPENPBR_BSDF");
    println!("cargo:rerun-if-env-changed=GLM");
    cc::Build::new()
        .cpp(true)
        .file("cpp/shim.cpp")
        .include(&openpbr)
        .include(&glm)
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .flag_if_supported("/fp:precise")
        .opt_level(2)
        .compile("openpbr_shim");
    cc::Build::new().cpp(true).file("cpp/gt7_ref.cpp").include(&gt7).flag_if_supported("/std:c++17").opt_level(2).compile("gt7_ref");
}
