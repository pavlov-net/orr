//! Builds the suite's kernels against orr-pack's shader root, with orr-pack's page and
//! record types standing in for the generated ones.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

const PACK_TYPES: &[(&str, &str)] = &[
    ("PageHeader", "::orr_pack::mesh::PageHeader"),
    ("Cluster", "::orr_pack::mesh::Cluster"),
    (
        "InstancePageHeader",
        "::orr_pack::instance::InstancePageHeader",
    ),
    ("InstanceRecord", "::orr_pack::instance::InstanceRecord"),
];

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let roots = env::var("DEP_ORR_PACK_SHADER_DIRS").expect("orr-pack shader roots");
    let result = Build::new().and_then(|mut b| {
        b = b.kernels(dir.join("src")).roots(&roots);
        for (slang, rust) in PACK_TYPES {
            b = b.extern_type(slang, rust);
        }
        b.compile()
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}
