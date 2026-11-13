//! Compiles orr_gpu's own kernels, publishes its shader root, which holds the `rtas` and
//! `sequence` libraries, and the Rust types of the `rtas` records, sets `cfg(orr_omm)` when the
//! shaders are built with micromaps, and with the `nsight-perf` feature generates the Nsight
//! Perf SDK bindings that counter frames use. Publishes `vulkan` for targets with the Vulkan
//! backend, for crates built on its interop.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

/// The `rtas` records, mirrored in `crate::rtas`.
const RTAS_RECORDS: &[&str] = &[
    "ClasBuild",
    "TemplateBuild",
    "Instantiate",
    "ClusterBlas",
    "Move",
    "PtlasWrite",
    "PtlasUpdate",
    "PtlasTranslate",
    "PtlasCommand",
];

fn main() {
    if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("apple") {
        println!("cargo::metadata=vulkan=1");
    }
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    println!("cargo::rustc-check-cfg=cfg(orr_omm)");
    let built = Build::new().and_then(|b| {
        RTAS_RECORDS
            .iter()
            .fold(b.kernels(&src), |b, t| {
                b.extern_type(t, &format!("crate::rtas::{t}"))
            })
            .extern_type("ClassicInstance", "crate::ClassicInstance")
            .publish_types("::orr_gpu")
            .compile()
    });
    match built {
        Ok(out) => {
            if out.omm {
                println!("cargo::rustc-cfg=orr_omm");
            }
        }
        Err(e) => panic!("{e}"),
    }
    #[cfg(feature = "nsight-perf")]
    perf_bindings();
}

/// Bindings to the Nsight Perf SDK's headers at `NVPERF_SDK_PATH` (the SDK or its `NvPerf`
/// folder), and each struct's `NVPA_STRUCT_SIZE` as its `StructSize::SIZE`, which the SDK
/// reads as the struct's version.
#[cfg(feature = "nsight-perf")]
fn perf_bindings() {
    const HEADERS: [&str; 6] = [
        "nvperf_common.h",
        "nvperf_host.h",
        "nvperf_target.h",
        "nvperf_device_target.h",
        "nvperf_vulkan_host.h",
        "nvperf_vulkan_target.h",
    ];
    println!("cargo::rerun-if-env-changed=NVPERF_SDK_PATH");
    if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        return;
    }
    let sdk = env::var_os("NVPERF_SDK_PATH")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("the nsight-perf feature needs NVPERF_SDK_PATH set to the Nsight Perf SDK")
        });
    let include = [sdk.join("include"), sdk.join("NvPerf").join("include")]
        .into_iter()
        .find(|d| d.join("nvperf_host.h").is_file())
        .unwrap_or_else(|| {
            panic!(
                "NVPERF_SDK_PATH is {}, which holds neither include/nvperf_host.h nor \
                 NvPerf/include/nvperf_host.h",
                sdk.display()
            )
        });
    let mut source = String::new();
    let mut sized = std::collections::BTreeMap::new();
    for h in HEADERS {
        let path = include.join(h);
        println!("cargo::rerun-if-changed={}", path.display());
        source += &format!("#include <{h}>\n");
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        // `#define <NAME> NVPA_STRUCT_SIZE(<struct>, <last field>)`
        for line in text.lines() {
            let Some((head, rest)) = line.split_once("NVPA_STRUCT_SIZE(") else {
                continue;
            };
            let head = head.trim();
            if !head.starts_with("#define ") || head == "#define" {
                continue;
            }
            let args = rest
                .split_once(')')
                .expect("NVPA_STRUCT_SIZE's arguments")
                .0;
            let ty = args.split(',').next().unwrap().trim();
            sized
                .entry(ty.to_owned())
                .or_insert_with(|| args.to_owned());
        }
    }
    source += "enum orr_struct_size {\n";
    for (ty, args) in &sized {
        source += &format!("    ORR_SIZE_{ty} = NVPA_STRUCT_SIZE({args}),\n");
    }
    source += "};\n";
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    bindgen::Builder::default()
        .header_contents("nvperf.h", &source)
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_type("NV(PW|PA)_.*")
        .allowlist_var("NV(PW|PA)_.*")
        .allowlist_type("orr_struct_size")
        .constified_enum_module("orr_struct_size")
        .layout_tests(false)
        .generate()
        .expect("Nsight Perf SDK bindings (bindgen needs libclang)")
        .write_to_file(out.join("nvperf.rs"))
        .expect("write Nsight Perf SDK bindings");
    let impls: String = sized
        .keys()
        .map(|ty| {
            format!(
                "impl StructSize for {ty} {{ const SIZE: usize = orr_struct_size::ORR_SIZE_{ty} as usize; }}\n"
            )
        })
        .collect();
    std::fs::write(out.join("nvperf_sizes.rs"), impls).expect("write Nsight Perf SDK sizes");
}
