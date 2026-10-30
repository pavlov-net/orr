//! Compiles the probe shaders with the pinned standalone Slang (`ORR_SLANGC`, default
//! `.tools/slang-2026.20`), validating each with the Vulkan SDK's spirv-val.

use std::path::{Path, PathBuf};
use std::process::Command;

fn run(cmd: &mut Command) {
    let status = cmd.status().unwrap_or_else(|e| panic!("failed to spawn {cmd:?}: {e}"));
    assert!(status.success(), "{cmd:?} failed");
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shaders = manifest.join("shaders");
    let slangc = std::env::var("ORR_SLANGC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../../.tools/slang-2026.20/bin/slangc.exe"));
    let sdk_bin = PathBuf::from(std::env::var("VULKAN_SDK").expect("VULKAN_SDK")).join("Bin");
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=ORR_SLANGC");
    println!("cargo:rerun-if-env-changed=PROBE_FP_PRECISE");
    let precise = std::env::var("PROBE_FP_PRECISE").is_ok_and(|v| v == "1");

    let slang = |file: &str, entry: &str, stage: &str, defines: &[&str], name: &str| {
        let heap = !defines.contains(&"ROOT_PUSH=1");
        let dst = out.join(format!("{name}.spv"));
        let mut cmd = Command::new(&slangc);
        cmd.arg(shaders.join(file))
            .args(["-target", "spirv", "-profile", "spirv_1_6", "-emit-spirv-directly"])
            .args(["-fvk-use-entrypoint-name", "-fvk-use-c-layout", "-matrix-layout-row-major"])
            .args(["-capability", "spvRayQueryKHR"])
            .args(["-entry", entry, "-stage", stage, "-o"])
            .arg(&dst);
        if precise {
            cmd.args(["-fp-mode", "precise"]);
        }
        if heap {
            cmd.args(["-capability", "spvDescriptorHeapEXT"]);
        }
        if stage != "compute" {
            cmd.args(["-capability", "spvShaderInvocationReorderEXT"]);
        }
        for d in defines {
            cmd.arg(format!("-D{d}"));
        }
        run(&mut cmd);
        validate(&sdk_bin, &dst);
    };
    let rg = "raygeneration";
    slang("pt.slang", "compute_main", "compute", &["STAGE_COMPUTE=1"], "pt_a");
    slang("pt.slang", "compute_main", "compute", &["STAGE_COMPUTE=1", "STATS=1"], "pt_a_stats");
    for (x, y) in [(8, 8), (16, 8), (16, 16), (32, 1)] {
        let (dx, dy) = (format!("WG_X={x}"), format!("WG_Y={y}"));
        slang("pt.slang", "compute_main", "compute", &["STAGE_COMPUTE=1", dx.as_str(), dy.as_str()], &format!("pt_a_{x}x{y}"));
    }
    slang("pt.slang", "raygen_main", rg, &[], "pt_b0");
    slang("pt.slang", "raygen_main", rg, &["REORDER=1", "KEY_BITS=3"], "pt_b");
    slang("pt.slang", "raygen_main", rg, &["REORDER=1", "KEY_BITS=0"], "pt_bh");
    slang("pt.slang", "raygen_main", rg, &["REORDER=1", "KEY_BITS=3", "PRE_HINT=1"], "pt_eb");
    slang("pt.slang", "raygen_main", rg, &["HINT_ONLY=1", "KEY_BITS=3"], "pt_bk");
    slang("pt.slang", "raygen_main", rg, &["HINT_ONLY=1", "KEY_BITS=3", "PRE_HINT=1"], "pt_ebk");
    slang("pt.slang", "raygen_main", rg, &["TRACE_PIPE=1"], "pt_c0");
    slang("pt.slang", "raygen_main", rg, &["TRACE_PIPE=1", "REORDER=1", "KEY_BITS=3"], "pt_c");
    slang("pt.slang", "raygen_main", rg, &["TRACE_PIPE=1", "REORDER=1", "KEY_BITS=0"], "pt_ch");
    slang("pt.slang", "raygen_main", rg, &["TRACE_PIPE=1", "REORDER=1", "KEY_BITS=3", "PRE_HINT=1"], "pt_ec");
    slang("hit.slang", "anyhit_main", "anyhit", &[], "anyhit");
    slang("hit.slang", "anyhit_main", "anyhit", &["ROOT_PUSH=1", "MINI_ALPHA_SIMPLE=1"], "anyhit_push");
    slang("hit.slang", "miss_main", "miss", &["ROOT_PUSH=1"], "miss_push");
    slang("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_SIMPLE=1"], "anyhit_simple");
    slang("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_ROOT=1"], "anyhit_root");
    slang("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_ROOT0=1"], "anyhit_root0");
    slang("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_GEOM=138"], "anyhit_geom");
    slang("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_DUMP=1"], "anyhit_dump");
    slang("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_CHECK=138"], "anyhit_check");
    slang("hit.slang", "anyhit_main", "anyhit", &["ROOT_PUSH=1", "MINI_ALPHA_ROOT=1"], "anyhit_push_root");
    let glsl = |file: &str, name: &str| {
        let dst = out.join(format!("{name}.spv"));
        run(Command::new(sdk_bin.join("glslangValidator.exe")).args(["-V", "--target-env", "vulkan1.4", "-e", "raygen_main", "--source-entrypoint", "main"]).arg(shaders.join(file)).arg("-o").arg(&dst));
        validate(&sdk_bin, &dst);
    };
    glsl("mini_from_query.rgen", "glsl_mini7_push");
    for n in 1..=15 {
        if n == 12 {
            continue;
        }
        let mini = format!("MINI={n}");
        slang("mini.slang", "raygen_main", rg, &[mini.as_str()], &format!("mini{n}"));
        slang("mini.slang", "raygen_main", rg, &[mini.as_str(), "ROOT_PUSH=1"], &format!("mini{n}_push"));
    }
    slang("hit.slang", "miss_main", "miss", &[], "miss");
}

fn validate(sdk_bin: &Path, spv: &Path) {
    run(Command::new(sdk_bin.join("spirv-val.exe"))
        .args(["--target-env", "vulkan1.4", "--scalar-block-layout"])
        .arg(spv));
}
