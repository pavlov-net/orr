//! Compiles the probe shaders with the pinned standalone Slang (`ORR_SLANGC`, default
//! `.tools/slang-2026.20`) and validates each with spirv-val. `trace_cluster_omm_khr` is
//! `trace_cluster` with the `OpacityMicromapIdKHR` execution mode added by disassembling and
//! reassembling with SPIR-V tools that know SPV_KHR_opacity_micromap (`ORR_SPIRV_TOOLS`,
//! default the Vulkan SDK 1.4.363 `Bin`).

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
    let tools = std::env::var("ORR_SPIRV_TOOLS").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("C:/VulkanSDK/1.4.363.0/Bin"));
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=ORR_SLANGC");
    println!("cargo:rerun-if-env-changed=ORR_SPIRV_TOOLS");

    let slang = |file: &str, defines: &[&str], name: &str| {
        let dst = out.join(format!("{name}.spv"));
        let mut cmd = Command::new(&slangc);
        cmd.arg(shaders.join(file))
            .args(["-target", "spirv", "-profile", "spirv_1_6", "-emit-spirv-directly"])
            .args(["-fvk-use-entrypoint-name", "-fvk-use-c-layout", "-matrix-layout-row-major"])
            .args(["-entry", "main", "-stage", "compute", "-o"])
            .arg(&dst);
        for d in defines {
            cmd.arg(format!("-D{d}"));
        }
        run(&mut cmd);
        validate(&tools, &dst);
        dst
    };
    slang("args.slang", &[], "args");
    let cluster = slang("trace.slang", &["WITH_CLUSTER=1"], "trace_cluster");
    slang("trace_lss.slang", &[], "trace_lss");
    slang("canopy.slang", &[], "canopy");
    omm_khr_mode(&tools, &cluster, &out.join("trace_cluster_omm_khr.spv"));
}

/// Adds `OpExecutionModeId <entry> OpacityMicromapIdKHR <true>` with its capability and
/// extension to a compiled module.
fn omm_khr_mode(tools: &Path, src: &Path, dst: &Path) {
    let text = Command::new(tools.join("spirv-dis.exe")).arg(src).arg("--raw-id").output().expect("spirv-dis");
    assert!(text.status.success(), "spirv-dis failed");
    let text = String::from_utf8(text.stdout).unwrap();
    let entry = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("OpEntryPoint GLCompute ").map(|r| r.split_whitespace().next().unwrap().to_owned()))
        .expect("entry point");
    let bool_ty = text.lines().find_map(|l| l.trim().strip_suffix(" = OpTypeBool").map(|r| r.trim().to_owned()));
    let mut out = String::new();
    let mut caps_done = false;
    let mut mode_done = false;
    let mut const_done = false;
    for line in text.lines() {
        let t = line.trim();
        if !caps_done && t.starts_with("OpExtension") {
            out += "OpCapability RayTracingOpacityMicromapExecutionModeKHR\nOpExtension \"SPV_KHR_opacity_micromap\"\n";
            caps_done = true;
        }
        if !mode_done && (t.starts_with("OpExecutionMode ") || t.starts_with("OpExecutionModeId ")) {
            out += &format!("OpExecutionModeId {entry} OpacityMicromapIdKHR %omm_true\n");
            mode_done = true;
        }
        out += line;
        out.push('\n');
        if !const_done && bool_ty.as_deref().is_some_and(|b| t.starts_with(b) && t.ends_with("OpTypeBool")) {
            out += &format!("%omm_true = OpConstantTrue {}\n", bool_ty.as_deref().unwrap());
            const_done = true;
        }
    }
    assert!(caps_done && mode_done, "unexpected module shape");
    if !const_done {
        // No bool type in the module: declare one after the first type instruction.
        let at = out.find("= OpType").and_then(|i| out[i..].find('\n').map(|j| i + j + 1)).expect("type section");
        out.insert_str(at, "%omm_bool = OpTypeBool\n%omm_true = OpConstantTrue %omm_bool\n");
    }
    let spvasm = dst.with_extension("spvasm");
    std::fs::write(&spvasm, out).unwrap();
    run(Command::new(tools.join("spirv-as.exe")).args(["--target-env", "vulkan1.4", "--preserve-numeric-ids"]).arg(&spvasm).arg("-o").arg(dst));
    validate(tools, dst);
}

fn validate(tools: &Path, spv: &Path) {
    run(Command::new(tools.join("spirv-val.exe")).args(["--target-env", "vulkan1.4", "--scalar-block-layout"]).arg(spv));
}
