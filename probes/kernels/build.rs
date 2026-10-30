//! Compiles the probe shaders with the pinned standalone Slang (`ORR_SLANGC`, default
//! `.tools/slang-2026.20`), validating each with the Vulkan SDK's spirv-val.

use std::path::{Path, PathBuf};
use std::process::Command;

fn run(cmd: &mut Command) {
    let status = cmd.status().unwrap_or_else(|e| panic!("failed to spawn {cmd:?}: {e}"));
    assert!(status.success(), "{cmd:?} failed");
}

fn validate(sdk_bin: &Path, spv: &Path) {
    run(Command::new(sdk_bin.join("spirv-val.exe")).args(["--target-env", "vulkan1.4", "--scalar-block-layout"]).arg(spv));
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

    let base = |file: &str, entry: &str, stage: &str, dst: &Path| {
        let mut cmd = Command::new(&slangc);
        cmd.arg(shaders.join(file))
            .args(["-target", "spirv", "-profile", "spirv_1_6", "-emit-spirv-directly"])
            .args(["-fvk-use-entrypoint-name", "-fvk-use-c-layout", "-matrix-layout-row-major"])
            .args(["-entry", entry, "-stage", stage, "-o"])
            .arg(dst);
        cmd
    };

    // probes/ser kernels (serlive binary).
    let ser = |file: &str, entry: &str, stage: &str, defines: &[&str], name: &str| {
        let heap = !defines.contains(&"ROOT_PUSH=1");
        let dst = out.join(format!("{name}.spv"));
        let mut cmd = base(&format!("ser/{file}"), entry, stage, &dst);
        cmd.args(["-capability", "spvRayQueryKHR"]);
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
    ser("pt.slang", "compute_main", "compute", &["STAGE_COMPUTE=1"], "pt_a");
    ser("pt.slang", "compute_main", "compute", &["STAGE_COMPUTE=1", "STATS=1"], "pt_a_stats");
    for (x, y) in [(8, 8), (16, 8), (16, 16), (32, 1)] {
        let (dx, dy) = (format!("WG_X={x}"), format!("WG_Y={y}"));
        ser("pt.slang", "compute_main", "compute", &["STAGE_COMPUTE=1", dx.as_str(), dy.as_str()], &format!("pt_a_{x}x{y}"));
    }
    ser("pt.slang", "raygen_main", rg, &[], "pt_b0");
    ser("pt.slang", "raygen_main", rg, &["REORDER=1", "KEY_BITS=3"], "pt_b");
    ser("pt.slang", "raygen_main", rg, &["REORDER=1", "KEY_BITS=0"], "pt_bh");
    ser("pt.slang", "raygen_main", rg, &["REORDER=1", "KEY_BITS=3", "PRE_HINT=1"], "pt_eb");
    ser("pt.slang", "raygen_main", rg, &["HINT_ONLY=1", "KEY_BITS=3"], "pt_bk");
    ser("pt.slang", "raygen_main", rg, &["HINT_ONLY=1", "KEY_BITS=3", "PRE_HINT=1"], "pt_ebk");
    ser("pt.slang", "raygen_main", rg, &["TRACE_PIPE=1"], "pt_c0");
    ser("pt.slang", "raygen_main", rg, &["TRACE_PIPE=1", "REORDER=1", "KEY_BITS=3"], "pt_c");
    ser("pt.slang", "raygen_main", rg, &["TRACE_PIPE=1", "REORDER=1", "KEY_BITS=0"], "pt_ch");
    ser("pt.slang", "raygen_main", rg, &["TRACE_PIPE=1", "REORDER=1", "KEY_BITS=3", "PRE_HINT=1"], "pt_ec");
    for words in [4, 8, 16, 32, 64] {
        let live = format!("LIVE_WORDS={words}");
        ser("pt.slang", "compute_main", "compute", &["STAGE_COMPUTE=1", live.as_str()], &format!("pt_a_live{words}"));
        ser("pt.slang", "raygen_main", rg, &["HINT_ONLY=1", "KEY_BITS=3", live.as_str()], &format!("pt_bk_live{words}"));
    }
    ser("hit.slang", "anyhit_main", "anyhit", &[], "anyhit");
    ser("hit.slang", "anyhit_main", "anyhit", &["ROOT_PUSH=1", "MINI_ALPHA_SIMPLE=1"], "anyhit_push");
    ser("hit.slang", "miss_main", "miss", &["ROOT_PUSH=1"], "miss_push");
    ser("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_SIMPLE=1"], "anyhit_simple");
    ser("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_ROOT=1"], "anyhit_root");
    ser("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_ROOT0=1"], "anyhit_root0");
    ser("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_GEOM=138"], "anyhit_geom");
    ser("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_DUMP=1"], "anyhit_dump");
    ser("hit.slang", "anyhit_main", "anyhit", &["MINI_ALPHA_CHECK=138"], "anyhit_check");
    ser("hit.slang", "anyhit_main", "anyhit", &["ROOT_PUSH=1", "MINI_ALPHA_ROOT=1"], "anyhit_push_root");
    ser("hit.slang", "miss_main", "miss", &[], "miss");
    let dst = out.join("glsl_mini7_push.spv");
    run(Command::new(sdk_bin.join("glslangValidator.exe"))
        .args(["-V", "--target-env", "vulkan1.4", "-e", "raygen_main", "--source-entrypoint", "main"])
        .arg(shaders.join("ser/mini_from_query.rgen"))
        .arg("-o")
        .arg(&dst));
    for n in (1..=15).filter(|&n| n != 12) {
        let mini = format!("MINI={n}");
        ser("mini.slang", "raygen_main", rg, &[mini.as_str()], &format!("mini{n}"));
        ser("mini.slang", "raygen_main", rg, &[mini.as_str(), "ROOT_PUSH=1"], &format!("mini{n}_push"));
    }

    // Kernel microbenchmarks (kernels binary). Heap pipelines throughout.
    let k = |file: &str, entry: &str, stage: &str, defines: &[&str], extra: &[&str], name: &str| {
        let dst = out.join(format!("{name}.spv"));
        let mut cmd = base(file, entry, stage, &dst);
        cmd.args(["-capability", "spvDescriptorHeapEXT"]);
        for d in defines {
            cmd.arg(format!("-D{d}"));
        }
        cmd.args(extra);
        run(&mut cmd);
        validate(&sdk_bin, &dst);
    };
    for (t, name) in [(0, "vbuf_image"), (1, "vbuf_linear"), (2, "vbuf_tiled")] {
        k("vbuf.slang", "raster", "compute", &[&format!("TARGET={t}")], &[], name);
    }
    for (p, suffix) in [(0, "ubo"), (1, "ptr")] {
        let d = format!("ROOT_PTR={p}");
        k("rootbench.slang", "params", "compute", &[&d], &[], &format!("root_params_{suffix}"));
        k("rootbench.slang", "pointers", "compute", &[&d], &[], &format!("root_pointers_{suffix}"));
        k("rootbench.slang", "mesh_main", "mesh", &[&d], &[], &format!("root_mesh_{suffix}"));
        k("rootbench.slang", "pixel_main", "pixel", &[&d], &[], &format!("root_pixel_{suffix}"));
    }
    k("coopvec.slang", "main", "compute", &["FORM=0"], &["-warnings-as-errors", "all", "-capability", "spvCooperativeVectorNV"], "coopvec_asm");
    k("coopvec.slang", "main", "compute", &["FORM=1"], &["-capability", "spvCooperativeVectorNV"], "coopvec_bab");
    for e in ["transmittance", "multiscatter", "skyview"] {
        k("sky.slang", e, "compute", &[], &[], &format!("sky_{e}"));
    }
    for e in ["transmittance", "multiscatter", "skyview"] {
        k("sky.slang", e, "compute", &[], &["-g2"], &format!("sky_g_{e}"));
    }
    // Crash-repro kernel: embedded -g2 debug info, and the stripped form with a sidecar.
    k("crash.slang", "main", "compute", &[], &["-g2"], "crash_full");
    let sidecar = out.join("crash.dbg.spv");
    k("crash.slang", "main", "compute", &[], &["-g2", "-separate-debug-info", "-separate-debug-info-output", sidecar.to_str().unwrap()], "crash_stripped");
    println!("cargo:rustc-env=CRASH_SIDECAR={}", sidecar.display());
    println!("cargo:rustc-env=CRASH_SOURCE={}", shaders.join("crash.slang").display());
}
