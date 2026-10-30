use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use orr_shaders::build::{Build, Error, Lint, Output};

/// Writes `files` under a fresh directory named for the test and returns it.
fn fixture(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("build")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    for (path, src) in files {
        let path = dir.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, src).unwrap();
    }
    dir
}

fn build(dir: &Path) -> Build {
    Build::new()
        .unwrap_or_else(|e| panic!("{e}"))
        .kernels(dir.join("src"))
        .out_dir(dir.join("out"))
}

fn lints(result: Result<Output, Error>) -> Vec<Lint> {
    match result {
        Err(Error::Lints(l)) => l.iter().map(|l| l.lint).collect(),
        Err(e) => panic!("expected lint failures, got: {e}"),
        Ok(_) => panic!("expected lint failures, the build passed"),
    }
}

const ROOT: &str = "struct Root { float4* out; uint count; uint pad; }\n[[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n";

fn kernel(body: &str) -> String {
    format!(
        "{ROOT}[shader(\"compute\")] [numthreads(32, 1, 1)]\nvoid main(uint3 id: SV_DispatchThreadID) {{ {body} }}\n"
    )
}

#[test]
fn clean_kernel_passes_every_lint() {
    let src = "[vk::constant_id(0)] const uint STEPS = 4;\n\
        struct Rec { float3 p; uint id; uint64_t key; }\n\
        struct Root { Rec* recs; uint64_t* vbuf; float4x4 m; uint4 v; uint count; uint pad0; uint pad1; uint pad2; }\n\
        [[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
        [shader(\"compute\")] [numthreads(32, 1, 1)]\n\
        void main(uint3 id: SV_DispatchThreadID) {\n\
            Rec r = root.recs[id.x];\n\
            InterlockedMax(root.vbuf[id.x], r.key);\n\
            root.recs[id.x].p = mul(root.m, float4(r.p, float(root.v.x + STEPS))).xyz;\n\
        }\n";
    let dir = fixture("clean", &[("src/k.slang", src)]);
    let out = build(&dir).compile().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out.compiled, ["k"]);
    let rust = fs::read_to_string(dir.join("out/orr_shaders.rs")).unwrap();
    assert!(
        rust.contains("pub recs: ::orr_shaders::abi::GpuPtr<Rec>"),
        "{rust}"
    );
}

#[test]
fn every_lint_has_a_failing_fixture() {
    let rec_root = |rec: &str| {
        format!(
            "struct Rec {{ {rec} }}\nstruct Root {{ Rec* r; uint count; uint pad; }}\n[[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
             [shader(\"compute\")] [numthreads(32, 1, 1)] void main(uint3 id: SV_DispatchThreadID) {{ root.r[id.x] = root.r[id.x + 1]; }}\n"
        )
    };
    let cases: Vec<(Lint, String)> = vec![
        (
            Lint::SingleRoot,
            format!("RWStructuredBuffer<uint> extra;\n{}", kernel("extra[id.x] = root.count;")),
        ),
        (
            Lint::RootLayout,
            "struct Root { float* out; float3 v; uint n; }\n[[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
             [shader(\"compute\")] [numthreads(1, 1, 1)] void main() { root.out[0] = root.v.x + root.n; }\n"
                .into(),
        ),
        (Lint::ImplicitPadding, rec_root("uint a; uint64_t b;")),
        (Lint::Bool, rec_root("uint a; bool b;")),
        (
            Lint::SpecIds,
            format!("[SpecializationConstant] const uint N = 4;\n{}", kernel("root.out[id.x] = float4(N);")),
        ),
        (
            Lint::HeapTokens,
            kernel("root.out[id.x] = getDescriptorFromHandle(DescriptorHandle<Texture2D<float4>>(uint2(root.count, 0))).Load(int3(0));"),
        ),
        (
            Lint::RayQuery,
            "struct Root { uint64_t tlas; float4* out; }\n[[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
             [shader(\"compute\")] [numthreads(32, 1, 1)] void main(uint3 id: SV_DispatchThreadID) {\n\
             RayDesc r; r.Origin = float3(id); r.Direction = float3(0, 0, 1); r.TMin = 0; r.TMax = 1;\n\
             RayQuery<RAY_FLAG_FORCE_OPAQUE> q; q.TraceRayInline(RaytracingAccelerationStructure(root.tlas), 0, 0xff, r); q.Proceed();\n\
             root.out[id.x] = float4(q.CommittedRayT()); }\n"
                .into(),
        ),
        (
            Lint::Reorder,
            format!("{ROOT}[shader(\"raygeneration\")] void main() {{ ReorderThread(DispatchRaysIndex().x, 1); root.out[DispatchRaysIndex().x] = 1; }}\n"),
        ),
        (
            Lint::CoopVec,
            kernel("CoopVec<half, 4> v = CoopVec<half, 4>(half(root.count)); root.out[id.x] = float4(v[0]);"),
        ),
        (
            Lint::Atomic64,
            "struct Root { uint64_t* v; uint n; uint pad; }\n[[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
             [shader(\"compute\")] [numthreads(32, 1, 1)] void main(uint3 id: SV_DispatchThreadID) { InterlockedAdd(root.v[id.x], 1ull); }\n"
                .into(),
        ),
    ];
    for (lint, src) in cases {
        let dir = fixture(&format!("lint-{lint}"), &[("src/k.slang", &src)]);
        let found = lints(build(&dir).compile());
        assert!(found.contains(&lint), "{lint}: found {found:?}");
        // Allowing the lint lets the same source through.
        let mut b = build(&dir);
        for l in &found {
            b = b.allow(*l);
        }
        b.compile()
            .unwrap_or_else(|e| panic!("{lint} allowed: {e}"));
    }
}

#[test]
fn import_names_and_struct_ownership_span_roots() {
    let shared = "public struct Shared { public float4 v; }\n";
    let user = "import shared;\nstruct Root { Shared* s; float4* out; }\n[[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
         [shader(\"compute\")] [numthreads(32, 1, 1)] void main(uint3 id: SV_DispatchThreadID) { root.out[id.x] = root.s[id.x].v; }\n";
    let dir = fixture(
        "roots",
        &[("src/k.slang", user), ("other/shared.slang", shared)],
    );
    let found = lints(build(&dir).include(dir.join("other")).compile());
    assert_eq!(found, [Lint::Ownership]);
    build(&dir)
        .include(dir.join("other"))
        .extern_type("Shared", "crate::Shared")
        .compile()
        .unwrap_or_else(|e| panic!("{e}"));

    let dir = fixture(
        "duplicate",
        &[
            ("src/k.slang", &kernel("root.out[id.x] = 1;")),
            ("src/common.slang", "float f() { return 1; }\n"),
            ("other/common.slang", "float g() { return 2; }\n"),
        ],
    );
    let found = lints(build(&dir).include(dir.join("other")).compile());
    assert_eq!(found, [Lint::DuplicateImport]);
}

#[test]
fn layout_mismatch_names_module_struct_and_field() {
    let src = "struct Rec { uint a; column_major float4x3 m; }\nstruct Root { Rec* r; float4* out; }\n\
        [[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
        [shader(\"compute\")] [numthreads(32, 1, 1)] void main(uint3 id: SV_DispatchThreadID) { root.out[id.x] = float4(root.r[id.x].m[1], 0); }\n";
    let dir = fixture("mismatch", &[("src/deep/k.slang", src)]);
    match build(&dir).compile() {
        Err(Error::Layout {
            module,
            structure,
            field,
            message,
        }) => {
            assert_eq!(
                (module.as_str(), structure.as_str(), field.as_str()),
                ("deep/k", "Rec", "m")
            );
            assert!(message.contains("column-major"), "{message}");
        }
        other => panic!("expected a layout mismatch, got {other:?}"),
    }
}

#[test]
fn editing_one_input_recompiles_only_its_modules() {
    let a = format!(
        "import common;\n{}",
        kernel("root.out[id.x] = float4(scale());")
    );
    let b = kernel("root.out[id.x] = 2;");
    let dir = fixture(
        "incremental",
        &[
            ("src/a.slang", &a),
            ("src/b.slang", &b),
            ("src/common.slang", "float scale() { return 1; }\n"),
        ],
    );
    let compiled = |dir: &Path| {
        build(dir)
            .compile()
            .unwrap_or_else(|e| panic!("{e}"))
            .compiled
    };
    assert_eq!(compiled(&dir), ["a", "b"]);
    assert!(compiled(&dir).is_empty());
    fs::write(dir.join("src/b.slang"), kernel("root.out[id.x] = 3;")).unwrap();
    assert_eq!(compiled(&dir), ["b"]);
    fs::write(
        dir.join("src/common.slang"),
        "float scale() { return 4; }\n",
    )
    .unwrap();
    assert_eq!(compiled(&dir), ["a"]);
}

#[test]
fn spirv_validation_runs_in_process() {
    let src = kernel(
        "float a = root.count; float b = a; float r = spirv_asm { result:$$float = OpIAdd $a $b }; root.out[id.x] = r;",
    );
    let dir = fixture("validation", &[("src/k.slang", &src)]);
    match build(&dir).compile() {
        Err(Error::Compile { diagnostics, .. }) => {
            assert!(
                diagnostics.contains("Validation of generated SPIR-V failed"),
                "{diagnostics}"
            );
        }
        other => panic!("invalid SPIR-V passed: {other:?}"),
    }
}

#[test]
#[allow(clippy::print_stderr)]
fn omm_variant_declares_the_mode_and_validates() {
    let src = "struct Root { uint64_t tlas; float4* out; }\n[[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
        void body(uint2 i) {\n\
        RayDesc r; r.Origin = float3(i, 0); r.Direction = float3(0, 0, 1); r.TMin = 0; r.TMax = 1;\n\
        RayQuery<RAY_FLAG_NONE> q; q.TraceRayInline(RaytracingAccelerationStructure(root.tlas), 0, 0xff, r);\n\
        while (q.Proceed()) { q.CommitNonOpaqueTriangleHit(); }\n\
        root.out[i.x] = float4(q.CommittedRayT()); }\n\
        [shader(\"raygeneration\")] void main_rays() { body(DispatchRaysIndex().xy); }\n\
        [shader(\"compute\")] [numthreads(8, 4, 1)] void main_compute(uint3 id: SV_DispatchThreadID) { body(id.xy); }\n";
    let dir = fixture("omm", &[("src/k.slang", src)]);
    build(&dir)
        .allow(Lint::RayQuery)
        .allow(Lint::HeapTokens)
        .compile()
        .unwrap_or_else(|e| panic!("{e}"));
    let bytes = fs::read(dir.join("out/k.omm.spv")).unwrap();
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    // OpExecutionModeId (331, 4 words) with OpacityMicromapIdKHR (6031), once per entry.
    let modes = words
        .windows(3)
        .filter(|w| w[0] == (4 << 16 | 331) && w[2] == 6031)
        .count();
    assert_eq!(modes, 2);

    // Only SPIRV-Tools v2026.4 or newer knows the mode (Vulkan SDK 1.4.363): ORR_SPIRV_TOOLS
    // names the directory holding that spirv-val.
    let Some(tools) = std::env::var_os("ORR_SPIRV_TOOLS") else {
        eprintln!("ORR_SPIRV_TOOLS unset; spirv-val not run");
        return;
    };
    let val = Path::new(&tools).join(format!("spirv-val{}", std::env::consts::EXE_SUFFIX));
    let out = Command::new(&val)
        .args(["--target-env", "vulkan1.4"])
        .arg(dir.join("out/k.omm.spv"))
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", val.display()));
    assert!(
        out.status.success(),
        "spirv-val rejected the OMM variant:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
