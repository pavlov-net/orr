use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use orr_shaders::build::{Build, DebugInfo, Error, LinkPoint, Lint, Output};

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
        .metal_check(false)
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
fn metal_check_compares_layouts_and_finds_invalid_msl() {
    // Gap-free under C layout, but Metal puts `v` at its native alignment, 16.
    let src = "struct P { float3 a; uint b; float3x4 m; uint16_t h; uint16_t h2; float2 c; }\n\
        struct Root { uint64_t addr; float4 v; P* p; uint2 u2; uint _pad0; uint _pad1; }\n\
        [[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
        [shader(\"compute\")] [numthreads(64, 1, 1)] void main(uint3 id: SV_DispatchThreadID) {\n\
        P x = root.p[id.x]; x.a += root.v.xyz; x.b += uint(root.addr) + root.u2.x; x.h += 1; root.p[id.x + 1] = x; }\n";
    let dir = fixture("metal-layout", &[("src/k.slang", src)]);
    assert_eq!(lints(build(&dir).compile()), [Lint::RootLayout; 2]);
    match build(&dir)
        .allow(Lint::RootLayout)
        .metal_check(true)
        .compile()
    {
        Err(Error::Lints(l)) => {
            assert!(l.iter().all(|l| l.lint == Lint::Metal), "{l:?}");
            assert!(
                l[0].message
                    .contains("struct Root field v is at offset 16 on Metal, 8 in SPIR-V"),
                "{l:?}"
            );
        }
        other => panic!("expected a Metal layout failure, got {other:?}"),
    }

    // A 64-bit max outside orr.atomic lowers to a fetch atomic Metal does not have.
    let src = "import orr.atomic;\n\
        struct Root { uint64_t* v; uint n; uint pad; }\n[[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
        [shader(\"compute\")] [numthreads(32, 1, 1)] void main(uint3 id: SV_DispatchThreadID) {\n\
        atomic_max_u64(root.v + id.x, id.x); InterlockedMax(root.v[id.x + 1], 1ull); }\n";
    let dir = fixture("metal-atomic", &[("src/k.slang", src)]);
    let found = build(&dir).metal_check(true).compile();
    match found {
        Err(Error::Lints(l)) => {
            assert_eq!(l.len(), 1, "{l:?}");
            assert!(l[0].message.contains("64-bit fetch atomic"), "{l:?}");
        }
        other => panic!("expected a Metal failure, got {other:?}"),
    }
}

/// Instructions of a SPIR-V module as (opcode, operands).
fn instructions(words: &[u32]) -> Vec<(u32, &[u32])> {
    let mut out = Vec::new();
    let mut at = 5;
    while let Some(&first) = words.get(at) {
        let count = (first >> 16) as usize;
        out.push((first & 0xffff, &words[at + 1..at + count]));
        at += count;
    }
    out
}

fn read_words(path: &Path) -> Vec<u32> {
    fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

#[test]
#[allow(clippy::print_stderr)]
fn omm_variant_declares_the_mode_and_forces_two_state() {
    const OP_CAPABILITY: u32 = 17;
    const OP_SPEC_CONSTANT: u32 = 50;
    const OP_EXECUTION_MODE_ID: u32 = 331;
    const OMM: u32 = 5381;
    const OMM_MODE: u32 = 6032;
    let src = "#include \"orr/root.h\"\nimport orr.abi;\nimport orr.trace;\n\
        struct Root { uint* out; Tlas scene; }\nORR_ROOT(Root);\n\
        void body(uint2 i) {\n\
        RayDesc r; r.Origin = float3(i, 0); r.Direction = float3(0, 0, 1); r.TMin = 0; r.TMax = 1;\n\
        Hit h = trace_closest(root.scene, r, MASK_OPAQUE, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, AcceptAll());\n\
        root.out[i.x] = h.hit ? h.instance_index : 0; }\n\
        [shader(\"raygeneration\")] void main_rays() { body(DispatchRaysIndex().xy); }\n\
        [shader(\"compute\")] [numthreads(8, 4, 1)] void main_compute(uint3 id: SV_DispatchThreadID) { body(id.xy); }\n";
    let dir = fixture("omm", &[("src/k.slang", src)]);
    build(&dir).compile().unwrap_or_else(|e| panic!("{e}"));

    let base = read_words(&dir.join("out/k.spv"));
    let variant = read_words(&dir.join("out/k.omm.spv"));
    let capabilities = |words: &[u32]| -> Vec<u32> {
        instructions(words)
            .into_iter()
            .filter(|(op, _)| *op == OP_CAPABILITY)
            .map(|(_, a)| a[0])
            .collect()
    };
    // The two-state constant: 0 in the module, the ray flag in the variant.
    let spec_values = |words: &[u32]| -> Vec<u32> {
        instructions(words)
            .into_iter()
            .filter(|(op, _)| *op == OP_SPEC_CONSTANT)
            .map(|(_, a)| a[2])
            .collect()
    };
    assert!(!capabilities(&base).contains(&OMM) && !capabilities(&base).contains(&OMM_MODE));
    assert!(capabilities(&variant).contains(&OMM) && capabilities(&variant).contains(&OMM_MODE));
    assert_eq!(spec_values(&base), [0]);
    assert_eq!(spec_values(&variant), [0x400]);
    let modes = instructions(&variant)
        .into_iter()
        .filter(|(op, a)| *op == OP_EXECUTION_MODE_ID && a[1] == 6031)
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
        .args([
            "--target-env",
            "vulkan1.4",
            "--uniform-buffer-standard-layout",
        ])
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

/// The surface interface in the shape the build's method table forwards.
const SURFACE: &str = "module surface;\n\
    public struct SurfaceInput { public uint surface; public float x; }\n\
    public struct GSurface { public float4 albedo; }\n\
    public interface ISurface {\n\
        static GSurface surface(SurfaceInput i, out float depth_offset);\n\
        static float coverage(SurfaceInput i) { return 1; }\n\
    }\n\
    public struct StandardSurface : ISurface {\n\
        public static GSurface surface(SurfaceInput i, out float d) { d = 0; return { float4(0) }; }\n\
    }\n";

/// A consumer surface returning `value`.
fn consumer_surface(ty: &str, value: &str) -> String {
    format!(
        "import surface;\nstruct {ty} : ISurface {{ static GSurface surface(SurfaceInput i, out float d) {{ d = 0; return {{ float4({value}) }}; }} }}\n"
    )
}

const MEDIA: &str = "module media;\n\
    public interface IMedium { static float density(uint64_t data); }\n\
    public struct NoMedium : IMedium { public static float density(uint64_t data) { return 0; } }\n";

/// A renderer root with a surface kernel, a medium kernel and one that links nothing, built
/// once; returns the fixture and the renderer's output.
fn link_fixture(name: &str) -> (PathBuf, Output) {
    let surface_kernel = format!(
        "import surface;\nextern struct Surface : ISurface = StandardSurface;\n{}",
        kernel(
            "SurfaceInput i = { root.count, 1.0 }; float d; root.out[id.x] = Surface.surface(i, d).albedo * Surface.coverage(i) + d;"
        )
    );
    let medium_kernel = format!(
        "import media;\nextern struct Medium : IMedium = NoMedium;\n{}",
        kernel("root.out[id.x] = Medium.density(0);")
    );
    let dir = fixture(
        name,
        &[
            ("src/surface.slang", SURFACE),
            ("src/media.slang", MEDIA),
            ("src/resolve.slang", &surface_kernel),
            ("src/fog.slang", &medium_kernel),
            ("src/plain.slang", &kernel("root.out[id.x] = 1;")),
            (
                "consumer/ground.slang",
                &consumer_surface("Ground", "i.x * 2"),
            ),
            ("consumer/grass.slang", &consumer_surface("Grass", "3")),
            (
                "consumer/air.slang",
                "import media;\nstruct Air : IMedium { static float density(uint64_t data) { return 0.5; } }\n",
            ),
        ],
    );
    let renderer = build(&dir).compile().unwrap_or_else(|e| panic!("{e}"));
    (dir, renderer)
}

fn consumer(dir: &Path, renderer: &Output) -> Build {
    Build::new()
        .unwrap_or_else(|e| panic!("{e}"))
        .roots(&renderer.shader_dirs)
        .link_points(&renderer.link_points.display().to_string())
        .metal_check(false)
        .out_dir(dir.join("consumer-out"))
}

#[test]
fn editing_a_linked_module_recompiles_only_the_kernels_it_links_into() {
    let (dir, renderer) = link_fixture("link-incremental");
    let ground = dir.join("consumer/ground.slang");
    let compiled = |extra: bool| {
        let b = consumer(&dir, &renderer)
            .link(LinkPoint::Surface, "ground", &ground, "Ground")
            .link(
                LinkPoint::Medium,
                "air",
                dir.join("consumer/air.slang"),
                "Air",
            );
        let b = if extra {
            b.link(
                LinkPoint::Surface,
                "grass",
                dir.join("consumer/grass.slang"),
                "Grass",
            )
        } else {
            b
        };
        b.compile().unwrap_or_else(|e| panic!("{e}")).compiled
    };
    assert_eq!(compiled(false), ["linked/fog", "linked/resolve"]);
    assert!(compiled(false).is_empty());
    fs::write(&ground, consumer_surface("Ground", "i.x * 4")).unwrap();
    assert_eq!(compiled(false), ["linked/resolve"]);
    assert_eq!(compiled(true), ["linked/resolve"]);
}

#[test]
fn link_calls_are_checked() {
    let (dir, renderer) = link_fixture("link-errors");
    let air = dir.join("consumer/air.slang");
    let ground = dir.join("consumer/ground.slang");
    let message = |b: Build| match b.compile() {
        Err(Error::Link { message }) => message,
        other => panic!("expected a link error, got {other:?}"),
    };
    let twice = consumer(&dir, &renderer)
        .link(LinkPoint::Medium, "air", &air, "Air")
        .link(LinkPoint::Medium, "fog", &air, "Air");
    assert!(message(twice).contains("at most 1"));
    let undeclared = consumer(&dir, &renderer).link(LinkPoint::Environment, "sky", &air, "Air");
    assert!(message(undeclared).contains("no kernel"));
    let name = consumer(&dir, &renderer).link(LinkPoint::Surface, "Ground", &ground, "Ground");
    assert!(message(name).contains("lower-case"));
}

/// Every `.spv` under `dir` by its path below it.
fn spirv_files(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for path in fs::read_dir(&d).unwrap().map(|e| e.unwrap().path()) {
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "spv") {
                out.push((
                    path.strip_prefix(dir).unwrap().to_path_buf(),
                    fs::read(&path).unwrap(),
                ));
            }
        }
    }
    out.sort();
    out
}

/// The `precompiled` lists of a build manifest's modules.
fn precompiled(out: &Output) -> Vec<(String, Vec<String>)> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&out.manifest).unwrap()).unwrap();
    manifest["modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let names = m["precompiled"].as_array().unwrap();
            (
                m["name"].as_str().unwrap().to_owned(),
                names
                    .iter()
                    .map(|n| n.as_str().unwrap().to_owned())
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn precompiled_modules_give_the_same_spirv_and_rust() {
    let shared = |scale: &str| {
        format!(
            "import orr.math;\npublic float3 lit(float3 n) {{ return safe_normalize(n) * {scale}; }}\n"
        )
    };
    let a = format!(
        "import orr.color;\nimport shared;\n{}",
        kernel("root.out[id.x] = float4(srgb_encode(lit(float3(root.count))), 1);")
    );
    let b = format!(
        "import orr.coherence;\nimport shared;\n{}",
        kernel(
            "root.out[id.x] = float4(lit(float3(1)), float(coherence_key(true, 1, root.count)));"
        )
    );
    let dir = fixture(
        "precompiled",
        &[
            ("src/a.slang", &a),
            ("src/b.slang", &b),
            ("src/shared.slang", &shared("2")),
        ],
    );
    let both = || {
        let [on, off] = [true, false].map(|on| {
            let out = dir.join(if on { "on" } else { "off" });
            let built = build(&dir)
                .debug_info(DebugInfo::Off)
                .asserts(true)
                .precompile(on)
                .out_dir(&out)
                .compile()
                .unwrap_or_else(|e| panic!("{e}"));
            let rust = fs::read_to_string(out.join("orr_shaders.rs"))
                .unwrap()
                .replace(
                    format!("{:?}", out.display().to_string()).trim_matches('"'),
                    "OUT",
                )
                .replace(&out.display().to_string(), "OUT");
            (built, spirv_files(&out), rust)
        });
        assert_eq!(on.1, off.1);
        assert_eq!(on.2, off.2);
        (on.0, off.0)
    };
    let (on, off) = both();
    assert_eq!(
        precompiled(&on),
        [
            (
                "a".to_owned(),
                vec![
                    "orr.color".to_owned(),
                    "orr.math".to_owned(),
                    "shared".to_owned()
                ]
            ),
            (
                "b".to_owned(),
                vec![
                    "orr.coherence".to_owned(),
                    "orr.math".to_owned(),
                    "shared".to_owned()
                ]
            ),
        ]
    );
    assert!(precompiled(&off).iter().all(|(_, p)| p.is_empty()));

    // An edited library is compiled again before its importers use it.
    fs::write(dir.join("src/shared.slang"), shared("3")).unwrap();
    let (on, _) = both();
    assert_eq!(on.compiled, ["a", "b"]);
}

#[test]
fn linked_kernels_load_precompiled_consumer_modules() {
    let (dir, renderer) = link_fixture("link-precompiled");
    let [on, off] = [true, false].map(|on| {
        let out = dir.join(if on { "linked-on" } else { "linked-off" });
        let built = consumer(&dir, &renderer)
            .link(
                LinkPoint::Surface,
                "ground",
                dir.join("consumer/ground.slang"),
                "Ground",
            )
            .link(
                LinkPoint::Surface,
                "grass",
                dir.join("consumer/grass.slang"),
                "Grass",
            )
            .link(
                LinkPoint::Medium,
                "air",
                dir.join("consumer/air.slang"),
                "Air",
            )
            .debug_info(DebugInfo::Off)
            .precompile(on)
            .out_dir(&out)
            .compile()
            .unwrap_or_else(|e| panic!("{e}"));
        (built, spirv_files(&out))
    });
    assert_eq!(on.1, off.1);
    let used = precompiled(&on.0);
    let resolve = &used.iter().find(|(n, _)| n == "resolve").unwrap().1;
    assert_eq!(resolve, &["grass", "ground", "surface"]);
}

#[test]
fn cooperative_vectors_stay_out_of_records() {
    // SPIR-V allows cooperative vectors only in function and private storage, so a root or
    // record holding one, directly or through a pointer, fails validation; weights go
    // behind an address instead.
    let cases = [
        (
            "struct W { CoopVec<half, 16> v; float s; }\nstruct Root { W* w; float* out; }\n",
            "root.out[id.x] = float(root.w[id.x].v[3]) * root.w[id.x].s;",
        ),
        (
            "struct Root { CoopVec<float, 4>* p; float* out; }\n",
            "root.out[id.x] = root.p[id.x][1];",
        ),
    ];
    for (i, (types, body)) in cases.into_iter().enumerate() {
        let src = format!(
            "{types}[[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
             [shader(\"compute\")] [numthreads(32, 1, 1)] void main(uint3 id: SV_DispatchThreadID) {{ {body} }}\n"
        );
        let dir = fixture(&format!("coopvec-{i}"), &[("src/k.slang", &src)]);
        match build(&dir).allow(Lint::CoopVec).compile() {
            Err(Error::Compile { diagnostics, .. }) => assert!(
                diagnostics.contains("Validation of generated SPIR-V failed"),
                "{diagnostics}"
            ),
            other => panic!("a cooperative vector in memory compiled: {other:?}"),
        }
    }
}

const OP_STRING: u32 = 7;
const OP_EXT_INST_IMPORT: u32 = 11;
const OP_EXT_INST: u32 = 12;

/// The SPIR-V literal string in `words`, up to its nul.
fn literal(words: &[u32]) -> String {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// The `NonSemantic.Shader.DebugInfo.100` instruction numbers a module uses.
fn debug_instructions(words: &[u32]) -> Vec<u32> {
    let ops = instructions(words);
    let Some(set) = ops.iter().find_map(|(op, a)| {
        (*op == OP_EXT_INST_IMPORT && literal(&a[1..]) == "NonSemantic.Shader.DebugInfo.100")
            .then_some(a[0])
    }) else {
        return Vec::new();
    };
    ops.iter()
        .filter(|(op, a)| *op == OP_EXT_INST && a[2] == set)
        .map(|(_, a)| a[3])
        .collect()
}

#[test]
fn separate_debug_info_strips_the_module_into_a_sidecar() {
    const DEBUG_SOURCE: u32 = 35;
    const DEBUG_LINE: u32 = 103;
    const DEBUG_BUILD_IDENTIFIER: u32 = 105;
    // A precompiled import, and an array in a pointee, which the model knows by its SPIR-V
    // wrapper struct's name.
    let src = "#include \"orr/debug.h\"\n\
        struct Rec { uint3 a[2]; uint b; }\n\
        struct Root { Rec* recs; uint count; uint pad; }\n\
        [[vk::binding(0, 0)]] ConstantBuffer<Root> root;\n\
        [shader(\"compute\")] [numthreads(32, 1, 1)]\n\
        void main(uint3 id: SV_DispatchThreadID) {\n\
            orr_assert(id.x < root.count, \"past the end\");\n\
            root.recs[id.x].b = root.recs[id.x].a[1].y;\n\
        }\n";
    let dir = fixture("separate_debug", &[("src/k.slang", src)]);
    let out = build(&dir)
        .debug_info(DebugInfo::Separate)
        .asserts(true)
        .compile()
        .unwrap_or_else(|e| panic!("{e}"));
    let stripped = read_words(&dir.join("out/k.spv"));
    let sidecar = dir.join("out/k.dbg.spv");
    let kept = debug_instructions(&stripped);
    let debug = debug_instructions(&read_words(&sidecar));
    assert!(kept.contains(&DEBUG_BUILD_IDENTIFIER), "{kept:?}");
    assert!(!kept.contains(&DEBUG_SOURCE) && !kept.contains(&DEBUG_LINE));
    assert!(debug.contains(&DEBUG_SOURCE) && debug.contains(&DEBUG_LINE));
    // The assert's printf format stays in the stripped module.
    let strings: Vec<String> = instructions(&stripped)
        .into_iter()
        .filter(|(op, _)| *op == OP_STRING)
        .map(|(_, a)| literal(&a[1..]))
        .collect();
    assert!(
        strings.iter().any(|s| s.ends_with(": past the end")),
        "{strings:?}"
    );

    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&out.manifest).unwrap()).unwrap();
    let listed = manifest["modules"][0]["debug_info"].as_str().unwrap();
    assert_eq!(
        Path::new(listed).canonicalize().unwrap(),
        sidecar.canonicalize().unwrap()
    );
    let rust = fs::read_to_string(dir.join("out/orr_shaders.rs")).unwrap();
    assert!(rust.contains(&format!("debug_info: {listed:?}")), "{rust}");

    build(&dir)
        .debug_info(DebugInfo::Embedded)
        .compile()
        .unwrap_or_else(|e| panic!("{e}"));
    let embedded = debug_instructions(&read_words(&dir.join("out/k.spv")));
    assert!(embedded.contains(&DEBUG_LINE));
    let rust = fs::read_to_string(dir.join("out/orr_shaders.rs")).unwrap();
    assert!(rust.contains("debug_info: \"\""), "{rust}");
}

#[test]
fn orr_assert_aborts_with_its_location_only_when_asserts_are_linked() {
    const OP_CONSTANT: u32 = 43;
    const OP_CONSTANT_COMPOSITE: u32 = 44;
    const OP_ABORT: u32 = 5121;
    let src = format!(
        "#include \"orr/debug.h\"\n{}",
        kernel(
            "orr_assert(id.x < root.count, \"thread %u past %u\", id.x, root.count);\n root.out[id.x] = float4(id.x);"
        )
    );
    let dir = fixture("asserts", &[("src/k.slang", &src)]);
    let imports = |words: &[u32]| -> Vec<String> {
        instructions(words)
            .into_iter()
            .filter(|(op, _)| *op == OP_EXT_INST_IMPORT)
            .map(|(_, a)| literal(&a[1..]))
            .collect()
    };

    build(&dir)
        .asserts(false)
        .compile()
        .unwrap_or_else(|e| panic!("{e}"));
    let off = read_words(&dir.join("out/k.spv"));
    assert!(!instructions(&off).iter().any(|(op, _)| *op == OP_ABORT));
    assert!(!imports(&off).iter().any(|i| i == "NonSemantic.DebugPrintf"));

    build(&dir)
        .asserts(true)
        .compile()
        .unwrap_or_else(|e| panic!("{e}"));
    let on = read_words(&dir.join("out/k.spv"));
    assert!(imports(&on).iter().any(|i| i == "NonSemantic.DebugPrintf"));
    // The abort's message: a struct holding the text as a uint array.
    let ops = instructions(&on);
    let value = |id: u32| {
        ops.iter()
            .find(|(op, a)| matches!(*op, OP_CONSTANT | OP_CONSTANT_COMPOSITE) && a[1] == id)
            .map(|(_, a)| &a[2..])
            .unwrap()
    };
    let (_, message) = ops.iter().find(|(op, _)| *op == OP_ABORT).unwrap();
    let fields = value(message[1]);
    let text: Vec<u32> = value(fields[0]).iter().map(|&id| value(id)[0]).collect();
    let text = literal(&text);
    let kernel_path = dir.join("src/k.slang").canonicalize().unwrap();
    let file = text
        .strip_suffix(":5: thread %u past %u")
        .unwrap_or_else(|| panic!("{text}"));
    assert_eq!(Path::new(file).canonicalize().unwrap(), kernel_path);
}
