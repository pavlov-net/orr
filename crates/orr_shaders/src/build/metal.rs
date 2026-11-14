//! The MSL emission check: a module's compute and mesh-pipeline entries compiled for Metal,
//! their root and pointee layouts compared with the SPIR-V, and the MSL searched for
//! constructs Metal rejects. Builds for Apple targets also compile the MSL into Metal
//! libraries ([`compile_library`]), which the modules embed.

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use super::model::{Model, Ty, text};
use super::sys;

/// Options of the Metal session; include roots follow. `ORR_METAL` (tested with `#if defined`) lets a kernel work around
/// what a target's Slang backend gets wrong where `__target_switch` cannot reach, as in entry
/// signatures.
pub(super) const OPTIONS: &[&str] = &[
    "-std",
    "2026",
    "-target",
    "metal",
    "-matrix-layout-row-major",
    "-DORR_METAL=1",
];

/// Every difference between the Metal layout of the root and its pointees and the SPIR-V
/// layout in `model`. `layout` must be the live Metal program layout.
pub(super) fn layout_issues(
    api: &sys::SlangLib,
    layout: *mut sys::SlangReflection,
    model: &Model,
) -> Vec<String> {
    let Some(root) = &model.root else {
        return Vec::new();
    };
    let mut walk = Walk {
        api,
        model,
        seen: HashSet::new(),
        issues: Vec::new(),
    };
    // SAFETY: `layout` is a live program layout; every pointer below comes from it.
    unsafe {
        for i in 0..api.spReflection_GetParameterCount(layout) {
            let vl = api.spReflection_GetParameterByIndex(layout, i);
            let var = api.spReflectionVariableLayout_GetVariable(vl);
            if text(api.spReflectionVariable_GetName(var)) != "root" {
                continue;
            }
            let buffer = api.spReflectionVariableLayout_GetTypeLayout(vl);
            let element = api.spReflectionTypeLayout_GetElementTypeLayout(buffer);
            if element.is_null() {
                walk.issues
                    .push("Metal reflection has no layout for the root".to_owned());
            } else {
                walk.structure(root, element);
            }
        }
    }
    walk.issues
}

struct Walk<'a> {
    api: &'a sys::SlangLib,
    model: &'a Model,
    seen: HashSet<String>,
    issues: Vec<String>,
}

const UNIFORM: sys::SlangParameterCategory =
    sys::SlangParameterCategory::SLANG_PARAMETER_CATEGORY_UNIFORM;

impl Walk<'_> {
    unsafe fn structure(&mut self, name: &str, tl: *mut sys::SlangReflectionTypeLayout) {
        if !self.seen.insert(name.to_owned()) {
            return;
        }
        let def = &self.model.structs[name];
        // SAFETY: `tl` comes from the live Metal layout.
        unsafe {
            let api = self.api;
            for i in 0..api.spReflectionTypeLayout_GetFieldCount(tl) {
                let fvl = api.spReflectionTypeLayout_GetFieldByIndex(tl, i);
                let var = api.spReflectionVariableLayout_GetVariable(fvl);
                let field_name = text(api.spReflectionVariable_GetName(var));
                let Some(field) = def.fields.iter().find(|f| f.name == field_name) else {
                    self.issues.push(format!(
                        "struct {name} field {field_name} is in the Metal layout only"
                    ));
                    continue;
                };
                let metal = api.spReflectionVariableLayout_GetOffset(fvl, UNIFORM);
                if metal != field.offset as usize {
                    self.issues.push(format!(
                        "struct {name} field {field_name} is at offset {metal} on Metal, {} in SPIR-V",
                        field.offset
                    ));
                }
                let ftl = api.spReflectionVariableLayout_GetTypeLayout(fvl);
                self.ty(&field.ty, ftl, name, &field_name);
            }
        }
    }

    unsafe fn ty(
        &mut self,
        ty: &Ty,
        tl: *mut sys::SlangReflectionTypeLayout,
        owner: &str,
        field: &str,
    ) {
        // SAFETY: `tl` comes from the live Metal layout.
        unsafe {
            let api = self.api;
            match ty {
                Ty::Struct(name) => self.structure(name, tl),
                Ty::Pointer(pointee) => {
                    let element = api.spReflectionTypeLayout_GetElementTypeLayout(tl);
                    if !element.is_null() {
                        self.ty(pointee, element, owner, field);
                    }
                }
                Ty::Array {
                    element, stride, ..
                } => {
                    let metal = api.spReflectionTypeLayout_GetElementStride(tl, UNIFORM);
                    if metal != *stride as usize {
                        self.issues.push(format!(
                            "struct {owner} field {field} has array stride {metal} on Metal, {stride} in SPIR-V"
                        ));
                    }
                    let inner = api.spReflectionTypeLayout_GetElementTypeLayout(tl);
                    if !inner.is_null() {
                        self.ty(element, inner, owner, field);
                    }
                }
                Ty::Matrix { .. } => {
                    let mode = api.spReflectionTypeLayout_GetMatrixLayoutMode(tl);
                    if mode != sys::SlangMatrixLayoutMode::SLANG_MATRIX_LAYOUT_ROW_MAJOR {
                        self.issues.push(format!(
                            "struct {owner} field {field} is not a row-major matrix on Metal"
                        ));
                    }
                }
                Ty::Scalar(_) | Ty::Vector(..) | Ty::Handle(_) => {}
            }
        }
    }
}

/// Constructs in emitted MSL that Metal rejects.
pub(super) fn msl_issues(msl: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_struct: Option<&str> = None;
    let mut depth = 0i32;
    for (i, line) in msl.lines().enumerate() {
        let line_no = i + 1;
        let t = line.trim();
        if let Some(name) = in_struct {
            if t.contains("intersection_query") {
                out.push(format!(
                    "MSL line {line_no}: a ray query is a member of struct {name}"
                ));
            }
            depth += brace_delta(t);
            if depth <= 0 {
                in_struct = None;
            }
        } else if let Some(rest) = t.strip_prefix("struct ")
            && !t.ends_with(';')
        {
            in_struct = Some(rest.split_whitespace().next().unwrap_or(rest));
            depth = brace_delta(t);
            if t.contains('{') && depth <= 0 {
                in_struct = None;
            }
        }
        if t.contains("atomic_fetch_")
            && ["atomic_ulong", "atomic_long", "vec<ulong", "vec<long"]
                .iter()
                .any(|w| t.contains(w))
        {
            out.push(format!(
                "MSL line {line_no}: a 64-bit fetch atomic, which Metal lacks (use orr_shaders.atomic)"
            ));
        }
    }
    for (line_no, signature) in entry_signatures(msl) {
        for p in entry_parameters(&signature).unwrap_or_default() {
            let builtin = ["mesh_grid_properties", "metal::mesh<"]
                .iter()
                .any(|b| p.trim_start().starts_with(b));
            if !p.contains("[[") && !builtin {
                out.push(format!(
                    "MSL line {line_no}: entry parameter `{}` has no binding",
                    p.trim()
                ));
            }
        }
    }
    out
}

/// The `[shader(...)]` stage names of ray tracing entries.
const RAY_TRACING_STAGES: [&str; 6] = [
    "\"raygeneration\"",
    "\"closesthit\"",
    "\"anyhit\"",
    "\"intersection\"",
    "\"miss\"",
    "\"callable\"",
];

/// `source` with the `[shader(...)]` attributes of its ray tracing entries blanked out, so a
/// Metal session, which rejects those entries when it loads a module, sees them as plain
/// functions. Line and column numbers stay the same.
pub(super) fn without_ray_tracing(source: &str) -> String {
    let mut out = source.to_owned();
    let mut from = 0;
    while let Some(i) = out[from..].find("[shader(").map(|i| i + from) {
        let rest = &out[i + "[shader(".len()..];
        let trimmed = rest.trim_start();
        let after_stage = RAY_TRACING_STAGES
            .iter()
            .find_map(|stage| trimmed.strip_prefix(stage))
            .map(str::trim_start)
            .and_then(|r| r.strip_prefix(")]"));
        match after_stage {
            Some(after) => {
                let end = out.len() - after.len();
                out.replace_range(i..end, &" ".repeat(end - i));
                from = end;
            }
            None => from = i + 1,
        }
    }
    out
}

/// The names of the entry functions `msl` declares. Slang renames some (`main` becomes
/// `main_0`), so these are the names a Metal library knows them by.
pub(super) fn entry_names(msl: &str) -> Vec<String> {
    entry_signatures(msl)
        .into_iter()
        .filter_map(|(_, s)| Some(s[..s.find('(')?].split_whitespace().last()?.to_owned()))
        .collect()
}

/// The Metal form of an entry point's compiled name: an MSL function name, with `_` for `.`.
pub(super) fn symbol(spirv: &str) -> String {
    spirv.replace('.', "_")
}

/// The Metal function name of the entry `name` among `declared`.
pub(super) fn metal_name<'a>(name: &str, declared: &'a [String]) -> Option<&'a str> {
    let renamed = format!("{name}_0");
    declared
        .iter()
        .find(|d| **d == name)
        .or_else(|| declared.iter().find(|d| **d == renamed))
        .map(String::as_str)
}

/// Each entry function's signature, from its stage attribute on, with its 1-based line.
/// Attributes such as `[[early_fragment_tests]]` can put the stage on a line of its own,
/// before the signature.
fn entry_signatures(msl: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = msl.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !ENTRY_STAGES.iter().any(|s| line.starts_with(s)) {
            continue;
        }
        if line.contains('(') {
            out.push((i + 1, (*line).to_owned()));
        } else if let Some(next) = lines.get(i + 1) {
            out.push((i + 2, format!("{line} {next}")));
        }
    }
    out
}

/// `msl` with the cull flag of each mesh entry's primitives marked `[[primitive_culled]]`.
/// Slang drops `SV_CullPrimitive` in MSL, leaving a plain `bool` member that Metal takes as
/// per-primitive data, so no mesh shader culled a primitive. The `bool` member without an
/// attribute of a mesh entry's primitive struct is that flag: no other primitive output is a
/// `bool`.
pub(super) fn primitive_culled(msl: &str) -> String {
    let primitives: Vec<String> = entry_signatures(msl)
        .iter()
        .filter_map(|(_, s)| {
            let args = &s[s.find("metal::mesh<")? + "metal::mesh<".len()..];
            let p = args.split(',').nth(1)?.trim();
            (p != "void").then(|| p.to_owned())
        })
        .collect();
    let mut out = String::with_capacity(msl.len() + 64);
    let mut in_primitive = false;
    for line in msl.split_inclusive('\n') {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("struct ") {
            in_primitive = primitives.iter().any(|p| p == name.trim());
        } else if in_primitive && t.starts_with('}') {
            in_primitive = false;
        } else if in_primitive
            && let Some(field) = t.strip_prefix("bool ").and_then(|f| f.strip_suffix(';'))
            && !field.contains("[[")
        {
            let indent = &line[..line.len() - line.trim_start().len()];
            out.push_str(&format!("{indent}bool {field} [[primitive_culled]];\n"));
            continue;
        }
        out.push_str(line);
    }
    out
}

/// `msl` with its ray queries traversing three instancing levels: the Metal backend stands
/// in for a cluster BLAS with an instance structure over one BLAS per cluster, under the
/// TLAS. Slang has no way to ask for it. Instance IDs then take a depth: 0, the TLAS's.
/// Fails when a query or an instance ID read is left as Slang wrote it, which means Slang's
/// MSL changed shape and the rewrite needs updating.
pub(super) fn multi_level(msl: &str) -> Result<String, String> {
    let mut out = msl.replace(
        "raytracing::triangle_data, raytracing::instancing>",
        "raytracing::triangle_data, raytracing::instancing, raytracing::max_levels<3>>",
    );
    let mut reads = Vec::new();
    for access in ["candidate", "committed"] {
        for id in ["instance_id", "user_instance_id"] {
            let read = format!("get_{access}_{id}(");
            out = out.replace(&format!("{read})"), &format!("{read}0)"));
            reads.push(read);
        }
    }
    let queries = out.matches("intersection_query<").count();
    let rewritten = out.matches("raytracing::max_levels<3>>").count();
    if queries != rewritten {
        return Err(format!(
            "{} of {queries} intersection queries are not `intersection_query<raytracing::triangle_data, raytracing::instancing>`, which the multi-level rewrite expects",
            queries - rewritten.min(queries)
        ));
    }
    if let Some(read) = reads.iter().find(|r| {
        out.match_indices(r.as_str())
            .any(|(i, _)| !out[i + r.len()..].starts_with("0)"))
    }) {
        return Err(format!(
            "`{read}` takes arguments the multi-level rewrite does not expect"
        ));
    }
    Ok(out)
}

/// The SDK whose Metal compiler builds libraries for the target Cargo builds for: `None`
/// off Apple targets.
pub(super) fn target_sdk() -> Option<&'static str> {
    let var = |name: &str| std::env::var(name).unwrap_or_default();
    if var("CARGO_CFG_TARGET_VENDOR") != "apple" {
        return None;
    }
    let sim = var("CARGO_CFG_TARGET_ABI") == "sim";
    Some(match var("CARGO_CFG_TARGET_OS").as_str() {
        "ios" if var("CARGO_CFG_TARGET_ABI") == "macabi" => "macosx",
        "ios" if sim => "iphonesimulator",
        "ios" => "iphoneos",
        "tvos" if sim => "appletvsimulator",
        "tvos" => "appletvos",
        "visionos" if sim => "xrsimulator",
        "visionos" => "xros",
        "watchos" if sim => "watchsimulator",
        "watchos" => "watchos",
        _ => "macosx",
    })
}

/// Compiles the MSL at `metal` into the library `out` with the Metal compiler of Xcode's
/// `sdk` ([`target_sdk`]). The error holds the compiler's error lines. Math is `safe`, as
/// SPIR-V's: Metal's default `fast` assumes no NaNs or infinities and reassociates. Nothing
/// fuses into FMAs either, since fusing differs between call sites and kernels rely on
/// evaluating one expression to the same bits everywhere (tessellated edges two patches
/// share).
pub(super) fn compile_library(metal: &Path, out: &Path, sdk: &str) -> Result<(), String> {
    let output = Command::new("xcrun")
        .args([
            "-sdk",
            sdk,
            "metal",
            "-std=metal4.0",
            "-fmetal-math-mode=safe",
            "-ffp-contract=off",
            "-o",
        ])
        .arg(out)
        .arg(metal)
        .output()
        .map_err(|e| {
            format!("cannot run `xcrun metal` ({e}): install Xcode and its Metal toolchain")
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let errors: Vec<&str> = stderr.lines().filter(|l| l.contains("error:")).collect();
    let errors = if errors.is_empty() {
        stderr.trim().to_owned()
    } else {
        errors.join("\n")
    };
    Err(format!(
        "Apple's Metal compiler rejected {}:\n{errors}",
        metal.display()
    ))
}

const ENTRY_STAGES: &[&str] = &["[[kernel]]", "[[mesh]]", "[[object]]", "[[fragment]]"];

fn brace_delta(line: &str) -> i32 {
    line.chars()
        .map(|c| match c {
            '{' => 1,
            '}' => -1,
            _ => 0,
        })
        .sum()
}

/// The parameters of an entry function's signature line.
fn entry_parameters(line: &str) -> Option<Vec<&str>> {
    if !ENTRY_STAGES.iter().any(|s| line.starts_with(s)) {
        return None;
    }
    let open = line.find('(')?;
    let mut depth = 0;
    let mut start = open + 1;
    let mut params = Vec::new();
    for (i, c) in line[open..].char_indices().map(|(i, c)| (i + open, c)) {
        match c {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => {
                depth -= 1;
                if depth == 0 {
                    let p = &line[start..i];
                    if !p.trim().is_empty() {
                        params.push(p);
                    }
                    return Some(params);
                }
            }
            ',' if depth == 1 => {
                params.push(&line[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    Some(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_what_metal_rejects() {
        let msl = "\
struct KernelContext_0
{
    metal::raytracing::acceleration_structure<metal::raytracing::instancing> tlas_0;
};
struct Tracer_0
{
    raytracing::intersection_query<raytracing::triangle_data, raytracing::instancing> q_0;
    float t_0;
};
struct Fwd_0;
    ulong _S6 = atomic_fetch_max_explicit(((atomic_ulong device*)(p)), v_0, memory_order_relaxed);
    uint _S10 = atomic_fetch_add_explicit(((atomic_uint device*)(c + ulong(i))), s_0, memory_order_relaxed);
    atomic_max_explicit((device atomic_ulong*)(p), v, memory_order_relaxed);
[[kernel]] void main_0(uint3 tid_0 [[thread_position_in_grid]], Root_0 constant* root_1 [[buffer(0)]], metal::raytracing::acceleration_structure<metal::raytracing::instancing> device* root_scene_tlas_1)
[[kernel]] void ok_0(uint3 tid_0 [[thread_position_in_grid]], Root_0 constant* root_1 [[buffer(0)]])
[[mesh]] void mesh_0(metal::mesh<Vertex_0, void, int(96), int(32), metal::topology::triangle> _slang_mesh, uint t_0 [[thread_index_in_threadgroup]])
";
        let issues = msl_issues(msl);
        assert_eq!(issues.len(), 3, "{issues:#?}");
        assert!(issues[0].contains("line 7") && issues[0].contains("Tracer_0"));
        assert!(issues[1].contains("line 11") && issues[1].contains("64-bit"));
        assert!(issues[2].contains("line 14") && issues[2].contains("root_scene_tlas_1"));
    }

    #[test]
    fn mesh_primitives_cull_by_their_flag() {
        let msl = "\
struct Primitive_0
{
    uint triangle_2 [[primitive_id]];
    bool cull_0;
};
struct Other_0
{
    bool kept_0;
};
[[mesh]] void cluster_mesh(uint3 g [[threadgroup_position_in_grid]], metal::mesh<Vertex_0, Primitive_0, int(256), int(128), metal::topology::triangle> _slang_mesh)
";
        let out = primitive_culled(msl);
        assert!(
            out.contains("    bool cull_0 [[primitive_culled]];\n"),
            "{out}"
        );
        assert!(out.contains("    bool kept_0;\n"), "{out}");
        assert!(out.contains("uint triangle_2 [[primitive_id]];"));
    }

    #[test]
    fn ray_tracing_attributes_blank_out_in_place() {
        let src = "[shader(\"compute\")]
void a() {}
[shader( \"raygeneration\" )]
void b() {}
[shader(\"closesthit\")]
void c() {}
";
        let out = without_ray_tracing(src);
        assert_eq!(out.len(), src.len());
        assert!(out.contains("[shader(\"compute\")]") && !out.contains("raygeneration"));
        assert!(!out.contains("closesthit"));
        assert_eq!(out.lines().nth(3), Some("void b() {}"));
    }
}
