//! The MSL emission check: a module's compute and mesh-pipeline entries compiled for Metal,
//! their root and pointee layouts compared with the SPIR-V, and the MSL searched for
//! constructs Metal rejects. Emission only: nothing here compiles MSL.

use std::collections::HashSet;

use super::model::{Model, Ty, text};
use super::sys;

/// Options of the Metal session; include roots follow.
pub(crate) const OPTIONS: &[&str] = &["-target", "metal", "-matrix-layout-row-major"];

/// Every difference between the Metal layout of the root and its pointees and the SPIR-V
/// layout in `model`. `layout` must be the live Metal program layout.
pub(crate) fn layout_issues(
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
pub(crate) fn msl_issues(msl: &str) -> Vec<String> {
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
                "MSL line {line_no}: a 64-bit fetch atomic, which Metal lacks (use orr.atomic)"
            ));
        }
        if let Some(params) = entry_parameters(t) {
            for p in params {
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
    }
    out
}

/// `source` with its `[shader("raygeneration")]` attributes blanked out, so a Metal session,
/// which rejects raygen entries when it loads a module, sees those entries as plain
/// functions. Line and column numbers stay the same.
pub(crate) fn without_raygen(source: &str) -> String {
    let mut out = source.to_owned();
    let mut from = 0;
    while let Some(i) = out[from..].find("[shader(").map(|i| i + from) {
        let rest = &out[i + "[shader(".len()..];
        let trimmed = rest.trim_start();
        let after_stage = trimmed
            .strip_prefix("\"raygeneration\"")
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
    const STAGES: &[&str] = &["[[kernel]]", "[[mesh]]", "[[object]]", "[[fragment]]"];
    if !STAGES.iter().any(|s| line.starts_with(s)) {
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
    fn raygen_attributes_blank_out_in_place() {
        let src = "[shader(\"compute\")]
void a() {}
[shader( \"raygeneration\" )]
void b() {}
";
        let out = without_raygen(src);
        assert_eq!(out.len(), src.len());
        assert!(out.contains("[shader(\"compute\")]") && !out.contains("raygeneration"));
        assert_eq!(out.lines().nth(3), Some("void b() {}"));
    }
}
