use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};

use super::layout::Plan;
use super::model::{Model, Scalar, Ty};

/// A rule the build checks on every module. [`Build::allow`](super::Build::allow) turns
/// one off.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Lint {
    /// `root` is the only shader parameter besides specialisation constants and the Metal
    /// bindings of `orr/abi.slang`: no other globals, no entry-point uniforms.
    SingleRoot,
    /// Root size is a multiple of 16; members are 32/64-bit scalars, 4-vectors, matrices
    /// with 4-wide rows, heap slots, TLAS addresses and pointers, each at its native
    /// alignment.
    RootLayout,
    /// Every field starts where the previous one ends and a struct's size is the sum of
    /// its fields.
    ImplicitPadding,
    /// No `bool` in roots or records.
    Bool,
    /// Specialisation constants take explicit ids (`[vk::constant_id(N)]`).
    SpecIds,
    /// `DescriptorHandle`, `getDescriptorFromHandle` and `RaytracingAccelerationStructure(`
    /// appear only in `orr/abi.slang`.
    HeapTokens,
    /// `RayQuery` appears only in `orr/trace.slang`.
    RayQuery,
    /// `ReorderThread` and `HitObject` appear only in `orr/coherence.slang`.
    Reorder,
    /// Cooperative-vector names appear only in `orr/coopvec.slang`.
    CoopVec,
    /// Every 64-bit atomic is a max whose result is unused.
    Atomic64,
    /// No two shader roots define the same top-level import name.
    DuplicateImport,
    /// A struct is generated only by the crate whose shader root declares it; others map
    /// it with [`Build::extern_type`](super::Build::extern_type).
    Ownership,
    /// The module's compute and mesh-pipeline entries emit MSL, lay out roots and pointees
    /// as the SPIR-V does, and the MSL has none of the constructs Metal rejects: 64-bit
    /// fetch atomics, a ray query inside a struct, a kernel parameter without a binding.
    /// Checked only with [`Build::metal_check`](super::Build::metal_check).
    Metal,
}

impl fmt::Display for Lint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// One lint failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LintError {
    pub lint: Lint,
    /// The module or file it was found in.
    pub module: String,
    pub message: String,
}

impl fmt::Display for LintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} [{}]", self.module, self.message, self.lint)
    }
}

struct TokenRule {
    lint: Lint,
    exact: &'static [&'static str],
    prefixes: &'static [&'static str],
    /// Flagged only when followed by `(`.
    calls: &'static [&'static str],
    /// The one file under `LIB_DIR` allowed to use them.
    home: Option<&'static str>,
}

const TOKEN_RULES: &[TokenRule] = &[
    TokenRule {
        lint: Lint::HeapTokens,
        exact: &["DescriptorHandle", "getDescriptorFromHandle"],
        prefixes: &[],
        calls: &["RaytracingAccelerationStructure"],
        home: Some("orr/abi.slang"),
    },
    TokenRule {
        lint: Lint::RayQuery,
        exact: &["RayQuery"],
        prefixes: &[],
        calls: &[],
        home: Some("orr/trace.slang"),
    },
    TokenRule {
        lint: Lint::Reorder,
        exact: &["ReorderThread", "HitObject"],
        prefixes: &[],
        calls: &[],
        home: Some("orr/coherence.slang"),
    },
    TokenRule {
        lint: Lint::CoopVec,
        exact: &[],
        prefixes: &["CoopVec", "coopVec", "OpCooperativeVector"],
        calls: &[],
        home: Some("orr/coopvec.slang"),
    },
    TokenRule {
        lint: Lint::SpecIds,
        exact: &["SpecializationConstant", "specialization_constant"],
        prefixes: &[],
        calls: &[],
        home: None,
    },
];

/// Identifiers with their line and whether `(` follows, outside comments and strings.
pub(crate) fn tokens(src: &str) -> Vec<(&str, u32, bool)> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let mut line = 1;
    while i < b.len() {
        match b[i] {
            b'\n' => {
                line += 1;
                i += 1;
            }
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i < b.len() && !(b[i] == b'*' && b.get(i + 1) == Some(&b'/')) {
                    line += u32::from(b[i] == b'\n');
                    i += 1;
                }
                i += 2;
            }
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                let mut j = i;
                while j < b.len() && (b[j] == b' ' || b[j] == b'\t') {
                    j += 1;
                }
                out.push((&src[start..i], line, b.get(j) == Some(&b'(')));
            }
            _ => i += 1,
        }
    }
    out
}

/// Token lints over one source file.
pub(crate) fn source_lints(path: &Path, src: &str, lib_dir: &Path) -> Vec<LintError> {
    let mut out = Vec::new();
    for (token, line, call) in tokens(src) {
        for rule in TOKEN_RULES {
            let hit = rule.exact.contains(&token)
                || rule.prefixes.iter().any(|p| token.starts_with(p))
                || (call && rule.calls.contains(&token));
            if !hit {
                continue;
            }
            if rule
                .home
                .is_some_and(|home| same_file(path, &lib_dir.join(home)))
            {
                continue;
            }
            let place = match rule.home {
                Some(home) => format!("; it belongs in {home}"),
                None => String::new(),
            };
            out.push(LintError {
                lint: rule.lint,
                module: format!("{}:{line}", path.display()),
                message: if rule.lint == Lint::SpecIds {
                    format!("`{token}` numbers constants implicitly; use [vk::constant_id(N)]")
                } else {
                    format!("`{token}`{place}")
                },
            });
        }
    }
    out
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Lints read from a compiled module's model and the generated layout.
pub(crate) fn model_lints(
    module: &str,
    model: &Model,
    plans: &HashMap<String, Plan>,
    lib_dir: &Path,
) -> Vec<LintError> {
    let mut out = Vec::new();
    let mut push = |lint, message: String| {
        out.push(LintError {
            lint,
            module: module.to_owned(),
            message,
        });
    };
    let abi = lib_dir.join("orr/abi.slang");
    for p in &model.stray_params {
        if p.file.as_deref().is_some_and(|f| same_file(f, &abi)) {
            continue;
        }
        let place = match &p.entry {
            Some(e) => format!("entry point {e} has uniform parameter {}", p.name),
            None => format!("global parameter {}", p.name),
        };
        push(
            Lint::SingleRoot,
            format!("{place}; `root` is the only parameter"),
        );
    }
    if let Some(root) = &model.root {
        let def = &model.structs[root];
        let size = plans[root].size;
        if !size.is_multiple_of(16) {
            push(
                Lint::RootLayout,
                format!("root {root} is {size} bytes, not a multiple of 16"),
            );
        }
        for f in &def.fields {
            match root_member_alignment(&f.ty) {
                None => push(
                    Lint::RootLayout,
                    format!(
                        "root field {} is {}; roots hold 32/64-bit scalars, 4-vectors, matrices with 4-wide rows, heap slots, Tlas and pointers",
                        f.name,
                        describe(&f.ty)
                    ),
                ),
                Some(a) if f.offset % a != 0 => push(
                    Lint::RootLayout,
                    format!(
                        "root field {} at offset {} is not at its native alignment {a}",
                        f.name, f.offset
                    ),
                ),
                Some(_) => {}
            }
        }
    }
    for (name, def) in &model.structs {
        for (next, bytes) in plans[name].gaps(def) {
            let place = match next {
                Some(f) => format!("before field {f}"),
                None => "at the end".to_owned(),
            };
            push(
                Lint::ImplicitPadding,
                format!(
                    "struct {name} has {bytes} bytes of implicit padding {place}; add explicit _pad fields"
                ),
            );
        }
        for f in &def.fields {
            if has_bool(&f.ty) {
                push(
                    Lint::Bool,
                    format!("struct {name} field {} is bool; use uint", f.name),
                );
            }
        }
    }
    for op in &model.bad_atomics {
        push(
            Lint::Atomic64,
            format!(
                "64-bit atomic (opcode {op}) is not a max with its result unused; use orr.atomic"
            ),
        );
    }
    out
}

fn has_bool(ty: &Ty) -> bool {
    match ty {
        Ty::Scalar(Scalar::Bool) | Ty::Vector(Scalar::Bool, _) => true,
        Ty::Array { element, .. } => has_bool(element),
        _ => false,
    }
}

/// The native alignment of an allowed root member.
fn root_member_alignment(ty: &Ty) -> Option<u32> {
    match ty {
        Ty::Scalar(s) if matches!(s.size(), 4 | 8) && *s != Scalar::Bool => Some(s.size()),
        Ty::Vector(s, 4) if matches!(s.size(), 4 | 8) && *s != Scalar::Bool => Some(4 * s.size()),
        Ty::Matrix {
            scalar, cols: 4, ..
        } if scalar.size() == 4 => Some(16),
        Ty::Handle(h) => Some(h.scalar().size()),
        Ty::Pointer(_) => Some(8),
        _ => None,
    }
}

fn describe(ty: &Ty) -> String {
    match ty {
        Ty::Scalar(s) => format!("{s:?}"),
        Ty::Vector(s, n) => format!("{s:?}x{n} vector"),
        Ty::Matrix { rows, cols, .. } => format!("a {rows}x{cols} matrix"),
        Ty::Array { len, .. } => format!("an array of {len}"),
        Ty::Struct(s) => format!("struct {s}"),
        Ty::Pointer(_) => "a pointer".into(),
        Ty::Handle(h) => format!("{h:?}"),
    }
}

/// Top-level import names (files and directories holding Slang) of each root.
pub(crate) fn duplicate_imports(roots: &[PathBuf]) -> Vec<LintError> {
    let mut seen: BTreeMap<String, &Path> = BTreeMap::new();
    let mut out = Vec::new();
    for root in roots {
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        let mut names: Vec<String> = entries
            .filter_map(Result::ok)
            .filter_map(|e| {
                let path = e.path();
                if path.is_dir() {
                    holds_slang(&path).then(|| e.file_name().to_string_lossy().into_owned())
                } else if is_slang(&path) {
                    path.file_stem().map(|s| s.to_string_lossy().into_owned())
                } else {
                    None
                }
            })
            .collect();
        names.sort();
        names.dedup();
        for name in names {
            if let Some(other) = seen.get(&name) {
                out.push(LintError {
                    lint: Lint::DuplicateImport,
                    module: root.display().to_string(),
                    message: format!(
                        "top-level import name `{name}` is also defined under {}",
                        other.display()
                    ),
                });
            } else {
                seen.insert(name, root);
            }
        }
    }
    out
}

fn is_slang(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("slang" | "slangh" | "h")
    )
}

fn holds_slang(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|e| {
            let p = e.path();
            if p.is_dir() {
                holds_slang(&p)
            } else {
                is_slang(&p)
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_skip_comments_and_strings() {
        let src = "// RayQuery\n/* HitObject\n */ x \"RayQuery\" RaytracingAccelerationStructure (a)\nRayQuery<0> q;";
        let t = tokens(src);
        let names: Vec<_> = t.iter().map(|(n, ..)| *n).collect();
        assert_eq!(
            names,
            ["x", "RaytracingAccelerationStructure", "a", "RayQuery", "q"]
        );
        assert_eq!(t[1], ("RaytracingAccelerationStructure", 3, true));
        assert_eq!(t[3].1, 4);
    }
}
