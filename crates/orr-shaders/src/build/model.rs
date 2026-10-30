//! What the build knows about one compiled module: entry points, specialisation constants
//! and every type reachable from the root, with offsets and strides read from the SPIR-V.
//!
//! Reflection supplies source-level types (struct names, generic arguments, `bool`, matrix
//! shapes, array lengths). It does not describe a pointer's pointee, but a pointer type's
//! full name spells it (`Ptr<View, ...>`), and `FindTypeByName` resolves that name.

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, c_char};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::slang::{Session, blob_text};
use super::spirv::{self, Type, decoration};
use super::sys;
use crate::Stage;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum Scalar {
    Bool,
    I8,
    U8,
    I16,
    U16,
    F16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
}

impl Scalar {
    /// Bytes in memory; `bool` is stored as a 32-bit value.
    pub(crate) fn size(self) -> u32 {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 | Self::F16 => 2,
            Self::Bool | Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }

    pub(crate) fn rust(self) -> &'static str {
        match self {
            Self::Bool | Self::U32 => "u32",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 | Self::F16 => "u16",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::F64 => "f64",
        }
    }

    fn from_spirv(t: &Type) -> Option<Self> {
        Some(match *t {
            Type::Bool => Self::Bool,
            Type::Int { width, signed } => match (width, signed) {
                (8, true) => Self::I8,
                (8, false) => Self::U8,
                (16, true) => Self::I16,
                (16, false) => Self::U16,
                (32, true) => Self::I32,
                (32, false) => Self::U32,
                (64, true) => Self::I64,
                (64, false) => Self::U64,
                _ => return None,
            },
            Type::Float { width: 16 } => Self::F16,
            Type::Float { width: 32 } => Self::F32,
            Type::Float { width: 64 } => Self::F64,
            _ => return None,
        })
    }

    fn from_slang(s: sys::SlangScalarType) -> Option<Self> {
        use sys::SlangScalarType as S;
        Some(match s {
            S::SLANG_SCALAR_TYPE_BOOL => Self::Bool,
            S::SLANG_SCALAR_TYPE_INT8 => Self::I8,
            S::SLANG_SCALAR_TYPE_UINT8 => Self::U8,
            S::SLANG_SCALAR_TYPE_INT16 => Self::I16,
            S::SLANG_SCALAR_TYPE_UINT16 => Self::U16,
            S::SLANG_SCALAR_TYPE_FLOAT16 => Self::F16,
            S::SLANG_SCALAR_TYPE_INT32 => Self::I32,
            S::SLANG_SCALAR_TYPE_UINT32 => Self::U32,
            S::SLANG_SCALAR_TYPE_FLOAT32 => Self::F32,
            S::SLANG_SCALAR_TYPE_INT64 => Self::I64,
            S::SLANG_SCALAR_TYPE_UINT64 => Self::U64,
            S::SLANG_SCALAR_TYPE_FLOAT64 => Self::F64,
            _ => return None,
        })
    }
}

/// The heap-slot and TLAS types of `orr.abi`, recognised by struct name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum Handle {
    Texture,
    StorageImage,
    Sampler,
    Tlas,
}

impl Handle {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "TextureIndex" => Self::Texture,
            "StorageImageIndex" => Self::StorageImage,
            "SamplerIndex" => Self::Sampler,
            "Tlas" => Self::Tlas,
            _ => return None,
        })
    }

    pub(crate) fn rust(self) -> &'static str {
        match self {
            Self::Texture => "::orr_shaders::abi::TextureIndex",
            Self::StorageImage => "::orr_shaders::abi::StorageImageIndex",
            Self::Sampler => "::orr_shaders::abi::SamplerIndex",
            Self::Tlas => "::orr_shaders::abi::Tlas",
        }
    }

    pub(crate) fn scalar(self) -> Scalar {
        if self == Self::Tlas {
            Scalar::U64
        } else {
            Scalar::U32
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum Ty {
    Scalar(Scalar),
    Vector(Scalar, u32),
    /// `rows` vectors of `cols` scalars in memory order when not `column_major`.
    Matrix {
        scalar: Scalar,
        rows: u32,
        cols: u32,
        stride: u32,
        column_major: bool,
    },
    Array {
        element: Box<Ty>,
        len: u32,
        stride: u32,
    },
    Struct(String),
    Pointer(Box<Ty>),
    Handle(Handle),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Field {
    pub(crate) name: String,
    pub(crate) ty: Ty,
    pub(crate) offset: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StructDef {
    /// The Slang name, with generic arguments (`Gen<float>`).
    pub(crate) name: String,
    pub(crate) fields: Vec<Field>,
    /// The size in bytes when the SPIR-V states it (an array or pointer stride).
    pub(crate) size: Option<u32>,
    /// The file that declares it, when reflection finds the declaration.
    pub(crate) file: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SpecConstant {
    pub(crate) name: String,
    pub(crate) id: u32,
    pub(crate) scalar: Scalar,
    /// The default value's bits; a bool is 0 or 1.
    pub(crate) default: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Entry {
    pub(crate) name: String,
    pub(crate) stage: Stage,
    pub(crate) workgroup: [u32; 3],
}

/// A global parameter other than the root and specialisation constants, or an entry-point
/// uniform parameter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StrayParam {
    pub(crate) name: String,
    pub(crate) entry: Option<String>,
    /// The file declaring a global parameter, when reflection finds it.
    pub(crate) file: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Model {
    pub(crate) entries: Vec<Entry>,
    pub(crate) root: Option<String>,
    pub(crate) structs: BTreeMap<String, StructDef>,
    pub(crate) spec: Vec<SpecConstant>,
    pub(crate) stray_params: Vec<StrayParam>,
    /// 64-bit atomics other than a max whose result is unused, by opcode.
    pub(crate) bad_atomics: Vec<u16>,
}

/// # Safety
/// `p` is null or a nul-terminated string that outlives the call.
pub(crate) unsafe fn text(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: Slang returns nul-terminated strings that live as long as the program.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

struct Extract<'a> {
    api: &'a sys::SlangLib,
    layout: *mut sys::SlangReflection,
    spv: &'a spirv::Module,
    decls: &'a Decls,
    structs: BTreeMap<String, StructDef>,
}

/// Declaring files of the structs and global variables of every loaded module, by name.
#[derive(Default)]
struct Decls {
    structs: HashMap<String, PathBuf>,
    globals: HashMap<String, PathBuf>,
}

/// Reads the model of a linked program. `layout` must stay alive for the call.
pub(crate) fn extract(
    session: &Session,
    layout: *mut sys::SlangReflection,
    words: &[u32],
) -> Result<Model, String> {
    let spv = spirv::Module::parse(words)?;
    let decls = declarations(session);
    let api = &session.lib.api;
    let mut x = Extract {
        api,
        layout,
        spv: &spv,
        decls: &decls,
        structs: BTreeMap::new(),
    };
    // SAFETY: `layout` is the live program layout of the session's linked program.
    unsafe { x.run() }
}

/// Every struct and module-scope variable declared in a module loaded into the session.
fn declarations(session: &Session) -> Decls {
    use sys::SlangDeclKind as K;
    let api = &session.lib.api;
    let mut out = Decls::default();
    let mut stack: Vec<(*mut sys::SlangReflectionDecl, bool)> = session
        .loaded_modules()
        .iter()
        .map(super::slang::Module::reflection)
        .filter(|d| !d.is_null())
        .map(|d| (d, true))
        .collect();
    // SAFETY: decls come from modules owned by the live session.
    unsafe {
        while let Some((decl, module_scope)) = stack.pop() {
            for i in 0..api.spReflectionDecl_getChildrenCount(decl) {
                let child = api.spReflectionDecl_getChild(decl, i);
                if child.is_null() {
                    continue;
                }
                let kind = api.spReflectionDecl_getKind(child);
                let map = match kind {
                    K::SLANG_DECL_KIND_STRUCT => Some(&mut out.structs),
                    K::SLANG_DECL_KIND_VARIABLE if module_scope => Some(&mut out.globals),
                    _ => None,
                };
                if let Some(map) = map
                    && let Some((file, _)) = session.decl_location(child)
                {
                    map.entry(text(api.spReflectionDecl_getName(child)))
                        .or_insert(file);
                }
                if kind != K::SLANG_DECL_KIND_FUNC {
                    stack.push((child, false));
                }
            }
        }
    }
    out
}

impl Extract<'_> {
    unsafe fn run(&mut self) -> Result<Model, String> {
        // SAFETY: every reflection pointer below comes from `self.layout`.
        unsafe {
            let api = self.api;
            let l = self.layout;
            let spec_defaults = self.spv.spec_defaults();
            let mut root = None;
            let mut spec = Vec::new();
            let mut stray_params = Vec::new();
            for i in 0..api.spReflection_GetParameterCount(l) {
                let vl = api.spReflection_GetParameterByIndex(l, i);
                let var = api.spReflectionVariableLayout_GetVariable(vl);
                let name = text(api.spReflectionVariable_GetName(var));
                let tl = api.spReflectionVariableLayout_GetTypeLayout(vl);
                let category = api.spReflectionTypeLayout_GetParameterCategory(tl);
                let ty = api.spReflectionTypeLayout_GetType(tl);
                let kind = api.spReflectionType_GetKind(ty);
                use sys::SlangParameterCategory as C;
                if category == C::SLANG_PARAMETER_CATEGORY_SPECIALIZATION_CONSTANT {
                    let id = api.spReflectionVariableLayout_GetOffset(vl, category) as u32;
                    if id == spirv::OMM_SPEC_ID {
                        continue;
                    }
                    let scalar = Scalar::from_slang(api.spReflectionType_GetScalarType(ty))
                        .ok_or_else(|| format!("specialisation constant {name} is not a scalar"))?;
                    let default = spec_defaults.get(&id).map_or(0, |(_, w)| {
                        u64::from(w[0]) | w.get(1).map_or(0, |&h| u64::from(h) << 32)
                    });
                    spec.push(SpecConstant {
                        name,
                        id,
                        scalar,
                        default,
                    });
                } else if root.is_none()
                    && kind == sys::SlangTypeKind::SLANG_TYPE_KIND_CONSTANT_BUFFER
                    && category == C::SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT
                    && api.spReflectionVariableLayout_GetOffset(vl, category) == 0
                    && api.spReflectionVariableLayout_GetSpace(vl, category) == 0
                {
                    let element = api.spReflectionType_GetElementType(ty);
                    let block = self
                        .spv
                        .root_block()
                        .ok_or_else(|| format!("root {name} is unused by every entry point, so the SPIR-V has no layout for it"))?;
                    match self.reflected(element, block, None, &name)? {
                        Ty::Struct(s) => root = Some(s),
                        _ => return Err(format!("the root {name} is not a struct")),
                    }
                } else {
                    let file = self.decls.globals.get(&name).cloned();
                    stray_params.push(StrayParam {
                        name,
                        entry: None,
                        file,
                    });
                }
            }
            spec.sort_by_key(|s| s.id);

            let mut entries = Vec::new();
            for i in 0..api.spReflection_getEntryPointCount(l) {
                let ep = api.spReflection_getEntryPointByIndex(l, i);
                let name = text(api.spReflectionEntryPoint_getName(ep));
                use sys::SlangStage as S;
                let stage = match api.spReflectionEntryPoint_getStage(ep) {
                    S::SLANG_STAGE_COMPUTE => Stage::Compute,
                    S::SLANG_STAGE_AMPLIFICATION => Stage::Task,
                    S::SLANG_STAGE_MESH => Stage::Mesh,
                    S::SLANG_STAGE_FRAGMENT => Stage::Pixel,
                    S::SLANG_STAGE_RAY_GENERATION => Stage::RayGen,
                    other => {
                        return Err(format!(
                            "entry point {name} has stage {}, which orr does not create pipelines for",
                            other.0
                        ));
                    }
                };
                let workgroup = if matches!(stage, Stage::Compute | Stage::Task | Stage::Mesh) {
                    let mut size = [0u64; 3];
                    api.spReflectionEntryPoint_getComputeThreadGroupSize(ep, 3, size.as_mut_ptr());
                    size.map(|s| s as u32)
                } else {
                    [1, 1, 1]
                };
                for p in 0..api.spReflectionEntryPoint_getParameterCount(ep) {
                    let vl = api.spReflectionEntryPoint_getParameterByIndex(ep, p);
                    let tl = api.spReflectionVariableLayout_GetTypeLayout(vl);
                    use sys::SlangParameterCategory as C;
                    let category = api.spReflectionTypeLayout_GetParameterCategory(tl);
                    if !matches!(
                        category,
                        C::SLANG_PARAMETER_CATEGORY_NONE
                            | C::SLANG_PARAMETER_CATEGORY_VARYING_INPUT
                            | C::SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT
                    ) {
                        let var = api.spReflectionVariableLayout_GetVariable(vl);
                        stray_params.push(StrayParam {
                            name: text(api.spReflectionVariable_GetName(var)),
                            entry: Some(name.clone()),
                            file: None,
                        });
                    }
                }
                entries.push(Entry {
                    name,
                    stage,
                    workgroup,
                });
            }

            let bad_atomics = self
                .spv
                .atomics64
                .iter()
                .filter(|a| {
                    !matches!(a.op, spirv::op::ATOMIC_UMAX | spirv::op::ATOMIC_SMAX)
                        || a.result_used
                })
                .map(|a| a.op)
                .collect();

            Ok(Model {
                entries,
                root,
                structs: std::mem::take(&mut self.structs),
                spec,
                stray_params,
                bad_atomics,
            })
        }
    }

    fn spirv(&self, id: u32) -> Result<&Type, String> {
        self.spv
            .types
            .get(&id)
            .ok_or_else(|| format!("SPIR-V type %{id} is not declared"))
    }

    /// The SPIR-V struct behind `id`: itself, or the `data` member of a layout wrapper.
    fn unwrap_storage(&self, id: u32) -> u32 {
        match self.spv.types.get(&id) {
            Some(Type::Struct { members }) if members.len() == 1 => {
                let wrapper =
                    self.spv.names.get(&id).is_some_and(|n| {
                        n.starts_with("_MatrixStorage_") || n.starts_with("_Array_")
                    });
                if wrapper { members[0] } else { id }
            }
            _ => id,
        }
    }

    fn is_column_major_storage(&self, id: u32) -> bool {
        self.spv
            .names
            .get(&id)
            .is_some_and(|n| n.starts_with("_MatrixStorage_") && n.contains("ColMajor"))
    }

    fn record_size(&mut self, ty: &Ty, size: Option<u32>) {
        if let (Ty::Struct(name), Some(size)) = (ty, size)
            && let Some(def) = self.structs.get_mut(name)
        {
            def.size = Some(size);
        }
    }

    /// A type from reflection, laid out by the SPIR-V type `id`. `member` is the SPIR-V
    /// struct and member index holding it, for member decorations.
    unsafe fn reflected(
        &mut self,
        rt: *mut sys::SlangReflectionType,
        id: u32,
        member: Option<(u32, u32)>,
        context: &str,
    ) -> Result<Ty, String> {
        // SAFETY: `rt` comes from the live program layout.
        unsafe {
            let api = self.api;
            use sys::SlangTypeKind as K;
            let kind = api.spReflectionType_GetKind(rt);
            match kind {
                K::SLANG_TYPE_KIND_SCALAR | K::SLANG_TYPE_KIND_ENUM => {
                    let scalar = Scalar::from_slang(api.spReflectionType_GetScalarType(rt))
                        .or_else(|| self.spirv(id).ok().and_then(Scalar::from_spirv))
                        .ok_or_else(|| format!("{context}: unsupported scalar type"))?;
                    if scalar != Scalar::Bool {
                        self.expect_scalar(id, scalar, context)?;
                    }
                    Ok(Ty::Scalar(scalar))
                }
                K::SLANG_TYPE_KIND_VECTOR => {
                    let element = api.spReflectionType_GetElementType(rt);
                    let scalar = Scalar::from_slang(api.spReflectionType_GetScalarType(element))
                        .ok_or_else(|| format!("{context}: unsupported vector element"))?;
                    let count = api.spReflectionType_GetElementCount(rt) as u32;
                    match self.spirv(id)? {
                        Type::Vector { count: c, .. } if *c == count => {}
                        other => {
                            return Err(format!(
                                "{context}: reflection says a {count}-vector, SPIR-V says {other:?}"
                            ));
                        }
                    }
                    Ok(Ty::Vector(scalar, count))
                }
                K::SLANG_TYPE_KIND_MATRIX => {
                    let rows = api.spReflectionType_GetRowCount(rt);
                    let cols = api.spReflectionType_GetColumnCount(rt);
                    let element = api.spReflectionType_GetElementType(rt);
                    let scalar = Scalar::from_slang(api.spReflectionType_GetScalarType(element))
                        .ok_or_else(|| format!("{context}: unsupported matrix element"))?;
                    let (vectors, len, stride, column_major) =
                        self.matrix_memory(id, member, context)?;
                    let column_major = column_major || (vectors, len) != (rows, cols);
                    Ok(Ty::Matrix {
                        scalar,
                        rows,
                        cols,
                        stride,
                        column_major,
                    })
                }
                K::SLANG_TYPE_KIND_ARRAY => {
                    let len = api.spReflectionType_GetElementCount(rt) as u32;
                    let element = api.spReflectionType_GetElementType(rt);
                    let array = self.unwrap_storage(id);
                    let (element_id, spirv_len) = match self.spirv(array)? {
                        Type::Array { element, length } => (*element, *length),
                        other => {
                            return Err(format!(
                                "{context}: reflection says an array, SPIR-V says {other:?}"
                            ));
                        }
                    };
                    if spirv_len != len {
                        return Err(format!(
                            "{context}: reflection says {len} elements, SPIR-V says {spirv_len}"
                        ));
                    }
                    let stride = self
                        .spv
                        .decoration(array, decoration::ARRAY_STRIDE)
                        .and_then(|d| d.first().copied())
                        .ok_or_else(|| format!("{context}: array has no stride"))?;
                    let element = self.reflected(element, element_id, None, context)?;
                    self.record_size(&element, Some(stride));
                    Ok(Ty::Array {
                        element: Box::new(element),
                        len,
                        stride,
                    })
                }
                K::SLANG_TYPE_KIND_STRUCT => self.struct_from_reflection(rt, id, context),
                K::SLANG_TYPE_KIND_POINTER => self.pointer(rt, id, context),
                other => Err(format!(
                    "{context}: type kind {} is not allowed in roots or records",
                    other.0
                )),
            }
        }
    }

    fn expect_scalar(&self, id: u32, scalar: Scalar, context: &str) -> Result<(), String> {
        let found = self.spirv(id)?;
        match Scalar::from_spirv(found) {
            Some(s) if s.size() == scalar.size() => Ok(()),
            _ => Err(format!(
                "{context}: reflection says {scalar:?}, SPIR-V says {found:?}"
            )),
        }
    }

    /// (vectors in memory, scalars per vector, vector stride, SPIR-V says column-major).
    fn matrix_memory(
        &self,
        id: u32,
        member: Option<(u32, u32)>,
        context: &str,
    ) -> Result<(u32, u32, u32, bool), String> {
        let vector_len = |v: u32| match self.spv.types.get(&v) {
            Some(Type::Vector { count, .. }) => Ok(*count),
            other => Err(format!("{context}: matrix vector is {other:?}")),
        };
        match self.spirv(id)? {
            Type::Matrix { column, count } => {
                let (parent, index) =
                    member.ok_or_else(|| format!("{context}: matrix outside a struct"))?;
                let stride = self
                    .spv
                    .member_decoration(parent, index, decoration::MATRIX_STRIDE)
                    .ok_or_else(|| format!("{context}: matrix has no stride"))?;
                let len = vector_len(*column)?;
                if self.spv.member_has(parent, index, decoration::ROW_MAJOR) {
                    // Slang's column-major matrices are SPIR-V row-major.
                    Ok((len, *count, stride, true))
                } else {
                    Ok((*count, len, stride, false))
                }
            }
            Type::Struct { .. } => {
                let data = self.unwrap_storage(id);
                let Type::Array { element, length } = self.spirv(data)? else {
                    return Err(format!("{context}: matrix storage without a data array"));
                };
                let stride = self
                    .spv
                    .decoration(data, decoration::ARRAY_STRIDE)
                    .and_then(|d| d.first().copied())
                    .ok_or_else(|| format!("{context}: matrix rows have no stride"))?;
                Ok((
                    *length,
                    vector_len(*element)?,
                    stride,
                    self.is_column_major_storage(id),
                ))
            }
            other => Err(format!(
                "{context}: reflection says a matrix, SPIR-V says {other:?}"
            )),
        }
    }

    /// Slang's full name of a type (`Pair<float>`, `Ptr<View, ...>`).
    unsafe fn full_name(&self, rt: *mut sys::SlangReflectionType) -> String {
        let mut blob: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `rt` comes from the live program layout; the blob comes back owned.
        unsafe {
            self.api
                .spReflectionType_GetFullName(rt, (&raw mut blob).cast())
        };
        blob_text(blob)
    }

    unsafe fn struct_from_reflection(
        &mut self,
        rt: *mut sys::SlangReflectionType,
        id: u32,
        context: &str,
    ) -> Result<Ty, String> {
        // SAFETY: `rt` comes from the live program layout.
        unsafe {
            let api = self.api;
            let name = self.full_name(rt);
            let short = text(api.spReflectionType_GetName(rt));
            if let Some(h) = Handle::from_name(&short) {
                return self.handle(h, id, context);
            }
            if self.structs.contains_key(&name) {
                return Ok(Ty::Struct(name));
            }
            let Type::Struct { members } = self.spirv(id)?.clone() else {
                return Err(format!(
                    "{context}: reflection says struct {name}, SPIR-V says otherwise"
                ));
            };
            let count = api.spReflectionType_GetFieldCount(rt);
            if count as usize != members.len() {
                return Err(format!(
                    "struct {name}: reflection has {count} fields, SPIR-V has {}",
                    members.len()
                ));
            }
            self.structs.insert(
                name.clone(),
                StructDef {
                    name: name.clone(),
                    fields: Vec::new(),
                    size: None,
                    file: self.decls.structs.get(&short).cloned(),
                },
            );
            let mut fields = Vec::new();
            for (i, &member) in members.iter().enumerate() {
                let var = api.spReflectionType_GetFieldByIndex(rt, i as u32);
                let field = text(api.spReflectionVariable_GetName(var));
                let context = format!("{name}.{field}");
                let ty = self.reflected(
                    api.spReflectionVariable_GetType(var),
                    member,
                    Some((id, i as u32)),
                    &context,
                )?;
                fields.push(self.field(id, i as u32, field, ty)?);
            }
            self.structs.get_mut(&name).expect("inserted above").fields = fields;
            Ok(Ty::Struct(name))
        }
    }

    fn field(&self, id: u32, index: u32, name: String, ty: Ty) -> Result<Field, String> {
        let offset = self
            .spv
            .member_decoration(id, index, decoration::OFFSET)
            .ok_or_else(|| format!("field {name} has no offset in the SPIR-V"))?;
        Ok(Field { name, ty, offset })
    }

    fn handle(&self, h: Handle, id: u32, context: &str) -> Result<Ty, String> {
        let ok = match self.spirv(id)? {
            Type::Struct { members } if members.len() == 1 => self
                .spv
                .types
                .get(&members[0])
                .and_then(Scalar::from_spirv)
                .is_some_and(|s| {
                    s == h.scalar() || (h.scalar() == Scalar::U32 && s == Scalar::I32)
                }),
            _ => false,
        };
        if ok {
            Ok(Ty::Handle(h))
        } else {
            Err(format!(
                "{context}: {h:?} must be a struct of one {:?}",
                h.scalar()
            ))
        }
    }

    /// A pointer: reflection names the pointee in the pointer's full name, the SPIR-V
    /// pointer type gives its layout and, as its stride, its size.
    unsafe fn pointer(
        &mut self,
        rt: *mut sys::SlangReflectionType,
        id: u32,
        context: &str,
    ) -> Result<Ty, String> {
        let Type::Pointer { storage, pointee } = *self.spirv(id)? else {
            return Err(format!(
                "{context}: reflection says a pointer, SPIR-V says otherwise"
            ));
        };
        if storage != spirv::STORAGE_PHYSICAL {
            return Err(format!(
                "{context}: pointer is not a physical storage buffer pointer"
            ));
        }
        let size = self
            .spv
            .decoration(id, decoration::ARRAY_STRIDE)
            .and_then(|d| d.first().copied());
        // SAFETY: `rt` comes from the live program layout.
        let full = unsafe { self.full_name(rt) };
        let target = first_type_argument(&full)
            .ok_or_else(|| format!("{context}: cannot read the pointee of {full}"))?;
        let (target, pointee) = match (target.strip_suffix("[]"), self.spirv(pointee)?) {
            (Some(element), Type::RuntimeArray { element: id }) => (element, *id),
            _ => (target, pointee),
        };
        let cname = std::ffi::CString::new(target).expect("no nul in type names");
        // SAFETY: `layout` is the live program layout.
        let prt = unsafe {
            self.api
                .spReflection_FindTypeByName(self.layout, cname.as_ptr())
        };
        if prt.is_null() {
            return Err(format!(
                "{context}: reflection cannot find the pointee type {target}"
            ));
        }
        // SAFETY: as above.
        let ty = unsafe { self.reflected(prt, pointee, None, context)? };
        self.record_size(&ty, size);
        Ok(Ty::Pointer(Box::new(ty)))
    }
}

/// `X` in `Ptr<X, ...>`, with any brackets inside `X` kept.
fn first_type_argument(full: &str) -> Option<&str> {
    let open = full.find('<')?;
    let mut depth = 0;
    for (i, c) in full[open + 1..].char_indices() {
        match c {
            '<' => depth += 1,
            '>' if depth == 0 => return Some(full[open + 1..open + 1 + i].trim()),
            '>' => depth -= 1,
            ',' if depth == 0 => return Some(full[open + 1..open + 1 + i].trim()),
            _ => {}
        }
    }
    None
}
