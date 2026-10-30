//! A SPIR-V reader for the few instructions the build needs, the opacity-micromap pass and
//! the string pass over stripped modules. Unknown instructions are skipped by their word
//! count.

use std::collections::{HashMap, HashSet};

const MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;

pub(crate) mod op {
    pub(crate) const NAME: u16 = 5;
    pub(crate) const MEMBER_NAME: u16 = 6;
    pub(crate) const STRING: u16 = 7;
    pub(crate) const EXTENSION: u16 = 10;
    pub(crate) const EXT_INST_IMPORT: u16 = 11;
    pub(crate) const EXT_INST: u16 = 12;
    pub(crate) const MEMORY_MODEL: u16 = 14;
    pub(crate) const ENTRY_POINT: u16 = 15;
    pub(crate) const EXECUTION_MODE: u16 = 16;
    pub(crate) const CAPABILITY: u16 = 17;
    pub(crate) const TYPE_BOOL: u16 = 20;
    pub(crate) const TYPE_INT: u16 = 21;
    pub(crate) const TYPE_FLOAT: u16 = 22;
    pub(crate) const TYPE_VECTOR: u16 = 23;
    pub(crate) const TYPE_MATRIX: u16 = 24;
    pub(crate) const TYPE_ARRAY: u16 = 28;
    pub(crate) const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub(crate) const TYPE_STRUCT: u16 = 30;
    pub(crate) const TYPE_POINTER: u16 = 32;
    pub(crate) const CONSTANT_TRUE: u16 = 41;
    pub(crate) const CONSTANT: u16 = 43;
    pub(crate) const SPEC_CONSTANT_TRUE: u16 = 48;
    pub(crate) const SPEC_CONSTANT_FALSE: u16 = 49;
    pub(crate) const SPEC_CONSTANT: u16 = 50;
    pub(crate) const FUNCTION: u16 = 54;
    pub(crate) const VARIABLE: u16 = 59;
    pub(crate) const DECORATE: u16 = 71;
    pub(crate) const MEMBER_DECORATE: u16 = 72;
    pub(crate) const EXECUTION_MODE_ID: u16 = 331;
    pub(crate) const ATOMIC_LOAD: u16 = 227;
    pub(crate) const ATOMIC_STORE: u16 = 228;
    pub(crate) const ATOMIC_XOR: u16 = 242;
    pub(crate) const ATOMIC_SMAX: u16 = 238;
    pub(crate) const ATOMIC_UMAX: u16 = 239;
    pub(crate) const ATOMIC_FMIN_EXT: u16 = 5614;
    pub(crate) const ATOMIC_FMAX_EXT: u16 = 5615;
    pub(crate) const ATOMIC_FADD_EXT: u16 = 6035;
    pub(crate) const RAY_QUERY_PROCEED_KHR: u16 = 4477;
    pub(crate) const VECTOR_SHUFFLE: u16 = 79;
    pub(crate) const COMPOSITE_EXTRACT: u16 = 81;
    pub(crate) const COMPOSITE_INSERT: u16 = 82;
    pub(crate) const SWITCH: u16 = 251;
}

pub(crate) mod decoration {
    pub(crate) const SPEC_ID: u32 = 1;
    pub(crate) const ROW_MAJOR: u32 = 4;
    pub(crate) const ARRAY_STRIDE: u32 = 6;
    pub(crate) const MATRIX_STRIDE: u32 = 7;
    pub(crate) const BINDING: u32 = 33;
    pub(crate) const DESCRIPTOR_SET: u32 = 34;
    pub(crate) const OFFSET: u32 = 35;
}

pub(crate) const STORAGE_UNIFORM: u32 = 2;
pub(crate) const STORAGE_PHYSICAL: u32 = 5349;
const CAPABILITY_OMM: u32 = 5381;
const CAPABILITY_OMM_EXECUTION_MODE: u32 = 6032;
const EXECUTION_MODE_OMM_ID: u32 = 6031;
const OMM_EXTENSION: &str = "SPV_KHR_opacity_micromap";
/// The specialisation constant id of `orr.trace`'s `OMM_2_STATE`, which carries the forced
/// two-state micromap ray flag: 0 in a module, the flag in its variant.
pub(crate) const OMM_SPEC_ID: u32 = 0x8000_0000;
const RAY_FLAG_FORCE_OMM_2_STATE: u32 = 0x400;

/// (decoration, literal operands) pairs.
type Decorations = Vec<(u32, Vec<u32>)>;

/// A SPIR-V type.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Type {
    Bool,
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Vector { component: u32, count: u32 },
    Matrix { column: u32, count: u32 },
    Array { element: u32, length: u32 },
    RuntimeArray { element: u32 },
    Struct { members: Vec<u32> },
    Pointer { storage: u32, pointee: u32 },
}

/// One instruction: its opcode and its words after the first.
#[derive(Clone, Copy)]
struct Inst<'a> {
    op: u16,
    args: &'a [u32],
}

fn instructions(words: &[u32]) -> impl Iterator<Item = (usize, Inst<'_>)> {
    let mut at = HEADER_WORDS;
    std::iter::from_fn(move || {
        let first = *words.get(at)?;
        let count = (first >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return None;
        }
        let inst = Inst {
            op: first as u16,
            args: &words[at + 1..at + count],
        };
        let start = at;
        at += count;
        Some((start, inst))
    })
}

/// A nul-terminated literal string, and the words after it.
fn string(words: &[u32]) -> (String, &[u32]) {
    let mut bytes = Vec::new();
    for (i, w) in words.iter().enumerate() {
        for b in w.to_le_bytes() {
            if b == 0 {
                return (
                    String::from_utf8_lossy(&bytes).into_owned(),
                    &words[i + 1..],
                );
            }
            bytes.push(b);
        }
    }
    (String::from_utf8_lossy(&bytes).into_owned(), &[])
}

fn encode_string(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A 64-bit atomic instruction.
#[derive(Clone, Debug)]
pub(crate) struct Atomic64 {
    pub(crate) op: u16,
    pub(crate) result_used: bool,
}

/// The parts of a module the build reads.
#[derive(Default, Debug)]
pub(crate) struct Module {
    pub(crate) names: HashMap<u32, String>,
    decorations: HashMap<u32, Decorations>,
    member_decorations: HashMap<(u32, u32), Decorations>,
    pub(crate) types: HashMap<u32, Type>,
    constants: HashMap<u32, u64>,
    /// Result type and literal words of spec constants; booleans carry one word, 0 or 1.
    spec_constants: HashMap<u32, (u32, Vec<u32>)>,
    variables: Vec<(u32, u32, u32)>,
    /// Entry point function ids.
    pub(crate) entries: Vec<u32>,
    pub(crate) atomics64: Vec<Atomic64>,
    pub(crate) ray_query: bool,
}

impl Module {
    pub(crate) fn parse(words: &[u32]) -> Result<Self, String> {
        if words.len() < HEADER_WORDS || words[0] != MAGIC {
            return Err("not a SPIR-V module".into());
        }
        let mut m = Self::default();
        let mut uses: HashSet<u32> = HashSet::new();
        let mut atomics: Vec<(u16, u32, u32)> = Vec::new();
        for (_, inst) in instructions(words) {
            let a = inst.args;
            match inst.op {
                op::NAME => {
                    m.names.insert(a[0], string(&a[1..]).0);
                }
                op::ENTRY_POINT => {
                    uses.extend(string(&a[2..]).1);
                    m.entries.push(a[1]);
                }
                op::TYPE_BOOL => {
                    m.types.insert(a[0], Type::Bool);
                }
                op::TYPE_INT => {
                    m.types.insert(
                        a[0],
                        Type::Int {
                            width: a[1],
                            signed: a[2] != 0,
                        },
                    );
                }
                op::TYPE_FLOAT => {
                    m.types.insert(a[0], Type::Float { width: a[1] });
                }
                op::TYPE_VECTOR => {
                    m.types.insert(
                        a[0],
                        Type::Vector {
                            component: a[1],
                            count: a[2],
                        },
                    );
                }
                op::TYPE_MATRIX => {
                    m.types.insert(
                        a[0],
                        Type::Matrix {
                            column: a[1],
                            count: a[2],
                        },
                    );
                }
                op::TYPE_ARRAY => {
                    let length = m.constants.get(&a[2]).copied().unwrap_or(0) as u32;
                    m.types.insert(
                        a[0],
                        Type::Array {
                            element: a[1],
                            length,
                        },
                    );
                }
                op::TYPE_RUNTIME_ARRAY => {
                    m.types.insert(a[0], Type::RuntimeArray { element: a[1] });
                }
                op::TYPE_STRUCT => {
                    m.types.insert(
                        a[0],
                        Type::Struct {
                            members: a[1..].to_vec(),
                        },
                    );
                }
                op::TYPE_POINTER => {
                    m.types.insert(
                        a[0],
                        Type::Pointer {
                            storage: a[1],
                            pointee: a[2],
                        },
                    );
                }
                op::CONSTANT => {
                    let value = u64::from(a[2]) | a.get(3).map_or(0, |&h| u64::from(h) << 32);
                    m.constants.insert(a[1], value);
                }
                op::SPEC_CONSTANT_TRUE | op::SPEC_CONSTANT_FALSE => {
                    let v = u32::from(inst.op == op::SPEC_CONSTANT_TRUE);
                    m.spec_constants.insert(a[1], (a[0], vec![v]));
                }
                op::SPEC_CONSTANT => {
                    m.spec_constants.insert(a[1], (a[0], a[2..].to_vec()));
                }
                op::VARIABLE => {
                    m.variables.push((a[1], a[0], a[2]));
                    uses.extend(&a[3..]);
                }
                op::DECORATE => {
                    m.decorations
                        .entry(a[0])
                        .or_default()
                        .push((a[1], a[2..].to_vec()));
                }
                op::MEMBER_DECORATE => {
                    m.member_decorations
                        .entry((a[0], a[1]))
                        .or_default()
                        .push((a[2], a[3..].to_vec()));
                }
                op::RAY_QUERY_PROCEED_KHR => {
                    m.ray_query = true;
                    uses.extend(&a[2..]);
                }
                op::COMPOSITE_EXTRACT => {
                    uses.insert(a[2]);
                }
                op::COMPOSITE_INSERT | op::VECTOR_SHUFFLE => uses.extend(&a[2..4]),
                op::SWITCH => {
                    uses.insert(a[0]);
                }
                o if is_atomic(o) => {
                    atomics.push((o, a[0], a[1]));
                    uses.extend(&a[2..]);
                }
                o if takes_literals(o) => {}
                _ => uses.extend(inst.args),
            }
        }
        for (o, ty, result) in atomics {
            if m.scalar_width(ty) == Some(64) {
                m.atomics64.push(Atomic64 {
                    op: o,
                    result_used: uses.contains(&result),
                });
            }
        }
        Ok(m)
    }

    fn scalar_width(&self, ty: u32) -> Option<u32> {
        match self.types.get(&ty)? {
            Type::Int { width, .. } | Type::Float { width } => Some(*width),
            Type::Vector { component, .. } => self.scalar_width(*component),
            _ => None,
        }
    }

    pub(crate) fn decoration(&self, id: u32, which: u32) -> Option<&[u32]> {
        self.decorations
            .get(&id)?
            .iter()
            .find(|(d, _)| *d == which)
            .map(|(_, v)| v.as_slice())
    }

    pub(crate) fn member_decoration(&self, id: u32, member: u32, which: u32) -> Option<u32> {
        self.member_decorations
            .get(&(id, member))?
            .iter()
            .find(|(d, _)| *d == which)
            .and_then(|(_, v)| v.first().copied())
    }

    pub(crate) fn member_has(&self, id: u32, member: u32, which: u32) -> bool {
        self.member_decorations
            .get(&(id, member))
            .is_some_and(|v| v.iter().any(|(d, _)| *d == which))
    }

    /// The struct type of the uniform block at descriptor set 0, binding 0.
    pub(crate) fn root_block(&self) -> Option<u32> {
        self.variables.iter().find_map(|&(id, ptr, storage)| {
            if storage != STORAGE_UNIFORM
                || self.decoration(id, decoration::DESCRIPTOR_SET) != Some(&[0])
                || self.decoration(id, decoration::BINDING) != Some(&[0])
            {
                return None;
            }
            match self.types.get(&ptr)? {
                Type::Pointer { pointee, .. } => Some(*pointee),
                _ => None,
            }
        })
    }

    /// Spec constant id to (scalar type id, literal words).
    pub(crate) fn spec_defaults(&self) -> HashMap<u32, (u32, Vec<u32>)> {
        self.spec_constants
            .iter()
            .filter_map(|(id, value)| {
                let spec_id = self.decoration(*id, decoration::SPEC_ID)?.first()?;
                Some((*spec_id, value.clone()))
            })
            .collect()
    }
}

/// Atomics with a result type; `OpAtomicStore` has none and is not checked.
fn is_atomic(o: u16) -> bool {
    (op::ATOMIC_LOAD..=op::ATOMIC_XOR).contains(&o) && o != op::ATOMIC_STORE
        || matches!(
            o,
            op::ATOMIC_FMIN_EXT | op::ATOMIC_FMAX_EXT | op::ATOMIC_FADD_EXT
        )
}

/// Instructions whose operands include literals that could be mistaken for ids.
fn takes_literals(o: u16) -> bool {
    matches!(
        o,
        op::NAME
            | op::MEMBER_NAME
            | op::EXTENSION
            | 11 // OpExtInstImport
            | 3 // OpSource
            | 7 // OpString
            | 8 // OpLine
            | op::MEMORY_MODEL
            | op::EXECUTION_MODE
            | op::CAPABILITY
            | op::TYPE_INT
            | op::TYPE_FLOAT
            | op::TYPE_VECTOR
            | op::TYPE_MATRIX
            | op::CONSTANT
            | op::SPEC_CONSTANT
            | op::DECORATE
            | op::MEMBER_DECORATE
    )
}

/// `stripped` with the strings of `full` that its instructions still name. Slang's
/// `-separate-debug-info` moves every `OpString` but the build identifier's to the sidecar,
/// so a stripped `printf` names a format the module no longer defines.
pub(crate) fn restore_strings(stripped: Vec<u32>, full: &[u32]) -> Vec<u32> {
    let defined: HashSet<u32> = instructions(&stripped)
        .filter(|(_, i)| i.op == op::STRING)
        .map(|(_, i)| i.args[0])
        .collect();
    let named: HashSet<u32> = instructions(&stripped)
        .filter(|(_, i)| i.op == op::EXT_INST)
        .flat_map(|(_, i)| i.args.get(4..).unwrap_or_default().iter().copied())
        .filter(|id| !defined.contains(id))
        .collect();
    let missing: Vec<u32> = instructions(full)
        .filter(|(_, i)| i.op == op::STRING && named.contains(&i.args[0]))
        .flat_map(|(at, i)| full[at..=at + i.args.len()].iter().copied())
        .collect();
    if missing.is_empty() {
        return stripped;
    }
    // Strings open the debug section, which follows the execution modes.
    let at = instructions(&stripped)
        .find(|(_, i)| {
            !matches!(
                i.op,
                op::CAPABILITY
                    | op::EXTENSION
                    | op::EXT_INST_IMPORT
                    | op::MEMORY_MODEL
                    | op::ENTRY_POINT
                    | op::EXECUTION_MODE
                    | op::EXECUTION_MODE_ID
            )
        })
        .map_or(stripped.len(), |(at, _)| at);
    let mut out = Vec::with_capacity(stripped.len() + missing.len());
    out.extend_from_slice(&stripped[..at]);
    out.extend(missing);
    out.extend_from_slice(&stripped[at..]);
    out
}

/// The module with `OpExecutionModeId <entry> OpacityMicromapIdKHR <true>` on every entry
/// point, plus its capability and extension. Where the module uses `orr.trace`'s
/// `OMM_2_STATE`, its value becomes the forced two-state ray flag and the micromap
/// capability is declared. `None` when the module has no ray query.
pub(crate) fn omm_variant(words: &[u32]) -> Option<Vec<u32>> {
    let module = Module::parse(words).ok()?;
    if !module.ray_query {
        return None;
    }
    let two_state = module
        .spec_constants
        .keys()
        .copied()
        .find(|id| module.decoration(*id, decoration::SPEC_ID) == Some(&[OMM_SPEC_ID]));
    let mut words = words.to_vec();
    if let Some(id) = two_state {
        let (at, _) =
            instructions(&words).find(|(_, i)| i.op == op::SPEC_CONSTANT && i.args[1] == id)?;
        words[at + 3] = RAY_FLAG_FORCE_OMM_2_STATE;
    }
    let words = &words[..];
    let mut bound = words[3];
    let mut new_id = || {
        let id = bound;
        bound += 1;
        id
    };
    let bool_ty = module
        .types
        .iter()
        .find(|(_, t)| **t == Type::Bool)
        .map(|(id, _)| *id);
    let (bool_ty, declare_bool) = match bool_ty {
        Some(id) => (id, false),
        None => (new_id(), true),
    };
    let true_id = new_id();

    let mut last_capability = None;
    let mut last_extension = None;
    let mut last_mode = None;
    let mut first_function = None;
    for (at, inst) in instructions(words) {
        let end = at + (words[at] >> 16) as usize;
        match inst.op {
            op::CAPABILITY => last_capability = Some(end),
            op::EXTENSION => last_extension = Some(end),
            op::ENTRY_POINT | op::EXECUTION_MODE | op::EXECUTION_MODE_ID => last_mode = Some(end),
            op::FUNCTION if first_function.is_none() => first_function = Some(at),
            _ => {}
        }
    }
    let after_capabilities = last_capability?;
    let after_extensions = last_extension.unwrap_or(after_capabilities);
    let after_modes = last_mode?;
    let before_functions = first_function.unwrap_or(words.len());

    let inst = |op: u16, args: &[u32]| {
        let mut v = vec![((args.len() as u32 + 1) << 16) | u32::from(op)];
        v.extend_from_slice(args);
        v
    };
    let mut capabilities = inst(op::CAPABILITY, &[CAPABILITY_OMM_EXECUTION_MODE]);
    if two_state.is_some() {
        capabilities.extend(inst(op::CAPABILITY, &[CAPABILITY_OMM]));
    }
    let mut inserts: Vec<(usize, Vec<u32>)> = vec![
        (after_capabilities, capabilities),
        (
            after_extensions,
            inst(op::EXTENSION, &encode_string(OMM_EXTENSION)),
        ),
        (
            after_modes,
            module
                .entries
                .iter()
                .flat_map(|&e| inst(op::EXECUTION_MODE_ID, &[e, EXECUTION_MODE_OMM_ID, true_id]))
                .collect(),
        ),
        (before_functions, {
            let mut v = Vec::new();
            if declare_bool {
                v.extend(inst(op::TYPE_BOOL, &[bool_ty]));
            }
            v.extend(inst(op::CONSTANT_TRUE, &[bool_ty, true_id]));
            v
        }),
    ];
    inserts.sort_by_key(|(at, _)| *at);

    let mut out = Vec::with_capacity(words.len() + 32);
    let mut copied = 0;
    for (at, words_in) in inserts {
        out.extend_from_slice(&words[copied..at]);
        out.extend(words_in);
        copied = at;
    }
    out.extend_from_slice(&words[copied..]);
    out[3] = bound;
    Some(out)
}
