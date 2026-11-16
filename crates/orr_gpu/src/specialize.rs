//! Folds a stage's specialization constants into its SPIR-V before the driver compiles it.
//!
//! The driver folds them too, but only after it has read the whole module, and a kernel made
//! for many feature masks is mostly code its mask rules out. [`specialize`] gives the constants
//! their values, follows them through the integer and boolean tests that read them, and removes
//! the blocks no path of the function reaches, with every other entry point's function, the
//! loads of local variables whose values nothing uses and the variables nothing left reads. A
//! variable the removed blocks also wrote, left one store that reaches its every load, gives
//! those loads the stored value, so a branch a mask rules out leaves no variable behind. It
//! changes nothing a reachable instruction computes.
//!
//! A test may reach its branch as a value or through a scalar local variable that only loads
//! and stores touch. A load of such a variable reads the store that reaches it, and where the
//! paths into a block bring different ones, what those of them that run agree on. Where no
//! store reaches a load on some path that runs, the pass knows nothing of what it reads.

use std::collections::{HashMap, HashSet};

use std::ffi::CStr;

use orr_shaders::abi::{SpecData, SpecEntry};

const HEADER_WORDS: usize = 5;
const BOUND: usize = 3;

const OP_UNDEF: u32 = 1;
const OP_SOURCE_CONTINUED: u32 = 2;
const OP_SOURCE: u32 = 3;
const OP_SOURCE_EXTENSION: u32 = 4;
const OP_NAME: u32 = 5;
const OP_MEMBER_NAME: u32 = 6;
const OP_STRING: u32 = 7;
const OP_LINE: u32 = 8;
const OP_EXTENSION: u32 = 10;
const OP_EXT_INST_IMPORT: u32 = 11;
const OP_MEMORY_MODEL: u32 = 14;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_CAPABILITY: u32 = 17;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_OPAQUE: u32 = 31;
const OP_TYPE_POINTER: u32 = 32;
const OP_TYPE_FORWARD_POINTER: u32 = 39;
const OP_CONSTANT_TRUE: u32 = 41;
const OP_CONSTANT_FALSE: u32 = 42;
const OP_CONSTANT: u32 = 43;
const OP_CONSTANT_NULL: u32 = 46;
const OP_SPEC_CONSTANT_TRUE: u32 = 48;
const OP_SPEC_CONSTANT_FALSE: u32 = 49;
const OP_SPEC_CONSTANT: u32 = 50;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;
const OP_FUNCTION_CALL: u32 = 57;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
const OP_PTR_ACCESS_CHAIN: u32 = 67;
const OP_VECTOR_EXTRACT_DYNAMIC: u32 = 77;
const OP_VECTOR_INSERT_DYNAMIC: u32 = 78;
const OP_VECTOR_SHUFFLE: u32 = 79;
const OP_COMPOSITE_CONSTRUCT: u32 = 80;
const OP_COMPOSITE_EXTRACT: u32 = 81;
const OP_COMPOSITE_INSERT: u32 = 82;
const OP_TRANSPOSE: u32 = 84;
const OP_SAMPLED_IMAGE: u32 = 86;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u32 = 87;
const OP_IMAGE_SAMPLE_DREF_IMPLICIT_LOD: u32 = 89;
const OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u32 = 90;
const OP_IMAGE_SAMPLE_PROJ_DREF_IMPLICIT_LOD: u32 = 93;
const OP_IMAGE_SAMPLE_PROJ_DREF_EXPLICIT_LOD: u32 = 94;
const OP_IMAGE_GATHER: u32 = 96;
const OP_IMAGE_DREF_GATHER: u32 = 97;
const OP_IMAGE_READ: u32 = 98;
const OP_IMAGE_WRITE: u32 = 99;
const OP_IMAGE: u32 = 100;
const OP_IMAGE_QUERY_SAMPLES: u32 = 107;
const OP_CONVERT_F_TO_U: u32 = 109;
const OP_S_NEGATE: u32 = 126;
const OP_S_MUL_EXTENDED: u32 = 152;
const OP_ANY: u32 = 154;
const OP_BIT_COUNT: u32 = 205;
const OP_ATOMIC_LOAD: u32 = 227;
const OP_ATOMIC_STORE: u32 = 228;
const OP_ATOMIC_EXCHANGE: u32 = 229;
const OP_ATOMIC_XOR: u32 = 242;
const OP_EXT_INST: u32 = 12;
/// GLSL.std.450's instructions that write through a pointer operand.
const GLSL_MODF: u32 = 35;
const GLSL_FREXP: u32 = 51;
const STORAGE_FUNCTION: u32 = 7;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const OP_COPY_OBJECT: u32 = 83;
const OP_BITCAST: u32 = 124;
const OP_LOGICAL_EQUAL: u32 = 164;
const OP_LOGICAL_NOT_EQUAL: u32 = 165;
const OP_LOGICAL_OR: u32 = 166;
const OP_LOGICAL_AND: u32 = 167;
const OP_LOGICAL_NOT: u32 = 168;
const OP_SELECT: u32 = 169;
const OP_I_EQUAL: u32 = 170;
const OP_S_LESS_THAN_EQUAL: u32 = 179;
const OP_SHIFT_RIGHT_LOGICAL: u32 = 194;
const OP_SHIFT_LEFT_LOGICAL: u32 = 196;
const OP_BITWISE_OR: u32 = 197;
const OP_BITWISE_XOR: u32 = 198;
const OP_BITWISE_AND: u32 = 199;
const OP_NOT: u32 = 200;
const OP_PHI: u32 = 245;
const OP_LOOP_MERGE: u32 = 246;
const OP_SELECTION_MERGE: u32 = 247;
const OP_LABEL: u32 = 248;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const OP_SWITCH: u32 = 251;
const OP_RETURN_VALUE: u32 = 254;
const OP_UNREACHABLE: u32 = 255;
const OP_LIFETIME_START: u32 = 256;
const OP_LIFETIME_STOP: u32 = 257;
const OP_NO_LINE: u32 = 317;
const OP_MODULE_PROCESSED: u32 = 330;
const OP_PTR_EQUAL: u32 = 401;
const OP_PTR_DIFF: u32 = 403;
const OP_TRACE_RAY: u32 = 4445;
const OP_EXECUTE_CALLABLE: u32 = 4446;
const OP_EXECUTION_MODE_ID: u32 = 331;
const OP_DECORATE_ID: u32 = 332;
const OP_TYPE_UNTYPED_POINTER: u32 = 4417;
const OP_DECORATE_STRING: u32 = 5632;
const OP_MEMBER_DECORATE_STRING: u32 = 5633;
const DECORATION_SPEC_ID: u32 = 1;

/// What a pass over the function knows of an id.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Value {
    /// Nothing reached it yet.
    Unreached,
    Constant(u32),
    Varying,
}

impl Value {
    /// What two ways of reaching one id leave known of it.
    fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unreached, v) | (v, Self::Unreached) => v,
            (Self::Constant(a), Self::Constant(b)) if a == b => self,
            _ => Self::Varying,
        }
    }
}

fn op(inst: &[u32]) -> u32 {
    inst[0] & 0xffff
}

/// The instructions of `code` from word `at` on, with the word each starts at; `None` where a
/// word count runs past the end.
fn instructions(code: &[u32], mut at: usize) -> impl Iterator<Item = Option<(usize, &[u32])>> {
    std::iter::from_fn(move || {
        if at >= code.len() {
            return None;
        }
        let count = (code[at] >> 16) as usize;
        let Some(inst) = code.get(at..at + count).filter(|_| count > 0) else {
            at = code.len();
            return Some(None);
        };
        let start = at;
        at += count;
        Some(Some((start, inst)))
    })
}

/// The bytes of the nul-terminated string that starts at `words`.
fn string(words: &[u32]) -> &[u8] {
    let bytes: &[u8] = bytemuck::cast_slice(words);
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    &bytes[..len]
}

/// Where a load of a local reads its value from.
#[derive(Clone, Copy)]
enum Definition {
    /// Not a load the pass follows.
    Unknown,
    /// No store reaches it: it reads an undefined value.
    None,
    /// The value a store wrote.
    Stored(u32),
    /// What the paths into a block bring, a [`Join`] of the function.
    Join(u32),
}

/// A local's value where a block's predecessors bring different definitions of it.
struct Join {
    /// Each predecessor's label and the definition at its end.
    sources: Vec<(u32, Definition)>,
    value: Value,
}

/// A block of a function: its label's id and its words, from the label to the next.
struct Block {
    id: u32,
    start: usize,
    end: usize,
    /// Where its last instruction starts.
    terminator: usize,
    /// Where its merge instruction starts.
    merge: Option<usize>,
}

/// The module's scalars and constants, and what the pass learns of each function's ids.
struct Folder<'a> {
    code: &'a [u32],
    /// Ids of the boolean and 32-bit integer types.
    scalar: Vec<bool>,
    /// Ids of the types of pointers to a function's variables of those types.
    scalar_pointer: Vec<bool>,
    /// Ids of the function's variables of those types that only loads and stores name.
    local: Vec<bool>,
    value: Vec<Value>,
    /// The definition each load of a local reads, by the load's id.
    reaching: Vec<Definition>,
    joins: Vec<Join>,
    /// Scratch for [`reaching_definitions`]: a local's place among its function's, by id.
    dense: Vec<u32>,
    /// Scratch for [`drop_unread_variables`].
    unread: Vec<bool>,
    /// Scratch for [`drop_unused_values`].
    pointer: Vec<bool>,
    uses: Vec<u32>,
    /// The id of the GLSL.std.450 instructions' import, `u32::MAX` where the module has none.
    glsl: u32,
    /// The ids a decoration names.
    decorated: Vec<bool>,
}

impl Folder<'_> {
    fn known(&self, id: u32) -> Value {
        self.value
            .get(id as usize)
            .copied()
            .unwrap_or(Value::Varying)
    }

    fn defined(&self, definition: Definition) -> Value {
        match definition {
            Definition::Unknown | Definition::None => Value::Varying,
            Definition::Stored(id) => self.known(id),
            Definition::Join(join) => self.joins[join as usize].value,
        }
    }

    /// What `inst`, an instruction with a result of a scalar type, computes; `None` for one
    /// the pass does not read, whose result varies.
    fn evaluate(&self, inst: &[u32], edges: &HashSet<(u32, u32)>, block: u32) -> Option<Value> {
        use Value::{Constant, Unreached, Varying};
        if inst.len() < 4 || !self.scalar.get(inst[1] as usize).copied().unwrap_or(false) {
            return None;
        }
        let a = self.known(inst[3]);
        let b = inst.get(4).map(|&id| self.known(id));
        let both = |f: fn(u32, u32) -> u32, absorbing: Option<u32>| {
            let b = b?;
            Some(match (a, b) {
                (Constant(x), Constant(y)) => Constant(f(x, y)),
                (Constant(x), _) | (_, Constant(x)) if Some(x) == absorbing => Constant(x),
                (Varying, _) | (_, Varying) => Varying,
                _ => Unreached,
            })
        };
        let one = |f: fn(u32) -> u32| {
            Some(match a {
                Constant(x) => Constant(f(x)),
                v => v,
            })
        };
        match op(inst) {
            OP_LOAD => self
                .local
                .get(inst[3] as usize)
                .copied()
                .unwrap_or(false)
                .then(|| self.defined(self.reaching[inst[2] as usize])),
            // A bit cast of anything but a 32-bit integer has an operand the pass holds no
            // value of.
            OP_COPY_OBJECT | OP_BITCAST => Some(a),
            OP_BITWISE_AND => both(|x, y| x & y, Some(0)),
            OP_BITWISE_OR => both(|x, y| x | y, Some(u32::MAX)),
            OP_BITWISE_XOR => both(|x, y| x ^ y, None),
            OP_NOT => one(|x| !x),
            OP_SHIFT_RIGHT_LOGICAL => match b? {
                Constant(y) if y >= 32 => Some(Varying),
                _ => both(|x, y| x >> y, None),
            },
            OP_SHIFT_LEFT_LOGICAL => match b? {
                Constant(y) if y >= 32 => Some(Varying),
                _ => both(|x, y| x << y, None),
            },
            OP_LOGICAL_AND => both(|x, y| u32::from(x != 0 && y != 0), Some(0)),
            OP_LOGICAL_OR => both(|x, y| u32::from(x != 0 || y != 0), Some(1)),
            OP_LOGICAL_EQUAL => both(|x, y| u32::from((x != 0) == (y != 0)), None),
            OP_LOGICAL_NOT_EQUAL => both(|x, y| u32::from((x != 0) != (y != 0)), None),
            OP_LOGICAL_NOT => one(|x| u32::from(x == 0)),
            OP_I_EQUAL..=OP_S_LESS_THAN_EQUAL => {
                // The operands are scalars where both are constants this pass holds.
                let compare: fn(u32, u32) -> u32 = match op(inst) - OP_I_EQUAL {
                    0 => |x, y| u32::from(x == y),
                    1 => |x, y| u32::from(x != y),
                    2 => |x, y| u32::from(x > y),
                    3 => |x, y| u32::from((x as i32) > (y as i32)),
                    4 => |x, y| u32::from(x >= y),
                    5 => |x, y| u32::from((x as i32) >= (y as i32)),
                    6 => |x, y| u32::from(x < y),
                    7 => |x, y| u32::from((x as i32) < (y as i32)),
                    8 => |x, y| u32::from(x <= y),
                    _ => |x, y| u32::from((x as i32) <= (y as i32)),
                };
                both(compare, None)
            }
            OP_SELECT => {
                let (x, y) = (b?, self.known(*inst.get(5)?));
                Some(match a {
                    Constant(c) => {
                        if c != 0 {
                            x
                        } else {
                            y
                        }
                    }
                    Unreached => Unreached,
                    Varying => match x.meet(y) {
                        Constant(_) if x == Unreached || y == Unreached => Unreached,
                        v => v,
                    },
                })
            }
            OP_PHI => Some(
                inst[3..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .filter(|pair| edges.contains(&(pair[1], block)))
                    .fold(Unreached, |v, pair| v.meet(self.known(pair[0]))),
            ),
            _ => None,
        }
    }
}

/// Whether `inst` is one of a module's names and decorations, whose first operand is the id
/// it describes.
fn annotation(inst: &[u32]) -> bool {
    matches!(
        op(inst),
        OP_NAME
            | OP_MEMBER_NAME
            | OP_DECORATE
            | OP_MEMBER_DECORATE
            | OP_DECORATE_ID
            | OP_DECORATE_STRING
            | OP_MEMBER_DECORATE_STRING
    )
}

/// Marks in `defined` the ids `inst` may define: a result is its first or second operand.
/// Literals in those places are left out where an instruction is known to hold one.
fn mark_defined(inst: &[u32], defined: &mut [bool]) {
    let places: &[usize] = match op(inst) {
        // An entry point's and an execution mode's function is defined where the function is.
        OP_SOURCE_CONTINUED | OP_SOURCE | OP_SOURCE_EXTENSION | OP_STRING | OP_LINE
        | OP_NO_LINE | OP_EXTENSION | OP_MEMORY_MODEL | OP_CAPABILITY | OP_MODULE_PROCESSED
        | OP_ENTRY_POINT | OP_EXECUTION_MODE | OP_EXECUTION_MODE_ID => &[],
        OP_EXT_INST_IMPORT
        | OP_TYPE_INT
        | OP_TYPE_FLOAT
        | OP_TYPE_OPAQUE
        | OP_TYPE_POINTER
        | OP_TYPE_FORWARD_POINTER
        | OP_TYPE_UNTYPED_POINTER
        | OP_SELECTION_MERGE
        | OP_LIFETIME_START
        | OP_LIFETIME_STOP => &[1],
        _ if annotation(inst) => return,
        _ => &[1, 2],
    };
    for &place in places {
        if let Some(slot) = inst.get(place).and_then(|&id| defined.get_mut(id as usize)) {
            *slot = true;
        }
    }
}

/// `code` with its entry point `entry` named `name` instead, and the debug name of the entry's
/// function with it: what a driver's tools show for the stage. `None` where the module has no
/// such entry point.
pub(crate) fn rename_entry(code: &[u32], entry: &[u8], name: &[u8]) -> Option<Vec<u32>> {
    let mut bytes = name.to_vec();
    bytes.resize((name.len() / 4 + 1) * 4, 0);
    let words = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c));
    // The instruction `inst` with the string from word `at` on replaced by the name.
    let renamed = |inst: &[u32], at: usize| {
        let old = string(&inst[at..]).len() / 4 + 1;
        let count = inst.len() - old + bytes.len() / 4;
        let mut out = vec![((count as u32) << 16) | op(inst)];
        out.extend_from_slice(&inst[1..at]);
        out.extend(words.clone());
        out.extend_from_slice(&inst[at + old..]);
        out
    };
    let mut function = None;
    let mut out = Vec::with_capacity(code.len() + bytes.len() / 2);
    out.extend_from_slice(code.get(..HEADER_WORDS)?);
    let mut rest = code.len();
    for item in instructions(code, HEADER_WORDS) {
        let (at, inst) = item?;
        match op(inst) {
            OP_FUNCTION => {
                rest = at;
                break;
            }
            OP_ENTRY_POINT if string(inst.get(3..)?) == entry => {
                function = inst.get(2).copied();
                out.extend(renamed(inst, 3));
            }
            OP_NAME if function.is_some() && inst.get(1).copied() == function => {
                out.extend(renamed(inst, 2));
            }
            _ => out.extend_from_slice(inst),
        }
    }
    out.extend_from_slice(&code[rest..]);
    function.map(|_| out)
}

/// How many instructions the functions of entry point `entry` of `code` hold once `spec` is
/// folded in, without debug lines: the size of the code a driver is handed for the stage.
pub fn handed_instructions(code: &[u32], entry: &CStr, spec: SpecData<'_>) -> u32 {
    let folded = specialize(code, entry.to_bytes(), spec.entries, spec.bytes);
    let code = folded.as_deref().unwrap_or(code);
    let mut inside = false;
    let mut count = 0;
    for (_, inst) in instructions(code, HEADER_WORDS).map_while(|i| i) {
        match op(inst) {
            OP_FUNCTION => inside = true,
            OP_FUNCTION_END => inside = false,
            OP_LINE | OP_NO_LINE => {}
            _ if inside => count += 1,
            _ => {}
        }
    }
    count
}

/// `code` with the constants `entries` name set to their values in `bytes`, the blocks of its
/// functions those values leave unreachable removed, and only entry point `entry` left.
/// `None` where there is nothing to fold or prune, or the module is not laid out as the pass
/// expects, so the caller keeps `code`.
pub(crate) fn specialize(
    code: &[u32],
    entry: &[u8],
    entries: &[SpecEntry],
    bytes: &[u8],
) -> Option<Vec<u32>> {
    let bound = *code.get(BOUND)? as usize;
    let given = |constant: u32| {
        let e = entries
            .iter()
            .find(|e| e.constant_id == constant && e.size == 4)?;
        let at = e.offset as usize;
        Some(u32::from_le_bytes(*bytes.get(at..)?.first_chunk()?))
    };

    // The module's declarations: its entry points, scalar types and constants.
    let mut spec_ids: HashMap<u32, u32> = HashMap::new();
    let mut scalar = vec![false; bound];
    let mut scalar_pointer = vec![false; bound];
    let mut value = vec![Value::Varying; bound];
    let mut frozen: HashMap<u32, u32> = HashMap::new();
    let mut kept_entry = None;
    let mut dropped: HashSet<u32> = HashSet::new();
    let mut functions_at = code.len();
    let mut glsl = u32::MAX;
    let mut decorated = vec![false; bound];
    for item in instructions(code, HEADER_WORDS) {
        let (at, inst) = item?;
        let id = |place: usize| inst.get(place).copied().filter(|&id| (id as usize) < bound);
        match op(inst) {
            OP_FUNCTION => {
                functions_at = at;
                break;
            }
            OP_EXT_INST_IMPORT if string(inst.get(2..)?) == b"GLSL.std.450" => glsl = id(1)?,
            OP_ENTRY_POINT => {
                let function = id(2)?;
                if string(inst.get(3..)?) == entry {
                    kept_entry = Some(function);
                } else {
                    dropped.insert(function);
                }
            }
            OP_DECORATE if inst.get(2) == Some(&DECORATION_SPEC_ID) => {
                spec_ids.insert(id(1)?, *inst.get(3)?);
            }
            OP_DECORATE | OP_DECORATE_ID | OP_DECORATE_STRING => decorated[id(1)? as usize] = true,
            OP_TYPE_BOOL => scalar[id(1)? as usize] = true,
            OP_TYPE_INT => scalar[id(1)? as usize] = inst.get(2) == Some(&32),
            OP_TYPE_POINTER => {
                scalar_pointer[id(1)? as usize] =
                    inst.get(2) == Some(&STORAGE_FUNCTION) && scalar[id(3)? as usize];
            }
            OP_CONSTANT_TRUE => value[id(2)? as usize] = Value::Constant(1),
            OP_CONSTANT_FALSE => value[id(2)? as usize] = Value::Constant(0),
            OP_CONSTANT | OP_CONSTANT_NULL if scalar[id(1)? as usize] => {
                let v = if op(inst) == OP_CONSTANT {
                    *inst.get(3)?
                } else {
                    0
                };
                value[id(2)? as usize] = Value::Constant(v);
            }
            OP_SPEC_CONSTANT_TRUE | OP_SPEC_CONSTANT_FALSE | OP_SPEC_CONSTANT => {
                let result = id(2)?;
                // A constant of another width keeps its place among the driver's.
                let fits = op(inst) != OP_SPEC_CONSTANT || scalar[id(1)? as usize];
                if let Some(v) = spec_ids
                    .get(&result)
                    .and_then(|&c| given(c))
                    .filter(|_| fits)
                {
                    let v = if op(inst) == OP_SPEC_CONSTANT {
                        v
                    } else {
                        u32::from(v != 0)
                    };
                    frozen.insert(result, v);
                    value[result as usize] = Value::Constant(v);
                }
            }
            _ => {}
        }
    }
    let kept_entry = kept_entry?;
    // Two entry points may name one function.
    dropped.remove(&kept_entry);
    if frozen.is_empty() && dropped.is_empty() {
        return None;
    }

    let mut folder = Folder {
        code,
        scalar,
        scalar_pointer,
        local: vec![false; bound],
        value,
        reaching: vec![Definition::Unknown; bound],
        joins: Vec::new(),
        dense: vec![u32::MAX; bound],
        unread: vec![false; bound],
        pointer: vec![false; bound],
        uses: vec![0; bound],
        glsl,
        decorated,
    };
    let mut body: Vec<u32> = Vec::with_capacity(code.len() - functions_at);
    // Undefined values a kept loop's removed continue block hands its header, by type.
    let mut undefined: HashMap<u32, u32> = HashMap::new();
    let mut next_id = bound as u32;

    let mut at = functions_at;
    while at < code.len() {
        // One function: from OpFunction to OpFunctionEnd.
        let mut end = at;
        let mut function = None;
        for item in instructions(code, at) {
            let (start, inst) = item?;
            if start == at {
                function = (op(inst) == OP_FUNCTION)
                    .then(|| inst.get(2).copied())
                    .flatten();
            }
            end = start + inst.len();
            if op(inst) == OP_FUNCTION_END {
                break;
            }
        }
        let function = function?;
        if !dropped.contains(&function) {
            fold_function(
                &mut folder,
                at,
                end,
                &mut body,
                &mut undefined,
                &mut next_id,
            )?;
        }
        at = end;
    }

    // What the kept instructions define decides which names and decorations stay.
    let mut defined = vec![false; next_id as usize];
    for item in instructions(code, HEADER_WORDS) {
        let (start, inst) = item?;
        if start >= functions_at {
            break;
        }
        mark_defined(inst, &mut defined);
    }
    for item in instructions(&body, 0) {
        mark_defined(item?.1, &mut defined);
    }

    let mut out = Vec::with_capacity(functions_at + body.len() + 3 * undefined.len());
    out.extend_from_slice(&code[..HEADER_WORDS]);
    out[BOUND] = next_id;
    for item in instructions(code, HEADER_WORDS) {
        let (start, inst) = item?;
        if start >= functions_at {
            break;
        }
        let target = inst.get(1).copied().unwrap_or(0);
        match op(inst) {
            OP_ENTRY_POINT if dropped.contains(&inst[2]) => {}
            OP_EXECUTION_MODE | OP_EXECUTION_MODE_ID if dropped.contains(&target) => {}
            OP_DECORATE
                if inst.get(2) == Some(&DECORATION_SPEC_ID) && frozen.contains_key(&target) => {}
            _ if annotation(inst) && !defined.get(target as usize).copied().unwrap_or(false) => {}
            OP_SPEC_CONSTANT_TRUE | OP_SPEC_CONSTANT_FALSE | OP_SPEC_CONSTANT
                if frozen.contains_key(&inst[2]) =>
            {
                let v = frozen[&inst[2]];
                let plain = match op(inst) {
                    OP_SPEC_CONSTANT => OP_CONSTANT,
                    _ if v != 0 => OP_CONSTANT_TRUE,
                    _ => OP_CONSTANT_FALSE,
                };
                out.push((inst[0] & 0xffff_0000) | plain);
                out.extend_from_slice(&inst[1..]);
                if plain == OP_CONSTANT {
                    let last = out.len() - 1;
                    out[last] = v;
                }
            }
            _ => out.extend_from_slice(inst),
        }
    }
    let mut undefined: Vec<(u32, u32)> = undefined.into_iter().collect();
    undefined.sort_unstable_by_key(|&(_, id)| id);
    for (ty, id) in undefined {
        out.extend_from_slice(&[(3 << 16) | OP_UNDEF, ty, id]);
    }
    out.extend_from_slice(&body);
    Some(out)
}

/// Appends to `out` the function in `code[at..end]` without the blocks its constants leave
/// unreachable.
fn fold_function(
    folder: &mut Folder<'_>,
    at: usize,
    end: usize,
    out: &mut Vec<u32>,
    undefined: &mut HashMap<u32, u32>,
    next_id: &mut u32,
) -> Option<()> {
    let code = folder.code;
    folder.joins.clear();
    // How many stores name each variable, as the function is written.
    let mut written: HashMap<u32, u32> = HashMap::new();
    // The scalar variables only loads and stores name. An id anywhere else leaves the variable
    // out.
    for item in instructions(&code[..end], at) {
        let (_, inst) = item?;
        if op(inst) == OP_STORE {
            *written.entry(*inst.get(1)?).or_default() += 1;
        }
        if op(inst) == OP_VARIABLE
            && inst.get(3) == Some(&STORAGE_FUNCTION)
            && folder
                .scalar_pointer
                .get(inst[1] as usize)
                .copied()
                .unwrap_or(false)
        {
            *folder.local.get_mut(inst[2] as usize)? = true;
        }
    }
    for item in instructions(&code[..end], at) {
        let (_, inst) = item?;
        for (place, &word) in inst.iter().enumerate().skip(1) {
            let named = match op(inst) {
                OP_VARIABLE => place == 2,
                OP_LOAD => place == 3,
                OP_STORE => place == 1,
                _ => false,
            };
            if !named
                && may_use(inst, place)
                && folder.local.get(word as usize).copied().unwrap_or(false)
            {
                folder.local[word as usize] = false;
            }
        }
    }
    // The blocks, and each instruction the pass reads marked as not reached yet.
    let mut blocks: Vec<Block> = Vec::new();
    let mut first_label = end;
    let mut function_end = end;
    let no_edges = HashSet::new();
    for item in instructions(&code[..end], at) {
        let (start, inst) = item?;
        match op(inst) {
            OP_LABEL => {
                if let Some(last) = blocks.last_mut() {
                    last.end = start;
                } else {
                    first_label = start;
                }
                blocks.push(Block {
                    id: *inst.get(1)?,
                    start,
                    end,
                    terminator: start,
                    merge: None,
                });
            }
            OP_FUNCTION_END => {
                function_end = start;
                if let Some(last) = blocks.last_mut() {
                    last.end = start;
                }
            }
            OP_LINE | OP_NO_LINE => {}
            _ => {
                let Some(block) = blocks.last_mut() else {
                    continue;
                };
                block.terminator = start;
                if matches!(op(inst), OP_LOOP_MERGE | OP_SELECTION_MERGE) {
                    block.merge = Some(start);
                }
                if inst.len() >= 4
                    && folder
                        .scalar
                        .get(inst[1] as usize)
                        .copied()
                        .unwrap_or(false)
                    && folder.evaluate(inst, &no_edges, 0).is_some()
                {
                    *folder.value.get_mut(inst[2] as usize)? = Value::Unreached;
                }
            }
        }
    }
    if blocks.is_empty() {
        out.extend_from_slice(&code[at..end]);
        return Some(());
    }
    let index: HashMap<u32, usize> = blocks.iter().enumerate().map(|(k, b)| (b.id, k)).collect();
    let inst_at = |start: usize| {
        let count = (code[start] >> 16) as usize;
        &code[start..start + count]
    };
    let loops = |b: &Block| b.merge.is_some_and(|m| op(inst_at(m)) == OP_LOOP_MERGE);
    let joins_at = reaching_definitions(folder, &blocks, &index)?;

    // Which blocks run and which values are constant, to a fixed point: a block runs once an
    // edge that runs reaches it, and a branch on a constant runs one edge.
    let mut runs = vec![false; blocks.len()];
    runs[0] = true;
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    loop {
        let mut changed = false;
        for k in 0..blocks.len() {
            if !runs[k] {
                continue;
            }
            let b = &blocks[k];
            for &join in &joins_at[k] {
                let Join { sources, value } = &folder.joins[join as usize];
                let met = sources
                    .iter()
                    .filter(|(from, _)| edges.contains(&(*from, b.id)))
                    .fold(*value, |v, &(_, definition)| {
                        v.meet(folder.defined(definition))
                    });
                if met != *value {
                    folder.joins[join as usize].value = met;
                    changed = true;
                }
            }
            for item in instructions(&code[..b.end], b.start) {
                let (_, inst) = item?;
                if inst.len() < 4 {
                    continue;
                }
                let Some(v) = folder.evaluate(inst, &edges, b.id) else {
                    continue;
                };
                let slot = folder.value.get_mut(inst[2] as usize)?;
                let met = slot.meet(v);
                if met != *slot {
                    *slot = met;
                    changed = true;
                }
            }
            let last = inst_at(b.terminator);
            let mut targets: Vec<u32> = Vec::new();
            match op(last) {
                OP_BRANCH => targets.push(*last.get(1)?),
                OP_BRANCH_CONDITIONAL => match folder.known(*last.get(1)?) {
                    Value::Constant(c) if !loops(b) => {
                        targets.push(*last.get(if c != 0 { 2 } else { 3 })?);
                    }
                    Value::Unreached if !loops(b) => {}
                    _ => targets.extend_from_slice(last.get(2..4)?),
                },
                OP_SWITCH => match folder.known(*last.get(1)?) {
                    Value::Constant(c) if !loops(b) => targets.push(switch_target(last, c)?),
                    Value::Unreached if !loops(b) => {}
                    _ => {
                        targets.push(*last.get(2)?);
                        targets
                            .extend(last.get(3..)?.as_chunks::<2>().0.iter().map(|pair| pair[1]));
                    }
                },
                _ => {}
            }
            for target in targets {
                if edges.insert((b.id, target)) {
                    changed = true;
                }
                let to = *index.get(&target)?;
                if !runs[to] {
                    runs[to] = true;
                    changed = true;
                }
            }
        }
        if changed {
            continue;
        }
        // A branch on a value nothing reached reads an undefined one: it stays as written.
        for (k, b) in blocks.iter().enumerate() {
            let last = inst_at(b.terminator);
            if runs[k] && matches!(op(last), OP_BRANCH_CONDITIONAL | OP_SWITCH) {
                let slot = folder.value.get_mut(*last.get(1)? as usize)?;
                if *slot == Value::Unreached {
                    *slot = Value::Varying;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // A block that runs keeps the merge and continue blocks its merge instruction names, as
    // stubs where they do not run.
    // A conditional branch on a constant becomes a branch and loses its merge instruction. A
    // switch keeps its own and only loses its other cases: blocks inside it may leave through
    // its merge block, which only a switch's or a loop's construct lets them do.
    let constant = |b: &Block, which: u32| {
        let last = inst_at(b.terminator);
        !loops(b) && op(last) == which && matches!(folder.known(last[1]), Value::Constant(_))
    };
    let folds = |b: &Block| constant(b, OP_BRANCH_CONDITIONAL);
    let mut merge_stub = vec![false; blocks.len()];
    let mut continue_stub: Vec<Option<u32>> = vec![None; blocks.len()];
    for (k, b) in blocks.iter().enumerate() {
        let Some(m) = b.merge.filter(|_| runs[k] && !folds(b)) else {
            continue;
        };
        let merge = inst_at(m);
        let to = *index.get(merge.get(1)?)?;
        merge_stub[to] = !runs[to];
        if op(merge) == OP_LOOP_MERGE {
            let to = *index.get(merge.get(2)?)?;
            if !runs[to] {
                continue_stub[to] = Some(b.id);
            }
        }
    }

    let function_at = out.len();
    out.extend_from_slice(&code[at..first_label]);
    for (k, b) in blocks.iter().enumerate() {
        if !runs[k] {
            if let Some(header) = continue_stub[k] {
                out.extend_from_slice(&[(2 << 16) | OP_LABEL, b.id, (2 << 16) | OP_BRANCH, header]);
            } else if merge_stub[k] {
                out.extend_from_slice(&[(2 << 16) | OP_LABEL, b.id, (1 << 16) | OP_UNREACHABLE]);
            }
            continue;
        }
        // The stub of this loop's continue block, a predecessor the block did not have.
        let stub = b
            .merge
            .map(inst_at)
            .filter(|m| op(m) == OP_LOOP_MERGE)
            .and_then(|m| index.get(&m[2]).filter(|&&to| !runs[to]).map(|_| m[2]));
        let folded = folds(b);
        for item in instructions(&code[..b.end], b.start) {
            let (start, inst) = item?;
            match op(inst) {
                OP_PHI => {
                    let first = out.len();
                    out.extend_from_slice(&inst[..3]);
                    for pair in inst[3..].as_chunks::<2>().0 {
                        if edges.contains(&(pair[1], b.id)) {
                            out.extend_from_slice(pair);
                        }
                    }
                    if let Some(stub) = stub {
                        let ty = inst[1];
                        let id = *undefined.entry(ty).or_insert_with(|| {
                            *next_id += 1;
                            *next_id - 1
                        });
                        out.extend_from_slice(&[id, stub]);
                    }
                    let count = (out.len() - first) as u32;
                    out[first] = (count << 16) | OP_PHI;
                }
                OP_SELECTION_MERGE if folded => {}
                _ if start == b.terminator && folded => {
                    let Value::Constant(c) = folder.known(inst[1]) else {
                        return None;
                    };
                    let target = *inst.get(if c != 0 { 2 } else { 3 })?;
                    out.extend_from_slice(&[(2 << 16) | OP_BRANCH, target]);
                }
                _ if start == b.terminator && constant(b, OP_SWITCH) => {
                    let Value::Constant(c) = folder.known(inst[1]) else {
                        return None;
                    };
                    out.extend_from_slice(&[
                        (3 << 16) | OP_SWITCH,
                        inst[1],
                        switch_target(inst, c)?,
                    ]);
                }
                _ => out.extend_from_slice(inst),
            }
        }
    }
    out.extend_from_slice(&code[function_end..end]);
    // A branch the pass folded leaves the values that fed it, the loads among them, and the
    // variables those loads read are left with stores alone. A variable its folded branches
    // also wrote may be left one store, whose value its loads read.
    loop {
        let before = out.len();
        drop_unused_values(
            out,
            function_at,
            folder.glsl,
            &mut folder.pointer,
            &mut folder.uses,
        )?;
        drop_unread_variables(out, function_at, &mut folder.unread)?;
        forward_single_stores(out, function_at, &written, &folder.decorated)?;
        if out.len() == before {
            break;
        }
    }
    merge_straight_blocks(out, function_at)
}

/// Merges into each block of the function in `out[from..]` that ends in a plain branch and
/// heads no construct the block it branches to, where nothing else branches to that block, no
/// construct names it as its merge or continue block, and it has no phi and heads no loop: the
/// chains of blocks a folded branch leaves become one block.
fn merge_straight_blocks(out: &mut Vec<u32>, from: usize) -> Option<()> {
    struct Span {
        label: u32,
        start: usize,
        end: usize,
        last: usize,
        header: bool,
        loops: bool,
        phi: bool,
    }
    let mut spans: Vec<Span> = Vec::new();
    let mut first_label = None;
    let mut function_end = out.len();
    let mut named: HashSet<u32> = HashSet::new();
    for item in instructions(out, from) {
        let (start, inst) = item?;
        match op(inst) {
            OP_LABEL => {
                match spans.last_mut() {
                    Some(last) => last.end = start,
                    None => first_label = Some(start),
                }
                spans.push(Span {
                    label: *inst.get(1)?,
                    start,
                    end: start,
                    last: start,
                    header: false,
                    loops: false,
                    phi: false,
                });
            }
            OP_FUNCTION_END => {
                function_end = start;
                if let Some(last) = spans.last_mut() {
                    last.end = start;
                }
            }
            OP_LINE | OP_NO_LINE => {}
            _ => {
                let Some(span) = spans.last_mut() else {
                    continue;
                };
                span.last = start;
                match op(inst) {
                    OP_PHI => span.phi = true,
                    OP_SELECTION_MERGE => {
                        span.header = true;
                        named.insert(*inst.get(1)?);
                    }
                    OP_LOOP_MERGE => {
                        span.header = true;
                        span.loops = true;
                        named.extend(inst.get(1..3)?);
                    }
                    _ => {}
                }
            }
        }
    }
    let first_label = first_label?;
    let inst_at = |start: usize| {
        let count = (out[start] >> 16) as usize;
        &out[start..start + count]
    };
    let index: HashMap<u32, usize> = spans
        .iter()
        .enumerate()
        .map(|(k, s)| (s.label, k))
        .collect();
    let mut predecessors: HashMap<u32, u32> = HashMap::new();
    for span in &spans {
        for target in targets(inst_at(span.last)) {
            *predecessors.entry(target).or_default() += 1;
        }
    }
    let mut next: Vec<Option<usize>> = vec![None; spans.len()];
    let mut merged = vec![false; spans.len()];
    for (k, span) in spans.iter().enumerate() {
        let last = inst_at(span.last);
        if op(last) != OP_BRANCH || span.header {
            continue;
        }
        let target = *last.get(1)?;
        let j = *index.get(&target)?;
        let t = &spans[j];
        if j != k
            && predecessors.get(&target) == Some(&1)
            && !named.contains(&target)
            && !t.phi
            && !t.loops
        {
            next[k] = Some(j);
            merged[j] = true;
        }
    }
    if !merged.contains(&true) {
        return Some(());
    }
    // A phi past a merged block names the block it merged into.
    let mut into: HashMap<u32, u32> = HashMap::new();
    let mut body: Vec<u32> = Vec::with_capacity(out.len() - from);
    body.extend_from_slice(&out[from..first_label]);
    for (k, span) in spans.iter().enumerate() {
        if merged[k] {
            continue;
        }
        let mut at = k;
        let mut begin = span.start;
        while let Some(j) = next[at] {
            body.extend_from_slice(&out[begin..spans[at].last]);
            into.insert(spans[j].label, span.label);
            begin = spans[j].start + (out[spans[j].start] >> 16) as usize;
            at = j;
        }
        body.extend_from_slice(&out[begin..spans[at].end]);
    }
    body.extend_from_slice(&out[function_end..]);
    let mut at = 0;
    while at < body.len() {
        let count = (body[at] >> 16) as usize;
        if body[at] & 0xffff == OP_PHI {
            for parent in (at + 4..at + count).step_by(2) {
                if let Some(&root) = into.get(&body[parent]) {
                    body[parent] = root;
                }
            }
        }
        at += count;
    }
    out.truncate(from);
    out.extend_from_slice(&body);
    Some(())
}

/// Whether `inst` computes a value and does nothing else, so it may go where nothing uses the
/// value: arithmetic, tests, conversions, composites, pointers, phis and the instructions of
/// GLSL.std.450 import `glsl` that write through no pointer. A load counts where it reads a local
/// variable or a place in one (`pointer`) and is not volatile.
fn pure(inst: &[u32], glsl: u32, pointer: &[bool]) -> bool {
    match op(inst) {
        OP_LOAD => {
            let volatile = inst.get(4).is_some_and(|&m| m & 1 != 0);
            let local = inst
                .get(3)
                .is_some_and(|&p| pointer.get(p as usize).copied().unwrap_or(false));
            local && !volatile
        }
        OP_ACCESS_CHAIN
        | OP_IN_BOUNDS_ACCESS_CHAIN
        | OP_PTR_ACCESS_CHAIN
        | OP_PHI
        | OP_VECTOR_EXTRACT_DYNAMIC..=OP_TRANSPOSE
        | OP_CONVERT_F_TO_U..=OP_BITCAST
        | OP_S_NEGATE..=OP_S_MUL_EXTENDED
        | OP_ANY..=OP_BIT_COUNT => true,
        OP_EXT_INST => {
            inst.get(3) == Some(&glsl) && !matches!(inst.get(4), Some(&(GLSL_MODF | GLSL_FREXP)))
        }
        _ => false,
    }
}

/// Removes from the function in `out[from..]` the values ([`pure`]) nothing uses, of GLSL.std.450
/// import `glsl`, and then those only they used. `pointer` and `uses` are clear for every id on
/// entry and on return.
fn drop_unused_values(
    out: &mut Vec<u32>,
    from: usize,
    glsl: u32,
    pointer: &mut [bool],
    uses: &mut [u32],
) -> Option<()> {
    let mut touched: Vec<u32> = Vec::new();
    let mut starts: Vec<usize> = Vec::new();
    for item in instructions(out, from) {
        let (start, inst) = item?;
        starts.push(start);
        let based = match op(inst) {
            OP_VARIABLE => inst.get(3) == Some(&STORAGE_FUNCTION),
            OP_ACCESS_CHAIN | OP_IN_BOUNDS_ACCESS_CHAIN | OP_PTR_ACCESS_CHAIN => pointer
                .get(*inst.get(3)? as usize)
                .copied()
                .unwrap_or(false),
            _ => false,
        };
        if based {
            *pointer.get_mut(inst[2] as usize)? = true;
            touched.push(inst[2]);
        }
        // A value names its own result alone.
        let skip = if pure(inst, glsl, pointer) { 3 } else { 1 };
        for (place, &word) in inst.iter().enumerate().skip(skip) {
            if !may_use(inst, place) {
                continue;
            }
            if let Some(count) = uses.get_mut(word as usize) {
                if *count == 0 {
                    touched.push(word);
                }
                *count += 1;
            }
        }
    }
    // Uses follow their values but for phis, so one pass from the end frees chains of values.
    let mut dropped = vec![false; starts.len()];
    for (k, &start) in starts.iter().enumerate().rev() {
        let count = (out[start] >> 16) as usize;
        let inst = &out[start..start + count];
        if inst.len() < 3
            || !pure(inst, glsl, pointer)
            || uses.get(inst[2] as usize).copied() != Some(0)
        {
            continue;
        }
        dropped[k] = true;
        for (place, &word) in inst.iter().enumerate().skip(3) {
            if !may_use(inst, place) {
                continue;
            }
            if let Some(count) = uses.get_mut(word as usize) {
                *count = count.saturating_sub(1);
            }
        }
    }
    let mut to = from;
    for (k, &start) in starts.iter().enumerate() {
        let count = (out[start] >> 16) as usize;
        if !dropped[k] {
            out.copy_within(start..start + count, to);
            to += count;
        }
    }
    out.truncate(to);
    for id in touched {
        pointer[id as usize] = false;
        uses[id as usize] = 0;
    }
    Some(())
}

/// Removes from the function in `out[from..]` its variables that only stores name, with
/// those stores. `unread` is false for every id on entry and on return.
fn drop_unread_variables(out: &mut Vec<u32>, from: usize, unread: &mut [bool]) -> Option<()> {
    let mut variables: Vec<u32> = Vec::new();
    for item in instructions(out, from) {
        let (_, inst) = item?;
        match op(inst) {
            OP_VARIABLE if inst.get(3) == Some(&STORAGE_FUNCTION) => {
                *unread.get_mut(inst[2] as usize)? = true;
                variables.push(inst[2]);
            }
            OP_VARIABLE => {}
            _ => {
                // A word that names a variable anywhere but as a store's pointer reads it.
                let skip = if op(inst) == OP_STORE { 2 } else { 1 };
                for (place, &word) in inst.iter().enumerate().skip(skip) {
                    if !may_use(inst, place) {
                        continue;
                    }
                    if let Some(slot) = unread.get_mut(word as usize) {
                        *slot = false;
                    }
                }
            }
        }
    }
    let mut to = from;
    let mut at = from;
    while at < out.len() {
        let count = (out[at] >> 16) as usize;
        let named = match out[at] & 0xffff {
            OP_VARIABLE => out[at + 2],
            OP_STORE => out[at + 1],
            _ => u32::MAX,
        };
        if !unread.get(named as usize).copied().unwrap_or(false) {
            out.copy_within(at..at + count, to);
            to += count;
        }
        at += count;
    }
    out.truncate(to);
    for variable in variables {
        unread[variable as usize] = false;
    }
    Some(())
}

/// The blocks a function's entry block reaches and their dominator tree, from each block's
/// successors by place, the entry block first.
struct Dominance {
    /// The blocks the entry block reaches, in reverse postorder.
    order: Vec<u32>,
    /// Each block's place in `order`, `u32::MAX` for one the entry block does not reach.
    number: Vec<u32>,
    predecessors: Vec<Vec<u32>>,
    /// Each reached block's immediate dominator, the entry block its own.
    dominator: Vec<u32>,
    depth: Vec<u32>,
    /// The dominator tree's preorder intervals, by which one block dominates another.
    enters: Vec<u32>,
    leaves: Vec<u32>,
}

impl Dominance {
    fn new(successors: &[Vec<u32>]) -> Self {
        const NONE: u32 = u32::MAX;
        let n = successors.len();
        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut number = vec![NONE; n];
        let mut seen = vec![false; n];
        let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
        seen[0] = true;
        while let Some((b, next)) = stack.last_mut() {
            if let Some(&to) = successors[*b as usize].get(*next) {
                *next += 1;
                if !seen[to as usize] {
                    seen[to as usize] = true;
                    stack.push((to, 0));
                }
            } else {
                order.push(*b);
                stack.pop();
            }
        }
        order.reverse();
        for (k, &b) in order.iter().enumerate() {
            number[b as usize] = k as u32;
        }
        let mut predecessors: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &b in &order {
            for &to in &successors[b as usize] {
                predecessors[to as usize].push(b);
            }
        }

        // Each block's immediate dominator (Cooper, Harvey and Kennedy).
        let mut dominator = vec![NONE; n];
        dominator[0] = 0;
        loop {
            let mut changed = false;
            for &b in order.iter().skip(1) {
                let mut new = NONE;
                for &p in &predecessors[b as usize] {
                    if dominator[p as usize] == NONE {
                        continue;
                    }
                    if new == NONE {
                        new = p;
                        continue;
                    }
                    let (mut x, mut y) = (p, new);
                    while x != y {
                        while number[x as usize] > number[y as usize] {
                            x = dominator[x as usize];
                        }
                        while number[y as usize] > number[x as usize] {
                            y = dominator[y as usize];
                        }
                    }
                    new = x;
                }
                if dominator[b as usize] != new {
                    dominator[b as usize] = new;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut depth = vec![0u32; n];
        let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &b in order.iter().skip(1) {
            // Reverse postorder visits a block after its dominator.
            depth[b as usize] = depth[dominator[b as usize] as usize] + 1;
            children[dominator[b as usize] as usize].push(b);
        }
        let mut enters = vec![0u32; n];
        let mut leaves = vec![0u32; n];
        let mut clock = 0;
        let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
        while let Some((b, next)) = stack.last_mut() {
            if *next == 0 {
                enters[*b as usize] = clock;
                clock += 1;
            }
            if let Some(&child) = children[*b as usize].get(*next) {
                *next += 1;
                stack.push((child, 0));
            } else {
                leaves[*b as usize] = clock;
                stack.pop();
            }
        }
        Self {
            order,
            number,
            predecessors,
            dominator,
            depth,
            enters,
            leaves,
        }
    }

    /// Whether block `x` dominates block `b`, both reached.
    fn dominates(&self, x: u32, b: u32) -> bool {
        let (x, b) = (x as usize, b as usize);
        self.enters[x] <= self.enters[b] && self.leaves[b] <= self.leaves[x]
    }

    fn reached(&self, b: u32) -> bool {
        self.number[b as usize] != u32::MAX
    }
}

/// Whether word `place` of `inst`, an instruction of a function, may name a value the
/// instruction reads: `Some(false)` for a literal, a label, the result or its type, `None`
/// for an instruction the pass does not know the operands of.
fn reads_value(inst: &[u32], place: usize) -> Option<bool> {
    // A memory access whose operands name a scope.
    let scoped = |mask: Option<&u32>| mask.is_some_and(|&m| m & 0x18 != 0);
    Some(match op(inst) {
        OP_LINE | OP_NO_LINE | OP_LABEL | OP_BRANCH | OP_LOOP_MERGE | OP_SELECTION_MERGE
        | OP_FUNCTION | OP_FUNCTION_END | OP_UNDEF | OP_UNREACHABLE => false,
        OP_VARIABLE => place == 4,
        OP_LOAD if scoped(inst.get(4)) => return None,
        OP_LOAD | OP_COMPOSITE_EXTRACT => place == 3,
        OP_STORE if scoped(inst.get(3)) => return None,
        OP_STORE => place <= 2,
        OP_COMPOSITE_INSERT | OP_VECTOR_SHUFFLE => place == 3 || place == 4,
        OP_EXT_INST => place == 3 || place >= 5,
        OP_BRANCH_CONDITIONAL | OP_SWITCH | OP_RETURN_VALUE => place == 1,
        OP_ATOMIC_STORE | OP_TRACE_RAY | OP_EXECUTE_CALLABLE => true,
        // The image operands' mask follows the coordinate, and a depth reference or a gather's
        // component where the instruction has one; the operands it names are ids.
        OP_IMAGE_SAMPLE_IMPLICIT_LOD..=OP_IMAGE_READ => {
            let mask = match op(inst) {
                OP_IMAGE_SAMPLE_DREF_IMPLICIT_LOD
                | OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD
                | OP_IMAGE_SAMPLE_PROJ_DREF_IMPLICIT_LOD
                | OP_IMAGE_SAMPLE_PROJ_DREF_EXPLICIT_LOD
                | OP_IMAGE_GATHER
                | OP_IMAGE_DREF_GATHER => 6,
                _ => 5,
            };
            place >= 3 && place != mask
        }
        OP_IMAGE_WRITE => place != 4,
        OP_FUNCTION_CALL
        | OP_ACCESS_CHAIN
        | OP_IN_BOUNDS_ACCESS_CHAIN
        | OP_PTR_ACCESS_CHAIN
        | OP_PHI
        | OP_COPY_OBJECT
        | OP_VECTOR_EXTRACT_DYNAMIC
        | OP_VECTOR_INSERT_DYNAMIC
        | OP_COMPOSITE_CONSTRUCT
        | OP_TRANSPOSE
        | OP_SAMPLED_IMAGE
        | OP_IMAGE..=OP_IMAGE_QUERY_SAMPLES
        | OP_CONVERT_F_TO_U..=OP_BITCAST
        | OP_S_NEGATE..=OP_S_MUL_EXTENDED
        | OP_ANY..=OP_BIT_COUNT
        | OP_ATOMIC_LOAD
        | OP_ATOMIC_EXCHANGE..=OP_ATOMIC_XOR
        | OP_PTR_EQUAL..=OP_PTR_DIFF => place >= 3,
        _ => return None,
    })
}

/// Whether word `place` of `inst`, an instruction of a function, may name an id the instruction
/// uses: every word but those [`reads_value`] knows for a literal, a label, the result or its
/// type. A literal is a number, whose value may equal any id's.
fn may_use(inst: &[u32], place: usize) -> bool {
    reads_value(inst, place) != Some(false)
}

/// Forwards to its loads the value of each variable of the function in `out[from..]` that the
/// pass left one store, where that store comes before every load on every path and only plain
/// loads and stores name the variable: the loads go, the values they gave name the stored one,
/// and the variable is left with its store alone. A variable `written` counts one store of, as
/// the function is written, keeps its loads: the pass takes out what its folded branches leave
/// behind, and leaves the rest of what a driver is handed as it was written. A load a
/// decoration names (`decorated`) keeps its variable.
fn forward_single_stores(
    out: &mut Vec<u32>,
    from: usize,
    written: &HashMap<u32, u32>,
    decorated: &[bool],
) -> Option<()> {
    // Each instruction's start and block, and each block's label and successors.
    let mut starts: Vec<usize> = Vec::new();
    let mut block_of: Vec<u32> = Vec::new();
    let mut labels: HashMap<u32, u32> = HashMap::new();
    let mut terminators: Vec<usize> = Vec::new();
    for item in instructions(out, from) {
        let (start, inst) = item?;
        if op(inst) == OP_LABEL {
            labels.insert(*inst.get(1)?, terminators.len() as u32);
            terminators.push(start);
        } else if let Some(last) = terminators.last_mut()
            && !matches!(op(inst), OP_LINE | OP_NO_LINE | OP_FUNCTION_END)
        {
            *last = start;
        }
        starts.push(start);
        block_of.push(terminators.len() as u32 - u32::from(!terminators.is_empty()));
    }
    if terminators.is_empty() {
        return Some(());
    }
    let inst_at = |start: usize| &out[start..start + (out[start] >> 16) as usize];

    struct Candidate {
        store: Option<(usize, u32)>,
        stores: u32,
        loads: Vec<usize>,
        kept: bool,
    }
    let mut candidates: HashMap<u32, Candidate> = HashMap::new();
    for &start in &starts {
        let inst = inst_at(start);
        if op(inst) == OP_VARIABLE
            && inst.len() == 4
            && inst[3] == STORAGE_FUNCTION
            && written.get(&inst[2]).is_some_and(|&n| n > 1)
        {
            candidates.insert(
                inst[2],
                Candidate {
                    store: None,
                    stores: 0,
                    loads: Vec::new(),
                    kept: false,
                },
            );
        }
    }
    if candidates.is_empty() {
        return Some(());
    }
    for (k, &start) in starts.iter().enumerate() {
        let inst = inst_at(start);
        match op(inst) {
            OP_VARIABLE | OP_LINE | OP_NO_LINE => continue,
            OP_STORE if inst.len() == 3 && candidates.contains_key(&inst[1]) => {
                let c = candidates.get_mut(&inst[1])?;
                c.stores += 1;
                c.store = Some((k, inst[2]));
                if let Some(c) = candidates.get_mut(&inst[2]) {
                    c.kept = true;
                }
                continue;
            }
            OP_LOAD if inst.len() == 4 && candidates.contains_key(&inst[3]) => {
                let c = candidates.get_mut(&inst[3])?;
                c.loads.push(k);
                c.kept |= decorated.get(inst[2] as usize).copied().unwrap_or(true);
                continue;
            }
            _ => {}
        }
        // A variable named anywhere else keeps its loads.
        for (place, &word) in inst.iter().enumerate().skip(1) {
            if !may_use(inst, place) {
                continue;
            }
            if let Some(c) = candidates.get_mut(&word) {
                c.kept = true;
            }
        }
    }

    // The store must dominate each load: come before it in its block, or in a block that
    // dominates the load's.
    let mut successors: Vec<Vec<u32>> = Vec::with_capacity(terminators.len());
    for &t in &terminators {
        successors.push(
            targets(inst_at(t))
                .map(|label| labels.get(&label).copied())
                .collect::<Option<Vec<u32>>>()?,
        );
    }
    let tree = Dominance::new(&successors);
    let mut value: HashMap<u32, u32> = HashMap::new();
    for c in candidates.values() {
        let Some((at, stored)) = c.store.filter(|_| !c.kept && c.stores == 1) else {
            continue;
        };
        let home = block_of[at];
        let dominated = c.loads.iter().all(|&load| {
            let b = block_of[load];
            if b == home {
                at < load
            } else {
                tree.reached(home) && tree.reached(b) && tree.dominates(home, b)
            }
        });
        if dominated {
            for &load in &c.loads {
                value.insert(inst_at(starts[load])[2], stored);
            }
        }
    }
    if value.is_empty() {
        return Some(());
    }

    // A forwarded value read where the pass does not know the operands keeps its variable.
    let mut kept: HashSet<u32> = HashSet::new();
    for &start in &starts {
        let inst = inst_at(start);
        for (place, word) in inst.iter().enumerate().skip(1) {
            if value.contains_key(word) && reads_value(inst, place).is_none() {
                kept.insert(*word);
            }
        }
    }
    if !kept.is_empty() {
        // The other loads of a variable one of whose loads stays keep theirs too.
        let stays: HashSet<u32> = candidates
            .iter()
            .filter(|(_, c)| {
                c.loads
                    .iter()
                    .any(|&load| kept.contains(&inst_at(starts[load])[2]))
            })
            .map(|(&v, _)| v)
            .collect();
        for v in stays {
            for &load in &candidates[&v].loads {
                value.remove(&inst_at(starts[load])[2]);
            }
        }
    }
    // A stored value that is itself a forwarded load's names that load's value.
    let resolve = |mut id: u32| {
        while let Some(&v) = value.get(&id) {
            id = v;
        }
        id
    };

    let mut rewritten: Vec<u32> = Vec::with_capacity(out.len() - from);
    for &start in &starts {
        let inst = inst_at(start);
        if op(inst) == OP_LOAD && inst.len() == 4 && value.contains_key(&inst[2]) {
            continue;
        }
        let first = rewritten.len();
        rewritten.extend_from_slice(inst);
        for place in 1..inst.len() {
            if reads_value(inst, place) == Some(true) {
                rewritten[first + place] = resolve(inst[place]);
            }
        }
    }
    out.truncate(from);
    out.extend_from_slice(&rewritten);
    Some(())
}

/// The labels `last`, a block's last instruction, may branch to.
fn targets(last: &[u32]) -> impl Iterator<Item = u32> + '_ {
    let (one, pairs): (&[u32], &[u32]) = match op(last) {
        OP_BRANCH => (last.get(1..2).unwrap_or(&[]), &[]),
        OP_BRANCH_CONDITIONAL => (last.get(2..4).unwrap_or(&[]), &[]),
        OP_SWITCH => (last.get(2..3).unwrap_or(&[]), last.get(3..).unwrap_or(&[])),
        _ => (&[], &[]),
    };
    one.iter()
        .copied()
        .chain(pairs.as_chunks::<2>().0.iter().map(|pair| pair[1]))
}

/// Finds the definition every load of a local of the function reads: fills `folder.reaching`
/// and `folder.joins`, and returns each block's joins.
///
/// A definition is the nearest store or join that dominates the load, as in SSA form, where
/// a variable's joins lie on the iterated dominance frontier of its stores. Only the joins a
/// load reads, directly or through another join, are made.
fn reaching_definitions(
    folder: &mut Folder<'_>,
    blocks: &[Block],
    index: &HashMap<u32, usize>,
) -> Option<Vec<Vec<u32>>> {
    const NONE: u32 = u32::MAX;
    let code = folder.code;
    let n = blocks.len();
    let mut joins_at: Vec<Vec<u32>> = vec![Vec::new(); n];

    // The blocks the entry block reaches, in reverse postorder, and each one's predecessors.
    let mut successors: Vec<Vec<u32>> = Vec::with_capacity(n);
    for b in blocks {
        let count = (code[b.terminator] >> 16) as usize;
        let last = code.get(b.terminator..b.terminator + count)?;
        let mut to = Vec::new();
        for target in targets(last) {
            to.push(*index.get(&target)? as u32);
        }
        successors.push(to);
    }
    let tree = Dominance::new(&successors);
    let Dominance {
        order,
        number,
        predecessors,
        dominator,
        depth,
        ..
    } = &tree;
    let dominates = |x: u32, b: u32| tree.dominates(x, b);
    let mut frontier: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &b in order {
        if predecessors[b as usize].len() < 2 {
            continue;
        }
        for &p in &predecessors[b as usize] {
            let mut runner = p;
            while runner != dominator[b as usize] {
                frontier[runner as usize].push(b);
                runner = dominator[runner as usize];
            }
        }
    }

    // Each local's stores, an initializer among them: the last of each block, in block order.
    let mut written: Vec<Vec<(u32, u32)>> = Vec::new();
    for (k, b) in blocks.iter().enumerate() {
        if number[k] == NONE {
            continue;
        }
        for item in instructions(&code[..b.end], b.start) {
            let (_, inst) = item?;
            let (variable, stored) = match op(inst) {
                OP_VARIABLE if inst.len() > 4 => (inst[2], inst[4]),
                OP_STORE if inst.len() >= 3 => (inst[1], inst[2]),
                _ => continue,
            };
            if !folder
                .local
                .get(variable as usize)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let v = &mut folder.dense[variable as usize];
            if *v == NONE {
                *v = written.len() as u32;
                written.push(Vec::new());
            }
            match written[*v as usize].last_mut() {
                Some(last) if last.0 == k as u32 => last.1 = stored,
                _ => written[*v as usize].push((k as u32, stored)),
            }
        }
    }
    // The blocks where a local's definitions may join: the iterated dominance frontier of
    // its stores.
    let mut may_join: Vec<Vec<u32>> = vec![Vec::new(); written.len()];
    let mut marked = vec![NONE; n];
    let mut work: Vec<u32> = Vec::new();
    for (v, written) in written.iter().enumerate() {
        work.extend(written.iter().map(|&(b, _)| b));
        while let Some(b) = work.pop() {
            for &f in &frontier[b as usize] {
                if marked[f as usize] != v as u32 {
                    marked[f as usize] = v as u32;
                    may_join[v].push(f);
                    work.push(f);
                }
            }
        }
    }

    // The definition of local `v` at the start of block `b`: the deepest of the joins that
    // dominate `b` and the stores of blocks that dominate it from above, a store over its
    // own block's join.
    let mut made: HashMap<(u32, u32), u32> = HashMap::new();
    let mut pending: Vec<(u32, u32, u32)> = Vec::new();
    let mut at_start =
        |v: u32, b: u32, joins: &mut Vec<Join>, pending: &mut Vec<(u32, u32, u32)>| {
            let mut best: Option<(u32, Result<u32, u32>)> = None;
            for &x in &may_join[v as usize] {
                if dominates(x, b) && best.is_none_or(|(d, _)| depth[x as usize] > d) {
                    best = Some((depth[x as usize], Err(x)));
                }
            }
            for &(x, stored) in &written[v as usize] {
                if x != b && dominates(x, b) && best.is_none_or(|(d, _)| depth[x as usize] >= d) {
                    best = Some((depth[x as usize], Ok(stored)));
                }
            }
            match best {
                None => Definition::None,
                Some((_, Ok(stored))) => Definition::Stored(stored),
                Some((_, Err(x))) => Definition::Join(*made.entry((v, x)).or_insert_with(|| {
                    joins.push(Join {
                        sources: Vec::new(),
                        value: Value::Unreached,
                    });
                    let join = joins.len() as u32 - 1;
                    pending.push((join, v, x));
                    join
                })),
            }
        };

    // A local's last store in the block at hand, where `stamp` names the block.
    let mut current = vec![(NONE, 0u32); written.len()];
    for (k, b) in blocks.iter().enumerate() {
        if number[k] == NONE {
            continue;
        }
        for item in instructions(&code[..b.end], b.start) {
            let (_, inst) = item?;
            let (variable, stored) = match op(inst) {
                OP_VARIABLE if inst.len() > 4 => (inst[2], Some(inst[4])),
                OP_STORE if inst.len() >= 3 => (inst[1], Some(inst[2])),
                OP_LOAD if inst.len() >= 4 => (inst[3], None),
                _ => continue,
            };
            let local = folder
                .local
                .get(variable as usize)
                .copied()
                .unwrap_or(false);
            let v = folder.dense.get(variable as usize).copied().unwrap_or(NONE);
            if !local {
                continue;
            }
            if let Some(stored) = stored {
                current[v as usize] = (k as u32, stored);
            } else {
                // A local nothing stores to has no definition anywhere.
                let definition = match current.get(v as usize) {
                    None => Definition::None,
                    Some(&(stamp, stored)) if stamp == k as u32 => Definition::Stored(stored),
                    Some(_) => at_start(v, k as u32, &mut folder.joins, &mut pending),
                };
                *folder.reaching.get_mut(inst[2] as usize)? = definition;
            }
        }
    }
    while let Some((join, v, b)) = pending.pop() {
        joins_at[b as usize].push(join);
        for &p in &predecessors[b as usize] {
            let stores = &written[v as usize];
            let definition = match stores.binary_search_by_key(&p, |&(x, _)| x) {
                Ok(found) => Definition::Stored(stores[found].1),
                Err(_) => at_start(v, p, &mut folder.joins, &mut pending),
            };
            folder.joins[join as usize]
                .sources
                .push((blocks[p as usize].id, definition));
        }
    }
    Some(joins_at)
}

/// The label an OpSwitch on a 32-bit selector of value `selector` branches to.
fn switch_target(inst: &[u32], selector: u32) -> Option<u32> {
    let cases = inst.get(3..)?;
    Some(
        cases
            .as_chunks::<2>()
            .0
            .iter()
            .find(|pair| pair[0] == selector)
            .map_or(*inst.get(2)?, |pair| pair[1]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const STORAGE_PRIVATE: u32 = 6;

    const OP_TYPE_VOID: u32 = 19;
    const OP_TYPE_FUNCTION: u32 = 33;
    const OP_RETURN: u32 = 253;
    const OP_I_NOT_EQUAL: u32 = 171;

    fn inst(op: u32, operands: &[u32]) -> Vec<u32> {
        let mut v = vec![((operands.len() as u32 + 1) << 16) | op];
        v.extend_from_slice(operands);
        v
    }

    fn text(s: &str) -> Vec<u32> {
        let mut b = s.as_bytes().to_vec();
        b.resize((b.len() / 4 + 1) * 4, 0);
        b.as_chunks()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect()
    }

    fn named(op: u32, head: &[u32], name: &str) -> Vec<u32> {
        let mut operands = head.to_vec();
        operands.extend(text(name));
        inst(op, &operands)
    }

    fn module(bound: u32, parts: &[Vec<u32>]) -> Vec<u32> {
        let mut code = vec![0x0723_0203, 0x0001_0600, 0, bound, 0];
        for p in parts {
            code.extend_from_slice(p);
        }
        code
    }

    const SPEC: [SpecEntry; 1] = [SpecEntry {
        constant_id: 7,
        offset: 0,
        size: 4,
    }];

    /// Ids: 1 void, 2 its function type, 3 bool, 4 uint, 5 the constant, 6 one, 7 zero.
    fn declarations(constant: Vec<u32>) -> Vec<Vec<u32>> {
        vec![
            inst(OP_TYPE_VOID, &[1]),
            inst(OP_TYPE_FUNCTION, &[2, 1]),
            inst(OP_TYPE_BOOL, &[3]),
            inst(OP_TYPE_INT, &[4, 32, 0]),
            constant,
            inst(OP_CONSTANT, &[4, 6, 1]),
            inst(OP_CONSTANT, &[4, 7, 0]),
        ]
    }

    /// Ids 40 to 43: a private bool and uint, whose stores keep the values a test looks for.
    fn sinks() -> Vec<Vec<u32>> {
        vec![
            inst(OP_TYPE_POINTER, &[40, STORAGE_PRIVATE, 3]),
            inst(OP_VARIABLE, &[40, 41, STORAGE_PRIVATE]),
            inst(OP_TYPE_POINTER, &[42, STORAGE_PRIVATE, 4]),
            inst(OP_VARIABLE, &[42, 43, STORAGE_PRIVATE]),
        ]
    }

    /// A branch on a test of the constant loses its dead side: the side's block and name go,
    /// the merge's phi keeps the live value, a loop whose body can no longer reach its
    /// continue block keeps that block as a stub with an undefined value for the header's
    /// phi, and the other entry point goes with its function.
    #[test]
    fn a_cleared_bit_removes_what_only_it_reaches() {
        let test = [
            inst(OP_BITWISE_AND, &[4, 11, 5, 6]),
            inst(OP_I_NOT_EQUAL, &[3, 12, 11, 7]),
        ];
        let mut parts = vec![
            named(OP_ENTRY_POINT, &[5, 8], "keep"),
            named(OP_ENTRY_POINT, &[5, 9], "drop"),
            inst(OP_EXECUTION_MODE, &[9, 17, 1, 1, 1]),
            named(OP_NAME, &[20], "gone"),
            named(OP_NAME, &[9], "drop"),
            named(OP_NAME, &[15], "kept"),
            inst(OP_DECORATE, &[5, DECORATION_SPEC_ID, 7]),
        ];
        parts.extend(declarations(inst(OP_SPEC_CONSTANT, &[4, 5, 1])));
        parts.extend(sinks());
        parts.extend([
            inst(OP_FUNCTION, &[1, 8, 0, 2]),
            inst(OP_LABEL, &[10]),
            test[0].clone(),
            test[1].clone(),
            inst(OP_SELECTION_MERGE, &[14, 0]),
            inst(OP_BRANCH_CONDITIONAL, &[12, 13, 14]),
            inst(OP_LABEL, &[13]),
            inst(OP_NOT, &[4, 20, 6]),
            inst(OP_BRANCH, &[14]),
            inst(OP_LABEL, &[14]),
            inst(OP_PHI, &[4, 15, 6, 10, 20, 13]),
            inst(OP_STORE, &[43, 15]),
            inst(OP_BRANCH, &[16]),
            inst(OP_LABEL, &[16]),
            inst(OP_PHI, &[4, 17, 7, 14, 21, 19]),
            inst(OP_LOOP_MERGE, &[18, 19, 0]),
            inst(OP_BRANCH, &[22]),
            inst(OP_LABEL, &[22]),
            inst(OP_STORE, &[43, 17]),
            inst(OP_BRANCH_CONDITIONAL, &[12, 19, 18]),
            inst(OP_LABEL, &[19]),
            inst(OP_NOT, &[4, 21, 17]),
            inst(OP_BRANCH, &[16]),
            inst(OP_LABEL, &[18]),
            inst(OP_RETURN, &[]),
            inst(OP_FUNCTION_END, &[]),
            inst(OP_FUNCTION, &[1, 9, 0, 2]),
            inst(OP_LABEL, &[23]),
            inst(OP_RETURN, &[]),
            inst(OP_FUNCTION_END, &[]),
        ]);
        let code = module(44, &parts);

        // The test's values feed folded branches alone, and go with them.
        let mut want = vec![
            named(OP_ENTRY_POINT, &[5, 8], "keep"),
            named(OP_NAME, &[15], "kept"),
        ];
        want.extend(declarations(inst(OP_CONSTANT, &[4, 5, 0])));
        want.extend(sinks());
        want.extend([
            inst(OP_UNDEF, &[4, 44]),
            inst(OP_FUNCTION, &[1, 8, 0, 2]),
            inst(OP_LABEL, &[10]),
            inst(OP_BRANCH, &[14]),
            inst(OP_LABEL, &[14]),
            inst(OP_PHI, &[4, 15, 6, 10]),
            inst(OP_STORE, &[43, 15]),
            inst(OP_BRANCH, &[16]),
            inst(OP_LABEL, &[16]),
            inst(OP_PHI, &[4, 17, 7, 14, 44, 19]),
            inst(OP_LOOP_MERGE, &[18, 19, 0]),
            inst(OP_BRANCH, &[22]),
            inst(OP_LABEL, &[22]),
            inst(OP_STORE, &[43, 17]),
            inst(OP_BRANCH, &[18]),
            inst(OP_LABEL, &[19]),
            inst(OP_BRANCH, &[16]),
            inst(OP_LABEL, &[18]),
            inst(OP_RETURN, &[]),
            inst(OP_FUNCTION_END, &[]),
        ]);
        assert_eq!(
            specialize(&code, b"keep", &SPEC, &0u32.to_le_bytes()),
            Some(module(45, &want))
        );
        // A constant the caller gives no value stays the driver's, and so does the branch.
        let other = [SpecEntry {
            constant_id: 8,
            ..SPEC[0]
        }];
        let kept = specialize(&code, b"keep", &other, &0u32.to_le_bytes()).unwrap();
        assert!(
            kept.windows(4)
                .any(|w| w == inst(OP_SPEC_CONSTANT, &[4, 5, 1]))
        );
        assert!(
            kept.windows(4)
                .any(|w| w == inst(OP_BRANCH_CONDITIONAL, &[12, 13, 14]))
        );
    }

    /// A test stored to a scalar local in both arms of a folded branch reaches the branch
    /// that loads it: only the store that runs counts, and once that branch folds nothing uses
    /// the load, which goes with the variable and its store. A variable something else names
    /// is not followed; a literal of its id's value, a switch's case, names nothing.
    #[test]
    fn a_test_reaches_its_branch_through_a_local() {
        let build = |escapes: bool, literal: bool| {
            let mut parts = vec![
                named(OP_ENTRY_POINT, &[5, 8], "keep"),
                inst(OP_DECORATE, &[5, DECORATION_SPEC_ID, 7]),
            ];
            parts.extend(declarations(inst(OP_SPEC_CONSTANT, &[4, 5, 1])));
            parts.extend([
                inst(OP_TYPE_POINTER, &[9, STORAGE_FUNCTION, 3]),
                inst(OP_CONSTANT_TRUE, &[3, 21]),
                inst(OP_CONSTANT_FALSE, &[3, 22]),
                inst(OP_UNDEF, &[4, 25]),
                inst(OP_FUNCTION, &[1, 8, 0, 2]),
                inst(OP_LABEL, &[10]),
                inst(OP_VARIABLE, &[9, 20, STORAGE_FUNCTION]),
                inst(OP_BITWISE_AND, &[4, 11, 5, 6]),
                inst(OP_I_NOT_EQUAL, &[3, 12, 11, 7]),
                inst(OP_SELECTION_MERGE, &[14, 0]),
                inst(OP_BRANCH_CONDITIONAL, &[12, 13, 15]),
                inst(OP_LABEL, &[13]),
                inst(OP_STORE, &[20, 21]),
                inst(OP_BRANCH, &[14]),
                inst(OP_LABEL, &[15]),
                inst(OP_STORE, &[20, 22]),
                inst(OP_BRANCH, &[14]),
                inst(OP_LABEL, &[14]),
            ]);
            if escapes {
                parts.push(inst(OP_COPY_OBJECT, &[9, 23, 20]));
                parts.push(inst(OP_STORE, &[23, 21]));
            }
            if literal {
                parts.extend([
                    inst(OP_SELECTION_MERGE, &[26, 0]),
                    inst(OP_SWITCH, &[25, 26, 20, 26]),
                    inst(OP_LABEL, &[26]),
                ]);
            }
            parts.extend([
                inst(OP_LOAD, &[3, 16, 20]),
                inst(OP_SELECTION_MERGE, &[18, 0]),
                inst(OP_BRANCH_CONDITIONAL, &[16, 17, 18]),
                inst(OP_LABEL, &[17]),
                inst(OP_BRANCH, &[18]),
                inst(OP_LABEL, &[18]),
                inst(OP_RETURN, &[]),
                inst(OP_FUNCTION_END, &[]),
            ]);
            module(27, &parts)
        };
        let folded = |escapes, literal| {
            specialize(
                &build(escapes, literal),
                b"keep",
                &SPEC,
                &0u32.to_le_bytes(),
            )
            .unwrap()
        };
        let has = |code: &[u32], i: Vec<u32>| code.windows(i.len()).any(|w| w == i);
        for literal in [false, true] {
            let through = folded(false, literal);
            assert!(!has(&through, inst(OP_LABEL, &[13])));
            assert!(!has(&through, inst(OP_LABEL, &[17])));
            assert!(!has(&through, inst(OP_BRANCH_CONDITIONAL, &[16, 17, 18])));
            assert!(!has(&through, inst(OP_LOAD, &[3, 16, 20])));
            assert!(!has(&through, inst(OP_STORE, &[20, 22])));
            assert!(!has(
                &through,
                inst(OP_VARIABLE, &[9, 20, STORAGE_FUNCTION])
            ));
        }
        let kept = folded(true, false);
        assert!(!has(&kept, inst(OP_LABEL, &[13])));
        assert!(has(&kept, inst(OP_BRANCH_CONDITIONAL, &[16, 17, 18])));
    }

    /// A variable a folded branch also wrote, left one store that comes before its load, gives
    /// the load that store's value and goes with them; one the function writes once keeps its
    /// load, as one whose store does not come before its load on every path does.
    #[test]
    fn a_folded_branch_leaves_no_variable_of_one_store() {
        let mut parts = vec![
            named(OP_ENTRY_POINT, &[5, 8], "keep"),
            inst(OP_DECORATE, &[5, DECORATION_SPEC_ID, 7]),
        ];
        parts.extend(declarations(inst(OP_SPEC_CONSTANT, &[4, 5, 1])));
        parts.extend(sinks());
        parts.extend([
            inst(OP_TYPE_POINTER, &[9, STORAGE_FUNCTION, 4]),
            // A test the pass knows nothing of.
            inst(OP_UNDEF, &[3, 31]),
            inst(OP_FUNCTION, &[1, 8, 0, 2]),
            inst(OP_LABEL, &[10]),
            inst(OP_VARIABLE, &[9, 20, STORAGE_FUNCTION]),
            inst(OP_VARIABLE, &[9, 21, STORAGE_FUNCTION]),
            inst(OP_VARIABLE, &[9, 22, STORAGE_FUNCTION]),
            inst(OP_BITWISE_AND, &[4, 11, 5, 6]),
            inst(OP_I_NOT_EQUAL, &[3, 12, 11, 7]),
            inst(OP_STORE, &[20, 6]),
            inst(OP_STORE, &[22, 6]),
            inst(OP_SELECTION_MERGE, &[16, 0]),
            inst(OP_BRANCH_CONDITIONAL, &[12, 13, 16]),
            inst(OP_LABEL, &[13]),
            inst(OP_STORE, &[20, 7]),
            inst(OP_BRANCH, &[16]),
            inst(OP_LABEL, &[16]),
            inst(OP_SELECTION_MERGE, &[14, 0]),
            inst(OP_BRANCH_CONDITIONAL, &[31, 17, 14]),
            inst(OP_LABEL, &[17]),
            inst(OP_STORE, &[21, 6]),
            inst(OP_BRANCH, &[14]),
            inst(OP_LABEL, &[14]),
            inst(OP_LOAD, &[4, 26, 20]),
            inst(OP_STORE, &[43, 26]),
            inst(OP_LOAD, &[4, 27, 21]),
            inst(OP_STORE, &[43, 27]),
            inst(OP_LOAD, &[4, 28, 22]),
            inst(OP_STORE, &[43, 28]),
            inst(OP_RETURN, &[]),
            inst(OP_FUNCTION_END, &[]),
        ]);
        let folded = specialize(&module(50, &parts), b"keep", &SPEC, &0u32.to_le_bytes()).unwrap();
        let has = |i: Vec<u32>| folded.windows(i.len()).any(|w| w == i);
        assert!(has(inst(OP_STORE, &[43, 6])));
        assert!(!has(inst(OP_LOAD, &[4, 26, 20])));
        assert!(!has(inst(OP_VARIABLE, &[9, 20, STORAGE_FUNCTION])));
        assert!(has(inst(OP_LOAD, &[4, 27, 21])));
        assert!(has(inst(OP_LOAD, &[4, 28, 22])));
    }

    /// A load reads the store that reaches it: of two stores in a row the nearer, and past a
    /// branch whose arms leave different values, neither. A variable nothing reads goes with
    /// its stores.
    #[test]
    fn a_load_reads_the_store_that_reaches_it() {
        let mut parts = vec![
            named(OP_ENTRY_POINT, &[5, 8], "keep"),
            inst(OP_DECORATE, &[5, DECORATION_SPEC_ID, 7]),
            inst(OP_DECORATE, &[30, DECORATION_SPEC_ID, 8]),
        ];
        parts.extend(declarations(inst(OP_SPEC_CONSTANT, &[4, 5, 1])));
        parts.extend(sinks());
        parts.extend([
            inst(OP_TYPE_POINTER, &[9, STORAGE_FUNCTION, 3]),
            inst(OP_CONSTANT_TRUE, &[3, 21]),
            inst(OP_CONSTANT_FALSE, &[3, 22]),
            // Given no value, so a test of it varies.
            inst(OP_SPEC_CONSTANT_TRUE, &[3, 30]),
            inst(OP_FUNCTION, &[1, 8, 0, 2]),
            inst(OP_LABEL, &[10]),
            inst(OP_VARIABLE, &[9, 20, STORAGE_FUNCTION]),
            inst(OP_VARIABLE, &[9, 28, STORAGE_FUNCTION]),
            inst(OP_STORE, &[28, 21]),
            inst(OP_STORE, &[20, 21]),
            inst(OP_LOAD, &[3, 16, 20]),
            inst(OP_SELECTION_MERGE, &[14, 0]),
            inst(OP_BRANCH_CONDITIONAL, &[16, 13, 14]),
            inst(OP_LABEL, &[13]),
            inst(OP_STORE, &[41, 21]),
            inst(OP_BRANCH, &[14]),
            inst(OP_LABEL, &[14]),
            inst(OP_STORE, &[20, 22]),
            inst(OP_LOAD, &[3, 19, 20]),
            inst(OP_SELECTION_MERGE, &[26, 0]),
            inst(OP_BRANCH_CONDITIONAL, &[30, 24, 25]),
            inst(OP_LABEL, &[24]),
            inst(OP_STORE, &[20, 21]),
            inst(OP_BRANCH, &[26]),
            inst(OP_LABEL, &[25]),
            inst(OP_SELECTION_MERGE, &[29, 0]),
            inst(OP_BRANCH_CONDITIONAL, &[19, 27, 29]),
            inst(OP_LABEL, &[27]),
            inst(OP_BRANCH, &[29]),
            inst(OP_LABEL, &[29]),
            inst(OP_BRANCH, &[26]),
            inst(OP_LABEL, &[26]),
            inst(OP_LOAD, &[3, 23, 20]),
            inst(OP_SELECTION_MERGE, &[18, 0]),
            inst(OP_BRANCH_CONDITIONAL, &[23, 17, 18]),
            inst(OP_LABEL, &[17]),
            inst(OP_BRANCH, &[18]),
            inst(OP_LABEL, &[18]),
            inst(OP_RETURN, &[]),
            inst(OP_FUNCTION_END, &[]),
        ]);
        let folded = specialize(&module(44, &parts), b"keep", &SPEC, &0u32.to_le_bytes()).unwrap();
        let has = |i: Vec<u32>| folded.windows(i.len()).any(|w| w == i);
        // The first load reads true, the second false.
        assert!(has(inst(OP_STORE, &[41, 21])));
        assert!(!has(inst(OP_BRANCH_CONDITIONAL, &[16, 13, 14])));
        assert!(!has(inst(OP_LABEL, &[27])));
        // The third reads true from one arm and false from the other.
        assert!(has(inst(OP_BRANCH_CONDITIONAL, &[23, 17, 18])));
        assert!(!has(inst(OP_VARIABLE, &[9, 28, STORAGE_FUNCTION])));
        assert!(!has(inst(OP_STORE, &[28, 21])));
        assert!(has(inst(OP_VARIABLE, &[9, 20, STORAGE_FUNCTION])));
    }

    /// The blocks a folded branch leaves in a row become one, and a phi past them names the
    /// block they merged into. A block with a phi keeps its own.
    #[test]
    fn a_folded_branch_leaves_one_block() {
        let mut parts = vec![
            named(OP_ENTRY_POINT, &[5, 8], "keep"),
            inst(OP_DECORATE, &[5, DECORATION_SPEC_ID, 7]),
        ];
        parts.extend(declarations(inst(OP_SPEC_CONSTANT, &[4, 5, 1])));
        parts.extend(sinks());
        parts.extend([
            inst(OP_FUNCTION, &[1, 8, 0, 2]),
            inst(OP_LABEL, &[10]),
            inst(OP_BITWISE_AND, &[4, 11, 5, 6]),
            inst(OP_I_NOT_EQUAL, &[3, 12, 11, 7]),
            inst(OP_SELECTION_MERGE, &[14, 0]),
            inst(OP_BRANCH_CONDITIONAL, &[12, 13, 15]),
            inst(OP_LABEL, &[13]),
            inst(OP_NOT, &[4, 20, 6]),
            inst(OP_BRANCH, &[14]),
            inst(OP_LABEL, &[15]),
            inst(OP_NOT, &[4, 25, 6]),
            inst(OP_BRANCH, &[16]),
            inst(OP_LABEL, &[16]),
            inst(OP_NOT, &[4, 26, 25]),
            inst(OP_BRANCH, &[14]),
            inst(OP_LABEL, &[14]),
            inst(OP_PHI, &[4, 27, 20, 13, 26, 16]),
            inst(OP_STORE, &[43, 27]),
            inst(OP_RETURN, &[]),
            inst(OP_FUNCTION_END, &[]),
        ]);
        let folded = specialize(&module(44, &parts), b"keep", &SPEC, &0u32.to_le_bytes()).unwrap();
        let has = |i: Vec<u32>| folded.windows(i.len()).any(|w| w == i);
        for label in [13, 15, 16] {
            assert!(!has(inst(OP_LABEL, &[label])));
        }
        // The test fed the folded branch alone.
        let row = [
            inst(OP_LABEL, &[10]),
            inst(OP_NOT, &[4, 25, 6]),
            inst(OP_NOT, &[4, 26, 25]),
            inst(OP_BRANCH, &[14]),
            inst(OP_LABEL, &[14]),
            inst(OP_PHI, &[4, 27, 26, 10]),
            inst(OP_STORE, &[43, 27]),
        ]
        .concat();
        assert!(has(row));
    }

    /// A switch on a constant keeps its merge instruction and its one live target: a block
    /// inside may leave through the merge block, as it may not through a selection's that is
    /// not its own.
    #[test]
    fn a_constant_switch_keeps_its_merge() {
        let mut parts = vec![
            named(OP_ENTRY_POINT, &[5, 8], "keep"),
            inst(OP_DECORATE, &[5, DECORATION_SPEC_ID, 7]),
        ];
        parts.extend(declarations(inst(OP_SPEC_CONSTANT, &[4, 5, 0])));
        let head = [
            inst(OP_FUNCTION, &[1, 8, 0, 2]),
            inst(OP_LABEL, &[10]),
            inst(OP_SELECTION_MERGE, &[13, 0]),
        ];
        let tail = [
            inst(OP_LABEL, &[12]),
            inst(OP_BRANCH, &[13]),
            inst(OP_LABEL, &[13]),
            inst(OP_RETURN, &[]),
            inst(OP_FUNCTION_END, &[]),
        ];
        let mut code_parts = parts.clone();
        code_parts.extend(head.iter().cloned());
        code_parts.extend([
            inst(OP_SWITCH, &[5, 11, 1, 12]),
            inst(OP_LABEL, &[11]),
            inst(OP_BRANCH, &[13]),
        ]);
        code_parts.extend(tail.iter().cloned());

        let mut want = vec![named(OP_ENTRY_POINT, &[5, 8], "keep")];
        want.extend(declarations(inst(OP_CONSTANT, &[4, 5, 1])));
        want.extend(head.iter().cloned());
        want.push(inst(OP_SWITCH, &[5, 12]));
        want.extend(tail.iter().cloned());
        assert_eq!(
            specialize(
                &module(14, &code_parts),
                b"keep",
                &SPEC,
                &1u32.to_le_bytes()
            ),
            Some(module(14, &want))
        );
    }
}
