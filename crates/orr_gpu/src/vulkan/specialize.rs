//! Folds a stage's specialization constants into its SPIR-V before the driver compiles it.
//!
//! The driver folds them too, but only after it has read the whole module, and a kernel made
//! for many feature masks is mostly code its mask rules out. [`specialize`] gives the constants
//! their values, follows them through the integer and boolean tests that read them, and removes
//! the blocks no path of the function reaches, with every other entry point's function and the
//! local variables nothing left reads. It changes nothing a reachable instruction computes.
//!
//! A test may reach its branch as a value or through a scalar local variable that only loads
//! and stores touch. A load of such a variable reads the store that reaches it, and where the
//! paths into a block bring different ones, what those of them that run agree on. Where no
//! store reaches a load on some path that runs, the pass knows nothing of what it reads.

use std::collections::{HashMap, HashSet};

use orr_shaders::abi::SpecEntry;

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
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
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
const OP_UNREACHABLE: u32 = 255;
const OP_LIFETIME_START: u32 = 256;
const OP_LIFETIME_STOP: u32 = 257;
const OP_NO_LINE: u32 = 317;
const OP_MODULE_PROCESSED: u32 = 330;
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

/// `code` with the constants `entries` name set to their values in `bytes`, the blocks of its
/// functions those values leave unreachable removed, and only entry point `entry` left.
/// `None` where `entries` names no constant of the module or the module is not laid out as
/// the pass expects, so the caller keeps `code`.
pub(super) fn specialize(
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
        Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
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
    for item in instructions(code, HEADER_WORDS) {
        let (at, inst) = item?;
        let id = |place: usize| inst.get(place).copied().filter(|&id| (id as usize) < bound);
        match op(inst) {
            OP_FUNCTION => {
                functions_at = at;
                break;
            }
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
    // The scalar variables only loads and stores name. An id anywhere else, a literal of its
    // value among them, leaves the variable out.
    for item in instructions(&code[..end], at) {
        let (_, inst) = item?;
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
            if !named && folder.local.get(word as usize).copied().unwrap_or(false) {
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
    drop_unread_variables(out, function_at, &mut folder.unread)
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
            // Their literals are not ids.
            OP_LINE | OP_NO_LINE | OP_VARIABLE => {}
            _ => {
                // A word that names a variable anywhere but as a store's pointer reads it. A
                // literal with a variable's id only keeps that variable.
                let skip = if op(inst) == OP_STORE { 2 } else { 1 };
                for &word in inst.iter().skip(skip) {
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

    // Each block's immediate dominator (Cooper, Harvey and Kennedy), then the dominator
    // tree's depths and its preorder intervals, by which one block dominates another.
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
    let dominates = |x: u32, b: u32| {
        enters[x as usize] <= enters[b as usize] && leaves[b as usize] <= leaves[x as usize]
    };
    let mut frontier: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &b in &order {
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
        b.chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
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
            named(OP_NAME, &[12], "kept"),
            inst(OP_DECORATE, &[5, DECORATION_SPEC_ID, 7]),
        ];
        parts.extend(declarations(inst(OP_SPEC_CONSTANT, &[4, 5, 1])));
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
            inst(OP_BRANCH, &[16]),
            inst(OP_LABEL, &[16]),
            inst(OP_PHI, &[4, 17, 7, 14, 21, 19]),
            inst(OP_LOOP_MERGE, &[18, 19, 0]),
            inst(OP_BRANCH, &[22]),
            inst(OP_LABEL, &[22]),
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
        let code = module(24, &parts);

        let mut want = vec![
            named(OP_ENTRY_POINT, &[5, 8], "keep"),
            named(OP_NAME, &[12], "kept"),
        ];
        want.extend(declarations(inst(OP_CONSTANT, &[4, 5, 0])));
        want.extend([
            inst(OP_UNDEF, &[4, 24]),
            inst(OP_FUNCTION, &[1, 8, 0, 2]),
            inst(OP_LABEL, &[10]),
            test[0].clone(),
            test[1].clone(),
            inst(OP_BRANCH, &[14]),
            inst(OP_LABEL, &[14]),
            inst(OP_PHI, &[4, 15, 6, 10]),
            inst(OP_BRANCH, &[16]),
            inst(OP_LABEL, &[16]),
            inst(OP_PHI, &[4, 17, 7, 14, 24, 19]),
            inst(OP_LOOP_MERGE, &[18, 19, 0]),
            inst(OP_BRANCH, &[22]),
            inst(OP_LABEL, &[22]),
            inst(OP_BRANCH, &[18]),
            inst(OP_LABEL, &[19]),
            inst(OP_BRANCH, &[16]),
            inst(OP_LABEL, &[18]),
            inst(OP_RETURN, &[]),
            inst(OP_FUNCTION_END, &[]),
        ]);
        assert_eq!(
            specialize(&code, b"keep", &SPEC, &0u32.to_le_bytes()),
            Some(module(25, &want))
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
    /// that loads it: only the store that runs counts. A variable something else names is not
    /// followed.
    #[test]
    fn a_test_reaches_its_branch_through_a_local() {
        let build = |escapes: bool| {
            let mut parts = vec![
                named(OP_ENTRY_POINT, &[5, 8], "keep"),
                inst(OP_DECORATE, &[5, DECORATION_SPEC_ID, 7]),
            ];
            parts.extend(declarations(inst(OP_SPEC_CONSTANT, &[4, 5, 1])));
            parts.extend([
                inst(OP_TYPE_POINTER, &[9, STORAGE_FUNCTION, 3]),
                inst(OP_CONSTANT_TRUE, &[3, 21]),
                inst(OP_CONSTANT_FALSE, &[3, 22]),
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
            module(24, &parts)
        };
        let folded =
            |escapes| specialize(&build(escapes), b"keep", &SPEC, &0u32.to_le_bytes()).unwrap();
        let has = |code: &[u32], i: Vec<u32>| code.windows(i.len()).any(|w| w == i);
        let through = folded(false);
        assert!(!has(&through, inst(OP_LABEL, &[13])));
        assert!(!has(&through, inst(OP_LABEL, &[17])));
        assert!(has(&through, inst(OP_STORE, &[20, 22])));
        assert!(!has(&through, inst(OP_BRANCH_CONDITIONAL, &[16, 17, 18])));
        let kept = folded(true);
        assert!(!has(&kept, inst(OP_LABEL, &[13])));
        assert!(has(&kept, inst(OP_BRANCH_CONDITIONAL, &[16, 17, 18])));
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
        let folded = specialize(&module(31, &parts), b"keep", &SPEC, &0u32.to_le_bytes()).unwrap();
        let has = |i: Vec<u32>| folded.windows(i.len()).any(|w| w == i);
        // The first load reads true, the second false.
        assert!(has(inst(OP_LABEL, &[13])));
        assert!(!has(inst(OP_BRANCH_CONDITIONAL, &[16, 13, 14])));
        assert!(!has(inst(OP_LABEL, &[27])));
        // The third reads true from one arm and false from the other.
        assert!(has(inst(OP_BRANCH_CONDITIONAL, &[23, 17, 18])));
        assert!(!has(inst(OP_VARIABLE, &[9, 28, STORAGE_FUNCTION])));
        assert!(!has(inst(OP_STORE, &[28, 21])));
        assert!(has(inst(OP_VARIABLE, &[9, 20, STORAGE_FUNCTION])));
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
