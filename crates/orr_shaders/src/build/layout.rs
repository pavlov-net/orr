//! The `#[repr(C)]` layout of the Rust the build generates, checked against the SPIR-V.

use std::collections::{BTreeMap, HashMap};

use super::model::{StructDef, Ty};

/// One member of a generated struct, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    Field(usize),
    /// Explicit padding bytes where the SPIR-V leaves a gap.
    Pad(u32),
}

#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub(crate) slots: Vec<Slot>,
    pub(crate) size: u32,
    pub(crate) align: u32,
}

impl Plan {
    /// Gaps the SPIR-V leaves: (field after the gap or `None` at the end, bytes).
    pub(crate) fn gaps<'a>(&self, def: &'a StructDef) -> Vec<(Option<&'a str>, u32)> {
        let mut out = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            if let Slot::Pad(n) = *slot {
                let next = self.slots[i + 1..].iter().find_map(|s| match s {
                    Slot::Field(f) => Some(def.fields[*f].name.as_str()),
                    Slot::Pad(_) => None,
                });
                out.push((next, n));
            }
        }
        out
    }
}

/// A place where the Rust layout cannot match the SPIR-V.
#[derive(Debug)]
pub(crate) struct Mismatch {
    pub(crate) structure: String,
    pub(crate) field: String,
    pub(crate) message: String,
}

pub(crate) struct Planner<'a> {
    structs: &'a BTreeMap<String, StructDef>,
    pub(crate) plans: HashMap<String, Plan>,
}

impl<'a> Planner<'a> {
    pub(crate) fn new(structs: &'a BTreeMap<String, StructDef>) -> Self {
        Self {
            structs,
            plans: HashMap::new(),
        }
    }

    /// Plans every struct; the first mismatch fails.
    pub(crate) fn plan_all(mut self) -> Result<HashMap<String, Plan>, Mismatch> {
        for name in self.structs.keys() {
            self.plan(name)?;
        }
        Ok(self.plans)
    }

    fn plan(&mut self, name: &str) -> Result<(u32, u32), Mismatch> {
        if let Some(p) = self.plans.get(name) {
            return Ok((p.size, p.align));
        }
        let def = &self.structs[name];
        let mut slots = Vec::new();
        let mut cursor = 0u32;
        let mut align = 1u32;
        for (i, field) in def.fields.iter().enumerate() {
            let mismatch = |message: String| Mismatch {
                structure: def.name.clone(),
                field: field.name.clone(),
                message,
            };
            let (size, field_align) = self.layout(&field.ty).map_err(&mismatch)?;
            align = align.max(field_align);
            if field.offset < cursor {
                return Err(mismatch(format!(
                    "SPIR-V offset {} overlaps the previous field, which ends at {cursor}",
                    field.offset
                )));
            }
            if field.offset % field_align != 0 {
                return Err(mismatch(format!(
                    "SPIR-V offset {} is not a multiple of the Rust alignment {field_align}",
                    field.offset
                )));
            }
            if field.offset > cursor {
                slots.push(Slot::Pad(field.offset - cursor));
            }
            slots.push(Slot::Field(i));
            cursor = field.offset + size;
        }
        let natural = cursor.next_multiple_of(align);
        let size = match def.size {
            Some(s) if s < cursor => {
                return Err(Mismatch {
                    structure: def.name.clone(),
                    field: def
                        .fields
                        .last()
                        .map(|f| f.name.clone())
                        .unwrap_or_default(),
                    message: format!(
                        "SPIR-V size {s} ends inside the fields, which end at {cursor}"
                    ),
                });
            }
            Some(s) if s % align != 0 => {
                return Err(Mismatch {
                    structure: def.name.clone(),
                    field: String::new(),
                    message: format!(
                        "SPIR-V size {s} is not a multiple of the Rust alignment {align}"
                    ),
                });
            }
            Some(s) => s,
            None => natural,
        };
        if size > cursor {
            slots.push(Slot::Pad(size - cursor));
        }
        self.plans
            .insert(name.to_owned(), Plan { slots, size, align });
        Ok((size, align))
    }

    /// (size, alignment) of the Rust type generated for `ty`.
    fn layout(&mut self, ty: &Ty) -> Result<(u32, u32), String> {
        Ok(match ty {
            Ty::Scalar(s) => (s.size(), s.size()),
            Ty::Vector(s, n) => (s.size() * n, s.size()),
            Ty::Matrix {
                scalar,
                rows,
                cols,
                stride,
                column_major,
            } => {
                if *column_major {
                    return Err(format!(
                        "column-major {rows}x{cols} matrix; generated Rust is row-major ([[_; {cols}]; {rows}])"
                    ));
                }
                if *stride != cols * scalar.size() {
                    return Err(format!(
                        "SPIR-V row stride {stride}, generated Rust row stride {}",
                        cols * scalar.size()
                    ));
                }
                (rows * cols * scalar.size(), scalar.size())
            }
            Ty::Array {
                element,
                len,
                stride,
            } => {
                let (size, align) = self.layout(element)?;
                if *stride != size {
                    return Err(format!(
                        "SPIR-V array stride {stride}, generated Rust element size {size}"
                    ));
                }
                (size * len, align)
            }
            Ty::Struct(name) => self
                .plan(name)
                .map_err(|m| format!("in {}.{}: {}", m.structure, m.field, m.message))?,
            Ty::Pointer(_) => (8, 8),
            Ty::Handle(h) => (h.scalar().size(), h.scalar().size()),
        })
    }
}
