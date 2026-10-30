//! SPIR-V rewrites applied when a stage is created.

use std::borrow::Cow;

const OP_CAPABILITY: u32 = 17;
const CAPABILITY_ABORT: u32 = 5120;
const OP_ENTRY_POINT: u32 = 15;
const OP_VARIABLE: u32 = 59;
const STORAGE_TASK_PAYLOAD: u32 = 5402;
const HEADER_WORDS: usize = 5;

/// Instruction word ranges of `code`, after the header.
fn instructions(code: &[u32]) -> impl Iterator<Item = (usize, &[u32])> {
    let mut at = HEADER_WORDS;
    std::iter::from_fn(move || {
        let count = (*code.get(at)? >> 16) as usize;
        let inst = code.get(at..at + count.max(1))?;
        let start = at;
        at += count.max(1);
        Some((start, inst))
    })
}

/// Whether the module declares `AbortKHR`, which needs `shaderAbort`.
pub(super) fn aborts(code: &[u32]) -> bool {
    instructions(code)
        .take_while(|(_, inst)| inst[0] & 0xffff == OP_CAPABILITY)
        .any(|(_, inst)| inst.get(1) == Some(&CAPABILITY_ABORT))
}

/// The interface ids of entry point `name`.
fn interface<'a>(code: &'a [u32], name: &[u8]) -> Option<&'a [u32]> {
    instructions(code).find_map(|(_, inst)| {
        if inst[0] & 0xffff != OP_ENTRY_POINT || inst.len() < 3 {
            return None;
        }
        let text: Vec<u8> = inst[3..].iter().flat_map(|w| w.to_le_bytes()).collect();
        let len = text.iter().position(|&b| b == 0)?;
        let words = len / 4 + 1;
        (text[..len] == *name).then(|| &inst[3 + words..])
    })
}

/// Makes `entry`'s task payload variable the first one the module declares.
///
/// One module holds every entry of a kernel file, so a task entry and a mesh entry each
/// declare a `TaskPayloadWorkgroupEXT` variable. On 621.01 a stage reads and writes its
/// payload as if the module's first such variable were its own, so a mesh stage whose
/// variable comes second reads memory the task stage never wrote.
pub(super) fn payload_first<'a>(code: &'a [u32], entry: &[u8]) -> Cow<'a, [u32]> {
    let Some(interface) = interface(code, entry) else {
        return Cow::Borrowed(code);
    };
    let payloads: Vec<(usize, &[u32])> = instructions(code)
        .filter(|(_, i)| i[0] & 0xffff == OP_VARIABLE && i.len() >= 4)
        .filter(|(_, i)| i[3] == STORAGE_TASK_PAYLOAD)
        .collect();
    let Some(&(own_at, own)) = payloads.iter().find(|(_, i)| interface.contains(&i[2])) else {
        return Cow::Borrowed(code);
    };
    let first_at = payloads[0].0;
    if own_at == first_at {
        return Cow::Borrowed(code);
    }
    let mut out = Vec::with_capacity(code.len());
    out.extend_from_slice(&code[..first_at]);
    out.extend_from_slice(own);
    out.extend_from_slice(&code[first_at..own_at]);
    out.extend_from_slice(&code[own_at + own.len()..]);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(op: u32, operands: &[u32]) -> Vec<u32> {
        let mut v = vec![((operands.len() as u32 + 1) << 16) | op];
        v.extend_from_slice(operands);
        v
    }

    fn name(s: &str) -> Vec<u32> {
        let mut b = s.as_bytes().to_vec();
        b.resize((b.len() / 4 + 1) * 4, 0);
        b.chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn mesh_payload_moves_ahead_of_task_payload() {
        let mut code = vec![0x0723_0203, 0x0001_0600, 0, 100, 0];
        let entry = |model: u32, id: u32, n: &str, iface: &[u32]| {
            let mut ops = vec![model, id];
            ops.extend(name(n));
            ops.extend_from_slice(iface);
            inst(OP_ENTRY_POINT, &ops)
        };
        let task = entry(5364, 1, "task_main", &[10]);
        let mesh = entry(5365, 2, "mesh_main", &[11, 12]);
        let pointer = inst(32, &[9, STORAGE_TASK_PAYLOAD, 8]);
        let task_var = inst(OP_VARIABLE, &[9, 10, STORAGE_TASK_PAYLOAD]);
        let mesh_var = inst(OP_VARIABLE, &[9, 11, STORAGE_TASK_PAYLOAD]);
        let other = inst(OP_VARIABLE, &[7, 12, 2]);
        for i in [&task, &mesh, &pointer, &task_var, &other, &mesh_var] {
            code.extend_from_slice(i);
        }

        assert!(matches!(
            payload_first(&code, b"task_main"),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            payload_first(&code, b"pixel_main"),
            Cow::Borrowed(_)
        ));
        let moved = payload_first(&code, b"mesh_main");
        let mut want = code[..HEADER_WORDS].to_vec();
        for i in [&task, &mesh, &pointer, &mesh_var, &task_var, &other] {
            want.extend_from_slice(i);
        }
        assert_eq!(*moved, want);
    }
}
