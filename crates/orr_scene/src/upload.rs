//! Upload records and the scatter that applies them.
//!
//! Every writer call puts its data in the frame's upload window and appends one record
//! naming the destination words. At `update_scenes` the records are trimmed so a later
//! record wins wherever two overlap, then one dispatch applies every contiguous record;
//! records that write one field of each table entry (strided, or masked bits of a word)
//! follow in their own dispatches when there are any.

use std::collections::BTreeMap;

use orr_gpu::{CommandList, ComputePipeline, Device, GpuAddress, GpuPtr, Hazards, Stages};
use rustc_hash::FxHashMap;

use crate::Result;
use crate::memory::UploadWindow;
use crate::shaders::upload::{Root, ScatterRecord};

const SCATTER_COPY: u32 = 0;
const SCATTER_FILL: u32 = 1;
const SCATTER_MASKED: u32 = 2;

/// Workgroup size of the scatter kernel.
const GROUP: u32 = 64;
/// Most workgroups one scatter dispatches; threads loop over the rest.
const MAX_GROUPS: u32 = 65_535;
/// Most words one scatter dispatch writes: a thread steps from its word by the threads
/// dispatched until it passes the last, in 32 bits.
const MAX_WORDS: u32 = u32::MAX - MAX_GROUPS * GROUP;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    /// Words from this address in the upload window.
    Copy(u64),
    Fill(u32),
    /// Only the `mask` bits of each destination word take `value`'s.
    Masked {
        value: u32,
        mask: u32,
    },
}

/// `count` words written to `dst`, `dst + 4 * stride`, ... Where two records write the
/// same word, the one with the higher `seq` wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    pub(crate) seq: u64,
    pub(crate) dst: u64,
    pub(crate) count: u32,
    pub(crate) stride: u32,
    pub(crate) op: Op,
}

impl Record {
    pub(crate) fn copy(seq: u64, dst: u64, src: u64, words: u32) -> Self {
        Self {
            seq,
            dst,
            count: words,
            stride: 1,
            op: Op::Copy(src),
        }
    }

    /// A copy of the `T` at `src` (or of the `count` values starting there) to `dst`.
    pub(crate) fn values<T>(seq: u64, dst: GpuPtr<T>, src: GpuPtr<T>, count: u32) -> Self {
        Self::copy(
            seq,
            dst.address().0,
            src.address().0,
            count * (size_of::<T>() / 4) as u32,
        )
    }

    pub(crate) fn fill(seq: u64, dst: u64, words: u32, value: u32) -> Self {
        Self {
            seq,
            dst,
            count: words,
            stride: 1,
            op: Op::Fill(value),
        }
    }

    /// Elements `from..to` of this record.
    fn part(&self, from: u64, to: u64) -> Self {
        let op = match self.op {
            Op::Copy(src) => Op::Copy(src + 4 * from),
            op => op,
        };
        Self {
            dst: self.dst + 4 * from * u64::from(self.stride),
            count: (to - from) as u32,
            op,
            ..*self
        }
    }
}

/// Disjoint half-open intervals.
#[derive(Default)]
struct Intervals(BTreeMap<u64, u64>);

impl Intervals {
    /// The parts of `s..e` not covered.
    fn uncovered(&self, s: u64, e: u64) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        let mut at = s;
        let first = self.0.range(..=s).next_back().map_or(s, |(&k, _)| k);
        for (&is, &ie) in self.0.range(first..e) {
            if ie <= at {
                continue;
            }
            if is > at {
                out.push((at, is));
            }
            at = ie;
            if at >= e {
                break;
            }
        }
        if at < e {
            out.push((at, e));
        }
        out
    }

    fn insert(&mut self, mut s: u64, mut e: u64) {
        let first = self.0.range(..=s).next_back().map_or(s, |(&k, _)| k);
        let touching: Vec<(u64, u64)> = self
            .0
            .range(first..=e)
            .filter(|&(_, &ie)| ie >= s)
            .map(|(&k, &v)| (k, v))
            .collect();
        for (k, v) in touching {
            self.0.remove(&k);
            s = s.min(k);
            e = e.max(v);
        }
        self.0.insert(s, e);
    }
}

/// Removes `lo..hi` from every range in `ranges`.
fn subtract(ranges: Vec<(u64, u64)>, lo: u64, hi: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::with_capacity(ranges.len() + 1);
    for (s, e) in ranges {
        if s < lo.min(e) {
            out.push((s, lo.min(e)));
        }
        if hi.max(s) < e {
            out.push((hi.max(s), e));
        }
    }
    out
}

/// Splits `records` into three passes run in order (contiguous writes, strided writes,
/// masked writes), trimmed so that no two records of a pass write the same word and each
/// word ends as if the records had applied in `seq` order. Strided records into one table
/// use one stride and masked records into one word one mask.
fn trim(mut records: Vec<Record>) -> [Vec<Record>; 3] {
    records.sort_by_key(|r| std::cmp::Reverse(r.seq));
    let mut words = Intervals::default();
    // Strided coverage per lattice (stride, first word mod stride), in lattice indices
    // (word / stride): full writes and masked writes.
    let mut full: FxHashMap<(u32, u64), Intervals> = FxHashMap::default();
    let mut masked: FxHashMap<(u32, u64), Intervals> = FxHashMap::default();
    let mut passes = [Vec::new(), Vec::new(), Vec::new()];
    for r in records.into_iter().filter(|r| r.count > 0) {
        let a = r.dst / 4;
        let n = u64::from(r.count);
        let is_masked = matches!(r.op, Op::Masked { .. });
        if r.stride == 1 && !is_masked {
            for (s, e) in words.uncovered(a, a + n) {
                passes[0].push(r.part(s - a, e - a));
            }
            words.insert(a, a + n);
            continue;
        }
        let stride = u64::from(r.stride);
        let key = (r.stride, a % stride);
        let g0 = a / stride;
        // Elements of this record in lattice indices: g0..g0 + n.
        let mut free = vec![(g0, g0 + n)];
        let later = if is_masked {
            vec![full.get(&key), masked.get(&key)]
        } else {
            vec![full.get(&key)]
        };
        for set in later.into_iter().flatten() {
            let mut at = g0;
            for (us, ue) in set.uncovered(g0, g0 + n) {
                free = subtract(free, at, us);
                at = ue;
            }
            free = subtract(free, at, g0 + n);
        }
        // Element i's word a + i * stride is covered by a later contiguous cs..ce when
        // cs <= a + i * stride < ce.
        let span = a + (n - 1) * stride + 1;
        let mut at = a;
        for (us, ue) in words.uncovered(a, span) {
            free = subtract(
                free,
                g0 + (at - a).div_ceil(stride),
                g0 + (us - a).div_ceil(stride),
            );
            at = ue;
        }
        free = subtract(free, g0 + (at - a).div_ceil(stride), g0 + n);
        let pass = if is_masked { 2 } else { 1 };
        passes[pass].extend(free.into_iter().map(|(s, e)| r.part(s - g0, e - g0)));
        let coverage = if is_masked { &mut masked } else { &mut full };
        coverage.entry(key).or_default().insert(g0, g0 + n);
    }
    passes
}

/// Splits a pass into the runs of records one dispatch each applies, no run past
/// [`MAX_WORDS`]: a record that would pass it goes on in the next.
fn dispatches(pass: &[Record]) -> Vec<Vec<Record>> {
    let mut out: Vec<Vec<Record>> = Vec::new();
    let mut words = 0;
    for r in pass {
        let mut from = 0;
        while from < r.count {
            if out.is_empty() || words == MAX_WORDS {
                out.push(Vec::new());
                words = 0;
            }
            let to = from + (r.count - from).min(MAX_WORDS - words);
            let run = out.last_mut().expect("pushed above");
            run.push(r.part(u64::from(from), u64::from(to)));
            words += to - from;
            from = to;
        }
    }
    out
}

/// Totals of one scatter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScatterTotals {
    pub(crate) records: u32,
    pub(crate) words: u64,
    pub(crate) dispatches: u32,
}

/// Writes one pass's GPU records and root into the window and dispatches it.
fn dispatch_pass(
    device: &Device,
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    pipeline: &ComputePipeline,
    records: &[Record],
    totals: &mut ScatterTotals,
) -> Result<()> {
    if records.is_empty() {
        return Ok(());
    }
    let (gpu, out) = window.slice::<ScatterRecord>(device, records.len() as u32)?;
    let mut first = 0u32;
    for (o, r) in out.iter_mut().zip(records) {
        let (op, src, value, mask) = match r.op {
            Op::Copy(src) => (SCATTER_COPY, src, 0, 0),
            Op::Fill(v) => (SCATTER_FILL, 0, v, 0),
            Op::Masked { value, mask } => (SCATTER_MASKED, 0, value, mask),
        };
        // SAFETY: both addresses lie in live GPU allocations, a scene table and the window.
        *o = unsafe {
            ScatterRecord {
                dst: GpuPtr::from_address(GpuAddress(r.dst)),
                src: GpuPtr::from_address(GpuAddress(src)),
                first,
                op: op | r.stride << 8,
                value,
                mask,
            }
        };
        first += r.count;
    }
    let groups = first.div_ceil(GROUP).min(MAX_GROUPS);
    let root_ptr = window.put_root(
        device,
        &Root {
            records: gpu,
            count: records.len() as u32,
            total: first,
            threads: groups * GROUP,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        },
    )?;
    list.dispatch(pipeline, root_ptr, [groups, 1, 1]);
    totals.records += records.len() as u32;
    totals.words += u64::from(first);
    totals.dispatches += 1;
    Ok(())
}

/// Records the scatter of `records`: the dispatches of each pass of [`trim`] in turn.
pub(crate) fn scatter(
    device: &Device,
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    pipeline: &ComputePipeline,
    records: Vec<Record>,
) -> Result<ScatterTotals> {
    let mut totals = ScatterTotals::default();
    for pass in &trim(records) {
        for (i, run) in dispatches(pass).iter().enumerate() {
            // A pass's records write no word twice; a later pass writes over an earlier one.
            if i == 0 && totals.dispatches > 0 {
                list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            }
            dispatch_pass(device, window, list, pipeline, run, &mut totals)?;
        }
    }
    Ok(totals)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Applies records to `mem` (word-addressed from 0) the way the GPU passes do.
    fn apply(mem: &mut [u32], src: &[u32], records: &[Record]) {
        for r in records {
            for i in 0..r.count as usize {
                let at = r.dst as usize / 4 + i * r.stride as usize;
                mem[at] = match r.op {
                    Op::Copy(s) => src[s as usize / 4 + i],
                    Op::Fill(v) => v,
                    Op::Masked { value, mask } => (mem[at] & !mask) | (value & mask),
                };
            }
        }
    }

    #[test]
    fn a_pass_past_one_dispatch_splits_into_runs_that_write_the_same_words() {
        let fill = |dst: u64, count: u32, stride: u32| Record {
            seq: 1,
            dst,
            count,
            stride,
            op: Op::Fill(7),
        };
        let copy = Record::copy(2, 1 << 40, 1 << 50, u32::MAX);
        let pass = [fill(0, 100, 1), copy, fill(1 << 36, u32::MAX, 14), copy];
        let runs = dispatches(&pass);
        assert_eq!(runs.len(), 4);
        let mut at = 0;
        let mut done = 0u64;
        for run in &runs {
            let words: u64 = run.iter().map(|r| u64::from(r.count)).sum();
            assert!(words <= u64::from(MAX_WORDS));
            for r in run {
                let whole = &pass[at];
                assert_eq!(*r, whole.part(done, done + u64::from(r.count)));
                done += u64::from(r.count);
                if done == u64::from(whole.count) {
                    at += 1;
                    done = 0;
                }
            }
        }
        assert_eq!((at, done), (pass.len(), 0));
    }

    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn trimmed_passes_equal_applying_in_order() {
        const WORDS: u64 = 600;
        let src: Vec<u32> = (0..4096u32)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        let mut state = 0x2545_f491_4f6c_dd1d;
        for case in 0..2000 {
            let mut records = Vec::new();
            for seq in 0..1 + rng(&mut state) % 12 {
                let stride = [1, 1, 12][(rng(&mut state) % 3) as usize];
                let count = 1 + (rng(&mut state) % 40) as u32;
                let span = (u64::from(count) - 1) * u64::from(stride) + 1;
                let dst = 4 * (rng(&mut state) % (WORDS - span));
                let op = match rng(&mut state) % 3 {
                    0 => Op::Copy(4 * (rng(&mut state) % 2000)),
                    k if k == 1 || stride == 1 => Op::Fill(rng(&mut state) as u32),
                    _ => Op::Masked {
                        value: rng(&mut state) as u32,
                        mask: 0xff,
                    },
                };
                records.push(Record {
                    seq,
                    dst,
                    count,
                    stride,
                    op,
                });
            }
            let mut expected = vec![7u32; WORDS as usize];
            apply(&mut expected, &src, &records);

            let passes = trim(records.clone());
            for pass in &passes {
                let mut writes = vec![0u32; WORDS as usize];
                for r in pass {
                    for i in 0..r.count as usize {
                        writes[r.dst as usize / 4 + i * r.stride as usize] += 1;
                    }
                }
                assert!(
                    writes.iter().all(|&w| w <= 1),
                    "case {case}: a pass writes a word twice"
                );
            }
            let mut got = vec![7u32; WORDS as usize];
            for pass in &passes {
                apply(&mut got, &src, pass);
            }
            assert_eq!(got, expected, "case {case}: {records:?}");
        }
    }
}
