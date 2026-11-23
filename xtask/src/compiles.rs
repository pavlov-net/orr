//! The run's report on the pipelines its test devices compiled, from their compile records,
//! which list the pipelines that came from a cache or the store too.
//!
//! Driver times move with the machine's load, by tens of percent between an idle and a busy
//! machine, so the report reads each compile against others of the same run: as a share of the
//! run's driver CPU, against a reference pipeline and per instruction handed to the driver.

use std::fmt::Write as _;
use std::time::Duration;

use orr_gpu::{CompileRecord, PipelineKind};

use crate::load::Load;

/// The pipeline the others' driver CPU is read against: the path tracer's pipeline that may meet
/// everything, which a view queues once its own pipelines are ready.
const REFERENCE: &str = "path pipeline 0xffffffffffffffff";

/// How many of the costliest compiles the report lists.
const LISTED: usize = 10;

/// The fields the listed compiles' columns show or leave out on purpose; a compile shows any
/// other after its name. A run's device and driver are the same for every compile, and a ray
/// pipeline's shaders are too many to list.
const KNOWN: &[&str] = &[
    "kind",
    "key",
    "device",
    "driver",
    "hit",
    "wall_us",
    "cpu_us",
    "serial_us",
    "longest_join_us",
    "joins",
    "stages",
    "instructions",
    "entries",
    "specs",
    "shaders",
];

/// One pipeline compile of a test target.
pub(crate) struct Compile {
    target: String,
    pub(crate) record: CompileRecord,
    /// The record's fields outside [`KNOWN`], as they appear in its line.
    other: Vec<String>,
}

/// What a test target's compile records file holds.
#[derive(Default)]
pub(crate) struct Records {
    pub(crate) compiles: Vec<Compile>,
    /// The pipelines that came from a cache or the store, which compiled nothing.
    pub(crate) hits: Vec<Compile>,
    /// Lines that are no whole record, such as the last of a process killed while writing it.
    pub(crate) unread: usize,
}

/// The compiles of `target` in `text`, the contents of its records file.
pub(crate) fn read(target: &str, text: &str) -> Records {
    let mut records = Records::default();
    for line in text.lines().filter(|l| !l.is_empty()) {
        let parsed = CompileRecord::parse(line).zip(CompileRecord::fields(line));
        let Some((record, (fields, _))) = parsed else {
            records.unread += 1;
            continue;
        };
        let list = if record.hit {
            &mut records.hits
        } else {
            &mut records.compiles
        };
        list.push(Compile {
            target: target.to_owned(),
            record,
            other: fields
                .filter(|(k, _)| !KNOWN.contains(k))
                .map(|(k, v)| format!("{k}={v}"))
                .collect(),
        });
    }
    records
}

fn secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

/// Driver CPU per instruction handed over, in microseconds, for a ray pipeline. Compute and
/// mesh pipelines compile one stage each, on one thread, at a cost per instruction of their own.
fn micros_per_instruction(r: &CompileRecord) -> Option<f64> {
    (r.kind == PipelineKind::Ray).then(|| secs(r.cost.cpu) * 1e6 / f64::from(r.cost.instructions))
}

/// The driver CPU of the reference pipeline this run, the mean of its compiles where targets
/// compiled it more than once.
fn reference_cpu(compiles: &[Compile]) -> Option<Duration> {
    let cpus: Vec<Duration> = compiles
        .iter()
        .filter(|c| c.record.name == REFERENCE)
        .map(|c| c.record.cost.cpu)
        .collect();
    let n = u32::try_from(cpus.len()).ok().filter(|&n| n > 0)?;
    Some(cpus.iter().sum::<Duration>() / n)
}

/// The report's lines on `compiles`, beside the count of pipelines `loaded` from a cache or the
/// store. Never an error, since driver times say nothing certain about a change on a shared
/// machine.
pub(crate) fn report(
    mut compiles: Vec<Compile>,
    loaded: usize,
    unread: usize,
    load: Option<&Load>,
) -> String {
    let mut out = String::new();
    if compiles.is_empty() {
        if loaded > 0 {
            let _ = writeln!(
                out,
                "\npipelines compiled: none; {loaded} came from a cache or the store"
            );
        }
        if unread > 0 {
            let _ = writeln!(out, "\ncompile records: {unread} lines hold no record");
        }
        return out;
    }
    compiles.sort_by_key(|c| std::cmp::Reverse(c.record.cost.cpu));
    let sum = |f: fn(&CompileRecord) -> Duration| -> Duration {
        compiles.iter().map(|c| f(&c.record)).sum()
    };
    let cpu = sum(|r| r.cost.cpu);
    let ray: Vec<&CompileRecord> = compiles
        .iter()
        .map(|c| &c.record)
        .filter(|r| r.kind == PipelineKind::Ray)
        .collect();
    let ray_instructions: u64 = ray.iter().map(|r| u64::from(r.cost.instructions)).sum();
    let per_instruction = if ray_instructions > 0 {
        let ray_cpu: Duration = ray.iter().map(|r| r.cost.cpu).sum();
        format!(
            "; ray pipelines {:.1} µs of driver CPU per instruction",
            secs(ray_cpu) * 1e6 / ray_instructions as f64
        )
    } else {
        String::new()
    };
    let _ = writeln!(
        out,
        "\npipelines compiled: {}, {:.1} s wall, {:.1} s driver CPU, {:.1} s of it \
         serial{per_instruction}; {loaded} more came from a cache or the store",
        compiles.len(),
        secs(sum(|r| r.cost.wall)),
        secs(cpu),
        secs(sum(|r| r.cost.serial)),
    );
    let reference = reference_cpu(&compiles);
    let _ = writeln!(
        out,
        "  {:>6}  {:>6}  {:>7}  {:>7}  {:>7}  {:>7}  {:>8}  {:>6}  name",
        "share", "ref", "cpu s", "wall s", "serial", "join", "instrs", "µs/in"
    );
    for c in compiles.iter().take(LISTED) {
        let r = &c.record;
        // Windows charges CPU in whole clock ticks, so a run of short compiles can read none.
        let share = if cpu.is_zero() {
            0.0
        } else {
            100.0 * secs(r.cost.cpu) / secs(cpu)
        };
        let ratio = reference.map_or_else(String::new, |d| {
            format!("{:.2}", secs(r.cost.cpu) / secs(d))
        });
        let per = micros_per_instruction(r).map_or_else(String::new, |p| format!("{p:.1}"));
        let mut name = format!("{} ({})", r.name, c.target);
        for field in &c.other {
            name.push(' ');
            name.push_str(field);
        }
        let _ = writeln!(
            out,
            "  {share:>5.1}%  {ratio:>6}  {:>7.2}  {:>7.2}  {:>7.2}  {:>7.2}  {:>8}  {per:>6}  {name}",
            secs(r.cost.cpu),
            secs(r.cost.wall),
            secs(r.cost.serial),
            secs(r.cost.longest_join),
            r.cost.instructions,
        );
    }
    if compiles.len() > LISTED {
        let rest: Duration = compiles[LISTED..].iter().map(|c| c.record.cost.cpu).sum();
        let _ = writeln!(
            out,
            "  and {} more, {:.1} s of driver CPU",
            compiles.len() - LISTED,
            secs(rest)
        );
    }
    let _ = match reference {
        Some(d) => writeln!(out, "  ref: driver CPU over {REFERENCE}'s {:.2} s", secs(d)),
        None => writeln!(out, "  ref: {REFERENCE} didn't compile this run"),
    };
    if unread > 0 {
        let _ = writeln!(out, "  {unread} lines of the records hold no record");
    }
    if let Some(load) = load {
        let _ = writeln!(
            out,
            "  load: the machine's CPUs {:.0} % busy over the tests, {:.0} % of that these compiles",
            100.0 * load.busy_share,
            100.0 * secs(cpu) / secs(load.busy),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_read_against_the_runs_reference() {
        let line = |name: &str, cpu_us: u64, instructions: u32, more: &str| {
            format!(
                "orr-compile kind=ray key=k device=d driver=v wall_us=1000000 cpu_us={cpu_us} \
                 serial_us=0 longest_join_us=0 joins=2 stages=3 instructions={instructions}{more} name={name}"
            )
        };
        let a = format!(
            "{}\n{}\norr-compile kind=ray key=k wall_us=5\n",
            line(REFERENCE, 2_000_000, 40_000, ""),
            line("path pipeline 0x3", 4_000_000, 80_000, " later=7"),
        );
        // A pipeline from the store, whose times the report must leave out.
        let b = format!(
            "{}\n{}\n",
            line(REFERENCE, 6_000_000, 40_000, ""),
            line("path pipeline 0x5", 9_000_000, 90_000, " hit=1"),
        );
        let mut compiled = read("a", &a);
        let mut b = read("b", &b);
        assert_eq!((compiled.unread, b.unread), (1, 0));
        assert_eq!(b.hits.len(), 1);
        compiled.compiles.append(&mut b.compiles);
        let text = report(compiled.compiles, b.hits.len(), 1, None);
        let rows: Vec<Vec<&str>> = text
            .lines()
            .filter(|l| l.contains(" (a)") || l.contains(" (b)"))
            .map(|l| l.split_whitespace().collect())
            .collect();
        // Share, ratio to the mean of the reference's two compiles, then µs per instruction.
        let columns = |row: &Vec<&str>| [row[0], row[1], row[7]].map(str::to_owned);
        assert_eq!(
            rows.iter().map(columns).collect::<Vec<_>>(),
            [
                ["50.0%", "1.50", "150.0"],
                ["33.3%", "1.00", "50.0"],
                ["16.7%", "0.50", "50.0"]
            ]
            .map(|r| r.map(str::to_owned))
        );
        assert!(rows[1].ends_with(&["(a)", "later=7"]));
        assert!(text.contains("12.0 s driver CPU"));
        assert!(text.contains("75.0 µs of driver CPU per instruction"));
        assert!(text.contains("1 more came from a cache or the store"));
        assert!(report(Vec::new(), 0, 0, None).is_empty());
    }
}
