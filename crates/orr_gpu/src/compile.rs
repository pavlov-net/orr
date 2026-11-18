//! What a pipeline compile cost the driver, and the line a device writes for each compile to the
//! file of [`DeviceDesc::compile_records`](crate::DeviceDesc::compile_records).

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// What one pipeline's compile cost the driver. A pipeline that came from the pipeline cache or
/// the store has none.
///
/// CPU times are the threads' own CPU time, kernel and user, spent inside the driver's creation
/// and join calls: `GetThreadTimes` on Windows, `CLOCK_THREAD_CPUTIME_ID` elsewhere. Windows
/// charges a thread whole clock ticks (15.625 ms by default), so there each figure is off by up
/// to a tick per call it sums, and a compile shorter than a tick may read 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileCost {
    /// From the creation call to the pipeline's completion, on the creating thread.
    pub wall: Duration,
    /// Driver CPU summed over every thread that worked on the compile.
    pub cpu: Duration,
    /// Driver CPU on the creating thread outside the joins: the creation call itself, which no
    /// other thread shares.
    pub serial: Duration,
    /// The most driver CPU one join took: one thread's calls from joining the deferred
    /// creation until the driver had no more work for it.
    pub longest_join: Duration,
    /// Joins of the deferred creation, the creating thread's among them; 0 for a creation on
    /// one thread.
    pub joins: u32,
    pub stages: u32,
    /// The instructions the stages' functions hand the driver, as
    /// [`handed_instructions`](crate::handed_instructions) counts them.
    pub instructions: u32,
}

/// What kind of pipeline a [`CompileRecord`] is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineKind {
    Compute,
    Mesh,
    Ray,
}

impl PipelineKind {
    fn name(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Mesh => "mesh",
            Self::Ray => "ray",
        }
    }
}

/// One compile, as a line of a compile records file:
///
/// ```text
/// orr-compile kind=ray key=<hex> wall_us=8410233 cpu_us=40921875 serial_us=2015625 longest_join_us=6890625 joins=32 stages=39 instructions=882114 name=path pipeline 0xffffffffffffffff
/// ```
///
/// Times are whole microseconds. `kind` is `compute`, `mesh` or `ray`; `key` is the hash that
/// names the pipeline's file in a pipeline store. `name` comes last and runs to the end of the
/// line, spaces and all. A reader skips fields it doesn't know.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileRecord {
    pub kind: PipelineKind,
    pub key: String,
    pub name: String,
    pub cost: CompileCost,
}

const PREFIX: &str = "orr-compile ";

impl fmt::Display for CompileRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.cost;
        write!(
            f,
            "{PREFIX}kind={} key={} wall_us={} cpu_us={} serial_us={} longest_join_us={} joins={} \
             stages={} instructions={} name={}",
            self.kind.name(),
            self.key,
            c.wall.as_micros(),
            c.cpu.as_micros(),
            c.serial.as_micros(),
            c.longest_join.as_micros(),
            c.joins,
            c.stages,
            c.instructions,
            self.name
        )
    }
}

impl CompileRecord {
    /// The record `line` holds; `None` for a line that is not a whole record.
    pub fn parse(line: &str) -> Option<Self> {
        let (fields, name) = Self::fields(line)?;
        let fields: HashMap<&str, &str> = fields.collect();
        let micros = |k| fields.get(k)?.parse().ok().map(Duration::from_micros);
        let count = |k| fields.get(k)?.parse().ok();
        Some(Self {
            kind: match *fields.get("kind")? {
                "compute" => PipelineKind::Compute,
                "mesh" => PipelineKind::Mesh,
                "ray" => PipelineKind::Ray,
                _ => return None,
            },
            key: (*fields.get("key")?).to_owned(),
            name: name.to_owned(),
            cost: CompileCost {
                wall: micros("wall_us")?,
                cpu: micros("cpu_us")?,
                serial: micros("serial_us")?,
                longest_join: micros("longest_join_us")?,
                joins: count("joins")?,
                stages: count("stages")?,
                instructions: count("instructions")?,
            },
        })
    }

    /// Every `key=value` field of a record line in line order, those [`Self::parse`] doesn't
    /// read among them, and its name; `None` for a line that is no record.
    pub fn fields(line: &str) -> Option<(impl Iterator<Item = (&str, &str)>, &str)> {
        let (fields, name) = line.strip_prefix(PREFIX)?.split_once(" name=")?;
        Some((fields.split(' ').filter_map(|f| f.split_once('=')), name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_read_back_as_written() {
        let record = CompileRecord {
            kind: PipelineKind::Ray,
            key: "ab12".into(),
            name: "path pipeline 0xffffffffffffffff".into(),
            cost: CompileCost {
                wall: Duration::from_micros(8_410_233),
                cpu: Duration::from_micros(40_921_875),
                serial: Duration::from_micros(2_015_625),
                longest_join: Duration::from_micros(6_890_625),
                joins: 32,
                stages: 39,
                instructions: 882_114,
            },
        };
        let line = record.to_string();
        assert_eq!(CompileRecord::parse(&line), Some(record.clone()));
        let newer = line.replace(" name=", " later=7 name=");
        assert_eq!(CompileRecord::parse(&newer), Some(record));
        let cut = line.split(" stages=").next().unwrap();
        assert_eq!(CompileRecord::parse(cut), None);
    }
}
