//! What a pipeline compile cost the driver and what it made, and the line a device writes for
//! each pipeline it creates to the file of
//! [`DeviceDesc::compile_records`](crate::DeviceDesc::compile_records).

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// What one pipeline's compile cost the driver, and what the driver reports of the code it made.
/// A pipeline that came from the pipeline cache or the store has none.
///
/// CPU times are the threads' own CPU time, kernel and user, spent inside the driver's creation
/// and join calls: `GetThreadTimes` on Windows, `CLOCK_THREAD_CPUTIME_ID` elsewhere. Windows
/// charges a thread whole clock ticks (15.625 ms by default), so there each figure is off by up
/// to a tick per call it sums, and a compile shorter than a tick may read 0.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    /// A compute or mesh pipeline's stages; empty for a ray tracing pipeline, whose `shaders`
    /// name its shaders.
    pub entries: Vec<StageEntry>,
    /// What the driver reports of the pipeline's executables; `None` where it doesn't report
    /// these figures, as for a ray tracing pipeline, without `Caps::pipeline_executable_info`, or
    /// on a driver whose statistics name none of them.
    pub code: Option<CodeStatistics>,
    /// A ray tracing pipeline's shaders, each once, with their stacks; empty for other kinds.
    pub shaders: Vec<RayShader>,
}

/// What the driver reports of a pipeline's machine code. Of a pipeline with several
/// executables, such as a mesh pipeline's task, mesh and pixel stages, the most any one takes,
/// and the code of all of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CodeStatistics {
    /// Registers per thread.
    pub registers: u32,
    /// Bytes of local memory per thread: spill, and arrays indexed at run time.
    pub local_bytes: u64,
    /// Bytes of shared memory per workgroup: `groupshared` variables, and the spill and arrays
    /// indexed at run time that a driver puts there before local memory.
    pub shared_bytes: u64,
    pub binary_bytes: u64,
}

/// A stage as the driver is handed it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageEntry {
    /// The name its entry point takes in the code the driver is handed.
    pub name: String,
    /// Its specialization constants' bytes, which tell apart the pipelines one entry makes.
    pub spec: Vec<u8>,
}

impl StageEntry {
    /// Its specialization data as a compile record writes it: hex bytes, or `-` for none.
    fn spec_hex(&self) -> String {
        if self.spec.is_empty() {
            "-".to_owned()
        } else {
            self.spec.iter().map(|b| format!("{b:02x}")).collect()
        }
    }
}

impl CompileCost {
    /// Its `entries` as a compile record's `entries` and `specs` fields write them.
    pub fn entry_fields(&self) -> (String, String) {
        let names: Vec<&str> = self.entries.iter().map(|e| e.name.as_str()).collect();
        let specs: Vec<String> = self.entries.iter().map(StageEntry::spec_hex).collect();
        (names.join(","), specs.join(","))
    }
}

/// A ray tracing shader as the driver is handed it, and the stack it keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RayShader {
    /// The stage's label, or where it has none the name its entry point takes in the code the
    /// driver is handed.
    pub shader: String,
    /// What it holds in memory across its calls and traces, as
    /// `vkGetRayTracingShaderGroupStackSizeKHR` reports it.
    pub stack_bytes: u64,
    /// As [`handed_instructions`](crate::handed_instructions) counts them.
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

/// One pipeline a device created, compiled or from a cache, as a line of a compile records
/// file, its fields in this order and those in brackets where they apply:
///
/// ```text
/// orr-compile kind=<kind> key=<hex> device=<device> driver=<driver> [hit=1] wall_us=<n>
///     cpu_us=<n> serial_us=<n> longest_join_us=<n> joins=<n> stages=<n> instructions=<n>
///     [entries=<entry>,... specs=<hex>,...]
///     [registers=<n> local_bytes=<n> shared_bytes=<n> binary_bytes=<n>]
///     [shaders=<shader>:<stack_bytes>:<instructions>,...] name=<name>
/// ```
///
/// A line holds no breaks. Times are whole microseconds. `kind` is `compute`, `mesh` or `ray`;
/// `key` is the hash that names the pipeline's file in a pipeline store. `device` and `driver`
/// are the device's name and its driver's name and version, as `Caps::device_name` and
/// `Caps::driver` give them, with `_` for each space. `hit=1` marks a pipeline that came from the
/// pipeline cache or the store, which cost the driver no compile: its times are 0, and the rest
/// is what the driver reports of it as of a compiled one. A device records such a pipeline once,
/// however often it creates it. `entries` lists a compute or mesh pipeline's stages by
/// [`StageEntry::name`], and `specs` their specialization data in the same order, as hex bytes or
/// `-` for none. The code statistics are [`CodeStatistics`]', where the driver reports them.
/// `shaders` lists a ray tracing pipeline's shaders. `name` comes last and runs to the end of the
/// line, spaces and all. A reader skips fields it doesn't know.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileRecord {
    pub kind: PipelineKind,
    pub key: String,
    /// The device that compiled it, with `_` for each space.
    pub device: String,
    /// The device's driver name and version, with `_` for each space.
    pub driver: String,
    /// It came from the pipeline cache or the store, and `cost`'s times are 0.
    pub hit: bool,
    pub name: String,
    pub cost: CompileCost,
}

const PREFIX: &str = "orr-compile ";

impl fmt::Display for CompileRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.cost;
        write!(
            f,
            "{PREFIX}kind={} key={} device={} driver={} wall_us={} cpu_us={} serial_us={} \
             longest_join_us={} joins={} stages={} instructions={}",
            self.kind.name(),
            self.key,
            self.device,
            self.driver,
            c.wall.as_micros(),
            c.cpu.as_micros(),
            c.serial.as_micros(),
            c.longest_join.as_micros(),
            c.joins,
            c.stages,
            c.instructions,
        )?;
        if self.hit {
            write!(f, " hit=1")?;
        }
        if !c.entries.is_empty() {
            let (names, specs) = c.entry_fields();
            write!(f, " entries={names} specs={specs}")?;
        }
        if let Some(code) = &c.code {
            write!(
                f,
                " registers={} local_bytes={} shared_bytes={} binary_bytes={}",
                code.registers, code.local_bytes, code.shared_bytes, code.binary_bytes
            )?;
        }
        for (i, s) in c.shaders.iter().enumerate() {
            let lead = if i == 0 { " shaders=" } else { "," };
            write!(f, "{lead}{}:{}:{}", s.shader, s.stack_bytes, s.instructions)?;
        }
        write!(f, " name={}", self.name)
    }
}

impl CompileRecord {
    /// The record `line` holds; `None` for a line that is not a whole record.
    pub fn parse(line: &str) -> Option<Self> {
        let (fields, name) = Self::fields(line)?;
        let fields: HashMap<&str, &str> = fields.collect();
        let micros = |k| fields.get(k)?.parse().ok().map(Duration::from_micros);
        let count = |k| fields.get(k)?.parse().ok();
        let bytes = |k| fields.get(k)?.parse().ok();
        Some(Self {
            kind: match *fields.get("kind")? {
                "compute" => PipelineKind::Compute,
                "mesh" => PipelineKind::Mesh,
                "ray" => PipelineKind::Ray,
                _ => return None,
            },
            key: (*fields.get("key")?).to_owned(),
            device: (*fields.get("device")?).to_owned(),
            driver: (*fields.get("driver")?).to_owned(),
            hit: match fields.get("hit") {
                None => false,
                Some(&"1") => true,
                Some(_) => return None,
            },
            name: name.to_owned(),
            cost: CompileCost {
                wall: micros("wall_us")?,
                cpu: micros("cpu_us")?,
                serial: micros("serial_us")?,
                longest_join: micros("longest_join_us")?,
                joins: count("joins")?,
                stages: count("stages")?,
                instructions: count("instructions")?,
                entries: match (fields.get("entries"), fields.get("specs")) {
                    (Some(names), Some(specs)) => names
                        .split(',')
                        .zip(specs.split(','))
                        .map(|(name, spec)| {
                            Some(StageEntry {
                                name: name.to_owned(),
                                spec: match spec {
                                    "-" => Vec::new(),
                                    hex => (0..hex.len())
                                        .step_by(2)
                                        .map(|i| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok())
                                        .collect::<Option<_>>()?,
                                },
                            })
                        })
                        .collect::<Option<_>>()?,
                    _ => Vec::new(),
                },
                code: match fields.get("registers") {
                    None => None,
                    Some(registers) => Some(CodeStatistics {
                        registers: registers.parse().ok()?,
                        local_bytes: bytes("local_bytes")?,
                        shared_bytes: bytes("shared_bytes")?,
                        binary_bytes: bytes("binary_bytes")?,
                    }),
                },
                shaders: match fields.get("shaders") {
                    None => Vec::new(),
                    Some(shaders) => shaders
                        .split(',')
                        .map(|s| {
                            let (s, instructions) = s.rsplit_once(':')?;
                            let (shader, stack_bytes) = s.rsplit_once(':')?;
                            Some(RayShader {
                                shader: shader.to_owned(),
                                stack_bytes: stack_bytes.parse().ok()?,
                                instructions: instructions.parse().ok()?,
                            })
                        })
                        .collect::<Option<_>>()?,
                },
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
        let cost = CompileCost {
            wall: Duration::from_micros(8_410_233),
            cpu: Duration::from_micros(40_921_875),
            serial: Duration::from_micros(2_015_625),
            longest_join: Duration::from_micros(6_890_625),
            joins: 32,
            stages: 39,
            instructions: 882_114,
            ..CompileCost::default()
        };
        let ray = CompileRecord {
            kind: PipelineKind::Ray,
            key: "ab12".into(),
            device: "NVIDIA_GeForce_RTX_5090".into(),
            driver: "NVIDIA_621.24".into(),
            hit: false,
            name: "path pipeline 0xffffffffffffffff".into(),
            cost: CompileCost {
                shaders: vec![
                    RayShader {
                        shader: "paths".into(),
                        stack_bytes: 480,
                        instructions: 21_345,
                    },
                    RayShader {
                        shader: "surface_hit.triangles.any.layer".into(),
                        stack_bytes: 0,
                        instructions: 9_876,
                    },
                ],
                ..cost.clone()
            },
        };
        let compute = CompileRecord {
            kind: PipelineKind::Compute,
            key: "cd34".into(),
            device: "NVIDIA_GeForce_RTX_5090".into(),
            driver: "NVIDIA_621.24".into(),
            hit: true,
            name: "resolve".into(),
            cost: CompileCost {
                entries: vec![
                    StageEntry {
                        name: "orr_scene.resolve".into(),
                        spec: vec![0xff, 0xff, 0xff, 0x7f],
                    },
                    StageEntry {
                        name: "orr_scene.resolve.pixel".into(),
                        spec: Vec::new(),
                    },
                ],
                code: Some(CodeStatistics {
                    registers: 255,
                    local_bytes: 1472,
                    shared_bytes: 0,
                    binary_bytes: 81_920,
                }),
                ..cost
            },
        };
        for record in [ray, compute] {
            let line = record.to_string();
            assert_eq!(CompileRecord::parse(&line), Some(record.clone()));
            let newer = line.replace(" name=", " later=7 name=");
            assert_eq!(CompileRecord::parse(&newer), Some(record));
            let cut = line.split(" stages=").next().unwrap();
            assert_eq!(CompileRecord::parse(cut), None);
        }
    }
}
