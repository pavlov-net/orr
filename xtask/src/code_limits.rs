//! Hard checks on what the driver reports of the pipelines a run used, read from the targets'
//! compile records: no spill in orr's compute and mesh kernels, and the path stages'
//! stacks within their budgets. Unlike times, these don't depend on what else the machine runs,
//! so a breach fails the run on the device and driver the limits were measured on
//! ([`MEASURED_DEVICE`], [`MEASURED_DRIVER`]). Elsewhere the figures differ, so breaches are
//! reported without failing, with a call to re-measure where only the driver differs.
//!
//! A run checks every pipeline it used, those from a cache or the pipeline store among them,
//! whose records the driver fills as when it compiled them: a pipeline that breaks a limit fails
//! every run that uses it. The test crates' own kernels run in no frame, so they aren't held.
//!
//! Spill is local memory past [`LOCAL_FLOOR`]: the driver's statistics don't tell spill from a
//! kernel's arrays indexed at run time, and no orr kernel keeps one there. Shared-memory spill
//! isn't held: the driver spills there first, but beside `groupshared` variables whose layout
//! the code doesn't fix, so no figure separates the two.

use std::fmt;

use orr_gpu::{CompileRecord, PipelineKind};

use crate::compiles::Compile;

/// The device and driver the limits were measured on, as compile records write them.
const MEASURED_DEVICE: &str = "NVIDIA_GeForce_RTX_5090";
const MEASURED_DRIVER: &str = "NVIDIA_621.24";

/// Local memory per thread that NVIDIA's driver reports for some kernels with no spill and no
/// array indexed at run time, where most report none: kernels of 34 to 151 registers with no
/// such array, and one of 249 registers, short of the 255 it could take.
const LOCAL_FLOOR: u64 = 16;

/// A kernel that spills today on [`MEASURED_DEVICE`] under [`MEASURED_DRIVER`], with the bead
/// that removes the spill. It may spill at any specialization; at `spec` it must, so that it
/// leaves the list once it stops.
struct KnownSpill {
    entries: &'static str,
    /// Specialization data as the record writes it, or empty for any.
    spec: &'static str,
    bead: &'static str,
}

/// `path_may` and `path_may2` of everything, in the specialization data of a masked support
/// kernel.
const MASKED_ALL: &str = "0000000000000000000000000000000001000000ffffffffffffffff00000000";

const KNOWN_SPILL: &[KnownSpill] = &[
    KnownSpill {
        entries: "orr_scene.resolve.resolve",
        spec: "ffffffff",
        bead: "orr-74ln",
    },
    KnownSpill {
        entries: "orr_scene.tessellate.dice",
        spec: "",
        bead: "orr-74ln",
    },
    KnownSpill {
        entries: "orr_scene.tessellate.split",
        spec: "",
        bead: "orr-74ln",
    },
    KnownSpill {
        entries: "orr_lighting.media.apply",
        spec: MASKED_ALL,
        bead: "orr-dmmc",
    },
    KnownSpill {
        entries: "orr_lighting.media.froxel_light.retrace_main",
        spec: MASKED_ALL,
        bead: "orr-cye4",
    },
    KnownSpill {
        entries: "orr_lighting.media.froxel_light.sample_main",
        spec: "00000000000000000000000000000000010000007ff83f3f105ffeff00000000",
        bead: "orr-cye4",
    },
    KnownSpill {
        entries: "orr_lighting.media.froxel.eye_column",
        spec: "",
        bead: "orr-cye4",
    },
    KnownSpill {
        entries: "orr_lighting.guides.resolve",
        spec: MASKED_ALL,
        bead: "orr-zte0",
    },
    KnownSpill {
        entries: "orr_lighting.dielectric.layer_guides",
        spec: "00000000000000000000000000000000010000007c50303e1014feff00000000",
        bead: "orr-zte0",
    },
    KnownSpill {
        entries: "orr_lighting.media.particle_draw.particle_mesh,\
                  orr_lighting.media.particle_draw.particle_pixel",
        spec: "",
        bead: "orr-idmb",
    },
    KnownSpill {
        entries: "orr_lighting.rays.shadow",
        spec: "0000000000000000000000000000000000000000fff83f3f135ffeff00000000",
        bead: "orr-p3iv",
    },
    KnownSpill {
        entries: "orr_lighting.media.smoke",
        spec: "",
        bead: "orr-fzqi",
    },
    KnownSpill {
        entries: "orr_scene.pixel_probe.probe",
        spec: "",
        bead: "orr-0fcp",
    },
    KnownSpill {
        entries: "orr_scene.raster.cluster_mesh,orr_scene.raster.alpha_pixel",
        spec: "",
        bead: "orr-xcux",
    },
];

/// The most bytes of stack each shader of a path pipeline may keep, by its label without the
/// linked surface it is made for: one for every kind of stage the stage table makes, each the
/// most it kept on [`MEASURED_DEVICE`] under [`MEASURED_DRIVER`] in the tests' pipelines, and in
/// pipelines with and without linked surfaces that may meet everything but the walk, the rich
/// lobes or emission, which make the kinds no test scene does.
const PATH_STACKS: &[(&str, u64)] = &[
    ("backgrounds", 448),
    ("box_medium_hit", 0),
    ("caster_photon", 128),
    ("cover_hit.copies", 0),
    ("cover_hit.producers", 0),
    ("deposit", 0),
    ("disperse.layer", 0),
    ("emitter_view.rich", 0),
    ("emitter_view.walk", 0),
    ("eye_interior", 0),
    ("eye_lens", 0),
    ("first_vertex", 448),
    ("first_vertex.rich", 608),
    ("first_vertex.walk", 672),
    ("first_view", 0),
    ("first_view.rich", 0),
    ("first_view.walk", 0),
    ("front", 0),
    ("front_met_hit", 0),
    ("gather", 0),
    ("light_box_hit", 0),
    ("light_met_hit", 0),
    ("map_gather", 0),
    ("medium_vertex", 192),
    ("miss_scatter", 0),
    ("miss_shadow", 0),
    ("paths", 480),
    ("photons", 128),
    ("proxy_shadow_hit", 0),
    ("shadow_hit.triangles", 0),
    ("stack_step", 0),
    ("strand_hit.strands", 448),
    ("strand_hit.strands.emitter", 480),
    ("strand_hit.strands.rich", 640),
    ("strand_hit.strands.rich.emitter", 672),
    ("strand_hit.strands.walk", 736),
    ("strand_hit.strands.walk.emitter", 768),
    ("surface_hit.copies", 448),
    ("surface_hit.copies.emitter", 480),
    ("surface_hit.copies.layer", 896),
    ("surface_hit.copies.layer.emitter", 960),
    ("surface_hit.copies.rich", 640),
    ("surface_hit.copies.rich.emitter", 640),
    ("surface_hit.copies.walk", 736),
    ("surface_hit.copies.walk.emitter", 736),
    ("surface_hit.producers", 480),
    ("surface_hit.producers.emitter", 480),
    ("surface_hit.producers.layer", 896),
    ("surface_hit.producers.layer.emitter", 960),
    ("surface_hit.producers.rich", 640),
    ("surface_hit.producers.rich.emitter", 672),
    ("surface_hit.producers.walk", 736),
    ("surface_hit.producers.walk.emitter", 768),
    ("updates", 448),
    ("volume_march", 0),
    ("walk_leaves.walk", 448),
    ("walk_meets.layer", 448),
];

/// A path stage's label without the surface segment (`.any` or `.surface<id>`), which a scene's
/// linked surface types decide.
fn stage_kind(label: &str) -> String {
    label
        .split('.')
        .filter(|s| {
            *s != "any"
                && !s
                    .strip_prefix("surface")
                    .is_some_and(|id| id.parse::<u32>().is_ok())
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// What breaks the limits in record `r`, a line each.
fn breaches(r: &CompileRecord) -> Vec<String> {
    let mut breaches = Vec::new();
    let c = &r.cost;
    let crate_name = c.entries.first().and_then(|e| e.name.split('.').next());
    let ours = crate_name.is_some_and(|n| n == "orr" || n.starts_with("orr_"));
    if let (Some(code), true) = (&c.code, ours) {
        let (entries, specs) = c.entry_fields();
        let spills = code.local_bytes > LOCAL_FLOOR;
        let figures = format!(
            "{} registers, {} B local, {} B shared",
            code.registers, code.local_bytes, code.shared_bytes
        );
        match KNOWN_SPILL.iter().find(|k| k.entries == entries) {
            None if spills => breaches.push(format!(
                "{} ({entries}, spec {specs}) spills: {figures}",
                r.name
            )),
            Some(k) if !spills && (k.spec.is_empty() || k.spec == specs) => {
                breaches.push(format!(
                    "{} ({entries}, spec {specs}) no longer spills ({figures}): take it off \
                     KNOWN_SPILL and close {}",
                    r.name, k.bead
                ));
            }
            _ => {}
        }
    }
    if r.kind == PipelineKind::Ray && r.name.starts_with("path pipeline") {
        for s in &c.stacks {
            let kind = stage_kind(&s.shader);
            match PATH_STACKS.iter().find(|(k, _)| *k == kind) {
                Some(&(_, most)) if s.bytes > most => breaches.push(format!(
                    "{}: {} keeps a stack of {} B, over its {most} B",
                    r.name, s.shader, s.bytes
                )),
                Some(_) => {}
                None => breaches.push(format!(
                    "{}: {} keeps a stack of {} B, and PATH_STACKS has no budget for {kind}",
                    r.name, s.shader, s.bytes
                )),
            }
        }
    }
    breaches
}

/// What breaks the limits among a run's compiles, by how far the limits hold where they ran.
#[derive(Default)]
pub(crate) struct Limits {
    /// On the device and driver the limits were measured on.
    failing: Vec<String>,
    /// On that device under other drivers, by driver.
    other_driver: Vec<(String, String)>,
    /// On other devices, by device and driver.
    other_device: Vec<(String, String)>,
}

impl Limits {
    /// Whether a breach fails the run.
    pub(crate) fn fails(&self) -> bool {
        !self.failing.is_empty()
    }
}

/// Checks the limits on `compiles`.
pub(crate) fn check<'a>(compiles: impl Iterator<Item = &'a Compile>) -> Limits {
    let mut limits = Limits::default();
    for r in compiles.map(|c| &c.record) {
        for breach in breaches(r) {
            if r.device != MEASURED_DEVICE {
                let at = format!("{} {}", r.device, r.driver);
                limits.other_device.push((at, breach));
            } else if r.driver != MEASURED_DRIVER {
                limits.other_driver.push((r.driver.clone(), breach));
            } else {
                limits.failing.push(breach);
            }
        }
    }
    limits.failing.sort();
    limits.failing.dedup();
    for list in [&mut limits.other_driver, &mut limits.other_device] {
        list.sort();
        list.dedup();
    }
    limits
}

/// Lines under a heading, the breaches grouped by where they ran.
fn grouped(
    f: &mut fmt::Formatter<'_>,
    breaches: &[(String, String)],
    heading: impl Fn(&str) -> String,
) -> fmt::Result {
    let mut last = None;
    for (at, breach) in breaches {
        if last != Some(at) {
            writeln!(f, "\n{}", heading(at))?;
            last = Some(at);
        }
        writeln!(f, "  {breach}")?;
    }
    Ok(())
}

impl fmt::Display for Limits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.failing.is_empty() {
            writeln!(f, "\ncode limits broken:")?;
            for breach in &self.failing {
                writeln!(f, "  {breach}")?;
            }
        }
        grouped(f, &self.other_driver, |driver| {
            format!(
                "code limits broken under {driver}, not failing: they were measured under \
                 {MEASURED_DRIVER}, and a driver update moves stacks and spill, so re-measure them"
            )
        })?;
        grouped(f, &self.other_device, |device| {
            format!(
                "code limits broken on {device}, not failing: they were measured on \
                 {MEASURED_DEVICE} {MEASURED_DRIVER}"
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiles;

    #[test]
    fn breaches_fail_only_on_the_device_and_driver_measured() {
        let lines = |device: &str, driver: &str| {
            format!(
                "orr-compile kind=compute key=k device={device} driver={driver} wall_us=1 \
                 cpu_us=1 serial_us=1 longest_join_us=0 joins=0 stages=1 instructions=1 \
                 entries=orr_none.spills specs=- registers=255 local_bytes=64 shared_bytes=0 \
                 binary_bytes=1 name=spills\n\
                 orr-compile kind=compute key=k device={device} driver={driver} wall_us=1 \
                 cpu_us=1 serial_us=1 longest_join_us=0 joins=0 stages=1 instructions=1 \
                 entries=test_suite.spills specs=- registers=255 local_bytes=64 shared_bytes=0 \
                 binary_bytes=1 name=a test's own\n\
                 orr-compile kind=ray key=k device={device} driver={driver} wall_us=1 cpu_us=1 \
                 serial_us=1 longest_join_us=0 joins=0 stages=1 instructions=1 \
                 stacks=no_such_stage.copies.surface3:16 name=path pipeline 0x1\n"
            )
        };
        let check_on = |device: &str, driver: &str| {
            let records = compiles::read("t", &lines(device, driver));
            assert_eq!(records.unread, 0);
            check(records.compiles.iter())
        };
        let here = check_on(MEASURED_DEVICE, MEASURED_DRIVER);
        assert!(here.fails());
        assert_eq!(here.failing.len(), 2, "{:?}", here.failing);
        assert!(
            here.failing
                .iter()
                .any(|b| b.contains("no budget for no_such_stage.copies"))
        );
        let updated = check_on(MEASURED_DEVICE, "other_driver");
        assert!(!updated.fails());
        assert_eq!(updated.other_driver.len(), 2);
        let elsewhere = check_on("other_device", MEASURED_DRIVER);
        assert!(!elsewhere.fails());
        assert_eq!(elsewhere.other_device.len(), 2);
    }
}
