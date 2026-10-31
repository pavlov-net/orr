//! The dense vegetation test scene's camera scripts with per-pass timings: renders each
//! script, prints the median and 90th percentile GPU time of every pass and the render
//! thread's time per step, and writes every figure the frames report to
//! `target/orr-reports/dense_vegetation-<git rev>.json`.
//!
//! `bench [--synthetic] [--script NAME] [--script-frames N] [--headless] [scene switches...]`
//! (see `examples::vegetation`). It runs all four scripts for 600 frames each unless told
//! otherwise, in a window unless `--headless`, at 1920 x 1080 unless `--extent` says.
#![allow(clippy::print_stdout)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use examples::vegetation::{COUNTS, Run, Sample, Script, Settings, run, settled};
use examples::{Options, assert_clean, device};
use orr::orr_gpu::MemoryKind;
use orr::{FrameStats, LightingStats};
use serde_json::{Map, Value, json};

const SCRIPT_FRAMES: u64 = 600;

fn main() {
    let options = Options::from_args();
    let mut settings = Settings::parse(&options.rest);
    if !options.rest.iter().any(|a| a == "--script") {
        settings.scripts = Script::ALL.to_vec();
    }
    settings.script_frames = settings.script_frames.or(Some(SCRIPT_FRAMES));
    let r = run(&options, &settings);
    for (script, samples) in &r.scripts {
        print_script(*script, settled(samples));
    }
    let path = write_report(&settings, &r);
    println!("\nwrote {}", path.display());
    assert_clean();
}

fn median(v: &[f64]) -> f64 {
    percentile(v, 0.5)
}

fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// GPU ms per pass name over `samples`, one value per sample (0 where a frame ran none).
fn pass_times(samples: &[Sample]) -> BTreeMap<&'static str, Vec<f64>> {
    let mut out: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    for (i, s) in samples.iter().enumerate() {
        for p in &s.stats.gpu {
            let v = out.entry(p.name).or_default();
            v.resize(i + 1, 0.0);
            v[i] += p.gpu_ms;
        }
    }
    for v in out.values_mut() {
        v.resize(samples.len(), 0.0);
    }
    out
}

fn cpu_times(samples: &[Sample]) -> BTreeMap<&'static str, Vec<f64>> {
    let mut out: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    for s in samples {
        for &(name, d) in &s.stats.cpu {
            out.entry(name).or_default().push(d.as_secs_f64() * 1e3);
        }
    }
    out
}

fn frame_gpu_ms(samples: &[Sample]) -> Vec<f64> {
    samples
        .iter()
        .map(|s| s.stats.gpu.iter().map(|p| p.gpu_ms).sum())
        .collect()
}

fn print_script(script: Script, samples: &[Sample]) {
    println!(
        "\n{}: {} frames, GPU frame {:.3} ms median, {:.3} ms p90",
        script.name(),
        samples.len(),
        median(&frame_gpu_ms(samples)),
        percentile(&frame_gpu_ms(samples), 0.9)
    );
    let mut passes: Vec<(&str, f64, f64)> = pass_times(samples)
        .into_iter()
        .map(|(n, v)| (n, median(&v), percentile(&v, 0.9)))
        .collect();
    passes.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("{:>36} {:>9} {:>9}", "pass", "median", "p90");
    for (name, m, p) in passes {
        println!("{name:>36} {m:>9.3} {p:>9.3}");
    }
    println!("{:>36} {:>9} {:>9}", "render thread step", "median", "p90");
    for (name, v) in cpu_times(samples) {
        println!(
            "{name:>36} {:>9.3} {:>9.3}",
            median(&v),
            percentile(&v, 0.9)
        );
    }
}

/// Median, 90th percentile and maximum of `v`.
fn spread(v: &[f64]) -> Value {
    json!({
        "median": median(v),
        "p90": percentile(v, 0.9),
        "max": v.iter().copied().fold(0.0, f64::max),
    })
}

/// `spread` of one figure of each sample.
fn figure(samples: &[Sample], f: impl Fn(&Sample) -> f64) -> Value {
    spread(&samples.iter().map(f).collect::<Vec<_>>())
}

/// `spread` of each named series.
fn spreads(series: BTreeMap<&'static str, Vec<f64>>) -> Value {
    Value::Object(
        series
            .into_iter()
            .map(|(name, v)| (name.to_owned(), spread(&v)))
            .collect(),
    )
}

/// `RasterView::counts` by name; the fourth word is a constant 1.
const RASTER_COUNTS: [(usize, &str); COUNTS - 1] = [
    (0, "visible_clusters"),
    (1, "visible_dropped"),
    (2, "work_dropped"),
    (4, "work_items"),
    (5, "mesh_opaque_draws"),
    (6, "mesh_alpha_draws"),
    (7, "software_clusters"),
    (8, "particle_clusters"),
];

fn script_report(script: Script, samples: &[Sample]) -> Value {
    let fig = |f: &dyn Fn(&FrameStats) -> f64| figure(samples, |s| f(&s.stats));
    let lit =
        |f: &dyn Fn(&LightingStats) -> f64| figure(samples, |s| s.lighting.as_ref().map_or(0.0, f));
    let raster: Map<String, Value> = RASTER_COUNTS
        .iter()
        .map(|&(i, name)| (name.to_owned(), figure(samples, |s| s.counts[i].into())))
        .collect();
    json!({
        "script": script.name(),
        "frames": samples.len(),
        "gpu_frame_ms": spread(&frame_gpu_ms(samples)),
        "gpu_pass_ms": spreads(pass_times(samples)),
        "render_thread_ms": spreads(cpu_times(samples)),
        "upload_bytes": fig(&|s| s.upload_bytes as f64),
        "upload_high_water": fig(&|s| s.upload_high_water as f64),
        "scatter_words": fig(&|s| s.scatter_words as f64),
        "ptlas": {
            "writes": fig(&|s| s.rt.writes.into()),
            "updates": fig(&|s| s.rt.updates.into()),
            "translations": fig(&|s| s.rt.translations.into()),
            "rebuilt": fig(&|s| s.rt.rebuilt.into()),
            "deferred": fig(&|s| s.rt.deferred.into()),
            "max_age": fig(&|s| s.rt.max_age.into()),
            "dropped": fig(&|s| s.rt.dropped.into()),
            "global": fig(&|s| s.rt.global.into()),
            "unplaced": fig(&|s| s.rt.unplaced.into()),
            "partitions": fig(&|s| s.rt.partitions.into()),
            "bytes": fig(&|s| s.rt.bytes as f64),
            "part_instances": fig(&|s| s.rt.parts.into()),
            "parts_refused": fig(&|s| s.rt.parts_refused.into()),
            "masks_refused": fig(&|s| s.rt.masks_refused.into()),
        },
        "producers": {
            "clusters": fig(&|s| s.producers.clusters.into()),
            "cluster_capacity": fig(&|s| s.producers.cluster_capacity.into()),
            "dropped_clusters": fig(&|s| s.producers.dropped_clusters.into()),
            "vertices": fig(&|s| s.producers.vertices.into()),
            "dropped_vertices": fig(&|s| s.producers.dropped_vertices.into()),
            "refused": fig(&|s| s.producers.refused.into()),
            "built_clas": fig(&|s| s.producers.built.into()),
        },
        "raster": raster,
        "lighting": {
            "directional_lights": lit(&|l| l.directional_lights.into()),
            "local_lights": lit(&|l| l.local_lights.into()),
            "emissive_entries": lit(&|l| l.emissive_entries.into()),
            "emissive_clusters": lit(&|l| l.emissive_clusters.into()),
            "dropped_entries": lit(&|l| l.dropped_entries.into()),
            "environment_probability": lit(&|l| l.environment_probability.into()),
            "cache_occupied": lit(&|l| l.world_cache.occupied.into()),
            "cache_queries": lit(&|l| l.world_cache.queries.into()),
            "cache_hits": lit(&|l| l.world_cache.hits.into()),
            "cache_updates": lit(&|l| l.world_cache.updates.into()),
        },
    })
}

/// `git rev-parse --short HEAD` in the repository, or "unknown".
fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".into(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        )
}

fn write_report(settings: &Settings, r: &Run) -> PathBuf {
    let caps = device().caps();
    let rev = git_rev();
    let memory: Map<String, Value> = [
        ("upload", MemoryKind::Upload),
        ("device_local", MemoryKind::DeviceLocal),
        ("readback", MemoryKind::Readback),
    ]
    .into_iter()
    .map(|(name, kind)| (format!("{name}_bytes"), r.memory.kind(kind).bytes.into()))
    .collect();
    let scripts: Vec<Value> = r
        .scripts
        .iter()
        .map(|(s, samples)| script_report(*s, settled(samples)))
        .collect();
    let report = json!({
        "rev": rev,
        "device": caps.device_name,
        "driver": caps.driver,
        "settings": format!("{settings:?}"),
        "trees": r.tree_count,
        "terrain_patches": r.terrain_patches,
        "load_ms": r.load_ms,
        "memory": memory,
        "scripts": scripts,
    });
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("target")
        .join("orr-reports");
    std::fs::create_dir_all(&dir).expect("report directory");
    let path = dir.join(format!("dense_vegetation-{rev}.json"));
    let text = serde_json::to_string_pretty(&report).expect("report")
        + "
";
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    path
}
