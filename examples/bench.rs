//! The dense vegetation test scene's camera scripts with per-pass timings: renders each
//! script, prints the median and 90th percentile GPU time of every pass and the render
//! thread's time per step, and writes every figure the frames report to
//! `target/orr-reports/dense_vegetation-<git rev>.json`. Built with the `nsight-perf` feature,
//! on a device with `Caps::perf_counters`, it also collects per-pass Nsight Perf SDK counters
//! for one frame of the first script (the middle one unless `--counter-frame N` says), prints
//! them as a table and adds them to the report.
//!
//! `bench [--synthetic] [--script NAME] [--script-frames N] [--headless] [scene switches...]`
//! (see `orr_examples::vegetation`). It runs all four scripts for 600 frames each unless told
//! otherwise, in a window unless `--headless`, at 1920 x 1080 unless `--extent` says.
#![allow(clippy::print_stdout)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use orr::orr_gpu::MemoryKind;
#[cfg(feature = "nsight-perf")]
use orr::orr_gpu::{CounterMetric, PassCounters, PassResult};
use orr::{FrameStats, LightingStats};
#[cfg(feature = "nsight-perf")]
use orr_examples::request_perf_counters;
use orr_examples::vegetation::{Run, Sample, Script, Settings, run, settled};
use orr_examples::{Options, assert_clean, device, git_rev, report_dir};
use serde_json::{Map, Value, json};

const SCRIPT_FRAMES: u64 = 600;

fn main() {
    #[cfg(feature = "nsight-perf")]
    request_perf_counters();
    let options = Options::from_args();
    let mut settings = Settings::parse(&options.rest);
    if !options.rest.iter().any(|a| a == "--script") {
        settings.scripts = Script::ALL.to_vec();
    }
    settings.script_frames = settings.script_frames.or(Some(SCRIPT_FRAMES));
    #[cfg(feature = "nsight-perf")]
    {
        let frames = options
            .frames
            .or(settings.script_frames)
            .unwrap_or(SCRIPT_FRAMES);
        settings.counter_frame = settings.counter_frame.or(Some(frames / 2));
    }
    let r = run(&options, &settings);
    for (script, samples) in &r.scripts {
        print_script(*script, settled(samples));
    }
    #[cfg(feature = "nsight-perf")]
    match &r.counters {
        Some((metrics, passes)) => print_counters(metrics, passes),
        None => println!("\nno counters: the log says why"),
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

/// The counter frame's passes, one row each: GPU ms and each metric's value, `-` for a metric
/// the device doesn't expose, then the metrics by column.
#[cfg(feature = "nsight-perf")]
fn print_counters(metrics: &[CounterMetric], passes: &[PassResult]) {
    println!("\ncounters, one row per pass:");
    print!("{:>36} {:>9}", "pass", "gpu ms");
    for i in 0..metrics.len() {
        print!(" {:>9}", format!("[{i}]"));
    }
    println!();
    for p in passes {
        print!("{:>36} {:>9.3}", p.name, p.gpu_ms);
        match &p.counters {
            Some(PassCounters::Values(values)) => {
                for v in values {
                    match v {
                        Some(v) => print!(" {v:>9.3}"),
                        None => print!(" {:>9}", "-"),
                    }
                }
            }
            Some(PassCounters::Dropped) => print!(" dropped"),
            None => print!(" none"),
        }
        println!();
    }
    for (i, m) in metrics.iter().enumerate() {
        let unit = if m.available {
            m.unit.as_str()
        } else {
            "not available"
        };
        println!("  [{i}] {} ({unit})", m.name);
    }
}

/// The counter frame's metrics and its passes, one row each with every metric by name; a
/// metric the device doesn't expose is `null`, and a pass whose range the SDK dropped says so.
#[cfg(feature = "nsight-perf")]
fn counter_report(metrics: &[CounterMetric], passes: &[PassResult]) -> Value {
    let rows: Vec<Value> = passes
        .iter()
        .map(|p| {
            let mut row = Map::new();
            row.insert("pass".into(), p.name.into());
            row.insert("gpu_ms".into(), p.gpu_ms.into());
            match &p.counters {
                Some(PassCounters::Values(values)) => {
                    for (m, v) in metrics.iter().zip(values) {
                        row.insert(m.name.clone(), (*v).into());
                    }
                }
                Some(PassCounters::Dropped) => {
                    row.insert("dropped".into(), true.into());
                }
                None => {}
            }
            Value::Object(row)
        })
        .collect();
    let metrics: Vec<Value> = metrics
        .iter()
        .map(|m| json!({ "name": m.name, "unit": m.unit, "available": m.available }))
        .collect();
    json!({ "metrics": metrics, "passes": rows })
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
const RASTER_COUNTS: [(usize, &str); 14] = [
    (0, "visible_clusters"),
    (1, "visible_dropped"),
    (2, "second_pass_work_dropped"),
    (4, "second_pass_work_items"),
    (5, "mesh_opaque_draws"),
    (6, "mesh_alpha_draws"),
    (7, "software_clusters"),
    (8, "particle_clusters"),
    (9, "occluded_instances"),
    (10, "occluded_clusters"),
    (11, "second_pass_clusters"),
    (12, "hidden_clusters"),
    (16, "layer_clusters"),
    (17, "smoke_clusters"),
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
        "blades": {
            "draw_items": figure(samples, |s| s.blades[0].into()),
            "caster_items": figure(samples, |s| s.blades[1].into()),
            "drawn": figure(samples, |s| s.blades[2].into()),
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
        "traversal": {
            "candidate_items": fig(&|s| s.cuts.candidate_items.into()),
            "candidate_clusters": fig(&|s| s.cuts.candidate_clusters.into()),
            "nodes": fig(&|s| s.cuts.nodes.into()),
            "nodes_dropped": fig(&|s| s.cuts.nodes_dropped.into()),
            "candidates_dropped": fig(&|s| s.cuts.candidates_dropped.into()),
            "rt_gathered": fig(&|s| s.cuts.rt_gathered.into()),
            "rt_cached": fig(&|s| s.cuts.rt_cached.into()),
            "rt_shared": fig(&|s| s.cuts.rt_shared.into()),
            "rt_fallback": fig(&|s| s.cuts.rt_fallback.into()),
            "rt_lost": fig(&|s| s.cuts.rt_lost.into()),
            "rt_empty": fig(&|s| s.cuts.rt_empty.into()),
            "rt_table_bytes": fig(&|s| s.cuts.rt_table_bytes as f64),
            "rt_built_blas": fig(&|s| s.cuts.rt_built.into()),
            "rt_built_clas": fig(&|s| s.cuts.rt_built_clas.into()),
            "rt_merged": fig(&|s| s.cuts.rt_merged.into()),
            "page_requests": fig(&|s| s.cuts.page_requests.into()),
        },
        "lighting": {
            "directional_lights": lit(&|l| l.directional_lights.into()),
            "local_lights": lit(&|l| l.local_lights.into()),
            "emissive_entries": lit(&|l| l.emissive_entries.into()),
            "emissive_clusters": lit(&|l| l.emissive_clusters.into()),
            "environment_probability": lit(&|l| l.environment_probability.into()),
            "cache_occupied": lit(&|l| l.world_cache.occupied.into()),
            "cache_queries": lit(&|l| l.world_cache.queries.into()),
            "cache_hits": lit(&|l| l.world_cache.hits.into()),
            "cache_updates": lit(&|l| l.world_cache.updates.into()),
        },
    })
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
        "reconstruction": r.reconstruction,
        "render_extent": r.render_extent,
        "blade_capacity": r.blades.map(|(c, bytes)| json!({
            "draw_items": c.items,
            "caster_items": c.caster_items,
            "caster_vertices": c.vertices,
            "drawn_blades": c.blades,
            "list_bytes": bytes,
        })),
        "trees": r.tree_count,
        "terrain_patches": r.terrain_patches,
        "load_ms": r.load_ms,
        "memory": memory,
        "scripts": scripts,
    });
    #[cfg(feature = "nsight-perf")]
    let report = {
        let mut report = report;
        report["counters"] = r
            .counters
            .as_ref()
            .map_or(Value::Null, |(metrics, passes)| {
                counter_report(metrics, passes)
            });
        report
    };
    let path = report_dir().join(format!("dense_vegetation-{rev}.json"));
    let text = serde_json::to_string_pretty(&report).expect("report")
        + "
";
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    path
}
