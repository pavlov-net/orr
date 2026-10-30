//! Probe: the render thread and the present path (gpu.md G0d, output.md 7 present probe).
//!
//! `present <case>`:
//! - `info`: surface formats, present modes and present-timing capabilities of a real
//!   full-screen window against the vulkaninfo capture; swapchain timing properties (VRR) per
//!   window kind, present mode and encoding.
//! - `thread [workers]`: render-thread CPU per step with a 4x metered batch, NGX RR and FG
//!   evaluate calls and the synthetic streaming load; time blocked in acquire and present.
//! - `flip`: SDR, PQ and scRGB full-screen segments, then 4x metered batches, with QPC segment
//!   marks for a PresentMon capture (see `presentmon.ps1`).
//! - `reflex`: low_latency2 markers keyed three ways around a 4x batch; vkGetLatencyTimingsNV
//!   reports checked against the keys.
//! - `pacing`: spacing of a 4x batch with present metering, with relative-time presents and
//!   unpaced, measured with present timing (first pixel out) and present_wait2.
//! - `hdrmeta [seconds]`: PQ peak-white pattern, MaxCLL alternating 400 and 4000 nits, for a
//!   by-eye check.
//!
//! Each case prints `ROW case | metric | value` lines and a `RESULT` line.
//! Build: `DLSS_SDK=<sdk> cargo build --release --target-dir ../../target/probes`.

#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments, clippy::missing_safety_doc)]

#[cfg(ngx)]
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, unused_imports, clippy::all)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/ngx.rs"));
}
mod frame;
mod gpu;
mod load;
#[cfg(ngx)]
mod ngx;
mod present;
mod win;

use ash::vk;
use frame::{Cfg, Engine, Key, Steps};
use present::{Chain, Opts};
use std::sync::Mutex;

#[macro_export]
macro_rules! spv {
    ($name:literal) => {
        ash::util::read_spv(&mut std::io::Cursor::new(&include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))[..])).expect("SPIR-V")
    };
}

static CASE: Mutex<String> = Mutex::new(String::new());

pub fn row(metric: &str, value: impl std::fmt::Display) {
    let case = CASE.lock().unwrap().clone();
    println!("ROW {case} | {metric} | {value}");
}

pub fn median(v: &mut [f64]) -> f64 {
    pct(v, 0.5)
}

pub fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// SplitMix64.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

pub fn quick() -> bool {
    std::env::var("PROBE_QUICK").is_ok_and(|v| v == "1")
}

fn frames(n: usize) -> usize {
    if quick() { n.min(40) } else { n }
}

struct Ctx {
    gpu: &'static gpu::Gpu,
    win: win::Window,
    surface: vk::SurfaceKHR,
}

unsafe fn setup(fullscreen: bool) -> Ctx {
    #[cfg(ngx)]
    let (ie, de) = ngx::required_extensions();
    #[cfg(not(ngx))]
    let (ie, de): (Vec<std::ffi::CString>, Vec<std::ffi::CString>) = (vec![], vec![]);
    let gpu: &'static gpu::Gpu = Box::leak(Box::new(gpu::Gpu::new(&ie, &de)));
    println!("DEVICE {} driver {} families render={} loading={} validation={}", gpu.name, gpu.driver, gpu.families[0], gpu.families[1], gpu.validation);
    if !ie.is_empty() || !de.is_empty() {
        row("ngx extensions", format!("instance {ie:?}, device {de:?}"));
    }
    let win = win::Window::new(fullscreen);
    let surface = gpu.surface(&win);
    Ctx { gpu, win, surface }
}

/// Presents `n` light frames (one image each) so timing properties settle.
unsafe fn settle(ctx: &mut Ctx, eng: &mut Engine, chain: &mut Chain, id: &mut u64, n: usize) {
    let cfg = Cfg { batch: 1, metered: false, content: false, ngx: false, sleep: false, markers: false, ..Cfg::default() };
    for _ in 0..n {
        ctx.win.pump();
        eng.frame(ctx.gpu, chain, *id, &cfg);
        *id += 1;
    }
}

unsafe fn refresh_ns(ctx: &Ctx, chain: &Chain) -> u64 {
    let (d, _, r) = chain.timing_properties(ctx.gpu);
    if r == vk::Result::SUCCESS && d > 0 { d } else { 6_060_606 }
}

fn interval_text(d: u64, i: u64, r: vk::Result) -> String {
    let kind = if i == u64::MAX {
        "VRR"
    } else if i == 0 {
        "unknown"
    } else if i == d {
        "FRR"
    } else {
        "FRR (interval divides duration)"
    };
    let iv = if i == u64::MAX { "UINT64_MAX".to_string() } else { i.to_string() };
    format!("refreshDuration {d} ns ({:.3} Hz), refreshInterval {iv} -> {kind} ({r:?})", if d > 0 { 1e9 / d as f64 } else { 0.0 })
}

// ---------------------------------------------------------------------------------------------
// info

fn vulkaninfo_formats() -> Vec<(String, String)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../.tools/display/vulkaninfo-621.01.txt");
    let Ok(text) = std::fs::read_to_string(path) else { return vec![] };
    let mut out = Vec::new();
    let mut in_surf = false;
    let mut fmt = String::new();
    for line in text.lines() {
        let l = line.trim();
        if l.starts_with("Presentable Surfaces") {
            in_surf = true;
        }
        if in_surf && l.starts_with("Present Modes") {
            break;
        }
        if in_surf {
            if let Some(f) = l.strip_prefix("format = FORMAT_") {
                fmt = f.to_string();
            }
            if let Some(c) = l.strip_prefix("colorSpace = COLOR_SPACE_") {
                out.push((fmt.clone(), c.to_string()));
            }
        }
    }
    out
}

unsafe fn info() {
    let mut ctx = setup(true);
    let gpu = ctx.gpu;
    let formats = gpu.surface_i.get_physical_device_surface_formats(gpu.pdev, ctx.surface).unwrap();
    let mine: Vec<(String, String)> = formats.iter().map(|f| (format!("{:?}", f.format), format!("{:?}", f.color_space))).collect();
    for (f, c) in &mine {
        row("surface format (full-screen window)", format!("{f} / {c}"));
    }
    let theirs = vulkaninfo_formats();
    row("surface format count", format!("full-screen window {}, vulkaninfo hidden window {}", mine.len(), theirs.len()));
    // ash prints e.g. B8G8R8A8_UNORM / SRGB_NONLINEAR; vulkaninfo prints B8G8R8A8_UNORM / SRGB_NONLINEAR_KHR.
    let key = |f: &str, c: &str| format!("{}|{}", f.trim_end_matches("_KHR"), c.trim_end_matches("_KHR").trim_end_matches("_EXT"));
    let mk: Vec<String> = mine.iter().map(|(f, c)| key(f, c)).collect();
    let tk: Vec<String> = theirs.iter().map(|(f, c)| key(f, c)).collect();
    row("surface formats only in full-screen window", format!("{:?}", mk.iter().filter(|m| !tk.contains(m)).collect::<Vec<_>>()));
    row("surface formats only in vulkaninfo capture", format!("{:?}", tk.iter().filter(|t| !mk.contains(t)).collect::<Vec<_>>()));
    let modes = gpu.surface_i.get_physical_device_surface_present_modes(gpu.pdev, ctx.surface).unwrap();
    row("present modes", format!("{modes:?}"));
    let (caps, tcaps, id2, wait2) = present::surface_caps(gpu, ctx.surface);
    row(
        "surface caps",
        format!(
            "images {}..{}, extent {}x{}, usage {:?}, presentId2 {id2}, presentWait2 {wait2}",
            caps.min_image_count, caps.max_image_count, caps.current_extent.width, caps.current_extent.height, caps.supported_usage_flags
        ),
    );
    row(
        "present timing caps",
        format!(
            "timing {}, absolute {}, relative {}, stages {:?}",
            tcaps.present_timing_supported, tcaps.present_at_absolute_time_supported, tcaps.present_at_relative_time_supported, tcaps.present_stage_queries
        ),
    );

    let mut id = 1u64;
    let mut chain = Chain::new(gpu, ctx.surface, present::SDR, &Opts::default(), None);
    let mut eng = Engine::new(gpu, &chain, false);
    for (fmt, mode) in [
        (present::SDR, vk::PresentModeKHR::FIFO),
        (present::SDR, vk::PresentModeKHR::IMMEDIATE),
        (present::PQ, vk::PresentModeKHR::FIFO),
        (present::SCRGB, vk::PresentModeKHR::FIFO),
    ] {
        eng.idle(gpu);
        chain = Chain::new(gpu, ctx.surface, fmt, &Opts { mode, ..Opts::default() }, Some(chain));
        eng.idle(gpu);
        settle(&mut ctx, &mut eng, &mut chain, &mut id, 240);
        let (d, i, r) = chain.timing_properties(gpu);
        let doms = chain.time_domains(gpu);
        row(&format!("timing properties full-screen {} {:?}", fmt.name, mode), interval_text(d, i, r));
        row(&format!("time domains full-screen {} {:?}", fmt.name, mode), format!("{doms:?}"));
        latency_stats(&ctx, &chain, &format!("full-screen {} {:?}", fmt.name, mode), vec![]);
    }
    eng.idle(gpu);
    chain.destroy(gpu);
    gpu.surface_i.destroy_surface(ctx.surface, None);
    drop(ctx.win);

    let win = win::Window::new(false);
    let surface = gpu.surface(&win);
    let mut ctx = Ctx { gpu, win, surface };
    let mut chain = Chain::new(gpu, ctx.surface, present::SDR, &Opts::default(), None);
    settle(&mut ctx, &mut eng, &mut chain, &mut id, 240);
    let (d, i, r) = chain.timing_properties(gpu);
    row("timing properties windowed SDR FIFO", interval_text(d, i, r));
    latency_stats(&ctx, &chain, "windowed SDR FIFO", vec![]);
    eng.idle(gpu);
    chain.destroy(gpu);
}

/// Present-timing feedback over recent presents: share of records with a first-pixel-out time,
/// queue-end to first-pixel-out latency, and the median interval of each stage.
unsafe fn latency_stats(ctx: &Ctx, chain: &Chain, label: &str, mut past: Vec<present::Past>) {
    past.extend(chain.past(ctx.gpu));
    let tick = ns_per_tick(chain) * 1e-6;
    let mut lat = Vec::new();
    let (mut qe, mut fpo) = (Vec::new(), Vec::new());
    for p in &past {
        let q = p.at(vk::PresentStageFlagsEXT::QUEUE_OPERATIONS_END);
        let f = p.at(vk::PresentStageFlagsEXT::IMAGE_FIRST_PIXEL_OUT);
        if let (Some(q), Some(f)) = (q, f) {
            lat.push((f as f64 - q as f64) * tick);
        }
        qe.extend(q);
        fpo.extend(f);
    }
    let iv = |v: &[u64]| median(&mut v.windows(2).map(|w| (w[1] as f64 - w[0] as f64) * tick).collect::<Vec<_>>());
    row(
        &format!("present timing, {label}"),
        format!(
            "{} records, {} with queue end, {} with first pixel out; queue end -> first pixel out median {:.3} ms; interval median queue end {:.3} ms, first pixel out {:.3} ms",
            past.len(),
            qe.len(),
            fpo.len(),
            median(&mut lat),
            iv(&qe),
            iv(&fpo)
        ),
    );
}

/// Nanoseconds per timestamp unit of the chain's time domain.
fn ns_per_tick(chain: &Chain) -> f64 {
    if chain.domain.is_some_and(|d| d.0 == vk::TimeDomainKHR::QUERY_PERFORMANCE_COUNTER) { 1e9 / win::qpc_raw().1 as f64 } else { 1.0 }
}

// ---------------------------------------------------------------------------------------------
// thread

#[derive(Default)]
struct Series {
    v: Vec<Steps>,
}

impl Series {
    fn stat(&self, name: &str, label: &str, f: impl Fn(&Steps) -> f64) {
        let mut c: Vec<f64> = self.v.iter().map(f).collect();
        let (m, p99, mx) = (median(&mut c), pct(&mut c, 0.99), pct(&mut c, 1.0));
        row(&format!("{label} {name}"), format!("median {m:.4} ms, p99 {p99:.4} ms, max {mx:.4} ms"));
    }
}

unsafe fn thread_case(args: &[String]) {
    let mut ctx = setup(true);
    let gpu = ctx.gpu;
    let workers: usize = args.first().and_then(|a| a.parse().ok()).unwrap_or(8);
    let mut chain = Chain::new(gpu, ctx.surface, present::SDR, &Opts { timing: false, ..Opts::default() }, None);
    let mut eng = Engine::new(gpu, &chain, true);
    let mut id = 1u64;
    settle(&mut ctx, &mut eng, &mut chain, &mut id, 60);
    let refresh = refresh_ns(&ctx, &chain);
    let (d, i, r) = chain.timing_properties(gpu);
    row("display", format!("{}x{} {}", chain.extent[0], chain.extent[1], interval_text(d, i, r)));
    eng.calibrate(gpu, 8.0);
    row("heavy iters", eng.heavy_iters);
    let ngx_on = eng.ngx.is_some();
    #[cfg(ngx)]
    let fg_on = eng.ngx.as_ref().is_some_and(|n| n.fg_ok);
    #[cfg(not(ngx))]
    let fg_on = false;
    row("ngx", format!("RR {ngx_on}, FG {fg_on}"));
    let cap_us = (4 * refresh / 1000) as u32;
    row("thread cycles per ms", format!("{:.0}", eng.cycles_per_ms));
    // (label, workers, BAR writes, decoded GB/s limit, cap, render thread at highest priority)
    let all: [(&str, usize, bool, f64, u32, bool); 9] = [
        ("cap+load", workers, true, 0.0, cap_us, false),
        ("cap", 0, true, 0.0, cap_us, false),
        ("cap+cached load", workers, false, 0.0, cap_us, false),
        ("cap+load high priority", workers, true, 0.0, cap_us, true),
        ("cap+4 workers", 4, true, 0.0, cap_us, false),
        ("nocap+load", workers, true, 0.0, 0, false),
        ("cap+load 7 GB/s", workers, true, 7.0, cap_us, false),
        ("cap+load 14 GB/s", workers, true, 14.0, cap_us, false),
        ("cap+load 21 GB/s", workers, true, 21.0, cap_us, false),
    ];
    let pick: Vec<usize> = match args.get(1).map(String::as_str) {
        Some("rate") => vec![1, 6, 7, 8],
        _ => (0..6).collect(),
    };
    for (label, w, bar, gbps, cap, high) in pick.into_iter().map(|i| all[i]) {
        chain.sleep_mode(gpu, true, false, cap);
        win::set_priority_highest(high);
        let load = if w > 0 { Some(load::Load::start(gpu, w, bar, gbps)) } else { None };
        let cfg = Cfg::default();
        for _ in 0..frames(120) {
            ctx.win.pump();
            eng.frame(gpu, &mut chain, id, &cfg);
            id += 1;
        }
        eng.gpu_ms.clear();
        let mut s = Series::default();
        let t0 = std::time::Instant::now();
        for _ in 0..frames(900) {
            if !ctx.win.pump() {
                break;
            }
            s.v.push(eng.frame(gpu, &mut chain, id, &cfg));
            id += 1;
        }
        let wall = t0.elapsed().as_secs_f64();
        let lr = load.map(load::Load::stop);
        let mut g = eng.gpu_ms.clone();
        row(
            &format!("{label} setup"),
            format!(
                "{} frames in {:.2} s ({:.2} fps base), cap {cap} us, workers {w}, load {}, GPU main list median {:.3} ms",
                s.v.len(),
                wall,
                s.v.len() as f64 / wall,
                lr.map_or("off".to_string(), |(g, c)| format!("{g:.2} GB/s into Upload, {c:.0} loading copies/s")),
                median(&mut g)
            ),
        );
        s.stat("slot wait", label, |x| x.slot_wait);
        s.stat("reflex sleep", label, |x| x.sleep);
        s.stat("record scene (50 passes)", label, |x| x.record_scene);
        s.stat("NGX RR evaluate", label, |x| x.ngx_rr);
        s.stat("record output (10 passes + compose)", label, |x| x.record_output);
        s.stat("NGX FG evaluate x3", label, |x| x.ngx_fg);
        s.stat("main submit", label, |x| x.submit);
        s.stat("acquire x4", label, |x| x.acquire);
        s.stat("acquire max single", label, |x| x.acquire_max);
        s.stat("compose-UI record x4", label, |x| x.compose);
        s.stat("present submit x4", label, |x| x.present_submit);
        s.stat("vkQueuePresentKHR x4", label, |x| x.present);
        s.stat("present max single", label, |x| x.present_max);
        s.stat("blocked in present path (acquire + present)", label, |x| x.acquire + x.present);
        s.stat("CPU run inside acquire + present (thread cycles)", label, |x| x.present_cpu);
        s.stat("waiting inside acquire + present (wall - CPU)", label, |x| (x.acquire + x.present - x.present_cpu).max(0.0));
        s.stat("present path (PresentStart..End)", label, |x| x.present_path);
        s.stat("output share: record output + RR + FG", label, |x| x.record_output + x.ngx_rr + x.ngx_fg);
        s.stat("output share: present path", label, |x| x.present_path);
        s.stat("orr-gpu share: main submit + acquires + present submits + presents", label, |x| x.submit + x.acquire + x.present_submit + x.present);
        s.stat("render thread total (excl. slot wait, sleep)", label, |x| x.total);
        let fails: u32 = s.v.iter().map(|x| x.ngx_fail).sum();
        let pfails: u32 = s.v.iter().map(|x| x.present_fail).sum();
        let mut fb: Vec<u32> = s.v.iter().map(|x| x.feedback).collect();
        fb.sort_unstable();
        fb.dedup();
        win::set_priority_highest(false);
        row(&format!("{label} failures"), format!("NGX evaluate failures {fails}, present/acquire failures {pfails}, presentConfigFeedback values {fb:?}"));
    }
    eng.idle(gpu);
    #[cfg(ngx)]
    if let Some(n) = eng.ngx.take() {
        n.shutdown(gpu);
    }
    chain.destroy(gpu);
}

// ---------------------------------------------------------------------------------------------
// flip

unsafe fn flip() {
    let mut ctx = setup(true);
    let gpu = ctx.gpu;
    let (c0, f) = win::qpc_raw();
    row("qpc", format!("frequency {f}, start {c0}"));
    let mut chain = Chain::new(gpu, ctx.surface, present::SDR, &Opts::default(), None);
    let mut eng = Engine::new(gpu, &chain, false);
    let mut id = 1u64;
    settle(&mut ctx, &mut eng, &mut chain, &mut id, 30);
    let refresh = refresh_ns(&ctx, &chain);
    let secs = if quick() { 0.5 } else { 5.0 };
    let segs: [(present::Format, u32, &str); 6] = [
        (present::SDR, 1, "SDR"),
        (present::PQ, 1, "PQ"),
        (present::SCRGB, 1, "scRGB"),
        (present::SDR, 4, "SDR 4x metered"),
        (present::PQ, 4, "PQ 4x metered"),
        (present::SCRGB, 4, "scRGB 4x metered"),
    ];
    for (fmt, batch, label) in segs {
        eng.idle(gpu);
        chain = Chain::new(gpu, ctx.surface, fmt, &Opts::default(), Some(chain));
        eng.idle(gpu);
        let cap = if batch > 1 { (u64::from(batch) * refresh / 1000) as u32 } else { 0 };
        chain.sleep_mode(gpu, true, false, cap);
        let cfg = Cfg { batch, metered: batch > 1, content: false, ngx: false, markers: true, ..Cfg::default() };
        for _ in 0..30 {
            ctx.win.pump();
            eng.frame(gpu, &mut chain, id, &cfg);
            id += 1;
            let _ = chain.past(gpu);
        }
        let (a, _) = win::qpc_raw();
        let t0 = std::time::Instant::now();
        let mut fails = 0;
        let mut past = Vec::new();
        while t0.elapsed().as_secs_f64() < secs {
            ctx.win.pump();
            fails += eng.frame(gpu, &mut chain, id, &cfg).present_fail;
            id += 1;
            past.extend(chain.past(gpu));
        }
        let (b, _) = win::qpc_raw();
        println!("SEGMENT {label} qpc {a} {b}");
        let (d, i, r) = chain.timing_properties(gpu);
        row(&format!("{label} timing properties"), interval_text(d, i, r));
        row(&format!("{label} present failures"), fails);
        latency_stats(&ctx, &chain, label, past);
    }
    eng.idle(gpu);
    chain.destroy(gpu);
}

// ---------------------------------------------------------------------------------------------
// reflex

unsafe fn reflex() {
    let mut ctx = setup(true);
    let gpu = ctx.gpu;
    let mut chain = Chain::new(gpu, ctx.surface, present::SDR, &Opts { timing: false, ..Opts::default() }, None);
    let mut eng = Engine::new(gpu, &chain, false);
    let mut id = 1u64;
    settle(&mut ctx, &mut eng, &mut chain, &mut id, 30);
    let refresh = refresh_ns(&ctx, &chain);
    eng.calibrate(gpu, 4.0);
    chain.sleep_mode(gpu, true, false, (4 * refresh / 1000) as u32);
    for key in [Key::Frame, Key::LastPresent, Key::FirstPresent] {
        let cfg = Cfg { key, ngx: false, ..Cfg::default() };
        let mut batches = Vec::new();
        for _ in 0..frames(240) {
            ctx.win.pump();
            let first = chain.next_id;
            let k = match key {
                Key::Frame => id,
                Key::LastPresent => first + 3,
                Key::FirstPresent => first,
            };
            eng.frame(gpu, &mut chain, id, &cfg);
            batches.push((k, first, chain.next_id - 1));
            id += 1;
        }
        eng.idle(gpu);
        std::thread::sleep(std::time::Duration::from_millis(200));
        let reps = chain.latency_timings(gpu);
        let used: std::collections::HashSet<u64> = batches.iter().map(|b| b.0).collect();
        let all_ids: std::collections::HashSet<u64> = batches.iter().flat_map(|b| b.1..=b.2).collect();
        let (mut matched, mut complete, mut ordered) = (0, 0, 0);
        let mut foreign = Vec::new();
        let mut zero = [0u32; 13];
        // Reports from this run only: keyed by one of its marker values, or by one of its present
        // ids that is not a marker value (a per-present report).
        let lo = batches.first().map_or(0, |b| b.0.min(b.1));
        let reps: Vec<_> = reps.into_iter().filter(|r| r.present_id >= lo || r.present_id == 0).collect();
        for r in &reps {
            if used.contains(&r.present_id) {
                matched += 1;
            } else {
                foreign.push((r.present_id, all_ids.contains(&r.present_id)));
            }
            let f = [
                r.input_sample_time_us,
                r.sim_start_time_us,
                r.sim_end_time_us,
                r.render_submit_start_time_us,
                r.render_submit_end_time_us,
                r.present_start_time_us,
                r.present_end_time_us,
                r.driver_start_time_us,
                r.driver_end_time_us,
                r.os_render_queue_start_time_us,
                r.os_render_queue_end_time_us,
                r.gpu_render_start_time_us,
                r.gpu_render_end_time_us,
            ];
            for (k, v) in f.iter().enumerate() {
                if *v == 0 {
                    zero[k] += 1;
                }
            }
            if f[1..].iter().all(|&v| v != 0) {
                complete += 1;
            }
            if r.sim_start_time_us <= r.sim_end_time_us
                && r.sim_end_time_us <= r.render_submit_start_time_us
                && r.render_submit_start_time_us <= r.render_submit_end_time_us
                && r.render_submit_end_time_us <= r.present_start_time_us
                && r.present_start_time_us <= r.present_end_time_us
                && r.gpu_render_start_time_us <= r.gpu_render_end_time_us
            {
                ordered += 1;
            }
        }
        row(
            &format!("{key:?} keys"),
            format!(
                "{} reports; {matched} keyed by a marker value, {} by another value (value, is a present id) {:?}; {complete} with every timestamp but input sample nonzero; {ordered} ordered",
                reps.len(),
                foreign.len(),
                &foreign[..foreign.len().min(6)]
            ),
        );
        row(
            &format!("{key:?} zero fields"),
            format!(
                "input {}, simS {}, simE {}, subS {}, subE {}, presS {}, presE {}, drvS {}, drvE {}, osQS {}, osQE {}, gpuS {}, gpuE {}",
                zero[0], zero[1], zero[2], zero[3], zero[4], zero[5], zero[6], zero[7], zero[8], zero[9], zero[10], zero[11], zero[12]
            ),
        );
        for r in reps.iter().rev().take(2) {
            let b = r.sim_start_time_us as i64;
            let d = |v: u64| v as i64 - b;
            row(
                &format!("{key:?} sample"),
                format!(
                    "id {}: simE {} subS {} subE {} presS {} presE {} drvS {} drvE {} osQS {} osQE {} gpuS {} gpuE {} (us after sim start)",
                    r.present_id,
                    d(r.sim_end_time_us),
                    d(r.render_submit_start_time_us),
                    d(r.render_submit_end_time_us),
                    d(r.present_start_time_us),
                    d(r.present_end_time_us),
                    d(r.driver_start_time_us),
                    d(r.driver_end_time_us),
                    d(r.os_render_queue_start_time_us),
                    d(r.os_render_queue_end_time_us),
                    d(r.gpu_render_start_time_us),
                    d(r.gpu_render_end_time_us),
                ),
            );
        }
        if let Some(r) = reps.last() {
            if let Some(b) = batches.iter().find(|b| b.0 == r.present_id) {
                row(&format!("{key:?} last report batch"), format!("marker value {}, present ids {}..={}", b.0, b.1, b.2));
            }
        }
    }
    eng.idle(gpu);
    chain.destroy(gpu);
}

// ---------------------------------------------------------------------------------------------
// pacing

unsafe fn pacing() {
    let mut ctx = setup(true);
    let gpu = ctx.gpu;
    // Stage queries are off (they stall present_wait2 on 621.01); `PROBE_STAGE_QUERIES=1` turns
    // them on to show that.
    let queries = std::env::var("PROBE_STAGE_QUERIES").is_ok();
    let mut chain = Chain::new(gpu, ctx.surface, present::SDR, &Opts { queries, ..Opts::default() }, None);
    let mut eng = Engine::new(gpu, &chain, false);
    let mut id = 1u64;
    settle(&mut ctx, &mut eng, &mut chain, &mut id, 60);
    let refresh = refresh_ns(&ctx, &chain);
    let (d, i, r) = chain.timing_properties(gpu);
    row("display", interval_text(d, i, r));
    row("time domain", format!("{:?} of {:?}", chain.domain, chain.time_domains(gpu)));
    chain.sleep_mode(gpu, true, false, 0);
    let qpc_ns = 1e9 / win::qpc_raw().1 as f64;
    for period in [4 * refresh, 30_000_000, 5 * refresh] {
        for mode in ["metered", "relative", "unpaced"] {
            let target = period / 4;
            let cfg = Cfg {
                batch: 4,
                metered: mode == "metered",
                relative_ns: if mode == "relative" { target } else { 0 },
                content: false,
                ngx: false,
                sleep: false,
                markers: true,
                ..Cfg::default()
            };
            let _ = chain.past(gpu);
            let mut past = Vec::new();
            let mut firsts = std::collections::HashSet::new();
            // A waiter thread blocks in vkWaitForPresent2KHR on each id in order and stamps the
            // return. The swapchain is externally synchronised per the spec; the probe shares it
            // with the render thread's acquire and present anyway, only to timestamp.
            let (tx, rx) = std::sync::mpsc::channel::<u64>();
            let sc = chain.sc;
            let waiter = std::thread::spawn(move || {
                let mut done: Vec<(u64, f64)> = Vec::new();
                while let Ok(p) = rx.recv() {
                    let w = vk::PresentWait2InfoKHR::default().present_id(p).timeout(500_000_000);
                    match gpu.wait2.wait_for_present2(sc, &w) {
                        Ok(()) | Err(vk::Result::SUBOPTIMAL_KHR) => done.push((p, win::qpc())),
                        Err(_) => {}
                    }
                }
                done
            });
            let label = format!("T {:.2} ms {mode}", period as f64 * 1e-6);
            let (seg0, _) = win::qpc_raw();
            let mut next = win::qpc();
            for _ in 0..frames(300) {
                ctx.win.pump();
                while win::qpc() < next {
                    std::hint::spin_loop();
                }
                next += period as f64 * 1e-9;
                let first = chain.next_id;
                firsts.insert(first);
                eng.frame(gpu, &mut chain, id, &cfg);
                for p in first..chain.next_id {
                    let _ = tx.send(p);
                }
                id += 1;
                past.extend(chain.past(gpu));
            }
            drop(tx);
            let mut done = waiter.join().unwrap();
            println!("SEGMENT {label} qpc {seg0} {}", win::qpc_raw().0);
            eng.idle(gpu);
            past.extend(chain.past(gpu));
            let tick = ns_per_tick(&chain) * 1e-9;
            let mut fpo: Vec<(u64, f64)> = past.iter().filter_map(|p| p.at(vk::PresentStageFlagsEXT::IMAGE_FIRST_PIXEL_OUT).map(|t| (p.id, t as f64 * tick))).collect();
            fpo.sort_by_key(|x| x.0);
            fpo.dedup_by_key(|x| x.0);
            let report = |name: &str, series: &[(u64, f64)]| {
                let mut all = Vec::new();
                let mut by_pos: [Vec<f64>; 4] = Default::default();
                for w in series.windows(2).skip(20) {
                    if w[1].0 != w[0].0 + 1 {
                        continue;
                    }
                    let dt = (w[1].1 - w[0].1) * 1e3;
                    all.push(dt);
                    let pos = (0..4u64).find(|k| w[1].0 >= *k && firsts.contains(&(w[1].0 - k))).unwrap_or(0);
                    by_pos[pos as usize].push(dt);
                }
                let ideal = target as f64 * 1e-6;
                let within = all.iter().filter(|x| (*x - ideal).abs() <= 1.0).count();
                let mut errs: Vec<f64> = all.iter().map(|x| (x - ideal).abs()).collect();
                let pos_txt: Vec<String> = by_pos.iter_mut().map(|v| format!("{:.2}", median(v))).collect();
                row(
                    &format!("{label} {name}"),
                    format!(
                        "{} intervals, ideal {ideal:.3} ms, within 1 ms {:.1}%, median |err| {:.3} ms, p95 |err| {:.3} ms, median interval into batch position 1/2/3/4 {}",
                        all.len(),
                        100.0 * within as f64 / all.len().max(1) as f64,
                        median(&mut errs.clone()),
                        pct(&mut errs, 0.95),
                        pos_txt.join(" / ")
                    ),
                );
            };
            report("first pixel out", &fpo);
            let mut qe: Vec<(u64, f64)> = past.iter().filter_map(|p| p.at(vk::PresentStageFlagsEXT::QUEUE_OPERATIONS_END).map(|t| (p.id, t as f64 * tick))).collect();
            qe.sort_by_key(|x| x.0);
            qe.dedup_by_key(|x| x.0);
            report("queue operations end", &qe);
            if std::env::var("PROBE_TRACE").is_ok() {
                let d0 = done.iter().map(|x| x.1).fold(f64::MAX, f64::min);
                let q0 = qe.iter().map(|x| x.1).fold(f64::MAX, f64::min);
                let dd: Vec<String> = done.iter().skip(40).take(16).map(|x| format!("{}:{:.2}", x.0, (x.1 - d0) * 1e3)).collect();
                let qq: Vec<String> = qe.iter().skip(40).take(16).map(|x| format!("{}:{:.2}", x.0, (x.1 - q0) * 1e3)).collect();
                eprintln!("{label} wait2 {}
{label} qend {}", dd.join(" "), qq.join(" "));
            }
            done.sort_by_key(|x| x.0);
            report("present_wait2 return", &done);
            let _ = qpc_ns;
        }
    }
    eng.idle(gpu);
    chain.destroy(gpu);
}

// ---------------------------------------------------------------------------------------------
// hdrmeta

unsafe fn hdrmeta(args: &[String]) {
    let mut ctx = setup(true);
    let gpu = ctx.gpu;
    let secs: f64 = args.first().and_then(|a| a.parse().ok()).unwrap_or(6.0);
    let mut chain = Chain::new(gpu, ctx.surface, present::PQ, &Opts { timing: false, ..Opts::default() }, None);
    let mut eng = Engine::new(gpu, &chain, false);
    let mut id = 1u64;
    let cfg = Cfg { batch: 1, metered: false, content: false, ngx: false, sleep: false, markers: false, peak: 4000.0, ..Cfg::default() };
    for cycle in 0..3 {
        for cll in [400.0f32, 4000.0] {
            let xy = |x: f32, y: f32| vk::XYColorEXT { x, y };
            let md = vk::HdrMetadataEXT::default()
                .display_primary_red(xy(0.708, 0.292))
                .display_primary_green(xy(0.170, 0.797))
                .display_primary_blue(xy(0.131, 0.046))
                .white_point(xy(0.3127, 0.3290))
                .max_luminance(cll)
                .min_luminance(0.0)
                .max_content_light_level(cll)
                .max_frame_average_light_level(cll.min(400.0));
            gpu.hdr.set_hdr_metadata(&[chain.sc], &[md]);
            println!("META cycle {cycle} MaxCLL {cll} nits");
            let t0 = std::time::Instant::now();
            while t0.elapsed().as_secs_f64() < if quick() { 0.3 } else { secs } {
                if !ctx.win.pump() {
                    break;
                }
                eng.frame(gpu, &mut chain, id, &cfg);
                id += 1;
            }
        }
    }
    eng.idle(gpu);
    let (d, i, r) = chain.timing_properties(gpu);
    row("PQ timing properties", interval_text(d, i, r));
    chain.destroy(gpu);
}

fn main() {
    let _tracy = tracy_client::Client::start();
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).cloned().unwrap_or_else(|| "info".into());
    *CASE.lock().unwrap() = case.clone();
    unsafe {
        match case.as_str() {
            "info" => info(),
            "thread" => thread_case(&args[2..]),
            "flip" => flip(),
            "reflex" => reflex(),
            "pacing" => pacing(),
            "hdrmeta" => hdrmeta(&args[2..]),
            _ => {
                println!("RESULT {case} FAIL unknown case");
                std::process::exit(2);
            }
        }
    }
    let (e, w) = gpu::validation_report();
    println!("RESULT {case} DONE (validation errors={e} warnings={w})");
}
