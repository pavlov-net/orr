//! Driver probe: NGX ray reconstruction and frame generation on a descriptor-heap device.
//!
//! One case per process (`run.sh` runs them all):
//! - `info`: device, NGX capabilities, required extensions, RR render sizes at 4K.
//! - `noheap`: RR on constant (cleared) inputs, for bisecting the heap-bind failure. Env:
//!   `PROBE_NO_HEAP` (device without descriptor heaps), `PROBE_PRE_DISPATCH=bind|push|dispatch[-other]`
//!   (heap commands submitted before each RR, `-other` from another command buffer),
//!   `PROBE_PRE_FRAMES=<list>`, `PROBE_PRE_QUEUE=compute`, `PROBE_RR_QUEUE=compute`,
//!   `PROBE_RESTORE=pipeline|set|full` (classic binds recorded right before RR).
//! - `interop <general|readonly> <none|rebind|restore|restore-norebind> <render|compute> <out>`:
//!   RR with orr's heaps bound earlier on the queue; `restore` records a classic descriptor-set
//!   bind right before RR; a heap-mode dispatch reads the output after it. Images in GENERAL or
//!   with RR inputs in SHADER_READ_ONLY_OPTIMAL; graphics or compute-only family.
//! - `color <709|2020> <seed> <std|flip|flipx|flipy> <out>`: 64 panning frames through RR with colour
//!   and albedo in Rec.709 or Rec.2020; saves the last output and the reference.
//! - `fg <wide|clip> <restore|plain> <out>`: FG 2x on PQ Rec.2020 backbuffers with out-of-Rec.709 tiles, or the same
//!   content clipped to Rec.709; saves the interpolated frame and the mid-frame reference.
//! - `dof <default|E|F> <guide 0|1> <out>`: RR with a blurred region, with or without
//!   `pInDepthOfFieldGuide`; result codes, NGX log lines, output.
//! - `cost`: GPU time of RR per mode and FG 2x / 4x at 4K output (validation off).
//! - `exposure <out>`: RR history after pre-exposure steps and ramps on a static scene.
//!
//! Build: `DLSS_SDK=<sdk> cargo build --release --target-dir ../../target/probes`.

mod gpu;
mod ngx;
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, clippy::all)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/ngx.rs"));
}

use ash::vk;
use gpu::{Buf, Gpu, Img, Mem};
use ngx::{Fg, FgInputs, Ngx, Quality, Rr, RrInputs};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[macro_export]
macro_rules! spv {
    ($name:literal) => {
        ash::util::read_spv(&mut std::io::Cursor::new(&include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))[..])).expect("valid SPIR-V")
    };
}

const OUTPUT: [u32; 2] = [3840, 2160];
const RENDER: [u32; 2] = [1920, 1080];

const REC2020: u32 = 1;
const WIDE: u32 = 2;
const CLIP: u32 = 4;
const DOF: u32 = 8;
const JITTER_FLIP_X: u32 = 16;
const JITTER_FLIP_Y: u32 = 32;

/// Mirrors `Root` in shaders/scene.slang (C layout).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Root {
    out: u64,
    rows: u64,
    img: [u32; 16],
    render: [u32; 2],
    output: [u32; 2],
    pan: [f32; 2],
    dpan: [f32; 2],
    jitter: [f32; 2],
    exposure: f32,
    frame: u32,
    seed: u32,
    flags: u32,
    nits: f32,
    pad: u32,
}

/// RR inputs at render resolution, RR output, FG backbuffer and interpolated image.
struct Images {
    color: Img,
    diffuse: Img,
    specular: Img,
    normals: Img,
    depth: Img,
    motion: Img,
    spec_motion: Img,
    dof: Img,
    out: Img,
    backbuffer: Img,
}

impl Images {
    unsafe fn new(gpu: &Gpu, render: [u32; 2], output: [u32; 2]) -> Images {
        unsafe {
            Images {
                color: gpu.image(render, vk::Format::R16G16B16A16_SFLOAT),
                diffuse: gpu.image(render, vk::Format::R8G8B8A8_UNORM),
                specular: gpu.image(render, vk::Format::R8G8B8A8_UNORM),
                normals: gpu.image(render, vk::Format::R16G16B16A16_SFLOAT),
                depth: gpu.image(render, vk::Format::R32_SFLOAT),
                motion: gpu.image(render, vk::Format::R16G16_SFLOAT),
                spec_motion: gpu.image(render, vk::Format::R16G16_SFLOAT),
                dof: gpu.image(render, vk::Format::R16_SFLOAT),
                out: gpu.image(output, vk::Format::R16G16B16A16_SFLOAT),
                backbuffer: gpu.image(output, vk::Format::A2B10G10R10_UNORM_PACK32),
            }
        }
    }
    fn slots(&self) -> [u32; 16] {
        let mut s = [0; 16];
        for (i, img) in [&self.color, &self.diffuse, &self.specular, &self.normals, &self.depth, &self.motion, &self.spec_motion, &self.dof, &self.out, &self.backbuffer]
            .iter()
            .enumerate()
        {
            s[i] = img.slot;
        }
        s
    }
    fn inputs(&self) -> [vk::Image; 8] {
        [self.color.image, self.diffuse.image, self.specular.image, self.normals.image, self.depth.image, self.motion.image, self.spec_motion.image, self.dof.image]
    }
    fn all(&self) -> Vec<vk::Image> {
        let mut v = self.inputs().to_vec();
        v.push(self.out.image);
        v.push(self.backbuffer.image);
        v
    }
    fn rr<'a>(&'a self, dof: bool) -> RrInputs<'a> {
        RrInputs {
            color: &self.color,
            diffuse: &self.diffuse,
            specular: &self.specular,
            normals: &self.normals,
            depth: &self.depth,
            motion: &self.motion,
            spec_motion: &self.spec_motion,
            output: &self.out,
            dof_guide: dof.then_some(&self.dof),
        }
    }
}

struct Pipelines {
    inputs: vk::Pipeline,
    reference: vk::Pipeline,
    backbuffer: vk::Pipeline,
    copy: vk::Pipeline,
    rows: vk::Pipeline,
}

impl Pipelines {
    unsafe fn new(gpu: &Gpu) -> Pipelines {
        unsafe {
            Pipelines {
                inputs: gpu.pipeline(&spv!("inputs_main"), c"inputs_main"),
                reference: gpu.pipeline(&spv!("reference_main"), c"reference_main"),
                backbuffer: gpu.pipeline(&spv!("backbuffer_main"), c"backbuffer_main"),
                copy: gpu.pipeline(&spv!("copy_main"), c"copy_main"),
                rows: gpu.pipeline(&spv!("rows_main"), c"rows_main"),
            }
        }
    }
}

/// Everything a case needs: device, NGX, pipelines, images, one command buffer.
struct Rig {
    gpu: Gpu,
    ngx: Ngx,
    p: Pipelines,
    img: Images,
    cmd: vk::CommandBuffer,
    queue: vk::Queue,
    out_buf: Buf,
    rows_buf: Buf,
    readback: Buf,
    render: [u32; 2],
    classic: gpu::Classic,
}

impl Rig {
    unsafe fn new(render: [u32; 2], compute_queue: bool, verbose_log: bool) -> Rig {
        unsafe {
            let (ie, de) = ngx::required_extensions();
            let unified = std::env::var("PROBE_NO_UNIFIED").is_err();
            let gpu = Gpu::new(&ie, &de, unified, std::env::var("PROBE_NO_HEAP").is_err());
            let ngx = Ngx::new(&gpu, verbose_log).unwrap_or_else(|e| panic!("NGX {e}"));
            let p = if gpu.heaps {
                Pipelines::new(&gpu)
            } else {
                let n = vk::Pipeline::null();
                Pipelines { inputs: n, reference: n, backbuffer: n, copy: n, rows: n }
            };
            let img = Images::new(&gpu, render, OUTPUT);
            let q = compute_queue as usize;
            let pool = gpu.command_pool(gpu.families[q]);
            let cmd = gpu.command_buffer(pool);
            let queue = gpu.queues[q];
            let px = (OUTPUT[0] * OUTPUT[1]) as u64;
            let out_buf = gpu.buffer(px * 16, Mem::Readback);
            let rows_buf = gpu.buffer(OUTPUT[1] as u64 * 4, Mem::Readback);
            let readback = gpu.buffer(px * 8, Mem::Readback);
            let scratch = gpu.image([8, 8], vk::Format::R8G8B8A8_UNORM);
            let classic = gpu::Classic::new(&gpu, &spv!("classic_main"), &scratch);
            let rig = Rig { gpu, ngx, p, img, cmd, queue, out_buf, rows_buf, readback, render, classic };
            rig.gpu.begin(rig.cmd);
            let mut all = rig.img.all();
            all.push(scratch.image);
            rig.gpu.layouts(rig.cmd, &all, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
            rig.gpu.submit_wait(rig.queue, rig.cmd);
            rig
        }
    }

    fn root(&self, frame: u32, flags: u32) -> Root {
        Root {
            out: self.out_buf.addr,
            rows: self.rows_buf.addr,
            img: self.img.slots(),
            render: self.render,
            output: OUTPUT,
            jitter: [0.0; 2],
            exposure: 1.0,
            frame,
            flags,
            nits: 200.0,
            ..Default::default()
        }
    }

    unsafe fn inputs(&self, root: &Root) {
        unsafe {
            self.gpu.push_root(self.cmd, root);
            self.gpu.dispatch(self.cmd, self.p.inputs, [self.render[0].div_ceil(8), self.render[1].div_ceil(8)]);
        }
    }

    unsafe fn create_rr(&self, quality: Quality, preset: u32) -> Result<Rr, sys::NVSDK_NGX_Result> {
        unsafe {
            self.gpu.begin(self.cmd);
            let r = Rr::new(self.cmd, self.render, OUTPUT, quality, preset);
            self.gpu.submit_wait(self.queue, self.cmd);
            r
        }
    }

    /// Reads the RR output back as f32 RGBA.
    unsafe fn read_out(&self) -> Vec<f32> {
        unsafe { self.read_rgba16f(&self.img.out) }
    }

    unsafe fn read_rgba16f(&self, img: &Img) -> Vec<f32> {
        unsafe {
            self.gpu.begin(self.cmd);
            self.gpu.copy_to_buffer(self.cmd, img, &self.readback);
            self.gpu.submit_wait(self.queue, self.cmd);
            let n = (img.extent[0] * img.extent[1] * 4) as usize;
            let h = std::slice::from_raw_parts(self.readback.ptr.cast::<u16>(), n);
            h.iter().map(|&v| f16_to_f32(v)).collect()
        }
    }

    unsafe fn read_backbuffer(&self, img: &Img) -> Vec<u32> {
        unsafe {
            self.gpu.begin(self.cmd);
            self.gpu.copy_to_buffer(self.cmd, img, &self.readback);
            self.gpu.submit_wait(self.queue, self.cmd);
            std::slice::from_raw_parts(self.readback.ptr.cast::<u32>(), (OUTPUT[0] * OUTPUT[1]) as usize).to_vec()
        }
    }

    /// Reference at output resolution, Rec.709, for `pan`.
    unsafe fn reference(&self, pan: f32, flags: u32) -> Vec<f32> {
        unsafe {
            let mut root = self.root(0, flags);
            root.pan = [pan, 0.0];
            self.gpu.begin(self.cmd);
            self.gpu.bind_heaps(self.cmd);
            self.gpu.push_root(self.cmd, &root);
            self.gpu.dispatch(self.cmd, self.p.reference, [OUTPUT[0].div_ceil(8), OUTPUT[1].div_ceil(8)]);
            self.gpu.submit_wait(self.queue, self.cmd);
            std::slice::from_raw_parts(self.out_buf.ptr.cast::<f32>(), (OUTPUT[0] * OUTPUT[1] * 4) as usize).to_vec()
        }
    }

    /// Mean Rec.709 luminance of the RR output (rows kernel, heap read).
    unsafe fn mean_luminance(&self) -> f64 {
        unsafe {
            let rows = std::slice::from_raw_parts(self.rows_buf.ptr.cast::<f32>(), OUTPUT[1] as usize);
            rows.iter().map(|&v| v as f64).sum::<f64>() / (OUTPUT[0] * OUTPUT[1]) as f64
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn halton(i: u32, base: u32) -> f32 {
    let (mut f, mut r, mut i) = (1.0f32, 0.0f32, i);
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

/// Halton(2,3) jitter in [-0.5, 0.5) render pixels (output.md 2.5).
fn jitter(frame: u32, render: [u32; 2]) -> [f32; 2] {
    let ratio = OUTPUT[0] as f32 / render[0] as f32;
    let phases = 32u32.max((8.0 * ratio * ratio).ceil() as u32);
    let i = frame % phases + 1;
    [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
}

fn save_f32(path: &Path, w: u32, h: u32, c: u32, data: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for v in [w, h, c] {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    f.write_all(bytes).unwrap();
}

fn save_u32(path: &Path, w: u32, h: u32, data: &[u32]) {
    let f: Vec<f32> = data.iter().map(|&v| f32::from_bits(v)).collect();
    save_f32(path, w, h, 1, &f);
}

fn finish(rig: Rig) {
    unsafe {
        rig.gpu.device.device_wait_idle().unwrap();
        rig.ngx.shutdown();
    }
    let (e, w) = gpu::validation_report();
    println!("validation {} errors {e} warnings {w}", if rig.gpu.validation { "on" } else { "off" });
}

fn print_log(filter: &[&str]) {
    for line in ngx::LOG.lock().unwrap().iter() {
        if filter.is_empty() || filter.iter().any(|f| line.to_lowercase().contains(&f.to_lowercase())) {
            println!("NGXLOG {line}");
        }
    }
}

const PAN_SPEED: f32 = 1.0 / RENDER[0] as f32;

/// Bisection aid: `PROBE_CLEAR_OUT` clears the RR output to 0.5 before RR; `PROBE_CLEAR=<names>`
/// overwrites the named inputs with constants.
unsafe fn debug_clears(rig: &Rig) {
    unsafe {
        let g = &rig.gpu;
        let clear = |img: &Img, v: [f32; 4]| {
            g.device.cmd_clear_color_image(rig.cmd, img.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: v }, &[gpu::color_range()]);
        };
        if std::env::var("PROBE_CLEAR_OUT").is_ok() {
            clear(&rig.img.out, [0.5; 4]);
        }
        let list = std::env::var("PROBE_CLEAR").unwrap_or_default();
        for name in list.split(',') {
            match name {
                "color" => clear(&rig.img.color, [0.5, 0.4, 0.3, 1.0]),
                "diffuse" => clear(&rig.img.diffuse, [0.5, 0.5, 0.5, 1.0]),
                "specular" => clear(&rig.img.specular, [0.04, 0.04, 0.04, 1.0]),
                "normals" => clear(&rig.img.normals, [0.0, 0.0, 1.0, 0.5]),
                "depth" => clear(&rig.img.depth, [0.02, 0.0, 0.0, 0.0]),
                "motion" => clear(&rig.img.motion, [0.0; 4]),
                "spec_motion" => clear(&rig.img.spec_motion, [0.0; 4]),
                _ => {}
            }
        }
        g.barrier(rig.cmd);
    }
}

/// How `rr_frame` records one frame.
#[derive(Clone, Copy)]
struct Opts {
    flags: u32,
    seed: u32,
    exposure: f32,
    pan_speed: f32,
    reset: bool,
    dof: bool,
    /// RR inputs in SHADER_READ_ONLY_OPTIMAL during RR, else GENERAL throughout.
    readonly: bool,
    /// A classic descriptor-set bind right before RR.
    restore: bool,
    /// Heap-mode copy and row kernels after RR.
    heap_after: bool,
    /// Heaps rebound and the root pushed again before `heap_after`.
    rebind: bool,
    /// `heap_after` includes the full-image copy into host memory (slow); rows always run.
    copy: bool,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts { flags: 0, seed: 1, exposure: 1.0, pan_speed: PAN_SPEED, reset: false, dof: false, readonly: false, restore: true, heap_after: true, rebind: true, copy: true }
    }
}

/// One RR frame recorded into one command buffer: heaps bound, inputs, RR, then (by `o`) a
/// heap-mode copy of the output.
unsafe fn rr_frame(rig: &Rig, rr: &mut Rr, frame: u32, o: Opts) -> sys::NVSDK_NGX_Result {
    unsafe {
        let g = &rig.gpu;
        let mut root = rig.root(frame, o.flags);
        root.pan = [frame as f32 * o.pan_speed, 0.0];
        root.dpan = [o.pan_speed, 0.0];
        root.jitter = jitter(frame, rig.render);
        root.exposure = o.exposure;
        root.seed = o.seed;
        g.begin(rig.cmd);
        g.bind_heaps(rig.cmd);
        rig.inputs(&root);
        g.barrier(rig.cmd);
        debug_clears(rig);
        if o.readonly {
            g.layouts(rig.cmd, &rig.img.inputs(), vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        if o.restore {
            rig.classic.record(g, rig.cmd, "set");
        }
        let r = rr.eval(rig.cmd, &rig.img.rr(o.dof), root.jitter, o.reset);
        g.barrier(rig.cmd);
        if o.readonly {
            g.layouts(rig.cmd, &rig.img.inputs(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::GENERAL);
        }
        if o.heap_after {
            if o.rebind {
                g.bind_heaps(rig.cmd);
                g.push_root(rig.cmd, &root);
            }
            if o.copy {
                g.dispatch(rig.cmd, rig.p.copy, [OUTPUT[0].div_ceil(8), OUTPUT[1].div_ceil(8)]);
            }
            g.dispatch(rig.cmd, rig.p.rows, [OUTPUT[1].div_ceil(64), 1]);
            g.barrier(rig.cmd);
        }
        g.submit_wait(rig.queue, rig.cmd);
        r
    }
}

fn info() {
    unsafe {
        let (ie, de) = ngx::required_extensions();
        println!("ngx instance extensions {:?}", ie);
        println!("ngx device extensions {:?}", de);
        let gpu = Gpu::new(&ie, &de, true, true);
        println!("device {} driver {}", gpu.name, gpu.driver);
        println!("families render {} compute {}", gpu.families[0], gpu.families[1]);
        let hp = gpu.heap_props;
        println!(
            "heap image descriptor {} B, buffer {} B, sampler {} B, reserved resource {} B, sampler {} B",
            hp.image_descriptor_size, hp.buffer_descriptor_size, hp.sampler_descriptor_size, hp.min_resource_heap_reserved_range, hp.min_sampler_heap_reserved_range
        );
        let ngx = Ngx::new(&gpu, false).unwrap();
        println!(
            "RR available {} FG available {} MultiFrameCountMax {} RR needs driver update {}",
            ngx.caps_i(sys::NVSDK_NGX_Parameter_SuperSamplingDenoising_Available),
            ngx.caps_i(sys::NVSDK_NGX_Parameter_FrameGeneration_Available),
            ngx.caps_ui(sys::NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax),
            ngx.caps_i(sys::NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver),
        );
        for q in [Quality::Performance, Quality::Balanced, Quality::Quality, Quality::Dlaa] {
            println!("RR render size {q:?} at 3840x2160: {:?}", ngx.rr_render_size(OUTPUT, q));
        }
        gpu.device.device_wait_idle().unwrap();
        ngx.shutdown();
        print_log(&["version", "dlssd", "dlssg", "snippet"]);
        let (e, w) = gpu::validation_report();
        println!("validation errors {e} warnings {w}");
    }
}

/// RR on a device without descriptor heaps, inputs filled by clears: does RR write its output?
fn noheap() {
    unsafe {
        let rr_compute = std::env::var("PROBE_RR_QUEUE").is_ok_and(|v| v == "compute");
        let pre_compute = std::env::var("PROBE_PRE_QUEUE").is_ok_and(|v| v == "compute");
        let pre_frames: Vec<u32> = std::env::var("PROBE_PRE_FRAMES").unwrap_or_default().split(',').filter_map(|v| v.parse().ok()).collect();
        let rig = Rig::new(RENDER, rr_compute, false);
        let g = &rig.gpu;
        println!("descriptor heaps enabled {}", g.heaps);
        let mut rr = rig.create_rr(Quality::Performance, 0).unwrap_or_else(|e| panic!("RR create {e:#x}"));
        let restore = std::env::var("PROBE_RESTORE").unwrap_or_default();
        let other = g.command_buffer(g.command_pool(g.families[pre_compute as usize]));
        let pre_queue = g.queues[pre_compute as usize];
        let frames: u32 = std::env::var("PROBE_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        for frame in 0..frames {
            let mut written = Vec::new();
            if let Ok(mode) = std::env::var("PROBE_PRE_DISPATCH")
                && (pre_frames.is_empty() || pre_frames.contains(&frame))
            {
                // bind | push | dispatch, in this command buffer or (suffix "-other") another one.
                let cmd = if mode.ends_with("-other") { other } else { rig.cmd };
                let mut root = rig.root(frame, 0);
                root.jitter = jitter(frame, RENDER);
                g.begin(cmd);
                g.bind_heaps(cmd);
                if mode.starts_with("push") || mode.starts_with("dispatch") {
                    g.push_root(cmd, &root);
                }
                if mode.starts_with("dispatch") {
                    g.dispatch(cmd, rig.p.inputs, [RENDER[0].div_ceil(8), RENDER[1].div_ceil(8)]);
                }
                g.submit_wait(if cmd == other { pre_queue } else { rig.queue }, cmd);
                written.push(frame);
            }
            g.begin(rig.cmd);
            let clear = |img: &Img, v: [f32; 4]| {
                g.device.cmd_clear_color_image(rig.cmd, img.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: v }, &[gpu::color_range()]);
            };
            clear(&rig.img.color, [0.5, 0.4, 0.3, 1.0]);
            clear(&rig.img.diffuse, [0.5, 0.5, 0.5, 1.0]);
            clear(&rig.img.specular, [0.04, 0.04, 0.04, 1.0]);
            clear(&rig.img.normals, [0.0, 0.0, 1.0, 0.5]);
            clear(&rig.img.depth, [0.02, 0.0, 0.0, 0.0]);
            clear(&rig.img.motion, [0.0; 4]);
            clear(&rig.img.spec_motion, [0.0; 4]);
            clear(&rig.img.out, [0.25; 4]);
            g.barrier(rig.cmd);
            if !restore.is_empty() {
                rig.classic.record(g, rig.cmd, &restore);
                g.barrier(rig.cmd);
            }
            let r = rr.eval(rig.cmd, &rig.img.rr(false), jitter(frame, RENDER), frame == 0);
            g.barrier(rig.cmd);
            g.submit_wait(rig.queue, rig.cmd);
            if !ngx::ok(r) {
                println!("frame {frame} evaluate {r:#x}");
            }
            let out = rig.read_out();
            println!("frame {frame} heap bound before {} output centre {:?}", !written.is_empty(), &out[(OUTPUT[0] * OUTPUT[1] / 2 + OUTPUT[0] / 2) as usize * 4..][..3]);
        }
        let img = rig.read_out();
        println!("output pixel 0 {:?}, centre {:?}", &img[0..4], &img[(OUTPUT[0] * OUTPUT[1] / 2 + OUTPUT[0] / 2) as usize * 4..][..4]);
        rr.release();
        print_log(&["warn", "error", "fail", "invalid"]);
        finish(rig);
    }
}

fn interop(layout: &str, heap: &str, queue: &str, out: &Path) {
    unsafe {
        let readonly = layout == "readonly";
        // The RR output is cleared to 0.5 before every evaluate; pixels still at 0.5 were not written.
        std::env::set_var("PROBE_CLEAR_OUT", "1");
        let rig = Rig::new(RENDER, queue == "compute", std::env::var("PROBE_VERBOSE").is_ok());
        println!("unified image layouts {}", rig.gpu.unified);
        let mut rr = rig.create_rr(Quality::Performance, 0).unwrap_or_else(|e| panic!("RR create {e:#x}"));
        let mut fails = 0;
        let frames = 24;
        for frame in 0..frames {
            let r = if heap == "none" {
                // Inputs in one submission, RR alone in a command buffer that never binds a heap.
                let g = &rig.gpu;
                let mut root = rig.root(frame, 0);
                root.pan = [frame as f32 * PAN_SPEED, 0.0];
                root.dpan = [PAN_SPEED, 0.0];
                root.jitter = jitter(frame, rig.render);
                root.seed = 1;
                g.begin(rig.cmd);
                g.bind_heaps(rig.cmd);
                rig.inputs(&root);
                g.barrier(rig.cmd);
                g.submit_wait(rig.queue, rig.cmd);
                g.begin(rig.cmd);
                debug_clears(&rig);
                if readonly {
                    g.layouts(rig.cmd, &rig.img.inputs(), vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                }
                let r = rr.eval(rig.cmd, &rig.img.rr(false), root.jitter, frame == 0);
                g.barrier(rig.cmd);
                if readonly {
                    g.layouts(rig.cmd, &rig.img.inputs(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::GENERAL);
                }
                g.submit_wait(rig.queue, rig.cmd);
                r
            } else {
                rr_frame(&rig, &mut rr, frame, Opts { reset: frame == 0, readonly, restore: heap.starts_with("restore"), rebind: !heap.ends_with("norebind"), ..Default::default() })
            };
            if !ngx::ok(r) {
                fails += 1;
                println!("frame {frame} RR evaluate {r:#x}");
            }
        }
        let input = rig.read_rgba16f(&rig.img.color);
        let in_mean: f64 = input.chunks(4).map(|p| p[1] as f64).sum::<f64>() / (input.len() / 4) as f64;
        println!("colour input mean green {in_mean:.4}");
        let img = rig.read_out();
        let finite = img.iter().all(|v| v.is_finite());
        let mean: f64 = img.chunks(4).map(|p| (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) as f64).sum::<f64>() / (img.len() / 4) as f64;
        let unwritten = img.chunks(4).filter(|p| p.iter().all(|&v| v == 0.5)).count();
        println!("RR evaluate failures {fails} of {frames}; output finite {finite}, mean luminance {mean:.4}, pixels not written {unwritten} of {}", img.len() / 4);
        if heap != "none" {
            let copied = std::slice::from_raw_parts(rig.out_buf.ptr.cast::<f32>(), img.len());
            let differ = copied.iter().zip(&img).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            println!("heap-mode copy after RR: {differ} of {} values differ from the transfer copy; row-kernel mean {:.4}", img.len(), rig.mean_luminance());
        }
        save_f32(&out.join(format!("interop_{layout}_{heap}_{queue}{}.f32", if rig.gpu.unified { "" } else { "_nounified" })), OUTPUT[0], OUTPUT[1], 4, &img);
        let reference = out.join("ref_interop.f32");
        if !reference.exists() {
            let r = rig.reference((frames - 1) as f32 * PAN_SPEED, 0);
            save_f32(&reference, OUTPUT[0], OUTPUT[1], 4, &r);
        }
        rr.release();
        print_log(&["warn", "error", "fail", "invalid"]);
        finish(rig);
    }
}

fn color(space: &str, seed: u32, sign: &str, out: &Path) {
    unsafe {
        let rig = Rig::new(RENDER, false, false);
        let mut rr = rig.create_rr(Quality::Performance, 0).unwrap_or_else(|e| panic!("RR create {e:#x}"));
        let jflags = match sign {
            "flip" => JITTER_FLIP_X | JITTER_FLIP_Y,
            "flipx" => JITTER_FLIP_X,
            "flipy" => JITTER_FLIP_Y,
            _ => 0,
        };
        let flags = if space == "2020" { REC2020 } else { 0 } | jflags;
        let frames = 64;
        for frame in 0..frames {
            let r = rr_frame(&rig, &mut rr, frame, Opts { flags, seed, reset: frame == 0, copy: false, ..Default::default() });
            assert!(ngx::ok(r), "RR evaluate {r:#x}");
        }
        let img = rig.read_out();
        let tag = if jflags == 0 { "std" } else { sign };
        save_f32(&out.join(format!("color_{space}_s{seed}_{tag}.f32")), OUTPUT[0], OUTPUT[1], 4, &img);
        let reference = out.join("color_reference.f32");
        if !reference.exists() {
            let r = rig.reference((frames - 1) as f32 * PAN_SPEED, 0);
            save_f32(&reference, OUTPUT[0], OUTPUT[1], 4, &r);
        }
        rr.release();
        finish(rig);
    }
}

fn fg(mode: &str, restore: bool, out: &Path) {
    unsafe {
        let rig = Rig::new(RENDER, false, false);
        let g = &rig.gpu;
        let interp = g.image(OUTPUT, vk::Format::A2B10G10R10_UNORM_PACK32);
        let truth = g.image(OUTPUT, vk::Format::A2B10G10R10_UNORM_PACK32);
        g.begin(rig.cmd);
        g.layouts(rig.cmd, &[interp.image, truth.image], vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
        let fgf = Fg::new(rig.cmd, OUTPUT, vk::Format::A2B10G10R10_UNORM_PACK32);
        g.submit_wait(rig.queue, rig.cmd);
        let mut fgf = match fgf {
            Ok(f) => f,
            Err(e) => {
                println!("FG create {e:#x}");
                print_log(&[]);
                finish(rig);
                return;
            }
        };
        let flags = if mode == "wide" { WIDE } else { WIDE | CLIP };
        let speed = 4.0 / OUTPUT[0] as f32;
        let frames = 16u32;
        for frame in 0..frames {
            let mut root = rig.root(frame, flags);
            root.pan = [frame as f32 * speed, 0.0];
            root.dpan = [speed, 0.0];
            g.begin(rig.cmd);
            g.bind_heaps(rig.cmd);
            rig.inputs(&root);
            g.push_root(rig.cmd, &root);
            g.dispatch(rig.cmd, rig.p.backbuffer, [OUTPUT[0].div_ceil(8), OUTPUT[1].div_ceil(8)]);
            g.device.cmd_clear_color_image(rig.cmd, interp.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[gpu::color_range()]);
            g.barrier(rig.cmd);
            if restore {
                rig.classic.record(g, rig.cmd, "set");
            }
            let r = fgf.eval(rig.cmd, &FgInputs { backbuffer: &rig.img.backbuffer, depth: &rig.img.depth, motion: &rig.img.motion, interpolated: &interp }, rig.render, 1, 1, frame as u64, frame == 0, speed, true);
            g.barrier(rig.cmd);
            g.submit_wait(rig.queue, rig.cmd);
            if !ngx::ok(r) {
                println!("frame {frame} FG evaluate {r:#x}");
            }
        }
        println!("OutputDisableInterpolation {}", fgf.interpolation_disabled());
        let got = rig.read_backbuffer(&interp);
        println!("FG restore {restore}: interpolated pixels not written {} of {}", got.iter().filter(|&&v| v == 0).count(), got.len());
        // Mid-frame truth: the backbuffer content half a frame back.
        let mut root = rig.root(0, flags);
        root.pan = [(frames as f32 - 1.5) * speed, 0.0];
        g.begin(rig.cmd);
        g.bind_heaps(rig.cmd);
        let mut troot = root;
        troot.img[9] = truth.slot;
        g.push_root(rig.cmd, &troot);
        g.dispatch(rig.cmd, rig.p.backbuffer, [OUTPUT[0].div_ceil(8), OUTPUT[1].div_ceil(8)]);
        g.submit_wait(rig.queue, rig.cmd);
        let want = rig.read_backbuffer(&truth);
        let real = rig.read_backbuffer(&rig.img.backbuffer);
        save_u32(&out.join(format!("fg_{mode}_interp.u32")), OUTPUT[0], OUTPUT[1], &got);
        save_u32(&out.join(format!("fg_{mode}_truth.u32")), OUTPUT[0], OUTPUT[1], &want);
        save_u32(&out.join(format!("fg_{mode}_real.u32")), OUTPUT[0], OUTPUT[1], &real);
        fgf.release();
        finish(rig);
    }
}

fn dof(preset: &str, guide: bool, out: &Path) {
    unsafe {
        let rig = Rig::new(RENDER, false, true);
        let p = match preset {
            "E" => 5,
            "F" => 6,
            _ => 0,
        };
        let mut rr = match rig.create_rr(Quality::Performance, p) {
            Ok(rr) => rr,
            Err(e) => {
                println!("RR create preset {preset} {e:#x}");
                print_log(&[]);
                finish(rig);
                return;
            }
        };
        println!("RR create preset {preset}: success");
        let frames = 64;
        let mut results = std::collections::BTreeMap::new();
        for frame in 0..frames {
            let r = rr_frame(&rig, &mut rr, frame, Opts { flags: DOF, reset: frame == 0, dof: guide, copy: false, ..Default::default() });
            *results.entry(r).or_insert(0) += 1;
        }
        println!("RR evaluate results {:?}", results.iter().map(|(k, v)| format!("{k:#x} x{v}")).collect::<Vec<_>>());
        let img = rig.read_out();
        save_f32(&out.join(format!("dof_{preset}_g{}.f32", guide as u32)), OUTPUT[0], OUTPUT[1], 4, &img);
        let reference = out.join("dof_reference.f32");
        if !reference.exists() {
            let r = rig.reference((frames - 1) as f32 * PAN_SPEED, DOF);
            save_f32(&reference, OUTPUT[0], OUTPUT[1], 4, &r);
        }
        rr.release();
        print_log(&["preset", "depthoffield", "dof", "guide", "error", "warn", "fail", "model", "snippet", "version"]);
        finish(rig);
    }
}

/// Median, 10th and 90th percentile of `v` (sorted in place).
fn stats(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.1), at(0.9))
}

unsafe fn timestamps(gpu: &Gpu) -> vk::QueryPool {
    unsafe { gpu.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None).unwrap() }
}

unsafe fn read_ms(gpu: &Gpu, pool: vk::QueryPool) -> f64 {
    unsafe {
        let mut t = [0u64; 2];
        gpu.device.get_query_pool_results(pool, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).unwrap();
        (t[1] - t[0]) as f64 * gpu.ts_period_ns / 1e6
    }
}

fn cost() {
    unsafe {
        let n = 150;
        let idle = Duration::from_millis(6);
        for q in [Quality::Performance, Quality::Balanced, Quality::Quality, Quality::Dlaa] {
            let probe = Rig::new(RENDER, false, false);
            let render = probe.ngx.rr_render_size(OUTPUT, q).unwrap();
            finish(probe);
            let rig = Rig::new(render, false, false);
            let g = &rig.gpu;
            let pool = timestamps(g);
            let before = g.vram_used();
            let mut rr = rig.create_rr(q, 0).unwrap_or_else(|e| panic!("RR create {e:#x}"));
            let mut ms = Vec::new();
            for i in 0..n + 10 {
                let mut root = rig.root(i, 0);
                root.jitter = jitter(i, render);
                root.seed = 1;
                g.begin(rig.cmd);
                g.device.cmd_reset_query_pool(rig.cmd, pool, 0, 2);
                g.bind_heaps(rig.cmd);
                rig.inputs(&root);
                g.barrier(rig.cmd);
                rig.classic.record(g, rig.cmd, "set");
                rr.eval(rig.cmd, &rig.img.rr(false), root.jitter, false);
                g.barrier(rig.cmd);
                g.device.cmd_write_timestamp2(rig.cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 0);
                let r = rr.eval(rig.cmd, &rig.img.rr(false), root.jitter, false);
                g.device.cmd_write_timestamp2(rig.cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 1);
                g.submit_wait(rig.queue, rig.cmd);
                assert!(ngx::ok(r));
                if i >= 10 {
                    ms.push(read_ms(g, pool));
                }
                std::thread::sleep(idle);
            }
            let after = g.vram_used();
            let (m, lo, hi) = stats(&mut ms);
            println!(
                "RR {q:?} render {}x{} -> 3840x2160: median {m:.3} ms (p10 {lo:.3}, p90 {hi:.3}), VRAM delta {:.1} MB, NGX stats {:?} MB",
                render[0],
                render[1],
                (after as f64 - before as f64) / 1e6,
                rig.ngx.rr_vram().map(|b| (b as f64 / 1e6 * 10.0).round() / 10.0)
            );
            rr.release();
            finish(rig);
        }
        for count in [1u32, 3] {
            let rig = Rig::new(RENDER, false, false);
            let g = &rig.gpu;
            let pool = timestamps(g);
            let interp = g.image(OUTPUT, vk::Format::A2B10G10R10_UNORM_PACK32);
            let before = g.vram_used();
            g.begin(rig.cmd);
            g.layouts(rig.cmd, &[interp.image], vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
            let fgf = Fg::new(rig.cmd, OUTPUT, vk::Format::A2B10G10R10_UNORM_PACK32);
            g.submit_wait(rig.queue, rig.cmd);
            let mut fgf = match fgf {
                Ok(f) => f,
                Err(e) => {
                    println!("FG create {e:#x}");
                    finish(rig);
                    continue;
                }
            };
            let speed = 4.0 / OUTPUT[0] as f32;
            let mut ms = Vec::new();
            let mut id = 0u64;
            for i in 0..n + 10 {
                let mut root = rig.root(i, WIDE);
                root.dpan = [speed, 0.0];
                g.begin(rig.cmd);
                g.device.cmd_reset_query_pool(rig.cmd, pool, 0, 2);
                g.bind_heaps(rig.cmd);
                for half in 0..2 {
                    root.pan = [id as f32 * speed, 0.0];
                    rig.inputs(&root);
                    g.push_root(rig.cmd, &root);
                    g.dispatch(rig.cmd, rig.p.backbuffer, [OUTPUT[0].div_ceil(8), OUTPUT[1].div_ceil(8)]);
                    g.barrier(rig.cmd);
                    rig.classic.record(g, rig.cmd, "set");
                    if half == 1 {
                        g.device.cmd_write_timestamp2(rig.cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 0);
                    }
                    for index in 1..=count {
                        let inputs = FgInputs { backbuffer: &rig.img.backbuffer, depth: &rig.img.depth, motion: &rig.img.motion, interpolated: &interp };
                        fgf.eval(rig.cmd, &inputs, rig.render, index, count, id, id == 0, speed, true);
                    }
                    if half == 1 {
                        g.device.cmd_write_timestamp2(rig.cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 1);
                    } else {
                        g.barrier(rig.cmd);
                        g.bind_heaps(rig.cmd);
                    }
                    id += 1;
                }
                g.submit_wait(rig.queue, rig.cmd);
                if i >= 10 {
                    ms.push(read_ms(g, pool));
                }
                std::thread::sleep(idle);
            }
            let after = g.vram_used();
            let (m, lo, hi) = stats(&mut ms);
            println!(
                "FG {}x at 3840x2160 ({count} evaluate{}): median {m:.3} ms (p10 {lo:.3}, p90 {hi:.3}), VRAM delta {:.1} MB",
                count + 1,
                if count > 1 { "s" } else { "" },
                (after as f64 - before as f64) / 1e6
            );
            fgf.release();
            finish(rig);
        }
    }
}

/// Mean RR output luminance per frame, divided by the frame's pre-exposure.
fn exposure(out: &Path) {
    unsafe {
        let rig = Rig::new(RENDER, false, false);
        let mut csv = String::from("run,frame,exposure,absolute_mean\n");
        let warm = 100u32;
        let runs: [(&str, fn(u32) -> f32, Option<u32>); 9] = [
            ("steady1", |_| 1.0, None),
            ("steady4", |_| 4.0, None),
            ("step_up", |f| if f < 100 { 1.0 } else { 4.0 }, None),
            ("step_up_reset", |f| if f < 100 { 1.0 } else { 4.0 }, Some(100)),
            ("step_down", |f| if f < 100 { 1.0 } else { 0.25 }, None),
            ("step_down_reset", |f| if f < 100 { 1.0 } else { 0.25 }, Some(100)),
            ("ramp_up", |f| 2f32.powf(0.05 * f.saturating_sub(100).min(120) as f32), None),
            ("ramp_down", |f| 2f32.powf(-0.05 * f.saturating_sub(100).min(120) as f32), None),
            ("ramp_up_slow", |f| 2f32.powf(0.0125 * f.saturating_sub(100).min(120) as f32), None),
        ];
        for (name, s, reset_at) in runs {
            eprintln!("exposure run {name}");
            let mut rr = rig.create_rr(Quality::Performance, 0).unwrap();
            for frame in 0..warm + 160 {
                if frame % 50 == 0 {
                    eprintln!("  frame {frame}");
                }
                let e = s(frame);
                let reset = frame == 0 || reset_at == Some(frame);
                let r = rr_frame(&rig, &mut rr, frame, Opts { exposure: e, pan_speed: 0.0, reset, copy: false, ..Default::default() });
                assert!(ngx::ok(r));
                if frame >= warm - 10 {
                    csv += &format!("{name},{frame},{e},{:.6}\n", rig.mean_luminance() / e as f64);
                }
            }
            rr.release();
        }
        std::fs::write(out.join("exposure.csv"), csv).unwrap();
        finish(rig);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = |i: usize| args.get(i).map(String::as_str).unwrap_or("");
    let out = |i: usize| {
        let p = PathBuf::from(args.get(i).cloned().unwrap_or_else(|| "out".into()));
        std::fs::create_dir_all(&p).unwrap();
        p
    };
    match a(0) {
        "info" => info(),
        "noheap" => noheap(),
        "interop" => interop(a(1), a(2), a(3), &out(4)),
        "color" => color(a(1), a(2).parse().unwrap(), a(3), &out(4)),
        "fg" => fg(a(1), a(2) != "plain", &out(3)),
        "dof" => dof(a(1), a(2) == "1", &out(3)),
        "cost" => cost(),
        "exposure" => exposure(&out(1)),
        _ => eprintln!("usage: ngx-interop info|interop|color|fg|dof|cost|exposure ..."),
    }
}
