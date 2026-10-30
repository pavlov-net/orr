//! The render-thread frame: a fixed list of record calls (60 passes, about 2000 dispatches,
//! mesh draws and barriers), NGX RR and FG evaluate calls, one main submit, then the present
//! path of contracts.md 13: for each image of the batch, acquire, compose-UI record, submit and
//! present, inside one PresentStart/PresentEnd bracket. Every step carries a Tracy span and a
//! wall-clock time.

use crate::gpu::{Buf, Gpu, Img, Mem};
use crate::present::Chain;
use ash::vk::{self, TaggedStructure};
use std::time::Instant;

pub const SLOTS: usize = 2;
const COMPUTE_DISPATCHES: u32 = 30;
const MESH_DRAWS: u32 = 40;
const PASSES: u32 = 60;
/// Passes 50..60 are output's (after RR).
const OUTPUT_FROM: u32 = 50;

#[repr(C)]
#[derive(Clone, Copy)]
struct PassRoot {
    dst: u64,
    n: u32,
    iters: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MeshRoot {
    offset: [f32; 2],
    scale: f32,
    shade: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ComposeRoot {
    pub extent: [u32; 2],
    pub encoding: u32,
    pub source: u32,
    pub tick: u32,
    pub batch: u32,
    pub paper: f32,
    pub peak: f32,
}

/// Which value the frame's latency markers and submit present-id carry.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Key {
    /// orr's frame id (contracts.md 14).
    Frame,
    /// The present id of the frame's last (rendered) present.
    LastPresent,
    /// The present id of the frame's first present.
    FirstPresent,
}

#[derive(Clone, Copy)]
pub struct Cfg {
    pub batch: u32,
    pub metered: bool,
    pub relative_ns: u64,
    pub content: bool,
    pub ngx: bool,
    pub key: Key,
    pub sleep: bool,
    pub markers: bool,
    pub peak: f32,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg { batch: 4, metered: true, relative_ns: 0, content: true, ngx: true, key: Key::Frame, sleep: true, markers: true, peak: 1000.0 }
    }
}

/// Wall-clock ms of each step of one frame.
#[derive(Default, Clone, Copy, Debug)]
pub struct Steps {
    pub slot_wait: f64,
    pub sleep: f64,
    pub record_scene: f64,
    pub ngx_rr: f64,
    pub record_output: f64,
    pub ngx_fg: f64,
    pub submit: f64,
    pub acquire: f64,
    pub acquire_max: f64,
    pub compose: f64,
    pub present_submit: f64,
    pub present: f64,
    pub present_max: f64,
    /// PresentStart to PresentEnd.
    pub present_path: f64,
    /// begin_frame (after the sleep) to PresentEnd, minus the slot wait.
    pub total: f64,
    /// CPU time the thread ran inside acquire and present calls (thread cycles).
    pub present_cpu: f64,
    pub feedback: u32,
    pub ngx_fail: u32,
    pub present_fail: u32,
}

pub struct Engine {
    pass: vk::Pipeline,
    mesh: vk::Pipeline,
    compose: vk::Pipeline,
    scratch: Buf,
    heavy: Buf,
    target: Img,
    pub backbuffer: Img,
    pub interp: Vec<Img>,
    pub inputs: Option<crate::ngx::Inputs>,
    pools: Vec<vk::CommandPool>,
    cmds: Vec<Vec<vk::CommandBuffer>>,
    queries: vk::QueryPool,
    pub timeline: vk::Semaphore,
    pub value: u64,
    slot_value: [u64; SLOTS],
    slot_frame: [u64; SLOTS],
    pub sleep_sem: vk::Semaphore,
    pub heavy_iters: u32,
    pub ngx: Option<crate::ngx::Ngx>,
    pub output: [u32; 2],
    pub render: [u32; 2],
    /// GPU ms of the main list, per slot, read when the slot is reused.
    pub gpu_ms: Vec<f64>,
    pub cycles_per_ms: f64,
}

const STORAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::STORAGE.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::TRANSFER_DST.as_raw() | vk::ImageUsageFlags::TRANSFER_SRC.as_raw(),
);

impl Engine {
    pub unsafe fn new(gpu: &Gpu, chain: &Chain, want_ngx: bool) -> Engine {
        unsafe {
            let output = chain.extent;
            let render = [output[0] / 2, output[1] / 2];
            let fmt = chain.fmt.format;
            let pass = gpu.compute_pipeline(&crate::spv!("pass"), gpu.layout);
            let compose = gpu.compute_pipeline(&crate::spv!("compose"), gpu.compose_layout);
            let mesh = gpu.mesh_pipeline(vk::Format::R8G8B8A8_UNORM);
            let scratch = gpu.buffer(64 << 20, Mem::Device);
            let heavy = gpu.buffer(16 << 20, Mem::Device);
            let target = gpu.image(render, vk::Format::R8G8B8A8_UNORM, STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT);
            let backbuffer = gpu.image(output, fmt, STORAGE);
            let interp: Vec<Img> = (0..7).map(|_| gpu.image(output, fmt, STORAGE)).collect();
            let pools: Vec<_> = (0..SLOTS).map(|_| gpu.command_pool(gpu.families[0])).collect();
            let cmds = pools.iter().map(|&p| gpu.command_buffers(p, 9)).collect();
            let queries = gpu
                .device
                .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2 * SLOTS as u32), None)
                .unwrap();
            gpu.device.reset_query_pool(queries, 0, 2 * SLOTS as u32);
            let mut e = Engine {
                pass,
                mesh,
                compose,
                scratch,
                heavy,
                target,
                backbuffer,
                interp,
                inputs: None,
                pools,
                cmds,
                queries,
                timeline: gpu.timeline(),
                value: 0,
                slot_value: [0; SLOTS],
                slot_frame: [0; SLOTS],
                sleep_sem: gpu.timeline(),
                heavy_iters: 64,
                ngx: None,
                output,
                render,
                gpu_ms: Vec::new(),
                cycles_per_ms: crate::win::cycles_per_ms(),
            };
            // Images to GENERAL once; NGX inputs and outputs too.
            let cmd = e.cmds[0][8];
            gpu.begin(cmd);
            let mut all = vec![e.target.image, e.backbuffer.image];
            all.extend(e.interp.iter().map(|i| i.image));
            if want_ngx && cfg!(ngx) {
                let i = crate::ngx::Inputs {
                    color: gpu.image(render, vk::Format::R16G16B16A16_SFLOAT, STORAGE),
                    depth: gpu.image(render, vk::Format::R32_SFLOAT, STORAGE),
                    motion: gpu.image(render, vk::Format::R16G16_SFLOAT, STORAGE),
                    diffuse: gpu.image(render, vk::Format::R8G8B8A8_UNORM, STORAGE),
                    specular: gpu.image(render, vk::Format::R8G8B8A8_UNORM, STORAGE),
                    normals: gpu.image(render, vk::Format::R16G16B16A16_SFLOAT, STORAGE),
                    rr_out: gpu.image(output, vk::Format::R16G16B16A16_SFLOAT, STORAGE),
                };
                all.extend([i.color.image, i.depth.image, i.motion.image, i.diffuse.image, i.specular.image, i.normals.image, i.rr_out.image]);
                e.inputs = Some(i);
            }
            for &img in &all {
                gpu.layout_barrier(cmd, img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
                let c = vk::ClearColorValue { float32: [0.2, 0.2, 0.2, 1.0] };
                gpu.device.cmd_clear_color_image(cmd, img, vk::ImageLayout::GENERAL, &c, &[crate::gpu::color_range()]);
            }
            gpu.barrier(cmd);
            #[cfg(ngx)]
            if want_ngx {
                match crate::ngx::Ngx::new(gpu, cmd, render, output, fmt) {
                    Ok(n) => e.ngx = Some(n),
                    Err(m) => crate::row("ngx", format!("unavailable: {m}")),
                }
            }
            gpu.device.end_command_buffer(cmd).unwrap();
            e.value += 1;
            e.submit(gpu, cmd, &[], None, 0, e.value);
            gpu.wait(e.timeline, e.value);
            e
        }
    }

    unsafe fn submit(&self, gpu: &Gpu, cmd: vk::CommandBuffer, wait: &[vk::Semaphore], signal_bin: Option<vk::Semaphore>, key: u64, value: u64) {
        unsafe {
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let waits: Vec<_> = wait.iter().map(|&s| vk::SemaphoreSubmitInfo::default().semaphore(s).stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)).collect();
            let mut sigs = vec![vk::SemaphoreSubmitInfo::default().semaphore(self.timeline).value(value).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            if let Some(b) = signal_bin {
                sigs.push(vk::SemaphoreSubmitInfo::default().semaphore(b).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS));
            }
            let mut lat = vk::LatencySubmissionPresentIdNV::default().present_id(key);
            let mut info = vk::SubmitInfo2::default().command_buffer_infos(&cbs).wait_semaphore_infos(&waits).signal_semaphore_infos(&sigs);
            if key != 0 {
                info = info.push(&mut lat);
            }
            gpu.device.queue_submit2(gpu.queues[0], &[info], vk::Fence::null()).expect("submit");
        }
    }

    /// Scales the heavy dispatch so the main list takes about `ms` on the GPU.
    pub unsafe fn calibrate(&mut self, gpu: &Gpu, ms: f64) {
        unsafe {
            for _ in 0..5 {
                let mut t = Vec::new();
                for _ in 0..5 {
                    let cmd = self.cmds[0][7];
                    gpu.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                    gpu.begin(cmd);
                    gpu.device.cmd_reset_query_pool(cmd, self.queries, 0, 2);
                    gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, self.queries, 0);
                    self.heavy_dispatch(gpu, cmd);
                    gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, self.queries, 1);
                    gpu.device.end_command_buffer(cmd).unwrap();
                    self.value += 1;
                    self.submit(gpu, cmd, &[], None, 0, self.value);
                    gpu.wait(self.timeline, self.value);
                    t.push(self.read_ms(gpu, 0));
                }
                let m = crate::median(&mut t);
                self.heavy_iters = ((f64::from(self.heavy_iters) * ms / m.max(0.01)) as u32).clamp(4, 1 << 22);
            }
        }
    }

    unsafe fn read_ms(&self, gpu: &Gpu, slot: usize) -> f64 {
        unsafe {
            let mut ts = [0u64; 2];
            gpu.device
                .get_query_pool_results(self.queries, 2 * slot as u32, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                .unwrap();
            (ts[1].wrapping_sub(ts[0])) as f64 * gpu.ts_period_ns * 1e-6
        }
    }

    unsafe fn heavy_dispatch(&self, gpu: &Gpu, cmd: vk::CommandBuffer) {
        unsafe {
            gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pass);
            let n = (self.heavy.size / 4) as u32;
            gpu.push(cmd, gpu.layout, vk::ShaderStageFlags::COMPUTE, &PassRoot { dst: self.heavy.addr, n, iters: self.heavy_iters });
            gpu.device.cmd_dispatch(cmd, n.div_ceil(64), 1, 1);
        }
    }

    /// Records passes [from, to).
    unsafe fn passes(&self, gpu: &Gpu, cmd: vk::CommandBuffer, from: u32, to: u32) {
        unsafe {
            let d = &gpu.device;
            for p in from..to {
                let raster = p < OUTPUT_FROM && p % 5 < 2;
                if raster {
                    let color = [vk::RenderingAttachmentInfo::default()
                        .image_view(self.target.view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .load_op(if p == 0 { vk::AttachmentLoadOp::CLEAR } else { vk::AttachmentLoadOp::LOAD })
                        .store_op(vk::AttachmentStoreOp::STORE)];
                    let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: self.render[0], height: self.render[1] } };
                    d.cmd_begin_rendering(cmd, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&color));
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.mesh);
                    d.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: self.render[0] as f32, height: self.render[1] as f32, min_depth: 0.0, max_depth: 1.0 }]);
                    d.cmd_set_scissor(cmd, 0, &[area]);
                    for k in 0..MESH_DRAWS {
                        let x = (k % 8) as f32 * 0.25 - 1.0;
                        let y = (k / 8) as f32 * 0.4 - 1.0;
                        let r = MeshRoot { offset: [x, y], scale: 0.1, shade: (p as f32) / 60.0 };
                        gpu.push(cmd, gpu.mesh_layout, vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT, &r);
                        gpu.mesh.cmd_draw_mesh_tasks(cmd, 1, 1, 1);
                    }
                    d.cmd_end_rendering(cmd);
                } else {
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pass);
                    for k in 0..COMPUTE_DISPATCHES {
                        let off = u64::from((p * COMPUTE_DISPATCHES + k) % 1024) * 16384;
                        gpu.push(cmd, gpu.layout, vk::ShaderStageFlags::COMPUTE, &PassRoot { dst: self.scratch.addr + off, n: 4096, iters: 4 });
                        d.cmd_dispatch(cmd, 64, 1, 1);
                    }
                    if p == 2 {
                        self.heavy_dispatch(gpu, cmd);
                    }
                }
                gpu.barrier(cmd);
            }
        }
    }

    pub unsafe fn compose_into(&self, gpu: &Gpu, cmd: vk::CommandBuffer, dst: vk::ImageView, src: vk::ImageView, root: &ComposeRoot) {
        unsafe {
            gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.compose);
            let di = [vk::DescriptorImageInfo::default().image_view(dst).image_layout(vk::ImageLayout::GENERAL)];
            let si = [vk::DescriptorImageInfo::default().image_view(src).image_layout(vk::ImageLayout::GENERAL)];
            let writes = [
                vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&di),
                vk::WriteDescriptorSet::default().dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&si),
            ];
            gpu.push_desc.cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, gpu.compose_layout, 0, &writes);
            gpu.push(cmd, gpu.compose_layout, vk::ShaderStageFlags::COMPUTE, root);
            gpu.device.cmd_dispatch(cmd, root.extent[0].div_ceil(8), root.extent[1].div_ceil(8), 1);
        }
    }

    /// One frame. `frame_id` is orr's frame id, strictly increasing from 1.
    pub unsafe fn frame(&mut self, gpu: &Gpu, chain: &mut Chain, frame_id: u64, cfg: &Cfg) -> Steps {
        unsafe {
            let mut s = Steps::default();
            let slot = (frame_id as usize) % SLOTS;
            let t0 = Instant::now();
            {
                let _sp = tracy_client::span!("begin_frame");
                if self.slot_value[slot] != 0 {
                    gpu.wait(self.timeline, self.slot_value[slot]);
                    if cfg.content && self.slot_frame[slot] != 0 {
                        let ms = self.read_ms(gpu, slot);
                        self.gpu_ms.push(ms);
                    }
                }
            }
            s.slot_wait = ms_since(t0);
            let t1 = Instant::now();
            if cfg.sleep && chain.low_latency {
                let _sp = tracy_client::span!("reflex_sleep");
                chain.sleep(gpu, self.sleep_sem, frame_id);
            }
            s.sleep = ms_since(t1);
            let t_start = Instant::now();
            let key = match cfg.key {
                Key::Frame => frame_id,
                Key::LastPresent => chain.next_id + u64::from(cfg.batch) - 1,
                Key::FirstPresent => chain.next_id,
            };
            let (sc, ll) = (chain.sc, chain.low_latency);
            let mark = |m: vk::LatencyMarkerNV| {
                if cfg.markers && ll {
                    gpu.latency.set_latency_marker(sc, &vk::SetLatencyMarkerInfoNV::default().present_id(key).marker(m));
                }
            };
            gpu.device.reset_command_pool(self.pools[slot], vk::CommandPoolResetFlags::empty()).unwrap();
            mark(vk::LatencyMarkerNV::SIMULATION_START);
            mark(vk::LatencyMarkerNV::SIMULATION_END);
            mark(vk::LatencyMarkerNV::RENDERSUBMIT_START);
            let cmd = self.cmds[slot][0];
            gpu.begin(cmd);
            let q0 = 2 * slot as u32;
            if cfg.content {
                gpu.device.cmd_reset_query_pool(cmd, self.queries, q0, 2);
                gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, self.queries, q0);
                let t = Instant::now();
                {
                    let _sp = tracy_client::span!("record_scene");
                    self.passes(gpu, cmd, 0, OUTPUT_FROM);
                }
                s.record_scene = ms_since(t);
                let t = Instant::now();
                if cfg.ngx {
                    if let (Some(n), Some(i)) = (self.ngx.as_mut(), self.inputs.as_ref()) {
                        let _sp = tracy_client::span!("ngx_rr");
                        if n.eval_rr(cmd, i, frame_id) != 1 {
                            s.ngx_fail += 1;
                        }
                    }
                    gpu.barrier(cmd);
                }
                s.ngx_rr = ms_since(t);
                let t = Instant::now();
                {
                    let _sp = tracy_client::span!("record_output");
                    self.passes(gpu, cmd, OUTPUT_FROM, PASSES);
                    let root = ComposeRoot {
                        extent: self.output,
                        encoding: chain.fmt.encoding,
                        source: 0,
                        tick: (frame_id * u64::from(cfg.batch)) as u32,
                        batch: cfg.batch,
                        paper: 250.0,
                        peak: cfg.peak,
                    };
                    self.compose_into(gpu, cmd, self.backbuffer.view, self.backbuffer.view, &root);
                    gpu.barrier(cmd);
                }
                s.record_output = ms_since(t);
                let t = Instant::now();
                if cfg.ngx && cfg.batch > 1 {
                    if let (Some(n), Some(i)) = (self.ngx.as_mut(), self.inputs.as_ref()) {
                        let _sp = tracy_client::span!("ngx_fg");
                        let count = cfg.batch - 1;
                        for k in 1..=count {
                            if n.eval_fg(cmd, &self.backbuffer, &i.depth, &i.motion, &self.interp[k as usize - 1], k, count, frame_id) != 1 {
                                s.ngx_fail += 1;
                            }
                        }
                        gpu.barrier(cmd);
                    }
                }
                s.ngx_fg = ms_since(t);
                gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, self.queries, q0 + 1);
            }
            gpu.device.end_command_buffer(cmd).unwrap();
            let t = Instant::now();
            {
                let _sp = tracy_client::span!("submit");
                self.value += 1;
                self.submit(gpu, cmd, &[], None, key, self.value);
            }
            s.submit = ms_since(t);
            self.slot_frame[slot] = if cfg.content { frame_id } else { 0 };
            mark(vk::LatencyMarkerNV::RENDERSUBMIT_END);

            let tp = Instant::now();
            mark(vk::LatencyMarkerNV::PRESENT_START);
            {
                let _sp = tracy_client::span!("present_path");
                for i in 0..cfg.batch {
                    let t = Instant::now();
                    let c0 = crate::win::cycles();
                    let acq = {
                        let _sp = tracy_client::span!("acquire");
                        chain.acquire(gpu, self.timeline)
                    };
                    let a = ms_since(t);
                    s.present_cpu += (crate::win::cycles() - c0) as f64 / self.cycles_per_ms;
                    s.acquire += a;
                    s.acquire_max = s.acquire_max.max(a);
                    let Ok((index, sem, aslot)) = acq else {
                        trace(&format!("acquire failed {acq:?} after {a:.1} ms"));
                        s.present_fail += 1;
                        continue;
                    };
                    let t = Instant::now();
                    let c = self.cmds[slot][1 + i as usize];
                    {
                        let _sp = tracy_client::span!("compose_ui");
                        gpu.begin(c);
                        let img = chain.images[index as usize];
                        gpu.layout_barrier(c, img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
                        let src = if cfg.content && cfg.ngx && self.ngx.as_ref().is_some_and(|n| n.fg_ok) && i + 1 < cfg.batch {
                            self.interp[i as usize].view
                        } else {
                            self.backbuffer.view
                        };
                        let root = ComposeRoot {
                            extent: chain.extent,
                            encoding: chain.fmt.encoding,
                            source: u32::from(cfg.content),
                            tick: (frame_id * u64::from(cfg.batch)) as u32 + i,
                            batch: cfg.batch,
                            paper: 250.0,
                            peak: cfg.peak,
                        };
                        self.compose_into(gpu, c, chain.views[index as usize], src, &root);
                        gpu.layout_barrier(c, img, vk::ImageLayout::GENERAL, vk::ImageLayout::PRESENT_SRC_KHR);
                        gpu.device.end_command_buffer(c).unwrap();
                    }
                    s.compose += ms_since(t);
                    let t = Instant::now();
                    {
                        let _sp = tracy_client::span!("present_submit");
                        self.value += 1;
                        self.submit(gpu, c, &[sem], Some(chain.present_sem(index)), key, self.value);
                        chain.consumed(aslot, self.value);
                    }
                    s.present_submit += ms_since(t);
                    let t = Instant::now();
                    let c0 = crate::win::cycles();
                    let (_, r, fb) = {
                        let _sp = tracy_client::span!("present");
                        let metered = if cfg.metered && i == 0 { Some(cfg.batch) } else { None };
                        chain.present(gpu, index, metered, cfg.relative_ns)
                    };
                    let p = ms_since(t);
                    s.present_cpu += (crate::win::cycles() - c0) as f64 / self.cycles_per_ms;
                    s.present += p;
                    s.present_max = s.present_max.max(p);
                    if i == 0 {
                        s.feedback = fb;
                    }
                    if r != vk::Result::SUCCESS && r != vk::Result::SUBOPTIMAL_KHR {
                        trace(&format!("present {r:?} feedback {fb}"));
                        s.present_fail += 1;
                    }
                }
            }
            mark(vk::LatencyMarkerNV::PRESENT_END);
            s.present_path = ms_since(tp);
            self.slot_value[slot] = self.value;
            s.total = ms_since(t_start);
            tracy_client::frame_mark();
            s
        }
    }

    pub unsafe fn idle(&self, gpu: &Gpu) {
        unsafe {
            gpu.wait(self.timeline, self.value);
        }
    }
}

/// Prints to stderr when `PROBE_TRACE` is set.
pub fn trace(what: &str) {
    if std::env::var("PROBE_TRACE").is_ok() {
        eprintln!("TRACE {what}");
    }
}

pub fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
