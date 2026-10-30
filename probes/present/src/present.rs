//! Swapchain with present_id2/present_wait2 ids, optional low_latency2, present timing and
//! present metering, the shapes contracts.md 12 and 13 describe.

use crate::gpu::Gpu;
use ash::vk::{self, TaggedStructure};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Format {
    pub format: vk::Format,
    pub space: vk::ColorSpaceKHR,
    pub encoding: u32,
    pub name: &'static str,
}

pub const SDR: Format = Format { format: vk::Format::B8G8R8A8_UNORM, space: vk::ColorSpaceKHR::SRGB_NONLINEAR, encoding: 0, name: "SDR" };
pub const PQ: Format = Format { format: vk::Format::A2B10G10R10_UNORM_PACK32, space: vk::ColorSpaceKHR::HDR10_ST2084_EXT, encoding: 1, name: "PQ" };
pub const SCRGB: Format = Format { format: vk::Format::R16G16B16A16_SFLOAT, space: vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT, encoding: 2, name: "scRGB" };

pub struct Opts {
    pub mode: vk::PresentModeKHR,
    pub min_images: u32,
    pub low_latency: bool,
    pub timing: bool,
    /// Request present-stage timestamps on each present. On 621.01 this stops
    /// vkWaitForPresent2KHR from returning while presents continue.
    pub queries: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { mode: vk::PresentModeKHR::FIFO, min_images: 5, low_latency: true, timing: true, queries: true }
    }
}

pub struct Chain {
    pub sc: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub fmt: Format,
    pub extent: [u32; 2],
    /// Acquire semaphores and the render timeline value whose submit waited each.
    acquire: Vec<(vk::Semaphore, u64)>,
    ring: usize,
    present_sems: Vec<vk::Semaphore>,
    pub next_id: u64,
    pub domain: Option<(vk::TimeDomainKHR, u64)>,
    pub stages: vk::PresentStageFlagsEXT,
    pub low_latency: bool,
    pub timing: bool,
    pub queries: bool,
}

/// One past presentation: present id and (stage, time) pairs.
#[derive(Clone, Debug)]
pub struct Past {
    pub id: u64,
    pub stages: Vec<(vk::PresentStageFlagsEXT, u64)>,
}

impl Past {
    pub fn at(&self, stage: vk::PresentStageFlagsEXT) -> Option<u64> {
        self.stages.iter().find(|s| s.0 == stage && s.1 != 0).map(|s| s.1)
    }
}

pub unsafe fn surface_caps(gpu: &Gpu, surface: vk::SurfaceKHR) -> (vk::SurfaceCapabilitiesKHR, vk::PresentTimingSurfaceCapabilitiesEXT<'static>, bool, bool) {
    unsafe {
        let mut timing = vk::PresentTimingSurfaceCapabilitiesEXT::default();
        let mut id2 = vk::SurfaceCapabilitiesPresentId2KHR::default();
        let mut wait2 = vk::SurfaceCapabilitiesPresentWait2KHR::default();
        let info = vk::PhysicalDeviceSurfaceInfo2KHR::default().surface(surface);
        let mut caps = vk::SurfaceCapabilities2KHR::default().push(&mut timing).push(&mut id2).push(&mut wait2);
        gpu.caps2_i.get_physical_device_surface_capabilities2(gpu.pdev, &info, &mut caps).unwrap();
        let c = caps.surface_capabilities;
        timing.p_next = std::ptr::null_mut();
        (c, timing, id2.present_id2_supported == vk::TRUE, wait2.present_wait2_supported == vk::TRUE)
    }
}

impl Chain {
    pub unsafe fn new(gpu: &Gpu, surface: vk::SurfaceKHR, fmt: Format, opts: &Opts, old: Option<Chain>) -> Chain {
        unsafe {
            let (caps, tcaps, _, _) = surface_caps(gpu, surface);
            let extent = caps.current_extent;
            let mut flags = vk::SwapchainCreateFlagsKHR::PRESENT_ID_2 | vk::SwapchainCreateFlagsKHR::PRESENT_WAIT_2;
            if opts.timing {
                flags |= vk::SwapchainCreateFlagsKHR::PRESENT_TIMING_EXT;
            }
            let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT;
            let mut latency = vk::SwapchainLatencyCreateInfoNV::default().latency_mode_enable(true);
            let modes = [opts.mode];
            let mut modes_ci = vk::SwapchainPresentModesCreateInfoKHR::default().present_modes(&modes);
            let old_sc = old.as_ref().map_or(vk::SwapchainKHR::null(), |o| o.sc);
            let mut ci = vk::SwapchainCreateInfoKHR::default()
                .flags(flags)
                .surface(surface)
                .min_image_count(opts.min_images.max(caps.min_image_count))
                .image_format(fmt.format)
                .image_color_space(fmt.space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(usage)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(opts.mode)
                .clipped(true)
                .old_swapchain(old_sc)
                .push(&mut modes_ci);
            if opts.low_latency {
                ci = ci.push(&mut latency);
            }
            let sc = gpu.swapchain.create_swapchain(&ci, None).unwrap_or_else(|e| panic!("swapchain {}: {e:?}", fmt.name));
            if let Some(o) = old {
                o.destroy(gpu);
            }
            let images = gpu.swapchain.get_swapchain_images(sc).unwrap();
            let views = images.iter().map(|&i| gpu.view(i, fmt.format)).collect();
            let acquire = (0..images.len() + 2).map(|_| (gpu.binary(), 0)).collect();
            let present_sems = images.iter().map(|_| gpu.binary()).collect();
            let mut chain = Chain {
                sc,
                images,
                views,
                fmt,
                extent: [extent.width, extent.height],
                acquire,
                ring: 0,
                present_sems,
                next_id: 1,
                domain: None,
                stages: tcaps.present_stage_queries,
                low_latency: opts.low_latency,
                timing: opts.timing,
                queries: opts.queries,
            };
            if opts.timing {
                let r = (gpu.timing.set_queue_size)(gpu.device.handle(), sc, 256);
                assert_eq!(r, vk::Result::SUCCESS, "present timing queue size");
                chain.domain = chain.pick_domain(gpu);
            }
            chain
        }
    }

    /// QueryPerformanceCounter if the swapchain offers it, else the first present-stage-local
    /// domain.
    unsafe fn pick_domain(&self, gpu: &Gpu) -> Option<(vk::TimeDomainKHR, u64)> {
        let ds = unsafe { self.time_domains(gpu) };
        ds.iter()
            .find(|d| d.0 == vk::TimeDomainKHR::QUERY_PERFORMANCE_COUNTER)
            .or_else(|| ds.iter().find(|d| d.0 == vk::TimeDomainKHR::PRESENT_STAGE_LOCAL_EXT))
            .or(ds.first())
            .copied()
    }

    pub unsafe fn time_domains(&self, gpu: &Gpu) -> Vec<(vk::TimeDomainKHR, u64)> {
        unsafe {
            let mut p = vk::SwapchainTimeDomainPropertiesEXT::default();
            let r = (gpu.timing.domains)(gpu.device.handle(), self.sc, &mut p, std::ptr::null_mut());
            if r != vk::Result::SUCCESS || p.time_domain_count == 0 {
                return vec![];
            }
            let n = p.time_domain_count as usize;
            let mut doms = vec![vk::TimeDomainKHR::default(); n];
            let mut ids = vec![0u64; n];
            p.p_time_domains = doms.as_mut_ptr();
            p.p_time_domain_ids = ids.as_mut_ptr();
            let _ = (gpu.timing.domains)(gpu.device.handle(), self.sc, &mut p, std::ptr::null_mut());
            doms.into_iter().zip(ids).take(p.time_domain_count as usize).collect()
        }
    }

    /// (refreshDuration, refreshInterval, result).
    pub unsafe fn timing_properties(&self, gpu: &Gpu) -> (u64, u64, vk::Result) {
        unsafe {
            let mut p = vk::SwapchainTimingPropertiesEXT::default();
            let mut counter = 0u64;
            let r = (gpu.timing.properties)(gpu.device.handle(), self.sc, &mut p, &mut counter);
            (p.refresh_duration, p.refresh_interval, r)
        }
    }

    /// Acquires an image. Returns (index, acquire semaphore, ring slot). `done` is the render
    /// timeline; a slot is reused only after the submit that waited its semaphore finished.
    pub unsafe fn acquire(&mut self, gpu: &Gpu, done: vk::Semaphore) -> Result<(u32, vk::Semaphore, usize), vk::Result> {
        unsafe {
            let slot = self.ring;
            self.ring = (self.ring + 1) % self.acquire.len();
            let (sem, v) = self.acquire[slot];
            if v != 0 {
                gpu.wait(done, v);
            }
            let (index, _) = gpu.swapchain.acquire_next_image(self.sc, 5_000_000_000, sem, vk::Fence::null())?;
            Ok((index, sem, slot))
        }
    }

    pub fn consumed(&mut self, slot: usize, value: u64) {
        self.acquire[slot].1 = value;
    }

    pub fn present_sem(&self, index: u32) -> vk::Semaphore {
        self.present_sems[index as usize]
    }

    /// Presents `index`, waiting its present semaphore. `metered` chains VkSetPresentConfigNV;
    /// `relative_ns` > 0 requests a relative-time present. Returns (present id, result,
    /// presentConfigFeedback).
    pub unsafe fn present(&mut self, gpu: &Gpu, index: u32, metered: Option<u32>, relative_ns: u64) -> (u64, vk::Result, u32) {
        unsafe {
            let id = self.next_id;
            self.next_id += 1;
            let ids = [id];
            let scs = [self.sc];
            let idx = [index];
            let waits = [self.present_sems[index as usize]];
            let mut pid = vk::PresentId2KHR::default().present_ids(&ids);
            let mut meter = vk::SetPresentConfigNV::default().num_frames_per_batch(metered.unwrap_or(0));
            let mut ti = [vk::PresentTimingInfoEXT::default()];
            if let Some((_, dom)) = self.domain {
                ti[0].time_domain_id = dom;
                ti[0].present_stage_queries = if self.queries { self.stages } else { vk::PresentStageFlagsEXT::empty() };
                if relative_ns > 0 {
                    ti[0].flags = vk::PresentTimingInfoFlagsEXT::PRESENT_AT_RELATIVE_TIME;
                    ti[0].target_time = relative_ns;
                    ti[0].target_time_domain_present_stage = vk::PresentStageFlagsEXT::IMAGE_FIRST_PIXEL_OUT;
                }
            }
            let mut timings = vk::PresentTimingsInfoEXT::default().timing_infos(&ti);
            let mut info = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&scs).image_indices(&idx).push(&mut pid);
            if metered.is_some() {
                info = info.push(&mut meter);
            }
            if self.timing && self.domain.is_some() && (self.queries || relative_ns > 0) {
                info = info.push(&mut timings);
            }
            let r = match gpu.swapchain.queue_present(gpu.queues[0], &info) {
                Ok(false) => vk::Result::SUCCESS,
                Ok(true) => vk::Result::SUBOPTIMAL_KHR,
                Err(e) => e,
            };
            (id, r, meter.present_config_feedback)
        }
    }

    /// Every timing record available now.
    pub unsafe fn past(&self, gpu: &Gpu) -> Vec<Past> {
        unsafe {
            let mut out = Vec::new();
            loop {
                let info = vk::PastPresentationTimingInfoEXT::default().swapchain(self.sc);
                let n = 64;
                let mut stages = vec![[vk::PresentStageTimeEXT::default(); 4]; n];
                let mut recs: Vec<vk::PastPresentationTimingEXT> = (0..n)
                    .map(|i| {
                        let mut r = vk::PastPresentationTimingEXT::default();
                        r.present_stage_count = 4;
                        r.p_present_stages = stages[i].as_mut_ptr();
                        r
                    })
                    .collect();
                let mut props = vk::PastPresentationTimingPropertiesEXT::default();
                props.presentation_timing_count = n as u32;
                props.p_presentation_timings = recs.as_mut_ptr();
                let r = (gpu.timing.past)(gpu.device.handle(), &info, &mut props);
                for rec in recs.iter().take(props.presentation_timing_count as usize) {
                    let st = std::slice::from_raw_parts(rec.p_present_stages, rec.present_stage_count as usize);
                    out.push(Past { id: rec.present_id, stages: st.iter().map(|s| (s.stage, s.time)).collect() });
                }
                if r != vk::Result::INCOMPLETE {
                    break;
                }
            }
            out
        }
    }

    pub unsafe fn sleep_mode(&self, gpu: &Gpu, on: bool, boost: bool, min_interval_us: u32) {
        unsafe {
            let info = vk::LatencySleepModeInfoNV::default().low_latency_mode(on).low_latency_boost(boost).minimum_interval_us(min_interval_us);
            gpu.latency.set_latency_sleep_mode(self.sc, Some(&info)).expect("sleep mode");
        }
    }

    /// vkLatencySleepNV signalling `sem` = `value`, then a host wait on it.
    pub unsafe fn sleep(&self, gpu: &Gpu, sem: vk::Semaphore, value: u64) {
        unsafe {
            let info = vk::LatencySleepInfoNV::default().signal_semaphore(sem).value(value);
            gpu.latency.latency_sleep(self.sc, &info).expect("latency sleep");
            gpu.wait(sem, value);
        }
    }

    pub unsafe fn latency_timings(&self, gpu: &Gpu) -> Vec<vk::LatencyTimingsFrameReportNV<'static>> {
        unsafe {
            let mut v = vec![vk::LatencyTimingsFrameReportNV::default(); 64];
            let mut info = vk::GetLatencyMarkerInfoNV::default();
            info.timing_count = v.len() as u32;
            info.p_timings = v.as_mut_ptr();
            gpu.latency.get_latency_timings(self.sc, &mut info);
            v.truncate(info.timing_count as usize);
            v
        }
    }

    /// Waits for the last present to complete (present_wait2) and the device to idle first.
    pub unsafe fn destroy(self, gpu: &Gpu) {
        unsafe {
            if self.next_id > 1 {
                let w = vk::PresentWait2InfoKHR::default().present_id(self.next_id - 1).timeout(1_000_000_000);
                let _ = gpu.wait2.wait_for_present2(self.sc, &w);
            }
            gpu.device.device_wait_idle().unwrap();
            for v in &self.views {
                gpu.device.destroy_image_view(*v, None);
            }
            for (s, _) in &self.acquire {
                gpu.device.destroy_semaphore(*s, None);
            }
            for s in &self.present_sems {
                gpu.device.destroy_semaphore(*s, None);
            }
            gpu.swapchain.destroy_swapchain(self.sc, None);
        }
    }
}
