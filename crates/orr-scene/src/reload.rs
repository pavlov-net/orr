//! Hot reload: one scan job at a time on the loading side, applied at step 1.

use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use orr_gpu::{ComputePipeline, Device, ShaderModule};
use orr_shaders::{Reload, Watcher};

use crate::loading::Loading;
use crate::pipelines::{Code, Pipelines, ReloadReport};

/// The least time from one scan's start to the next.
const SCAN_PERIOD: Duration = Duration::from_millis(250);

/// What a scan job hands back.
struct Scanned {
    watcher: Watcher,
    /// The recipes the scan saw.
    recipes: usize,
    /// New pipelines by recipe index.
    swaps: Vec<(usize, ComputePipeline)>,
    code: Vec<(&'static ShaderModule, Arc<Code>)>,
    reports: Vec<ReloadReport>,
}

pub(crate) struct Reloader {
    /// `None` while a scan job holds it.
    watcher: Option<Watcher>,
    /// Modules to watch once the watcher is back.
    unwatched: Vec<&'static ShaderModule>,
    scan: Option<Arc<Mutex<Option<Scanned>>>>,
    started: Option<Instant>,
}

impl Reloader {
    pub(crate) fn new() -> Self {
        Self {
            watcher: Some(Watcher::new()),
            unwatched: Vec::new(),
            scan: None,
            started: None,
        }
    }

    pub(crate) fn watch(&mut self, module: &'static ShaderModule) {
        match &mut self.watcher {
            Some(w) => watch(w, module),
            None => self.unwatched.push(module),
        }
    }

    /// Step 1: swaps in the pipelines of a finished scan and queues the next scan when the
    /// last one started at least [`SCAN_PERIOD`] ago. Returns what the finished scan did.
    pub(crate) fn poll(
        &mut self,
        pipelines: &mut Pipelines,
        loading: &Loading,
        device: &Device,
    ) -> Vec<ReloadReport> {
        let mut reports = Vec::new();
        let done = self.scan.as_ref().and_then(|s| s.lock().unwrap().take());
        if let Some(done) = done {
            self.scan = None;
            for (index, pipeline) in done.swaps {
                pipelines.replace(index, pipeline);
            }
            // Pipelines created during the scan took the SPIR-V it replaced.
            for index in done.recipes..pipelines.recipes.len() {
                let recipe = Arc::clone(&pipelines.recipes[index]);
                let Some((_, code)) = done.code.iter().find(|(m, _)| ptr::eq(*m, recipe.module))
                else {
                    continue;
                };
                match recipe.create(device, Some(code)) {
                    Ok(p) => pipelines.replace(index, p),
                    Err(e) => log::warn!("hot reload of {} failed: {e}", recipe.module.name),
                }
            }
            for (module, code) in done.code {
                pipelines.reloaded(module, code);
            }
            let mut watcher = done.watcher;
            for m in self.unwatched.drain(..) {
                watch(&mut watcher, m);
            }
            self.watcher = Some(watcher);
            reports = done.reports;
        }
        if self.scan.is_none() && self.started.is_none_or(|t| t.elapsed() >= SCAN_PERIOD) {
            let mut watcher = self.watcher.take().expect("no scan holds the watcher");
            let recipes = pipelines.recipes.clone();
            let seen = recipes.len();
            let device = device.clone();
            let slot = Arc::new(Mutex::new(None));
            let out = Arc::clone(&slot);
            loading.spawn(move || {
                let (mut swaps, mut code, mut reports) = (Vec::new(), Vec::new(), Vec::new());
                for reload in watcher.scan() {
                    let kernel = reload.module().name;
                    let (module, new) = match reload {
                        Reload::Module {
                            module,
                            spirv,
                            spirv_omm,
                            ..
                        } => (module, Arc::new(Code { spirv, spirv_omm })),
                        Reload::Rejected { reason, .. } => {
                            log::warn!("hot reload of {kernel} rejected: {reason}");
                            reports.push(ReloadReport::Rejected { kernel, reason });
                            continue;
                        }
                    };
                    let made: crate::Result<Vec<_>> = recipes
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| ptr::eq(r.module, module))
                        .map(|(i, r)| Ok((i, r.create(&device, Some(&*new))?)))
                        .collect();
                    match made {
                        Ok(made) => {
                            log::info!("hot reload swaps {} pipelines of {kernel}", made.len());
                            reports.push(ReloadReport::Swapped {
                                kernel,
                                pipelines: made.len() as u32,
                            });
                            swaps.extend(made);
                            code.push((module, new));
                        }
                        Err(e) => {
                            let reason = format!("pipeline creation failed: {e}");
                            log::warn!("hot reload of {kernel} rejected: {reason}");
                            reports.push(ReloadReport::Rejected { kernel, reason });
                        }
                    }
                }
                *out.lock().unwrap() = Some(Scanned {
                    watcher,
                    recipes: seen,
                    swaps,
                    code,
                    reports,
                });
            });
            self.scan = Some(slot);
            self.started = Some(Instant::now());
        }
        reports
    }
}

fn watch(watcher: &mut Watcher, module: &'static ShaderModule) {
    if let Err(e) = watcher.watch(module) {
        log::warn!("hot reload cannot watch {}: {e}", module.name);
    }
}
