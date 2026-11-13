//! The support kernels made for what a scene may hold: the guide resolve, the layer guides, the
//! froxels' light pass with the thin shadow tracer, and media apply. Each inlines hit
//! resolution, surface evaluation or shadow rays, so like the staged path kernels it compiles for
//! a mask (`specialize.slang`) when a view first needs it, and a view renders lit once the ones
//! its frame runs are ready.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use orr_scene::orr_gpu::{ComputePipeline, Device, ShaderModule, ShaderStage, SpecData};
use orr_scene::{Loading, SceneFeatures, Urgency};
use orr_shaders::abi::SpecEntry;

use crate::media::links_medium;
use crate::shaders::dielectric::layer_guides;
use crate::shaders::guides::resolve;
use crate::shaders::media::{apply, froxel_light};
use crate::shaders::rays::shadow;
use crate::staged::{Late, MAY_ALL, PATH_MAY_ID};
use crate::{Error, KernelSet, module};

/// A support kernel made per mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Masked {
    Guides,
    LayerGuides,
    FroxelWalk,
    FroxelSample,
    ShadowTrace,
    FroxelRetrace,
    FroxelAccumulate,
    MediaApply,
}

const MASKED: [Masked; 8] = [
    Masked::Guides,
    Masked::LayerGuides,
    Masked::FroxelWalk,
    Masked::FroxelSample,
    Masked::ShadowTrace,
    Masked::FroxelRetrace,
    Masked::FroxelAccumulate,
    Masked::MediaApply,
];

/// The froxels' light pass, which runs from one set.
pub(crate) const FROXEL_LIGHT: [Masked; 5] = [
    Masked::FroxelWalk,
    Masked::FroxelSample,
    Masked::ShadowTrace,
    Masked::FroxelRetrace,
    Masked::FroxelAccumulate,
];

/// A masked kernel's module, the layout of the module's specialization constants and their
/// default bytes, its entry point and its name.
type Source = (
    &'static ShaderModule,
    &'static [SpecEntry],
    Vec<u8>,
    &'static str,
    &'static str,
);

impl Masked {
    fn source(self) -> Source {
        let froxel = |entry, name| {
            (
                &froxel_light::MODULE,
                froxel_light::Spec::ENTRIES,
                froxel_light::Spec::default().bytes().to_vec(),
                entry,
                name,
            )
        };
        match self {
            Self::Guides => (
                &resolve::MODULE,
                resolve::Spec::ENTRIES,
                resolve::Spec::default().bytes().to_vec(),
                "main",
                "guide resolve",
            ),
            Self::LayerGuides => (
                &layer_guides::MODULE,
                layer_guides::Spec::ENTRIES,
                layer_guides::Spec::default().bytes().to_vec(),
                "main",
                "layer guides",
            ),
            Self::FroxelWalk => froxel("walk_main", "froxel walk"),
            Self::FroxelSample => froxel("sample_main", "froxel sample"),
            Self::ShadowTrace => (
                &shadow::MODULE,
                shadow::Spec::ENTRIES,
                shadow::Spec::default().bytes().to_vec(),
                "main",
                "shadow trace",
            ),
            Self::FroxelRetrace => froxel("retrace_main", "froxel retrace"),
            Self::FroxelAccumulate => froxel("accumulate_main", "froxel accumulate"),
            Self::MediaApply => (
                &apply::MODULE,
                apply::Spec::ENTRIES,
                apply::Spec::default().bytes().to_vec(),
                "main",
                "media apply",
            ),
        }
    }
}

/// Compiles `kernel` for a scene that may hold `may`.
fn compile(
    device: &Device,
    kernels: &KernelSet,
    kernel: Masked,
    may: u64,
) -> Result<ComputePipeline, Error> {
    let (own, entries, mut bytes, entry, name) = kernel.source();
    // The mask's low word is `path_may`, which every masked kernel takes, and its high word
    // `path_may2`, which a kernel takes where it evaluates surfaces or resolves hits.
    let word = |id| entries.iter().find(|e| e.constant_id == id);
    let at = word(PATH_MAY_ID).expect("a masked support kernel takes the mask");
    bytes[at.offset as usize..][..4].copy_from_slice(&(may as u32).to_le_bytes());
    if let Some(at) = word(SceneFeatures::SURFACE_MAY_ID) {
        bytes[at.offset as usize..][..4].copy_from_slice(&((may >> 32) as u32).to_le_bytes());
    }
    let spec = SpecData {
        entries,
        bytes: &bytes,
    };
    let started = Instant::now();
    let stage = ShaderStage::new(module(kernels, own)?, entry, spec)?;
    let pipeline = device.create_compute_pipeline(&stage, &format!("{name} {may:#x}"))?;
    log::debug!(
        "{name} for {may:#x} made in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    Ok(pipeline)
}

/// Which of the masked kernels a view's frame runs beside the guide resolve, which every
/// frame runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Runs {
    pub(crate) layer_guides: bool,
    pub(crate) froxel_light: bool,
    pub(crate) media_apply: bool,
}

impl Runs {
    const ALL: Self = Self {
        layer_guides: true,
        froxel_light: true,
        media_apply: true,
    };

    fn kernels(self) -> impl Iterator<Item = Masked> {
        MASKED.into_iter().filter(move |k| match k {
            Masked::Guides => true,
            Masked::LayerGuides => self.layer_guides,
            Masked::FroxelWalk
            | Masked::FroxelSample
            | Masked::ShadowTrace
            | Masked::FroxelRetrace
            | Masked::FroxelAccumulate => self.froxel_light,
            Masked::MediaApply => self.media_apply,
        })
    }
}

/// The masked kernels, the mask a frame runs them with and which of them it runs.
#[derive(Clone, Copy)]
pub(crate) struct MaskedFrame<'a> {
    pub(crate) kernels: &'a MaskedSupport,
    pub(crate) may: u64,
    pub(crate) runs: Runs,
}

impl MaskedFrame<'_> {
    /// [`MaskedSupport::take`] for the frame's mask.
    pub(crate) fn take(self, kernel: Masked) -> Result<Arc<SupportSet>, Error> {
        self.kernels.take(self.may, &[kernel])
    }

    /// [`MaskedSupport::take`] of `kernels` together, for the frame's mask.
    pub(crate) fn take_all(self, kernels: &[Masked]) -> Result<Arc<SupportSet>, Error> {
        self.kernels.take(self.may, kernels)
    }
}

/// The masked kernels' pipelines for one mask, each made when a view first runs it.
pub(crate) struct SupportSet {
    may: u64,
    pipelines: [Late<ComputePipeline>; 8],
}

impl SupportSet {
    fn new(may: u64) -> Self {
        Self {
            may,
            pipelines: MASKED.map(|k| Late::new(k.source().4)),
        }
    }

    fn late(&self, kernel: Masked) -> &Late<ComputePipeline> {
        &self.pipelines[kernel as usize]
    }

    /// `kernel`'s pipeline once it has compiled.
    pub(crate) fn get(&self, kernel: Masked) -> Option<&ComputePipeline> {
        self.late(kernel).get()
    }

    /// What the set's pipelines may meet.
    pub(crate) fn may(&self) -> u64 {
        self.may
    }

    fn holds(&self, runs: Runs) -> bool {
        runs.kernels().all(|k| self.get(k).is_some())
    }
}

/// The masked support kernels of one device: a set of pipelines per mask a view has asked
/// for, kept while the kernels live. The masks of a scene are few, and a set a view moved on
/// from is the one it comes back to when the scene loses what it gained.
pub(crate) struct MaskedSupport {
    device: Device,
    kernels: KernelSet,
    linked_medium: bool,
    /// The device and loading side pipelines compile on once [`Self::attach`] names them; on
    /// `device` and the calling thread before that.
    compiler: OnceLock<(Device, Arc<Loading>)>,
    sets: Mutex<Vec<Arc<SupportSet>>>,
    /// A view's pass has declared [`crate::ViewPass::particle_lighting`], which a view's
    /// request ahead of its pass cannot know.
    particle_lighting: AtomicBool,
}

impl MaskedSupport {
    pub(crate) fn new(device: &Device, kernels: &KernelSet) -> Self {
        Self {
            device: device.clone(),
            kernels: *kernels,
            linked_medium: links_medium(kernels),
            compiler: OnceLock::new(),
            sets: Mutex::new(Vec::new()),
            particle_lighting: AtomicBool::new(false),
        }
    }

    /// Whether the kernels link a medium.
    pub(crate) fn linked_medium(&self) -> bool {
        self.linked_medium
    }

    /// Makes later pipelines on `device` as jobs on `loading`.
    pub(crate) fn attach(&self, device: &Device, loading: &Arc<Loading>) {
        let _ = self.compiler.set((device.clone(), Arc::clone(loading)));
    }

    /// Notes that a pass declared particle lighting, and says whether one had before.
    pub(crate) fn declare_particle_lighting(&self) -> bool {
        self.particle_lighting.swap(true, Ordering::Relaxed)
    }

    /// Whether a pass has declared particle lighting.
    pub(crate) fn particle_lighting(&self) -> bool {
        self.particle_lighting.load(Ordering::Relaxed)
    }

    fn set(&self, may: u64) -> Arc<SupportSet> {
        let mut sets = self.sets.lock().unwrap();
        if let Some(set) = sets.iter().find(|s| s.may == may) {
            return Arc::clone(set);
        }
        let set = Arc::new(SupportSet::new(may));
        sets.push(Arc::clone(&set));
        set
    }

    fn start(&self, set: &SupportSet, runs: Runs, urgency: Urgency) {
        let (device, loading) = match self.compiler.get() {
            Some((device, loading)) => (device, Some((&**loading, urgency))),
            None => (&self.device, None),
        };
        for kernel in runs.kernels() {
            let (kernels, may) = (self.kernels, set.may);
            set.late(kernel)
                .start(device, loading, move |d| compile(d, &kernels, kernel, may));
        }
    }

    /// The first compile failure among `runs`' kernels of `set`.
    fn failure(set: &SupportSet, runs: Runs) -> Result<(), Error> {
        match runs.kernels().find_map(|k| set.late(k).failure()) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Asks for the pipelines of `runs` that hold `may`, ahead of everything else the loading
    /// side has, and behind those the ones that may meet everything where `general`: they run a
    /// scene that comes to hold more than its own were made for. Says whether pipelines that
    /// hold `may` are ready for all of `runs`.
    pub(crate) fn ask(&self, may: u64, runs: Runs, general: bool) -> Result<bool, Error> {
        let exact = self.set(may);
        self.start(&exact, runs, Urgency::Needed);
        Self::failure(&exact, runs)?;
        if exact.holds(runs) {
            return Ok(true);
        }
        // With no loading side a pipeline compiles where it is first run.
        if !general || may == MAY_ALL || self.compiler.get().is_none() {
            return Ok(false);
        }
        let all = self.set(MAY_ALL);
        self.start(&all, Runs::ALL, Urgency::Useful);
        Ok(all.holds(runs))
    }

    /// Queues every pipeline that may meet everything, behind what views cannot render without.
    pub(crate) fn start_general(&self) {
        if self.compiler.get().is_some() {
            self.start(&self.set(MAY_ALL), Runs::ALL, Urgency::Useful);
        }
    }

    /// The set whose `kernels` a frame that may meet `may` runs, all from one set: the one made
    /// for `may` once their pipelines are ready, else the one that may meet everything where
    /// they are, else the one made for `may` after waiting for them, running queued loading jobs
    /// on this thread meanwhile.
    pub(crate) fn take(&self, may: u64, kernels: &[Masked]) -> Result<Arc<SupportSet>, Error> {
        let holds = |set: &SupportSet| kernels.iter().all(|&k| set.get(k).is_some());
        let exact = self.set(may);
        if holds(&exact) {
            return Ok(exact);
        }
        let all = self
            .sets
            .lock()
            .unwrap()
            .iter()
            .find(|s| s.may == MAY_ALL)
            .cloned();
        if let Some(all) = all.filter(|all| holds(all)) {
            return Ok(all);
        }
        for &kernel in kernels {
            self.wait(&exact, kernel)?;
        }
        Ok(exact)
    }

    fn wait(&self, set: &SupportSet, kernel: Masked) -> Result<(), Error> {
        let (device, loading) = match self.compiler.get() {
            Some((device, loading)) => (device, Some(&**loading)),
            None => (&self.device, None),
        };
        let (kernels, may) = (self.kernels, set.may);
        set.late(kernel)
            .start(device, loading.map(|l| (l, Urgency::Needed)), move |d| {
                compile(d, &kernels, kernel, may)
            });
        set.late(kernel).wait(loading).map(|_| ())
    }

    /// Waits for the pipelines of `runs` made for exactly `may`.
    pub(crate) fn wait_exact(&self, may: u64, runs: Runs) -> Result<(), Error> {
        let exact = self.set(may);
        for kernel in runs.kernels() {
            self.wait(&exact, kernel)?;
        }
        Ok(())
    }

    /// How many pipelines a frame or a caller waits for are still compiling: those asked for as
    /// needed that no pipeline that may meet everything stands in for ([`Self::take`]), but
    /// where `exact` (a caller that waits takes those made for its mask, [`Self::wait_exact`]).
    pub(crate) fn outstanding(&self, exact: bool) -> u32 {
        let sets = self.sets.lock().unwrap();
        let general = sets.iter().find(|s| s.may == MAY_ALL).filter(|_| !exact);
        let awaited = |s: &SupportSet, k: Masked| {
            s.late(k).needed() && !general.is_some_and(|g| g.get(k).is_some())
        };
        sets.iter()
            .map(|s| MASKED.into_iter().filter(|&k| awaited(s, k)).count())
            .sum::<usize>() as u32
    }
}
