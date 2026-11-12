//! Headless GPU tests of orr_scene. Every test runs on `orr_scene::testing`'s device with
//! validation on and fails if the layer reports anything.

use std::sync::Arc;
use std::thread::JoinHandle;

use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshInput, MeshSettings, Source};
use orr_gpu::{ComputePipeline, ShaderModule, ShaderStage, SpecData};
use orr_math::{Vec2, Vec3};
use orr_pack::material::Material;
use orr_pack::{AssetId, Span, Target};
use orr_scene::testing::{device, mount_pending};
use orr_scene::{FrameDesc, FrameTime, LoadState, Loading, ModelId, PackSource, Scenes};

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// `hot/`, compiled from the copy in [`HOT_SRC`].
pub mod hot {
    include!(concat!(env!("OUT_DIR"), "/hot/orr_shaders.rs"));
}

/// The copy of `hot/` its modules compile from, for tests that edit it.
pub const HOT_SRC: &str = concat!(env!("OUT_DIR"), "/hot-src");

/// orr_scene's kernels with `surfaces/parallax.slang` linked as surface 1.
pub mod linked {
    include!(concat!(env!("OUT_DIR"), "/linked/orr_kernels.rs"));
}

/// Threads lent to the loading side until dropped.
pub struct Workers {
    loading: Arc<Loading>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    pub fn start(scenes: &Scenes, n: usize) -> Self {
        let loading = scenes.loading();
        let threads = (0..n)
            .map(|_| {
                let l = Arc::clone(&loading);
                std::thread::spawn(move || l.run_worker())
            })
            .collect();
        Self { loading, threads }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.loading.stop_workers();
        for t in self.threads.drain(..) {
            t.join().unwrap();
        }
    }
}

/// The compute pipeline of `module`'s `entry`.
pub fn pipeline(module: &'static ShaderModule, entry: &str) -> ComputePipeline {
    let stage = ShaderStage::new(module, entry, SpecData::default()).unwrap();
    device()
        .create_compute_pipeline(&stage, module.name)
        .unwrap()
}

/// Mounts a pack of one opaque model, a unit quad of one section, and returns the model once
/// the mount is ready, with the workers that loaded it.
pub fn mount_quad(scenes: &mut Scenes) -> (Workers, ModelId) {
    let mesh = MeshInput {
        positions: vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::new(1.0, 0.0, 1.0)],
        normals: vec![Vec3::Y; 4],
        uvs: Some(vec![Vec2::ZERO; 4]),
        indices: vec![0, 2, 1, 1, 2, 3],
        sections: vec![Span { first: 0, count: 6 }],
        ..MeshInput::default()
    };
    let asset = AssetSpec {
        name: "quad".into(),
        source: Source::Mesh {
            mesh,
            materials: vec![Material::default()],
            settings: MeshSettings::default(),
        },
    };
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, vec![asset])],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let workers = Workers::start(scenes, 1);
    let m = scenes.mount(PackSource::Memory(bytes));
    let frame = FrameDesc {
        time: FrameTime::default(),
        scenes: &[],
        low_latency: None,
    };
    while mount_pending(scenes, m) {
        scenes.begin_frame(&frame).unwrap().end(&[]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    if let LoadState::Failed(e) = scenes.mount_state(m) {
        panic!("the quad failed to mount: {e}");
    }
    (
        workers,
        scenes.model(m, AssetId::from_name("quad")).unwrap(),
    )
}
