//! The device memory of a renderer with an empty scene and two views of it, a raster view and
//! a lit one, against what holds it: a table or a list of the scene's content holds nothing, so
//! its allocation is the smallest the device makes, and every allocation that is larger states
//! what it follows. The target holds this test alone: the device's ledger of live allocations
//! by label is then this renderer's.

use orr::{
    FrameDesc, FrameTime, Lens, LitView, PathEntry, Projection, Renderer, RendererDesc, SceneDesc,
    TextureFilter, ViewDesc, ViewHooks, ViewLighting, ViewOutput, ViewTarget,
};
use orr_gpu::{Format, MemoryKind, TextureDesc, TextureMemory, TextureUsage};
use orr_math::{DVec3, Quat, UVec2, Vec2};
use orr_scene::testing::{assert_clean, device, wait};

/// What an allocation follows when it is not a table or a list of the scene's content.
#[derive(Clone, Copy)]
enum Follows {
    /// The device: its heaps and what every device holds.
    Device,
    /// A numeric table, a record or a ring of roots of a size the code fixes.
    Fixed,
    /// The setting named.
    Setting(&'static str),
    /// The views' extents.
    Extent,
    /// Content the scene does not hold: a size stated up front, which is to follow the content
    /// instead.
    Reserved,
}

use Follows::{Device, Extent, Fixed, Reserved, Setting};

impl std::fmt::Display for Follows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device => f.write_str("the device"),
            Fixed => f.write_str("a fixed size"),
            Setting(name) => f.write_str(name),
            Extent => f.write_str("the views' extents"),
            Reserved => f.write_str("a size stated up front"),
        }
    }
}

const WORLD_CACHE: Follows = Setting("SceneLightingSettings::world_cache");
const CAUSTICS: Follows = Setting("SceneLightingSettings::caustics");
const MESH_PAGES: Follows = Setting("Budgets::mesh_page_pool");

/// Every label whose allocations are not the device's smallest, and what it follows.
const STATED: &[(&str, Follows)] = &[
    ("orr resource heap", Device),
    ("orr raygen records", Device),
    ("orr sampler heap", Device),
    ("orr empty TLAS", Device),
    ("orr empty TLAS scratch", Device),
    ("texture streams", Device),
    // The renderer.
    ("CLAS pool bits", MESH_PAGES),
    ("CLAS pool runs", MESH_PAGES),
    ("mesh page table", Reserved),
    ("mesh page bases", Reserved),
    ("texture feedback records", Reserved),
    ("texture feedback readback", Reserved),
    ("materials", Fixed),
    ("meshes", Fixed),
    ("bsdf tables", Fixed),
    ("ltc", Fixed),
    ("ideal_dielectric_energy_complement", Fixed),
    ("ideal_dielectric_avg_energy_complement", Fixed),
    ("ideal_dielectric_reflection_ratio", Fixed),
    ("ideal_metal_energy_complement", Fixed),
    ("ideal_metal_avg_energy_complement", Fixed),
    ("opaque_dielectric_energy_complement", Fixed),
    ("opaque_dielectric_avg_energy_complement", Fixed),
    ("tessellation patterns", Fixed),
    ("tessellation template addresses", Fixed),
    ("fft glare table", Fixed),
    ("glare tables", Fixed),
    ("subsurface table", Fixed),
    ("histogram", Fixed),
    ("output roots", Fixed),
    ("upload window", Setting("Budgets::upload_block")),
    // The scene and its lighting.
    ("scene root", Fixed),
    ("world cache samples", WORLD_CACHE),
    ("world cache radiance", WORLD_CACHE),
    ("world cache checks", WORLD_CACHE),
    ("world cache roots", Fixed),
    ("world cache resolve roots", Fixed),
    ("caustic grid deposits", CAUSTICS),
    ("caustic grid irradiance", CAUSTICS),
    ("caustic grid checks", CAUSTICS),
    ("caustic grid photons", CAUSTICS),
    ("producer caster cdf", CAUSTICS),
    ("caustic roots", Fixed),
    ("caustic resolve roots", Fixed),
    ("caster roots", Fixed),
    (
        "emissive cluster cdf",
        Setting("SceneLightingSettings::emissive_clusters"),
    ),
    ("light tiles", Fixed),
    ("light tile roots", Fixed),
    ("directional roots", Fixed),
    ("environment map", Fixed),
    ("environment pyramid", Fixed),
    ("environment partials", Fixed),
    ("environment pyramid roots", Fixed),
    ("environment bake roots", Fixed),
    // The scene's traversal and the cuts it builds.
    ("traversal queue", Reserved),
    ("candidate cut", Reserved),
    ("touched page entries", Reserved),
    ("wanted page entries", Reserved),
    ("wanted page errors", Reserved),
    ("wanted page buckets", Fixed),
    ("page requests", Setting("Budgets::traversal_requests")),
    ("cut readback", Reserved),
    ("cut args", Fixed),
    ("cut admission buckets", Fixed),
    ("cut keys", Reserved),
    ("built cut clusters", Reserved),
    ("cut CLAS addresses", Reserved),
    ("cut BLAS", Reserved),
    ("cut BLAS scratch", Reserved),
    ("cut BLAS records", Reserved),
    ("cut BLAS jobs", Reserved),
    ("cut BLAS addresses", Reserved),
    ("cut BLAS sizes", Reserved),
    ("cut part lists", Reserved),
    // The views.
    ("visible clusters", Reserved),
    ("opaque clusters", Reserved),
    ("alpha clusters", Reserved),
    ("software clusters", Reserved),
    ("particle clusters", Reserved),
    ("coverage clusters", Reserved),
    ("smoke clusters", Reserved),
    ("cluster work", Reserved),
    ("preprocess", Fixed),
    ("view targets", Extent),
    ("view image", Extent),
    ("hzb", Extent),
    ("depth", Extent),
    ("motion", Extent),
    ("staged path state", Extent),
    ("staged path records", Extent),
    ("staged update path records", Extent),
    ("staged path lists", Extent),
    ("staged path queues", Fixed),
    ("layer guides", Extent),
    ("light feedback", Extent),
    ("local light samplers", Extent),
    ("overlay front words", Extent),
    ("overlay depth", Extent),
    ("overlay motion", Extent),
    ("overlay weight", Extent),
    ("glare chain", Extent),
    ("blur tile rows", Extent),
    ("blur tiles", Extent),
    ("blur neighbors", Extent),
    ("frosted glass roots", Fixed),
    ("froxel roots", Fixed),
    ("froxel volumes", Fixed),
    ("guide roots", Fixed),
    ("layer tile roots", Fixed),
    ("local light sampler roots", Fixed),
    ("media apply roots", Fixed),
    ("overlay front roots", Fixed),
    ("particle lighting", Fixed),
];

const PROBE: &str = "smallest allocation";

/// Bytes of the smallest allocation of `kind`, by making one.
fn smallest(kind: MemoryKind) -> u64 {
    let _probe = device().gpu_malloc::<u8>(kind, 1, PROBE).unwrap();
    let ledger = device().memory_breakdown();
    let held = ledger
        .labels
        .iter()
        .find(|l| &*l.label == PROBE && l.kind == kind);
    held.expect("the probe is live").bytes
}

fn view(scene: orr::SceneId, history: orr::ViewHistoryId) -> ViewDesc {
    ViewDesc {
        scene,
        history,
        position: DVec3::ZERO,
        rotation: Quat::IDENTITY,
        projection: Projection {
            fov_y: 1.0,
            near: 0.1,
        },
        lens: Lens::default(),
        render_extent: UVec2::new(640, 360),
        output_extent: UVec2::new(640, 360),
        jitter: Vec2::ZERO,
        camera_cut: false,
        mip_bias: 0.0,
        texture_filter: TextureFilter::Hardware,
    }
}

/// Three frames of an empty scene through a raster view, then three through a lit one, leave
/// nothing in device memory but what [`STATED`] names and, for every table and list the scene
/// has nothing to put in, one allocation of the device's smallest size. A table or a list
/// sized before the scene holds anything for it fails here by its label, and so does a label
/// stated that holds no more than a table of nothing.
#[test]
fn an_empty_scene_holds_no_memory_for_content() {
    let kinds = MemoryKind::ALL;
    let granules = kinds.map(smallest);
    let desc = RendererDesc {
        path_entry: PathEntry::Compute,
        ..RendererDesc::default()
    };
    let mut r = Renderer::new(device(), desc).unwrap();
    let scene = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    let raster = r.create_raster_view_history().unwrap();
    let lit = r.create_view_history().unwrap();
    r.wait_paths().unwrap();
    let image = device()
        .create_texture(
            &TextureDesc::d2(
                Format::Rgba16Float,
                640,
                360,
                TextureUsage::SAMPLED | TextureUsage::STORAGE,
            ),
            TextureMemory::Dedicated,
            "view image",
        )
        .unwrap();
    let list = [scene];
    let mut last = None;
    for i in 0..6 {
        let mut f = r
            .begin_frame(FrameDesc {
                time: FrameTime::default(),
                scenes: &list,
                display: None,
            })
            .unwrap();
        f.update_scenes(&mut |_| {}).unwrap();
        if i < 3 {
            f.raster_view(&view(scene, raster)).unwrap();
        } else {
            let v = view(scene, lit);
            f.render_view(
                &LitView {
                    scene,
                    history: lit,
                    position: v.position,
                    rotation: v.rotation,
                    projection: v.projection,
                    lens: v.lens,
                    camera_cut: false,
                    not_rendering_scene: false,
                    screenshot: None,
                    texture_filter: v.texture_filter,
                    target: ViewTarget::Texture(&image),
                },
                &ViewLighting::default(),
                &ViewOutput::default(),
                &mut ViewHooks::default(),
            )
            .unwrap();
        }
        last = Some(f.end().unwrap());
    }
    wait(last.unwrap());

    let ledger = device().memory_breakdown();
    let mut wrong = Vec::new();
    for l in ledger.labels.iter().filter(|l| &*l.label != PROBE) {
        let kind = kinds.iter().position(|&k| k == l.kind);
        let granule = kind.map(|k| granules[k]);
        let empty = granule.is_some_and(|g| l.bytes == l.allocations * g);
        let stated = STATED.iter().find(|(label, _)| *label == &*l.label);
        match (stated, empty) {
            (None, false) => wrong.push(format!(
                "{:?} {:?}: {} B in {} allocations, with no content and nothing stated",
                l.kind, l.label, l.bytes, l.allocations
            )),
            (Some((_, follows)), true) => wrong.push(format!(
                "{:?} {:?}: stated to follow {follows}, but it holds nothing",
                l.kind, l.label
            )),
            _ => {}
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_clean();
}
