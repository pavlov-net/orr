//! One tree asset swaying in the wind, path traced on a ground under a cooked sky and a sun.
//! The tree is a glTF file, cooked in process into an archive under the OS temp directory
//! (kept up to date by the cook's own state, so later runs start at once), or an archive
//! cooked earlier. Its rigid parts come from a rigid skin (each vertex on one joint; the
//! skeleton gives pivots and the hierarchy), or with `--part-attribute` from a per-vertex
//! integer attribute such as `_ID` (every part then hangs on the body about its cooked
//! pivot). The example's kernel writes the parts' poses into a GPU-written pose group in
//! scene compute every frame; raster and ray tracing both take them. A joint node's glTF
//! extras may hold wind hints, `{"wind": {"stiffness": s, "flutter": f}}` with both in
//! 0..1; without them a joint bends more the deeper it is in the skeleton. The camera
//! frames the tree's bounds and orbits it, or with `--still` stays put. Where NGX is available
//! DLSS ray reconstruction denoises it at the mode that suits the output size.
//!
//! `tree-viewer <file.glb | file.gltf | archive.orr> [--model NAME] [--node PATTERN]
//! [--part-attribute NAME] [--still] [--frames N] [--headless] [--out file.png]`: `--node`
//! cooks each glTF node whose name holds the pattern as its own model (not with skins),
//! `--model` picks the archive's model (default: the first by name). Headless runs output
//! 480 x 270 for 120 frames unless told otherwise and write the last frame as a PNG.

use std::collections::HashMap;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use examples::mesh::MeshBuilder;
use examples::scene::{FOV_Y, SUN, sun};
use examples::shaders::tree_wind;
use examples::sky::{self, Sky};
use examples::window::Window;
use examples::{
    Capture, Options, Workers, assert_clean, device, enable_dlss, entry_head, fixed_time,
    mount_all_at,
};
use orr::orr_gpu::{
    Format, MemoryKind, QueueKind, SpecData, TextureDesc, TextureMemory, TextureUsage,
    TimelinePoint,
};
use orr::orr_pack::material::Material;
use orr::orr_pack::mesh::MeshInfo;
use orr::orr_pack::model::ModelInfo;
use orr::orr_pack::skeleton::{MAX_JOINTS, NO_PARENT, SkeletonInfo};
use orr::orr_pack::{Archive, AssetId, Class, Target};
use orr::{
    Affine3, CompilingKernels, Display, DisplayRequest, FrameDesc, FrameTime, GroupWriter,
    InstanceFlags, InstancesDesc, Lens, LightId, LitView, PackSource, Pose, Projection, Renderer,
    RendererDesc, SceneDesc, SceneLightingSettings, ScreenshotRequest, TextureFilter,
    TransformGroupDesc, ViewHooks, ViewLighting, ViewOutput, ViewTarget,
};
use orr_cook::{
    ArchiveSpec, AssetSpec, CookOptions, GltfSettings, Manifest, MeshSettings, NodeSelect,
    PartAttribute, Source,
};
use orr_math::camera::rh::view;
use orr_math::{DVec3, Mat4, Vec2, Vec3, Vec4};

const TREE: &str = "tree";
const GROUND: &str = "ground";
const SKY: &str = "sky";

const HEADLESS_EXTENT: [u32; 2] = [480, 270];
const HEADLESS_FRAMES: u64 = 120;
/// The frame id the scene is created in, after the packs have mounted.
const FIRST_FRAME: u64 = 500;
/// Unit xz direction the wind blows toward.
const WIND: Vec2 = Vec2::new(0.8, 0.6);
/// Rad the most flexible chain of joints leans at its tip in a full gust, summed over it.
const TIP_BEND: f32 = 0.22;
/// Rad of flutter on the joints that flutter most.
const FLUTTER: f32 = 0.06;
/// Radians per second.
const ORBIT_RATE: f64 = 0.15;
const USAGE: &str = "usage: tree-viewer <file.glb | file.gltf | archive.orr> [--model NAME] \
    [--node PATTERN] [--part-attribute NAME] [--still] [--frames N] [--headless] \
    [--out file.png]";

fn main() {
    let options = Options::from_args();
    if let Some(c) = run(&options) {
        let path = options.out_path("tree-viewer");
        c.save_png(&path);
        log::info!("wrote {}", path.display());
    }
    assert_clean();
}

/// The example's own arguments.
struct Args {
    input: PathBuf,
    model: Option<String>,
    node: Option<String>,
    part_attribute: Option<String>,
}

impl Args {
    fn parse(rest: &[String]) -> Self {
        let (mut input, mut model, mut node, mut part_attribute) = (None, None, None, None);
        let mut it = rest.iter();
        while let Some(a) = it.next() {
            let mut value = || {
                it.next()
                    .cloned()
                    .unwrap_or_else(|| panic!("{a} takes a value"))
            };
            match a.as_str() {
                "--model" => model = Some(value()),
                "--node" => node = Some(value()),
                "--part-attribute" => part_attribute = Some(value()),
                "--still" => {}
                _ if input.is_none() && !a.starts_with("--") => input = Some(PathBuf::from(a)),
                _ => panic!("unknown argument {a}; {USAGE}"),
            }
        }
        Args {
            input: input.expect(USAGE),
            model,
            node,
            part_attribute,
        }
    }

    fn is_gltf(&self) -> bool {
        self.input
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("glb") || e.eq_ignore_ascii_case("gltf"))
    }
}

/// Cooks the glTF file into an archive under the OS temp directory, named after the file and
/// the settings; the cook leaves an archive whose inputs did not change as it is.
fn cook_tree(args: &Args) -> PathBuf {
    let source = std::fs::canonicalize(&args.input)
        .unwrap_or_else(|e| panic!("{}: {e}", args.input.display()));
    let settings = GltfSettings {
        models: args
            .node
            .iter()
            .map(|n| NodeSelect { node: n.clone() })
            .collect(),
        parts_from_attribute: args.part_attribute.as_ref().map(|name| PartAttribute {
            name: name.clone(),
            parts: MAX_JOINTS as u16,
            ..PartAttribute::default()
        }),
        parts_from_skin: args.part_attribute.is_none(),
        ..GltfSettings::default()
    };
    let mut h = DefaultHasher::new();
    (&source, &args.node, &args.part_attribute).hash(&mut h);
    let dir = std::env::temp_dir().join("orr-tree-viewer");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let stem = source
        .file_stem()
        .map_or("tree".into(), |s| s.to_string_lossy());
    let out = dir.join(format!("{stem}-{:016x}.orr", h.finish()));
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::File(out.clone()),
            vec![AssetSpec {
                name: TREE.into(),
                source: Source::Gltf {
                    path: source,
                    settings,
                },
            }],
        )],
    };
    let started = Instant::now();
    orr_cook::cook(&manifest, &CookOptions::default()).expect("cook");
    log::info!(
        "cooked {} in {:.1} s",
        out.display(),
        started.elapsed().as_secs_f64()
    );
    out
}

/// Blob 0 of entry `id`, which must exist.
fn head(archive: &Archive, id: AssetId) -> Vec<u8> {
    entry_head(archive, id).unwrap_or_else(|| panic!("no readable entry {id:?}"))
}

/// The tree model's name: `wanted`, or the archive's first model by name.
fn model_name(archive: &Archive, wanted: Option<&str>) -> String {
    let mut names: Vec<&str> = archive
        .entries()
        .iter()
        .filter(|e| e.class == Class::Model as u16)
        .map(|e| archive.name(e))
        .collect();
    names.sort_unstable();
    match wanted {
        Some(w) => {
            assert!(
                names.contains(&w),
                "no model {w}; the archive holds {names:?}"
            );
            w.to_string()
        }
        None => names
            .first()
            .expect("the archive holds no model")
            .to_string(),
    }
}

/// A joint's wind hints from its glTF node's extras, each 0..1.
#[derive(Clone, Copy, Default)]
struct Hint {
    stiffness: Option<f32>,
    flutter: Option<f32>,
}

/// Wind hints by node name from a glTF file's JSON (a `.glb`'s first chunk).
fn wind_hints(path: &Path) -> HashMap<String, Hint> {
    let fail = |e: &dyn std::fmt::Display| -> ! { panic!("{}: {e}", path.display()) };
    let mut file = File::open(path).unwrap_or_else(|e| fail(&e));
    let mut header = [0u8; 20];
    let json = match file.read_exact(&mut header) {
        Ok(()) if header.starts_with(b"glTF") && header[16..20] == *b"JSON" => {
            let len = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
            let mut json = vec![0; len as usize];
            file.read_exact(&mut json).unwrap_or_else(|e| fail(&e));
            json
        }
        _ => std::fs::read(path).unwrap_or_else(|e| fail(&e)),
    };
    let doc: serde_json::Value = serde_json::from_slice(&json).unwrap_or_else(|e| fail(&e));
    let value = |w: &serde_json::Value, key| w.get(key)?.as_f64().map(|v| v as f32);
    doc["nodes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|n| {
            let w = n.get("extras")?.get("wind")?;
            let hint = Hint {
                stiffness: value(w, "stiffness"),
                flutter: value(w, "flutter"),
            };
            Some((n.get("name")?.as_str()?.to_string(), hint))
        })
        .collect()
}

/// The model-space origin of a joint's bind space, from its inverse bind.
fn bind_origin(inverse_bind: &orr::orr_pack::Affine3) -> Vec3 {
    let [r0, r1, r2] = inverse_bind.rows.map(Vec4::from);
    Mat4::from_cols(r0, r1, r2, Vec4::W)
        .transpose()
        .inverse()
        .w_axis
        .truncate()
}

/// The tree's joints for the pose kernel and the joint each part follows. With a skeleton,
/// its joints pivot at their bind positions and hang on their parents; without one, each
/// part is a joint on the body. A joint bends by its flexibility (1 - stiffness, or its
/// depth over the deepest joint's without a hint), scaled so the most flexible chain bends
/// `TIP_BEND` in all; joints with a flutter hint flutter by it, others only at the tips.
/// The body's joint stays still, since the renderer does not pose part 0.
fn rig(
    archive: &Archive,
    model: &ModelInfo,
    hints: &HashMap<String, Hint>,
) -> (Vec<tree_wind::Joint>, Vec<u32>) {
    let mesh = MeshInfo::parse(&head(archive, model.header.mesh)).expect("mesh");
    let (pivots, parents, joint_hints, part_joints): (
        Vec<Vec3>,
        Vec<Option<usize>>,
        Vec<Hint>,
        Vec<u32>,
    ) = if model.header.skeleton == AssetId::NONE {
        let n = mesh.parts.len();
        (
            mesh.parts.iter().map(|p| Vec3::from(p.pivot)).collect(),
            (0..n).map(|i| (i > 0).then_some(0)).collect(),
            vec![Hint::default(); n],
            (0..n as u32).collect(),
        )
    } else {
        let s = SkeletonInfo::parse(&head(archive, model.header.skeleton)).expect("skeleton");
        (
            s.inverse_binds.iter().map(bind_origin).collect(),
            s.parents
                .iter()
                .map(|&p| (p != NO_PARENT).then_some(usize::from(p)))
                .collect(),
            s.names
                .iter()
                .map(|n| hints.get(n).copied().unwrap_or_default())
                .collect(),
            mesh.parts.iter().map(|p| u32::from(p.joint)).collect(),
        )
    };
    let n = pivots.len();
    assert!(
        n <= MAX_JOINTS,
        "{n} joints; the pose kernel takes {MAX_JOINTS}"
    );
    let mut depth = vec![0u32; n];
    let mut leaf = vec![true; n];
    for j in 0..n {
        if let Some(p) = parents[j] {
            depth[j] = depth[p] + 1;
            leaf[p] = false;
        }
    }
    let body = part_joints[0] as usize;
    let deepest = depth.iter().copied().max().unwrap_or(0).max(1) as f32;
    let flexibility: Vec<f32> = (0..n)
        .map(|j| match joint_hints[j].stiffness {
            _ if j == body => 0.0,
            Some(s) => (1.0 - s).clamp(0.0, 1.0),
            None => depth[j] as f32 / deepest,
        })
        .collect();
    let mut chain = vec![0.0f32; n];
    for j in 0..n {
        chain[j] = flexibility[j] + parents[j].map_or(0.0, |p| chain[p]);
    }
    let longest = chain.iter().copied().fold(0.0, f32::max).max(1e-6);
    let most = joint_hints
        .iter()
        .filter_map(|h| h.flutter)
        .fold(0.0, f32::max);
    let flutter = |j: usize| match joint_hints[j].flutter {
        _ if j == body => 0.0,
        Some(f) if most > 0.0 => f / most,
        Some(_) => 0.0,
        None if leaf[j] => 1.0,
        None => 0.0,
    };
    let joints = (0..n)
        .map(|j| tree_wind::Joint {
            pivot: pivots[j].to_array(),
            parent: parents[j].map_or(u32::MAX, |p| p as u32),
            bend: TIP_BEND * flexibility[j] / longest,
            flutter: FLUTTER * flutter(j),
            phase: (j as u32).wrapping_mul(2_654_435_761) as f32 / u32::MAX as f32
                * std::f32::consts::TAU,
            _pad: 0,
        })
        .collect();
    (joints, part_joints)
}

/// The ground, `half` m to each side, and the sky, cooked into an in-memory pack.
fn cook_surroundings(half: f32) -> Arc<[u8]> {
    let soil = Material {
        base_color: [0.32, 0.28, 0.22],
        specular_roughness: 0.9,
        ..Material::default()
    };
    let mut floor = MeshBuilder::default();
    floor.quad(
        0,
        [
            Vec3::new(-half, 0.0, half),
            Vec3::new(half, 0.0, half),
            Vec3::new(half, 0.0, -half),
            Vec3::new(-half, 0.0, -half),
        ],
        16,
    );
    let sky = Sky::new("tree-viewer", SUN);
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                sky.asset(SKY),
                AssetSpec {
                    name: GROUND.into(),
                    source: Source::Mesh {
                        mesh: floor.finish(),
                        materials: vec![soil],
                        settings: MeshSettings::default(),
                    },
                },
            ],
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .expect("cook")
        .memory
        .remove(0)
}

/// Runs the example; headless runs and runs with `--out` return their last frame.
fn run(options: &Options) -> Option<Capture> {
    let gpu = device();
    let args = Args::parse(&options.rest);
    let (path, hints) = if args.is_gltf() {
        (cook_tree(&args), wind_hints(&args.input))
    } else {
        (args.input.clone(), HashMap::new())
    };
    let archive = Archive::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let name = model_name(&archive, args.model.as_deref());
    let info = ModelInfo::parse(&head(&archive, AssetId::from_name(&name))).expect("model");
    let (joints, part_joints) = rig(&archive, &info, &hints);
    let parts = part_joints.len() as u32;
    let (lo, hi) = (
        Vec3::from(info.header.bounds.min),
        Vec3::from(info.header.bounds.max),
    );
    let center = (lo + hi) * 0.5;
    let radius = (hi - lo).length() * 0.5;
    log::info!(
        "{name}: {parts} parts, {} joints, bounds {lo} to {hi}",
        joints.len()
    );

    let mut window = (!options.headless).then(|| Window::new("orr tree-viewer", 1280, 720));
    let desc = RendererDesc::default();
    let mut display = window.as_ref().map(|w| {
        Display::new(
            gpu,
            w.window_handle(),
            w.display_handle(),
            &DisplayRequest::default(),
            desc.frames_in_flight,
        )
        .expect("display")
    });
    let mut renderer = Renderer::new(gpu, desc).expect("renderer");
    enable_dlss(&mut renderer);
    let _workers = Workers::new(renderer.loading(), 2 + CompilingKernels::JOBS);
    let surroundings = cook_surroundings((radius * 8.0).max(40.0));
    let (mounts, mut frame) = mount_all_at(
        &mut renderer,
        vec![PackSource::File(path), PackSource::Memory(surroundings)],
        FIRST_FRAME,
    );
    let tree = renderer
        .model(mounts[0], AssetId::from_name(&name))
        .unwrap_or_else(|e| panic!("model {name}: {e}"));
    let ground = renderer
        .model(mounts[1], AssetId::from_name(GROUND))
        .expect("ground model");
    let environment = sky::environment(&renderer, gpu, mounts[1], SKY);

    let scene = renderer
        .create_scene(
            &SceneDesc {
                instance_capacity: 4,
                group_capacity: 4,
                light_capacity: 1,
                material_set_capacity: 16,
                emissive_capacity: 16,
                part_capacity: parts.next_power_of_two(),
                ptlas_partitions: 16,
                global_instances: 16,
                ..SceneDesc::default()
            },
            &SceneLightingSettings::default(),
        )
        .expect("scene");
    let history = renderer.create_view_history().expect("view history");
    options.wait_for_paths(&mut renderer);
    frame += 1;
    let listed = [scene];
    let f = renderer
        .begin_frame(FrameDesc {
            time: fixed_time(frame),
            scenes: &listed,
            display: None,
        })
        .expect("begin frame");
    let mut w = f.scene(scene).expect("scene writer");
    let group = w
        .create_transform_group(&TransformGroupDesc {
            count: 2,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .expect("transform group");
    w.transforms(group, 0, 2)
        .expect("transforms")
        .fill(Affine3::IDENTITY);
    let poses = w
        .create_transform_group(&TransformGroupDesc {
            count: parts,
            writer: GroupWriter::Gpu,
            anchor: DVec3::ZERO,
        })
        .expect("pose group");
    for (model, first, pose) in [
        (
            tree,
            0,
            Some(Pose {
                group: poses,
                index: 0,
                variants: 1,
                stride: 0,
            }),
        ),
        (ground, 1, None),
    ] {
        w.add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: group,
            first,
            pose,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1.0e4,
            animation_range: if pose.is_some() { 1.0e4 } else { 0.0 },
            user_tag: 0,
        })
        .expect("instances");
    }
    w.set_light(LightId(0), Some(&sun())).expect("sun");
    w.set_environment(environment.gpu().address());
    drop(w);
    f.end().expect("end frame");

    let wind = renderer
        .compute_pipeline(&tree_wind::MODULE, "main", SpecData::default())
        .expect("wind pipeline");
    let mut joint_table = gpu
        .gpu_malloc::<tree_wind::Joint>(MemoryKind::Upload, joints.len() as u64, "tree joints")
        .expect("joints");
    joint_table
        .as_mut_slice()
        .expect("joints")
        .copy_from_slice(&joints);
    let mut part_table = gpu
        .gpu_malloc::<u32>(MemoryKind::Upload, u64::from(parts), "tree part joints")
        .expect("part joints");
    part_table
        .as_mut_slice()
        .expect("part joints")
        .copy_from_slice(&part_joints);

    let headless_target = options.headless.then(|| {
        let [w, h] = HEADLESS_EXTENT;
        gpu.create_texture(
            &TextureDesc::d2(
                Format::Rgba8Unorm,
                w,
                h,
                TextureUsage::STORAGE | TextureUsage::SAMPLED,
            ),
            TextureMemory::Dedicated,
            "tree-viewer target",
        )
        .expect("headless target")
    });
    let capture = options.headless || options.out.is_some();
    let frames = options
        .frames
        .or(options.headless.then_some(HEADLESS_FRAMES));
    let distance = radius / (0.5 * FOV_Y).tan() * 0.9;
    let started = Instant::now();
    let mut previous = 0.0;
    let mut last: Option<(u64, TimelinePoint)> = None;
    for i in 0.. {
        if frames.is_some_and(|n| i >= n) {
            break;
        }
        if let Some(w) = window.as_mut()
            && !w.pump_showing_paths(&renderer)
        {
            break;
        }
        let final_frame = frames.is_some_and(|n| i + 1 == n);
        // Headless runs step a fixed clock, so they repeat exactly.
        let t = if options.headless {
            fixed_time(i).absolute
        } else {
            started.elapsed().as_secs_f64()
        };
        let time = FrameTime {
            absolute: fixed_time(frame + 1).absolute + t,
            delta: (t - previous) as f32,
        };
        previous = t;
        let angle = 0.6
            + if options.has("--still") {
                0.0
            } else {
                t * ORBIT_RATE
            };
        let eye = center
            + Vec3::new(
                angle.sin() as f32 * distance * 0.97,
                distance * 0.25,
                angle.cos() as f32 * distance * 0.97,
            );

        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &listed,
                display: display.as_mut(),
            })
            .expect("begin frame");
        let id = f.id();
        let target = match (&headless_target, f.display()) {
            (Some(t), _) => ViewTarget::Texture(t),
            (None, Some(_)) => ViewTarget::Display,
            (None, None) => {
                last = Some((id, f.end().expect("end frame")));
                continue;
            }
        };
        f.scene(scene)
            .expect("scene writer")
            .set_focus(eye.as_dvec3());
        let view = LitView {
            scene,
            history,
            position: eye.as_dvec3(),
            rotation: view::look_at_quat(eye, center, Vec3::Y).inverse(),
            projection: Projection {
                fov_y: FOV_Y,
                near: 0.05,
            },
            lens: Lens::default(),
            reset_history: i == 0,
            texture_filter: TextureFilter::Hardware,
            target,
        };
        let output = ViewOutput {
            screenshot: (capture && final_frame)
                .then_some(ScreenshotRequest::Display { include_ui: false }),
            ..ViewOutput::default()
        };
        f.prepare_view(&view, &ViewLighting::default(), &output)
            .expect("prepare view");
        f.update_scenes(&mut |c| {
            let root = c
                .put_root(&tree_wind::Root {
                    scene: c.root,
                    joints: joint_table.gpu(),
                    part_joints: part_table.gpu(),
                    group: poses.0,
                    joint_count: joints.len() as u32,
                    time: t as f32,
                    wind_x: WIND.x,
                    wind_z: WIND.y,
                    _pad: 0,
                })
                .expect("wind root");
            c.cmd.dispatch(c.pipeline(wind), root, [1, 1, 1]);
        })
        .expect("update scenes");
        f.render_view(
            &view,
            &ViewLighting::default(),
            &output,
            &mut ViewHooks::default(),
        )
        .expect("render view");
        last = Some((id, f.end().expect("end frame")));
    }

    if let Some(d) = display.as_ref() {
        d.wait_idle();
    }
    if let Some((_, point)) = last {
        gpu.queue(QueueKind::Render)
            .expect("render queue")
            .wait(point, Duration::from_secs(10))
            .expect("wait");
    }
    let captured = last
        .filter(|_| capture)
        .and_then(|(id, _)| renderer.screenshots(history, id).pop())
        .map(|shot| Capture::from_screenshot(&shot));
    drop(headless_target);
    drop(environment);
    drop(display);
    drop(renderer);
    drop(window);
    gpu.collect_garbage();
    captured
}
