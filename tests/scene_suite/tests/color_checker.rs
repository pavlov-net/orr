//! The ColorChecker through cook, shading and capture, against the colors of BabelColor's
//! average reflectance spectra of the chart under CIE D65, which orr_color's matrices and
//! curves play no part in.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use orr::{
    DirectionalLight, DiscProfile, Exposure, FrameDesc, FrameTime, GroupWriter, InstanceFlags,
    InstancesDesc, Lens, Light, LitView, LoadState, PackSource, Projection, Renderer, RendererDesc,
    SceneDesc, ScreenshotRequest, TextureFilter, TransformGroupDesc, ViewHooks, ViewLighting,
    ViewOutput, ViewTarget,
};
use orr_color::{
    CIE_1931_CMF, CIE_1931_CMF_FIRST_NM, CIE_1931_CMF_STEP_NM, Chromaticity, Primaries,
    RgbPrimaries, srgb8_encode,
};
use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshInput, MeshSettings, Source};
use orr_math::{Affine3, DVec3, Vec2, Vec3};
use orr_pack::material::{Material, slot};
use orr_pack::{AssetId, Span, Target};
use orr_scene::testing::{assert_clean, device, wait};

const NAMES: [&str; 24] = [
    "dark skin",
    "light skin",
    "blue sky",
    "foliage",
    "blue flower",
    "bluish green",
    "orange",
    "purplish blue",
    "moderate red",
    "purple",
    "yellow green",
    "orange yellow",
    "blue",
    "green",
    "red",
    "yellow",
    "magenta",
    "cyan",
    "white 9.5",
    "neutral 8",
    "neutral 6.5",
    "neutral 5",
    "neutral 3.5",
    "black 2",
];

const COLUMNS: usize = 6;
const ROWS: usize = 4;
/// Texels on a patch's side: whole BC7 blocks, and a texture small enough to sit in its mip
/// tail, so it is resident from mount.
const PATCH: usize = 32;
const CHART_M: [f32; 2] = [1.2, 0.8];
/// The charts' left edges, in m.
const LEFT: [f32; 2] = [-1.3, 0.1];
const EXTENT: [u32; 2] = [576, 192];
const FOV_Y: f32 = 0.5;
/// Lux on the charts, so that a perfect diffuse white reads 100 cd/m2.
const ILLUMINANCE: f64 = 100.0 * std::f64::consts::PI;
/// Largest CIE76 Delta E*ab of a patch from its reference. The pipeline measures at most 1.0,
/// from the 8-bit codes, BC7 and the G-buffer's 9-bit base color.
const MAX_DELTA_E: f64 = 1.5;

/// A file `tools/fetch-slang.sh` puts at `rel` under `ORR_TOOLS`, or under the nearest
/// `.tools` that holds it.
fn fetched(rel: &str) -> Option<String> {
    let path = match std::env::var_os("ORR_TOOLS").filter(|t| !t.is_empty()) {
        Some(t) => PathBuf::from(t).join(rel),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .map(|d| d.join(".tools").join(rel))
            .find(|p| p.exists())?,
    };
    std::fs::read_to_string(path).ok()
}

/// The CIE XYZ of each patch, row-major from dark skin, and of the perfect diffuse white, by
/// the CIE 1931 2-degree observer under D65 at BabelColor's 10 nm samples, white at Y = 1.
fn reference(cgats: &str, d65: &str) -> ([DVec3; 24], DVec3) {
    let d65 = |nm: u32| {
        d65.lines()
            .find_map(|l| {
                let (n, v) = l.split_once(',')?;
                (n.trim().parse() == Ok(nm)).then(|| v.trim().parse::<f64>().unwrap())
            })
            .unwrap_or_else(|| panic!("D65 has no {nm} nm"))
    };
    let format = &cgats[cgats.find("BEGIN_DATA_FORMAT").unwrap()..];
    let fields: Vec<&str> = format.lines().nth(1).unwrap().split_whitespace().collect();
    let first: u32 = fields[2]
        .strip_prefix("SPECTRAL_NM_")
        .unwrap()
        .parse()
        .unwrap();
    let observer = |nm: u32| {
        let row = (f64::from(nm) - CIE_1931_CMF_FIRST_NM) / CIE_1931_CMF_STEP_NM;
        DVec3::from(CIE_1931_CMF[row as usize]) * d65(nm)
    };
    let weights: Vec<DVec3> = (0..fields.len() - 2)
        .map(|i| observer(first + 10 * i as u32))
        .collect();
    let white: DVec3 = weights.iter().sum();
    let scale = 1.0 / white.y;
    let rows = cgats
        .lines()
        .skip_while(|l| l.trim() != "BEGIN_DATA")
        .skip(1)
        .take_while(|l| l.trim() != "END_DATA");
    let mut patches = [None; 24];
    for line in rows {
        let mut f = line.split_whitespace();
        let name = f.nth(1).unwrap().as_bytes();
        let (column, row) = (usize::from(name[0] - b'A'), usize::from(name[1] - b'1'));
        let xyz: DVec3 = f
            .zip(&weights)
            .map(|(r, &w)| r.parse::<f64>().unwrap() * w)
            .sum();
        patches[row * COLUMNS + column] = Some(xyz * scale);
    }
    (patches.map(Option::unwrap), white * scale)
}

fn lab(xyz: DVec3, white: DVec3) -> DVec3 {
    let f = |t: f64| {
        const D: f64 = 6.0 / 29.0;
        if t > D * D * D {
            t.cbrt()
        } else {
            t / (3.0 * D * D) + 4.0 / 29.0
        }
    };
    let [x, y, z] = (xyz / white).to_array().map(f);
    DVec3::new(116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z))
}

/// An 8-bit chart of `linear` per patch, sRGB-encoded, with `chunks` before its texels.
fn chart_png(path: &Path, linear: &[DVec3; 24], chunks: &[(&[u8; 4], &[u8])]) {
    let [w, h] = [COLUMNS * PATCH, ROWS * PATCH];
    let texels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let patch = (i / w / PATCH) * COLUMNS + i % w / PATCH;
            linear[patch].to_array().map(|c| srgb8_encode(c as f32))
        })
        .collect();
    let mut enc = png::Encoder::new(std::fs::File::create(path).unwrap(), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().unwrap();
    for (name, data) in chunks {
        writer
            .write_chunk(png::chunk::ChunkType(**name), data)
            .unwrap();
    }
    writer.write_image_data(&texels).unwrap();
}

/// The two charts side by side facing +z, each with its texture as base color on a diffuse
/// surface with no specular lobe.
fn pack(textures: [PathBuf; 2]) -> Arc<[u8]> {
    let mut mesh = MeshInput {
        uvs: Some(Vec::new()),
        ..MeshInput::default()
    };
    let [w, h] = CHART_M;
    for left in LEFT {
        let first = mesh.positions.len() as u32;
        let (x0, x1, y0, y1) = (left, left + w, -0.5 * h, 0.5 * h);
        mesh.positions.extend([
            Vec3::new(x0, y1, 0.0),
            Vec3::new(x1, y1, 0.0),
            Vec3::new(x1, y0, 0.0),
            Vec3::new(x0, y0, 0.0),
        ]);
        mesh.normals.extend([Vec3::Z; 4]);
        let uvs = mesh.uvs.as_mut().unwrap();
        uvs.extend([Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y]);
        mesh.indices.extend([0, 2, 1, 0, 3, 2].map(|i| first + i));
        mesh.sections.push(Span {
            first: mesh.indices.len() as u32 - 6,
            count: 6,
        });
    }
    let names = ["tex/srgb", "tex/p3"];
    let materials = names.map(|name| {
        let mut m = Material {
            base_color: [1.0; 3],
            specular_weight: 0.0,
            diffuse_roughness: 0.0,
            ..Material::default()
        };
        m.textures[slot::BASE_COLOR].texture = AssetId::from_name(name);
        m
    });
    let mut assets: Vec<AssetSpec> = names
        .into_iter()
        .zip(textures)
        .map(|(name, path)| AssetSpec {
            name: name.into(),
            source: Source::Texture {
                path,
                settings: Default::default(),
            },
        })
        .collect();
    assets.push(AssetSpec {
        name: "charts".into(),
        source: Source::Mesh {
            mesh,
            materials: materials.to_vec(),
            settings: MeshSettings::default(),
        },
    });
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, assets)],
    };
    orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0)
}

fn time(frame: u32) -> FrameTime {
    FrameTime {
        absolute: f64::from(frame) / 60.0,
        delta: 1.0 / 60.0,
    }
}

/// The charts lit head-on, seen head-on: the scene-referred capture of the third frame, in
/// linear Rec.2020 cd/m2.
fn render(pack: &Arc<[u8]>) -> orr::Screenshot {
    let mut r = Renderer::new(device(), RendererDesc::default()).unwrap();
    let loading = r.loading();
    let worker = {
        let l = Arc::clone(&loading);
        std::thread::spawn(move || l.run_worker())
    };
    let mut frame = 0;
    let mount = r.mount(PackSource::Memory(Arc::clone(pack)));
    let start = Instant::now();
    while matches!(r.mount_state(mount), LoadState::Pending) {
        frame += 1;
        let f = r
            .begin_frame(FrameDesc {
                time: time(frame),
                scenes: &[],
                display: None,
            })
            .unwrap();
        wait(f.end().unwrap());
        assert!(start.elapsed() < Duration::from_secs(60), "mount pending");
    }
    if let LoadState::Failed(e) = r.mount_state(mount) {
        panic!("mount failed: {e}");
    }
    let model = r.model(mount, AssetId::from_name("charts")).unwrap();
    let scene = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    let history = r.create_view_history().unwrap();
    r.wait_paths().unwrap();
    let eye = Vec3::new(0.0, 0.0, distance());
    let output = ViewOutput {
        exposure: Exposure::Manual { ev100: 10.0 },
        ..ViewOutput::default()
    };
    let listed = [scene];
    let mut shot = None;
    for i in 0..3 {
        frame += 1;
        let mut f = r
            .begin_frame(FrameDesc {
                time: time(frame),
                scenes: &listed,
                display: None,
            })
            .unwrap();
        let id = f.id();
        if i == 0 {
            let mut s = f.scene(scene).unwrap();
            let group = s
                .create_transform_group(&TransformGroupDesc {
                    count: 1,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::ZERO,
                })
                .unwrap();
            s.transforms(group, 0, 1).unwrap()[0] = Affine3::IDENTITY.into();
            s.add_instances(&InstancesDesc {
                model,
                copies: 1,
                transforms: group,
                first: 0,
                pose: None,
                material_set: None,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                rt_range: 1000.0,
                animation_range: 0.0,
                user_tag: 0,
            })
            .unwrap();
            s.add_light(&Light::Directional(DirectionalLight {
                direction_to_light: Vec3::Z,
                illuminance: Vec3::splat(ILLUMINANCE as f32),
                angular_radius: 4.65e-3,
                disc: DiscProfile::Uniform,
            }))
            .unwrap();
            s.set_focus(DVec3::ZERO);
        }
        f.update_scenes(&mut |_| {}).unwrap();
        let view = LitView {
            scene,
            history,
            position: eye.as_dvec3(),
            rotation: orr_math::camera::rh::view::look_at_quat(eye, Vec3::ZERO, Vec3::Y).inverse(),
            projection: Projection {
                fov_y: FOV_Y,
                near: 0.05,
            },
            lens: Lens::default(),
            camera_cut: false,
            not_rendering_scene: false,
            screenshot: (i == 2).then_some(ScreenshotRequest::SceneReferred { rec709: false }),
            texture_filter: TextureFilter::Hardware,
            target: ViewTarget::Radiance { extent: EXTENT },
        };
        f.render_view(
            &view,
            &ViewLighting::default(),
            &output,
            &mut ViewHooks::default(),
        )
        .unwrap();
        wait(f.end().unwrap());
        if i == 2 {
            shot = r.screenshots(history, id).pop();
        }
    }
    loading.stop_workers();
    worker.join().unwrap();
    shot.expect("a scene-referred capture")
}

/// The eye's distance from the charts' plane: the view's height spans 1 m there.
fn distance() -> f32 {
    0.5 / (0.5 * FOV_Y).tan()
}

/// The mean linear Rec.2020 of the middle third of `patch` on chart `chart`.
fn patch_mean(shot: &orr::Screenshot, chart: usize, patch: usize) -> DVec3 {
    let (column, row) = (patch % COLUMNS, patch / COLUMNS);
    let side = CHART_M[0] / COLUMNS as f32;
    let center = Vec2::new(
        LEFT[chart] + side * (column as f32 + 0.5),
        0.5 * CHART_M[1] - side * (row as f32 + 0.5),
    );
    let pixels_per_m = EXTENT[1] as f32;
    let at = Vec2::new(
        0.5 * EXTENT[0] as f32 + center.x * pixels_per_m,
        0.5 * EXTENT[1] as f32 - center.y * pixels_per_m,
    );
    let half = (side * pixels_per_m / 6.0) as i32;
    let mut sum = DVec3::ZERO;
    let mut n = 0.0;
    for y in -half..half {
        for x in -half..half {
            let (px, py) = (at.x as i32 + x, at.y as i32 + y);
            let i = 4 * (py as usize * EXTENT[0] as usize + px as usize);
            let p = &shot.rgba[i..i + 3];
            sum += DVec3::new(p[0].into(), p[1].into(), p[2].into());
            n += 1.0;
        }
    }
    sum / n
}

/// Every patch of an 8-bit sRGB chart, and every patch of an 8-bit Display P3 chart that cICP
/// tags, reads back within [`MAX_DELTA_E`] of BabelColor's average chart under D65. The sRGB
/// chart leaves out the patch that Rec.709 can't hold, cyan, which clips there.
#[test]
#[allow(clippy::print_stderr)]
fn color_checker_patches_match_babelcolor_under_d65() {
    let (Some(cgats), Some(d65)) = (
        fetched("babelcolor/CC_Avg30_spectrum_CGATS.txt"),
        fetched("cie/CIE_std_illum_D65.csv"),
    ) else {
        eprintln!("skipped: run ORR_VALIDATION_DATA=1 tools/fetch-slang.sh");
        return;
    };
    let (patches, white) = reference(&cgats, &d65);
    // The charts' encodings and the capture share D65 as (1, 1, 1); scaling to it keeps each
    // patch's CIELAB, which is taken against the reference's own white.
    let d65_white = Chromaticity::D65.to_xyz(1.0);
    let encoded = |p: RgbPrimaries| patches.map(|xyz| p.xyz_to_rgb() * (xyz / white * d65_white));
    let rec709 = encoded(RgbPrimaries::REC709);
    let p3 = encoded(RgbPrimaries::DISPLAY_P3);
    assert!(p3.iter().all(|c| c.min_element() >= 0.0), "{p3:?}");

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("color_checker");
    std::fs::create_dir_all(&dir).unwrap();
    let textures = [dir.join("srgb.png"), dir.join("p3.png")];
    chart_png(&textures[0], &rec709, &[]);
    // H.273 primaries 12 (Display P3) and transfer 13 (sRGB), full range.
    chart_png(&textures[1], &p3, &[(b"cICP", &[12, 13, 0, 1])]);
    let shot = render(&pack(textures));
    assert!(shot.scene_referred);
    assert_eq!(shot.primaries, Primaries::Rec2020);
    assert_eq!(shot.size, EXTENT);

    let to_xyz = RgbPrimaries::REC2020.rgb_to_xyz();
    let white_nits = ILLUMINANCE / std::f64::consts::PI;
    let mut failed = Vec::new();
    for (chart, (name, encoded)) in [("sRGB", rec709), ("Display P3", p3)].iter().enumerate() {
        for patch in (0..24).filter(|&p| encoded[p].min_element() >= 0.0) {
            let want = lab(patches[patch], white);
            let xyz = to_xyz * patch_mean(&shot, chart, patch) / white_nits;
            let got = lab(xyz, d65_white);
            let delta = got.distance(want);
            if delta > MAX_DELTA_E {
                failed.push(format!(
                    "{name} chart, {}: Lab {got:.2} against {want:.2}, Delta E {delta:.2}",
                    NAMES[patch]
                ));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    assert_clean();
}
