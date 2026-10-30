//! Mesh pipelines: a G-buffer pass against a CPU rasteriser, blending and scissors.

use gpu_suite::shaders::{blend, gbuffer};
use gpu_suite::{assert_clean, device, readback, render, run, upload};
use orr_gpu::{
    Blend, ClearColor, ColorAttachment, ColorLoad, ColorMask, ColorTarget, CompareOp,
    DepthAttachment, DepthLoad, DepthTarget, Format, Hazards, MeshArgs, MeshPipelineDesc,
    Rendering, ShaderStage, SpecData, Stages, Texture, TextureDesc, TextureMemory, TextureRegion,
    TextureUsage,
};

const W: u32 = 256;
const H: u32 = 192;
const SALT: u32 = 0x5a17;

fn target(format: Format, usage: TextureUsage, name: &str) -> Texture {
    let desc = TextureDesc::d2(format, W, H, usage);
    device()
        .create_texture(&desc, TextureMemory::Dedicated, name)
        .unwrap()
}

fn half(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Triangles with distinct depths, every seventh hidden.
fn triangles() -> Vec<gbuffer::Triangle> {
    let n = 150u32;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut order: Vec<u32> = (0..n).collect();
    for i in (1..n as usize).rev() {
        order.swap(i, (next() * (i + 1) as f32) as usize % (i + 1));
    }
    (0..n)
        .map(|i| {
            let z = 0.05 + 0.9 * order[i as usize] as f32 / n as f32;
            let (cx, cy, r) = (next() * 2.2 - 1.1, next() * 2.2 - 1.1, 0.1 + next() * 0.5);
            let mut corner = |t: f32| {
                let a = t * std::f32::consts::TAU + next() * 1.5;
                [cx + r * a.cos(), cy + r * a.sin(), z, 1.0]
            };
            gbuffer::Triangle {
                a: corner(0.0),
                b: corner(1.0 / 3.0),
                c: corner(2.0 / 3.0),
                motion: [i as f32 / 8.0, -(i as f32) / 16.0],
                id: 1000 + i,
                hidden: u32::from(i % 7 == 3),
            }
        })
        .collect()
}

/// The front-most visible triangle at each pixel centre: `Some(None)` for background,
/// `None` where a pixel centre lies within 0.01 px of a visible triangle's edge.
fn reference(tris: &[gbuffer::Triangle]) -> Vec<Option<Option<usize>>> {
    let to_px = |v: [f32; 4]| [(v[0] + 1.0) * 0.5 * W as f32, (v[1] + 1.0) * 0.5 * H as f32];
    let mut out = vec![Some(None); (W * H) as usize];
    for (i, t) in tris.iter().enumerate().filter(|(_, t)| t.hidden == 0) {
        let v = [to_px(t.a), to_px(t.b), to_px(t.c)];
        let area =
            (v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0]);
        for y in 0..H {
            for x in 0..W {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let dist = |e: usize| {
                    let (a, b) = (v[e], v[(e + 1) % 3]);
                    let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
                    cross * area.signum() / (b[0] - a[0]).hypot(b[1] - a[1])
                };
                let nearest = dist(0).min(dist(1)).min(dist(2));
                if nearest < -0.01 {
                    continue;
                }
                let px = &mut out[(y * W + x) as usize];
                if nearest < 0.01 {
                    *px = None;
                } else if let Some(cur) = px
                    && cur.is_none_or(|c: usize| tris[c].a[2] < t.a[2])
                {
                    *cur = Some(i);
                }
            }
        }
    }
    out
}

#[test]
fn gbuffer_matches_cpu_reference() {
    let d = device();
    let tris = triangles();
    let visible = tris.iter().filter(|t| t.hidden == 0).count() as u64;
    let gbuf_a = target(Format::Rgba32Uint, TextureUsage::COLOR, "gbuf_a");
    let gbuf_b = target(Format::Rg32Uint, TextureUsage::COLOR, "gbuf_b");
    let motion = target(Format::Rg16Float, TextureUsage::COLOR, "motion");
    let depth = target(Format::D32Float, TextureUsage::DEPTH, "depth");

    let spec = gbuffer::Spec { salt: SALT }.bytes();
    let stage = |entry| {
        ShaderStage::new(
            &gbuffer::MODULE,
            entry,
            SpecData {
                entries: gbuffer::Spec::ENTRIES,
                bytes: &spec,
            },
        )
        .unwrap()
    };
    let colors = [
        ColorTarget::new(Format::Rgba32Uint),
        ColorTarget::new(Format::Rg32Uint),
        ColorTarget::new(Format::Rg16Float),
    ];
    let p = d
        .create_mesh_pipeline(
            &MeshPipelineDesc {
                task: Some(stage("task_main")),
                pixel: Some(stage("pixel_main")),
                colors: &colors,
                depth: Some(DepthTarget {
                    format: Format::D32Float,
                    compare: CompareOp::GreaterOrEqual,
                    write: true,
                }),
                ..MeshPipelineDesc::new(stage("mesh_main"))
            },
            "gbuffer",
        )
        .unwrap();

    let tri_buf = upload(&tris, "gbuffer triangles");
    let root = upload(
        &[gbuffer::Root {
            triangles: tri_buf.gpu,
            triangle_count: tris.len() as u32,
            pad: 0,
        }],
        "gbuffer root",
    );
    // Two draws of every triangle; the count stops the two one-group draws after them.
    let groups = (tris.len() as u32).div_ceil(32);
    let args = upload(
        &[
            MeshArgs {
                x: groups,
                y: 1,
                z: 1,
            },
            MeshArgs {
                x: groups,
                y: 1,
                z: 1,
            },
            MeshArgs { x: 1, y: 1, z: 1 },
            MeshArgs { x: 1, y: 1, z: 1 },
        ],
        "gbuffer args",
    );
    let count = upload(&[2u32], "gbuffer count");

    let px = u64::from(W * H);
    let out_a = readback::<[u32; 4]>(px, "gbuf_a out");
    let out_b = readback::<[u32; 2]>(px, "gbuf_b out");
    let out_m = readback::<[u16; 2]>(px, "motion out");
    let out_d = readback::<f32>(px, "depth out");

    let mut list = render().command_list().unwrap();
    for t in [&gbuf_a, &gbuf_b, &motion, &depth] {
        list.discard(t);
    }
    list.begin_pass("gbuffer", true);
    list.begin_rendering(&Rendering {
        extent: [W, H],
        colors: &[
            ColorAttachment {
                texture: &gbuf_a,
                load: ColorLoad::Clear(ClearColor::Uint([0; 4])),
                store: true,
            },
            ColorAttachment {
                texture: &gbuf_b,
                load: ColorLoad::Clear(ClearColor::Uint([7, 7, 0, 0])),
                store: true,
            },
            ColorAttachment {
                texture: &motion,
                load: ColorLoad::Clear(ClearColor::Float([0.5, -0.5, 0.0, 0.0])),
                store: true,
            },
        ],
        depth: Some(DepthAttachment {
            texture: &depth,
            load: DepthLoad::Clear(0.0),
            store: true,
        }),
    });
    list.draw_mesh_tasks_indirect_count(&p, root.gpu, args.gpu, count.gpu, 4);
    list.end_rendering();
    list.end_pass();
    list.barrier(
        Stages::COLOR_OUT | Stages::DEPTH,
        Stages::TRANSFER,
        Hazards::empty(),
    );
    let whole = TextureRegion::whole_mip(gbuf_a.desc(), 0);
    list.copy_from_texture(out_a.gpu.cast(), &gbuf_a, &whole);
    list.copy_from_texture(out_b.gpu.cast(), &gbuf_b, &whole);
    list.copy_from_texture(out_m.gpu.cast(), &motion, &whole);
    list.copy_from_texture(out_d.gpu.cast(), &depth, &whole);
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    let point = run(list);

    let stats = d.pass_results(point).unwrap()[0].stats.unwrap();
    assert_eq!(stats.task, 2 * u64::from(groups) * 32, "task invocations");
    assert_eq!(stats.mesh, 2 * visible * 3, "mesh invocations");

    let expected = reference(&tris);
    let (a, b, m, z) = (
        out_a.as_slice().unwrap(),
        out_b.as_slice().unwrap(),
        out_m.as_slice().unwrap(),
        out_d.as_slice().unwrap(),
    );
    let (mut checked, mut covered, mut wrong) = (0, 0, Vec::new());
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            let Some(front) = expected[i] else { continue };
            checked += 1;
            let ok = match front {
                None => {
                    a[i] == [0; 4]
                        && b[i] == [7, 7]
                        && [half(m[i][0]), half(m[i][1])] == [0.5, -0.5]
                        && z[i] == 0.0
                }
                Some(t) => {
                    covered += 1;
                    let tri = &tris[t];
                    let close = |v: f32| (v - tri.a[2]).abs() <= 1e-6;
                    a[i] == [tri.id, x, y, SALT]
                        && b[i][0] == t as u32
                        && close(f32::from_bits(b[i][1]))
                        && [half(m[i][0]), half(m[i][1])] == tri.motion
                        && close(z[i])
                }
            };
            if !ok && wrong.len() < 8 {
                wrong.push((x, y, front, a[i], b[i], z[i]));
            }
        }
    }
    assert!(
        checked > (W * H) as usize * 9 / 10,
        "only {checked} pixels away from edges"
    );
    assert!(covered > checked / 4, "only {covered} covered pixels");
    assert!(
        wrong.is_empty(),
        "pixels differing from the reference: {wrong:?}"
    );
    assert_clean();
}

#[test]
fn blend_mask_and_scissor() {
    let d = device();
    let tex = target(Format::Rgba16Float, TextureUsage::COLOR, "blend target");
    let pipeline = |blend: Blend, write: ColorMask| {
        let stage = |e| ShaderStage::new(&blend::MODULE, e, SpecData::default()).unwrap();
        d.create_mesh_pipeline(
            &MeshPipelineDesc {
                pixel: Some(stage("pixel_main")),
                colors: &[ColorTarget {
                    format: Format::Rgba16Float,
                    blend: Some(blend),
                    write,
                }],
                ..MeshPipelineDesc::new(stage("mesh_main"))
            },
            "blend",
        )
        .unwrap()
    };
    let over = pipeline(Blend::PREMULTIPLIED, ColorMask::ALL);
    let add = pipeline(Blend::ADDITIVE, ColorMask::R | ColorMask::G | ColorMask::B);
    let over_root = upload(
        &[blend::Root {
            color: [0.5, 0.25, 0.0, 0.5],
        }],
        "over root",
    );
    let add_root = upload(
        &[blend::Root {
            color: [0.125, 0.125, 0.125, 0.5],
        }],
        "add root",
    );
    let out = readback::<[u16; 4]>(u64::from(W * H), "blend out");

    let mut list = render().command_list().unwrap();
    list.discard(&tex);
    list.begin_rendering(&Rendering {
        extent: [W, H],
        colors: &[ColorAttachment {
            texture: &tex,
            load: ColorLoad::Clear(ClearColor::Float([0.25, 0.5, 0.75, 1.0])),
            store: true,
        }],
        depth: None,
    });
    list.draw_mesh_tasks(&over, over_root.gpu, [1, 1, 1]);
    let (offset, extent) = ([8u32, 16u32], [20u32, 1000u32]);
    list.set_scissor(offset, extent);
    list.draw_mesh_tasks(&add, add_root.gpu, [1, 1, 1]);
    list.end_rendering();
    list.barrier(Stages::COLOR_OUT, Stages::TRANSFER, Hazards::empty());
    list.copy_from_texture(
        out.gpu.cast(),
        &tex,
        &TextureRegion::whole_mip(tex.desc(), 0),
    );
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    run(list);

    let px = out.as_slice().unwrap();
    let mut wrong = 0;
    for y in 0..H {
        for x in 0..W {
            let inside = (offset[0]..offset[0] + extent[0]).contains(&x) && y >= offset[1];
            let want = if inside {
                [0.75, 0.625, 0.5, 1.0]
            } else {
                [0.625, 0.5, 0.375, 1.0]
            };
            wrong += usize::from(px[(y * W + x) as usize].map(half) != want);
        }
    }
    assert_eq!(wrong, 0, "pixels differing from the blend reference");
    assert_clean();
}
