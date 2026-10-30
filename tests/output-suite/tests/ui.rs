//! The egui painter, the UI layer and compose-UI: reference images of egui frames presented
//! as UI-only frames, the painter's textures, clips and large meshes against expected
//! pixels, and compose-UI's blend and display screenshots with UI against an f64 model for
//! every encoding.

use std::path::PathBuf;

use egui::epaint::{ImageDelta, Mesh, Vertex};
use egui::{
    ClippedPrimitive, Color32, ColorImage, Context, Event, FullOutput, Modifiers, PointerButton,
    Pos2, RawInput, Rect, TextureId, TextureOptions, TexturesDelta, Ui, pos2, vec2,
};
use half::f16;
use orr_egui::Painter;
use orr_output::{
    CapturePass, ComposePass, ComposeRoot, ComposeSource, DisplayRequest, OutputFormat,
    OutputKernels, OutputView, ResolvedDisplay, UiLayer, pq_decode, pq_encode, srgb_decode,
    srgb_encode,
};
use orr_scene::orr_gpu::{
    Allocation, Format, Hazards, MemoryKind, QueueKind, Stages, StorageImageIndex, Texture,
    TextureDesc, TextureIndex, TextureMemory, TextureRegion, TextureUsage, ViewDesc,
};
use orr_scene::testing::{
    Image, Pixels, Tolerance, assert_clean, check, device, read_texture, wait,
};

const REC709_TO_REC2020: [[f64; 3]; 3] = [
    [0.627_403_9, 0.329_283_0, 0.043_313_1],
    [0.069_097_3, 0.919_540_4, 0.011_362_3],
    [0.016_391_4, 0.088_013_3, 0.895_595_3],
];
const REC2020_TO_REC709: [[f64; 3]; 3] = [
    [1.660_491_0, -0.587_641_1, -0.072_849_9],
    [-0.124_550_5, 1.132_899_9, -0.008_349_4],
    [-0.018_150_8, -0.100_578_9, 1.118_729_7],
];

fn mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    m.map(|r| r[0] * v[0] + r[1] * v[1] + r[2] * v[2])
}

fn reference(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("references")
        .join(name)
}

/// Every encoding at the paper white of a 240-nit SDR white on a 1000-nit monitor.
fn displays() -> [(ResolvedDisplay, Format); 3] {
    let resolve = |encoding| {
        DisplayRequest {
            encoding: Some(encoding),
            paper_white_nits: Some(240.0),
            peak_nits: Some(1000.0),
            ..DisplayRequest::default()
        }
        .resolve(
            &[
                OutputFormat::Sdr,
                OutputFormat::Hdr10Pq,
                OutputFormat::ScRgb,
            ],
            None,
            false,
        )
    };
    [
        (resolve(OutputFormat::Sdr), Format::Bgra8Unorm),
        (resolve(OutputFormat::Hdr10Pq), Format::A2b10g10r10Unorm),
        (resolve(OutputFormat::ScRgb), Format::Rgba16Float),
    ]
}

/// A storage texture and its heap views.
struct Target {
    texture: Texture,
    store: StorageImageIndex,
    sampled: TextureIndex,
}

impl Target {
    fn new(format: Format, extent: [u32; 2], name: &str) -> Self {
        let gpu = device();
        let desc = TextureDesc::d2(
            format,
            extent[0],
            extent[1],
            TextureUsage::STORAGE | TextureUsage::SAMPLED,
        );
        let texture = gpu
            .create_texture(&desc, TextureMemory::Dedicated, name)
            .unwrap();
        let view = ViewDesc::default();
        let store = gpu.write_storage(gpu.alloc_image_slot().unwrap(), &texture, &view);
        let sampled = gpu.write_sampled(gpu.alloc_image_slot().unwrap(), &texture, &view);
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.discard(&texture);
        wait(queue.submit(vec![list], &[]).unwrap());
        Self {
            texture,
            store,
            sampled,
        }
    }

    fn read(&self) -> Vec<u8> {
        read_texture(
            &self.texture,
            &TextureRegion::whole_mip(self.texture.desc(), 0),
        )
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        device().free_image_slot_after_submitted(self.store.0);
        device().free_image_slot_after_submitted(self.sampled.0);
    }
}

/// Signals of an image in `format`, normalised: [0, 1] code values, or floats for scRGB.
fn signals(bytes: &[u8], format: Format) -> Vec<[f64; 4]> {
    match format {
        Format::Bgra8Unorm => bytes
            .chunks_exact(4)
            .map(|t| [t[2], t[1], t[0], t[3]].map(|c| f64::from(c) / 255.0))
            .collect(),
        Format::Rgba8Unorm => bytes
            .chunks_exact(4)
            .map(|t| [t[0], t[1], t[2], t[3]].map(|c| f64::from(c) / 255.0))
            .collect(),
        Format::A2b10g10r10Unorm => bytes
            .chunks_exact(4)
            .map(|t| {
                let v = u32::from_le_bytes(t.try_into().unwrap());
                let c = |shift: u32| f64::from((v >> shift) & 0x3ff) / 1023.0;
                [c(0), c(10), c(20), f64::from(v >> 30) / 3.0]
            })
            .collect(),
        Format::Rgba16Float => bytes
            .chunks_exact(8)
            .map(|t| std::array::from_fn(|c| f16::from_le_bytes([t[2 * c], t[2 * c + 1]]).to_f64()))
            .collect(),
        _ => unreachable!(),
    }
}

/// The f64 model of `decode_display` and `encode_display` (`display.slang`).
fn decode(signal: [f64; 3], d: &ResolvedDisplay) -> [f64; 3] {
    let pw = f64::from(d.paper_white_nits);
    match d.encoding {
        OutputFormat::Sdr => signal.map(srgb_decode),
        OutputFormat::Hdr10Pq => mul(&REC2020_TO_REC709, signal.map(pq_decode)).map(|v| v / pw),
        OutputFormat::ScRgb => signal.map(|v| v * 80.0 / pw),
    }
}

fn encode(linear: [f64; 3], d: &ResolvedDisplay) -> [f64; 3] {
    let pw = f64::from(d.paper_white_nits);
    match d.encoding {
        OutputFormat::Sdr => linear.map(|v| srgb_encode(v.clamp(0.0, 1.0))),
        OutputFormat::Hdr10Pq => mul(&REC709_TO_REC2020, linear).map(|v| pq_encode(v * pw)),
        OutputFormat::ScRgb => linear.map(|v| v * pw / 80.0),
    }
}

/// Whether `got` matches `want` for `d`: every channel within one code value (two RGBA16F
/// steps for scRGB), or within 1e-3 of the texel's brightest channel once decoded to nits.
/// The second clause covers channels near zero in saturated colours, which come out of the
/// Rec.709 / Rec.2020 cancellations in f32.
fn matches(got: [f64; 3], want: [f64; 3], d: &ResolvedDisplay) -> bool {
    let nits = |s: [f64; 3]| decode(s, d).map(|v| v * f64::from(d.paper_white_nits));
    let close = |g: f64, w: f64| match d.encoding {
        OutputFormat::Sdr => (g - w).abs() * 255.0 <= 1.0 + 1e-6,
        OutputFormat::Hdr10Pq => (g - w).abs() * 1023.0 <= 1.0 + 1e-6,
        OutputFormat::ScRgb => (g - w).abs() <= w.abs() / 512.0 + 1e-6,
    };
    let (gn, wn) = (nits(got), nits(want));
    let brightest = wn.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
    (0..3).all(|c| close(got[c], want[c]) || (gn[c] - wn[c]).abs() <= 1e-3 * brightest)
}

/// egui frames painted into a UI layer and presented as UI-only frames: the UI over black.
struct UiFrames {
    painter: Painter,
    kernels: OutputKernels,
    layer: UiLayer,
    roots: Vec<ComposeRoot>,
    frame: u64,
}

impl UiFrames {
    fn new(extent: [u32; 2]) -> Self {
        let gpu = device();
        Self {
            painter: Painter::new(gpu, 2).unwrap(),
            kernels: OutputKernels::new(gpu).unwrap(),
            layer: UiLayer::new(gpu, extent).unwrap(),
            roots: (0..3).map(|_| ComposeRoot::new(gpu).unwrap()).collect(),
            frame: 0,
        }
    }

    /// One frame: clear the layer, apply `delta`, paint each of `paints` and compose the
    /// layer over black into each of `present`'s targets.
    fn frame(
        &mut self,
        delta: TexturesDelta,
        paints: &[&[ClippedPrimitive]],
        pixels_per_point: f32,
        present: &[(&ResolvedDisplay, &Target)],
    ) {
        let gpu = device();
        self.frame += 1;
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        self.layer.clear(&mut list);
        self.painter
            .update_textures(&mut list, self.frame, delta)
            .unwrap();
        for primitives in paints {
            self.painter
                .paint(
                    &mut list,
                    self.frame,
                    self.layer.texture(),
                    primitives,
                    pixels_per_point,
                )
                .unwrap();
            list.barrier(
                Stages::COLOR_OUT,
                Stages::COLOR_OUT | Stages::COMPUTE,
                Hazards::empty(),
            );
        }
        for (&(display, target), root) in present.iter().zip(&mut self.roots) {
            self.kernels.compose(
                &mut list,
                root,
                &ComposePass {
                    source: ComposeSource::Black,
                    ui: Some(self.layer.index()),
                    target: target.store,
                    extent: self.layer.extent(),
                    display,
                },
            );
        }
        wait(queue.submit(vec![list], &[]).unwrap());
    }

    /// Runs `ui` for `frames` egui frames 1/60 s apart, so opening animations finish,
    /// painting each, and presents the last into a target per display. `click` clicks the
    /// primary button at a point on a frame. Returns the presented images.
    fn run(
        &mut self,
        frames: u32,
        click: Option<(u32, Pos2)>,
        mut ui: impl FnMut(&mut Ui),
    ) -> Vec<Vec<[f64; 4]>> {
        let ctx = Context::default();
        let [w, h] = self.layer.extent();
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(w as f32, h as f32))),
            predicted_dt: 1.0 / 60.0,
            ..RawInput::default()
        };
        let displays = displays();
        let targets: Vec<_> = displays
            .iter()
            .map(|&(_, format)| Target::new(format, [w, h], "presented"))
            .collect();
        for i in 0..frames {
            let FullOutput {
                textures_delta,
                shapes,
                pixels_per_point,
                ..
            } = ctx.run_ui(
                RawInput {
                    time: Some(f64::from(i) / 60.0),
                    events: click_events(click, i),
                    ..input.clone()
                },
                &mut ui,
            );
            let primitives = ctx.tessellate(shapes, pixels_per_point);
            let present: Vec<_> = if i + 1 == frames {
                displays.iter().map(|(d, _)| d).zip(&targets).collect()
            } else {
                Vec::new()
            };
            self.frame(textures_delta, &[&primitives], pixels_per_point, &present);
        }
        displays
            .iter()
            .zip(&targets)
            .map(|((_, format), t)| signals(&t.read(), *format))
            .collect()
    }
}

/// A press on frame `at` and a release on the next.
fn click_events(click: Option<(u32, Pos2)>, frame: u32) -> Vec<Event> {
    let Some((at, pos)) = click else {
        return Vec::new();
    };
    let button = |pressed| Event::PointerButton {
        pos,
        button: PointerButton::Primary,
        pressed,
        modifiers: Modifiers::default(),
    };
    match frame.checked_sub(at) {
        Some(0) => vec![Event::PointerMoved(pos), button(true)],
        Some(1) => vec![button(false)],
        _ => Vec::new(),
    }
}

/// The SDR image as 8-bit RGBA for the reference check.
fn sdr_image(extent: [u32; 2], sdr: &[[f64; 4]]) -> Image {
    Image {
        width: extent[0],
        height: extent[1],
        pixels: Pixels::Rgba8(
            sdr.iter()
                .flat_map(|p| p.map(|c| (c * 255.0).round() as u8))
                .collect(),
        ),
    }
}

/// HDR keeps the UI's SDR look: the PQ and scRGB images equal the SDR image re-encoded for
/// their display, at its paper white.
fn assert_same_look(images: &[Vec<[f64; 4]>]) {
    let displays = displays();
    let sdr = &displays[0].0;
    for (i, (d, _)) in displays.iter().enumerate().skip(1) {
        for (p, (got, s)) in images[i].iter().zip(&images[0]).enumerate() {
            let want = encode(decode([s[0], s[1], s[2]], sdr), d);
            let got = [got[0], got[1], got[2]];
            assert!(
                matches(got, want, d),
                "{:?} pixel {p}: {got:?} against the SDR image's {want:?}",
                d.encoding
            );
        }
    }
}

#[test]
fn egui_demo_windows_match_reference() {
    let extent = [1280, 800];
    let mut demo = egui_demo_lib::DemoWindows::default();
    let images = UiFrames::new(extent).run(60, None, |ui| demo.ui(ui));
    check(
        &reference("egui_demo"),
        &sdr_image(extent, &images[0]),
        Tolerance::PerPixel(1.0 / 255.0),
    );
    assert_same_look(&images);
    assert_clean();
}

#[test]
fn egui_color_test_matches_reference() {
    let extent = [640, 2600];
    let mut test = egui_demo_lib::ColorTest::default();
    // Opens the "Color test" section, which compares vertex and texture gradients.
    let open = Some((10, pos2(50.0, 437.0)));
    let images = UiFrames::new(extent).run(60, open, |ui| {
        egui::CentralPanel::default().show(ui, |ui| test.ui(ui));
    });
    check(
        &reference("egui_color_test"),
        &sdr_image(extent, &images[0]),
        Tolerance::PerPixel(1.0 / 255.0),
    );
    assert_same_look(&images);
    assert_clean();
}

/// A quad from `min` to `max` in points with texture coordinates 0..1 over it.
fn quad(mesh: &mut Mesh, min: Pos2, max: Pos2, color: Color32) {
    mesh.add_rect_with_uv(
        Rect::from_min_max(min, max),
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        color,
    );
}

fn primitive(mesh: Mesh, clip: Rect) -> ClippedPrimitive {
    ClippedPrimitive {
        clip_rect: clip,
        primitive: egui::epaint::Primitive::Mesh(mesh),
    }
}

/// A deterministic byte stream.
fn bytes(seed: u64) -> impl FnMut() -> u8 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 56) as u8
    }
}

#[test]
fn painter_draws_textures_clips_and_split_meshes() {
    let gpu = device();
    const GRID: u32 = 1024;
    let extent = [GRID + 80, GRID];
    let mut frames = UiFrames::new(extent);
    let everywhere = Rect::from_min_size(Pos2::ZERO, vec2(extent[0] as f32, extent[1] as f32));

    // A grid of one-pixel quads, 2^21 triangles, drawn with nearest sampling of a texture
    // of random texels: a triangle drawn twice or not at all shows.
    let mut random = bytes(7);
    let texels: Vec<Color32> = (0..GRID * GRID)
        .map(|_| Color32::from_rgb(random(), random(), random()))
        .collect();
    let grid_image = ColorImage::new([GRID as usize; 2], texels.clone());
    let mut grid = Mesh::with_texture(TextureId::Managed(1));
    for y in 0..=GRID {
        for x in 0..=GRID {
            grid.vertices.push(Vertex {
                pos: pos2(x as f32, y as f32),
                uv: pos2(x as f32 / GRID as f32, y as f32 / GRID as f32),
                color: Color32::WHITE,
            });
        }
    }
    for y in 0..GRID {
        for x in 0..GRID {
            let i = y * (GRID + 1) + x;
            grid.indices
                .extend([i, i + 1, i + GRID + 1, i + 1, i + GRID + 2, i + GRID + 1]);
        }
    }

    // A 4x4 texture whose middle 2x2 a later frame replaces.
    let a = Color32::from_rgb(10, 200, 90);
    let b = Color32::from_rgb(250, 20, 140);
    let small = ColorImage::new([4, 4], vec![a; 16]);
    let patch = ColorImage::new([2, 2], vec![b; 4]);
    // A user texture: RGBA8 UNORM in its own slot, read as linear.
    let user = gpu
        .create_texture(
            &TextureDesc::d2(Format::Rgba8Unorm, 2, 2, TextureUsage::SAMPLED),
            TextureMemory::Dedicated,
            "user texture",
        )
        .unwrap();
    let user_slot = gpu.alloc_image_slot().unwrap();
    gpu.write_sampled(user_slot, &user, &ViewDesc::default());
    let user_texels: [u8; 16] = [
        255, 0, 0, 255, 0, 128, 0, 255, 0, 0, 64, 255, 200, 200, 200, 255,
    ];
    let mut upload: Allocation<u8> = gpu
        .gpu_malloc(MemoryKind::Upload, 16, "user texels")
        .unwrap();
    upload.as_mut_slice().unwrap().copy_from_slice(&user_texels);
    let queue = gpu.queue(QueueKind::Render).unwrap();
    let mut list = queue.command_list().unwrap();
    list.discard(&user);
    list.copy_to_texture(&user, &TextureRegion::whole_mip(user.desc(), 0), upload.gpu);
    wait(queue.submit(vec![list], &[]).unwrap());

    let x0 = GRID as f32 + 8.0;
    let primitives = |frame2: bool| {
        let mut small_mesh = Mesh::with_texture(TextureId::Managed(2));
        quad(
            &mut small_mesh,
            pos2(x0, 0.0),
            pos2(x0 + 4.0, 4.0),
            Color32::WHITE,
        );
        let mut user_mesh = Mesh::with_texture(TextureId::User(u64::from(user_slot)));
        quad(
            &mut user_mesh,
            pos2(x0 + 10.0, 0.0),
            pos2(x0 + 12.0, 2.0),
            Color32::WHITE,
        );
        // A solid quad tinted by a premultiplied vertex colour, clipped to its middle.
        let mut solid = Mesh::with_texture(TextureId::Managed(3));
        quad(
            &mut solid,
            pos2(x0, 10.0),
            pos2(x0 + 60.0, 70.0),
            Color32::from_rgba_premultiplied(200, 100, 50, 200),
        );
        let clip = Rect::from_min_max(pos2(x0 + 10.0, 20.0), pos2(x0 + 30.0, 40.0));
        let mut out = vec![
            primitive(small_mesh, everywhere),
            primitive(user_mesh, everywhere),
            primitive(solid, clip),
        ];
        if frame2 {
            out.insert(0, primitive(grid.clone(), everywhere));
        }
        out
    };

    let mut first = TexturesDelta::default();
    first.push(
        TextureId::Managed(1),
        ImageDelta::full(grid_image, TextureOptions::NEAREST),
    );
    first.push(
        TextureId::Managed(2),
        ImageDelta::full(small, TextureOptions::NEAREST),
    );
    first.push(
        TextureId::Managed(3),
        ImageDelta::full(
            ColorImage::new([1, 1], vec![Color32::WHITE]),
            TextureOptions::LINEAR,
        ),
    );
    frames.frame(first, &[&primitives(false)], 1.0, &[]);

    // The second frame patches texture 2, paints in two calls and frees texture 3, which
    // stays drawable for the rest of the frame.
    let mut second = TexturesDelta::default();
    second.push(
        TextureId::Managed(2),
        ImageDelta::partial([1, 1], patch, TextureOptions::NEAREST),
    );
    second.free(TextureId::Managed(3));
    let all = primitives(true);
    let (grid_part, rest) = all.split_at(1);
    frames.frame(second, &[grid_part, rest], 1.0, &[]);
    let layer = read_texture(
        frames.layer.texture(),
        &TextureRegion::whole_mip(frames.layer.texture().desc(), 0),
    );
    let pixel = |x: u32, y: u32| {
        let i = 4 * (y * extent[0] + x) as usize;
        [layer[i], layer[i + 1], layer[i + 2], layer[i + 3]]
    };
    let near = |got: [u8; 4], want: [u8; 4]| got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 1);

    let mut wrong = 0;
    for y in 0..GRID {
        for x in 0..GRID {
            if !near(pixel(x, y), texels[(y * GRID + x) as usize].to_array()) {
                wrong += 1;
            }
        }
    }
    assert_eq!(wrong, 0, "grid pixels differing from their texels");

    let x0 = x0 as u32;
    for y in 0..4 {
        for x in 0..4 {
            let inside = (1..3).contains(&x) && (1..3).contains(&y);
            let want = if inside { b } else { a }.to_array();
            assert!(
                near(pixel(x0 + x, y), want),
                "patched texel {x},{y}: {:?}",
                pixel(x0 + x, y)
            );
        }
    }
    for (i, t) in user_texels.chunks_exact(4).enumerate() {
        let (x, y) = (i as u32 % 2, i as u32 / 2);
        let want = [0, 1, 2].map(|c| (srgb_encode(f64::from(t[c]) / 255.0) * 255.0).round() as u8);
        let want = [want[0], want[1], want[2], t[3]];
        assert!(
            near(pixel(x0 + 10 + x, y), want),
            "user texel {x},{y}: {:?} against {want:?}",
            pixel(x0 + 10 + x, y)
        );
    }
    for y in 10..70 {
        for x in 0..60 {
            let inside = (10..30).contains(&x) && (20..40).contains(&y);
            let want = if inside { [200, 100, 50, 200] } else { [0; 4] };
            assert!(
                near(pixel(x0 + x, y), want),
                "clipped quad at {x},{y}: {:?}",
                pixel(x0 + x, y)
            );
        }
    }
    assert_clean();
}

/// Whether a capture's display-linear `got` matches `want`: within 1e-3 of the texel's
/// brightest channel, as `matches`. `orr.color`'s f32 PQ signal is within 1.3e-5 of ST 2084,
/// about 1.5e-4 in nits, and a capture over a PQ backbuffer decodes, encodes and decodes.
fn capture_matches(got: &[f32], want: [f64; 3]) -> bool {
    let brightest = want.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
    (0..3).all(|c| (f64::from(got[c]) - want[c]).abs() <= 1e-3 * brightest + 1e-6)
}

#[test]
fn compose_ui_matches_model() {
    let gpu = device();
    let extent = [256u32, 64];
    let n = (extent[0] * extent[1]) as usize;
    // Display-linear Rec.709 over 0..4 x paper white in eight hues, some outside Rec.709's
    // gamut, and a UI layer of every coverage with every fifth texel empty.
    let hues = [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.5, 0.2],
        [0.2, 0.6, 1.0],
        [1.1, -0.05, 0.0],
        [0.4, 0.4, 0.4],
    ];
    let linear: Vec<[u16; 4]> = (0..n)
        .map(|i| {
            let (x, y) = (i as u32 % extent[0], i as u32 / extent[0]);
            let scale = 4.0 * x as f32 / (extent[0] - 1) as f32;
            let h = hues[(y / 8) as usize];
            [h[0] * scale, h[1] * scale, h[2] * scale, 1.0].map(|v| f16::from_f32(v).to_bits())
        })
        .collect();
    let mut random = bytes(11);
    let ui: Vec<[u8; 4]> = (0..n)
        .map(|i| {
            if i % 5 == 0 {
                return [0; 4];
            }
            let a = random();
            let mut c = || (u32::from(random()) * (u32::from(a) + 1) / 256) as u8;
            [c(), c(), c(), a]
        })
        .collect();

    let source = Target::new(Format::Rgba16Float, extent, "linear");
    let layer = UiLayer::new(gpu, extent).unwrap();
    let mut upload: Allocation<u8> = gpu
        .gpu_malloc(MemoryKind::Upload, (n * 12) as u64, "compose ui inputs")
        .unwrap();
    {
        let mapped = upload.as_mut_slice().unwrap();
        mapped[..n * 8].copy_from_slice(bytemuck::cast_slice(&linear));
        mapped[n * 8..].copy_from_slice(bytemuck::cast_slice(&ui));
    }
    let kernels = OutputKernels::new(gpu).unwrap();
    for (display, format) in displays() {
        let backbuffer = Target::new(format, extent, "backbuffer");
        let presented = Target::new(format, extent, "presented");
        let black = Target::new(format, extent, "ui only");
        let mut roots: Vec<ComposeRoot> = (0..3).map(|_| ComposeRoot::new(gpu).unwrap()).collect();
        let mut view = OutputView::new(gpu, 3).unwrap();
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.discard(layer.texture());
        let whole = TextureRegion::whole_mip(source.texture.desc(), 0);
        list.copy_to_texture(&source.texture, &whole, upload.gpu);
        list.copy_to_texture(layer.texture(), &whole, upload.gpu.offset((n * 8) as u64));
        list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        let pass = |source, ui, target: &Target| ComposePass {
            source,
            ui,
            target: target.store,
            extent,
            display: &display,
        };
        kernels.compose(
            &mut list,
            &mut roots[0],
            &pass(ComposeSource::Linear(source.sampled), None, &backbuffer),
        );
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        let ui_index = Some(layer.index());
        kernels.compose(
            &mut list,
            &mut roots[1],
            &pass(
                ComposeSource::Encoded(backbuffer.sampled),
                ui_index,
                &presented,
            ),
        );
        kernels.compose(
            &mut list,
            &mut roots[2],
            &pass(ComposeSource::Black, ui_index, &black),
        );
        // Display screenshots with UI: over tonemap's output, over the backbuffer (frame 1)
        // and over black (frame 2).
        for (frame, source) in [
            ComposeSource::Linear(source.sampled),
            ComposeSource::Encoded(backbuffer.sampled),
            ComposeSource::Black,
        ]
        .into_iter()
        .enumerate()
        {
            let capture = CapturePass {
                source,
                ui: ui_index,
                extent,
                display: &display,
                frame: frame as u64,
            };
            kernels.capture(&mut list, &mut view, &capture).unwrap();
        }
        wait(queue.submit(vec![list], &[]).unwrap());
        let shots = view.screenshots(2);
        assert_eq!(shots.iter().map(|s| s.frame).collect::<Vec<_>>(), [0, 1, 2]);

        let (bb_bytes, presented_bytes) = (backbuffer.read(), presented.read());
        let bb = signals(&bb_bytes, format);
        let got = signals(&presented_bytes, format);
        let got_black = signals(&black.read(), format);
        let texel = bb_bytes.len() / n;
        for i in 0..n {
            let u = ui[i].map(|c| f64::from(c) / 255.0);
            let over = |under: [f64; 3]| {
                let blended =
                    [0, 1, 2].map(|c| srgb_decode(u[c] + (1.0 - u[3]) * srgb_encode(under[c])));
                encode(blended, &display)
            };
            if ui[i] == [0; 4] {
                assert_eq!(
                    presented_bytes[i * texel..(i + 1) * texel],
                    bb_bytes[i * texel..(i + 1) * texel],
                    "{:?} texel {i} without UI changed",
                    display.encoding
                );
            } else {
                let want = over(decode([bb[i][0], bb[i][1], bb[i][2]], &display));
                let g = [got[i][0], got[i][1], got[i][2]];
                assert!(
                    matches(g, want, &display),
                    "{:?} texel {i}: {g:?} against {want:?}",
                    display.encoding
                );
            }
            let under = linear[i].map(|v| f16::from_bits(v).to_f64());
            let want = decode(over([under[0], under[1], under[2]]), &display);
            let g = &shots[0].rgba[4 * i..4 * i + 4];
            assert!(
                capture_matches(g, want),
                "{:?} capture texel {i}: {g:?} against {want:?}",
                display.encoding
            );
            let want = decode(
                if ui[i] == [0; 4] {
                    [bb[i][0], bb[i][1], bb[i][2]]
                } else {
                    over(decode([bb[i][0], bb[i][1], bb[i][2]], &display))
                },
                &display,
            );
            let g = &shots[1].rgba[4 * i..4 * i + 4];
            assert!(
                capture_matches(g, want),
                "{:?} backbuffer capture texel {i}: {g:?} against {want:?}",
                display.encoding
            );
            let want = over([0.0; 3]);
            let g = &shots[2].rgba[4 * i..4 * i + 4];
            assert!(
                capture_matches(g, decode(want, &display)),
                "{:?} UI-only capture texel {i}: {g:?}",
                display.encoding
            );
            let g = [got_black[i][0], got_black[i][1], got_black[i][2]];
            assert!(
                matches(g, want, &display),
                "{:?} UI-only texel {i}: {g:?} against {want:?}",
                display.encoding
            );
        }
    }
    assert_clean();
}
