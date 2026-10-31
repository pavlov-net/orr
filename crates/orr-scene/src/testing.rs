//! Headless testing: one device with validation on, readback, and comparison against
//! reference images. `ORR_BLESS=1` rewrites the references instead of comparing.
//!
//! 8-bit images are display-encoded (sRGB) and compare with FLIP; float images compare per
//! pixel.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use bytemuck::Pod;
use orr_gpu::{
    Device, DeviceDesc, GpuPtr, Hazards, MemoryKind, QueueKind, QueueRequest, Stages, Texture,
    TextureRegion, TimelinePoint,
};

struct StderrLog;

impl log::Log for StderrLog {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Info
    }

    #[allow(clippy::print_stderr)]
    fn log(&self, r: &log::Record<'_>) {
        if self.enabled(r.metadata()) {
            eprintln!("{} {}", r.level(), r.args());
        }
    }

    fn flush(&self) {}
}

/// The pipeline cache file the test binaries of one build profile share, beside them under
/// `target/<profile>/`. Shader edits leave stale pipelines in it, so it starts over once it
/// outgrows 256 MiB.
fn pipeline_cache_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let path = exe.parent()?.parent()?.join("test-pipeline-cache.bin");
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > 256 << 20) {
        let _ = std::fs::remove_file(&path);
    }
    Some(path)
}

/// The process's test device: every queue kind, validation on, messages logged to stderr,
/// pipelines cached in `target/<profile>/test-pipeline-cache.bin` (written by [`assert_clean`]).
pub fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| {
        let _ = log::set_logger(&StderrLog);
        log::set_max_level(log::LevelFilter::Info);
        let cache = pipeline_cache_path();
        let desc = DeviceDesc {
            name: "orr-test",
            pipeline_cache: cache.as_deref(),
            validation: true,
            queues: QueueRequest {
                render_compute: true,
                async_compute: true,
                copy: true,
                present: true,
            },
            ..DeviceDesc::default()
        };
        Device::new(&desc).expect("test device")
    })
}

/// Writes [`device`]'s pipeline cache, then fails the calling test if the validation layer
/// has reported anything on it.
pub fn assert_clean() {
    if let Err(e) = device().save_pipeline_cache() {
        log::warn!("{e}");
    }
    assert_eq!(
        device().validation_messages(),
        0,
        "validation messages (see the log)"
    );
}

/// Waits for `point` on [`device`].
pub fn wait(point: TimelinePoint) {
    let render = device().queue(QueueKind::Render).unwrap();
    render.wait(point, Duration::from_secs(10)).expect("wait");
}

/// Sets what copies of `model` add to the emissive list, as model loading does when the
/// model becomes resident.
pub fn set_model_emission(
    scenes: &mut crate::Scenes,
    model: crate::ModelId,
    emission: Option<crate::ModelEmission>,
) {
    scenes.set_model_emission(model, emission);
}

/// Copies `count` values at `src` back to the CPU, after all work submitted so far.
pub fn read_buffer<T: Pod>(src: GpuPtr<T>, count: u64) -> Vec<T> {
    read_buffer_on(device(), src, count)
}

/// [`read_buffer`] on device `d`.
pub fn read_buffer_on<T: Pod>(d: &Device, src: GpuPtr<T>, count: u64) -> Vec<T> {
    let out = d
        .gpu_malloc::<T>(MemoryKind::Readback, count, "test readback")
        .unwrap();
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
    list.copy(out.gpu.cast(), src.cast(), out.size);
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    let point = render.submit(vec![list], &[]).unwrap();
    render.wait(point, Duration::from_secs(10)).expect("wait");
    out.as_slice().unwrap().to_vec()
}

/// Copies `region` of `t` back to the CPU, tightly packed, after all work submitted so far.
pub fn read_texture(t: &Texture, region: &TextureRegion) -> Vec<u8> {
    read_texture_on(device(), t, region)
}

/// [`read_texture`] on device `d`.
pub fn read_texture_on(d: &Device, t: &Texture, region: &TextureRegion) -> Vec<u8> {
    let bytes = region.bytes(t.desc().format);
    let out = d
        .gpu_malloc::<u8>(MemoryKind::Readback, bytes, "test readback")
        .unwrap();
    let render = d.queue(QueueKind::Render).unwrap();
    let mut list = render.command_list().unwrap();
    list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
    list.copy_from_texture(out.gpu, t, region);
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    let point = render.submit(vec![list], &[]).unwrap();
    render.wait(point, Duration::from_secs(10)).expect("wait");
    out.as_slice().unwrap().to_vec()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pixels {
    /// Display-encoded RGBA, 4 bytes per pixel.
    Rgba8(Vec<u8>),
    /// Linear RGBA, 4 floats per pixel.
    Rgba32F(Vec<f32>),
}

/// An image to compare, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Pixels,
}

impl Image {
    fn extension(&self) -> &'static str {
        match self.pixels {
            Pixels::Rgba8(_) => "png",
            Pixels::Rgba32F(_) => "exr",
        }
    }

    fn save(&self, path: &Path) {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).unwrap();
        }
        let saved = match &self.pixels {
            Pixels::Rgba8(p) => image::RgbaImage::from_raw(self.width, self.height, p.clone())
                .expect("image size")
                .save(path),
            Pixels::Rgba32F(p) => image::Rgba32FImage::from_raw(self.width, self.height, p.clone())
                .expect("image size")
                .save(path),
        };
        saved.unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    }

    fn load(path: &Path, like: &Image) -> Image {
        let img = image::open(path).unwrap_or_else(|e| {
            panic!(
                "reading reference {}: {e} (ORR_BLESS=1 writes it)",
                path.display()
            )
        });
        let (width, height) = (img.width(), img.height());
        let pixels = match like.pixels {
            Pixels::Rgba8(_) => Pixels::Rgba8(img.into_rgba8().into_raw()),
            Pixels::Rgba32F(_) => Pixels::Rgba32F(img.into_rgba32f().into_raw()),
        };
        Image {
            width,
            height,
            pixels,
        }
    }
}

/// FLIP error of an image against its reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipError {
    pub mean: f32,
    pub p99: f32,
}

fn rgb8(i: &Image) -> nv_flip::FlipImageRgb8 {
    let Pixels::Rgba8(p) = &i.pixels else {
        panic!("FLIP compares 8-bit display-encoded images");
    };
    let rgb: Vec<u8> = p.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
    nv_flip::FlipImageRgb8::with_data(i.width, i.height, &rgb)
}

/// LDR FLIP of `test` against `reference` at the default viewing distance.
pub fn flip(reference: &Image, test: &Image) -> FlipError {
    assert_eq!(
        (reference.width, reference.height),
        (test.width, test.height),
        "image sizes differ"
    );
    let map = nv_flip::flip(
        rgb8(reference),
        rgb8(test),
        nv_flip::DEFAULT_PIXELS_PER_DEGREE,
    );
    let mut pool = nv_flip::FlipPool::from_image(&map);
    FlipError {
        mean: pool.mean(),
        p99: pool.get_percentile(0.99, false),
    }
}

/// The largest per-channel difference: in [0, 1] for 8-bit images, in the images' units
/// for float images. A NaN on one side only counts as infinite.
pub fn max_difference(reference: &Image, test: &Image) -> f32 {
    assert_eq!(
        (reference.width, reference.height),
        (test.width, test.height),
        "image sizes differ"
    );
    match (&reference.pixels, &test.pixels) {
        (Pixels::Rgba8(a), Pixels::Rgba8(b)) => a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f32::from(x.abs_diff(y)) / 255.0)
            .fold(0.0, f32::max),
        (Pixels::Rgba32F(a), Pixels::Rgba32F(b)) => a
            .iter()
            .zip(b)
            .map(|(&x, &y)| match (x.is_nan(), y.is_nan()) {
                (true, true) => 0.0,
                (false, false) if x == y => 0.0,
                (false, false) => (x - y).abs(),
                _ => f32::INFINITY,
            })
            .fold(0.0, f32::max),
        _ => panic!("images of different pixel types"),
    }
}

/// How close an image must be to its reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tolerance {
    /// 8-bit images: FLIP mean and 99th percentile at most these.
    Flip { mean: f32, p99: f32 },
    /// Largest per-channel difference at most this ([`max_difference`]).
    PerPixel(f32),
}

/// Compares `image` with the reference at `reference`, whose extension is set from the
/// pixel type (`.png` or `.exr`). With `ORR_BLESS=1` writes `image` as the reference
/// instead. On a mismatch the image is written to the system temp directory and the test
/// panics naming both files.
pub fn check(reference: &Path, image: &Image, tolerance: Tolerance) {
    let path = reference.with_extension(image.extension());
    if std::env::var_os("ORR_BLESS").is_some_and(|v| v == "1") {
        image.save(&path);
        return;
    }
    let expected = Image::load(&path, image);
    let failure = if (expected.width, expected.height) != (image.width, image.height) {
        Some(format!(
            "size {}x{} against the reference's {}x{}",
            image.width, image.height, expected.width, expected.height
        ))
    } else {
        match tolerance {
            Tolerance::Flip { mean, p99 } => {
                let e = flip(&expected, image);
                (e.mean > mean || e.p99 > p99)
                    .then(|| format!("FLIP mean {} p99 {} (allowed {mean}, {p99})", e.mean, e.p99))
            }
            Tolerance::PerPixel(max) => {
                let d = max_difference(&expected, image);
                (d > max).then(|| format!("largest difference {d} (allowed {max})"))
            }
        }
    };
    if let Some(why) = failure {
        let name = path.file_name().unwrap_or_default();
        let actual: PathBuf = std::env::temp_dir().join("orr-test-output").join(name);
        image.save(&actual);
        panic!(
            "{} differs from its reference: {why}; this run's image is {}",
            path.display(),
            actual.display()
        );
    }
}

/// The global model, mesh and material tables.
pub fn global_tables(
    scenes: &crate::Scenes,
) -> (
    GpuPtr<crate::ModelGpu>,
    GpuPtr<crate::MeshGpu>,
    GpuPtr<crate::MaterialGpu>,
) {
    scenes.tables()
}

/// The texture of cooked texture `id` in a ready mount.
pub fn mounted_texture(
    scenes: &crate::Scenes,
    m: crate::MountId,
    id: orr_pack::AssetId,
) -> Option<std::sync::Arc<Texture>> {
    scenes.texture_object(m, id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(width: u32, height: u32) -> Image {
        let mut p = Vec::new();
        for y in 0..height {
            for x in 0..width {
                p.extend([(x * 255 / width) as u8, (y * 255 / height) as u8, 96, 255]);
            }
        }
        Image {
            width,
            height,
            pixels: Pixels::Rgba8(p),
        }
    }

    #[test]
    fn references_round_trip_and_catch_a_changed_block() {
        let dir = std::env::temp_dir().join(format!("orr-testing-{}", std::process::id()));
        let reference = dir.join("gradient");
        let img = gradient(64, 48);
        img.save(&reference.with_extension("png"));
        let tight = Tolerance::Flip {
            mean: 1e-4,
            p99: 1e-3,
        };
        check(&reference, &img, tight);

        let mut changed = img.clone();
        let Pixels::Rgba8(p) = &mut changed.pixels else {
            unreachable!()
        };
        for y in 10..18 {
            for x in 20..28 {
                p[(y * 64 + x) * 4] = 255;
            }
        }
        let e = flip(&img, &changed);
        assert!(e.mean > 1e-3 && e.p99 > 0.1, "{e:?}");
        let caught = std::panic::catch_unwind(|| check(&reference, &changed, tight));
        assert!(caught.is_err(), "a changed block passed");

        let float = Image {
            width: 2,
            height: 1,
            pixels: Pixels::Rgba32F(vec![1.0, 2.0, 3.0, 1.0, 1e4, 0.0, 0.5, 1.0]),
        };
        float.save(&dir.join("float.exr"));
        check(&dir.join("float"), &float, Tolerance::PerPixel(0.0));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
