//! Shading a raster view with the `sun_shadow` kernel: each pixel of the view's G-buffer under
//! a sun with one traced shadow ray and a constant sky, composed to the frame, and on request
//! read back for a headless capture.

use orr::orr_gpu::{Allocation, Format, Hazards, MemoryKind, SpecData, Stages};
use orr::orr_output::Image;
use orr::{Frame, PipelineId, RasterView, Renderer};
use orr_math::Vec3;

use crate::shaders::{capture, sun_shadow};
use crate::{Capture, device};

/// Sun illuminance and sky radiance, relative to paper white.
const SUN_ILLUMINANCE: f32 = 3.0;
const SKY: [f32; 3] = [0.18, 0.24, 0.34];

pub struct SunShade {
    shade: PipelineId,
    encode: PipelineId,
    roots: Allocation<sun_shadow::Root>,
    capture_root: Allocation<capture::Root>,
    readback: Option<Allocation<u32>>,
    capture_extent: [u32; 2],
    image: Option<Image>,
    name: String,
    sun: Vec3,
}

impl SunShade {
    /// Shades under `sun`, the unit direction to it, and with `capture_extent` can capture an
    /// image of that size.
    pub fn new(
        renderer: &mut Renderer,
        name: &str,
        frames_in_flight: u32,
        capture_extent: Option<[u32; 2]>,
        sun: Vec3,
    ) -> Self {
        let gpu = device();
        let shade = renderer
            .compute_pipeline(&sun_shadow::MODULE, "main", SpecData::default())
            .expect("shade pipeline");
        let encode = renderer
            .compute_pipeline(&capture::MODULE, "main", SpecData::default())
            .expect("capture pipeline");
        let roots = gpu
            .gpu_malloc::<sun_shadow::Root>(
                MemoryKind::Upload,
                u64::from(frames_in_flight),
                &format!("{name} roots"),
            )
            .expect("roots");
        let capture_root = gpu
            .gpu_malloc::<capture::Root>(MemoryKind::Upload, 1, &format!("{name} capture root"))
            .expect("capture root");
        let readback = capture_extent.map(|[w, h]| {
            gpu.gpu_malloc::<u32>(
                MemoryKind::Readback,
                u64::from(w * h),
                &format!("{name} capture"),
            )
            .expect("readback")
        });
        Self {
            shade,
            encode,
            roots,
            capture_root,
            readback,
            capture_extent: capture_extent.unwrap_or_default(),
            image: None,
            name: name.to_owned(),
            sun,
        }
    }

    /// Shades `view` into an image of `extent` with the kernel's `mode`, encodes it for the
    /// capture on the final frame and composes it to the frame.
    pub fn record(
        &mut self,
        f: &mut Frame<'_>,
        view: &RasterView,
        extent: [u32; 2],
        mode: u32,
        final_frame: bool,
    ) {
        let fresh = self
            .image
            .as_ref()
            .is_none_or(|im| im.texture().desc().extent[..2] != extent);
        if fresh {
            self.image = Some(
                Image::new(
                    device(),
                    Format::Rgba16Float,
                    extent,
                    &format!("{} shaded", self.name),
                )
                .expect("image"),
            );
        }
        let target = self.image.as_ref().expect("image");
        let roots = self.roots.as_mut_slice().expect("roots");
        let slot = f.id() % roots.len() as u64;
        roots[slot as usize] = sun_shadow::Root {
            scene: view.scene,
            view: view.view,
            gbuffer: view.gbuffer,
            target: target.storage,
            mode,
            sun: [self.sun.x, self.sun.y, self.sun.z, SUN_ILLUMINANCE],
            sky: [SKY[0], SKY[1], SKY[2], 1.0],
        };
        let (shade, encode) = (f.pipeline(self.shade), f.pipeline(self.encode));
        let groups = [extent[0].div_ceil(8), extent[1].div_ceil(8), 1];
        let list = f.list().expect("frame list");
        if fresh {
            list.discard(target.texture());
        }
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.begin_pass("sun shadow", false);
        list.dispatch(shade, self.roots.gpu().offset(slot), groups);
        list.end_pass();
        if let Some(out) = self.readback.as_ref().filter(|_| final_frame) {
            self.capture_root.as_mut_slice().expect("capture root")[0] = capture::Root {
                pixels: out.gpu(),
                source: target.sampled,
                width: extent[0],
                height: extent[1],
                pad0: 0,
                pad1: 0,
            };
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            list.dispatch(encode, self.capture_root.gpu(), groups);
            list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        }
        f.compose(target.sampled).expect("compose");
    }

    /// The final frame's capture, once its work has finished.
    pub fn capture(self) -> Option<Capture> {
        let extent = self.capture_extent;
        self.readback
            .map(|words| Capture::from_readback(extent, &words))
    }
}
