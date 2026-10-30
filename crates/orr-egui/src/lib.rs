//! An egui painter for orr: draws egui's tessellated meshes into a UI layer with a mesh
//! shader, blending in gamma space like egui's reference painters.
//!
//! The UI layer is an RGBA8 UNORM texture with `COLOR` usage holding premultiplied,
//! sRGB-encoded colour. Managed egui textures live in the texture heap as RGBA8 UNORM
//! textures with one mip holding egui's sRGB-encoded bytes, so filtering happens in gamma
//! space as egui expects. `TextureId::User(n)` draws image slot `n` directly: it holds linear
//! colour (an sRGB format decodes in the sampler), filtered linearly, clamped and encoded to
//! sRGB. Paint callbacks are not drawn.

use std::collections::HashMap;
use std::mem::{offset_of, size_of};

use egui::epaint::{ImageData, Primitive, Vertex};
use egui::{
    ClippedPrimitive, TextureFilter, TextureId, TextureOptions, TextureWrapMode, TexturesDelta,
};
use orr_gpu::{
    AddressMode, Allocation, Blend, ColorAttachment, ColorLoad, ColorTarget, CommandList, Device,
    Error, Filter, Format, GpuPtr, Hazards, MemoryKind, MeshPipeline, MeshPipelineDesc, Rendering,
    SamplerDesc, SamplerIndex, ShaderStage, SpecData, Stages, Texture, TextureDesc, TextureIndex,
    TextureMemory, TextureRegion, TextureUsage, ViewDesc,
};

mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

use shaders::painter::{self, EguiVertex, Root};

// The shader reads egui's vertices in place.
const _: () = {
    assert!(size_of::<Vertex>() == size_of::<EguiVertex>());
    assert!(offset_of!(Vertex, uv) == offset_of!(EguiVertex, uv));
    assert!(offset_of!(Vertex, color) == offset_of!(EguiVertex, color));
};

/// Triangles one mesh workgroup draws (`painter.slang`).
const GROUP: u32 = 32;
/// Workgroups per draw: Vulkan's minimum `maxMeshWorkGroupCount`.
const MAX_GROUPS: u32 = 65_535;
/// Bytes of a new upload block, unless one allocation needs more.
const BLOCK_BYTES: u64 = 1 << 20;
/// Alignment of draw roots in the upload blocks: the largest uniform-buffer offset
/// alignment Vulkan allows.
const ROOT_ALIGN: u64 = 256;

/// Paints egui output into a UI layer. Holds one list of upload blocks per frame in flight.
pub struct Painter {
    device: Device,
    pipeline: MeshPipeline,
    slots: Vec<FrameSlot>,
    /// The frame of the latest call.
    frame: Option<u64>,
    textures: HashMap<u64, Managed>,
    /// Textures egui freed this frame, released once the frame's paints are recorded.
    to_free: Vec<u64>,
    user_sampler: SamplerIndex,
}

/// A managed egui texture in the heap.
struct Managed {
    texture: Texture,
    slot: u32,
    sampler: SamplerIndex,
}

/// Upload blocks and released textures of one frame slot.
#[derive(Default)]
struct FrameSlot {
    blocks: Vec<Allocation<u8>>,
    /// The block being filled and the bytes used in it.
    block: usize,
    used: u64,
    /// Textures the slot's frame stopped using, released when the slot is reused.
    released: Vec<Managed>,
}

impl Painter {
    /// Creates the pipeline. `frames_in_flight` is the renderer's: upload blocks of frame
    /// `id` are reused by frame `id + frames_in_flight`, which must start only once frame
    /// `id` has completed.
    pub fn new(gpu: &Device, frames_in_flight: u32) -> Result<Self, Error> {
        let stage = |entry| ShaderStage::new(&painter::MODULE, entry, SpecData::default());
        let colors = [ColorTarget {
            blend: Some(Blend::PREMULTIPLIED),
            ..ColorTarget::new(Format::Rgba8Unorm)
        }];
        let pipeline = gpu.create_mesh_pipeline(
            &MeshPipelineDesc {
                pixel: Some(stage("pixel_main")?),
                colors: &colors,
                ..MeshPipelineDesc::new(stage("mesh_main")?)
            },
            "egui",
        )?;
        Ok(Self {
            device: gpu.clone(),
            pipeline,
            slots: (0..frames_in_flight.max(1))
                .map(|_| FrameSlot::default())
                .collect(),
            frame: None,
            textures: HashMap::new(),
            to_free: Vec::new(),
            user_sampler: gpu.sampler(&sampler_desc(TextureOptions::LINEAR))?,
        })
    }

    /// Records the texture changes of one egui frame: call it before [`Painter::paint`] with
    /// the frame's `textures_delta`. Textures it frees stay drawable until the next frame.
    pub fn update_textures(
        &mut self,
        cmd: &mut CommandList<'_>,
        frame_id: u64,
        mut delta: TexturesDelta,
    ) -> Result<(), Error> {
        self.begin(frame_id);
        let set = std::mem::take(&mut delta.set);
        self.to_free
            .extend(delta.free.drain().filter_map(|id| match id {
                TextureId::Managed(id) => Some(id),
                TextureId::User(_) => None,
            }));
        let mut copied = false;
        for (id, deltas) in set {
            let TextureId::Managed(id) = id else {
                continue;
            };
            for d in &deltas {
                let ImageData::Color(image) = &d.image;
                let [w, h] = image.size.map(|s| s as u32);
                if d.pos.is_none() {
                    let desc = TextureDesc::d2(Format::Rgba8Unorm, w, h, TextureUsage::SAMPLED);
                    let texture = self.device.create_texture(
                        &desc,
                        TextureMemory::Dedicated,
                        "egui texture",
                    )?;
                    let slot = self.device.alloc_image_slot()?;
                    self.device
                        .write_sampled(slot, &texture, &ViewDesc::default());
                    cmd.discard(&texture);
                    let new = Managed {
                        texture,
                        slot,
                        sampler: self.device.sampler(&sampler_desc(d.options))?,
                    };
                    if let Some(old) = self.textures.insert(id, new) {
                        self.slot(frame_id).released.push(old);
                    }
                }
                if !self.textures.contains_key(&id) {
                    continue;
                }
                let src = self.put(frame_id, bytemuck::cast_slice(&image.pixels), 16)?;
                let texture = &self.textures[&id].texture;
                let [x, y] = d.pos.unwrap_or([0, 0]).map(|p| p as u32);
                let region = TextureRegion {
                    offset: [x, y, 0],
                    extent: [w, h, 1],
                    ..TextureRegion::whole_mip(texture.desc(), 0)
                };
                cmd.copy_to_texture(texture, &region, src);
                copied = true;
            }
        }
        if copied {
            cmd.barrier(Stages::TRANSFER, Stages::PIXEL, Hazards::empty());
        }
        Ok(())
    }

    /// Draws `primitives` over what `ui_layer` holds, inside one rendering pass. Order the
    /// layer's earlier writes before it; readers of what it draws need a barrier from
    /// [`Stages::COLOR_OUT`].
    pub fn paint(
        &mut self,
        cmd: &mut CommandList<'_>,
        frame_id: u64,
        ui_layer: &Texture,
        primitives: &[ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<(), Error> {
        self.begin(frame_id);
        let [width, height, _] = ui_layer.desc().extent;
        let pixel = |v: f32, max: u32| (v * pixels_per_point).round().clamp(0.0, max as f32) as u32;
        let mut draws = Vec::new();
        for p in primitives {
            let Primitive::Mesh(mesh) = &p.primitive else {
                continue;
            };
            let clip = p.clip_rect;
            let (x0, y0) = (pixel(clip.min.x, width), pixel(clip.min.y, height));
            let (x1, y1) = (pixel(clip.max.x, width), pixel(clip.max.y, height));
            let triangles = (mesh.indices.len() / 3) as u32;
            if x1 <= x0 || y1 <= y0 || triangles == 0 {
                continue;
            }
            let (texture, sampler, linear) = match mesh.texture_id {
                TextureId::Managed(id) => match self.textures.get(&id) {
                    Some(m) => (TextureIndex(m.slot), m.sampler, 0),
                    None => continue,
                },
                TextureId::User(n) => (TextureIndex(n as u32), self.user_sampler, 1),
            };
            let vertices = self.put(frame_id, bytemuck::cast_slice(&mesh.vertices), 16)?;
            let indices = self.put(frame_id, bytemuck::cast_slice(&mesh.indices), 16)?;
            let per_draw = GROUP * MAX_GROUPS;
            for first in (0..triangles).step_by(per_draw as usize) {
                let n = (triangles - first).min(per_draw);
                let root = Root {
                    vertices: vertices.cast(),
                    indices: indices.cast(),
                    scale_x: 2.0 * pixels_per_point / width as f32,
                    scale_y: 2.0 * pixels_per_point / height as f32,
                    first_triangle: first,
                    triangles: n,
                    texture,
                    sampler,
                    linear,
                    _pad0: 0,
                };
                let root = self.put(frame_id, bytemuck::bytes_of(&root), ROOT_ALIGN)?;
                draws.push(([x0, y0], [x1 - x0, y1 - y0], root, n.div_ceil(GROUP)));
            }
        }
        cmd.begin_rendering(&Rendering {
            extent: [width, height],
            colors: &[ColorAttachment {
                texture: ui_layer,
                load: ColorLoad::Keep,
                store: true,
            }],
            depth: None,
        });
        for (offset, extent, root, groups) in draws {
            cmd.set_scissor(offset, extent);
            cmd.draw_mesh_tasks(&self.pipeline, root.cast::<Root>(), [groups, 1, 1]);
        }
        cmd.end_rendering();
        Ok(())
    }

    fn slot(&mut self, frame_id: u64) -> &mut FrameSlot {
        let n = self.slots.len() as u64;
        &mut self.slots[(frame_id % n) as usize]
    }

    /// On the first call of a new frame: hands the previous frame's freed textures to its
    /// slot and resets the new frame's slot, whose previous frame has completed.
    fn begin(&mut self, frame_id: u64) {
        if self.frame == Some(frame_id) {
            return;
        }
        if let Some(previous) = self.frame {
            for id in std::mem::take(&mut self.to_free) {
                if let Some(m) = self.textures.remove(&id) {
                    self.slot(previous).released.push(m);
                }
            }
        }
        self.frame = Some(frame_id);
        let device = self.device.clone();
        let slot = self.slot(frame_id);
        slot.block = 0;
        slot.used = 0;
        for m in slot.released.drain(..) {
            device.free_image_slot_after_submitted(m.slot);
        }
    }

    /// Copies `bytes` into the frame slot's upload blocks at `align` and returns their
    /// address, adding a block when the current ones are full.
    fn put(&mut self, frame_id: u64, bytes: &[u8], align: u64) -> Result<GpuPtr<u8>, Error> {
        let device = self.device.clone();
        let slot = self.slot(frame_id);
        let len = bytes.len() as u64;
        loop {
            if let Some(block) = slot.blocks.get_mut(slot.block) {
                let at = slot.used.next_multiple_of(align);
                if at + len <= block.size {
                    let mapped = block.as_mut_slice().expect("upload memory is mapped");
                    mapped[at as usize..(at + len) as usize].copy_from_slice(bytes);
                    slot.used = at + len;
                    return Ok(block.gpu.offset(at));
                }
                slot.block += 1;
                slot.used = 0;
                continue;
            }
            let size = BLOCK_BYTES.max(len);
            slot.blocks
                .push(device.gpu_malloc(MemoryKind::Upload, size, "egui upload")?);
        }
    }
}

impl Drop for Painter {
    fn drop(&mut self) {
        let released = self.slots.iter_mut().flat_map(|s| s.released.drain(..));
        for m in released.chain(self.textures.drain().map(|(_, m)| m)) {
            self.device.free_image_slot_after_submitted(m.slot);
        }
    }
}

fn sampler_desc(o: TextureOptions) -> SamplerDesc {
    let filter = |f| match f {
        TextureFilter::Nearest => Filter::Nearest,
        TextureFilter::Linear => Filter::Linear,
    };
    let address = match o.wrap_mode {
        TextureWrapMode::ClampToEdge => AddressMode::ClampToEdge,
        TextureWrapMode::Repeat => AddressMode::Repeat,
        TextureWrapMode::MirroredRepeat => AddressMode::MirroredRepeat,
    };
    SamplerDesc {
        mag: filter(o.magnification),
        min: filter(o.minification),
        mip: Filter::Nearest,
        address: [address; 3],
        max_lod: 0.0,
        ..SamplerDesc::default()
    }
}
