//! Paints a finished GPUI scene into orr's transparent UI layer.
//!
//! A host implementing `gpui::PlatformWindow` returns [Painter::sprite_atlas] from its
//! `sprite_atlas` hook and passes the scene from `draw` to [Painter::prepare].
//! Return false from `is_subpixel_rendering_supported` so GPUI generates grayscale glyphs
//! for the transparent layer. Record [Painter::paint] from the renderer's UI callback.

mod atlas;
mod scene;
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use gpui::{PlatformAtlas, Scene};
use orr_gpu::{
    Blend, ColorAttachment, ColorLoad, ColorTarget, CommandList, Device, Error, Format, Hazards,
    MemoryKind, PrimitiveTopology, QueueKind, ROOT_ALIGN, Rendering, SamplerDesc, SamplerIndex,
    ShaderStage, SpecData, Stages, Texture, TextureDesc, TextureIndex, TextureMemory,
    TextureRegion, TextureUsage, VertexPipeline, VertexPipelineDesc, ViewDesc,
    interop::defer_after_submitted,
};

use atlas::{Atlas, Page, Tile};
use scene::SceneData;
use shaders::painter::{self, Root};

/// A prepared frame can be painted multiple times, including into layers for generated frames.
pub struct PreparedFrame {
    scene: SceneData,
    atlas: Arc<Atlas>,
}

/// Owns a GPUI sprite atlas and records its scene batches on the renderer's graphics queue.
pub struct Painter {
    device: Device,
    pipeline: VertexPipeline,
    atlas: Arc<Atlas>,
    textures: HashMap<u32, PageTexture>,
    sampler: SamplerIndex,
}

struct PageTexture {
    texture: Texture,
    index: TextureIndex,
    initialized: bool,
}

impl PageTexture {
    fn new(device: &Device, page: &Page) -> Result<Self, Error> {
        let [width, height] = page.extent;
        let desc = TextureDesc::d2(page.format, width, height, TextureUsage::SAMPLED);
        let texture =
            device.create_texture(&desc, TextureMemory::Dedicated, "GPUI sprite atlas")?;
        let index =
            device.write_sampled(device.alloc_image_slot()?, &texture, &ViewDesc::default());
        Ok(Self {
            texture,
            index,
            initialized: false,
        })
    }
}

impl Painter {
    pub fn new(device: &Device) -> Result<Self, Error> {
        let stage = |entry| ShaderStage::new(&painter::MODULE, entry, SpecData::default());
        let pipeline = device.create_vertex_pipeline(
            &VertexPipelineDesc {
                topology: PrimitiveTopology::TriangleStrip,
                pixel: Some(stage("pixel_main")?),
                colors: &[ColorTarget {
                    blend: Some(Blend::PREMULTIPLIED),
                    ..ColorTarget::new(Format::Rgba8Unorm)
                }],
                ..VertexPipelineDesc::new(stage("vertex_main")?)
            },
            "GPUI",
        )?;
        Ok(Self {
            device: device.clone(),
            pipeline,
            atlas: Arc::new(Atlas::new(device.caps().limits.max_image_extent_2d)),
            textures: HashMap::new(),
            sampler: device.sampler(&SamplerDesc::default())?,
        })
    }

    /// Supply this to the GPUI platform window's sprite-atlas hook. Use grayscale text
    /// antialiasing for a layer whose background is composed later. Polychrome bitmaps
    /// supplied to the atlas are straight BGRA.
    pub fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        Arc::clone(&self.atlas) as _
    }

    /// Prepare the finished scene from the platform window's draw callback. New atlas pixels
    /// are submitted to the Render queue here and are ready in queue order for [Self::paint].
    /// Scenes containing external surfaces or native views require host composition and
    /// return an error.
    pub fn prepare(&mut self, scene: &Scene) -> Result<PreparedFrame, Error> {
        let atlas = Arc::clone(&self.atlas);
        let scene = {
            let mut state = atlas.0.lock().unwrap();
            let scene = SceneData::from_gpui(scene, &state.backend)?;
            self.collect_atlas(&mut state.backend.pages);
            self.upload_atlas(&state.backend.pages, &scene.tiles)?;
            scene
        };
        Ok(PreparedFrame { scene, atlas })
    }

    /// Record a prepared scene on the Render queue, over the layer's existing contents.
    /// The layer is RGBA8 UNORM with premultiplied sRGB-encoded color. Order earlier writes to
    /// the layer before this call and barrier from [Stages::COLOR_OUT] before consuming it.
    pub fn paint(
        &self,
        cmd: &mut CommandList<'_>,
        layer: &Texture,
        frame: &PreparedFrame,
    ) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.atlas, &frame.atlas) {
            return Err(Error::Invalid(
                "GPUI frame belongs to another painter".into(),
            ));
        }
        let scene = &frame.scene;
        if scene.batches.is_empty() {
            return Ok(());
        }
        let [width, height, _] = layer.desc().extent;
        let instances = bytemuck::cast_slice(scene.instances.as_slice());
        let curves = bytemuck::cast_slice(scene.curves.as_slice());
        let curves_offset = (instances.len() as u64).next_multiple_of(16);
        let roots_offset = (curves_offset + curves.len() as u64).next_multiple_of(ROOT_ALIGN);
        let stride = (size_of::<Root>() as u64).next_multiple_of(ROOT_ALIGN);
        let mut upload = self.device.gpu_malloc::<u8>(
            MemoryKind::Upload,
            roots_offset + scene.batches.len() as u64 * stride,
            "GPUI draw data",
        )?;
        let gpu = upload.gpu();
        let mapped = upload.as_mut_slice().expect("upload memory is mapped");
        mapped[..instances.len()].copy_from_slice(instances);
        let offset = curves_offset as usize;
        mapped[offset..offset + curves.len()].copy_from_slice(curves);

        if !scene.tiles.is_empty() {
            cmd.barrier(Stages::TRANSFER, Stages::PIXEL, Hazards::empty());
        }
        cmd.begin_rendering(&Rendering {
            extent: [width, height],
            colors: &[ColorAttachment {
                texture: layer,
                load: ColorLoad::Keep,
                store: true,
            }],
            depth: None,
        });
        for (i, batch) in scene.batches.iter().enumerate() {
            let texture = batch
                .page
                .map_or(TextureIndex::NONE, |page| self.textures[&page].index);
            let root = Root {
                instances: gpu.cast(),
                curves: gpu.offset(curves_offset).cast(),
                scale_x: 2.0 / width as f32,
                scale_y: 2.0 / height as f32,
                first: batch.first,
                kind: batch.kind,
                texture,
                sampler: self.sampler,
                _pad0: 0,
                _pad1: 0,
            };
            let offset = roots_offset + i as u64 * stride;
            let bytes = bytemuck::bytes_of(&root);
            let start = offset as usize;
            mapped[start..start + bytes.len()].copy_from_slice(bytes);
            cmd.draw(
                &self.pipeline,
                gpu.offset(offset).cast::<Root>(),
                4,
                batch.len,
            );
        }
        cmd.end_rendering();
        if !scene.tiles.is_empty() {
            let tiles: Vec<_> = scene.tiles.values().map(Arc::clone).collect();
            defer_after_submitted(&self.device, move || drop(tiles));
        }
        Ok(())
    }

    fn collect_atlas(&mut self, pages: &mut [Page]) {
        self.textures.retain(|&id, texture| {
            let page = &mut pages[id as usize];
            page.reclaim();
            if page.is_empty() {
                self.device.free_image_slot_after_submitted(texture.index.0);
                return false;
            }
            true
        });
    }

    fn upload_atlas(
        &mut self,
        pages: &[Page],
        tiles: &HashMap<(u32, u32), Arc<Tile>>,
    ) -> Result<(), Error> {
        let mut to_upload = Vec::new();
        let mut byte_len = 0u64;
        for (&(page, _), tile) in tiles {
            let bytes = tile.bytes.lock().unwrap();
            if let Some(bytes) = bytes.as_ref() {
                to_upload.push((page, tile, byte_len));
                byte_len += (bytes.len() as u64).next_multiple_of(16);
            }
        }
        if to_upload.is_empty() {
            return Ok(());
        }
        let mut new_pages = HashSet::new();
        for &(page, _, _) in &to_upload {
            if let std::collections::hash_map::Entry::Vacant(entry) = self.textures.entry(page) {
                entry.insert(PageTexture::new(&self.device, &pages[page as usize])?);
            }
        }
        let queue = self
            .device
            .queue(QueueKind::Render)
            .expect("a device has a Render queue");
        let mut list = queue.command_list()?;
        let mut upload =
            self.device
                .gpu_malloc::<u8>(MemoryKind::Upload, byte_len, "GPUI sprite pixels")?;
        let gpu = upload.gpu();
        let mapped = upload.as_mut_slice().expect("upload memory is mapped");
        list.barrier(
            Stages::PIXEL | Stages::TRANSFER,
            Stages::TRANSFER,
            Hazards::empty(),
        );
        for &(page, tile, offset) in &to_upload {
            let texture = &self.textures[&page].texture;
            let bytes = tile.bytes.lock().unwrap();
            let bytes = bytes.as_ref().expect("pending sprite");
            let at = offset as usize;
            mapped[at..at + bytes.len()].copy_from_slice(bytes);
            let [x, y, width, height] = tile.bounds;
            if new_pages.insert(page) && !self.textures[&page].initialized {
                list.discard(texture);
            }
            list.copy_to_texture(
                texture,
                &TextureRegion {
                    offset: [x, y, 0],
                    extent: [width, height, 1],
                    ..TextureRegion::whole_mip(texture.desc(), 0)
                },
                gpu.offset(offset),
            );
        }
        queue.submit(vec![list], &[])?;
        for (page, tile, _) in to_upload {
            self.textures.get_mut(&page).unwrap().initialized = true;
            *tile.bytes.lock().unwrap() = None;
        }
        Ok(())
    }
}

impl Drop for Painter {
    fn drop(&mut self) {
        for page in self.textures.values() {
            self.device.free_image_slot_after_submitted(page.index.0);
        }
    }
}
