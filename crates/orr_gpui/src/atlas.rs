use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use anyhow::{Result, ensure};
use etagere::{AllocId, AtlasAllocator, size2};
use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds,
    DevicePixels, PlatformAtlas, Size, TileId, point,
};
use orr_gpu::Format;

const PAGE_SIZE: i32 = 1024;

pub(crate) struct Tile {
    pub(crate) bounds: [u32; 4],
    pub(crate) bytes: Mutex<Option<Box<[u8]>>>,
}

pub(crate) struct Page {
    pub(crate) id: AtlasTextureId,
    pub(crate) extent: [u32; 2],
    pub(crate) format: Format,
    allocator: AtlasAllocator,
    tiles: HashMap<u32, (AllocId, Arc<Tile>)>,
    retired: Vec<(AllocId, Weak<Tile>)>,
}

impl Page {
    pub(crate) fn reclaim(&mut self) {
        self.retired.retain(|(id, tile)| {
            let in_use = tile.strong_count() != 0;
            if !in_use {
                self.allocator.deallocate(*id);
            }
            in_use
        });
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tiles.is_empty() && self.retired.is_empty()
    }

    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<etagere::Allocation> {
        self.reclaim();
        self.allocator.allocate(size2(size.width.0, size.height.0))
    }

    pub(crate) fn tile(&self, tile: AtlasTile) -> Option<Arc<Tile>> {
        self.tiles
            .get(&tile.tile_id.0)
            .map(|(_, pixels)| Arc::clone(pixels))
    }
}

pub(crate) struct Pages {
    pub(crate) pages: Vec<Page>,
    next_tile: u32,
    max_extent: i32,
}

impl AtlasBackend for Pages {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        let format = match kind {
            AtlasTextureKind::Monochrome => Format::R8Unorm,
            AtlasTextureKind::Polychrome => Format::Bgra8Unorm,
            AtlasTextureKind::Subpixel => {
                anyhow::bail!("subpixel text requires the scene background")
            }
        };
        let [width, height] = [size.width.0, size.height.0];
        ensure!(
            width > 0 && height > 0 && width <= self.max_extent && height <= self.max_extent,
            "sprite extent {width}x{height} outside GPU limits"
        );
        ensure!(
            bytes.len() as u64 == width as u64 * height as u64 * u64::from(format.block_bytes()),
            "sprite bitmap size does not match its extent and format"
        );
        let tile_id = self.next_tile;
        self.next_tile = tile_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("atlas tile IDs exhausted"))?;
        let available = self
            .pages
            .iter_mut()
            .filter(|page| page.id.kind == kind)
            .find_map(|page| page.allocate(size).map(|rect| (page, rect)));
        let (page, allocation) = match available {
            Some(entry) => entry,
            None => {
                let base = PAGE_SIZE.min(self.max_extent);
                let extent = [width.max(base), height.max(base)];
                let index = u32::try_from(self.pages.len())?;
                let mut page = Page {
                    id: AtlasTextureId { index, kind },
                    extent: extent.map(|n| n as u32),
                    format,
                    allocator: AtlasAllocator::new(size2(extent[0], extent[1])),
                    tiles: HashMap::new(),
                    retired: Vec::new(),
                };
                let allocation = page.allocate(size).expect("empty page fits the sprite");
                self.pages.push(page);
                (self.pages.last_mut().unwrap(), allocation)
            }
        };
        let origin = allocation.rectangle.min;
        page.tiles.insert(
            tile_id,
            (
                allocation.id,
                Arc::new(Tile {
                    bounds: [
                        origin.x as u32,
                        origin.y as u32,
                        width as u32,
                        height as u32,
                    ],
                    bytes: Mutex::new(Some(bytes.into())),
                }),
            ),
        );
        Ok(AtlasTile {
            texture_id: page.id,
            tile_id: TileId(tile_id),
            padding: 0,
            bounds: Bounds {
                origin: point(DevicePixels(origin.x), DevicePixels(origin.y)),
                size,
            },
        })
    }

    fn remove(&mut self, tile: AtlasTile) {
        let page = &mut self.pages[tile.texture_id.index as usize];
        let (id, pixels) = page
            .tiles
            .remove(&tile.tile_id.0)
            .expect("GPUI tile is live");
        page.retired.push((id, Arc::downgrade(&pixels)));
    }
}

pub(crate) struct Atlas(pub(crate) Mutex<AtlasState<Pages>>);

impl Atlas {
    pub(crate) fn new(max_extent: u32) -> Self {
        Self(Mutex::new(AtlasState::new(Pages {
            pages: Vec::new(),
            next_tile: 0,
            max_extent: i32::try_from(max_extent).unwrap_or(i32::MAX),
        })))
    }
}

impl PlatformAtlas for Atlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        self.0.lock().unwrap().get_or_insert_with(key, build)
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().unwrap().remove(key);
    }
}
