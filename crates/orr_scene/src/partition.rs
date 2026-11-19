//! Cooked layouts ([`orr_pack::layout`]) resolved against the ready mounts, and their
//! partitions placed in a scene in one call each.
//!
//! A placed partition holds its pages' copies and its lights and decals, under ids the scene
//! hands out as it does any caller's. Unmounting the layout's mount empties its lights and
//! decals with its copies; the partition holds them until [`SceneWriter::remove_partition`].

use orr_gpu::TextureIndex;
use orr_math::{Quat, Vec3};
use orr_pack::AssetId;
use orr_pack::layout::{CameraSource, Group, LightSource};

use crate::decals::{Decal, DecalChannels, DecalId};
use crate::lights::{Cone, DirectionalLight, DiscProfile, Light, LightId};
use crate::mount::{MountId, Mounts, Resident};
use crate::scene::{InstanceFlags, InstancePageRef, InstanceRange, ModelId, SceneWriter};
use crate::scenes::insert;
use crate::{Error, PageId, Result};

/// A cooked layout of a mount, its assets resolved: pages in the layout's mount, models,
/// materials and textures in it and then in the other ready mounts.
#[derive(Clone, Debug)]
pub struct Layout {
    pub mount: MountId,
    pub groups: Vec<Group>,
    pub partitions: Vec<Partition>,
}

/// What a partition of a layout holds. [`SceneWriter::add_partition`] places its `pages`,
/// lights and decals; the consumer places its `posed_pages` and poses its `posed` models, as it
/// does any such page or model.
#[derive(Clone, Debug)]
pub struct Partition {
    /// Indices into [`Layout::groups`], ascending; empty for what no group holds.
    pub groups: Vec<u32>,
    /// The pages of its instance sources that place what they hold as stored, each source's in
    /// the cook's order of their cells.
    pub pages: Vec<PageId>,
    /// The pages that need what only the consumer has, as `pages` are ordered: those of a
    /// skinned model ([`SceneWriter::add_skinned_page`]), whose copies take palettes, of a
    /// model with a skeleton, whose parts take poses ([`InstancePageRef::poses`]), and those
    /// that hang from a joint ([`InstancePageRef::frame`]).
    pub posed_pages: Vec<PageId>,
    /// The models it holds that no page places, in scene space, which the consumer poses by
    /// their skeletons.
    pub posed: Vec<ModelId>,
    pub lights: Vec<Light>,
    pub decals: Vec<Decal>,
    pub cameras: Vec<CameraSource>,
    pub environment: Option<LayoutEnvironment>,
}

/// An environment map of a layout: `map` looked up at `rotation` of a world direction, times
/// `color`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutEnvironment {
    pub map: TextureIndex,
    /// The texture's `TextureHeader::primaries`.
    pub primaries: u8,
    pub rotation: Quat,
    pub color: Vec3,
}

/// A partition of a layout to place, and what its copies take as the copies of a page take
/// them ([`InstancePageRef`]).
#[derive(Clone, Copy, Debug)]
pub struct PartitionRef<'a> {
    pub layout: &'a Layout,
    /// Index into [`Layout::partitions`].
    pub partition: usize,
    pub flags: InstanceFlags,
    /// m from the scene focus within which the copies are traced.
    pub rt_range: f32,
    /// m from the scene focus within which the copies take their pose.
    pub animation_range: f32,
    /// Never read by orr.
    pub user_tag: u32,
}

/// A partition placed in a scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PartitionId(pub u32);

/// What a placed partition holds in its scene.
pub(crate) struct PlacedPartition {
    /// The layout's mount.
    pub(crate) mount: MountId,
    ranges: Vec<InstanceRange>,
    pub(crate) lights: Vec<LightId>,
    pub(crate) decals: Vec<DecalId>,
}

fn light(l: &LightSource) -> Light {
    match *l {
        // UsdLux and glTF state no profile over a distant light's disc.
        LightSource::Directional {
            direction_to_light,
            illuminance,
            angular_radius,
        } => Light::Directional(DirectionalLight {
            direction_to_light,
            illuminance,
            angular_radius,
            disc: DiscProfile::Uniform,
        }),
        LightSource::Sphere {
            position,
            radius,
            intensity,
            cone,
        } => Light::Sphere {
            position,
            radius,
            intensity,
            cone: cone.map(|c| Cone {
                direction: c.direction,
                inner: c.inner,
                outer: c.outer,
            }),
        },
        LightSource::Rect {
            center,
            edge_x,
            edge_y,
            luminance,
        } => Light::Rect {
            center,
            edge_x,
            edge_y,
            luminance,
        },
    }
}

/// What `get` finds of `asset` in the first of `search` that holds it.
fn find<T>(
    search: &[&Resident],
    m: MountId,
    asset: AssetId,
    get: impl Fn(&Resident) -> Option<T>,
) -> Result<T> {
    search
        .iter()
        .find_map(|r| get(r))
        .ok_or(Error::Missing { mount: m, asset })
}

impl Mounts {
    /// Layout `id` of ready mount `m`, resolved.
    pub(crate) fn layout(&self, m: MountId, id: AssetId) -> Result<Layout> {
        let info = self.lookup(m, id, |r| &r.layouts)?;
        let search = self.search_from(m)?;
        let home = search[0];
        let partitions = info
            .partitions
            .iter()
            .map(|p| {
                let pages = |sources: &[AssetId]| {
                    let mut pages = Vec::new();
                    for &s in sources {
                        let missing = Error::Missing { mount: m, asset: s };
                        pages.extend_from_slice(home.sources.get(&s).ok_or(missing)?);
                    }
                    Ok::<_, Error>(pages)
                };
                let posed = p
                    .posed
                    .iter()
                    .map(|&a| find(&search, m, a, |r| r.models.get(&a).copied()))
                    .collect::<Result<_>>()?;
                let decals = p
                    .decals
                    .iter()
                    .map(|d| {
                        let a = d.material;
                        let material = find(&search, m, a, |r| r.materials.get(&a).copied())?;
                        Ok(Decal {
                            center: d.center,
                            rotation: d.rotation,
                            half_extent: d.half_extent,
                            material,
                            channels: DecalChannels(d.channels),
                            opacity: d.opacity,
                            fade_angles: d.fade_angles,
                            depth_fade: d.depth_fade,
                            order: d.order,
                        })
                    })
                    .collect::<Result<_>>()?;
                let environment = p
                    .environment
                    .map(|e| {
                        let a = e.texture;
                        let (map, primaries) = find(&search, m, a, |r| {
                            Some((*r.textures.get(&a)?, r.texture_primaries[&a]))
                        })?;
                        Ok::<_, Error>(LayoutEnvironment {
                            map,
                            primaries,
                            rotation: e.rotation,
                            color: e.color,
                        })
                    })
                    .transpose()?;
                Ok(Partition {
                    groups: p.groups.clone(),
                    pages: pages(&p.sources)?,
                    posed_pages: pages(&p.posed_sources)?,
                    posed,
                    lights: p.lights.iter().map(light).collect(),
                    decals,
                    cameras: p.cameras.clone(),
                    environment,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Layout {
            mount: m,
            groups: info.groups.clone(),
            partitions,
        })
    }
}

impl SceneWriter<'_> {
    /// Places partition `r.partition` of `r.layout` in the scene from this frame on: its
    /// [`Partition::pages`], and its lights and decals under ids the scene hands out. Fails when
    /// a light, decal or copy finds no room, after removing what it placed, as
    /// [`Self::remove_partition`] does.
    pub fn add_partition(&mut self, r: &PartitionRef<'_>) -> Result<PartitionId> {
        let p = r
            .layout
            .partitions
            .get(r.partition)
            .ok_or(Error::UnknownPartition(r.partition))?;
        let mut placed = PlacedPartition {
            mount: r.layout.mount,
            ranges: Vec::new(),
            lights: Vec::new(),
            decals: Vec::new(),
        };
        if let Err(e) = self.place_partition(r, p, &mut placed) {
            self.unplace(&placed)?;
            return Err(e);
        }
        let cpu = &mut *self.scene.cpu.lock().unwrap();
        Ok(PartitionId(insert(&mut cpu.partitions, placed)))
    }

    fn place_partition(
        &mut self,
        r: &PartitionRef<'_>,
        p: &Partition,
        placed: &mut PlacedPartition,
    ) -> Result<()> {
        for &page in &p.pages {
            placed.ranges.push(self.add_instance_page(&InstancePageRef {
                page,
                frame: None,
                poses: &[],
                material_sets: &[],
                flags: r.flags,
                rt_range: r.rt_range,
                animation_range: r.animation_range,
                user_tag: r.user_tag,
            })?);
        }
        for l in &p.lights {
            placed.lights.push(self.add_light(l)?);
        }
        for d in &p.decals {
            placed.decals.push(self.add_decal(d)?);
        }
        Ok(())
    }

    /// Removes what `placed` holds, all of it even where one removal fails, as the partition's
    /// record is gone: the first failure.
    fn unplace(&mut self, placed: &PlacedPartition) -> Result<()> {
        for &range in &placed.ranges {
            self.remove(range);
        }
        let lights = placed.lights.iter().map(|&id| self.remove_light(id));
        let lights: Vec<Result<()>> = lights.collect();
        let decals = placed.decals.iter().map(|&id| self.remove_decal(id));
        let decals: Vec<Result<()>> = decals.collect();
        lights.into_iter().chain(decals).collect()
    }

    /// Removes placed partition `id` from this frame on: its copies, lights and decals, whose
    /// slots and ids are handed out again once this frame has completed.
    pub fn remove_partition(&mut self, id: PartitionId) -> Result<()> {
        let placed = {
            let cpu = &mut *self.scene.cpu.lock().unwrap();
            let slot = cpu.partitions.get_mut(id.0 as usize);
            slot.and_then(Option::take)
                .ok_or(Error::UnknownPlacedPartition(id))?
        };
        self.unplace(&placed)
    }
}
