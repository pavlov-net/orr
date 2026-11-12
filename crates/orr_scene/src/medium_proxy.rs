//! Medium proxies as the API takes them, their table records, resident volumes and the one
//! AABB BLAS every proxy instance traces.

use std::sync::Arc;

use bytemuck::Zeroable;
use orr_gpu::{
    Allocation, ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicKind, CommandList,
    Device, GpuAddress, GpuPtr, Hazards, MemoryKind, RtasFlags, Stages,
};
use orr_math::{DVec3, Mat3, Quat, Vec3};
use orr_pack::Aabb;
use orr_pack::volume::VolumeGpu;

use crate::MediumProxyGpu;
use crate::Result;
use crate::world::WorldPos;

/// `medium_proxy.slang` constants.
const PROXY_LIVE: u32 = 1;

/// A medium proxy of a scene: its entry in the scene's proxy table, which holds every entry
/// up to the highest id set, as the PTLAS holds an instance for each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MediumProxyId(pub u32);

/// A resident volume (`orr_pack::volume`), valid until its pack is unmounted.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeRef {
    pub volume: GpuPtr<VolumeGpu>,
    /// Every sample outside reads each grid's background, in the volume's space (m).
    pub bounds: Aabb,
    /// Grid names, by grid index.
    pub grids: Arc<[String]>,
}

impl VolumeRef {
    /// The index of the grid named `name`.
    pub fn grid(&self, name: &str) -> Option<u32> {
        self.grids.iter().position(|g| g == name).map(|i| i as u32)
    }

    /// Grid `grid` as a proxy's density over the volume's bounds. Panics if the volume has no
    /// grid `grid`.
    pub fn density(&self, grid: u32) -> ProxyDensity {
        assert!(
            (grid as usize) < self.grids.len(),
            "grid {grid} of a volume of {} grids",
            self.grids.len()
        );
        ProxyDensity::Grid {
            volume: self.volume,
            grid,
            bounds: self.bounds,
        }
    }
}

/// What fills a medium proxy's box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProxyDensity {
    /// Density 1 throughout a box of half extents `half_extent` (m) around the origin of the
    /// volume space.
    Uniform { half_extent: Vec3 },
    /// Grid `grid` of the resident volume at `volume`, over the box `bounds` of its space:
    /// a cooked volume ([`VolumeRef::density`]) or a consumer's own in the same layout. A grid
    /// at or past the record's `grid_count` reads as density 0, an empty box.
    Grid {
        volume: GpuPtr<VolumeGpu>,
        grid: u32,
        bounds: Aabb,
    },
}

/// A box of participating medium. Shadow rays take its transmittance, marching its density
/// along their span inside the box, and media apply marches it along camera rays; path and
/// cache rays do not see it. Its volume space sits in the world at `position`, turned by `rotation` and scaled by `scale`.
/// Coefficients are linear Rec.2020.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediumProxy {
    pub position: DVec3,
    pub rotation: Quat,
    /// Above 0.
    pub scale: f32,
    pub density: ProxyDensity,
    /// Extinction per unit density, 1/m.
    pub extinction: Vec3,
    /// Scattering over extinction.
    pub albedo: Vec3,
    /// Henyey-Greenstein anisotropy.
    pub anisotropy: f32,
}

/// The table record of `proxy`.
pub(crate) fn proxy_gpu(proxy: &MediumProxy, cell_log2: u32) -> MediumProxyGpu {
    let (center, half, volume, grid) = match proxy.density {
        ProxyDensity::Uniform { half_extent } => (Vec3::ZERO, half_extent, GpuPtr::NULL, 0),
        ProxyDensity::Grid {
            volume,
            grid,
            bounds,
        } => {
            let (lo, hi) = (Vec3::from(bounds.min), Vec3::from(bounds.max));
            ((lo + hi) * 0.5, (hi - lo) * 0.5, volume, grid)
        }
    };
    let half = half.max(Vec3::splat(1e-6));
    let r = Mat3::from_quat(proxy.rotation.normalize());
    let world = proxy.position + (r * (center * proxy.scale)).as_dvec3();
    let w = WorldPos::new(world, cell_log2);
    let axes = r * Mat3::from_diagonal(half * proxy.scale);
    MediumProxyGpu {
        cell: w.cell.to_array(),
        flags: PROXY_LIVE,
        offset: w.offset.to_array(),
        grid,
        axis_x: axes.x_axis.to_array(),
        anisotropy: proxy.anisotropy,
        axis_y: axes.y_axis.to_array(),
        axis_z: axes.z_axis.to_array(),
        box_center: center.to_array(),
        box_half: half.to_array(),
        extinction: proxy.extinction.to_array(),
        albedo: proxy.albedo.to_array(),
        volume,
        ..Zeroable::zeroed()
    }
}

/// The BLAS of one AABB over the cube [-1, 1]^3, which every proxy instance traces.
pub(crate) struct ProxyBlas {
    blas: ClassicAs,
    built: bool,
    _storage: Allocation<u8>,
    _boxes: Allocation<[f32; 6]>,
    scratch: Option<Allocation<u8>>,
    desc: ClassicDesc,
}

impl ProxyBlas {
    pub(crate) fn new(device: &Device) -> Result<Self> {
        let mut boxes = device.gpu_malloc::<[f32; 6]>(MemoryKind::Upload, 1, "proxy box")?;
        boxes.as_mut_slice().expect("upload is mapped")[0] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let desc = ClassicDesc {
            kind: ClassicKind::Blas,
            flags: RtasFlags::FAST_TRACE,
            geometry: ClassicGeometry::Aabbs {
                boxes: boxes.gpu(),
                count: 1,
                opaque: false,
            },
        };
        let sizes = device.classic_sizes(&desc);
        let storage =
            device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "proxy BLAS")?;
        let blas = device.create_classic_as(ClassicKind::Blas, sizes.result_max, &storage, 0)?;
        let scratch = device.gpu_malloc::<u8>(
            MemoryKind::DeviceLocal,
            sizes.scratch.max(256),
            "proxy BLAS scratch",
        )?;
        Ok(Self {
            blas,
            built: false,
            _storage: storage,
            _boxes: boxes,
            scratch: Some(scratch),
            desc,
        })
    }

    /// Records the build and its barrier the first time; the BLAS address after it. A later frame frees the
    /// build's scratch, which waits for the submitted build.
    pub(crate) fn record(&mut self, list: &mut CommandList<'_>) -> GpuAddress {
        if self.built {
            self.scratch = None;
        } else {
            let scratch = self.scratch.as_ref().expect("held until built");
            list.build_classic(&ClassicBuild {
                desc: &self.desc,
                dst: &self.blas,
                scratch: scratch.gpu().address(),
            });
            list.barrier(Stages::AS_BUILD, Stages::AS_BUILD, Hazards::ACCEL_STRUCT);
            self.built = true;
        }
        self.blas.address()
    }
}
