//! Classic BLAS and TLAS over caller memory, and opacity micromaps, which Metal lacks.
//!
//! A classic acceleration structure lives in a DeviceLocal block's placement heap, and its
//! address is its resource ID tagged in bit 63 ([`CLASSIC_TAG`]); a cluster BLAS or PTLAS
//! address is memory whose header holds the ID (`cluster.rs`). `abi.slang`'s `Tlas.get()`
//! and the instance conversion below take either. TLAS builds first rewrite the instance
//! records, laid out as the Vulkan backend's, into Metal's indirect instance descriptors.

use std::sync::{Arc, Mutex, OnceLock};

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString};
use objc2_metal::{
    MTL4AccelerationStructureBoundingBoxGeometryDescriptor, MTL4AccelerationStructureDescriptor,
    MTL4AccelerationStructureGeometryDescriptor,
    MTL4AccelerationStructureTriangleGeometryDescriptor, MTL4BufferRange,
    MTL4ComputeCommandEncoder, MTL4InstanceAccelerationStructureDescriptor,
    MTL4LibraryFunctionDescriptor, MTL4PrimitiveAccelerationStructureDescriptor,
    MTLAccelerationStructure, MTLAccelerationStructureDescriptor,
    MTLAccelerationStructureInstanceDescriptorType, MTLAccelerationStructureUsage,
    MTLAttributeFormat, MTLComputePipelineState, MTLDevice, MTLHeap, MTLIndexType, MTLResource,
};

use super::command::CommandList;
use super::device::{Device, Shared, metal_error};
use super::memory::{Allocation, MemoryKind, malloc};
use super::{Obj, lock, not_implemented};
use crate::{Error, GpuAddress, GpuPtr, Hazards, RtasFlags, RtasSizes, Stages, Tlas};

/// Bytes of Metal's `MTLIndirectAccelerationStructureInstanceDescriptor`.
const METAL_INSTANCE_BYTES: u64 = 72;

/// Rewrites [`ClassicInstance`] records into Metal's indirect instance descriptors. The flag
/// bits mean the same in both: culling off, counter-clockwise front, opaque, non-opaque.
const INSTANCE_CONVERT: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Orr { float t[12]; uint custom_mask; uint sbt_flags; ulong blas; };
struct Mtl { packed_float3 m[4]; uint options; uint mask; uint ift_offset; uint user_id; ulong as_id; };
ulong resolve(ulong a, thread bool& cluster) {
    cluster = false;
    if (a == 0) return 0;
    if ((a >> 63) != 0) return a & ~(1ul << 63);
    uint kind = *reinterpret_cast<device const uint*>(a);
    cluster = kind == 3 || kind == 4;
    return reinterpret_cast<device const ulong*>(a)[1];
}
struct Args { device const Orr* src; device Mtl* dst; uint count; uint pad; };
kernel void convert(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    Orr v = a.src[i];
    Mtl o;
    for (int c = 0; c < 3; c++) o.m[c] = packed_float3(v.t[c], v.t[4 + c], v.t[8 + c]);
    o.m[3] = packed_float3(v.t[3], v.t[7], v.t[11]);
    o.options = (v.sbt_flags >> 24) & 0xf;
    o.mask = v.blas == 0 ? 0 : v.custom_mask >> 24;
    o.ift_offset = 0;
    bool cluster;
    o.as_id = resolve(v.blas, cluster);
    o.user_id = (v.custom_mask & 0xffffff) | (cluster ? 1u << 31 : 0u);
    a.dst[i] = o;
}
// The per-triangle data of a classic triangle BLAS, as cluster BLAS give theirs (cluster.msl):
// no cluster, geometry 0, the triangle, then its corners for hits to fetch.
struct DataArgs { device const float* positions; device const uint* indices; device uint* out; uint count; uint pad; };
kernel void classic_data(constant DataArgs& a [[buffer(0)]], uint t [[thread_position_in_grid]]) {
    if (t >= a.count) return;
    device uint* o = a.out + t * 12;
    o[0] = 0xffffffffu;
    o[1] = 0;
    o[2] = t;
    for (uint c = 0; c < 3; c++) {
        uint v = a.indices ? a.indices[t * 3 + c] : t * 3 + c;
        for (uint k = 0; k < 3; k++) o[3 + c * 3 + k] = as_type<uint>(a.positions[v * 3 + k]);
    }
}
"#;

/// Bytes of a triangle's primitive data, as `cluster.rs` lays it out.
const TRIANGLE_DATA_BYTES: u64 = 48;

/// The arguments of `classic_data`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DataArgs {
    positions: u64,
    indices: u64,
    out: u64,
    count: u32,
    pad: u32,
}

/// The arguments of [`INSTANCE_CONVERT`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConvertArgs {
    src: u64,
    dst: u64,
    count: u32,
    pad: u32,
}

bitflags::bitflags! {
    /// Per-instance flags of a [`ClassicInstance`].
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct InstanceFlags: u8 {
        /// Both triangle faces hit.
        const CULL_DISABLE = 1 << 0;
        /// Counter-clockwise triangles face front.
        const FLIP_FACING = 1 << 1;
        const FORCE_OPAQUE = 1 << 2;
        const FORCE_NO_OPAQUE = 1 << 3;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ClassicKind {
    Blas,
    Tlas,
}

/// What a classic acceleration structure is built from.
#[derive(Clone, Copy, Debug)]
pub enum ClassicGeometry {
    /// One triangle geometry; `indices` null for unindexed triangles.
    Triangles {
        positions: GpuPtr<[f32; 3]>,
        vertex_count: u32,
        indices: GpuPtr<[u32; 3]>,
        triangle_count: u32,
        opaque: bool,
    },
    /// One linear-swept-sphere geometry; needs `Caps::lss`, which Metal lacks.
    Lss {
        positions: GpuPtr<[f32; 3]>,
        radii: GpuPtr<f32>,
        indices: GpuPtr<u32>,
        segment_count: u32,
        indexing: LssIndexing,
        end_caps: LssEndCaps,
        opaque: bool,
    },
    /// `count` axis-aligned boxes, each `[min x, min y, min z, max x, max y, max z]`.
    Aabbs {
        boxes: GpuPtr<[f32; 6]>,
        count: u32,
        opaque: bool,
    },
    /// `capacity` [`ClassicInstance`] records at `records`; a record with BLAS address 0 is
    /// inactive.
    Instances { records: GpuAddress, capacity: u32 },
}

/// How [`ClassicGeometry::Lss`] indices name each segment's vertices.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LssIndexing {
    List,
    Successive,
}

/// Which [`ClassicGeometry::Lss`] segment ends are closed by a sphere.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LssEndCaps {
    None,
    Chained,
}

#[derive(Clone, Copy, Debug)]
pub struct ClassicDesc {
    pub kind: ClassicKind,
    pub flags: RtasFlags,
    pub geometry: ClassicGeometry,
}

/// One instance of a classic TLAS, laid out as the Vulkan backend's.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClassicInstance {
    /// Row-major 3x4 object-to-world transform.
    pub transform: [f32; 12],
    /// Custom index in bits 0-23, visibility mask in bits 24-31.
    pub custom_index_and_mask: u32,
    /// Shader-binding-table offset in bits 0-23 (always 0 in orr), flags in bits 24-31.
    pub sbt_offset_and_flags: u32,
    /// The BLAS address; 0 makes the instance inactive.
    pub blas: u64,
}

impl ClassicInstance {
    pub fn new(
        transform: [[f32; 4]; 3],
        custom_index: u32,
        mask: u8,
        flags: InstanceFlags,
        blas: GpuAddress,
    ) -> Self {
        assert!(
            custom_index < 1 << 24,
            "custom index {custom_index} needs more than 24 bits"
        );
        Self {
            transform: bytemuck::cast(transform),
            custom_index_and_mask: custom_index | u32::from(mask) << 24,
            sbt_offset_and_flags: u32::from(flags.bits()) << 24,
            blas: blas.0,
        }
    }
}

/// A classic acceleration structure in caller memory. Dropping it releases it once
/// everything submitted before the drop has completed; its memory stays the caller's.
pub struct ClassicAs {
    raw: Obj<ProtocolObject<dyn MTLAccelerationStructure>>,
    /// The resource ID.
    id: u64,
    kind: ClassicKind,
    shared: Arc<Shared>,
}

/// Bit 63 of a classic acceleration structure's address, which is its resource ID: no
/// device address sets it.
pub(super) const CLASSIC_TAG: u64 = 1 << 63;

impl ClassicAs {
    /// The address instances and traces name it by: its resource ID, tagged.
    pub fn address(&self) -> GpuAddress {
        GpuAddress(self.id | CLASSIC_TAG)
    }

    /// The TLAS kernels trace.
    pub fn tlas(&self) -> Tlas {
        assert_eq!(self.kind, ClassicKind::Tlas, "tlas() of a BLAS");
        Tlas(self.id | CLASSIC_TAG)
    }
}

impl Drop for ClassicAs {
    fn drop(&mut self) {
        let raw = self.raw.clone();
        self.shared.defer(Box::new(move || drop(raw)));
    }
}

/// One classic build: `desc` into `dst`, with scratch at `scratch`.
#[derive(Clone, Copy)]
pub struct ClassicBuild<'a> {
    pub desc: &'a ClassicDesc,
    pub dst: &'a ClassicAs,
    pub scratch: GpuAddress,
}

/// The opacity states a micromap stores per micro-triangle.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OmmFormat {
    TwoState,
    FourState,
}

/// How many micromap triangles of one subdivision level and format a build holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MicromapUsage {
    pub count: u32,
    pub level: u32,
    pub format: OmmFormat,
}

/// One micromap triangle: its states start at byte `data_offset` of the build's data.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MicromapTriangle {
    pub data_offset: u32,
    pub level: u16,
    pub format: u16,
}

impl MicromapTriangle {
    pub fn new(data_offset: u32, level: u16, format: OmmFormat) -> Self {
        Self {
            data_offset,
            level,
            // `VkOpacityMicromapFormatKHR`'s values.
            format: match format {
                OmmFormat::TwoState => 1,
                OmmFormat::FourState => 2,
            },
        }
    }
}

/// An opacity micromap build: `triangles` describe the states packed in `data`.
#[derive(Clone, Copy, Debug)]
pub struct MicromapDesc<'a> {
    pub flags: RtasFlags,
    pub usage: &'a [MicromapUsage],
    pub data: GpuAddress,
    pub triangles: GpuPtr<MicromapTriangle>,
}

/// An opacity micromap; Metal has none, so [`Device::create_micromap`] fails.
pub struct Micromap {
    never: std::convert::Infallible,
}

impl Micromap {
    pub fn address(&self) -> GpuAddress {
        match self.never {}
    }
}

impl Device {
    /// The result and scratch sizes of building `d`. The result size covers the heap
    /// alignment, so a block of it holds the structure at offset 0.
    pub fn classic_sizes(&self, d: &ClassicDesc) -> RtasSizes {
        let desc = descriptor(d, None, Some(0));
        let base: &MTLAccelerationStructureDescriptor = &desc;
        let sizes = self
            .shared
            .device
            .accelerationStructureSizesWithDescriptor(base);
        // SAFETY: a size Metal just reported.
        let placed = unsafe {
            self.shared
                .device
                .heapAccelerationStructureSizeAndAlignWithSize(sizes.accelerationStructureSize)
        };
        RtasSizes {
            result_max: placed.size as u64,
            scratch: sizes.buildScratchBufferSize as u64,
        }
    }

    /// A classic acceleration structure of `size` bytes at `offset` of `block`, a
    /// DeviceLocal allocation.
    pub fn create_classic_as(
        &self,
        kind: ClassicKind,
        size: u64,
        block: &Allocation<u8>,
        offset: u64,
    ) -> Result<ClassicAs, Error> {
        let heap = block.raw().heap.as_ref().ok_or_else(|| {
            Error::Invalid("acceleration structures go in DeviceLocal blocks".to_owned())
        })?;
        // SAFETY: the range lies within the block's heap, which outlives the structure.
        let raw =
            unsafe { heap.newAccelerationStructureWithSize_offset(size as usize, offset as usize) }
                .ok_or_else(|| Error::Metal {
                    call: "newAccelerationStructureWithSize",
                    message: format!("{size} bytes at {offset}"),
                })?;
        raw.setLabel(Some(&NSString::from_str(match kind {
            ClassicKind::Blas => "classic blas",
            ClassicKind::Tlas => "classic tlas",
        })));
        let id = super::heap::id_bits(raw.gpuResourceID());
        Ok(ClassicAs {
            raw: Obj(raw),
            id,
            kind,
            shared: Arc::clone(&self.shared),
        })
    }

    pub fn micromap_sizes(&self, d: &MicromapDesc<'_>) -> RtasSizes {
        let _ = d;
        not_implemented("Device::micromap_sizes")
    }

    pub fn create_micromap(
        &self,
        size: u64,
        block: &Allocation<u8>,
        offset: u64,
    ) -> Result<Micromap, Error> {
        let _ = (size, block, offset);
        Err(Error::NotImplemented("Device::create_micromap"))
    }
}

impl CommandList<'_> {
    /// Builds `b.desc` into `b.dst`, then makes it visible to traces, builds and kernels.
    pub fn build_classic(&mut self, b: &ClassicBuild<'_>) {
        let instances = match b.desc.geometry {
            ClassicGeometry::Instances { records, capacity } => {
                Some(self.convert_instances(records, capacity))
            }
            _ => None,
        };
        let data = match b.desc.geometry {
            ClassicGeometry::Triangles {
                positions,
                indices,
                triangle_count,
                ..
            } => {
                Some(self.triangle_data(positions.address().0, indices.address().0, triangle_count))
            }
            _ => None,
        };
        let desc = descriptor(b.desc, instances, data);
        let e = self.compute();
        // SAFETY: the inputs, destination and scratch are live device memory, as on every
        // backend.
        unsafe {
            e.buildAccelerationStructure_descriptor_scratchBuffer(
                &b.dst.raw,
                &desc,
                MTL4BufferRange {
                    bufferAddress: b.scratch.0,
                    length: u64::MAX,
                },
            );
        }
        self.keep.push(Box::new(b.dst.raw.clone()));
        self.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }

    /// Writes the per-triangle data of a classic triangle BLAS, in memory the list keeps;
    /// returns its address. Hits fetch their corners from it, since Metal cannot.
    fn triangle_data(&mut self, positions: u64, indices: u64, count: u32) -> u64 {
        let shared = Arc::clone(&self.queue.shared);
        let dst = malloc::<u8>(
            &shared,
            MemoryKind::DeviceLocal,
            TRIANGLE_DATA_BYTES * u64::from(count.max(1)),
            "triangle data",
        )
        .expect("triangle data");
        let address = dst.gpu().address().0;
        let state = shared
            .accel_kernel("classic_data")
            .expect("the triangle data kernel");
        let args = DataArgs {
            positions,
            indices,
            out: address,
            count,
            pad: 0,
        };
        self.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.dispatch_state(
            &state,
            bytemuck::bytes_of(&args),
            [count.div_ceil(64).max(1), 1, 1],
            [64, 1, 1],
        );
        self.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
        self.keep.push(Box::new(dst));
        address
    }

    /// Rewrites `capacity` instance records at `records` into Metal's descriptors, in
    /// memory the list keeps; returns their address.
    fn convert_instances(&mut self, records: GpuAddress, capacity: u32) -> u64 {
        let shared = Arc::clone(&self.queue.shared);
        let dst = malloc::<u8>(
            &shared,
            MemoryKind::DeviceLocal,
            METAL_INSTANCE_BYTES * u64::from(capacity.max(1)),
            "metal instances",
        )
        .expect("instance descriptors");
        let address = dst.gpu().address().0;
        let state = shared
            .instance_convert()
            .expect("the instance conversion kernel");
        let args = ConvertArgs {
            src: records.0,
            dst: address,
            count: capacity,
            pad: 0,
        };
        self.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.dispatch_state(
            &state,
            bytemuck::bytes_of(&args),
            [capacity.div_ceil(64).max(1), 1, 1],
            [64, 1, 1],
        );
        self.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
        self.keep.push(Box::new(dst));
        address
    }

    /// Writes the bytes `a` compacts into to `dst`, once its build has completed on the GPU.
    pub fn classic_compacted_size(&mut self, a: &ClassicAs, dst: GpuPtr<u64>) {
        self.barrier(Stages::AS_BUILD, Stages::AS_BUILD, Hazards::empty());
        let e = self.compute();
        // SAFETY: the structure is live and `dst` is eight bytes of the caller's memory.
        unsafe {
            e.writeCompactedAccelerationStructureSize_toBuffer(
                &a.raw,
                MTL4BufferRange {
                    bufferAddress: dst.address().0,
                    length: 8,
                },
            );
        }
        self.keep.push(Box::new(a.raw.clone()));
        self.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::empty());
    }

    /// Copies `src` into `dst`, created with at least its compacted size.
    pub fn compact_classic(&mut self, dst: &ClassicAs, src: &ClassicAs) {
        self.barrier(Stages::AS_BUILD, Stages::AS_BUILD, Hazards::empty());
        let e = self.compute();
        // SAFETY: both structures are live until the submit completes.
        unsafe {
            e.copyAndCompactAccelerationStructure_toAccelerationStructure(&src.raw, &dst.raw)
        };
        self.keep.push(Box::new(src.raw.clone()));
        self.keep.push(Box::new(dst.raw.clone()));
        self.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }

    pub fn build_micromap(&mut self, d: &MicromapDesc<'_>, dst: &Micromap, scratch: GpuAddress) {
        let _ = (d, scratch);
        match dst.never {}
    }
}

impl Shared {
    /// The compute pipeline of [`INSTANCE_CONVERT`], compiled once.
    fn instance_convert(&self) -> Result<Obj<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
        self.accel_kernel("convert")
    }

    /// The compute pipeline of kernel `name` of [`INSTANCE_CONVERT`], compiled once.
    fn accel_kernel(
        &self,
        name: &'static str,
    ) -> Result<Obj<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
        type States = Mutex<
            Vec<(
                &'static str,
                Obj<ProtocolObject<dyn MTLComputePipelineState>>,
            )>,
        >;
        static STATES: OnceLock<States> = OnceLock::new();
        let states = STATES.get_or_init(Mutex::default);
        if let Some((_, s)) = lock(states).iter().find(|(n, _)| *n == name) {
            return Ok(s.clone());
        }
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(INSTANCE_CONVERT), None)
            .map_err(|e| metal_error("newLibraryWithSource", &e))?;
        let f = MTL4LibraryFunctionDescriptor::new();
        f.setName(Some(&NSString::from_str(name)));
        f.setLibrary(Some(&library));
        let state = self.compute_state(&f, [64, 1, 1], name)?;
        lock(states).push((name, state.clone()));
        Ok(state)
    }
}

/// The Metal build descriptor of `d`; `instances` is the address of the converted instance
/// descriptors of a TLAS build, `None` when only sizing.
fn descriptor(
    d: &ClassicDesc,
    instances: Option<u64>,
    data: Option<u64>,
) -> Retained<MTL4AccelerationStructureDescriptor> {
    let range = |address: u64, bytes: u64| MTL4BufferRange {
        bufferAddress: address,
        length: bytes,
    };
    let desc: Retained<MTL4AccelerationStructureDescriptor> = match d.geometry {
        ClassicGeometry::Triangles {
            positions,
            vertex_count,
            indices,
            triangle_count,
            opaque,
        } => {
            let g = MTL4AccelerationStructureTriangleGeometryDescriptor::new();
            // SAFETY: addresses and counts are the caller's, as on every backend; Metal reads
            // them only when the build runs.
            unsafe {
                g.setVertexBuffer(range(positions.address().0, 12 * u64::from(vertex_count)));
                g.setVertexStride(12);
                g.setTriangleCount(triangle_count as usize);
                if indices != GpuPtr::NULL {
                    g.setIndexBuffer(range(indices.address().0, 12 * u64::from(triangle_count)));
                }
            }
            if let Some(data) = data {
                // SAFETY: `data` holds a record per triangle once its kernel has run.
                unsafe {
                    g.setPrimitiveDataBuffer(range(
                        data,
                        TRIANGLE_DATA_BYTES * u64::from(triangle_count),
                    ));
                    g.setPrimitiveDataElementSize(TRIANGLE_DATA_BYTES as usize);
                    g.setPrimitiveDataStride(TRIANGLE_DATA_BYTES as usize);
                }
            }
            g.setVertexFormat(MTLAttributeFormat::Float3);
            g.setIndexType(MTLIndexType::UInt32);
            g.setOpaque(opaque);
            primitive(Retained::into_super(g))
        }
        ClassicGeometry::Aabbs {
            boxes,
            count,
            opaque,
        } => {
            let g = MTL4AccelerationStructureBoundingBoxGeometryDescriptor::new();
            g.setBoundingBoxBuffer(range(boxes.address().0, 24 * u64::from(count)));
            // SAFETY: as above.
            unsafe {
                g.setBoundingBoxStride(24);
                g.setBoundingBoxCount(count as usize);
            }
            g.setOpaque(opaque);
            primitive(Retained::into_super(g))
        }
        ClassicGeometry::Lss { .. } => not_implemented("linear swept spheres"),
        ClassicGeometry::Instances { capacity, .. } => {
            let t = MTL4InstanceAccelerationStructureDescriptor::new();
            // SAFETY: the converted descriptors hold `capacity` records.
            unsafe {
                t.setInstanceDescriptorBuffer(range(
                    instances.unwrap_or(0),
                    METAL_INSTANCE_BYTES * u64::from(capacity),
                ));
                t.setInstanceDescriptorStride(METAL_INSTANCE_BYTES as usize);
                t.setInstanceCount(capacity as usize);
            }
            t.setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::Indirect);
            Retained::into_super(t)
        }
    };
    if d.flags.contains(RtasFlags::FAST_BUILD) {
        desc.setUsage(MTLAccelerationStructureUsage::PreferFastBuild);
    }
    desc
}

fn primitive(
    g: Retained<MTL4AccelerationStructureGeometryDescriptor>,
) -> Retained<MTL4AccelerationStructureDescriptor> {
    let p = MTL4PrimitiveAccelerationStructureDescriptor::new();
    p.setGeometryDescriptors(Some(&NSArray::from_retained_slice(&[g])));
    Retained::into_super(p)
}

/// [`Device::empty_tlas`]'s TLAS and its memory.
pub(super) struct EmptyTlas {
    pub(super) tlas: ClassicAs,
    _storage: Allocation<u8>,
}

/// Builds a TLAS with one inactive instance on the render queue and waits for it.
pub(super) fn build_empty_tlas(device: &Device) -> Result<EmptyTlas, Error> {
    let mut instances =
        device.gpu_malloc::<ClassicInstance>(MemoryKind::Upload, 1, "orr empty TLAS instance")?;
    instances.as_mut_slice().unwrap()[0] = ClassicInstance::zeroed();
    let desc = ClassicDesc {
        kind: ClassicKind::Tlas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Instances {
            records: instances.gpu().address(),
            capacity: 1,
        },
    };
    let sizes = device.classic_sizes(&desc);
    let storage =
        device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "orr empty TLAS")?;
    let scratch = device.gpu_malloc::<u8>(
        MemoryKind::DeviceLocal,
        sizes.scratch.max(1),
        "orr empty TLAS scratch",
    )?;
    let tlas = device.create_classic_as(ClassicKind::Tlas, sizes.result_max, &storage, 0)?;
    let queue = device
        .queue(crate::QueueKind::Render)
        .expect("the render queue always exists");
    let mut list = queue.command_list()?;
    list.build_classic(&ClassicBuild {
        desc: &desc,
        dst: &tlas,
        scratch: scratch.gpu().address(),
    });
    let point = queue.submit(vec![list], &[])?;
    queue.wait(point, std::time::Duration::from_secs(10))?;
    Ok(EmptyTlas {
        tlas,
        _storage: storage,
    })
}
