//! Cooperative-vector weight matrices: their sizes, and conversion into the layouts kernels
//! multiply with.

use ash::vk;

use super::command::CommandList;
use super::device::{Device, Shared};
use super::memory::Allocation;
use crate::{ComponentType, Error, GpuAddress};

/// Matrices start at multiples of this many bytes. The driver neither faults nor reports
/// a misaligned matrix, so conversions check it.
pub const MATRIX_ALIGN: u64 = 64;

/// How a cooperative-vector matrix lays out its elements.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
    /// The device's layout for inference; kernels multiply with matrices in it.
    InferencingOptimal,
    /// The device's layout for training.
    TrainingOptimal,
}

/// The element type and layout of a matrix. `stride` is the bytes from one row (row-major)
/// or column (column-major) to the next, 0 for tightly packed; the optimal layouts ignore it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MatrixFormat {
    pub component: ComponentType,
    pub layout: MatrixLayout,
    pub stride: u32,
}

impl MatrixFormat {
    /// Tightly packed in `layout`.
    pub fn new(component: ComponentType, layout: MatrixLayout) -> Self {
        Self {
            component,
            layout,
            stride: 0,
        }
    }
}

/// A `rows` x `columns` matrix converted from `src_format` to `dst_format`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MatrixConversion {
    pub rows: u32,
    pub columns: u32,
    pub src_format: MatrixFormat,
    pub dst_format: MatrixFormat,
}

pub(crate) fn component(t: vk::ComponentTypeKHR) -> ComponentType {
    use vk::ComponentTypeKHR as C;
    match t {
        C::FLOAT16 => ComponentType::F16,
        C::FLOAT32 => ComponentType::F32,
        C::FLOAT64 => ComponentType::F64,
        C::SINT8 => ComponentType::I8,
        C::SINT16 => ComponentType::I16,
        C::SINT32 => ComponentType::I32,
        C::SINT64 => ComponentType::I64,
        C::UINT8 => ComponentType::U8,
        C::UINT16 => ComponentType::U16,
        C::UINT32 => ComponentType::U32,
        C::UINT64 => ComponentType::U64,
        C::FLOAT8_E4M3_EXT => ComponentType::E4m3,
        C::FLOAT8_E5M2_EXT => ComponentType::E5m2,
        other => ComponentType::Other(other.as_raw()),
    }
}

fn vk_component(t: ComponentType) -> vk::ComponentTypeKHR {
    use vk::ComponentTypeKHR as C;
    match t {
        ComponentType::F16 => C::FLOAT16,
        ComponentType::F32 => C::FLOAT32,
        ComponentType::F64 => C::FLOAT64,
        ComponentType::I8 => C::SINT8,
        ComponentType::I16 => C::SINT16,
        ComponentType::I32 => C::SINT32,
        ComponentType::I64 => C::SINT64,
        ComponentType::U8 => C::UINT8,
        ComponentType::U16 => C::UINT16,
        ComponentType::U32 => C::UINT32,
        ComponentType::U64 => C::UINT64,
        ComponentType::E4m3 => C::FLOAT8_E4M3_EXT,
        ComponentType::E5m2 => C::FLOAT8_E5M2_EXT,
        ComponentType::Other(raw) => C::from_raw(raw),
    }
}

fn vk_layout(l: MatrixLayout) -> vk::CooperativeVectorMatrixLayoutNV {
    use vk::CooperativeVectorMatrixLayoutNV as L;
    match l {
        MatrixLayout::RowMajor => L::ROW_MAJOR,
        MatrixLayout::ColumnMajor => L::COLUMN_MAJOR,
        MatrixLayout::InferencingOptimal => L::INFERENCING_OPTIMAL,
        MatrixLayout::TrainingOptimal => L::TRAINING_OPTIMAL,
    }
}

/// Bytes per element of `t`.
fn element_bytes(t: ComponentType) -> u32 {
    match t {
        ComponentType::I8 | ComponentType::U8 | ComponentType::E4m3 | ComponentType::E5m2 => 1,
        ComponentType::F16 | ComponentType::I16 | ComponentType::U16 => 2,
        ComponentType::F32 | ComponentType::I32 | ComponentType::U32 => 4,
        ComponentType::F64 | ComponentType::I64 | ComponentType::U64 => 8,
        ComponentType::Other(raw) => panic!("component type {raw} has no known size"),
    }
}

/// The stride of `f` for `rows` x `columns`: the caller's, or tightly packed.
fn stride(f: &MatrixFormat, rows: u32, columns: u32) -> u32 {
    if f.stride != 0 {
        return f.stride;
    }
    let lines = match f.layout {
        MatrixLayout::RowMajor => columns,
        MatrixLayout::ColumnMajor => rows,
        _ => return 0,
    };
    lines * element_bytes(f.component)
}

/// The conversion info of `rows` x `columns` from `src` to `dst`, sizes and data left for
/// the caller.
fn info(
    rows: u32,
    columns: u32,
    src: &MatrixFormat,
    dst: &MatrixFormat,
) -> vk::ConvertCooperativeVectorMatrixInfoNV<'static> {
    vk::ConvertCooperativeVectorMatrixInfoNV {
        src_component_type: vk_component(src.component),
        dst_component_type: vk_component(dst.component),
        num_rows: rows,
        num_columns: columns,
        src_layout: vk_layout(src.layout),
        src_stride: stride(src, rows, columns) as usize,
        dst_layout: vk_layout(dst.layout),
        dst_stride: stride(dst, rows, columns) as usize,
        ..Default::default()
    }
}

impl Shared {
    fn coop_fn(&self) -> &ash::nv::cooperative_vector::Device {
        self.coop_fn
            .as_ref()
            .expect("matrix conversion needs cooperative vectors (Caps::cooperative_vector)")
    }

    /// Bytes of a `rows` x `columns` matrix in `f`.
    fn matrix_bytes(&self, rows: u32, columns: u32, f: &MatrixFormat) -> Result<usize, Error> {
        let row_major = MatrixFormat::new(f.component, MatrixLayout::RowMajor);
        let src = vec![0u8; (rows * stride(&row_major, rows, columns)) as usize];
        let mut bytes = 0usize;
        let mut info = info(rows, columns, &row_major, f);
        info.src_size = src.len();
        info.src_data = vk::DeviceOrHostAddressConstKHR {
            host_address: src.as_ptr().cast(),
        };
        info.p_dst_size = &mut bytes;
        // SAFETY: a size query (null destination) over a live host source.
        unsafe { self.coop_fn().convert_cooperative_vector_matrix(&info) }
            .map_err(Error::vk("vkConvertCooperativeVectorMatrixNV"))?;
        Ok(bytes)
    }
}

fn check_aligned(address: u64) {
    assert!(
        address.is_multiple_of(MATRIX_ALIGN),
        "matrix at {address:#x} is not a multiple of {MATRIX_ALIGN} bytes"
    );
}

impl Device {
    /// Bytes a `rows` x `columns` matrix takes in `format`. Needs cooperative vectors.
    pub fn matrix_bytes(
        &self,
        rows: u32,
        columns: u32,
        format: MatrixFormat,
    ) -> Result<u64, Error> {
        self.shared
            .matrix_bytes(rows, columns, &format)
            .map(|b| b as u64)
    }

    /// Converts the matrix in host memory `src` into mapped `dst` at `offset` (a multiple of
    /// [`MATRIX_ALIGN`] from the allocation's address), on the calling thread. Returns the
    /// bytes written. Needs cooperative vectors.
    pub fn convert_matrix(
        &self,
        c: &MatrixConversion,
        src: &[u8],
        dst: &mut Allocation<u8>,
        offset: u64,
    ) -> Result<u64, Error> {
        let shared = &self.shared;
        check_aligned(dst.gpu.address().0 + offset);
        let mut bytes = shared.matrix_bytes(c.rows, c.columns, &c.dst_format)?;
        let size = dst.size;
        let mapped = dst
            .as_mut_slice()
            .ok_or_else(|| Error::Invalid("matrix destination is not mapped".into()))?;
        if offset + bytes as u64 > size {
            return Err(Error::Invalid(format!(
                "{bytes} matrix bytes at offset {offset} in an allocation of {size}"
            )));
        }
        let mut info = info(c.rows, c.columns, &c.src_format, &c.dst_format);
        info.src_size = src.len();
        info.src_data = vk::DeviceOrHostAddressConstKHR {
            host_address: src.as_ptr().cast(),
        };
        info.p_dst_size = &mut bytes;
        info.dst_data = vk::DeviceOrHostAddressKHR {
            host_address: mapped[offset as usize..].as_mut_ptr().cast(),
        };
        // SAFETY: both host ranges are live and hold the sizes given.
        unsafe { shared.coop_fn().convert_cooperative_vector_matrix(&info) }
            .map_err(Error::vk("vkConvertCooperativeVectorMatrixNV"))?;
        Ok(bytes as u64)
    }
}

impl CommandList<'_> {
    /// Converts the matrix at `src` into `dst` (a multiple of [`MATRIX_ALIGN`]) on the GPU.
    /// Later reads of the result wait on a barrier from [`crate::Stages::TRANSFER`]. Needs
    /// cooperative vectors and a graphics or compute queue.
    pub fn convert_matrix(&mut self, c: &MatrixConversion, src: GpuAddress, dst: GpuAddress) {
        let shared = &self.queue().shared;
        assert!(
            self.queue().shaders,
            "matrix conversion runs on a graphics or compute queue"
        );
        check_aligned(dst.0);
        let bytes = |f| {
            shared
                .matrix_bytes(c.rows, c.columns, f)
                .unwrap_or_else(|e| panic!("matrix size: {e}"))
        };
        let mut dst_size = bytes(&c.dst_format);
        let mut info = info(c.rows, c.columns, &c.src_format, &c.dst_format);
        info.src_size = bytes(&c.src_format);
        info.src_data = vk::DeviceOrHostAddressConstKHR {
            device_address: src.0,
        };
        info.p_dst_size = &mut dst_size;
        info.dst_data = vk::DeviceOrHostAddressKHR {
            device_address: dst.0,
        };
        // SAFETY: recording on a graphics or compute queue; the addresses are the caller's
        // live allocations.
        unsafe {
            shared
                .coop_fn()
                .cmd_convert_cooperative_vector_matrix(self.cmd(), &[info]);
        }
    }
}
