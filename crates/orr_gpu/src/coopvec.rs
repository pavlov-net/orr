//! Cooperative-vector matrix formats, alike on every backend.

use crate::ComponentType;

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
