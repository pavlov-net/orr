//! Types shared by Rust and Slang: GPU addresses, heap slots and specialisation data.

use core::fmt;
use core::marker::PhantomData;

use bytemuck::{Pod, Zeroable};

/// A GPU virtual address; 0 is null.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug, Pod, Zeroable)]
pub struct GpuAddress(pub u64);

/// A typed GPU address: a `T*` in Slang.
#[repr(transparent)]
pub struct GpuPtr<T: 'static>(u64, PhantomData<fn() -> T>);

impl<T> GpuPtr<T> {
    /// The null pointer.
    pub const NULL: Self = Self(0, PhantomData);

    /// The untyped address.
    pub const fn address(self) -> GpuAddress {
        GpuAddress(self.0)
    }

    /// The pointer `elements` values of `T` further on.
    pub const fn offset(self, elements: u64) -> Self {
        Self(self.0 + elements * size_of::<T>() as u64, PhantomData)
    }

    /// The same address as a pointer to `U`.
    pub const fn cast<U>(self) -> GpuPtr<U> {
        GpuPtr(self.0, PhantomData)
    }

    /// Types an address.
    ///
    /// # Safety
    /// The address must be null or point at memory laid out as `T` for as long as a shader
    /// reads or writes it through this pointer.
    pub const unsafe fn from_address(address: GpuAddress) -> Self {
        Self(address.0, PhantomData)
    }
}

impl<T> Clone for GpuPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GpuPtr<T> {}
impl<T> PartialEq for GpuPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for GpuPtr<T> {}
impl<T> Default for GpuPtr<T> {
    fn default() -> Self {
        Self::NULL
    }
}
impl<T> fmt::Debug for GpuPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GpuPtr({:#x})", self.0)
    }
}
// SAFETY: a single u64 with no padding; PhantomData is zero-sized; all bit patterns valid.
unsafe impl<T> Zeroable for GpuPtr<T> {}
// SAFETY: as above; `fn() -> T` keeps the type Send, Sync and 'static regardless of `T`.
unsafe impl<T: 'static> Pod for GpuPtr<T> {}

macro_rules! slot {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Pod, Zeroable)]
        pub struct $name(pub u32);

        impl $name {
            /// No slot.
            pub const NONE: Self = Self(u32::MAX);
        }
    };
}

slot!(
    /// A sampled image view's slot in the resource heap.
    TextureIndex
);
slot!(
    /// A storage image view's slot in the resource heap; shares the slot space of
    /// [`TextureIndex`].
    StorageImageIndex
);
slot!(
    /// A sampler's slot in the sampler heap.
    SamplerIndex
);

/// The device address of a TLAS or partitioned TLAS. Never 0 when traced.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Pod, Zeroable)]
pub struct Tlas(pub u64);

/// One specialisation constant's place in a [`SpecData`] byte block.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Pod, Zeroable)]
pub struct SpecEntry {
    pub constant_id: u32,
    pub offset: u32,
    pub size: u32,
}

/// Specialisation constant values: entries index into `bytes`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecData<'a> {
    pub entries: &'a [SpecEntry],
    pub bytes: &'a [u8],
}
