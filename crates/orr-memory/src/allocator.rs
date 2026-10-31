use std::ffi::{CStr, CString, c_void};
use std::ptr::{self, NonNull};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use crate::{Result, check, sys};

/// Blocks on heaps over 1 GiB; smaller heaps take blocks of an eighth of the heap.
const LARGE_HEAP_BLOCK: u64 = 256 << 20;

/// What [`Allocator::new`] needs: the loaded instance and device and the features enabled
/// on them.
#[derive(Clone, Copy)]
pub struct AllocatorDesc<'a> {
    pub entry: &'a ash::Entry,
    pub instance: &'a ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: &'a ash::Device,
    /// The API version the instance and device were created with, Vulkan 1.4 at most.
    pub api_version: u32,
    /// `VK_EXT_memory_budget` is enabled: budgets come from the OS instead of estimates.
    pub memory_budget: bool,
    /// `VK_EXT_memory_priority` is enabled with `memoryPriority`: pools pass their priority.
    pub memory_priority: bool,
}

/// Sub-allocates buffers and images from large `VkDeviceMemory` blocks of its [`Pool`]s.
/// Every method may be called from any thread.
pub struct Allocator {
    raw: sys::VmaAllocator,
    /// `vkAllocateMemory` calls, counted by VMA's callback; boxed so its address is stable.
    allocations: Box<AtomicU64>,
    pools: Mutex<Vec<sys::VmaPool>>,
}

// SAFETY: VMA allocators are internally synchronized (created without
// `VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT`).
unsafe impl Send for Allocator {}
// SAFETY: as above.
unsafe impl Sync for Allocator {}

/// The blocks of one memory type at one priority, from [`Allocator::create_pool`]. Pools
/// live as long as their allocator.
#[derive(Clone, Copy, Debug)]
pub struct Pool {
    raw: sys::VmaPool,
    memory_type: u32,
}

// SAFETY: a pool handle is used only through its internally synchronized allocator.
unsafe impl Send for Pool {}
// SAFETY: as above.
unsafe impl Sync for Pool {}

impl Pool {
    pub fn memory_type(&self) -> u32 {
        self.memory_type
    }
}

/// A [`Pool`] to create.
#[derive(Clone, Copy, Debug)]
pub struct PoolDesc<'a> {
    /// Shown in [`Allocator::stats_json`].
    pub name: &'a str,
    pub memory_type: u32,
    /// 0 to 1: under memory pressure the OS keeps higher-priority blocks in VRAM first.
    pub priority: f32,
}

bitflags::bitflags! {
    /// How to place an allocation.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
    pub struct AllocationFlags: u32 {
        /// Persistently mapped; the pool's memory type must be host-visible.
        const MAPPED = 1 << 0;
        /// Its own `VkDeviceMemory`. Allocations over half a block get one anyway, as do
        /// resources whose driver prefers it.
        const DEDICATED = 1 << 1;
        /// Fail instead of going over the heap's budget.
        const WITHIN_BUDGET = 1 << 2;
        /// Other resources may be bound into the memory: a dedicated allocation is made
        /// without naming its resource, so images can be placed in a buffer's memory.
        const CAN_ALIAS = 1 << 3;
    }
}

/// Memory from an [`Allocator`]: `size` bytes at `offset` in `memory`. It stays until
/// freed through the allocator; dropping it leaks it.
#[derive(Debug)]
pub struct Allocation {
    raw: sys::VmaAllocation,
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    pub size: u64,
    pub memory_type: u32,
    /// The start of the allocation in the mapping, with [`AllocationFlags::MAPPED`].
    pub mapped: Option<NonNull<u8>>,
}

// SAFETY: the handle is used only through its internally synchronized allocator; the
// mapping is plain memory.
unsafe impl Send for Allocation {}
// SAFETY: as above.
unsafe impl Sync for Allocation {}

/// One memory heap's use and budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeapBudget {
    /// `VkDeviceMemory` blocks the allocator holds on the heap, dedicated ones included.
    pub blocks: u32,
    pub allocations: u32,
    pub block_bytes: u64,
    pub allocation_bytes: u64,
    /// What this process uses on the heap, everything included; estimated without
    /// `VK_EXT_memory_budget`.
    pub usage: u64,
    /// What the OS lets this process use; estimated without `VK_EXT_memory_budget`.
    pub budget: u64,
}

unsafe extern "C" fn on_allocate(
    _: sys::VmaAllocator,
    _: u32,
    _: vk::DeviceMemory,
    _: vk::DeviceSize,
    user: *mut c_void,
) {
    // SAFETY: `user` is the allocator's boxed counter, which outlives VMA's use of it.
    let count = unsafe { &*user.cast::<AtomicU64>() };
    count.fetch_add(1, Ordering::Relaxed);
}

impl Allocator {
    /// Creates the allocator, with buffer device addresses on every block.
    ///
    /// # Safety
    ///
    /// The handles in `desc` are live, the device has `bufferDeviceAddress` and the
    /// extensions `desc` names enabled, and the device outlives the allocator and
    /// everything allocated from it.
    pub unsafe fn new(desc: &AllocatorDesc<'_>) -> Result<Self> {
        let allocations = Box::<AtomicU64>::default();
        let callbacks = sys::VmaDeviceMemoryCallbacks {
            pfnAllocate: Some(on_allocate),
            pfnFree: None,
            pUserData: ptr::from_ref::<AtomicU64>(&allocations).cast_mut().cast(),
        };
        let functions = sys::VmaVulkanFunctions {
            vkGetInstanceProcAddr: Some(desc.entry.static_fn().get_instance_proc_addr),
            vkGetDeviceProcAddr: Some(desc.instance.fp_v1_0().get_device_proc_addr),
            ..Default::default()
        };
        let mut flags = sys::VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        if desc.memory_budget {
            flags |= sys::VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        }
        if desc.memory_priority {
            flags |= sys::VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
        }
        let info = sys::VmaAllocatorCreateInfo {
            flags,
            physicalDevice: desc.physical,
            device: desc.device.handle(),
            preferredLargeHeapBlockSize: LARGE_HEAP_BLOCK,
            pDeviceMemoryCallbacks: &callbacks,
            pVulkanFunctions: &functions,
            instance: desc.instance.handle(),
            vulkanApiVersion: desc.api_version,
            ..Default::default()
        };
        let mut allocator = ptr::null_mut();
        // SAFETY: valid create info; VMA copies the callbacks and functions.
        check("vmaCreateAllocator", unsafe {
            sys::vmaCreateAllocator(&info, &mut allocator)
        })?;
        Ok(Self {
            raw: allocator,
            allocations,
            pools: Mutex::default(),
        })
    }

    /// A pool of blocks of `desc.memory_type`: 256 MiB blocks on heaps over 1 GiB, an
    /// eighth of the heap on smaller ones. It never takes another memory type.
    pub fn create_pool(&self, desc: &PoolDesc<'_>) -> Result<Pool> {
        let info = sys::VmaPoolCreateInfo {
            memoryTypeIndex: desc.memory_type,
            priority: desc.priority,
            ..Default::default()
        };
        let mut pool = ptr::null_mut();
        // SAFETY: a live allocator and valid create info.
        check("vmaCreatePool", unsafe {
            sys::vmaCreatePool(self.raw, &info, &mut pool)
        })?;
        if let Ok(name) = CString::new(desc.name) {
            // SAFETY: VMA copies the name.
            unsafe { sys::vmaSetPoolName(self.raw, pool, name.as_ptr()) };
        }
        self.pools.lock().unwrap().push(pool);
        Ok(Pool {
            raw: pool,
            memory_type: desc.memory_type,
        })
    }

    /// VMA gives every allocation, dedicated ones too, the priority of its pool.
    fn create_info(pool: Pool, flags: AllocationFlags, align: u64) -> sys::VmaAllocationCreateInfo {
        let mut f = 0;
        if flags.contains(AllocationFlags::MAPPED) {
            f |= sys::VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        if flags.contains(AllocationFlags::DEDICATED) {
            f |= sys::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        }
        if flags.contains(AllocationFlags::WITHIN_BUDGET) {
            f |= sys::VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        }
        if flags.contains(AllocationFlags::CAN_ALIAS) {
            f |= sys::VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT;
        }
        sys::VmaAllocationCreateInfo {
            flags: f,
            pool: pool.raw,
            minAlignment: align,
            ..Default::default()
        }
    }

    fn finish(
        &self,
        raw: sys::VmaAllocation,
        info: &sys::VmaAllocationInfo,
        name: &str,
    ) -> Allocation {
        if let Ok(name) = CString::new(name) {
            // SAFETY: a live allocation; VMA copies the name.
            unsafe { sys::vmaSetAllocationName(self.raw, raw, name.as_ptr()) };
        }
        Allocation {
            raw,
            memory: info.deviceMemory,
            offset: info.offset,
            size: info.size,
            memory_type: info.memoryType,
            mapped: NonNull::new(info.pMappedData.cast()),
        }
    }

    /// Creates a buffer bound to new memory from `pool`, at an offset that is a multiple of
    /// `align` (a power of two) and of the buffer's own alignment.
    ///
    /// # Safety
    ///
    /// `info` is valid for the device.
    pub unsafe fn create_buffer(
        &self,
        pool: Pool,
        info: &vk::BufferCreateInfo<'_>,
        align: u64,
        flags: AllocationFlags,
        name: &str,
    ) -> Result<(vk::Buffer, Allocation)> {
        assert!(align.is_power_of_two(), "alignment {align}");
        let create = Self::create_info(pool, flags, align);
        let mut buffer = vk::Buffer::null();
        let mut allocation = ptr::null_mut();
        let mut out = sys::VmaAllocationInfo::default();
        // SAFETY: forwarded from the caller; VMA reads the info only during the call.
        check("vmaCreateBuffer", unsafe {
            sys::vmaCreateBuffer(
                self.raw,
                ptr::from_ref(info).cast(),
                &create,
                &mut buffer,
                &mut allocation,
                &mut out,
            )
        })?;
        Ok((buffer, self.finish(allocation, &out, name)))
    }

    /// Creates an image bound to new memory from `pool`, in its own memory with
    /// [`AllocationFlags::DEDICATED`] or when the driver prefers it.
    ///
    /// # Safety
    ///
    /// `info` is valid for the device.
    pub unsafe fn create_image(
        &self,
        pool: Pool,
        info: &vk::ImageCreateInfo<'_>,
        flags: AllocationFlags,
        name: &str,
    ) -> Result<(vk::Image, Allocation)> {
        let create = Self::create_info(pool, flags, 1);
        let mut image = vk::Image::null();
        let mut allocation = ptr::null_mut();
        let mut out = sys::VmaAllocationInfo::default();
        // SAFETY: as above.
        check("vmaCreateImage", unsafe {
            sys::vmaCreateImage(
                self.raw,
                ptr::from_ref(info).cast(),
                &create,
                &mut image,
                &mut allocation,
                &mut out,
            )
        })?;
        Ok((image, self.finish(allocation, &out, name)))
    }

    /// Memory from `pool` meeting `requirements`, bound to nothing.
    pub fn allocate(
        &self,
        pool: Pool,
        requirements: &vk::MemoryRequirements,
        flags: AllocationFlags,
        name: &str,
    ) -> Result<Allocation> {
        let create = Self::create_info(pool, flags, 1);
        let mut allocation = ptr::null_mut();
        let mut out = sys::VmaAllocationInfo::default();
        // SAFETY: a live allocator and valid create info.
        check("vmaAllocateMemory", unsafe {
            sys::vmaAllocateMemory(self.raw, requirements, &create, &mut allocation, &mut out)
        })?;
        Ok(self.finish(allocation, &out, name))
    }

    /// Binds `image` at `offset` bytes into `a`.
    ///
    /// # Safety
    ///
    /// `image` is live and unbound, and its memory requirements fit at `offset` in `a`.
    pub unsafe fn bind_image(&self, a: &Allocation, offset: u64, image: vk::Image) -> Result<()> {
        // SAFETY: forwarded from the caller.
        check("vmaBindImageMemory2", unsafe {
            sys::vmaBindImageMemory2(self.raw, a.raw, offset, image, ptr::null())
        })
    }

    /// Destroys `buffer` and frees `a`, which [`Allocator::create_buffer`] made together.
    ///
    /// # Safety
    ///
    /// No pending GPU work uses either.
    pub unsafe fn destroy_buffer(&self, buffer: vk::Buffer, a: Allocation) {
        // SAFETY: forwarded from the caller; each is destroyed once.
        unsafe { sys::vmaDestroyBuffer(self.raw, buffer, a.raw) };
    }

    /// Destroys `image` and frees `a`, which [`Allocator::create_image`] made together.
    ///
    /// # Safety
    ///
    /// No pending GPU work uses either.
    pub unsafe fn destroy_image(&self, image: vk::Image, a: Allocation) {
        // SAFETY: as above.
        unsafe { sys::vmaDestroyImage(self.raw, image, a.raw) };
    }

    /// Frees `a` at once: nothing here waits for the GPU.
    ///
    /// # Safety
    ///
    /// No pending GPU work uses it, and nothing bound to it is used again.
    pub unsafe fn deallocate(&self, a: Allocation) {
        // SAFETY: as above.
        unsafe { sys::vmaFreeMemory(self.raw, a.raw) };
    }

    /// Starts frame `index`: the budget is fetched again from the OS.
    pub fn set_frame(&self, index: u32) {
        // SAFETY: a live allocator.
        unsafe { sys::vmaSetCurrentFrameIndex(self.raw, index) };
    }

    /// Use and budget per memory heap.
    pub fn heap_budgets(&self) -> Vec<HeapBudget> {
        let mut props = ptr::null();
        // SAFETY: a live allocator; VMA keeps the properties for its lifetime.
        let heaps = unsafe {
            sys::vmaGetMemoryProperties(self.raw, &mut props);
            (*props).memory_heap_count as usize
        };
        let mut budgets: [sys::VmaBudget; vk::MAX_MEMORY_HEAPS] =
            std::array::from_fn(|_| sys::VmaBudget::default());
        // SAFETY: room for every heap.
        unsafe { sys::vmaGetHeapBudgets(self.raw, budgets.as_mut_ptr()) };
        budgets[..heaps]
            .iter()
            .map(|b| HeapBudget {
                blocks: b.statistics.blockCount,
                allocations: b.statistics.allocationCount,
                block_bytes: b.statistics.blockBytes,
                allocation_bytes: b.statistics.allocationBytes,
                usage: b.usage,
                budget: b.budget,
            })
            .collect()
    }

    /// VMA's JSON statistics: totals per heap, memory type and pool, and with `detailed`
    /// every block and named allocation.
    pub fn stats_json(&self, detailed: bool) -> String {
        let mut s = ptr::null_mut();
        // SAFETY: a live allocator; the string is freed once, after the copy.
        unsafe {
            sys::vmaBuildStatsString(self.raw, &mut s, u32::from(detailed));
            let json = CStr::from_ptr(s).to_string_lossy().into_owned();
            sys::vmaFreeStatsString(self.raw, s);
            json
        }
    }

    /// `vkAllocateMemory` calls this allocator has made.
    pub fn device_memory_allocations(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // SAFETY: the pools and allocator are live and destroyed once, pools first.
        unsafe {
            for pool in self.pools.get_mut().unwrap().drain(..) {
                sys::vmaDestroyPool(self.raw, pool);
            }
            sys::vmaDestroyAllocator(self.raw);
        }
    }
}
