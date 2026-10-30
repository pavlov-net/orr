// The one 64-bit atomic kernels use.
module atomic;

// Stores max(*p, v) into `*p`. There is no result: Metal's 64-bit max returns nothing.
[ForceInline]
public void atomic_max_u64(uint64_t* p, uint64_t v)
{
    __target_switch
    {
    case metal:
        metal_atomic_max_u64(p, v);
    default:
        InterlockedMax(*p, v);
    }
}

void metal_atomic_max_u64(uint64_t* p, uint64_t v)
{
    __intrinsic_asm "atomic_max_explicit((device atomic_ulong*)($0), $1, memory_order_relaxed)";
}
