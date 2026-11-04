// Atomics Metal cannot take in their plain form: the one 64-bit atomic kernels use, and
// atomics on one component of a vector in GPU memory, whose address MSL does not take.
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

// Adds `x` to component `i` of `v`, which is in GPU memory.
[ForceInline]
public void atomic_add_at<let N : int>(__ref vector<uint, N> v, int i, uint x)
{
    __target_switch
    {
    case metal:
        metal_atomic_add_at(v, i, x);
    default:
        InterlockedAdd(v[i], x);
    }
}

// Stores min(v[i], x) into component `i` of `v`, which is in GPU memory.
[ForceInline]
public void atomic_min_at<let N : int>(__ref vector<uint, N> v, int i, uint x)
{
    __target_switch
    {
    case metal:
        metal_atomic_min_at(v, i, x);
    default:
        InterlockedMin(v[i], x);
    }
}

// Stores max(v[i], x) into component `i` of `v`, which is in GPU memory.
[ForceInline]
public void atomic_max_at<let N : int>(__ref vector<uint, N> v, int i, uint x)
{
    __target_switch
    {
    case metal:
        metal_atomic_max_at(v, i, x);
    default:
        InterlockedMax(v[i], x);
    }
}

// `$0` is the vector's address: the component is an offset from it, not `&v[i]`.
void metal_atomic_add_at<let N : int>(__ref vector<uint, N> v, int i, uint x)
{
    __intrinsic_asm "atomic_fetch_add_explicit((device atomic_uint*)($0) + ($1), $2, memory_order_relaxed)";
}

void metal_atomic_min_at<let N : int>(__ref vector<uint, N> v, int i, uint x)
{
    __intrinsic_asm "atomic_fetch_min_explicit((device atomic_uint*)($0) + ($1), $2, memory_order_relaxed)";
}

void metal_atomic_max_at<let N : int>(__ref vector<uint, N> v, int i, uint x)
{
    __intrinsic_asm "atomic_fetch_max_explicit((device atomic_uint*)($0) + ($1), $2, memory_order_relaxed)";
}
