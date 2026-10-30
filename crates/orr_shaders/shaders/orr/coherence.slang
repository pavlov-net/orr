// Coherence points for kernel bodies shared by a raygen and a compute entry.
//
// A body is `void body<C : ICoherence>(uint2 index)`. The raygen entry calls
// `body<Reorder>(DispatchRaysIndex().xy)` and the compute entry `body<NoReorder>(id.xy)`.
// The body calls `C.point(key, bits)` in uniform control flow after traversal and before
// shading; lanes with equal keys run on together after a `Reorder` point.
module coherence;

// The bits of a coherence key: hit or miss, the 2-bit cost class, then the linked surface id.
// A build that links surfaces sets it to 3 plus the bit length of its highest surface id.
public extern static const uint KEY_BITS = 3;

// The coherence key of a hit or miss with its material's cost class and surface id.
public uint coherence_key(bool hit, uint cost_class, uint surface)
{
    return uint(hit) | ((cost_class & 3u) << 1) | (surface << 3);
}

public interface ICoherence
{
    // `key` uses its low `bits` bits.
    static void point(uint key, uint bits);
}

// Reorders lanes by key. Raygen entries only.
public struct Reorder : ICoherence
{
    [ForceInline]
    public static void point(uint key, uint bits)
    {
        ReorderThread(key, bits);
    }
}

// No reordering, for compute entries.
public struct NoReorder : ICoherence
{
    [ForceInline]
    public static void point(uint key, uint bits) {}
}
