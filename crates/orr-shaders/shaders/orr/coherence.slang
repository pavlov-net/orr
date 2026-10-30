// Coherence points for kernel bodies shared by a raygen and a compute entry.
//
// A body is `void body<C : ICoherence>(uint2 index)`. The raygen entry calls
// `body<Reorder>(DispatchRaysIndex().xy)` and the compute entry `body<NoReorder>(id.xy)`.
// The body calls `C.point(key, bits)` in uniform control flow after traversal and before
// shading; lanes with equal keys run on together after a `Reorder` point.
module coherence;

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
