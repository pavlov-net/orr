// Sequences of device-generated commands. Kernels fill `Sequence` and write it with `store`;
// `CommandList::execute_generated` runs the array, binding each sequence's pipeline, pushing
// its root and dispatching or drawing its workgroups.
module sequence;

public static const uint SEQUENCE_STRIDE = 24;

public struct Sequence
{
    public uint pipeline; // index into the execution set
    public uint64_t root; // the root pointer the pipeline reads
    public uint3 groups;  // workgroup counts; task workgroups for mesh sets with a task stage
}

// Writes sequence `index` of the array at `sequences`, a multiple of 4.
public void store(uint64_t sequences, uint index, Sequence v)
{
    uint* p = (uint*)(sequences + uint64_t(index) * SEQUENCE_STRIDE);
    p[0] = v.pipeline;
    p[1] = uint(v.root);
    p[2] = uint(v.root >> 32);
    p[3] = v.groups.x;
    p[4] = v.groups.y;
    p[5] = v.groups.z;
}
