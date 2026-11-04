// Writes a cooked instance page's slots from its records once they are resident: each record
// takes its prototype's instance record and emissive list entries.
#include "orr_root.h"
import orr_pack.instance;
import orr_scene.lights;
import orr_scene.tables;

// A prototype of a placed page: what each of its copies takes.
public struct PagePrototypeGpu
{
    public InstanceGpu instance;
    public uint first_span;        // in PageExpandRoot.spans
    public uint span_count;
    public uint2 _pad;
}

public struct PageExpandRoot
{
    public SceneRoot* scene;
    public InstanceRecord* records;
    public PagePrototypeGpu* prototypes;
    public EmissiveGpu* spans;
    public uint first_slot;
    public uint count;
    public uint prototype_count;
    public uint first_entry;       // the page's first instance entry of the emissive list
    public uint entry_stride;      // entries per record
    public uint _pad0;
    public uint64_t _pad1;
}
ORR_ROOT(PageExpandRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void expand(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.count)
        return;
    uint prototype = root.records[i].prototype;
    if (prototype >= root.prototype_count)
        return;
    SceneRoot* s = root.scene;
    uint slot = root.first_slot + i;
    PagePrototypeGpu p = root.prototypes[prototype];
    InstanceGpu inst = p.instance;
    inst.group_index = i;
    s.instances[slot] = inst;
    for (uint k = 0; k < p.span_count; k++)
    {
        EmissiveGpu e = root.spans[p.first_span + k];
        e.source = slot;
        s.emissive[s.producer_capacity + root.first_entry + i * root.entry_stride + k] = e;
    }
}
