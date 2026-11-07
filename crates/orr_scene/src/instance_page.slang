// Writes a cooked instance page's slots from its records once they are resident: each record
// takes its prototype's instance record and emissive list entries.
#include "orr_root.h"
import orr_pack.instance;
import orr_scene.lights;
import orr_scene.tables;

// A prototype of a placed page: what each of its copies takes.
public struct PagePrototypeGpu
{
    public InstanceGpu instance;   // `first_emissive` is the slot's own
    public uint emissive_count;    // the emissive and caster section ranks of its model
    public uint3 _pad;
}

public struct PageExpandRoot
{
    public SceneRoot* scene;
    public InstanceRecord* records;
    public PagePrototypeGpu* prototypes;
    public uint first_slot;
    public uint count;
    public uint prototype_count;
    public uint first_entry;       // the page's first instance entry of the emissive list
    public uint entry_stride;      // entries per record
    public uint linear_rows;       // rows of the page's linear table
}
ORR_ROOT(PageExpandRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void expand(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.count)
        return;
    InstanceRecord r = root.records[i];
    uint prototype = r.prototype;
    if (prototype >= root.prototype_count)
        return;
    if ((r.scale & INDEXED) != 0 && r.rot >= AXIS_ROTATIONS + root.linear_rows)
        return;
    SceneRoot* s = root.scene;
    uint slot = root.first_slot + i;
    PagePrototypeGpu p = root.prototypes[prototype];
    uint entry = s.producer_capacity + root.first_entry + i * root.entry_stride;
    InstanceGpu inst = p.instance;
    inst.group_index = i;
    inst.first_emissive = p.emissive_count != 0 ? entry : NO_EMISSIVE;
    inst.emissive_count = p.emissive_count;
    s.instances[slot] = inst;
    for (uint k = 0; k < p.emissive_count; k++)
    {
        EmissiveGpu e = (EmissiveGpu)0;
        e.kind = EMISSIVE_INSTANCE;
        e.source = slot;
        e.rank = k;
        e.set = NO_SET;
        s.emissive[entry + k] = e;
    }
}
