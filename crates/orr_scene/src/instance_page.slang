// Writes a cooked instance page's slots from its records once they are resident: each record
// takes its prototype's instance record and emissive list entries.
#include "orr_root.h"
import orr_pack.instance;
import orr_scene.lights;
import orr_scene.tables;

// A prototype of a placed page: what each of its copies takes.
public struct PagePrototypeGpu
{
    public InstanceGpu instance;   // `first_emissive` and `emissive_count` are the slot's own
    public uint records;           // the page's records that name it
    public uint3 _pad;
}

// The emissive list entries of a prototype's copies: `count` each from `first` past the page's
// first entry, in the order of their ranks.
public struct PageEntriesGpu
{
    public uint first;
    public uint count;
}

public struct PageExpandRoot
{
    public SceneRoot* scene;
    public InstanceRecord* records;
    public PagePrototypeGpu* prototypes;
    public PageEntriesGpu* entries; // per prototype
    public uint first_slot;
    public uint count;
    public uint prototype_count;
    public uint first_entry;       // the page's first instance entry of the emissive list
    public uint exact_rows;        // rows of the page's exact table
    public uint entries_only;      // 1: the slots are written, and take new entries alone
    public uint _pad0;
    public uint _pad1;
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
    if ((r.scale & INDEXED) != 0 && r.rot >= AXIS_ROTATIONS + root.exact_rows)
        return;
    PagePrototypeGpu p = root.prototypes[prototype];
    if (r.rank >= p.records)
        return;
    SceneRoot* s = root.scene;
    uint slot = root.first_slot + i;
    PageEntriesGpu entries = root.entries[prototype];
    uint entry = root.first_entry + entries.first + r.rank * entries.count;
    uint first = entries.count != 0 ? entry : NO_EMISSIVE;
    if (root.entries_only != 0)
    {
        s.instances[slot].first_emissive = first;
        s.instances[slot].emissive_count = entries.count;
    }
    else
    {
        InstanceGpu inst = p.instance;
        inst.group_index = i;
        inst.first_emissive = first;
        inst.emissive_count = entries.count;
        s.instances[slot] = inst;
    }
    for (uint k = 0; k < entries.count; k++)
    {
        EmissiveGpu e = EmissiveGpu();
        e.kind = EMISSIVE_INSTANCE;
        e.source = slot;
        e.rank = k;
        e.set = NO_SET;
        s.emissive[entry + k] = e;
    }
}
