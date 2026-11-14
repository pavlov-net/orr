// A scene's hit table for one ray tracing pipeline: per run, a record per ray type, each the
// handle of the pipeline's group for what the run serves and the run's `HitData`. The host writes
// the runs a table has not seen yet, by ranges: those of rows made, changed or gone since it last
// wrote, or all of them. A run no row holds writes a record of zeros, which runs no shader.
//
// A run's path and shadow groups are its geometry's (a copy's cooked or skinned clusters, a
// producer's clusters, strands) for its material's linked surface: the pipeline holds those
// groups once per pin, a surface id its shaders are made for (`scene_may.slang`'s
// `hit_surface`), and once for any id, which a material whose back shows another id takes.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.runs;
import orr_scene.tables;

// The groups the handle table holds one of, in its order (orr_scene::HitGroups).
[Host] public static const uint KIND_LAYERS = 0;
[Host] public static const uint KIND_MEDIUM_SHADOW = 1;
[Host] public static const uint KIND_MEDIUM_LAYERS = 2;
[Host] public static const uint KIND_LIGHT_LAYERS = 3;
[Host] public static const uint KIND_WALK_MEET = 4;
[Host] public static const uint KIND_STACK_MEET = 5;
[Host] public static const uint FIXED_KINDS = 6;
// The groups it holds `pins` of after those: one for any surface id, then, where surfaces link,
// one per pin from id 0 on.
[Host] public static const uint KIND_PATH = 0;          // a copy's clusters
[Host] public static const uint KIND_PRODUCER_PATH = 1; // a producer entry's clusters
[Host] public static const uint KIND_STRAND_PATH = 2;   // a strand entry's
[Host] public static const uint KIND_SHADOW = 3;
[Host] public static const uint PINNED_KINDS = 4;
// No group: a record of zeros.
static const uint SLOT_NONE = 0xffffffff;

// The runs one workgroup of `runs` writes at most: a range of the table's dispatch.
[Host] public static const uint RUNS_PER_GROUP = 4096;

public struct HitTableRoot
{
    public SceneRoot* scene;
    public uint* records;           // the table
    public uint* handles;           // FIXED_KINDS + PINNED_KINDS * pins handles of `handle_words` words, zero for none
    public uint2* ranges;           // per workgroup, the first run it writes and how many, RUNS_PER_GROUP at most
    public uint stride_words;       // a record's
    public uint handle_words;
    public uint surfaces;           // linked surface types, ids 1 to `surfaces`
    public uint pins;               // 1 where none link, else `surfaces` + 2
}
ORR_ROOT(HitTableRoot);

// The handle table slot of pinned kind `kind` at pin `pin`.
uint pinned(uint kind, uint pin)
{
    return FIXED_KINDS + kind * root.pins + pin;
}

// The pin of the groups that shade material `material`: its linked surface id's, or with a back
// of another id the pin of any id.
uint material_pin(SceneRoot* s, uint material)
{
    if (root.surfaces == 0 || material == MATERIAL_NONE)
        return 0;
    MaterialGpu* m = s.materials + material;
    uint id = m.surface <= root.surfaces ? m.surface : 0;
    if ((m.flags & MATERIAL_HAS_BACK) != 0)
    {
        uint back = s.materials[m.back].surface;
        if ((back <= root.surfaces ? back : 0) != id)
            return 0;
    }
    return 1 + id;
}

// Writes run `run`'s records: per ray type the handle of slot `slots[type]` and `data`.
void write_run(uint run, uint3 slots, HitData data)
{
    for (uint type = 0; type < RAY_TYPES; type++)
    {
        uint* r = root.records + (RAY_TYPES * run + type) * root.stride_words;
        uint slot = slots[type];
        for (uint w = 0; w < root.handle_words; w++)
            r[w] = slot == SLOT_NONE ? 0 : root.handles[slot * root.handle_words + w];
        uint* d = r + root.handle_words;
        d[0] = data.material;
        d[1] = data.flags;
        d[2] = data.back;
        d[3] = data.attributes;
    }
}

// The records of run `run` as `SceneRoot.runs` describes it.
void write_records(SceneRoot* s, uint run)
{
    RunGpu g = s.runs[run];
    uint kind = g.word & ROW_KIND_MASK;
    uint3 none = uint3(SLOT_NONE);
    if (kind == ROW_NONE)
    {
        write_run(run, none, proxy_data(CLASS_OPAQUE));
        return;
    }
    if (kind == ROW_FIXED)
    {
        if (run == RUN_MEDIUM)
            write_run(run, uint3(SLOT_NONE, KIND_MEDIUM_SHADOW, KIND_MEDIUM_LAYERS), proxy_data(CLASS_MEDIUM));
        else if (run == RUN_LIGHT)
            write_run(run, uint3(SLOT_NONE, SLOT_NONE, KIND_LIGHT_LAYERS), proxy_data(CLASS_LIGHT));
        else
            write_run(run, uint3(KIND_WALK_MEET, KIND_STACK_MEET, SLOT_NONE), proxy_data(CLASS_OPAQUE));
        return;
    }
    uint material;
    uint class_id;
    uint attributes = 0;
    uint path = KIND_PATH;
    if (kind == ROW_PRODUCER)
    {
        // The entry's record and its row change together, a run per material.
        ProducerGpu p = s.producers[g.source];
        material = p.materials[g.word >> ROW_SLOT_SHIFT];
        class_id = p.opacity == PRODUCER_SECTIONS ? material_class(s.materials + material) : p.class_id;
        path = p.strands != 0 ? KIND_STRAND_PATH : KIND_PRODUCER_PATH;
    }
    else
    {
        material = kind == ROW_MODEL ? s.model_materials[g.source] : kind == ROW_SET ? s.material_sets[g.source] : s.page_material_sets[g.source];
        class_id = (g.word >> ROW_CLASS_SHIFT) & HIT_CLASS_MASK;
        attributes = (g.word >> ROW_ATTRIBUTES_SHIFT) & 0xff;
    }
    uint pin = material_pin(s, material);
    HitData data = hit_data(s, material, class_id, (g.word & ROW_SOLID) != 0, attributes);
    write_run(run, uint3(pinned(path, pin), pinned(KIND_SHADOW, pin), KIND_LAYERS), data);
}

// One workgroup per range of runs.
[shader("compute")]
[numthreads(64, 1, 1)]
void runs(uint3 group: SV_GroupID, uint t: SV_GroupThreadID)
{
    uint2 range = root.ranges[group.x];
    for (uint k = t; k < range.y; k += 64)
        write_records(root.scene, range.x + k);
}
