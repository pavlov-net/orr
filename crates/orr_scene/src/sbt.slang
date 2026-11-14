// A scene's hit table for one ray tracing pipeline: per run, a record per ray type, each the
// handle of the pipeline's group for what the run serves and the run's `HitData`. A run is one
// section of a material row: `runs` writes those whose material an array of the scene's tables
// names, `producer_runs` an entry's. Records of unused runs stay as they were: no instance's
// contribution names them.
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
public static const uint KIND_LAYERS = 0;
public static const uint KIND_MEDIUM_SHADOW = 1;
public static const uint KIND_MEDIUM_LAYERS = 2;
public static const uint KIND_LIGHT_LAYERS = 3;
public static const uint KIND_WALK_MEET = 4;
public static const uint KIND_STACK_MEET = 5;
public static const uint FIXED_KINDS = 6;
// The groups it holds `pins` of after those: one for any surface id, then, where surfaces link,
// one per pin from id 0 on.
public static const uint KIND_PATH = 0;          // a copy's clusters
public static const uint KIND_PRODUCER_PATH = 1; // a producer entry's clusters
public static const uint KIND_STRAND_PATH = 2;   // a strand entry's
public static const uint KIND_SHADOW = 3;
public static const uint KIND_LAYER_PATH = 4;          // a copy's dielectric and presence layer clusters
public static const uint KIND_PRODUCER_LAYER_PATH = 5; // a producer entry's
public static const uint PINNED_KINDS = 6;
// No group: a record of zeros.
static const uint SLOT_NONE = 0xffffffff;

public struct HitTableRoot
{
    public SceneRoot* scene;
    public uint* records;           // the table
    public uint* handles;           // FIXED_KINDS + PINNED_KINDS * pins handles of `handle_words` words, zero for none
    public uint stride_words;       // a record's
    public uint handle_words;
    public uint surfaces;           // linked surface types, ids 1 to `surfaces`
    public uint pins;               // 1 where none link, else `surfaces` + 2
    public uint64_t _pad;
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

// The kind of the path ray groups of a run of class `class_id` on geometry whose path rays take
// kind `path`: layers on clusters resolve as layers. Strands are opaque.
uint path_kind(uint path, uint class_id)
{
    if (path == KIND_STRAND_PATH || (class_id != CLASS_DIELECTRIC && class_id != CLASS_PRESENCE))
        return path;
    return path == KIND_PATH ? KIND_LAYER_PATH : KIND_PRODUCER_LAYER_PATH;
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
        d[3] = data._spare;
    }
}

// One thread per run outside the producer rows.
[shader("compute")]
[numthreads(64, 1, 1)]
void runs(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    uint run = id.x;
    if (run >= s.run_count || (run >= s.run_producers && run < s.run_pages))
        return;
    if (run == RUN_MEDIUM)
    {
        write_run(run, uint3(SLOT_NONE, KIND_MEDIUM_SHADOW, KIND_MEDIUM_LAYERS), proxy_data(CLASS_MEDIUM));
        return;
    }
    if (run == RUN_LIGHT)
    {
        write_run(run, uint3(SLOT_NONE, SLOT_NONE, KIND_LIGHT_LAYERS), proxy_data(CLASS_LIGHT));
        return;
    }
    if (run == RUN_WALK)
    {
        write_run(run, uint3(KIND_WALK_MEET, KIND_STACK_MEET, SLOT_NONE), proxy_data(CLASS_OPAQUE));
        return;
    }
    uint material;
    if (run < s.run_sets)
        material = s.model_materials[run - s.run_models];
    else if (run < s.run_producers)
        material = s.material_sets[run - s.run_sets];
    else
        material = s.page_material_sets[run - s.run_pages];
    uint class_id = material == MATERIAL_NONE ? CLASS_OPAQUE : material_class(s.materials + material);
    uint pin = material_pin(s, material);
    write_run(run, uint3(pinned(path_kind(KIND_PATH, class_id), pin), pinned(KIND_SHADOW, pin), KIND_LAYERS), hit_data(s, material, class_id));
}

// One workgroup per producer entry, a thread per material of a set entry.
[shader("compute")]
[numthreads(64, 1, 1)]
void producer_runs(uint3 group: SV_GroupID, uint t: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    uint entry = group.x;
    if (entry >= s.producer_capacity)
        return;
    ProducerGpu p = s.producers[entry];
    uint path = p.strands != 0 ? KIND_STRAND_PATH : KIND_PRODUCER_PATH;
    for (uint slot = t; slot < p.material_count; slot += 64)
    {
        uint material = p.materials[slot];
        uint class_id = p.opacity == PRODUCER_SECTIONS ? material_class(s.materials + material) : p.class_id;
        uint pin = material_pin(s, material);
        write_run(s.run_producers + p.runs + slot, uint3(pinned(path_kind(path, class_id), pin), pinned(KIND_SHADOW, pin), KIND_LAYERS), hit_data(s, material, class_id));
    }
}
