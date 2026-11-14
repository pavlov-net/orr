// A scene's hit table for one ray tracing pipeline: per run, a record per ray type, each the
// handle of the pipeline's group for what the run serves and the run's `HitData`. A run is one
// section of a material row: `runs` writes those whose material an array of the scene's tables
// names, `producer_runs` an entry's. Records of unused runs stay as they were: no instance's
// contribution names them.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.runs;
import orr_scene.tables;

// What a group serves, the handle table's order (orr_scene::HitGroups).
public static const uint KIND_PATH = 0;
public static const uint KIND_SHADOW = 1;
public static const uint KIND_PARTICLE_LAYERS = 2;
public static const uint KIND_MEDIUM_SHADOW = 3;
public static const uint KIND_MEDIUM_LAYERS = 4;
public static const uint KIND_LIGHT_LAYERS = 5;
public static const uint KINDS = 6;
// No group: a record of zeros.
static const uint KIND_NONE = 0xffffffff;

public struct HitTableRoot
{
    public SceneRoot* scene;
    public uint* records;           // the table
    public uint* handles;           // KINDS handles of `handle_words` words, zero for none
    public uint stride_words;       // a record's
    public uint handle_words;
}
ORR_ROOT(HitTableRoot);

// Writes run `run`'s records: per ray type the handle of group kind `kinds[type]` and `data`.
void write_run(uint run, uint3 kinds, HitData data)
{
    for (uint type = 0; type < RAY_TYPES; type++)
    {
        uint* r = root.records + (RAY_TYPES * run + type) * root.stride_words;
        uint kind = kinds[type];
        for (uint w = 0; w < root.handle_words; w++)
            r[w] = kind == KIND_NONE ? 0 : root.handles[kind * root.handle_words + w];
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
        write_run(run, uint3(KIND_NONE, KIND_MEDIUM_SHADOW, KIND_MEDIUM_LAYERS), proxy_data(CLASS_MEDIUM));
        return;
    }
    if (run == RUN_LIGHT)
    {
        write_run(run, uint3(KIND_NONE, KIND_NONE, KIND_LIGHT_LAYERS), proxy_data(CLASS_LIGHT));
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
    write_run(run, uint3(KIND_PATH, KIND_SHADOW, KIND_NONE), hit_data(s, material, class_id));
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
    for (uint slot = t; slot < p.material_count; slot += 64)
    {
        uint material = p.materials[slot];
        uint class_id = p.opacity == PRODUCER_SECTIONS ? material_class(s.materials + material) : p.class_id;
        uint layers = class_id == CLASS_PARTICLE ? KIND_PARTICLE_LAYERS : KIND_NONE;
        write_run(s.run_producers + p.runs + slot, uint3(KIND_PATH, KIND_SHADOW, layers), hit_data(s, material, class_id));
    }
}
