// The runs of a scene's hit tables. A run is one section of a material row: a model's section
// materials, a set of a copy's material sets or a page's, or a producer entry's materials. It holds
// a hit record per ray type; a trace's record is its instance's contribution, RAY_TYPES times
// the first run of the row it shows, plus RAY_TYPES times the hit's geometry index (its cluster's
// section, or its producer cluster's material) plus the ray type. Runs are laid out as
// `SceneRoot.run_*` give: the medium proxies' and the light proxies' runs and the walk meet's,
// then a run per entry of `model_materials`, of `material_sets`, of the producer rows and of
// `page_material_sets`.
module runs;

import orr_shaders.trace;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.tables;
import orr_scene.hit;

// Ray types: a trace's hit record offset, with RAY_TYPES its stride.
public static const uint RAY_TYPE_PATH = 0;
public static const uint RAY_TYPE_SHADOW = 1;
public static const uint RAY_TYPE_LAYERS = 2;
public static const uint RAY_TYPES = 3;

// The contribution of an instance whose row starts at run `run`. Runs past what 24 bits of
// contribution reach hold no records, since `Frame::hit_table` refuses such a layout; their
// instances keep a valid contribution all the same.
public uint contribution(uint run)
{
    return (RAY_TYPES * run) & 0xffffff;
}

// The runs every medium proxy and every light proxy shows, and the run no instance shows, whose
// records a raygen sets on a traced hit to run a meet's closest-hit on it: the walk meet's at
// RAY_TYPE_PATH, the stack meet's at RAY_TYPE_SHADOW.
public static const uint RUN_MEDIUM = 0;
public static const uint RUN_LIGHT = 1;
public static const uint RUN_WALK = 2;
public static const uint FIXED_RUNS = 3;

// `HitData.flags`: the CLASS_* of the run's material in the low bits, then HIT_*.
public static const uint HIT_CLASS_MASK = 7;
public static const uint HIT_ALPHA_TESTED = 1 << 3;
public static const uint HIT_THIN_WALL = 1 << 4;      // a dielectric that bounds no interior
public static const uint HIT_CROSSES_INTERIOR = 1 << 5;
public static const uint HIT_EMITS = 1 << 6;
// The material or its back may give its surface a coat, fuzz, thin film or subsurface weight: it
// has one of their weights above 0, or a linked surface, which may set any.
public static const uint HIT_LOBED = 1 << 7;

// A hit record's data, after its group's handle: what its shaders start from.
public struct HitData
{
    public uint material;          // the section's MaterialId, MATERIAL_NONE for none
    public uint flags;             // HIT_* and the material's class
    public uint back;              // the material's back, MATERIAL_NONE for none
    public uint _spare;
}

// The data of a run of material `material` and class `class_id`.
public HitData hit_data(SceneRoot* s, uint material, uint class_id)
{
    HitData d = { material, class_id, MATERIAL_NONE, 0 };
    if (material == MATERIAL_NONE)
        return d;
    uint flags = s.materials[material].flags;
    if ((flags & MATERIAL_ALPHA_TESTED) != 0)
        d.flags |= HIT_ALPHA_TESTED;
    if (class_id == CLASS_DIELECTRIC)
        d.flags |= crosses_interior(s, material) ? HIT_CROSSES_INTERIOR : HIT_THIN_WALL;
    if ((flags & MATERIAL_EMISSIVE) != 0)
        d.flags |= HIT_EMITS;
    if ((flags & MATERIAL_HAS_BACK) != 0)
        d.back = s.materials[material].back;
    if (material_lobed(s.materials + material) || (d.back != MATERIAL_NONE && material_lobed(s.materials + d.back)))
        d.flags |= HIT_LOBED;
    return d;
}

bool material_lobed(MaterialGpu* m)
{
    return m.surface != 0 || m.coat_weight > 0.0 || m.fuzz_weight > 0.0 || m.thin_film_weight > 0.0 || m.subsurface_weight > 0.0;
}

// The data of the medium or light proxies' run.
public HitData proxy_data(uint class_id)
{
    HitData d = { MATERIAL_NONE, class_id, MATERIAL_NONE, 0 };
    return d;
}

// The first run of the row copy record `inst` (at its chain step) shows: run `first + section`
// holds `section_material(s, inst, model, section)`.
public uint copy_run(SceneRoot* s, InstanceGpu inst)
{
    ModelGpu* model = s.models + inst.model;
    if (inst.material_base == NO_MATERIAL_SET)
        return s.run_models + model.materials;
    uint at = ((inst.variant >> 8) & 0xff) * model.section_count;
    if ((inst.material_base & PAGE_MATERIAL_SET) != 0)
        return s.run_pages + (inst.material_base & ~PAGE_MATERIAL_SET) + at;
    return s.run_sets + inst.material_base + at;
}

// The first run of producer record `p`'s row: run `first + material` holds its material.
public uint producer_run(SceneRoot* s, ProducerGpu p)
{
    return s.run_producers + p.runs;
}
