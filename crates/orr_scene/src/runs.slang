// The runs of a scene's hit tables. A run is one section of a material row: a model's section
// materials, one set of a copy's material set or of a page's, or a producer entry's materials. It
// holds a hit record per ray type; a trace's record is its instance's contribution, RAY_TYPES times
// the first run of the row it shows, plus RAY_TYPES times the hit's geometry index (its cluster's
// section, or its producer cluster's material) plus the ray type. A row takes its runs from the
// scene's one allocator when it is made and keeps them until it goes, so a table that grows moves
// no run and no instance takes another contribution for it. `SceneRoot.runs` says per run where
// its material is and what its geometry is: the hit tables' kernels write records from it.
module runs;

import orr_shaders.abi;
import orr_shaders.trace;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.tables;
import orr_scene.hit;

// Ray types: a trace's hit record offset, with RAY_TYPES its stride.
public static const uint RAY_TYPE_PATH = 0;
public static const uint RAY_TYPE_SHADOW = 1;
[Host] public static const uint RAY_TYPES = 2;

// The contribution of an instance whose row starts at run `run`. Runs past what 24 bits of
// contribution reach hold no records, since `Frame::hit_table` refuses such a layout; their
// instances keep a valid contribution all the same.
public uint contribution(uint run)
{
    return (RAY_TYPES * run) & 0xffffff;
}

// The runs every medium proxy and every light proxy shows. Every scene takes them first.
public static const uint RUN_MEDIUM = 0;
public static const uint RUN_LIGHT = 1;
[Host] public static const uint FIXED_RUNS = 2;

// What a run serves (`RunGpu.word` bits 0-3).
[Host] public static const uint ROW_NONE = 0;     // nothing: a record of zeros, as free runs hold
[Host] public static const uint ROW_FIXED = 1;    // RUN_MEDIUM or RUN_LIGHT, by its index
[Host] public static const uint ROW_MODEL = 2;    // a copy's section of `model_materials[source]`
[Host] public static const uint ROW_SET = 3;      // a copy's section of `material_sets[source]`
[Host] public static const uint ROW_PAGE = 4;     // a copy's section of `page_material_sets[source]`
[Host] public static const uint ROW_PRODUCER = 5; // material `word >> 16` of producer entry `source`
[Host] public static const uint ROW_KIND_MASK = 15;
// A copy's row: the CLASS_* its section cooked with in bits 4-6, and ROW_SOLID where the section
// is solid glass (`CLUSTER_SOLID`). Its clusters trace by them, whatever material a set shows.
[Host] public static const uint ROW_CLASS_SHIFT = 4;
[Host] public static const uint ROW_SOLID = 1 << 7;
// A copy's row: its mesh's MESH_* vertex attributes in bits 8-15, which its coverage reads.
[Host] public static const uint ROW_ATTRIBUTES_SHIFT = 8;
// A producer's row: the material slot in bits 16-31.
[Host] public static const uint ROW_SLOT_SHIFT = 16;

// `HitData.flags`: the CLASS_* its candidates and hits take in the low bits, then HIT_*. The class
// is what the hit's geometry traces as: a copy's section's cooked class, which a material set
// does not change, or a producer entry's (the material's with PRODUCER_SECTIONS). Candidates are
// filtered by it, as the instance masks and opacity follow it, and the vertex shades as it.
public static const uint HIT_CLASS_MASK = 7;
public static const uint HIT_ALPHA_TESTED = 1 << 3;
public static const uint HIT_THIN_WALL = 1 << 4;      // of the dielectric class, its material bounds no interior
public static const uint HIT_CROSSES_INTERIOR = 1 << 5;
public static const uint HIT_EMITS = 1 << 6;
// The material or its back may give its surface a coat, fuzz, thin film or subsurface weight
// (`MaterialGpu.lobes`): it has one of their weights above 0, or its linked surface type
// declares one.
public static const uint HIT_LOBED = 1 << 7;
// Solid glass, a section cooked as a dielectric with an interior: a plane stack's ray stops at it.
public static const uint HIT_SOLID = 1 << 8;

// A hit record's data, after its group's handle: what its shaders start from, HIT_DATA_BYTES of it.
[Host] public static const uint HIT_DATA_BYTES = 16;
public struct HitData
{
    public uint material;          // the section's MaterialId, MATERIAL_NONE for none
    public uint flags;             // HIT_* and the class
    public uint back;              // the material's back, MATERIAL_NONE for none
    public uint attributes;        // a copy's mesh's MESH_* vertex attributes, which its coverage reads; 0 on producer geometry

    public uint class_id()
    {
        return flags & HIT_CLASS_MASK;
    }

    public bool has(uint flag)
    {
        return (flags & flag) != 0;
    }
}

// The data of a run of material `material` on geometry of class `class_id` with vertex
// `attributes`, solid glass where `solid`.
public HitData hit_data(SceneRoot* s, uint material, uint class_id, bool solid, uint attributes)
{
    static_assert(sizeof(HitData) == HIT_DATA_BYTES, "a run's data is not HIT_DATA_BYTES");
    HitData d = { material, class_id | (solid ? HIT_SOLID : 0), MATERIAL_NONE, attributes };
    if (material == MATERIAL_NONE)
        return d;
    uint flags = s.materials[material].flags;
    if ((flags & MATERIAL_ALPHA_TESTED) != 0)
        d.flags |= HIT_ALPHA_TESTED;
    if (class_id == CLASS_DIELECTRIC)
        d.flags |= crosses_interior(s, material) ? HIT_CROSSES_INTERIOR : HIT_THIN_WALL;
    if ((flags & MATERIAL_HAS_BACK) != 0)
        d.back = s.materials[material].back;
    if (material_emits(s.materials + material) || (d.back != MATERIAL_NONE && material_emits(s.materials + d.back)))
        d.flags |= HIT_EMITS;
    if (material_lobed(s.materials + material) || (d.back != MATERIAL_NONE && material_lobed(s.materials + d.back)))
        d.flags |= HIT_LOBED;
    return d;
}

// Whether a surface of `m` may emit: its own emission is not zero, or its linked surface type
// declares one.
bool material_emits(MaterialGpu* m)
{
    return (m.lobes & MATERIAL_LOBE_EMISSION) != 0;
}

bool material_lobed(MaterialGpu* m)
{
    return (m.lobes & MATERIAL_LOBES_RICH) != 0;
}

// The data of the medium or light proxies' run.
public HitData proxy_data(uint class_id)
{
    HitData d = { MATERIAL_NONE, class_id, MATERIAL_NONE, 0 };
    return d;
}

// The first run of the row copy record `inst` (at its chain step) shows: run `first + section`
// holds `section_material(s, inst, model, section)`. A record past chain step 0 shows its model's
// own materials.
public uint copy_run(SceneRoot* s, InstanceGpu inst)
{
    return inst.material_base == NO_MATERIAL_SET ? s.model_runs[inst.model] : inst.run;
}

// The first run of producer record `p`'s row: run `first + material` holds its material.
public uint producer_run(SceneRoot* s, ProducerGpu p)
{
    return p.runs;
}
