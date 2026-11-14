// What the debug kernels read of the traced scene: a ray's closest hit as the path rays meet
// opaque and alpha-tested geometry, named by its copy, mesh, cluster and the cut the copy's
// PTLAS instance holds, and the scene's sun. Only debug kernels import it.
module traced_debug;

import orr_shaders.abi;
import orr_shaders.trace;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.surface;
import orr_scene.tables;
import orr_scene.visibility;
import orr_lighting.path;

public static const uint TRACED_NONE = 0xffffffff;

// A traced ray's closest hit.
public struct TracedHit
{
    public uint hit;              // 0 on a miss
    public uint class_id;         // CLASS_*
    public uint instance_index;   // PTLAS instance index
    public uint copy;             // slot copy, TRACED_NONE for producer entries
    public int cluster_id;        // -1 outside cluster BLAS
    public uint primitive;
    public uint mesh_cluster;     // TRACED_NONE outside cluster BLAS
    public uint model;
    public uint mesh;
    public uint rt_lod;           // SlotStateGpu.rt_lod of the copy's slot
    public float t;
    public float _pad;
    public float4 normal;         // the hit's geometric normal, render space
}

// The closest hit along `dir` from `origin` within `t_max` of the geometry in `mask`, its
// alpha-tested candidates decided by the surface's coverage, opacity micromaps forced to two
// states, as the path rays trace.
public TracedHit traced_hit<S : ISurfaces>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint mask)
{
    RayCone cone = { 0.0, 0.0 };
    HitContext context = hit_context(HIT_SECONDARY, 1, cone, 0.0);
    Alpha<S> a = { s, v, dir, context };
    float3 positions[3];
    Hit h = trace_closest(s.tlas, path_ray(origin, dir, t_max), mask, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
    TracedHit o = TracedHit();
    o.copy = TRACED_NONE;
    o.mesh_cluster = TRACED_NONE;
    o.model = TRACED_NONE;
    o.mesh = TRACED_NONE;
    o.rt_lod = TRACED_NONE;
    o.cluster_id = -1;
    if (!h.hit)
        return o;
    o.hit = 1;
    o.class_id = h.class_id;
    o.instance_index = h.instance_index;
    o.cluster_id = h.cluster_id;
    o.primitive = h.primitive_index;
    o.t = h.t;
    uint copy;
    if (h.cluster_id >= 0 && hit_copy(s, h.instance_index, copy))
    {
        o.copy = copy;
        InstanceGpu inst = copy_record(s, copy);
        o.model = inst.model;
        o.mesh = s.models[inst.model].mesh;
        o.mesh_cluster = s.clusters[h.cluster_id].mesh_cluster;
        o.rt_lod = s.slot_state[copy_slot(copy)].rt_lod;
    }
    if (h.triangle && h.cluster_id >= 0)
    {
        SurfaceHit sh = resolve_hit<S>(s, v, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, context);
        o.normal = float4(sh.surface.geometric_normal, 0.0);
    }
    return o;
}

// The direction toward the scene's first directional light; false without one.
public bool sun_direction(SceneRoot* s, out float3 dir)
{
    dir = float3(0.0);
    for (uint i = 0; i < s.light_capacity; i++)
    {
        if (s.lights[i].kind == LIGHT_DIRECTIONAL)
        {
            dir = s.lights[i].direction;
            return true;
        }
    }
    return false;
}
