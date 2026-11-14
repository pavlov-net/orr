// What a shader of a path's vertex loads of its slot, and the parts it stores back.
module load;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_shaders.pipeline;
import orr_color.luminance;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.runs;
import orr_scene.tables;
import orr_scene.candidates;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.froxels;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.subsurface;
import orr_lighting.interiors;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.launch.slot;
import orr_lighting.path.launch.record;

// What a shader of a path's vertex loads: its pixel, loop, ray, interiors, wavelengths, record and
// the context they give.
public struct LaunchVertex
{
    public uint4* at;
    public PathFixed f;
    public PathLoop p;
    public float t;
    public uint reach;
    public LaunchRay ray;                  // as the slot holds it
    public ShadeState c;
    public LaunchRecord record;
    public PathContext x;
}

public LaunchVertex load_vertex(PathRoot r, uint slot)
{
    LaunchVertex v;
    v.at = slot_words(r, slot);
    v.f.pixel = launch_pixel(r, slot);
    LaunchLoop l = load_part<LaunchLoop, LOOP_WORDS>(v.at + LOOP_AT);
    v.p = launch_loop_of(r, l, v.f.pixel);
    v.t = l.t;
    v.reach = l.reach;
    v.ray = load_part<LaunchRay, RAY_WORDS>(v.at + RAY_AT);
    v.c = launch_state(layer_ray(v.ray));
    if (interiors_tracked(r))
    {
        LaunchInterior i = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(v.at + interior_at(r));
        v.c.interior = i.interior;
        v.c.ray.start = i.start;
        v.p.entered = i.entered;
        v.p.entered_at = i.entered_at;
        v.p.entered_normal = i.entered_normal;
    }
    if (stack_kept(r))
        v.c.stack = load_part<PathStack, PATH_STACK_WORDS>(v.at + stack_at(r));
    if (spectral_tracked(r))
        v.c.spectral = load_part<Spectral, SLOT_SPECTRAL_PART>(v.at + spectral_at(r));
    v.record = launch_record(r, slot);
    v.x = path_context(r, v.record, v.f.pixel);
    return v;
}

public void store_loop_part(LaunchVertex v)
{
    store_part<LaunchLoop, LOOP_WORDS>(v.at + LOOP_AT, launch_loop_from(v.p, v.t, v.reach));
}

public void store_ray_part(LaunchVertex v)
{
    store_part<LaunchRay, RAY_WORDS>(v.at + RAY_AT, launch_ray(v.c.ray, 0.0, 0));
}

public PackedSurface load_surface_part(LaunchVertex v)
{
    return load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT);
}

public PackedLayer load_layer_part(PathRoot r, LaunchVertex v)
{
    PackedLayer pl;
    pl.surface = load_surface_part(v);
    pl.layer = load_part<LayerWords, LAYER_WORDS>(v.at + layer_at(r));
    return pl;
}

// Stores vertex `v`'s shadow ray `q`.
public void store_shadow_part(PathRoot r, LaunchVertex v, PathShadowRay q)
{
    ShadowJob job = { q.contribution, q.light, q.origin, q.t_max, q.dir, q.flags };
    store_part<ShadowJob, SHADOW_WORDS>(v.at + SHADOW_AT, job);
    if (interiors_tracked(r))
    {
        ShadowInterior i = { q.start, q.surface, q.point };
        store_part<ShadowInterior, SLOT_SHADOW_INTERIOR_WORDS>(v.at + interior_at(r) + SLOT_INTERIOR_WORDS, i);
    }
}

public LaunchWalk load_walk_part(LaunchVertex v)
{
    return load_part<LaunchWalk, WALK_STATE_WORDS>(v.at + WALK_AT);
}
