// What a shader of a path's vertex loads of its slot at once (`load_vertex`), and the parts it
// stores back: every part a vertex's shaders may read, which a shader that reads few loads by
// word from the slot instead.
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
// the context they give. `t`, the length of the ray that reached the vertex (the camera ray's
// vertex's distance from the eye), is its caller's to set.
public struct LaunchVertex
{
    public uint slot;
    public uint4* at;                       // the slot's hot words
    public PathFixed f;
    public PathLoop p;
    public float t;
    public uint reach;
    public LaunchRay ray;                   // as the slot holds it
    public ShadeState c;
    public LaunchRecord record;
    public PathContext x;
}

public LaunchVertex load_vertex(PathRoot r, uint slot)
{
    LaunchVertex v;
    v.slot = slot;
    v.at = hot_words(r, slot);
    v.f.pixel = launch_pixel(r, slot);
    v.p = load_path_loop(r, v.at, v.f.pixel, v.reach);
    v.t = 0.0;
    v.ray = load_launch_ray(v.at);
    v.c = launch_state(layer_ray(v.ray));
    if (interiors_tracked(r))
    {
        LaunchInterior i = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(interior_words(r, slot));
        v.c.interior = i.interior;
        v.c.ray.start = i.start;
        v.p.entered = i.entered;
        v.p.entered_at = i.entered_at;
        v.p.entered_normal = i.entered_normal;
    }
    if (spectral_tracked(r))
        v.c.spectral = load_part<Spectral, SLOT_SPECTRAL_PART>(spectral_words(r, slot));
    v.record = launch_record(r, slot);
    v.x = path_context(r, v.record, v.f.pixel);
    if (stack_kept(r))
        v.c.stack = load_path_stack(r, slot, v.ray, v.c.interior, v.p.depth - 1, !v.x.photon && !v.x.one_technique);
    return v;
}

public void store_loop_part(LaunchVertex v)
{
    store_path_loop(v.at, v.p, v.reach);
}

public void store_ray_part(LaunchVertex v)
{
    store_launch_ray(v.at, launch_ray(v.c.ray, 0.0, 0));
}

public PackedSurface load_surface_part(LaunchVertex v)
{
    return load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT);
}

public PackedLayer load_layer_part(PathRoot r, LaunchVertex v)
{
    PackedLayer pl;
    pl.surface = load_surface_part(v);
    pl.layer = load_part<LayerWords, LAYER_WORDS>(layer_words(r, v.slot));
    return pl;
}

// Stores vertex `v`'s shadow ray `q`.
public void store_shadow_part(PathRoot r, LaunchVertex v, PathShadowRay q)
{
    store_shadow_ray(r, v.slot, q);
}

public LaunchWalk load_walk_part(PathRoot r, LaunchVertex v)
{
    return load_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot));
}
