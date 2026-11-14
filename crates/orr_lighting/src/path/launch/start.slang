// The starts of a path, by source, and the G-buffer's surface as its first vertex.
module start;

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
import orr_lighting.path.launch.steps;
import orr_lighting.path.launch.slot;
import orr_lighting.path.launch.record;
import orr_lighting.path.launch.load;
import orr_lighting.path.launch.shade_surface;

// The G-buffer's surface as the path's first vertex, reached with `p` and its ray `ray`, which
// the path starts on where `start`, else reaches by its camera ray: STEP_VERTEX with STEP_EMITS
// where it emits, or STEP_ENDED for a sky pixel, which leaves the sky.
uint reach_gbuffer(PathRoot r, uint4* at, inout PathLoop p, LayerRay ray, inout LaunchRecord record, uint2 pixel, bool start)
{
    set_flag(p.flags, LOOP_GBUFFER_START, false);
    if (!start)
        p.iteration++;
    PathFixed f = { pixel };
    PackedSurface ps;
    if (!arrive_gbuffer(r, f, p, record, start, ps, ray.dir))
    {
        store_part<LaunchLoop, LOOP_WORDS>(at + LOOP_AT, launch_loop_from(p, 0.0, 0));
        return STEP_ENDED;
    }
    store_part<PackedSurface, SURFACE_WORDS>(at + SURFACE_AT, ps);
    store_part<LaunchRay, RAY_WORDS>(at + RAY_AT, launch_ray(ray, 0.0, 0));
    store_part<LaunchLoop, LOOP_WORDS>(at + LOOP_AT, launch_loop_from(p, 0.0, REACH_FIRST | REACH_GBUFFER));
    return vertex_step(ps);
}

// Starts `slot`'s path of `pixel` from `start` with `rng`, counting `rays` rays already: its
// camera ray, or its first vertex on the G-buffer's surface, resolved here.
public uint begin_launch(PathRoot r, uint slot, uint2 pixel, PathStart start, Sampler rng, uint rays, inout LaunchRecord record)
{
    PathFixed f;
    PathLoop p;
    ShadeState c;
    start_state(r, pixel, start, rng, record.photon(), f, p, c);
    p.rays = rays;
    uint4* at = slot_words(r, slot);
    store_tracked_parts(r, at, p, c);
    store_spectral_part(r, at, c.spectral);
    if (start.mask != 0)
    {
        store_part<LaunchRay, RAY_WORDS>(at + RAY_AT, launch_ray(c.ray, start.t_max, start.mask, start.gbuffer ? LAUNCH_GBUFFER : 0));
        store_part<LaunchLoop, LOOP_WORDS>(at + LOOP_AT, launch_loop_from(p, 0.0, 0));
        return STEP_RAY;
    }
    return reach_gbuffer(r, at, p, c.ray, record, pixel, true);
}

// The start of a view's path of the slice's `slot`th pixel (`view_start`). Debug views of the
// first vertex alone write the pixel here.
public uint launch_start_view<S : ISurfaces>(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    Sampler rng = path_sampler(r, pixel);
    if (first_vertex_mode(r))
    {
        first_vertex_view<S>(r, pixel, rng);
        return STEP_DONE;
    }
    PathStart start = view_start<S>(r, pixel, rng);
    LaunchRecord record = launch_record(r, slot);
    store_part<StoredView, RECORD_WORDS>(record.words, view_record_start(r, pixel));
    return begin_launch(r, slot, pixel, start, rng, 0, record);
}

// The update path of the slice's `slot`th block (`update_path_start`).
public uint launch_start_update<S : ISurfaces>(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    if (any(pixel >= r.gbuffer.extent))
        return STEP_DONE;
    LaunchRecord record = launch_record(r, slot);
    store_part<CacheUpdate, UPDATE_RECORD_WORDS>(record.words, CacheUpdate());
    return begin_launch(r, slot, pixel, update_path_start<S>(r, pixel), path_sampler(r, pixel), 0, record);
}

// The background layer's path of the slice's `slot`th pixel (`background_path_start`).
public uint launch_start_background(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    Sampler rng = path_sampler(r, pixel);
    PathStart start;
    if (!background_path_start(r, pixel, rng, start))
        return STEP_DONE;
    LaunchRecord record = launch_record(r, slot);
    store_part<StoredView, RECORD_WORDS>(record.words, view_record_start(r, pixel));
    return begin_launch(r, slot, pixel, start, rng, 0, record);
}

// Caustic photon `r.slice_base + slot`: its start up to its shadow ray toward its light, which
// the raygen traces before the photon's first vertex resolves from its given hit, adjoint,
// depositing in `r.caustics`.
public uint launch_start_photon<S : ISurfaces>(PathRoot r, uint slot)
{
    uint index = r.slice_base + slot;
    if (index >= r.caustics.photon_count || !(r.caustics.source.total > 0.0))
        return STEP_DONE;
    uint2 pixel = launch_pixel(r, slot);
    Sampler rng = path_sampler(r, pixel);
    PhotonStart p;
    PhotonRay light;
    if (!photon_emit<S>(r, index, rng, p, light))
        return STEP_DONE;
    LaunchRecord record = launch_record(r, slot);
    store_part<CausticPhoton, RECORD_WORDS>(record.words, CausticPhoton(p.entry));
    begin_launch(r, slot, pixel, photon_path_start(p), rng, 0, record);
    uint4* at = slot_words(r, slot);
    ValueHit given = value_hit(p.hit, p.positions, p.dir);
    store_part<ValueHit, GIVEN_WORDS>(at + SURFACE_AT, given);
    // What the interiors around the start take of the flux on the way to the light, as its
    // trace in mode SHADOW_PHOTON takes it beside the candidates' product.
    float scale;
    float3 beer = exp(-straight_depth(r.scene, p.interior, light.t_max, false, scale));
    ShadowJob job = { beer, NO_LIGHT, light.origin, light.t_max, light.dir, 0 };
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, job);
    return STEP_GIVEN;
}

// The G-buffer's surface as the vertex of `slot`'s camera ray, which found no layer before it.
public uint launch_gbuffer(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    return reach_gbuffer(r, v.at, v.p, v.c.ray, v.record, v.f.pixel, false);
}
