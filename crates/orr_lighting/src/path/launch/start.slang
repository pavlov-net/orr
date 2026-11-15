// The starts of a path, by source, and the G-buffer's surface as its first vertex. A start whose
// interiors depend on the surfaces a ray crosses asks the raygen for them (STEP_CROSS) and folds
// them in when it runs again.
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
import orr_lighting.path.launch.boundary;
import orr_lighting.path.launch.hit;
import orr_lighting.path.launch.shade_surface;

// The ray of hot words `at` as its trace half gives it: its origin, its direction and its flags,
// with no MIS half.
public LayerRay traced_ray(uint4* at)
{
    RayOrigin o = load_word<RayOrigin>(at + RAY_ORIGIN_AT);
    RayDir d = load_word<RayDir>(at + RAY_DIR_AT);
    LaunchRay v = { o.origin, d.word, d.dir, o.reach, float3(0.0), 0.0, float3(0.0), 0.0 };
    return layer_ray(v);
}

// The G-buffer's surface as the path's first vertex, reached with `p`, which the path starts on
// where `start`, else reaches by its camera ray: STEP_VERTEX, with STEP_EMITS where its emission
// counts and STEP_UNSHADED where its pixel, in no interior, runs no path this frame; or for a sky
// pixel STEP_SKY, with the camera ray's direction in the ray words, where the path is a view's,
// else STEP_ENDED. It stores the loop.
uint reach_gbuffer(PathRoot r, uint4* at, inout PathLoop p, inout LaunchRecord record, uint2 pixel, bool start)
{
    PathFixed f = { pixel };
    PackedSurface ps;
    float3 dir;
    if (!arrive_gbuffer(r, f, p, record, start, ps, dir))
    {
        store_path_loop(at, p);
        // Only a view shows the sky.
        return from_view(r) ? STEP_SKY : STEP_ENDED;
    }
    store_part<PackedSurface, SURFACE_WORDS>(at + SURFACE_AT, ps);
    store_path_loop(at, p);
    uint step = vertex_step() | STEP_FIRST | STEP_GBUFFER;
    if (may(MAY_EMISSIVE) && surface_emits(ps) && has(p.flags, LOOP_COUNTED))
        step |= STEP_EMITS;
    if (!has(p.flags, LOOP_INSIDE) && !runs_path(r, pixel))
        step |= STEP_UNSHADED;
    return step;
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
    uint4* at = hot_words(r, slot);
    if (c.interior.count > 0)
        store_interiors(r, slot, p, c);
    // Only a photon starts with a plane stack, the one its light's direction gives.
    if (c.stack.valid)
        keep_path_stack(r, slot, p, c.stack);
    else
        set_flag(p.flags, LOOP_STACK_RAY, true);
    if (start.mask != 0)
    {
        store_path_ray(r, at, p, c, record, path_context(r, record, pixel), start.t_max, start.mask, start.gbuffer ? LAUNCH_GBUFFER : 0, start.offset);
        store_loop_state(at, p);
        store_loop_radiance(at, p);
        return STEP_RAY;
    }
    store_launch_ray(at, launch_ray(c.ray, 0.0, 0));
    return reach_gbuffer(r, at, p, record, pixel, true);
}

// Asks for the crossings of the ray from `origin` along unit `dir` to `t_max`, which the interiors
// `slot`'s path starts in take once the raygen has traced them: STEP_CROSS for its start's step. A
// root that tracks no interiors has no surface to cross.
public uint ask_crossings(PathRoot r, uint slot, float3 origin, float3 dir, float t_max)
{
    if (!interiors_tracked(r))
        return 0;
    store_crossing_ray(r, slot, origin, dir, t_max);
    return STEP_CROSS;
}

// The interiors at `origin`, a point of the lens, in `interior` (`lens_interior`): `camera_medium`,
// to flip at each surface between the eye and it, which `slot`'s start asks for (STEP_CROSS).
public uint lens_crossed(PathRoot r, uint slot, float3 origin, out Interior interior)
{
    interior = Interior();
    if (r.camera_medium != MATERIAL_NONE)
        interior.cross(r.camera_medium, true);
    float3 eye = r.view.eye_render.xyz;
    float t = length(origin - eye);
    return ask_crossings(r, slot, eye, (origin - eye) / t, t);
}

// The interiors around `pixel`'s G-buffer surface, which its `pinhole_start` ray `start` reaches, in
// `start.interior` (`gbuffer_interior`): on a tile the layer guides traced, those at its near-plane
// point, to flip at each surface the ray crosses before it, which `slot`'s start asks for
// (STEP_CROSS); elsewhere none.
uint gbuffer_crossed(PathRoot r, uint slot, uint2 pixel, inout PathStart start)
{
    start.interior = Interior();
    LayerGuide* layers = r.layers;
    uint2 extent = r.gbuffer.extent;
    if (layers == nullptr || traced_tiles(layers, extent)[layer_tile(extent, pixel)] == 0)
        return 0;
    start.interior = Interior(layers[pixel.y * extent.x + pixel.x].near_interior);
    if (!isfinite(start.t_max))
        return 0;
    return ask_crossings(r, slot, start.origin, start.dir, start.t_max);
}

// The start of a view's path of the slice's `slot`th pixel (`view_start`): through the thin lens,
// inside the interiors at its point of the lens; on a tile the layer guides traced, a camera ray
// that looks for layers before the G-buffer's surface, inside the interiors at its near-plane
// point; else from the G-buffer's surface.
public uint launch_start_view(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    Sampler rng = path_sampler(r, pixel);
    PathStart start;
    uint cross = 0;
    if (thin_lens(r))
    {
        start = lens_start(r, pixel, r.aperture_radius, r.focus_distance, 0.0, MASK_OPAQUE | MASK_DIELECTRIC, rng);
        cross = lens_crossed(r, slot, start.origin, start.interior);
    }
    else
    {
        bool traced = pixel_traced(r, pixel);
        start = pinhole_start(r.view, r.gbuffer, pixel, traced);
        if (traced)
            start.interior = Interior(r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].near_interior);
    }
    LaunchRecord record = launch_record(r, slot);
    store_part<StoredView, RECORD_WORDS>(record.words, view_record_start(r, pixel));
    return begin_launch(r, slot, pixel, start, rng, 0, record) | cross;
}

// A view's debug view of the first vertex alone of the slice's `slot`th pixel, which it writes.
public uint launch_start_first_vertex<S : ISurfaces>(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    Sampler rng = path_sampler(r, pixel);
    first_vertex_view<S>(r, pixel, rng);
    return STEP_DONE;
}

// The update path of the slice's `slot`th block (`update_path_start`): from its G-buffer surface,
// inside the interiors around it.
public uint launch_start_update(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    if (any(pixel >= r.gbuffer.extent))
        return STEP_DONE;
    PathStart start = pinhole_start(r.view, r.gbuffer, pixel, false);
    uint cross = gbuffer_crossed(r, slot, pixel, start);
    LaunchRecord record = launch_record(r, slot);
    store_part<CacheUpdate, UPDATE_RECORD_WORDS>(record.words, CacheUpdate());
    return begin_launch(r, slot, pixel, start, path_sampler(r, pixel), 0, record) | cross;
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
public uint launch_start_photon<S : ISurfaces, M : IMedium>(PathRoot r, uint slot)
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
    uint4* at = hot_words(r, slot);
    ValueHit given = value_hit(p.hit, p.positions, p.dir);
    store_part<ValueHit, GIVEN_WORDS>(at + SURFACE_AT, given);
    // What the interiors around the start and the linked medium take of the flux on the way to
    // the light, beside what its trace in mode SHADOW_PHOTON lets through.
    float scale;
    float3 beer = exp(-straight_depth(r.scene, p.interior, light.t_max, false, scale)) * medium_transmittance<M>(r.scene.medium, light.origin, light.dir, light.t_max);
    ShadowJob job = { beer, NO_LIGHT, light.origin, light.t_max, light.dir, 0 };
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, job);
    return STEP_GIVEN;
}

// The G-buffer's surface as the vertex of `slot`'s camera ray, which found no layer before it. It
// loads the loop and the ray's trace half, whose reach and launch bits it clears.
public uint launch_gbuffer(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot);
    uint2 pixel = launch_pixel(r, slot);
    PathLoop p = load_path_loop(r, at, pixel);
    LaunchRay v = launch_ray(traced_ray(at), 0.0, 0);
    RayOrigin o = { v.origin, v.t_max };
    RayDir d = { v.dir, v.flags };
    store_word(at + RAY_ORIGIN_AT, o);
    store_word(at + RAY_DIR_AT, d);
    LaunchRecord record = launch_record(r, slot);
    return reach_gbuffer(r, at, p, record, pixel, false);
}
