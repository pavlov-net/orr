// The finishes, each STEP_DONE but where a path runs again.
module finish;

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
import orr_lighting.path.launch.start;

// The end of the path of a view's pixel `pixel` with loop `p` and record `record`, in its first
// pass or `second` (`view_pass_end` up to its second pass's start): the pixel's radiance, with
// the sky its camera ray met where the mode shows it, as the pixel's outputs in the first pass,
// whose radiance texel is in `texel`, and as its background's luminance in the second. True where
// the pixel's second pass follows, with `record` readied for it.
bool view_pass_done<E : IEnvironment, M : IMedium>(PathRoot r, uint2 pixel, PathLoop p, inout StoredView record, bool second, out float4 texel)
{
    ViewConstants* v = r.view;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float3 pixel_radiance = view_exposed(record) + pre_exposure * p.radiance;
    bool sky = any(p.sky > 0.0) && path_mode(r) != PATH_INDIRECT && path_mode(r) != PATH_SUBSURFACE;
    bool two = two_passes(r, pixel);
    texel = float4(0.0);
    if (sky)
        pixel_radiance += pre_exposure * p.sky * sky_radiance<E, M>(r.set, v.eye_render.xyz, camera_start(r, pixel).dir);
    if (second)
    {
        if (frost_pass(r))
            r.layers[index].background = guide_luminance(pixel_radiance);
        return false;
    }
    PathFixed f = { pixel };
    texel = view_outputs(r, f, p, record, pixel_radiance);
    if (!two)
        return false;
    // The second pass's path takes the G-buffer's emission anew.
    record.own = float3(0.0);
    return true;
}

// The second pass of a view's `pixel` from the G-buffer's surface, inside what holds it: on its
// first call (`crossings` 0) false where the interiors around it wait on crossings `slot`'s finish
// asks for, which its next call folds in.
bool second_pass_start(PathRoot r, uint slot, uint crossings, uint2 pixel, out PathStart start)
{
    PathStart camera_ray = camera_start(r, pixel);
    start = PathStart();
    bool crossed;
    if (thin_lens(r))
        crossed = lens_crossed(r, slot, crossings, camera_ray.origin, start.interior);
    else
    {
        start = pinhole_start(r.view, r.gbuffer, pixel, true);
        start.interior = Interior(r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].near_interior);
        crossed = !isfinite(start.t_max) || start_crossed(r, slot, crossings, start.origin, start.dir, start.t_max, start.interior);
    }
    start.dir = camera_ray.dir;
    start.throughput = float3(1.0);
    start.mask = 0;
    start.gbuffer = true;
    return crossed;
}

// What a view does with `slot`'s finished path in pass `pass` (`view_pass_end`). A frosted
// pixel's first pass starts the second from the G-buffer's surface, with the sampler, rays and
// record the first left: the step it starts with, once the crossings it may ask for are traced
// (`crossings`, data word 2), on which it runs only that start.
public uint launch_finish_view<E : IEnvironment, M : IMedium>(PathRoot r, uint slot, uint pass, uint crossings)
{
    uint4* at = hot_words(r, slot);
    uint2 pixel = launch_pixel(r, slot);
    LaunchRecord record = launch_record(r, slot);
    PathStart start;
    if (crossings != 0)
    {
        second_pass_start(r, slot, crossings, pixel, start);
        Sampler rng = path_sampler(r, pixel);
        rng.dimension = load_word<LoopRadiance>(at + LOOP_RADIANCE_AT).dimension;
        return begin_launch(r, slot, pixel, start, rng, load_word<LoopCounts>(at + LOOP_COUNTS_AT).rays, record);
    }
    uint reach;
    PathLoop p = load_path_loop(r, at, pixel, reach);
    bool second = pass != 0;
    // The pixel's rays count both passes: the second reads the texel the first pass's shader
    // wrote.
    RWTexture2D<float4> radiance = r.radiance.get<RWTexture2D<float4>>();
    bool rays = path_mode(r) == PATH_RAYS;
    if (second && rays)
        radiance[pixel] = float4(float(p.rays), load_texel_coherent(radiance, pixel).yzw);
    StoredView view = load_part<StoredView, RECORD_WORDS>(record.words);
    float4 texel;
    if (!view_pass_done<E, M>(r, pixel, p, view, second, texel))
        return STEP_DONE;
    if (rays)
        store_texel_coherent(radiance, pixel, texel);
    store_part<StoredView, RECORD_WORDS>(record.words, view);
    if (!second_pass_start(r, slot, 0, pixel, start))
        return STEP_CROSS;
    return begin_launch(r, slot, pixel, start, p.rng, p.rays, record);
}

// The end of an update path, whose record adds to the cache, or of a photon, which its entry's
// yield counts where it deposited. It loads the loop's radiance and counts.
public uint launch_finish_record(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot);
    float3 radiance = load_word<LoopRadiance>(at + LOOP_RADIANCE_AT).radiance;
    LaunchRecord record = launch_record(r, slot);
    record.finish(r, radiance, load_word<LoopCounts>(at + LOOP_COUNTS_AT).layers);
    return STEP_DONE;
}

// What the background layer does with `slot`'s finished path: its texel, the pre-exposed radiance
// in RGB and the reverse-Z depth of its hit in A (0 on a miss). It loads the loop's radiance.
public uint launch_finish_background<E : IEnvironment, M : IMedium>(PathRoot r, uint slot)
{
    float3 radiance = load_word<LoopRadiance>(hot_words(r, slot) + LOOP_RADIANCE_AT).radiance;
    LaunchRecord record = launch_record(r, slot);
    background_texel<E, M>(r, launch_pixel(r, slot), load_part<StoredView, RECORD_WORDS>(record.words), radiance);
    return STEP_DONE;
}
