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
// pass or `second`: the pixel's radiance, which holds the sky its camera ray met, as the pixel's
// outputs in the first pass, whose radiance texel is in `texel`, and as its background's luminance
// in the second. True where the pixel's second pass follows, with `record` readied for it.
bool view_pass_done(PathRoot r, uint2 pixel, PathLoop p, inout StoredView record, bool second, out float4 texel)
{
    ViewConstants* v = r.view;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float3 pixel_radiance = view_exposed(record) + pre_exposure * p.radiance;
    texel = float4(0.0);
    if (second)
    {
        if (frost_pass(r))
            r.layers[index].background = guide_luminance(pixel_radiance);
        return false;
    }
    PathFixed f = { pixel };
    texel = view_outputs(r, f, p, record, pixel_radiance);
    if (!two_passes(r, pixel))
        return false;
    // The second pass's path takes the G-buffer's emission anew.
    record.own = float3(0.0);
    return true;
}

// The second pass of a view's `pixel` from the G-buffer's surface, inside what holds it, to flip
// at the crossings `slot`'s finish asks for with the STEP_CROSS this returns.
uint second_pass_start(PathRoot r, uint slot, uint2 pixel, out PathStart start)
{
    PathStart camera_ray = camera_start(r, pixel);
    start = PathStart();
    uint cross = 0;
    if (thin_lens(r))
        cross = lens_crossed(r, slot, camera_ray.origin, start.interior);
    else
    {
        start = pinhole_start(r.view, r.gbuffer, pixel, true);
        start.interior = Interior(r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].near_interior);
        if (isfinite(start.t_max))
            cross = ask_crossings(r, slot, start.origin, start.dir, start.t_max);
    }
    start.dir = camera_ray.dir;
    start.throughput = float3(1.0);
    start.mask = 0;
    start.gbuffer = true;
    return cross;
}

// What a view does with `slot`'s finished path in pass `pass` (`view_pass_end`). A frosted
// pixel's first pass starts the second from the G-buffer's surface, with the sampler, rays and
// record the first left: the step it starts with.
public uint launch_finish_view(PathRoot r, uint slot, uint pass)
{
    uint4* at = hot_words(r, slot);
    uint2 pixel = launch_pixel(r, slot);
    LaunchRecord record = launch_record(r, slot);
    PathStart start;
    PathLoop p = load_path_loop(r, at, pixel);
    bool second = pass != 0;
    // The pixel's rays count both passes: the second reads the texel the first pass's shader
    // wrote.
    RWTexture2D<float4> radiance = r.radiance.get<RWTexture2D<float4>>();
    bool rays = path_mode(r) == PATH_RAYS;
    if (second && rays)
        radiance[pixel] = float4(float(p.rays), load_texel_coherent(radiance, pixel).yzw);
    StoredView view = load_part<StoredView, RECORD_WORDS>(record.words);
    float4 texel;
    if (!view_pass_done(r, pixel, p, view, second, texel))
        return STEP_DONE;
    if (rays)
        store_texel_coherent(radiance, pixel, texel);
    store_part<StoredView, RECORD_WORDS>(record.words, view);
    uint cross = second_pass_start(r, slot, pixel, start);
    return begin_launch(r, slot, pixel, start, p.rng, p.rays, record) | cross;
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

// What the background layer does with `slot`'s finished path: its texel, the pre-exposed
// radiance, which holds the sky its camera ray met, in RGB and the reverse-Z depth of its first
// vertex in A (0 where it met nothing).
public uint launch_finish_background(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    ViewConstants* v = r.view;
    float3 radiance = load_word<LoopRadiance>(hot_words(r, slot) + LOOP_RADIANCE_AT).radiance;
    StoredView record = load_part<StoredView, RECORD_WORDS>(launch_record(r, slot).words);
    float depth = 0.0;
    if (record.first_distance > 0.0)
        depth = v.clip_from_view[2][3] / (camera_start(r, pixel).axis * view_first_t(record));
    r.dof_background.get<RWTexture2D<float4>>()[pixel] = float4(apply_hdr_ceiling(v.exposure[v.exposure_slot].pre_exposure * radiance), depth);
    return STEP_DONE;
}
