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
import orr_lighting.path.launch.load;
import orr_lighting.path.launch.start;

// What a view does with `slot`'s finished path in pass `pass` (`view_pass_end`). A frosted
// pixel's first pass starts the second from the G-buffer's surface, with the sampler, rays and
// record the first left: the step it starts with.
public uint launch_finish_view<S : ISurfaces, E : IEnvironment, M : IMedium>(PathRoot r, uint slot, uint pass)
{
    LaunchVertex v = load_vertex(r, slot);
    uint2 pixel = v.f.pixel;
    bool second = pass != 0;
    // The pixel's rays count both passes: the second reads the texel the first pass's shader
    // wrote.
    RWTexture2D<float4> radiance = r.radiance.get<RWTexture2D<float4>>();
    bool rays = path_mode(r) == PATH_RAYS;
    if (second && rays)
        radiance[pixel] = float4(float(v.p.rays), load_texel_coherent(radiance, pixel).yzw);
    StoredView record = load_part<StoredView, RECORD_WORDS>(v.record.words);
    float4 texel;
    PathStart start;
    if (!view_pass_end<S, E, M>(r, v.f, v.p, record, second, texel, start))
        return STEP_DONE;
    if (rays)
        store_texel_coherent(radiance, pixel, texel);
    store_part<StoredView, RECORD_WORDS>(v.record.words, record);
    return begin_launch(r, slot, pixel, start, v.p.rng, v.p.rays, v.record);
}

// The end of an update path, whose record adds to the cache, or of a photon, which its entry's
// yield counts where it deposited.
public uint launch_finish_record(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    v.record.finish(r, v.p.radiance, v.p.layers);
    return STEP_DONE;
}

// What the background layer does with `slot`'s finished path: its texel, the pre-exposed radiance
// in RGB and the reverse-Z depth of its hit in A (0 on a miss).
public uint launch_finish_background<E : IEnvironment, M : IMedium>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    background_texel<E, M>(r, v.f.pixel, load_part<StoredView, RECORD_WORDS>(v.record.words), v.p.radiance);
    return STEP_DONE;
}
