// Shadow rays: a vertex's job made ready where it is made, so the raygen adds the job's
// contribution times what its trace let through; a photon's ray toward its light; and a coated
// emitter's view.
module shadow;

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

// Whether the shadow ray of flags `flags` of a path of `r` is a directional light's that a plane
// stack may bend (`path_light`'s `bend`).
public bool shadow_bends(PathRoot r, uint flags)
{
    return stack_kept(r) && has(flags, SHADOW_DIRECTIONAL) && has(flags, SHADOW_BEND);
}

// Makes shadow ray `q` ready as the raygen takes it, where the vertex makes it: its contribution
// times the transmittance of the interiors it starts in over its length and, unless it goes to the
// environment, of the linked medium. The raygen multiplies in what its trace lets through. A plane
// stack's walk replaces the contribution of a ray it bends with its legs'. The interiors the ray
// starts in: those its vertex arrived with, crossed at its surface where the ray goes into it.
public Interior shadow_start<M : IMedium>(PathRoot r, inout PathShadowRay q)
{
    Interior inside = Interior();
    float3 start = q.origin;
    if (interiors_tracked(r))
    {
        inside = q.arrived;
        start = q.start;
        if (q.surface != MATERIAL_NONE)
            inside.cross(q.surface, has(q.flags, SHADOW_INTO), q.point);
    }
    if (!any(q.contribution > 0.0))
        return inside;
    float3 vis = float3(1.0);
    if (!has(q.flags, SHADOW_ENVIRONMENT))
        vis *= medium_transmittance<M>(r.scene.medium, q.origin, q.dir, q.t_max);
    float scale;
    float3 depth = straight_depth(r.scene, inside, q.t_max, false, scale);
    if (q.t_max > 0.0 && isfinite(q.t_max))
        depth *= segment_length(start, q.origin, q.dir, q.t_max) / q.t_max;
    q.contribution *= vis * exp(-depth);
    return inside;
}

// The shadow ray in `slot`'s words as a ray, with its flags in `flags` and the payload its any-hit
// shaders read: in mode `mode` (SHADOW_PATH for the shadow ray a vertex left, SHADOW_PHOTON for a
// photon's toward its light), its candidates of the context of a ray at path depth `depth`.
public RayDesc launch_shadow(PathRoot r, uint slot, uint mode, out uint flags, out uint4 payload)
{
    uint4* at = hot_words(r, slot);
    ShadowOrigin o = load_word<ShadowOrigin>(at + SHADOW_ORIGIN_AT);
    ShadowDir d = load_word<ShadowDir>(at + SHADOW_DIR_AT);
    flags = d.flags;
    ShadowCover c = mode == SHADOW_PHOTON ? shadow_cover(mode, SHADOW_MASK, 0, false)
        : shadow_cover(mode, SHADOW_MASK, (d.flags >> SHADOW_DEPTH_SHIFT) + 1, has(d.flags, SHADOW_FEEDBACK));
    payload = c.words();
    return path_ray(o.origin, d.dir, o.t_max);
}

// Multiplies `slot`'s shadow job's contribution by `by`.
public void scale_shadow(PathRoot r, uint slot, float3 by)
{
    uint4* at = hot_words(r, slot) + SHADOW_CONTRIBUTION_AT;
    ShadowContribution q = load_word<ShadowContribution>(at);
    q.contribution *= by;
    store_word(at, q);
}

// Adds to `slot`'s path what its shadow ray of flags `flags` let through, `passed` of its job's
// contribution, in the path's wavelengths' colors, and gives the light its share for a view's
// record. The loop's radiance word alone.
public void add_shadow(PathRoot r, uint slot, uint flags, float3 passed)
{
    uint4* at = hot_words(r, slot);
    ShadowContribution q = load_word<ShadowContribution>(at + SHADOW_CONTRIBUTION_AT);
    float3 added = q.contribution * passed;
    if (all(added == 0.0))
        return;
    if (spectral_tracked(r))
        added = path_spectral(r, slot, load_word<LoopThroughput>(at + LOOP_THROUGHPUT_AT).flags).color(added);
    if (has(flags, SHADOW_COUNTED))
        add_radiance(at, added);
    if (q.light != NO_LIGHT)
    {
        LaunchRecord record = launch_record(r, slot);
        record.light(q.light, luminance_rec2020(added));
    }
}

// What a photon's shadow ray toward its light lets through of its flux, `passed` of its job's
// contribution: STEP_GIVEN, or STEP_DONE where none is left.
public uint launch_light_add(PathRoot r, uint slot, float3 passed)
{
    uint4* at = hot_words(r, slot);
    ShadowContribution q = load_word<ShadowContribution>(at + SHADOW_CONTRIBUTION_AT);
    LoopThroughput l = load_word<LoopThroughput>(at + LOOP_THROUGHPUT_AT);
    l.throughput *= q.contribution * passed;
    store_word(at + LOOP_THROUGHPUT_AT, l);
    return any(l.throughput > 0.0) ? STEP_GIVEN : STEP_DONE;
}

// The emitter view shader's body: what the coat and fuzz of the emitter `slot`'s shadow ray
// reached pass of its light toward the vertex (`emitter_view`), into its contribution.
public void launch_emitter_view<S : ISurfaces>(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot) + SHADOW_CONTRIBUTION_AT;
    ShadowContribution q = load_word<ShadowContribution>(at);
    if (!any(q.contribution > 0.0))
        return;
    LightView l = load_part<LightView, SLOT_LIGHT_PART>(light_words(r, slot));
    ResolvedLightSample c = no_candidate();
    c.dir = l.dir;
    c.light = l.light;
    c.position = l.position;
    c.triangle = l.triangle;
    c.point = l.point;
    c.view = true;
    q.contribution *= emitter_view_layer<CoverageFeedback<S>>(r.set, c, l.exterior_ior);
    store_word(at, q);
}
