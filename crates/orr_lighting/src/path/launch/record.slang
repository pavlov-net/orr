// A path's record in the pipeline, in its slot of `PathStages.records`.
module record;

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

// The record of a path in the pipeline, in its slot of `PathStages.records`, by the root's source.
// An update path's vertices go to its own callable (`launch_cache_vertex`), which `vertex` asks
// for.
public struct LaunchRecord : IVertexRecord
{
    public bool view;
    public bool background;
    public bool update;
    public bool caustic;
    public uint4* words;
    // `vertex` handed the record the vertex, and the sampler dimension it took for it, which the
    // callable draws.
    public bool recorded;
    public uint dimension;
    // The slot's stack hit, where the root keeps the stack part: what the raygen's probe met,
    // which `stack_owned` reads.
    public uint4* probe;

    public bool view_record()
    {
        return view || background;
    }

    public bool may_end(uint scattered, bool diffuse, bool wide, bool last)
    {
        if (update)
            return scattered >= 2;
        if (caustic)
            return false;
        return scattered >= 1 && (diffuse || wide || last);
    }

    public bool walks()
    {
        return view_record();
    }

    public bool adjoint()
    {
        return caustic;
    }

    public bool photon()
    {
        return caustic;
    }

    public bool spectral()
    {
        return !update;
    }

    public bool particles()
    {
        return !update;
    }

    [mutating]
    public void arrive(PathRoot r, PathVertex x, Bsdf b, float t, bool layer)
    {
        if (!view_record())
            return;
        StoredView p = load_part<StoredView, RECORD_WORDS>(words);
        view_arrive(p, x, b, t, layer);
        store_part<StoredView, RECORD_WORDS>(words, p);
    }

    [mutating]
    public void expose(float3 e)
    {
        if (!view_record())
            return;
        StoredView p = load_part<StoredView, RECORD_WORDS>(words);
        p.own += e;
        store_part<StoredView, RECORD_WORDS>(words, p);
    }

    // The probe a shader asks for of a stack the path crossed (STEP_PROBE) is the raygen's,
    // traced before the shader runs again.
    public bool stack_owned<S : ISurfaces>(PathRoot r, float3 origin, float3 dir, PathStack stack, uint depth, bool dispersion, bool feedback)
    {
        if (!stack.valid || stack.crossed == 0)
            return false;
        StackFace face;
        return stack_face(r.scene, stack_hit(load_part<StackHit, STACK_HIT_WORDS>(probe)), dispersion, face)
            && abs(dot(face.normal, stack.normal)) >= STACK_PARALLEL;
    }

    [mutating]
    public void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng)
    {
        recorded = update;
        // `CacheUpdate.vertex` draws one dimension.
        dimension = rng.dimension;
        if (update)
            rng.next();
    }

    [mutating]
    public void deposit(PathRoot r, LayerVertex l, Bsdf b, float3 flux, inout Sampler rng)
    {
        if (!caustic)
            return;
        CausticPhoton c = load_part<CausticPhoton, RECORD_WORDS>(words);
        c.deposit(r, l, b, flux, rng);
        store_part<CausticPhoton, RECORD_WORDS>(words, c);
    }

    [mutating]
    public void light(uint id, float contribution)
    {
        if (!view_record() || !view_light_counts(contribution))
            return;
        StoredViewLights p = load_part<StoredViewLights, 1>(words);
        view_light(p, id, contribution);
        store_part<StoredViewLights, 1>(words, p);
    }

    [mutating]
    public void finish(PathRoot r, float3 radiance, uint layers)
    {
        if (update)
        {
            CacheUpdate u = load_part<CacheUpdate, UPDATE_RECORD_WORDS>(words);
            u.finish(r, radiance, layers);
        }
        else if (caustic)
        {
            CausticPhoton c = load_part<CausticPhoton, RECORD_WORDS>(words);
            c.finish(r, radiance, layers);
        }
    }
}

public LaunchRecord launch_record(PathRoot r, uint slot)
{
    LaunchRecord record;
    record.view = from_view(r);
    record.background = from_background(r);
    record.update = from_update(r);
    record.caustic = from_photon(r);
    // The records' stride is layout, which the source gives alone.
    record.words = r.stages.records + slot * (r.source == PATH_SOURCE_UPDATE ? UPDATE_RECORD_WORDS : RECORD_WORDS);
    record.recorded = false;
    record.dimension = 0;
    record.probe = slot_words(r, slot) + stack_at(r) + STACK_HIT_AT;
    return record;
}

// The hit `h` holds, as `stack_face` reads it.
public Hit stack_hit(StackHit h)
{
    Hit k = Hit();
    k.hit = (h.flags & STACK_HIT_MET) != 0;
    k.triangle = (h.flags & STACK_HIT_TRIANGLE) != 0;
    k.t = h.t;
    k.instance_index = h.instance;
    k.class_id = h.class_id;
    k.primitive_index = h.primitive;
    k.cluster_id = h.cluster;
    k.barycentrics = h.barycentrics;
    return k;
}
