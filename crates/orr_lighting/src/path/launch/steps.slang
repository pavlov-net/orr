// What the pipeline's shaders and its raygen hand each other: the callables in the order of
// the table, the steps a shader returns, the vertex classes and how the raygen calls a callable.
module steps;

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

// The pipeline's callables, in the order of its table (`orr_lighting::launch`). A path ray's hit
// shader runs its whole vertex; a vertex no hit shader ran, one with something in front of it or
// along its ray, a walk's or the G-buffer's, runs by its class (AT_*) in CALL_VERTEX + class. The
// raygen runs what a vertex leaves an entry for in the scratch: the caustic photons' gather, a
// photon's deposit and an update path's cache vertex.
[Host] public static const uint CALL_START_VIEW = 0;
[Host] public static const uint CALL_START_UPDATE = 1;
[Host] public static const uint CALL_START_PHOTON = 2;
[Host] public static const uint CALL_START_BACKGROUND = 3;
[Host] public static const uint CALL_RESOLVE_GIVEN = 4;
[Host] public static const uint CALL_EMISSION = 5;
[Host] public static const uint CALL_VERTEX = 6;        // AT_CLASSES of them, by class
[Host] public static const uint CALL_GATHER = CALL_VERTEX + AT_CLASSES;
[Host] public static const uint CALL_DEPOSIT = CALL_GATHER + 1;
[Host] public static const uint CALL_CACHE_VERTEX = CALL_DEPOSIT + 1;
[Host] public static const uint CALL_BENT = CALL_CACHE_VERTEX + 1;
[Host] public static const uint CALL_FINISH_VIEW = CALL_BENT + 1;
[Host] public static const uint CALL_FINISH_RECORD = CALL_FINISH_VIEW + 1;
[Host] public static const uint CALL_FINISH_BACKGROUND = CALL_FINISH_RECORD + 1;
[Host] public static const uint CALL_WALK_END = CALL_FINISH_BACKGROUND + 1;
[Host] public static const uint CALL_SEGMENT = CALL_WALK_END + 1;
[Host] public static const uint CALL_LAYERS = CALL_SEGMENT + 1;
[Host] public static const uint CALL_PROXY_MARCH = CALL_LAYERS + 1;
[Host] public static const uint CALL_GBUFFER = CALL_PROXY_MARCH + 1;
[Host] public static const uint CALL_STACK_STEP = CALL_GBUFFER + 1;
[Host] public static const uint CALL_STACK_MISS = CALL_STACK_STEP + 1;
[Host] public static const uint CALL_EMITTER_VIEW = CALL_STACK_MISS + 1;
[Host] public static const uint CALL_START_FIRST_VERTEX = CALL_EMITTER_VIEW + 1;  // a view's debug view of its first vertex alone
[Host] public static const uint CALL_CROSS_FOLD = CALL_START_FIRST_VERTEX + 1;
[Host] public static const uint CALLABLES = CALL_CROSS_FOLD + 1;

// What a shader hands the raygen in its payload's first word.
public static const uint STEP_RAY = 1 << 0;      // the path has a ray to trace
public static const uint STEP_VERTEX = 1 << 1;   // the path is at the vertex its surface words hold
public static const uint STEP_EMITS = 1 << 2;    // whose emission toward the path counts
public static const uint STEP_SHADOW = 1 << 3;   // the vertex's shade left a shadow job
public static const uint STEP_ENDED = 1 << 4;    // the path ended: its finish runs
public static const uint STEP_DONE = 1 << 5;     // nothing of the path is left to run: no finish follows
// A start, or a finish that starts a pass, left in the slot's shadow ray words a ray whose
// crossings of interiors' surfaces the path's interiors take before it goes on
// (`launch/boundary.slang`).
public static const uint STEP_CROSS = 1 << 6;
public static const uint STEP_WALK = 1 << 7;     // the path walks from its walk part
public static const uint STEP_GIVEN = 1 << 9;    // the path starts on the hit its surface words give, once its shadow ray has passed
// The path starts on a sky pixel: the miss shader runs on the camera ray its ray words hold.
public static const uint STEP_SKY = 1 << 10;
// A photon's deposit waits on the probe of the plane stack it crossed, whose ray its shade left in
// the slot's shadow ray.
public static const uint STEP_PROBE = 1 << 11;
// From STEP_AT_SHIFT, with STEP_VERTEX, the class of the vertex (AT_*), which selects its
// callable, CALL_VERTEX + class.
public static const uint STEP_AT_SHIFT = 12;
[Host] public static const uint AT_SURFACE = 0;         // an opaque vertex, its surface words a `PackedSurface`
[Host] public static const uint AT_EXIT = 1;            // a subsurface walk's exit
[Host] public static const uint AT_LEAVE = 2;           // an interior's walk leaving it
[Host] public static const uint AT_SCATTER = 3;         // a scatter in an interior's medium, its surface words a `MediumVertex`
[Host] public static const uint AT_LAYER = 4;           // a dielectric or presence layer, its words a `PackedLayer`
[Host] public static const uint AT_CLASSES = 5;
public static const uint STEP_AT_MASK = 7;

// How the path reached the vertex a step names, which the raygen hands its shaders: the vertex
// is the path's first; it is the G-buffer's surface, which its start reached and counted; a
// subsurface walk's exit is the vertex the walk entered, where it stopped at its cap; and the
// first vertex's pixel runs no path this frame, so it only emits.
public static const uint STEP_FIRST = 1 << 16;
public static const uint STEP_GBUFFER = 1 << 17;
public static const uint STEP_CAPPED = 1 << 18;
public static const uint STEP_UNSHADED = 1 << 19;
// A hit's shader ran its whole vertex: the step is its shade's, with the vertex's class and
// STEP_FIRST.
public static const uint STEP_SHADED = 1 << 23;
// What a vertex's shade leaves in the scratch for the raygen to run after it: the entry of the
// caustic photons' gather, of a photon's deposit, of an update path's cache vertex.
public static const uint STEP_GATHER = 1 << 20;
public static const uint STEP_DEPOSIT = 1 << 21;
public static const uint STEP_CACHE_VERTEX = 1 << 22;

// The class (AT_*) of the vertex `step` names.
public uint step_at(uint step)
{
    return (step >> STEP_AT_SHIFT) & STEP_AT_MASK;
}

// What a shader that leaves the path at the opaque vertex of `ps` hands the raygen, but whether
// its emission counts.
public uint vertex_step()
{
    return STEP_VERTEX | (AT_SURFACE << STEP_AT_SHIFT);
}

// What a shadow ray's trace leaves its raygen to do: its light is a coated emitter's, whose view
// its end takes; it is a directional light's that ended at an opaque surface or solid glass,
// before which a plane stack's probe looks for a face.
public static const uint TAKE_VIEW = 1 << 1;
public static const uint TAKE_BENDS = 1 << 2;

// `launch_stack_step`'s phases, its data's second word: the probe's hit is in the slot; a leg's
// stack ray and segment have run.
public static const uint STACK_PROBE = 0;
public static const uint STACK_LEG = 1;

// What it hands the raygen: a leg follows, whose ray the shadow job holds; its stack hit is the
// probe's, which the slot holds already; the light came through, and the bent light's callable
// weighs it. None of them: no light came through.
public static const uint STACK_ON = 1 << 0;
public static const uint STACK_PROBED = 1 << 1;
public static const uint STACK_LIT = 1 << 2;

// Runs callable `index` with `words` as its data, and returns its data.
public uint4 data_call(uint index, uint4 words)
{
    Payload c;
    c.words = words;
    call(index, c);
    return c.words;
}

// The first word of the data callable `index` hands back, run with `word` in its data's second
// word.
public uint step_call(uint index, uint word = 0)
{
    return data_call(index, uint4(0, word, 0, 0)).x;
}
