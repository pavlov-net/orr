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

// The pipeline's callables, in the order of its table (`orr_lighting::launch`). Next-event
// estimation and the scatter of a vertex run by its class: an opaque vertex's as a plain
// surface's, which the pipeline makes without the lobes and walks a plain one lacks, or a rich
// one's.
[Host] public static const uint CALL_START_VIEW = 0;
[Host] public static const uint CALL_START_UPDATE = 1;
[Host] public static const uint CALL_START_PHOTON = 2;
[Host] public static const uint CALL_START_BACKGROUND = 3;
[Host] public static const uint CALL_RESOLVE_GIVEN = 4;
[Host] public static const uint CALL_EMISSION = 5;
[Host] public static const uint CALL_DIRECT = 6;        // AT_CLASSES of them, by class
[Host] public static const uint CALL_SCATTER = CALL_DIRECT + AT_CLASSES;  // AT_CLASSES of them, by class
[Host] public static const uint CALL_DEPOSIT = CALL_SCATTER + AT_CLASSES;
[Host] public static const uint CALL_CACHE_VERTEX = 19;
[Host] public static const uint CALL_FINISH_VIEW = 20;
[Host] public static const uint CALL_FINISH_RECORD = 21;
[Host] public static const uint CALL_FINISH_BACKGROUND = 22;
[Host] public static const uint CALL_WALK_END = 23;
[Host] public static const uint CALL_SEGMENT = 24;
[Host] public static const uint CALL_LAYERS = 25;
[Host] public static const uint CALL_PROXY_MARCH = 26;
[Host] public static const uint CALL_GBUFFER = 27;
[Host] public static const uint CALL_STACK_STEP = 28;
[Host] public static const uint CALL_EMITTER_VIEW = 29;
[Host] public static const uint CALL_START_FIRST_VERTEX = 30;   // a view's debug view of its first vertex alone
[Host] public static const uint CALLABLES = 31;

// What the raygen hands a vertex's direct or deposit callable in its data's second word: its
// first call; its call again once the raygen has traced the probe it asked for (STEP_PROBE); and
// its second phase, which takes a directional light that a plane stack bent. The vertex's
// emission, direct and deposit callables take the length of the ray that reached the vertex in
// its third word and the step that reached it in its fourth.
public static const uint DIRECT_FIRST = 0;
public static const uint DIRECT_PROBED = 1;
public static const uint DIRECT_BENT = 2;

// What a shader hands the raygen in its payload's first word.
public static const uint STEP_RAY = 1 << 0;      // the path has a ray to trace
public static const uint STEP_VERTEX = 1 << 1;   // the path is at the vertex its surface words hold
public static const uint STEP_EMITS = 1 << 2;    // which emits
public static const uint STEP_SHADOW = 1 << 3;   // next-event estimation left a shadow ray
public static const uint STEP_ENDED = 1 << 4;    // the path ended: its finish runs
public static const uint STEP_DONE = 1 << 5;     // nothing of the path is left to run: no finish follows
// A start or a finish asks for the surfaces of an interior that the ray it left in the slot's ray
// words crosses (`launch/boundary.slang`), before it runs again.
public static const uint STEP_CROSS = 1 << 6;
public static const uint STEP_WALK = 1 << 7;     // the path walks from its walk part
public static const uint STEP_INVOKE = 1 << 8;   // the walk's last step met a surface whose hit shader goes on
public static const uint STEP_GIVEN = 1 << 9;    // the path starts on the hit its surface words give, once its shadow ray has passed
public static const uint STEP_RECORD = 1 << 10;  // the record takes the vertex: an update path's cache vertex
// The shader decided nothing and asks the raygen for a plane stack's probe, whose ray it left in
// the slot's shadow ray, before it runs again.
public static const uint STEP_PROBE = 1 << 11;
// From STEP_AT_SHIFT, with STEP_VERTEX, the class of the vertex (AT_*), which selects its direct
// callable, CALL_DIRECT + class, and its scatter, CALL_SCATTER + class.
public static const uint STEP_AT_SHIFT = 12;
[Host] public static const uint AT_PLAIN = 0;           // an opaque vertex whose surface has no lobe a plain one lacks
[Host] public static const uint AT_RICH = 1;            // an opaque vertex whose surface may have one (`surface_lobed`)
[Host] public static const uint AT_EXIT = 2;            // a subsurface walk's exit
[Host] public static const uint AT_LEAVE = 3;           // an interior's walk leaving it
[Host] public static const uint AT_SCATTER = 4;         // a scatter in an interior's medium, its surface words a `MediumVertex`
[Host] public static const uint AT_LAYER = 5;           // a dielectric or presence layer, its words a `PackedLayer`
public static const uint AT_CLASSES = 6;
static const uint STEP_AT_MASK = 7;

// How the path reached the vertex a step names, which the raygen hands its shaders: the vertex
// is the path's first; it is the G-buffer's surface; the loop has not yet taken the account of
// the ray that reached it, which the first of its shaders to load the loop takes
// (`arrive_vertex`); and a segment along that ray drew the march's jitter.
public static const uint STEP_FIRST = 1 << 16;
public static const uint STEP_GBUFFER = 1 << 17;
public static const uint STEP_ARRIVES = 1 << 18;
public static const uint STEP_SEGMENTED = 1 << 19;

// The class (AT_*) of the vertex `step` names.
public uint step_at(uint step)
{
    return (step >> STEP_AT_SHIFT) & STEP_AT_MASK;
}

// What a path ray asks of the raygen beside its trace (`launch_path_ray`): the instance mask of a
// LAYERS trace in front of its hit in the low bits, 0 for none, and a segment.
public static const uint FRONT_SEGMENT = 1 << 8;

// What a shader that leaves the path at the vertex of `ps` hands the raygen.
public uint vertex_step(PackedSurface ps)
{
    return STEP_VERTEX | (surface_emits(ps) ? STEP_EMITS : 0) | ((surface_lobed(ps) ? AT_RICH : AT_PLAIN) << STEP_AT_SHIFT);
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
// probe's, which the slot holds already; the light came through, and next-event estimation's
// second phase takes it (`direct_bent`). None of them: no light came through.
public static const uint STACK_ON = 1 << 0;
public static const uint STACK_PROBED = 1 << 1;
public static const uint STACK_LIT = 1 << 2;

// Runs callable `index` with `word` in its data's second word and `t` in its third, and returns
// its data: the step it hands back first.
public uint4 data_call(uint index, uint word = 0, float t = 0.0)
{
    Payload c;
    c.words = uint4(0, word, asuint(t), 0);
    call(index, c);
    return c.words;
}

// `data_call`'s step.
public uint step_call(uint index, uint word = 0)
{
    return data_call(index, word).x;
}
