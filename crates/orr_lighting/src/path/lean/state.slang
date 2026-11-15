// What a path of the lean tracer holds between its shaders: LEAN_HOT_WORDS words of 16 bytes in its
// slot, packed, which every shader of the common path loads and stores whole, and the budget the
// tracer's stages are held to (`tests/lighting_suite/tests/budget.rs`).
//
// A path's ray, its throughput and its pending shadow job are all a hit's shader leaves for the
// raygen: the raygen holds the slot's address across a trace or an invoke and nothing of the path.
// What only some paths need (the interiors a path lies in, a plane stack, wavelengths, a walk)
// lives in cold parts of the slot that the shaders of those features load, by the hot flags.
module state;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.pipeline;

// The budget of the common path: what a plain ray pays whatever else its pipeline links.
// A path's hot state.
[Host] public static const uint LEAN_HOT_WORDS = 6;
[Host] public static const uint LEAN_HOT_BYTES = 16 * LEAN_HOT_WORDS;
// What a raygen keeps across a trace or an invoke, bytes: the driver's stack size of the raygen
// beyond its hit object.
[Host] public static const uint RAYGEN_LIVE_BYTES = 32;
// A ray's payload.
[Host] public static const uint PAYLOAD_BYTES = 16;
// Registers of the plain opaque hit shader and of an alpha test's any-hit, as their compute
// stand-ins report them, with no local or shared memory: a launch runs at the count of its
// largest shader up to 128, past which that shader spills.
[Host] public static const uint HIT_REGISTERS = 128;
[Host] public static const uint ANY_HIT_REGISTERS = 64;

// `PathHot.flags`.
public static const uint HOT_SHADOW = 1 << 0;       // a shadow job is pending: the raygen traces it before the path's ray
public static const uint HOT_SHADOW_FAR = 1 << 1;   // its target is a direction: the light lies at infinity
public static const uint HOT_RAY = 1 << 2;          // the path has a ray to trace
public static const uint HOT_CAMERA = 1 << 3;       // which is its camera ray: its vertex is the path's first
public static const uint HOT_DIFFUSE = 1 << 4;      // the ray left a diffuse lobe
public static const uint HOT_COUNTED = 1 << 5;      // light its ray meets counts toward the root's mode
public static const uint HOT_MEDIUM = 1 << 6;       // the ray left a point in a medium, which has no normal
// From HOT_DEPTH_SHIFT, the vertices the path has scattered at, HOT_DEPTH_MASK at most.
public static const uint HOT_DEPTH_SHIFT = 8;
public static const uint HOT_DEPTH_MASK = 0xff;
// From HOT_DIMENSION_SHIFT, the sampler dimension the path's next draw takes.
public static const uint HOT_DIMENSION_SHIFT = 16;

// A path's hot state as its shaders hold it.
public struct PathHot
{
    public float3 origin;           // of its ray
    public uint flags;              // HOT_*, its depth and its sampler dimension
    public float3 dir;              // of its ray, unit
    public float pdf;               // the density its vertex's scatter gave `dir`; 0 for an ideal-specular one
    public float3 throughput;
    public float run;               // the probability its vertex ran next-event estimation
    public float2 cone;             // the ray cone's width at the origin and its spread
    public float3 normal;           // the shading normal of the vertex the ray left; none with HOT_MEDIUM
    public float3 radiance;         // gathered so far
    public uint light;              // the pending shadow job's light, for the feedback its light gives
    public float3 shadow_origin;
    public float3 shadow_target;    // the point on the light its ray ends before, or with HOT_SHADOW_FAR its direction
    public float3 shadow_light;     // what the job adds where its ray meets nothing

    public uint depth()
    {
        return (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    }

    public uint dimension()
    {
        return flags >> HOT_DIMENSION_SHIFT;
    }

    [mutating]
    public void set_depth(uint depth)
    {
        flags = (flags & ~(HOT_DEPTH_MASK << HOT_DEPTH_SHIFT)) | (min(depth, HOT_DEPTH_MASK) << HOT_DEPTH_SHIFT);
    }

    [mutating]
    public void set_dimension(uint dimension)
    {
        flags = (flags & 0xffff) | (dimension << HOT_DIMENSION_SHIFT);
    }
}

// The hot words of a path, in order:
//   0  origin, flags
//   1  dir as a 2 x 16-bit octahedral code, normal as one, throughput as three halves and run as a half
//   2  the cone as two halves, a spare word, the shadow job's light as rgb9e5, the job's light id
//   3  radiance, pdf
//   4  shadow origin, a spare word
//   5  shadow target, a spare word
// A direction keeps 16 bits a component: under 4e-5 rad, which no scatter's lobe resolves.
public PathHot load_hot(uint4* at)
{
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    uint4 w2 = load_coherent(at + 2);
    PathHot p;
    p.origin = asfloat(w0.xyz);
    p.flags = w0.w;
    p.dir = oct_decode_snorm16(w1.x);
    p.normal = oct_decode_snorm16(w1.y);
    float2 b_run = unpack_half2x16(w1.w);
    p.throughput = float3(unpack_half2x16(w1.z), b_run.x);
    p.run = b_run.y;
    p.cone = unpack_half2x16(w2.x);
    p.shadow_light = unpack_rgb9e5(w2.z);
    p.light = w2.w;
    float4 w3 = asfloat(load_coherent(at + 3));
    p.radiance = w3.xyz;
    p.pdf = w3.w;
    p.shadow_origin = float3(0.0);
    p.shadow_target = float3(0.0);
    return p;
}

// Stores `p` but its shadow job's ray, which `store_shadow` stores where one is pending.
public void store_hot(uint4* at, PathHot p)
{
    store_coherent(at, uint4(asuint(p.origin), p.flags));
    store_coherent(at + 1, uint4(oct_encode_snorm16(p.dir), oct_encode_snorm16(p.normal), pack_half2x16(p.throughput.xy), pack_half2x16(float2(p.throughput.z, p.run))));
    store_coherent(at + 2, uint4(pack_half2x16(p.cone), 0, pack_rgb9e5(p.shadow_light), p.light));
    store_coherent(at + 3, uint4(asuint(p.radiance), asuint(p.pdf)));
}

public void store_shadow(uint4* at, PathHot p)
{
    store_coherent(at + 4, uint4(asuint(p.shadow_origin), 0));
    store_coherent(at + 5, uint4(asuint(p.shadow_target), 0));
}
