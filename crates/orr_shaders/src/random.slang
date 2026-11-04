// Hashes and sample sequences.
//
// Stochastic choices draw from `Sampler`: an Owen-scrambled Sobol sequence indexed by the
// frame and seeded by a hash of the pixel, padded by hashing the dimension, so every choice
// takes its own dimension and no pattern is shared across the screen.
module random;

// PCG hash (Jarzynski and Olano 2020).
public uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

public uint3 pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

public uint4 pcg4d(uint4 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

public uint hash_combine(uint seed, uint v)
{
    return pcg(seed ^ pcg(v));
}

// A float in [0, 1) from the top 24 bits.
public float unit_float(uint x)
{
    return float(x >> 8) * (1.0 / 16777216.0);
}

// A float in [0, 1) hashed from `x`.
public float pcg_float(uint x)
{
    return unit_float(pcg(x));
}

// Interleaved gradient noise (Jimenez 2014): a value in [0, 1) per pixel position.
public float interleaved_gradient_noise(float2 p)
{
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

// The R2 sequence (Roberts 2018), exact in fixed point for every index.
public float2 r2(uint n)
{
    return float2(unit_float(n * 3242174889u + 2147483648u), unit_float(n * 2447445413u + 2147483648u));
}

// Sobol point `index` in dimension `dim` (0-3) as 32-bit fixed point, with the direction numbers
// of Joe and Kuo's new-joe-kuo-6.21201. Each direction number follows from the ones before it by
// its dimension's primitive polynomial, so none is read from memory: a table indexed at run time
// costs a copy of it per lookup in thread-local memory.
public uint sobol(uint index, uint dim)
{
    if (dim == 0)
        return reversebits(index);
    if (dim == 1)
    {
        // Its generator is Pascal's matrix mod 2: bit j of the point sums the index bits whose
        // positions hold j's set bits, which a butterfly per bit of j computes.
        uint y = index;
        y ^= (y >> 1) & 0x55555555;
        y ^= (y >> 2) & 0x33333333;
        y ^= (y >> 4) & 0x0f0f0f0f;
        y ^= (y >> 8) & 0x00ff00ff;
        y ^= y >> 16;
        return reversebits(y);
    }
    uint x = 0;
    uint a = 0x80000000;
    uint b = 0xc0000000;
    uint c = 0x20000000;
    for (; index != 0; index >>= 1)
    {
        x ^= a & (0u - (index & 1u));
        uint next = b ^ a ^ (a >> dim);
        a = b;
        if (dim == 2)
        {
            b = next;
        }
        else
        {
            b = c;
            c = next;
        }
    }
    return x;
}

// Owen scrambling of 32-bit fixed point (Burley 2020, with Vegdahl's permutation): every
// bit flips by a hash of the bits above it, so stratification over power-of-two intervals
// survives.
public uint nested_uniform_scramble(uint x, uint seed)
{
    x = reversebits(x);
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return reversebits(x);
}

// Point `index` of a four-dimensional Owen-scrambled Sobol sequence, its order shuffled by
// `seed` as well (Burley 2020).
public float4 owen_sobol4(uint index, uint seed)
{
    uint i = nested_uniform_scramble(index, seed);
    return float4(owen_sobol(i, 0, seed), owen_sobol(i, 1, seed), owen_sobol(i, 2, seed), owen_sobol(i, 3, seed));
}

// Dimension `dim` of a shuffled point.
float owen_sobol(uint shuffled, uint dim, uint seed)
{
    return unit_float(nested_uniform_scramble(sobol(shuffled, dim), hash_combine(seed, dim)));
}

// One point of `owen_sobol4`, each half computed where it is used: most of its users read `xy`,
// and `zw` costs a loop per dimension.
public struct OwenSobol4
{
    public uint shuffled;
    public uint seed;

    public float2 xy()
    {
        return float2(owen_sobol(shuffled, 0, seed), owen_sobol(shuffled, 1, seed));
    }

    public float2 zw()
    {
        return float2(owen_sobol(shuffled, 2, seed), owen_sobol(shuffled, 3, seed));
    }
}

// Samples for one pixel and frame. Each `next*` call takes the next dimension; its points
// over the frames of one pixel are a scrambled Sobol sequence.
public struct Sampler
{
    public uint index;
    public uint seed;
    public uint dimension;

    // `stream` separates independent sample sets of one pixel and frame.
    public __init(uint2 pixel, uint frame, uint stream = 0)
    {
        index = frame;
        seed = pcg3d(uint3(pixel, stream)).x;
        dimension = 0;
    }

    [mutating]
    public float next()
    {
        uint s = next_seed();
        return owen_sobol(nested_uniform_scramble(index, s), 0, s);
    }

    [mutating]
    public float2 next2()
    {
        uint s = next_seed();
        uint i = nested_uniform_scramble(index, s);
        return float2(owen_sobol(i, 0, s), owen_sobol(i, 1, s));
    }

    [mutating]
    public float4 next4()
    {
        return owen_sobol4(index, next_seed());
    }

    // `next4`, its components computed by the caller as it needs them.
    [mutating]
    public OwenSobol4 next4_parts()
    {
        uint s = next_seed();
        OwenSobol4 point = { nested_uniform_scramble(index, s), s };
        return point;
    }

    [mutating]
    uint next_seed()
    {
        uint s = hash_combine(seed, dimension);
        dimension += 1;
        return s;
    }
}
