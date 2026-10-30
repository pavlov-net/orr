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

// The R2 sequence (Roberts 2018), exact in fixed point for every index.
public float2 r2(uint n)
{
    return float2(unit_float(n * 3242174889u + 2147483648u), unit_float(n * 2447445413u + 2147483648u));
}

// Sobol direction numbers for four dimensions (Joe and Kuo, new-joe-kuo-6.21201).
static const uint SOBOL_DIRECTIONS[4][32] = {
    { 0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000, 0x02000000, 0x01000000,
      0x00800000, 0x00400000, 0x00200000, 0x00100000, 0x00080000, 0x00040000, 0x00020000, 0x00010000,
      0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800, 0x00000400, 0x00000200, 0x00000100,
      0x00000080, 0x00000040, 0x00000020, 0x00000010, 0x00000008, 0x00000004, 0x00000002, 0x00000001 },
    { 0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
      0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
      0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
      0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff },
    { 0x80000000, 0xc0000000, 0x60000000, 0x90000000, 0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
      0x68800000, 0x9cc00000, 0xee600000, 0x55900000, 0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
      0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000, 0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
      0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590, 0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555 },
    { 0x80000000, 0xc0000000, 0x20000000, 0x50000000, 0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
      0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000, 0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
      0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000, 0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
      0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050, 0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093 },
};

// Sobol point `index` in dimension `dim` (0-3) as 32-bit fixed point.
public uint sobol(uint index, uint dim)
{
    uint x = 0;
    for (uint bit = 0; index != 0; bit++, index >>= 1)
    {
        if ((index & 1) != 0)
            x ^= SOBOL_DIRECTIONS[dim][bit];
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

    [mutating]
    uint next_seed()
    {
        uint s = hash_combine(seed, dimension);
        dimension += 1;
        return s;
    }
}
