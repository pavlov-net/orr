// Software raster of procedural micro-triangles into a 64-bit visibility buffer with
// atomic max. TARGET 0: R64_UINT storage image from the resource heap; 1: buffer by address,
// linear rows; 2: buffer by address, 8x8 pixel tiles. Mirrors `raster_ref` in main.rs.
#ifndef TARGET
#define TARGET 0
#endif

struct Root
{
    uint64_t* vis;
    uint img;
    uint width;
    uint height;
    uint cell;   // cell size in pixels; two triangles per cell
    uint cells_x;
    uint cells_y;
    uint layers;
    uint ntri;
};
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

int edge(int2 a, int2 b, int2 p)
{
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Through a heap handle Slang infers no image format, and an image atomic needs one: the third
// argument is Slang's ImageFormat value for r64ui (40, slang-image-format-defs.h).
typealias VisImage = RWTexture2D<uint64_t, 0, 40>;

void store(uint x, uint y, uint64_t v)
{
#if TARGET == 0
    VisImage img = getDescriptorFromHandle(DescriptorHandle<VisImage>(uint2(root.img, 0)));
    InterlockedMax(img[uint2(x, y)], v);
#elif TARGET == 1
    InterlockedMax(root.vis[y * root.width + x], v);
#else
    uint tile = (y >> 3) * (root.width >> 3) + (x >> 3);
    InterlockedMax(root.vis[tile * 64 + (y & 7) * 8 + (x & 7)], v);
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void raster(uint3 id: SV_DispatchThreadID)
{
    uint t = id.x;
    if (t >= root.ntri)
        return;
    uint per_layer = root.cells_x * root.cells_y * 2;
    uint layer = t / per_layer;
    uint c = (t % per_layer) >> 1;
    uint half_ = t & 1;
    int s = int(root.cell) * 256;
    int2 base = int2(int(c % root.cells_x), int(c / root.cells_x)) * s;
    base += int2(int(pcg(layer * 7919u) & 255u), int(pcg(layer * 104729u) & 255u)) * s / 256;
    int2 corner[3] = { int2(0, 0), int2(s, 0), int2(0, s) };
    if (half_ == 1)
    {
        corner[0] = int2(s, 0);
        corner[1] = int2(s, s);
        corner[2] = int2(0, s);
    }
    int2 v[3];
    for (uint k = 0; k < 3; ++k)
    {
        uint h = pcg(t * 3u + k);
        int2 j = int2(int(h & 0xffffu), int(h >> 16)) - 32768;
        v[k] = base + corner[k] + j * s / 131072;
    }
    if (edge(v[0], v[1], v[2]) < 0)
    {
        int2 tmp = v[1];
        v[1] = v[2];
        v[2] = tmp;
    }
    int2 lo = min(v[0], min(v[1], v[2]));
    int2 hi = max(v[0], max(v[1], v[2]));
    int x0 = max((lo.x - 128 + 255) >> 8, 0);
    int y0 = max((lo.y - 128 + 255) >> 8, 0);
    int x1 = min((hi.x - 128) >> 8, int(root.width) - 1);
    int y1 = min((hi.y - 128) >> 8, int(root.height) - 1);
    uint depth = 0x3f800000u - (pcg(t ^ 0xa511e9b3u) >> 9); // positive float bits, larger is nearer
    uint64_t value = (uint64_t(depth) << 32) | uint64_t(t);
    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            int2 p = int2(x * 256 + 128, y * 256 + 128);
            if (edge(v[0], v[1], p) >= 0 && edge(v[1], v[2], p) >= 0 && edge(v[2], v[0], p) >= 0)
                store(uint(x), uint(y), value);
        }
    }
}
