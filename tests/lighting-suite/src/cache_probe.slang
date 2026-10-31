// orr-lighting's world cache cell by cell: keys of points, samples added in order by one
// thread, and every cell's slot as the resolve left it.
#include "orr/root.h"
import world_cache;

struct KeyCase
{
    int4 origin_cell;       // w = cell_log2
    float3 position;        // render space
    int k;                  // voxel edge 2^k m
    float3 normal;
    uint level;
}

struct KeyOut
{
    uint2 key;              // low, high word
    uint hash;
    uint check;
}

struct CacheSample
{
    uint2 key;
    uint _pad0;
    uint _pad1;
    float4 value;
}

struct Root
{
    WorldCacheGpu* cache;
    KeyCase* cases;
    KeyOut* keys;
    CacheSample* samples;
    uint4* cells;           // check, resolved rgb9e5, resolved state, samples this frame
    uint count;
    uint capacity;
}
ORR_ROOT(Root);

uint64_t key_of(uint2 k)
{
    return uint64_t(k.x) | (uint64_t(k.y) << 32);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void keys(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    KeyCase c = root.cases[id.x];
    uint64_t key = cache_key(c.origin_cell, c.position, c.k, c.level, normal_bucket(c.normal));
    KeyOut o;
    o.key = uint2(uint(key), uint(key >> 32));
    o.hash = cache_hash(key);
    o.check = cache_check(key);
    root.keys[id.x] = o;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void add()
{
    for (uint i = 0; i < root.count; i++)
        cache_add(root.cache, key_of(root.samples[i].key), root.samples[i].value.xyz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cells(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.capacity)
        return;
    WorldCacheGpu* c = root.cache;
    root.cells[i] = uint4(c.checks[i], c.resolved[i], c.accumulated[i].w);
}
