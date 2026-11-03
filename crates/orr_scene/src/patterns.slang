// The CLAS templates of the tessellation patterns: one thread per pattern and opacity
// writes its template build record over the moment-curve positions, opaque templates first.
#include "orr_shaders/root.h"
import orr_shaders.abi;
import orr_gpu.rtas;
import orr_scene.tessellation;

public struct TemplatesRoot
{
    public PatternTable* patterns;
    public uint8_t* records;
    public float3* curve;         // template positions: vertex i on the moment curve
    public uint* count;           // the build's record count
}
ORR_ROOT(TemplatesRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void templates(uint3 id: SV_DispatchThreadID)
{
    if (id.x == 0)
        *root.count = 2 * PATTERNS;
    if (id.x >= 2 * PATTERNS)
        return;
    PatternTable* table = root.patterns;
    PatternGpu p = table.patterns[id.x % PATTERNS];
    bool opaque = id.x < PATTERNS;
    TemplateBuild r = (TemplateBuild)0;
    r.cluster.cluster_id = 0;
    r.cluster.tri_count = p.tri_count;
    r.cluster.vertex_count = p.vertex_count;
    r.cluster.index_format = 1;
    r.cluster.base_geometry_and_flags = geometry_index_and_flags(0, opaque ? GEOMETRY_OPAQUE : 0);
    r.cluster.omm_index_format = 4;
    r.cluster.vertices = (uint64_t)root.curve;
    r.cluster.indices = (uint64_t)(table.triangles + 3 * p.tri_first);
    store((uint64_t)root.records, id.x, r);
}
