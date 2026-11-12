// A scene's producer figures for a frame's statistics: `sum` adds up what the set entries of
// the producer table emitted, dropped and built, after the frame's builds.
#include "orr_root.h"
import orr_scene.producer;
import orr_scene.tessellation;
import orr_scene.tables;

// The sums over a scene's set entries, as `ProducerStats` and `TessellationStats` name them.
public struct ProducerFigures
{
    public uint entries;
    public uint clusters;
    public uint cluster_capacity;
    public uint dropped_clusters;
    public uint vertices;
    public uint vertex_capacity;
    public uint dropped_vertices;
    public uint segments;
    public uint segment_capacity;
    public uint dropped_segments;
    public uint refused;
    public uint built;
    public uint tessellated;        // tessellated entries, which the fields below cover
    public uint patches;
    public uint subpatches;
    public uint split_dropped;
    public uint dice_dropped;
    public uint weld_dropped;
    public uint unseen_edges;
    public uint welded;
    public float lowest_rate_scale; // of the tessellated entries; unset without one
    public uint _pad;
}

public struct FiguresRoot
{
    public SceneRoot* scene;
    public ProducerFigures* figures;
    public TessStateGpu* tess_state; // by producer entry
    public uint entries;            // of the producer table
    public uint _pad;
}
ORR_ROOT(FiguresRoot);

static const uint GROUP = 128;

groupshared ProducerFigures parts[GROUP];

void add(inout ProducerFigures a, ProducerFigures b)
{
    if (b.tessellated != 0)
        a.lowest_rate_scale = a.tessellated != 0 ? min(a.lowest_rate_scale, b.lowest_rate_scale) : b.lowest_rate_scale;
    a.entries += b.entries;
    a.clusters += b.clusters;
    a.cluster_capacity += b.cluster_capacity;
    a.dropped_clusters += b.dropped_clusters;
    a.vertices += b.vertices;
    a.vertex_capacity += b.vertex_capacity;
    a.dropped_vertices += b.dropped_vertices;
    a.segments += b.segments;
    a.segment_capacity += b.segment_capacity;
    a.dropped_segments += b.dropped_segments;
    a.refused += b.refused;
    a.built += b.built;
    a.tessellated += b.tessellated;
    a.patches += b.patches;
    a.subpatches += b.subpatches;
    a.split_dropped += b.split_dropped;
    a.dice_dropped += b.dice_dropped;
    a.weld_dropped += b.weld_dropped;
    a.unseen_edges += b.unseen_edges;
    a.welded += b.welded;
}

// The figures of producer entry `i`, which is set.
ProducerFigures entry_figures(SceneRoot* s, uint i)
{
    ProducerBatch* b = s.batches + i;
    ProducerStateGpu* st = s.producer_state + i;
    uint clusters = b.cluster_count[0];
    uint vertices = b.vertex_count[0];
    // Only a strand entry counts segments.
    uint segments = b.segment_count != nullptr ? b.segment_count[0] : 0;
    ProducerFigures f = (ProducerFigures)0;
    f.entries = 1;
    f.clusters = min(clusters, b.max_clusters);
    f.cluster_capacity = b.max_clusters;
    f.dropped_clusters = clusters - f.clusters;
    f.vertices = min(vertices, b.max_vertices);
    f.vertex_capacity = b.max_vertices;
    f.dropped_vertices = vertices - f.vertices;
    f.segments = min(segments, b.max_segments);
    f.segment_capacity = b.max_segments;
    f.dropped_segments = segments - f.segments;
    f.refused = st.refused;
    f.built = st.clusters;
    if (s.producers[i].tessellated != 0)
    {
        TessStateGpu* t = root.tess_state + i;
        f.tessellated = 1;
        f.patches = t.patches;
        f.subpatches = t.subpatches;
        f.split_dropped = t.split_dropped;
        f.dice_dropped = t.dice_dropped;
        f.weld_dropped = t.weld_dropped;
        f.unseen_edges = t.unseen;
        f.welded = t.welded;
        // A state no frame has stepped holds no scale, which `tessellate` reads as 1.
        f.lowest_rate_scale = t.rate_scale > 0.0 ? t.rate_scale : 1.0;
    }
    return f;
}

// One workgroup. Thread `t` sums entries `t`, `t + GROUP` and so on; thread 0 then adds the
// threads' sums and writes the scene's.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void sum(uint t: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ProducerFigures f = (ProducerFigures)0;
    for (uint i = t; i < root.entries; i += GROUP)
    {
        // An unset entry is all zero.
        if (s.batches[i].cluster_count != nullptr)
            add(f, entry_figures(s, i));
    }
    parts[t] = f;
    GroupMemoryBarrierWithGroupSync();
    if (t != 0)
        return;
    for (uint j = 1; j < GROUP; j++)
        add(f, parts[j]);
    *root.figures = f;
}
