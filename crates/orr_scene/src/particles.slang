// The clusters of particle producer entries a view's culling kept, for transparency hooks:
// one mesh workgroup per cluster of the view's particle list. The pixel stage finds its
// surface with `primary_hit` on the cluster, user words included. Smoke entries' clusters are
// listed apart, for the froxels.
module particles;

import orr_shaders.abi;
import orr_scene.producer;
__exported import orr_scene.visibility;

// A particle cluster's draw: its visible cluster, batch and entry-to-eye placement.
public struct ParticleDraw
{
    public uint visible;           // in the view's visible clusters
    public VisibleCluster vc;
    public ProducerBatch batch;
    public ProducerClusterGpu cluster;
    public Placement placement;
}

// The draw of mesh workgroup `group` of a particle draw with the raster record `r` it took;
// false past the list.
public bool particle_draw(SceneRoot* s, ViewConstants* v, RasterGpu* r, uint3 group, out ParticleDraw d)
{
    d = (ParticleDraw)0;
    uint index = grid_index(group) + r.draw_first;
    return index < r.counts[COUNT_PARTICLES] && listed_draw(s, v, r, r.particles[index], d);
}

// The draw of workgroup `group` of the smoke clusters; false past the list.
public bool smoke_draw(SceneRoot* s, ViewConstants* v, RasterGpu* r, uint3 group, out ParticleDraw d)
{
    d = (ParticleDraw)0;
    uint index = grid_index(group);
    return index < r.counts[COUNT_SMOKE] && listed_draw(s, v, r, r.smoke[index], d);
}

bool listed_draw(SceneRoot* s, ViewConstants* v, RasterGpu* r, uint visible, inout ParticleDraw d)
{
    d.visible = visible;
    d.vc = r.gbuffer.clusters[visible];
    uint entry;
    if (!visible_producer(s, d.vc, entry))
        return false;
    d.batch = s.batches[entry];
    d.cluster = d.batch.clusters[d.vc.cluster];
    return visible_placement(s, v, d.vc, false, d.placement);
}

// Vertex `i` of the draw in jittered clip space.
public float4 particle_vertex(ViewConstants* v, ParticleDraw d, uint i)
{
    float3 q = place(d.placement, d.batch.positions[d.cluster.first_vertex + i]);
    return mul(v.clip_from_view_jittered, float4(view_from_relative(v, q), 1.0));
}

// The cluster-local corners of triangle `t` of the draw.
public uint3 particle_triangle(ParticleDraw d, uint t)
{
    return cluster_corners(d.cluster, t);
}
