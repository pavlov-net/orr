// The clusters of particle producer entries a view's culling kept, for transparency hooks:
// one mesh workgroup per cluster of the view's particle list. The pixel stage finds its
// surface with `primary_hit` on the cluster, user words included.
module particles;

import orr.abi;
import producer;
__exported import visibility;

// A particle cluster's draw: its visible cluster, batch and entry-to-eye placement.
public struct ParticleDraw
{
    public uint visible;           // in the view's visible clusters
    public VisibleCluster vc;
    public ProducerBatch batch;
    public ProducerClusterGpu cluster;
    public Placement placement;
}

// The draw of mesh workgroup `group` of the particle draws; false past the list.
public bool particle_draw(SceneRoot* s, ViewConstants* v, RasterGpu* r, uint3 group, out ParticleDraw d)
{
    d = (ParticleDraw)0;
    uint index = grid_index(group);
    if (index >= r.counts[COUNT_PARTICLES])
        return false;
    d.visible = r.particles[index];
    d.vc = r.gbuffer.clusters[d.visible];
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
