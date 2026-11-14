// What rays trace of the blade field: one workgroup per caster item derives its candidates
// and emits those rays trace as one cluster of the entry, with the cluster topology of that
// many: an exact cell's blades as they are drawn, and elsewhere one-triangle chords, but for
// the blades of exact cells. A drawn field's are those the cast range's fade keeps, whatever
// the draw keeps; a stored field's those the draw keeps, as wide as it draws them. Each
// vertex's word is its blade's color, as the draw's linked geometry gives a drawn triangle.
#include "orr_root.h"
import orr_examples_grass.blades;
import orr_scene.producer;
import orr_scene.tables;
import orr_scene.visibility;

struct Root
{
    SceneRoot* scene;
    Field* field;
}
ORR_ROOT(Root);

groupshared uint alive[2];
groupshared uint first_vertex;

[shader("compute")]
[numthreads(CASTER_SLICE_CHORD, 1, 1)]
void main(uint3 group: SV_GroupID, uint lane: SV_GroupThreadID)
{
    Field* f = root.field;
    uint index = grid_index(group);
    if (index >= min(f.counts[FIELD_CASTER_ITEMS], f.caster_capacity))
        return;
    uint item = f.caster_items[index];
    bool exact = (item & ITEM_EXACT) != 0;
    uint size = exact ? CASTER_SLICE_EXACT : CASTER_SLICE_CHORD;
    uint tile = item_tile(item);
    uint stride;
    uint blade = caster_candidate(f, tile, item, lane, stride);
    if (lane < 2)
        alive[lane] = 0;
    GroupMemoryBarrierWithGroupSync();

    Blade b = Blade();
    float cast = 0.0;
    if (lane < size && blade < f.ranks * CELL_COUNT)
    {
        b = field_blade(f, tile, blade);
        // A chord stands for the blades of cells that are not exact.
        bool own = exact || (item & ITEM_FAR) != 0 || !cell_exact(f, tile, blade % CELL_COUNT);
        if (f.stored != 0)
        {
            float widen = blade_drawn(f, b, blade);
            cast = !own ? 0.0 : exact ? widen : chord_width(widen);
        }
        else if (own && blade_traced(f, b, cast_distance(f, b.root.xz)))
        {
            cast = exact ? 1.0 : chord_width(float(stride));
        }
        if (cast > 0.0)
            InterlockedOr(alive[lane >> 5], 1u << (lane & 31));
    }
    GroupMemoryBarrierWithGroupSync();

    uint count = countbits(alive[0]) + countbits(alive[1]);
    if (count == 0)
        return;
    ProducerBatch batch = root.scene.batches[f.producer];
    uint per = exact ? FULL_VERTICES : FAR_VERTICES;
    if (lane == 0)
        first_vertex = alloc_vertices(batch, count * per);
    GroupMemoryBarrierWithGroupSync();

    if (first_vertex == NO_ROOM)
        return;
    if (cast > 0.0)
    {
        uint below = lane < 32 ? countbits(alive[0] & ((1u << lane) - 1)) : countbits(alive[0]) + countbits(alive[1] & ((1u << (lane - 32)) - 1));
        float2 sway = blade_sway(f, b, false);
        bool previous = batch.previous != nullptr;
        float2 before = previous ? blade_sway(f, b, true) : sway;
        uint color = blade_color(b);
        for (uint k = 0; k < per; k++)
        {
            float2 at = exact ? full_template(k) : far_template(k);
            uint v = first_vertex + below * per + k;
            batch.positions[v] = blade_point(b, cast, sway, at);
            if (previous)
                batch.previous[v] = blade_point(b, cast, before, at);
            batch.attrs[v] = producer_attributes(blade_normal(b, sway, at), float2(0.5 + 0.5 * at.x, at.y));
            batch.vertex_user[v] = color;
        }
    }
    if (lane == 0)
        emit_template_cluster(batch, (exact ? 0 : CASTER_SLICE_EXACT) + count - 1, first_vertex, 0, 0);
}
