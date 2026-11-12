// What rays trace of the blade field: one workgroup per caster item derives its candidates
// and emits those that are drawn as one cluster of the entry, with the cluster topology of
// that many: an exact cell's blades as they are drawn, and elsewhere one-triangle chords, but
// for the blades of exact cells. Each vertex's word is its blade's color, as the draw's linked
// geometry gives a drawn triangle.
#include "orr_root.h"
import orr_examples.blades;
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
    uint blade = caster_candidate(f, item, lane);
    if (lane < 2)
        alive[lane] = 0;
    GroupMemoryBarrierWithGroupSync();

    Blade b = (Blade)0;
    float cast = 0.0;
    if (lane < size && blade < f.ranks * CELL_COUNT)
    {
        b = field_blade(f, tile, blade);
        cast = blade_drawn(f, b, blade);
        if (!exact)
        {
            bool traced = (item & ITEM_FAR) == 0 && cell_exact(f, tile, blade % CELL_COUNT);
            cast = traced || cast_distance(f, b.root.xz) >= f.cast_range ? 0.0 : chord_width(f, cast);
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
        float2 sway = blade_sway(f, b, f.time);
        float2 before = blade_sway(f, b, f.previous_time);
        uint color = blade_color(b);
        for (uint k = 0; k < per; k++)
        {
            float2 at = exact ? full_template(k) : far_template(k);
            uint v = first_vertex + below * per + k;
            batch.positions[v] = blade_point(b, cast, sway, at);
            if (batch.previous != nullptr)
                batch.previous[v] = blade_point(b, cast, before, at);
            batch.attrs[v] = producer_attributes(blade_normal(b, sway, at), float2(0.5 + 0.5 * at.x, at.y));
            batch.vertex_user[v] = color;
        }
    }
    if (lane == 0)
        emit_template_cluster(batch, (exact ? 0 : CASTER_SLICE_EXACT) + count - 1, first_vertex, 0, 0);
}
