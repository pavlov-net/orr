// The blade field's work lists for a frame, from the camera alone. `clear` zeroes the counts.
// `tiles` runs one thread per tile: a tile within range lists a draw item per slice of the
// ranks its nearest point keeps, where its box meets the view's frustum, and within the cast
// range, in view or not, its caster items: per exact cell one for each run of those ranks, and
// per block of cells that reaches inside the cast range one for each run of the ranks chords
// stand for. `args` sizes the draw and the caster dispatch to the lists.
#include "orr_root.h"
import orr_examples.blades;
import orr_scene.visibility;

struct Root
{
    Field* field;
    uint64_t _pad0;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void clear()
{
    for (uint i = 0; i < FIELD_COUNTS; i++)
        root.field.counts[i] = 0;
}

// `count` items of `tile` with `flags` from the list `items` of `capacity`: slices
// `first_slice`, then every `step` on.
void list(uint* items, uint* counter, uint capacity, uint tile, uint count, uint first_slice, uint step, uint flags)
{
    uint first;
    InterlockedAdd(*counter, count, first);
    for (uint j = 0; j < count && first + j < capacity; j++)
        items[first + j] = tile | ((first_slice + j * step) << ITEM_SLICE_SHIFT) | flags;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void tiles(uint3 id: SV_DispatchThreadID)
{
    Field* f = root.field;
    uint tile = id.x;
    if (tile >= f.tiles * f.tiles)
        return;
    float3 lo, hi;
    tile_box(f, tile, lo, hi);
    float d = box_distance(f, lo, hi);
    uint kept = kept_ranks(f, blade_keep(f, d));
    if (kept == 0)
        return;
    uint flags = d >= f.detail ? ITEM_FAR : 0;
    // What rays trace is decided by `cast_distance`, which is at least the distance on the
    // ground plane: a tile farther than a range that way holds nothing inside it.
    float flat = length(max(max(lo.xz - field_eye(f).xz, field_eye(f).xz - hi.xz), 0.0));
    if (flat < f.cast_range)
    {
        uint* counter = f.counts + FIELD_CASTER_ITEMS;
        uint runs = (kept + CASTER_SLICE_EXACT - 1) / CASTER_SLICE_EXACT;
        for (uint cell = 0; cell < CELL_COUNT && flags == 0 && flat < f.exact; cell++)
        {
            if (cell_exact(f, tile, cell))
                list(f.caster_items, counter, f.caster_capacity, tile, runs, cell, CELL_COUNT, ITEM_EXACT);
        }
        uint chords = (kept + f.cast_stride - 1) / f.cast_stride;
        runs = (chords + CHORD_RUN - 1) / CHORD_RUN;
        // A block is four cells a side, and its blades stand inside it.
        float cell = f.tile_size / float(CELLS);
        float reach = 2.0 * cell * sqrt(2.0) * cast_slope(f);
        for (uint block = 0; block < CELL_COUNT / CHORD_BLOCK; block++)
        {
            float2 corner = tile_corner(f, tile) + float2(cell_place(block * CHORD_BLOCK)) * cell;
            if (cast_distance(f, corner + 2.0 * cell) - reach < f.cast_range)
                list(f.caster_items, counter, f.caster_capacity, tile, runs, block, CHORD_BLOCK, flags);
        }
    }
    float3 center = 0.5 * (lo + hi);
    float radius = 0.5 * length(hi - lo);
    for (uint k = 0; k < 5; k++)
    {
        float4 p = f.planes[k];
        if (dot(p.xyz, center) + p.w < -radius * length(p.xyz))
            return;
    }
    list(f.items, f.counts + FIELD_ITEMS, f.item_capacity, tile, kept * CELL_COUNT / SLICE, 0, 1, flags);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void args()
{
    Field* f = root.field;
    write_grid(f.draw_args, min(f.counts[FIELD_ITEMS], f.item_capacity));
    write_grid(f.draw_args + 3, min(f.counts[FIELD_CASTER_ITEMS], f.caster_capacity));
    f.draw_args[6] = 1;
}
