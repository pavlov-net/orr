// The blade field's work lists for a frame, from the camera alone. `clear` zeroes the counts.
// `tiles` runs one thread per tile: a tile within range lists a draw item per slice of the
// ranks its nearest point keeps, where its box meets the view's frustum, and within the cast
// range, in view or not, its caster items over the ranks rays trace, every rank of a drawn
// field and the kept ones of a stored field: per exact cell one for each square of its strata
// that holds such a rank, and per block of cells that reaches inside the cast range and holds
// a cell that is not exact, for each square of its cells sized to its chords
// (`chord_square_cells`), one for each run of the chords its stride leaves (`block_stride`).
// `args` sizes the draw and the caster dispatch to the lists.
#include "orr_root.h"
import orr_examples_grass.blades;
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
    uint flags = d >= f.detail ? ITEM_FAR : 0;
    // Rays trace every rank of a drawn field, whatever the draw keeps.
    uint traced = f.stored != 0 ? kept : f.ranks;
    // What rays trace is decided by `cast_distance`, which is at least the distance on the
    // ground plane: a tile farther than a range that way holds nothing inside it.
    float flat = length(max(max(lo.xz - field_eye(f).xz, field_eye(f).xz - hi.xz), 0.0));
    if (flat < f.cast_range && traced > 0)
    {
        uint* counter = f.counts + FIELD_CASTER_ITEMS;
        uint squares = min(f.strata * f.strata, traced);
        // The blocks with a cell that is not exact, whose chords stand for its blades.
        uint chorded = (1u << (CELL_COUNT / CHORD_BLOCK)) - 1;
        if (flags == 0 && flat < f.exact)
        {
            chorded = 0;
            for (uint cell = 0; cell < CELL_COUNT; cell++)
            {
                if (cell_exact(f, tile, cell))
                    list(f.caster_items, counter, f.caster_capacity, tile, squares, cell, CELL_COUNT, ITEM_EXACT);
                else
                    chorded |= 1u << (cell / CHORD_BLOCK);
            }
        }
        // A block is four cells a side, and its blades stand inside it.
        float cell = f.tile_size / float(CELLS);
        float reach = 2.0 * cell * sqrt(2.0) * cast_slope(f);
        for (uint block = 0; block < CELL_COUNT / CHORD_BLOCK; block++)
        {
            float cast = cast_distance(f, block_center(f, tile, block));
            if ((chorded & (1u << block)) == 0 || cast - reach >= f.cast_range)
                continue;
            uint stride = chord_stride(f, cast);
            uint chords = (traced + stride - 1) / stride;
            uint cells = chord_square_cells(chords);
            uint run = CASTER_SLICE_CHORD / cells;
            uint runs = (chords + run - 1) / run;
            uint side = firstbithigh(cells) / 2;
            // Each square's runs, the squares in order.
            uint count = (CHORD_BLOCK / cells) * runs;
            uint first;
            InterlockedAdd(*counter, count, first);
            for (uint j = 0; j < count && first + j < f.caster_capacity; j++)
            {
                uint slice = block | (side << CHORD_SIDE_SHIFT) | ((j / runs) << CHORD_SQUARE_SHIFT) | ((j % runs) << CHORD_RUN_SHIFT);
                f.caster_items[first + j] = tile | (slice << ITEM_SLICE_SHIFT) | flags;
            }
        }
    }
    if (kept == 0)
        return;
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
