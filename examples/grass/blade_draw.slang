// The blade field's draw, from a view's opaque hook: one task workgroup per draw item, one
// mesh workgroup per window of its slice with a blade to draw. The task stage tests the item's
// tile against the pass's HZB and each window's square of cells against the frustum, derives
// the blades of the windows in view, keeps the ones their distance draws, in lane order, and
// takes a visible record per window with any; the mesh stage expands the kept blades of its
// window and reads nothing but the payload.
#include "orr_root.h"
import orr_examples_grass.blades;
import orr_scene.drawn_raster;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    RasterGpu* raster;
    Field* field;
    uint second;         // the raster's second pass: the items its first found occluded
    uint _pad0;
    uint64_t _pad1;
}
ORR_ROOT(Root);

// A blade the task stage kept.
struct Kept
{
    float3 root;
    float yaw;
    float height;
    float width;         // as drawn
    float lean;
    uint lane;           // its place in the slice
    float2 sway;
}

struct Payload
{
    Placement placement; // the entry's, relative to the eye
    uint far;            // one triangle a blade
    uint windows[WINDOWS]; // per mesh workgroup: its window, its first kept blade << 8, their count << 16
    uint visible[WINDOWS]; // and its visible record
    Kept kept[SLICE];
}

groupshared Payload payload;
groupshared uint alive[SLICE / 32];
groupshared uint recorded;           // the windows with a record
groupshared uint records[WINDOWS];
groupshared uint shown;
groupshared float2 tile_heights;     // the lowest and highest a blade of the tile reaches

// Whether the frustum holds any of the square of cells that window `window` of `size` blades
// of `item` draws in, by the sphere around what its blades reach.
bool window_in_view(Field* f, uint item, uint window, uint size)
{
    float cell = f.tile_size / float(CELLS);
    float side = size == WINDOW_FULL ? 2.0 * cell : 4.0 * cell;
    uint first = (item_slice(item) * SLICE + window * size) % CELL_COUNT;
    float2 corner = tile_corner(f, item_tile(item)) + float2(cell_place(first)) * cell;
    float3 center = float3(corner.x + 0.5 * side, 0.5 * (tile_heights.x + tile_heights.y), corner.y + 0.5 * side);
    float half_side = 0.5 * side + blade_reach_max(f);
    float radius = length(float3(half_side, 0.5 * (tile_heights.y - tile_heights.x), half_side));
    for (uint k = 0; k < 5; k++)
    {
        float4 p = f.planes[k];
        if (dot(p.xyz, center) + p.w < -radius * length(p.xyz))
            return false;
    }
    return true;
}

// Kept blades in lanes below `lane`.
uint kept_below(uint lane)
{
    uint n = 0;
    for (uint w = 0; w < SLICE / 32; w++)
    {
        uint bits = alive[w];
        if (lane < 32 * w + 32)
            bits &= lane > 32 * w ? (1u << (lane - 32 * w)) - 1 : 0;
        n += countbits(bits);
    }
    return n;
}

[shader("amplification")]
[numthreads(SLICE, 1, 1)]
void task(uint3 group: SV_GroupID, uint lane: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    Field* f = root.field;
    uint index = grid_index(group);
    bool listed = index < min(f.counts[FIELD_ITEMS], f.item_capacity);
    uint item = listed ? f.items[index] : 0;
    uint tile = item_tile(item);
    bool second = root.second != 0;
    if (lane < SLICE / 32)
        alive[lane] = 0;
    if (lane == 0)
    {
        recorded = 0;
        bool go = listed && (!second || f.item_state[index] != 0);
        Placement now = (Placement)0;
        if (go)
            go = drawn_placement(s, v, f.producer, false, now);
        if (go)
        {
            Placement before;
            if (!drawn_placement(s, v, f.producer, true, before))
                before = now;
            float3 lo, hi;
            tile_box(f, tile, lo, hi);
            tile_heights = float2(lo.y, hi.y);
            bool occluded = drawn_occluded(v, r, second, now, before, lo, hi);
            go = !occluded;
            if (!second)
                f.item_state[index] = occluded ? 1 : 0;
        }
        else if (listed && !second)
        {
            f.item_state[index] = 0;
        }
        shown = go ? 1 : 0;
        payload.placement = now;
        payload.far = (item & ITEM_FAR) != 0 ? 1 : 0;
    }
    GroupMemoryBarrierWithGroupSync();

    bool far = (item & ITEM_FAR) != 0;
    uint size = far ? WINDOW_FAR : WINDOW_FULL;
    Kept k = (Kept)0;
    bool keep = false;
    uint blade = item_slice(item) * SLICE + lane;
    if (shown != 0 && blade < f.ranks * CELL_COUNT && window_in_view(f, item, lane / size, size))
    {
        Blade b = field_blade(f, tile, blade);
        float widen = blade_drawn(f, b, blade);
        if (widen > 0.0)
        {
            keep = true;
            k.root = b.root;
            k.yaw = b.yaw;
            k.height = b.height;
            k.width = b.width * widen * (far ? CHORD_WIDTH : 1.0);
            k.lean = b.lean;
            k.lane = lane;
            k.sway = blade_sway(f, b, false);
            InterlockedOr(alive[lane >> 5], 1u << (lane & 31));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (keep)
        payload.kept[kept_below(lane)] = k;
    bool window = lane < SLICE / size;
    uint first = window ? kept_below(lane * size) : 0;
    uint count = window ? kept_below(lane * size + size) - first : 0;
    if (count > 0)
    {
        uint visible;
        if (drawn_visible(s, r, f.producer, 0, (index << WORD_ITEM_SHIFT) | lane, visible))
        {
            records[lane] = visible;
            InterlockedOr(recorded, 1u << lane);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (count > 0 && (recorded & (1u << lane)) != 0)
    {
        uint g = countbits(recorded & ((1u << lane) - 1));
        payload.windows[g] = lane | (first << 8) | (count << 16);
        payload.visible[g] = records[lane];
    }
    if (lane == 0 && !second)
        InterlockedAdd(f.counts[FIELD_BLADES], kept_below(SLICE));
    DispatchMesh(countbits(recorded), 1, 1, payload);
}

struct Primitive
{
    uint id : SV_PrimitiveID;
}

// A window's most vertices and triangles, of full blades or far ones.
static const uint MESH_VERTICES = WINDOW_FULL * FULL_VERTICES > WINDOW_FAR * FAR_VERTICES ? WINDOW_FULL * FULL_VERTICES : WINDOW_FAR * FAR_VERTICES;
static const uint MESH_TRIANGLES = WINDOW_FULL * FULL_TRIANGLES > WINDOW_FAR ? WINDOW_FULL * FULL_TRIANGLES : WINDOW_FAR;

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void mesh(
    uint3 group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    in payload Payload p,
    out indices uint3 triangles[MESH_TRIANGLES],
    out vertices DrawnVertex vertices[MESH_VERTICES],
    out primitives Primitive primitives[MESH_TRIANGLES])
{
    uint w = p.windows[group.x];
    uint window = w & 255;
    uint first = (w >> 8) & 255;
    uint count = w >> 16;
    bool far = p.far != 0;
    uint per_vertices = far ? FAR_VERTICES : FULL_VERTICES;
    uint per_triangles = far ? 1 : FULL_TRIANGLES;
    uint size = far ? WINDOW_FAR : WINDOW_FULL;
    SetMeshOutputCounts(count * per_vertices, count * per_triangles);
    for (uint i = thread; i < count * per_vertices; i += 32)
    {
        Kept k = p.kept[first + i / per_vertices];
        Blade b;
        b.root = k.root;
        b.yaw = k.yaw;
        b.height = k.height;
        b.width = k.width;
        b.lean = k.lean;
        b.seed = 0;
        uint corner = i % per_vertices;
        float3 x = blade_point(b, 1.0, k.sway, far ? far_template(corner) : full_template(corner));
        DrawnVertex o;
        o.position = drawn_clip(root.view, place(p.placement, x));
        o.visible = p.visible[group.x];
        vertices[i] = o;
    }
    for (uint t = thread; t < count * per_triangles; t += 32)
    {
        uint j = t / per_triangles;
        uint triangle = t % per_triangles;
        triangles[t] = (far ? uint3(0, 1, 2) : full_triangle(triangle)) + j * per_vertices;
        Primitive o;
        o.id = (p.kept[first + j].lane - window * size) * per_triangles + triangle;
        primitives[t] = o;
    }
}

[shader("pixel")]
void pixel(DrawnVertex v, uint primitive: SV_PrimitiveID)
{
    drawn_pixel(root.raster, v.position, v.visible, primitive);
}
