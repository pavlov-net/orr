// A field of blades derived from hashes alone: a grid of square tiles over a height raster,
// each of CELL_COUNT cells in Morton order. A tile's candidate blades come in ranks, one blade
// a cell: candidate `index` is rank `index / CELL_COUNT` of cell `index % CELL_COUNT`. The
// ranks below a share that falls with distance from the eye are drawn, so thinning leaves
// every cell its share, and consecutive candidates stand side by side. Every stage that needs
// a blade derives it again from its tile and index: the draw's task stage, the caster kernel,
// and the material resolve through `BladeGeometry`. Nothing is kept per blade.
//
// Tiles nearer the eye than `detail` draw full blades (11 vertices, 9 triangles); farther ones
// one triangle a blade. A draw item is a tile and a slice of SLICE candidate blades, with what
// the item kernel decided for the tile, so every stage reads one decision. A mesh workgroup
// draws one window of a slice: WINDOW_FULL full blades or WINDOW_FAR far ones. Its visible
// record's word names the item and the window, and a primitive's id the blade in the window
// and the triangle in the blade, so a pixel leads back to its blade without the stages'
// culling.
//
// Rays trace less: the blades themselves in the cells nearer than `exact`, each caster item one
// cell's CASTER_SLICE_EXACT ranks, and elsewhere a chord for every `cast_stride`th rank, as
// many times as wide, each item four such ranks of a block of 16 cells. Both ranges measure a
// distance that grows faster outside the view cone (`cast_distance`), so most of what is
// traced lies where the camera looks.
module blades;

import orr_shaders.math;
import orr_shaders.random;
import orr_examples.field;

public static const uint CELLS = 16;          // per side of a tile
public static const uint CELL_COUNT = CELLS * CELLS;
public static const uint SLICE = 128;
public static const uint WINDOW_FULL = 4;
public static const uint WINDOW_FAR = 16;
public static const uint FULL_VERTICES = 11;
public static const uint FULL_TRIANGLES = 9;
public static const uint FAR_VERTICES = 3;
// The most mesh workgroups a slice takes.
public static const uint WINDOWS = SLICE / WINDOW_FULL;

// A one-triangle blade is this much wider than the full blade it stands for, which keeps its
// area.
public static const float CHORD_WIDTH = 1.15;
// The most a blade widens for the blades its distance drops.
public static const float WIDEN_MAX = 8.0;

// A record word: the draw item's index in the high bits, the window in the low.
public static const uint WORD_ITEM_SHIFT = 5;
public static const uint WORD_WINDOW_MASK = 31;

// An item: the tile, then the slice, then what the item kernel decided for the tile, and for a
// caster item which kind it is.
public static const uint ITEM_TILE_MASK = 0xffff;
public static const uint ITEM_SLICE_SHIFT = 16;
public static const uint ITEM_SLICE_MASK = 0x3fff;
public static const uint ITEM_FAR = 1u << 30;    // the tile draws one triangle a blade
public static const uint ITEM_EXACT = 1u << 31;  // a caster item of one cell's blades

public uint item_tile(uint item)
{
    return item & ITEM_TILE_MASK;
}

public uint item_slice(uint item)
{
    return (item >> ITEM_SLICE_SHIFT) & ITEM_SLICE_MASK;
}

// `Field.counts`.
public static const uint FIELD_ITEMS = 0;        // draw items asked for
public static const uint FIELD_CASTER_ITEMS = 1; // caster items asked for
public static const uint FIELD_BLADES = 2;       // blades the draw's first pass kept
public static const uint FIELD_COUNTS = 4;

// The field this frame.
public struct Field
{
    public float* heights;      // side x side samples, spacing m apart from (origin_x, origin_z)
    public uint* items;         // this frame's draw items
    public uint* item_state;    // per draw item: 1 where the draw's first pass found it occluded
    public uint* caster_items;  // this frame's caster items: an exact one's slice is its cell and 256 times its run of ranks, a chord one's its block and 16 times its run
    public uint* counts;        // FIELD_*
    public uint* draw_args;     // the draw's task workgroups x, y, z; then the caster kernel's; then 1
    public float4 planes[5];    // the view's frustum in the entry's space: inside where dot(xyz, p) + w >= 0
    public uint side;
    public float spacing;
    public float origin_x;
    public float origin_z;
    public float corner_x;      // the tiles' first corner, entry space
    public float corner_z;
    public uint tiles;          // per side
    public float tile_size;     // m
    public uint ranks;          // candidate blades per cell
    public uint item_capacity;
    public uint caster_capacity;
    public float full;          // m from the eye: every candidate is drawn nearer
    public float range;         // m: none past it
    public float detail;        // m: tiles nearer draw full blades
    public float exact;         // m of `cast_distance`: cells nearer, of tiles that draw full blades, trace the blades they draw
    public float cast_range;    // m of `cast_distance`: no blade past it is traced
    public uint cast_stride;    // elsewhere a chord stands for this many ranks
    public float forward_x;     // the view's direction on the ground plane, unit
    public float forward_z;
    public float cast_cone;     // the cosine of the angle from it within which `cast_distance` is the distance
    public float cast_stretch;  // how many times the distance it is directly behind the eye
    public float wind;          // tip sway as a share of height
    public float time;          // s
    public float previous_time;
    public float eye_x;         // entry space
    public float eye_y;
    public float eye_z;
    public uint producer;       // the entry
}

public struct Blade
{
    public float3 root;         // entry space
    public float yaw;
    public float height;        // m
    public float width;         // m: half the root's
    public float lean;          // the tip's forward reach as a share of height
    public uint seed;
}

public Heights field_heights(Field* f)
{
    Heights h = { f.heights, f.side, f.spacing, f.origin_x, f.origin_z };
    return h;
}

public float3 field_eye(Field* f)
{
    return float3(f.eye_x, f.eye_y, f.eye_z);
}

// The first corner of tile `tile`.
public float2 tile_corner(Field* f, uint tile)
{
    return float2(f.corner_x, f.corner_z) + float2(tile % f.tiles, tile / f.tiles) * f.tile_size;
}

// The tallest a blade stands and the farthest its tip reaches sideways, wind included.
public float blade_height_max()
{
    return 0.7;
}

public float blade_reach_max(Field* f)
{
    return blade_height_max() * (0.6 + 0.8 * f.wind) + 0.02 * WIDEN_MAX * CHORD_WIDTH;
}

// The entry-space box of the blades of `tile`.
public void tile_box(Field* f, uint tile, out float3 lo, out float3 hi)
{
    float2 c = tile_corner(f, tile);
    Heights h = field_heights(f);
    float y0 = 1e30;
    float y1 = -1e30;
    for (uint k = 0; k < 9; k++)
    {
        float2 p = c + float2(k % 3, k / 3) * (0.5 * f.tile_size);
        float y = height_at(h, p.x, p.y);
        y0 = min(y0, y);
        y1 = max(y1, y);
    }
    // The raster between the samples lies within its slope across half a tile.
    float slack = 0.5 * (y1 - y0);
    float reach = blade_reach_max(f);
    lo = float3(c.x - reach, y0 - slack, c.y - reach);
    hi = float3(c.x + f.tile_size + reach, y1 + slack + blade_height_max(), c.y + f.tile_size + reach);
}

// The distance from the eye to the nearest point of the box `lo..hi`.
public float box_distance(Field* f, float3 lo, float3 hi)
{
    float3 e = field_eye(f);
    return length(max(max(lo - e, e - hi), 0.0));
}

// The column and row of cell `cell` in its tile.
public uint2 cell_place(uint cell)
{
    uint2 v = uint2(cell, cell >> 1) & 0x55;
    v = (v | (v >> 1)) & 0x33;
    v = (v | (v >> 2)) & 0x0f;
    return v;
}

// Around the eye the direction to a point says little, so `cast_distance` is the distance.
public static const float CAST_NEAR = 2.0;

// How far the point at `xz` counts from the eye for what rays trace: its distance on the
// ground plane, and past CAST_NEAR the part outside the view cone longer, up to
// `cast_stretch` times directly behind. The kernels that compare it must agree, so each
// operation rounds alone.
public float cast_distance(Field* f, float2 xz)
{
    precise float dx = xz.x - f.eye_x;
    precise float dz = xz.y - f.eye_z;
    precise float xx = dx * dx;
    precise float zz = dz * dz;
    precise float d2 = xx + zz;
    precise float d = sqrt(d2);
    if (d <= CAST_NEAR)
        return d;
    precise float ax = dx * f.forward_x;
    precise float az = dz * f.forward_z;
    precise float along = ax + az;
    precise float c = along / d;
    precise float gap = f.cast_cone - c;
    precise float span = f.cast_cone + 1.0;
    precise float behind = max(gap, 0.0) / span;
    precise float more = (f.cast_stretch - 1.0) * behind;
    precise float past = d - CAST_NEAR;
    precise float longer = past * more;
    precise float stretched = d + longer;
    return stretched;
}

// How much `cast_distance` changes at most per m, for a bound on it over a region: along the
// direction from the eye, and as the direction turns.
public float cast_slope(Field* f)
{
    return f.cast_stretch + (f.cast_stretch - 1.0) / (f.cast_cone + 1.0);
}

// The center of `cell` of `tile` on the ground plane.
public float2 cell_center(Field* f, uint tile, uint cell)
{
    return tile_corner(f, tile) + (float2(cell_place(cell)) + 0.5) * (f.tile_size / float(CELLS));
}

// Whether rays trace the blades `cell` of `tile` draws, for a tile that draws full blades.
public bool cell_exact(Field* f, uint tile, uint cell)
{
    return cast_distance(f, cell_center(f, tile, cell)) < f.exact;
}

// Candidate blade `index` of `tile`.
public Blade field_blade(Field* f, uint tile, uint index)
{
    uint s = pcg(pcg(tile) + index * 0x9e3779b9u);
    float2 cell = float2(cell_place(index % CELL_COUNT)) + float2(pcg_float(s), pcg_float(s + 1));
    float2 xz = tile_corner(f, tile) + cell * (f.tile_size / float(CELLS));
    Blade b;
    b.root = float3(xz.x, height_at(field_heights(f), xz.x, xz.y), xz.y);
    b.yaw = TWO_PI * pcg_float(s + 2);
    b.height = 0.3 + 0.4 * pcg_float(s + 3);
    b.width = 0.012 + 0.008 * pcg_float(s + 4);
    b.lean = 0.15 + 0.45 * pcg_float(s + 5);
    b.seed = s;
    return b;
}

// The share of a tile's candidates drawn `distance` m from the eye: all within `full`, then
// as many as keep their number per pixel, dissolving over the last fifth of `range`.
public float blade_keep(Field* f, float distance)
{
    float near = f.full / max(distance, f.full);
    return near * near * saturate((f.range - distance) / (0.2 * f.range));
}

// How much wider a drawn blade is where `keep` of the candidates are drawn, holding the
// ground they cover.
public float blade_widen(float keep)
{
    return min(1.0 / max(keep, 1.0 / WIDEN_MAX), WIDEN_MAX);
}

// What draws of blade `b`, candidate `index` of its tile: its width's scale, or 0 where its
// distance drops it. The last rank kept thins blade by blade.
public float blade_drawn(Field* f, Blade b, uint index)
{
    float keep = blade_keep(f, distance(field_eye(f), b.root));
    return float(index / CELL_COUNT) + pcg_float(b.seed + 8) < keep * float(f.ranks) ? blade_widen(keep) : 0.0;
}

// The ranks of a tile that hold a blade to draw, where the share its nearest point keeps is
// `keep`.
public uint kept_ranks(Field* f, float keep)
{
    return min(uint(ceil(keep * float(f.ranks))), f.ranks);
}

// The tip's sideways sway at wind time `time`, in x and z.
public float2 blade_sway(Field* f, Blade b, float time)
{
    float3 r = b.root;
    float gust = sin(time * 1.7 + r.x * 0.35 + r.z * 0.21) * 0.6 + sin(time * 3.1 + r.x * 0.9 + float(b.seed & 255) * 0.03) * 0.2;
    return f.wind * b.height * gust * float2(1.0, 0.4);
}

// A full blade's vertex `k`: across (-1 left, 1 right, 0 the tip) and along (0 root, 1 tip).
public float2 full_template(uint k)
{
    return k == 10 ? float2(0.0, 1.0) : float2((k & 1) != 0 ? 1.0 : -1.0, float(k >> 1) * 0.2);
}

// A one-triangle blade's vertex `k`: the root's two corners and the tip.
public float2 far_template(uint k)
{
    return k == 2 ? float2(0.0, 1.0) : float2(k == 1 ? 1.0 : -1.0, 0.0);
}

public uint3 full_triangle(uint t)
{
    uint j = t & ~1u;
    return t == 8 ? uint3(8, 9, 10) : (t & 1) == 0 ? uint3(j, j + 1, j + 3) : uint3(j, j + 3, j + 2);
}

// The point of blade `b` at `at` of a template, `scale` times as wide, swayed by `sway`.
public float3 blade_point(Blade b, float scale, float2 sway, float2 at)
{
    float3 across = float3(cos(b.yaw), 0.0, sin(b.yaw));
    float3 forward = float3(-across.z, 0.0, across.x);
    float bend = at.y * at.y;
    float taper = 1.0 - 0.85 * at.y;
    return b.root + across * (at.x * b.width * scale * taper) + forward * (b.lean * b.height * bend)
        + float3(sway.x, 0.0, sway.y) * bend + float3(0.0, b.height * at.y, 0.0);
}

// The unit normal of blade `b` there: its face's, turned outward toward its edges.
public float3 blade_normal(Blade b, float2 sway, float2 at)
{
    float3 across = float3(cos(b.yaw), 0.0, sin(b.yaw));
    float3 forward = float3(-across.z, 0.0, across.x);
    float3 along = forward * (2.0 * b.lean * b.height * at.y) + float3(sway.x, 0.0, sway.y) * (2.0 * at.y) + float3(0.0, b.height, 0.0);
    return normalize(normalize(cross(along, across)) + across * (0.5 * at.x));
}

// How far the middle of full blade `b` stands off the triangle of its root and tip.
public float blade_sag(Blade b, float2 sway)
{
    float2 forward = float2(-sin(b.yaw), cos(b.yaw));
    return 0.25 * length(forward * (b.lean * b.height) + sway);
}

// The blade's color, linear Rec.2020, as its vertices' word for the surface: 8 bits a channel.
public uint blade_color(Blade b)
{
    float dry = pcg_float(b.seed + 6);
    float3 c = lerp(float3(0.16, 0.36, 0.08), float3(0.42, 0.44, 0.14), dry * dry);
    uint3 q = uint3(round(saturate(c) * 255.0));
    return q.x | (q.y << 8) | (q.z << 16);
}

public float3 blade_color_from(uint word)
{
    return float3(word & 255, (word >> 8) & 255, (word >> 16) & 255) / 255.0;
}

// The candidates a caster item covers, which emits one cluster of those it keeps: full blades
// (at most 154 vertices and 126 triangles) or chords (192 and 64).
public static const uint CASTER_SLICE_EXACT = 14;
public static const uint CASTER_SLICE_CHORD = 64;
// A chord item's cells, and the strided ranks it takes of each.
public static const uint CHORD_BLOCK = 16;
public static const uint CHORD_RUN = CASTER_SLICE_CHORD / CHORD_BLOCK;

// Candidate `lane` of caster item `item`: its index in the tile.
public uint caster_candidate(Field* f, uint item, uint lane)
{
    uint slice = item_slice(item);
    if ((item & ITEM_EXACT) != 0)
        return ((slice / CELL_COUNT) * CASTER_SLICE_EXACT + lane) * CELL_COUNT + slice % CELL_COUNT;
    uint rank = ((slice / CHORD_BLOCK) * CHORD_RUN + lane / CHORD_BLOCK) * f.cast_stride;
    return rank * CELL_COUNT + (slice % CHORD_BLOCK) * CHORD_BLOCK + lane % CHORD_BLOCK;
}

// How wide a chord is for a blade drawn `widen` times as wide.
public float chord_width(Field* f, float widen)
{
    return min(widen * CHORD_WIDTH * float(f.cast_stride), WIDEN_MAX * CHORD_WIDTH);
}
