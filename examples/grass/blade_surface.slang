// What orr's kernels link for the blade field: `BladeGeometry` derives a drawn pixel's triangle
// again from its record's word and its primitive's id, and `BladeSurface` shades blades,
// drawn or traced, by the color its vertices' words carry.
module blade_surface;

import orr_examples_grass.blades;
import orr_scene.drawn;
import orr_scene.surface;

public struct BladeGeometry : IDrawn
{
    public static DrawnTriangle triangle(DrawnInput i)
    {
        Field* f = (Field*)i.data;
        uint item = f.items[i.word >> WORD_ITEM_SHIFT];
        bool far = (item & ITEM_FAR) != 0;
        uint per = far ? 1 : FULL_TRIANGLES;
        uint window = (i.word & WORD_WINDOW_MASK) * (far ? WINDOW_FAR : WINDOW_FULL);
        uint index = item_slice(item) * SLICE + window + i.primitive / per;
        Blade b = field_blade(f, item_tile(item), index);
        float widen = blade_widen(blade_keep(f, distance(field_eye(f), b.root))) * (far ? CHORD_WIDTH : 1.0);
        float2 sway = blade_sway(f, b, false);
        float2 before = blade_sway(f, b, true);
        uint3 corners = far ? uint3(0, 1, 2) : full_triangle(i.primitive % per);
        DrawnTriangle t;
        for (uint j = 0; j < 3; j++)
        {
            float2 at = far ? far_template(corners[j]) : full_template(corners[j]);
            t.position[j] = blade_point(b, widen, sway, at);
            t.previous[j] = blade_point(b, widen, before, at);
            t.normal[j] = blade_normal(b, sway, at);
            t.uv[j] = float2(0.5 + 0.5 * at.x, at.y);
        }
        t.cluster_user = 0;
        t.vertex_user = uint3(blade_color(b));
        // An exact cell traces this blade as drawn where it is drawn no wider, and a far blade's
        // chord lies in its plane.
        bool exact = !far && widen <= 1.0 && cell_exact(f, item_tile(item), index % CELL_COUNT)
            && blade_traced(f, b, cast_distance(f, b.root.xz));
        t.error = far || exact ? 0.0 : blade_sag(b, sway);
        return t;
    }
}

public struct BladeSurface : ISurface
{
    public override static void adjust(SurfaceInput i, inout GSurface g, inout float depth_offset)
    {
        // Darker toward the root, where less light reaches.
        float3 color = blade_color_from(i.vertex_user.x) * lerp(0.35, 1.0, saturate(i.uv[0].uv.y));
        g.bsdf.base_color = color;
        g.bsdf.subsurface_color = color;
    }
}
