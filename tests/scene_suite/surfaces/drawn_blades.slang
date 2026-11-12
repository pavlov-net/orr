// The linked geometry of the drawn blade field (`src/drawn_blades.slang`): a record's word is
// its mesh workgroup's index, and the triangle the mesh stage gave the primitive's id comes
// back with the words `blades.slang` gives the same blade in a batch.
module drawn_blades;

import orr_scene.drawn;
import scene_suite.blade_shape;

public struct DrawnBlades : IDrawn
{
    public static DrawnTriangle triangle(DrawnInput i)
    {
        BladeField* f = (BladeField*)i.data;
        uint blade = i.word * DRAWN_GROUP + i.primitive / 3;
        uint3 corners = blade_triangle(i.primitive % 3);
        DrawnTriangle t;
        for (uint j = 0; j < 3; j++)
        {
            uint k = corners[j];
            t.position[j] = blade_vertex(f.z, blade, k, f.time);
            t.previous[j] = blade_vertex(f.z, blade, k, f.previous_time);
            t.normal[j] = float3(0.0, 0.0, 1.0);
            t.uv[j] = blade_uv(k);
            t.vertex_user[j] = 8 * blade + k;
        }
        t.cluster_user = f.base + blade;
        t.error = 0.0;
        return t;
    }
}
