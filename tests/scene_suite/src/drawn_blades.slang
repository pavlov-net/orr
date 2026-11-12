// The blades of `blade_shape.slang` drawn from a view's opaque hook: one mesh workgroup per
// DRAWN_GROUP blades, whose visible record's word is the workgroup's index. A primitive's id
// is three times its blade's place in the workgroup plus its triangle.
#include "orr_root.h"
import orr_scene.drawn_raster;
import scene_suite.blade_shape;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    RasterGpu* raster;
    BladeField* field;
}
ORR_ROOT(Root);

struct Primitive
{
    uint id : SV_PrimitiveID;
}

groupshared uint record;

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void mesh(
    uint3 group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    out indices uint3 triangles[3 * DRAWN_GROUP],
    out vertices DrawnVertex vertices[5 * DRAWN_GROUP],
    out primitives Primitive primitives[3 * DRAWN_GROUP])
{
    BladeField* f = root.field;
    uint first = group.x * DRAWN_GROUP;
    uint blades = first < f.count ? min(f.count - first, DRAWN_GROUP) : 0;
    if (thread == 0)
    {
        uint visible;
        record = blades > 0 && drawn_visible(root.scene, root.raster, f.producer, 0, group.x, visible) ? visible : 0xffffffffu;
    }
    GroupMemoryBarrierWithGroupSync();
    Placement p;
    if (record == 0xffffffffu || !drawn_placement(root.scene, root.view, f.producer, false, p))
    {
        SetMeshOutputCounts(0, 0);
        return;
    }
    SetMeshOutputCounts(5 * blades, 3 * blades);
    if (thread < 5 * blades)
    {
        DrawnVertex o;
        o.position = drawn_clip(root.view, place(p, blade_vertex(f.z, first + thread / 5, thread % 5, f.time)));
        o.visible = record;
        vertices[thread] = o;
    }
    if (thread < 3 * blades)
    {
        triangles[thread] = blade_triangle(thread % 3) + 5 * (thread / 3);
        Primitive o;
        o.id = thread;
        primitives[thread] = o;
    }
}

[shader("pixel")]
void pixel(DrawnVertex v, uint primitive: SV_PrimitiveID)
{
    drawn_pixel(root.raster, v.position, v.visible, primitive);
}
