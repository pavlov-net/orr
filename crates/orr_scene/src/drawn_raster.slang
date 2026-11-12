// What the task, mesh and pixel stages of a drawn entry's draw call, from a view's opaque
// hook: the entry's placement, a test against the pass's HZB, a visible record per mesh
// workgroup, and the visibility buffer write.
//
// A draw's root holds the pass's scene, view and raster record, which the hook hands it. The
// mesh stage outputs `DrawnVertex` vertices and gives each primitive an `SV_PrimitiveID` below
// 128; the pixel stage calls `drawn_pixel`. Each mesh workgroup takes one visible record,
// whose word and the primitive's id are all the entry's linked `IDrawn` gets to derive the
// triangle again (`orr_scene.drawn`): what the stages decide by rounding, such as a level of
// detail, goes into the word, since the resolve may round the other way.
//
// The view's raster draws in two passes while it holds the previous frame's HZB
// (`RASTER_OCCLUSION`). A stage that culls by `drawn_occluded` keeps what the first pass
// rejects, by any record of its own, and its second-pass draw tests and draws only that.
module drawn_raster;

import orr_shaders.atomic;
import orr_scene.producer;
__exported import orr_scene.visibility;

public struct DrawnVertex
{
    public float4 position : SV_Position;
    public nointerpolation uint visible : VISIBLE;
}

// The placement of producer entry `entry` relative to the view's eye, this frame's or with
// `previous` last frame's relative to this frame's eye; false when its group hides it.
public bool drawn_placement(SceneRoot* s, ViewConstants* v, uint entry, bool previous, out Placement p)
{
    return producer_placement(s, s.producers[entry], previous, v.eye_cell.xyz, v.eye_offset.xyz, uint(v.eye_cell.w), p);
}

// The clip-space position the raster draws eye-relative `relative` at.
public float4 drawn_clip(ViewConstants* v, float3 relative)
{
    return mul(v.clip_from_view_jittered, float4(view_from_relative(v, relative), 1.0));
}

// Whether the pass with raster record `r` finds the box `lo..hi` of the entry's space
// occluded: the second pass (`second`) through placement `now` against the first pass's HZB,
// the first through last frame's placement `before` against last frame's HZB. False in a first
// pass that holds no HZB.
public bool drawn_occluded(ViewConstants* v, RasterGpu* r, bool second, Placement now, Placement before, float3 lo, float3 hi)
{
    if (second)
        return box_occluded(v, r, now, lo, hi, false);
    return (r.flags & RASTER_OCCLUSION) != 0 && box_occluded(v, r, before, lo, hi, true);
}

// A visible record for the triangles one mesh workgroup draws of producer entry `entry`, with
// material `material` of the entry's materials: its index for `DrawnVertex.visible`. `word`
// is the stages' own. False when the material is outside the entry's or the view's list is
// full, which the list's next frames grow out of; the workgroup then draws nothing.
public bool drawn_visible(SceneRoot* s, RasterGpu* r, uint entry, uint material, uint word, out uint visible)
{
    visible = 0;
    ProducerGpu p = s.producers[entry];
    if (material >= p.material_count)
        return false;
    VisibleCluster vc;
    vc.page = nullptr;
    vc.instance = s.producer_base + entry;
    vc.cluster = word;
    vc.local = VISIBLE_DRAWN;
    vc.material = p.materials[material];
    return add_visible(r, vc, visible);
}

// Writes the fragment at `position` (`SV_Position`) of primitive `primitive` of visible record
// `visible` where it is the nearest the pixel has.
public void drawn_pixel(RasterGpu* r, float4 position, uint visible, uint primitive)
{
    uint2 pixel = uint2(position.xy);
    atomic_max_u64(r.gbuffer.visibility + pixel.y * r.gbuffer.extent.x + pixel.x, visibility_word(position.z, visible, primitive));
}
