// A surface that reports its hit's context: base color holds the kind, the path depth and
// the cone width, specular roughness the cone spread.
module probe;

import orr_scene.surface;

public struct Probe : ISurface
{
    public override static void adjust(SurfaceInput i, inout PlainSurface s, inout float depth_offset)
    {
        s.base_color = float3(float(i.context.kind), float(i.context.depth), i.context.cone_width);
        s.specular_roughness = i.context.cone_spread;
    }
}
