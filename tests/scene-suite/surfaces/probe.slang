// A surface that reports its hit's context: base color holds the kind, the path depth and
// the cone width, specular roughness the cone spread.
module probe;

import surface;

public struct Probe : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        GSurface s = StandardSurface.surface(i, depth_offset);
        s.bsdf.base_color = float3(float(i.context.kind), float(i.context.depth), i.context.cone_width);
        s.bsdf.specular_roughness = i.context.cone_spread;
        return s;
    }
}
