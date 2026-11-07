// An unlit surface that shows its hit's path depth: red at depths 0 and 1, green from 2 on.
module context;

import orr_scene.surface;

public static const float3 NEAR = float3(40.0, 4.0, 4.0);
public static const float3 FAR = float3(4.0, 40.0, 4.0);

// Written out rather than built on `StandardSurface`: a linked type that calls it adds a copy of
// it at every surface site of every kernel, and the path kernels' cold compile grows faster than
// their size.
public struct Context : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        depth_offset = 0.0;
        GSurface s = (GSurface)0;
        s.normal = i.normal;
        s.geometric_normal = i.geometric_normal;
        s.bsdf.specular_ior = 1.5;
        s.bsdf.coat_normal = i.normal;
        s.bsdf.emission = i.context.depth < 2 ? NEAR : FAR;
        s.flags = GBUFFER_UNLIT;
        return s;
    }
}
