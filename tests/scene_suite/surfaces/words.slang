// A surface that reports a hit's producer words: base color holds the cluster word and the
// first two corners' vertex words, emission the third corner's, the user tag and the hit
// kind.
module words;

import surface;

public struct Words : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        depth_offset = 0.0;
        GSurface s = (GSurface)0;
        s.normal = i.normal;
        s.geometric_normal = i.geometric_normal;
        s.bsdf.base_color = float3(float(i.cluster_user), float(i.vertex_user.x), float(i.vertex_user.y));
        s.bsdf.emission = float3(float(i.vertex_user.z), float(i.user_tag), float(i.context.kind));
        return s;
    }
}
