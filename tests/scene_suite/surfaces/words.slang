// A surface that reports a hit's producer words: base color holds the cluster word and the
// first two corners' vertex words, emission the third corner's, the user tag and the hit
// kind.
module words;

import orr_scene.surface;

public struct Words : ISurface
{
    public override static bool surface(SurfaceInput i, out GSurface g, out float depth_offset)
    {
        depth_offset = 0.0;
        g = GSurface();
        g.normal = i.normal;
        g.geometric_normal = i.geometric_normal;
        g.bsdf.base_color = float3(float(i.cluster_user), float(i.vertex_user.x), float(i.vertex_user.y));
        g.bsdf.emission = float3(float(i.vertex_user.z), float(i.user_tag), float(i.context.kind));
        return true;
    }
}
