// A card that shows a tilted quad behind it: coverage keeps the pixels whose view ray meets
// the quad, and the depth offset moves each one back onto it. The quad lies on the plane
// z = -DEPTH - SLOPE x in the card's object space, within HALF of its axis in x and y.
module parallax;

import orr.math;
import surface;

static const float SLOPE = 0.3;
static const float DEPTH = 0.2;
static const float HALF = 0.4;

// The object-space point where the view ray meets the plane, and the render-space distance
// from the card to it.
float3 behind(SurfaceInput i, out float t)
{
    float3 d = mul(linear_inverse(i.object_to_render), i.view_direction);
    float3 p = i.object_position;
    t = (-DEPTH - SLOPE * p.x - p.z) / (d.z + SLOPE * d.x);
    return p + t * d;
}

public struct Parallax : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        GSurface s = StandardSurface.surface(i, depth_offset);
        float t;
        behind(i, t);
        depth_offset = max(t, 0.0);
        float3 n = normalize(mul(transpose(linear_inverse(i.object_to_render)), float3(SLOPE, 0.0, 1.0)));
        s.normal = n;
        s.geometric_normal = n;
        return s;
    }

    public override static float coverage(SurfaceInput i)
    {
        float t;
        float3 h = behind(i, t);
        return t >= 0.0 && all(abs(h.xy) <= HALF) ? 1.0 : 0.0;
    }
}
