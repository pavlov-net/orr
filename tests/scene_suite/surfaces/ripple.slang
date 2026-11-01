// Ripples along the normal as a function of uv: displacement id 1 in the suite's linked
// kernels, `ripple` in the tessellation tests.
import displacement;

struct Ripple : IDisplacement
{
    static float3 displace(DisplacementInput i)
    {
        float h = 0.05 * sin(9.0 * i.uv[0].uv.x) * cos(7.0 * i.uv[0].uv.y);
        return i.normal * h;
    }
}
