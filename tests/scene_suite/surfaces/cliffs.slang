// Terraces along u with walls a few centimeters wide: displacement id 2 in the suite's
// linked kernels, `cliffs` in the tessellation tests.
import orr_scene.displacement;

struct Cliffs : IDisplacement
{
    static float3 displace(DisplacementInput i)
    {
        float h = 0.2 * clamp(12.0 * sin(9.0 * i.uv[0].uv.x), -1.0, 1.0);
        return i.normal * h;
    }
}
