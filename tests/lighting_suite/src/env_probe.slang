// Sampling the baked environment: a histogram of `environment_sample` over equal-area
// latitude-longitude bins, and the bins' probabilities integrated from `environment_pdf`.
#include "orr/root.h"
import orr.math;
import orr.random;
import environment;

static const uint BINS_PHI = 32;
static const uint BINS_Y = 16;

struct Root
{
    EnvironmentGpu* environment;
    uint* histogram;
    // Samples whose returned density differs from `environment_pdf` by more than 1e-4.
    uint* mismatches;
    float* bin_probability;
    uint samples_per_thread;
    uint seed;
    // Sub-samples per bin side for the integral.
    uint subdivisions;
    uint _pad;
}
ORR_ROOT(Root);

uint bin_of(float3 d)
{
    float phi = atan2(d.z, d.x) * INV_TWO_PI + 0.5;
    uint i = min(uint(phi * BINS_PHI), BINS_PHI - 1);
    uint j = min(uint((d.y * 0.5 + 0.5) * BINS_Y), BINS_Y - 1);
    return j * BINS_PHI + i;
}

float3 bin_dir(float2 p)
{
    float phi = (p.x - 0.5) * TWO_PI;
    float y = 2.0 * p.y - 1.0;
    float r = sqrt(max(0.0, 1.0 - y * y));
    return float3(r * cos(phi), y, r * sin(phi));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sample(uint3 id: SV_DispatchThreadID)
{
    for (uint k = 0; k < root.samples_per_thread; k++)
    {
        uint4 h = pcg4d(uint4(id.x, k, root.seed, 7));
        float3 u = float3(h.xyz >> 8) * (1.0 / 16777216.0);
        float pdf;
        float3 d = environment_sample(root.environment, u, pdf);
        InterlockedAdd(root.histogram[bin_of(d)], 1);
        float check = environment_pdf(root.environment, d);
        if (abs(check - pdf) > 1e-4 * pdf)
            InterlockedAdd(root.mismatches[0], 1);
    }
}

// One workgroup per bin; each writes its bin's probability.
[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint bin = group.x;
    uint n = root.subdivisions;
    float2 lo = float2(bin % BINS_PHI, bin / BINS_PHI) / float2(BINS_PHI, BINS_Y);
    float sum = 0.0;
    for (uint k = lane; k < n * n; k += 64)
    {
        float2 p = lo + (float2(k % n, k / n) + 0.5) / float(n) / float2(BINS_PHI, BINS_Y);
        sum += environment_pdf(root.environment, bin_dir(p));
    }
    groupshared_sum(lane, sum);
    if (lane == 0)
        root.bin_probability[bin] = sums[0] * (4.0 * PI / float(BINS_PHI * BINS_Y)) / float(n * n);
}

static groupshared float sums[64];

void groupshared_sum(uint lane, float v)
{
    sums[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32; s > 0; s >>= 1)
    {
        if (lane < s)
            sums[lane] += sums[lane + s];
        GroupMemoryBarrierWithGroupSync();
    }
}
