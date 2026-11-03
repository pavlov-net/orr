// Local exposure's bilateral grid (exposure.slang has the layout): blurs the meter's sums with
// a binomial 1 4 6 4 1 over the bins, then across the cells and down them, leaving out taps
// past the grid's edges, and clears the sums for the next frame. `clear` zeroes a new grid's
// sums and pivot.
#include "orr_shaders/root.h"
import orr_shaders.abi;
import orr_output.exposure;

struct Root
{
    // Two words per bin, as the meter adds them.
    uint* sums;
    // The blurred grid, float2(EV sum in EV, count) per bin, which tonemap slices.
    float2* grid;
    // The grid between the blur's passes.
    float2* scratch;
    // Adapt's pivot, which `clear` zeroes.
    float2* pivot;
    // Cells across and down.
    uint width;
    uint height;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

// A multiple of LOCAL_BINS, so each group holds whole cells.
static const uint GROUP = 64;

static groupshared float2 bins[GROUP];

// The binomial weight `d` taps from the center, for |d| <= 2.
float binomial(int d)
{
    return d == 0 ? 6.0 : (abs(d) == 1 ? 4.0 : 1.0);
}

uint bins_total()
{
    return root.width * root.height * LOCAL_BINS;
}

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void clear(uint3 id: SV_DispatchThreadID)
{
    if (id.x < 2 * bins_total())
        root.sums[id.x] = 0;
    if (id.x == 0)
        *root.pivot = float2(0.0);
}

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void blur_bins(uint3 id: SV_DispatchThreadID, uint lane: SV_GroupIndex)
{
    bool inside = id.x < bins_total();
    float2 v = 0.0;
    if (inside)
    {
        v = float2(float(root.sums[2 * id.x + 1]) / LOCAL_EV_STEPS, float(root.sums[2 * id.x]));
        root.sums[2 * id.x] = 0;
        root.sums[2 * id.x + 1] = 0;
    }
    bins[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    if (!inside)
        return;
    int bin = int(lane % LOCAL_BINS);
    float2 sum = 0.0;
    for (int d = -2; d <= 2; d++)
    {
        int b = bin + d;
        if (b >= 0 && b < int(LOCAL_BINS))
            sum += binomial(d) * bins[int(lane) + d];
    }
    root.grid[id.x] = sum;
}

// Across the cells, `step` cells apart: 1 along a row, the grid's width down a column.
void blur_cells(uint i, uint along, uint length, uint step, float2* from, float2* to)
{
    float2 sum = 0.0;
    for (int d = -2; d <= 2; d++)
    {
        int at = int(along) + d;
        if (at >= 0 && at < int(length))
            sum += binomial(d) * from[int(i) + d * int(step * LOCAL_BINS)];
    }
    to[i] = sum;
}

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void blur_rows(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= bins_total())
        return;
    uint cell = id.x / LOCAL_BINS;
    blur_cells(id.x, cell % root.width, root.width, 1, root.grid, root.scratch);
}

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void blur_columns(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= bins_total())
        return;
    uint cell = id.x / LOCAL_BINS;
    blur_cells(id.x, cell / root.width, root.height, root.width, root.scratch, root.grid);
}
