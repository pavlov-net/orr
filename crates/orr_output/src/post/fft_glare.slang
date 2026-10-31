// FFT glare: the exposed radiance at 1/2 or 1/4 resolution convolved with the diaphragm's
// PSF by FFT in shared memory, its edge pixels repeated out to the kernel's radius. The
// kernel is the PSF's energy per pixel (`build/fft.rs`) within that radius, less the source
// pixel's own, which tonemap keeps at full resolution as `w0 * exposed`. The dispatches
// share one root:
//
// - `lens` (one thread): the F-number. When it moved more than REBUILD_STOPS since the kernel
//   was built, or the view's buffers are new, it writes the kernel dispatches' group counts,
//   else zero groups.
// - `kernel_rows` and `kernel_columns`: the kernel's spectrum, real since the kernel is even,
//   kept for the quadrant kx <= nx / 2, ky <= ny / 2.
// - `rows`: each image row's real-to-complex FFT.
// - `columns`: per column of the row spectra and channel, the FFT, its product with the
//   kernel's spectrum and the inverse FFT; it also writes w0.
// - `inverse_rows`: each row's complex-to-real inverse FFT into `target`.
//
// Row spectra hold nx / 2 + 1 complex values per row and channel, the rest being their
// conjugates. Past the image, rows and columns repeat its last one out to the kernel's
// radius and, wrapped around, its first one; the rest are zero.
#include "orr/root.h"
import orr.abi;
import orr.math;
import view;
import glare_lens;

static const uint THREADS = 256;
// The longest transform: rows of up to 4096 pixels take 2048-point complex FFTs, columns
// 4096-point ones.
static const uint MAX_FFT = 4096;
// Complex values per thread of a row's half-length transform.
static const uint PER_THREAD = 2048 / THREADS;
static const float REBUILD_STOPS = 1.0 / 128.0;
// The kernel table's layout, `build/fft.rs`'s constants: level-0 texels per axis of its
// quadrant, its levels, the texels a summed box spans at least, the F-number it holds and
// its texels per reference pixel.
static const uint TABLE = 256;
static const uint LEVELS = 9;
static const float BOX_TEXELS = 8.0;
static const float REFERENCE_F = 22.0;
static const float TEXELS_PER_PIXEL = 2.0;

// The diffraction spikes past the kernel's table, from `build/fft.rs`.
struct FftKernel
{
    // Per channel, each spike's energy per unit length past the table times the distance
    // squared, in level-0 texels.
    float4 kappa;
    // Unit spike directions, two per entry.
    float4 spikes[8];
    uint spike_count;
}

struct Root
{
    ViewConstants* view;
    // The table's levels, each its quadrant row by row; xyz per channel.
    float4* table;
    FftKernel* kernel;
    // The F-number the kernel was built at (as bits), then the group counts of
    // `kernel_rows` and `kernel_columns`.
    uint* state;
    // Per channel, the kernel's spectrum over (nx / 2 + 1) x (ny / 2 + 1).
    float* spectrum;
    // Per channel, row spectra of (nx / 2 + 1) complex values, `plane_rows` rows.
    float2* plane;
    // w0 per channel.
    float4* weights;
    // The radiance, at full resolution.
    TextureIndex source;
    // The glare at the convolution's resolution.
    StorageImageIndex target;
    // The image at the convolution's resolution.
    uint width;
    uint height;
    uint log2_nx;
    uint log2_ny;
    uint radius;
    uint plane_rows;
    uint first_level;
    // 1 on the view's first frame with these buffers.
    uint fresh;
    // Reference pitch over the view's pixel pitch.
    float pitch_ratio;
    // Stops the HDR rendition's F-number moves down: log2(peak / GT7_SDR_NITS), 0 on SDR.
    float hdr_shift;
    float strength;
    uint _pad;
}
ORR_ROOT(Root);

static groupshared float2 data[MAX_FFT];

float2 cmul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 conj(float2 a)
{
    return float2(a.x, -a.y);
}

// e^(sign 2 pi i k / n).
float2 twiddle(uint k, uint n, float sign)
{
    float a = sign * TWO_PI * (float(k) / float(n));
    return float2(cos(a), sin(a));
}

uint bit_reverse(uint i, uint log2n)
{
    return reversebits(i) >> (32 - log2n);
}

// In-place radix-2 FFT of `data` from bit-reversed order to natural order; `sign` -1 is the
// forward transform, 1 the inverse without the 1/n.
void fft_dit(uint tid, uint log2n, float sign)
{
    uint n = 1u << log2n;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 0; s < log2n; s++)
    {
        uint half = 1u << s;
        for (uint b = tid; b < n / 2; b += THREADS)
        {
            uint j = b & (half - 1);
            uint i0 = ((b >> s) << (s + 1)) + j;
            uint i1 = i0 + half;
            float2 a = data[i0];
            float2 t = cmul(twiddle(j, 2 * half, sign), data[i1]);
            data[i0] = a + t;
            data[i1] = a - t;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// In-place radix-2 FFT of `data` from natural order to bit-reversed order.
void fft_dif(uint tid, uint log2n, float sign)
{
    uint n = 1u << log2n;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = log2n; s > 0; s--)
    {
        uint half = 1u << (s - 1);
        for (uint b = tid; b < n / 2; b += THREADS)
        {
            uint j = b & (half - 1);
            uint i0 = ((b >> (s - 1)) << s) + j;
            uint i1 = i0 + half;
            float2 a = data[i0];
            float2 c = data[i1];
            data[i0] = a + c;
            data[i1] = cmul(a - c, twiddle(j, 2 * half, sign));
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

uint row_length()
{
    return (1u << (root.log2_nx - 1)) + 1;
}

float2* plane_row(uint c, uint row)
{
    return root.plane + (c * root.plane_rows + row) * row_length();
}

float* spectrum_at(uint c, uint kx, uint ky)
{
    uint rows = (1u << (root.log2_ny - 1)) + 1;
    return root.spectrum + (c * rows + ky) * row_length() + kx;
}

// Writes a real row's spectrum (bins 0 to nx / 2) to `out` from the FFT in `data` of its
// samples packed as even + i odd.
void split_real(uint tid, float2* out)
{
    uint m = 1u << (root.log2_nx - 1);
    uint nx = 2 * m;
    for (uint k = tid; k <= m; k += THREADS)
    {
        float2 zk = data[k & (m - 1)];
        float2 zm = conj(data[(m - k) & (m - 1)]);
        float2 e = 0.5 * (zk + zm);
        float2 d = 0.5 * (zk - zm);
        // -i d
        float2 o = float2(d.y, -d.x);
        out[k] = e + cmul(twiddle(k, nx, -1.0), o);
    }
}

// The exposed radiance averaged over the full-resolution pixels of pixel (x, y).
float3 image_pixel(uint x, uint y, float scale)
{
    Texture2D<float4> source = root.source.get<Texture2D<float4>>();
    uint f = 1u << root.first_level;
    float3 sum = 0.0;
    for (uint b = 0; b < f; b++)
        for (uint a = 0; a < f; a++)
            sum += glare_exposed(source.Load(int3(x * f + a, y * f + b, 0)).rgb, scale);
    return sum / float(f * f);
}

// The image index at padded position `i` along an axis of `n` holding `size` pixels; -1
// for zero.
int padded(uint i, uint size, uint n)
{
    if (i < size)
        return int(i);
    if (i < size + root.radius)
        return int(size) - 1;
    if (i >= n - root.radius)
        return 0;
    return -1;
}

// The kernel's energy per channel at `[dx, dy]` pixels from the source, for pixels `u`
// table texels across (`Kernel::energy` in `build/fft.rs`).
float3 kernel_energy(int dx, int dy, float u)
{
    FftKernel* k = root.kernel;
    float2 x = float2(float(dx) - 0.5, float(dx) + 0.5) * u;
    float2 y = float2(float(dy) - 0.5, float(dy) + 0.5) * u;
    float t = float(TABLE);
    float3 e = 0.0;
    float2 cx = float2(max(x.x, -t), min(x.y, t));
    float2 cy = float2(max(y.x, -t), min(y.y, t));
    if (cx.x < cx.y && cy.x < cy.y)
    {
        uint level = 0;
        while (level + 1 < LEVELS && BOX_TEXELS * float(1u << (level + 1)) <= u)
            level++;
        float g = float(1u << level);
        uint n = TABLE >> level;
        uint offset = 0;
        for (uint l = 0; l < level; l++)
            offset += (TABLE >> l) * (TABLE >> l);
        int ax0 = int(floor(cx.x / g));
        int ax1 = int(ceil(cx.y / g));
        int ay0 = int(floor(cy.x / g));
        int ay1 = int(ceil(cy.y / g));
        for (int b = ay0; b < ay1; b++)
        {
            float wy = (min(cy.y, float(b + 1) * g) - max(cy.x, float(b) * g)) / g;
            uint j = uint(b >= 0 ? b : -b - 1);
            for (int a = ax0; a < ax1; a++)
            {
                float wx = (min(cx.y, float(a + 1) * g) - max(cx.x, float(a) * g)) / g;
                uint i = uint(a >= 0 ? a : -a - 1);
                e += wx * wy * root.table[offset + j * n + i].xyz;
            }
        }
    }
    for (uint s = 0; s < k.spike_count; s++)
    {
        float4 pair = k.spikes[s / 2];
        float2 d = (s & 1) == 0 ? pair.xy : pair.zw;
        // The spike's chord through the box, past the table's edge.
        float near = t / max(abs(d.x), abs(d.y));
        float far = 1e30;
        bool hit = true;
        if (d.x == 0.0)
            hit = hit && x.x < 0.0 && x.y > 0.0;
        else
        {
            float a = x.x / d.x;
            float b = x.y / d.x;
            near = max(near, min(a, b));
            far = min(far, max(a, b));
        }
        if (d.y == 0.0)
            hit = hit && y.x < 0.0 && y.y > 0.0;
        else
        {
            float a = y.x / d.y;
            float b = y.y / d.y;
            near = max(near, min(a, b));
            far = min(far, max(a, b));
        }
        if (hit && far > near)
            e += k.kappa.xyz * (1.0 / near - 1.0 / far);
    }
    return e;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lens()
{
    float n = glare_f_number(root.view, root.hdr_shift, root.pitch_ratio);
    float built = asfloat(root.state[0]);
    bool rebuild = root.fresh != 0 || !(abs(log2(n / built)) <= REBUILD_STOPS);
    if (rebuild)
        root.state[0] = asuint(n);
    root.state[1] = rebuild ? 2 * root.radius + 1 : 0;
    root.state[2] = 1;
    root.state[3] = 1;
    root.state[4] = rebuild ? (1u << (root.log2_nx - 1)) + 1 : 0;
    root.state[5] = 3;
    root.state[6] = 1;
}

// One group per kernel row, dy = group - radius, stored at row `group`.
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void kernel_rows(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint log2m = root.log2_nx - 1;
    uint m = 1u << log2m;
    uint nx = 2 * m;
    int dy = int(group.x) - int(root.radius);
    int r = int(root.radius);
    float u = float(1u << root.first_level) * TEXELS_PER_PIXEL * REFERENCE_F
            / asfloat(root.state[0]);
    float3 even[PER_THREAD];
    float3 odd[PER_THREAD];
    [ForceUnroll]
    for (uint q = 0; q < PER_THREAD; q++)
    {
        uint i = tid + q * THREADS;
        even[q] = 0.0;
        odd[q] = 0.0;
        if (i >= m)
            continue;
        for (uint h = 0; h < 2; h++)
        {
            uint x = 2 * i + h;
            int dx = x < nx / 2 ? int(x) : int(x) - int(nx);
            float3 e = 0.0;
            if (dx * dx + dy * dy <= r * r && (dx != 0 || dy != 0))
                e = kernel_energy(dx, dy, u);
            if (h == 0)
                even[q] = e;
            else
                odd[q] = e;
        }
    }
    [ForceUnroll]
    for (uint c = 0; c < 3; c++)
    {
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint q = 0; q < PER_THREAD; q++)
        {
            uint i = tid + q * THREADS;
            if (i < m)
                data[bit_reverse(i, log2m)] = float2(even[q][c], odd[q][c]);
        }
        fft_dit(tid, log2m, -1.0);
        split_real(tid, plane_row(c, group.x));
    }
}

// One group per column kx of the row spectra (x) and channel (y).
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void kernel_columns(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint kx = group.x;
    uint c = group.y;
    uint ny = 1u << root.log2_ny;
    int r = int(root.radius);
    for (uint y = tid; y < ny; y += THREADS)
    {
        int dy = y < ny / 2 ? int(y) : int(y) - int(ny);
        float2 v = 0.0;
        if (abs(dy) <= r)
            v = plane_row(c, uint(dy + r))[kx];
        data[bit_reverse(y, root.log2_ny)] = v;
    }
    fft_dit(tid, root.log2_ny, -1.0);
    for (uint ky = tid; ky <= ny / 2; ky += THREADS)
        spectrum_at(c, kx, ky)[0] = data[ky].x;
}

// One group per image row.
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void rows(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint y = group.x;
    uint log2m = root.log2_nx - 1;
    uint m = 1u << log2m;
    uint nx = 2 * m;
    float scale = glare_exposure_scale(root.view);
    float3 even[PER_THREAD];
    float3 odd[PER_THREAD];
    [ForceUnroll]
    for (uint q = 0; q < PER_THREAD; q++)
    {
        uint i = tid + q * THREADS;
        even[q] = 0.0;
        odd[q] = 0.0;
        if (i >= m)
            continue;
        int a = padded(2 * i, root.width, nx);
        int b = padded(2 * i + 1, root.width, nx);
        if (a >= 0)
            even[q] = image_pixel(uint(a), y, scale);
        if (b >= 0)
            odd[q] = image_pixel(uint(b), y, scale);
    }
    [ForceUnroll]
    for (uint c = 0; c < 3; c++)
    {
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint q = 0; q < PER_THREAD; q++)
        {
            uint i = tid + q * THREADS;
            if (i < m)
                data[bit_reverse(i, log2m)] = float2(even[q][c], odd[q][c]);
        }
        fft_dit(tid, log2m, -1.0);
        split_real(tid, plane_row(c, y));
    }
}

// One group per column kx of the row spectra (x) and channel (y).
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void columns(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint kx = group.x;
    uint c = group.y;
    uint ny = 1u << root.log2_ny;
    if (kx == 0 && c == 0 && tid == 0)
    {
        float3 dc = float3(spectrum_at(0, 0, 0)[0], spectrum_at(1, 0, 0)[0], spectrum_at(2, 0, 0)[0]);
        root.weights[0] = float4(1.0 - root.strength * dc, 0.0);
    }
    for (uint y = tid; y < ny; y += THREADS)
    {
        int row = padded(y, root.height, ny);
        data[bit_reverse(y, root.log2_ny)] = row >= 0 ? plane_row(c, uint(row))[kx] : 0.0;
    }
    fft_dit(tid, root.log2_ny, -1.0);
    // The forward transforms are unnormalized; the inverse row transform of half length
    // leaves a factor nx / 2, the inverse column transform ny.
    float norm = root.strength / float((1u << (root.log2_nx - 1)) * ny);
    for (uint ky = tid; ky < ny; ky += THREADS)
        data[ky] *= norm * spectrum_at(c, kx, min(ky, ny - ky))[0];
    fft_dif(tid, root.log2_ny, 1.0);
    for (uint y = tid; y < root.height; y += THREADS)
        plane_row(c, y)[kx] = data[bit_reverse(y, root.log2_ny)];
}

// One group per image row.
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void inverse_rows(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint y = group.x;
    uint log2m = root.log2_nx - 1;
    uint m = 1u << log2m;
    uint nx = 2 * m;
    float3 even[PER_THREAD];
    float3 odd[PER_THREAD];
    [ForceUnroll]
    for (uint c = 0; c < 3; c++)
    {
        float2* row = plane_row(c, y);
        GroupMemoryBarrierWithGroupSync();
        for (uint k = tid; k < m; k += THREADS)
        {
            float2 xk = row[k];
            float2 xm = conj(row[m - k]);
            float2 e = 0.5 * (xk + xm);
            float2 o = 0.5 * cmul(xk - xm, twiddle(k, nx, 1.0));
            // e + i o
            data[k] = e + float2(-o.y, o.x);
        }
        fft_dif(tid, log2m, 1.0);
        [ForceUnroll]
        for (uint q = 0; q < PER_THREAD; q++)
        {
            uint i = tid + q * THREADS;
            float2 z = i < m ? data[bit_reverse(i, log2m)] : 0.0;
            even[q][c] = z.x;
            odd[q][c] = z.y;
        }
    }
    RWTexture2D<float4> target = root.target.get<RWTexture2D<float4>>();
    [ForceUnroll]
    for (uint q = 0; q < PER_THREAD; q++)
    {
        uint x = 2 * (tid + q * THREADS);
        if (x < root.width)
            target[uint2(x, y)] = float4(max(even[q], 0.0), 1.0);
        if (x + 1 < root.width)
            target[uint2(x + 1, y)] = float4(max(odd[q], 0.0), 1.0);
    }
}
