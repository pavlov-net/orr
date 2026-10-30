// Polyphony's GT7 sample as the reference, built twice: as published (f32) and with every
// float and float function widened to double. peak_nits 0 selects its SDR calibration.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <math.h>

// libstdc++ before GCC 14 lacks std::powf and friends, which the sample calls.
namespace std {
using ::exp2f;
using ::expf;
using ::log2f;
using ::powf;
}

#define main gt7_sample_main
namespace f32 {
#include "gt7_tone_mapping.cpp"
}
#define float double
#define powf pow
#define expf exp
#define log2f log2
#define exp2f exp2
#define max(a, b) max<double>(a, b)
namespace f64 {
#include "gt7_tone_mapping.cpp"
}
#undef float
#undef max
#undef main

template <typename Tm, typename T>
static void run(float peak_nits, const float* in, float* out, unsigned count)
{
    Tm tm;
    if (peak_nits == 0.0f)
        tm.initializeAsSDR();
    else
        tm.initializeAsHDR(peak_nits);
    for (unsigned i = 0; i < count; ++i)
    {
        T x[3] = { in[3 * i], in[3 * i + 1], in[3 * i + 2] };
        T y[3];
        tm.applyToneMapping(x, y);
        for (int c = 0; c < 3; ++c)
            out[3 * i + c] = static_cast<float>(y[c]);
    }
}

extern "C" void gt7_reference_f32(float peak_nits, const float* in, float* out, unsigned count)
{
    run<f32::GT7ToneMapping, float>(peak_nits, in, out, count);
}

extern "C" void gt7_reference_f64(float peak_nits, const float* in, float* out, unsigned count)
{
    run<f64::GT7ToneMapping, double>(peak_nits, in, out, count);
}
