// Links Polyphony's GT7 sample (fetched into .tools/display/gt7) for the port check.
#define main gt7_sample_main
#include "gt7_tone_mapping.cpp"
#undef main
extern "C" void orr_gt7_ref_sdr(const float* in, float* out) {
    static GT7ToneMapping tm = [] { GT7ToneMapping t; t.initializeAsSDR(); return t; }();
    tm.applyToneMapping(in, out);
}
