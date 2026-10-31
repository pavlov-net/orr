// C entry points over orr-lighting's `bsdf` (array LUTs) for the CPU harness. Directions are
// in the shading frame (normal +z) and pass as float arrays.
import bsdf;
import bsdf_inputs;
import openpbr_inputs;

float3 load3(float* p) { return float3(p[0], p[1], p[2]); }
void store3(float* p, float3 v) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }

void store_sample(BsdfSample s, float* wi, float* weight, float* pdf, uint* lobe)
{
    store3(wi, s.wi);
    store3(weight, s.weight);
    *pdf = s.pdf;
    *lobe = s.lobe;
}

export __extern_cpp uint lighting_bsdf_inputs_size() { return sizeof(Inputs); }

export __extern_cpp uint lighting_bsdf_size() { return sizeof(Bsdf); }

export __extern_cpp void lighting_bsdf_prepare(Inputs* m, float* view, uint collapse, Bsdf* out)
{
    *out = prepare(*m, load3(view), collapse != 0);
}

export __extern_cpp void lighting_bsdf_eval(Bsdf* p, float* wi, float* diffuse, float* specular)
{
    float3 d;
    float3 total = p.eval(load3(wi), d);
    store3(diffuse, d);
    store3(specular, total - d);
}

export __extern_cpp float lighting_bsdf_pdf(Bsdf* p, float* wi)
{
    return p.pdf(load3(wi));
}

export __extern_cpp void lighting_bsdf_sample(Bsdf* p, float* u, float* wi, float* weight, float* pdf, uint* lobe)
{
    store_sample(p.sample(load3(u)), wi, weight, pdf, lobe);
}

export __extern_cpp void lighting_bsdf_subsurface_entry(Bsdf* p, float* u, float* wi, float* weight, float* pdf, uint* lobe)
{
    store_sample(p.subsurface_entry(load3(u)), wi, weight, pdf, lobe);
}

// Diffuse, specular, transmission and subsurface, three floats each.
export __extern_cpp void lighting_bsdf_albedo(Bsdf* p, float* out)
{
    LobeAlbedo a = p.albedo();
    store3(out, a.diffuse);
    store3(out + 3, a.specular);
    store3(out + 6, a.transmission);
    store3(out + 9, a.subsurface);
}

export __extern_cpp void lighting_bsdf_emission(Bsdf* p, float* out)
{
    store3(out, p.emission());
}

export __extern_cpp void lighting_bsdf_volume(Inputs* m, float* extinction, float* albedo, float* anisotropy)
{
    SubsurfaceVolume v = bsdf_subsurface_volume(bsdf_params(*m));
    store3(extinction, v.extinction);
    store3(albedo, v.albedo);
    *anisotropy = v.anisotropy;
}

// The fitting harness: `prepare` split at `Bsdf.finish`, then eval and pdf with the
// coefficients `fit` and their forward derivatives along each coefficient.
static const uint FIT_COEFFICIENTS = 3;

BsdfFit load_fit(float* p)
{
    BsdfFit f = { p[0], p[1], p[2] };
    return f;
}

// Collapsed `prepare` up to `Bsdf.finish`.
Bsdf prepare_layers(Inputs m, float3 view, out BsdfLayers layers)
{
    BsdfParams p = bsdf_collapse(bsdf_params(m));
    return bsdf_prepare_layers(p, float3(0.0, 0.0, 1.0), view, m.exterior_ior, layers);
}

[ForwardDifferentiable]
float4 fitted(no_diff Bsdf layered, no_diff BsdfLayers layers, BsdfFit fit, no_diff float3 wi)
{
    Bsdf s = layered;
    s.finish(layers, fit);
    return float4(s.eval(wi), s.pdf(wi));
}

export __extern_cpp uint lighting_bsdf_fit_coefficients() { return FIT_COEFFICIENTS; }

export __extern_cpp uint lighting_bsdf_layers_size() { return sizeof(BsdfLayers); }

// The constants `bsdf_prepare` uses.
export __extern_cpp void lighting_bsdf_fit_constants(float* out)
{
    out[0] = FOLD_COAT;
    out[1] = FOLD_COAT_SLOPE;
    out[2] = FOLD_FUZZ;
}

export __extern_cpp void lighting_bsdf_fit_prepare(Inputs* m, float* view, Bsdf* out, BsdfLayers* layers)
{
    *out = prepare_layers(*m, load3(view), *layers);
}

// Eval (3 floats) and pdf toward `wi`, and in `d_eval` (3 per coefficient) and `d_pdf` their
// derivatives along each coefficient.
export __extern_cpp void lighting_bsdf_fit_eval(Bsdf* layered, BsdfLayers* layers, float* fit, float* wi, float* eval, float* pdf, float* d_eval, float* d_pdf)
{
    BsdfFit f = load_fit(fit);
    float3 w = load3(wi);
    float4 r = fitted(*layered, *layers, f, w);
    store3(eval, r.xyz);
    *pdf = r.w;
    for (uint k = 0; k < FIT_COEFFICIENTS; k++)
    {
        float unit[FIT_COEFFICIENTS] = { 0.0, 0.0, 0.0 };
        unit[k] = 1.0;
        DifferentialPair<float4> d = fwd_diff(fitted)(*layered, *layers, diffPair(f, load_fit(&unit[0])), w);
        store3(d_eval + 3 * k, d.d.xyz);
        d_pdf[k] = d.d.w;
    }
}

// Collapsed `prepare` with the coefficients `fit` in place of the `FOLD_*` constants.
export __extern_cpp void lighting_bsdf_prepare_fit(Inputs* m, float* view, float* fit, Bsdf* out)
{
    BsdfLayers layers;
    *out = prepare_layers(*m, load3(view), layers);
    out.finish(layers, load_fit(fit));
}
