// The real-time OpenPBR 1.1.2 BSDF.
//
// `bsdf_prepare` takes a shading point's `BsdfParams`, its shading normal and the direction
// toward the viewer, and returns a `Bsdf` that evaluates, samples and gives densities for
// light directions. Directions are unit vectors pointing away from the surface. Values are
// BSDF times the cosine of the light direction; densities are per unit solid angle, and for
// an ideal-specular (delta) lobe `BsdfSample.pdf` is the discrete probability of the lobe.
//
// The model follows adobe/openpbr-bsdf, the reference: EON diffuse, anisotropic GGX with F82
// metal and the specular weight applied as a change of IOR, rough transmission, thin-walled
// diffuse and specular transmission, thin-film iridescence, a coat with darkening and fuzz
// from the Zeltner sheen LTC, each layer scaling the ones below by its albedo complement seen
// from the view, and energy compensation from adobe's tables. Kernels that import this module
// read the tables from the heap slots `orr_lighting.openpbr` names (specialization constant 104);
// defining `ORR_OPENPBR_ARRAY_LUTS` compiles them in instead (CPU builds).
//
// Differences from the reference: reflection-only GGX lobes sample bounded visible normals (Eto and Tokuyoshi 2023); a
// roughness below 0.01 is an ideal mirror and refraction; subsurface transmission is its own
// lobe (`LOBE_VOLUME`), which `Bsdf.subsurface_entry` samples alone for a random walk; lobes
// are chosen by their estimated albedo, glossy ones favored by up to `SPECULAR_BIAS`; no
// dispersion; and refraction scales radiance, unless `bsdf_prepare` is told the path is adjoint,
// which is the reference's form.
//
// `bsdf_prepare` is `bsdf_prepare_layers` followed by `Bsdf.finish` with the fitted `FOLD_*`
// constants. `finish`, `eval` and `pdf` are forward-differentiable in `finish`'s `BsdfFit`;
// the functions marked `[TreatAsDifferentiable]` do not depend on it.
module bsdf;

import orr_shaders.math;
import orr_shaders.sampling;
import orr_lighting.openpbr;
import orr_lighting.specialize;
// `BsdfParams` and its flags.
__exported import orr_scene.gbuffer;

// `BsdfSample.lobe` flags (adobe's `OpenPBR_BsdfLobeType` values).
public static const uint LOBE_REFLECTION = 1 << 0;
public static const uint LOBE_TRANSMISSION = 1 << 1;
public static const uint LOBE_DIFFUSE = 1 << 2;
public static const uint LOBE_GLOSSY = 1 << 3;
// An ideal-specular (delta) direction.
public static const uint LOBE_SPECULAR = 1 << 4;
// Transmission into a subsurface medium.
public static const uint LOBE_VOLUME = 1 << 5;

// GGX roughness below which the specular, coat and thin-walled transmission lobes are delta.
public static const float MIRROR_ROUGHNESS = 0.01;
// The largest factor by which lobe selection favors a smooth glossy lobe over its albedo
// share; it falls to 1 at roughness 1.
public static const float SPECULAR_BIAS = 2.0;

// `bsdf_collapse` moves the coat's and fuzz's albedo onto the base: a share goes on the
// specular lobe's shape and the rest on a cosine lobe. The coat's share is `FOLD_COAT` plus
// `FOLD_COAT_SLOPE` times the coat roughness, the fuzz's `FOLD_FUZZ`, each at most 1. Fitted
// to the reference's eval over the harness grid (`bsdf-fit` in tests/lighting_math).
public static const float FOLD_COAT = 0.47870687;
public static const float FOLD_COAT_SLOPE = 2.7307014;
public static const float FOLD_FUZZ = 0.5356808;

// The coefficients `Bsdf.finish` takes, the `FOLD_*` constants in `bsdf_prepare`.
public struct BsdfFit : IDifferentiable
{
    public float fold_coat;
    public float fold_coat_slope;
    public float fold_fuzz;
}

public struct BsdfSample
{
    public float3 wi;
    public float3 weight;               // BSDF x cos / pdf, over every lobe
    public float3 diffuse;              // what of `weight` the diffuse lobes give, as `Bsdf.eval`'s `diffuse`
    public float pdf;                   // 0: no sample
    public uint lobe;                   // LOBE_*
    public uint component;              // the lobe drawn, for `Bsdf.delta_value`
}

// Directional albedo from the view, by kind, for guides and lobe selection.
public struct LobeAlbedo : IDifferentiable
{
    public float3 diffuse;              // diffuse reflection and thin-walled diffuse transmission
    public float3 specular;             // glossy and mirror reflection: specular, coat, fuzz
    public float3 transmission;         // glossy and ideal refraction, thin-walled specular transmission
    public float3 subsurface;           // into the subsurface medium
}

public struct SubsurfaceVolume
{
    public float3 extinction;           // 1/m
    public float3 albedo;               // single-scattering
    public float anisotropy;
}

static const uint COAT = 0;
static const uint FUZZ = 1;
static const uint SPECULAR = 2;
static const uint WALK = 3;
static const uint DIELECTRIC_MS = 4;
static const uint METAL_MS = 5;
static const uint DIFFUSE = 6;
static const uint THIN_SPECULAR = 7;
static const uint THIN_DIFFUSE = 8;
static const uint LOBES = 9;

static const float MIRROR_ALPHA = MIRROR_ROUGHNESS * MIRROR_ROUGHNESS;
// adobe's floors: the microfacet alpha, and the alpha below which no energy is visibly lost.
static const float MIN_ALPHA = 1e-6;
static const float MIN_MS_ALPHA = 0.04 * 0.04;
// The smallest view cosine; flatter views lean up to it.
static const float MIN_COS = 1e-4;
// The IOR the specular weight's limit on F0 leaves of an IOR of 0, which layers above divide by.
static const float MIN_IOR = 2.5e-5;
static const float LARGEST_BELOW_ONE = 0.99999994;
// The reference's largest F0, past which an IOR leaves what a float holds.
static const float MAX_F0 = 0.9999;

// What `bsdf_prepare_layers` leaves for `Bsdf.finish`: directional albedo estimates per lobe
// without the coat and fuzz folded, and the lobe-selection inputs.
public struct BsdfLayers
{
    bool collapsed;
    float e_ss;                         // the specular lobe's single-scattering albedo
    float coat_roughness;
    float3 e[LOBES];
    LobeAlbedo estimate;                // specular left out
    float3 specular_single;             // the specular lobe's reflection coefficient under the layers
    float3 specular_multiple;           // the multiple-scattering lobes' reflection
    float glossy_bias;
    float coat_bias;
    bool dms_floor;
    bool mms_floor;
}

// Prepared at one shading point for one view.
public struct Bsdf : IDifferentiable
{
    no_diff float3 t;
    no_diff float3 b;
    no_diff float3 n;                   // toward the view
    no_diff float3 v;                   // the view in (t, b, n)
    no_diff bool inside;
    no_diff float3 layer;               // what the coat and fuzz pass of the view to the base

    no_diff float2 alpha;
    no_diff float alpha_iso;
    no_diff bool mirror;
    no_diff bool bounded;               // reflection only: bounded visible-normal sampling
    no_diff float eta_refraction;
    no_diff float radiance_scale;       // of refraction: 1 / eta_refraction^2, 1 for an adjoint path or a thin wall
    no_diff float eta_transparent;
    no_diff float eta_opaque;
    no_diff float3 reflect_transparent;
    no_diff float3 reflect_opaque;
    no_diff float3 reflect_const;
    no_diff bool fresnel_mix;           // BSDF_FRESNEL_MIX, seen from outside
    no_diff float3 mix_f0;
    no_diff float mix_weight;           // 1 without `fresnel_mix`
    no_diff bool metal_flat;
    no_diff float metal;
    no_diff float3 f0;
    no_diff float3 f82;
    no_diff float film;
    no_diff float film_thickness;
    no_diff float film_exterior;
    no_diff float film_ior;
    no_diff float3 film_interior;
    no_diff float3 transmit;
    no_diff float3 walk;
    no_diff float walk_ms_share;
    float3 fold;                        // collapsed coat and fuzz albedo on the specular shape, over its single-scattering albedo
    float3 fold_diffuse;                // collapsed coat and fuzz albedo on a cosine lobe

    no_diff float3 dms_reflect;
    no_diff float3 dms_transmit;
    no_diff float dms_ratio;
    no_diff float dms_ec;
    no_diff float3 mms;
    no_diff float mms_ec;
    // Average energy complements of the multiple-scattering lobes, 1 where no evaluation needs
    // them, and the view's Smith term: none depends on the light direction.
    no_diff float mms_average;
    no_diff float dms_average_above;
    no_diff float dms_average_below;
    no_diff float g1v;

    no_diff float3 diffuse;
    no_diff float3 thin_diffuse;
    no_diff float diffuse_roughness;
    no_diff float diffuse_comp;
    no_diff float3 thin_specular;

    no_diff float coat_presence;        // of the coat lobe, under the fuzz
    no_diff float3 coat_tint;
    no_diff float coat_tint_presence;
    no_diff float coat_eta;
    no_diff float2 coat_alpha;
    no_diff bool coat_mirror;
    no_diff bool own_frames;            // the coat and fuzz do not shade in (t, b, n)
    no_diff float3x3 coat_basis;        // rows: its axes in (t, b, n)
    no_diff float3 coat_v;              // the view in the coat's frame
    no_diff float3x3 fuzz_basis;
    no_diff float3 fuzz_v;

    no_diff float fuzz_presence;
    no_diff float3 fuzz_tint;
    no_diff float3 fuzz_ltc;            // a_inv, b_inv, albedo at the view

    no_diff float3 emitted;
    float probability[LOBES];
    LobeAlbedo estimate;
    no_diff float3 thin_diffuse_albedo;  // the thin-walled diffuse transmission's share of `estimate.diffuse`

    // Completes `bsdf_prepare_layers` with `fit`: folds the coat and fuzz when collapsed, and
    // sets the albedo estimates and lobe probabilities.
    [mutating, ForwardDifferentiable]
    public void finish(no_diff BsdfLayers layers, BsdfFit fit)
    {
        float3 e[LOBES] = layers.e;
        LobeAlbedo a = layers.estimate;
        fold = float3(0.0);
        fold_diffuse = float3(0.0);
        if (layers.collapsed)
        {
            float3 glossy = e[COAT] * saturate(fit.fold_coat + fit.fold_coat_slope * layers.coat_roughness)
                + e[FUZZ] * saturate(fit.fold_fuzz);
            fold = glossy / max(layers.e_ss, 0.05);
            fold_diffuse = e[COAT] + e[FUZZ] - glossy;
            e[SPECULAR] += fold * layers.e_ss;
            e[DIFFUSE] += fold_diffuse;
            e[COAT] = float3(0.0);
            e[FUZZ] = float3(0.0);
        }
        a.specular = e[COAT] + e[FUZZ] + fold_diffuse + (layers.specular_single + fold) * layers.e_ss
            + layers.specular_multiple;
        // Table interpolation can leave a complement a rounding error below zero.
        estimate.diffuse = max(a.diffuse, 0.0);
        thin_diffuse_albedo = max(layers.e[THIN_DIFFUSE], 0.0);
        estimate.specular = max(a.specular, 0.0);
        estimate.transmission = max(a.transmission, 0.0);
        estimate.subsurface = max(a.subsurface, 0.0);

        float sum = 0.0;
        for (uint i = 0; i < LOBES; i++)
        {
            float w = max(max_component(e[i]), 0.0);
            if (i == SPECULAR || i == THIN_SPECULAR)
                w *= layers.glossy_bias;
            if (i == COAT)
                w *= layers.coat_bias;
            probability[i] = w;
            sum += w;
        }
        // The multiple-scattering lobes keep a share where their estimate rounds to nothing.
        if (layers.dms_floor)
            probability[DIELECTRIC_MS] = max(probability[DIELECTRIC_MS], 1e-3 * sum);
        if (layers.mms_floor)
            probability[METAL_MS] = max(probability[METAL_MS], 1e-3 * sum);
        sum = 0.0;
        for (uint i = 0; i < LOBES; i++)
            sum += probability[i];
        for (uint i = 0; i < LOBES; i++)
            probability[i] = sum > 0.0 ? probability[i] / sum : 0.0;
    }

    // Directional albedo from the view, by kind.
    public LobeAlbedo albedo()
    {
        return estimate;
    }

    // Directional albedo from the view of what `eval`'s `diffuse` holds: on the view's side,
    // diffuse reflection with the collapsed coat and fuzz's cosine lobe, and on the other, a
    // thin wall's diffuse transmission.
    public float3 diffuse_reflection()
    {
        return max(estimate.diffuse - thin_diffuse_albedo, 0.0) + fold_diffuse;
    }

    public float3 diffuse_transmission()
    {
        return thin_diffuse_albedo;
    }

    // `BsdfParams.emission` as seen from the view: through the coat and fuzz, none from inside.
    public float3 emission()
    {
        return emitted;
    }

    // What `emission` scales `BsdfParams.emission` by: the coat and fuzz, 0 from inside.
    public float3 emission_layer()
    {
        return inside ? float3(0.0) : layer;
    }

    // BSDF x cos toward `wi`, over every non-delta lobe.
    [ForwardDifferentiable]
    public float3 eval(no_diff float3 wi)
    {
        float density;
        float3 diffuse;
        return evaluate(local(wi), density, diffuse);
    }

    // `eval`, and in `diffuse` the share of the diffuse reflection and thin-walled diffuse
    // transmission lobes and of the collapsed coat and fuzz's cosine lobe.
    public float3 eval(float3 wi, out float3 diffuse)
    {
        float density;
        return evaluate(local(wi), density, diffuse);
    }

    // `eval`, and in `density` what `pdf` gives for `wi`.
    public float3 eval_and_pdf(float3 wi, out float density)
    {
        float3 diffuse;
        return evaluate(local(wi), density, diffuse);
    }

    // `eval_and_pdf`, and in `diffuse` the diffuse lobes' share `eval` gives.
    public float3 eval_and_pdf(float3 wi, out float density, out float3 diffuse)
    {
        return evaluate(local(wi), density, diffuse);
    }

    // Whether `eval` gives 0 toward every direction: each lobe with a share is delta.
    public bool specular_only()
    {
        bool glossy = (probability[SPECULAR] > 0.0 || probability[THIN_SPECULAR] > 0.0) && !mirror;
        bool coat = probability[COAT] > 0.0 && !coat_mirror;
        // At a mirror the walk lobe evaluates its multiple-scattering share alone.
        bool walk = probability[WALK] > 0.0 && !(mirror && dms_ec == 0.0);
        return !(glossy || coat || walk || probability[FUZZ] > 0.0 || probability[DIFFUSE] > 0.0 || probability[METAL_MS] > 0.0
            || probability[THIN_DIFFUSE] > 0.0 || probability[DIELECTRIC_MS] > 0.0);
    }

    // The density `sample` draws `wi` with, over every non-delta lobe.
    [ForwardDifferentiable]
    public float pdf(no_diff float3 wi)
    {
        float density;
        float3 diffuse;
        evaluate(local(wi), density, diffuse);
        return density;
    }

    // One direction from three uniform numbers in [0, 1).
    public BsdfSample sample(float3 u)
    {
        BsdfSample r = no_sample();
        float x = u.x;
        uint k = pick(x);
        if (k == LOBES)
            return r;
        float2 uv = float2(x, u.y);
        float3 l = float3(0.0);
        uint lobe = LOBE_GLOSSY;
        // How far `l` lies on the lobe's side of its own frame, where that isn't (t, b, n).
        float facing = 0.0;
        // A delta direction: its coefficient and its probability within the lobe.
        bool delta = false;
        float3 w = float3(0.0);
        float within = 1.0;
        bool ok = true;
        switch (k)
        {
        case COAT:
        {
            if (!may(MAY_COAT))
                return r;
            delta = coat_mirror;
            float3 c = delta ? float3(-coat_v.xy, coat_v.z) : sample_ggx_vndf_bounded(coat_v, coat_alpha, uv);
            facing = c.z;
            l = own_frames ? mul(c, coat_basis) : c;
            w = float3(coat_presence * openpbr_fresnel(coat_eta, coat_v.z));
            lobe |= LOBE_REFLECTION;
            break;
        }
        case FUZZ:
        {
            if (!may(MAY_FUZZ))
                return r;
            float3 f = sample_fuzz(uv);
            facing = f.z;
            l = own_frames ? mul(f, fuzz_basis) : f;
            lobe |= LOBE_REFLECTION;
            break;
        }
        case SPECULAR:
        {
            delta = mirror;
            if (!mirror && bounded)
            {
                l = sample_ggx_vndf_bounded(v, alpha, uv);
                lobe |= LOBE_REFLECTION;
                break;
            }
            float3 h = mirror ? float3(0.0, 0.0, 1.0) : openpbr_sample_aniso_ggx_smith_vndf(alpha, v, uv);
            float c = dot(v, h);
            float3 rc, tc;
            coefficients(abs(c), rc, tc);
            tc *= transmit;
            float pr = reflect_probability(rc, tc);
            if (c <= 0.0 || u.z < pr)
            {
                l = reflect(-v, h);
                lobe |= LOBE_REFLECTION;
                w = rc * layer * leaving(l) + fold;
                within = pr;
            }
            else
            {
                ok = openpbr_refract(v, h, c, eta_refraction, l);
                lobe |= LOBE_TRANSMISSION;
                w = tc * layer * leaving(l) * radiance_scale;
                within = 1.0 - pr;
            }
            break;
        }
        case WALK:
            lobe |= LOBE_TRANSMISSION | LOBE_VOLUME;
            if (u.z < walk_ms_share)
            {
                l = mirrored(sample_uniform_hemisphere(uv));
            }
            else
            {
                delta = mirror;
                float3 h = mirror ? float3(0.0, 0.0, 1.0) : openpbr_sample_aniso_ggx_smith_vndf(alpha, v, uv);
                ok = openpbr_refract(v, h, dot(v, h), eta_refraction, l);
                if (mirror)
                {
                    float3 rc, tc;
                    coefficients(v.z, rc, tc);
                    w = walk * tc * layer;
                    within = 1.0 - walk_ms_share;
                }
            }
            break;
        case DIELECTRIC_MS:
        {
            bool up = uv.x < dms_ratio;
            float z = up ? uv.x / dms_ratio : (uv.x - dms_ratio) / (1.0 - dms_ratio);
            z = min(z, LARGEST_BELOW_ONE);
            l = sample_uniform_hemisphere(float2(z, uv.y));
            if (!up)
                l = mirrored(l);
            lobe |= up ? LOBE_REFLECTION : LOBE_TRANSMISSION;
            break;
        }
        case METAL_MS:
            l = sample_uniform_hemisphere(uv);
            lobe |= LOBE_REFLECTION;
            break;
        case DIFFUSE:
            l = sample_cosine_hemisphere(uv);
            lobe = LOBE_REFLECTION | LOBE_DIFFUSE;
            break;
        case THIN_SPECULAR:
            delta = mirror;
            l = delta ? -v : mirrored(sample_ggx_vndf_bounded(v, alpha, uv));
            w = thin_specular * layer;
            lobe |= LOBE_TRANSMISSION;
            break;
        default:
            l = mirrored(sample_cosine_hemisphere(uv));
            lobe = LOBE_TRANSMISSION | LOBE_DIFFUSE;
            break;
        }
        bool up = (lobe & LOBE_REFLECTION) != 0;
        bool layered = k == COAT || k == FUZZ;
        if (!ok || (layered ? facing <= 0.0 : up ? l.z <= 0.0 : l.z >= 0.0))
            return r;
        if (delta)
        {
            r.pdf = probability[k] * within;
            r.weight = w / r.pdf;
            lobe = (lobe & ~LOBE_GLOSSY) | LOBE_SPECULAR;
        }
        else
        {
            float3 diffuse;
            float3 f = evaluate(l, r.pdf, diffuse);
            if (!(r.pdf > 0.0))
            {
                r.pdf = 0.0;
                return r;
            }
            r.weight = f / r.pdf;
            r.diffuse = diffuse / r.pdf;
        }
        r.wi = normalize(from_local(l, t, b, n));
        r.lobe = lobe;
        r.component = k;
        return r;
    }

    // What a delta sample of `component` reaching `wi` has here: BSDF x cos over the
    // direction's measure, and in `chance` the probability `sample` draws it. Zero where this
    // BSDF's delta direction for that lobe is not `wi`, or the lobe is not delta.
    public float3 delta_value(uint component, float3 wi, out float chance)
    {
        chance = 0.0;
        float3 l = local(wi);
        float3 own = float3(0.0);
        float3 w = float3(0.0);
        float within = 1.0;
        switch (component)
        {
        case COAT:
            if (!may(MAY_COAT) || !coat_mirror)
                return float3(0.0);
            own = float3(-coat_v.xy, coat_v.z);
            if (own_frames)
                own = mul(own, coat_basis);
            w = float3(coat_presence * openpbr_fresnel(coat_eta, coat_v.z));
            break;
        case SPECULAR:
        {
            if (!mirror)
                return float3(0.0);
            float3 rc, tc;
            coefficients(v.z, rc, tc);
            tc *= transmit;
            float pr = reflect_probability(rc, tc);
            if (l.z > 0.0)
            {
                own = float3(-v.xy, v.z);
                w = rc * layer * leaving(own) + fold;
                within = pr;
            }
            else
            {
                if (!openpbr_refract(v, float3(0.0, 0.0, 1.0), v.z, eta_refraction, own))
                    return float3(0.0);
                w = tc * layer * leaving(own) * radiance_scale;
                within = 1.0 - pr;
            }
            break;
        }
        case THIN_SPECULAR:
            if (!mirror)
                return float3(0.0);
            own = -v;
            w = thin_specular * layer;
            break;
        default:
            return float3(0.0);
        }
        if (dot(normalize(own), l) < 1.0 - 1e-6)
            return float3(0.0);
        chance = probability[component] * within;
        return w;
    }

    // One direction into the subsurface medium from the walk lobe alone: `weight` is its
    // BSDF x cos over its own density. `pdf` is 0 when the material has no walk lobe.
    public BsdfSample subsurface_entry(float3 u)
    {
        BsdfSample r = no_sample();
        if (all(walk == 0.0))
            return r;
        float3 l;
        bool specular = false;
        if (u.x < walk_ms_share)
        {
            l = mirrored(sample_uniform_hemisphere(u.yz));
        }
        else
        {
            float x = min((u.x - walk_ms_share) / (1.0 - walk_ms_share), LARGEST_BELOW_ONE);
            float3 h = mirror ? float3(0.0, 0.0, 1.0) : openpbr_sample_aniso_ggx_smith_vndf(alpha, v, float2(x, u.y));
            if (!openpbr_refract(v, h, dot(v, h), eta_refraction, l) || l.z >= 0.0)
                return r;
            specular = mirror;
        }
        // The mirror's refraction and the walk lobe take their coefficients from one call.
        Facet m = (Facet)0;
        if (!mirror)
            m = facet(l);
        if (specular || m.valid)
            coefficients(specular ? v.z : m.c, m.rc, m.tc);
        if (specular)
        {
            r.wi = normalize(from_local(l, t, b, n));
            r.pdf = 1.0 - walk_ms_share;
            r.weight = walk * m.tc * layer / r.pdf;
            r.lobe = LOBE_TRANSMISSION | LOBE_SPECULAR | LOBE_VOLUME;
            return r;
        }
        float pdf;
        float3 f = walk_lobe(l, m, pdf) * layer * leaving(l);
        if (!(pdf > 0.0))
            return r;
        r.wi = normalize(from_local(l, t, b, n));
        r.pdf = pdf;
        r.weight = f / pdf;
        r.lobe = LOBE_TRANSMISSION | LOBE_GLOSSY | LOBE_VOLUME;
        return r;
    }

    [TreatAsDifferentiable]
    float3 local(float3 wi)
    {
        return to_local(wi, t, b, n);
    }

    // The lobe for `x`, which is remapped to [0, 1) within it; LOBES when every lobe is empty.
    uint pick(inout float x)
    {
        float sum = 0.0;
        uint last = LOBES;
        for (uint k = 0; k < LOBES; k++)
        {
            float p = probability[k];
            if (p > 0.0)
            {
                if (x < sum + p)
                {
                    x = min((x - sum) / p, LARGEST_BELOW_ONE);
                    return k;
                }
                last = k;
            }
            sum += p;
        }
        x = LARGEST_BELOW_ONE;
        return last;
    }

    // BSDF x cos, density and the diffuse lobes' share at local `l` over the non-delta lobes.
    [ForwardDifferentiable]
    float3 evaluate(no_diff float3 l, out float pdf, out float3 diffuse_share)
    {
        pdf = 0.0;
        diffuse_share = float3(0.0);
        if (l.z == 0.0)
            return float3(0.0);
        float3 diffuse_base = float3(0.0);
        float3 diffuse_top = float3(0.0);
        float3 base = float3(0.0);
        float3 top = float3(0.0);
        float lobe_pdf;
        // The specular and walk lobes share the microfacet toward `l` and its coefficients.
        bool specular = probability[SPECULAR] > 0.0 && !mirror;
        bool walks = !(l.z > 0.0) && probability[WALK] > 0.0 && !mirror;
        Facet micro = (Facet)0;
        if ((specular && (l.z > 0.0 || any(transmit != 0.0))) || walks)
            micro = facet_coefficients(l);
        if (may(MAY_COAT) && probability[COAT] > 0.0 && !coat_mirror)
        {
            float3 c = in_frame(l, coat_basis);
            if (c.z > 0.0)
            {
                top += coat_lobe(c, lobe_pdf);
                pdf += probability[COAT] * lobe_pdf;
            }
        }
        if (may(MAY_FUZZ) && probability[FUZZ] > 0.0)
        {
            float3 f = in_frame(l, fuzz_basis);
            if (f.z > 0.0)
            {
                top += fuzz_lobe(f, lobe_pdf);
                pdf += probability[FUZZ] * lobe_pdf;
            }
        }
        if (l.z > 0.0)
        {
            if (probability[DIFFUSE] > 0.0)
            {
                diffuse_base += eon(diffuse, l) * l.z;
                diffuse_top = fold_diffuse * l.z * INV_PI;
                top += diffuse_top;
                pdf += probability[DIFFUSE] * l.z * INV_PI;
            }
            if (probability[METAL_MS] > 0.0)
            {
                base += mms * metal_ms(l.z) * l.z * INV_PI;
                pdf += probability[METAL_MS] * INV_TWO_PI;
            }
        }
        else
        {
            float3 m = mirrored(l);
            if (probability[THIN_DIFFUSE] > 0.0)
            {
                diffuse_base += eon(thin_diffuse, m) * m.z;
                pdf += probability[THIN_DIFFUSE] * m.z * INV_PI;
            }
            if (probability[THIN_SPECULAR] > 0.0 && !mirror)
            {
                base += thin_specular * ggx_reflection(m, lobe_pdf);
                pdf += probability[THIN_SPECULAR] * lobe_pdf;
            }
            if (probability[WALK] > 0.0)
            {
                base += walk_lobe(l, micro, lobe_pdf);
                pdf += probability[WALK] * lobe_pdf;
            }
        }
        if (specular)
        {
            float shape;
            base += specular_lobe(l, micro, lobe_pdf, shape);
            top += fold * shape;
            pdf += probability[SPECULAR] * lobe_pdf;
        }
        if (probability[DIELECTRIC_MS] > 0.0)
        {
            base += (l.z > 0.0 ? dms_reflect : dms_transmit * radiance_scale) * dielectric_ms(l.z) * abs(l.z) * INV_PI;
            pdf += probability[DIELECTRIC_MS] * INV_TWO_PI * (l.z > 0.0 ? dms_ratio : 1.0 - dms_ratio);
        }
        float3 below = layer * leaving(l);
        diffuse_share = diffuse_base * below + diffuse_top;
        return top + (base + diffuse_base) * below;
    }

    // GGX reflection with a white coefficient, D G / (4 cos_v), and its bounded sampling density.
    [TreatAsDifferentiable]
    float ggx_reflection(float3 l, out float pdf)
    {
        float3 h = normalize(v + l);
        pdf = ggx_vndf_bounded_pdf(v, l, alpha);
        return openpbr_eval_aniso_ggx(h, alpha) * g1v
            * openpbr_eval_aniso_smith_g1(l, alpha) / (4.0 * v.z);
    }

    // The microfacet toward local `l`, without its coefficients: the half vector's for `l` above
    // the surface, else the one that refracts the view into `l`.
    [TreatAsDifferentiable]
    Facet facet(float3 l)
    {
        Facet f = (Facet)0;
        if (l.z > 0.0)
        {
            float3 h = normalize(v + l);
            f.d = openpbr_eval_aniso_ggx(h, alpha);
            f.c = abs(dot(v, h));
            f.valid = true;
            return f;
        }
        f.c = refraction(l, f.jacobian);
        f.valid = f.jacobian != 0.0;
        if (f.valid)
            f.d = openpbr_eval_aniso_ggx(refraction_half(l), alpha);
        return f;
    }

    // `facet`, with its coefficients where it is valid.
    [TreatAsDifferentiable]
    Facet facet_coefficients(float3 l)
    {
        Facet f = facet(l);
        if (f.valid)
            coefficients(f.c, f.rc, f.tc);
        return f;
    }

    // The specular lobe below the layers toward `l`, with its `facet` and coefficients, and in
    // `shape` the white reflection the collapsed layers' `fold` scales.
    [TreatAsDifferentiable]
    float3 specular_lobe(float3 l, Facet m, out float pdf, out float shape)
    {
        pdf = 0.0;
        shape = 0.0;
        float g = g1v * openpbr_eval_aniso_smith_g1(l, alpha);
        if (l.z > 0.0)
        {
            if (bounded)
                pdf = ggx_vndf_bounded_pdf(v, l, alpha);
            else
                pdf = reflect_probability(m.rc, m.tc * transmit) * m.d * g1v / (4.0 * v.z);
            shape = m.d * g / (4.0 * v.z);
            return m.rc * shape;
        }
        if (all(transmit == 0.0) || !m.valid)
            return float3(0.0);
        float3 tc = m.tc * transmit;
        pdf = (1.0 - reflect_probability(m.rc, tc)) * m.d * g1v * m.jacobian;
        return m.d * g * m.jacobian * tc * radiance_scale;
    }

    // Subsurface transmission toward `l` below the surface, with its `facet` and coefficients:
    // GGX refraction plus its share of the dielectric multiple scattering.
    [TreatAsDifferentiable]
    float3 walk_lobe(float3 l, Facet m, out float pdf)
    {
        pdf = walk_ms_share * INV_TWO_PI;
        float3 f = walk * dielectric_ms(l.z) * abs(l.z) * INV_PI;
        if (mirror || !m.valid)
            return f;
        pdf += (1.0 - walk_ms_share) * m.d * g1v * m.jacobian;
        return f + m.d * g1v * openpbr_eval_aniso_smith_g1(l, alpha) * m.jacobian * walk * m.tc;
    }

    float3 refraction_half(float3 l)
    {
        return -normalize(v + l * eta_refraction);
    }

    // |v.h| for the microfacet that refracts the view into `l`, and the Jacobian
    // eta^2 |v.h| |l.h| / ((v.h + eta l.h)^2 cos_v); 0 when no microfacet does.
    float refraction(float3 l, out float jacobian)
    {
        jacobian = 0.0;
        float3 h = refraction_half(l);
        if ((eta_refraction >= 1.0) != (h.z > 0.0))
            return 0.0;
        float c = dot(v, h);
        float lc = dot(l, h);
        if (c * lc >= 0.0)
            return 0.0;
        float denom = c + eta_refraction * lc;
        jacobian = eta_refraction * eta_refraction * abs(c) * abs(lc) / (denom * denom * v.z);
        return abs(c);
    }

    // `l` in a layer's frame.
    [TreatAsDifferentiable]
    float3 in_frame(float3 l, float3x3 basis)
    {
        return own_frames ? mul(basis, l) : l;
    }

    // `l` in the coat's frame.
    [TreatAsDifferentiable]
    float3 coat_lobe(float3 l, out float pdf)
    {
        float3 h = normalize(coat_v + l);
        float d = openpbr_eval_aniso_ggx(h, coat_alpha);
        float g = openpbr_eval_aniso_smith_g1(coat_v, coat_alpha) * openpbr_eval_aniso_smith_g1(l, coat_alpha);
        pdf = ggx_vndf_bounded_pdf(coat_v, l, coat_alpha);
        return float3(coat_presence * d * g * openpbr_fresnel(coat_eta, abs(dot(coat_v, h))) / (4.0 * coat_v.z));
    }

    // `l` in the fuzz's frame.
    [TreatAsDifferentiable]
    float3 fuzz_lobe(float3 l, out float pdf)
    {
        pdf = ltc_eval(view_aligned(l, float2(fuzz_v.x, -fuzz_v.y)));
        return fuzz_presence * fuzz_ltc.z * fuzz_tint * pdf;
    }

    float ltc_eval(float3 l)
    {
        float a_inv = fuzz_ltc.x;
        float3 o = float3(a_inv * l.x + fuzz_ltc.y * l.z, a_inv * l.y, l.z);
        float len2 = dot(o, o);
        if (len2 == 0.0 || a_inv == 0.0 || l.z <= 0.0)
            return 0.0;
        float k = a_inv / len2;
        return INV_PI * l.z * k * k;
    }

    float3 sample_fuzz(float2 uv)
    {
        float3 o = sample_cosine_hemisphere(uv);
        float a = 1.0 / fuzz_ltc.x;
        float3 l = normalize(float3(o.x * a - o.z * fuzz_ltc.y * a, o.y * a, o.z));
        return view_aligned(l, fuzz_v.xy);
    }

    [TreatAsDifferentiable]
    float3 eon(float3 rho, float3 l)
    {
        return openpbr_f_EON(rho, diffuse_roughness, v, l, false) * diffuse_comp
            * weighted_complement(openpbr_look_up_opaque_dielectric_energy_complement(eta_opaque, alpha_iso, l.z), mix_weight);
    }

    [TreatAsDifferentiable]
    float metal_ms(float cos_l)
    {
        float f = mms_ec * openpbr_look_up_ideal_metal_energy_complement(alpha_iso, cos_l) / mms_average;
        return min(f, 1.0 / cos_l);
    }

    // adobe's dielectric multiple-scattering factor toward cosine `cos_l` (negative below),
    // the reflection ratio included.
    [TreatAsDifferentiable]
    float dielectric_ms(float cos_l)
    {
        if (dms_ec == 0.0)
            return 0.0;
        float c = abs(cos_l);
        float eta = cos_l > 0.0 ? eta_refraction : 1.0 / eta_refraction;
        float f = dms_ec * openpbr_look_up_ideal_dielectric_energy_complement(eta, alpha_iso, c)
            / (cos_l > 0.0 ? dms_average_above : dms_average_below);
        f *= cos_l > 0.0 ? dms_ratio : 1.0 - dms_ratio;
        return min(f, 1.0 / c);
    }

    // Reflection coefficient and transmission factor (before the transmission scale) at
    // microfacet cosine `c`, with the thin film.
    void coefficients(float c, out float3 reflection, out float3 transmission)
    {
        float ft;
        if (fresnel_mix)
        {
            float3 r = mix_reflection(mix_f0, mix_weight, c);
            reflection = (reflect_transparent + reflect_opaque) * r;
            ft = max_component(r);
        }
        else
        {
            ft = openpbr_fresnel(eta_transparent, c);
            // The two differ only from inside.
            float fo = ft;
            if (eta_opaque != eta_transparent)
                fo = openpbr_fresnel(eta_opaque, c);
            reflection = reflect_transparent * ft + reflect_opaque * fo;
        }
        if (metal > 0.0)
            reflection += metal * (metal_flat ? f0 : openpbr_metal_schlick_with_f82_tint(f0, f82, c));
        transmission = float3(1.0 - ft);
        if (may(MAY_FILM) && film > 0.0)
        {
            bool dielectric = any(reflect_transparent > 0.0) || any(reflect_opaque > 0.0);
            OpenPBR_ThinFilmResults r = openpbr_thin_film_and_base_reflectance(
                c, film_exterior, film_ior, film_interior, false, f0, f82, dielectric,
                metal > 0.0, film_thickness, OpenPBR_BaseRgbWavelengths_nm);
            float3 dielectric_film = mix_weight * r.reflectance_dielectric;
            float3 film_reflection = (reflect_transparent + reflect_opaque) * dielectric_film + metal * r.reflectance_metal;
            reflection = lerp(reflection, film_reflection, film);
            // No inverse colors: the layer below keeps what the largest channel leaves.
            float3 film_passed = fresnel_mix ? float3(1.0 - max_component(dielectric_film)) : 1.0 - dielectric_film;
            transmission = lerp(transmission, saturate(film_passed), film);
        }
        reflection += reflect_const;
    }

    // The coat's color on light leaving the base toward local `l`.
    [TreatAsDifferentiable]
    float3 leaving(float3 l)
    {
        if (inside ? l.z >= 0.0 : l.z <= 0.0)
            return float3(1.0);
        return passage(abs(in_frame(l, coat_basis).z));
    }

    // The coat's color along a direction at cosine `c` > 0 outside the base.
    float3 passage(float c)
    {
        if (coat_tint_presence == 0.0 || all(coat_tint >= 1.0))
            return float3(1.0);
        return lerp(float3(1.0), coat_transmittance(coat_tint, coat_eta, c), coat_tint_presence);
    }
}

// The transmittance of one passage through a coat of relative IOR `eta`, along a direction at
// cosine `c` > 0 outside it. `color` is OpenPBR's `coat_color`, the square of this at `c` = 1
// for light in and back out.
float3 coat_transmittance(float3 color, float eta, float c)
{
    float s2 = (1.0 - c * c) / (eta * eta);
    float refracted = sqrt(max(0.0, 1.0 - s2));
    return pow(sqrt(max(color, 0.0)), float3(1.0 / max(refracted, 1e-6)));
}

// A specular microfacet toward a direction: GGX's density `d` there, its cosine `c` to the view,
// the refraction Jacobian below the surface, and the coefficients at `c`, which callers fill.
// Not `valid` where no microfacet refracts the view into the direction.
struct Facet
{
    float d;
    float c;
    float jacobian;
    bool valid;
    float3 rc;
    float3 tc;
}

BsdfSample no_sample()
{
    BsdfSample r;
    r.wi = float3(0.0);
    r.weight = float3(0.0);
    r.diffuse = float3(0.0);
    r.pdf = 0.0;
    r.lobe = 0;
    r.component = LOBES;
    return r;
}

float reflect_probability(float3 reflection, float3 transmission)
{
    float r = max_component(reflection);
    float t = max_component(transmission);
    return r + t > 0.0 ? r / (r + t) : 1.0;
}

// `l` mirrored through the surface plane.
[TreatAsDifferentiable]
float3 mirrored(float3 l)
{
    return float3(l.xy, -l.z);
}

// `l` rotated about +z by the azimuth of `xy`.
float3 view_aligned(float3 l, float2 xy)
{
    float r2 = dot(xy, xy);
    if (r2 == 0.0)
        return l;
    float k = rsqrt(r2);
    float c = xy.x * k;
    float s = xy.y * k;
    return float3(c * l.x - s * l.y, c * l.y + s * l.x, l.z);
}

// `v` leaned up to the smallest view cosine.
float3 leaned(float3 v)
{
    if (v.z >= MIN_COS)
        return v;
    float r2 = dot(v.xy, v.xy);
    return r2 > 0.0 ? float3(v.xy * sqrt((1.0 - MIN_COS * MIN_COS) / r2), MIN_COS) : float3(0.0, 0.0, 1.0);
}

// The sheen LTC (a_inv, b_inv, albedo) at roughness `alpha` and view cosine `mu`, filtered as
// the reference filters it.
float3 ltc_coeffs(float alpha, float mu)
{
    OpenPBR_FuzzLobe_CoatingLobe_AggregateLobe lobe = {};
    lobe.alpha = alpha;
    return openpbr_disney_sheen_fetch_coeffs(lobe, float3(0.0, 0.0, mu));
}

float sq(float x)
{
    return x * x;
}

float fourth_root(float x)
{
    return sqrt(sqrt(x));
}

// What the layer below a specular lobe keeps of `complement`, the share a full weight leaves it,
// at BSDF_FRESNEL_MIX's `weight`.
float weighted_complement(float complement, float weight)
{
    return (1.0 - weight) + weight * complement;
}

// BSDF_FRESNEL_MIX's reflection at cosine `c`, from `f0` and `weight`.
float3 mix_reflection(float3 f0, float weight, float c)
{
    float x = 1.0 - c;
    float x2 = x * x;
    return weight * (f0 + (1.0 - f0) * (x2 * x2 * x));
}

// A thin wall's tint along the refracted path, from its tint at normal incidence.
float3 view_tint(float3 tint, bool thin, float c, float eta)
{
    if (!thin)
        return tint;
    float s2 = (1.0 - c * c) / (eta * eta);
    if (s2 >= 1.0)
        return float3(0.0);
    return pow(max(tint, 0.0), float3(1.0 / sqrt(1.0 - s2)));
}

// Energy a GGX dielectric interface at relative IOR `eta` reflects, seen at cosine `mu`.
float dielectric_energy(float eta, float alpha, float mu)
{
    if (alpha < MIRROR_ALPHA)
        return openpbr_fresnel(eta, mu);
    return 1.0 - openpbr_look_up_opaque_dielectric_energy_complement(eta, alpha, mu);
}

// `p` without coat, fuzz and thin film, for deeper path vertices: the base keeps their
// complement and the coat's color, and their directional albedo moves onto the base
// specular lobe and a cosine lobe (the `FOLD_*` shares), so the directional albedo from the
// view is kept.
public BsdfParams bsdf_collapse(BsdfParams p)
{
    p.flags = (p.flags | BSDF_COLLAPSED) & ~BSDF_THIN_FILM;
    return p;
}

// The interior medium of a subsurface material (adobe's mapping from color and radius).
public SubsurfaceVolume bsdf_subsurface_volume(BsdfParams p)
{
    OpenPBR_HomogeneousVolume m = openpbr_create_subsurface_volume_from_openpbr_params(
        1.0, p.subsurface_radius, p.subsurface_color, p.subsurface_anisotropy);
    SubsurfaceVolume r;
    r.extinction = m.extinction_coefficient;
    r.albedo = m.albedo;
    r.anisotropy = m.anisotropy;
    return r;
}

// One estimate of the share of what reaches a subsurface material's surface from inside
// that leaves toward `wi` (unit, on the side of shading normal `n`): the weight of the walk
// lobe, prepared from `wi`, for the direction `u` samples from it, over the subsurface
// weight. Its mean over `u` is the walk lobe's directional albedo from `wi`, so the
// interface and the layers above it as light from `wi` crosses them. 0 for materials
// without a walk lobe.
public float3 bsdf_walk_exit(BsdfParams p, float3 n, float3 wi, float3 u)
{
    Bsdf s = bsdf_prepare(p, n, wi);
    float sss = max_component(s.walk);
    if (!(sss > 0.0))
        return float3(0.0);
    BsdfSample e = s.subsurface_entry(u);
    return e.pdf > 0.0 ? e.weight / sss : float3(0.0);
}

// `p` with its subsurface share shaded as diffuse reflection in the subsurface color, the
// limit of a vanishing mean free path. Thin-walled materials are left as they are.
public BsdfParams bsdf_subsurface_diffuse(BsdfParams p)
{
    if ((p.flags & BSDF_THIN_WALLED) != 0)
        return p;
    p.base_color = lerp(p.base_color, p.subsurface_color, p.subsurface_weight);
    p.subsurface_weight = 0.0;
    return p;
}

// What `straight_transmission` reads of a surface's parameters and of a ray along `wi`
// (unit) through it, before the medium around it is known.
public struct StraightSurface
{
    public float3 color;            // the transmission color
    public float c;                 // `wi` . the unit geometric normal on the material's front side
    public float3 specular_color;
    public float specular_ior;
    public float3 coat_color;
    public float specular_weight;
    public float coat_weight;       // 0 without a coat
    public float coat_ior;
    public float share;             // (1 - metalness) x transmission weight
    public uint flags;              // BSDF_THIN_WALLED and BSDF_FRESNEL_MIX of the surface's
}

public StraightSurface straight_surface(BsdfParams p, float3 n, float3 wi)
{
    StraightSurface r;
    r.color = p.transmission_color;
    r.share = (1.0 - p.metalness) * p.transmission_weight;
    r.c = dot(wi, n);
    r.specular_color = p.specular_color;
    r.specular_ior = p.specular_ior;
    r.coat_color = p.coat_color;
    r.specular_weight = p.specular_weight;
    r.coat_weight = (p.flags & BSDF_COAT) != 0 ? p.coat_weight : 0.0;
    r.coat_ior = p.coat_ior;
    r.flags = p.flags & (BSDF_THIN_WALLED | BSDF_FRESNEL_MIX);
    return r;
}

// What a surface lets through straight along `wi`, without bending: its transmission share,
// surface tint and coat, times 1 - Fresnel at its interface against the medium of IOR
// `exterior_ior` around it. `wi` is the light's direction outside the surface, so Fresnel is
// the exterior side's from either side and light at any angle gets in. With `window`, for light
// from every direction outside, `wi` is its direction on the ray's own side: light reaches a
// point inside only within the critical angle, 0 beyond it, and what reflects there makes up
// the rest.
public float3 straight_transmission(StraightSurface s, float exterior_ior, bool window = false)
{
    bool thin = (s.flags & BSDF_THIN_WALLED) != 0;
    bool inside = window && s.c > 0.0 && !thin;
    float mu = max(abs(s.c), MIN_COS);
    bool mix = (s.flags & BSDF_FRESNEL_MIX) != 0;
    float weight = mix ? 1.0 : s.specular_weight;
    float ior = exterior_ior * openpbr_apply_specular_weight_to_ior(s.specular_ior / exterior_ior, weight);
    float eta = inside ? exterior_ior / ior : ior / exterior_ior;
    float reflected = mix && !inside ? max_component(mix_reflection(s.specular_color, s.specular_weight, mu)) : openpbr_fresnel(eta, mu);
    float3 t = s.share * (1.0 - reflected) * view_tint(s.color, thin && !mix, mu, eta);
    if (s.coat_weight > 0.0)
    {
        float coat_eta = max(s.coat_ior, MIN_IOR) / exterior_ior;
        t *= lerp(float3(1.0), coat_transmittance(s.coat_color, coat_eta, mu) * (1.0 - openpbr_fresnel(coat_eta, mu)), s.coat_weight);
    }
    return max(t, 0.0);
}

// `bsdf_straight_transmission` of `p` along `wi` with `n` the unit geometric normal on the
// material's front side.
public float3 bsdf_straight_transmission(BsdfParams p, float3 n, float3 wi, float exterior_ior = 1.0, bool window = false)
{
    return straight_transmission(straight_surface(p, n, wi), exterior_ior, window);
}

// The base specular lobe's roughness once the layers above widen it, and in `coat_roughness`
// the coat's once the fuzz widens it.
float widened_roughness(BsdfParams p, float coat_weight, float fuzz_weight, float coat_eta, bool mix, out float coat_roughness)
{
    float fuzz_adjust = (p.fuzz_color.x + p.fuzz_color.y + p.fuzz_color.z) / 3.0 * p.fuzz_roughness * 0.005;
    coat_roughness = lerp(p.coat_roughness,
        fourth_root(min(1.0, sq(sq(p.coat_roughness)) + fuzz_adjust * sq(sq(p.fuzz_roughness)))), fuzz_weight);
    float coat_adjust = 1.0 - (coat_eta >= 1.0 ? 1.0 / coat_eta : coat_eta);
    return lerp(p.specular_roughness,
        fourth_root(min(1.0, sq(sq(p.specular_roughness)) + coat_adjust * sq(sq(coat_roughness)))), mix ? 0.0 : coat_weight);
}

// Whether a thin wall of `p` in a medium of IOR `exterior_ior` is an ideal mirror and window:
// what `Bsdf.mirror` gives it, so a ray through it keeps its direction exactly.
public bool bsdf_thin_mirror(BsdfParams p, float exterior_ior)
{
    float coat_weight = (p.flags & BSDF_COAT) != 0 ? p.coat_weight : 0.0;
    float fuzz_weight = (p.flags & BSDF_FUZZ) != 0 ? p.fuzz_weight : 0.0;
    float anisotropy = (p.flags & BSDF_ANISOTROPY) != 0 ? p.specular_anisotropy : 0.0;
    bool mix = (p.flags & BSDF_FRESNEL_MIX) != 0;
    float coat_eta = coat_weight > 0.0 ? max(p.coat_ior, MIN_IOR) / exterior_ior : 1.0;
    float coat_roughness;
    float roughness = widened_roughness(p, coat_weight, fuzz_weight, coat_eta, mix, coat_roughness);
    float2 alpha = openpbr_compute_anisotropic_alpha(sq(roughness), anisotropy, false, MIN_ALPHA);
    return max(alpha.x, alpha.y) < MIRROR_ALPHA;
}

// A `Bsdf` for a path that has no vertex yet.
public Bsdf bsdf_unprepared()
{
    Bsdf s;
    s.emitted = float3(0.0);
    return s;
}

// Prepares `p` at shading normal `n` (unit) seen along `wo` (unit, toward the viewer), with
// `exterior_ior` the medium on the side `n` points to. A view within 1e-4 of the surface
// plane leans up to it. Refraction scales radiance by the squared ratio of the two IORs, so
// a path traced from the camera and leaving through another surface cancels the scale. A path
// carrying flux from a light is `adjoint`: its refraction keeps flux unscaled.
public Bsdf bsdf_prepare(BsdfParams p, float3 n, float3 wo, float exterior_ior = 1.0, bool adjoint = false)
{
    BsdfLayers layers;
    Bsdf s = bsdf_prepare_layers(p, n, wo, exterior_ior, adjoint, layers);
    BsdfFit fit = { FOLD_COAT, FOLD_COAT_SLOPE, FOLD_FUZZ };
    s.finish(layers, fit);
    return s;
}

// `bsdf_prepare` up to the fitted coefficients: `Bsdf.finish` with `layers` completes it.
public Bsdf bsdf_prepare_layers(BsdfParams p, float3 n, float3 wo, float exterior_ior, bool adjoint, out BsdfLayers layers)
{
    Bsdf s;
    bool thin = (p.flags & BSDF_THIN_WALLED) != 0;
    bool collapsed = (p.flags & BSDF_COLLAPSED) != 0;
    // A pipeline whose surfaces have none of a lobe (`may`) reads its weight as the 0 it is.
    float coat_weight = may(MAY_COAT) && (p.flags & BSDF_COAT) != 0 ? p.coat_weight : 0.0;
    float fuzz_weight = may(MAY_FUZZ) && (p.flags & BSDF_FUZZ) != 0 ? p.fuzz_weight : 0.0;
    float film_weight = may(MAY_FILM) && (p.flags & BSDF_THIN_FILM) != 0 ? p.thin_film_weight : 0.0;
    if (!may(MAY_SUBSURFACE))
        p.subsurface_weight = 0.0;
    float anisotropy = (p.flags & BSDF_ANISOTROPY) != 0 ? p.specular_anisotropy : 0.0;

    bool back = dot(wo, n) < 0.0;
    s.inside = back && !thin;
    s.n = back ? -n : n;
    float3 t0, b0;
    orthonormal_basis(n, t0, b0);
    s.t = cos(p.tangent_angle) * t0 + sin(p.tangent_angle) * b0;
    s.b = cross(s.n, s.t);
    float3 v = to_local(wo, s.t, s.b, s.n);
    v = leaned(v);
    s.v = v;
    float mu = v.z;
    s.coat_v = v;
    s.fuzz_v = v;
    s.coat_basis = float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    s.fuzz_basis = s.coat_basis;
    s.own_frames = coat_weight > 0.0 && (p.flags & BSDF_COAT_FRAME) != 0;
    if (s.own_frames)
    {
        float3 cn = dot(wo, p.coat_normal) < 0.0 ? -p.coat_normal : p.coat_normal;
        float3 ct0, cb0;
        orthonormal_basis(p.coat_normal, ct0, cb0);
        float3 ct = cos(p.coat_tangent_angle) * ct0 + sin(p.coat_tangent_angle) * cb0;
        float3 cb = cross(cn, ct);
        s.coat_basis = float3x3(to_local(ct, s.t, s.b, s.n), to_local(cb, s.t, s.b, s.n), to_local(cn, s.t, s.b, s.n));
        s.coat_v = leaned(mul(s.coat_basis, v));
        // The reference's fuzz lies between the base and the coat.
        float3 fn = safe_normalize(lerp(float3(0.0, 0.0, 1.0), s.coat_basis[2], coat_weight), float3(0.0, 0.0, 1.0));
        float3 ft, fb;
        orthonormal_basis(fn, ft, fb);
        s.fuzz_basis = float3x3(ft, fb, fn);
        s.fuzz_v = leaned(mul(s.fuzz_basis, v));
    }

    // IORs: the specular weight moves the base IOR toward the medium above it. A Fresnel mix's
    // base sees neither.
    bool mix = (p.flags & BSDF_FRESNEL_MIX) != 0;
    bool enter_coat = coat_weight > 0.0 && !s.inside;
    float coat_layer_ior = max(p.coat_ior, MIN_IOR);
    float coat_ior = enter_coat && !mix ? lerp(exterior_ior, coat_layer_ior, coat_weight) : exterior_ior;
    float specular_ior = coat_ior * openpbr_apply_specular_weight_to_ior(p.specular_ior / coat_ior, mix ? 1.0 : p.specular_weight);
    float eta_r;
    float eta_t;
    float eta_o;
    float coat_eta = 1.0;
    if (s.inside)
    {
        eta_r = exterior_ior / specular_ior;
        eta_t = eta_r;
        eta_o = specular_ior / exterior_ior;
    }
    else
    {
        eta_r = specular_ior / exterior_ior;
        eta_t = eta_r;
        if (enter_coat)
        {
            // No total internal reflection at the base under a coat that does not bend light.
            eta_t = coat_ior > specular_ior && specular_ior >= exterior_ior ? coat_ior / specular_ior : specular_ior / coat_ior;
            coat_eta = coat_layer_ior / exterior_ior;
        }
        eta_o = eta_t;
        if (mix)
            eta_o = openpbr_ior_from_f0(min(max_component(p.specular_color), MAX_F0));
    }
    s.fresnel_mix = mix && !s.inside;
    s.mix_f0 = p.specular_color;
    s.mix_weight = s.fresnel_mix ? p.specular_weight : 1.0;
    eta_r = eta_r >= 1.0 ? max(eta_r, 1.0001) : min(eta_r, 1.0 / 1.0001);
    s.eta_refraction = eta_r;
    s.radiance_scale = adjoint || thin ? 1.0 : 1.0 / (eta_r * eta_r);
    s.eta_transparent = eta_t;
    s.eta_opaque = eta_o;

    float coat_roughness;
    float roughness = widened_roughness(p, coat_weight, fuzz_weight, coat_eta, mix, coat_roughness);
    s.alpha_iso = sq(roughness);
    s.alpha = openpbr_compute_anisotropic_alpha(s.alpha_iso, anisotropy, false, MIN_ALPHA);
    s.mirror = max(s.alpha.x, s.alpha.y) < MIRROR_ALPHA;
    float coat_alpha = max(sq(coat_roughness), MIN_ALPHA);
    s.coat_alpha = openpbr_compute_anisotropic_alpha(coat_alpha, p.coat_anisotropy, false, MIN_ALPHA);
    s.coat_mirror = max(s.coat_alpha.x, s.coat_alpha.y) < MIRROR_ALPHA;
    s.coat_eta = coat_eta;

    // The base: metal over transmission over subsurface over opaque dielectric.
    float dielectric = 1.0 - p.metalness;
    float opaque = (1.0 - p.transmission_weight) * (1.0 - p.subsurface_weight) * dielectric;
    float trans = dielectric * p.transmission_weight;
    float sss = (1.0 - p.transmission_weight) * dielectric * p.subsurface_weight;
    // OpenPBR takes the metal's weight in 0..1; only the dielectric's goes past it.
    float darkened_metal = p.metalness * (mix ? 1.0 : saturate(p.specular_weight));
    float3 edge = mix ? float3(1.0) : p.specular_color;
    s.metal_flat = (p.flags & BSDF_METAL_FLAT) != 0;
    float3 metal_average = s.metal_flat ? p.base_color : openpbr_metal_average_fresnel_with_f82_tint(p.base_color, edge);

    // Coat darkening from internal reflections (OpenPBR 1.1.2).
    float k_smooth = openpbr_average_fresnel(coat_eta);
    float k_rough = 1.0 - (1.0 - k_smooth) / sq(coat_eta);
    float specularity = opaque * openpbr_average_fresnel(eta_o) + (sss + trans) * openpbr_average_fresnel(eta_t) + p.metalness;
    float k = lerp(k_smooth, k_rough, lerp(1.0, roughness, specularity));
    float3 base_albedo = darkened_metal * metal_average + opaque * lerp(p.base_color, float3(1.0), openpbr_average_fresnel(eta_o))
        + sss * p.subsurface_color + float3(trans);
    float3 darkening = lerp(float3(1.0), (1.0 - k) / (1.0 - base_albedo * k), coat_weight * p.coat_darkening);

    // The coat and fuzz seen from the view.
    s.coat_tint = p.coat_color;
    s.coat_tint_presence = coat_weight;
    float coat_presence = s.inside ? 0.0 : coat_weight;
    float coat_reflected = coat_presence > 0.0 ? coat_presence * dielectric_energy(coat_eta, coat_alpha, s.coat_v.z) : 0.0;
    float3 coat_in = s.passage(s.coat_v.z) * (1.0 - coat_reflected) * darkening;
    float fuzz_presence = s.inside ? 0.0 : fuzz_weight;
    s.fuzz_tint = p.fuzz_color;
    s.fuzz_ltc = float3(0.0);
    if (fuzz_presence != 0.0)
        s.fuzz_ltc = ltc_coeffs(p.fuzz_roughness, s.fuzz_v.z);
    float fuzz_in = 1.0 - fuzz_presence * s.fuzz_ltc.z;
    s.layer = s.inside ? float3(1.0) : coat_in * fuzz_in;
    s.emitted = s.inside ? float3(0.0) : p.emission * coat_in * fuzz_in;

    // Specular reflection and transmission.
    float3 specular_tint = mix ? float3(1.0) : p.specular_color;
    s.reflect_transparent = s.inside ? float3(trans + sss) : specular_tint * (thin && !mix ? sss : trans + sss);
    s.reflect_opaque = specular_tint * opaque;
    float3 tinted = trans * view_tint(p.transmission_color, thin && !mix, mu, eta_r);
    bool walk_on = !s.inside && !thin && sss > 0.0;
    s.transmit = thin ? float3(0.0) : tinted + (walk_on ? 0.0 : sss);
    s.walk = float3(walk_on ? sss : 0.0);
    s.bounded = all(s.transmit == 0.0);
    s.metal = darkened_metal;
    s.f0 = p.base_color;
    s.f82 = edge;
    s.film = film_weight > 0.0 && !collapsed ? film_weight * openpbr_thin_film_presence_multiplier(p.thin_film_thickness) : 0.0;
    s.film_thickness = p.thin_film_thickness;
    s.film_exterior = s.inside ? specular_ior : coat_ior;
    s.film_interior = float3(s.inside ? coat_ior : specular_ior);
    if (may(MAY_FILM) && s.fresnel_mix && s.film > 0.0)
    {
        float3 root = sqrt(min(p.specular_color, MAX_F0));
        s.film_interior = (1.0 + root) / (1.0 - root);
    }
    s.film_ior = p.thin_film_ior;

    // Thin walls: a window's reflection and transmission, and subsurface as diffuse.
    s.reflect_const = float3(0.0);
    s.thin_specular = float3(0.0);
    s.thin_diffuse = float3(0.0);
    float3 thin_reflect_diffuse = float3(0.0);
    if (thin)
    {
        if (mix)
        {
            s.thin_specular = tinted * (1.0 - max_component(mix_reflection(p.specular_color, p.specular_weight, mu)));
        }
        else
        {
            s.reflect_const = p.specular_color * trans * openpbr_thin_wall_fresnel(eta_t, mu);
            s.thin_specular = tinted * (1.0 - openpbr_thin_wall_fresnel(eta_r, mu));
        }
        float forward = (p.subsurface_anisotropy + 1.0) * 0.5;
        float3 albedo = sss * p.subsurface_color;
        s.thin_diffuse = albedo * forward;
        thin_reflect_diffuse = albedo * (1.0 - forward);
    }

    // Energy lost to multiple scattering between microfacets, returned as wide lobes.
    s.dms_ec = 0.0;
    s.dms_ratio = 0.5;
    s.mms_ec = 0.0;
    s.mms_average = 1.0;
    s.dms_average_above = 1.0;
    s.dms_average_below = 1.0;
    s.g1v = openpbr_eval_aniso_smith_g1(s.v, s.alpha);
    s.dms_reflect = s.inside ? float3(trans + sss) : specular_tint * s.mix_weight * (trans + sss);
    s.dms_transmit = tinted + (walk_on ? 0.0 : sss);
    float metal_ec = openpbr_look_up_ideal_metal_energy_complement(s.alpha_iso, mu);
    if (s.alpha_iso >= MIN_MS_ALPHA)
    {
        // Each of these multiplies the base's transmitting or walking share.
        if (walk_on || any(s.dms_reflect != 0.0) || any(s.dms_transmit != 0.0))
        {
            s.dms_ratio = openpbr_look_up_ideal_dielectric_reflection_ratio(eta_r, s.alpha_iso);
            s.dms_ec = openpbr_look_up_ideal_dielectric_energy_complement(eta_r, s.alpha_iso, mu);
            s.dms_average_above = openpbr_clamp_average_energy_complement_above_zero(openpbr_look_up_ideal_dielectric_average_energy_complement(eta_r, s.alpha_iso));
            if (walk_on || any(s.dms_transmit > 0.0))
                s.dms_average_below = openpbr_clamp_average_energy_complement_above_zero(openpbr_look_up_ideal_dielectric_average_energy_complement(1.0 / eta_r, s.alpha_iso));
        }
        s.mms_ec = metal_ec;
        if (s.metal > 0.0)
            s.mms_average = openpbr_clamp_average_energy_complement_above_zero(openpbr_look_up_ideal_metal_average_energy_complement(s.alpha_iso));
    }
    s.mms = metal_average * metal_average * darkened_metal;

    // Diffuse under the specular layer. The complement at the view multiplies the diffuse lobes
    // alone. Their average is read without them too: a Fresnel mix's weight can cancel it, and
    // the infinite ratio times 0 is not 0.
    s.diffuse_roughness = p.diffuse_roughness;
    s.diffuse = p.base_color * opaque + thin_reflect_diffuse;
    float diffuse_ec = 1.0;
    if (any(s.diffuse != 0.0) || any(s.thin_diffuse != 0.0))
        diffuse_ec = openpbr_look_up_opaque_dielectric_energy_complement(eta_o, s.alpha_iso, mu);
    s.diffuse_comp = weighted_complement(diffuse_ec, s.mix_weight)
        / weighted_complement(openpbr_clamp_average_energy_complement_above_zero(openpbr_look_up_opaque_dielectric_average_energy_complement(eta_o, s.alpha_iso)), s.mix_weight);

    // Directional albedo estimates without the coat and fuzz: `finish` adds them, folded or
    // not, and turns the estimates into lobe probabilities.
    float e_ss = 1.0 - metal_ec;
    float3 below = s.layer * s.leaving(v);
    float3 rc;
    float3 tc;
    s.coefficients(mu, rc, tc);
    s.coat_presence = collapsed ? 0.0 : coat_presence * fuzz_in;
    s.fuzz_presence = collapsed ? 0.0 : fuzz_presence;
    float3 walk_ss = s.walk * tc * e_ss;
    float3 walk_ms = s.walk * (1.0 - s.dms_ratio) * s.dms_ec;
    s.walk_ms_share = max_component(walk_ms) > 0.0 ? max_component(walk_ms) / (max_component(walk_ss) + max_component(walk_ms)) : 0.0;

    layers.collapsed = collapsed;
    layers.e_ss = e_ss;
    layers.coat_roughness = coat_roughness;
    layers.e[COAT] = float3(coat_reflected * fuzz_in);
    layers.e[FUZZ] = fuzz_presence * s.fuzz_ltc.z * p.fuzz_color;
    layers.e[SPECULAR] = (rc + s.transmit * tc) * e_ss * below;
    layers.e[WALK] = (walk_ss + walk_ms) * below;
    layers.e[DIELECTRIC_MS] = (s.dms_reflect * s.dms_ratio + s.dms_transmit * (1.0 - s.dms_ratio)) * s.dms_ec * below;
    layers.e[METAL_MS] = s.mms * s.mms_ec * below;
    layers.e[DIFFUSE] = s.diffuse * s.diffuse_comp * below;
    layers.e[THIN_SPECULAR] = s.thin_specular * e_ss * below;
    layers.e[THIN_DIFFUSE] = s.thin_diffuse * s.diffuse_comp * below;
    layers.estimate.diffuse = layers.e[DIFFUSE] + layers.e[THIN_DIFFUSE];
    layers.estimate.specular = float3(0.0);
    layers.specular_single = rc * below;
    layers.specular_multiple = (s.dms_reflect * s.dms_ratio * s.dms_ec + s.mms * s.mms_ec) * below;
    layers.estimate.transmission = (s.transmit * tc * e_ss + s.dms_transmit * (1.0 - s.dms_ratio) * s.dms_ec) * below
        + layers.e[THIN_SPECULAR];
    layers.estimate.subsurface = layers.e[WALK];
    layers.glossy_bias = lerp(SPECULAR_BIAS, 1.0, roughness);
    layers.coat_bias = lerp(SPECULAR_BIAS, 1.0, coat_roughness);
    layers.dms_floor = s.dms_ec > 0.0 && (any(s.dms_reflect > 0.0) || any(s.dms_transmit > 0.0));
    layers.mms_floor = s.mms_ec > 0.0 && s.metal > 0.0;
    return s;
}
