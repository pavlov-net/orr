// The real-time OpenPBR 1.1.1 BSDF.
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
// read the tables from the heap slots `orr.openpbr` names (specialisation constant 104);
// defining `ORR_OPENPBR_ARRAY_LUTS` compiles them in instead (CPU builds).
//
// Differences from the reference: one shading normal for every layer and an isotropic coat;
// reflection-only GGX lobes sample bounded visible normals (Eto and Tokuyoshi 2023); a
// roughness below 0.01 is an ideal mirror and refraction; subsurface transmission is its own
// lobe (`LOBE_VOLUME`), which `Bsdf.subsurface_entry` samples alone for a random walk; lobes
// are chosen by their estimated albedo, glossy ones favoured by up to `SPECULAR_BIAS`; and no
// dispersion.
module bsdf;

import orr.math;
import orr.sampling;
import orr.openpbr;

// `BsdfParams.flags`.
public static const uint BSDF_THIN_WALLED = 1 << 0;
public static const uint BSDF_COAT = 1 << 1;
public static const uint BSDF_FUZZ = 1 << 2;
public static const uint BSDF_THIN_FILM = 1 << 3;
public static const uint BSDF_ANISOTROPY = 1 << 4;
// Set by `bsdf_collapse`.
public static const uint BSDF_COLLAPSED = 1 << 5;

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
// The largest factor by which lobe selection favours a smooth glossy lobe over its albedo
// share; it falls to 1 at roughness 1.
public static const float SPECULAR_BIAS = 2.0;

// OpenPBR inputs at a shading point, linear Rec.2020 and absolute units.
public struct BsdfParams
{
    public float3 base_color;           // base_weight x base_color
    public float metalness;
    public float3 specular_color;
    public float specular_weight;
    public float specular_roughness;
    public float specular_anisotropy;
    public float specular_ior;
    public float tangent_angle;         // rad: the anisotropy direction, from orthonormal_basis(n)'s tangent toward its bitangent
    public float diffuse_roughness;
    public float transmission_weight;
    public float3 transmission_color;   // tint applied at the surface: thin-walled or zero depth; 1 when the interior medium carries the colour
    public float subsurface_weight;
    public float3 subsurface_color;
    public float subsurface_anisotropy;
    public float3 subsurface_radius;    // m, mean free path per channel (radius x radius_scale)
    public float coat_weight;
    public float3 coat_color;
    public float coat_roughness;
    public float coat_ior;
    public float coat_darkening;
    public float fuzz_weight;
    public float3 fuzz_color;
    public float fuzz_roughness;
    public float thin_film_weight;
    public float thin_film_thickness;   // nm
    public float thin_film_ior;
    public float3 emission;             // any units; `Bsdf.emission` attenuates it by the coat and fuzz
    public float3 scale;                // multiplies every lobe: the directional occlusion, and the layers `bsdf_collapse` removes
    public uint flags;                  // BSDF_*: the lobes that are on and thin-walled geometry
}

public struct BsdfSample
{
    public float3 wi;
    public float3 weight;               // BSDF x cos / pdf, over every lobe
    public float pdf;                   // 0: no sample
    public uint lobe;                   // LOBE_*
}

// Directional albedo from the view, by kind, for guides and lobe selection.
public struct LobeAlbedo
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
static const float LARGEST_BELOW_ONE = 0.99999994;


// Prepared at one shading point for one view.
public struct Bsdf
{
    float3 t;
    float3 b;
    float3 n;                           // toward the view
    float3 v;                           // the view in (t, b, n)
    float3 scale;
    bool inside;
    float3 layer;                       // what the coat and fuzz pass of the view to the base

    float2 alpha;
    float alpha_iso;
    bool mirror;
    bool bounded;                       // reflection only: bounded visible-normal sampling
    float eta_refraction;
    float eta_transparent;
    float eta_opaque;
    float3 reflect_transparent;
    float3 reflect_opaque;
    float3 reflect_const;
    float metal;
    float3 f0;
    float3 f82;
    float film;
    float film_thickness;
    float film_exterior;
    float film_ior;
    float film_interior;
    float3 transmit;
    float3 walk;
    float walk_ms_share;
    float3 fold;                        // collapsed coat and fuzz albedo over the specular lobe's single-scattering albedo

    float3 dms_reflect;
    float3 dms_transmit;
    float dms_ratio;
    float dms_ec;
    float3 mms;
    float mms_ec;

    float3 diffuse;
    float3 thin_diffuse;
    float diffuse_roughness;
    float diffuse_comp;
    float3 thin_specular;

    float coat_presence;                // of the coat lobe, under the fuzz
    float3 coat_tint;
    float coat_tint_presence;
    float coat_eta;
    float coat_alpha;
    bool coat_mirror;

    float fuzz_presence;
    float3 fuzz_tint;
    float3 fuzz_ltc;                    // a_inv, b_inv, albedo at the view

    float3 emitted;
    float probability[LOBES];
    LobeAlbedo estimate;

    // Directional albedo from the view, by kind.
    public LobeAlbedo albedo()
    {
        return estimate;
    }

    // `BsdfParams.emission` as seen from the view: through the coat and fuzz, none from inside.
    public float3 emission()
    {
        return emitted;
    }

    // BSDF x cos toward `wi`, over every non-delta lobe.
    public float3 eval(float3 wi)
    {
        float density;
        float3 diffuse;
        return evaluate(local(wi), density, diffuse);
    }

    // `eval`, and in `diffuse` the share of the diffuse reflection and thin-walled diffuse
    // transmission lobes.
    public float3 eval(float3 wi, out float3 diffuse)
    {
        float density;
        return evaluate(local(wi), density, diffuse);
    }

    // The density `sample` draws `wi` with, over every non-delta lobe.
    public float pdf(float3 wi)
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
        // A delta direction: its coefficient and its probability within the lobe.
        bool delta = false;
        float3 w = float3(0.0);
        float within = 1.0;
        bool ok = true;
        switch (k)
        {
        case COAT:
            delta = coat_mirror;
            l = delta ? float3(-v.xy, v.z) : sample_ggx_vndf_bounded(v, float2(coat_alpha), uv);
            w = float3(coat_presence * openpbr_fresnel(coat_eta, v.z));
            lobe |= LOBE_REFLECTION;
            break;
        case FUZZ:
            l = sample_fuzz(uv);
            lobe |= LOBE_REFLECTION;
            break;
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
                w = rc * layer * leaving(l.z) + fold;
                within = pr;
            }
            else
            {
                ok = openpbr_refract(v, h, c, eta_refraction, l);
                lobe |= LOBE_TRANSMISSION;
                w = tc * layer * leaving(l.z);
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
        if (!ok || (up ? l.z <= 0.0 : l.z >= 0.0))
            return r;
        if (delta)
        {
            r.pdf = probability[k] * within;
            r.weight = w * scale / r.pdf;
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
        }
        r.wi = normalize(from_local(l, t, b, n));
        r.lobe = lobe;
        return r;
    }

    // One direction into the subsurface medium from the walk lobe alone: `weight` is its
    // BSDF x cos over its own density. `pdf` is 0 when the material has no walk lobe.
    public BsdfSample subsurface_entry(float3 u)
    {
        BsdfSample r = no_sample();
        if (all(walk == 0.0))
            return r;
        float3 l;
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
            if (mirror)
            {
                float3 rc, tc;
                coefficients(v.z, rc, tc);
                r.wi = normalize(from_local(l, t, b, n));
                r.pdf = 1.0 - walk_ms_share;
                r.weight = walk * tc * layer * scale / r.pdf;
                r.lobe = LOBE_TRANSMISSION | LOBE_SPECULAR | LOBE_VOLUME;
                return r;
            }
        }
        float pdf;
        float3 f = walk_lobe(l, pdf) * layer * leaving(l.z);
        if (!(pdf > 0.0))
            return r;
        r.wi = normalize(from_local(l, t, b, n));
        r.pdf = pdf;
        r.weight = f * scale / pdf;
        r.lobe = LOBE_TRANSMISSION | LOBE_GLOSSY | LOBE_VOLUME;
        return r;
    }

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
    float3 evaluate(float3 l, out float pdf, out float3 diffuse_share)
    {
        pdf = 0.0;
        diffuse_share = float3(0.0);
        if (l.z == 0.0)
            return float3(0.0);
        float3 diffuse_base = float3(0.0);
        float3 base = float3(0.0);
        float3 top = float3(0.0);
        float lobe_pdf;
        if (l.z > 0.0)
        {
            if (probability[COAT] > 0.0 && !coat_mirror)
            {
                top += coat_lobe(l, lobe_pdf);
                pdf += probability[COAT] * lobe_pdf;
            }
            if (probability[FUZZ] > 0.0)
            {
                top += fuzz_lobe(l, lobe_pdf);
                pdf += probability[FUZZ] * lobe_pdf;
            }
            if (probability[DIFFUSE] > 0.0)
            {
                diffuse_base += eon(diffuse, l) * l.z;
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
                base += walk_lobe(l, lobe_pdf);
                pdf += probability[WALK] * lobe_pdf;
            }
        }
        if (probability[SPECULAR] > 0.0 && !mirror)
        {
            float3 folded;
            base += specular_lobe(l, lobe_pdf, folded);
            top += folded;
            pdf += probability[SPECULAR] * lobe_pdf;
        }
        if (probability[DIELECTRIC_MS] > 0.0)
        {
            base += (l.z > 0.0 ? dms_reflect : dms_transmit) * dielectric_ms(l.z) * abs(l.z) * INV_PI;
            pdf += probability[DIELECTRIC_MS] * INV_TWO_PI * (l.z > 0.0 ? dms_ratio : 1.0 - dms_ratio);
        }
        float3 below = layer * leaving(l.z) * scale;
        diffuse_share = diffuse_base * below;
        return top * scale + (base + diffuse_base) * below;
    }

    // GGX reflection with a white coefficient, D G / (4 cos_v), and its bounded sampling density.
    float ggx_reflection(float3 l, out float pdf)
    {
        float3 h = normalize(v + l);
        pdf = ggx_vndf_bounded_pdf(v, l, alpha);
        return openpbr_eval_aniso_ggx(h, alpha) * openpbr_eval_aniso_smith_g1(v, alpha)
            * openpbr_eval_aniso_smith_g1(l, alpha) / (4.0 * v.z);
    }

    // The specular lobe below the layers, and the collapsed layers' share in `folded`.
    float3 specular_lobe(float3 l, out float pdf, out float3 folded)
    {
        pdf = 0.0;
        folded = float3(0.0);
        float g1v = openpbr_eval_aniso_smith_g1(v, alpha);
        float g = g1v * openpbr_eval_aniso_smith_g1(l, alpha);
        if (l.z > 0.0)
        {
            float3 h = normalize(v + l);
            float d = openpbr_eval_aniso_ggx(h, alpha);
            float3 rc, tc;
            coefficients(abs(dot(v, h)), rc, tc);
            if (bounded)
                pdf = ggx_vndf_bounded_pdf(v, l, alpha);
            else
                pdf = reflect_probability(rc, tc * transmit) * d * g1v / (4.0 * v.z);
            float shape = d * g / (4.0 * v.z);
            folded = fold * shape;
            return rc * shape;
        }
        if (all(transmit == 0.0))
            return float3(0.0);
        float jacobian;
        float c = refraction(l, jacobian);
        if (jacobian == 0.0)
            return float3(0.0);
        float d = openpbr_eval_aniso_ggx(refraction_half(l), alpha);
        float3 rc, tc;
        coefficients(c, rc, tc);
        tc *= transmit;
        pdf = (1.0 - reflect_probability(rc, tc)) * d * g1v * jacobian;
        return d * g * jacobian * tc;
    }

    // Subsurface transmission: GGX refraction plus its share of the dielectric multiple
    // scattering.
    float3 walk_lobe(float3 l, out float pdf)
    {
        pdf = walk_ms_share * INV_TWO_PI;
        float3 f = walk * dielectric_ms(l.z) * abs(l.z) * INV_PI;
        if (mirror)
            return f;
        float jacobian;
        float c = refraction(l, jacobian);
        if (jacobian == 0.0)
            return f;
        float d = openpbr_eval_aniso_ggx(refraction_half(l), alpha);
        float3 rc, tc;
        coefficients(c, rc, tc);
        float g1v = openpbr_eval_aniso_smith_g1(v, alpha);
        pdf += (1.0 - walk_ms_share) * d * g1v * jacobian;
        return f + d * g1v * openpbr_eval_aniso_smith_g1(l, alpha) * jacobian * walk * tc;
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

    float3 coat_lobe(float3 l, out float pdf)
    {
        float3 h = normalize(v + l);
        float2 a = float2(coat_alpha);
        float d = openpbr_eval_aniso_ggx(h, a);
        float g = openpbr_eval_aniso_smith_g1(v, a) * openpbr_eval_aniso_smith_g1(l, a);
        pdf = ggx_vndf_bounded_pdf(v, l, a);
        return float3(coat_presence * d * g * openpbr_fresnel(coat_eta, abs(dot(v, h))) / (4.0 * v.z));
    }

    float3 fuzz_lobe(float3 l, out float pdf)
    {
        pdf = ltc_eval(view_aligned(l, float2(v.x, -v.y)));
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
        return view_aligned(l, v.xy);
    }

    float3 eon(float3 rho, float3 l)
    {
        return openpbr_f_EON(rho, diffuse_roughness, v, l, false) * diffuse_comp
            * openpbr_look_up_opaque_dielectric_energy_complement(eta_opaque, alpha_iso, l.z);
    }

    float metal_ms(float cos_l)
    {
        float f = mms_ec * openpbr_look_up_ideal_metal_energy_complement(alpha_iso, cos_l)
            / openpbr_clamp_average_energy_complement_above_zero(openpbr_look_up_ideal_metal_average_energy_complement(alpha_iso));
        return min(f, 1.0 / cos_l);
    }

    // adobe's dielectric multiple-scattering factor toward cosine `cos_l` (negative below),
    // the reflection ratio included.
    float dielectric_ms(float cos_l)
    {
        if (dms_ec == 0.0)
            return 0.0;
        float c = abs(cos_l);
        float eta = cos_l > 0.0 ? eta_refraction : 1.0 / eta_refraction;
        float f = dms_ec * openpbr_look_up_ideal_dielectric_energy_complement(eta, alpha_iso, c)
            / openpbr_clamp_average_energy_complement_above_zero(openpbr_look_up_ideal_dielectric_average_energy_complement(eta, alpha_iso));
        f *= cos_l > 0.0 ? dms_ratio : 1.0 - dms_ratio;
        return min(f, 1.0 / c);
    }

    // Reflection coefficient and transmission factor (before the transmission scale) at
    // microfacet cosine `c`, with the thin film.
    void coefficients(float c, out float3 reflection, out float3 transmission)
    {
        float ft = openpbr_fresnel(eta_transparent, c);
        float fo = openpbr_fresnel(eta_opaque, c);
        reflection = reflect_transparent * ft + reflect_opaque * fo;
        if (metal > 0.0)
            reflection += metal * openpbr_metal_schlick_with_f82_tint(f0, f82, c);
        transmission = float3(1.0 - ft);
        if (film > 0.0)
        {
            bool dielectric = any(reflect_transparent > 0.0) || any(reflect_opaque > 0.0);
            OpenPBR_ThinFilmResults r = openpbr_thin_film_and_base_reflectance(
                c, film_exterior, film_ior, float3(film_interior), false, f0, f82, dielectric,
                metal > 0.0, film_thickness, OpenPBR_BaseRgbWavelengths_nm);
            float3 film_reflection = (reflect_transparent + reflect_opaque) * r.reflectance_dielectric + metal * r.reflectance_metal;
            reflection = lerp(reflection, film_reflection, film);
            transmission = lerp(transmission, saturate(1.0 - r.reflectance_dielectric), film);
        }
        reflection += reflect_const;
    }

    // The coat's colour on light leaving the base toward local cosine `c`.
    float3 leaving(float c)
    {
        if (inside ? c >= 0.0 : c <= 0.0)
            return float3(1.0);
        return passage(abs(c));
    }

    // The coat's colour along a direction at cosine `c` > 0 outside the base.
    float3 passage(float c)
    {
        if (coat_tint_presence == 0.0 || all(coat_tint >= 1.0))
            return float3(1.0);
        float s2 = (1.0 - c * c) / (coat_eta * coat_eta);
        float refracted = sqrt(max(0.0, 1.0 - s2));
        float3 through = pow(sqrt(max(coat_tint, 0.0)), float3(1.0 / max(refracted, 1e-6)));
        return lerp(float3(1.0), through, coat_tint_presence);
    }
}

BsdfSample no_sample()
{
    BsdfSample r;
    r.wi = float3(0.0);
    r.weight = float3(0.0);
    r.pdf = 0.0;
    r.lobe = 0;
    return r;
}

float reflect_probability(float3 reflection, float3 transmission)
{
    float r = max3(reflection);
    float t = max3(transmission);
    return r + t > 0.0 ? r / (r + t) : 1.0;
}

// `l` mirrored through the surface plane.
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

// The sheen LTC (a_inv, b_inv, albedo) at roughness `alpha` and view cosine `mu`, filtered as
// the reference filters it.
float3 ltc_coeffs(float alpha, float mu)
{
    OpenPBR_FuzzLobe_CoatingLobe_AggregateLobe lobe = {};
    lobe.alpha = alpha;
    return openpbr_disney_sheen_fetch_coeffs(lobe, float3(0.0, 0.0, mu));
}

float max3(float3 v)
{
    return max(v.x, max(v.y, v.z));
}

float sq(float x)
{
    return x * x;
}

float fourth_root(float x)
{
    return sqrt(sqrt(x));
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
// complement and the coat's colour, and their directional albedo moves onto the base
// specular lobe, so the directional albedo from the view is kept.
public BsdfParams bsdf_collapse(BsdfParams p)
{
    p.flags = (p.flags | BSDF_COLLAPSED) & ~BSDF_THIN_FILM;
    return p;
}

// The interior medium of a subsurface material (adobe's mapping from colour and radius).
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

// Prepares `p` at shading normal `n` (unit) seen along `wo` (unit, toward the viewer), with
// `exterior_ior` the medium on the side `n` points to. A view within 1e-4 of the surface
// plane leans up to it.
public Bsdf bsdf_prepare(BsdfParams p, float3 n, float3 wo, float exterior_ior = 1.0)
{
    Bsdf s;
    bool thin = (p.flags & BSDF_THIN_WALLED) != 0;
    bool collapsed = (p.flags & BSDF_COLLAPSED) != 0;
    float coat_weight = (p.flags & BSDF_COAT) != 0 ? p.coat_weight : 0.0;
    float fuzz_weight = (p.flags & BSDF_FUZZ) != 0 ? p.fuzz_weight : 0.0;
    float film_weight = (p.flags & BSDF_THIN_FILM) != 0 ? p.thin_film_weight : 0.0;
    float anisotropy = (p.flags & BSDF_ANISOTROPY) != 0 ? p.specular_anisotropy : 0.0;

    bool back = dot(wo, n) < 0.0;
    s.inside = back && !thin;
    s.n = back ? -n : n;
    float3 t0, b0;
    orthonormal_basis(n, t0, b0);
    s.t = cos(p.tangent_angle) * t0 + sin(p.tangent_angle) * b0;
    s.b = cross(s.n, s.t);
    float3 v = to_local(wo, s.t, s.b, s.n);
    if (v.z < MIN_COS)
    {
        float r2 = dot(v.xy, v.xy);
        v = r2 > 0.0 ? float3(v.xy * sqrt((1.0 - MIN_COS * MIN_COS) / r2), MIN_COS) : float3(0.0, 0.0, 1.0);
    }
    s.v = v;
    float mu = v.z;
    s.scale = p.scale;

    // IORs: the specular weight moves the base IOR toward the medium above it.
    bool enter_coat = coat_weight > 0.0 && !s.inside;
    float coat_ior = enter_coat ? lerp(exterior_ior, p.coat_ior, coat_weight) : exterior_ior;
    float specular_ior = coat_ior * openpbr_apply_specular_weight_to_ior(p.specular_ior / coat_ior, p.specular_weight);
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
            coat_eta = p.coat_ior / exterior_ior;
        }
        eta_o = eta_t;
    }
    eta_r = eta_r >= 1.0 ? max(eta_r, 1.0001) : min(eta_r, 1.0 / 1.0001);
    s.eta_refraction = eta_r;
    s.eta_transparent = eta_t;
    s.eta_opaque = eta_o;

    // Roughness, widened by the layers above.
    float fuzz_adjust = (p.fuzz_color.x + p.fuzz_color.y + p.fuzz_color.z) / 3.0 * p.fuzz_roughness * 0.005;
    float coat_roughness = lerp(p.coat_roughness,
        fourth_root(min(1.0, sq(sq(p.coat_roughness)) + fuzz_adjust * sq(sq(p.fuzz_roughness)))), fuzz_weight);
    float coat_adjust = 1.0 - (coat_eta >= 1.0 ? 1.0 / coat_eta : coat_eta);
    float roughness = lerp(p.specular_roughness,
        fourth_root(min(1.0, sq(sq(p.specular_roughness)) + coat_adjust * sq(sq(coat_roughness)))), coat_weight);
    s.alpha_iso = sq(roughness);
    s.alpha = openpbr_compute_anisotropic_alpha(s.alpha_iso, anisotropy, false, MIN_ALPHA);
    s.mirror = max(s.alpha.x, s.alpha.y) < MIRROR_ALPHA;
    s.coat_alpha = max(sq(coat_roughness), MIN_ALPHA);
    s.coat_mirror = s.coat_alpha < MIRROR_ALPHA;
    s.coat_eta = coat_eta;

    // The base: metal over transmission over subsurface over opaque dielectric.
    float dielectric = 1.0 - p.metalness;
    float opaque = (1.0 - p.transmission_weight) * (1.0 - p.subsurface_weight) * dielectric;
    float trans = dielectric * p.transmission_weight;
    float sss = (1.0 - p.transmission_weight) * dielectric * p.subsurface_weight;
    float darkened_metal = p.metalness * p.specular_weight;
    float3 metal_average = openpbr_metal_average_fresnel_with_f82_tint(p.base_color, p.specular_color);

    // Coat darkening from internal reflections (OpenPBR 1.1.1).
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
    float coat_reflected = coat_presence > 0.0 ? coat_presence * dielectric_energy(coat_eta, s.coat_alpha, mu) : 0.0;
    float3 coat_in = s.passage(mu) * (1.0 - coat_reflected) * darkening;
    float fuzz_presence = s.inside ? 0.0 : fuzz_weight;
    s.fuzz_tint = p.fuzz_color;
    s.fuzz_ltc = ltc_coeffs(p.fuzz_roughness, mu);
    float fuzz_in = 1.0 - fuzz_presence * s.fuzz_ltc.z;
    s.layer = s.inside ? float3(1.0) : coat_in * fuzz_in;
    s.emitted = s.inside ? float3(0.0) : p.emission * coat_in * fuzz_in;

    // Specular reflection and transmission.
    s.reflect_transparent = s.inside ? float3(trans + sss) : p.specular_color * (thin ? sss : trans + sss);
    s.reflect_opaque = p.specular_color * opaque;
    float3 tinted = trans * view_tint(p.transmission_color, thin, mu, eta_r);
    bool walk_on = !s.inside && !thin && sss > 0.0;
    s.transmit = thin ? float3(0.0) : tinted + (walk_on ? 0.0 : sss);
    s.walk = float3(walk_on ? sss : 0.0);
    s.bounded = all(s.transmit == 0.0);
    s.metal = darkened_metal;
    s.f0 = p.base_color;
    s.f82 = p.specular_color;
    s.film = film_weight > 0.0 && !collapsed ? film_weight * openpbr_thin_film_presence_multiplier(p.thin_film_thickness) : 0.0;
    s.film_thickness = p.thin_film_thickness;
    s.film_exterior = s.inside ? specular_ior : coat_ior;
    s.film_interior = s.inside ? coat_ior : specular_ior;
    s.film_ior = p.thin_film_ior;

    // Thin walls: a window's reflection and transmission, and subsurface as diffuse.
    s.reflect_const = float3(0.0);
    s.thin_specular = float3(0.0);
    s.thin_diffuse = float3(0.0);
    float3 thin_reflect_diffuse = float3(0.0);
    if (thin)
    {
        s.reflect_const = p.specular_color * trans * openpbr_thin_wall_fresnel(eta_t, mu);
        s.thin_specular = tinted * (1.0 - openpbr_thin_wall_fresnel(eta_r, mu));
        float forward = (p.subsurface_anisotropy + 1.0) * 0.5;
        float3 albedo = sss * p.subsurface_color;
        s.thin_diffuse = albedo * forward;
        thin_reflect_diffuse = albedo * (1.0 - forward);
    }

    // Energy lost to multiple scattering between microfacets, returned as wide lobes.
    s.dms_ec = 0.0;
    s.dms_ratio = 0.5;
    s.mms_ec = 0.0;
    if (s.alpha_iso >= MIN_MS_ALPHA)
    {
        s.dms_ratio = openpbr_look_up_ideal_dielectric_reflection_ratio(eta_r, s.alpha_iso);
        s.dms_ec = openpbr_look_up_ideal_dielectric_energy_complement(eta_r, s.alpha_iso, mu);
        s.mms_ec = openpbr_look_up_ideal_metal_energy_complement(s.alpha_iso, mu);
    }
    s.dms_reflect = s.inside ? float3(trans + sss) : p.specular_color * (trans + sss);
    s.dms_transmit = tinted + (walk_on ? 0.0 : sss);
    s.mms = metal_average * metal_average * darkened_metal;

    // Diffuse under the specular layer.
    s.diffuse_roughness = p.diffuse_roughness;
    s.diffuse_comp = openpbr_look_up_opaque_dielectric_energy_complement(eta_o, s.alpha_iso, mu)
        / openpbr_clamp_average_energy_complement_above_zero(openpbr_look_up_opaque_dielectric_average_energy_complement(eta_o, s.alpha_iso));
    s.diffuse = p.base_color * opaque + thin_reflect_diffuse;

    // Directional albedo estimates: they choose lobes and answer `albedo`.
    float e_ss = 1.0 - openpbr_look_up_ideal_metal_energy_complement(s.alpha_iso, mu);
    float3 below = s.layer * s.leaving(mu);
    float3 rc;
    float3 tc;
    s.fold = float3(0.0);
    s.coefficients(mu, rc, tc);
    float3 coat_albedo = float3(coat_reflected * fuzz_in);
    float3 fuzz_albedo = fuzz_presence * s.fuzz_ltc.z * p.fuzz_color;
    s.coat_presence = coat_presence * fuzz_in;
    s.fuzz_presence = fuzz_presence;
    if (collapsed)
    {
        s.fold = (coat_albedo + fuzz_albedo) / max(e_ss, 0.05);
        coat_albedo = float3(0.0);
        fuzz_albedo = float3(0.0);
        s.coat_presence = 0.0;
        s.fuzz_presence = 0.0;
    }
    float3 walk_ss = s.walk * tc * e_ss;
    float3 walk_ms = s.walk * (1.0 - s.dms_ratio) * s.dms_ec;
    s.walk_ms_share = max3(walk_ms) > 0.0 ? max3(walk_ms) / (max3(walk_ss) + max3(walk_ms)) : 0.0;

    float3 e[LOBES];
    e[COAT] = coat_albedo;
    e[FUZZ] = fuzz_albedo;
    e[SPECULAR] = (rc + s.transmit * tc) * e_ss * below + s.fold * e_ss;
    e[WALK] = (walk_ss + walk_ms) * below;
    e[DIELECTRIC_MS] = (s.dms_reflect * s.dms_ratio + s.dms_transmit * (1.0 - s.dms_ratio)) * s.dms_ec * below;
    e[METAL_MS] = s.mms * s.mms_ec * below;
    e[DIFFUSE] = s.diffuse * s.diffuse_comp * below;
    e[THIN_SPECULAR] = s.thin_specular * e_ss * below;
    e[THIN_DIFFUSE] = s.thin_diffuse * s.diffuse_comp * below;

    s.estimate.diffuse = (e[DIFFUSE] + e[THIN_DIFFUSE]) * p.scale;
    s.estimate.specular = (e[COAT] + e[FUZZ] + (rc * below + s.fold) * e_ss
        + (s.dms_reflect * s.dms_ratio * s.dms_ec + s.mms * s.mms_ec) * below) * p.scale;
    s.estimate.transmission = ((s.transmit * tc * e_ss + s.dms_transmit * (1.0 - s.dms_ratio) * s.dms_ec) * below
        + e[THIN_SPECULAR]) * p.scale;
    s.estimate.subsurface = e[WALK] * p.scale;
    // Table interpolation can leave a complement a rounding error below zero.
    s.estimate.diffuse = max(s.estimate.diffuse, 0.0);
    s.estimate.specular = max(s.estimate.specular, 0.0);
    s.estimate.transmission = max(s.estimate.transmission, 0.0);
    s.estimate.subsurface = max(s.estimate.subsurface, 0.0);

    float bias = lerp(SPECULAR_BIAS, 1.0, roughness);
    float sum = 0.0;
    for (uint i = 0; i < LOBES; i++)
    {
        float w = max(max3(e[i]), 0.0);
        if (i == SPECULAR || i == THIN_SPECULAR)
            w *= bias;
        if (i == COAT)
            w *= lerp(SPECULAR_BIAS, 1.0, coat_roughness);
        s.probability[i] = w;
        sum += w;
    }
    // The multiple-scattering lobes keep a share where their estimate rounds to nothing.
    if (s.dms_ec > 0.0 && (any(s.dms_reflect > 0.0) || any(s.dms_transmit > 0.0)))
        s.probability[DIELECTRIC_MS] = max(s.probability[DIELECTRIC_MS], 1e-3 * sum);
    if (s.mms_ec > 0.0 && s.metal > 0.0)
        s.probability[METAL_MS] = max(s.probability[METAL_MS], 1e-3 * sum);
    sum = 0.0;
    for (uint i = 0; i < LOBES; i++)
        sum += s.probability[i];
    for (uint i = 0; i < LOBES; i++)
        s.probability[i] = sum > 0.0 ? s.probability[i] / sum : 0.0;
    return s;
}
