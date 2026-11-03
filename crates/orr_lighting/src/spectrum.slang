// Hero-wavelength sampling for dispersion. Colors stay linear Rec.2020: a color's value at a
// wavelength is its dot product with `spectrum_basis` there, and a wavelength's value returns
// to Rec.2020 through the CIE 1931 2-deg observer (Wyman, Sloan and Shirley's multi-lobe fit)
// and `SPECTRUM_TO_REC2020`, so that a color sent through sampled wavelengths and back keeps
// its value on average.
module spectrum;

import orr_color.chromaticity;
import orr_color.primaries;
import orr.openpbr;

public static const float WAVELENGTH_MIN = 360.0;   // nm
public static const float WAVELENGTH_MAX = 830.0;
// Wavelengths a dispersive path carries: the hero and its stratified companions.
public static const uint WAVELENGTHS = 4;

// Takes the sum of `spectrum_rec2020(l) * dot(rgb, spectrum_basis(l))` over the visible range
// back to `rgb`: the inverse of that integral's matrix, over each channel's integral.
public static const float3x3 SPECTRUM_TO_REC2020 = float3x3(
    1.151455483e-02, -3.223699765e-03, -2.811293073e-04,
    -3.931714443e-03, 1.422114913e-02, -5.150260166e-05,
    -3.080867227e-04, -8.290504610e-04, 1.145013428e-02);

// The Rec.2020 color matching functions at `nm`.
public float3 spectrum_rec2020(float nm)
{
    return mul(XYZ_TO_REC2020, cie_xyz(nm));
}

// The weights of a color's channels at `nm`: the positive parts of the Rec.2020 color matching
// functions there, summing to 1, so gray is flat.
public float3 spectrum_basis(float nm)
{
    float3 c = max(spectrum_rec2020(nm), 0.0);
    return c / max(c.x + c.y + c.z, 1e-12);
}

// The density `sample_wavelength` draws with, per nm, over the visible range (Radziszewski,
// Boryczko and Alda's fit to the luminance response, as pbrt uses it).
public float wavelength_pdf(float nm)
{
    if (nm < WAVELENGTH_MIN || nm > WAVELENGTH_MAX)
        return 0.0;
    float c = cosh(0.0072 * (nm - 538.0));
    return 0.0039398042 / (c * c);
}

public float sample_wavelength(float u)
{
    return 538.0 - 138.888889 * atanh(0.85691062 - 1.82750197 * u);
}

// A path's wavelengths, hero first, and their densities.
public struct Wavelengths
{
    public float4 nm;
    public float4 pdf;
}

// The hero wavelength from `u` and companions at `u` offset by quarters.
public Wavelengths sample_wavelengths(float u)
{
    Wavelengths w;
    for (uint i = 0; i < WAVELENGTHS; i++)
    {
        float ui = frac(u + float(i) / float(WAVELENGTHS));
        w.nm[i] = clamp(sample_wavelength(ui), WAVELENGTH_MIN, WAVELENGTH_MAX);
        w.pdf[i] = wavelength_pdf(w.nm[i]);
    }
    return w;
}

// The matrix that takes a path's contribution, a color, to Rec.2020 through wavelengths `w`,
// each wavelength's estimate scaled by `weight`.
public float3x3 spectral_transfer(Wavelengths w, float4 weight)
{
    float3x3 m = float3x3(0.0);
    for (uint i = 0; i < WAVELENGTHS; i++)
    {
        if (!(weight[i] != 0.0) || !(w.pdf[i] > 0.0))
            continue;
        float3 c = spectrum_rec2020(w.nm[i]) * (weight[i] / (float(WAVELENGTHS) * w.pdf[i]));
        float3 b = spectrum_basis(w.nm[i]);
        m += float3x3(c.x * b, c.y * b, c.z * b);
    }
    return mul(SPECTRUM_TO_REC2020, m);
}

// OpenPBR's IOR at `nm` for IOR `ior` at 587.6 nm and Abbe number `abbe` over `scale`: the
// Cauchy fit through both. `ior` unchanged where `scale` or `abbe` is not above 0.
public float dispersed_ior(float ior, float scale, float abbe, float nm)
{
    if (!(scale > 0.0) || !(abbe > 0.0))
        return ior;
    return openpbr_dispersion_adjusted_ior(ior, 20.0 * scale / abbe, nm);
}
