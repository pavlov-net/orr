// RGB primaries: matrices between linear RGB spaces and CIE XYZ, `mul(M, rgb)`. They are
// orr_color::RgbPrimaries's f64 matrices rounded to f32. Every space here has D65 white.
module primaries;

public static const float2 D65_XY = float2(0.3127, 0.329);

// The primaries ids stored in cooked textures, orr_color::Primaries.
public static const uint PRIMARIES_REC709 = 0;
public static const uint PRIMARIES_DISPLAY_P3 = 1;
public static const uint PRIMARIES_REC2020 = 2;

public static const float3x3 REC709_TO_XYZ = float3x3(
    0.4123908, 0.35758433, 0.1804808,
    0.212639, 0.71516865, 0.07219232,
    0.019330818, 0.11919478, 0.95053214);
public static const float3x3 XYZ_TO_REC709 = float3x3(
    3.24097, -1.5373832, -0.49861076,
    -0.96924365, 1.8759675, 0.04155506,
    0.05563008, -0.20397696, 1.0569715);
public static const float3x3 P3_TO_XYZ = float3x3(
    0.48657095, 0.2656677, 0.19821729,
    0.22897457, 0.69173855, 0.07928691,
    0.0, 0.04511338, 1.0439444);
public static const float3x3 XYZ_TO_P3 = float3x3(
    2.493497, -0.9313836, -0.4027108,
    -0.829489, 1.7626641, 0.023624687,
    0.03584583, -0.07617239, 0.9568845);
public static const float3x3 REC2020_TO_XYZ = float3x3(
    0.63695806, 0.1446169, 0.16888097,
    0.2627002, 0.67799807, 0.059301715,
    0.0, 0.028072692, 1.0609851);
public static const float3x3 XYZ_TO_REC2020 = float3x3(
    1.7166512, -0.35567078, -0.2533663,
    -0.6666843, 1.6164812, 0.015768547,
    0.017639857, -0.042770613, 0.94210315);
public static const float3x3 REC709_TO_REC2020 = float3x3(
    0.6274039, 0.32928303, 0.043313067,
    0.06909729, 0.9195404, 0.011362315,
    0.01639144, 0.088013306, 0.89559525);
public static const float3x3 REC2020_TO_REC709 = float3x3(
    1.660491, -0.5876411, -0.07284986,
    -0.12455048, 1.1328999, -0.008349422,
    -0.018150764, -0.1005789, 1.1187297);
public static const float3x3 P3_TO_REC2020 = float3x3(
    0.75383306, 0.19859737, 0.047569595,
    0.04574385, 0.9417772, 0.012478931,
    -0.0012103403, 0.017601717, 0.9836086);
public static const float3x3 REC2020_TO_P3 = float3x3(
    1.3435782, -0.28217968, -0.06139858,
    -0.065297455, 1.0757879, -0.010490463,
    0.0028217873, -0.019598495, 1.0167767);
