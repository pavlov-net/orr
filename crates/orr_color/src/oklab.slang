// Oklab from linear Rec.709 RGB with D65 white, and back.
// Matrices from Björn Ottosson, public domain.
// https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
module oklab;

static const float3x3 REC709_TO_LMS = float3x3(
    0.4122214708, 0.5363325363, 0.0514459929,
    0.2119034982, 0.6806995451, 0.1073969566,
    0.0883024619, 0.2817188376, 0.6299787005);
static const float3x3 LMS_TO_OKLAB = float3x3(
    0.2104542553, 0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050, 0.4505937099,
    0.0259040371, 0.7827717662, -0.8086757660);
static const float3x3 OKLAB_TO_LMS = float3x3(
    1.0, 0.3963377774, 0.2158037573,
    1.0, -0.1055613458, -0.0638541728,
    1.0, -0.0894841775, -1.2914855480);
static const float3x3 LMS_TO_REC709 = float3x3(
    4.0767416621, -3.3077115913, 0.2309699292,
    -1.2684380046, 2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147, 1.7076147010);

public float3 oklab_from_rec709(float3 rgb)
{
    float3 lms = mul(REC709_TO_LMS, rgb);
    return mul(LMS_TO_OKLAB, sign(lms) * pow(abs(lms), 1.0 / 3.0));
}

public float3 rec709_from_oklab(float3 lab)
{
    float3 lms = mul(OKLAB_TO_LMS, lab);
    return mul(LMS_TO_REC709, lms * lms * lms);
}
