module oklab;

// Oklab conversion coefficients: CSS Color 4, W3C Software and Document License.
// https://www.w3.org/TR/css-color-4/#color-conversion-code
// Convert linear Rec.709 RGB to Oklab.
public float3 linear_rec709_to_oklab(float3 rgb)
{
    float3 lms = mul(float3x3(
        0.4122214708, 0.5363325363, 0.0514459929,
        0.2119034982, 0.6806995451, 0.1073969566,
        0.0883024619, 0.2817188376, 0.6299787005), rgb);
    return mul(float3x3(
        0.2104542553, 0.7936177850, -0.0040720468,
        1.9779984951, -2.4285922050, 0.4505937099,
        0.0259040371, 0.7827717662, -0.8086757660), sign(lms) * pow(abs(lms), 1.0 / 3.0));
}

public float3 oklab_to_linear_rec709(float3 lab)
{
    float3 lms = mul(float3x3(
        1.0, 0.3963377774, 0.2158037573,
        1.0, -0.1055613458, -0.0638541728,
        1.0, -0.0894841775, -1.2914855480), lab);
    return mul(float3x3(
        4.0767416621, -3.3077115913, 0.2309699292,
        -1.2684380046, 2.6097574011, -0.3413193965,
        -0.0041960863, -0.7034186147, 1.7076147010), lms * lms * lms);
}
