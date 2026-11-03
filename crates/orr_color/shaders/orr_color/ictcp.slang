// ICtCp (Rec. ITU-R BT.2100, PQ form) from linear Rec.2020 in nits, and back. Above 10000
// nits the forward direction follows the PQ curve on, so brighter colors keep their order;
// the inverse clamps the signal like pq_decode.
module ictcp;

import orr_color.transfer;

static const float3x3 REC2020_TO_LMS = float3x3(
    1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0,
    683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0,
    99.0 / 4096.0, 309.0 / 4096.0, 3688.0 / 4096.0);
static const float3x3 LMS_TO_REC2020 = float3x3(
    3.43660669, -2.50645212, 0.0698454243,
    -0.791329556, 1.98360045, -0.192270896,
    -0.0259498997, -0.0989137147, 1.12486361);
static const float3x3 LMS_TO_ICTCP = float3x3(
    2048.0 / 4096.0, 2048.0 / 4096.0, 0.0,
    6610.0 / 4096.0, -13613.0 / 4096.0, 7003.0 / 4096.0,
    17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0);
static const float3x3 ICTCP_TO_LMS = float3x3(
    1.0, 0.00860903704, 0.111029625,
    1.0, -0.00860903704, -0.111029625,
    1.0, 0.560031336, -0.320627175);

public float3 ictcp_from_rec2020(float3 nits)
{
    return mul(LMS_TO_ICTCP, pq_curve(max(mul(REC2020_TO_LMS, nits) / PQ_MAX_NITS, 0.0)));
}

public float3 rec2020_from_ictcp(float3 ictcp)
{
    return mul(LMS_TO_REC2020, pq_decode(mul(ICTCP_TO_LMS, ictcp)));
}
