// Skeletons and clips as pose kernels read them in place: the layouts of orr_pack::skeleton
// and orr_pack::clip, the clip sampler `Clip::sample` is the reference for, and the joint
// transforms a skinning palette is built from. Every function takes blob addresses; the
// module holds no bindings. Records are read as 4-byte words, as the blobs align them.
module skeleton;

import orr_shaders.math;

// `SkeletonHeader.parents` entry of a root joint.
public static const uint NO_PARENT = 0xffff;

// Scale and shear, then rotation, then translation, relative to the parent joint.
public struct Trs
{
    public float4 rotation;        // unit quaternion, xyzw
    public float3 translation;
    public float3 scale;
    public float3 shear;           // unit y along x, unit z along x, unit z along y
}

// Trs record `i` of a table of 52-byte records at `p`.
public Trs load_trs(uint8_t* p, uint i)
{
    float* f = (float*)(p + 52 * i);
    Trs t;
    t.rotation = float4(f[0], f[1], f[2], f[3]);
    t.translation = float3(f[4], f[5], f[6]);
    t.scale = float3(f[7], f[8], f[9]);
    t.shear = float3(f[10], f[11], f[12]);
    return t;
}

// Row-major [R S | t], as `Trs::to_affine`.
public float3x4 trs_affine(Trs t)
{
    float x = t.rotation.x;
    float y = t.rotation.y;
    float z = t.rotation.z;
    float w = t.rotation.w;
    float3 s = t.scale;
    float3 h = t.shear;
    float3 ax = float3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
    float3 ay = float3(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
    float3 az = float3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
    float3 c0 = ax * s.x;
    float3 c1 = ax * h.x + ay * s.y;
    float3 c2 = ax * h.y + ay * h.z + az * s.z;
    return float3x4(
        float4(c0.x, c1.x, c2.x, t.translation.x),
        float4(c0.y, c1.y, c2.y, t.translation.y),
        float4(c0.z, c1.z, c2.z, t.translation.z));
}

// Spherical interpolation on the shorter arc, normalized lerp above a dot of 0.9995, as
// `orr_pack::skeleton::slerp`.
public float4 slerp(float4 a, float4 b, float u)
{
    float d = dot(a, b);
    float sign = d < 0.0 ? -1.0 : 1.0;
    d *= sign;
    float wa = 1.0 - u;
    float wb = u;
    if (d <= 0.9995)
    {
        float theta = acos(d);
        float s = sin(theta);
        wa = sin((1.0 - u) * theta) / s;
        wb = sin(u * theta) / s;
    }
    float4 q = wa * a + wb * sign * b;
    return q / length(q);
}

public Trs trs_interpolate(Trs a, Trs b, float u)
{
    Trs o;
    o.rotation = slerp(a.rotation, b.rotation, u);
    o.translation = a.translation + (b.translation - a.translation) * u;
    o.scale = a.scale + (b.scale - a.scale) * u;
    o.shear = a.shear + (b.shear - a.shear) * u;
    return o;
}

// A skeleton blob (`SkeletonHeader` and its tables).
public struct Skeleton
{
    public uint8_t* blob;
}

public uint skeleton_joints(Skeleton k)
{
    return *(uint*)k.blob;
}

// Byte offset of table `t` (0 parents, 1 rest, 2 inverse binds) in the blob.
uint table_at(Skeleton k, uint t)
{
    return ((uint*)k.blob)[2 + 2 * t];
}

// The parent of joint `j`, or NO_PARENT; parents come before their children.
public uint skeleton_parent(Skeleton k, uint j)
{
    uint word = ((uint*)(k.blob + table_at(k, 0)))[j >> 1];
    return (word >> (16 * (j & 1))) & 0xffff;
}

public Trs skeleton_rest(Skeleton k, uint j)
{
    return load_trs(k.blob + table_at(k, 1), j);
}

float3x4 load_affine(float* f)
{
    return float3x4(float4(f[0], f[1], f[2], f[3]), float4(f[4], f[5], f[6], f[7]), float4(f[8], f[9], f[10], f[11]));
}

// Model space to joint `j`'s bind space.
public float3x4 skeleton_inverse_bind(Skeleton k, uint j)
{
    return load_affine((float*)(k.blob + table_at(k, 2)) + 12 * j);
}

// Model space from the space of the root joints: `SkeletonHeader.root`, after its table spans.
public float3x4 skeleton_root(Skeleton k)
{
    return load_affine((float*)(k.blob + 48));
}

// The skinning matrix (palette entry) of joint `j` from its model-space transform:
// `model x inverse_bind`. A joint's model-space transform is its parent's times its local one
// (`affine_mul(parent, trs_affine(local))`), a root's parent being `skeleton_root`.
public float3x4 skeleton_palette(Skeleton k, uint j, float3x4 model)
{
    return affine_mul(model, skeleton_inverse_bind(k, j));
}

// A clip: its blob 0 (`ClipHeader`) and blob 1 (frame-major Trs samples).
public struct Clip
{
    public uint8_t* header;
    public uint8_t* samples;
}

// `ClipHeader` words: the skeleton's 16-byte AssetId, then these.
public uint clip_joints(Clip c)
{
    return ((uint*)c.header)[4];
}

public uint clip_frames(Clip c)
{
    return ((uint*)c.header)[5];
}

public float clip_rate(Clip c)
{
    return ((float*)c.header)[6];
}

public float clip_duration(Clip c)
{
    return ((float*)c.header)[7];
}

// Joint `j`'s local transform at frame `f`.
public Trs clip_frame(Clip c, uint f, uint j)
{
    return load_trs(c.samples, f * clip_joints(c) + j);
}

// Joint `j`'s local transform `t` seconds into the clip, clamped to it: the neighboring
// frames interpolated, as `Clip::sample`.
public Trs clip_sample(Clip c, float t, uint j)
{
    uint frames = clip_frames(c);
    if (frames <= 1)
        return clip_frame(c, 0, j);
    float x = clamp(t * clip_rate(c), 0.0, float(frames - 1));
    uint f = min(uint(x), frames - 2);
    return trs_interpolate(clip_frame(c, f, j), clip_frame(c, f + 1, j), x - float(f));
}
