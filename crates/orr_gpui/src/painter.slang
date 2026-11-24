#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.math;
import orr_color.alpha;
import orr_color.luminance;
import orr_color.oklab;
import orr_color.transfer;

struct Fill
{
    float4 color0;
    float4 color1;
    float2 stops;
    float parameter;
    uint kind;
    uint color_space;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

struct Instance
{
    float4 bounds;
    float4 clip;
    // Fill or shadow bounds, or the sprite's full atlas allocation in UV space.
    float4 shape;
    float4 radii;
    float4 edges;
    float4 color;
    float4 uv;
    Fill fill;
    float2 transform_x;
    float2 transform_y;
    float2 offset;
    float parameter;
    uint flags;
    uint curve_start;
    uint curve_count;
    int backdrop;
    uint _pad0;
}

struct Curve
{
    float2 start;
    float2 end;
    float2 quadratic;
    float2 linear;
    float winding;
    uint _pad0;
}

struct Root
{
    Instance* instances;
    Curve* curves;
    float scale_x;
    float scale_y;
    uint first;
    uint kind;
    TextureIndex texture;
    SamplerIndex sampler;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

static const uint QUAD = 0;
static const uint SHADOW = 1;
static const uint UNDERLINE = 2;
static const uint PATH = 3;
static const uint MASK_SPRITE = 4;
static const uint COLOR_SPRITE = 5;

struct Vertex
{
    float4 position : SV_Position;
    float2 local : TEXCOORD0;
    nointerpolation uint index : TEXCOORD1;
}

[shader("vertex")]
Vertex vertex_main(uint corner : SV_VertexID, uint instance : SV_InstanceID)
{
    uint index = root.first + instance;
    Instance data = root.instances[index];
    float4 bounds = data.bounds;
    bool transformed = root.kind == MASK_SPRITE;
    if (!transformed)
        bounds = float4(max(bounds.xy, data.clip.xy), min(bounds.zw, data.clip.zw));

    float2 xy = float2((corner & 1) != 0 ? bounds.z : bounds.x, (corner & 2) != 0 ? bounds.w : bounds.y);
    if (any(bounds.zw <= bounds.xy))
        xy = bounds.xy;
    float2 window = xy;
    if (transformed)
        window = float2(dot(data.transform_x, xy), dot(data.transform_y, xy)) + data.offset;
    Vertex v;
    v.position = float4(window * float2(root.scale_x, root.scale_y) - 1.0, 0.0, 1.0);
    v.local = xy;
    v.index = index;
    return v;
}

float4 fill_color(Fill fill, float4 bounds, float2 p)
{
    if (fill.kind == 1)
    {
        float theta = radians(fill.parameter);
        float2 size = bounds.zw - bounds.xy;
        float2 direction = normalize(float2(sin(theta), -cos(theta)) * size);
        float extent = abs(direction.x) > abs(direction.y) ? size.x : size.y;
        float along = dot(p - (bounds.xy + bounds.zw) * 0.5, direction) / extent + 0.5;
        float span = fill.stops.y - fill.stops.x;
        float t = span == 0.0 ? step(fill.stops.x, along) : saturate((along - fill.stops.x) / span);
        float4 color = lerp(fill.color0, fill.color1, t);
        if (fill.color_space == 1)
        {
            float3 a = linear_rec709_to_oklab(srgb_decode(fill.color0.rgb));
            float3 b = linear_rec709_to_oklab(srgb_decode(fill.color1.rgb));
            color.rgb = srgb_encode(oklab_to_linear_rec709(lerp(a, b, t)));
        }
        return color;
    }
    if (fill.kind == 2)
    {
        float width = floor(fill.parameter / 65535.0) / 255.0;
        float gap = fmod(fill.parameter, 65535.0) / 255.0;
        if (width + gap == 0.0)
            return float4(0.0);
        float period = (width + gap) * sqrt(0.5);
        float phase = frac((p.x + p.y - bounds.x - bounds.y) * sqrt(0.5) / period) * period;
        float distance = min(phase, period - phase) - width * sqrt(0.5) * 0.5;
        return float4(fill.color0.rgb, fill.color0.a * saturate(0.5 - distance));
    }
    if (fill.kind == 3)
    {
        if (fill.parameter <= 0.0)
            return float4(0.0);
        float2 tile = floor((p - bounds.xy) / fill.parameter);
        return float4(fill.color0.rgb, fill.color0.a * saturate(fmod(tile.x + tile.y, 2.0)));
    }
    return fill.color0;
}

float signed_box(float2 p, float4 bounds)
{
    float2 d = max(bounds.xy - p, p - bounds.zw);
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float corner_radius(float2 p, float4 bounds, float4 radii)
{
    float2 center = (bounds.xy + bounds.zw) * 0.5;
    if (p.y < center.y)
        return p.x < center.x ? radii.x : radii.y;
    return p.x < center.x ? radii.w : radii.z;
}

float rounded_box(float2 p, float4 bounds, float4 radii)
{
    float radius = corner_radius(p, bounds, radii);
    float4 reduced = bounds + float4(radius, radius, -radius, -radius);
    return signed_box(p, reduced) - radius;
}

float inner_box(float2 p, Instance data)
{
    // Zero-width sides have no border antialiasing.
    float4 insets = select(data.edges == 0.0, float4(-0.5), data.edges);
    float4 inner = data.bounds + insets * float4(1.0, 1.0, -1.0, -1.0);
    if (any(inner.zw <= inner.xy))
        return 1.0;

    float2 center = (data.bounds.xy + data.bounds.zw) * 0.5;
    bool left = p.x < center.x;
    bool top = p.y < center.y;
    float radius = corner_radius(p, data.bounds, data.radii);
    float2 widths = float2(left ? insets.x : insets.z, top ? insets.y : insets.w);
    float2 axes = max(radius - widths, 0.0);
    float2 d = max(inner.xy - p, p - inner.zw);
    float2 from_corner = d + axes;
    if (any(from_corner <= 0.0) || any(axes <= 0.0))
        return signed_box(p, inner);

    // First-order signed distance to the ellipse formed by unequal border insets.
    float2 normalized = from_corner / axes;
    return (dot(normalized, normalized) - 1.0) / (2.0 * length(normalized / axes));
}

float dashed_edge(float2 p, Instance data)
{
    float2 size = data.bounds.zw - data.bounds.xy;
    float2 local = p - data.bounds.xy;
    float2 centered = local - 0.5 * size;
    float2 to_edge = abs(centered) - 0.5 * size;
    bool top = centered.y < 0.0;
    bool left = centered.x < 0.0;
    uint side = to_edge.y >= to_edge.x ? (top ? 0 : 2) : (left ? 3 : 1);

    // Side lengths and ending corners in clockwise order, starting at the top edge.
    float4 widths = data.edges.yzwx;
    float4 radii = data.radii.yzwx;
    float4 side_speed = select(widths > 0.0, 1.0 / (3.0 * widths), float4(0.0));
    float4 along = float4(local, size - local);
    const float dash = 2.0 / 3.0;
    float speed = side_speed[side];
    float position = along[side] * speed;
    float length = size[side & 1] * speed - dash;
    bool rounded = any(radii > 0.0);
    if (rounded)
    {
        float4 corner_widths = max(widths, widths.yzwx);
        float4 arc_speed = select(corner_widths > 0.0, 1.0 / (3.0 * corner_widths), float4(0.0));
        float4 straight = (size.xyxy - radii.wxyz - radii) * side_speed;
        float4 arc = (0.5 * PI) * radii * arc_speed;
        float4 segment = straight + arc;
        float4 start = float4(0.0, segment.x, segment.x + segment.y, segment.x + segment.y + segment.z);
        length = start.w + segment.w;
        position = start[side] + (along[side] - radii.wxyz[side]) * speed;

        uint corner = top ? (left ? 3 : 0) : (left ? 2 : 1);
        float radius = radii[corner];
        float2 from_center = to_edge + radius;
        if (radius > 0.0 && all(from_center >= 0.0) && any(from_center > 0.0))
        {
            float2 axes = (corner & 1) == 0 ? from_center : from_center.yx;
            speed = arc_speed[corner];
            position = start[corner] + straight[corner] + atan2(axes.x, axes.y) * radius * speed;
        }
    }
    if (speed == 0.0)
        return 0.0;
    if (length < (rounded ? 1.0 : dash))
        return 1.0;
    float period = length / max(floor(length), 1.0);
    float distance = abs(frac((position - dash * 0.5) / period + 0.5) - 0.5) * period - dash * 0.5;
    return saturate(0.5 - distance / speed);
}

float4 quad_color(float2 p, Instance data)
{
    float4 fill = fill_color(data.fill, data.shape, p);
    bool bordered = any(data.edges > 0.0);
    if (!bordered && !any(data.radii > 0.0) && data.fill.kind == 0)
        return premultiply_alpha(fill);
    float outer = saturate(0.5 - rounded_box(p, data.bounds, data.radii));
    if (!bordered)
        return premultiply_alpha(fill, outer);
    float border = saturate(0.5 + inner_box(p, data));
    if (data.flags != 0 && border > 0.0)
        border *= dashed_edge(p, data);

    // Take both colors at full coverage before masking the outer edge.
    return over_premultiplied(premultiply_alpha(data.color, border), premultiply_alpha(fill)) * outer;
}

float corner_inset(float depth, float radius)
{
    if (depth >= radius)
        return 0.0;
    return radius - sqrt(depth * (2.0 * radius - depth));
}

float shadow_coverage(float2 p, Instance data)
{
    float sigma = data.parameter;
    float4 box = data.shape;
    float alpha;
    if (sigma == 0.0)
        alpha = saturate(0.5 - rounded_box(p, box, data.radii));
    else
    {
        // Integrate the Gaussian over each horizontal interval of the rounded shape.
        float first = max(box.y, p.y - 4.0 * sigma);
        float last = min(box.w, p.y + 4.0 * sigma);
        float dy = max(last - first, 0.0) / 12.0;
        alpha = 0.0;
        for (uint i = 0; i < 12 && dy > 0.0; i++)
        {
            float y = first + (float(i) + 0.5) * dy;
            float top = y - box.y;
            float bottom = box.w - y;
            float lo = box.x + max(corner_inset(top, data.radii.x), corner_inset(bottom, data.radii.w));
            float hi = box.z - max(corner_inset(top, data.radii.y), corner_inset(bottom, data.radii.z));
            float horizontal = standard_normal_cdf((hi - p.x) / sigma) - standard_normal_cdf((lo - p.x) / sigma);
            float v = (y - p.y) / sigma;
            alpha += horizontal * exp(-0.5 * v * v);
        }
        alpha = saturate(alpha * (INV_SQRT_TWO_PI * dy / sigma));
    }
    if (data.flags != 0)
        return (1.0 - alpha) * saturate(0.5 - rounded_box(p, data.bounds, data.edges));
    return alpha;
}

float axis_intersection(float a, float b, float c, float direction)
{
    float den = b + direction * sqrt(max(b * b - 4.0 * a * c, 0.0));
    return den == 0.0 ? 0.0 : saturate(-2.0 * c / den);
}

float path_coverage(float2 p, Instance data)
{
    float2 winding = float2(data.backdrop);
    float2 weight = float2(0.0);
    for (uint i = 0; i < data.curve_count; i++)
    {
        Curve c = root.curves[data.curve_start + i];
        float left = min(c.start.x, c.end.x);
        if (left >= p.x + 0.5)
            break;
        if (c.start.y >= p.y + 0.5)
            continue;

        // Each listed endpoint accounts for an unlisted chain between the tile's
        // reference corner and the pixel. The path builder supplies the reference winding.
        winding.x += c.winding * (
            float(c.end.x <= data.offset.x) * (float(c.end.y <= p.y) - float(c.end.y <= data.offset.y)) -
            float(c.start.x <= data.offset.x) * (float(c.start.y <= p.y) - float(c.start.y <= data.offset.y)));
        winding.y += c.winding * (float(c.start.y <= data.offset.y && c.start.x <= p.x) -
            float(c.end.y <= data.offset.y && c.end.x <= p.x));

        if (c.start.y <= p.y && p.y < c.end.y)
        {
            float t = axis_intersection(c.quadratic.y, c.linear.y, c.start.y - p.y, 1.0);
            float x = c.start.x + t * (c.linear.x + t * c.quadratic.x);
            float distance = p.x - x;
            winding.x += c.winding * saturate(0.5 + distance);
            weight.x = max(weight.x, saturate(1.0 - 2.0 * abs(distance)));
        }
        float right = max(c.start.x, c.end.x);
        if (left <= p.x && p.x < right)
        {
            float direction = c.end.x > c.start.x ? 1.0 : -1.0;
            float t = axis_intersection(c.quadratic.x, c.linear.x, c.start.x - p.x, direction);
            float y = c.start.y + t * (c.linear.y + t * c.quadratic.y);
            float distance = p.y - y;
            winding.y -= c.winding * direction * saturate(0.5 + distance);
            weight.y = max(weight.y, saturate(1.0 - 2.0 * abs(distance)));
        }
    }
    float sum = weight.x + weight.y;
    float coverage = min(abs(winding.x), abs(winding.y));
    if (sum > 0.0)
        coverage = max(coverage, abs(dot(winding, weight) / sum));
    return data.flags != 0 ? 1.0 - abs(1.0 - 2.0 * frac(coverage * 0.5)) : saturate(coverage);
}

float underline_coverage(float2 p, Instance data)
{
    if (data.flags == 0)
        return 1.0;
    float height = data.bounds.w - data.bounds.y;
    float rate = TWO_PI * data.parameter / (height * height);
    float phase = (p.x - data.bounds.x) * rate;
    float amplitude = 0.8 * data.parameter;
    float slope = amplitude * rate * cos(phase);
    float center = (data.bounds.y + data.bounds.w) * 0.5 + amplitude * sin(phase);
    float distance = abs(p.y - center) * rsqrt(1.0 + slope * slope);
    return saturate(0.5 + 0.5 * data.parameter - distance);
}

float4 sprite(Instance data, float2 p)
{
    Texture2D<float4> texture = root.texture.get<Texture2D<float4>>();
    uint w, h;
    texture.GetDimensions(w, h);
    float2 inset = 0.5 / float2(w, h);
    float2 relative = (p - data.bounds.xy) / (data.bounds.zw - data.bounds.xy);
    float2 uv = clamp(lerp(data.uv.xy, data.uv.zw, relative), data.shape.xy + inset, data.shape.zw - inset);
    return texture.SampleLevel(root.sampler.get(), uv, 0.0);
}

[shader("pixel")]
float4 pixel_main(Vertex v) : SV_Target
{
    Instance data = root.instances[v.index];
    float2 p = v.position.xy;
    switch (root.kind)
    {
        case QUAD:
            return quad_color(p, data);
        case SHADOW:
            return premultiply_alpha(data.color, shadow_coverage(p, data));
        case UNDERLINE:
            return premultiply_alpha(data.color, underline_coverage(p, data));
        case PATH:
            return premultiply_alpha(fill_color(data.fill, data.shape, p), path_coverage(p, data));
        case MASK_SPRITE:
            if (any(p < data.clip.xy) || any(p >= data.clip.zw))
                return float4(0.0);
            return premultiply_alpha(data.color, sprite(data, v.local).r);
        default:
            float4 texel = sprite(data, v.local);
            if (data.flags == 0 && !any(data.radii > 0.0))
                return premultiply_alpha(texel, data.parameter);
            if (data.flags != 0)
                texel.rgb = float3(luma_rec709(texel.rgb));
            return premultiply_alpha(texel, data.parameter * saturate(0.5 - rounded_box(p, data.bounds, data.radii)));
    }
}
