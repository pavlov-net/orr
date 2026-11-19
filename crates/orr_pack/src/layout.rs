//! Layout class: what a cooked source holds besides its assets, in one mount blob
//! ([`LayoutInfo`]).
//!
//! A layout lists the source's groups, each a name and typed attributes, and its partitions,
//! each the content that belongs to exactly one set of groups: its instance sources, whose pages
//! place the models it holds, the models it holds that no page places, and its lights, decals,
//! cameras and environment in world space. A partition in no group holds what no group holds.
//! Groups say nothing of how a consumer combines them, such as whether content in two groups
//! shows when either is on or only when both are.
//!
//! The header's spans name tables in the same blob: `first` is a byte offset and `count` a
//! record count. Spans in records index other tables, `first` the first index, and strings are
//! spans of the string table's UTF-8 bytes.

use bytemuck::{Pod, Zeroable};
use orr_math::{DQuat, DVec3, Quat, Vec3};

use crate::blob::{put_table, read, span, table, text};
use crate::{AssetId, Error, Result, Span};

pub const SCHEMA: u16 = 4;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct LayoutHeader {
    pub groups: Span,
    pub partitions: Span,
    pub attributes: Span,
    /// Attribute values by kind ([`AttributeRecord::values`]): `f64` numbers, `i64` integers,
    /// and strings, each a [`Span`] of the string table.
    pub reals: Span,
    pub integers: Span,
    pub texts: Span,
    /// The group indices partitions name, `u32` each.
    pub memberships: Span,
    /// Instance sources, plain and posed, and posed models, an [`AssetId`] each.
    pub sources: Span,
    pub posed: Span,
    pub lights: Span,
    pub decals: Span,
    pub cameras: Span,
    pub environments: Span,
    pub strings: Span,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GroupRecord {
    pub name: Span,
    /// Indices into the attribute table.
    pub attributes: Span,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct PartitionRecord {
    /// Indices into the membership table.
    pub groups: Span,
    /// Indices into the source, posed, light, decal and camera tables.
    pub sources: Span,
    pub posed_sources: Span,
    pub posed: Span,
    pub lights: Span,
    pub decals: Span,
    pub cameras: Span,
    /// Index into the environment table, or [`NONE`].
    pub environment: u32,
    pub _pad: u32,
}

/// No environment.
pub const NONE: u32 = u32::MAX;

/// `AttributeRecord::kind`: the element kind, with [`ATTR_ARRAY`] for an array of them.
const ATTR_BOOL: u32 = 1;
const ATTR_INT: u32 = 2;
const ATTR_UINT64: u32 = 3;
const ATTR_FLOAT: u32 = 4;
const ATTR_FLOAT2: u32 = 5;
const ATTR_FLOAT3: u32 = 6;
const ATTR_FLOAT4: u32 = 7;
const ATTR_INT2: u32 = 8;
const ATTR_INT3: u32 = 9;
const ATTR_INT4: u32 = 10;
const ATTR_QUAT: u32 = 11;
const ATTR_MATRIX2: u32 = 12;
const ATTR_MATRIX3: u32 = 13;
const ATTR_MATRIX4: u32 = 14;
const ATTR_STRING: u32 = 15;
const ATTR_TOKEN: u32 = 16;
const ATTR_ASSET: u32 = 17;
const ATTR_PATH_EXPRESSION: u32 = 18;
const ATTR_ARRAY: u32 = 1 << 31;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct AttributeRecord {
    pub name: Span,
    pub kind: u32,
    pub _pad: u32,
    /// Its elements' values, one element for a value that is no array: of the integer table
    /// for bools (0 or 1), integers (a `uint64` as its bits) and integer vectors, of the string
    /// spans for strings, tokens, asset paths and path expressions, and of the real table for
    /// the rest, a quaternion as (x, y, z, w) and a matrix row by row.
    pub values: Span,
}

/// `LightRecord::kind`.
const LIGHT_DIRECTIONAL: u32 = 1;
const LIGHT_SPHERE: u32 = 2;
const LIGHT_RECT: u32 = 3;
const LIGHT_POINT: u32 = 4;

/// `LightRecord::flags`: the light's shaping is set.
const LIGHT_SHAPED: u32 = 1;
/// `LightRecord::flags`: surfaces cast no shadow from the light.
const LIGHT_UNSHADOWED: u32 = 2;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct LightRecord {
    /// Point, sphere and rect: the center.
    pub position: [f64; 3],
    pub kind: u32,
    /// Sphere: the radius; directional: the angular radius.
    pub radius: f32,
    /// Directional: illuminance; point and sphere: intensity; rect: luminance.
    pub color: [f32; 3],
    pub cone_angle: f32,
    /// Directional: toward the light; shaped: the shaping's axis.
    pub direction: [f32; 3],
    pub cone_softness: f32,
    pub edge_x: [f32; 3],
    pub flags: u32,
    pub edge_y: [f32; 3],
    pub focus: f32,
    pub focus_tint: [f32; 3],
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct DecalRecord {
    pub center: [f64; 3],
    pub channels: u32,
    pub opacity: f32,
    pub rotation: [f32; 4],
    pub half_extent: [f32; 3],
    pub depth_fade: f32,
    pub fade_angles: [f32; 2],
    pub order: i32,
    pub _pad: u32,
    pub material: AssetId,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct CameraRecord {
    pub position: [f64; 3],
    pub name: Span,
    pub rotation: [f32; 4],
    pub focal_length: f32,
    pub aperture: [f32; 2],
    pub near: f32,
    pub f_stop: f32,
    pub focus_distance: f32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct EnvironmentRecord {
    pub texture: AssetId,
    pub rotation: [f32; 4],
    pub color: [f32; 3],
    pub _pad: u32,
}

/// A typed value of a group's attribute, of USD's attribute value types: its signed and 32-bit
/// integers are [`Self::Int`], its `half`, `float`, `double` and `timecode` numbers
/// [`Self::Float`] and their vectors likewise, whatever role they state. Each kind has its
/// homogeneous array.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    Bool(bool),
    Int(i64),
    Uint64(u64),
    Float(f64),
    Float2([f64; 2]),
    Float3([f64; 3]),
    Float4([f64; 4]),
    Int2([i32; 2]),
    Int3([i32; 3]),
    Int4([i32; 4]),
    Quat(DQuat),
    /// Row by row, as USD states a matrix, which takes row vectors.
    Matrix2([f64; 4]),
    Matrix3([f64; 9]),
    Matrix4([f64; 16]),
    String(String),
    /// A name from a fixed set, as USD's `token`.
    Token(String),
    /// An asset path as authored.
    Asset(String),
    /// A path expression in USD's text form.
    PathExpression(String),
    BoolArray(Vec<bool>),
    IntArray(Vec<i64>),
    Uint64Array(Vec<u64>),
    FloatArray(Vec<f64>),
    Float2Array(Vec<[f64; 2]>),
    Float3Array(Vec<[f64; 3]>),
    Float4Array(Vec<[f64; 4]>),
    Int2Array(Vec<[i32; 2]>),
    Int3Array(Vec<[i32; 3]>),
    Int4Array(Vec<[i32; 4]>),
    QuatArray(Vec<DQuat>),
    Matrix2Array(Vec<[f64; 4]>),
    Matrix3Array(Vec<[f64; 9]>),
    Matrix4Array(Vec<[f64; 16]>),
    StringArray(Vec<String>),
    TokenArray(Vec<String>),
    AssetArray(Vec<String>),
    PathExpressionArray(Vec<String>),
}

/// An attribute value's elements as stored: the table it is in, flattened.
enum Values<'a> {
    Reals(Vec<f64>),
    Integers(Vec<i64>),
    Texts(Vec<&'a str>),
}

impl AttributeValue {
    /// Its record kind and elements.
    fn encode(&self) -> (u32, Values<'_>) {
        use AttributeValue as A;
        use Values::{Integers as I, Reals as R, Texts as T};
        let wide = |v: &[i32]| I(v.iter().map(|&x| i64::from(x)).collect());
        let quats = |q: &[DQuat]| R(q.iter().flat_map(DQuat::to_array).collect());
        fn texts(s: &[String]) -> Vec<&str> {
            s.iter().map(String::as_str).collect()
        }
        let (kind, values) = match self {
            A::Bool(b) => (ATTR_BOOL, I(vec![i64::from(*b)])),
            A::Int(i) => (ATTR_INT, I(vec![*i])),
            A::Uint64(u) => (ATTR_UINT64, I(vec![*u as i64])),
            A::Float(f) => (ATTR_FLOAT, R(vec![*f])),
            A::Float2(f) => (ATTR_FLOAT2, R(f.to_vec())),
            A::Float3(f) => (ATTR_FLOAT3, R(f.to_vec())),
            A::Float4(f) => (ATTR_FLOAT4, R(f.to_vec())),
            A::Int2(i) => (ATTR_INT2, wide(i)),
            A::Int3(i) => (ATTR_INT3, wide(i)),
            A::Int4(i) => (ATTR_INT4, wide(i)),
            A::Quat(q) => (ATTR_QUAT, quats(std::slice::from_ref(q))),
            A::Matrix2(m) => (ATTR_MATRIX2, R(m.to_vec())),
            A::Matrix3(m) => (ATTR_MATRIX3, R(m.to_vec())),
            A::Matrix4(m) => (ATTR_MATRIX4, R(m.to_vec())),
            A::String(s) => (ATTR_STRING, T(vec![s])),
            A::Token(s) => (ATTR_TOKEN, T(vec![s])),
            A::Asset(s) => (ATTR_ASSET, T(vec![s])),
            A::PathExpression(s) => (ATTR_PATH_EXPRESSION, T(vec![s])),
            A::BoolArray(b) => {
                let bits = b.iter().map(|&b| i64::from(b));
                (ATTR_BOOL | ATTR_ARRAY, I(bits.collect()))
            }
            A::IntArray(i) => (ATTR_INT | ATTR_ARRAY, I(i.clone())),
            A::Uint64Array(u) => {
                let bits = u.iter().map(|&u| u as i64);
                (ATTR_UINT64 | ATTR_ARRAY, I(bits.collect()))
            }
            A::FloatArray(f) => (ATTR_FLOAT | ATTR_ARRAY, R(f.clone())),
            A::Float2Array(f) => (ATTR_FLOAT2 | ATTR_ARRAY, R(f.as_flattened().to_vec())),
            A::Float3Array(f) => (ATTR_FLOAT3 | ATTR_ARRAY, R(f.as_flattened().to_vec())),
            A::Float4Array(f) => (ATTR_FLOAT4 | ATTR_ARRAY, R(f.as_flattened().to_vec())),
            A::Int2Array(i) => (ATTR_INT2 | ATTR_ARRAY, wide(i.as_flattened())),
            A::Int3Array(i) => (ATTR_INT3 | ATTR_ARRAY, wide(i.as_flattened())),
            A::Int4Array(i) => (ATTR_INT4 | ATTR_ARRAY, wide(i.as_flattened())),
            A::QuatArray(q) => (ATTR_QUAT | ATTR_ARRAY, quats(q)),
            A::Matrix2Array(m) => (ATTR_MATRIX2 | ATTR_ARRAY, R(m.as_flattened().to_vec())),
            A::Matrix3Array(m) => (ATTR_MATRIX3 | ATTR_ARRAY, R(m.as_flattened().to_vec())),
            A::Matrix4Array(m) => (ATTR_MATRIX4 | ATTR_ARRAY, R(m.as_flattened().to_vec())),
            A::StringArray(s) => (ATTR_STRING | ATTR_ARRAY, T(texts(s))),
            A::TokenArray(s) => (ATTR_TOKEN | ATTR_ARRAY, T(texts(s))),
            A::AssetArray(s) => (ATTR_ASSET | ATTR_ARRAY, T(texts(s))),
            A::PathExpressionArray(s) => (ATTR_PATH_EXPRESSION | ATTR_ARRAY, T(texts(s))),
        };
        (kind, values)
    }
}

/// The tables attribute values are read from.
struct ValueTables<'a> {
    reals: &'a [f64],
    integers: &'a [i64],
    texts: &'a [Span],
    strings: &'a [u8],
}

const MALFORMED_VALUE: Error = Error::Malformed("layout attribute value");

/// `v` as elements of `N` values.
fn elements<T: Copy, const N: usize>(v: &[T]) -> Result<Vec<[T; N]>> {
    match v.as_chunks::<N>() {
        (whole, []) => Ok(whole.to_vec()),
        _ => Err(MALFORMED_VALUE),
    }
}

/// `v` as one element of `N` values.
fn element<T: Copy, const N: usize>(v: &[T]) -> Result<[T; N]> {
    v.try_into().map_err(|_| MALFORMED_VALUE)
}

fn bit(i: i64) -> Result<bool> {
    match i {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(MALFORMED_VALUE),
    }
}

fn narrow(v: &[i64]) -> Result<Vec<i32>> {
    v.iter()
        .map(|&i| i32::try_from(i).map_err(|_| MALFORMED_VALUE))
        .collect()
}

fn quats(v: &[f64]) -> Result<Vec<DQuat>> {
    Ok(elements::<_, 4>(v)?
        .into_iter()
        .map(DQuat::from_array)
        .collect())
}

fn attribute_value(t: &ValueTables<'_>, r: &AttributeRecord) -> Result<AttributeValue> {
    use AttributeValue as A;
    let array = r.kind & ATTR_ARRAY != 0;
    let kind = r.kind & !ATTR_ARRAY;
    let reals = || slice(t.reals, r.values);
    let ints = || slice(t.integers, r.values);
    let texts = || {
        slice(t.texts, r.values)?
            .iter()
            .map(|&s| string(t.strings, s))
            .collect::<Result<Vec<_>>>()
    };
    let text = || match <[String; 1]>::try_from(texts()?) {
        Ok([s]) => Ok(s),
        Err(_) => Err(MALFORMED_VALUE),
    };
    let one_int = || Ok::<_, Error>(element::<_, 1>(ints()?)?[0]);
    Ok(match (kind, array) {
        (ATTR_BOOL, false) => A::Bool(bit(one_int()?)?),
        (ATTR_INT, false) => A::Int(one_int()?),
        (ATTR_UINT64, false) => A::Uint64(one_int()? as u64),
        (ATTR_FLOAT, false) => A::Float(element::<_, 1>(reals()?)?[0]),
        (ATTR_FLOAT2, false) => A::Float2(element(reals()?)?),
        (ATTR_FLOAT3, false) => A::Float3(element(reals()?)?),
        (ATTR_FLOAT4, false) => A::Float4(element(reals()?)?),
        (ATTR_INT2, false) => A::Int2(element(&narrow(ints()?)?)?),
        (ATTR_INT3, false) => A::Int3(element(&narrow(ints()?)?)?),
        (ATTR_INT4, false) => A::Int4(element(&narrow(ints()?)?)?),
        (ATTR_QUAT, false) => A::Quat(DQuat::from_array(element(reals()?)?)),
        (ATTR_MATRIX2, false) => A::Matrix2(element(reals()?)?),
        (ATTR_MATRIX3, false) => A::Matrix3(element(reals()?)?),
        (ATTR_MATRIX4, false) => A::Matrix4(element(reals()?)?),
        (ATTR_STRING, false) => A::String(text()?),
        (ATTR_TOKEN, false) => A::Token(text()?),
        (ATTR_ASSET, false) => A::Asset(text()?),
        (ATTR_PATH_EXPRESSION, false) => A::PathExpression(text()?),
        (ATTR_BOOL, true) => A::BoolArray(ints()?.iter().map(|&i| bit(i)).collect::<Result<_>>()?),
        (ATTR_INT, true) => A::IntArray(ints()?.to_vec()),
        (ATTR_UINT64, true) => A::Uint64Array(ints()?.iter().map(|&i| i as u64).collect()),
        (ATTR_FLOAT, true) => A::FloatArray(reals()?.to_vec()),
        (ATTR_FLOAT2, true) => A::Float2Array(elements(reals()?)?),
        (ATTR_FLOAT3, true) => A::Float3Array(elements(reals()?)?),
        (ATTR_FLOAT4, true) => A::Float4Array(elements(reals()?)?),
        (ATTR_INT2, true) => A::Int2Array(elements(&narrow(ints()?)?)?),
        (ATTR_INT3, true) => A::Int3Array(elements(&narrow(ints()?)?)?),
        (ATTR_INT4, true) => A::Int4Array(elements(&narrow(ints()?)?)?),
        (ATTR_QUAT, true) => A::QuatArray(quats(reals()?)?),
        (ATTR_MATRIX2, true) => A::Matrix2Array(elements(reals()?)?),
        (ATTR_MATRIX3, true) => A::Matrix3Array(elements(reals()?)?),
        (ATTR_MATRIX4, true) => A::Matrix4Array(elements(reals()?)?),
        (ATTR_STRING, true) => A::StringArray(texts()?),
        (ATTR_TOKEN, true) => A::TokenArray(texts()?),
        (ATTR_ASSET, true) => A::AssetArray(texts()?),
        (ATTR_PATH_EXPRESSION, true) => A::PathExpressionArray(texts()?),
        _ => return Err(Error::Malformed("layout attribute kind")),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub value: AttributeValue,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub name: String,
    pub attributes: Vec<Attribute>,
}

/// A light, in world space (meters, Y up), colors linear Rec.709, as `orr_scene`'s `Light`
/// takes it. Surfaces cast no shadow from a point, sphere or rect light without `shadows`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightSource {
    Directional {
        direction_to_light: Vec3,
        /// Lux.
        illuminance: Vec3,
        /// Radians.
        angular_radius: f32,
    },
    Point {
        position: DVec3,
        /// Candela.
        intensity: Vec3,
        shaping: Option<ShapingSource>,
        shadows: bool,
    },
    Sphere {
        position: DVec3,
        radius: f32,
        /// Candela.
        intensity: Vec3,
        shaping: Option<ShapingSource>,
        shadows: bool,
    },
    /// One-sided, emitting toward `edge_x` x `edge_y`.
    Rect {
        center: DVec3,
        edge_x: Vec3,
        edge_y: Vec3,
        /// Nits.
        luminance: Vec3,
        shaping: Option<ShapingSource>,
        shadows: bool,
    },
}

/// How a light's emission falls off around `axis`, in UsdLux `ShapingAPI`'s terms: none past
/// `cone_angle` (radians), smoothly to 0 over the outer `cone_softness` share of it (in
/// [0, 1]), and by `lerp(focus_tint, 1, |cos|^focus)` (`focus` at least 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapingSource {
    pub axis: Vec3,
    pub cone_angle: f32,
    pub cone_softness: f32,
    pub focus: f32,
    pub focus_tint: Vec3,
}

/// A box decal, as `orr_scene`'s `Decal` takes it, its material an asset of the archive or of
/// another the consumer mounts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecalSource {
    pub center: DVec3,
    pub rotation: Quat,
    pub half_extent: Vec3,
    pub material: AssetId,
    pub channels: u32,
    pub opacity: f32,
    pub fade_angles: [f32; 2],
    pub depth_fade: f32,
    /// Where decals overlap, higher orders are written over lower ones.
    pub order: i32,
}

/// An authored view: a camera at `position` turned by `rotation` from looking down -Z with +Y
/// up.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraSource {
    pub name: String,
    pub position: DVec3,
    pub rotation: Quat,
    /// Meters.
    pub focal_length: f32,
    /// Width and height of the film back, meters.
    pub aperture: [f32; 2],
    /// Meters.
    pub near: f32,
    /// 0 for no depth of field.
    pub f_stop: f32,
    /// Meters.
    pub focus_distance: f32,
}

/// An environment map: `texture` looked up at `rotation` of a world direction, times `color`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvironmentSource {
    pub texture: AssetId,
    pub rotation: Quat,
    pub color: Vec3,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Partition {
    /// Indices into [`LayoutInfo::groups`], ascending.
    pub groups: Vec<u32>,
    /// Its instance sources whose pages place what they hold as stored.
    pub sources: Vec<AssetId>,
    /// Its instance sources whose pages the consumer places: pages of a skinned model, whose
    /// copies take palettes, of a model with a skeleton, whose parts take poses, and pages that
    /// hang from a joint, whose records stand in its bind space.
    pub posed_sources: Vec<AssetId>,
    /// The models it holds that no page places, in scene space: the consumer poses them by
    /// their skeletons.
    pub posed: Vec<AssetId>,
    pub lights: Vec<LightSource>,
    pub decals: Vec<DecalSource>,
    pub cameras: Vec<CameraSource>,
    pub environment: Option<EnvironmentSource>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayoutInfo {
    pub groups: Vec<Group>,
    pub partitions: Vec<Partition>,
}

/// The records `s` indexes in `t`.
fn slice<T>(t: &[T], s: Span) -> Result<&[T]> {
    span(t, s).ok_or(Error::Malformed("layout index span"))
}

/// The string at span `s` of the string table.
fn string(strings: &[u8], s: Span) -> Result<String> {
    text(strings, s)
        .map(str::to_owned)
        .ok_or(Error::Malformed("layout string"))
}

fn light(r: &LightRecord) -> Result<LightSource> {
    let position = DVec3::from_array(r.position);
    let color = Vec3::from_array(r.color);
    let shaping = (r.flags & LIGHT_SHAPED != 0).then(|| ShapingSource {
        axis: Vec3::from_array(r.direction),
        cone_angle: r.cone_angle,
        cone_softness: r.cone_softness,
        focus: r.focus,
        focus_tint: Vec3::from_array(r.focus_tint),
    });
    let shadows = r.flags & LIGHT_UNSHADOWED == 0;
    Ok(match r.kind {
        LIGHT_DIRECTIONAL => LightSource::Directional {
            direction_to_light: Vec3::from_array(r.direction),
            illuminance: color,
            angular_radius: r.radius,
        },
        LIGHT_POINT => LightSource::Point {
            position,
            intensity: color,
            shaping,
            shadows,
        },
        LIGHT_SPHERE => LightSource::Sphere {
            position,
            radius: r.radius,
            intensity: color,
            shaping,
            shadows,
        },
        LIGHT_RECT => LightSource::Rect {
            center: position,
            edge_x: Vec3::from_array(r.edge_x),
            edge_y: Vec3::from_array(r.edge_y),
            luminance: color,
            shaping,
            shadows,
        },
        _ => return Err(Error::Malformed("layout light kind")),
    })
}

fn light_record(l: &LightSource) -> LightRecord {
    let mut r = LightRecord::zeroed();
    let shaping = match *l {
        LightSource::Directional {
            direction_to_light,
            illuminance,
            angular_radius,
        } => {
            r.kind = LIGHT_DIRECTIONAL;
            r.direction = direction_to_light.to_array();
            r.color = illuminance.to_array();
            r.radius = angular_radius;
            None
        }
        LightSource::Point {
            position,
            intensity,
            shaping,
            shadows,
        } => {
            r.kind = LIGHT_POINT;
            r.position = position.to_array();
            r.color = intensity.to_array();
            r.flags = if shadows { 0 } else { LIGHT_UNSHADOWED };
            shaping
        }
        LightSource::Sphere {
            position,
            radius,
            intensity,
            shaping,
            shadows,
        } => {
            r.kind = LIGHT_SPHERE;
            r.position = position.to_array();
            r.radius = radius;
            r.color = intensity.to_array();
            r.flags = if shadows { 0 } else { LIGHT_UNSHADOWED };
            shaping
        }
        LightSource::Rect {
            center,
            edge_x,
            edge_y,
            luminance,
            shaping,
            shadows,
        } => {
            r.kind = LIGHT_RECT;
            r.position = center.to_array();
            r.edge_x = edge_x.to_array();
            r.edge_y = edge_y.to_array();
            r.color = luminance.to_array();
            r.flags = if shadows { 0 } else { LIGHT_UNSHADOWED };
            shaping
        }
    };
    if let Some(s) = shaping {
        r.flags |= LIGHT_SHAPED;
        r.direction = s.axis.to_array();
        r.cone_angle = s.cone_angle;
        r.cone_softness = s.cone_softness;
        r.focus = s.focus;
        r.focus_tint = s.focus_tint.to_array();
    }
    r
}

/// The string table being written.
#[derive(Default)]
struct Strings(Vec<u8>);

impl Strings {
    fn add(&mut self, s: &str) -> Span {
        let first = self.0.len() as u32;
        self.0.extend_from_slice(s.as_bytes());
        Span {
            first,
            count: s.len() as u32,
        }
    }
}

impl LayoutInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let h: LayoutHeader = read(blob, 0).ok_or(Error::Malformed("layout header"))?;
        let strings: Vec<u8> = table(blob, h.strings)?;
        let attributes: Vec<AttributeRecord> = table(blob, h.attributes)?;
        let reals: Vec<f64> = table(blob, h.reals)?;
        let integers: Vec<i64> = table(blob, h.integers)?;
        let texts: Vec<Span> = table(blob, h.texts)?;
        let values = ValueTables {
            reals: &reals,
            integers: &integers,
            texts: &texts,
            strings: &strings,
        };
        let memberships: Vec<u32> = table(blob, h.memberships)?;
        let sources: Vec<AssetId> = table(blob, h.sources)?;
        let posed: Vec<AssetId> = table(blob, h.posed)?;
        let lights: Vec<LightRecord> = table(blob, h.lights)?;
        let decals: Vec<DecalRecord> = table(blob, h.decals)?;
        let cameras: Vec<CameraRecord> = table(blob, h.cameras)?;
        let environments: Vec<EnvironmentRecord> = table(blob, h.environments)?;
        let groups = table::<GroupRecord>(blob, h.groups)?
            .iter()
            .map(|g| {
                let attributes = slice(&attributes, g.attributes)?
                    .iter()
                    .map(|a| {
                        Ok(Attribute {
                            name: string(&strings, a.name)?,
                            value: attribute_value(&values, a)?,
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(Group {
                    name: string(&strings, g.name)?,
                    attributes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let partitions = table::<PartitionRecord>(blob, h.partitions)?
            .iter()
            .map(|p| {
                let groups = slice(&memberships, p.groups)?.to_vec();
                if groups
                    .iter()
                    .any(|&g| g as usize >= h.groups.count as usize)
                    || !groups.is_sorted_by(|a, b| a < b)
                {
                    return Err(Error::Malformed("layout partition groups"));
                }
                let environment = match p.environment {
                    NONE => None,
                    i => {
                        let e = environments
                            .get(i as usize)
                            .ok_or(Error::Malformed("layout environment index"))?;
                        Some(EnvironmentSource {
                            texture: e.texture,
                            rotation: Quat::from_array(e.rotation),
                            color: Vec3::from_array(e.color),
                        })
                    }
                };
                Ok(Partition {
                    groups,
                    sources: slice(&sources, p.sources)?.to_vec(),
                    posed_sources: slice(&sources, p.posed_sources)?.to_vec(),
                    posed: slice(&posed, p.posed)?.to_vec(),
                    lights: slice(&lights, p.lights)?
                        .iter()
                        .map(light)
                        .collect::<Result<_>>()?,
                    decals: slice(&decals, p.decals)?
                        .iter()
                        .map(|d| DecalSource {
                            center: DVec3::from_array(d.center),
                            rotation: Quat::from_array(d.rotation),
                            half_extent: Vec3::from_array(d.half_extent),
                            material: d.material,
                            channels: d.channels,
                            opacity: d.opacity,
                            fade_angles: d.fade_angles,
                            depth_fade: d.depth_fade,
                            order: d.order,
                        })
                        .collect(),
                    cameras: slice(&cameras, p.cameras)?
                        .iter()
                        .map(|c| {
                            Ok(CameraSource {
                                name: string(&strings, c.name)?,
                                position: DVec3::from_array(c.position),
                                rotation: Quat::from_array(c.rotation),
                                focal_length: c.focal_length,
                                aperture: c.aperture,
                                near: c.near,
                                f_stop: c.f_stop,
                                focus_distance: c.focus_distance,
                            })
                        })
                        .collect::<Result<_>>()?,
                    environment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { groups, partitions })
    }

    /// Blob bytes. Panics unless every partition's groups are ascending indices of
    /// [`Self::groups`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.groups.len();
        for p in &self.partitions {
            assert!(
                p.groups.is_sorted_by(|a, b| a < b) && p.groups.iter().all(|&g| (g as usize) < n),
                "partition groups are ascending group indices"
            );
        }
        let mut strings = Strings::default();
        let span = |first: usize, count: usize| Span {
            first: first as u32,
            count: count as u32,
        };
        let mut attributes = Vec::new();
        let mut reals = Vec::new();
        let mut integers = Vec::new();
        let mut texts = Vec::new();
        let mut groups = Vec::with_capacity(n);
        for g in &self.groups {
            let first = attributes.len();
            for a in &g.attributes {
                let (kind, values) = a.value.encode();
                let values = match values {
                    Values::Reals(v) => {
                        let s = span(reals.len(), v.len());
                        reals.extend(v);
                        s
                    }
                    Values::Integers(v) => {
                        let s = span(integers.len(), v.len());
                        integers.extend(v);
                        s
                    }
                    Values::Texts(v) => {
                        let s = span(texts.len(), v.len());
                        texts.extend(v.into_iter().map(|t| strings.add(t)));
                        s
                    }
                };
                attributes.push(AttributeRecord {
                    name: strings.add(&a.name),
                    kind,
                    _pad: 0,
                    values,
                });
            }
            groups.push(GroupRecord {
                name: strings.add(&g.name),
                attributes: span(first, g.attributes.len()),
            });
        }
        let mut memberships = Vec::new();
        let mut sources = Vec::new();
        let mut posed = Vec::new();
        let mut lights = Vec::new();
        let mut decals = Vec::new();
        let mut cameras = Vec::new();
        let mut environments = Vec::new();
        let mut partitions = Vec::with_capacity(self.partitions.len());
        for p in &self.partitions {
            partitions.push(PartitionRecord {
                groups: span(memberships.len(), p.groups.len()),
                sources: span(sources.len(), p.sources.len()),
                posed_sources: span(sources.len() + p.sources.len(), p.posed_sources.len()),
                posed: span(posed.len(), p.posed.len()),
                lights: span(lights.len(), p.lights.len()),
                decals: span(decals.len(), p.decals.len()),
                cameras: span(cameras.len(), p.cameras.len()),
                environment: p.environment.map_or(NONE, |_| environments.len() as u32),
                _pad: 0,
            });
            memberships.extend_from_slice(&p.groups);
            sources.extend_from_slice(&p.sources);
            sources.extend_from_slice(&p.posed_sources);
            posed.extend_from_slice(&p.posed);
            lights.extend(p.lights.iter().map(light_record));
            decals.extend(p.decals.iter().map(|d| DecalRecord {
                center: d.center.to_array(),
                channels: d.channels,
                opacity: d.opacity,
                rotation: d.rotation.to_array(),
                half_extent: d.half_extent.to_array(),
                depth_fade: d.depth_fade,
                fade_angles: d.fade_angles,
                order: d.order,
                _pad: 0,
                material: d.material,
            }));
            cameras.extend(p.cameras.iter().map(|c| CameraRecord {
                position: c.position.to_array(),
                name: strings.add(&c.name),
                rotation: c.rotation.to_array(),
                focal_length: c.focal_length,
                aperture: c.aperture,
                near: c.near,
                f_stop: c.f_stop,
                focus_distance: c.focus_distance,
                _pad: [0; 2],
            }));
            environments.extend(p.environment.iter().map(|e| EnvironmentRecord {
                texture: e.texture,
                rotation: e.rotation.to_array(),
                color: e.color.to_array(),
                _pad: 0,
            }));
        }
        let mut out = vec![0; size_of::<LayoutHeader>()];
        let h = LayoutHeader {
            lights: put_table(&mut out, &lights),
            decals: put_table(&mut out, &decals),
            cameras: put_table(&mut out, &cameras),
            attributes: put_table(&mut out, &attributes),
            reals: put_table(&mut out, &reals),
            integers: put_table(&mut out, &integers),
            texts: put_table(&mut out, &texts),
            environments: put_table(&mut out, &environments),
            groups: put_table(&mut out, &groups),
            partitions: put_table(&mut out, &partitions),
            memberships: put_table(&mut out, &memberships),
            sources: put_table(&mut out, &sources),
            posed: put_table(&mut out, &posed),
            strings: put_table(&mut out, &strings.0),
        };
        out[..size_of::<LayoutHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let info = LayoutInfo {
            groups: vec![
                Group {
                    name: "/World/Layers/Night".into(),
                    attributes: vec![
                        Attribute {
                            name: "state".into(),
                            value: AttributeValue::Token("activated".into()),
                        },
                        Attribute {
                            name: "count".into(),
                            value: AttributeValue::Int(-3),
                        },
                        Attribute {
                            name: "on".into(),
                            value: AttributeValue::Bool(true),
                        },
                        Attribute {
                            name: "tint".into(),
                            value: AttributeValue::Float3([0.5, 0.25, 1.0]),
                        },
                        Attribute {
                            name: "label".into(),
                            value: AttributeValue::String("Nacht".into()),
                        },
                        Attribute {
                            name: "id".into(),
                            value: AttributeValue::Uint64(u64::MAX - 1),
                        },
                        Attribute {
                            name: "cell".into(),
                            value: AttributeValue::Int2([-7, i32::MAX]),
                        },
                        Attribute {
                            name: "turn".into(),
                            value: AttributeValue::Quat(DQuat::from_rotation_z(0.5)),
                        },
                        Attribute {
                            name: "frame".into(),
                            value: AttributeValue::Matrix3([
                                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                            ]),
                        },
                        Attribute {
                            name: "texture".into(),
                            value: AttributeValue::Asset("./night.exr".into()),
                        },
                        Attribute {
                            name: "tags".into(),
                            value: AttributeValue::TokenArray(vec![
                                "a".into(),
                                "".into(),
                                "c".into(),
                            ]),
                        },
                        Attribute {
                            name: "flags".into(),
                            value: AttributeValue::BoolArray(vec![true, false]),
                        },
                        Attribute {
                            name: "spots".into(),
                            value: AttributeValue::Float3Array(vec![
                                [1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                            ]),
                        },
                        Attribute {
                            name: "cells".into(),
                            value: AttributeValue::Int3Array(vec![[1, -2, 3]]),
                        },
                        Attribute {
                            name: "ids".into(),
                            value: AttributeValue::Uint64Array(vec![u64::MAX, 0]),
                        },
                        Attribute {
                            name: "none".into(),
                            value: AttributeValue::FloatArray(Vec::new()),
                        },
                    ],
                },
                Group {
                    name: "/World/Layers/Candles".into(),
                    attributes: vec![],
                },
            ],
            partitions: vec![
                Partition::default(),
                Partition {
                    groups: vec![0, 1],
                    sources: vec![AssetId::from_name("a"), AssetId::from_name("b")],
                    posed_sources: vec![AssetId::from_name("skinned")],
                    posed: vec![AssetId::from_name("rig")],
                    lights: vec![
                        LightSource::Directional {
                            direction_to_light: Vec3::Y,
                            illuminance: Vec3::splat(1e5),
                            angular_radius: 0.0047,
                        },
                        LightSource::Sphere {
                            position: DVec3::new(1e6, 2.0, -3.0),
                            radius: 0.1,
                            intensity: Vec3::new(10.0, 20.0, 30.0),
                            shaping: Some(ShapingSource {
                                axis: -Vec3::Y,
                                cone_angle: 0.5,
                                cone_softness: 0.25,
                                focus: 2.0,
                                focus_tint: Vec3::new(0.1, 0.2, 0.3),
                            }),
                            shadows: true,
                        },
                        LightSource::Point {
                            position: DVec3::ZERO,
                            intensity: Vec3::ONE,
                            shaping: None,
                            shadows: false,
                        },
                        LightSource::Rect {
                            center: DVec3::new(0.0, 3.0, 0.0),
                            edge_x: Vec3::X,
                            edge_y: Vec3::Z,
                            luminance: Vec3::splat(500.0),
                            shaping: None,
                            shadows: true,
                        },
                    ],
                    decals: vec![DecalSource {
                        center: DVec3::new(4.0, 0.0, 1e7),
                        rotation: Quat::from_rotation_y(0.3),
                        half_extent: Vec3::new(1.0, 2.0, 0.5),
                        material: AssetId::from_name("m"),
                        channels: 5,
                        opacity: 0.75,
                        fade_angles: [1.0, 1.5],
                        depth_fade: 0.25,
                        order: -2,
                    }],
                    cameras: vec![CameraSource {
                        name: "editor".into(),
                        position: DVec3::new(-1.0, 1.7, 9.0),
                        rotation: Quat::from_rotation_x(-0.1),
                        focal_length: 0.035,
                        aperture: [0.036, 0.024],
                        near: 0.1,
                        f_stop: 0.0,
                        focus_distance: 5.0,
                    }],
                    environment: Some(EnvironmentSource {
                        texture: AssetId::from_name("sky"),
                        rotation: Quat::from_rotation_y(1.0),
                        color: Vec3::new(1.0, 0.9, 0.8),
                    }),
                },
                Partition {
                    groups: vec![1],
                    sources: vec![AssetId::from_name("c")],
                    posed_sources: vec![AssetId::from_name("hung"), AssetId::from_name("d")],
                    ..Partition::default()
                },
            ],
        };
        assert_eq!(LayoutInfo::parse(&info.to_bytes()).unwrap(), info);
    }
}
