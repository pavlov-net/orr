//! Reading a composed stage: the resolver that records every file the stage reads, value
//! conversions, transform stacks and primvars.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use openusd::ar::Resolver;
use openusd::{ar, gf, usd, vt};
use openusd_schemas::geom::{self, Interpolation, XformableExt};
use orr_math::{DMat4, DQuat, DVec3};
use orr_pack::Hash32;

use super::VariantSelection;
use crate::shading;
use crate::state::{Input, hash_file, hash_path};
use crate::texture::color::ImageColor;
use crate::texture::tiles::UDIM_TILES;

/// Identifier of the session layer that carries the manifest's variant selections.
pub(super) const SESSION: &str = "orr_cook_session.usda";

/// Files the stage read, with the hash of the bytes it was served.
#[derive(Default)]
pub(super) struct Reads {
    files: BTreeMap<PathBuf, Hash32>,
    /// Where assets that did not resolve or read would appear.
    absent: BTreeSet<PathBuf>,
    /// The tiles of each `<UDIM>` identifier, by number, with their resolved paths.
    tiles: BTreeMap<String, Vec<(u16, String)>>,
    pub(super) session: String,
}

/// The default resolver, serving files from read-only mappings of them, recording every file it
/// serves and serving the session layer.
pub(super) struct Recorder {
    inner: ar::DefaultResolver,
    reads: Rc<RefCell<Reads>>,
}

impl Recorder {
    pub(super) fn new(reads: Rc<RefCell<Reads>>) -> Self {
        // SAFETY: the cook never writes its sources. Another program that rewrites one in place
        // while the cook runs breaks the promise; on Windows a mapped file is opened with write
        // sharing denied, which refuses that program's open, and elsewhere nothing stops it.
        let inner = unsafe { ar::DefaultResolver::new().map_files() };
        Self { inner, reads }
    }

    /// `asset_path` as a path of a file: a bare one stays bare until a file appears beside the
    /// layer, so that is where it would appear.
    fn beside(&self, asset_path: &str, anchor: Option<&ar::ResolvedPath>) -> String {
        let id = self.inner.create_identifier(asset_path, anchor);
        if Path::new(&id).is_absolute() {
            return id;
        }
        self.inner
            .create_identifier(&format!("./{asset_path}"), anchor)
    }

    /// The identifier of `udim`, the asset path `asset_path`, after finding its tiles as
    /// `UsdShadeUdimUtils` does, each the way a path of its own would resolve. Where a tile does
    /// not resolve is where it would appear.
    fn tiled_identifier(
        &self,
        asset_path: &str,
        udim: &ar::UdimPath,
        anchor: Option<&ar::ResolvedPath>,
    ) -> String {
        let id = self.beside(asset_path, anchor);
        if self.reads.borrow().tiles.contains_key(&id) {
            return id;
        }
        let anchored = |tile: &str| self.inner.create_identifier(tile, anchor);
        let found = udim.resolve_tiles(&self.inner, Some(&anchored));
        let mut found = found.into_iter().peekable();
        let mut tiles = Vec::new();
        for n in UDIM_TILES {
            match found.next_if(|t| t.number == n) {
                Some(t) => tiles.push((n, t.path.to_string_lossy().into_owned())),
                None => {
                    let file = outer(Path::new(&self.beside(&udim.tile(n), anchor)));
                    self.reads.borrow_mut().absent.insert(file);
                }
            }
        }
        self.reads.borrow_mut().tiles.insert(id.clone(), tiles);
        id
    }

    /// The bytes of an asset (an image) by its resolved path, as a view of the mapped file that
    /// holds them, recording its file, which is watched for a change when it can't be read.
    pub(super) fn read(&self, resolved: &str) -> io::Result<ar::SharedBuffer> {
        let read = self
            .open_asset(&ar::ResolvedPath::new(resolved))
            .and_then(ar::Asset::into_buffer)
            .map(ar::AssetBuffer::into_shared);
        if read.is_err() {
            let file = outer(Path::new(resolved));
            self.reads.borrow_mut().absent.insert(file);
        }
        read
    }
}

/// The file a resolved path lives in: a package's own file for one of its entries.
fn outer(resolved: &Path) -> PathBuf {
    let s = resolved.to_string_lossy();
    match s.find('[') {
        Some(i) if s.ends_with(']') => PathBuf::from(&s[..i]),
        _ => resolved.to_owned(),
    }
}

impl Resolver for Recorder {
    fn create_identifier(&self, asset_path: &str, anchor: Option<&ar::ResolvedPath>) -> String {
        if asset_path == SESSION {
            return SESSION.into();
        }
        if let Some(udim) = ar::UdimPath::parse(asset_path) {
            return self.tiled_identifier(asset_path, &udim, anchor);
        }
        let id = self.inner.create_identifier(asset_path, anchor);
        if !asset_path.is_empty() && self.inner.resolve(&id).is_none() {
            let file = outer(Path::new(&self.beside(asset_path, anchor)));
            self.reads.borrow_mut().absent.insert(file);
        }
        id
    }

    fn resolve(&self, asset_path: &str) -> Option<ar::ResolvedPath> {
        if asset_path == SESSION {
            return Some(ar::ResolvedPath::new(SESSION));
        }
        if let Some(tiles) = self.reads.borrow().tiles.get(asset_path) {
            return (!tiles.is_empty()).then(|| ar::ResolvedPath::new(asset_path));
        }
        self.inner.resolve(asset_path)
    }

    fn resolve_for_new_asset(&self, asset_path: &str) -> Option<ar::ResolvedPath> {
        self.inner.resolve_for_new_asset(asset_path)
    }

    fn open_asset(&self, resolved: &ar::ResolvedPath) -> io::Result<Box<dyn ar::Asset>> {
        if resolved.to_string_lossy() == SESSION {
            let text = self.reads.borrow().session.clone().into_bytes();
            return Ok(Box::new(io::Cursor::new(text)));
        }
        let file = outer(resolved);
        let asset = self.inner.open_asset(resolved)?;
        let whole = file.as_path() == &**resolved;
        let mut reads = self.reads.borrow_mut();
        let std::collections::btree_map::Entry::Vacant(e) = reads.files.entry(file) else {
            return Ok(asset);
        };
        if !whole {
            // A package's entry is hashed by the package's whole file.
            let hash = hash_file(e.key()).map_err(io::Error::other)?;
            e.insert(hash);
            return Ok(asset);
        }
        // The hash is of the bytes served, which a file that is not mapped is read into.
        let buffer = asset.into_buffer()?;
        e.insert(Hash32::of(&buffer));
        Ok(Box::new(io::Cursor::new(buffer)))
    }

    fn is_context_dependent_path(&self, asset_path: &str) -> bool {
        self.inner.is_context_dependent_path(asset_path)
    }

    fn in_memory_identifier<'a>(&self, asset_path: &'a str) -> std::borrow::Cow<'a, str> {
        self.inner.in_memory_identifier(asset_path)
    }

    fn identity(&self) -> String {
        self.inner.identity()
    }

    fn cache_id(&self) -> Option<u64> {
        self.inner.cache_id()
    }
}

impl Reads {
    /// The tiles of a resolved `<UDIM>` path.
    pub(super) fn tiles(&self, resolved: &str) -> Option<&[(u16, String)]> {
        self.tiles.get(resolved).map(Vec::as_slice)
    }

    pub(super) fn inputs(&self) -> Vec<Input> {
        let read = self.files.iter().map(|(path, &hash)| Input {
            path: path.clone(),
            hash,
        });
        let absent = self
            .absent
            .iter()
            .filter(|p| !self.files.contains_key(*p))
            .map(|path| Input {
                path: path.clone(),
                // A package can stand here without the entry that was asked of it.
                hash: hash_path(path).unwrap_or(Input::ABSENT),
            });
        read.chain(absent).collect()
    }
}

/// A session layer selecting `variants`.
pub(super) fn session_layer(variants: &[VariantSelection]) -> Result<String, String> {
    // Nested `over`s per prim path; selections on each.
    #[derive(Default)]
    struct Over {
        selections: Vec<(String, String)>,
        children: BTreeMap<String, Over>,
    }
    let mut root = Over::default();
    for VariantSelection {
        prim,
        set,
        selection,
    } in variants
    {
        let mut at = &mut root;
        let names: Vec<&str> = prim.trim_start_matches('/').split('/').collect();
        if prim.is_empty() || !prim.starts_with('/') || names.iter().any(|n| n.is_empty()) {
            return Err(format!(
                "variant prim {prim:?} is not an absolute prim path"
            ));
        }
        for n in names {
            at = at.children.entry(n.to_string()).or_default();
        }
        at.selections.push((set.clone(), selection.clone()));
    }
    fn write(out: &mut String, name: &str, o: &Over, depth: usize) {
        let pad = "    ".repeat(depth);
        out.push_str(&format!("{pad}over \"{name}\""));
        if !o.selections.is_empty() {
            out.push_str(" (\n");
            out.push_str(&format!("{pad}    variants = {{\n"));
            for (set, sel) in &o.selections {
                out.push_str(&format!("{pad}        string {set} = \"{sel}\"\n"));
            }
            out.push_str(&format!("{pad}    }}\n{pad})"));
        }
        out.push_str(&format!("\n{pad}{{\n"));
        for (n, c) in &o.children {
            write(out, n, c, depth + 1);
        }
        out.push_str(&format!("{pad}}}\n"));
    }
    let mut out = String::from("#usda 1.0\n\n");
    for (n, c) in &root.children {
        write(&mut out, n, c, 0);
    }
    Ok(out)
}

/// Every number of a value, flattened: vectors by component, quaternions as stored
/// (w, x, y, z), matrices row by row.
pub(super) fn numbers(v: &vt::Value) -> Option<Vec<f64>> {
    use openusd::gf::f16;
    use vt::Value as V;
    fn cast<T: bytemuck::Pod, S: bytemuck::Pod + Into<f64>>(v: &[T]) -> Vec<f64> {
        bytemuck::cast_slice::<T, S>(v)
            .iter()
            .map(|&s| s.into())
            .collect()
    }
    fn half<T: bytemuck::Pod>(v: &[T]) -> Vec<f64> {
        bytemuck::cast_slice::<T, f16>(v)
            .iter()
            .map(|s| f64::from(s.to_f32()))
            .collect()
    }
    let one = |x: f64| Some(vec![x]);
    match v {
        V::Bool(b) => one(f64::from(u8::from(*b))),
        V::BoolVec(b) => Some(b.iter().map(|&b| f64::from(u8::from(b))).collect()),
        V::Uchar(x) => one(f64::from(*x)),
        V::UcharVec(x) => Some(x.iter().map(|&x| f64::from(x)).collect()),
        V::Int(x) => one(f64::from(*x)),
        V::IntVec(x) => Some(x.iter().map(|&x| f64::from(x)).collect()),
        V::Uint(x) => one(f64::from(*x)),
        V::UintVec(x) => Some(x.iter().map(|&x| f64::from(x)).collect()),
        V::Int64(x) => one(*x as f64),
        V::Int64Vec(x) => Some(x.iter().map(|&x| x as f64).collect()),
        V::Uint64(x) => one(*x as f64),
        V::Uint64Vec(x) => Some(x.iter().map(|&x| x as f64).collect()),
        V::Half(x) => one(f64::from(x.to_f32())),
        V::HalfVec(x) => Some(half(x)),
        V::Float(x) => one(f64::from(*x)),
        V::FloatVec(x) => Some(x.iter().map(|&x| f64::from(x)).collect()),
        V::Double(x) => one(*x),
        V::DoubleVec(x) => Some(x.clone()),
        V::Vec2f(x) => Some(cast::<_, f32>(std::slice::from_ref(x))),
        V::Vec3f(x) => Some(cast::<_, f32>(std::slice::from_ref(x))),
        V::Vec4f(x) => Some(cast::<_, f32>(std::slice::from_ref(x))),
        V::Vec2d(x) => Some(cast::<_, f64>(std::slice::from_ref(x))),
        V::Vec3d(x) => Some(cast::<_, f64>(std::slice::from_ref(x))),
        V::Vec4d(x) => Some(cast::<_, f64>(std::slice::from_ref(x))),
        V::Vec2i(x) => Some(cast::<_, i32>(std::slice::from_ref(x))),
        V::Vec3i(x) => Some(cast::<_, i32>(std::slice::from_ref(x))),
        V::Vec4i(x) => Some(cast::<_, i32>(std::slice::from_ref(x))),
        V::Vec2h(x) => Some(half(std::slice::from_ref(x))),
        V::Vec3h(x) => Some(half(std::slice::from_ref(x))),
        V::Vec4h(x) => Some(half(std::slice::from_ref(x))),
        V::Vec2fVec(x) => Some(cast::<_, f32>(x)),
        V::Vec3fVec(x) => Some(cast::<_, f32>(x)),
        V::Vec4fVec(x) => Some(cast::<_, f32>(x)),
        V::Vec2dVec(x) => Some(cast::<_, f64>(x)),
        V::Vec3dVec(x) => Some(cast::<_, f64>(x)),
        V::Vec4dVec(x) => Some(cast::<_, f64>(x)),
        V::Vec2iVec(x) => Some(cast::<_, i32>(x)),
        V::Vec3iVec(x) => Some(cast::<_, i32>(x)),
        V::Vec4iVec(x) => Some(cast::<_, i32>(x)),
        V::Vec2hVec(x) => Some(half(x)),
        V::Vec3hVec(x) => Some(half(x)),
        V::Vec4hVec(x) => Some(half(x)),
        V::Quatf(x) => Some(cast::<_, f32>(std::slice::from_ref(x))),
        V::Quatd(x) => Some(cast::<_, f64>(std::slice::from_ref(x))),
        V::Quath(x) => Some(half(std::slice::from_ref(x))),
        V::QuatfVec(x) => Some(cast::<_, f32>(x)),
        V::QuatdVec(x) => Some(cast::<_, f64>(x)),
        V::QuathVec(x) => Some(half(x)),
        V::Matrix2d(m) => Some(m.0.to_vec()),
        V::Matrix3d(m) => Some(m.0.to_vec()),
        V::Matrix4d(m) => Some(m.0.to_vec()),
        V::Matrix2dVec(m) => Some(m.iter().flat_map(|m| m.0).collect()),
        V::Matrix3dVec(m) => Some(m.iter().flat_map(|m| m.0).collect()),
        V::Matrix4dVec(m) => Some(m.iter().flat_map(|m| m.0).collect()),
        _ => None,
    }
}

/// Every integer of a value, as stored: an id past 2^53 does not survive an `f64`.
pub(super) fn integers(v: &vt::Value) -> Option<Vec<i64>> {
    use vt::Value as V;
    Some(match v {
        V::Uchar(x) => vec![i64::from(*x)],
        V::UcharVec(x) => x.iter().map(|&x| i64::from(x)).collect(),
        V::Int(x) => vec![i64::from(*x)],
        V::IntVec(x) => x.iter().map(|&x| i64::from(x)).collect(),
        V::Uint(x) => vec![i64::from(*x)],
        V::UintVec(x) => x.iter().map(|&x| i64::from(x)).collect(),
        V::Int64(x) => vec![*x],
        V::Int64Vec(x) => x.clone(),
        V::Uint64(x) => vec![*x as i64],
        V::Uint64Vec(x) => x.iter().map(|&x| x as i64).collect(),
        _ => return None,
    })
}

/// The numbers in one element of an array value; `None` for a value that is no array.
fn element_width(v: &vt::Value) -> Option<usize> {
    use vt::Value as V;
    match v {
        V::BoolVec(_)
        | V::UcharVec(_)
        | V::IntVec(_)
        | V::UintVec(_)
        | V::Int64Vec(_)
        | V::Uint64Vec(_)
        | V::HalfVec(_)
        | V::FloatVec(_)
        | V::DoubleVec(_) => Some(1),
        V::Vec2fVec(_) | V::Vec2dVec(_) | V::Vec2iVec(_) | V::Vec2hVec(_) => Some(2),
        V::Vec3fVec(_) | V::Vec3dVec(_) | V::Vec3iVec(_) | V::Vec3hVec(_) => Some(3),
        V::Vec4fVec(_)
        | V::Vec4dVec(_)
        | V::Vec4iVec(_)
        | V::Vec4hVec(_)
        | V::QuatfVec(_)
        | V::QuatdVec(_)
        | V::QuathVec(_)
        | V::Matrix2dVec(_) => Some(4),
        V::Matrix3dVec(_) => Some(9),
        V::Matrix4dVec(_) => Some(16),
        _ => None,
    }
}

/// A USD matrix (row vectors, translation in the last row) as a column-vector transform.
pub(super) fn matrix(rows: &[f64]) -> DMat4 {
    DMat4::from_cols_slice(&rows[..16])
}

/// A quaternion stored (w, x, y, z).
pub(super) fn quat(wxyz: &[f64]) -> DQuat {
    DQuat::from_xyzw(wxyz[1], wxyz[2], wxyz[3], wxyz[0])
}

/// `rgb` in `space` as linear Rec.709; a space that names no conversion leaves it as it stands.
pub(super) fn to_rec709(space: ImageColor, rgb: DVec3) -> DVec3 {
    match space {
        ImageColor::Named(s) => DVec3::from(s.to_rec709(rgb.as_vec3().to_array()).map(f64::from)),
        _ => rgb,
    }
}

/// Reads attributes of one prim at one time.
pub(super) struct PrimReader<'a> {
    pub(super) prim: &'a usd::Prim,
    pub(super) time: usd::TimeCode,
}

impl PrimReader<'_> {
    pub(super) fn path(&self) -> &str {
        self.prim.path().as_str()
    }

    pub(super) fn value(&self, name: &str) -> Result<Option<vt::Value>, String> {
        self.prim
            .attribute(name)
            .get_at::<vt::Value>(self.time)
            .map_err(|e| format!("{} {name}: {e}", self.path()))
    }

    pub(super) fn numbers(&self, name: &str) -> Result<Option<Vec<f64>>, String> {
        match self.value(name)? {
            None => Ok(None),
            Some(v) => numbers(&v)
                .map(Some)
                .ok_or_else(|| format!("{} {name}: not numbers", self.path())),
        }
    }

    pub(super) fn token(&self, name: &str) -> Result<Option<String>, String> {
        Ok(match self.value(name)? {
            Some(vt::Value::Token(t)) => Some(t.as_str().to_string()),
            Some(vt::Value::String(s)) => Some(s),
            _ => None,
        })
    }

    /// The color space `UsdColorSpaceAPI` resolves for color attribute `name`: its own
    /// `colorSpace`, else its prim's or the nearest ancestor's `colorSpace:name`, else linear
    /// Rec.709 (`pxr/usd/usd/schema.usda`).
    pub(super) fn color_space(&self, name: &str) -> Result<String, String> {
        let attr = self.prim.attribute(name);
        let space = usd::ColorSpaceAPI::compute_attribute_name(&attr, None)
            .map_err(|e| format!("{} {name}: {e}", self.path()))?;
        Ok(space.map_or_else(
            || gf::color_space_names::LINEAR_REC709.to_string(),
            |s| s.as_str().to_string(),
        ))
    }

    /// `rgb`, a value of color attribute `name` in the space [`Self::color_space`] resolves, as
    /// linear Rec.709. A space that names no conversion ("data", "raw", "identity") leaves it as
    /// it stands, and one the cook doesn't know fails.
    pub(super) fn color(&self, name: &str, rgb: DVec3) -> Result<DVec3, String> {
        let space = self.color_space(name)?;
        ImageColor::named(&space)
            .map(|c| to_rec709(c, rgb))
            .ok_or_else(|| {
                format!(
                    "{} {name}: color space {space} is not one the cook converts",
                    self.path()
                )
            })
    }

    /// This reader at `time` instead.
    pub(super) fn at(&self, time: f64) -> PrimReader<'_> {
        PrimReader {
            prim: self.prim,
            time: usd::TimeCode::new(time),
        }
    }

    /// Whether attribute `name` holds time samples of more than one value.
    pub(super) fn varies(&self, name: &str) -> Result<bool, String> {
        let times = self.prim.attribute(name).time_sample_times();
        let times = times.map_err(|e| format!("{} {name}: {e}", self.path()))?;
        let Some((&first, rest)) = times.split_first() else {
            return Ok(false);
        };
        let value = self.at(first).value(name)?;
        for &t in rest {
            if self.at(t).value(name)? != value {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// A prim's transform stack (`UsdGeomXformable`), read once and evaluated at any time. A prim
/// that is not `Xformable` has an empty one.
#[derive(Clone, Default)]
pub(super) struct Stack {
    /// The prim's path, which errors name.
    path: String,
    query: geom::XformQuery,
    /// The times its ops are sampled at, in order; none where each op holds one value. Values
    /// tell that and matrices can't: a full turn ends where it began.
    pub(super) times: Vec<f64>,
}

/// Whether an op of `kind` takes `value`, as `xformOp.h` types them: a double, float or half
/// for one axis, a vector of three for `translate`, `scale` and the three-axis rotations, a
/// quaternion for `orient` and a `matrix4d` for `transform`.
fn op_takes(kind: geom::XformOpKind, value: &vt::Value) -> bool {
    use geom::XformOpKind as K;
    use vt::Value as V;
    match kind {
        K::Transform => matches!(value, V::Matrix4d(_)),
        K::Orient => matches!(value, V::Quatd(_) | V::Quatf(_) | V::Quath(_)),
        K::TranslateX
        | K::TranslateY
        | K::TranslateZ
        | K::ScaleX
        | K::ScaleY
        | K::ScaleZ
        | K::RotateX
        | K::RotateY
        | K::RotateZ => matches!(value, V::Double(_) | V::Float(_) | V::Half(_)),
        K::Translate
        | K::Scale
        | K::RotateXYZ
        | K::RotateXZY
        | K::RotateYXZ
        | K::RotateYZX
        | K::RotateZXY
        | K::RotateZYX => matches!(value, V::Vec3d(_) | V::Vec3f(_) | V::Vec3h(_)),
    }
}

impl Stack {
    pub(super) fn read(prim: &usd::Prim) -> Result<Self, String> {
        let path = prim.path().as_str();
        let fail = |e: openusd::Error| format!("{path}: {e}");
        let Some(xformable) = geom::Xformable::from_prim(prim.clone()).map_err(fail)? else {
            return Ok(Self {
                path: path.to_string(),
                ..Self::default()
            });
        };
        let (ops, reset) = xformable.ordered_xform_ops().map_err(fail)?;
        let mut times = Vec::new();
        let mut moves = false;
        for op in &ops {
            let name = op.name();
            let Some(kind) = op.kind() else {
                return Err(format!("{path}: {name} is not an xform op"));
            };
            let fail = |e: openusd::Error| format!("{path} {name}: {e}");
            let value = |t: Option<f64>| {
                let v = op.query().get_at::<vt::Value>(t.map(usd::TimeCode::new));
                let v = v.map_err(fail)?;
                match &v {
                    Some(v) if !op_takes(kind, v) => Err(format!(
                        "{path}: xform op {} holds a value it does not take",
                        kind.as_token()
                    )),
                    _ => Ok(v),
                }
            };
            let held = op.query().time_sample_times().map_err(fail)?;
            match held.split_first() {
                Some((&first, rest)) => {
                    let first = value(Some(first))?;
                    for &t in rest {
                        moves = moves || value(Some(t))? != first;
                    }
                }
                None => {
                    value(None)?;
                }
            }
            times.extend(held);
        }
        if moves {
            times.sort_by(f64::total_cmp);
            times.dedup();
        } else {
            times.clear();
        }
        Ok(Self {
            path: path.to_string(),
            query: geom::XformQuery::from_ops(ops, reset),
            times,
        })
    }

    /// The local transform at `time`.
    pub(super) fn local(&self, time: usd::TimeCode) -> Result<DMat4, String> {
        let m = self.query.local_transform(time);
        let m = m.map_err(|e| format!("{}: {e}", self.path))?;
        Ok(matrix(&m.0))
    }

    /// Whether the local transform starts from the stage, not the parent.
    pub(super) fn resets(&self) -> bool {
        self.query.resets_xform_stack()
    }
}

/// A primvar's elements (of `width` numbers each), its interpolation and indices.
pub(super) struct Primvar {
    values: Vec<f64>,
    width: usize,
    pub(super) interp: Interpolation,
    pub(super) indices: Option<Vec<i32>>,
}

impl Primvar {
    /// Element `i` of the primvar's domain (point, face or face-vertex, by interpolation).
    fn get(&self, i: usize) -> Option<&[f64]> {
        let i = match &self.indices {
            Some(ix) => usize::try_from(*ix.get(i)?).ok()?,
            None => i,
        };
        self.values.get(i * self.width..(i + 1) * self.width)
    }

    /// The numbers of each element.
    pub(super) fn width(&self) -> usize {
        self.width
    }

    /// How many of the `domain` elements hold no value.
    pub(super) fn missing(&self, domain: usize) -> usize {
        (0..domain).filter(|&i| self.get(i).is_none()).count()
    }

    /// The element for a corner: `point`, `face` and `corner` index into each domain.
    pub(super) fn at(&self, point: usize, face: usize, corner: usize) -> Option<&[f64]> {
        self.get(match self.interp {
            Interpolation::Constant => 0,
            Interpolation::Uniform => face,
            Interpolation::Vertex | Interpolation::Varying => point,
            Interpolation::FaceVarying => corner,
        })
    }
}

impl PrimReader<'_> {
    /// Primvar `name` (`primvars:name`) of the prim; `None` for a name no primvar has, which
    /// holds no value, as `GetPrimvar` gives an invalid primvar there.
    fn geom_primvar(&self, name: &str) -> Result<Option<geom::Primvar>, String> {
        let api = geom::PrimvarsAPI::from_prim_unchecked(self.prim.clone());
        match api.primvar(name) {
            Ok(p) => Ok(Some(p)),
            Err(openusd_schemas::SchemaError::InvalidPrimvarName { .. }) => Ok(None),
            Err(e) => Err(format!("{} primvar {name}: {e}", self.path())),
        }
    }

    /// Primvar `name` (`primvars:name`), or a plain attribute `fallback` with vertex
    /// interpolation; `None` when neither holds values.
    pub(super) fn primvar(
        &self,
        name: &str,
        width: usize,
        fallback: Option<&str>,
    ) -> Result<Option<Primvar>, String> {
        let Some((primvar, attr)) = self.elements(name, width, fallback)? else {
            return Ok(None);
        };
        if !primvar.values.len().is_multiple_of(width) {
            return Err(format!("{} {attr}: not {width}-wide", self.path()));
        }
        Ok(Some(primvar))
    }

    /// [`Self::primvar`] and the attribute that holds it, whatever its last element falls short
    /// of.
    fn elements(
        &self,
        name: &str,
        width: usize,
        fallback: Option<&str>,
    ) -> Result<Option<(Primvar, String)>, String> {
        let Some(primvar) = self.geom_primvar(name)? else {
            return Ok(None);
        };
        let full = primvar.attribute().name().to_string();
        let held = |attr: &str| Ok::<_, String>(self.numbers(attr)?.filter(|v| !v.is_empty()));
        let fail = |attr: &str, e: openusd::Error| format!("{} {attr}: {e}", self.path());
        if let Some(values) = held(&full)? {
            let interp = primvar.interpolation().map_err(|e| fail(&full, e))?;
            let indices = primvar.indices(self.time).map_err(|e| fail(&full, e))?;
            let primvar = Primvar {
                values,
                width,
                interp,
                indices,
            };
            return Ok(Some((primvar, full)));
        }
        let Some(f) = fallback else {
            return Ok(None);
        };
        let Some(values) = held(f)? else {
            return Ok(None);
        };
        // The plain attribute's interpolation falls back to vertex.
        let interp = self
            .prim
            .attribute(f)
            .metadata::<Interpolation>(geom::tokens::INTERPOLATION)
            .map_err(|e| fail(f, e))?
            .unwrap_or(Interpolation::Vertex);
        let primvar = Primvar {
            values,
            width,
            interp,
            indices: None,
        };
        Ok(Some((primvar, f.to_string())))
    }

    /// Primvar `name` of tangent frame vectors, as wide as its type declares: three components, or
    /// four where the last is a bitangent sign.
    pub(super) fn frame_primvar(&self, name: &str) -> Result<Option<Primvar>, String> {
        let Some(primvar) = self.geom_primvar(name)? else {
            return Ok(None);
        };
        let full = primvar.attribute().name();
        let Some(value) = self.value(full)? else {
            return Ok(None);
        };
        match element_width(&value) {
            Some(w) => match w * self.element_size(&primvar)? {
                width @ (3 | 4) => self.primvar(name, width, None),
                width => Err(format!(
                    "{} {full}: {width} components, not a vector of three or four",
                    self.path()
                )),
            },
            None => self.primvar(name, 3, None),
        }
    }

    /// Primvar `name` as its elements stand, each as wide as its type and `elementSize` make it.
    /// A last element the array falls short of holds no value.
    pub(super) fn wide_primvar(&self, name: &str) -> Result<Option<Primvar>, String> {
        let Some(primvar) = self.geom_primvar(name)? else {
            return Ok(None);
        };
        let Some(value) = self.value(primvar.attribute().name())? else {
            return Ok(None);
        };
        let Some(width) = self.stride(&primvar, &value)? else {
            return Ok(None);
        };
        Ok(self.elements(name, width, None)?.map(|(p, _)| p))
    }

    /// The numbers of one element of `primvar` holding `value`, if it holds numbers.
    fn stride(&self, primvar: &geom::Primvar, value: &vt::Value) -> Result<Option<usize>, String> {
        let Some(width) = element_width(value).or_else(|| numbers(value).map(|v| v.len())) else {
            return Ok(None);
        };
        Ok(Some(width * self.element_size(primvar)?))
    }

    /// Whether primvar `name` holds values.
    pub(super) fn holds_primvar(&self, name: &str) -> Result<bool, String> {
        let Some(primvar) = self.geom_primvar(name)? else {
            return Ok(false);
        };
        let held = self.numbers(primvar.attribute().name())?;
        Ok(held.is_some_and(|v| !v.is_empty()))
    }

    /// `primvar`'s `elementSize`: how many array elements make one of its values.
    fn element_size(&self, primvar: &geom::Primvar) -> Result<usize, String> {
        let name = primvar.attribute().name();
        let size = primvar
            .element_size()
            .map_err(|e| format!("{} {name}: {e}", self.path()))?;
        usize::try_from(size)
            .ok()
            .filter(|&s| s > 0)
            .ok_or_else(|| format!("{} {name}: elementSize {size}", self.path()))
    }

    /// Primvar `name` as a value for the whole prim, whatever the width of its elements. One
    /// that holds no numbers, or whose index misses its values, gives a reader nothing. The stream
    /// makes room for `reach` times its largest value.
    pub(super) fn whole_primvar(&self, name: &str, reach: f64) -> Result<shading::Primvar, String> {
        let Some(primvar) = self.geom_primvar(name)? else {
            return Ok(shading::Primvar::Missing);
        };
        let full = primvar.attribute().name();
        let Some(value) = self.value(full)? else {
            return Ok(shading::Primvar::Missing);
        };
        let Some(values) = numbers(&value).filter(|v| !v.is_empty()) else {
            return Ok(shading::Primvar::Missing);
        };
        let width = self.stride(&primvar, &value)?.unwrap_or(values.len());
        let fail = |e: openusd::Error| format!("{} {full}: {e}", self.path());
        if primvar.interpolation().map_err(fail)? != Interpolation::Constant {
            let channels = width.min(4);
            let peak = values
                .chunks(width)
                .flat_map(|v| v.iter().take(channels))
                .fold(0.0, |p: f64, x| p.max(x.abs()));
            return Ok(shading::Primvar::Varying {
                width: channels as u8,
                scale: shading::stream_scale(reach * peak),
            });
        }
        let primvar = Primvar {
            values,
            width,
            interp: Interpolation::Constant,
            indices: primvar.indices(self.time).map_err(fail)?,
        };
        Ok(match primvar.at(0, 0, 0) {
            Some(v) => shading::Primvar::Constant(v.iter().map(|&x| x as f32).collect()),
            None => shading::Primvar::Missing,
        })
    }
}
