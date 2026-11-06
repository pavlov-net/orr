//! Reading a composed stage: the resolver that records every file the stage reads, value
//! conversions, transforms and primvars.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use openusd::ar::Resolver;
use openusd::{ar, sdf, usd};
use orr_math::{DMat4, DQuat, DVec3};
use orr_pack::Hash32;

use super::VariantSelection;
use crate::shading;
use crate::state::{Input, hash_path};
use crate::texture::tiles::{UDIM, UDIM_TILES};

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

/// The default resolver, recording every file it serves and serving the session layer.
pub(super) struct Recorder {
    inner: ar::DefaultResolver,
    reads: Rc<RefCell<Reads>>,
}

impl Recorder {
    pub(super) fn new(reads: Rc<RefCell<Reads>>) -> Self {
        Self {
            inner: ar::DefaultResolver::new(),
            reads,
        }
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

    /// The identifier of an asset path holding `<UDIM>`, after finding each of its tiles the way
    /// a path of its own would resolve.
    fn tiled_identifier(&self, asset_path: &str, anchor: Option<&ar::ResolvedPath>) -> String {
        let id = self.beside(asset_path, anchor);
        if self.reads.borrow().tiles.contains_key(&id) {
            return id;
        }
        let mut found = Vec::new();
        for n in UDIM_TILES {
            let tile = asset_path.replace(UDIM, &n.to_string());
            match self
                .inner
                .resolve(&self.inner.create_identifier(&tile, anchor))
            {
                Some(r) => found.push((n, r.to_string_lossy().into_owned())),
                None => {
                    let file = outer(Path::new(&self.beside(&tile, anchor)));
                    self.reads.borrow_mut().absent.insert(file);
                }
            }
        }
        self.reads.borrow_mut().tiles.insert(id.clone(), found);
        id
    }

    /// Reads an asset (an image) by its resolved path, recording its file, which is watched for
    /// a change when it can't be read.
    pub(super) fn read(&self, resolved: &str) -> io::Result<Vec<u8>> {
        let read = self
            .open_asset(&ar::ResolvedPath::new(resolved))
            .and_then(|mut asset| asset.read_all());
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
        if asset_path.contains(UDIM) {
            return self.tiled_identifier(asset_path, anchor);
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
        let bytes = self.inner.open_asset(resolved)?.read_all()?;
        let whole = file.as_path() == &**resolved;
        if let std::collections::btree_map::Entry::Vacant(e) =
            self.reads.borrow_mut().files.entry(file)
        {
            let hash = if whole {
                Hash32::of(&bytes)
            } else {
                Hash32::of(&std::fs::read(e.key())?)
            };
            e.insert(hash);
        }
        Ok(Box::new(io::Cursor::new(bytes)))
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
pub(super) fn numbers(v: &sdf::Value) -> Option<Vec<f64>> {
    use openusd::gf::f16;
    use sdf::Value as V;
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
        V::Matrix4d(m) => Some(m.0.to_vec()),
        V::Matrix4dVec(m) => Some(m.iter().flat_map(|m| m.0).collect()),
        _ => None,
    }
}

/// The numbers in one element of an array value; `None` for a value that is no array.
fn element_width(v: &sdf::Value) -> Option<usize> {
    use sdf::Value as V;
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
        | V::QuathVec(_) => Some(4),
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

/// Reads attributes of one prim at one time.
pub(super) struct PrimReader<'a> {
    pub(super) prim: &'a usd::Prim,
    pub(super) time: usd::TimeCode,
}

impl PrimReader<'_> {
    pub(super) fn path(&self) -> &str {
        self.prim.path().as_str()
    }

    pub(super) fn value(&self, name: &str) -> Result<Option<sdf::Value>, String> {
        self.prim
            .attribute(name)
            .get_at::<sdf::Value>(self.time)
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

    pub(super) fn ints(&self, name: &str) -> Result<Option<Vec<i64>>, String> {
        Ok(self
            .numbers(name)?
            .map(|v| v.into_iter().map(|x| x as i64).collect()))
    }

    pub(super) fn token(&self, name: &str) -> Result<Option<String>, String> {
        Ok(match self.value(name)? {
            Some(sdf::Value::Token(t)) => Some(t.as_str().to_string()),
            Some(sdf::Value::String(s)) => Some(s),
            _ => None,
        })
    }

    pub(super) fn tokens(&self, name: &str) -> Result<Vec<String>, String> {
        Ok(match self.value(name)? {
            Some(sdf::Value::TokenVec(t)) => t.iter().map(|t| t.as_str().to_string()).collect(),
            Some(sdf::Value::StringVec(s)) => s,
            _ => Vec::new(),
        })
    }

    pub(super) fn metadata_token(&self, name: &str, key: &str) -> Option<String> {
        let a = self.prim.attribute(name);
        a.get_metadata::<sdf::Value>(key)
            .ok()
            .flatten()
            .and_then(|v| match v {
                sdf::Value::Token(t) => Some(t.as_str().to_string()),
                sdf::Value::String(s) => Some(s),
                _ => None,
            })
    }

    /// This reader at `time` instead.
    pub(super) fn at(&self, time: f64) -> PrimReader<'_> {
        PrimReader {
            prim: self.prim,
            time: usd::TimeCode::new(time),
        }
    }

    pub(super) fn targets(&self, name: &str) -> Result<Vec<sdf::Path>, String> {
        self.prim
            .relationship(name)
            .targets()
            .map_err(|e| format!("{} {name}: {e}", self.path()))
    }

    /// The prim's local transform from its `xformOpOrder`, and whether it resets the
    /// parent's.
    pub(super) fn local_transform(&self) -> Result<(DMat4, bool), String> {
        let order = self.tokens("xformOpOrder")?;
        let mut m = DMat4::IDENTITY;
        let mut reset = false;
        for op in &order {
            if op == "!resetXformStack!" {
                reset = true;
                m = DMat4::IDENTITY;
                continue;
            }
            let (invert, name) = match op.strip_prefix("!invert!") {
                Some(n) => (true, n),
                None => (false, op.as_str()),
            };
            let Some(v) = self.numbers(name)? else {
                continue;
            };
            let kind = name
                .strip_prefix("xformOp:")
                .and_then(|k| k.split(':').next())
                .ok_or_else(|| format!("{}: {name} is not an xform op", self.path()))?;
            let deg = |x: f64| x.to_radians();
            let axis = |a: char, x: f64| match a {
                'X' => DMat4::from_rotation_x(deg(x)),
                'Y' => DMat4::from_rotation_y(deg(x)),
                _ => DMat4::from_rotation_z(deg(x)),
            };
            let op_m = match kind {
                "translate" if v.len() == 3 => DMat4::from_translation(DVec3::from_slice(&v)),
                "scale" if v.len() == 1 => DMat4::from_scale(DVec3::splat(v[0])),
                "scale" if v.len() == 3 => DMat4::from_scale(DVec3::from_slice(&v)),
                "translateX" if v.len() == 1 => DMat4::from_translation(DVec3::X * v[0]),
                "translateY" if v.len() == 1 => DMat4::from_translation(DVec3::Y * v[0]),
                "translateZ" if v.len() == 1 => DMat4::from_translation(DVec3::Z * v[0]),
                "scaleX" if v.len() == 1 => DMat4::from_scale(DVec3::new(v[0], 1.0, 1.0)),
                "scaleY" if v.len() == 1 => DMat4::from_scale(DVec3::new(1.0, v[0], 1.0)),
                "scaleZ" if v.len() == 1 => DMat4::from_scale(DVec3::new(1.0, 1.0, v[0])),
                "rotateX" if v.len() == 1 => axis('X', v[0]),
                "rotateY" if v.len() == 1 => axis('Y', v[0]),
                "rotateZ" if v.len() == 1 => axis('Z', v[0]),
                "orient" if v.len() == 4 => DMat4::from_quat(quat(&v).normalize()),
                "transform" if v.len() == 16 => matrix(&v),
                "rotateXYZ" | "rotateXZY" | "rotateYXZ" | "rotateYZX" | "rotateZXY"
                | "rotateZYX"
                    if v.len() == 3 =>
                {
                    // rotateZYX applies Z first, and its value is (x, y, z) whatever the order.
                    kind[6..].chars().fold(DMat4::IDENTITY, |r, a| {
                        axis(a, v[usize::from(a as u8 - b'X')]) * r
                    })
                }
                k => return Err(format!("{}: xform op {k}", self.path())),
            };
            m *= if invert { op_m.inverse() } else { op_m };
        }
        Ok((m, reset))
    }
}

/// Interpolation of a primvar's values over a mesh.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum Interp {
    Constant,
    Uniform,
    Vertex,
    FaceVarying,
}

/// A primvar's elements (of `width` numbers each), its interpolation and indices.
pub(super) struct Primvar {
    values: Vec<f64>,
    width: usize,
    pub(super) interp: Interp,
    pub(super) indices: Option<Vec<i64>>,
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

    /// How many of the `domain` elements hold no value.
    pub(super) fn missing(&self, domain: usize) -> usize {
        (0..domain).filter(|&i| self.get(i).is_none()).count()
    }

    /// The element for a corner: `point`, `face` and `corner` index into each domain.
    pub(super) fn at(&self, point: usize, face: usize, corner: usize) -> Option<&[f64]> {
        self.get(match self.interp {
            Interp::Constant => 0,
            Interp::Uniform => face,
            Interp::Vertex => point,
            Interp::FaceVarying => corner,
        })
    }
}

impl PrimReader<'_> {
    /// Primvar `name` (`primvars:name`), or a plain attribute `fallback` with vertex
    /// interpolation; `None` when neither holds values.
    pub(super) fn primvar(
        &self,
        name: &str,
        width: usize,
        fallback: Option<&str>,
    ) -> Result<Option<Primvar>, String> {
        let full = format!("primvars:{name}");
        let held = |attr: &str| Ok::<_, String>(self.numbers(attr)?.filter(|v| !v.is_empty()));
        let (values, attr) = match (held(&full)?, fallback) {
            (Some(v), _) => (v, full.as_str()),
            (None, Some(f)) => match held(f)? {
                Some(v) => (v, f),
                None => return Ok(None),
            },
            (None, None) => return Ok(None),
        };
        let interp = match self.metadata_token(attr, "interpolation").as_deref() {
            Some("constant") => Interp::Constant,
            Some("uniform") => Interp::Uniform,
            Some("faceVarying") => Interp::FaceVarying,
            Some("vertex" | "varying") => Interp::Vertex,
            // A primvar unauthored is constant; the plain attribute has vertex as its fallback.
            None if attr == full => Interp::Constant,
            None => Interp::Vertex,
            Some(i) => return Err(format!("{} {attr}: interpolation {i}", self.path())),
        };
        let indices = if attr == full {
            self.ints(&format!("{full}:indices"))?
        } else {
            None
        };
        if !values.len().is_multiple_of(width) {
            return Err(format!("{} {attr}: not {width}-wide", self.path()));
        }
        Ok(Some(Primvar {
            values,
            width,
            interp,
            indices,
        }))
    }

    /// Primvar `name` of tangent frame vectors, as wide as its type declares: three components, or
    /// four where the last is a bitangent sign.
    pub(super) fn frame_primvar(&self, name: &str) -> Result<Option<Primvar>, String> {
        let full = format!("primvars:{name}");
        let Some(value) = self.value(&full)? else {
            return Ok(None);
        };
        match element_width(&value).map(|w| w * self.element_size(&full)) {
            Some(width @ (3 | 4)) => self.primvar(name, width, None),
            Some(width) => Err(format!(
                "{} {full}: {width} components, not a vector of three or four",
                self.path()
            )),
            None => self.primvar(name, 3, None),
        }
    }

    /// Whether primvar `name` holds values.
    pub(super) fn holds_primvar(&self, name: &str) -> Result<bool, String> {
        let held = self.numbers(&format!("primvars:{name}"))?;
        Ok(held.is_some_and(|v| !v.is_empty()))
    }

    /// Attribute `name`'s `elementSize`: how many array elements make one of its values.
    pub(super) fn element_size(&self, name: &str) -> usize {
        self.prim
            .attribute(name)
            .get_metadata::<sdf::Value>("elementSize")
            .ok()
            .flatten()
            .and_then(|v| numbers(&v))
            .map_or(1, |v| v[0] as usize)
            .max(1)
    }

    /// Primvar `name` as a value for the whole prim, whatever the width of its elements. One
    /// that holds no numbers, or whose index misses its values, gives a reader nothing.
    pub(super) fn whole_primvar(&self, name: &str) -> Result<shading::Primvar, String> {
        let full = format!("primvars:{name}");
        let Some(value) = self.value(&full)? else {
            return Ok(shading::Primvar::Missing);
        };
        let Some(values) = numbers(&value).filter(|v| !v.is_empty()) else {
            return Ok(shading::Primvar::Missing);
        };
        let width = element_width(&value).unwrap_or(values.len()) * self.element_size(&full);
        if self
            .metadata_token(&full, "interpolation")
            .is_some_and(|i| i != "constant")
        {
            return Ok(shading::Primvar::Varying(width.min(4) as u8));
        }
        let primvar = Primvar {
            values,
            width,
            interp: Interp::Constant,
            indices: self.ints(&format!("{full}:indices"))?,
        };
        Ok(match primvar.at(0, 0, 0) {
            Some(v) => shading::Primvar::Constant(v.iter().map(|&x| x as f32).collect()),
            None => shading::Primvar::Missing,
        })
    }
}
