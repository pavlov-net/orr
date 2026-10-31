//! The cook run: sources become units (one cook job each, usually one entry), units link
//! from the archive being replaced or the cache when their keys match, the rest cook on the
//! cook's pool, and every archive is written in a fixed order.
//!
//! Scheduling: jobs start largest first while the estimated working set of the running
//! jobs stays under [`CookOptions::working_set`] (a job over it runs alone). Finished units
//! wait for their turn to be written; while the bytes waiting exceed
//! [`CookOptions::reorder_window`], only the jobs the next unwritten units wait on start.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::Instant;

use glam::{Vec2, Vec3, Vec4};

use orr_pack::material::{Material, slot};
use orr_pack::model::{ModelHeader, ModelInfo};
use orr_pack::texture::TextureInfo;
use orr_pack::{
    Aabb, Archive, ArchiveWriter, AssetId, BlobIn, Class, GDeflate, Hash32, Span, Target,
};

use crate::assembly::{self, Assembly};
use crate::cache::Cache;
use crate::import::Scene;
use crate::instances::{self, Catalog, ModelFacts};
use crate::key::{self, Key, ZSTD_LEVEL};
use crate::mesh::maps::{Displacement, Opacity, Raster};
use crate::mesh::{
    self, DisplacementSpec, MapSpec, MeshData, MeshInput, OpacitySpec, PartData, Pivots,
    SectionData, SectionMaps,
};
use crate::raster::{self, RasterReport};
use crate::report::{
    self, ArchiveReport, AssemblyReport, AssemblyUnitReport, AssetReport, Outcome,
};
use crate::state::{self, Input, Memo, Record};
use crate::subd::{self, CageData, CageInput, SubdSettings};
use crate::texture::material::{
    ImageFacts, ImageFile, MaterialTextures, Packer, TexRef, TextureJob, TextureRules, image_facts,
};
use crate::texture::neural::{self, NeuralJob};
use crate::texture::{self, ENCODERS, Wrap};
use crate::{
    CookOptions, Cooked, Error, GltfSettings, Instances, Manifest, MeshSettings, Placement,
    PrototypeSpec, Result, Source, TextureSettings, UsdSettings, mount,
};

/// Peak bytes of a subdivision cage cook per cage face, an estimate.
const SUBD_BYTES_PER_FACE: u64 = 16 << 10;

/// Peak bytes of a mesh cook per source triangle (P2's clodBuild process peak, an upper
/// bound).
const MESH_BYTES_PER_TRIANGLE: u64 = 480;

pub(crate) fn cook(manifest: &Manifest, opts: &CookOptions) -> Result<Cooked> {
    let start = Instant::now();
    crate::alloc::install_meshopt_arena();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.jobs)
        .build()
        .map_err(|e| orr_pack::Error::from(std::io::Error::other(e)))?;
    let shared = Shared {
        memo: Memo::default(),
        cache: opts.cache.as_deref().map(Cache::new).transpose()?,
    };
    let threads = pool.current_num_threads();
    let mut run = Run::new(manifest, opts, &shared)?;
    pool.in_place_scope(|scope| run.run(scope, threads))?;
    let mut cooked = run.finish()?;
    cooked.report.seconds = start.elapsed().as_secs_f64();
    if let Some(path) = &opts.report {
        let json = serde_json::to_vec_pretty(&cooked.report).expect("report serialises");
        std::fs::write(path, json).map_err(|source| Error::Write {
            path: path.clone(),
            source,
        })?;
    }
    Ok(cooked)
}

/// What jobs read.
struct Shared {
    memo: Memo,
    cache: Option<Cache>,
}

struct Unit<'m> {
    archive: usize,
    asset: usize,
    /// Entry name; the source name for instance pages.
    name: String,
    class: u16,
    key: Hash32,
    work: Option<Work<'m>>,
    /// Units this one waits for.
    deps: Vec<usize>,
    working_set: u64,
    /// Source triangles of a mesh, for the report.
    triangles: u64,
    state: State,
    /// For a done mesh or subdivision cage: its bounds and what its model's instance pages
    /// need.
    mesh: Option<(Aabb, ModelFacts)>,
}

enum Work<'m> {
    Blobs {
        schema: u16,
        blobs: Vec<(Vec<u8>, bool)>,
    },
    /// The mesh, its settings and its sections' cooked maps, read from the texture units
    /// it waits for when it starts.
    Mesh(Arc<MeshData>, MeshSettings, Vec<SectionMaps>),
    Texture(TextureJob, Arc<Vec<ImageFile>>),
    Neural(NeuralJob, Arc<Vec<ImageFile>>),
    TextureFile(PathBuf, Hash32, TextureSettings),
    /// The cage, its section materials and its settings.
    Subd(Arc<CageData>, Vec<Material>, SubdSettings),
    /// A model of the mesh and subdivision cage units in `deps`, these materials and this
    /// skeleton.
    Model(Vec<AssetId>, AssetId),
    /// The same key as `deps[0]`, whose result it shares.
    Same,
    /// The placements, the cell size, the catalog without models, and the mesh unit of
    /// each prototype's model.
    Instances(Cow<'m, Instances>, u8, Catalog, Vec<(String, usize)>),
    Raster(raster::Plan),
}

enum State {
    Waiting,
    Running,
    Done(Done),
    /// Where its entries are, kept while a mesh that reads them has not started.
    Written(Option<From>),
}

struct Done {
    from: From,
    /// Entries in write order.
    entries: Vec<usize>,
    outcome: Outcome,
    seconds: f64,
    /// Bytes held in memory until written.
    memory: u64,
}

/// Where a unit's entries are.
#[derive(Clone)]
enum From {
    Memory(Arc<Archive>),
    Cache(PathBuf),
    /// The archive the cook replaces.
    Old,
}

/// The archive a file target replaces, with its entries by cook key.
struct Old {
    archive: Archive,
    by_key: HashMap<Hash32, Vec<usize>>,
}

/// A scene or material document to import.
struct Import<'m> {
    archive: usize,
    asset: usize,
    name: &'m str,
    path: &'m Path,
    kind: ImportKind<'m>,
    identity: Hash32,
}

#[derive(Copy, Clone, Debug)]
enum ImportKind<'m> {
    Gltf(&'m GltfSettings),
    Usd(&'m UsdSettings),
    MaterialX {
        material: &'m str,
        textures: &'m TextureRules,
    },
}

/// A unit an import made; `deps` name units of the same import.
struct NewUnit {
    name: String,
    class: u16,
    key: Hash32,
    work: Work<'static>,
    deps: Vec<String>,
    working_set: u64,
    triangles: u64,
}

impl NewUnit {
    /// A unit with no dependencies and no triangles.
    fn leaf(name: &str, class: u16, key: Hash32, working_set: u64, work: Work<'static>) -> Self {
        Self {
            name: name.into(),
            class,
            key,
            work,
            deps: Vec::new(),
            working_set,
            triangles: 0,
        }
    }
}

struct Imported {
    files: Vec<Input>,
    units: Vec<NewUnit>,
    /// Placements of the instanced models, cooked as instance unit `<asset>/instances`.
    instances: Option<Instances>,
    assemblies: Vec<Assembly>,
}

/// An instance source waiting for the keys of the models it places.
struct PendingInstances<'m> {
    archive: usize,
    asset: usize,
    name: String,
    src: Cow<'m, Instances>,
    /// The scene source record the unit's key joins: (archive, record).
    record: Option<(usize, usize)>,
}

/// A cooked unit: where its entries are, the job's seconds and its report details.
type UnitResult = (From, f64, Detail);

/// What a cooked unit adds to its first entry's report.
#[derive(Default)]
struct Detail {
    raster: Option<RasterReport>,
    neural: Option<report::NeuralReport>,
    warnings: Vec<String>,
}

enum Msg {
    Import(usize, std::thread::Result<Result<Imported>>),
    Unit(usize, std::thread::Result<Result<UnitResult>>),
}

struct Run<'m, 's> {
    manifest: &'m Manifest,
    opts: &'m CookOptions,
    shared: &'s Shared,
    units: Vec<Unit<'m>>,
    /// Units by archive and name, and by name in the first archive that has the name.
    by_name: HashMap<(usize, String), usize>,
    first_by_name: HashMap<String, usize>,
    /// Per archive: the GDeflate codec of its payload blobs, if it asks for one.
    codecs: Vec<Option<Arc<dyn GDeflate>>>,
    old: Vec<Option<Old>>,
    /// Per archive: units in write order, the next one to write, and the writer (none
    /// while the archive matches the one it replaces).
    order: Vec<Vec<usize>>,
    cursor: Vec<usize>,
    writers: Vec<Option<ArchiveWriter>>,
    /// Per archive: records of its scene sources and every input file, for the cookstate.
    records: Vec<Vec<Record>>,
    files: Vec<Vec<PathBuf>>,
    imports: Vec<Import<'m>>,
    instances: Vec<PendingInstances<'m>>,
    /// Assemblies by model unit.
    assemblies: HashMap<usize, Assembly>,
    assets: Vec<AssetReport>,
    /// Report details of cooked units, by unit, until written.
    details: HashMap<usize, Detail>,
    /// Units whose dependencies are done, largest first; models and shared results apart,
    /// as they finish on the coordinator.
    ready: BTreeSet<(Reverse<u64>, usize)>,
    local: Vec<usize>,
    pending: Vec<usize>,
    dependents: Vec<Vec<usize>>,
    /// Per unit: mesh units that read its texture and have not started.
    map_users: Vec<usize>,
    window: u64,
}

impl<'m, 's> Run<'m, 's> {
    fn new(manifest: &'m Manifest, opts: &'m CookOptions, shared: &'s Shared) -> Result<Self> {
        let n = manifest.archives.len();
        let mut run = Self {
            manifest,
            opts,
            shared,
            units: Vec::new(),
            by_name: HashMap::new(),
            first_by_name: HashMap::new(),
            codecs: Vec::with_capacity(n),
            old: Vec::with_capacity(n),
            order: vec![Vec::new(); n],
            cursor: vec![0; n],
            writers: Vec::new(),
            records: vec![Vec::new(); n],
            files: vec![Vec::new(); n],
            imports: Vec::new(),
            instances: Vec::new(),
            assemblies: HashMap::new(),
            assets: Vec::new(),
            details: HashMap::new(),
            ready: BTreeSet::new(),
            local: Vec::new(),
            pending: Vec::new(),
            dependents: Vec::new(),
            map_users: Vec::new(),
            window: 0,
        };
        for (a, spec) in manifest.archives.iter().enumerate() {
            let codec: Option<Arc<dyn GDeflate>> = match spec.gdeflate {
                false => None,
                #[cfg(feature = "gdeflate")]
                true => Some(Arc::new(crate::gdeflate::Libdeflate::new(
                    key::GDEFLATE_LEVEL,
                )?)),
                #[cfg(not(feature = "gdeflate"))]
                true => return Err(Error::NoGDeflate { archive: a }),
            };
            run.codecs.push(codec);
            let mut previous = HashMap::new();
            let old = match &spec.output {
                Target::File(path) if !opts.force => {
                    previous = state::load(path, &shared.memo)
                        .into_iter()
                        .map(|r| (r.identity, r))
                        .collect();
                    Archive::open(path).ok().map(|archive| {
                        let mut by_key: HashMap<Hash32, Vec<usize>> = HashMap::new();
                        for (i, e) in archive.entries().iter().enumerate() {
                            by_key.entry(e.cook_key).or_default().push(i);
                        }
                        Old { archive, by_key }
                    })
                }
                _ => None,
            };
            run.old.push(old);
            // Texture assets first, so that library meshes find the textures their
            // materials name; reports keep the manifest's order.
            let first = run.assets.len();
            run.assets
                .extend(spec.assets.iter().map(|asset| AssetReport {
                    name: asset.name.clone(),
                    archive: a,
                    ..AssetReport::default()
                }));
            let n = spec.assets.len();
            let is_texture = |i: &usize| matches!(spec.assets[*i].source, Source::Texture { .. });
            let order = (0..n)
                .filter(is_texture)
                .chain((0..n).filter(|i| !is_texture(i)));
            let mut textures = HashMap::new();
            for i in order {
                let asset = &spec.assets[i];
                run.source(a, first + i, asset, &previous, &textures)?;
                if is_texture(&i) {
                    let u = &run.units[run.by_name[&(a, asset.name.clone())]];
                    textures.insert(AssetId::from_name(&asset.name), (u.name.clone(), u.key));
                }
            }
        }
        Ok(run)
    }

    /// Turns a source into units, or into a pending import or instance set, reporting as
    /// asset `index`. Library meshes read `textures`: the archive's texture assets by id,
    /// with their unit names and keys.
    fn source(
        &mut self,
        a: usize,
        index: usize,
        asset: &'m crate::AssetSpec,
        previous: &HashMap<Hash32, Record>,
        textures: &HashMap<AssetId, (String, Hash32)>,
    ) -> Result<()> {
        let name = asset.name.as_str();
        match &asset.source {
            Source::Gltf { path, settings } => {
                self.queue_import(a, index, name, path, ImportKind::Gltf(settings), previous)?;
            }
            Source::Usd { path, settings } => {
                self.queue_import(a, index, name, path, ImportKind::Usd(settings), previous)?;
            }
            Source::MaterialX {
                path,
                material,
                textures,
            } => {
                let kind = ImportKind::MaterialX { material, textures };
                self.queue_import(a, index, name, path, kind, previous)?;
            }
            Source::Texture { path, settings } => {
                let hash = self.shared.memo.hash(path)?;
                self.input(a, index, path);
                let key = Key::new(Class::Texture, name)
                    .str(ENCODERS)
                    .debug(settings)
                    .hash(hash)
                    .finish();
                let unit = NewUnit::leaf(
                    name,
                    Class::Texture as u16,
                    key,
                    texture_file_working_set(path),
                    Work::TextureFile(path.clone(), hash, settings.clone()),
                );
                self.add_units(a, index, vec![unit]);
            }
            Source::User {
                class,
                schema,
                version,
                bytes,
            } => {
                if *class < Class::USER_FIRST {
                    return Err(Error::UserClass {
                        asset: name.into(),
                        class: *class,
                    });
                }
                let key = Key::new(*class, name)
                    .bytes(&schema.to_le_bytes())
                    .bytes(&version.to_le_bytes())
                    .bytes(bytes)
                    .finish();
                let work = Work::Blobs {
                    schema: *schema,
                    blobs: vec![(bytes.clone(), false)],
                };
                let unit = NewUnit::leaf(name, *class, key, bytes.len() as u64, work);
                self.add_units(a, index, vec![unit]);
            }
            Source::Raster(set) => {
                let planned = raster::plan(name, set, &self.shared.memo)?;
                for f in &planned.files {
                    self.input(a, index, f);
                }
                let unit = NewUnit::leaf(
                    name,
                    Class::RasterSet as u16,
                    planned.key,
                    raster::WORKING_SET,
                    Work::Raster(planned.plan),
                );
                self.add_units(a, index, vec![unit]);
            }
            Source::Instances(src) => self.instances.push(PendingInstances {
                archive: a,
                asset: index,
                name: name.into(),
                src: Cow::Borrowed(src),
                record: None,
            }),
            Source::Mesh {
                mesh,
                materials,
                settings,
            } => {
                let (units, assembly) = mesh_units(name, mesh, materials, settings, textures)?;
                self.add_units(a, index, units);
                self.add_assemblies(a, assembly);
            }
            Source::Subd {
                cage,
                materials,
                settings,
            } => {
                let units = subd_units(name, cage, materials, settings)?;
                self.add_units(a, index, units);
            }
            Source::Material(patch) => {
                let mut m = Material::default();
                patch.apply(&mut m);
                self.add_units(a, index, vec![material_unit(name.into(), &m)]);
            }
        }
        Ok(())
    }

    /// Links a scene source's recorded units, or queues it for import.
    fn queue_import(
        &mut self,
        a: usize,
        asset: usize,
        name: &'m str,
        path: &'m Path,
        kind: ImportKind<'m>,
        previous: &HashMap<Hash32, Record>,
    ) -> Result<()> {
        let identity = Key::new(0u16, name)
            .hash(key::salt())
            .str(&path.to_string_lossy())
            .debug(&kind)
            .finish();
        let identity = key::stored_as(identity, self.codecs[a].is_some());
        if let Some(record) = previous.get(&identity)
            && self.link_record(a, asset, record)?
        {
            return Ok(());
        }
        self.imports.push(Import {
            archive: a,
            asset,
            name,
            path,
            kind,
            identity,
        });
        Ok(())
    }

    fn input(&mut self, a: usize, asset: usize, path: &Path) {
        self.files[a].push(path.to_owned());
        self.assets[asset].inputs.push(path.to_owned());
    }

    fn add(&mut self, unit: Unit<'m>) -> usize {
        let i = self.units.len();
        let a = unit.archive;
        self.by_name.insert((a, unit.name.clone()), i);
        let units = &self.units;
        self.first_by_name
            .entry(unit.name.clone())
            .and_modify(|u| {
                if a < units[*u].archive {
                    *u = i;
                }
            })
            .or_insert(i);
        self.order[a].push(i);
        self.units.push(unit);
        i
    }

    /// The unit named `name` in archive `a`, else in the first archive that has one, so
    /// the lookup does not depend on the order imports finish in.
    fn named(&self, a: usize, name: &str) -> Option<usize> {
        self.by_name
            .get(&(a, name.to_owned()))
            .or_else(|| self.first_by_name.get(name))
            .copied()
    }

    /// Links every unit of a recorded source if its files are unchanged and every key is
    /// in the replaced archive or the cache.
    fn link_record(&mut self, a: usize, asset: usize, record: &Record) -> Result<bool> {
        let unchanged = record
            .files
            .iter()
            .all(|f| self.shared.memo.hash(&f.path).is_ok_and(|h| h == f.hash));
        if !unchanged {
            return Ok(false);
        }
        let mut links = Vec::with_capacity(record.units.len());
        for &key in &record.units {
            match self.find_link(a, key) {
                Some(l) => links.push((key, l)),
                None => return Ok(false),
            }
        }
        for (key, (from, entries, outcome)) in links {
            let (mut name, class) = self.with_archive(a, &from, |arc| {
                let e = &arc.entries()[entries[0]];
                (arc.name(e).to_owned(), e.class)
            })?;
            // Instance pages are named `<source>/<cell>/<page>`; the unit is the source.
            if class == Class::InstancePage as u16
                && let Some(source) = name.rsplitn(3, '/').nth(2)
            {
                name = source.to_owned();
            }
            let u = self.add(Unit {
                archive: a,
                asset,
                name,
                class,
                key,
                work: None,
                deps: Vec::new(),
                working_set: 0,
                triangles: 0,
                state: State::Waiting,
                mesh: None,
            });
            self.done(u, from, Some(entries), outcome, 0.0)?;
        }
        for f in &record.files {
            self.input(a, asset, &f.path);
        }
        self.add_assemblies(a, record.assemblies.iter().cloned());
        self.records[a].push(record.clone());
        Ok(true)
    }

    /// The entries holding `key`: in the replaced archive, else in the cache.
    fn find_link(&self, a: usize, key: Hash32) -> Option<(From, Vec<usize>, Outcome)> {
        if self.opts.force {
            return None;
        }
        if let Some(entries) = self.old[a].as_ref().and_then(|o| o.by_key.get(&key)) {
            return Some((From::Old, entries.clone(), Outcome::Archive));
        }
        let cache = self.shared.cache.as_ref()?;
        let entries = (0..cache.find(key)?.entries().len()).collect();
        Some((From::Cache(cache.path(key)), entries, Outcome::Cache))
    }

    fn with_archive<T>(&self, a: usize, from: &From, f: impl FnOnce(&Archive) -> T) -> Result<T> {
        Ok(match from {
            From::Memory(arc) => f(arc),
            From::Old => f(&self.old[a]
                .as_ref()
                .expect("linked from the old archive")
                .archive),
            From::Cache(path) => f(&Archive::open(path)?),
        })
    }

    /// Marks a unit done: its entries (`None`: all of them) in write order and, for a mesh,
    /// the facts its model and instance pages need.
    fn done(
        &mut self,
        u: usize,
        from: From,
        entries: Option<Vec<usize>>,
        outcome: Outcome,
        seconds: f64,
    ) -> Result<()> {
        let a = self.units[u].archive;
        let class = self.units[u].class;
        let name = self.units[u].name.clone();
        let (entries, mesh) = self.with_archive(a, &from, |arc| {
            let mut entries = entries.unwrap_or_else(|| (0..arc.entries().len()).collect());
            if class == Class::InstancePage as u16 {
                entries.sort_by_key(|&e| instances::page_order(&name, arc.name(&arc.entries()[e])));
            }
            let e = &arc.entries()[entries[0]];
            let mesh = match Class::from_u16(class) {
                Some(Class::Mesh) => mesh_facts(arc, e),
                Some(Class::Subd) => subd_facts(arc, e),
                _ => None,
            };
            (entries, mesh)
        })?;
        let memory = match &from {
            From::Memory(arc) => arc.header().file_len,
            _ => 0,
        };
        self.window += memory;
        self.units[u].state = State::Done(Done {
            from,
            entries,
            outcome,
            seconds,
            memory,
        });
        self.units[u].mesh = mesh;
        let dependents = self.dependents.get_mut(u).map(std::mem::take);
        for d in dependents.into_iter().flatten() {
            self.pending[d] -= 1;
            if self.pending[d] == 0 {
                self.make_ready(d);
            }
        }
        Ok(())
    }

    fn make_ready(&mut self, u: usize) {
        if let Some(Work::Model(..) | Work::Same) = self.units[u].work {
            self.local.push(u);
        } else {
            self.ready.insert((Reverse(self.units[u].working_set), u));
        }
    }

    fn run(&mut self, scope: &rayon::Scope<'m>, threads: usize) -> Result<()>
    where
        's: 'm,
    {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.import_all(scope, threads, &tx, &rx)?;
        self.instance_units()?;
        self.link_all()?;
        self.open_writers()?;
        self.cook_all(scope, threads, &tx, &rx)
    }

    /// Imports every changed scene source, largest file first.
    fn import_all(
        &mut self,
        scope: &rayon::Scope<'m>,
        threads: usize,
        tx: &mpsc::Sender<Msg>,
        rx: &mpsc::Receiver<Msg>,
    ) -> Result<()>
    where
        's: 'm,
    {
        let size: Vec<u64> = self
            .imports
            .iter()
            .map(|i| std::fs::metadata(i.path).map_or(0, |m| m.len() * 8))
            .collect();
        let mut queue: Vec<usize> = (0..self.imports.len()).collect();
        queue.sort_by_key(|&i| Reverse(size[i]));
        let mut queue = queue.into_iter().peekable();
        let (mut running, mut bytes) = (0usize, 0u64);
        loop {
            while let Some(&i) = queue.peek() {
                let ws = size[i];
                if running >= threads || (running > 0 && bytes + ws > self.opts.working_set) {
                    break;
                }
                queue.next();
                running += 1;
                bytes += ws;
                let (tx, shared) = (tx.clone(), self.shared);
                let imp = &self.imports[i];
                let (name, path, kind) = (imp.name, imp.path, imp.kind);
                scope.spawn(move |_| {
                    let r =
                        catch_unwind(AssertUnwindSafe(|| import(name, path, kind, &shared.memo)));
                    let _ = tx.send(Msg::Import(i, r));
                });
            }
            if running == 0 {
                return Ok(());
            }
            let Ok(Msg::Import(i, r)) = rx.recv() else {
                unreachable!("only imports run");
            };
            running -= 1;
            bytes -= size[i];
            let imported = r.unwrap_or_else(|p| resume_unwind(p))?;
            self.add_import(i, imported);
        }
    }

    /// Adds units whose dependencies name units of the same list.
    fn add_units(&mut self, a: usize, asset: usize, units: Vec<NewUnit>) -> Vec<Hash32> {
        let mut keys = Vec::new();
        let mut deps = Vec::new();
        for nu in units {
            let key = key::stored_as(nu.key, self.codecs[a].is_some());
            keys.push(key);
            deps.push(nu.deps);
            self.add(Unit {
                archive: a,
                asset,
                name: nu.name,
                class: nu.class,
                key,
                work: Some(nu.work),
                deps: Vec::new(),
                working_set: nu.working_set,
                triangles: nu.triangles,
                state: State::Waiting,
                mesh: None,
            });
        }
        let first = self.units.len() - deps.len();
        for (u, names) in (first..).zip(deps) {
            self.units[u].deps = names
                .iter()
                .map(|n| self.by_name[&(a, n.clone())])
                .collect();
        }
        keys
    }

    /// Registers assemblies of models just added to archive `a`.
    fn add_assemblies(&mut self, a: usize, assemblies: impl IntoIterator<Item = Assembly>) {
        for assembly in assemblies {
            let u = self.by_name[&(a, assembly.model.clone())];
            self.assemblies.insert(u, assembly);
        }
    }

    fn add_import(&mut self, i: usize, imported: Imported) {
        let (a, asset, identity) = {
            let imp = &self.imports[i];
            (imp.archive, imp.asset, imp.identity)
        };
        let keys = self.add_units(a, asset, imported.units);
        for f in &imported.files {
            self.input(a, asset, &f.path);
        }
        self.add_assemblies(a, imported.assemblies.iter().cloned());
        self.records[a].push(Record {
            identity,
            files: imported.files,
            units: keys,
            assemblies: imported.assemblies,
        });
        if let Some(src) = imported.instances {
            self.instances.push(PendingInstances {
                archive: a,
                asset,
                name: format!("{}/instances", self.assets[asset].name),
                src: Cow::Owned(src),
                record: Some((a, self.records[a].len() - 1)),
            });
        }
    }

    /// Keys and dependencies of instance sources, from the keys of the models and
    /// materials they name.
    fn instance_units(&mut self) -> Result<()> {
        let by_id: HashMap<(usize, AssetId), usize> = self
            .by_name
            .iter()
            .map(|((a, n), &u)| ((*a, AssetId::from_name(n)), u))
            .collect();
        for pending in std::mem::take(&mut self.instances) {
            let PendingInstances {
                archive: a,
                asset,
                name,
                mut src,
                record,
            } = pending;
            let expanded = assembly::expand(&name, &src, |model| {
                self.named(a, model).and_then(|u| self.assemblies.get(&u))
            })?;
            if let Some(expanded) = expanded {
                src = Cow::Owned(expanded);
            }
            let cell_log2 = self.manifest.archives[a].cell_log2;
            let mut key =
                Key::new(Class::InstancePage, &name).hash(instances::input_hash(&src, cell_log2));
            let mut meshes = Vec::new();
            for p in &src.prototypes {
                let model = self
                    .named(a, &p.model)
                    .filter(|&u| self.units[u].class == Class::Model as u16)
                    .ok_or_else(|| Error::UnknownModel {
                        asset: name.clone(),
                        model: p.model.clone(),
                    })?;
                key = key.hash(self.units[model].key);
                for part in self.parts_of(model, &by_id)? {
                    meshes.push((p.model.clone(), part));
                }
                for s in &p.sections {
                    if let Some(m) = &s.material {
                        let u = self
                            .named(a, m)
                            .filter(|&u| self.units[u].class == Class::Material as u16)
                            .ok_or_else(|| Error::UnknownMaterial {
                                asset: name.clone(),
                                material: m.clone(),
                            })?;
                        key = key.hash(self.units[u].key);
                    }
                }
            }
            // Prototypes may share a model; each model's parts join once.
            meshes.sort_unstable();
            meshes.dedup();
            let mut deps: Vec<usize> = meshes.iter().map(|m| m.1).collect();
            deps.sort_unstable();
            deps.dedup();
            let key = key::stored_as(key.finish(), self.codecs[a].is_some());
            if let Some((ra, r)) = record {
                self.records[ra][r].units.push(key);
            }
            let working_set = src.records.len() as u64 * 256;
            self.add(Unit {
                archive: a,
                asset,
                name,
                class: Class::InstancePage as u16,
                key,
                work: Some(Work::Instances(src, cell_log2, Catalog::new(), meshes)),
                deps,
                working_set,
                triangles: 0,
                state: State::Waiting,
                mesh: None,
            });
        }
        Ok(())
    }

    /// The mesh and subdivision cage units of a model unit.
    fn parts_of(
        &self,
        model: usize,
        by_id: &HashMap<(usize, AssetId), usize>,
    ) -> Result<Vec<usize>> {
        let u = &self.units[model];
        if let Some(Work::Model(..)) = &u.work {
            return Ok(u.deps.clone());
        }
        let State::Done(d) = &u.state else {
            unreachable!("a model unit without work is linked");
        };
        let header = self.with_archive(u.archive, &d.from, |arc| {
            let e = &arc.entries()[d.entries[0]];
            let mut head = vec![0; arc.blobs(e)[0].raw_len as usize];
            arc.read(&arc.blobs(e)[0], &mut head)?;
            Ok::<_, Error>(ModelInfo::parse(&head)?.header)
        })??;
        [header.mesh, header.subd]
            .into_iter()
            .filter(|&id| id != AssetId::NONE)
            .map(|id| {
                by_id
                    .get(&(u.archive, id))
                    .copied()
                    .ok_or_else(|| Error::UnknownModel {
                        asset: u.name.clone(),
                        model: format!("geometry {id}"),
                    })
            })
            .collect()
    }

    /// Links every unit whose key is in the replaced archive or the cache; the others wait
    /// for their dependencies.
    fn link_all(&mut self) -> Result<()> {
        self.pending = vec![0; self.units.len()];
        self.dependents = vec![Vec::new(); self.units.len()];
        self.map_users = vec![0; self.units.len()];
        for u in 0..self.units.len() {
            if !matches!(self.units[u].state, State::Waiting) {
                continue;
            }
            if let Some((from, entries, outcome)) =
                self.find_link(self.units[u].archive, self.units[u].key)
            {
                self.units[u].work = None;
                self.done(u, from, Some(entries), outcome, 0.0)?;
            }
        }
        // Units with equal keys (the same asset in several archives) cook once.
        let mut first = HashMap::new();
        for u in 0..self.units.len() {
            if !matches!(self.units[u].state, State::Waiting) {
                continue;
            }
            if let Some(&f) = first.get(&self.units[u].key) {
                self.units[u].work = Some(Work::Same);
                self.units[u].deps = vec![f];
            } else {
                first.insert(self.units[u].key, u);
            }
        }
        for u in 0..self.units.len() {
            if !matches!(self.units[u].state, State::Waiting) {
                continue;
            }
            if let Some(Work::Mesh(data, ..)) = &self.units[u].work {
                let a = self.units[u].archive;
                for unit in data.sections.iter().flat_map(|s| s.maps.units()) {
                    self.map_users[self.by_name[&(a, unit.to_owned())]] += 1;
                }
            }
            let deps = self.units[u].deps.clone();
            for d in deps {
                if !matches!(self.units[d].state, State::Done(_)) {
                    self.pending[u] += 1;
                    self.dependents[d].push(u);
                }
            }
            if self.pending[u] == 0 {
                self.make_ready(u);
            }
        }
        Ok(())
    }

    /// Sorts each archive's units into write order and opens a writer for every archive
    /// that does not match the one it replaces.
    fn open_writers(&mut self) -> Result<()> {
        for a in 0..self.order.len() {
            let mut order = std::mem::take(&mut self.order[a]);
            order.sort_by(|&x, &y| {
                let k = |u: usize| {
                    let unit = &self.units[u];
                    (unit.class == Class::InstancePage as u16, unit.name.as_str())
                };
                k(x).cmp(&k(y))
            });
            self.order[a] = order;
            let unchanged = self.old[a].as_ref().is_some_and(|old| {
                let mut linked = 0;
                let all_old = self.order[a].iter().all(|&u| match &self.units[u].state {
                    State::Done(d) if matches!(d.from, From::Old) => {
                        linked += d.entries.len();
                        true
                    }
                    _ => false,
                });
                all_old && linked == old.archive.entries().len()
            });
            self.writers.push(if unchanged {
                None
            } else {
                let mut w = ArchiveWriter::create(self.manifest.archives[a].output.clone())?;
                w.set_zstd_level(ZSTD_LEVEL);
                Some(w)
            });
        }
        Ok(())
    }

    fn cook_all(
        &mut self,
        scope: &rayon::Scope<'m>,
        threads: usize,
        tx: &mpsc::Sender<Msg>,
        rx: &mpsc::Receiver<Msg>,
    ) -> Result<()>
    where
        's: 'm,
    {
        let (mut running, mut bytes) = (0usize, 0u64);
        loop {
            while let Some(u) = self.local.pop() {
                self.finish_local(u)?;
            }
            self.write_ready()?;
            if self
                .cursor
                .iter()
                .zip(&self.order)
                .all(|(c, o)| *c == o.len())
            {
                return Ok(());
            }
            let candidates: Vec<usize> = if self.window > self.opts.reorder_window {
                (0..self.order.len())
                    .filter_map(|a| {
                        self.order[a]
                            .get(self.cursor[a])
                            .and_then(|&u| self.blocking(u))
                    })
                    .collect()
            } else {
                self.ready.iter().map(|&(_, u)| u).collect()
            };
            for u in candidates {
                let ws = self.units[u].working_set;
                if running >= threads || (running > 0 && bytes + ws > self.opts.working_set) {
                    break;
                }
                if self.ready.remove(&(Reverse(ws), u)) {
                    running += 1;
                    bytes += ws;
                    self.spawn(scope, tx, u);
                }
            }
            assert!(
                running > 0 || !self.local.is_empty(),
                "the cook has nothing to run and entries left to write"
            );
            if running == 0 {
                continue;
            }
            let Ok(Msg::Unit(u, r)) = rx.recv() else {
                unreachable!("only units run");
            };
            running -= 1;
            bytes -= self.units[u].working_set;
            let (from, seconds, detail) = r.unwrap_or_else(|p| resume_unwind(p))?;
            self.details.insert(u, detail);
            self.done(u, from, None, Outcome::Cooked, seconds)?;
        }
    }

    /// A ready unit that `u`, the next unit to write, waits on.
    fn blocking(&self, u: usize) -> Option<usize> {
        match self.units[u].state {
            State::Waiting if self.pending[u] == 0 => self
                .ready
                .contains(&(Reverse(self.units[u].working_set), u))
                .then_some(u),
            State::Waiting => self.units[u].deps.iter().find_map(|&d| self.blocking(d)),
            _ => None,
        }
    }

    /// The cooked maps a mesh unit's sections read, from the texture units it waited for.
    fn section_maps(&mut self, u: usize, mesh: &MeshData) -> Result<Vec<SectionMaps>> {
        let a = self.units[u].archive;
        let mut users = Vec::new();
        let mut read = |unit: &str, all: bool, wrap| -> Result<Vec<Raster>> {
            let t = self.by_name[&(a, unit.to_owned())];
            users.push(t);
            let from = match &self.units[t].state {
                State::Done(d) => &d.from,
                State::Written(Some(f)) => f,
                _ => unreachable!("a mesh starts after the textures it reads"),
            };
            self.with_archive(a, from, |arc| {
                texture_mips(arc, AssetId::from_name(unit), all, wrap)
            })?
        };
        let mut out = Vec::with_capacity(mesh.sections.len());
        for s in &mesh.sections {
            let opacity = match &s.maps.opacity {
                OpacitySpec::None => None,
                OpacitySpec::Constant(pass) => Some(Opacity::Constant(*pass)),
                OpacitySpec::Unknown => Some(Opacity::Unknown),
                OpacitySpec::Texture {
                    unit, cutoff, wrap, ..
                } => Some(Opacity::Texture {
                    mip0: read(unit, false, *wrap)?.remove(0),
                    cutoff: *cutoff,
                }),
            };
            let displacement = match &s.maps.displacement {
                None => None,
                Some(d) => Some(Displacement {
                    scale: d.scale,
                    offset: d.offset,
                    min: d.min,
                    max: d.max,
                    mips: match &d.texture {
                        Some((unit, _, wrap)) => read(unit, true, *wrap)?,
                        None => Vec::new(),
                    },
                }),
            };
            out.push(SectionMaps {
                opacity,
                displacement,
            });
        }
        for t in users {
            self.map_users[t] -= 1;
            if self.map_users[t] == 0 && matches!(self.units[t].state, State::Written(_)) {
                self.units[t].state = State::Written(None);
            }
        }
        Ok(out)
    }

    fn spawn(&mut self, scope: &rayon::Scope<'m>, tx: &mpsc::Sender<Msg>, u: usize)
    where
        's: 'm,
    {
        let mut work = self.units[u].work.take().expect("waiting units have work");
        if let Work::Instances(_, _, catalog, meshes) = &mut work {
            for (model, mesh) in meshes.drain(..) {
                if let Some((_, facts)) = self.units[mesh].mesh {
                    catalog
                        .entry(model)
                        .and_modify(|f| *f = f.join(facts))
                        .or_insert(facts);
                }
            }
        }
        if let Work::Mesh(data, _, maps) = &mut work {
            match self.section_maps(u, data) {
                Ok(m) => *maps = m,
                Err(e) => {
                    let _ = tx.send(Msg::Unit(u, Ok(Err(e))));
                    self.units[u].state = State::Running;
                    return;
                }
            }
        }
        self.units[u].state = State::Running;
        let unit = &self.units[u];
        let (name, class, key) = (unit.name.clone(), unit.class, unit.key);
        let gdeflate = self.codecs[unit.archive].clone();
        let (tx, shared) = (tx.clone(), self.shared);
        scope.spawn(move |_| {
            let r = catch_unwind(AssertUnwindSafe(|| {
                let start = Instant::now();
                let (from, detail) = cook_unit(shared, &name, class, key, gdeflate, work)?;
                Ok((from, start.elapsed().as_secs_f64(), detail))
            }));
            let _ = tx.send(Msg::Unit(u, r));
        });
    }

    /// Finishes a unit on the coordinator: a model, whose mesh is done and the rest is a
    /// header, or a unit sharing the result of one with the same key.
    fn finish_local(&mut self, u: usize) -> Result<()> {
        let start = Instant::now();
        let (materials, skeleton) = match self.units[u].work.take() {
            Some(Work::Model(materials, skeleton)) => (materials, skeleton),
            Some(Work::Same) => {
                let State::Done(d) = &self.units[self.units[u].deps[0]].state else {
                    unreachable!("shared results are done and not yet written");
                };
                let (from, outcome) = (d.from.clone(), d.outcome);
                return self.done(u, from, None, outcome, 0.0);
            }
            _ => unreachable!("only models and shared results finish here"),
        };
        let mut bounds = Aabb::EMPTY;
        let (mut mesh, mut subd) = (AssetId::NONE, AssetId::NONE);
        for &part in &self.units[u].deps {
            let (b, _) = self.units[part]
                .mesh
                .ok_or(orr_pack::Error::Malformed("geometry header"))?;
            bounds.extend(b.min);
            bounds.extend(b.max);
            let id = AssetId::from_name(&self.units[part].name);
            if self.units[part].class == Class::Subd as u16 {
                subd = id;
            } else {
                mesh = id;
            }
        }
        let bytes = ModelInfo {
            header: ModelHeader {
                kind: ModelHeader::MODEL,
                _pad: [0; 3],
                bounds,
                mesh,
                subd,
                skeleton,
                materials: Span::default(),
            },
            materials,
        }
        .to_bytes();
        let unit = &self.units[u];
        let work = Work::Blobs {
            schema: orr_pack::model::SCHEMA,
            blobs: vec![(bytes, true)],
        };
        let gdeflate = self.codecs[unit.archive].clone();
        let (from, _) = cook_unit(
            self.shared,
            &unit.name,
            unit.class,
            unit.key,
            gdeflate,
            work,
        )?;
        self.done(
            u,
            from,
            None,
            Outcome::Cooked,
            start.elapsed().as_secs_f64(),
        )
    }

    /// Writes every unit that is next in its archive's order and done.
    fn write_ready(&mut self) -> Result<()> {
        for a in 0..self.order.len() {
            while let Some(&u) = self.order[a].get(self.cursor[a]) {
                if !matches!(self.units[u].state, State::Done(_)) {
                    break;
                }
                let State::Done(d) =
                    std::mem::replace(&mut self.units[u].state, State::Written(None))
                else {
                    unreachable!("checked above");
                };
                if self.map_users[u] > 0 {
                    self.units[u].state = State::Written(Some(d.from.clone()));
                }
                let unit = &self.units[u];
                let (asset, working_set, triangles) =
                    (unit.asset, unit.working_set, unit.triangles);
                let writer = self.writers[a].as_mut();
                let old = self.old[a].as_ref().map(|o| &o.archive);
                let mut reports = Vec::new();
                let write = |arc: &Archive| -> Result<()> {
                    let mut writer = writer;
                    for &e in &d.entries {
                        let entry = &arc.entries()[e];
                        if let Some(w) = writer.as_deref_mut() {
                            w.copy_entry(arc, entry)?;
                        }
                        reports.push(report::entry(
                            arc,
                            entry,
                            d.outcome,
                            d.seconds,
                            working_set,
                            triangles,
                        ));
                    }
                    Ok(())
                };
                match &d.from {
                    From::Memory(arc) => write(arc)?,
                    From::Old => write(old.expect("linked from the old archive"))?,
                    From::Cache(path) => write(&Archive::open(path)?)?,
                }
                if let (Some(d), Some(first)) = (self.details.remove(&u), reports.first_mut()) {
                    first.raster = d.raster;
                    first.neural = d.neural;
                    first.warnings = d.warnings;
                }
                self.assets[asset].entries.extend(reports);
                self.window -= d.memory;
                self.cursor[a] += 1;
            }
        }
        Ok(())
    }

    /// Completes the archives, then saves cookstates and trims the cache.
    fn finish(self) -> Result<Cooked> {
        let mut cooked = Cooked::default();
        let Self {
            manifest,
            opts,
            shared,
            writers,
            records,
            files,
            mut assets,
            units,
            assemblies,
            ..
        } = self;
        for (u, a) in assemblies {
            let asset = &mut assets[units[u].asset];
            let report = assembly_report(&a, &asset.entries);
            asset.assemblies.push(report);
        }
        for a in &mut assets {
            a.assemblies.sort_by(|x, y| x.model.cmp(&y.model));
        }
        for (a, (spec, writer)) in manifest.archives.iter().zip(writers).enumerate() {
            let entries = assets
                .iter()
                .filter(|r| r.archive == a)
                .map(|r| r.entries.len())
                .sum();
            let written = writer.is_some();
            let mut report = ArchiveReport {
                output: None,
                written,
                entries,
                bytes: 0,
            };
            match writer.map(ArchiveWriter::finish).transpose()?.flatten() {
                Some(bytes) => {
                    report.bytes = bytes.len() as u64;
                    cooked.memory.push(bytes);
                }
                None => {
                    if let Target::File(path) = &spec.output {
                        report.output = Some(path.clone());
                        report.bytes = std::fs::metadata(path).map_or(0, |m| m.len());
                    }
                }
            }
            cooked.report.archives.push(report);
        }
        for (a, spec) in manifest.archives.iter().enumerate() {
            if let Target::File(path) = &spec.output {
                state::save(path, &shared.memo, &files[a], records[a].clone())?;
            }
        }
        if let (Some(cache), Some(cap)) = (&shared.cache, opts.cache_cap) {
            cache.evict(cap)?;
        }
        cooked.report.assets = assets;
        Ok(cooked)
    }
}

/// An assembly's units, copies and repeated triangles and bytes, from the entries of its
/// asset.
fn assembly_report(a: &Assembly, entries: &[report::EntryReport]) -> AssemblyReport {
    let mut out = AssemblyReport {
        model: a.model.clone(),
        triangles: a.source_triangles,
        kept_triangles: a.kept_triangles,
        ..AssemblyReport::default()
    };
    let mut copies = vec![0u32; a.units.len()];
    for p in &a.placements {
        copies[p.unit as usize] += 1;
    }
    for (unit, copies) in a.units.iter().zip(copies) {
        let mesh = format!("{}/mesh", unit.model);
        let stored = entries
            .iter()
            .find(|e| e.name == mesh)
            .map_or(0, |e| e.stored);
        let extra = u64::from(copies.saturating_sub(1));
        out.repeated_triangles += extra * u64::from(unit.triangles);
        out.repeated_bytes += extra * stored;
        out.units.push(AssemblyUnitReport {
            model: unit.model.clone(),
            copies,
            triangles: unit.triangles,
            stored,
        });
    }
    out
}

fn mesh_facts(a: &Archive, e: &orr_pack::Entry) -> Option<(Aabb, ModelFacts)> {
    let b = a.blobs(e).first()?;
    let mut head = vec![0; b.raw_len as usize];
    a.read(b, &mut head).ok()?;
    let info = orr_pack::mesh::MeshInfo::parse(&head).ok()?;
    Some((
        info.header.bounds,
        ModelFacts {
            instances: (info.parts.len() as u32).max(1),
            sections: info.sections.len() as u32,
            position_exponent: info.header.position_exponent,
        },
    ))
}

fn subd_facts(a: &Archive, e: &orr_pack::Entry) -> Option<(Aabb, ModelFacts)> {
    let b = a.blobs(e).first()?;
    let mut head = vec![0; b.raw_len as usize];
    a.read(b, &mut head).ok()?;
    let info = orr_pack::subd::SubdInfo::parse(&head).ok()?;
    Some((
        info.header.bounds,
        ModelFacts {
            instances: 1,
            sections: info.sections.len() as u32,
            position_exponent: i8::MIN,
        },
    ))
}

/// Estimated peak of a texture cook: 64 B per source texel when the header gives the size.
fn texture_file_working_set(path: &Path) -> u64 {
    image::ImageReader::open(path)
        .and_then(image::ImageReader::with_guessed_format)
        .ok()
        .and_then(|r| r.into_dimensions().ok())
        .map_or_else(
            || std::fs::metadata(path).map_or(0, |m| m.len() * 64),
            |(w, h)| u64::from(w) * u64::from(h) * 64,
        )
}

/// Imports a scene or material document and plans its units.
fn import(name: &str, path: &Path, kind: ImportKind<'_>, memo: &Memo) -> Result<Imported> {
    let (scene, rules) = match kind {
        ImportKind::Gltf(s) => (crate::gltf::import(name, path, memo, s)?, &s.textures),
        ImportKind::Usd(s) => (crate::usd::import(name, path, s)?, &s.textures),
        ImportKind::MaterialX { material, textures } => (
            crate::materialx::import(name, path, memo, material)?,
            textures,
        ),
    };
    plan_scene(name, scene, rules)
}

/// The entry name of material `suffix` of asset `name`.
fn material_name(name: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        name.to_string()
    } else {
        format!("{name}/material/{suffix}")
    }
}

/// Plans an imported scene's units: the skeletons and clips, the packed textures, the
/// materials, and a mesh and a model per model.
fn plan_scene(name: &str, mut src: Scene, rules: &TextureRules) -> Result<Imported> {
    let mut units = Vec::new();
    let mut skeletons = Vec::with_capacity(src.skeletons.len());
    for sk in std::mem::take(&mut src.skeletons) {
        let prefix = match sk.prefix.as_str() {
            "" => name.to_string(),
            p => format!("{name}/{p}"),
        };
        let skeleton_name = format!("{prefix}/skeleton");
        let bytes = sk.info.to_bytes();
        let key = Key::new(Class::Skeleton, &skeleton_name)
            .bytes(&bytes)
            .finish();
        let skeleton = (AssetId::from_name(&skeleton_name), Some(key));
        skeletons.push(skeleton);
        let working_set = bytes.len() as u64;
        let work = Work::Blobs {
            schema: orr_pack::skeleton::SCHEMA,
            blobs: vec![(bytes, true)],
        };
        let class = Class::Skeleton as u16;
        units.push(NewUnit::leaf(&skeleton_name, class, key, working_set, work));
        for (clip_name, mut clip) in sk.clips {
            clip.header.skeleton = skeleton.0;
            let clip_name = format!("{prefix}/clip/{clip_name}");
            let (head, samples) = clip.to_bytes();
            let key = Key::new(Class::Clip, &clip_name)
                .bytes(&head)
                .bytes(&samples)
                .finish();
            let working_set = (head.len() + samples.len()) as u64;
            let work = Work::Blobs {
                schema: orr_pack::clip::SCHEMA,
                blobs: vec![(head, true), (samples, false)],
            };
            let class = Class::Clip as u16;
            units.push(NewUnit::leaf(&clip_name, class, key, working_set, work));
        }
    }

    let mut used: Vec<usize> = src
        .materials
        .iter()
        .flat_map(|m| m.textures.refs().map(|r| r.image))
        .collect();
    used.sort_unstable();
    used.dedup();
    let facts: HashMap<usize, ImageFacts> = image_facts(name, &src.images, &used)?;
    let mut packer = Packer::new(name, &src.images, &facts, rules);
    // Per material: its id, key and cooked parameters.
    let mut materials = Vec::new();
    for m in &src.materials {
        let mut cooked = m.material;
        packer.material(&m.suffix, &mut cooked, &m.textures)?;
        let unit = material_unit(material_name(name, &m.suffix), &cooked);
        materials.push((AssetId::from_name(&unit.name), unit.key, cooked));
        units.push(unit);
    }
    let jobs = std::mem::take(&mut packer.jobs);
    let sets = std::mem::take(&mut packer.neural);
    drop(packer);
    let textures: HashMap<AssetId, (String, Hash32)> = jobs
        .iter()
        .map(|j| (AssetId::from_name(&j.name), (j.name.clone(), j.key)))
        .collect();

    // Materials of statically displaced sections without their displacement, which the
    // geometry holds.
    let mut baked: HashMap<usize, (AssetId, Hash32)> = HashMap::new();
    let mut assemblies = Vec::new();
    for model in src.models {
        let model_name = match &model.node {
            None => name.to_string(),
            Some(n) => format!("{name}/{n}"),
        };
        let mut mesh = model.mesh;
        let mut deps = Vec::new();
        let mut model_materials = Vec::with_capacity(model.materials.len());
        for (sec, &mi) in mesh.sections.iter_mut().zip(&model.materials) {
            let t = &src.materials[mi].textures;
            sec.maps = map_spec(&materials[mi].2, t, &textures);
            deps.extend(sec.maps.units().map(str::to_string));
            model_materials.push(if bakes_displacement(&model.settings, &sec.maps) {
                *baked.entry(mi).or_insert_with(|| {
                    let suffix = &src.materials[mi].suffix;
                    let unit = static_material_unit(&material_name(name, suffix), &materials[mi].2);
                    let id = (AssetId::from_name(&unit.name), unit.key);
                    units.push(unit);
                    id
                })
            } else {
                (materials[mi].0, materials[mi].1)
            });
        }
        deps.sort_unstable();
        deps.dedup();
        let skeleton = model
            .skeleton
            .map_or((AssetId::NONE, None), |s| skeletons[s]);
        let mesh = (!mesh.sections.is_empty()).then(|| MeshPart {
            name: format!("{model_name}/mesh"),
            mesh,
            settings: model.settings,
            deps,
        });
        let subd = model.subd.map(|c| {
            model_materials.extend(
                c.materials
                    .iter()
                    .map(|&mi| (materials[mi].0, materials[mi].1)),
            );
            SubdPart {
                name: format!("{model_name}/subd"),
                cage: c.cage,
                materials: c.materials.iter().map(|&mi| materials[mi].2).collect(),
                settings: c.settings,
            }
        });
        let (model, assembly) = model_units(model_name, mesh, subd, &model_materials, skeleton);
        units.extend(model);
        assemblies.extend(assembly);
    }
    let instances = (!src.instances.is_empty())
        .then(|| scene_instances(name, &src.instances))
        .transpose()?;
    let images = Arc::new(src.images);
    for job in jobs {
        let (name, key, working_set) = (job.name.clone(), job.key, job.working_set(&images));
        let work = Work::Texture(job, Arc::clone(&images));
        let class = Class::Texture as u16;
        units.push(NewUnit::leaf(&name, class, key, working_set, work));
    }
    for set in sets {
        let (name, key, working_set) = (set.name.clone(), set.key, set.working_set(&images));
        let work = Work::Neural(set, Arc::clone(&images));
        let class = Class::NeuralTextureSet as u16;
        units.push(NewUnit::leaf(&name, class, key, working_set, work));
    }
    Ok(Imported {
        files: src.files,
        units,
        instances,
        assemblies,
    })
}

/// The instance source of a scene's instanced models: one prototype per model, a placement
/// per instance.
fn scene_instances(name: &str, models: &[(String, Vec<glam::Mat4>)]) -> Result<Instances> {
    let mut src = Instances::default();
    for (p, (node, transforms)) in models.iter().enumerate() {
        src.prototypes.push(PrototypeSpec {
            model: format!("{name}/{node}"),
            ..PrototypeSpec::default()
        });
        for (i, m) in transforms.iter().enumerate() {
            let (scale, rotation, translation) = m.to_scale_rotation_translation();
            let back = glam::Mat4::from_scale_rotation_translation(scale, rotation, translation);
            let tolerance = 1e-4 * m.abs().to_cols_array().into_iter().fold(1.0, f32::max);
            if !back.abs_diff_eq(*m, tolerance) {
                return Err(Error::Import {
                    asset: name.into(),
                    message: format!(
                        "{node}: instance {i} has a shear, which a placement cannot hold"
                    ),
                });
            }
            src.records.push(Placement {
                position: translation.as_dvec3(),
                rotation,
                scale,
                prototype: p as u16,
            });
        }
    }
    Ok(src)
}

/// Whether a section with `maps` bakes its displacement into the geometry.
fn bakes_displacement(settings: &MeshSettings, maps: &MapSpec) -> bool {
    settings.displace_static.is_some()
        && maps
            .displacement
            .as_ref()
            .is_some_and(|d| d.texture.is_some())
}

/// Material `name`'s variant for statically displaced sections, `<name>/static`: without
/// its displacement, which the geometry holds.
fn static_material_unit(name: &str, m: &Material) -> NewUnit {
    let mut m = *m;
    m.textures[slot::DISPLACEMENT] = bytemuck::Zeroable::zeroed();
    m.displacement_scale = 0.0;
    m.displacement_offset = 0.0;
    m.displacement_min = 0.0;
    m.displacement_max = 0.0;
    material_unit(format!("{name}/static"), &m)
}

/// A material entry holding `m`.
fn material_unit(name: String, m: &Material) -> NewUnit {
    let bytes = bytemuck::bytes_of(m).to_vec();
    let key = Key::new(Class::Material, &name).bytes(&bytes).finish();
    let working_set = bytes.len() as u64;
    let work = Work::Blobs {
        schema: orr_pack::material::SCHEMA,
        blobs: vec![(bytes, true)],
    };
    NewUnit::leaf(&name, Class::Material as u16, key, working_set, work)
}

/// The cooked textures a section with material `m` reads at mesh cook time.
fn map_spec(
    m: &Material,
    t: &MaterialTextures,
    textures: &HashMap<AssetId, (String, Hash32)>,
) -> MapSpec {
    let bound = |s: usize| {
        let b = m.textures[s];
        (b.texture != AssetId::NONE).then(|| (textures.get(&b.texture), b.uv_set))
    };
    let wrap = |r: Option<&TexRef>| r.map_or([Wrap::Repeat; 2], |r| r.wrap);
    let opacity = match (m.alpha_mode, bound(slot::OPACITY)) {
        (Material::MASK, None) => OpacitySpec::Constant(m.geometry_opacity >= m.alpha_cutoff),
        (Material::MASK, Some((Some((unit, key)), 0))) => OpacitySpec::Texture {
            unit: unit.clone(),
            key: *key,
            cutoff: m.alpha_cutoff,
            wrap: wrap(t.opacity.as_ref()),
        },
        (Material::MASK, Some(_)) => OpacitySpec::Unknown,
        _ => OpacitySpec::None,
    };
    let displacement = bound(slot::DISPLACEMENT).map(|b| DisplacementSpec {
        scale: m.displacement_scale,
        offset: m.displacement_offset,
        min: m.displacement_min,
        max: m.displacement_max,
        texture: match b {
            (Some((unit, key)), 0) => Some((unit.clone(), *key, wrap(t.displacement.as_ref()))),
            _ => None,
        },
    });
    MapSpec {
        opacity,
        displacement,
    }
}

/// A mesh, its materials and a model from a library caller's mesh, whose materials name
/// textures among `textures`, and the model's assembly.
fn mesh_units(
    name: &str,
    input: &MeshInput,
    materials: &[Material],
    settings: &MeshSettings,
    textures: &HashMap<AssetId, (String, Hash32)>,
) -> Result<(Vec<NewUnit>, Option<Assembly>)> {
    let fail = |message: String| Error::Mesh {
        asset: name.into(),
        message,
    };
    if materials.len() != input.sections.len() {
        return Err(fail(format!(
            "{} materials for {} sections",
            materials.len(),
            input.sections.len()
        )));
    }
    let mut units = Vec::new();
    let mesh_name = format!("{name}/mesh");
    let mut sections = Vec::new();
    let mut materials_out = Vec::new();
    let mut deps = Vec::new();
    for (i, (m, span)) in materials.iter().zip(&input.sections).enumerate() {
        let maps = map_spec(m, &MaterialTextures::default(), textures);
        deps.extend(maps.units().map(str::to_string));
        let mut unit = material_unit(format!("{name}/material/{i}"), m);
        if bakes_displacement(settings, &maps) {
            let baked = static_material_unit(&unit.name, m);
            units.push(std::mem::replace(&mut unit, baked));
        }
        materials_out.push((AssetId::from_name(&unit.name), unit.key));
        units.push(unit);
        let first = span.first as usize;
        sections.push(SectionData::new(
            first..first + span.count as usize,
            m,
            maps,
        ));
    }
    deps.sort_unstable();
    deps.dedup();
    let n = input.positions.len();
    let mesh = MeshData {
        positions: input.positions.iter().map(Vec3::to_array).collect(),
        normals: input.normals.iter().map(Vec3::to_array).collect(),
        tangents: input
            .tangents
            .as_ref()
            .map(|t| t.iter().map(Vec4::to_array).collect()),
        uvs: input.uvs.as_ref().map_or_else(
            || vec![[0.0; 2]; n],
            |u| u.iter().map(Vec2::to_array).collect(),
        ),
        indices: input.indices.clone(),
        sections,
        skin: None,
        parts: input.parts.as_ref().map(|p| PartData {
            values: p.values.clone(),
            count: p.count,
            by_section: p.by_section,
            pivots: p.pivots.as_ref().map_or(Pivots::Origin, |v| {
                Pivots::PerVertex(v.iter().map(Vec3::to_array).collect())
            }),
            joints: None,
        }),
        min_exponent: None,
    };
    let part = MeshPart {
        name: mesh_name,
        mesh,
        settings: settings.clone(),
        deps,
    };
    let (model, assembly) = model_units(
        name.into(),
        Some(part),
        None,
        &materials_out,
        (AssetId::NONE, None),
    );
    units.extend(model);
    Ok((units, assembly))
}

/// A mesh to cook for a model: its entry name, data, settings and the units its maps read.
struct MeshPart {
    name: String,
    mesh: MeshData,
    settings: MeshSettings,
    deps: Vec<String>,
}

/// A subdivision cage to cook for a model: its entry name, cage, section materials and
/// settings.
struct SubdPart {
    name: String,
    cage: CageData,
    materials: Vec<Material>,
    settings: SubdSettings,
}

/// The mesh and cage units of a model and the model unit, with these materials (id and key,
/// mesh sections then cage sections) and skeleton (id and key). With assemblies, also a mesh
/// and a model per repeated unit, and the assembly placing them; the model's mesh then
/// holds the rest.
fn model_units(
    model_name: String,
    mut mesh: Option<MeshPart>,
    subd: Option<SubdPart>,
    materials: &[(AssetId, Hash32)],
    skeleton: (AssetId, Option<Hash32>),
) -> (Vec<NewUnit>, Option<Assembly>) {
    let mut units = Vec::new();
    let mut found = None;
    if let Some(part) = &mut mesh
        && let Some(settings) = &part.settings.assemblies
        && let Some(split) = assembly::split(&part.mesh, settings, part.settings.position_step_log2)
    {
        let source_triangles = part.mesh.triangles();
        let mut list = Vec::with_capacity(split.units.len());
        for (i, (mesh, sections)) in split.units.into_iter().enumerate() {
            let name = format!("{model_name}/unit/{i}");
            let unit_materials: Vec<_> = sections
                .iter()
                .map(|&s| materials[usize::from(s)])
                .collect();
            list.push(assembly::Unit {
                model: name.clone(),
                sections,
                triangles: mesh.triangles() as u32,
            });
            let mut deps: Vec<String> = mesh
                .sections
                .iter()
                .flat_map(|s| s.maps.units())
                .map(str::to_string)
                .collect();
            deps.sort_unstable();
            deps.dedup();
            let unit_mesh = MeshPart {
                name: format!("{name}/mesh"),
                mesh,
                settings: part.settings.clone(),
                deps,
            };
            units.extend(geometry_units(
                name,
                Some(unit_mesh),
                None,
                &unit_materials,
                (AssetId::NONE, None),
            ));
        }
        part.mesh = split.remainder;
        found = Some(Assembly {
            model: model_name.clone(),
            units: list,
            placements: split.placements,
            source_triangles,
            kept_triangles: part.mesh.triangles(),
        });
    }
    units.extend(geometry_units(model_name, mesh, subd, materials, skeleton));
    (units, found)
}

/// The mesh and cage units of a model and the model unit, with these materials (id and key,
/// mesh sections then cage sections) and skeleton (id and key).
fn geometry_units(
    model_name: String,
    mesh: Option<MeshPart>,
    subd: Option<SubdPart>,
    materials: &[(AssetId, Hash32)],
    skeleton: (AssetId, Option<Hash32>),
) -> Vec<NewUnit> {
    let mut units = Vec::new();
    let mut model_key = Key::new(Class::Model, &model_name);
    if let Some(m) = mesh {
        let key = Key::new(Class::Mesh, &m.name)
            .debug(&m.settings)
            .hash(m.mesh.hash())
            .finish();
        model_key = model_key.hash(key);
        let triangles = m.mesh.triangles();
        units.push(NewUnit {
            name: m.name,
            class: Class::Mesh as u16,
            key,
            working_set: triangles * MESH_BYTES_PER_TRIANGLE,
            triangles,
            work: Work::Mesh(Arc::new(m.mesh), m.settings, Vec::new()),
            deps: m.deps,
        });
    }
    if let Some(c) = subd {
        let mut key = Key::new(Class::Subd, &c.name)
            .debug(&c.settings)
            .hash(c.cage.hash());
        for m in &c.materials {
            key = key.bytes(bytemuck::bytes_of(m));
        }
        let key = key.finish();
        model_key = model_key.hash(key);
        units.push(NewUnit::leaf(
            &c.name,
            Class::Subd as u16,
            key,
            c.cage.faces() * SUBD_BYTES_PER_FACE,
            Work::Subd(Arc::new(c.cage), c.materials, c.settings),
        ));
    }
    if let Some(key) = skeleton.1 {
        model_key = model_key.hash(key);
    }
    for m in materials {
        model_key = model_key.hash(m.1);
    }
    let deps = units.iter().map(|u| u.name.clone()).collect();
    units.push(NewUnit {
        name: model_name,
        class: Class::Model as u16,
        key: model_key.finish(),
        working_set: 0,
        triangles: 0,
        work: Work::Model(materials.iter().map(|m| m.0).collect(), skeleton.0),
        deps,
    });
    units
}

/// A cage, its materials and a model from a library caller's cage.
fn subd_units(
    name: &str,
    cage: &CageInput,
    materials: &[Material],
    settings: &SubdSettings,
) -> Result<Vec<NewUnit>> {
    subd::check(cage, materials.len()).map_err(|message| Error::Mesh {
        asset: name.into(),
        message,
    })?;
    let mut units = Vec::new();
    let mut ids = Vec::new();
    for (i, m) in materials.iter().enumerate() {
        let unit = material_unit(format!("{name}/material/{i}"), m);
        ids.push((AssetId::from_name(&unit.name), unit.key));
        units.push(unit);
    }
    let part = SubdPart {
        name: format!("{name}/subd"),
        cage: CageData {
            cage: cage.clone(),
            skin: None,
        },
        materials: materials.to_vec(),
        settings: settings.clone(),
    };
    units.extend(geometry_units(
        name.into(),
        None,
        Some(part),
        &ids,
        (AssetId::NONE, None),
    ));
    Ok(units)
}

/// One channel of every mip (all of them, or mip 0 only) of a cooked texture entry.
fn texture_mips(arc: &Archive, id: AssetId, all: bool, wrap: [Wrap; 2]) -> Result<Vec<Raster>> {
    let e = arc.expect(id, Class::Texture as u16, orr_pack::texture::SCHEMA)?;
    let blobs = arc.blobs(e);
    let mut head = vec![0; blobs[0].raw_len as usize];
    arc.read(&blobs[0], &mut head)?;
    let info = TextureInfo::parse(&head)?;
    let h = info.header;
    let mips = if all { u32::from(h.mip_count) } else { 1 };
    let mut out = Vec::with_capacity(mips as usize);
    let mut blob = (usize::MAX, Vec::new());
    for mip in 0..mips {
        let b = h.mip_blob(mip);
        if blob.0 != b {
            blob = (b, vec![0; blobs[b].raw_len as usize]);
            arc.read(&blobs[b], &mut blob.1)?;
        }
        let bytes = info.mip_texels(&blob.1, mip, 0)?;
        let [w, hh] = h.mip_size(mip);
        let texels = match h.vk_format {
            orr_pack::texture::vk::R16_UNORM => bytes
                .chunks_exact(2)
                .map(|c| f32::from(u16::from_le_bytes([c[0], c[1]])) / 65535.0)
                .collect(),
            f => texture::decode_blocks(f, w, hh, &bytes)
                .map_err(|_| orr_pack::Error::Malformed("texture format the mesh cook reads"))?
                .px
                .iter()
                .map(|p| p[0])
                .collect(),
        };
        out.push(Raster {
            w,
            h: hh,
            texels,
            wrap,
        });
    }
    Ok(out)
}

/// Cooks one unit into a one-unit archive: in the cache when there is one, else in memory.
/// With a `gdeflate` codec, its payload blobs are GDeflate pages.
fn cook_unit(
    shared: &Shared,
    name: &str,
    class: u16,
    key: Hash32,
    gdeflate: Option<Arc<dyn GDeflate>>,
    work: Work<'_>,
) -> Result<(From, Detail)> {
    let mut detail = Detail::default();
    let path = shared.cache.as_ref().map(|c| c.path(key));
    let mut w = ArchiveWriter::create(path.clone().map_or(Target::Memory, Target::File))?;
    w.set_zstd_level(ZSTD_LEVEL);
    w.set_gdeflate(gdeflate);
    match work {
        Work::Blobs { schema, blobs } => {
            let blobs: Vec<BlobIn> = blobs
                .iter()
                .map(|(b, m)| if *m { mount(b) } else { BlobIn::new(b) })
                .collect();
            w.add(name, class, schema, key, &blobs)?;
        }
        Work::Mesh(data, settings, maps) => {
            let cooked = mesh::cook(name, &data, &settings, &maps)?;
            drop((data, maps));
            detail.warnings = cooked.warnings;
            let info = cooked.info.to_bytes();
            let pinned = cooked.info.header.pinned_pages as usize;
            let mut blobs = vec![mount(&info)];
            blobs.extend(cooked.pages.iter().enumerate().map(|(i, p)| BlobIn {
                mount: i < pinned,
                ..BlobIn::new(p)
            }));
            blobs.extend(cooked.node_pages.iter().enumerate().map(|(i, p)| BlobIn {
                mount: i == 0,
                ..BlobIn::new(p)
            }));
            w.add(name, class, orr_pack::mesh::SCHEMA, key, &blobs)?;
        }
        Work::Texture(job, images) => {
            let built = job.run(&images)?;
            drop(images);
            texture::add(&mut w, name, key, &built)?;
        }
        Work::Neural(set, images) => {
            let built = set.run(&images)?;
            drop(images);
            neural::add(&mut w, name, key, &built)?;
            detail.neural = Some(built.report);
        }
        Work::TextureFile(path, hash, settings) => {
            let bytes = state::read_checked(&path, hash)?;
            let built = texture::cook_file(name, &bytes, &settings)?;
            texture::add(&mut w, name, key, &built)?;
        }
        Work::Subd(data, materials, settings) => {
            let cooked =
                subd::cook(&data, &materials, &settings).map_err(|message| Error::Mesh {
                    asset: name.into(),
                    message,
                })?;
            let info = cooked.info.to_bytes();
            let mut blobs = vec![mount(&info)];
            blobs.extend(cooked.pages.iter().map(|p| BlobIn::new(p)));
            w.add(name, class, orr_pack::subd::SCHEMA, key, &blobs)?;
        }
        Work::Model(..) | Work::Same => unreachable!("finished on the coordinator"),
        Work::Instances(src, cell_log2, catalog, _) => {
            instances::cook(&mut w, name, &src, cell_log2, &catalog, key)?;
        }
        Work::Raster(plan) => {
            let scratch = raster::scratch_dir(shared.cache.as_ref().map(Cache::dir));
            detail.raster = Some(raster::cook(&mut w, name, key, &plan, &scratch)?);
        }
    }
    let from = match (w.finish()?, path) {
        (Some(bytes), _) => From::Memory(Arc::new(Archive::from_memory(bytes)?)),
        (None, Some(path)) => From::Cache(path),
        (None, None) => unreachable!("a file target has a path"),
    };
    Ok((from, detail))
}
