//! The cook run: sources become units (one cook job each, usually one entry), units link
//! from the archive being replaced or the cache when their keys match, the rest cook on the
//! cook's pool, and every archive is written in a fixed order.
//!
//! Scheduling: jobs start largest first while the estimated working set of the running
//! jobs stays under [`CookOptions::working_set`] (a job over it runs alone). Finished units
//! wait for their turn to be written; while the bytes waiting exceed
//! [`CookOptions::reorder_window`], only the jobs the next unwritten units wait on start.

use std::cmp::Reverse;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::Instant;

use orr_pack::model::{ModelHeader, ModelInfo};
use orr_pack::{Aabb, Archive, ArchiveWriter, AssetId, BlobIn, Class, Hash32, Span, Target};

use crate::cache::Cache;
use crate::instances::{self, Catalog, ModelFacts};
use crate::key::{self, Key, ZSTD_LEVEL};
use crate::mesh::{self, MeshData};
use crate::raster::{self, RasterReport};
use crate::report::{self, ArchiveReport, AssetReport, Outcome};
use crate::state::{self, Input, Memo, Record};
use crate::texture::material::{ImageFacts, ImageFile, Packer, TextureJob, image_facts};
use crate::texture::{self, ENCODERS};
use crate::{
    CookOptions, Cooked, Error, GltfSettings, Instances, Manifest, MeshSettings, Result, Source,
    TextureSettings, mount,
};

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
    /// For a done mesh: its bounds and what its model's instance pages need.
    mesh: Option<(Aabb, ModelFacts)>,
}

enum Work<'m> {
    Blobs {
        schema: u16,
        blobs: Vec<(Vec<u8>, bool)>,
    },
    Mesh(Arc<MeshData>, MeshSettings),
    Texture(TextureJob, Arc<Vec<ImageFile>>),
    TextureFile(PathBuf, Hash32, TextureSettings),
    /// A model of the mesh in `deps[0]`, these materials and this skeleton.
    Model(Vec<AssetId>, AssetId),
    /// The same key as `deps[0]`, whose result it shares.
    Same,
    /// The placements, the cell size, the catalog without models, and the mesh unit of
    /// each prototype's model.
    Instances(&'m Instances, u8, Catalog, Vec<(String, usize)>),
    Raster(raster::Plan),
}

enum State {
    Waiting,
    Running,
    Done(Done),
    Written,
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

/// A glTF source to import.
struct Import<'m> {
    archive: usize,
    asset: usize,
    name: &'m str,
    path: &'m Path,
    settings: &'m GltfSettings,
    identity: Hash32,
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

struct Imported {
    files: Vec<Input>,
    units: Vec<NewUnit>,
}

/// A cooked unit: where its entries are, the job's seconds and a raster set's report.
type UnitResult = (From, f64, Option<RasterReport>);

enum Msg {
    Import(usize, std::thread::Result<Result<Imported>>),
    Unit(usize, std::thread::Result<Result<UnitResult>>),
}

struct Run<'m, 's> {
    manifest: &'m Manifest,
    opts: &'m CookOptions,
    shared: &'s Shared,
    units: Vec<Unit<'m>>,
    by_name: HashMap<String, usize>,
    old: Vec<Option<Old>>,
    /// Per archive: units in write order, the next one to write, and the writer (none
    /// while the archive matches the one it replaces).
    order: Vec<Vec<usize>>,
    cursor: Vec<usize>,
    writers: Vec<Option<ArchiveWriter>>,
    /// Per archive: records of its glTF sources and every input file, for the cookstate.
    records: Vec<Vec<Record>>,
    files: Vec<Vec<PathBuf>>,
    imports: Vec<Import<'m>>,
    instances: Vec<(usize, usize, &'m Instances)>,
    assets: Vec<AssetReport>,
    /// Reports of cooked raster sets, by unit, until written.
    rasters: HashMap<usize, RasterReport>,
    /// Units whose dependencies are done, largest first; models and shared results apart,
    /// as they finish on the coordinator.
    ready: BTreeSet<(Reverse<u64>, usize)>,
    local: Vec<usize>,
    pending: Vec<usize>,
    dependents: Vec<Vec<usize>>,
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
            old: Vec::with_capacity(n),
            order: vec![Vec::new(); n],
            cursor: vec![0; n],
            writers: Vec::new(),
            records: vec![Vec::new(); n],
            files: vec![Vec::new(); n],
            imports: Vec::new(),
            instances: Vec::new(),
            assets: Vec::new(),
            rasters: HashMap::new(),
            ready: BTreeSet::new(),
            local: Vec::new(),
            pending: Vec::new(),
            dependents: Vec::new(),
            window: 0,
        };
        for (a, spec) in manifest.archives.iter().enumerate() {
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
            for asset in &spec.assets {
                run.source(a, asset, &previous)?;
            }
        }
        Ok(run)
    }

    /// Turns a source into units, or into a pending import or instance set.
    fn source(
        &mut self,
        a: usize,
        asset: &'m crate::AssetSpec,
        previous: &HashMap<Hash32, Record>,
    ) -> Result<()> {
        let name = asset.name.as_str();
        let index = self.assets.len();
        self.assets.push(AssetReport {
            name: name.into(),
            archive: a,
            ..AssetReport::default()
        });
        match &asset.source {
            Source::Gltf { path, settings } => {
                let identity = Key::new(0u16, name)
                    .hash(key::salt())
                    .str(&path.to_string_lossy())
                    .debug(settings)
                    .finish();
                if let Some(record) = previous.get(&identity)
                    && self.link_record(a, index, record)?
                {
                    return Ok(());
                }
                self.imports.push(Import {
                    archive: a,
                    asset: index,
                    name,
                    path,
                    settings,
                    identity,
                });
            }
            Source::Texture { path, settings } => {
                let hash = self.shared.memo.hash(path)?;
                self.input(a, index, path);
                let key = Key::new(Class::Texture, name)
                    .str(ENCODERS)
                    .debug(settings)
                    .hash(hash)
                    .finish();
                let working_set = texture_file_working_set(path);
                self.add(Unit {
                    archive: a,
                    asset: index,
                    name: name.into(),
                    class: Class::Texture as u16,
                    key,
                    work: Some(Work::TextureFile(path.clone(), hash, settings.clone())),
                    deps: Vec::new(),
                    working_set,
                    triangles: 0,
                    state: State::Waiting,
                    mesh: None,
                });
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
                self.add(Unit {
                    archive: a,
                    asset: index,
                    name: name.into(),
                    class: *class,
                    key,
                    work: Some(Work::Blobs {
                        schema: *schema,
                        blobs: vec![(bytes.clone(), false)],
                    }),
                    deps: Vec::new(),
                    working_set: bytes.len() as u64,
                    triangles: 0,
                    state: State::Waiting,
                    mesh: None,
                });
            }
            Source::Raster(set) => {
                let planned = raster::plan(name, set, &self.shared.memo)?;
                for f in &planned.files {
                    self.input(a, index, f);
                }
                self.add(Unit {
                    archive: a,
                    asset: index,
                    name: name.into(),
                    class: Class::RasterSet as u16,
                    key: planned.key,
                    work: Some(Work::Raster(planned.plan)),
                    deps: Vec::new(),
                    working_set: raster::WORKING_SET,
                    triangles: 0,
                    state: State::Waiting,
                    mesh: None,
                });
            }
            Source::Instances(src) => self.instances.push((a, index, src)),
        }
        Ok(())
    }

    fn input(&mut self, a: usize, asset: usize, path: &Path) {
        self.files[a].push(path.to_owned());
        self.assets[asset].inputs.push(path.to_owned());
    }

    fn add(&mut self, unit: Unit<'m>) -> usize {
        let i = self.units.len();
        self.by_name.insert(unit.name.clone(), i);
        self.order[unit.archive].push(i);
        self.units.push(unit);
        i
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
            let (name, class) = self.with_archive(a, &from, |arc| {
                let e = &arc.entries()[entries[0]];
                (arc.name(e).to_owned(), e.class)
            })?;
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
        let archive = self.shared.cache.as_ref()?.find(key)?;
        let entries = (0..archive.entries().len()).collect();
        Some((
            From::Cache(self.shared.cache.as_ref()?.path(key)),
            entries,
            Outcome::Cache,
        ))
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
            let mesh = (class == Class::Mesh as u16)
                .then(|| mesh_facts(arc, &arc.entries()[entries[0]]))
                .flatten();
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

    /// Imports every changed glTF source, largest file first.
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
                let (name, path, settings) = (imp.name, imp.path, imp.settings);
                scope.spawn(move |_| {
                    let r = catch_unwind(AssertUnwindSafe(|| {
                        import_gltf(name, path, settings, &shared.memo)
                    }));
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

    fn add_import(&mut self, i: usize, imported: Imported) {
        let (a, asset, identity) = {
            let imp = &self.imports[i];
            (imp.archive, imp.asset, imp.identity)
        };
        let mut keys = Vec::new();
        let mut deps = Vec::new();
        for nu in imported.units {
            keys.push(nu.key);
            deps.push(nu.deps);
            self.add(Unit {
                archive: a,
                asset,
                name: nu.name,
                class: nu.class,
                key: nu.key,
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
            self.units[u].deps = names.iter().map(|n| self.by_name[n]).collect();
        }
        for f in &imported.files {
            self.input(a, asset, &f.path);
        }
        self.records[a].push(Record {
            identity,
            files: imported.files,
            units: keys,
        });
    }

    /// Keys and dependencies of instance sources, from the keys of the models and
    /// materials they name.
    fn instance_units(&mut self) -> Result<()> {
        let by_id: HashMap<AssetId, usize> = self
            .by_name
            .iter()
            .map(|(n, &u)| (AssetId::from_name(n), u))
            .collect();
        let materials: HashSet<String> = self
            .units
            .iter()
            .filter(|u| u.class == Class::Material as u16)
            .map(|u| u.name.clone())
            .collect();
        for (a, asset, src) in std::mem::take(&mut self.instances) {
            let name = self.assets[asset].name.clone();
            let cell_log2 = self.manifest.archives[a].cell_log2;
            let mut key =
                Key::new(Class::InstancePage, &name).hash(instances::input_hash(src, cell_log2));
            let mut meshes = Vec::new();
            for p in &src.prototypes {
                let model = self
                    .by_name
                    .get(&p.model)
                    .copied()
                    .filter(|&u| self.units[u].class == Class::Model as u16)
                    .ok_or_else(|| Error::UnknownModel {
                        asset: name.clone(),
                        model: p.model.clone(),
                    })?;
                key = key.hash(self.units[model].key);
                meshes.push((p.model.clone(), self.mesh_of(model, &by_id)?));
                for s in &p.sections {
                    if let Some(m) = &s.material {
                        let u = self
                            .by_name
                            .get(m)
                            .copied()
                            .filter(|&u| self.units[u].class == Class::Material as u16)
                            .ok_or_else(|| Error::UnknownMaterial {
                                asset: name.clone(),
                                material: m.clone(),
                            })?;
                        key = key.hash(self.units[u].key);
                    }
                }
            }
            let mut deps: Vec<usize> = meshes.iter().map(|m| m.1).collect();
            deps.sort_unstable();
            deps.dedup();
            let catalog = Catalog {
                materials: materials.clone(),
                ..Catalog::default()
            };
            self.add(Unit {
                archive: a,
                asset,
                name,
                class: Class::InstancePage as u16,
                key: key.finish(),
                work: Some(Work::Instances(src, cell_log2, catalog, meshes)),
                deps,
                working_set: src.records.len() as u64 * 256,
                triangles: 0,
                state: State::Waiting,
                mesh: None,
            });
        }
        Ok(())
    }

    /// The mesh unit of a model unit.
    fn mesh_of(&self, model: usize, by_id: &HashMap<AssetId, usize>) -> Result<usize> {
        let u = &self.units[model];
        if let Some(Work::Model(..)) = &u.work {
            return Ok(u.deps[0]);
        }
        let State::Done(d) = &u.state else {
            unreachable!("a model unit without work is linked");
        };
        let mesh = self.with_archive(u.archive, &d.from, |arc| {
            let e = &arc.entries()[d.entries[0]];
            let mut head = vec![0; arc.blobs(e)[0].raw_len as usize];
            arc.read(&arc.blobs(e)[0], &mut head)?;
            Ok::<_, Error>(ModelInfo::parse(&head)?.header.mesh)
        })??;
        by_id
            .get(&mesh)
            .copied()
            .ok_or_else(|| Error::UnknownModel {
                asset: u.name.clone(),
                model: format!("mesh {mesh}"),
            })
    }

    /// Links every unit whose key is in the replaced archive or the cache; the others wait
    /// for their dependencies.
    fn link_all(&mut self) -> Result<()> {
        self.pending = vec![0; self.units.len()];
        self.dependents = vec![Vec::new(); self.units.len()];
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
            let (from, seconds, raster) = r.unwrap_or_else(|p| resume_unwind(p))?;
            if let Some(report) = raster {
                self.rasters.insert(u, report);
            }
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

    fn spawn(&mut self, scope: &rayon::Scope<'m>, tx: &mpsc::Sender<Msg>, u: usize)
    where
        's: 'm,
    {
        let mut work = self.units[u].work.take().expect("waiting units have work");
        if let Work::Instances(_, _, catalog, meshes) = &mut work {
            for (model, mesh) in meshes.drain(..) {
                if let Some((_, facts)) = self.units[mesh].mesh {
                    catalog.models.insert(model, facts);
                }
            }
        }
        self.units[u].state = State::Running;
        let unit = &self.units[u];
        let (name, class, key) = (unit.name.clone(), unit.class, unit.key);
        let (tx, shared) = (tx.clone(), self.shared);
        scope.spawn(move |_| {
            let r = catch_unwind(AssertUnwindSafe(|| {
                let start = Instant::now();
                let (from, raster) = cook_unit(shared, &name, class, key, work)?;
                Ok((from, start.elapsed().as_secs_f64(), raster))
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
        let mesh = self.units[u].deps[0];
        let (bounds, _) = self.units[mesh]
            .mesh
            .ok_or(orr_pack::Error::Malformed("mesh header"))?;
        let bytes = ModelInfo {
            header: ModelHeader {
                kind: ModelHeader::MODEL,
                _pad: [0; 3],
                bounds,
                mesh: AssetId::from_name(&self.units[mesh].name),
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
        let (from, _) = cook_unit(self.shared, &unit.name, unit.class, unit.key, work)?;
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
                let State::Done(d) = std::mem::replace(&mut self.units[u].state, State::Written)
                else {
                    unreachable!("checked above");
                };
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
                if let (Some(r), Some(first)) = (self.rasters.remove(&u), reports.first_mut()) {
                    first.raster = Some(r);
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
            assets,
            ..
        } = self;
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

/// Imports a glTF file and plans its units: the mesh, the packed textures, the materials,
/// the skeleton and clips, and the model.
fn import_gltf(name: &str, path: &Path, settings: &GltfSettings, memo: &Memo) -> Result<Imported> {
    let mut src = crate::gltf::import(name, path, memo, &settings.clips)?;
    let mut units = Vec::new();
    let mut skeleton = (AssetId::NONE, None);
    if let Some(info) = src.skeleton.take() {
        let skeleton_name = format!("{name}/skeleton");
        let bytes = info.to_bytes();
        let key = Key::new(Class::Skeleton, &skeleton_name)
            .bytes(&bytes)
            .finish();
        skeleton = (AssetId::from_name(&skeleton_name), Some(key));
        units.push(NewUnit {
            name: skeleton_name,
            class: Class::Skeleton as u16,
            key,
            working_set: bytes.len() as u64,
            triangles: 0,
            work: Work::Blobs {
                schema: orr_pack::skeleton::SCHEMA,
                blobs: vec![(bytes, true)],
            },
            deps: Vec::new(),
        });
        for (clip_name, mut clip) in std::mem::take(&mut src.clips) {
            clip.header.skeleton = skeleton.0;
            let clip_name = format!("{name}/clip/{clip_name}");
            let (head, samples) = clip.to_bytes();
            let key = Key::new(Class::Clip, &clip_name)
                .bytes(&head)
                .bytes(&samples)
                .finish();
            units.push(NewUnit {
                name: clip_name,
                class: Class::Clip as u16,
                key,
                working_set: (head.len() + samples.len()) as u64,
                triangles: 0,
                work: Work::Blobs {
                    schema: orr_pack::clip::SCHEMA,
                    blobs: vec![(head, true), (samples, false)],
                },
                deps: Vec::new(),
            });
        }
    }

    let mesh_name = format!("{name}/mesh");
    let mesh_key = Key::new(Class::Mesh, &mesh_name)
        .debug(&settings.mesh)
        .hash(src.mesh.hash())
        .finish();

    let mut used: Vec<usize> = src
        .materials
        .iter()
        .flat_map(|(_, _, t)| {
            [
                t.base_color,
                t.metal_rough,
                t.normal.map(|n| n.0),
                t.occlusion.map(|o| o.0),
                t.emissive,
            ]
        })
        .flatten()
        .map(|r| r.image)
        .collect();
    used.sort_unstable();
    used.dedup();
    let facts: HashMap<usize, ImageFacts> = image_facts(name, &src.images, &used)?;
    let mut packer = Packer::new(name, &src.images, &facts, &settings.textures);
    let mut materials = Vec::new();
    let mut model_key = Key::new(Class::Model, name).hash(mesh_key);
    if let Some(key) = skeleton.1 {
        model_key = model_key.hash(key);
    }
    for (suffix, m, textures) in &src.materials {
        let mut m = *m;
        packer.material(suffix, &mut m, textures)?;
        let material_name = format!("{name}/material/{suffix}");
        let bytes = bytemuck::bytes_of(&m).to_vec();
        let key = Key::new(Class::Material, &material_name)
            .bytes(&bytes)
            .finish();
        model_key = model_key.hash(key);
        materials.push(AssetId::from_name(&material_name));
        units.push(NewUnit {
            name: material_name,
            class: Class::Material as u16,
            key,
            working_set: bytes.len() as u64,
            triangles: 0,
            work: Work::Blobs {
                schema: orr_pack::material::SCHEMA,
                blobs: vec![(bytes, true)],
            },
            deps: Vec::new(),
        });
    }
    let jobs = std::mem::take(&mut packer.jobs);
    drop(packer);
    let images = Arc::new(src.images);
    for job in jobs {
        units.push(NewUnit {
            name: job.name.clone(),
            class: Class::Texture as u16,
            key: job.key,
            working_set: job.working_set(&images),
            triangles: 0,
            work: Work::Texture(job, Arc::clone(&images)),
            deps: Vec::new(),
        });
    }
    let triangles = src.mesh.triangles();
    units.push(NewUnit {
        name: mesh_name.clone(),
        class: Class::Mesh as u16,
        key: mesh_key,
        working_set: triangles * MESH_BYTES_PER_TRIANGLE,
        triangles,
        work: Work::Mesh(Arc::new(src.mesh), settings.mesh.clone()),
        deps: Vec::new(),
    });
    units.push(NewUnit {
        name: name.into(),
        class: Class::Model as u16,
        key: model_key.finish(),
        working_set: 0,
        triangles: 0,
        work: Work::Model(materials, skeleton.0),
        deps: vec![mesh_name],
    });
    Ok(Imported {
        files: src.files,
        units,
    })
}

/// Cooks one unit into a one-unit archive: in the cache when there is one, else in memory.
fn cook_unit(
    shared: &Shared,
    name: &str,
    class: u16,
    key: Hash32,
    work: Work<'_>,
) -> Result<(From, Option<RasterReport>)> {
    let mut detail = None;
    let path = shared.cache.as_ref().map(|c| c.path(key));
    let mut w = ArchiveWriter::create(path.clone().map_or(Target::Memory, Target::File))?;
    w.set_zstd_level(ZSTD_LEVEL);
    match work {
        Work::Blobs { schema, blobs } => {
            let blobs: Vec<BlobIn> = blobs
                .iter()
                .map(|(b, m)| if *m { mount(b) } else { BlobIn::new(b) })
                .collect();
            w.add(name, class, schema, key, &blobs)?;
        }
        Work::Mesh(data, settings) => {
            let cooked = mesh::cook(name, &data, &settings)?;
            drop(data);
            let info = cooked.info.to_bytes();
            let pinned = cooked.info.header.pinned_pages as usize;
            let mut blobs = vec![mount(&info)];
            blobs.extend(cooked.pages.iter().enumerate().map(|(i, p)| BlobIn {
                mount: i < pinned,
                ..BlobIn::new(p)
            }));
            w.add(name, class, orr_pack::mesh::SCHEMA, key, &blobs)?;
        }
        Work::Texture(job, images) => {
            let built = job.run(&images)?;
            drop(images);
            texture::add(&mut w, name, key, &built)?;
        }
        Work::TextureFile(path, hash, settings) => {
            let bytes = state::read_checked(&path, hash)?;
            let built = texture::cook_file(name, &bytes, &settings)?;
            texture::add(&mut w, name, key, &built)?;
        }
        Work::Model(..) | Work::Same => unreachable!("finished on the coordinator"),
        Work::Instances(src, cell_log2, catalog, _) => {
            instances::cook(&mut w, name, src, cell_log2, &catalog, key)?;
        }
        Work::Raster(plan) => {
            let scratch = raster::scratch_dir(shared.cache.as_ref().map(Cache::dir));
            detail = Some(raster::cook(&mut w, name, key, &plan, &scratch)?);
        }
    }
    let from = match (w.finish()?, path) {
        (Some(bytes), _) => From::Memory(Arc::new(Archive::from_memory(bytes)?)),
        (None, Some(path)) => From::Cache(path),
        (None, None) => unreachable!("a file target has a path"),
    };
    Ok((from, detail))
}
