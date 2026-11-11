//! The cook run: sources become units (one cook job each, usually one entry), units link
//! from the archive being replaced or the cache when their keys match, the rest cook on
//! threads of their own with their parallel loops on the rayon pool, and every archive is
//! written in a fixed order.
//!
//! Scheduling: the jobs the next unwritten units wait on start first, then the rest by the
//! largest planned job among them and the units waiting on them, so the textures of the
//! largest meshes cook first, each while the estimated working set of the running jobs stays
//! under [`CookOptions::working_set`] with it (a job over it runs alone); no job starts past
//! one that does not fit. Finished units wait for their turn to be written, which a writer
//! thread does while the coordinator starts jobs. While the bytes waiting in memory exceed
//! [`CookOptions::reorder_window`], jobs of a file archive without a cache write their units
//! to `<archive>.units/` beside it, removed once written, and the jobs of an archive in
//! memory start only when the next unwritten units wait on them.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::thread::Scope;
use std::time::Instant;

use ctt::Quality;
use orr_math::{Vec2, Vec3, Vec4};

use orr_pack::material::{Material, slot};
use orr_pack::model::{ModelHeader, ModelInfo};
use orr_pack::texture::TextureInfo;
use orr_pack::tiles::TextureTiles;
use orr_pack::{
    Aabb, Archive, ArchiveWriter, AssetId, BlobIn, Class, GDeflate, Hash32, Span, Target,
};

use crate::assembly::{self, Assembly};
use crate::cache::Cache;
use crate::decoded::Decoded;
use crate::import::{PlacedModel, Scene};
use crate::instances::{self, Catalog, ModelFacts, PageFrame};
use crate::key::{self, Key};
use crate::mesh::maps::{
    Displacement, Emission, EmissionMap, Map, NormalMap, Opacity, OpacityMap, OpacityRead, Passes,
    Raster, Reach, Tiles,
};
use crate::mesh::{
    self, DisplacementSpec, DisplacementTexture, EmissionSpec, MapRead, MapSpec, MapTexture,
    MeshData, MeshInput, MeshSource, NormalSpec, OpacitySpec, PartData, Pivots, SectionData,
    SectionMaps,
};
use crate::raster::{self, RasterReport};
use crate::report::{
    self, ArchiveReport, AssemblyReport, AssemblyUnitReport, AssetReport, Outcome, TangentReport,
};
use crate::state::{self, Input, Memo, Record, RecordUnit};
use crate::subd::{self, CageData, CageInput, SubdSettings};
use crate::texture::material::{
    ImageData, ImageFile, ImageReads, ImageRules, Packer, TextureJob, TextureRules, image_facts,
    settle_tiles, stand_in,
};
use crate::texture::neural::{self, NeuralJob};
use crate::texture::{self, Img, Wrap};
use crate::volume::{self, VolumeReport};
use crate::{
    CookOptions, Cooked, Error, GltfSettings, Instances, Linear, Manifest, MeshSettings, Placement,
    PrototypeSpec, Result, Source, TextureSettings, UsdSettings, mount,
};

/// Each of the [`DECODED_CACHES`] caches of decoded data holds up to this part of the working
/// set, which the jobs' estimates leave free.
const DECODED_SHARE: u64 = 16;
const DECODED_CACHES: u64 = 4;

/// Peak bytes of a subdivision cage cook per cage face, an estimate.
const SUBD_BYTES_PER_FACE: u64 = 16 << 10;

/// Estimated peak bytes of an instance unit's cook per placement.
const INSTANCE_BYTES_PER_PLACEMENT: u64 = 256;

/// The estimates that plan a scene source's units, which place them in the payload: its
/// record carries their planned bytes, so these enter the source's identity, and a warm cook
/// drops records planned otherwise and writes the payload as a clean cook does. Each
/// estimate's constants, then a version to raise when an estimate changes in another way.
const PLAN: &[&[u64]] = &[
    &[
        SUBD_BYTES_PER_FACE,
        INSTANCE_BYTES_PER_PLACEMENT,
        MESH_BYTES_PER_TRIANGLE,
        DAG_BYTES_PER_TRIANGLE,
        DAG_BYTES_PER_VERTEX,
        SUBSURFACE_BYTES_PER_TRIANGLE,
    ],
    &texture::material::ESTIMATE,
    &[neural::BYTES_PER_CHANNEL_TEXEL],
    &[1],
];

/// Peak bytes of a discrete-level mesh cook per source triangle besides its maps, an upper
/// bound.
const MESH_BYTES_PER_TRIANGLE: u64 = 480;

/// Peak bytes of a DAG mesh cook past its read besides its maps, per source triangle and per
/// source vertex, as vertices cost apart from triangles: fitted to the job peaks of a large
/// scene's meshes that peaked past the read, a byte under a peak weighed far above a byte
/// over it, since a job over its estimate takes working set others count on.
const DAG_BYTES_PER_TRIANGLE: u64 = 74;
const DAG_BYTES_PER_VERTEX: u64 = 196;

/// Peak bytes a DAG mesh with subsurface sections adds per source triangle, as it cooks its
/// groups in one batch for thickness to read every cluster: the least that covers meshes of
/// few and of many vertices per triangle cooked with a subsurface material.
const SUBSURFACE_BYTES_PER_TRIANGLE: u64 = 163;

/// Bytes a mesh cook holds per decoded texel channel of the maps its sections read
/// ([`map_bytes`]): an f32 each. Textures decode one at a time, each taking
/// [`MAP_DECODE_BYTES_PER_CHANNEL`] more per channel while it decodes, while the mesh is read
/// and before the geometry's peak.
const MAP_BYTES_PER_CHANNEL: u64 = 4;
const MAP_DECODE_BYTES_PER_CHANNEL: u64 = 9;

/// Peak bytes of reading a mesh per source triangle and per source vertex, which its maps
/// decode beside: fitted to the read peaks of a large scene's meshes as the DAG constants are.
const LOAD_BYTES_PER_TRIANGLE: u64 = 131;
const LOAD_BYTES_PER_VERTEX: u64 = 66;

pub(crate) fn cook(manifest: &Manifest, opts: &CookOptions) -> Result<Cooked> {
    let start = Instant::now();
    crate::alloc::install_meshopt_arena();
    crate::profile::start();
    // A pool built earlier in the process stays; the cook runs on whichever is there.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.jobs)
        .start_handler(|i| crate::profile::name_thread(&format!("rayon {i}")))
        .build_global();
    let shared = Shared {
        memo: Memo::default(),
        cache: opts.cache.as_deref().map(Cache::new).transpose()?,
        quality: opts.quality,
        zstd_level: opts
            .zstd_level
            .unwrap_or_else(|| crate::quality::zstd_level(opts.quality)),
        images: Decoded::new(opts.working_set / DECODED_SHARE, Img::bytes),
        opacity: Decoded::new(opts.working_set / DECODED_SHARE, OpacityMap::bytes),
        emission: Decoded::new(opts.working_set / DECODED_SHARE, EmissionMap::bytes),
        normals: Decoded::new(opts.working_set / DECODED_SHARE, NormalMap::bytes),
    };
    let threads = rayon::current_num_threads();
    let mut run = Run::new(manifest, opts, &shared)?;
    // Jobs run on threads of their own and their parallel loops on the pool, so a pool
    // thread waiting on one job's loop never starts another job on its stack.
    if let Err(e) = std::thread::scope(|scope| run.run(scope, threads)) {
        run.remove_scratch();
        return Err(e);
    }
    let mut cooked = run.finish()?;
    cooked.report.seconds = start.elapsed().as_secs_f64();
    cooked.report.heap_peak = crate::job::total_peak();
    if let Some(path) = &opts.report {
        let json = serde_json::to_vec_pretty(&cooked.report).expect("report serializes");
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
    quality: Quality,
    zstd_level: i32,
    /// Source images by content hash and mip, for the texture jobs that read them.
    images: Decoded<Img>,
    /// Cooked opacity mips by cook key and cutoff, for the mesh jobs that read them.
    opacity: Decoded<OpacityMap>,
    /// Cooked emission luminance by cook key, for the mesh jobs that read it.
    emission: Decoded<EmissionMap>,
    /// Cooked normal maps' mip 0 by cook key, for the tangent checks that read them.
    normals: Decoded<NormalMap>,
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
    /// The working set planned before anything cooked, which places the unit in the payload.
    planned: u64,
    /// Source triangles of a mesh, for the report.
    triangles: u64,
    state: State,
    /// For a done mesh or subdivision cage: its bounds and what its model's instance pages
    /// need.
    mesh: Option<(Aabb, ModelFacts)>,
    /// For a done texture: its header, which the estimates of the meshes reading it count.
    texture: Option<orr_pack::texture::TextureHeader>,
}

enum Work<'m> {
    Blobs {
        schema: u16,
        blobs: Vec<(Vec<u8>, bool)>,
    },
    /// The mesh, its settings and the archives holding the cooked texture units its
    /// sections read, by unit name, filled when it starts.
    Mesh(MeshSource, MeshSettings, HashMap<String, Arc<Archive>>),
    Texture(Box<TextureJob>, Arc<Vec<ImageFile>>),
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
    Volume(volume::Plan<'m>),
}

enum State {
    Waiting,
    Running,
    Done(Done),
    /// Where its entries are, kept while a mesh that reads them has not finished.
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
    /// A unit archive in the archive's scratch directory, removed once written and read.
    Scratch(PathBuf),
    /// The archive the cook replaces.
    Old,
}

/// The archive a file target replaces, with its entries by cook key.
struct Old {
    archive: Arc<Archive>,
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
struct NewUnit<'m> {
    name: String,
    class: u16,
    key: Hash32,
    work: Work<'m>,
    deps: Vec<String>,
    working_set: u64,
    triangles: u64,
}

impl<'m> NewUnit<'m> {
    /// A unit with no dependencies and no triangles.
    fn leaf(name: &str, class: u16, key: Hash32, working_set: u64, work: Work<'m>) -> Self {
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
    units: Vec<NewUnit<'static>>,
    /// Placements of the instanced models, cooked as instance units `<asset>/instances`, then
    /// `<asset>/instances/<k>` for each further 65536 models and each model with a skeleton.
    instances: Vec<Instances>,
    assemblies: Vec<Assembly>,
    /// For the asset's report and its record.
    warnings: Vec<String>,
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

/// A library mesh waiting for the materials its sections' backs name, which imports plan.
struct PendingMesh<'m> {
    archive: usize,
    asset: usize,
    name: &'m str,
    mesh: &'m MeshInput,
    materials: &'m [Material],
    settings: &'m MeshSettings,
}

/// A cooked unit: where its entries are, the job's seconds and its report details.
type UnitResult = (From, f64, Detail);

/// What a cooked unit adds to its first entry's report.
#[derive(Default)]
struct Detail {
    /// The job's measured peak bytes.
    peak: Option<u64>,
    tangents: Option<mesh::TangentCheck>,
    raster: Option<RasterReport>,
    volume: Option<VolumeReport>,
    neural: Option<report::NeuralReport>,
    warnings: Vec<String>,
}

enum Msg {
    Import(usize, std::thread::Result<Result<Imported>>),
    Unit(usize, std::thread::Result<Result<UnitResult>>),
    /// The writer thread wrote a unit's entries, with their reports.
    Written(usize, std::thread::Result<Result<Vec<report::EntryReport>>>),
}

/// A done unit for the writer thread to copy into its archive and report, in write order.
struct Write {
    unit: usize,
    archive: usize,
    from: WriteFrom,
    entries: Vec<usize>,
    outcome: Outcome,
    seconds: f64,
    working_set: u64,
    triangles: u64,
    detail: Option<Detail>,
    waited: Option<f64>,
}

enum WriteFrom {
    Archive(Arc<Archive>),
    Path(PathBuf),
}

/// The writer thread: copies each unit's entries into its archive's writer, when it has one,
/// and reports them, until the coordinator stops sending.
fn write_units(
    writers: &mut [Option<ArchiveWriter>],
    writes: &mpsc::Receiver<Write>,
    tx: &mpsc::Sender<Msg>,
) {
    crate::profile::name_thread("cook writer");
    for w in writes {
        let unit = w.unit;
        let r = catch_unwind(AssertUnwindSafe(|| write_unit(writers, w)));
        if tx.send(Msg::Written(unit, r)).is_err() {
            return;
        }
    }
}

fn write_unit(writers: &mut [Option<ArchiveWriter>], w: Write) -> Result<Vec<report::EntryReport>> {
    let _z = crate::profile::zone!("archive.write");
    let opened;
    let arc = match &w.from {
        WriteFrom::Archive(arc) => arc.as_ref(),
        WriteFrom::Path(path) => {
            opened = Archive::open(path)?;
            &opened
        }
    };
    let mut reports = Vec::with_capacity(w.entries.len());
    for &e in &w.entries {
        let entry = &arc.entries()[e];
        if let Some(writer) = writers[w.archive].as_mut() {
            writer.copy_entry(arc, entry)?;
        }
        reports.push(report::entry(
            arc,
            entry,
            w.outcome,
            w.seconds,
            w.working_set,
            w.triangles,
        ));
    }
    for r in &mut reports {
        r.peak = w.detail.as_ref().and_then(|d| d.peak);
        r.budget_wait = w.waited;
    }
    if let (Some(d), Some(first)) = (w.detail, reports.first_mut()) {
        if let (Some(m), Some(t)) = (first.mesh.as_mut(), d.tangents) {
            m.tangents = Some(tangent_report(m.vertices, t));
        }
        first.raster = d.raster;
        first.volume = d.volume;
        first.neural = d.neural;
        first.warnings = d.warnings;
    }
    Ok(reports)
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
    /// Per archive: where its units go once the reorder window is full, for a file archive
    /// cooked without a cache.
    scratch: Vec<Option<PathBuf>>,
    /// Units holding each scratch archive (units of equal keys share one) that are not yet
    /// written and read.
    scratch_users: HashMap<PathBuf, usize>,
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
    meshes: Vec<PendingMesh<'m>>,
    /// Assemblies by model unit.
    assemblies: HashMap<usize, Assembly>,
    assets: Vec<AssetReport>,
    /// Report details of cooked units, by unit, until written.
    details: HashMap<usize, Detail>,
    /// Units whose dependencies are done, by [`Self::ready_key`]; models and shared results
    /// apart, as they finish on the coordinator.
    ready: BTreeSet<(Reverse<u64>, usize)>,
    local: Vec<usize>,
    pending: Vec<usize>,
    dependents: Vec<Vec<usize>>,
    /// Per unit: its planned working set or the largest of any unit that waits on it, so the
    /// units the biggest jobs wait on start first.
    priority: Vec<u64>,
    /// Per unit: mesh units that read its texture and have not finished.
    map_users: Vec<usize>,
    /// Per running mesh unit: the texture units it reads, one per read.
    map_reads: HashMap<usize, Vec<usize>>,
    /// Units the writer thread has not yet written, with the bytes of their entries held in
    /// memory.
    writing: HashMap<usize, u64>,
    /// Entry reports of written units.
    reports: HashMap<usize, Vec<report::EntryReport>>,
    /// The unit first in line that does not fit the budget, and since when.
    held: Option<(usize, Instant)>,
    /// Seconds units waited there.
    waited: HashMap<usize, f64>,
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
            scratch: Vec::with_capacity(n),
            scratch_users: HashMap::new(),
            old: Vec::with_capacity(n),
            order: vec![Vec::new(); n],
            cursor: vec![0; n],
            writers: Vec::new(),
            records: vec![Vec::new(); n],
            files: vec![Vec::new(); n],
            imports: Vec::new(),
            instances: Vec::new(),
            meshes: Vec::new(),
            assemblies: HashMap::new(),
            assets: Vec::new(),
            details: HashMap::new(),
            ready: BTreeSet::new(),
            local: Vec::new(),
            pending: Vec::new(),
            dependents: Vec::new(),
            priority: Vec::new(),
            map_users: Vec::new(),
            map_reads: HashMap::new(),
            writing: HashMap::new(),
            reports: HashMap::new(),
            held: None,
            waited: HashMap::new(),
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
            run.scratch.push(match (&spec.output, &shared.cache) {
                (Target::File(path), None) => {
                    let mut dir = path.as_os_str().to_owned();
                    dir.push(".units");
                    Some(PathBuf::from(dir))
                }
                _ => None,
            });
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
                        Old {
                            archive: Arc::new(archive),
                            by_key,
                        }
                    })
                }
                _ => None,
            };
            run.old.push(old);
            let first = run.assets.len();
            run.assets
                .extend(spec.assets.iter().map(|asset| AssetReport {
                    name: asset.name.clone(),
                    archive: a,
                    ..AssetReport::default()
                }));
            for (i, asset) in spec.assets.iter().enumerate() {
                run.source(a, first + i, asset, &previous)?;
            }
        }
        Ok(run)
    }

    /// Turns a source into units, or into a pending import, instance set or mesh, reporting
    /// as asset `index`.
    fn source(
        &mut self,
        a: usize,
        index: usize,
        asset: &'m crate::AssetSpec,
        previous: &HashMap<Hash32, Record>,
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
                let header = texture::file_header(path);
                if header.is_some_and(|h| h.1) {
                    let warning = texture::first_frame_warning(name);
                    self.assets[index].warnings.push(warning);
                }
                let key = Key::new(Class::Texture, name)
                    .str(&texture::encoders(self.shared.quality))
                    .debug(settings)
                    .hash(hash)
                    .finish();
                let unit = NewUnit::leaf(
                    name,
                    Class::Texture as u16,
                    key,
                    texture_file_working_set(path, header.map(|h| h.0), settings),
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
                let planned = raster::plan(name, set, &self.shared.memo, self.shared.quality)?;
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
            Source::Volume { input, settings } => {
                let planned = volume::plan(name, input, settings, &self.shared.memo)?;
                if let Some(f) = &planned.file {
                    self.input(a, index, f);
                }
                let unit = NewUnit::leaf(
                    name,
                    Class::Volume as u16,
                    planned.key,
                    planned.working_set,
                    Work::Volume(planned.plan),
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
            } => self.meshes.push(PendingMesh {
                archive: a,
                asset: index,
                name,
                mesh,
                materials,
                settings,
            }),
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
            .debug(&self.shared.quality)
            .debug(&PLAN)
            .finish();
        let identity = self.stored(a, identity);
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
            .all(|f| match self.shared.memo.hash(&f.path) {
                Ok(h) => h == f.hash,
                Err(_) => f.hash == Input::ABSENT,
            });
        if !unchanged {
            return Ok(false);
        }
        let mut links = Vec::with_capacity(record.units.len());
        for &RecordUnit { key, planned } in &record.units {
            match self.find_link(a, key) {
                Some(l) => links.push((key, planned, l)),
                None => return Ok(false),
            }
        }
        for (key, planned, (from, entries, outcome)) in links {
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
                planned,
                triangles: 0,
                state: State::Waiting,
                mesh: None,
                texture: None,
            });
            self.done(u, from, Some(entries), outcome, 0.0)?;
        }
        for f in &record.files {
            self.input(a, asset, &f.path);
        }
        self.add_assemblies(a, record.assemblies.iter().cloned());
        self.assets[asset]
            .warnings
            .extend_from_slice(&record.warnings);
        self.records[a].push(record.clone());
        Ok(true)
    }

    /// The key archive `a` stores an entry of key `key` under.
    fn stored(&self, a: usize, key: Hash32) -> Hash32 {
        key::stored_as(key, self.shared.zstd_level, self.codecs[a].is_some())
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
            From::Cache(path) | From::Scratch(path) => f(&Archive::open(path)?),
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
        let (entries, mesh, texture) = self.with_archive(a, &from, |arc| {
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
            // Its head is a mount blob, in the index already read.
            let texture = (class == Class::Texture as u16)
                .then(|| texture_head(arc, e.id).map(|(_, info)| info.header))
                .transpose();
            texture.map(|t| (entries, mesh, t))
        })??;
        let memory = match &from {
            From::Memory(arc) => arc.header().file_len,
            From::Scratch(path) => {
                *self.scratch_users.entry(path.clone()).or_default() += 1;
                0
            }
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
        self.units[u].texture = texture;
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
            return;
        }
        // A mesh's maps are cooked once it is ready, so its estimate can count them.
        if let Some(Work::Mesh(source, settings, _)) = &self.units[u].work {
            let a = self.units[u].archive;
            let header = |unit: &str| {
                self.units[self.by_name[&(a, unit.to_owned())]]
                    .texture
                    .expect("a mesh is ready after the textures it reads")
            };
            let maps = map_bytes(source.sections(), settings, header);
            let load = source.triangles() * LOAD_BYTES_PER_TRIANGLE
                + source.vertices() * LOAD_BYTES_PER_VERTEX
                + maps.held
                + maps.decode_channels * MAP_DECODE_BYTES_PER_CHANNEL;
            let geometry = mesh_working_set(source, settings) + maps.held + maps.micromaps;
            self.units[u].working_set = load.max(geometry);
        }
        self.ready.insert(self.ready_key(u));
    }

    /// Unit `u`'s place in [`Self::ready`]: highest priority first.
    fn ready_key(&self, u: usize) -> (Reverse<u64>, usize) {
        (Reverse(self.priority[u]), u)
    }

    fn run<'t>(&mut self, scope: &'t Scope<'t, '_>, threads: usize) -> Result<()>
    where
        'm: 't,
        's: 't,
    {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.import_all(scope, threads, &tx, &rx)?;
        self.mesh_sources()?;
        self.instance_units()?;
        self.link_all()?;
        self.open_writers()?;
        // Half as many jobs as threads: their parallel loops take the rest, and more jobs
        // at once cost more CPU time and memory without finishing sooner.
        self.cook_all(scope, (threads / 2).max(1), &tx, &rx)
    }

    /// Imports every changed scene source, largest file first.
    fn import_all<'t>(
        &mut self,
        scope: &'t Scope<'t, '_>,
        threads: usize,
        tx: &mpsc::Sender<Msg>,
        rx: &mpsc::Receiver<Msg>,
    ) -> Result<()>
    where
        'm: 't,
        's: 't,
    {
        let _z = crate::profile::zone!("import");
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
                scope.spawn(move || {
                    let r = catch_unwind(AssertUnwindSafe(|| import(name, path, kind, shared)));
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

    /// Adds units whose dependencies name units of the same list; returns their keys and
    /// planned working sets.
    fn add_units(&mut self, a: usize, asset: usize, units: Vec<NewUnit<'m>>) -> Vec<RecordUnit> {
        let mut out = Vec::new();
        let mut deps = Vec::new();
        for nu in units {
            let key = self.stored(a, nu.key);
            out.push(RecordUnit {
                key,
                planned: nu.working_set,
            });
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
                planned: nu.working_set,
                triangles: nu.triangles,
                state: State::Waiting,
                mesh: None,
                texture: None,
            });
        }
        let first = self.units.len() - deps.len();
        for (u, names) in (first..).zip(deps) {
            self.units[u].deps = names
                .iter()
                .map(|n| self.by_name[&(a, n.clone())])
                .collect();
        }
        out
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
        let units = self.add_units(a, asset, imported.units);
        for f in &imported.files {
            self.input(a, asset, &f.path);
        }
        self.add_assemblies(a, imported.assemblies.iter().cloned());
        self.assets[asset]
            .warnings
            .extend_from_slice(&imported.warnings);
        self.records[a].push(Record {
            identity,
            files: imported.files,
            units,
            assemblies: imported.assemblies,
            warnings: imported.warnings,
        });
        for (k, src) in imported.instances.into_iter().enumerate() {
            let base = format!("{}/instances", self.assets[asset].name);
            self.instances.push(PendingInstances {
                archive: a,
                asset,
                name: match k {
                    0 => base,
                    k => format!("{base}/{k}"),
                },
                src: Cow::Owned(src),
                record: Some((a, self.records[a].len() - 1)),
            });
        }
    }

    /// Units of library meshes, whose sections' backs name materials of any planned unit or
    /// library mesh. A material's textures are the texture units of the archive holding it.
    fn mesh_sources(&mut self) -> Result<()> {
        let meshes = std::mem::take(&mut self.meshes);
        let units: HashMap<(usize, AssetId), usize> = self
            .by_name
            .iter()
            .map(|((a, n), &u)| ((*a, AssetId::from_name(n)), u))
            .collect();
        let first_units: HashMap<AssetId, usize> = self
            .first_by_name
            .iter()
            .map(|(n, &u)| (AssetId::from_name(n), u))
            .collect();
        let mut library: HashMap<(usize, AssetId), Material> = HashMap::new();
        let mut first_library: HashMap<AssetId, (usize, Material)> = HashMap::new();
        for p in &meshes {
            for (i, m) in p.materials.iter().enumerate() {
                let id = AssetId::from_name(&format!("{}/material/{i}", p.name));
                library.insert((p.archive, id), *m);
                let first = first_library.entry(id).or_insert((p.archive, *m));
                if p.archive < first.0 {
                    *first = (p.archive, *m);
                }
            }
        }
        for p in meshes {
            let a = p.archive;
            let is_material = |u: &&usize| self.units[**u].class == Class::Material as u16;
            let read = |u: usize| -> Result<Material> {
                Ok(bytemuck::pod_read_unaligned(&self.unit_blob(u)?))
            };
            // As `named`: the archive's own material, else the first archive's that has one,
            // with the archive holding it.
            let material = |id: AssetId| -> Result<Option<(usize, Material)>> {
                if let Some(m) = library.get(&(a, id)) {
                    return Ok(Some((a, *m)));
                }
                if let Some(&u) = units.get(&(a, id)).filter(is_material) {
                    return read(u).map(|m| Some((a, m)));
                }
                let unit = first_units.get(&id).filter(is_material);
                match (unit, first_library.get(&id)) {
                    (Some(&u), Some(&(la, m))) if la < self.units[u].archive => Ok(Some((la, m))),
                    (Some(&u), _) => read(u).map(|m| Some((self.units[u].archive, m))),
                    (None, l) => Ok(l.copied()),
                }
            };
            let mut backs = Vec::with_capacity(p.materials.len());
            for m in p.materials {
                backs.push(match m.back {
                    AssetId::NONE => None,
                    id => Some(material(id)?.ok_or_else(|| Error::UnknownMaterial {
                        asset: p.name.into(),
                        material: id.to_string(),
                    })?),
                });
            }
            let mut textures: HashMap<usize, HashMap<AssetId, MapTexture>> =
                HashMap::from([(a, HashMap::new())]);
            let fronts = p.materials.iter().map(|m| (a, m));
            for (archive, m) in fronts.chain(backs.iter().flatten().map(|(b, m)| (*b, m))) {
                let held = textures.entry(archive).or_default();
                for b in &m.textures {
                    if b.texture != AssetId::NONE
                        && !held.contains_key(&b.texture)
                        && let Some(t) = self.map_texture(&units, archive, b.texture)?
                    {
                        held.insert(b.texture, t);
                    }
                }
            }
            let backs: Vec<_> = backs
                .iter()
                .map(|b| b.as_ref().map(|(archive, m)| (m, &textures[archive])))
                .collect();
            let (new, assembly) = mesh_units(
                p.name,
                p.mesh,
                p.materials,
                &backs,
                p.settings,
                &textures[&a],
            )?;
            self.add_units(a, p.asset, new);
            self.add_assemblies(a, assembly);
        }
        Ok(())
    }

    /// Texture `id` of archive `a`, by `units`, as a mesh reads it: a cooked texture or a set
    /// of tiles.
    fn map_texture(
        &self,
        units: &HashMap<(usize, AssetId), usize>,
        a: usize,
        id: AssetId,
    ) -> Result<Option<MapTexture>> {
        let Some(&u) = units.get(&(a, id)) else {
            return Ok(None);
        };
        let unit = &self.units[u];
        Ok(match Class::from_u16(unit.class) {
            Some(Class::Texture) => Some(MapTexture::lone(&unit.name, unit.key)),
            Some(Class::TextureTiles) => {
                let set = TextureTiles::parse(&self.unit_blob(u)?)?;
                let mut tiles = Vec::with_capacity(set.tiles.len());
                for t in &set.tiles {
                    let tile = &self.units[units[&(a, t.texture)]];
                    tiles.push(([t.column, t.row], tile.name.clone(), tile.key));
                }
                Some(MapTexture {
                    unit: unit.name.clone(),
                    key: unit.key,
                    tiles,
                })
            }
            _ => None,
        })
    }

    /// The first blob of unit `u`: its planned bytes, else its linked entry's.
    fn unit_blob(&self, u: usize) -> Result<Vec<u8>> {
        let unit = &self.units[u];
        if let Some(Work::Blobs { blobs, .. }) = &unit.work {
            return Ok(blobs[0].0.clone());
        }
        let State::Done(done) = &unit.state else {
            unreachable!("a unit planned without its bytes is linked");
        };
        self.with_archive(unit.archive, &done.from, |arc| {
            let blob = &arc.blobs(&arc.entries()[done.entries[0]])[0];
            let mut bytes = vec![0; blob.raw_len as usize];
            arc.read(blob, &mut bytes).map(|()| bytes)
        })?
        .map_err(Into::into)
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
            let key = self.stored(a, key.finish());
            let working_set = src.records.len() as u64 * INSTANCE_BYTES_PER_PLACEMENT;
            if let Some((ra, r)) = record {
                self.records[ra][r].units.push(RecordUnit {
                    key,
                    planned: working_set,
                });
            }
            self.add(Unit {
                archive: a,
                asset,
                name,
                class: Class::InstancePage as u16,
                key,
                work: Some(Work::Instances(src, cell_log2, Catalog::new(), meshes)),
                deps,
                working_set,
                planned: working_set,
                triangles: 0,
                state: State::Waiting,
                mesh: None,
                texture: None,
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
                for unit in data.sections().iter().flat_map(|s| s.maps.units()) {
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
        }
        self.priority = self.units.iter().map(|u| u.planned).collect();
        // Dependents are added after the units they wait on, mostly; the loop settles the
        // rest.
        loop {
            let mut raised = false;
            for u in (0..self.units.len()).rev() {
                let most = self.dependents[u]
                    .iter()
                    .map(|&d| self.priority[d])
                    .max()
                    .unwrap_or(0);
                if most > self.priority[u] {
                    self.priority[u] = most;
                    raised = true;
                }
            }
            if !raised {
                break;
            }
        }
        for u in 0..self.units.len() {
            if matches!(self.units[u].state, State::Waiting) && self.pending[u] == 0 {
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
                    let instances = unit.class == Class::InstancePage as u16;
                    (instances, Reverse(unit.planned), unit.name.as_str())
                };
                k(x).cmp(&k(y))
            });
            self.order[a] = order;
            // A writer appends each entry's name and blob records after the previous entry's,
            // so the old archive is the one this cook would write when every entry is linked
            // from it and each follows on from the one before in write order.
            let unchanged = self.old[a].as_ref().is_some_and(|old| {
                let entries = old.archive.entries();
                let (mut linked, mut name, mut blob) = (0, 0, 0);
                let in_order = self.order[a].iter().all(|&u| match &self.units[u].state {
                    State::Done(d) if matches!(d.from, From::Old) => d.entries.iter().all(|&e| {
                        let e = &entries[e];
                        let next = e.name.offset == name && e.first_blob == blob;
                        linked += 1;
                        name += e.name.len;
                        blob += e.blob_count;
                        next
                    }),
                    _ => false,
                });
                in_order && linked == entries.len()
            });
            self.writers.push(if unchanged {
                None
            } else {
                let mut w = ArchiveWriter::create(self.manifest.archives[a].output.clone())?;
                w.set_zstd_level(self.shared.zstd_level);
                Some(w)
            });
        }
        Ok(())
    }

    fn cook_all<'t>(
        &mut self,
        scope: &'t Scope<'t, '_>,
        jobs: usize,
        tx: &mpsc::Sender<Msg>,
        rx: &mpsc::Receiver<Msg>,
    ) -> Result<()>
    where
        'm: 't,
        's: 't,
    {
        // The writer thread copies done units into the archives, so the coordinator keeps
        // starting jobs while it writes.
        let (writes, queue) = mpsc::channel::<Write>();
        let mut writers = std::mem::take(&mut self.writers);
        let wtx = tx.clone();
        let writer = scope.spawn(move || {
            write_units(&mut writers, &queue, &wtx);
            writers
        });
        let (mut running, mut bytes) = (0usize, 0u64);
        let budget =
            self.opts.working_set - DECODED_CACHES * (self.opts.working_set / DECODED_SHARE);
        loop {
            while let Some(u) = self.local.pop() {
                self.finish_local(u)?;
            }
            self.send_ready(&writes);
            let sent = self
                .cursor
                .iter()
                .zip(&self.order)
                .all(|(c, o)| *c == o.len());
            if sent && self.writing.is_empty() {
                break;
            }
            // The units the next unwritten ones wait on start first.
            let mut candidates: Vec<usize> = (0..self.order.len())
                .filter_map(|a| {
                    self.order[a]
                        .get(self.cursor[a])
                        .and_then(|&u| self.blocking(u))
                })
                .collect();
            // A full window holds back the jobs of archives without a scratch directory.
            let full = self.window > self.opts.reorder_window;
            candidates.extend(
                self.ready
                    .iter()
                    .map(|&(_, u)| u)
                    .filter(|&u| !full || self.scratch[self.units[u].archive].is_some()),
            );
            // A unit that does not fit stays first in line: a smaller one started past it
            // would hold budget it waits for.
            let mut head = None;
            for u in candidates {
                if running >= jobs {
                    break;
                }
                if !self.ready.contains(&self.ready_key(u)) {
                    continue;
                }
                let ws = self.units[u].working_set;
                if running > 0 && bytes + ws > budget {
                    head = Some(u);
                    break;
                }
                self.ready.remove(&self.ready_key(u));
                running += 1;
                bytes += ws;
                self.spawn(scope, tx, u);
            }
            match (self.held, head) {
                (Some((h, _)), Some(u)) if h == u => {}
                (held, head) => {
                    if let Some((h, since)) = held {
                        *self.waited.entry(h).or_default() += since.elapsed().as_secs_f64();
                    }
                    self.held = head.map(|u| (u, Instant::now()));
                }
            }
            crate::profile::plot!("jobs running", running as f64);
            crate::profile::plot!("jobs working set GiB", bytes as f64 / f64::from(1 << 30));
            assert!(
                running > 0 || !self.local.is_empty() || !self.writing.is_empty(),
                "the cook has nothing to run and entries left to write"
            );
            if running == 0 && self.writing.is_empty() {
                continue;
            }
            match rx.recv().expect("the coordinator holds a sender") {
                Msg::Unit(u, r) => {
                    running -= 1;
                    bytes -= self.units[u].working_set;
                    if let Some(reads) = self.map_reads.remove(&u) {
                        self.release_maps(reads);
                    }
                    let (from, seconds, detail) = r.unwrap_or_else(|p| resume_unwind(p))?;
                    self.details.insert(u, detail);
                    self.done(u, from, None, Outcome::Cooked, seconds)?;
                }
                Msg::Written(u, r) => {
                    let reports = r.unwrap_or_else(|p| resume_unwind(p))?;
                    self.written(u, reports);
                }
                Msg::Import(..) => unreachable!("imports are done"),
            }
        }
        drop(writes);
        self.writers = writer.join().unwrap_or_else(|p| resume_unwind(p));
        Ok(())
    }

    /// A ready unit that `u`, the next unit to write, waits on.
    fn blocking(&self, u: usize) -> Option<usize> {
        match self.units[u].state {
            State::Waiting if self.pending[u] == 0 => {
                self.ready.contains(&self.ready_key(u)).then_some(u)
            }
            State::Waiting => self.units[u].deps.iter().find_map(|&d| self.blocking(d)),
            _ => None,
        }
    }

    /// The archives holding the cooked texture units a mesh unit's sections read.
    fn map_archives(
        &mut self,
        u: usize,
        sections: &[SectionData],
    ) -> Result<HashMap<String, Arc<Archive>>> {
        let a = self.units[u].archive;
        let units = || sections.iter().flat_map(|s| s.maps.units());
        let reads = units().map(|t| self.by_name[&(a, t.to_owned())]).collect();
        self.map_reads.insert(u, reads);
        let mut archives = HashMap::new();
        for unit in units() {
            if archives.contains_key(unit) {
                continue;
            }
            let t = self.by_name[&(a, unit.to_owned())];
            let from = match &self.units[t].state {
                State::Done(d) => &d.from,
                State::Written(Some(f)) => f,
                _ => unreachable!("a mesh starts after the textures it reads"),
            };
            let archive = match from {
                From::Memory(arc) => Arc::clone(arc),
                From::Old => Arc::clone(
                    &self.old[a]
                        .as_ref()
                        .expect("linked from the old archive")
                        .archive,
                ),
                From::Cache(path) | From::Scratch(path) => Arc::new(Archive::open(path)?),
            };
            archives.insert(unit.to_owned(), archive);
        }
        Ok(archives)
    }

    /// Lets go of the texture units a finished mesh unit read.
    fn release_maps(&mut self, reads: Vec<usize>) {
        for t in reads {
            self.map_users[t] -= 1;
            if self.map_users[t] == 0
                && !self.writing.contains_key(&t)
                && matches!(self.units[t].state, State::Written(_))
                && let State::Written(Some(from)) =
                    std::mem::replace(&mut self.units[t].state, State::Written(None))
            {
                self.release(&from);
            }
        }
    }

    fn spawn<'t>(&mut self, scope: &'t Scope<'t, '_>, tx: &mpsc::Sender<Msg>, u: usize)
    where
        'm: 't,
        's: 't,
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
            match self.map_archives(u, data.sections()) {
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
        let spill = self.scratch[unit.archive]
            .clone()
            .filter(|_| self.window > self.opts.reorder_window);
        let (tx, shared) = (tx.clone(), self.shared);
        let meter = crate::job::Meter::new();
        scope.spawn(move || {
            crate::profile::name_thread("cook job");
            let r = catch_unwind(AssertUnwindSafe(|| {
                let entered = crate::job::enter(Some(meter.job(&name)));
                let z = crate::profile::zone!("job", &name);
                let start = Instant::now();
                let (from, mut detail) =
                    cook_unit(shared, &name, class, key, gdeflate, spill.as_deref(), work)?;
                let seconds = start.elapsed().as_secs_f64();
                drop(z);
                drop(entered);
                detail.peak = meter.peak();
                Ok((from, seconds, detail))
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
            bounds.union(&b);
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
            None,
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

    /// Lets go of a written unit's scratch archive, removed when no unit holds it.
    fn release(&mut self, from: &From) {
        let From::Scratch(path) = from else { return };
        let users = self
            .scratch_users
            .get_mut(path)
            .expect("held scratch archive");
        *users -= 1;
        if *users == 0 {
            self.scratch_users.remove(path);
            let _ = std::fs::remove_file(path);
        }
    }

    /// Sends every unit that is next in its archive's order and done to the writer thread.
    fn send_ready(&mut self, writes: &mpsc::Sender<Write>) {
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
                let from = match &d.from {
                    From::Memory(arc) => WriteFrom::Archive(Arc::clone(arc)),
                    From::Old => WriteFrom::Archive(Arc::clone(
                        &self.old[a]
                            .as_ref()
                            .expect("linked from the old archive")
                            .archive,
                    )),
                    From::Cache(path) | From::Scratch(path) => WriteFrom::Path(path.clone()),
                };
                let unit = &self.units[u];
                let write = Write {
                    unit: u,
                    archive: a,
                    from,
                    entries: d.entries,
                    outcome: d.outcome,
                    seconds: d.seconds,
                    working_set: unit.working_set,
                    triangles: unit.triangles,
                    detail: self.details.remove(&u),
                    waited: self.waited.remove(&u),
                };
                self.units[u].state = State::Written(Some(d.from));
                self.writing.insert(u, d.memory);
                writes
                    .send(write)
                    .expect("the writer runs until the cook ends");
                self.cursor[a] += 1;
            }
        }
    }

    /// Takes a unit the writer thread has written: its reports, and its entries' memory and
    /// scratch archive unless a mesh still reads them.
    fn written(&mut self, u: usize, reports: Vec<report::EntryReport>) {
        let memory = self.writing.remove(&u).expect("sent to the writer");
        self.window -= memory;
        self.reports.insert(u, reports);
        if self.map_users[u] == 0
            && let State::Written(Some(from)) =
                std::mem::replace(&mut self.units[u].state, State::Written(None))
        {
            self.release(&from);
        }
    }

    fn remove_scratch(&self) {
        for dir in self.scratch.iter().flatten() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    /// Completes the archives, then saves cookstates and trims the cache.
    fn finish(self) -> Result<Cooked> {
        let mut cooked = Cooked::default();
        self.remove_scratch();
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
            mut reports,
            ..
        } = self;
        // Reports list each asset's entries by unit name, instance pages last.
        let mut by_name: Vec<usize> = (0..units.len()).collect();
        by_name.sort_by(|&x, &y| {
            let k = |u: usize| {
                let unit = &units[u];
                (unit.class == Class::InstancePage as u16, unit.name.as_str())
            };
            k(x).cmp(&k(y))
        });
        for u in by_name {
            if let Some(r) = reports.remove(&u) {
                assets[units[u].asset].entries.extend(r);
            }
        }
        // By unit, so models of one name keep a fixed order.
        let mut assemblies: Vec<(usize, Assembly)> = assemblies.into_iter().collect();
        assemblies.sort_unstable_by_key(|a| a.0);
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
            let finished = {
                let _z = crate::profile::zone!("archive.complete");
                writer.map(ArchiveWriter::finish).transpose()?.flatten()
            };
            match finished {
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

fn tangent_report(vertices: u64, t: mesh::TangentCheck) -> TangentReport {
    let per_vertex = orr_pack::mesh::attr_stride(orr_pack::mesh::MeshHeader::TANGENT)
        - orr_pack::mesh::attr_stride(0);
    TangentReport {
        angle_p99: t.angle_p99,
        texels: t.texels,
        derived: t.derive,
        bytes_saved: if t.derive {
            vertices * u64::from(per_vertex)
        } else {
            0
        },
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
            finest_exponent: info.header.position_exponent,
            bounds: info.header.bounds,
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
            finest_exponent: i8::MIN,
            bounds: info.header.bounds,
        },
    ))
}

/// Estimated peak of a texture cook: an environment's from its source and map
/// ([`texture::environment_working_set`]), others 64 B per source texel when the header gives
/// the size, else per file byte.
fn texture_file_working_set(
    path: &Path,
    size: Option<[u32; 2]>,
    settings: &TextureSettings,
) -> u64 {
    let file = || std::fs::metadata(path).map_or(0, |m| m.len());
    let texels = size.map(|[w, h]| u64::from(w) * u64::from(h));
    match settings.usage {
        crate::Usage::Environment { .. } => {
            texture::environment_working_set(file(), texels, settings.max_size)
        }
        _ => texels.unwrap_or_else(file) * 64,
    }
}

/// Imports a scene or material document and plans its units.
fn import(name: &str, path: &Path, kind: ImportKind<'_>, shared: &Shared) -> Result<Imported> {
    let memo = &shared.memo;
    // Before the importer, which reads them as it goes.
    let check = |clips: &crate::ClipSettings| {
        clips.check().map_err(|message| Error::Import {
            asset: name.into(),
            message,
        })
    };
    let (scene, rules) = match kind {
        ImportKind::Gltf(s) => {
            check(&s.clips)?;
            (crate::gltf::import(name, path, memo, s)?, &s.textures)
        }
        ImportKind::Usd(s) => {
            check(&s.clips)?;
            (crate::usd::import(name, path, s)?, &s.textures)
        }
        ImportKind::MaterialX { material, textures } => (
            crate::materialx::import(name, path, memo, material)?,
            textures,
        ),
    };
    plan_scene(name, scene, rules, memo, shared.quality)
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
fn plan_scene(
    name: &str,
    mut src: Scene,
    rules: &TextureRules,
    memo: &Memo,
    quality: Quality,
) -> Result<Imported> {
    let mut units = Vec::new();
    let mut skeletons = Vec::with_capacity(src.skeletons.len());
    let mut skeleton_names = Vec::with_capacity(src.skeletons.len());
    let reaches = src.reaches();
    for (sk, reach) in std::mem::take(&mut src.skeletons).into_iter().zip(reaches) {
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
        skeleton_names.push(skeleton_name.clone());
        let working_set = bytes.len() as u64;
        let work = Work::Blobs {
            schema: orr_pack::skeleton::SCHEMA,
            blobs: vec![(bytes, true)],
        };
        let class = Class::Skeleton as u16;
        units.push(NewUnit::leaf(&skeleton_name, class, key, working_set, work));
        for (clip_name, source) in sk.clips {
            let mut clip = crate::skin::resample(&sk.info, &source, &reach, &src.clips);
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
    // A set reads through its tiles.
    let images = &src.images;
    let read_as = |i: usize| -> Vec<usize> {
        match images.get(i).and_then(ImageFile::tiles) {
            Some(tiles) => tiles.iter().map(|t| t.image).collect(),
            None => vec![i],
        }
    };
    let sets = used.clone();
    let used: Vec<usize> = used.into_iter().flat_map(read_as).collect();
    let colors: HashSet<usize> = src
        .materials
        .iter()
        .flat_map(|m| m.textures.colors().flat_map(|r| read_as(r.image)))
        .collect();
    // A height's range is what it stores, and a remap can take a color's below 0.
    let whole: HashSet<usize> = src
        .materials
        .iter()
        .flat_map(|m| {
            let remapped = m.textures.colors().filter(|r| r.remap.is_some());
            let across = [slot::DISPLACEMENT_U, slot::DISPLACEMENT_V]
                .into_iter()
                .filter_map(|s| m.textures.more.get(&s).and_then(Option::as_ref));
            m.textures.displacement.iter().chain(remapped).chain(across)
        })
        .flat_map(|r| read_as(r.image))
        .collect();
    let normals: HashSet<usize> = src
        .materials
        .iter()
        .flat_map(|m| [&m.textures.normal, &m.textures.coat_normal])
        .flatten()
        .filter(|n| n.0.rules == ImageRules::Gltf)
        .flat_map(|n| read_as(n.0.image))
        .collect();
    // glTF states no value for an image it can't read.
    let optional: HashSet<usize> = src
        .materials
        .iter()
        .flat_map(|m| m.textures.refs())
        .filter(|r| r.rules != ImageRules::Gltf)
        .flat_map(|r| read_as(r.image))
        .collect();
    let reads = ImageReads {
        whole,
        normals,
        colors,
        optional,
    };
    let (mut facts, mut unread) = image_facts(name, &mut src.images, &used, &reads, memo)?;
    settle_tiles(&sets, &mut src.images, &mut facts, &mut unread);
    let mut warnings = std::mem::take(&mut src.warnings);
    warnings.extend(
        unread
            .iter()
            .map(|(&i, why)| texture::unread_warning(&src.images[i].name, why)),
    );
    let mut solids = HashMap::new();
    for m in &mut src.materials {
        stand_in(
            &mut m.textures,
            &unread,
            &mut src.images,
            &mut facts,
            &mut solids,
        );
    }
    warnings.extend(
        used.iter()
            .filter(|i| facts.get(*i).is_some_and(|f| f.animated))
            .map(|&i| texture::first_frame_warning(&src.images[i].name)),
    );
    let unscaled: BTreeSet<usize> = src
        .materials
        .iter()
        .flat_map(|m| m.textures.unscaled_normals(&facts))
        .collect();
    warnings.extend(
        unscaled
            .into_iter()
            .map(|i| texture::unscaled_normal_warning(&src.images[i].name)),
    );
    for i in &used {
        if let ImageData::File { path, read, .. } = &src.images[*i].data {
            src.files.push(Input {
                path: path.clone(),
                hash: read.map_or_else(|| memo.hash(path).unwrap_or(Input::ABSENT), |r| r.0),
            });
        }
    }
    let mut packer = Packer::new(name, &src.images, &facts, rules, quality);
    // Per material: its id, key and cooked parameters.
    let mut materials = Vec::new();
    for m in &src.materials {
        let mut cooked = m.material;
        packer.material(&m.suffix, &mut cooked, &m.textures)?;
        if let Some(back) = m.back {
            let back = material_name(name, &src.materials[back].suffix);
            cooked.back = AssetId::from_name(&back);
        }
        let unit = material_unit(material_name(name, &m.suffix), &cooked);
        materials.push((AssetId::from_name(&unit.name), unit.key, cooked));
        units.push(unit);
    }
    let clipped = packer.clipped(src.materials.iter().map(|m| &m.textures))?;
    if let Some(&first) = clipped.first() {
        let name = &src.images[first].name;
        warnings.push(texture::gamut_warning(&match clipped.len() - 1 {
            0 => name.clone(),
            more => format!("{name} and {more} more images"),
        }));
    }
    let jobs = std::mem::take(&mut packer.jobs);
    let sets = std::mem::take(&mut packer.neural);
    let tile_sets = std::mem::take(&mut packer.tile_sets);
    drop(packer);
    let mut textures: HashMap<AssetId, MapTexture> = jobs
        .iter()
        .map(|j| {
            (
                AssetId::from_name(&j.name),
                MapTexture::lone(&j.name, j.key),
            )
        })
        .collect();
    for set in tile_sets {
        let held = TextureTiles::parse(&set.bytes).expect("a planned set parses");
        let tiles = held.tiles.iter().map(|t| {
            let tile = &textures[&t.texture];
            ([t.column, t.row], tile.unit.clone(), tile.key)
        });
        let texture = MapTexture {
            unit: set.name.clone(),
            key: set.key,
            tiles: tiles.collect(),
        };
        textures.insert(AssetId::from_name(&set.name), texture);
        let work = Work::Blobs {
            schema: orr_pack::tiles::SCHEMA,
            blobs: vec![(set.bytes, true)],
        };
        let class = Class::TextureTiles as u16;
        units.push(NewUnit::leaf(&set.name, class, set.key, 0, work));
    }

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
        for (sec, &mi) in mesh.sections_mut().iter_mut().zip(&model.materials) {
            let cooked = &materials[mi].2;
            let back = src.materials[mi].back.map(|b| &materials[b].2);
            // Packing binds the textures that decide a section's class and emission.
            *sec = section_data(
                sec.indices.clone(),
                cooked,
                back.map(|b| (b, &textures)),
                &textures,
            );
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
        let mesh = (!mesh.sections().is_empty()).then(|| MeshPart {
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
        let (model, assembly) = model_units(model_name, mesh, subd, &model_materials, skeleton)?;
        units.extend(model);
        assemblies.extend(assembly);
    }
    let (alone, shared): (Vec<_>, Vec<_>) = src.instances.iter().partition(|m| m.alone);
    let mut instances = scene_instances(name, &shared);
    for m in alone {
        let mut pages = scene_instances(name, &[m]);
        for src in &mut pages {
            src.frame = m.frame.map(|(skeleton, joint)| PageFrame {
                skeleton: skeleton_names[skeleton].clone(),
                joint: u32::from(joint),
            });
            src.motion.clone_from(&m.motion);
        }
        instances.extend(pages);
    }
    let images = Arc::new(src.images);
    for job in jobs {
        let (name, key, working_set) = (job.name.clone(), job.key, job.working_set(&images));
        let work = Work::Texture(Box::new(job), Arc::clone(&images));
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
        warnings,
    })
}

/// The instance sources of a scene's instanced models: one prototype per model, a placement
/// per instance, and a source per 65536 models, the most a source's prototype index names.
fn scene_instances(name: &str, models: &[&PlacedModel]) -> Vec<Instances> {
    models
        .chunks(1 << 16)
        .map(|chunk| {
            let mut src = Instances::default();
            for (p, m) in chunk.iter().enumerate() {
                src.prototypes.push(PrototypeSpec {
                    model: format!("{name}/{}", m.model),
                    ..PrototypeSpec::default()
                });
                src.records.extend(m.placed.iter().map(|q| Placement {
                    position: q.position.to_array(),
                    linear: Linear::Matrix(q.linear.to_cols_array_2d()),
                    prototype: p as u16,
                }));
            }
            src
        })
        .collect()
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
fn static_material_unit(name: &str, m: &Material) -> NewUnit<'static> {
    let mut m = *m;
    m.textures[slot::DISPLACEMENT] = bytemuck::Zeroable::zeroed();
    m.textures[slot::DISPLACEMENT_U] = bytemuck::Zeroable::zeroed();
    m.textures[slot::DISPLACEMENT_V] = bytemuck::Zeroable::zeroed();
    m.displacement_tangent = [[0.0; 4]; 2];
    m.displacement_scale = 0.0;
    m.displacement_offset = 0.0;
    m.displacement_min = 0.0;
    m.displacement_max = 0.0;
    m.displacement_world = 0;
    material_unit(format!("{name}/static"), &m)
}

/// A section of `indices` drawn with `m`, whose back is `back` where it has one of its own,
/// with the textures each binds. A section whose front emits nothing weighs its clusters by its back's
/// emission.
fn section_data(
    indices: Range<usize>,
    m: &Material,
    back: Option<(&Material, &HashMap<AssetId, MapTexture>)>,
    textures: &HashMap<AssetId, MapTexture>,
) -> SectionData {
    let mut maps = map_spec(m, textures);
    if let Some((b, held)) = back.filter(|(b, _)| !m.emits() && b.emits()) {
        maps.emission = map_spec(b, held).emission;
    }
    SectionData::new(indices, m, back.map(|b| b.0), maps)
}

/// A material entry holding `m`.
fn material_unit(name: String, m: &Material) -> NewUnit<'static> {
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
fn map_spec(m: &Material, textures: &HashMap<AssetId, MapTexture>) -> MapSpec {
    let bound = |s: usize| {
        let b = m.textures[s];
        (b.texture != AssetId::NONE).then(|| (textures.get(&b.texture), usize::from(b.uv_set & 1)))
    };
    let read = |s: usize| MapRead {
        wrap: m.textures[s].wrap.map(Wrap::of_binding),
        uv: m.uv_transforms[s],
        outside: m.textures[s].outside,
    };
    let opacity = match (m.alpha_mode, bound(slot::OPACITY)) {
        // Vertices state an opacity no triangle-wide test can stand for.
        (Material::MASK, _) if m.textures[slot::OPACITY].vertex != 0 => OpacitySpec::Unknown,
        (Material::MASK, None) => OpacitySpec::Constant(m.geometry_opacity >= m.alpha_cutoff),
        (Material::MASK, Some((Some(texture), uv_set))) => OpacitySpec::Texture {
            texture: texture.clone(),
            cutoff: m.alpha_cutoff,
            read: read(slot::OPACITY),
            uv_set,
        },
        (Material::MASK, Some(_)) => OpacitySpec::Unknown,
        _ => OpacitySpec::None,
    };
    let height = |s: usize, [scale, offset, min, max]: [f32; 4]| {
        bound(s).map(|b| DisplacementSpec {
            scale,
            offset,
            min,
            max,
            texture: b.0.map(|t| DisplacementTexture {
                texture: t.clone(),
                read: read(s),
                uv_set: b.1,
            }),
        })
    };
    let displacement = height(
        slot::DISPLACEMENT,
        [
            m.displacement_scale,
            m.displacement_offset,
            m.displacement_min,
            m.displacement_max,
        ],
    );
    let tangent_displacement = [
        height(slot::DISPLACEMENT_U, m.displacement_tangent[0]),
        height(slot::DISPLACEMENT_V, m.displacement_tangent[1]),
    ];
    let frame_set = usize::from(m.tangent_set());
    let normal = match bound(slot::NORMAL).filter(|_| !m.object_normal(slot::NORMAL)) {
        _ if mesh::reads_tangent(m) => NormalSpec::Tangent { frame_set },
        None => NormalSpec::None,
        Some((Some(t), uv_set)) => NormalSpec::Texture {
            texture: t.clone(),
            read: read(slot::NORMAL),
            uv_set,
            frame_set,
        },
        Some((None, _)) => NormalSpec::Tangent { frame_set },
    };
    let emission = match bound(slot::EMISSION) {
        Some((Some(texture), uv_set)) if m.emits() => Some(EmissionSpec {
            texture: texture.clone(),
            read: read(slot::EMISSION),
            uv_set,
        }),
        _ => None,
    };
    MapSpec {
        tangent_displacement,
        opacity,
        displacement,
        normal,
        emission,
        uv1: mesh::reads_uv1(m),
    }
}

/// A mesh, its materials and a model from a library caller's mesh, and the model's assembly.
/// `backs` holds the material each one's back names with the textures it binds, and `textures`
/// the textures the materials bind.
fn mesh_units(
    name: &str,
    input: &MeshInput,
    materials: &[Material],
    backs: &[Option<(&Material, &HashMap<AssetId, MapTexture>)>],
    settings: &MeshSettings,
    textures: &HashMap<AssetId, MapTexture>,
) -> Result<(Vec<NewUnit<'static>>, Option<Assembly>)> {
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
    for (i, ((m, back), span)) in materials.iter().zip(backs).zip(&input.sections).enumerate() {
        let first = span.first as usize;
        let section = section_data(first..first + span.count as usize, m, *back, textures);
        deps.extend(section.maps.units().map(str::to_string));
        let mut unit = material_unit(format!("{name}/material/{i}"), m);
        if bakes_displacement(settings, &section.maps) {
            let baked = static_material_unit(&unit.name, m);
            units.push(std::mem::replace(&mut unit, baked));
        }
        materials_out.push((AssetId::from_name(&unit.name), unit.key));
        units.push(unit);
        sections.push(section);
    }
    deps.sort_unstable();
    deps.dedup();
    let n = input.positions.len();
    let mesh = MeshData {
        moves: None,
        colors: input
            .colors
            .as_ref()
            .map(|c| c.iter().map(Vec4::to_array).collect()),
        positions: input.positions.iter().map(Vec3::to_array).collect(),
        normals: input.normals.iter().map(Vec3::to_array).collect(),
        tangents: input
            .tangents
            .as_ref()
            .map(|t| t.iter().map(Vec4::to_array).collect()),
        uvs: std::iter::once(input.uvs.as_ref().map_or_else(
            || vec![[0.0; 2]; n],
            |u| u.iter().map(Vec2::to_array).collect(),
        ))
        .chain(
            input
                .uvs1
                .as_ref()
                .map(|u| u.iter().map(Vec2::to_array).collect()),
        )
        .collect(),
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
        mesh: MeshSource::Data(Box::new(mesh)),
        settings: settings.clone(),
        deps,
    };
    let (model, assembly) = model_units(
        name.into(),
        Some(part),
        None,
        &materials_out,
        (AssetId::NONE, None),
    )?;
    units.extend(model);
    Ok((units, assembly))
}

/// A mesh to cook for a model: its entry name, data, settings and the units its maps read.
struct MeshPart {
    name: String,
    mesh: MeshSource,
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
) -> Result<(Vec<NewUnit<'static>>, Option<Assembly>)> {
    let mut units = Vec::new();
    let mut found = None;
    // The split reads a checked mesh and compares tangents; other meshes get both when they
    // cook.
    if let Some(part) = &mut mesh
        && part.settings.assemblies.is_some()
        && let MeshSource::Data(data) = &mut part.mesh
    {
        mesh::check(data, &part.settings).map_err(|message| Error::Mesh {
            asset: part.name.clone(),
            message,
        })?;
        data.compute_tangents();
    }
    if let Some(part) = &mut mesh
        && let Some(settings) = &part.settings.assemblies
        && let MeshSource::Data(data) = &part.mesh
        && let Some(split) = assembly::split(data, settings, part.settings.position_step_log2)
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
                mesh: MeshSource::Data(Box::new(mesh)),
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
        part.mesh = MeshSource::Data(Box::new(split.remainder));
        found = Some(Assembly {
            model: model_name.clone(),
            units: list,
            placements: split.placements,
            source_triangles,
            kept_triangles: part.mesh.triangles(),
        });
    }
    units.extend(geometry_units(model_name, mesh, subd, materials, skeleton));
    Ok((units, found))
}

/// The mesh and cage units of a model and the model unit, with these materials (id and key,
/// mesh sections then cage sections) and skeleton (id and key).
fn geometry_units(
    model_name: String,
    mesh: Option<MeshPart>,
    subd: Option<SubdPart>,
    materials: &[(AssetId, Hash32)],
    skeleton: (AssetId, Option<Hash32>),
) -> Vec<NewUnit<'static>> {
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
            working_set: mesh_working_set(&m.mesh, &m.settings),
            triangles,
            work: Work::Mesh(m.mesh, m.settings, HashMap::new()),
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
) -> Result<Vec<NewUnit<'static>>> {
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

/// Mip 0 of opacity texture `unit` (cook key `key`) under `cutoff`, from `cache` when held.
fn opacity_map(
    archives: &HashMap<String, Arc<Archive>>,
    unit: &str,
    key: Hash32,
    cutoff: f32,
    at: MapRead,
    cache: &Decoded<OpacityMap>,
) -> Result<Arc<OpacityMap>> {
    cache.get((key, cutoff.to_bits()), || {
        let mip0 = texture_mips(&archives[unit], AssetId::from_name(unit), false, at, 1)?.remove(0);
        if mip0.w > u32::from(u16::MAX) {
            return Err(Error::Texture {
                asset: unit.to_string(),
                message: format!("opacity texture {} texels wide, over 65535", mip0.w),
            });
        }
        Ok(OpacityMap {
            w: mip0.w,
            h: mip0.h,
            passes: Arc::new(Passes::new(&mip0, cutoff)),
            texels: mip0.texels,
        })
    })
}

/// Mip 0 of normal map `unit` (cook key `key`), from `cache` when held.
fn normal_map(
    archives: &HashMap<String, Arc<Archive>>,
    unit: &str,
    key: Hash32,
    at: MapRead,
    cache: &Decoded<NormalMap>,
) -> Result<Arc<NormalMap>> {
    cache.get((key, 0), || {
        let mips = texture_mips(&archives[unit], AssetId::from_name(unit), false, at, 2)?;
        let [x, y]: [Raster; 2] = mips
            .try_into()
            .unwrap_or_else(|_| panic!("a mip of {unit} has two channels"));
        Ok(NormalMap {
            w: x.w,
            h: x.h,
            texels: [x.texels, y.texels],
        })
    })
}

/// Maps [`decode_maps`] decoded for [`section_maps`], held for it by cook key: a cache keeps
/// no map over its limit, and other jobs' maps can push one out.
#[derive(Default)]
struct Prefetched {
    /// By key and cutoff bits.
    opacity: HashMap<(Hash32, u32), Arc<OpacityMap>>,
    normals: HashMap<Hash32, Arc<NormalMap>>,
    emission: HashMap<Hash32, Arc<EmissionMap>>,
}

/// Decodes, through the caches, the lone textures [`section_maps`] reads for `sections`.
/// Tile sets wait for the mesh: its uvs pick their tiles.
fn decode_maps(
    sections: &[SectionData],
    archives: &HashMap<String, Arc<Archive>>,
    normals: bool,
    shared: &Shared,
) -> Result<Prefetched> {
    let _z = crate::profile::zone!("mesh.decode_maps");
    let mut held = Prefetched::default();
    for s in sections {
        if let OpacitySpec::Texture {
            texture,
            cutoff,
            read,
            ..
        } = &s.maps.opacity
            && texture.tiles.is_empty()
            && let Entry::Vacant(v) = held.opacity.entry((texture.key, cutoff.to_bits()))
        {
            v.insert(opacity_map(
                archives,
                &texture.unit,
                texture.key,
                *cutoff,
                *read,
                &shared.opacity,
            )?);
        }
        if let NormalSpec::Texture { texture, read, .. } = &s.maps.normal
            && normals
            && texture.tiles.is_empty()
            && let Entry::Vacant(v) = held.normals.entry(texture.key)
        {
            v.insert(normal_map(
                archives,
                &texture.unit,
                texture.key,
                *read,
                &shared.normals,
            )?);
        }
        if let Some(e) = &s.maps.emission
            && e.texture.tiles.is_empty()
            && let Entry::Vacant(v) = held.emission.entry(e.texture.key)
        {
            let unit = &e.texture.unit;
            v.insert(emission_map_cached(
                archives,
                unit,
                e.texture.key,
                &shared.emission,
            )?);
        }
    }
    Ok(held)
}

/// The [`emission_map`] of emission texture `unit` (cook key `key`), from `cache` when held.
fn emission_map_cached(
    archives: &HashMap<String, Arc<Archive>>,
    unit: &str,
    key: Hash32,
    cache: &Decoded<EmissionMap>,
) -> Result<Arc<EmissionMap>> {
    cache.get((key, 0), || {
        emission_map(&archives[unit], AssetId::from_name(unit))
    })
}

/// The cooked maps a mesh's sections read, from `held` or the archives holding their texture
/// units.
fn section_maps(
    m: &MeshData,
    archives: &HashMap<String, Arc<Archive>>,
    normals: bool,
    shared: &Shared,
    held: &Prefetched,
) -> Result<Vec<SectionMaps>> {
    let _z = crate::profile::zone!("mesh.maps");
    let read = |unit: &str, all: bool, read, channels| {
        texture_mips(
            &archives[unit],
            AssetId::from_name(unit),
            all,
            read,
            channels,
        )
    };
    // Sections reading one texture share its texels: a tile set's mips can run to gigabytes.
    let mut heights: HashMap<Hash32, Vec<Raster>> = HashMap::new();
    let mut out = Vec::with_capacity(m.sections.len());
    for s in &m.sections {
        let uvs = |set: usize| {
            let corners = m.indices[s.indices.clone()].iter();
            corners.map(move |&i| m.uvs[set][i as usize])
        };
        let opacity = match &s.maps.opacity {
            OpacitySpec::None => None,
            OpacitySpec::Constant(pass) => Some(Opacity::Constant(*pass)),
            OpacitySpec::Unknown => Some(Opacity::Unknown),
            OpacitySpec::Texture {
                texture,
                cutoff,
                read: at,
                uv_set,
            } => {
                let texture_read = |unit: &str, key: Hash32, at: MapRead| {
                    let map = match held.opacity.get(&(key, cutoff.to_bits())) {
                        Some(map) => Arc::clone(map),
                        None => opacity_map(archives, unit, key, *cutoff, at, &shared.opacity)?,
                    };
                    Ok(OpacityRead {
                        mip0: Raster {
                            w: map.w,
                            h: map.h,
                            texels: Arc::clone(&map.texels),
                            wrap: at.wrap,
                            uv: at.uv,
                            outside: at.outside[0],
                        },
                        passes: Arc::clone(&map.passes),
                    })
                };
                let missing = |outside: [f32; 4]| OpacityRead::solid(outside[0], *cutoff);
                Some(Opacity::Texture {
                    map: read_map(texture, *at, uvs(*uv_set), texture_read, missing)?,
                    cutoff: *cutoff,
                    uv_set: *uv_set,
                })
            }
        };
        let mut height = |d: &DisplacementSpec| -> Result<Displacement> {
            Ok(Displacement {
                scale: d.scale,
                offset: d.offset,
                min: d.min,
                max: d.max,
                map: match &d.texture {
                    Some(t) => Some(read_map(
                        &t.texture,
                        t.read,
                        uvs(t.uv_set),
                        |unit, key, at| {
                            let mips = match heights.entry(key) {
                                Entry::Occupied(e) => e.into_mut(),
                                Entry::Vacant(e) => e.insert(read(unit, true, at, 1)?),
                            };
                            let mip = |m: &Raster| Raster {
                                texels: Arc::clone(&m.texels),
                                wrap: at.wrap,
                                uv: at.uv,
                                outside: at.outside[0],
                                ..*m
                            };
                            Ok(mips.iter().map(mip).collect())
                        },
                        |outside| vec![Raster::solid(outside[0])],
                    )?),
                    None => None,
                },
                uv_set: d.texture.as_ref().map_or(0, |t| t.uv_set),
            })
        };
        let displacement = s.maps.displacement.as_ref().map(&mut height).transpose()?;
        let [u, v] = &s.maps.tangent_displacement;
        let tangent_displacement = [
            u.as_ref().map(&mut height).transpose()?,
            v.as_ref().map(&mut height).transpose()?,
        ];
        let normal = match &s.maps.normal {
            NormalSpec::Texture {
                texture,
                read: at,
                uv_set,
                ..
            } if normals => {
                let pair = |unit: &str, key, at: MapRead| {
                    let map = match held.normals.get(&key) {
                        Some(map) => Arc::clone(map),
                        None => normal_map(archives, unit, key, at, &shared.normals)?,
                    };
                    Ok([0, 1].map(|c| Raster {
                        w: map.w,
                        h: map.h,
                        texels: Arc::clone(&map.texels[c]),
                        wrap: at.wrap,
                        uv: at.uv,
                        outside: at.outside[c],
                    }))
                };
                let missing = |o: [f32; 4]| [Raster::solid(o[0]), Raster::solid(o[1])];
                Some(read_map(texture, *at, uvs(*uv_set), pair, missing)?)
            }
            _ => None,
        };
        let emission = match &s.maps.emission {
            None => None,
            Some(e) => {
                // A set's tiles share their primaries, which its slot's values are in too.
                let unit = e.texture.units().next().expect("a texture has a unit");
                let y = luminance_weights(&archives[unit], AssetId::from_name(unit))?;
                let luminance = |[r, g, b, _]: [f32; 4]| y.x * r + y.y * g + y.z * b;
                let tile = |unit: &str, key: Hash32, at: MapRead| {
                    let map = match held.emission.get(&key) {
                        Some(map) => Arc::clone(map),
                        None => emission_map_cached(archives, unit, key, &shared.emission)?,
                    };
                    Ok(Raster {
                        w: map.w,
                        h: map.h,
                        texels: Arc::clone(&map.texels),
                        wrap: at.wrap,
                        uv: at.uv,
                        outside: luminance(at.outside),
                    })
                };
                let missing = |outside| Raster::solid(luminance(outside));
                Some(Emission {
                    map: read_map(&e.texture, e.read, uvs(e.uv_set), tile, missing)?,
                    uv_set: e.uv_set,
                })
            }
        };
        out.push(SectionMaps {
            opacity,
            displacement,
            tangent_displacement,
            normal,
            emission,
        });
    }
    Ok(out)
}

/// `t` under `at` as a map of what `texture` makes of each of its textures. Tiles read clamped in
/// their own squares, and `missing` makes what reads where a set has none from the slot's values.
/// Of a set, it holds the tiles a mesh with vertex uvs `uvs` can reach.
fn read_map<T>(
    t: &MapTexture,
    at: MapRead,
    uvs: impl Iterator<Item = [f32; 2]>,
    mut texture: impl FnMut(&str, Hash32, MapRead) -> Result<T>,
    missing: impl FnOnce([f32; 4]) -> T,
) -> Result<Map<T>> {
    if t.tiles.is_empty() {
        return Ok(Map::Image(texture(&t.unit, t.key, at)?));
    }
    let tile = MapRead {
        wrap: [Wrap::Clamp; 2],
        uv: [0.0; 6],
        ..at
    };
    let reach = Reach::new(at.uv, uvs);
    let tiles = t
        .tiles
        .iter()
        .filter(|(cell, ..)| reach.holds(*cell))
        .map(|([column, row], unit, key)| Ok(((*row, *column), texture(unit, *key, tile)?)));
    Ok(Map::Tiles(Tiles {
        uv: at.uv,
        tiles: tiles.collect::<Result<_>>()?,
        missing: missing(at.outside),
    }))
}

/// The blobs of a cooked texture and its parsed blob 0.
fn texture_head(arc: &Archive, id: AssetId) -> Result<(&[orr_pack::Blob], TextureInfo)> {
    let e = arc.expect(id, Class::Texture as u16, orr_pack::texture::SCHEMA)?;
    let blobs = arc.blobs(e);
    let mut head = vec![0; blobs[0].raw_len as usize];
    arc.read(&blobs[0], &mut head)?;
    Ok((blobs, TextureInfo::parse(&head)?))
}

/// Luminance per channel in a cooked texture's primaries.
fn luminance_weights(arc: &Archive, id: AssetId) -> Result<Vec3> {
    let primaries = texture_head(arc, id)?.1.header.primaries;
    header_luminance(primaries)
}

/// Luminance per channel in primaries `id`.
fn header_luminance(id: u8) -> Result<Vec3> {
    let primaries =
        orr_color::Primaries::from_id(id).ok_or(orr_pack::Error::Malformed("texture primaries"))?;
    Ok(primaries.rgb().luminance_weights().as_vec3())
}

/// The luminance, in the texture's primaries, of the first mip of a cooked emission texture
/// no larger than [`EmissionMap::SIDE_MAX`] a side, or of its last.
fn emission_map(arc: &Archive, id: AssetId) -> Result<EmissionMap> {
    let (blobs, info) = texture_head(arc, id)?;
    let h = info.header;
    let mip = (0..u32::from(h.mip_count))
        .find(|&m| h.mip_size(m).into_iter().max() <= Some(EmissionMap::SIDE_MAX))
        .unwrap_or_else(|| u32::from(h.mip_count) - 1);
    let b = h.mip_blob(mip);
    let mut blob = vec![0; blobs[b].raw_len as usize];
    arc.read(&blobs[b], &mut blob)?;
    let bytes = info.mip_texels(&blob, mip, 0)?;
    let [w, hh] = h.mip_size(mip);
    let srgb = h.vk_format == orr_pack::texture::vk::BC7_SRGB;
    let planes = mip_planes(h.vk_format, w, hh, &bytes, 3)?;
    let y = header_luminance(h.primaries)?;
    let linear = |c: f32| if srgb { texture::srgb_to_linear(c) } else { c };
    let [r, g, b] = [&planes[0], &planes[1], &planes[2]];
    let texels = (0..r.len())
        .map(|i| y.x * linear(r[i]) + y.y * linear(g[i]) + y.z * linear(b[i]))
        .collect();
    Ok(EmissionMap {
        w,
        h: hh,
        texels: Arc::new(texels),
    })
}

/// The first `channels` channels of a cooked mip, one plane each, as the texture reads them.
fn mip_planes(format: u32, w: u32, h: u32, bytes: &[u8], channels: usize) -> Result<Vec<Vec<f32>>> {
    use orr_pack::texture::vk;
    let unread = |_| orr_pack::Error::Malformed("texture format the mesh cook reads");
    let pairs = || bytes.as_chunks::<2>().0.iter().copied();
    let red = |r: Vec<f32>| {
        let n = r.len();
        let mut planes = vec![r];
        planes.extend([0.0, 0.0, 1.0].map(|v| vec![v; n]));
        planes.truncate(channels);
        planes
    };
    let planes: Vec<Vec<f32>> = match format {
        vk::R16_UNORM => red(pairs()
            .map(|b| f32::from(u16::from_le_bytes(b)) / 65535.0)
            .collect()),
        vk::R16_SNORM => red(pairs()
            .map(|b| (f32::from(i16::from_le_bytes(b)) / 32767.0).max(-1.0))
            .collect()),
        vk::R32_SFLOAT => red(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()),
        f if channels == 1 => {
            vec![texture::decode_channel(f, w, h, bytes, 0).map_err(unread)?]
        }
        f => {
            let px = texture::decode_blocks(f, w, h, bytes).map_err(unread)?.px;
            (0..channels)
                .map(|c| px.iter().map(|p| p[c]).collect())
                .collect()
        }
    };
    Ok(planes)
}

/// Mip 0, or every mip with `all`, of a cooked texture: per mip, its first `channels`
/// channels as rasters.
fn texture_mips(
    arc: &Archive,
    id: AssetId,
    all: bool,
    read: MapRead,
    channels: usize,
) -> Result<Vec<Raster>> {
    let (blobs, info) = texture_head(arc, id)?;
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
        let planes = mip_planes(h.vk_format, w, hh, &bytes, channels)?;
        for (c, texels) in planes.into_iter().enumerate() {
            out.push(Raster {
                w,
                h: hh,
                texels: Arc::new(texels),
                wrap: read.wrap,
                uv: read.uv,
                outside: read.outside[c],
            });
        }
    }
    Ok(out)
}

/// The estimated peak bytes of a mesh cook besides its maps.
fn mesh_working_set(mesh: &MeshSource, settings: &MeshSettings) -> u64 {
    let triangles = mesh.triangles();
    match settings.lod {
        mesh::LodMode::Dag => {
            let subsurface = mesh.sections().iter().any(|s| s.subsurface);
            triangles * DAG_BYTES_PER_TRIANGLE
                + mesh.vertices() * DAG_BYTES_PER_VERTEX
                + u64::from(subsurface) * triangles * SUBSURFACE_BYTES_PER_TRIANGLE
        }
        mesh::LodMode::Discrete { .. } => triangles * MESH_BYTES_PER_TRIANGLE,
    }
}

/// What a mesh cook holds of the maps its sections read ([`section_maps`]).
struct MapBytes {
    /// The decoded maps, each texture once per use ([`MapUse`]).
    held: u64,
    /// The opacity micromaps bake from: a byte per mip 0 texel, per section that bakes them.
    micromaps: u64,
    /// The most texel channels one texture use decodes.
    decode_channels: u64,
}

/// A use of a texture, which holds one decoded map of it.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum MapUse {
    /// Mip 0 under the cutoff of these bits, and its [`Passes`].
    Opacity(u32),
    /// Mip 0, two channels; only while the tangent check runs.
    Normal,
    /// Every mip, one channel.
    Height,
    /// The luminance of the mip at most [`EmissionMap::SIDE_MAX`] a side, decoded from three
    /// channels.
    Emission,
}

/// [`MapBytes`] of these sections, with `header` giving a texture unit's header.
fn map_bytes(
    sections: &[SectionData],
    settings: &MeshSettings,
    header: impl Fn(&str) -> orr_pack::texture::TextureHeader,
) -> MapBytes {
    let mut seen = HashSet::new();
    let mut out = MapBytes {
        held: 0,
        micromaps: 0,
        decode_channels: 0,
    };
    let mut add = |texture: &MapTexture, use_: MapUse, out: &mut MapBytes| {
        for unit in texture.units() {
            if !seen.insert((unit.to_owned(), use_)) {
                continue;
            }
            let h = header(unit);
            let texels = |mip: u32| {
                let [w, h] = h.mip_size(mip);
                u64::from(w) * u64::from(h)
            };
            let (held, decoded) = match use_ {
                MapUse::Opacity(_) => {
                    let [w, hh] = h.mip_size(0);
                    out.held += Passes::bytes_of(w, hh);
                    (texels(0), texels(0))
                }
                MapUse::Normal => (2 * texels(0), 2 * texels(0)),
                MapUse::Height => {
                    let all = (0..u32::from(h.mip_count)).map(texels).sum();
                    (all, all)
                }
                MapUse::Emission => {
                    let mip = (0..u32::from(h.mip_count))
                        .find(|&m| h.mip_size(m).into_iter().max() <= Some(EmissionMap::SIDE_MAX))
                        .unwrap_or_else(|| u32::from(h.mip_count) - 1);
                    (texels(mip), 3 * texels(mip))
                }
            };
            out.held += held * MAP_BYTES_PER_CHANNEL;
            out.decode_channels = out.decode_channels.max(decoded);
        }
    };
    for s in sections {
        let m = &s.maps;
        if let OpacitySpec::Texture {
            texture, cutoff, ..
        } = &m.opacity
        {
            add(texture, MapUse::Opacity(cutoff.to_bits()), &mut out);
            let sizes: Vec<_> = texture.units().map(|u| header(u).mip_size(0)).collect();
            out.micromaps += s.micromap_bytes(settings, &sizes);
        }
        if settings.tangent_error_max.is_some()
            && let NormalSpec::Texture { texture, .. } = &m.normal
        {
            add(texture, MapUse::Normal, &mut out);
        }
        let heights = m
            .displacement
            .iter()
            .chain(m.tangent_displacement.iter().flatten());
        for d in heights {
            if let Some(t) = &d.texture {
                add(&t.texture, MapUse::Height, &mut out);
            }
        }
        if let Some(e) = &m.emission {
            add(&e.texture, MapUse::Emission, &mut out);
        }
    }
    out
}

/// Cooks one unit into a one-unit archive: in the cache when there is one, else in `spill`
/// when given, else in memory. With a `gdeflate` codec, its payload blobs are GDeflate pages.
fn cook_unit(
    shared: &Shared,
    name: &str,
    class: u16,
    key: Hash32,
    gdeflate: Option<Arc<dyn GDeflate>>,
    spill: Option<&Path>,
    work: Work<'_>,
) -> Result<(From, Detail)> {
    let mut detail = Detail::default();
    let scratch = match (&shared.cache, spill) {
        (None, Some(dir)) => {
            std::fs::create_dir_all(dir).map_err(|source| Error::Write {
                path: dir.to_owned(),
                source,
            })?;
            Some(dir.join(key::hex(key)))
        }
        _ => None,
    };
    let path = shared
        .cache
        .as_ref()
        .map(|c| c.path(key))
        .or_else(|| scratch.clone());
    let mut w = ArchiveWriter::create(path.clone().map_or(Target::Memory, Target::File))?;
    w.set_zstd_level(shared.zstd_level);
    w.set_gdeflate(gdeflate);
    match work {
        Work::Blobs { schema, blobs } => {
            let blobs: Vec<BlobIn> = blobs
                .iter()
                .map(|(b, m)| if *m { mount(b) } else { BlobIn::new(b) })
                .collect();
            w.add(name, class, schema, key, &blobs)?;
        }
        Work::Mesh(source, settings, archives) => {
            // The tangent check reads normal maps, and every section reading one has
            // tangents once they are computed.
            let normals = settings.tangent_error_max.is_some();
            let sections = source.sections().to_vec();
            let job = crate::job::current();
            let (data, decoded) = std::thread::scope(|s| {
                let decode = s.spawn(|| {
                    crate::profile::name_thread("cook maps");
                    let _job = crate::job::enter(job);
                    decode_maps(&sections, &archives, normals, shared)
                });
                let data = source.read(name, &shared.memo);
                (data, decode.join().unwrap_or_else(|p| resume_unwind(p)))
            });
            let mut data = data?;
            let held = decoded?;
            let uv_warnings = data.pad_uv_sets();
            mesh::check(&data, &settings).map_err(|message| Error::Mesh {
                asset: name.into(),
                message,
            })?;
            data.compute_tangents();
            let maps = section_maps(&data, &archives, normals, shared, &held)?;
            drop((held, archives));
            let gdeflate = w.gdeflate();
            let compress = mesh::Compress {
                zstd_level: w.zstd_level(),
                gdeflate: gdeflate.as_deref(),
            };
            let cooked = mesh::cook(name, data, &settings, maps, compress)?;
            detail.warnings = cooked.warnings;
            detail.warnings.extend(uv_warnings);
            detail.tangents = cooked.tangents;
            let info = cooked.info.to_bytes();
            let _z = crate::profile::zone!("archive.add");
            w.add_streamed(name, class, orr_pack::mesh::SCHEMA, key, |e| {
                e.push(&mount(&info))?;
                cooked.pages.iter().try_for_each(|p| e.push_encoded(p))?;
                for (i, p) in cooked.node_pages.iter().enumerate() {
                    e.push(&BlobIn {
                        mount: i == 0,
                        ..BlobIn::new(p)
                    })?;
                }
                Ok::<_, orr_pack::Error>(())
            })?;
        }
        Work::Texture(job, images) => {
            let built = job.run(&images, &shared.images)?;
            drop(images);
            texture::add(&mut w, name, key, &built)?;
        }
        Work::Neural(set, images) => {
            let built = set.run(&images, &shared.images)?;
            drop(images);
            neural::add(&mut w, name, key, &built)?;
            detail.neural = Some(built.report);
        }
        Work::TextureFile(path, hash, settings) => {
            let bytes = state::read_checked(&path, hash)?;
            let built = texture::cook_file(name, &bytes, &settings, shared.quality)?;
            if built.clips {
                detail.warnings.push(texture::gamut_warning(name));
            }
            texture::add(&mut w, name, key, &built)?;
        }
        Work::Subd(data, materials, settings) => {
            let cooked =
                subd::cook(&data, &materials, &settings).map_err(|message| Error::Mesh {
                    asset: name.into(),
                    message,
                })?;
            detail.warnings = cooked.warnings;
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
        Work::Volume(plan) => {
            detail.volume = Some(volume::cook(&mut w, name, key, &plan)?);
        }
    }
    let finished = {
        let _z = crate::profile::zone!("archive.finish");
        w.finish()?
    };
    let from = match (finished, path) {
        (Some(bytes), _) => From::Memory(Arc::new(Archive::from_memory(bytes)?)),
        (None, Some(path)) if scratch.is_some() => From::Scratch(path),
        (None, Some(path)) => From::Cache(path),
        (None, None) => unreachable!("a file target has a path"),
    };
    Ok((from, detail))
}
