//! Runtime mesh edits. An edit is a model of its own over a mesh of its own: the pages that
//! hold what it touches are per-edit copies a consumer kernel writes at the scene compute
//! point, every other page table entry and cluster entry follows the source mesh. Edited
//! clusters stay at level 0 and their CLAS rebuild on the render queue right after the
//! scene compute point of a frame that marks the edit dirty; the edit's copies gather their
//! ray-tracing cut every frame, since cached cuts hold unedited CLAS.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use orr_gpu::{
    Allocation, CommandList, Device, GpuAddress, GpuPtr, Hazards, MemoryKind, Micromap, RtasIo,
    RtasOp, Stages, TimelinePoint,
};
use orr_pack::Aabb;
use orr_pack::mesh::{Group, MeshHeader, PageHeader};

use crate::accel::{
    self, AccelKernels, ClasBatch, ClasSource, ClusterBuild, ClusterInfo, PageMicromap,
};
use crate::loading::{BlobSource, Loading, Msg, decoded, landed, read_stored};
use crate::memory::{Block, UploadWindow};
use crate::mount::{PAGE, StagedCopy, page_infos, submit_copies};
use crate::producer::scratch_align;
use crate::scene::ModelId;
use crate::shaders::accel::{AccelRoot, ClasJob};
use crate::shaders::edit::{EditJob, EditRoot};
use crate::stream::StreamedPage;
use crate::{ClusterRef, EditGpu, EditGroupGpu, Error, MeshGpu, Result};

/// Spheres an edit tests its groups against one by one; more pin the whole mesh.
pub(crate) const EDIT_SPHERES: usize = 64;

/// What [`crate::Scenes::create_edit`] edits.
#[derive(Clone, Copy, Debug)]
pub struct EditDesc<'a> {
    pub model: ModelId,
    /// Spheres (center xyz, radius, mesh space) around what the edit changes; `None`, or more
    /// than 64, the whole mesh. Every level 0 group meeting one is held at level 0 and handed
    /// to the consumer kernel; the rest of the mesh keeps its level of detail. A mesh cooked
    /// with discrete levels has no groups to choose from: its edit holds all of level 0 and
    /// its copies draw nothing coarser.
    pub touched: Option<&'a [[f32; 4]]>,
    /// A box (mesh space) the edited triangles stay inside; `None`: the model's bounds.
    pub bounds: Option<Aabb>,
}

/// An edit, valid until [`crate::Scenes::destroy_edit`] or until its source's pack is
/// unmounted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EditId(pub u32);

/// A ready edit, for consumer kernels and instances.
#[derive(Clone, Copy, Debug)]
pub struct EditRef {
    /// The edit's model: its copies draw and trace the edited mesh, at rest pose. Its copies
    /// are placed with [`crate::SceneWriter::add_instances`].
    pub model: ModelId,
    /// The edit's mesh record. `pages[group.page]` is the page a kernel writes through
    /// `orr_pack.mesh`'s `store_triangle` and `store_position`.
    pub mesh: GpuPtr<MeshGpu>,
    /// The groups held at level 0.
    pub groups: GpuPtr<EditGroupGpu>,
    pub group_count: u32,
}

/// What an edit reads of its source mesh.
pub(crate) struct EditSource {
    pub entry: u32,
    pub header: MeshHeader,
    pub groups: Vec<Group>,
    pub pages: Vec<StreamedPage>,
    /// Sections whose every triangle is non-opaque.
    pub non_opaque: Vec<bool>,
    /// A discrete mesh's level 0: its first mesh cluster and cluster count.
    pub finest_level: Option<(u32, u32)>,
}

/// An edit's page table and cluster entries, which a streaming source keeps in step for the
/// pages the edit has no copy of.
pub(crate) struct Mirror {
    pub table: GpuPtr<u64>,
    pub clusters: GpuPtr<ClusterRef>,
    /// By page: the edit holds a copy.
    pub own: Vec<bool>,
}

/// The pages an edit copies and the groups it holds at level 0: with `spheres`, every page
/// holding a group that meets one at any level, since the traversal reads each of them
/// around what is pinned; whole, the pages of level 0.
pub(crate) fn plan(
    groups: &[Group],
    spheres: Option<&[[f32; 4]]>,
) -> (Vec<u32>, Vec<EditGroupGpu>) {
    let meets = |g: &Group| {
        spheres.is_none_or(|spheres| {
            spheres.iter().any(|t| {
                let s = g.simplified;
                let d = [s.center[0] - t[0], s.center[1] - t[1], s.center[2] - t[2]];
                let r = s.radius + t[3];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r * r
            })
        })
    };
    let mut pages: Vec<u32> = groups
        .iter()
        .filter(|g| meets(g) && (spheres.is_some() || g.depth == 0))
        .map(|g| g.page)
        .collect();
    pages.sort_unstable();
    pages.dedup();
    let held = groups
        .iter()
        .filter(|g| g.depth == 0 && meets(g))
        .map(|g| EditGroupGpu {
            page: g.page,
            first_cluster: u32::from(g.first_cluster),
            cluster_count: u32::from(g.cluster_count),
            _pad: 0,
        })
        .collect();
    (pages, held)
}

/// The pages an edit of a discrete mesh copies and the groups it holds: all of level 0, the
/// mesh clusters `first..first + count`, as one group per page.
pub(crate) fn plan_level(
    pages: &[StreamedPage],
    (first, count): (u32, u32),
) -> (Vec<u32>, Vec<EditGroupGpu>) {
    let held: Vec<EditGroupGpu> = (0u32..)
        .zip(pages)
        .filter_map(|(page, p)| {
            let start = first.max(p.first);
            let end = (first + count).min(p.first + p.clusters);
            (start < end).then_some(EditGroupGpu {
                page,
                first_cluster: start - p.first,
                cluster_count: end - start,
                _pad: 0,
            })
        })
        .collect();
    (held.iter().map(|g| g.page).collect(), held)
}

/// An edit on the loading side: its pages' reads, decodes and copies.
pub(crate) struct EditTask {
    pub id: EditId,
    pub loading: std::sync::Weak<Loading>,
    pub device: Device,
    pub source: Arc<BlobSource>,
    pub stream: Arc<EditSource>,
    pub own: Vec<u32>,
    pub cancel: Arc<AtomicBool>,
    pub micromaps_per_submit: u32,
    pub done: Mutex<EditDone>,
    pub outstanding: AtomicUsize,
}

/// What the jobs of an edit have made.
#[derive(Default)]
pub(crate) struct EditDone {
    pub block: Option<(Allocation<u8>, Vec<u64>)>,
    /// The pages decoded, laid out as `block`.
    pub staging: Option<Allocation<u8>>,
    /// By own page.
    pub clusters: Vec<Vec<ClusterInfo>>,
    pub regions: Vec<Option<PageMicromap>>,
    pub error: Option<Error>,
}

/// A finished edit, handed to the render thread once its copies have landed.
pub(crate) struct EditBatch {
    pub id: EditId,
    pub point: Option<TimelinePoint>,
    pub result: Result<EditLoaded, Arc<Error>>,
}

impl EditBatch {
    pub(crate) fn landed(&self, device: &Device) -> bool {
        landed(device, self.point)
    }
}

/// An edit's pages, resident in its block.
pub(crate) struct EditLoaded {
    pub _block: Allocation<u8>,
    /// By own page: its address, clusters and micromap.
    pub addresses: Vec<u64>,
    pub clusters: Vec<Vec<ClusterInfo>>,
    pub micromaps: Vec<Option<(Micromap, Allocation<u8>)>>,
}

impl EditTask {
    fn end(self: &Arc<Self>) {
        if self.outstanding.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(l) = self.loading.upgrade()
        {
            l.send(Msg::EditDone(Arc::clone(self)));
        }
    }
}

/// Starts an edit's page reads on the loading thread.
pub(crate) fn start(task: &Arc<EditTask>) {
    let s = &task.stream;
    let mut at = 0u64;
    let places: Vec<u64> = task
        .own
        .iter()
        .map(|&p| {
            let here = at;
            at += u64::from(s.pages[p as usize].bytes).next_multiple_of(256);
            here
        })
        .collect();
    let bytes = at.max(16);
    let d = &task.device;
    let blocks = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, bytes, "mesh edit pages")
        .and_then(|b| {
            Ok((
                b,
                d.gpu_malloc::<u8>(MemoryKind::Upload, bytes, "mesh edit staging")?,
            ))
        });
    let (block, staging) = match blocks {
        Ok(b) => b,
        Err(e) => {
            task.done.lock().unwrap().error = Some(e.into());
            task.end();
            return;
        }
    };
    let base = block.gpu().address().0;
    let mapped = Block::of(&staging);
    {
        let mut done = task.done.lock().unwrap();
        done.block = Some((block, places.iter().map(|o| base + o).collect()));
        done.staging = Some(staging);
        done.clusters = vec![Vec::new(); task.own.len()];
        done.regions = vec![None; task.own.len()];
    }
    let archive = &task.source.archive;
    let blobs = archive.blobs(&archive.entries()[s.entry as usize]);
    for (i, &page) in task.own.iter().enumerate() {
        let p = s.pages[page as usize];
        let Some(&b) = blobs.get(p.blob as usize) else {
            let e = orr_pack::Error::Malformed("mesh page outside its entry");
            task.done.lock().unwrap().error = Some(e.into());
            continue;
        };
        task.outstanding.fetch_add(1, Ordering::AcqRel);
        let t = Arc::clone(task);
        let at = places[i];
        read_stored(
            &task.source,
            b,
            None,
            Box::new(move |stored| {
                let result = stored.and_then(|stored| {
                    decoded(&t.source.archive, &b, None, stored, |bytes| {
                        PAGE.with_borrow_mut(|resident| {
                            resident.resize(p.bytes as usize, 0);
                            orr_pack::mesh::decode_page(bytes, resident)?;
                            let wrote = mapped.write_mapped(at, resident);
                            assert!(wrote, "staging is mapped");
                            let h: PageHeader =
                                bytemuck::pod_read_unaligned(&resident[..size_of::<PageHeader>()]);
                            let view = orr_pack::mesh::PageView::new(resident, &t.stream.header)?;
                            let records = (0..view.cluster_count()).map(|c| view.cluster(c));
                            let (clusters, region) = page_infos(&h, records, page);
                            let mut done = t.done.lock().unwrap();
                            done.clusters[i] = clusters;
                            done.regions[i] = region;
                            Ok(())
                        })
                    })
                });
                if let Err(e) = result {
                    t.done.lock().unwrap().error.get_or_insert(e);
                }
                t.end();
            }),
        );
    }
    task.end();
}

/// Ends an edit on the loading thread: submits its copies and builds its pages' micromaps.
pub(crate) fn finish(loading: &Loading, device: &Device, task: &Arc<EditTask>) {
    let done = std::mem::take(&mut *task.done.lock().unwrap());
    let EditDone {
        block,
        staging,
        clusters,
        regions,
        error,
    } = done;
    let cancelled = task.cancel.load(Ordering::Relaxed);
    let (result, point) = match (error, block) {
        (Some(e), _) => (Err(Arc::new(e)), None),
        (None, _) if cancelled => (Err(Arc::new(Error::UnknownEdit(task.id))), None),
        (None, None) => (Err(Arc::new(Error::UnknownEdit(task.id))), None),
        (None, Some((block, addresses))) => {
            let copies: Vec<StagedCopy> = staging
                .into_iter()
                .map(|src| StagedCopy::Buffer {
                    src,
                    dst: block.gpu().address().0,
                })
                .collect();
            let built = submit_copies(device, &copies, &[], None).and_then(|copied| {
                let pages: Vec<(u64, Option<PageMicromap>)> =
                    addresses.iter().copied().zip(regions).collect();
                accel::page_micromaps(device, &pages, task.micromaps_per_submit, copied)
                    .map(|(m, point)| (m, point.or(copied)))
            });
            match built {
                Ok((micromaps, point)) => (
                    Ok(EditLoaded {
                        _block: block,
                        addresses,
                        clusters,
                        micromaps,
                    }),
                    point,
                ),
                Err(e) => (Err(Arc::new(e)), None),
            }
        }
    };
    loading.push_edit(EditBatch {
        id: task.id,
        point,
        result,
    });
}

/// One implicit CLAS build of an edit and where its results go.
struct Build {
    batch: ClasBatch,
    jobs: Vec<EditJob>,
    /// Offset of its implicit results in the edit's CLAS block.
    at: u64,
}

/// An edit's CLAS: the clusters that rebuild when it is dirty and those built once, and what
/// their results fill.
pub(crate) struct EditClas {
    edited: Vec<Build>,
    frozen: Vec<Build>,
    block: Option<Allocation<u8>>,
    /// Triangles one rebuild takes.
    pub triangles: u32,
    pub built: bool,
}

/// What [`EditClas::new`] builds from.
pub(crate) struct ClasInputs<'a> {
    /// By own page: its address, micromap address (or 0) and clusters.
    pub pages: Vec<(u64, u64, &'a [ClusterInfo])>,
    /// The cluster id of each own page's first cluster.
    pub cluster_bases: Vec<u32>,
    pub header: &'a MeshHeader,
    pub non_opaque: &'a [bool],
    /// Clusters of coarser levels are built once only when the cut may take them.
    pub whole: bool,
    pub clas_triangles: u32,
}

impl EditClas {
    pub(crate) fn new(device: &Device, i: &ClasInputs<'_>) -> Result<Self> {
        let caps = device.caps();
        let traced = caps.cluster.max_tris > 0 && caps.cluster.max_partitions > 0;
        let omm = caps.omm && caps.cluster.omm_in_clusters;
        let step = i.header.step();
        let truncate_bits = u32::from(i.header.clas_truncate_bits);
        let mut bytes = 0u64;
        let mut lists = [Vec::new(), Vec::new()];
        for (k, list) in lists.iter_mut().enumerate() {
            let level0 = k == 0;
            if !traced || (!level0 && i.whole) {
                continue;
            }
            for with_omm in [false, true] {
                let mut members = Vec::new();
                for (pi, &(_, micromap, clusters)) in i.pages.iter().enumerate() {
                    for (local, c) in clusters.iter().enumerate() {
                        let uses =
                            omm && c.micromap && micromap != 0 && !i.non_opaque[c.section as usize];
                        if uses == with_omm && (c.self_error == 0.0) == level0 && c.tri_count > 0 {
                            members.push((pi, local));
                        }
                    }
                }
                let tris: Vec<u32> = members
                    .iter()
                    .map(|&(pi, l)| i.pages[pi].2[l].tri_count)
                    .collect();
                for range in accel::batches(&tris, i.clas_triangles) {
                    let run = &members[range];
                    let sources = run.iter().map(|&(pi, local)| {
                        let (page, micromap, clusters) = i.pages[pi];
                        let c = &clusters[local];
                        ClasSource {
                            page,
                            micromap,
                            cluster: c,
                            cluster_id: i.cluster_bases[pi] + local as u32,
                            step,
                            truncate_bits,
                            non_opaque: i.non_opaque[c.section as usize],
                        }
                    });
                    let batch = accel::clas_batch(sources, ClusterBuild::Clas { omm: with_omm });
                    let jobs = run
                        .iter()
                        .map(|&(pi, local)| EditJob {
                            // SAFETY: the page lives in the edit's block.
                            page: unsafe { GpuPtr::from_address(GpuAddress(i.pages[pi].0)) },
                            local: local as u32,
                            cluster_id: i.cluster_bases[pi] + local as u32,
                            step,
                            _pad: 0,
                        })
                        .collect();
                    let at = bytes;
                    bytes += device
                        .rtas_sizes(&batch.desc)
                        .result_max
                        .next_multiple_of(256);
                    list.push(Build { batch, jobs, at });
                }
            }
        }
        let [edited, frozen] = lists;
        let block = (bytes > 0)
            .then(|| device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, bytes, "mesh edit CLAS"))
            .transpose()?;
        let triangles = edited.iter().map(|b| b.batch.triangles).sum();
        Ok(Self {
            edited,
            frozen,
            block,
            triangles,
            built: false,
        })
    }
}

/// The frame's edit rebuilds: positions, flags and scratch regions of one grow-only block.
pub(crate) struct EditRecorder<'a> {
    pub device: &'a Device,
    pub window: &'a UploadWindow,
    pub kernels: &'a AccelKernels,
    pub clusters: GpuPtr<ClusterRef>,
}

impl EditRecorder<'_> {
    /// Records the rebuild of `edits` on `list`; `scratch` is the grow-only block every frame
    /// reuses, on the one queue whose lists start with a full barrier.
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        edits: &mut [&mut EditClas],
        scratch: &mut Option<Allocation<u8>>,
    ) -> Result<()> {
        struct Regions {
            vertices: u64,
            flags: u64,
            omm: u64,
            records: u64,
            count: u64,
            scratch: u64,
        }
        let d = self.device;
        let stride = u64::from(d.rtas_args_stride(RtasOp::BuildClas));
        let align = scratch_align(d);
        let mut size = 0u64;
        let mut region = |bytes: u64| {
            let at = size.next_multiple_of(align);
            size = at + bytes.max(16);
            at
        };
        let mut todo: Vec<(&Build, Regions, bool, GpuAddress)> = Vec::new();
        for e in edits.iter() {
            let block = e.block.as_ref().map_or(0, |b| b.gpu().address().0);
            let frozen = if e.built { &[][..] } else { &e.frozen[..] };
            for (b, edited) in e
                .edited
                .iter()
                .map(|b| (b, true))
                .chain(frozen.iter().map(|b| (b, false)))
            {
                let sizes = d.rtas_sizes(&b.batch.desc);
                todo.push((
                    b,
                    Regions {
                        vertices: region(12 * u64::from(b.batch.vertices)),
                        flags: region(4 * u64::from(b.batch.triangles)),
                        omm: region(4 * u64::from(b.batch.triangles)),
                        records: region(stride * b.jobs.len() as u64),
                        count: region(4),
                        scratch: region(sizes.scratch),
                    },
                    edited,
                    GpuAddress(block + b.at),
                ));
            }
        }
        if todo.is_empty() {
            for e in edits.iter_mut() {
                e.built = true;
            }
            return Ok(());
        }
        if scratch.as_ref().is_none_or(|s| s.size() < size) {
            *scratch = Some(d.gpu_malloc::<u8>(
                MemoryKind::DeviceLocal,
                size.next_power_of_two().max(1 << 20),
                "mesh edit scratch",
            )?);
        }
        let base = scratch.as_ref().expect("made above").gpu().address().0;
        let ptr = |at: u64| -> GpuPtr<u8> {
            // SAFETY: the address lies in the scratch block.
            unsafe { GpuPtr::from_address(GpuAddress(base + at)) }
        };
        let mut roots = Vec::with_capacity(todo.len());
        for (b, r, edited, dst) in &todo {
            let n = b.jobs.len() as u32;
            let (jobs, out) = self.window.slice::<EditJob>(d, n)?;
            out.copy_from_slice(&b.jobs);
            let (addresses, _) = self.window.slice::<u64>(d, n)?;
            let root = self.window.put_root(
                d,
                &EditRoot {
                    jobs,
                    clusters: self.clusters,
                    addresses,
                    count: n,
                    _pad: 0,
                },
            )?;
            if *edited {
                list.dispatch(&self.kernels.edit_bounds, root, [n.div_ceil(64), 1, 1]);
            }
            let (clas_jobs, out) = self.window.slice::<ClasJob>(d, n)?;
            out.copy_from_slice(&b.batch.jobs);
            let clas = self.window.put_root(
                d,
                &AccelRoot {
                    clas: clas_jobs,
                    records: ptr(r.records),
                    record_count: ptr(r.count).cast(),
                    vertices: ptr(r.vertices).cast(),
                    triangle_flags: ptr(r.flags).cast(),
                    omm_indices: ptr(r.omm).cast(),
                    count: n,
                    ..bytemuck::Zeroable::zeroed()
                },
            )?;
            roots.push((root, clas, addresses, *dst));
        }
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        for (&(_, clas, ..), (b, ..)) in roots.iter().zip(&todo) {
            list.dispatch(&self.kernels.clas, clas, [b.jobs.len() as u32, 1, 1]);
        }
        list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
        for (&(_, _, addresses, dst), (b, r, ..)) in roots.iter().zip(&todo) {
            list.rtas_execute(
                &b.batch.desc,
                &RtasIo {
                    args: GpuAddress(base + r.records),
                    count: ptr(r.count).cast(),
                    dst_implicit: dst,
                    dst_addresses: addresses,
                    dst_sizes: GpuPtr::NULL,
                    scratch: GpuAddress(base + r.scratch),
                    src: GpuAddress(0),
                },
            );
        }
        list.barrier(
            Stages::AS_BUILD,
            Stages::COMPUTE | Stages::AS_BUILD,
            Hazards::ACCEL_STRUCT,
        );
        for (&(root, ..), (b, ..)) in roots.iter().zip(&todo) {
            let n = b.jobs.len() as u32;
            list.dispatch(&self.kernels.edit_refs, root, [n.div_ceil(64), 1, 1]);
        }
        list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        for e in edits.iter_mut() {
            e.built = true;
        }
        Ok(())
    }
}

/// An edit's small tables in one mapped block: the pin spheres, the groups, the one empty
/// cached cut its copies fall back on and, when its source streams, its page stamps.
pub(crate) struct EditTables {
    pub _block: Allocation<u8>,
    pub edit: GpuPtr<EditGpu>,
    pub groups: GpuPtr<EditGroupGpu>,
    pub rt_cut: GpuPtr<crate::RtCutGpu>,
    pub stamps: GpuPtr<u32>,
}

impl EditTables {
    pub(crate) fn new(
        device: &Device,
        spheres: Option<&[[f32; 4]]>,
        groups: &[EditGroupGpu],
        pages: u32,
        stamps: bool,
    ) -> Result<Self> {
        let sphere_bytes = 16 * spheres.map_or(0, <[_]>::len) as u64;
        let group_bytes = size_of_val(groups) as u64;
        let stamp_bytes = if stamps { 4 * u64::from(pages) } else { 0 };
        let edit_at = 0u64;
        let spheres_at = (size_of::<EditGpu>() as u64).next_multiple_of(16);
        let groups_at = spheres_at + sphere_bytes;
        let cut_at = (groups_at + group_bytes).next_multiple_of(16);
        let stamps_at = cut_at + size_of::<crate::RtCutGpu>() as u64;
        let bytes = stamps_at + stamp_bytes;
        let a = device.gpu_malloc::<u8>(MemoryKind::Upload, bytes, "mesh edit tables")?;
        let block = Block::of(&a);
        let base = block.gpu();
        let at = |o: u64| GpuAddress(base + o);
        // SAFETY: each address lies in the block, at its table's offset.
        let (spheres_gpu, edit, groups_gpu, rt_cut, stamps_gpu) = unsafe {
            (
                GpuPtr::<[f32; 4]>::from_address(at(spheres_at)),
                GpuPtr::<EditGpu>::from_address(at(edit_at)),
                GpuPtr::<EditGroupGpu>::from_address(at(groups_at)),
                GpuPtr::<crate::RtCutGpu>::from_address(at(cut_at)),
                GpuPtr::<u32>::from_address(at(stamps_at)),
            )
        };
        let record = EditGpu {
            spheres: spheres_gpu,
            groups: groups_gpu,
            sphere_count: spheres.map_or(0, <[_]>::len) as u32,
            group_count: groups.len() as u32,
            whole: u32::from(spheres.is_none()),
            _pad: 0,
        };
        let cut: crate::RtCutGpu = bytemuck::Zeroable::zeroed();
        let wrote = block.write_mapped(edit_at, bytemuck::bytes_of(&record))
            && block.write_mapped(
                spheres_at,
                bytemuck::cast_slice(spheres.unwrap_or_default()),
            )
            && block.write_mapped(groups_at, bytemuck::cast_slice(groups))
            && block.write_mapped(cut_at, bytemuck::bytes_of(&cut))
            && block.write_mapped(stamps_at, &vec![0; stamp_bytes as usize]);
        assert!(wrote, "edit tables are mapped");
        Ok(Self {
            _block: a,
            edit,
            groups: groups_gpu,
            rt_cut,
            stamps: if stamps { stamps_gpu } else { GpuPtr::NULL },
        })
    }
}
