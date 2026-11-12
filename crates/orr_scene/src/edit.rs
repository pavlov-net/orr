//! Runtime mesh edits. An edit is a model of its own over a mesh of its own: the pages that
//! hold what it touches are per-edit copies a consumer kernel writes at the scene compute
//! point, whose clusters take cluster ids of the edit's, and every other page table entry and
//! page base follows the source mesh, whose clusters it traces under the source's ids. Edited
//! clusters stay at level 0 and their CLAS rebuild on the render queue right after the
//! scene compute point of a frame that marks the edit dirty; the edit's copies gather their
//! ray-tracing cut every frame, since cached cuts hold unedited CLAS. A skinned mesh's edit
//! rebuilds cluster templates instead, which its copies instantiate. The block of an edit's
//! pages also holds their state before the frame's kernels, which motion reads.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use orr_gpu::{
    Allocation, CommandList, Device, GpuAddress, GpuPtr, Hazards, MemoryKind, Micromap, RtasIo,
    Stages, TimelinePoint,
};
use orr_math::Vec3;
use orr_pack::Aabb;
use orr_pack::mesh::{Cluster, Group, MeshHeader, PageHeader};

use crate::accel::{
    self, AccelKernels, ClasBatch, ClasSource, ClusterBuild, ClusterInfo, PageMicromap,
};
use crate::loading::{BlobSource, Loading, Msg, Queue, decoded, landed, read_stored};
use crate::memory::{Block, UploadWindow};
use crate::mount::{PAGE, StagedCopy, page_infos, submit_copies};
use crate::producer::scratch_align;
use crate::scene::ModelId;
use crate::shaders::accel::{AccelRoot, ClasJob};
use crate::shaders::edit::{EditJob, EditRoot};
use crate::stream::StreamedPage;
use crate::{ClusterRef, EditGpu, EditGroupGpu, Error, PartGpu, Result, SkinMeshGpu};

/// Spheres an edit tests its groups against one by one; more pin the whole mesh.
pub(crate) const EDIT_SPHERES: usize = 64;

/// What [`crate::Scenes::create_edit`] edits.
#[derive(Clone, Copy, Debug)]
pub struct EditDesc<'a> {
    pub model: ModelId,
    /// Spheres (center xyz, radius, mesh space) around what the edit changes; `None`, or more
    /// than 64, the whole mesh. Every level 0 group meeting one is held at level 0 and handed
    /// to the consumer kernel; the rest of the mesh keeps its level of detail. A mesh cooked
    /// with discrete levels has no groups to choose from and a skinned mesh's copies take one
    /// cut: their edits hold all of level 0 and their copies draw nothing coarser.
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
    /// The edit's model: its copies draw and trace the edited mesh. They are placed like
    /// the source's, with [`crate::SceneWriter::add_instances`] and its poses, or with
    /// [`crate::SceneWriter::add_skinned`] for a skinned mesh.
    pub model: ModelId,
    /// The edit's mesh: its record is `SceneRoot::meshes[mesh]`, read through the root of the
    /// frame, since the mesh table moves as it grows. `pages[group.page]` of the record is the
    /// page a kernel writes through `orr_pack.mesh`'s `store_triangle` and `store_position`,
    /// after `store_anchor` when a vertex goes below its cluster's anchor. Vertices it moves have
    /// motion from where they were the frame before.
    pub mesh: u32,
    /// The groups held at level 0.
    pub groups: GpuPtr<EditGroupGpu>,
    pub group_count: u32,
}

/// What an edit reads of its source mesh.
pub(crate) struct EditSource {
    pub(crate) entry: u32,
    pub(crate) header: MeshHeader,
    pub(crate) groups: Arc<[Group]>,
    pub(crate) pages: Vec<StreamedPage>,
    /// Sections whose every triangle is non-opaque.
    pub(crate) non_opaque: Vec<bool>,
    /// A discrete mesh's level 0: its first mesh cluster and cluster count.
    pub(crate) finest_level: Option<(u32, u32)>,
}

/// An edit's entries of the mesh page table, which a streaming source keeps in step for the
/// pages the edit has no copy of.
pub(crate) struct Mirror {
    /// Its first entry.
    pub(crate) entries: u32,
    /// By page: the edit holds a copy.
    pub(crate) own: Vec<bool>,
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
                let r = s.radius + t[3];
                Vec3::from(s.center).distance_squared(Vec3::from_slice(t)) <= r * r
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

/// The mesh clusters of level 0 in an edit's `own` pages, ascending.
pub(crate) fn level0_clusters(
    own: &[u32],
    pages: &[StreamedPage],
    clusters: &[Vec<ClusterInfo>],
) -> Vec<u32> {
    own.iter()
        .zip(clusters)
        .flat_map(|(&page, clusters)| {
            clusters
                .iter()
                .zip(pages[page as usize].first..)
                .filter(|(c, _)| {
                    c.self_error == 0.0 && c.tri_count > 0 && c.flags & Cluster::MERGED == 0
                })
                .map(|(_, id)| id)
        })
        .collect()
}

/// An edit on the loading side: its pages' reads, decodes and copies.
pub(crate) struct EditTask {
    pub(crate) id: EditId,
    pub(crate) loading: std::sync::Weak<Loading>,
    pub(crate) device: Device,
    pub(crate) source: Arc<BlobSource>,
    pub(crate) stream: Arc<EditSource>,
    pub(crate) own: Vec<u32>,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) micromaps_per_submit: u32,
    pub(crate) done: Mutex<EditDone>,
    pub(crate) outstanding: AtomicUsize,
}

/// What the jobs of an edit have made.
#[derive(Default)]
pub(crate) struct EditDone {
    block: Option<(Allocation<u8>, Vec<u64>)>,
    /// The pages decoded, laid out as `block`.
    staging: Option<Allocation<u8>>,
    /// By own page.
    clusters: Vec<Vec<ClusterInfo>>,
    regions: Vec<Option<PageMicromap>>,
    error: Option<Error>,
}

/// A finished edit, handed to the render thread once its copies have landed.
pub(crate) struct EditBatch {
    pub(crate) id: EditId,
    pub(crate) point: Option<TimelinePoint>,
    pub(crate) result: Result<EditLoaded, Arc<Error>>,
}

impl EditBatch {
    pub(crate) fn landed(&self, device: &Device) -> bool {
        landed(device, self.point)
    }
}

/// An edit's pages, resident in its block.
pub(crate) struct EditLoaded {
    pub(crate) block: Allocation<u8>,
    /// Bytes of the pages; their state before this frame's kernels follows them.
    pub(crate) pages_bytes: u64,
    /// By own page: its address, clusters and micromap.
    pub(crate) addresses: Vec<u64>,
    pub(crate) clusters: Vec<Vec<ClusterInfo>>,
    pub(crate) micromaps: Vec<Option<(Micromap, Allocation<u8>)>>,
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
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, 2 * bytes, "mesh edit pages")
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
            Queue::Stream,
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
                            if view.cluster_count() != p.clusters {
                                let e = orr_pack::Error::Malformed("edit page cluster count");
                                return Err(e.into());
                            }
                            let records: Vec<Cluster> =
                                (0..view.cluster_count()).map(|c| view.cluster(c)).collect();
                            let (clusters, region) =
                                page_infos(&h, &records, page, &t.stream.groups)?;
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
    let canceled = task.cancel.load(Ordering::Relaxed);
    let (result, point) = match (error, block) {
        (Some(e), _) => (Err(Arc::new(e)), None),
        (None, _) if canceled => (Err(Arc::new(Error::UnknownEdit(task.id))), None),
        (None, None) => (Err(Arc::new(Error::UnknownEdit(task.id))), None),
        (None, Some((block, addresses))) => {
            let pages_bytes = block.size() / 2;
            let copies: Vec<StagedCopy> = staging
                .into_iter()
                .map(|src| StagedCopy::Buffer {
                    src,
                    dst: block.gpu().address().0,
                })
                .collect();
            let built = submit_copies(device, &copies, &[]).and_then(|copied| {
                let pages: Vec<(u64, Option<PageMicromap>)> =
                    addresses.iter().copied().zip(regions).collect();
                accel::page_micromaps(device, &pages, task.micromaps_per_submit, copied)
                    .map(|(m, point)| (m, point.or(copied)))
            });
            match built {
                Ok((micromaps, point)) => (
                    Ok(EditLoaded {
                        block,
                        pages_bytes,
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

/// An edit's CLAS, or a skinned mesh's edit's templates: the clusters that rebuild when it is
/// dirty and those built once, and what their results fill.
pub(crate) struct EditClas {
    edited: Vec<Build>,
    frozen: Vec<Build>,
    block: Option<Allocation<u8>>,
    /// A skinned mesh's edit: its template table, by mesh cluster.
    templates: GpuPtr<u64>,
    /// Triangles one rebuild takes.
    pub(crate) triangles: u32,
    pub(crate) built: bool,
}

/// What [`EditClas::new`] builds from.
pub(crate) struct ClasInputs<'a> {
    /// By own page: its address, micromap address (or 0) and clusters.
    pub(crate) pages: Vec<(u64, u64, &'a [ClusterInfo])>,
    /// The cluster id and the mesh cluster of each own page's first cluster.
    pub(crate) cluster_bases: Vec<u32>,
    pub(crate) firsts: Vec<u32>,
    pub(crate) header: &'a MeshHeader,
    pub(crate) non_opaque: &'a [bool],
    /// Clusters of coarser levels are built once only when the cut may take them.
    pub(crate) whole: bool,
    pub(crate) clas_triangles: u32,
    /// A skinned mesh's edit: the template table its level 0's builds fill.
    pub(crate) templates: Option<GpuPtr<u64>>,
}

impl EditClas {
    pub(crate) fn new(device: &Device, i: &ClasInputs<'_>) -> Result<Self> {
        let caps = device.caps();
        let traced = caps.cluster.max_tris > 0 && caps.cluster.max_partitions > 0;
        let template = i.templates.is_some();
        let omm = !template && caps.omm && caps.cluster.omm_in_clusters;
        let step = i.header.step();
        // Skinned positions are off the position grid, so templates keep every bit.
        let truncate_bits = if template {
            0
        } else {
            u32::from(i.header.clas_truncate_bits)
        };
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
                    let build = if template {
                        ClusterBuild::Template
                    } else {
                        ClusterBuild::Clas { omm: with_omm }
                    };
                    let batch = accel::clas_batch(sources, build);
                    let jobs = run
                        .iter()
                        .map(|&(pi, local)| EditJob {
                            // SAFETY: the page lives in the edit's block.
                            page: unsafe { GpuPtr::from_address(GpuAddress(i.pages[pi].0)) },
                            local: local as u32,
                            cluster_id: i.cluster_bases[pi] + local as u32,
                            step,
                            mesh_cluster: i.firsts[pi] + local as u32,
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
            templates: i.templates.unwrap_or(GpuPtr::NULL),
            triangles,
            built: false,
        })
    }
}

/// The frame's edit rebuilds: positions, flags and scratch regions of one grow-only block.
pub(crate) struct EditRecorder<'a> {
    pub(crate) device: &'a Device,
    pub(crate) window: &'a UploadWindow,
    pub(crate) kernels: &'a AccelKernels,
    pub(crate) clusters: GpuPtr<ClusterRef>,
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
            bounds: u64,
            records: u64,
            count: u64,
            scratch: u64,
        }
        struct Todo<'b> {
            build: &'b Build,
            regions: Regions,
            edited: bool,
            dst: GpuAddress,
            templates: GpuPtr<u64>,
        }
        let d = self.device;
        let align = scratch_align(d);
        let mut size = 0u64;
        let mut region = |bytes: u64| {
            let at = size.next_multiple_of(align);
            size = at + bytes.max(16);
            at
        };
        let mut todo: Vec<Todo<'_>> = Vec::new();
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
                let stride = u64::from(d.rtas_args_stride(b.batch.desc.op));
                todo.push(Todo {
                    build: b,
                    regions: Regions {
                        vertices: region(12 * u64::from(b.batch.vertices)),
                        flags: region(4 * u64::from(b.batch.triangles)),
                        omm: region(4 * u64::from(b.batch.triangles)),
                        bounds: region(32 * b.jobs.len() as u64),
                        records: region(stride * b.jobs.len() as u64),
                        count: region(4),
                        scratch: region(sizes.scratch),
                    },
                    edited,
                    dst: GpuAddress(block + b.at),
                    templates: e.templates,
                });
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
        for t in &todo {
            let (b, r) = (t.build, &t.regions);
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
                    templates: t.templates,
                    count: n,
                    _pad0: 0,
                    _pad1: 0,
                },
            )?;
            if t.edited {
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
                    bounds: ptr(r.bounds).cast(),
                    count: n,
                    ..bytemuck::Zeroable::zeroed()
                },
            )?;
            roots.push((root, clas, addresses, t.dst));
        }
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        for (&(_, clas, ..), t) in roots.iter().zip(&todo) {
            let groups = t.build.jobs.len() as u32;
            list.dispatch(&self.kernels.clas, clas, [groups, 1, 1]);
        }
        list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
        for (&(_, _, addresses, dst), t) in roots.iter().zip(&todo) {
            let r = &t.regions;
            list.rtas_execute(
                &t.build.batch.desc,
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
        for (&(root, ..), t) in roots.iter().zip(&todo) {
            let n = t.build.jobs.len() as u32;
            list.dispatch(&self.kernels.edit_refs, root, [n.div_ceil(64), 1, 1]);
        }
        list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        for e in edits.iter_mut() {
            e.built = true;
        }
        Ok(())
    }
}

/// What a skinned mesh's edit changes of its skin table: its one cut is level 0, and its
/// templates are its own.
pub(crate) struct EditSkin<'a> {
    pub(crate) source: SkinMeshGpu,
    /// Level 0's mesh clusters, ascending.
    pub(crate) cut: &'a [u32],
}

/// What [`EditTables::new`] lays out.
pub(crate) struct EditTablesDesc<'a> {
    pub(crate) spheres: Option<&'a [[f32; 4]]>,
    pub(crate) groups: &'a [EditGroupGpu],
    /// The block of its pages and their bytes.
    pub(crate) pages: (GpuPtr<u8>, u64),
    /// Its mesh's clusters, which a skinned mesh's edit takes a template of each.
    pub(crate) clusters: u32,
    /// The cooked sets' row of level 0: past the pinned cut's row on a DAG mesh.
    pub(crate) row: u32,
    /// Its animated parts, which share the box its edited triangles stay inside.
    pub(crate) parts: &'a [PartGpu],
    pub(crate) skin: Option<EditSkin<'a>>,
}

/// An edit's small tables in one mapped block: the pin spheres, the groups, the one empty
/// cached cut its copies fall back on, its animated parts and, when its source is skinned, its
/// skin table.
pub(crate) struct EditTables {
    _block: Allocation<u8>,
    pub(crate) edit: GpuPtr<EditGpu>,
    pub(crate) groups: GpuPtr<EditGroupGpu>,
    pub(crate) rt_cut: GpuPtr<crate::RtCutGpu>,
    pub(crate) parts: GpuPtr<PartGpu>,
    pub(crate) skin: GpuPtr<SkinMeshGpu>,
    pub(crate) templates: GpuPtr<u64>,
}

impl EditTables {
    pub(crate) fn new(device: &Device, d: &EditTablesDesc<'_>) -> Result<Self> {
        let spheres = d.spheres.unwrap_or_default();
        let mut bytes = 0u64;
        let mut take = |n: u64| {
            let at = bytes.next_multiple_of(16);
            bytes = at + n;
            at
        };
        let edit_at = take(size_of::<EditGpu>() as u64);
        let spheres_at = take(size_of_val(spheres) as u64);
        let groups_at = take(size_of_val(d.groups) as u64);
        let cut_at = take(size_of::<crate::RtCutGpu>() as u64);
        let parts_at = take(size_of_val(d.parts) as u64);
        let skin_at = take(size_of::<SkinMeshGpu>() as u64);
        let templates_bytes = if d.skin.is_some() {
            8 * u64::from(d.clusters)
        } else {
            0
        };
        let templates_at = take(templates_bytes);
        let cut = d.skin.as_ref().map_or(&[][..], |s| s.cut);
        let cut_clusters_at = take(size_of_val(cut) as u64);
        let cut_first_at = take(8);
        let cut_error_at = take(4);
        let a = device.gpu_malloc::<u8>(MemoryKind::Upload, bytes, "mesh edit tables")?;
        let block = Block::of(&a);
        let at = |o: u64| -> GpuPtr<u8> {
            // SAFETY: each offset above lies in the block.
            unsafe { GpuPtr::from_address(GpuAddress(block.gpu() + o)) }
        };
        let record = EditGpu {
            spheres: at(spheres_at).cast(),
            groups: at(groups_at).cast(),
            pages: d.pages.0,
            pages_bytes: d.pages.1,
            sphere_count: spheres.len() as u32,
            group_count: d.groups.len() as u32,
            whole: u32::from(d.spheres.is_none()),
            _pad: 0,
        };
        let empty_cut = crate::RtCutGpu {
            row: d.row,
            ..bytemuck::Zeroable::zeroed()
        };
        let wrote = block.write_mapped(edit_at, bytemuck::bytes_of(&record))
            && block.write_mapped(spheres_at, bytemuck::cast_slice(spheres))
            && block.write_mapped(groups_at, bytemuck::cast_slice(d.groups))
            && block.write_mapped(cut_at, bytemuck::bytes_of(&empty_cut))
            && block.write_mapped(parts_at, bytemuck::cast_slice(d.parts));
        assert!(wrote, "edit tables are mapped");
        let skin = d.skin.as_ref().map(|s| SkinMeshGpu {
            templates: at(templates_at).cast(),
            cut_clusters: at(cut_clusters_at).cast(),
            cut_first: at(cut_first_at).cast(),
            cut_error: at(cut_error_at).cast(),
            cut_count: 1,
            ..s.source
        });
        if let Some(skin) = &skin {
            let wrote = block.write_mapped(skin_at, bytemuck::bytes_of(skin))
                && block.write_mapped(templates_at, &vec![0; templates_bytes as usize])
                && block.write_mapped(cut_clusters_at, bytemuck::cast_slice(cut))
                && block.write_mapped(cut_first_at, bytemuck::cast_slice(&[0, cut.len() as u32]))
                && block.write_mapped(cut_error_at, bytemuck::bytes_of(&0.0f32));
            assert!(wrote, "edit tables are mapped");
        }
        Ok(Self {
            edit: at(edit_at).cast(),
            groups: at(groups_at).cast(),
            rt_cut: at(cut_at).cast(),
            parts: at(parts_at).cast(),
            skin: if skin.is_some() {
                at(skin_at).cast()
            } else {
                GpuPtr::NULL
            },
            templates: at(templates_at).cast(),
            _block: a,
        })
    }
}
