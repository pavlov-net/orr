//! Streamed units: the render thread's residency step at step 1, and the loading side's
//! reads that place each unit in the stream pool.
//!
//! At step 1 the render thread takes the stream batches whose copies have landed and
//! writes each unit's address into its table entry, hands the requests since the last step
//! to the [`Residency`] manager, writes null over the entries of the units it evicts (their
//! memory returns once the frames that could read it have completed) and sends the loads
//! to the loading thread. The frame's submit waits the taken batches' points.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use orr_gpu::{Device, GpuPtr, MemoryKind, QueueKind, TimelinePoint};

use crate::loading::{BlobSource, Loading, Msg, decoded, read_stored};
use crate::memory::{Pool, PoolSlot};
use crate::mount::{Apply, StagedCopy, submit_copies, write};
use crate::residency::{
    Key, Limits, Plan, Priority, Residency, ResidencyClass, ResidencyStats, UnitDesc,
};
use crate::{Budgets, Error, MountId, Result};

/// Where a streamed unit's bytes are and where its address goes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamUnit {
    pub entry: u32,
    pub blob: u32,
    pub bytes: u32,
    /// The table entry that holds the unit's address while it is resident, else null.
    pub table: GpuPtr<u64>,
}

struct MountStreams {
    source: Arc<BlobSource>,
    /// By unit; `None` for units the mount placed.
    units: Vec<Option<StreamUnit>>,
    /// Streamed units per mesh of the mount.
    meshes: Vec<Vec<u32>>,
    /// The unit of each node page, per mesh of the mount.
    node_pages: Vec<Vec<u32>>,
}

pub(crate) struct Streaming {
    residency: Residency,
    pool: Arc<Mutex<Pool>>,
    mounts: HashMap<MountId, MountStreams>,
    resident: HashMap<Key, (PoolSlot, GpuPtr<u64>)>,
    /// Slots freed once the frame has completed.
    retired: Vec<(u64, PoolSlot)>,
}

impl Streaming {
    pub(crate) fn new(device: &Device, b: &Budgets) -> Self {
        let mut budgets = [0; 4];
        budgets[ResidencyClass::MeshPages as usize] = b.mesh_page_pool;
        budgets[ResidencyClass::HierarchyPages as usize] = b.hierarchy_page_pool;
        budgets[ResidencyClass::Textures as usize] = b.texture_pool;
        budgets[ResidencyClass::InstancePages as usize] = b.instance_page_pool;
        let kind = if device.caps().rebar {
            MemoryKind::Upload
        } else {
            MemoryKind::DeviceLocal
        };
        Self {
            residency: Residency::new(Limits {
                budgets,
                io_requests: b.io_requests,
                io_bytes: b.io_bytes,
            }),
            pool: Arc::new(Mutex::new(Pool::new(kind, b.stream_block, "stream pool"))),
            mounts: HashMap::new(),
            resident: HashMap::new(),
            retired: Vec::new(),
        }
    }

    pub(crate) fn stats(&self) -> &ResidencyStats {
        self.residency.stats()
    }

    /// Adds a mount's units; `units[i].1` says how to load unit `i` when it starts absent.
    pub(crate) fn register(
        &mut self,
        mount: MountId,
        source: Arc<BlobSource>,
        units: Vec<(UnitDesc, Option<StreamUnit>)>,
        meshes: Vec<Vec<u32>>,
        node_pages: Vec<Vec<u32>>,
    ) -> Result<()> {
        let descs: Vec<UnitDesc> = units.iter().map(|(d, _)| d.clone()).collect();
        self.residency.register(mount, &descs)?;
        let units = units.into_iter().map(|(_, s)| s).collect();
        self.mounts.insert(
            mount,
            MountStreams {
                source,
                units,
                meshes,
                node_pages,
            },
        );
        Ok(())
    }

    /// Drops a mount's units; their memory is freed once `frame` has completed.
    pub(crate) fn unregister(&mut self, mount: MountId, frame: u64) {
        if self.mounts.remove(&mount).is_none() {
            return;
        }
        self.residency.unregister(mount);
        let gone: Vec<Key> = self
            .resident
            .keys()
            .filter(|k| k.mount == mount)
            .copied()
            .collect();
        for k in gone {
            let (slot, _) = self.resident.remove(&k).expect("resident");
            self.retired.push((frame, slot));
        }
    }

    /// Requests every streamed unit of mesh `mesh` of `mount` for the next step.
    pub(crate) fn prefetch(&mut self, mount: MountId, mesh: u32, cell_distance: u32) {
        let Some(m) = self.mounts.get(&mount) else {
            return;
        };
        let p = Priority {
            error: 0.0,
            cell_distance,
        };
        for &unit in &m.meshes[mesh as usize] {
            self.residency.request(Key { mount, unit }, p);
        }
    }

    /// Node page `page` of mesh `mesh` of `mount`: requested for the next step with the
    /// projected error it would refine, or read by a traversal in `frame`.
    pub(crate) fn node_page(
        &mut self,
        mount: MountId,
        mesh: u32,
        page: u32,
        request: Option<f32>,
        frame: u64,
    ) {
        let Some(&unit) = self
            .mounts
            .get(&mount)
            .and_then(|m| m.node_pages.get(mesh as usize))
            .and_then(|p| p.get(page as usize))
        else {
            return;
        };
        let k = Key { mount, unit };
        match request {
            Some(error) => self.residency.request(
                k,
                Priority {
                    error,
                    cell_distance: 0,
                },
            ),
            None => self.residency.touch(k, frame),
        }
    }

    /// Step 1: applies the landed batches, plans this step and sends its loads. Returns the
    /// points the frame's submit waits.
    pub(crate) fn step(
        &mut self,
        frame: u64,
        loading: &Arc<Loading>,
        w: &mut Apply<'_>,
    ) -> Result<Vec<TimelinePoint>> {
        let mut points = Vec::new();
        for batch in loading.take_streams(w.device) {
            points.extend(batch.point.map(|(_, p)| p));
            for l in batch.loads {
                let ok = l.result.is_ok();
                let live = self.residency.landed(l.key, ok, frame);
                match (live && ok, l.slot) {
                    (true, Some(slot)) => {
                        w.put(l.table, &[l.gpu])?;
                        self.resident.insert(l.key, (slot, l.table));
                    }
                    (_, Some(slot)) => self.pool.lock().unwrap().free(slot),
                    (_, None) => {}
                }
            }
        }
        let Plan { loads, evictions } = self.residency.plan(frame);
        for k in evictions {
            let (slot, table) = self.resident.remove(&k).expect("evicted unit is resident");
            w.put(table, &[0])?;
            self.retired.push((frame - 1, slot));
        }
        let loads: Vec<StreamLoad> = loads
            .into_iter()
            .map(|key| {
                let m = &self.mounts[&key.mount];
                StreamLoad {
                    key,
                    source: Arc::clone(&m.source),
                    unit: m.units[key.unit as usize].expect("a streamed unit"),
                }
            })
            .collect();
        if !loads.is_empty() {
            let task = StreamTask {
                pool: Arc::clone(&self.pool),
                device: w.device.clone(),
                outstanding: AtomicUsize::new(1),
                done: Mutex::new(Vec::with_capacity(loads.len())),
                copies: Mutex::new(Vec::new()),
                loads,
            };
            loading.send(Msg::Stream(Arc::new(task)));
        }
        Ok(points)
    }

    /// Frees the slots retired at or before `completed`.
    pub(crate) fn retire(&mut self, completed: u64) {
        let mut pool = self.pool.lock().unwrap();
        self.retired.retain(|&(frame, slot)| {
            let done = frame <= completed;
            if done {
                pool.free(slot);
            }
            !done
        });
    }
}

struct StreamLoad {
    key: Key,
    source: Arc<BlobSource>,
    unit: StreamUnit,
}

/// One step's loads on the loading side.
pub(crate) struct StreamTask {
    pool: Arc<Mutex<Pool>>,
    device: Device,
    loads: Vec<StreamLoad>,
    /// The start chain plus one per read.
    outstanding: AtomicUsize,
    done: Mutex<Vec<Landed>>,
    copies: Mutex<Vec<StagedCopy>>,
}

/// A finished load.
pub(crate) struct Landed {
    key: Key,
    slot: Option<PoolSlot>,
    gpu: u64,
    table: GpuPtr<u64>,
    result: Result<(), Arc<Error>>,
}

/// A stream task's loads, handed to the render thread once their copies have landed.
pub(crate) struct StreamBatch {
    pub point: Option<(QueueKind, TimelinePoint)>,
    loads: Vec<Landed>,
}

impl StreamTask {
    fn end(self: &Arc<Self>, loading: &Loading) {
        if self.outstanding.fetch_sub(1, Ordering::AcqRel) == 1 {
            loading.send(Msg::StreamDone(Arc::clone(self)));
        }
    }
}

/// Places a stream task's units on the loading thread: a pool slot each, then its read.
pub(crate) fn start(loading: &Arc<Loading>, task: &Arc<StreamTask>) {
    for load in &task.loads {
        let (key, u) = (load.key, load.unit);
        let landed = move |slot, gpu, result| Landed {
            key,
            slot,
            gpu,
            table: u.table,
            result,
        };
        let archive = &load.source.archive;
        let blob = archive
            .entries()
            .get(u.entry as usize)
            .and_then(|e| archive.blobs(e).get(u.blob as usize).copied())
            .ok_or(orr_pack::Error::Malformed(
                "streamed blob outside its entry",
            ));
        let blob = match blob {
            Ok(b) if b.codec() == orr_pack::Codec::GDeflate => Err(orr_pack::Error::GDeflate),
            b => b,
        };
        let b = match blob {
            Ok(b) => b,
            Err(e) => {
                task.done
                    .lock()
                    .unwrap()
                    .push(landed(None, 0, Err(Arc::new(e.into()))));
                continue;
            }
        };
        let (slot, block) = match task
            .pool
            .lock()
            .unwrap()
            .alloc(&task.device, u64::from(u.bytes))
        {
            Ok(s) => s,
            Err(e) => {
                task.done
                    .lock()
                    .unwrap()
                    .push(landed(None, 0, Err(Arc::new(e))));
                continue;
            }
        };
        task.outstanding.fetch_add(1, Ordering::AcqRel);
        let t = Arc::clone(task);
        let src = Arc::clone(&load.source);
        let l = Arc::downgrade(loading);
        read_stored(
            &load.source,
            b,
            Box::new(move |stored| {
                let result = stored.and_then(|stored| {
                    decoded(&src.archive, &b, stored, |bytes| {
                        if bytes.len() != u.bytes as usize {
                            return Err(orr_pack::Error::Malformed("streamed blob length").into());
                        }
                        if let Some(copy) = write(&t.device, block, 0, bytes)? {
                            t.copies.lock().unwrap().push(copy);
                        }
                        Ok(())
                    })
                });
                t.done.lock().unwrap().push(landed(
                    Some(slot),
                    block.gpu,
                    result.map_err(Arc::new),
                ));
                if let Some(l) = l.upgrade() {
                    t.end(&l);
                }
            }),
        );
    }
    task.end(loading);
}

/// Ends a stream task on the loading thread: submits its staged copies and hands the batch
/// to the render thread.
pub(crate) fn finish(loading: &Loading, device: &Device, task: &Arc<StreamTask>) {
    let copies = std::mem::take(&mut *task.copies.lock().unwrap());
    let mut loads = std::mem::take(&mut *task.done.lock().unwrap());
    let point = match submit_copies(device, &copies, &[]) {
        Ok(p) => p,
        Err(e) => {
            let e = Arc::new(e);
            for l in &mut loads {
                l.result = Err(Arc::clone(&e));
            }
            None
        }
    };
    loading.push_stream(StreamBatch { point, loads });
}
