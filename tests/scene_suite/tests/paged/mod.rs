//! A cooked DAG mesh's hierarchy split into a top page and one streamed node page, at a size a
//! test cooks quickly: the layout the cook writes for hierarchies over one node page.

use std::sync::Arc;

use orr_pack::mesh::{MeshHeader, MeshInfo, Node, NodePage};
use orr_pack::{Archive, ArchiveWriter, AssetId, BlobIn, Target};

/// `pack` with mesh `mesh`'s hierarchy, cooked resident or in one node page, split into the
/// top page the cook writes (the roots, and the children of every node above the leaf of a
/// group in a pinned page) and one streamed page of every other node, which every group with
/// its leaf there names.
pub(crate) fn split_hierarchy(pack: &Arc<[u8]>, mesh: AssetId) -> Arc<[u8]> {
    let a = Archive::from_memory(Arc::clone(pack)).unwrap();
    let mut w = ArchiveWriter::create(Target::Memory).unwrap();
    for e in a.entries() {
        if e.id != mesh {
            w.copy_entry(&a, e).unwrap();
            continue;
        }
        let blobs = a.blobs(e);
        let read = |i: usize| {
            let mut v = vec![0; blobs[i].raw_len as usize];
            a.read(&blobs[i], &mut v).unwrap();
            v
        };
        let mut info = MeshInfo::parse(&read(0)).unwrap();
        let nodes: Vec<Node> = match info.node_pages.as_slice() {
            [] => std::mem::take(&mut info.nodes),
            [p] => orr_pack::mesh::nodes(&read(p.blob as usize)).unwrap(),
            _ => panic!("the hierarchy is in one page"),
        };
        let children = |n: &Node| {
            n.child_first as usize..(n.child_first as usize + usize::from(n.child_count))
        };
        // Children follow their parents.
        let mut above = vec![false; nodes.len()];
        for (i, n) in nodes.iter().enumerate().rev() {
            above[i] = match usize::try_from(n.group) {
                Ok(g) => info.groups[g].page < info.header.pinned_pages,
                Err(_) => children(n).any(|c| above[c]),
            };
        }
        let roots = info
            .roots
            .iter()
            .filter(|&&r| r != MeshHeader::NO_ROOT)
            .count();
        let inner = nodes.iter().zip(&above).filter(|(n, _)| n.group < 0);
        let floor = roots
            + inner
                .clone()
                .filter(|&(_, &a)| a)
                .map(|(n, _)| usize::from(n.child_count))
                .sum::<usize>();
        for (n, &a) in inner {
            assert_eq!(
                children(n).end <= floor,
                a,
                "the cook puts the top page first"
            );
        }
        assert!(floor < nodes.len(), "nodes lie below the top page");
        for (i, n) in nodes.iter().enumerate() {
            if let Ok(g) = usize::try_from(n.group) {
                info.groups[g].node_page = u32::from(i >= floor);
            }
        }
        let first_blob = 1 + info.pages.len() as u32;
        info.node_pages = vec![
            NodePage {
                first_node: 0,
                node_count: floor as u32,
                blob: first_blob,
                parent_page: NodePage::TOP,
            },
            NodePage {
                first_node: floor as u32,
                node_count: (nodes.len() - floor) as u32,
                blob: first_blob + 1,
                parent_page: 0,
            },
        ];
        let blob0 = info.to_bytes();
        let pages: Vec<(Vec<u8>, bool)> = (1..first_blob as usize)
            .map(|i| (read(i), blobs[i].is_mount()))
            .collect();
        let (top, rest) =
            bytemuck::cast_slice::<Node, u8>(&nodes).split_at(floor * size_of::<Node>());
        let blob = |bytes, mount| BlobIn {
            mount,
            ..BlobIn::new(bytes)
        };
        let mut list = vec![blob(&blob0, true)];
        list.extend(pages.iter().map(|(p, mount)| blob(p, *mount)));
        list.extend([blob(top, true), blob(rest, false)]);
        w.add(a.name(e), e.class, e.schema, e.cook_key, &list)
            .unwrap();
    }
    w.finish().unwrap().unwrap()
}
