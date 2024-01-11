use core::slice;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::marker;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use heed::types::Bytes;
use heed::{BytesDecode, BytesEncode, RoTxn};
use memmap2::Mmap;
use rand::seq::index;
use rand::Rng;
use roaring::RoaringBitmap;

use crate::internals::{KeyCodec, Leaf, NodeCodec};
use crate::key::{Prefix, PrefixCodec};
use crate::node::Node;
use crate::{Database, Distance, Error, ItemId, Result};

/// A structure to store the tree nodes out of the heed database.
pub struct TmpNodes<DE> {
    file: BufWriter<File>,
    ids: Vec<ItemId>,
    bounds: Vec<usize>,
    deleted: RoaringBitmap,
    _marker: marker::PhantomData<DE>,
}

impl<'a, DE: BytesEncode<'a>> TmpNodes<DE> {
    /// Creates an empty `TmpNodes`.
    pub fn new() -> heed::Result<TmpNodes<DE>> {
        Ok(TmpNodes {
            file: tempfile::tempfile().map(BufWriter::new)?,
            ids: Vec::new(),
            bounds: vec![0],
            deleted: RoaringBitmap::new(),
            _marker: marker::PhantomData,
        })
    }

    /// Creates an empty `TmpNodes` in the defined folder.
    pub fn new_in(path: &Path) -> heed::Result<TmpNodes<DE>> {
        Ok(TmpNodes {
            file: tempfile::tempfile_in(path).map(BufWriter::new)?,
            ids: Vec::new(),
            bounds: vec![0],
            deleted: RoaringBitmap::new(),
            _marker: marker::PhantomData,
        })
    }

    /// Add a new node in the file.
    /// Items do not need to be ordered.
    pub fn put(
        // TODO move that in the type
        &mut self,
        item: ItemId,
        data: &'a DE::EItem,
    ) -> heed::Result<()> {
        assert!(item != ItemId::MAX);
        let bytes = DE::bytes_encode(data).map_err(heed::Error::Encoding)?;
        self.file.write_all(&bytes)?;
        let last_bound = self.bounds.last().unwrap();
        self.bounds.push(last_bound + bytes.len());
        self.ids.push(item);

        // in the current algorithm, we should never insert a node that was deleted before
        debug_assert!(!self.deleted.contains(item));

        Ok(())
    }

    /// Mark a node to delete from the DB.
    pub fn remove_from_db(&mut self, item: ItemId) {
        self.deleted.insert(item);
    }

    /// Mark a node to delete from the DB and delete it from the tmp nodes as well.
    /// Panic if the node wasn't inserted in the tmp_nodes before calling this method.
    pub fn remove(&mut self, item: ItemId) -> heed::Result<()> {
        self.remove_from_db(item);
        // In the current algorithm, we're supposed to find the node in the two last positions.
        if let Some(el) = self.ids.iter_mut().rev().take(2).find(|i| **i == item) {
            *el = u32::MAX;
        } else {
            unreachable!();
        }
        Ok(())
    }

    /// Converts it into a readers to be able to read the nodes.
    pub fn into_bytes_reader(self) -> Result<TmpNodesReader> {
        let file = self.file.into_inner().map_err(|iie| iie.into_error())?;
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential)?;
        Ok(TmpNodesReader { mmap, ids: self.ids, bounds: self.bounds, deleted: self.deleted })
    }
}

/// A reader of nodes stored in a file.
pub struct TmpNodesReader {
    mmap: Mmap,
    ids: Vec<ItemId>,
    bounds: Vec<usize>,
    deleted: RoaringBitmap,
}

impl TmpNodesReader {
    /// The number of nodes stored in this file.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn to_delete(&self) -> impl Iterator<Item = ItemId> + '_ {
        self.deleted.iter()
    }

    /// Returns an forward iterator over the nodes.
    pub fn to_insert(&self) -> impl Iterator<Item = (ItemId, &[u8])> {
        self.ids
            .iter()
            .zip(self.bounds.windows(2))
            .map(|(id, bounds)| {
                let [start, end] = [bounds[0], bounds[1]];
                (*id, &self.mmap[start..end])
            })
            .filter(|(id, _)| *id != ItemId::MAX)
    }
}

/// A concurrent ID generate that will never return the same ID twice.
#[derive(Debug)]
pub struct ConcurrentNodeIds {
    /// The current tree node ID we should use.
    current: Arc<AtomicU32>,
    /// The total number of tree node IDs used.
    used: Arc<AtomicU64>,
    /// A list of IDs to exhaust first before picking IDs from `current`.
    /// When cloning this type, the available node ids are not cloned.
    available: RoaringBitmap,
}

impl ConcurrentNodeIds {
    /// Creates the ID generator starting at the given number.
    pub fn new(used: RoaringBitmap) -> ConcurrentNodeIds {
        let last_id = used.iter().last().map(|id| id + 1).unwrap_or(0);
        let used_ids = used.len();
        let available = RoaringBitmap::from_sorted_iter(0..last_id).unwrap() - used;

        ConcurrentNodeIds {
            current: Arc::new(AtomicU32::new(last_id)),
            used: Arc::new(AtomicU64::new(used_ids)),
            available,
        }
    }

    /// Returns and increment the ID you can use as a NodeId.
    pub fn next(&mut self) -> Result<u32> {
        if self.used.fetch_add(1, Ordering::Relaxed) > u32::MAX as u64 {
            Err(Error::DatabaseFull)
        } else {
            match self.available.select(0) {
                Some(id) => {
                    // remove_smallest has a faster access pattern than remove
                    self.available.remove_smallest(1);
                    Ok(id)
                }
                None => Ok(self.current.fetch_add(1, Ordering::Relaxed)),
            }
        }
    }

    /// Returns the number of used ids in total.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Split a concurrent node ids into `n` parts..
    pub fn split_in(self, n: usize) -> Vec<ConcurrentNodeIds> {
        if n <= 1 {
            return vec![self];
        }

        let Self { current, used, available } = self;
        let chunk_size = available.len() / n as u64;
        let mut ret = Vec::new();

        let mut iter = available.into_iter();
        for _ in 0..(n - 1) {
            let available = iter.by_ref().take(chunk_size as usize).collect();
            ret.push(Self { current: current.clone(), used: used.clone(), available });
        }
        // the last element is going to contain everything remaining
        ret.push(Self { current, used, available: iter.collect() });

        ret
    }

    /// Recompose a vector of concurrent node ids into a single concurrent node ids.
    /// Return `None` if called on an empty list.
    pub fn merge(concurrents_node_ids: Vec<ConcurrentNodeIds>) -> Option<Self> {
        concurrents_node_ids.into_iter().reduce(|left, right| Self {
            current: left.current,
            used: left.used,
            available: left.available | right.available,
        })
    }
}

/// Cloning this type doesn't clone the available node ids
impl Clone for ConcurrentNodeIds {
    fn clone(&self) -> Self {
        Self {
            current: self.current.clone(),
            used: self.used.clone(),
            available: RoaringBitmap::new(),
        }
    }
}

/// A struture used to keep a list of the leaf nodes in the tree.
///
/// It is safe to share between threads as the pointer are pointing
/// in the mmapped file and the transaction is kept here and therefore
/// no longer touches the database.
pub struct ImmutableLeafs<'t, D> {
    leaf_ids: RoaringBitmap,
    constant_length: Option<usize>,
    offsets: Vec<*const u8>,
    _marker: marker::PhantomData<(&'t (), D)>,
}

impl<'t, D: Distance> ImmutableLeafs<'t, D> {
    /// Creates the structure by fetching all the leaf pointers
    /// and keeping the transaction making the pointers valid.
    pub fn new(rtxn: &'t RoTxn, database: Database<D>, index: u16) -> heed::Result<Self> {
        let mut leaf_ids = RoaringBitmap::new();
        let mut constant_length = None;
        let mut offsets = Vec::new();

        let iter = database
            .remap_types::<PrefixCodec, Bytes>()
            .prefix_iter(rtxn, &Prefix::item(index))?
            .remap_key_type::<KeyCodec>();

        for result in iter {
            let (key, bytes) = result?;
            let item_id = key.node.unwrap_item();
            assert_eq!(*constant_length.get_or_insert(bytes.len()), bytes.len());
            assert!(leaf_ids.push(item_id));
            offsets.push(bytes.as_ptr());
        }

        Ok(ImmutableLeafs { leaf_ids, constant_length, offsets, _marker: marker::PhantomData })
    }

    /// Returns the leafs identified by the given ID.
    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<Leaf<'t, D>>> {
        let len = match self.constant_length {
            Some(len) => len,
            None => return Ok(None),
        };
        let ptr = match self
            .leaf_ids
            .rank(item_id)
            .checked_sub(1)
            .and_then(|offset| self.offsets.get(offset as usize))
        {
            Some(ptr) => *ptr,
            None => return Ok(None),
        };
        let bytes = unsafe { slice::from_raw_parts(ptr, len) };
        NodeCodec::bytes_decode(bytes).map_err(heed::Error::Decoding).map(|node| node.leaf())
    }
}

unsafe impl<D> Sync for ImmutableLeafs<'_, D> {}

/// A subset of leafs that are accessible for read.
pub struct ImmutableSubsetLeafs<'t, D> {
    subset: &'t RoaringBitmap,
    leafs: &'t ImmutableLeafs<'t, D>,
}

impl<'t, D: Distance> ImmutableSubsetLeafs<'t, D> {
    /// Creates a subset view of the available leafs.
    pub fn from_item_ids(leafs: &'t ImmutableLeafs<'t, D>, subset: &'t RoaringBitmap) -> Self {
        ImmutableSubsetLeafs { subset, leafs }
    }

    /// Returns the leafs identified by the given ID in the subset.
    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<Leaf<'t, D>>> {
        if self.subset.contains(item_id) {
            self.leafs.get(item_id)
        } else {
            Ok(None)
        }
    }

    /// Randomly selects two leafs verified to be different.
    pub fn choose_two<R: Rng>(&self, rng: &mut R) -> heed::Result<Option<[Leaf<'t, D>; 2]>> {
        let indexes = index::sample(rng, self.subset.len() as usize, 2);
        let first = match self.subset.select(indexes.index(0) as u32) {
            Some(item_id) => self.leafs.get(item_id)?,
            None => None,
        };
        let second = match self.subset.select(indexes.index(1) as u32) {
            Some(item_id) => self.leafs.get(item_id)?,
            None => None,
        };
        Ok(first.zip(second).map(|(a, b)| [a, b]))
    }

    /// Randomly select one leaf out of this subset.
    pub fn choose<R: Rng>(&self, rng: &mut R) -> heed::Result<Option<Leaf<'t, D>>> {
        if self.subset.is_empty() {
            Ok(None)
        } else {
            let ubound = (self.subset.len() - 1) as u32;
            let index = rng.gen_range(0..=ubound);
            match self.subset.select(index) {
                Some(item_id) => self.leafs.get(item_id),
                None => Ok(None),
            }
        }
    }
}

/// A struture used to keep a list of all the tree nodes in the tree.
///
/// It is safe to share between threads as the pointer are pointing
/// in the mmapped file and the transaction is kept here and therefore
/// no longer touches the database.
pub struct ImmutableTrees<'t, D> {
    tree_ids: RoaringBitmap,
    offsets: Vec<*const u8>,
    lengths: Vec<usize>,
    _marker: marker::PhantomData<(&'t (), D)>,
}

impl<'t, D: Distance> ImmutableTrees<'t, D> {
    /// Creates the structure by fetching all the root pointers
    /// and keeping the transaction making the pointers valid.
    pub fn new(rtxn: &'t RoTxn, database: Database<D>, index: u16) -> heed::Result<Self> {
        let mut tree_ids = RoaringBitmap::new();
        let mut offsets = Vec::new();
        let mut lengths = Vec::new();

        let iter = database
            .remap_types::<PrefixCodec, Bytes>()
            .prefix_iter(rtxn, &Prefix::tree(index))?
            .remap_key_type::<KeyCodec>();

        for result in iter {
            let (key, bytes) = result?;
            let tree_id = key.node.unwrap_tree();
            assert!(tree_ids.push(tree_id));
            offsets.push(bytes.as_ptr());
            lengths.push(bytes.len());
        }

        Ok(ImmutableTrees { tree_ids, lengths, offsets, _marker: marker::PhantomData })
    }

    /// Returns the tree node identified by the given ID.
    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<Node<'t, D>>> {
        let (ptr, len) = match self.tree_ids.rank(item_id).checked_sub(1).and_then(|offset| {
            self.offsets.get(offset as usize).zip(self.lengths.get(offset as usize))
        }) {
            Some((ptr, len)) => (*ptr, *len),
            None => return Ok(None),
        };

        let bytes = unsafe { slice::from_raw_parts(ptr, len) };
        NodeCodec::bytes_decode(bytes).map_err(heed::Error::Decoding).map(Some)
    }
}

unsafe impl<D> Sync for ImmutableTrees<'_, D> {}
