use core::slice;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::marker;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use heed::types::Bytes;
use heed::{BytesDecode, BytesEncode, RoTxn};
use memmap2::Mmap;
use nohash::{BuildNoHashHasher, IntMap};
use rand::seq::index;
use rand::Rng;
use roaring::RoaringBitmap;

use crate::internals::{KeyCodec, Leaf, NodeCodec};
use crate::key::{Key, Prefix, PrefixCodec};
use crate::node::Node;
use crate::writer::BuildOption;
use crate::{Database, Distance, Error, ItemId, MainStep, Result, WriterProgress};

#[derive(Default, Debug)]
enum TmpNodesState {
    Writing(BufWriter<File>),
    Reading(BufReader<File>),
    // Ugly trick because otherwise I can't take the value out of the enum. Can we do better?
    // The enum should never be let in this state.
    #[default]
    Invalid,
}

impl TmpNodesState {
    pub fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        let this = match std::mem::take(self) {
            TmpNodesState::Writing(mut writer) => {
                writer.write_all(bytes)?;
                TmpNodesState::Writing(writer)
            }
            TmpNodesState::Reading(reader) => {
                let mut writer = BufWriter::new(reader.into_inner());
                writer.write_all(bytes)?;
                TmpNodesState::Writing(writer)
            }
            TmpNodesState::Invalid => unreachable!(),
        };

        debug_assert!(!matches!(this, TmpNodesState::Invalid));

        *self = this;
        Ok(())
    }

    pub fn read_all(&mut self, (start, end): (usize, usize)) -> Result<Vec<u8>> {
        debug_assert!(start < end);
        let mut buffer = vec![0; end - start];

        let this = match std::mem::take(self) {
            TmpNodesState::Writing(mut writer) => {
                writer.flush()?;
                let mut reader = BufReader::new(writer.into_inner().expect("Could not convert the writer to a file even thought it was flushed right before"));
                reader.seek(SeekFrom::Start(start as u64))?;
                reader.read_exact(&mut buffer)?;
                TmpNodesState::Reading(reader)
            }
            TmpNodesState::Reading(mut reader) => {
                reader.seek(SeekFrom::Start(start as u64))?;
                reader.read_exact(&mut buffer)?;
                TmpNodesState::Reading(reader)
            }
            TmpNodesState::Invalid => unreachable!(),
        };

        debug_assert!(!matches!(this, TmpNodesState::Invalid));

        *self = this;
        Ok(buffer)
    }

    pub fn into_inner(self) -> Result<File> {
        let file = match self {
            TmpNodesState::Writing(mut writer) => {
                writer.flush()?;
                writer.into_inner().expect("Could not convert the writer to a file even thought it was flushed right before")
            }
            TmpNodesState::Reading(reader) => reader.into_inner(),
            TmpNodesState::Invalid => unreachable!(),
        };
        Ok(file)
    }
}

/// A structure to store the tree nodes out of the heed database.
/// You should avoid alternating between writing and reading as it flushes the file on each operation and lose the read buffer.
/// The structure is optimized for reading the last written nodes.
pub struct TmpNodes<DE> {
    file: TmpNodesState,
    ids: Vec<ItemId>,
    bounds: Vec<usize>,
    deleted: RoaringBitmap,
    _marker: marker::PhantomData<DE>,
}

impl<'a, D: Distance> TmpNodes<D> {
    /// Creates an empty `TmpNodes`.
    pub fn new() -> heed::Result<TmpNodes<D>> {
        Ok(TmpNodes {
            file: TmpNodesState::Writing(tempfile::tempfile().map(BufWriter::new)?),
            ids: Vec::new(),
            bounds: vec![0],
            deleted: RoaringBitmap::new(),
            _marker: marker::PhantomData,
        })
    }

    /// Creates an empty `TmpNodes` in the defined folder.
    pub fn new_in(path: &Path) -> heed::Result<TmpNodes<D>> {
        Ok(TmpNodes {
            file: TmpNodesState::Writing(tempfile::tempfile_in(path).map(BufWriter::new)?),
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
        data: &'a Node<D>,
    ) -> Result<()> {
        assert!(item != ItemId::MAX);
        let bytes = NodeCodec::bytes_encode(data).map_err(heed::Error::Encoding)?;
        self.file.write_all(&bytes)?;
        let last_bound = self.bounds.last().unwrap();
        self.bounds.push(last_bound + bytes.len());
        self.ids.push(item);

        // in the current algorithm, we should never insert a node that was deleted before
        debug_assert!(!self.deleted.contains(item));

        Ok(())
    }

    /// Get the node at the given item id.
    /// Ignore the remapped ids and deletions, only suitable when appending to the file.
    /// A flush will be executed on the file if the previous operation was a write.
    pub fn get(&mut self, item: ItemId) -> Result<Option<Node<'static, D>>> {
        // In our current implementation, when we starts retrieving the nodes, it's always the nodes of the last tree,
        // so it makes sense to search in reverse order.
        let Some(position) = self.ids.iter().rev().position(|id| *id == item) else {
            return Ok(None);
        };
        let bounds = &self.bounds[self.bounds.len() - position - 2..self.bounds.len() - position];
        let bytes = self.file.read_all((bounds[0], bounds[1]))?;
        Ok(Some(NodeCodec::bytes_decode(&bytes).map_err(heed::Error::Decoding)?.into_owned()))
    }

    /// Delete the tmp_nodes and the node in the database.
    pub fn remove(&mut self, item: ItemId) {
        let deleted = self.deleted.insert(item);
        debug_assert!(deleted, "Removed the same item with id {item} twice");
    }

    /// Converts it into a readers to read the nodes.
    pub fn into_bytes_reader(self) -> Result<TmpNodesReader> {
        let file = self.file.into_inner()?;
        // safety: No one should move our files around
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
    pub fn to_delete(&self) -> impl Iterator<Item = ItemId> + '_ {
        self.deleted.iter()
    }

    /// Returns an forward iterator over the nodes.
    pub fn to_insert(&self) -> impl Iterator<Item = (ItemId, &[u8])> {
        self.ids
            .iter()
            .zip(self.bounds.windows(2))
            .filter(|(&id, _)| !self.deleted.contains(id))
            .map(|(id, bounds)| {
                let [start, end] = [bounds[0], bounds[1]];
                (*id, &self.mmap[start..end])
            })
    }
}

/// A concurrent ID generate that will never return the same ID twice.
#[derive(Debug)]
pub struct ConcurrentNodeIds {
    /// The current tree node ID we should use if there is no other IDs available.
    current: AtomicU32,
    /// The total number of tree node IDs used.
    used: AtomicU64,

    /// A list of IDs to exhaust before picking IDs from `current`.
    available: RoaringBitmap,
    /// The current Nth ID to select in the bitmap.
    select_in_bitmap: AtomicU32,
    /// Tells if you should look in the roaring bitmap or if all the IDs are already exhausted.
    look_into_bitmap: AtomicBool,
}

impl ConcurrentNodeIds {
    /// Creates an ID generator returning unique IDs, avoiding the specified used IDs.
    pub fn new(used: RoaringBitmap) -> ConcurrentNodeIds {
        let last_id = used.max().map_or(0, |id| id + 1);
        let used_ids = used.len();
        let available = RoaringBitmap::from_sorted_iter(0..last_id).unwrap() - used;

        ConcurrentNodeIds {
            current: AtomicU32::new(last_id),
            used: AtomicU64::new(used_ids),
            select_in_bitmap: AtomicU32::new(0),
            look_into_bitmap: AtomicBool::new(!available.is_empty()),
            available,
        }
    }

    /// Returns a new unique ID and increase the count of IDs used.
    pub fn next(&self) -> Result<u32> {
        if self.used.fetch_add(1, Ordering::Relaxed) > u32::MAX as u64 {
            Err(Error::DatabaseFull)
        } else if self.look_into_bitmap.load(Ordering::Relaxed) {
            let current = self.select_in_bitmap.fetch_add(1, Ordering::Relaxed);
            match self.available.select(current) {
                Some(id) => Ok(id),
                None => {
                    self.look_into_bitmap.store(false, Ordering::Relaxed);
                    Ok(self.current.fetch_add(1, Ordering::Relaxed))
                }
            }
        } else {
            Ok(self.current.fetch_add(1, Ordering::Relaxed))
        }
    }
}

/// A struture used to keep a list of the leaf nodes in the tree.
///
/// It is safe to share between threads as the pointer are pointing
/// in the mmapped file and the transaction is kept here and therefore
/// no longer touches the database.
pub struct ImmutableLeafs<'t, D> {
    leafs: IntMap<ItemId, *const u8>,
    constant_length: Option<usize>,
    _marker: marker::PhantomData<(&'t (), D)>,
}

impl<'t, D: Distance> ImmutableLeafs<'t, D> {
    /// Creates the structure by fetching all the leaf pointers
    /// and keeping the transaction making the pointers valid.
    pub fn new(
        rtxn: &'t RoTxn,
        options: &BuildOption,
        database: Database<D>,
        items: &RoaringBitmap,
        index: u16,
    ) -> heed::Result<Self> {
        (options.progress)(WriterProgress { main: MainStep::RetrievingTheItems, sub: None });
        let mut leafs =
            IntMap::with_capacity_and_hasher(items.len() as usize, BuildNoHashHasher::default());
        let mut constant_length = None;

        for item_id in items {
            let bytes =
                database.remap_data_type::<Bytes>().get(rtxn, &Key::item(index, item_id))?.unwrap();
            assert_eq!(*constant_length.get_or_insert(bytes.len()), bytes.len());

            let ptr = bytes.as_ptr();
            leafs.insert(item_id, ptr);
        }

        Ok(ImmutableLeafs { leafs, constant_length, _marker: marker::PhantomData })
    }

    /// Returns the leafs identified by the given ID.
    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<Leaf<'t, D>>> {
        let len = match self.constant_length {
            Some(len) => len,
            None => return Ok(None),
        };
        let ptr = match self.leafs.get(&item_id) {
            Some(ptr) => *ptr,
            None => return Ok(None),
        };

        // safety:
        // - ptr: The pointer comes from LMDB. Since the database cannot be written to, it is still valid.
        // - len: All the items share the same dimensions and are the same size
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

    pub fn len(&self) -> u64 {
        self.subset.len()
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
    trees: IntMap<ItemId, (usize, *const u8)>,
    _marker: marker::PhantomData<(&'t (), D)>,
}

impl<'t, D: Distance> ImmutableTrees<'t, D> {
    /// Creates the structure by fetching all the root pointers
    /// and keeping the transaction making the pointers valid.
    pub fn new(
        rtxn: &'t RoTxn,
        options: &BuildOption,
        database: Database<D>,
        index: u16,
        nb_trees: u64,
    ) -> heed::Result<Self> {
        (options.progress)(WriterProgress { main: MainStep::RetrievingTheTreeNodes, sub: None });
        let mut trees =
            IntMap::with_capacity_and_hasher(nb_trees as usize, BuildNoHashHasher::default());

        let iter = database
            .remap_types::<PrefixCodec, Bytes>()
            .prefix_iter(rtxn, &Prefix::tree(index))?
            .remap_key_type::<KeyCodec>();

        for result in iter {
            let (key, bytes) = result?;
            let tree_id = key.node.unwrap_tree();
            trees.insert(tree_id, (bytes.len(), bytes.as_ptr()));
        }

        Ok(ImmutableTrees { trees, _marker: marker::PhantomData })
    }

    /// Returns the tree node identified by the given ID.
    pub fn get(&self, item_id: ItemId) -> heed::Result<Option<Node<'t, D>>> {
        let (ptr, len) = match self.trees.get(&item_id) {
            Some((len, ptr)) => (*ptr, *len),
            None => return Ok(None),
        };

        // safety:
        // - ptr: The pointer comes from LMDB. Since the database cannot be written to, it is still valid.
        // - len: The len cannot change either
        let bytes = unsafe { slice::from_raw_parts(ptr, len) };
        NodeCodec::bytes_decode(bytes).map_err(heed::Error::Decoding).map(Some)
    }
}

unsafe impl<D> Sync for ImmutableTrees<'_, D> {}
