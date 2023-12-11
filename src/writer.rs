use std::any::TypeId;
use std::borrow::Cow;
use std::path::PathBuf;

use heed::types::{Bytes, DecodeIgnore};
use heed::{MdbError, PutFlags, RoTxn, RwTxn};
use rand::{Rng, SeedableRng};
use rayon::iter::repeatn;
use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::distance::Distance;
use crate::internals::{KeyCodec, Side};
use crate::item_iter::ItemIter;
use crate::node::{Descendants, ItemIds, Leaf, SplitPlaneNormal, UnalignedF32Slice};
use crate::node_id::NodeMode;
use crate::parallel::{
    ConcurrentNodeIds, ImmutableLeafs, ImmutableSubsetLeafs, ImmutableTrees, TmpNodes,
    TmpNodesReader,
};
use crate::reader::item_leaf;
use crate::roaring::RoaringBitmapCodec;
use crate::{
    Database, Error, ItemId, Key, Metadata, MetadataCodec, Node, NodeCodec, NodeId, Prefix,
    PrefixCodec, Result,
};

/// A writer to store new items, remove existing ones,
/// and build the search tree to query the nearest
/// neighbors to items or vectors.
#[derive(Debug)]
pub struct Writer<D: Distance> {
    database: Database<D>,
    index: u16,
    dimensions: usize,
    /// The folder in which tempfile will write its temporary files.
    tmpdir: Option<PathBuf>,
}

impl<D: Distance> Writer<D> {
    /// Creates a new writer from a database, index and dimensions.
    pub fn new(database: Database<D>, index: u16, dimensions: usize) -> Result<Writer<D>> {
        let database: Database<D> = database.remap_data_type();
        Ok(Writer { database, index, dimensions, tmpdir: None })
    }

    /// Returns a writer after having deleted the tree nodes and rewrote all the items
    /// for the new [`Distance`] format to be able to modify items safely.
    pub fn prepare_changing_distance<ND: Distance>(self, wtxn: &mut RwTxn) -> Result<Writer<ND>> {
        if TypeId::of::<ND>() != TypeId::of::<D>() {
            clear_tree_nodes(wtxn, self.database, self.index)?;

            let mut cursor = self.database.iter_mut(wtxn)?;
            while let Some((item_id, node)) = cursor.next().transpose()? {
                match node {
                    Node::Leaf(Leaf { header: _, vector }) => {
                        let new_leaf = Node::Leaf(Leaf {
                            header: ND::new_header(&vector),
                            vector: Cow::Owned(vector.into_owned()),
                        });
                        unsafe {
                            // safety: We do not keep a reference to the current value, we own it.
                            cursor.put_current_with_options::<NodeCodec<ND>>(
                                PutFlags::empty(),
                                &item_id,
                                &new_leaf,
                            )?
                        };
                    }
                    Node::Descendants(_) | Node::SplitPlaneNormal(_) => panic!(),
                }
            }
        }

        let Writer { database, index, dimensions, tmpdir } = self;
        Ok(Writer { database: database.remap_data_type(), index, dimensions, tmpdir })
    }

    /// Specifies the folder in which arroy will write temporary files when building the tree.
    ///
    /// If specified it uses the [`tempfile::tempfile_in`] function, otherwise it will
    /// use the default [`tempfile::tempfile`] function which uses the OS temporary directory.
    pub fn set_tmpdir(&mut self, path: impl Into<PathBuf>) {
        self.tmpdir = Some(path.into());
    }

    /// Returns an `Option`al vector previous stored in this database.
    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> Result<Option<Vec<f32>>> {
        Ok(item_leaf(self.database, self.index, rtxn, item)?.map(|leaf| leaf.vector.into_owned()))
    }

    /// Returns `true` if the index is empty.
    pub fn is_empty(&self, rtxn: &RoTxn) -> Result<bool> {
        self.iter(rtxn).map(|mut iter| iter.next().is_none())
    }

    /// Returns `true` if the database contains the given item.
    pub fn contains_item(&self, rtxn: &RoTxn, item: ItemId) -> Result<bool> {
        self.database
            .remap_data_type::<DecodeIgnore>()
            .get(rtxn, &Key::item(self.index, item))
            .map(|opt| opt.is_some())
            .map_err(Into::into)
    }

    /// Returns an iterator over the items vector.
    pub fn iter<'t>(&self, rtxn: &'t RoTxn) -> Result<ItemIter<'t, D>> {
        Ok(ItemIter {
            inner: self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter(rtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>(),
        })
    }

    /// Add an item associated to a vector in the database.
    pub fn add_item(&self, wtxn: &mut RwTxn, item: ItemId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(Error::InvalidVecDimension {
                expected: self.dimensions,
                received: vector.len(),
            });
        }

        let vector = UnalignedF32Slice::from_slice(vector);
        let leaf = Leaf { header: D::new_header(vector), vector: Cow::Borrowed(vector) };
        self.database.put(wtxn, &Key::item(self.index, item), &Node::Leaf(leaf))?;
        let mut updated = self
            .database
            .remap_data_type::<RoaringBitmapCodec>()
            .get(wtxn, &Key::updated(self.index))?
            .unwrap_or_default();
        updated.insert(item);
        self.database.remap_data_type::<RoaringBitmapCodec>().put(
            wtxn,
            &Key::updated(self.index),
            &updated,
        )?;

        Ok(())
    }

    /// Attempt to append an item into the database. It is generaly faster to append an item than insert it.
    ///
    /// There are two conditions for an item to be successfully appended:
    ///  - The last item ID in the database is smaller than the one appended.
    ///  - The index of the database is the highest one.
    pub fn append_item(&self, wtxn: &mut RwTxn, item: ItemId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(Error::InvalidVecDimension {
                expected: self.dimensions,
                received: vector.len(),
            });
        }

        let vector = UnalignedF32Slice::from_slice(vector);
        let leaf = Leaf { header: D::new_header(vector), vector: Cow::Borrowed(vector) };
        let key = Key::item(self.index, item);
        match self.database.put_with_flags(wtxn, PutFlags::APPEND, &key, &Node::Leaf(leaf)) {
            Ok(()) => (),
            Err(heed::Error::Mdb(MdbError::KeyExist)) => return Err(Error::InvalidItemAppend),
            Err(e) => return Err(e.into()),
        }
        let mut updated = self
            .database
            .remap_data_type::<RoaringBitmapCodec>()
            .get(wtxn, &Key::updated(self.index))?
            .unwrap_or_default();
        updated.insert(item);
        self.database.remap_data_type::<RoaringBitmapCodec>().put(
            wtxn,
            &Key::updated(self.index),
            &updated,
        )?;

        Ok(())
    }

    /// Deletes an item stored in this database and returns `true` if it existed.
    pub fn del_item(&self, wtxn: &mut RwTxn, item: ItemId) -> Result<bool> {
        if self.database.delete(wtxn, &Key::item(self.index, item))? {
            let mut updated = self
                .database
                .remap_data_type::<RoaringBitmapCodec>()
                .get(wtxn, &Key::updated(self.index))?
                .unwrap_or_default();
            updated.insert(item);
            self.database.remap_data_type::<RoaringBitmapCodec>().put(
                wtxn,
                &Key::updated(self.index),
                &updated,
            )?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Removes everything in the database, user items and internal tree nodes.
    pub fn clear(&self, wtxn: &mut RwTxn) -> Result<()> {
        let mut cursor = self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter_mut(wtxn, &Prefix::all(self.index))?
            .remap_key_type::<DecodeIgnore>();

        while let Some((_id, _node)) = cursor.next().transpose()? {
            // safety: we don't have any reference to the database
            unsafe { cursor.del_current() }?;
        }

        Ok(())
    }

    fn next_tree_node(&self, rtxn: &RoTxn) -> Result<ItemId> {
        Ok(self
            .database
            .remap_key_type::<PrefixCodec>()
            .rev_prefix_iter(rtxn, &Prefix::tree(self.index))?
            .remap_types::<KeyCodec, DecodeIgnore>()
            .next()
            .transpose()?
            .map(|(key, _)| key.node.item)
            .unwrap_or_default())
    }

    /// Generates a forest of `n_trees` trees.
    ///
    /// More trees give higher precision when querying at the cost of more disk usage.
    /// After calling build, no more items can be added.
    ///
    /// This function is using rayon to spawn threads. It can be configured
    /// by using the [`rayon::ThreadPoolBuilder`] and the
    /// [`rayon::ThreadPool::install`] to use it.
    pub fn build<R: Rng + SeedableRng>(
        self,
        wtxn: &mut RwTxn,
        rng: &mut R,
        n_trees: Option<usize>,
    ) -> Result<()> {
        log::debug!("started preprocessing the items...");

        D::preprocess(wtxn, |wtxn| {
            Ok(self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter_mut(wtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>())
        })?;

        let item_indices = self.item_indices(wtxn)?;
        let n_items = item_indices.len();
        let metadata = self
            .database
            .remap_data_type::<MetadataCodec>()
            .get(wtxn, &Key::metadata(self.index))?;
        let updated_items = self
            .database
            .remap_data_type::<RoaringBitmapCodec>()
            .get(wtxn, &Key::updated(self.index))?
            .unwrap_or_default();
        let deleted_items = &updated_items - &item_indices;
        let mut roots =
            metadata.as_ref().map_or(Vec::new(), |metadata| metadata.roots.iter().collect());

        log::debug!("Getting a reference to your {} items...", n_items);

        // TODO: do not starts at 0
        let concurrent_node_ids = ConcurrentNodeIds::new(self.next_tree_node(wtxn)?);
        let frozzen_reader = FrozzenReader {
            // TODO: We may not need the leafs or the trees. Can we do something about it?
            leafs: &ImmutableLeafs::new(wtxn, self.database, self.index)?,
            trees: &ImmutableTrees::new(wtxn, self.database, self.index)?,
            dimensions: self.dimensions,
            n_items,
            // The globally incrementing node ids that are shared between threads.
            concurrent_node_ids: &concurrent_node_ids,
        };

        // TODO: Insert the updated elements into the already existing trees
        if let Some(ref metadata) = metadata {
            // If there was metadata in the index we should delete them to let rest of the indexing process go as planned

            log::debug!(
                "started inserting new items {} in {} trees...",
                n_items,
                metadata.roots.len()
            );
            self.update_trees(rng, &metadata, &updated_items, &deleted_items, &frozzen_reader)?;

            // todo!("{:?}", updated_items);
        }

        // In any case we may be missing a bunch of root nodes after inserting the new elements

        // TODO: Create the new trees if necessary
        log::debug!("started building trees for {} items...", n_items);
        log::debug!(
            "running {} parallel tree building...",
            n_trees.map_or_else(|| "an unknown number of".to_string(), |n| n.to_string())
        );

        let n_trees_to_build = n_trees
            .zip(metadata)
            .map(|(n_trees, metadata)| n_trees.saturating_sub(metadata.roots.len()));
        let (mut thread_roots, tmp_nodes) =
            self.build_trees(rng, n_trees_to_build, &item_indices, &frozzen_reader)?;
        log::debug!("started writing the tree nodes of {} trees...", tmp_nodes.len());
        for (i, tmp_node) in tmp_nodes.into_iter().enumerate() {
            log::debug!("started writing the {} tree nodes of the {i}nth trees...", tmp_node.len());
            for (item_id, item_bytes) in tmp_node.iter() {
                let key = Key::tree(self.index, item_id);
                self.database.remap_data_type::<Bytes>().put(wtxn, &key, item_bytes)?;
            }
        }

        roots.append(&mut thread_roots);

        log::debug!("started writing the metadata...");

        let metadata = Metadata {
            dimensions: self.dimensions.try_into().unwrap(),
            items: item_indices,
            roots: ItemIds::from_slice(&roots),
            distance: D::name(),
        };
        match self.database.remap_data_type::<MetadataCodec>().put(
            wtxn,
            &Key::metadata(self.index),
            &metadata,
        ) {
            Ok(_) => (),
            Err(heed::Error::Mdb(MdbError::KeyExist)) => return Err(Error::DatabaseFull),
            Err(e) => return Err(e.into()),
        }

        Ok(())
    }

    fn update_trees<R: Rng + SeedableRng>(
        &self,
        rng: &mut R,
        metadata: &Metadata,
        updated_items: &RoaringBitmap,
        deleted_items: &RoaringBitmap,
        frozen_reader: &FrozzenReader<D>,
    ) -> Result<Vec<TmpNodesReader>> {
        let roots: Vec<_> = metadata.roots.iter().collect();

        repeatn(rng.next_u64(), metadata.roots.len())
            .zip(roots)
            .map(|(seed, root)| {
                log::debug!("started updating tree {root:X}...");
                let mut rng = R::seed_from_u64(seed.wrapping_add(root as u64));
                let mut tmp_nodes: TmpNodes<NodeCodec<D>> = match self.tmpdir.as_ref() {
                    Some(path) => TmpNodes::new_in(path)?,
                    None => TmpNodes::new()?,
                };
                let _id = self.update_nodes_in_file(
                    frozen_reader,
                    &mut rng,
                    NodeId::tree(root),
                    updated_items,
                    &mut tmp_nodes,
                )?;
                log::debug!("finished generating tree {root:X}");
                // make_tree will NEVER return a leaf when called as root
                tmp_nodes.into_bytes_reader()
            })
            .collect()
    }

    fn update_nodes_in_file<R: Rng>(
        &self,
        frozen_reader: &FrozzenReader<D>,
        rng: &mut R,
        current_node: NodeId,
        item_indices: &RoaringBitmap,
        tmp_nodes: &mut TmpNodes<NodeCodec<D>>,
    ) -> Result<Option<NodeId>> {
        match current_node.mode {
            NodeMode::Item => {
                // We were called on a specific item, we should create a descendants node
                let mut descendants = item_indices.clone();
                descendants.insert(current_node.item);
                tmp_nodes.put(
                    current_node.item,
                    &Node::Descendants(Descendants { descendants: Cow::Owned(descendants) }),
                )?;
                return Ok(Some(current_node));
            }
            NodeMode::Tree => {
                match frozen_reader.trees.get(current_node.item)?.expect("should not happen") {
                    Node::Leaf(_) => unreachable!("we searched in the trees, that's impossible"),
                    Node::Descendants(Descendants { descendants }) => {
                        // insert all of our IDs in the descendants
                        let new_descendants = descendants.into_owned() | item_indices;
                        tmp_nodes.put(
                            current_node.item,
                            &Node::Descendants(Descendants {
                                descendants: Cow::Owned(new_descendants),
                            }),
                        )?;
                        return Ok(Some(current_node));
                    }
                    Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                        // Iterato over all nodes in item_indices and regenerate two roaring bitmaps
                        let mut left_ids = RoaringBitmap::new();
                        let mut right_ids = RoaringBitmap::new();

                        for leaf in item_indices {
                            let query = frozen_reader.leafs.get(leaf)?.unwrap();
                            match D::side(&normal, &query, rng) {
                                Side::Left => left_ids.insert(leaf),
                                Side::Right => right_ids.insert(leaf),
                            };
                        }

                        self.update_nodes_in_file(frozen_reader, rng, left, &left_ids, tmp_nodes)?;
                        self.update_nodes_in_file(
                            frozen_reader,
                            rng,
                            right,
                            &right_ids,
                            tmp_nodes,
                        )?;

                        Ok(None)
                    }
                }
            }
            NodeMode::Metadata => panic!("Should never happens"),
        }
    }

    fn build_trees<R: Rng + SeedableRng>(
        &self,
        rng: &mut R,
        n_trees: Option<usize>,
        item_indices: &RoaringBitmap,
        frozen_reader: &FrozzenReader<D>,
    ) -> Result<(Vec<ItemId>, Vec<TmpNodesReader>)> {
        let n_items = item_indices.len();
        let concurrent_node_ids = frozen_reader.concurrent_node_ids;

        // TODO: Do not enter if we have enough tree
        repeatn(rng.next_u64(), n_trees.unwrap_or(usize::MAX))
            .enumerate()
            // Stop generating trees once the number of tree nodes are generated
            // but continue to generate trees if the number of trees is specified
            .take_any_while(|_| match n_trees {
                Some(_) => true,
                None => concurrent_node_ids.current() < n_items,
            })
            .map(|(i, seed)| {
                log::debug!("started generating tree {i:X}...");
                let mut rng = R::seed_from_u64(seed.wrapping_add(i as u64));
                let mut tmp_nodes = match self.tmpdir.as_ref() {
                    Some(path) => TmpNodes::new_in(path)?,
                    None => TmpNodes::new()?,
                };
                let root_id =
                    make_tree_in_file(frozen_reader, &mut rng, item_indices, true, &mut tmp_nodes)?;
                log::debug!("finished generating tree {i:X}");
                // make_tree will NEVER return a leaf when called as root
                Ok((root_id.unwrap_tree(), tmp_nodes.into_bytes_reader()?))
            })
            .collect()
    }

    // Fetches the item's ids, not the tree nodes ones.
    fn item_indices(&self, wtxn: &mut RwTxn<'_>) -> heed::Result<RoaringBitmap> {
        let mut indices = RoaringBitmap::new();
        for result in self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(wtxn, &Prefix::item(self.index))?
            .remap_key_type::<KeyCodec>()
        {
            let (i, _) = result?;
            indices.push(i.node.unwrap_item());
        }

        Ok(indices)
    }
}

/// Represents the final version of the leafs and contains
/// useful informations to synchronize the building threads.
#[derive(Clone)]
struct FrozzenReader<'a, D: Distance> {
    leafs: &'a ImmutableLeafs<'a, D>,
    trees: &'a ImmutableTrees<'a, D>,
    dimensions: usize,
    n_items: u64,
    concurrent_node_ids: &'a ConcurrentNodeIds,
}

/// Creates a tree of nodes from the frozzen items that lives
/// in the database and generates descendants, split normal
/// and root nodes in files that will be stored in the database later.
fn make_tree_in_file<D: Distance, R: Rng>(
    reader: &FrozzenReader<D>,
    rng: &mut R,
    item_indices: &RoaringBitmap,
    is_root: bool,
    tmp_nodes: &mut TmpNodes<NodeCodec<D>>,
) -> Result<NodeId> {
    // we simplify the max descendants (_K) thing by considering
    // that we can fit as much descendants as the number of dimensions
    let max_descendants = reader.dimensions as u64;

    if item_indices.len() == 1 && !is_root {
        return Ok(NodeId::item(item_indices.min().unwrap()));
    }

    if item_indices.len() <= max_descendants
        && (!is_root || reader.n_items <= max_descendants || item_indices.len() == 1)
    {
        let item_id = reader.concurrent_node_ids.next().try_into().unwrap();
        let item = Node::Descendants(Descendants { descendants: Cow::Borrowed(item_indices) });
        tmp_nodes.put(item_id, &item)?;
        return Ok(NodeId::tree(item_id));
    }

    let children = ImmutableSubsetLeafs::from_item_ids(reader.leafs, item_indices);
    let mut children_left = RoaringBitmap::new();
    let mut children_right = RoaringBitmap::new();
    let mut remaining_attempts = 3;

    let mut normal = loop {
        children_left.clear();
        children_right.clear();

        let normal = D::create_split(&children, rng)?;
        for item_id in item_indices.iter() {
            let node = children.get(item_id)?.unwrap();
            match D::side(UnalignedF32Slice::from_slice(&normal), &node, rng) {
                Side::Left => children_left.push(item_id),
                Side::Right => children_right.push(item_id),
            };
        }

        if split_imbalance(children_left.len(), children_right.len()) < 0.95
            || remaining_attempts == 0
        {
            break normal;
        }

        remaining_attempts -= 1;
    };

    // If we didn't find a hyperplane, just randomize sides as a last option
    // and set the split plane to zero as a dummy plane.
    if split_imbalance(children_left.len(), children_right.len()) > 0.99 {
        randomly_split_children(rng, item_indices, &mut children_left, &mut children_right);
        normal.fill(0.0);
    }

    let normal = SplitPlaneNormal {
        normal: Cow::Owned(normal),
        left: make_tree_in_file(reader, rng, &children_left, false, tmp_nodes)?,
        right: make_tree_in_file(reader, rng, &children_right, false, tmp_nodes)?,
    };

    let new_node_id = reader.concurrent_node_ids.next().try_into().unwrap();
    tmp_nodes.put(new_node_id, &Node::SplitPlaneNormal(normal))?;

    Ok(NodeId::tree(new_node_id))
}

/// Randomly and efficiently splits the items into the left and right children vectors.
fn randomly_split_children<R: Rng>(
    rng: &mut R,
    item_indices: &RoaringBitmap,
    children_left: &mut RoaringBitmap,
    children_right: &mut RoaringBitmap,
) {
    children_left.clear();
    children_right.clear();

    // Split it in half and put the right half into the right children's vector
    for item_id in item_indices {
        match Side::random(rng) {
            Side::Left => children_left.push(item_id),
            Side::Right => children_right.push(item_id),
        };
    }
}

/// Clears everything but the leafs nodes (items).
/// Starts from the last node and stops at the first leaf.
fn clear_tree_nodes<D: Distance>(
    wtxn: &mut RwTxn,
    database: Database<D>,
    index: u16,
) -> Result<()> {
    database.delete(wtxn, &Key::metadata(index))?;
    let mut cursor = database
        .remap_types::<PrefixCodec, DecodeIgnore>()
        .prefix_iter_mut(wtxn, &Prefix::tree(index))?
        .remap_key_type::<DecodeIgnore>();
    while let Some((_id, _node)) = cursor.next().transpose()? {
        // safety: we keep no reference into the database between operations
        unsafe { cursor.del_current()? };
    }

    Ok(())
}

fn split_imbalance(left_indices_len: u64, right_indices_len: u64) -> f64 {
    let ls = left_indices_len as f64;
    let rs = right_indices_len as f64;
    let f = ls / (ls + rs + f64::EPSILON); // Avoid 0/0
    f.max(1.0 - f)
}
