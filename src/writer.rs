use std::any::TypeId;
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use heed::types::{Bytes, DecodeIgnore, Unit};
use heed::{MdbError, PutFlags, RoTxn, RwTxn};
use rand::{Rng, SeedableRng};
use rayon::iter::repeatn;
use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::distance::Distance;
use crate::internals::{KeyCodec, Side};
use crate::item_iter::ItemIter;
use crate::node::{Descendants, ItemIds, Leaf, SplitPlaneNormal};
use crate::node_id::NodeMode;
use crate::parallel::{
    ConcurrentNodeIds, ImmutableLeafs, ImmutableSubsetLeafs, ImmutableTrees, TmpNodes,
    TmpNodesReader,
};
use crate::reader::item_leaf;
use crate::unaligned_vector::UnalignedVector;
use crate::version::{Version, VersionCodec};
use crate::{
    Database, Error, ItemId, Key, Metadata, MetadataCodec, Node, NodeCodec, NodeId, Prefix,
    PrefixCodec, Result,
};

/// The options available when building the arroy database.
pub struct ArroyBuilder<'a, D: Distance, R: Rng + SeedableRng> {
    writer: &'a Writer<D>,
    rng: &'a mut R,
    inner: BuildOption<'a>,
}

/// Helps you understand what is happening inside of arroy during an indexing process.
#[derive(Debug)]
pub struct WriterProgress {
    /// The `main` part describes what's going on overall.
    pub main: MainStep,
    /// Sometimes, when a part takes a lot of time, you'll get a substep describing with more details what's going on.
    pub sub: Option<SubStep>,
}

/// When a `MainStep` takes too long, it may output a sub-step that gives you more details about the progression we've made on the current step.
#[derive(Debug)]
pub struct SubStep {
    /// The name of what is being updated.
    pub unit: &'static str,
    /// The `current` iteration we're at. It's stored in an `AtomicU32` so arroy can update it very quickly without calling your closure again.
    pub current: Arc<AtomicU32>,
    /// The `max`imum number of iteration it'll do before updating the `MainStep` again.
    pub max: u32,
}

impl SubStep {
    fn new(unit: &'static str, max: u32) -> (Self, Arc<AtomicU32>) {
        let current = Arc::new(AtomicU32::new(0));
        (Self { unit, current: current.clone(), max }, current)
    }
}

/// Some steps arroy will go through during an indexing process.
/// Some steps may be skipped in certain cases, and the name of the variant
/// the order in which they appear and the time they take is unspecified, and
/// might change from one version to the next one.
/// It's recommended not to assume anything from this enum.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, enum_iterator::Sequence)]
#[allow(missing_docs)]
pub enum MainStep {
    PreProcessingTheItems,
    RetrievingTheItemsIds,
    RetrieveTheUpdatedItems,
    WritingTheDescendantsAndMetadata,
    RemoveItems,
    InsertItemsInCurrentTrees,
    IncrementalIndexLargeDescendants,
    WriteTheMetadata,
}

/// The options available when building the arroy database.
struct BuildOption<'a> {
    n_trees: Option<usize>,
    split_after: Option<usize>,
    available_memory: Option<usize>,
    cancel: Box<dyn Fn() -> bool + 'a + Sync + Send>,
    progress: Box<dyn Fn(WriterProgress) + 'a + Sync + Send>,
}

impl Default for BuildOption<'_> {
    fn default() -> Self {
        Self {
            n_trees: None,
            split_after: None,
            available_memory: None,
            cancel: Box::new(|| false),
            progress: Box::new(|_| ()),
        }
    }
}

impl<'a, D: Distance, R: Rng + SeedableRng> ArroyBuilder<'a, D, R> {
    /// The number of trees to build. If not set arroy will determine the best amount to build for your number of vectors by itself.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use arroy::{Writer, distances::Euclidean};
    /// # let (writer, wtxn): (Writer<Euclidean>, heed::RwTxn) = todo!();
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// let mut rng = StdRng::seed_from_u64(13);
    /// writer.builder(&mut rng).n_trees(10).build(&mut wtxn);
    /// ```
    pub fn n_trees(&mut self, n_trees: usize) -> &mut Self {
        self.inner.n_trees = Some(n_trees);
        self
    }

    /// Configure the maximum number of items stored in a descendant node.
    ///
    /// This is only applied to the newly created or updated tree node.
    /// If the value is modified while working on an already existing database,
    /// the nodes that don't need to be updated won't be recreated.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use arroy::{Writer, distances::Euclidean};
    /// # let (writer, wtxn): (Writer<Euclidean>, heed::RwTxn) = todo!();
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// let mut rng = StdRng::seed_from_u64(92);
    /// writer.builder(&mut rng).split_after(1000).build(&mut wtxn);
    /// ```
    pub fn split_after(&mut self, split_after: usize) -> &mut Self {
        self.inner.split_after = Some(split_after);
        self
    }

    /// Configure the maximum memory arroy can use to build its trees in bytes.
    ///
    /// This value is used as a hint; arroy may still consume too much memory, especially if the value is too low.
    /// If not specified, arroy will use as much memory as possible but keep in mind that if arroy tries to use more
    /// memory than you have, it'll become very slow.
    ///
    /// In this case, it will randomly read the disk as pages will be invalidated by other reads, and OS cache will
    /// become useless.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use arroy::{Writer, distances::Euclidean};
    /// # let (writer, wtxn): (Writer<Euclidean>, heed::RwTxn) = todo!();
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// let mut rng = StdRng::seed_from_u64(92);
    /// let memory = 1024 * 1024 * 1024 * 4; // 4 GiB
    /// writer.builder(&mut rng).available_memory(memory).build(&mut wtxn);
    /// ```
    pub fn available_memory(&mut self, memory: usize) -> &mut Self {
        self.inner.available_memory = Some(memory);
        self
    }

    /// Provide a closure that can cancel the indexing process early if needed.
    /// There is no guarantee on when the process is going to cancel itself, but
    /// arroy will try to stop as soon as possible once the closure returns `true`.
    ///
    /// Since the closure is not mutable and will be called from multiple threads
    /// at the same time it's encouraged to make it quick to execute. A common
    /// way to use it is to fetch an `AtomicBool` inside it that can be set
    /// from another thread without lock.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use arroy::{Writer, distances::Euclidean};
    /// # let (writer, wtxn): (Writer<Euclidean>, heed::RwTxn) = todo!();
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let stops_after = AtomicBool::new(false);
    ///
    /// // Cancel the task after one minute
    /// std::thread::spawn(|| {
    ///     let one_minute = std::time::Duration::from_secs(60);
    ///     std::thread::sleep(one_minute);
    ///     stops_after.store(true, Ordering::Relaxed);
    /// });
    ///
    /// let mut rng = StdRng::seed_from_u64(92);
    /// writer.builder(&mut rng).cancel(|| stops_after.load(Ordering::Relaxed)).build(&mut wtxn);
    /// ```
    pub fn cancel(&mut self, cancel: impl Fn() -> bool + 'a + Sync + Send) -> &mut Self {
        self.inner.cancel = Box::new(cancel);
        self
    }

    /// The provided closure is called between all the indexing steps.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use arroy::{Writer, distances::Euclidean};
    /// # let (writer, wtxn): (Writer<Euclidean>, heed::RwTxn) = todo!();
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use std::sync::atomic::{AtomicBool, Ordering};
    ///
    /// let mut rng = StdRng::seed_from_u64(4729);
    /// writer.builder(&mut rng).progress(|progress| println!("{progress:?}")).build(&mut wtxn);
    /// ```
    pub fn progress(&mut self, progress: impl Fn(WriterProgress) + 'a + Sync + Send) -> &mut Self {
        self.inner.progress = Box::new(progress);
        self
    }

    /// Generates a forest of `n_trees` trees.
    ///
    /// More trees give higher precision when querying at the cost of more disk usage.
    ///
    /// This function is using rayon to spawn threads. It can be configured
    /// by using the [`rayon::ThreadPoolBuilder`] and the
    /// [`rayon::ThreadPool::install`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use arroy::{Writer, distances::Euclidean};
    /// # let (writer, wtxn): (Writer<Euclidean>, heed::RwTxn) = todo!();
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// let mut rng = StdRng::seed_from_u64(92);
    /// writer.builder(&mut rng).build(&mut wtxn);
    /// ```
    pub fn build(&mut self, wtxn: &mut RwTxn) -> Result<()> {
        self.writer.build(wtxn, self.rng, &self.inner)
    }
}

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
    pub fn new(database: Database<D>, index: u16, dimensions: usize) -> Writer<D> {
        let database: Database<D> = database.remap_data_type();
        Writer { database, index, dimensions, tmpdir: None }
    }

    /// Returns a writer after having deleted the tree nodes and rewrote all the items
    /// for the new [`Distance`] format to be able to modify items safely.
    pub fn prepare_changing_distance<ND: Distance>(self, wtxn: &mut RwTxn) -> Result<Writer<ND>> {
        if TypeId::of::<ND>() != TypeId::of::<D>() {
            clear_tree_nodes(wtxn, self.database, self.index)?;

            let mut cursor = self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter_mut(wtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>();
            while let Some((item_id, node)) = cursor.next().transpose()? {
                match node {
                    Node::Leaf(Leaf { header: _, vector }) => {
                        let vector = vector.to_vec();
                        let vector = UnalignedVector::from_vec(vector);
                        let new_leaf = Node::Leaf(Leaf { header: ND::new_header(&vector), vector });
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
        Ok(item_leaf(self.database, self.index, rtxn, item)?.map(|leaf| {
            let mut vec = leaf.vector.to_vec();
            vec.truncate(self.dimensions);
            vec
        }))
    }

    /// Returns `true` if the index is empty.
    pub fn is_empty(&self, rtxn: &RoTxn) -> Result<bool> {
        self.iter(rtxn).map(|mut iter| iter.next().is_none())
    }

    /// Returns `true` if the index needs to be built before being able to read in it.
    pub fn need_build(&self, rtxn: &RoTxn) -> Result<bool> {
        Ok(self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(rtxn, &Prefix::updated(self.index))?
            .remap_key_type::<KeyCodec>()
            .next()
            .is_some()
            || self
                .database
                .remap_data_type::<DecodeIgnore>()
                .get(rtxn, &Key::metadata(self.index))?
                .is_none())
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

        let vector = UnalignedVector::from_slice(vector);
        let leaf = Leaf { header: D::new_header(&vector), vector };
        self.database.put(wtxn, &Key::item(self.index, item), &Node::Leaf(leaf))?;
        self.database.remap_data_type::<Unit>().put(wtxn, &Key::updated(self.index, item), &())?;

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

        let vector = UnalignedVector::from_slice(vector);
        let leaf = Leaf { header: D::new_header(&vector), vector };
        let key = Key::item(self.index, item);
        match self.database.put_with_flags(wtxn, PutFlags::APPEND, &key, &Node::Leaf(leaf)) {
            Ok(()) => (),
            Err(heed::Error::Mdb(MdbError::KeyExist)) => return Err(Error::InvalidItemAppend),
            Err(e) => return Err(e.into()),
        }
        // We cannot append here because the items appear after the updated keys
        self.database.remap_data_type::<Unit>().put(wtxn, &Key::updated(self.index, item), &())?;

        Ok(())
    }

    /// Deletes an item stored in this database and returns `true` if it existed.
    pub fn del_item(&self, wtxn: &mut RwTxn, item: ItemId) -> Result<bool> {
        if self.database.delete(wtxn, &Key::item(self.index, item))? {
            self.database.remap_data_type::<Unit>().put(
                wtxn,
                &Key::updated(self.index, item),
                &(),
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
            .remap_types::<DecodeIgnore, DecodeIgnore>();

        while let Some((_id, _node)) = cursor.next().transpose()? {
            // safety: we don't have any reference to the database
            unsafe { cursor.del_current() }?;
        }

        Ok(())
    }

    fn used_tree_node(&self, rtxn: &RoTxn) -> Result<RoaringBitmap> {
        Ok(self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter(rtxn, &Prefix::tree(self.index))?
            .remap_types::<KeyCodec, DecodeIgnore>()
            .try_fold(RoaringBitmap::new(), |mut bitmap, used| -> Result<RoaringBitmap> {
                bitmap.insert(used?.0.node.item);
                Ok(bitmap)
            })
            .unwrap_or_default())
    }

    // we simplify the max descendants (_K) thing by considering
    // that we can fit as much descendants as the number of dimensions
    fn fit_in_descendant(&self, opt: &BuildOption, n: u64) -> bool {
        let max_in_descendant = opt.split_after.unwrap_or(self.dimensions) as u64;
        n <= max_in_descendant
    }

    /// Returns an [`ArroyBuilder`] to configure the available options to build the database.
    pub fn builder<'a, R: Rng + SeedableRng>(&'a self, rng: &'a mut R) -> ArroyBuilder<'a, D, R> {
        ArroyBuilder { writer: self, rng, inner: BuildOption::default() }
    }

    fn build<R: Rng + SeedableRng>(
        &self,
        wtxn: &mut RwTxn,
        rng: &mut R,
        options: &BuildOption,
    ) -> Result<()> {
        self.pre_process_items(wtxn, options)?;
        let item_indices = self.item_indices(wtxn, options)?;
        let n_items = item_indices.len();
        // updated items can be an update, an addition or a removed item
        let updated_items = self.reset_and_retrieve_updated_items(wtxn, options)?;

        if self.fit_in_descendant(options, item_indices.len()) {
            return self.clear_db_and_create_a_single_leaf(wtxn, options, item_indices);
        }

        let to_delete = updated_items.clone();
        let to_insert = &item_indices & &updated_items;

        let metadata = self
            .database
            .remap_data_type::<MetadataCodec>()
            .get(wtxn, &Key::metadata(self.index))?;
        let mut roots =
            metadata.as_ref().map_or_else(Vec::new, |metadata| metadata.roots.iter().collect());
        // we should not keep a reference to the metadata since they're going to be moved by LMDB
        drop(metadata);

        tracing::debug!("Getting a reference to your {} items...", n_items);

        if (options.cancel)() {
            return Err(Error::BuildCancelled);
        }

        let used_node_ids = self.used_tree_node(wtxn)?;
        let nb_tree_nodes = used_node_ids.len();
        let concurrent_node_ids = ConcurrentNodeIds::new(used_node_ids);

        let target_n_trees = self.target_n_trees(
            options,
            &item_indices,
            updated_items,
            &to_delete,
            &roots,
            nb_tree_nodes,
        );
        self.delete_extra_trees(wtxn, &mut roots, target_n_trees)?;

        // Before taking any references on the DB, remove all the items we must remove.
        self.delete_items_from_trees(wtxn, options, &mut roots, &to_delete)?;

        let mut large_descendants = self.insert_items_in_current_trees(
            wtxn,
            rng,
            options,
            to_insert,
            &roots,
            nb_tree_nodes,
            &concurrent_node_ids,
        )?;
        // Create a new descendant that contains all items for every missing trees
        let nb_missing_trees = target_n_trees.saturating_sub(roots.len());
        for _ in 0..nb_missing_trees {
            let new_id = concurrent_node_ids.next()?;
            roots.push(new_id);
            large_descendants.insert(new_id);
            self.database.put(
                wtxn,
                &Key::tree(self.index, new_id),
                &Node::Descendants(Descendants { descendants: Cow::Borrowed(&item_indices) }),
            )?;
        }

        self.incremental_index_large_descendants(
            wtxn,
            rng,
            options,
            concurrent_node_ids,
            large_descendants,
        )?;

        tracing::debug!("write the metadata...");
        (options.progress)(WriterProgress { main: MainStep::WriteTheMetadata, sub: None });
        let metadata = Metadata {
            dimensions: self.dimensions.try_into().unwrap(),
            items: item_indices,
            roots: ItemIds::from_slice(&roots),
            distance: D::name(),
        };
        self.database.remap_data_type::<MetadataCodec>().put(
            wtxn,
            &Key::metadata(self.index),
            &metadata,
        )?;

        Ok(())
    }

    /// Remove extraneous trees if any
    fn delete_extra_trees(
        &self,
        wtxn: &mut RwTxn,
        roots: &mut Vec<u32>,
        target_n_trees: usize,
    ) -> Result<(), Error> {
        let extraneous_tree = roots.len().saturating_sub(target_n_trees);

        for _ in 0..extraneous_tree {
            if roots.is_empty() {
                break;
            }
            // We want to remove the oldest trees first
            let root = roots.swap_remove(0);
            self.delete_tree(wtxn, NodeId::tree(root))?;
        }

        Ok(())
    }

    fn target_n_trees(
        &self,
        options: &BuildOption,
        item_indices: &RoaringBitmap,
        updated_items: RoaringBitmap,
        to_delete: &RoaringBitmap,
        roots: &[u32],
        nb_tree_nodes: u64,
    ) -> usize {
        match options.n_trees {
            Some(n) => n,
            // In the case we never made any tree we can roughly guess how many trees we want to build in total
            None if roots.is_empty() => {
                // Full Binary Tree Theorem: The number of leaves in a non-empty full binary tree is one more than the number of internal nodes.
                // Source: https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/BinaryTreeFullThm.html
                //
                // That means we can exactly find the minimal number of tree node required to hold all the items
                // 1. How many descendants do we need:
                let descendant_required = item_indices.len() / self.dimensions as u64;
                // 2. Find the number of tree nodes required per trees
                let tree_nodes_per_tree = descendant_required + 1;
                // 3. Find the number of tree required to get as many tree nodes as item:
                let nb_trees = item_indices.len() / tree_nodes_per_tree;

                nb_trees as usize + 1
            }
            None =>
            // By looking at the number of tree_node per tree with the previous number of items
            // we can guess how many tree we'll need with the new number of items.
            {
                // 1. Estimate the number of nodes per tree; the division is safe because we ensured there was at least one root node above.
                let nodes_per_tree = nb_tree_nodes / roots.len() as u64;
                // 2. Estimate the number of tree we need to have AT LEAST as much tree-nodes than items
                //    /!\ We must use the old number of items here
                let old_n_items = (item_indices - updated_items).union_len(to_delete);

                // TODO: check that again
                (old_n_items / nodes_per_tree) as usize
            }
        }
    }

    /// Loop over the list of large descendants and split them into sub trees with respect to the available memory.
    fn incremental_index_large_descendants<R: Rng + SeedableRng>(
        &self,
        wtxn: &mut RwTxn,
        rng: &mut R,
        options: &BuildOption,
        concurrent_node_ids: ConcurrentNodeIds,
        mut large_descendants: RoaringBitmap,
    ) -> Result<(), Error> {
        (options.progress)(WriterProgress {
            main: MainStep::IncrementalIndexLargeDescendants,
            sub: None,
        });

        while let Some(descendant_id) = large_descendants.select(0) {
            large_descendants.remove_smallest(1);
            let node = self.database.get(wtxn, &Key::tree(self.index, descendant_id))?.unwrap();
            let Node::Descendants(Descendants { descendants }) = node else { unreachable!() };
            let mut descendants = descendants.into_owned();

            // For each steps of the loop we starts by creating a new sub-tree with as many items as possible
            // and then insert all the remaining items that couldn't be selected into this new created tree.
            let (leafs, to_insert) = ImmutableLeafs::new(
                wtxn,
                self.database,
                self.index,
                &mut descendants,
                options.available_memory.unwrap_or(usize::MAX),
            )?;
            let frozen_reader = FrozzenReader {
                leafs: &leafs,
                trees: &ImmutableTrees::empty(),
                concurrent_node_ids: &concurrent_node_ids,
            };

            let mut tmp_nodes = match self.tmpdir.as_ref() {
                Some(path) => TmpNodes::new_in(path)?,
                None => TmpNodes::new()?,
            };
            let (root_id, nb_new_tree_nodes) =
                self.make_tree_in_file(options, &frozen_reader, rng, &to_insert, &mut tmp_nodes)?;
            // We cannot update our father so we're going to overwrite the new root node as ourselves.
            tmp_nodes.remap(root_id.item, descendant_id);

            let tmp_nodes = tmp_nodes.into_bytes_reader()?;
            // We never delete anything while building trees
            for (item_id, item_bytes) in tmp_nodes.to_insert() {
                let key = Key::tree(self.index, item_id);
                self.database.remap_data_type::<Bytes>().put(wtxn, &key, item_bytes)?;
            }

            let descendants_became_too_large = self.insert_items_in_current_trees(
                wtxn,
                rng,
                options,
                descendants,
                &[descendant_id],
                nb_new_tree_nodes,
                &concurrent_node_ids,
            )?;
            large_descendants |= descendants_became_too_large;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_items_in_current_trees<R: Rng + SeedableRng>(
        &self,
        wtxn: &mut RwTxn,
        rng: &mut R,
        options: &BuildOption,
        mut to_insert: RoaringBitmap,
        roots: &[u32],
        nb_tree_nodes: u64,
        concurrent_node_ids: &ConcurrentNodeIds,
    ) -> Result<RoaringBitmap> {
        (options.progress)(WriterProgress { main: MainStep::InsertItemsInCurrentTrees, sub: None });

        let mut large_descendants = RoaringBitmap::new();

        if roots.is_empty() {
            return Ok(large_descendants);
        }

        while !to_insert.is_empty() {
            let immutable_tree_nodes =
                ImmutableTrees::new(wtxn, self.database, self.index, nb_tree_nodes)?;
            let (leafs, to_insert) = ImmutableLeafs::new(
                wtxn,
                self.database,
                self.index,
                &mut to_insert,
                // We let the indexing process uses 2/3 of the memory for the items and the last third for the tree.
                options
                    .available_memory
                    .map_or(usize::MAX, |memory| (memory as f64 * 2.0 / 3.0).floor() as usize),
            )?;
            let frozzen_reader =
                FrozzenReader { leafs: &leafs, trees: &immutable_tree_nodes, concurrent_node_ids };
            let tmp_descendant_to_write =
                self.insert_items_in_tree(options, rng, roots, &to_insert, &frozzen_reader)?;

            for (tmp_node, descendants) in tmp_descendant_to_write.iter() {
                large_descendants |= descendants;

                for item_id in tmp_node.to_delete() {
                    let key = Key::tree(self.index, item_id);
                    self.database.remap_data_type::<Bytes>().delete(wtxn, &key)?;
                }
                for (item_id, item_bytes) in tmp_node.to_insert() {
                    let key = Key::tree(self.index, item_id);
                    self.database.remap_data_type::<Bytes>().put(wtxn, &key, item_bytes)?;
                }
            }
        }
        Ok(large_descendants)
    }

    fn reset_and_retrieve_updated_items(
        &self,
        wtxn: &mut RwTxn,
        options: &BuildOption,
    ) -> Result<RoaringBitmap, Error> {
        tracing::debug!("reset and retrieve the updated items...");
        (options.progress)(WriterProgress { main: MainStep::RetrieveTheUpdatedItems, sub: None });
        let mut updated_items = RoaringBitmap::new();
        let mut updated_iter = self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter_mut(wtxn, &Prefix::updated(self.index))?
            .remap_key_type::<KeyCodec>();
        while let Some((key, _)) = updated_iter.next().transpose()? {
            let inserted = updated_items.push(key.node.item);
            debug_assert!(inserted, "The keys should be sorted by LMDB");
            // Safe because we don't hold any reference to the database currently
            unsafe {
                updated_iter.del_current()?;
            }
        }
        Ok(updated_items)
    }

    fn clear_db_and_create_a_single_leaf(
        &self,
        wtxn: &mut RwTxn,
        options: &BuildOption,
        item_indices: RoaringBitmap,
    ) -> Result<(), Error> {
        tracing::debug!("We can fit every elements in a single descendant node, we can skip all the build process");
        (options.progress)(WriterProgress {
            main: MainStep::WritingTheDescendantsAndMetadata,
            sub: None,
        });
        self.database
            .remap_data_type::<Bytes>()
            .delete_range(wtxn, &(Key::tree(self.index, 0)..=Key::tree(self.index, ItemId::MAX)))?;
        let mut roots = Vec::new();
        if !item_indices.is_empty() {
            // if we have more than 0 elements we need to create a descendant node

            self.database.put(
                wtxn,
                &Key::tree(self.index, 0),
                &Node::Descendants(Descendants { descendants: Cow::Borrowed(&item_indices) }),
            )?;
            roots.push(0);
        }
        tracing::debug!("write the metadata...");
        let metadata = Metadata {
            dimensions: self.dimensions.try_into().unwrap(),
            items: item_indices,
            roots: ItemIds::from_slice(&roots),
            distance: D::name(),
        };
        self.database.remap_data_type::<MetadataCodec>().put(
            wtxn,
            &Key::metadata(self.index),
            &metadata,
        )?;
        tracing::debug!("write the version...");
        let version = Version {
            major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
            minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
            patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
        };
        self.database.remap_data_type::<VersionCodec>().put(
            wtxn,
            &Key::version(self.index),
            &version,
        )?;
        Ok(())
    }

    fn pre_process_items(&self, wtxn: &mut RwTxn, options: &BuildOption) -> Result<(), Error> {
        tracing::debug!("started preprocessing the items...");
        (options.progress)(WriterProgress { main: MainStep::PreProcessingTheItems, sub: None });
        if (options.cancel)() {
            return Err(Error::BuildCancelled);
        }
        D::preprocess(wtxn, |wtxn| {
            Ok(self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter_mut(wtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>())
        })?;
        if (options.cancel)() {
            return Err(Error::BuildCancelled);
        };
        Ok(())
    }

    // TODO: Should we return the list of empty descendants to double check at the very end of the indexing process if
    // they're still empty and should be removed?
    fn delete_items_from_trees(
        &self,
        wtxn: &mut RwTxn,
        options: &BuildOption,
        roots: &mut [u32],
        to_delete: &RoaringBitmap,
    ) -> Result<()> {
        (options.progress)(WriterProgress { main: MainStep::RemoveItems, sub: None });

        let mut tmp_nodes: TmpNodes<NodeCodec<D>> = match self.tmpdir.as_ref() {
            Some(path) => TmpNodes::new_in(path)?,
            None => TmpNodes::new()?,
        };

        // TODO: Parallelize
        for root in roots.iter_mut() {
            let (new_root, _) =
                self.delete_items_in_file(options, wtxn, *root, &mut tmp_nodes, to_delete)?;
            *root = new_root;
        }
        roots.sort_unstable();

        let tmp_nodes = tmp_nodes.into_bytes_reader()?;
        for item_id in tmp_nodes.to_delete() {
            let key = Key::tree(self.index, item_id);
            self.database.remap_data_type::<Bytes>().delete(wtxn, &key)?;
        }
        for (item_id, item_bytes) in tmp_nodes.to_insert() {
            let key = Key::tree(self.index, item_id);
            self.database.remap_data_type::<Bytes>().put(wtxn, &key, item_bytes)?;
        }
        Ok(())
    }

    /// Remove items in O(n). We must explore the whole list of items.
    /// That could be reduced to O(log(n)) if we had a `RoTxn` of the previous state of the database.
    fn delete_items_in_file(
        &self,
        opt: &BuildOption,
        rtxn: &RoTxn,
        current_node: ItemId,
        tmp_nodes: &mut TmpNodes<NodeCodec<D>>,
        to_delete: &RoaringBitmap,
    ) -> Result<(ItemId, RoaringBitmap)> {
        if (opt.cancel)() {
            return Err(Error::BuildCancelled);
        }
        match self.database.get(rtxn, &Key::tree(self.index, current_node))?.unwrap() {
            Node::Leaf(_) => unreachable!(),
            Node::Descendants(Descendants { descendants }) => {
                let len = descendants.len();
                let mut new_descendants = descendants.into_owned();
                new_descendants -= to_delete;

                if len != new_descendants.len() {
                    // update the descendants
                    tmp_nodes.put(
                        current_node,
                        &Node::Descendants(Descendants {
                            descendants: Cow::Borrowed(&new_descendants),
                        }),
                    )?;
                }
                Ok((current_node, new_descendants))
            }
            Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                let (new_left, left_items) = match left.mode {
                    NodeMode::Tree => {
                        let (id, items) =
                            self.delete_items_in_file(opt, rtxn, left.item, tmp_nodes, to_delete)?;
                        (NodeId::tree(id), items)
                    }
                    NodeMode::Item => {
                        if to_delete.contains(left.item) {
                            (left, RoaringBitmap::new())
                        } else {
                            (left, RoaringBitmap::from_sorted_iter(Some(left.item)).unwrap())
                        }
                    }
                    NodeMode::Metadata | NodeMode::Updated => unreachable!(),
                };
                let (new_right, right_items) = match right.mode {
                    NodeMode::Tree => {
                        let (id, items) =
                            self.delete_items_in_file(opt, rtxn, right.item, tmp_nodes, to_delete)?;
                        (NodeId::tree(id), items)
                    }
                    NodeMode::Item => {
                        if to_delete.contains(right.item) {
                            (right, RoaringBitmap::new())
                        } else {
                            (right, RoaringBitmap::from_sorted_iter(Some(right.item)).unwrap())
                        }
                    }
                    NodeMode::Metadata | NodeMode::Updated => unreachable!(),
                };

                let total_items = &left_items | &right_items;

                if self.fit_in_descendant(opt, total_items.len()) {
                    // Since we're shrinking we KNOW that new_left and new_right are descendants
                    // thus we can delete them directly knowing there is no sub-tree to look at.
                    if new_left.mode == NodeMode::Tree {
                        tmp_nodes.remove(new_left.item);
                    }
                    if new_right.mode == NodeMode::Tree {
                        tmp_nodes.remove(new_right.item);
                    }

                    tmp_nodes.put(
                        current_node,
                        &Node::Descendants(Descendants {
                            descendants: Cow::Owned(total_items.clone()),
                        }),
                    )?;

                    // we should merge both branch and update ourselves to be a single descendant node
                    Ok((current_node, total_items))

                // If we don't have any items in either the left or right we can delete ourselves and point directly to our child
                } else if left_items.is_empty() {
                    if new_left.mode == NodeMode::Tree {
                        tmp_nodes.remove(new_left.item);
                    }
                    tmp_nodes.remove(current_node);
                    Ok((new_right.item, total_items))
                } else if right_items.is_empty() {
                    if new_right.mode == NodeMode::Tree {
                        tmp_nodes.remove(new_right.item);
                    }
                    tmp_nodes.remove(current_node);
                    Ok((new_left.item, total_items))
                } else {
                    // if either the left or the right changed we must update ourselves inplace
                    if new_left != left || new_right != right {
                        tmp_nodes.put(
                            current_node,
                            &Node::SplitPlaneNormal(SplitPlaneNormal {
                                normal,
                                left: new_left,
                                right: new_right,
                            }),
                        )?;
                    }

                    // TODO: Should we update the normals if something changed?

                    Ok((current_node, total_items))
                }
            }
        }
    }

    /// Insert items in the specified trees without creating new tree.
    /// Return the list of nodes modified that must be inserted into the database and
    /// the roaring bitmap of descendants that became too large in the process.
    #[allow(clippy::too_many_arguments)]
    fn insert_items_in_tree<R: Rng + SeedableRng>(
        &self,
        opt: &BuildOption,
        rng: &mut R,
        roots: &[u32],
        to_insert: &RoaringBitmap,
        frozen_reader: &FrozzenReader<D>,
    ) -> Result<Vec<(TmpNodesReader, RoaringBitmap)>> {
        repeatn(rng.next_u64(), roots.len())
            .zip(roots)
            .map(|(seed, root)| {
                tracing::debug!("started updating tree {root:X}...");
                let mut rng = R::seed_from_u64(seed.wrapping_add(*root as u64));
                let mut tmp_descendant: TmpNodes<NodeCodec<D>> = match self.tmpdir.as_ref() {
                    Some(path) => TmpNodes::new_in(path)?,
                    None => TmpNodes::new()?,
                };
                let root_node = NodeId::tree(*root);
                let mut large_descendants = RoaringBitmap::new();
                self.insert_items_in_file(
                    opt,
                    frozen_reader,
                    &mut rng,
                    root_node,
                    to_insert,
                    &mut large_descendants,
                    &mut tmp_descendant,
                )?;

                tracing::debug!("finished updating tree {root:X}");
                Ok((tmp_descendant.into_bytes_reader()?, large_descendants))
            })
            .collect()
    }

    /// Find all the descendants that matches the list of items to insert and write them to a file
    #[allow(clippy::too_many_arguments)]
    fn insert_items_in_file<R: Rng>(
        &self,
        opt: &BuildOption,
        frozen_reader: &FrozzenReader<D>,
        rng: &mut R,
        current_node: NodeId,
        to_insert: &RoaringBitmap,
        large_descendants: &mut RoaringBitmap,
        tmp_nodes: &mut TmpNodes<NodeCodec<D>>,
    ) -> Result<ItemId> {
        if (opt.cancel)() {
            return Err(Error::BuildCancelled);
        }
        match current_node.mode {
            NodeMode::Item => {
                // We were called on a specific item, we should create a descendants node
                let mut new_items = RoaringBitmap::from_iter([current_node.item]);
                new_items |= to_insert;

                if !self.fit_in_descendant(opt, new_items.len()) {
                    large_descendants.insert(current_node.item);
                }

                if new_items.len() > 1 {
                    let node_id = frozen_reader.concurrent_node_ids.next()?;
                    let node_id = NodeId::tree(node_id);
                    tmp_nodes.put(
                        node_id.item,
                        &Node::Descendants(Descendants {
                            descendants: Cow::Owned(new_items.clone()),
                        }),
                    )?;
                    Ok(node_id.item)
                } else {
                    Ok(current_node.item)
                }
            }
            NodeMode::Tree => {
                match frozen_reader.trees.get(current_node.item)?.unwrap() {
                    Node::Leaf(_) => unreachable!(),
                    Node::Descendants(Descendants { descendants }) => {
                        let mut new_descendants = descendants.clone().into_owned();
                        // insert all of our IDs in the descendants
                        new_descendants |= to_insert;

                        if !self.fit_in_descendant(opt, new_descendants.len()) {
                            large_descendants.insert(current_node.item);
                        }

                        if descendants.as_ref() != &new_descendants {
                            // otherwise we can just update our descendants
                            tmp_nodes.put(
                                current_node.item,
                                &Node::Descendants(Descendants {
                                    descendants: Cow::Owned(new_descendants.clone()),
                                }),
                            )?;
                        }
                        Ok(current_node.item)
                    }
                    Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                        // Split the to_insert into two bitmaps on the left and right of this normal
                        let mut left_ids = RoaringBitmap::new();
                        let mut right_ids = RoaringBitmap::new();

                        if normal.is_zero() {
                            randomly_split_children(rng, to_insert, &mut left_ids, &mut right_ids);
                        } else {
                            for leaf in to_insert {
                                let node = frozen_reader.leafs.get(leaf)?.unwrap();
                                match D::side(&normal, &node, rng) {
                                    Side::Left => left_ids.insert(leaf),
                                    Side::Right => right_ids.insert(leaf),
                                };
                            }
                        }

                        let new_left = self.insert_items_in_file(
                            opt,
                            frozen_reader,
                            rng,
                            left,
                            &left_ids,
                            large_descendants,
                            tmp_nodes,
                        )?;
                        let new_right = self.insert_items_in_file(
                            opt,
                            frozen_reader,
                            rng,
                            right,
                            &right_ids,
                            large_descendants,
                            tmp_nodes,
                        )?;

                        if new_left != left.item || new_right != right.item {
                            tmp_nodes.put(
                                current_node.item,
                                &Node::SplitPlaneNormal(SplitPlaneNormal {
                                    normal,
                                    left: NodeId::item(new_left),
                                    right: NodeId::item(new_right),
                                }),
                            )?;
                            Ok(current_node.item)
                        } else {
                            Ok(current_node.item)
                        }
                    }
                }
            }
            NodeMode::Metadata => unreachable!(),
            NodeMode::Updated => unreachable!(),
        }
    }

    /// Creates a tree of nodes from the frozzen items that lives
    /// in the database and generates descendants, split normal
    /// and root nodes in files that will be stored in the database later.
    /// Return the root node + the total number of tree node generated.
    fn make_tree_in_file<R: Rng>(
        &self,
        opt: &BuildOption,
        reader: &FrozzenReader<D>,
        rng: &mut R,
        item_indices: &RoaringBitmap,
        tmp_nodes: &mut TmpNodes<NodeCodec<D>>,
    ) -> Result<(NodeId, u64)> {
        if (opt.cancel)() {
            return Err(Error::BuildCancelled);
        }
        if item_indices.len() == 1 {
            return Ok((NodeId::item(item_indices.min().unwrap()), 0));
        }

        if self.fit_in_descendant(opt, item_indices.len()) {
            let item_id = reader.concurrent_node_ids.next()?;
            let item = Node::Descendants(Descendants { descendants: Cow::Borrowed(item_indices) });
            tmp_nodes.put(item_id, &item)?;
            return Ok((NodeId::tree(item_id), 1));
        }

        let children = ImmutableSubsetLeafs::from_item_ids(reader.leafs, item_indices);
        let mut children_left = Vec::with_capacity(children.len() as usize);
        let mut children_right = Vec::with_capacity(children.len() as usize);
        let mut remaining_attempts = 3;

        let mut normal = loop {
            children_left.clear();
            children_right.clear();

            let normal = D::create_split(&children, rng)?;
            for item_id in item_indices.iter() {
                let node = reader.leafs.get(item_id)?.unwrap();
                match D::side(&normal, &node, rng) {
                    Side::Left => children_left.push(item_id),
                    Side::Right => children_right.push(item_id),
                };
            }

            if split_imbalance(children_left.len() as u64, children_right.len() as u64) < 0.95
                || remaining_attempts == 0
            {
                break normal;
            }

            remaining_attempts -= 1;
        };

        // If we didn't find a hyperplane, just randomize sides as a last option
        // and set the split plane to zero as a dummy plane.
        let (children_left, children_right) =
            if split_imbalance(children_left.len() as u64, children_right.len() as u64) > 0.99 {
                let mut children_left = RoaringBitmap::new();
                let mut children_right = RoaringBitmap::new();
                randomly_split_children(rng, item_indices, &mut children_left, &mut children_right);
                UnalignedVector::reset(&mut normal);

                (children_left, children_right)
            } else {
                (
                    RoaringBitmap::from_sorted_iter(children_left).unwrap(),
                    RoaringBitmap::from_sorted_iter(children_right).unwrap(),
                )
            };

        let (left, l) = self.make_tree_in_file(opt, reader, rng, &children_left, tmp_nodes)?;
        let (right, r) = self.make_tree_in_file(opt, reader, rng, &children_right, tmp_nodes)?;
        let normal = SplitPlaneNormal { normal, left, right };

        let new_node_id = reader.concurrent_node_ids.next()?;
        tmp_nodes.put(new_node_id, &Node::SplitPlaneNormal(normal))?;

        Ok((NodeId::tree(new_node_id), l + r + 1))
    }

    fn delete_tree(&self, wtxn: &mut RwTxn, node: NodeId) -> Result<()> {
        let key = Key::new(self.index, node);
        match self.database.get(wtxn, &key)?.ok_or(Error::missing_key(key))? {
            // the leafs are shared between the trees, we MUST NOT delete them.
            Node::Leaf(_) => Ok(()),
            Node::Descendants(_) => {
                self.database.delete(wtxn, &key).map(|_| ()).map_err(Error::from)
            }
            Node::SplitPlaneNormal(SplitPlaneNormal { normal: _, left, right }) => {
                self.delete_tree(wtxn, left)?;
                self.delete_tree(wtxn, right)?;
                self.database.delete(wtxn, &key).map(|_| ()).map_err(Error::from)
            }
        }
    }

    // Fetches the item's ids, not the tree nodes ones.
    fn item_indices(&self, wtxn: &mut RwTxn, options: &BuildOption) -> heed::Result<RoaringBitmap> {
        tracing::debug!("started retrieving all the items ids...");
        (options.progress)(WriterProgress { main: MainStep::RetrievingTheItemsIds, sub: None });

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
    concurrent_node_ids: &'a ConcurrentNodeIds,
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
