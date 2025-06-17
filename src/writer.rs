use std::any::TypeId;
use std::borrow::Cow;
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use crossbeam::channel::{bounded, Sender};
use heed::types::{Bytes, DecodeIgnore, Unit};
use heed::{MdbError, PutFlags, RoTxn, RwTxn};
use nohash::{BuildNoHashHasher, IntMap};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::repeatn;
use rayon::{current_num_threads, prelude::*, Scope};
use roaring::RoaringBitmap;
use thread_local::ThreadLocal;

use crate::distance::Distance;
use crate::internals::{KeyCodec, Side};
use crate::item_iter::ItemIter;
use crate::node::{Descendants, ItemIds, Leaf, SplitPlaneNormal};
use crate::parallel::{
    ConcurrentNodeIds, ImmutableLeafs, ImmutableSubsetLeafs, ImmutableTrees, TmpNodes,
};
use crate::reader::item_leaf;
use crate::unaligned_vector::UnalignedVector;
use crate::version::{Version, VersionCodec};
use crate::{
    Database, Error, ItemId, Key, Metadata, MetadataCodec, Node, NodeCodec, Prefix, PrefixCodec,
    Result,
};

/// The options available when building the arroy database.
pub struct ArroyBuilder<'a, D: Distance, R: Rng + SeedableRng + Send + Sync> {
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
    #[allow(unused)] // Keeping this method because we'll need it once the code is parallelized
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
    RetrievingTheUsedTreeNodes,
    DeletingExtraTrees,
    RemoveItemsFromExistingTrees,
    InsertItemsInCurrentTrees,
    IncrementalIndexLargeDescendants,
    WriteTheMetadata,
}

/// The options available when building the arroy database.
pub(crate) struct BuildOption<'a> {
    pub(crate) n_trees: Option<usize>,
    pub(crate) split_after: Option<usize>,
    pub(crate) available_memory: Option<usize>,
    pub(crate) cancel: Box<dyn Fn() -> bool + 'a + Sync + Send>,
    pub(crate) progress: Box<dyn Fn(WriterProgress) + 'a + Sync + Send>,
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

impl BuildOption<'_> {
    pub(crate) fn cancelled(&self) -> Result<(), Error> {
        if (self.cancel)() {
            Err(Error::BuildCancelled)
        } else {
            Ok(())
        }
    }
}

impl<'a, D: Distance, R: Rng + SeedableRng + Send + Sync> ArroyBuilder<'a, D, R> {
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

    fn used_tree_node(&self, rtxn: &RoTxn, options: &BuildOption) -> Result<RoaringBitmap> {
        (options.progress)(WriterProgress {
            main: MainStep::RetrievingTheUsedTreeNodes,
            sub: None,
        });
        Ok(self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter(rtxn, &Prefix::tree(self.index))?
            .remap_types::<KeyCodec, DecodeIgnore>()
            .try_fold(RoaringBitmap::new(), |mut bitmap, used| -> Result<RoaringBitmap> {
                options.cancelled()?;
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
    pub fn builder<'a, R: Rng + SeedableRng + Send + Sync>(
        &'a self,
        rng: &'a mut R,
    ) -> ArroyBuilder<'a, D, R> {
        ArroyBuilder { writer: self, rng, inner: BuildOption::default() }
    }

    fn build<R: Rng + SeedableRng + Send + Sync>(
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

        tracing::debug!("Getting a reference to your {n_items} items...");

        let used_node_ids = self.used_tree_node(wtxn, options)?;
        let nb_tree_nodes = used_node_ids.len();
        let concurrent_node_ids = ConcurrentNodeIds::new(used_node_ids);

        let target_n_trees = target_n_trees(options, self.dimensions as u64, &item_indices, &roots);
        self.delete_extra_trees(wtxn, options, &mut roots, target_n_trees)?;

        // Before taking any references on the DB, remove all the items we must remove.
        self.delete_items_from_trees(wtxn, options, &mut roots, &to_delete)?;

        // From this point on, we're not going to write anything to the DB until the very end.
        // Each thread will have its own TmpNodes and we're going to write them all to the DB at the end.

        let leafs = ImmutableLeafs::new(wtxn, self.database, &item_indices, self.index)?;
        let immutable_tree_nodes =
            ImmutableTrees::new(wtxn, self.database, self.index, nb_tree_nodes)?;
        let frozen_reader = FrozzenReader {
            leafs: &leafs,
            trees: &immutable_tree_nodes,
            concurrent_node_ids: &concurrent_node_ids,
        };

        let files_tls = Arc::new(ThreadLocal::new());

        // The next method is called from multiple place so we have to update the progress here
        (options.progress)(WriterProgress { main: MainStep::InsertItemsInCurrentTrees, sub: None });
        let mut descendants =
            self.insert_items_in_current_trees(rng, options, to_insert, &roots, &frozen_reader)?;

        // Create a new descendant that contains all items for every missing trees
        let nb_missing_trees = target_n_trees.saturating_sub(roots.len() as u64);
        for _ in 0..nb_missing_trees {
            let new_id = concurrent_node_ids.next()?;
            roots.push(new_id);
            descendants.insert(new_id, item_indices.clone());
        }

        // When a task fails, a message must be sent in this channel.
        // When a taks starts it must check if the channel contains something and stop asap.
        // The channel will be openend only after all the tasks have been stopped.
        let (error_snd, error_rcv) = bounded(1);

        rayon::scope(|s| {
            let frozen_reader = &frozen_reader;
            let files_tls = files_tls.clone();
            let error_snd = error_snd.clone();

            // Spawn a thread that will create all the tasks
            s.spawn(move |s| {
                let rng = StdRng::from_seed(rng.gen());
                let ret = self.insert_descendants_in_file_and_spawn_tasks(
                    rng,
                    options,
                    &error_snd,
                    s,
                    frozen_reader,
                    files_tls,
                    descendants,
                );
                if let Err(e) = ret {
                    let _ = error_snd.try_send(e);
                }
            });
        });

        if let Ok(e) = error_rcv.try_recv() {
            return Err(e);
        }

        let files_tls = Arc::into_inner(files_tls).expect("Threads have all finished their works");
        for file in files_tls.into_iter() {
            let tmp_nodes = file.into_inner().into_bytes_reader()?;
            for (item_id, item_bytes) in tmp_nodes.to_insert() {
                self.database.remap_data_type::<Bytes>().put(
                    wtxn,
                    &Key::tree(self.index, item_id),
                    item_bytes,
                )?;
            }
        }

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
        self.database.remap_data_type::<VersionCodec>().put(
            wtxn,
            &Key::version(self.index),
            &Version::current(),
        )?;

        Ok(())
    }

    /// Remove extraneous trees if any
    fn delete_extra_trees(
        &self,
        wtxn: &mut RwTxn,
        options: &BuildOption,
        roots: &mut Vec<u32>,
        target_n_trees: u64,
    ) -> Result<(), Error> {
        (options.progress)(WriterProgress { main: MainStep::DeletingExtraTrees, sub: None });
        let extraneous_tree = roots.len().saturating_sub(target_n_trees as usize);

        for _ in 0..extraneous_tree {
            options.cancelled()?;
            if roots.is_empty() {
                break;
            }
            // We want to remove the oldest trees first
            let root = roots.swap_remove(0);
            self.delete_tree(wtxn, root)?;
        }

        Ok(())
    }

    /// Loop over the items of the specified descendant and explode it into a tree with respect to the available memory.
    /// Returns the new descendants that are ready to store in the database.
    /// Push more tasks to the scope for all the descendants that are still too large to fit in memory.
    /// Write the tree squeleton to its local tmp file. That file must be written to the DB at the end.
    #[allow(clippy::too_many_arguments)]
    fn incremental_index_large_descendant<'scope, R: Rng + SeedableRng + Send + Sync>(
        &'scope self,
        mut rng: R,
        options: &'scope BuildOption,
        error_snd: &Sender<Error>,
        scope: &Scope<'scope>,
        descendant: (ItemId, RoaringBitmap),
        frozen_reader: &'scope FrozzenReader<D>,
        tmp_nodes: Arc<ThreadLocal<RefCell<TmpNodes<D>>>>,
    ) -> Result<()> {
        (options.progress)(WriterProgress {
            main: MainStep::IncrementalIndexLargeDescendants,
            sub: None,
        });
        options.cancelled()?;
        if error_snd.is_full() {
            return Ok(());
        }

        let tmp_node = tmp_nodes.get_or_try(|| match self.tmpdir.as_ref() {
            Some(path) => TmpNodes::new_in(path).map(RefCell::new),
            None => TmpNodes::new().map(RefCell::new),
        })?;
        // Safe to borrow mut here because we're the only thread running with this variable
        let mut tmp_node = tmp_node.borrow_mut();
        let mut descendants = IntMap::<ItemId, RoaringBitmap>::default();
        let (descendant_id, mut to_insert) = descendant;

        let available_memory =
            options.available_memory.unwrap_or(usize::MAX) / current_num_threads();

        // safe to unwrap because we know the descendant is large
        let items_for_tree =
            fit_in_memory::<D, R>(available_memory, &mut to_insert, self.dimensions, &mut rng)
                .unwrap();

        let (root_id, _nb_new_tree_nodes) = self.make_tree_in_file(
            options,
            frozen_reader,
            error_snd,
            &mut rng,
            &items_for_tree,
            &mut descendants,
            Some(descendant_id),
            &mut tmp_node,
        )?;
        assert_eq!(root_id, descendant_id);

        while let Some(to_insert) =
            fit_in_memory::<D, R>(available_memory, &mut to_insert, self.dimensions, &mut rng)
        {
            options.cancelled()?;
            if error_snd.is_full() {
                return Ok(());
            }

            insert_items_in_descendants_from_tmpfile(
                options,
                frozen_reader,
                &mut tmp_node,
                error_snd,
                &mut rng,
                descendant_id,
                &to_insert,
                &mut descendants,
            )?;
        }

        drop(tmp_node);
        self.insert_descendants_in_file_and_spawn_tasks(
            rng,
            options,
            error_snd,
            scope,
            frozen_reader,
            tmp_nodes,
            descendants,
        )?;

        Ok(())
    }

    /// Explore the IntMap of descendants and when a descendant is too large to fit in memory, spawn a task to index it.
    /// Otherwise, insert the descendant in the tempfile.
    #[allow(clippy::too_many_arguments)]
    fn insert_descendants_in_file_and_spawn_tasks<'scope, R: Rng + SeedableRng + Send + Sync>(
        &'scope self,
        mut rng: R,
        options: &'scope BuildOption,
        error_snd: &Sender<Error>,
        scope: &Scope<'scope>,
        frozen_reader: &'scope FrozzenReader<'_, D>,
        tmp_nodes: Arc<ThreadLocal<RefCell<TmpNodes<D>>>>,
        descendants: IntMap<ItemId, RoaringBitmap>,
    ) -> Result<(), Error> {
        options.cancelled()?;
        if error_snd.is_full() {
            return Ok(());
        }
        let tmp_node = tmp_nodes.get_or_try(|| match self.tmpdir.as_ref() {
            Some(path) => TmpNodes::new_in(path).map(RefCell::new),
            None => TmpNodes::new().map(RefCell::new),
        })?;
        // Safe to borrow mut here because we're the only thread running with this variable
        let mut tmp_node = tmp_node.borrow_mut();

        for (item_id, item_indices) in descendants.into_iter() {
            options.cancelled()?;
            if error_snd.is_full() {
                return Ok(());
            }
            if self.fit_in_descendant(options, item_indices.len()) {
                tmp_node.put(
                    item_id,
                    &Node::Descendants(Descendants { descendants: Cow::Borrowed(&item_indices) }),
                )?;
            } else {
                let tmp_nodes = tmp_nodes.clone();
                let rng = StdRng::from_seed(rng.gen());
                let error_snd = error_snd.clone();
                scope.spawn(move |s| {
                    let ret = self.incremental_index_large_descendant(
                        rng,
                        options,
                        &error_snd,
                        s,
                        (item_id, item_indices),
                        frozen_reader,
                        tmp_nodes,
                    );
                    if let Err(e) = ret {
                        let _ = error_snd.try_send(e);
                    }
                });
            }
        }

        Ok(())
    }

    fn insert_items_in_current_trees<R: Rng + SeedableRng>(
        &self,
        rng: &mut R,
        options: &BuildOption,
        mut to_insert: RoaringBitmap,
        roots: &[ItemId],
        frozen_reader: &FrozzenReader<D>,
    ) -> Result<IntMap<ItemId, RoaringBitmap>> {
        if roots.is_empty() {
            return Ok(IntMap::default());
        }

        let mut descendants = IntMap::<ItemId, RoaringBitmap>::default();

        while let Some(to_insert) = fit_in_memory::<D, R>(
            options.available_memory.unwrap_or(usize::MAX),
            &mut to_insert,
            self.dimensions,
            rng,
        ) {
            options.cancelled()?;

            let desc = self.insert_items_in_tree(options, rng, roots, &to_insert, frozen_reader)?;
            for (item_id, desc) in desc {
                descendants.entry(item_id).or_default().extend(desc);
            }
        }

        Ok(descendants)
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
            options.cancelled()?;
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
        options.cancelled()?;
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
        let version = Version::current();
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
        options.cancelled()?;
        D::preprocess(wtxn, |wtxn| {
            Ok(self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter_mut(wtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>())
        })?;
        Ok(())
    }

    fn delete_items_from_trees(
        &self,
        wtxn: &mut RwTxn,
        options: &BuildOption,
        roots: &mut [ItemId],
        to_delete: &RoaringBitmap,
    ) -> Result<()> {
        (options.progress)(WriterProgress {
            main: MainStep::RemoveItemsFromExistingTrees,
            sub: None,
        });

        let mut tmp_nodes: TmpNodes<D> = match self.tmpdir.as_ref() {
            Some(path) => TmpNodes::new_in(path)?,
            None => TmpNodes::new()?,
        };

        for root in roots.iter_mut() {
            options.cancelled()?;
            let (new_root, _) =
                self.delete_items_in_file(options, wtxn, *root, &mut tmp_nodes, to_delete)?;
            *root = new_root;
        }
        roots.sort_unstable();

        let tmp_nodes = tmp_nodes.into_bytes_reader()?;
        for item_id in tmp_nodes.to_delete() {
            options.cancelled()?;
            let key = Key::tree(self.index, item_id);
            self.database.remap_data_type::<Bytes>().delete(wtxn, &key)?;
        }
        for (item_id, item_bytes) in tmp_nodes.to_insert() {
            options.cancelled()?;
            let key = Key::tree(self.index, item_id);
            self.database.remap_data_type::<Bytes>().put(wtxn, &key, item_bytes)?;
        }
        Ok(())
    }

    /// Remove items in O(n). We must explore the whole list of items.
    /// That could be reduced to O(log(n)) if we had a `RoTxn` of the previous state of the database.
    /// Return the new node id and the list of all the items contained in this branch.
    /// If there is too many items to create a single descendant, we return `None`.
    fn delete_items_in_file(
        &self,
        options: &BuildOption,
        rtxn: &RoTxn,
        current_node: ItemId,
        tmp_nodes: &mut TmpNodes<D>,
        to_delete: &RoaringBitmap,
    ) -> Result<(ItemId, Option<RoaringBitmap>)> {
        options.cancelled()?;
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
                Ok((current_node, Some(new_descendants)))
            }
            Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                let (new_left, left_items) =
                    self.delete_items_in_file(options, rtxn, left, tmp_nodes, to_delete)?;

                let (new_right, right_items) =
                    self.delete_items_in_file(options, rtxn, right, tmp_nodes, to_delete)?;

                match (left_items, right_items) {
                    (Some(left_items), right_items) if left_items.is_empty() => {
                        tmp_nodes.remove(new_left);
                        tmp_nodes.remove(current_node);
                        Ok((new_right, right_items))
                    }
                    (left_items, Some(right_items)) if right_items.is_empty() => {
                        tmp_nodes.remove(new_right);
                        tmp_nodes.remove(current_node);
                        Ok((new_left, left_items))
                    }
                    (Some(left_items), Some(right_items)) => {
                        let total_items = left_items.len() + right_items.len();
                        if self.fit_in_descendant(options, total_items) {
                            let total_items = left_items | right_items;
                            // Since we're shrinking we KNOW that new_left and new_right are descendants
                            // thus we can delete them directly knowing there is no sub-tree to look at.
                            tmp_nodes.remove(new_left);
                            tmp_nodes.remove(new_right);

                            tmp_nodes.put(
                                current_node,
                                &Node::Descendants(Descendants {
                                    descendants: Cow::Borrowed(&total_items),
                                }),
                            )?;

                            // we should merge both branch and update ourselves to be a single descendant node
                            Ok((current_node, Some(total_items)))
                        } else {
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
                            Ok((current_node, None))
                        }
                    }
                    (None, Some(_)) | (Some(_), None) | (None, None) => {
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
                        Ok((current_node, None))
                    }
                }
            }
        }
    }

    /// Insert items in the specified trees without creating new tree nodes.
    /// Return the list of nodes modified that must be inserted into the database and
    /// the roaring bitmap of descendants that became too large in the process.
    fn insert_items_in_tree<R: Rng + SeedableRng>(
        &self,
        opt: &BuildOption,
        rng: &mut R,
        roots: &[ItemId],
        to_insert: &RoaringBitmap,
        frozen_reader: &FrozzenReader<D>,
    ) -> Result<IntMap<ItemId, RoaringBitmap>> {
        repeatn(rng.next_u64(), roots.len())
            .zip(roots)
            .map(|(seed, root)| {
                opt.cancelled()?;
                tracing::debug!("started updating tree {root:X}...");
                let mut rng = R::seed_from_u64(seed.wrapping_add(*root as u64));
                let mut descendants_to_update = IntMap::with_hasher(BuildNoHashHasher::default());
                insert_items_in_descendants_from_frozen_reader(
                    opt,
                    frozen_reader,
                    &mut rng,
                    *root,
                    to_insert,
                    &mut descendants_to_update,
                )?;

                tracing::debug!("finished updating tree {root:X}");
                Ok(descendants_to_update)
            })
            .reduce(
                || Ok(IntMap::with_hasher(BuildNoHashHasher::default())),
                |acc, descendants_to_update| match (acc, descendants_to_update) {
                    (Err(e), _) | (_, Err(e)) => Err(e),
                    (Ok(mut acc), Ok(descendants_to_update)) => {
                        for (item_id, descendants) in descendants_to_update.into_iter() {
                            acc.entry(item_id).or_default().extend(descendants.clone());
                        }
                        Ok(acc)
                    }
                },
            )
    }

    /// Creates a tree of nodes from the frozzen items that lives
    /// in the database and generates descendants, split normal
    /// and root nodes in files that will be stored in the database later.
    /// Return the root node + the total number of tree node generated.
    #[allow(clippy::too_many_arguments)]
    fn make_tree_in_file<R: Rng>(
        &self,
        opt: &BuildOption,
        reader: &FrozzenReader<D>,
        error_snd: &Sender<Error>,
        rng: &mut R,
        item_indices: &RoaringBitmap,
        descendants: &mut IntMap<ItemId, RoaringBitmap>,
        next_id: Option<ItemId>,
        tmp_nodes: &mut TmpNodes<D>,
    ) -> Result<(ItemId, u64)> {
        opt.cancelled()?;
        if error_snd.is_full() {
            // We can return anything as the whole process is being stopped and nothing will be written
            return Ok((0, 0));
        }
        if self.fit_in_descendant(opt, item_indices.len()) {
            let item_id = next_id.map(Ok).unwrap_or_else(|| reader.concurrent_node_ids.next())?;
            // Don't write the descendants to the tmp nodes yet because they may become too large later
            descendants.insert(item_id, item_indices.clone());
            return Ok((item_id, 1));
        }

        let children = ImmutableSubsetLeafs::from_item_ids(reader.leafs, item_indices);
        let mut children_left = Vec::with_capacity(children.len() as usize);
        let mut children_right = Vec::with_capacity(children.len() as usize);
        let mut remaining_attempts = 3;

        let mut normal = loop {
            opt.cancelled()?;
            children_left.clear();
            children_right.clear();

            let normal = D::create_split(&children, rng)?;
            for item_id in item_indices.iter() {
                let node = reader.leafs.get(item_id)?.unwrap();
                match D::side(&normal, &node) {
                    Side::Left => children_left.push(item_id),
                    Side::Right => children_right.push(item_id),
                };
            }

            if split_imbalance(children_left.len() as u64, children_right.len() as u64) < 0.95
                || remaining_attempts == 0
            {
                break Some(normal);
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
                normal = None;

                (children_left, children_right)
            } else {
                (
                    RoaringBitmap::from_sorted_iter(children_left).unwrap(),
                    RoaringBitmap::from_sorted_iter(children_right).unwrap(),
                )
            };

        let (left, l) = self.make_tree_in_file(
            opt,
            reader,
            error_snd,
            rng,
            &children_left,
            descendants,
            None,
            tmp_nodes,
        )?;
        let (right, r) = self.make_tree_in_file(
            opt,
            reader,
            error_snd,
            rng,
            &children_right,
            descendants,
            None,
            tmp_nodes,
        )?;
        let normal = SplitPlaneNormal { normal, left, right };

        let new_node_id = next_id.map(Ok).unwrap_or_else(|| reader.concurrent_node_ids.next())?;
        tmp_nodes.put(new_node_id, &Node::SplitPlaneNormal(normal))?;

        Ok((new_node_id, l + r + 1))
    }

    fn delete_tree(&self, wtxn: &mut RwTxn, node: ItemId) -> Result<()> {
        let key = Key::tree(self.index, node);
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
    fn item_indices(&self, wtxn: &mut RwTxn, options: &BuildOption) -> Result<RoaringBitmap> {
        tracing::debug!("started retrieving all the items ids...");
        (options.progress)(WriterProgress { main: MainStep::RetrievingTheItemsIds, sub: None });

        let mut indices = RoaringBitmap::new();
        for result in self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(wtxn, &Prefix::item(self.index))?
            .remap_key_type::<KeyCodec>()
        {
            options.cancelled()?;
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

/// Return the number of trees we should have at the end of the indexing process.
/// The number may be bigger or smaller than the current number of trees in the database
/// but won't shrink too quickly if the number of items slightly decreases.
pub(crate) fn target_n_trees(
    options: &BuildOption,
    dimensions: u64,
    item_indices: &RoaringBitmap,
    roots: &[u32],
) -> u64 {
    match options.n_trees {
        Some(n) => n as u64,
        // In the case we never made any tree we can roughly guess how many trees we want to build in total
        None => {
            // Full Binary Tree Theorem: The number of leaves in a non-empty full binary tree is one more than the number of internal nodes.
            // Source: https://opendsa-server.cs.vt.edu/ODSA/Books/CS3/html/BinaryTreeFullThm.html
            //
            // That means we can exactly find the minimal number of tree node required to hold all the items
            // 1. How many descendants do we need:
            let descendant_required = item_indices.len() / dimensions;
            // 2. Find the number of tree nodes required per trees
            let tree_nodes_per_tree = descendant_required + 1;
            // 3. Find the number of tree required to get as many tree nodes as item:
            let mut nb_trees = item_indices.len() / tree_nodes_per_tree;

            // 4. We don't want to shrink too quickly when a user remove some documents.
            //    We're only going to shrink if we should remove more than 20% of our trees.
            if (roots.len() as u64) > nb_trees {
                let tree_to_remove = roots.len() as u64 - nb_trees;
                if (tree_to_remove as f64 / nb_trees as f64) < 0.20 {
                    nb_trees = roots.len() as u64;
                }
            }

            nb_trees
        }
    }
}

/// Find all the descendants that matches the list of items to insert and add them to the descendants_to_update map
#[allow(clippy::too_many_arguments)]
fn insert_items_in_descendants_from_frozen_reader<D: Distance, R: Rng>(
    opt: &BuildOption,
    frozen_reader: &FrozzenReader<D>,
    rng: &mut R,
    current_node: ItemId,
    to_insert: &RoaringBitmap,
    descendants_to_update: &mut IntMap<ItemId, RoaringBitmap>,
) -> Result<()> {
    opt.cancelled()?;
    match frozen_reader.trees.get(current_node)?.unwrap() {
        Node::Leaf(_) => unreachable!(),
        Node::Descendants(Descendants { descendants }) => {
            descendants_to_update.insert(current_node, descendants.into_owned() | to_insert);
        }
        Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
            // Split the to_insert into two bitmaps on the left and right of this normal
            let mut left_ids = RoaringBitmap::new();
            let mut right_ids = RoaringBitmap::new();

            match normal {
                None => {
                    randomly_split_children(rng, to_insert, &mut left_ids, &mut right_ids);
                }
                Some(ref normal) => {
                    for leaf in to_insert {
                        let node = frozen_reader.leafs.get(leaf)?.unwrap();
                        match D::side(normal, &node) {
                            Side::Left => left_ids.insert(leaf),
                            Side::Right => right_ids.insert(leaf),
                        };
                    }
                }
            }

            if !left_ids.is_empty() {
                insert_items_in_descendants_from_frozen_reader(
                    opt,
                    frozen_reader,
                    rng,
                    left,
                    &left_ids,
                    descendants_to_update,
                )?;
            }
            if !right_ids.is_empty() {
                insert_items_in_descendants_from_frozen_reader(
                    opt,
                    frozen_reader,
                    rng,
                    right,
                    &right_ids,
                    descendants_to_update,
                )?;
            }
        }
    }
    Ok(())
}

/// Find all the descendants that matches the list of items to insert and add them to the descendants_to_update map
#[allow(clippy::too_many_arguments)]
fn insert_items_in_descendants_from_tmpfile<D: Distance, R: Rng>(
    opt: &BuildOption,
    // We still need this to read the leafs
    frozen_reader: &FrozzenReader<D>,
    // Must be mutable because we're going to seek and read in it
    tmp_nodes: &mut TmpNodes<D>,
    error_snd: &Sender<Error>,
    rng: &mut R,
    current_node: ItemId,
    to_insert: &RoaringBitmap,
    descendants_to_update: &mut IntMap<ItemId, RoaringBitmap>,
) -> Result<()> {
    opt.cancelled()?;
    if error_snd.is_full() {
        return Ok(());
    }
    match tmp_nodes.get(current_node)? {
        Some(Node::Leaf(_) | Node::Descendants(_)) => unreachable!(),
        None => {
            *descendants_to_update.get_mut(&current_node).unwrap() |= to_insert;
        }
        Some(Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right })) => {
            // Split the to_insert into two bitmaps on the left and right of this normal
            let mut left_ids = RoaringBitmap::new();
            let mut right_ids = RoaringBitmap::new();

            match normal {
                None => {
                    randomly_split_children(rng, to_insert, &mut left_ids, &mut right_ids);
                }
                Some(ref normal) => {
                    for leaf in to_insert {
                        let node = frozen_reader.leafs.get(leaf)?.unwrap();
                        match D::side(normal, &node) {
                            Side::Left => left_ids.insert(leaf),
                            Side::Right => right_ids.insert(leaf),
                        };
                    }
                }
            }

            if !left_ids.is_empty() {
                insert_items_in_descendants_from_tmpfile(
                    opt,
                    frozen_reader,
                    tmp_nodes,
                    error_snd,
                    rng,
                    left,
                    &left_ids,
                    descendants_to_update,
                )?;
            }
            if !right_ids.is_empty() {
                insert_items_in_descendants_from_tmpfile(
                    opt,
                    frozen_reader,
                    tmp_nodes,
                    error_snd,
                    rng,
                    right,
                    &right_ids,
                    descendants_to_update,
                )?;
            }
        }
    }
    Ok(())
}

/// Returns the items from the `to_insert` that fit in memory.
/// If there is no items to insert anymore, returns `None`.
/// If everything fits in memory, returns the `to_insert` bitmap.
fn fit_in_memory<D: Distance, R: Rng>(
    memory: usize,
    to_insert: &mut RoaringBitmap,
    dimensions: usize,
    rng: &mut R,
) -> Option<RoaringBitmap> {
    if to_insert.is_empty() {
        return None;
    } else if to_insert.len() <= dimensions as u64 {
        // We need at least dimensions + one extra item to create a split.
        // If we return less than that it won't be used.
        return Some(std::mem::take(to_insert));
    }

    let page_size = page_size::get();
    let nb_page_allowed = (memory as f64 / page_size as f64).floor() as usize;
    let largest_item_size = D::size_of_item(dimensions);
    let nb_items_per_page = page_size / largest_item_size;
    let nb_page_per_item = (largest_item_size as f64 / page_size as f64).ceil() as usize;

    let nb_items = if nb_items_per_page > 1 {
        debug_assert_eq!(nb_page_per_item, 1);
        nb_page_allowed * nb_items_per_page
    } else if nb_page_per_item > 1 {
        debug_assert_eq!(nb_items_per_page, 1);
        nb_page_allowed / nb_page_per_item
    } else {
        nb_page_allowed
    };

    if nb_items as u64 >= to_insert.len() {
        return Some(std::mem::take(to_insert));
    }

    let mut items = RoaringBitmap::new();

    for _ in 0..nb_items {
        let idx = rng.gen_range(0..to_insert.len());
        // Safe to unwrap because we know nb_items is smaller than the number of items in the bitmap
        let item = to_insert.select(idx as u32).unwrap();
        items.push(item);
        to_insert.remove(item);
    }

    Some(items)
}
