use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt::Write;
use std::iter::repeat;
use std::num::NonZeroUsize;
use std::{io, marker};

use heed::RoTxn;
use ordered_float::OrderedFloat;

use crate::distance::Distance;
use crate::internals::{KeyCodec, Side};
use crate::item_iter::ItemIter;
use crate::node::{Descendants, ItemIds, Leaf, SplitPlaneNormal, UnalignedF32Slice};
use crate::{
    Database, Error, ItemId, Key, MetadataCodec, Node, NodeId, Prefix, PrefixCodec, Result, Stats,
    TreeStats,
};

/// A reader over the arroy trees and user items.
#[derive(Debug)]
pub struct Reader<'t, D: Distance> {
    database: Database<D>,
    index: u16,
    roots: ItemIds<'t>,
    dimensions: usize,
    n_items: usize,
    _marker: marker::PhantomData<D>,
}

impl<'t, D: Distance> Reader<'t, D> {
    /// Returns a reader over the database with the specified [`Distance`] type.
    pub fn open(rtxn: &'t RoTxn, index: u16, database: Database<D>) -> Result<Reader<'t, D>> {
        let metadata_key = Key::metadata(index);
        let metadata = match database.remap_data_type::<MetadataCodec>().get(rtxn, &metadata_key)? {
            Some(metadata) => metadata,
            None => return Err(Error::MissingMetadata),
        };

        if D::name() != metadata.distance {
            return Err(Error::UnmatchingDistance {
                expected: metadata.distance.to_owned(),
                received: D::name(),
            });
        }

        Ok(Reader {
            database: database.remap_data_type(),
            index,
            roots: metadata.roots,
            dimensions: metadata.dimensions.try_into().unwrap(),
            n_items: metadata.n_items.try_into().unwrap(),
            _marker: marker::PhantomData,
        })
    }

    /// Returns the number of dimensions in the index.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the number of trees in the index.
    pub fn n_trees(&self) -> usize {
        self.roots.len()
    }

    /// Returns the number of vectors stored in the index.
    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// Returns the index of this reader in the database.
    pub fn index(&self) -> u16 {
        self.index
    }

    /// Returns the stats of the trees of this database.
    pub fn stats(&self, rtxn: &RoTxn) -> Result<Stats> {
        fn recursive_depth<D: Distance>(
            rtxn: &RoTxn,
            database: Database<D>,
            index: u16,
            node_id: NodeId,
        ) -> Result<TreeStats> {
            match database.get(rtxn, &Key::new(index, node_id))?.unwrap() {
                Node::Leaf(_) => Ok(TreeStats { depth: 1, dummy_normals: 0 }),
                Node::Descendants(_) => Ok(TreeStats { depth: 1, dummy_normals: 0 }),
                Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                    let left = recursive_depth(rtxn, database, index, left)?;
                    let right = recursive_depth(rtxn, database, index, right)?;
                    let is_zero_normal = normal.iter().all(|f| f == 0.0) as usize;

                    Ok(TreeStats {
                        depth: 1 + left.depth.max(right.depth),
                        dummy_normals: left.dummy_normals + right.dummy_normals + is_zero_normal,
                    })
                }
            }
        }

        let tree_stats: Result<Vec<_>> = self
            .roots
            .iter()
            .map(NodeId::tree)
            .map(|root| recursive_depth::<D>(rtxn, self.database, self.index, root))
            .collect();

        Ok(Stats { tree_stats: tree_stats? })
    }

    /// Returns the number of nodes in the index. Useful to run an exhaustive search.
    pub fn n_nodes(&self, rtxn: &'t RoTxn) -> Result<Option<NonZeroUsize>> {
        Ok(NonZeroUsize::new(self.database.len(rtxn)? as usize))
    }

    /// Returns the vector for item `i` that was previously added.
    pub fn item_vector(&self, rtxn: &'t RoTxn, item: ItemId) -> Result<Option<Vec<f32>>> {
        Ok(item_leaf(self.database, self.index, rtxn, item)?.map(|leaf| leaf.vector.into_owned()))
    }

    /// Returns an iterator over the items vector.
    pub fn iter(&self, rtxn: &'t RoTxn) -> Result<ItemIter<'t, D>> {
        Ok(ItemIter {
            inner: self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter(rtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>(),
        })
    }

    /// Returns the `count` closests items from `item`.
    ///
    /// During the query it will inspect up to `search_k` nodes which defaults
    /// to `n_trees * count` if not provided. `search_k` gives you a run-time
    /// tradeoff between better accuracy and speed.
    pub fn nns_by_item(
        &self,
        rtxn: &'t RoTxn,
        item: ItemId,
        count: usize,
        search_k: Option<NonZeroUsize>,
    ) -> Result<Option<Vec<(ItemId, f32)>>> {
        match item_leaf(self.database, self.index, rtxn, item)? {
            Some(leaf) => self.nns_by_leaf(rtxn, &leaf, count, search_k).map(Some),
            None => Ok(None),
        }
    }

    /// Returns the `count` closest items from the provided `vector`.
    ///
    /// See [`Reader::nns_by_item`] for more details.
    pub fn nns_by_vector(
        &self,
        rtxn: &'t RoTxn,
        vector: &[f32],
        count: usize,
        search_k: Option<NonZeroUsize>,
    ) -> Result<Vec<(ItemId, f32)>> {
        if vector.len() != self.dimensions {
            return Err(Error::InvalidVecDimension {
                expected: self.dimensions(),
                received: vector.len(),
            });
        }

        let vector = UnalignedF32Slice::from_slice(vector);
        let leaf = Leaf { header: D::new_header(vector), vector: Cow::Borrowed(vector) };
        self.nns_by_leaf(rtxn, &leaf, count, search_k)
    }

    fn nns_by_leaf(
        &self,
        rtxn: &'t RoTxn,
        query_leaf: &Leaf<D>,
        count: usize,
        search_k: Option<NonZeroUsize>,
    ) -> Result<Vec<(ItemId, f32)>> {
        // Since the datastructure describes a kind of btree, the capacity is something in the order of:
        // The number of root nodes + log2 of the total number of vectors.
        let mut queue = BinaryHeap::with_capacity(self.roots.len() + self.n_items.ilog2() as usize);
        let search_k = search_k.map_or(count * self.roots.len(), NonZeroUsize::get);

        // Insert all the root nodes and associate them to the highest distance.
        queue.extend(repeat(OrderedFloat(f32::INFINITY)).zip(self.roots.iter().map(NodeId::tree)));

        let mut nns = Vec::new();
        while nns.len() < search_k {
            let (OrderedFloat(dist), item) = match queue.pop() {
                Some(out) => out,
                None => break,
            };

            match self.database.get(rtxn, &Key::new(self.index, item))?.unwrap() {
                Node::Leaf(_) => nns.push(item.unwrap_item()),
                Node::Descendants(Descendants { descendants }) => nns.extend(descendants.iter()),
                Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                    let margin = D::margin_no_header(&normal, &query_leaf.vector);
                    queue.push((OrderedFloat(D::pq_distance(dist, margin, Side::Left)), left));
                    queue.push((OrderedFloat(D::pq_distance(dist, margin, Side::Right)), right));
                }
            }
        }

        // Get distances for all items
        // To avoid calculating distance multiple times for any items, sort by id and dedup by id.
        nns.sort_unstable();
        nns.dedup();

        let mut nns_distances = Vec::with_capacity(nns.len());
        for nn in nns {
            let leaf = match self.database.get(rtxn, &Key::item(self.index, nn))?.unwrap() {
                Node::Leaf(leaf) => leaf,
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => unreachable!(),
            };
            let distance = D::built_distance(query_leaf, &leaf);
            nns_distances.push(Reverse((OrderedFloat(distance), nn)));
        }

        let mut sorted_nns = BinaryHeap::from(nns_distances);
        let capacity = count.min(sorted_nns.len());
        let mut output = Vec::with_capacity(capacity);
        while let Some(Reverse((OrderedFloat(dist), item))) = sorted_nns.pop() {
            if output.len() == capacity {
                break;
            }
            output.push((item, D::normalized_distance(dist)));
        }

        Ok(output)
    }

    /// Write the internal arroy graph in dot format into the provided writer.
    pub fn plot_internals(&self, rtxn: &RoTxn, mut writer: impl io::Write) -> Result<()> {
        writeln!(writer, "digraph {{")?;
        writeln!(writer, "\tlabel=metadata")?;
        writeln!(writer)?;

        for tree in self.roots.iter().take(1) {
            // subgraph {
            //   a -> b
            //   a -> b
            //   b -> a
            // }

            // Start creating the graph
            writeln!(writer, "\tsubgraph {{")?;
            // writeln!(writer, "\t\tlabel={}", tree)?;
            writeln!(writer, "\t\troot [color=red]")?;
            writeln!(writer, "\t\troot -> {tree}")?;

            let mut explore = vec![Key::tree(self.index, tree)];
            while let Some(key) = explore.pop() {
                let node = self.database.get(rtxn, &key)?.unwrap();
                match node {
                    Node::Leaf(_) => (),
                    Node::Descendants(iter) => writeln!(
                        writer,
                        "\t\t{} [label=\"{}-{}\"]",
                        key.node.item,
                        key.node.item,
                        iter.descendants.len()
                    )?,
                    Node::SplitPlaneNormal(SplitPlaneNormal { normal: _, left, right }) => {
                        writeln!(writer, "\t\t{} -> {}", key.node.item, left.item)?;
                        writeln!(writer, "\t\t{} -> {}", key.node.item, right.item)?;
                        explore.push(Key::tree(self.index, left.item));
                        explore.push(Key::tree(self.index, right.item));
                    }
                }
            }

            writeln!(writer, "\t}}")?;
        }

        writeln!(writer, "}}")?;

        Ok(())
    }
}

pub fn item_leaf<'a, D: Distance>(
    database: Database<D>,
    index: u16,
    rtxn: &'a RoTxn,
    item: ItemId,
) -> Result<Option<Leaf<'a, D>>> {
    match database.get(rtxn, &Key::item(index, item))? {
        Some(Node::Leaf(leaf)) => Ok(Some(leaf)),
        Some(Node::SplitPlaneNormal(_)) => Ok(None),
        Some(Node::Descendants(_)) => Ok(None),
        None => Ok(None),
    }
}
