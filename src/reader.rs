use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::repeat;
use std::marker;
use std::num::NonZeroUsize;

use heed::{Database, RoTxn};
use ordered_float::OrderedFloat;

use crate::node::{Descendants, Leaf, SplitPlaneNormal};
use crate::{Distance, ItemId, MetadataCodec, Node, NodeCodec, NodeId, Result, Side, BEU32};

pub struct Reader<D: Distance> {
    database: heed::Database<BEU32, NodeCodec<D>>,
    roots: Vec<NodeId>,
    dimensions: usize,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance + 'static> Reader<D> {
    pub fn open<U>(rtxn: &RoTxn, database: Database<BEU32, U>) -> Result<Reader<D>> {
        let metadata = match database.remap_data_type::<MetadataCodec>().get(rtxn, &u32::MAX)? {
            Some(metadata) => metadata,
            None => todo!("Invalid database"),
        };

        Ok(Reader {
            database: database.remap_data_type(),
            roots: metadata.root_nodes.to_vec(),
            dimensions: metadata.dimensions,
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

    /// Returns the number of nodes in the index. Useful to run an exhaustive search.
    pub fn n_nodes(&self, rtxn: &RoTxn) -> Result<Option<NonZeroUsize>> {
        Ok(NonZeroUsize::new(self.database.len(rtxn)? as usize))
    }

    /// Returns the vector for item `i` that was previously added.
    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> Result<Option<Vec<f32>>> {
        Ok(item_leaf(self.database, rtxn, item)?.map(|leaf| leaf.vector.into_owned()))
    }

    /// Returns the `count` closest items from `item`.
    ///
    /// During the query it will inspect up to `search_k` nodes which defaults
    /// to `n_trees * count` if not provided. `search_k` gives you a run-time
    /// tradeoff between better accuracy and speed.
    pub fn nns_by_item(
        &self,
        rtxn: &RoTxn,
        item: ItemId,
        count: usize,
        search_k: Option<NonZeroUsize>,
    ) -> Result<Option<Vec<(ItemId, f32)>>> {
        match item_leaf(self.database, rtxn, item)? {
            Some(leaf) => self.nns_by_leaf(rtxn, &leaf, count, search_k).map(Some),
            None => Ok(None),
        }
    }

    /// Returns the `count` closest items from the provided `vector`.
    ///
    /// See [`Reader::nns_by_item`] for more details.
    pub fn nns_by_vector(
        &self,
        rtxn: &RoTxn,
        vector: &[f32],
        count: usize,
        search_k: Option<NonZeroUsize>,
    ) -> Result<Vec<(ItemId, f32)>> {
        if vector.len() != self.dimensions {
            return Err(crate::Error::InvalidVecDimension {
                expected: self.dimensions(),
                received: vector.len(),
            });
        }

        let leaf = Leaf { header: D::new_header(vector), vector: Cow::Borrowed(vector) };
        self.nns_by_leaf(rtxn, &leaf, count, search_k)
    }

    fn nns_by_leaf(
        &self,
        rtxn: &RoTxn,
        query_leaf: &Leaf<D>,
        count: usize,
        search_k: Option<NonZeroUsize>,
    ) -> Result<Vec<(ItemId, f32)>> {
        // TODO define the capacity
        let mut queue = BinaryHeap::new();
        let search_k = search_k.map_or(count * self.roots.len(), NonZeroUsize::get);

        // Insert all the root nodes and associate them to the highest distance.
        queue.extend(repeat(OrderedFloat(f32::INFINITY)).zip(self.roots.iter().copied()));

        let mut nns = Vec::<ItemId>::new();
        while nns.len() < search_k {
            let (OrderedFloat(dist), item) = match queue.pop() {
                Some(out) => out,
                None => break,
            };

            match self.database.get(rtxn, &item)?.unwrap() {
                Node::Leaf(_) => nns.push(item),
                // TODO introduce an iterator method to avoid deserializing into a vector
                Node::Descendants(Descendants { mut descendants }) => match descendants {
                    Cow::Borrowed(descendants) => nns.extend_from_slice(descendants),
                    Cow::Owned(ref mut descendants) => nns.append(descendants),
                },
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
            let leaf = match self.database.get(rtxn, &nn)?.unwrap() {
                Node::Leaf(leaf) => leaf,
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => panic!("Shouldn't happen"),
            };
            let distance = D::distance(query_leaf, &leaf);
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
}

pub fn item_leaf<'a, D: Distance + 'static>(
    database: Database<BEU32, NodeCodec<D>>,
    rtxn: &'a RoTxn,
    item: ItemId,
) -> Result<Option<Leaf<'a, D>>> {
    match database.get(rtxn, &item)? {
        Some(Node::Leaf(leaf)) => Ok(Some(leaf)),
        Some(Node::SplitPlaneNormal(_)) => Ok(None),
        Some(Node::Descendants(_)) => Ok(None),
        None => Ok(None),
    }
}
