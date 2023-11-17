use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::repeat;
use std::marker;
use std::num::NonZeroUsize;

use bytemuck::pod_collect_to_vec;
use heed::types::ByteSlice;
use heed::{Database, RoTxn};
use ordered_float::OrderedFloat;

use crate::node::{Descendants, Leaf, SplitPlaneNormal};
use crate::{Distance, ItemId, Node, NodeCodec, NodeId, Side, BEU32};

// TODO use a "metadata" key to store and check the dimensions and distance type
pub struct Reader<D: Distance> {
    database: heed::Database<BEU32, NodeCodec<D>>,
    roots: Vec<NodeId>,
    dimensions: usize,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance + 'static> Reader<D> {
    pub fn open<U>(
        rtxn: &RoTxn,
        database: Database<BEU32, U>,
        dimensions: usize,
    ) -> heed::Result<Reader<D>> {
        let roots = match database.remap_data_type::<ByteSlice>().get(rtxn, &u32::MAX)? {
            Some(roots_bytes) => pod_collect_to_vec(roots_bytes),
            None => Vec::new(),
        };

        Ok(Reader {
            database: database.remap_data_type(),
            roots,
            dimensions,
            _marker: marker::PhantomData,
        })
    }

    /// Returns the number of trees in the index.
    pub fn n_trees(&self) -> usize {
        self.roots.len()
    }

    /// Returns the vector for item `i` that was previously added.
    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> heed::Result<Option<Vec<f32>>> {
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
    ) -> heed::Result<Option<Vec<(ItemId, f32)>>> {
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
    ) -> heed::Result<Vec<(ItemId, f32)>> {
        assert_eq!(
            vector.len(),
            self.dimensions,
            "invalid vector dimensions, provided {} but expected {}",
            vector.len(),
            self.dimensions
        );

        let leaf = Leaf { header: D::new_header(vector), vector: Cow::Borrowed(vector) };
        self.nns_by_leaf(rtxn, &leaf, count, search_k)
    }

    fn nns_by_leaf(
        &self,
        rtxn: &RoTxn,
        query_leaf: &Leaf<D>,
        count: usize,
        search_k: Option<NonZeroUsize>,
    ) -> heed::Result<Vec<(ItemId, f32)>> {
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
) -> heed::Result<Option<Leaf<'a, D>>> {
    match database.get(rtxn, &item)? {
        Some(Node::Leaf(leaf)) => Ok(Some(leaf)),
        Some(Node::SplitPlaneNormal(_)) => Ok(None),
        Some(Node::Descendants(_)) => Ok(None),
        None => Ok(None),
    }
}