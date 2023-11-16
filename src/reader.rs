use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::repeat;
use std::marker;
use std::str::pattern::CharPredicateSearcher;

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

impl<D: Distance> Reader<D> {
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

    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> heed::Result<Option<Vec<f32>>> {
        Ok(item_leaf(self.database, rtxn, item)?.map(|leaf| leaf.vector))
    }

    pub fn nns_by_item(
        &self,
        rtxn: &RoTxn,
        item: ItemId,
        count: usize,
        search_k: Option<usize>, // TODO consider Option<NonZeroUsize>
    ) -> heed::Result<Option<Vec<(ItemId, f32)>>> {
        match item_leaf(self.database, rtxn, item)? {
            Some(leaf) => self.nns_by_leaf(rtxn, &leaf, count, search_k).map(Some),
            None => Ok(None),
        }
    }

    pub fn nns_by_vector(
        &self,
        rtxn: &RoTxn,
        vector: &[f32],
        count: usize,
        search_k: Option<usize>, // TODO consider Option<NonZeroUsize>
    ) -> heed::Result<Vec<(ItemId, f32)>> {
        assert_eq!(
            vector.len(),
            self.dimensions,
            "invalid vector dimensions, provided {} but expected {}",
            vector.len(),
            self.dimensions
        );

        let leaf = Leaf { header: D::new_header(vector), vector: vector.to_vec() };
        self.nns_by_leaf(rtxn, &leaf, count, search_k)
    }

    fn nns_by_leaf(
        &self,
        rtxn: &RoTxn,
        query_leaf: &Leaf<D>,
        count: usize,
        search_k: Option<usize>, // TODO consider Option<NonZeroUsize>
    ) -> heed::Result<Vec<(ItemId, f32)>> {
        // TODO define the capacity
        let mut queue = BinaryHeap::new();
        let search_k = search_k.unwrap_or(count * self.roots.len());

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
                Node::Descendants(Descendants { mut descendants }) => nns.append(&mut descendants),
                Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                    let margin = D::margin(&normal, &query_leaf.vector);
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
            nns_distances.push(Reverse((OrderedFloat(D::distance(query_leaf, &leaf)), nn)));
        }

        let mut sorted_nns = BinaryHeap::from(nns_distances);
        let capacity = count.min(sorted_nns.len());
        let mut output = Vec::with_capacity(capacity);
        while let Some(Reverse((OrderedFloat(dist), item))) = sorted_nns.pop() {
            output.push((item, D::normalized_distance(dist)));
            if output.len() == capacity {
                break;
            }
        }

        Ok(output)
    }
}

pub fn item_leaf<D: Distance>(
    database: Database<BEU32, NodeCodec<D>>,
    rtxn: &RoTxn,
    item: ItemId,
) -> heed::Result<Option<Leaf<D>>> {
    match database.get(rtxn, &item)? {
        Some(Node::Leaf(leaf)) => Ok(Some(leaf)),
        Some(Node::SplitPlaneNormal(_)) => Ok(None),
        Some(Node::Descendants(_)) => Ok(None),
        None => Ok(None),
    }
}
