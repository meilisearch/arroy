use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker;

use bytemuck::pod_collect_to_vec;
use heed::types::ByteSlice;
use heed::{Database, RoTxn};
use ordered_float::OrderedFloat;

use crate::node::{Leaf, SplitPlaneNormal};
use crate::priority_queue::BinaryHeapItem;
use crate::{Distance, ItemId, Node, NodeCodec, NodeId, BEU32};

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
        item_vector(self.database, rtxn, item)
    }

    pub fn nns_by_item(
        &self,
        rtxn: &RoTxn,
        item: ItemId,
        count: usize,
        search_k: Option<usize>, // TODO consider Option<NonZeroUsize>
    ) -> heed::Result<Option<Vec<(ItemId, f32)>>> {
        match self.item_vector(rtxn, item)? {
            Some(vector) => self.nns_by_vector(rtxn, &vector, count, search_k).map(Some),
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

        let capacity = count; // count.min(self.size).max(1)
        let search_k = search_k.unwrap_or(capacity * self.roots.len());

        let mut pq = BinaryHeap::with_capacity(capacity);
        for &item in &self.roots {
            pq.push(BinaryHeapItem { item, ord: OrderedFloat(f32::MAX) });
        }

        let mut nearest_neighbors = Vec::with_capacity(search_k);
        while !pq.is_empty() && nearest_neighbors.len() < search_k {
            if let Some(BinaryHeapItem { item: top_node_id, ord: top_node_margin }) = pq.pop() {
                match self.database.get(rtxn, &top_node_id)?.unwrap() {
                    Node::Leaf(_) => nearest_neighbors.push(top_node_id),
                    Node::Descendants(mut descendants) => {
                        nearest_neighbors.append(&mut descendants.descendants);
                    }
                    Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                        let margin = D::margin(&normal, vector);
                        // NOTE: Hamming has different logic to calculate margin
                        pq.push(BinaryHeapItem {
                            item: left,
                            ord: OrderedFloat(top_node_margin.0.min(margin)),
                        });
                        pq.push(BinaryHeapItem {
                            item: right,
                            ord: OrderedFloat(top_node_margin.0.min(-margin)),
                        });
                    }
                }
            }
        }

        nearest_neighbors.sort_unstable();

        let mut sorted_nns = BinaryHeap::with_capacity(nearest_neighbors.len());
        let mut nn_id_last = None;
        for nn_id in nearest_neighbors {
            if Some(nn_id) == nn_id_last {
                continue;
            }
            nn_id_last = Some(nn_id);
            let s = match self.database.get(rtxn, &nn_id)?.unwrap() {
                Node::Leaf(leaf) => leaf.vector,
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => continue,
            };
            sorted_nns.push(Reverse(BinaryHeapItem {
                item: nn_id,
                ord: OrderedFloat(D::non_normalized_distance(&s, vector)),
            }));
        }

        let final_result_capacity = count.min(sorted_nns.len());
        let mut output = Vec::with_capacity(final_result_capacity);
        while let Some(Reverse(heap_item)) = sorted_nns.pop() {
            if output.len() == final_result_capacity {
                break;
            }
            let BinaryHeapItem { item, ord: OrderedFloat(dist) } = heap_item;
            output.push((item, D::normalize_distance(dist)));
        }

        Ok(output)
    }
}

pub fn item_vector<D: Distance>(
    database: Database<BEU32, NodeCodec<D>>,
    rtxn: &RoTxn,
    item: ItemId,
) -> heed::Result<Option<Vec<f32>>> {
    match database.get(rtxn, &item)? {
        Some(Node::Leaf(Leaf { header: _, vector })) => Ok(Some(vector)),
        Some(Node::SplitPlaneNormal(_)) => Ok(None),
        Some(Node::Descendants(_)) => Ok(None),
        None => Ok(None),
    }
}
