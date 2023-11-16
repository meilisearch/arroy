use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem::size_of;
use std::{iter, mem};

use bytemuck::{bytes_of, cast_slice, pod_collect_to_vec, Pod, Zeroable};
use byteorder::{ByteOrder, NativeEndian};
use heed::types::ByteSlice;
use heed::{Database, RoTxn, RwTxn};
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::distance::{
    cosine_distance_no_simd, dot_product_no_simd, euclidean_distance_no_simd,
    manhattan_distance_no_simd, minkowski_margin,
};
use crate::priority_queue::BinaryHeapItem;
use crate::DistanceType;

// An big endian-encoded u32
pub type BEU32 = heed::types::U32<heed::byteorder::BE>;

pub struct HeedReader<D> {
    dimension: usize,
    size: usize,
    max_descendants: usize,
    offset_before_children: usize,
    node_header_size: usize,
    database: Database<BEU32, ByteSlice>,
    roots: Vec<u32>,
}

impl<D: Distance> HeedReader<D> {
    pub fn new(
        rtxn: &RoTxn,
        database: Database<BEU32, ByteSlice>,
        dimensions: usize,
    ) -> heed::Result<HeedReader<D>> {
        let (offset_before_children, node_header_size, max_descendants) = match distance_type {
            DistanceType::Angular => (4, NodeHeaderAngular::header_size(), dimensions + 2),
            DistanceType::Euclidean | DistanceType::Manhattan => {
                (8, NodeHeaderMinkowski::header_size(), dimensions + 2)
            }
            // DistanceType::Hamming => (4, 12),
            DistanceType::Dot => (4, NodeHeaderDot::header_size(), dimensions + 3),
        };

        let mut roots = Vec::new();
        let mut m = None;
        for result in database.rev_iter(rtxn)? {
            let (id, node_bytes) = result?;
            let n_descendants =
                Node::from_bytes(node_bytes, distance_type, max_descendants).n_descendants();
            if m.map_or(true, |m| n_descendants == m) {
                roots.push(id);
                m = Some(n_descendants);
            } else {
                break;
            }
        }

        // hacky fix: since the last root precedes the copy of all roots, delete it
        if roots.len() > 1 {
            if let Some((first_id, last_id)) = roots.first().zip(roots.last()) {
                let first_node_bytes = database.get(rtxn, first_id)?.unwrap();
                let last_node_bytes = database.get(rtxn, last_id)?.unwrap();
                let first = Node::from_bytes(first_node_bytes, distance_type, max_descendants);
                let last = Node::from_bytes(last_node_bytes, distance_type, max_descendants);

                let first_children = match first {
                    Node::Leaf(leaf) => leaf.header.children_id_slice(),
                    Node::SplitPlaneNormal(normal) => normal.header.children_id_slice(),
                    Node::Descendants(_) => panic!("invalid root node"),
                };

                let last_children = match last {
                    Node::Leaf(leaf) => leaf.header.children_id_slice(),
                    Node::SplitPlaneNormal(normal) => normal.header.children_id_slice(),
                    Node::Descendants(_) => panic!("invalid root node"),
                };

                if first_children[0] == last_children[0] {
                    roots.pop();
                }
            }
        }

        Ok(HeedReader {
            dimension: dimensions,
            distance_type,
            offset_before_children,
            node_header_size,
            max_descendants,
            database,
            roots,
            size: m.unwrap() as usize,
        })
    }

    pub fn load_from_tree(
        wtxn: &mut RwTxn,
        database: Database<BEU32, ByteSlice>,
        dimensions: usize,
        distance_type: DistanceType,
        tree_bytes: &[u8],
    ) -> heed::Result<()> {
        let node_header_size = match distance_type {
            DistanceType::Angular => NodeHeaderAngular::header_size(),
            DistanceType::Euclidean | DistanceType::Manhattan => NodeHeaderMinkowski::header_size(),
            // DistanceType::Hamming => (4, 12),
            DistanceType::Dot => NodeHeaderDot::header_size(),
        };

        database.clear(wtxn)?;

        let flags = heed::PutFlags::APPEND;
        let node_size = node_header_size + (size_of::<f32>() * dimensions);
        for (i, chunk) in tree_bytes.chunks(node_size).enumerate() {
            let i = i.try_into().unwrap();
            assert_eq!(chunk.len(), node_size);
            database.put_with_flags(wtxn, flags, &i, chunk)?;
        }

        Ok(())
    }

    // TODO we must unbuild to do that safely
    pub fn add_item(&self, wtxn: &mut RwTxn, item: u32, vector: &[f32]) -> heed::Result<()> {
        assert_eq!(
            vector.len(),
            self.dimension,
            "invalid vector dimensions, provided {} but expected {}",
            vector.len(),
            self.dimension
        );

        // TODO use with_capacity
        let mut vec = Vec::new();
        let leaf = Leaf::from_vector(self.distance_type, vector);
        self.database.put(wtxn, &item, leaf.append_to_vec(&mut vec))
    }

    pub fn build<R: Rng>(&self, wtxn: &mut RwTxn, mut rng: R, n_trees: usize) -> heed::Result<()> {
        assert!(n_trees >= 1, "please provide a n_trees >= 1 for now");

        todo!("preprocess stuff");
        todo!("n_nodes = n_items (no tree nodes yet)");

        let mut thread_roots = Vec::new();
        while thread_roots.len() < n_trees {
            // TODO use a roaring bitmap?
            let mut indices = Vec::new();
            for result in self.database.iter(wtxn)? {
                let (i, _) = result?;
                indices.push(i);
            }

            let tree_root_id = self.make_tree(wtxn, &indices, true)?;
            thread_roots.push(tree_root_id);
        }

        todo!("copy roots in the database");
        todo!("postprocess stuff");

        Ok(())
    }

    fn make_tree(&self, wtxn: &mut RwTxn, indices: &[u32], is_root: bool) -> heed::Result<u32> {
        // The basic rule is that if we have <= _K items, then it's a leaf node, otherwise it's a split node.
        // There's some regrettable complications caused by the problem that root nodes have to be "special":
        // 1. We identify root nodes by the arguable logic that _n_items == n->n_descendants, regardless of how many descendants they actually have
        // 2. Root nodes with only 1 child need to be a "dummy" parent
        // 3. Due to the _n_items "hack", we need to be careful with the cases where _n_items <= _K or _n_items > _K
        if indices.len() == 1 && !is_root {
            return Ok(indices[0]);
        }

        // if (indices.size() <= (size_t)_K && (!is_root || (size_t)_n_items <= (size_t)_K || indices.size() == 1)) {
        let n_items: usize = 0; // We must count the number of items
        if indices.len() <= self.max_descendants
            && (!is_root || n_items <= self.max_descendants || indices.len() == 1)
        {
            let item_id: u32 = self.database.len(wtxn)?.try_into().unwrap();

            // threaded_build_policy.lock_n_nodes();
            // _allocate_size(_n_nodes + 1, threaded_build_policy);
            // S item = _n_nodes++;
            // threaded_build_policy.unlock_n_nodes();

            // threaded_build_policy.lock_shared_nodes();
            // Node* m = _get(item);
            // m->n_descendants = is_root ? _n_items : (S)indices.size();

            // We identify root nodes by the arguable logic that _n_items == n->n_descendants,
            // regardless of how many descendants they actually have
            let n_descendants = if is_root { n_items } else { indices.len() } as u32;
            let node = Descendants { n_descendants, descendants_bytes: cast_slice(indices) };

            let mut vec = Vec::new();
            self.database.put(wtxn, &item_id, node.append_to_vec(&mut vec))?;
            return Ok(item_id);
        }

        let mut children = Vec::new();
        for item_id in indices {
            let node_bytes = self.database.get(wtxn, item_id)?.unwrap();
            match Node::from_bytes(node_bytes, self.distance_type, self.max_descendants) {
                Node::Leaf(leaf) => children.push(leaf),
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => {
                    panic!("a children must be a leaf ???")
                }
            }
        }

        let mut children_indices[2] = [Vec::new(); 2];
        let mut m = None;

        for _ in 0..3 {
            children_indices[0].clear();
            children_indices[1].clear();
            D::create_split(children, _f, _s, _random, m);

            for item_id in indices {
                let node_bytes = self.database.get(wtxn, item_id)?.unwrap();
                let n = match Node::from_bytes(node_bytes, self.distance_type, self.max_descendants) {
                    Node::Leaf(leaf) => leaf,
                    Node::Descendants(_) | Node::SplitPlaneNormal(_) => {
                        panic!("a children must be a leaf ???")
                    }
                };

                bool side = D::side(m, n, _f, _random);
                children_indices[side].push(item_id);
            }

            if _split_imbalance(children_indices[0], children_indices[1]) < 0.95 {
                break;
            }
        }

        return Ok(0);
    }

    pub fn item_vector(&self, rtxn: &RoTxn, item: u32) -> heed::Result<Option<Vec<f32>>> {
        match self.database.get(rtxn, &item)? {
            Some(node_bytes) => {
                match Node::from_bytes(node_bytes, self.distance_type, self.max_descendants) {
                    Node::Leaf(leaf) => Ok(Some(leaf.vector())),
                    Node::Descendants(_) | Node::SplitPlaneNormal(_) => Ok(None),
                }
            }
            None => Ok(None),
        }
    }

    pub fn nns_by_item(
        &self,
        rtxn: &RoTxn,
        item: u32,
        n_results: usize,
        search_k: Option<usize>,
    ) -> heed::Result<Option<Vec<(u32, f32)>>> {
        match self.item_vector(rtxn, item)? {
            Some(vector) => self.nns_by_vector(rtxn, &vector, n_results, search_k).map(Some),
            None => Ok(None),
        }
    }

    pub fn nns_by_vector(
        &self,
        rtxn: &RoTxn,
        query_vector: &[f32],
        n_results: usize,
        search_k: Option<usize>,
        // should_include_distance: bool,
    ) -> heed::Result<Vec<(u32, f32)>> {
        assert_eq!(
            query_vector.len(),
            self.dimension,
            "invalid vector dimensions, provided {} but expected {}",
            query_vector.len(),
            self.dimension
        );

        let result_capacity = n_results.min(self.size).max(1);
        let search_k = search_k.unwrap_or(result_capacity * self.roots.len());

        let mut pq = BinaryHeap::with_capacity(result_capacity);
        for &item in &self.roots {
            pq.push(BinaryHeapItem { item, ord: OrderedFloat(f32::MAX) });
        }

        let mut nearest_neighbors = Vec::with_capacity(search_k);
        while !pq.is_empty() && nearest_neighbors.len() < search_k {
            if let Some(BinaryHeapItem { item: top_node_id, ord: top_node_margin }) = pq.pop() {
                let node_bytes = self.database.get(rtxn, &top_node_id)?.unwrap();
                match Node::from_bytes(node_bytes, self.distance_type, self.max_descendants) {
                    Node::Leaf(_) => nearest_neighbors.push(top_node_id),
                    Node::Descendants(descendants) => {
                        nearest_neighbors.extend(descendants.descendants_ids())
                    }
                    Node::SplitPlaneNormal(normal) => {
                        let margin = normal.margin(query_vector);
                        let [child_0, child_1] = normal.header.children_id_slice();
                        // NOTE: Hamming has different logic to calculate margin
                        pq.push(BinaryHeapItem {
                            item: child_1,
                            ord: OrderedFloat(top_node_margin.0.min(margin)),
                        });
                        pq.push(BinaryHeapItem {
                            item: child_0,
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
            let bytes = self.database.get(rtxn, &nn_id)?.unwrap();
            let s = match Node::from_bytes(bytes, self.distance_type, self.max_descendants) {
                Node::Leaf(leaf) => leaf.vector(),
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => continue,
            };
            // TODO use a tuple to simplify things
            sorted_nns.push(Reverse(BinaryHeapItem {
                item: nn_id,
                ord: OrderedFloat(self.distance_type.distance_no_norm(&s, query_vector)),
            }));
        }

        let final_result_capacity = n_results.min(sorted_nns.len());
        let mut output = Vec::with_capacity(final_result_capacity);
        while let Some(Reverse(heap_item)) = sorted_nns.pop() {
            if output.len() == final_result_capacity {
                break;
            }
            let BinaryHeapItem { item, ord: OrderedFloat(dist) } = heap_item;
            output.push((item, self.distance_type.normalized_distance(dist)));
        }

        Ok(output)
    }
}
