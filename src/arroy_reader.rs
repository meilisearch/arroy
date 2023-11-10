use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem::size_of;
use std::{fmt, iter};

use bytemuck::allocation::pod_collect_to_vec;
use byteorder::{ByteOrder, NativeEndian};
use ordered_float::OrderedFloat;

use crate::distance::{
    cosine_distance_no_simd, dot_product_no_simd, euclidean_distance_no_simd,
    manhattan_distance_no_simd, minkowski_margin,
};
use crate::node::{Node, NodeHeaderAngular, NodeHeaderDot, NodeHeaderMinkowski};
use crate::priority_queue::BinaryHeapItem;

pub struct ArroyReader<'a> {
    pub dimension: usize,
    pub distance_type: DistanceType,
    pub node_size: usize,
    pub size: usize,
    pub(crate) max_descendants: i32,
    pub(crate) offset_before_children: usize,
    pub(crate) node_header_size: usize,
    pub(crate) storage: &'a [u8],
    pub(crate) roots: Vec<usize>,
}

impl<'a> ArroyReader<'a> {
    pub fn new(storage: &'a [u8], dimension: usize, distance_type: DistanceType) -> ArroyReader {
        let (offset_before_children, node_header_size, max_descendants) = match distance_type {
            DistanceType::Angular => (4, NodeHeaderAngular::header_size(), dimension + 2),
            DistanceType::Euclidean | DistanceType::Manhattan => {
                (8, NodeHeaderMinkowski::header_size(), dimension + 2)
            }
            // DistanceType::Hamming => (4, 12),
            DistanceType::Dot => (4, NodeHeaderDot::header_size(), dimension + 3),
        };

        let index_size = storage.len();
        let node_size = node_header_size + (size_of::<f32>() * dimension);

        let mut roots = Vec::new();
        let mut m = None;
        let mut i = index_size - node_size;
        loop {
            let n_descendants = NativeEndian::read_i32(&storage[i..]);
            if m.map_or(true, |m| n_descendants == m) {
                roots.push(i / node_size);
                m = Some(n_descendants);
            } else {
                break;
            }
            match i.checked_sub(node_size) {
                Some(new) => i = new,
                None => break,
            }
        }

        // hacky fix: since the last root precedes the copy of all roots, delete it
        if roots.len() > 1 {
            if let Some((first, last)) = roots.first().zip(roots.last()) {
                let first_descendant =
                    get_nth_descendant_id(storage, first * node_size, offset_before_children, 0);
                let last_descendant =
                    get_nth_descendant_id(storage, last * node_size, offset_before_children, 0);
                if first_descendant == last_descendant {
                    roots.pop();
                }
            }
        }

        ArroyReader {
            dimension,
            distance_type,
            offset_before_children,
            node_header_size,
            max_descendants: max_descendants as i32,
            node_size,
            storage,
            roots,
            size: m.unwrap() as usize,
        }
    }

    pub fn item_vector(&self, item_index: usize) -> Option<Vec<f32>> {
        if item_index < self.size {
            let node_offset = item_index * self.node_size;
            Some(self.node_slice_with_offset(node_offset))
        } else {
            None
        }
    }

    pub fn nns_by_vector(
        &self,
        query_vector: &[f32],
        n_results: usize,
        search_k: Option<usize>,
        // should_include_distance: bool,
    ) -> Vec<(usize, f32)> {
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
            pq.push(BinaryHeapItem { item: item as i32, ord: OrderedFloat(f32::MAX) });
        }

        let mut nearest_neighbors = Vec::with_capacity(search_k);
        while !pq.is_empty() && nearest_neighbors.len() < search_k {
            if let Some(BinaryHeapItem { item: top_node_id_i32, ord: top_node_margin }) = pq.pop() {
                let top_node_id = top_node_id_i32 as usize;
                let top_node = Node::new_with_id(
                    top_node_id,
                    self.node_size,
                    self.distance_type,
                    self.storage,
                );
                let top_node_header = top_node.header;
                let top_node_offset = top_node.offset;
                let n_descendants = top_node_header.get_n_descendant();
                if n_descendants == 1 && top_node_id < self.size {
                    nearest_neighbors.push(top_node_id_i32);
                } else if n_descendants <= self.max_descendants {
                    let children_ids = self.descendant_ids(top_node_offset, n_descendants as usize);
                    nearest_neighbors.extend(children_ids);
                } else {
                    let v = self.node_slice_with_offset(top_node_offset);
                    let margin = self.margin(&v, query_vector, top_node_offset);
                    let [child_0, child_1] = top_node_header.get_children_id_slice();
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
        nearest_neighbors.sort_unstable();
        let mut sorted_nns = BinaryHeap::with_capacity(nearest_neighbors.len());
        let mut nn_id_last = -1;
        for nn_id in nearest_neighbors {
            if nn_id == nn_id_last {
                continue;
            }
            nn_id_last = nn_id;
            let node =
                Node::new_with_id(nn_id as usize, self.node_size, self.distance_type, self.storage);
            let n_descendants = node.header.get_n_descendant();
            if n_descendants != 1 {
                continue;
            }

            let s = self.node_slice_with_offset(nn_id as usize * self.node_size);
            sorted_nns.push(Reverse(BinaryHeapItem {
                item: nn_id,
                ord: OrderedFloat(self.distance_no_norm(&s, query_vector)),
            }));
        }

        let final_result_capacity = n_results.min(sorted_nns.len());
        let mut output = Vec::with_capacity(final_result_capacity);
        for _ in 0..final_result_capacity {
            match sorted_nns.pop() {
                Some(Reverse(BinaryHeapItem { item, ord: OrderedFloat(dist) })) => {
                    output.push((item as usize, self.normalized_distance(dist)));
                }
                None => break,
            }
        }

        output
    }

    pub fn nns_by_item(
        &self,
        item: usize,
        n_results: usize,
        search_k: Option<usize>,
    ) -> Option<Vec<(usize, f32)>> {
        let query_vector = self.item_vector(item)?;
        Some(self.nns_by_vector(&query_vector, n_results, search_k))
    }

    fn descendant_ids(&self, node_offset: usize, n: usize) -> impl Iterator<Item = i32> + 'a {
        let offset = node_offset + self.offset_before_children;
        let mut remaining = &self.storage[offset..offset + n * size_of::<i32>()];
        iter::from_fn(move || {
            if remaining.is_empty() {
                None
            } else {
                let number = NativeEndian::read_i32(remaining);
                remaining = &remaining[size_of::<i32>()..];
                Some(number)
            }
        })
    }

    fn node_slice_with_offset(&self, node_offset: usize) -> Vec<f32> {
        let offset = node_offset + self.node_header_size;
        let size = self.dimension * size_of::<f32>();
        pod_collect_to_vec(&self.storage[offset..offset + size])
    }

    fn margin(&self, v1: &[f32], v2: &[f32], node_offset: usize) -> f32 {
        match self.distance_type {
            DistanceType::Angular => dot_product_no_simd(v1, v2),
            DistanceType::Euclidean | DistanceType::Manhattan => {
                let bias = NativeEndian::read_f32(&self.storage[node_offset + 4..]);
                minkowski_margin(v1, v2, bias)
            }
            DistanceType::Dot => {
                let dot = NativeEndian::read_f32(&self.storage[node_offset + 12..]).powi(2);
                dot_product_no_simd(v1, v2) + dot
            }
        }
    }

    fn distance_no_norm(&self, v1: &[f32], v2: &[f32]) -> f32 {
        match self.distance_type {
            DistanceType::Angular => cosine_distance_no_simd(v1, v2),
            DistanceType::Euclidean => euclidean_distance_no_simd(v1, v2),
            DistanceType::Manhattan => manhattan_distance_no_simd(v1, v2),
            DistanceType::Dot => -dot_product_no_simd(v1, v2),
        }
    }

    fn normalized_distance(&self, distance: f32) -> f32 {
        match self.distance_type {
            DistanceType::Angular | DistanceType::Euclidean => distance.sqrt(),
            DistanceType::Dot => -distance,
            DistanceType::Manhattan => distance,
        }
    }
}

impl fmt::Debug for ArroyReader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arroy")
            .field("dimension", &self.dimension)
            .field("distance_type", &self.distance_type)
            .field("node_size", &self.node_size)
            .field("size", &self.size)
            .field("max_descendants", &self.max_descendants)
            .field("offset_before_children", &self.offset_before_children)
            .field("node_header_size", &self.node_header_size)
            .field("roots", &self.roots)
            .finish()
    }
}

pub(crate) fn get_nth_descendant_id(
    storage: &[u8],
    node_offset: usize,
    offset_before_children: usize,
    n: usize,
) -> usize {
    let child_offset = node_offset + offset_before_children + n * size_of::<i32>();
    NativeEndian::read_i32(&storage[child_offset..]) as usize
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
#[repr(u8)]
pub enum DistanceType {
    Angular = 0,
    Euclidean = 1,
    Manhattan = 2,
    // Hamming = 3,
    Dot = 4,
}
