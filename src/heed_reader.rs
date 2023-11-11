use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem::size_of;
use std::{iter, mem};

use bytemuck::{pod_collect_to_vec, Pod, Zeroable};
use byteorder::{ByteOrder, NativeEndian};
use heed::types::ByteSlice;
use heed::{BoxedError, BytesDecode, BytesEncode, Database, RoTxn, RwTxn};
use ordered_float::OrderedFloat;

use crate::distance::{
    cosine_distance_no_simd, dot_product_no_simd, euclidean_distance_no_simd,
    manhattan_distance_no_simd, minkowski_margin,
};
use crate::priority_queue::BinaryHeapItem;
use crate::DistanceType;

// An big endian-encoded u32
pub type BEU32 = heed::types::U32<heed::byteorder::BE>;

pub struct HeedReader {
    dimension: usize,
    // TODO make the distance a generic type
    distance_type: DistanceType,
    size: usize,
    max_descendants: usize,
    offset_before_children: usize,
    node_header_size: usize,
    database: Database<BEU32, ByteSlice>,
    roots: Vec<u32>,
}

impl HeedReader {
    pub fn new(
        rtxn: &RoTxn,
        database: Database<BEU32, ByteSlice>,
        dimensions: usize,
        distance_type: DistanceType,
    ) -> heed::Result<HeedReader> {
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

impl DistanceType {
    fn distance_no_norm(&self, v1: &[f32], v2: &[f32]) -> f32 {
        match self {
            DistanceType::Angular => cosine_distance_no_simd(v1, v2),
            DistanceType::Euclidean => euclidean_distance_no_simd(v1, v2),
            DistanceType::Manhattan => manhattan_distance_no_simd(v1, v2),
            DistanceType::Dot => -dot_product_no_simd(v1, v2),
        }
    }

    fn normalized_distance(&self, distance: f32) -> f32 {
        match self {
            DistanceType::Angular | DistanceType::Euclidean => distance.sqrt(),
            DistanceType::Dot => -distance,
            DistanceType::Manhattan => distance,
        }
    }
}

pub enum NodeCodec {}

impl<'a> BytesEncode<'a> for NodeCodec {
    type EItem = (NodeHeader, &'a [u8]);

    fn bytes_encode(item: &'a Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        todo!()
    }
}

impl<'a> BytesDecode<'a> for NodeCodec {
    type DItem = (NodeHeader, &'a [u8]);

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        Ok(NodeHeader::from_bytes(bytes, DistanceType::Angular))
    }
}

/*
 * We store a binary tree where each node has two things
 * - A vector associated with it
 * - Two children
 * All nodes occupy the same amount of memory
 * All nodes with n_descendants == 1 are leaf nodes.
 * A memory optimization is that for nodes with 2 <= n_descendants <= K,
 * we skip the vector. Instead we store a list of all descendants. K is
 * determined by the number of items that fits in the space of the vector.
 * For nodes with n_descendants == 1 the vector is a data point.
 * For nodes with n_descendants > K the vector is the normal of the split plane.
 * Note that we can't really do sizeof(node<T>) because we cheat and allocate
 * more memory to be able to fit the vector outside
 */
enum Node<'a> {
    Leaf(Leaf<'a>),
    Descendants(Descendants<'a>),
    SplitPlaneNormal(SplitPlaneNormal<'a>),
}

impl<'a> Node<'a> {
    fn from_bytes(
        bytes: &'a [u8],
        distance_type: DistanceType,
        max_descendants: usize,
    ) -> Node<'a> {
        let n_descendants = NativeEndian::read_u32(bytes);
        if n_descendants == 1 {
            let (header, vector_bytes) = NodeHeader::from_bytes(bytes, distance_type);
            Node::Leaf(Leaf { header, vector_bytes })
        } else if n_descendants as usize <= max_descendants {
            let offset = NodeHeaderAngular::offset_before_children();
            let length = n_descendants as usize * size_of::<u32>();
            let descendants_bytes = &bytes[offset..offset + length];
            Node::Descendants(Descendants { n_descendants, descendants_bytes })
        } else {
            let (header, normal_bytes) = NodeHeader::from_bytes(bytes, distance_type);
            Node::SplitPlaneNormal(SplitPlaneNormal { header, normal_bytes })
        }
    }

    fn n_descendants(&self) -> u32 {
        match self {
            Node::Leaf(leaf) => leaf.header.n_descendants(),
            Node::Descendants(Descendants { n_descendants, .. }) => *n_descendants,
            Node::SplitPlaneNormal(normal) => normal.header.n_descendants(),
        }
    }
}

struct Leaf<'a> {
    pub header: NodeHeader,
    vector_bytes: &'a [u8],
}

impl Leaf<'_> {
    fn vector(&self) -> Vec<f32> {
        let Leaf { header: _, vector_bytes } = self;
        pod_collect_to_vec(vector_bytes)
    }
}

struct Descendants<'a> {
    n_descendants: u32,
    descendants_bytes: &'a [u8],
}

impl<'a> Descendants<'a> {
    fn descendants_ids(&self) -> impl Iterator<Item = u32> + 'a {
        let Descendants { n_descendants: _, descendants_bytes: mut remaining } = self;
        iter::from_fn(move || {
            if remaining.is_empty() {
                None
            } else {
                let number = NativeEndian::read_u32(remaining);
                remaining = &remaining[size_of::<u32>()..];
                Some(number)
            }
        })
    }
}

struct SplitPlaneNormal<'a> {
    pub header: NodeHeader,
    normal_bytes: &'a [u8],
}

impl SplitPlaneNormal<'_> {
    fn normal(&self) -> Vec<f32> {
        let SplitPlaneNormal { header: _, normal_bytes } = self;
        pod_collect_to_vec(normal_bytes)
    }

    fn margin(&self, v2: &[f32]) -> f32 {
        let v1 = self.normal();
        match self.header {
            NodeHeader::Angular(_) => dot_product_no_simd(&v1, v2),
            NodeHeader::Minkowski(NodeHeaderMinkowski { bias, .. }) => {
                minkowski_margin(&v1, v2, bias)
            }
            NodeHeader::Dot(NodeHeaderDot { dot_factor, .. }) => {
                dot_product_no_simd(&v1, v2) + dot_factor.powi(2)
            }
        }
    }
}

#[repr(C)]
pub enum NodeHeader {
    Angular(NodeHeaderAngular),
    Minkowski(NodeHeaderMinkowski),
    Dot(NodeHeaderDot),
}

impl NodeHeader {
    pub fn from_bytes(bytes: &[u8], distance_type: DistanceType) -> (NodeHeader, &[u8]) {
        match distance_type {
            DistanceType::Angular => {
                let (header, remaining) = NodeHeaderAngular::read(bytes);
                (NodeHeader::Angular(header), remaining)
            }
            DistanceType::Euclidean | DistanceType::Manhattan => {
                let (header, remaining) = NodeHeaderMinkowski::read(bytes);
                (NodeHeader::Minkowski(header), remaining)
            }
            DistanceType::Dot => {
                let (header, remaining) = NodeHeaderDot::read(bytes);
                (NodeHeader::Dot(header), remaining)
            }
        }
    }

    pub fn n_descendants(&self) -> u32 {
        match self {
            NodeHeader::Angular(h) => h.n_descendants,
            NodeHeader::Minkowski(h) => h.n_descendants,
            NodeHeader::Dot(h) => h.n_descendants,
        }
    }

    pub fn children_id_slice(&self) -> [u32; 2] {
        match self {
            NodeHeader::Angular(h) => h.children,
            NodeHeader::Minkowski(h) => h.children,
            NodeHeader::Dot(h) => h.children,
        }
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderAngular {
    n_descendants: u32,
    children: [u32; 2],
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderMinkowski {
    n_descendants: u32,
    bias: f32,
    children: [u32; 2],
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderDot {
    n_descendants: u32,
    children: [u32; 2],
    dot_factor: f32,
}

impl NodeHeaderAngular {
    fn read(bytes: &[u8]) -> (NodeHeaderAngular, &[u8]) {
        let (left, right) = bytes.split_at(size_of::<Self>());
        let array: [u8; size_of::<Self>()] = left.try_into().unwrap();
        (bytemuck::cast(array), right)
    }

    pub const fn header_size() -> usize {
        size_of::<NodeHeaderAngular>()
    }

    pub const fn offset_before_children() -> usize {
        size_of::<u32>()
    }
}

impl NodeHeaderMinkowski {
    fn read(bytes: &[u8]) -> (NodeHeaderMinkowski, &[u8]) {
        let (left, right) = bytes.split_at(size_of::<Self>());
        let array: [u8; size_of::<Self>()] = left.try_into().unwrap();
        (bytemuck::cast(array), right)
    }

    pub const fn header_size() -> usize {
        size_of::<NodeHeaderMinkowski>()
    }

    pub const fn offset_before_children() -> usize {
        size_of::<u32>() + size_of::<f32>()
    }
}

impl NodeHeaderDot {
    fn read(bytes: &[u8]) -> (NodeHeaderDot, &[u8]) {
        let (left, right) = bytes.split_at(size_of::<Self>());
        let array: [u8; size_of::<Self>()] = left.try_into().unwrap();
        (bytemuck::cast(array), right)
    }

    pub const fn header_size() -> usize {
        size_of::<NodeHeaderDot>()
    }

    pub const fn offset_before_children() -> usize {
        size_of::<u32>()
    }
}
