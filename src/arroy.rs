use std::borrow::Cow;
use std::error::Error;
use std::mem::size_of;
use std::{fmt, marker};

use bytemuck::checked::cast_slice;
use bytemuck::{bytes_of, pod_collect_to_vec, pod_read_unaligned, Pod, Zeroable};
use byteorder::{BigEndian, ByteOrder};
use heed::types::{ByteSlice, DecodeIgnore};
use heed::{BytesDecode, BytesEncode, Database, RoTxn, RwTxn};
use rand::seq::SliceRandom;
use rand::Rng;

use crate::distance::dot_product_no_simd;

/// An big endian-encoded u32.
pub type BEU32 = heed::types::U32<heed::byteorder::BE>;

/// An external item id.
pub type ItemId = u32;

/// An internal node id.
type NodeId = u32;

// TODO use a "metadata" key to store and check the dimensions and distance type
pub struct Reader<D: Distance> {
    database: heed::Database<BEU32, NodeCodec<D>>,
    dimensions: usize,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance> Reader<D> {
    pub fn open<U>(dimensions: usize, database: Database<BEU32, U>) -> Reader<D> {
        Reader { database: database.remap_data_type(), dimensions, _marker: marker::PhantomData }
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

        todo!()
    }
}

pub struct Writer<D: Distance> {
    database: heed::Database<BEU32, NodeCodec<D>>,
    dimensions: usize,
    // non-initiliazed until build is called.
    n_items: usize,
    roots: Vec<NodeId>,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance> Writer<D> {
    pub fn prepare<U>(
        wtxn: &mut RwTxn,
        dimensions: usize,
        database: Database<BEU32, U>,
    ) -> heed::Result<Writer<D>> {
        let database = database.remap_data_type();
        clear_tree_nodes(wtxn, database)?;
        Ok(Writer {
            database,
            dimensions,
            n_items: 0,
            roots: Vec::new(),
            _marker: marker::PhantomData,
        })
    }

    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> heed::Result<Option<Vec<f32>>> {
        item_vector(self.database, rtxn, item)
    }

    pub fn add_item(&self, wtxn: &mut RwTxn, item: ItemId, vector: &[f32]) -> heed::Result<()> {
        // TODO make this not an assert
        assert_eq!(
            vector.len(),
            self.dimensions,
            "invalid vector dimensions, provided {} but expected {}",
            vector.len(),
            self.dimensions
        );

        // TODO find a way to not allocate the vector
        let leaf = Leaf { header: D::new_header(vector), vector: vector.to_vec() };
        self.database.put(wtxn, &item, &Node::Leaf(leaf))
    }

    pub fn del_item(&self, wtxn: &mut RwTxn, item: ItemId) -> heed::Result<bool> {
        todo!()
    }

    pub fn clear(&self, wtxn: &mut RwTxn) -> heed::Result<()> {
        self.database.clear(wtxn)
    }

    pub fn build<R: Rng>(
        mut self,
        wtxn: &mut RwTxn,
        mut rng: R,
        n_trees: Option<usize>,
    ) -> heed::Result<Reader<D>> {
        // D::template preprocess<T, S, Node>(_nodes, _s, _n_items, _f);

        self.n_items = self.database.len(wtxn)? as usize;
        let last_item_id = self.last_node_id(wtxn)?;

        let mut thread_roots = Vec::new();
        loop {
            match n_trees {
                Some(n_trees) if thread_roots.len() >= n_trees => break,
                None if self.database.len(wtxn)? >= 2 * self.n_items as u64 => break,
                _ => (),
            }

            let mut indices = Vec::new();
            // Only fetch the item's ids, not the tree nodes ones
            for result in self.database.remap_data_type::<DecodeIgnore>().iter(wtxn)? {
                let (i, _) = result?;
                if last_item_id.map_or(true, |last| i > last) {
                    break;
                }
                indices.push(i);
            }

            let tree_root_id = self.make_tree(wtxn, indices, true, &mut rng)?;
            thread_roots.push(tree_root_id);
        }

        self.roots.append(&mut thread_roots);

        // Also, copy the roots into the highest key of the database (u32::MAX).
        // This way we can load them faster without reading the whole database.
        match self.database.get(wtxn, &u32::MAX)? {
            Some(_) => panic!("The database is full. We cannot write the root nodes ids"),
            None => {
                self.database.remap_data_type::<ByteSlice>().put(
                    wtxn,
                    &u32::MAX,
                    cast_slice(self.roots.as_slice()),
                )?;
            }
        }

        // D::template postprocess<T, S, Node>(_nodes, _s, _n_items, _f);

        Ok(Reader {
            database: self.database,
            dimensions: self.dimensions,
            _marker: marker::PhantomData,
        })
    }

    /// Creates a tree of nodes from the items the user provided
    /// and generates descendants, split normal and root nodes.
    fn make_tree<R: Rng>(
        &self,
        wtxn: &mut RwTxn,
        indices: Vec<u32>,
        is_root: bool,
        rng: &mut R,
    ) -> heed::Result<NodeId> {
        // we simplify the max descendants (_K) thing by considering
        // that we can fit as much descendants as the number of dimensions
        let max_descendants = self.dimensions;

        if indices.len() == 1 && !is_root {
            return Ok(indices[0]);
        }

        if indices.len() <= max_descendants
            && (!is_root || self.n_items <= max_descendants || indices.len() == 1)
        {
            let item_id = match self.last_node_id(wtxn)? {
                Some(last_id) => last_id.checked_add(1).unwrap(),
                None => 0,
            };

            let item = Node::Descendants(Descendants { descendants: indices });
            self.database.put(wtxn, &item_id, &item)?;
            return Ok(item_id);
        }

        let mut children = Vec::new();
        for node_id in &indices {
            let node = self.database.get(wtxn, node_id)?.unwrap();
            let leaf = node.leaf().unwrap();
            children.push(leaf);
        }

        let mut children_left = Vec::new();
        let mut children_right = Vec::new();
        let mut remaining_attempts = 3;

        let mut m = loop {
            children_left.clear();
            children_right.clear();

            let m = D::create_split(&children, rng);
            for (&node_id, node) in indices.iter().zip(&children) {
                match D::side(&m, node, rng) {
                    Side::Left => children_left.push(node_id),
                    Side::Right => children_right.push(node_id),
                }
            }

            if split_imbalance(children_left.len(), children_right.len()) < 0.95
                || remaining_attempts == 0
            {
                break m;
            }

            remaining_attempts -= 1;
        };

        // If we didn't find a hyperplane, just randomize sides as a last option
        // and set the split plane to zero as a dummy plane.
        while split_imbalance(children_left.len(), children_right.len()) > 0.99 {
            children_left.clear();
            children_right.clear();

            m.normal.fill(0.0);

            for &node_id in &indices {
                match Side::random(rng) {
                    Side::Left => children_left.push(node_id),
                    Side::Right => children_right.push(node_id),
                }
            }
        }

        // TODO make sure to run _make_tree for the smallest child first (for cache locality)

        m.left = self.make_tree(wtxn, children_left, false, rng)?;
        m.right = self.make_tree(wtxn, children_right, false, rng)?;

        let new_node_id = match self.last_node_id(wtxn)? {
            Some(last_id) => last_id.checked_add(1).unwrap(),
            None => 0,
        };

        self.database.put(wtxn, &new_node_id, &Node::SplitPlaneNormal(m))?;
        Ok(new_node_id)
    }

    fn last_node_id(&self, rtxn: &RoTxn) -> heed::Result<Option<NodeId>> {
        match self.database.remap_data_type::<DecodeIgnore>().last(rtxn)? {
            Some((last_id, _)) => Ok(Some(last_id)),
            None => Ok(None),
        }
    }
}

/// Clears everything but the leafs nodes (items).
/// Starts from the last node and stops at the first leaf.
fn clear_tree_nodes<D: Distance>(
    wtxn: &mut RwTxn,
    database: Database<BEU32, NodeCodec<D>>,
) -> heed::Result<()> {
    database.delete(wtxn, &u32::MAX)?;
    let mut cursor = database.rev_iter_mut(wtxn)?;
    while let Some((_id, node)) = cursor.next().transpose()? {
        if node.leaf().is_none() {
            unsafe { cursor.del_current()? };
        } else {
            break;
        }
    }
    Ok(())
}

fn split_imbalance(left_indices_len: usize, right_indices_len: usize) -> f64 {
    let ls = left_indices_len as f64;
    let rs = right_indices_len as f64;
    let f = ls / (ls + rs + f64::EPSILON); // Avoid 0/0
    f.max(1.0 - f)
}

fn item_vector<D: Distance>(
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

fn two_means<D: Distance, R: Rng>(rng: &mut R, leafs: &[Leaf<D>], cosine: bool) -> [Leaf<D>; 2] {
    // This algorithm is a huge heuristic. Empirically it works really well, but I
    // can't motivate it well. The basic idea is to keep two centroids and assign
    // points to either one of them. We weight each centroid by the number of points
    // assigned to it, so to balance it.

    const ITERATION_STEPS: usize = 200;

    let mut random_nodes = leafs.choose_multiple(rng, 2);
    let mut leaf_p = random_nodes.next().unwrap().clone();
    let mut leaf_q = random_nodes.next().unwrap().clone();

    if cosine {
        D::normalize(&mut leaf_p);
        D::normalize(&mut leaf_q);
    }

    D::init(&mut leaf_p);
    D::init(&mut leaf_q);

    let mut ic = 1.0;
    let mut jc = 1.0;
    for _ in 0..ITERATION_STEPS {
        let node_k = leafs.choose(rng).unwrap();
        let di = ic * D::distance(&leaf_p, node_k);
        let dj = jc * D::distance(&leaf_q, node_k);
        let norm = if cosine { D::norm(&node_k.vector) } else { 1.0 };
        if norm.is_nan() || norm <= 0.0 {
            continue;
        }
        if di < dj {
            Distance::update_mean(&mut leaf_p, node_k, norm, ic);
            Distance::init(&mut leaf_p);
            ic += 1.0;
        } else if dj < di {
            Distance::update_mean(&mut leaf_q, node_k, norm, jc);
            Distance::init(&mut leaf_q);
            jc += 1.0;
        }
    }

    [leaf_p, leaf_q]
}

#[derive(Debug, Copy, Clone)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    fn random<R: Rng>(rng: &mut R) -> Side {
        if rng.gen() {
            Side::Left
        } else {
            Side::Right
        }
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
#[derive(Debug, Clone)]
pub enum Node<D: Distance> {
    Leaf(Leaf<D>),
    Descendants(Descendants),
    SplitPlaneNormal(SplitPlaneNormal),
}

const LEAF_TAG: u8 = 0;
const DESCENDANTS_TAG: u8 = 1;
const SPLIT_PLANE_NORMAL_TAG: u8 = 2;

impl<D: Distance> Node<D> {
    fn leaf(self) -> Option<Leaf<D>> {
        if let Node::Leaf(leaf) = self {
            Some(leaf)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Leaf<D: Distance> {
    pub header: D::Header,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Descendants {
    descendants: Vec<NodeId>,
}

#[derive(Debug, Clone)]
pub struct SplitPlaneNormal {
    pub normal: Vec<f32>,
    pub left: NodeId,
    pub right: NodeId,
}

pub struct NodeCodec<D>(D);

impl<'a, D: Distance + 'a> BytesEncode<'a> for NodeCodec<D> {
    type EItem = Node<D>;

    fn bytes_encode(item: &Self::EItem) -> Result<Cow<'a, [u8]>, Box<dyn Error + Send + Sync>> {
        let mut bytes = Vec::new();
        match item {
            Node::Leaf(Leaf { header, vector }) => {
                bytes.push(LEAF_TAG);
                bytes.extend_from_slice(bytes_of(header));
                bytes.extend_from_slice(cast_slice(vector));
            }
            Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                bytes.push(SPLIT_PLANE_NORMAL_TAG);
                bytes.extend_from_slice(&left.to_be_bytes());
                bytes.extend_from_slice(&right.to_be_bytes());
                bytes.extend_from_slice(cast_slice(normal));
            }
            Node::Descendants(Descendants { descendants }) => {
                bytes.push(DESCENDANTS_TAG);
                bytes.extend_from_slice(cast_slice(descendants));
            }
        }
        Ok(Cow::Owned(bytes))
    }
}

impl<'a, D: Distance + 'a> BytesDecode<'a> for NodeCodec<D> {
    type DItem = Node<D>;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, Box<dyn Error + Send + Sync>> {
        match bytes {
            [LEAF_TAG, bytes @ ..] => {
                let (header_bytes, remaining) = bytes.split_at(size_of::<D::Header>());
                let header = pod_read_unaligned(header_bytes);
                let vector = pod_collect_to_vec(remaining);
                Ok(Node::Leaf(Leaf { header, vector }))
            }
            [SPLIT_PLANE_NORMAL_TAG, bytes @ ..] => {
                let left = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                let right = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                Ok(Node::SplitPlaneNormal(SplitPlaneNormal {
                    normal: pod_collect_to_vec(bytes),
                    left,
                    right,
                }))
            }
            [DESCENDANTS_TAG, bytes @ ..] => {
                Ok(Node::Descendants(Descendants { descendants: pod_collect_to_vec(bytes) }))
            }
            unknown => panic!("What the fuck is an {unknown:?}"),
        }
    }
}

pub trait Distance: Sized + Clone + fmt::Debug {
    type Header: Pod + Zeroable + fmt::Debug;

    fn name() -> &'static str;
    fn new_header(vector: &[f32]) -> Self::Header;
    fn distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32;
    fn norm(v: &[f32]) -> f32;
    fn normalize(node: &mut Leaf<Self>);
    fn init(node: &mut Leaf<Self>);
    fn update_mean(mean: &mut Leaf<Self>, new_node: &Leaf<Self>, norm: f32, c: f32);
    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> SplitPlaneNormal;
    fn side<R: Rng>(plane: &SplitPlaneNormal, node: &Leaf<Self>, rng: &mut R) -> Side;
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderAngular {
    norm: f32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderMinkowski {
    bias: f32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderDot {
    dot_factor: f32,
}

#[derive(Debug, Clone)]
pub enum Angular {}

impl Distance for Angular {
    type Header = NodeHeaderAngular;

    fn name() -> &'static str {
        "angular"
    }

    fn new_header(vector: &[f32]) -> Self::Header {
        NodeHeaderAngular { norm: Self::norm(vector) }
    }

    fn distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        // want to calculate (a/|a| - b/|b|)^2
        // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
        // = 2 - 2cos
        let pq = dot_product_no_simd(&p.vector, &q.vector);
        let ppqq = p.header.norm * q.header.norm;
        if ppqq > 0.0 {
            2.0 - 2.0 * pq / ppqq.sqrt()
        } else {
            2.0 // cos is 0
        }
    }

    fn norm(v: &[f32]) -> f32 {
        dot_product_no_simd(v, v).sqrt()
    }

    fn normalize(node: &mut Leaf<Self>) {
        let norm = Self::norm(&node.vector);
        if norm > 0.0 {
            node.vector.iter_mut().for_each(|x| *x /= norm);
        }
    }

    fn init(node: &mut Leaf<Self>) {
        node.header.norm = dot_product_no_simd(&node.vector, &node.vector);
    }

    fn update_mean(mean: &mut Leaf<Self>, new_node: &Leaf<Self>, norm: f32, c: f32) {
        mean.vector
            .iter_mut()
            .zip(&new_node.vector)
            .for_each(|(x, n)| *x = (*x * c + *n / norm) / (c + 1.0));
    }

    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> SplitPlaneNormal {
        let [node_p, node_q] = two_means(rng, children, true);
        let vector = node_p.vector.iter().zip(node_q.vector.iter()).map(|(&p, &q)| p - q).collect();
        let mut normal = Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector };
        Self::normalize(&mut normal);
        // TODO we are returning invalid left and rights
        SplitPlaneNormal { normal: normal.vector, left: u32::MAX, right: u32::MAX }
    }

    fn side<R: Rng>(plane: &SplitPlaneNormal, node: &Leaf<Self>, rng: &mut R) -> Side {
        let dot = dot_product_no_simd(&plane.normal, &node.vector);
        if dot > 0.0 {
            Side::Right
        } else if dot < 0.0 {
            Side::Left
        } else {
            Side::random(rng)
        }
    }
}
