use std::borrow::Cow;
use std::error::Error;
use std::marker;
use std::mem::size_of;

use bytemuck::checked::cast_slice;
use bytemuck::{bytes_of, pod_collect_to_vec, pod_read_unaligned, Pod, Zeroable};
use byteorder::{BigEndian, ByteOrder};
use heed::types::{ByteSlice, DecodeIgnore};
use heed::{BytesDecode, BytesEncode, Database, RoTxn, RwTxn};
use rand::seq::SliceRandom;
use rand::Rng;

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
    n_items: usize,
    roots: Vec<NodeId>,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance> Writer<D> {
    pub fn open<U>(dimensions: usize, database: Database<BEU32, U>) -> Writer<D> {
        Writer {
            database: database.remap_data_type(),
            dimensions,
            n_items: todo!(),
            roots: Vec::new(),
            _marker: marker::PhantomData,
        }
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

        // _n_nodes = _n_items;
        // todo!("clear all the nodes but the items");

        self.n_items = self.database.len(wtxn)? as usize;
        let last_item_id = match self.last_node_id(wtxn)? {
            Some(last_id) => last_id,
            None => todo!(),
        };

        let mut thread_roots = Vec::new();
        loop {
            match n_trees {
                Some(n_trees) if thread_roots.len() >= n_trees => break,
                None if self.database.len(wtxn)? >= 2 * self.n_items as u64 => break,
                _ => (),
            }

            let mut indices = Vec::new();
            for result in self.database.remap_data_type::<DecodeIgnore>().iter(wtxn)? {
                let (i, _) = result?;
                indices.push(i);
                if i > last_item_id {
                    break;
                }
            }

            let tree_root_id = self.make_tree(wtxn, indices, true, &mut rng)?;
            thread_roots.push(tree_root_id);
        }

        self.roots.append(&mut thread_roots);

        // Also, copy the roots into the last segment of the database
        // This way we can load them faster without reading the whole file
        // TODO do not do that, store the root ids into the metadata field
        let n_nodes = self.database.len(wtxn)?;
        for (i, id) in self.roots.iter().enumerate() {
            let root_bytes = self.database.remap_data_type::<ByteSlice>().get(wtxn, id)?.unwrap();
            let root_vec = root_bytes.to_vec();
            let end_root_id = (n_nodes + i as u64).try_into().unwrap();
            self.database.remap_data_type::<ByteSlice>().put(wtxn, &end_root_id, &root_vec)?;
        }

        // D::template postprocess<T, S, Node>(_nodes, _s, _n_items, _f);

        Ok(Reader {
            database: self.database,
            dimensions: self.dimensions,
            _marker: marker::PhantomData,
        })
    }

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

        let last_node_id = match self.last_node_id(wtxn)? {
            Some(last_id) => last_id,
            None => todo!(),
        };

        // The basic rule is that if we have <= _K items, then it's a leaf node, otherwise it's a split node.
        // There's some regrettable complications caused by the problem that root nodes have to be "special":
        // 1. We identify root nodes by the arguable logic that _n_items == n->n_descendants, regardless of how many descendants they actually have
        // 2. Root nodes with only 1 child need to be a "dummy" parent
        // 3. Due to the _n_items "hack", we need to be careful with the cases where _n_items <= _K or _n_items > _K
        if indices.len() == 1 && !is_root {
            return Ok(indices[0]);
        }

        if indices.len() <= max_descendants
            && (!is_root || self.n_items <= max_descendants || indices.len() == 1)
        {
            //   threaded_build_policy.lock_n_nodes();
            //   _allocate_size(_n_nodes + 1, threaded_build_policy);
            //   S item = _n_nodes++;
            //   threaded_build_policy.unlock_n_nodes();
            let item = last_node_id + 1;

            // let node = if is_root {
            //     Node::Root(Root { children: () })
            // } else {
            //     // Node::
            //     unimplemented!()
            // };

            return Ok(item);

            //   threaded_build_policy.lock_shared_nodes();
            //   Node* m = _get(item);
            //   m->n_descendants = is_root ? _n_items : (S)indices.size();

            //   // Using std::copy instead of a loop seems to resolve issues #3 and #13,
            //   // probably because gcc 4.8 goes overboard with optimizations.
            //   // Using memcpy instead of std::copy for MSVC compatibility. #235
            //   // Only copy when necessary to avoid crash in MSVC 9. #293
            //   if (!indices.empty())
            //     memcpy(m->children, &indices[0], indices.size() * sizeof(S));

            //   threaded_build_policy.unlock_shared_nodes();
            //   return item;
        }

        let mut children = Vec::new();
        for node_id in &indices {
            let node = self.database.get(wtxn, node_id)?.unwrap();
            children.push(node);
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
        // m->n_descendants = is_root ? _n_items : (S)indices.size();
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
        Some(Node::Root(_)) => Ok(None),
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
        let di = ic * D::distance(&leaf_p.vector, &node_k.vector);
        let dj = jc * D::distance(&leaf_q.vector, &node_k.vector);
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
#[derive(Clone)]
pub enum Node<D: Distance> {
    Leaf(Leaf<D>),
    Descendants(Descendants),
    SplitPlaneNormal(SplitPlaneNormal),
    Root(Root),
}

const LEAF_TAG: u8 = 0;
const DESCENDANTS_TAG: u8 = 1;
const SPLIT_PLANE_NORMAL_TAG: u8 = 2;
const ROOT_TAG: u8 = 3;

#[derive(Clone)]
pub struct Leaf<D: Distance> {
    pub header: D::Header,
    pub vector: Vec<f32>,
}

#[derive(Clone)]
pub struct Descendants {
    descendants: Vec<NodeId>,
}

#[derive(Clone)]
pub struct SplitPlaneNormal {
    // pub header: D::Header,
    pub normal: Vec<f32>,
    pub left: NodeId,
    pub right: NodeId,
}

#[derive(Clone)]
pub struct Root {
    pub children: Vec<NodeId>,
}

struct NodeCodec<D>(D);

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
            Node::SplitPlaneNormal(SplitPlaneNormal { /* header,*/ normal, left, right }) => {
                bytes.push(SPLIT_PLANE_NORMAL_TAG);
                // bytes.extend_from_slice(bytes_of(header));
                // TODO return error on try_into ???
                // let normal_len: u32 = normal.len().try_into().unwrap();
                // let left_len: u32 = left.len().try_into().unwrap();

                // bytes.extend_from_slice(&normal_len.to_be_bytes());
                // bytes.extend_from_slice(&left_len.to_be_bytes());

                // bytes.extend_from_slice(cast_slice(normal));
                // bytes.extend_from_slice(cast_slice(left));
                // bytes.extend_from_slice(cast_slice(right));
                todo!()
            }
            _ => todo!(),
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
                let normal_len = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                let left_len = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                let (normal_bytes, bytes) = &bytes.split_at(normal_len as usize);
                let (left_bytes, right_bytes) = &bytes.split_at(left_len as usize);
                // Ok(Node::SplitPlaneNormal(SplitPlaneNormal {
                //     normal: pod_collect_to_vec(normal_bytes),
                //     left: pod_collect_to_vec(left_bytes),
                //     right: pod_collect_to_vec(right_bytes),
                // }))
                todo!()
            }
            unknown => panic!("What the fuck is an {unknown:?}"),
        }
    }
}

pub trait Distance: Sized + Clone {
    type Header: Pod + Zeroable;

    fn name() -> &'static str;
    fn new_header(vector: &[f32]) -> Self::Header;
    fn distance(p: &[f32], q: &[f32]) -> f32;
    fn norm(v: &[f32]) -> f32;
    fn normalize(node: &mut Leaf<Self>);
    fn init(node: &mut Leaf<Self>);
    fn update_mean(mean: &mut Leaf<Self>, new_node: &Leaf<Self>, norm: f32, c: f32);
    fn create_split<R: Rng>(children: &[Node<Self>], rng: &mut R) -> SplitPlaneNormal;
    fn side<R: Rng>(plane: &SplitPlaneNormal, node: &Node<Self>, rng: &mut R) -> Side;
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
