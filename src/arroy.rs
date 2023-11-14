use std::borrow::Cow;
use std::error::Error;
use std::mem::size_of;
use std::{cmp, marker};

use bytemuck::checked::cast_slice;
use bytemuck::{bytes_of, cast, pod_collect_to_vec, pod_read_unaligned, Pod, Zeroable};
use byteorder::{BigEndian, ByteOrder};
use heed::types::{ByteSlice, DecodeIgnore};
use heed::{BytesDecode, BytesEncode, Database, RoTxn, RwTxn};
use rand::seq::index;
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
        let last_item_id = match self.database.last(wtxn)? {
            Some((i, _)) => i,
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

        let last_node_id = match self.database.last(wtxn)? {
            Some((i, _)) => i,
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

            let node = if is_root {
                Node::Root(Root { children: () })
            } else {
                // Node::
                unimplemented!()
            };

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

        // threaded_build_policy.lock_shared_nodes();
        // vector<Node*> children;
        // for (size_t i = 0; i < indices.size(); i++) {
        //   S j = indices[i];
        //   Node* n = _get(j);
        //   if (n)
        //     children.push_back(n);
        // }

        // vector<S> children_indices[2];
        // Node* m = (Node*)alloca(_s);

        // for (int attempt = 0; attempt < 3; attempt++) {
        //   children_indices[0].clear();
        //   children_indices[1].clear();
        //   D::create_split(children, _f, _s, _random, m);

        //   for (size_t i = 0; i < indices.size(); i++) {
        //     S j = indices[i];
        //     Node* n = _get(j);
        //     if (n) {
        //       bool side = D::side(m, n, _f, _random);
        //       children_indices[side].push_back(j);
        //     } else {
        //       annoylib_showUpdate("No node for index %d?\n", j);
        //     }
        //   }

        //   if (_split_imbalance(children_indices[0], children_indices[1]) < 0.95)
        //     break;
        // }
        // threaded_build_policy.unlock_shared_nodes();

        // // If we didn't find a hyperplane, just randomize sides as a last option
        // while (_split_imbalance(children_indices[0], children_indices[1]) > 0.99) {
        //   if (_verbose)
        //     annoylib_showUpdate("\tNo hyperplane found (left has %zu children, right has %zu children)\n",
        //       children_indices[0].size(), children_indices[1].size());

        //   children_indices[0].clear();
        //   children_indices[1].clear();

        //   // Set the vector to 0.0
        //   for (int z = 0; z < _f; z++)
        //     m->v[z] = 0;

        //   for (size_t i = 0; i < indices.size(); i++) {
        //     S j = indices[i];
        //     // Just randomize...
        //     children_indices[_random.flip()].push_back(j);
        //   }
        // }

        // int flip = (children_indices[0].size() > children_indices[1].size());

        // m->n_descendants = is_root ? _n_items : (S)indices.size();
        // for (int side = 0; side < 2; side++) {
        //   // run _make_tree for the smallest child first (for cache locality)
        //   m->children[side^flip] = _make_tree(children_indices[side^flip], false, _random, threaded_build_policy);
        // }

        // threaded_build_policy.lock_n_nodes();
        // _allocate_size(_n_nodes + 1, threaded_build_policy);
        // S item = _n_nodes++;
        // threaded_build_policy.unlock_n_nodes();

        // threaded_build_policy.lock_shared_nodes();
        // memcpy(_get(item), m, _s);
        // threaded_build_policy.unlock_shared_nodes();

        // return item;

        todo!()
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
        Some(Node::Root(_)) => Ok(None),
        None => Ok(None),
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
enum Node<D: Distance> {
    Leaf(Leaf<D>),
    SplitPlaneNormal(SplitPlaneNormal),
    Root(Root),
}

const LEAF_TAG: u8 = 0;
const SPLIT_PLANE_NORMAL_TAG: u8 = 1;
const ROOT_TAG: u8 = 2;

struct Leaf<D: Distance> {
    pub header: D::Header,
    pub vector: Vec<f32>,
}

struct SplitPlaneNormal {
    // pub header: D::Header,
    pub normal: Vec<f32>,
    // TODO make the deser lazy
    pub left: Vec<NodeId>,
    pub right: Vec<NodeId>,
}

struct Root {
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
                let normal_len: u32 = normal.len().try_into().unwrap();
                let left_len: u32 = left.len().try_into().unwrap();

                bytes.extend_from_slice(&normal_len.to_be_bytes());
                bytes.extend_from_slice(&left_len.to_be_bytes());

                bytes.extend_from_slice(cast_slice(normal));
                bytes.extend_from_slice(cast_slice(left));
                bytes.extend_from_slice(cast_slice(right));
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
                let normal_len = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                let left_len = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                let (normal_bytes, bytes) = &bytes.split_at(normal_len as usize);
                let (left_bytes, right_bytes) = &bytes.split_at(left_len as usize);
                Ok(Node::SplitPlaneNormal(SplitPlaneNormal {
                    normal: pod_collect_to_vec(normal_bytes),
                    left: pod_collect_to_vec(left_bytes),
                    right: pod_collect_to_vec(right_bytes),
                }))
            }
            unknown => panic!("What the fuck is an {unknown:?}"),
        }
    }
}

pub trait Distance {
    type Header: Pod + Zeroable;

    fn name() -> &'static str;
    fn new_header(vector: &[f32]) -> Self::Header;
    fn distance(p: &[f32], q: &[f32]) -> f32;
    fn norm(v: &[f32]) -> f32;
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
