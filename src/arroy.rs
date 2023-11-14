use std::borrow::Cow;
use std::error::Error;
use std::marker;
use std::mem::size_of;

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
    roots: Vec<NodeId>,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance> Writer<D> {
    pub fn open<U>(dimensions: usize, database: Database<BEU32, U>) -> Writer<D> {
        Writer {
            database: database.remap_data_type(),
            dimensions,
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
        todo!("clear all the nodes but the items");

        let n_items = self.database.len(wtxn)?;
        let last_item_id = match self.database.last(wtxn)? {
            Some((i, _)) => i,
            None => todo!(),
        };

        let mut thread_roots = Vec::new();
        loop {
            match n_trees {
                Some(n_trees) if thread_roots.len() >= n_trees => break,
                None if self.database.len(wtxn)? >= 2 * n_items => break,
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
        todo!()
    }
}

fn item_vector<D: Distance>(
    database: Database<BEU32, NodeCodec<D>>,
    rtxn: &RoTxn,
    item: ItemId,
) -> heed::Result<Option<Vec<f32>>> {
    match database.get(rtxn, &item)? {
        Some(Node::Leaf(Leaf { header: _, vector })) => Ok(Some(vector)),
        Some(Node::SplitPlaneNormal(_)) => Ok(None),
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
}

const LEAF_TAG: u8 = 0;
const SPLIT_PLANE_NORMAL_TAG: u8 = 1;

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
