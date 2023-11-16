use std::marker;

use bytemuck::pod_collect_to_vec;
use heed::types::ByteSlice;
use heed::{Database, RoTxn};

use crate::node::Leaf;
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
        leaf: &Leaf<D>,
        count: usize,
        search_k: Option<usize>, // TODO consider Option<NonZeroUsize>
    ) -> heed::Result<Vec<(ItemId, f32)>> {
        todo!()
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
