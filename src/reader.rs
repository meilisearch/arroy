use std::marker;

use heed::{Database, RoTxn};

use crate::node::Leaf;
use crate::{Distance, ItemId, Node, NodeCodec, BEU32};

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

    pub(crate) fn from_database_and_dimensions(
        database: Database<BEU32, NodeCodec<D>>,
        dimensions: usize,
    ) -> Reader<D> {
        Reader { database, dimensions, _marker: marker::PhantomData }
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
