use std::borrow::Cow;

use crate::{Distance, ItemId, Leaf, Node, NodeCodec, Result, BEU32};

pub struct ItemIter<'t, D: Distance> {
    pub(crate) inner: heed::RoIter<'t, BEU32, NodeCodec<D>>,
}

impl<'t, D: Distance> Iterator for ItemIter<'t, D> {
    type Item = Result<(ItemId, Cow<'t, [f32]>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(Ok((item_id, node))) => match node {
                Node::Leaf(Leaf { header: _, vector }) => Some(Ok((item_id, vector))),
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => None,
            },
            Some(Err(e)) => Some(Err(e.into())),
            None => None,
        }
    }
}
