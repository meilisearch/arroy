use std::borrow::Cow;

use heed::iteration_method::MoveThroughDuplicateValues;
use heed::DefaultComparator;

use crate::{Distance, ItemId, KeyCodec, Leaf, Node, NodeCodec, Result};

pub struct ItemIter<'t, D: Distance> {
    pub(crate) inner:
        heed::RoPrefix<'t, KeyCodec, NodeCodec<D>, DefaultComparator, MoveThroughDuplicateValues>,
    // pub(crate) inner: heed::RoPrefix<'t, KeyCodec, NodeCodec<D>>,
}

impl<'t, D: Distance> Iterator for ItemIter<'t, D> {
    type Item = Result<(ItemId, Cow<'t, [f32]>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(Ok((key, node))) => match node {
                Node::Leaf(Leaf { header: _, vector }) => Some(Ok((key.node.item, vector))),
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => None,
            },
            Some(Err(e)) => Some(Err(e.into())),
            None => None,
        }
    }
}
