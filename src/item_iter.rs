use crate::distance::Distance;
use crate::internals::KeyCodec;
use crate::node::Leaf;
use crate::{ItemId, Node, NodeCodec, Result};

/// An iterator over the items (vectors) stored in an arroy index.
///
/// Returned by [`Reader::iter`](crate::Reader::iter).
pub struct ItemIter<'t, D: Distance> {
    pub(crate) inner: heed::RoPrefix<'t, KeyCodec, NodeCodec<D>>,
}

impl<D: Distance> Iterator for ItemIter<'_, D> {
    // TODO think about exposing the UnalignedF32Slice type
    type Item = Result<(ItemId, Vec<f32>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(Ok((key, node))) => match node {
                Node::Leaf(Leaf { header: _, vector }) => {
                    Some(Ok((key.node.item, vector.to_vec())))
                }
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => None,
            },
            Some(Err(e)) => Some(Err(e.into())),
            None => None,
        }
    }
}
