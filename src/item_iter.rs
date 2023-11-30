use crate::{Distance, ItemId, KeyCodec, Leaf, Node, NodeCodec, Result};

pub struct ItemIter<'t, D: Distance> {
    pub(crate) inner: heed::RoPrefix<'t, KeyCodec, NodeCodec<D>>,
}

impl<'t, D: Distance> Iterator for ItemIter<'t, D> {
    // TODO think about exposing the UnalignedF32Slice type
    type Item = Result<(ItemId, Vec<f32>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(Ok((key, node))) => match node {
                Node::Leaf(Leaf { header: _, vector }) => Some(Ok((key.item, vector.into_owned()))),
                Node::Descendants(_) | Node::SplitPlaneNormal(_) => None,
            },
            Some(Err(e)) => Some(Err(e.into())),
            None => None,
        }
    }
}
