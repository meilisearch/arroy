use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use heed::{RwPrefix, RwTxn};
use rand::Rng;

use super::two_means;
use crate::distance::Distance;
use crate::internals::KeyCodec;
use crate::node::{Leaf, UnalignedF32Slice};
use crate::parallel::ImmutableSubsetLeafs;
use crate::spaces::simple::dot_product;
use crate::{Node, NodeCodec};

/// In mathematics, the dot product or scalar product is an algebraic
/// operation that takes two equal-length sequences of numbers
/// (usually coordinate vectors), and returns a single number.
#[derive(Debug, Clone)]
pub enum DotProduct {}

/// The header of DotProduct leaf nodes.
#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderDotProduct {
    extra_dim: f32,
    /// An extra constant term to determine the offset of the plane
    norm: f32,
}

impl Distance for DotProduct {
    type Header = NodeHeaderDotProduct;

    fn name() -> &'static str {
        "dot-product"
    }

    fn new_header(_vector: &UnalignedF32Slice) -> Self::Header {
        // We compute the norm when we preprocess the vector, before generating the tree nodes.
        NodeHeaderDotProduct { extra_dim: 0.0, norm: 0.0 }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        // When index is already built, we don't need angular distances to retrieve NNs
        // Thus, we can return dot product scores itself
        -dot_product(&p.vector, &q.vector)
    }

    fn non_built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        // Calculated by analogy with the angular case
        let pp = p.header.norm;
        let qq = q.header.norm;
        let pq = dot_product(&p.vector, &q.vector) + p.header.extra_dim * q.header.extra_dim;
        let ppqq = pp * qq;

        if ppqq >= f32::MIN_POSITIVE {
            2.0 - 2.0 * pq / ppqq.sqrt()
        } else {
            2.
        }
    }

    fn norm(leaf: &Leaf<Self>) -> f32 {
        let dot = dot_product(&leaf.vector, &leaf.vector);
        (dot + leaf.header.extra_dim * leaf.header.extra_dim).sqrt()
    }

    fn normalized_distance(d: f32) -> f32 {
        -d
    }

    fn normalize(node: &mut Leaf<Self>) {
        let norm = Self::norm(node);
        if norm > 0.0 {
            node.vector.to_mut().iter_mut().for_each(|x| *x /= norm);
            node.header.extra_dim /= norm;
        }
    }

    fn init(node: &mut Leaf<Self>) {
        node.header.norm = dot_product(&node.vector, &node.vector);
    }

    fn create_split<R: Rng>(
        children: &ImmutableSubsetLeafs<Self>,
        rng: &mut R,
    ) -> heed::Result<Vec<f32>> {
        let [node_p, node_q] = two_means(rng, children, true)?;
        let vector = node_p.vector.iter().zip(node_q.vector.iter()).map(|(p, q)| p - q).collect();
        let mut normal = Leaf::<Self> {
            header: NodeHeaderDotProduct { norm: 0.0, extra_dim: 0.0 },
            vector: Cow::Owned(vector),
        };
        normal.header.extra_dim = node_p.header.extra_dim - node_q.header.extra_dim;
        Self::normalize(&mut normal);

        Ok(normal.vector.into_owned())
    }

    fn margin(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        dot_product(&p.vector, &q.vector) + p.header.extra_dim * q.header.extra_dim
    }

    fn margin_no_header(p: &UnalignedF32Slice, q: &UnalignedF32Slice) -> f32 {
        dot_product(p, q)
    }

    fn preprocess(
        wtxn: &mut RwTxn,
        new_iter: impl for<'a> Fn(
            &'a mut RwTxn,
        ) -> heed::Result<RwPrefix<'a, KeyCodec, NodeCodec<Self>>>,
    ) -> heed::Result<()> {
        // Highly inspired by the DotProduct::preprocess function:
        // https://github.com/spotify/annoy/blob/2be37c9e015544be2cf60c431f0cccc076151a2d/src/annoylib.h#L661-L694
        //
        // This uses a method from Microsoft Research for transforming inner product spaces to cosine/angular-compatible spaces.
        // (Bachrach et al., 2014, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf)

        // Step one: compute the norm of each vector and find the maximum norm
        let mut max_norm = 0.0;
        for result in new_iter(wtxn)? {
            let (_item_id, node) = result?;
            let leaf = match node.leaf() {
                Some(leaf) => leaf,
                None => break,
            };

            let norm = Self::norm_no_header(&leaf.vector);
            max_norm = f32::max(max_norm, norm);
        }

        // Step two: set each vector's extra dimension to sqrt(max_norm^2 - norm^2)
        // Note: we put that in a dedicated header value
        let mut cursor = new_iter(wtxn)?;
        while let Some((item_id, node)) = cursor.next().transpose()? {
            let leaf = match node.leaf() {
                Some(leaf) => leaf,
                None => break,
            };

            let node_norm = Self::norm_no_header(&leaf.vector);
            let squared_norm_diff = (max_norm * max_norm) - (node_norm * node_norm);

            let mut leaf = leaf.into_owned();
            leaf.header.norm = max_norm * max_norm;
            leaf.header.extra_dim = squared_norm_diff.sqrt();

            // safety: We do not keep a reference to the current value, we own it.
            unsafe { cursor.put_current(&item_id, &Node::Leaf(leaf))? };
        }

        Ok(())
    }
}
