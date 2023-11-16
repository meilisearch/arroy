use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::{dot_product_no_simd, two_means};
use crate::node::{Leaf, SplitPlaneNormal};
use crate::Distance;

#[derive(Debug, Clone)]
pub enum Euclidean {}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderEuclidean {
    /// An extra constant term to determine the offset of the plane
    bias: f32,
}

impl Distance for Euclidean {
    type Header = NodeHeaderEuclidean;

    fn new_header(_vector: &[f32]) -> Self::Header {
        NodeHeaderEuclidean { bias: 0.0 }
    }

    fn distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        // Don't use dot-product: avoid catastrophic cancellation in
        // https://github.com/spotify/annoy/issues/314.
        p.vector.iter().zip(q.vector.iter()).map(|(&p, &q)| (p - q) * (p - q)).sum()
    }

    fn init(_node: &mut Leaf<Self>) {}

    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> SplitPlaneNormal<'static> {
        let [node_p, node_q] = two_means(rng, children, false);
        let vector = node_p.vector.iter().zip(node_q.vector.iter()).map(|(&p, &q)| p - q).collect();
        let mut normal = Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector };
        Self::normalize(&mut normal);

        let mut bias = 0.0;
        let iter = normal.vector.iter().zip(node_p.vector.iter()).zip(node_q.vector.iter());
        iter.for_each(|((&n, &p), &q)| {
            bias += -n * (p + q) / 2.0;
        });
        normal.header.bias = bias;

        // TODO we are returning invalid left and rights
        SplitPlaneNormal { normal: normal.vector, left: u32::MAX, right: u32::MAX }
    }

    fn margin(p: &Leaf<Self>, q: &[f32]) -> f32 {
        p.header.bias + dot_product_no_simd(&p.vector, q)
    }

    fn margin_no_header(p: &[f32], q: &[f32]) -> f32 {
        dot_product_no_simd(p, q)
    }
}
