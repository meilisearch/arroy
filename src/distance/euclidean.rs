use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::two_means;
use crate::distance::Distance;
use crate::node::{Leaf, UnalignedF32Slice};
use crate::spaces::simple::{dot_product, euclidean_distance};

/// The Euclidean distance between two points in Euclidean space
/// is the length of the line segment between them.
///
/// `d(p, q) = sqrt((p - q)²)`
#[derive(Debug, Clone)]
pub enum Euclidean {}

/// The header of Euclidean leaf nodes.
#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderEuclidean {
    /// An extra constant term to determine the offset of the plane
    bias: f32,
}

impl Distance for Euclidean {
    type Header = NodeHeaderEuclidean;

    fn name() -> &'static str {
        "euclidean"
    }

    fn new_header(_vector: &UnalignedF32Slice) -> Self::Header {
        NodeHeaderEuclidean { bias: 0.0 }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        euclidean_distance(&p.vector, &q.vector)
    }

    fn init(_node: &mut Leaf<Self>) {}

    fn create_split<R: Rng>(children: &[Leaf<Self>], rng: &mut R) -> Vec<f32> {
        let [node_p, node_q] = two_means(rng, children, false);
        let vector = node_p.vector.iter().zip(node_q.vector.iter()).map(|(p, q)| p - q).collect();
        let mut normal =
            Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: Cow::Owned(vector) };
        Self::normalize(&mut normal);

        normal.header.bias = normal
            .vector
            .iter()
            .zip(node_p.vector.iter())
            .zip(node_q.vector.iter())
            .map(|((n, p), q)| -n * (p + q) / 2.0)
            .sum();

        normal.vector.into_owned()
    }

    fn margin(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        p.header.bias + dot_product(&p.vector, &q.vector)
    }

    fn margin_no_header(p: &UnalignedF32Slice, q: &UnalignedF32Slice) -> f32 {
        dot_product(p, q)
    }
}
