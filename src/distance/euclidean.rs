use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::two_means;
use crate::distance::Distance;
use crate::node::{Leaf, UnalignedVector};
use crate::parallel::ImmutableSubsetLeafs;
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

    fn new_header(_vector: &UnalignedVector) -> Self::Header {
        NodeHeaderEuclidean { bias: 0.0 }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        euclidean_distance(&p.vector, &q.vector)
    }

    fn init(_node: &mut Leaf<Self>) {}

    fn create_split<'a, R: Rng>(
        children: &'a ImmutableSubsetLeafs<Self>,
        rng: &mut R,
    ) -> heed::Result<Cow<'a, UnalignedVector>> {
        let [node_p, node_q] = two_means(rng, children, false)?;
        let vector: Vec<_> =
            node_p.vector.iter_f32().zip(node_q.vector.iter_f32()).map(|(p, q)| p - q).collect();
        let mut normal = Leaf {
            header: NodeHeaderEuclidean { bias: 0.0 },
            vector: Self::craft_owned_unaligned_vector_from_f32(vector),
        };
        Self::normalize(&mut normal);

        normal.header.bias = normal
            .vector
            .iter_f32()
            .zip(node_p.vector.iter_f32())
            .zip(node_q.vector.iter_f32())
            .map(|((n, p), q)| -n * (p + q) / 2.0)
            .sum();

        Ok(normal.vector)
    }

    fn margin(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        p.header.bias + dot_product(&p.vector, &q.vector)
    }

    fn margin_no_header(p: &UnalignedVector, q: &UnalignedVector) -> f32 {
        dot_product(p, q)
    }
}
