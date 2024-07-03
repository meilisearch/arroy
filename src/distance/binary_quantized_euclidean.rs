use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::two_means;
use crate::distance::Distance;
use crate::node::{Leaf, UnalignedVector};
use crate::parallel::ImmutableSubsetLeafs;
use crate::spaces::simple::dot_product;

/// The Euclidean distance between two points in Euclidean space
/// is the length of the line segment between them.
///
/// `d(p, q) = sqrt((p - q)²)`
#[derive(Debug, Clone)]
pub enum BinaryQuantizedEuclidean {}

/// The header of BinaryQuantizedEuclidean leaf nodes.
#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderBinaryQuantizedEuclidean {
    /// An extra constant term to determine the offset of the plane
    bias: f32,
}

impl Distance for BinaryQuantizedEuclidean {
    type Header = NodeHeaderBinaryQuantizedEuclidean;

    fn name() -> &'static str {
        "binary quantized euclidean"
    }

    fn new_header(_vector: &UnalignedVector) -> Self::Header {
        NodeHeaderBinaryQuantizedEuclidean { bias: 0.0 }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        binary_quantized_euclidean_distance(&p.vector, &q.vector)
    }

    fn init(_node: &mut Leaf<Self>) {}

    fn create_split<'a, R: Rng>(
        children: &'a ImmutableSubsetLeafs<Self>,
        rng: &mut R,
    ) -> heed::Result<Cow<'a, UnalignedVector>> {
        let [node_p, node_q] = two_means(rng, children, false)?;
        let vector: Vec<f32> =
            node_p.vector.iter_f32().zip(node_q.vector.iter_f32()).map(|(p, q)| p - q).collect();
        let mut normal = Leaf {
            header: NodeHeaderBinaryQuantizedEuclidean { bias: 0.0 },
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

fn binary_quantized_euclidean_distance(u: &UnalignedVector, v: &UnalignedVector) -> f32 {
    u.as_bytes().iter().zip(v.as_bytes()).map(|(u, v)| (u ^ v).count_ones()).sum::<u32>() as f32
}
