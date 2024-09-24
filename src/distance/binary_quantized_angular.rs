use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::{two_means_binary_quantized as two_means, Angular};
use crate::distance::Distance;
use crate::node::Leaf;
use crate::parallel::ImmutableSubsetLeafs;
use crate::spaces::simple::dot_product_binary_quantized;
use crate::unaligned_vector::{BinaryQuantized, UnalignedVector};

/// The Cosine similarity is a measure of similarity between two
/// non-zero vectors defined in an inner product space. Cosine similarity
/// is the cosine of the angle between the vectors.
/// /!\ This distance function is binary quantized, which means it loses all its precision
///     and their scalar values are converted to `-1` or `1`.
#[derive(Debug, Clone)]
pub enum BinaryQuantizedAngular {}

/// The header of `BinaryQuantizedAngular` leaf nodes.
#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderBinaryQuantizedAngular {
    norm: f32,
}

impl Distance for BinaryQuantizedAngular {
    const DEFAULT_OVERSAMPLING: usize = 3;

    type Header = NodeHeaderBinaryQuantizedAngular;
    type VectorCodec = BinaryQuantized;

    fn name() -> &'static str {
        "binary quantized angular"
    }

    fn new_header(vector: &UnalignedVector<Self::VectorCodec>) -> Self::Header {
        NodeHeaderBinaryQuantizedAngular { norm: Self::norm_no_header(vector) }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        let pn = p.header.norm;
        let qn = q.header.norm;
        let pq = dot_product_binary_quantized(&p.vector, &q.vector);
        let pnqn = pn * qn;
        if pnqn != 0.0 {
            let cos = pq / pnqn;
            // cos is [-1; 1]
            // cos =  0. -> 0.5
            // cos = -1. -> 1.0
            // cos =  1. -> 0.0
            (1.0 - cos) / 2.0
        } else {
            0.0
        }
    }

    /// Normalizes the distance returned by the distance method.
    fn normalized_distance(d: f32, _dimensions: usize) -> f32 {
        d
    }

    fn norm_no_header(v: &UnalignedVector<Self::VectorCodec>) -> f32 {
        dot_product_binary_quantized(v, v).sqrt()
    }

    fn init(node: &mut Leaf<Self>) {
        node.header.norm = dot_product_binary_quantized(&node.vector, &node.vector).sqrt();
    }

    fn create_split<'a, R: Rng>(
        children: &'a ImmutableSubsetLeafs<Self>,
        rng: &mut R,
    ) -> heed::Result<Leaf<'static, Self>> {
        let [node_p, node_q] = two_means::<Self, Angular, R>(rng, children, true)?;
        let vector: Vec<f32> =
            node_p.vector.iter().zip(node_q.vector.iter()).map(|(p, q)| p - q).collect();
        let unaligned_vector = UnalignedVector::from_vec(vector);
        let mut normal = Leaf {
            header: NodeHeaderBinaryQuantizedAngular { norm: 0.0 },
            vector: unaligned_vector,
        };
        Self::normalize(&mut normal);

        Ok(normal)
    }

    fn margin_no_header(
        p: &UnalignedVector<Self::VectorCodec>,
        q: &UnalignedVector<Self::VectorCodec>,
    ) -> f32 {
        dot_product_binary_quantized(p, q)
    }
}
