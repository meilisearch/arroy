use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::{two_means_binary_quantized as two_means, Angular};
use crate::distance::Distance;
use crate::node::Leaf;
use crate::parallel::ImmutableSubsetLeafs;
use crate::unaligned_vector::{self, BinaryQuantized, UnalignedVector};

/// The Cosine similarity is a measure of similarity between two
/// non-zero vectors defined in an inner product space. Cosine similarity
/// is the cosine of the angle between the vectors.
#[derive(Debug, Clone)]
pub enum BinaryQuantizedAngular {}

/// The header of BinaryQuantizedAngular leaf nodes.
#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderBinaryQuantizedAngular {
    norm: f32,
}

impl Distance for BinaryQuantizedAngular {
    const DEFAULT_OVERSAMPLING: usize = 3;

    type Header = NodeHeaderBinaryQuantizedAngular;
    type VectorCodec = unaligned_vector::BinaryQuantized;

    fn name() -> &'static str {
        "binary quantized angular"
    }

    fn new_header(vector: &UnalignedVector<Self::VectorCodec>) -> Self::Header {
        NodeHeaderBinaryQuantizedAngular { norm: Self::norm_no_header(vector) }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        let pn = p.header.norm;
        let qn = q.header.norm;
        let pq = dot_product(&p.vector, &q.vector);
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
        dot_product(v, v).sqrt()
    }

    fn init(node: &mut Leaf<Self>) {
        node.header.norm = dot_product(&node.vector, &node.vector).sqrt();
    }

    fn create_split<'a, R: Rng>(
        children: &'a ImmutableSubsetLeafs<Self>,
        rng: &mut R,
    ) -> heed::Result<Cow<'a, UnalignedVector<Self::VectorCodec>>> {
        let [node_p, node_q] = two_means::<Self, Angular, R>(rng, children, true)?;
        let vector: Vec<f32> =
            node_p.vector.iter().zip(node_q.vector.iter()).map(|(p, q)| p - q).collect();
        let unaligned_vector = UnalignedVector::from_vec(vector);
        let mut normal = Leaf {
            header: NodeHeaderBinaryQuantizedAngular { norm: 0.0 },
            vector: unaligned_vector,
        };
        Self::normalize(&mut normal);

        Ok(normal.vector)
    }

    fn margin_no_header(
        p: &UnalignedVector<Self::VectorCodec>,
        q: &UnalignedVector<Self::VectorCodec>,
    ) -> f32 {
        dot_product(p, q)
    }
}

fn bits(mut word: u8) -> [f32; 8] {
    let mut ret = [0.0; 8];
    for i in 0..8 {
        let bit = word & 1;
        word >>= 1;
        if bit == 0 {
            ret[i] = -1.0;
            // ret[i] = 0.0;
        } else {
            ret[i] = 1.0;
        }
    }

    ret
}

fn dot_product(u: &UnalignedVector<BinaryQuantized>, v: &UnalignedVector<BinaryQuantized>) -> f32 {
    // /!\ If the number of dimensions is not a multiple of the `Word` size, we'll xor 0 bits at the end, which will generate a lot of 1s.
    //     This may or may not impact relevancy since the 1s will be added to every vector.
    // u.as_bytes().iter().zip(v.as_bytes()).map(|(u, v)| (u | v).count_ones()).sum::<u32>() as f32

    u.as_bytes()
        .iter()
        .zip(v.as_bytes())
        .flat_map(|(u, v)| {
            let u = bits(*u);
            let v = bits(*v);
            u.into_iter().zip(v).map(|(u, v)| u * v)
        })
        .sum::<f32>()
}

fn squared_euclidean_distance(
    u: &UnalignedVector<BinaryQuantized>,
    v: &UnalignedVector<BinaryQuantized>,
) -> f32 {
    // /!\ If the number of dimensions is not a multiple of the `Word` size, we'll xor 0 bits at the end, which will generate a lot of 1s.
    //     This may or may not impact relevancy since the 1s will be added to every vector.
    // u.as_bytes().iter().zip(v.as_bytes()).map(|(u, v)| (u ^ v).count_ones()).sum::<u32>() as f32

    u.as_bytes()
        .iter()
        .zip(v.as_bytes())
        .flat_map(|(u, v)| {
            let u = bits(*u);
            let v = bits(*v);
            u.into_iter().zip(v).map(|(u, v)| (u - v) * (u - v))
        })
        .sum::<f32>()
}
