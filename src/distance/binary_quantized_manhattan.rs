use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use rand::Rng;

use super::{two_means_binary_quantized as two_means, Manhattan};
use crate::distance::Distance;
use crate::node::Leaf;
use crate::parallel::ImmutableSubsetLeafs;
use crate::unaligned_vector::{self, BinaryQuantized, UnalignedVector};

/// A taxicab geometry or a Manhattan geometry is a geometry whose usual distance function
/// or metric of Euclidean geometry is replaced by a new metric in which the distance between
/// two points is the sum of the absolute differences of their Cartesian coordinates.
#[derive(Debug, Clone)]
pub enum BinaryQuantizedManhattan {}

/// The header of BinaryQuantizedEuclidean leaf nodes.
#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderBinaryQuantizedManhattan {
    /// An extra constant term to determine the offset of the plane
    bias: f32,
}

impl Distance for BinaryQuantizedManhattan {
    const DEFAULT_OVERSAMPLING: usize = 3;

    type Header = NodeHeaderBinaryQuantizedManhattan;
    type VectorCodec = unaligned_vector::BinaryQuantized;

    fn name() -> &'static str {
        "binary quantized manhattan"
    }

    fn new_header(_vector: &UnalignedVector<Self::VectorCodec>) -> Self::Header {
        NodeHeaderBinaryQuantizedManhattan { bias: 0.0 }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        manhattan_distance(&p.vector, &q.vector)
    }

    /// Normalizes the distance returned by the distance method.
    fn normalized_distance(d: f32, dimensions: usize) -> f32 {
        d.max(0.0) / dimensions as f32
    }

    fn norm_no_header(v: &UnalignedVector<Self::VectorCodec>) -> f32 {
        let ones = v.as_bytes().iter().flat_map(|b| bits(*b)).sum::<f32>();
        ones.sqrt()
    }

    fn init(_node: &mut Leaf<Self>) {}

    fn create_split<'a, R: Rng>(
        children: &'a ImmutableSubsetLeafs<Self>,
        rng: &mut R,
    ) -> heed::Result<Cow<'a, UnalignedVector<Self::VectorCodec>>> {
        let [node_p, node_q] = two_means::<Self, Manhattan, R>(rng, children, false)?;
        let vector: Vec<f32> =
            node_p.vector.iter().zip(node_q.vector.iter()).map(|(p, q)| p - q).collect();
        let mut normal = Leaf {
            header: NodeHeaderBinaryQuantizedManhattan { bias: 0.0 },
            vector: UnalignedVector::from_slice(&vector),
        };
        Self::normalize(&mut normal);

        Ok(Cow::Owned(normal.vector.into_owned()))
    }

    fn margin(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        p.header.bias + dot_product(&p.vector, &q.vector)
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

fn manhattan_distance(
    u: &UnalignedVector<BinaryQuantized>,
    v: &UnalignedVector<BinaryQuantized>,
) -> f32 {
    // u.as_bytes().iter().zip(v.as_bytes()).map(|(u, v)| (u ^ v).count_ones()).sum::<u32>() as f32

    u.as_bytes()
        .iter()
        .zip(v.as_bytes())
        .flat_map(|(u, v)| {
            let u = bits(*u);
            let v = bits(*v);
            u.into_iter().zip(v).map(|(u, v)| (u - v).abs())
        })
        .sum::<f32>()
}
