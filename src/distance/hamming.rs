use crate::distance::Distance;
use crate::node::Leaf;
use crate::parallel::ImmutableSubsetLeafs;
use crate::unaligned_vector::{Binary, UnalignedVector};
use bytemuck::{Pod, Zeroable};
use rand::Rng;
use std::borrow::Cow;

/// The Hamming distance between two vectors is the number of positions at
/// which the corresponding symbols are different.
///
/// `d(u,v) = ||u ^ v||₁`
///
/// /!\ This distance function is binary, which means it loses all its precision
///     and their scalar values are converted to `0` or `1` under the rule
///     `x > 0.0 => 1`, otherwise `0`
#[derive(Debug, Clone)]
pub enum Hamming {}

/// The header of BinaryEuclidean leaf nodes.
#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderHamming {}

impl Distance for Hamming {
    const DEFAULT_OVERSAMPLING: usize = 3;

    type Header = NodeHeaderHamming;
    type VectorCodec = Binary;

    fn name() -> &'static str {
        "hamming"
    }

    fn new_header(_vector: &UnalignedVector<Self::VectorCodec>) -> Self::Header {
        NodeHeaderHamming {}
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        hamming_bitwise_fast(p.vector.as_bytes(), q.vector.as_bytes())
    }

    fn normalized_distance(d: f32, dimensions: usize) -> f32 {
        d.max(0.0) / dimensions as f32
    }

    fn norm_no_header(v: &UnalignedVector<Self::VectorCodec>) -> f32 {
        v.as_bytes().iter().map(|b| b.count_ones() as i32).sum::<i32>() as f32
    }

    fn init(_node: &mut Leaf<Self>) {}

    fn create_split<'a, R: Rng>(
        children: &'a ImmutableSubsetLeafs<Self>,
        rng: &mut R,
    ) -> heed::Result<Cow<'a, UnalignedVector<Self::VectorCodec>>> {
        // unlike other distances which build a seperating hyperplane we
        // construct an LSH by bit sampling and store the random bit in a one-hot 
        // vector
        // https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Bit_sampling_for_Hamming_distance

        const ITERATION_STEPS: usize = 200;

        let is_valid_split = |v: &UnalignedVector<Self::VectorCodec>, rng: &mut R| {
            let mut count = 0;
            for _ in 0..ITERATION_STEPS {
                let u = children.choose(rng)?.unwrap().vector;
                if <Self as Distance>::margin_no_header(v, u.as_ref()) > 0.0 {
                    count += 1;
                }
            }
            Ok::<bool, heed::Error>(count > 0 && count < ITERATION_STEPS)
        };

        // first try random index
        let dim = children.choose(rng)?.unwrap().vector.len();
        let mut n: Vec<f32> = vec![0.0; dim];
        let idx = rng.gen_range(0..dim);
        n[idx] = 1.0;
        let mut normal = UnalignedVector::from_vec(n);

        if is_valid_split(&normal, rng)? {
            return Ok(Cow::Owned(normal.into_owned()));
        }

        // otherwise brute-force search for a splitting coordinate
        for j in 0..dim {
            let mut n: Vec<f32> = vec![0.0; dim];
            n[j] = 1.0;
            normal = UnalignedVector::from_vec(n);

            if is_valid_split(&normal, rng)? {
                return Ok(Cow::Owned(normal.into_owned()));
            }
        }

        // fallback
        Ok(Cow::Owned(normal.into_owned()))
    }

    fn margin_no_header(
        p: &UnalignedVector<Self::VectorCodec>,
        q: &UnalignedVector<Self::VectorCodec>,
    ) -> f32 {
        // p is a mask with 1 bit set
        let ret =
            p.as_bytes().iter().zip(q.as_bytes()).map(|(u, v)| (u & v).count_ones()).sum::<u32>();
        ret as f32
    }
}

#[inline]
pub fn hamming_bitwise_fast(u: &[u8], v: &[u8]) -> f32 {
    // based on : https://github.com/emschwartz/hamming-bitwise-fast
    // Explicitly structuring the code as below lends itself to SIMD optimizations by
    // the compiler -> https://matklad.github.io/2023/04/09/can-you-trust-a-compiler-to-optimize-your-code.html
    assert_eq!(u.len(), v.len());

    type BitPackedWord = u64;
    const CHUNK_SIZE: usize = std::mem::size_of::<BitPackedWord>();

    let mut distance = u
        .chunks_exact(CHUNK_SIZE)
        .zip(v.chunks_exact(CHUNK_SIZE))
        .map(|(u_chunk, v_chunk)| {
            let u_val = BitPackedWord::from_ne_bytes(u_chunk.try_into().unwrap());
            let v_val = BitPackedWord::from_ne_bytes(v_chunk.try_into().unwrap());
            (u_val ^ v_val).count_ones()
        })
        .sum::<u32>();

    if u.len() % CHUNK_SIZE != 0 {
        distance += u
            .chunks_exact(CHUNK_SIZE)
            .remainder()
            .iter()
            .zip(v.chunks_exact(CHUNK_SIZE).remainder())
            .map(|(u_byte, v_byte)| (u_byte ^ v_byte).count_ones())
            .sum::<u32>();
    }

    distance as f32
}
