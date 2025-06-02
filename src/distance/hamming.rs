use crate::distance::Distance;
use crate::internals::Side;
use crate::node::Leaf;
use crate::parallel::ImmutableSubsetLeafs;
use crate::unaligned_vector::{Binary, UnalignedVector};
use bytemuck::{Pod, Zeroable};
use rand::Rng;

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
pub struct NodeHeaderHamming {
    idx: usize,
}

impl Distance for Hamming {
    const DEFAULT_OVERSAMPLING: usize = 3;

    type Header = NodeHeaderHamming;
    type VectorCodec = Binary;

    fn name() -> &'static str {
        "hamming"
    }

    fn new_header(_vector: &UnalignedVector<Self::VectorCodec>) -> Self::Header {
        NodeHeaderHamming { idx: 0 }
    }

    fn built_distance(p: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        hamming_bitwise_fast(p.vector.as_bytes(), q.vector.as_bytes())
    }

    fn normalized_distance(d: f32, _: usize) -> f32 {
        d
    }

    fn norm_no_header(v: &UnalignedVector<Self::VectorCodec>) -> f32 {
        v.as_bytes().iter().map(|b| b.count_ones() as i32).sum::<i32>() as f32
    }

    fn init(_node: &mut Leaf<Self>) {}

    fn create_split<'a, R: Rng>(
        children: &'a ImmutableSubsetLeafs<Self>,
        rng: &mut R,
    ) -> heed::Result<Leaf<'a, Self>> {
        // unlike other distances which build a seperating hyperplane we
        // construct an LSH by bit sampling and storing the splitting index
        // in the node header.
        // https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Bit_sampling_for_Hamming_distance

        const ITERATION_STEPS: usize = 200;

        let is_valid_split = |n: &Leaf<'a, Self>, rng: &mut R| {
            let mut count = 0;
            for _ in 0..ITERATION_STEPS {
                let u = children.choose(rng)?.unwrap();
                if <Self as Distance>::margin(n, &u).is_sign_positive() {
                    count += 1;
                }
            }
            Ok::<bool, heed::Error>(count > 0 && count < ITERATION_STEPS)
        };

        // first try random index
        let dim = children.choose(rng)?.unwrap().vector.len();
        let idx = rng.gen_range(0..dim);
        let mut normal =
            Leaf { header: NodeHeaderHamming { idx }, vector: UnalignedVector::from_vec(vec![]) };

        if is_valid_split(&normal, rng)? {
            return Ok(normal);
        }

        // otherwise brute-force search for a splitting coordinate
        for j in 0..dim {
            normal.header.idx = j;
            if is_valid_split(&normal, rng)? {
                return Ok(normal);
            }
        }

        // fallback
        Ok(normal)
    }

    fn margin(n: &Leaf<Self>, q: &Leaf<Self>) -> f32 {
        let v = q.vector.to_vec();
        if v[n.header.idx] == 1.0 {
            return 1.0;
        } else {
            return -1.0;
        }
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
