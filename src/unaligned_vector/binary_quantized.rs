use std::borrow::Cow;
use std::mem::{size_of, transmute};
use std::slice::ChunksExact;

use super::{SizeMismatch, UnalignedVector, UnalignedVectorCodec};

/// The type of the words used to quantize a vector
type QuantizedWord = u64;
/// The size of the words used to quantize a vector
const QUANTIZED_WORD_SIZE: usize = QuantizedWord::BITS as usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryQuantized {}

impl UnalignedVectorCodec for BinaryQuantized {
    fn from_bytes(bytes: &[u8]) -> Result<Cow<UnalignedVector<Self>>, SizeMismatch> {
        let rem = bytes.len() % size_of::<QuantizedWord>();
        if rem == 0 {
            // safety: `UnalignedVector` is transparent
            Ok(Cow::Borrowed(unsafe { transmute::<&[u8], &UnalignedVector<Self>>(bytes) }))
        } else {
            Err(SizeMismatch { vector_codec: "binary quantized", rem })
        }
    }

    fn from_slice(slice: &[f32]) -> Cow<'static, UnalignedVector<Self>> {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Cow::Owned(unsafe { from_slice_neon(slice) });
            }
        }
        Cow::Owned(from_slice_non_optimized(slice))
    }

    fn from_vec(vec: Vec<f32>) -> Cow<'static, UnalignedVector<Self>> {
        Cow::Owned(Self::from_slice(&vec).into_owned())
    }

    fn to_vec(vec: &UnalignedVector<Self>) -> Vec<f32> {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { to_vec_neon(vec) };
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") {
                return unsafe { to_vec_see(vec) };
            }
        }
        to_vec_non_optimized(vec)
    }

    fn iter(vec: &UnalignedVector<Self>) -> impl ExactSizeIterator<Item = f32> + '_ {
        BinaryQuantizedIterator {
            current_element: 0,
            // Force the pulling of the first word
            current_iteration: QUANTIZED_WORD_SIZE,
            iter: vec.vector.chunks_exact(size_of::<QuantizedWord>()),
        }
    }

    fn len(vec: &UnalignedVector<Self>) -> usize {
        (vec.vector.len() / size_of::<QuantizedWord>()) * QUANTIZED_WORD_SIZE
    }

    fn is_zero(vec: &UnalignedVector<Self>) -> bool {
        vec.as_bytes().iter().all(|b| *b == 0)
    }
}

fn from_slice_non_optimized(slice: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(slice.len() / QUANTIZED_WORD_SIZE);
    for chunk in slice.chunks(QUANTIZED_WORD_SIZE) {
        let mut word: QuantizedWord = 0;
        for scalar in chunk.iter().rev() {
            word <<= 1;
            word += scalar.is_sign_positive() as QuantizedWord;
        }
        output.extend_from_slice(&word.to_ne_bytes());
    }
    output
}

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
unsafe fn from_slice_neon(slice: &[f32]) -> Vec<u8> {
    use core::arch::aarch64::*;

    let iterations = slice.len() / size_of::<QuantizedWord>();
    // The size of the returned vector must be a multiple of a word
    let reminder = slice.len() % size_of::<QuantizedWord>();
    let mut len = iterations;
    if len % size_of::<QuantizedWord>() != 0 {
        len += size_of::<QuantizedWord>() - len % size_of::<QuantizedWord>();
    } else if reminder != 0 {
        // if we generated a valid number of Word but we're missing a few bits
        // then we need to add a full Word at the end.
        len += size_of::<QuantizedWord>();
    }
    let mut ret = vec![0; len];
    let ptr = slice.as_ptr();

    #[allow(clippy::needless_range_loop)]
    for i in 0..iterations {
        unsafe {
            let lane = vld1q_f32(ptr.add(i * 8));
            let lane = vcltzq_f32(lane);
            let lane = vmvnq_u32(lane);
            let mask: Vec<u32> = vec![
                0b_00000000_00000000_00000000_00000001,
                0b_00000000_00000000_00000000_00000010,
                0b_00000000_00000000_00000000_00000100,
                0b_00000000_00000000_00000000_00001000,
            ];
            let mask = vld1q_u32(mask.as_ptr());
            let lane = vandq_u32(lane, mask);

            let left = vaddvq_u32(lane) as u8;

            let lane = vld1q_f32(ptr.add(i * 8 + 4));
            let lane = vcltzq_f32(lane);
            let lane = vmvnq_u32(lane);
            let mask: Vec<u32> = vec![
                0b_00000000_00000000_00000000_00010000,
                0b_00000000_00000000_00000000_00100000,
                0b_00000000_00000000_00000000_01000000,
                0b_00000000_00000000_00000000_10000000,
            ];
            let mask = vld1q_u32(mask.as_ptr());
            let lane = vandq_u32(lane, mask);

            let right = vaddvq_u32(lane) as u8;

            ret[i] = left | right;
        }
    }

    // Since we're iterating on bytes two by two.
    // If we had a number of dimensions not dividible by 8 we may be
    // missing some bits in the last byte.
    let reminder = slice.len() % size_of::<QuantizedWord>();
    if reminder != 0 {
        let mut rem: u8 = 0;
        for r in slice[slice.len() - reminder..].iter().rev() {
            rem <<= 1;
            let r = r.is_sign_positive();
            rem |= r as u8;
        }
        ret[iterations] = rem;
    }

    ret
}

fn to_vec_non_optimized(vec: &UnalignedVector<BinaryQuantized>) -> Vec<f32> {
    vec.iter().collect()
}

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
unsafe fn to_vec_neon(vec: &UnalignedVector<BinaryQuantized>) -> Vec<f32> {
    use core::arch::aarch64::*;

    let mut output: Vec<f32> = vec![0.0; vec.len()];
    let output_ptr = output.as_mut_ptr();
    let bytes = vec.as_bytes();
    let low_mask = [0b_0000_0001, 0b_0000_0010, 0b_0000_0100, 0b_0000_1000];
    let high_mask = [0b_0001_0000, 0b_0010_0000, 0b_0100_0000, 0b_1000_0000];
    let ones = unsafe { vld1q_dup_f32(&1.0) };
    let minus = unsafe { vld1q_dup_f32(&-1.0) };

    for (current_byte, base) in bytes.iter().enumerate() {
        unsafe {
            let base = *base as u32;
            let base = vld1q_dup_u32(&base);
            for (i, mask) in [low_mask, high_mask].iter().enumerate() {
                let mask = vld1q_u32(mask.as_ptr());
                let mask = vandq_u32(base, mask);
                // 0xffffffff if equal to zero and 0x00000000 otherwise
                let mask = vceqzq_u32(mask);
                let lane = vbslq_f32(mask, minus, ones);
                let offset = output_ptr.add(current_byte * 8 + i * 4);
                vst1q_f32(offset, lane);
            }
        }
    }

    output
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "arm64ec")))]
unsafe fn to_vec_sse(vec: &UnalignedVector<BinaryQuantized>) -> Vec<f32> {
    use core::arch::x86_64::*;

    let mut output: Vec<f32> = vec![0.0; vec.len()];
    let output_ptr = output.as_mut_ptr();
    let bytes = vec.as_bytes();
    let low_mask = [0b_0000_0001, 0b_0000_0010, 0b_0000_0100, 0b_0000_1000];
    let high_mask = [0b_0001_0000, 0b_0010_0000, 0b_0100_0000, 0b_1000_0000];
    let ones = unsafe { _mm_set1_ps(1.0) };
    let minus = unsafe { _mm_set1_ps(-1.0) };

    for (current_byte, base) in bytes.iter().enumerate() {
        unsafe {
            let base = _mm_set1_epi32(*base as i32);
            for (i, mask) in [low_mask, high_mask].iter().enumerate() {
                let mask = _mm_set_epi32(mask[3], mask[2], mask[1], mask[0]);
                let mask = _mm_and_si128(base, mask);
                // 0xffffffff if equal to zero and 0x00000000 otherwise
                let mask = _mm_cmpeq_epi32(mask, _mm_setzero_si128());
                let lane = _mm_blendv_ps(ones, minus, _mm_castsi128_ps(mask));
                let offset = output_ptr.add(current_byte * 8 + i * 4);
                _mm_store_ps(offset, lane);
            }
        }
    }

    output
}

// Dedicated to mm256 (AVX). Doesn't provide any real perf gain.
// #[cfg(not(any(target_arch = "aarch64", target_arch = "arm64ec")))]
// unsafe fn to_vec_avx(vec: &UnalignedVector<BinaryQuantized>) -> Vec<f32> {
//     use core::arch::x86_64::*;

//     let mut output: Vec<f32> = vec![0.0; vec.len()];
//     let output_ptr = output.as_mut_ptr();
//     let bytes = vec.as_bytes();
//     let mask = [
//         0b_0000_0001,
//         0b_0000_0010,
//         0b_0000_0100,
//         0b_0000_1000,
//         0b_0001_0000,
//         0b_0010_0000,
//         0b_0100_0000,
//         0b_1000_0000,
//     ];
//     let ones = unsafe { _mm256_set1_ps(1.0) };
//     let minus = unsafe { _mm256_set1_ps(-1.0) };

//     for (current_byte, base) in bytes.iter().enumerate() {
//         unsafe {
//             let base = _mm256_set1_epi32(*base as i32);
//             let mask = _mm256_set_epi32(
//                 mask[7], mask[6], mask[5], mask[4], mask[3], mask[2], mask[1], mask[0],
//             );
//             let mask = _mm256_and_si256(base, mask);
//             // 0xffffffff if equal to zero and 0x00000000 otherwise
//             let mask = _mm256_cmpeq_epi32(mask, _mm256_setzero_si256());
//             let lane = _mm256_blendv_ps(ones, minus, _mm256_castsi256_ps(mask));
//             let offset = output_ptr.add(current_byte * 8);
//             // I don't understand why this is not aligned and I cannot use _mm256_store_ps
//             _mm256_storeu_ps(offset, lane);
//         }
//     }

//     output
// }

pub struct BinaryQuantizedIterator<'a> {
    current_element: QuantizedWord,
    current_iteration: usize,
    iter: ChunksExact<'a, u8>,
}

impl Iterator for BinaryQuantizedIterator<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_iteration >= QUANTIZED_WORD_SIZE {
            let bytes = self.iter.next()?;
            self.current_element = QuantizedWord::from_ne_bytes(bytes.try_into().unwrap());
            self.current_iteration = 0;
        }

        let bit = self.current_element & 1;
        self.current_element >>= 1;
        self.current_iteration += 1;

        Some(bit as f32 * 2.0 - 1.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.iter.size_hint();
        let rem = QUANTIZED_WORD_SIZE - self.current_iteration;

        (low * QUANTIZED_WORD_SIZE + rem, high.map(|h| h * QUANTIZED_WORD_SIZE + rem))
    }
}

impl ExactSizeIterator for BinaryQuantizedIterator<'_> {
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        debug_assert_eq!(upper, Some(lower));
        lower
    }
}

#[cfg(test)]
mod test {
    use insta::{assert_debug_snapshot, assert_snapshot};

    use super::*;
    use crate::internals::UnalignedVectorCodec;

    #[test]
    fn test_from_slice() {
        let original = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9];
        let vector = BinaryQuantized::from_slice(&original);

        let internal = vector.as_bytes().iter().map(|b| format!("{b:08b}\n")).collect::<String>();
        assert_snapshot!(internal, @r###"
        10101011
        00000000
        00000000
        00000000
        00000000
        00000000
        00000000
        00000000
        "###);

        let iter_vec: Vec<_> = BinaryQuantized::iter(&vector).take(original.len()).collect();
        assert_debug_snapshot!(iter_vec, @r###"
        [
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
        ]
        "###);
        let mut vec_vec: Vec<_> = BinaryQuantized::to_vec(&vector);
        vec_vec.truncate(original.len());
        assert_debug_snapshot!(vec_vec, @r###"
        [
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
        ]
        "###);

        assert_eq!(vec_vec, iter_vec);
    }

    #[test]
    fn super_truc() {
        let original = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9];
        let vector = BinaryQuantized::from_slice(&original);
        let iter_vec: Vec<_> = BinaryQuantized::iter(&vector).take(original.len()).collect();
        assert_debug_snapshot!(iter_vec, @r###"
        [
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
        ]
        "###);
        let mut vec_vec: Vec<_> = BinaryQuantized::to_vec(&vector);
        vec_vec.truncate(original.len());
        assert_debug_snapshot!(vec_vec, @r###"
        [
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
        ]
        "###);

        assert_eq!(vec_vec, iter_vec);
    }

    use proptest::collection::vec;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_truc_1(
            original in vec(-50f32..=50.2, 0..512)
        ){
            let vector = BinaryQuantized::from_slice(&original);
            let iter_vec: Vec<_> = to_vec_non_optimized(&vector);
            let vec_vec: Vec<_> = BinaryQuantized::to_vec(&vector);

            assert_eq!(vec_vec, iter_vec);
        }

        #[test]
        fn prop_truc_2(
            original in vec(-50f32..=50.2, 0..516)
        ){
            let vector1 = BinaryQuantized::from_slice(&original);
            let vector2 = from_slice_non_optimized(&original);

            assert_eq!(vector1.as_bytes(), &vector2);
        }
    }
}
