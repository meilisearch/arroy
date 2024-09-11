use std::{
    borrow::Cow,
    mem::{size_of, transmute},
    slice::ChunksExact,
};

use ordered_float::Float;

use super::{SizeMismatch, UnalignedVector, UnalignedVectorCodec};

/// The type of the words used to quantize a vector
type QuantizedWord = u64;
/// The size of the words used to quantize a vector
const QUANTIZED_WORD_SIZE: usize = QuantizedWord::BITS as usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryQuantized {}

impl UnalignedVectorCodec for BinaryQuantized {
    fn from_bytes(bytes: &[u8]) -> Result<Cow<UnalignedVector<Self>>, SizeMismatch> {
        // let rem = bytes.len() % size_of::<QuantizedWord>();
        let rem = 0;
        if rem == 0 {
            // safety: `UnalignedVector` is transparent
            Ok(Cow::Borrowed(unsafe { transmute(bytes) }))
        } else {
            Err(SizeMismatch { vector_codec: "binary quantized", rem })
        }
    }

    fn from_slice(slice: &[f32]) -> Cow<'static, UnalignedVector<Self>> {
        let output = unsafe { from_slice_simd(slice) };

        Cow::Owned(output)
    }

    fn from_vec(vec: Vec<f32>) -> Cow<'static, UnalignedVector<Self>> {
        Cow::Owned(Self::from_slice(&vec).into_owned())
    }

    fn to_vec(vec: &UnalignedVector<Self>) -> Vec<f32> {
        unsafe { to_vec_simd(vec) }
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

unsafe fn from_slice_simd(slice: &[f32]) -> Vec<u8> {
    println!("from slice called with {slice:?}");
    use core::arch::aarch64::*;

    let iterations = slice.len() / 8;
    let reminder = slice.len() % 8;
    let mut ret = Vec::with_capacity(iterations + (reminder != 0) as usize);

    let ptr = slice.as_ptr();

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

            ret.push(left | right);
        }
    }

    let mut rem: QuantizedWord = 0;
    for r in slice[slice.len() - reminder..].iter().rev() {
        rem <<= 1;
        let r = r.is_sign_positive();
        rem |= r as QuantizedWord;
    }
    ret.extend(rem.to_ne_bytes());

    println!("from slice returns {ret:?}");
    ret
}

unsafe fn to_vec_simd(vec: &UnalignedVector<BinaryQuantized>) -> Vec<f32> {
    use core::arch::aarch64::*;

    let mut output: Vec<f32> = vec![0.0; vec.len()];
    let output_ptr = output.as_mut_ptr();
    let bytes = vec.as_bytes();

    for (current_byte, base) in bytes.iter().enumerate() {
        let base = *base as u32;
        let low_mask = [0b_0000_0001, 0b_0000_0010, 0b_0000_0100, 0b_0000_1000];
        let high_mask = [0b_0001_0000, 0b_0010_0000, 0b_0100_0000, 0b_1000_0000];

        for (i, mask) in [low_mask, high_mask].iter().enumerate() {
            unsafe {
                let lane = vld1q_dup_u32(&base as *const u32);
                let lane = vandq_u32(lane, vld1q_u32(mask.as_ptr()));
                let lane = vceqzq_u32(lane);
                // Make the exponent right (either 1 or -1)
                //             sign exponent mantissa
                let mask: u32 = 0b0_01111111_00000000000000000000000;
                let lane = vorrq_u32(lane, vld1q_dup_u32(&mask as *const u32));
                //             sign exponent mantissa
                let mask: u32 = 0b1_01111111_00000000000000000000000;
                let lane = vandq_u32(lane, vld1q_dup_u32(&mask as *const u32));
                let lane = vreinterpretq_f32_u32(lane);
                vst1q_f32(output_ptr.add(current_byte * 8 + i * 4), lane);
            }
        }
    }

    output
}

pub struct BinaryQuantizedIterator<'a> {
    current_element: usize,
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
