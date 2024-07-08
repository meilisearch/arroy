use std::{
    borrow::Cow,
    mem::{size_of, transmute},
    slice::ChunksExact,
};

use super::{SizeMismatch, UnalignedVector, UnalignedVectorCodec};

/// The type of the words used to quantize a vector
type QuantizedWord = usize;
/// The size of the words used to quantize a vector
const QUANTIZED_WORD_SIZE: usize = QuantizedWord::BITS as usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryQuantized {}

impl UnalignedVectorCodec for BinaryQuantized {
    fn from_bytes(bytes: &[u8]) -> Result<Cow<UnalignedVector<Self>>, SizeMismatch> {
        let rem = bytes.len() % size_of::<QuantizedWord>();
        if rem == 0 {
            // safety: `UnalignedVector` is transparent
            Ok(Cow::Borrowed(unsafe { transmute(bytes) }))
        } else {
            Err(SizeMismatch { vector_codec: "binary quantized", rem })
        }
    }

    fn from_slice(slice: &[f32]) -> Cow<'static, UnalignedVector<Self>> {
        let mut output: Vec<u8> = Vec::with_capacity(slice.len() / QUANTIZED_WORD_SIZE);
        for chunk in slice.chunks(QUANTIZED_WORD_SIZE) {
            let mut word: QuantizedWord = 0;
            for bit in chunk.iter().rev() {
                word <<= 1;
                word += bit.is_sign_positive() as QuantizedWord;
            }
            output.extend_from_slice(&word.to_ne_bytes());
        }

        Cow::Owned(output)
    }

    fn from_vec(vec: Vec<f32>) -> Cow<'static, UnalignedVector<Self>> {
        Cow::Owned(Self::from_slice(&vec).into_owned())
    }

    fn iter(vec: &UnalignedVector<Self>) -> impl Iterator<Item = f32> + '_ {
        BinaryQuantizedIterator {
            current_element: 0,
            // Force the pulling of the first word
            current_iteration: QUANTIZED_WORD_SIZE,
            iter: vec.vector.chunks_exact(size_of::<QuantizedWord>()),
        }
    }

    fn len(vec: &UnalignedVector<Self>) -> usize {
        vec.vector.len() / size_of::<QuantizedWord>()
    }
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

        Some(bit as f32)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.iter.size_hint();
        let rem = QUANTIZED_WORD_SIZE - self.current_iteration;

        (low + rem, high.map(|h| h + rem))
    }
}

impl ExactSizeIterator for BinaryQuantizedIterator<'_> {
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        debug_assert_eq!(upper, Some(lower));
        lower
    }
}
