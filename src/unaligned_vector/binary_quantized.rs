use std::{
    borrow::Cow,
    mem::{size_of, transmute},
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
        vec.vector
            .chunks_exact(size_of::<QuantizedWord>())
            .map(|bytes| QuantizedWord::from_ne_bytes(bytes.try_into().unwrap()))
            .flat_map(|mut word| {
                let mut ret = vec![0.0; QUANTIZED_WORD_SIZE];
                for index in 0..QUANTIZED_WORD_SIZE {
                    let bit = word & 1;
                    word >>= 1;
                    if bit == 1 {
                        ret[index] = 1.0;
                    }
                }
                ret
            })
    }

    fn len(vec: &UnalignedVector<Self>) -> usize {
        vec.vector.len() / size_of::<QuantizedWord>()
    }
}
