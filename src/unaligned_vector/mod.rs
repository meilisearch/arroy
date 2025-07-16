use std::{
    borrow::{Borrow, Cow},
    fmt,
    marker::PhantomData,
    mem::transmute,
};

pub use binary::Binary;
pub use binary_quantized::BinaryQuantized;

use bytemuck::pod_collect_to_vec;

mod binary;
mod binary_quantized;
mod f32;

#[cfg(test)]
mod binary_quantized_test;

#[cfg(test)]
mod binary_test;

/// Determine the way the vectors should be read and written from the database
pub trait UnalignedVectorCodec: std::borrow::ToOwned + Sized {
    /// Creates an unaligned vector from a slice of bytes.
    /// Don't allocate.
    fn from_bytes(bytes: &[u8]) -> Result<Cow<UnalignedVector<Self>>, SizeMismatch>;

    /// Creates an unaligned vector from a slice of f32.
    /// May allocate depending on the codec.
    fn from_slice(slice: &[f32]) -> Cow<UnalignedVector<Self>>;

    /// Creates an unaligned slice of f32 wrapper from a slice of f32.
    /// The slice is already known to be of the right length.
    fn from_vec(vec: Vec<f32>) -> Cow<'static, UnalignedVector<Self>>;

    /// Converts the `UnalignedVector` to an aligned vector of `f32`.
    /// It's strictly equivalent to `.iter().collect()` but the performances
    /// are better.
    fn to_vec(vec: &UnalignedVector<Self>) -> Vec<f32>;

    /// Returns an iterator of f32 that are read from the vector.
    /// The f32 are copied in memory and are therefore, aligned.
    fn iter(vec: &UnalignedVector<Self>) -> impl ExactSizeIterator<Item = f32> + '_;

    /// Returns the len of the vector in terms of elements.
    fn len(vec: &UnalignedVector<Self>) -> usize;

    /// Returns true if all the elements in the vector are equal to 0.
    fn is_zero(vec: &UnalignedVector<Self>) -> bool;

    /// Returns the size of an item in bytes.
    fn size_of_item(dimensions: usize) -> usize;
}

/// A wrapper struct that is used to read unaligned vectors directly from memory.
#[repr(transparent)]
pub struct UnalignedVector<Codec: UnalignedVectorCodec> {
    format: PhantomData<fn() -> Codec>,
    vector: [u8],
}

impl<Codec: UnalignedVectorCodec> UnalignedVector<Codec> {
    /// Creates an unaligned vector from a slice of bytes.
    /// Don't allocate.
    pub fn from_bytes(bytes: &[u8]) -> Result<Cow<UnalignedVector<Codec>>, SizeMismatch> {
        Codec::from_bytes(bytes)
    }

    /// Creates an unaligned vector from a slice of f32.
    /// May allocate depending on the codec.
    pub fn from_slice(slice: &[f32]) -> Cow<UnalignedVector<Codec>> {
        Codec::from_slice(slice)
    }

    /// Creates an unaligned slice of f32 wrapper from a slice of f32.
    /// The slice is already known to be of the right length.
    pub fn from_vec(vec: Vec<f32>) -> Cow<'static, UnalignedVector<Codec>> {
        Codec::from_vec(vec)
    }

    /// Returns an iterator of f32 that are read from the vector.
    /// The f32 are copied in memory and are therefore, aligned.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        Codec::iter(self)
    }

    /// Returns true if all the elements in the vector are equal to 0.
    pub fn is_zero(&self) -> bool {
        Codec::is_zero(self)
    }

    /// Returns an allocated and aligned `Vec<f32>`.
    pub fn to_vec(&self) -> Vec<f32> {
        Codec::to_vec(self)
    }

    /// Returns the len of the vector in terms of elements.
    pub fn len(&self) -> usize {
        Codec::len(self)
    }

    /// Creates an unaligned slice of something. It's up to the caller to ensure
    /// it will be used with the same type it was created initially.
    pub(crate) fn from_bytes_unchecked(bytes: &[u8]) -> &Self {
        unsafe { transmute(bytes) }
    }

    /// Returns the original raw slice of bytes.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.vector
    }

    /// Returns wether it is empty or not.
    pub fn is_empty(&self) -> bool {
        self.vector.is_empty()
    }
    /// Returns the raw pointer to the start of this slice.
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.vector.as_ptr()
    }
}

/// Returned in case you tried to make an unaligned vector from a slice of bytes that don't have the right number of elements
#[derive(Debug, thiserror::Error)]
#[error(
    "Slice of bytes contains {rem} too many bytes to be decoded with the {vector_codec} codec."
)]
pub struct SizeMismatch {
    /// The name of the codec used.
    vector_codec: &'static str,
    /// The number of bytes remaining after decoding as many words as possible.
    rem: usize,
}

impl<Codec: UnalignedVectorCodec> ToOwned for UnalignedVector<Codec> {
    type Owned = Vec<u8>;

    fn to_owned(&self) -> Self::Owned {
        pod_collect_to_vec(&self.vector)
    }
}

impl<Codec: UnalignedVectorCodec> Borrow<UnalignedVector<Codec>> for Vec<u8> {
    fn borrow(&self) -> &UnalignedVector<Codec> {
        UnalignedVector::from_bytes_unchecked(self)
    }
}

impl<Codec: UnalignedVectorCodec> fmt::Debug for UnalignedVector<Codec> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();

        struct Number(f32);
        impl fmt::Debug for Number {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:0.4}", self.0)
            }
        }

        let vec = self.to_vec();
        for v in vec.iter().take(10) {
            list.entry(&Number(*v));
        }
        if vec.len() < 10 {
            return list.finish();
        }

        // With binary quantization we may be padding with a lot of zeros
        if vec[10..].iter().all(|v| *v == 0.0) {
            list.entry(&"0.0, ...");
        } else {
            list.entry(&"other ...");
        }

        list.finish()
    }
}
