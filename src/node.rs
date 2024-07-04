use std::borrow::{Borrow, Cow};
use std::fmt;
use std::marker::PhantomData;
use std::mem::{size_of, transmute};

use bytemuck::{bytes_of, cast_slice, pod_collect_to_vec, pod_read_unaligned};
use byteorder::{ByteOrder, NativeEndian};
use heed::{BoxedError, BytesDecode, BytesEncode};
use roaring::RoaringBitmap;

use crate::distance::Distance;
use crate::{ItemId, NodeId};

#[derive(Clone)]
pub enum Node<'a, D: Distance> {
    Leaf(Leaf<'a, D>),
    Descendants(Descendants<'a>),
    SplitPlaneNormal(SplitPlaneNormal<'a>),
}

impl<D: Distance> fmt::Debug for Node<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Leaf(leaf) => f.debug_tuple("Leaf").field(&leaf).finish(),
            Node::Descendants(desc) => f.debug_tuple("Descendants").field(&desc).finish(),
            Node::SplitPlaneNormal(split) => f
                .debug_tuple("SplitPlaneNormal")
                .field(&DisplaySplitPlaneNormal::<D>(split, PhantomData))
                .finish(),
        }
    }
}

const LEAF_TAG: u8 = 0;
const DESCENDANTS_TAG: u8 = 1;
const SPLIT_PLANE_NORMAL_TAG: u8 = 2;

impl<'a, D: Distance> Node<'a, D> {
    pub fn leaf(self) -> Option<Leaf<'a, D>> {
        if let Node::Leaf(leaf) = self {
            Some(leaf)
        } else {
            None
        }
    }
}

/// Small structure used to implement `Debug` for the `Leaf` and the `SplitPlaneNormal`.
struct DisplayVec(Vec<f32>);
impl fmt::Debug for DisplayVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        self.0.iter().for_each(|float| {
            list.entry(&format_args!("{:.4?}", float));
        });
        list.finish()
    }
}

/// A leaf node which corresponds to the vector inputed
/// by the user and the distance header.
#[derive(Clone)]
pub struct Leaf<'a, D: Distance> {
    /// The header of this leaf.
    pub header: D::Header,
    /// The vector of this leaf.
    pub vector: Cow<'a, UnalignedVector>,
}

impl<D: Distance> fmt::Debug for Leaf<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vec = DisplayVec(D::read_unaligned_vector(&self.vector));
        f.debug_struct("Leaf").field("header", &self.header).field("vector", &vec).finish()
    }
}

impl<D: Distance> Leaf<'_, D> {
    /// Converts the leaf into an owned version of itself by cloning
    /// the internal vector. Doing so will make it mutable.
    pub fn into_owned(self) -> Leaf<'static, D> {
        Leaf { header: self.header, vector: Cow::Owned(self.vector.into_owned()) }
    }
}

/// A wrapper struct that is used to read unaligned vectors directly from memory.
#[repr(transparent)]
pub struct UnalignedVector([u8]);

/// The type of the words used to quantize a vector
type QuantizedWord = usize;
/// The size of the words used to quantize a vector
const QUANTIZED_WORD_SIZE: usize = QuantizedWord::BITS as usize;

impl UnalignedVector {
    /// Creates an unaligned slice of something. It's up to the caller to ensure
    /// it will be used with the same type it was created initially.
    pub(crate) fn reset(vector: &mut Cow<'_, UnalignedVector>) {
        match vector {
            Cow::Borrowed(slice) => *vector = Cow::Owned(vec![0; slice.as_bytes().len()]),
            Cow::Owned(bytes) => bytes.fill(0),
        }
    }
    /// Creates an unaligned slice of something. It's up to the caller to ensure
    /// it will be used with the same type it was created initially.
    pub(crate) fn from_bytes_unchecked(bytes: &[u8]) -> &Self {
        unsafe { transmute(bytes) }
    }

    /// Creates an unaligned slice of f32 wrapper from a slice of bytes.
    pub(crate) fn f32_vectors_from_bytes(bytes: &[u8]) -> Result<&Self, SizeMismatch> {
        if bytes.len() % size_of::<f32>() == 0 {
            // safety: `UnalignedF32Slice` is transparent
            Ok(unsafe { transmute(bytes) })
        } else {
            Err(SizeMismatch)
        }
    }

    /// Creates an unaligned slice of f32 wrapper from a slice of f32.
    /// The slice is already known to be of the right length.
    pub(crate) fn f32_vectors_from_f32_slice(slice: &[f32]) -> &Self {
        Self::f32_vectors_from_bytes(cast_slice(slice)).unwrap()
    }

    /// Creates an unaligned slice of f32 wrapper from a slice of f32.
    /// The slice is already known to be of the right length.
    pub(crate) fn owned_f32_vectors_from_f32_slice(vec: Vec<f32>) -> Cow<'static, Self> {
        let bytes = vec.into_iter().flat_map(|f| f.to_ne_bytes()).collect();
        Cow::Owned(bytes)
    }

    /// Creates a binary quantized wrapper from a slice of bytes.
    pub(crate) fn quantized_vectors_from_bytes(bytes: &[u8]) -> Result<&Self, SizeMismatch> {
        if bytes.len() % size_of::<QuantizedWord>() == 0 {
            // safety: `UnalignedF32Slice` is transparent
            Ok(unsafe { transmute(bytes) })
        } else {
            Err(SizeMismatch)
        }
    }

    /// Creates a binary quantized unaligned slice of bytes from a slice of f32.
    /// Will allocate.
    pub(crate) fn binary_quantized_vectors_from_slice(slice: &[f32]) -> Cow<'static, Self> {
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

    /// Returns the original raw slice of bytes.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Return the number of f32 that fits into this slice.
    pub(crate) fn f32_len(&self) -> usize {
        self.0.len() / size_of::<f32>()
    }

    /// Returns wether it is empty or not.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns an iterator of f32 that are read from the slice.
    /// The f32 are copied in memory and are therefore, aligned.
    pub(crate) fn iter_f32(&self) -> impl Iterator<Item = f32> + '_ {
        self.0.chunks_exact(size_of::<f32>()).map(NativeEndian::read_f32)
    }

    /// Returns an iterator of f32 that are read from the binary quantized slice.
    /// The f32 are copied in memory and are therefore, aligned.
    pub(crate) fn iter_binary_quantized(&self) -> impl Iterator<Item = f32> + '_ {
        self.0
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

    /// Returns the raw pointer to the start of this slice.
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
}

#[derive(Debug, thiserror::Error)]
#[error("invalid slice of float dimension")]
pub struct SizeMismatch;

impl ToOwned for UnalignedVector {
    type Owned = Vec<u8>;

    fn to_owned(&self) -> Self::Owned {
        pod_collect_to_vec(&self.0)
    }
}

impl Borrow<UnalignedVector> for Vec<u8> {
    fn borrow(&self) -> &UnalignedVector {
        UnalignedVector::from_bytes_unchecked(self)
    }
}

#[derive(Clone)]
pub struct Descendants<'a> {
    // A descendants node can only contains references to the leaf nodes.
    // We can get and store their ids directly without the `Mode`.
    pub descendants: Cow<'a, RoaringBitmap>,
}

impl fmt::Debug for Descendants<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let descendants = self.descendants.iter().collect::<Vec<_>>();
        f.debug_struct("Descendants").field("descendants", &descendants).finish()
    }
}

#[derive(Clone)]
pub struct ItemIds<'a> {
    bytes: &'a [u8],
}

impl<'a> ItemIds<'a> {
    pub fn from_slice(slice: &[u32]) -> ItemIds<'_> {
        ItemIds::from_bytes(cast_slice(slice))
    }

    pub fn from_bytes(bytes: &[u8]) -> ItemIds<'_> {
        ItemIds { bytes }
    }

    pub fn raw_bytes(&self) -> &[u8] {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len() / size_of::<ItemId>()
    }

    pub fn iter(&self) -> impl Iterator<Item = ItemId> + 'a {
        self.bytes.chunks_exact(size_of::<ItemId>()).map(NativeEndian::read_u32)
    }
}

impl fmt::Debug for ItemIds<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        self.iter().for_each(|integer| {
            list.entry(&integer);
        });
        list.finish()
    }
}

#[derive(Clone)]
pub struct SplitPlaneNormal<'a> {
    pub left: NodeId,
    pub right: NodeId,
    pub normal: Cow<'a, UnalignedVector>,
}

/// Wraps a `SplitPlaneNormal` with its distance type to display it.
/// The distance is required to be able to read the normal.
pub struct DisplaySplitPlaneNormal<'a, D: Distance>(&'a SplitPlaneNormal<'a>, PhantomData<D>);
impl<D: Distance> fmt::Debug for DisplaySplitPlaneNormal<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let normal = DisplayVec(D::read_unaligned_vector(&self.0.normal));
        f.debug_struct("SplitPlaneNormal")
            .field("left", &self.0.left)
            .field("right", &self.0.right)
            .field("normal", &normal)
            .finish()
    }
}

/// The codec used internally to encode and decode nodes.
pub struct NodeCodec<D>(D);

impl<'a, D: Distance> BytesEncode<'a> for NodeCodec<D> {
    type EItem = Node<'a, D>;

    fn bytes_encode(item: &Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let mut bytes = Vec::new();
        match item {
            Node::Leaf(Leaf { header, vector }) => {
                bytes.push(LEAF_TAG);
                bytes.extend_from_slice(bytes_of(header));
                bytes.extend_from_slice(vector.as_bytes());
            }
            Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                bytes.push(SPLIT_PLANE_NORMAL_TAG);
                bytes.extend_from_slice(&left.to_bytes());
                bytes.extend_from_slice(&right.to_bytes());
                bytes.extend_from_slice(normal.as_bytes());
            }
            Node::Descendants(Descendants { descendants }) => {
                bytes.push(DESCENDANTS_TAG);
                descendants.serialize_into(&mut bytes)?;
            }
        }
        Ok(Cow::Owned(bytes))
    }
}

impl<'a, D: Distance> BytesDecode<'a> for NodeCodec<D> {
    type DItem = Node<'a, D>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        match bytes {
            [LEAF_TAG, bytes @ ..] => {
                let (header_bytes, remaining) = bytes.split_at(size_of::<D::Header>());
                let header = pod_read_unaligned(header_bytes);
                let vector = D::craft_unaligned_vector_from_bytes(remaining)?;

                Ok(Node::Leaf(Leaf { header, vector }))
            }
            [SPLIT_PLANE_NORMAL_TAG, bytes @ ..] => {
                let (left, bytes) = NodeId::from_bytes(bytes);
                let (right, bytes) = NodeId::from_bytes(bytes);
                Ok(Node::SplitPlaneNormal(SplitPlaneNormal {
                    normal: D::craft_unaligned_vector_from_bytes(bytes)?,
                    left,
                    right,
                }))
            }
            [DESCENDANTS_TAG, bytes @ ..] => Ok(Node::Descendants(Descendants {
                descendants: Cow::Owned(RoaringBitmap::deserialize_from(bytes)?),
            })),
            unknown => panic!("What the fuck is an {unknown:?}"),
        }
    }
}
