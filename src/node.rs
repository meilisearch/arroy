use std::borrow::{Borrow, Cow};
use std::fmt;
use std::mem::{size_of, transmute};

use bytemuck::{bytes_of, cast_slice, pod_collect_to_vec, pod_read_unaligned};
use byteorder::{ByteOrder, NativeEndian};
use heed::{BoxedError, BytesDecode, BytesEncode};
use roaring::RoaringBitmap;

use crate::distance::Distance;
use crate::{ItemId, NodeId};

#[derive(Debug, Clone)]
pub enum Node<'a, D: Distance> {
    Leaf(Leaf<'a, D>),
    Descendants(Descendants<'a>),
    SplitPlaneNormal(SplitPlaneNormal<'a>),
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

/// A leaf node which corresponds to the vector inputed
/// by the user and the distance header.
#[derive(Debug, Clone)]
pub struct Leaf<'a, D: Distance> {
    /// The header of this leaf.
    pub header: D::Header,
    /// The vector of this leaf.
    pub vector: Cow<'a, UnalignedF32Slice>,
}

impl<D: Distance> Leaf<'_, D> {
    /// Converts the leaf into an owned version of itself by cloning
    /// the internal vector. Doing so will make it mutable.
    pub fn into_owned(self) -> Leaf<'static, D> {
        Leaf { header: self.header, vector: Cow::Owned(self.vector.into_owned()) }
    }
}

/// A wrapper struct that is used to read unaligned floats directly from memory.
#[repr(transparent)]
pub struct UnalignedF32Slice([u8]);

impl UnalignedF32Slice {
    /// Creates an unaligned slice of f32 wrapper from a slice of bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, SizeMismatch> {
        if bytes.len() % size_of::<f32>() == 0 {
            Ok(unsafe { transmute(bytes) })
        } else {
            Err(SizeMismatch)
        }
    }

    /// Creates an unaligned slice of f32 wrapper from a slice of f32.
    /// The slice is already known to be of the right length.
    pub fn from_slice(slice: &[f32]) -> &Self {
        Self::from_bytes(cast_slice(slice)).unwrap()
    }

    /// Returns the original raw slice of bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Return the number of f32 that fits into this slice.
    pub fn len(&self) -> usize {
        self.0.len() / size_of::<f32>()
    }

    /// Returns wether it is empty or not.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns an iterator of f32 that are read from the slice.
    /// The f32 are copied in memory and are therefore, aligned.
    #[allow(clippy::needless_lifetimes)]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        self.0.chunks_exact(size_of::<f32>()).map(NativeEndian::read_f32)
    }

    /// Returns the raw pointer to the start of this slice.
    pub fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
}

#[derive(Debug, thiserror::Error)]
#[error("invalid slice of float dimension")]
pub struct SizeMismatch;

impl ToOwned for UnalignedF32Slice {
    type Owned = Vec<f32>;

    fn to_owned(&self) -> Self::Owned {
        pod_collect_to_vec(&self.0)
    }
}

impl Borrow<UnalignedF32Slice> for Vec<f32> {
    fn borrow(&self) -> &UnalignedF32Slice {
        UnalignedF32Slice::from_slice(&self[..])
    }
}

impl fmt::Debug for UnalignedF32Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct SmallF32(f32);
        impl fmt::Debug for SmallF32 {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_fmt(format_args!("{:.4?}", self.0))
            }
        }

        let mut list = f.debug_list();
        self.iter().for_each(|float| {
            list.entry(&SmallF32(float));
        });
        list.finish()
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

#[derive(Debug, Clone)]
pub struct SplitPlaneNormal<'a> {
    pub normal: Cow<'a, UnalignedF32Slice>,
    pub descendants: Cow<'a, RoaringBitmap>,

    pub left: NodeId,
    pub right: NodeId,
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
            Node::SplitPlaneNormal(SplitPlaneNormal { normal, descendants, left, right }) => {
                bytes.push(SPLIT_PLANE_NORMAL_TAG);
                bytes.extend_from_slice(&left.to_bytes());
                bytes.extend_from_slice(&right.to_bytes());
                descendants.serialize_into(&mut bytes)?;
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
                let vector = UnalignedF32Slice::from_bytes(remaining).map(Cow::Borrowed)?;
                Ok(Node::Leaf(Leaf { header, vector }))
            }
            [SPLIT_PLANE_NORMAL_TAG, bytes @ ..] => {
                let (left, bytes) = NodeId::from_bytes(bytes);
                let (right, bytes) = NodeId::from_bytes(bytes);
                let descendants: Cow<'static, RoaringBitmap> =
                    Cow::Owned(RoaringBitmap::deserialize_from(bytes)?);
                let bytes = &bytes[descendants.serialized_size()..];

                Ok(Node::SplitPlaneNormal(SplitPlaneNormal {
                    left,
                    right,
                    descendants,
                    normal: UnalignedF32Slice::from_bytes(bytes).map(Cow::Borrowed)?,
                }))
            }
            [DESCENDANTS_TAG, bytes @ ..] => Ok(Node::Descendants(Descendants {
                descendants: Cow::Owned(RoaringBitmap::deserialize_from(bytes)?),
            })),
            unknown => panic!("What the fuck is an {unknown:?}"),
        }
    }
}
