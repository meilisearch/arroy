use std::borrow::Cow;
use std::fmt;
use std::mem::size_of;

use bytemuck::{bytes_of, cast_slice, pod_read_unaligned};
use byteorder::{BigEndian, ByteOrder, NativeEndian};
use heed::{BoxedError, BytesDecode, BytesEncode};
use roaring::RoaringBitmap;

use crate::distance::Distance;
use crate::node_id::NodeId;
use crate::unaligned_vector::UnalignedVector;
use crate::ItemId;

#[derive(Clone, Debug)]
pub enum Node<'a, D: Distance> {
    Leaf(Leaf<'a, D>),
    Descendants(Descendants<'a>),
    SplitPlaneNormal(SplitPlaneNormal<'a, D>),
}

/// A node generic over the version of the database.
/// Should only be used while reading from the database.
#[derive(Clone, Debug)]
pub enum GenericReadNode<'a, D: Distance> {
    Leaf(Leaf<'a, D>),
    Descendants(Descendants<'a>),
    SplitPlaneNormal(GenericReadSplitPlaneNormal<'a, D>),
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
pub struct Leaf<'a, D: Distance> {
    /// The header of this leaf.
    pub header: D::Header,
    /// The vector of this leaf.
    pub vector: Cow<'a, UnalignedVector<D::VectorCodec>>,
}

impl<D: Distance> fmt::Debug for Leaf<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Leaf").field("header", &self.header).field("vector", &self.vector).finish()
    }
}

impl<D: Distance> Clone for Leaf<'_, D> {
    fn clone(&self) -> Self {
        Self { header: self.header, vector: self.vector.clone() }
    }
}

impl<D: Distance> Leaf<'_, D> {
    /// Converts the leaf into an owned version of itself by cloning
    /// the internal vector. Doing so will make it mutable.
    pub fn into_owned(self) -> Leaf<'static, D> {
        Leaf { header: self.header, vector: Cow::Owned(self.vector.into_owned()) }
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

pub struct SplitPlaneNormal<'a, D: Distance> {
    pub left: ItemId,
    pub right: ItemId,
    pub normal: Option<Leaf<'a, D>>,
}

impl<D: Distance> fmt::Debug for SplitPlaneNormal<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = format!("SplitPlaneNormal<{}>", D::name());
        let mut debug = f.debug_struct(&name);

        debug.field("left", &self.left).field("right", &self.right);
        match &self.normal {
            Some(normal) => debug.field("normal", &normal),
            None => debug.field("normal", &"none"),
        };
        debug.finish()
    }
}

impl<D: Distance> Clone for SplitPlaneNormal<'_, D> {
    fn clone(&self) -> Self {
        Self { left: self.left, right: self.right, normal: self.normal.clone() }
    }
}

pub struct GenericReadSplitPlaneNormal<'a, D: Distance> {
    // Before version 0.7.0 the split plane normal was stored as a `NodeId` and could point directly to items.
    pub left: NodeId,
    pub right: NodeId,
    // Before version 0.7.0 instead of storing `None` for a missing normal, we were
    // storing a vector filled with zeros, that will be overwritten while creating this type.
    pub normal: Option<Leaf<'a, D>>,
}

impl<D: Distance> fmt::Debug for GenericReadSplitPlaneNormal<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = format!("GenericReadSplitPlaneNormal<{}>", D::name());
        let mut debug = f.debug_struct(&name);

        debug.field("left", &self.left).field("right", &self.right);
        match &self.normal {
            Some(normal) => debug.field("normal", &normal),
            None => debug.field("normal", &"none"),
        };
        debug.finish()
    }
}

impl<D: Distance> Clone for GenericReadSplitPlaneNormal<'_, D> {
    fn clone(&self) -> Self {
        Self { left: self.left, right: self.right, normal: self.normal.clone() }
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
                bytes.extend_from_slice(&left.to_be_bytes());
                bytes.extend_from_slice(&right.to_be_bytes());
                if let Some(normal) = normal {
                    bytes.extend_from_slice(bytes_of(&normal.header));
                    bytes.extend_from_slice(normal.vector.as_bytes());
                }
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
                let vector = UnalignedVector::<D::VectorCodec>::from_bytes(remaining)?;

                Ok(Node::Leaf(Leaf { header, vector }))
            }
            [SPLIT_PLANE_NORMAL_TAG, bytes @ ..] => {
                let left = BigEndian::read_u32(bytes);
                let bytes = &bytes[std::mem::size_of_val(&left)..];
                let right = BigEndian::read_u32(bytes);
                let bytes = &bytes[std::mem::size_of_val(&right)..];
                let normal = if bytes.is_empty() {
                    None
                } else {
                    let (header_bytes, remaining) = bytes.split_at(size_of::<D::Header>());
                    let header = pod_read_unaligned::<D::Header>(header_bytes);
                    let vector = UnalignedVector::<D::VectorCodec>::from_bytes(remaining)?;
                    Some(Leaf { header, vector })
                };
                Ok(Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }))
            }
            [DESCENDANTS_TAG, bytes @ ..] => Ok(Node::Descendants(Descendants {
                descendants: Cow::Owned(RoaringBitmap::deserialize_from(bytes)?),
            })),
            unknown => panic!(
                "Did not recognize node tag type: {unknown:?} while decoding a node from v0.7.0"
            ),
        }
    }
}

/// The codec used internally during read operations to decode nodes to a common interface from the v0.4.0.
pub struct GenericReadNodeCodecFromV0_4_0<D>(D);

impl<'a, D: Distance> BytesDecode<'a> for GenericReadNodeCodecFromV0_4_0<D> {
    type DItem = GenericReadNode<'a, D>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        match bytes {
            [LEAF_TAG, bytes @ ..] => {
                let (header_bytes, remaining) = bytes.split_at(size_of::<D::Header>());
                let header = pod_read_unaligned(header_bytes);
                let vector = UnalignedVector::<D::VectorCodec>::from_bytes(remaining)?;

                Ok(GenericReadNode::Leaf(Leaf { header, vector }))
            }
            [SPLIT_PLANE_NORMAL_TAG, bytes @ ..] => {
                // From v0.4.0 to v0.5.0 included, the children were stored as `NodeId` and could point directly to items.
                let (left, bytes) = NodeId::from_bytes(bytes);
                let (right, bytes) = NodeId::from_bytes(bytes);
                // And the normal could not be null, but it could be a vector filled with zeros.
                let vector = UnalignedVector::<D::VectorCodec>::from_bytes(bytes)?;
                let normal = if vector.is_zero() {
                    None
                } else {
                    let header = D::new_header(&vector);
                    Some(Leaf { header, vector })
                };
                Ok(GenericReadNode::SplitPlaneNormal(GenericReadSplitPlaneNormal { normal, left, right }))
            }
            [DESCENDANTS_TAG, bytes @ ..] => Ok(GenericReadNode::Descendants(Descendants {
                descendants: Cow::Owned(RoaringBitmap::deserialize_from(bytes)?),
            })),
            unknown => panic!("Did not recognize node tag type: {unknown:?} while decoding a generic read node from v0.4.0"),
        }
    }
}

/// The codec used internally during read operations to decode nodes to a common interface from the v0.7.0.
pub struct GenericReadNodeCodecFromV0_7_0<D>(D);

impl<'a, D: Distance> BytesDecode<'a> for GenericReadNodeCodecFromV0_7_0<D> {
    type DItem = GenericReadNode<'a, D>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        NodeCodec::bytes_decode(bytes).map(|node| match node {
            Node::SplitPlaneNormal(split_plane_normal) => {
                GenericReadNode::SplitPlaneNormal(GenericReadSplitPlaneNormal {
                    // From v0.6.0 the split plane normal always points to a tree node.
                    left: NodeId::tree(split_plane_normal.left),
                    right: NodeId::tree(split_plane_normal.right),
                    normal: split_plane_normal.normal,
                })
            }
            Node::Descendants(descendants) => GenericReadNode::Descendants(descendants),
            Node::Leaf(leaf) => GenericReadNode::Leaf(leaf),
        })
    }
}

/// The codec used internally during read operations to decode nodes to a common interface from the v0.4.0.
pub struct WriteNodeCodecForV0_5_0<D>(D);

impl<'a, D: Distance> BytesEncode<'a> for WriteNodeCodecForV0_5_0<D> {
    // Since the dimension of the vector has been lost while converting to a generic node, we need to get it back.
    type EItem = (GenericReadNode<'a, D>, usize);

    fn bytes_encode(item: &Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        // It's ok to clone and be slow because that only happens once when upgrading from v0.4.0 to v0.5.0.
        match &item.0 {
            // The leaf didn't change between v0.4.0 and today.
            GenericReadNode::Leaf(leaf) => {
                Ok(NodeCodec::bytes_encode(&Node::Leaf(leaf.clone()))?.into_owned().into())
            }
            // The descendants didn't change between v0.4.0 and today.
            GenericReadNode::Descendants(descendants) => {
                Ok(NodeCodec::bytes_encode(&Node::<D>::Descendants(descendants.clone()))?
                    .into_owned()
                    .into())
            }
            GenericReadNode::SplitPlaneNormal(GenericReadSplitPlaneNormal {
                left,
                right,
                normal,
            }) => {
                // Original code at: https://github.com/meilisearch/arroy/blob/5b748bac2c69c65a97980901b02067a3a545e357/src/node.rs#L152-L157
                let mut bytes = Vec::new();
                bytes.push(SPLIT_PLANE_NORMAL_TAG);
                bytes.extend_from_slice(&left.to_bytes());
                bytes.extend_from_slice(&right.to_bytes());
                match normal {
                    Some(normal) => bytes.extend_from_slice(normal.vector.as_bytes()),
                    // If the normal is None, we need to write a vector filled with zeros.
                    None => bytes.extend_from_slice(&vec![0; item.1]),
                }
                Ok(Cow::Owned(bytes))
            }
        }
    }
}
