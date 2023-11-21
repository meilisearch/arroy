use std::borrow::Cow;
use std::mem::size_of;

use bytemuck::{bytes_of, cast_slice, pod_read_unaligned};
use byteorder::{BigEndian, ByteOrder, NativeEndian};
use heed::{BoxedError, BytesDecode, BytesEncode};

use crate::{aligned_or_collect_vec, Distance, NodeId};

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

#[derive(Debug, Clone)]
pub struct Leaf<'a, D: Distance> {
    pub header: D::Header,
    pub vector: Cow<'a, [f32]>,
}

impl<D: Distance> Leaf<'_, D> {
    pub fn into_owned(self) -> Leaf<'static, D> {
        Leaf { header: self.header, vector: Cow::Owned(self.vector.into_owned()) }
    }
}

#[derive(Debug, Clone)]
pub struct Descendants<'a> {
    pub descendants: NodeIds<'a>,
}

#[derive(Debug, Clone)]
pub struct NodeIds<'a> {
    bytes: &'a [u8],
}

impl<'a> NodeIds<'a> {
    pub fn from_slice(slice: &[u32]) -> NodeIds<'_> {
        NodeIds::from_bytes(cast_slice(slice))
    }

    pub fn from_bytes(bytes: &[u8]) -> NodeIds<'_> {
        NodeIds { bytes }
    }

    pub fn raw_bytes(&self) -> &[u8] {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len() / size_of::<NodeId>()
    }

    pub fn iter(&self) -> impl Iterator<Item = NodeId> + 'a {
        self.bytes.chunks_exact(size_of::<NodeId>()).map(NativeEndian::read_u32)
    }
}

#[derive(Debug, Clone)]
pub struct SplitPlaneNormal<'a> {
    pub normal: Cow<'a, [f32]>,
    pub left: NodeId,
    pub right: NodeId,
}

pub struct NodeCodec<D>(D);

impl<'a, D: Distance> BytesEncode<'a> for NodeCodec<D> {
    type EItem = Node<'a, D>;

    fn bytes_encode(item: &Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let mut bytes = Vec::new();
        match item {
            Node::Leaf(Leaf { header, vector }) => {
                bytes.push(LEAF_TAG);
                bytes.extend_from_slice(bytes_of(header));
                bytes.extend_from_slice(cast_slice(vector));
            }
            Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }) => {
                bytes.push(SPLIT_PLANE_NORMAL_TAG);
                bytes.extend_from_slice(&left.to_be_bytes());
                bytes.extend_from_slice(&right.to_be_bytes());
                bytes.extend_from_slice(cast_slice(normal));
            }
            Node::Descendants(Descendants { descendants }) => {
                bytes.push(DESCENDANTS_TAG);
                bytes.extend_from_slice(descendants.raw_bytes());
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
                let vector = aligned_or_collect_vec(remaining);
                Ok(Node::Leaf(Leaf { header, vector }))
            }
            [SPLIT_PLANE_NORMAL_TAG, bytes @ ..] => {
                let left = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                let right = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                Ok(Node::SplitPlaneNormal(SplitPlaneNormal {
                    normal: aligned_or_collect_vec(bytes),
                    left,
                    right,
                }))
            }
            [DESCENDANTS_TAG, bytes @ ..] => {
                Ok(Node::Descendants(Descendants { descendants: NodeIds::from_bytes(bytes) }))
            }
            unknown => panic!("What the fuck is an {unknown:?}"),
        }
    }
}
