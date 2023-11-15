use std::borrow::Cow;
use std::error::Error;
use std::mem::size_of;

use bytemuck::{bytes_of, cast_slice, pod_collect_to_vec, pod_read_unaligned};
use byteorder::{BigEndian, ByteOrder};
use heed::{BytesDecode, BytesEncode};

use crate::{Distance, NodeId};

#[derive(Debug, Clone)]
pub enum Node<D: Distance> {
    Leaf(Leaf<D>),
    Descendants(Descendants),
    SplitPlaneNormal(SplitPlaneNormal),
}

const LEAF_TAG: u8 = 0;
const DESCENDANTS_TAG: u8 = 1;
const SPLIT_PLANE_NORMAL_TAG: u8 = 2;

impl<D: Distance> Node<D> {
    pub fn leaf(self) -> Option<Leaf<D>> {
        if let Node::Leaf(leaf) = self {
            Some(leaf)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Leaf<D: Distance> {
    pub header: D::Header,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Descendants {
    pub descendants: Vec<NodeId>,
}

#[derive(Debug, Clone)]
pub struct SplitPlaneNormal {
    pub normal: Vec<f32>,
    pub left: NodeId,
    pub right: NodeId,
}

pub struct NodeCodec<D>(D);

impl<'a, D: Distance + 'a> BytesEncode<'a> for NodeCodec<D> {
    type EItem = Node<D>;

    fn bytes_encode(item: &Self::EItem) -> Result<Cow<'a, [u8]>, Box<dyn Error + Send + Sync>> {
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
                bytes.extend_from_slice(cast_slice(descendants));
            }
        }
        Ok(Cow::Owned(bytes))
    }
}

impl<'a, D: Distance + 'a> BytesDecode<'a> for NodeCodec<D> {
    type DItem = Node<D>;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, Box<dyn Error + Send + Sync>> {
        match bytes {
            [LEAF_TAG, bytes @ ..] => {
                let (header_bytes, remaining) = bytes.split_at(size_of::<D::Header>());
                let header = pod_read_unaligned(header_bytes);
                let vector = pod_collect_to_vec(remaining);
                Ok(Node::Leaf(Leaf { header, vector }))
            }
            [SPLIT_PLANE_NORMAL_TAG, bytes @ ..] => {
                let left = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                let right = BigEndian::read_u32(bytes);
                let bytes = &bytes[size_of::<u32>()..];
                Ok(Node::SplitPlaneNormal(SplitPlaneNormal {
                    normal: pod_collect_to_vec(bytes),
                    left,
                    right,
                }))
            }
            [DESCENDANTS_TAG, bytes @ ..] => {
                Ok(Node::Descendants(Descendants { descendants: pod_collect_to_vec(bytes) }))
            }
            unknown => panic!("What the fuck is an {unknown:?}"),
        }
    }
}
