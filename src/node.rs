use std::mem;

use crate::DistanceType;
use bytemuck::{Pod, Zeroable};

pub(crate) struct Node {
    pub offset: usize,
    pub header: NodeHeader,
}

impl Node {
    pub fn new_with_id(
        id: usize,
        node_size: usize,
        index_type: DistanceType,
        storage: &[u8],
    ) -> Node {
        let offset = id * node_size;
        Node {
            offset,
            header: NodeHeader::new(offset, index_type, storage),
        }
    }
}

#[repr(C)]
pub(crate) enum NodeHeader {
    Angular(NodeHeaderAngular),
    Minkowski(NodeHeaderMinkowski),
    Dot(NodeHeaderDot),
}

impl NodeHeader {
    pub fn new(offset: usize, distance_type: DistanceType, storage: &[u8]) -> NodeHeader {
        match distance_type {
            DistanceType::Angular => NodeHeader::Angular(NodeHeaderAngular::read(storage, offset)),
            DistanceType::Euclidean | DistanceType::Manhattan => {
                NodeHeader::Minkowski(NodeHeaderMinkowski::read(storage, offset))
            }
            DistanceType::Dot => NodeHeader::Dot(NodeHeaderDot::read(storage, offset)),
        }
    }

    pub fn get_n_descendant(&self) -> i32 {
        match self {
            NodeHeader::Angular(h) => h.n_descendants,
            NodeHeader::Minkowski(h) => h.n_descendants,
            NodeHeader::Dot(h) => h.n_descendants,
        }
    }

    pub fn get_children_id_slice(&self) -> [i32; 2] {
        match self {
            NodeHeader::Angular(h) => h.children,
            NodeHeader::Minkowski(h) => h.children,
            NodeHeader::Dot(h) => h.children,
        }
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderAngular {
    n_descendants: i32,
    children: [i32; 2],
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderMinkowski {
    n_descendants: i32,
    bias: f32,
    children: [i32; 2],
}

#[repr(C)]
#[derive(Pod, Zeroable, Debug, Clone, Copy)]
pub struct NodeHeaderDot {
    n_descendants: i32,
    children: [i32; 2],
    dot_factor: f32,
}

impl NodeHeaderAngular {
    fn read(storage: &[u8], offset: usize) -> NodeHeaderAngular {
        let array: [u8; mem::size_of::<Self>()] = storage[offset..offset + mem::size_of::<Self>()]
            .try_into()
            .unwrap();
        bytemuck::cast(array)
    }

    pub const fn header_size() -> usize {
        mem::size_of::<NodeHeaderAngular>()
    }
}

impl NodeHeaderMinkowski {
    fn read(storage: &[u8], offset: usize) -> NodeHeaderMinkowski {
        let array: [u8; mem::size_of::<Self>()] = storage[offset..offset + mem::size_of::<Self>()]
            .try_into()
            .unwrap();
        bytemuck::cast(array)
    }

    pub const fn header_size() -> usize {
        mem::size_of::<NodeHeaderMinkowski>()
    }
}

impl NodeHeaderDot {
    fn read(storage: &[u8], offset: usize) -> NodeHeaderDot {
        let array: [u8; mem::size_of::<Self>()] = storage[offset..offset + mem::size_of::<Self>()]
            .try_into()
            .unwrap();
        bytemuck::cast(array)
    }

    pub const fn header_size() -> usize {
        mem::size_of::<NodeHeaderDot>()
    }
}
