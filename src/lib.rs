mod distance;
mod node;
mod reader;
mod spaces;
mod writer;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::mem::size_of;

use bytemuck::{cast_slice, pod_collect_to_vec, try_cast_slice, Pod, Zeroable};
use byteorder::{BigEndian, ByteOrder};
pub use distance::{
    Angular, Distance, Euclidean, Manhattan, NodeHeaderAngular, NodeHeaderEuclidean,
    NodeHeaderManhattan,
};
pub use node::{Leaf, Node, NodeCodec};
use rand::Rng;
pub use reader::Reader;
pub use writer::Writer;

/// An big endian-encoded u32.
pub type BEU32 = heed::types::U32<heed::byteorder::BE>;

/// An external item id.
pub type ItemId = u32;

/// An internal node id.
type NodeId = u32;

#[derive(Debug, Copy, Clone)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    pub fn random<R: Rng>(rng: &mut R) -> Side {
        if rng.gen() {
            Side::Left
        } else {
            Side::Right
        }
    }
}

fn aligned_or_collect_vec<T: Pod + Zeroable>(bytes: &[u8]) -> Cow<[T]> {
    use bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned;
    match try_cast_slice(bytes) {
        Ok(casted) => Cow::Borrowed(casted),
        Err(TargetAlignmentGreaterAndInputNotAligned) => Cow::Owned(pod_collect_to_vec(bytes)),
        Err(e) => panic!("casting slices failed: {e}"),
    }
}

#[derive(Debug)]
struct Metadata<'a> {
    dimensions: usize,
    root_nodes: Cow<'a, [NodeId]>,
}

struct MetadataCodec;

impl<'a> heed::BytesEncode<'a> for MetadataCodec {
    type EItem = Metadata<'a>;

    fn bytes_encode(item: &'a Self::EItem) -> Result<std::borrow::Cow<'a, [u8]>, heed::BoxedError> {
        let mut output: Vec<u8> =
            Vec::with_capacity(size_of::<u32>() + item.root_nodes.len() * size_of::<u32>());
        output.extend_from_slice(&(item.dimensions as u32).to_be_bytes());
        output.extend_from_slice(cast_slice(&item.root_nodes));

        Ok(Cow::Owned(output))
    }
}

impl<'a> heed::BytesDecode<'a> for MetadataCodec {
    type DItem = Metadata<'a>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, heed::BoxedError> {
        let dimensions = BigEndian::read_u32(bytes);
        let bytes = &bytes[size_of::<u32>()..];
        let root_nodes = aligned_or_collect_vec(bytes);

        Ok(Metadata { dimensions: dimensions as usize, root_nodes })
    }
}
