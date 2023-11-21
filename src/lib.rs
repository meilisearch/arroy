mod distance;
mod error;
mod item_iter;
mod node;
mod reader;
mod spaces;
mod writer;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytemuck::{pod_collect_to_vec, try_cast_slice, Pod, Zeroable};
use byteorder::{BigEndian, ByteOrder};
pub use distance::{
    Angular, Distance, Euclidean, Manhattan, NodeHeaderAngular, NodeHeaderEuclidean,
    NodeHeaderManhattan,
};
pub use error::Error;
use heed::BoxedError;
use node::NodeIds;
pub use node::{Leaf, Node, NodeCodec};
use rand::Rng;
pub use reader::Reader;
pub use writer::Writer;

pub type Result<T, E = Error> = std::result::Result<T, E>;

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

pub static ALIGNED_VECTOR: AtomicUsize = AtomicUsize::new(0);
pub static NON_ALIGNED_VECTOR: AtomicUsize = AtomicUsize::new(0);

fn aligned_or_collect_vec<T: Pod + Zeroable>(bytes: &[u8]) -> Cow<[T]> {
    use bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned;
    match try_cast_slice(bytes) {
        Ok(casted) => {
            ALIGNED_VECTOR.fetch_add(1, Ordering::SeqCst);
            Cow::Borrowed(casted)
        }
        Err(TargetAlignmentGreaterAndInputNotAligned) => {
            NON_ALIGNED_VECTOR.fetch_add(1, Ordering::SeqCst);
            Cow::Owned(pod_collect_to_vec(bytes))
        }
        Err(e) => panic!("casting slices failed: {e}"),
    }
}

#[derive(Debug)]
struct Metadata<'a> {
    dimensions: u32,
    n_items: u32,
    roots: NodeIds<'a>,
}

enum MetadataCodec {}

impl<'a> heed::BytesEncode<'a> for MetadataCodec {
    type EItem = Metadata<'a>;

    fn bytes_encode(item: &'a Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let mut output = Vec::with_capacity(size_of::<u32>() + item.roots.len() * size_of::<u32>());
        output.extend_from_slice(&item.dimensions.to_be_bytes());
        output.extend_from_slice(&item.n_items.to_be_bytes());
        output.extend_from_slice(item.roots.raw_bytes());

        Ok(Cow::Owned(output))
    }
}

impl<'a> heed::BytesDecode<'a> for MetadataCodec {
    type DItem = Metadata<'a>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        let dimensions = BigEndian::read_u32(bytes);
        let bytes = &bytes[size_of::<u32>()..];
        let n_items = BigEndian::read_u32(bytes);
        let bytes = &bytes[size_of::<u32>()..];

        Ok(Metadata { dimensions, n_items, roots: NodeIds::from_bytes(bytes) })
    }
}
