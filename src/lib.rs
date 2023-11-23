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

use bytemuck::{pod_collect_to_vec, try_cast_slice, Pod, Zeroable};
use byteorder::{BigEndian, ByteOrder};
pub use distance::{
    Angular, Distance, DotProduct, Euclidean, Manhattan, NodeHeaderAngular, NodeHeaderDotProduct,
    NodeHeaderEuclidean, NodeHeaderManhattan,
};
pub use error::Error;
use heed::BoxedError;
use node::NodeIds;
pub use node::{Leaf, Node, NodeCodec};
use rand::Rng;
pub use reader::Reader;
pub use writer::Writer;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The database required by arroy for reading or writing operations.
pub type Database<D> = heed::Database<KeyCodec, NodeCodec<D>>;

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

/// /!\ Changing the value of the enum can be DB-breaking /!\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum NodeMode {
    Item = 0,
    Tree = 1,
    Root = 2,
}

impl TryFrom<u8> for NodeMode {
    type Error = String;

    fn try_from(v: u8) -> std::result::Result<Self, Self::Error> {
        match v {
            v if v == NodeMode::Item as u8 => Ok(NodeMode::Item),
            v if v == NodeMode::Tree as u8 => Ok(NodeMode::Tree),
            v if v == NodeMode::Root as u8 => Ok(NodeMode::Root),
            v => Err(format!("Could not convert {v} as a `NodeMode`.")),
        }
    }
}

/// This whole structure must fit in an u64 so we can tell LMDB to optimize its storage.
/// The `prefix` is specified by the user and is used to differentiate between multiple arroy indexes.
/// The `mode` indicates what we're looking at.
/// The `item` point to a specific node.
/// If the mode is:
///  - `Item`: we're looking at a `Leaf` node.
///  - `Tree`: we're looking at one of the internal generated node from arroy. Could be a descendants, a split plane or anything.
///  - `Root`: There is only one item at `0` that contains the header required to read the index.
#[derive(Debug, Copy, Clone)]
pub struct Key {
    /// The prefix specified by the user.
    pub prefix: u16,
    // Indicate what the item represent.
    pub mode: NodeMode,
    /// The item we want to get.
    pub item: u32,
    /// Unused space.
    _padding: u8,
}

impl Key {
    pub const fn new(prefix: u16, mode: NodeMode, item: u32) -> Self {
        Self { prefix, mode, item, _padding: 0 }
    }

    pub const fn root(prefix: u16) -> Self {
        Self::new(prefix, NodeMode::Root, 0)
    }

    pub const fn item(prefix: u16, item: u32) -> Self {
        Self::new(prefix, NodeMode::Item, item)
    }

    pub const fn tree(prefix: u16, item: u32) -> Self {
        Self::new(prefix, NodeMode::Tree, item)
    }
}

pub enum KeyCodec {}

impl<'a> heed::BytesEncode<'a> for KeyCodec {
    type EItem = Key;

    fn bytes_encode(item: &'a Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let mut output = Vec::with_capacity(size_of::<u64>());
        output.extend_from_slice(&item.prefix.to_be_bytes());
        output.extend_from_slice(&(item.mode as u8).to_be_bytes());
        output.extend_from_slice(&item.item.to_be_bytes());
        output.extend_from_slice(&item._padding.to_be_bytes());

        Ok(Cow::Owned(output))
    }
}

impl heed::BytesDecode<'_> for KeyCodec {
    type DItem = Key;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, BoxedError> {
        let prefix = BigEndian::read_u16(bytes);
        let bytes = &bytes[size_of::<u16>()..];
        let mode = bytes[0].try_into()?;
        let bytes = &bytes[size_of::<u8>()..];
        let item = BigEndian::read_u32(bytes);
        // We don't need to deserialize the unused space

        Ok(Key { prefix, mode, item, _padding: 0 })
    }
}

/// This is used to query part of a key.
#[derive(Debug, Copy, Clone)]
pub struct Prefix {
    /// The prefix specified by the user.
    prefix: u16,
    // Indicate what the item represent.
    mode: Option<NodeMode>,
}

impl Prefix {
    pub const fn all(prefix: u16) -> Self {
        Self { prefix, mode: None }
    }

    pub const fn item(prefix: u16) -> Self {
        Self { prefix, mode: Some(NodeMode::Item) }
    }

    pub const fn tree(prefix: u16) -> Self {
        Self { prefix, mode: Some(NodeMode::Tree) }
    }
}

enum PrefixCodec {}

impl<'a> heed::BytesEncode<'a> for PrefixCodec {
    type EItem = Prefix;

    fn bytes_encode(item: &'a Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let mode_used = item.mode.is_some() as usize;
        let mut output = Vec::with_capacity(size_of::<u16>() + mode_used);

        output.extend_from_slice(&item.prefix.to_be_bytes());
        if let Some(mode) = item.mode {
            output.extend_from_slice(&(mode as u8).to_be_bytes());
        }

        Ok(Cow::Owned(output))
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
