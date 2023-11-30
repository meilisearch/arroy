#![warn(missing_docs)]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/meilisearch/arroy/main/assets/arroy-electric-clusters.ico?raw=true"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/meilisearch/arroy/main/assets/arroy-electric-clusters-logo.png?raw=true"
)]

mod distance;
mod error;
mod item_iter;
mod key;
mod node;
mod node_id;
mod reader;
mod spaces;
mod writer;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::mem::size_of;

use byteorder::{BigEndian, ByteOrder};
pub use distance::Distance;
pub use error::Error;
use heed::BoxedError;
use key::{Key, Prefix, PrefixCodec};
use node::{ItemIds, Node, NodeCodec};
use node_id::{NodeId, NodeMode};
pub use reader::Reader;
pub use writer::Writer;

/// The set of types used by the [`Distance`] trait.
pub mod internals {
    use rand::Rng;

    pub use crate::distance::{
        NodeHeaderAngular, NodeHeaderDotProduct, NodeHeaderEuclidean, NodeHeaderManhattan,
    };
    pub use crate::key::KeyCodec;
    pub use crate::node::{Leaf, UnalignedF32Slice};

    /// A type that is used to decide on
    /// which side of a plane we move an item.
    #[derive(Debug, Copy, Clone)]
    pub enum Side {
        /// The left side.
        Left,
        /// The right side.
        Right,
    }

    impl Side {
        pub(crate) fn random<R: Rng>(rng: &mut R) -> Side {
            if rng.gen() {
                Side::Left
            } else {
                Side::Right
            }
        }
    }
}

/// The set of distances implementing the [`Distance`] and supported by arroy.
pub mod distances {
    pub use crate::distance::{Angular, DotProduct, Euclidean, Manhattan};
}

/// A custom Result type that is returning an arroy error by default.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The database required by arroy for reading or writing operations.
pub type Database<D> = heed::Database<internals::KeyCodec, NodeCodec<D>>;

/// An identifier for the items stored in the database.
pub type ItemId = u32;

#[derive(Debug)]
struct Metadata<'a> {
    dimensions: u32,
    n_items: u32,
    roots: ItemIds<'a>,
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

        Ok(Metadata { dimensions, n_items, roots: ItemIds::from_bytes(bytes) })
    }
}
