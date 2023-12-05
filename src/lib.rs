//! Arroy ([Approximate Rearest Reighbors][1] Oh Yeah) is a Rust library with the interface of the [Annoy Python library][2] to search for vectors in space that are close to a given query vector. It is based on LMDB, a memory-mapped key-value store, so many processes may share the same data and atomically modify the vectors.
//!
//! [1]: https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor
//! [2]: https://github.com/spotify/annoy/#full-python-api
//!
//! # Examples
//!
//! Open an LMDB database, store some vectors in it and query the top 20 nearest items from the first vector. This is the most trivial way to use arroy and it's fairly easy. Just do not forget to [`Writer::build`] and [`heed::RwTxn::commit`] when you are done inserting your items.
//!
//! ```
//! use std::num::NonZeroUsize;
//!
//! use arroy::distances::Euclidean;
//! use arroy::{Database as ArroyDatabase, Writer, Reader};
//! use rand::rngs::StdRng;
//! use rand::{Rng, SeedableRng};
//!
//! /// That's the 200MiB size limit we allow LMDB to grow.
//! const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dir = tempfile::tempdir()?;
//! let env = heed::EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path())?;
//!
//! // we will open the default LMDB unnamed database
//! let mut wtxn = env.write_txn()?;
//! let db: ArroyDatabase<Euclidean> = env.create_database(&mut wtxn, None)?;
//!
//! // Now we can give it to our arroy writer
//! let index = 0;
//! let dimensions = 5;
//! let writer = Writer::<Euclidean>::prepare(&mut wtxn, db, index, dimensions)?;
//!
//! // let's write some vectors
//! writer.add_item(&mut wtxn, 0,    &[0.8,  0.49, 0.27, 0.76, 0.94])?;
//! writer.add_item(&mut wtxn, 1,    &[0.66, 0.86, 0.42, 0.4,  0.31])?;
//! writer.add_item(&mut wtxn, 2,    &[0.5,  0.95, 0.7,  0.51, 0.03])?;
//! writer.add_item(&mut wtxn, 100,  &[0.52, 0.33, 0.65, 0.23, 0.44])?;
//! writer.add_item(&mut wtxn, 1000, &[0.18, 0.43, 0.48, 0.81, 0.29])?;
//!
//! // You can specify the number of trees to use or specify None.
//! let mut rng = StdRng::seed_from_u64(42);
//! writer.build(&mut wtxn, &mut rng, None)?;
//!
//! // By committing, other readers can query the database in parallel.
//! wtxn.commit()?;
//!
//! let mut rtxn = env.read_txn()?;
//! let reader = Reader::<Euclidean>::open(&rtxn, index, db)?;
//! let n_results = 20;
//!
//! // You can increase the quality of the results by forcing arroy to search into more nodes.
//! // This multiplier is arbitrary but basically the higher, the better the results, the slower the query.
//! let is_precise = true;
//! let search_k = if is_precise {
//!     NonZeroUsize::new(n_results * reader.n_trees() * 15)
//! } else {
//!     None
//! };
//!
//! // Similar searching can be achieved by requesting the nearest neighbors of a given item.
//! let item_id = 0;
//! let arroy_results = reader.nns_by_item(&rtxn, item_id, n_results, search_k, None)?.unwrap();
//! # Ok(()) }
//! ```

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
mod stats;
mod writer;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::ffi::CStr;
use std::mem::size_of;

use byteorder::{BigEndian, ByteOrder};
pub use distance::Distance;
pub use error::Error;
use heed::BoxedError;
use key::{Key, Prefix, PrefixCodec};
use node::{ItemIds, Node, NodeCodec};
use node_id::{NodeId, NodeMode};
pub use reader::Reader;
pub use stats::{Stats, TreeStats};
pub use writer::Writer;

/// The set of types used by the [`Distance`] trait.
pub mod internals {
    use rand::Rng;

    pub use crate::distance::{
        NodeHeaderAngular, NodeHeaderDotProduct, NodeHeaderEuclidean, NodeHeaderManhattan,
    };
    pub use crate::key::KeyCodec;
    pub use crate::node::{Leaf, NodeCodec, UnalignedF32Slice};

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
    distance: &'a str,
}

enum MetadataCodec {}

impl<'a> heed::BytesEncode<'a> for MetadataCodec {
    type EItem = Metadata<'a>;

    fn bytes_encode(item: &'a Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let Metadata { dimensions, n_items, roots, distance } = item;
        debug_assert!(!distance.as_bytes().iter().any(|&b| b == 0));

        let mut output = Vec::with_capacity(
            size_of::<u32>() + roots.len() * size_of::<u32>() + distance.len() + 1,
        );
        output.extend_from_slice(distance.as_bytes());
        output.push(0);
        output.extend_from_slice(&dimensions.to_be_bytes());
        output.extend_from_slice(&n_items.to_be_bytes());
        output.extend_from_slice(roots.raw_bytes());

        Ok(Cow::Owned(output))
    }
}

impl<'a> heed::BytesDecode<'a> for MetadataCodec {
    type DItem = Metadata<'a>;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        let distance = CStr::from_bytes_until_nul(bytes)?.to_str()?;
        let bytes = &bytes[distance.len() + 1..];
        let dimensions = BigEndian::read_u32(bytes);
        let bytes = &bytes[size_of::<u32>()..];
        let n_items = BigEndian::read_u32(bytes);
        let bytes = &bytes[size_of::<u32>()..];

        Ok(Metadata { dimensions, n_items, roots: ItemIds::from_bytes(bytes), distance })
    }
}
