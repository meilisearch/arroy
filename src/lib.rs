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
//! let env = unsafe { heed::EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path()) }?;
//!
//! // we will open the default LMDB unnamed database
//! let mut wtxn = env.write_txn()?;
//! let db: ArroyDatabase<Euclidean> = env.create_database(&mut wtxn, None)?;
//!
//! // Now we can give it to our arroy writer
//! let index = 0;
//! let dimensions = 5;
//! let writer = Writer::<Euclidean>::new(db, index, dimensions);
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
//! writer.builder(&mut rng).build(&mut wtxn)?;
//!
//! // By committing, other readers can query the database in parallel.
//! wtxn.commit()?;
//!
//! let mut rtxn = env.read_txn()?;
//! let reader = Reader::<Euclidean>::open(&rtxn, index, db)?;
//! let n_results = 20;
//!
//! let mut query = reader.nns(n_results);
//!
//! // You can increase the quality of the results by forcing arroy to search into more nodes.
//! // This multiplier is arbitrary but basically the higher, the better the results, the slower the query.
//! let is_precise = true;
//! if is_precise {
//!     query.search_k(NonZeroUsize::new(n_results * reader.n_trees() * 15).unwrap());
//! }
//!
//! // Similar searching can be achieved by requesting the nearest neighbors of a given item.
//! let item_id = 0;
//! let arroy_results = query.by_item(&rtxn, item_id)?.unwrap();
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
mod metadata;
mod node;
mod node_id;
mod parallel;
mod reader;
mod roaring;
mod spaces;
mod stats;
mod writer;

#[cfg(test)]
mod tests;
mod unaligned_vector;

pub use distance::Distance;
use distances::Cosine;
pub use error::Error;
use heed::{
    types::{Bytes, LazyDecode, Unit},
    RoTxn, RwTxn, Unspecified,
};
use internals::KeyCodec;
use key::{Key, Prefix, PrefixCodec};
use metadata::{Metadata, MetadataCodec};
use node::{Node, NodeCodec};
use node_id::{NodeId, NodeMode};
pub use reader::{QueryBuilder, Reader};
use roaring::RoaringBitmapCodec;
pub use stats::{Stats, TreeStats};
pub use writer::{ArroyBuilder, Writer};

/// The set of types used by the [`Distance`] trait.
pub mod internals {
    use rand::Rng;

    pub use crate::distance::{
        NodeHeaderBinaryQuantizedCosine, NodeHeaderBinaryQuantizedEuclidean,
        NodeHeaderBinaryQuantizedManhattan, NodeHeaderCosine, NodeHeaderDotProduct,
        NodeHeaderEuclidean, NodeHeaderManhattan,
    };
    pub use crate::key::KeyCodec;
    pub use crate::node::{Leaf, NodeCodec};
    pub use crate::unaligned_vector::{SizeMismatch, UnalignedVector, UnalignedVectorCodec};

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
    pub use crate::distance::{
        BinaryQuantizedCosine, BinaryQuantizedEuclidean, BinaryQuantizedManhattan, Cosine,
        DotProduct, Euclidean, Manhattan,
    };
}

/// A custom Result type that is returning an arroy error by default.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The database required by arroy for reading or writing operations.
pub type Database<D> = heed::Database<internals::KeyCodec, NodeCodec<D>>;

/// An identifier for the items stored in the database.
pub type ItemId = u32;

// ################ The updating code ################

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
enum OldNodeMode {
    Item = 0,
    Tree = 1,
    Metadata = 2,
}

impl TryFrom<u8> for OldNodeMode {
    type Error = String;

    fn try_from(v: u8) -> std::result::Result<Self, Self::Error> {
        match v {
            v if v == OldNodeMode::Item as u8 => Ok(OldNodeMode::Item),
            v if v == OldNodeMode::Tree as u8 => Ok(OldNodeMode::Tree),
            v if v == OldNodeMode::Metadata as u8 => Ok(OldNodeMode::Metadata),
            v => Err(format!("Could not convert {v} as a `NodeMode`.")),
        }
    }
}

/// Upgrade arroy from v0.4 to v0.5 without rebuilding the trees
/// /!\ This is only valid if the arroy database was generated by Meilisearch. Do not try to use it yourself.
pub fn ugrade_from_prev_version(
    rtxn: &RoTxn,
    read_database: heed::Database<Unspecified, Unspecified>,
    wtxn: &mut RwTxn,
    write_database: heed::Database<Unspecified, Unspecified>,
) -> Result<()> {
    // We need to update EVERY single nodes, thus we can clear the whole DB initially
    write_database.clear(wtxn)?;

    // Then we **must** iterate over everything in the database to be sure we don't miss anything.
    for ret in read_database.remap_types::<internals::KeyCodec, LazyDecode<Bytes>>().iter(rtxn)? {
        let (mut key, value) = ret?;
        let old_mode = OldNodeMode::try_from(key.node.mode as u8)
            .map_err(|_| Error::CannotDecodeKeyMode { mode: key.node.mode })?;

        match old_mode {
            OldNodeMode::Item => {
                key.node.mode = NodeMode::Item;
                // In case of an item there is nothing else to do
                write_database.remap_types::<KeyCodec, Bytes>().put(
                    wtxn,
                    &key,
                    value.remap::<Bytes>().decode().unwrap(),
                )?;
            }
            OldNodeMode::Tree => {
                key.node.mode = NodeMode::Tree;
                // Meilisearch is only using Cosine distance at this point
                let mut tree_node = value.remap::<NodeCodec<Cosine>>().decode().unwrap();
                // The leaf and descendants tree node don't contains any node mode
                if let Node::SplitPlaneNormal(split) = &mut tree_node {
                    let left_old_mode = OldNodeMode::try_from(split.left.mode as u8)
                        .map_err(|_| Error::CannotDecodeKeyMode { mode: split.left.mode })?;
                    split.left.mode = match left_old_mode {
                        OldNodeMode::Item => NodeMode::Item,
                        OldNodeMode::Tree => NodeMode::Tree,
                        OldNodeMode::Metadata => NodeMode::Metadata,
                    };

                    let right_old_mode = OldNodeMode::try_from(split.right.mode as u8)
                        .map_err(|_| Error::CannotDecodeKeyMode { mode: split.right.mode })?;
                    split.right.mode = match right_old_mode {
                        OldNodeMode::Item => NodeMode::Item,
                        OldNodeMode::Tree => NodeMode::Tree,
                        OldNodeMode::Metadata => NodeMode::Metadata,
                    };
                }
                write_database
                    .remap_types::<KeyCodec, NodeCodec<Cosine>>()
                    .put(wtxn, &key, &tree_node)?;
            }
            OldNodeMode::Metadata => {
                match key.index {
                    0 => {
                        key.node.mode = NodeMode::Metadata;
                        // The distance has been renamed
                        let mut metadata = value.remap::<MetadataCodec>().decode().unwrap();
                        metadata.distance = Cosine::name();
                        write_database
                            .remap_types::<KeyCodec, MetadataCodec>()
                            .put(wtxn, &key, &metadata)?;
                    }
                    1 => {
                        key.node.mode = NodeMode::Updated;
                        // In this case we have a roaring bitmap of document id
                        // that we must re-insert as multiple values
                        let updated =
                            value.remap::<RoaringBitmapCodec>().decode().unwrap_or_default();
                        for item in updated {
                            key.node.item = item;
                            write_database.remap_types::<KeyCodec, Unit>().put(wtxn, &key, &())?;
                        }
                    }
                    _ => (),
                }
            }
        };
    }

    Ok(())
}
