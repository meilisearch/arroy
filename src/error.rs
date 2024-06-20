use std::io;

use crate::{key::Key, node_id::NodeMode, ItemId};

/// The different set of errors that arroy can encounter.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An internal error mostly related to LMDB or encoding/decoding.
    #[error(transparent)]
    Heed(#[from] heed::Error),

    /// IO error
    #[error(transparent)]
    Io(#[from] io::Error),

    /// The user is trying to insert or search for a vector that is not of the right dimensions.
    #[error("Invalid vector dimensions. Got {received} but expected {expected}")]
    InvalidVecDimension {
        /// The expected number of dimensions.
        expected: usize,
        /// The dimensions given by the user.
        received: usize,
    },

    /// An internal error returned when arroy cannot generate internal IDs.
    #[error("Database full. Arroy cannot generate enough internal IDs for your items")]
    DatabaseFull,

    /// The user tried to append an item in the database but the last inserted item
    /// is highler or equal to this one.
    #[error("Item cannot be appended into the database")]
    InvalidItemAppend,

    /// The user is trying to query a database with a distance that is not of the right type.
    #[error("Invalid distance provided. Got {received} but expected {expected}")]
    UnmatchingDistance {
        /// The expected distance type.
        expected: String,
        /// The distance given by the user.
        received: &'static str,
    },

    /// Arroy is not able to find the metadata for a given index.
    /// It is probably because the user forget to build the database.
    #[error(
        "Metadata are missing on index {0}, You must build your database before attempting to read it"
    )]
    MissingMetadata(u16),

    /// The last time items in the database were updated, the [`Writer::build`] method wasn't called.
    #[error("The trees have not been built after an update on index {0}")]
    NeedBuild(u16),

    /// Internal error
    #[error("Internal error: {mode}({item}) is missing in index `{index}`")]
    MissingKey {
        /// The index that caused the error
        index: u16,
        /// The kind of item that was being queried
        mode: &'static str,
        /// The item ID queried
        item: ItemId,
    },
}

impl Error {
    pub(crate) fn missing_key(key: Key) -> Self {
        Self::MissingKey {
            index: key.index,
            mode: match key.node.mode {
                NodeMode::Item => "Item",
                NodeMode::Tree => "Tree",
                NodeMode::Metadata => "Metadata",
            },
            item: key.node.item,
        }
    }
}
