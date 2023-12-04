use std::io;

/// The different set of errors that arroy can encounter.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An internal error mostly related to LMDB or encoding/decoding.
    #[error(transparent)]
    Heed(#[from] heed::Error),

    /// IO error
    #[error(transparent)]
    IO(#[from] io::Error),

    /// The user is trying to insert or search for a vector that is not of the right dimensions.
    #[error("Invalid vector dimensions. Got {received} but expected {expected}.")]
    InvalidVecDimension {
        /// The expected number of dimensions.
        expected: usize,
        /// The dimensions given by the user.
        received: usize,
    },

    /// An internal error returned when arroy cannot generate internal IDs.
    #[error("Database full. Try to use lower vector IDs.")]
    DatabaseFull,

    /// The user is trying to query a database with a distance that is not of the right type.
    #[error("Invalid distance provided. Got {received} but expected {expected}.")]
    UnmatchingDistance {
        /// The expected distance type.
        expected: String,
        /// The distance given by the user.
        received: &'static str,
    },

    /// Arroy is not able to find the metadata for a given index.
    /// It is probably because the user forget to build the database.
    #[error("Metadata are missing, did you build your database before trying to read it.")]
    MissingMetadata,
}
