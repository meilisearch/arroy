#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Heed(#[from] heed::Error),

    #[error("Invalid vector dimensions. Got {received} but expected {expected}.")]
    InvalidVecDimension { expected: usize, received: usize },

    #[error("Database full. Try to use lower vector IDs.")]
    DatabaseFull,

    #[error("Metadata are missing, did you build your database before trying to read it.")]
    MissingMetadata,
}
