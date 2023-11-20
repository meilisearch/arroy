#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Heed(#[from] heed::Error),

    #[error("Invalid vector dimensions. Got {received} but expected {expected}")]
    InvalidVecDimension { expected: usize, received: usize },
}
