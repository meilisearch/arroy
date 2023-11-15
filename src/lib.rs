mod arroy;
mod arroy_reader;
mod distance;
// mod heed_reader;
mod node;
mod priority_queue;

pub use arroy::{Angular, NodeCodec, NodeHeaderAngular, Reader, Writer, BEU32};
pub use arroy_reader::ArroyReader;
// pub use heed_reader::HeedReader;

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
#[repr(u8)]
pub enum DistanceType {
    Angular = 0,
    Euclidean = 1,
    Manhattan = 2,
    // Hamming = 3,
    Dot = 4,
}
