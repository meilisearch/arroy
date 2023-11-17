mod distance;
mod node;
mod reader;
mod spaces;
mod writer;

#[cfg(test)]
mod tests;

pub use distance::{Angular, Distance, Euclidean, Manhattan};
pub use node::{Leaf, Node, NodeCodec};
use rand::Rng;
pub use reader::Reader;
pub use writer::Writer;

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
