use std::borrow::Cow;
use std::fmt;
use std::mem::size_of;

use byteorder::{BigEndian, ByteOrder};
use heed::BoxedError;

use crate::ItemId;

/// This whole structure must fit in an u64 so we can tell LMDB to optimize its storage.
/// The `index` is specified by the user and is used to differentiate between multiple arroy indexes.
/// Its last bit represent the mode we're in:
/// - 0 means we're manipulating an item node.
/// - 1 means we're manipulating a tree node.
/// The metadata are stored at the last tree node.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    // The prefix specified by the user on the first 7 bits
    // And the last bit for the mode.
    // - 0 => Item
    // - 1 => Tree
    pub index: u8,
    pub item: ItemId,
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[derive(Debug)]
        enum Mode {
            Item,
            Tree,
            Metadata,
        }
        let mode = match self.index & 1 {
            0 => Mode::Item,
            1 if self.item == ItemId::MAX => Mode::Metadata,
            1 => Mode::Tree,
            _ => unreachable!(),
        };

        f.debug_struct("Key")
            .field("index", &(self.index & 0b1111_1110))
            .field("mode", &mode)
            .field("item", &self.item)
            .finish()
    }
}

impl Key {
    const ITEM_MASK: u8 = 0b1111_1110;
    const TREE_MASK: u8 = !Self::ITEM_MASK;

    pub const fn new(index: u8, item: ItemId) -> Self {
        Self { index, item }
    }

    pub const fn metadata(index: u8) -> Self {
        Self::new(index | Self::TREE_MASK, ItemId::MAX)
    }

    pub const fn item(index: u8, item: ItemId) -> Self {
        Self::new(index & Self::ITEM_MASK, item)
    }

    pub const fn tree(index: u8, item: ItemId) -> Self {
        Self::new(index | Self::TREE_MASK, item)
    }

    #[track_caller]
    pub fn unwrap_item(&self) -> ItemId {
        if self.index & 1 != 0 {
            panic!("Unwrap item called on a tree node");
        }
        self.item
    }

    #[track_caller]
    pub fn unwrap_tree(&self) -> ItemId {
        if self.index & 1 != 1 {
            panic!("Unwrap tree called on an item node");
        }
        self.item
    }

    pub fn to_bytes(&self) -> [u8; 5] {
        let mut output = [0; 5];
        output[0] = self.index;

        let item_bytes = self.item.to_be_bytes();
        debug_assert_eq!(item_bytes.len(), output.len() - 1);

        output[1..].copy_from_slice(&item_bytes);

        output
    }

    pub fn from_bytes(bytes: &[u8]) -> (Self, &[u8]) {
        let index = bytes[0];
        let item = BigEndian::read_u32(&bytes[1..]);

        (Self::new(index, item), &bytes[size_of::<u8>() + size_of::<ItemId>()..])
    }
}

pub enum KeyCodec {}

impl<'a> heed::BytesEncode<'a> for KeyCodec {
    type EItem = Key;

    fn bytes_encode(key: &'a Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let mut output = Vec::with_capacity(size_of::<u8>() + size_of::<u32>());
        output.push(key.index);
        output.extend_from_slice(&key.item.to_be_bytes());

        Ok(Cow::Owned(output))
    }
}

impl heed::BytesDecode<'_> for KeyCodec {
    type DItem = Key;

    fn bytes_decode(bytes: &[u8]) -> Result<Self::DItem, BoxedError> {
        let index = bytes[0];
        let bytes = &bytes[size_of::<u8>()..];
        let item = BigEndian::read_u32(bytes);

        Ok(Key::new(index, item))
    }
}

/// This is used to query part of a key.
#[derive(Debug, Copy, Clone)]
pub struct Prefix {
    /// The index specified by the user + the mode.
    index: u8,
}

impl Prefix {
    pub const fn item(index: u8) -> Self {
        Self { index: index & Key::ITEM_MASK }
    }

    pub const fn tree(index: u8) -> Self {
        Self { index: index & Key::TREE_MASK }
    }
}

pub enum PrefixCodec {}

impl<'a> heed::BytesEncode<'a> for PrefixCodec {
    type EItem = Prefix;

    fn bytes_encode(prefix: &'a Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        Ok(Cow::Owned([prefix.index].to_vec()))
    }
}
