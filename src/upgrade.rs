//! Everything related to the upgrade process.

use std::borrow::Cow;

use heed::{
    types::{Bytes, LazyDecode, Unit},
    RoTxn, RwTxn,
};
use roaring::RoaringBitmap;

use crate::{
    distance::Cosine,
    key::{Key, KeyCodec, Prefix, PrefixCodec},
    metadata::MetadataCodec,
    node::{
        Descendants, GenericReadNode, GenericReadNodeCodecFromV0_4_0, GenericReadSplitPlaneNormal,
        Node, SplitPlaneNormal, WriteNodeCodecForV0_5_0,
    },
    node_id::NodeMode,
    roaring::RoaringBitmapCodec,
    version::{Version, VersionCodec},
    Database, Distance, Error, Result,
};

/// Upgrade a cosine-based arroy database from v0.4 to v0.5 without rebuilding the trees.
pub fn cosine_from_0_4_to_0_5(
    rtxn: &RoTxn,
    read_database: Database<Cosine>,
    wtxn: &mut RwTxn,
    write_database: Database<Cosine>,
) -> Result<()> {
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

    // We need to update EVERY single nodes, thus we can clear the whole DB initially
    write_database.clear(wtxn)?;

    // To write the split plane normal we need to know the dimension of the vectors.
    // Since the `OldNodeMode` starts by the item we're guaranteed to see the dimension before the split plane normal.
    let mut dimension = None;

    // Then we **must** iterate over everything in the database to be sure we don't miss anything.
    for ret in read_database
        .remap_data_type::<LazyDecode<GenericReadNodeCodecFromV0_4_0<Cosine>>>()
        .iter(rtxn)?
    {
        let (mut key, value) = ret?;
        let old_mode = OldNodeMode::try_from(key.node.mode as u8)
            .map_err(|_| Error::CannotDecodeKeyMode { mode: key.node.mode })?;

        match old_mode {
            OldNodeMode::Item => {
                key.node.mode = NodeMode::Item;
                let item = value.decode().unwrap();
                match item {
                    GenericReadNode::Leaf(leaf) => {
                        dimension = Some(leaf.vector.len());
                    }
                    _ => unreachable!("The v0.4.0 arroy database is corrupted."),
                }
                // In case of an item there is nothing else to do
                write_database.remap_data_type::<Bytes>().put(
                    wtxn,
                    &key,
                    value.remap::<Bytes>().decode().unwrap(),
                )?;
            }
            OldNodeMode::Tree => {
                key.node.mode = NodeMode::Tree;
                // Meilisearch is only using Cosine distance at this point
                let mut tree_node = value.decode().unwrap();
                // The leaf and descendants tree node don't contains any node mode
                if let GenericReadNode::SplitPlaneNormal(split) = &mut tree_node {
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
                write_database.remap_data_type::<WriteNodeCodecForV0_5_0<Cosine>>().put(
                    wtxn,
                    &key,
                    &(tree_node, dimension.unwrap()),
                )?;
            }
            OldNodeMode::Metadata => {
                match key.node.item {
                    0 => {
                        key.node.mode = NodeMode::Metadata;
                        // The distance has been renamed
                        let mut metadata = value.remap::<MetadataCodec>().decode().unwrap();
                        metadata.distance = Cosine::name();
                        write_database
                            .remap_data_type::<MetadataCodec>()
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
                            write_database.remap_data_type::<Unit>().put(wtxn, &key, &())?;
                        }
                    }
                    other => {
                        let bytes = value.remap::<MetadataCodec>().decode().unwrap();
                        panic!("Unexpected {other} with value: {bytes:?}");
                    }
                }
            }
        };
    }

    Ok(())
}

/// Upgrade an arroy database from v0.5 to v0.6 without rebuilding the trees.
/// The only addition between the two versions is that we now store the version of the database.
pub fn from_0_5_to_0_6<C: Distance>(
    rtxn: &RoTxn,
    read_database: Database<C>,
    wtxn: &mut RwTxn,
    write_database: Database<C>,
) -> Result<()> {
    let version = Version { major: 0, minor: 6, patch: 0 };

    // Note that we have to write the versions into each database.
    // The reason is that the Keys are prefixed by the index
    // and that all indexes (u16) are valid.
    for index in 0..=u16::MAX {
        let metadata = Key::metadata(index);
        if read_database.remap_data_type::<MetadataCodec>().get(rtxn, &metadata)?.is_some() {
            write_database.remap_data_type::<VersionCodec>().put(
                wtxn,
                &Key::version(index),
                &version,
            )?;
        }
    }

    Ok(())
}

/// Upgrade an arroy database from v0.6 to the current version.
///
/// What changed:
/// - `SplitPlaneNormal::normal` is now `Option<u64>`
/// - `SplitPlaneNormal::normal` is now `None` if the normal is the zero vector
/// - `SplitPlaneNormal::normal` does not point to an item directly anymore and
///     only store a simple `ItemId` instead of the `NodeId` we had before
/// - The version must be written in each index
pub fn from_0_6_to_current<C: Distance>(
    rtxn: &RoTxn,
    read_database: Database<C>,
    wtxn: &mut RwTxn,
    write_database: Database<C>,
) -> Result<()> {
    let version = Version::current();

    for index in 0..=u16::MAX {
        let metadata = Key::metadata(index);
        if write_database.remap_data_type::<MetadataCodec>().get(wtxn, &metadata)?.is_some() {
            write_database.remap_data_type::<VersionCodec>().put(
                wtxn,
                &Key::version(index),
                &version,
            )?;
        } else {
            // If the metadata is not present, it means the index was never built and there is no nodes to update.
            continue;
        }

        let mut last_tree_id = match read_database
            .remap_key_type::<PrefixCodec>()
            .rev_prefix_iter(rtxn, &Prefix::tree(index))?
            .remap_types::<KeyCodec, Bytes>()
            .next()
        {
            Some(ret) => ret?.0.node.item,
            // If there is no tree nodes at all, there is nothing else to update in this version
            None => continue,
        };

        for ret in read_database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter(rtxn, &Prefix::tree(index))?
            .remap_types::<KeyCodec, GenericReadNodeCodecFromV0_4_0<C>>()
        {
            let (key, node) = ret?;
            if let GenericReadNode::SplitPlaneNormal(GenericReadSplitPlaneNormal {
                normal,
                left,
                right,
            }) = node
            {
                let left = match left.mode {
                    NodeMode::Item => {
                        last_tree_id += 1;
                        write_database.put(
                            wtxn,
                            &Key::tree(index, last_tree_id),
                            &Node::Descendants(Descendants {
                                descendants: Cow::Owned(RoaringBitmap::from_iter(Some(left.item))),
                            }),
                        )?;
                        last_tree_id
                    }
                    NodeMode::Tree => left.item,
                    NodeMode::Metadata => unreachable!("Metadata cannot be linked to a split node"),
                    NodeMode::Updated => unreachable!("Updated cannot be linked to a split node"),
                };
                let right = match right.mode {
                    NodeMode::Item => {
                        last_tree_id += 1;
                        write_database.put(
                            wtxn,
                            &Key::tree(index, last_tree_id),
                            &Node::Descendants(Descendants {
                                descendants: Cow::Owned(RoaringBitmap::from_iter(Some(right.item))),
                            }),
                        )?;
                        last_tree_id
                    }
                    NodeMode::Tree => right.item,
                    NodeMode::Metadata => unreachable!("Metadata cannot be linked to a split node"),
                    NodeMode::Updated => unreachable!("Updated cannot be linked to a split node"),
                };

                write_database.put(
                    wtxn,
                    &key,
                    &Node::SplitPlaneNormal(SplitPlaneNormal { normal, left, right }),
                )?;
            }
        }
    }

    Ok(())
}
