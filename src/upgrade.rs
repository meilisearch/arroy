//! Everything related to the upgrade process.

use heed::{
    types::{Bytes, LazyDecode, Unit},
    RoTxn, RwTxn,
};
use std::collections::HashMap;
use roaring::RoaringBitmap;

use crate::{
    distance::Cosine,
    key::Key,
    metadata::MetadataCodec,
    node::{Descendants, Node, NodeCodec, SplitPlaneNormal},
    node_id::{NodeId, NodeMode},
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

    // Then we **must** iterate over everything in the database to be sure we don't miss anything.
    for ret in read_database.remap_data_type::<LazyDecode<NodeCodec<Cosine>>>().iter(rtxn)? {
        let (mut key, value) = ret?;
        let old_mode = OldNodeMode::try_from(key.node.mode as u8)
            .map_err(|_| Error::CannotDecodeKeyMode { mode: key.node.mode })?;

        match old_mode {
            OldNodeMode::Item => {
                key.node.mode = NodeMode::Item;
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
                write_database.put(wtxn, &key, &tree_node)?;
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
pub fn from_0_5_to_0_6<C: Distance>(
    rtxn: &RoTxn,
    read_database: Database<C>,
    wtxn: &mut RwTxn,
    write_database: Database<C>,
) -> Result<()> {
    let version = Version {
        major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
        minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
        patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
    };

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

/// Upgrade an arroy database from v0.6 to v0.7 without rebuilding the trees.
/// This upgrade makes two changes:
/// 1. Ensures SplitPlaneNormal nodes never point directly to items, adding intermediary nodes
/// 2. Makes the normal vector field optional, with null for zero vectors
pub fn from_0_6_to_0_7<C: Distance>(
    rtxn: &RoTxn,
    read_database: Database<C>,
    wtxn: &mut RwTxn,
    write_database: Database<C>,
) -> Result<()> {
    
    // Update version
    let version = Version {
        major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
        minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
        patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
    };
    
    // Track new intermediate nodes we create (original_item_id -> new_node_id mapping)
    let mut intermediate_nodes: HashMap<u32, u32> = HashMap::new();
    
    // Find the highest node ID currently in use to generate new IDs
    let mut next_node_id = 0;
    for ret in read_database.remap_data_type::<LazyDecode<NodeCodec<C>>>().iter(rtxn)? {
        let (key, _) = ret?;
        if key.node.mode == NodeMode::Tree {
            next_node_id = next_node_id.max(key.node.item + 1);
        }
    }
    
    // First pass: Copy all nodes that don't need modification
    for ret in read_database.remap_data_type::<LazyDecode<NodeCodec<C>>>().iter(rtxn)? {
        let (key, value) = ret?;
        
        // Skip non-tree nodes
        if key.node.mode != NodeMode::Tree {
            continue;
        }
        
        let node = value.decode().unwrap();
        if let Node::SplitPlaneNormal(ref split) = node {
            // Check if left or right points directly to an item
            if split.left.mode == NodeMode::Item || split.right.mode == NodeMode::Item {
                // Process this node in the next pass
                continue;
            }
        }
        
        // For all other nodes, copy as is (but still need to adapt to new SplitPlaneNormal format)
        if let Node::SplitPlaneNormal(ref split) = node {
            let normal_opt = if let Some(ref normal) = &split.normal {
                if normal.is_zero() {
                    None
                } else {
                    Some(normal.clone())
                }
            } else {
                None
            };
            
            let updated_node = Node::SplitPlaneNormal(SplitPlaneNormal {
                left: split.left,
                right: split.right,
                normal: normal_opt,
            });
            
            write_database.put(wtxn, &key, &updated_node)?;
        } else {
            // Copy non-SplitPlaneNormal nodes as is
            write_database.remap_data_type::<Bytes>().put(
                wtxn,
                &key,
                value.remap::<Bytes>().decode().unwrap(),
            )?;
        }
    }
    
    // Second pass: Process SplitPlaneNormal nodes with direct item links
    for ret in read_database.remap_data_type::<LazyDecode<NodeCodec<C>>>().iter(rtxn)? {
        let (key, value) = ret?;
        
        // Skip non-tree nodes
        if key.node.mode != NodeMode::Tree {
            continue;
        }
        
        let node = value.decode().unwrap();
        if let Node::SplitPlaneNormal(ref split) = node {
            // Check if left or right points directly to an item
            if split.left.mode == NodeMode::Item || split.right.mode == NodeMode::Item {
                // Create a new SplitPlaneNormal with intermediary nodes
                let new_left = if split.left.mode == NodeMode::Item {
                    let item_id = split.left.unwrap_item();
                    
                    // Check if we already created an intermediate node for this item
                    if let Some(&node_id) = intermediate_nodes.get(&item_id) {
                        NodeId::tree(node_id)
                    } else {
                        // Create a new intermediary Descendants node for this item
                        let mut descendants = RoaringBitmap::new();
                        descendants.push(item_id);
                        
                        let new_node_id = next_node_id;
                        next_node_id += 1;
                        intermediate_nodes.insert(item_id, new_node_id);
                        
                        let new_node = Node::Descendants(Descendants { 
                            descendants: std::borrow::Cow::Owned(descendants) 
                        });
                        
                        // Save the new node
                        let new_key = Key::new(key.index, NodeId::tree(new_node_id));
                        write_database.put(wtxn, &new_key, &new_node)?;
                        
                        NodeId::tree(new_node_id)
                    }
                } else {
                    split.left
                };
                
                let new_right = if split.right.mode == NodeMode::Item {
                    let item_id = split.right.unwrap_item();
                    
                    // Check if we already created an intermediate node for this item
                    if let Some(&node_id) = intermediate_nodes.get(&item_id) {
                        NodeId::tree(node_id)
                    } else {
                        // Create a new intermediary Descendants node for this item
                        let mut descendants = RoaringBitmap::new();
                        descendants.push(item_id);
                        
                        let new_node_id = next_node_id;
                        next_node_id += 1;
                        intermediate_nodes.insert(item_id, new_node_id);
                        
                        let new_node = Node::Descendants(Descendants { 
                            descendants: std::borrow::Cow::Owned(descendants) 
                        });
                        
                        // Save the new node
                        let new_key = Key::new(key.index, NodeId::tree(new_node_id));
                        write_database.put(wtxn, &new_key, &new_node)?;
                        
                        NodeId::tree(new_node_id)
                    }
                } else {
                    split.right
                };
                
                // Optimize the normal vector to be None if it's all zeros
                let normal_opt = if let Some(ref normal) = &split.normal {
                    if normal.is_zero() {
                        None
                    } else {
                        Some(normal.clone())
                    }
                } else {
                    None
                };
                
                // Create the updated split node
                let updated_node = Node::SplitPlaneNormal(SplitPlaneNormal {
                    left: new_left,
                    right: new_right,
                    normal: normal_opt,
                });
                
                // Save the updated node
                write_database.put(wtxn, &key, &updated_node)?;
            }
        }
    }
    
    // Update version for all indexes
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
