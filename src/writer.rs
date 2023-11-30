use std::any::TypeId;
use std::borrow::Cow;
use std::marker;

use heed::types::DecodeIgnore;
use heed::{MdbError, PutFlags, RoTxn, RwTxn};
use rand::Rng;

use crate::item_iter::ItemIter;
use crate::node::{Descendants, ItemIds, Leaf, SplitPlaneNormal, UnalignedF32Slice};
use crate::reader::item_leaf;
use crate::{
    Database, Distance, Error, ItemId, Key, KeyCodec, Metadata, MetadataCodec, Node, NodeCodec,
    Prefix, PrefixCodec, Result, Side,
};

#[derive(Debug)]
pub struct Writer<D: Distance> {
    database: Database<D>,
    index: u8,
    dimensions: usize,
    // non-initiliazed until build is called.
    n_items: usize,
    // non-initiliazed until build is called.
    next_tree_id: u32,
    // We know the root nodes points to tree-nodes.
    roots: Vec<ItemId>,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance> Writer<D> {
    pub fn prepare<U>(
        wtxn: &mut RwTxn,
        database: heed::Database<KeyCodec, U>,
        index: u8,
        dimensions: usize,
    ) -> Result<Writer<D>> {
        if index > 0b0111_1111 {
            return Err(Error::InvalidIndex { received: index });
        }
        // we keep the lowest bit for our mode
        let index = index << 1;

        let database: Database<D> = database.remap_data_type();
        clear_tree_nodes(wtxn, database, index)?;
        Ok(Writer {
            database,
            index,
            dimensions,
            n_items: 0,
            next_tree_id: 0,
            roots: Vec::new(),
            _marker: marker::PhantomData,
        })
    }

    pub fn prepare_changing_distance<ND: Distance>(self, wtxn: &mut RwTxn) -> Result<Writer<ND>> {
        if TypeId::of::<ND>() != TypeId::of::<D>() {
            clear_tree_nodes(wtxn, self.database, self.index)?;

            let mut cursor = self.database.iter_mut(wtxn)?;
            while let Some((item_id, node)) = cursor.next().transpose()? {
                match node {
                    Node::Leaf(Leaf { header: _, vector }) => {
                        let new_leaf = Node::Leaf(Leaf {
                            header: ND::new_header(&vector),
                            vector: Cow::Owned(vector.into_owned()),
                        });
                        unsafe {
                            // safety: We do not keep a reference to the current value, we own it.
                            cursor.put_current_with_options::<NodeCodec<ND>>(
                                PutFlags::empty(),
                                &item_id,
                                &new_leaf,
                            )?
                        };
                    }
                    Node::Descendants(_) | Node::SplitPlaneNormal(_) => panic!(),
                }
            }
        }

        let Writer { database, index, dimensions, n_items, next_tree_id, roots, _marker: _ } = self;
        Ok(Writer {
            database: database.remap_data_type(),
            index,
            dimensions,
            n_items,
            next_tree_id,
            roots,
            _marker: marker::PhantomData,
        })
    }

    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> Result<Option<Vec<f32>>> {
        Ok(item_leaf(self.database, self.index, rtxn, item)?.map(|leaf| leaf.vector.into_owned()))
    }

    /// Returns an iterator over the items vector.
    pub fn iter<'t>(&self, rtxn: &'t RoTxn) -> Result<ItemIter<'t, D>> {
        Ok(ItemIter {
            inner: self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter(rtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>(),
        })
    }

    /// Add an item associated to a vector in the database.
    pub fn add_item(&self, wtxn: &mut RwTxn, item: ItemId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(Error::InvalidVecDimension {
                expected: self.dimensions,
                received: vector.len(),
            });
        }

        let vector = UnalignedF32Slice::from_slice(vector);
        let leaf = Leaf { header: D::new_header(vector), vector: Cow::Borrowed(vector) };
        Ok(self.database.put(wtxn, &Key::item(self.index, item), &Node::Leaf(leaf))?)
    }

    pub fn del_item(&self, wtxn: &mut RwTxn, item: ItemId) -> Result<bool> {
        Ok(self.database.delete(wtxn, &Key::item(self.index, item))?)
    }

    pub fn clear(&self, wtxn: &mut RwTxn) -> Result<()> {
        let mut cursor = self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter_mut(wtxn, &Prefix::item(self.index))?
            .remap_key_type::<DecodeIgnore>();

        while let Some((_id, _node)) = cursor.next().transpose()? {
            // safety: we don't have any reference to the database
            unsafe { cursor.del_current() }?;
        }
        drop(cursor);

        let mut cursor = self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter_mut(wtxn, &Prefix::tree(self.index))?
            .remap_key_type::<DecodeIgnore>();

        while let Some((_id, _node)) = cursor.next().transpose()? {
            // safety: we don't have any reference to the database
            unsafe { cursor.del_current() }?;
        }

        Ok(())
    }

    pub fn build<R: Rng>(
        mut self,
        wtxn: &mut RwTxn,
        mut rng: R,
        n_trees: Option<usize>,
    ) -> Result<()> {
        D::preprocess(wtxn, |wtxn| {
            Ok(self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter_mut(wtxn, &Prefix::item(self.index))?
                .remap_key_type::<KeyCodec>())
        })?;

        let item_indices = self.item_indices(wtxn)?;
        self.n_items = item_indices.len();

        let mut thread_roots = Vec::new();
        loop {
            match n_trees {
                Some(n_trees) if thread_roots.len() >= n_trees => break,
                None if self.next_tree_id as usize >= self.n_items => break,
                _ => (),
            }

            let tree_root_id = self.make_tree(wtxn, &item_indices, true, &mut rng)?;
            // make_tree must NEVER return a leaf when called as root
            thread_roots.push(tree_root_id.unwrap_tree());
        }

        self.roots.append(&mut thread_roots);

        // Also, copy the roots into the highest key of the database (u32::MAX).
        // This way we can load them faster without reading the whole database.
        let metadata = Metadata {
            dimensions: self.dimensions.try_into().unwrap(),
            n_items: self.n_items.try_into().unwrap(),
            roots: ItemIds::from_slice(&self.roots),
        };
        match self.database.remap_data_type::<MetadataCodec>().put_with_flags(
            wtxn,
            PutFlags::NO_OVERWRITE,
            &Key::metadata(self.index),
            &metadata,
        ) {
            Ok(_) => (),
            Err(heed::Error::Mdb(MdbError::KeyExist)) => return Err(Error::DatabaseFull),
            Err(e) => return Err(e.into()),
        }

        Ok(())
    }

    /// Creates a tree of nodes from the items the user provided
    /// and generates descendants, split normal and root nodes.
    fn make_tree<R: Rng>(
        &mut self,
        wtxn: &mut RwTxn,
        item_indices: &[ItemId],
        is_root: bool,
        rng: &mut R,
    ) -> Result<Key> {
        // we simplify the max descendants (_K) thing by considering
        // that we can fit as much descendants as the number of dimensions
        let max_descendants = self.dimensions;

        if item_indices.len() == 1 && !is_root {
            return Ok(Key::item(self.index, item_indices[0]));
        }

        if item_indices.len() <= max_descendants
            && (!is_root || self.n_items <= max_descendants || item_indices.len() == 1)
        {
            let item_id = self.create_item_id()?;

            let item =
                Node::Descendants(Descendants { descendants: ItemIds::from_slice(item_indices) });
            self.database.put(wtxn, &Key::tree(self.index, item_id), &item)?;
            return Ok(Key::tree(self.index, item_id));
        }

        let mut children = Vec::new();
        for &item_id in item_indices {
            let node = self.database.get(wtxn, &Key::item(self.index, item_id))?.unwrap();
            let leaf = node.leaf().unwrap();
            children.push(leaf);
        }

        let mut children_left = Vec::new();
        let mut children_right = Vec::new();
        let mut remaining_attempts = 3;

        let mut normal = loop {
            children_left.clear();
            children_right.clear();

            let normal = D::create_split(&children, rng);
            for (&node_id, node) in item_indices.iter().zip(&children) {
                match D::side(UnalignedF32Slice::from_slice(&normal), node, rng) {
                    Side::Left => children_left.push(node_id),
                    Side::Right => children_right.push(node_id),
                }
            }

            if split_imbalance(children_left.len(), children_right.len()) < 0.95
                || remaining_attempts == 0
            {
                break normal;
            }

            remaining_attempts -= 1;
        };

        // If we didn't find a hyperplane, just randomize sides as a last option
        // and set the split plane to zero as a dummy plane.
        while split_imbalance(children_left.len(), children_right.len()) > 0.99 {
            children_left.clear();
            children_right.clear();
            normal.fill(0.0);

            for &node_id in item_indices {
                match Side::random(rng) {
                    Side::Left => children_left.push(node_id),
                    Side::Right => children_right.push(node_id),
                }
            }
        }

        let normal = SplitPlaneNormal {
            normal: Cow::Owned(normal),
            left: self.make_tree(wtxn, &children_left, false, rng)?,
            right: self.make_tree(wtxn, &children_right, false, rng)?,
        };

        let new_node_id = self.create_item_id()?;
        self.database.put(
            wtxn,
            &Key::tree(self.index, new_node_id),
            &Node::SplitPlaneNormal(normal),
        )?;

        Ok(Key::tree(self.index, new_node_id))
    }

    fn create_item_id(&mut self) -> Result<ItemId> {
        let old = self.next_tree_id;
        self.next_tree_id = self.next_tree_id.checked_add(1).ok_or(Error::DatabaseFull)?;

        Ok(old)
    }

    /// Returns the index this writer was opened with
    pub fn index(&self) -> u8 {
        self.index >> 1
    }

    // Fetches the item's ids, not the tree nodes ones.
    fn item_indices(&self, wtxn: &mut RwTxn<'_>) -> heed::Result<Vec<ItemId>> {
        let mut indices = Vec::new();
        for result in self
            .database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(wtxn, &Prefix::item(self.index))?
            .remap_key_type::<KeyCodec>()
        {
            let (key, _) = result?;
            indices.push(key.unwrap_item());
        }

        Ok(indices)
    }
}

/// Clears everything but the leafs nodes (items).
/// Starts from the last node and stops at the first leaf.
fn clear_tree_nodes<D: Distance>(wtxn: &mut RwTxn, database: Database<D>, index: u8) -> Result<()> {
    database.delete(wtxn, &Key::metadata(index))?;
    let mut cursor = database
        .remap_types::<PrefixCodec, DecodeIgnore>()
        .prefix_iter_mut(wtxn, &Prefix::tree(index))?
        .remap_key_type::<DecodeIgnore>();
    while let Some((_id, _node)) = cursor.next().transpose()? {
        // safety: we keep no reference into the database between operations
        unsafe { cursor.del_current()? };
    }

    Ok(())
}

fn split_imbalance(left_indices_len: usize, right_indices_len: usize) -> f64 {
    let ls = left_indices_len as f64;
    let rs = right_indices_len as f64;
    let f = ls / (ls + rs + f64::EPSILON); // Avoid 0/0
    f.max(1.0 - f)
}
