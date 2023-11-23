use std::any::TypeId;
use std::borrow::Cow;
use std::marker;

use heed::types::DecodeIgnore;
use heed::{MdbError, PutFlags, RoTxn, RwTxn};
use rand::Rng;

use crate::item_iter::ItemIter;
use crate::node::{Descendants, ItemIds, Leaf};
use crate::reader::item_leaf;
use crate::{
    Database, Distance, Error, ItemId, Key, KeyCodec, Metadata, MetadataCodec, Node, NodeCodec,
    NodeId, Prefix, PrefixCodec, Reader, Result, Side,
};

#[derive(Debug)]
pub struct Writer<D: Distance> {
    database: Database<D>,
    prefix: u16,
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
        prefix: u16,
        dimensions: usize,
    ) -> Result<Writer<D>> {
        let database: Database<D> = database.remap_data_type();
        clear_tree_nodes(wtxn, database, prefix)?;
        Ok(Writer {
            database,
            prefix,
            dimensions,
            n_items: 0,
            next_tree_id: 0,
            roots: Vec::new(),
            _marker: marker::PhantomData,
        })
    }

    pub fn prepare_changing_distance<ND: Distance>(self, wtxn: &mut RwTxn) -> Result<Writer<ND>> {
        if TypeId::of::<ND>() != TypeId::of::<D>() {
            clear_tree_nodes(wtxn, self.database, self.prefix)?;

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

        let Writer { database, prefix, dimensions, n_items, next_tree_id, roots, _marker: _ } =
            self;
        Ok(Writer {
            database: database.remap_data_type(),
            prefix,
            dimensions,
            n_items,
            next_tree_id,
            roots,
            _marker: marker::PhantomData,
        })
    }

    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> Result<Option<Vec<f32>>> {
        Ok(item_leaf(self.database, self.prefix, rtxn, item)?.map(|leaf| leaf.vector.into_owned()))
    }

    /// Returns an iterator over the items vector.
    pub fn iter<'t>(&self, rtxn: &'t RoTxn) -> Result<ItemIter<'t, D>> {
        Ok(ItemIter {
            inner: self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter(rtxn, &Prefix::item(self.prefix))?
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

        let leaf = Leaf { header: D::new_header(vector), vector: Cow::Borrowed(vector) };
        Ok(self.database.put(wtxn, &Key::item(self.prefix, item), &Node::Leaf(leaf))?)
    }

    pub fn del_item(&self, wtxn: &mut RwTxn, item: ItemId) -> Result<bool> {
        Ok(self.database.delete(wtxn, &Key::item(self.prefix, item))?)
    }

    pub fn clear(&self, wtxn: &mut RwTxn) -> Result<()> {
        let mut cursor = self
            .database
            .remap_key_type::<PrefixCodec>()
            .prefix_iter_mut(wtxn, &Prefix::all(self.prefix))?
            .remap_key_type::<KeyCodec>();

        while let Some((_id, _node)) = cursor.next().transpose()? {
            // We don't have any reference to the database
            unsafe { cursor.del_current() }?;
        }

        Ok(())
    }

    pub fn build<'t, R: Rng>(
        mut self,
        wtxn: &'t mut RwTxn,
        mut rng: R,
        n_trees: Option<usize>,
    ) -> Result<Reader<'t, D>> {
        D::preprocess(wtxn, |wtxn| {
            Ok(self
                .database
                .remap_key_type::<PrefixCodec>()
                .prefix_iter_mut(wtxn, &Prefix::item(self.prefix))?
                .remap_key_type::<KeyCodec>())
        })?;

        self.n_items = self.n_items(wtxn)? as usize;

        let mut thread_roots = Vec::new();
        loop {
            match n_trees {
                Some(n_trees) if thread_roots.len() >= n_trees => break,
                None if self.next_tree_id as usize >= self.n_items => break,
                _ => (),
            }

            let mut indices = Vec::new();
            // Only fetch the item's ids, not the tree nodes ones
            for result in self
                .database
                .remap_types::<PrefixCodec, DecodeIgnore>()
                .prefix_iter(wtxn, &Prefix::item(self.prefix))?
                .remap_key_type::<KeyCodec>()
            {
                let (i, _) = result?;
                indices.push(i.node);
            }

            let tree_root_id = self.make_tree(wtxn, indices, true, &mut rng)?;
            // make_tree must NEVER return a leaf
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
            &Key::root(self.prefix),
            &metadata,
        ) {
            Ok(_) => (),
            Err(heed::Error::Mdb(MdbError::KeyExist)) => return Err(Error::DatabaseFull),
            Err(e) => return Err(e.into()),
        }

        Reader::open(wtxn, self.prefix, self.database)
    }

    /// Creates a tree of nodes from the items the user provided
    /// and generates descendants, split normal and root nodes.
    fn make_tree<R: Rng>(
        &mut self,
        wtxn: &mut RwTxn,
        indices: Vec<NodeId>,
        is_root: bool,
        rng: &mut R,
    ) -> Result<NodeId> {
        // we simplify the max descendants (_K) thing by considering
        // that we can fit as much descendants as the number of dimensions
        let max_descendants = self.dimensions;

        if indices.len() == 1 && !is_root {
            return Ok(indices[0]);
        }

        if indices.len() <= max_descendants
            && (!is_root || self.n_items <= max_descendants || indices.len() == 1)
        {
            let item_id = self.create_item_id()?;

            // If we reached this point, we know we only holds leaf in our indices.
            let indices: Vec<ItemId> =
                indices.into_iter().map(|node_id| node_id.unwrap_item()).collect();
            let item =
                Node::Descendants(Descendants { descendants: ItemIds::from_slice(&indices) });
            self.database.put(wtxn, &Key::tree(self.prefix, item_id), &item)?;
            return Ok(NodeId::tree(item_id));
        }

        let mut children = Vec::new();
        for node_id in &indices {
            let node = self.database.get(wtxn, &Key::new(self.prefix, *node_id))?.unwrap();
            let leaf = node.leaf().unwrap();
            children.push(leaf);
        }

        let mut children_left = Vec::new();
        let mut children_right = Vec::new();
        let mut remaining_attempts = 3;

        let mut m = loop {
            children_left.clear();
            children_right.clear();

            let m = D::create_split(&children, rng);
            for (&node_id, node) in indices.iter().zip(&children) {
                match D::side(&m, node, rng) {
                    Side::Left => children_left.push(node_id),
                    Side::Right => children_right.push(node_id),
                }
            }

            if split_imbalance(children_left.len(), children_right.len()) < 0.95
                || remaining_attempts == 0
            {
                break m;
            }

            remaining_attempts -= 1;
        };

        // If we didn't find a hyperplane, just randomize sides as a last option
        // and set the split plane to zero as a dummy plane.
        while split_imbalance(children_left.len(), children_right.len()) > 0.99 {
            children_left.clear();
            children_right.clear();

            m.normal.to_mut().fill(0.0);

            for &node_id in &indices {
                match Side::random(rng) {
                    Side::Left => children_left.push(node_id),
                    Side::Right => children_right.push(node_id),
                }
            }
        }

        // TODO make sure to run _make_tree for the smallest child first (for cache locality)
        m.left = self.make_tree(wtxn, children_left, false, rng)?;
        m.right = self.make_tree(wtxn, children_right, false, rng)?;

        let new_node_id = self.create_item_id()?;

        self.database.put(
            wtxn,
            &Key::tree(self.prefix, new_node_id),
            &Node::SplitPlaneNormal(m),
        )?;
        Ok(NodeId::tree(new_node_id))
    }

    fn create_item_id(&mut self) -> Result<ItemId> {
        let old = self.next_tree_id;
        self.next_tree_id = self.next_tree_id.checked_add(1).ok_or(Error::DatabaseFull)?;

        Ok(old)
    }

    fn n_items(&self, rtxn: &RoTxn) -> Result<u64> {
        self.database
            .remap_types::<PrefixCodec, DecodeIgnore>()
            .prefix_iter(rtxn, &Prefix::item(self.prefix))?
            .remap_key_type::<DecodeIgnore>()
            .try_fold(0, |acc, el| -> Result<u64> {
                el?;
                Ok(acc + 1)
            })
    }
}

/// Clears everything but the leafs nodes (items).
/// Starts from the last node and stops at the first leaf.
fn clear_tree_nodes<D: Distance>(
    wtxn: &mut RwTxn,
    database: Database<D>,
    prefix: u16,
) -> Result<()> {
    database.delete(wtxn, &Key::root(prefix))?;
    let mut cursor = database
        .remap_types::<PrefixCodec, DecodeIgnore>()
        .prefix_iter_mut(wtxn, &Prefix::tree(prefix))?
        .remap_key_type::<KeyCodec>();
    while let Some((_id, _node)) = cursor.next().transpose()? {
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
