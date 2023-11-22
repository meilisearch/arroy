use std::any::TypeId;
use std::borrow::Cow;
use std::marker;

use heed::types::DecodeIgnore;
use heed::{Database, MdbError, PutFlags, RoTxn, RwTxn};
use rand::Rng;

use crate::item_iter::ItemIter;
use crate::node::{Descendants, Leaf, NodeIds};
use crate::reader::item_leaf;
use crate::{
    Distance, Error, ItemId, Metadata, MetadataCodec, Node, NodeCodec, NodeId, Reader, Result,
    Side, BEU32,
};

#[derive(Debug)]
pub struct Writer<D: Distance> {
    database: heed::Database<BEU32, NodeCodec<D>>,
    dimensions: usize,
    // non-initiliazed until build is called.
    n_items: usize,
    roots: Vec<NodeId>,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance> Writer<D> {
    pub fn prepare<U>(
        wtxn: &mut RwTxn,
        database: Database<BEU32, U>,
        dimensions: usize,
    ) -> Result<Writer<D>> {
        let database = database.remap_data_type();
        clear_tree_nodes(wtxn, database)?;
        Ok(Writer {
            database,
            dimensions,
            n_items: 0,
            roots: Vec::new(),
            _marker: marker::PhantomData,
        })
    }

    pub fn prepare_changing_distance<ND: Distance>(self, wtxn: &mut RwTxn) -> Result<Writer<ND>> {
        if TypeId::of::<ND>() != TypeId::of::<D>() {
            clear_tree_nodes(wtxn, self.database)?;

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
                            cursor
                                .put_current_with_data_codec::<NodeCodec<ND>>(&item_id, &new_leaf)?
                        };
                    }
                    Node::Descendants(_) | Node::SplitPlaneNormal(_) => panic!(),
                }
            }
        }

        let Writer { database, dimensions, n_items, roots, _marker: _ } = self;
        Ok(Writer {
            database: database.remap_data_type(),
            dimensions,
            n_items,
            roots,
            _marker: marker::PhantomData,
        })
    }

    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> Result<Option<Vec<f32>>> {
        Ok(item_leaf(self.database, rtxn, item)?.map(|leaf| leaf.vector.into_owned()))
    }

    /// Returns an iterator over the items vector.
    pub fn iter<'t>(&self, rtxn: &'t RoTxn) -> Result<ItemIter<'t, D>> {
        self.database.iter(rtxn).map(|inner| ItemIter { inner }).map_err(Into::into)
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
        Ok(self.database.put(wtxn, &item, &Node::Leaf(leaf))?)
    }

    pub fn del_item(&self, wtxn: &mut RwTxn, item: ItemId) -> Result<bool> {
        Ok(self.database.delete(wtxn, &item)?)
    }

    pub fn clear(&self, wtxn: &mut RwTxn) -> Result<()> {
        Ok(self.database.clear(wtxn)?)
    }

    pub fn build<'t, R: Rng>(
        mut self,
        wtxn: &'t mut RwTxn,
        mut rng: R,
        n_trees: Option<usize>,
    ) -> Result<Reader<'t, D>> {
        D::preprocess(wtxn, |wtxn| self.database.iter_mut(wtxn))?;

        self.n_items = self.database.len(wtxn)? as usize;
        let last_item_id = self.last_node_id(wtxn)?;

        let mut thread_roots = Vec::new();
        loop {
            match n_trees {
                Some(n_trees) if thread_roots.len() >= n_trees => break,
                None if self.database.len(wtxn)? >= 2 * self.n_items as u64 => break,
                _ => (),
            }

            let mut indices = Vec::new();
            // Only fetch the item's ids, not the tree nodes ones
            for result in self.database.remap_data_type::<DecodeIgnore>().iter(wtxn)? {
                let (i, _) = result?;
                if last_item_id.map_or(true, |last| i > last) {
                    break;
                }
                indices.push(i);
            }

            let tree_root_id = self.make_tree(wtxn, indices, true, &mut rng)?;
            thread_roots.push(tree_root_id);
        }

        self.roots.append(&mut thread_roots);

        // Also, copy the roots into the highest key of the database (u32::MAX).
        // This way we can load them faster without reading the whole database.
        let metadata = Metadata {
            dimensions: self.dimensions.try_into().unwrap(),
            n_items: self.n_items.try_into().unwrap(),
            roots: NodeIds::from_slice(&self.roots),
        };
        match self.database.remap_data_type::<MetadataCodec>().put_with_flags(
            wtxn,
            PutFlags::NO_OVERWRITE,
            &u32::MAX,
            &metadata,
        ) {
            Ok(_) => (),
            Err(heed::Error::Mdb(MdbError::KeyExist)) => return Err(Error::DatabaseFull),
            Err(e) => return Err(e.into()),
        }

        Reader::open(wtxn, self.database)
    }

    /// Creates a tree of nodes from the items the user provided
    /// and generates descendants, split normal and root nodes.
    fn make_tree<R: Rng>(
        &self,
        wtxn: &mut RwTxn,
        indices: Vec<u32>,
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
            let item_id = match self.last_node_id(wtxn)? {
                Some(last_id) => last_id.checked_add(1).ok_or(Error::DatabaseFull)?,
                None => 0,
            };

            let item =
                Node::Descendants(Descendants { descendants: NodeIds::from_slice(&indices) });
            self.database.put(wtxn, &item_id, &item)?;
            return Ok(item_id);
        }

        let mut children = Vec::new();
        for node_id in &indices {
            let node = self.database.get(wtxn, node_id)?.unwrap();
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

        let new_node_id = match self.last_node_id(wtxn)? {
            Some(last_id) => last_id.checked_add(1).unwrap(),
            None => 0,
        };

        self.database.put(wtxn, &new_node_id, &Node::SplitPlaneNormal(m))?;
        Ok(new_node_id)
    }

    fn last_node_id(&self, rtxn: &RoTxn) -> Result<Option<NodeId>> {
        match self.database.remap_data_type::<DecodeIgnore>().last(rtxn)? {
            Some((last_id, _)) => Ok(Some(last_id)),
            None => Ok(None),
        }
    }
}

/// Clears everything but the leafs nodes (items).
/// Starts from the last node and stops at the first leaf.
fn clear_tree_nodes<D: Distance>(
    wtxn: &mut RwTxn,
    database: Database<BEU32, NodeCodec<D>>,
) -> Result<()> {
    database.delete(wtxn, &u32::MAX)?;
    let mut cursor = database.rev_iter_mut(wtxn)?;
    while let Some((_id, node)) = cursor.next().transpose()? {
        if node.leaf().is_none() {
            unsafe { cursor.del_current()? };
        } else {
            break;
        }
    }

    Ok(())
}

fn split_imbalance(left_indices_len: usize, right_indices_len: usize) -> f64 {
    let ls = left_indices_len as f64;
    let rs = right_indices_len as f64;
    let f = ls / (ls + rs + f64::EPSILON); // Avoid 0/0
    f.max(1.0 - f)
}
