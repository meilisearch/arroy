use std::marker;

use bytemuck::checked::cast_slice;
use heed::types::{ByteSlice, DecodeIgnore};
use heed::{Database, RoTxn, RwTxn};
use rand::Rng;

use crate::node::{Descendants, Leaf};
use crate::reader::item_vector;
use crate::{Distance, ItemId, Node, NodeCodec, NodeId, Side, BEU32};

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
        dimensions: usize,
        database: Database<BEU32, U>,
    ) -> heed::Result<Writer<D>> {
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

    pub fn item_vector(&self, rtxn: &RoTxn, item: ItemId) -> heed::Result<Option<Vec<f32>>> {
        item_vector(self.database, rtxn, item)
    }

    pub fn add_item(&self, wtxn: &mut RwTxn, item: ItemId, vector: &[f32]) -> heed::Result<()> {
        // TODO make this not an assert
        assert_eq!(
            vector.len(),
            self.dimensions,
            "invalid vector dimensions, provided {} but expected {}",
            vector.len(),
            self.dimensions
        );

        // TODO find a way to not allocate the vector
        let leaf = Leaf { header: D::new_header(vector), vector: vector.to_vec() };
        self.database.put(wtxn, &item, &Node::Leaf(leaf))
    }

    pub fn del_item(&self, wtxn: &mut RwTxn, item: ItemId) -> heed::Result<bool> {
        todo!()
    }

    pub fn clear(&self, wtxn: &mut RwTxn) -> heed::Result<()> {
        self.database.clear(wtxn)
    }

    pub fn build<R: Rng>(
        mut self,
        wtxn: &mut RwTxn,
        mut rng: R,
        n_trees: Option<usize>,
    ) -> heed::Result<()> {
        // D::template preprocess<T, S, Node>(_nodes, _s, _n_items, _f);

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
        match self.database.get(wtxn, &u32::MAX)? {
            Some(_) => panic!("The database is full. We cannot write the root nodes ids"),
            None => {
                self.database.remap_data_type::<ByteSlice>().put(
                    wtxn,
                    &u32::MAX,
                    cast_slice(self.roots.as_slice()),
                )?;
            }
        }

        // D::template postprocess<T, S, Node>(_nodes, _s, _n_items, _f);

        Ok(())
    }

    /// Creates a tree of nodes from the items the user provided
    /// and generates descendants, split normal and root nodes.
    fn make_tree<R: Rng>(
        &self,
        wtxn: &mut RwTxn,
        indices: Vec<u32>,
        is_root: bool,
        rng: &mut R,
    ) -> heed::Result<NodeId> {
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
                Some(last_id) => last_id.checked_add(1).unwrap(),
                None => 0,
            };

            let item = Node::Descendants(Descendants { descendants: indices });
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

            m.normal.fill(0.0);

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

    fn last_node_id(&self, rtxn: &RoTxn) -> heed::Result<Option<NodeId>> {
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
) -> heed::Result<()> {
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

#[cfg(test)]
mod test {
    use std::fmt::{Display, Write};

    use bytemuck::pod_collect_to_vec;
    use heed::types::LazyDecode;
    use heed::{Env, EnvOpenOptions, Unspecified};
    use tempfile::TempDir;

    use super::*;
    use crate::Angular;

    struct DatabaseHandle {
        pub env: Env,
        pub database: Database<BEU32, Unspecified>,
        #[allow(unused)]
        pub tempdir: TempDir,
    }

    impl Display for DatabaseHandle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let rtxn = self.env.read_txn().unwrap();
            for result in self
                .database
                .remap_data_type::<LazyDecode<NodeCodec<Angular>>>()
                .iter(&rtxn)
                .unwrap()
            {
                let (i, lazy_node) = result.unwrap();
                if i != u32::MAX {
                    let node = lazy_node.decode().unwrap();
                    writeln!(f, "{i}: {node:?}")?;
                } else {
                    let roots_bytes = self
                        .database
                        .remap_data_type::<ByteSlice>()
                        .get(&rtxn, &i)
                        .unwrap()
                        .unwrap();
                    let roots: Vec<u32> = pod_collect_to_vec(roots_bytes);
                    writeln!(f, "\nu32::MAX: {roots:?}")?;
                }
            }
            Ok(())
        }
    }

    fn create_database() -> DatabaseHandle {
        let dir = tempfile::tempdir().unwrap();
        let env = EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()).unwrap();
        let mut wtxn = env.write_txn().unwrap();
        let database: Database<BEU32, Unspecified> = env.create_database(&mut wtxn, None).unwrap();
        wtxn.commit().unwrap();
        DatabaseHandle { env, database, tempdir: dir }
    }

    #[test]
    fn write_one_vector_in_one_tree() {
        let handle = create_database();
        let mut wtxn = handle.env.write_txn().unwrap();
        let writer = Writer::<Angular>::prepare(&mut wtxn, 3, handle.database).unwrap();
        writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

        let rng = rand::thread_rng();
        let _reader = writer.build(&mut wtxn, rng, Some(1)).unwrap();
        wtxn.commit().unwrap();

        insta::assert_display_snapshot!(handle, @r###"
        0: Leaf(Leaf { header: NodeHeaderAngular { norm: 2.236068 }, vector: [0.0, 1.0, 2.0] })
        1: Descendants(Descendants { descendants: [0] })

        u32::MAX: [1]
        "###);
    }

    #[test]
    fn write_one_vector_in_multiple_trees() {
        let handle = create_database();
        let mut wtxn = handle.env.write_txn().unwrap();
        let writer = Writer::<Angular>::prepare(&mut wtxn, 3, handle.database).unwrap();
        writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

        let rng = rand::thread_rng();
        let _reader = writer.build(&mut wtxn, rng, Some(10)).unwrap();
        wtxn.commit().unwrap();

        insta::assert_display_snapshot!(handle, @r###"
        0: Leaf(Leaf { header: NodeHeaderAngular { norm: 2.236068 }, vector: [0.0, 1.0, 2.0] })
        1: Descendants(Descendants { descendants: [0] })
        2: Descendants(Descendants { descendants: [0] })
        3: Descendants(Descendants { descendants: [0] })
        4: Descendants(Descendants { descendants: [0] })
        5: Descendants(Descendants { descendants: [0] })
        6: Descendants(Descendants { descendants: [0] })
        7: Descendants(Descendants { descendants: [0] })
        8: Descendants(Descendants { descendants: [0] })
        9: Descendants(Descendants { descendants: [0] })
        10: Descendants(Descendants { descendants: [0] })

        u32::MAX: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        "###);
    }
}
