use std::borrow::Cow;
use std::marker;

use bytemuck::checked::cast_slice;
use heed::types::{ByteSlice, DecodeIgnore};
use heed::{Database, RoTxn, RwTxn};
use rand::Rng;

use crate::node::{Descendants, Leaf};
use crate::reader::item_leaf;
use crate::{Distance, ItemId, Node, NodeCodec, NodeId, Side, BEU32};

pub struct Writer<D: Distance> {
    database: heed::Database<BEU32, NodeCodec<D>>,
    dimensions: usize,
    // non-initiliazed until build is called.
    n_items: usize,
    roots: Vec<NodeId>,
    _marker: marker::PhantomData<D>,
}

impl<D: Distance + 'static> Writer<D> {
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
        Ok(item_leaf(self.database, rtxn, item)?.map(|leaf| leaf.vector.into_owned()))
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
        let leaf = Leaf { header: D::new_header(vector), vector: Cow::Borrowed(vector) };
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

            let item = Node::Descendants(Descendants { descendants: Cow::Owned(indices) });
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

    fn last_node_id(&self, rtxn: &RoTxn) -> heed::Result<Option<NodeId>> {
        match self.database.remap_data_type::<DecodeIgnore>().last(rtxn)? {
            Some((last_id, _)) => Ok(Some(last_id)),
            None => Ok(None),
        }
    }
}

/// Clears everything but the leafs nodes (items).
/// Starts from the last node and stops at the first leaf.
fn clear_tree_nodes<D: Distance + 'static>(
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
pub mod test {
    use std::fmt::Display;

    use bytemuck::pod_collect_to_vec;
    use heed::types::LazyDecode;
    use heed::{Env, EnvOpenOptions, Unspecified};
    use rand::SeedableRng;
    use tempfile::TempDir;

    use super::*;
    use crate::Angular;

    pub struct DatabaseHandle {
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

    pub fn create_database() -> DatabaseHandle {
        let dir = tempfile::tempdir().unwrap();
        let env = EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()).unwrap();
        let mut wtxn = env.write_txn().unwrap();
        let database: Database<BEU32, Unspecified> = env.create_database(&mut wtxn, None).unwrap();
        wtxn.commit().unwrap();
        DatabaseHandle { env, database, tempdir: dir }
    }

    pub fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::from_seed(std::array::from_fn(|_| 42))
    }

    #[test]
    fn write_one_vector_in_one_tree() {
        let handle = create_database();
        let mut wtxn = handle.env.write_txn().unwrap();
        let writer = Writer::<Angular>::prepare(&mut wtxn, 3, handle.database).unwrap();
        writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

        writer.build(&mut wtxn, rng(), Some(1)).unwrap();
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

        writer.build(&mut wtxn, rng(), Some(10)).unwrap();
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

    #[test]
    fn write_vectors_until_there_is_a_descendants() {
        let handle = create_database();
        let mut wtxn = handle.env.write_txn().unwrap();
        let writer = Writer::<Angular>::prepare(&mut wtxn, 3, handle.database).unwrap();
        for i in 0..3 {
            let id = i;
            let i = i as f32;
            writer.add_item(&mut wtxn, id, &[i, i, i]).unwrap();
        }

        writer.build(&mut wtxn, rng(), Some(1)).unwrap();
        wtxn.commit().unwrap();

        insta::assert_display_snapshot!(handle, @r###"
        0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0, 0.0, 0.0] })
        1: Leaf(Leaf { header: NodeHeaderAngular { norm: 1.7320508 }, vector: [1.0, 1.0, 1.0] })
        2: Leaf(Leaf { header: NodeHeaderAngular { norm: 3.4641016 }, vector: [2.0, 2.0, 2.0] })
        3: Descendants(Descendants { descendants: [0, 1, 2] })

        u32::MAX: [3]
        "###);
    }

    #[test]
    fn write_vectors_until_there_is_a_split() {
        let handle = create_database();
        let mut wtxn = handle.env.write_txn().unwrap();
        let writer = Writer::<Angular>::prepare(&mut wtxn, 3, handle.database).unwrap();
        for i in 0..4 {
            let id = i;
            let i = i as f32;
            writer.add_item(&mut wtxn, id, &[i, i, i]).unwrap();
        }

        writer.build(&mut wtxn, rng(), Some(1)).unwrap();
        wtxn.commit().unwrap();

        insta::assert_display_snapshot!(handle, @r###"
        0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0, 0.0, 0.0] })
        1: Leaf(Leaf { header: NodeHeaderAngular { norm: 1.7320508 }, vector: [1.0, 1.0, 1.0] })
        2: Leaf(Leaf { header: NodeHeaderAngular { norm: 3.4641016 }, vector: [2.0, 2.0, 2.0] })
        3: Leaf(Leaf { header: NodeHeaderAngular { norm: 5.196152 }, vector: [3.0, 3.0, 3.0] })
        4: Descendants(Descendants { descendants: [1, 2, 3] })
        5: SplitPlaneNormal(SplitPlaneNormal { normal: [0.0, 0.0, 0.0], left: 0, right: 4 })

        u32::MAX: [5]
        "###);
    }

    #[test]
    fn write_a_lot_of_random_points() {
        let handle = create_database();
        let mut wtxn = handle.env.write_txn().unwrap();
        let writer = Writer::<Angular>::prepare(&mut wtxn, 30, handle.database).unwrap();
        let mut rng = rng();
        for id in 0..100 {
            let vector: [f32; 30] = std::array::from_fn(|_| rng.gen());
            writer.add_item(&mut wtxn, id, &vector).unwrap();
        }

        writer.build(&mut wtxn, rng, Some(10)).unwrap();
        wtxn.commit().unwrap();

        // we skip all the leaf to avoid flooding ourselves too much
        let s = handle
            .to_string()
            .lines()
            .skip_while(|line| line.contains("Leaf"))
            .collect::<Vec<&str>>()
            .join("\n");
        insta::assert_display_snapshot!(s, @r###"
        100: Descendants(Descendants { descendants: [16, 20, 21, 22, 24, 30, 31, 34, 35, 36, 39, 43, 46, 52, 54, 56, 59, 66, 67, 72, 80, 85, 89, 90, 91, 92, 93] })
        101: Descendants(Descendants { descendants: [8, 13, 19, 28, 41, 45, 51, 60, 63, 64, 68, 78, 83, 87] })
        102: SplitPlaneNormal(SplitPlaneNormal { normal: [0.01625332, -0.03969399, -0.43729657, 0.034380652, 0.40609843, 0.030681998, 0.019480277, -0.09044069, 0.026958808, -0.021964513, -0.1427401, 0.012944222, -0.21064779, 0.29124254, 0.10624939, -0.013896785, 0.06552808, 0.06193905, 0.11892321, -0.10734607, -0.053229064, 0.10669195, -0.21512093, 0.29982722, -0.24642143, -0.35716367, 0.12232927, -0.013265043, -0.28481594, 0.013008841], left: 100, right: 101 })
        103: Descendants(Descendants { descendants: [4, 12, 15, 18, 32, 49, 53, 57, 58, 61, 69, 75, 81, 82, 84, 88, 94, 99] })
        104: Descendants(Descendants { descendants: [6, 17, 23, 37, 47, 55, 71, 73, 74, 77, 79, 86, 95] })
        105: SplitPlaneNormal(SplitPlaneNormal { normal: [0.18556727, 0.3279048, -0.31312305, -0.07537248, -0.13704272, -0.07991075, -0.10438369, 0.10597492, 0.14081477, 0.031024769, 0.018912876, -0.078412734, -0.369496, 0.037646126, 0.027761098, 0.17990145, 0.1120954, -0.26636386, -0.42894384, 0.35052556, 0.13845795, 0.07326738, 0.068004765, 0.12580833, 0.05191296, -0.14754887, 0.11841216, -0.18376996, -0.0053913547, -0.031483363], left: 103, right: 104 })
        106: Descendants(Descendants { descendants: [0, 1, 2, 3, 5, 7, 9, 10, 11, 14, 25, 26, 27, 29, 33, 38, 40, 42, 44, 48, 50, 62, 65, 70, 76, 96, 97, 98] })
        107: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.27855277, 0.011969785, 0.08146074, 0.1828015, 0.05819895, 0.2746265, -0.032797504, -0.17927803, -0.3089917, 0.033607554, -0.26319543, -0.09568312, -0.157078, 0.10759606, -0.358909, 0.08823572, 0.03850366, -0.24742849, 0.19027866, 0.31661242, -0.020296905, -0.051165197, -0.051584527, -0.21315901, 0.24884838, 0.16896838, 0.07521751, 0.08604065, 0.23332472, -0.118525855], left: 105, right: 106 })
        108: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.17736381, -0.006371736, 0.016802901, -0.036275286, -0.052111678, 0.081598125, 0.064470895, 0.354013, 0.046185274, 0.4671995, 0.11379122, 0.08996324, 0.10815429, 0.17561986, -0.026565742, -0.14255291, -0.13535392, -0.1996413, -0.13677888, 0.1806355, -0.041689802, -0.13973509, -0.35580137, -0.25388643, 0.4133126, -0.12658316, 0.0036246115, 0.0019280507, -0.09425121, 0.07381116], left: 102, right: 107 })
        109: Descendants(Descendants { descendants: [8, 10, 13, 29, 33, 34, 44, 45, 48, 50, 51, 56, 61, 63, 74, 76, 83, 86, 89, 92] })
        110: Descendants(Descendants { descendants: [2, 5, 9, 12, 15, 16, 23, 24, 43, 46, 47, 58, 59, 64, 67, 71, 72, 78, 80, 87, 94, 97, 98] })
        111: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.13424322, 0.36610976, 0.24592736, -0.28743353, -0.05910448, -0.017567419, 0.10537477, 0.06917676, 0.20936698, 0.23979133, 0.38355905, -0.06542441, -0.2488073, -0.04689046, 0.031155385, 0.025637489, -0.21002293, 0.12072455, 0.0647727, -0.17110738, -0.079629414, -0.09692957, 0.12968001, 0.042183563, -0.38921526, -0.1024931, 0.21619704, -0.12755194, 0.041157763, 0.116194725], left: 109, right: 110 })
        112: Descendants(Descendants { descendants: [1, 4, 6, 11, 19, 21, 25, 27, 28, 30, 37, 39, 40, 41, 42, 52, 54, 62, 66, 73, 77, 91, 93, 96] })
        113: Descendants(Descendants { descendants: [0, 3, 7, 14, 26, 31, 35, 36, 57, 65, 69, 70, 81, 85, 90, 95] })
        114: Descendants(Descendants { descendants: [17, 18, 20, 22, 32, 38, 49, 53, 55, 60, 68, 75, 79, 82, 84, 88, 99] })
        115: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.0524527, -0.06305983, -0.11649934, -0.0012920233, -0.09681346, 0.061425507, -0.0025004994, -0.13958338, 0.10907478, 0.045828592, 0.28380275, 0.28759366, 0.16504854, -0.07613818, 0.4856375, -0.25591096, -0.029699776, 0.23163868, 0.015009494, -0.43509173, -0.19215453, 0.046200734, 0.05329473, -0.06933652, -0.19649859, -0.15992823, 0.10840128, 0.009714843, -0.24587722, -0.09041427], left: 113, right: 114 })
        116: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.11771611, -0.09553633, -0.11839958, 0.024758026, -0.10539523, -0.0821912, 0.32237732, -0.055203393, 0.07181618, 0.38039342, 0.012460858, 0.082575664, 0.33113036, -0.02417131, 0.23031001, -0.4256573, -0.13603504, 0.3103537, 0.122073494, -0.08940767, -0.054621104, -0.17133562, -0.029347489, -0.07284014, 0.16973229, 0.01526716, -0.35222134, 0.0077801137, -0.0013044429, 0.08286023], left: 112, right: 115 })
        117: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.04148136, 0.049241483, 0.07580433, 0.11914422, -0.19602169, -0.08884349, -0.012469575, 0.025993342, 0.45847654, 0.04171318, 0.1559513, 0.22236335, 0.06879678, 0.19559413, -0.09868415, 0.38653633, 0.046592344, 0.051169366, -0.12997848, -0.10077684, -0.13274859, 0.0028357538, -0.049805783, -0.20889023, 0.1568575, -0.11485026, -0.36617252, -0.25997236, 0.21146172, -0.24729285], left: 111, right: 116 })
        118: Descendants(Descendants { descendants: [1, 3, 5, 9, 10, 14, 15, 18, 22, 30, 31, 32, 39, 42, 52, 62, 70, 80, 89] })
        119: Descendants(Descendants { descendants: [2, 4, 6, 7, 11, 17, 29, 33, 36, 55, 66, 67, 86, 93, 95, 96, 97] })
        120: SplitPlaneNormal(SplitPlaneNormal { normal: [0.2342128, 0.08030951, 0.08339744, 0.16865982, 0.16633713, -0.013527187, 0.17700669, 0.051341347, -0.07183851, 0.0151873315, -0.122949496, -0.17701778, -0.015515677, -0.12534827, 0.369863, -0.023749009, 0.2768047, -0.17695546, -0.38956937, -0.08695174, -0.016948096, 0.0055830698, -0.28424183, -0.07797943, -0.012121763, -0.22438335, 0.32286566, -0.2760743, 0.048060566, 0.22045591], left: 118, right: 119 })
        121: Descendants(Descendants { descendants: [0, 8, 12, 20, 23, 25, 34, 35, 44, 46, 49, 53, 58, 59, 63, 69, 71, 74, 75, 81, 82, 84, 88, 92, 94, 98, 99] })
        122: SplitPlaneNormal(SplitPlaneNormal { normal: [0.17183405, -0.18494184, -0.2625619, 0.13870606, 0.09213427, 0.030140208, 0.06310458, -0.25403965, 0.3118504, -0.045314312, 0.22484277, 0.02225946, 0.23127884, -0.113136314, 0.0107076075, -0.34929472, 0.014875503, 0.032533947, 0.27901745, -0.30990648, -0.040843535, -0.11738862, -0.043644257, 0.018016404, -0.08426814, -0.23303093, 0.22558996, 0.23751251, -0.25724313, 0.060656644], left: 120, right: 121 })
        123: Descendants(Descendants { descendants: [13, 19, 24, 38, 41, 43, 45, 47, 48, 50, 51, 56, 57, 61, 64, 65, 72, 76, 78, 83] })
        124: Descendants(Descendants { descendants: [16, 21, 26, 27, 28, 37, 40, 54, 60, 68, 73, 77, 79, 85, 87, 90, 91] })
        125: SplitPlaneNormal(SplitPlaneNormal { normal: [0.03130058, 0.18156847, 0.21285993, 0.103604324, -0.2386227, -0.20907225, -0.18743752, -0.15101728, 0.16725044, -0.0011657355, 0.19045946, -0.054160558, -0.10246557, 0.10960757, -0.3632132, 0.081641525, 0.2050039, -0.23030327, -0.2126656, -0.054425605, -0.30343473, 0.2702454, -0.09622262, 0.34853375, 0.10632425, -0.18891807, 0.046799064, -0.12465511, 0.12978117, 0.049722943], left: 123, right: 124 })
        126: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.1312112, 0.18818863, -0.38253522, 0.097510636, 0.0584542, 0.32583308, 0.10893987, -0.19848756, 0.06520269, -0.13890485, -0.07045288, -0.17264637, -0.25586605, 0.2804965, -0.055645097, 0.06551341, 0.17929886, -0.15850309, 0.028581414, 0.0038058402, 0.050736904, -0.024026657, 0.007670757, 0.4152641, -0.20077547, 0.19806197, -0.08027669, -0.18796057, -0.10909973, -0.23206556], left: 122, right: 125 })
        127: Descendants(Descendants { descendants: [0, 1, 4, 7, 9, 10, 11, 29, 31, 33, 34, 35, 42, 44, 48, 55, 65, 70, 74, 86, 92, 97, 98] })
        128: Descendants(Descendants { descendants: [13, 20, 22, 23, 28, 36, 37, 38, 39, 45, 50, 58, 63, 76, 80, 83, 89, 93, 95, 96] })
        129: SplitPlaneNormal(SplitPlaneNormal { normal: [0.05694411, 0.24910937, 0.090614505, 0.17520176, -0.03844601, -0.06812057, -0.1742166, 0.060665842, -0.12455139, -0.017917173, -0.14942999, -0.05900423, 0.14140002, 0.08407566, 0.23110114, -0.0867853, -0.17448236, -0.009301468, -0.01670315, -0.17423654, 0.21752901, 0.18124764, 0.23612213, 0.3659228, -0.33379167, 0.22607204, -0.36081666, -0.20914808, -0.15171781, -0.18849924], left: 127, right: 128 })
        130: Descendants(Descendants { descendants: [2, 3, 6, 14, 15, 17, 21, 25, 26, 27, 32, 40, 47, 49, 54, 57, 62, 68, 69, 71, 73, 77, 79, 81, 85, 88, 90, 91, 94, 99] })
        131: Descendants(Descendants { descendants: [5, 8, 12, 16, 18, 19, 24, 30, 41, 43, 46, 51, 52, 53, 56, 59, 60, 61, 64, 66, 67, 72, 75, 78, 82, 84, 87] })
        132: SplitPlaneNormal(SplitPlaneNormal { normal: [0.098814845, -0.21892492, -0.04639159, -0.13089101, 0.29918295, 0.08164901, 0.09908062, -0.17102747, -0.12293781, -0.36477786, -0.1867438, -0.070044816, -0.044486552, -0.119812116, 0.17438447, -0.08032481, -0.050395753, 0.22227244, 0.3252506, -0.20421635, 0.3183799, -0.06811254, 0.27253973, -0.07561059, -0.32370785, -0.085278004, 0.13021345, 0.05063233, 0.01253736, 0.17162144], left: 130, right: 131 })
        133: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.28883654, 0.2736963, 0.21895072, -0.08614311, -0.14646798, 0.218014, 0.031434696, -0.19213581, 0.41353047, -0.048396006, 0.23310447, -0.015197888, -0.0306501, 0.12079779, 0.07184795, -0.00026274633, 0.15419127, 0.24716407, -0.080473825, -0.39063436, -0.1056468, 8.900969e-5, -0.08700987, 0.08667995, -0.28154245, -0.13779715, -0.09668366, 0.15219508, -0.16997732, 0.031068167], left: 129, right: 132 })
        134: Descendants(Descendants { descendants: [1, 2, 10, 11, 13, 19, 24, 29, 30, 36, 39, 42, 44, 45, 46, 50, 56, 63, 65, 66, 67, 72, 80, 89, 97, 98] })
        135: Descendants(Descendants { descendants: [3, 6, 8, 16, 21, 27, 28, 37, 40, 54, 57, 62, 68, 73, 77, 85, 90, 91, 93, 95, 96] })
        136: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.0029770124, 0.23770188, 0.08868875, -0.065938905, -0.31842995, -0.03518789, 0.15655333, -0.03466768, 0.33669183, 0.10654795, 0.07451913, 0.19329932, -0.09715014, 0.030263763, -0.1400719, 0.2539012, 0.048709035, -0.053919815, -0.27023074, -0.25293326, -0.1145704, 0.007920524, -0.10564127, 0.5353334, -0.06440899, -0.06979195, -0.09587791, -0.20349908, -0.103223234, -0.1317844], left: 134, right: 135 })
        137: Descendants(Descendants { descendants: [4, 12, 15, 18, 20, 22, 23, 41, 43, 47, 49, 51, 53, 55, 58, 59, 61, 64, 69, 71, 76, 78, 79, 82, 83, 94, 99] })
        138: Descendants(Descendants { descendants: [0, 5, 7, 9, 14, 17, 25, 26, 31, 32, 33, 34, 35, 38, 48, 52, 60, 70, 74, 75, 81, 84, 86, 87, 88, 92] })
        139: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.23784819, -0.09385218, 0.13433951, 0.1302502, -0.2713463, 0.025226684, 0.0057616625, -0.0098076975, -0.08409845, -0.16942787, -0.33935776, 0.058660924, -0.104487896, -0.18655704, -0.24545807, 0.0021513824, 0.1722549, 0.07163203, 0.04722417, 0.270348, -0.14381427, 0.09457607, -0.010112073, -0.11641689, 0.47588605, -0.19342247, -0.08141719, 0.20445323, 0.28450572, 0.13489985], left: 137, right: 138 })
        140: SplitPlaneNormal(SplitPlaneNormal { normal: [0.031611905, -0.15087008, -0.14485994, -0.1564125, -0.053694956, -0.021092102, 0.19271319, 0.11383851, 0.13658114, 0.035488863, 0.050951112, 0.5708897, 0.021888867, -0.12603271, 0.15070206, -0.119693734, -0.163055, 0.29782015, 0.18377481, -0.20127891, -0.10118867, -0.2713996, -0.04997851, -0.18716405, 0.16679673, -0.14121817, -0.026741985, 0.21980959, -0.2589983, 0.025834382], left: 136, right: 139 })
        141: Descendants(Descendants { descendants: [4, 8, 12, 17, 22, 23, 37, 43, 47, 52, 53, 55, 64, 68, 71, 73, 74, 78, 87, 95] })
        142: Descendants(Descendants { descendants: [15, 18, 20, 24, 32, 35, 38, 49, 57, 58, 59, 60, 61, 65, 69, 75, 76, 79, 82, 83, 84, 88, 94, 99] })
        143: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.19239663, -0.27655548, 0.112270854, 0.3315285, 0.018722609, 0.30358502, -0.026712867, -0.091236606, 0.011320183, 0.080924205, -0.052898224, 0.29491597, 0.2784867, -0.12497674, 0.05269807, -0.27761108, -0.113762066, 0.050172582, 0.27963692, -0.16606395, -0.19824645, -0.024618482, -0.11107985, -0.2286693, 0.07190561, 0.2703224, -0.26687855, 0.007265952, 0.066743955, -0.114967555], left: 141, right: 142 })
        144: Descendants(Descendants { descendants: [0, 2, 6, 7, 9, 11, 14, 25, 29, 33, 34, 36, 40, 44, 48, 50, 56, 63, 77, 81, 86, 89, 92, 98] })
        145: Descendants(Descendants { descendants: [1, 5, 10, 13, 19, 26, 30, 31, 39, 41, 42, 46, 51, 62, 66, 70, 72, 85, 91] })
        146: Descendants(Descendants { descendants: [3, 16, 21, 27, 28, 45, 54, 67, 80, 90, 93, 96, 97] })
        147: SplitPlaneNormal(SplitPlaneNormal { normal: [0.16650625, 0.19149053, 0.3135367, 0.06213022, 0.10943303, -0.20039327, 0.058175124, 0.29041374, -0.020198382, 0.06427184, 0.048072472, -0.2131601, -0.074241415, 0.042433996, 0.12720534, -0.08725447, -0.056408316, -0.18833943, 0.086455815, -0.044081382, -0.03063445, 0.030401165, -0.24449702, 0.44077677, -0.2519855, -0.17355765, -0.10227632, -0.3992569, -0.06917604, 0.19473612], left: 145, right: 146 })
        148: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.18260413, 0.06320929, -0.055755503, -0.21661519, 0.10735892, -0.09589254, 0.10757786, -0.20054871, 0.034027994, 0.06764255, 0.13642153, -0.029665237, -0.19284418, 0.20193517, -0.1385175, 0.46306923, 0.11512843, 0.14657307, 0.056569442, -0.09560224, 0.06451906, 0.12966657, 0.1639076, 0.047254913, -0.35267997, 0.09693259, -0.10757247, -0.090299316, 0.33578426, -0.37402594], left: 144, right: 147 })
        149: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.0642175, 0.100616895, 0.21108045, 0.21055491, 0.04798969, 0.1748321, -0.36787733, -0.1483772, -0.17960916, -0.102965966, -0.06196047, -0.38384473, -0.15657893, 0.14972475, -0.36109754, 0.109144755, 0.22337553, -0.30172336, -0.05349713, 0.29302213, -0.020268407, -0.032836825, 0.08316328, -0.059157453, 0.13349834, 0.14169748, 0.022270858, 0.14461756, 0.14294842, 0.045416445], left: 143, right: 148 })
        150: Descendants(Descendants { descendants: [1, 3, 14, 15, 16, 21, 25, 27, 37, 40, 41, 42, 62, 77, 90, 91, 96, 98] })
        151: Descendants(Descendants { descendants: [2, 8, 12, 13, 18, 19, 28, 30, 36, 39, 45, 46, 51, 52, 54, 56, 66, 67, 68, 72, 73, 78, 80, 85, 87, 89, 92, 93] })
        152: SplitPlaneNormal(SplitPlaneNormal { normal: [0.12879752, -0.03803886, -0.073910356, 0.122484855, 0.2766287, -0.20582354, 0.097139, -0.21201412, -0.42800406, -0.058100864, -0.28332734, -0.15469314, 0.15667287, 0.018061234, 0.27831256, 0.026416773, 0.12758882, 0.17699721, -0.06847727, -0.055192467, 0.22137809, -0.055586252, 0.3153764, -0.0068500787, -0.276034, -0.09205558, -0.09806346, 0.07311849, 0.18933807, 0.22803518], left: 150, right: 151 })
        153: Descendants(Descendants { descendants: [0, 4, 7, 9, 10, 11, 24, 26, 29, 33, 34, 38, 44, 48, 50, 57, 61, 63, 65, 70, 76, 83, 86, 97] })
        154: Descendants(Descendants { descendants: [5, 6, 17, 20, 22, 23, 31, 32, 35, 43, 47, 49, 53, 55, 58, 59, 60, 64, 69, 71, 74, 75, 79, 81, 82, 84, 88, 94, 95, 99] })
        155: SplitPlaneNormal(SplitPlaneNormal { normal: [0.017928407, -0.016501768, 0.11772263, -0.14731649, -0.31905007, -0.26658154, 0.082067646, -0.015782926, 0.31785077, -0.12597542, 0.38731834, 0.23332672, -0.07515248, 0.066881336, 0.21481149, -0.1360667, -0.03098562, 0.32398272, 0.14981648, -0.19419862, 0.016375758, -0.009273337, 0.26013294, 0.023178319, 0.02844188, -0.34974656, -0.09547516, -0.024495449, -0.1048415, -0.09687797], left: 153, right: 154 })
        156: SplitPlaneNormal(SplitPlaneNormal { normal: [0.040711954, -0.08914529, -0.1805117, 0.06410551, 0.14557914, 0.08505917, 0.23401214, 0.007859579, -0.13221383, 0.29239056, 0.039778166, 0.20651713, 0.17177214, -0.11956052, 0.425181, -0.32198694, -0.21845435, 0.08067835, 0.21261747, -0.0014900899, 0.012805599, -0.09140584, -0.219866, -0.24325836, 0.24150439, -0.07735292, 0.0029572074, -0.31386444, -0.12031868, -0.047034983], left: 152, right: 155 })
        157: Descendants(Descendants { descendants: [2, 3, 4, 6, 7, 11, 15, 23, 27, 28, 47, 54, 55, 57, 62, 63, 66, 67, 73, 86, 87, 94, 95, 96, 97, 99] })
        158: Descendants(Descendants { descendants: [14, 17, 18, 19, 20, 26, 40, 42, 49, 60, 68, 69, 72, 75, 77, 79, 81, 84, 88, 91] })
        159: Descendants(Descendants { descendants: [0, 8, 9, 12, 25, 35, 37, 52, 53, 64, 71, 74, 98] })
        160: SplitPlaneNormal(SplitPlaneNormal { normal: [0.24873787, -0.15348019, -0.25156528, -0.19840913, -0.001288626, -0.32185066, 0.14885728, 0.13258436, -0.27334258, -0.07619485, -0.028077316, -0.042886484, 0.094783776, -0.33924598, -0.08856041, 0.027427612, -0.0037186889, -0.21609825, 0.23013714, 0.07689291, 0.3911563, 0.0067356406, 0.22926123, -0.0064194314, -0.068800054, -0.19760348, 0.12588142, 0.1637023, 0.07486653, 0.22250946], left: 158, right: 159 })
        161: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.19110389, -0.14020018, -0.27589774, 0.15650764, -0.074112944, 0.10051508, -0.043884125, -0.16025354, 0.47288913, -0.23930365, -0.0010959144, 0.26004055, 0.013729734, -0.016586266, -0.12559721, 0.040822547, 0.05280213, 0.16323669, 0.23179184, -0.084966786, -0.2265896, -0.03554115, 0.37635878, -0.17177078, 0.16676624, -0.13028221, -0.03562441, 0.21590918, -0.0667475, -0.14941435], left: 157, right: 160 })
        162: Descendants(Descendants { descendants: [1, 10, 13, 29, 31, 32, 33, 34, 36, 44, 45, 48, 50, 70, 89, 90, 92, 93] })
        163: Descendants(Descendants { descendants: [5, 16, 21, 22, 24, 30, 38, 39, 41, 43, 46, 51, 56, 58, 59, 61, 65, 76, 78, 80, 82, 83, 85] })
        164: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.07888287, 0.20454118, 0.017643586, -0.36178023, 0.13447018, 0.18570124, 0.105694324, 0.055130225, 0.26246232, 0.20461608, 0.088159636, -0.020598607, -0.10875567, -0.06031052, 0.21960914, 0.1559918, -0.22492622, 0.30719995, 0.060890317, -0.36892045, 0.16200861, -0.14118609, 0.099450916, 0.11916227, -0.38983616, -0.009986793, -0.13290069, -0.14889522, 0.0021677325, -0.0059990613], left: 162, right: 163 })
        165: SplitPlaneNormal(SplitPlaneNormal { normal: [0.10190071, -0.17172524, -0.11427366, 0.111685805, -0.045642603, 0.32032213, -0.2150584, -0.068413995, -0.2068874, -0.2366226, -0.28964862, -0.053019058, -0.011843923, 0.05915935, -0.014828069, -2.0931711e-5, -0.11198175, -0.14662306, 0.24134658, -0.035214365, 0.14111027, -0.25094754, 0.36904633, 0.1756086, 0.072693974, 0.38884625, -0.26793262, 0.07328632, -0.0034730998, -0.11185236], left: 161, right: 164 })
        166: Descendants(Descendants { descendants: [0, 2, 4, 6, 7, 9, 11, 14, 28, 29, 42, 44, 63, 74, 81, 86, 96, 97, 98, 99] })
        167: Descendants(Descendants { descendants: [1, 5, 10, 13, 19, 25, 26, 27, 30, 31, 32, 33, 34, 36, 39, 40, 48, 50, 62, 65, 70, 84, 85, 89, 91, 92, 93] })
        168: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.2502523, -0.122984335, -0.1472623, -0.051168714, -0.23660395, 0.1640492, 0.092374586, -0.33055833, -0.11776838, -0.028082361, -0.07169091, 0.1789267, -0.05319624, 0.009076259, -0.12235238, 0.07716192, 0.096866295, -0.0019910936, 0.08133792, -0.11651183, 0.033711, 0.060958967, 0.4296489, 0.22519411, 0.026167046, 0.3692936, 0.0045700674, 0.10633795, 0.15766934, -0.42262092], left: 166, right: 167 })
        169: Descendants(Descendants { descendants: [3, 8, 16, 21, 22, 23, 24, 37, 43, 45, 46, 47, 52, 54, 55, 64, 66, 67, 68, 69, 71, 72, 73, 77, 78, 80, 87, 90, 95] })
        170: Descendants(Descendants { descendants: [12, 15, 17, 18, 20, 35, 38, 41, 49, 51, 53, 56, 57, 58, 59, 60, 61, 75, 76, 79, 82, 83, 88, 94] })
        171: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.2220714, -0.44854346, 0.036275726, 0.0255127, -0.0069580204, 0.19252448, 0.14070556, -0.018338572, 0.02597122, 0.023465466, -0.12835452, 0.45027652, 0.28853863, -0.12607528, 0.21642616, -0.044019, -0.0671061, 0.12988785, 0.15407647, -0.3413349, -0.0389065, -0.13625199, -0.06538583, -0.20873953, 0.21120332, 0.106182635, -0.10197272, -0.012247764, -0.15733947, 0.03415892], left: 169, right: 170 })
        172: SplitPlaneNormal(SplitPlaneNormal { normal: [0.028435115, 0.15386719, -0.11415848, -0.1559428, 0.049400114, -0.15226397, 0.15010566, 0.11809572, 0.18941736, 0.12587875, 0.30251285, 0.069149904, -0.1728105, -0.059560023, 0.26466438, -0.09032245, -0.18104646, 0.23139672, 0.1682387, -0.30709746, 0.005550602, -0.08355586, 0.22693199, 0.24149981, -0.488041, -0.026979148, -0.08059744, -0.15866515, -0.14127068, 0.010886688], left: 168, right: 171 })
        173: Descendants(Descendants { descendants: [10, 11, 13, 19, 24, 26, 28, 29, 30, 33, 34, 36, 42, 44, 45, 46, 50, 51, 52, 56, 63, 64, 72, 76, 86, 89, 93, 97] })
        174: Descendants(Descendants { descendants: [0, 1, 16, 25, 31, 32, 35, 40, 41, 48, 49, 60, 65, 70, 85, 90, 91, 92, 98] })
        175: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.00853715, -0.25198168, -0.042705003, -0.15998664, -0.44207978, 0.029163469, 0.2800755, -0.0807462, 0.35243085, -0.15530515, 0.27395794, 0.27044085, 0.045883097, -0.098055094, -0.37794983, 0.1296113, -0.13526912, -0.060749177, 0.24635799, -0.07984116, -0.07452147, -0.015919844, -0.06970881, 0.056868616, 0.19047818, -0.03767905, 0.06190846, -0.1112239, -0.054655172, -0.005871846], left: 173, right: 174 })
        176: Descendants(Descendants { descendants: [2, 3, 4, 5, 6, 7, 9, 12, 14, 15, 17, 18, 27, 38, 39, 59, 62, 66, 67, 69, 74, 75, 80, 81, 84, 88, 94, 96, 99] })
        177: Descendants(Descendants { descendants: [8, 20, 21, 22, 23, 37, 43, 47, 53, 54, 55, 57, 58, 61, 68, 71, 73, 77, 78, 79, 82, 83, 87, 95] })
        178: SplitPlaneNormal(SplitPlaneNormal { normal: [0.24580881, -0.035068728, -0.49108055, -0.15485871, 0.004550411, -0.030795772, 0.100407355, -0.106373765, 0.27296832, -0.075598314, 0.05893523, -0.044649325, -0.07328395, -0.12034062, 0.26863864, -0.0654176, -0.055185746, -0.033357237, 0.02186325, -0.040906813, 0.019569982, 0.21569723, 0.15597622, 0.489017, -0.23309052, 0.03669589, -0.24700235, -0.14732295, -0.04211076, -0.10954111], left: 176, right: 177 })
        179: SplitPlaneNormal(SplitPlaneNormal { normal: [0.066175275, 0.18135375, 0.27109244, -0.22674069, -0.07872372, -0.15182653, 0.09111304, 0.13856085, 0.13552883, 0.20629387, 0.2128292, 0.14701003, 0.041535135, 0.12427932, 0.0122273415, -0.13799655, -0.28499222, 0.33582252, -0.04419542, -0.16844393, 0.111133575, 0.08805257, -0.34991947, 0.10205429, -0.17144446, -0.23618957, -0.1392775, -0.21064599, -0.18111064, 0.24051411], left: 175, right: 178 })

        u32::MAX: [108, 117, 126, 133, 140, 149, 156, 165, 172, 179]
        "###);
    }
}
