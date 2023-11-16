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
        5: SplitPlaneNormal(SplitPlaneNormal { normal: [0.57735026, 0.57735026, 0.57735026], left: 0, right: 4 })

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
        100: Descendants(Descendants { descendants: [8, 10, 13, 16, 19, 35, 42, 43, 45, 51, 52, 53, 55, 64, 71, 72, 85, 90, 98] })
        101: Descendants(Descendants { descendants: [1, 2, 4, 6, 14, 21, 28, 37, 39, 50, 54, 62, 66, 67, 73, 96, 97] })
        102: SplitPlaneNormal(SplitPlaneNormal { normal: [0.07512467, 0.19600277, 0.30218643, 0.1358408, -0.06765547, 0.14609312, -0.0372204, 0.2714758, -0.19893101, 0.1393551, 0.04210694, 0.14259106, -0.18715931, 0.22170284, -0.26130807, 0.1470251, 0.015006348, -0.23208928, -0.12128476, -0.01372178, 0.23710534, 0.05887076, -0.3387812, -0.06531115, 0.14381309, -0.062506974, -0.3071456, -0.30082035, -0.080220915, 0.16254048], left: 100, right: 101 })
        103: Descendants(Descendants { descendants: [0, 11, 25, 29, 30, 34, 36, 40, 41, 44, 46, 56, 59, 61, 63, 74, 78, 84, 89, 91, 99] })
        104: SplitPlaneNormal(SplitPlaneNormal { normal: [0.2351645, -0.16763529, 0.0032193537, 0.15251963, -0.23367284, 0.254, -0.14019762, -0.13237454, -0.17220105, -0.18898742, -0.1517085, 0.10685223, 0.18287121, -0.17689519, 0.17754751, -0.27445936, 0.066891424, 0.20223533, 0.0051120576, -0.031727653, -0.09837184, -0.29704186, -0.10143661, -0.059491586, 0.26624924, 0.23713762, 0.01452003, 0.2625767, -0.318517, 0.09879091], left: 102, right: 103 })
        105: Descendants(Descendants { descendants: [12, 15, 18, 20, 22, 23, 24, 27, 77, 80, 82, 92] })
        106: SplitPlaneNormal(SplitPlaneNormal { normal: [0.017020548, -0.13664687, 0.24865846, -0.27404806, -0.038590383, -0.1727712, -0.096779786, 0.23450656, -0.16535814, 0.3384994, -0.050891604, 0.26691857, -0.13146721, -0.20495231, -0.16840728, -0.26956043, -0.20619334, 0.21813172, -0.063203774, -0.1146148, -0.16152546, 0.16085884, -0.13667649, 0.28168127, -0.045050178, 0.14505465, -0.030338116, 0.20585902, -0.13632523, 0.17366599], left: 104, right: 105 })
        107: Descendants(Descendants { descendants: [3, 5, 7, 9, 17, 26, 32, 33, 38, 47, 49, 57, 58, 60, 65, 68, 69, 75, 76, 79, 81, 83, 86, 87, 88, 93, 94, 95] })
        108: Descendants(Descendants { descendants: [31, 48, 70] })
        109: SplitPlaneNormal(SplitPlaneNormal { normal: [0.17374937, -0.2074397, -0.2366163, 0.026750235, -0.024439065, 0.16448697, -0.07386261, -0.18236269, -0.19260435, -0.19956972, -0.17504321, 0.02866819, -0.0199318, 0.041983683, -0.23952922, -0.06559596, 0.020130573, -0.25018507, 0.33427045, 0.37141722, -0.06410406, 0.025915662, -0.18296248, 0.33077487, 0.31504673, -0.04849188, 0.021577489, -0.11820342, -0.22910075, -0.0510319], left: 107, right: 108 })
        110: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.2978123, 0.14917336, -0.16992433, 0.28751895, -0.21072413, 0.16392203, 0.29887488, 0.1332026, -0.1774698, 0.31578198, -0.251642, 0.2480368, 0.014649618, 0.090864636, 0.0038475967, -0.069699004, -0.20857874, 0.20700143, 0.06201296, -0.03297842, -0.20889233, -0.24701324, 0.01958788, -0.12601674, 0.04643118, 0.15201731, -0.11314642, -0.24744803, -0.13204284, -0.028801732], left: 106, right: 109 })
        111: Descendants(Descendants { descendants: [6, 8, 11, 13, 19, 28, 37, 40, 45, 48, 56, 63, 77, 96] })
        112: Descendants(Descendants { descendants: [25, 30, 46, 51, 52, 62, 66, 72, 85, 91] })
        113: Descendants(Descendants { descendants: [14, 15, 16, 17, 18, 20, 24, 35, 41, 54, 69, 75, 82, 84, 92, 94, 98] })
        114: Descendants(Descendants { descendants: [0, 1, 3, 10, 21, 22, 36, 39, 42, 43, 50, 55, 61, 71, 73, 80, 81, 89, 90, 93] })
        115: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.08616266, 0.033412486, -0.164263, 0.08642972, 0.2890862, -0.007520519, -0.039247, 0.3267576, -0.26378027, 0.0049049957, -0.29907674, -0.25905177, -0.064000115, 0.01647497, 0.28144208, 0.027787197, -0.1393059, -0.28978747, 0.12694037, 0.33511278, 0.25880876, -0.013966175, 0.052289877, 0.11122343, 0.20159356, 0.018648764, -0.14316064, 0.105117954, -0.03378276, -0.25847235], left: 113, right: 114 })
        116: SplitPlaneNormal(SplitPlaneNormal { normal: [0.009079916, -0.22287004, -0.046115704, 0.21839239, 0.008868565, -0.31696707, 0.13397971, 0.25642464, 0.18026976, 0.08405914, 0.20240486, 0.26379353, -0.26745376, 0.21193798, 0.25998512, -0.28408122, -0.16353317, -0.12970923, 0.13924436, 0.16377705, -0.13627727, 0.07268423, -0.24407917, 0.035786588, 0.027352773, 0.039019745, 0.035453763, -0.23021826, -0.12485576, 0.23389305], left: 112, right: 115 })
        117: Descendants(Descendants { descendants: [26, 32, 38, 49, 57, 65, 68, 70, 76, 79, 83, 88, 95] })
        118: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.27237818, 0.14496854, -0.21366176, 0.30878115, -0.15805566, 0.1820099, 0.30054784, 0.12279179, -0.21265727, 0.303942, -0.31090719, 0.23709111, 0.0089869015, 0.10635952, -0.015571969, -0.07511403, -0.18018949, 0.17617936, 0.06772791, -0.022857614, -0.19393873, -0.24600413, -0.0119969, -0.13631943, 0.05404094, 0.112686716, -0.060763612, -0.2608047, -0.13969055, 0.014371483], left: 116, right: 117 })
        119: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.20157184, 0.053609703, 0.14279659, -0.25516254, -0.2645582, 0.22426972, 0.17378002, 0.08463165, 0.13757567, 0.13214216, 0.21560438, 0.097934745, -0.022120731, -0.26910487, 0.033941783, -0.27326304, -0.2138507, 0.27332002, 0.16856982, -0.014706516, 0.12492885, -0.1124529, 0.2585665, -0.27101004, 0.19879077, 0.21943599, -0.16480158, 0.14560415, -0.037496503, -0.13072692], left: 111, right: 118 })
        120: Descendants(Descendants { descendants: [2, 4, 5, 7, 9, 12, 23, 27, 29, 31, 33, 34, 44, 47, 53, 58, 59, 60, 64, 67, 74, 78, 86, 87, 97, 99] })
        121: SplitPlaneNormal(SplitPlaneNormal { normal: [0.038754288, -0.07314909, -0.021897526, -0.24806738, 0.22658992, 0.1946724, 0.02409565, 0.12649195, -0.28221837, 0.07055125, -0.099768616, -0.040067237, -0.11034741, 0.053359985, 0.059217848, -0.15166254, -0.019978272, 0.38337454, -0.20914552, -0.044461973, 0.16224925, -0.2111162, 0.048164748, -0.27814186, 0.09948333, -0.2680944, 0.31974638, -0.057342686, -0.3117292, 0.24382202], left: 119, right: 120 })
        122: Descendants(Descendants { descendants: [4, 8, 10, 12, 13, 22, 31, 32, 33, 35, 36, 39, 43, 44, 45, 50, 51, 56, 57, 61, 64, 68, 70, 76, 80, 82, 87, 92, 93] })
        123: Descendants(Descendants { descendants: [0, 2, 5, 7, 9, 18, 20, 23, 24, 38, 48, 49, 53, 58, 63, 65, 67, 71, 74, 75, 83, 84, 88, 94, 97, 99] })
        124: SplitPlaneNormal(SplitPlaneNormal { normal: [0.047452178, 0.05660816, 0.14478551, -0.011553997, -0.14058505, 0.0063598566, 0.10824247, 0.11860974, 0.1598137, 0.20487851, 0.19628702, -0.07455351, 0.17309944, -0.21000703, -0.21933752, -0.19525412, -0.28957596, 0.098847486, -0.08357282, 0.20806158, -0.19278549, 0.15689662, -0.36888304, -0.07025165, 0.13208911, 0.018903926, 0.3529716, -0.37066787, -0.15482654, 0.0741927], left: 122, right: 123 })
        125: Descendants(Descendants { descendants: [3, 15, 28, 29, 47, 55, 59, 77, 79, 89, 95] })
        126: SplitPlaneNormal(SplitPlaneNormal { normal: [0.07747901, 0.15225424, -0.062926866, 0.009303651, -0.051261406, 0.16285722, -0.27703866, 0.24440615, 0.2879295, 0.13938846, 0.017345557, 0.09703846, -0.11877166, 0.108684614, -0.2940979, 0.018591447, -0.16951865, -0.31860572, -0.3579106, -0.26222986, -0.19182189, -0.106773935, 0.18642902, 0.262897, -0.0073928363, -0.008613328, 0.06589841, -0.17023048, 0.12665184, 0.21830714], left: 124, right: 125 })
        127: Descendants(Descendants { descendants: [1, 6, 17, 21, 25, 26, 27, 34, 37, 46, 54, 62, 66, 73, 78, 81, 85, 86, 90, 96] })
        128: Descendants(Descendants { descendants: [11, 14, 16, 19, 30, 40, 41, 42, 52, 60, 69, 72, 91, 98] })
        129: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.23827817, 0.039545957, -0.151788, 0.1874096, 0.34964973, 0.09983087, -0.23132268, -0.13189809, 0.1538636, -0.114739485, -0.2988155, 0.16393514, -0.09491986, 0.17827144, -0.1612445, -0.03885938, 0.012398998, 0.23331435, 0.25895903, 0.123083785, -0.23443566, 0.17893146, 0.23435864, -0.20394728, -0.21310434, -0.18621472, 0.19652256, -0.08075367, -0.012971398, -0.11563114], left: 127, right: 128 })
        130: SplitPlaneNormal(SplitPlaneNormal { normal: [0.14854711, 0.22860974, -0.07622298, -0.04811397, -0.21838346, 0.12429746, -0.26221788, -0.17528069, 0.25290996, -0.2460538, 0.2572168, -0.09773754, -0.23075935, -0.05927029, -0.22399013, 0.18284199, 0.24384497, 0.12722042, -0.22570312, 0.18032914, -0.17356178, 0.13615556, -0.24740273, -0.11312684, 0.13934772, 0.19432063, 0.13986954, -0.059161615, 0.020991879, -0.22777791], left: 126, right: 129 })
        131: Descendants(Descendants { descendants: [0, 9, 15, 19, 20, 42, 49, 58, 67, 69, 71, 75, 81, 82, 84, 86, 87, 88, 98, 99] })
        132: Descendants(Descendants { descendants: [5, 12, 13, 16, 24, 25, 33, 34, 40, 44, 47, 48, 50, 78, 80, 92] })
        133: SplitPlaneNormal(SplitPlaneNormal { normal: [0.05045362, 0.13465884, 0.051753167, -0.22256595, -0.19589958, 0.17735893, -0.20359863, -0.043273874, -0.4195213, 0.009078773, -0.0655253, -0.2548648, -0.020661347, 0.25042954, -0.065845504, 0.068045616, 0.060106914, -0.3681754, -0.08489071, 0.09531681, 0.23613034, -0.13257512, 0.14700377, 0.23729563, 0.05341431, 0.2770314, 0.28981146, -0.129207, -0.07230826, 0.065483116], left: 131, right: 132 })
        134: Descendants(Descendants { descendants: [3, 6, 8, 14, 17, 21, 22, 23, 35, 37, 41, 43, 52, 53, 54, 55, 60, 61, 64, 68, 73, 74, 77, 90, 91, 95] })
        135: SplitPlaneNormal(SplitPlaneNormal { normal: [0.09186934, -0.12703975, -0.2252305, -0.17153096, -0.07355476, -0.21699151, 0.35265493, 0.2630534, 0.114089884, 0.0015537725, 0.18412569, 0.14767201, -0.14333858, 0.08882348, -0.14624925, 0.2915095, 0.22422187, -0.21869026, -0.2286786, 0.018034607, 0.14271836, 0.22011063, -0.026747813, 0.23563698, -0.037646614, -0.09860609, -0.2878852, -0.20478316, 0.010920738, -0.18392794], left: 133, right: 134 })
        136: Descendants(Descendants { descendants: [4, 7, 31, 32, 36, 38, 51, 56, 59, 62, 76, 89, 93, 96] })
        137: Descendants(Descendants { descendants: [1, 2, 10, 11, 18, 26, 27, 28, 29, 30, 39, 45, 46, 57, 63, 65, 66, 70, 72, 79, 83, 85, 94, 97] })
        138: SplitPlaneNormal(SplitPlaneNormal { normal: [0.016165696, 0.014578785, -0.29359114, -0.11199103, 0.38107425, 0.08303776, -0.12391554, -0.16702424, 0.23195735, 0.28698927, 0.12584835, -0.17416814, 0.08014528, 0.12581654, -0.11915606, -0.2417498, -0.06500719, 0.20996104, 0.17617878, -0.14133063, -0.09232797, -0.00026623608, -0.3037671, -0.20328346, -0.07943627, -0.08045709, -0.14277832, 0.22664638, 0.24708739, 0.20286429], left: 136, right: 137 })
        139: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.038118232, -0.18535481, 0.1959418, 0.08826405, 0.23524666, 0.19772182, 0.121950954, -0.14070152, -0.14548893, -0.12852849, -0.30070242, -0.17723657, 0.20533475, 0.26598668, -0.23571838, 0.25161576, -0.01818603, 0.11656165, -0.24589254, -0.114202134, -0.118261896, 0.031496443, -0.1725158, -0.070035785, -0.05086292, 0.32278886, -0.25999272, -0.13932124, 0.2203908, 0.13480008], left: 135, right: 138 })
        140: Descendants(Descendants { descendants: [1, 2, 3, 4, 7, 9, 10, 11, 12, 23, 29, 36, 39, 42, 44, 46, 50, 51, 55, 62, 66, 69, 72, 80, 81, 86, 94, 97, 99] })
        141: Descendants(Descendants { descendants: [6, 13, 27, 28, 45, 54, 96] })
        142: SplitPlaneNormal(SplitPlaneNormal { normal: [0.18633184, -0.07616294, -0.20345284, 0.24905209, 0.19958362, -0.21601209, -0.13100593, -0.12997484, -0.028687386, -0.14925154, -0.18486242, -0.0036261308, 0.058437344, 0.23379844, -0.023118118, 0.28227118, 0.19407402, -0.29444352, -0.09820317, -0.0701076, -0.17722899, 0.15586302, -0.20842032, 0.3615964, -0.23019543, -0.25506136, 0.13819805, -0.16331913, -0.0053956313, 0.08527704], left: 140, right: 141 })
        143: Descendants(Descendants { descendants: [26, 31, 33, 38, 47, 57, 58, 65, 68, 70, 95] })
        144: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.28438136, 0.114256166, -0.1944772, 0.2702586, -0.25035194, 0.18248953, 0.29032463, 0.12114966, -0.1309323, 0.26298591, -0.2564117, 0.31349733, 0.021200716, 0.04608762, 0.011885774, -0.10108522, -0.21906826, 0.19125354, 0.112180695, -0.09758495, -0.20912051, -0.24212435, 0.07996029, -0.07434901, 0.036770593, 0.14003766, -0.0897412, -0.23819593, -0.16677201, -0.04982607], left: 142, right: 143 })
        145: Descendants(Descendants { descendants: [5, 17, 19, 24, 25, 40, 49, 60, 61, 64, 67, 71, 73, 84, 88, 98] })
        146: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.19418193, -0.2657932, -0.25853494, 0.026822418, -0.1978374, 0.28967768, 0.30820748, -0.18031043, 0.27307263, -0.14681008, 0.079177134, 0.14115956, -0.24989071, -0.019359324, -0.03514404, -0.18978901, 0.1452782, 0.16095576, 0.00044193017, -0.26500845, -0.03126643, 0.12663086, 0.13951364, 0.14938854, 0.095209114, -0.28223723, 0.21008869, -0.1946499, 0.05738086, -0.03258677], left: 144, right: 145 })
        147: Descendants(Descendants { descendants: [0, 8, 14, 15, 18, 20, 21, 22, 32, 34, 35, 37, 41, 43, 48, 53, 59, 63, 74, 75, 76, 77, 79, 82, 83, 87, 89, 90, 92, 93] })
        148: Descendants(Descendants { descendants: [16, 30, 52, 56, 78, 85, 91] })
        149: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.08773824, 0.25581914, 0.082716465, -0.22445627, 0.04278438, 0.29809254, -0.16355422, -0.25095022, -0.17740946, -0.02680602, -0.14471799, -0.30831257, 0.2452354, -0.17891939, -0.22216423, 0.29828, 0.16228424, 0.117015906, -0.10575321, -0.12664022, 0.14204077, -0.05446369, 0.20119031, -0.119357504, -0.045411784, -0.01862851, -0.033166356, 0.24365757, 0.18195447, -0.26352516], left: 147, right: 148 })
        150: SplitPlaneNormal(SplitPlaneNormal { normal: [0.30622315, 0.100945964, -0.13788669, 0.13827197, -0.29385835, -0.0033370205, -0.046480738, -0.06308946, 0.2387045, -0.2971582, -0.1446312, 0.22472551, 0.22734858, -0.15566528, -0.26574245, 0.07281537, -0.21353607, 0.232632, -0.061111744, -0.20117609, 0.055164874, 0.018907728, 0.18297762, 0.29820114, -0.14793766, -0.033491135, -0.13394944, 0.11497521, -0.26827338, -0.03148576], left: 146, right: 149 })
        151: Descendants(Descendants { descendants: [2, 4, 7, 11, 15, 23, 36, 38, 53, 58, 59, 67, 70, 74, 86, 87, 93, 96, 99] })
        152: Descendants(Descendants { descendants: [0, 3, 5, 6, 9, 14, 17, 18, 27, 28, 29, 31, 33, 39, 42, 52, 55, 62, 80, 81, 84, 89, 97] })
        153: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.16379543, -0.06276204, -0.005408549, 0.22107954, 0.2679068, -0.057813484, -0.28257698, -0.23605163, 0.23740928, -0.17467123, 0.06736259, -0.12871064, -0.14886813, 0.1626736, 0.046967443, 0.21713087, -0.30124572, -0.148557, 0.13490392, 0.24188665, -0.26423967, 0.2002983, 0.20933527, -0.22460698, 0.18994719, -0.03279104, 0.039122447, 0.24177766, -0.053290498, -0.09983266], left: 151, right: 152 })
        154: Descendants(Descendants { descendants: [1, 20, 32, 34, 41, 43, 44, 47, 48, 49, 50, 51, 56, 61, 65, 75, 76, 83, 88, 94] })
        155: Descendants(Descendants { descendants: [10, 13, 19, 25, 26, 35, 57, 63, 92] })
        156: Descendants(Descendants { descendants: [8, 12, 16, 21, 22, 24, 30, 37, 40, 45, 46, 54, 60, 64, 66, 68, 69, 71, 72, 73, 77, 78, 79, 82, 85, 90, 91, 95, 98] })
        157: SplitPlaneNormal(SplitPlaneNormal { normal: [0.3115682, 0.28927645, 0.2523173, 0.05425417, -0.15380275, 0.20269677, -0.016378755, 0.070204906, 0.23779537, -0.10872922, 0.23373792, -0.14801607, -0.13876896, 0.037792403, 0.16538362, 0.083147, -0.2271411, 0.21089928, 0.0029315082, -0.26753104, -0.21603179, -0.18004946, -0.0013675551, 0.29242358, -0.13768663, -0.2062605, -0.122831725, -0.14715792, -0.17737056, 0.14449558], left: 155, right: 156 })
        158: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.015219687, -0.0019976492, 0.115951635, -0.17132875, -0.12245842, -0.2809623, -0.31395188, -0.1381453, 0.13871874, 0.2398817, 0.22069253, -0.21929854, -0.190981, 0.07552084, -0.23581828, 0.05965723, 0.23915319, 0.17554599, -0.16757233, -0.003963544, -0.0840577, 0.20441361, 0.18762816, 0.30781606, 0.22337954, -0.22507516, 0.07816997, -0.08990541, 0.08762784, 0.23045357], left: 154, right: 157 })
        159: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.09668621, -0.09571796, -0.2655864, 0.045932144, 0.021625938, 0.18263707, -0.01501707, -0.14925094, 0.2838182, -0.081388794, 0.28683776, -0.24848302, 0.08366109, 0.078468196, -0.12840554, -0.17889087, 0.18657413, -0.0010680249, 0.18294568, -0.2514951, -0.0040860535, 0.11569139, 0.28087255, 0.27778414, -0.138305, 0.20550853, -0.042032223, 0.07859159, -0.11139619, -0.42068604], left: 153, right: 158 })
        160: Descendants(Descendants { descendants: [8, 12, 13, 22, 34, 35, 44, 45, 48, 51, 52, 58, 60, 61, 68, 71, 75, 76, 80, 82, 83, 92, 98] })
        161: Descendants(Descendants { descendants: [16, 23, 24, 41, 43, 46, 59, 64, 78] })
        162: SplitPlaneNormal(SplitPlaneNormal { normal: [0.13905475, 0.23415704, -0.122892424, -0.30303857, -0.1089201, 0.10190301, 0.06292753, -0.06630676, 0.23696962, -0.20000139, -0.022146469, -0.30348113, -0.24750462, -0.081734516, 0.02116819, 0.3087747, -0.24119404, 0.20078039, -0.031812593, 0.024293413, 0.17746526, -0.22400925, -0.16034837, 0.21984017, -0.2559303, 0.19016914, 0.2067811, -0.025204135, -0.18345737, 0.0360538], left: 160, right: 161 })
        163: Descendants(Descendants { descendants: [5, 6, 7, 9, 14, 19, 21, 31, 33, 42, 62, 66, 67, 74, 86, 91, 93] })
        164: Descendants(Descendants { descendants: [1, 2, 3, 15, 18, 26, 37, 38, 39, 47, 50, 54, 57, 65, 69, 72, 73, 79, 85, 87, 89, 90, 94, 95, 96, 97] })
        165: Descendants(Descendants { descendants: [0, 4, 10, 11, 20, 29, 30, 32, 36, 53, 55, 63, 70, 84, 99] })
        166: SplitPlaneNormal(SplitPlaneNormal { normal: [0.33167005, -0.16893864, -0.33315122, 0.036242615, -0.0326411, 0.10227442, 0.21544313, -0.24039292, -0.12666139, -0.24660012, -0.27485433, -0.030760687, 0.19547744, -0.1220246, 0.27265236, -0.079632714, 0.30512875, -0.058160767, 0.025702644, -0.15725802, -0.031029755, 0.18889722, 0.23054363, -0.2733389, 0.015234866, -0.1415607, 0.13020445, 0.015420088, 0.13192609, 0.070992135], left: 164, right: 165 })
        167: Descendants(Descendants { descendants: [17, 25, 27, 28, 40, 49, 56, 77, 81, 88] })
        168: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.20766939, -0.10970753, 0.2653233, -0.08886781, -0.18234791, 0.23403256, -0.23465951, -0.22243048, 0.22945118, -0.16242902, -0.12650165, 0.0010314796, -0.23802875, 0.1951399, -0.06943394, 0.07252531, 0.1728604, -0.2610344, 0.17088427, -0.03478795, -0.19177267, 0.020603105, -0.040997706, 0.18667907, 0.15318401, -0.2129217, 0.14773335, 0.21973448, -0.29233366, 0.21816865], left: 166, right: 167 })
        169: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.22177796, -0.15320264, -0.1969053, -0.03139439, 0.07391595, -0.111777976, 0.20589228, -0.07646896, 0.15424904, 0.27029252, -0.24587509, -0.17263246, 0.3554633, 0.31249732, 0.009477227, 0.051041324, -0.012116123, 0.031438272, 0.020387273, -0.23552006, 0.031008128, -0.07576056, -0.12292671, 0.3085485, -0.07381264, 0.19886869, -0.33125263, -0.017494086, 0.22296113, 0.16970628], left: 163, right: 168 })
        170: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.30400214, 0.188252, 0.28256142, 0.027029159, -0.20130241, -0.060291797, -0.14205988, -0.03620351, 0.029914524, 0.06380362, 0.09045495, 0.14286101, 0.26897863, 0.01866373, -0.24270363, 0.26903385, 0.07506793, 0.071448, -0.4040054, 0.20213911, 0.15066616, 0.15683414, -0.15786536, -0.15624462, 0.19651376, 0.09818253, 0.06753538, -0.14432889, 0.13390842, -0.29214942], left: 162, right: 169 })
        171: Descendants(Descendants { descendants: [0, 1, 4, 5, 6, 7, 8, 9, 14, 26, 28, 31, 36, 39, 40, 48, 54, 57, 62, 65, 66, 69, 70, 89, 90, 91, 93, 96, 99] })
        172: Descendants(Descendants { descendants: [10, 11, 12, 13, 15, 19, 20, 22, 25, 30, 32, 33, 34, 35, 42, 45, 51, 52, 53, 56, 59, 60, 63, 74, 75, 83, 84, 87, 88, 92] })
        173: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.08000439, -0.24016437, -0.21199584, -0.10424361, 0.003905079, -0.07120454, 0.03791336, -0.29365635, 0.109126315, -0.14152998, -0.09632567, -0.04024281, 0.2586265, -0.2653413, 0.23776904, -0.18124272, -0.05925753, 0.3253085, 0.12782276, -0.031658906, -0.27994817, -0.11111147, 0.30306232, 0.032973118, -0.024701731, 0.0886601, 0.29942212, 0.30443743, 0.07277702, -0.10681153], left: 171, right: 172 })
        174: Descendants(Descendants { descendants: [38, 49, 58, 68, 76, 79, 95] })
        175: Descendants(Descendants { descendants: [2, 16, 29, 41, 44, 46, 50, 71, 81, 98] })
        176: Descendants(Descendants { descendants: [3, 17, 18, 21, 23, 24, 27, 37, 43, 47, 55, 61, 64, 67, 72, 73, 77, 78, 80, 82, 85, 86, 94, 97] })
        177: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.087727495, 0.009702844, 0.06308284, -0.2717029, -0.20532838, -0.1931675, 0.22511299, -0.09963676, 0.23731165, 0.067836545, -0.069502555, 0.07120837, -0.21979475, 0.09406604, 0.29413944, 0.23954326, -0.20381849, 0.27736098, -0.12943143, 0.19914658, -0.19487625, -0.054376885, -0.065456375, 0.35654917, -0.17174238, 0.15188561, 0.023070866, -0.2650788, 0.12984817, -0.122480355], left: 175, right: 176 })
        178: SplitPlaneNormal(SplitPlaneNormal { normal: [0.28898108, -0.10890765, 0.14718525, -0.3555476, 0.21010998, -0.1304838, -0.2619405, -0.06358745, 0.22309433, -0.2771766, 0.3170322, -0.30664554, -0.053757805, -0.052663665, 0.055480912, 0.05318984, 0.17122732, -0.1830722, -0.09919934, 0.061733074, 0.23203634, 0.21026404, -0.05529353, 0.15386564, -0.070047095, -0.14994225, 0.08434543, 0.22077192, 0.07279335, 0.014194834], left: 174, right: 177 })
        179: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.010640355, 0.21099214, -0.263487, -0.19562171, -0.14897494, 0.26582825, 0.055530738, 0.3179208, 0.053357124, 0.21491961, 0.18731804, -0.28105342, -0.008714122, 0.14630932, 0.31172734, -0.037761472, -0.14933966, -0.12479238, -0.21280523, 0.14816938, 0.0401444, -0.15164319, 0.04425687, -0.039325036, -0.30904126, 0.17583486, 0.09461181, 0.098950274, -0.2983562, -0.007075098], left: 173, right: 178 })
        180: Descendants(Descendants { descendants: [3, 12, 13, 15, 16, 18, 21, 22, 23, 37, 40, 41, 43, 45, 50, 51, 54, 55, 59, 61, 64, 71, 78, 80, 81, 82, 93, 96] })
        181: Descendants(Descendants { descendants: [5, 26, 31, 38, 47, 49, 58, 68, 76, 79, 87, 88, 95] })
        182: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.28667116, 0.1326681, -0.20549229, 0.3270434, -0.17404889, 0.17767026, 0.24701081, 0.12744027, -0.17735028, 0.32877162, -0.31087804, 0.22469367, 0.08393165, 0.079274125, -0.04347192, -0.07618159, -0.17002766, 0.19724123, 0.020022431, -0.029019328, -0.23613313, -0.2240476, -0.0031697787, -0.17801692, 0.047794532, 0.17347695, -0.09741677, -0.23120005, -0.04674965, -0.0004111095], left: 180, right: 181 })
        183: Descendants(Descendants { descendants: [2, 4, 6, 7, 8, 10, 11, 14, 17, 25, 27, 57, 62, 67, 69, 70, 72, 74, 84, 86, 91, 97, 98, 99] })
        184: Descendants(Descendants { descendants: [1, 24, 28, 29, 33, 34, 36, 44, 46, 66, 89, 92] })
        185: SplitPlaneNormal(SplitPlaneNormal { normal: [0.11163518, -0.091684364, 0.21451074, 0.16163953, -0.01672313, -0.21266669, -0.2410006, 0.09720285, -0.11640142, -0.20520923, -0.0126737235, -0.024624502, 0.23180963, -0.019899482, -0.029534647, -0.20020178, -0.33934832, -0.30057472, -0.02621115, 0.025668044, 0.04806987, -0.04318429, 0.4827723, 0.14067277, -0.07445168, 0.26426566, -0.15630065, -0.12909606, 0.1320705, 0.21128531], left: 183, right: 184 })
        186: Descendants(Descendants { descendants: [0, 9, 19, 20, 30, 32, 35, 39, 42, 48, 52, 53, 56, 60, 63, 65, 73, 75, 77, 83, 85, 90, 94] })
        187: SplitPlaneNormal(SplitPlaneNormal { normal: [0.27801692, -0.28623977, -0.21869783, 0.0156020755, 0.014425076, 0.043122824, -0.19792765, 0.028452026, 0.21655455, 0.082219996, -0.14696294, 0.16551793, -0.12521836, -0.28517702, 0.01889106, 0.22256573, -0.16308567, 0.04190303, 0.27387938, -0.23535715, -0.058356192, 0.14612688, 0.25905842, 0.2128086, 0.17069347, 0.096501775, -0.2461433, -0.1829393, 0.05233679, -0.26447338], left: 185, right: 186 })
        188: SplitPlaneNormal(SplitPlaneNormal { normal: [0.09136044, -0.20095804, -0.04395628, 0.032766636, 0.040519398, 0.12617387, -0.11339119, 0.09111433, -0.0018338374, -0.107752524, -0.058151815, -0.18705884, 0.31028968, -0.2787389, -0.143532, 0.17139296, 0.2941596, 0.126218, -0.17769712, 0.14020826, -0.35156244, 0.14606436, -0.30209154, -0.18206203, 0.09535721, 0.09443327, 0.18523015, -0.15850836, 0.31774366, 0.18483673], left: 182, right: 187 })
        189: Descendants(Descendants { descendants: [1, 8, 10, 21, 30, 33, 44, 46, 50, 76, 89, 97] })
        190: Descendants(Descendants { descendants: [6, 13, 20, 27, 28, 36, 42, 45, 51, 52, 54, 56, 57, 63, 64, 66, 68, 72, 80, 90, 91, 93] })
        191: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.056643445, -0.009598799, 0.06416115, -0.11487001, -0.18850172, -0.2794085, 0.17507307, -0.23688273, 0.12199674, 0.008846225, -0.1885696, 0.05961105, -0.17505163, 0.0497728, 0.29167438, 0.28199455, -0.11473312, 0.23321094, -0.11355979, 0.25118113, -0.24063236, 0.023328466, 0.003137189, 0.34222984, -0.15609282, 0.20070219, -0.007608417, -0.2514093, 0.23297492, -0.1610175], left: 189, right: 190 })
        192: Descendants(Descendants { descendants: [16, 19, 22, 24, 25, 32, 35, 37, 40, 41, 48, 60, 61, 73, 83, 85, 87, 92] })
        193: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.21524209, -0.25281727, -0.20982465, 0.026099956, -0.20718324, 0.29952615, 0.30809334, -0.12745614, 0.28130475, -0.06832879, 0.11263453, 0.13417348, -0.2483901, 0.011091851, -0.006047227, -0.23838294, 0.1199709, 0.20070542, -0.013786022, -0.25023773, -0.038864702, 0.07109124, 0.08031567, 0.056563333, 0.16844945, -0.32344967, 0.23277709, -0.21130155, 0.049890444, 0.0069886995], left: 191, right: 192 })
        194: Descendants(Descendants { descendants: [0, 12, 15, 23, 34, 43, 47, 49, 53, 58, 59, 67, 71, 75, 78, 82, 86, 94, 98, 99] })
        195: Descendants(Descendants { descendants: [2, 3, 4, 5, 7, 9, 11, 14, 17, 18, 26, 29, 31, 38, 39, 55, 62, 65, 69, 70, 74, 77, 79, 81, 84, 88, 95, 96] })
        196: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.29658064, 0.22112957, 0.21590166, 0.093435995, -0.18239443, -0.05156224, -0.21838573, -0.121946946, 0.006590102, -0.004166745, 0.09002511, 0.10478942, 0.24134445, 0.0023022497, -0.26534277, 0.31488854, 0.12436819, -0.011401531, -0.3599604, 0.17142293, 0.12918179, 0.23578753, -0.04616434, -0.07559721, 0.12792516, 0.118260324, 0.07733487, -0.1297757, 0.17799312, -0.35044104], left: 194, right: 195 })
        197: SplitPlaneNormal(SplitPlaneNormal { normal: [0.20949899, -0.12729417, 0.29229268, -0.2236024, -0.21741313, 0.08036112, 0.22580796, 0.23834857, 0.17365675, 0.14337134, 0.1608216, 0.020262048, 0.25507376, 0.085016266, -0.15118209, -0.08273563, -0.14824979, 0.31538838, -0.114383996, 0.048942104, 0.008884673, -0.25266656, -0.14698087, -0.3156628, 0.016986044, -0.16778988, 0.07786804, -0.16921316, -0.26362154, 0.09613866], left: 193, right: 196 })

        u32::MAX: [110, 121, 130, 139, 150, 159, 170, 179, 188, 197]
        "###);
    }
}
