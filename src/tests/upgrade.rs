use std::num::NonZeroUsize;

use heed::EnvOpenOptions;

use super::DatabaseHandle;
use crate::distance::Euclidean;
use crate::tests::reader::NnsRes;
use crate::upgrade::from_0_6_to_current;
use crate::{Database, Reader};

#[test]
fn upgrade_v0_6_to_v0_7() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy("src/tests/assets/v0_6/data.mdb", dir.path().join("data.mdb")).unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()) }.unwrap();
    let rtxn = env.read_txn().unwrap();
    let database: Database<Euclidean> = env.open_database(&rtxn, None).unwrap().unwrap();

    /* The original database in v0.6 looks like this:
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [0], distance: "euclidean" }
    Tree 0: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Item(0), right: Tree(4), normal: [1.0000, 0.0000] }) // This tree node is pointing to the item 0 which is illegal, it must be rewritten
    Tree 1: Descendants(Descendants { descendants: [1, 5] })
    Tree 2: Descendants(Descendants { descendants: [3, 4] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(2), right: Item(2), normal: [0.0000, 0.0000] }) // The normal should become None after the upgrade and the Item(2) should be rewritten
    Tree 4: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(3), normal: [0.0000, 0.0000] }) // The normal should become None after the upgrade
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    */

    // First step is to know if we can still read into the database before the upgrade
    let reader = Reader::open(&rtxn, 0, database).unwrap();
    // The version is wrong but that's ok because the nodes format didn't change between v0.4 and v0.6 included.
    // This bug has been fixed in v0.7.
    insta::assert_snapshot!(reader.version(), @"v0.4.0");
    insta::assert_snapshot!(reader.n_items(), @r#"
    6
    "#);
    insta::assert_snapshot!(reader.n_trees(), @r#"
    1
    "#);
    insta::assert_snapshot!(reader.dimensions(), @r#"
    2
    "#);
    insta::assert_snapshot!(format!("{:?}", reader.item_ids()), @r#"
    RoaringBitmap<[0, 1, 2, 3, 4, 5]>
    "#);
    insta::assert_snapshot!(format!("{:?}", reader.stats(&rtxn).unwrap()), @"Stats { leaf: 6, tree_stats: [TreeStats { depth: 4, dummy_normals: 2, split_nodes: 3, descendants: 2 }] }");
    insta::assert_snapshot!(format!("{:?}", reader.item_vector(&rtxn, 0).unwrap()), @"Some([0.0, 0.0])");
    insta::assert_snapshot!(format!("{:?}", reader.item_vector(&rtxn, 25).unwrap()), @"None");

    let nns = reader
        .nns(3)
        .search_k(NonZeroUsize::new(100).unwrap())
        .by_vector(&rtxn, &[1.0, 0.0])
        .unwrap();
    insta::assert_snapshot!(NnsRes(Some(nns)), @r"
    id(1): distance(0)
    id(0): distance(1)
    id(2): distance(1)
    ");

    let mut wtxn = env.write_txn().unwrap();
    from_0_6_to_current(&rtxn, database, &mut wtxn, database).unwrap();
    wtxn.commit().unwrap();
    drop(rtxn);

    let handle = DatabaseHandle { env: env.clone(), database, tempdir: dir };
    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 7, patch: 0 }
    Tree 0: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: 5, right: 4, normal: [1.0000, 0.0000] })
    Tree 1: Descendants(Descendants { descendants: [1, 5] })
    Tree 2: Descendants(Descendants { descendants: [3, 4] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: 2, right: 6, normal: "none" })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: 1, right: 3, normal: "none" })
    Tree 5: Descendants(Descendants { descendants: [0] })
    Tree 6: Descendants(Descendants { descendants: [2] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);
}
