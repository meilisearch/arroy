use std::num::NonZeroUsize;

use heed::EnvOpenOptions;

use super::DatabaseHandle;
use crate::distance::Euclidean;
use crate::tests::reader::NnsRes;
use crate::upgrade::from_0_6_to_current;
use crate::{Database, Reader};

#[test]
fn simple_upgrade_v0_6_to_v0_7() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy("src/tests/assets/v0_6/smol.mdb", dir.path().join("data.mdb")).unwrap();
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
    Tree 0: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: 5, right: 4, normal: Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] } })
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

// Same test as above but with a larger database. See its original snapshot here: https://github.com/meilisearch/arroy/blob/f52bf0560f5ceef27946bf0522730649be46ccdd/src/tests/snapshots/arroy__tests__writer__write_and_update_lot_of_random_points-2.snap
#[test]
fn large_upgrade_v0_6_to_v0_7() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy("src/tests/assets/v0_6/large.mdb", dir.path().join("data.mdb")).unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()) }.unwrap();
    let rtxn = env.read_txn().unwrap();
    let database: Database<Euclidean> = env.open_database(&rtxn, None).unwrap().unwrap();

    // First step is to know if we can still read into the database before the upgrade
    let reader = Reader::open(&rtxn, 0, database).unwrap();
    // The version is wrong but that's ok because the nodes format didn't change between v0.4 and v0.6 included.
    // This bug has been fixed in v0.7.
    insta::assert_snapshot!(reader.version(), @"v0.4.0");
    insta::assert_snapshot!(reader.n_items(), @"100");
    insta::assert_snapshot!(reader.n_trees(), @"10");
    insta::assert_snapshot!(reader.dimensions(), @"30");
    insta::assert_snapshot!(format!("{:?}", reader.item_ids()), @"RoaringBitmap<100 values between 0 and 99>");
    insta::assert_snapshot!(format!("{:?}", reader.stats(&rtxn).unwrap()), @"Stats { leaf: 100, tree_stats: [TreeStats { depth: 4, dummy_normals: 0, split_nodes: 4, descendants: 5 }, TreeStats { depth: 5, dummy_normals: 0, split_nodes: 5, descendants: 6 }, TreeStats { depth: 4, dummy_normals: 0, split_nodes: 4, descendants: 5 }, TreeStats { depth: 7, dummy_normals: 0, split_nodes: 6, descendants: 7 }, TreeStats { depth: 5, dummy_normals: 0, split_nodes: 5, descendants: 6 }, TreeStats { depth: 5, dummy_normals: 0, split_nodes: 6, descendants: 7 }, TreeStats { depth: 4, dummy_normals: 0, split_nodes: 4, descendants: 5 }, TreeStats { depth: 6, dummy_normals: 0, split_nodes: 6, descendants: 7 }, TreeStats { depth: 5, dummy_normals: 0, split_nodes: 4, descendants: 5 }, TreeStats { depth: 6, dummy_normals: 0, split_nodes: 5, descendants: 6 }] }");
    insta::assert_snapshot!(format!("{:?}", reader.item_vector(&rtxn, 0).unwrap()), @"Some([0.59189945, 0.9953131, 0.7271174, 0.7734485, 0.5760655, 0.8882299, 0.84973, 0.08173108, 0.39887708, 0.33842397, 0.16736221, 0.13506532, 0.7610012, 0.50516164, 0.51428705, 0.7101963, 0.44652337, 0.7144127, 0.31324244, 0.43315363, 0.98117304, 0.21394211, 0.8465342, 0.27935255, 0.70608264, 0.44866508, 0.9707988, 0.6317311, 0.94693947, 0.17849642])");
    insta::assert_snapshot!(format!("{:?}", reader.item_vector(&rtxn, 25).unwrap()), @"Some([0.011625171, 0.53228873, 0.39399207, 0.13821805, 0.19865465, 0.7286784, 0.40262043, 0.14423728, 0.59565574, 0.03397578, 0.54211503, 0.80171144, 0.88514394, 0.5250775, 0.2614928, 0.4367664, 0.94518125, 0.05161941, 0.7546513, 0.5079431, 0.72314125, 0.47682863, 0.36076427, 0.3593862, 0.99203247, 0.5132183, 0.9997714, 0.8521869, 0.58587575, 0.5980581])");

    let nns = reader
        .nns(3)
        .search_k(NonZeroUsize::new(100).unwrap())
        .by_vector(&rtxn, &[0.0; 30])
        .unwrap();
    insta::assert_snapshot!(NnsRes(Some(nns)), @r"
    id(92): distance(2.4881108)
    id(24): distance(2.5068686)
    id(78): distance(2.5809734)
    ");

    let mut wtxn = env.write_txn().unwrap();
    from_0_6_to_current(&rtxn, database, &mut wtxn, database).unwrap();
    wtxn.commit().unwrap();
    drop(rtxn);

    let handle = DatabaseHandle { env: env.clone(), database, tempdir: dir };
    insta::assert_snapshot!(handle);
}
