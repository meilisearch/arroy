use heed::EnvOpenOptions;

use super::DatabaseHandle;
use crate::distance::Euclidean;
use crate::upgrade::from_0_6_to_0_7;
use crate::Database;

#[test]
fn upgrade_v0_6_to_v0_7() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy("src/tests/assets/v0_6/data.mdb", dir.path().join("data.mdb")).unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()) }.unwrap();
    let mut rtxn = env.read_txn().unwrap();
    let database: Database<Euclidean> = env.open_database(&mut rtxn, None).unwrap().unwrap();

    let mut wtxn = env.write_txn().unwrap();

    /* The original database in v0.6 looks like this:
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [0], distance: "euclidean" }
    Tree 0: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Item(0), right: Tree(4), normal: [1.0000, 0.0000] })
    Tree 1: Descendants(Descendants { descendants: [1, 5] })
    Tree 2: Descendants(Descendants { descendants: [3, 4] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(2), right: Item(2), normal: [0.0000, 0.0000] }) // The normal should become None after the upgrade
    Tree 4: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(3), normal: [0.0000, 0.0000] }) // The normal should become None after the upgrade
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    */

    from_0_6_to_0_7(&mut rtxn, database, &mut wtxn, database).unwrap();
    wtxn.commit().unwrap();
    drop(rtxn);

    let handle = DatabaseHandle { env: env.clone(), database, tempdir: dir };
    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Item(0), right: Tree(4), normal: [1.0000, 0.0000] })
    Tree 1: Descendants(Descendants { descendants: [1, 5] })
    Tree 2: Descendants(Descendants { descendants: [3, 4] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(2), right: Item(2), normal: "none" })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(3), normal: "none" })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);
}
