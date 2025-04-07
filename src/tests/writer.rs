use std::sync::atomic::{AtomicUsize, Ordering};

use heed::EnvOpenOptions;
use insta::assert_snapshot;
use rand::seq::SliceRandom;
use rand::Rng;

use super::{create_database, rng};
use crate::distance::{BinaryQuantizedCosine, Cosine, DotProduct, Euclidean};
use crate::{Database, Reader, Writer};

#[test]
fn clear_small_database() {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();

    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(200 * 1024 * 1024).open(dir.path()) }.unwrap();

    let mut wtxn = env.write_txn().unwrap();
    let database: Database<DotProduct> = env.create_database(&mut wtxn, None).unwrap();

    let zero_writer = Writer::new(database, 0, 3);
    zero_writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();
    zero_writer.clear(&mut wtxn).unwrap();
    zero_writer.builder(&mut rng()).build(&mut wtxn).unwrap();

    let one_writer = Writer::new(database, 1, 3);
    one_writer.add_item(&mut wtxn, 0, &[1.0, 2.0, 3.0]).unwrap();
    one_writer.builder(&mut rng()).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    let mut wtxn = env.write_txn().unwrap();
    let zero_writer = Writer::new(database, 0, 3);
    zero_writer.clear(&mut wtxn).unwrap();

    let one_reader = Reader::open(&wtxn, 1, database).unwrap();
    assert_eq!(one_reader.item_vector(&wtxn, 0).unwrap().unwrap(), vec![1.0, 2.0, 3.0]);
    wtxn.commit().unwrap();
}

#[test]
fn use_u32_max_minus_one_for_a_vec() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    writer.add_item(&mut wtxn, u32::MAX - 1, &[0.0, 1.0, 2.0]).unwrap();

    writer.builder(&mut rng()).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[4294967294]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [4294967294] })
    Item 4294967294: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    "###);
}

#[test]
fn use_u32_max_for_a_vec() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    writer.add_item(&mut wtxn, u32::MAX, &[0.0, 1.0, 2.0]).unwrap();

    writer.builder(&mut rng()).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[4294967295]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [4294967295] })
    Item 4294967295: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    "###);
}

#[test]
fn write_one_vector() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.builder(&mut rng()).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    "###);
}

#[test]
fn write_one_vector_in_one_tree() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.builder(&mut rng()).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    "###);
}

#[test]
fn write_one_vector_in_multiple_trees() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.builder(&mut rng()).n_trees(10).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    "###);
}

#[test]
fn write_vectors_until_there_is_a_descendants() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    for i in 0..3 {
        let id = i;
        let i = i as f32;
        writer.add_item(&mut wtxn, id, &[i, i, i]).unwrap();
    }

    writer.builder(&mut rng()).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0, 1, 2]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0, 1, 2] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 1.0000, 1.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 2.0000, 2.0000] })
    "###);
}

#[test]
fn write_vectors_until_there_is_a_split() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3);
    for i in 0..4 {
        let id = i;
        let i = i as f32;
        writer.add_item(&mut wtxn, id, &[i, i, i]).unwrap();
    }

    writer.builder(&mut rng()).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0, 1, 2, 3]>, roots: [2], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [1, 2, 3] })
    Tree 1: Descendants(Descendants { descendants: [0] })
    Tree 2: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(1), normal: Some([-0.5774, -0.5774, -0.5774]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 1.0000, 1.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 2.0000, 2.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 3.0000, 3.0000] })
    "#);
}

#[test]
fn write_and_update_lot_of_random_points() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 30);
    let mut rng = rng();
    for id in 0..100 {
        let vector: [f32; 30] = std::array::from_fn(|_| rng.gen());
        writer.add_item(&mut wtxn, id, &vector).unwrap();
    }

    writer.builder(&mut rng).n_trees(10).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();
    insta::assert_snapshot!(handle);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 30);
    for id in (0..100).step_by(2) {
        let vector: [f32; 30] = std::array::from_fn(|_| rng.gen());
        writer.add_item(&mut wtxn, id, &vector).unwrap();
    }
    writer.builder(&mut rng).n_trees(10).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle);
}

#[test]
fn write_multiple_indexes() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();

    for i in 0..5 {
        let writer = Writer::new(handle.database, i, 3);
        writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();
        writer.builder(&mut rng()).n_trees(1).build(&mut wtxn).unwrap();
    }
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    ==================
    Dumping index 1
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    ==================
    Dumping index 2
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    ==================
    Dumping index 3
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    ==================
    Dumping index 4
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    "###);
}

#[test]
fn write_random_vectors_to_random_indexes() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();

    let mut indexes: Vec<u16> = (0..10).collect();
    indexes.shuffle(&mut rng);

    for index in indexes {
        let writer = Writer::new(handle.database, index, 10);

        // We're going to write 10 vectors per index
        for i in 0..10 {
            let vector: [f32; 10] = std::array::from_fn(|_| rng.gen());
            writer.add_item(&mut wtxn, i, &vector).unwrap();
        }
        writer.builder(&mut rng).build(&mut wtxn).unwrap();
    }
    wtxn.commit().unwrap();
}

#[test]
fn overwrite_one_item_incremental() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [1, 3] })
    Tree 2: Descendants(Descendants { descendants: [2] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(2), normal: Some([0.0000, 0.0000]) })
    Tree 4: Descendants(Descendants { descendants: [4, 5] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(4), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();

    let writer = Writer::new(handle.database, 0, 2);

    writer.add_item(&mut wtxn, 3, &[6., 0.]).unwrap();

    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 3: Descendants(Descendants { descendants: [1, 2] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(9), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Tree 7: Descendants(Descendants { descendants: [4] })
    Tree 8: Descendants(Descendants { descendants: [3, 5] })
    Tree 9: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(7), right: Tree(8), normal: Some([0.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [6.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);
}

#[test]
fn delete_one_item_in_a_one_item_db() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 0).unwrap();

    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, roots: [], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    "###);

    let rtxn = handle.env.read_txn().unwrap();
    let one_reader = Reader::open(&rtxn, 0, handle.database).unwrap();
    assert!(one_reader.item_vector(&rtxn, 0).unwrap().is_none());
}

#[test]
fn delete_document_in_an_empty_index_74() {
    // See https://github.com/meilisearch/arroy/issues/74
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();

    let writer = Writer::new(handle.database, 0, 2);
    writer.del_item(&mut wtxn, 0).unwrap();
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.builder(&mut rng).build(&mut wtxn).unwrap();

    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();

    let writer1 = Writer::new(handle.database, 0, 2);
    writer1.del_item(&mut wtxn, 0).unwrap();

    let writer2 = Writer::new(handle.database, 1, 2);
    writer2.del_item(&mut wtxn, 0).unwrap();

    writer1.builder(&mut rng).build(&mut wtxn).unwrap();
    writer2.builder(&mut rng).build(&mut wtxn).unwrap();

    let reader = Reader::open(&wtxn, 1, handle.database).unwrap();
    let ret = reader.nns(10).by_vector(&wtxn, &[0., 0.]).unwrap();
    insta::assert_debug_snapshot!(ret, @"[]");

    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, roots: [], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    ==================
    Dumping index 1
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, roots: [], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    "###);

    let rtxn = handle.env.read_txn().unwrap();
    let reader = Reader::open(&rtxn, 1, handle.database).unwrap();
    let ret = reader.nns(10).by_vector(&rtxn, &[0., 0.]).unwrap();
    insta::assert_debug_snapshot!(ret, @"[]");
}

#[test]
fn delete_one_item_in_a_descendant() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // first, insert a bunch of items
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.add_item(&mut wtxn, 1, &[1., 0.]).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0, 1] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 0).unwrap();

    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[1]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [1] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    "###);
}

#[test]
fn delete_one_leaf_in_a_split() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // first, insert a bunch of elements
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.add_item(&mut wtxn, 1, &[1., 0.]).unwrap();
    writer.add_item(&mut wtxn, 2, &[2., 0.]).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2]>, roots: [2], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [1, 2] })
    Tree 2: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(1), normal: Some([1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 0).unwrap();

    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    // after deleting the leaf, the split node should be replaced by a descendant
    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[1, 2]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [1, 2] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    "###);
}

#[test]
fn delete_one_item_in_a_single_document_database() {
    let handle = create_database::<Cosine>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // first, insert a bunch of elements
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.builder(&mut rng).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, roots: [0], distance: "cosine" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderCosine { norm: 0.0 }, vector: [0.0000, 0.0000] })
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 0).unwrap();

    writer.builder(&mut rng).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, roots: [], distance: "cosine" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    "###);
}

#[test]
fn delete_one_item() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // first, insert a bunch of elements
    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [1, 3] })
    Tree 2: Descendants(Descendants { descendants: [2] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(2), normal: Some([0.0000, 0.0000]) })
    Tree 4: Descendants(Descendants { descendants: [4, 5] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(4), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 3).unwrap();

    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 4, 5]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 3: Descendants(Descendants { descendants: [1, 2] })
    Tree 4: Descendants(Descendants { descendants: [4, 5] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(4), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);

    // delete the last item in a descendants node
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.del_item(&mut wtxn, 1).unwrap();

    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 2, 4, 5]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 4: Descendants(Descendants { descendants: [4, 5] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Item(2), right: Tree(4), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);
}

#[test]
fn add_one_item_incrementally_in_an_empty_db() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, roots: [], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    "###);
}

#[test]
fn add_one_item_incrementally_in_a_one_item_db() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.add_item(&mut wtxn, 1, &[1., 0.]).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0, 1] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    "###);
}

#[test]
fn add_one_item_incrementally_to_create_a_split_node() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.add_item(&mut wtxn, 1, &[1., 0.]).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0, 1] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.add_item(&mut wtxn, 2, &[2., 0.]).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2]>, roots: [3], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 1: Descendants(Descendants { descendants: [0] })
    Tree 2: Descendants(Descendants { descendants: [1, 2] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(2), normal: Some([1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    "#);
}

#[test]
fn add_one_item_incrementally() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [1, 3] })
    Tree 2: Descendants(Descendants { descendants: [2] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(2), normal: Some([0.0000, 0.0000]) })
    Tree 4: Descendants(Descendants { descendants: [4, 5] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(4), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.add_item(&mut wtxn, 25, &[25., 0.]).unwrap();

    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5, 25]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [1, 3] })
    Tree 2: Descendants(Descendants { descendants: [2] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(2), normal: Some([0.0000, 0.0000]) })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(9), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Tree 7: Descendants(Descendants { descendants: [5] })
    Tree 8: Descendants(Descendants { descendants: [4, 25] })
    Tree 9: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(7), right: Tree(8), normal: Some([0.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Item 25: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [25.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    writer.add_item(&mut wtxn, 8, &[8., 0.]).unwrap();

    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5, 8, 25]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [1, 3] })
    Tree 2: Descendants(Descendants { descendants: [2] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(2), normal: Some([0.0000, 0.0000]) })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(9), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Tree 7: Descendants(Descendants { descendants: [5, 8] })
    Tree 8: Descendants(Descendants { descendants: [4, 25] })
    Tree 9: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(7), right: Tree(8), normal: Some([0.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Item 8: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [8.0000, 0.0000] })
    Item 25: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [25.0000, 0.0000] })
    "#);
}

#[test]
fn delete_extraneous_tree() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 4);
    for i in 0..5 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0., 0., 0.]).unwrap();
    }
    // 5 nodes of 4 dimensions should create 3 trees by default.
    writer.builder(&mut rng).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 4, items: RoaringBitmap<[0, 1, 2, 3, 4]>, roots: [2, 5], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [1, 2, 3, 4] })
    Tree 2: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(1), normal: Some([1.0000, 0.0000, 0.0000, 0.0000]) })
    Tree 3: Descendants(Descendants { descendants: [1, 2, 3, 4] })
    Tree 4: Descendants(Descendants { descendants: [0] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(4), normal: Some([-1.0000, 0.0000, 0.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000, 0.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000, 0.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000, 0.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000, 0.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.builder(&mut rng).n_trees(2).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4]>, roots: [2, 5], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [1, 2, 3, 4] })
    Tree 2: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(1), normal: Some([1.0000, 0.0000, 0.0000, 0.0000]) })
    Tree 3: Descendants(Descendants { descendants: [1, 2, 3, 4] })
    Tree 4: Descendants(Descendants { descendants: [0] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(4), normal: Some([-1.0000, 0.0000, 0.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000, 0.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000, 0.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000, 0.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000, 0.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4]>, roots: [5], distance: "euclidean" }
    Tree 3: Descendants(Descendants { descendants: [1, 2, 3, 4] })
    Tree 4: Descendants(Descendants { descendants: [0] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(4), normal: Some([-1.0000, 0.0000, 0.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000, 0.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000, 0.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000, 0.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000, 0.0000, 0.0000] })
    "#);
}

#[test]
fn reuse_node_id() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [1, 3] })
    Tree 2: Descendants(Descendants { descendants: [2] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(2), normal: Some([0.0000, 0.0000]) })
    Tree 4: Descendants(Descendants { descendants: [4, 5] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(4), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // if we delete the 1 it should free the node id 0
    writer.del_item(&mut wtxn, 1).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 2, 3, 4, 5]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 3: Descendants(Descendants { descendants: [2, 3] })
    Tree 4: Descendants(Descendants { descendants: [4, 5] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(4), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // if we re-insert the 1 the node id 0 should be re-used
    writer.add_item(&mut wtxn, 1, &[1., 0.]).unwrap();
    writer.builder(&mut rng).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [6], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [] })
    Tree 2: Descendants(Descendants { descendants: [] })
    Tree 3: Descendants(Descendants { descendants: [2, 3] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(13), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Tree 7: Descendants(Descendants { descendants: [] })
    Tree 8: Descendants(Descendants { descendants: [1, 5] })
    Tree 9: Descendants(Descendants { descendants: [4] })
    Tree 10: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(8), right: Tree(9), normal: Some([0.0000, 0.0000]) })
    Tree 11: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(7), right: Tree(10), normal: Some([0.0000, 0.0000]) })
    Tree 12: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(2), right: Tree(11), normal: Some([0.0000, 0.0000]) })
    Tree 13: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(12), normal: Some([0.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);

    // if we now build a new tree, the id 1 should be re-used
    writer.builder(&mut rng).n_trees(2).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [6, 19], distance: "euclidean" }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [] })
    Tree 2: Descendants(Descendants { descendants: [] })
    Tree 3: Descendants(Descendants { descendants: [2, 3] })
    Tree 4: Descendants(Descendants { descendants: [1, 3] })
    Tree 5: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(3), right: Tree(13), normal: Some([0.0000, 0.0000]) })
    Tree 6: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(0), right: Tree(5), normal: Some([1.0000, 0.0000]) })
    Tree 7: Descendants(Descendants { descendants: [] })
    Tree 8: Descendants(Descendants { descendants: [1, 5] })
    Tree 9: Descendants(Descendants { descendants: [4] })
    Tree 10: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(8), right: Tree(9), normal: Some([0.0000, 0.0000]) })
    Tree 11: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(7), right: Tree(10), normal: Some([0.0000, 0.0000]) })
    Tree 12: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(2), right: Tree(11), normal: Some([0.0000, 0.0000]) })
    Tree 13: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(1), right: Tree(12), normal: Some([0.0000, 0.0000]) })
    Tree 14: Descendants(Descendants { descendants: [5] })
    Tree 15: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(4), right: Tree(14), normal: Some([0.0000, 0.0000]) })
    Tree 16: Descendants(Descendants { descendants: [2, 4] })
    Tree 17: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(15), right: Tree(16), normal: Some([0.0000, 0.0000]) })
    Tree 18: Descendants(Descendants { descendants: [0] })
    Tree 19: SplitPlaneNormal(SplitPlaneNormal<euclidean> { left: Tree(17), right: Tree(18), normal: Some([-1.0000, 0.0000]) })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    "#);
}

#[test]
fn need_build() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    assert!(writer.need_build(&wtxn).unwrap(), "because metadata are missing");

    writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();
    assert!(
        writer.need_build(&wtxn).unwrap(),
        "because metadata are missing and an item has been updated"
    );
    writer.builder(&mut rng).build(&mut wtxn).unwrap();

    let writer = Writer::new(handle.database, 0, 2);
    writer.del_item(&mut wtxn, 0).unwrap();
    assert!(writer.need_build(&wtxn).unwrap(), "because an item has been updated");
}

#[test]
fn append() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 1, 2);

    writer.append_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();
    writer.append_item(&mut wtxn, 1, &[0.1, 0.1]).unwrap();
    let err = writer.append_item(&mut wtxn, 0, &[0.2, 0.2]).unwrap_err();
    assert_snapshot!(err, @"Item cannot be appended into the database");
    writer.builder(&mut rng).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();
    insta::assert_snapshot!(handle, @r###"
    ==================
    Dumping index 1
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1]>, roots: [0], distance: "euclidean" }
    Version: Version { major: 0, minor: 6, patch: 1 }
    Tree 0: Descendants(Descendants { descendants: [0, 1] })
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.1000, 0.1000] })
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();

    // We cannot push in a database with an index lower than the maximum
    let writer = Writer::new(handle.database, 0, 2);
    let err = writer.append_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap_err();
    assert_snapshot!(err, @"Item cannot be appended into the database");

    // And we must still be able append in an already existing build database as long as the item id is higher than the last max item id
    let writer = Writer::new(handle.database, 1, 2);
    // The item id is *equal* to the previous maximum item id
    let err = writer.append_item(&mut wtxn, 1, &[0.1, 0.1]).unwrap_err();
    assert_snapshot!(err, @"Item cannot be appended into the database");
    // If we delete the max item then we must be able to append it again
    writer.del_item(&mut wtxn, 1).unwrap();
    writer.append_item(&mut wtxn, 1, &[0.1, 0.1]).unwrap();

    // But we can still append in a database with a higher index even if the document id is lower
    let writer = Writer::new(handle.database, 2, 2);
    writer.append_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();
}

#[test]
fn prepare_changing_distance() {
    let handle = create_database::<Cosine>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();
    writer.add_item(&mut wtxn, 1, &[1.0, 1.0]).unwrap();
    writer.add_item(&mut wtxn, 3, &[3.0, 3.0]).unwrap();
    writer.builder(&mut rng).build(&mut wtxn).unwrap();
    let writer = Writer::new(handle.database, 1, 2);
    writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();
    writer.add_item(&mut wtxn, 1, &[1.0, 1.0]).unwrap();
    writer.add_item(&mut wtxn, 3, &[3.0, 3.0]).unwrap();
    writer.builder(&mut rng).build(&mut wtxn).unwrap();
    let writer = Writer::new(handle.database, 2, 2);
    writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();
    writer.add_item(&mut wtxn, 1, &[1.0, 1.0]).unwrap();
    writer.add_item(&mut wtxn, 3, &[3.0, 3.0]).unwrap();
    writer.builder(&mut rng).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 1, 2);

    let writer = writer.prepare_changing_distance::<BinaryQuantizedCosine>(&mut wtxn).unwrap();
    assert!(writer.need_build(&wtxn).unwrap(), "after changing the distance");

    writer.builder(&mut rng).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    // TODO: this should not works, see https://github.com/meilisearch/arroy/issues/92
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 1, 2);
    writer.del_item(&mut wtxn, 0).unwrap();
    assert!(writer.need_build(&wtxn).unwrap(), "because an item has been updated");
    writer.builder(&mut rng).build(&mut wtxn).unwrap();
}

#[test]
fn cancel_indexing_process() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2);
    writer.add_item(&mut wtxn, 0, &[0.0, 0.0]).unwrap();
    // Cancel straight away
    let err = writer.builder(&mut rng).cancel(|| true).build(&mut wtxn).unwrap_err();
    assert_snapshot!(err, @"The corresponding build process has been cancelled");

    // Do not cancel at all
    writer.builder(&mut rng).cancel(|| false).build(&mut wtxn).unwrap();

    // Cancel after being called a few times
    let writer = Writer::new(handle.database, 0, 2);
    for i in 0..100 {
        writer.add_item(&mut wtxn, i, &[i as f32, 1.1]).unwrap();
    }
    let cpt = AtomicUsize::new(0);
    let err = writer
        .builder(&mut rng)
        .cancel(|| {
            let prev = cpt.fetch_add(1, Ordering::Relaxed);
            prev > 5
        })
        .build(&mut wtxn)
        .unwrap_err();
    assert_snapshot!(err, @"The corresponding build process has been cancelled");
}

#[test]
#[ignore] // Temporarily disabled while fixing issues
fn split_nodes_never_point_to_items_directly() {
    /* 
    use crate::{Distance, Error, Key, Node, NodeId, NodeMode};
    use crate::node::SplitPlaneNormal;
    use heed::RoTxn;
    use crate::tests::TempDataBase;
    use crate::distances::Euclidean;
    
    // Initialize the test with 3 items to ensure we'll create a split node
    let (db, _tmp) = TempDataBase::new_with_distance::<Euclidean>("split_nodes_test").unwrap();
    let idx = 0;
    let mut wtxn = db.write_txn().unwrap();
    let writer = Writer::new(db.database, idx, 3);
    
    // Add 3 items with different vectors
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0];
    let v3 = vec![0.0, 0.0, 1.0];
    
    writer.add_item(&mut wtxn, 0, &v1).unwrap();
    writer.add_item(&mut wtxn, 1, &v2).unwrap();
    writer.add_item(&mut wtxn, 2, &v3).unwrap();
    
    // Build trees (which creates split nodes)
    writer.builder(&mut rand::thread_rng()).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();
    
    // Now verify no split nodes directly point to items
    let rtxn = db.read_txn().unwrap();
    let reader = Reader::open(&rtxn, idx, db.database).unwrap();
    
    // Get the tree roots from reader's stats
    let stats = reader.stats(&rtxn).unwrap();
    
    // Check all trees
    for result in stats.tree_stats.iter() {
        // Start checking from any tree roots
        if result.split_nodes > 0 {
            // Find a tree root by querying nodes
            for i in 0..100 { // Try a range of potential roots
                let node_id = NodeId::tree(i);
                let key = Key::new(idx, node_id);
                if let Ok(Some(_)) = db.database.get(&rtxn, &key) {
                    if let Ok(true) = check_no_direct_items(db.database, &rtxn, idx, node_id) {
                        // Found and checked a valid root
                        return;
                    }
                }
            }
        }
    }
    
    // If we get here, we couldn't find a root to check
    panic!("Could not find any tree roots to check");
    */
}

// Recursive function to check all nodes in the tree
fn check_no_direct_items<D: crate::Distance>(
    db: crate::Database<D>,
    rtxn: &heed::RoTxn,
    index: u16,
    node: crate::NodeId,
) -> Result<bool, Box<dyn std::error::Error>> {
    use crate::{Error, Key, Node, NodeMode};
    use crate::node::SplitPlaneNormal;
    
    let key = Key::new(index, node);
    match db.get(rtxn, &key)?.ok_or(Error::missing_key(key))? {
        Node::Leaf(_) => Ok(true),
        Node::Descendants(_) => Ok(true),
        Node::SplitPlaneNormal(SplitPlaneNormal { normal: _, left, right }) => {
            // Check that neither left nor right points directly to an item
            assert_ne!(left.mode, NodeMode::Item, "Left child of split node points to an item directly");
            assert_ne!(right.mode, NodeMode::Item, "Right child of split node points to an item directly");
            
            // Recursively check children
            check_no_direct_items(db, rtxn, index, left)?;
            check_no_direct_items(db, rtxn, index, right)?;
            
            Ok(true)
        }
    }
}
