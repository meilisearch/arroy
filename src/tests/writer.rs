use rand::seq::SliceRandom;
use rand::Rng;

use super::{create_database, rng};
use crate::distance::Euclidean;
use crate::Writer;

#[test]
fn use_u32_max_minus_one_for_a_vec() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, u32::MAX - 1, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 4294967294: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [4294967294] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0], distance: "euclidean" }
    "###);
}

#[test]
fn use_u32_max_for_a_vec() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, u32::MAX, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 4294967295: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [4294967295] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0], distance: "euclidean" }
    "###);
}

#[test]
fn write_one_vector() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), None).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0], distance: "euclidean" }
    "###);
}

#[test]
fn write_one_vector_in_one_tree() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0], distance: "euclidean" }
    "###);
}

#[test]
fn write_one_vector_in_multiple_trees() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), Some(10)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Tree 1: Descendants(Descendants { descendants: [0] })
    Tree 2: Descendants(Descendants { descendants: [0] })
    Tree 3: Descendants(Descendants { descendants: [0] })
    Tree 4: Descendants(Descendants { descendants: [0] })
    Tree 5: Descendants(Descendants { descendants: [0] })
    Tree 6: Descendants(Descendants { descendants: [0] })
    Tree 7: Descendants(Descendants { descendants: [0] })
    Tree 8: Descendants(Descendants { descendants: [0] })
    Tree 9: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], distance: "euclidean" }
    "###);
}

#[test]
fn write_vectors_until_there_is_a_descendants() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    for i in 0..3 {
        let id = i;
        let i = i as f32;
        writer.add_item(&mut wtxn, id, &[i, i, i]).unwrap();
    }

    writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [1.0000, 1.0000, 1.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [2.0000, 2.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0, 1, 2] })
    Root: Metadata { dimensions: 3, n_items: 3, roots: [0], distance: "euclidean" }
    "###);
}

#[test]
fn write_vectors_until_there_is_a_split() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    for i in 0..4 {
        let id = i;
        let i = i as f32;
        writer.add_item(&mut wtxn, id, &[i, i, i]).unwrap();
    }

    writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [1.0000, 1.0000, 1.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [2.0000, 2.0000, 2.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [3.0000, 3.0000, 3.0000] })
    Tree 0: Descendants(Descendants { descendants: [1, 2, 3] })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { normal: [-0.5774, -0.5774, -0.5774], left: NodeId { mode: Tree, item: 0 }, right: NodeId { mode: Item, item: 0 } })
    Root: Metadata { dimensions: 3, n_items: 4, roots: [1], distance: "euclidean" }
    "###);
}

#[test]
fn write_a_lot_of_random_points() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::prepare(&mut wtxn, handle.database, 0, 30).unwrap();
    let mut rng = rng();
    for id in 0..100 {
        let vector: [f32; 30] = std::array::from_fn(|_| rng.gen());
        writer.add_item(&mut wtxn, id, &vector).unwrap();
    }

    writer.build(&mut wtxn, &mut rng, Some(10)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle);
}

#[test]
fn write_multiple_indexes() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();

    for i in 0..5 {
        let writer = Writer::prepare(&mut wtxn, handle.database, i, 3).unwrap();
        writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();
        writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    }
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0], distance: "euclidean" }
    ==================
    Dumping index 1
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0], distance: "euclidean" }
    ==================
    Dumping index 2
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0], distance: "euclidean" }
    ==================
    Dumping index 3
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0], distance: "euclidean" }
    ==================
    Dumping index 4
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, n_items: 1, roots: [0], distance: "euclidean" }
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
        let writer = Writer::prepare(&mut wtxn, handle.database, index, 10).unwrap();

        // We're going to write 10 vectors per index
        for i in 0..10 {
            let vector: [f32; 10] = std::array::from_fn(|_| rng.gen());
            writer.add_item(&mut wtxn, i, &vector).unwrap();
        }
        writer.build(&mut wtxn, &mut rng, None).unwrap();
    }
    wtxn.commit().unwrap();
}
