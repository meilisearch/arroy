use rand::Rng;

use super::{create_database, rng};
use crate::{Euclidean, Writer};

#[test]
fn use_u32_max_minus_one_for_a_vec() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, u32::MAX - 1, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, rng(), Some(1)).unwrap();
    insta::assert_display_snapshot!(handle, @"");
}

#[test]
fn use_u32_max_for_a_vec() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, u32::MAX, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, rng(), Some(1)).unwrap();
    insta::assert_display_snapshot!(handle, @"");
}

#[test]
fn write_one_vector_in_one_tree() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0, 1.0, 2.0] })
    Tree 0: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })

    root node: Metadata { dimensions: 3, n_items: 1, roots: ItemIds { bytes: [0, 0, 0, 0] } }
    "###);
}

#[test]
fn write_one_vector_in_multiple_trees() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, rng(), Some(10)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0, 1.0, 2.0] })
    Tree 0: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })
    Tree 1: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })
    Tree 2: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })
    Tree 3: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })
    Tree 4: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })
    Tree 5: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })
    Tree 6: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })
    Tree 7: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })
    Tree 8: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })
    Tree 9: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0] } })

    root node: Metadata { dimensions: 3, n_items: 1, roots: ItemIds { bytes: [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0] } }
    "###);
}

#[test]
fn write_vectors_until_there_is_a_descendants() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    for i in 0..3 {
        let id = i;
        let i = i as f32;
        writer.add_item(&mut wtxn, id, &[i, i, i]).unwrap();
    }

    writer.build(&mut wtxn, rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0, 0.0, 0.0] })
    Item 1: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [1.0, 1.0, 1.0] })
    Item 2: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [2.0, 2.0, 2.0] })
    Tree 0: Descendants(Descendants { descendants: ItemIds { bytes: [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0] } })

    root node: Metadata { dimensions: 3, n_items: 3, roots: ItemIds { bytes: [0, 0, 0, 0] } }
    "###);
}

#[test]
fn write_vectors_until_there_is_a_split() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 3).unwrap();
    for i in 0..4 {
        let id = i;
        let i = i as f32;
        writer.add_item(&mut wtxn, id, &[i, i, i]).unwrap();
    }

    writer.build(&mut wtxn, rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    Item 0: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [0.0, 0.0, 0.0] })
    Item 1: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [1.0, 1.0, 1.0] })
    Item 2: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [2.0, 2.0, 2.0] })
    Item 3: Leaf(Leaf { header: NodeHeaderAngular { norm: 0.0 }, vector: [3.0, 3.0, 3.0] })
    Tree 0: Descendants(Descendants { descendants: ItemIds { bytes: [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0] } })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { normal: [0.57735026, 0.57735026, 0.57735026], left: NodeId { mode: Item, item: 0 }, right: NodeId { mode: Tree, item: 0 } })

    root node: Metadata { dimensions: 3, n_items: 4, roots: ItemIds { bytes: [1, 0, 0, 0] } }
    "###);
}

#[test]
fn write_a_lot_of_random_points() {
    let handle = create_database();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::<Euclidean>::prepare(&mut wtxn, handle.database, 0, 30).unwrap();
    let mut rng = rng();
    for id in 0..100 {
        let vector: [f32; 30] = std::array::from_fn(|_| rng.gen());
        writer.add_item(&mut wtxn, id, &vector).unwrap();
    }

    writer.build(&mut wtxn, rng, Some(10)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle);
}
