use rand::seq::SliceRandom;
use rand::Rng;

use super::{create_database, rng};
use crate::distance::Euclidean;
use crate::Writer;

#[test]
fn use_u32_max_minus_one_for_a_vec() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, u32::MAX - 1, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 4294967294: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [4294967294] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[4294967294]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn use_u32_max_for_a_vec() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, u32::MAX, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 4294967295: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [4294967295] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[4294967295]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn write_one_vector() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), None).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn write_one_vector_in_one_tree() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn write_one_vector_in_multiple_trees() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3).unwrap();
    writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();

    writer.build(&mut wtxn, &mut rng(), Some(10)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn write_vectors_until_there_is_a_descendants() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3).unwrap();
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
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 1.0000, 1.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 2.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0, 1, 2] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0, 1, 2]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn write_vectors_until_there_is_a_split() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 3).unwrap();
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
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 1.0000, 1.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 2.0000, 2.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 3.0000, 3.0000] })
    Tree 0: Descendants(Descendants { descendants: [1, 2, 3] })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { left: Tree(0), right: Item(0), normal: [-0.5774, -0.5774, -0.5774] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0, 1, 2, 3]>, roots: [1], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn write_and_update_lot_of_random_points() {
    let handle = create_database::<Euclidean>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 30).unwrap();
    let mut rng = rng();
    for id in 0..100 {
        let vector: [f32; 30] = std::array::from_fn(|_| rng.gen());
        writer.add_item(&mut wtxn, id, &vector).unwrap();
    }

    writer.build(&mut wtxn, &mut rng, Some(10)).unwrap();
    wtxn.commit().unwrap();
    insta::assert_display_snapshot!(handle);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 30).unwrap();
    for id in (0..100).step_by(2) {
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
        let writer = Writer::new(handle.database, i, 3).unwrap();
        writer.add_item(&mut wtxn, 0, &[0.0, 1.0, 2.0]).unwrap();
        writer.build(&mut wtxn, &mut rng(), Some(1)).unwrap();
    }
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    ==================
    Dumping index 1
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    ==================
    Dumping index 2
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    ==================
    Dumping index 3
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    ==================
    Dumping index 4
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 1.0000, 2.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 3, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
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
        let writer = Writer::new(handle.database, index, 10).unwrap();

        // We're going to write 10 vectors per index
        for i in 0..10 {
            let vector: [f32; 10] = std::array::from_fn(|_| rng.gen());
            writer.add_item(&mut wtxn, i, &vector).unwrap();
        }
        writer.build(&mut wtxn, &mut rng, None).unwrap();
    }
    wtxn.commit().unwrap();
}

#[test]
fn overwrite_one_item_incremental() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [1, 3] })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { left: Tree(0), right: Item(2), normal: [0.0000, 0.0000] })
    Tree 2: Descendants(Descendants { descendants: [4, 5] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal { left: Tree(1), right: Tree(2), normal: [0.0000, 0.0000] })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(3), normal: [1.0000, 0.0000] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [4], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();

    let writer = Writer::new(handle.database, 0, 2).unwrap();

    writer.add_item(&mut wtxn, 3, &[6., 0.]).unwrap();

    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [6.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { left: Item(1), right: Tree(5), normal: [0.0000, 0.0000] })
    Tree 2: Descendants(Descendants { descendants: [4, 5] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal { left: Tree(1), right: Tree(2), normal: [0.0000, 0.0000] })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(3), normal: [1.0000, 0.0000] })
    Tree 5: Descendants(Descendants { descendants: [2, 3] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [4], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn delete_one_item_in_a_one_item_db() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    writer.del_item(&mut wtxn, 0).unwrap();

    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, roots: [], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn delete_one_item_in_a_descendant() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    // first, insert a bunch of items
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.add_item(&mut wtxn, 1, &[1., 0.]).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [0, 1] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    writer.del_item(&mut wtxn, 0).unwrap();

    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [1] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[1]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn delete_one_leaf_in_a_split() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    // first, insert a bunch of elements
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.add_item(&mut wtxn, 1, &[1., 0.]).unwrap();
    writer.add_item(&mut wtxn, 2, &[2., 0.]).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [1, 2] })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(0), normal: [1.0000, 0.0000] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2]>, roots: [1], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    writer.del_item(&mut wtxn, 0).unwrap();

    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    // after deleting the leaf, the split node should be replaced by a descendant
    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Tree 1: Descendants(Descendants { descendants: [1, 2] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[1, 2]>, roots: [1], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn delete_one_item() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    // first, insert a bunch of elements
    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [1, 3] })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { left: Tree(0), right: Item(2), normal: [0.0000, 0.0000] })
    Tree 2: Descendants(Descendants { descendants: [4, 5] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal { left: Tree(1), right: Tree(2), normal: [0.0000, 0.0000] })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(3), normal: [1.0000, 0.0000] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [4], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    writer.del_item(&mut wtxn, 3).unwrap();

    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Tree 1: Descendants(Descendants { descendants: [1, 2] })
    Tree 2: Descendants(Descendants { descendants: [4, 5] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal { left: Tree(1), right: Tree(2), normal: [0.0000, 0.0000] })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(3), normal: [1.0000, 0.0000] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 4, 5]>, roots: [4], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    // delete the last item in a descendants node
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    writer.del_item(&mut wtxn, 1).unwrap();

    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Tree 2: Descendants(Descendants { descendants: [4, 5] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal { left: Item(2), right: Tree(2), normal: [0.0000, 0.0000] })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(3), normal: [1.0000, 0.0000] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 2, 4, 5]>, roots: [4], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn add_one_item_incrementally_in_an_empty_db() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[]>, roots: [], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn add_one_item_incrementally_in_a_one_item_db() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [0] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();
    writer.add_item(&mut wtxn, 1, &[1., 0.]).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [0, 1] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn add_one_item_incrementally_to_create_a_split_node() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();
    writer.add_item(&mut wtxn, 0, &[0., 0.]).unwrap();
    writer.add_item(&mut wtxn, 1, &[1., 0.]).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [0, 1] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1]>, roots: [0], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();
    writer.add_item(&mut wtxn, 2, &[2., 0.]).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Tree 1: Descendants(Descendants { descendants: [1, 2] })
    Tree 2: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(1), normal: [1.0000, 0.0000] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2]>, roots: [2], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn add_one_item_incrementally() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    for i in 0..6 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0.]).unwrap();
    }
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [1, 3] })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { left: Tree(0), right: Item(2), normal: [0.0000, 0.0000] })
    Tree 2: Descendants(Descendants { descendants: [4, 5] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal { left: Tree(1), right: Tree(2), normal: [0.0000, 0.0000] })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(3), normal: [1.0000, 0.0000] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5]>, roots: [4], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    writer.add_item(&mut wtxn, 25, &[25., 0.]).unwrap();

    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Item 25: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [25.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [1, 3] })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { left: Tree(0), right: Tree(5), normal: [0.0000, 0.0000] })
    Tree 2: Descendants(Descendants { descendants: [4, 5] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal { left: Tree(1), right: Tree(2), normal: [0.0000, 0.0000] })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(3), normal: [1.0000, 0.0000] })
    Tree 5: Descendants(Descendants { descendants: [2, 25] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5, 25]>, roots: [4], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();

    writer.add_item(&mut wtxn, 8, &[8., 0.]).unwrap();

    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000] })
    Item 4: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [4.0000, 0.0000] })
    Item 5: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [5.0000, 0.0000] })
    Item 8: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [8.0000, 0.0000] })
    Item 25: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [25.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [1, 3] })
    Tree 1: SplitPlaneNormal(SplitPlaneNormal { left: Tree(0), right: Tree(7), normal: [0.0000, 0.0000] })
    Tree 2: Descendants(Descendants { descendants: [4, 5] })
    Tree 3: SplitPlaneNormal(SplitPlaneNormal { left: Tree(1), right: Tree(2), normal: [0.0000, 0.0000] })
    Tree 4: SplitPlaneNormal(SplitPlaneNormal { left: Item(0), right: Tree(3), normal: [1.0000, 0.0000] })
    Tree 6: Descendants(Descendants { descendants: [8, 25] })
    Tree 7: SplitPlaneNormal(SplitPlaneNormal { left: Tree(6), right: Item(2), normal: [0.0000, 0.0000] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3, 4, 5, 8, 25]>, roots: [4], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}

#[test]
fn delete_extraneous_tree() {
    let handle = create_database::<Euclidean>();
    let mut rng = rng();

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 4).unwrap();
    for i in 0..4 {
        writer.add_item(&mut wtxn, i, &[i as f32, 0., 0., 0.]).unwrap();
    }
    // 4 nodes of 4 dimensions should create 3 trees by default.
    writer.build(&mut wtxn, &mut rng, None).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000, 0.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000, 0.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000, 0.0000, 0.0000] })
    Tree 0: Descendants(Descendants { descendants: [0, 1, 2, 3] })
    Tree 1: Descendants(Descendants { descendants: [0, 1, 2, 3] })
    Tree 2: Descendants(Descendants { descendants: [0, 1, 2, 3] })
    Tree 3: Descendants(Descendants { descendants: [0, 1, 2, 3] })
    Root: Metadata { dimensions: 4, items: RoaringBitmap<[0, 1, 2, 3]>, roots: [0, 1, 2, 3], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(2)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000, 0.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000, 0.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000, 0.0000, 0.0000] })
    Tree 2: Descendants(Descendants { descendants: [0, 1, 2, 3] })
    Tree 3: Descendants(Descendants { descendants: [0, 1, 2, 3] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3]>, roots: [2, 3], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);

    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 2).unwrap();
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    wtxn.commit().unwrap();

    insta::assert_display_snapshot!(handle, @r###"
    ==================
    Dumping index 0
    Item 0: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [0.0000, 0.0000, 0.0000, 0.0000] })
    Item 1: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [1.0000, 0.0000, 0.0000, 0.0000] })
    Item 2: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [2.0000, 0.0000, 0.0000, 0.0000] })
    Item 3: Leaf(Leaf { header: NodeHeaderEuclidean { bias: 0.0 }, vector: [3.0000, 0.0000, 0.0000, 0.0000] })
    Tree 3: Descendants(Descendants { descendants: [0, 1, 2, 3] })
    Root: Metadata { dimensions: 2, items: RoaringBitmap<[0, 1, 2, 3]>, roots: [3], distance: "euclidean" }
    updated_item_ids: RoaringBitmap<[]>
    "###);
}
