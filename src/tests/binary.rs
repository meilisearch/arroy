use crate::{
    distance::Hamming,
    tests::{create_database, rng},
    Writer,
};

#[test]
fn write_and_retrieve_binary_vector() {
    let handle = create_database::<Hamming>();
    let mut wtxn = handle.env.write_txn().unwrap();
    let writer = Writer::new(handle.database, 0, 16);
    writer
        .add_item(
            &mut wtxn,
            0,
            &[
                -2.0, -1.0, 0.0, -0.1, 2.0, 2.0, -12.4, 21.2, -2.0, -1.0, 0.0, 1.0, 2.0, 2.0,
                -12.4, 21.2,
            ],
        )
        .unwrap();
    let vec = writer.item_vector(&wtxn, 0).unwrap().unwrap();
    insta::assert_debug_snapshot!(vec, @r###"
    [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
    ]
    "###);

    writer.builder(&mut rng()).n_trees(1).build(&mut wtxn).unwrap();
    wtxn.commit().unwrap();

    insta::assert_snapshot!(handle, @r#"
    ==================
    Dumping index 0
    Root: Metadata { dimensions: 16, items: RoaringBitmap<[0]>, roots: [0], distance: "hamming" }
    Version: Version { major: 0, minor: 7, patch: 0 }
    Tree 0: Descendants(Descendants { descendants: [0] })
    Item 0: Leaf(Leaf { header: NodeHeaderHamming  { idx: "0" }, vector: [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, "other ..."] })
    "#);
}
