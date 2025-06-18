use std::borrow::Cow;

use insta::assert_debug_snapshot;
use roaring::RoaringBitmap;

use crate::{
    distance::{Cosine, NodeHeaderCosine},
    internals::UnalignedVector,
    node::{Descendants, Leaf, Node, SplitPlaneNormal},
    parallel::TmpNodes,
};

#[test]
fn test_put_and_get_tmp_nodes() {
    let mut tmp_nodes = TmpNodes::<Cosine>::new().unwrap();
    for i in 0..10 {
        let node = Node::Descendants(Descendants {
            descendants: Cow::Owned(RoaringBitmap::from_iter(&[i + 0, i + 1, i + 2])),
        });
        tmp_nodes.put(i, &node).unwrap();
    }

    assert_debug_snapshot!(tmp_nodes.get(0).unwrap().unwrap(), @r"
    Descendants(
        Descendants {
            descendants: [
                0,
                1,
                2,
            ],
        },
    )
    ");
    assert_debug_snapshot!(tmp_nodes.get(9).unwrap().unwrap(), @r"
    Descendants(
        Descendants {
            descendants: [
                9,
                10,
                11,
            ],
        },
    )
    ");
    assert_debug_snapshot!(tmp_nodes.get(10).unwrap(), @"None");

    // We start at 11 so there will be a hole at the id 10
    for i in 11..20 {
        let normal =
            if i % 2 == 0 { Some(UnalignedVector::from_vec(vec![i as f32])) } else { None };
        let node = Node::SplitPlaneNormal(SplitPlaneNormal {
            left: i * 2,
            right: i * 2 + 1,
            normal: normal.map(|v| Leaf { header: NodeHeaderCosine { norm: 0. }, vector: v }),
        });
        tmp_nodes.put(i, &node).unwrap();
    }

    assert_debug_snapshot!(tmp_nodes.get(10).unwrap(), @"None");
    assert_debug_snapshot!(tmp_nodes.get(11).unwrap().unwrap(), @r#"
    SplitPlaneNormal(
        SplitPlaneNormal<cosine> {
            left: 22,
            right: 23,
            normal: "none",
        },
    )
    "#);

    assert_debug_snapshot!(tmp_nodes.get(15).unwrap().unwrap(), @r#"
    SplitPlaneNormal(
        SplitPlaneNormal<cosine> {
            left: 30,
            right: 31,
            normal: "none",
        },
    )
    "#);

    assert_debug_snapshot!(tmp_nodes.get(19).unwrap().unwrap(), @r#"
    SplitPlaneNormal(
        SplitPlaneNormal<cosine> {
            left: 38,
            right: 39,
            normal: "none",
        },
    )
    "#);

    assert_debug_snapshot!(tmp_nodes.get(20).unwrap(), @"None");

    // can we still get the previous nodes correctly?
    assert_debug_snapshot!(tmp_nodes.get(3).unwrap().unwrap(), @r"
    Descendants(
        Descendants {
            descendants: [
                3,
                4,
                5,
            ],
        },
    )
    ");
}
